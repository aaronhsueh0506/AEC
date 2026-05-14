#!/usr/bin/env python3
"""v3.14 Arc-P Sprint 01 — Per-band ERL audit on 8 worst-FS listen cases.

Objective: quantify how much the scalar erl_estimate (production default ~0.3)
diverges from per-band actual ERL across 3 frequency bands:
  Band 0: 0–1 kHz  (LF — tight coupling, low ERL physically expected)
  Band 1: 1–4 kHz  (MF — transition)
  Band 2: 4–8 kHz  (HF — loose coupling, higher ERL physically expected)

The F3.1-v3 mic-excess formula:
    excess_ratio[k] = max(error_psd[k] - far_lw[k] * erl_estimate, 0) / (error_psd[k] + eps)

uses a scalar erl_estimate clipped to [0.001, 1.0]. In FS (no NE speech), this
scalar is typically ~0.3 post-convergence. In HF bands where real ERL is higher
(0.5–0.8), the scalar 0.3 UNDER-estimates ERL → excess formula over-fires →
dt_per_bin[HF] is too high → RES over-suppresses HF echo as if it were NE.

This script:
  1. Runs each of the 8 cases through the standard BALANCED AEC
  2. Collects per-frame per-bin (error_psd, far_lw, echo_psd) during FS-converged frames
  3. Computes per-band instantaneous ERL = mean(error_psd[band]) / mean(far_lw[band])
     as a proxy for true echo coupling (valid in FS where error ≈ residual echo)
  4. Aggregates mean/median/p10/p90 per band per bucket
  5. Quantifies scalar-0.3 bias = (actual_per_band_erl - 0.3) per band

Environment gate: this script is audit-only, default-OFF in production.
Set AEC_PER_BAND_ERL_AUDIT=1 to enable (informational only, not required —
the script always runs the audit logic directly without modifying aec.py).

Bench config: preset=BALANCED, filter_length=832 (52ms), cng=True, seed=42.

Usage:
    python3 tools/research/v3_14_p_s1_erl_audit.py \\
        --listen-dir /path/to/listen/v3_12_worst_fs \\
        --dataset-dir wav/aec_challenge_blind \\
        --out results/v3_14_p_s1_erl_audit

Output:
    results/v3_14_p_s1_erl_audit/<NN>_<stem>_per_band_erl.json
    results/v3_14_p_s1_erl_audit/summary.json

The summary JSON feeds docs/v3_14_p_s1_erl_audit.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


# 8 listen cases from v3.12 worst-FS exercise.
# Full stem used to locate mic/lpb in dataset_dir.
LISTEN_CASES = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',               'FS_static',   False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',               'FS_static',   False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',               'FS_static',   False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',               'FS_static',   False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',               'FS_static',   False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement', 'FS_movement', True),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement', 'FS_movement', True),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  'FS_movement', True),
]

# Frequency band boundaries in Hz.
# 3-band split tuned to ERL physics:
#   LF  0–1000  Hz: tight coupling (speaker → mic direct path + floor)
#   MF  1000–4000 Hz: transition region, coupling rolls off
#   HF  4000–8000 Hz: loose coupling (far-field scatter, HF room attenuation)
BAND_EDGES_HZ = [(0, 1000), (1000, 4000), (4000, 8000)]
BAND_NAMES    = ['LF_0_1k', 'MF_1_4k', 'HF_4_8k']

# FS-converged frame gating thresholds (conservative — only trust frames
# where the filter is well-converged and far-end is active)
ERLE_CONVERGED_THRESHOLD = 0.3   # erle_factor ≥ 0.3 → converged enough
FAR_POWER_THRESHOLD       = 1e-4  # far_power > 1e-4 → far-end active
DT_INDICATOR_THRESHOLD    = 0.15  # dt_indicator < 0.15 → FS (not DT)


def _estimate_delay(mic, ref, sr, max_delay_ms=1024.0):
    """GCC-PHAT delay estimate (matches E2 Path 3 default max_delay_ms=1024)."""
    max_d = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref))
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx
    return delay, float(confidence)


def _band_bins(fft_size, sr, band_edges_hz):
    """Return list of (lo_bin, hi_bin) index pairs for each band."""
    n_freqs = fft_size // 2 + 1
    freq_per_bin = sr / fft_size
    bands = []
    for lo_hz, hi_hz in band_edges_hz:
        lo_bin = int(round(lo_hz / freq_per_bin))
        hi_bin = min(int(round(hi_hz / freq_per_bin)), n_freqs - 1)
        lo_bin = max(lo_bin, 0)
        bands.append((lo_bin, hi_bin))
    return bands


def _safe_erl_ratio(error_psd_band, far_lw_band, eps=1e-12):
    """Instantaneous per-band ERL proxy: mean(error_psd) / mean(far_lw).

    In FS-converged frames: error ≈ residual echo ≈ far × ERL → this
    ratio estimates ERL. Clipped to [0.001, 2.0] for numeric safety.
    """
    far_mean = float(np.mean(far_lw_band)) + eps
    err_mean = float(np.mean(error_psd_band)) + eps
    return float(np.clip(err_mean / far_mean, 0.001, 2.0))


def audit_one_case(case_num, stem, scenario, movement, dataset_dir, listen_dir):
    """Run one case and collect per-band ERL statistics.

    Returns a dict with per-band ERL distribution + scalar-bias quantification.
    """
    from aec import AEC, AecConfig, AecMode, AecPreset

    # --- Locate audio files ---
    # stem examples:
    #   '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk'
    #   '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement'
    # All files (static + movement) live in farend_singletalk/ directory.
    stem_id = stem.split('_')[0]  # e.g. '7GTxyTksSUqCnP5y0ILG4A'

    # Try dataset dir (wav/aec_challenge_blind/farend_singletalk/<stem>_mic.wav)
    mic_path = None
    lpb_path = None
    for sub in ['farend_singletalk']:
        candidate_mic = os.path.join(dataset_dir, sub, f'{stem}_mic.wav')
        candidate_lpb = os.path.join(dataset_dir, sub, f'{stem}_lpb.wav')
        if os.path.exists(candidate_mic) and os.path.exists(candidate_lpb):
            mic_path, lpb_path = candidate_mic, candidate_lpb
            break

    if mic_path is None:
        # Fallback: listen_dir short filenames (only works for static FS cases)
        listen_mic = os.path.join(listen_dir, f'{case_num}_{stem_id[:5]}_FSstatic_mic.wav')
        listen_lpb = os.path.join(listen_dir, f'{case_num}_{stem_id[:5]}_FSstatic_lpb.wav')
        if os.path.exists(listen_mic) and os.path.exists(listen_lpb):
            mic_path, lpb_path = listen_mic, listen_lpb

    if mic_path is None:
        return {'error': f'audio not found for {stem}', 'case': case_num, 'stem': stem}

    mic_wav, sr_m = sf.read(mic_path)
    lpb_wav, sr_l = sf.read(lpb_path)
    assert sr_m == sr_l == 16000, f'Expected 16 kHz, got mic={sr_m} lpb={sr_l}'
    sr = 16000
    mic_wav = mic_wav.astype(np.float32)
    lpb_wav = lpb_wav.astype(np.float32)

    # --- GCC-PHAT delay pre-align (matches E2 Path 3 bench config) ---
    delay_samp, delay_conf = _estimate_delay(mic_wav, lpb_wav, sr, max_delay_ms=1024.0)
    if delay_samp > 0:
        lpb_aligned = np.concatenate([np.zeros(delay_samp, dtype=np.float32), lpb_wav])
        n = min(len(mic_wav), len(lpb_aligned))
        mic_aligned = mic_wav[:n]
        lpb_aligned = lpb_aligned[:n]
    else:
        n = min(len(mic_wav), len(lpb_wav))
        mic_aligned = mic_wav[:n]
        lpb_aligned = lpb_wav[:n]

    # --- Build AEC (BALANCED / fl=832 / cng=True) ---
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr,
        mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False,
        enable_shadow=True,
        enable_res=True,
        enable_cng=True,
        use_kalman=True,
    )
    np.random.seed(42)
    aec = AEC(cfg)
    hop = aec.hop_size  # 160 @ 16 kHz

    # Compute band bin ranges (after AEC is created so fft_size is known)
    fft_sz = aec.filter.fft_size
    band_bins = _band_bins(fft_sz, sr, BAND_EDGES_HZ)
    freq_per_bin = sr / fft_sz

    # Per-band per-frame ERL accumulator.
    # Stratified into filter_converged=True vs False frames.
    # ERL proxy = error_psd / far_lw (valid when filter cancels well;
    # inflated when filter doesn't cancel because error ≈ mic, not residual echo).
    band_erl_samples   = [[] for _ in BAND_NAMES]   # all qualifying frames
    band_erl_conv      = [[] for _ in BAND_NAMES]   # filter_converged=True only
    band_erl_noconv    = [[] for _ in BAND_NAMES]   # filter_converged=False
    band_error_psd_samples = [[] for _ in BAND_NAMES]
    band_far_lw_samples    = [[] for _ in BAND_NAMES]
    band_excess_ratio_scalar  = [[] for _ in BAND_NAMES]  # with scalar 0.3
    band_excess_ratio_perband = [[] for _ in BAND_NAMES]  # with per-band ERL
    # Also track excess_ratio using scalar 0.3 on converged frames only
    band_excess_scalar_conv   = [[] for _ in BAND_NAMES]
    band_excess_perband_conv  = [[] for _ in BAND_NAMES]
    n_converged_fs_frames = 0

    # Gating thresholds (relaxed for worst-FS audit — these cases are intentionally
    # challenging, so filter may not reach full convergence):
    #   erle_windowed_db >= 0 dB (any positive ERLE) → filter has some traction
    #     NOTE: for very bad cases, we relax to -99 dB (disabled) so we still capture
    #     frames where long_window_far_psd has warmed up. We rely on far_power_db
    #     gate to avoid silence. The F3.1 path itself only requires filter_converged
    #     (not a strict ERLE threshold) and lw_ready.
    #   far_power_db     > -55 dB → far-end active
    #   dt_confidence    < 0.30   → FS frame (broad — these are FS cases)
    ERLE_WIN_DB_THRESH = -99.0  # disabled: allow any converged frame
    FAR_PWR_DB_THRESH  = -55.0
    DT_CONF_THRESH     = 0.30

    # Per-frame scalars
    erl_scalar_vals    = []  # scalar _erl_estimate (linear, from erl_db)
    dt_confidence_vals = []
    erle_win_db_vals   = []

    n_total_frames   = 0
    n_fs_frames      = 0  # FS-converged gated frames used in audit
    n_early_frames   = 0  # skipped: not yet converged

    pos = 0
    while pos + hop <= n:
        blk_mic = mic_aligned[pos: pos + hop]
        blk_lpb = lpb_aligned[pos: pos + hop]
        aec.process(blk_mic, blk_lpb)
        n_total_frames += 1

        stats = aec.get_stats()
        # erle_windowed_db: ~10s decaying-window ERLE in dB
        erle_win_db = float(stats.erle_windowed_db)
        # dt_confidence: combined DT confidence [0,1]
        dt_conf  = float(stats.dt_confidence)
        # erl_db: 10*log10(_erl_estimate); convert back to linear
        erl_db_v = float(stats.erl_db)
        erl_lin  = 10.0 ** (erl_db_v / 10.0)
        # far_power_db for far-active gate
        far_pw_db = float(stats.far_power_db)

        erl_scalar_vals.append(erl_lin)
        dt_confidence_vals.append(dt_conf)
        erle_win_db_vals.append(erle_win_db)

        # Gate: far active + low DT + long-window PSD ready.
        # NOTE: We do NOT require filter_converged here, because the worst-FS
        # cases in this 8-case bundle rarely converge (delay mismatch prevents it).
        # The ERL bias audit targets the _far_lw/error_psd_ ratio at the per-bin
        # level — this is meaningful whenever far_lw has warmed up (>0 updates)
        # and far-end energy is present. This is exactly the same gating condition
        # as the lw_ready flag used by F3.1-v3 in production (plus far active + no DT).
        filter_converged = bool(stats.filter_converged)

        # Access per-bin arrays from res (ResFilter) — needed for lw_ready check
        res = aec.res
        if res is None:
            n_early_frames += 1
            pos += hop
            continue
        residual_est = res._residual_est
        lw_ready = (residual_est is not None
                    and residual_est._long_window_n_updates > 0)

        if (not lw_ready
                or far_pw_db < FAR_PWR_DB_THRESH
                or dt_conf > DT_CONF_THRESH):
            n_early_frames += 1
            pos += hop
            continue

        error_psd = res.error_psd                          # shape (n_freqs,)
        far_lw    = residual_est._long_window_far_psd      # shape (n_freqs,)

        if error_psd is None or far_lw is None:
            n_early_frames += 1
            pos += hop
            continue

        # Additional far_power check from far_lw (avoid silent far-end)
        far_lw_mean = float(np.mean(far_lw))
        if far_lw_mean < 1e-8:
            n_early_frames += 1
            pos += hop
            continue

        n_fs_frames += 1
        if filter_converged:
            n_converged_fs_frames += 1

        # Per-band ERL computation
        # In FS frames: error_psd ≈ residual_echo ≈ far_lw × ERL(k)
        # so per-bin ERL_proxy(k) = error_psd(k) / far_lw(k)
        # This is the actual coupling ratio the scalar erl_estimate is trying to capture.
        # Caveat: when filter_converged=False, error_psd ≈ mic input (not residual echo),
        # so the ERL proxy is inflated. Stratify to expose this.
        for bi, (bname, (lo, hi)) in enumerate(zip(BAND_NAMES, band_bins)):
            err_band = error_psd[lo:hi+1]
            far_band = far_lw[lo:hi+1]
            if len(err_band) == 0 or len(far_band) == 0:
                continue

            inst_erl = _safe_erl_ratio(err_band, far_band)
            band_erl_samples[bi].append(inst_erl)
            band_error_psd_samples[bi].append(float(np.mean(err_band)))
            band_far_lw_samples[bi].append(float(np.mean(far_band)))

            # Stratify by convergence
            if filter_converged:
                band_erl_conv[bi].append(inst_erl)
            else:
                band_erl_noconv[bi].append(inst_erl)

            # Excess ratio with scalar 0.3 (production behavior)
            excess_sc = np.maximum(err_band - far_band * 0.3, 0.0)
            er_sc     = float(np.mean(np.clip(excess_sc / (err_band + 1e-10), 0.0, 1.0)))
            band_excess_ratio_scalar[bi].append(er_sc)

            # Excess ratio with per-band actual ERL (ideal behavior)
            excess_pb = np.maximum(err_band - far_band * inst_erl, 0.0)
            er_pb     = float(np.mean(np.clip(excess_pb / (err_band + 1e-10), 0.0, 1.0)))
            band_excess_ratio_perband[bi].append(er_pb)

            # Converged-only excess comparison
            if filter_converged:
                band_excess_scalar_conv[bi].append(er_sc)
                band_excess_perband_conv[bi].append(er_pb)

        pos += hop

    # --- Aggregate stats per band ---
    def _stats(arr):
        if len(arr) == 0:
            return {'mean': None, 'median': None, 'p10': None, 'p90': None,
                    'min': None, 'max': None, 'n': 0}
        a = np.asarray(arr, dtype=np.float64)
        return {
            'mean':   float(np.mean(a)),
            'median': float(np.median(a)),
            'p10':    float(np.percentile(a, 10)),
            'p90':    float(np.percentile(a, 90)),
            'min':    float(np.min(a)),
            'max':    float(np.max(a)),
            'n':      int(len(a)),
        }

    band_results = {}
    for bi, bname in enumerate(BAND_NAMES):
        erl_arr     = band_erl_samples[bi]
        erl_cv      = band_erl_conv[bi]
        erl_nc      = band_erl_noconv[bi]
        er_sc_arr   = band_excess_ratio_scalar[bi]
        er_pb_arr   = band_excess_ratio_perband[bi]
        er_sc_cv    = band_excess_scalar_conv[bi]
        er_pb_cv    = band_excess_perband_conv[bi]

        # Bias sign convention:
        # bias > 0 means actual ERL > 0.3 (scalar UNDER-estimates coupling in this band)
        # → excess_ratio over-fires → dt_per_bin[k] too high → over-suppresses NE
        # In FS: residual echo is the dominant signal, so actual ERL in converged frames
        #        reflects true room coupling. If ERL > 0.3 in HF → scalar is LF-biased.
        bias_vals   = [e - 0.3 for e in erl_arr]          # actual - 0.3
        bias_cv     = [e - 0.3 for e in erl_cv]
        bias_nc     = [e - 0.3 for e in erl_nc]

        band_results[bname] = {
            'band_hz':         list(BAND_EDGES_HZ[bi]),
            'band_bins':       list(band_bins[bi]),
            # All qualifying frames (lw_ready + far_active + low_DT)
            'erl_all':               _stats(erl_arr),
            'bias_actual_minus_030': _stats(bias_vals),   # > 0 = actual > 0.3
            'excess_ratio_scalar030':      _stats(er_sc_arr),
            'excess_ratio_perband':        _stats(er_pb_arr),
            'excess_ratio_delta':          _stats([pb - sc for sc, pb in zip(er_sc_arr, er_pb_arr)]),
            # Filter-converged-only (most reliable ERL proxy)
            'erl_converged':          _stats(erl_cv),
            'bias_converged':         _stats(bias_cv),
            'excess_scalar_conv':     _stats(er_sc_cv),
            'excess_perband_conv':    _stats(er_pb_cv),
            'excess_delta_conv':      _stats([pb - sc for sc, pb in zip(er_sc_cv, er_pb_cv)]),
            # Non-converged frames (ERL proxy inflated by mic leakthrough)
            'erl_nonconverged':       _stats(erl_nc),
            'bias_nonconverged':      _stats(bias_nc),
        }

    # Scalar ERL summary
    sc_erl_stats      = _stats(erl_scalar_vals)
    erle_win_stats    = _stats(erle_win_db_vals)
    dt_conf_stats     = _stats(dt_confidence_vals)

    return {
        'case': case_num,
        'stem': stem,
        'scenario': scenario,
        'movement': movement,
        'delay_samp': int(delay_samp),
        'delay_conf': float(delay_conf),
        'n_total_frames': int(n_total_frames),
        'n_fs_frames': int(n_fs_frames),
        'n_converged_fs_frames': int(n_converged_fs_frames),
        'n_early_frames': int(n_early_frames),
        'fs_frame_pct': float(n_fs_frames / max(n_total_frames, 1)),
        'conv_frame_pct': float(n_converged_fs_frames / max(n_total_frames, 1)),
        'fft_size': int(fft_sz),
        'freq_per_bin_hz': float(freq_per_bin),
        'scalar_erl_linear': sc_erl_stats,
        'erle_windowed_db': erle_win_stats,
        'dt_confidence': dt_conf_stats,
        'bands': band_results,
    }


def build_summary(all_results, scalar_baseline=0.3):
    """Cross-case summary: per-band ERL distribution + scalar bias profile."""
    buckets = {'FS_static': [], 'FS_movement': []}
    for r in all_results:
        if 'error' in r:
            continue
        b = r['scenario']
        if b in buckets:
            buckets[b].append(r)

    summary = {'per_bucket': {}, 'global': {}, 'scalar_baseline': scalar_baseline}

    for bucket, cases in buckets.items():
        if not cases:
            summary['per_bucket'][bucket] = {'n_cases': 0}
            continue
        per_band = {bname: {'erl_mean': [], 'erl_median': [], 'erl_p10': [], 'erl_p90': [],
                            'erl_cv_mean': [], 'erl_cv_p10': [], 'erl_cv_p90': [],
                            'bias_mean': [], 'bias_cv_mean': [],
                            'excess_sc_mean': [], 'excess_pb_mean': [],
                            'excess_sc_cv_mean': [], 'excess_pb_cv_mean': []}
                    for bname in BAND_NAMES}
        for r in cases:
            for bname in BAND_NAMES:
                bd = r['bands'].get(bname, {})
                erl = bd.get('erl_all', {})
                erl_cv = bd.get('erl_converged', {})
                bias = bd.get('bias_actual_minus_030', {})
                bias_cv = bd.get('bias_converged', {})
                ex_sc = bd.get('excess_ratio_scalar030', {})
                ex_pb = bd.get('excess_ratio_perband', {})
                ex_sc_cv = bd.get('excess_scalar_conv', {})
                ex_pb_cv = bd.get('excess_perband_conv', {})
                if erl.get('mean') is not None:
                    per_band[bname]['erl_mean'].append(erl['mean'])
                    per_band[bname]['erl_median'].append(erl['median'])
                    per_band[bname]['erl_p10'].append(erl['p10'])
                    per_band[bname]['erl_p90'].append(erl['p90'])
                if erl_cv.get('mean') is not None:
                    per_band[bname]['erl_cv_mean'].append(erl_cv['mean'])
                    per_band[bname]['erl_cv_p10'].append(erl_cv.get('p10', erl_cv['mean']))
                    per_band[bname]['erl_cv_p90'].append(erl_cv.get('p90', erl_cv['mean']))
                if bias.get('mean') is not None:
                    per_band[bname]['bias_mean'].append(bias['mean'])
                if bias_cv.get('mean') is not None:
                    per_band[bname]['bias_cv_mean'].append(bias_cv['mean'])
                if ex_sc.get('mean') is not None:
                    per_band[bname]['excess_sc_mean'].append(ex_sc['mean'])
                if ex_pb.get('mean') is not None:
                    per_band[bname]['excess_pb_mean'].append(ex_pb['mean'])
                if ex_sc_cv.get('mean') is not None:
                    per_band[bname]['excess_sc_cv_mean'].append(ex_sc_cv['mean'])
                if ex_pb_cv.get('mean') is not None:
                    per_band[bname]['excess_pb_cv_mean'].append(ex_pb_cv['mean'])

        bucket_summary = {'n_cases': len(cases), 'bands': {}}
        for bname in BAND_NAMES:
            d = per_band[bname]

            def agg(lst):
                if not lst:
                    return None
                a = np.asarray(lst)
                return {'mean': float(np.mean(a)), 'min': float(np.min(a)),
                        'max': float(np.max(a))}

            bucket_summary['bands'][bname] = {
                # All qualifying frames
                'erl_mean_across_cases':    agg(d['erl_mean']),
                'erl_median_across_cases':  agg(d['erl_median']),
                'erl_p10_across_cases':     agg(d['erl_p10']),
                'erl_p90_across_cases':     agg(d['erl_p90']),
                'bias_actual_minus_030':    agg(d['bias_mean']),  # > 0: actual > 0.3
                'excess_ratio_scalar030':   agg(d['excess_sc_mean']),
                'excess_ratio_perband':     agg(d['excess_pb_mean']),
                # Filter-converged frames only (most reliable)
                'erl_conv_mean':            agg(d['erl_cv_mean']),
                'erl_conv_p10':             agg(d['erl_cv_p10']),
                'erl_conv_p90':             agg(d['erl_cv_p90']),
                'bias_conv_mean':           agg(d['bias_cv_mean']),
                'excess_sc_conv':           agg(d['excess_sc_cv_mean']),
                'excess_pb_conv':           agg(d['excess_pb_cv_mean']),
            }
        summary['per_bucket'][bucket] = bucket_summary

    # Global (all 8 cases pooled)
    all_valid = [r for r in all_results if 'error' not in r]
    for bname in BAND_NAMES:
        all_erl = []
        all_erl_cv = []
        for r in all_valid:
            bd = r['bands'].get(bname, {})
            erl = bd.get('erl_all', {})
            erl_cv = bd.get('erl_converged', {})
            if erl.get('mean') is not None:
                all_erl.extend([erl['mean'], erl['median']])  # weight roughly equally
            if erl_cv.get('mean') is not None:
                all_erl_cv.append(erl_cv['mean'])
        g = {}
        if all_erl:
            a = np.asarray(all_erl)
            g.update({
                'erl_grand_mean': float(np.mean(a)),
                'erl_grand_p10':  float(np.percentile(a, 10)),
                'erl_grand_p90':  float(np.percentile(a, 90)),
                'bias_from_030':  float(np.mean(a)) - scalar_baseline,
            })
        if all_erl_cv:
            b = np.asarray(all_erl_cv)
            g.update({
                # Converged-only ERL: most reliable per-band ERL estimate
                'erl_conv_mean':      float(np.mean(b)),
                'erl_conv_p10':       float(np.percentile(b, 10)),
                'erl_conv_p90':       float(np.percentile(b, 90)),
                'bias_conv_from_030': float(np.mean(b)) - scalar_baseline,
            })
        summary['global'][bname] = g

    return summary


def print_summary_table(summary):
    """Print human-readable tables: all-frames + converged-frames."""
    print('\n=== v3.14 Arc-P Sprint 01: Per-Band ERL Audit ===')
    print(f'Scalar baseline: {summary["scalar_baseline"]:.2f}')
    print('ERL proxy = error_psd / far_lw per band. Converged rows most reliable.\n')

    # Table 1: All qualifying frames (lw_ready + far_active + low_DT)
    print('-- ALL qualifying frames (may include non-converged; ERL inflated by mic leakthrough) --')
    print(f'{"Band":<14} {"Bucket":<13} {"ERL_mean":<10} {"ERL_p10":<9} {"ERL_p90":<9} '
          f'{"Bias(act-0.3)":<15} {"ER_sc030":<10} {"ER_perband":<10}')
    print('-' * 95)
    for bucket, bdata in summary['per_bucket'].items():
        if bdata.get('n_cases', 0) == 0:
            continue
        for bname in BAND_NAMES:
            bd  = bdata['bands'].get(bname, {})
            em  = (bd.get('erl_mean_across_cases') or {})
            ep  = (bd.get('erl_p10_across_cases') or {})
            ep9 = (bd.get('erl_p90_across_cases') or {})
            bi  = (bd.get('bias_actual_minus_030') or {})
            es  = (bd.get('excess_ratio_scalar030') or {})
            epb = (bd.get('excess_ratio_perband') or {})
            print(f'{bname:<14} {bucket:<13} '
                  f'{em.get("mean",0):.3f}      '
                  f'{ep.get("mean",0):.3f}     '
                  f'{ep9.get("mean",0):.3f}     '
                  f'{bi.get("mean",0):+.3f}           '
                  f'{es.get("mean",0):.3f}      '
                  f'{epb.get("mean",0):.3f}')

    print()
    # Table 2: Filter-converged frames only (most reliable)
    print('-- CONVERGED frames only (filter_converged=True; ERL proxy is genuine room coupling) --')
    print(f'{"Band":<14} {"Bucket":<13} {"ERL_mean":<10} {"ERL_p10":<9} {"ERL_p90":<9} '
          f'{"Bias(act-0.3)":<15} {"ER_sc030":<10} {"ER_perband":<10}')
    print('-' * 95)
    for bucket, bdata in summary['per_bucket'].items():
        if bdata.get('n_cases', 0) == 0:
            continue
        for bname in BAND_NAMES:
            bd  = bdata['bands'].get(bname, {})
            em  = (bd.get('erl_conv_mean') or {})
            ep  = (bd.get('erl_conv_p10') or {})
            ep9 = (bd.get('erl_conv_p90') or {})
            bi  = (bd.get('bias_conv_mean') or {})
            es  = (bd.get('excess_sc_conv') or {})
            epb = (bd.get('excess_pb_conv') or {})
            if em.get('mean') is None:
                print(f'{bname:<14} {bucket:<13} (no converged frames)')
                continue
            print(f'{bname:<14} {bucket:<13} '
                  f'{em.get("mean",0):.3f}      '
                  f'{ep.get("mean",0):.3f}     '
                  f'{ep9.get("mean",0):.3f}     '
                  f'{bi.get("mean",0):+.3f}           '
                  f'{es.get("mean",0):.3f}      '
                  f'{epb.get("mean",0):.3f}')

    print()
    print('Global per-band ERL (all frames / converged frames):')
    for bname in BAND_NAMES:
        g = summary['global'].get(bname, {})
        print(f'  {bname:<14}: '
              f'all:  mean={g.get("erl_grand_mean",0):.3f}  '
              f'p10={g.get("erl_grand_p10",0):.3f}  p90={g.get("erl_grand_p90",0):.3f}  '
              f'bias={g.get("bias_from_030",0):+.3f}  |  '
              f'conv: mean={g.get("erl_conv_mean",0):.3f}  '
              f'p10={g.get("erl_conv_p10",0):.3f}  p90={g.get("erl_conv_p90",0):.3f}  '
              f'bias={g.get("bias_conv_from_030",0):+.3f}')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-dir', default=None,
                        help='Path to listen/v3_12_worst_fs/ (for fallback short filenames)')
    parser.add_argument('--dataset-dir',
                        default=os.path.join(_REPO, 'wav', 'aec_challenge_blind'),
                        help='Root of wav/aec_challenge_blind/ (used to load mic/lpb)')
    parser.add_argument('--out', default=os.path.join(_REPO, 'results', 'v3_14_p_s1_erl_audit'),
                        help='Output directory for per-case JSONs + summary')
    parser.add_argument('--cases', nargs='*', default=None,
                        help='Limit to specific case numbers (e.g. --cases 01 03)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    listen_dir = args.listen_dir or ''
    dataset_dir = args.dataset_dir

    all_results = []
    for case_num, stem, scenario, movement in LISTEN_CASES:
        if args.cases and case_num not in args.cases:
            continue
        print(f'[{case_num}] {stem[:30]:<30} ({scenario}) ...', end=' ', flush=True)
        t0 = time.time()
        try:
            result = audit_one_case(case_num, stem, scenario, movement, dataset_dir, listen_dir)
        except Exception as e:
            result = {'error': str(e), 'case': case_num, 'stem': stem}
        elapsed = time.time() - t0

        if 'error' in result:
            print(f'ERROR: {result["error"]}')
        else:
            fs_pct = result['fs_frame_pct'] * 100
            n_fs  = result['n_fs_frames']
            print(f'OK ({elapsed:.1f}s, {n_fs} FS frames = {fs_pct:.0f}%)')

        # Save per-case JSON
        stem_short = stem.split('_')[0][:8]
        out_json = out_dir / f'{case_num}_{stem_short}_{scenario}_per_band_erl.json'
        with open(out_json, 'w') as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

    # Build and save summary
    summary = build_summary(all_results)
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary_table(summary)
    print(f'Per-case JSONs + summary written to: {out_dir}')
    return all_results, summary


if __name__ == '__main__':
    main()
