#!/usr/bin/env python3
"""E5.S1 saturation audit — listen 8-case per-frame sat trace.

Captures per-frame saturation state on the same 8 worst-FS cases
covered by E2.S1 delay audit. Combines two information sources:

  1. OFFLINE raw-signal metrics (computed from mic.wav / lpb.wav
     directly, no AEC run needed):
       - mic_sat_rate at 0.90 / 0.95 / 0.99 / 0.999
       - lpb_sat_rate at 0.90 / 0.95 / 0.99 / 0.999
       - per-frame mic / lpb sat_level estimate (peak / margin)
       - longest consecutive saturation run (samples)
       - mic <-> lpb saturation alignment (per-frame Pearson r)
       - inter-event gap distribution (hangover length)

  2. ONLINE AEC trace (one stream run through the bench-equivalent
     AEC, sampling `get_stats().saturation_level` per frame). This
     captures the smoothed `_saturation_level` value F-E5 gates fire
     on, which has an asymmetric EMA (attack 0.3 / release 0.98) so
     stays high long after raw clip events end.

Bench config mirrors the standard 800-case run: preset=BALANCED,
filter_length=832, cng=True, seed=0.

Audit-only: no production code modified. The F-E5 bundle stays
default OFF (f_e5_enabled=False) since we want to characterise the
production behaviour, not the candidate path.

Output dir: `results/v3_13_e5_s1_audit/`
  per-case JSON: <NN>_<stem16>.json
  summary table: summary.md

Usage:
    python3 tools/research/e5_s1_sat_audit.py \\
        --out results/v3_13_e5_s1_audit
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


# 8 listen cases preserved from v3.12 worst-FS listen exercise (2026-05-13).
# Same numbering as E2.S1; cross-reference docs/v3_13_e2_s1_verdict.md.
LISTEN_CASES = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                'FS_static',   False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',                'FS_static',   False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',                'FS_static',   False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',                'FS_static',   False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',                'FS_static',   False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',  'FS_movement', True ),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',  'FS_movement', True ),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  'FS_movement', True ),
]


# Cross-reference notes from listen exercise (v3_13_e2_s1_verdict.md).
LISTEN_NOTES = {
    '01': 'delay ~10000 samp + clipping',
    '02': 'delay ~1200 samp + clipping + radio-like',
    '03': 'gain 變化, mic 品質差 (E1)',
    '04': 'delay ~9800 samp',
    '05': 'delay ~6000 samp',
    '06': 'delay ~1800 samp + clipping',
    '07': 'delay ~640 samp + 嚴重 NL',
    '08': 'delay ~10000 samp + occasional sat',
}


# Cross-reference E2.S1 delay grouping. A = delay-broken, B = delay-OK.
E2_S1_GROUP = {
    '01': 'A', '02': 'B', '03': 'B', '04': 'A',
    '05': 'A', '06': 'B', '07': 'B', '08': 'A',
}


def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    """Bit-for-bit port of eval_aec_challenge.estimate_delay (GCC-PHAT)."""
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
    peak_val_phat = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx_phat = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val_phat / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx_phat
    return delay, float(confidence)


def _signal_clip_rates(name, sig):
    """Multi-threshold per-sample clip-rate (matches E2.S1 mic_clip_*)."""
    s = sig.astype(np.float64)
    peak = float(np.max(np.abs(s)))
    rms = float(np.sqrt(np.mean(s ** 2)))
    abs_s = np.abs(s)
    return {
        f'{name}_peak': peak,
        f'{name}_rms': rms,
        f'{name}_peak_dbfs': 20 * np.log10(peak + 1e-12),
        f'{name}_clip_90': float(np.mean(abs_s > 0.90)),
        f'{name}_clip_95': float(np.mean(abs_s > 0.95)),
        f'{name}_clip_99': float(np.mean(abs_s > 0.99)),
        f'{name}_clip_999': float(np.mean(abs_s > 0.999)),
    }


def _longest_run(mask):
    """Longest True run in a 1-D boolean mask (in samples)."""
    if not np.any(mask):
        return 0
    # Find run lengths via diff of cumulative indices.
    runs = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return int(max(runs)) if runs else 0


def _inter_event_gaps(mask):
    """Gap lengths between consecutive saturation events (in samples).

    A "saturation event" = contiguous run of True samples; the gap is
    the number of False samples between two consecutive events. Used
    as a hangover-length distribution proxy: short gaps = rapid re-
    saturation; long gaps = isolated events.
    """
    if not np.any(mask):
        return []
    in_event = False
    gap_lens = []
    cur_gap = 0
    # We only want gaps BETWEEN events (so skip the leading gap before
    # the first event, and ignore any trailing zeros).
    seen_first_event = False
    for v in mask:
        if v:
            if seen_first_event and not in_event:
                # End of a gap. Push and reset.
                gap_lens.append(cur_gap)
            cur_gap = 0
            in_event = True
            seen_first_event = True
        else:
            in_event = False
            cur_gap += 1
    return gap_lens


def _per_frame_sat_metrics(name, sig, hop=160):
    """Per-frame saturation summary at hop=160 (10 ms @ 16 kHz).

    Returns dict with:
      <name>_frame_peak       — per-frame max(|sig|) array (n_frames,)
      <name>_frame_sat_level  — 1 - dynamic_margin per frame, in [0, 1)
                                where margin = 1 - peak. Encodes how
                                close the frame approached full scale.
      <name>_frame_sat_flag90 — bool per-frame ≥ 0.90 peak
      <name>_frame_sat_flag95 — bool per-frame ≥ 0.95 peak
      <name>_frame_sat_flag99 — bool per-frame ≥ 0.99 peak
    """
    n = len(sig)
    n_frames = n // hop
    s = sig.astype(np.float64)
    # Reshape into frames; drop trailing partial.
    sframe = s[:n_frames * hop].reshape(n_frames, hop)
    frame_peak = np.max(np.abs(sframe), axis=1)
    frame_sat_level = np.clip(frame_peak - 0.0, 0.0, 1.0)  # = peak in [0,1]
    return {
        f'{name}_n_frames': int(n_frames),
        f'{name}_frame_peak': frame_peak,
        f'{name}_frame_sat_level': frame_sat_level,
        f'{name}_frame_sat_flag90': (frame_peak >= 0.90),
        f'{name}_frame_sat_flag95': (frame_peak >= 0.95),
        f'{name}_frame_sat_flag99': (frame_peak >= 0.99),
    }


def _stream_online_sat(mic, lpb_aligned, sr):
    """Run AEC over the case, collect per-frame `_saturation_level`.

    Matches bench config (BALANCED / fl=832 / cng=True / seed=0). The
    F-E5 bundle stays default OFF — production path. Returns:

      online_sat_level  — float array (n_frames,) of the smoothed value
      online_sat_ref    — float array, per-frame value of internal
                          `_sat_detector_ref.saturation_level`
      online_sat_mic    — float array, per-frame value of internal
                          `_sat_detector_mic.saturation_level`
      online_e5_gate_*  — counts of how often each F-E5 gate WOULD fire
                          (we check the live state, F-E5 itself is OFF
                          so no behaviour change).
    """
    from aec import AEC, AecConfig, AecMode, AecPreset
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True,
        # F-E5 stays OFF (production path being audited).
        f_e5_enabled=False,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    n = min(len(mic), len(lpb_aligned))
    pos = 0
    sat_level, sat_ref_arr, sat_mic_arr = [], [], []
    # F-E5 condition counters
    e5_mic_softclip_fires = 0     # sat_mic > f_e5_mic_softclip_threshold (0.3)
    e5_main_mu_freeze = 0         # _saturation_level > 0.5
    e5_error_psd_reset = 0        # prev > 0.5 AND now < 0.2 (sat-to-clean)
    e5_shadow_rise_mask = 0       # _saturation_level > 0.5 AND rise event would fire
    prev_sat = 0.0
    # Also a few related state counters we want to characterise.
    mic_clip_freeze = 0           # _sat_detector_mic.saturation_level > 0.8
                                  # (this is the SOURCE clamp, always ON,
                                  # not gated by f_e5_enabled)
    softclip_ref_fires = 0        # sat_ref > 0.1 → soft-clip on ref
                                  # (config flag saturation_softclip_ref=True
                                  # in BALANCED so this ALWAYS fires when sat_ref>0.1)
    shadow_safety_fires = 0       # shadow safety gate (sat_safe = sat<0.5)
                                  # active when sat>=0.5 → shadow_mu_scale=0.1

    while pos + hop <= n:
        aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
        s = aec.get_stats()
        sl = float(s.saturation_level)
        sr_val = float(aec._sat_detector_ref.saturation_level) if aec._sat_detector_ref else 0.0
        sm_val = float(aec._sat_detector_mic.saturation_level) if aec._sat_detector_mic else 0.0
        sat_level.append(sl)
        sat_ref_arr.append(sr_val)
        sat_mic_arr.append(sm_val)

        # F-E5 / E5-1 mic softclip
        if sm_val > 0.3:
            e5_mic_softclip_fires += 1
        # F-E5 / E5-2 main mu freeze
        if sl > 0.5:
            e5_main_mu_freeze += 1
        # F-E5 / E5-3 error_psd reset (sat → clean transition)
        if prev_sat > 0.5 and sl < 0.2:
            e5_error_psd_reset += 1
        # F-E5 / E5-4 shadow_rise mask (we don't know if a rise event fired
        # without intercepting the EPC; what we DO know is the threshold
        # condition. This is the upper bound on how often E5-4 would matter.)
        if sl > 0.5:
            e5_shadow_rise_mask += 1

        # Source-clamp + shadow safety always active counters
        if sm_val > 0.8:
            mic_clip_freeze += 1
        if sr_val > 0.1:
            softclip_ref_fires += 1
        if sl >= 0.5:
            shadow_safety_fires += 1
        prev_sat = sl
        pos += hop

    return {
        'online_sat_level':       np.asarray(sat_level, dtype=np.float64),
        'online_sat_ref':         np.asarray(sat_ref_arr, dtype=np.float64),
        'online_sat_mic':         np.asarray(sat_mic_arr, dtype=np.float64),
        'online_n_frames':        len(sat_level),
        'e5_mic_softclip_fires':  e5_mic_softclip_fires,
        'e5_main_mu_freeze':      e5_main_mu_freeze,
        'e5_error_psd_reset':     e5_error_psd_reset,
        'e5_shadow_rise_mask':    e5_shadow_rise_mask,
        'mic_clip_freeze':        mic_clip_freeze,
        'softclip_ref_fires':     softclip_ref_fires,
        'shadow_safety_fires':    shadow_safety_fires,
    }


def audit_one(case_num, stem, scenario, movement, dataset_dir, hop=160):
    t0 = time.time()
    sub = 'farend_singletalk'
    mic_path = os.path.join(dataset_dir, sub, f'{stem}_mic.wav')
    lpb_path = os.path.join(dataset_dir, sub, f'{stem}_lpb.wav')
    mic, sr_m = sf.read(mic_path)
    lpb, sr_l = sf.read(lpb_path)
    assert sr_m == sr_l, f'sr mismatch: {sr_m} vs {sr_l}'
    sr = int(sr_m)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)

    # GCC-PHAT pre-alignment (matches bench).
    gcc_delay, gcc_conf = _estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < gcc_delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[gcc_delay:] = lpb[: n - gcc_delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]
    lpb_raw = lpb[:n]  # for raw-signal sat (lpb before alignment)

    # Per-sample clip-rate (same metric as E2.S1 mic_clip_*)
    mic_stats = _signal_clip_rates('mic', mic)
    # lpb_RAW stats (BEFORE alignment) — better proxy for speaker-side
    # full-scale than aligned lpb (alignment only shifts samples, but
    # offline clip rate is shift-invariant; lpb_raw kept for clarity).
    lpb_stats = _signal_clip_rates('lpb', lpb_raw)

    # Longest consecutive-saturation run at threshold 0.95
    mic_run95 = _longest_run(np.abs(mic) > 0.95)
    lpb_run95 = _longest_run(np.abs(lpb_raw) > 0.95)
    mic_run99 = _longest_run(np.abs(mic) > 0.99)
    lpb_run99 = _longest_run(np.abs(lpb_raw) > 0.99)

    # Inter-event gap distribution (hangover proxy) on 0.95-clip events
    mic_gaps = _inter_event_gaps(np.abs(mic) > 0.95)
    lpb_gaps = _inter_event_gaps(np.abs(lpb_raw) > 0.95)

    def _gap_stats(name, gaps):
        if not gaps:
            return {f'{name}_n_events': 0,
                    f'{name}_gap_median_samp': 0,
                    f'{name}_gap_p25_samp': 0,
                    f'{name}_gap_p75_samp': 0,
                    f'{name}_gap_min_samp': 0,
                    f'{name}_gap_max_samp': 0}
        arr = np.asarray(gaps)
        return {f'{name}_n_events': int(len(gaps) + 1),  # gaps + 1 events
                f'{name}_gap_median_samp': int(np.median(arr)),
                f'{name}_gap_p25_samp': int(np.percentile(arr, 25)),
                f'{name}_gap_p75_samp': int(np.percentile(arr, 75)),
                f'{name}_gap_min_samp': int(arr.min()),
                f'{name}_gap_max_samp': int(arr.max())}

    mic_gap_stats = _gap_stats('mic', mic_gaps)
    lpb_gap_stats = _gap_stats('lpb', lpb_gaps)

    # Per-frame metrics (10 ms frames at 16 kHz)
    pf_mic = _per_frame_sat_metrics('mic_pf', mic, hop=hop)
    pf_lpb = _per_frame_sat_metrics('lpb_pf', lpb_raw, hop=hop)
    n_frames = min(pf_mic['mic_pf_n_frames'], pf_lpb['lpb_pf_n_frames'])

    # Mic↔lpb saturation alignment: per-frame Pearson r on sat_level
    # (peak per frame, in [0,1]). Tells whether mic clips when lpb
    # clips (alignment of speaker-induced clipping into mic).
    mlvl = pf_mic['mic_pf_frame_sat_level'][:n_frames]
    llvl = pf_lpb['lpb_pf_frame_sat_level'][:n_frames]
    if mlvl.std() > 1e-8 and llvl.std() > 1e-8:
        align_r = float(np.corrcoef(mlvl, llvl)[0, 1])
    else:
        align_r = 0.0

    # Per-frame coincidence rates: fraction of frames where mic≥0.95
    # AND lpb≥0.95 simultaneously (10 ms granularity)
    m95 = pf_mic['mic_pf_frame_sat_flag95'][:n_frames]
    l95 = pf_lpb['lpb_pf_frame_sat_flag95'][:n_frames]
    both95 = float(np.mean(m95 & l95))
    mic_only95 = float(np.mean(m95 & ~l95))
    lpb_only95 = float(np.mean(~m95 & l95))
    either95 = float(np.mean(m95 | l95))

    # ONLINE stream — F-E5 stays OFF (production path)
    online = _stream_online_sat(mic, lpb_aligned, sr)

    # F-E5 condition fire-rates as % of frames
    online_n = max(online['online_n_frames'], 1)
    e5_gate_rates = {
        'e5_mic_softclip_pct':  100.0 * online['e5_mic_softclip_fires'] / online_n,
        'e5_main_mu_freeze_pct':100.0 * online['e5_main_mu_freeze'] / online_n,
        'e5_error_psd_reset_n': int(online['e5_error_psd_reset']),
        'e5_shadow_rise_mask_pct': 100.0 * online['e5_shadow_rise_mask'] / online_n,
        'mic_clip_freeze_pct':  100.0 * online['mic_clip_freeze'] / online_n,
        'softclip_ref_pct':     100.0 * online['softclip_ref_fires'] / online_n,
        'shadow_safety_pct':    100.0 * online['shadow_safety_fires'] / online_n,
    }

    summary = {
        'case_num': case_num,
        'stem': stem,
        'scenario': scenario,
        'movement': movement,
        'e2_s1_group': E2_S1_GROUP.get(case_num, '?'),
        'n_samples': int(n),
        'duration_s': float(n / sr),
        'gcc_phat_initial_delay_samples': int(gcc_delay),
        'gcc_phat_initial_delay_ms': float(gcc_delay * 1000.0 / sr),
        'gcc_phat_initial_confidence': float(gcc_conf),
        # raw clip-rate (sample-level)
        **mic_stats,
        **lpb_stats,
        # longest-run (samples) at 0.95 / 0.99
        'mic_longest_run_95_samp': mic_run95,
        'lpb_longest_run_95_samp': lpb_run95,
        'mic_longest_run_99_samp': mic_run99,
        'lpb_longest_run_99_samp': lpb_run99,
        'mic_longest_run_95_ms': float(mic_run95 * 1000.0 / sr),
        'lpb_longest_run_95_ms': float(lpb_run95 * 1000.0 / sr),
        # gap stats (hangover proxy)
        **mic_gap_stats, **lpb_gap_stats,
        # per-frame alignment
        'n_frames_offline': int(n_frames),
        'mic_lpb_sat_align_r': align_r,
        'mic_lpb_both_95_pct': 100.0 * both95,
        'mic_only_95_pct': 100.0 * mic_only95,
        'lpb_only_95_pct': 100.0 * lpb_only95,
        'either_95_pct': 100.0 * either95,
        # online stream
        'online_n_frames': online['online_n_frames'],
        'online_sat_level_mean': float(online['online_sat_level'].mean()),
        'online_sat_level_p95':  float(np.percentile(online['online_sat_level'], 95)),
        'online_sat_level_max':  float(online['online_sat_level'].max()),
        'online_sat_ref_mean':   float(online['online_sat_ref'].mean()),
        'online_sat_ref_p95':    float(np.percentile(online['online_sat_ref'], 95)),
        'online_sat_ref_max':    float(online['online_sat_ref'].max()),
        'online_sat_mic_mean':   float(online['online_sat_mic'].mean()),
        'online_sat_mic_p95':    float(np.percentile(online['online_sat_mic'], 95)),
        'online_sat_mic_max':    float(online['online_sat_mic'].max()),
        # F-E5 gate fire-rates (production, F-E5 OFF)
        **e5_gate_rates,
        'elapsed_s': time.time() - t0,
    }

    # Per-frame trace (light: subsample at 10 frame intervals to keep
    # JSON small enough to inspect). We KEEP full arrays as compressed
    # int8 quantiles via percentile bins instead.
    quantiles = [0, 25, 50, 75, 90, 95, 99, 100]
    def _q(arr):
        return {f'p{q}': float(np.percentile(arr, q)) for q in quantiles}
    trace_quantiles = {
        'mic_pf_sat_level':  _q(pf_mic['mic_pf_frame_sat_level']),
        'lpb_pf_sat_level':  _q(pf_lpb['lpb_pf_frame_sat_level']),
        'online_sat_level':  _q(online['online_sat_level']),
        'online_sat_ref':    _q(online['online_sat_ref']),
        'online_sat_mic':    _q(online['online_sat_mic']),
    }
    summary['trace_quantiles'] = trace_quantiles

    return summary


def main():
    ap = argparse.ArgumentParser(description='E5.S1 saturation audit (8-case listen)')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'[e5-s1] {len(LISTEN_CASES)} listen cases, out={args.out}',
          flush=True)

    all_summaries = []
    for case_num, stem, scenario, movement in LISTEN_CASES:
        print(f'[e5-s1] case {case_num} {stem[:24]}...', flush=True)
        summary = audit_one(case_num, stem, scenario, movement, args.dataset)
        with open(os.path.join(args.out, f'{case_num}_{stem[:16]}.json'),
                  'w') as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

    # Summary table
    lines = []
    lines.append('# E5.S1 saturation audit summary')
    lines.append('')
    lines.append('Per-case saturation distribution on v3.12 worst-FS '
                 'listen 8-case (FS_static + FS_movement). Captures '
                 'raw-signal sample-level clip rates, per-frame '
                 'saturation levels, mic-lpb sat alignment, and the '
                 '`_saturation_level` smoothed value the F-E5 bundle '
                 'gates fire on.')
    lines.append('')
    lines.append('Bench config: preset=BALANCED, fl=832, cng=True, '
                 'use_kalman=True, seed=0, **F-E5 OFF** (production).')
    lines.append('')
    lines.append('## Sample-level clip rates (vs E2.S1 mic_clip_*)')
    lines.append('')
    lines.append('| # | Stem | mic peak (dBFS) | mic clip% (95/99/999) | '
                 'lpb peak (dBFS) | lpb clip% (95/99/999) | mic longest 95 run (ms) | '
                 'lpb longest 95 run (ms) | Group | Listen |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---|---|')
    for s in all_summaries:
        lines.append(
            f"| {s['case_num']} | {s['stem'][:16]} | "
            f"{s['mic_peak_dbfs']:.2f} | "
            f"{s['mic_clip_95']*100:.2f}/{s['mic_clip_99']*100:.2f}/"
            f"{s['mic_clip_999']*100:.3f}% | "
            f"{s['lpb_peak_dbfs']:.2f} | "
            f"{s['lpb_clip_95']*100:.2f}/{s['lpb_clip_99']*100:.2f}/"
            f"{s['lpb_clip_999']*100:.3f}% | "
            f"{s['mic_longest_run_95_ms']:.2f} | "
            f"{s['lpb_longest_run_95_ms']:.2f} | "
            f"{s['e2_s1_group']} | "
            f"{LISTEN_NOTES.get(s['case_num'], '')} |"
        )
    lines.append('')
    lines.append('## Per-frame mic-lpb alignment & online smoothed saturation')
    lines.append('')
    lines.append('| # | mic-lpb sat r | both>0.95% | mic-only>0.95% | '
                 'lpb-only>0.95% | online sat mean | online sat p95 | online sat max |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for s in all_summaries:
        lines.append(
            f"| {s['case_num']} | {s['mic_lpb_sat_align_r']:.3f} | "
            f"{s['mic_lpb_both_95_pct']:.2f}% | "
            f"{s['mic_only_95_pct']:.2f}% | "
            f"{s['lpb_only_95_pct']:.2f}% | "
            f"{s['online_sat_level_mean']:.3f} | "
            f"{s['online_sat_level_p95']:.3f} | "
            f"{s['online_sat_level_max']:.3f} |"
        )
    lines.append('')
    lines.append('## F-E5 gate fire-rates (production, F-E5 OFF — would-fire characterization)')
    lines.append('')
    lines.append('These columns answer "if F-E5 were ON, what fraction of '
                 'frames would each gate fire on?" Existing always-on gates '
                 '(mic clip freeze ≥0.8, softclip ref ≥0.1, shadow safety '
                 '≥0.5) are shown for context.')
    lines.append('')
    lines.append('| # | E5-1 mic softclip (sat_mic>0.3) | E5-2 mu freeze (sat>0.5) | '
                 'E5-3 sat→clean reset (events) | E5-4 shadow_rise mask (sat>0.5) | '
                 '[mic clip freeze >0.8] | [softclip ref >0.1] | [shadow safety >0.5] |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for s in all_summaries:
        lines.append(
            f"| {s['case_num']} | "
            f"{s['e5_mic_softclip_pct']:.2f}% | "
            f"{s['e5_main_mu_freeze_pct']:.2f}% | "
            f"{s['e5_error_psd_reset_n']} | "
            f"{s['e5_shadow_rise_mask_pct']:.2f}% | "
            f"{s['mic_clip_freeze_pct']:.2f}% | "
            f"{s['softclip_ref_pct']:.2f}% | "
            f"{s['shadow_safety_pct']:.2f}% |"
        )
    lines.append('')
    lines.append('## Interpretation cues')
    lines.append('')
    lines.append('- **sample-level clip% > 1%**: literal full-scale clipping audible')
    lines.append('- **mic-lpb sat r > 0.3**: mic clips co-occur with lpb clips → '
                 'speaker → mic acoustic path saturates together (typical NL profile)')
    lines.append('- **online sat level mean > 0.1**: production `_saturation_level` '
                 'smoothed value spends significant time above the softclip-ref gate')
    lines.append('- **E5-2 mu freeze % > 1%**: F-E5 mu freeze would meaningfully '
                 'change the filter trajectory on this case')
    lines.append('- **softclip ref pct vs E5-1 mic softclip pct**: asymmetry — '
                 'large delta means production already soft-clips ref but not mic')
    lines.append('')

    summary_path = os.path.join(args.out, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[e5-s1] wrote {summary_path}')

    # Aggregated JSON
    with open(os.path.join(args.out, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
