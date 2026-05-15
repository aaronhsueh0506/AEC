#!/usr/bin/env python3
"""Sprint 09 — Huber loss NLMS prototype (Arc H clipping-aware NLMS).

Standalone NumPy prototype comparing L2 NLMS vs Huber-loss NLMS in the
presence of clipping/saturation.  NO aec.py import; pure NumPy filter loop.

Background
----------
v3.13 E5 closure showed that 5 listen cases with severe clipping cannot be
fixed by amplitude-domain interventions (freeze, mask, gating).  The root
cause is that L2 NLMS gradient ∝ e(n)² — clipping bursts produce large
residuals → large gradient → divergence / over-shoot of filter weights.

Huber loss bounds the gradient:

    dL/de = e          if |e| ≤ δ   (quadratic regime — normal adaptation)
    dL/de = δ·sign(e)  if |e| > δ   (linear regime — clipping burst)

This caps the per-sample gradient to δ regardless of residual magnitude,
preventing the divergence trap.

Experiments
-----------
1. Synthetic clipping bench  — sine sweep + FIR room + hard-clip at 0.7.
   Controlled: we know the true filter → can measure weight-error norm.
2. Real listen cases (cases 01/02/07 from v3_12_worst_fs) — peak=1.0
   (hard-clipped mic).  Metric: ERLE trajectory + weight stability.

Both use time-domain block NLMS (frame_size = hop = 160 samples; no FFT).
This is the canonical prototype framing before any PBFDKF wiring (Sprint H.S2).

δ scan: [0.03, 0.07, 0.15, 0.30, 0.60] — covers literature 0.05–0.5 range
with one point below and one above to establish monotone behaviour.

Usage
-----
    python3 tools/research/v3_14_h_s1_huber_proto.py
    python3 tools/research/v3_14_h_s1_huber_proto.py --out /tmp/h_s1_results

Output: JSON results + summary printed to stdout + stored to out_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
LISTEN_DIR = os.path.join(
    os.path.expanduser('~'), 'Desktop', 'novatek', 'SE', 'AEC',
    'listen', 'v3_12_worst_fs',
)

# Listen cases with known heavy clipping (peak == 1.0000 in mic.wav).
# Cases selected from e5_s1_sat_audit: 01/02/07 have mic peak=1.0.
LISTEN_CASES = [
    ('01', '7GTxy_FSstatic',  '01_7GTxy_FSstatic'),
    ('02', 'IrQvq_FSstatic',  '02_IrQvq_FSstatic'),
    ('07', 'IrQvq_FSmovement','07_IrQvq_FSmovement'),
]

SR = 16000
HOP = 160          # 10 ms frame
FILTER_LEN = 256   # short filter for prototype speed; enough for room echo
MU_BASE = 0.02     # step size for both L2 and Huber NLMS
EPS_POWER = 1e-6   # regularisation denominator
DELTA_SCAN = [0.03, 0.07, 0.15, 0.30, 0.60]  # Huber δ values to scan


# ---------------------------------------------------------------------------
# Core NLMS implementations
# ---------------------------------------------------------------------------

def nlms_l2(
    mic: np.ndarray,
    ref: np.ndarray,
    filter_len: int = FILTER_LEN,
    mu: float = MU_BASE,
    eps: float = EPS_POWER,
    w_true: Optional[np.ndarray] = None,
) -> Dict:
    """Standard L2-loss NLMS (time-domain, sample-by-sample).

    Returns per-frame metrics: ERLE (dB), weight-error norm, residual power.
    """
    n = min(len(mic), len(ref))
    w = np.zeros(filter_len, dtype=np.float64)
    x_buf = np.zeros(filter_len, dtype=np.float64)  # ref delay line

    # Per-frame accumulators
    n_frames = n // HOP
    erle_db = np.full(n_frames, np.nan)
    w_err_norm = np.full(n_frames, np.nan) if w_true is not None else None
    w_norm = np.zeros(n_frames)
    mic_pwr_frames = np.zeros(n_frames)
    res_pwr_frames = np.zeros(n_frames)

    mic_f = mic[:n].astype(np.float64)
    ref_f = ref[:n].astype(np.float64)

    for fr in range(n_frames):
        s = fr * HOP
        e_frame = np.zeros(HOP)
        for i in range(HOP):
            # Shift reference into delay line
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = ref_f[s + i]

            # AEC output: filter estimate of echo
            y_hat = np.dot(w, x_buf)
            e = mic_f[s + i] - y_hat

            # Normalised step
            x_pwr = np.dot(x_buf, x_buf) + eps
            w += (mu / x_pwr) * e * x_buf

            e_frame[i] = e

        mic_blk = mic_f[s: s + HOP]
        mic_pwr = float(np.mean(mic_blk ** 2))
        res_pwr = float(np.mean(e_frame ** 2))
        mic_pwr_frames[fr] = mic_pwr
        res_pwr_frames[fr] = res_pwr
        if mic_pwr > 1e-10 and res_pwr > 1e-10:
            erle_db[fr] = 10.0 * np.log10(mic_pwr / res_pwr)
        w_norm[fr] = float(np.linalg.norm(w))
        if w_true is not None:
            w_err_norm[fr] = float(np.linalg.norm(w - w_true))

    return {
        'erle_db': erle_db,
        'w_norm': w_norm,
        'w_err_norm': w_err_norm,
        'mic_pwr': mic_pwr_frames,
        'res_pwr': res_pwr_frames,
        'w_final': w.copy(),
    }


def nlms_huber(
    mic: np.ndarray,
    ref: np.ndarray,
    delta: float,
    filter_len: int = FILTER_LEN,
    mu: float = MU_BASE,
    eps: float = EPS_POWER,
    w_true: Optional[np.ndarray] = None,
) -> Dict:
    """Huber-loss NLMS (time-domain, sample-by-sample).

    Gradient of Huber loss w.r.t. the residual e:
        g(e) = e            if |e| <= delta
        g(e) = delta*sign(e) if |e| > delta

    Weight update: w += (mu / ||x||²) * g(e) * x
    Large residuals are bounded to delta — prevents divergence during clipping.
    """
    n = min(len(mic), len(ref))
    w = np.zeros(filter_len, dtype=np.float64)
    x_buf = np.zeros(filter_len, dtype=np.float64)

    n_frames = n // HOP
    erle_db = np.full(n_frames, np.nan)
    w_err_norm = np.full(n_frames, np.nan) if w_true is not None else None
    w_norm = np.zeros(n_frames)
    mic_pwr_frames = np.zeros(n_frames)
    res_pwr_frames = np.zeros(n_frames)
    # Track how often Huber clipping fires (fraction of samples)
    huber_clip_rate = np.zeros(n_frames)

    mic_f = mic[:n].astype(np.float64)
    ref_f = ref[:n].astype(np.float64)

    for fr in range(n_frames):
        s = fr * HOP
        e_frame = np.zeros(HOP)
        clip_count = 0
        for i in range(HOP):
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = ref_f[s + i]

            y_hat = np.dot(w, x_buf)
            e = mic_f[s + i] - y_hat

            # Huber gradient (clipped residual for update)
            if abs(e) <= delta:
                g = e
            else:
                g = delta * np.sign(e)
                clip_count += 1

            x_pwr = np.dot(x_buf, x_buf) + eps
            w += (mu / x_pwr) * g * x_buf

            e_frame[i] = e  # track actual residual, not gradient

        mic_blk = mic_f[s: s + HOP]
        mic_pwr = float(np.mean(mic_blk ** 2))
        res_pwr = float(np.mean(e_frame ** 2))
        mic_pwr_frames[fr] = mic_pwr
        res_pwr_frames[fr] = res_pwr
        if mic_pwr > 1e-10 and res_pwr > 1e-10:
            erle_db[fr] = 10.0 * np.log10(mic_pwr / res_pwr)
        w_norm[fr] = float(np.linalg.norm(w))
        huber_clip_rate[fr] = clip_count / HOP
        if w_true is not None:
            w_err_norm[fr] = float(np.linalg.norm(w - w_true))

    return {
        'erle_db': erle_db,
        'w_norm': w_norm,
        'w_err_norm': w_err_norm,
        'mic_pwr': mic_pwr_frames,
        'res_pwr': res_pwr_frames,
        'w_final': w.copy(),
        'huber_clip_rate': huber_clip_rate,
    }


# ---------------------------------------------------------------------------
# Clipping burst detector — finds frames where input was clipped
# ---------------------------------------------------------------------------

def clipping_burst_mask(mic: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Per-frame boolean mask: True if frame contains clipped samples."""
    n = len(mic)
    n_frames = n // HOP
    mask = np.zeros(n_frames, dtype=bool)
    for fr in range(n_frames):
        blk = mic[fr * HOP: (fr + 1) * HOP]
        if np.any(np.abs(blk) > threshold):
            mask[fr] = True
    return mask


def recovery_time_frames(w_err: np.ndarray, burst_mask: np.ndarray,
                          threshold_ratio: float = 0.5) -> List[int]:
    """Frames to recover below threshold after each clipping burst end.

    For each burst end (burst_mask goes False → False), measure how many
    frames until w_err_norm drops back below 50% of its local peak.
    Returns list of recovery times (frames); empty if no bursts.
    """
    recovery_times = []
    in_burst = False
    burst_end = -1
    peak_err = 0.0
    for i, b in enumerate(burst_mask):
        if b:
            in_burst = True
            if w_err is not None and not np.isnan(w_err[i]):
                peak_err = max(peak_err, w_err[i])
        else:
            if in_burst:
                # Burst just ended
                burst_end = i
                tgt = peak_err * threshold_ratio
                # Look ahead for recovery
                rt = None
                for j in range(i, min(i + 500, len(w_err))):
                    if w_err is not None and not np.isnan(w_err[j]) and w_err[j] < tgt:
                        rt = j - burst_end
                        break
                if rt is not None:
                    recovery_times.append(rt)
                in_burst = False
                peak_err = 0.0
    return recovery_times


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(result: Dict, burst_mask: np.ndarray,
                      w_true: Optional[np.ndarray] = None) -> Dict:
    """Compute summary scalars from per-frame arrays."""
    erle = result['erle_db']
    valid = ~np.isnan(erle)
    w_norm = result['w_norm']

    # ERLE stats
    erle_mean = float(np.nanmean(erle)) if valid.any() else float('nan')
    erle_p50  = float(np.nanpercentile(erle, 50)) if valid.any() else float('nan')
    erle_p10  = float(np.nanpercentile(erle, 10)) if valid.any() else float('nan')

    # During / after burst ERLE (burst frames vs non-burst frames)
    erle_during_burst = float(np.nanmean(erle[burst_mask])) if burst_mask.any() else float('nan')
    erle_non_burst = float(np.nanmean(erle[~burst_mask])) if (~burst_mask).any() else float('nan')

    # W norm stability (std over time — lower = more stable)
    w_norm_std = float(np.std(w_norm))
    w_norm_max = float(np.max(w_norm))
    w_norm_mean = float(np.mean(w_norm))

    # W-error metrics (only if w_true provided)
    w_err = result.get('w_err_norm')
    if w_err is not None and not np.all(np.isnan(w_err)):
        w_err_mean = float(np.nanmean(w_err))
        w_err_peak = float(np.nanmax(w_err))
        w_err_final = float(w_err[-1]) if not np.isnan(w_err[-1]) else float('nan')
        rt = recovery_time_frames(w_err, burst_mask)
        recovery_ms_mean = float(np.mean(rt) * HOP * 1000.0 / SR) if rt else float('nan')
        recovery_ms_p75  = float(np.percentile(rt, 75) * HOP * 1000.0 / SR) if rt else float('nan')
    else:
        w_err_mean = w_err_peak = w_err_final = float('nan')
        recovery_ms_mean = recovery_ms_p75 = float('nan')

    # Huber clip rate
    hcr = result.get('huber_clip_rate')
    huber_clip_pct = float(100.0 * np.mean(hcr)) if hcr is not None else float('nan')

    return {
        'erle_mean_db': erle_mean,
        'erle_p50_db': erle_p50,
        'erle_p10_db': erle_p10,
        'erle_during_burst_db': erle_during_burst,
        'erle_non_burst_db': erle_non_burst,
        'w_norm_std': w_norm_std,
        'w_norm_max': w_norm_max,
        'w_norm_mean': w_norm_mean,
        'w_err_mean': w_err_mean,
        'w_err_peak': w_err_peak,
        'w_err_final': w_err_final,
        'recovery_ms_mean': recovery_ms_mean,
        'recovery_ms_p75': recovery_ms_p75,
        'huber_clip_pct': huber_clip_pct,
    }


# ---------------------------------------------------------------------------
# Experiment 1 — Synthetic clipping bench
# ---------------------------------------------------------------------------

def make_synthetic_clip(duration_s: float = 10.0, sr: int = SR,
                         clip_level: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic clipping scenario.

    Reference: sine sweep (100–3000 Hz).
    Room filter: random min-phase FIR (filter_len taps).
    Mic: room-filtered ref + hard-clip at clip_level + small noise.
    Returns: (mic, ref, w_true) where w_true is the room FIR.
    """
    rng = np.random.default_rng(42)
    n = int(duration_s * sr)

    # Sine sweep reference
    t = np.linspace(0, duration_s, n)
    f_start, f_end = 100.0, 3000.0
    ref = np.sin(2 * np.pi * (f_start + (f_end - f_start) / (2 * duration_s) * t) * t)
    # Amplitude modulation to create burst variations
    env = 0.5 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
    ref = ref * env * 0.8

    # Room FIR: exponential decay with random peaks (simulates multi-path)
    fir_len = FILTER_LEN
    tau = fir_len / 4
    w_true = rng.standard_normal(fir_len) * np.exp(-np.arange(fir_len) / tau)
    # Normalise so ||w_true|| = 1
    w_true /= np.linalg.norm(w_true)

    # Room echo (linear convolution, trimmed)
    echo = np.convolve(ref, w_true)[:n]

    # Mic = echo + noise; then hard-clip
    noise = rng.standard_normal(n) * 0.005
    mic_raw = echo + noise
    mic_clipped = np.clip(mic_raw, -clip_level, clip_level)

    return mic_clipped.astype(np.float32), ref.astype(np.float32), w_true.astype(np.float64)


def run_synthetic_experiment(out_dir: Optional[str] = None) -> Dict:
    """Experiment 1: synthetic clipping."""
    print('[H.S1] === Experiment 1: Synthetic clipping bench ===', flush=True)
    mic, ref, w_true = make_synthetic_clip(duration_s=15.0, clip_level=0.7)
    burst_mask = clipping_burst_mask(mic, threshold=0.65)
    n_burst_frames = int(burst_mask.sum())
    n_frames = len(burst_mask)
    print(f'[H.S1]   Signal: {len(mic)/SR:.1f}s, clip>0.65: {n_burst_frames}/{n_frames} '
          f'frames ({100*n_burst_frames/n_frames:.1f}%)', flush=True)

    results = {}

    # L2 baseline
    t0 = time.time()
    l2_res = nlms_l2(mic, ref, filter_len=FILTER_LEN, mu=MU_BASE, w_true=w_true)
    l2_metrics = aggregate_metrics(l2_res, burst_mask, w_true=w_true)
    l2_metrics['runtime_s'] = time.time() - t0
    results['L2'] = l2_metrics
    print(f'[H.S1]   L2:     ERLE={l2_metrics["erle_mean_db"]:+.2f} dB  '
          f'W_err_mean={l2_metrics["w_err_mean"]:.4f}  '
          f'W_norm_std={l2_metrics["w_norm_std"]:.4f}  '
          f'recovery={l2_metrics["recovery_ms_mean"]:.0f}ms  '
          f'({l2_metrics["runtime_s"]:.1f}s)', flush=True)

    # Huber δ scan
    for delta in DELTA_SCAN:
        t0 = time.time()
        h_res = nlms_huber(mic, ref, delta=delta, filter_len=FILTER_LEN, mu=MU_BASE, w_true=w_true)
        m = aggregate_metrics(h_res, burst_mask, w_true=w_true)
        m['runtime_s'] = time.time() - t0
        m['delta'] = delta
        key = f'Huber_d{delta:.2f}'.replace('.', 'p')
        results[key] = m
        print(f'[H.S1]   Huber δ={delta:.2f}: ERLE={m["erle_mean_db"]:+.2f} dB  '
              f'W_err_mean={m["w_err_mean"]:.4f}  '
              f'W_norm_std={m["w_norm_std"]:.4f}  '
              f'clip_fires={m["huber_clip_pct"]:.1f}%  '
              f'recovery={m["recovery_ms_mean"]:.0f}ms', flush=True)

    return results


# ---------------------------------------------------------------------------
# Experiment 2 — Real listen cases
# ---------------------------------------------------------------------------

def gcc_phat_delay(mic: np.ndarray, ref: np.ndarray,
                   sr: int = SR, max_delay_ms: float = 1024.0) -> int:
    """GCC-PHAT delay estimate (matches eval_aec_challenge pattern)."""
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
    conf = peak_val / (rms + 1e-10)
    if conf < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        return int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    return peak_idx


def run_listen_case(case_num: str, stem_short: str, prefix: str) -> Optional[Dict]:
    """Run L2 vs Huber on one listen case (mic + lpb)."""
    mic_path = os.path.join(LISTEN_DIR, f'{prefix}_mic.wav')
    lpb_path = os.path.join(LISTEN_DIR, f'{prefix}_lpb.wav')
    if not os.path.exists(mic_path) or not os.path.exists(lpb_path):
        print(f'[H.S1]   SKIP {case_num}: files not found', flush=True)
        return None

    mic, sr_m = sf.read(mic_path)
    lpb, sr_l = sf.read(lpb_path)
    assert sr_m == sr_l == SR, f'Expected sr={SR}, got mic={sr_m} lpb={sr_l}'

    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)

    # GCC-PHAT pre-alignment
    delay = gcc_phat_delay(mic, lpb, sr=SR, max_delay_ms=1024.0)
    n = min(len(mic), len(lpb))
    if 0 < delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[delay:] = lpb[: n - delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]

    mic_peak = float(np.max(np.abs(mic)))
    clip99 = float(np.mean(np.abs(mic) > 0.99) * 100)
    print(f'[H.S1]   Case {case_num} ({stem_short}): peak={mic_peak:.4f} '
          f'clip99={clip99:.2f}% delay={delay} samp ({delay*1000//SR} ms)',
          flush=True)

    burst_mask = clipping_burst_mask(mic, threshold=0.95)
    n_burst = int(burst_mask.sum())
    print(f'[H.S1]     Clipping burst frames (>0.95): {n_burst}/{len(burst_mask)} '
          f'({100*n_burst/max(1,len(burst_mask)):.1f}%)', flush=True)

    case_results: Dict = {
        'case_num': case_num,
        'stem_short': stem_short,
        'mic_peak': mic_peak,
        'clip99_pct': clip99,
        'gcc_delay_samp': delay,
        'n_burst_frames': n_burst,
        'n_frames': len(burst_mask),
    }

    # L2 baseline
    t0 = time.time()
    l2_res = nlms_l2(mic, lpb_aligned, filter_len=FILTER_LEN, mu=MU_BASE)
    m = aggregate_metrics(l2_res, burst_mask)
    m['runtime_s'] = time.time() - t0
    case_results['L2'] = m
    print(f'[H.S1]     L2: ERLE={m["erle_mean_db"]:+.2f} dB  '
          f'W_norm_std={m["w_norm_std"]:.4f}  '
          f'burst_ERLE={m["erle_during_burst_db"]:+.2f} dB', flush=True)

    # Huber δ scan
    for delta in DELTA_SCAN:
        t0 = time.time()
        h_res = nlms_huber(mic, lpb_aligned, delta=delta,
                           filter_len=FILTER_LEN, mu=MU_BASE)
        m = aggregate_metrics(h_res, burst_mask)
        m['runtime_s'] = time.time() - t0
        m['delta'] = delta
        key = f'Huber_d{delta:.2f}'.replace('.', 'p')
        case_results[key] = m
        print(f'[H.S1]     Huber δ={delta:.2f}: ERLE={m["erle_mean_db"]:+.2f} dB  '
              f'W_norm_std={m["w_norm_std"]:.4f}  '
              f'burst_ERLE={m["erle_during_burst_db"]:+.2f} dB  '
              f'clip_fires={m["huber_clip_pct"]:.1f}%', flush=True)

    return case_results


def run_listen_experiments() -> List[Dict]:
    print('\n[H.S1] === Experiment 2: Real listen cases ===', flush=True)
    all_results = []
    for case_num, stem_short, prefix in LISTEN_CASES:
        print(f'\n[H.S1] --- Case {case_num}: {stem_short} ---', flush=True)
        r = run_listen_case(case_num, stem_short, prefix)
        if r is not None:
            all_results.append(r)
    return all_results


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def _fmt(v: float, digits: int = 2) -> str:
    if v != v:  # isnan
        return 'N/A'
    return f'{v:.{digits}f}'


def print_summary_table(synth_results: Dict, listen_results: List[Dict]) -> str:
    lines = []
    lines.append('')
    lines.append('=' * 70)
    lines.append('H.S1 RESULTS SUMMARY')
    lines.append('=' * 70)

    lines.append('\n--- Synthetic clipping (w_true known, 15s, clip@0.7) ---')
    header = f"{'Variant':<20} {'ERLE_mean':>10} {'ERLE_burst':>10} {'W_err_mean':>12} {'W_norm_std':>10} {'Recov_ms':>10} {'HuberFire%':>10}"
    lines.append(header)
    lines.append('-' * 85)
    for key, m in synth_results.items():
        lines.append(
            f"{key:<20} {_fmt(m['erle_mean_db']):>10} "
            f"{_fmt(m['erle_during_burst_db']):>10} "
            f"{_fmt(m['w_err_mean'], 4):>12} "
            f"{_fmt(m['w_norm_std'], 4):>10} "
            f"{_fmt(m['recovery_ms_mean'], 0):>10} "
            f"{_fmt(m['huber_clip_pct'] if 'huber_clip_pct' in m and not (m['huber_clip_pct']!=m['huber_clip_pct']) else float('nan'), 1):>10}"
        )

    if listen_results:
        lines.append('\n--- Listen cases (real clipping, no w_true) ---')
        lines.append(f"{'Case':<8} {'Variant':<20} {'ERLE_mean':>10} {'ERLE_burst':>10} "
                     f"{'ERLE_nonburst':>13} {'W_norm_std':>10} {'HuberFire%':>10}")
        lines.append('-' * 90)
        for cr in listen_results:
            for key in ['L2'] + [f'Huber_d{d:.2f}'.replace('.', 'p') for d in DELTA_SCAN]:
                if key not in cr:
                    continue
                m = cr[key]
                lines.append(
                    f"{cr['case_num']:<8} {key:<20} "
                    f"{_fmt(m['erle_mean_db']):>10} "
                    f"{_fmt(m['erle_during_burst_db']):>10} "
                    f"{_fmt(m['erle_non_burst_db']):>13} "
                    f"{_fmt(m['w_norm_std'], 4):>10} "
                    f"{_fmt(m.get('huber_clip_pct', float('nan')), 1):>10}"
                )

    lines.append('')
    lines.append('=' * 70)
    return '\n'.join(lines)


def select_best_delta(synth_results: Dict, listen_results: List[Dict]) -> Dict:
    """Score each δ across all experiments to recommend range for H.S2."""
    # Score = (Huber ERLE - L2 ERLE) + 0.5*(L2 W_norm_std - Huber W_norm_std)
    # Higher is better.
    delta_scores: Dict[float, List[float]] = {d: [] for d in DELTA_SCAN}

    l2_synth = synth_results.get('L2', {})
    for d in DELTA_SCAN:
        key = f'Huber_d{d:.2f}'.replace('.', 'p')
        m = synth_results.get(key, {})
        if m:
            erle_gain = m.get('erle_mean_db', float('nan')) - l2_synth.get('erle_mean_db', float('nan'))
            stability_gain = l2_synth.get('w_norm_std', float('nan')) - m.get('w_norm_std', float('nan'))
            w_err_reduction = l2_synth.get('w_err_mean', float('nan')) - m.get('w_err_mean', float('nan'))
            score = erle_gain + 0.5 * stability_gain + 0.5 * w_err_reduction
            if score == score:  # not nan
                delta_scores[d].append(score)

    for cr in listen_results:
        l2_m = cr.get('L2', {})
        for d in DELTA_SCAN:
            key = f'Huber_d{d:.2f}'.replace('.', 'p')
            m = cr.get(key, {})
            if m:
                erle_gain = m.get('erle_mean_db', float('nan')) - l2_m.get('erle_mean_db', float('nan'))
                stability_gain = l2_m.get('w_norm_std', float('nan')) - m.get('w_norm_std', float('nan'))
                score = erle_gain + 0.5 * stability_gain
                if score == score:
                    delta_scores[d].append(score)

    scored = {}
    for d, scores in delta_scores.items():
        if scores:
            scored[d] = float(np.mean(scores))

    if not scored:
        return {'recommended_delta': None, 'delta_scores': {}}

    best_d = max(scored, key=lambda x: scored[x])
    # Recommend range: ±1 step around best
    deltas_sorted = sorted(scored.keys())
    idx = deltas_sorted.index(best_d)
    lo = deltas_sorted[max(0, idx - 1)]
    hi = deltas_sorted[min(len(deltas_sorted) - 1, idx + 1)]

    return {
        'recommended_delta': best_d,
        'recommended_range': [lo, hi],
        'delta_scores': scored,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='H.S1 Huber NLMS prototype')
    ap.add_argument('--out', default='/tmp/v3_14_h_s1_results',
                    help='Output directory for JSON results')
    ap.add_argument('--skip-synthetic', action='store_true',
                    help='Skip synthetic experiment (for speed during debug)')
    ap.add_argument('--skip-listen', action='store_true',
                    help='Skip listen case experiments')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t_total = time.time()

    synth_results = {}
    if not args.skip_synthetic:
        synth_results = run_synthetic_experiment(out_dir=args.out)
        with open(os.path.join(args.out, 'synthetic_results.json'), 'w') as f:
            # Convert arrays to list for JSON serialisation
            serialisable = {}
            for k, v in synth_results.items():
                serialisable[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                                   for kk, vv in v.items()}
            json.dump(serialisable, f, indent=2)

    listen_results = []
    if not args.skip_listen:
        listen_results = run_listen_experiments()
        with open(os.path.join(args.out, 'listen_results.json'), 'w') as f:
            serialisable = []
            for cr in listen_results:
                scase = {}
                for k, v in cr.items():
                    if isinstance(v, dict):
                        scase[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                                    for kk, vv in v.items()}
                    else:
                        scase[k] = v
                serialisable.append(scase)
            json.dump(serialisable, f, indent=2)

    # Summary
    summary = print_summary_table(synth_results, listen_results)
    print(summary)

    # Delta recommendation
    rec = select_best_delta(synth_results, listen_results)
    print(f'\n[H.S1] Delta recommendation: best={rec.get("recommended_delta")} '
          f'range={rec.get("recommended_range")}')
    print(f'[H.S1] Delta scores: '
          + ', '.join(f'δ={d:.2f}:{s:.3f}' for d, s in
                      sorted(rec.get('delta_scores', {}).items())))

    with open(os.path.join(args.out, 'delta_recommendation.json'), 'w') as f:
        json.dump(rec, f, indent=2)

    total_elapsed = time.time() - t_total
    print(f'\n[H.S1] Total elapsed: {total_elapsed:.1f}s')
    print(f'[H.S1] Results written to: {args.out}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
