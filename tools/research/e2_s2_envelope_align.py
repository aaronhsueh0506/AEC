#!/usr/bin/env python3
"""E2.S2 Path 2 — envelope-based pre-alignment prototype.

Group A (cases 01/04/05/08) have low GCC-PHAT confidence on raw
mic/lpb because speaker NL distortion breaks the linear cross-
correlation relationship. The amplitude envelope (RMS / Hilbert
magnitude), however, is preserved through NL: speech bursts still
produce envelope peaks in mic at the same temporal offset as lpb.

Pipeline:
  1. Compute short-time RMS envelope of mic and lpb (10 ms hop)
  2. GCC-PHAT on the envelopes → coarse delay in envelope frames
  3. Convert envelope-frame delay back to sample delay
  4. Sub-sample refinement: phase-based GCC-PHAT on original signals
     within ±hop-samples window around the coarse peak
  5. Compare against GCC-PHAT raw + online tracker (E2.S1 audit values)

Goal: produce a robust delay estimate for Group A cases where current
pre-alignment fails. If envelope method delivers, online tracker
residual collapses → filter_length=832 sufficient → no C-port memory
hit needed.

Output: results/v3_13_e2_s2_envelope/summary.md + per-case JSON
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


LISTEN_CASES = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                False, 10882),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',                False,  1475),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',                False,  2523),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',                False,  9855),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',                False,  5653),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',  True,   1701),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',  True,    644),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  True,  10690),
]
# Last column = E2.S1 estimated true delay (GCC pre-align + online residual)


def _gcc_phat(a, b, max_d):
    """Plain GCC-PHAT, returns (peak_idx, peak_val, par)."""
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    A = np.fft.rfft(a, n=fft_size)
    B = np.fft.rfft(b, n=fft_size)
    cross = A * np.conj(B)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val = np.max(np.abs(xcorr[:max_search + 1]))
    peak_idx = int(np.argmax(np.abs(xcorr[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr[:max_search + 1] ** 2))
    par = float(peak_val / (rms + 1e-10))
    return peak_idx, float(peak_val), par


def _short_time_rms(x, win_size, hop_size):
    """Sliding-window RMS envelope."""
    n_frames = (len(x) - win_size) // hop_size + 1
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop_size: i * hop_size + win_size]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
    return out


def envelope_align(mic, lpb, sr, env_hop_ms=10.0, env_win_ms=20.0,
                   max_delay_ms=700.0):
    """Two-stage envelope → sub-sample alignment.

    Returns dict with envelope_coarse_samples, refined_samples,
    raw_gcc_samples (for comparison), per-stage PAR, conf.
    """
    n = min(len(mic), len(lpb))
    mic = mic[:n]; lpb = lpb[:n]

    # --- Stage 0: raw GCC-PHAT (baseline) ---
    max_d_raw = int(max_delay_ms * sr / 1000)
    raw_idx, raw_peak, raw_par = _gcc_phat(mic, lpb, max_d_raw)

    # --- Stage 1: envelope GCC-PHAT ---
    env_hop = int(env_hop_ms * sr / 1000)
    env_win = int(env_win_ms * sr / 1000)
    mic_env = _short_time_rms(mic, env_win, env_hop)
    lpb_env = _short_time_rms(lpb, env_win, env_hop)

    # Search envelope in frame domain
    max_d_env = int(max_delay_ms * sr / 1000 / env_hop)
    env_idx, env_peak, env_par = _gcc_phat(mic_env, lpb_env, max_d_env)
    env_coarse_samp = env_idx * env_hop

    # --- Stage 2: sub-sample refinement near envelope peak ---
    # Within ±env_hop window of envelope-found peak, run phase-GCC on raw
    search_low = max(0, env_coarse_samp - env_hop)
    search_high = min(n - 1, env_coarse_samp + env_hop)
    # For sub-sample refine: re-run GCC on full-resolution within window
    # Implementation: simple — shift lpb by candidate samples, compute
    # correlation peak in the local window
    # (For prototype, snap to env_coarse rather than refine)
    refined_samp = env_coarse_samp

    # --- Compute refined PAR on original signals via local GCC ---
    # Mask: only search ±env_hop range
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    A = np.fft.rfft(mic.astype(np.float64), n=fft_size)
    B = np.fft.rfft(lpb.astype(np.float64), n=fft_size)
    cross = A * np.conj(B)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    win_corr = xcorr_phat[search_low: search_high + 1]
    if len(win_corr) > 0:
        local_idx = int(np.argmax(np.abs(win_corr)))
        refined_samp = search_low + local_idx
        local_peak = float(np.abs(win_corr[local_idx]))
        local_rms = float(np.sqrt(np.mean(win_corr ** 2)))
        local_par = local_peak / (local_rms + 1e-10)
    else:
        local_par = 0.0

    return {
        'raw_gcc_samples': int(raw_idx),
        'raw_gcc_par': raw_par,
        'env_coarse_samples': int(env_coarse_samp),
        'env_par': env_par,
        'env_hop_samples': env_hop,
        'env_win_samples': env_win,
        'refined_samples': int(refined_samp),
        'refined_par_local': float(local_par),
    }


def audit_one(case_num, stem, movement, true_delay, dataset_dir):
    sub = 'farend_singletalk'
    mic, sr_m = sf.read(os.path.join(dataset_dir, sub, f'{stem}_mic.wav'))
    lpb, sr_l = sf.read(os.path.join(dataset_dir, sub, f'{stem}_lpb.wav'))
    assert sr_m == sr_l
    sr = int(sr_m)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    n = min(len(mic), len(lpb))

    result = envelope_align(mic[:n], lpb[:n], sr)
    result['case_num'] = case_num
    result['stem'] = stem
    result['movement'] = movement
    result['true_delay_e2s1'] = int(true_delay)
    result['err_raw_gcc'] = int(result['raw_gcc_samples'] - true_delay)
    result['err_env_coarse'] = int(result['env_coarse_samples'] - true_delay)
    result['err_refined'] = int(result['refined_samples'] - true_delay)
    return result


def main():
    ap = argparse.ArgumentParser(description='E2.S2 Path 2: envelope pre-align')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', default='results/v3_13_e2_s2_envelope')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f'[e2-s2-env] {len(LISTEN_CASES)} cases, out={args.out}', flush=True)
    all_results = []
    for case_num, stem, movement, true_delay in LISTEN_CASES:
        t0 = time.time()
        r = audit_one(case_num, stem, movement, true_delay, args.dataset)
        dt = time.time() - t0
        print(f'[e2-s2-env] {case_num} {stem[:24]:24s} '
              f'true={r["true_delay_e2s1"]:5d} '
              f'raw={r["raw_gcc_samples"]:5d}(err{r["err_raw_gcc"]:+d}) '
              f'env={r["env_coarse_samples"]:5d}(err{r["err_env_coarse"]:+d}) '
              f'ref={r["refined_samples"]:5d}(err{r["err_refined"]:+d}) '
              f'{dt:.1f}s', flush=True)
        all_results.append(r)
        with open(os.path.join(args.out, f'{case_num}_{stem[:16]}.json'),
                  'w') as f:
            json.dump(r, f, indent=2)

    # Summary table
    lines = []
    lines.append('# E2.S2 Path 2 — envelope-based pre-alignment results')
    lines.append('')
    lines.append('Cases 01/04/05/08 = Group A (delay-broken, low raw-GCC conf).')
    lines.append('Goal: envelope method should put **err_env_coarse**')
    lines.append('within ±500 samples of true delay (≈ filter_length/2).')
    lines.append('')
    lines.append('## Per-case')
    lines.append('')
    lines.append('| # | Stem | Group | True | Raw GCC (samp / err) | Env coarse (samp / err) | Refined (samp / err) | Env PAR | Raw PAR |')
    lines.append('|---|---|---|---:|---|---|---|---:|---:|')
    for r in all_results:
        group = 'A' if r['case_num'] in ('01','04','05','08') else 'B'
        lines.append(
            f"| {r['case_num']} | {r['stem'][:16]} | {group} | "
            f"{r['true_delay_e2s1']} | "
            f"{r['raw_gcc_samples']} / {r['err_raw_gcc']:+d} | "
            f"{r['env_coarse_samples']} / {r['err_env_coarse']:+d} | "
            f"{r['refined_samples']} / {r['err_refined']:+d} | "
            f"{r['env_par']:.2f} | {r['raw_gcc_par']:.2f} |"
        )
    lines.append('')
    lines.append('## Verdict criteria')
    lines.append('')
    lines.append('- **Group A success**: |err_env_coarse| < 500 samples on all 4 cases (01/04/05/08)')
    lines.append('- **Group B preserved**: |err_env_coarse| < 500 samples on 02/03/06/07 (must not regress)')
    lines.append('- **If both hold**: envelope alignment is a valid C-port candidate; fl=832 stays')
    lines.append('- **If only Group B preserved**: envelope misses Group A → need different approach')
    lines.append('- **If Group B regresses**: envelope is overfit / unstable → abandon')

    summary = os.path.join(args.out, 'summary.md')
    with open(summary, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[e2-s2-env] wrote {summary}')

    with open(os.path.join(args.out, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
