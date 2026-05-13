#!/usr/bin/env python3
"""E2.S2 Path 3 — test extended bench pre-alignment search range.

Hypothesis: bench `eval_aec_challenge.estimate_delay` uses
`max_delay_ms=250.0` (4000 samples). Group A cases (01/04/05/08)
have true delays 5653-10882 samples → PHAT peak lies OUTSIDE search
range → confidence collapses → fallback to non-PHAT returns garbage.

The online DelayEstimator inside AEC uses `max_delay_ms=1024.0`
(16384 samples), so the residual gets found correctly inside the
AEC. But the bench pre-alignment NEVER re-aligns lpb based on the
residual.

Fix: extend bench `estimate_delay` max_delay_ms from 250 → 1024
to match the online tracker.

This script:
  1. Re-renders 8 listen cases with max_delay_ms=1024 in pre-align
  2. Compares delay value and (more importantly) processed audio
     against E2.S1 baseline
  3. If Group A residuals collapse → search-range was the bottleneck

Output: results/v3_13_e2_s2_extended_align/<stem>_ours.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


LISTEN_CASES = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',                False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',                False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',                False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',                False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',  True ),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',  True ),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  True ),
]


def estimate_delay_extended(mic, ref, sr, max_delay_ms=1024.0):
    """Same algorithm as eval_aec_challenge.estimate_delay, but with
    max_delay_ms parameterizable.
    """
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


def render_one(stem, movement, dataset, out_dir, max_delay_ms):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    sub = 'farend_singletalk'
    mic, sr = sf.read(os.path.join(dataset, sub, f'{stem}_mic.wav'))
    lpb, _  = sf.read(os.path.join(dataset, sub, f'{stem}_lpb.wav'))
    mic = mic.astype(np.float32); lpb = lpb.astype(np.float32)
    delay, conf = estimate_delay_extended(mic, lpb, sr, max_delay_ms)
    n = min(len(mic), len(lpb))
    if 0 < delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[delay:] = lpb[: n - delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]
    if movement:
        delay_kw = dict(enable_delay_est=True, delay_est_period_s=0.25,
                        delay_est_init_s=0.2)
    else:
        delay_kw = dict(enable_delay_est=False)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=int(sr), mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True, **delay_kw,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos: pos + hop] = aec.process(mic[pos: pos + hop],
                                           lpb_aligned[pos: pos + hop])
        pos += hop
    out_path = os.path.join(out_dir, f'{stem}_ours.wav')
    sf.write(out_path, out, int(sr), subtype='PCM_16')
    return stem, delay, conf, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description='E2.S2 Path 3: extended pre-align')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out-root', default='results')
    ap.add_argument('--max-delay-ms', type=float, default=1024.0,
                    help='match online DelayEstimator default')
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, 'v3_13_e2_s2_extended_align')
    os.makedirs(out_dir, exist_ok=True)

    print(f'[e2-s2-ext] max_delay_ms={args.max_delay_ms} (vs bench default 250) '
          f'→ {out_dir}', flush=True)
    t0 = time.time()
    for num, stem, mvt in LISTEN_CASES:
        _, dly, conf, dt = render_one(stem, mvt, args.dataset, out_dir,
                                      args.max_delay_ms)
        print(f'[e2-s2-ext] {num} {stem[:24]:24s} '
              f'delay={dly:5d} conf={conf:.2f} {dt:.1f}s', flush=True)
    print(f'[e2-s2-ext] DONE in {time.time()-t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
