#!/usr/bin/env python3
"""E2.S3 — F-DelayTrack `delay_est_period_s` scan on FS_movement bucket.

Per hazy-lynx plan: scan period ∈ {0.05, 0.10, 0.25 (production), 0.50}
to find best for movement cases. Path 3 (max_delay_ms=1024) is ON
for all variants — we're evaluating the online tracker period on
top of the fixed pre-alignment.

Per Phase 0 discipline (linear-first sequencing): evaluate via linear
ERLE on `_ours_nores.wav` (RES-disabled), NOT AECMOS post-RES. AECMOS
will mask front-end gains via the RES press-NE behavior identified
in E2.S2.

Output per case: `<stem>_ours.wav` (full) + `<stem>_ours_nores.wav`
(linear) in `results/v3_13_e2_s3_period_{N}ms/`.

Linear ERLE aggregation: `tools/research/e2_s3_linear_summary.py` post-processes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


def _estimate_delay(mic, ref, sr, max_delay_ms=1024.0):
    """Path 3 extended pre-align (max=1024 to match online tracker)."""
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
    return delay


def list_fs_movement_cases(dataset):
    """Return list of (stem, mic_path, lpb_path) for FS_movement bucket."""
    sub = os.path.join(dataset, 'farend_singletalk')
    cases = []
    for f in sorted(os.listdir(sub)):
        if f.endswith('_mic.wav') and '_with_movement_' in f:
            stem = f[:-len('_mic.wav')]
            mic_p = os.path.join(sub, f)
            lpb_p = os.path.join(sub, f.replace('_mic.wav', '_lpb.wav'))
            if os.path.isfile(lpb_p):
                cases.append((stem, mic_p, lpb_p))
    return cases


def render_one(stem, mic_p, lpb_p, out_dir, period_s):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    mic, sr = sf.read(mic_p)
    lpb, _  = sf.read(lpb_p)
    mic = mic.astype(np.float32); lpb = lpb.astype(np.float32)
    delay = _estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[delay:] = lpb[: n - delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]

    # Full pipeline (ours)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=int(sr), mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True,
        enable_delay_est=True,
        delay_est_period_s=period_s,
        delay_est_init_s=0.2,
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
    sf.write(os.path.join(out_dir, f'{stem}_ours.wav'), out, int(sr),
             subtype='PCM_16')

    # Linear-only (no RES) — same delay/period config
    cfg_lin = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=int(sr), mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=False,
        enable_cng=True, use_kalman=True,
        enable_delay_est=True,
        delay_est_period_s=period_s,
        delay_est_init_s=0.2,
    )
    np.random.seed(0)
    aec_lin = AEC(cfg_lin)
    out_lin = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out_lin[pos: pos + hop] = aec_lin.process(mic[pos: pos + hop],
                                                   lpb_aligned[pos: pos + hop])
        pos += hop
    sf.write(os.path.join(out_dir, f'{stem}_ours_nores.wav'), out_lin,
             int(sr), subtype='PCM_16')
    return stem, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description='E2.S3 delay_est_period_s scan on FS_movement')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out-root', default='results')
    ap.add_argument('--periods', nargs='+', type=float,
                    default=[0.05, 0.10, 0.25, 0.50])
    args = ap.parse_args()

    cases = list_fs_movement_cases(args.dataset)
    print(f'[e2-s3] {len(cases)} FS_movement cases × {len(args.periods)} periods',
          flush=True)

    for period in args.periods:
        period_ms = int(period * 1000)
        out_dir = os.path.join(args.out_root,
                               f'v3_13_e2_s3_period_{period_ms}ms')
        os.makedirs(out_dir, exist_ok=True)
        print(f'[e2-s3] period={period}s → {out_dir}', flush=True)
        t0 = time.time()
        for i, (stem, mic_p, lpb_p) in enumerate(cases):
            try:
                _, dt = render_one(stem, mic_p, lpb_p, out_dir, period)
            except Exception as e:
                print(f'[e2-s3] period={period} {stem[:24]} ERR: {e}',
                      flush=True)
                continue
            if (i + 1) % 20 == 0:
                print(f'[e2-s3] period={period} {i+1}/{len(cases)} '
                      f'last={dt:.1f}s', flush=True)
        print(f'[e2-s3] period={period} DONE in {time.time()-t0:.0f}s',
              flush=True)
    print('[e2-s3] all periods done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
