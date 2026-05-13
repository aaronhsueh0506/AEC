#!/usr/bin/env python3
"""E2.S2 Path 1 — filter_length scan on listen 8-case.

Renders the 8 worst-FS listen cases at multiple `filter_length`
values to quantify dataset coverage as function of filter coverage.
This is Python-only research; C-port stays at fl=832 per v3.13 plan.

Pre-alignment: SAME as bench harness (GCC-PHAT _estimate_delay).
This isolates the question "given current pre-alignment, how much
fl is needed to cover residual delay?"

Output dirs: results/v3_13_e2_s2_fl_{N}/<stem>_ours.wav
Summary: results/v3_13_e2_s2_fl_scan/ (after AECMOS scoring)

Per E2.S1 verdict, expected outcomes:
  fl=832:   covers nothing in Group A (case 01/04/05/08)
  fl=2080:  covers nothing (residuals 2647+)
  fl=4160:  covers case 05 (residual 2647)
  fl=6400:  covers cases 05 + (partial) 08 (7634)
  fl=12288: covers all Group A
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
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',                False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',                False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',                False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',                False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',  True ),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',  True ),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  True ),
]


def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    """Bit-for-bit port of eval_aec_challenge.estimate_delay."""
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


def render_one(stem, movement, dataset, out_dir, fl):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    sub = 'farend_singletalk'
    mic, sr = sf.read(os.path.join(dataset, sub, f'{stem}_mic.wav'))
    lpb, _  = sf.read(os.path.join(dataset, sub, f'{stem}_lpb.wav'))
    mic = mic.astype(np.float32); lpb = lpb.astype(np.float32)
    delay = _estimate_delay(mic, lpb, sr)
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
        filter_length=fl,
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
    return stem, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description='E2.S2 Path 1: fl scan on listen 8-case')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out-root', default='results')
    ap.add_argument('--fl-list', nargs='+', type=int,
                    default=[832, 2080, 4160, 6400, 12288])
    args = ap.parse_args()

    for fl in args.fl_list:
        out_dir = os.path.join(args.out_root, f'v3_13_e2_s2_fl_{fl}')
        os.makedirs(out_dir, exist_ok=True)
        print(f'[e2-s2] fl={fl} → {out_dir}', flush=True)
        t0 = time.time()
        for _, stem, mvt in LISTEN_CASES:
            _, dt = render_one(stem, mvt, args.dataset, out_dir, fl)
            print(f'[e2-s2] fl={fl} {stem[:24]} {dt:.1f}s', flush=True)
        print(f'[e2-s2] fl={fl} DONE in {time.time()-t0:.0f}s', flush=True)
    print('[e2-s2] all fl values done')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
