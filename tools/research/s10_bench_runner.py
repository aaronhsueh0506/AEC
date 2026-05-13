#!/usr/bin/env python3
"""S10 bench runner — `res_noise_floor_refined` A/B harness.

Mirrors S7 bench runner pattern (eval_aec_challenge byte-equal:
GCC-PHAT delay, np.random.seed(0), preset path, hop loop) plus a
`--flag-on` switch for `res_noise_floor_refined`. S9-A.2 H1
verification: flag-ON drops `noise_floor_psd` in FS-confident bins
(coh² < 0.1) from `mean(error_psd) × 0.01` to per-bin
`error_psd × 0.005`.

Usage:
    python3 tools/research/s10_bench_runner.py \\
        --out results/v3_12_s10_off -j 4
    python3 tools/research/s10_bench_runner.py \\
        --out results/v3_12_s10_on  -j 4 --flag-on
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


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

    # Primary: GCC-PHAT (sharp peak for most cases)
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


def _run_one(mic_path: str, lpb_path: str, out_path: str,
              fl: int, enable_cng: bool, flag_on: bool, is_movement: bool):
    from aec import AEC, AecConfig, AecMode, AecPreset

    t0 = time.time()
    try:
        mic, sr_m = sf.read(mic_path)
        lpb, sr_l = sf.read(lpb_path)
        if sr_m != sr_l:
            return (os.path.basename(out_path), 0.0,
                    f'sr_mismatch:{sr_m},{sr_l}')
        sr = int(sr_m)
        mic = mic.astype(np.float32)
        lpb = lpb.astype(np.float32)

        delay = _estimate_delay(mic, lpb, sr)
        n = min(len(mic), len(lpb))
        if delay > 0 and delay < n:
            lpb_aligned = np.zeros(n, dtype=np.float32)
            lpb_aligned[delay:] = lpb[: n - delay]
        else:
            lpb_aligned = lpb[:n]
        mic = mic[:n]

        # Match eval_aec_challenge.py delay-est-per-case logic: movement
        # cases get online tracking, static cases get fixed delay only.
        if is_movement:
            delay_est_kw = dict(enable_delay_est=True,
                                delay_est_period_s=0.25,
                                delay_est_init_s=0.2)
        else:
            delay_est_kw = dict(enable_delay_est=False)

        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr, mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False, enable_shadow=True, enable_res=True,
            enable_cng=enable_cng,
            use_kalman=True,
            res_noise_floor_refined=flag_on,
            **delay_est_kw,
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

        sf.write(out_path, out, sr, subtype='PCM_16')
        return (os.path.basename(out_path), time.time() - t0, 'ok')
    except Exception as e:
        return (os.path.basename(out_path), time.time() - t0, f'err:{e}')


def _collect_cases(dataset_dir: str):
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        scen_dir = Path(dataset_dir) / scenario
        if not scen_dir.is_dir():
            continue
        for mic_f in sorted(scen_dir.glob('*_mic.wav')):
            stem = mic_f.name[: -len('_mic.wav')]
            lpb_f = scen_dir / f'{stem}_lpb.wav'
            if lpb_f.is_file():
                is_mvt = '_with_movement' in stem
                cases.append((stem, scenario, is_mvt, str(mic_f), str(lpb_f)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description='S10 res_noise_floor_refined bench runner')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    ap.add_argument('--filter', type=int, default=832)
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-cng', action='store_true')
    ap.add_argument('--flag-on', action='store_true',
                    help='Enable res_noise_floor_refined=True (default OFF)')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = _collect_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.skip_existing:
        before = len(cases)
        cases = [c for c in cases
                  if not os.path.isfile(os.path.join(args.out,
                                                      f'{c[0]}_ours.wav'))]
        print(f'[s10-bench] skip-existing: {before - len(cases)} done, '
              f'{len(cases)} to run', flush=True)
    enable_cng = not args.no_cng

    print(f'[s10-bench] cases={len(cases)} jobs={args.jobs} '
          f'flag_on={args.flag_on} cng={enable_cng} out={args.out}',
          flush=True)

    t_start = time.time()
    n_done = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for stem, scen, is_mvt, mic_p, lpb_p in cases:
            out_p = os.path.join(args.out, f'{stem}_ours.wav')
            futures[pool.submit(
                _run_one, mic_p, lpb_p, out_p,
                args.filter, enable_cng, args.flag_on, is_mvt,
            )] = stem
        for fut in as_completed(futures):
            name, dt, status = fut.result()
            n_done += 1
            if status != 'ok':
                n_err += 1
                print(f'[s10-bench] ERR {name}: {status}', flush=True)
            if n_done % 100 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[s10-bench] {n_done}/{len(cases)} errors={n_err} '
                      f'elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    total = time.time() - t_start
    print(f'[s10-bench] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{len(cases)}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
