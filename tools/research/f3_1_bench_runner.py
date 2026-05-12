#!/usr/bin/env python3
"""F3.1 bench runner — case-level parallel 800-case AEC runner.

Mirrors `eval_aec_challenge.py:run_ours` per-case behaviour (same delay
pre-alignment, same np.random.seed(0) determinism, same preset path,
same hop loop) but parallelises across cases with N workers, since
the production script only parallelises across the 3 scenarios.

Output layout matches `bench_aecmos.py` expectations: every
`<stem>_ours.wav` lands flat in `--out`. Bucket assignment is
derived by `bench_aecmos.py` from the stem suffix.

Usage:
    python3 tools/research/f3_1_bench_runner.py --out /tmp/f3_1_baseline -j 4
    AEC_USE_MIC_EXCESS=1 python3 tools/research/f3_1_bench_runner.py --out /tmp/f3_1_on -j 4
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
    """Bit-for-bit port of eval_aec_challenge.estimate_delay."""
    max_d = int(sr * max_delay_ms / 1000)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    # Cross-correlate via FFT.
    nfft = 1 << int(np.ceil(np.log2(n + max_d)))
    M = np.fft.rfft(mic, n=nfft)
    R = np.fft.rfft(ref, n=nfft)
    xc = np.fft.irfft(M * np.conj(R), n=nfft)
    candidates = xc[: max_d + 1]
    return int(np.argmax(candidates))


def _run_one(mic_path: str, lpb_path: str, out_path: str,
              fl: int, use_mic_excess: bool, enable_cng: bool) -> tuple[str, float, str]:
    """Worker entrypoint: process one (mic, lpb) pair → ours.wav.
    Returns (stem, elapsed_seconds, status)."""
    from aec import AEC, AecConfig, AecMode, AecPreset

    t0 = time.time()
    try:
        mic, sr_m = sf.read(mic_path)
        lpb, sr_l = sf.read(lpb_path)
        if sr_m != sr_l:
            return (os.path.basename(out_path), 0.0, f'sr_mismatch:{sr_m},{sr_l}')
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

        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr, mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False, enable_shadow=True, enable_res=True,
            enable_cng=enable_cng,
            use_kalman=True,
            enable_delay_est=False,
            use_mic_excess_evidence=use_mic_excess,
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


def _collect_cases(dataset_dir: str) -> list[tuple[str, str, str, str]]:
    """Return list of (stem, scenario, mic_path, lpb_path)."""
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        scen_dir = Path(dataset_dir) / scenario
        if not scen_dir.is_dir():
            continue
        # Files are `<stem>_<scenario>_mic.wav` and `<stem>_<scenario>_lpb.wav`.
        for mic_f in sorted(scen_dir.glob(f'*_{scenario}_mic.wav')):
            stem = mic_f.name.replace('_mic.wav', '')
            lpb_f = scen_dir / f'{stem}_lpb.wav'
            if lpb_f.is_file():
                cases.append((stem, scenario, str(mic_f), str(lpb_f)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description='F3.1 case-parallel bench runner')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind',
                    help='AEC Challenge blind dataset root')
    ap.add_argument('--out', required=True, help='Output dir for <stem>_ours.wav')
    ap.add_argument('--filter', type=int, default=832,
                    help='Filter length samples (default 832 = 52ms @ 16k)')
    ap.add_argument('-j', '--jobs', type=int, default=4, help='Parallel workers')
    ap.add_argument('--limit', type=int, default=None,
                    help='Cap number of cases (debug)')
    ap.add_argument('--no-cng', action='store_true',
                    help='Disable CNG (default: on)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = _collect_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    use_mic_excess = os.environ.get('AEC_USE_MIC_EXCESS', '0').lower() not in (
        '0', 'false', 'off', 'no', '')
    enable_cng = not args.no_cng

    print(f'[f3.1-bench] cases={len(cases)} jobs={args.jobs} '
          f'use_mic_excess={use_mic_excess} cng={enable_cng} out={args.out}',
          flush=True)

    t_start = time.time()
    n_done = 0
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for stem, scen, mic_p, lpb_p in cases:
            out_p = os.path.join(args.out, f'{stem}_ours.wav')
            futures[pool.submit(
                _run_one, mic_p, lpb_p, out_p,
                args.filter, use_mic_excess, enable_cng,
            )] = stem

        for fut in as_completed(futures):
            name, dt, status = fut.result()
            n_done += 1
            if status != 'ok':
                n_err += 1
                print(f'[f3.1-bench] ERR {name}: {status}', flush=True)
            if n_done % 50 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[f3.1-bench] {n_done}/{len(cases)} '
                      f'errors={n_err} elapsed={elapsed:.0f}s eta={eta:.0f}s',
                      flush=True)

    total = time.time() - t_start
    print(f'[f3.1-bench] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{len(cases)}', flush=True)
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
