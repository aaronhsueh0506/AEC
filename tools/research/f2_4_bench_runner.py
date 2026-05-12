#!/usr/bin/env python3
"""F2.4 bench runner — mu holdoff no-reset ablation.

_update_simple_mu_ratio resets holdoff=20 on every DT frame; marginal DT
(ratio oscillating around _simple_mu_ratio) keeps resetting holdoff so mu
never releases. Fix: only arm holdoff on fresh onset (holdoff==0).
Toggle via `AEC_MU_HOLDOFF_NO_RESET=1`.

Usage:
    # Baseline
    python3 tools/research/f2_4_bench_runner.py --out /tmp/f2_4_baseline -j 4

    # Ablation (holdoff no-reset)
    AEC_MU_HOLDOFF_NO_RESET=1 \
        python3 tools/research/f2_4_bench_runner.py --out /tmp/f2_4_on -j 4
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
    max_d = int(sr * max_delay_ms / 1000)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    nfft = 1 << int(np.ceil(np.log2(n + max_d)))
    M = np.fft.rfft(mic, n=nfft)
    R = np.fft.rfft(ref, n=nfft)
    xc = np.fft.irfft(M * np.conj(R), n=nfft)
    candidates = xc[: max_d + 1]
    return int(np.argmax(candidates))


def _run_one(mic_path: str, lpb_path: str, out_path: str,
             fl: int, enable_cng: bool,
             mu_holdoff_no_reset: bool = False) -> tuple[str, float, str]:
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
            mu_holdoff_no_reset=mu_holdoff_no_reset,
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
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        scen_dir = Path(dataset_dir) / scenario
        if not scen_dir.is_dir():
            continue
        for mic_f in sorted(scen_dir.glob('*_mic.wav')):
            stem = mic_f.name[: -len('_mic.wav')]
            lpb_f = scen_dir / f'{stem}_lpb.wav'
            if lpb_f.is_file():
                cases.append((stem, scenario, str(mic_f), str(lpb_f)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description='F2.4 mu holdoff no-reset ablation runner')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True, help='Output dir for <stem>_ours.wav')
    ap.add_argument('--filter', type=int, default=832)
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-cng', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = _collect_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    if args.skip_existing:
        before = len(cases)
        cases = [c for c in cases
                 if not os.path.isfile(os.path.join(args.out, f'{c[0]}_ours.wav'))]
        print(f'[f2.4-bench] skip-existing: {before - len(cases)} done, '
              f'{len(cases)} to run', flush=True)

    mu_holdoff_no_reset = os.environ.get('AEC_MU_HOLDOFF_NO_RESET', '0').lower() not in (
        '0', 'false', 'off', 'no', '')
    enable_cng = not args.no_cng

    print(f'[f2.4-bench] cases={len(cases)} jobs={args.jobs} '
          f'mu_holdoff_no_reset={mu_holdoff_no_reset} cng={enable_cng} out={args.out}',
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
                args.filter, enable_cng, mu_holdoff_no_reset,
            )] = stem

        for fut in as_completed(futures):
            name, dt, status = fut.result()
            n_done += 1
            if status != 'ok':
                n_err += 1
                print(f'[f2.4-bench] ERR {name}: {status}', flush=True)
            if n_done % 50 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[f2.4-bench] {n_done}/{len(cases)} '
                      f'errors={n_err} elapsed={elapsed:.0f}s eta={eta:.0f}s',
                      flush=True)

    total = time.time() - t_start
    print(f'[f2.4-bench] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{len(cases)}', flush=True)
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
