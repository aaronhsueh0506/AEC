"""P52 Phase B Task B.5 — final flag-path validation.

Run AEC on 800 cases twice through the production `use_res_refactored`
config flag (NOT via monkey-patch). Snapshot once with flag=False (must
equal pre-B.5 production) and once with flag=True (must equal B.4
refactored result). Then a 50-case timing comparison.

Usage:
  python tools/research/p52_phase_b_b5_final.py snapshot --flag {false,true} --out path
  python tools/research/p52_phase_b_b5_final.py timing --flag {false,true} --out path
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
CNG_SEED = 42
TIMING_SAMPLE_SIZE = 50
TIMING_SAMPLE_SEED = 42


def discover():
    out = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir(): continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                out.append((stem, subset, str(mic), str(lpb)))
    return out


def _run_one(args):
    stem, subset, mic_p, lpb_p, use_flag, capture_time = args
    np.random.seed(CNG_SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=sr, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True, use_res_refactored=use_flag,
    )
    a = AEC(cfg)
    hop = a.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    t0 = time.perf_counter() if capture_time else 0.0
    while pos + hop <= n:
        out[pos:pos + hop] = a.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    elapsed = (time.perf_counter() - t0) if capture_time else 0.0
    return stem, out[:pos], elapsed, type(a.res).__name__


def cmd_snapshot(args):
    cases = discover()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    flag = (args.flag == 'true')
    print(f'[flag={flag}] {len(cases)} cases j={args.jobs}', flush=True)
    t0 = time.time()
    results = {}
    jobs = [(s, sub, m, l, flag, False) for (s, sub, m, l) in cases]
    cls_seen = set()
    with Pool(args.jobs) as p:
        for i, (stem, out, _, cls_name) in enumerate(p.imap_unordered(_run_one, jobs)):
            results[stem] = out
            cls_seen.add(cls_name)
            if (i + 1) % 100 == 0 or (i + 1) == len(cases):
                print(f'  {i+1}/{len(cases)} ({time.time()-t0:.0f}s)', flush=True)
    np.savez(args.out, **results)
    print(f'[flag={flag}] wrote {args.out} ({time.time()-t0:.0f}s) res_classes={cls_seen}', flush=True)


def cmd_timing(args):
    cases = random.Random(TIMING_SAMPLE_SEED).sample(discover(), TIMING_SAMPLE_SIZE)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    flag = (args.flag == 'true')
    print(f'[timing flag={flag}] {len(cases)} cases sequential', flush=True)
    rows = []
    # Warmup
    np.random.seed(0)
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=16000, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True, use_res_refactored=flag,
    )
    warm = AEC(cfg)
    warm.process(np.zeros(warm.hop_size, dtype=np.float32),
                 np.zeros(warm.hop_size, dtype=np.float32))
    t_all = time.time()
    for stem, subset, mic_p, lpb_p in cases:
        _, _, elapsed, _ = _run_one((stem, subset, mic_p, lpb_p, flag, True))
        rows.append({'stem': stem, 'subset': subset, 'elapsed_s': elapsed})
    total = sum(r['elapsed_s'] for r in rows)
    print(f'[timing flag={flag}] total={total:.3f}s ({time.time()-t_all:.1f}s wall)', flush=True)
    with open(args.out, 'w') as f:
        json.dump({'flag': flag, 'total_s': total, 'rows': rows}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('snapshot')
    a.add_argument('--flag', required=True, choices=['true', 'false'])
    a.add_argument('--out', required=True)
    a.add_argument('-j', '--jobs', type=int, default=4)
    a.set_defaults(func=cmd_snapshot)
    b = sub.add_parser('timing')
    b.add_argument('--flag', required=True, choices=['true', 'false'])
    b.add_argument('--out', required=True)
    b.set_defaults(func=cmd_timing)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
