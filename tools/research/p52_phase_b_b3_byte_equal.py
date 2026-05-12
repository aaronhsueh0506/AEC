"""P52 Phase B Task B.3 — Module 1 ResidualEstimator byte-equal validation.

Run AEC on 100 deterministically-sampled cases (random.Random(42).sample(...))
twice:

  legacy       production ResFilter (`_stage_residual_model` in-class)
  refactored   `ResFilterRefactored` subclass with `_stage_residual_model`
               delegated to `residual_estimator()` in res_refactored package

Compare per-frame, per-sample float32 output.

Hard bar (P52 §3.6):
  ≥ 99.99% samples within atol=1e-6, rtol=1e-5
  zero cases with mean |delta| > 0.1 dB voice-band

Internal target (advisory): 100% exact match (zero drift).

Usage:
  python tools/research/p52_phase_b_b3_byte_equal.py snapshot --tag legacy     --out /tmp/p52_b3/legacy.npz
  python tools/research/p52_phase_b_b3_byte_equal.py snapshot --tag refactored --out /tmp/p52_b3/refactored.npz
  python tools/research/p52_phase_b_b3_byte_equal.py diff --pre legacy.npz --post refactored.npz --out verdict.json
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

import aec  # noqa: E402
from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SAMPLE_SIZE = 100
SAMPLE_SEED = 42
CNG_SEED = 42


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


def sample_cases():
    cases = discover()
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(cases, SAMPLE_SIZE)


def _run_one(args):
    stem, subset, mic_p, lpb_p, tag = args
    if tag == 'refactored':
        from res_refactored.res_filter_refactored import ResFilterRefactored
        aec.ResFilter = ResFilterRefactored
    np.random.seed(CNG_SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=sr, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True,
    )
    a = AEC(cfg)
    hop = a.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = a.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    return stem, out[:pos]


def cmd_snapshot(args):
    cases = sample_cases()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f'[{args.tag}] {len(cases)} cases (seed={SAMPLE_SEED}), j={args.jobs}', flush=True)
    t0 = time.time()
    results = {}
    jobs = [(s, sub, m, l, args.tag) for (s, sub, m, l) in cases]
    with Pool(args.jobs) as p:
        for i, (stem, out) in enumerate(p.imap_unordered(_run_one, jobs)):
            results[stem] = out
            if (i + 1) % 10 == 0 or (i + 1) == len(cases):
                dt = time.time() - t0
                print(f'  {i+1}/{len(cases)} ({dt:.0f}s)', flush=True)
    np.savez(args.out, **results)
    print(f'[{args.tag}] wrote {args.out} ({time.time()-t0:.0f}s)', flush=True)


def cmd_diff(args):
    pre = np.load(args.pre)
    post = np.load(args.post)
    common = sorted(set(pre.files) & set(post.files))
    total = 0
    exact = 0
    close = 0
    cases_byte_identical = 0
    case_max = {}
    for stem in common:
        a = pre[stem].astype(np.float32)
        b = post[stem].astype(np.float32)
        m = min(len(a), len(b))
        if m == 0: continue
        a = a[:m]; b = b[:m]
        d = np.abs(a - b)
        total += m
        exact += int(np.sum(a == b))
        close += int(np.sum(np.isclose(a, b, atol=1e-6, rtol=1e-5)))
        if np.array_equal(a, b):
            cases_byte_identical += 1
        case_max[stem] = float(d.max())
    frac_exact = exact / max(1, total)
    frac_close = close / max(1, total)
    summary = {
        'cases_total': len(common),
        'cases_byte_identical': cases_byte_identical,
        'total_samples': total,
        'samples_exact_match': exact,
        'samples_close_atol_1e-6_rtol_1e-5': close,
        'fraction_exact': frac_exact,
        'fraction_close': frac_close,
        'pass_99_99': frac_close >= 0.9999,
        'pass_100_exact': frac_exact == 1.0,
        'top10_max_delta': dict(sorted(case_max.items(), key=lambda kv: -kv[1])[:10]),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
    sys.exit(0 if summary['pass_99_99'] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('snapshot')
    a.add_argument('--tag', required=True, choices=['legacy', 'refactored'])
    a.add_argument('--out', required=True)
    a.add_argument('-j', '--jobs', type=int, default=4)
    a.set_defaults(func=cmd_snapshot)
    b = sub.add_parser('diff')
    b.add_argument('--pre', required=True)
    b.add_argument('--post', required=True)
    b.add_argument('--out', default=None)
    b.set_defaults(func=cmd_diff)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
