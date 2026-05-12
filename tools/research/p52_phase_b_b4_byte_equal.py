"""P52 Phase B Task B.4 — full 800-case byte-equal sweep.

Run AEC on ALL 800 AEC-Challenge cases twice (legacy ResFilter vs
ResFilterRefactored), compare per-sample float32 output across the full
corpus.

Hard bar (P52 §3.6):
  ≥ 99.99% (frame, bin) within atol=1e-6, rtol=1e-5
  Zero frames > 0.1 dB voice-band mean gain difference (sample-domain
  proxy: zero cases with mean |delta| > 0.1 dB)

Internal advisory target: 100% exact match (zero drift).

Usage:
  python tools/research/p52_phase_b_b4_byte_equal.py snapshot --tag legacy     --out /tmp/p52_b4/legacy.npz -j 4
  python tools/research/p52_phase_b_b4_byte_equal.py snapshot --tag refactored --out /tmp/p52_b4/refactored.npz -j 4
  python tools/research/p52_phase_b_b4_byte_equal.py diff --pre legacy.npz --post refactored.npz --out verdict.json
"""
from __future__ import annotations

import argparse
import json
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
    cases = discover()
    if args.limit:
        cases = cases[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f'[{args.tag}] {len(cases)} cases j={args.jobs}', flush=True)
    t0 = time.time()
    results = {}
    jobs = [(s, sub, m, l, args.tag) for (s, sub, m, l) in cases]
    with Pool(args.jobs) as p:
        for i, (stem, out) in enumerate(p.imap_unordered(_run_one, jobs)):
            results[stem] = out
            if (i + 1) % 50 == 0 or (i + 1) == len(cases):
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
    case_mean_db_drift = {}
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
        # mean dB drift (sample-domain proxy): 20·log10(rms(b)/rms(a))
        rms_a = float(np.sqrt(np.mean(a * a) + 1e-20))
        rms_b = float(np.sqrt(np.mean(b * b) + 1e-20))
        case_mean_db_drift[stem] = 20.0 * float(np.log10(rms_b / rms_a))
    frac_exact = exact / max(1, total)
    frac_close = close / max(1, total)
    over_01db = sum(1 for v in case_mean_db_drift.values() if abs(v) > 0.1)
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
        'cases_over_0.1dB_mean_drift': over_01db,
        'pass_zero_over_0.1dB': over_01db == 0,
        'top10_max_delta': dict(sorted(case_max.items(), key=lambda kv: -kv[1])[:10]),
        'top10_db_drift': dict(sorted(case_mean_db_drift.items(),
                                       key=lambda kv: -abs(kv[1]))[:10]),
    }
    print(json.dumps({k: v for k, v in summary.items()
                       if k not in ('top10_max_delta', 'top10_db_drift')},
                      indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
    sys.exit(0 if (summary['pass_99_99'] and summary['pass_zero_over_0.1dB']) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('snapshot')
    a.add_argument('--tag', required=True, choices=['legacy', 'refactored'])
    a.add_argument('--out', required=True)
    a.add_argument('-j', '--jobs', type=int, default=4)
    a.add_argument('--limit', type=int, default=0)
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
