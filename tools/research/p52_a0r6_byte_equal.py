"""P52 A.0R.6 — 800-case byte-equal sanity for rename + trace flag.

Run aec.py on all 800 cases with deterministic CNG seed, capture per-frame
samples of the output to a tagged snapshot. The two snapshots compared:

  pre   commit 410c238 (postmortem HEAD; pre-rename ShadowCopyController)
  post  current HEAD   (PathChangeRegimeHandler, trace flag default-OFF)

Byte-equal criterion: ≥ 99.99 % of all samples within float32 epsilon
(atol=1e-6, rtol=1e-5). Stricter — exact identity — is preferred.

Two subcommands:

  snapshot --tag {pre,post} --out path  emit per-case 1-D float32 output to
                                         a single npz keyed by stem
  diff    --pre p --post p              compute per-case max|delta|,
                                         frame-equal fraction, write JSON
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

from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SEED = 42


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


def _one(args):
    stem, subset, mic_p, lpb_p = args
    np.random.seed(SEED)
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
    aec = AEC(cfg)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    return stem, out[:pos]


def cmd_snapshot(args):
    cases = discover()
    if args.limit:
        cases = cases[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f'[{args.tag}] {len(cases)} cases, j={args.jobs}', flush=True)
    t0 = time.time()
    results = {}
    with Pool(args.jobs) as p:
        for i, (stem, out) in enumerate(p.imap_unordered(_one, cases)):
            results[stem] = out
            if (i + 1) % 50 == 0 or (i + 1) == len(cases):
                dt = time.time() - t0
                print(f'  {i+1}/{len(cases)} ({dt:.0f}s, {dt/(i+1):.2f}s/case)',
                      flush=True)
    np.savez(args.out, **results)
    print(f'[{args.tag}] wrote {args.out} ({time.time()-t0:.0f}s)', flush=True)


def cmd_diff(args):
    pre = np.load(args.pre)
    post = np.load(args.post)
    common = sorted(set(pre.files) & set(post.files))
    print(f'[diff] pre={len(pre.files)} post={len(post.files)} common={len(common)}',
          flush=True)
    total_samples = 0
    samples_equal_exact = 0
    samples_close = 0
    cases_byte_identical = 0
    case_max_abs_delta = {}
    for stem in common:
        a = pre[stem].astype(np.float32)
        b = post[stem].astype(np.float32)
        m = min(len(a), len(b))
        if m == 0:
            continue
        a = a[:m]; b = b[:m]
        d = np.abs(a - b)
        total_samples += m
        samples_equal_exact += int(np.sum(a == b))
        samples_close += int(np.sum(np.isclose(a, b, atol=1e-6, rtol=1e-5)))
        if np.array_equal(a, b):
            cases_byte_identical += 1
        case_max_abs_delta[stem] = float(d.max())
    frac_exact = samples_equal_exact / max(1, total_samples)
    frac_close = samples_close / max(1, total_samples)
    summary = {
        'cases_total': len(common),
        'cases_byte_identical': cases_byte_identical,
        'total_samples': total_samples,
        'samples_exact_match': samples_equal_exact,
        'samples_close_atol_1e-6_rtol_1e-5': samples_close,
        'fraction_exact': frac_exact,
        'fraction_close': frac_close,
        'pass_99_99': frac_close >= 0.9999,
        'top10_max_delta': dict(sorted(case_max_abs_delta.items(),
                                       key=lambda kv: -kv[1])[:10]),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'[diff] wrote {args.out}', flush=True)
    sys.exit(0 if summary['pass_99_99'] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('snapshot')
    a.add_argument('--tag', required=True)
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
