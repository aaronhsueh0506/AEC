"""P52 A.0R.8 — runtime sanity for trace_p52_regime_handler default-OFF.

Pre-Path-3 baseline = HEAD before A.0 retirement (eac5325^ → 8c20f6d).
Post-Path-3        = current branch HEAD (22bd3d9 or later).
Both runs use trace_p52_regime_handler=False (default).

Deterministic 50-case sample selected via random.Random(42).sample(stems, 50).

Pass: total wall time delta < 1 %; per-case p95 delta < 2 %.

Two modes:
  measure  --tag {pre,post}  run 50 cases sequentially (j=1 to keep timing
                              deterministic), write per-case timing CSV
  compare  --pre p --post p  load two timing CSVs, compute deltas, emit
                              JSON verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SAMPLE_SIZE = 50
SAMPLE_SEED = 42
CNG_SEED = 42

# Acceptance bars
TOTAL_BAR_PCT = 1.0
P95_BAR_PCT = 2.0


def discover():
    cases = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir(): continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                cases.append((stem, subset, str(mic), str(lpb)))
    return cases


def sample_cases():
    cases = discover()
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(cases, SAMPLE_SIZE)


def _run_one(stem, subset, mic_p, lpb_p):
    np.random.seed(CNG_SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    cfg_kwargs = dict(
        sample_rate=sr, filter_length=832, mode=AecMode.PBFDKF,
        enable_cng=True, enable_res=True, enable_shadow=True,
    )
    # Pre-Path-3 aec.py has no `trace_p52_regime_handler` field. Only pass
    # it when post (i.e. the field exists on the dataclass).
    try:
        cfg = AecConfig.from_preset(PRESET_MAP['balanced'],
                                     trace_p52_regime_handler=False,
                                     **cfg_kwargs)
    except TypeError:
        cfg = AecConfig.from_preset(PRESET_MAP['balanced'], **cfg_kwargs)
    aec = AEC(cfg)
    hop = aec.hop_size

    t0 = time.perf_counter()
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    elapsed = time.perf_counter() - t0
    return elapsed, pos


def cmd_measure(args):
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cases = sample_cases()
    print(f'[{args.tag}] {len(cases)} sampled cases (seed={SAMPLE_SEED})', flush=True)
    # Warmup: process one short noise frame to amortise import / first-touch cost
    np.random.seed(0)
    cfg_kwargs = dict(sample_rate=16000, filter_length=832, mode=AecMode.PBFDKF,
                      enable_cng=True, enable_res=True, enable_shadow=True)
    try:
        warm_cfg = AecConfig.from_preset(PRESET_MAP['balanced'],
                                          trace_p52_regime_handler=False,
                                          **cfg_kwargs)
    except TypeError:
        warm_cfg = AecConfig.from_preset(PRESET_MAP['balanced'], **cfg_kwargs)
    warm_aec = AEC(warm_cfg)
    warm = np.zeros(warm_aec.hop_size, dtype=np.float32)
    warm_aec.process(warm, warm)

    rows = []
    t0 = time.time()
    for i, (stem, subset, mic_p, lpb_p) in enumerate(cases):
        elapsed, samples = _run_one(stem, subset, mic_p, lpb_p)
        rows.append({
            'stem': stem, 'subset': subset, 'samples': samples,
            'elapsed_s': elapsed,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            dt = time.time() - t0
            print(f'  {i+1}/{len(cases)} ({dt:.1f}s)', flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'[{args.tag}] wrote {args.out} (total {time.time()-t0:.1f}s)', flush=True)


def cmd_compare(args):
    def load(p):
        with open(p) as f:
            return {r['stem']: r for r in csv.DictReader(f)}
    pre = load(args.pre)
    post = load(args.post)
    common = sorted(set(pre) & set(post))
    deltas_s = []
    deltas_pct = []
    rows = []
    pre_total = 0.0
    post_total = 0.0
    for stem in common:
        a = float(pre[stem]['elapsed_s'])
        b = float(post[stem]['elapsed_s'])
        pre_total += a
        post_total += b
        delta_s = b - a
        delta_pct = 100.0 * delta_s / a if a > 0 else 0.0
        deltas_s.append(delta_s)
        deltas_pct.append(delta_pct)
        rows.append({'stem': stem, 'subset': pre[stem]['subset'],
                     'pre_s': f'{a:.3f}', 'post_s': f'{b:.3f}',
                     'delta_s': f'{delta_s:+.3f}', 'delta_pct': f'{delta_pct:+.2f}'})

    total_delta_pct = 100.0 * (post_total - pre_total) / pre_total if pre_total > 0 else 0.0
    p95_pct = float(np.percentile(np.abs(deltas_pct), 95))
    p50_pct = float(np.median(deltas_pct))

    pass_total = abs(total_delta_pct) < TOTAL_BAR_PCT
    pass_p95 = p95_pct < P95_BAR_PCT
    verdict = 'PASS' if (pass_total and pass_p95) else 'FAIL'

    summary = {
        'verdict': verdict,
        'cases': len(common),
        'pre_total_s': pre_total,
        'post_total_s': post_total,
        'total_delta_s': post_total - pre_total,
        'total_delta_pct': total_delta_pct,
        'total_bar_pct': TOTAL_BAR_PCT,
        'total_pass': pass_total,
        'per_case_delta_pct_p50': p50_pct,
        'per_case_delta_pct_p95': p95_pct,
        'p95_bar_pct': P95_BAR_PCT,
        'p95_pass': pass_p95,
        'rows': rows,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'},
                      indent=2, sort_keys=False))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2, sort_keys=False)
    sys.exit(0 if verdict == 'PASS' else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('measure'); a.add_argument('--tag', required=True); a.add_argument('--out', required=True); a.set_defaults(func=cmd_measure)
    b = sub.add_parser('compare'); b.add_argument('--pre', required=True); b.add_argument('--post', required=True); b.add_argument('--out', default=None); b.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
