"""P52 Task A.0 validation — per-frame ERLE_main snapshot + delta comparison.

Two modes:

  snapshot  Iterate the 800-case AEC Challenge dataset, run aec.py per case
            with the j4 standard (preset=balanced, fl=832, cng=True), and
            capture per-frame erle_inst_db (= the main filter's instantaneous
            ERLE) into a tagged CSV file.

  compare   Take two snapshot files (pre / post A.0) and report:
              - per-frame absolute delta histogram
              - fraction of frames with |Δ| > 0.1 dB
              - per-case mean ERLE_main delta, flagging cases with mean
                regression > 0.5 dB (the v1.1 §2.6 fail criterion)

Pass criterion (v1.1 §2.6 Task A.0):
  - frames with |Δ| > 0.1 dB  ≤ 1.0 % of total frames
  - cases with mean Δ < -0.5 dB  == 0

Usage:
  python tools/research/p52_task_a0_validation.py snapshot \\
      --tag pre_a0 --out /tmp/p52_a0/pre.csv -j 4

  python tools/research/p52_task_a0_validation.py snapshot \\
      --tag post_a0 --out /tmp/p52_a0/post.csv -j 4

  python tools/research/p52_task_a0_validation.py compare \\
      --pre /tmp/p52_a0/pre.csv --post /tmp/p52_a0/post.csv \\
      --report /tmp/p52_a0/verdict.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET_ROOT = _REPO / 'wav' / 'aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')

# Acceptance bars (v1.1 §2.6 Task A.0)
FRAME_DELTA_BAR_DB = 0.1
FRAME_DELTA_MAX_FRAC = 0.01   # ≤ 1 % frames with |Δ|>0.1 dB
CASE_MEAN_REGRESSION_BAR_DB = 0.5  # mean Δ_ERLE_main < -0.5 dB → fail


def discover_cases():
    """Return list of (stem, subset, mic_path, lpb_path) for all 800 cases."""
    cases = []
    for subset in SUBSETS:
        subset_dir = DATASET_ROOT / subset
        if not subset_dir.is_dir():
            continue
        for mic in sorted(subset_dir.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = subset_dir / f'{stem}_lpb.wav'
            if not lpb.is_file():
                continue
            cases.append((stem, subset, str(mic), str(lpb)))
    return cases


def _run_one(args):
    """Worker: run AEC on one case, return per-frame erle_inst_db list."""
    stem, subset, mic_path, lpb_path = args
    mic, sr = sf.read(mic_path, dtype='float32')
    lpb, _ = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if lpb.ndim > 1:
        lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'],
        sample_rate=sr,
        filter_length=832,
        mode=AecMode.PBFDKF,
        enable_cng=True,
        enable_res=True,
        enable_shadow=True,
    )
    aec = AEC(cfg)
    hop = aec.hop_size
    erle = []
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        s = aec.get_stats()
        erle.append(float(s.erle_inst_db))
        pos += hop
    return stem, subset, erle


def cmd_snapshot(args):
    cases = discover_cases()
    print(f'[snapshot] {len(cases)} cases discovered; tag={args.tag} jobs={args.jobs}',
          flush=True)
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.limit:
        cases = cases[:args.limit]
        print(f'[snapshot] LIMITED to {len(cases)} cases', flush=True)

    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_run_one, cases)):
                results.append(r)
                if (i + 1) % 25 == 0 or (i + 1) == len(cases):
                    dt = time.time() - t0
                    print(f'  {i+1}/{len(cases)} ({dt:.0f}s, '
                          f'{dt/(i+1):.2f}s/case)', flush=True)
    else:
        results = []
        for i, c in enumerate(cases):
            results.append(_run_one(c))
            if (i + 1) % 10 == 0 or (i + 1) == len(cases):
                dt = time.time() - t0
                print(f'  {i+1}/{len(cases)} ({dt:.0f}s)', flush=True)

    # Write CSV: rows = (stem, subset, frame_idx, erle_inst_db)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stem', 'subset', 'frame', 'erle_inst_db'])
        for stem, subset, erle in results:
            for fi, e in enumerate(erle):
                w.writerow([stem, subset, fi, f'{e:.4f}'])
    print(f'[snapshot] wrote {args.out} ({time.time()-t0:.0f}s total)', flush=True)


def _load_snapshot(path):
    """Return dict: stem → (subset, np.ndarray of erle_inst_db)."""
    out = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        cur_stem = None
        cur_subset = None
        cur = []
        for row in r:
            stem, subset, _frame, erle = row
            if stem != cur_stem:
                if cur_stem is not None:
                    out[cur_stem] = (cur_subset, np.array(cur, dtype=np.float32))
                cur_stem = stem
                cur_subset = subset
                cur = []
            cur.append(float(erle))
        if cur_stem is not None:
            out[cur_stem] = (cur_subset, np.array(cur, dtype=np.float32))
    return out


def cmd_compare(args):
    pre = _load_snapshot(args.pre)
    post = _load_snapshot(args.post)
    common = sorted(set(pre) & set(post))
    print(f'[compare] pre={len(pre)} post={len(post)} common={len(common)}',
          flush=True)

    total_frames = 0
    frames_over_bar = 0
    case_means = {}   # stem → mean delta
    per_subset_means = {s: [] for s in SUBSETS}

    for stem in common:
        subset, a = pre[stem]
        _, b = post[stem]
        n = min(len(a), len(b))
        if n == 0:
            continue
        d = b[:n] - a[:n]
        total_frames += n
        frames_over_bar += int(np.sum(np.abs(d) > FRAME_DELTA_BAR_DB))
        m = float(np.mean(d))
        case_means[stem] = m
        per_subset_means[subset].append(m)

    frac_over = frames_over_bar / max(1, total_frames)
    regressing = {s: m for s, m in case_means.items()
                  if m < -CASE_MEAN_REGRESSION_BAR_DB}

    pass_frame = frac_over <= FRAME_DELTA_MAX_FRAC
    pass_case = len(regressing) == 0
    verdict = 'PASS' if (pass_frame and pass_case) else 'FAIL'

    summary = {
        'verdict': verdict,
        'total_cases': len(common),
        'total_frames': total_frames,
        'frame_delta_bar_db': FRAME_DELTA_BAR_DB,
        'frames_over_bar': frames_over_bar,
        'fraction_over_bar': frac_over,
        'fraction_bar_max': FRAME_DELTA_MAX_FRAC,
        'frame_test_pass': pass_frame,
        'case_regression_bar_db': CASE_MEAN_REGRESSION_BAR_DB,
        'cases_regressing': len(regressing),
        'case_test_pass': pass_case,
        'regressing_cases': dict(sorted(regressing.items(), key=lambda kv: kv[1])[:20]),
        'per_subset_mean_delta_db': {
            s: float(np.mean(v)) if v else None
            for s, v in per_subset_means.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=False))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(summary, f, indent=2, sort_keys=False)
        print(f'[compare] wrote {args.report}', flush=True)
    sys.exit(0 if verdict == 'PASS' else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_snap = sub.add_parser('snapshot')
    ap_snap.add_argument('--tag', required=True)
    ap_snap.add_argument('--out', required=True, help='output CSV path')
    ap_snap.add_argument('-j', '--jobs', type=int, default=4)
    ap_snap.add_argument('--limit', type=int, default=0,
                         help='if >0, only run first N cases (smoke)')
    ap_snap.set_defaults(func=cmd_snapshot)

    ap_cmp = sub.add_parser('compare')
    ap_cmp.add_argument('--pre', required=True)
    ap_cmp.add_argument('--post', required=True)
    ap_cmp.add_argument('--report', default=None)
    ap_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
