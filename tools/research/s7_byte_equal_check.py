#!/usr/bin/env python3
"""S7 byte-equal scaffold check (Step 5).

Compares S7 flag-OFF outputs (results/v3_12_s7_off/) against the
v3.11.2 / v3_12_s3 baseline (results/v3_12_s3_candidate/). For each
matching `<stem>_ours.wav`, runs md5 + max-diff; reports per-bucket
PASS / FAIL counts.

Design ref: docs/v3_12_phase3b_v3_design.md §5.2 — flag-OFF must
produce >= 99.99% byte-equal output across 800 cases.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    if len(sys.argv) != 3:
        print('usage: s7_byte_equal_check.py <candidate_dir> <baseline_dir>')
        return 2
    cand = Path(sys.argv[1])
    base = Path(sys.argv[2])
    if not cand.is_dir() or not base.is_dir():
        print(f'missing dir: {cand} / {base}')
        return 2

    cand_wavs = sorted(cand.glob('*_ours.wav'))
    n_total = 0
    n_md5_equal = 0
    n_zero_diff = 0
    n_diff = 0
    worst = []  # (max_diff, stem)
    bucket_count = {b: [0, 0, 0] for b in
                    ('FS_static', 'FS_movement', 'DT_static',
                     'DT_movement', 'NE')}

    def _bucket(stem):
        if '_farend_singletalk' in stem:
            return 'FS_movement' if '_with_movement' in stem else 'FS_static'
        if '_doubletalk' in stem:
            return 'DT_movement' if '_with_movement' in stem else 'DT_static'
        if '_nearend_singletalk' in stem:
            return 'NE'
        return 'OTHER'

    for cwav in cand_wavs:
        bwav = base / cwav.name
        if not bwav.is_file():
            continue
        n_total += 1
        with open(cwav, 'rb') as f:
            md_c = hashlib.md5(f.read()).hexdigest()
        with open(bwav, 'rb') as f:
            md_b = hashlib.md5(f.read()).hexdigest()
        stem = cwav.name.replace('_ours.wav', '')
        b = _bucket(stem)
        bucket_count[b][0] += 1
        if md_c == md_b:
            n_md5_equal += 1
            bucket_count[b][1] += 1
            continue
        # Different md5; quantify
        a, _ = sf.read(cwav)
        bb, _ = sf.read(bwav)
        if len(a) != len(bb):
            mxd = -1.0
        else:
            mxd = float(np.max(np.abs(a - bb)))
        if mxd == 0.0:
            n_zero_diff += 1
            bucket_count[b][1] += 1  # zero-diff still counts as match (md5 may differ on metadata)
        else:
            n_diff += 1
            bucket_count[b][2] += 1
            worst.append((mxd, stem))

    worst.sort(reverse=True)

    print(f'=== S7 byte-equal scaffold check ===')
    print(f'Candidate: {cand}')
    print(f'Baseline : {base}')
    print(f'Total matched cases: {n_total}')
    print(f'  md5-equal: {n_md5_equal} ({n_md5_equal/n_total*100:.2f}%)')
    print(f'  zero-diff (md5 differ on metadata but samples identical): {n_zero_diff}')
    print(f'  nonzero-diff: {n_diff} ({n_diff/n_total*100:.2f}%)')
    if worst:
        print(f'Top-10 nonzero-diff cases:')
        for d, s in worst[:10]:
            print(f'  max_diff={d:.4e}  {s}')
    print()
    print(f'{"bucket":<14} {"n":>4} {"match":>6} {"diff":>5} {"match%":>8}')
    for b, (n, m, d) in bucket_count.items():
        pct = m / n * 100 if n else 0.0
        print(f'{b:<14} {n:>4} {m:>6} {d:>5} {pct:>7.2f}%')

    # PASS criterion: 99.99% byte-equal globally (per design §5.2)
    pass_pct = (n_md5_equal + n_zero_diff) / n_total * 100 if n_total else 0
    print()
    print(f'PASS criterion (>= 99.99%): {pass_pct:.4f}% → '
          f'{"PASS" if pass_pct >= 99.99 else "FAIL"}')
    return 0 if pass_pct >= 99.99 else 1


if __name__ == '__main__':
    raise SystemExit(main())
