#!/usr/bin/env python3
"""Build the Tier 1 subset case list for v3.15 §1.5b onwards bench infrastructure.

Source pools (deduped):
  1. Arc T S1 cohort tail + controls (8) — docs/v3_15_arc_t_s1_design_and_verdict.md
  2. Arc G S4 listen pack candidates (5) — docs/v3_15_arc_g_closure.md
  3. xrtntuju regression-listen clips (7) — memory/project_xrtntuju_regression_clip.md
  4. Arc F NE breakers (2) — docs/v3_15_arc_f_closure.md
  5. Stratified random (~40, 8/bucket × 5 buckets, seed=42) — excludes pools 1-4

After dedupe + dataset existence verification, writes
tools/research/v3_15_subset_cases.txt with a header documenting source pools,
bucket counts, dedup count, and generation date.

Usage:
    python3 tools/research/v3_15_build_subset.py [--dataset PATH] [--out PATH]

Defaults to dataset at AEC repo (../AEC/wav/aec_challenge_blind), and out at
this worktree's tools/research/v3_15_subset_cases.txt.
"""
import argparse
import datetime
import os
import random
import sys
from collections import OrderedDict, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# Default dataset: try worktree-local first, then sibling AEC repo
_DEFAULT_DATASET_CANDIDATES = [
    os.path.join(_REPO, 'wav', 'aec_challenge_blind'),
    '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind',
]
_DEFAULT_OUT = os.path.join(_HERE, 'v3_15_subset_cases.txt')

# ----------------------------------------------------------------------------
# Source pools (extracted verbatim from referenced docs / memories)
# ----------------------------------------------------------------------------

# Pool 1 — Arc T S1 cohort tail + controls
# (5 TAIL + 3 CTRL listed in docs/v3_15_arc_t_s1_design_and_verdict.md "Cases:" block)
POOL_ARC_T = [
    ('TAIL_canonical',         'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk'),
    ('TAIL_named_outlier',     '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement'),
    ('TAIL_arc_f_breaker',     '3UAwzzOa40aCXQAmEdpwww_farend_singletalk_with_movement'),
    ('TAIL_xqvgr_dt_mvmt',     'XqvGR01tJkan17zltLs38Q_doubletalk_with_movement'),
    ('TAIL_arc_m_v2_breaker',  'Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement'),
    ('CTRL_fs_static',         'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk'),
    ('CTRL_dt_static_v2',      'NN7yhG2XTEqq46X8X0yLfA_doubletalk'),
    ('CTRL_ne_only',           '014AzuqPZku2004NbTTmcA_nearend_singletalk'),
]

# Pool 2 — Arc G S4 listen pack candidates
# (5 CAND rows from docs/v3_15_arc_g_closure.md "S4 listen pack" table; the role
#  labels in the doc — _FS_movement / _DT_movement — are abbreviations of
#  _farend_singletalk_with_movement / _doubletalk_with_movement.)
POOL_ARC_G = [
    ('Arc_G_CAND_dt',          'MeQ3WL4hykKuT2761h0xFg_doubletalk'),
    ('Arc_G_CAND_dt',          'QkRkwwFKVEar0WtcuvJsZg_doubletalk'),
    ('Arc_G_CAND_fs_mvmt',     'OX2l6zV7nkmmSkVA3ETLKg_farend_singletalk_with_movement'),
    ('Arc_G_CAND_fs',          'Y91uE2tRg0SUB2a9XjT30w_farend_singletalk'),
    ('Arc_G_CAND_dt_mvmt',     'WH0jN3PY40es2S0LsxmkkQ_doubletalk_with_movement'),
]

# Pool 3 — xrtntuju regression-listen clips
# (7 stems from memory/project_xrtntuju_regression_clip.md; XqvGR01t doubletalk
#  appears twice in the second table — same stem, different windows; deduped.)
POOL_XRTNTUJU = [
    ('xrtntuju_dt',            'XRTnTUjU5kS0mejzCqyCiw_doubletalk'),
    ('xrtntuju_dt',            'LHsrJBRGnUKiMC2mihEr0g_doubletalk'),
    ('xrtntuju_dt_mvmt',       'SgKY30fjT0G8e3kQL0RHSQ_doubletalk_with_movement'),
    ('xrtntuju_dt_mvmt',       'afHuFvflAkaH7Pr85kheUQ_doubletalk_with_movement'),
    ('xrtntuju_dt_mvmt',       'XqvGR01tJkan17zltLs38Q_doubletalk_with_movement'),
    ('xrtntuju_dt',            'XqvGR01tJkan17zltLs38Q_doubletalk'),  # win_00041 / win_00036
]

# Pool 4 — Arc F NE breakers (2 cases with Δdeg ~-0.143/-0.150 in V1 table)
# Stems abbreviated in doc; full stems resolved from dataset listing.
POOL_ARC_F_NE = [
    ('Arc_F_NE_breaker',       'wJVPo4lexUK40x0nuK0KWg_nearend_singletalk'),
    ('Arc_F_NE_breaker',       'E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk'),
]


# ----------------------------------------------------------------------------
# Bucket inference
# ----------------------------------------------------------------------------

def infer_bucket(stem):
    """Infer Tier-1 bucket from filename suffix."""
    if '_farend_singletalk_with_movement' in stem:
        return 'FS_movement'
    if '_farend_singletalk' in stem:
        return 'FS_static'
    if '_nearend_singletalk' in stem:
        return 'NE'
    if '_doubletalk_with_movement' in stem:
        return 'DT_movement'
    if '_doubletalk' in stem:
        return 'DT_static'
    return 'UNKNOWN'


def stem_to_subdir(stem):
    """Map a stem to its dataset subdirectory."""
    if '_farend_singletalk' in stem:
        return 'farend_singletalk'
    if '_nearend_singletalk' in stem:
        return 'nearend_singletalk'
    if '_doubletalk' in stem:
        return 'doubletalk'
    return None


def stem_exists(dataset_dir, stem):
    """Check both _mic.wav and _lpb.wav exist for a stem."""
    sd = stem_to_subdir(stem)
    if sd is None:
        return False
    base = os.path.join(dataset_dir, sd, stem)
    return (os.path.isfile(base + '_mic.wav')
            and os.path.isfile(base + '_lpb.wav'))


def discover_all_stems(dataset_dir):
    """Walk dataset_dir and return all stems (UUID + suffix) grouped by bucket."""
    by_bucket = defaultdict(list)
    for sd in ('farend_singletalk', 'nearend_singletalk', 'doubletalk'):
        d = os.path.join(dataset_dir, sd)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('_mic.wav'):
                continue
            stem = f[:-len('_mic.wav')]
            # Sanity check the matching _lpb exists
            if not os.path.isfile(os.path.join(d, stem + '_lpb.wav')):
                continue
            bucket = infer_bucket(stem)
            if bucket != 'UNKNOWN':
                by_bucket[bucket].append(stem)
    for k in by_bucket:
        by_bucket[k].sort()
    return by_bucket


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None,
                    help='Dataset root (auto-detect if omitted)')
    ap.add_argument('--out', default=_DEFAULT_OUT,
                    help='Output .txt path (default: %(default)s)')
    ap.add_argument('--per-bucket-random', type=int, default=8,
                    help='Stratified random count per bucket (default 8)')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed (default 42)')
    args = ap.parse_args()

    # Resolve dataset
    dataset = args.dataset
    if dataset is None:
        for c in _DEFAULT_DATASET_CANDIDATES:
            if os.path.isdir(c):
                dataset = c
                break
    if dataset is None or not os.path.isdir(dataset):
        sys.exit(f'ERROR: dataset not found. Tried: {_DEFAULT_DATASET_CANDIDATES}')
    print(f'[INFO] dataset = {dataset}', file=sys.stderr)

    # Combine source pools, preserving (role, stem) tuples for tracing
    all_pool = (POOL_ARC_T + POOL_ARC_G + POOL_XRTNTUJU + POOL_ARC_F_NE)

    # Verify pool stems exist; track drops
    pool_stems = OrderedDict()  # stem -> first role label
    pool_drops = []
    for role, stem in all_pool:
        if not stem_exists(dataset, stem):
            pool_drops.append((role, stem))
            continue
        if stem not in pool_stems:
            pool_stems[stem] = role

    # Stratified random fill — exclude already-selected stems
    rng = random.Random(args.seed)
    by_bucket_all = discover_all_stems(dataset)
    pool_set = set(pool_stems.keys())

    random_pick = OrderedDict()
    random_drops = []
    for bucket in ('FS_static', 'FS_movement', 'NE', 'DT_static', 'DT_movement'):
        candidates = [s for s in by_bucket_all.get(bucket, []) if s not in pool_set]
        if len(candidates) < args.per_bucket_random:
            print(f'[WARN] bucket {bucket}: only {len(candidates)} candidates '
                  f'(< requested {args.per_bucket_random})', file=sys.stderr)
        chosen = rng.sample(candidates, min(args.per_bucket_random, len(candidates)))
        chosen.sort()
        for s in chosen:
            random_pick[s] = f'random_{bucket}'

    # Final ordered list — pools first, then random (stable, sorted within group)
    final = OrderedDict()
    for s, role in pool_stems.items():
        final[s] = role
    for s, role in random_pick.items():
        if s not in final:
            final[s] = role

    # Bucket counts
    bucket_counts = defaultdict(int)
    for s in final:
        bucket_counts[infer_bucket(s)] += 1

    # Pool dedup count: how many pool entries (after existence) collapsed
    pool_total_entries = sum(1 for _, s in all_pool
                              if stem_exists(dataset, s))
    pool_dedup_dropped = pool_total_entries - len(pool_stems)

    # Write output
    today = datetime.date.today().isoformat()
    header = [
        '# Tier 1 subset case list for v3.15 §1.5b onwards',
        f'# Generated: {today}',
        f'# Source pools:',
        f'#   1. Arc T S1 cohort tail + controls (8) — docs/v3_15_arc_t_s1_design_and_verdict.md',
        f'#   2. Arc G S4 listen pack candidates (5) — docs/v3_15_arc_g_closure.md',
        f'#   3. xrtntuju regression-listen clips (7) — memory/project_xrtntuju_regression_clip.md',
        f'#   4. Arc F NE breakers (2) — docs/v3_15_arc_f_closure.md',
        f'#   5. Stratified random ({args.per_bucket_random}/bucket × 5 buckets, seed={args.seed}) — excludes pools 1-4',
        f'#',
        f'# Total stems: {len(final)}',
        f'# Bucket counts:',
        f'#   FS_static    = {bucket_counts["FS_static"]}',
        f'#   FS_movement  = {bucket_counts["FS_movement"]}',
        f'#   NE           = {bucket_counts["NE"]}',
        f'#   DT_static    = {bucket_counts["DT_static"]}',
        f'#   DT_movement  = {bucket_counts["DT_movement"]}',
        f'#',
        f'# Pool entries (after existence check): {pool_total_entries}',
        f'# Pool dedup collapses: {pool_dedup_dropped}',
        f'# Pool stems dropped (not in dataset): {len(pool_drops)}',
    ]
    if pool_drops:
        header.append('#   Dropped: ' + ', '.join(f'{r}={s}' for r, s in pool_drops))
    header.append('#')
    header.append('# Build script: tools/research/v3_15_build_subset.py')
    header.append('')

    lines = list(header)
    for stem in final:
        lines.append(stem)
    body = '\n'.join(lines) + '\n'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write(body)

    # Stderr summary
    print(f'[INFO] wrote {args.out}', file=sys.stderr)
    print(f'[INFO] total stems: {len(final)}', file=sys.stderr)
    for b in ('FS_static', 'FS_movement', 'NE', 'DT_static', 'DT_movement'):
        print(f'[INFO]   {b:12s} = {bucket_counts[b]}', file=sys.stderr)
    print(f'[INFO] pool entries: {pool_total_entries}, '
          f'dedup collapses: {pool_dedup_dropped}, '
          f'dropped: {len(pool_drops)}', file=sys.stderr)


if __name__ == '__main__':
    main()
