#!/usr/bin/env python3
"""A/B comparator for two AECMOS scores.json files (v3.22 bench helper).

    python3 python/ab_compare.py <baseline_scores.json> <cand_scores.json> [--worst N]

Prints per-bucket mean echo/deg for baseline + candidate + Δ, catastrophic
regression counts (Δ <= -0.5), and the worst-N per-case regressions per metric.
Bucket order is scenario-grouped (FS / DT / NE) per the bench discipline.
"""
import argparse
import json
from collections import defaultdict

BUCKETS = ['FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE']
CATASTROPHIC = -0.5


def load(path):
    d = json.load(open(path))
    return d.get('label', path), d['scores']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline')
    ap.add_argument('candidate')
    ap.add_argument('--worst', type=int, default=8)
    args = ap.parse_args()

    bl_label, bl = load(args.baseline)
    cd_label, cd = load(args.candidate)

    stems = sorted(set(bl) & set(cd))
    missing = (set(bl) ^ set(cd))
    print(f"baseline={bl_label} (n={len(bl)})  candidate={cd_label} (n={len(cd)})")
    print(f"matched cases: {len(stems)}" + (f"  [unmatched: {len(missing)}]" if missing else ""))
    print()

    # per-bucket aggregation
    agg = defaultdict(lambda: {'b_echo': 0.0, 'b_deg': 0.0,
                               'c_echo': 0.0, 'c_deg': 0.0, 'n': 0})
    per_case = []  # (stem, bucket, decho, ddeg)
    for s in stems:
        bk = bl[s]['bucket']
        de = cd[s]['echo'] - bl[s]['echo']
        dd = cd[s]['deg'] - bl[s]['deg']
        a = agg[bk]
        a['b_echo'] += bl[s]['echo']; a['b_deg'] += bl[s]['deg']
        a['c_echo'] += cd[s]['echo']; a['c_deg'] += cd[s]['deg']
        a['n'] += 1
        per_case.append((s, bk, de, dd))

    print(f"{'bucket':<13} {'n':>4} | {'echo_bl':>8} {'echo_cd':>8} {'Δecho':>8} | "
          f"{'deg_bl':>7} {'deg_cd':>7} {'Δdeg':>8}")
    print('-' * 86)
    sum_de = sum_dd = 0.0
    for bk in BUCKETS:
        if bk not in agg:
            continue
        a = agg[bk]; n = a['n']
        be, bd = a['b_echo'] / n, a['b_deg'] / n
        ce, cd_ = a['c_echo'] / n, a['c_deg'] / n
        print(f"{bk:<13} {n:>4} | {be:>8.3f} {ce:>8.3f} {ce-be:>+8.3f} | "
              f"{bd:>7.3f} {cd_:>7.3f} {cd_-bd:>+8.3f}")
    # corpus totals (sum of per-case deltas, by metric)
    for s, bk, de, dd in per_case:
        sum_de += de; sum_dd += dd
    print('-' * 86)
    print(f"Σ per-case  Δecho={sum_de:+.3f}  Δdeg={sum_dd:+.3f}   "
          f"(mean Δecho={sum_de/len(stems):+.4f}, Δdeg={sum_dd/len(stems):+.4f})")

    # catastrophic regressions
    cat_e = [(s, bk, de) for s, bk, de, dd in per_case if de <= CATASTROPHIC]
    cat_d = [(s, bk, dd) for s, bk, de, dd in per_case if dd <= CATASTROPHIC]
    print(f"\ncatastrophic (Δ<=-0.5):  echo {len(cat_e)} cases | deg {len(cat_d)} cases")

    print(f"\nworst-{args.worst} Δecho regressions:")
    for s, bk, de, dd in sorted(per_case, key=lambda t: t[2])[:args.worst]:
        print(f"  {de:+.3f}  (Δdeg {dd:+.3f})  [{bk}]  {s}")
    print(f"worst-{args.worst} Δdeg regressions:")
    for s, bk, de, dd in sorted(per_case, key=lambda t: t[3])[:args.worst]:
        print(f"  {dd:+.3f}  (Δecho {de:+.3f})  [{bk}]  {s}")


if __name__ == '__main__':
    main()
