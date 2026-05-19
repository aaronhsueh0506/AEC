"""v3.21.1 patch gate verification.

bench_aecmos.py's ``verdict: ok`` only checks FS_echo regress (<-0.02) and
NE_deg regress (<-0.01) — it does NOT check DT_echo, DT_deg, or FS_deg.
This script applies the full v3.21.1 HARD bars + SOFT checks defined in
the patch plan, prints a verdict table, and exits 0 only if all HARD
bars pass.

Usage::

    python3 python/check_v3_21_1_gates.py \\
        --new results_v3_21_1/scores.json \\
        --baseline docs/bench/v3_21_3aadd2d_baseline/balanced_aec3_scores.json

HARD bars (exit 1 on any failure):
  * Every bucket Δdeg >= -0.005
  * NE bucket absolute deg_mean >= 4.00  (preserve v3.21.0 NE win)
  * DT_static absolute deg_mean >= 2.30  (above AEC2 floor)
  * DT_movement absolute deg_mean >= 2.30

SOFT checks (warn only):
  * At least one echo bucket Δ > 0 (lift expected from A1 per-bin work)
  * Cohort tail: no case Δecho or Δdeg < -0.05 in worst-20 of any bucket
"""
import argparse
import json
import sys
from typing import Dict, List, Tuple


HARD_DEG_DELTA_FLOOR = -0.005
HARD_NE_DEG_ABS_FLOOR = 4.00
HARD_DT_DEG_ABS_FLOOR = 2.30
COHORT_TAIL_REGRESSION = 0.05

BUCKETS = ['FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE']


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--new', required=True, help='new scores.json (post-change)')
    p.add_argument('--baseline', required=True, help='baseline scores.json (pre-change)')
    args = p.parse_args()

    new = load_scores(args.new)
    base = load_scores(args.baseline)

    print('v3.21.1 GATE CHECK')
    print('=' * 82)
    print(f'  new:      {args.new}')
    print(f'  baseline: {args.baseline}')
    print()
    print(f"{'Bucket':<14} {'Δecho':>10} {'Δdeg':>10} {'new_echo':>10} {'new_deg':>10} {'HARD':>10}")
    print('-' * 82)

    hard_pass = True
    soft_echo_lift = False

    for b in BUCKETS:
        n_summary = new['summary'].get(b)
        b_summary = base['summary'].get(b)
        if n_summary is None or b_summary is None:
            print(f'{b:<14} MISSING in new or baseline — HARD FAIL')
            hard_pass = False
            continue

        de = n_summary['echo_mean'] - b_summary['echo_mean']
        dd = n_summary['deg_mean'] - b_summary['deg_mean']
        new_echo = n_summary['echo_mean']
        new_deg = n_summary['deg_mean']

        bucket_hard_pass = True
        reasons: List[str] = []

        if dd < HARD_DEG_DELTA_FLOOR:
            bucket_hard_pass = False
            reasons.append(f'Δdeg<{HARD_DEG_DELTA_FLOOR}')
        if b == 'NE' and new_deg < HARD_NE_DEG_ABS_FLOOR:
            bucket_hard_pass = False
            reasons.append(f'NE_deg<{HARD_NE_DEG_ABS_FLOOR}')
        if b in ('DT_static', 'DT_movement') and new_deg < HARD_DT_DEG_ABS_FLOOR:
            bucket_hard_pass = False
            reasons.append(f'DT_deg<{HARD_DT_DEG_ABS_FLOOR}')

        if not bucket_hard_pass:
            hard_pass = False

        if de > 0:
            soft_echo_lift = True

        verdict = 'PASS' if bucket_hard_pass else 'FAIL'
        suffix = f"  [{','.join(reasons)}]" if reasons else ''
        print(f"{b:<14} {de:+10.3f} {dd:+10.3f} {new_echo:10.3f} {new_deg:10.3f} {verdict:>10}{suffix}")

    print('-' * 82)

    cohort_regressions: Dict[str, List[Tuple[str, float, float]]] = {b: [] for b in BUCKETS}
    new_scores = new.get('scores', {})
    base_scores = base.get('scores', {})
    for stem, n_case in new_scores.items():
        b_case = base_scores.get(stem)
        if b_case is None:
            continue
        bucket = n_case.get('bucket')
        if bucket not in BUCKETS:
            continue
        de_case = n_case['echo'] - b_case['echo']
        dd_case = n_case['deg'] - b_case['deg']
        if de_case < -COHORT_TAIL_REGRESSION or dd_case < -COHORT_TAIL_REGRESSION:
            cohort_regressions[bucket].append((stem, de_case, dd_case))

    soft_cohort_clean = all(len(v) == 0 for v in cohort_regressions.values())

    print()
    print(f'COHORT TAIL (per-case Δ > {COHORT_TAIL_REGRESSION} regressions):')
    for b in BUCKETS:
        regs = cohort_regressions[b]
        print(f'  {b}: {len(regs)} cases')
        for stem, de_c, dd_c in regs[:5]:
            print(f'    - {stem}: Δecho {de_c:+.3f} Δdeg {dd_c:+.3f}')

    print()
    print(f'HARD bars     : {"PASS" if hard_pass else "FAIL"}')
    print(f'SOFT echo lift: {"YES (≥1 bucket Δecho > 0)" if soft_echo_lift else "no lift detected"}')
    print(f'SOFT cohort   : {"clean" if soft_cohort_clean else "regressions present (see above)"}')
    print()
    print(f'VERDICT: {"PASS" if hard_pass else "FAIL"}')
    return 0 if hard_pass else 1


if __name__ == '__main__':
    sys.exit(main())
