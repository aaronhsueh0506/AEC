"""800-case AEC3 RefinedFilterUpdateGain X² source parity fix verdict.

This candidate aligns PBFDKF RefinedFilterUpdateGain with AEC3:
  X² source = partition-summed render power (Σ_p |X_buf[p][k]|²),
used in mu denominator / noise gate / H_error decay. W update
direction stays per-partition (UNCHANGED — not a summed-direction
rewrite).

Reads scores.json from A_off (pre-parity: X² = latest partition only)
and B_on (parity ON: X² = partition-summed, matching AEC3). Reports:

  - Bucket means INCLUDING all cases
  - Bucket means EXCLUDING XRTnTUjU_DT_static (no-clean-convergence
    stress per project_xrtntuju_dt_static_stress memory)
  - Worst-N stress list per bucket (echo + deg)
  - Per-case Δ table (sorted by Δdeg descending for normal, separate
    list for XRTnTUjU stems)
  - Pareto verdict (Δecho vs Δdeg per bucket; geomean of bucket ratios
    excluding the stress case)

Usage:
  python3 python/scripts/partition_summed_x2_800_verdict.py \\
      --a /path/scores_A_off/scores.json \\
      --b /path/scores_B_on/scores.json \\
      --out report.md
"""
from __future__ import annotations

import argparse
import json
import sys

XRTNTUJU_DT_STATIC = 'XRTnTUjU5kS0mejzCqyCiw_doubletalk'


def load(path: str) -> dict:
    raw = json.load(open(path))
    scores = raw.get('scores', raw)
    # Filter to per-case entries
    return {k: v for k, v in scores.items()
            if isinstance(v, dict) and 'echo' in v and 'deg' in v}


def bucket_mean(per_case: dict, predicate=None) -> dict:
    buckets: dict[str, list] = {}
    for stem, v in per_case.items():
        if predicate is not None and not predicate(stem, v):
            continue
        buckets.setdefault(v['bucket'], []).append(v)
    return {b: {
        'n': len(vs),
        'echo_mean': sum(x['echo'] for x in vs) / len(vs),
        'deg_mean': sum(x['deg'] for x in vs) / len(vs),
    } for b, vs in buckets.items()}


def per_case_delta(A: dict, B: dict) -> list[dict]:
    out = []
    for s in sorted(set(A) | set(B)):
        a = A.get(s); b = B.get(s)
        if a is None or b is None:
            continue
        out.append({
            'stem': s,
            'bucket': a['bucket'],
            'A_echo': a['echo'], 'B_echo': b['echo'],
            'A_deg': a['deg'], 'B_deg': b['deg'],
            'd_echo': b['echo'] - a['echo'],
            'd_deg': b['deg'] - a['deg'],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='scores_A_off/scores.json')
    ap.add_argument('--b', required=True, help='scores_B_on/scores.json')
    ap.add_argument('--out', required=True, help='output .md path')
    ap.add_argument('--top-n', type=int, default=20)
    args = ap.parse_args()

    A = load(args.a); B = load(args.b)
    n_common = len(set(A) & set(B))
    print(f'cases A={len(A)} B={len(B)} common={n_common}', file=sys.stderr)

    deltas = per_case_delta(A, B)
    # XRTnTUjU_DT_static = stress; exclude from normal aggregate
    is_xrt = lambda s: s == XRTNTUJU_DT_STATIC

    # Bucket means including and excluding stress
    b_all_A = bucket_mean(A)
    b_all_B = bucket_mean(B)
    b_norm_A = bucket_mean(A, lambda s, v: not is_xrt(s))
    b_norm_B = bucket_mean(B, lambda s, v: not is_xrt(s))

    lines: list[str] = []
    lines.append('# AEC3 RefinedFilterUpdateGain X² source parity fix — 800-case verdict')
    lines.append('')
    lines.append('Config:')
    lines.append('- Mechanism: align PBFDKF RefinedFilterUpdateGain X² source with AEC3.')
    lines.append('  X²[k] = Σ_p |X_buf[p][k]|² (partition-summed render power, matching')
    lines.append('  AEC3 `render_buffer.cc::SpectralSum`) instead of latest partition only.')
    lines.append('  Used in: mu denominator / noise gate (silent-far floor) / H_error decay.')
    lines.append('  W update outer product still uses per-partition X[p] — direction UNCHANGED.')
    lines.append('- A_off: pre-parity (X² = latest partition only)')
    lines.append('- B_on : parity ON (X² = partition-summed, AEC3 parity)')
    lines.append('- Both runs: mic HPF ON / ref HPF OFF (intended HPF baseline),')
    lines.append('  preset balanced, filter 832 (52ms), --cng, --parallel j4')
    lines.append('')
    lines.append('Acceptance (3-way INDEPENDENT split):')
    lines.append('1. nores LF artifact reduced on cohort tail — see 6-case audit (NOT in this report).')
    lines.append('2. 800-case AECMOS Pareto-safe vs A_off — THIS report.')
    lines.append('3. XRTnTUjU_DT_static stress: SEPARATE state-guard arc.')
    lines.append('   Any Δdeg here reflects PRE-EXISTING gate-3 latch bug exposed by parity fix,')
    lines.append('   NOT a formula problem (see project_usable_linear_gate3_latch_bug memory).')
    lines.append('   Handled by separate state-guard work, NOT by reverting parity fix.')
    lines.append('')
    lines.append(f'Cases: A={len(A)}, B={len(B)}, common={n_common}.')
    lines.append('')
    lines.append('XRTnTUjU_DT_static = stress / no-clean-convergence case '
                 '(per project_xrtntuju_dt_static_stress memory). '
                 'EXCLUDED from normal aggregate; kept as worst-N stress. '
                 'Parity fix EXPOSES the gate-3 binary latch on convergence_seen; '
                 'it does NOT cause it. Reverting parity to "hide" this would also '
                 'revert the nores improvement and the AEC3 alignment.')
    lines.append('')

    # --- Bucket aggregate -----------------------------------------------
    lines.append('## Bucket means (Δ vs A_off, NORMAL aggregate — XRTnTUjU_DT_static excluded)')
    lines.append('')
    lines.append('| bucket | n_A | n_B | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for b in sorted(set(b_norm_A) & set(b_norm_B)):
        a, c = b_norm_A[b], b_norm_B[b]
        de = c['echo_mean'] - a['echo_mean']
        dd = c['deg_mean'] - a['deg_mean']
        lines.append(f'| {b} | {a["n"]} | {c["n"]} | {a["echo_mean"]:.3f} | {c["echo_mean"]:.3f} | '
                     f'{de:+.3f} | {a["deg_mean"]:.3f} | {c["deg_mean"]:.3f} | {dd:+.3f} |')
    lines.append('')
    lines.append('## Bucket means (Δ vs A_off, INCLUDING stress, for completeness)')
    lines.append('')
    lines.append('| bucket | n_A | n_B | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for b in sorted(set(b_all_A) & set(b_all_B)):
        a, c = b_all_A[b], b_all_B[b]
        de = c['echo_mean'] - a['echo_mean']
        dd = c['deg_mean'] - a['deg_mean']
        lines.append(f'| {b} | {a["n"]} | {c["n"]} | {a["echo_mean"]:.3f} | {c["echo_mean"]:.3f} | '
                     f'{de:+.3f} | {a["deg_mean"]:.3f} | {c["deg_mean"]:.3f} | {dd:+.3f} |')
    lines.append('')

    # --- XRTnTUjU stress watch ------------------------------------------
    lines.append('## XRTnTUjU_DT_static stress watch')
    lines.append('')
    xrt = next((d for d in deltas if d['stem'] == XRTNTUJU_DT_STATIC), None)
    if xrt is None:
        lines.append('(case not in cohort)')
    else:
        lines.append(f'A_off: echo={xrt["A_echo"]:.3f}, deg={xrt["A_deg"]:.3f}')
        lines.append(f'B_on : echo={xrt["B_echo"]:.3f}, deg={xrt["B_deg"]:.3f}')
        lines.append(f'**Δecho = {xrt["d_echo"]:+.3f}, Δdeg = {xrt["d_deg"]:+.3f}** '
                     '(stress tolerance bar = −2.0 deg; '
                     'state-guard arc target, NOT a parity-fix revert)')
    lines.append('')

    # --- Worst-N per bucket (NORMAL only) -------------------------------
    lines.append(f'## Top {args.top_n} worst Δdeg per bucket (NORMAL, XRTnTUjU excluded)')
    lines.append('')
    by_bucket: dict[str, list[dict]] = {}
    for d in deltas:
        if d['stem'] == XRTNTUJU_DT_STATIC:
            continue
        by_bucket.setdefault(d['bucket'], []).append(d)
    for b in sorted(by_bucket):
        lst = sorted(by_bucket[b], key=lambda x: x['d_deg'])[:args.top_n]
        lines.append(f'### {b}')
        lines.append('| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for d in lst:
            lines.append(f'| `{d["stem"]}` | {d["A_echo"]:.3f} | {d["B_echo"]:.3f} | {d["d_echo"]:+.3f} | '
                         f'{d["A_deg"]:.3f} | {d["B_deg"]:.3f} | {d["d_deg"]:+.3f} |')
        lines.append('')

    lines.append(f'## Top {args.top_n} worst Δecho per bucket (NORMAL, XRTnTUjU excluded)')
    lines.append('')
    for b in sorted(by_bucket):
        lst = sorted(by_bucket[b], key=lambda x: x['d_echo'])[:args.top_n]
        lines.append(f'### {b}')
        lines.append('| stem | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|')
        for d in lst:
            lines.append(f'| `{d["stem"]}` | {d["A_echo"]:.3f} | {d["B_echo"]:.3f} | {d["d_echo"]:+.3f} | '
                         f'{d["A_deg"]:.3f} | {d["B_deg"]:.3f} | {d["d_deg"]:+.3f} |')
        lines.append('')

    with open(args.out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
