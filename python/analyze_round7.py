#!/usr/bin/env python3
"""Round 7 Phase 0 analysis: filter-trajectory + transition trace separation.

Compares per-frame R7 fields between worst-20 and best-20 DT_movement cases
(rank-locked by `baseline_v381_seeded.deg`).

Go/No-Go gate (upgraded from naive 30% relative gap):
- Continuous fields: Cohen's-d-like effect size = |mean(worst) - mean(best)| /
  std(800-case population). PASS if effect_size >= 0.5 AND absolute delta
  meaningful (>= 0.05 for normalized fields, >= 5 percentage points for
  fractions, >= 1.0 dB for power-db fields).
- Event-count fields (discrete fire counts): PASS if worst-20 hit-rate
  (worst20 cases with count > 0) >= 6/20 AND worst20 hit-rate >= 2x best20.
- Post-event response fields: continuous rule + at least 8/20 worst-20 cases
  have non-zero sample count (avoids meaningless 0/0 = 0 separators).

GO if at least one field passes; STOP otherwise.

Reads:
    experiments/round7_phase0/states.json
    experiments/baseline_v381_seeded/scores.json

Writes:
    experiments/round7_phase0/analysis.md
"""
import json
import os
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


EVENT_COUNT_FIELDS = {
    'r7_event_delay_first_count', 'r7_event_delay_shift_count',
    'r7_event_epv_count', 'r7_event_shadow_rise_count',
    'r7_delay_delta_count',
}
POST_EVENT_FIELDS = {
    'r7_post_delay_first_inst_erle_mean',
    'r7_post_delay_shift_inst_erle_mean',
    'r7_post_epv_inst_erle_mean',
    'r7_post_shadow_rise_inst_erle_mean',
}
DB_FIELDS = {
    'r7_nores_pwr_db_mean', 'r7_final_pwr_db_mean',
}
PCT_FIELDS = {
    'r7_p_max_active_pct', 'r7_p_floor_active_pct',
    'r7_once_conv_pct', 'r7_epc_force_active_pct',
    'r7_far_active_pct',
}

R7_FIELD_GROUPS = {
    'transition events (event-count rule)': [
        'r7_event_delay_first_count', 'r7_event_delay_shift_count',
        'r7_event_epv_count', 'r7_event_shadow_rise_count',
    ],
    'delay dynamics': [
        'r7_delay_samples_mean', 'r7_delay_samples_max',
        'r7_delay_delta_max_abs', 'r7_delay_delta_count',
    ],
    'P-override / forced state (pct rule)': [
        'r7_p_max_active_pct', 'r7_p_floor_active_pct',
        'r7_epc_force_active_pct',
    ],
    'convergence (pct rule)': [
        'r7_once_conv_pct',
    ],
    'output power (dB / continuous)': [
        'r7_far_active_pct',
        'r7_nores_pwr_db_mean', 'r7_final_pwr_db_mean',
        'r7_nores_echo_proxy_mean', 'r7_nores_echo_proxy_max',
        'r7_res_required_gain_mean',
    ],
    'adaptation signals (continuous)': [
        'r7_shadow_advantage_mean', 'r7_inst_erle_smooth_mean',
        'r7_main_err_smooth_mean', 'r7_shadow_err_smooth_mean',
        'r7_filter_w_norm_mean', 'r7_shadow_w_norm_mean',
        'r7_mu_scale_mean',
    ],
    'post-event recovery (continuous + sample-count guard)': sorted(POST_EVENT_FIELDS),
}


def evaluate_field(field, worst_vals, best_vals, all_vals, worst_states=None):
    """Return (pass_bool, reason, w_mean, b_mean)."""
    w_mean = float(np.mean(worst_vals)) if worst_vals else 0.0
    b_mean = float(np.mean(best_vals)) if best_vals else 0.0
    abs_delta = abs(w_mean - b_mean)

    if field in EVENT_COUNT_FIELDS:
        # Event-count rule: worst hit-rate >= 6/20 AND >= 2x best hit-rate
        w_hit = sum(1 for v in worst_vals if v > 0)
        b_hit = sum(1 for v in best_vals if v > 0)
        n_w = max(len(worst_vals), 1)
        n_b = max(len(best_vals), 1)
        w_rate = w_hit / n_w
        b_rate = b_hit / n_b
        passed = (w_hit >= 6) and (w_rate >= 2.0 * b_rate)
        reason = (f'event-count: w_hit={w_hit}/{n_w}, b_hit={b_hit}/{n_b}, '
                  f'ratio={w_rate / max(b_rate, 0.01):.1f}x')
        return passed, reason, w_mean, b_mean

    pop_std = float(np.std(all_vals)) if all_vals else 0.0
    effect_size = abs_delta / max(pop_std, 1e-6)

    if field in PCT_FIELDS:
        abs_floor = 0.05
    elif field in DB_FIELDS:
        abs_floor = 1.0
    else:
        abs_floor = 0.05

    if field in POST_EVENT_FIELDS:
        count_field_name = field.replace('_mean', '_count')
        if worst_states is not None:
            non_zero = sum(1 for s in worst_states if s.get(count_field_name, 0) > 0)
            if non_zero < 8:
                return (False, f'post-event: only {non_zero}/20 worst cases have samples',
                        w_mean, b_mean)

    passed = (effect_size >= 0.5) and (abs_delta >= abs_floor)
    reason = f'continuous: effect={effect_size:.2f}, |Δ|={abs_delta:.4f} (floor={abs_floor})'
    return passed, reason, w_mean, b_mean


def main():
    states = json.load(open(f'{_REPO}/experiments/round7_phase0/states.json'))
    base = json.load(open(f'{_REPO}/experiments/baseline_v381_seeded/scores.json'))['scores']

    dtmv = sorted(
        [(s, base[s]['deg']) for s in base if base[s].get('bucket') == 'DT_movement'],
        key=lambda x: x[1]
    )
    worst20_stems = [s for s, _ in dtmv[:20] if s in states]
    best20_stems = [s for s, _ in dtmv[-20:] if s in states]
    worst20 = [states[s] for s in worst20_stems]
    best20 = [states[s] for s in best20_stems]

    md = ['# Round 7 — Phase 0 filter-trajectory analysis', '']
    md.append(f'Cases: {len(states)} (baseline = baseline_v381_seeded, rank-locked by deg)')
    md.append(f'DT_movement worst-20 mean baseline_deg = '
              f'{np.mean([base[s]["deg"] for s in worst20_stems]):.3f}')
    md.append(f'DT_movement best-20 mean baseline_deg = '
              f'{np.mean([base[s]["deg"] for s in best20_stems]):.3f}')
    md.append('')
    md.append('Go/No-Go gate (upgraded):')
    md.append('- continuous: effect_size ≥ 0.5 AND |Δ| ≥ category floor (0.05 / 1 dB / 5 pp)')
    md.append('- event-count: worst-20 hit-rate ≥ 6/20 AND ≥ 2× best-20 hit-rate')
    md.append('- post-event: continuous rule + ≥ 8/20 worst-20 with non-zero sample')
    md.append('')

    md.append('## Per-bucket means (sanity check)')
    md.append('')
    md.append('| Bucket | n | once_conv% | shadow_adv | inst_erle_sm | nores_echo_proxy | res_gain | far_active% |')
    md.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    by_b = defaultdict(list)
    for stem, st in states.items():
        bk = st.get('bucket')
        if bk:
            by_b[bk].append(st)
    for bk in ['FS_static', 'FS_movement', 'NE', 'DT_static', 'DT_movement']:
        rs = by_b.get(bk, [])
        if not rs:
            continue
        n = len(rs)
        m = lambda f: float(np.mean([r.get(f, 0.0) for r in rs]))
        md.append(
            f"| {bk} | {n} | {m('r7_once_conv_pct') * 100:.1f} | "
            f"{m('r7_shadow_advantage_mean'):.3f} | {m('r7_inst_erle_smooth_mean'):.3f} | "
            f"{m('r7_nores_echo_proxy_mean'):.3f} | {m('r7_res_required_gain_mean'):.3f} | "
            f"{m('r7_far_active_pct') * 100:.1f} |"
        )
    md.append('')

    passed_separators = []
    md.append('## DT_movement worst-20 vs best-20 (rank-locked by baseline_deg)')
    md.append('')
    for group_name, fields in R7_FIELD_GROUPS.items():
        md.append(f'### {group_name}')
        md.append('')
        md.append('| field | worst-20 | best-20 | gate | reason |')
        md.append('|---|---:|---:|---|---|')
        for f in fields:
            w_vals = [s.get(f, 0.0) for s in worst20]
            b_vals = [s.get(f, 0.0) for s in best20]
            all_vals = [s.get(f, 0.0) for s in states.values()]
            passed, reason, w, b = evaluate_field(
                f, w_vals, b_vals, all_vals, worst_states=worst20,
            )
            gate = '**PASS**' if passed else 'no'
            md.append(f'| {f} | {w:.3f} | {b:.3f} | {gate} | {reason} |')
            if passed:
                passed_separators.append((f, w, b, reason))
        md.append('')

    md.append('## Phase 0 Go/No-Go decision')
    md.append('')
    md.append(f'PASS-gate fields: **{len(passed_separators)}**')
    if passed_separators:
        md.append('')
        md.append('| field | worst-20 | best-20 | reason |')
        md.append('|---|---:|---:|---|')
        for f, w, b, r in passed_separators:
            md.append(f'| {f} | {w:.3f} | {b:.3f} | {r} |')
        md.append('')
        md.append('**Decision: GO** — proceed to Phase 0.5 signal split + Phase 1.')
        md.append('Top separators above point to which transition / signal type is binding.')
    else:
        md.append('')
        md.append('**Decision: STOP** — no R7 field separates worst-20 from best-20 at the')
        md.append('upgraded gate. Filter trajectory is also at Pareto on this dataset.')
        md.append('Recommend close R7, hold v3.8.1 baseline, escalate to NN postfilter or')
        md.append('out-of-scope filter trajectory rewrite.')
    md.append('')

    if passed_separators:
        md.append('## Worst-20 DT_mv per-case detail (top separators)')
        md.append('')
        top = [f for f, *_ in passed_separators[:5]]
        if top:
            header = '| stem | base_deg | ' + ' | '.join(t.replace('r7_', '') for t in top) + ' |'
            sep = '|---|---:|' + '---:|' * len(top)
            md.append(header)
            md.append(sep)
            for s, _ in dtmv[:20]:
                if s not in states:
                    continue
                row = f'| {s[:30]} | {base[s]["deg"]:.2f} | '
                row += ' | '.join(f'{states[s].get(t, 0.0):.3f}' for t in top)
                row += ' |'
                md.append(row)
            md.append('')

    out = '\n'.join(md)
    with open(f'{_REPO}/experiments/round7_phase0/analysis.md', 'w') as f:
        f.write(out)
    print(out)


if __name__ == '__main__':
    main()
