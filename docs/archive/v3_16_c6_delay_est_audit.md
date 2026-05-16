# v3.16 C6 — DelayEst audit (Phase 1, 2026-05-15)

**Status**: design + script landed; trace dump pending.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 1.0 (per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` §10).

---

## 1. Purpose

Adjudicate whether DelayEst (`DelayEstimator` GCC-PHAT estimator,
[python/aec.py:1198](../python/aec.py#L1198)) is the upstream root cause
for the meta-pattern across v3.15 closures (cohort tail `qNvSMyU` class,
Arc M V1 FS_movement damage, Arc F cohort tail damage, Arc G destructive
W reset, §1.1 H5 hypothesis on DT-NE) and the HK-2 pcb1N case (delay
~125 ms > fl 52 ms; mic-lpb cross-correlation r=0.071). Output gates
v3.16 Phase 3-4 candidate ROI re-evaluation (v3.16-A / v3.16-B / C7 / C9).

---

## 2. Hypotheses

| H | Statement | Predicted observable |
|---|---|---|
| **H1** | DelayEst is the upstream root cause; movement / cohort tail / pcb1N closures fire because the linear filter trains on a stale or wrong delay → phantom echo → inflated `residual_echo_psd` → RES over-suppresses on NE-only frames OR fails to suppress on FS-only frames | DelayEst issues (PAR < `par_low_threshold=5.0`, top1/top2 ambiguous with `top2_par/top1_par > 0.5`, `did_estimate=False` for long stretches, `estimated_delay` drifts > 50 samples between consecutive estimates, init mis-lock at `init_done=True` boundary) **PRECEDE** filter issues (high `residual_psd_linear`, `epc_active=True`, `cohort_tail_T=True`, `divergence > 0.3`) by ≥ 100 ms |
| **H2** | DelayEst behaves correctly; closures are mechanism walls (Q/R steady-state, W destructive zero-out, RES gate trade-off). DelayEst trace is normal in bad-frame windows; filter issues coincide with mechanism events independent of delay | DelayEst PAR ≥ 8.0 (solid), top2/top1 < 0.3, `estimated_delay` stable within ±10 samples, no init mis-lock; filter issues fire AT/AFTER the corresponding mechanism event (e.g. EPC at actual echo-path-change frame) |

**Mixed verdict**: 30-50% upstream attribution → conditional H1 (delay-aware
candidate worth opening, but not the primary root cause).

---

## 3. Methodology

### 3.1 Trace surface (existing, audit-only)

- `AecConfig.trace_delay_est = True` enables
  `DelayEstimator._trace_rows` capture: per-`accumulate()` call dump of
  `estimated_delay`, `last_par`, `top1/2/3 lag+height+par`, `init_done`,
  `did_estimate`, `samples_accumulated`, `in_init_window`, `confidence`,
  `is_solid`. Already wired
  ([python/aec.py:292-298](../python/aec.py#L292), built-in trace
  emit at [python/aec.py:1334-1393](../python/aec.py#L1334)).
- `AEC.get_stats()` per-frame: `filter_state`, `divergence`,
  `epc_active`, `cohort_tail_T` (Arc T S1 detector signal, default ON
  in BALANCED via §10.S0b), `dt_active`, `delay_samples` (in-use
  delay),`shadow_advantage`, `main_paused`, `res_using_render`,
  `mic/far/error/echo/error PSD dB`.
- `AEC._diag` per-frame: `residual_psd_linear` /
  `residual_psd_render` / `residual_render_blend` / `erl_estimate`
  / `epv_gain_ratio` / `erle_slope_db_per_s` / `filter_state` (string
  internal version).

Zero behaviour change vs production default config — `trace_delay_est`
is opt-in.

### 3.2 Case set (Tier A deep audit, 10 cases)

Per `tools/research/v3_16_c6_tier_a_cases.txt`:

| # | Stem | Selection rationale |
|---|---|---|
| 1 | `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` | cohort tail catastrophe (P52 A.0 outlier; v3.13 / v3.14 / v3.15 all hit) |
| 2 | `XqvGR01tJkan17zltLs38Q_doubletalk_with_movement` | Arc M V1 FS_movement damage candidate |
| 3 | `pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk` | HK-2 reframed (delay ~125 ms > fl 52 ms; r=0.071) |
| 4 | `XRTnTUjU5kS0mejzCqyCiw_doubletalk` | xrtntuju DT regression carrier (v3.13 E2 Path 3 unmask) |
| 5 | `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement` | FS_movement |
| 6 | `Hp5g1asacUCt5rJVLO1FuQ_doubletalk_with_movement` | DT_movement |
| 7 | `WH0jN3PY40es2S0LsxmkkQ_doubletalk_with_movement` | DT_movement |
| 8 | `jtYTdZm3lUmFVNibJWq8YQ_doubletalk` | F2.4 mu holdoff regression case |
| 9 | `OX2l6zV7nkmmSkVA3ETLKg_farend_singletalk_with_movement` | FS_movement |
| 10 | `Y91uE2tRg0SUB2a9XjT30w_farend_singletalk` | CTRL — FS_static well-behaved baseline |

CTRL (case 10) provides "DelayEst-good + filter-good" reference; cases
1-9 cover all v3.15 closure failure modes.

### 3.3 Audit script

[`tools/research/v3_16_c6_delay_est_audit.py`](../tools/research/v3_16_c6_delay_est_audit.py)
runs each stem with `trace_delay_est=True`, captures per-frame stats +
DelayEst trace, writes per-case JSON to `/tmp/v3_16_c6_audit/`.

Output:
- `/tmp/v3_16_c6_audit/<stem>.json` — full trace per case
- `/tmp/v3_16_c6_audit/summary.csv` — per-case aggregate (epc fire-rate,
  cohort_tail_T fire-rate, divergence p95, DelayEst PAR mean/p10,
  top1/top2 ratio mean, estimated_delay percentiles, ERLE windowed mean)

Run command:
```bash
python3 tools/research/v3_16_c6_delay_est_audit.py \
    --cases tools/research/v3_16_c6_tier_a_cases.txt \
    --dataset wav/aec_challenge_blind \
    --out /tmp/v3_16_c6_audit/
```

Wall-time estimate: ~10 cases × ~10 sec/case = ~2 min. No bench
parallelisation (each case gets full trace; sequential is fine for
audit phase).

### 3.4 Adjudication procedure (post-trace)

For each case, time-align the following per-frame series:
1. **DelayEst issue indicators**: `last_par < 5`, `top2_par/top1_par > 0.5`,
   `estimated_delay` jump > 50 samples, `did_estimate=False` streak > 30
   frames after `init_done`.
2. **Filter issue indicators**: `residual_psd_linear` percentile rank
   > 90th in case, `epc_active=True`, `cohort_tail_T=True`,
   `divergence > 0.3`, `shadow_advantage > 1.5`.
3. **Lead-lag**: for each filter-issue burst, find the nearest preceding
   DelayEst-issue burst and record the lead time (ms). Aggregate the
   lead-time distribution.

H1 supported if ≥ 60% of filter-issue bursts have a preceding DelayEst
issue within 250 ms. H2 supported if < 30% have any preceding DelayEst
issue. Mixed: 30-60% → conditional H1.

### 3.5 Hard bar (verdict acceptance)

Per [v3.15 §0.4](../../../.claude/plans/se-aec-aec-main-hazy-lynx.md#§04--negative-result-acceptance-protocol):
pre-stated thresholds.

| Outcome | Threshold | Action |
|---|---|---|
| **H1 STRONG** | ≥ 60% lead-lag attribution; PAR < 5 in ≥ 30% of bad-frame windows | OPEN delay-aware mechanism arc (3-6 sprints LOE per plan §C6); REORDER Phase 3-4: v3.16-A may need delay sanity gate; v3.16-B / C7 / C9 ROI re-evaluated |
| **H1 CONDITIONAL** | 30-60% lead-lag attribution | OPEN narrow scope: delay-confidence gate downstream of DelayEst (use `is_solid` already in trace) without changing estimator core; defer mechanism arc |
| **H2** | < 30% lead-lag; PAR ≥ 8 in ≥ 70% of bad-frame windows | CLOSE C6 audit per §0.4; do NOT open mechanism arc; Phase 3-4 ROI estimates stand; v3.15 closures attributed to mechanism walls (already documented in respective closure docs) |

---

## 4. Substrate notes

- DelayEst `max_delay_ms = 1024` (since v3.13 E2 Path 3 ship); pcb1N
  delay ~125 ms is well within range. If pcb1N's `estimated_delay`
  matches the ~2000-sample static delay, that's evidence the estimator
  resolves correctly even though the filter (`fl=832 = 52 ms`) cannot
  cover it — i.e. delay > fl is not a DelayEst failure but a separate
  filter-tail-coverage issue that lives in C9 scope, not C6 scope.
- Arc T cohort_tail_T detector is default ON in BALANCED preset
  (v3.15 §10.S0b). `cohort_tail_T=True` frames are the candidate "bad
  windows" for cohort tail catastrophe class.

---

## 5. Output deliverables

1. **Trace JSONs** (per-case, gitignored) at `/tmp/v3_16_c6_audit/`.
2. **Summary CSV** at `/tmp/v3_16_c6_audit/summary.csv`.
3. **Verdict doc** at [`docs/v3_16_c6_delay_est_audit_verdict.md`](v3_16_c6_delay_est_audit_verdict.md)
   with adjudicated H1 / H2 / mixed and recommended Phase 1.5 / 3 / 4
   re-ordering implications.
