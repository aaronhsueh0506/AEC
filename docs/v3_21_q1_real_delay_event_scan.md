# v3.21 Q1 Path 1 — real-delay-event cohort scan (2026-05-25)

**Status**: read-only artifact survey. No algorithm change, no rendering,
no benchmark, no version bump. Continues the Path 1 step from
[docs/v3_21_q1_trace_audit.md](v3_21_q1_trace_audit.md) §5.4.

**Question**: Do any existing trace / diag artifacts on disk already
identify real `delay_first` / `delay_shift` events (as opposed to
EPV / shadow_rise pseudo-`delay_change` dispatched under production
Phase D wiring)?

**Scope guard**:
- Does NOT modify the Q1 verdict (still **C — insufficient evidence**,
  per [trace audit §0](v3_21_q1_trace_audit.md)).
- Does NOT reclassify EPV / shadow_rise as real delay events.
- Does NOT run any new render. No 800-case AECMOS.
- Does NOT touch Phase D, Phase G, M2/M3 W=0, usable_linear,
  PathChangeRegimeHandler, RES tuning. No v3.22 framing.

---

## 0. TL;DR

**Result: ZERO HITS.** No existing artifact on disk captures the
per-frame `event_delay_first` / `event_delay_shift` / `delay_samples`
diag fields that the orchestrator emits. Two compounding reasons:

1. **The trace scripts that emit per-frame JSON (`v3_21_20_phase_e_trace.py`,
   `v3_21_20_phase_g_isolation.py`) explicitly set `enable_delay_est=False`**
   and pre-align the ref/mic via one-shot xcorr at render start
   (`phase_g_isolation.py:76, 90-95`). In those traces the delay
   estimator never runs → `delay_first` and `delay_shift` cannot fire.

2. **The trace scripts that capture per-frame data don't persist
   `event_delay_*` / `div_counts` / `delay_samples`** even though the
   orchestrator computes them ([orchestrator.py:3448-3495](AEC/python/modules/orchestrator.py)).
   Per-frame fields persisted are limited to 14–31 columns covering
   EPV / shadow_rise / H_error_mean / W_norm / mu — not the
   delay-event family. The 800-case bench (`out_800_*`) emits WAV only,
   no diag.

**Recommendation**: per Path 1 fallback in the trace audit
([§5.4 path 2](v3_21_q1_trace_audit.md)), proceed to **synthetic
delay-shift cohort construction** as the next step. The user must
authorise the construction explicitly; this scan does NOT trigger any
rendering.

A second-line alternative (cheaper than synthetic cohort, but requires
user approval for ONE rendering pass): add the four `event_delay_*`
fields to an existing trace script and re-render a small candidate set
of 5–10 cases (DT_movement and FS_movement buckets are the most
likely to fire real delay events because the lpb path is
non-stationary). This is documented in §4 as an optional micro-bench
the user may authorise instead of synthetic.

---

## 1. Where the orchestrator emits delay-event markers

Verified in [python/modules/orchestrator.py](AEC/python/modules/orchestrator.py):

| Field | Source line | Semantic |
|---|---|---|
| `self._diag['div_source_last']` | 3448 | Last fired EPC reset source (string: `'delay_first'`, `'delay_shift'`, `'epv'`, `'shadow_rise'`, or `''`) |
| `self._diag['div_counts']` | 3449-3451 | Running per-source counter (dict, monotonically non-decreasing) |
| `self._diag['delay_samples']` | 3484 | Current `_current_delay` value (samples; -1 if not set) |
| `self._diag['delay_delta']` | 3486 | Per-frame change in `delay_samples` (0 if same; magnitude == sample shift on a `delay_shift` event) |
| `self._diag['event_delay_first']` | 3494 (loop) | **True on the frame `delay_first` fires** (computed via `div_counts['delay_first']` increment vs prev frame) |
| `self._diag['event_delay_shift']` | 3494 (loop) | True on the frame `delay_shift` fires |
| `self._diag['event_epv']` | 3494 (loop) | True on the frame EPV fires |
| `self._diag['event_shadow_rise']` | 3494 (loop) | True on the frame shadow_rise fires |
| `self._diag['p_max_override_remaining']` | 3498 | Legacy P diagnostic countdown (out of Q1 scope, retained here only for completeness) |

**These fields exist in `self._diag` on every frame.** They are NOT
persisted unless a trace script reads them.

Site that fires `delay_first` (real delay-change event #1):
- [orchestrator.py:1866-1889](AEC/python/modules/orchestrator.py) — Path A
  in delay-est pipeline (first acquisition). Triggers
  `self._maybe_mark_diverged('delay_first')` which increments
  `_round3_div_counts['delay_first']`.

Site that fires `delay_shift` (real delay-change event #2):
- [orchestrator.py:1921-1959](AEC/python/modules/orchestrator.py) — Path B
  (shift detection: `abs(new_delay - self._current_delay) > 32` AND
  confidence ≥ 0.5 AND consistent-consecutive). Triggers
  `self._maybe_mark_diverged('delay_shift')`.

Both call `filter.handle_echo_path_change(delay_change=True, ...)`
under production default flags (`use_aec3_handle_echo_path_change=True`),
which fires `H_error_per_bin.fill(10000)`.

---

## 2. Inventory of existing trace artifacts (read-only audit)

### 2.1 800-case bench outputs (`out_800_*`)

| Directory | Contents | Per-case diag | `event_delay_*` captured? |
|---|---|---|---|
| `out_800_v3_21_15_V0/` | 2400 WAV files (`*_ours.wav` + `*_ours_nores.wav` × 800 cases) | None | **No** |
| `out_800_v3_21_20_min_noFixA/` | 2400 WAV files | None | **No** |

Bench harness `eval_aec_challenge.py` does not emit per-case diag JSON.
**Cannot use to scan for delay events without re-rendering.**

### 2.2 12-case stress cohort outputs (`out_12_*`)

13 directories total (BASE, ALLON, fix-variants, …). Inspected
representative samples:

| Directory | Contents | `event_delay_*`? |
|---|---|---|
| `out_12_v3_21_20_ALLON_v2/` | WAV only | **No** |
| `out_12_A_baseline/` | WAV only | **No** |
| `out_12_v3_21_20_fixab/` | WAV only | **No** |
| (others same pattern) | — | — |

### 2.3 Phase A state JSONs (`out_v3_21_20_phase_a/`)

3 cases (nVUnxqHLr, XRTnTUjU, wVYS_mvmt), 2 variants (BASE, ALLON).
Format: `state` field is a dict with 26 per-frame columns:

```
frames_total, frames_use_capture, usable_linear, transparent_mode_active,
dominant_nearend_like_state, fq_usable, fq_convergence_seen,
fq_far_active_recent, filter_converged, mu_scale, filter_w_norm,
shadow_w_norm, mic_power_db, far_power_db, error_power_db,
erle_inst_db, erl_db, dt_from_energy, dt_from_shadow,
dt_from_coherence, dt_active, epc_active, res_gain_mean_db,
echo_psd_mean_db, error_psd_mean_db, frame_idx
```

**`event_delay_*` and `delay_samples`: NOT present.**

Only `epc_active` (boolean) is captured — a derived signal that fires
on EPV / shadow_rise / delay_first / delay_shift indistinguishably, so
useless for separating real-delay from pseudo-delay events.

### 2.4 Phase E trace JSONs (`out_v3_21_20_phase_e_trace/`)

3 cases × 4 variants (A_off, E_on, F_off, F_on). 31 per-frame columns:

```
frame, t_s, mic_rms, ref_rms, out_rms,
epv_event_raw, epv_event_suppressed, epc_active, epc_active_now,
epc_hangover_count, epc_render_forced_remaining,
converged, filter_converged_now, filter_once_converged,
filter_w_norm, shadow_w_norm, main_paused,
p_max_override_active, p_max_override_remaining,
usable_linear, usable_linear_estimate_v1, usable_linear_estimate_v2,
erle_reset_signal, copy_err_baseline, epv_gain_ratio,
H_error_mean, H_error_max,
call_counter_refined, poor_exc_counter_refined,
call_counter_shadow, poor_exc_counter_shadow
```

**`event_delay_*` and `delay_samples`: NOT present.** Only EPV is
captured as an event signal (`epv_event_raw`).

Script ([`v3_21_20_phase_e_trace.py:100`](AEC/python/v3_21_20_phase_e_trace.py))
sets `enable_delay_est=False` — so even if `event_delay_*` were
persisted, they would always be 0 in these traces.

### 2.5 Phase G isolation trace JSONs (`out_v3_21_20_phase_g_trace/`)

3 cases × 2 variants (A_off, G). 14 per-frame columns:

```
frame, t_s, mic_rms, out_rms,
epv_event_raw, epv_event_suppressed, epc_active, epc_active_now,
filter_w_norm, shadow_w_norm, erle_reset_signal,
H_error_mean, call_counter_refined, poor_exc_counter_refined
```

**`event_delay_*` and `delay_samples`: NOT present.** Same
`enable_delay_est=False` caveat as Phase E
([`v3_21_20_phase_g_isolation.py:76, 90-95`](AEC/python/v3_21_20_phase_g_isolation.py)).

### 2.6 Older per-frame trace (`trace_v3_21_15_interaction/`)

Multiple cases as `*_V{0,1,2,3}.npz`. 38 per-frame arrays covering
A.2 / A.3 / `usable_linear` / convergence / poor_excitation /
gain-stage attribution / x2 LF/MF/HF / W_norm / E2 / shadow_W.

**`event_delay_*`, `div_counts`, `delay_samples`, `delay_delta`: NOT
present.** Different audit purpose.

### 2.7 800-case aggregated summary (`trace_v3_21_16_800/`)

800 cases × {V0, V3}, format `*__V{0,3}__summary.npz` containing
mean/std per case (no per-frame). 60+ fields — `a2_zero_*`, `dn_*`,
`fq_*`, `main_W_norm_mean`, `shadow_*`, `poor_excitation_*`,
`x2_*_mean`, `nores_e_mean`, `res_gain_*`, etc.

**`event_delay_*` and `delay_samples`: NOT present** in either V0 or
V3. Aggregated stats only — even if a case fires a single delay event,
it would not be visible in mean/std of 800-1000 frame windows.

### 2.8 No CSV or log files

`find` over `/Users/mingyu/Desktop/novatek/SE/AEC -type f \( -name "*.csv"
-o -name "*.log" \)` returns nothing. The default render scripts emit
WAV; the trace scripts emit JSON / NPZ; no plaintext logs persist
delay-event markers either.

---

## 3. Why every existing trace is blind to real delay events

Two compounding causes summarised:

### 3.1 Trace-script instrumentation gap

`run_one_case.py` captures `s.delay_samples` and `s.delay_ms` via
AecStats per frame ([run_one_case.py:160](AEC/python/run_one_case.py))
to its CSV (when `--csv` is passed). But:
- `run_one_case.py` is a single-case-with-PNG helper, not a cohort
  trace tool.
- No existing `out_*` directory contains a CSV emitted by this script.
- AecStats does **not** include `event_delay_first` / `event_delay_shift`
  in its 33 documented stats fields (see
  [python/modules/dataclasses.py](AEC/python/modules/dataclasses.py)).
  The boolean event markers live in `self._diag` (per-call dict) and
  need to be read directly from the AEC instance after `process()`.

The cohort-trace scripts (`v3_21_20_phase_{a,e,g}_trace.py`) read
`aec._diag` but pick only the subset of fields relevant to the
EPC / W_norm / H_error / usable_linear audits they were authored for.
The `event_delay_*` family was added to `_diag` for a different audit
(Round 7 trace, lines 3477-3495) and was not back-filled into the
cohort-trace scripts.

### 3.2 Trace-render config gap

`v3_21_20_phase_e_trace.py` and `v3_21_20_phase_g_isolation.py` both
explicitly set:

```python
overrides = dict(enable_cng=True, enable_delay_est=False)
# ...
delay = estimate_delay(mic[:n], ref[:n], sr)
if 0 < delay < n:
    ref_aligned = np.zeros_like(ref)
    ref_aligned[delay:] = ref[:n - delay]
```

By **disabling `enable_delay_est`** and pre-aligning ref/mic via a
one-shot xcorr at render start, those scripts intentionally remove
the variability the delay estimator would introduce — useful for the
EPC / H_error variance audits those scripts targeted, but means
`delay_first` and `delay_shift` are structurally impossible to fire in
those renders, regardless of what `_diag` is persisted.

### 3.3 Production bench config

`eval_aec_challenge.py` runs production-default settings with
`enable_delay_est=True` (the bench config that produced
`out_800_v3_21_15_V0/` etc.). So real delay events DO fire there — but
the bench emits WAV only and discards `_diag` after each case.

This means **real delay-event evidence has never been persisted by any
existing artifact in the repo.**

---

## 4. Candidate list and zero-hit declaration

Per the task spec:

> 2. Produce a candidate list:
>    - 3–6 real delay-event cases if available
>    - include at least one DT/stress and one normal guard if possible
>    - for each case, report event frame(s), bucket, available diag
>      fields, and whether raw artifact already exists

**Result: candidate list size = 0.** No existing artifact identifies a
single real-delay-event case. Reporting per the structured fallback:

> 3. If no real delay_first / delay_shift cases exist in current
>    artifacts: explicitly report zero-hit result, recommend
>    synthetic delay-shift cohort as next step.

### Recommended next steps (user must pick + authorise)

**Path A — synthetic delay-shift cohort (cleanest)**

Construct 3–6 controlled cases by mechanical post-processing of
existing AEC challenge wav files: take a clean DT_static case (e.g.
`014AzuqPZku2004NbTTmcA_nearend_singletalk` from the 800-case set)
and insert known `lpb`-channel delay shifts at known times
(e.g. +200 sample shift at t=10 s, +500 at t=20 s, −300 at t=30 s).
Then render with current production main + a small Q1-purpose trace
script that persists `event_delay_first` / `event_delay_shift` /
`delay_samples` / `H_error_mean` / `out_rms` per frame.

Cost:
- Synthetic data prep: ~50 lines of pure numpy `np.roll` work + WAV I/O.
  Mechanical, no algorithm change.
- Render: 3–6 cases at ~2× real-time = 1–3 minutes.
- Trace script: small read-only extension to one of the existing
  `v3_21_20_phase_*_trace.py` scripts. Add 4 `_diag` field reads;
  re-enable `enable_delay_est=True`; remove the pre-align hop. No
  algorithm code change.

Verifies whether elevated steady leakage handles real delay shifts
adequately AND gives us a controlled cohort with known event times for
clean trajectory analysis.

**Path B — production-config 800-case trace re-render with delay-event diag**

Re-run the 800-case bench with `enable_delay_est=True` (production
default) + a trace script that persists the four `event_delay_*`
fields + `delay_samples` + `H_error_mean` + `out_rms` per frame.
Identify the subset of cases that fire real `delay_first` /
`delay_shift` events.

Cost:
- Re-render: ~2 hours wall (800 cases × few seconds each × 4 workers).
  Same render-config as the standard bench; only the diag dump field
  set differs.
- Storage: 800 cases × ~10 KB JSON each = ~10 MB. Trivial.
- No algorithm change.

This is more expensive than Path A but provides empirical evidence of
how often real delay events actually occur in the production bench
cohort. If the answer is "≤ 5 cases out of 800", Q1 is essentially
moot in practice (the stance-(b) functional-equivalence claim is
vacuously close because there's nothing to test on).

**Path C — surgical 5–10 case mini-bench**

Compromise: pick 5–10 DT_movement + FS_movement cases from the
800-case set (these are the most-likely-to-have-delay-shifts buckets
because the lpb path is non-stationary by construction). Re-render
ONLY those with production-default config + the extended diag dump.
Inspect `event_delay_*` counts per case.

Cost:
- Re-render: ~30 seconds wall.
- Storage: 5–10 JSON files, ~100 KB total.
- No algorithm change.

Likely fastest path to either confirming a small candidate list or
confirming the "delay events are rare in the bench" finding.

### Selection guidance

- If the user wants the cleanest answer to Q1 (controlled event
  timing, deterministic trajectory math): **Path A**.
- If the user wants to know whether Q1 is even relevant on the
  production cohort: **Path C** first (cheap), promote to **Path B**
  only if the small sample shows non-zero events.
- If the user wants to skip Q1 empirical work entirely and close it
  by documentation: that's the original
  [trace-audit §5.4 path 3](v3_21_q1_trace_audit.md) (close as
  stance (b) with corrected framing per the §4 math finding).

**This scan does NOT make the selection. The user picks.**

---

## 5. What this scan does NOT do

- Does NOT render anything.
- Does NOT modify any algorithm code.
- Does NOT modify any trace script in this round (the §4 paths
  describe what a future extension would need; no code is written
  here).
- Does NOT modify the Q1 verdict (still C — insufficient evidence per
  [docs/v3_21_q1_trace_audit.md](v3_21_q1_trace_audit.md)).
- Does NOT reclassify EPV / shadow_rise as real delay events. The 8
  H_error reset events observed in
  [out_v3_21_20_phase_g_trace/](../out_v3_21_20_phase_g_trace/) are
  all EPV / shadow_rise dispatched as `delay_change=True` under Phase D
  wiring — they fire the H_error reset code path but do NOT exercise
  delay-change physics, and the trace audit §3.4 has them on the
  record.
- Does NOT touch Phase D / Phase G / M2 / M3 / usable_linear /
  PathChangeRegimeHandler / RES.
- Does NOT use `_p_max_override` as a Q1 adapter.
- Does NOT propose a v3.22 framing.
