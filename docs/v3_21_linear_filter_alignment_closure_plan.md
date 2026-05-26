# v3.21 Linear Filter Alignment — Closure Plan

**Date**: 2026-05-25  
**Scope**: Convergence of v3.21.x AEC3 linear-filter alignment; classifying remaining gaps;
authorizing ONLY Variant F Gate 0 and usable_linear read-only trace.  
**Authority**: After this plan, no new variant or flag flip is authorized until Variant F
Gate 0 result is known.

---

## 0. Corrected source facts

### W=0 claim RETRACTED

Prior documentation (v3.21 alignment sweep draft, audit B, q1 phase 2 verdict, roadmap)
incorrectly stated "AEC3 delay_change = W=0 / full ZeroFilter / W reset". This is retracted.

**Verified from `docs/aec3_extracts/src/aec3/adaptive_fir_filter.cc:460-509`**:

```cpp
void ZeroFilter(size_t old_size, size_t new_size, ...) {
    for (size_t p = old_size; p < new_size; ++p) { H[p][ch].Clear(); }
}
void AdaptiveFirFilter::HandleEchoPathChange() {
    ZeroFilter(current_size_partitions_, max_size_partitions_, &H_);  // no-op when equal
}
```

At AEC3 defaults (`current_size = max_size = 13`):
- `HandleEchoPathChange()` = `ZeroFilter(13, 13)` = loop `for p=13; p<13` = **empty range**
- `SetSizePartitions(12, true)` → `ZeroFilter(13, 12)` = also **empty range (13>12)**

AEC3 retains stale W in partitions 0–11. Partition 12 becomes inactive (not zeroed).
Python `zero_filter=False` **matches AEC3** — both systems retain stale W after delay_change.

**Python `W.fill(0)` = non-AEC3 ablation** — MORE aggressive than AEC3 (zeros all active
partitions). NOT a v3.21 alignment candidate. If ever tested must be labelled as ablation.

### Corrected files (2026-05-25)
- `docs/v3_21_linear_filter_alignment_sweep.md` §0 + §1.2 S1 + §5 + §6 + §8
- `docs/v3_21_q1_short_window_final_sanity.md` §Structural conclusion + §ARC OPEN note
- `docs/v3_21_audit_b_m1_m4_parity.md` header retraction block
- Memory: `project_v3_21_alignment_roadmap.md` + `project_v3_21_q1_phase2_verdict.md`
  + `project_v3_21_a1_gate1_verdict.md`

---

## 1. Complete linear-filter alignment matrix

Legend:
- **MATCH**: behavior identical (or threshold-equivalent under float→int16 rescaling)
- **APPROX**: different formula, same semantic intent; confirmed no behavioral cliff
- **FUNCTIONAL ADAPTATION**: documented PBFDKF divergence from AEC3, accepted
- **IMPL-OFF**: implemented behind flag, default-OFF (byte-equal to baseline when OFF)
- **MISSING**: not ported; no flag
- **NON-AEC3 ABLATION**: implemented but more aggressive or structurally different from AEC3
- **CLOSED NO-SURFACE**: behavior gap exists but gate never fires in practice

All rows evaluated for Variant D config unless noted otherwise.

### echo_remover.cc

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| E1 | UseRefinedOutput (refined-vs-coarse selection) | IMPL-OFF | **HIGH** | OFF | Primary gap. AEC3 default-ON. Variant F enables. |
| E2 | FormLinearFilterOutput 30-sample crossfade | APPROX | LOW | ON | sqrt-Hann/OLA hop-aligned ≈ AEC3 SignalTransition crossfade |
| E3 | UseLinearFilterOutput Y vs E final selection | IMPL-OFF | N/A | ON | Active in D. AEC3 production default. |
| E4 | nearend_spectrum = UsableLinear ? E² : Y² | MATCH | LOW | ON | |
| E5 | ResidualEchoEstimator input selection | APPROX | MED | ON | |
| E6 | HandleEchoPathChange trigger (outer) | MATCH | N/A | ON | Triggers inner S1-S6 chain |
| E7 | ExitInitialState: SuppressionGain exit | PARTIAL | MED | ON | SG exit ported; filter config switch MISSING (S3/S5) |

### subtractor.cc / adaptive_fir_filter.cc

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| S1 | HandleEchoPathChange — ZeroFilter(current, max) | FUNCTIONAL ADAPTATION | MED | ON (=no-op) | AEC3 ZeroFilter(13,13)=no-op; Python `zero_filter=False` matches. Python W.fill(0) = NON-AEC3 ABLATION. |
| S2 | H_error reset to 10000 on delay_change | MATCH | LOW | ON | |
| S3 | Config switch → refined_initial (transient gain) on delay_change | MISSING | MED | NO | PBFDKF has single fixed config. No IMPL-OFF substrate. |
| S4 | SetSizePartitions: shrink 13→12 on delay_change | MISSING | LOW | NO | Fixed partition count PBFDKF. Functional adaptation documented. |
| S5 | ExitInitialState config switch: grow 12→13 on convergence | MISSING | LOW | NO | Symmetric with S4. Fixed partition count. |
| S6 | CoarseFilter::HandleEchoPathChange — ZeroFilter(coarse) | FUNCTIONAL ADAPTATION | MED | NO | Same no-op semantics as S1. Shadow path not in D. |
| S7 | Refined-then-coarse update order | MATCH | LOW | ON | |
| S8 | E²/S²/Y² block metrics | MATCH | LOW | ON | |

### subtractor_output_analyzer.cc

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| A1 | refined_filter_converged: e² < 0.5·y² ∧ y² > threshold | MATCH | LOW | ON | |
| A2 | coarse_filter_converged_strict: e² < 0.05·y² | MATCH | LOW | ON | |
| A3 | coarse_filter_converged_relaxed: e² < 0.3·y² | IMPL-OFF | LOW | NO | No consumer (TransparentMode ignores it) |
| A4 | all_filters_diverged: min(e²_ref, e²_coa) > 1.5·y² | APPROX | MED | ON | Bridge proxy divergence_indicator |

### aec_state.cc / filter_quality.py

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| AS1 | FilterAnalyzer peak-based delays | IMPL-OFF | MED | NO | filter_analyzer_enabled=False in AecStateConfig |
| AS2 | FilteringQualityAnalyzer 4-gate AND | MATCH | LOW | ON | |
| AS3 | Gate-3: external_delay OR convergence_seen | MATCH | MED | ON | Logic correct; but see AS4 |
| AS4 | External delay quality: REFINED vs permissive | **HIGH** | HIGH | ON (permissive) | Python `usable_linear_trusted_external_delay_only=False` → any `_current_delay >= 0` = ext_delay. AEC3 requires matched-filter quality. Secondary gap. |
| AS5 | UsableLinearEstimate() | MATCH | LOW | ON | |
| AS6 | InitialState duration (5s conservative / 2.5s normal) | MATCH | LOW | ON | |
| AS7 | AecState full_reset on delay_change | MATCH | LOW | ON | |

### refined_filter_update_gain.cc

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| R1 | X² source: partition-summed SpectralSum | MATCH | LOW | ON | use_partition_summed_x2_for_h_error_gain=True (BALANCED default) |
| R2 | E² source: instantaneous current-block | IMPL-OFF | MED | NO | Python uses 0.95 EMA → ~2× mu inflation first 200 ms post-reset |
| R3 | H_error conditional refresh (e²_ref ≤ e²_coa → leakage_converged) | MATCH | LOW | ON | Scalar default path |
| R4 | H_error per-bin refresh | IMPL-OFF | MED | NO | |
| R5 | Noise gate: X² < noise_gate_power → mu=0 | MATCH | LOW | ON | |
| R6 | Poor excitation gate + call counter | MATCH | LOW | ON | |
| R7 | Narrowband mask (RenderSignalAnalyzer) | MATCH | LOW | ON | |
| R8 | Saturation gate | MATCH | LOW | ON | |
| R9 | H_error init = 10000 on cold-start | MATCH | LOW | ON | |

### coarse_filter_update_gain.cc (shadow, Variant D = all OFF)

| ID | AEC3 Node | Status | Risk | Variant D | Notes |
|---|---|---|---|---|---|
| C1 | X² partition-summed in shadow mu denom | IMPL-OFF | MED | NO | Affects shadow convergence speed and coarse fallback quality |
| C2 | Noise gate: X² < noise_gate → mu=0 | IMPL-OFF | MED | NO | Same |
| C3 | Poor excitation + call counter gate | IMPL-OFF | MED | NO | Same |
| C4 | Narrowband mask | IMPL-OFF | LOW | NO | |
| C5 | Saturation gate | IMPL-OFF | LOW | NO | |
| C6 | HandleEchoPathChange — ZeroFilter(coarse) | FUNCTIONAL ADAPTATION | MED | NO | No-op semantics same as S1/S6. Covered by S6. |
| C7 | Fixed-gain LMS shadow (`rate / X²`, no Kalman) | FUNCTIONAL ADAPTATION | MED | N/A | Python shadow = PBFDKF Kalman. Documented divergence. |

**C1-C5 disposition**: NOT irrelevant. All five OFF in Variant D. They affect coarse output
quality, `coarse_filter_converged` / `any_filter_converged` signals (consumed by
TransparentMode + FilteringQualityAnalyzer), and — once E1 (UseRefinedOutput) is enabled —
the quality of the coarse fallback path. A poorly-converged shadow degrades the coarse
fallback even when E1 fires correctly.  
**Decision**: Do NOT enable C1–C5 in Variant F. Trace Variant F Gate 0 first; if coarse
fallback fires meaningfully, evaluate C1–C5 impact in a subsequent authorized variant.

---

## 2. Primary gaps (ordered by mechanistic connection to Variant D failure)

### Gap 1 — UseRefinedOutput (HIGH, E1)

AEC3 `echo_remover.cc:112-130` runs a per-frame predicate on every frame:
```
cond1: e²_coarse < 0.9·e²_refined AND y² > 30²·64 AND (s²_ref > 60²·64 OR s²_coa > 60²·64)
cond2: e²_coarse < e²_refined AND y² < e²_refined   ← refined diverged
```
When either condition fires → output switches to coarse (shadow) filter for that frame.

During delay_change re-convergence with stale W: refined filter has high e²_refined (stale
subtraction) → cond2 fires → AEC3 routes RES/SuppressionGain through coarse path.
Variant D always uses refined path — no coarse rescue during divergence.

**Variant F** = D + `use_refined_output_selection_for_linear_path=True` ports this.

**Prior result** (v3.21.8, UseRefinedOutput without ULO): 37% cond2 rate in XRTnTUjU
DT_static frames → catastrophic DT_static regression (shadow NLMS not converged on
no-clean-convergence cases). Variant F has ULO companion which provides Y-path when
`usable_linear=False` — interaction is different from v3.21.8.

### Gap 2 — Permissive usable_linear gate (HIGH, AS4)

AEC3 gate-3 (`aec_state.cc:432-433`): `external_delay OR convergence_seen`.  
AEC3 `external_delay` comes from matched-filter EchoPathDelayEstimator — promoted only
when confidence is robust.

Python (`orchestrator.py:4007-4023`, `filter_quality.py:111-115`) with
`usable_linear_trusted_external_delay_only=False` (default): ANY `_current_delay >= 0`
provides `external_delay`. Gate-3 latches True immediately at first valid delay estimate,
compressing the Y-path window and extending the stale-W E-path window.

**Gate 3 read-only trace** (`v3_21_a1_f_gate0.py`) measures `Δframes` between
`ext_delay_available` and `convergence_seen` per DT_mvmt case. Result informs whether
a Variant G with `usable_linear_trusted_external_delay_only=True` is warranted.

---

## 3. Authorized experiments (2026-05-25)

### PRIMARY: Variant F Gate 0 trace — COMPLETED (2026-05-25)

**Script**: `python/v3_21_a1_f_gate0.py`  
**Config**: `use_full_delay_change_chain=True` + `use_linear_filter_output_selection_for_final_output=True` + `use_refined_output_selection_for_linear_path=True`  
**No W.fill(0), no R1/R3, no 800-case**

**Step 1 — Gate 0 trace** (DT_mvmt + DT_static focus cases):
```
trace_hf_chain=True → _hf_chain_trace per frame
```
Fields collected: `uro_fire`, `uro_cond1`, `uro_cond2`, `uro_path`, `uro_s2_refined`,
`uro_s2_coarse`, `ext_delay_available`, `convergence_seen` (all default-OFF, byte-equal).

**Gate 0 criteria**:
- G0.1: UseRefinedOutput fires ≥1 DT_mvmt failing-window frame
- G0.2: cond2 fires during delay_first re-convergence window (intended rescue)
- G0.3: coarse path selected in DT_mvmt re-convergence windows
- G0.4: DT_static stable cond2 rate < 15% (prevents v3.21.8 regression)

**If Gate 0 PASS** → run 12-case AECMOS (script continues automatically).  
**If Gate 0 FAIL** → diagnose; classify as CLOSED NO-SURFACE if cond2 zero-fire.

**Verdict doc**: `docs/v3_21_a1_f_gate0_verdict.md` (auto-written by script).

### PARALLEL: Gate 3 read-only trace

Included in `python/v3_21_a1_f_gate0.py` Gate 3 section.  
No config change. Outputs `Δframes` per DT_mvmt case.  
Result informs whether Variant G (trusted gate) is warranted — no code change authorized.

---

## 4. Unresolved items requiring separate authorization

These items were identified during the alignment sweep or during Variant F execution.
They go in this matrix; no new variant or sprint is authorized without explicit user authorization.

| Item | Description | Status after Variant F | Authorization needed |
|---|---|---|---|
| R2 (instantaneous E²) | AEC3 uses current-block E² in mu denom; Python uses 0.95 EMA → ~2× mu inflation first 200 ms post-reset | Deferred | Separate variant; explicit authorization |
| C1–C5 (shadow gates) | All 5 shadow protection flags OFF; degrade coarse fallback quality → limits UseRefinedOutput effectiveness on xFk7 / ZJYUt0O0 | **Coarse surface CONFIRMED by Variant F Gate 0**; shadow quality is likely limiting factor on worst cases | Authorized as next candidate — requires user authorization. Suggest: Variant H = F + C1+C2+C3 |
| Variant G (trusted gate) | `usable_linear_trusted_external_delay_only=True`; extends Y-path by Δframes=23–71 | **Gate 3 trace DONE**: Δframes=23/56/71 — permissive gate compresses Y-path | Authorized as candidate — requires user authorization. Can combine with C1–C5 |
| UseRefinedOutput broad fire (36–60% DT_mvmt) | cond2 fires during all DT_movement, not just re-convergence; mechanism works for wVYSGVTT (+0.117) but insufficient for xFk7/ZJYUt0O0 | NEW — found during Variant F run | Document; do NOT treat as defect to fix without evidence — may be AEC3 same behavior |
| S3/S4 (partition resize) | AEC3 shrinks 13→12 on delay_change; Python fixed at 13 | LOW risk; minor filter-reach difference | Separate v3.21.x parity arc; low priority; deprioritize vs C1-C5/G |
| C7 (shadow = PBFDKF vs NLMS) | Python shadow Kalman vs AEC3 coarse fixed-gain LMS | Fundamental architectural divergence; documented | Not a v3.21.x parity target; PBFDKF shadow is intentional |

---

## 4b. Variant F results summary (2026-05-25)

| Metric | Result |
|---|---|
| Gate 0 | **PASS** — all 4 criteria met |
| Gate 3 Δframes | 23 (ZJYUt0O0) / 56 (wVYSGVTT) / 71 (xFk7) |
| Gate 1 12-case | **FAIL** |
| DT_mvmt F−A mean | −0.351 (slight improvement vs D −0.392) |
| DT_mvmt wVYSGVTT | **F−A = +0.117** (rescued by UseRefinedOutput) |
| DT_mvmt xFk7 | F−A = −0.692 (rescue insufficient) |
| DT_mvmt ZJYUt0O0 | F−A = −0.478 (rescue insufficient; EPV events complicate) |
| DT_static F−A | **+0.185** (improved — ULO insulates from UseRefinedOutput damage) |
| FS F−A | −0.364 mvmt / −0.094 static (improved, except 1 static case regressed) |
| Category A | **NOT VALID** — wVYSGVTT +0.117 proves partial rescue is possible |

**Key conclusion**: UseRefinedOutput works (wVYSGVTT recovered). The remaining failure on
xFk7 / ZJYUt0O0 is attributed to shadow quality (C1–C5 all OFF) degrading the coarse
fallback during re-convergence. Gate 3 compression (23–71 frames) is a secondary contributor.

---

## 5. Explicitly forbidden items

| Item | Reason |
|---|---|
| `W.fill(0)` / `zero_filter=True` as alignment candidate | Non-AEC3 ablation (RETRACTED). If ever tested: separate arc, explicit user authorization, must be labelled ablation. |
| 800-case bench | Not authorized until 12-case AECMOS PASS |
| Category A classification (PBFDKF structurally incompatible) | **NOT VALID** — Variant F wVYSGVTT +0.117 proves PBFDKF is NOT structurally incompatible; UseRefinedOutput with C1-C5 and/or trusted gate may complete the rescue |
| R1/R3 functional adaptation (NE-gate ERLE reset, skip ERLE reset) | Not authorized. Requires Category A proven first. |
| v3.22 framing of A.1 work | A.1 is v3.21.x AEC3 alignment — must complete in v3.21.x |
| C1–C5 enabling in Variant F | Not authorized for Variant F. Evaluate after Gate 0 confirms coarse fallback surface. |
| New variants beyond F during Gate 0 cycle | If new doubts found, put in §4 matrix above; do not open new sprint. |

---

## 6. Instrumentation changes (2026-05-25)

All changes are trace-only. Byte-equal verified (max diff = 0.00e+00 with `trace_hf_chain=False`).

**`python/modules/orchestrator.py`**:
- `self._last_uro_frame_trace = {}` initialized in `__init__` (near `_hf_chain_trace = []`)
- `_aec3_select_linear_filter_output`: stores per-frame `{fire, cond1, cond2, path, s2_refined, s2_coarse}` in `self._last_uro_frame_trace`
- Fallback branch in `_aec3_post`: sets `_last_uro_frame_trace` to zeros (fire=False, path='refined')
- `_hf_chain_trace.append`: 8 new fields — `uro_fire`, `uro_cond1`, `uro_cond2`, `uro_path`, `uro_s2_refined`, `uro_s2_coarse`, `ext_delay_available`, `convergence_seen`
- All 8 fields only populated when `trace_hf_chain=True`; no production impact

---

## 7. Hard rules (unchanged)

- **A.1 remains OPEN** until user explicitly closes
- **No 800-case** without explicit user authorization
- **No v3.22 framing** of A.1
- **R1/R3 NOT authorized** (requires Category A proven first)
- **No Category A claim** until Variant F Gate 0 + 12-case complete
- **No Variant E** (W.fill(0) deprecated as non-AEC3 ablation)
- **No new variants** until Variant F Gate 0 result is known
- **New doubts → §4 matrix**, not new sprint

---

## 8. Next step

Run `python3 python/v3_21_a1_f_gate0.py` from the AEC repo root.

Script runs Gate 0 trace on DT_mvmt (3 cases) + DT_static (4 cases) with `trace_hf_chain=True`,
prints per-case stats, evaluates G0.1–G0.4, prints Gate 3 trace, auto-writes
`docs/v3_21_a1_f_gate0_verdict.md`, and if Gate 0 PASS continues to 12-case AECMOS.

Prerequisite: `wav/v3_21_8_cohort/` cohort WAV files present.
