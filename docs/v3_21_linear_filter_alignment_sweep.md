# v3.21 Linear Filter AEC3 Alignment Sweep

**Date**: 2026-05-25  
**Scope**: Read-only audit + per-frame divergence analysis for Variant D Gate 1 FAIL  
**Method**: AEC3 C++ reference (`docs/aec3_extracts/src/aec3/`) vs Python implementation  
**Directive**: Do NOT close arc, propose R1/R3, or run 800-case until every node below is classified.

---

## 0. Summary verdict

**Verdict: B — INCONCLUSIVE**

**⚠ CORRECTION (2026-05-25)**: Prior draft incorrectly stated AEC3 zeros filter weights
(`W=0`) on delay change. This is RETRACTED. The corrected AEC3 behavior is documented in
§1.2 S1 (ZeroFilter semantics analysis). Python W.fill(0) is NOT AEC3 parity — it would be
a non-AEC3 ablation and cannot be labelled a v3.21 alignment candidate.

Variant D Gate 1 FAIL remains inconclusive because two materially significant AEC3 linear
path behaviors are absent from Variant D:

1. **UseRefinedOutput NOT enabled** (HIGH — default-OFF in Python):
   AEC3 runs `UseRefinedOutput` by default (`enable_coarse_filter_output_usage=True`).
   During re-convergence, the refined filter (stale W) produces high `e2_refined`; AEC3
   falls back to coarse output via the `UseRefinedOutput` predicate. Variant D always uses
   the refined path — no coarse fallback during divergence.

2. **Permissive `usable_linear` gate** (HIGH — external_delay shortcut active):
   Python's gate-3 latches `usable_linear=True` as soon as `_current_delay >= 0`, regardless
   of filter convergence. AEC3 requires `external_delay` from the matched-filter estimator
   OR `convergence_seen`. Python compresses the Y-path protection window and extends the
   stale-W refined E-path window vs AEC3.

Both gaps have direct mechanistic connection to the DT_mvmt damage in Variant D. The Gate 1
failure does NOT prove PBFDKF is structurally incompatible with AEC3 delay-change
composition — it proves UseRefined-only + permissive gate is insufficient.

**Category A classification is NOT valid until these two gaps are tested.**

---

## 1. AEC3 → Python master alignment matrix

Sources:
- AEC3 reference: `docs/aec3_extracts/src/aec3/`  
- Python: `python/modules/orchestrator.py`, `filters.py`, `state/aec_state.py`,
  `state/filter_quality.py`, `state/initial_state.py`

Legend:
- **MATCH**: behavior identical or threshold-equivalent
- **IMPL-OFF**: implemented, default-OFF (exists behind flag)
- **APPROX**: approximation (different formula, same semantic intent)
- **MISSING**: not ported
- **DIV**: intentional divergence (known, documented deviation)

Variant D config: BALANCED + `use_full_delay_change_chain=True` +
`use_linear_filter_output_selection_for_final_output=True`; all other fields = BALANCED
defaults.

### 1.1 echo_remover.cc

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| E1 | `UseRefinedOutput` (refined-vs-coarse selection) | `echo_remover.cc:112-130` | `orchestrator.py:3883` behind `use_refined_output_selection_for_linear_path` | IMPL-OFF | OFF | **NO** | HIGH |
| E2 | `FormLinearFilterOutput` crossfade (30-sample window) | `echo_remover.cc:134-147` | OLA/sqrt-Hann hop-aligned | APPROX | ON | YES (implicit) | LOW |
| E3 | `UseLinearFilterOutput` Y vs E selection | `echo_remover.cc:475` | `orchestrator.py:4513` behind `use_linear_filter_output_selection_for_final_output` | IMPL-OFF | OFF | **YES** | N/A (active) |
| E4 | `nearend_spectrum = UsableLinearEstimate() ? E2 : Y2` | `echo_remover.cc:452` | `orchestrator.py` via `_aec3_state.usable_linear_estimate()` | MATCH | ON | YES | LOW |
| E5 | `ResidualEchoEstimator` input selection | `echo_remover.cc:490-492` | `_aec3_ree.estimate(s2_linear=echo_psd)` | APPROX | ON | YES | MED |
| E6 | `HandleEchoPathChange` → `subtractor_.HandleEchoPathChange()` | `echo_remover.cc:401` | `orchestrator.py:1902` (delay_first) / `1991` (delay_shift) | MATCH-OUTER | ON | YES | N/A (triggers inner) |
| E7 | `TransitionTriggered` → `ExitInitialState()` | `echo_remover.cc:419` | `orchestrator.py:4095-4099` | PARTIAL | ON | YES | MED |

**E1 (UseRefinedOutput) detail**: AEC3 `UseRefinedOutput` is a per-frame predicate. When
`e2_coarse < 0.9 × e2_refined AND y2 > 30²·kBlockSize AND (s2_refined OR s2_coarse > 60²·kBlockSize)`,
or when `e2_coarse < e2_refined AND y2 < e2_refined` (refined diverged), the output path
switches to the coarse (shadow) filter. During delay-change re-convergence the refined filter
will have diverged (high e2_refined), so AEC3 would switch to coarse for those frames.
Variant D never makes this switch.

**E7 (ExitInitialState) partial**: AEC3 `ExitInitialState()` performs two things:
(a) `SuppressionGain::SetInitialState(false)` — **PORTED** at `orchestrator.py:4098`;
(b) config switch `refined_initial → refined` which grows filter length from 12→13 partitions
— **NOT PORTED** (Python PBFDKF has a single fixed partition count; no initial vs converged
config split).

### 1.2 subtractor.cc

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| S1 | `HandleEchoPathChange` — `ZeroFilter(current, max)` | `adaptive_fir_filter.cc:506-509`; `subtractor.cc:148-175` | `filters.py:PBFDKF.handle_echo_path_change(zero_filter=False)` | **APPROX** | N/A | YES (see S1 detail) | **MED** |
| S2 | `HandleEchoPathChange` — H_error reset to `kHErrorInitial=10000` | `refined_filter_update_gain.cc:53-68` | `filters.py:PBFDKF.handle_echo_path_change:506` | MATCH | N/A | YES | LOW |
| S3 | `HandleEchoPathChange` — config switch to `refined_initial` | `subtractor.cc:162-163` | NOT PORTED | MISSING | N/A | NO | MED |
| S4 | `HandleEchoPathChange` — partition size reset | `subtractor.cc:164-165` | NOT PORTED | MISSING | N/A | NO | LOW |
| S5 | `ExitInitialState` — config switch to `refined` (grow partitions) | `subtractor.cc:177-186` | NOT PORTED | MISSING | N/A | NO | LOW |
| S6 | `CoarseFilter::HandleEchoPathChange` — `ZeroFilter(current,max)` (same semantics as S1 = no-op when current=max) + gain counter reset | `subtractor.cc:168-169` + `coarse_filter_update_gain.cc` | `filters.py:PBFDAF.handle_echo_path_change(zero_filter=False)` | **APPROX** | N/A | NO | MED |
| S7 | Refined vs coarse filter update order (refined first) | `subtractor.cc:259, 287` | orchestrator: main filter before shadow | MATCH | ON | YES | LOW |
| S8 | `E2 / S2 / Y2` block metrics | `subtractor_output.cc:ComputeMetrics` | `orchestrator.py:3919-3948` | MATCH | ON | YES | LOW |

**S1 (ZeroFilter semantics) detail — CORRECTED ⚠**:

**Prior claim RETRACTED**: AEC3 does NOT unconditionally zero W on delay change.

Actual AEC3 code (`adaptive_fir_filter.cc:460-509`):
```cpp
// ZeroFilter: clears partitions in [old_size, new_size) only
void ZeroFilter(size_t old_size, size_t new_size, ...) {
  for (size_t p = old_size; p < new_size; ++p) { H[p][ch].Clear(); }
}

void AdaptiveFirFilter::HandleEchoPathChange() {
  // TODO(peah): Check the value and purpose of the code below.
  ZeroFilter(current_size_partitions_, max_size_partitions_, &H_);
}
```
When the filter operates at normal size (`current=max=13`):
- `ZeroFilter(13, 13)` → loop `for p=13; p<13` → **empty range, no zeroing**

`subtractor.cc:158-159` then calls:
```cpp
refined_filters_[ch]->SetSizePartitions(refined_initial.length_blocks, true);  // shrinks 13→12
```
`SetSizePartitions(12, immediate=true)`:
- `old_size=13`, `current_size→12`, `ZeroFilter(13, 12)` → **empty range (13>12), no zeroing**

**Conclusion**: AEC3 retains stale W in partitions 0–11 after delay change. Partition 12
(the 13th) becomes inactive (current_size=12), but its coefficients are not zeroed. Both
AEC3 and Python start re-convergence from stale W. Python retains 13 active partitions vs
AEC3's 12 — a MED difference in filter reach, not a fundamental behavioral gap.

**Python W.fill(0) is NOT AEC3 parity.** It would zero all 13 partitions — MORE aggressive
than AEC3, which zeros zero. If implemented, it must be labelled as non-AEC3 ablation and
cannot be a v3.21 alignment ship candidate.

Python comparison:
```python
# orchestrator.py:1902-1911 (A.1 delay_first)
self.filter.handle_echo_path_change(delay_change=True, zero_filter=False)       # retains W
self.shadow_filter.handle_echo_path_change(delay_change=True, zero_filter=False) # retains W
```
Python `zero_filter=False` matches AEC3 semantics (no active-partition zeroing). The H_error
reset to 10000 in both systems is correct (see S2). The MED difference is partition count
(Python=13, AEC3 shrinks to 12) — covered by S3/S4.

### 1.3 subtractor_output_analyzer.cc

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| A1 | `refined_filter_converged`: e2 < 0.5·y2 ∧ y2 > 2500·kBlock | `subtractor_output_analyzer.cc:46-47` | `orchestrator.py:3922` | MATCH | ON | YES | LOW |
| A2 | `coarse_filter_converged_strict`: e2 < 0.05·y2 | `subtractor_output_analyzer.cc:48-49` | `orchestrator.py:3948` | MATCH | ON | YES | LOW |
| A3 | `coarse_filter_converged_relaxed`: e2 < 0.3·y2, lower floor | `subtractor_output_analyzer.cc:50-51` | `orchestrator.py:3965` behind `coarse_filter_converged_relaxed_enabled` | IMPL-OFF | OFF | NO | LOW (TM consumer absent) |
| A4 | `all_filters_diverged`: min(e2_ref, e2_coa) > 1.5·y2 | `subtractor_output_analyzer.cc:52-53` | `bridge.divergence_indicator > 1.0` (proxy) | APPROX | ON | YES | MED |

**A3 no-consumer note**: AEC3 uses `coarse_filter_converged_relaxed` only in
`TransparentModeImpl` (HMM variant). Our Python `LegacyTransparentModeImpl` ignores it
(`transparent_mode.cc:150: bool /*any_coarse_filter_converged*/`). So even if enabled, it
has no behavioral effect. Low risk.

### 1.4 aec_state.cc

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| AS1 | `FilterDelay` — FilterAnalyzer peak-based delays | `aec_state.cc:199-206` | `state/aec_state.py:267-280` gated by `enable_filter_analyzer` | IMPL-OFF | FilterAnalyzer ON (`filter_analyzer_enabled=True` in config.py) but AecStateConfig.enable_filter_analyzer=False | Depends on AecState construction | MED |
| AS2 | `FilteringQualityAnalyzer` 4-gate AND | `aec_state.cc:420-437` | `state/filter_quality.py:100-121` | MATCH | ON | YES | LOW |
| AS3 | Gate-3: `external_delay OR convergence_seen` | `aec_state.cc:432-433` | `filter_quality.py:115` | MATCH | ON | YES | MED (see AS4) |
| AS4 | External delay quality: REFINED/COARSE distinction | `aec_state.cc:366-383` | `orchestrator.py:4007-4024` | APPROX | ON | YES | **HIGH** |
| AS5 | `UsableLinearEstimate()` | `aec_state.cc:452` (echo_remover.cc) | `state/aec_state.py:147-151` | MATCH | ON | YES | LOW |
| AS6 | `InitialStateActive` duration: conservative=5s, normal=2.5s | `aec_state.cc:343-348` | `state/initial_state.py` | MATCH | ON | YES | LOW |
| AS7 | `_full_reset` on delay change | `aec_state.cc:145-157` | `state/aec_state.py:360-374` | MATCH | ON | YES | LOW |

**AS4 (external delay quality) detail — HIGH risk**:

AEC3 `FilteringQualityAnalyzer` gate-3 passes when `external_delay.has_value() OR convergence_seen`. The AEC3 external delay is a high-quality estimate from the `EchoPathDelayEstimator` (matched-filter based; only promotes when confidence is robust).

Python (`orchestrator.py:4007-4024`):
```python
if self._delay_active and self._current_delay >= 0:
    _trusted_only = bool(getattr(self.config, 'usable_linear_trusted_external_delay_only', False))
    if _trusted_only:
        # High-confidence gate (flag OFF by default)
    else:
        ext_delay = DelayEstimate(quality=DelayQuality.REFINED, delay=int(self._current_delay))
```

With `usable_linear_trusted_external_delay_only=False` (default), ANY active `_current_delay`
(including the very first solid estimate) promotes as `external_delay`. Gate-3 latches True
immediately at `_current_delay >= 0`, regardless of filter convergence. This is **more
permissive** than AEC3 and causes `usable_linear` to flip True sooner after delay_first
— before the filter has re-converged — compressing the Y-path protection window and
extending the E-path damage window.

**This amplifies the UseRefinedOutput gap**: usable_linear=True fires sooner in Python than
AEC3 would, shortening the Y-path coverage and extending the window where the refined E-path
(with stale W, no coarse fallback) is used.

### 1.5 refined_filter_update_gain.cc

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| R1 | X² source: partition-summed `SpectralSum` | `subtractor.cc:196-213` | `filters.py:PBFDKF._update_weights_aec3` behind `_use_partition_summed_x2_for_h_error_gain` | IMPL-OFF | **ON** (config.py:278 `bool = True`) | YES | LOW (all A/B/D share same X²) |
| R2 | E² source in mu denominator: instantaneous current-block | `refined_filter_update_gain.cc:103-107` | `filters.py:PBFDKF._update_weights_aec3:773` behind `_use_current_e2_refined_in_h_error_denominator` | IMPL-OFF | OFF | **NO** | MED |
| R3 | H_error conditional refresh: E2_ref ≤ E2_coa → leakage_converged | `refined_filter_update_gain.cc:128-138` | `filters.py:PBFDKF._h_error_refresh` | MATCH (scalar default) | ON (scalar) | YES | LOW |
| R4 | H_error per-bin refresh path | `refined_filter_update_gain.cc:128-138` | `filters.py:PBFDKF._h_error_refresh` behind `_use_per_bin_h_error_refresh` | IMPL-OFF | OFF | **NO** | MED |
| R5 | Noise gate: X² < noise_gate_power → mu=0 | `refined_filter_update_gain.cc:105-110` | `filters.py:PBFDKF._update_weights_aec3:793` | MATCH | ON | YES | LOW |
| R6 | Poor excitation gate: startup call counter | `refined_filter_update_gain.cc:96-99` | `filters.py:PBFDKF._update_weights:543-546` | MATCH | ON | YES | LOW |
| R7 | Narrowband mask: `RenderSignalAnalyzer.MaskRegionsAroundNarrowBands` | `refined_filter_update_gain.cc:113-114` | `filters.py:PBFDKF._update_weights:571-574` | MATCH | ON | YES | LOW |
| R8 | Saturation gate | `refined_filter_update_gain.cc:96` | `filters.py:PBFDKF._update_weights:546` | MATCH | ON | YES | LOW |
| R9 | H_error init = 10000 on cold-start | `refined_filter_update_gain.cc:31, 44-46` | `filters.py:PBFDKF.H_error_per_bin init via aec3_scale.H_ERROR_INIT_FLOAT` | MATCH | ON | YES | LOW |
| R10 | Startup partition-sum X² hybrid | `subtractor.cc:196-213` (fixed partition-sum) | `filters.py:PBFDKF._update_weights_aec3:757` startup fallback to single-partition | APPROX | ON | YES | LOW |

**R1 note**: `use_partition_summed_x2_for_h_error_gain` defaults to `True` in `config.py:278`.
All three variants (A, B, D) inherit this from BALANCED preset. Because A and D share the same
X² source, the partition-sum X² is NOT the cause of D's DT regression vs A.

**R2 (instantaneous E2) gap**: AEC3 uses current-block E2_refined in the mu denominator
(instantaneous, no smoothing). Python default uses `_error_psd` (0.95 EMA of `|error_spec|²`,
~200 ms time constant). During re-convergence with stale W, the smoothed `_error_psd` reflects
pre-event error levels (post-convergence, near-zero). The mu denominator is then:
`0.5 × H_error × X² + n_part × _error_psd_stale`. With `_error_psd_stale ≈ 0`, the denominator
collapses to `0.5 × 10000 × X² = 5000·X²` — mu = `10000 / (5000·X²) = 2/X²`, 2× AEC3's
instantaneous value. Over-aggressive update during the first ~200 ms post-reset.

### 1.6 coarse_filter_update_gain.cc (shadow filter)

| # | AEC3 Node | AEC3 Ref | Python Location | Status | Default | D Enabled | Risk |
|---|-----------|----------|-----------------|--------|---------|-----------|------|
| C1 | X² partition-summed in mu denom (`rate / X²`) | `coarse_filter_update_gain.cc:64-72` | `filters.py:PBFDAF._update_weights` behind `_use_partition_summed_x2_for_shadow_mu` | IMPL-OFF | OFF | **NO** | MED |
| C2 | Noise gate: X² < noise_gate → mu=0 | `coarse_filter_update_gain.cc:66-71` | `filters.py:PBFDAF._update_weights` behind `_use_aec3_noise_gate_for_shadow` | IMPL-OFF | OFF | **NO** | MED |
| C3 | Poor excitation + call counter gate | `coarse_filter_update_gain.cc:51-61` | `filters.py:PBFDAF._update_weights` behind `_use_poor_excitation_gate_for_shadow` | IMPL-OFF | OFF | **NO** | MED |
| C4 | Narrowband mask | `coarse_filter_update_gain.cc:75` | `filters.py:PBFDAF._update_weights` behind `_use_narrowband_mask_for_shadow` | IMPL-OFF | OFF | **NO** | LOW |
| C5 | Saturation gate | `coarse_filter_update_gain.cc:56-57` | `filters.py:PBFDAF._update_weights` behind `_use_saturation_gate_for_shadow` | IMPL-OFF | OFF | **NO** | LOW |
| C6 | Coarse `HandleEchoPathChange` — `ZeroFilter(current,max)` (same semantics as S1/S6 = no-op when current=max) | `subtractor.cc:168-169` | `filters.py:PBFDAF.handle_echo_path_change(zero_filter=False)` | **APPROX** | N/A | **NO** | MED |
| C7 | Fixed-gain LMS (`rate / X²`, no persistent H_error) | `coarse_filter_update_gain.cc:68` | Python shadow uses PBFDKF Kalman (heavier than AEC3 coarse) | DIV | N/A | N/A | MED |

**C6 + C1-C5 interaction**: Shadow (coarse) filter in Variant D has W=stale and no AEC3
protection gates (C1–C5 all OFF). The shadow flags are NOT irrelevant: they affect coarse
output quality, `coarse_filter_converged` / `any_filter_converged` signals consumed by
TransparentMode and FilteringQualityAnalyzer, and — critically — the UseRefinedOutput
fallback (E1/G.A) which switches to coarse output when refined diverges. A poorly-converged
shadow (no C1–C5) degrades the quality of the coarse fallback path, making UseRefinedOutput
less effective even when enabled.

However, enabling C1–C5 before confirming that UseRefinedOutput (E1/Variant F) has a
meaningful coarse-fallback surface is premature. **Decision: trace Variant F first; after
confirming coarse fallback fires meaningfully, evaluate whether C1–C5 affect the fallback
quality.**

---

## 2. Suspected gap deep-dive (per user request)

### G.A — Does D enable `use_refined_output_selection_for_linear_path`?

**No.** Confirmed at `config.py:459`:
```python
use_refined_output_selection_for_linear_path: bool = False
```
`_cfg_d()` in `v3_21_a1_d_12case.py` does not set this flag. **D runs only the refined
(main) filter path in all frames.** During re-convergence, when the refined filter diverges
(high e2_refined), AEC3 would switch to the coarse output. Variant D cannot make this switch.

### G.B — Does D enable `use_coarse_e2_time_domain_parity`?

**No.** Confirmed at `config.py:482`:
```python
use_coarse_e2_time_domain_parity: bool = False
```
The default Parseval approximation (`orchestrator.py:3938-3944`) is used for `_e2_coarse`.
The `coarse_filter_converged` gate at line 3948 evaluates this approximation. Since both A
and D use the same Parseval path, this flag is NOT a differentiator between A and D behavior.
Risk: LOW for A vs D comparison.

### G.C — Does D enable shadow coarse protection flags?

**No — all five OFF.** Confirmed from `v3_21_a1_d_12case.py:_cfg_d()` which uses
`AecConfig.from_preset(AecPreset.BALANCED, ...)` without setting any shadow flags:
```
use_partition_summed_x2_for_shadow_mu=False  (config.py:498)
use_aec3_noise_gate_for_shadow=False         (config.py:504)
use_poor_excitation_gate_for_shadow=False    (config.py:510)
use_narrowband_mask_for_shadow=False         (config.py:517)
use_saturation_gate_for_shadow=False         (config.py:522)
```
These flags are NOT irrelevant (see §1.6 C1–C5 note). They affect coarse output quality,
`coarse_filter_converged` / `any_filter_converged` signals, and — once UseRefinedOutput is
enabled in Variant F — the quality of the coarse fallback path. A shadow filter without
noise gate or poor-excitation gate diverges more during re-convergence, making the
UseRefinedOutput coarse fallback less effective even when the predicate fires.

**Disposition**: Do NOT enable C1–C5 in Variant F. Trace Variant F Gate 0 first to
confirm coarse fallback fires and characterise how often cond1/cond2 select coarse. Then
evaluate whether C1–C5 improve fallback quality in a subsequent variant.

### G.D — Does D enable `usable_linear_trusted_external_delay_only`?

**No.** Confirmed at `config.py:586`:
```python
usable_linear_trusted_external_delay_only: bool = False
```
With this OFF, `_current_delay >= 0` immediately provides `ext_delay` to the
`FilteringQualityAnalyzer` gate-3 (`orchestrator.py:4022`). Gate-3 latches True as
soon as the legacy delay tracker has any estimate, regardless of filter convergence.

**Risk: HIGH**. AEC3 requires a high-quality matched-filter delay or `convergence_seen`.
Python's permissive external_delay shortcut causes `usable_linear=True` to fire sooner
after delay_first, compressing the Y-path protection window and extending the E-path
damage window. Combined with the W=non-zero gap, this worsens DT_mvmt damage.

### G.E — Does D use `use_current_e2_refined_in_h_error_denominator`?

**No.** Confirmed at `config.py:401`:
```python
use_current_e2_refined_in_h_error_denominator: bool = False
```
Python uses smoothed `_error_psd` (0.95 EMA). During the first ~200 ms post-reset,
`_error_psd` reflects the pre-event (post-convergence, near-zero) error level, making the
mu denominator underestimate the actual error. This inflates mu by ~2× during the critical
first 200 ms of re-convergence (see R2 analysis above).

### G.F — Does D have a startup X² hybrid divergence?

`use_partition_summed_x2_for_h_error_gain=True` (BALANCED default, `config.py:278`).
Python's startup hybrid at `filters.py:PBFDKF._update_weights_aec3:757`:
```python
if self._call_counter > self._partition_sum_x2_startup_hops:   # 500 hops = 5 s
    X2 = partition_sum
else:
    X2 = single_partition
```
AEC3 always uses partition-summed X² (`SpectralSum` in `render_buffer`). The Python hybrid
uses single-partition X² for the first 500 hops (5 s), then switches to partition-sum.

After `handle_echo_path_change()`, `_call_counter=0` is reset (line 193 in filters.py).
So for the first 500 hops after delay_first, the mu denominator uses single-partition X²
(smaller X², larger mu). This produces faster initial adaptation but diverges from AEC3
behavior during the same critical window where W is stale and H_error=10000.

---

## 3. Per-frame first-divergence analysis

Data source: Gate 0 trace summary (`docs/v3_21_a1_combo_gate0_trace_verdict.md`) + code
trace. Per-frame CSV not available; columns marked `[TRACE]` come from Gate 0 summary data;
columns marked `[INFERRED]` are derived from code analysis + known invariants.

### 3.1 xFk7igecuke0R5JMfREyDg — D−A = −0.706 (worst case)

delay_first = frame 104 (1040 ms from start)

| Frame window | delay_event | initial_state | usable_linear | external_delay source | refined_conv / coarse_conv | UseRefinedOutput | E/Y decision | Estimated E-path integrity | First Python↔AEC3 divergence |
|---|---|---|---|---|---|---|---|---|---|
| 0–103 | none | False | True [INFERRED] | delay tracker active | — | N/A (E1 OFF) | E [TRACE] | GOOD (converged) | — |
| **104** | delay_first | **True** [TRACE] | **False** [TRACE] | present (permissive) | False / False [INFERRED] | N/A (E1 OFF) | **Y** [TRACE] | — (Y-path used) | **DIVERGENCE START: both retain stale W; Python has no coarse fallback (UseRefinedOutput absent)** |
| 105–143 | none | True | False [TRACE] | present | False / False | N/A | Y [TRACE] | — (Y-path, no filter output used) | W adapting stale→new, H_error=10000 |
| **~144** | none | True | **True** [INFERRED] | present (gate-3 already latched) | False / False [INFERRED] | N/A | **E** [INFERRED] | **CORRUPTED** (stale W + large gain) | **Python uses E; AEC3 would extend Y or use coarse** |
| 144–394 | none | True | True | present | False→True [INFERRED] | N/A | E | Improving but corrupted during 144–250 approx | Stale W + H_error=10000 → wrong-cancellation |
| **395** | none | **False** (transition) | True | present | True [INFERRED] | N/A | E | Better (W re-converged somewhat) | Python: SG exits initial state; AEC3: also ExitInitialState + config switch |

Y_total in Gate 1 trace = 193 (Gate 0 window = 40). Re-fire events (EPV/shadow_rise) caused
additional Y-path frames outside the initial_state window.

**First divergence**: Frame 104 — delay change fires. Both AEC3 and Python retain stale W;
partition count difference (Python=13 vs AEC3 shrinks to 12) is MED risk. The key AEC3/Python
divergence is NOT W zeroing but the absence of UseRefinedOutput: as W adapts from stale, the
refined filter produces high e2_refined during re-convergence — AEC3 would switch to coarse
output; Python D cannot.
**Second divergence (gate-3 compression)**: ~Frame 144 — Python's permissive gate-3 latches
`usable_linear=True` via `ext_delay` shortly after delay_first. AEC3 may wait until
`convergence_seen`. Python compresses Y-path protection and extends stale-W refined E-path.

### 3.2 wVYSGVTTakih9twI4xlDWQ — D−A = −0.110

delay_first = frame 75 (750 ms)

| Frame window | delay_event | initial_state | usable_linear | E/Y | First divergence |
|---|---|---|---|---|---|
| 0–74 | none | False | True | E | — |
| **75** | delay_first | True | False | **Y** | **both AEC3 and Python retain stale W; Python keeps 13 partitions vs AEC3 shrinks to 12** |
| 76–134 | none | True | False | Y | W adapting stale→new |
| **~135** | none | True | **True** | **E** | Python E-path with stale W begins |
| 135–453 | none | True | True | E | Recovery; B−A=−0.269 so filter had significant re-convergence burden |
| **454** | none | False (transition) | True | E | SG exits initial; D−B=+0.159 (partial recovery by Y-path coverage) |

Y_total in Gate 1 = 135 (Gate 0 window = 60, initial_state duration 379 frames).

### 3.3 ZJYUt0O0AEKSQ9LJ8z7t0A — D−A = −0.361

delay_first = frame 141 (1410 ms)

| Frame window | delay_event | initial_state | usable_linear | E/Y | Notes |
|---|---|---|---|---|---|
| 0–140 | none | False | True | E | Converged |
| **141** | delay_first | True | False | **Y** | both retain stale W; Python has no coarse fallback |
| 142–224 | none | True | False | Y | W adapting stale (84 frames) |
| **~225** | none | True | **True** | **E** | Stale W E-path begins (B−A=−0.072 → this B regression is smallest of 3) |
| 225–650 | EPV=[465,897,1153] SR=[3772] | True | Mixed | E/Y | EPV events force Y-path re-entries (explains higher Y_total=225 vs Gate 0=84) |
| **651** | none | False | True | E | Transition |

D−B = −0.289 (ZJYUt0O0 is the ONLY case where D is WORSE than B): Y-path fires
(Y_total=225) but during the EPV/SR re-entry windows, the additional Y-path frames may
interfere with already-re-converged filter output, causing regression vs B (which has no
Y-path interference). This asymmetry demonstrates that the Y-path companion is not monotone
across cases.

---

## 4. Gaps confirmed not present in Variant D (checklist completion)

Per user request, explicit confirmation of each gap item:

| Item | Suspected Gap | Confirmed Absent in D? | Impact on D failure verdict |
|---|---|---|---|
| G.A | `use_refined_output_selection_for_linear_path` (UseRefinedOutput) | **YES, absent** | HIGH — coarse fallback missing during refined divergence |
| G.B | `use_coarse_e2_time_domain_parity` | YES, absent — but NOT a D vs A differentiator (both use Parseval) | NEUTRAL for A vs D comparison |
| G.C | Shadow coarse protection flags (C1-C5) | **YES, all five absent** | MED — affects coarse fallback quality once E1 enabled; NOT irrelevant (see §1.6) |
| G.D | `usable_linear_trusted_external_delay_only` | **YES, absent** — permissive shortcut active | HIGH — compresses Y-path window, extends stale-W E-path |
| G.E | `use_partition_summed_x2_for_h_error_gain` | **PRESENT** (BALANCED default=True, shared by A/B/D) | NEUTRAL (not a differentiator) |
| S1 | W=0 on delay change — **RETRACTED** (see §1.2 S1) | AEC3 also retains stale W (`ZeroFilter(current,max)` = no-op); Python `zero_filter=False` matches AEC3 | NOT a gap. Python W.fill(0) = non-AEC3 ablation |
| S3/S4 | Partition resize 13→12 on delay change (AEC3 shrinks, Python fixed) | **YES, absent** | MED — minor difference in active filter reach |
| — | Instantaneous E2 in mu denominator (`use_current_e2_refined_in_h_error_denominator`) | **YES, absent** | MED — inflates mu ~2× during first 200 ms post-reset |
| — | ExitInitialState filter config switch (partition resize on exit) | **YES, absent** (SuppressionGain-only) | MED — minor initial-phase asymmetry |

---

## 5. Verdict classification

**Verdict: B — Variant D failure is INCONCLUSIVE because remaining AEC3 linear path gaps exist.**

Specific statement: Gate 1 FAIL on all 3 DT_mvmt cases for Variant D (full AEC3 composition)
is NOT attributable solely to PBFDKF structural incompatibility. Two AEC3 behaviors with
direct mechanistic connection to DT_mvmt damage are absent:

**⚠ Prior Gap 1 (W=0, CRITICAL) — RETRACTED (2026-05-25)**:
`AdaptiveFirFilter::HandleEchoPathChange()` = `ZeroFilter(current_size, max_size)`.
When `current_size = max_size = 13`: range [13, 13) = empty → no zeroing.
`SetSizePartitions(12, true)` → `ZeroFilter(13, 12)` = empty range → no zeroing.
AEC3 retains stale W in partitions 0–11, drops partition 12 (inactive but not zeroed).
Python `zero_filter=False` matches AEC3: both retain stale W.
**Python W.fill(0) = non-AEC3 ablation — NOT a v3.21 alignment candidate.**

**Gap 1 (UseRefinedOutput coarse fallback, HIGH)**:
- AEC3: when refined diverges (high e2_refined during stale-W re-convergence), `UseRefinedOutput`
  predicate fires → output switches to coarse (shadow) filter
- Python Variant D: refined output always used; `use_refined_output_selection_for_linear_path=False`
- Mechanism: AEC3 has a rescue path via coarse output during refined divergence; Python D
  has none — all re-convergence damage goes directly to the E-path output

**Gap 2 (permissive usable_linear gate, HIGH)**:
- AEC3 gate-3: `external_delay` requires matched-filter quality OR `convergence_seen`
- Python: `usable_linear_trusted_external_delay_only=False` → gate-3 latches True as soon
  as `_current_delay >= 0` (any active estimate), regardless of filter convergence
- Mechanism: usable_linear=True fires 40–84 frames after delay_first; AEC3 may wait until
  genuine convergence evidence → Python compresses Y-path, extends E-path damage window

**Secondary differences (MED, not primary cause)**:
- Partition resize 13→12 on delay (S3/S4): minor effect on filter reach
- Shadow flags C1–C5 all OFF: degrades coarse output quality, relevant once E1 (UseRefinedOutput) enabled

**Consequences**:
1. Category A classification (PBFDKF structurally incompatible with full AEC3 delay-change
   composition) remains UNVALIDATED. The 2-gap combination (no UseRefinedOutput + permissive
   gate) makes Variant D failure expected from a parity perspective.

2. R1/R3 functional adaptations (NE gate ERLE reset / skip ERLE reset on delay-change)
   are NOT authorized. The re-convergence environment (no coarse fallback, permissive
   usable_linear) is not AEC3-equivalent. Fixing the environment first is prerequisite.

---

## 6. Next actions (authorized 2026-05-25 by user)

### PRIMARY: Variant F — D + UseRefinedOutput Gate 0 trace

**Config**: D (`use_full_delay_change_chain=True` + `use_linear_filter_output_selection_for_final_output=True`)
plus `use_refined_output_selection_for_linear_path=True`. No W.fill(0), no R1/R3, no 800-case.

**Rationale**: UseRefinedOutput (E1/G.A) is the primary gap confirmed in this audit. It is
the AEC3 rescue path when refined diverges during re-convergence. Variant D has stale W but
AEC3 does too — the key protection AEC3 has that D lacks is the coarse fallback.

**Prior warning** (`docs/v3_21_8_12case_verdict.md`): UseRefinedOutput alone (v3.21.8,
no ULO) caused catastrophic DT_static regression — 37% of XRTnTUjU frames fired
`cond2_refined_diverged` → shadow NLMS selected in stable-DT frames. Variant F combines
with ULO which provides Y-path bridge when `usable_linear=False`; interaction with D is
different from v3.21.8 which had no ULO companion.

**Step 1 — Gate 0 trace** (read-only, before any AECMOS):

Required trace fields per frame for all 12 cohort cases (instrument in `_aec3_post` trace,
default-OFF, byte-equal guard):
```
use_refined_output_fire         # bool: UseRefinedOutput predicate fired this frame
cond1_selected                  # bool: cond1 (e2_coarse < 0.9*e2_refined, normal) selected coarse
cond2_selected                  # bool: cond2 (refined diverged, e2_coarse < e2_refined) selected coarse
selected_path                   # str: 'refined' | 'coarse'
e2_refined                      # float: refined filter error power
e2_coarse                       # float: coarse filter error power
y2                              # float: capture power
s2_refined                      # float: refined filter echo power
s2_coarse                       # float: coarse filter echo power
usable_linear                   # bool: usable_linear_estimate (existing field)
initial_state_active            # bool: AecState.initial_state_active (existing field)
```

**Gate 0 pass criteria**:
- G0.1: UseRefinedOutput fires on ≥ 1 DT_mvmt failing-window frame (not zero-fire)
- G0.2: cond2 (refined diverged) fires during delay_first re-convergence window (the intended rescue case)
- G0.3: selected_path='coarse' is present in DT_mvmt failing windows
- G0.4: DT_static stable-state cond2 fire rate is LOW (< 15% of frames) — prevents v3.21.8 regression
  (v3.21.8 had 37% on XRTnTUjU in stable frames; this must be verified with ULO companion)

**If Gate 0 passes**: run 12-case AECMOS (Variant F vs A, F vs B, F vs D).
**If Gate 0 fails** (cond2 zero-fire or cond2 fires in stable-DT at 37%+ rate): diagnose
before proceeding. Cond2 zero-fire = coarse output never needed (shadow converges faster
than expected). High cond2 in stable = v3.21.8 problem not resolved by ULO companion.

---

### PARALLEL: Gate 3 read-only trace (ext_delay shortcut vs trusted-only)

**Goal**: Determine whether `usable_linear_trusted_external_delay_only=False` (current default)
causes `usable_linear=True` to fire too early in the DT_mvmt regression frames, and whether
switching to `trusted_only=True` would extend the Y-path protection window.

**Method** (read-only, no production change):
1. On the 3 DT_mvmt failing cases (xFk7, wVYSGVTT, ZJYUt0O0), trace:
   - `ext_delay_available`: True when `_current_delay >= 0` (current permissive condition)
   - `convergence_seen`: True when filter has genuinely converged (gate-3 strict condition)
   - Frame offset from delay_first when each latches True
2. Compute per-case: `Δframes = convergence_seen_frame - ext_delay_available_frame`
   — this is how many extra Y-path frames AEC3-strict-gate would provide vs current
3. Cross-check: `Y_total` in Gate 1 trace (193/135/225) vs gate-3 latch offset — if
   `Δframes > 0`, the trusted-only gate would extend Y-path coverage by that amount

**This is trace-only.** No config change, no flag flip. Result feeds the decision on whether
`usable_linear_trusted_external_delay_only=True` should be in a Variant G (after Variant F
Gate 0/12-case completes).

---

### DEPRECATED: Variant E (W.fill(0)) — reclassified as non-AEC3 ablation

**Retracted as v3.21 alignment candidate.** AEC3's `HandleEchoPathChange()` =
`ZeroFilter(current=13, max=13)` = no-op. `SetSizePartitions(12, true)` also = no-op for
active partitions. AEC3 retains stale W. Python `zero_filter=False` matches AEC3 behavior.
Python W.fill(0) would be MORE aggressive than AEC3, not parity.

If W.fill(0) is tested in the future, it must be labelled as a non-AEC3 ablation hypothesis
under a separate arc (e.g., v3.22 or post-v3.21 investigation). It cannot be shipped as
v3.21 AEC3 alignment work.

---

## 7. Instrumentation gap (for full per-frame divergence table)

The following fields are NOT in the current trace output and would be needed to produce a
complete per-frame first-divergence table for the 3 DT_mvmt cases:

| Field | Source | Default-OFF addition required |
|---|---|---|
| `refined_filter_converged` (separate from OR) | `_refined_conv` already computed in `_aec3_post:3922` | Add to `_diag` (1 line) |
| `coarse_filter_converged` (separate from OR) | `_coarse_conv` already computed in `_aec3_post:3948` | Add to `_diag` (1 line) |
| `usable_linear_gate3_reason` (external_delay vs convergence_seen) | `filter_quality.py:_overall_usable` logic | Requires minor exposure |
| `UseRefinedOutput decision` (would-be: refined vs coarse) | Can be computed from E1 predicate even when E1 is OFF (informational) | Optional informational trace |
| `h_error_mean` | Already in `_diag` (lines 4101-4106) | Already available |
| `W_norm_main` | `np.linalg.norm(filter.W)` | 1 line in `_diag` |

All of these are default-OFF additions. They do NOT change production behavior. A byte-equal
guard run confirming no output change before and after adding the fields is the only
prerequisite.

---

## 8. Arc disposition

A.1 remains **OPEN**.

| Condition | Status |
|---|---|
| Gate 1 12-case AECMOS complete (D vs A) | DONE (2026-05-25) — FAIL |
| Full AEC3 linear path audit complete | **DONE (2026-05-25, this doc, corrected)** |
| W=0 (ZeroFilter) gap — RETRACTED | AEC3 also retains stale W; Python `zero_filter=False` matches |
| UseRefinedOutput gap confirmed absent in D | YES — primary gap |
| Permissive usable_linear gate confirmed | YES — secondary gap |
| Category A classification valid | **NOT VALID** — primary gaps (E1 + G.D) not yet tested |
| R1/R3 functional adaptation authorized | NO — prohibited until Category A proven |
| Variant E (W.fill(0)) | **DEPRECATED** — non-AEC3 ablation, not v3.21 alignment candidate |
| Variant F Gate 0 trace | **DONE (2026-05-25)** — PASS; see `docs/v3_21_a1_f_gate0_verdict.md` |
| Variant F Gate 1 12-case AECMOS | **DONE (2026-05-25)** — FAIL; wVYSGVTT +0.117 rescued; xFk7 −0.692 / ZJYUt0O0 −0.478 |
| Gate 3 ext_delay shortcut trace | **DONE (2026-05-25)** — Δframes=23/56/71; shortcut fires at delay_first |
| C1–C5 shadow gates | Coarse fallback CONFIRMED active; shadow quality limiting; next candidate |
| Variant G (trusted gate) | Δframes confirmed; next candidate after C1-C5 or combined |
| Category A classification | **STILL NOT VALID** — wVYSGVTT +0.117 rescue proves PBFDKF not structurally incompatible |
| Next steps | Require user authorization: Variant H (F+C1-C5) and/or Variant G |
| Variant F 12-case AECMOS | Pending Gate 0 pass |

**Hard rules (unchanged)**:
- No 800-case without explicit user authorization
- No v3.22 framing of A.1
- No R1/R3 without explicit user authorization
- No new variant tests without explicit user authorization
