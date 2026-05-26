# v3.21 AEC3 Alignment Closure Ledger

Generated: 2026-05-26 — M0–M4 12-case complete  
Production baseline: v3.21.6 (`docs/bench/v3_21_6_baseline/result.md`)  
Ship gate: Gate 0 PASS → 12-case M0–M4 FAIL (URO all variants) → no 800-case run  
**Status**: v3.21 ships unchanged at v3.21.6. One OPEN item remains: `use_full_delay_change_chain`.

## Legend

| Verdict | Meaning |
|---|---|
| **SHIPPED** | Default-ON; included in v3.21.x production; 800-case verified |
| **NOSHIP** | Tested; failed ship criteria; default-OFF retained as substrate |
| **SUBSTRATE** | Implemented; not yet tested OR tested at insufficient scope; default-OFF |
| **ZERO-FIRE** | Fires ≤ 2 % of frames on 800-case cohort; no behavioral impact |
| **NOOP** | Correct AEC3 port but threshold-bound no-op in practice on our pipeline |
| **BROKEN** | Wiring defect; cannot be safely enabled; dead code |
| **OPEN** | Pending test in this release matrix (M0–M4) |

---

## Part A — Refined filter update gain (PBFDKF main)

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `use_partition_summed_x2_for_h_error_gain` | **False** | `SpectralSum` Σ_p X²[k] → mu denom / noise-gate / H_error decay | **NOSHIP** | 800-case Cat C: ~26 DT cases Δdeg ≤ −0.5 (worst nVUnxqHLr −1.475); FS echo −0.04–0.06 acceptable. Default reverted to False for v3.21 closure; requires fresh 12-case + 800-case. Ref: `docs/v3_21_7_800case_verdict.md` |
| `use_per_bin_h_error_refresh` | False | Per-bin `H_error` selector (refined_filter_update_gain.cc:128-138) | **NOSHIP** | 800-case cohort tail: 82 echo + 54 deg per-case regressions > 0.05 dB; worst Δdeg −0.437 LHsrJBRGn. Ref: v3.21.1 / v3.21.4 U4.A retest |
| `use_current_e2_refined_in_h_error_denominator` | False | Current-block `E²_refined` (vs EMA-smoothed) in mu denominator | **SUBSTRATE** | 12-case matrix not completed; companion to `use_partition_summed_x2_for_h_error_gain`; both must be tested together. Ref: `docs/v3_21_12_refined_filter_update_gain_input_parity_plan.md` |
| `use_aec3_handle_echo_path_change` | **True** | H_error=10000 + counter reset on EPC (refined_filter_update_gain.cc:53-68) | **SHIPPED** | Default-ON since Phase D. Load-bearing: jtYTdZm3 +0.155 dB. |
| `use_aec3_zero_filter_on_epc` | False | W=0 on delay-change event (AdaptiveFirFilter::ZeroFilter) | **NOSHIP** | 12-case: nVUnxqHLr −0.510, jtYTdZm −0.300 under EPC classification bug (Phase E). W=0 is destructive on PathChangeRegimeHandler cohort tail. |
| `use_aec3_epc_classification` | False | Correct delay-change vs gain-change EchoPathVariability dispatch | **SUBSTRATE** | Phase F prerequisite for correct `use_aec3_zero_filter_on_epc`; not yet 12-case tested. |
| `skip_aec3_gain_change_dispatch_at_epv_shadow_rise` | False | Decouple AecState gain_change from EPV+shadow_rise sites | **SUBSTRATE** | Phase G isolation experiment. `erle_reset_signal` identical in A_off vs F_off → AecState side decorative on synthetic PBFDKF events. Not a ship candidate; audit-only. |

---

## Part B — UseRefinedOutput + FormLinearFilterOutput

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `use_refined_output_selection_for_linear_path` | False | Per-frame coarse-vs-refined selection (echo_remover.cc:112-147) | **NOSHIP** | M1 12-case (2026-05-26): xFk7igec DT_mvmt Δdeg=−1.009; 9xjhi FS_static Δecho=−1.517; DT_mvmt bucket mean Δdeg=−0.378; all G1/G2/G3 fail. Root: URO fires cond2 (refined-diverged) on un-converged shadow → coarse selected → coarse under-converged → DT suppressed / FS echo uncancelled. Shadow-converged precondition required (v3.22 scope). Ref: `docs/v3_21_uro_crossfade_12case_verdict.md` |
| `form_linear_filter_crossfade_enabled` | False | 30-sample SignalTransition ramp + WindowedPaddedFft FFT memory (signal_transition.cc + echo_remover.cc:134) | **SUBSTRATE** | Cannot ship independently (no-op without URO). Semantics corrected 2026-05-26: from_time = current block × prev selector; e_old_ = _form_prev_output_time. Mixed M1→M2 delta: jtYTdZm3 +0.184 (crossfade smooths transition correctly), wVYSGVTT −0.158, overall swamped by URO failure. Retain as substrate for v3.22 URO + crossfade combined design. |
| `use_coarse_e2_time_domain_parity` | False | Coarse e² block-window consistency (subtractor_output.cc:38-48) | **NOOP** | 12-case: 0/24 file diffs ON vs baseline. Mechanism correct AEC3 port but threshold-bound (≤ 5% AND y² > 50²·hop) — stress frames fail threshold under both old and new formula; normal frames lock on refined_conv first. Ref: `docs/v3_21_9_12case_verdict.md` |
| `coarse_filter_converged_relaxed_enabled` | False | `coarse_filter_converged_relaxed` signal for TransparentMode HMM | **SUBSTRATE** | No consumer: LegacyTransparentMode ignores relaxed signal; `transparent_mode_enabled=False` by default. Cannot affect behavior until both HMM variant AND transparent_mode_enabled are active. Audit-only. |
| `aec3_misadj_trace_enabled` | False | AEC3 `inv_misadjustment` trace (e²/y² ratio, no behaviour change) | **SUBSTRATE** | Audit-only; zero behaviour change. v3.23 G.0 A.1: legacy estimator fired 0/20188 frames. |
| `use_aec3_filter_misadjustment_parity` | False | AEC3 direction parity: e²/y² > 10 → shrink W (vs legacy grow-W) | **SUBSTRATE** | Not yet 12-case tested. Different failure mode from legacy estimator. |

---

## Part C — Final output selection (UseLinearFilterOutput)

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `use_linear_filter_output_selection_for_final_output` | False | Y-vs-E final output (echo_remover.cc:475): gain applied to Y when `usable_linear=False` | **SUBSTRATE** | PAUSED after v3.21.13 BE-combo showed mechanistic trap: B's cleaner filter latches `usable_linear` early on XRTnTUjU → use_capture collapses 59.4%→7.4% → Y-fallback neutralized. Standalone E verdict: DT_static −0.335, DT_mvmt +0.298, FS_static −0.138. Not in this 12-case matrix. Ref: `docs/v3_21_13_use_linear_filter_output_verdict.md` |

---

## Part D — Shadow (coarse) filter protection — C1–C5

All five flags are AEC3 `CoarseFilterUpdateGain::Compute` ports for the PBFDAF shadow filter.

| Flag | C# | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|---|
| `use_partition_summed_x2_for_shadow_mu` | C1 | False | Shadow mu denom: Σ_p X²[k] | **NOSHIP** | 12-case: wVYSGVTTakih9twI4xlDWQ −0.148 alone; −0.445 in ALL combo. Ref: `docs/v3_21_14_verdict.md` |
| `use_aec3_noise_gate_for_shadow` | C2 | False | Shadow noise gate: mu[k]=0 when X²[k] < NOISE_GATE | **NOSHIP** | 800-case (A.2+A.3 combo): 6 FS catastrophic regressions (worst wlAXM0iD −1.054); high-variance on borderline X². Individual C2-only split not completed. Ref: `docs/v3_21_15_verdict.md` |
| `use_poor_excitation_gate_for_shadow` | C3 | False | Shadow startup gate: poor_signal_excitation + call_counter | **NOSHIP** | 800-case (A.2+A.3 combo): same catastrophic cluster as C2. Individual C3-only split not completed. |
| `use_narrowband_mask_for_shadow` | C4 | False | Shadow narrowband mask: zero mu on tonal bins | **ZERO-FIRE** | v3.21.14 A.4: 0/90 fire rate on DT_mvmt cohort. No behavioral impact on standard cohort. |
| `use_saturation_gate_for_shadow` | C5 | False | Shadow saturation gate: return early on saturated_capture | **ZERO-FIRE** | v3.21.14 A.5: 2/90 fire rate on DT_mvmt cohort. Effectively zero-impact. |

---

## Part E — Q1 delay-change chain + AEC3 SetConfig parity

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `use_full_delay_change_chain` | False | Full AEC3 delay-change cascade: H_error + poor_excitation + AecState._full_reset + suppression_gain.set_initial_state on delay_first/delay_shift | **OPEN** | Sole remaining open parity item per v3.21.x alignment roadmap (2026-05-25). Gate 0 trace verdict at `docs/v3_21_a1_gate0_trace_verdict.md`. Not in M0–M4 crossfade matrix (independent arc; schedule after M4 if 12-case passes). |
| `use_q1_true_delay_transient_leakage` | False | AEC3-like transient leakage burst on real delay events | **NOSHIP** | All variants exhausted (Phase 2.1 R1 through Phase 3.1). No consistent win without interaction with EPV/SR sites. Ref: `docs/v3_21_q1_*` series |
| `use_q1_terminate_on_non_delay_epc` | False | Q1 re-entry guard: terminate on EPV/shadow_rise during Q1 | **NOSHIP** | Companion to `use_q1_true_delay_transient_leakage`; closed with Q1 arc. |

---

## Part F — usable_linear gate-3 ablation knobs

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `usable_linear_convergence_hops_required` | 0 (int) | Graduated convergence latch: counter ≥ N vs binary latch | **SUBSTRATE** | Not independently tested on 800-case. Gate-3 ablation substrate for no-clean-convergence stress audit. |
| `usable_linear_require_filter_analyzer_consistent` | False | AND-gate gate-3 with FilterAnalyzer consistent signal | **SUBSTRATE** | Requires `filter_analyzer_enabled=True` (already default-ON). Not tested on 800-case. |
| `usable_linear_disable_external_delay_shortcut` | False | Drop ext_delay branch: only convergence satisfies gate 3 | **SUBSTRATE** | Most-aggressive variant; not tested on 800-case. |
| `usable_linear_trusted_external_delay_only` | False | Restrict ext_delay to fixed/is_solid sources only | **SUBSTRATE** | M3 12-case (2026-05-26): M3 vs M2 Δ = 0.000 on all 12 cases (zero effect on this cohort). Semantics gap is real but this flag cannot be evaluated while URO (M1) fails catastrophically. Retain for v3.22 combined design. |

---

## Part G — SaturationDetector + saturation inputs

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `saturation_subtractor_inputs_enabled` | False | Feed s_refined + s_coarse max-abs to AecState.update() (echo estimates) | **SUBSTRATE** | M4 12-case (2026-05-26): M4 vs M3 Δ = 0.000 on all 12 cases (zero-fire on this cohort, consistent with v3.21.14 A.5 audit). Cannot be evaluated while URO (M1) fails. Retain as substrate. |

---

## Part H — AecState substrate (FilterAnalyzer / TransparentMode / SDE / Stationarity)

| Flag | Default | AEC3 mechanism | Verdict | Evidence |
|---|---|---|---|---|
| `filter_analyzer_enabled` | **True** | FilterAnalyzer port: IFFT partitions → 3-tap HPF → peak detection → ConsistentFilterDetector | **SHIPPED** | v3.21.6 P1: FS_static +0.059 / FS_movement +0.036; DT ±0.01. |
| `transparent_mode_enabled` | False | AEC3 LegacyTransparentMode (2-state HMM: good_quality / poor_quality) | **SUBSTRATE** | v3.21.6 P2: ported + block-unit constants corrected. ALL-ON cycle shows mis-fire on stress cases (TransparentMode gates usable_linear → over-suppression). Default-OFF preserved. |
| `filter_quality_enabled` | False | FilteringQualityAnalyzer via old orchestrator stub | **BROKEN** | Dead v3.18 wiring: `modules/filter_quality.py` signature mismatch with `state/filter_quality.py`. Correct port lives in AecState composition (aec_state.py:103-111). This flag is a no-op and cannot be safely enabled. |
| `aec3_post_stationarity_zero_enabled` | **True** | R²=0 on stationary bins when filter converged (EchoAudibility.use_stationarity_properties inverse) | **SHIPPED** | v3.21.5 Sprint B: OFF gives 60+ DT cohort-tail cases 1-2 dB formant attenuation (xQEUtY2 −0.602 worst). Load-bearing safety net compensating for incomplete AEC3 port. |
| `signal_dependent_erle_sections` | 0 (int) | SignalDependentErleEstimator: per-section per-subband ERLE refinement | **SUBSTRATE** | 0=OFF. Tested at sections=2 in v3.21.20 ALL-ON cycle as part of 8-flag combo; catastrophic stress regression observed but not isolated. AEC3 production ships sections=1 (degenerate). Not independently tested. |
| `coarse_filter_converged_relaxed_enabled` | False | audit-only: `coarse_filter_converged_relaxed` signal | **SUBSTRATE** | No consumer under LegacyTransparentMode (transparent_mode.cc:150 ignores it). Audit trace only. |

---

## Part I — Shipped fixed changes (no flag)

These v3.21.x changes are always-on and do not have a runtime flag:

| Change | Version | AEC3 mechanism | Status |
|---|---|---|---|
| `e2_y2_clamp_enabled=True` | v3.21.5 | AEC3 echo_remover.cc:495-501 E²=min(E²,Y²) clamp | **SHIPPED** |
| `filter_analyzer_enabled=True` | v3.21.6 P1 | FilterAnalyzer ConsistentFilterDetector | **SHIPPED** |
| `use_aec3_handle_echo_path_change=True` | Phase D | H_error reset on EPC | **SHIPPED** |
| Legacy ResFilter 9-stage chain retired | v3.21.0 | Replaced by AEC3-aligned _aec3_post | **SHIPPED** |
| AecState + ResidualEchoEstimator + SuppressionGain + CNG OLA | v3.21.0 | AEC3-aligned post-filter | **SHIPPED** |
| SubbandErleEstimator per-frame e²<0.5y² convergence gate | v3.21 | Replaces legacy 10-frame ERLE latch | **SHIPPED** |
| PBFDKF H_error += factor×ERL conditional refresh | v3.21.1 | refined_filter_update_gain.cc:128-138 | **SHIPPED** |

---

## Part J — 12-case test matrix (M0–M4)

Run: 2026-05-26. M0 byte-equal to plain BALANCED: **PASS**.

| Label | Flags vs M0 | Purpose | Gate result | Key evidence |
|---|---|---|---|---|
| M0 | all candidates OFF; `use_partition_summed_x2_for_h_error_gain=False` | v3.21.6-equivalent anchor | G5 PASS | byte-equal to plain BALANCED |
| M1 | `use_refined_output_selection_for_linear_path=True` | URO binary switch only | G1/G2/G3 FAIL | xFk7igec Δdeg=−1.009; 9xjhi Δecho=−1.517; DT_mvmt bucket mean=−0.378 |
| M2 | M1 + `form_linear_filter_crossfade_enabled=True` | URO + SignalTransition crossfade | G1/G2/G3 FAIL | xFk7igec Δdeg=−0.955; inherits M1 catastrophe; jtYTdZm3 +0.184 isolated win |
| M3 | M2 + `usable_linear_trusted_external_delay_only=True` | Trusted ext_delay gate | G1/G2/G3 FAIL | M3−M2 delta = 0 on all cases; trusted-delay gate has no effect on this cohort |
| M4 | M3 + `saturation_subtractor_inputs_enabled=True` | Full A/B/C/D substrate ON | G1/G2/G3 FAIL | M4−M3 delta = 0; saturation inputs zero-fire on this cohort |

**Not in matrix**: `use_linear_filter_output_selection_for_final_output` (PAUSED SUBSTRATE), C1–C5 shadow flags (NOSHIP/ZERO-FIRE), `use_full_delay_change_chain` (independent arc).

---

## Part K — Open items / deferred work

All Part B/F/G OPEN items resolved via M0–M4 12-case (2026-05-26). Remaining:

| Item | Status | Next action |
|---|---|---|
| `use_full_delay_change_chain` (A.1 delay-change chain) | OPEN | Schedule independently; Gate 0 trace verdict already exists. Must be done before v3.21 can be declared fully closed. |
| URO shadow-converged precondition | DEFERRED → v3.22 | Root cause of M1 failure. Shadow filter must converge to stable state before URO fires cond2. Requires v3.22 shadow design changes (not AEC3 parity — beyond-AEC3). |
| C2/C3 individual split (A.2-only / A.3-only 800-case) | DEFERRED | Not in v3.21 scope. |
| `use_partition_summed_x2_for_h_error_gain=True` (fresh test) | DEFERRED | Requires independent 12-case + 800-case; not in v3.21 closure. |
| v3.10.5 vs v3.21 difference analysis | DEFERRED | Arc A after v3.21 closure. |
| v3.22 beyond-AEC3 optimisation | DEFERRED | Arc B after v3.21 closure; requires independent authorisation. |

---

## Closure rule

v3.21 is closed when:
1. All "OPEN" items in this ledger are resolved (pass → SHIPPED, fail → NOSHIP/SUBSTRATE)
2. 800-case bench shows Pareto-positive or Pareto-neutral vs v3.21.6 baseline
3. 4-stress cases (nVUnxqHLr / XRTnTUjU / 9xjhi / MYrVxVEM) each Δdeg ≥ −0.10

**2026-05-26 state**: Items 2 and 3 are not triggered (no new ship candidate passes 12-case).
Item 1 has one remaining OPEN: `use_full_delay_change_chain`. All M0–M4 URO/crossfade candidates
resolve to NOSHIP/SUBSTRATE. v3.21 production = v3.21.6 (unchanged).
4. `__version__` bumped + CHANGELOG updated
