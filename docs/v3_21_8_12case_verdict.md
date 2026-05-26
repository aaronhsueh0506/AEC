# v3.21.8 — AEC3 UseRefinedOutput + FormLinearFilterOutput 12-case verdict (CLOSED)

**Date**: 2026-05-22

**Final verdict (2026-05-22)**: **Cat C BLOCKED-STRESS, not shippable.** v3.21.8 reframed as AEC3 **field-trial substrate** (not production-default parity). Code preserved with `use_refined_output_selection_for_linear_path: bool = False` default OFF. Do NOT proceed to 800-case A/B. Do NOT stack partition_summed_x2 on top.

**Reason for downgrade from "primary root cause parity gap" to "field-trial substrate"**: AEC3 itself treats UseRefinedOutput / FormLinearFilterOutput as field-trial mechanisms (per [echo_remover.cc:430-444](aec3_extracts/src/aec3/echo_remover.cc) — `use_coarse_filter_output_` config defaults to False; production AEC3 always uses refined). Our pipeline missing the field-trial substrate is true, but the "missing" classification overstated its role; with the substrate enabled, the cond2 (refined-diverged) firing rate exposes a shadow-not-converged downstream hole on no-clean-convergence stress cases (XRTnTUjU and similar). See "Corrected root-cause" below.

## Implementation summary

- Flag: `use_refined_output_selection_for_linear_path: bool = False` ([config.py:221](AEC/python/modules/config.py))
- Env: `AEC_USE_REFINED_OUTPUT_SELECTION=1`
- Code:
  - Capture shadow output: orchestrator.py `_last_shadow_output_time = self.shadow_filter.process(...)` at the existing shadow.process() call site.
  - Method `_aec3_select_linear_filter_output` ([orchestrator.py:3304+](AEC/python/modules/orchestrator.py)): AEC3 UseRefinedOutput predicate + spectral selection between `filter.error_spec_windowed/echo_spec` (refined) and `(error_spec_windowed + echo_spec - shadow.echo_spec)/shadow.echo_spec` (coarse).
  - Wired into `_aec3_post` `_PSD_SCALE` block: when flag ON, `echo_psd` + `error_spec` use the selected filter's spectra.
- Byte-equal: 24/24 MD5 PASS vs v3.21.6 baseline (flag default OFF).

## 12-case cohort

| Category | n | Stems |
|---|---:|---|
| **Stress** (must recover) | 4 | XRTnTUjU_doubletalk; nVUnxqHLr_doubletalk; MYrVxVEMxkaE_doubletalk; xFk7igec_doubletalk_with_movement |
| FS_static | 3 | 9xjhi; qNvSMyU; xQEUtY2 |
| FS_movement | 1 | 0I0XMl3M |
| DT_static normal | 1 | jtYTdZm3 |
| DT_movement normal | 2 | wVYSGV; ZJYUt0O0A |
| NE | 1 | 014AzuqPZku |

Both runs use intended HPF (mic ON / ref OFF), preset balanced, filter 832, --cng, --parallel j4.

## Per-case Δ (B − A; A = v3.21.6 baseline, B = v3.21.8 flag ON)

### STRESS cases (must recover Δdeg ≥ 0)

| stem | A_deg | B_deg | **Δdeg** | A_echo | B_echo | Δecho |
|---|---:|---:|---:|---:|---:|---:|
| XRTnTUjU_doubletalk | 3.929 | 3.235 | **−0.694** | 4.081 | 4.288 | +0.207 |
| nVUnxqHLr_doubletalk | 3.223 | 2.103 | **−1.120** | 4.687 | 4.665 | −0.022 |
| MYrVxVEM_doubletalk | 2.869 | 1.643 | **−1.226** | 4.065 | 4.183 | +0.119 |
| xFk7igec_doubletalk_with_movement | 2.882 | 2.360 | **−0.522** | 4.012 | 4.161 | +0.149 |

**All 4 stress cases REGRESS on deg**: −0.522 to −1.226 dB. Gate FAIL.

### NORMAL cases (must not regress)

| stem | bucket | A_deg | B_deg | Δdeg | Δecho |
|---|---|---:|---:|---:|---:|
| 014AzuqPZku NE | NE | 4.354 | 4.354 | 0 | 0 |
| 0I0XMl3M FS_mv | FS_movement | 4.999 | 4.999 | 0 | +0.214 |
| 9xjhi FS | FS_static | 5.000 | 5.000 | 0 | +0.332 |
| qNvSMyU FS | FS_static | 4.999 | 4.999 | 0 | +0.134 |
| xQEUtY2 FS | FS_static | 5.000 | 5.000 | 0 | +0.067 |
| jtYTdZm DT_static | DT_static | 2.347 | 2.568 | +0.221 | +0.052 |
| wVYSGV DT_mv | DT_movement | 2.203 | 2.759 | **+0.556** | +0.028 |
| ZJYUt0O0A DT_mv | DT_movement | 1.991 | 2.690 | **+0.700** | −0.164 |

**8/8 normal cases improve or neutral**. Normal no-regress: PASS.

## Comparison vs partition_summed_x2 alone (Gate 1)

| stress case | partition_summed_x2 Δdeg | v3.21.8 Δdeg | delta |
|---|---:|---:|---|
| XRTnTUjU | −1.253 | −0.694 | v3.21.8 less regressive |
| nVUnxqHLr | −1.475 | −1.120 | less regressive |
| MYrVxVEM | −1.346 | −1.226 | less regressive |
| xFk7igec | −1.057 | −0.522 | less regressive |

v3.21.8 reduces stress damage by 30-50 % vs partition_summed_x2 alone, but DOES NOT recover. Both interventions hurt stress (independently).

## Correction 2026-05-22 — shadow IS PBFDAF (NLMS) and is the AEC3-equivalent coarse filter

Initial analysis incorrectly claimed `shadow_filter` was PBFDKF with Q×3.5. That was wrong. Verified via runtime introspection on BALANCED preset:

```
shadow_class_nlms = True  (BALANCED default in config.py)
shadow filter class    = PBFDAF
shadow has Q attr      = False (no Kalman state)
shadow_mu_nlms         = 0.5
```

Our shadow IS the AEC3-equivalent bounded NLMS coarse filter. The Gate 2 `#2a UseRefinedOutput` premise (shadow ≈ AEC3 coarse) is therefore architecturally valid. But the v3.21.8 implementation, when enabled, exposes a downstream hole on no-clean-convergence stress cases — described in "Corrected root cause" below.

## Per-frame trace evidence (XRTnTUjU, full utterance 3487 frames, flag ON)

| metric | count | % |
|---|---:|---:|
| frames_total | 3487 | 100 % |
| use_refined | 2183 | 62.6 % |
| **use_coarse** | **1304** | **37.4 %** |
| ↳ cond1 fired (coarse-cleaner) | 446 | 12.8 % |
| ↳ cond2 fired (refined-diverged) | **1274** | **36.5 %** |

37 % of frames switch to coarse, dominated by `cond2: e2_coarse < e2_refined AND y2 < e2_refined` (refined residual is larger than mic energy → AEC3 deems refined "diverged"). cond1 has the `s2_refined > 60²·N OR s2_coarse > 60²·N` precondition (so coarse selection requires SOME filter to have a non-trivial echo estimate); **cond2 has no such precondition**.

## Corrected root cause (2026-05-22)

XRTnTUjU lacks a clean FS preamble (DT-first, preprocessed). Consequence:

- **Refined PBFDKF** Kalman with high Q on a non-converging signal exhibits local divergence (e_refined energy > y2 on 36 % of frames) → cond2 fires
- **Shadow NLMS** with mu = 0.5 is bounded but ALSO has not converged → `shadow.echo_spec ≈ 0`

When cond2 fires and our spectral selection swaps `error_spec_windowed → near_spec_windowed − shadow.echo_spec`:

```
coarse_error_spec_windowed ≈ near_spec_windowed − 0 = near_spec_windowed
→ error_psd ≈ near_psd
→ ResidualEchoEstimator: "all of mic is echo"
→ SuppressionGain ≈ 0
→ DT speech entirely suppressed
```

This is the OPPOSITE of UseRefinedOutput's AEC3 intent. In AEC3 production, `use_coarse_filter_output_` defaults to False ([echo_remover.cc:430-444](aec3_extracts/src/aec3/echo_remover.cc)), so this hole is not exposed. Field-trial deployments that enable the substrate run with cohorts where shadow has had time to converge before cond2 can fire.

## Mechanism summary table

| component | AEC3 production behavior | Our v3.21.8 behavior | Outcome on stress (no-clean-convergence) |
|---|---|---|---|
| `use_coarse_filter_output_` outer gate | False (default) → always refined | Always inside selection (flag-ON path) | Selection runs by default when flag is ON |
| cond1 precondition `s²_refined>60²·N OR s²_coarse>60²·N` | Gates coarse-cleaner branch | Same | OK |
| cond2 (refined-diverged) precondition | None | None | **No shadow-converged gate → fires before shadow has tracked echo** |
| Selected coarse spectrum quality | NLMS coarse adapts in field-trial cohorts (assumed converged) | NLMS shadow `echo_spec ≈ 0` on stress | RES misinterprets as "all echo" → catastrophic gain |

## Reframing — v3.21.8 as field-trial substrate

Per AEC3's own production-default (`use_coarse_filter_output_` = False), our v3.21.8 is NOT a missing production parity. It is the **field-trial substrate**:

| Aspect | Classification |
|---|---|
| Code surface | Preserved (config flag + helper method + spectral selection + env hook + hysteresis state) |
| Default | OFF (`use_refined_output_selection_for_linear_path: bool = False`) |
| Byte-equal vs v3.21.6 baseline (flag OFF) | PASS (24/24 MD5) |
| Ship status | NOT shippable as production-default; available for opt-in research / future cohort with NLMS-coarse-converged guarantee |
| Gate 2 reclassification of `#2a` | "AEC3 field-trial mechanism, default-OFF parity. NOT a production parity gap." |

## Why NOT pursue shadow-converged precondition (deferred to v3.22)

A shadow-converged gate on cond2 (e.g., `s2_coarse > 60²·hop / 32768²` required before cond2 fires) would close the no-clean-convergence hole. **But it is NOT AEC3 parity** — AEC3 cond2 has no such gate, because AEC3's NLMS coarse in field-trial cohorts is assumed to have a clean preamble.

Per the v3.21.x = AEC3 parity rule, shadow-converged precondition is **out of scope for v3.21.x**. If it survives a future audit, it goes to v3.22 (beyond-AEC3 optimization).

## Next-step direction (chosen 2026-05-22)

**A**: Close v3.21.8 as field-trial substrate, Cat C, not shippable. ✓ (this document)

**C**: Open next v3.21.x parity patch for **#1c coarse e2/y2 window mismatch**. Pure formula fix in `_aec3_post`'s convergence check: replace `_e2_coarse` Parseval-over-fft_size reconstruction with time-domain block sum-of-squares over hop, matching AEC3 `SubtractorOutput::ComputeMetrics` semantics. Independent of shadow convergence; default-OFF flag with byte-equal guard. See follow-up sub-patch doc.

**NOT chosen**: shadow-converged precondition (B; defer to v3.22), pivot to v3.22 wholesale (D; premature), other Gate 2 candidates require independent A/B and are queued behind #1c.

## Forbidden actions

- Do NOT ship v3.21.8 to production (default OFF stays)
- Do NOT stack partition_summed_x2 on top of v3.21.8 (precondition not met)
- Do NOT introduce V2 / counter / shadow-converged gating in the v3.21.x cycle
- Do NOT delete research branches (Cat C preserves debug evidence)

## Status

- v3.21.8 candidate (in current form) **NOT shipping** — stress recovery gate FAIL
- Production main unchanged
- v3.21.7 partition_summed_x2 still PAUSED (Cat C BLOCKED-STRESS)
- Research branches preserved (Cat C rule)
- Implementation preserved (byte-equal default OFF; reusable as substrate if Option A pursued)

## Forbidden actions (still in force)

- Do NOT proceed to 800-case A/B for v3.21.8 (12-case gate FAILED)
- Do NOT stack partition_summed_x2 on top (precondition not met)
- Do NOT delete research branches (Cat C rule)
- Do NOT mix V2 gate3 / convergence-counter variants

## Awaiting decision

Which option (A / B / C / D) to pursue next.
