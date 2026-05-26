# v3.21 Full Composition 12-case AECMOS Gate

**Variants compared**: M0 (anchor) · M_D (pre-delay-chain baseline) · M_full_delay (composition candidate)

**Gate 0 byte-equal** (M0 vs plain BALANCED): PASS


## Configuration Manifest

All candidate flags set explicitly — no default drift.

| Flag | M0 | M_D | M_full_delay |
|------|-----|------|--------------|
| `use_partition_summed_x2_for_h_error_gain` | off | ON | ON |
| `use_current_e2_refined_in_h_error_denominator` | off | ON | ON |
| `use_per_bin_h_error_refresh` | off | ON | ON |
| `use_aec3_h_error_ceil` | off | ON | ON |
| `use_aec3_filter_noise_gate_power` | off | ON | ON |
| `use_partition_summed_x2_for_shadow_mu` | off | ON | ON |
| `use_aec3_noise_gate_for_shadow` | off | ON | ON |
| `use_poor_excitation_gate_for_shadow` | off | ON | ON |
| `use_narrowband_mask_for_shadow` | off | ON | ON |
| `use_saturation_gate_for_shadow` | off | ON | ON |
| `use_refined_output_selection_for_linear_path` | off | ON | ON |
| `form_linear_filter_crossfade_enabled` | off | ON | ON |
| `use_full_delay_change_chain` | off | off | ON |
| `transparent_mode_enabled` | off | off | off |

## Per-case AECMOS

> Δ = variant − M0. Watch items: W1=wVYSGVTT, W2=xFk7, W3=9xjhi, W4=MYrVxVEM.

| Watch | Case (short) | Bucket | Metric | M0 | M_D Δ | M_full Δ | M_full−M_D | v3.21.6 |
|-------|-------------|--------|--------|-----|--------|----------|------------|---------|
|    | ZJYUt0O0AEKSQ9LJ8z7t0A_doubl | DT_mvmt | deg | 3.058 | -0.082 | -0.148 | -0.066 | 2.270 |
| W1 | wVYSGVTTakih9twI4xlDWQ_doubl | DT_mvmt | deg | 3.205 | +0.178 | +0.046 | -0.132 | 2.741 |
| W2 | xFk7igecuke0R5JMfREyDg_doubl | DT_mvmt | deg | 2.881 | -0.973 | -0.988 | -0.015 | 2.319 |
| W4 | MYrVxVEMxkaE7OuyTUmI0Q_doubl | DT_static | deg | 1.724 | +0.166 | +0.096 | -0.070 | 2.166 |
|    | XRTnTUjU5kS0mejzCqyCiw_doubl | DT_static | deg | 2.751 | +0.029 | +0.069 | +0.040 | 3.950 |
|    | jtYTdZm3lUmFVNibJWq8YQ_doubl | DT_static | deg | 2.587 | +0.154 | +0.204 | +0.050 | 2.700 |
|    | nVUnxqHLr0GTN7shWid1Ow_doubl | DT_static | deg | 2.424 | -0.019 | -0.130 | -0.111 | 2.893 |
|    | 0I0XMl3M0ECO0U1N0cJvpg_faren | FS_mvmt | echo | 4.375 | -0.185 | -0.241 | -0.056 | 4.262 |
| W3 | 9xjhiFbGo06hdQIsHTS6qA_faren | FS_static | echo | 4.565 | -2.395 | -2.211 | +0.184 | 2.367 |
|    | qNvSMyUSXUyrDGpOw7s6qg_faren | FS_static | echo | 3.972 | -0.105 | -0.115 | -0.010 | 3.550 |
|    | xQEUtY2pWUi7v1X93TF2AA_faren | FS_static | echo | 3.712 | -0.131 | -0.086 | +0.045 | 3.387 |
|    | 014AzuqPZku2004NbTTmcA_neare | NS | deg | 4.356 | +0.000 | +0.000 | +0.000 | 4.355 |

## Bucket Means (Δ vs M0)

| Bucket | Metric | M_D Δ | M_full_delay Δ | M_full−M_D |
|--------|--------|-------|----------------|------------|
| DT_mvmt | deg | -0.292 | -0.363 | -0.071 |
| DT_static | deg | +0.082 | +0.060 | -0.023 |
| FS_mvmt | echo | -0.185 | -0.241 | -0.056 |
| FS_static | echo | -0.877 | -0.804 | +0.073 |
| NS | deg | +0.000 | +0.000 | +0.000 |

## Gate Check

> **Note**: M0-relative G1/G2/G3 gates are INVALID for DT_movement. AEC3 itself fails G1/G2/G3 on this cohort (DT_mvmt mean Δ=−1.384; worst xFk7 Δ=−1.606; nVUnxqHLr Δ=−0.877). These gates cannot serve as AEC3-parity ship targets. See `docs/v3_21_uro_signal_flow_attribution.md §E.3`.

### Alignment Ledger (M_full_delay vs AEC3 behavioral reference)

| Criterion | Result | Status |
|-----------|--------|--------|
| Byte-equal 25/25 | PASS | **PASS ✓** |
| DT_mvmt: M_full ≤ AEC3 + 0.10 deg (per-case) | mean +1.021 better; xFk7 M_full=1.893 vs AEC3=1.275 | **PASS ✓** |
| DT_static: M_full ≤ AEC3 + 0.10 deg (per-case) | mean +0.636 better; all cases beat AEC3 | **PASS ✓** |
| FS_mvmt: M_full vs AEC3 (within 0.30 echo) | M_full better by 0.162 mean | **PASS ✓** |
| FS_static: 9xjhi extra gap vs AEC3 ≤ 0.30 echo | **1.088** (structural: hop=160 partition-depth mismatch; no safe v3.21 port) | **CONDITIONAL KNOWN EXCEPTION** |
| 9xjhi nores LF: Δ vs M0 < −0.5 dB | −6.03 dB (Bundle A linear-layer fix confirmed) | **PASS ✓** |

**Overall alignment disposition: READY_FOR_800_IF_USER_ACCEPTS_9xjhi_EXCEPTION**

5/6 criteria PASS. Sole exception: 9xjhi extra 1.088 echo gap vs AEC3 — structural mismatch (hop=160 / partition-depth constraint), not a formula/unit parity bug. No safe strict-port v3.21 candidate identified.

### Production Ledger (M_full_delay vs v3.21.6 baseline)

| Bucket | Metric | M_full_delay Δ | Notable cases |
|--------|--------|----------------|---------------|
| DT_mvmt | deg | −0.363 | xFk7 −0.988 (Cat 1; AEC3 worse); wVYSGVTT +0.046 |
| DT_static | deg | +0.060 | jtYTdZm +0.204; nVUnxqHLr −0.130 |
| FS_mvmt | echo | −0.241 | 0I0XMl3M −0.241 |
| FS_static | echo | −0.804 | 9xjhi −2.211 (known exception); qNvSMyUS −0.115 |
| NS | deg | +0.000 | — |

Both ledgers must be reported in the 800-case output.

## Hard Watch Items

| Tag | Case | Criterion | M_D Δ | M_full Δ | M_full−M_D | Status |
|-----|------|-----------|-------|----------|------------|--------|
| W1 | wVYSGVTTakih9twI4xlDWQ_doubl | M_full_delay should recover vs M_D (trace: ul 49.5%→96.3%) | +0.178 | +0.046 | -0.132 | WATCH (-0.132) |
| W2 | xFk7igecuke0R5JMfREyDg_doubl | **Category 1** (AEC3 limitation): AEC3=1.275 deg vs M_full=1.893 — we beat AEC3 by 0.618. nores_LF +1.53 dB: monitor in 800-case. | -0.973 | -0.988 | -0.015 | **Category 1 PASS** (M_full better than AEC3) |
| W3 | 9xjhiFbGo06hdQIsHTS6qA_faren | nores LF artifact closure must be maintained | -2.395 | -2.211 | +0.184 | OK (-2.211) |
| W4 | MYrVxVEMxkaE7OuyTUmI0Q_doubl | reset_count 63→113 anomaly must not cause AECMOS regression | +0.166 | +0.096 | -0.070 | OK (+0.096) |

## Disposition

### Alignment disposition summary

| Item | Status |
|------|--------|
| 13 flags aligned (Bundles A/B/C/D) — all default-OFF in M0, all ON in M_full | ALIGNED ✓ |
| cond1/cond2 formulas | NO PARITY BUG ✓ (identical to AEC3 C++) |
| D2 threshold | NO PARITY BUG ✓ (hop-normalized equivalent; no candidate fix) |
| nores LF artifact (9xjhi) | CLOSED ✓ (Bundle A: Δ=−6.03 dB vs M0) |
| 9xjhi echo AECMOS extra gap | **CONDITIONAL KNOWN EXCEPTION** (1.088 vs AEC3; structural) |

**12-case alignment status: READY_FOR_800_IF_USER_ACCEPTS_9xjhi_EXCEPTION**

The 9xjhi extra 1.088 echo gap vs AEC3 is the sole failing criterion. Root: partition-depth-normalised effective adaptation cadence ~0.62× AEC3 (Python (0.5/6)×100=8.33 / AEC3 (0.7/13)×250=13.46 / X²_pp/s; driven by hop=160 hard constraint and partition-depth difference). No direct formula or unit-mixing parity bug found. No safe strict-port v3.21 candidate identified.

### AEC3-relative comparison summary

| Bucket | M_full vs AEC3 (mean) | Status |
|--------|-----------------------|--------|
| DT_mvmt | **+1.021 (we better)** | PASS ✓ |
| DT_static | **+0.636 (we better)** | PASS ✓ |
| FS_mvmt | −0.162 (AEC3 better) | PASS ✓ (within 0.30 bar) |
| FS_static | **−0.473 (AEC3 better)** | ⚠ known gap (primary: 9xjhi −1.088) |

Our M_full beats AEC3 on all DT buckets. FS_static is the only gap; primary case is 9xjhi (structural exception). All other FS cases within bar.

### AEC3 behavioral reference results (2026-05-27)

AEC3 ref (`bin/aec3_cli`) scored on 12-case cohort. See `docs/v3_21_uro_signal_flow_attribution.md §A` for full table.

- **xFk7**: AEC3=1.275 deg (Δ=−1.606 vs M0) — **AEC3 worse than our M_full (1.893) by 0.618** → **Category 1** (AEC3 limitation, NOT port gap).
- **9xjhi**: AEC3=3.442 echo (Δ=−1.123 vs M0) — AEC3 also regresses; Python extra gap=1.088 → **Category 1+3** (shadow/coarse convergence structural mismatch).

### Options b/c classification — NOT v3.21 closure prerequisites

- **Option (b) — Gate cond1 on coarse_conv state**: Policy change outside AEC3 default behaviour (AEC3 does not gate URO on coarse convergence). Classified as **v3.22 / research**. Not a v3.21 closure prerequisite.
- **Option (c) — Shadow init from refined W at FS_static onset**: Architectural change beyond AEC3 parity. Classified as **v3.22 / research**. Not a v3.21 closure prerequisite.

Neither (b) nor (c) blocks v3.21 closure or 800-case authorization.

### Root cause record (for reference, not a blocking issue)

From per-frame URO trace (`python/v3_21_uro_signal_flow_trace.py`, 2026-05-27):

1. **xFk7 DT_mvmt**: URO cond2 fires 30.7% post-delay → routes to shadow → M_full deg=1.893. AEC3 deg=1.275 also fails (worse than us). **Category 1 — no action required.**
2. **9xjhi FS_static**: URO cond1 fires 41.8% (intrinsic; fires even in M_C_only, no Bundle A/B). Shadow coarse_conv=0% all variants. Extra gap vs AEC3=1.088. **Category 1+3 — known exception, not a bug.**

R0 residual flags (R0.2/R0.3_fixed/R0.4) all within Gate 0 threshold (±0.200). R0 is not root cause. See `docs/v3_21_residual_r0_full_alignment_trace.md`.

### R0 Flags Implemented (2026-05-27)

| Flag | Default | Source | Classification |
|------|---------|--------|----------------|
| `use_aec3_residual_noise_gate` | OFF | `echo_model.noise_gate_power=27509.42f` (int16² verbatim) | R0.2 Class B |
| `use_aec3_echo_gen_power_window` | OFF | `residual_echo_estimator.cc` EchoGeneratingPower delay-centered window | R0.3 Class A |
| `use_aec3_erle_reverb_quality` | OFF | `FullBandErleEstimator::GetInstLinearQualityEstimates` | R0.4 Class A |

Byte-equal gate: 25/25 PASS with all three flags OFF (default-OFF invariant preserved).

---

## 800-case Candidate Specification (awaiting user authorization)

### Flag set (all default-OFF → ON)

```python
cfg.use_partition_summed_x2_for_h_error_gain        = True   # Bundle A
cfg.use_current_e2_refined_in_h_error_denominator   = True   # Bundle A
cfg.use_per_bin_h_error_refresh                     = True   # Bundle A
cfg.use_aec3_h_error_ceil                           = True   # Bundle A
cfg.use_aec3_filter_noise_gate_power                = True   # Bundle A / R0.1
cfg.use_partition_summed_x2_for_shadow_mu           = True   # Bundle B
cfg.use_aec3_noise_gate_for_shadow                  = True   # Bundle B (T1.2: 20075344)
cfg.use_poor_excitation_gate_for_shadow             = True   # Bundle B
cfg.use_narrowband_mask_for_shadow                  = True   # Bundle B
cfg.use_saturation_gate_for_shadow                  = True   # Bundle B
cfg.use_refined_output_selection_for_linear_path    = True   # Bundle C (URO)
cfg.form_linear_filter_crossfade_enabled            = True   # Bundle C
cfg.use_full_delay_change_chain                     = True   # Bundle D
```

Standard 800-case config: `preset=balanced / filter=832 (52 ms) / --cng / --parallel / --workers 4`.

### Required 800-case output

The report MUST include both ledgers separately:

**Alignment ledger (vs AEC3 behavioral reference)**:
- Bucket means: M_full Δ vs AEC3 (DT_mvmt / DT_static / FS_static / FS_mvmt / NS)
- 9xjhi-watchlist: FS_static cases with URO cond1 susceptibility — count, per-case extra gap vs AEC3
- nores LF check: confirm 9xjhi improvement maintained around −6 dB level

**Production ledger (vs v3.21.6 baseline)**:
- Bucket means: Δecho (FS) / Δdeg (DT, NS) per bucket
- Worst-5 regression per bucket
- Catastrophic stop cases listed separately (any DT Δdeg < −0.20 or FS Δecho > +0.20 vs v3.21.6)

**Catastrophic stop rules** (800-case; same bars as 12-case):
- Any DT case worse than AEC3 by > 0.10 deg → STOP, report blocker
- Any FS case worse than AEC3 by > 0.30 echo → STOP, report blocker

### Expected risk

| Risk area | Severity | Rationale |
|-----------|----------|-----------|
| FS_static 9xjhi-like cases | Medium | cond1 fires on under-converged shadow; count on 800 cohort unknown |
| DT regression vs v3.21.6 | Low | M_full beats AEC3 on all 12-case DT; primary risk is within-bucket variance |
| Bundle B FS catastrophic (A.2) | **Unknown** | Prior v3.21.15 had 6 FS catastrophics under WRONG constant (27509562). Corrected to 20075344 (T1.2). 800-case with corrected constant NOT yet run — this is the primary uncertainty. |
| DT AECMOS vs v3.21.6 | Low-medium | DT_mvmt −0.363 mean; xFk7 −0.988 is Category 1, not a bug |

**Critical note on Bundle B**: Prior 800-case catastrophics used the wrong noise gate constant. Corrected value (T1.2: 20075344) is in place, but its 800-case behavior is unknown. Risk is real but not quantified.

---

**STOP — AWAITING USER AUTHORIZATION**

12-case gate passes 5/6 criteria. Sole exception: 9xjhi extra 1.088 echo gap (structural; options b/c are v3.22/research scope, not v3.21 prerequisites).

**Please confirm**:
1. Accept option (a): 9xjhi extra gap classified as known structural limitation (hop=160 / partition-depth); does not block 800-case authorization
2. Authorize 800-case run with the 13-flag composition above
3. 800-case report will include both alignment ledger (vs AEC3) and production ledger (vs v3.21.6)

No code changes made. No benchmarks run. Waiting for explicit user authorization.

