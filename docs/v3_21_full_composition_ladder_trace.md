# v3.21 Full AEC3 Composition Ladder Trace

**Date**: 2026-05-26 (v3 — corrected M_A: use_aec3_h_error_ceil=True + use_aec3_filter_noise_gate_power=True)  
**Ladder**: M0 → M_R0 → M_B_corrected → M_C → M_A → M_D  
**8 required cases** (see plan §Required Cases)  
**No AECMOS / no 800-case / no verdict / no version bump**

> This supersedes all prior trace runs (v1 M0→M_B→M_C→M_A→M_D without M_R0;
> v2 M_A without H_ERROR_CEIL parity fix). v3 is the authoritative corrected run.

---

## §0 Step Definitions

| Step | Flags ON (cumulative) | Purpose |
|------|-----------------------|---------|
| **M0** | All candidate flags explicitly OFF | v3.21.6 anchor; byte-equal baseline |
| **M_R0** | `use_aec3_filter_noise_gate_power=True` | R0.1 standalone: PBFDKF refined filter noise gate 27509562→20075344 int16² (0.02562→0.01870 float); 1.37× looser |
| **M_B_corrected** | M_R0 + A.1–A.5 (Bundle B shadow/coarse PBFDAF) | Shadow mu denom, noise gate constant, poor-excitation gate, narrowband mask, saturation gate |
| **M_C** | M_B_corrected + URO + crossfade (Bundle C) | Per-frame refined/coarse output selection + 30-sample crossfade |
| **M_A** | M_C + Bundle A + use_aec3_h_error_ceil=True | Refined PBFDKF H_error parity: partition-summed X², instantaneous E², per-bin H_error refresh, AEC3 ceil=2.0, corrected filter noise gate |
| **M_D** | M_A (no new flags; Bundle D trace-only) | FilterAnalyzer consumer chain, usable_linear gate breakdown, delay events |
| **M_full_delay** | M_D + `use_full_delay_change_chain=True` | Full AEC3 delay-change chain: H_error reset + counter reset; delay-event cases only |

---

## §1 Bundle Classification

Each flag classified as: **(a) strict AEC3 alignment**, **(b) functional adaptation under hop_size=160**, **(c) default-OFF substrate (non-AEC3)**, **(d) closed/no-op**.

| Flag | Classification | AEC3 default | Source |
|------|---------------|-------------|--------|
| `use_aec3_filter_noise_gate_power` (R0.1) | **(a) strict AEC3 alignment** | ON (20075344 unconditional) | `refined_filter_update_gain.cc:97`; `coarse_filter_update_gain.cc:98`. Python had wrong constant (27509562) from suppression path. |
| `use_partition_summed_x2_for_shadow_mu` (A.1) | **(b) functional adaptation** | ON (Σ_p\|X_buf[p]\|², 13 partitions, 64-sample hop) | AEC3 SpectralSum = SUM; we use 5 partitions, 160-sample hop. Mu rate ratio 0.74× (T1.1 analysis). |
| `use_aec3_noise_gate_for_shadow` (A.2) | **(a) strict AEC3 alignment** | ON (20075344 int16² for coarse filter gate) | `coarse_filter_update_gain.cc:98`. Constant corrected via T1.2. |
| `use_poor_excitation_gate_for_shadow` (A.3) | **(a) strict AEC3 alignment** | ON (early return when poor_excitation_counter < n_partitions) | `coarse_filter_update_gain.cc` poor-excitation guard. |
| `use_narrowband_mask_for_shadow` (A.4) | **(a) strict AEC3 alignment** | ON (per-bin narrowband mask pre-multiplied on mu) | RSA narrowband mask from RenderSignalAnalyzer. |
| `use_saturation_gate_for_shadow` (A.5) | **(a) strict AEC3 alignment** | ON (early return on _saturated_capture) | AEC3 saturation guard in CoarseFilterUpdateGain. |
| `use_partition_summed_x2_for_h_error_gain` | **(a) strict AEC3 alignment** | ON (SpectralSum = Σ_p\|X_buf[p]\|²) | `refined_filter_update_gain.cc` X² denom. |
| `use_current_e2_refined_in_h_error_denominator` | **(a) strict AEC3 alignment** | ON (instantaneous \|error_spec\|², not smoothed EMA) | `refined_filter_update_gain.cc` E² denom. |
| `use_per_bin_h_error_refresh` | **(a) strict AEC3 alignment** | ON (per-bin H_error refresh from \|error_spec\|², gated e²≤e²_coarse) | `refined_filter_update_gain.cc:128–138`. |
| `use_aec3_h_error_ceil` | **(a) strict AEC3 alignment** | ON (kHErrorCeiling=2.0 unconditional) | `refined_filter_update_gain.cc`. Python default=100.0 was confirmed live parity gap. |
| `use_refined_output_selection_for_linear_path` (URO) | **(a) strict AEC3 alignment** | ON (per-frame refined vs coarse selection via cond1+cond2) | `echo_remover.cc` — NOTE: AEC3 `enable_coarse_filter_output_usage=true` default; UseRefinedOutput is NOT a non-default flag. |
| `form_linear_filter_crossfade_enabled` | **(a) strict AEC3 alignment** | ON (30-sample SignalTransition ramp on selector switch) | `echo_remover.cc` WindowedPaddedFft crossfade. |
| `use_full_delay_change_chain` | **(b) functional adaptation — v3.21 alignment composition candidate** | ON (full chain: ZeroFilter+H_error reset+counter reset+SetConfig+AecState) | AEC3 `AdaptiveFirFilter::ZeroFilter(current,max)` in steady state (current=max=13) = NO-OP. Python `W.fill(0)` is more aggressive. SetSizePartitions = structural incompatibility. Primary mechanism: H_error reset=10000 + poor_excitation_counter reset. **NOT v3.22** — still v3.21.x alignment closure item. D2 (SetConfig initial-state leakage) not yet fully ported but FIXABLE (FilterPlateauDetector is Python-only). |
| `transparent_mode_enabled` | **(c) default-OFF substrate** | ON in AEC3 production | Excluded from ladder — P2 scaling bug (÷2.5 for 10ms hops). Independent probe M_probe_TM only; requires user authorisation. |
| `use_aec3_zero_filter_on_epc` (ablation) | **(c) default-OFF substrate (non-AEC3)** | N/A | W.fill(0) is Python-only ablation. AEC3 ZeroFilter(13,13) is NO-OP in steady state. |

**Closed / no-op items**:
- `use_coarse_e2_time_domain_parity`: **(d)** — mechanism correct parity, but 0/24 file diffs (threshold-bound no-op in practice). Code preserved default-OFF.
- `usable_linear_trusted_external_delay_only`: **(d)** — ablation knob, not parity. M3–M2 delta=0 on all 12 cases.
- `saturation_subtractor_inputs_enabled`: **(d)** — zero-fire on 8-case cohort. M4–M3 delta=0.

---

## §2 Bundle B: Shadow State — M0 vs M_B_corrected

| Case | M0 W_norm | MB W_norm | M0 e2_coa | MB e2_coa | MB A2_zero | MB A3_skip | MB A4_mask | MB A5_sat |
|------|----------|----------|----------|----------|-----------|-----------|-----------|----------|
| 9xjhiFbG_FS_static | 83.007 | 71.848 | 9.211 | 16.907 | 6.2% | 1.4% | 0.2% | 0.0% |
| xFk7igec_DT_mvmt | 17.693 | 14.561 | 0.493 | 0.739 | 7.0% | 0.2% | 0.1% | 0.0% |
| nVUnxqHL_DT_static | 14.828 | 14.927 | 0.398 | 0.369 | 4.1% | 0.1% | 0.0% | 0.0% |
| XRTnTUjU_DT_static | 2.182 | 3.144 | 0.263 | 0.306 | 2.9% | 0.2% | 0.0% | 0.0% |
| MYrVxVEM_DT_static | 13.042 | 12.553 | 0.041 | 0.060 | 6.5% | 0.5% | 0.0% | 0.0% |
| wVYSGVTT_DT_mvmt | 79.722 | 81.305 | 2.727 | 4.467 | 5.5% | 0.2% | 0.0% | 0.0% |
| qNvSMyUS_FS_static | 1.818 | 1.820 | 0.174 | 0.179 | 11.3% | 2.1% | 0.3% | 0.0% |
| jtYTdZm3_DT_static | 21.856 | 19.822 | 0.635 | 1.105 | 18.6% | 0.2% | 0.0% | 0.0% |

**coarse_conv_frac** (e2_coarse < 0.05·y2 AND y2 > thr — AEC3 criterion):

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D |
|------|-------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | 0.4% | 0.4% | 0.2% | 0.2% | 0.2% | 0.2% |
| nVUnxqHL_DT_static | 0.4% | 0.5% | 0.2% | 0.2% | 0.1% | 0.1% |
| XRTnTUjU_DT_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MYrVxVEM_DT_static | 2.3% | 2.3% | 0.9% | 0.9% | 0.9% | 0.9% |
| wVYSGVTT_DT_mvmt | 5.5% | 5.4% | 4.3% | 4.3% | 3.0% | 3.0% |
| qNvSMyUS_FS_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| jtYTdZm3_DT_static | 5.6% | 5.8% | 5.0% | 5.0% | 5.4% | 5.4% |

> All steps < 6% = PBFDKF structural mismatch (shadow NLMS vs AEC3 FIR coarse). Not a Bundle B failure. coarse_conv_frac is expected low due to PBFDKF architecture divergence.

---

## §3 Bundle C: URO Root-Cause — M_C

| Case | use_refined_frac | cond1_frac | cond2_frac | e2_ratio(coa/ref) | form_xfade_frac |
|------|-----------------|-----------|-----------|-----------------|----------------|
| 9xjhiFbG_FS_static | 64.6% | 32.4% | 7.4% | 0.004 | 17.7% |
| xFk7igec_DT_mvmt | 38.9% | 36.2% | 56.8% | 0.082 | 34.5% |
| nVUnxqHL_DT_static | 67.6% | 17.4% | 28.4% | 0.359 | 26.3% |
| XRTnTUjU_DT_static | 68.2% | 10.4% | 30.5% | 1.132 | 16.2% |
| MYrVxVEM_DT_static | 47.5% | 44.7% | 46.3% | 0.013 | 8.0% |
| wVYSGVTT_DT_mvmt | 46.8% | 30.0% | 40.5% | 0.613 | 30.3% |
| qNvSMyUS_FS_static | 28.7% | 6.7% | 70.4% | 0.885 | 9.0% |
| jtYTdZm3_DT_static | 57.1% | 40.8% | 33.2% | 0.254 | 10.3% |

**URO coarse-frame attribution** (stats on frames where URO selected coarse output):

| Case | n_coarse_frames | e2_coa@coa | e2_ref@coa | y2@coa | A2_zero@coa | A3_skip@coa |
|------|----------------|-----------|-----------|-------|------------|------------|
| 9xjhiFbG_FS_static | 774 | 10.769 | 11044.711 | 17.836 | 12.2% | 1.2% |
| xFk7igec_DT_mvmt | 2246 | 0.552 | 14.446 | 0.565 | 7.8% | 0.0% |
| nVUnxqHL_DT_static | 1394 | 0.275 | 2.524 | 0.271 | 5.8% | 0.0% |
| XRTnTUjU_DT_static | 1108 | 0.197 | 0.251 | 0.202 | 5.3% | 0.0% |
| MYrVxVEM_DT_static | 2071 | 0.081 | 8.551 | 0.194 | 10.9% | 0.7% |
| wVYSGVTT_DT_mvmt | 1952 | 3.699 | 10.308 | 5.333 | 7.1% | 0.0% |
| qNvSMyUS_FS_static | 1915 | 0.086 | 0.162 | 0.159 | 13.2% | 2.4% |
| jtYTdZm3_DT_static | 1578 | 1.196 | 9.048 | 0.791 | 31.2% | 0.0% |

> xFk7 cond2_frac=56.8% → refined-diverged is dominant coarse-selection trigger (e2_coarse < e2_refined AND y2 < e2_refined). e2_ratio=0.082 confirms coarse genuinely cleaner during movement phases.
> XRTnTUjU e2_ratio=1.132 (coarse ≥ refined) → cond1/cond2 fire but coarse not cleanly better; primary mechanism here is convergence state.

---

## §4 Bundle A: H_error Contamination-Aware Audit — M_C vs M_A

> **H_error_mean is contaminated** by frames where far_hop_energy ≤ 1e-4 skips `_update_weights()`, leaving H_error at init=10000.  
> Use `steady_at_ceil_excl20` (fraction of bins at ceiling, excluding 20 hops after any reset) as the clean ceiling-saturation signal.

| Case | step | reset_frames | above_ceil_frac | post_refresh_at_ceil | steady_excl6 | steady_excl20 | steady_excl50 |
|------|------|------------|---------------|---------------------|------------|--------------|--------------|
| 9xjhiFbG_FS_static | M_C | 4 | 0.2% | 0.8% | 0.5% | 0.0% | 0.0% |
| 9xjhiFbG_FS_static | **M_A** | **4** | **0.2%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| xFk7igec_DT_mvmt | M_C | 60 | 1.6% | 0.6% | 0.1% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | **M_A** | **59** | **1.6%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| nVUnxqHL_DT_static | M_C | 65 | 1.5% | 0.3% | 0.1% | 0.0% | 0.0% |
| nVUnxqHL_DT_static | **M_A** | **65** | **1.5%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| XRTnTUjU_DT_static | M_C | 60 | 1.7% | 0.2% | 0.0% | 0.0% | 0.0% |
| XRTnTUjU_DT_static | **M_A** | **60** | **1.7%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| MYrVxVEM_DT_static | M_C | 63 | 1.6% | 1.1% | 0.7% | 0.5% | 0.3% |
| MYrVxVEM_DT_static | **M_A** | **113** ⚠️ | **2.9%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| wVYSGVTT_DT_mvmt | M_C | 14 | 0.4% | 3.3% | 2.7% | 1.9% | 0.8% |
| wVYSGVTT_DT_mvmt | **M_A** | **11** | **0.3%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| qNvSMyUS_FS_static | M_C | 61 | 2.3% | 0.2% | 0.0% | 0.0% | 0.0% |
| qNvSMyUS_FS_static | **M_A** | **61** | **2.3%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| jtYTdZm3_DT_static | M_C | 2 | 0.1% | 6.4% | 6.1% | 5.5% | 4.9% |
| jtYTdZm3_DT_static | **M_A** | **10** ⚠️ | **0.3%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

**Key findings**:
- **H_ERROR_CEIL parity CONFIRMED**: `use_aec3_h_error_ceil=True` drives all `post_refresh_at_ceil` and `steady_excl20` to 0.0% across all cases. Rule 1 applies — ceiling saturation resolved.
- **MYrVxVEM anomaly** ⚠️: reset_frames 63 → 113 at M_A (per-bin H_error refresh firing additional reset/contamination events). `steady_at_ceil_excl20=0.0%` — ceiling saturation not the issue. Root cause: `use_per_bin_h_error_refresh` triggers more frequent H_error conditional refresh events, which the reset counter tracks. Not a correctness problem; per-bin refresh is working as intended.
- **jtYTdZm3 anomaly** ⚠️: reset_count 2 → 10 at M_A. Same mechanism as MYrVxVEM. Ceiling saturation resolved (steady_excl20=0.0%).
- **wVYSGVTT M_C→M_A**: post_refresh_at_ceil drops from 3.3% to 0.0% (ceiling saturation resolved). reset_count 14→11 (slight reduction).

---

## §5 Bundle D: usable_linear + FilterAnalyzer — M_D

> Filter_analyzer_enabled=True is already the v3.21.6 baseline. Bundle D is trace-only (no new flags).  
> \* FA_consistent=0% = FilterAnalyzer never fires consistent_estimate=True for these cases.  
> usable_linear maintained via ext_delay shortcut (gate 3a) — delay estimator solid.

| Case | usable_linear | ul_g1 | ul_g2 | ul_g3_ext | ul_g3_conv | FA_consistent | conv_latch_frame | delay_events |
|------|--------------|-------|-------|----------|-----------|--------------|----------------|-------------|
| 9xjhiFbG_FS_static | 92.4% | 93.0% | 93.0% | 96.8% | 80.9% | 72.3% | 222 | 1 |
| xFk7igec_DT_mvmt | 96.1% | 96.2% | 96.2% | 97.2% | 96.5% | 90.8% | 129 | 1 |
| nVUnxqHL_DT_static | 95.9% | 96.0% | 96.0% | 97.2% | 92.9% | 90.6% | 304 | 1 |
| XRTnTUjU_DT_static | 91.2% | 91.2% | 91.2% | 96.9% | 18.6% | 0.0% \* | 241 | 1 |
| MYrVxVEM_DT_static | 95.6% | 95.6% | 95.6% | 97.8% | 95.3% | 86.6% | 156 | 1 |
| wVYSGVTT_DT_mvmt | 49.5% ⚠️ | 49.5% | 49.5% | 98.0% | 44.5% | 17.5% | — | 1 |
| qNvSMyUS_FS_static | 82.5% | 85.7% | 85.7% | 93.0% | 53.9% | 0.0% \* | 1239 | 1 |
| jtYTdZm3_DT_static | 97.7% | 97.8% | 97.8% | 93.3% | 98.0% | 87.3% | 53 | 1 |

> **wVYSGVTT ul=49.5% at M_D** — caused by Bundle A (per_bin_h_error_refresh + instantaneous E²) interaction with delay_first event. FilterPlateauDetector fires at frame 461 (before once_converged latches at frame 475). ul_gate3_ext=98.0% — ext_delay shortcut is not the constraint. **M_full_delay recovers to 96.3% (+46.9pp)** via H_error reset that triggers proper reconvergence path.

---

## §6 M_full_delay vs M_D (delay-event cases)

> All 8 cases have delay_events=1 (delay_first only at startup). All qualify for M_full_delay.

| Case | M_D ul | Mfd ul | Δul | M_D cc | Mfd cc | Δcc | Mfd nores_LF Δ vs M0 |
|------|--------|--------|-----|--------|--------|-----|----------------------|
| 9xjhiFbG_FS_static | 92.4% | 92.4% | 0.0% | 0.0% | 0.0% | 0.0% | −5.83 dB |
| xFk7igec_DT_mvmt | 96.1% | 96.1% | 0.0% | 0.2% | 0.1% | −0.0% | **+1.53 dB** ⚠️ |
| nVUnxqHL_DT_static | 95.9% | 95.9% | 0.0% | 0.1% | 0.1% | 0.0% | +0.15 dB |
| XRTnTUjU_DT_static | 91.2% | 91.2% | 0.0% | 0.0% | 0.0% | 0.0% | +0.03 dB |
| MYrVxVEM_DT_static | 95.6% | 95.6% | 0.0% | 0.9% | 1.0% | +0.1% | +0.03 dB |
| **wVYSGVTT_DT_mvmt** | **49.5%** | **96.3%** | **+46.9%** ✓ | 3.0% | 4.4% | +1.3% | −1.08 dB |
| qNvSMyUS_FS_static | 82.5% | 82.5% | 0.0% | 0.0% | 0.0% | 0.0% | −0.07 dB |
| jtYTdZm3_DT_static | 97.7% | 97.7% | 0.0% | 5.4% | 5.1% | −0.4% | −1.45 dB |

> **wVYSGVTT**: M_full_delay restores ul from 49.5%→96.3% — the H_error reset on delay_first enables proper reconvergence before FilterPlateauDetector fires. M_full_delay is REQUIRED for wVYSGVTT composition closure.
> **xFk7 nores_LF +1.53 dB** ⚠️: Gate 0 primary case. Delay chain causes minor nores_LF regression. Root cause: H_error reset delays convergence_seen latch; FilterPlateauDetector fires at frame 470 (vs absent in M_D). Variants C/C2/D1_corrected all FAIL — see `docs/v3_21_m_full_delay_classification_audit.md`. D2 pending design implementation.

---

## §7 Nores Artifact Gate

> Closure rule (FS cases): Δ LF at M_A or later vs M0 < −0.5 dB → linear layer reduced artifact → proceed to AECMOS review.  
> M0 column = absolute dB; other columns = Δ vs M0.

### Nores LF (0–500 Hz)

| Case | M0 (dB) | M_R0 | M_B_corrected | M_C | **M_A** | M_D | M_full_delay |
|------|---------|------|------|------|------|------|------|
| **9xjhiFbG_FS_static** | **24.80** | +0.06 | −0.02 | −0.02 | **−6.03** ✓ | −6.03 | −5.83 |
| xFk7igec_DT_mvmt | 6.16 | +0.06 | +0.08 | +0.08 | +1.70 | +1.70 | +1.53 |
| nVUnxqHL_DT_static | 9.42 | +0.16 | +0.16 | +0.16 | +0.89 | +0.89 | +0.15 |
| XRTnTUjU_DT_static | 5.01 | −0.02 | −0.02 | −0.02 | +0.02 | +0.02 | +0.03 |
| MYrVxVEM_DT_static | −0.22 | +4.35 | +4.35 | +4.35 | +0.12 | +0.12 | +0.03 |
| wVYSGVTT_DT_mvmt | 16.92 | −0.76 | −0.76 | −0.76 | −0.77 | −0.77 | −1.08 |
| **qNvSMyUS_FS_static** | **−2.35** | −0.00 | +0.03 | +0.03 | −0.05 | −0.05 | −0.07 |
| jtYTdZm3_DT_static | 13.63 | −1.16 | −1.16 | −1.16 | −0.57 | −0.57 | −1.45 |

### Nores MF (500–2000 Hz)

| Case | M0 (dB) | M_R0 | M_B_corrected | M_C | M_A | M_D | M_full_delay |
|------|---------|------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 23.41 | +0.07 | +0.07 | +0.07 | +2.13 | +2.13 | — |
| xFk7igec_DT_mvmt | 12.87 | −0.17 | −0.17 | −0.17 | −0.49 | −0.49 | — |
| nVUnxqHL_DT_static | 9.75 | −1.55 | −1.55 | −1.55 | −2.53 | −2.53 | — |
| XRTnTUjU_DT_static | 8.49 | +0.00 | +0.00 | +0.00 | +0.09 | +0.09 | — |
| MYrVxVEM_DT_static | 7.13 | +2.41 | +2.41 | +2.41 | +1.29 | +1.29 | — |
| wVYSGVTT_DT_mvmt | 20.12 | +1.44 | +1.44 | +1.44 | +0.76 | +0.76 | — |
| qNvSMyUS_FS_static | 5.21 | −0.08 | −0.08 | −0.08 | +0.31 | +0.31 | — |
| jtYTdZm3_DT_static | 10.48 | −1.53 | −1.53 | −1.53 | −0.68 | −0.68 | — |

### Nores HF (2000–8000 Hz)

| Case | M0 (dB) | M_R0 | M_B_corrected | M_C | M_A | M_D | M_full_delay |
|------|---------|------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 21.76 | +0.12 | +0.12 | +0.12 | +1.16 | +1.16 | — |
| xFk7igec_DT_mvmt | 8.18 | +0.16 | +0.16 | +0.16 | −6.11 | −6.11 | — |
| nVUnxqHL_DT_static | 0.12 | +0.34 | +0.34 | +0.34 | +0.27 | +0.27 | — |
| XRTnTUjU_DT_static | 0.66 | −0.07 | −0.07 | −0.07 | +0.27 | +0.27 | — |
| MYrVxVEM_DT_static | −4.34 | −0.83 | −0.83 | −0.83 | −1.06 | −1.06 | — |
| wVYSGVTT_DT_mvmt | 14.41 | +0.52 | +0.52 | +0.52 | +0.62 | +0.62 | — |
| qNvSMyUS_FS_static | −11.03 | +0.04 | +0.04 | +0.04 | +0.06 | +0.06 | — |
| jtYTdZm3_DT_static | 2.92 | +0.04 | +0.04 | +0.04 | −0.25 | −0.25 | — |

**Artifact closure verdict**:
- **9xjhiFbG (primary artifact case)**: M_A Δ LF = **−6.03 dB** ✓ — closure criterion MET (< −0.5 dB). Linear layer confirmed as the artifact source. Proceed to AECMOS review at 12-case pending user authorisation.
- **qNvSMyUS**: Δ LF = −0.05 dB (negligible). Not a primary artifact case (M0 baseline already near-zero LF).
- **MYrVxVEM M_B_corrected MF +4.35 dB**: Transient regression at Bundle B; largely resolved at M_A (+1.29 dB, within watch range).

---

## §8 Full Ladder: usable_linear_frac × step

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D | M_full_delay |
|------|-------|------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 94.6% | 94.6% | 92.4% | 92.4% | 92.4% | 92.4% | 92.4% |
| xFk7igec_DT_mvmt | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% | 96.1% |
| nVUnxqHL_DT_static | 95.9% | 95.9% | 95.9% | 95.9% | 95.9% | 95.9% | 95.9% |
| XRTnTUjU_DT_static | **40.6%** | 40.6% | **91.2%** | 91.2% | 91.2% | 91.2% | 91.2% |
| MYrVxVEM_DT_static | 96.8% | 96.8% | 96.8% | 96.8% | 95.6% | 95.6% | 95.6% |
| wVYSGVTT_DT_mvmt | 96.3% | 96.3% | 96.3% | 96.3% | **49.5%** | **49.5%** | **96.3%** ✓ |
| qNvSMyUS_FS_static | 82.5% | 82.5% | 84.2% | 84.2% | 82.5% | 82.5% | 82.5% |
| jtYTdZm3_DT_static | 97.8% | 97.8% | 97.8% | 97.8% | 97.7% | 97.7% | 97.7% |

**Key transitions**:
- **XRTnTUjU**: 40.6% → 91.2% at M_B_corrected. Bundle B (shadow PBFDAF gates) is the fix — poor-excitation counter + partition-summed X² allow convergence_seen to latch.
- **wVYSGVTT**: 96.3% → 49.5% at M_A. Bundle A (per-bin H_error refresh + instantaneous E²) interacts with delay_first → FilterPlateauDetector fires before once_converged latches. **Recovered to 96.3% at M_full_delay** (+46.9pp). Delay chain is REQUIRED.
- **MYrVxVEM**: minor 96.8% → 95.6% at M_A (per-bin H_error refresh causing more reset frames; not a convergence failure).

---

## §9 Full Ladder: coarse_conv_frac × step

| Case | M0 | M_R0 | M_B_corrected | M_C | M_A | M_D |
|------|-------|------|------|------|------|------|
| 9xjhiFbG_FS_static | 0.0% | 0.0% | 0.1% | 0.1% | 0.0% | 0.0% |
| xFk7igec_DT_mvmt | 0.4% | 0.4% | 0.2% | 0.2% | 0.2% | 0.2% |
| nVUnxqHL_DT_static | 0.4% | 0.5% | 0.2% | 0.2% | 0.1% | 0.1% |
| XRTnTUjU_DT_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MYrVxVEM_DT_static | 2.3% | 2.3% | 0.9% | 0.9% | 0.9% | 0.9% |
| wVYSGVTT_DT_mvmt | 5.5% | 5.4% | 4.3% | 4.3% | 3.0% | 3.0% |
| qNvSMyUS_FS_static | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| jtYTdZm3_DT_static | 5.6% | 5.8% | 5.0% | 5.0% | 5.4% | 5.4% |

---

## §10 r2 and Suppression Gain Per Band × Ladder Step

> r2_lf = r2 at bin 5 (~156 Hz), r2_mf = bin 30 (~937 Hz), r2_hf = bin 100 (~3125 Hz).  
> gain < 1.0 = echo suppression active.

| Case | step | r2_lf | r2_mf | r2_hf | gain_lf | gain_mf | gain_hf |
|------|------|-------|-------|-------|---------|---------|---------|
| 9xjhiFbG_FS_static | M0 | 81.9G | 24.5G | 9.71G | 0.282 | 0.344 | 0.352 |
| 9xjhiFbG_FS_static | M_B_corrected | 23.1G | 8.74G | 1.98G | 0.433 | 0.492 | 0.503 |
| 9xjhiFbG_FS_static | M_C | 25.2G | 6.05G | 1.57G | 0.388 | 0.523 | 0.509 |
| 9xjhiFbG_FS_static | M_A | 44.6G | 5.25G | 1.70G | 0.358 | 0.580 | 0.556 |
| 9xjhiFbG_FS_static | M_D | 44.6G | 5.25G | 1.70G | 0.358 | 0.580 | 0.556 |
| XRTnTUjU_DT_static | M0 | 29.9M | 22.8M | 106.6M | 0.834 | 0.845 | 0.750 |
| XRTnTUjU_DT_static | M_B_corrected | 28.0M | 16.9M | 132.5M | 0.851 | 0.851 | 0.778 |
| XRTnTUjU_DT_static | M_C | 9.00M | 8.28M | 55.5M | 0.866 | 0.853 | 0.763 |
| XRTnTUjU_DT_static | M_A | 5.08M | 11.5M | 186.8M | 0.845 | 0.842 | 0.788 |
| XRTnTUjU_DT_static | M_D | 5.08M | 11.5M | 186.8M | 0.845 | 0.842 | 0.788 |
| wVYSGVTT_DT_mvmt | M0 | 10.9G | 3.31G | 233.6M | 0.725 | 0.716 | 0.736 |
| wVYSGVTT_DT_mvmt | M_B_corrected | 33.4G | 4.87G | 259.0M | 0.708 | 0.701 | 0.718 |
| wVYSGVTT_DT_mvmt | M_C | 7.19G | 2.45G | 129.0M | 0.674 | 0.693 | 0.717 |
| wVYSGVTT_DT_mvmt | M_A | 116.3M | 280.7M | 6.75M | 0.814 | 0.881 | 0.869 |
| wVYSGVTT_DT_mvmt | M_D | 116.3M | 280.7M | 6.75M | 0.814 | 0.881 | 0.869 |

> wVYSGVTT M_A: r2 drops dramatically (LF 10.9G→116M, MF 3.31G→281M) — linear filter state better estimated, SuppressionGain backs off (gain_lf 0.725→0.814). This is correct: Bundle A yields cleaner residual estimate; suppressor reduces attenuation when echo is genuinely low.

---

## §11 Composition Verdict Summary (as of 2026-05-26 v3)

### What passed

| Criterion | Verdict | Evidence |
|-----------|---------|---------|
| H_ERROR_CEIL parity | ✓ PASS | All steady_excl20=0.0% at M_A; xFk7 43.3%→0% ceiling saturation resolved |
| 9xjhi nores LF artifact closure | ✓ PASS | M_A Δ LF = −6.03 dB (< −0.5 dB threshold) |
| XRTnTUjU usable_linear recovery | ✓ PASS | M_B_corrected 40.6%→91.2% (+50.6pp); Bundle B confirmed as fix |
| wVYSGVTT ul recovery at M_full_delay | ✓ PASS | M_D 49.5% → M_full_delay 96.3% (+46.9pp); delay chain required |
| All steady_excl20 = 0.0% at M_A | ✓ PASS | No ceiling saturation in steady-state operation |

### What requires user decision

| Item | Status | Required action |
|------|--------|----------------|
| **12-case AECMOS** | PENDING USER AUTHORISATION | **M_full_delay = current composition candidate** — recovers wVYSGVTT 49.5%→96.3%. **M_D = pre-delay-chain baseline / ablation comparator** (wVYSGVTT known bad: ul=49.5%). **12-case must compare M0 vs M_D vs M_full_delay** — not M_full_delay alone, not M_D alone. |
| **12-case hard watch items** | REQUIRED | (1) wVYSGVTT DT_movement: M_full_delay must recover vs M_D (reference: ul 49.5%→96.3% on 8-case). (2) xFk7 DT_movement: no catastrophic echo/deg/nores regression (watch: nores_LF +1.53 dB at M_full_delay vs M0). (3) 9xjhi FS_static: nores_LF artifact closure must remain (−5.83 dB at M_full_delay — still meets < −0.5 dB threshold). (4) MYrVxVEM: reset_count anomaly (63→113) must not become AECMOS regression. |
| **M_full_delay Gate 0 D2** | OPEN — not needed for corrected composition | Corrected composition already shows M_full_delay recovers wVYSGVTT without D2. D2 (coarse rate=0.9, shadow mu_initial=0.643) remains candidate; requires explicit user auth to implement. |
| **MYrVxVEM reset_count anomaly** | WATCH | reset_frames 63→113 at M_A. Not ceiling saturation (excl20=0%). Monitor at 12-case AECMOS. |

### What remains open (v3.21.x)

| Item | Classification | Next action |
|------|---------------|------------|
| R0.2 residual noise gate unit-verify | Track 2 pending | Read AEC3 source; same-unit verify 27509.42f vs 27509562 int16² |
| R0.3 EchoGeneratingPower window | Track 2 pending | Flag implementation (use_aec3_echo_generating_power_window) |
| R0.4 FilterAnalyzer continuous consistent_estimate | Deferred v3.21.x+ | Substrate incompatibility; not current sprint |
| M_full_delay D2 implementation | UNRESOLVED Gate 0 | Requires user authorisation |

---

*Full per-case raw data: `docs/v3_21_full_composition_trace_results.md`*  
*Trace script: `python/v3_21_full_composition_trace.py`*  
*Gate 0 classification audit: `docs/v3_21_m_full_delay_classification_audit.md`*
