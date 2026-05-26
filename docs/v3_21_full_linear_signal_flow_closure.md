# v3.21 Full AEC3 Linear Signal-Flow Closure Audit

**Date**: 2026-05-26  
**Status**: ACTIVE — Phase 1 trace in progress  
**Supersedes**: v3.21.20 ALL-ON audio trace (root cause found: composition incomplete, not individual flag bugs)

---

## 1. Background and Constraint

M1–M4 12-case matrix (URO + crossfade patch-ladder) failed G1/G2/G3 across all variants. **This
does NOT close URO as NOSHIP.** M1–M4 tested URO on top of an incomplete coarse/shadow composition
(A.1–A.5 all default-OFF). AEC3's URO coarse fallback surface is untrusted without Bundle B: the
composition is incomplete, making URO's cond1/cond2 decisions based on unreliable coarse filter
state.

All previously-isolated NOSHIP results (partition_summed_x2, per_bin_h_error_refresh, shadow gates,
URO) are **historical isolated results only** — they carry zero weight as full-composition verdicts
and must NOT be used to close any flag as NOSHIP.

**Hard constraints (permanent)**:
- 不改 dominant_ne order
- 不啟用 C1–C5 ship
- 不跑 800-case until user authorises after AECMOS pass
- 不 merge / version bump
- 不開 v3.22 until full AEC3 composition tested and proven PBFDKF-incompatible
- Do NOT classify any flag as NOSHIP based on isolated-patch failure alone

---

## 2. AEC3-Default Classification Table

| Flag | AEC3 default | Our default | Composition include? |
|------|-------------|-------------|----------------------|
| `use_partition_summed_x2_for_h_error_gain` | ON | OFF | **YES — Bundle A (M_A step)** |
| `use_current_e2_refined_in_h_error_denominator` | ON | OFF | **YES — Bundle A (M_A step)** |
| `use_per_bin_h_error_refresh` | ON | OFF | **YES — Bundle A (M_A step; prior NOSHIP = isolated only)** |
| H_ERROR_CEIL parity (`use_aec3_h_error_ceil`) | ON (ceil=2.0) | OFF (Python=100.0) | **Candidate — trace H_error_at_ceil_frac first; add to M_A if live** |
| `use_partition_summed_x2_for_shadow_mu` | ON | OFF | **YES — Bundle B (M_B step)** |
| `use_aec3_noise_gate_for_shadow` | ON | OFF | **YES — Bundle B (M_B step; prior 800-case fail = isolated only)** |
| `use_poor_excitation_gate_for_shadow` | ON | OFF | **YES — Bundle B (M_B step)** |
| `use_narrowband_mask_for_shadow` | ON | OFF | **YES — Bundle B (M_B step; 12-case zero-fire ≠ 800-case)** |
| `use_saturation_gate_for_shadow` | ON | OFF | **YES — Bundle B (M_B step; 12-case zero-fire ≠ 800-case)** |
| `use_refined_output_selection_for_linear_path` | ON | OFF | **YES — Bundle C (M_C step; M1–M4 = composition-incomplete, not NOSHIP)** |
| `form_linear_filter_crossfade_enabled` | ON | OFF | **YES — Bundle C (M_C step)** |
| `filter_analyzer_enabled` | ON | **ON (True)** | **TRACE ONLY — already baseline; not a new ladder flag** |
| `use_full_delay_change_chain` | ON | OFF | **YES — M_full_delay only; delay-event cases only** |
| `transparent_mode_enabled` | ON | OFF | **NO — independent probe M_probe_TM; user auth required; P2 bug** |
| `usable_linear_trusted_external_delay_only` | non-default | OFF | NO (ablation knob) |
| `usable_linear_require_filter_analyzer_consistent` | non-default | OFF | NO (ablation knob) |
| `usable_linear_disable_external_delay_shortcut` | non-default | OFF | NO (ablation knob) |
| `signal_dependent_erle_sections=2` | ON | 0 | OUT OF SCOPE (residual layer) |
| `coarse_filter_converged_relaxed_enabled` | field-trial | OFF | NO |

---

## 3. H_ERROR_CEIL Known Mismatch

| | Python | AEC3 |
|--|--------|------|
| H_error ceiling | `H_ERROR_CEIL_FLOAT = 1e2` (100.0) in `aec3_scale.py:84` | `error_ceil = 2.0` in `refined_filter_update_gain.cc` |
| H_error floor | `H_ERROR_FLOOR_FLOAT = 1e-3` | same |
| Post-reset init | `H_ERROR_INIT_FLOAT` (from `aec3_scale.py`) | 10000 (AEC3 `kInitialErrorEstimate`) |

**Audit rule**: Trace `H_error_at_ceil_frac` (fraction of bins at ceiling post-clip). If live
(> 5% of bins at ceiling in normal operation, post-startup), introduce `use_aec3_h_error_ceil`
as a default-off candidate for Bundle A and re-run M_A with it. If dormant (< 1%), document as
known mismatch with negligible behavioral impact.

**PBFDKF mu formula** (`filters.py:791–796`):
```
denom = 0.5·H_error[k]·X2[k] + n·E2[k] + δ
mu[k] = H_error[k] / denom
H_error[k] -= 0.5·mu[k]·X2[k]·H_error[k]   # decay
H_error[k] clipped to [1e-3, 100.0]           # Python; AEC3: [floor, 2.0]
```

---

## 4. Bundle Definitions

### Bundle A — Refined PBFDKF H_error parity

All four items are **AEC3 default ON** (unconditional in `refined_filter_update_gain.cc`).
All prior isolated results are historical only.

| Flag | AEC3 behavior | Historical isolated result |
|------|---------------|---------------------------|
| `use_partition_summed_x2_for_h_error_gain` | X² denom = Σ_p\|X_buf[p]\|² (SpectralSum) | v3.21.7 800-case: DT_static Cat C Δdeg −1.475 — **isolated, not composition** |
| `use_current_e2_refined_in_h_error_denominator` | E² = instantaneous \|error\|² not smoothed EMA | Not bench-tested in isolation |
| `use_per_bin_h_error_refresh` | per-bin H_error refresh, gate e²_ref ≤ e²_coarse per bin | Prior NOSHIP = isolated test; rescinded as composition verdict |
| H_ERROR_CEIL parity | AEC3 ceil=2.0; Python=100.0 | **Known mismatch** — trace H_error_at_ceil_frac |

**Shadow `_update_weights` execution order** (`filters.py:273–339`):  
A.5 (sat) → A.3 (poor_excite) → stationary → A.4 (narrowband) → A.1 (X²) → A.2 (noise gate) → final mu_eff

### Bundle B — Shadow/Coarse PBFDAF parity (A.1–A.5)

All five are **AEC3 default ON**. All prior isolated results rescinded as composition verdicts.

| Flag | AEC3 behavior | Historical isolated result |
|------|---------------|---------------------------|
| `use_partition_summed_x2_for_shadow_mu` (A.1) | shadow mu denom = Σ_p\|X_buf[p]\|² | Not bench-tested |
| `use_aec3_noise_gate_for_shadow` (A.2) | hard-zero mu where X² < NOISE_GATE | v3.21.15 isolated 800-case FS catastrophic — **composition-incomplete** |
| `use_poor_excitation_gate_for_shadow` (A.3) | early return when poor_excitation_counter < n_part OR startup | v3.21.15 tested isolated — historical only |
| `use_narrowband_mask_for_shadow` (A.4) | per-bin narrowband mask pre-multiplied on mu | v3.21.14 zero-fire on 12-case cohort — not representative of 800-case |
| `use_saturation_gate_for_shadow` (A.5) | early return on `_saturated_capture` | v3.21.14 zero-fire on cohort — historical only |

### Bundle C — Output selection parity

**Corrected AEC3 default**: `enable_coarse_filter_output_usage=true` (default in production).
`use_coarse_filter_output_` is NOT a compile-time gate — it's a per-frame decision variable.
`FormLinearFilterOutput` **always runs** in AEC3; refined is preferred, coarse selected only when
cond1 or cond2 fires.

| Flag | AEC3 behavior | Historical isolated result |
|------|---------------|---------------------------|
| `use_refined_output_selection_for_linear_path` (URO) | Per-frame refined vs coarse via cond1+cond2 | M1–M4 fail — **composition-incomplete; NOT NOSHIP verdict** |
| `form_linear_filter_crossfade_enabled` | 30-sample SignalTransition ramp on selector switch | jtYTdZm3 isolated +0.184; overwhelmed by URO failures in M1–M4 |

**URO cond logic** (`orchestrator.py:3830–3836`):
```
cond1 (coarse_cleaner):   e2_coarse < 0.9·e2_refined  AND  y2 > thr_30  AND  (s2_ref > thr_60 OR s2_coa > thr_60)
cond2 (refined_diverged): e2_coarse < e2_refined        AND  y2 < e2_refined
use_refined = NOT (cond1 OR cond2)
```

**URO dependency**: `e2_coarse` quality depends directly on Bundle B. Without B, the coarse/shadow
composition is incomplete, making URO's coarse fallback surface untrusted.

### Bundle D — AecState / FilterQuality consumer parity

**FilterAnalyzer status**: `filter_analyzer_enabled` default **True** in AecConfig (line 162) →
passed as `enable_filter_analyzer=True` to `AecStateConfig` at orchestrator.py:693. Already ON in
v3.21.6 baseline. **No new flag added — Bundle D is trace-only.**

| Item | AEC3 | Python default | Ladder action |
|------|------|---------------|---------------|
| `filter_analyzer_enabled` | ON | **ON (True)** | TRACE ONLY — already baseline |
| `use_full_delay_change_chain` | ON | OFF | M_full_delay only; delay-event cases only |

**Ablation knobs excluded from ladder** (not AEC3 default-on alignment):
- `usable_linear_require_filter_analyzer_consistent`
- `usable_linear_disable_external_delay_shortcut`

**M_probe_TM** (independent probe — NOT part of main ladder):
- `transparent_mode_enabled` has P2 scaling bug (÷2.5 for 10ms hops vs AEC3 4ms blocks)
- Requires **user explicit authorisation** before running
- NOT run in same round as main ladder

---

## 5. Dependency Ladder

```
M0  (anchor — all candidate flags explicitly OFF)
 └─► M_B  (M0 + Bundle B: A.1–A.5 shadow/coarse PBFDAF gates)
       └─► M_C  (M_B + Bundle C: URO + crossfade)
             └─► M_A  (M_C + Bundle A: refined H_error parity + H_ERROR_CEIL audit)
                   └─► M_D  (M_A + Bundle D trace; no new flags — filter_analyzer already ON)
                         └─► M_full_delay  (M_D + delay-chain; delay-event cases only)

Independent probe (user authorisation required; NOT same round):
  M_probe_TM  (M_D + transparent_mode_enabled; P2 scaling bug caveat)
```

**Rule**: Each layer adds exactly the flags from one bundle. Attribution trace at each step
identifies which bundle caused which behavioral change.

### M0 — Baseline (v3.21.6 anchor)

```python
# M0 — all candidate flags explicitly overridden (do NOT rely on AecConfig defaults)
cfg.use_partition_summed_x2_for_h_error_gain = False
cfg.use_current_e2_refined_in_h_error_denominator = False
cfg.use_per_bin_h_error_refresh = False
cfg.use_partition_summed_x2_for_shadow_mu = False
cfg.use_aec3_noise_gate_for_shadow = False
cfg.use_poor_excitation_gate_for_shadow = False
cfg.use_narrowband_mask_for_shadow = False
cfg.use_saturation_gate_for_shadow = False
cfg.use_refined_output_selection_for_linear_path = False
cfg.form_linear_filter_crossfade_enabled = False
cfg.use_full_delay_change_chain = False
cfg.transparent_mode_enabled = False  # explicitly OFF; M_probe_TM is separate
# filter_analyzer_enabled: keep True (already ON in v3.21.6 baseline)
```

### M_B — Bundle B ON (shadow/coarse PBFDAF)

```python
# M_B = M0 + Bundle B
cfg.use_partition_summed_x2_for_shadow_mu = True   # A.1
cfg.use_aec3_noise_gate_for_shadow = True           # A.2
cfg.use_poor_excitation_gate_for_shadow = True      # A.3
cfg.use_narrowband_mask_for_shadow = True           # A.4
cfg.use_saturation_gate_for_shadow = True           # A.5
```

**Attribution gate**: confirm e2_coarse more reliable than M0 before proceeding.

### M_C — M_B + Bundle C (URO + crossfade)

```python
# M_C = M_B + Bundle C
cfg.use_refined_output_selection_for_linear_path = True
cfg.form_linear_filter_crossfade_enabled = True
```

**Corrected version of M1–M4**: URO now runs with full coarse composition.

### M_A — M_C + Bundle A (refined H_error parity)

```python
# M_A = M_C + Bundle A
cfg.use_partition_summed_x2_for_h_error_gain = True
cfg.use_current_e2_refined_in_h_error_denominator = True
cfg.use_per_bin_h_error_refresh = True
# use_aec3_h_error_ceil: added if H_error_at_ceil_frac is live (>5%)
```

### M_D — M_A + Bundle D (trace-only)

```python
# M_D = M_A (no new flags — filter_analyzer_enabled already True)
# Bundle D: trace FilterAnalyzer consumer chain + gate behavior + delay_event_count
```

### M_full_delay — M_D + full delay-change chain

```python
# M_full_delay = M_D + delay chain
cfg.use_full_delay_change_chain = True
```

Only on cases with `delay_event_count > 0` from M_D trace.

---

## 6. Nores Artifact Closure Gate

**Rule**: AECMOS alone is NOT sufficient for v3.21 closure. The nores LF grid/fan artifact on
`9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` must be addressed.

**Closure criterion**: At M_C or beyond, `nores_LF_extra_energy_db` must decrease vs M0 on the
primary artifact case, OR the artifact must be confirmed as a residual-layer issue (defer to
separate arc — do NOT declare v3.21 linear closure either way).

**Artifact bands** (measured in `_ours_nores.wav` = linear output without RES):
- LF: 0–500 Hz (FFT bins 0–4 at 16kHz/160-sample hop = 100 Hz/bin)
- MF: 500–2000 Hz (bins 5–19)
- HF: 2000–8000 Hz (bins 20–80)

---

## 7. Required Cases

| Case | Bucket | Delay events? | Why required |
|------|--------|--------------|-------------|
| `9xjhiFbGo06hdQIsHTS6qA` | FS_static | No | nores LF artifact primary; artifact closure gate |
| `xFk7igecuke0R5JMfREyDg` | DT_mvmt | TBD (M0 trace) | Gate 0; delay_first verified; URO stress |
| `nVUnxqHLr0GTN7shWid1Ow` | DT_static | No | Worst ALL-ON regressor (−0.954) |
| `XRTnTUjU5kS0mejzCqyCiw` | DT_static | No | 2nd worst ALL-ON (−0.569); use_capture collapse |
| `MYrVxVEMxkaE7OuyTUmI0Q` | DT_static | No | 4th stress; ALL-ON −0.436 |
| `wVYSGVTTakih9twI4xlDWQ` | DT_mvmt | TBD | DT_mvmt stress; shadow A.1 −0.148 isolated |
| `qNvSMyUSXUyrDGpOw7s6qg` | FS_static | No | Clean FS guard; load-bearing on spectral_floor |
| `jtYTdZm3lUmFVNibJWq8YQ` | DT_static | No | Clean DT guard; crossfade +0.184 isolated |

---

## 8. Previously Closed Items (carry-forward)

| Item | Verdict | Evidence | Composition verdict? |
|------|---------|---------|---------------------|
| `use_coarse_e2_time_domain_parity` | NOOP | v3.21.9: 0/24 file diffs | Yes — no-op under both compositions |
| `usable_linear_trusted_external_delay_only` | SUBSTRATE (non-default) | M3–M2 delta=0, 12 cases | Yes — ablation knob, not parity |
| `saturation_subtractor_inputs_enabled` | SUBSTRATE (zero-fire) | M4–M3 delta=0, 12 cases | Yes — confirmed zero-fire |
| Gate 0 `use_full_delay_change_chain` | PASS (Gate 0 only) | `v3_21_a1_gate0_trace_verdict.md`: W ratio 0.9766 | Gate 0 only; Gate 1 = M_full_delay |
| URO M1–M4 | Historical isolated fail | Composition-incomplete (no Bundle B) | Rescinded; must re-test in M_C |
| Crossfade M1–M4 | Historical isolated result | jtYTdZm3 +0.184 isolated; overwhelmed | Rescinded; re-test in M_C |

---

## 9. Trace Results — Bundle B (M0 → M_B delta)

*Populated by `python/v3_21_full_composition_trace.py` — see below.*

### Bundle B: Shadow state

| Case | M0 shadow_W_norm | M_B shadow_W_norm | M_B e2_coarse_mean | M_B A2_zero_frac | M_B A3_skip_frac | M_B coarse_conv_frac |
|------|-----------------|-------------------|-------------------|-----------------|-----------------|---------------------|
| 9xjhi_FS | — | — | — | — | — | — |
| xFk7_DT_mv | — | — | — | — | — | — |
| nVUnxqHLr_DT | — | — | — | — | — | — |
| XRTnTUjU_DT | — | — | — | — | — | — |
| MYrVxVEM_DT | — | — | — | — | — | — |
| wVYSGVTT_DT_mv | — | — | — | — | — | — |
| qNvSMy_FS | — | — | — | — | — | — |
| jtYTdZm3_DT | — | — | — | — | — | — |

**Attribution gate**: confirm `e2_coarse_mean` decreased and `coarse_conv_frac` increased vs M0
before proceeding to M_C.

---

## 10. Trace Results — Bundle C (M_B → M_C delta)

### URO root-cause trace (M_C — URO ON)

| Case | use_refined_frac | cond1_frac | cond2_frac | e2_coarse / e2_refined | form_transition_frac |
|------|-----------------|-----------|-----------|----------------------|---------------------|
| 9xjhi_FS | — | — | — | — | — |
| xFk7_DT_mv | — | — | — | — | — |
| nVUnxqHLr_DT | — | — | — | — | — |
| XRTnTUjU_DT | — | — | — | — | — |
| MYrVxVEM_DT | — | — | — | — | — |
| wVYSGVTT_DT_mv | — | — | — | — | — |
| qNvSMy_FS | — | — | — | — | — |
| jtYTdZm3_DT | — | — | — | — | — |

### Nores artifact gate (FS cases — LF extra energy vs M0)

| Case | M0 nores_LF_db | M_B nores_LF_db | M_C nores_LF_db | Δ(M_C − M0) | Artifact reduced? |
|------|---------------|----------------|----------------|------------|-----------------|
| 9xjhi_FS (primary) | — | — | — | — | — |
| qNvSMy_FS | — | — | — | — | — |

---

## 11. Trace Results — Bundle A (M_C → M_A delta)

### H_error distribution

| Case | M_C H_error_mean | M_A H_error_mean | M_C H_error_at_ceil_frac | M_A H_error_at_ceil_frac | H_ERROR_CEIL live? |
|------|-----------------|-----------------|------------------------|--------------------------|-------------------|
| 9xjhi_FS | — | — | — | — | — |
| xFk7_DT_mv | — | — | — | — | — |
| nVUnxqHLr_DT | — | — | — | — | — |
| XRTnTUjU_DT | — | — | — | — | — |
| MYrVxVEM_DT | — | — | — | — | — |
| wVYSGVTT_DT_mv | — | — | — | — | — |
| qNvSMy_FS | — | — | — | — | — |
| jtYTdZm3_DT | — | — | — | — | — |

**H_ERROR_CEIL verdict (to be filled)**: live / dormant → `use_aec3_h_error_ceil` candidate added / not added.

---

## 12. Trace Results — Bundle D (M_A → M_D; trace only)

### FilterAnalyzer + usable_linear + delay events

| Case | usable_linear_frac | ul_gate3_conv_frac | ul_gate3_ext_delay_frac | FA_consistent_frac | delay_event_count | has_delay_events |
|------|-------------------|-------------------|------------------------|------------------|-----------------|-----------------|
| 9xjhi_FS | — | — | — | — | — | — |
| xFk7_DT_mv | — | — | — | — | — | — |
| nVUnxqHLr_DT | — | — | — | — | — | — |
| XRTnTUjU_DT | — | — | — | — | — | — |
| MYrVxVEM_DT | — | — | — | — | — | — |
| wVYSGVTT_DT_mv | — | — | — | — | — | — |
| qNvSMy_FS | — | — | — | — | — | — |
| jtYTdZm3_DT | — | — | — | — | — | — |

**M_full_delay gate**: cases with `delay_event_count > 0` → run M_full_delay.

---

## 13. Classification Rules

- **AEC3 live dependency**: unconditional in AEC3 production → must be in full composition ladder. Prior isolated NOSHIP labels are explicitly rescinded for all Bundle A/B/C flags.
- **AEC3 non-default / no live consumer**: field-trial or no active consumer → can be documented closed without full composition test.
- **PBFDKF structural divergence**: only valid AFTER M_D or M_full_delay tested and fails; not before.
- **Historical isolated result**: M1–M4, v3.21.7 Cat C, v3.21.15 isolated 800-case — provide attribution context only; do NOT override composition verdict.

---

*Results populated by `python/v3_21_full_composition_trace.py` run on 2026-05-26.*
