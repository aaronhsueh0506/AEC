# v3.21 Residual R0 Parity Audit

**Date**: 2026-05-26  
**Scope**: Residual/reverb AEC3 parity gaps that must be ruled out before classifying A2/A3 regression as PBFDKF structural (Class C). All items are read-only traces or default-OFF candidate flags; no production algorithm changes.  
**Cases**: 9xjhiFbGo06hdQ (FS_static G1) | qNvSMyUSXUyrDG (FS_static G2) | xFk7igecuke0R5 (DT_mvmt G3)  
**Constraint**: no 800-case, no merge, no version bump.  
**Status**: Trace complete (2026-05-26). All R0 items classified. A/B parity gaps ruled out as cause of A2 regression. See §Verdict.

---

## R0.1 — Filter Noise Gate Constant

### AEC3 vs Python

| Path | Python constant | int16² value | float[-1,1]² | AEC3 constant | AEC3 float | Ratio Py/AEC3 |
|------|----------------|--------------|--------------|---------------|------------|---------------|
| Filter refined mu gate | `NOISE_GATE_POWER_FLOAT` | 27509562 | 0.02562 | 20075344 int16² | 0.01870 | **1.370×** |
| Filter coarse mu gate | same constant | 27509562 | 0.02562 | 20075344 int16² | 0.01870 | **1.370×** |

### Source locations

- **Python filter gate**: `python/modules/aec3_scale.py:109`
  ```python
  NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562.0)  # 0.0256 float
  ```
  Comment: `"Fix A reverted; wVYS regression test"` — a prior correction for FFT_SIZE Parseval was reverted. This is a separate issue from AEC3 constant parity.

- **Python filter gate application**: `python/modules/filters.py:811`
  ```python
  _noise_gate = np.float32(_aec3_scale.NOISE_GATE_POWER_FLOAT)
  mu_aec3 = np.where(X2 >= _noise_gate, mu_aec3, np.float32(0.0))
  ```

- **AEC3 source**: `refined_filter_update_gain.cc` — `filter.refined.noise_gate = 20075344` int16².

### Trace results (late-phase ng_delta per band)

| Case | ng_delta_lf | ng_delta_mf | ng_delta_hf | Changes under A2? |
|------|------------|------------|------------|-------------------|
| 9xjhiFbG | +0.0153 | +0.0237 | +0.0346 | **NO** (identical M0/A2/A2_A3) |
| qNvSMyUS | +0.0163 | +0.0300 | +0.0348 | **NO** |
| xFk7igec | +0.0117 | +0.0228 | +0.0274 | **NO** |

**Key finding**: The ng_delta (Python gates this many MORE bins than AEC3) is CONSTANT across M0/A2_only/A2_A3. A2 changes the E² denominator but NOT X² values, so the filter gate fire rate is unaffected. The 1.37× mismatch is a static bias — it does NOT interact with A2 and does NOT cause the A2-specific regression.

### Candidate flag

`use_aec3_filter_noise_gate_power` (default OFF, preserves byte-equal):
```python
if cfg.use_aec3_filter_noise_gate_power:
    _noise_gate = np.float32(psd_int16_to_float(20075344.0))  # AEC3: 0.01870
else:
    _noise_gate = np.float32(NOISE_GATE_POWER_FLOAT)           # current: 0.02562
```

### Classification

**B** — Python unit bug (wrong constant for filter update gate; 1.37× too high). Fixable in v3.21 with default-OFF candidate flag. **NOT a cause of A2 regression** — static bias only, independent of A2/A3.

---

## R0.2 — Residual Echo_Model.noise_gate_power Unit

### AEC3 vs Python

| Path | Python stored | Python unit assumption | AEC3 spec | AEC3 unit | Ratio |
|------|---------------|------------------------|-----------|-----------|-------|
| Residual noise gate (nonlinear mode) | 27509562.0 | int16² (= float×32768²) | 27509.42f | unknown (need AEC3 source) | **~1000×** |

### Mode analysis (from trace)

The residual noise gate applies ONLY when `usable_linear=False` (nonlinear mode). Trace confirms:
- All 3 primary cases: `usable_linear` late-phase ≈ 1.0 (100% linear mode in steady state)
- Early frames (first 40 hops): some nonlinear frames, but gate effect only during warmup

**Implication**: For these 4 cases, the residual noise gate is irrelevant to the A2/A3 regression (late-phase behavior). The regression is in linear mode, and this gate doesn't apply.

### Classification

**B-candidate** — Unit discrepancy traced; requires AEC3 source unit verification. Effect confined to `usable_linear=False` frames (warmup only on these cases). **NOT a cause of A2 regression** — gate fires only during warmup, not in steady-state where regression occurs. Do NOT change `noise_gate_power` until unit verified.

---

## R0.3 — Nonlinear EchoGeneratingPower Window (render_pre_window_size)

### AEC3 vs Python

| Parameter | Python | AEC3 |
|-----------|--------|------|
| `render_pre_window_size` | 0 | 1 |
| `render_post_window_size` | 1 | 1 |
| `render_history_size` | 2 slots | 3 slots |

### Trace results

`x2_delta` (3-slot − 2-slot window difference) is near-zero in steady-state for these cases (continuous FS signal; no fast render transients). The window delta matters only at onset frames in nonlinear mode.

Since all primary cases are in linear mode (usable_linear=True) for the relevant regression phase, R0.3 does not affect the A2 regression outcome.

### Candidate flag

`use_aec3_echo_generating_power_window` (default OFF):
```python
if cfg.use_aec3_echo_generating_power_window:
    self._render_pre_window_size = 1
    self._render_post_window_size = 1
    # → history_size = 3
```

### Classification

**A** — AEC3 parity gap. Python pre=0 vs AEC3 pre=1. Minor for steady-state FS; relevant for transient-onset nonlinear frames. **NOT a cause of A2 regression** — effect confined to nonlinear mode frames which are not the regression locus.

---

## R0.4 — ReverbFrequencyResponse / ReverbDecay Parity

### Chain dependency

```
W (filter taps, n_partitions × n_freqs)
  → |W|² (frequency_response matrix, n_partitions × n_freqs)
  → filter_delay_blocks (direct path partition index, from FilterAnalyzer)
  → linear_filter_quality (update gate + EMA speed)
  → freq_resp_direct = frequency_response[filter_delay_blocks]
  → freq_resp_tail   = frequency_response[-1]
  → instant_decay    = sum(freq_resp_tail[1:]) / sum(freq_resp_direct[1:])
  → smoothed avg_decay += 0.2 × quality × (instant_decay − avg_decay)
  → tail_response = freq_resp_direct × avg_decay
```

### Trace results (late-phase, key fields)

| Case | Variant | delay_bl | direct_energy | tail_energy | instant_decay | avg_decay | skip_frac |
|------|---------|----------|--------------|------------|--------------|-----------|-----------|
| 9xjhiFbG | M0      | 0.0 | 1.88e+03 | 1.41e+02 | 0.0755 | 0.0759 | 0.564 |
| 9xjhiFbG | A2_only | 0.0 | **3.67e+02** | **2.23e+01** | **0.0675** | **0.0740** | **0.790** |
| 9xjhiFbG | A2_A3   | 0.0 | 3.26e+02 | 2.11e+01 | 0.0725 | 0.0858 | 0.826 |
| qNvSMyUS | M0      | 3.4 | 2.15e+00 | 1.53e+00 | 0.7967 | 0.0897 | 0.999 |
| qNvSMyUS | A2_only | **2.6** | 1.84e+00 | 1.77e+00 | **1.0406** | **0.2625** | 0.999 |
| qNvSMyUS | A2_A3   | **4.2** | 1.67e+00 | 1.72e+00 | 1.0358 | 0.1677 | 1.000 |
| xFk7igec | M0      | 4.0 | 1.57e+03 | 1.22e+03 | 0.7077 | 0.7737 | 0.919 |
| xFk7igec | A2_only | **0.0** | **6.49e+01** | **4.63e+00** | **0.0714** | **0.0733** | 0.868 |
| xFk7igec | A2_A3   | 2.2 | 9.35e+01 | 6.35e+01 | 0.5029 | 0.1044 | 0.900 |

### Per-case analysis

**9xjhiFbG (FS_static G1)**:
- `direct_energy` drops **5.1×** under A2 (1.88e3 → 3.67e2)
- `instant_decay` slightly lower (0.0755 → 0.0675, −10.6%)
- `avg_decay` nearly unchanged (0.0759 → 0.0740, −2.5%)
- `skip_frac` jumps from 56% to 79% — reverb model updates LESS under A2 (binary quality gate fires less often)
- **Consequence**: `tail_response = direct_energy × avg_decay` drops 5× because `direct_energy` dropped 5×
- `avg_decay` barely changes because the reverb model updates LESS (not more) under A2
- **Root cause of reverb tail collapse**: W magnitude reduction under A2 (direct_energy −5×), NOT filter_quality parity gap
- The filter_quality binary approximation causes SLOWER reverb adaptation (higher skip_frac) — this damps the avg_decay tracking, not accelerates it
- **R0.4 filter_quality parity gap is non-blocking**: even if we fixed it (continuous quality), the avg_decay would still be ~0.074 (matches current value) but the direct_energy would still be 3.67e2 → tail_response still 5× lower than M0

**qNvSMyUS (FS_static G2)**:
- `delay_blocks` shifts from 3.4 (M0) to 2.6 (A2) — FilterAnalyzer detects different direct-path partition under A2
- `instant_decay` HIGHER under A2 (0.797 → 1.041) — because different partition is used as "direct"
- `avg_decay` inflates 3× (0.090 → 0.263) — a few update frames (skip_frac=0.999 → ~3 frames) at high instant_decay drive avg_decay up
- `tail_response = direct_energy × avg_decay` inflates → R² inflates → over-suppression → echo distortion
- **Root cause**: FilterAnalyzer delay_blocks shift under A2 → wrong reverb direct partition → avg_decay inflates from erroneous instant_decay measurements
- FilterAnalyzer delay_blocks shift is a structural consequence of A2 changing W energy distribution → delay peak moves

**xFk7igecuke0 (DT_mvmt G3)**:
- `delay_blocks` **collapses from 4.0 to 0.0** under A2_only — FilterAnalyzer completely fails to detect the room delay
- `direct_energy` drops 24× (1.57e3 → 6.49e1) — W energy peak moves away from partition 4
- `avg_decay` drops 10× (0.774 → 0.073) — reverb model adapts to (wrong) near-zero delay
- BUT: primary G3 failure is A3 diverged false-positive (62% HF bins), not reverb model
- **Root cause**: A2 causes W to converge away from partition 4 (movement case) → FilterAnalyzer loses delay track

### Parity item: filter_quality binary vs continuous

| Source | Python | AEC3 |
|--------|--------|------|
| quality signal | binary: `1.0 if (_aec3_converged and _filter_converged_enough) else None` | continuous: `FilterAnalyzer::consistent_estimate` ∈ [0, 1] |
| smoothing_constant | `0.2 × 1.0 = 0.20` when converged | `0.2 × consistent_estimate` ∈ [0.0, 0.20] |

**Trace evidence**: Under A2, `skip_frac` increases (56%→79% for 9xjhi), meaning Python's binary gate actually causes FEWER reverb updates under A2. The reverb model adapts SLOWER, not faster, under A2. The filter_quality parity gap does NOT amplify the reverb tail collapse — it dampens it.

**Verdict on R0.4 parity**: The filter_quality binary approximation is a real parity gap. But it is NOT the blocker for A2/A3 classification. The reverb tail collapse is driven by W magnitude changes (direct_energy drops), which is a PBFDKF structural consequence.

### Classification

**A** — AEC3 parity gap (filter_quality binary vs continuous). Real parity issue that should be fixed via `FilterAnalyzer.consistent_estimate` port. **NOT a cause of A2 regression** — reverb tail collapse is driven by W magnitude reduction (structural), not by smoothing speed. Does not block A2/A3 verdict.

---

## R0.5 — SuppressionGain Per-Band Reason Histogram

### Trace results (late-phase, reason fractions + per-band R²)

| Case | Variant | G_lf | G_mf | G_hf | min_lf | min_mf | lim_lf | lim_hf | r2_lf | r2_mf | r2_hf |
|------|---------|------|------|------|--------|--------|--------|--------|-------|-------|-------|
| 9xjhiFbG | M0 | 0.783 | 0.940 | 0.554 | 0.042 | 0.003 | 0.147 | 0.315 | 9.94e+10 | 2.01e+11 | 1.23e+10 |
| 9xjhiFbG | A2_only | 0.736 | 0.886 | 0.515 | 0.042 | 0.008 | 0.162 | 0.348 | **3.99e+10** | **6.47e+10** | **4.86e+09** |
| 9xjhiFbG | A2_A3 | 0.735 | 0.881 | 0.524 | 0.046 | 0.009 | 0.158 | 0.336 | 4.23e+10 | 6.28e+10 | 4.63e+09 |
| qNvSMyUS | M0 | 0.661 | 0.832 | 0.552 | 0.021 | 0.001 | 0.121 | 0.307 | 7.39e+08 | 2.47e+08 | 1.63e+06 |
| qNvSMyUS | A2_only | 0.676 | 0.819 | 0.537 | 0.020 | 0.001 | 0.132 | 0.324 | **2.37e+09** | **2.72e+08** | **1.25e+06** |
| qNvSMyUS | A2_A3 | 0.724 | 0.879 | 0.452 | 0.022 | 0.001 | 0.149 | 0.486 | 2.12e+09 | 2.88e+08 | 1.20e+06 |
| xFk7igec | M0 | 0.846 | 0.923 | 0.768 | 0.081 | 0.004 | 0.138 | 0.192 | 2.19e+10 | 1.75e+11 | 1.01e+11 |
| xFk7igec | A2_only | 0.740 | 0.937 | 0.626 | 0.051 | 0.004 | 0.116 | 0.288 | **3.60e+09** | **9.30e+09** | **1.45e+08** |
| xFk7igec | A2_A3 | 0.718 | 0.920 | 0.567 | 0.038 | 0.004 | 0.117 | 0.333 | 1.32e+10 | 1.21e+10 | 7.31e+08 |

### Key findings

1. **G-path dominates** in all cases/variants (G_lf 0.66–0.85, G_mf 0.82–0.94). The SG regression is NOT caused by gain-rule clamping (min/max binding) — it is driven by R² changes feeding the GainToNoAudibleEcho path.

2. **R² drops under A2** for 9xjhiFbG and xFk7 (LF: −2.5× and −6×). This is the proximate cause of the echo regression — lower R² → SG opens gain → echo passes through.

3. **R² inflates under A2** for qNvSMyUS LF: +3.2× (from 7.39e8 to 2.37e9). This contradicts the expectation of "less suppression" — however, the G2 echo failure (qNvSMy Δecho=−0.328) may be driven by a different frequency band or by W-level cancellation changes, not by SG gain direction.

4. **lim_hf fraction** (HF limiter binding): slightly higher under A2 for all cases (0.315→0.348 for 9xjhi). HF limiter firing is expected behavior; slight increase is consistent with HF R² changes.

5. **min_gain not binding** for any case/band (min_lf ≤ 0.08, min_mf ≤ 0.01). The SG is not hitting the minimum gain floor — it has headroom to suppress more if R² were higher.

### Classification

**B-closed** — Reason histogram confirms R² is the proximate cause (G-path, not clamped). R² drops under A2 are explained by reverb tail collapse (from R0.4 analysis: W magnitude reduction). No new parity issue found in SG gain rules. The regression path is: A2 → W magnitude changes → reverb tail collapses → R² drops → G-path gain opens → echo leaks.

---

## Summary: R0 Parity Item Classification

| Item | Python vs AEC3 | Classification | Cause of A2 regression? | Action |
|------|---------------|----------------|------------------------|--------|
| R0.1 Filter noise gate | 27509562 vs 20075344 int16² (1.37×) | **B** — unit bug | **NO** — ng_delta constant across variants | Candidate flag `use_aec3_filter_noise_gate_power` (standalone v3.21 item) |
| R0.2 Residual noise gate unit | 27509562 vs 27509.42 (~1000× if same units) | **B-candidate** | **NO** — only affects warmup nonlinear frames | Verify AEC3 units; low priority |
| R0.3 Render pre-window | pre=0 vs AEC3 pre=1 (2 vs 3 slots) | **A** — parity gap | **NO** — only affects nonlinear mode (not regression locus) | Candidate flag `use_aec3_echo_generating_power_window` (standalone v3.21 item) |
| R0.4 ReverbFrequencyResponse filter_quality | binary 0/1 vs AEC3 continuous consistent_estimate | **A** — parity gap | **NO** — causes SLOWER not faster adaptation; not root cause | Port FilterAnalyzer.consistent_estimate (long-term; non-blocking for A2 verdict) |
| R0.5 SG reason histogram | 2-bin → extended to per-band | **B-closed** | **NO** — G-path dominates; R² is proximate cause | No action needed |

---

## R0 Trace Conclusion

**A2/A3 regression is Class C (PBFDKF structural mismatch) in isolation.**

All five R0 parity gaps have been traced and ruled out as causes:
- R0.1: static constant bias, not interaction with A2
- R0.2: confined to warmup frames; irrelevant in steady-state regression
- R0.3: confined to nonlinear-mode frames; not the regression locus
- R0.4: binary filter_quality causes SLOWER reverb adaptation under A2 (higher skip_frac), not faster; does not amplify collapse
- R0.5: R² changes drive the G-path gain; no gain-rule clamping

The actual mechanism:
1. A2 (instantaneous E² denominator) changes PBFDKF mu dynamics
2. Filter W converges to different magnitude state under A2
   - 9xjhiFbG: direct_energy drops 5× → tail_response drops 5× → R² drops 2.5× → echo leaks
   - qNvSMyUS: FilterAnalyzer delay_blocks shifts (W energy distribution changes) → wrong reverb partition → avg_decay inflates 3× → R² inflates → echo distortion
   - xFk7igec: FilterAnalyzer completely fails (delay_blocks = 0 instead of 4) under A2; A3 diverged false-positive independent mechanism
3. These are all structural consequences of A2 changing the PBFDKF update dynamics

**This is a composition-incomplete result.** The isolated Class C finding does NOT constitute NOSHIP:
- Tested without Bundle B (shadow/coarse parity) — shadow quality feeds e2_coarse, which affects A3 gate and FilterAnalyzer indirectly
- Tested without Bundle C (URO + crossfade) — output selection state
- Full AEC3 linear composition (M_B → M_C → M_A in hazy-lynx plan) may change the W trajectory and FilterAnalyzer behavior under A2
- The `docs/v3_21_full_linear_signal_flow_closure.md` dependency ladder must be executed before any NOSHIP verdict

**v3.21 must-fix items (standalone, unrelated to A2 verdict)**:
- R0.1: `use_aec3_filter_noise_gate_power` candidate flag (B-confirmed)
- R0.3: `use_aec3_echo_generating_power_window` candidate flag (A-confirmed)
- R0.4: FilterAnalyzer.consistent_estimate port (A-confirmed; long-term item)
