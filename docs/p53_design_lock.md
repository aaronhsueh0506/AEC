# P53 Design Lock — Non-NN PBFDKF Tracking + RES Replacement (Classical-Only)

**Status**: Design lock. Branch fork from main HEAD (post p52-phase-b-closed) only after Step 0 evidence audit complete.
**Scope**: PBFDKF main internals (preserve PBFDKF as filter class) + RES architecture replacement (Module 1-5 can be redesigned, leveraging P52 Phase B refactor).
**Out of scope**:
- Any neural network / learned component (user constraint)
- OM-LSA-based RES (user reserves OM-LSA for downstream noise reduction)
- ShadowCopyController / PathChangeRegimeHandler removal (P52 Phase A confirmed load-bearing; preserved untouched)
- Trace chain P-number iteration pattern (anti-loophole)

---

## 0. Context

### 0.1 What P52 left

- Main = PBFDKF (preserved by user constraint)
- PathChangeRegimeHandler = catastrophic-defence safety net for ~7/800 wildly_nonstationary cases (preserved)
- AcousticRegimeClassifier = analysis-only regime tagging (`stable / mildly_nonstationary / wildly_nonstationary`)
- ResFilterRefactored = 5-module byte-equal wrapper available behind `use_res_refactored` flag (default OFF, +1.48% dispatch overhead)
- 800/800 byte-identical to production. 9 listener-anchored cases unchanged.
- ResState migration (§3.4 target architecture) deferred.

### 0.2 What P52 did NOT explore (the gap P53 addresses)

Trace chain P30–P52 explored:
- RES single-axis fixes (P30–P46) — closed
- Shadow filter for reconvergence (P47–P51) — closed
- Textbook PBFDKF + shadow (P52 Phase A) — closed via Path 3
- RES modular refactor (P52 Phase B) — closed, opt-in flag

**Never explored**:
1. **Adaptive process noise covariance Q estimation** in main PBFDKF (Mehra 1970 onward, AQ-VBAKF 2023)
2. **Strong tracking fading factor** on state covariance P (Zhou & Frank 1996)
3. **Interacting Multiple Model (IMM)** PBFDKF bank for regime switching (Blom & Bar-Shalom 1988)
4. **Variational Bayesian AEC** joint linear + nonlinearity (Malik & Enzner 2013)
5. **RES architecture replacement** by Habets 2008 joint dereverb + RES, Faller-Chen 2005 spectral envelope AES, Lee & Kim 2007 four-hypothesis SPP, Hoshuyama nonlinear RES, coherence-based SPP

### 0.3 Innovation sequence statistics — the missing evidence

The P52 post-mortem (`docs/p52_a0_postmortem.md`) found wildly_nonstationary acoustic regime where `W L2 0.03 → 0.78, ERLE 0 → -27 dB` — textbook divergence into anti-correlated output.

**What was never measured**: innovation sequence statistics on the cohort.
- `innovation[k, t] = mic[k, t] − predicted_echo[k, t]`
- Innovation covariance `C_obs[t]` over sliding window
- Expected covariance `C_exp[t]` from PBFDKF Kalman recursion (`X P X* + R`)
- Orthogonality test ratio `r[t] = trace(C_obs) / trace(C_exp)`

Theory:
- `r ≈ 1` — Kalman model assumptions hold (Q, R correctly specified)
- `r > 1` — innovation larger than expected; **Q likely underestimated** (state model too rigid for actual path variability)
- `r < 1` — innovation smaller than expected; over-fitting risk

This single measurement disambiguates which P53 direction is the right primary commit. Without it, choosing between Direction 1 (adaptive Q) vs Direction 4 (IMM) vs RES replacement is speculation.

### 0.4 References (all non-NN, all classical)

**Linear filter directions**:
- Mehra (1970) IEEE TAC "On the identification of variances and adaptive Kalman filtering"
- Yang, Wu, Jiang (2017) IEEE TASLP "Frequency-Domain Adaptive Kalman Filter With Fast Recovery of Abrupt Echo-Path Changes"
- Zhou & Frank (1996) Int. J. Adaptive Control "Strong tracking filtering of nonlinear time-varying stochastic systems"
- Hu et al. (2018) "Adaptive strong tracking Kalman filter"
- Blom & Bar-Shalom (1988) IEEE TAC "Interacting multiple model algorithm"
- Gao et al. (2017) Int. J. Control Auto. Sys. "Interacting Multiple Model estimation-based adaptive robust UKF"
- Wang (2023) Electronics "Self-Tuning Process Noise in Variational Bayesian Adaptive Kalman Filter"
- Malik & Enzner (2013) IEEE TSP "A Variational Bayesian Learning Approach for Nonlinear Acoustic Echo Control"
- Schrammen, Kühl, Markovich-Golan, Jax (2019) ICASSP "Efficient nonlinear acoustic echo cancellation by dual-stage multi-channel Kalman filtering"

**RES directions**:
- Habets, Gannot, Cohen (2008) IEEE TASLP "Joint Dereverberation and Residual Echo Suppression of Speech Signals in Noisy Environments"
- Faller & Chen (2005) IEEE TSAP "Suppressing Acoustic Echo in a Spectral Envelope Space"
- Lee & Kim (2007) "A Statistical Model-Based Residual Echo Suppression" (four-hypothesis SPP)
- Hoshuyama & Sugiyama "Acoustic echo suppressor based on frequency-domain model of highly nonlinear residual echo"
- Park & Chang (2008) IEEE SPL "Frequency domain acoustic echo suppression based on soft decision"
- Park et al. (2020) "Residual Echo Suppression Considering Harmonic Distortion and Temporal Correlation"
- Schwartz, Braun, Gannot, Habets (2015) "Maximum likelihood estimation of late reverberant PSD"
- Lebart, Boucher, Denbigh (2001) "A new method based on spectral subtraction for speech dereverberation"

---

## 1. Phase Overview

| Phase | Goal | Time-box | Hard Tests | Gating |
|---|---|---|---|---|
| Step 0 | Innovation audit on 9 listener + 7 wildly cases | 1 day | T0 (binary decision tree) | Read-only, no code change |
| Phase L | Linear filter tracking improvement (1 direction selected by T0) | 3–4 weeks | TL1–TL4 | Gated on T0 outcome |
| Phase R | RES replacement (1 direction selected by T0 + Phase L outcome) | 4–6 weeks | TR1–TR4 | Parallel with Phase L after Step 0 |
| Phase E | Combined evaluation | 2 weeks | TE1–TE3 | Gated on Phase L + R both pass |

Total wall time: 9–13 weeks. Phase L and R parallel after Step 0.

---

## 2. Step 0 — Innovation Audit (Evidence Collection)

### 2.1 Goal

Measure innovation sequence statistics on 9 listener-anchored cases + 7 wildly_nonstationary cohort (from A.0R.7) to:
1. Confirm or refute Q-underestimation hypothesis on cohort
2. Decide which linear filter direction (1, 2, 3, or 4) to commit
3. Decide whether RES replacement should target reverb-dominant (Direction R1) or path-change-dominant (Direction R2) failure mode

### 2.2 Measurement spec

For each frame `t` of each case, compute per frequency bin `k`:

```
innovation[k, t] = mic_freq[k, t] − sum_p X[k, p, t] * H_main[k, p, t-1]
                   (i.e., the prediction residual before main filter update)

innovation_power[k, t] = |innovation[k, t]|²

C_obs[k, t]  = EMA(innovation_power[k, t], alpha=0.95)
               (observed innovation variance, EMA-smoothed)

C_exp[k, t]  = far_psd[k, t] * P_diag[k, t] + R[k, t]
               (expected innovation variance per Kalman theory)

r[k, t]      = C_obs[k, t] / max(C_exp[k, t], eps)
               (orthogonality ratio)

r_voice[t]   = mean(r[k, t]) over voice band bins (300–3000 Hz)
```

Per case, aggregate:
- `r_voice` mean, median, p90, p99 over far_active frames
- `r_voice` mean conditional on regime_class (stable / mildly / wildly)
- Per-frame trajectory plot

### 2.3 Implementation tasks

#### Task 0.1: Read-only trace flag

Add `trace_p53_innovation = bool` (default False) to AecConfig. When enabled, in PBFDKF main filter update step, compute and dump:
- `innovation[k, t]` (complex)
- `innovation_power[k, t]`
- `P_diag[k, t]` (already computed in main; just expose)
- `R[k, t]` (already computed)
- `far_psd[k, t]` (already computed)

Dump to npz per case under `/tmp/p53_audit/`.

**No production code change**. Trace flag default OFF. Logic of PBFDKF unchanged.

#### Task 0.2: Analysis tool

Pure post-processing tool reading npz dumps:
```
tools/research/p53_innovation_audit.py
```

Compute aggregates per spec §2.2. Output:
- `docs/p53_innovation_audit.md` — narrative + tables
- `/tmp/p53_audit/audit_summary.json` — per-case stats

#### Task 0.3: Run audit on target cases

- 9 listener-anchored windows (from trace chain anchor list)
- 7 wildly_nonstationary cases (from A.0R.7 §"wildly cohort" list)
- 12 stable cohort baseline cases (random sample, same seed as A.0R.8 for reproducibility)

Total 28 case runs. Estimated 30–60 minutes runtime.

### 2.4 Decision tree (T0)

Based on per-case `r_voice` aggregates:

```
T0.A — Q underestimation confirmed on wildly cohort:
  r_voice mean on wildly cases ≥ 3.0 AND on stable cases < 1.5
  → Phase L commits Direction 1 (Adaptive Q estimation)
  → Phase R commits Direction R3 (Lee & Kim four-hypothesis SPP) or R1 based on reverb evidence

T0.B — Innovation large but bin-localized (path change signal):
  r_voice mean on wildly cases ≥ 3.0, but per-bin distribution shows
  isolated spikes rather than uniform elevation
  → Phase L commits Direction 2 (Strong Tracking Filter, P inflation)
  → Phase R commits Direction R2 (Faller-Chen spectral envelope AES)

T0.C — Regime-class-dependent innovation magnitude:
  r_voice mean differs significantly across regime classes
  (stable < 1.5, mildly 1.5–3, wildly > 3)
  → Phase L commits Direction 4 (IMM PBFDKF bank with N=3 regimes)
  → Phase R commits Direction R1 (Habets joint dereverb + RES)

T0.D — Innovation small / r_voice ≈ 1 everywhere:
  No Q underestimation signal
  → Phase L commits Direction 3 (Variational Bayesian linear + nonlinearity)
  OR Phase L close, Phase R only
  → Phase R commits Direction R1 (Habets joint dereverb + RES) — focuses on RES side

T0.E — Mixed signal / ambiguous:
  Outcomes do not cleanly fit A/B/C/D
  → Close P53 Phase L (no clear evidence path)
  → Phase R commits Direction R3 (Lee & Kim) as least-risky RES upgrade
```

### 2.5 Anti-loophole

- Step 0 is **read-only**. No production code logic change.
- T0 thresholds (`r_voice ≥ 3.0`, `< 1.5`, etc.) are **locked at design-lock signing**. No mid-audit threshold tuning.
- If T0 outcome falls in T0.E (ambiguous), commit T0.E branch. Do not re-run audit with different windowing / band / EMA parameters hoping for cleaner result.

### 2.6 Time-box

1 day total (Tasks 0.1 + 0.2 + 0.3 + verdict writeup).

---

## 3. Phase L — Linear Filter Tracking (one direction selected by T0)

### 3.1 Direction 1 — Adaptive Q Estimation (T0.A outcome)

**Math** (innovation-based Q adaptation, Mehra 1970 style):

```
For each bin k, each time t:
  Q_adaptive[k, t] = alpha_q * Q_adaptive[k, t-1] +
                     (1 - alpha_q) * max(0, C_obs[k, t] - C_exp_minus_Q[k, t])

  where:
    C_exp_minus_Q[k, t] = far_psd[k, t] * P_diag[k, t] + R[k, t] − Q[k, t-1]
                          (expected variance contribution from non-Q sources)
    alpha_q ∈ [0.95, 0.99]  (EMA smoothing; initial 0.97)

Replace Q_main[k] (currently fixed) with Q_adaptive[k, t]
in PBFDKF Kalman recursion.
```

Bounds:
- `Q_adaptive ∈ [Q_min, Q_max]`, `Q_min = Q_current`, `Q_max = Q_current * 100`
- Prevents Q from going negative (always ≥ baseline) or exploding

#### Tasks (Direction 1)

- L1.1: Implement `AdaptiveQEstimator` class consuming innovation + P_diag + R per frame
- L1.2: Wire into PBFDKF main update step, behind flag `use_adaptive_q` (default False)
- L1.3: Synthetic test harness:
  - 50 sequences with abrupt path change @ 10s
  - Stable, mildly, wildly synthetic conditions (3 × 50 = 150 sequences)
- L1.4: 800-case shadow run (adaptive Q tracked but original Q used in production decision) — collect statistics on adaptive Q trajectory
- L1.5: 800-case active run (`use_adaptive_q=True`)

#### Hard tests (Direction 1)

- **TL1**: Synthetic harness reconvergence
  - Mean reconvergence time on wildly synthetic ≤ 0.6 × baseline
  - Mean ERLE on stable synthetic within 0.5 dB of baseline (no regression)

- **TL2**: 800-case mean ERLE
  - ≥ 760/800 cases within ±0.5 dB of baseline mean ERLE
  - ≥ 50/800 cases improve ≥ 1 dB

- **TL3**: 9 listener-anchored cases
  - Filter state distribution (mature / coarse_learning / ...) on these cases shifts ≥ 10 percentage points toward `mature` or `refined_usable`

- **TL4**: PathChangeRegimeHandler interaction
  - Handler fire counts on the target case (qNvSMyU...) ≤ pre-Adaptive-Q baseline fire counts
  - (i.e., adaptive Q should reduce or at minimum not increase need for catastrophic defence)

#### Time-box

3 weeks

### 3.2 Direction 2 — Strong Tracking Filter (T0.B outcome)

**Math** (Zhou & Frank 1996):

```
After computing P_predicted in standard PBFDKF Kalman step:
  V[k, t] = innovation[k, t] * conj(innovation[k, t])  (1x1 in per-bin case)
  N[k, t] = V[k, t] − R[k, t]
  M[k, t] = far_psd[k, t] * P_predicted[k, t]
  
  fading_factor[k, t] = max(1.0, N[k, t] / M[k, t])
  
  P_inflated[k, t] = fading_factor[k, t] * P_predicted[k, t]

Replace P_predicted with P_inflated in Kalman gain computation.
```

#### Tasks (Direction 2)

- L2.1: Implement `StrongTrackingPInflator` 
- L2.2: Wire into PBFDKF main update step, behind flag `use_strong_tracking` (default False)
- L2.3–L2.5: Test harness + 800-case runs (mirror L1.3–L1.5)

#### Hard tests (Direction 2)

Same TL1–TL4 structure as Direction 1, replace "adaptive Q" with "strong tracking".

Additional:
- **TL2.5**: Fading factor distribution stability
  - On stable cohort: fading_factor mean < 1.5 (i.e., does not over-fire on normal data)

#### Time-box

2 weeks (lighter than Direction 1 — single multiplier inside existing loop)

### 3.3 Direction 3 — Variational Bayesian Linear + Nonlinearity (T0.D outcome)

Implement Malik & Enzner (2013) framework. Composite state-space:
```
state = [echo_path_coefficients, nonlinearity_expansion_coefficients]
```
Joint variational lower bound maximization.

**Cost**: large. Estimated 6 weeks. Only commit if T0.D outcome AND Phase R direction is small (R3).

#### Time-box

6 weeks. **High-cost direction**, only if T0.D + minimal Phase R load.

### 3.4 Direction 4 — IMM PBFDKF Bank (T0.C outcome)

**Architecture** (Blom & Bar-Shalom 1988):

```
N = 3 PBFDKF instances:
  Model 1: Q_stable, R_normal    (matches stable regime)
  Model 2: Q_mildly, R_normal    (Q_mildly = 3 × Q_stable)
  Model 3: Q_wildly, R_normal    (Q_wildly = 10 × Q_stable)

Per-frame mixing:
  1. Mixing probabilities mu_ij = Markov transition matrix × posterior probs
  2. Mixed initial conditions for each model
  3. Each model runs Kalman step → likelihood Lambda_i
  4. Update mode probabilities mu_i ∝ Lambda_i * prior_i
  5. Output: probability-weighted state estimate

Markov transition matrix initialization:
  T = [[0.95, 0.04, 0.01],
       [0.05, 0.90, 0.05],
       [0.01, 0.04, 0.95]]
```

Output to RES: probability-weighted residual echo PSD.

**Memory cost**: 3× per-bin state covariance + 3× filter coefficients.
**Compute cost**: 3× Kalman recursion + mixing logic. Estimated +200% PBFDKF cost.

#### Tasks (Direction 4)

- L4.1: Implement `IMMPBFDKFBank` with 3 sub-filters
- L4.2: Mixing logic + mode probability tracking
- L4.3: Wire into AEC, behind flag `use_imm_bank` (default False)
- L4.4–L4.6: Test harness + 800-case runs

#### Hard tests (Direction 4)

- **TL1–TL4** as in Direction 1
- **TL5**: Mode probability dynamics
  - On wildly cases, mode 3 probability dominates ≥ 60% of frames
  - On stable cases, mode 1 probability dominates ≥ 90% of frames
  - On mildly cases, no single mode dominates (no mode > 80% mean)
- **TL6**: Compute cost
  - Per-frame runtime ≤ 3× single-PBFDKF baseline (acceptable since memory matters more for embedded; if user signals compute budget, may need 2-model variant)

#### Time-box

4 weeks

---

## 4. Phase R — RES Replacement (one direction selected by T0)

### 4.1 Common Phase R foundation

Reuse P52 Phase B modular refactor as Module 1–5 interface. Phase R replaces Module logic, not interface. Specifically:

- **Module 1 (ResidualEstimator)** signature unchanged; internal logic replaced
- **Module 2 (GainComputer)** signature unchanged; internal logic replaced
- **Module 3 (SpectralShaper)** + **Module 4 (TemporalSmoother)** + **Module 5 (NoiseFloorAndCng)** may be reduced to no-op or simplified depending on Phase R direction

This means Phase R **builds on Phase B success**, doesn't redo it. Phase B's `use_res_refactored` flag mechanism reused.

### 4.2 Direction R1 — Habets Joint Dereverb + RES (T0.C or T0.D outcome)

**Background**: Trace chain P47.1 identified reverb additive contamination at `aec.py:2394` (`residual_echo_psd += reverb_gain * reverb_psd * reverb_gate`). Current Module 1 adds reverb PSD heuristically. Habets 2008 provides principled estimator for late reverberant spectral variance.

**Math** (Lebart 2001 + Habets 2008):

```
Late reverberation power spectral density model:
  P_late[k, t] = P_early[k, t-N_R] * exp(-2 * delta * N_R * T_s)

  where:
    delta = 3 * ln(10) / T60  (decay constant)
    T60   = reverberation time (estimable from impulse response or assumed)
    N_R   = number of frames after direct path (typically 50 ms / T_s)
    T_s   = STFT hop time

Postfilter gain:
  G_optimal[k, t] = SNR_early[k, t] / (SNR_early[k, t] + 1)
  SNR_early[k, t] = P_speech[k, t] / (P_late[k, t] + P_residual_echo[k, t] + P_noise[k, t])
```

Joint estimation of `P_speech`, `P_late`, `P_residual_echo`, `P_noise` using maximum likelihood from microphone PSD + linear filter output.

#### Tasks (Direction R1)

- R1.1: Implement `LateReverbPsdEstimator` (Lebart 2001 model)
- R1.2: Implement `JointMlPsdEstimator` for {speech, late, echo, noise}
- R1.3: Replace Module 1 (`ResidualEstimator`) logic with Habets joint estimator
- R1.4: Replace Module 2 (`GainComputer`) logic with G_optimal formula
- R1.5: Simplify Module 3-5 (most ad-hoc smoothing becomes redundant under principled Wiener gain)
- R1.6: Synthetic + 800-case validation

#### Hard tests (Direction R1)

- **TR1**: 800-case stability
  - Mean ERLE within ±1 dB of baseline on ≥ 95% cases
- **TR2**: 9 listener-anchored cases NE preservation
  - Subjective listener evaluation: NE clarity equal-or-better than baseline on ≥ 6/9 windows
  - (subjective bar; trace chain anchor source)
- **TR3**: High-reverb subset improvement
  - On cases with estimated T60 > 400ms, mean RES output PESQ improves ≥ 0.1
- **TR4**: No NE damage regression on FS-only frames
  - FS frames: RES gain ≥ 0.95 mean on far-active-only frames (no spurious NE-like suppression)

#### Time-box

5 weeks

### 4.3 Direction R2 — Faller-Chen Spectral Envelope AES (T0.B outcome)

**Background**: Faller & Chen (2005) AES does NOT estimate impulse response. It estimates **spectral envelope** of echo using short-time spectral modification. Envelope changes slowly even when underlying path is abrupt — fits T0.B's "path-change-dominant" scenario.

**Math** (simplified):

```
For each STFT bin k, time frame t:
  D_envelope[k, t] = G_aes[k, t] * X_envelope[k, t]
  
  G_aes[k, t]      = adaptive coupling gain, EMA-updated:
                     G_aes[k, t] = alpha_g * G_aes[k, t-1] +
                                   (1-alpha_g) * |E[k, t]| / max(|X_envelope[k, t]|, eps)
                     (where E is post-PBFDKF residual; X_envelope is far-end envelope)
  
  Suppression gain:
    H_supp[k, t] = max(epsilon, 1 − D_envelope[k, t] / max(|Y[k, t]|, eps))^beta
    beta ∈ [1.0, 2.0]
  
  Output: H_supp[k, t] * Y[k, t]
```

This **replaces Module 1+2** entirely. Module 3-5 remain for smoothing + NFL + CNG (lighter touch).

#### Tasks (Direction R2)

- R2.1: Implement `SpectralEnvelopeEstimator`
- R2.2: Implement `EnvelopeBasedSuppressionGain`
- R2.3: Replace Module 1+2 logic
- R2.4: Tune `alpha_g`, `beta` on synthetic harness (these are R2-internal, locked once Phase R hard tests start)
- R2.5: 800-case validation

#### Hard tests (Direction R2)

- **TR1**: 800-case stability (as in R1)
- **TR2**: 9 listener-anchored cases NE preservation
- **TR3**: Path-change subset improvement
  - On wildly_nonstationary regime cases, mean RES output ERLE ≥ baseline + 1 dB
  - (R2 trades steady-state ERLE for change tolerance, so improvement is on the difficult subset)
- **TR4**: FS-only frames (as in R1)

#### Time-box

4 weeks

### 4.4 Direction R3 — Lee & Kim Four-Hypothesis SPP (T0.A or T0.E outcome)

**Background**: Lee & Kim (2007) framework has four hypotheses per TF bin:
- H00: no NE, no residual echo (silence)
- H01: no NE, residual echo present (FS)
- H10: NE present, no residual echo (NE-only after perfect AEC)
- H11: NE and residual echo present (DT)

Bayes posterior computed from observation likelihoods. Suppression gain weighted by hypothesis probabilities.

**Math**:

```
For each TF bin (k, t):
  Compute likelihoods L_hij based on assumed Gaussian PDFs of NE / echo PSDs
  Compute posteriors p_hij = (priors * L_hij) / sum(...)
  
  G_lee[k, t] = p_h00 * G_silence + p_h01 * G_fs + p_h10 * 1.0 + p_h11 * G_dt
  
  where:
    G_silence ∈ [0, ε_floor]  (suppress strongly)
    G_fs ≈ Wiener gain w.r.t. echo
    G_dt ≈ adjusted Wiener weighing NE preservation
```

**Why for T0.A**: Adaptive Q in Phase L improves linear filter; Lee & Kim in Phase R uses SPP to protect NE. Combination addresses both upstream (filter convergence) and downstream (NE preservation).

**Why for T0.E (fallback)**: Lee & Kim is the lowest-risk RES upgrade. SPP framework is well-established, not requiring deep co-design with linear filter.

#### Tasks (Direction R3)

- R3.1: Implement `FourHypothesisSPPEstimator`
- R3.2: Implement `BayesianGainComputer`
- R3.3: Replace Module 2+3 (GainComputer + SpectralShaper)
- R3.4: Module 1 (ResidualEstimator) untouched (uses current logic), so reverb contamination from P47.1 remains — caveat noted
- R3.5: 800-case validation

#### Hard tests (Direction R3)

- **TR1**: 800-case stability
- **TR2**: 9 listener-anchored cases NE preservation (primary metric: NE preservation, as Lee & Kim's main strength is SPP-driven NE protection)
- **TR3**: DT subset improvement
  - On DT-rich cases, mean PESQ improves ≥ 0.05 vs baseline
- **TR4**: FS-only frames

#### Time-box

3 weeks (lighter than R1, since Module 1 untouched)

---

## 5. Phase E — Combined Evaluation

### 5.1 Gate

Phase E opens after:
- ✓ Phase L direction passes TL1–TL4 (+TL5–TL6 if Direction 4)
- ✓ Phase R direction passes TR1–TR4

If either fails, Phase E does not open. P53 closes with whichever passed shipping.

### 5.2 Goal

Quantify combined effect of Phase L + Phase R on:
- 9 listener-anchored cases (trace chain anchor regression set)
- 800-case full corpus
- Synthetic test sets (Phase L's TL1 harness + Phase R's hard test scenarios)

### 5.3 Evaluation framework

Re-use P52 §4.3 methodology with additions:
- AECMOS scoring (if speechmos venv available) — collected per A.0R errata
- Per-regime (stable / mildly / wildly) breakdown of all metrics
- PathChangeRegimeHandler fire frequency tracked
- Filter state distribution (P49 enum) tracked

### 5.4 Success criteria

- **TE1**: 9 listener-anchored cases — improvement on ≥ 5/9 by ≥ 2 dB OR subjective NE preservation improvement
- **TE2**: 800-case — no regression > 0.5 dB ERLE on ≥ 95% cases
- **TE3**: PathChangeRegimeHandler fire reduction — fire count on wildly cases reduced ≥ 30% (since Phase L should reduce need for catastrophic defence)

### 5.5 Decision tree

```
TE1 pass AND TE2 pass AND TE3 pass:
  → Ship combined as new production default
  → P53 closes with trace chain 5-case improvement

TE1 pass AND TE2 pass AND TE3 fail:
  → Ship combined; document TE3 finding
  → Handler still load-bearing; consider future investigation

TE1 fail AND TE2 pass:
  → Ship combined as opt-in (use_adaptive_q / use_strong_tracking / etc. defaults False)
  → Trace chain 5 cases remain unsolved
  → P53 closes with infrastructure improvement, no production behavior change

TE2 fail:
  → Investigate regression source
  → Roll back failing direction (L or R individually) to opt-in flag
  → Ship passing direction
```

### 5.6 Time-box

2 weeks

---

## 6. Anti-Loophole Rules

### 6.1 Constants locked at design-lock signing

All numeric constants in §2.2 (alpha=0.95), §2.4 (T0 thresholds), §3.1–§3.4 (alpha_q, q_ratio, fading_factor formula, IMM Markov transition matrix), §4.2–§4.4 (alpha_g, beta, T60 default) are locked.

Allowed: post-Phase E, opening new design lock with updated constants.
Forbidden: mid-Phase tuning.

### 6.2 No P-number sub-iteration

P53a, P53b, etc. forbidden. Phase L / R / E only.

### 6.3 PBFDKF main filter class preserved

Per user constraint, PBFDKF remains the main filter class. Modifications add behavior (adaptive Q, P inflation, IMM mixing) but do not replace PBFDKF with another filter class (e.g., RLS, NLMS).

### 6.4 PathChangeRegimeHandler untouched

P52 Phase A confirmed handler is load-bearing on ≥ 1/800 case. P53 does not modify handler logic. Phase L improvements should *reduce need for handler firing*, not replace handler.

### 6.5 OM-LSA excluded from Phase R

Per user constraint (OM-LSA reserved for downstream noise reduction). Phase R directions (R1, R2, R3) all avoid OM-LSA gain formula.

### 6.6 Each Phase has single output, single decision

- Phase L output: pass/fail of all 4 (or 6 for Direction 4) hard tests
- Phase R output: pass/fail of all 4 hard tests
- Phase E output: one decision tree branch executed

No "we passed 3 of 4, let's revisit".

### 6.7 Reuse P52 infrastructure, not methodology

**Allowed reuse**:
- 800-case corpus
- 9 listener-anchored windows
- A.0R.7 regime classifier (analysis-only)
- Phase B Module 1-5 interface (Phase R builds on)
- P49 RenderSignalAnalyzer (M1 from main)
- AECMOS scoring (when speechmos available)

**Forbidden reuse**:
- P51 shadow filter implementations
- P30-P51 hypothesis-test iteration pattern
- Trace chain P-number naming (P53 uses Step 0 / Phase L / R / E)

### 6.8 Post-mortem before retry

Per P52 methodology lesson: if any Phase fails hard tests, write post-mortem doc before any new design work. Post-mortem ≠ retry permission.

---

## 7. File / Branch / Commit Structure

### 7.1 Branch

```
main (post p52-phase-b-closed)
  └── p53-design-lock-doc       # this document
        ├── p53-step-0-audit    # Step 0 evidence collection
        ├── p53-phase-l-{adaptive_q | stf | vb | imm}  # Phase L development (one chosen)
        ├── p53-phase-r-{habets | faller | lee}        # Phase R development (one chosen)
              └── p53-phase-e   # opens after L + R merge to main
```

Branches named after selected direction post-Step-0.

### 7.2 File layout

```
python/
  aec.py                                       # existing
  aec_shadow.py                                # from P52 (preserved)
  path_change_detector.py                      # from P52 (preserved)
  
  pbfdkf_extensions/                           # NEW Phase L
    __init__.py
    adaptive_q.py            # Direction 1
    strong_tracking.py       # Direction 2
    variational_bayesian.py  # Direction 3
    imm_bank.py              # Direction 4
    (only the selected one is implemented)
  
  res_refactored/                              # from P52 Phase B
    residual_estimator.py                      # Phase R may replace
    gain_computer.py                           # Phase R may replace
    spectral_shaper.py                         # Phase R may simplify
    temporal_smoother.py                       # Phase R may simplify
    noise_floor_cng.py                         # Phase R untouched (preserves CNG)
    res_filter_refactored.py
  
  res_replacements/                            # NEW Phase R
    __init__.py
    habets_joint.py          # Direction R1
    faller_chen_envelope.py  # Direction R2
    lee_kim_spp.py           # Direction R3
    (only the selected one is implemented)

tools/research/
  p53_step_0_innovation_audit.py
  p53_phase_l_tl1_synthetic.py
  p53_phase_l_tl2_800case.py
  p53_phase_l_tl3_listener.py
  p53_phase_l_tl4_handler_interaction.py
  p53_phase_r_tr1_800case.py
  p53_phase_r_tr2_listener.py
  p53_phase_r_tr3_subset.py
  p53_phase_r_tr4_fs_frames.py
  p53_phase_e_te1_combined_listener.py
  p53_phase_e_te2_combined_800case.py
  p53_phase_e_te3_handler_reduction.py

docs/
  p53_design_lock.md                # this document
  p53_innovation_audit.md           # Step 0 verdict
  p53_phase_l_{direction}_verdict.md
  p53_phase_r_{direction}_verdict.md
  p53_phase_e_verdict.md
```

### 7.3 Commit discipline (per P52 lesson)

Single concept per commit. Each Phase verdict written before merge. No retroactive verdict edits.

---

## 8. Execution Order Summary

```
Day 1:        Commit p53-design-lock-doc to main
              Fork p53-step-0-audit branch

Day 2:        Implement Task 0.1 (trace flag)
              Implement Task 0.2 (analysis tool)
              Run Task 0.3 (28 cases)
              Write docs/p53_innovation_audit.md
              T0 decision tree executed

Day 3:        Fork p53-phase-l-{chosen} branch
              Fork p53-phase-r-{chosen} branch (parallel)

Day 3 to Week 4-5:
              Phase L development on its branch
              Phase R development on its branch
              Independent hard tests on each branch
              Each writes own verdict doc

Week 5-6:     Merge Phase L (if pass) → main
              Merge Phase R (if pass) → main
              Tag p53-phase-l-{chosen}-closed, p53-phase-r-{chosen}-closed

Week 6:       Fork p53-phase-e from merged main

Week 7-8:     Phase E evaluation
              Decision tree execution
              Write docs/p53_phase_e_verdict.md
              Tag p53-closed
```

---

## 9. What success looks like (and what it doesn't)

### 9.1 Best case (TE1 + TE2 + TE3 all pass)

- 5 listener-anchored cases finally show ≥ 2 dB ERLE improvement
- 800-case no regression
- PathChangeRegimeHandler fire reduction shows architectural fix is real
- Production default shipped with Phase L + R combined

### 9.2 Acceptable case (TE2 pass + others mixed)

- No 800-case regression (production safe)
- Phase L + R available as opt-in flags
- Some listener cases improve, others unchanged
- Documents which specific failure mode remains

### 9.3 Honest acknowledgment

If T0 outcome is T0.E (ambiguous) and Phase R chooses R3 (least-risky), and Phase E TE1 fails (5 cases still not solved), **this is still meaningful**:

- 800-case stability improved or unchanged (TE2 pass)
- Direction tested with rigor, evidence documented
- Trace chain final closure: "5 cases require non-classical methods (NN / oracle dataset), confirmed across two design locks (P52 + P53)"

This is the strongest possible classical statement. Not failure.

---

## 10. References Summary

All references are non-NN, all classical signal processing. No reference requires training data, GPU, or learned components.

Linear filter:
- Mehra 1970 (foundational adaptive Q)
- Yang 2017 (FDKF reconvergence; informs but P53 goes beyond R-reset)
- Zhou & Frank 1996 (strong tracking)
- Blom & Bar-Shalom 1988 (IMM)
- Wang 2023 (AQ-VBAKF — modern adaptive Q)
- Malik & Enzner 2013 (variational Bayesian AEC)

RES:
- Habets 2008 (joint dereverb + RES — direct fit for P47.1 finding)
- Faller & Chen 2005 (envelope-space AES — fit for path change tolerance)
- Lee & Kim 2007 (four-hypothesis SPP)
- Lebart 2001 (reverberation tail model)
- Schwartz et al. 2015 (late reverb PSD ML estimator)

---

End of P53 design lock.
