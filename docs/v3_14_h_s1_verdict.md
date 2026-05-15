# v3.14 Arc H — Sprint 09 Verdict: Huber Loss NLMS Prototype

**Date**: 2026-05-14
**Sprint**: H.S1 — Standalone Huber NLMS prototype (clipping-aware adaptation)
**Script**: `tools/research/v3_14_h_s1_huber_proto.py`
**Status**: CLOSED — mechanism falsified for saturation problem class; CONDITIONAL PASS for impulsive artifact class

---

## Background

v3.13 E5 closure established that 5 listen cases (cases 01/02/06/07/08 in
`listen/v3_12_worst_fs/`) cannot be fixed by amplitude-domain interventions
(freeze, spectral mask, saturation gating).  Arc H tests whether Huber-loss
NLMS addresses the same cases from the gradient-bounding angle.

**Hypothesis**: Hard-clipping in mic signal creates large residuals → L2 NLMS
gradient ∝ e² → runaway weight update during clipping bursts → Huber caps
gradient at δ → stabilises filter during bursts.

---

## Experiments

### Experiment 1: Synthetic clipping bench (controlled, w_true known)

- 15s sine sweep (100–3000 Hz) convolved with random FIR room filter.
- Mic = echo + noise, hard-clipped at ±0.7 amplitude.
- Clipping burst frames (>0.65): 297/1500 = 19.8%.
- Both L2 and Huber share identical μ=0.02, filter_len=256.

| Variant       | ERLE_mean (dB) | ERLE_burst (dB) | W_err_mean | W_norm_std | Huber_fire% |
|---------------|---------------:|----------------:|-----------:|-----------:|------------:|
| L2            |         +23.42 |          +20.00 |     0.9758 |     0.2160 |         N/A |
| Huber δ=0.03  |         +22.86 |          +17.84 |     0.9734 |     0.2122 |        11.4 |
| Huber δ=0.07  |         +23.29 |          +19.57 |     0.9749 |     0.2154 |         5.7 |
| Huber δ=0.15  |         +23.41 |          +19.96 |     0.9757 |     0.2159 |         1.7 |
| Huber δ=0.30  |         +23.42 |          +20.00 |     0.9758 |     0.2160 |         0.0 |
| Huber δ=0.60  |         +23.42 |          +20.00 |     0.9758 |     0.2160 |         0.0 |

**Finding**: No meaningful difference.  δ ≥ 0.30 is identical to L2 (zero
clipping events in gradient domain).  Small δ values (0.03–0.07) actually
*hurt* ERLE by capping the gradient during non-burst frames, slowing
convergence.

**Root cause**: Mic is clipped at ±0.7 (bounded).  Reference (lpb) is
unclipped.  Therefore residual `e = mic - ŷ` is bounded at
|e| ≤ |mic| + |ŷ| ≤ 0.7 + ||w||·||x||.  In steady-state, ||w||·||x|| is
small, so residuals don't exceed ~0.7–1.0.  No impulsive events → Huber
δ ≥ 0.30 never fires.  The gradient is already bounded by the signal level.

---

### Experiment 2: Real listen cases (cases 01/02/07)

Selected cases with mic peak = 1.0000 (hard-clipped at full-scale), which
were the primary v3.13 E5 targets.

| Case | Variant      | ERLE_mean (dB) | ERLE_burst (dB) | W_norm_std | Huber_fire% | Burst frames |
|------|--------------|---------------:|----------------:|-----------:|------------:|-------------:|
| 01   | L2           |          +1.03 |           +2.48 |     1.1429 |         N/A |   2/2167 (0.1%) |
| 01   | Huber δ=0.03 |          -1.04 |           +0.54 |     1.2404 |        33.3 |              |
| 01   | Huber δ=0.60 |          +0.96 |           +2.34 |     1.1473 |         1.6 |              |
| 02   | L2           |          +4.25 |           +5.63 |     4.2248 |         N/A |   9/2169 (0.4%) |
| 02   | Huber δ=0.03 |          +3.21 |           +4.10 |     3.1738 |        46.7 |              |
| 02   | Huber δ=0.60 |          +4.23 |           +5.58 |     4.2446 |         1.3 |              |
| 07   | L2           |          +2.92 |           +2.95 |     2.2111 |         N/A |   4/2515 (0.2%) |
| 07   | Huber δ=0.03 |          +1.63 |           +2.59 |     1.8078 |        65.4 |              |
| 07   | Huber δ=0.60 |          +2.69 |           +2.86 |     2.3751 |         6.9 |              |

**Finding**: Huber consistently degrades ERLE across all listen cases.  Three
independent effects explain this:

1. **Huber_fire% >> burst_frame%**: Case 02 fires Huber on 46.7% of samples
   with δ=0.03, but only 0.4% of frames are burst frames.  The high fire rate
   is driven by the large residual amplitude of poorly-cancelled echo, not by
   clipping spikes.  Huber is capping the gradient on the majority of
   *non-burst* frames, slowing convergence.

2. **Burst frame count is negligible**: 0.1–0.4% of frames contain any sample
   above 0.95.  Even if Huber perfectly protected those frames, the effect on
   total ERLE would be < 0.01 dB.

3. **The listen cases' pathology is not gradient-divergence**: They fail due
   to (a) large delays (case 01: 681 ms GCC-PHAT offset), (b) severe
   nonlinear distortion (case 07: "severe NL" in listen notes), (c) path
   movement (FS_movement cases).  None of these are addressable by
   gradient bounding.

---

### Experiment 3: Adversarial — Impulsive spikes (correct use case)

To validate Huber does work when it should, a controlled experiment added
50 random impulse spikes (amplitude 5–20×) to a clean signal.  This is the
scenario where Huber is designed to help.

| Variant       | ERLE_mean (dB) | W_norm_max | W_err_final |
|---------------|---------------:|-----------:|------------:|
| L2 (with spikes) |       +28.45 |      0.707 |      0.6606 |
| Huber δ=0.15  |         +28.94 |      0.412 |      0.6613 |
| Huber δ=0.30  |         +28.95 |      0.416 |      0.6595 |
| L2 (clean)    |         +30.43 |      0.416 |      0.6597 |

**Finding**: Huber DOES help for impulsive spikes.  W_norm_max drops from 0.707
to 0.412 (42% reduction), confirming the gradient-bounding mechanism works as
designed when residuals contain true super-Gaussian outliers (spikes >> 1).
However, mic saturation at ±1.0 is NOT this regime.

---

## δ-Parameter Scan Results

δ scan over [0.03, 0.07, 0.15, 0.30, 0.60]:

```
δ=0.03: score=-1.070  (hurts: clips gradient on normal-amplitude frames)
δ=0.07: score=-0.806  (hurts: still clips too aggressively)
δ=0.15: score=-0.562  (hurts: clips ~15% of samples on listen cases)
δ=0.30: score=-0.308  (near-neutral on synthetic; slight hurt on real)
δ=0.60: score=-0.101  (near-zero effect — almost never clips)
```

Monotone: larger δ = less harm, converging to L2 behaviour.  No δ value
shows a net gain on the v3_12_worst_fs listen cases.

---

## Root Cause Analysis: Why Huber Doesn't Fix Saturation

The Huber gradient bound is:

    g(e) = δ·sign(e)   when |e| > δ

For this to be beneficial, we need residuals `e` that are:
1. Transiently large (spikes) — bounded by Huber → filter recovers
2. Normally bounded (most of the time) — Huber transparent (g = e)

The AEC saturation problem does NOT fit this profile:

- Mic saturates at ±1.0.  Residual = mic − ŷ ≤ 1.0 + ||w||·||x||.
  This is bounded, not impulsive.
- The residual floor during clipping is caused by nonlinear distortion
  (clipping introduces harmonic overtones not present in lpb) — a *model
  mismatch*, not a gradient magnitude problem.
- Huber reduces gradient on large-residual frames (which are the majority
  in poorly-converged cases), slowing adaptation.  This is counterproductive
  when convergence is the actual bottleneck.

The correct tool for bounded nonlinear distortion is **Volterra expansion**
(v3.14 primary arc) — it augments the linear model with polynomial basis
functions that can represent the clipping nonlinearity.

---

## Conclusions and Recommendations

### Arc H verdict: CLOSED — mechanism falsified for saturation problem class

Huber-loss NLMS is NOT the right tool for AEC echo residuals caused by
loudspeaker / microphone saturation.  The mechanism assumptions
(impulse-like gradient blowup) do not match the actual distortion profile
(bounded nonlinear residual floor).

### δ-range recommendation for H.S2

**Recommendation: Do NOT proceed to H.S2 (aec.py wire sprint).**

No δ value shows ERLE improvement on the target listen cases.  The
mechanism is falsified at the prototype stage.  Proceeding to wiring would
waste a sprint.

If Huber is revisited in a different context (e.g., impulsive noise
robustness for non-AEC applications, or as a safety layer against spike
artifacts in preprocessing), the recommended δ range would be
**0.15–0.30** — these values:
- Fire only on true outlier residuals (|e| > 0.15 or 0.30)
- Are transparent to the normal adaptation signal
- Show W_norm_max reduction without ERLE regression in the impulse experiment

### Next action

The v3.14 primary arc remains **Volterra non-linear inverse filter** on
`feature/v3.14-volterra`.  Arc H closes here.

---

## Artefacts

- Script: `tools/research/v3_14_h_s1_huber_proto.py`
- Raw results: `/tmp/v3_14_h_s1_results/` (synthetic + listen JSON, delta recommendation JSON)
- Runtime: ~16s total (all 3 experiments, 3 listen cases, 5 δ values each)
