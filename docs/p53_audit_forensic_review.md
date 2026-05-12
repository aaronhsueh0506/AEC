# P53 Audit Forensic Review — T0.E outcome scrutiny

**Date**: 2026-05-12
**Source data**: `/tmp/p53_audit/audit_summary.json` + per-case npz dumps
**Scope**: forensic answer to three pre-registered questions; no audit re-run, no T0 retune, no R3 implementation.

---

## §1 Stable cohort `r_voice = 5.67` — why so far from textbook ≈ 1?

### §1.1 Three competing hypotheses

| ID | Statement | Verdict |
|---|---|---|
| α | Audit captures wrong quantity (P_updated instead of P_predicted) | **REJECTED** |
| β | `R = EMA(error_psd)` double-counts innovation into denominator | **PARTIALLY CONFIRMED** (correlated noise) |
| γ | Production R is heuristic, not Kalman measurement noise | **CONFIRMED** |

### §1.2 Hypothesis α: P_predicted vs P_updated

The trace hook sits at [aec.py:1136](../python/aec.py#L1136), immediately after the denominator is computed and **before** the per-partition K-gain / W / P updates run. `self.P[p]` at this point is the t−1 frame's final value, which is also the t-frame's *predictor* (PBFDKF folds the additive Q into the update step `(1−KX)·P + Q_gated`, so there is no separate `P_predicted = P_prev + Q` line). My `P_diag = Σ_p self.P[p]` is therefore the right quantity. **α rejected.**

### §1.3 Hypothesis β: correlated noise in numerator and denominator

Production R chain ([aec.py:1088-1095](../python/aec.py#L1088)):

```
error_psd        = |error_spec|²                              ← innovation_power
self._error_psd  = 0.95 * self._error_psd + 0.05 * error_psd  ← same EMA as audit's C_obs
self.R           = max(self._error_psd, δ)
R_scale          = 0.1 + 0.9 * (1 - mu_mean)                  ∈ [0.1, 1.0]
self.R           = self.R * R_scale
```

The audit's `C_obs[k, t] = EMA(|error_spec|²; α=0.95)` is **bit-identical to `self._error_psd`**. So:

- `C_obs ≈ R / R_scale`
- `r = C_obs / (M + R)` where `M = far_psd · P_diag`

In the high-`M` regime: `r ≈ C_obs / M`, so the ratio reflects how much innovation power exceeds the `far_psd · P_diag` Kalman prediction. Useful diagnostic.

In the low-`M` regime: `r ≈ C_obs / R = R_scale⁻¹ ∈ [1, 10]`. The ratio collapses to a pure function of `mu_mean` (the gain controller's DT-vs-FS state), not the filter's underlying tracking quality. **This is the contamination.**

Empirically on the stable cohort (Q1 component table):

| case | C_obs (vb mean) | R | M=fP_diag | M/R | r_audit |
|---|---|---|---|---|---|
| ZtGitIxrzU0ILwu0HACaaw_doubletalk | 7.71e+00 | 2.57e+00 | 1.67e+00 | 0.65 | 1.815 |
| wr2CEenGL0Oec4b1GW5Zww_doubletalk | 8.90e-01 | 2.15e-01 | 3.30e+00 | 15.4 | 0.253 |
| ON-NE-zero (uLl640) | 1.49e-01 | 3.28e-02 | 2.12e-02 | 0.65 | 2.765 |

Cases with `M ≪ R` (NE-dominant, filter converged) produce `r > 1` purely from `R_scale⁻¹`. **β partially confirmed**: the numerator/denominator share the same EMA chain, so `r` cannot be interpreted as pure Mehra-1970 orthogonality.

### §1.4 Hypothesis γ: heuristic R is not measurement noise

Textbook Kalman `R` is **a-priori** measurement-noise variance (mic preamp + ambient acoustic noise) — *independent* of the residual. Production R is `EMA(innovation²) · R_scale`, a deliberate AEC heuristic ([aec.py:1090](../python/aec.py#L1090) — "Update measurement noise estimate from error PSD"), giving the filter an adaptive ratio for DT-vs-FS gating. **The Mehra-1970 orthogonality test premise that R is fixed is violated by design.**

### §1.5 Aggregation artifact (separate from α/β/γ)

Re-aggregating the same per-bin data four ways:

| Aggregation | stable | wildly | listener |
|---|---|---|---|
| `mean_k(C_obs/C_exp)` then `mean_t` (audit form) | **5.67** | 1.75 | 2.96 |
| `mean_t(Σ_k C_obs / Σ_k C_exp)` (voice-band integral) | 3.59 | 0.81 | 2.62 |
| `mean(C_obs) / mean(C_exp)` pooled | 1.37 | 0.58 | 0.82 |
| `mean_t(median_k(C_obs/C_exp))` (robust) | 2.89 | 0.80 | 1.82 |

The audit's mean-of-ratios is dominated by per-bin spikes (single-bin small denominators). Two NE-only stable samples (`uLl640`: 33 frames; `WnDjVFW`: 90 frames) contribute means of 20.3 and 14.6 respectively. Pooled or integral forms compress everything by 2–4×.

**Important**: under every aggregation form, **stable ≥ wildly** still holds. The T0.A signature (`wildly ≥ 3.0 AND stable < 1.5`) fails under every form, not just the audit's. The qualitative finding is robust; only the magnitudes are inflated.

### §1.6 §1 verdict

Stable `r_voice = 5.67` is not 1.0 because (a) γ holds — production R is a heuristic, not Kalman measurement noise, and (b) the mean-of-ratios aggregation amplifies low-denominator bins. Under aggregation forms that suppress this, `r_voice` lands in 1.4–3.6 on stable, which is closer to textbook 1.0 but still elevated — consistent with γ contamination on top of any genuine residual.

---

## §2 Wildly cohort `r_voice = 1.75 < stable 5.67` — what mechanism?

### §2.1 Per-cohort component magnitudes (voice band, far-active)

| Component | stable mean | wildly mean | listener mean |
|---|---|---|---|
| `P_diag` | 1.05e−01 | **5.98e−02** | 9.91e−02 |
| `R` | 6.53e−01 | 6.92e−01 | 7.66e−01 |
| `far_psd` | 2.13e+01 | **3.70e+01** | 5.28e+01 |
| `innovation_power` | 2.30e+00 | 2.10e+00 | 2.47e+00 |
| `M = far_psd · P_diag` | ≈ 2.24 | ≈ 2.22 | ≈ 5.23 |

Three observations:

1. **`innovation_power` is the same** across cohorts (2.1–2.5). Wildly is not generating systematically larger innovations on a voice-band-mean basis.
2. **`P_diag` on the wildly cohort is 40 % smaller** than stable. The PBFDKF state covariance on the wildly cohort tail is *less inflated*, not more, in the long-run mean.
3. **`far_psd` on wildly is 1.75× stable**. Wildly cohort has more far-end activity, which would naturally invite larger `M = far_psd · P_diag`. The smaller `P_diag` partially compensates.

### §2.2 Why is `P_diag` smaller on wildly?

Two mechanisms in production that pull `P` down on the wildly cohort tail:

1. **`PathChangeRegimeHandler.main_paused`** ([aec.py §PathChangeRegimeHandler](../python/aec.py#L3577)) pauses main filter updates during catastrophic-divergence frames. When main is paused, `_update_weights` doesn't run, so the audit captures nothing for those frames — they're excluded from the `r_voice` mean. The remaining frames on wildly cases are those where the handler decided **not** to fire heavy-machinery, i.e. the post-recovery / quiescent frames where `P` has settled.
2. **`reverse_copy` from shadow** copies the more-converged shadow filter's coefficients into main, which incidentally also reflects in lower steady-state `P` on the post-copy frames captured here.

Both mechanisms mean the audit, sampled across whole recordings, sees mostly the *post-defence* steady state on wildly cases — not the divergence transients. The divergence frames themselves are either masked (paused) or short-lived.

### §2.3 Implication for Phase L premise

Direction 1 (Adaptive Q via Mehra 1970) hypothesises that wildly cases need *larger* Q to allow `P` to grow and track faster. The data shows the opposite long-run behaviour: production already *suppresses* `P` on wildly cases through the handler, and the audit cannot see Q-rate effects on the transients because those frames are gated out.

Direction 2 (Strong Tracking P-inflation) similarly assumes innovation > expected → inflate `P`. The audit can't observe this signature because (a) innovation on wildly cohort matches stable, (b) `P` is already smaller, and (c) the handler has already taken corrective action on the very frames where the signature would appear.

Direction 4 (IMM bank) hypothesises that distinct regimes have distinct `(Q, R)` settings. The audit shows the production system has effectively already implemented a *binary* regime switch via the handler (`main_paused` vs run-normally), and the regime labels don't track this binary cleanly (per P52 §3a: only 1 of 7 wildly cases has handler heavy-fire). The IMM premise of clean regime separation is contradicted by the same A.0R.7 evidence that the design lock §0.2 cited.

**Direction 1/2/4 share a premise — that an adaptive linear-filter modification can do work the production handler is already doing.** The forensic data shows the handler-controller joint system has already taken the lever; the audit is observing post-intervention steady state, not unsolved Q-rate gaps.

### §2.4 §2 verdict

`wildly < stable` is not a bug in the audit and not a refutation of Q-underestimation in some abstract textbook PBFDKF. It is direct evidence that the **production PBFDKF + `PathChangeRegimeHandler` joint system** has already absorbed the corrective action that Directions 1/2/4 would add. Adding adaptive-Q or P-inflation on top would be redundant at best and conflicting at worst (Direction 1's larger Q would fight the handler's reverse_copy; Direction 2's inflated P would fight the handler's main_paused).

---

## §3 Catastrophe case `qNvSMyU... r_voice = 0.37` — does the metric scope cover divergence?

### §3.1 Per-100-frame trajectory (voice band means)

```
    t     P_diag    inn_pwr      C_obs       M=fP          R        r
    0  6.000e-02  2.051e-04  2.051e-04  6.226e-04  7.798e-03    0.024
  100  1.303e-01  1.158e+00  2.233e+00  5.982e-01  8.294e-01    1.564
  200  4.588e-02  1.137e-02  5.427e-02  1.154e+00  5.427e-02    0.045
  300  4.359e-02  3.537e-06  5.114e-04  5.354e-01  2.921e-04    0.001
  ...
 1400  8.943e-02  3.526e-03  3.330e-01  4.151e-02  8.242e-02    2.687
 1500  4.312e-02  1.593e-02  1.158e-02  4.559e+00  1.158e-02    0.003
```

**Top-5 innovation-power frames** (where divergence is most acute):

| t | `P_diag` | `inn_pwr` | `M = f·P` | `r` |
|---|---|---|---|---|
| 26  | 6.72e−02 | 1.53e+02 | 1.33e+01 | 0.69 |
| 757 | 4.52e−02 | 1.27e+02 | 1.53e+00 | 2.13 |
| 41  | 8.79e−02 | 9.72e+01 | 5.21e−01 | 1.17 |
| 27  | 6.85e−02 | 8.78e+01 | 1.33e+01 | 0.77 |
| 758 | 4.83e−02 | 8.67e+01 | 2.47e+00 | 1.79 |

### §3.2 Two findings

1. **Even at the worst-divergence frames, `r ∈ [0.7, 2.1]`** — never >> 3. The catastrophic-blowup frames have `inn_pwr ~ 150`, but `M = far_psd · P_diag` also rises to 1–13, so the ratio stays bounded.
2. **The case mean `r = 0.37` is dominated by the quiescent post-recovery frames** (300, 400, 500, 600, …) where `inn_pwr ≈ 10⁻⁵`. The catastrophe events at t=26-27, 41, 757-758 are 5 frames out of 1749 — about 0.3 % of the recording.

The whole-case `r_voice` average is therefore an inherently poor detector for transient divergence. Even a perfect Q-underestimation diagnostic would be washed out by the 99.7 % steady-state mass.

### §3.3 Implication for audit metric scope

The audit was designed to provide *whole-case* aggregates per the design lock §2.2 spec. Detecting transient divergence would require:

- Per-frame `r` trajectory analysis (we have this — see §3.1) and
- A *defence-event-aligned* aggregation (e.g. condition on `PathChangeRegimeHandler.fire == True` frames)

But aligning on handler-fire is exactly the regime-label arm/no-arm question that A.0R.7 already answered: only 1 of 7 wildly cases has heavy-machinery handler fires, and the trigger pattern doesn't align with the acoustic regime label. So the same A.0R.7 finding that scoped Path 2 out also limits what the audit can do.

### §3.4 §3 verdict

The audit metric **as defined by design lock §2.2 is fundamentally not the right scope** for detecting transient catastrophic divergence. It is a whole-case-mean metric; the events it would need to flag are 0.3 % of frames and are already actively gated by the production handler. Any redesigned audit that conditions on handler-fire frames would essentially re-implement A.0R.7's analysis, which has already been done and supports the §6.4 P52 Path 3 conclusion.

---

## §4 Final assessment — which outcome stands?

Mapping forensic findings to the three pre-declared outcomes:

| Outcome | Statement | Forensic verdict |
|---|---|---|
| α | Audit implementation bug → fix + re-run | **NO** — α/§1.2 rejected; aggregation form artifact is a magnitude bias but does not flip the sign |
| β | Production-controller joint system doesn't follow textbook → walk back to R3 with evidence; Directions 1/2/4 permanently scoped out | **YES** — §2.3 + §2.4 |
| γ | R estimator degenerate → P53 close with documented finding | **PARTIALLY** — γ/§1.4 confirmed, but R3 is still a constructive path on the RES side, so total close-out would discard a usable upgrade |

The dominant outcome is **β**. γ is real but doesn't justify closing the entire arc when Phase R = R3 is independently motivated by the P47.1 reverb-additive-contamination finding (carried forward in design lock §4.4 R3.4 caveat).

---

## §5 Recommendation

**Proceed to Phase R = R3 (Lee-Kim four-hypothesis SPP), Phase L permanently scoped out.**

Concrete next steps:

1. **Merge `p53-step-0-audit` → main** (carries trace hook, audit tool, listener case list, verdict doc + this forensic review).
2. **Tag `p53-step-0-closed-T0E`** on the merge commit.
3. **Fork `p53-phase-r-lee`** from the merged main.
4. **Begin R3 implementation per design lock §4.4** (R3.1 `FourHypothesisSPPEstimator`, R3.2 `BayesianGainComputer`, R3.3 wire into M2+M3, R3.4 M1 untouched / P47.1 caveat preserved, R3.5 800-case validation).
5. **Update `docs/p52_errata.md`** with entry "T0 threshold methodology scope limitation":

   P52 design lock §2.4 T0 thresholds (r_voice mean ≥ 3.0 etc.) implicitly
   assumed whole-case mean would detect Q-related signatures. P53 forensic §3
   found this scope limitation: transient divergence events are < 1 % of
   frames, washed out by quiescent mass. Future audit-style methodologies
   should either (a) condition on handler-fire-aligned frame subsets, or
   (b) use different statistical scope (max, p99, or event-conditional means).
   P52 spec NOT modified; errata documents methodology learning.

**Pre-requisite for any future Phase L**:

A forensic model of the production PBFDKF + `PathChangeRegimeHandler` joint
system's effective state-space dynamics. Without this model, any Phase L
direction risks fighting handler interventions on the same lever (Q-scale,
P-magnitude, W-norm). The forensic data in §2 of this review provides
starting evidence; a full model would require trace coverage of handler-fire
vs non-fire frame populations on the full 800-case corpus, not just the
7 wildly cases sampled here.

**Direction 1 / 2 / 3 / 4 permanently scoped out of P53.** Reopening them requires a fresh design lock that:

- Does not rely on the Mehra-1970 orthogonality test on production code (use a fixed-`R` shadow filter instead), or
- Demonstrates the production `R` heuristic can be modified without breaking the DT-vs-FS gating it currently provides (would itself be a separate arc).

These conditions are deliberately stringent; the forensic evidence shows the production PBFDKF + handler system has already taken the corrective lever Phase L was meant to add. Any new Phase L work would need to first walk back the handler intervention — that is exactly the work P52 Phase A closed via Path 3, which deliberately preserved the handler as load-bearing.

**Phase E re-scope**: with Phase L closed, Phase E becomes Phase-R-only evaluation (2 weeks). Hard-test set TE2 (no 800-case regression) and TE1 (9 listener cases) carry forward as written. The original TE3 (handler fire reduction ≥ 30 %) is **replaced** by:

**TE3′ (replaces design lock §5.4 TE3) — Handler-fire frequency parity**:
- Per regime cohort (stable / mildly / wildly): handler fire counts on Phase R vs production baseline within ±10 % per cohort.
- Pass criterion: Phase R does **not** increase handler load (R3 should be RES-layer-only, handler-neutral).
- Fail interpretation: if R3 increases handler fires, R3 has an unintended interaction with upstream filter state (architectural concern).
- Rationale: R3 modifies RES (downstream of main filter); handler operates on main filter state (upstream of RES). R3 should be handler-neutral by construction. The parity test verifies no architectural leakage. The original TE3 (≥ 30 % reduction) was premised on Phase L tracking improvement, which is now permanently scoped out per §5.

Total revised P53 wall time: 3 weeks Phase R + 2 weeks Phase E = **5 weeks**, down from 6–8 weeks originally scoped.

---

## §6 Methodology lesson (P53 lasting contribution)

Production AEC effective dynamics ≠ textbook PBFDKF. Future classical architecture work in this codebase should begin with a forensic model of the production PBFDKF + handler + R-heuristic compound system, before applying textbook Kalman / RES literature directly.

The forensic review framework demonstrated here generalises to future audits:

(a) Three pre-registered hypotheses (α/β/γ pattern) prevents post-hoc rationalization.
(b) Per-cohort component magnitude tables expose mechanism, not just symptoms.
(c) Multi-aggregation sign-robustness check (mean-of-ratios vs pooled vs integral) separates qualitative findings from quantitative noise.

Per P52 methodology lesson "post-mortem before retry", P53 adds **"production effective dynamics ≠ textbook before applying textbook"**.

These two lessons compose: when retry temptation arises after a failed test, first ask whether the test methodology assumed textbook dynamics; if yes, forensically verify those dynamics hold in production before any retry design.

---

End of forensic review.
