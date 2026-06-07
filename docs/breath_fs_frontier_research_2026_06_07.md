# Breath-vs-FS Pareto frontier — literature + Speex/AEC3 dissection (2026-06-07)

Multi-agent research workflow output. Goal: break the breath-vs-FS Pareto wall
(preserve near-end breath like Speex AND suppress non-stationary FS echo like AEC3),
DSP-only. Reports 1-4 = research arms; final section = ranked candidate synthesis.

## SYNTHESIS — ranked candidates

I have enough material to synthesize. Let me produce the ranked candidate set directly.

---

# Ranked DSP-only candidates to move the breath-vs-FS Pareto frontier outward

The four reports converge on one hard truth that matches our own falsified-premise memory: **no single-frame, render-referenced feature separates near-end breath from non-stationary far-echo when the linear filter is blind.** Every method that yields a credible echo estimate does so by adapting a reference-driven model that is double-talk-toxic and must be frozen — i.e. it helps the FS bar but not breath directly. So the only genuinely NEW information available falls into four classes:

1. **Filter-STATE** (per-bin convergence / Kalman P-trace) — separable because "is the linear filter trustworthy in THIS bin" is a far/near-independent property (Report 3 P1, Report 4 §2/§6).
2. **TEMPORAL context** of the echo's own statistics — the residual is a filtered/distorted slow render envelope; breath is not predictable from the render's recent past (Report 2 §1, §4).
3. **Different DOMAIN** that survives MSC≈0 — envelope/magnitude (phase-robust) and near-end-side higher-order statistics that never touch the render (Report 2 §5, §6).
4. **Restoring linear credibility** by removing a slow time-warp (SRO/clock-drift compensation) — turns a chunk of the 64% back into a modelable case (Report 1 §6).

Below are the five candidates worth engineering time, ranked by expected payoff against OUR wall and OUR hard FS bar.

---

## #1 — Per-bin convergence-gated leak (Speex `leak_per_bin` × AEC3 NonLinear R2, selected per-bin by filter state)

**Mechanism.** Replace the scalar `residual = leak · |Ŷ|²` with a per-bin blend selected by *filter convergence state*, not by a near/far feature:

```
leak[k]     = clip( Pey[k]/Pyy[k], MIN_LEAK_k, 1 )          # per-bin power-covariance (Report 3)
residual[k] = max( leak[k]·|Ŷ_k|² , conv_floor[k]·X2_aligned[k] )
```

where `conv_floor[k]` is the AEC3-style delay-aligned, window-maxed, stationary-subtracted render power at **unity gain** (Report 4 §6), gated ON only in bins where the linear filter is NOT converged. We already expose per-bin PBFDKF `P`/Kalman gain — use `P_trace[k]`/Kalman convergence directly as `conv_floor[k]` (high P-trace → high render floor; converged → ~0), rather than re-deriving from ERLE.

**What NEW information breaks the wall.** It does not try to discriminate near from far with a feature at all. It makes the FS-vs-breath decision on a **filter-state signal** — "is the linear estimate credible in this bin." That IS separable where every near/far feature overlaps: converged ql7 HF bins keep the credible `leak·|Ŷ|²` (Speex-style breath-preserving leak); unconverged movement bins fall back to a bounded render-power floor (AEC3-style FS holding). The per-bin granularity is the key advance over our scalar leak — breath living in a few HF bins where the filter IS converged is preserved even while adjacent blind bins catch the non-stationary echo.

**Integration point.** Drop-in replacement for the residual-echo PSD λ̂_echo handed to the RES Wiener/MMSE stage. No new adaptive filter. Source the per-bin convergence from existing PBFDKF state; source `X2_aligned` from the existing delay estimate with a ±1-block window-max.

**Cost.** Low. Per-bin covariance EMAs (Pey/Pyy) + a window-max render buffer. A few hundred MACs/frame at 257 bins. No extra FFT.

**FS-bar risk (would it leak ql7?).** Low for ql7, which is HIGH-coherence and converges (clears the 3 dB `e²<0.5y²` bar) → stays on the `leak·|Ŷ|²` path exactly as today. The risk is the *opposite*: in unconverged bins the unity-gain render floor could OVER-suppress breath that coexists with a loud render — but that is bounded by Report 3 P3 (floor-not-gate) and is the Pareto knob, not a leak. The real FS hazard is a **mis-set delay** feeding `X2_aligned` to the wrong bins; the ±1-block window-max (Report 4 §6 item 1) is the mitigation, and it must be validated that the floor does not fire in *converged* bins (P2 below).

**Pair with (mandatory):** Report 3 P2 — per-bin leak-adaptation freeze (`update Pey/Pyy only when |Ŷ_k|² > β·See_k`) so breath cannot corrupt the leak; and Report 3 P3 / Report 2 §4 — use the result as a soft **gain floor** (`p·gain + (1−p)·floor`), never a hard gate.

---

## #2 — Decision-directed a-priori SER + MMSE-LSA gain rule (replace the hard RES gate)

**Mechanism.** Replace our hard Wiener/power gate — the documented near-end killer — with an **MMSE-LSA** (log-spectral-amplitude) gain driven by a **decision-directed a-priori SER**:

```
ξ̂[k,ℓ] = α·|Ŝ[k,ℓ−1]|² / λ̂_echo[k,ℓ] + (1−α)·max(γ[k,ℓ]−1, 0)
```

The DD recursion's α-weighted *previous-frame near-end estimate* IS the temporal-context term: once near-end is asserted, ξ̂ stays elevated for several frames (built-in hangover), so the gain does not slam shut on a transient render-power spike — exactly the breath-onset case. LSA is the canonically least-distorting suppression rule; Speex's preprocessor is a crude version of it (Report 2 §4, Report 3 §3).

**What NEW information breaks the wall.** Temporal memory on the *near-end side*. It does not discriminate per-frame; it integrates a soft speech-presence/SER over a few frames, smoothing the gain across breath onsets where a per-frame gate collapses to the floor.

**Integration point.** Replaces the gain-computation layer that sits on top of λ̂_echo. Composes directly with #1 (which supplies a better λ̂_echo). This is the structural fix to "RES Wiener gain is the nearend killer" in our memory.

**Cost.** Low — we already compute most terms (post/prior SNR, DD prior).

**FS-bar risk (would it leak ql7?).** Medium and must be validated. LSA under-suppresses relative to hard Wiener at low SER, so the α (DD weight) and SPP prior set the Speex↔AEC3 Pareto point. The claim is *less breath damage per dB of echo*, not free echo. ql7 risk is bounded because λ̂_echo from #1 is credible on ql7 (converged) → ξ̂ low → gain suppresses normally. The hazard is on the unconverged class if α is too aggressive (hangover holds the gain open into real echo) — sweepable.

---

## #3 — Temporal+harmonic residual-echo PSD with a conservative DT switch (Song & Shin 2020)

**Mechanism.** Model the current-frame residual-echo PSD as a regression on (a) the linear echo estimate at **harmonically-related bins** (the loudspeaker-nonlinearity term the linear filter cannot model — `i·j+k=m` stencil, Report 1 §1) and (b) **linear echo / residual / mic estimates in ADJACENT past frames** (explicit inter-frame regression). A DTD tuned to a deliberately LOW false-reject rate gates ONLY whether the past-mic term folds in — it is a conservative switch on the temporal term, never the discriminator (Report 2 §1).

**What NEW information breaks the wall.** Two new axes at once: cross-bin harmonic coupling (a nonlinear estimate the 52 ms filter is structurally blind to) and temporal regression of the echo's OWN statistics (which persist across frames even at MSC≈0 because the residual is a slow filtered render envelope). The polarity is breath-safe: both terms ADD residual PSD on FS (more suppression) and are switched off on suspected DT.

**Integration point.** Feeds a better λ̂_echo into #2's gain rule. The harmonic term and temporal term should be wired as **separately ablatable** flags — they are class-dependent (harmonic → ql7-like static-weak; temporal → movement).

**Cost.** Moderate. Past-frame buffers of |E|²/|Ŷ|²/|Y|² + a small per-bin regression/EMA + a sparse harmonic stencil. Fits 257-bin/160-hop directly.

**FS-bar risk (would it leak ql7?).** Low — this raises FS suppression. ql7 benefits (the harmonic term models the distortion the linear filter misses). The risk is the opposite Pareto direction (over-suppress breath if the conservative DT switch false-rejects too rarely), controlled by the DT switch's operating point. Caveat: the harmonic term weakens on clock-drift/time-warp residuals (not harmonic), which fall back to the temporal term — so isolate the two on the 800-case.

---

## #4 — SRO / clock-drift diagnostic + compensation (restore linear credibility)

**Mechanism.** Estimate sample-rate offset from the **phase-slope drift of the ref–mic cross-spectrum over time** (a linearly-growing-across-frames phase ramp is the SRO signature), then compensate by frame-step control + fractional-delay phase rotation in overlap-save — no explicit resampler (Report 1 §6).

**What NEW information breaks the wall.** It does not discriminate at all — it REMOVES a slow time-warp that is destroying MSC, restoring the linear filter's credibility on whatever fraction of the 64% is SRO-driven. Those cases then preserve breath the normal way (converged → #1 keeps them on the leak path). Orthogonal to all nonlinear modeling.

**Integration point.** Upstream of the linear filter (delay/alignment stage). Diagnostic-first: it does not even need to ship to be valuable — it tells us how much of the 64% is recoverable vs truly nonlinear.

**Cost.** Diagnostic: trivial (cross-spectrum phase-slope tracker, logging-only). Compensation: moderate (fractional-delay phase rotation per frame).

**FS-bar risk (would it leak ql7?).** None for the diagnostic. For compensation: ql7 is static (no drift) → phase slope ~0 → no compensation applied → unaffected. Risk is a false SRO estimate on a *moving* path (movement also drifts phase) injecting a spurious resample — must gate the SRO estimator on far-stationarity and require a *persistent linear* phase ramp (movement gives erratic, not linear, drift).

---

## #5 — Temporal/per-band kurtosis of the error as a soft SPP prior (highest-novelty cheap probe)

**Mechanism.** Compute windowed 4th-moment (kurtosis/negentropy) of the error per band over a short window; feed it as a **soft speech-presence prior** into #2's ξ̂, never as a standalone gate (Report 2 §6). Near-end speech is heavy-tailed/time-localized; reverberant, clock-drift-smeared far-echo is closer to Gaussian.

**What NEW information breaks the wall.** It is the rare feature that **never consults the render or the linear estimate** — so MSC<0.05 is irrelevant by construction. This is genuinely outside the failed feature set (all of which were render-referenced or single-frame magnitude/coherence). Temporal kurtosis-per-band is a feature with temporal context we have NOT probed.

**Integration point.** A soft multiplicative prior on the SPP in #2. Trivial to log first.

**Cost.** Low (windowed per-band 4th moment).

**FS-bar risk (would it leak ql7?).** Low if used only as a soft prior. **Honest caveat — this is the one most likely to disappoint:** when near-end IS breath (low-energy, noise-like, UNVOICED), its kurtosis advantage over echo shrinks — exactly our hardest sub-case. So it may protect voiced near-end but not the breath we most care about. Probe-before-believe.

---

## #1 RECOMMENDATION + cheap first validation (kill-before-800-case)

**#1 (per-bin convergence-gated leak) is the recommendation.** It is the only candidate that improves the **trade-off curve itself** rather than sliding along it: it replaces a scalar far-power gate with a per-bin, filter-state-selected echo-power model, using a signal (per-bin convergence) that is provably separable where every near/far feature overlaps. It reuses machinery we already have (PBFDKF P-trace, delay, Speex leak math), it is cheap, and its FS risk is structurally bounded to ql7 staying on the unchanged converged path.

**Cheap first experiment — LOGGING-ONLY, no mechanism change, single representative case from each class (ql7 static-weak, one movement/low-coherence FS, one DT-with-breath):**

Per bin per frame, dump four traces:
1. `usable_per_bin[k]` — the per-bin convergence flag from PBFDKF P-trace / `e²<0.5y²`.
2. `leak[k]·|Ŷ_k|²` — the Speex-style residual estimate.
3. `conv_floor[k]·X2_aligned[k]` — the candidate unity-gain render floor.
4. The oracle echo PSD (from the ref-only / near-muted rendering we already use for ground-truth WAVs in the breath arc).

**The single kill-criterion:** On the movement/low-coherence FS case, does `usable_per_bin` actually go LOW (and `conv_floor·X2` track the oracle echo) in the bins where `leak·|Ŷ|²` under-estimates the oracle? AND on the DT-with-breath case, does `usable_per_bin` stay HIGH (so the breath bins keep the leak path, NOT the render floor) in the breath bins?

- If `usable_per_bin` is HIGH in the leaking FS bins (filter falsely thinks it converged) → the convergence signal is not separable → **KILL** before any mechanism work. This is the exact "probe trap" our memory warns about (`_aec3_state` recreated every hop → re-fetch/class-patch; a once-captured ref reads a dead object → false usable_linear=0%) — so the probe MUST re-read per-bin state every hop.
- If `usable_per_bin` is LOW across the breath bins too (over-triggers, would route breath to the render floor) → it will over-suppress breath → **KILL** or demote to a Pareto-only lever.

This is a one-case, logging-only probe answerable in an afternoon, and it directly tests the load-bearing assumption (per-bin convergence is separable where features are not) before committing to the residual-PSD rewrite or any 800-case run.

---

## Dead-ends — explicitly do NOT pursue (under our constraints)

- **Two-path / shadow-filter divergence as a discriminator** (Report 2 §2). Blind on the 64%: when the main has no credible estimate, a shadow on the same render is equally blind — both track noise, "shadow beats main" carries no near-end info. Already CLOSED FAIL in our memory (F2.2 EMA diverged-streak, P52 shadow-orthogonality, `feedback_shadow_stays_nlms`). Use shadow only as a conservative DT *gate input*, never the discriminator.
- **Coherence DTD with adaptive threshold** (Report 2 §3). Our class is *defined* by MSC<0.05 → the achievable max collapses → the adaptive threshold has no dynamic range. Useful only on the modelable ql7 cases / as a per-class router, not on the wall.
- **All normalized cross-correlation / normalized-error-power / Ŷ-Y coherence DTDs** (Report 2 §8). Render-referenced → identically dead at MSC<0.05. We already falsified E²/Y², delay-aligned X-Y coherence, Ŷ-Y coherence as single-frame features. Skip the entire bucket.
- **Full Volterra / orthogonalized power filter / SA-HGM as a replacement CANCELLER** (Report 1 §3/§5). Hundreds of thousands of kernel coefficients (full Volterra) or a 6-month frequency-domain arc (SA-HGM); fragile in DT (worse conditioning than our already-under-converging PBFDKF → more destructive false-DT divergence, per our regime-handler scar tissue F1.2/F2.2); and it helps the static harmonic-distortion (ql7-ish, high-coherence) cases we *already* handle while doing nothing for the low-coherence movement wall. The power-filter *PSD* (extracted into #1/#3) captures the useful part at ~1% of the cost and risk. The full canceller is NOT a primary candidate.
- **AEC3 StationarityEstimator residual-scaling.** It is OFF by default and, even enabled, zeroes R2 only when the *render* is stationary — mutually exclusive with the non-stationary-echo class by construction (Report 4 §3). It would help breath only during stationary far-noise, which is not our wall.
- **AEC3 DominantNearendDetector.** A far-vs-near *level* test on bins 1–15 keyed on R2 (render power); structurally cannot fire during far-active breath (Report 4 §4). It is the same far-power gate we already have, not a new discriminator.
- **Cepstral-peak / harmonicity near-end discriminator** (Report 2 §7). Voiced-only — breath/fricatives are UNVOICED, no pitch peak — so it cannot protect the breath we care about. No prior art uses it for DT echo discrimination. Rank below #5; pursue only if voiced-near-end protection is separately needed.

**Unifying skeptical bottom line (matches our falsified-premise memory):** the entire nonlinear-AEC literature gets credible estimates by adapting a reference-driven model that is frozen in DT — these are FS-bar levers. The only paths to breath are (a) **filter-STATE selection** (#1, the one method that moves the curve), (b) **temporal memory** (#2/#3), (c) a **render-independent near-end statistic** (#5), or (d) **restoring linear credibility** (#4). Expect every one to be a Pareto move except #1 (curve-shaping) and #4 (recovers cases outright where SRO is the cause). Probe #1 first; it is cheap to kill.
---
## Report 1 — Nonlinear / non-stationary echo estimate

I now have comprehensive coverage across all method families plus the clock-drift sub-problem (which the problem statement names as one driver of the non-modelable echo). I have enough concrete mechanism, cost, and skeptical material to write the report.

---

# Nonlinear / non-stationary AEC research scout — getting a credible echo estimate where the linear filter fails

Scope reminder and verdict up front: the goal is to **preserve near-end breath during far-end activity without leaking the non-stationary echo, on the ~64% of FS cases where MSC(ref,mic) < 0.05**. The honest framing of the literature is this: almost every "nonlinear AEC" method below buys you a *better-converged echo estimate* — which helps the FS-echo bar — but **none of them solves the near-end/non-stationary-echo discrimination problem during double-talk on its own.** Every method that gives a credible nonlinear echo estimate from the reference does so by *adapting a model from the reference*, and that adaptation is **double-talk-toxic** — it must be frozen by a DTD when near-end is present, which is exactly the moment your breath problem lives. So the literature splits cleanly into two buckets, and you should treat them differently:

- **Bucket A — get a credible echo estimate (helps the FS bar, does NOT help breath directly):** Volterra / power-filter / FLAF / Hammerstein-group / particle-filter nonlinear cancellers, and HDRES-style residual-echo PSD estimators.
- **Bucket B — what actually moves the breath problem:** the *only* lever any of these papers gives toward near-end preservation is **a better residual-echo PSD estimate**, so the RES Wiener gain stops over-subtracting because it has a *credible* echo PSD instead of falling back to gating on far-end power. That is the thread worth pulling. The discrimination itself (near vs non-stationary echo with no credible linear estimate) is acknowledged as an open problem in this entire literature — they all sidestep it with a DTD.

Below, each family with mechanism, why it could be credible where linear fails, cost at your geometry (160-hop / 512-FFT / 257 bins / 16 kHz), portability, and what would NOT survive your hard FS bar.

---

## 1. HDRES — Harmonic-Distortion Residual Echo Suppression (Bendersky, Stokes, Malvar) — the highest-relevance method

**Core mechanism** (from US8213598B2 and the Microsoft Research paper "Nonlinear residual acoustic echo suppression for high levels of harmonic distortion," Bendersky/Stokes/Malvar, ~2008): this operates *entirely in the STFT magnitude domain on top of your existing linear AEC* — it is a residual-echo PSD estimator, not a new canceller. The residual-echo magnitude in bin *m* is modeled as a **linear combination of linear-echo-estimate magnitudes in harmonically-related bins**:

  D̂ᵣ(κ,m) = Σᵢ Σⱼ Σₖ δ(i,j,k,m) · Wᵣ(i,j,k) · |X′(κ,i)|,  with δ=1 iff i·j + k = m

i.e. bin *m* is fed by fundamental bin *i*, its *j*-th harmonic, within a ±K bin window. Coefficients Wᵣ are adapted by NLMS on the error ξ = |E| − |D̂ᵣ|. The suppression gain is a magnitude-subtraction Wiener-type:

  G(κ,m) = max{ Ē(κ,m) − β·D̄ᵣ(κ,m), N̄(κ,m) } / Ē(κ,m),  β≈0.95, phase preserved.

**Why credible where the linear filter fails:** the non-modelable echo on your 64% is, in large part, *loudspeaker/amplifier harmonic distortion* — energy at 2f, 3f… that has **no linear transfer-function relationship to the reference at f**, so MSC(ref,mic) collapses and the 52ms linear filter literally cannot represent it. HDRES models exactly that cross-bin (f → harmonics) mapping that a linear filter cannot, so it can predict residual-echo energy in bins where your filter is blind. This is the single most portable, lowest-cost idea in the whole literature: it is a bolt-on to your RES, magnitude-domain, ~257-bin, and the reported result is **+14.9 dB ERLE over linear-only at negligible double-talk speech-quality loss** — which is the closest published claim to your actual target.

**Cost:** trivial at your geometry. It is per-bin NLMS over a sparse harmonic stencil; a handful of taps per bin, magnitude-domain, no extra FFTs. Easily real-time on a DSP.

**The skeptical kill-shot for YOUR problem (this is the crux):** HDRES does NOT separate near-end from non-stationary echo. It *relies on a double-talk detector*. The patent is explicit: "If adaptation occurs when near-end voice is present, even for a short period, near-end voice distortion increases considerably." Its near-end protection is entirely (a) **freeze/rollback Wᵣ adaptation during DT** (discard the last T₁ steps on ST→DT transition, re-enable only after T₂ consecutive ST frames), and (b) **noise-floor clamp N̄ in the gain**. So HDRES gives you a *credible echo-PSD estimate during clean FS* (good for the FS bar), but in double-talk it falls back to a frozen model + power-gating — i.e. **the same failure mode you already have**, just with a smarter frozen estimate. It would very likely *pass and probably help your FS-echo bar* (better PSD → less reliance on raw far-power gating on the harmonic-distortion cases) but it will **not, by itself, preserve breath during active double-talk**, because the breath problem is precisely when adaptation is frozen. Verdict: **port it for the FS bar; it is the most plausible "credible estimate" win, but do not expect it to be the breath fix.** The breath fix would have to come from the frozen model being accurate enough that the RES floor can be raised — a Pareto move, not a free lunch, consistent with your MEMORY's "breath = Pareto choice" conclusion.

(Refs: US8213598B2 Bendersky/Stokes/Malvar, Microsoft; "Residual Echo Suppression Considering Harmonic Distortion and Temporal Correlation," Appl. Sci. 10(15):5291, 2020 — extends HDRES with a second adjacent-frame temporal-correlation term and **two separate magnitude estimates, one for FS one for DT, switched by a low-false-reject DTD** — same DTD dependence.)

---

## 2. Power-filter residual-echo PSD (Kuech & Kellermann, ICASSP 2007) — the most "RES-native" and DTD-robust option

**Core mechanism** ("Nonlinear Residual Echo Suppression using a Power Filter Model of the Acoustic Echo Path," Kuech/Kellermann; cf. IWAENC 2005 "Residual Echo PSD Estimation based on…"): model the echo path as a **power filter** — a basis expansion of the reference into fixed nonlinear basis signals (the reference and its squared/higher even-power terms), each passed through its own linear coupling. The key trick for RES is that they estimate the **residual-echo PSD** directly from these basis-signal PSDs via *adaptively estimated coupling factors*, and — critically — by working in the PSD/basis-expansion form, **the PSD estimate is independent of the room-impulse-response length** (you don't need a long, converged nonlinear filter; you estimate a few scalar coupling coefficients between basis-signal power and residual power). That PSD then drives a Wiener postfilter.

**Why credible where the linear filter fails:** it never tries to invert the nonlinear path sample-by-sample; it only needs a *statistical* (power-domain) relation between reference-derived nonlinear basis powers and the leftover echo power. So even when MSC ≈ 0 (no usable phase relationship), the *power* of the harmonic/nonlinear echo is still predictable from the *power* of the basis signals — which is exactly the regime your problem describes ("non-stationary / loudspeaker nonlinearity / clock drift" destroying coherence but not destroying the power relationship). This is conceptually the **canonical, principled successor to Speex's `leak = cov(|E|²,|Ŷ|²)/var(|Ŷ|²)`** — Speex estimates ONE scalar leak from the linear echo power; the power-filter method estimates **per-basis, per-band coupling factors** from *nonlinear* basis powers. That is a direct, mechanism-level upgrade of the lever your MEMORY already calls a live axis (`leak_per_bin`).

**Cost:** moderate but bounded. A few basis signals (e.g. linear + cubic, or linear + |x|·x) × per-band coupling factors. No per-tap nonlinear filter. At 257 bins with, say, 2–3 basis branches, this is a few hundred extra MACs/frame — cheap.

**Why it may survive the FS bar where HDRES is shakier on breath:** power-domain coupling factors are *much less DT-toxic* than HDRES's NLMS-on-magnitude, because power covariance is more robust to a brief near-end excursion than sample-aligned magnitude adaptation (this is precisely why Speex's leak survives DT while a sample-domain filter diverges). So this is the family most likely to let you **raise the RES gain floor in DT without leaking FS echo**, because you'd have a credible *nonlinear* echo PSD instead of falling back to raw far-power gating. **This is the one I'd prototype first for the breath problem specifically.**

**Skeptical flags:** (1) it still over-estimates if the coupling factors are learned during reverberant tails or movement — you'd need to gate factor *updates* (not the gain) on far-stationarity. (2) The published evaluations are FS-ERLE-centric; near-end preservation in DT is asserted, not the headline. (3) It assumes the nonlinearity is *memoryless-ish* (power filter = memoryless polynomial + linear filter cascade); strong **clock drift is NOT a memoryless nonlinearity** — drift is a slow time-warp that this model does not represent, so on your clock-drift subset this PSD will *also* mis-estimate. Treat clock drift separately (see §6).

---

## 3. Volterra / power filters as a replacement CANCELLER (Stenger/Kellermann; Kuech/Mitnacht/Kellermann orthogonalized power filters, ICASSP 2005)

**Mechanism:** replace/augment the linear filter with a 2nd-order Volterra filter, or its efficient reparametrization, the **orthogonalized power filter** = parallelized cascade of a memoryless polynomial nonlinearity followed by a linear filter, with the basis signals **orthogonalized following the (non-stationary) input statistics** to fix the otherwise-catastrophic eigenvalue spread / slow convergence of raw Volterra.

**Why credible where linear fails:** it explicitly models loudspeaker harmonic distortion (the dominant non-modelable component on small/cheap speakers — Kuech/Mitnacht/Kellermann motivate it with "miniature speakers in cellular handsets… well modeled by a memoryless nonlinearity"). A converged 2nd-order Volterra/power filter *can* produce a phase-accurate echo estimate where the linear filter produces garbage.

**Cost — this is where it gets dangerous at your geometry.** Full 2nd-order Volterra coefficient count is ~L + L(L+1)/2; for a 52 ms / ~832-tap-equivalent memory this is *hundreds of thousands* of kernel coefficients — utterly infeasible. The orthogonalized power filter and FLAF (§4) exist precisely to dodge this, by factoring into (small memoryless nonlinearity) × (one linear filter the same size as yours). Even so you're adding 1–3 extra full-length adaptive filters.

**Skeptical flags — why this likely does NOT survive your bar as a primary move:** (1) **Convergence/double-talk fragility.** Your MEMORY already documents that the linear PBFDKF *under-converges* (~2.5 dB linear ERLE) on the hard cases and that adding state to it is delicate; a nonlinear filter has worse conditioning and a far more destructive false-DT divergence. The same "shadow stays NLMS / regime handler is destructive when falsely triggered" lessons apply doubly. (2) On the **movement/clock-drift** subset, the nonlinearity is *not* the dominant problem — a Volterra filter still needs a stable linear path, which is exactly what's missing. So Volterra helps the *static harmonic-distortion* (ql7-ish, high-coherence) cases you already handle, and does little for the low-coherence movement cases that are the real wall. **Verdict: high cost, helps the cases you're least worried about, fragile in DT — not a primary candidate. The power-filter *PSD* (§2) extracts the useful part of this idea at 1% of the cost and risk.**

---

## 4. Functional-Link Adaptive Filters — FLAF / split-FLAF / collaborative-FLAF (Comminiello, Scarpiniti, Azpicueta-Ruiz, Uncini, IEEE TASLP 2013; Sicuranza & Carini)

**Mechanism:** expand each reference sample through a **fixed trigonometric functional-link basis** and adaptively filter the expansion. The canonical expansion (confirmed from the block-oriented FLAF paper, Ganjimala & Mula 2023):

  φⱼ(x(n−i)) = { x(n−i) for j=0;  sin(pπx(n−i)) for j=2p−1;  cos(pπx(n−i)) for j=2p },  p=1…P, total Q = 2P+1 links.

**Split-FLAF** runs a linear NLMS filter (your existing one) and a *separate* nonlinear FLAF in parallel, each adapted independently; **collaborative/combined-FLAF** convex-combines a linear-only and a nonlinear filter with an adaptive mixing parameter so that "when the system is essentially linear, the combination weight collapses to the linear filter" — i.e. **built-in robustness against over-modeling.**

**Why credible where linear fails:** the sin/cos expansion captures odd/even loudspeaker nonlinearities (saturation, clipping) that a linear filter cannot, and the FLAF avoids Volterra's coefficient explosion.

**Cost — favorable, this is the practical nonlinear-canceller choice.** Complexity is **O(M·Q), linear in the number of expansion terms Q**, not exponential. Concrete from the Hammerstein-block FLAF: at filter length M=1024 with Q=7 links, ~9,200 mults/sample. At your geometry (and a shorter ~832-tap filter) a split-FLAF with Q≈5–7 is a few thousand MACs/sample on top of your linear path — real-time feasible on a capable DSP, though not free.

**Portability:** **split-FLAF is the single most drop-in nonlinear *canceller*** for your pipeline: keep your PBFDKF as the linear path, add a parallel trigonometric-FLAF nonlinear path summed before the RES; the RES stays as-is. The **combined-FLAF mixing parameter is attractive** because it auto-disables the nonlinear path on linear cases — which protects you on the movement/reverb cases where the nonlinear model would otherwise just add variance.

**Skeptical flags:** (1) Same DT problem — FLAF NLMS adaptation must be frozen in double-talk or it diverges into near-end; the FLAF papers assume a DTD. So, like §3, it **improves the FS estimate but does not preserve breath in DT.** (2) The trigonometric basis is tuned to *memoryless saturation*; it does nothing for **clock drift or movement-driven coherence loss**, which are a large part of your 64%. (3) Adding a second full adaptive filter doubles your divergence surface — and your MEMORY's whole history (F1.2/F2.2 regime-handler regressions) warns that *more adaptive state = more destructive false fires* on the cohort tail. **Verdict: the best nonlinear-canceller option if you want one, lower-risk than Volterra via the combined-filter auto-off, but still an FS-bar play, not a breath play.**

---

## 5. Significance-Aware Hammerstein Group Models (Halimeh/Huemmer/Kellermann, EURASIP JASP 2016; PBSA-HGM EUSIPCO 2016) — the efficiency-optimized industrial form

**Mechanism:** decompose the nonlinear echo path into (a) a **direct-path region** modeled by a group of parallel Hammerstein models (fixed nonlinear preprocessor + linear filters), from whose linear kernels a **nonlinear preprocessor is estimated by cross-correlation**, and (b) a **remaining (reverberant tail) region** modeled by one cheap Hammerstein model reusing that preprocessor. Partitioned-block frequency-domain realization (PBSA-HGM) for real-time.

**Why credible / cost:** this is the *most computationally efficient* principled nonlinear canceller (the "significance-aware" idea = spend coefficients only where they matter, near the direct path), and the frequency-domain partitioned-block form maps naturally onto your existing block-frequency-domain pipeline. It is essentially the production-grade version of §3/§4.

**Skeptical flags:** sophisticated, but it is still a *canceller* with the same DTD dependence and the same "doesn't address movement/clock-drift coherence loss" limitation. Higher integration effort than HDRES/power-filter-PSD for the same FS-bar-only benefit. **Verdict: the right answer if you decide to invest in a real nonlinear canceller and want frequency-domain efficiency, but it is a 6-month arc, not a quick lever, and it does not change the breath calculus.**

---

## 6. The part of your 64% that NONE of the above fixes: clock drift / SRO and movement

This is the most important skeptical point, because the problem statement lumps "loudspeaker nonlinearity / clock drift / non-stationary" together, but **they require different mechanisms and only the first is addressed by §1–5.**

- **Clock drift / sample-rate offset (SRO):** even ~4 ppm offset destroys MSC and makes the residual look exactly like non-modelable echo, but it is **a slow time-warp, not a memoryless nonlinearity** — Volterra/FLAF/power-filter models *cannot* represent it. The canonical DSP fix is **SRO estimation + compensation**: estimate the offset from the **phase drift of the ref–mic coherence over time** (the slope of cross-spectrum phase vs frequency drifting linearly across frames is the SRO signature), then compensate by **frame-step control + fractional-delay phase rotation (overlap-save)**, no explicit resampler needed (refs: "On Dealing with Sampling Rate Mismatches…" Kellermann group; US7120259B1 "Adaptive estimation and compensation of clock drift in AEC"; "Sample Rate Offset Compensated AEC for Multi-Device Scenarios," 2024). **This is genuinely high-value and orthogonal to everything else** — if any meaningful fraction of your low-coherence FS cases are SRO-driven, compensating SRO would *restore the linear filter's credibility* (raise MSC) and let your existing pipeline preserve breath the normal way. Strongly recommend a diagnostic: measure cross-spectrum phase-slope drift on your low-MSC FS cohort to see how much of the 64% is SRO vs true nonlinearity. This is the highest-leverage, lowest-listed-elsewhere idea in this report.
- **Movement (echo-path change during the case):** your MEMORY already has this correctly framed as a convergence/regime problem, not a nonlinearity. None of §1–5 helps; it is the linear filter's tracking that matters. Noted only to keep the taxonomy honest — don't let "nonlinear AEC" papers distract from the movement subset.

---

## Bottom-line recommendations, ranked by expected value vs your hard FS bar

1. **SRO/clock-drift diagnostic first (§6).** Cheap, orthogonal, potentially restores linear-filter credibility on a chunk of the 64% — which would let you preserve breath *without* any nonlinear modeling. Measure coherence phase-slope drift before committing to nonlinear arcs.
2. **Power-filter residual-echo PSD (§2) — most likely to actually move breath.** It is the principled, *power-domain (DT-robust)*, per-band, *nonlinear* generalization of the exact Speex `leak` lever your MEMORY already treats as live. It gives a credible *nonlinear* echo PSD without needing a converged long filter, and power covariance survives brief DT better than magnitude NLMS — so it's the best shot at *raising the RES floor in DT without leaking FS echo*. Prototype it as `leak_per_bin` driven by linear + squared basis powers.
3. **HDRES (§1) — best FS-bar win, easiest port, but DTD-gated so not a breath fix by itself.** Bolt-on magnitude-domain harmonic-bin RES; cheap; +14.9 dB ERLE reported. Expect it to improve the harmonic-distortion FS cases; do not expect breath preservation in active DT (adaptation frozen there).
4. **Combined/split-FLAF (§4) — if you want a real nonlinear canceller, this is the lowest-risk one** (auto-off mixing on linear cases, O(M·Q) cost). Still FS-bar-only and adds divergence surface; weigh against your regime-handler scar tissue.
5. **Volterra / orthogonalized power filter (§3) and SA-HGM (§5):** principled but heavier; SA-HGM is the efficient industrial form if you commit to a canceller arc. Neither changes the breath calculus.

**The unifying skeptical conclusion**, which matches your existing MEMORY: the *entire* nonlinear-AEC literature gets credible echo estimates by **adapting a reference-driven model that must be frozen in double-talk**. None of them publishes a feature that separates near-end from non-stationary echo *while the model is frozen* — they all defer to a DTD. So these methods are **FS-echo-bar levers (better credible estimate → less reliance on raw far-power gating), and the only path they offer to the breath problem is an *indirect* one: a credible enough echo PSD that you can safely raise the RES gain floor.** That is still a Pareto move (FS headroom traded for breath), exactly as your "breath = Pareto choice, not a correctness fix" line already concluded — but **the power-filter PSD (§2) is the one method that improves the trade-off curve itself** rather than just sliding along it, because it replaces a scalar far-power gate with a per-band nonlinear echo-power model.

### Key sources
- Bendersky, Stokes, Malvar, "Nonlinear Residual Acoustic Echo Suppression for High Levels of Harmonic Distortion," Microsoft Research (~2008); US Patent 8,213,598 B2 — [microsoft.com](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/diegobenderskyhdres.pdf), [patents.google.com/US8213598B2](https://patents.google.com/patent/US8213598B2/en)
- "Residual Echo Suppression Considering Harmonic Distortion and Temporal Correlation," Appl. Sci. 10(15):5291, 2020 — [mdpi.com](https://www.mdpi.com/2076-3417/10/15/5291)
- Kuech & Kellermann, "Nonlinear Residual Echo Suppression using a Power Filter Model of the Acoustic Echo Path," ICASSP 2007 — [researchgate.net](https://www.researchgate.net/publication/4248831_Nonlinear_Residual_Echo_Suppression_using_a_Power_Filter_Model_of_the_Acoustic_Echo_Path); IWAENC 2005 PSD paper — [iwaenc.org](https://www.iwaenc.org/proceedings/2005/papers/S04-06.pdf)
- Kuech, Mitnacht, Kellermann, "Nonlinear Acoustic Echo Cancellation using Adaptive Orthogonalized Power Filters," ICASSP 2005 — [ieeexplore](https://ieeexplore.ieee.org/document/1415657/); Orthogonalized power filters, Signal Processing — [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0165168405003191)
- Comminiello, Scarpiniti, Azpicueta-Ruiz, Arenas-García, Uncini, "Functional Link Adaptive Filters for Nonlinear Acoustic Echo Cancellation," IEEE TASLP 2013 — [ieeexplore](https://ieeexplore.ieee.org/document/6488741/); Ganjimala & Mula, "A Low Complexity Block-oriented FLAF Algorithm," 2023 — [arxiv 2310.10276](https://arxiv.org/html/2310.10276)
- Halimeh/Huemmer/Kellermann, "Significance-aware filtering for nonlinear AEC," EURASIP JASP 2016 — [asp-eurasipjournals](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-016-0410-7); PBSA-HGM, EUSIPCO 2016 — [ieeexplore](https://ieeexplore.ieee.org/document/7760555/)
- Huemmer & Kellermann, "Elitist Particle Filter based on Evolutionary Strategies for Nonlinear AEC," IEEE/ACM TASLP 2018 / arXiv 1604.04198 — [arxiv](https://arxiv.org/abs/1604.04198)
- Schwarz, Hofmann, Kellermann, "Spectral feature-based nonlinear residual echo suppression," WASPAA 2013 — [ieeexplore](https://ieeexplore.ieee.org/document/6701825/)
- Stenger & Kellermann et al., "Nonlinear Acoustic Echo Cancellation based on Volterra Filters" — [ieeexplore](https://ieeexplore.ieee.org/document/3333988); Enzner, "Partitioned residual echo power estimation," Eur. Trans. Telecomm. 2002 — [wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/ett.4460130205)
- Clock drift / SRO: "On Dealing with Sampling Rate Mismatches in BSS and AEC" (Kellermann group) — [researchgate](https://www.researchgate.net/publication/4295706); US7120259B1 — [patents.google.com](https://patents.google.com/patent/US7120259B1/en); "Sample Rate Offset Compensated AEC for Multi-Device Scenarios," 2024 — [researchgate](https://www.researchgate.net/publication/384652143)
---
## Report 2 — DT-robust RES + stateful near-end

I have comprehensive coverage. Synthesizing now.

---

# Research Scout Report: Double-Talk-Robust, Near-End-Preserving RES when the linear filter gives no credible estimate

**Framing of your wall (so the recommendations are scoped honestly):** Your problem is that on the ~64% non-modelable FS class, MSC(X,Y)<0.05 means *every coherence/cross-correlation feature that depends on the linear filter or on X-Y alignment is dead by construction* — and you've already proven the single-frame branch-independent features overlap. So the literature splits cleanly into two buckets: (A) mechanisms that still secretly depend on a usable estimate or X-Y coherence (these will NOT help you — I flag them) and (B) mechanisms that add **temporal context, a different signal domain, or a near-end-side model** that does not require X-Y coherence. Only bucket B is worth your engineering time. Most of the AEC DTD canon is bucket A.

---

## 1. Temporal / inter-frame residual-echo modeling with a DT-safe gate — STRONGEST direct hit

**Song & Shin, "Residual Echo Suppression Considering Harmonic Distortion and Temporal Correlation," *Applied Sciences* 10(15):5291, 2020.** ([mdpi.com](https://www.mdpi.com/2076-3417/10/15/5291), [researchgate](https://www.researchgate.net/publication/343338076_Residual_Echo_Suppression_Considering_Harmonic_Distortion_and_Temporal_Correlation))

- **Mechanism:** Models residual-echo PSD in the current frame as correlated with (a) the *linear echo estimate at harmonically-related bins* in the current frame — this is the loudspeaker-nonlinearity term, the part your linear filter cannot model; and (b) **linear echo estimates, residual echo estimates, and microphone signals in ADJACENT (past) frames** — an explicit inter-frame regression. The residual-echo PSD estimate is thus a multi-tap-in-time, multi-bin-in-frequency predictor, not a single-frame power gate.
- **The DT-protection trick (this is the part you want):** they adopt a DTD *tuned to a deliberately LOW false-rejection rate of double-talk* — i.e. it errs toward declaring DT. Its only job is to decide whether past frames' microphone signal can be folded into the residual-echo estimate. When DT is even plausibly present, the past-mic term is excluded so near-end energy never leaks into the residual-echo PSD and never gets subtracted from the near-end. This decouples "estimate the non-stationary residual from temporal context" from "don't eat the near-end."
- **Why it can beat your single-frame features:** it never tries to *classify* near-end vs far-echo from one frame. It uses temporal regression of the *echo's own* statistics (which persist across frames even when X-Y coherence is zero, because the residual is still a filtered/distorted version of a slowly-varying render envelope), and uses the DTD only as a conservative *switch on the temporal term*, not as the discriminator. The non-stationary echo is partially predictable from its own recent past; breath is not predictable from the render's recent past.
- **Cost:** moderate — a few past-frame buffers of |E|², |Ŷ|², |Y|², plus a small per-bin regression/EMA. Fits your 257-bin/160-hop geometry directly.
- **FS-echo risk:** LOW — because the temporal/harmonic term *adds* residual-echo PSD on FS (more suppression), and is only switched off during suspected DT. This is the opposite polarity to your Speex leak under-suppression.
- **Skeptical caveat:** the harmonic-bin term assumes the dominant nonlinearity is harmonic distortion of a tonal-ish render; for clock-drift / time-warp residuals the harmonic-bin coupling weakens, and you fall back to the temporal-regression term. Worth isolating the two terms in your 800-case rig because their value is class-dependent (harmonic term helps ql7-like static-weak, temporal term helps movement).

This is the single paper most aligned with your stated need ("a feature/mechanism with TEMPORAL CONTEXT … given single-frame features failed"). Recommend it as candidate #1.

---

## 2. Two-path / shadow-filter DTD — you already have the substrate, but it's bucket A for your hard class

**Foreground/background two-filter DTD; ITU-G.168 robust-EC lineage; dual-H architecture.** ([EP0872962A2 foreground/background](https://patents.google.com/patent/EP0872962A2/en), [US6266409 dual-H](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6266409), [vocal.com DTD](https://vocal.com/echo-cancellation/double-talk-detector-in-echo-cancellation/))

- **Mechanism:** A constrained/standby shadow filter runs alongside the main. DT is declared when the shadow suddenly out-performs the main (or when the main's error jumps while the shadow's doesn't), because near-end onset perturbs the two filters differently. Update/suppression is frozen on the main during DT.
- **Why temporal context helps in principle:** the *divergence trajectory* between two filters is a multi-frame signal, more robust than instantaneous ERLE.
- **Why it likely does NOT crack your 64% wall (skeptical):** on the non-modelable class the *main filter itself has no credible estimate* (MSC<0.05). A shadow filter built on the same render input is equally blind — both filters track noise, so "shadow beats main" carries no near-end information. Your own MEMORY already records this exact failure (`feedback_shadow_stays_nlms`, F2.2 EMA diverged-streak CLOSED FAIL, P52 shadow-orthogonality). **I flag this as a known dead end for your hard class** — include only as a DT *gate input* (à la Song&Shin's conservative switch), never as the discriminator. The literature's two-path DTD works because it assumes a *converged* main filter; that assumption is exactly what fails on your 64%.

---

## 3. Coherence DTD with ADAPTIVE THRESHOLD (temporal max-tracking) — a cheap temporal upgrade, but coherence-dependent

**Tashev, "Coherence Based Double Talk Detector with Adaptive Threshold," Microsoft Research.** ([microsoft.com PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Tashev_ET11_DoubleTalkDetector_final.pdf))

- **Mechanism (from the PDF):** standard MSC(X,Y) statistic, but instead of a fixed threshold it runs a *dual-time-constant peak tracker* on the coherence: a fast-attack/slow-decay integrator tracks the running maximum coherence the channel achieves in clean far-only conditions, and the decision threshold is scaled between zero and that tracked max (with a floor for very low SNR). Adds hysteresis to prevent ringing. The temporal context is in the *threshold normalization*, not the statistic.
- **Why it can beat a fixed single-frame coherence threshold:** it self-calibrates to the channel's achievable coherence (which is far below 1 under reverb/noise), so "low coherence = DT" is judged relative to *what this channel does when there's no near-end*, not an absolute 0.5.
- **Why it still won't crack your wall (skeptical):** your non-modelable class is *defined by MSC<0.05* — the achievable max is already near the floor, so the adaptive threshold collapses and the statistic has no dynamic range. Tashev himself notes coherence "is less suitable when the mic is away from the loudspeaker / high noise." Useful as a cheap robustness improvement on your *modelable* cases (ql7 high-coherence static), and as a per-class router, but not a solution for the 64%.
- **Cost:** trivial (two EMAs + hysteresis). **FS risk:** low. Portable immediately.

The general adaptive-threshold idea (normalize the statistic by its own recent far-only distribution) is the reusable nugget — apply it to *whichever* statistic you adopt, not just coherence.

---

## 4. Decision-directed a-priori SER + MMSE-LSA RES gain — the near-end-preserving GAIN RULE you're missing

**Habets/Cohen lineage; Ephraim-Malah MMSE-LSA applied to residual echo; soft-decision integrated AES.**
- Faller/Cohen statistical-model RES and soft-decision AES ([Frequency-Domain AES Based on Soft Decision](https://www.researchgate.net/publication/221491798_Frequency_Domain_Acoustic_Echo_Suppression_Based_on_Soft_Decision), [integrated AES+NR soft-decision, EURASIP 2012](https://link.springer.com/article/10.1186/1687-6180-2012-11))
- Habets, "MMSE Log-Spectral Amplitude Estimator for Multiple Interferences," IWAENC 2006 ([israelcohen.com PDF](https://israelcohen.com/wp-content/uploads/2018/05/IWAENC2006_Habets.pdf)) and Habets et al., "Joint Dereverberation and Residual Echo Suppression in Noisy Environments" ([researchgate](https://www.researchgate.net/publication/224332472)).
- "The a priori SDR (signal-to-disturbance ratio) Estimation Techniques with Reduced Speech Distortion for AES/ANS" ([researchgate](https://www.researchgate.net/publication/220242534)).

- **Mechanism:** replace the hard Wiener/power gate (your "RES kills the breath") with an **MMSE-LSA gain driven by a decision-directed a-priori SER** ξ̂(k,ℓ) = α·|Ŝ(k,ℓ−1)|²/λ̂echo(k,ℓ) + (1−α)·max(γ−1,0). The DD recursion's α-weighted *previous-frame near-end estimate* is itself the temporal-context term — it gives a frame of "memory" that a frame of high instantaneous residual is consistent with ongoing near-end, smoothing the gain across breath onsets and preventing the per-frame collapse to the floor. LSA (log-domain) is the canonical *least speech-distorting* suppression rule and is exactly what Speex's preprocessor is a crude version of.
- **Why it preserves breath:** the DD ξ̂ has hangover built in — once near-end is asserted, the a-priori SER stays elevated for several frames, so the gain does not slam shut on a transient render-power spike. Combined with an SPP (speech-presence-probability) soft gate, you get graceful protection rather than a binary DT freeze.
- **Why it's not magic (skeptical):** the gain is only as good as λ̂echo, the residual-echo PSD. If you feed it the Speex power-covariance leak estimate, you inherit Speex's leak on the non-modelable class; if you feed it Song&Shin's temporal/harmonic estimate (§1), the two compose well. So this is the *gain-rule layer* to sit on top of a better PSD estimate, not a discriminator by itself.
- **Cost:** low; you likely already compute most terms. **FS risk:** medium — LSA under-suppresses relative to hard Wiener at low SER, so you must validate the FS AECMOS bar; the DD α and the SPP prior are the knobs that set your Speex-vs-AEC3 Pareto point *with far less near-end damage per dB of echo* than a power gate. This is the principled replacement for your "min-gain floor axis."

---

## 5. Phase-robust echo-MAGNITUDE estimation in a smoothed/envelope domain — a *different domain* that survives MSC≈0

**Faller & Chen, "Suppressing Acoustic Echo in a Spectral Envelope Space," IEEE TSA, 2005** ([researchgate](https://www.researchgate.net/publication/3334140)); **Faller & Tournery, "Robust Acoustic Echo Control Using a Simple Echo Path Model"** ([semantic scholar](https://www.semanticscholar.org/paper/Robust-Acoustic-ECHO-Control-using-a-Simple-ECHO-Faller-Tournery/bc3142b1995f938c77badd9dabcf8b32a35a078c)).

- **Mechanism:** estimate the echo's *spectral envelope* (power smoothed over frequency) from the render's spectral envelope via a short adaptive coloration+delay model — and **discard phase entirely**. The gain is computed from estimated echo magnitude only.
- **Why it's relevant to your wall:** MSC<0.05 is a *phase/fine-structure* coherence collapse (clock drift and movement destroy phase alignment first). The *envelope* coupling between render and echo degrades far more gracefully — a drifting/non-stationary echo still has a render-correlated magnitude envelope even when bin-wise phase coherence is gone. So an envelope-space echo-magnitude estimate can remain credible exactly where your linear filter's complex estimate dies. This is a candidate replacement for `usable_linear` on the non-modelable class: when complex-domain MSC<0.05, fall back to an *envelope-domain* echo estimate rather than to a pure render-power gate.
- **Skeptical caveat:** envelope-space is coarser; it over-suppresses near-end whose envelope happens to track the render's (the same DT failure, milder). But it is strictly more selective than render-power gating, and it is the canonical "robust to echo-path change / phase" estimator. Pairs naturally with the §4 LSA gain.
- **Cost:** low–moderate (a few-tap subband coloration filter + delay). **FS risk:** low.

---

## 6. Near-end-SIDE model: kurtosis / non-Gaussianity (estimate-free discriminator)

**"Wavelet-based ICA using maximisation of non-Gaussianity for AEC during double-talk," *Applied Acoustics*, 2015** ([sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0003682X15001139)); broader HOS-DTD reviews.

- **Mechanism:** near-end speech and (Gaussianized) residual echo have *different higher-order statistics*. Maximizing non-Gaussianity (kurtosis/negentropy) of the output isolates the near-end *without a DTD and without an X-Y coherence test* — it's a near-end-side property, not an X-dependent one. Reported ~5 dB ERLE gain in their setup.
- **Why this is the conceptually right axis for you:** it is the rare mechanism that does **not** consult the render or the linear estimate at all, so MSC<0.05 is irrelevant to it. Voiced/breath near-end has heavy-tailed, time-localized structure; a reverberant non-stationary echo of distant render speech is closer to Gaussian after the room/clock-drift smearing. **Temporal kurtosis of the error per band** (estimated over a short window) is a single scalar-per-band feature with temporal context that you have NOT listed among your failed single-frame features — worth a direct 800-case probe before any ICA machinery.
- **Skeptical caveats:** (a) ICA proper needs multiple observations / is heavy and unstable in real time — do not port the full FastICA; port only the *kurtosis-as-a-feature* idea. (b) When the near-end is *also* breath (low-energy, noise-like), its kurtosis advantage over echo shrinks — this is exactly your hardest sub-case, so temper expectations. (c) Reverberant FS echo of *near-end-of-the-far-room* speech can itself be peaky. Treat as a *complementary* discriminator feeding the SPP prior (§4), not a standalone gate.
- **Cost:** low (windowed 4th-moment per band). **FS risk:** low if used as a soft prior. This is the highest-novelty cheap probe in the report.

---

## 7. Cepstral / pitch / harmonicity near-end discriminator — promising in theory, thin in the AEC-DT literature (a finding in itself)

Searches for pitch/cepstral-peak *as a DT echo discriminator* returned VAD and ASR front-end work ([Cepstral Peak Prominence harmonicity, PHCC](https://www.isca-archive.org/icslp_2000/gu00_icslp.html), [LTSV/spectral-flatness VAD, EURASIP 2013](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/1687-4722-2013-21)), but essentially **no DSP-only paper uses cepstral-peak/harmonicity to separate near-end from non-stationary far-echo during DT.**

- **Why it's still worth a probe:** voiced near-end has a sharp cepstral peak at its *own* pitch period; reverberant/drifted far-echo of far-room speech has its harmonicity *smeared* by the room IR and any clock drift, so its cepstral peak is lower and at a different quefrency. A **cepstral-peak-prominence-of-the-error-vs-CPP-of-the-delay-aligned-render** ratio is a temporal-context feature in a *different domain* that your listed feature set does not cover.
- **Skeptical caveats:** breath/fricatives are *unvoiced* — no pitch — so CPP protects voiced near-end but NOT the breath you most care about (your MEMORY's "breath arc"). And ql7 static-weak FS at high coherence may retain harmonicity. So CPP is at best a partial, voiced-only protector. Given your breath focus, rank this below §1/§4/§6.
- The *absence* of prior art here means: if it works on your data it's genuinely novel, but the literature gives you no reason to expect it to crack the unvoiced-breath case.

---

## 8. Things to explicitly NOT pursue (bucket A — they assume a usable estimate)

- **Normalized cross-correlation DTD (Benesty/Gänsler 2001/2006; Tashev's references [3][4]; fast-NCC).** ([researchgate NCC](https://www.researchgate.net/publication/221266425_Normalized_Double-Talk_Detection_Based_on_Microphone_and_AEC_Error_Cross-Correlation), [fast-NCC ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165168405003166)) — all are X-Y / Ŷ-Y correlation statistics; identically dead at MSC<0.05. Your "DELAY-ALIGNED X-Y coherence" already covers this family's information and failed. Skip.
- **Coherence-based DTD as the discriminator** (partial-coherence DTD, EUSIPCO 1998; vocal.com) — same wall.
- **Two-path/shadow divergence as discriminator** (§2) — blind when the main has no estimate; your own F2.2/P52 closures confirm.
- **Normalized-error-power DTD** ([IEEE 4555455](https://ieeexplore.ieee.org/document/4555455/)) — your E²/Y² feature, already failed.

The unifying reason all of bucket A fails *for you specifically*: they are **render-referenced** statistics, and your defining condition is that the render reference has lost its statistical link to the mic. The only escape routes are render-referenced-but-ENVELOPE-domain (§5), render-referenced-but-TEMPORAL (§1), or NOT-render-referenced near-end-side models (§4 DD/SPP memory, §6 kurtosis, §7 cepstral).

---

## Recommended evaluation order (cheap→expensive, by expected payoff on the breath/64% wall)

1. **§6 temporal/per-band kurtosis of the error** as a soft SPP prior — cheapest, highest novelty, render-independent, directly probes a feature you haven't tried. One-day 800-case probe.
2. **§4 DD a-priori-SER + MMSE-LSA gain** replacing your hard RES gate — the principled, least-distorting near-end-preserving *gain rule*; sets your Speex↔AEC3 Pareto point with less breath damage per dB. This is the structural fix to "RES is the near-end killer" in your MEMORY.
3. **§1 Song&Shin temporal+harmonic residual-echo PSD with a low-false-reject DT gate** — feeds a *better λ̂echo* into the §4 gain; raises FS suppression while the conservative gate protects DT. Isolate the harmonic term vs the temporal term per class.
4. **§5 Faller envelope-space echo-magnitude estimate** as the `usable_linear` fallback when complex MSC<0.05 — replaces render-power gating with a phase-robust magnitude estimate.
5. **§3 Tashev adaptive-threshold normalization** applied to whichever statistic survives — cheap robustness, and a per-class router for your ql7-vs-movement split.
6. **§7 cepstral-peak** only if you still need voiced-near-end protection after the above; note it does nothing for unvoiced breath.

**Honest bottom line:** no paper claims to separate near-end from *non-stationary, render-decorrelated* echo on a single frame — consistent with your empirical wall. Every credible lever in the literature buys separation by either (a) adding *temporal memory* (DD-SER, temporal residual regression, adaptive thresholds), (b) moving to a *coherence-robust domain* (spectral envelope/magnitude), or (c) consulting a *near-end-side* statistic that never touches the render (kurtosis/non-Gaussianity, cepstral harmonicity). Combine (b)+(a)+(c) as PSD-estimate / gain-rule / soft-prior layers respectively; do not expect any one of them to be a clean DT discriminator on its own.

**Key sources:** [Song & Shin 2020, Appl.Sci.](https://www.mdpi.com/2076-3417/10/15/5291) · [Tashev coherence-adaptive DTD](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Tashev_ET11_DoubleTalkDetector_final.pdf) · [Habets MMSE-LSA multiple interferences 2006](https://israelcohen.com/wp-content/uploads/2018/05/IWAENC2006_Habets.pdf) · [Habets joint dereverb+RES](https://www.researchgate.net/publication/224332472) · [a-priori SDR reduced-distortion AES/ANS](https://www.researchgate.net/publication/220242534) · [soft-decision AES](https://www.researchgate.net/publication/221491798) · [Faller & Chen spectral-envelope AES 2005](https://www.researchgate.net/publication/3334140) · [Faller & Tournery robust echo control](https://www.semanticscholar.org/paper/Robust-Acoustic-ECHO-Control-using-a-Simple-ECHO-Faller-Tournery/bc3142b1995f938c77badd9dabcf8b32a35a078c) · [wavelet-ICA non-Gaussianity DT 2015](https://www.sciencedirect.com/science/article/abs/pii/S0003682X15001139) · [Benesty/Gänsler fast-NCC DTD](https://www.sciencedirect.com/science/article/abs/pii/S0165168405003166) (bucket A, flagged) · [foreground/background two-filter EP0872962A2](https://patents.google.com/patent/EP0872962A2/en) (bucket A, flagged) · [Cohen/Berdugo DTD-aided RES 2022](https://israelcohen.com/wp-content/uploads/2022/09/acoustics-04-00039.pdf) (NN, taxonomy only).

Extracted source PDFs saved at: `/Users/mingyu/.claude/projects/-Users-mingyu-Desktop-novatek-SE/2cc3ad9d-517c-44a3-9fac-9e87ea2d5a7c/tool-results/webfetch-1780815052738-ys102f.pdf` (Tashev) and `.../webfetch-1780815052577-83ysa5.pdf` (Cohen DTD-RES).
---
## Report 3 — Speex leak mechanism dissection

I now have the complete mechanism with verbatim lines and line numbers across both files. I have everything needed to write the dissection and the concrete proposal. Let me compile the final answer.

---

# Speex MDF + Preprocess: leak/residual mechanism, why it preserves breath, why it leaks FS — and a better leak estimate

Source: `xiph/speexdsp` tag **SpeexDSP-1.2.1** (matches the installed binding's headers `/opt/homebrew/include/speex/`, v1.2.1). No `.c` on disk — it's the compiled wheel `_speexdsp.cpython-39-darwin.so`; I dissected the canonical 1.2.1 source. Files: `libspeexdsp/mdf.c` (echo canceller) and `libspeexdsp/preprocess.c` (RES). Geometry note: Speex MDF is a time-domain-partitioned NLMS, NOT a per-bin Kalman like our PBFDKF — that distinction is load-bearing for portability (see §5).

## 1. The leak estimate (`mdf.c`)

The leak is a **scalar** (one number per frame, not per-bin) measuring "what fraction of the filter-output power `|Ŷ|²` is actually correlated with the residual error power `|E|²`." It is a normalized **power-domain covariance** computed on the *spectrally-whitened* PSDs:

```
MIN_LEAK = .005f                                    // mdf.c ~L96

Eh = PSEUDOFLOAT(st->Rf[j] - st->Eh[j]);            // ~L1038  whitened error PSD
Yh = PSEUDOFLOAT(st->Yf[j] - st->Yh[j]);            // ~L1039  whitened echo-est PSD
Pey = FLOAT_ADD(Pey, FLOAT_MULT(Eh,Yh));            // ~L1041  Σ_j (cov) over bins
Pyy = FLOAT_ADD(Pyy, FLOAT_MULT(Yh,Yh));            // ~L1042  Σ_j (var)

st->Eh[j] = (1-spec_average)*st->Eh[j] + spec_average*st->Rf[j];   // ~L1048 running mean
st->Yh[j] = (1-spec_average)*st->Yh[j] + spec_average*st->Yf[j];   // ~L1049

tmp32 = MULT16_32_Q15(st->beta0, Syy);              // ~L1051  adaptive time constant…
if (tmp32 > MULT16_32_Q15(st->beta_max, See))       // ~L1052  …clamped by error energy
   tmp32 = MULT16_32_Q15(st->beta_max, See);
alpha   = FLOAT_DIV32(tmp32, See);                  // ~L1054  alpha ∝ Syy/See, capped
alpha_1 = FLOAT_SUB(FLOAT_ONE, alpha);              // ~L1055

st->Pey = alpha_1*st->Pey + alpha*Pey;              // ~L1057  leak smoothing
st->Pyy = alpha_1*st->Pyy + alpha*Pyy;              // ~L1058

if (st->Pey < MIN_LEAK*st->Pyy) st->Pey = MIN_LEAK*st->Pyy;   // ~L1062 floor 0.5%
if (st->Pey > st->Pyy)          st->Pey = st->Pyy;            // ~L1064 cap 100%
st->leak_estimate = FLOAT_DIVU(st->Pey, st->Pyy) << 14;       // ~L1067  ∈ [0.005, 1.0]
```

Key facts:
- **`leak_estimate = cov(|E|,|Ŷ|) / var(|Ŷ|)`** across frequency, on the *de-meaned* (whitened) PSDs `Rf−Eh`, `Yf−Yh`. Subtracting the slowly-tracked means `Eh`,`Yh` is a **spectral-flatness / decorrelation** trick: it removes the stationary baseline so the covariance is driven by *co-fluctuation* of error and echo-estimate power. A near-end voiced segment fluctuates `|E|` without a matching `|Ŷ|` fluctuation → contributes ~0 to `Pey`, raising `Pyy` only → leak goes **down**, not up.
- **Time constant is adaptive**: `alpha ∝ Syy/See`, capped at `beta_max`. When echo dominates the error (`Syy` large, `See` mostly echo) leak adapts fast; when the error is dominated by near-end/noise (`See` large vs `Syy`) `alpha` shrinks → leak is **frozen**, not updated by near-end energy. This is the first near-end protection.
- **Floor `MIN_LEAK = 0.005`** (−23 dB): never claims *zero* echo. **Cap = 1.0**: never claims more residual echo than the entire filter output power.

## 2. How the residual-echo PSD is exported (`mdf.c speex_echo_get_residual`, ~L770–790)

```c
void speex_echo_get_residual(SpeexEchoState *st, spx_word32_t *residual_echo, int len)
   st->y[i] = MULT16_16_Q15(st->window[i], st->last_y[i]);     // ~L775 window LAST filter output
   spx_fft(st->fft_table, st->y, st->Y);
   power_spectrum(st->Y, residual_echo, N);                    // ~L777 residual_echo[i] = |Ŷ[i]|²
   // float path:
   if (st->leak_estimate > .5) leak2 = 1;  else leak2 = 2*st->leak_estimate;   // ~L785
   residual_echo[i] = MULT16_32_Q15(leak2, residual_echo[i]);  // ~L790 scale echo PSD by leak
```

So the residual echo PSD handed to the preprocessor is **`residual_echo[i] = leak2 · |Ŷ[i]|²`**, where `leak2 = min(1, 2·leak_estimate)`. Crucially it is a **single scalar `leak2` applied flat across all 257 bins** — the per-bin shape comes only from `|Ŷ|²`, never from any per-bin error correlation.

## 3. How the RES uses it (`preprocess.c`)

```c
NOISE_SUPPRESS_DEFAULT  -15;  ECHO_SUPPRESS_DEFAULT  -40;  ECHO_SUPPRESS_ACTIVE_DEFAULT -15;  // ~L23-25
st->echo_noise[i] = MAX32(MULT16_32_Q15(QCONST16(.6f,15), st->echo_noise[i]),
                          st->residual_echo[i]);             // ~L311  echo treated as extra noise, 0.6 decay
```

Echo is folded into the noise estimate (`echo_noise`), then the **decision-directed MMSE (Ephraim-Malah / OM-LSA-style)** runs over `noise + echo_noise + reverb`:

```c
tot_noise = 1 + noise[i] + echo_noise[i] + reverb_estimate[i];      // a posteriori SNR denom
post  = ps[i]/tot_noise - 1;                                        // a posteriori
gamma = .1 + .89*(old_ps/(old_ps+tot_noise))²;                      // DD weight
prior = gamma*max(0,post) + (1-gamma)*(old_ps/tot_noise);           // a priori SNR
... MMSE/qcurve gain ...
```

Two near-end-protecting layers sit on top:

```c
// global speech-presence Pframe via qcurve over smoothed SNR (zeta):
Pframe = .1 + .899*qcurve(Zframe/nbands);                                            // ~L?
effective_echo_suppress = (1-Pframe)*echo_suppress + Pframe*echo_suppress_active;    // ~L419

// gain floor (how hard echo is allowed to be suppressed):
noise_gain = min(1, exp(0.11513*noise_suppress)/2);                                  // ~L200
gain_ratio = min(1, exp(.2302585*(effective_echo_suppress-noise_suppress))/2);       // ~L201
gain_floor[i] = noise_gain * sqrt( (noise[i] + gain_ratio*echo[i]) / (noise[i]+echo[i]) );  // ~L?

// per-bin speech presence p blends MMSE gain toward the floor (NOT to zero):
tmp = p*sqrt(gain[i]) + (1-p)*sqrt(gain_floor[i]);  gain2[i] = tmp²;                 // ~L517-518
```

## 4. WHY it preserves breath, and WHY it leaks FS

**Why breath survives (4 stacked reasons):**
1. **Decorrelation in the leak covariance** — breath raises `|E|` without a correlated `|Ŷ|` rise; via `Eh/Yh` mean subtraction this *lowers* leak rather than raising it. The estimator structurally refuses to attribute near-end-driven error to echo.
2. **The leak is a single global scalar with a 0.5%→100% range and a `2×` cap on `leak2`** — it never spikes per-bin. A breath occupying a few HF bins cannot push the residual-echo PSD up in those bins, because the per-bin shape is `|Ŷ|²` (the *echo* estimate), and the breath isn't in `Ŷ`.
3. **`effective_echo_suppress` defaults to a *floor of −40 dB* (inactive) / −15 dB (active)** — `gain_ratio` lets echo be attenuated only by that bounded amount; the **gain floor is never 0**. Even when the RES decides "echo here," it pulls the gain to `gain_floor`, which is `sqrt`-mixed with the noise floor — low-level near-end leaks through at the floor instead of being gated to silence.
4. **`p`-blend (`p·gain + (1−p)·floor`)** means decision is soft, per-bin; a breath with even modest speech-presence `p` is partly preserved.

The net effect is exactly your observation: Speex **globally under-suppresses**. The `MIN_LEAK` floor and `2×` cap bound the residual estimate *low*, and the `−40/−15 dB` echo-suppress floors bound how hard it acts — so it cannot zero out low-level near-end.

**Why it leaks FS (the same mechanism, failure mode):**
- The leak is **one scalar over the whole spectrum** and the residual shape is **`|Ŷ|²` from the linear filter**. On your 64% non-modelable class, the true residual echo is **not** a scaled copy of `|Ŷ|²` — it's loudspeaker-nonlinear / clock-drifted / non-stationary, so it lives in bins where `|Ŷ|²` is *small* (the filter never modeled it). `leak2·|Ŷ|²` is therefore **tiny exactly where the real residual is large** → RES sees no echo there → leaks.
- The **covariance under-estimates** when `E` and `Ŷ` are real but *phase-incoherent*: nonlinear/drifted echo de-correlates `|E|` from `|Ŷ|` in the same way near-end does. The leak estimator **cannot tell non-stationary far-echo from near-end** — both look like "error uncorrelated with the linear echo estimate." This is your wall, restated in Speex's own variables: low `Pey/Pyy` is ambiguous between breath and non-modelable echo.
- `MIN_LEAK=0.005` and `leak2≤1` cap the residual *low* by design. Good for breath, fatal for a loud non-linear residual.

**The RER path (`mdf.c` ~L1490, the float branch) is the most portable single line:**
```c
RER = (.0001*Sxx + 3.*MULT16_32_Q15(st->leak_estimate,Syy)) / See;   // float path
if (RER < Sey*Sey/(1+See*Syy)) RER = Sey*Sey/(1+See*Syy);            // coherence lower bound
if (RER > .5) RER = .5;                                              // cap at 50%
```
This is the "residual-to-error ratio." Note the **`Sey²/(See·Syy)` lower bound** — that is exactly a (delay-free) **magnitude coherence²** between error `e` and echo-estimate `y` in the *time domain*. Speex uses coherence only as a *floor* on RER (never lets RER drop below the coherence), and `.0001*Sxx` injects a small far-power term. This is the same coherence feature your audit found insufficient — Speex doesn't solve the discriminator either; it just floors low and caps at 0.5.

## 5. Concrete proposal: a near-end-protected, convergence-aware, per-bin leak

Three portable ideas, ranked by expected payoff against your wall. None require NN; all are DSP and fit the 160/512/257 geometry.

**(P1) Per-bin leak with a convergence-floor that decouples it from `|Ŷ|²` — the core fix for FS leak.**
Speex's fatal flaw is `residual = scalar_leak · |Ŷ|²`. On non-modelable echo the real residual is *not* shaped like `|Ŷ|²`. Replace the scalar with a **per-bin leak covariance** computed on whitened PSDs (you already have `Rf`,`Yf` equivalents):
```
Pey[k] = α·(|E_k|−Eh_k)(|Ŷ_k|−Yh_k) + (1−α)·Pey[k]
Pyy[k] = α·(|Ŷ_k|−Yh_k)²          + (1−α)·Pyy[k]
leak[k] = clip(Pey[k]/Pyy[k], MIN_LEAK_k, 1)
```
but then form the residual PSD as
```
residual[k] = max( leak[k]·|Ŷ_k|² ,  conv_floor[k]·|X_aligned,k|² )
```
where the second term is a **render-power floor gated by per-bin filter convergence**: `conv_floor[k]` is large where the linear filter is *not* converged (your `usable_linear`/ERLE per bin is low) and ~0 where it is converged. This is the missing piece: on the 64% class where `|Ŷ|²` is uninformative, you fall back to a *bounded* render-power residual **only in the unconverged bins**, so you don't blow away the converged-bin near-end. This is a principled blend of Speex's leak (converged bins) and AEC3's NonLinear render path (unconverged bins), selected **per bin** by convergence — not globally by far-power. It directly attacks "leak on non-modelable echo" while keeping the leak-floor breath protection on the modelable bins.

**(P2) Near-end-protected leak adaptation (port the `alpha ∝ Syy/See` freeze, make it per-bin).**
Speex already freezes leak adaptation when error is near-end-dominated (`alpha` shrinks). Port this *per bin*: only update `Pey[k]/Pyy[k]` when `|Ŷ_k|² > β·See_k` (echo present in that bin). In bins dominated by near-end, **hold** the last converged leak instead of letting it drift. This stops a breath from corrupting the leak estimate, and—combined with P1's convergence floor—means breath bins keep their (low, converged) leak while non-modelable echo bins use the render floor. Cost: one per-bin gate, no new feature needed.

**(P3) Coherence-floored RER as a *soft* gain floor, not a hard echo decision.**
Adopt Speex's RER structure but use *your* delay-aligned coherence as the lower bound (you already compute it), and—critically—use RER to set the **gain floor**, not to zero the gain. Mirror `gain_floor = noise_gain·sqrt((noise + gain_ratio·echo)/(noise+echo))` and the **`p·gain + (1−p)·floor` soft blend**. This is the breath-preservation mechanism that AEC3 lacks: AEC3 gates to a hard NonLinear suppression; Speex blends to a *bounded floor*. Porting the soft-floor blend (with `echo_suppress_active = −15 dB`, not −∞) gives you a controllable knob: how much FS to trade for breath is literally the `effective_echo_suppress` value, set per-frame by speech-presence `Pframe`. This is the "min-gain floor axis / gentle preset" lever your memory already flagged — Speex shows the canonical form: floor it, don't gate it, and let `p`/`Pframe` interpolate.

**Why this can beat the wall where single-frame features failed:** your discriminator search failed because *no single-frame scalar* separates non-modelable FS-echo from breath. P1 doesn't try to separate them with a feature — it changes the **residual model per bin by filter convergence state** (a *filter*-state signal, not a near/far signal), so the FS-vs-breath decision is made by "is the linear filter trustworthy in this bin," which *is* separable (converged ql7 HF bins → leak·|Ŷ|² is credible; unconverged movement bins → render floor catches the leak). P2/P3 then ensure the breath that survives in converged bins is floored-not-gated. The trade you keep paying is P3's `echo_suppress_active` — but now it's a per-bin, convergence-aware floor instead of a global far-power gate, which is the whole point.

**Cautions for porting into PBFDKF:** Speex's leak runs on a single NLMS path; your PBFDKF already exposes per-bin `P`/Kalman gain — use that directly as `conv_floor[k]` (high `P_trace`/low Kalman convergence → high render floor) rather than re-deriving convergence from ERLE. And keep `MIN_LEAK`-style flooring **per bin** (your memory notes a scalar `erl_estimate` is the #1 per-band candidate — same disease). Do NOT port `leak2 = 2×leak` globally; that 2× cap is what makes Speex leak FS — replace with the per-bin `max(leak·|Ŷ|², conv_floor·|X|²)`.

**Files / lines referenced (SpeexDSP-1.2.1):**
- `libspeexdsp/mdf.c` ~L96 `MIN_LEAK=.005`; ~L1038-1067 leak covariance (`Eh/Yh/Pey/Pyy/leak_estimate`); ~L1490 `RER` float formula `(.0001*Sxx + 3.*leak*Syy)/See` with `Sey²/(See·Syy)` coherence floor and `.5` cap; ~L770-790 `speex_echo_get_residual` (`leak2 = 2·leak`, `residual = leak2·|Ŷ|²`); ~L618 `spec_average = frame_size/sampling_rate`.
- `libspeexdsp/preprocess.c` ~L23-25 `ECHO_SUPPRESS_DEFAULT=-40 / ACTIVE=-15 / NOISE_SUPPRESS=-15`; ~L311 `echo_noise = max(.6·echo_noise, residual_echo)`; ~L200-208 `compute_gain_floor` (`0.11513`, `.2302585`, `sqrt((noise+gain_ratio·echo)/(noise+echo))`); ~L419 `effective_echo_suppress = (1-Pframe)·suppress + Pframe·suppress_active`; ~L517-518 `gain2 = (p·sqrt(gain) + (1-p)·sqrt(gain_floor))²`.

No source `.c` exists on this machine (only the compiled wheel at `/Users/mingyu/Desktop/novatek/SE/.venv/lib/python3.9/site-packages/speexdsp/_speexdsp.cpython-39-darwin.so` and headers at `/opt/homebrew/include/speex/`); the above is the canonical 1.2.1 source matching the installed binding.
---
## Report 4 — AEC3 broad-robustness dissection

I now have full coverage of the mechanism. Default reverb decay = `default_len = 0.83` (mild = `nearend_len = 0.83`), so reverb is appended to R2 as a 0.83-decay leaky accumulation of render power. I have all the pieces needed to write the complete dissection with the precise breath-side reasoning. Let me compile the final answer.

# AEC3 Broad-Robustness Dissection: the render-power NonLinear path, and why it cannot preserve breath

All file:line refs are `/tmp/src/modules/audio_processing/aec3/` unless noted. Verified against the second extract dir (identical logic, line numbers within ±10). The verified-faithful `aec3_cli.cc:100` constructs a **default** `EchoCanceller3Config` with no field trials — this is the exact config whose breath-cut the memory measured, and it pins down which mechanisms are actually live (below).

## 1. The two-branch fork — where AEC3 switches linear → render-power R2

`residual_echo_estimator.cc:241` is the entire fork:

```
if (aec_state.UsableLinearEstimate()) {          // line 241 — LINEAR branch
    LinearEstimate(S2_linear, erle, R2);         // line 253: R2[k] = S2_linear[k] / erle[k]
    AddReverb(R2);
} else {                                          // line 263 — NONLINEAR branch (ql7 lands here)
    EchoGeneratingPower(...) -> X2;              // line 277
    NonLinearEstimate(echo_path_gain, X2, R2);   // line 291: R2[k] = X2[k] * echo_path_gain
    AddReverb(R2);                                // line 300 (model_reverb_in_nonlinear_mode=true)
}
```

**The NonLinear R2 (lines 110-119, 264, 277-292):**
- `echo_path_gain = GetEchoPathGain(..., early=true)` → `gain_amplitude²` where `gain_amplitude = ep_strength.default_gain = 1.0` (config.h:133). So in non-transparent mode **echo_path_gain = 1.0** — R2 is literally the windowed render power, with no shrinkage.
- `X2` is **not** the instantaneous render bin. `EchoGeneratingPower` (lines 132-166) takes, per bin, the **max** render power over a delay window `[delay−render_pre_window_size, delay+render_post_window_size]` = `[delay−1, delay+1]` blocks (config.h:174-175), aligned at `MinDirectPathFilterDelay()`. This gated-max is the credibility trick (see §6).
- A soft noise gate (`ApplyNoiseGate`, lines 122-130, `noise_gate_power=27509`, `noise_gate_slope=0.3`) and a stationary-noise-floor subtraction (lines 286-289, `stationary_gate_slope=10 × X2_noise_floor`) remove the render's own stationary noise so steady far-end noise doesn't drive suppression.
- Reverb is then **added** (lines 295-301): `model_reverb_in_nonlinear_mode = true` (config.h:176), so even in the NonLinear branch a leaky reverb tail (`reverb_model.cc:29-40`, decay `default_len=0.83`) is layered on top, keyed off `echo_path_gain` again (`UpdateReverbNoFreqShaping`, line 397).

Net: on the non-modelable FS case **R2 ≈ delay-aligned, window-maxed render power × 1.0 + 0.83-decay reverb tail**. This is the "credible-enough" echo estimate. It is structurally **the Speex-style power estimate but at gain 1.0 instead of a covariance-leak fraction** — that is exactly why AEC3 holds FS echo where Speex leaks it.

## 2. The gate: `UsableLinearEstimate` / `any_filter_converged` (aec_state)

`UsableLinearEstimate()` (aec_state.h:54-57) = `filter_quality_state_.LinearFilterUsable() && use_linear_filter`. `LinearFilterUsable()` is set in `FilteringQualityAnalyzer::Update` (aec_state.cc:421-462):

```
sufficient_data_to_converge_at_startup = filter_update_blocks_since_start_ > 0.4*kNumBlocksPerSecond   // ~0.4s active render
sufficient_data_to_converge_at_reset   = ... > 0.2*kNumBlocksPerSecond
overall_usable = startup && reset && (external_delay || convergence_seen_) && !transparent_mode   // lines 445-455
```

`convergence_seen_` latches on `any_filter_converged` (line 433), which comes from **`subtractor_output_analyzer.cc:46-47`**:

```
refined_filter_converged = e2_refined < 0.5f * y2 && y2 > 50*50*kBlockSize     // only 3 dB error-reduction!
coarse_filter_converged_strict = e2_coarse < 0.05f * y2 && y2 > ...
filters_converged = refined || coarse_strict
```

**This is the crux for ql7-class FS echo.** The convergence bar is `e²<0.5y²` — a mere **3 dB** of linear cancellation, with `y2` above an absolute floor. So:
- **ql7 (static weak-linear, HIGH coherence):** filter gets ≥3 dB → `convergence_seen_` latches → `UsableLinearEstimate=true` → LINEAR branch, R2 = S2_linear/ERLE. The render-power path is **not** what saves ql7; the *linear* path does, because the bar to qualify as "converged" is very low. ERLE then divides (`max_l=4, max_h=1.5`, i.e. capped at 6 dB LF / 1.8 dB HF — config.h:124-125).
- **Non-stationary / movement / clock-drift (LOW coherence, e²≈y²):** never clears 3 dB → `convergence_seen_` stays false → `UsableLinearEstimate=false` → **NonLinear render-power branch**, R2 = X2 × 1.0. This is the fallback that holds FS echo when the filter is blind. Once it falls here, it is identical-in-kind to Speex's power gate, just at unity gain and delay-window-maxed.

Note `convergence_seen_` is a **latch** (`|| any_filter_converged`, line 433) reset only on echo-path change (`Reset()`, aec_state.cc:170-173 → `filter_quality_state_.Reset()`). Once a case has *ever* shown 3 dB, AEC3 stays in the linear branch unless an echo-path-change reset fires.

## 3. StationarityEstimator — present, but OFF in the default config

`stationarity_estimator.cc` builds a per-bin "render is stationary" flag: accumulate render power over a `kWindowLength` window, compare to a min-statistics noise floor (`EstimateBandStationarity`, lines 100-121: `stationary = acum_power < 10 × noise`), with hangover (lines 131-140) and 3-bin smoothing (lines 142-155).

It feeds suppression in exactly **one** place — `residual_echo_estimator.cc:306-316`:
```
if (aec_state.UseStationarityProperties()) {
    GetResidualEchoScaling(residual_scaling);    // per bin: 0.0 if render stationary, else 1.0
    R2[ch][k] *= residual_scaling[k];            // ZEROES R2 on stationary-render bins
}
```
`GetResidualEchoScaling` (echo_audibility.h:40-51) sets `residual_scaling[band]=0` when the render is stationary in that band (and the filter has had time to converge, aec_state.cc:115-126). **A zero R2 means no echo penalty → near-end fully preserved in that bin.**

**But `UseStationarityProperties()` returns `config_.echo_audibility.use_stationarity_properties` (aec_state.h:74-76), which defaults to `false` (config.h:149) and is only turned on by the `WebRTC-Aec3EnforceStationarityProperties` field trial (echo_canceller3.cc:461-462).** The default `aec3_cli` config does **not** set it. So in the AEC3 the memory measured, this entire per-bin near-end-preserving scaler is **inert**. That is the single biggest reason "faithful AEC3 cuts breath as hard as us": its one near-end-aware residual lever is compiled-in but switched off by default.

## 4. Near-end awareness that IS live by default — DominantNearendDetector

This is AEC3's only active near-end protection in the default config, and it operates on the **suppressor gain**, not on R2. `suppression_gain.cc:216` selects `nearend_params_` vs `normal_params_` based on `dominant_nearend_detector_->IsNearendState()`. The nearend tuning is far gentler (config.h:218: `enr_transparent=1.09, enr_suppress=1.1` vs normal `0.3/0.4`) — i.e. in nearend state it tolerates much higher echo-to-nearend before suppressing.

The detector (`dominant_nearend_detector.cc:37-81`) is a **low-frequency ENR/SNR test on bins 1–15** (lines 46-49):
```
ne_sum  = Σ nearend[1..15];  echo_sum = Σ R2[1..15];  noise_sum = Σ comfort_noise[1..15]
nearend if (echo_sum < 0.25*ne_sum) && (ne_sum > 30*noise_sum)   // lines 58-60, trigger 12 blocks
exit nearend if echo_sum > 10*ne_sum                              // lines 72-74
```
With `use_unbounded_echo_spectrum=true` (config.h:236, suppression_gain.cc:404-405) the `echo_sum` here is **R2_unbounded** — the same render-power estimate.

**Why this fails the breath case — the core mechanical answer:** `echo_sum` is R2, which in the NonLinear branch is `render_power × 1.0`. During far-end singletalk with a quiet near-end breath, the render is loud → R2 is large → `echo_sum > 0.25·ne_sum` is *false* → the detector stays out of nearend state → `normal_params_` (aggressive 0.3/0.4 masking) apply → breath is cut. The detector keys near-end-ness on **R2 vs near**, and R2 is large *precisely because the far-end is active*, which is also exactly when breath occurs. It cannot fire during the very condition that endangers breath. It is a double-talk *level* detector, not a near-end *presence* detector — structurally the same wall the memory already documented ("RES falls back to gating on far-end POWER").

## 5. Onset detection / onset-ERLE — does not touch the NonLinear branch

`subband_erle_estimator.cc:144-160, 192-212`: onset compensation tracks a separate lower `erle_during_onsets_` and bleeds the live ERLE toward it (×0.97) after a hold, so the first blocks of a render onset are not over-credited with ERLE. This only modulates the **LINEAR** branch's `R2 = S2_linear/ERLE` (a smaller ERLE → larger R2 → more suppression at onsets). It is a convergence-safety mechanism, **not** a near-end lever, and it is irrelevant to the NonLinear render-power branch where ql7-fallback echo lives.

## 6. Why the render-power path is *credible* (the real porting target)

What makes AEC3's NonLinear R2 trustworthy where a naive `X²` would be garbage, and what is genuinely worth porting:
1. **Delay alignment + window-max** (`EchoGeneratingPower`, lines 132-166; window `[delay−1, delay+1]`): R2 is the render power *at the acoustic delay*, taking the max over a 3-block window so a slightly-wrong delay or a render transient still produces a non-zero, time-correct echo bound. This is what your DELAY-ALIGNED X-Y feature was trying to be, used here as a *power bound* rather than a discriminator.
2. **Stationary-render subtraction** (lines 286-289) + **noise gate** (lines 122-130): the render's own steady noise floor (`min_statistics`, `UpdateRenderNoisePower` lines 326-360) is removed, so only the *non-stationary, time-varying* render power (the part that actually produces audible non-linear echo) drives R2. This is the closest AEC3 gets to "non-stationary echo" awareness — it is in the **render**, not the mic.
3. **Reverb tail at decay 0.83** (lines 295-301, reverb_model.cc:29-40): a leaky-integrated render-power tail is added, covering the part of the echo the 52 ms filter can't reach.

**Porting verdict:** items (1)+(2)+(3) are the mechanism that gives AEC3 strength on the ql7/movement class — a delay-aligned, window-maxed, stationary-subtracted render-power R2 at gain 1.0, plus a 0.83 reverb tail. This is portable into your RES as a *fallback R2 when usable_linear is closed*, and it would close your FS-echo leak relative to Speex. **But it is the same power-gating you already do** — it holds FS echo by being echo-priority, exactly because R2 scales with render power.

## 7. The precise reason it cannot help the breath side

The breath problem is fundamentally unsolved by anything in AEC3's default path, for one structural reason that recurs at every stage:

- **R2 in the NonLinear branch is a monotone function of render power** (line 116: `R2 = X2 × echo_path_gain`, echo_path_gain=1.0, no near-end term). Every downstream near-end decision keys off R2: `GainToNoAudibleEcho` uses `enr = echo/nearend` (suppression_gain.cc:219), and `DominantNearendDetector` uses `echo_sum vs ne_sum` (lines 58-60). When the far-end is active and the near-end is a faint breath, R2 is large and `nearend` is small → high ENR → suppression fires → breath dies. There is **no signal anywhere in this branch that distinguishes "loud render that produced echo" from "loud render coexisting with quiet near-end breath,"** because the only near-end evidence used is the *level ratio*, and that ratio is unfavorable exactly during far-end activity.
- The **one** mechanism that *could* preserve breath without leaking echo — the StationarityEstimator residual-scaling (§3), which zeroes R2 per-bin when the *render* is stationary — is **OFF by default** and, even when on, only helps when the *far-end* is stationary noise (it preserves near-end during stationary far-end). It does **nothing** for non-stationary far-echo, which is the unsolved class: a non-stationary render is by definition *not* flagged stationary, so `residual_scaling=1.0`, R2 unscaled, breath cut. **The stationarity lever and the non-modelable-echo class are mutually exclusive by construction.**

So AEC3 confirms your falsified-premise memory exactly: it is echo-priority by design; its render-power NonLinear path is a strength to port for FS-echo holding, but it carries **zero** near-end-discriminating information — it preserves breath only via (a) the gentle `nearend_tuning` once a *low-frequency level* test fires (which cannot fire during far-active breath), and (b) a default-OFF render-stationarity scaler that is orthogonal to the non-stationary-echo class. There is no AEC3 component that separates near-end breath from non-stationary far-echo when the linear filter gives no credible estimate; AEC3 simply chooses echo over breath at that point.

## Portable / not-portable summary

- **PORT (closes FS leak, echo-priority):** the NonLinear R2 construction — delay-aligned window-max render power (residual_echo_estimator.cc:132-166) × unity echo_path_gain, minus stationary-render noise floor (lines 286-289, 326-360), plus 0.83-decay reverb tail (lines 295-301), gated by the **3 dB** `e²<0.5y²` convergence latch (subtractor_output_analyzer.cc:46-47) deciding linear-vs-render. This is what gives ql7/movement a credible-enough echo bound. It will **not** preserve breath — it is render-power-keyed.
- **PORT (conditional breath win, but ONLY for stationary far-end):** StationarityEstimator → per-bin `residual_scaling∈{0,1}` (stationarity_estimator.cc + echo_audibility.h:40-51), wired at residual_echo_estimator.cc:306-316. Off by default; enable via the config field. Helps breath during *stationary* far-noise; **useless** on the non-stationary class.
- **NOT PORTABLE for the unsolved case:** `DominantNearendDetector` (dominant_nearend_detector.cc) — it is a far-vs-near *level* test on bins 1–15 using R2, structurally unable to fire during far-active breath; it is the same far-power gate you already have, not a new discriminator.
- **Bottom line:** AEC3 has no mechanism that distinguishes near-end breath from non-stationary far-echo when the filter is blind. Its broad robustness comes from a *better echo bound* (render-power at gain 1.0, delay-aligned, stationary-subtracted), not from better near-end discrimination. The breath wall is intrinsic to the render-power formulation; no single-frame feature inside AEC3 separates the two, consistent with every discriminator you have already falsified.