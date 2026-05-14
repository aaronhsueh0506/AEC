# v3.14 Volterra non-linear inverse — design lock

**Status**: DRAFT design lock (research + design only; no production code change). Foundation document for v3.14 implementation sprints. Implementation candidate is multi-quarter (≥6 months) and explicitly out of v3.13 scope.

**Date**: 2026-05-14
**Branch**: `feature/v3.11-route-a` (worktree `agent-a30454b52f11e68d0`)
**Predecessor closure**: [docs/v3_13_arc_closure.md](v3_13_arc_closure.md)
**Predecessor mechanism docs**: [docs/v3_13_e4_s6a_s6b_verdict.md](v3_13_e4_s6a_s6b_verdict.md), [docs/v3_13_e5_closure_verdict.md](v3_13_e5_closure_verdict.md)

---

## 1. Background

### 1.1 What v3.13 ruled out

Two architectural arcs (E4 NLP amplitude mask, E5 saturation deepening) closed CANNOT SHIP within v3.13. Both arcs landed on the same physics wall:

* **E4 (amplitude mask)**: harmonic-pinned σ=50 Hz Gaussian mask attenuating residual `Y[k,t]` by a real-valued multiplicative gain `m[k,t] ∈ [g_min_lin, 1.0]`. Linear ERLE was metric-positive (+0.49 to +1.60 dB across an aggression sweep g_min ∈ {-12, -18, -24, -30} dB). Subjective listen verification across 3 listen-validated NL cases at all four aggression levels: **NO AUDIBLE NL REDUCTION** at any aggression level. At -30 dB the mask is provably attacking (voice formants begin disappearing) but the perceived NL character (爆掉 / 無線電) is unchanged. Asymptote pattern (+0.4 dB ERLE gain across 18 dB more attack) confirms the mask saturates against the harmonic peaks while the dominant NL energy lives off-peak / phase / transient.

* **E5 (filter-protection action)**: per-frame Pearson `r(mic, lpb)` in 0.7-0.95 mic-peak band, threshold 0.35, 5-frame hold. Detector quality is excellent (0/200 NE FP on smoke; 14-56% real-NL fire rate). Four action variants (S2 / S3 / S4a / S4b) all sit on a single FS-vs-DT trade-off line with slope ≈ -0.5 dB DT per +1 dB FS. Best variant (S4b) achieves FS_static Δecho +0.095 but DT_static Δdeg -0.052 — far past the -0.005 hard bar. Mechanism: the same correlation signature appears on FS-NL frames (acoustic NL — desired fire) AND DT high-echo frames (mic = echo + NE voice; same cross-band correlation). The detector cannot orthogonalise.

### 1.2 The physics wall (restated)

Both closures hit a **common ceiling**: any **multiplicative spectral mask** `m[k,t] · Y[k,t]` only modulates amplitude per frequency bin. It cannot:

1. **Change phase** — `m[k,t]` is real-valued; the spectrum gets scaled, not rotated. Loudspeaker overdrive ("爆掉") generates significant phase distortion that no real-valued mask can address.
2. **Undo time-domain non-linear transfer** — the loudspeaker / amp NL `mic_NL = h_nl(ref)` is a memory-bearing non-linearity; the inverse `h_nl⁻¹` is itself memory-bearing. Frequency-domain amplitude masks have no time-domain memory of the NL transfer.
3. **Resolve sub-frame transients** — codec NL ("無線電") and clip-attack artifacts are below the FFT analysis-window resolution. Per-frame amplitude scaling cannot localise to the transient.
4. **Distinguish frame types** — amplitude detectors fire identically on FS-NL and DT high-echo (E5 closure); even a perfectly precise amplitude detector cannot escape the trade-off line.

The architectural escape is **time-domain non-linear modelling** — explicit identification of the NL transfer `h_nl(ref → mic_NL)` followed by either (a) inverse application on the captured mic, or (b) generation of a synthetic NL-corrupted reference for a downstream linear AEC to subtract. Both subclasses are realisable via canonical Volterra-series adaptive filters.

### 1.3 The canonical fix

The literature's load-bearing insight (Stenger-Kellermann 2000, Mossi et al 2010, Kuech-Kellermann 2005, Kuech-Mabande-Enzner 2014, Malik-Enzner 2012) is that loudspeaker / amp-side non-linearity is well-modelled as a **memoryless or short-memory non-linear preprocessor cascaded with a linear FIR** (Hammerstein form), and its identification is well-posed via **second-order or third-order Volterra kernels** updated by adaptive algorithms (NLMS / RLS / state-space Kalman). Once the NL transfer is identified, the **subtractive** path (synthesise NL-corrupted echo and subtract from mic) reaches into phase and time-domain transient terms — which the multiplicative mask family cannot.

This is exactly the path WebRTC AEC3 takes with its `subtractive_nonlinear` variant alongside the linear filter — the residual_echo_estimator switches between linear-mode (S2_linear / ERLE) and nonlinear-mode (X² · echo_path_gain) per frame, but the upstream subtraction stage is non-linear-aware.

The v3.14 arc is the implementation of this canonical fix in our PBFDKF + ResFilter pipeline, using detector substrate harvested from v3.13 (E4.S2 SubtractiveNLP + E5.S3 mic-lpb correlation gate).

---

## 2. Literature review

The literature canon for Volterra-based NLAEC spans 1996-2025. We summarise each load-bearing reference with mechanism, position, and applicability to our PBFDKF substrate.

### 2.1 Birkett & Goubran 1995 (NNSP) — early NL preprocessor

* **Title**: "Nonlinear echo cancellation using a partial adaptive time-delay neural network" (1995 NNSP); follow-up 1996 with autocorrelation-based pre-distorter.
* **Mechanism**: time-delay feedforward neural network (TDNN) modelling loudspeaker NL, cascaded with NLMS linear filter. Memoryless polynomial pre-distorter variant in 1996.
* **Position**: pre-AEC on the reference path (the NL processor models the speaker → its inverse synthesises the NL-corrupted echo replica that the linear AEC subtracts).
* **Adaptation**: NLMS on linear stage; gradient-descent on NL stage.
* **Applicability**: historical foundation. The TDNN flavour conflicts with `feedback_no_nn_mention`; the polynomial pre-distorter is the v3.14-relevant lineage and is generalised by Stenger-Kellermann 2000.
* **URL**: [semanticscholar.org](https://www.semanticscholar.org/paper/Acoustic-echo-cancellation-using-NLMS-neural-Birkett-Goubran/f53de6969fd9af7a5502a4f76e7600df2b8d5753)

### 2.2 Stenger & Kellermann 2000 (Signal Processing) — memoryless NL preprocessor

* **Title**: "Adaptation of a memoryless preprocessor for nonlinear acoustic echo cancelling", *Signal Processing*, vol. 80, pp. 1747-1760, Sept 2000.
* **Mechanism**: cascade of memoryless polynomial / hard-clip preprocessor → linear FIR. The preprocessor models loudspeaker / amp memoryless NL (saturation, soft / hard clip). Joint adaptation of both stages with step-size normalisation. RLS or signal-orthogonalisation can be used for the polynomial stage; standard NLMS for the linear stage.
* **Position**: pre-PBFDKF on the reference signal. The reference `ref` is fed through the polynomial → produces `ref_NL`; the linear filter then identifies the linear acoustic path against `ref_NL`. Sometimes called the **Hammerstein structure**.
* **Cost**: O(N) per frame for polynomial of order N (typical N=3-5); negligible vs PBFDKF.
* **Convergence**: paper reports ~10 dB improvement at amplitude peaks vs linear-only.
* **Limitation**: memoryless model — cannot capture frequency-dependent NL (amp inductive ringing, speaker hysteresis). Real loudspeakers are *short-memory* non-linear.
* **Applicability**: **HIGH** — this is the canonical Hammerstein baseline that v3.14 should reproduce as Sprint S3 first-pass before any Volterra extension.
* **URL**: [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0165168400000852)

### 2.3 Stenger & Kellermann 2001 (Eusipco / IEEE TSP follow-on) — adaptive Volterra

* **Title**: "Adaptive Volterra filters for non-linear acoustic echo cancellation"; later extended in IEEE TSP.
* **Mechanism**: full second-order Volterra series `y[n] = Σ h₁[k] x[n-k] + Σ Σ h₂[k₁,k₂] x[n-k₁] x[n-k₂]`. The 2nd-order kernel `h₂` captures memory-bearing NL (intermodulation between delayed reference samples).
* **Position**: pre-PBFDKF on the reference; the Volterra output replaces the linear-only echo estimate; PBFDKF is then applied to remove the residual linear+NL echo.
* **Cost**: O(K²) per frame for kernel length K (or O(K) for diagonal-only kernels); significantly more than memoryless polynomial. Memory: K² coefficients for full kernel (e.g. K=64 → 4096 coefficients).
* **Convergence**: NLMS on Volterra kernel converges slowly (the kernel autocorrelation matrix is highly conditioned). RLS converges faster but is O(K²) compute.
* **Limitation**: full kernel storage prohibitive at long K; typically diagonal-only or banded kernels are used.
* **Applicability**: **HIGH** — this is the canonical Volterra extension. v3.14 should consider 2nd-order diagonal Volterra (not full) as Sprint S5 if memoryless polynomial proves insufficient.

### 2.4 Mossi et al. 2010 (IEEE Workshop) — assessment of linear-AEC robustness to NL

* **Title**: "An assessment of linear adaptive filter performance with non-linear distortions" (IEEE WASPAA 2010); related cascaded-NLAEC variants in IEEE-ICASSP 2011.
* **Mechanism**: characterises the failure modes of pure-linear AEC under loudspeaker NL — provides ERLE ceiling estimate (typically 15-20 dB cap when NL is present, vs 30-40 dB possible in linear regime). Compares APA, PBFDAF, PBFDKF, and AP-FRLS on identical NL-injected corpus.
* **Position**: diagnostic / benchmark paper — not an algorithm proposal.
* **Applicability**: **HIGH** — methodology for **how to construct the bench cohort**. Provides the synth-NL-injection methodology (tanh / cubic / hard-clip on ref before linear room IR convolution) that v3.14 Sprint S2 will need to validate detector + suppressor outside the 5-case real cohort.
* **Lesson for us**: per E4.S6 verdict, synth NL alone is not sufficient (the synth signature is too weak vs real NL). v3.14 must combine the Mossi synth corpus AND the v3.13 5/5 real cohort.
* **URL**: [ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/5495904/)

### 2.5 Kuech & Kellermann 2005 (IEEE TSP) — partitioned-block frequency-domain Volterra

* **Title**: "Partitioned block frequency-domain adaptive second-order Volterra filter", *IEEE Trans. Signal Processing*, 2005.
* **Mechanism**: extends partitioned-block frequency-domain LMS (PBFDLMS) — the algorithmic ancestor of our PBFDKF — to second-order Volterra. Allows different memory lengths for linear and quadratic kernels. Per-bin step-size normalisation (analogous to per-bin Kalman P / R in PBFDKF).
* **Position**: replaces our linear-only PBFDKF as a **single fused linear+quadratic** block-frequency-domain identifier. Pre-RES.
* **Cost**: O((N + N²) · log B) per partitioned block of length B; significantly more than linear-only PBFDKF but tractable. Memory: N + N² × n_partitions × n_freqs in DFT domain.
* **Convergence**: reports ~5-10 dB additional ERLE on NL corpus vs linear-only PBFDLMS at 5x compute.
* **Applicability**: **MEDIUM-HIGH** — this is the architectural cousin of our existing PBFDKF. Could be the v3.14 Sprint S6 unification path: **fuse the NL identification into PBFDKF itself** rather than as a separate preprocessor. But per P52 §1 `PBFDKF + ShadowFilter + PathChangeRegimeHandler` invariant, modifying PBFDKF body is high-risk; preprocessor cascade (Stenger-Kellermann form) is lower-risk.
* **URL**: [ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/1381748/)

### 2.6 Kuech, Mabande & Enzner 2014 (ICASSP) — state-space partitioned-block

* **Title**: "State-space architecture of the partitioned-block-based acoustic echo controller" (ICASSP 2014).
* **Mechanism**: Kalman-filter formulation of PBFD adaptive filter with explicit observation noise R per bin. Provides the canonical bridge from our existing PBFDKF (which is exactly this state-space derivation) to non-linear extensions.
* **Position**: not directly a non-linear paper, but the state-space framework is what supports the non-linear extension in Malik-Enzner 2012.
* **Applicability**: **HIGH** — this is the formal grounding of our PBFDKF. v3.14 inherits this framework and extends it.

### 2.7 Malik & Enzner — state-space frequency-domain NLAEC

* **Title**: "State-space frequency-domain adaptive filtering for nonlinear acoustic echo cancellation", *IEEE TASLP*.
* **Mechanism**: combines state-space Kalman PBFD framework with Volterra kernel. Per-bin Kalman gain on linear coefficients + per-bin step-size on quadratic kernel.
* **Position**: identical signal-flow position to Kuech-Kellermann 2005 (single fused identifier). Adds the Kalman uncertainty bookkeeping that our PBFDKF already has.
* **Applicability**: **MEDIUM-HIGH** — natural fit if we go the Kuech 2005 fused-identifier route. But same risk as 2.5 (modifies PBFDKF body).
* **URL**: [semanticscholar.org](https://www.semanticscholar.org/paper/State-Space-Frequency-Domain-Adaptive-Filtering-for-Malik-Enzner/5f26898e5c0fdab7a952652225b1f10c2e468b63)

### 2.8 Hoshuyama & Sugiyama 2003 / 2006 — spectral correlation NL suppressor

* **Title**: "Nonlinear acoustic echo suppressor based on spectral correlation between residual echo and echo replica" (IEICE 2006); earlier IWAENC 2003 abstracts.
* **Mechanism**: spectral correlation between residual `e` and echo replica `ŷ`; per-bin gain control. NOT a subtractive Volterra; it's a *post-processor suppressor* that conceptually sits where E4 / E5 sit in our stack.
* **Position**: post-PBFDKF on the residual. Multiplicative gain per bin.
* **Applicability**: **LOW for v3.14** — this falls into the amplitude-mask family that v3.13 closures explicitly ruled out. Listed for completeness (sometimes cited as "Hammerstein-like" but the action is multiplicative gain, not subtractive). DO NOT use this as the v3.14 mechanism.
* **URL**: [globals.ieice.org](https://globals.ieice.org/en_transactions/fundamentals/10.1093/ietfec/e89-a.11.3254/_p)

### 2.9 WebRTC AEC3 — production reference implementation

* **Title**: WebRTC AEC3 `subtractive_nonlinear` + `residual_echo_estimator`.
* **Mechanism**: dual-mode estimation. **Linear mode**: `R²[ch][k] = S²_linear[ch][k] / erle[ch][k]` — divides linear-stage power estimate by ERLE. **Nonlinear mode**: `R²[ch][k] = X²[k] · echo_path_gain` — multiplies render power by a learned per-bin gain. Reverberation modelled separately and added to residual. Soft noise gate prevents over-suppression.
* **Position**: complete production stack. The `subtractive_nonlinear` block sits before the residual_echo_estimator and produces a non-linear-aware echo estimate; the estimator provides a per-bin power model that drives the suppressor gain.
* **Cost**: known production budget — runs in real-time on mobile.
* **Applicability**: **HIGH for reference** but **CAUTION on porting**. AEC3's `R²[k] = X²[k] · gain` non-linear branch is a *power model*, not a subtractive Volterra; it falls back to it only when linear estimates are unavailable. Per P58 trap (AEC3-pattern RES restructure CLOSED FAIL), copying AEC3's residual structure wholesale was destructive on FS. v3.14 should mine AEC3 for **gain estimation pattern + reverberation model + soft noise gate**, not for `subtractive_nonlinear` mechanism (which AEC3 itself does not document fully publicly).
* **URL**: [chromium.googlesource.com](https://chromium.googlesource.com/external/webrtc/+/refs/heads/main/modules/audio_processing/aec3/residual_echo_estimator.cc)

### 2.10 Speex AEC — open-source NLP module

* **Title**: SpeexDSP echo canceller and NLP (`speex_echo_state_t`).
* **Mechanism**: 2-stage residual echo suppression. Stage 1 produces input to DTD; Stage 2 is the "really nonlinear output". Uses cross-spectral coherence between `ref` and `mic` to estimate per-bin gain. Not a Volterra; closer to a coherence-driven residual gain (also amplitude-mask family).
* **Applicability**: **LOW** — same family as v3.13 closures. Useful as comparison baseline; not a v3.14 mechanism candidate.
* **URL**: [speex.org](https://www.speex.org/)

### 2.11 Functional Link Adaptive Filters (FLAF) — Comminiello & Uncini 2013

* **Title**: "Functional Link Adaptive Filters for Nonlinear Acoustic Echo Cancellation", *IEEE TASLP*; multiple follow-ups (split-FLAF, CC-FLAF, ICC-FLAF, VSSCC-FLAF).
* **Mechanism**: replaces Volterra kernel with a non-linear feature expansion (trigonometric basis of the input, then linear adaptive filter on the expanded features). Hammerstein-like family.
* **Position**: cascade with linear FIR, identical to Stenger-Kellermann 2000 in signal-flow position, but with a different choice of NL basis.
* **Cost**: O(K · M) for K input samples × M expansion functions; comparable to memoryless polynomial if M is small.
* **Applicability**: **MEDIUM** — viable Sprint S5 alternative if classical Volterra proves slow-converging on our cohort. The feature-expansion view is also more amenable to compute-budget tuning.
* **URL**: [ieeexplore.ieee.org](https://ieeexplore.ieee.org/document/6488741/)

### 2.12 Hermite non-linear filters / pipelined HNF — 2024 update

* **Title**: "Nonlinear acoustic echo cancellation based on pipelined Hermite filters", *Signal Processing*, 2024.
* **Mechanism**: Hermite polynomial basis (orthogonal on Gaussian-like inputs) for NL expansion; pipelined cascade of NL stage and linear stage.
* **Applicability**: **MEDIUM** — recent literature; same Hammerstein family but with orthogonal basis (faster convergence than raw polynomial). Worth a Sprint S5 ablation.
* **URL**: [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0165168424000896)

### 2.13 Synthesis — what the literature says

The canon converges on these points:

1. **Hammerstein structure (memoryless or short-memory NL preprocessor → linear FIR)** is the canonical baseline. Stenger-Kellermann 2000 is the load-bearing reference.
2. **Polynomial vs FLAF vs Hermite** is a basis-function choice; first-pass should use polynomial (simplest, most-cited).
3. **Volterra (full kernel)** captures intermodulation memory but at O(K²) cost; **diagonal Volterra** is the practical compromise.
4. **Adaptation algorithm**: NLMS is the slow-but-cheap baseline; RLS is the fast-but-expensive baseline; state-space Kalman per Kuech-Mabande-Enzner 2014 is what we have. Per-bin step-size normalisation (frequency-domain) is essentially mandatory for usable convergence on typical 64-128 ms kernels.
5. **Pipeline position**: the **subtractive** approach (NL preprocessor → linear AEC subtracts NL-aware echo replica) is the canonical fix for real loudspeaker NL. The **residual gain** approach (post-AEC amplitude mask) is the v3.13-closed family.
6. **Validation**: synth NL alone is insufficient for tuning detectors targeted at real NL. A real-NL cohort + listen-validation gate is mandatory (E4 S6a/S6b lesson).
7. **NN flavours exist** but are explicitly out of v3.14 scope per `feedback_no_nn_mention`. The DSP arc is well-developed and not exhausted.

---

## 3. Design decision matrix

We score 5 candidate architectures on 7 axes. Score legend: **3 = excellent, 2 = good, 1 = poor, 0 = unacceptable.**

| Architecture | NL improve potential | Compute cost | Adapt stability | DT robustness | Implementation cmplx | Reuse v3.13 detector | Memory footprint | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A. Memoryless polynomial pre-PBFDKF (Stenger-Kellermann 2000 Hammerstein)** | 2 | 3 | 3 | 3 | 3 | Yes (3) | 3 | **20 / 21** |
| **B. Diagonal 2nd-order Volterra pre-PBFDKF** | 3 | 2 | 2 | 2 | 2 | Yes (3) | 2 | **16 / 21** |
| **C. Full 2nd-order Volterra fused-into-PBFDKF (Kuech 2005)** | 3 | 1 | 1 | 1 | 1 | Yes (3) | 1 | **11 / 21** |
| **D. FLAF / Hermite basis pre-PBFDKF** | 2 | 3 | 3 | 3 | 2 | Yes (3) | 3 | **19 / 21** |
| **E. Spectral residual gain (post-PBFDKF)** | 0 (E4/E5 ruled out) | 3 | 3 | 0 | 3 | Yes (3) | 3 | **15 / 21** but **NL improve potential = 0 → DISQUALIFIED** |

### 3.1 Per-axis scoring justification

**NL improvement potential**:
* B / C have full Volterra structure; theoretically address memory-bearing NL.
* A / D address memoryless / short-memory NL only.
* E is in the amplitude-mask family ruled out by v3.13.

**Compute cost** (vs current PBFDKF baseline):
* PBFDKF baseline at fl=832, hop=160, n_partitions=14: ~10⁵ mul/frame.
* A: +O(N) for polynomial of order N=3-5 → +5-10 mul/sample → ~+10⁴ mul/frame (~10% overhead).
* B: +O(K) per band for diagonal Volterra K=32 → ~+3·10⁴ mul/frame (~30% overhead).
* C: +O(K²) per band → 5-10x baseline.
* D: depends on M expansion functions (M=8 typical) → ~+10⁴ mul/frame (similar to A).
* E: residual mask → negligible.

**Adaptation stability**:
* A / D / E adapt independently of PBFDKF; no state coupling. P52 invariant preserved.
* B requires its own adaptive state (kernel weights + per-bin Kalman gain); state coupling is contained.
* C modifies PBFDKF body (P52 invariant violation; high regression risk per P58 trap).

**DT robustness**:
* A / D adapt only on far-active frames; DT bypass via existing F-DTD gate is straightforward.
* B has additional state that needs DT freeze; manageable.
* C: PBFDKF DT protection is currently load-bearing on cohort tail (P52 invariant); fusing Volterra into PBFDKF risks cohort tail regression.
* E: ruled out by E5 trade-off.

**Implementation complexity** (sprint count estimate):
* A: 6-8 sprints (single polynomial preprocessor, NLMS adaptation, integration with PBFDKF reference path).
* B: 8-10 sprints (kernel design + adaptation tuning + per-bin step-size).
* C: 12-16 sprints (PBFDKF body refactor + state-space Volterra extension; high P52 risk).
* D: 7-9 sprints (FLAF expansion + adaptation; comparable to A).
* E: 4-6 sprints (but NL improvement = 0; do not pursue).

**Reuse v3.13 detector substrate**:
* All architectures: detector (E4.S2 + E5.S3) gates the **adaptation** of the NL stage (freeze when nl_confidence < threshold; adapt when confidence is high). Yes for all.

**Memory footprint** (Audio_ALG production budget; static-memory model in NR/AEC C-port):
* A: +N coefficients + 1 input buffer ≤ 1 KB. Easy.
* B: +K · n_freqs coefficients (K=32, n_freqs=257) ≈ 32 KB. Manageable.
* C: +K² · n_partitions · n_freqs ≈ 1-10 MB. Prohibitive on embedded budget.
* D: +M · K coefficients ≈ 1-10 KB. Easy.
* E: negligible.

### 3.2 Outcome

**A wins the matrix (20/21)**. D is a near-tie (19/21) and remains as a Sprint S5 alternative if A's memoryless polynomial proves insufficient on real-cohort listen verification. B is the Sprint S6 escalation path if A+D both saturate. C is **disqualified** as primary on P52 invariant (PBFDKF body modification) and memory budget. E is disqualified on v3.13 closure mechanism.

---

## 4. Recommended architecture

### 4.1 Headline

**Hammerstein structure**: memoryless polynomial NL preprocessor on the reference path, cascaded with the existing PBFDKF as the linear identifier. The polynomial coefficients adapt jointly with PBFDKF via NLMS, gated on `nl_confidence` from the v3.13 E4.S2 + E5.S3 detector ensemble. Pipeline position: **pre-PBFDKF on the reference**, with both PBFDKF and the polynomial seeing a downsampled-or-windowed adaptive update at every hop.

### 4.2 Block diagram

```
                        ┌─────────────────────────────┐
ref (16 kHz, hop=160) ──┤ HPF + Saturation + DelayEst ├──┐
                        └─────────────────────────────┘  │
                                                         ▼
                                          ┌────────────────────────┐
                                          │ Polynomial NL prepro   │
                                          │ ref_NL = Σ_k h_k · ref^k│  ← adapt h_k
                                          │ (k = 1..3 or 1..5)     │     via NLMS
                                          └──────────┬─────────────┘     gated on
                                                     │                   nl_confidence
                                                     ▼
                                            ┌─────────────────┐
mic (16 kHz, hop=160) ──► HPF ──► ─────────►│ PBFDKF (linear) │── error ──► ResFilter ──► out
                                            │ + ShadowFilter  │             │
                                            │ + PathChangeReg │             │ nl_confidence ──► gate adaptation
                                            └─────────────────┘             │ feed-back to
                                                ▲                           │ polynomial
                                                │                           │
                                                └───────────────────────────┘
                                                       PBFDKF state
                                                       (filter_state, far_active,
                                                        cancel_ratio, sat_level)
                                                              │
                                                              ▼
                                                    ┌──────────────────┐
                                                    │ Detector (E4.S2  │
                                                    │ + E5.S3 ensemble)│
                                                    │ → nl_confidence  │
                                                    └──────────────────┘
```

### 4.3 Pipeline position — pre-PBFDKF on reference (not mic, not post-RES)

Three positions were considered; pre-PBFDKF on **reference** wins:

| Position | Pro | Con | Verdict |
|---|---|---|---|
| **Pre-PBFDKF on REF** | NL preprocessor models speaker → AEC subtracts NL-aware replica → reaches phase / transient | requires joint adaptation with linear filter | **CHOSEN** |
| Pre-PBFDKF on MIC (NL inverse on mic) | conceptually simplest | inverting an unknown loudspeaker NL is ill-posed; can damage NE voice | rejected (NE damage risk) |
| Post-RES on output | minimal coupling with PBFDKF | mathematically reduces to amplitude mask if applied per-bin → v3.13 closure family | rejected (E4 closure) |

Pre-PBFDKF on reference is the canonical Hammerstein form (Stenger-Kellermann 2000) and matches our existing reference processing chain (HPF → Saturation → DelayEst → PBFDKF). The polynomial is inserted between DelayEst and PBFDKF on the reference path only; the mic path is untouched.

### 4.4 Kernel structure

* **Order**: 3rd-order polynomial as first pass: `ref_NL[n] = h₁ · ref[n] + h₂ · ref[n]² + h₃ · ref[n]³`. Three coefficients, one per term. Higher orders (5, 7) reserved as Sprint S5 ablation if 3rd-order proves insufficient.
* **Memoryless**: no delay terms in the polynomial. Loudspeaker memory is captured by the linear FIR downstream; the polynomial captures instantaneous NL only. This matches Stenger-Kellermann 2000 design.
* **Per-band**: NO. Polynomial operates broadband on the time-domain reference. Per-band variant deferred to Sprint S6 if listen-validation reveals frequency-dependent NL signature.
* **Init**: `h₁ = 1.0, h₂ = 0.0, h₃ = 0.0` (pure linear pass-through). Byte-equal flag-OFF behaviour by construction.

### 4.5 Adaptation algorithm

* **Algorithm**: NLMS on polynomial coefficients. Update rule:
  ```
  ref_NL[n] = h₁ · ref[n] + h₂ · ref[n]² + h₃ · ref[n]³
  e[n] = mic[n] − PBFDKF_estimate(ref_NL)
  // gradient w.r.t. h_k is ref[n]^k (held over the update block)
  for k in 1..3:
      h_k ← h_k + μ_NL · ref[n]^k · e_proj[n] / (Σ ref[n]^(2k) + δ)
  ```
  Update operates over the same block as PBFDKF (one update per hop). `e_proj[n]` is the time-domain projection of the frequency-domain error from PBFDKF onto the current frame. Step-size `μ_NL` separate from PBFDKF `mu`; initial value 0.05 (much slower than PBFDKF's 0.3 — polynomial coefficients are global, must converge slowly to avoid divergence).

* **Why NOT RLS**: O(N²) compute is acceptable for N=3 (negligible) but RLS introduces a forgetting factor that interacts badly with PBFDKF's per-bin Kalman R; we want the polynomial to be the slow-stable substrate. NLMS for v3.14 baseline; RLS deferred to Sprint S5 if convergence is too slow.

* **Why NOT Kalman**: per P55 trap (Enzner-Vary canonical Wiener gain failed bench), canonical Kalman on Volterra kernels is not guaranteed to outperform simple NLMS on our cohort. Reserve for Sprint S5 ablation only if NLMS is the bottleneck.

### 4.6 Detector wiring — gate adaptation, not output

The v3.13 detector ensemble (E4.S2 SubtractiveNLP + E5.S3 mic-lpb correlation gate) outputs `nl_confidence ∈ [0, 1]` per hop. v3.14 wires this into the **polynomial adaptation rate**, not the output:

```
μ_NL_effective = μ_NL · nl_confidence    if nl_confidence > 0.3
μ_NL_effective = 0                       otherwise (freeze)
```

* When `nl_confidence` is high → polynomial adapts → captures the NL transfer.
* When `nl_confidence` is low (NE bucket; pure voice; quiet) → polynomial frozen → no false adaptation that would corrupt converged coefficients.
* When `nl_confidence` is moderate (transitional / partial NL) → adaptation proportional → graceful tracking.

This avoids the E5 trade-off because the detector gates the *learning rate*, not the *output gain*. False detector fires cost adaptation step but never produce an audible output artefact (the polynomial has identity init, no clamp-to-zero behaviour).

### 4.7 Stability safeguards

Inherited from PBFDKF + ShadowFilter + PathChangeRegimeHandler invariant (P52). Additional polynomial-specific safeguards:

1. **Coefficient leakage**: `h_k ← (1 − λ) · h_k + update_term` with `λ = 1e-5` per hop. Drains stale coefficients on path change.
2. **Norm clipping**: enforce `‖h‖₂ ≤ 5.0` after update. Prevents catastrophic divergence (polynomial coefficients can blow up under DT-mistaken adaptation). On clip event: log + reset `μ_NL_effective = 0` for next 50 frames (cooldown).
3. **Identity-init regression check**: 800-case bench A/B with `μ_NL = 0` MUST produce byte-equal output to baseline (within `atol=1e-6, rtol=1e-5`). This is the **anti-loophole hard bar** — implementation must preserve byte-equal flag-OFF behaviour.
4. **Cohort-tail catastrophe defence**: per P52 invariant, the regime handler `boost_q + pause_main` MUST continue to fire on cohort tail (qNvSMyU). Polynomial adaptation MUST also freeze when `_main_paused` is True (the regime handler is signalling unstable echo path; polynomial adaptation in this regime is catastrophic).
5. **Saturation gate**: when `_saturation_level > 0.5` (existing F-E5 threshold), freeze polynomial adaptation for the frame. Prevents the polynomial from learning the saturation distortion as the NL transfer.

### 4.8 What this design does NOT do

Explicitly out of scope for v3.14 baseline architecture:

* **Per-band polynomial**: defer until Sprint S6.
* **Higher-than-3rd-order polynomial**: defer to Sprint S5 ablation.
* **Volterra kernel** (memory-bearing NL): defer to Sprint S6 escalation.
* **PBFDKF body modification**: explicitly forbidden by P52 invariant.
* **Output-side suppression** (mask, gate): explicitly forbidden by E4/E5 closure.

---

## 5. Sprint plan

10 sprints, each 1-2 weeks, total 10-20 weeks (≈3-5 months). Compatible with the "6+ months" multi-quarter budget per closure docs.

### Sprint S1 — Cohort sizing + oracle generation (1 week)

**Goal**: lock the listen-validated cohort. Generate per-case oracle outputs (linear-only baseline + ideal NL-cancelled target, where attainable).

* Confirm 5/5 listen-validated NL cohort from E4 (`Gsy0lC5`, `9xjhiFb`, `IrQvqOTC_mvmt`, `WTdBhX`, `m4789f`).
* Add 3 NE control cases + 3 DT control cases (no NL) for false-positive verification.
* Generate Mossi-style synth-NL corpus (tanh / cubic / hard-clip applied to clean ref before linear room IR convolution). 20 synth cases for compute-budget tuning + adaptation algorithm verification.
* Listen materials: rendered baseline (linear-only PBFDKF) for each cohort + control case.
* Acceptance: cohort + oracle artefacts checked into `wav/v3_14_volterra_cohort/` + `listen/v3_14_volterra_listen/`.

### Sprint S2 — Detector integration verification (1 week)

**Goal**: confirm the v3.13 detector ensemble outputs reliable `nl_confidence` on the v3.14 cohort.

* Wire E4.S2 SubtractiveNLP (already in `aec.py`, default OFF) to emit per-hop `nl_confidence` on the cohort.
* Wire E5.S3 mic-lpb correlation gate (currently in worktree branch `worktree-agent-a3c6eebf4ab4289a4`) into `aec.py` as default-OFF detector. Add `e5_correlation_detector_enabled` flag.
* Verify both detectors fire on 5/5 NL cohort, 0/3 NE control, 0/3 DT control.
* Combine into ensemble: `nl_confidence = max(e4_nl_conf, e5_corr_conf)` (initial heuristic; refine in S3).
* Acceptance: detector ensemble output trace per hop committed; per-cohort fire-rate table matches v3.13 verdicts.

### Sprint S3 — Polynomial preprocessor prototype (offline, single case) (2 weeks)

**Goal**: build the Stenger-Kellermann polynomial preprocessor in offline form, tune on one case.

* Implement `class NonlinearPreprocessor` in a research module (`tools/research/v3_14_volterra/preprocessor.py`); not yet in `aec.py`.
* 3rd-order polynomial; NLMS adaptation with `μ_NL = 0.05`; identity init.
* Drive with Case A (Gsy0lC5) reference + mic; iterate `μ_NL` and order to maximise per-frame ERLE on far-active frames.
* Listen verification on Case A: render `out_baseline.wav` (linear-only) and `out_v3_14.wav` (with polynomial preprocessor) and listen-A/B.
* Acceptance: ≥ 1 dB ERLE gain on Case A AND audible NL character improvement (subjective listen).

### Sprint S4 — Multi-case bench (real NL cohort + synth cohort) (2 weeks)

**Goal**: validate polynomial preprocessor on full v3.14 cohort (5 real + 20 synth + 6 controls).

* Render all 31 cases with offline preprocessor.
* Measure ERLE delta vs baseline on each.
* Listen verification on 5 real NL cases + 6 controls.
* Iterate `μ_NL`, polynomial order, leakage λ, cohort-tail-defence freeze conditions.
* Acceptance criteria:
  - 5/5 real NL: ≥ 0.5 dB ERLE gain AND audible improvement (load-bearing per E4.S6 lesson — listen gate first, metric second).
  - 0/6 controls: no audible damage; ERLE delta within ±0.1 dB.
  - Synth corpus: ≥ 3 dB ERLE on 80%+ of synth cases (Mossi-comparable target).
* Failure mode: if listen verification fails on real cohort, escalate to Sprint S5 (alternative basis).

### Sprint S5 — Alternative basis ablation (conditional, 2 weeks)

**Goal**: only if S4 fails real-cohort listen gate. Try alternative NL bases.

* Variant A: 5th-order polynomial (extend to 5 coefficients).
* Variant B: FLAF expansion (8 trigonometric basis functions).
* Variant C: Hermite polynomial basis (orthogonal to Gaussian-like inputs; faster convergence).
* Variant D: short-memory polynomial (3rd-order with 2-sample memory: `ref_NL[n] = poly(ref[n], ref[n-1], ref[n-2])`).
* Per-variant: iterate on Case A first (Sprint S3 protocol), then run S4 protocol on full cohort.
* Acceptance: same as S4. If still failing → escalate to Sprint S5.5 (diagonal Volterra) or close arc.

### Sprint S5.5 — Diagonal 2nd-order Volterra (conditional, 3 weeks)

**Goal**: only if S5 fails. Implement diagonal Volterra kernel.

* Kernel: `y[n] = h₁ · x[n] + Σ_k h₂[k] · x[n] · x[n-k]` (diagonal 2nd-order).
* K = 32 (memoryless cross-products) or K = 64 (longer memory).
* Per-bin step-size normalisation (Kuech-Kellermann 2005 algorithm); compute cost ~3x baseline.
* Same listen + bench protocol as S4.
* Acceptance: same as S4 with allowance for higher compute cost.

### Sprint S6 — Production integration + flag wiring (1 week)

**Goal**: port the winning preprocessor into `aec.py` as default-OFF flag-gated module.

* Add `class NonlinearPreprocessor` to `aec.py` near `class PBFDKF` (around line 1359).
* Wire into `AEC.process()` reference path: `ref` → HPF → Saturation → `polynomial.process(ref)` → DelayEst → PBFDKF.
* Add `AecConfig.volterra_enabled` flag (default `False`); port-control flag.
* Add `AecConfig.volterra_order`, `volterra_mu`, `volterra_leakage` knobs.
* Add detector ensemble wiring: `e5_correlation_detector_enabled` (port from worktree).
* Acceptance: byte-equal output to baseline with flag OFF (within atol=1e-6, rtol=1e-5) on 800-case BALANCED render.

### Sprint S7 — 800-case bench A/B (1 week)

**Goal**: full-cohort statistical verdict.

* 800-case j=4 BALANCED render with `volterra_enabled=True` vs OFF baseline.
* AECMOS scoring; per-bucket Δecho / Δdeg.
* Hard bars (per acceptance criteria §7):
  - FS_static Δecho ≥ +0.05
  - FS_movement Δecho ≥ 0
  - DT_static Δdeg ≥ -0.005
  - DT_movement Δdeg ≥ -0.005
  - NE Δdeg ≥ -0.005
  - Cohort tail (qNvSMyU): Δecho ≥ -0.05
* If hard bars met → proceed to S8. If failed → diagnose + iterate detector wiring / gate threshold.

### Sprint S8 — Listen verification on 800-case worst regressions (1 week)

**Goal**: per E4.S6a/S6b lesson, listen gate is load-bearing. Listen the worst-K regressing cases AND best-K improving cases.

* Top 10 worst Δdeg DT regressions: render baseline + v3.14, A/B listen.
* Top 10 best ΔAECMOS NL improvements: render + listen, confirm subjective improvement.
* Cohort tail (qNvSMyU): listen + measure to confirm cohort-tail catastrophe defence intact.
* Acceptance: 0/10 worst-DT regressions audible; ≥ 7/10 best-improvement cases audibly improved.

### Sprint S9 — C-port (2-3 weeks)

**Goal**: port Python preprocessor to C with byte-equal Python parity.

* Add `c_impl/aec_volterra.c` mirroring Python `NonlinearPreprocessor`.
* Static-memory budget: polynomial coefficients (12 bytes for order=3), update buffer (~1 KB).
* Build with `-ffp-contract=off` per CLAUDE.md.
* Byte-equal verification: 100% of bench cases match Python within `atol=1e-6, rtol=1e-5` per CLAUDE.md anti-loophole.
* Audio_ALG integration: confirm static-memory model in NR/AEC C-port budget; no malloc.

### Sprint S10 — Final productisation (1 week)

**Goal**: BALANCED preset switch + verdict commit.

* Update BALANCED preset to default `volterra_enabled = True` (after S7+S8 acceptance).
* Update `__version__` in `aec.py` to `v3.14`.
* Write `docs/v3_14_volterra_verdict.md` with sprint-by-sprint summary, bench numbers, listen verdicts.
* Commit + tag `v3.14-shipped`.
* Update `docs/CHANGELOG.md` + `docs/SUMMARY.md`.
* Announce closure on `docs/aec_v3_evolution.md`.

---

## 6. Risk register

Hard-won lessons from P3 / P50 / P52 / P55 / P58 / F-series / E4 / E5 must inform v3.14 design. Each risk paired with mitigation.

### R1 — P50 trap: over-protect creates trade-off

* **Source**: P50 Phase 2 `nearend_protect_dt` G3 FS Δecho -1.328 mean / -2.741 worst (8× past audibility bound) DROPPED.
* **Mechanism**: detector that gates suppressor on "we think this is NE" fires on FS frames where echo masquerades as NE → over-protects → kills FS cancellation.
* **v3.14 risk**: detector gating polynomial adaptation could fire spuriously on FS frames → freeze polynomial → no NL learning on the very frames we want to learn from.
* **Mitigation**: gate the *adaptation rate* (proportional to `nl_confidence`), not a binary freeze. Degraded adaptation under uncertainty is recoverable; binary freeze is not. Verify on cohort tail (qNvSMyU) that polynomial adaptation continues at low rate even when detector confidence is partial.

### R2 — P52 trap: cohort tail catastrophe defence

* **Source**: P52 Task A.0 retired ShadowCopyController failed by 1.88% frames > 0.1 dB on cohort tail (qNvSMyU). Path 3 reverted retirement; renamed to PathChangeRegimeHandler; preserved as load-bearing defence.
* **Mechanism**: 0.8% of cohort sits in extreme echo-path-nonstationarity regime (ERL_decile_std p99.2). PBFDKF + Shadow + Regime handler defence is non-trivial; modifying it risks unbounded mis-adaptation.
* **v3.14 risk**: polynomial preprocessor adaptation on cohort tail could be catastrophic. The polynomial coefficients are global (broadband), so a misconverged polynomial damages all frames, not just NL frames.
* **Mitigation**: polynomial adaptation MUST freeze when `PathChangeRegimeHandler._main_paused` is True. Norm clipping (§4.7 safeguard 2) prevents the polynomial from accumulating catastrophic gain even if freeze fails. 800-case S7 hard bar: cohort tail Δecho ≥ -0.05.

### R3 — P55 trap: canonical doesn't always work on bench

* **Source**: P55 Phase 1.2 Enzner-Vary canonical Wiener gain failed bench (DT-FS +7.01 dB vs 20 dB bar). Both canonical discriminators (Jung 2011 + Enzner-Vary) failed.
* **Mechanism**: canonical literature mechanisms are tuned on academic corpora that may not match our cohort statistics. "Canonical" doesn't mean "ships on our bench".
* **v3.14 risk**: Stenger-Kellermann 2000 polynomial may converge cleanly on academic NLAEC corpora but produce neutral or negative deltas on AEC Challenge corpus.
* **Mitigation**: Sprint S3 single-case validation BEFORE multi-case S4 BEFORE 800-case S7. Iterate algorithm parameters at each stage. Reserve Sprint S5 (alternative basis ablation) for the canonical-doesn't-work case. Listen-validation gate (S4 + S8) catches the failure pattern early.

### R4 — P58 trap: AEC3-pattern restructure without evidence-first

* **Source**: P58 Phase 2 closed FAIL — copying AEC3 ErleEstimator + mode-switched ResidualEchoEstimator + Wiener SuppressorGain into our pipeline produced FS Δecho -0.674 / overall geomean 0.96. Mechanism: AEC3 ERLE-divide produces tiny residual when FS-converged → Wiener gain ≈ 1 → no post-suppression; legacy 4-cap chain was load-bearing on FS.
* **v3.14 risk**: temptation to wholesale-port AEC3's `subtractive_nonlinear` block. AEC3's design is co-tuned with its own residual estimator; lifting the subtractive block out of context will fail on our pipeline.
* **Mitigation**: build the polynomial preprocessor from the canonical literature (Stenger-Kellermann 2000), not from AEC3 source-code copy. AEC3 is a reference for **gain estimation pattern + soft noise gate**, not for the subtractive mechanism. Per `feedback_aec_code_review_accuracy`: existing aec.py tuning is load-bearing; isolate each addition via 800-case bench.

### R5 — F2.2 trap: EMA threshold sweep without anchoring

* **Source**: F2.2 v1+v2 closed FAIL — raising threshold from 0.7 to 0.85 reduced fire frequency but each false fire was MORE destructive. Sweep without anchoring the underlying mechanism is destructive.
* **v3.14 risk**: tempted to sweep `μ_NL` / polynomial order / leakage λ as bench-driven hyperparameter optimization without understanding the mechanism. This is the hazy-lynx anti-pattern.
* **Mitigation**: Sprint S3 mechanism understanding (single-case ERLE + listen verification) MUST precede S4-S5 hyperparameter iteration. Document which parameter changes which behaviour BEFORE bench A/B. Per `feedback_no_shortcut_use_canonical`: prefer canonical mechanisms over sweep-and-hope.

### R6 — E4 trap: ERLE-positive but perceptually neutral

* **Source**: E4.S6a/S6b — linear ERLE +0.49 to +1.60 dB across aggression sweep but 0/3 listen-validated cases showed audible NL reduction. Mask was working; just attacking the wrong place.
* **v3.14 risk**: polynomial preprocessor produces ERLE gain on bench but no audible NL improvement on listen materials. The polynomial may be learning the *linear* transfer (which PBFDKF was already capturing) rather than the *non-linear* transfer.
* **Mitigation**: Sprint S4 listen-verification gate is **load-bearing**. ERLE alone is necessary but not sufficient. Per E4.S6 lesson: render listen materials with polynomial active, verify subjective change, THEN run 800-case bench. The cancel_ratio EMA from E4.S2 detector can serve as an "is the polynomial doing something different from PBFDKF" sanity check.

### R7 — E5 trap: filter-protection action creates FS-DT trade-off

* **Source**: E5.S2/S3/S4a/S4b — all four variants on FS-vs-DT trade-off line (slope -0.5 dB DT per +1 dB FS). Same correlation signature on FS-NL frames AND DT high-echo.
* **v3.14 risk**: polynomial preprocessor mis-adapting on DT high-echo frames could damage DT performance. The polynomial sees `ref` only (not mic), so the DT signature on the reference path doesn't directly trigger mis-adaptation. BUT the gradient `e_proj[n]` is computed from PBFDKF error which contains NE voice on DT frames → polynomial could pick up the NE voice as "NL" signal.
* **Mitigation**: gate polynomial adaptation on `filter_state ∈ {refined_usable, coarse_learning}` AND `far_active=True` AND `not _suspicious_dt`. The existing F-DTD gate orthogonalises FS from DT at the gradient level. Verify on Sprint S4 with DT control cases (target: 0/3 audible damage).

### R8 — E5 detector lesson: precise detector + wrong action still fails

* **Source**: E5.S3 had perfect detector (0% NE FP, 14-56% real-NL fire) but the action (filter-protection) was on the wrong axis (amplitude trade-off line).
* **v3.14 risk**: the v3.13 detector substrate is reusable but the v3.14 action (polynomial adaptation) must be the right axis. If polynomial mis-converges even with perfect detector, the architecture is wrong.
* **Mitigation**: pipeline position (pre-PBFDKF on REF) is chosen specifically to escape the post-output amplitude axis. The polynomial is **subtractive** (synthesises NL-aware echo, PBFDKF subtracts) not **multiplicative** (gain on output). This is the architectural escape from the trade-off line.

### R9 — Compute / memory budget overrun

* **Source**: Audio_ALG production budget (NR/AEC C-port; static memory; no malloc; kiss_fft footprint).
* **v3.14 risk**: 3rd-order polynomial preprocessor is +O(N) per sample → ~10% PBFDKF compute overhead. If we escalate to diagonal Volterra (Sprint S5.5) compute jumps to 3x.
* **Mitigation**: Sprint S6 production integration validates memory + compute budget BEFORE bench A/B. If diagonal Volterra is required and exceeds budget, ship the polynomial-only variant with a follow-on v3.15 candidate for embedded compute upgrade.

### R10 — Detector substrate code drift

* **Source**: E5.S3 detector lives in worktree `worktree-agent-a3c6eebf4ab4289a4` (uncommitted to main per E5 closure); could drift from current `aec.py` over time.
* **v3.14 risk**: by the time Sprint S2 starts (~weeks-to-months from now), detector substrate may have diverged from the production pipeline.
* **Mitigation**: Sprint S2 first action is to port E5.S3 detector into `aec.py` as default-OFF (preserve E5 closure intent: research substrate, not production). Once in `aec.py`, the detector tracks production code naturally.

---

## 7. Acceptance criteria — what does "v3.14 ships" mean?

### 7.1 Listen gate (load-bearing per E4.S6 lesson)

* **5/5 listen-validated NL cohort** (Gsy0lC5, 9xjhiFb, IrQvqOTC_mvmt, WTdBhX, m4789f): show **audible improvement** in 爆掉 / 無線電 perceptual character vs baseline.
* **0/6 controls** (3 NE + 3 DT cases): no audible voice damage, no musical noise, no transient artefacts.
* **Top-10 best-improvement bench cases** (post-S7): ≥ 7/10 audibly improved.
* **Top-10 worst-regression bench cases** (post-S7): 0/10 audibly degraded.

### 7.2 800-case bench (BALANCED, fl=832, cng=True, j=4)

Hard bars (FAIL → no ship):

| Bucket | Metric | Bar |
|---|---|---:|
| FS_static | Δecho | ≥ **+0.05** |
| FS_movement | Δecho | ≥ **0.0** |
| DT_static | Δdeg | ≥ **-0.005** |
| DT_movement | Δdeg | ≥ **-0.005** |
| NE | Δdeg | ≥ **-0.005** |
| Cohort tail (qNvSMyU) | Δecho | ≥ **-0.05** |

Soft targets (informational; not blocking):

| Bucket | Metric | Target |
|---|---|---:|
| FS_static | Δecho | ≥ +0.10 |
| FS_movement | Δecho | ≥ +0.05 |
| DT bucket | Δdeg | ≥ 0.0 |
| Overall AECMOS geomean | — | ≥ +0.02 |

### 7.3 C-port byte-equal Python parity (per CLAUDE.md anti-loophole)

* `c_impl/aec_volterra.c` matches Python `NonlinearPreprocessor` within `atol=1e-6, rtol=1e-5` on 100% of frames across 800-case bench.
* `-ffp-contract=off` enforced in CFLAGS.
* `AEC_FP32_WAV=1` (default fp32 PCM output) byte-equal verified.

### 7.4 Memory + compute budget (Audio_ALG production model)

* Polynomial coefficients: ≤ 64 bytes (order-3 polynomial = 12 bytes; with leakage state + counters ≤ 64 bytes total).
* Update buffer: ≤ 4 KB (1 frame of fp32 reference at hop=160 → 640 bytes; with double-buffer ≤ 1.5 KB).
* No malloc; static memory only; compatible with NR/AEC C-port memory model.
* Compute overhead: ≤ +15% vs PBFDKF baseline (measured on Audio_ALG integration build).

### 7.5 Anti-regression on baseline behaviour (byte-equal flag-OFF)

* `volterra_enabled = False`: 800-case BALANCED render byte-equal to v3.13 production within `atol=1e-6, rtol=1e-5` on 100% of frames.
* This is the **anti-loophole** hard bar: implementation must be additive, not modifying any existing pipeline behaviour when flag is OFF.

---

## 8. Deferred items (NOT in v3.14 scope)

### 8.1 Higher-than-3rd-order Volterra (compute prohibitive)

* Order ≥ 5 polynomials: deferred to Sprint S5 ablation only if 3rd-order proves insufficient.
* Order ≥ 7: explicitly out of scope. Compute overhead exceeds budget.

### 8.2 Full 2nd-order Volterra kernel (memory prohibitive)

* Full K² kernel storage at K=64 → 4096 coefficients per band × 257 bands × 14 partitions ≈ 14M coefficients. Prohibitive on embedded budget.
* Diagonal-only variant deferred to Sprint S5.5 (conditional).
* Full Kuech-Kellermann 2005 fused-into-PBFDKF: deferred to v3.15+ after PBFDKF body P52 invariant is re-evaluated.

### 8.3 Neural network NL processor

* Per `feedback_no_nn_mention`: NN is independent arc, not a v3.14 fallback.
* The DSP arc is well-developed and not exhausted; v3.14 commits to canonical DSP path.
* Reference: `project_p50_phase1_6_decision`, `feedback_no_nn_mention`.

### 8.4 Full re-architecture of PBFDKF

* P52 invariant: `PBFDKF + ShadowFilter + PathChangeRegimeHandler` is load-bearing on cohort tail. Modifying PBFDKF body forbidden in v3.14 scope.
* Kuech-Kellermann 2005 fused identifier mode would require this; deferred to v3.15+ design lock.

### 8.5 Output-side NL suppressor

* Per E4 closure: amplitude mask family ruled out for real loudspeaker / codec NL.
* Per E5 closure: filter-protection action on amplitude detector creates FS-DT trade-off.
* No Sprint in v3.14 should re-attempt amplitude / multiplicative mask on output. Use the v3.13 closures as the no-go bar.

### 8.6 Per-band polynomial / per-band NL kernel

* Broadband polynomial in v3.14 baseline. Per-band variant deferred to Sprint S6 escalation if listen-verification reveals frequency-dependent NL signature on cohort.

### 8.7 NLAEC for stereo / multi-channel

* Single-channel only (1 mic + 1 ref) per CLAUDE.md repo scope. Multi-channel NLAEC out of v3.14 scope.

### 8.8 Phase 3 RES architectural refactor (per Phase 3 audit verdict)

* Phase 3 RES audit (S4-S5 closed; S6-S9 deprioritized). Cosmetic cleanup only; no behaviour-change leverage.
* Deferred to v3.15+ as candidate; not v3.14.

### 8.9 F-HFR per-band Q/R + E1 mic_dynamic_margin

* Listed as v3.14 candidates in arc closure but lower priority than Volterra arc.
* Defer to v3.15 unless explicitly authorised by user.

---

## 9. Open questions for user before implementation starts

These need user direction before Sprint S1 kickoff:

1. **Cohort sizing**: 5 listen-validated NL cases is small. Should v3.14 Sprint S1 expand the cohort by listen-auditing additional FS bucket cases with M3 > 7.5 (lower threshold than v3.13's M3 > 9.0)? More cohort = better generalisation but more listen-verification work.

2. **Synth-NL methodology weight**: per E4.S6 lesson, synth-NL is insufficient for tuning. Should Sprint S5 ablations rely on real cohort only (5 cases) or use synth corpus as adaptation-rate / kernel-order tuning surface (with real cohort as final acceptance)?

3. **Cohort-tail authorisation**: cohort tail (qNvSMyU) NEVER receives polynomial adaptation (regime handler `_main_paused` gate). This means qNvSMyU-class cases get baseline behaviour. Acceptable, OR should v3.14 explore allowing partial polynomial adaptation under restrictive conditions (e.g. when both PBFDKF and shadow are stable AND polynomial coefficients are within 50% of init)? Higher risk; reserved for v3.15+.

4. **Sprint S5 fallback**: if S4 fails real-cohort listen gate, escalate to S5 alternative basis (FLAF / Hermite / longer polynomial)? Or close arc with documented amplitude-mask ceiling + Volterra-3rd-order-insufficient finding?

5. **Detector ensemble combination**: `nl_confidence = max(e4_conf, e5_conf)` is heuristic. Should Sprint S2 explore weighted ensemble or boolean AND (more conservative)? Trade-off is fire rate vs precision.

6. **Production candidate branch model**: v3.13 left `feature/v3.11-route-a` as production candidate; v3.14 work could go on a new `feature/v3.14-volterra` branch (file-disjoint with future P52 work) or continue on v3.11-route-a. Recommend new branch per P52 §6.4 file-disjoint model.

---

## 10. Cross-references

### v3.13 closure docs
* [docs/v3_13_arc_closure.md](v3_13_arc_closure.md) — v3.13 final summary
* [docs/v3_13_e4_s6a_s6b_verdict.md](v3_13_e4_s6a_s6b_verdict.md) — amplitude-mask family closure
* [docs/v3_13_e5_closure_verdict.md](v3_13_e5_closure_verdict.md) — filter-protection trade-off closure
* [docs/v3_13_phase3_res_audit_verdict.md](v3_13_phase3_res_audit_verdict.md) — RES architectural surface limited
* [docs/v3_13_e4_s2_design_lock.md](v3_13_e4_s2_design_lock.md) — E4.S2 detector design (substrate)
* [docs/v3_13_e5_s1_verdict.md](v3_13_e5_s1_verdict.md) — E5.S1 audit (acoustic NL ≠ digital clipping)

### Risk register sources
* [docs/p52_a0_postmortem.md](p52_a0_postmortem.md) — cohort-tail catastrophe defence (P52 invariant)
* [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md) — PathChangeRegimeHandler retention
* `project_p55_phase1_2_verdict` — P55 canonical Wiener gain failure (memory)
* `project_p58_phase2_verdict` — P58 AEC3-pattern restructure failure (memory)
* `feedback_aec_code_review_accuracy` — existing aec.py tuning is load-bearing
* `feedback_no_nn_mention` — NN out of v3.14 scope
* `feedback_no_shortcut_use_canonical` — canonical mechanisms over sweep-and-hope
* `feedback_holistic_trace_before_change` — trace before code change

### Production code references
* `python/aec.py` — single-file algorithm reference (~7600 lines)
* `python/aec.py` `class SubtractiveNLP` (line 4624) — E4.S2 detector substrate
* `python/aec.py` `class PBFDKF` (line 1359) — linear identifier (insertion point for polynomial preprocessor on reference path)
* `python/aec.py` `class ResFilter` (line 1738) — residual stage (NOT modified in v3.14)
* `python/aec.py` `class AecConfig` (line 150) — preset / flag definitions
* [CLAUDE.md](../CLAUDE.md) — repo conventions (byte-equal, BALANCED preset, j=4 bench)
* [docs/aec_methods.md](aec_methods.md) — canonical algorithm reference
* [docs/aec_v3_evolution.md](aec_v3_evolution.md) — trace-driven evolution history

### Literature canon
* [Stenger & Kellermann 2000 — memoryless preprocessor](https://www.sciencedirect.com/science/article/abs/pii/S0165168400000852)
* [Stenger & Kellermann 2001 — adaptive Volterra](https://ieeexplore.ieee.org/document/759811)
* [Birkett & Goubran 1995 — TDNN preprocessor](https://www.semanticscholar.org/paper/Acoustic-echo-cancellation-using-NLMS-neural-Birkett-Goubran/f53de6969fd9af7a5502a4f76e7600df2b8d5753)
* [Mossi et al 2010 — assessment of linear AEC under NL](https://ieeexplore.ieee.org/document/5495904/)
* [Kuech & Kellermann 2005 — partitioned-block frequency-domain Volterra](https://ieeexplore.ieee.org/document/1381748/)
* [Malik & Enzner — state-space frequency-domain NLAEC](https://www.semanticscholar.org/paper/State-Space-Frequency-Domain-Adaptive-Filtering-for-Malik-Enzner/5f26898e5c0fdab7a952652225b1f10c2e468b63)
* [Hoshuyama & Sugiyama 2006 — spectral correlation NL suppressor (REJECTED)](https://globals.ieice.org/en_transactions/fundamentals/10.1093/ietfec/e89-a.11.3254/_p)
* [Comminiello & Uncini 2013 — Functional Link Adaptive Filters](https://ieeexplore.ieee.org/document/6488741/)
* [Hermite NL filters 2024](https://www.sciencedirect.com/science/article/abs/pii/S0165168424000896)
* [WebRTC AEC3 residual_echo_estimator.cc](https://chromium.googlesource.com/external/webrtc/+/refs/heads/main/modules/audio_processing/aec3/residual_echo_estimator.cc)
* [Speex AEC](https://www.speex.org/)
* [pyaec — Python reference implementations of Volterra / FLAF / FDKF](https://github.com/ewan-xu/pyaec)

---

## 11. Verdict closure (this document)

This is a DESIGN LOCK draft. **No implementation work begins until user authorises Sprint S1**. The document is committed as `feature/v3.11-route-a` worktree-only artefact (uncommitted) per task spec; promotion to `main` after user review.

**Recommended next step**: user reviews §9 open questions and authorises Sprint S1 cohort sizing.

**Estimated total v3.14 budget**: 10-20 weeks (≈ 3-5 months). Compatible with the 6+ month "canonical breakthrough arc" framing per v3.13 closure.

**Estimated word count**: ~6700 words.

**Status**: DRAFT pending user §9 review.
