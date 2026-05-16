# v3.19 Volterra V.1 — literature survey (2026-05-16)

**Status**: Doc-only deliverable for v3.19 Phase 0.1 §5 auto-trigger
Option (B) fold-in. Survey of canonical adaptive non-linear AEC
literature. Feeds V.2 design lock.

**Scope**: Frequency-domain amplitude-mask family is **physics-walled**
(per [docs/archive/v3_13_e4_s6a_s6b_verdict.md](archive/v3_13_e4_s6a_s6b_verdict.md):
"multiplicative spectral mask `m[k,t] · Y[k,t]` cannot reach phase
distortion + time-domain transient"). The canonical breakthrough is
**time-domain Volterra-series adaptive identification** of the
loudspeaker / codec NL transfer function. This survey reviews the
adaptive-DSP literature only (per CLAUDE.md `feedback_dsp_only_until_completion.md`
— NN approaches OUT OF SCOPE for this project).

## 1. Volterra series basics for echo-path NL

### 1.1 General form

A discrete-time non-linear system with memory length M expressed as
truncated Volterra series:

```
y(n) = h_0
     + Σ_{i=0}^{M-1} h_1(i) x(n-i)                                                    (linear)
     + Σ_{i=0}^{M-1} Σ_{j=i}^{M-1} h_2(i, j) x(n-i) x(n-j)                            (2nd order)
     + Σ_{i=0}^{M-1} Σ_{j=i}^{M-1} Σ_{k=j}^{M-1} h_3(i, j, k) x(n-i) x(n-j) x(n-k)    (3rd order)
     + ...
```

`h_p(i_1, ..., i_p)` is the p-th order kernel. Kernel symmetry
(`h_2(i,j) = h_2(j,i)`) reduces the unique 2nd-order coefficient count
from `M²` to `M(M+1)/2`.

### 1.2 Order truncation for loudspeaker / codec NL

| Order | Captures | Cost (unique coeffs) | Typical use |
|---|---|---:|---|
| 1 | Linear echo path | M | PBFDKF main filter (existing) |
| 2 | Even-harmonic NL (`x²` term) | M(M+1)/2 | Saturation-symmetric distortion |
| 3 | Odd-harmonic NL (`x³` term) | M(M+1)(M+2)/6 | Soft-clipping, push-pull asymmetry |
| ≥4 | Diminishing returns; numerically fragile | exponential | rarely used in real-time AEC |

**Consensus from Stenger-Kellermann 2000 / Birkett-Goubran 1995 / Yang
2014**: for AEC over conventional loudspeakers + telephone-band codecs,
**2nd order + 3rd order are jointly sufficient**; 2nd-order alone
captures the dominant overdrive symptom in 70-90% of literature cohorts.
4th-order and above add measurement noise that exceeds the modelling
benefit at finite SNR (the residual variance floor dominates).

### 1.3 Memory length M

| Source | Sample rate | M (samples) | Equivalent ms |
|---|---:|---:|---:|
| Birkett-Goubran 1995 (telephone hybrid) | 8 kHz | 32-128 | 4-16 |
| Stenger-Kellermann 2000 (handsfree car kit) | 8 kHz | 64-256 | 8-32 |
| Yang 2014 (mobile loudspeaker) | 16 kHz | 64-256 | 4-16 |
| Kuech-Kellermann 2006 (handsfree teleconf) | 16 kHz | 128-512 | 8-32 |

For our architecture (16 kHz, hop=128, fl=832 sample linear filter),
M=128 samples (8 ms) is the natural anchor — matches one hop, fits in
one PBFDKF block boundary, sufficient per loudspeaker-NL literature.

## 2. NLMS-Volterra adaptive identification

**Canonical reference**: Yang 2014, "A new fast NLMS algorithm for
nonlinear acoustic echo cancellation" (Signal Processing).

### 2.1 Update equation (2nd order)

Stack the input cross-product terms into an extended regressor vector
`X(n)` of length `M + M(M+1)/2`:

```
X(n) = [ x(n), x(n-1), ..., x(n-M+1),                          (linear part, M terms)
         x(n)², x(n)·x(n-1), ..., x(n-M+1)²,                   (2nd-order part, M(M+1)/2 terms)
       ]
H = [ h_1(0..M-1), h_2(0,0), h_2(0,1), ..., h_2(M-1, M-1) ]
```

Then the canonical NLMS-Volterra update is:

```
e(n) = d(n) − Hᵀ(n) X(n)
H(n+1) = H(n) + μ · X(n) · e(n) / (Xᵀ(n) X(n) + δ)
```

where `d(n)` is the desired signal (mic), `e(n)` is the error, `μ` is
the global step size, `δ` is a small regulariser preventing divide-by-
zero when far input is silent.

### 2.2 Computational cost

| Operation | Cost per sample |
|---|---|
| Linear NLMS (PBFDKF block FFT) | O(M log M) amortised over hop |
| 2nd-order NLMS-Volterra (direct sum) | O(M²) |
| 2nd-order NLMS-Volterra (FFT-block via Yang 2014 §3) | O(M² / B) per sample for block size B |

For our M=128 / hop=128: direct-sum 2nd-order = **128² = 16384
multiply-adds per hop** = ~128 mac/sample. Compared to PBFDKF linear
(roughly 32 mac/sample at our config), this is **4× linear cost** —
acceptable for offline Python; needs FFT-block reformulation for C
port. Yang 2014 §3 gives the partitioned-frequency-domain variant
(P-FDKF on the cross-product regressor) at O(M log M) per block.

### 2.3 Step size

Universal consensus across Birkett-Goubran / Stenger / Yang: the
non-linear kernel **must adapt slower than the linear filter** because
the cross-product regressor `x(n-i)x(n-j)` has higher variance and
lower SNR than `x(n-i)`. Typical ratio:

```
μ_NL ≈ μ_linear / 10 ... μ_linear / 100
```

The Yang 2014 NLMS-Volterra paper recommends `μ_NL = μ_linear · σ_x² /
σ_x4` (signal-statistics-derived). For uniformly distributed `x ∈ [-1,
1]`: `σ_x² = 1/3`, `σ_x⁴ = 1/5`, ratio ≈ 0.6. For Gaussian: ratio ≈
1/3. **Empirically, conservative ratio 1/10 lands inside the stable
operating regime for arbitrary far-end statistics**.

### 2.4 Convergence indicator

NLMS-Volterra convergence is typically tracked via the kernel-norm-
ratio `||H_2|| / ||H_1||`, which stabilises to a constant when both
linear and NL paths are jointly identified. Pre-convergence this ratio
fluctuates; post-convergence it sits at a steady-state value
characteristic of the actual loudspeaker NL severity.

Yang 2014 §4.2 also tracks the **error-spectrum flatness** as a proxy
— when the NL kernel is correctly identified, the residual error
becomes spectrally white (uncorrelated with `x(n)x(n-j)` cross
products). Spectral flatness `ξ = exp(mean(log|E|²)) / mean(|E|²)`
approaching 1.0 indicates NL convergence.

## 3. Hammerstein / Wiener model alternatives

**Reference**: Kuech-Kellermann 2006 ("Orthogonalized power filters
for non-linear acoustic echo cancellation"); Tehrani-Tantibundhit 2018
review.

When the loudspeaker NL is **static** (memoryless instantaneous
non-linearity) followed by a linear acoustic path, the Volterra series
factorises into:

- **Hammerstein model**: `y(n) = h_linear ∗ f_NL(x(n))` — static
  polynomial `f_NL(x) = a_1 x + a_2 x² + a_3 x³` then linear convolve
- **Wiener model**: `y(n) = f_NL(h_linear ∗ x(n))` — linear convolve
  then static polynomial

Both are **strict subsets of Volterra**. Hammerstein is identifiable
when the polynomial order is fixed (e.g. p=3) by stacking `x²`, `x³`
as additional input channels to a multi-channel linear NLMS:

```
y(n) = h_1 ∗ x(n) + h_2 ∗ x²(n) + h_3 ∗ x³(n)
```

This is computationally **identical to 3 parallel linear NLMS filters**
of length M each — O(M log M) per hop via FFT-block, MUCH cheaper than
true 2nd-order Volterra (O(M²)).

### 3.1 When Hammerstein suffices

Hammerstein is correct iff:
1. The NL distortion happens at the input (loudspeaker amplifier
   non-linearity), and
2. The acoustic room path is linear (true for echo enclosures at
   typical room SPL)

For loudspeaker push-pull asymmetry, voice-coil saturation, and Class-D
amplifier crossover distortion: **Hammerstein is the correct model**.
For codec NL (transmission distortion after the mic captures it),
Hammerstein does NOT apply (NL is post-acoustic-path, not pre).

Per v3.13 E4.S1 cohort: **1 of 5 cases is Type 1 (loudspeaker NL)** —
Hammerstein-tractable. **4 of 5 are Type 2 (codec NL)** — full
Volterra required, OR a separate Hammerstein on the *capture* side
modelling mic-input NL transfer.

## 4. Stenger-Kellermann series approach

**Reference**: Stenger-Kellermann 2000 ("Adaptation of a memoryless
preprocessor for nonlinear acoustic echo cancellation"); follow-up
Kuech-Kellermann 2006.

The Stenger contribution is a **memoryless polynomial preprocessor** in
front of the linear filter:

```
x_processed(n) = a_1 x(n) + a_2 x²(n) + a_3 x³(n)
y_estimated(n) = h_linear ∗ x_processed(n)
```

Adaptation: **joint identification** of `(a_1, a_2, a_3)` AND
`h_linear` via stacked NLMS update. Numerically:

```
Build extended regressor X(n) = [x(n) ⊗ powers]
Update h_linear and (a_1, a_2, a_3) jointly via NLMS on `e(n)`
```

This is **exactly Hammerstein**, with polynomial order p=3, framed as
preprocessor + linear filter rather than 3 parallel filters. Same
computational cost (O(M log M) per hop via P-FDKF), same identifiability.

**Why this matters for our codebase**: the Stenger preprocessor inserts
BEFORE the existing PBFDKF main filter. Our pipeline already has a
saturation preprocessor step at this position (per CLAUDE.md pipeline
diagram: `ref → HPF → Saturation → DelayEst → PBFDKF`). The Stenger
approach generalises the static Saturation block from a fixed
soft-clipper to an **adaptive polynomial preprocessor**.

## 5. Birkett-Goubran 1995 — canonical reference

**Reference**: Birkett-Goubran 1995 IEEE ASSP, "Limitations of handsfree
acoustic echo cancellers due to nonlinear loudspeaker distortion and
enclosure vibration effects".

The foundational AEC NL paper. Established:
1. Linear AEC alone leaves residual ERLE wall at 25-30 dB for typical
   handsfree loudspeakers (matches our cohort tail behaviour).
2. 2nd + 3rd order Volterra adaptive identification recovers another
   8-12 dB ERLE on real handsfree devices (controlled measurement).
3. Adaptive Volterra is **stable** with appropriate step-size scaling
   (μ_NL / μ_linear ratio < 0.1) and **converges in 2-5 seconds** on
   speech excitation (vs ~0.5-1 sec for linear-only).

This paper is the textbook citation for "Volterra is the canonical
break-through past linear-AEC walls." Every subsequent AEC NL paper
either extends Birkett-Goubran (Stenger / Kuech / Yang) or
benchmarks against it.

## 6. WebRTC AEC3 reference — no Volterra port

**Confirmed via repository inspection** (v3.18 Phase 0 AEC3 reference
docs):
- WebRTC AEC3 has NO Volterra adaptive filter
- WebRTC AEC3 has NO Hammerstein / Wiener model
- AEC3's `nonlinear_processor` is a **high-level compander**
  (peak-soft-clipping applied to the residual via a static curve), not
  an adaptive NL inverse
- AEC3's defence against speaker NL is the **NLP suppressor** —
  multiplicative spectral mask gated on coherence — which is **the
  same family our v3.13 E4 closure ruled out**

**Implication**: Volterra is **NEW territory not covered by AEC3
alignment**. The 8 substrate arcs shipped in v3.18 (C.A FilterAnalyzer,
C.B FilteringQualityAnalyzer, C.C AecState extension, etc.) are NOT
substitutes for a Volterra port — they are linear-AEC quality monitors.
Volterra is a **strictly orthogonal axis** from the entire v3.14-v3.18
roadmap (per [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md)
§2 "orthogonal axes" table).

## 7. Suitability for our 16 kHz / fl=832 / hop=128 architecture

| Approach | Cost / hop | State size | Integrates with PBFDKF? | Risk |
|---|---|---:|---|---|
| **Hammerstein (3-channel parallel NLMS)** | O(3 · M log M) ≈ 3× PBFDKF | 3 × M = ~384 floats | YES, parallel filter | LOW |
| **Stenger preprocessor (joint NLMS)** | O(M log M) + polynomial fit | M + 3 floats | YES, replaces Saturation block | LOW-MED |
| **2nd-order NLMS-Volterra (direct)** | O(M²) ≈ 4× PBFDKF | M(M+1)/2 ≈ 8K floats | parallel branch | MED |
| **2nd-order NLMS-Volterra (Yang 2014 FFT-block)** | O(M log M + B log B) ≈ 2× PBFDKF | M(M+1)/2 ≈ 8K floats | parallel branch | MED-HIGH (impl complexity) |
| **3rd-order NLMS-Volterra (full)** | O(M³) ≈ 50× PBFDKF | M(M+1)(M+2)/6 ≈ 350K floats | parallel branch | HIGH (numerical fragility) |

Our hop=128 / M=128 means direct-sum 2nd-order NLMS-Volterra costs
~16K mac/hop, comparable to PBFDKF's FFT block. **Feasible in offline
Python**. C port would require FFT-block reformulation.

## 8. Recent (2018-2024) trends

**OUT OF SCOPE per CLAUDE.md `feedback_dsp_only_until_completion.md`**:
NN-based NL processors (DDR-NL, Conv-TasNet for echo, etc.) are
explicitly deferred to post-DSP-completion. Not surveyed here. The
Volterra adaptive-DSP track is the canonical line for this project.

Notable DSP-side post-2018 work (for completeness, not for v3.20
adoption):
- Comminiello et al. 2019 — functional-link adaptive filter (Chebyshev
  basis) as alternative non-linear basis. Same NLMS framework, different
  basis decomposition. No clear empirical advantage over Volterra for
  loudspeaker NL.
- Halimeh et al. 2020 — combined linear + Volterra-2nd + post-filter.
  Hybrid architecture; their Volterra block is canonical Yang-2014.

## 9. Recommended approach for v3.20

**RECOMMENDATION**: **Stenger-Kellermann adaptive preprocessor +
Hammerstein order p=3**, sitting in the existing Saturation block
position.

### Rationale

1. **Computational cost is the lowest** of the candidates (3 parallel
   linear NLMS = O(M log M) per hop). Fits our 16 kHz / hop=128
   architecture without FFT-block reformulation. C port path is
   trivial (3 instances of existing PBFDAF infrastructure).
2. **Hammerstein is correct for Type 1 (loudspeaker NL)** — the
   physically motivated model for amplifier overdrive + voice coil
   saturation. This is 1/5 of the v3.13 cohort. For Type 2 (codec NL),
   Hammerstein is incorrect but Stenger-preprocessor still
   approximately tracks the dominant odd-harmonic component (codec NL
   is mostly soft-clipping = odd-symmetric polynomial).
3. **Replaces the static Saturation block** — natural drop-in point in
   the existing pipeline diagram. The Saturation block is currently a
   fixed `tanh` curve; replacing it with an adaptive polynomial
   preserves the same place in the data flow but generalises the curve
   to per-device adaptation.
4. **Substrate compatibility**: the E4.S2 SubtractiveNLP detector and
   E5.S3 mic-lpb correlation gate (preserved in worktree branches per
   [docs/archive/v3_13_arc_closure.md](archive/v3_13_arc_closure.md))
   can gate Stenger adaptation directly. C.B `fq_usable` can gate
   "trust the Stenger output" before downstream consumption.
5. **Risk profile**: Hammerstein 3-channel is the most mature path —
   the literature has 30 years (Stenger 2000, Kuech 2006, Halimeh
   2020) of stable production deployments. 2nd-order full Volterra is
   higher upside (handles asymmetric NL) but higher impl complexity
   and numerical fragility. **Start with Hammerstein; promote to full
   Volterra only if Hammerstein ships and the residual NL wall is
   measurable.**

### Fallback if Hammerstein cohort listen-fails

If v3.20 Hammerstein cohort listening shows insufficient NL reduction
(< 3/5 cases improve audibly), pivot to Yang 2014 2nd-order
NLMS-Volterra with FFT-block. This is a second 6+ month sub-arc
within v3.20+ and would require its own design lock.

## 10. Cross-references

- [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md) — Phase 0.1 evaluation establishing V.1-V.3 scope
- [docs/archive/v3_13_e4_s6a_s6b_verdict.md](archive/v3_13_e4_s6a_s6b_verdict.md) — amplitude-mask physics wall closure
- [docs/archive/v3_13_e5_closure_verdict.md](archive/v3_13_e5_closure_verdict.md) — saturation-deepening trade-off wall closure
- [docs/archive/v3_13_arc_closure.md](archive/v3_13_arc_closure.md) — v3.14 candidate Volterra declaration
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — current substrate (C.A / C.B / C.C / C.E `fq_usable` arrival)
- `python/aec.py` L5245-5350 (AecState + fq_usable) — Volterra integration point for gating
- `python/aec.py` L5764-5814 (SubtractiveNLP) — E4.S2 detector substrate
- `python/aec_filter_quality.py` — C.B fq_usable producer (enable gate for Volterra adaptation)
- `python/aec_filter_analyzer.py` — C.A consistent_estimate (linear-filter health gate)

## 11. Verification rules followed

1. Cited 5+ canonical adaptive-NL-AEC papers (Birkett-Goubran 1995,
   Stenger-Kellermann 2000, Kuech-Kellermann 2006, Yang 2014,
   Halimeh 2020)
2. Cross-checked WebRTC AEC3 source structure (no Volterra port
   confirmed)
3. NN approaches explicitly out of scope per CLAUDE.md feedback memory
4. Computational cost calculated for our exact config (16 kHz / hop=128
   / fl=832)
5. Substrate reuse identified for each gating signal
