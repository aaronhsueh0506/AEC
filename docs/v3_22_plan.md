# v3.22 Plan — Beyond-AEC3 Optimization (DSP-only)

> **Status: ROADMAP — not active.** Gated on v3.21 CLOSE. Drafted 2026-05-29.
> v3.21 = AEC3 *alignment* (closing now). v3.22 = beyond-AEC3 *optimization* only —
> everything the v3.21 audit found "aligned to AEC3 but suboptimal for *our* pipeline"
> lands here. No code is written under this plan until v3.21 is closed and the user
> authorises a v3.22 kickoff.

---

## 0. Scope & framing

- **Baseline** = v3.21-closed (includes the corrected `active_render` etc., pending the
  Tier-C validation verdict).
- **Hard constraints (unchanged)**: hop/block/fft = 160/320/512; mic-path HPF=ON /
  ref HPF=OFF (locked); **DSP-only**.
- **Discipline**: matched-magnitude AECMOS Pareto (less echo auto-lifts deg ≠ a win).
  Reference points are **absolute: beat AEC2 + beat AEC3 deg** — not yesterday's baseline.
- **Dev discipline**: every candidate behind a default-OFF flag (byte-equal preserved)
  until an isolated 800-case A/B passes; never batch.

---

## 1. The movable-gap map (from the 11.48 dB decomposition + v3.21 findings)

**Architecture note (verified `filters.py:88-90`)**: `block_size = 2×hop = 320` is the
**real, energy-relevant frame**; `fft_size = 512` is only the next-pow2 *zero-pad* of 320.
Per-bin |X|² energy therefore scales with **2×hop = 320 points, NOT fft_size = 512** — the
zero-pad adds bins (finer resolution) but no energy (white-noise demo: per-bin ∝ frame
samples, fft-independent).

| Gap | Size | v3.22 addressable? |
|---|---|---|
| Bin **resolution** (257 vs 65 bins = 31 vs 125 Hz) | — | **N/A** — ours is *finer*, not a deficit. |
| Per-bin **energy magnitude** (frame 2×hop=320 vs AEC3 64 → 5× ≈ **7 dB**) | most of the "6–8 dB" | **Accountable** — a frame-length units artifact in the *diagnostic*, normalisable / floor-scalable (Arc 3); NOT an irreducible ceiling. |
| **Shadow convergence quality** | ~2–3 dB | **YES → Arc 1** |
| **Painted-black perceptual** (HF wipe / formant valley / fricative) | perceptual | **YES** — AEC3-inherent, Speex-free → **Arc 2** |

**Reframe (2026-05-29 architecture note)**: the "6–8 dB FFT-resolution ceiling" in the
11.48 dB decomposition was **largely the 2×hop=320 vs 64-sample frame-energy units** (5× ≈
7 dB) of comparing our per-bin |echo_spec|² against AEC3's — an apples-to-oranges units
artifact in the *diagnostic*, not a real performance gap (AECMOS scores the time-domain
output, which has no such units issue). Only the estimation *method* (partition-conv vs
windowed subtraction) is genuinely structural. So the movable budget is **larger** than
first framed. The two thrusts remain **shadow quality** (score gap) and **painted-black /
non-linear** (the perceptual gap the user has repeatedly flagged).

---

## 2. Tier-1 arcs (highest value)

### Arc 1 — Shadow Convergence Quality

- **Problem (evidence)**: across all of v3.21, the AEC3 coarse-fallback (URO) and
  poor-coarse rescue mechanisms closed NO-SHIP, and the common root was
  `coarse_conv ≈ 0%` — the shadow PBFDAF/NLMS never catches up to the refined Kalman
  when the refined filter is genuinely better (the 9xjhi FS_static "Category-3"
  shadow-quality gap; the rescue Pareto failures in `docs/v3_21_poor_coarse_rescue_*`).
  AEC3's rescue/fallback design *assumes* a usable coarse — ours doesn't have one.
- **Beyond-AEC3 directions**:
  1. Variable step-size / regularisation redesign of the shadow update so it converges
     where the refined filter does.
  2. **Differential-Huber "orthogonal shadow"** (the Arc H' idea — see
     `project_shadow_orthogonality_arc_h_prime`): make the shadow track an
     *independent* error landscape rather than a degraded copy of main, so it carries
     information main lacks.
  3. Kalman shadow with a distinct Q (re-examined beyond AEC3's homogeneous coarse).
- **Payoff**: closes the FS_static score gap; makes URO / rescue actually net-positive
  (they were starved by the bad coarse). Reaches into the ~2–3 dB movable budget.
- **Effort**: medium-large (filter redesign + its own validation arc).
- **Dependency**: settles the shadow design that Arc 6 (Python↔C reconcile) consumes.

### Arc 2 — Non-linear / Painted-black (Speex-direction)

User-confirmed: AEC3's own spectrum shows the painted-black artefact; **Speex does
not** → the fix is a Speex-style, beyond-AEC3 approach. Four sub-paths, sequenced
stopgap → deep:

- **B0 — Speex study (diagnostic-first)**: determine *what* Speex's preprocessor /
  residual suppression does differently that avoids painted-black; port the principle.
- **B1 — HF min-gain floor during DNE**: protective amplitude floor when the
  dominant-nearend detector fires. **Substrate already in code**
  (`hf_min_gain_floor_during_dne_enabled`, default-OFF). Weeks-level A/B.
- **B2 — LF filter-failure R² injection**: the 200–500 Hz ref-bleed "two-pitch / 重音".
  **Substrate already in code** (`enable_lf_filter_failure_r2_injection`, default-OFF).
- **B3 — Volterra non-linear inverse filter**: the canonical deep fix. The artefact's
  core is phase distortion + a time-domain transient that a multiplicative spectral
  mask structurally cannot reach (confirmed in the v3.13 E4/E5 amplitude-mask closure).
  Major multi-month arc; highest eventual value, biggest effort.
- **Sequence**: B1/B2 as quick stopgap layers + B0 study in parallel → B3 as the real
  fix once B0 informs the method.

---

## 3. Tier-2 arcs (grown directly from the 2026-05-29 conversion audit)

### Arc 3 — Signal-adaptive per-bin floor
- **Why**: per-bin |X|² energy scales with the real frame = **2×hop = 320** (not fft_size).
  So the floor factor vs AEC3's 64-sample frame is **(2×hop)/64 = 5×** broadband,
  **((2×hop)/64)² = 25×** tonal — signal-dependent; no single constant is right. (The
  shipped `fft_density` uses `(fft_size//2)/64 = 256/64 = 4×` — wrong basis: it counts
  zero-padded *bins*, not real-frame energy. Fix the derivation to the 2×hop energy basis.)
  A tonal-vs-broadband adaptive floor would protect exactly the weak HF bins (valleys /
  fricatives) that get painted black. Beyond-AEC3 (AEC3 uses a fixed floor).
- **Payoff**: directly targets the painted-black valley/fricative wipe — complements Arc 2.

### Arc 4 — Pipeline-tuned thresholds
- **Why**: if the Tier-C validation shows the *AEC3-true* values (`active_render`
  9.31e-6, `dne` 12, etc.) are **worse** for our hop/fft than the shipped tuning, then
  our pipeline's optimum genuinely differs from AEC3 → tune for *our* pipeline. Pure
  v3.22 tuning (the honest "we diverge because our FFT/hop differs" case).
- **Dependency**: **gated on the Tier-C validation result** (task running now).

### Arc 5 — Per-estimator EMA invariant
- **Why**: C1-C3 proved ERLE wants "preserve count" (don't convert α); but envelope
  trackers (CNG Y²/N², reverb) may legitimately want "preserve seconds". Choose the
  invariant *per estimator* empirically rather than one-size-fits-all. Includes
  validating/reverting the CNG converted EMAs (currently unconditional, never A/B'd).

---

## 4. Tier-3 — Integration

### Arc 6 — Python↔C shadow reconcile
- **Why**: Python shadow = PBFDAF/NLMS (no Q); C shadow = PBFDKF/Kalman with Q
  (`c_impl/src/aec.c`). Real divergence — "bit-equal Python↔C" can't hold for the shadow.
- **When**: *after* Arc 1 fixes the shadow design — reconcile C to the **final** shadow,
  not the current one (else we'd reconcile twice).

---

## 5. Sequencing

1. *(gated on v3.21 CLOSE)* establish the v3.22 baseline + kickoff authorisation.
2. **Quick stopgap**: B1 / B2 (stubs exist, weeks) + Arc 4 (if Tier-C flags it).
3. **Tier-1 deep**: Arc 1 (shadow quality) ∥ B0 (Speex study).
4. **Major arc**: B3 (Volterra) — after B0 defines the method.
5. **Opportunistic**: Arc 3, Arc 5 woven in.
6. **Integration**: Arc 6 once the shadow design is final.

---

## 6. Constraints (permanent)

- hop/block/fft = 160/320/512 locked; HPF locked; **DSP-only**.
- Matched-magnitude AECMOS Pareto; absolute target = beat AEC2 + beat AEC3 deg.
- Byte-equal dev discipline; isolate every change via 800-case; never batch.
- No version numbers in variable/config names — use mechanism descriptors.

---

## 7. Substrate already in place (default-OFF, ready for v3.22)

- `hf_min_gain_floor_during_dne_enabled` + `hf_min_gain_floor_during_dne_db` (Arc 2 / B1).
- `enable_lf_filter_failure_r2_injection` + its band/ratio/inject params (Arc 2 / B2).
- E4.S2 + E5.S3 non-linear detector substrate (reusable for B0/B3).
- The 2026-05-29 audit's corrected constants + comments (active_render 9.31e-6,
  fft_density frame-basis) inform Arc 3 / Arc 4.

---

## 8. Open decisions (resolve at v3.22 kickoff — user calls)

1. **First thrust**: shadow quality (score gap) vs painted-black (perceptual)?
2. **Volterra (B3)**: kick off directly, or B1/B2 stopgap first then B3?
3. Any arc missing from this map?

---

*Roadmap only. No code, no merge, no version bump under this plan until v3.21 is
closed and a v3.22 kickoff is authorised.*
