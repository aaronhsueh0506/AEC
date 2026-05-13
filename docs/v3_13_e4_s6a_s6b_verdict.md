# v3.13 E4.S6a + S6b verdict — amplitude-mask family closure

**Status**: E4 amplitude-mask family CLOSED. CANNOT SHIP. Suppressor
algorithm retired. Detector retained as research substrate (default OFF).

**Date**: 2026-05-14

## Headline

Listen verification across 4 aggression levels (g_min ∈ {-12, -18, -24,
-30} dB) on 3 listen-validated NL cases (Gsy0lC5 Type 1 / 9xjhiFb Type 2
/ IrQvqOTC_mvmt Type 2) shows **NO AUDIBLE NL REDUCTION** at any level,
including the most aggressive -30 dB where voice formants begin
disappearing. Linear ERLE is metric-positive (+0.49 to +1.60 dB
sweep range) but perceptually neutral on NL character.

Conclusion: harmonic-pinned amplitude mask in frequency domain is
structurally insufficient for real loudspeaker / codec NL. The
perceived NL character (爆掉 / 無線電 / 通話類) is dominantly phase
distortion + time-domain transient artifact, which a multiplicative
spectral mask `m[k,t] · Y[k,t]` cannot reach.

## S6a verdict (g_min = -12 dB, 2026-05-14)

3 listen cases rendered with `apply_suppressor(g_min_e4_db=-12)`.

User listen verdict: **"頻譜看起來 suppressed 有多壓一點，但聽感沒有
太大差異；爆掉或訊號品質差的問題都差不多"**.

Maps to S6a README acceptance matrix "No audible change → tune mask
before 800-case". S7 BLOCKED pending mask redesign.

## S6b verdict (g_min sweep -18 / -24 / -30 dB, 2026-05-14)

Per-case ERLE on signal-active frames:

| Case | -12 dB | -18 dB | -24 dB | -30 dB | -12→-30 gain |
|---|---:|---:|---:|---:|---:|
| A Gsy0lC5 (Type 1) | +1.11 | +1.35 | +1.50 | +1.60 | **+0.49 dB** |
| B 9xjhiFb (Type 2) | +0.67 | +0.84 | +0.96 | +1.04 | +0.37 dB |
| 07 IrQvqOTC mvmt | +0.68 | +0.85 | +0.98 | +1.07 | +0.39 dB |

Aggression sweep produces only **+0.4 dB ERLE gain across 18 dB more
attack** — strong asymptote, indicating harmonic peaks already
attenuated near floor; remaining echo / NL energy is broadband
(off-peak) and unreachable by harmonic-pinned mask.

User listen verdict: **"每一個聽起來 NL 都沒改善"** ＋ **"頻譜上來看
確實有砍比較多，甚至有語音消失，但是聽感的非線性影響太重大了"**.

Three observations from user verdict:
1. ERLE asymptote (data side) → harmonic peaks not where NL lives
2. -30 dB does damage voice spectrum (proves mask IS attacking) → mask
   is working as designed
3. NL perceptual character unchanged across full 18 dB sweep → NL
   character is NOT in attenuated harmonic peak amplitudes

## Mechanism — why amplitude mask cannot fix this NL

Multiplicative spectral mask:
```
Y_masked[k, t] = m[k, t] · Y[k, t]
```
where `m[k, t] ∈ [g_min_lin, 1.0]` only modulates amplitude per bin.

Loudspeaker physical NL (爆掉) and codec NL (無線電) introduce:
1. **Broadband intermodulation products** (off-harmonic) — amplitude
   mask can attack these only via broad mask, which damages voice
2. **Phase distortion** — multiplicative mask cannot change phase
   (mask is real-valued; spectrum gets scaled, not rotated)
3. **Time-domain transients** (爆 attack, click artifacts) — mask
   operates per-frame in frequency; transient localization is below
   FFT analysis window resolution

The amplitude family's intrinsic ceiling: at narrow harmonic pinning
(σ=50 Hz), mask precisely attacks peaks but misses 80%+ of NL energy
distribution. At broad pitched-region (σ=400 Hz or uniform 0-2 kHz),
mask damages voice formants. There is no σ in between where mask is
both perceptually meaningful AND voice-safe.

This is the canonical reason WebRTC AEC3 and similar production
systems use a **time-domain non-linear inverse processor**
(SubtractiveNonlinear), not a frequency-domain amplitude mask, for
real NL distortion.

## Why ERLE-positive but perceptually neutral

Linear ERLE measures broadband energy reduction (mic vs final output).
Mask attenuates harmonics → spectrum has notches → measurable energy
drop. But perception integrates across many cues (envelope shape, phase
coherence, transient character) that mask doesn't touch.

This is the classic failure mode where a metric optimizer ships
something perceptually worthless. The listen gate (S6a / S6b) is
load-bearing — without it, S7 800-case AECMOS A/B would have shown
+ΔAECMOS (or near-neutral), and we'd ship a non-improvement.

Memory implication: amplitude-domain ERLE + isolated bench evidence
is NECESSARY but NOT SUFFICIENT for NLP design validation. Subjective
listen verification on perceptually-validated cohort is the
load-bearing acceptance signal for all future NLP work.

## What's preserved as research substrate

The detector design (E4.S2 - E4.S4.1) is **kept** in aec.py as
flag-gated default-OFF code:
- `SubtractiveNLP` class — autocorrelation pitch tracker
- AecConfig flags: `e4_nlp_enabled` (default False), threshold, gates
- Detector pipeline: `filter_state ∈ {refined_usable, coarse_learning}`
  ∧ `far_active` ∧ `raw_output_rms ≥ 0.05` ∧ `cancel_ratio > 1.05`

S4 800-case validation: NE FP 14.31% → 0.00% after S4.1 cancel_ratio
gate. 5/5 NL cohort fires 19.6 - 44.1%. Detector is reusable.

The detector fires reliably on real NL frames without false-positive on
NE. For any v3.14 NLP work (Volterra inverse, neural NL processor, or
other), the detector can identify which frames need NL processing.

## What's retired

The suppressor algorithm (E4.S5):
- Fixed-attenuation harmonic-pinned mask via `apply_suppressor()` in
  `tools/research/e4_s6_suppressor.py`
- AecConfig fields for suppressor mask parameters (`g_min_e4_db`,
  `time_alpha`, etc.) — if any were added, remove or leave default-OFF
- S5 design lock document `docs/v3_13_e4_s5_suppressor_design.md`
  remains as historical reference

The suppressor is NOT being wired into `aec.py` `AEC.process()`.
S7 (porting) is CANCELLED.

## E4 arc lessons (forward)

1. **Linear ERLE is metric-positive but perceptually insufficient
   for NL**. Future NL design validation must include listen gate
   before metric A/B. The S6a/S6b discipline is a model — render
   listen materials with mask applied, verify subjective change,
   THEN run 800-case AECMOS.

2. **Synth NL methodology is unviable for real-NL-tuned detector**.
   Plan §3.3 S5 synth corpus target (+3 dB) was unmet because tanh /
   cubic / hardclip injection produces too-weak autocorrelation
   signature. Real listen-validated cohort is load-bearing. Keep
   this preserved (5/5 cases on FS bucket M3 > 9.0).

3. **Amplitude mask family is ruled out for real loudspeaker / codec
   NL**. v3.14 NLP arc must use time-domain Volterra non-linear
   inverse or equivalent — not another frequency-domain mask variant.
   Do not iterate further on σ / aggression / mask shape.

4. **Detector design is reusable**. Pitch tracker + cancel_ratio
   gate is a useful NL frame identifier independent of suppressor
   choice. Retain as v3.14 substrate.

## v3.13 arc state after closure

- E2 Delay: **SHIPPED** (Path 3 pre-align max=1024 ms)
- E4 NLP: **CLOSED CANNOT SHIP** (amplitude family ruled out;
  Volterra v3.14)
- E5 Saturation deepening: ACTIVE (S1 audit shipped; S2-S4 pending)
- F-HFR per-band Q/R: v3.14 candidate
- E1 mic_dynamic_margin: v3.14 candidate

## v3.14 — Volterra design lock placeholder

When v3.14 opens, the E4 successor arc design lock should cover:
- Volterra kernel order (2nd order vs 3rd order)
- Adaptation algorithm (LMS / NLMS / RLS on Volterra coefficients)
- Detector reuse (gate Volterra adaptation on `nl_confidence > 0`)
- Position in pipeline (pre-PBFDKF on mic? Post-RES? Mic + ref both
  Volterra-inverted?)
- Compute budget (Volterra is O(K²) for kernel length K — significant
  vs current PBFDKF)

This is out of v3.13 scope.

## Artifacts

- Suppressor: [tools/research/e4_s6_suppressor.py](../tools/research/e4_s6_suppressor.py) (retained for historical reference, not wired)
- Isolated bench: [tools/research/e4_s6_isolated_bench.py](../tools/research/e4_s6_isolated_bench.py)
- S6b sweep harness: [tools/research/e4_s6b_aggression_sweep.py](../tools/research/e4_s6b_aggression_sweep.py)
- Listen materials: `listen/v3_13_e4_s6_listen/` (3 cases × 5 versions each)
- S6 isolated bench verdict (real cohort): [docs/v3_13_e4_s6_verdict.md](v3_13_e4_s6_verdict.md)
- S5 design lock: [docs/v3_13_e4_s5_suppressor_design.md](v3_13_e4_s5_suppressor_design.md)
- Detector design lock: [docs/v3_13_e4_s2_design_lock.md](v3_13_e4_s2_design_lock.md)

## Verification rules followed

1. 5-case listen-validated cohort as primary acceptance
2. Subjective listen verification gate before 800-case A/B
3. Sweep over aggression to disambiguate H1 (low) vs H2 (shape)
4. Mechanism analysis traced to fundamental physics
   (amplitude mask + phase distortion incompatibility)
5. Detector substrate retention (no destructive code removal)
