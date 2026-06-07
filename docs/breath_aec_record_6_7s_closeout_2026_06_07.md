# aec_record 6–7s breath/fricative — CLOSEOUT (research record)
Date 2026-06-07 · branch `feature/render-correlation` · production stays **v3.22.4 byte-equal**

## Verdict (one line)
`aec_record` 6–7s is **not a single HF breath bug** — it is **two interleaved Pareto trade-offs**.
No reference-/render-correlation discriminator buys back the audible part for free. **balanced stays
byte-equal; gentle (−20) is an opt-in speech-preserving operating point, NOT a fix.** Step 2
(render-correlation / near-priority masking) is closed as a default lever.

## How the case decomposes (do NOT average 6–7s — credit: codex split)
Layer attribution per fine band, `mic → NORES (linear) → PROD (RES/SG)`; `near_fr = 1 − coherence(mic,far)`:

### 6.2–6.5 s — ONSET (mid/broadband)
| band | mic | NORES | PROD | lin_cut | RES_cut | near_fr |
|------|-----|-------|------|---------|---------|---------|
| 250–500 | 5.1 | 0.4 | −11.7 | −4.7 | **−12.2** | 0.69 |
| 1–2k | −12.8 | −16.9 | −27.4 | −4.1 | **−10.5** | **0.10** |
| 2–3k | −24.1 | −26.0 | −31.1 | −1.9 | −5.1 | 0.72 |

Deep cuts at 250-500 / 1-2k, but `near_fr` there is low (0.10–0.69) → the onset **correlates with far**
(it is double-talk; far is active). Both linear and RES participate. A *default-safe* detector cannot
claim this as near without leaking the (real, co-active) echo. Some 1–3k near-priority headroom exists
but the discriminator is insufficient.

### 6.5–7.0 s — FRICATIVE
| band | mic | NORES | PROD | lin_cut | RES_cut | near_fr |
|------|-----|-------|------|---------|---------|---------|
| 80–500 | ~15 | ~14 | ~14 | ~−1 | ~0 | 0.97 (preserved ✓) |
| 1–2k | −15.1 | −15.3 | −19.0 | −0.2 | −3.7 | 0.84 |
| 2–3k | −16.0 | −15.9 | −19.2 | +0.1 | −3.3 | 0.76 |
| 3–4k | −13.3 | −13.5 | −17.9 | −0.2 | −4.4 | 0.84 |
| **4–6k** | −27.0 | −24.7 | **−49.6** | **+2.3** | **−24.9** | **0.13** |
| 6–8k | −33.8 | −33.4 | −45.6 | +0.3 | −12.2 | 0.36 |

- **4–6k is where it perceptually dies, but `near_fr = 0.13`** → that band is **echo-dominated**; the
  fricative air is a minority component buried under far-correlated echo (linear even *adds* +2.3 dB).
  **Any reference/render discriminator sees mostly echo there → cannot preserve it without releasing
  real echo. Essentially closed.**
- The only genuinely near-like, RES-cut region is **1–3k** (`near_fr` 0.76–0.84, RES cuts 3–4 dB) =
  the body/articulation, not the air.

## Why the linear stage is NOT the lever (answers "is linear already good?")
Three-way per-band, breath window, vs mic (negative = cut):
```
2–8k mean suppress:   FINAL −4.53 dB    NORES (linear only) −0.20 dB    Speex −1.03 dB
FS echo 0.2–2s:       FINAL −19.1 dB    NORES −4.7 dB                   Speex −5.0 dB
```
- **Our linear stage already preserves the breath BETTER than Speex** (NORES −0.20 vs Speex −1.03).
  The −4.53 dB breath cut is **100 % the RES**, not the subtractor.
- **Speex only looks "perfect" because it is a weak canceller** (−5.0 dB FS echo ≈ our linear-only
  −4.7). Its gentleness = our NORES operating point. Our RES buys +14 dB FS echo at the cost of the
  breath. So this is an RES-aggressiveness trade, correctly located.

## Why the masking / near-priority discriminator fails as a default (the kill-tests)
1. **Invalid-control correction.** The 6–7s-average "does not separate" was driven by `0I0X-FSmv`,
   which has **no real echo** (echo_presence −1.2 dB, coh 0.004, bestlag≈0). Removed as a leak control.
2. **Window-averaged, 1–3k DOES separate** vs valid real-echo controls and a 20-case FS population:
   ```
   band  breath  FS_med  FS_p75  FS_p90  FS_max   (near_frac × env_detrack, far-active)
   1–2k   0.68    0.08    0.16    0.23    0.56   breath > FS_max
   2–3k   0.74    0.03    0.17    0.36    0.45   breath > FS_max
   3–4k   0.45    0.09    0.25    0.43    0.77   FS_max 0.77 leaks
   ```
3. **Per-frame (real-pipeline) gate kills it.** 250 ms causal window + 8-frame hangover, threshold sweep:
   ```
   2–3k:  thr 0.65 → breath_open 60 %  but  FS_leak 12.1 %   (1–2k worse: 34 % / 11 %)
   ```
   The window-average smoothed away FS per-frame flicker; per-frame, ~12–17 % of FS far-active frames
   spike above any breath-opening threshold. A 12 % per-frame 1–3k floor-lift on far-active frames will
   break the **FS_movement > 3.5** hard gate. Not default-safe.

Other axes already falsified this arc (re-verified, not memory-trusted): render-power ERL floor
(−35 dB wrong direction), SRO/clock-drift (erratic, r²≈0), HDRES harmonic (breath HF coh 0.013),
gain-rule replacement (gain is AEC3 ENR/EMR `GainToNoAudibleEcho`, already near-end-aware), mu[HF]-damp
(+0.009 corpus). `env_detrack` here also instantiates and closes the background review's "untested third
axis" (multi-frame envelope/inter-frame correlation): moving echo decorrelates in magnitude too, so it
flags FS-movement as near-like.

## Decision
- **balanced = v3.22.4, byte-equal.** Do not move balanced for this single case.
- **gentle (`min_gain_floor_far_active_db = −20`) = opt-in near-priority preset.** Documented as a
  **speech-preserving operating point, not a repair.** On this case it is itself a weak lever
  (−28→−20 recovers only +0.6 dB breath at ~0.9 dB FS cost — the breath is ENR/EMR-fire-driven, not
  floor-clamped).
- **Step 2 (render-correlation / near-priority masking) closed for default.** 4–6k unrecoverable
  (echo-dominated); 1–3k separable only window-averaged, per-frame FS-leak too high; recoverable part is
  the 1–3k body, not the audible 4–6k air. Not worth audio A/B — it would re-derive a known trade-off.
- **Not pursued (deliberate):** SPP-control + longer hangover to push 2–3k FS-leak 12 %→<3 %. Even if
  successful it recovers only the 1–3k body, returns to detector-tuning micro-loops, and is gated out by
  the user. Recorded as the one remaining (low-yield) lever, not taken.

## Repro (logging-only, byte-equal probes under /tmp)
`linear_review.py` (3-way per-band), `split_full.py` (codex split + layer attribution + per-segment
separation), `echo_presence.py` (control validity), `align_check.py` (block-align artifact test),
`fs_population.py` (20-case leak distribution), `perframe_feasibility.py` (per-frame open/leak sweep),
`gentle_preset_curve.py` (operating-point curve).
