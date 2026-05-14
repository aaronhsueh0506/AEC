# v3.13 E5 Saturation deepening — final closure verdict

**Status**: E5 arc CLOSED. CANNOT SHIP. All four variants (S2 / S3 / S4a / S4b)
sit on a fundamental FS-vs-DT trade-off line; none achieves Pareto
improvement vs baseline. Filter-protection actions on acoustic-NL frames
are trade-off bound by mechanism.

**Date**: 2026-05-14

## Headline

| Variant | FS_static Δecho | DT_static Δdeg | DT_movement Δdeg | Verdict |
|---|---:|---:|---:|---|
| Baseline | 0 | 0 | 0 | reference |
| **S2** lower threshold (sat_level > 0.05) | +0.076 | **-0.049** | **-0.021** | FAIL DT bar |
| **S3** mic-lpb correlation gate (r > 0.35) | +0.073 | **-0.043** | **-0.018** | FAIL DT bar |
| **S4(a)** S3 + state-gated freeze (only when filter unstable) | +0.080 | **-0.043** | **-0.018** | FAIL DT bar (= S3) |
| **S4(b)** S3 detector + shadow_rise mask only (no mu freeze) | **+0.095** | **-0.052** | **-0.024** | FAIL DT bar (worst) |

Hard bars: `FS / DT bucket Δdeg ≥ -0.005`. All variants FAIL DT bar by 4-10x.

## Mechanism — why all four variants are trade-off-bound

E5.S1 audit (commit 1786f08, docs/v3_13_e5_s1_verdict.md): listen 8-case
"clipping" perception is **acoustic-path NL distortion** (peaks in 0.7-0.95
mic amplitude band), NOT digital ADC clipping (clip@99 = 0% on all 8).
F-E5 production bundle's existing actions (mic soft-clip, main_mu freeze,
error_psd reset, shadow_rise mask) gate on `_saturation_level`, which
maxes 0.086 on listen cases — well below the 0.3 / 0.5 thresholds.

E5 sub-arcs explored two question dimensions:
1. **Detector**: amplitude threshold (S2) vs correlation-based (S3/S4)
2. **Action**: combined freeze + mask (S2/S3) vs state-gated freeze (S4a)
   vs mask only (S4b)

### S2 verdict (commit pending) — lower F-E5 threshold to 0.05

Approach: simply lower `f_e5_main_mu_sat_threshold` from 0.5 → 0.05 so
existing F-E5 actions (mu freeze + shadow_rise mask) fire on acoustic-NL
frames where `_saturation_level` reaches ~0.05-0.086.

Result: FS_static +0.076 / DT_static -0.049 → FAIL.

Mechanism: amplitude threshold is imprecise. On DT frames where echo+voice
both raise sample amplitude into the 0.7-0.95 band, sat_level also rises
above 0.05 → freeze fires → main filter doesn't track DT echo → Δdeg drops.

### S3 verdict — mic-lpb correlation gate (new detector)

Approach: per-frame Pearson r(mic, lpb) within 0.7-0.95 mic peak band.
Fire when r > 0.35 AND far_active. NEW detector independent of F-E5.

Smoke verification (3 NL cases + 1 NE control):
- Gsy0lC5 (Type 1 NL): 14.4% fires
- IrQvqOTC FS_static: 32.7% fires
- IrQvqOTC FS_movement: 56.2% fires
- 014AzuqPZku (NE): **0% fires** (clean separation)

Smoke detector quality = excellent. Byte-equal flag-OFF verified.

Result on 800-case: FS_static +0.073 / DT_static -0.043 → FAIL.

Mechanism: detector itself is precise (proven by 0% NE FP), but the same
correlation signature (mic responding to lpb in 0.7-0.95 band) appears in:
- FS-NL frames (acoustic NL — desired fire)
- DT high-echo frames (mic = echo + NE voice; echo correlates with lpb)
**Detector cannot distinguish FS-NL from DT high-echo** because both have
identical correlation signature in the band of interest. Same trade-off
as S2.

### S4(a) verdict — state-gated freeze (filter_state aware)

Hypothesis: DT regression in S2/S3 came from freezing main_mu when
filter_state == 'refined_usable' (filter is already stable; freeze does
no protective work, just disrupts adaptation). Solution: only freeze when
filter_state ∈ {coarse_learning, suspicious_dt, diverged, startup}.

Implementation: `_e5_s3_latched_raw AND _prev_filter_state not in
('refined_usable', 'idle')`.

Result: **identical to S3** (FS_static +0.080 vs S3 +0.073, DT_static
-0.043 same as S3). State-gating made almost NO difference.

Mechanism: when the correlation gate fires on FS-NL or DT high-echo
frames, filter_state is RARELY 'refined_usable'. NL itself perturbs the
filter into coarse_learning / suspicious_dt; DT high-echo frames may
also be in non-refined state. Therefore the state filter doesn't
exclude any meaningful subset of fires. State-gating is the wrong axis.

### S4(b) verdict — shadow_rise-only (drop mu freeze)

Hypothesis: freeze action is the source of FS gain AND DT loss. Removing
freeze should drop both FS gain and DT loss back to baseline.

Implementation: drop `_e5_s3_latched OR` from mu freeze condition (line
6087); keep shadow_rise mask at line 6363 unchanged.

Result: FS_static **+0.095 (HIGHEST)** / DT_static **-0.052 (WORST)**
→ FAIL.

Mechanism: hypothesis was reversed. Removing mu freeze:
- INCREASED FS gain (+0.095 vs S3 +0.073) — freeze was actually
  suppressing some valid FS adaptation
- INCREASED DT loss (-0.052 vs S3 -0.043) — freeze was providing some
  DT protection (preventing wrong adaptation during NL)

Reality: **shadow_rise mask** (preventing false EPC fire on NL frames) is
the dominant FS-vs-DT trade-off mechanism. The mu freeze action only
slightly shifts the trade-off (small FS reduction in exchange for small
DT protection).

## The fundamental trade-off

All four variants land on a single FS-vs-DT trade-off line:

```
DT_static Δdeg
   0 +-------- baseline
     |
-0.02|
     |
-0.04|   . S3, S4(a)         . S2
     |
-0.06|       . S4(b)
     +---------------------------------
     0    +0.05    +0.075   +0.10   FS_static Δecho
```

Slope: ~ -0.5 dB DT per +1 dB FS (regression : improvement ratio).

This is the **canonical FS-vs-DT trade-off** that any amplitude-layer NL
detector + filter-protection action will sit on. The mechanism is:
1. Detector fires on mic-lpb correlation in 0.7-0.95 band
2. SAME correlation signature appears on FS-NL frames AND DT high-echo
3. Action (freeze / mask) helps FS-NL, hurts DT high-echo
4. Cannot distinguish frame types → cannot decouple actions
5. Trade-off is geometric — moving along the line, not off it

To break the trade-off, need a detector that distinguishes FS-NL from
DT high-echo OR an action that helps both. Within amplitude / frequency
domain, neither is achievable:
- Amplitude detector cannot distinguish (both produce same evidence)
- Multiplicative mask action cannot fix NL perceptually (E4 closure)

The canonical breakthrough requires **time-domain Volterra non-linear
inverse filter** that explicitly models the NL transfer h_nl(ref → mic_NL)
and applies its inverse. This is a structural change exceeding v3.13
scope, deferred to v3.14 (see E4.S6a/S6b verdict commit 3e10621).

## What's preserved as research substrate

The S3 correlation detector code remains in worktree `s4a` and `s4b`
branches (uncommitted to main). For v3.14 reuse:
- Per-frame Pearson r(mic, lpb) in 0.7-0.95 mic peak band
- Threshold r > 0.35 (E5.S1 audit anchored)
- Hold counter 5 frames (hysteresis)
- 0% NE false positive on smoke
- 14-56% fire rate on real NL cases
- Independent of F-E5 amplitude threshold

This detector is REUSABLE for any future NL processing arc (Volterra,
neural NL inverse, or other). It identifies WHICH frames need NL
processing — the design problem for v3.14 is the ACTION on those frames.

## What's retired

- E5.S2 threshold lowering (FAIL by DT regression)
- E5.S3 correlation gate + mu freeze action (FAIL same trade-off)
- E5.S4(a) state-gated freeze (FAIL — state filter misses true cause)
- E5.S4(b) shadow_rise-only (FAIL same trade-off, different position)

Worktrees `agent-a3c6eebf4ab4289a4`, `s4a`, `s4b` can be removed after
this verdict is committed. The detector design + verdict serve as
v3.14 inputs.

## E5 arc lessons (forward)

1. **Acoustic NL ≠ digital clipping**. F-E5 bundle's design assumed
   digital signature (sample > 0.95 + flat-top); acoustic NL lives in
   0.7-0.95 band with smooth amplitude. Rule: detector must match
   the actual NL signature, not the production assumption.

2. **Detector precision ≠ Pareto improvement**. S3 had perfect detector
   (0% NE FP, 14-56% fire on NL) but still hit the FS-DT trade-off.
   Precise detection on the wrong feature is still trade-off bound.

3. **State-gating doesn't help when state is correlated with detection**.
   S4(a) hypothesis was that 'refined_usable' state was the DT-harm
   source; reality is filter is rarely in that state when correlation
   gate fires. State filter must orthogonalize the noise — here it didn't.

4. **The trade-off slope is the architectural limit**. Moving along the
   line (different actions, different gates) shifts position but doesn't
   break it. Breaking it requires a detector that distinguishes the
   physical mechanism (FS-NL has speaker-driven distortion; DT high-echo
   has speaker + NE voice — distinguishable in time-domain non-linear
   transfer modeling).

5. **shadow_rise mask is the bigger lever than mu freeze**. Counter to
   intuition. EPC false-fire prevention helped FS more than mu freeze
   (and hurt DT more). Future EPC reform (v3.14+ Phase 4) should
   consider this evidence.

## v3.13 arc state after closure

| Arc | Status |
|---|---|
| E2 Delay | ✅ SHIPPED (Path 3 max_delay 250→1024 ms) |
| E4 NLP | ❌ CLOSED CANNOT SHIP (amplitude family ruled out) |
| **E5 Saturation deepening** | ❌ **CLOSED CANNOT SHIP** (filter protection trade-off bound) |
| F-HFR per-band Q/R | ⏳ v3.14 candidate |
| E1 mic_dynamic_margin | ⏳ v3.14 candidate |

**Front-end work for v3.13 is now complete (per user directive 2026-05-14:
"如果前端filter做完都沒問題的話 後端res就接著往下做"). Next: back-end
Phase 3 RES canonical refactor (gain_floor unification + 4-cap ranked
priority + per-state ENR tuple).**

## v3.14 NL processing arc — design lock placeholder

E4 + E5 closures both point to v3.14 Volterra non-linear inverse design:
- **Detector**: reuse E5.S3 correlation gate (proven 0% NE FP, 14-56% NL fire)
- **Action**: time-domain Volterra inverse filter h_nl⁻¹(mic_NL → mic_clean)
- **Pipeline**: pre-PBFDKF on mic OR post-RES on output (TBD)
- **Adaptation**: NLMS / RLS on Volterra coefficients
- **Position**: TBD — depends on whether NL appears in mic input or
  in residual after AEC
- **Compute**: Volterra is O(K²) for kernel length K; significant vs PBFDKF

This is a 6+ month dedicated arc. Open after v3.13 closes.

## Verification rules followed

1. 800-case j4 with `preset=balanced / fl=832 / cng=True`
2. Worktree isolation per S4 sub-variant (file-disjoint per design lock)
3. Byte-equal flag-OFF verification on S3 detector
4. Smoke verification on 3 NL + 1 NE control cases
5. Cohort tail check: all variants stay within Δecho ±0.05 on cohort tail
6. NE bucket Δdeg = -0.002 across all variants (= existing baseline noise)

## Artifacts (committed or in worktrees)

- E5.S1 audit + verdict: commit 1786f08, docs/v3_13_e5_s1_verdict.md
- E5.S2 env override: python/eval_aec_challenge.py (env AEC_F_E5_MAIN_MU_THRESHOLD)
- E5.S3 correlation detector: worktree `worktree-agent-a3c6eebf4ab4289a4` (commit 12aa172)
- E5.S4(a) state-gated freeze: worktree `s4a-state-gated`
- E5.S4(b) shadow_rise-only: worktree `s4b-shadow-only`
- Bench results: results/v3_13_e5_s2/, results/v3_13_e5_s3/, results/v3_13_e5_s4a/, results/v3_13_e5_s4b/
- This verdict: docs/v3_13_e5_closure_verdict.md
