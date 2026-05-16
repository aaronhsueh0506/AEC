# v3.19 Volterra V.3 — cohort sizing (2026-05-16)

**Status**: Doc-only deliverable for v3.19 Phase 0.1 §5 auto-trigger
Option (B) fold-in. Re-evaluates the listening cohort for v3.20
Volterra arc. Consumes V.1 + V.2 + v3.18 C.E single-case extremes.

**Anchor**: v3.13 E4.S1 verdict (`docs/archive/v3_13_e4_s1_verdict.md`)
established the original 5/5 listen-validated NL cohort on FS bucket
M3 > 9.0. v3.18 C.E closeout (`docs/v3_18_c_e_closeout.md`) surfaced
new high-magnitude single-case extremes worth re-listening for NL
signature.

## 1. v3.13 NL cohort (5/5 listen-validated)

Source: [docs/archive/v3_13_e4_s1_verdict.md](archive/v3_13_e4_s1_verdict.md)
§"Listen validation (2026-05-13)". User listen verdicts captured on
`_mic.wav` + `_lpb.wav` pairs in `listen/v3_13_e4_s1_nl_candidates/`.

| # | Stem | Bucket | M3 | M2 | M5 | Listen verdict | Sub-type |
|---|---|---|---:|---:|---:|---|---|
| A | `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | FS_static | 11.92 | 23.40 | 0.53 | 喇叭推爆 (lpb 正常) | **Type 1** loudspeaker physical NL |
| B | `9xjhiFbGo06hdQIsHTS6qA_farend_singletalk` | FS_static | 10.66 | 21.13 | 0.77 | 失真/非線性, 類似無線電/廣播/通話 | **Type 2** transmission codec NL |
| C | `IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement` | FS_movement | 9.68 | 16.50 | 0.74 | 失真/非線性, 同 Type 2 | **Type 2** |
| D | `WTdBhXa080WJEeGDde9BGA_farend_singletalk` | FS_static | 9.56 | 26.74 | 0.84 | 失真/非線性, 同 Type 2 | **Type 2** |
| E | `m4789fdio0q92zjf9gvh1Q_farend_singletalk` | FS_static | 9.55 | 21.83 | 0.73 | 失真/非線性, 同 Type 2 | **Type 2** |

### 1.1 NL-signature analysis per case

| # | Stem | NL signature | Volterra path for fix |
|---|---|---|---|
| A | Gsy0lC5 | Loudspeaker amplifier overdrive — odd-harmonic comb (H3/H5/H7 visible in residual spectrum), saturation-type curve | **Hammerstein-tractable** (V.1 §3.1). Stenger preprocessor with `a_3` capturing soft-clip; `a_1`/`a_3` jointly identified. Expected primary win case for v3.20. |
| B | 9xjhiFb | Codec NL — bandpass-shaped + broadband intermodulation, "無線電/廣播" character. Phase distortion present. | Hammerstein partial — `a_3` captures odd-symmetric component. Phase distortion partially reachable via `a_2` cross-coupling. Expected MED improvement. |
| C | IrQvqOTC_mvmt | Codec NL Type 2 + path movement (FS_movement bucket). Same NL signature as B but compounded with adaptation tracking. | Same as B; preprocessor adapts during steady FS windows between movements. Hardest case (path tracking + NL identification simultaneous). |
| D | WTdBhX | Codec NL Type 2; highest M2 (26.74) in cohort = strongest broadband NL content | Hammerstein partial; broadband NL component may exceed polynomial-3 modelling capacity. Candidate for Yang 2014 fallback if Hammerstein insufficient. |
| E | m4789f | Codec NL Type 2; standard Type 2 profile | Same as B. |

### 1.2 Status as gating cohort

All 5 cases:
- Pass M3 > 9.0 FS-bucket detector (100% precision; 100% recall on
  validated set)
- Are reproducible on demand from `wav/aec_challenge_blind/farend_singletalk/`
  and `wav/aec_challenge_blind/farend_singletalk_with_movement/`
- Have rendered listening materials preserved in
  `listen/v3_13_e4_s1_nl_candidates/`
- Failed v3.13 E4.S6a/S6b at -12 / -18 / -24 / -30 dB amplitude mask
  ("NO AUDIBLE NL REDUCTION at any level") — this is the BASELINE
  Volterra must beat
- Do NOT overlap with xrtntuju 5-clip regression set (per E4.S1 §"xrtntuju
  cross-reference")
- Do NOT overlap with cohort tail qNvSMyU (P52 PathChangeRegimeHandler
  invariant)

## 2. New candidates from v3.18 cycle

Source: [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) §"Per-case
extremes". v3.18 C.E `c_e_res_use_fq_usable` ON produced unprecedented
single-case AECMOS movement. The two extreme single-case shifts deserve
NL-signature listen-check before v3.20 cohort lock.

| Stem | Bucket | C.E Δecho | C.E Δdeg | Listen check question |
|---|---|---:|---:|---|
| `sUQrHEPA_DT_static` | DT_static | -0.209 | **+1.2000** | Largest single-case Δdeg ever. Did C.E relax RES on a frame that has phase-distortion NL? If YES → Volterra adds DT-side win on same frame. |
| `N2rQLbnp_FS_static` | FS_static | **-0.490** | +0.001 | Worst FS Δecho ever from C.E. Is the failed RES suppression preserving a loudspeaker-NL residual that Volterra would have removed at the *source*? |
| `oQK3bV_DT_movement` | DT_movement | +0.170 | +0.7501 | Win-win on C.E. NL signature here? |
| `wZvBJ5R4_DT_movement` | DT_movement | +0.224 | +0.5763 | Win-win on C.E. NL signature here? |

### 2.1 Why these matter for v3.20

The v3.18 C.E closeout finding (Pareto wall +1.2 Δdeg DT vs -0.490
Δecho FS) is consistent with Volterra being an **orthogonal axis** —
if Volterra removes the *source* NL, the FS residual ENR drops, and
RES can relax suppression with no echo cost. This is the v3.20
"compound win" hypothesis: Volterra-on UNLOCKS C.E.

For the v3.20 cohort, these 4 candidates are **stretch-goal listening
materials** (not in-cohort gates). If Volterra-on produces audible
improvement on sUQrHEPA / N2rQLbnp, that's evidence the Pareto wall
identified in v3.18 C.E is broken by adding the upstream Volterra layer.

### 2.2 Listen-check before v3.20 cohort lock

**Action**: Before v3.20 Phase 0 kickoff, render `_mic.wav` + `_lpb.wav`
for the 4 candidates above and have user perform NL-character listen.
Score each as:
- **NL-positive**: 爆掉 / 無線電 / saturation transient audible → add to
  v3.20 cohort
- **NL-negative**: clean linear echo / DT mix only, no NL character →
  exclude (v3.20 cohort stays at 5)
- **Ambiguous**: unclear → use as stretch goal only

Expected outcome (priors): N2rQLbnp_FS_static likely NL-positive (FS
worst with extreme Δecho fits the cohort tail M3 profile); sUQrHEPA
likely NL-negative (DT_static, DT signatures usually NE-dominated). But
verify with listen before commit.

## 3. Pre-bench oracle thresholds

For each cohort case, state the expected per-case ΔERLE (or audible
character shift) Volterra should achieve to count as "audibly reduced
NL". Calibrated against the v3.13 E4.S6a "no audible reduction"
outcome at -12 dB amplitude mask (which produced +1.11 dB ERLE on case
A but **zero perceived NL change**).

### 3.1 Calibration baseline

E4.S6a (12 dB amplitude mask) at full aggression:
- Gsy0lC5 ERLE: +1.11 dB → **perceptually neutral**
- 9xjhiFb ERLE: +0.67 dB → **perceptually neutral**
- IrQvqOTC_mvmt ERLE: +0.68 dB → **perceptually neutral**

E4.S6b (30 dB mask) at maximum aggression:
- Gsy0lC5 ERLE: +1.60 dB → **damages voice formants + still no NL char change**

**Conclusion from v3.13 E4 closure**: linear ERLE > +1.5 dB on these
cases is **insufficient evidence** of NL improvement. Perceptual NL
character requires fundamentally different signal-domain change (per
V.1: time-domain Volterra inverse, not frequency-domain mask).

### 3.2 v3.20 Volterra acceptance thresholds (per-case)

| Case | Min ΔERLE | Listen verdict required for PASS |
|---|---:|---|
| A Gsy0lC5 (Type 1) | **+3 dB** | "輕微推爆" or "正常" (shift away from "推爆") |
| B 9xjhiFb (Type 2) | +2 dB | "可接受" or "正常" (shift away from "失真/無線電") |
| C IrQvqOTC_mvmt (Type 2 + mvmt) | +1.5 dB | "明顯改善" but maintaining echo cancellation through path change |
| D WTdBhX (Type 2 broadband) | +1.5 dB | "可接受" or "明顯改善" |
| E m4789f (Type 2) | +2 dB | "可接受" or "正常" |

**Type 1 (case A) has the highest threshold (+3 dB)** because
Hammerstein is the physically-correct model — Volterra should achieve
substantial ERLE gain there. **Type 2 (B-E) have lower threshold
(+1.5 to +2 dB)** because Hammerstein partial reach; passing one of
these (even at +2 dB ERLE) requires perceptual confirmation that the
"無線電" character is reduced, not just dB count.

### 3.3 Why these thresholds are higher than E4.S6 baseline

E4.S6 amplitude mask achieved up to +1.60 dB ERLE on case A with
**zero perceptual change**. v3.20 must beat this AND deliver
perceptual change. The +3 dB threshold for case A is **2× the
E4.S6b ceiling** — well above the amplitude-mask family's reach by
construction.

### 3.4 Aggregate cohort PASS criterion

**PASS**: 5/5 cases meet BOTH per-case ΔERLE AND listen verdict.

**PARTIAL PASS** (triggers V.1 §9 fallback to Yang 2014 2nd-order
NLMS-Volterra): 3-4/5 cases pass.

**FAIL** (closes v3.20 cycle): ≤ 2/5 cases pass.

## 4. Cohort size justification — why 5? why not 10 or 20?

### 4.1 5 is the natural M3 > 9.0 threshold cohort

Per v3.13 E4.S1 audit: across all 800 cases, **exactly 5 land at M3 >
9.0 within the FS bucket**. This is the empirically-anchored
"definitely-NL" tier. M3 > FS p95 (7.87) widens to 15 cases but adds
10 unvalidated candidates that may dilute the cohort with non-NL
content (per E4.S1 §"Detector precision" table).

### 4.2 Listen-fatigue ergonomics

User listening sessions in v3.13 E4 closeout (S6a + S6b) used 3 cases
× 5 versions each = 15 listening clips per session. Verdict notes
showed user fatigue patterns after ~12 clips. For v3.20 the
expected listen burden per phase is:
- Phase 2.2 baseline (μ=0.1): 5 cases × 2 versions (ON / OFF) = 10
  clips — comfortable single-session
- Phase 2.3 μ sweep (4 values): 5 × 4 = 20 clips — borderline; should
  split across 2 sessions
- Phase 2.4 edge cases (NE + xrtntuju): +5-7 additional clips

**A 10-case cohort would push Phase 2.3 to 40 clips per session —
fatigue territory.** A 20-case cohort makes the listen gates impractical.

5 is the upper bound for single-session listenability.

### 4.3 A/B click-through ergonomics

Per v3.13 E4.S6a/S6b methodology: paired listening (`_mic.wav` →
`_volterra_off.wav` → `_volterra_on.wav`) requires ~30 sec per case
to form a verdict. 5 cases × 30 sec × 2 passes (ON / OFF) ≈ 5 min
total verification time — fits a single review block.

10 cases would be 10-15 min — acceptable but tedious. 20 cases would
be 20-30 min — beyond typical session attention.

### 4.4 Bench-cost amortisation

Each isolated-bench render of cohort takes ~10-20 sec per case on
local CPU (per v3.13 E4.S6 timings). Full sweep over μ_NL grid
× cohort size:
- 5 cases × 4 μ values = 20 renders × 15 sec = 5 min
- 10 cases × 4 μ values = 40 renders × 15 sec = 10 min
- 20 cases × 4 μ values = 80 renders × 15 sec = 20 min

All tractable, but the 800-case Phase 3 bench costs (~45-90 min for
full A/B per CLAUDE.md) dominate. Cohort renders are negligible at
any size. Bench cost is NOT the constraint; listening cost IS.

### 4.5 Statistical signal-to-noise

5 cases gives:
- Single-case loss/gain on PASS = ±20% cohort signal
- Per-case ERLE std typically ~0.5 dB on bench reproducibility
- Aggregate cohort ERLE std ≈ 0.5/√5 ≈ 0.22 dB

For acceptance thresholds in §3.2 (+1.5 to +3 dB), this is **6-13×
SNR**. Adequate.

10 cases would give 9-18× SNR — diminishing return for 2× listen
burden.

### 4.6 Stretch goal — expansion from §2 candidates

If §2 listen-check yields 1-3 NL-positive cases from
{sUQrHEPA, N2rQLbnp, oQK3bV, wZvBJ5R4}, **add to v3.20 cohort as
"stretch candidates"** (do NOT block PASS on stretch cases). This
provides:
- Compound-win evidence for Volterra + C.E synergy hypothesis
- Coverage of DT bucket NL (currently 5/5 cohort is FS bucket only)
- Up to 8-case total cohort while preserving the 5-case core PASS gate

## 5. Cohort-tail invariant (qNvSMyU)

Per CLAUDE.md `feedback_aec_code_review_accuracy.md`: qNvSMyU FS_static
case is P52 PathChangeRegimeHandler's load-bearing cohort-tail
defence. v3.20 Volterra MUST NOT regress this case (per V.2 §6.3
hard bar: Δecho ≥ -0.020).

qNvSMyU is **NOT** in the v3.13 NL cohort (per E4.S1 §"Detector
precision" — M3 below 9.0 threshold). qNvSMyU's failure mode is
**linear-filter divergence under acoustic regime change**, not
loudspeaker NL. Volterra preprocessor should be **gated OFF** during
qNvSMyU-like regime change via the adapt-gate's `consistent_estimate`
condition (which fails when filter is unstable).

**Validation step in v3.20 Phase 3.2**: confirm Volterra adapt-gate
fire rate on qNvSMyU = ~0% (preprocessor stays at identity throughout).
If fire rate > 5% on qNvSMyU, the adapt-gate is mis-calibrated.

## 6. xrtntuju 5-clip invariant

Per CLAUDE.md memory `project_xrtntuju_regression_clip.md`: 5 DT
positive listening windows on `xrtntuju*` stems. v3.20 Volterra MUST
NOT regress any (per V.2 §6.4 hard bar).

xrtntuju cases are **DT-bucket**; Volterra preprocessor adapt-gate
should freeze on DT (per V.2 §4.2 `dt_combined > 0.3` freeze). With
correct gating, Volterra preprocessor is at identity during xrtntuju
DT windows → byte-equal output to baseline.

**Validation step in v3.20 Phase 3.3**: render xrtntuju 5-clip with
volterra_on and compare to baseline. Expected: byte-equal (or within
1e-6 numerical tolerance per `-ffp-contract=off` parity).

## 7. Listening material checklist for v3.20

For v3.20 Phase 2.1, render the following materials:

```
listen/v3_20_volterra_phase2/
├── core_cohort/                          # 5 cases (§1)
│   ├── A_Gsy0lC5/
│   │   ├── _mic.wav                       # original
│   │   ├── _lpb.wav                       # reference
│   │   ├── _volterra_off.wav              # baseline (v3.20 base)
│   │   ├── _volterra_on_mu_p1.wav         # μ_NL_scale = 0.1
│   │   ├── _volterra_on_mu_p05.wav        # μ_NL_scale = 0.05 (Phase 2.3)
│   │   ├── _volterra_on_mu_p2.wav         # μ_NL_scale = 0.2 (Phase 2.3)
│   │   └── _volterra_on_mu_p5.wav         # μ_NL_scale = 0.5 (Phase 2.3)
│   ├── B_9xjhiFb/   (same structure)
│   ├── C_IrQvqOTC_mvmt/   (same)
│   ├── D_WTdBhX/   (same)
│   └── E_m4789f/   (same)
├── stretch_cohort/                       # 0-4 cases from §2 listen-check
│   └── ...
└── edge_cases/                            # invariant verification
    ├── qNvSMyU_FS_static/                # cohort tail
    └── xrtntuju_5clip/                    # 5 DT positive windows
```

Render commands (per v3.13 E4.S6 pattern):
```bash
python3 python/aec.py wav/aec_challenge_blind/farend_singletalk/<stem>_mic.wav \
    wav/aec_challenge_blind/farend_singletalk/<stem>_lpb.wav \
    listen/v3_20_volterra_phase2/<group>/<name>/_volterra_off.wav \
    --preset balanced --filter 832 --cng --enable-res

python3 python/aec.py wav/aec_challenge_blind/farend_singletalk/<stem>_mic.wav \
    wav/aec_challenge_blind/farend_singletalk/<stem>_lpb.wav \
    listen/v3_20_volterra_phase2/<group>/<name>/_volterra_on_mu_p1.wav \
    --preset balanced --filter 832 --cng --enable-res \
    --volterra-enabled --volterra-mu-scale 0.1
```

(CLI flag wiring is part of v3.20 Phase 0 work; this is the target spec.)

## 8. Cross-references

- [docs/v3_19_volterra_v1_literature_survey.md](v3_19_volterra_v1_literature_survey.md) — V.1 (Type 1 vs Type 2 modelling theory)
- [docs/v3_19_volterra_v2_design_lock.md](v3_19_volterra_v2_design_lock.md) — V.2 (adapt-gate / freeze-gate spec)
- [docs/archive/v3_13_e4_s1_verdict.md](archive/v3_13_e4_s1_verdict.md) — Original 5/5 NL cohort
- [docs/archive/v3_13_e4_s6a_s6b_verdict.md](archive/v3_13_e4_s6a_s6b_verdict.md) — Amplitude-mask listen baseline (NO AUDIBLE NL REDUCTION across full -12 to -30 dB sweep)
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — Single-case extremes (sUQrHEPA / N2rQLbnp / oQK3bV / wZvBJ5R4)
- [docs/archive/v3_13_e5_closure_verdict.md](archive/v3_13_e5_closure_verdict.md) — E5.S3 detector (mic-lpb correlation; reused as adapt-gate OR-arm)
- CLAUDE.md `feedback_aec_code_review_accuracy.md` — cohort tail qNvSMyU invariant
- CLAUDE.md memory `project_xrtntuju_regression_clip.md` — xrtntuju 5-clip invariant

## 9. Verification rules followed

1. Cohort anchored to v3.13 E4.S1 listen verdict (5/5 user-validated)
2. Sub-type categorisation preserved (1 Type 1 + 4 Type 2) — drives per-case ERLE thresholds
3. v3.18 C.E single-case extremes evaluated as stretch candidates (listen-check before v3.20)
4. ERLE acceptance thresholds calibrated AGAINST v3.13 E4.S6b ceiling (linear ERLE alone insufficient)
5. Cohort size justified on 4 dimensions: detector threshold + listen fatigue + A/B ergonomics + bench cost
6. Cohort-tail (qNvSMyU) + xrtntuju 5-clip invariants explicitly checked in §5 / §6
7. Listening material structure compatible with v3.13 E4.S6 listening pattern
