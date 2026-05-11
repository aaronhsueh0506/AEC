# P52 A.0R.7 — 800-case regime distribution + handler fire profile

**Date**: 2026-05-11
**Branch**: `feature/p52-phase-a-shadow`
**Classifier commit (locked)**: `401e62b` (A.0R.3) — thresholds
`stable_max_db = 9.43`, `mild_max_db = 21.04`, anchored to post-mortem
800-case ERL_decile_std distribution (p90 / p99).
**AEC code commit**: `22bd3d9` (Path 3 HEAD)
**Run command**: `python tools/research/p52_a0r7_regime_distribution.py --out /tmp/p52_a0r7/dist.json -j 4`
**Run wall time**: 561 s (j=4)

## Verdict: **Outcome A** — distribution as expected; proceed to A.0R.8 + merge

| Sanity check | Expected | Actual |
|---|---|---|
| Sum of regime counts | 800 | **800** ✓ |
| `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` in `wildly_nonstationary` | yes (per post-mortem p99.2) | **yes**, std = 23.39 ✓ |
| No NaN / Inf in `erl_decile_std_db` | 0 cases | **0** ✓ |
| `wildly_nonstationary` count | 5 – 15 | **7** ✓ |
| `mildly_nonstationary` count | 50 – 100 | **59** ✓ |
| `stable` count | 685 – 745 | **734** ✓ |

## Methodology

Per case (mic + lpb wav):
1. **Classify** via `AcousticRegimeClassifier()` (`python/aec_p52_regime_classifier.py`).
2. **Run AEC** with `trace_p52_regime_handler = True` (preset = balanced,
   fl = 832, cng = True, `np.random.seed(42)` per case).
3. **Tally** boost_q / reverse_copy / main_paused fire frames from the
   per-frame trace dump.

## Aggregate statistics

### ERL_decile_std distribution (n = 800)

```
min        =  0.00 dB
median     =  2.30 dB
p90        =  8.58 dB     (cohort p90 in A.0R.3 thresholds was 9.43 dB
                            — this re-measurement gives 8.58; classifier
                            uses the locked 9.43 anchor)
p99        = 20.30 dB     (vs 21.04 dB anchor — locked)
max        = 32.03 dB
```

Histogram (40-column ASCII):

```
[ 0.00- 1.00)  n= 193  ########################################
[ 1.00- 2.00)  n= 173  ###################################
[ 2.00- 3.00)  n= 119  ########################
[ 3.00- 5.00)  n= 136  ############################
[ 5.00- 7.00)  n=  67  #############
[ 7.00- 9.43)  n=  46  #########         ← stable / mildly boundary at 9.43
[ 9.43-12.00)  n=  29  ######
[12.00-15.00)  n=  15  ###
[15.00-18.00)  n=  10  ##
[18.00-21.04)  n=   5  #                 ← mildly / wildly boundary at 21.04
[21.04-25.00)  n=   4
[25.00-30.00)  n=   1
[30.00-50.00)  n=   2
```

Long-tail distribution: 80 % of cohort is below 5 dB ERL_decile_std,
mass thins quickly past 9 dB, a handful of extreme outliers > 21 dB.

### Per-regime handler fire profile

| Regime | n | boost_q (mean / med / max) | reverse_copy (mean / med / max) | main_paused frames (mean / med / max) |
|---|---:|---|---|---|
| `wildly_nonstationary` | 7 | 3.14 / 0 / **22** | 4.29 / 3 / 11 | 22.71 / 0 / **159** |
| `mildly_nonstationary` | 59 | 0.37 / 0 / 17 | 3.88 / 3 / 17 | 3.37 / 0 / 188 |
| `stable` | 734 | 0.43 / 0 / 13 | 2.49 / 1 / 32 | 3.14 / 0 / 112 |

## Critical finding — handler fires concentrate on the *catastrophe* case, not on regime as such

Within the wildly cohort, **only the post-mortem target case
(`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`) has heavy intervention**:

| Case | std (dB) | boost_q | reverse_copy | main_paused frames |
|---|---:|---:|---:|---:|
| SgKY30fjT0G8e3kQL0RHSQ_doubletalk | 32.03 | 0 | 3 | **0** |
| LQhlYoXXiUevFuxMKwWB0Q_doubletalk | 31.13 | 0 | 2 | **0** |
| 0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement | 26.09 | 0 | 4 | **0** |
| P10GsQvhskKx3fB06Zv4Yg_farend_singletalk | 23.91 | 0 | 3 | **0** |
| s90M7MOTBkqaV4nQPLhKbA_doubletalk | 23.57 | 0 | 4 | **0** |
| **qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk** | **23.39** | **22** | **11** | **159** |
| LQhlYoXXiUevFuxMKwWB0Q_farend_singletalk | 21.93 | 0 | 3 | **0** |

6 of the 7 wildly cases have **zero** `boost_q` and **zero** `main_paused`
frames — only some `reverse_copy` (main→shadow sync; the gentler, non-
disruptive action). The post-mortem catastrophe case is the **single
outlier** even within the wildly cohort.

Reading: ERL_decile_std measures **echo path nonstationarity** but does
not by itself predict catastrophic main filter divergence. The target
case combines high nonstationarity **and** a specific failure trajectory
(W L2 grows but in the wrong direction) that triggers the
`boost_q + pause_main` escalation. The other 6 wildly cases have similar
acoustic nonstationarity but main filter happens to track without
diverging — only `reverse_copy` is needed to keep shadow in sync.

**Path 2 ROI implication (informational)**: a regime-aware intervention
arming only on wildly cases would, at best, affect 1 / 800 cases (the
catastrophe target) with measurable ERLE impact, not 7. The cohort tail
for this specific failure mode is even narrower than the regime
distribution suggests. Path 2 design (when / if it opens) must
explicitly justify why a regime trigger would catch the *catastrophe*
sub-population, not just the *nonstationarity* sub-population.

## §3a — Mildly_nonstationary cohort controller fire breakdown

Parallel data point to the wildly-cohort finding above: does the
controller fire heavily across the 59 mildly cases, or is the pattern
again "one outlier + bulk gentle / silent"?

### Aggregate (n = 59)

| Metric | Mean | Median | Max | Sum |
|---|---:|---:|---:|---:|
| `boost_q_count` | 0.37 | 0 | **17** | 22 |
| `reverse_copy_count` | 3.88 | 3 | 17 | 229 |
| `main_paused_frames` | 3.37 | 0 | **188** | 199 |

### Activity classes

| Class | n | Share |
|---|---:|---:|
| Zero handler fires (bq=0, rev=0, pause=0) | 14 | 23.7 % |
| Reverse-copy only (bq=0, pause=0, rev>0) | 41 | 69.5 % |
| **Any `boost_q` > 0 or `main_paused` > 0** | **4** | **6.8 %** |

55 of 59 cases (93 %) have zero `boost_q` and zero `main_paused` frames —
again, the heavy escalation machinery is silent. 41 cases incur only
`reverse_copy` (gentle shadow←main sync); 14 cases trigger nothing at
all (acoustic nonstationarity present per ERL_decile_std but the
controller sees no divergence to act on).

### The 4 mildly cases with heavy escalation

| Case | std (dB) | boost_q | reverse_copy | main_paused frames |
|---|---:|---:|---:|---:|
| `MkSLte0FTkqybGcLTwA3Tw_farend_singletalk_with_movement` | 12.98 | **17** | 6 | **188** |
| `pU21kfoo0UOz0fPMJFfydg_doubletalk` | 11.93 | 3 | 10 | 8 |
| `SUYzW4QT30yxKUq7OGvZKg_farend_singletalk` | 10.08 | 1 | 7 | 2 |
| `tl5UFRCXZkyL6EoWVl09xA_doubletalk` | 9.59 | 1 | 0 | 1 |

The top mildly case (`MkSLte0F…`, std = 12.98 dB) has `boost_q = 17` and
`main_paused = 188` — comparable in magnitude to the wildly catastrophe
target (`qNvSMyU…`, std = 23.39, bq = 22, pause = 159). Yet its
ERL_decile_std (12.98) places it well inside the mildly band (9.43 –
21.04), not in the wildly tail. This is direct evidence that the
catastrophe-defence trigger **does not align with the
ERL_decile_std-defined regime label**: a "merely mildly" case can
require the same heavy intervention as a wildly case, and 6 of 7 wildly
cases need only gentle reverse_copy.

### Cohort-wide consolidation

Combining the wildly (§3) and mildly (§3a) findings across the 66
"non-stable" cases (7 + 59):

| Pattern | Wildly (n=7) | Mildly (n=59) | Combined |
|---|---:|---:|---:|
| Heavy escalation (`boost_q > 0` or `main_paused > 0`) | 1 | 4 | **5 / 66** (7.6 %) |
| Reverse-copy only | 6 | 41 | 47 / 66 (71 %) |
| Zero fires | 0 | 14 | 14 / 66 (21 %) |

Across the full 800-case cohort the catastrophic-defence machinery
(`boost_q + main_paused`) fires meaningfully on **5 non-stable cases
(1 wildly + 4 mildly) plus at least 2 stable-cohort cases**
(`QEeKiaN…` bq=5 pause=40 and `nyT6FUU…` bq=3 pause=25, see §4 below)
— roughly **7 cases out of 800** (0.9 %). Reverse_copy (gentle sync)
fires on tens of cases across all three regimes. The remaining
~93–94 % of the cohort either is silent or uses only a handful of
frames of light activity.

### Path 2 ROI implication (updated)

The A.0R.7 wildly-only finding suggested Path 2 ROI ≈ 1 / 800. The
mildly-cohort data extends this: there is **no clean way to arm a
Path 2 regime-triggered intervention** using ERL_decile_std alone:

- A "wildly-only" trigger would fire on 7 cases but only 1 needs the
  heavy machinery (precision 1/7 ≈ 14 %; recall 1/5 = 20 %).
- A "wildly ∪ mildly" trigger would fire on 66 cases for 5 of ~7
  catastrophe-candidates (precision 5/66 ≈ 7.6 %; recall 5/7 ≈ 71 %;
  misses the 2 stable-cohort heavy-fire cases entirely).
- A "wildly ∪ mildly ∪ stable-with-heavy-fires" trigger would catch all
  ~7 — but the stable-cohort criterion is itself the existing
  controller's `boost_q + main_paused` decision; this is circular and
  Path 2 would contribute nothing new.

Reading: **the controller's frame-level escalation decision is the
discriminator that works**, not a precomputed acoustic regime label.
A Path 2 redesign that arms on regime classification alone would either
over-trigger (66 cases for a 5-case need) or under-trigger (7-case
wildly catches only 1 of 5 catastrophe candidates). This is
informational only — Path 2 remains explicitly out of v1.1 scope per
[p52_phase_a_verdict.md](p52_phase_a_verdict.md); when / if Path 2
opens, the anchor must be a frame-level divergence signal, not a
pre-AEC acoustic regime label.

## Full case enumeration

### `wildly_nonstationary` (n = 7)

```
SgKY30fjT0G8e3kQL0RHSQ_doubletalk                                  std=32.03
LQhlYoXXiUevFuxMKwWB0Q_doubletalk                                  std=31.13
0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement             std=26.09
P10GsQvhskKx3fB06Zv4Yg_farend_singletalk                           std=23.91
s90M7MOTBkqaV4nQPLhKbA_doubletalk                                  std=23.57
qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk         (postmortem target) std=23.39
LQhlYoXXiUevFuxMKwWB0Q_farend_singletalk                           std=21.93
```

### `mildly_nonstationary` (n = 59, top-15 by std)

```
LHsrJBRGnUKiMC2mihEr0g_doubletalk_with_movement      std=20.42
vSZmpMJI0kKv30P2GhgV1Q_farend_singletalk              std=20.30
kOtW70qgikKm0F9OEQw22A_farend_singletalk              std=19.22
veoTpvS3mkaNkmCI6iEMVA_farend_singletalk              std=18.33
TZ6TJFCbfkKAVrS64Sf08Q_doubletalk                     std=18.05
xNr7L0xsLUG4B9oUqW0V4Q_doubletalk_with_movement       std=17.29
mNqvVawOEUSLrFsSVX0xYg_nearend_singletalk             std=17.16
nV9v63E5CUKtKTjha8dtdQ_farend_singletalk_with_movement std=17.10
WqEYNwalSUebZxaeYVay2g_doubletalk_with_movement       std=17.06
vfTQx88jikCT7BocQ0Hgyw_nearend_singletalk             std=16.36
49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement       std=16.34
BYRb7rMHZUOVwHO90KRg9Q_farend_singletalk_with_movement std=16.27
SgKY30fjT0G8e3kQL0RHSQ_doubletalk_with_movement       std=15.99
49IIo03GZ0CYQOmeA3A0BA_doubletalk                     std=15.77
49IIo03GZ0CYQOmeA3A0BA_farend_singletalk              std=15.10
... (44 more cases between std=15.10 and std=9.43; full list in
     /tmp/p52_a0r7/dist.json)
```

### `stable` (n = 734)

Listing the top-5 (closest to the `stable`/`mildly` boundary at 9.43 dB):

```
QYHsugUAcUWEQ9WghnG0Jw_farend_singletalk              std=9.43  bq=0
QEeKiaNiDECfqXTRrDFWWw_farend_singletalk              std=9.41  bq=5  pause=40
ksP3OuSnpUa9Si2ttiUSoA_farend_singletalk              std=9.38  bq=0
nyT6FUUdu0W8UpvjP1rRgQ_farend_singletalk_with_movement std=9.36  bq=3  pause=25
sUQrHEPAoEmIvHclpi1tRQ_farend_singletalk              std=9.34  bq=0
```

Two stable-cohort cases also incur `boost_q` + `main_paused` activity
(QEeKiaN…, nyT6FUU…) — further evidence that the controller is not
strictly correlated with regime label. Full stable list in
`/tmp/p52_a0r7/dist.json` (734 entries).

## Sanity check on the classifier behaviour

| Item | Result |
|---|---|
| 800 / 800 cases produce a valid `erl_decile_std_db` (no NaN / Inf) | ✓ |
| `regime` always one of `{stable, mildly_nonstationary, wildly_nonstationary}` | ✓ |
| Sum of regime counts = 800 | ✓ |
| Threshold logic matches A.0R.3 source: `< 9.43` → stable; `[9.43, 21.04)` → mildly; `≥ 21.04` → wildly | ✓ |
| Target case classified as `wildly_nonstationary` with std = 23.39 (matches post-mortem manual measurement) | ✓ |
| All `boost_q_count` / `reverse_copy_count` / `main_paused_frames` are non-negative ints | ✓ |
| Target case fire counts match Step 1 post-mortem trace: `boost_q=22`, `reverse_copy=11`, `main_paused=159` | ✓ |

No classifier bug surfaced. Thresholds remain locked at A.0R.3 commit `401e62b`.

## Anti-loophole

- Classifier thresholds not modified.
- No production code touched by A.0R.7 (analysis-only).
- Classifier output not wired to any production decision (still
  guarded by `python/test_p52_regime.py::AntiLoopholeTests`).

## Disposition

Outcome A satisfied. Proceed to **A.0R.8 runtime sanity**; then re-verify
A.0R.6 byte-equal on current HEAD and merge `feature/p52-phase-a-shadow`
to `main`.

## Cross-references

- Classifier source: [python/aec_p52_regime_classifier.py](../python/aec_p52_regime_classifier.py)
- Classifier tests: [python/test_p52_regime.py](../python/test_p52_regime.py)
- Distribution driver: [tools/research/p52_a0r7_regime_distribution.py](../tools/research/p52_a0r7_regime_distribution.py)
- Raw per-case JSON: `/tmp/p52_a0r7/dist.json`
- Post-mortem: [p52_a0_postmortem.md](p52_a0_postmortem.md)
- Phase A verdict: [p52_phase_a_verdict.md](p52_phase_a_verdict.md)
