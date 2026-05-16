# v3.13 E2.S4 verdict — movement persistence classifier prototype

**Status**: CLOSED INFORMATIVE. Classifier is a useful diagnostic but
**not a deployment gate**. Path 3 (already justified by E2.S2) and the
classifier surface the same underlying defect (bench pre-align failure
on long-delay cases). v0 thresholds saved for E2.S5 as a triage signal.

**Date**: 2026-05-13

## Scope

Per v3.13 plan E2.S4: produce a per-case movement-persistence
classifier from the online `DelayEstimator._trace_rows` (audit-only,
no AEC behaviour change). Test on the 8-case listen set and on the
full FS_static (n=169) + FS_movement (n=131) buckets.

Classification (v0 thresholds, under production max=250ms GCC
pre-align):

| Class | Rule |
|---|---|
| STATIC | online-trace delay std < 30 samp |
| INTERMITTENT_STEP | std≥30 + 1-2 big jumps + ≥80% within ±50 samp of median |
| PERSISTENT_MOVEMENT | std≥100 (spread) OR many big jumps with low clustering |
| UNCERTAIN | solid_fraction < 20% OR mean PAR < 3 |

## Headline distribution

| Bucket | n | STATIC | INTERMITTENT | PERSISTENT | UNCERTAIN |
|---|---:|---:|---:|---:|---:|
| FS_static | 169 | 150 (89%) | 6 (4%) | 13 (8%) | 0 |
| FS_movement | 131 | 110 (84%) | 3 (2%) | 18 (14%) | 0 |

**Finding 1**: FS_movement is only modestly more "movement-dominated"
than FS_static under the production pre-align. 84% of FS_movement
cases classify STATIC at the online tracker once GCC has locked.
The bucket naming reflects how the dataset was generated (simulated
movement injected), not what the AEC actually sees in its
within-session delay trace.

**Finding 2**: The 18 FS_movement PERSISTENT cases correlate strongly
with bench GCC pre-align failure:

| Stem (top 9 by std) | GCC initial delay (samp) | std (samp) | range (samp) |
|---|---:|---:|---:|
| `V0JqgjlrB0Ke9y91r0rxNw` | 798 | 6140 | 13298 |
| `OwkV685H2Em8jjTOAbhbow` | 1890 | 3845 | 12987 |
| `sRCs6SKo6kC0xire475q0A` | 2455 | 3764 | 11513 |
| `KgCF0xsN8EibTdJ2Yac2dw` | 10 | 3221 | 10745 |
| `khqZY41lNEyIvMf2ZNJuVA` | 13 | 2510 | 8306 |
| `xFk7igecuke0R5JMfREyDg` | 7 | 2378 | 5774 |
| `SgKY30fjT0G8e3kQL0RHSQ` | 1 | 2023 | 6789 |
| `lxLsvT1rY0mdtZuRogM06Q` | 1119 | 1964 | 9284 |
| `0I0XMl3M0ECO0U1N0cJvpg` | 2 | 1720 | 4367 |

8 of these 9 cases have GCC initial delay ≤ 2455 samp (≤ 153 ms) but
within-session range up to 13,298 samp (832 ms). This means the bench
pre-align lands on one peak, then the online tracker is chasing a
much-larger residual delay — consistent with the E2.S2 finding that
extending bench pre-align to max=1024 (Path 3) unlocks +13.5 dB on
the worst case (S22FC).

## Cross-reference with listen 8-case

Re-classified using the existing E2.S1 audit JSON (same threshold v0,
same max=250 GCC):

| # | Stem | GCC delay | class | std | listen note |
|---|---|---:|---|---:|---|
| 01 | 7GTxy_FS_static | 415 | STATIC | 11.1 | delay ~10000 samp + clip |
| 02 | IrQvq_FS_static | 1475 | STATIC | 0.2 | delay ~1200 + clip + radio |
| 03 | pcb1N_FS_static | 2523 | STATIC | 5.9 | mic quality (E1) |
| 04 | S22FC_FS_static | 0 | STATIC | 8.0 | delay ~9800 samp |
| 05 | hVqUm_FS_static | 3006 | STATIC | 29.5 | delay ~6000 samp |
| 06 | 5bJUo_FS_mvmt | 1701 | STATIC | 1.9 | delay ~1800 + clip |
| 07 | IrQvq_FS_mvmt | 644 | STATIC | 3.0 | NL distortion |
| 08 | S22FC_FS_mvmt | 3056 | INTERMITTENT_STEP | 99.6 | delay ~10000 + sat |

**Finding 3**: 7 of 8 listen cases classify STATIC, including the
4 known-delay-broken ones (01/04/05/06). The classifier is misled by
GCC pre-align failure — when the bench pre-align lands on a wrong
peak, the online tracker can still lock to that wrong peak stably,
producing low std. **STATIC under max=250 ≠ correctly aligned**.

This is the classifier's fundamental limitation under production
pre-align: it captures tracker stability, not delay correctness.

## Mechanism

The online F-DelayTrack runs *after* the bench GCC pre-align. Its
search range (max_delay_ms=1024 by default) is large, but its update
rate is conservative — it only reports a new estimate every
`delay_est_period_s` (production = 0.25s) and requires solid PAR. If
the bench pre-align is wrong by >1024 ms, the online tracker can't
find the right peak either (out of its window after re-alignment).

So under production pre-align, the classifier sees three distinct
populations:

| Population | Trace appearance | Class |
|---|---|---|
| Correctly aligned, no movement | std≈0, high PAR | STATIC ✓ |
| Correctly aligned, real movement | std spreads over time | PERSISTENT ✓ |
| Pre-align broken, locked to wrong peak | std≈0, high PAR | STATIC ✗ (false negative) |

The classifier reliably catches the second column (correctly-aligned
real-movement) but cannot distinguish columns 1 and 3 without
external evidence (e.g., linear ERLE).

## What's deployable / not deployable

**Deployable now**:
- The classifier output as a **diagnostic surface** (zero behaviour
  change). Cases classified PERSISTENT under default settings are
  flagged for offline review — useful for cohort tail catastrophe
  monitoring even without acting on it.

**NOT deployable as a gate**:
- Cannot use STATIC vs PERSISTENT as a gate to conditionally trigger
  aggressive online tracking, because the most-broken cases (listen
  01/04/05/06) all classify STATIC due to bench pre-align failure
  locking the tracker.

**Path forward — re-classify under Path 3 pre-align**:
- Re-run the classifier with `AEC_MAX_DELAY_MS=1024` (env override
  added in commit `1e14c0f`). Under Path 3 pre-align, the bench
  aligns within 1024 ms — listen 01/04/05/06 should then either
  classify STATIC (correctly aligned, no remaining movement) or
  PERSISTENT (correctly aligned, real movement). This *delta* between
  max=250 and max=1024 classifications is the deployable signal:
  cases that switch class are confirmed pre-align-broken; cases that
  stay PERSISTENT have genuine movement.

That delta-classification is logical but **defer to E2.S5**, where
it can directly inform the conditional Path 3 deployment criteria
(if any subset benefits from staying at max=250 to avoid DT
regression unmasking).

## Cross-arc consistency check

Triple-confirmation across the E2 sub-arcs of the same defect (bench
pre-align too narrow on subset of long-delay cases):

| Sub-arc | Evidence | Signature |
|---|---|---|
| E2.S2 Path 3 | Linear ERLE on 7 DT regressors +1.14 to +3.61 dB | Filter sees real delay |
| E2.S3 period scan | Flat (Δ<0.003 dB any period) | Once aligned, no live drift |
| E2.S4 classifier | 18 FS_movement PERSISTENT cases have GCC delay 1-2455 samp + range up to 13298 | Online tracker visible range > pre-align landing |

All three arcs detect the same subset of cases through different
metrics. The defect is structural (bench harness, fixable in one
line) — not an algorithmic deficit in F-DelayTrack itself.

## Threshold sensitivity (informational)

The v0 thresholds (std=30 / 100 samp) are calibrated to make the
classifier produce useful three-class output on this dataset. Lower
static threshold (e.g., 10 samp) would push more cases into
INTERMITTENT/PERSISTENT; higher (e.g., 50 samp) would consolidate
more into STATIC. The current values match the audible threshold
(~2-6 ms range; one PBFDKF tap = 1/16000 s ≈ 63 µs, so 30 samp ≈ 2 ms
relative drift below filter resolution).

## Artifacts

- Renderer: [tools/research/e2_s4_movement_classifier.py](../tools/research/e2_s4_movement_classifier.py)
- Output: `results/v3_13_e2_s4_fs_full/summary.md` + `results.json` (gitignored)
- E2.S1 audit JSON reused for listen 8-case replay

## Verification rules followed

1. Production max=250 GCC pre-align (matches bench)
2. seed=0, fl=832, BALANCED preset (matches bench)
3. `trace_delay_est=True` for diagnostic only (zero-impact on AEC output)
4. Full FS_static + FS_movement (n=300) for distribution

## Plan implications

- E2.S5 deploy A/B: classifier output → diagnostic only; deploy Path 3
  unconditionally (per E2.S2 net positive) with DT regression accepted
  to v3.14+ RES re-eval
- v3.13.X (dual filter sub-arc): consider adding classifier output to
  AecStats / regime trace as a permanent diagnostic surface
- v3.14+: re-evaluate classifier under Path 3 pre-align; if delta-
  classification (max=250 vs max=1024) gives clean signal, that's a
  real deployment gate candidate
