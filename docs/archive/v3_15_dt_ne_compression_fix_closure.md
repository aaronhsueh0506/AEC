# v3.15 §1.2 — DT-NE compression fix CLOSED (CANNOT SHIP)

**Date**: 2026-05-15
**Branch**: `feature/v3.15`
**Sprint**: §1.2.S4+S5 — combined verdict + closure
**Implementation preserved**: commit `6b279bb` (substrate, default OFF, byte-equal flag-OFF PASS)

## Result summary

Two configurations bench'd against `results/v3_14_baseline/scores.json`
(production `main` HEAD `b3273de`):

### Variant A — full §1.2 (per-state scale 2.0 in pre-conv states + per-bin override 2.0)

| bucket | n | mean Δecho | mean Δdeg | worst Δecho | worst Δdeg |
|---|---:|---:|---:|---:|---:|
| DT_static | 186 | -0.0607 | **+0.1275** | -0.628 | -0.157 |
| DT_movement | 114 | -0.0739 | **+0.1397** | -0.561 | -0.142 |
| FS_static | 169 | **-0.2002** | +0.0001 | -1.049 | -0.001 |
| FS_movement | 131 | **-0.1558** | +0.0001 | -0.874 | -0.001 |
| NE | 200 | -0.0000 | +0.0077 | -0.002 | -0.044 |
| Cohort tail qNvSMyU | — | **-0.3952** | -0.000 | — | — |

### Variant B — per-bin override only (state_scale = 1.0 everywhere)

| bucket | n | mean Δecho | mean Δdeg | worst Δecho | worst Δdeg |
|---|---:|---:|---:|---:|---:|
| DT_static | 186 | -0.0350 | **+0.0647** | -0.372 | -0.154 |
| DT_movement | 114 | -0.0389 | **+0.0692** | -0.331 | -0.277 |
| FS_static | 169 | **-0.1059** | +0.0001 | -0.763 | -0.001 |
| FS_movement | 131 | **-0.0765** | +0.0001 | -0.529 | -0.000 |
| NE | 200 | -0.0000 | +0.0049 | -0.002 | -0.028 |
| Cohort tail qNvSMyU | — | **-0.2750** | -0.000 | — | — |

### Hard bar check (both fail)

| Bar | Limit | Variant A | Variant B | Verdict |
|---|---:|---:|---:|---|
| DT combined Δdeg | ≥ +0.020 | **+0.134** ✓ | **+0.067** ✓ | both PASS |
| FS_static Δecho | ≥ -0.020 | -0.200 ✗ | -0.106 ✗ | both FAIL (10× / 5× over) |
| FS_movement Δecho | ≥ -0.020 | -0.156 ✗ | -0.077 ✗ | both FAIL (7.8× / 3.8× over) |
| Cohort tail qNvSMyU Δecho | ≥ -0.050 | -0.395 ✗ | -0.275 ✗ | both FAIL (7.9× / 5.5× over) |
| NE Δdeg | ≥ -0.005 | +0.008 ✓ | +0.005 ✓ | both PASS |

## Mechanism analysis: same FS-vs-DT wall as v3.13 E5

Both variants deliver DT-NE preservation gains as predicted by §1.1
audit (+0.067 to +0.134 dB Δdeg, well above hard bar). But the same
gate-relaxation mechanism that preserves NE on unconverged-state
NE-evidence bins **also passes echo through on those same bins**.

The information-theoretic reason: in `coarse_learning` state, the
linear filter's `residual_echo_psd` is inflated (poor echo estimate).
F3.1-v3 mic-excess (`dt_per_bin`) computes "mic exceeds expected echo"
— but "expected echo" is itself inflated, so `dt_per_bin > 0.5` fires
on **both** NE-only bins **AND** "NE+echo" bins. The mechanism cannot
selectively spare NE without also sparing echo, because **the AEC
literally cannot distinguish them at the per-bin level when the
filter is unconverged**.

**Slope**: Variant A: +0.134 DT / -0.200 FS_static = ratio 0.67. Variant
B: +0.067 DT / -0.106 FS_static = ratio 0.63. v3.13 E5 closure ratio
was ~2.0 (DT loss per FS gain in opposite direction). v3.15 §1.2 angle
is more favourable, but the wall is still there.

## Comparison with v3.13 E5 closure (same family)

v3.13 E5 closure verdict (`docs/v3_13_e5_closure.md`):
> "Filter-protection mechanism is fundamentally trade-off-bound;
> v3.14 Volterra needed."

v3.15 §1.2 is the SAME class of mechanism (RES-side gate relaxation
to address compression in unconverged states) hitting the SAME wall
from the opposite direction.

## Root cause is upstream

Per `feedback_no_shortcut_use_canonical`: canonical solution is to
**fix the linear filter convergence** so unconverged-state behavior
becomes rare, not to relax the RES gate to mask the symptom.

The §1.1 audit found worst-DT cases stuck in `coarse_learning`
40-99% of frames. If the filter converged into `refined_usable`
faster:
- `residual_echo_psd` becomes accurate
- `dt_per_bin` (F3.1-v3 mic-excess) becomes reliable at distinguishing
  NE from echo
- ENR gate behaves correctly (suppresses echo, preserves NE)
- DT-NE compression problem self-resolves

**Canonical mechanisms** (already in v3.15 plan, Phase E):
- §1.4 Arc M: movement-aware adaptive Q schedule (faster Q-boost
  in DT during dynamic paths)
- §1.4 Arc G: gain-change per-band W reset (faster recovery from
  mic-gain shifts that prevent convergence)
- §1.6 Arc F: per-band Q/R Kalman schedule (faster HF convergence
  while keeping LF stable)

Combined, these target the **convergence speed** that determines
how often the filter sits in `coarse_learning`. If pre-convergence
fraction drops from current 40-99% to e.g. 10-30% on the DT cohort,
DT-NE compression should self-resolve without needing §1.2.

## Disposition

**§1.2 CLOSED CANNOT SHIP.** Same family as v3.13 E5.

**Substrate retained**: `dt_ne_compression_fix` flag + per-state scale
+ per-bin override implementation kept as default-OFF research substrate.
Reusable if a future arc adds state confidence that lets us gate the
mechanism on "filter is well-converged AND NE evidence reliable" — but
that's exactly the condition where the legacy gate already works.

**Plan revision (autonomous decision per user §1.2 directive)**:
- §1.3 Arc D merge: also DEFERRED. Arc D's per-state ENR plumbing is
  the same family (per-state gate relaxation). The audit + this
  closure both confirm RES-side per-state gate-relax cannot escape
  the FS-vs-DT wall.
- Skip directly to Phase E (canonical convergence arcs): §1.6 Arc F,
  §1.4 Arc M+G, §1.5 Arc T.
- After Phase E ships, re-measure DT-NE compression. If residual
  debt remains, revisit §1.2 with state-conditional gating
  (per-bin override ONLY in `refined_usable` where dt_per_bin
  is trustworthy).

## Audit invariants retained

§1.1 audit findings remain load-bearing for v3.15 sprint design:
- DT cases stuck in `coarse_learning` 40-99% — supports Arc M / Arc G
  rationale
- DTD never fires on worst-DT cohort — supports Arc T preemption
- F3.1-v3 mic-excess works when filter converged — supports keeping
  current per-bin evidence pipeline (don't redesign mic-excess)
- 4-cap chain is roughly neutral — supports §1.7 candidate to retire
  `epc_dt_cap` dead code

## v3.13 E2 Path 3 debt status

v3.13 E2 Path 3 DT debt (-0.050 DT_static / -0.025 DT_movement)
**remains unrecovered** by RES-side mechanism. Closure deferred to
post-Phase E re-measurement. If Arc M/F/G drives `coarse_learning`
frame fraction down, debt closes by reduction of pre-conv frames
(not by RES policy change).

§0.5 invariant debt updated: "v3.13 E2 Path 3 DT debt → target:
Phase E (Arc M+F+G) convergence improvement, NOT §1.2 RES policy".

## Files committed

- `python/aec.py` — `dt_ne_compression_fix` flag, per-state scale dict,
  per-bin override. Default OFF byte-equal sanity 5/5 PASS.
- `python/eval_aec_challenge.py` — env overrides
  `AEC_DT_NE_COMPRESSION_FIX`, `AEC_DT_NE_STATE_SCALE`,
  `AEC_DT_NE_PER_BIN_{THRESH,SCALE}`.
- `docs/v3_15_dt_ne_audit.md` — §1.1 audit verdict (H2 confirmed).
- `docs/v3_15_dt_ne_compression_fix_closure.md` — this doc.

## Next

Proceed directly to Phase D (B3 + B6 quick bug fixes) and Phase E
(canonical convergence arcs §1.6 Arc F, §1.4 Arc M+G, §1.5 Arc T).
