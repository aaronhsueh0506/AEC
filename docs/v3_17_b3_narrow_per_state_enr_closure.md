# v3.17 Phase B.3 — C2.v2 narrow per-state ENR CLOSED CANNOT SHIP (2026-05-15)

**Branch**: `feature/v3.17` HEAD `182cf89`.
**Sprint**: Phase B.3 (per [`docs/v3_17_plan.md`](v3_17_plan.md) §B.3).
**Verdict**: CLOSE per §0.4 — narrow scope mechanism barely fires;
DT bucket Δdeg = +0.001 (vs hard bar +0.015). Structural confirmation
of v3.15 §1.1 H1 hypothesis (DT cases dominantly in `refined_usable`).

---

## 1. Mechanism tested

Used existing v3.15 §1.2 substrate (`dt_ne_compression_fix` flag,
`dt_ne_state_scale` dict, `dt_ne_per_bin_scale`) with **narrow** config:
- `suspicious_dt = 1.5` (only state with non-1.0 scale)
- All others (idle/startup/coarse_learning/refined_usable/diverged) = 1.0
- `dt_ne_per_bin_scale = 1.0` (per-bin override disabled)

Hypothesis (per v3.17 plan §B.3): unlike v3.15 §1.2's wide scope (which
broke FS by relaxing gate in `coarse_learning` where filter is
unconverged), `suspicious_dt` state has filter mu reduced — output is
trustworthy enough that ENR relax doesn't pass spurious echo.

## 2. 60-case AECMOS bench result

Δ vs v3.16.0 baseline (`/tmp/v3_17_b1_off_scores/scores.json`):

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.001 | +0.000 |
| FS_movement | −0.007 | −0.000 |
| DT_static | −0.001 | **+0.001** |
| DT_movement | +0.000 | +0.000 |
| NE | +0.000 | +0.000 |

15 wav files differed (8 cases / 60). Mechanism barely fires.

Hard bar check:
- DT bucket Δdeg ≥ +0.015 → **FAIL** (+0.001 actual, 15× short)
- All other guards within bar (no harm done — but no benefit either)

## 3. Root cause: confirms v3.15 §1.1 H1 hypothesis

v3.15 §1.1 DT-NE compression audit identified H1 hypothesis: "DT cases
mostly in `refined_usable` state (where Arc D / per-state ENR is
byte-equal locked)". B.3 narrow result CONFIRMS this — by restricting
the lever to `suspicious_dt`, we observed that the state simply
doesn't fire often enough during DT cases to move the bucket means.

This is the inverse confirmation of v3.15 §1.2 closure:
- Wide (§1.2): scaling `coarse_learning` (which fires often during DT
  due to convergence churn) → breaks FS
- Narrow (B.3): scaling only `suspicious_dt` (which protects against
  DT-on-NE confusion but rarely fires standalone) → no DT recovery

The v3.13 E2 Path 3 DT debt (DT_static −0.050 / DT_movement −0.025)
is therefore confirmed STRUCTURALLY unrecoverable through RES-side
ENR gate relaxation alone. Same family verdict as v3.13 E5 closure.

## 4. Per §0.4 verdict

> "If lever moves bucket-mean < 0.002 dB AND fires < 5% of frames
> after 3 A/B sweeps, ship as substrate for dependent arc"

DT bucket lever moves +0.001 dB (under 0.002 dB threshold). Mechanism
fires but barely affects output (state rarely active). Substrate
already exists (commit `6b279bb` from v3.15 §1.2). No new code needed
to ship — the existing flag + state_scale dict can be tuned by
downstream users wanting different state weighting.

**Action**: no new code, no commit. Document closure as v3.17 Phase B.3
verdict. Existing substrate (default-OFF flag) preserved.

## 5. v3.18+ canonical path (per v3.15 §1.2 closure)

> "Per `feedback_no_shortcut_use_canonical`: canonical solution is to
> **fix the linear filter convergence** so unconverged-state behavior
> becomes rare, not to relax the RES gate to mask the symptom."

v3.13 E2 Path 3 DT debt persists. Recovery requires linear-filter
convergence improvement (faster Q tuning, better delay tracking, or
new state classification). All 4 v3.15 attempts (Arc M/G/F/T family)
hit walls. v3.17 B.1 (movement-rate DelayEst) hit a wall. The DT-NE
debt closure target moves to v3.18+ with a fundamentally new mechanism.

## 6. Next: Phase B.2

Phase B.2 (C9.v2 reverb-aware multi-feature trigger, pcb1N target) is
next. B.2 is independent of B.3 mechanism (RES-side override on a
different trigger), so B.3 closure does NOT block B.2.
