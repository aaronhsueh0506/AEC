# v3.21 Q1 Phase 2 — 9-case AECMOS Quick Check

**Candidate:** `use_q1_true_delay_transient_leakage` (A=OFF vs B=ON)
**Config:** BALANCED + enable_delay_est=True + enable_cng=True; no pre-alignment
**Cohort:** 9 cases (4×DT_mvmt + 3×FS_mvmt + 1×DT_static + 1×FS_static)
**Phase 1 prerequisite:** PASS (leakage + H_error + smoothing + restore all mechanically correct)

---

## Overall Verdict: FAIL — do not proceed to 12-case

| Gate | Result | Detail |
|---|---|---|
| Gate 1: No DT_mvmt case regression > 0.05 dB | **FAIL** | 2/4 cases regress: Hp5g1asacU −0.951, XRTnTUjU −0.163 |
| Gate 2: No catastrophic outlier \|Δ\| > 0.20 | **FAIL** | 4 cases flagged (see table below) |
| Gate 3: nVUnxqHLr EPV suppression verdict | **BENEFICIAL** | Δdeg = +0.329 (false-alarm EPV suppression) |

---

## Bucket Means

Primary metric: DT buckets → deg (higher = better); FS buckets → echo (lower = better).

| Bucket | N | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg | Primary Δ |
|---|---|---|---|---|---|---|---|---|
| DT_mvmt | 4 | 3.637 | 3.698 | +0.061 | 3.173 | 2.898 | −0.275 | **deg=−0.275** |
| DT_static | 1 | 4.454 | 4.517 | +0.063 | 2.446 | 2.775 | +0.329 | deg=+0.329 |
| FS_mvmt | 3 | 3.812 | 3.955 | +0.143 | 5.000 | 4.999 | −0.000 | **echo=+0.143** |
| FS_static | 1 | 2.892 | 3.325 | +0.433 | 5.000 | 5.000 | −0.000 | **echo=+0.433** |

---

## Per-case Scores

| Label | Case | A_echo | B_echo | Δecho | A_deg | B_deg | Δdeg | Notes |
|---|---|---|---|---|---|---|---|---|
| DT_mvmt | 49IIo03GZ0 | 4.330 | 4.417 | +0.087 | 2.817 | 2.831 | +0.014 | ✓ |
| DT_mvmt | 7GTxyTksSU | 2.912 | 2.912 | +0.000 | 3.754 | 3.754 | +0.000 | ✓ (no events) |
| DT_mvmt | Hp5g1asacU | 4.360 | 4.457 | +0.097 | 2.785 | 1.833 | **−0.951** | **FAIL Gate 1+2** |
| DT_mvmt | XRTnTUjU5k | 2.947 | 3.007 | +0.061 | 3.337 | 3.174 | **−0.163** | **FAIL Gate 1+2** |
| FS_mvmt | 0I0XMl3M0E | 4.462 | 4.446 | −0.016 | 4.999 | 4.999 | −0.000 | ✓ |
| FS_mvmt | Fi80N5kW9U | 3.759 | 4.077 | **+0.318** | 5.000 | 4.999 | −0.000 | FAIL Gate 2 |
| FS_mvmt | wlAXM0iDgk | 3.215 | 3.342 | +0.126 | 5.000 | 5.000 | −0.000 | watch |
| DT_static | nVUnxqHLr0 | 4.454 | 4.517 | +0.063 | 2.446 | 2.775 | **+0.329** | EPV beneficial |
| FS_static | 9xjhiFbGo0 | 2.892 | 3.325 | **+0.433** | 5.000 | 5.000 | −0.000 | FAIL Gate 2 |

---

## Root Cause Analysis — Why Phase 2 FAILS

### Mechanism 1: DT_mvmt catastrophic regression (Hp5g1asacU, Δdeg = −0.951)

**Phase 1 trace:** delay_first at frame 90 (t=0.90 s). EPV fires at frame 372 (t=3.72 s) in **both** OFF and ON runs.

**What happens in the ON run:**

Q1 arms at frame 90 → elevated lc = 0.0125 for 250 hops, then linear smooth 100 hops.
At EPV frame 372: Q1 counter rem = 350 − (372 − 90) = **68** (already in smoothing zone).
lc at frame 372 ≈ (68/100)×0.0125 + (32/100)×0.0025 = **0.0093** (still 3.7× elevated).

When EPV fires:
- Filter reset runs → W/P/R cleared
- H_error resets to 10000 (via `handle_echo_path_change`)
- Q1 counter continues from rem=68, elevated lc continues for 68 more hops

After EPV in ON run: filter starts from W=0, H_error=10000, AND lc=0.0093 (elevated).
The Kalman gain is proportional to P/(P + σ²) and H_error is in the numerator of the leakage injection. With BOTH H_error=10000 AND lc=0.0093, the effective Kalman gain is enormously amplified vs the OFF run (where EPV fires with steady lc=0.0025). This drives the filter to over-adapt aggressively toward the post-EPV input — if that input contains DT speech, the filter mistakes NE speech for echo path change → DT speech suppression → deg catastrophically damaged.

**OFF run:** EPV fires at frame 372 with normal lc=0.0025 → H_error=10000 but Kalman gain is moderated by steady leakage → filter re-adapts more carefully → better DT preservation.

The damage is not the EPV itself; it is **EPV + elevated Q1 leakage combining multiplicatively**.

### Mechanism 2: DT_mvmt moderate regression (XRTnTUjU, Δdeg = −0.163)

**Phase 1 trace:** delay_first at frame 154 (t=1.54 s). No EPV in this case. Q1 active from frame 154 to 504 (3.5 s).

At hop +11 (frame 165): H_on = 0.8965 vs H_off = 0.2653 → ON run Kalman gain **3.4× higher**.
This elevated gain persists for the full 350-hop Q1 window. During DT speech that falls in this window, the filter adapts more aggressively toward near-end-contaminated observations → partial NE suppression → deg −0.163.

XRTnTUjU is the less severe case because no EPV interaction amplifies the damage.

### Mechanism 3: FS regression via filter de-convergence (Fi80N5kW9U Δecho = +0.318; 9xjhiFbGo0 Δecho = +0.433)

**Phase 1 traces:**
- Fi80N5kW9U: delay_first at frame 103 (FS_mvmt). Q1 active 103–453 (3.5 s).
- 9xjhiFbGo0: delay_first at frame 69 (FS_static). Q1 active 69–419 (3.5 s).

Both are FS (no near-end speech). Why does elevated leakage harm FS echo cancellation?

PBFDKF convergence direction: W is updated toward minimizing E. After a real delay change, W needs to shift to track the new echo path. The elevated leakage makes this faster — correct in principle. BUT at lc=5×, the H_error per-bin grows too fast → per-hop Kalman gain shoots above its stable operating point → filter **overshoots** the new echo path minimum and begins oscillating → steady-state E is higher than it would be under normal steady leakage → more residual echo.

AEC3 calibration note: The transient parameters (LC_TRANSIENT = 1.25e-2, LD_TRANSIENT = 1.25) were ported from AEC3's `refined_initial` values, which are calibrated for a purely NLMS-style (partition-by-partition) shadow filter with bounded gain. In PBFDKF, the Kalman gain computation already incorporates full state information: P, X²-sum, H_error. When H_error is elevated 5× via elevated lc injection, the effective Kalman step can become unstable (P→∞ trajectory). The 350-hop duration (3.5 s) is far longer than the NLMS equivalent needs.

---

## nVUnxqHLr DT_static — EPV Suppression Analysis

**Phase 1 finding:** OFF run EPV fires at t=4.26 s (frame 426). ON run: no EPV; shadow_rise at t=11.11 s.

**Phase 2 verdict: BENEFICIAL** — ON deg = 2.775 vs OFF deg = 2.446; Δdeg = **+0.329**.

In this case, the OFF-run EPV was a false alarm: the filter had not finished post-delay-first warm-up at t=4.26 s; its divergence metric triggered EPV spuriously. The H_error=10000 reset caused aggressive re-adaptation during DT speech. Q1 in the ON run accelerated convergence enough to pass the EPC detector threshold without triggering the false EPV, and the subsequent stable convergence trajectory preserved DT speech better.

Key difference from the DT_mvmt failures: no EPV fires in the ON run → no H_error=10000 reset → no multiplicative amplification of Q1 leakage. Q1 acts as a moderate convergence accelerator, not a destabilizer.

| Metric | A (OFF) | B (ON) | Δ |
|---|---|---|---|
| echo | 4.454 | 4.517 | +0.063 |
| deg | 2.446 | 2.775 | **+0.329** |

---

## Root-Cause Summary

The FAIL is **not a code bug** — Phase 1 confirmed the mechanism is implemented correctly. The FAIL is a **parameter mis-calibration** for PBFDKF:

| Parameter | Current (AEC3 ported) | Problem |
|---|---|---|
| LC_TRANSIENT | 0.0125 (5× steady) | Too large for PBFDKF; H_error amplification is non-linear |
| Duration | 250 + 100 = 350 hops (3.5 s) | Too long; DT speech spans this window in all 4 DT_mvmt cases |
| EPV interaction | None (Q1 does not reset on EPV) | EPV + elevated lc multiplicatively amplifies Kalman gain |
| DT gate | None | Elevated leakage fires during DT speech as well as FS periods |

The one success (nVUnxqHLr) occurs because no EPV fires in the ON run and the DT speech window falls after Q1 smoothing has already reduced lc back toward steady.

---

## Recalibration Candidates (Phase 2.1 design — requires authorization)

These are NOT implemented. Listed for Phase 2.1 design discussion.

| Option | Description | Expected effect |
|---|---|---|
| R1: Reduce amplitude | LC_TRANSIENT = 2×-3× steady (0.005–0.0075) | Limits H_error overshoot; may lose some convergence speed |
| R2: Reduce duration | transient_hops = 50–100 (not 250) | Limits DT exposure window; most echo path stabilizes within 1 s |
| R3: DT gate | Suppress Q1 leakage boost when nearend_dominant or DT-active | Prevents DT speech damage; requires reliable NE detector |
| R4: EPV sync | Reset Q1 counter when EPV fires | Prevents Q1 + EPV multiplicative amplification |
| R5: Convergence-gated | Only apply when filter H_error > convergence threshold | Avoids destabilizing already-converged FS cases |

R4 (EPV sync) directly addresses the Hp5g1asacU catastrophe.
R2 (shorter duration) directly addresses XRTnTUjU and FS_static regressions.
R1+R4 combination is the minimum-change candidate.

---

## Phase 2 Gate

| Rule | Status |
|---|---|
| Gate 1: No DT_mvmt case regression > 0.05 dB | **FAIL** |
| Gate 2: No catastrophic outlier \|Δ\| > 0.20 | **FAIL** |
| Gate 3: nVUnxqHLr EPV suppression beneficial | **PASS (beneficial +0.329)** |

**Hard rules (unchanged):**
- No 12-case bench until Phase 2 passes
- No 800-case bench until Phase 3 passes
- No algorithm changes without explicit user authorization
