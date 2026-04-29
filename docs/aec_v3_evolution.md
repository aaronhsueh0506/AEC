# AEC v3.x Evolution — Trace-Driven Multi-Axis Tuning

This document records the design rationale, trace findings, and decisions for
v3.0–v3.4. See `CHANGELOG.md` for terse version notes; this is the deeper
"why" alongside the code comments.

For algorithm description (filter, RES pipeline, gain rules), see
[aec_methods.md](aec_methods.md). For WebRTC AEC3 reference, see
[aec3_reference.md](aec3_reference.md). For the original detector decoupling
plan, see [signal_flow_constraints.md](signal_flow_constraints.md) and
[multi_point_change_plan_v3_1.md](multi_point_change_plan_v3_1.md).

---

## Goal of v3.x

Close DT/FS echo gap vs WebRTC AEC2 baseline on AECMOS, while preserving
DT_static deg lead and keeping NE deg above 4.0.

Starting point (v3.0.2 = bit-exact v2.8.1, CNG=True):

| bucket | ours | aec2 | Δ |
|---|---:|---:|---:|
| FS_static echo | 3.303 | 3.457 | −0.154 |
| FS_movement echo | 3.659 | 3.519 | +0.140 |
| DT_static echo | 3.986 | 4.331 | −0.345 |
| DT_movement echo | 4.065 | 4.149 | −0.084 |
| DT deg | 2.563 | 2.389 | +0.174 |
| NE deg | 4.016 | 4.098 | −0.082 |

(Note: pre-v3 evaluations used `enable_cng=False` for parity; numbers above
are CNG=True production-default.)

---

## Methodology — trace-driven, not ablation-driven

11 ablation variants in v3.0.x failed (B2 R1 split, C1 S1/S2/S3, C2 E3 no-reset
etc.). The lesson: AEC3-borrowed detector patterns don't translate; we needed
to find what's actually wrong in our pipeline, not copy theirs.

Method that worked:

1. **Identify worst N cases** by Δ vs AEC2 (per bucket).
2. **Trace per-frame, per-bin** through the residual pipeline:
   `attribute → 4 caps → reverb → echo_boost → fallback → nearend → enr → gain`
3. **Find the binding constraint** at leak hotspots (where ours_psd >> aec2_psd).
4. **Fix it; verify with smoke trace; then 800-case AECMOS.**

Tools: [diag_leak_hotspot.py](../python/diag_leak_hotspot.py),
[diag_full_chain.py](../python/diag_full_chain.py),
[diag_freq_temporal.py](../python/diag_freq_temporal.py),
[diag_multi_trace.py](../python/diag_multi_trace.py).

---

## v3.1.0 — render-mode-aware RES caps

**Trace finding** (worst static-DT case ZtGitIxr): filter never converges →
render-based mode active 98% of frames → `residual_echo_psd = far × ERL`
estimate. Then 4 sequential caps applied:
- `echo_psd × 2.0` cap dropped 71% of render estimate
- `error_psd × dt_suppress` cap dropped another large fraction
- `min_ne_from_dt` floor inflated nearend_est → ENR collapsed → gain stayed high

**Fix**: skip the echo and dt caps when `using_render_based`; halve the
min_ne_from_dt floor in render-mode (tunable via `render_min_ne_factor=0.5`).

**Result**: DT echo gap closed 13% (-0.252 → -0.218 vs v2.8.1 baseline).

---

## v3.2.0 — ERL outlier protection + render-mode gain ceiling

**Multi-case trace** (22 worst cases including 5 winners as control):

Surprise findings:
- `once_converged` is NOT a winner/loser discriminator (winners had 0% conv too)
- mean enr / mean gain doesn't predict AECMOS — transient leakage matters
- Gain p90=1.0 in 10% of DT×far frames in losers (no suppression that 10%)

Real discriminator: **ERL clamping % AND median ERL value**.
- Losers: ERL clamped at 0.3 in 90%+ of frames; median 0.4–0.95 (NE-corrupted)
- Winners: ERL not clamped; median 0.003–0.2 (filter learned echo cleanly)

Mechanism chain explaining the leak:
```
NE in mic → ERL = mic/far ↑ corrupted
         → render_ceil = far × min(ERL×2, 1) saturated at far × 1.0
         → residual capped at error_psd
         → error_psd inflated by dt_shaped × 0.5 floor
         → enr per-bin sometimes ≈ 0 → gain → g_max=1.0 → echo bursts leak
```

**Three coordinated fixes**:
- **Axis 1 — ERL outlier protection**: skip ERL update when `inst_erl ≥ 1.5`
  (mic > far is physically implausible; clip max to 1.0 from 10.0).
- **Axis 2 — Render-mode gain ceiling**: hard cap `gain ≤ 0.6` when
  `using_render AND far_active > 0.3`. Prevents transient leaks.
- **Axis 3 — Relax error_psd cap in render-mode**: `residual ≤ error_psd × 1.5`
  instead of `× 1.0`. Defensive; mostly no-op after Axis 1.

Why three together: only doing Axis 2 hurts DT deg; only doing Axis 1 leaves
the cap; only doing Axis 3 the estimate direction is still wrong.

**Result vs v3.1.0**: FS_echo +0.031, DT_echo +0.018, DT_deg −0.033, NE +0.008.

---

## v3.3.0 — Reverb tail tuning + error-based render fallback

**Trace finding** (5 worst-leak cases via `diag_leak_hotspot.py`):

At leak hotspots, lpb_NOW is small but lpb peaked 60–320ms ago. Pattern across
5/5 cases:

| stem | lpb peak past | peak/now ratio |
|---|---|---|
| WYKA2 #0 | -3 frames (60ms) | 6.2× |
| ZtGitIxr #1 | -6 frames (120ms) | 800× |
| s0oJq #0 | -15 frames (300ms) | 1.2× |
| m4789fd #0 | -7 frames (140ms) | 77× |
| wr54we #2 | -8 frames (160ms) | 260,000× |

**Root cause**: AEC2 has reverb-tail attribution; we use only current frame's
lpb. Echo at frame t comes from past lpb arriving with delay/RIR.

**Discovery**: codebase already had IIR reverb at `ResFilter.process()` (line
~1568) but with parameters too conservative for our PBFDKF architecture:
- `decay=0.65` → TC ≈ 50ms (much shorter than typical RT60 ~130ms)
- `gain=1.4`
- DT-gate kills reverb 70% during DT, exactly when we need it most

**Two coordinated fixes**:
- **Tune existing IIR reverb** (preset balanced):
  - `reverb_decay 0.65 → 0.85` (TC ~130ms)
  - `reverb_gain 1.4 → 1.6`
  - `ne_reverb_factor` formula `0.3 + 0.7×far×(1−dt)` → `0.7 + 0.3×far×(1−dt)`
    (DT-gate weakened — DT is when reverb matters most, not least)
- **Error-based render fallback** in `attribute_legacy`:
  ```python
  render_based_echo = max(far × ERL, error_psd × far_conf × 0.7)
  ```
  Where `far_conf` is per-bin "active echo path" indicator. When filter never
  converges, `far × ERL` underestimates by 100×; falling back to "70% of error
  is echo" recovers attribution.

**Iteration history**:
- rc7: max-window FIR reverb buffer in `attribute_legacy` — overlapped with
  existing IIR; reverted in favor of tuning the existing IIR.
- rc8: error-based fallback only — won FS_static for the first time (+0.005).
- rc9 = rc8 + IIR tuning — final v3.3.0.

**Result vs v3.2.0**: FS_echo +0.045, DT_echo +0.030, DT_deg −0.032,
NE +0.008. FS_static reaches +0.031 vs AEC2 (won).

---

## v3.4.0 — Skip render_ceil + NE/DT-aware reverb tuning

**Trace finding** (`diag_full_chain.py` on v3.3.0): at hotspot frame 706 bin
32, `attribute` returned residual ≈ 167 (from error-based fallback) but the
`render_ceil` cap at line 1554 (`residual ≤ far_now × ERL × 2 ≈ 2.24`) cut it
back down. enr collapsed → gain stayed high → echo leaked.

The cap exists to bound a possibly-wrong linear filter, but render-mode IS the
fallback for when filter is wrong — the cap was self-defeating.

**Three axes**:
- **Axis 1**: skip `render_ceil` when `using_render_based`. Tested gating on
  `not_once_converged` (rc11): didn't help DT_movement deg, lost FS echo —
  convergence is not the deg-leak discriminator.
- **Axis 2**: NE-protect reverb hard-cut. `reverb_gate=0` when
  `far_activity < 0.1`. far_activity EMA decays slowly — without hard-cut,
  reverb tail lingers 200ms+ post-far-burst, hurting NE-only frames.
- **Axis 3**: DT-aware fallback factor.
  `fallback_factor = 0.7 - 0.2 × clip(dt_for_fs, 0, 1)`. Strong DT (NE present)
  → less of error_psd is echo → smaller fallback floor (down to 0.5×).
  Protects DT_movement deg.

**Result vs v3.3.0**: FS_static echo +0.034, DT_static echo +0.008,
DT_movement deg −0.024, NE deg −0.002.

---

## v3.5 candidates (rc12-14) — all reverted

**rc12** (Axis 4: fast reverb decay when far instantaneous silent): hurt leak
+30% on echo hotspots. Reverb-tail-fallback (long decay for echo attribution)
contradicts fast-collapse-for-NE; cannot satisfy both with one reverb signal.

**rc13** (Axis 6: per-bin NE-dominant gain floor, WebRTC-style): falsely
triggered on echo leak hotspots. The test `error_psd > expected_echo × 5` is
true for both NE-dominant bins AND under-attributed echo bins; no
discrimination without a reliable echo estimate (which is exactly what's wrong).

**rc14** (Axis 5: `far_for_conf = max(far_psd, reverb_psd × 0.5)` in fallback):
smoke −2~−5% leak energy, AECMOS Δ < 0.01. Marginal — Axis 5 is mostly
redundant with the existing reverb mechanism that already adds reverb to
residual.

**Why these failed** — at hotspot bin 32 with v3.4.0 numbers:
```
attribute fallback → residual ≈ 18 (vs ideal 124)        6.9× short
+ reverb adds  ≈ 5.6                  → residual ≈ 24    5.2× short
nearend = max(error − residual, min_ne) ≈ 100             200× too high
enr = residual/nearend ≈ 0.23                              1000× too low
Wiener gain → 0.81 → Axis 2 cap → 0.6                      300× too high
output = 0.36 × mic = 44.5 (target 0.0004)              100,000× leak
```

To bridge this, fallback factor needs ~0.95 (vs current 0.5–0.7), which
requires reliable per-bin echo-vs-NE discrimination. Without it, pushing
factor toward 1.0 destroys NE in DT.

---

## Where v3.4.0 hits the ceiling — and Route B

The reverb + error-based fallback framework is at its limit. Further DT echo
gains require a per-bin **echo-vs-NE discriminator** that current pipeline
lacks:

- **WebRTC AEC3** uses long-history coherence + multi-band cross-correlation.
- **NN postfilters** (DTLN-AEC, TEACAEC) learn this discrimination from data.

**Route B candidate** (next iteration): long-history coherence per-bin.
Maintain `cross_spec_history(mic, lpb_history) / |lpb_history|² × |mic|²`
per-bin. Use as `confidence` in fallback factor:
```
fallback_factor = 0.3 + 0.6 × coh_history     # range [0.3, 0.9]
```
NE bins have low historical coherence with far → factor stays low (preserve);
echo bins have high historical coherence → factor pushes toward 0.9
(aggressive suppression).

Risks: implementation complexity, additional state, computational cost,
potential noise in coherence estimate during transients.

---

## v3.4.0 final scoreboard vs AEC2 (CNG=True, 800 cases)

| bucket | n | ours_e | aec2_e | Δecho | ours_d | aec2_d | Δdeg |
|---|---:|---:|---:|---:|---:|---:|---:|
| FS | 300 | 3.675 | 3.484 | **+0.190** | 4.999 | 4.999 | 0 |
| FS_static | 169 | 3.522 | 3.457 | **+0.065** ✓ | — | — | 0 |
| FS_movement | 131 | 3.871 | 3.519 | **+0.353** | — | — | 0 |
| DT | 300 | 4.107 | 4.262 | −0.155 | 2.393 | 2.389 | **+0.004** |
| DT_static | 186 | 4.098 | 4.331 | −0.233 | 2.440 | 2.304 | **+0.136** |
| DT_movement | 114 | 4.123 | 4.149 | −0.027 | 2.315 | 2.528 | −0.213 |
| NE | 200 | 4.998 | 4.998 | 0 | 4.008 | 4.098 | −0.091 |

**Win rate**: Echo 47% (DT 100/300), Deg 42%.

Compared to v3.0.2 baseline:
- DT_static echo gap closed 32% (−0.345 → −0.233)
- FS reaches +0.190 lead (was −0.014 net)
- DT_static deg buffer reduced from +0.314 → +0.136 (cost paid for echo gain)
- NE deg from −0.082 → −0.091 (still > 4.0)
