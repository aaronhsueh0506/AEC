# AEC v3.x Evolution — Trace-Driven Multi-Axis Tuning

This document records the design rationale, trace findings, and decisions for
v3.0–v3.4. See `CHANGELOG.md` for terse version notes; this is the deeper
"why" alongside the code comments.

For algorithm description (filter, RES pipeline, gain rules), see
[aec_methods.md](aec_methods.md). For WebRTC AEC3 reference, see
[aec3_reference.md](aec3_reference.md). For the original detector decoupling
plan, see [archive/signal_flow_constraints_v3.0.2.md](archive/signal_flow_constraints_v3.0.2.md)
and [archive/multi_point_change_plan_v3_1.md](archive/multi_point_change_plan_v3_1.md).

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

---

## v3.5.x — Y2 fallback + nonlinear/saturation handling

**Trace finding**: at hotspot frames where `using_render_based=1` AND
post-filter signal still has high amplitude (`error_max_abs > 0.05`),
the residual_echo_psd estimate from `far × ERL` is sometimes orders of
magnitude too small → ENR drops near zero → soft-gate gives gain ≈ 1.0
(no suppression) → audible echo leak.

**Change**: AEC3-style "saturated_echo" Y2 fallback. When in render-mode
AND error peak high, force `residual_echo_psd ≥ mic_psd × 0.5`. Trace
verified WYKA2 worst-leak hotspot ratio dropped 111,827× → 5×.

**v3.5.0 result**: FS echo gains +0.06~+0.10. **But**: DT_static deg
−0.049, DT_movement deg −0.035 (NE damage cost).

(This change was later **fully ablated in v3.8.0 ABL-2** — see below.)

Also added: nonlinear echo mode (boost residual_echo when `saturation_level >
0.3` sustained), HF harmonic mapping (HF echo floor from LF echo when speaker
distortion present).

---

## v3.6.0 — Filter length 32 → 52ms (PR-D1)

**AEC3 alignment**: AEC3 default uses 13 partitions × 4ms = 52ms. We had 32ms.

**Change**: bump default `filter_length` from `sample_rate × 32 / 1000` to
`sample_rate × 52 / 1000` for sample rates < 44100 Hz.

**Result**: FS echo +0.03~+0.07 (more reverb tail captured). DT_static deg
−0.066, DT_movement deg −0.022, NE drops to 4.000 floor. Echo-prioritized
trade.

---

## v3.6.1 — DT-from-frame-zero stats detector

**Trace finding**: in DT-only-from-frame-0 cases (no FS warmup), filter
never converges → render-based mode never trusted → echo leaks. Existing
DT detection requires far-active history.

**Change**: add stats-only DT detector that compares mic energy to a
running noise floor (no far reference required). When triggered, gates
`mu_scale` floor to allow filter to learn even in DT.

(Detailed spec: `aec_methods.md` appendix E.)

---

## v3.7.0 — G1 KX blended P-update

**Trace finding** (KX deep trace + GPT analysis): in DT movement, `mu_scale`
drops to ~0.2 to slow W-update against NE contamination, but the P
covariance update was using K_optimal × X (full update) regardless → P
shrinks 72% during DT, then DT ends and P is too low → K too low → main
filter recovery 100+ frames slower than necessary.

**Root cause**: P-update was decoupled from W-update. P used optimal Kalman
gain even when W used scaled mu.

**Change**: blended KX for P-update —
```python
KX_blended = mu_mean × (K_optimal × X) + (1 - mu_mean) × (K_scaled × X)
```
When filter trusts itself (mu_mean ≈ 1), use full Kalman P-update; when
DT-suppressed (mu_mean < 1), use scaled KX matching W-update intensity.

**Result vs v3.6.1**: all 5 buckets neutral or positive. DT_movement deg
+0.012 (recovery faster). FS echo +0.001~+0.003 (filter learns better
post-DT). Confirmed root-cause fix, not Pareto sliding.

---

## v3.7.1 — PR-B drop render-based linear_failed (e2 floor removal)

**Trace finding**: `linear_failed = (using_render AND erle_factor < 0.2)`
branch fires 35% of DT_movement frames at `effective_dt = 0.008`
(false-negative). When fired, `residual_echo = max(residual_echo, error_psd × 0.9)`,
but `error_psd` in DT contains both echo AND near-end speech → using
e2 as residual_echo floor structurally over-suppresses near-end.

**WebRTC AEC3 reference**: AEC3 `residual_echo_estimator.cc` NEVER uses
error_psd as floor — relies entirely on `render × ERLE_smoothed`.

**Change**: remove the `using_render AND erle_factor < 0.2` branch. Keep
only `erl > 1.2` branch (later proven dead in v3.8.1).

**Result vs v3.7.0**: DT_movement deg +0.011, NE +0.002, DT_static deg
+0.004. Cost: FS_movement echo −0.016 (still leads AEC2 by +0.422).

---

## v3.8.0 — ABL-1 + ABL-2 (architectural cleanup family)

Continued v3.7.1 PR-B's AEC3-aligned cleanup. Two more legacy floors with
the same structural mistake.

**ABL-1 (drop v3.3 error_based_floor in ResidualEchoEstimator)**: removed
```python
far_conf = far_psd / (far_psd + error_psd × 0.05 + 1e-10)
fallback_factor = 0.7 - 0.2 × clip(dt_for_fs, 0, 1)
error_based_floor = error_psd × far_conf × fallback_factor
render_based_echo = max(render_based_echo, error_based_floor)
```
Same lesson: error_psd contains NE during DT.

**ABL-2 (drop v3.5 Y2 fallback `mic_psd × 0.5`)**: removed
```python
if render_based AND error_peak > 0.05:
    residual_echo_psd = max(residual_echo_psd, mic_psd × 0.5)
```
Trigger condition `error_peak > 0.05` is structurally NE-blind — DT speech
also produces high error peaks. Floor `mic_psd × 0.5` then equates the
near-end+echo mixture with echo proxy.

**Kept**: `render_dt_gain_ceil = 0.6` (ABL-3 reverted) — ablation showed
removal causes −0.046 FS_movement echo for only +0.008 DT_movement deg
(Pareto ratio 0.17, much worse than ABL-2's 0.67). Load-bearing.

**Result vs v3.7.1**: DT_movement deg +0.050 (gap to AEC2 closed from
−0.309 → −0.259, −16%). DT_static deg +0.033. NE +0.009. Cost: FS_static
echo −0.076, FS_movement echo −0.078 (FS still leads AEC2 by +0.344).
Preset-independent across all 4 presets.

---

## v3.8.1 — ABL-4 dead-branch cleanup + diagnostics hygiene

**ABL-4**: trace verified `self._erl_estimate` clipped to `[0.001, 1.0]`
in update path, so `erl_estimate > 1.2` gate fires 0.00% across all 5
buckets. Branch was retained as v3.7.1 PR-B "physical mic/far defense"
but value is structurally bounded → defense impossible to engage. Dead
code. Removed.

**Diagnostics hygiene**: `AEC.reset()` now clears `_far_active_blocks`
and `_dt_from_zero_count` getattr-lazy counters that were leaking
across batch eval cases.

**Result**: all metrics within AECMOS noise (±0.001) vs v3.8.0. Pure cleanup.

**Rejected**: delay_first ERL re-init, delay_shift ERL cap. Both
code-review-correct but Δ ≤ 0.003 on AEC Challenge dataset (delay-shift
events rare). Will revisit if movement-heavy multi-shift dataset surfaces.

---

## H1 exploration (DominantNearendDetector) — Pareto-bound, stashed

Concurrent with v3.8.0 ablation, attempted AEC3-style filter-independent
DT detector to gate PBFDKF main mu_scale during loud-DT (where 5-site
trace showed `eff_dt ≈ 0` and `gain_after_smoothing ≈ 1.0` — root cause
of remaining DT_movement deg gap is linear filter learning NE into W,
NOT residual filter over-suppression).

7 variants (v1-v7) tested. Best variant (v1, loose threshold) gave
+0.18 DT_movement deg at cost of −0.34 FS_static echo. Latest variant
(v7, with all bug fixes from code review) gave neutral result on this
dataset.

**Verdict**: detector based on raw mic/far ENR is on the same Pareto
curve as PR-F (per-bin coh2). Breaking the Pareto requires architectural
primitives we don't have:
- Phase coherence (R3 in plan): physical phase relation between echo
  and reference, separable from speech in DT.
- Masking-aware suppression (R7 in plan): psycho-acoustic masking
  threshold to allow inaudible echo to leak under NE.
- NN postfilter (DTLN-aec / `~/.claude/plans/jazzy-brewing-castle.md`):
  joint mic+far time-history mapping.

H1 implementation stashed (`git stash` ID: H1 implementation backup).
Architectural finding preserved in plan for v3.9+ stretch.

---

## v3.8.1 final scoreboard vs AEC2 (BALANCED, 800 cases)

| bucket | ours | aec2 | gap | gap @ v3.0.2 |
|---|---:|---:|---:|---:|
| FS_static echo | 3.801 | 3.457 | **+0.344** | +0.218 |
| FS_movement echo | 3.863 | 3.519 | **+0.344** | +0.412 |
| NE deg | 4.002 | 4.098 | −0.096 | −0.098 |
| DT_static echo | 4.256 | 4.331 | −0.075 | −0.149 |
| DT_static deg | 2.257 | 2.304 | −0.047 | +0.022 |
| DT_movement echo | 4.144 | 4.149 | −0.005 | +0.002 |
| DT_movement deg | 2.269 | 2.528 | **−0.259** | −0.270 |

Compared to v3.0.2:
- FS_static echo lead +0.126
- DT_static echo gap closed 50% (−0.149 → −0.075)
- DT_static deg held above AEC2 throughout v3.x (+0.022 → −0.047 still ≤ 1 bucket gap)
- DT_movement deg gap closed only 4% (−0.270 → −0.259) — remains the largest gap
- NE deg: virtually unchanged (~−0.10)

**Conclusion**: v3.8.1 is the AEC3-architectural-alignment endpoint via
floor cleanup. Further DT_movement deg improvement requires new primitives
(R3 phase coherence / R7 masking / NN postfilter), not floor tuning.
