# F1.1 Verdict — reverse_copy P Reset

**Date**: 2026-05-12  
**Status**: CLOSED — NO MEASURABLE EFFECT

---

## Hypothesis

When `reverse_copy` fires (shadow←main sync, main-winning case), shadow receives
main's W but retains its old high-Q P. Stale P → wrong Kalman gain K for copied W
for ~15 frames. Fix: re-arm shadow's `_p_max_override=1.0` + `_p_floor_beta=1.0`
for 15 frames so K recalibrates quickly. Also re-arm main for faster
re-adaptation post-copy.

## Implementation Issue Found and Fixed

**Bug**: The original implementation used `hasattr(filt, '_p_max_override')` as the
PBFDKF guard. `_p_max_override` is a dynamic attribute — deleted when its frame
counter expires (`del self._p_max_override_frames`, aec.py:1179). Outside active
EPC overrides, the attribute doesn't exist → `hasattr` returns False → the fix
never executed.

**Fix**: Changed guard to `hasattr(filt, 'P')`, since `P` is always set as an
instance variable on PBFDKF. Verified on qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk:
14 reverse_copy events, P override correctly set to 1.0 / 15 frames on all of them.

## Bench Results (with fix)

800-case A/B vs f2_5_baseline: 0 regressions / 0 improvements (threshold ±0.05).
All bucket mean deltas < 0.001. Cohort tail: worst Δecho = −0.0001.

## Root Cause of No Effect

`reverse_copy` fires only when main is clearly winning over shadow
(`main_err_smooth < shadow_err_smooth * 0.65`). This happens in stable convergence
periods where the acoustic path is already well-estimated. At that point:

1. Shadow's P recalibration only affects shadow's subsequent K and W evolution
2. Shadow W is not directly output — it only matters if shadow later wins (shadow→main copy)
3. The chain (reverse_copy P reset → better shadow K → better shadow W → better
   shadow→main copy) is 2nd-order and too weak to move AECMOS scores
4. AECMOS resolves ~0.05 dB; the effect is at 0.001 dB level (50× below threshold)

## Contrast with Plan's B1 Description

Plan B1 describes: "copy 後 main 的 P 仍是舊的 → Kalman gain K 完全錯誤達數 frame".
This concern is valid for the normal copy direction (shadow→main, when shadow wins).
In that case, main gets shadow's W but main's P may be stale — the wrong K for the
newly copied W on the main output path. F1.1 targeted reverse_copy (shadow←main),
which is the less impactful direction since shadow output is secondary.

The analogous fix for the normal copy direction (shadow→main: reset main's P when
it receives shadow's W) would be more impactful but requires different gating
since normal copy fires during convergence events (boost_q + pause_main flow),
not via a separate `copy_weights_from()` call. Deferred to future work if normal
copy becomes more frequent.

## Disposition

- `reverse_copy_p_reset` flag remains in aec.py as default-OFF (no harm)
- Not promoted to balanced preset
- `hasattr(filt, 'P')` guard fix is kept (correct behavior for when the flag is ON)
- F1.1 closes; the plan's B1 concern is real but the reverse_copy direction
  is too low-frequency and too secondary to move AECMOS at this point

## Plan Completion Summary

With F1.1 closed, all plan items from `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
have been adjudicated:

| Item | Verdict |
|---|---|
| F3.1 v3 per-bin excess metric | GREEN-PASS → balanced default |
| F2.1 EPC state hard-reset | CLOSED FAIL |
| F2.2 diverged-streak EMA reset | CLOSED FAIL |
| F2.3 Yang 2017 EPC R-reset | PASS → balanced default |
| F2.4 mu holdoff no-reset | CONDITIONAL PASS → balanced default |
| F1.2 streak_only gate | CLOSED FAIL |
| F1.3 delay_reliable gate | CLOSED — analytically dead |
| F2.5 prev_dtd_conf two-stage | CLOSED — analytically dead |
| F1.1 reverse_copy P reset | CLOSED — no measurable effect |
