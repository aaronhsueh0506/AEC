# v3.15 §1.0.S1 — B4 verdict: quiescent re-sync 'converged' dead branch (CLOSED, FIXED)

**Date**: 2026-05-14
**Branch**: `feature/v3.15`
**Baseline**: `b3273de` (main HEAD, v3.14 Arc P + R + S-orth.A all ON)
**Sprint**: §1.0.S1 (Phase A.0 v3.14 substrate re-verification, sprint 1 of 4)

## Bug

[python/aec.py:6364-6365](../python/aec.py#L6364-L6365) (pre-fix):

```python
_is_quiescent = (
    far_excited
    and hasattr(self.filter, '_error_psd')
    and getattr(self, '_prev_filter_state', 'idle')
        in ('refined_usable', 'converged')
)
```

The tuple `('refined_usable', 'converged')` mixes vocabularies from two
parallel state systems:

- **Internal P3f string state** (`_prev_filter_state` / `_filter_state` /
  `_diag['filter_state']`): values are `'idle'`, `'startup'`,
  `'diverged'`, `'suspicious_dt'`, `'refined_usable'`, `'coarse_learning'`
  (set by [aec.py:7180-7199](../python/aec.py#L7180-L7199)).
- **Public `AecFilterState` enum** (returned by `get_filter_state()`):
  values include `CONVERGED = "converged"` ([aec.py:64](../python/aec.py#L64)),
  used only by `AecStats`.

The string `'converged'` is from the *enum* vocabulary; `_prev_filter_state`
only ever holds the *internal-string* vocabulary. Result: the `'converged'`
branch was structurally unreachable — it could never fire.

## Impact

S-orth.A's quiescent safety regularization (designed to nudge shadow
`_error_psd` back toward main when drift exceeds 3× in steady FS) was
gated on `_prev_filter_state in ('refined_usable', 'converged')`. The
'converged' branch never contributed because it could never trigger.
The `'refined_usable'` branch still fires correctly, so the safety net
is **only partially functional, not fully broken** — but the dead branch
was a code-clarity hazard suggesting wider coverage than actually exists.

## Fix

[python/aec.py:6361-6371](../python/aec.py#L6361-L6371):

```python
# Quiescent safety regularization (Option B):
# When filter is converged and shadow _error_psd has drifted
# more than 3× away from main's in either direction, nudge
# shadow back by 10% blend per frame.  Fires only in steady
# FS (refined_usable + far_excited) so it cannot corrupt the
# non-stationary path where orthogonality matters most.
# B4 fix (2026-05-14): drop dead 'converged' branch — that
# string belongs to AecFilterState enum, not the internal
# P3f state machine (lines 7180-7199), which only sets
# 'refined_usable' for steady FS.
_is_quiescent = (
    far_excited
    and hasattr(self.filter, '_error_psd')
    and getattr(self, '_prev_filter_state', 'idle')
        == 'refined_usable'
)
```

## Verification

**5-case byte-equal sanity** (BALANCED + fl=832 + `--no-cng`):

| Case | cohort | hash match |
|---|---|---|
| `014AzuqPZku2004NbTTmcA_nearend_singletalk` | NE | EQUAL |
| `0KjzXA3g20qsd8zmSekADw_farend_singletalk` | FS | EQUAL |
| `0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement` | FS_movement | EQUAL |
| `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` | DT | EQUAL |
| `49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement` | DT_movement | EQUAL |

5/5 PASS. The fix is structurally byte-equal on production paths because
the dead branch was unreachable via the real state machine.

**Note on `--no-cng`**: `python/aec.py` CLI does not call
`np.random.seed(...)` (B3 will fix this); CNG output is non-deterministic
across CLI invocations. We use `--no-cng` for byte-equal sanity so the
diff isolates B4 logic, not CNG noise. CNG path is unaffected by B4
because the quiescent re-sync block is upstream of CNG.

**Synthetic-case test not constructed**: the plan called for a case
exercising the previously-dead branch, but since `_prev_filter_state`
is **never** set to `'converged'` by the state machine, any such test
would require monkey-patching state, not testing real behaviour. The
dead branch is genuinely structurally unreachable, so removing it is
provably non-destructive.

## Outstanding invariant debt closed

Per §0.5: "Arc S-orth.A safety regularization partially defeated (B4)" —
**CLOSED**. The remaining `'refined_usable'` branch still fires under
the same conditions as before; the dead-code removal is non-functional
but eliminates the misleading code-clarity hazard. The S-orth.A safety
net was never broader than `refined_usable` in practice; B4 removes
only the illusion that it was.

## Next

Proceed to §1.0.S2 (B5 trace + dispose `_shadow_copy_err_baseline`).
