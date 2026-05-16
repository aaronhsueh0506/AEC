# v3.14 Housekeeping B1 + B2 Verdict

**Date**: 2026-05-14
**Branch**: `feature/v3.14-housekeeping`
**Scope**: Code review fixes B1 (LOW-MED) + B2 (LOW) from 3-agent code review 2026-05-14

---

## B1 — PBFDKF.reset() P-override cleanup (LOW-MED)

### Bug description
`PBFDKF.reset()` cleaned up dynamic P-override attributes (`_p_max_override`,
`_p_max_override_frames`, `_p_floor_beta`, `_p_floor_beta_frames`) using
`hasattr + delattr`.  This is functionally correct for single-call semantics,
but fragile under a specific latent scenario:

- Override is armed: `_p_max_override = 1.0`, `_p_max_override_frames = 30`
- Countdown expires: `_p_max_override` set to 0.5 (default), `_p_max_override_frames` deleted
- `reset()` is called: `_p_max_override` (now = 0.5) remains as an instance attr
  and IS deleted (hasattr is True), so the first-call case is handled correctly.
- A **second** `reset()` call on the already-reset filter is clean — no latent issue
  in normal two-call sequences.

The real latency risk is that `hasattr + delattr` is a two-step operation that
could theoretically raise `AttributeError` if the attr is deleted between the
two calls in a threaded context (not an issue for the single-threaded AEC
pipeline, but still an antipattern).  The `try/except AttributeError` form is
the canonical Python idiom for unconditional cleanup and is self-documenting.

### Fix applied
`python/aec.py`, `PBFDKF.reset()` (around line 1395):

Changed from:
```python
for attr in ('_p_max_override', '_p_max_override_frames',
             '_p_floor_beta', '_p_floor_beta_frames'):
    if hasattr(self, attr):
        delattr(self, attr)
```

Changed to:
```python
for attr in ('_p_max_override', '_p_max_override_frames',
             '_p_floor_beta', '_p_floor_beta_frames'):
    try:
        delattr(self, attr)
    except AttributeError:
        pass
```

Added explanatory comment documenting the lifecycle and the rationale for the
unconditional pattern.

### Unit tests added
`python/test_p52_regime.py` — new class `PBFDKFResetTests` (5 tests):

1. `test_reset_on_fresh_filter_is_clean` — attrs absent before and after reset on new filter
2. `test_reset_during_active_p_override_clears_all_four_attrs` — **core B1 test**: armed mid-countdown, all 4 attrs removed
3. `test_reset_after_countdown_expires_base_attr_also_cleared` — post-countdown residue cleared
4. `test_second_reset_is_idempotent` — double reset does not raise AttributeError
5. `test_getattr_fallback_works_after_reset` — getattr fallback returns correct defaults after reset

All 5 new tests PASS. Total test suite: 18 tests, 0 failures.

---

## B2 — AecStats.filter_state type annotation contract (LOW)

### Bug description
`AecStats.filter_state` was annotated as `AecFilterState` (enum) but the
accompanying comment described it ambiguously.  The internal `_diag['filter_state']`
/ `_prev_filter_state` string fields (values: `'idle'`, `'startup'`, `'diverged'`,
`'suspicious_dt'`, `'refined_usable'`, `'coarse_learning'`) are a SEPARATE P3f
finer-grained state machine distinct from the 7-value `AecFilterState` enum
returned by `get_filter_state()`.  Without explicit documentation, a consumer
reading `AecStats.filter_state` could incorrectly assume it uses the P3f string
vocabulary.

The actual data path was already correct:
- `AecStats.filter_state` is populated by `get_stats()` via `self.get_filter_state()`
- `get_filter_state()` correctly returns `AecFilterState` enum
- External `DiagPrinter` uses `s.filter_state.value` — requires enum; works correctly
- `run_one_case.py` line 154 uses `d.get('filter_state', 'startup')` from `_diag`
  (the P3f string) — this is a separate field, NOT `AecStats.filter_state`

The bug was annotation/documentation clarity, not a runtime type error.

### Fix applied
`python/aec.py`:

1. **`AecStats.filter_state` declaration** (~line 81): Added 7-line docblock comment
   explicitly distinguishing the public enum from the internal P3f string machine,
   listing both vocabularies, and clarifying the usage contract.

2. **`get_filter_state()` docstring** (~line 7280): Expanded docstring to document
   the 7-value enum vocabulary and explicitly note the distinction from
   `_diag['filter_state']` / `_prev_filter_state`.

No change to the `filter_state: AecFilterState` type annotation (already correct),
no change to `get_filter_state()` return type, no change to `get_stats()` assembly.

### Convention chosen
**Option (b) lite**: Keep `AecStats.filter_state: AecFilterState` (already
correct); keep `get_filter_state()` returning enum (already correct); document
clearly that the P3f string state is separate. No external consumer changes needed.

---

## Byte-equal verification

**Method**: Python-in-process comparison with `np.random.seed(0)` before each
`AEC(cfg)` instantiation (matching `eval_aec_challenge.py` determinism protocol).
Config: `preset=balanced, filter_length=832, enable_res=True, use_kalman=True`.

| Case         | Result         |
|--------------|----------------|
| DT_static    | PASS (byte-equal) |
| DT_movement  | PASS (byte-equal) |
| FS_static    | PASS (byte-equal) |
| FS_movement  | PASS (byte-equal) |
| NE           | PASS (byte-equal) |

**5/5 byte-equal.** Both fixes produce zero output change, confirming they are
latent bug fixes that do not alter observable production behaviour.

---

## Summary

| Fix | Severity | Files changed | Tests added | Byte-equal |
|-----|----------|--------------|-------------|------------|
| B1 PBFDKF.reset() unconditional cleanup | LOW-MED | `python/aec.py` | 5 in `test_p52_regime.py` | 5/5 PASS |
| B2 AecStats.filter_state annotation clarity | LOW | `python/aec.py` | 0 (annotation only) | 5/5 PASS |

No behavior change.  No new knobs.  No preset changes.  Production `balanced`
candidate (`feature/v3.11-route-a`) unchanged.
