# v3.15 §1.0.S2 — B5 verdict: _shadow_copy_err_baseline (CLOSED, doc-only)

**Date**: 2026-05-14
**Branch**: `feature/v3.15`
**Baseline**: `feature/v3.15` HEAD (B4 fix committed)
**Sprint**: §1.0.S2 (Phase A.0 sprint 2 of 4)

## Trace findings

`grep -n "_shadow_copy_err_baseline" python/aec.py`:

```
684:    #   _shadow_copy_err_baseline shadow's view of "best error in stable FS"
```

**Single occurrence — only in the docblock**. Never initialised, never
read, never written.

Cross-check `_shadow_mu_holdoff` (same docblock list):

```
685:    #   _shadow_mu_holdoff      shadow's own mu-holdoff counter (independent of main)
5448:        self._shadow_mu_holdoff = 0   # independent of main's _simple_mu_holdoff
5605:        if hasattr(self, '_shadow_mu_holdoff'):
5606:            self._shadow_mu_holdoff = 0
5861:        if hasattr(self, '_shadow_mu_holdoff'):
5862:            self._shadow_mu_holdoff = 0
```

Initialised at 5448 and reset at 5605 / 5861, but **no readers** — the
actual mu-holdoff logic at [aec.py:6022-6037](../python/aec.py#L6022-L6037)
uses `self._simple_mu_holdoff` (the single AEC-level counter) for both
main and shadow paths.

**Conclusion**: S-orth.A is shipped with **partial decoupling** —
`_error_psd` and `R` are decoupled and wired into `shadow_filter` each
frame (lines 6349-6384); `_copy_err_baseline` and `mu_holdoff` are still
coupled (single instances of each).

## Disposition

The plan allowed two paths: (a) wire `_shadow_copy_err_baseline` into
the quiescent re-sync block, OR (b) remove the doc comment.

**Wiring requires NEW MECHANISM, not bug fix**:

- `_copy_err_baseline` is owned by `PathChangeRegimeHandler`
  ([aec.py:4693](../python/aec.py#L4693)) and tracks
  `min(main_err_smooth, shadow_err_smooth)` during stable FS
  ([aec.py:4751](../python/aec.py#L4751)). A "shadow-only" baseline
  would require a parallel handler state plus a use case (currently no
  consumer reads such a quantity).
- `_simple_mu_holdoff` is set in
  `AEC._update_simple_mu_ratio` ([aec.py:6029](../python/aec.py#L6029))
  off the AEC's `_simple_mu_ratio`. A decoupled shadow holdoff would
  need a parallel `_shadow_simple_mu_ratio` mechanism, plus a way to
  apply it to shadow's mu independent of main.

Per Phase A.0 scope (prerequisite cleanup, not new design), and per
user directive to run linear-filter arcs autonomously to completion,
the right choice is **REMOVE doc claim, document as RESERVED**.

## Fix

[python/aec.py:674-697](../python/aec.py#L674-L697) — docblock rewritten:

- Lists only the two fields actually wired (`_shadow_error_psd` /
  `_shadow_R`).
- Calls out `_shadow_copy_err_baseline` and `_shadow_mu_holdoff` as
  RESERVED, with the design surface they would require (regime handler
  redesign / parallel mu-ratio mechanism).
- `_shadow_mu_holdoff` init/reset kept untouched (no-cost future
  wiring point).

[python/aec.py:908-911](../python/aec.py#L908-L911) — BALANCED preset
comment aligned with actual wiring scope.

## Verification

Doc-only changes (no code path modified). Byte-equal by construction.
No 5-case sanity needed — the change cannot affect any execution path.

## Outstanding invariant debt closed

Per §0.5: "Arc S-orth.A wiring possibly incomplete (B5)" — **CLOSED**
as documentation-incomplete-not-implementation-incomplete. The aspect
that was "missing" is genuinely future work, not a bug; aligning docs
with reality removes the misleading claim.

Future arc opportunity (NOT in v3.15 scope): a follow-on `S-orth.A.B`
arc could wire shadow-decoupled `_copy_err_baseline` and `mu_holdoff`
if a use case in a §1.4 (Arc M+G) sub-sprint emerges. Logged as a
v3.16 candidate for §1.7 RES audit doc.

## Next

Proceed to §1.0.S3 (S-orth.A retroactive nores listen).
