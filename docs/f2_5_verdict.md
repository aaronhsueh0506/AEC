# F2.5 Verdict — prev_dtd_conf Two-Stage Hangover

**Date**: 2026-05-12  
**Status**: CLOSED — ANALYTICALLY DEAD IN PRODUCTION CONFIG

---

## Hypothesis

`prev_dtd_conf` decays ×0.9 per frame (TC ≈ 10 frames / 100ms), faster than R
EMA recovery (α=0.95, TC ≈ 20 frames). Proposed fix: attack fast (1 frame),
hold 10 frames, then ×0.9 decay → total release ~300ms.

## Bench Result

800-case A/B: all 800 Δecho = 0.0000, Δdeg = 0.0000. Both runs bit-identical.

## Root Cause of No Effect

`prev_dtd_conf` lives exclusively in `_get_dtd_mu_scale()` (aec.py:4656),
which is only called from `_compute_mu_scale()` (aec.py:4626), which is only
invoked when `enable_dtd=True` (aec.py:4900).

The standard bench configuration and balanced preset both use `enable_dtd=False`.
In that path, `_get_simple_mu_scale()` is used instead — it relies on
`_simple_mu_ratio` / `_simple_mu_holdoff`, not `prev_dtd_conf`. The F2.5 flag
has zero effect in the production configuration.

## Analogy

Identical closure mechanism to F1.3 (delay_reliable gate softening) — the
targeted variable is unreachable from the production code path.

## Disposition

- `dtd_conf_two_stage` flag remains in aec.py as default-OFF dead substrate
  (no harm, no effect)
- Flag is NOT promoted to balanced preset defaults
- The analogous `_simple_mu_holdoff` mechanism is already addressed by F2.4
  (CONDITIONAL PASS, promoted to balanced)

## Next Steps

Proceed to F1.1 (reverse_copy P reset) — HIGH ROI, next in plan.
