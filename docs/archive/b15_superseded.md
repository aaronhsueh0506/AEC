# B-15 raw_dt delay alignment — SUPERSEDED

**Status: Abandoned, superseded by B-16.**

B-15 spec (`docs/spec_b15_raw_dt_delay_alignment.md`) attempted to
fix the PZ7V FS#84 leak by replacing the `raw_dt` energy-mismatch
formula with a delay-aligned `echo_est_pwr` term. Implementation
reached Stage 1B (working-tree only, never committed to main).
Benchmark showed no PZ7V leak improvement. Approach abandoned
in favour of B-16 (`docs/spec_b16_raw_dt_jump_veto.md`).

## Root cause of B-15 failure

The replacement formula used `self.filter.echo_spec` magnitude as
the delay-aligned echo power reference. But `filter.echo_spec`
lags onset because **the filter itself is what's being frozen**
by the cascade B-15 was trying to break (see
`docs/PZ7V_FS84_root_cause.md` §2 and the Task 3 live diagnostic
in the B-16 investigation).

Circular dependency: we were using the frozen filter's output to
unfreeze the filter.

Stage 1B implementation on the experiment branch showed PZ7V leak
unchanged (verified via live diagnostic t=9.9-11.0 s per-frame
trace).

## Why B-16 works where B-15 didn't

B-15 attempted a **formula-level fix** (replace the `raw_dt`
computation with a better-behaved formula).
B-16 attempts a **signal-level veto** (detect the anomaly
signature, fall back to a slow EMA of the legacy formula).

The raw_dt cascade root cause is:

```
gain jump → raw_dt single-frame jump → mu_scale crushed →
filter frozen → error stays large → RES passes mic through → leak
```

B-15 tried to fix the first arrow (how `raw_dt` computes), which
required a reference signal that wasn't already corrupted by the
cascade. **No such signal exists** in the current architecture:

- `filter.echo_spec` is corrupted (filter is frozen)
- `shadow_adv` saturates under Kalman K-denominator
  self-normalisation (see
  `docs/aec3_full_architecture_analysis.md §2.3`)
- Coherence is live but not consumed on the `enable_dtd=False`
  path used by the 800-case evaluation

B-16 breaks the third arrow (mu_scale's response to raw_dt)
without needing a clean reference. Slow EMA is a valid fallback
*when* the detector identifies an anomaly sustained for 2 frames.
The detector uses the raw_dt signal's own rate-of-change, which
is corrupted in a *known pattern* (single-frame jump) that
differentiates from DT near-end speech onsets (gradual).

## Preserved artefacts

- **Spec**: `docs/spec_b15_raw_dt_delay_alignment.md` — kept as
  design record. Spec commits on main: `718b66d` (initial),
  `c544c67` (unit/smooth/invariant details), `f72b412`
  (`_conv_counter` fallback policy), `45c3239` (FFT unit
  consistency + semantics), `7731387` (dedicated
  `_b15_blend_counter`).
- **Branch**: `experiment/pbfdkf-no-epc-baseline` — contains B-15
  Stage 1B code + `AEC_EXP_NO_EPC` flag, retained as historical
  reference for the EPC experiment that confirmed EPC is
  cosmetic at PZ7V onset (`experiment/pbfdkf-no-epc-baseline`
  with `AEC_EXP_NO_EPC=1` yields bit-exact −13.83 dB leak).
- **B-15 Stage 1B implementation**: never merged. Code in branch
  above only.

## Lessons documented in B-16 design

The B-15 → B-16 pivot produced design-discipline improvements,
codified in the B-16 spec:

1. **Physics-first validation before spec**. B-16 Task 3 live
   diagnostic (per-frame filter / DTD / RES state probe)
   confirmed the cascade chain end-to-end *before* the fix was
   designed. B-15 designed from a hypothesis that had not been
   end-to-end-validated.
2. **Threshold validation across DT / FS / NE population before
   committing**. B-16 Task 4 (1 DT + 1 NE + PZ7V) and Task 5
   (3 DT + 2 FS-non-onset) cross-case threshold calibration;
   saw the boundary (XTqo_mv single-frame 0.823) that a
   smaller sample would have missed.
3. **Invariants as grep-verifiable assertions**. B-16 spec §5
   lists 5 invariants, each with an explicit `grep` verification
   command and a line-number window. B-15 spec had verbal
   design rules but no mechanical check.
4. **Dedicated fail-mode planning**. B-16 spec §7 enumerates
   expected failure modes with escalation procedures (e.g. "if
   PZ7V leak unchanged with veto firing, probe `dt_from_shadow`
   / `dt_from_energy`"). B-15 spec §8 rollback was a single
   paragraph without specific diagnostic steps.

These lessons are reflected in the Phase 2+ experiment queue
(`docs/aec3_full_architecture_analysis.md §8`).

## Reference chain

- Original incident: `docs/PZ7V_FS84_root_cause.md`
- B-15 spec: `docs/spec_b15_raw_dt_delay_alignment.md`
- B-16 spec (current): `docs/spec_b16_raw_dt_jump_veto.md`
- B-16 Stage 1D results: `docs/b16_stage_1d_results.md`
- Architecture context: `docs/aec3_full_architecture_analysis.md`
