# AEC Post-v3.10.4 Plan

Date: 2026-05-06  
Baseline: `main` at v3.10.4 + P3c Phase 1a fast path  
Rule: do not mix trace-only work with behavior changes. If a gate fails, stop that branch and write a negative log.

---

## Current State

### Shipped / merged

- AEC `main` contains v3.10.4 release work.
- Audio_ALG `main` points its `lib/aec` submodule at AEC `main`.
- P3c Phase 1a delay fast path is shipped:
  - `n_updates >= 2`
  - high PAR
  - same lag for two consecutive updates
  - TTFS median improved from 4.09 s to 3.57 s
  - wrong locks: 0 / 800
  - AECMOS: bit-identical to v3.10.4
- C parity for the P3c Phase 1a fast path is complete.

### Closed negative work

- P3e DT advisory gate: closed.
- P3f Mini AecState mu gate: closed.
- P3g RES linear-vs-render switch: closed.
- P3h sustained-diverged reset: closed.
- P3c Phase 1b `n_updates >= 1` gate: trace shows unsafe; close after committing the negative log.

### Default-off diagnostics kept

- P3f state trace fields:
  - `main_err_ratio`
  - `shadow_err_ratio`
  - `p3f_shadow_advantage`
  - `erle_slope_db_per_s`
  - `post_reset_age_ms`
  - `filter_state`
  - `usable_linear`
- P3g residual-source audit fields:
  - linear residual estimate
  - render residual estimate
  - render blend diagnostics
- P3e/P3f/P3h behavior toggles remain default off.

---

## Immediate Cleanup

### Step 0.1 — Commit P3c Phase 1b negative

Files:

- `python/trace_delay_acquisition.py`
- `docs/research_log_p3c_phase1b_negative.md`

Summary to record:

- `n_updates >= 1` single-update delay locking is unsafe.
- Same-lag rate from update 1 to update 2 is only 73.25 % among ever-solid cases.
- Wrong-lock pool: 180 / 673 = 26.75 %.
- PAR cannot separate correct vs wrong first update:
  - even PAR >= 150 still has wrong locks.
- 7GT variants are not reliably helped:
  - 2 / 3 7GT variants would wrong-lock with single-update gating.
- Decision:
  - close Phase 1b-A.
  - do not implement behavior.
  - do not start partial-buffer FFT in this release path.

Acceptance:

- `python3 -m py_compile python/trace_delay_acquisition.py`
- commit and push.

---

## P4x — Legacy Plan B/C Re-evaluation

Goal: re-evaluate two old v3.9.0 branches that target the original issue: RES can suppress near-end during DT. These are distinct from the P3 arc because they do not touch delay, mu, reset, or linear-vs-render switching.

Baseline for all tests: current `main`, not the old v3.9.0 branch base.

### Phase 0 — Audit only

#### Plan B

Branch: `feature/v3.9.0-B-gamma-primary`  
Commit: `c50a0aa`  
Scope: small `dt_per_bin` formula change.

Current formula:

```python
dt_per_bin = np.maximum(
    np.full(self.n_freqs, effective_dt, dtype=np.float32),
    1.0 - coh2
)
```

Plan B formula:

```python
dt_per_bin_gamma = (1.0 - coh2).astype(np.float32)
if effective_dt > 0.5:
    floor_lift = float((effective_dt - 0.5) * 2.0)
    dt_per_bin = np.maximum(dt_per_bin_gamma, floor_lift)
else:
    dt_per_bin = dt_per_bin_gamma
```

Audit questions:

- Does `dt_per_bin` only affect RES / gain decisions?
- Does this formula still suffer from FS post-cancel low-coherence contamination?
- Does it interact with Plan A high-band behavior?
- What existing diag fields are enough, and what extra fields are needed?

Suggested diag:

- `dt_per_bin_mean`
- `dt_per_bin_hf_mean`
- `effective_dt`
- `coh2_hf_mean`
- `gain_hf_mean`
- `res_echo_hf`

#### Plan C

Branch: `feature/v3.9.0-subband-res`  
Commit: `3ffcf8f`  
Scope: larger RES scalar-to-subband redesign.

Candidate variables:

- `_dt_band[4]`
- `_erle_band[4]`
- `_over_sub_band[4]`
- `_over_sub_per_bin[n_freqs]`

Bands:

- 0-1 kHz
- 1-2 kHz
- 2-4 kHz
- 4-8 kHz

Audit questions:

- Is `_dt_band` still coherence-based?
- Does FS post-cancel also light up `_dt_band`?
- Is `_erle_band` stable enough to drive over-subtraction?
- Does `python/test_subband_res.py` still apply to current main?
- Can Plan C be turned into a dry-run candidate without touching output?

Output:

- `docs/research_log_p4x_legacy_bc_audit.md`

Gate to continue:

- Plan B can be cleanly rebased to current main.
- Plan C can be represented as diagnostic-only candidate fields before behavior.

---

## P4B — Plan B Behavior A/B

Do this before Plan C because it is smaller and easier to interpret.

### Variant

Apply Plan B formula on top of current `main`.

Do not change:

- delay estimator
- mu / adaptation
- reset / recovery
- RES linear-vs-render source
- Plan A kernel / HF cap

### Bench

Run BALANCED 800-case bench.

Must report:

- FS_static echo
- FS_movement echo
- DT_static echo / deg
- DT_movement echo / deg
- NE deg
- target DT high-band muffled cases
- 7GT sanity only, not primary gate

### Gate

Pass only if:

- DT deg or NE deg improves by at least +0.02.
- FS_static and FS_movement echo regression are each <= 0.02.
- DT echo regression <= 0.02.
- Improvement is not dominated by a single case.

Decision:

- Pass: continue to preset sanity and C parity planning.
- FS regression: close Plan B.
- Bit-identical or too small: close Plan B.
- Echo improves but deg regresses: document as non-target improvement; do not ship.

Output:

- `docs/research_log_p4b_gamma_primary.md`

---

## P4C — Plan C Subband RES

Only start P4C if Plan B has signal but is insufficient, or if P4B/P4x traces show scalar RES decisions are the bottleneck.

### Phase 2 — Dry-run only

Compute candidate fields without changing output:

- `_dt_band[4]`
- `_erle_band[4]`
- `_over_sub_band_candidate[4]`
- `_over_sub_per_bin_candidate`

Compare candidate vs current over-subtraction by cohort.

Trace cohorts:

- FS_static
- FS_movement
- DT_static
- DT_movement
- NE
- known DT high-band muffled cases

Gate:

- FS candidate over-subtraction is not looser than current.
- DT/NE high-band candidate shows selective protection.
- Candidate has better cohort separation than current scalar behavior.
- FS post-cancel is not misread as DT/NE.

Output:

- `docs/research_log_p4c_subband_res_dryrun.md`

### Phase 3 — Behavior A/B

Only if Phase 2 passes.

Start with soft blend:

```text
over_sub_final = 0.7 * current_over_sub + 0.3 * subband_over_sub
```

Variants:

- blend = 0.3
- blend = 0.5
- blend = 1.0 only if 0.3 and 0.5 are safe

Gate:

- FS echo regression <= 0.02.
- DT/NE deg improvement >= 0.03.
- DT echo regression <= 0.02.
- BALANCED passes.
- SOFT sanity passes before any ship decision.

---

## P4W — WebRTC-like Nearend-Protected Suppression Mode

This is the long-path architecture track. Do not start it until P4B/C are closed or clearly insufficient.

Goal: borrow WebRTC AEC3's state-driven suppressor idea, not a single formula.

### Phase 0 — Trace-only

Add diagnostic features:

Nearend evidence:

- high-band nearend excess
- high-band excess ratio
- high-band envelope modulation
- spectral flatness

Echo dominance:

- long-window render high-band power
- echo-to-error ratio
- render sync score
- far activity

Filter reliability:

- `filter_state`
- `usable_linear`
- `main_err_ratio`
- `shadow_err_ratio`
- `erle_win_db`
- `post_reset_age_ms`

Suppressor behavior:

- high-band gain mean/min
- RES high-band echo estimate
- current render/linear residual source

Gate:

- AUROC FS-vs-(NE + positive-DT) >= 0.80.
- FS_movement false positives are low.
- Target DT/NE frames are separable.

### Phase 1 — State classifier, diag-only

Candidate states:

- `echo_dominant`
- `nearend_dominant`
- `mixed_dt`
- `uncertain`
- `filter_unusable`

No output change.

### Phase 2 — Gain dry-run

Compute candidate gain next to current gain, but do not apply it.

Rules to test:

- nearend-dominant: higher high-band gain floor, weaker suppression
- mixed-DT: moderate gain floor lift, slower attack, faster release
- echo-dominant: current or stronger suppression
- filter-unusable: conservative render-based protection

### Phase 3 — Behavior

Only if dry-run passes.

Use soft gain lift, not hard switch:

```text
final_gain = min(current_gain + w * protection_lift, max_allowed_gain)
```

Do not change:

- delay
- mu
- reset
- RES source
- Plan A kernel

---

## Deferred / Do Not Start Now

- P2 fallback v2:
  - only restart if a future trace finds a real alignment-lost cohort.
  - do not use it for 7GT; 7GT is aligned.
- P3c partial-buffer FFT:
  - separate larger delay-estimator rewrite.
  - not a release-path tweak.
- More mu / reset / DT-advisory variants:
  - P3e/P3f/P3h closed.
- Direct Plan C behavior without dry-run.
- Direct WebRTC-like behavior without trace and gain dry-run.

---

## Recommended Order

1. Commit and push P3c Phase 1b negative log.
2. Run P4x audit for Plan B/C.
3. Run P4B Plan B behavior A/B.
4. If justified, run P4C dry-run.
5. If P4C dry-run passes, run P4C soft-blend behavior.
6. If B/C do not solve the nearend suppression problem, start P4W WebRTC-like nearend-protected suppressor trace.

