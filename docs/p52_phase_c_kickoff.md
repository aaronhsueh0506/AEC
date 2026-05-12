# P52 Phase C — Kickoff (prep doc, pending user authorize)

**Date**: 2026-05-12
**Phase A closure**: `p52-phase-a-closed-path3` tag on `main@df2d2f8`
**Phase B closure**: pending merge of `feature/p52-phase-b-refactor`
(B.5 PASS as of `a4c87af` + new commits).

This document is **preparation only**. T6 / T7 / T8 execution and any
AECMOS scoring requires explicit user authorize.

## Phase C scope under Path 3 closure of Phase A

P52 v1.1 §4.5 defines a decision tree for Phase C entry. Phase A closed
via **Path 3** (rename + observability + classifier, *not* a new
intervention; production behaviour unchanged from pre-rename HEAD on
800-case byte-equal). Therefore the "Phase A unavailable / no Phase A
intervention" branch of §4.5 applies — Phase C reduces from the
4-configuration matrix to a 2-configuration comparison:

| Original 4-config matrix | Path 3 reality | Phase C action |
|---|---|---|
| Baseline | Production `main` (unchanged) | Reference |
| Phase A only | **Not applicable** — Path 3 ≠ new shadow-driven intervention | Skip |
| Phase B only | `use_res_refactored=True`, identical AECMOS to Baseline (proven byte-equal) | **Evaluate** |
| Combined | **Not applicable** for same reason | Skip |

## What Phase C will and will not measure

### Will measure

- **T7 (800-case regression)**: AECMOS Δ between Baseline and Phase B
  only. Expected: **identically zero** (byte-equal proves no audio
  output change). Run as a formality to satisfy the v1.1 hard-test list.
- **T6 (9 listener-anchored cases)**: same expectation, identically
  zero Δ.
- **T8 (filter state distribution)**: same expectation, no main filter
  change (Phase B does not touch `PBFDKF`).

### Will not measure

- Phase A vs Baseline AECMOS — Path 3 introduced no new intervention,
  so any Δ is by construction zero (verified at sample level in
  `p52_phase_a_verdict.md` §A.0R.6).
- Combined config — same reason.

### Optional add-on (Phase C-prime)

If the user wants to extract additional value from Phase C beyond
satisfying the v1.1 hard-test list, two add-on evaluations could be
run on the same `feature/p52-phase-b-refactor` branch:

1. **Timing audit on full 800 cases**: extend B.5 Step 3.3 (50-case
   sequential) to 800 cases via parallel run, capturing per-case
   timing distributions. Informs the cost of subclass-and-delegate at
   scale.
2. **Trace flag fire profile**: `trace_p52_regime_handler=True` on the
   800-case corpus to update A.0R.7 numbers post-Phase-B-merge (verify
   none of the regime handler decisions change after the swap — which
   they cannot, byte-equal proves it).

Both add-ons are informational. Neither is required by v1.1 spec.

## Configurations & tooling

### Configuration A — Baseline

`AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000,
filter_length=832, mode=AecMode.PBFDKF, enable_cng=True,
enable_res=True, enable_shadow=True)` — i.e. `use_res_refactored=False`
(default).

### Configuration B — Phase B only

Same as Baseline + `use_res_refactored=True`.

### Render command (Phase B only, mirroring Baseline)

```bash
python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
    --preset balanced --filter 832 --cng \
    -o out_phase_b/ -j 4
```

`eval_aec_challenge.py` does not currently expose
`--use-res-refactored`. Minimal change required: add a CLI flag in
`eval_aec_challenge.py` that sets `cfg.use_res_refactored = True`
before AEC instantiation. **Phase B branch isolation §6.4 permits
this** — `eval_aec_challenge.py` is tooling, not production code.

### Scoring

```bash
python3 python/bench_aecmos.py out_phase_b/ results_phase_b/ \
    --baseline results_baseline/scores.json
```

Expected output: Δ = 0.000 ± 1e-6 on every subset / cluster (because
input WAVs are byte-identical).

## Decision tree §4.5 applied to Path 3 closure

§4.5 originally branches on T7 outcome:

- T7 PASS (Δ within bar): ship Combined / A-only / B-only as production
  default depending on T6+T8.
- T7 FAIL: investigate regression, may revert.

Path 3 reality: T7 is **expected PASS by construction** (byte-equal).
The remaining decision is: does `use_res_refactored=True` become the new
production default?

| Outcome | Action |
|---|---|
| T7 PASS (byte-equal confirmed at AECMOS surface) | Promote `use_res_refactored=True` to default `True` in `AecConfig`, retire `ResFilter` legacy `_stage_*` methods in a follow-on |
| T7 FAIL (any Δ surfaces) | **Critical** — investigate; AECMOS Δ > 0 with sample byte-equal == 0 would indicate a scoring tool bug, not a real regression |

## Time-box

§4.6 specifies 2 weeks. With Path 3 closure narrowing the matrix and
byte-equal predicting zero AECMOS Δ, actual execution time should be
substantially less (~3 days for the formal T6/T7/T8 run + verdict
write-up).

## Authorize gates

Phase C **does not start** until user explicitly authorizes. Pre-start
checklist:

- [ ] Phase B branch merged to `main`
- [ ] Tag `p52-phase-b-closed` created on the merge commit
- [ ] `eval_aec_challenge.py` CLI flag added (small tooling change)
- [ ] Baseline AECMOS scores rendered (if not already cached from
  recent runs)

After authorize:

1. T7 (800-case AECMOS Δ Baseline vs Phase B only)
2. T6 (9 listener-anchored cases)
3. T8 (filter state distribution sanity)
4. Phase C verdict doc
5. Decision: promote `use_res_refactored` default to `True`? (user)

## Cross-references

- Design lock: [p52_design_lock_v1.1.md](p52_design_lock_v1.1.md)
- Phase A verdict: [p52_phase_a_verdict.md](p52_phase_a_verdict.md)
- Phase B verdict: [p52_phase_b_verdict.md](p52_phase_b_verdict.md)
- B.5 verdict: [p52_phase_b_b5_verdict.md](p52_phase_b_b5_verdict.md)
