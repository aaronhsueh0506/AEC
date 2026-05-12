# P52 Phase B Task B.5 — final flag-path validation

**Date**: 2026-05-12
**Branch**: `feature/p52-phase-b-refactor`
**Status**: **PASS** (all four validation steps green)

## Verdict at a glance

| Step | Test | Bar | Result |
|---|---|---|---|
| 1 (gap closure) | M3 / M4 / M5 isolated byte-equal (100-case) | 100% exact | **PASS** (M1..M{2,3,4,5} all 100/100) |
| 3.1 | flag=False vs B.4 legacy (800-case) | byte-equal | **PASS** (800/800 identical) |
| 3.2 | flag=True vs B.4 refactored (800-case) | byte-equal | **PASS** (800/800 identical) |
| 3.3 | 50-case sequential timing flag=False vs flag=True | total < 5%, max < 20% | **PASS** (+1.48% total, +13.18% max) |

Tool: [tools/research/p52_phase_b_b5_final.py](../tools/research/p52_phase_b_b5_final.py)

## Step 1 — per-module gap closure

Documented as §B.4-supp in [p52_phase_b_b4_verdict.md](p52_phase_b_b4_verdict.md).
All four cumulative configurations (M1..M{2,3,4,5}) produce 100/100 byte-
identical output to legacy; compensating-drift theoretical bug verified
absent. Commit `a4c87af`.

## Step 3.1 — flag=False produces byte-equal-to-production output

Pre-snapshot: `/tmp/p52_b4/legacy.npz` (production `aec.ResFilter`,
captured at branch HEAD before B.5 commit via the B.4 tool's monkey-patch-
disabled path).

Post-snapshot: `/tmp/p52_b5/flag_false.npz` (current branch HEAD
`a4c87af`, with `use_res_refactored=False` via the new `AecConfig` field
— NOT monkey-patch).

Both snapshots ran the same `_run_one` core: identical config / seed /
processing order. Worker process diagnostic confirmed
`res_classes={'ResFilter'}` (i.e. the false-path branch in `AEC.__init__`
selected legacy `ResFilter`).

```
cases_byte_identical:                   800 / 800
total_samples:                          325,123,520
fraction_exact:                         1.0
fraction_close (atol=1e-6, rtol=1e-5):  1.0
cases_over_0.1dB_mean_drift:            0
```

B.5 flag wiring is correctly a **no-op when OFF** — production behaviour
unchanged.

## Step 3.2 — flag=True produces byte-equal-to-B.4 refactored output

Pre-snapshot: `/tmp/p52_b4/refactored.npz` (monkey-patch path
`aec.ResFilter = ResFilterRefactored`).

Post-snapshot: `/tmp/p52_b5/flag_true.npz` (`use_res_refactored=True` via
`AecConfig` field). Worker process diagnostic confirmed
`res_classes={'ResFilterRefactored'}`.

```
cases_byte_identical:                   800 / 800
total_samples:                          325,123,520
fraction_exact:                         1.0
fraction_close (atol=1e-6, rtol=1e-5):  1.0
cases_over_0.1dB_mean_drift:            0
```

The production flag path is **equivalent** to the B.4 monkey-patch path
at the sample level. Both lead to the identical refactored execution.

## Step 3.3 — timing comparison (flag=False vs flag=True)

50 cases, deterministic seed=42 sample, sequential `j=1` execution, one
zero-frame warmup per A.0R.8 methodology.

| Quantity | flag=False (s) | flag=True (s) | Δ |
|---|---:|---:|---:|
| Total wall time | 121.940 | 123.739 | **+1.48 %** |
| Per-case median Δ% | — | — | +1.52 % |
| Per-case |Δ%| p50 | — | — | 2.27 % |
| Per-case |Δ%| p95 | — | — | 9.10 % |
| Per-case Δ% range | — | — | [−7.47 %, +13.18 %] |

| Pass criterion | Bar | Actual | Result |
|---|---|---|---|
| Total wall delta | < 5 % | +1.48 % | **PASS** |
| Per-case egregious outlier | none > 20 % | max +13.18 % | **PASS** |

Worst three cases (all sub-1 s `nearend_singletalk`, consistent with
A.0R.8 finding that sub-1 s cases have inflated relative jitter):

| Case | flag=False (s) | flag=True (s) | Δ% |
|---|---:|---:|---:|
| `NL23aL0w3E6huGfizi8xeg_nearend_singletalk` | 0.790 | 0.895 | +13.18 % |
| `06S6EY1JpU2qpe409kUaew_nearend_singletalk` | 0.793 | 0.883 | +11.26 % |
| `w0ogzwvJ7EmiHTCzx7sgwA_nearend_singletalk` | 0.830 | 0.911 | +9.75 % |

### Source of the +1.48 % overhead

The flag=True path adds **one Python method-override dispatch + one free-
function call per stage per frame** (5 stages × ~hop rate ≈ 500
additional indirections / second of audio). This is the documented cost
of the subclass-and-delegate pattern (per Phase B verdict § "Subclass-
and-delegate, deferred `ResState`"). Per §5.5 anti-loophole, **B.5 does
not optimise**:

- The overhead is not a logic change; it is a structural cost of the
  refactor pattern.
- The total is below the pre-locked 5 % bar (note: itself well above
  the A.0R.8 noise floor 6.36 % p95).
- A future `ResState` migration (post-Phase-C) may eliminate part of
  this dispatch cost by inlining stages into the orchestrator, but that
  is **deferred** to its own task.

Comparison to A.0R.8 baseline: A.0R.8 (Path 3 rename + trace flag wiring
only, no module extraction) measured −0.43 % total / 6.36 % same-code
noise floor. B.5 measures +1.48 % total — i.e. the real overhead of the
five-stage extraction is roughly +1.91 percentage points beyond the rename
baseline. p95 9.10 % vs noise floor 6.36 % → ~2.7 percentage-point excess,
mostly carried by short NE cases.

## Anti-loophole compliance (B.5)

- §5.1 (constants locked): no AECMOS threshold, hard test, or design-lock
  constant modified.
- §5.5 (no RES logic change): per Step 3.1 + 3.2 byte-equal results,
  RES output is sample-identical to legacy in both flag positions.
- §5.6 (reuse trace infra, not methodology): tooling reuses 800-case
  corpus + the A.0R.8 timing methodology; no new hypothesis-loop
  iteration.

## Phase B closure status

All Phase B tasks now complete and verified:

| Task | Status | Verdict |
|---|---|---|
| B.1 inventory | DONE | `research_log_p52_phase_b_inventory.md` |
| B.2 stubs | DONE | (commit message of `5d01e5b`) |
| B.3 Module 1-5 extraction | DONE | `p52_phase_b_module_{1,2,3,4,5}_verdict.md` |
| B.4 800-case byte-equal + gap closure | **PASS** | `p52_phase_b_b4_verdict.md` (with §B.4-supp) |
| B.5 flag wiring + final validation | **PASS** | this doc |
| Phase B verdict aggregate | **CLOSED** | `p52_phase_b_verdict.md` |

## Next

`feature/p52-phase-b-refactor` is **ready to merge to `main`** after
push. Per §6.4 second-branch rule, the post-merge `main` should be tagged
`p52-phase-b-closed`. Phase C kickoff doc to follow.

## Cross-references

- Phase B verdict: [p52_phase_b_verdict.md](p52_phase_b_verdict.md)
- B.4 verdict: [p52_phase_b_b4_verdict.md](p52_phase_b_b4_verdict.md)
- Tool: [tools/research/p52_phase_b_b5_final.py](../tools/research/p52_phase_b_b5_final.py)
- Artefacts: `/tmp/p52_b5/{flag_false,flag_true}.npz`,
  `/tmp/p52_b5/{diff_3_1,diff_3_2,timing_false,timing_true}.json`
