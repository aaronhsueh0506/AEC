# P52 Task A.0 — Verdict: **FAIL** (Phase A CLOSED)

**Branch**: `feature/p52-phase-a-shadow`
**Date**: 2026-05-11
**Bench standard**: `feedback_bench_j4` — 800 cases, preset=balanced, fl=832 (52 ms), cng=True, j=4

## Acceptance bars (v1.1 §2.6)

| Bar | Rule | Result | Pass |
|---|---|---|---|
| Frame | `\|Δ_ERLE_main\|` ≤ 0.1 dB on ≥ 99 % frames | 98.12 % within | **FAIL** (0.88 pp short) |
| Case  | zero cases with mean Δ_ERLE_main < −0.5 dB | 1 case at −0.559 dB | **FAIL** |

## Numbers

| Quantity | Value |
|---|---:|
| Total cases compared | 800 |
| Total frames | 2,032,022 |
| Frames with `\|Δ\| > 0.1 dB` | 38,223 (1.88 %) |
| Cases regressing > 0.5 dB | 1 |
| Per-subset mean Δ (dB): doubletalk | +0.002 |
| Per-subset mean Δ (dB): farend_singletalk | +0.003 |
| Per-subset mean Δ (dB): nearend_singletalk | 0.000 |
| Worst regressing case | `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` Δ = **−0.559 dB** |

Snapshot artefacts: `/tmp/p52_a0/pre.csv` (main HEAD `8c20f6d`), `/tmp/p52_a0/post.csv`
(A.0-retired HEAD `eac5325`), comparison `/tmp/p52_a0/verdict.json`.

## Interpretation (v1.1 §2.6 fail mechanism)

Per-subset mean deltas (≤ 0.003 dB across all three subsets) confirm **the
average effect of ShadowCopyController retirement is essentially zero**. Both
acceptance bars fail by narrow margins — 0.88 pp on frame coverage, 0.06 dB on
the worst case. There is no broad regression.

However the failure mode itself is the load-bearing signal the design lock
anticipated. The single case `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`
regressing −0.559 dB demonstrates that **at least one cohort case relied on
in-flight shadow-copy / Q-boost / main-pause machinery for its ERLE_main to
stay within 0.5 dB of the production trajectory**. The frame-level 1.88 % >
0.1 dB rate is the cumulative footprint of those same load-bearing fires
scattered across other cases (each below the per-case 0.5 dB mean threshold
but individually visible at the per-frame scale).

This is exactly the v1.1 §2.6 fail action prerequisite:

> A.0 fail means copy controller IS load-bearing on production main ERLE —
> the v1.0 design premise that shadow is "just an information source" is
> invalid on this cohort, and the entire dual-filter-as-detector architecture
> is unviable here.

## Action taken

Per v1.1 §2.6 fail action + §5.1 / §5.3 anti-loophole (no constant retuning,
binary pass/fail per Phase):

1. **Revert** A.0 retirement commit on `feature/p52-phase-a-shadow`.
2. **Close Phase A.** A0 pre-flight (§0.4) does not run; Tasks A.1 – A.6 do
   not run; T1 – T4 do not run.
3. Per v1.1 §4.5 decision-tree: Phase C will follow the **"Phase B only / A
   unmerged"** branch when it opens (i.e., if Phase B passes T5, ship
   `ResFilterRefactored` as the production default; Phase A research artefacts
   are not retained on this branch).
4. **Phase B is unaffected** (separate branch `feature/p52-phase-b-refactor`)
   and continues per its own 6-week schedule.

## What this verdict does NOT do (anti-loophole)

- Does not retune the frame-coverage bar or case-regression bar.
- Does not retain `ShadowCopyController` retirement as a runtime toggle.
- Does not attempt "partial migration" (e.g. retire only `reverse_copy` and
  keep `boost_q` / `pause_main`). v1.1 §2.6 explicitly forbids this.
- Does not pivot to a different shadow architecture under the P52 design lock.

## Disposition of code

The A.0 retirement commit `eac5325` is reverted on this branch. The branch
itself is preserved for historical traceability and the anomaly notes
(`phase_a_anomaly_notes.md`) are retained as a research artefact.

## Cross-references

- Design lock: [docs/p52_design_lock_v1.1.md §2.6, §4.5, §5.1, §5.3](p52_design_lock_v1.1.md)
- Anomaly notes: [phase_a_anomaly_notes.md](phase_a_anomaly_notes.md)
- Bench standard: `~/.claude/projects/.../memory/feedback_bench_j4.md`
- Pre-decessor context: [research_log_aec_p25_p58_summary.md](research_log_aec_p25_p58_summary.md)
  — non-NN AEC arc was already exhausted before P52; this verdict adds one
  more architectural negative result to the file (PBFDKF-native shadow as
  detector-only is not viable on this cohort under current production
  `_alpha_r=0.95` / `shadow_q_ratio=3.0`).
