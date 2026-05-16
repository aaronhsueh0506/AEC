# P52 Phase A — Verdict (Path 3 closure)

**Branch**: `feature/p52-phase-a-shadow`
**Date**: 2026-05-11
**Status**: **CLOSED with regime handler renamed and documented** (not "closed and reverted").

## What Phase A actually accomplished

| # | Task | Status | Commit |
|---|---|---|---|
| Phase 0 | Design lock v1.1 | DONE | (separate branch `feature/p52-design-lock-v1.1` `dd63288`) |
| A.0 | Retire ShadowCopyController | **FAIL** | `eac5325` → reverted `3236f6c`; verdict `6cf2731` |
| A.0 post-mortem | Forensic trace on the 1 regressing case | DONE | `410c238` |
| A.0R.1 | **Rename** ShadowCopyController → PathChangeRegimeHandler | DONE | `e681f4d` |
| A.0R.2 | **Observability**: per-frame regime trace flag (default-OFF) | DONE | `710b105` |
| A.0R.3 | **AcousticRegimeClassifier** (analysis-only) | DONE | `401e62b` |
| A.0R.4 | Unit tests (13 cases) | DONE | `401e62b` |
| A.0R.5 | **This verdict doc** | DONE | (this commit) |
| A.0R.6 | 800-case byte-equal sanity | **PASS (100 % cases byte-identical)** | this commit |
| A0 pre-flight (§0.4) | Discriminator on T1 harness | **NOT RUN** (gated by A.0 fail) | — |
| A.1 – A.6 (PBFDKFCore extraction, ERLE util, PathChangeDetector, R-reset hook, integration, trace flag) | **NOT RUN** (gated by A.0 fail) | — |
| T1 – T4 hard tests | **NOT RUN** | — |

## Path 3 reframe

P52 v1.1 §2.6 framed A.0 as: "retire ShadowCopyController under the premise
shadow is just an information source." A.0 800-case run **falsified that
premise** narrowly (1 case regressing −0.56 dB; 1.88 % frames over 0.1 dB
bar). v1.1 §2.6 fail action was revert + close Phase A — done.

The post-mortem ([p52_a0_postmortem.md](p52_a0_postmortem.md)) sharpened
the failure mechanism: the controller is a **catastrophe defence on the
top ≤ 1 % of cohort echo-path-nonstationarity**, not a margin squeezer or
legacy artifact. The cohort tail (target case at ERL_decile_std p99.2) has
no stable post-change echo path; high Kalman gain (Yang-2017 fast-recovery
prescription) would accelerate mis-adaptation, not robustness. The
controller's `boost_q + pause_main` force main W → 0 — the only known
defence for this regime.

Three closure paths were available:
1. **Close + revert (Path 1)** — done by `6cf2731`; treat as cohort dead-end.
2. **Reopen Phase A with new spec (Path 2)** — would require new design
   lock; outside the v1.1 anti-loophole "no mid-Phase re-spec" rule.
3. **Rename + document (Path 3)** — recognise the controller's correctness
   by design, add observability, document the regime that warrants it.

**Path 3 was executed**: Tasks A.0R.1 through A.0R.6 above.

## A.0R.6 — 800-case byte-equal sanity result

Tool: `tools/research/p52_a0r6_byte_equal.py` (snapshot + diff).
Standard: preset=balanced / fl=832 / cng=True / j4 / `np.random.seed(42)` per
case. Pre = HEAD `410c238` (post-mortem; pre-rename `ShadowCopyController`).
Post = current HEAD (post-rename `PathChangeRegimeHandler`, trace flag
default-OFF, classifier present but unused).

| Quantity | Value |
|---|---:|
| Cases byte-identical (`np.array_equal`) | **800 / 800** |
| Total samples compared | 325,123,520 |
| Samples exact match | 325,123,520 (100 %) |
| Samples within `atol=1e-6, rtol=1e-5` | 325,123,520 (100 %) |
| Top-10 case max abs delta | 0.0 across the board |
| T5-style ≥ 99.99 % bar | **PASS** |

Snapshot artefacts: `/tmp/p52_a0r6/{pre,post}.npz`, `/tmp/p52_a0r6/verdict.json`.

The rename + observability + classifier modules introduce **zero numeric
drift** in production behaviour.

## Observability now available (default-OFF)

| Surface | API | What it shows |
|---|---|---|
| Regime trace flag | `AecConfig.trace_p52_regime_handler = True` | per-frame `_regime_trace_rows` list on `AEC`; columns include `boost_q_fired / reverse_copy_fired / main_paused_fired / w_l2_{before,after} / q_max_{before,after} / shadow_w_l2_{before,after} / erle_main_{before,after} / copy_counter / copy_err_baseline`. |
| Trace dump | `aec.dump_regime_trace(path)` | writes the captured rows to `.npz`. |
| Regime classifier | `from aec_p52_regime_classifier import AcousticRegimeClassifier` | offline: pass `(mic, lpb)`, get `RegimeClassification` with regime ∈ {stable, mildly_nonstationary, wildly_nonstationary}, decile-std/ptp dB, and per-decile ERL trace. **Must not be wired into production paths** — enforced by `python/test_p52_regime.py::AntiLoopholeTests`. |

## Hard guarantees (anti-loophole)

- Production behaviour identical to pre-rename HEAD on the 800-case bench (A.0R.6).
- Trace flag default-OFF — zero overhead, byte-equal when off.
- Classifier output not consumed by `aec.py` (test enforces).
- No legacy alias for `ShadowCopyController` retained.
- Original handler logic unchanged (A.0R.1 was pure rename).
- C-impl in `c_impl/` retains the legacy `ShadowCopyController` / `shadow_copy_*` names; out of scope for the Python rename (the C port is downstream and will follow when synced).

## Pointers to deeper material

- Design lock: [p52_design_lock_v1.1.md](p52_design_lock_v1.1.md)
- Task A.0 verdict: [research_log_p52_task_a0_verdict.md](research_log_p52_task_a0_verdict.md)
- Post-mortem: [p52_a0_postmortem.md](p52_a0_postmortem.md)
- Anomalies: [phase_a_anomaly_notes.md](phase_a_anomaly_notes.md)
- Cohort summary: [research_log_aec_p25_p58_summary.md](research_log_aec_p25_p58_summary.md)

## Disposition

**Phase A is now CLOSED with regime handler renamed and documented.**
Production `main` is **unchanged** (this branch is unmerged). Whether
to merge the rename + observability into `main` is a separate user decision,
independent of Phase B / Phase C progress on the other branch.

Phase B (separate branch `feature/p52-phase-b-refactor`) continues on its
own 6-week schedule. After Phase B T5 passes, Phase C will run with one
of two configs depending on a future user decision:

- Phase B branch alone merged to main → §4.5 "Phase B only / A unmerged" path
- This branch also merged (Path 3 outcome) → Phase C "B + observability" path

Path 2 (regime-aware redesign, e.g. arming a `wildly_nonstationary`-only
intervention) is **not within v1.1 scope**. It would require a new design
lock that explicitly anchors on the post-mortem finding rather than the
v1.0 / v1.1 "shadow as information source" premise.
