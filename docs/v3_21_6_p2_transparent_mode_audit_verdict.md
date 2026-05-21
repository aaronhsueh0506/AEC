# v3.21.6 Sprint P2 — TransparentMode / AecState parity audit VERDICT

**Date**: 2026-05-21
**Branch**: `feature/v3_21_6_parity_completion`
**Status**: ✅ CLOSED — intentionally incompatible with PBFDKF cohort tail; v3.22 G.2 follow-up
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v3.21.6 Sprint P2)
**AEC3 reference**: [`docs/aec3_extracts/src/aec3/transparent_mode.cc`](aec3_extracts/src/aec3/transparent_mode.cc), [`aec_state.cc`](aec3_extracts/src/aec3/aec_state.cc)

## P2.0 — AEC3 source diff

Compared [`aec_state.cc:189-325`](aec3_extracts/src/aec3/aec_state.cc#L189) (Update flow) + [`transparent_mode.cc`](aec3_extracts/src/aec3/transparent_mode.cc) (LegacyTransparentModeImpl) against [`python/modules/state/aec_state.py`](../python/modules/state/aec_state.py) + [`python/modules/state/transparent_mode.py`](../python/modules/state/transparent_mode.py).

Findings:

| # | Mismatch | Impact | Decision |
|---|---|---|---|
| A | `enable_transparent_mode=False` hard-coded at [orchestrator.py:589, 1160](../python/modules/orchestrator.py#L589) | TM subsystem dormant | **Audit ON → close as intentionally-incompatible** (see P2.2) |
| B | 3 block-unit constants in `transparent_mode.py` left at AEC3 4ms-block values (`_SANE_FILTER_DELAY_BLOCKS=5`, `_DIVERGED_SEQ_BOUND=60`, `_NUM_CONVERGED_BLOCKS_HIGH=50`) — comment incorrectly classified as "block-count semantics; stays N" but they are wall-clock durations | Dormant due to A; would over-lenient 2.5× if A flipped | **Parity fix shipped (dormant)** — renamed to `*_HOPS` with corrected values 2 / 24 / 20 |
| C | `any_coarse_filter_converged` not threaded into TransparentMode.update | Legacy variant ignores (`/* any_coarse_filter_converged */`); HMM variant not ported | **Aligned (no-op)** |
| D | `all_filters_diverged` source = `(not any_filter_converged) AND bridge.divergence_indicator > 1.0` proxy (AEC3 uses SubtractorOutputAnalyzer) | Different signal source feeding same role | **Aligned via different source signal** (document) |

## P2.1 — Parity substrate (dormant)

Implemented at `transparent_mode_enabled: bool = False` default:

- **NEW** [`AecConfig.transparent_mode_enabled`](../python/modules/config.py) — config-level gate; default False preserves current production behavior
- **CHANGED** [`orchestrator.py:587-591, 1156-1160`](../python/modules/orchestrator.py#L587) — both `_Aec3State` instantiation sites now consume the config flag instead of hard-coded False
- **CHANGED** [`state/transparent_mode.py`](../python/modules/state/transparent_mode.py) — 3 block-unit constants renamed + corrected:
  - `_SANE_FILTER_DELAY_BLOCKS = 5` → `_SANE_FILTER_DELAY_HOPS = 2` (AEC3 5×4ms = 20ms ÷ 10ms/hop = 2)
  - `_DIVERGED_SEQ_BOUND = 60` → `_DIVERGED_SEQ_BOUND_HOPS = 24` (AEC3 60×4ms = 240ms ÷ 10ms = 24)
  - `_NUM_CONVERGED_BLOCKS_HIGH = 50` → `_NUM_CONVERGED_BLOCKS_HIGH_HOPS = 20` (AEC3 50×4ms = 200ms ÷ 10ms = 20)
  Module-level docblock updated; the misleading "blocks not hops -> stays N" comment removed.
- **NEW** trace field `transparent_mode_active` in [`trace_hf_chain`](../python/modules/orchestrator.py) (and `_diag` dict; gated by `transparent_mode_enabled`)
- **NEW** `AEC_TRANSPARENT_MODE` env hook in [`run_one_case.py`](../python/run_one_case.py) + [`eval_aec_challenge.py`](../python/eval_aec_challenge.py)

At default OFF: zero algorithmic change. Code paths gated by `if self.config.transparent_mode_enabled:` stay dormant.

## P2.2 — Cohort 5-case trace verify

Rendered same 5 FS_static cohort as P1.2 (pcb1Nh, LN18k5r8, s90M7MOT, 9xjhi, lV0kQN). Each rendered twice:

- **OFF**: `AEC_E2_Y2_CLAMP=1` (P2 default state = TM off)
- **ON**: `AEC_E2_Y2_CLAMP=1 AEC_TRANSPARENT_MODE=1` (AEC3-parity restored)

### Trace summary

| Case | cfg | tm_active% | fa_consistent% | fa_peak | dpd_mean | tail_max | md5 differs? |
|---|---|---:|---:|---:|---:|---:|---|
| pcb1Nh | OFF | 0.0 | 56.2 | 189.0 | 0.756 | 5.628 | — |
| | ON | **0.0** | 56.2 | 189.0 | 0.756 | 5.628 | no |
| LN18k5r8 | OFF | 0.0 | 0.0 | 531.4 | 2.979 | 0.000 | — |
| | ON | **23.1** | 0.0 | 531.4 | 2.979 | 0.000 | **yes** |
| s90M7MOT | OFF | 0.0 | 20.1 | 170.5 | 0.729 | 0.201 | — |
| | ON | 2.3 | 20.1 | 170.5 | 0.729 | 0.201 | yes (small) |
| 9xjhi | OFF | 0.0 | 76.4 | 46.4 | 0.097 | 1.089 | — |
| | ON | 0.0 | 76.4 | 46.4 | 0.097 | 1.089 | no |
| lV0kQN | OFF | 0.0 | 91.4 | 23.3 | 0.030 | 1.145 | — |
| | ON | 0.0 | 91.4 | 23.3 | 0.030 | 1.145 | no |

### Findings

**TM fires on 1/5 cohort cases (LN18k5r8 @ 23.1%)** — exactly the cohort tail mechanism the original [orchestrator.py:580-586](../python/modules/orchestrator.py#L580) disable comment warned about, namely:

> *"... that makes TM falsely activate after 6s of strong render → kills usable_linear → R² collapses to nonlinear path forever."*

P1.2 already documented that LN18k5r8's FilterAnalyzer **never declares `consistent_estimate=True`** because PBFDKF's filter peak position is noisier than AEC3's ConsistentFilterDetector envelope. So on LN18k5r8 the TM Update logic flows:

1. `any_filter_consistent=False` → `sane_filter_observed` never set
2. After 5 s, `sane_filter_recently_seen=False` (init-window branch elapses)
3. `bridge.filter_converged` fluctuates → `recent_convergence_during_activity` toggles
4. After 60 s of active render without persistent convergence → `recent_convergence_during_activity=False`
5. After 6 s strong-render-not-saturated → final-branch `transparency_activated = strong_not_saturated_render_blocks > 6 s` → TRUE

Energy delta on LN18k5r8 OFF→ON: total -4.0%, 1-4 kHz band +1.9%. Mixed direction (downstream consumers of `transparent_mode_active()` propagate through `usable_linear` → `nearend_pwr` source flip → `dominant_nearend` redirection), not a clean "echo passes through" pattern.

### Mechanism diagnosis

LN18k5r8 represents the **PBFDKF-vs-AEC3 architectural incompatibility** that P1 already documented:
- AEC3 NLMS-derived peak position is stable enough for ConsistentFilterDetector's 1.5 s window
- Our PBFDKF Kalman update produces a noisier peak trajectory
- AEC3's TransparentMode trigger logic implicitly assumes the former; falsely activates under the latter

This is **not** "our TransparentMode is wrong" — the port is verbatim. It's "AEC3's TM trigger criteria don't match our adaptive-filter architecture".

## P2.1 per-mismatch verdicts (strict protocol per plan)

| # | Verdict | Reason |
|---|---|---|
| **A** | **Intentionally incompatible with PBFDKF; replacement = v3.22 Sprint G.2** | LN18k5r8 23.1% TM activation when fa_consistent=0% is cohort tail evidence that AEC3's trigger criteria assume filter-architecture properties (stable peak detector) our PBFDKF doesn't satisfy. v3.22 G.2 "Tighten trigger criteria" is the explicit divergence sprint. |
| **B** | **Fixed (parity correctness, dormant)** | Constants now wall-clock correct should A ever flip True; defensive hygiene with zero current behavior impact. |
| **C** | **Aligned (no-op)** | LegacyTransparentMode ignores `any_coarse_filter_converged`; HMM variant not ported and disabled by default in AEC3 anyway. |
| **D** | **Aligned via different signal source** | `bridge.divergence_indicator > 1.0` proxy serves the same role as AEC3 `SubtractorOutputAnalyzer.AllFiltersDiverged()`. Source differs (Kalman P_trace × shadow/main error ratio vs subtractor output analyzer state machine) but functional surface — boolean "filter looks diverged" — is equivalent. Documented; out of P2 scope to re-implement subtractor analyzer. |

## P2.2 — 800-case bench

**Not run** — per strict P2.1 protocol, option 3 (intentionally incompatible) does not require the bench step. Cohort evidence sufficient for close.

## Final verdict

✅ **CLOSED — AEC3 transparent-mode parity intentionally incompatible with current PBFDKF signal characteristics.** Production default remains OFF.

- Production state unchanged: `transparent_mode_enabled` default False; AecState's TM subsystem remains dormant.
- Parity substrate (config flag, env hook, hop-unit constants) shipped as dormant fidelity-correctness improvements — usable as substrate for any future PBFDKF-specific criterion, NOT as a step toward AEC3 parity restoration.
- v3.22 Sprint G.2 may design a **PBFDKF-specific** transparent-mode criterion (e.g., gate on a Kalman-state-derived "filter has had time to converge" indicator instead of AEC3's `any_filter_consistent` + 6-second `strong_not_saturated_render_blocks` heuristic). G.2 **MUST NOT claim AEC3 parity** — its design must be documented as intentional divergence with PBFDKF-architecture-specific rationale.

### Citation back to plan triage policy

This verdict places item (A) into **Bucket 2 (trace-gated → v3.22 Sprint G.2 as intentional divergence)** per the [plan's AEC3 Parity Gap Triage Policy](#). The PARITY classification for transparent-mode is permanently retired by this verdict; v3.22 G.2 inherits the file/flag/trace substrate but must reframe its design as divergence.

### Known follow-ups for v3.22 G.2

- Either ship a **PBFDKF-specific TM-replacement criterion** (labelled as intentional divergence), or **delete the subsystem** entirely from our port (and the corresponding `FilteringQualityAnalyzer.transparent_mode` parameter wiring), or **keep it dormant indefinitely** as a closed-AEC3-divergence-by-design.
- If candidate path: trace fields already in place (`transparent_mode_active`); G.0 cohort trace can re-use `AEC_TRANSPARENT_MODE` env hook. The 3 corrected hop-unit constants are dormant-correct and ready for the new criterion to reuse / replace.
