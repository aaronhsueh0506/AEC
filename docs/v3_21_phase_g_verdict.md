# v3.21.20 Phase G — isolation experiment verdict (2026-05-24)

**Status**: CLOSED no-ship. Isolation experiment — answers the audit
question and informs future Phase D mechanism documentation. **Does NOT
close M2 / M3 / M1 / M4.**

## Experiment definition

**Question**: Is the AecState `gain_change` dispatch
(`_aec3_pending_gain_change = True` → `_aec3_state.handle_echo_path_change(gain_change=True)`
→ `erle_estimator.reset(False)`) at EPV + shadow_rise sites decorative
versus the load-bearing filter-side H_error reset?

**Independent variable**: at EPV + shadow_rise sites only, gate the
`self._aec3_pending_gain_change = True` setter behind a new flag
`skip_aec3_gain_change_dispatch_at_epv_shadow_rise` (default OFF, byte-
equal to production).

**Held constant**:
- Filter-side `filter.handle_echo_path_change(delay_change=True, gain_change=False, zero_filter=False)` — unchanged.
- Legacy companion machinery (`_arc_m_q_boost` / `_p_max_override` / `_p_floor_beta` / `_erl_estimate` cap / `_epc_render_forced_remaining` / `_apply_epc_state_reset` / `_maybe_mark_diverged` / `_f_e3_handle_epc_fire` / dtd_coherence dampening) — unchanged.
- delay_first / delay_shift sites — unchanged (`_aec3_pending_delay_change` still queued there).
- M2 W=0 / M3 shadow zero / M1 / M4 / usable_linear / PathChangeRegimeHandler — unchanged.

**Variants**:
- **A_off** (control): current production / Phase D wiring, no Phase G flag.
- **G**: A_off + `skip_aec3_gain_change_dispatch_at_epv_shadow_rise=True`.

## Per-case event counts

| Case | Variant | EPV fires | shadow_rise fires (est) | filter H_error resets | `_aec3_pending_gain_change` dispatches |
|---|---|:---:|:---:|:---:|:---:|
| nVUnxqHLr | A_off | 0 | 2 | 2 | 2 |
| nVUnxqHLr | G | 0 | 2 | 2 | **0** |
| jtYTdZm | A_off | 1 | 2 | 3 | 3 |
| jtYTdZm | G | 1 | 2 | 3 | **0** |
| wVYS (mvmt) | A_off | 0 | 3 | 3 | 3 |
| wVYS (mvmt) | G | 0 | 3 | 3 | **0** |

(Last column = epv_fires + shadow_rise_fires in A_off; structurally 0 in G.)

**Confirmed**:
- ✓ G drops `_aec3_pending_gain_change` dispatches at EPV + shadow_rise to 0.
- ✓ G preserves filter-side H_error reset count (3/2/3 matches A).
- ✓ G preserves EPV / shadow_rise detection fire counts (event detection unchanged).
- ✗ The `erle_reset_signal` diagnostic label (orchestrator:3415-3429) is **NOT** the AecState dispatch — it's just `epc_active`-derived (label = 2 when EPC active, label = 0 otherwise). Both A and G show `erle_reset_signal = 2` on EPC frames because both have identical EPC detection. The round-1 audit doc's reliance on this signal to infer "AecState fires identically" was a misread; corrected here.

## Audio equality

| Case | A md5 | G md5 | equal | rms_diff | max_abs_diff |
|---|---|---|:---:|---:|---:|
| nVUnxqHLr | 2ad274ed | 2403681d | **NO** | 1.28e−04 | 7.78e−03 |
| jtYTdZm | a5350cfe | 67d47797 | **NO** | 1.57e−04 | 9.28e−03 |
| wVYS (mvmt) | 54809a19 | 56dd0694 | **NO** | 1.28e−03 | 6.21e−02 |

A vs G is **NOT byte-equal** on any of the 3 conflict cases. Max
amplitude divergence reaches 0.062 (~−24 dBFS) on wVYS movement —
audible delta. **Per rule "only run 12-case if 3-case is clean" → 12-
case NOT run.**

## Per-case AECMOS (A vs G, 3 cases)

| Bucket | Case | A_echo | G_echo | Δecho | A_deg | G_deg | Δdeg |
|---|---|---:|---:|---:|---:|---:|---:|
| DT_static | nVUnxqHLr | 4.657 | 4.659 | +0.001 | 3.299 | 3.302 | +0.003 |
| DT_static | jtYTdZm | 4.513 | 4.512 | −0.001 | 2.589 | 2.589 | −0.000 |
| DT_movement | wVYS | 4.281 | 4.273 | −0.008 | 2.161 | 2.197 | +0.036 |

All deltas are **within AECMOS noise** (typically ±0.05 deg or ±0.05
echo on a single case). No case crosses the ±0.10 worst-case flag.

## Side-effect causing audio divergence

**Identified mechanism**: `_aec3_state.handle_echo_path_change(gain_change=True)`
calls `self._erle_estimator.reset(delay_change=False)` ([state/aec_state.py:224](AEC/python/modules/state/aec_state.py#L224)).

ErleEstimator.reset(False) clears:
- `_fullband.reset()` — fullband ERLE state
- `_subband.reset()` — per-bin subband ERLE state
- `_sde.reset()` (if SDE enabled — disabled in current production)
- Does NOT reset `_blocks_since_reset` counter (only delay_change=True does)

ERLE consumers downstream:
- [residual/residual_echo_estimator.py:253](AEC/python/modules/residual/residual_echo_estimator.py#L253) — `erle = aec_state.erle(onset_compensated)` feeds the per-bin echo PSD estimation under AEC3-aligned RES.

**Causal chain when AecState dispatch is skipped (G variant)**:
1. EPV/shadow_rise fires → filter-side H_error reset fires (kept).
2. AecState ERLE reset SKIPPED → `_subband` + `_fullband` retain pre-EPC
   state (which is the converged-filter high-ERLE estimate).
3. After EPC, the filter is briefly re-adapting (Kalman gain re-armed via
   H_error reset). True echo cancellation drops transiently. But the
   stale-high ERLE makes RES think the filter is still doing a great job.
4. RES underestimates needed suppression → residual echo passes through
   for ~10-30 frames until ERLE naturally decays.
5. Audio diverges in this window: G has slightly more residual echo
   than A.

**Quantitative effect**: max_abs_diff up to 0.062 (single sample on
wVYS movement) but AECMOS deltas ±0.036 — bit-perceptible, AECMOS-
neutral on the 3 cases tested. On a different cohort the AECMOS impact
could be larger; do not extrapolate.

## Verdict

- AecState ERLE reset on EPV + shadow_rise is **load-bearing in the bit-
  level sense** (output diverges when removed) but **AECMOS-neutral on
  the 3 conflict cases** (Δdeg / Δecho within noise).
- Keep `_aec3_pending_gain_change = True` at EPV + shadow_rise sites
  (do NOT ship Phase G as default).
- **Updated Phase D mechanism summary**: Phase D's load-bearing benefit
  has TWO components:
  1. **Filter-side H_error + counter reset** (primary, large
     impact: −0.104 DT_static / −0.018 FS_static echo / 6–17% utterance
     output RMS swing when removed via Phase F).
  2. **AecState ERLE estimator reset** (secondary, small impact:
     bit-level non-byte-equal, AECMOS ±0.036 on 3 cases).
- **Reframe per Audit B (2026-05-24, [docs/v3_21_audit_b_m1_m4_parity.md](v3_21_audit_b_m1_m4_parity.md))**:
  - The AecState ERLE reset on `gain_change` is **AEC3-valid** —
    `aec_state.cc:165-167` does exactly this. Our `state/aec_state.py:222-224`
    is AEC3 parity.
  - The AEC3 `RefinedFilterUpdateGain` on `gain_change` is a TODO STUB
    (no H_error reset). Our filter-side H_error reset on EPV/shadow_rise
    is NOT in AEC3.
  - Therefore the right reframe is: **EPV / shadow_rise as gain_change
    sources are PBFDKF-specific** (AEC3 doesn't have these detectors);
    the gain_change HANDLER side-effects we fire are a mix:
    `erle_estimator.Reset(false)` is AEC3 parity; H_error reset is not.
  - Current Phase D/G behavior is **legacy PBFDKF compensation that
    happens to use the AEC3 gain_change handler as the adapter for the
    AecState-side reset**, not v3.21 parity overall.

**Per user rule "Do not classify Phase D as v3.21.x parity if its
benefit is not AEC3-equivalent"**: Phase D is classified **v3.22
intentional divergence** (PBFDKF compensation), already in production,
not subject to re-tuning by the v3.21.x AEC3 parity arc.

## What Phase G did NOT close

Per experiment definition rules — **M2 / M3 / M1 / M4 remain OPEN.**

- M2 (`AdaptiveFirFilter::ZeroFilter` W=0) — only fires safely on real
  delay change events. 12-case has 0 such events (per round-1 audit).
  Status: substrate kept default-OFF; needs cohort with delay events.
- M3 (`CoarseFilterUpdateGain::HandleEchoPathChange` shadow handler) —
  ported alongside M2; same fate. Status: substrate kept default-OFF.
- M1 (`SubtractorOutputAnalyzer::HandleEchoPathChange`) — never ported.
  Status: NOT investigated; no port; no flag.
- M4 — never identified in our code; placeholder label.

## Code change shipped this round

| File | Change |
|---|---|
| [python/modules/config.py](AEC/python/modules/config.py) | Added `skip_aec3_gain_change_dispatch_at_epv_shadow_rise: bool = False` with 17-line docstring (default OFF, byte-equal). |
| [python/modules/orchestrator.py](AEC/python/modules/orchestrator.py) | Gated 2 lines (EPV site ~2522, shadow_rise site ~2605) with the new flag. Both sites' default behaviour unchanged when flag OFF. |
| [python/modules/epc.py](AEC/python/modules/epc.py) | Round-1 fix: added `AecEventType` to import (classifier bug). |
| [python/v3_21_20_phase_g_isolation.py](AEC/python/v3_21_20_phase_g_isolation.py) | New: render + trace + event-count + MD5 compare for 3 cases. |
| [python/v3_21_20_phase_g_score.py](AEC/python/v3_21_20_phase_g_score.py) | New: AECMOS score 3 cases A vs G. |

**No production-default behaviour change.** All edits are flag-gated;
default OFF preserves current production wiring exactly.

## Recommended next separate arc

Two open arcs per user's prompt; do not mix.

### B (recommended next) — M1 / M4 / SubtractorOutputAnalyzer state-reset parity audit

**Why first**:
- **Pure paper audit** — read AEC3 source for `SubtractorOutputAnalyzer::HandleEchoPathChange`
  + `RefinedFilterUpdateGain` + any other downstream state-reset
  consumers on the AEC3 EPC chain. Map to our pipeline. Identify any
  state-reset side-effect we don't currently dispatch.
- **No 800-case run required** (matches current rules forbidding it).
- **Cheap to investigate** before committing to A.
- **Could explain Phase E's secondary failure mode**: M1 reset clears
  `filters_converged_` which feeds the AecState same-frame stale-state
  risk identified in round-2 audit §4.3. If M1 is the missing piece,
  Phase E might be re-attemptable (still gated on delay events only).
- **Decision criterion**: does AEC3 have a `delay_change`-triggered
  state reset we don't dispatch, that would change the post-EPC behaviour
  on our PBFDKF + startup-gate + companion architecture?

**Out of scope for B**: any code change. Audit-only.

### A (defer) — true delay-event M2 / M3 audit

**Why defer**:
- Requires a render with delay events. The 12-case cohort has 0; the
  800-case cohort is forbidden by current rules.
- Pre-requisite: B should close first to know if M1/M4 are missing
  pieces that change the delay-event behaviour.
- Without B, an A audit would re-test M2/M3 in incomplete-port state
  and could mis-attribute failures.

## File index

- This verdict: [docs/v3_21_phase_g_verdict.md](AEC/docs/v3_21_phase_g_verdict.md)
- Round-1 + round-2 architecture audit: [docs/v3_21_epc_architecture_audit.md](AEC/docs/v3_21_epc_architecture_audit.md)
- Code under test: [python/modules/orchestrator.py:2514-2530, 2601-2611](AEC/python/modules/orchestrator.py)
