# v3.22 /simplify Audit — Variable Naming & Redundant Code

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Scope**: `python/aec.py`, `python/modules/{orchestrator,config,filters,epc}.py`, `python/modules/residual/suppression_gain.py` (~6909 lines total)
**Status**: Read-only audit. No code edits. 82 findings flagged. **Action plan is tiered by risk; user decides which tier to execute.**

## Top High-Impact Findings (verified by grep)

| # | Severity | Finding | Action |
|---|---|---|---|
| 14 | high | `_handle_delay_change_full()` defined at `orchestrator.py:190` but NEVER called anywhere (verified `grep` shows definition only) | Delete (10 LOC method + 23-line docstring) |
| 25 | high | `final_error_power` + `final_error_power_sum` written per-frame at orchestrator.py:2975/2978 + init/reset paths but NEVER read (verified) | Delete the 6 init/reset assignments + the 2 update lines |
| 26 | high | `_stat_far_hangover` initialised in 3 reset paths, never read or written elsewhere | Delete |
| 27 | high | `confidence_history = deque(maxlen=1000)` appended every frame at L3161, never read | Delete |
| 28 | high | `_misadjustment_reset_done_count` + 3 related config flags (`filter_misadjustment_use_fq_usable`, `filter_misadjustment_reset_done_frames`, `filter_misadjustment_threshold_phase3`) never consumed | Delete |
| 29 | high | `_misadjustment_fire_count` incremented but never read | Delete |
| 75 | **blocking-verify** | `_use_aec3_h_error` set True at `filters.py:345` and never written elsewhere. The legacy P-based Kalman branches at L398, L413, and the 124-line L440-563 block run only when False → all dead in production | Verify (toggle to False manually + run test to confirm path is no-op) then delete |
| 79 | high | `SubbandNlms = PBFDKF` backward-compat alias at `filters.py:714` referenced nowhere in `python/` | Delete |

## Config Flag Cleanup (~150 LOC opportunity)

18 dead AecConfig flags — consumed only via `eval_aec_challenge.py` env hooks targeting retired ResFilter. Verified retirement via `grep` of supposed consumers.

| Section | Flags | Lines |
|---|---|---|
| Plan A/B legacy | `plan_a_kernel_tight`, `plan_a_hf_cap_2k`, `plan_a_stat_mask_7k`, `plan_b_dt_per_bin_gamma` | config.py:317-326 |
| HF cap conditional | `hf_cap_conditional`, `hf_cap_metric_threshold` | config.py:359-360 |
| v3.12 RES per-state | `res_consume_filter_state`, `res_unified_gain_floor`, `res_dt_per_bin_unified`, `res_cap2_fs_loosen`, `res_noise_floor_refined` | config.py:511, 526, 543, 555, 570 |
| v3.14 Arc R per-band ENR | `res_per_band_enr`, `enr_t_ne_per_band`, `enr_s_ne_per_band` | config.py:712-720 |
| v3.15 §1.2 DT-NE compression | `dt_ne_compression_fix`, `dt_ne_state_scale`, `dt_ne_per_bin_thresh`, `dt_ne_per_bin_scale` | config.py:815-830 |
| v3.18 D.1 Subband NE (TOP-LEVEL — duplicates `SuppressorConfig.subband_nearend_detection`) | `subband_ne_detect_enabled` + 6 sub-fields | config.py:852-858 |
| v3.18 D.2 RES mask profile swap | `res_mask_profile_swap_enabled`, `res_mask_last_lf_band`, `res_mask_first_hf_band`, 4 tuples, `res_mask_ne_gate_dt`, `res_mask_swap_mode`, `res_mask_fs_overlay_*` | config.py:889-910 |
| v3.18 B1 Dominant NE (TOP-LEVEL — duplicates `SuppressorConfig.dominant_nearend_detection`) | `dominant_ne_detect_enabled` + 7 sub-fields | config.py:926-935 |

**Before acting**: cross-check with bench team that no active A/B uses these env hooks.

## Closed-Cycle Name Leakage (violates `feedback_no_version_in_var_names.md`)

Closed-cycle prefixes still present in live attribute / diag-key / config-field names. Each rename is medium-risk (touches state semantics across init/reset/process paths):

- `_round3_*` (v3.10 cycle marker) — orchestrator.py:113-118, 3072-3107, 3118-3158
- `_r7_*` — orchestrator.py:3119
- `_p3f_*` — orchestrator.py:826-827, 1339-1340, 2696, 2770, 2824 + L80 property
- `_p3h_*` — verify by grep
- `_p4b_*` diag keys — orchestrator.py:2638-2642
- `g_stage_*` diag keys — orchestrator.py:3113-3116
- `filter_misadjustment_threshold_phase3` — config.py:108
- `_round3_raw_dt_pre_epc` — orchestrator.py:2533

## Unused Imports

- orchestrator.py:11 — `import os` shadowed inside `AEC.__init__:579`
- orchestrator.py:15 — `from typing import List, Optional, Tuple` — `List`, `Tuple` unused
- orchestrator.py:33 — `RenderActivityState`, `FilterConvergenceState` unused
- orchestrator.py:34 — `RegimeHandlerDecision`, `AecEventType` unused
- orchestrator.py:30 — `_PB_MODES` unused
- config.py:10 — `from typing import Optional, List` unused
- config.py:11 — `import numpy as np` unused
- filters.py:14 — `from collections import deque` unused

## Duplicate Computation / Logic

| # | Finding | File:line |
|---|---|---|
| 16 | `_reverb_tail_max` computed twice per frame (production block + trace block; only production is the source of truth) | orchestrator.py:3522 vs 3701 |
| 35 | Diag-dict init duplicated 3 times (init + reset + derived-state-reset) with drifting key sets | orchestrator.py:1351-1365, 1072-1097, 909-932 |
| 44 | `_in_post_reset_warmup` calc duplicated | orchestrator.py:1736-1744 vs 1472-1476 |
| 45 | `if isinstance(mu_scale, np.ndarray): mu_scale *= _f else: mu_scale *= _f` — both branches identical (numpy broadcast) | orchestrator.py:1758-1761 |
| 46 | Same isinstance anti-pattern at L1776-1779 | orchestrator.py:1776 |
| 47 | `for filt in [self.filter, self.shadow_filter]` Q-boost loop duplicated at 2 sites | orchestrator.py:2235-2249, 2293-2310 |
| 48 | DTD coherence reset `self.dtd_coherence.confidence *= 0.3` in BOTH branches of split | orchestrator.py:2287-2288, 2291-2292 |
| 24 (revised) | `_inst_erle_smooth = 1.0` repeated across 5 reset paths (init + reset + derived-state-reset + 2 sub-resets) | orchestrator.py:897, 968, 1049, 1142, 1303 |
| 33 | `_pending_delay` / `_pending_delay_ttl` use `hasattr`+`del` at 3 sites instead of init-as-None | orchestrator.py:1626-1633, 1399-1402, 1124-1125 |
| 34 | `getattr(self, '_far_active_blocks', 0)` lazy-fallback redundant — reset() always sets it | orchestrator.py:2893-2898 |
| 38-40 | Inline imports inside hot path | orchestrator.py:573, 1180, 608, 1674, 3403, 3414 |
| 23 | `self._aec3_ree = ResidualEchoEstimator(...)` initialized at __init__:591 AND reset:1182 — same parameters | orchestrator.py:591 vs 1182 |
| 78 | `_enable_p53_trace` legacy P-path trace capture (block dead if #75 acts) | filters.py:484-500 |
| 77 | `_enable_kx_trace` legacy P-path trace state | filters.py:347-358 |

## Source-of-Truth Duplications

| Subject | Old (dead) | Live (canonical) |
|---|---|---|
| Subband NE detection | `AecConfig.subband_ne_detect_enabled` + 6 sub-fields (config.py:852-858) | `SuppressorConfig.subband_nearend_detection` (suppression_gain.py:185) |
| Dominant NE detection | `AecConfig.dominant_ne_detect_enabled` + 7 sub-fields (config.py:926-935) | `SuppressorConfig.dominant_nearend_detection` (suppression_gain.py:182) |
| Stationarity zeroing | `AecConfig.aec3_post_stationarity_zero_enabled` (DEPRECATED ALIAS per v3.21.6 P3) | `SuppressorConfig.echo_audibility.use_stationarity_properties` |
| `_shadow_advantage` | property at orchestrator.py:80 | `_shadow_advantage_p3f` instance at L2696 etc. — different definitions |

## Structure Smells

- `AEC.__init__` is ~700 lines (orchestrator.py:319-1021); sets >120 instance attributes; could extract `_init_delay() / _init_filter() / _init_aec3_post() / _init_diagnostics()`
- `process()` is ~1611 lines (L1555-3166); 12 clearly separable sub-sections per agent finding #49
- `_aec3_post()` is ~570 lines; trace block alone is ~200 lines (L3641-3843)
- `_reset_filter_derived_state()` is ~190 lines + 60-line docstring (L1214-1402); cleared-state list hand-maintained vs `__init__` defaults — drift-prone

## Suggested Tiered Action Plan

### Tier 1 — Trivial mechanical cleanups (low risk, ~30 min, byte-equal trivially preserved)

- Unused imports (8 sites)
- `_handle_delay_change_full` deletion (#14)
- `SubbandNlms` alias deletion (#79)
- Inline `import os` shadow in __init__:579 (#19)
- isinstance redundancies #45, #46
- Hoist `self.dtd_coherence.confidence *= 0.3` outside split (#48)
- Single `import dataclasses as _dc` to module-level (#40)

**Risk**: byte-equal trivially preserved (no semantic changes). Verify with `check_byte_equal.py` after.

### Tier 2 — Dead state cleanup (medium-low risk, byte-equal preserved if attributes truly unused)

- `final_error_power` / `final_error_power_sum` write-only (#25): ~8 LOC drop
- `_stat_far_hangover` (#26): ~3 LOC drop
- `confidence_history` (#27): ~2 LOC drop
- `_misadjustment_reset_done_count` + 3 config flags (#28): ~10 LOC
- `_misadjustment_fire_count` (#29): ~2 LOC

**Risk**: byte-equal preserved if grep confirms no external consumers. Verify each individually.

### Tier 3 — Dead config flag retirement (medium risk, requires bench-team check)

18 ResFilter-retired flags (table above) — drop config field + env hook + any docstring. ~150 LOC opportunity. **Verify with mingyu** that none is being toggled in active A/B runs / external scripts.

### Tier 4 — Legacy code path retirement (high risk, requires regression bench)

- 124-line legacy P-Kalman path in `filters.py:440-563` (#75) — under permanently-True `_use_aec3_h_error`. Plus the `_enable_kx_trace` and `_enable_p53_trace` legacy capture blocks (#77, #78).
- `_aec3_ree` double-init at L591 + L1182 dedupe (#23) — research substrate semantics may change

**Risk**: large blast radius; needs `check_byte_equal.py` + full 800-case bench after retirement.

### Tier 5 — Naming hygiene (high risk per `feedback_no_version_in_var_names.md`; touches state across init/reset)

Rename `_round3_*` / `_p3f_*` / `_p3h_*` / `_p4b_*` / `g_stage_*` / `filter_misadjustment_threshold_phase3` — closed-cycle prefixes still in live names. Each rename touches multiple files (init / reset / derived-state-reset / diag dict / external consumers).

**Risk**: highest; mechanical rename but each affected attribute must be checked for external consumers (test files, debug tooling, downstream NR / Audio_ALG).

### Tier 6 — Structural extractions (largest payoff, multi-day work, full regression suite)

- Extract `_default_diag_dict()` helper from 3 duplicated init/reset sites (#35)
- Extract `_init_delay() / _init_filter() / _init_aec3_post() / _init_diagnostics()` from monolithic `__init__` (#50)
- Break `process()` into 12 sub-method calls (#49)
- Extract `_capture_hf_chain_trace(locals_)` from `_aec3_post()` trace block (#52)

**Risk**: substantial refactor; requires full regression suite + 800-case bench.

## Recommendation

Pick Tier 1 + Tier 2 + Tier 3 (subject to bench-team verification) as a single "cleanup PR" against `feature/v3_22_optimization`. Estimated ~250 LOC removed, byte-equal preserved, no behavior change. Defers Tier 4-6 to a future dedicated refactor cycle (which can be its own plan).

Skip Tier 5 entirely until the next algorithm cycle — renaming closed-cycle attributes mid-cycle creates merge conflicts with any in-flight research substrate.

Open questions for user:

1. Are any of the v3.22 substrate flags (`e_stat_aware_ne_proxy_enabled`, `reverb_tail_dead_fallback_enabled`, `transparent_mode_enabled`) being used in bench-team A/B beyond v3.22's own audit scope? If not, we can also retire them in Tier 3 (or keep dormant per cycle close decision).

2. Cleanup PR target: against `feature/v3_22_optimization` (so the audit closes out v3.22 substrate-clean), or against `main` directly (since main is v3.21.6 + the cleanup is byte-equal, it could land cleanly)?
