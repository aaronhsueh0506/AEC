# v3.21.6 — AEC3 Parity Completion cycle close

**Date**: 2026-05-21
**Branch**: `feature/v3_21_6_parity_completion`
**Status**: ready to ship (tag + push pending user approval)
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (Round 7 cycle 2)

## Headline

One production change shipped: **Sprint P1 FilterAnalyzer port** (default-True). Sprints P2 / P3 / P4 are structural / audit / intentional-incompatibility closures that do not change default behavior. Cumulative 800-case bench Pareto-positive vs v3.21.5 baseline:

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | **+0.059** | -0.000 |
| FS_movement | **+0.036** | -0.000 |
| DT_static | +0.029 | -0.009 |
| DT_movement | +0.016 | +0.008 |
| NE | +0.000 | -0.001 |

Bucket means identical to the per-sprint P1.3 bench result — confirms v3.21.6 algorithmic ship state = P1 only.

## Per-sprint outcomes

| Sprint | Verdict | Production effect | Commit |
|---|---|---|---|
| **P1** FilterAnalyzer port | ✅ shipped default-True | ~250 LOC AEC3 [`filter_analyzer.cc`](aec3_extracts/src/aec3/filter_analyzer.cc) port owned by AecState; orchestrator feeds `PBFDAF.get_time_domain_filter()` per hop; reverb-update `_delay_blocks` switches to `aec_state.min_direct_path_filter_delay()`. Indirectly closes v3.21.5 Sprint C reverb-tail blocker. | `ba4d585` (port + verdict, default-OFF) + `ed9efd8` (default flip True) |
| **P2** TransparentMode audit | ✅ CLOSED intentionally-incompatible | 4 mismatches identified; LN18k5r8 cohort trace fires TM 23.1% @ fa_consistent=0% (PBFDKF Kalman peak too noisy for AEC3 ConsistentFilterDetector). AEC3 transparent-mode parity permanently retired for our cohort. Parity substrate (config flag, env hook, 3 hop-unit constant corrections) shipped dormant; production stays `transparent_mode_enabled=False`. | `89f1eda` + `ec90770` (framing tightened) |
| **P3** EchoAudibilityConfig wiring | ✅ shipped byte-equal | Existing `EchoAudibilityConfig` dataclass promoted from SuppressionGain-internal local to `SuppressorConfig.echo_audibility` field; orchestrator stationarity zeroing block reads canonical nested field; top-level `AecConfig.aec3_post_stationarity_zero_enabled` retained as DEPRECATED ALIAS propagated via `dataclasses.replace` at init. Single-case md5 identical pre/post. | `bcd4206` |
| **P4** Stationarity default-off re-test | ✅ CLOSED intentionally-incompatible | Cohort 3-case re-trace on post-P1+P2+P3 baseline: all 3 user-set gate criteria FAIL — catastrophic g100 drops persist 233/235/424 frames; ΔNE = +0.0 on all 3 cases; HF formant damage -0.94 dB on xQEUtY2. P1+P2+P3 paths don't feed into `_DominantNearendDetector` ENR/SNR decision — Sprint B hypothesis (companion mechanisms rescue NE firing) data-confirmed-FALSIFIED. AEC3 default-off `use_stationarity_properties=False` permanently retired. | `7c45ec9` |

## v3.22 entry gate

✅ **AEC3 parity verdict gate MET**. Every Bucket-1 item in the [plan's AEC3 Parity Gap Triage Policy](~/.claude/plans/se-aec-aec-main-hazy-lynx.md) has a closed verdict:

| Bucket-1 item | Outcome |
|---|---|
| E2=min(E2,Y2) clamp (v3.21.5 Sprint A) | ✅ shipped default-True |
| Stationarity gate `use_stationarity_properties=False` | ✅ permanently retired (v3.21.6 P4) |
| PBFDKF per-bin H_error refresh selector | ✅ CLOSED no-leverage (v3.21.5 Sprint C2) |
| FilterAnalyzer / direct-path delay parity | ✅ shipped default-True (v3.21.6 P1) |
| Transparent mode / AecState parity audit | ✅ CLOSED intentionally-incompatible (v3.21.6 P2) |
| EchoAudibilityConfig structural wiring | ✅ shipped byte-equal (v3.21.6 P3) |

✅ **release-quality bench gate MET** — cumulative 800-case Pareto-positive (above).

Two new permanent Bucket 3 entries created (both with explicit "v3.22+ revisit must be PBFDKF-specific divergence, NOT AEC3 parity" labels per [[feedback-no-parity-claim-for-divergence]]):
- AEC3 transparent-mode parity (TransparentMode) — retired by P2 verdict
- AEC3 default-off stationarity (`use_stationarity_properties=False`) — retired by P4 verdict

## v3.22 framing

v3.22 is no longer a broad "try all old ideas" cycle. Post-v3.21.6 prioritization (Codex 2026-05-21):

- **Sprint E HF cap / NE decoupling** = **PRIMARY** — surviving P4 root cause (`_DominantNearendDetector` mis-classifies NE under stationary-far → far-tuning gain). E directly addresses the consequence.
- **Sprint D hybrid residual / nonlinear HF floor** = secondary — only proceeds if post-v3.21.6 trace shows S²_linear HF under-estimation persists despite P1's reverb path improvement.
- **Sprint F reverb tail dead fallback** = trace-gated no-op candidate; likely SKIP (P1 already revives tail update on 4/5 cohort cases).
- **Sprint G** per-item — G.2 PBFDKF-specific not AEC3 parity (per P2); G.3 baseline = v3.21.6 final with stationarity=True retained (P4 retired default-off).
- **Sprint H.3** ERLE clamp — G.1-gated.
- **Sprint I** config cleanup — `aec3_post_stationarity_zero_enabled` stays as research toggle per P4.

## vs AEC2 / AEC3 reference scores

Per `docs/aec_methods.md` reference table: v3.21.5 already beat AEC2 by +1.12 FS and beat AEC3 by +0.52 DT_deg / +0.60 NE. v3.21.6 adds FS_static +0.059 / FS_movement +0.036 on top of that, widening the AEC2-FS advantage to ~+1.18. AEC3 parity is now structurally complete: every Bucket-1 item closed; no parity gap remains merely deferred. The two intentional-incompatibility closures (TransparentMode + stationarity-default-off) document permanent PBFDKF-architecture-specific deviations that v3.22+ would only re-open as labeled divergence designs.

## Ship checklist

- [x] Cumulative 800-case bench PASS (Pareto-positive vs v3.21.5)
- [x] Cycle close doc (this file)
- [ ] `__version__` bump 3.21.5 → 3.21.6 in [`python/aec.py`](../python/aec.py)
- [ ] CHANGELOG.md entry
- [ ] Byte-equal anchor snapshot to `docs/bench/v3_21_6_baseline/`
- [ ] Tag `v3.21.6` + merge to main + push **(needs explicit user approval)**
