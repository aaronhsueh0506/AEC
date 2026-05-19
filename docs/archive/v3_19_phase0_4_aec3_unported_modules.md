# v3.19 Phase 0.4 — AEC3 unported modules scope decision (2026-05-16)

**Status**: DECIDED — split decision per module:
- **SuppressionGain**: SURVEY folded into Phase 1.1; PORT deferred to v3.20+
- **ResidualEchoEstimator**: SURVEY deferred (no Phase 1 overlap); PORT
  deferred to v3.20+
**Verdict**: Phase 1.1 RES branch inventory absorbs an additional
SuppressionGain side-by-side reading. No new sub-phase. ResidualEcho
stays untouched in v3.19.

## 1. Origin

v3.18 Phase 0 fetched 6 AEC3 modules to
[docs/aec3_extracts/](aec3_extracts/). Mapping docs landed for 4 of
them; **2 got partial mapping but no port arc**:

| Module | Mapping doc | v3.18 status |
|---|---|---|
| `subband_nearend_detector.cc` | [aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md) | Phase D-γ (CLOSED) |
| `dominant_nearend_detector.cc` | [aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md) | Phase D-γ (CLOSED) |
| `reverb_model.cc` | [aec3_reverb_mapping.md](aec3_reverb_mapping.md) | unported (v3.17 B.2 substrate) |
| `reverb_decay_estimator.cc` | [aec3_reverb_mapping.md](aec3_reverb_mapping.md) | unported (v3.17 B.2 substrate) |
| **`suppression_gain.cc`** (514 lines) | [aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md) | unported, **partial mapping only** |
| **`residual_echo_estimator.cc`** (432 lines) | [aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md) | unported, **partial mapping only** |

The two un-arc'd modules are the focus of Phase 0.4.

## 2. Module overlap analysis with v3.19 Phase 1 work

### 2.1 SuppressionGain.cc — HIGH overlap

AEC3 `SuppressionGain::GetGain`
([aec3_residual_pipeline_mapping.md §3](aec3_residual_pipeline_mapping.md#3-suppression-gain-structure-aec3))
is the per-bin gain envelope producer:

```
DominantNearendDetector → IsNearendState
LowerBandGain():
    GetMaxGain(): max_gain[k] = clamp(last_gain[k] * inc_factor, ...)
    GetMinGain(): min_gain[k] = noise_limit / weighted_residual_echo[k]
    SolveForGainSquares(...): per-bin Wiener-like gain in [min, max]
GainParameters::SetConfig(): per-bin LF→HF interpolated thresholds
```

Our equivalent is **spread across 4 ResFilter stages**:
- `_stage_gain_compute` (softgate_emr / spectral_floor)
- `_stage_gain_postprocess` (epc_dt_cap / quiet_mask / 3bin_smooth /
  hf_cap / pre_temporal)
- `_stage_temporal_smoothing` (last_gain × inc_factor approximation)
- `_stage_noise_floor_cng` (noise_limit equivalent)

**Phase 1.1 scope**: enumerate ResFilter branches reading
`filter_converged`, classify per-branch downstream effect on
`effective_g_min` / `res_gain_mean` / Wiener gain output.

Direct overlap with SuppressionGain. Phase 1.1 reading SuppressionGain
side-by-side surfaces:
- Which of our stages corresponds to AEC3's `min_gain` / `max_gain`
  envelope (informs per-branch flag naming + design alignment)
- AEC3's per-bin GainParameters anchors (LF/HF interpolation) — useful
  context for v3.20+ per-band axis (Phase 0.3 D-γ retry)
- AEC3's `nearend_smoother` (averaging across 8 frames) vs our scalar
  `effective_dt` / `_filter_converged` — informs Phase 2 retry logic

### 2.2 ResidualEchoEstimator.cc — LOW overlap

AEC3 `ResidualEchoEstimator::Estimate`
([aec3_residual_pipeline_mapping.md §2](aec3_residual_pipeline_mapping.md#2-residual-echo-estimator-structure-aec3))
is the R2 producer with 3 modes:
- SaturatedEcho: `R2 = Y2` (mic passthrough)
- UsableLinearEstimate: `R2 = S2_linear / Erle + reverb_tail`
- non-linear fallback: `R2 = X2_gated * echo_path_gain + reverb_tail`

Our `ResFilter._stage_residual_model` already produces equivalent
`residual_echo_psd`. The 3-mode switch is structurally what
v3.18 P58 Phase 2 attempted (CLOSED FAIL: AEC3 ERLE-divide produces
tiny residual when FS-converged → Wiener gain ≈ 1 → no
post-suppression; legacy 4-cap chain was load-bearing on FS).

**Phase 1 does not touch `_stage_residual_model`**. Phase 1 acts on
the gain pipeline downstream. ResidualEchoEstimator port would
re-open the P58 closure — out of v3.19 scope.

## 3. Decision

### 3.1 SuppressionGain — fold SURVEY into Phase 1.1 (no new sub-phase)

Phase 1.1 scope expands from:

> Inventory ResFilter branches reading `filter_converged`: per-branch
> downstream effect on effective_g_min / res_gain_mean / Wiener gain
> output

To:

> Inventory ResFilter branches reading `filter_converged`: per-branch
> downstream effect on effective_g_min / res_gain_mean / Wiener gain
> output. **Side-by-side annotate AEC3 SuppressionGain
> equivalent** (per-stage AEC3 line citation; identify branch ↔
> SuppressionGain section mapping) for v3.20+ port reference.

LOE delta: +1 sprint to Phase 1 total (1.1 grows 1 → 2 sprints; 1.2-1.7
unchanged).

**Rationale**: doc-only addition. Cost is small. Output strongly
informs both Phase 1 design and v3.20+ SuppressionGain port arc.
No risk to Phase 1 hard bar.

### 3.2 SuppressionGain PORT — v3.20+ backlog (entry already present)

Backlog entry stays:

> AEC3 suppression_gain + residual_echo_estimator port | 4-6 sprints
> | per Phase 0.4 if NOT folded into Phase 1.1 | v3.18 Phase 0 fetch

Update entry note: "Port deferred to v3.20+ even with Phase 1.1
SURVEY fold-in — port is a mechanism replacement (Wiener-like envelope
+ DominantNearendDetector co-design) requiring its own design lock
and 800-case ship arc, separate from Phase 1's per-branch flag
discipline."

### 3.3 ResidualEchoEstimator — defer fully to v3.20+

- No Phase 1.1 overlap.
- Port re-opens v3.18 P58 closure pattern (FS regression risk).
- Backlog entry stays.

## 4. v3.19 Phase 1.1 scope update

Updated sprint table for Phase 1 in plan:

| Sprint | Action | Hard bar |
|---|---|---|
| 1.1a | Inventory ResFilter branches reading `filter_converged`: per-branch downstream effect on effective_g_min / res_gain_mean / Wiener gain output | doc only |
| 1.1b | Side-by-side annotate AEC3 SuppressionGain equivalent: per branch cite AEC3 line ranges, identify ↔ section mapping | doc only |
| 1.2 | Design: per-branch flag (`c_e_branch_<name>_use_fq_usable`); 5-10 branches expected | design lock + 60% confidence check |
| 1.3 | Implement per-branch flags; default OFF; 5-case byte-equal flag-OFF | byte-equal md5 PASS |
| 1.4 | Single-branch sweep: enable each branch alone on 60-case AECMOS; rank by (Δdeg_DT − \|Δecho_FS\|) | ranking doc |
| 1.5 | Combine top-ranked branches; 60-case A/B | DT Δdeg ≥ +0.020 AND FS Δecho ≥ −0.010 |
| 1.6 | If 1.5 PASS: nores listen verification on qNvSMyU + 0I0XMl3M + xrtntuju 5-clip + DT NE-preserve cases | no audible regression |
| 1.7 | 800-case + ship gate (user re-auth) | hard bar all PASS |

Total Phase 1: 9-13 sprints (was 8-12).

## 5. Cross-references

- [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md) — partial mapping doc for both modules
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — Phase 1
  motivation
- [project_p58_phase2_verdict](../../../.claude/projects/-Users-mingyu-Desktop-novatek-SE/memory/project_p58_phase2_verdict.md)
  — P58 ResidualEchoEstimator port closed FAIL pattern
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` — v3.19 cycle plan
  (Phase 1 sprint table to be updated per §4)
