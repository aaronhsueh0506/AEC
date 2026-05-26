# v3.21.7 Gate 2 — State-input parity arc verdict (2026-05-22)

**Trigger**: Gate 1 returned Cat C BLOCKED-STRESS (~26 DT cases at Δdeg ≤ −0.5; partition_summed_x2 candidate paused).

**Goal**: Determine whether the DT stress cluster is an AEC3 parity gap (returns to v3.21.x cycle) or a beyond-AEC3 design need (deferred to v3.22).

**Source references**: AEC3 extracts at [docs/aec3_extracts/src/aec3/](aec3_extracts/src/aec3/).

---

## Trace target #1 — SubtractorOutputAnalyzer parity

AEC3 [subtractor_output.h](aec3_extracts/src/aec3/subtractor_output.h) + [subtractor_output_analyzer.cc](aec3_extracts/src/aec3/subtractor_output_analyzer.cc):

```
struct SubtractorOutput {
  e_refined, e_coarse  // time-domain block (kBlockSize=64)
  s_refined, s_coarse  // echo estimates
  E_refined, E2_refined, E2_coarse  // spectral
  s2_refined, s2_coarse, e2_refined, e2_coarse, y2  // block sum-of-squares
  s_refined_max_abs, s_coarse_max_abs
};

// SubtractorOutputAnalyzer::Update emits:
refined_filter_converged   = e2_refined < 0.5  × y2 AND y2 > 50²·kBlockSize
coarse_filter_converged_strict  = e2_coarse < 0.05 × y2 AND y2 > 50²·kBlockSize
coarse_filter_converged_relaxed = e2_coarse < 0.3  × y2 AND y2 > 20²·kBlockSize
filter_diverged            = min(e2_refined,e2_coarse) > 1.5 × y2 AND y2 > 30²·kBlockSize

any_filter_converged       = refined OR coarse_strict   (per channel OR)
any_coarse_filter_converged = coarse_relaxed
all_filters_diverged       = AND of filter_diverged
```

Our implementation at [orchestrator.py:3320-3354](../python/modules/orchestrator.py#L3320-L3354):

```
_y2_time     = sum(near_end²)            over hop=160 samples (post-mic-HPF) ✓
_e2_refined  = sum(raw_output²)          over hop=160 samples ✓
_y2_threshold = 3.73e-4  (= 50²·64 / 32768² × 160/64)  ≈ AEC3-scale-equivalent ✓

_e2_coarse = (2·Σ|E[1:-1]|² + |E[0]|² + |E[-1]|²) / fft_size
  → Parseval reconstruction over fft_size=512 samples ✗ WINDOW MISMATCH

_refined_conv = _e2_refined < 0.5  × _y2_time AND _y2_time > _y2_threshold ✓
_coarse_conv  = _e2_coarse  < 0.05 × _y2_time AND _y2_time > _y2_threshold
_aec3_converged = _refined_conv OR _coarse_conv
```

### Findings

| # | Finding | Impact | Classification |
|---|---|---|---|
| 1a | `_y2_time` block window = hop (160) ≠ AEC3 kBlockSize (64), but `_y2_threshold` scaled by (160/64) ✓ | Scale-equivalent (assumes stationary signal) | OK |
| 1b | `_e2_refined` window = hop (160), matches `_y2_time` window ✓ | Refined ratio dimensionally consistent | OK |
| **1c** | **`_e2_coarse` window = fft_size (512) via Parseval, vs `_y2_time` window = hop (160)** | **Coarse e2 is ~3.2× larger than hop-equivalent (512/160). Effective threshold 0.05 → 0.016 — `coarse_conv` fires ~3× LESS than AEC3 intended.** | **v3.21.x parity gap** |
| 1d | Missing `coarse_filter_converged_relaxed` (e2_coarse < 0.3 × y2 AND y2 > 20²·kBlockSize) | AEC3 emits this as separate signal `any_coarse_filter_converged` for downstream consumers | **v3.21.x parity gap** |
| 1e | Missing `all_filters_diverged` (min(e2_refined, e2_coarse) > 1.5 × y2 AND y2 > 30²·kBlockSize) | AEC3 uses this for echo-path-change detection + filter reset trigger | **v3.21.x parity gap** |
| 1f | `near_end` source = post-mic-HPF ✓ | Matches AEC3 capture-mic semantics | OK |

### Stress-cluster implication for 1c

XRTnTUjU damaged frames 239-263 show `coarse_conv=0%` despite refined_conv=20%. With the corrected 0.05 threshold (rather than effective 0.016), coarse_conv would fire more often. **However, fixing 1c alone may not help the stress cluster** — the gate-3 latch would still fire on the very first coarse_conv. The window mismatch is a parity bug but not the proximal stress-cluster cause.

---

## Trace target #2 — Linear output selection parity (UseRefinedOutput + FormLinearFilterOutput)

AEC3 [echo_remover.cc:112-147](aec3_extracts/src/aec3/echo_remover.cc):

```
bool UseRefinedOutput(subtractor_output) {
  // Prefer coarse when coarse residual significantly cleaner AND signal present:
  if (e2_coarse < 0.9 × e2_refined && y2 > 30²·kBlockSize &&
      (s2_refined > 60²·kBlockSize || s2_coarse > 60²·kBlockSize)) return false;

  // Refined diverged (refined e2 worse than both coarse e2 AND y2):
  if (e2_coarse < e2_refined && y2 < e2_refined) return false;

  return true;  // default: refined
}

void FormLinearFilterOutput(refined_filter_output_last_selected,
                            use_refined_output, subtractor_output, output) {
  SignalTransition(
    refined_filter_output_last_selected ? subtractor_output.e_refined
                                        : subtractor_output.e_coarse,
    use_refined_output ? subtractor_output.e_refined
                       : subtractor_output.e_coarse,
    output);
}
```

`EchoRemoverImpl::ProcessCapture` (echo_remover.cc:430-444) then routes the chosen output to AecState + RES.

Our implementation: grep `use_refined|UseRefined|coarse.*output|FormLinearFilter` across `python/modules/` returns **EMPTY** for output-selection. `raw_output` in `process()` is **only** from `self.filter.process(...)` (main / refined filter). No coarse fallback. No SignalTransition.

### Findings

| # | Finding | Impact | Classification |
|---|---|---|---|
| **2a** | **`UseRefinedOutput` mechanism is COMPLETELY MISSING** | AEC3 `echo_remover.cc:430-444` selects per-frame when `use_coarse_filter_output_` outer gate is ON. **AEC3 production default for that gate is FALSE** — production always uses refined. UseRefinedOutput is a field-trial mechanism, not production parity. v3.21.8 implemented + 12-case bench (2026-05-22): on stress no-clean-convergence cases (XRTnTUjU et al.), shadow NLMS is also un-converged → spectral substitution feeds RES `error ≈ near` → SuppressionGain catastrophically suppresses DT speech. See [docs/v3_21_8_12case_verdict.md](AEC/docs/v3_21_8_12case_verdict.md). | **DOWNGRADED 2026-05-22: AEC3 field-trial substrate, NOT production parity gap.** Code preserved default-OFF; not shippable. |
| **2b** | **`FormLinearFilterOutput` smooth transition between e_refined ↔ e_coarse missing** | AEC3 does sample-by-sample crossfade to avoid glitches at switch boundary; our pipeline has no concept of switch | **v3.21.x parity gap (depends on 2a)** |

### Stress-cluster implication for 2a/2b

XRTnTUjU + stress cluster mechanism: parity fix (partition_summed_x2) cleans linear residual modestly → triggers brief refined_conv → convergence_seen latches → usable_linear stays True for whole utterance → SuppressionGain trusts S²/E² over-aggressively → DT damage.

**If AEC3's UseRefinedOutput were present**: when refined filter has high residual on damaged DT windows, AEC3 would switch to e_coarse output (shadow filter, less aggressive). The downstream AecState would see the LOWER-residual coarse output rather than the refined latch-trigger.

The nores-LF-reduction-on-stress-cluster observation (parity fix cleans linear filter ON the same cases that regress on Δdeg) is consistent: parity fix IS making the linear filter cleaner, but with no coarse-fallback to capture that cleaner-than-refined moments, the state machine over-commits to a brittle convergence signal.

**This is the primary candidate for fixing the stress cluster.** Implementing UseRefinedOutput + FormLinearFilterOutput is direct AEC3 parity — belongs in v3.21.x, possibly v3.21.8.

---

## Trace target #3 — FilterMisadjustment / ScaleFilter parity

AEC3 [subtractor.cc:336-360](aec3_extracts/src/aec3/subtractor.cc):

```
void FilterMisadjustmentEstimator::Update(SubtractorOutput& output) {
  // Accumulates inv_misadjustment ~ e2_refined / y2
  update = output.e2_refined / output.y2;  // (or similar; e²/y²)
  if (update < inv_misadjustment_ || overhang_ > 0) {
    inv_misadjustment_ += 0.1 × (update - inv_misadjustment_);
    overhang_ = ...
  }
}

// Subtractor::Process (subtractor.cc:239-247):
filter_misadjustment_estimators_[ch].Update(output);
if (IsAdjustmentNeeded()) {                          // inv_misadjustment_ > 10
  float scale = GetMisadjustment();                  // = 2 / sqrt(inv_misadjustment_)
  refined_filter.ScaleFilter(scale);                 // scale W DOWN
  ...
  filter_misadjustment_estimators_[ch].Reset();
}
```

AEC3 direction: large `e2_refined/y2` → `inv_misadjustment` grows → if > 10 → scale W **DOWN** (to reduce echo estimate magnitude).

Our implementation per [memory project_v3_23_g0_verdict](../../.claude/projects/-Users-mingyu-Desktop-novatek-SE/memory/project_v3_23_g0_verdict.md): `_update_misadjustment_estimator` ([orchestrator.py:190-263](../python/modules/orchestrator.py#L190-L263)) uses `echo_psd/(far_psd × ERL)`. AEC3 reference verdict (v3.23 G.0 A.1) classified the legacy implementation as **opposite-direction** (fires on SMALL ratio → scale UP). On the G.0 7-case cohort, the legacy estimator fired 0/20188 frames → `INFORMATIVE_NULL`.

### Findings

| # | Finding | Impact | Classification |
|---|---|---|---|
| 3a | Formula direction inverted vs AEC3 (per v3.23 G.0 A.1 verdict, already established) | Estimator fires on SMALL ratio not LARGE; effectively dormant on most cases (0/20188 G.0 fires). Misadjustment-protection mechanism non-functional. | **v3.21.x parity gap (re-confirmed; not new this cycle)** |
| 3b | Trigger threshold + scale formula need separate audit on a stress-cluster cohort | G.0 cohort had 0 fires → cannot evaluate AEC3-aligned trigger on this evidence | Re-trace on stress-cluster cohort if 3a fixed |

### Stress-cluster implication for 3a/3b

If misadjustment estimator were AEC3-direction-correct AND fired on stress-cluster cases (high e2_refined/y2 during refined divergence), it would scale W DOWN → reduce echo estimate aggression → smaller filter output → less downstream over-suppression.

Secondary fix relative to #2a. The G.0 cohort INFORMATIVE_NULL on the legacy was per the OLD cohort; with v3.21.7 stress cohort, re-evaluation is warranted post-3a-fix.

---

## Gate 2 Summary

| # | AEC3 mechanism | Our code | Classification | Stress-cluster relevance |
|---|---|---|---|---|
| **1c** | coarse e2/y2 ratio window consistency | Window mismatch (e2 over 512, y2 over 160) | **v3.21.x parity gap** | Secondary — affects coarse_conv fire rate but latch still fires on first hit |
| **1d** | `coarse_filter_converged_relaxed` | Missing | **v3.21.x parity gap** | Indirect — downstream consumers may behave differently |
| **1e** | `all_filters_diverged` | Missing | **v3.21.x parity gap** | Indirect — echo-path-change detection signal missing |
| **2a** | `UseRefinedOutput` per-frame selection | **Completely missing** | **v3.21.x parity gap** | **PRIMARY ROOT CAUSE CANDIDATE** for stress cluster |
| **2b** | `FormLinearFilterOutput` smooth transition | Missing (depends on 2a) | **v3.21.x parity gap** | Required if 2a fixed |
| **3a** | `FilterMisadjustmentEstimator` direction | Opposite direction (re-confirmed from v3.23 G.0) | **v3.21.x parity gap** | Secondary — independent protection layer |

### Verdict

**ALL THREE TRACED TARGETS = v3.21.x parity gaps.** None are beyond-AEC3 design need.

**Classification of stress-cluster mechanism**: AEC3 parity gap, NOT beyond-AEC3 design. The state-input parity arc continues within v3.21.x. v3.21.7 candidate (partition_summed_x2) is paused; the parity-fix work continues with **#2a UseRefinedOutput** as the highest-priority next sub-patch (likely **v3.21.8**), with #1c (coarse window fix) and #1d/1e (missing signals) as supporting cleanup.

### Recommended ship-track ordering (post-Gate-2)

1. **v3.21.8 candidate** — implement UseRefinedOutput + FormLinearFilterOutput (gap #2a + #2b). Default-OFF flag (`use_refined_output_selection_for_linear_path`). Bench 12-case cohort first; if stress cluster recovers → 800-case A/B vs v3.21.6 baseline.
2. **v3.21.9 candidate (or bundled)** — fix coarse e2 window mismatch (#1c). Add `_e2_coarse_block` over hop or scale Parseval result by hop/fft_size. Independent of UseRefinedOutput; can be A/B'd separately.
3. **v3.21.10 candidate** — add missing coarse_relaxed + all_filters_diverged signals (#1d + #1e). Wire into AecState consumers.
4. **v3.21.11 candidate (deferred)** — fix FilterMisadjustment direction (#3a). Re-evaluate on stress cohort.
5. **partition_summed_x2 (current v3.21.7)** — resumes evaluation AFTER #2a lands. The Pareto picture should improve once coarse fallback rescues stress cases.

### Forbidden actions during Gate 2 → next sub-patch transition

- Do NOT delete / archive / merge research branches (Gate 1 was Cat C, not Cat A)
- Do NOT implement V2 FilterAnalyzer-AND gate-3 (still v3.22 diagnostic only, not parity)
- Do NOT mix Route A / G.0 trace infra into v3.21.8 production code
- Do NOT bundle parity gaps into one mega-patch; each lands isolated with its own A/B

### Memory updates

- Update `project_aec3_refined_filter_update_gain_x2_parity`: status = paused; v3.21.7 evaluation re-runs after v3.21.8 (UseRefinedOutput) lands
- Update `project_usable_linear_gate3_latch_bug`: confirm UseRefinedOutput-missing is the upstream cause; gate-3 latch is the symptom path
- New memory: `project_aec3_use_refined_output_missing` — v3.21.8 candidate
