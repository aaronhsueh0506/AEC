# v3.21.12 — RefinedFilterUpdateGain input-parity audit plan (ACTIVE 2026-05-22)

**Scope**: AEC3 alignment / parity (v3.21.x line). NOT v3.22 optimization.
**Status**: TRACE PLAN, not implementation. Code added is default-OFF.
**Supersedes**: #1d / #1e downstream signal arcs paused for this round per user directive.

## Why this target

Code reading (Explore agent 2026-05-22) confirms a structural input mismatch in our `_update_weights_aec3()`:

### AEC3 reference

[`docs/aec3_extracts/src/aec3/refined_filter_update_gain.cc:103-107`](aec3_extracts/src/aec3/refined_filter_update_gain.cc):

```cpp
// mu = H_error / (0.5 * H_error * X2 + n * E2)
for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
  if (X2[k] >= current_config_.noise_gate) {
    mu[k] = H_error_[k] /
            (0.5f * H_error_[k] * X2[k] + size_partitions * E2_refined[k]);
  } else {
    mu[k] = 0.f;
  }
}
```

Key facts:

- `X2[k]` is from `render_buffer.SpectralSum(...)` (`subtractor.cc` `Process` call site) → partition-summed `Σ_p |X_buf[p][k]|²`.
- `E2_refined[k]` is from `SubtractorOutput.E2_refined` → **current-block** per-bin spectrum of the post-filter error `e_refined` (FFT-ed inside `Subtractor::Process` then `Spectrum()` per bin, line 252-257 of subtractor.cc). It is **NOT** a smoothed PSD.
- `size_partitions` is the number of frequency-domain partitions of the refined filter (call site passes `refined_filters_[ch]->SizePartitions()` = our `n_partitions`).

### Our Python deviations

[`python/modules/filters.py:602-616`](python/modules/filters.py) `_update_weights_aec3`:

```python
if self._use_partition_summed_x2_for_h_error_gain:
    X2 = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)   # AEC3 parity
else:
    X2 = (np.abs(X_latest) ** 2).astype(np.float32)                  # latest partition only

denom_aec3 = (
    np.float32(0.5) * self.H_error_per_bin * X2
    + n_part * self._error_psd                                       # ← smoothed EMA
    + delta32
)
```

Mismatch summary:

| Input | AEC3 | Ours (default) | Status |
|---|---|---|---|
| `X²[k]` | `render_buffer.SpectralSum` = Σ_p `|X_buf[p][k]|²` (partition-summed) | latest partition only `|X_buf[curr_p][k]|²` | already gated behind `use_partition_summed_x2_for_h_error_gain` (v3.21.7, paused Cat C BLOCKED-STRESS) |
| `E2_refined[k]` | current-block `|error_spec|²` (no smoothing) | `self._error_psd` = 0.95 EMA of `|error_spec|²` (`filters.py:435`, `_alpha_r=0.95`, init 1e-2) | **NEW MISMATCH — not gated by any flag yet** |
| `n_partitions` | `refined_filters_[ch]->SizePartitions()` | `n_part` (orchestrator) | parity OK |
| `delta` | implicit via `noise_gate` config | `delta32` additive in denom + noise_gate gate above | parity OK (both function as silent-far floor) |
| `noise_gate` | `X²[k] >= current_config_.noise_gate` zeroes mu | `X² >= NOISE_GATE_POWER_FLOAT` zeroes mu (filters.py:629) | parity OK |
| `H_error[k]` decay | `H_error -= 0.5 * mu * X² * H_error` per-bin | same (filters.py per-bin path) | parity OK |
| `H_error[k]` refresh | per-bin: `E²_refined ≤ E²_coarse → leakage_converged * erl`, else `leakage_diverged * erl` | per-bin (filters.py:678-707) | parity present but uses `|error_spec|²` (instantaneous) vs `_e2_coarse_per_bin` — **also instantaneous-not-smoothed; check if `_e2_coarse_per_bin` follows AEC3 semantics** |

### Why this matters — mechanism hypothesis

`self._error_psd` is heavily smoothed (α = 0.95 → 95-sample time constant ≈ 200 ms at 10-ms hop). The denominator term `n_part × _error_psd`:

- **Under-reacts to a sudden divergence**: when refined PBFDKF locally diverges (e.g. echo-path change), `|error_spec|²` spikes but `_error_psd` lags ~200 ms behind. Smaller-than-actual denominator → **mu OVERSHOOTS**, the adaptation step is too large → divergence persists or worsens before the EMA catches up.
- **Over-reacts to a sudden cleaning**: when residual drops suddenly, `_error_psd` stays high → denominator stays high → **mu under-shoots**, adaptation slows. Convergence is slower than AEC3.

AEC3 uses the CURRENT block's `E2_refined` so the denominator tracks the actual residual instantaneously — the K-step adaptation matches the actual signal regime in real-time, which is exactly what a Kalman-filter-style update is supposed to do.

### Tie-in with v3.21.7 Cat C stress regression

v3.21.7 partition_summed_x2 ON: refined PBFDKF locally diverges on no-clean-convergence stress (XRTnTUjU cond2 = 37% frames `e2_coarse < e2_refined AND y2 < e2_refined`).

Possible root cause: the smoothed denominator interacts pathologically with partition_summed_x2:

- Partition-summed X² is larger than latest-X² (more energy in denom). This is correct AEC3 parity for the X² term.
- But our smoothed `_error_psd` lags behind on the residual spike that follows.
- Combined: denominator's X² term scales up correctly, but the E² term doesn't react fast enough → mu becomes mis-balanced → adaptation overshoots → divergence persists.

If correct, then **C (raw E2_refined only)** or **D (both)** should reduce refined-divergence frame count vs v3.21.7 B (partition_summed_x2 only).

## Forbidden actions (carried over)

- Do NOT enable production-default change; both flags stay default-OFF.
- Do NOT revive UseRefinedOutput (v3.21.8) as ship candidate.
- Do NOT use shadow as AEC3 coarse-output fallback in any wiring.
- Do NOT add state gate-3 counter / FilterAnalyzer-AND / shadow-converged precondition / PBFDKF-specific logic. These are not AEC3 parity.
- Do NOT modify AEC3 thresholds (noise_gate / leakage_converged / leakage_diverged / etc.).
- Do NOT touch `_alpha_r` smoothing coefficient itself (that would be intentional divergence — the parity question is "which signal to plug in", not "how to smooth it").
- Do NOT pivot to v3.22; this is parity, belongs in v3.21.x.
- Do NOT run 800-case until trace + 12-case gate pass.
- Do NOT run /simplify or branch cleanup.

## Implementation plan

### Step 1 — default-OFF flag (this round)

Add `use_current_e2_refined_in_h_error_denominator: bool = False` to `AecConfig`. When True, `_update_weights_aec3` uses **current-block raw** `|error_spec|²` in the denominator instead of smoothed `self._error_psd`. When False (default): byte-equal to v3.21.6.

PBFDKF object exposes the flag via `_use_current_e2_refined_in_h_error_denominator` attribute set from config in `__init__`.

The flag is named in mechanism terms (no version number / no design-iteration suffix) per [feedback_no_version_in_var_names.md].

Optional companion (NOT shipping this round, audit-only):
- Trace counters: `frames_total`, `refined_diverged_frames` (`|error_spec|² mean > y² mean` per hop), per-bin LF/MF/HF aggregates of `|error_spec|² / _error_psd` (ratio of instant to smoothed).

### Step 2 — env hook

`AEC_CURRENT_E2_REFINED_DENOM=1` → `config_overrides['use_current_e2_refined_in_h_error_denominator'] = True`.

### Step 3 — byte-equal sanity (mandatory pre-trace)

Render 7-case cohort with flag OFF. Compare MD5 to v3.21.6 anchor (or to current v3.21.6 baseline render with all other parity flags OFF including the new one). Must be 7/7 PASS before proceeding.

### Step 4 — A/B/C/D trace cohort

Cohort: 7 cases (existing v3.21.8 cohort).

- A: baseline — all v3.21.x flags OFF (this is current production-equivalent under intended HPF policy).
- B: partition_summed_x2 only (`AEC_PARTITION_SUMMED_X2=1`).
- C: current E2_refined denominator only (`AEC_CURRENT_E2_REFINED_DENOM=1`).
- D: both flags ON (`AEC_PARTITION_SUMMED_X2=1 AEC_CURRENT_E2_REFINED_DENOM=1`).

For each variant, on each case, collect:

#### Audit metrics (primary)

1. **Output `_ours_nores.wav`** for LF/MF/HF artifact energy comparison (`*_nores.wav` band-energy ratios vs mic; smaller LF ratio = less artifact).
2. **`refined_diverged_frames`**: count of hops where time-domain `e²_refined > y²` (i.e. AEC3 SubtractorOutputAnalyzer would say "refined diverged"). Per-case rate.
3. **`mu_lf`**: mean `mu` value across LF bins (0-1 kHz) per hop, averaged over case.
4. **`H_error_lf`**: mean `H_error_per_bin` LF per hop, averaged over case.
5. **`X2_lf` / `E2_refined_lf` / `_error_psd_lf`**: mean per-band per-hop, to show the magnitude relationship that drives the denominator.
6. **`W_lf` / `dW_lf` (W update magnitude)**: norm of LF partition weights + per-hop weight delta norm — directly tests whether C/D stabilises adaptation.

#### Downstream diagnostics (secondary, do NOT gate decision)

7. `usable_linear` / `refined_conv` / `coarse_conv` / `convergence_seen` per-hop fire rates. Same per-case rate.
8. `_aec3_misadj_trace` (from v3.21.10) — does the raw-E²-denominator change the inv_misadjustment statistics?
9. `_coarse_relaxed_trace` (from v3.21.11) — for sanity; expected unchanged.

### Step 5 — decision gate

Apply IN ORDER:

1. **nores LF metric**: if C or D reduces nores LF artifact ratio on the nores-artifact case (whatever it is — XRTnTUjU is the candidate per v3.21.7 evidence) vs A, proceed. If neither does, close as no-op.
2. **XRTnTUjU stress state**: refined_diverged_frames count on XRTnTUjU under C/D vs B vs A. If C or D < B AND C or D ≤ A, the parity fix helps both nores AND stress. If C/D ≥ B (no improvement on stress despite nores improvement), this is the same Cat C class as v3.21.7. Stop without AECMOS — Cat C is paused, not shipped.
3. **12-case AECMOS** (only if both 1 and 2 pass): render full cohort A/B/C/D, score AECMOS, verify normal-bucket non-regression + worst-N stress acceptable. Single stress case does not kill candidate; cluster does (per Gate 1 verdict gate framing).

### Step 6 — verdict

| Outcome | Decision |
|---|---|
| C/D improves nores + reduces XRTnTUjU stress refined-diverged frames | **SHIP CANDIDATE** for v3.21.x (after 12-case AECMOS confirms). Likely supersedes v3.21.7 Cat C status. |
| C/D no-op (same divergence frames, same nores) | **CLOSE NO-OP**. Re-open #1d/#1e arcs. |
| C/D regresses nores or worsens stress | **CLOSE REJECTED**. Re-open #1d/#1e. |
| C improves nores but worsens stress (or vice versa) | **PAUSED SUBSTRATE**. Document trade-off, do not enable by default. Re-open #1d/#1e. |

## Expected conclusion format

For each variant a single-line per-case table:

```
case | nores_lf_ratio | refined_div_rate | mu_lf | H_err_lf | X2_lf | E2_lf | _err_psd_lf | refined_conv | coarse_conv | usable_linear
```

Plus per-band aggregates (LF / MF / HF) and per-bucket aggregates (stress / normal DT / FS).

Final write-up: `docs/v3_21_12_refined_filter_update_gain_input_parity_verdict.md` with:

1. EXACT / APPROX / WRONG verdict on X² and E²_refined inputs vs AEC3 (with code-line citations).
2. Per-case trace table for A/B/C/D.
3. nores artifact spectral evidence (band-energy ratios).
4. XRTnTUjU refined_diverged_frame rate per variant.
5. Recommendation: ship / paused substrate / close no-op.

## Memory anchors in force

- `project_aec3_refined_filter_update_gain_x2_parity` — v3.21.7 partition_summed_x2 origin doc + Gate 2 evidence
- `project_usable_linear_gate3_latch_bug` — state arc, do NOT touch this round
- `project_xrtntuju_dt_static_stress` — stress classification (XRTnTUjU has no clean FS preamble)
- `project_aec_hpf_lock` — intended HPF policy (mic ON / ref OFF)
- `feedback_holistic_trace_before_change` — trace-evidence-first before code change
- `feedback_no_version_in_var_names` — flag name uses mechanism descriptor, not "_v2"
