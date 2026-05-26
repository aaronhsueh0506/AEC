# v3.21 Residual R0 Full Alignment Trace

**Status**: COMPLETE (2026-05-27)

## Scope

Targeted attribution trace for the two 12-case gate failures:
- **xFk7** (DT_mvmt): catastrophic Δdeg −0.988 at M_full_delay
- **9xjhi** (FS_static): nores LF improved −6 dB but AECMOS echo collapsed −2.211

Variants: M0 / M_R0 / M_D / M_full / M_full+R0.2 / M_full+R0.3_fixed / M_full+R0.4 / M_full+R0

## AEC3 Reference Status

`bin/aec3_cli` is a behavioral AEC3 reference binary (arm64 Mach-O, compiled from WebRTC source).

**Confirmed**: NOT a source-parity claim. "AEC3 PASS/FAIL" requires running `bin/aec3_cli` on
the same 12 cases and comparing AECMOS. Without those numbers, this doc can only claim
"source-code parity for specific flags" — not behavioral AEC3 equivalence.

AEC3 reference NOT RUN; source parity only.

Run: `python3 python/v3_21_aec3_reference_aecmos.py` for 12-case AEC3 vs M0 AECMOS comparison.

## R0 Flags Implemented (2026-05-27)

| Flag | Default | Source | Classification |
|------|---------|--------|----------------|
| `use_aec3_residual_noise_gate` | OFF | `echo_model.noise_gate_power=27509.42f` (int16²) | R0.2 Class B |
| `use_aec3_echo_gen_power_window` | OFF | `residual_echo_estimator.cc` EchoGeneratingPower delay-centered window | R0.3 Class A |
| `use_aec3_erle_reverb_quality` | OFF | `FullBandErleEstimator::GetInstLinearQualityEstimates` (`fullband_erle_estimator.cc`) | R0.4 Class A |

### R0.2 — Residual noise gate unit audit

**Source**: AEC3 `echo_canceller3_config.h` `EchoModel::noise_gate_power = 27509.42f`.
**Unit**: int16² scale — same unit as `far_psd = float_spec² × 32768²` in our residual path.
The AEC3 value `27509.42` is already in the float PSD int16²-referenced scale (NOT `psd_int16_to_float`).
Python bug: default `noise_gate_power = 27509562.0` (1000× too large).
Fix: `RESIDUAL_NOISE_GATE_POWER = 27509.42` in `aec3_scale.py` — int16² verbatim, no conversion.

**Filter path is separate**: The filter weight-update gate (20075344 int16² → `FILTER_NOISE_GATE_POWER_FLOAT = 0.01870`)
is a completely independent constant. Do NOT mix R0.2 residual gate with the filter gate fix.

### R0.3 — EchoGeneratingPower strict AEC3 delay-centered window

**Prior implementation (pre-fix)**: `pre=1, post=1` with a recent-N ring buffer (not delay-centered).
Stored last N frames and took element-wise max — always picks most recent frames regardless of
acoustic delay. This is NOT AEC3 `residual_echo_estimator.cc` parity.

**AEC3 source** (`residual_echo_estimator.cc:133-165`):
```
EchoGeneratingPower:
  idx_start = max(0, filter_delay_blocks - pre_blocks)
  idx_stop  = filter_delay_blocks + post_blocks
  x2 = element-wise max over render_buf[idx_start : idx_stop+1]
```
The window is centered on `filter_delay_blocks` — the acoustic echo path delay in blocks.

**Fix (2026-05-27)**:
- `estimate()` now accepts `filter_delay_blocks: int = 0`
- Orchestrator passes `filter_delay_blocks=_delay_blocks` to `estimate()`
- When `use_aec3_echo_gen_window=True`: maintain `_delay_render_buf` deque (maxlen=16, index 0=current)
  - `idx_start = max(0, delay - pre)`, `idx_stop = min(len-1, delay + post)`, pre=post=1
  - `x2 = max over _delay_render_buf[idx_start : idx_stop+1]`
- When flag OFF: legacy recent-N ring buffer unchanged (byte-equal preserved)

**Trace diagnostic fields** (per-hop, in `_hf_chain_trace`):
`echo_gen_delay_blocks`, `echo_gen_idx_start`, `echo_gen_idx_stop`

### R0.4 — ERLE instantaneous quality (continuous)

**Source**: `FullBandErleEstimator::GetInstLinearQualityEstimates` (`fullband_erle_estimator.cc`).
**NOT** from FilterAnalyzer. FilterAnalyzer produces `consistent_estimate` for filter convergence —
R0.4 is the *reverb model update quality* from the ERLE side, not the filter side.

**Implementation**: `_update_erle_inst_quality()` in `orchestrator.py` — ports rolling 6-hop ERLE:
- `quality = (erle_log2 − min_log2) / (max_log2 − min_log2)` with instant-up / EMA-down (α=0.07)
- Render energy gate: `X2_total > 44015068 × 257` (int16² threshold × n_bins)
- Returns None when accumulator not yet populated (equivalent to binary None = skip reverb update)

## Trace Results (2026-05-27)

### xFk7 (DT_mvmt)

**Bucket**: DT_mvmt | **Primary metric**: deg | **M0 ref**: 2.881

| Variant | deg | Δ vs M0 | Δ vs M_full | ul% | cond1% | cond2% | revQ% | erle |
|---------|-----|---------|-------------|-----|--------|--------|-------|------|
| M0 | 2.881 | −0.000 | +0.988 | 4.9 | 0.0 | 0.0 | 5.8 | −4.6 |
| M_R0 | 2.839 | −0.042 | +0.946 | 5.1 | 0.0 | 0.0 | 6.1 | −4.5 |
| M_D | 1.908 | −0.973 | +0.015 | 14.6 | 20.4 | 13.3 | 14.3 | 0.4 |
| M_full | 1.893 | −0.988 | 0.000 | 6.9 | 16.6 | 31.5 | 15.1 | 0.6 |
| M_full+R0.2 | 1.893 | −0.988 | **0.000** | 6.9 | 16.6 | 31.5 | 15.1 | 0.6 |
| M_full+R0.3_fixed | 1.897 | −0.984 | **+0.004** | 6.9 | 16.6 | 31.5 | 15.1 | 0.6 |
| M_full+R0.4 | 2.065 | −0.816 | **+0.172** | 6.9 | 16.6 | 31.5 | 35.6 | 0.6 |
| M_full+R0 | 2.063 | −0.818 | **+0.170** | 6.9 | 16.6 | 31.5 | 35.6 | 0.6 |

**Gate 0 (R0)**: PASS — no flag regresses vs M_full by more than −0.2 threshold.

**R0.3 EchoGeneratingPower window** (mean across all hops):

| | filter_delay_blocks | idx_start | idx_stop |
|---|---|---|---|
| M_full (legacy recent-N, flag OFF) | 0.0 (not tracked) | 0.0 (not tracked) | 0.0 (not tracked) |
| M_full+R0.3_fixed (delay-centered) | 0.0 | 0.0 | 1.0 |

**R0.3 finding**: `filter_delay_blocks=0` on average for xFk7 in M_full. This is because
FilterAnalyzer returns 0 (pre-convergence headroom default) when the filter is not converged —
and xFk7 at M_full has ul=6.9% (mostly non-linear mode = filter not converged). With delay=0,
window=[max(0,-1)=0, 0+1=1] = frames [0,1] = same as legacy recent-N pre=0/post=1 window [0,1].
**Effective impact: +0.004 (negligible).** Delay-centered window cannot differ from legacy window
when filter_delay_blocks=0; structural fix has no effect on xFk7.

**Attribution**:
- R0.2 (noise gate): **zero effect** — failure is in linear path (URO cond2=31.5%), not non-linear path.
  usable_linear=6.9% → noise gate only affects non-linear frames (6.9%); negligible.
- R0.3_fixed (delay-centered): **+0.004 negligible** — filter_delay_blocks=0 throughout (not converged);
  delay-centered window degenerates to legacy window [0,1]. Not a residual path fix.
- R0.4 (ERLE quality): **+0.172 partial recovery** — revQ 15.1%→35.6%. Continuous ERLE quality
  gives the reverb model better estimates during degraded convergence. Partial fix only.
- **Root cause**: ul=6.9%, cond2=31.5%. URO fires "refined-diverged" on 31.5% of frames during
  movement → routes output to un-converged coarse filter. This is a linear-path convergence gap,
  not a residual-path issue. No R0 flag can fix it.
- **M_D vs M_full**: the delay chain INCREASES cond2 from 13.3%→31.5% and DECREASES ul from
  14.6%→6.9% for xFk7. The H_error reset on delay_first prevents rapid reconvergence.

### 9xjhi (FS_static)

**Bucket**: FS_static | **Primary metric**: echo | **M0 ref**: 4.565

| Variant | echo | Δ vs M0 | Δ vs M_full | ul% | cond1% | cond2% | revQ% | erle |
|---------|------|---------|-------------|-----|--------|--------|-------|------|
| M0 | 4.565 | +0.000 | +2.211 | 33.5 | 0.0 | 0.0 | 33.2 | 1.0 |
| M_R0 | 4.561 | −0.004 | +2.207 | 33.3 | 0.0 | 0.0 | 32.8 | 1.0 |
| M_D | 2.170 | −2.395 | −0.184 | 0.0 | 42.6 | 6.0 | 8.0 | 0.9 |
| M_full | 2.354 | −2.211 | 0.000 | 0.0 | 41.9 | 6.5 | 8.7 | 0.9 |
| M_full+R0.2 | 2.354 | −2.211 | **0.000** | 0.0 | 41.9 | 6.5 | 8.7 | 0.9 |
| M_full+R0.3_fixed | 2.354 | −2.211 | **0.000** | 0.0 | 41.9 | 6.5 | 8.7 | 0.9 |
| M_full+R0.4 | 2.306 | −2.259 | **−0.048** | 0.0 | 41.9 | 6.5 | 28.2 | 0.9 |
| M_full+R0 | 2.306 | −2.259 | **−0.048** | 0.0 | 41.9 | 6.5 | 28.2 | 0.9 |

**Gate 0 (R0)**: PASS — no flag regresses vs M_full by more than −0.2 threshold.
(R0.4 is −0.048 vs M_full, within the −0.200 threshold.)

**R0.3 EchoGeneratingPower window** (mean across all hops):

| | filter_delay_blocks | idx_start | idx_stop |
|---|---|---|---|
| M_full (legacy recent-N, flag OFF) | 0.0 (not tracked) | 0.0 (not tracked) | 0.0 (not tracked) |
| M_full+R0.3_fixed (delay-centered) | 0.1 | 0.1 | 1.1 |

**R0.3 finding**: `filter_delay_blocks≈0` for 9xjhi as well (FS_static, ul=0.0% throughout —
non-linear path always active, but filter still not converged at M_full). Window is effectively
[0,1] in both cases. **Zero improvement** — same as R0.2.

**Attribution**:
- R0.2 (noise gate): **zero effect** — 9xjhi is FS_static; usable_linear=0.0% at M_D/M_full
  (URO cond1 dominates). Non-linear path runs always, but noise gate does NOT affect R² because:
  ul=0% → non-linear path → X² used for R² = echo_path_gain × X², gated by noise_gate.
  Result: zero effect. The cond1=41.9% routing dominates — the linear path output is selected
  to coarse (un-converged shadow), not the residual path computation.
- R0.3_fixed (delay-centered): **zero effect** — filter_delay_blocks≈0 (not converged), window=[0,1]
  is identical to legacy. No change to R² computation.
- R0.4 (ERLE quality): **−0.048 slight regression** — revQ 8.7%→28.2%. The continuous ERLE
  quality causes more frequent reverb model updates, slightly worsening the residual estimate
  on this FS_static case. Within Gate 0 threshold but worth monitoring.
- **Root cause**: ul=0.0%, cond1=41.9%. URO fires "coarse-cleaner" on 41.9% of frames →
  routes output to coarse filter output. Coarse (shadow) filter is not converged on this case.
  The coarse-cleaner condition fires because the full Bundle A composition (per_bin_h_error_refresh)
  reduces e2_refined aggressively → e2_coarse < 0.9×e2_refined triggers cond1.
  This is a linear-path composition gap between Bundle A + Bundle B behavior for FS_static.

## R0 Gate 0 Summary (2026-05-27)

| Flag | xFk7 Δ vs M_full | 9xjhi Δ vs M_full | Gate 0 (Δ < −0.2) |
|------|-----------------|-------------------|---------------------|
| R0.2 | 0.000 | 0.000 | **PASS** |
| R0.3_fixed | +0.004 | 0.000 | **PASS** |
| R0.4 | +0.172 | −0.048 | **PASS** |
| All R0 | +0.170 | −0.048 | **PASS** |

**R0 Gate 0 = PASS (all flags, 2026-05-27).**

R0.3 comparison vs prior implementation (pre=1 recent-N only):
- Prior: xFk7 +0.002, 9xjhi 0.000 (ring-buffer pre=1, not delay-centered)
- Fixed: xFk7 +0.004, 9xjhi 0.000 (delay-centered, filter_delay_blocks=0 throughout)
- Conclusion: strict AEC3 delay-centered window does not change Gate 0 verdict.
  Both implementations degenerate to window=[0,1] when filter_delay_blocks=0 (not converged).

## 12-Case Projection: M_full+R0 Will NOT Pass Gate

Gate 0 passes but **M_full+R0 will NOT fix the 12-case G1/G2/G3 failures**:
- xFk7 with all R0: deg=2.063, still −0.818 vs M0 (G2 threshold is −0.20; FAIL)
- 9xjhi with all R0: echo=2.306, still −2.259 vs M0 (G3 threshold is −0.10; FAIL)

The failures are structural — in the linear output selection path (URO), not the residual path.

**Blocked by**:
1. xFk7: H_error reset on delay_first → cond2=31.5% → URO falls back to un-converged coarse
2. 9xjhi: Bundle A per_bin_h_error_refresh → cond1=41.9% → URO routes to un-converged shadow

**R0 residual flags are correctly implemented** (Gate 0 PASS, source parity verified).
They represent genuine AEC3 alignment improvements to the residual path but are not the
root cause of the 12-case failures. Those failures require linear-path fixes (URO cond1/cond2).

## Gate Rule

```
Gate 0 (R0): Each R0 flag individually must not catastrophically regress
             xFk7 deg or 9xjhi echo vs M_full baseline. Threshold: Δ < -0.2.
  → PASS (2026-05-27)

Gate 1 (12-case): M_full+R0 must pass G1/G2/G3/G4 → re-run 12-case script.
  → BLOCKED: xFk7/9xjhi failures are structural (linear path), not residual.
     Running 12-case with M_full+R0 will reproduce the same G1/G2/G3 failures.
     Need structural fix (URO cond1/cond2 root cause resolution) before 12-case.

Gate 2 (800-case): Requires explicit user authorization.
  → BLOCKED pending Gate 1.
```

## Status

- [x] R0.2 implemented — `use_aec3_residual_noise_gate` (27509562→27509.42 int16² verbatim)
- [x] R0.3 implemented + fixed (2026-05-27) — `use_aec3_echo_gen_power_window`
  - Prior: recent-N ring buffer with pre=1 (not delay-centered)
  - Fixed: strict AEC3 delay-centered window; `filter_delay_blocks` passed to `estimate()`
  - Finding: no behavioral difference when filter_delay_blocks=0 (not converged); xFk7 +0.004
- [x] R0.4 implemented — `use_aec3_erle_reverb_quality`
  - Source: `FullBandErleEstimator::GetInstLinearQualityEstimates` (NOT FilterAnalyzer)
- [x] Byte-equal 25/25 PASS (all R0 flags default-OFF invariant preserved)
- [x] Targeted trace COMPLETE (2026-05-27): xFk7 × 9xjhi × 8 variants
- [x] R0 Gate 0: PASS (2026-05-27)
- [ ] 12-case M_full+R0: BLOCKED — structural linear-path failures persist
- [ ] AEC3 behavioral reference: PENDING (run `v3_21_aec3_reference_aecmos.py`)
- [ ] 800-case: BLOCKED (user authorization required, Gate 1 not passed)
