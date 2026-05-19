# AEC Algorithm Methods (v3.10.4 reference)

**Note (v3.21.0 release)**: this document captures the v3.10.4 deep
algorithm reference. The current release is **v3.21.0** (2026-05-19),
which retires the legacy `ResFilter` 9-stage chain documented below
in favour of `AEC._aec3_post` (AecState + ResidualEchoEstimator +
SuppressionGain + CNG, AEC3-aligned). Module references like
`python/modules/res_filter.py` no longer exist in v3.21 — see
[../CHANGELOG.md](../CHANGELOG.md) for the migration and
[refactor_modules_layout.md](refactor_modules_layout.md) for the
current module layout. The PBFDKF / shadow filter / DelayEstimator /
DTD / detector sections remain accurate; the RES post-filter section
is preserved as historical reference.

**Version**: v3.10.4 reference. Python `aec.py` `__version__ = "3.21.0"`.
Cumulative trail: v3.8.4 Plan A HF preservation (smoothing kernel +
HF cap anchor), v3.10.0 delay + recovery state machine (max_delay 512,
plateau detector, long-window EMA, mu-scale delay-confidence ceiling),
v3.10.1 long-window EMA refinements, v3.10.2–v3.10.3 Codex review
fixes, v3.10.4 widens `max_delay_ms` 512 → 1024. See
[../CHANGELOG.md](../CHANGELOG.md) for the full v3.10.4 → v3.21.0 path.

This document is the **deep algorithm specification**. For other angles:

| Document | Purpose |
|---|---|
| [aec_algorithm_guide.html](aec_algorithm_guide.html) | Presentation overview (figures, scores, debug) |
| [c_user_and_integration_guide.md](c_user_and_integration_guide.md) | C API, build, integration rules, streaming contract |
| [CHANGELOG.md](CHANGELOG.md) | Version-by-version changes |
| [SUMMARY.md](SUMMARY.md) | R1–R16 algorithm investigation log |
| [aec_v3_evolution.md](aec_v3_evolution.md) | Historical v3.0–v3.4 design rationale |

All numerical defaults below are anchored to `python/aec.py` (the
authoritative reference). The C port (`c_impl/`) is bit-exact aligned and
inherits the same defaults via `aec_config_from_preset()`.

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [Adaptive filter family](#2-adaptive-filter-family)
   - 2.1 LMS / 2.2 NLMS / 2.3 FDAF / 2.4 PBFDAF / 2.5 PBFDKF
3. [Double-talk detection & adaptation control](#3-double-talk-detection--adaptation-control)
4. [Residual Echo Suppressor (RES)](#4-residual-echo-suppressor-res)
5. [Pre-processing](#5-pre-processing)
6. [Convergence control](#6-convergence-control)
7. [Preset system](#7-preset-system)
8. [Python ↔ C parity](#8-python--c-parity)

Appendices:
- A. [v3.8.1 baseline metrics](#appendix-a--v381-baseline-metrics)
- B. [Investigation log](#appendix-b--investigation-log)
- C. [Known limitations](#appendix-c--known-limitations)
- D. [Removed / legacy paths](#appendix-d--removed--legacy-paths)
- [References](#references)

---

## 1. System architecture

### 1.1 Signal flow

The pipeline runs every 10 ms (one hop). Each block:

```
mic ──► HPF ──► Saturation ──┐
                              ▼
ref ──► HPF ──► Saturation ─► Delay-Est / Ref-Align ─┐
                                                      ▼
                              ┌──────── PBFDKF (main) ─────────┐
                              │   ↑           ↓                 │
                              │ DTD / EPC   Shadow filter       │
                              │  ↓ mu_scale  ↑ copy gating      │
                              └────────┬───────────────────────┘
                                       ▼
                                 error = mic − echo_est
                                       │
                              ┌────────┴────────┐
                              │ Output limiter  │ (|out| ≤ |mic|)
                              └────────┬────────┘
                                       ▼
                              ┌─────────────────┐
                              │ RES post-filter │
                              │  ① Echo PSD     │
                              │  ② ENR gate     │
                              │  ③ Spectral floor│
                              │  ④ CNG (optional)│
                              └────────┬────────┘
                                       ▼
                                clean output
```

**Algorithmic latency**: ~10 ms (one hop) when RES enabled (RES OLA
adds one hop); ≈ 0 with `enable_res=False`.

### 1.2 Public API

Python:

```python
from aec import AEC, AecConfig, AecPreset

cfg  = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000)
aec  = AEC(cfg)
hop  = aec.hop_size                       # 80 / 160 / 480
out  = aec.process(mic_hop, ref_hop)      # 1 hop in → 1 hop out
stats = aec.get_stats()                   # AecStats dataclass
aec.reset()
```

C:

```c
AecConfig cfg;
aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

Aec aec;
aec_create(&aec, &cfg);            /* heap path */
/* or aec_init(&aec, pool, n, &cfg) for static memory */

int hop = aec_hop_size(&aec);
aec_process(&aec, mic_hop, ref_hop, out_hop);
aec_destroy(&aec);
```

### 1.3 Modes

`AecMode` (`python/aec.py:30`):

| Mode | Domain | Production? |
|---|---|---|
| `LMS`     | time, sample-by-sample      | reference only |
| `NLMS`    | time, sample-by-sample      | reference only |
| `FDAF`    | freq, single block          | reference only |
| `PBFDAF`  | freq, partitioned, NLMS adapt | reference only |
| `PBFDKF`  | freq, partitioned, Kalman adapt | **production** |
| `SUBBAND` | alias of `PBFDKF`           | back-compat |

The first four are kept for educational comparison; the only mode
implemented in C and shipped in production is `PBFDKF`.

### 1.4 Frame & block sizes

All sizes derive from `sample_rate`. See `__post_init__` in
`AecConfig` and `aec.c:aec_derive_sizes`.

| Field | 8 kHz | 16 kHz | 48 kHz |
|---|---|---|---|
| `frame_size` (20 ms) | 160 | 320 | 960 |
| `hop_size` (10 ms)   | 80  | 160 | 480 |
| `fft_size`           | 256 | 512 | 1024 |
| `n_freqs`            | 129 | 257 | 513 |
| `filter_length` (52 ms default; 64 ms ≥ 44.1 k in Python) | 416 | 832 | 3072* |
| `n_partitions` (52 ms / 10 ms) | 6 | 6 | 6 |

> \* Python auto-picks 64 ms at SR ≥ 44.1 k (3072 samples / 6 partitions).
> The C port uses 52 ms across all SRs unless overridden.

---

## 2. Adaptive filter family

### 2.1 LMS — time-domain reference

```
y[n] = w[n]ᵀ · x[n]
e[n] = d[n] − y[n]
w[n+1] = leak · w[n] + μ · e[n] · x[n]
```

`x[n]` is a length-`L` ref tap-vector. `μ` ≈ 0.001–0.05.

Educational baseline. Not used in production (slow convergence, sensitive
to ref level scaling).

### 2.2 NLMS — normalized

```
w[n+1] = leak · w[n] + μ · e[n] · x[n] / (‖x[n]‖² + δ)
```

With `μ ∈ [0.1, 0.8]`, `δ` regularization, optional weight-norm clamp
(divergence guard). Faster and level-invariant compared with LMS.

Implemented as time-domain reference. Production uses the partitioned
frequency-domain formulation (PBFDKF) instead.

### 2.3 FDAF — frequency-domain adaptive filter

50 % overlap-save block adaptation. Each block:

```
X = FFT(x_block)         # length 2L
Y = X ⊙ W
y = real(IFFT(Y))[L:]    # save second half
e = d − y
E = FFT([0; e])          # zero-pad
P = α · P + (1−α) · |X|²
W = W + μ · conj(X) ⊙ E / (P + δ)
```

Projects ref into freq, multiplies by current weights, computes error,
normalizes by power-spectrum estimate `P`, updates weights.

Single block ⇒ filter length = block size. Convergence is fast for
short echo tails (≤ 10 ms) but cannot represent long room responses.

### 2.4 PBFDAF — partitioned-block FDAF

Splits the impulse response into `P` partitions of size `L_p = block / 2`:

```
W₀, W₁, …, W_{P-1}     each length 2 L_p
echo_freq = Σ_k X_k ⊙ W_k     # X_k = ref delayed by k blocks
W_k ← W_k + μ · conj(X_k) ⊙ E / (P_freq + δ)
```

Each partition learns one section of the room impulse response. With
P=6 and 10 ms hop, the filter spans **60 ms** total — long enough for
typical phone / IoT room reverb tails.

### 2.5 PBFDKF — partitioned-block frequency-domain Kalman (production)

Replaces NLMS step-size with a per-bin Kalman gain. Conceptually each
frequency bin has its own scalar Kalman filter tracking the optimal
weight value.

#### Update equations

```
predict:  P_k ← P_k + Q                  (process-noise grow)
gain:     K_k = P_k / (P_k + R_k · |X_k|²)
update:   W_k ← W_k + K_k · conj(X_k) ⊙ E
          P_k ← (1 − K_k · |X_k|²) · P_k
```

Per bin and per partition, tracking weight-error covariance `P_k`,
process noise `Q`, measurement noise `R_k` (estimated from the residual
PSD).

The advantage: **adaptation rate is automatically tuned by signal
quality**. Where `|X|²` is large (loud ref), the gain `K` shrinks to
prevent overshoot. Where the filter is uncertain (`P_k` large), gain
grows to learn faster.

#### Two-stage Q switching

After convergence, `Q` is reduced to dampen residual jitter:

```
not yet converged → Q_high   (kalman_q_high)
converged & stable → Q_low   (kalman_q_low = 1e-6)
```

Per-preset `Q_high`:

| Preset | `kalman_q_high` |
|---|---|
| MILD       | `1.5e-3` |
| SOFT       | `1.5e-3` |
| BALANCED   | `1.0e-3` |
| AGGRESSIVE | `7.0e-4` |
| MAXIMUM    | `7.0e-4` |

`Q_low = 1e-6` is shared. The lower bound is protected by `P_floor =
1e-4` so that the filter can still respond to sudden echo-path changes.

> Earlier versions had `Q_low = 1e-5`; this was lowered to `1e-6` in v3
> after measuring lower steady-state misadjustment.

#### Why not single-stage NLMS?

PBFDKF avoids two NLMS pathologies:

1. **Step-size sensitivity to ref level** — NLMS needs careful `μ`
   tuning; Kalman gain auto-scales with `|X|²`.
2. **Slow recovery after EPC** — NLMS uses a fixed `μ`; PBFDKF can
   raise `P_floor` on EPC and recover in ~200 ms vs 1–3 s for NLMS.

#### Reference

S. Malik & G. Enzner, *State-Space Frequency-Domain Adaptive Filtering for
Nonlinear Acoustic Echo Cancellation* (IEEE TASLP, 2012) — derivation of
the FDKF with per-bin Kalman gain.

---

## 3. Double-talk detection & adaptation control

The filter must stop adapting when near-end speech is present, otherwise
it learns the speech as part of the echo path and cancels it.

### 3.1 Three-line DTD ensemble

The production ensemble combines three independent signals into a
single `dt_confidence ∈ [0, 1]`:

| Source | Detector | Strength |
|---|---|---|
| Energy | `dt_from_energy` | mic ≫ ERL × ref → near-end present |
| Shadow filter | `dt_from_shadow` | shadow-vs-main error advantage |
| Coherence | `dt_from_coherence` | fullband C² between ref & error |

These are independent failure modes — energy detector misses when ERL
estimate is wrong; shadow detector misses when shadow has not converged;
coherence detector misses on stationary far-end. Combining them
(soft-max with hangover) gives robust DT triggering across scenarios.

`dt_active` = `dt_confidence > 0.5`.

### 3.2 Coherence-based DTD (production gate)

Computed every block in the frequency domain:

```
S_xx = α · S_xx + (1−α) · |X|²        # ref PSD
S_ee = α · S_ee + (1−α) · |E|²        # error PSD
S_xe = α · S_xe + (1−α) · X · conj(E) # cross-PSD
C²   = |S_xe|² / (S_xx · S_ee)        # magnitude-squared coherence
```

Defaults (`AecConfig`):

| Param | Value | Meaning |
|---|---|---|
| `dtd_coh_alpha`        | 0.85 | PSD smoothing (~6-block TC) |
| `dtd_coh_high`         | 0.6  | C² > → no DT |
| `dtd_coh_low`          | 0.3  | C² < → DT |
| `dtd_coh_energy_floor` | 0.1  | min |E|² / |X|² to fire |
| `dtd_coh_hangover`     | 3    | release lag |
| `dtd_coh_release`      | 0.1  | confidence decay rate |

When ref and error are highly correlated (C² high), the residual is
echo — keep adapting. When they decorrelate (C² low) and energy is
non-trivial, near-end speech is in the error — freeze.

### 3.3 Output-vs-input divergence

Safety net: even if DTD fails, an output that exceeds the input must
not enter the weight update.

```
if |out|² > divergence_factor × |mic|²:
    treat as diverged → drive mu_scale → 0
```

`dtd_divergence_factor = 1.5`.

### 3.4 Confidence → mu_scale

The combined `dt_confidence` is mapped to a multiplicative gate on the
adaptation rate:

```
mu_scale = 1 − dt_confidence × (1 − mu_min_ratio)
```

`dtd_mu_min_ratio = 0.05`. So even at full DT, 5 % of nominal step
remains, which lets the filter resume tracking quickly after DT ends.

### 3.5 Shadow filter copy gating

A second filter (`shadow`) runs in parallel at a higher adaptation rate
(`shadow_q_ratio` × main `Q`). The criterion for copying its weights
into the main filter:

```
shadow_advantage = main_err / shadow_err
if shadow_advantage > shadow_copy_threshold (0.65)
   for shadow_copy_hysteresis (3) frames:
       main.W ← shadow.W
```

Shadow only helps when it tracks better than main — the hysteresis
prevents flip-flopping. After EPC, shadow typically wins for ~10
frames and copies, recovering main filter fast.

### 3.6 EPC (echo-path change) detection

Driven by total error rise + shadow advantage:

| Param | Value |
|---|---|
| `epc_delta_threshold` | 0.3 |
| `epc_total_rise`      | 1.5 |
| `epc_hangover`        | 20 frames (~200 ms) |
| `epc_mu_floor`        | 0.5 (mu_scale lower bound during EPC) |

When EPC fires:

1. `P` covariance is bumped (allow fast re-learning)
2. `Q` is held at `Q_high`
3. Shadow filter is reset to track from scratch
4. RES temporarily switches to render-based estimate
5. Hangover lasts 20 frames before resuming normal adaptation

### 3.7 Filter state machine

`AecFilterState` enum (`python/aec.py:52`):

| State | Triggers | Effect |
|---|---|---|
| `WARMUP`         | first `warmup_frames` frames | high `mu`, no DT freeze |
| `CONVERGING`     | erle_inst rising | normal adaptation |
| `CONVERGED`      | erle_windowed > 10 dB | switch to `Q_low` |
| `DT_ACTIVE`      | `dt_confidence > 0.5` | mu_scale ≈ 0 |
| `EPC_RECOVERY`   | EPC fires | `Q_high`, `P` bump |
| `STATIONARY_FAR` | far-end CV² low for ≥ N frames | RES extra protection |
| `DIVERGED`       | output ≫ mic | mu_scale ≈ 0, alarm |

Only one state at a time; priority order is the table top-down.

### 3.8 Filter recovery state machine (v3.10.0+)

Beyond EPC, two new recovery paths catch slow-degradation modes that
EPC's threshold-based trigger misses.

#### `FilterPlateauDetector`

Watches windowed ERLE during DT episodes — when ERLE stays low for too
long despite active far-end and active DT signal, the linear filter
has likely learned a corrupted impulse response. Reset taps + downstream
derived state to give the filter a clean adaptation window.

| Param | Value |
|---|---|
| consecutive low-ERLE frame budget | 50 |
| grace period (frames after start / reset) | 400 |
| `far_active_ratio` gate | > 0.5 |
| `dt_signal_ratio` gate (current frame, v3.10.3 F4) | > 0.10 |
| `MAX_ATTEMPTS` per case | 2 |

On fire (v3.10.3 H3): cumulative counters `_frame_count`,
`_far_active_count`, `_dt_signal_count` reset (otherwise they survive
into the second attempt and bias its threshold). Action calls
`_reset_filter_derived_state(reason='plateau',
preserve_render_ema=True)`.

#### `_reset_filter_derived_state(reason, preserve_render_ema)`

Shared helper introduced in v3.10.2 to make plateau and delay-first
resets symmetric. Resets:

- main filter taps (`W_freq`) + `P` covariance
- shadow filter state
- `error_power` EMA AND `near_power` EMA (v3.10.3 F3 — the prior
  helper reset only `error_power`, producing a transient ERLE spike
  that fed back into convergence detection)
- `_pending_delay` (v3.10.3 F5)
- ResFilter / ResidualEchoEstimator state via
  `ResFilter.reset(preserve_long_window_ema=...)` →
  `ResidualEchoEstimator.reset(preserve_long_window_ema=...)`
  (v3.10.2). When the input-side render PSD is still meaningful
  (plateau or delay_shift), preserve the long-window EMA;
  delay-first acquisition wipes it.

#### `ResidualEchoEstimator` long-window far-PSD EMA (v3.10.0+)

A delay-agnostic echo template tracked at `alpha = 0.993`
(time-constant ~30 s @ 16 kHz / 100 fps). Used as the render-based
fallback's far-end estimate so it is robust to short-term silences in
the far signal.

- v3.10.1: EMA updates **every far-active frame** regardless of
  current mode (constant time-constant, not gated by render-vs-direct
  selection — keeps the template warm in case the direct path later
  fails).
- v3.10.1: render-based fallback path blends 70 % long-window + 30 %
  instantaneous; warmup-gated by `n_updates / 100` so the very first
  cycles still favor instantaneous data while the EMA fills in.

### 3.9 NLMS-mode extras

For LMS / NLMS modes (not production), an additional **weight L2-norm
clamp** prevents divergence:

```
if ‖w‖² > w_norm_max:
    w *= w_norm_max / ‖w‖²
```

PBFDKF doesn't need this because Kalman gain self-stabilizes.

---

## 4. Residual Echo Suppressor (RES)

The linear filter caps at ~20–30 dB ERLE (limited by speaker non-linearity,
modeling error, reverb tail). RES is a frequency-domain post-filter
applying a per-bin gain `g(k) ∈ [g_min(k), 1]` on the linear AEC output
to push residual echo further down.

Three logical stages, executed every block (10 ms):

```
linear AEC output ──► ① Residual Echo PSD ──► ② ENR Gain Gate ──► ③ Spectral Floor ──► out
                                                                          ↓
                                                                         CNG (optional)
```

### 4.1 Framework: OLA + sqrt-Hann

| Param | Value |
|---|---|
| Analysis window | 50 % overlap, sqrt-Hann |
| Synthesis window | Hann (matching) |
| Block | `2 × hop_size` |
| Latency | one hop (~10 ms) |

Sqrt-Hann analysis × Hann synthesis is the AEC3 convention; it matches
COLA at 50 % overlap and avoids modulation artifacts when bin gains
change rapidly.

### 4.2 Stage ① — Residual echo PSD

Goal: estimate `S_echo(k)` = "how much echo power remains at bin k
after the linear AEC". Two estimators are blended:

#### Direct (filter-based)

```
S_echo_direct(k) = |echo_spec(k)|² / ERLE(k)
```

`echo_spec` is the filter's predicted echo (already computed in PBFDKF
step). `ERLE(k)` is the per-bin echo-return-loss-enhancement, tracked
from the ratio of mic PSD to error PSD over recent blocks.

Per-bin ERLE update conditions (only update when reliable):

```
update if  filter_converged
       and dt_indicator < 0.3
       and far_activity > floor
```

#### Render-based (fallback)

```
S_echo_render(k) = |X(k)|² × ERL_avg
```

where `ERL_avg` is the average echo-return-loss tracked from
`mean(|mic|²) / mean(|ref|²)` during far-only periods. Independent of
filter state.

#### Blend by convergence

```
λ = clip(erle_factor, 0, 1)
S_echo(k) = λ · S_echo_direct(k) + (1 − λ) · S_echo_render(k)
```

`erle_factor` rises 0 → 1 as `erle_windowed_db` exceeds the convergence
threshold. So during cold-start RES uses render-based; once converged
it leans on the direct estimate for precision.

### 4.3 Stage ① — Reverb tail model (optional)

A first-order IIR adds the tail of past blocks' residual:

```
S_tail(k) = decay · S_tail(k) + gain · S_echo_now(k)
S_echo(k) ← max(S_echo(k), S_tail(k))
```

Per-preset:

| Preset | `res_reverb_decay` | `res_reverb_gain` |
|---|---|---|
| MILD       | 0.45 | 0.4 |
| SOFT       | 0.6  | 0.8 |
| BALANCED   | 0.85 | 1.6 |
| AGGRESSIVE | 0.7  | 2.0 |
| MAXIMUM    | 0.8  | 3.0 |

BALANCED uses a longer `decay` (0.85, TC ≈ 130 ms) to model real-room
RT60. AGGRESSIVE/MAXIMUM use higher `gain` to compensate for shorter
decay when faster reaction is preferred.

### 4.4 Stage ② — ENR gain gate

ENR = `S_echo(k) / S_nearend(k)`. The gate is a soft sigmoid on
`ENR(k)`:

```
gain(k) = sigmoid((threshold − ENR_dB) / softness)
```

- `threshold` — center of the transition. ENR above ⇒ suppress.
- `softness` — sigmoid width. Hard ⇒ musical noise; soft ⇒ residual leak.

#### Per-preset ENR scale

`res_enr_scale` shifts the threshold:

| Preset | `res_enr_scale` | Effect |
|---|---|---|
| MILD       | 1.15 | shift ~1 dB later (NE-friendlier) |
| SOFT       | 1.00 | AEC3 default; conservative |
| BALANCED   | 0.85 | shift ~1.5 dB earlier |
| AGGRESSIVE | 0.70 | shift ~3 dB earlier |
| MAXIMUM    | 0.50 | shift ~6 dB earlier |

#### DT relax

When DTD signals double-talk, the threshold is raised so the same ENR
produces a higher gain (less suppression):

```
threshold_DT = threshold + DT_relax_dB    (typ. +4 to +8 dB)
```

#### Spectral over-subtraction

A dynamic over-subtraction term adds echo PSD margin:

```
over_sub = res_over_sub_base
         + res_over_sub_scale × erle_factor
         − res_dt_reduction × dt_indicator
```

Higher `over_sub` ⇒ more aggressive subtraction ⇒ stronger gain
suppression. The `dt_reduction` term backs off during DT.

Per-preset:

| Preset | `over_sub_base` | `over_sub_scale` | `dt_reduction` |
|---|---|---|---|
| MILD       | 1.5  | 2.5  | 4.5 |
| SOFT       | 2.5  | 4.0  | 3.5 |
| BALANCED   | 5.0  | 9.0  | 2.5 |
| AGGRESSIVE | 7.0  | 12.0 | 1.5 |
| MAXIMUM    | 10.0 | 15.0 | 0.5 |

### 4.5 Stage ③ — Spectral floor

Two floors are blended; the larger of the two clamps the gain.

#### Fixed floor

`res_g_min_db`, per-preset:

| Preset | `res_g_min_db` |
|---|---|
| MILD       | −25 dB |
| SOFT       | −35 dB |
| BALANCED   | −55 dB |
| AGGRESSIVE | −65 dB |
| MAXIMUM    | −72 dB |

#### Spectral-shape-preserving floor (`res_spectral_floor`)

Tracks the error envelope and lets the floor follow spectral shape
rather than a flat dB number. Prevents notches sounding "hollow":

```
env(k) = 0.95 · env(k) + 0.05 · |E(k)|²       # error envelope EMA
norm   = env(k) / mean(env)                    # shape indicator
floor(k) = res_spectral_floor_db
         + 10·log10(norm)                      # shape-preserving offset
```

`res_spectral_floor_db` per preset:

| Preset | `res_spectral_floor_db` |
|---|---|
| MILD       | −18 dB |
| SOFT       | −25 dB |
| BALANCED   | −38 dB |
| AGGRESSIVE | −45 dB |
| MAXIMUM    | −55 dB |

#### Final floor

```
g_min(k) = max( fixed_floor, spectral_floor(k), per_bin_ne_protect(k) )
gain(k)  = max( gain_from_ENR(k), g_min(k) )
```

`per_bin_ne_protect` is a per-frequency near-end protection ceiling
(`res_ne_protect_db`, ranging −7 dB MILD → −30 dB MAXIMUM); see §4.7.

### 4.6 Multi-ERLE estimation

Two estimators feed the per-bin ERLE used in §4.2:

| Estimator | Domain | Update window |
|---|---|---|
| `FilterErleEstimator`   | per-bin | every block, FS-only |
| `FullbandErleEstimator` | scalar | every block |

The fullband estimator provides `erle_inst_db`, `erle_windowed_db`
(used for state machine + ENR gate). The per-bin estimator drives the
direct echo-PSD computation.

Cross-check: when fullband ERLE indicates convergence but per-bin ERLE
disagrees, we trust per-bin (some frequencies may not be well-modeled
even if average is good).

### 4.7 Per-bin near-end gate

Pre-convergence (filter still learning) the per-bin floor gives moderate
protection so a louder near-end speaker is not crushed:

```
ne_protect(k) = res_ne_protect_db   if erle_factor < 0.3
              = stricter             else
```

Per-preset: see `res_ne_protect_db` table in §4.5.

### 4.8 CNG — Comfort Noise Generation

Optional (`enable_cng`; default ON for BALANCED, OFF elsewhere). Goal:
fill spectral holes left by aggressive suppression so the listener
doesn't hear "dead air" (which sounds like a dropped call).

#### Freeze-gated minima tracking

```
if not actively_suppressing(k):
    noise_psd(k) = α · noise_psd(k) + (1−α) · |E(k)|²
# When suppressing, freeze — don't let CNG learn echo
```

Min-statistics: track the per-bin minimum over the last ~1.5 s of
"frozen" snapshots. Provides a robust noise-floor estimate independent
of speech content.

#### AEC3-style crossfade

```
if gain(k) < g_min(k):
    cng(k) = sqrt(noise_psd(k)) · complex_gaussian()
    out(k) = cng(k)         # replace, not add
```

Complex Gaussian (random magnitude AND phase) avoids tonal artifacts
that pure-magnitude noise would create.

> **Don't stack CNGs.** If the AEC's `enable_cng=1`, never run another
> noise-fill stage downstream. Double-shaping creates audible tonal
> artifacts.

### 4.9 Postprocess gains (AEC3-style)

Applied after the per-bin gain mask:

| Step | Purpose |
|---|---|
| DC consistency  | bins 0–1 follow bin 2 (avoid DC dip) |
| HF cap          | bins above ~2 kHz capped at 2-kHz bin gain |
| Temporal smooth | activity-driven release (faster down, slower up) |
| Rate limit      | ±6 dB per frame (`res_max_drop_db_per_frame`) |
| Divergence override | force bypass when `dt_active` & filter diverged |

These gains are co-tuned; per-knob tweaking will likely break audible
quality. Pick a preset rather than tuning individually.

### 4.10 RES signal flow summary

```
linear AEC out
      │
      ▼
┌──────────────────────────────────────────┐
│ Stage ① — Echo PSD                        │
│   direct: echo_spec/ERLE                  │
│   render: |X|² · ERL                      │
│   blend by erle_factor                    │
│   + reverb tail IIR                       │
└──────────────────────────────────────────┘
      │ S_echo(k)
      ▼
┌──────────────────────────────────────────┐
│ Stage ② — ENR gain gate                   │
│   ENR = S_echo / S_nearend                │
│   gain = sigmoid(threshold − ENR)         │
│   over_sub adjustment (erle, dt)          │
│   DT relax → raise threshold              │
└──────────────────────────────────────────┘
      │ raw_gain(k)
      ▼
┌──────────────────────────────────────────┐
│ Stage ③ — Spectral floor                  │
│   max(fixed_floor, spectral_floor,         │
│       ne_protect)                          │
│   final_gain = max(raw_gain, floor)        │
└──────────────────────────────────────────┘
      │ gain(k)
      ▼
   CNG (if enabled & gain < floor)
      │
      ▼
   IFFT → OLA → out
```

---

## 5. Pre-processing

### 5.1 High-pass filter

80 Hz Butterworth biquad (`hpf.h` / `hpf.c`). Removes DC, 50/60 Hz hum,
rumble. Applied to both mic and ref before everything else.

```
H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
```

Coefficients are SR-dependent (`2π·f_c / sr`); auto-recomputed in
`hpf_init`.

### 5.2 Saturation detector + soft-clip

#### Detection

```
saturation_level = EMA( fraction of |sample| > saturation_threshold )
saturation_threshold = 0.95   # 95% of full-scale
```

Asymmetric EMA: fast attack (~20 ms), slow release (~200 ms).

#### Soft-clip on ref

When `saturation_softclip_ref = True`, ref is passed through a tanh-like
compressor:

```
ref' = sign(ref) · tanh(|ref| / threshold)
```

The filter learns from the soft-clipped ref, which better matches what
the speaker actually produces (real speakers hard-clip too).

#### RES over-sub boost

When `saturation_level > 0`, RES adds `saturation_over_sub_boost = 3.0`
to `over_sub` to compensate for the broadband non-linear products
created by clipping (which the linear filter cannot model).

### 5.3 Delay estimation (GCC-PHAT)

`DelayEstimator` accumulates short overlapping segments of mic & ref,
computes a phase-transformed cross-correlation, and tracks the lag with
maximum coherence:

```
segment_size = 2 · max_delay_samples
overlap      = segment_size / 2

X = FFT(ref_seg)
Y = FFT(mic_seg)
S_xy = α · S_xy + (1−α) · X · conj(Y)
phat = real(IFFT( S_xy / (|S_xy| + ε) ))
delay = argmax(|phat|)
```

Defaults (v3.10.4):

| Param | Value |
|---|---|
| `max_delay_ms`         | 1024 (was 250 ≤ v3.8.3, 512 in v3.10.0–v3.10.3) |
| `delay_est_init_s`     | 0.3  |
| `delay_est_period_s`   | 0.5  |
| `delay_buffer_ms` (render ring) | 2048 (was 1024 in v3.10.0–v3.10.3) |
| `seg_size`             | auto-scaled to `2 × max_delay_samples` |

The ring buffer is sized to ~2× `max_delay_ms` to keep headroom for
GCC-PHAT segments across late-arriving render frames (matches WebRTC
AEC3's `kRenderTransferQueueSizeFrames = 100` design pattern but with
double the depth). v3.10.4's bump from 512 → 1024 ms tracks WebRTC's
older AEC implementation, which holds ~1 s of far-end history; AEC3's
512 ms cap was found to miss BT/mobile clock-skew cases in our 800-case
bench.

#### Confidence / `is_solid` (v3.10.0+)

```
confidence = clip((par - PAR_LO) / (PAR_HI - PAR_LO), 0, 1)
             with PAR_LO = 5.0, PAR_HI = 8.0
is_solid   = confidence >= 1.0   (par >= 8.0)
```

Used by:
- **Two-path delay-acquisition gate** (split into independent ifs in
  v3.10.2 — the solid-shift path was unreachable when wrapped under
  an outer `is_solid` check):
  - **Path A — first acquisition**: heavy reset via shared helper
    `_reset_filter_derived_state(reason='delay_first',
    preserve_render_ema=False)`, demands `is_solid`.
  - **Path B — delay shift**: independent gate, requires
    `confidence ≥ 0.5` AND two consecutive consistent estimates
    (two-strike), calls `force_delay()` + temporary `Q_high` boost,
    plus `_reset_filter_derived_state(reason='delay_shift',
    preserve_render_ema=True)` (added in v3.10.3 M4 — the prior path
    relied on `mark_diverged` only and left taps misaligned for
    50–100 frames).
- **`mu_scale` delay-confidence ceiling**: `mu ≤ 0.5 + 0.5 *
  delay_conf` (capped at 0.5 when delay uncertain). Skipped during
  the post-reset warmup window (v3.10.3 H2) so it does not fight
  the high-Q boost.

#### `_pending_delay` (v3.10.3+)

Cross-cycle staging buffer for two-strike consistency. Cleared on
`AEC.reset()` and from inside the helper (F5). TTL = 3 cycles (H1)
to prevent pairing with stale rogue estimates after the consistency
window has lapsed. Diagnostic counters (`_round3_div_counts` etc.)
explicitly survive recovery (L7 revert) — only `AEC.reset()` clears
them.

### 5.4 Reference alignment

A ring buffer applies the estimated delay before the filter sees the
ref:

```c
ref_ring[(write++) & (size − 1)] = ref_in[i]
ref_aligned[i] = ref_ring[(read++) & (size − 1)]    # read = write − delay
```

`ref_ring_size = max_delay_samp + 4096`.

When the delay estimate first acquires (transitioning from −1 to a
valid value), the filter is reset — the prior weights are no longer
valid for the new alignment.

> **Offline batch with pre-aligned files**: set
> `enable_delay_est = False` to skip the ~300 ms learning loss from the
> reset on first acquisition.

---

## 6. Convergence control

### 6.1 Warmup

`warmup_frames` (BALANCED: 80 = 800 ms). During warmup:

- `mu_scale` forced high (no DT freeze applied)
- `Q` held at `Q_high`
- RES uses render-based echo estimate exclusively

Reason: at startup we have no reliable `ERL` or `ERLE`; freezing the
filter on a false DT trigger would prevent any learning.

### 6.2 Two-stage Q switching

After convergence, `Q` shrinks from `kalman_q_high` (~1e-3) to
`kalman_q_low = 1e-6`. This reduces steady-state misadjustment by
~10 dB.

The transition uses hysteresis to prevent flip-flopping:

```
converged_streak = 0
for each frame:
    if erle_inst_db > THRESH_HIGH and erle_windowed_db > THRESH_LOW:
        converged_streak += 1
    else:
        converged_streak = 0
    if converged_streak > N: → Q_low
    if EPC fires:           → Q_high
```

`P_floor = 1e-4` ensures the filter can still respond to perturbations
even at `Q_low`.

### 6.3 EPC detection

See §3.6. Triggers Q_high reset, P bump, shadow reset, RES render
fallback, 200 ms hangover.

### 6.4 Output limiter

```
out_limited = sign(out) · min(|out|, |mic| × limit_factor)
```

Safety net: even with a diverged filter, the output never exceeds the
input magnitude. Prevents listener-side clipping when something goes
wrong.

### 6.5 Output noise gate (optional)

Suppresses output below a minimum threshold during far-only periods to
remove low-level filter chatter. Defaults to off; enable via config if
the use case has very quiet far-end signal.

---

## 7. Preset system

`AecPreset` selects a coherent set of RES + adaptive-filter parameters
tuned for a target FS-suppression / DT-preservation trade-off.

### 7.1 Full preset table

Source: `python/aec.py:280-410` (`AecConfig.from_preset`).

| Parameter | MILD | SOFT | BALANCED ★ | AGGRESSIVE | MAXIMUM |
|---|---|---|---|---|---|
| **RES echo estimation** | | | | | |
| `res_echo_method`       | direct | direct | direct | direct | direct |
| `res_gain_type`         | enr    | enr    | enr    | enr    | enr |
| `res_enable_reverb`     | True   | True   | True   | True   | True |
| `res_reverb_decay`      | 0.45   | 0.6    | 0.85   | 0.7    | 0.8 |
| `res_reverb_gain`       | 0.4    | 0.8    | 1.6    | 2.0    | 3.0 |
| `res_alpha_echo_psd`    | 0.6    | 0.5    | 0.4    | 0.3    | 0.2 |
| `res_alpha_error_psd`   | 0.6    | 0.6    | 0.5    | 0.4    | 0.3 |
| `res_enr_scale`         | 1.15   | 1.0    | 0.85   | 0.7    | 0.5 |
| **RES suppression** | | | | | |
| `res_g_min_db`          | −25    | −35    | −55    | −65    | −72 |
| `res_over_sub_base`     | 1.5    | 2.5    | 5.0    | 7.0    | 10.0 |
| `res_over_sub_scale`    | 2.5    | 4.0    | 9.0    | 12.0   | 15.0 |
| `res_dt_reduction`      | 4.5    | 3.5    | 2.5    | 1.5    | 0.5 |
| `res_spectral_floor_db` | −18    | −25    | −38    | −45    | −55 |
| `res_ne_protect_db`     | −7     | −10    | −16    | −22    | −30 |
| **Comfort noise** | | | | | |
| `enable_cng`            | True   | True   | True   | True   | True |
| **Adaptive filter** | | | | | |
| `kalman_q_high`         | 1.5e-3 | 1.5e-3 | 1.0e-3 | 7.0e-4 | 7.0e-4 |
| `warmup_frames`         | 80     | 80     | 80     | 80     | 100 |
| `shadow_q_ratio`        | 3.0    | 3.0    | 3.5    | 4.0    | 5.0 |
| `shadow_mu_min`         | 0.5    | 0.5    | 0.6    | 0.7    | 0.9 |

Shared (not preset-dependent):

| Parameter | Value |
|---|---|
| `kalman_q_low`            | 1e-6 |
| `mu`                      | 0.3 (LMS/NLMS only) |
| `delta`                   | 1e-8 |
| `dtd_coh_alpha`           | 0.85 |
| `dtd_coh_high` / `low`    | 0.6 / 0.3 |
| `epc_hangover`            | 20 frames |
| `epc_mu_floor`            | 0.5 |
| `max_delay_ms`            | 1024 (v3.10.4; was 250 ≤ v3.8.3 / 512 v3.10.0–v3.10.3) |
| `delay_buffer_ms`         | 2048 (v3.10.4; was 1024 v3.10.0–v3.10.3) |
| `highpass_cutoff_hz`      | 80 |
| `saturation_threshold`    | 0.95 |

### 7.2 Selection guide

| Scenario | Preset | Why |
|---|---|---|
| Echo 安靜、希望幾乎不動到近端 | MILD | Minimum-touch RES; NE intelligibility trumps echo cleanup |
| 輕度 echo、近端優先 (= 舊 v3.8.2 MILD) | SOFT | Light RES with audible echo cleanup; balanced for sensitive NE |
| General phone / video call         | **BALANCED ★** | Co-optimised default |
| Automotive, factory, high noise    | AGGRESSIVE | Stronger FS suppression OK |
| IoT speaker, mic-near-speaker      | MAXIMUM | Extreme coupling needs extreme RES |

### 7.3 Don't tune individual knobs

The 14 RES parameters are **co-tuned**. Changing one in isolation almost
always degrades quality elsewhere. The 4 presets sweep the FS / DT
Pareto front; pick whichever side suits the use case.

If you must tune, do it via the high-level dial:

- More FS suppression: lower preset → higher (mild → balanced → aggressive → maximum)
- More NE preservation: higher preset → lower
- Longer reverb tails: increase `filter_length_ms` (52 → 100 ms; cost: memory)

---

## 8. Python ↔ C parity

### 8.1 Module mapping

| Python class / module (`python/aec.py`) | C module (`c_impl/`) |
|---|---|
| `AEC` (top-level orchestration)                  | `aec.h` / `aec.c` |
| `AecConfig`, `AecPreset`                          | `aec.h:AecConfig`, `aec.h:AecPreset` |
| `HighpassFilter`                                  | `hpf.h` / `hpf.c` |
| `SaturationDetector`                              | `saturation.h` / `saturation.c` |
| `DelayEstimator`                                  | `delay_est.h` / `delay_est.c` |
| `PBFDAFFilter`, `PBFDKFFilter`                    | `pbfdkf.h` / `pbfdkf.c` |
| `FilterErleEstimator`, `FullbandErleEstimator`   | `erle.h` / `erle.c` |
| `RenderActivity`, `FilterConvergence`, `DTAnalyzer` | `detectors.h` / `detectors.c` |
| `EchoPathChangeDetector`, `ShadowCopyController` | `epc_shadow.h` / `epc_shadow.c` |
| `ResidualEchoEstimator`                           | `residual_echo.h` / `residual_echo.c` |
| `ResFilter`                                       | `res_filter.h` / `res_filter.c` |
| `DTD` (geigel / divergence / coherence)           | `dtd.h` / `dtd.c` |
| `AecDebugLogger`                                  | `aec_debug.h` / `aec_debug.c` |
| FFT (numpy `pocketfft`)                            | `fft_fp64.c` (radix-2 fp64) |

### 8.2 Bit-exactness

The C port at v3.8.3 was **bit-identical** to Python v3.8.1 across all
4 presets and FS / DT / NE scenarios. The C port v3.10.4 follows the
Python v3.10.4 release in parallel; bit-exact verification is
re-established once the C port lands. Verified at v3.8.3 by:

1. MD5 comparison of output WAVs across the 800-case AEC Challenge
   blind set.
2. Static-memory path (`aec_init` with caller-supplied pool) verified
   to produce the same MD5 as the heap path (`aec_create`).

The fp64 FFT (`fft_fp64.c`) was chosen specifically to match numpy's
`pocketfft`, which is fp64-internal even for fp32 inputs. Switching to
a fp32 FFT (e.g., kiss_fft) would break this guarantee.

### 8.3 Determinism guarantees

To preserve bit-exactness across compilers:

- **`-ffp-contract=off` is mandatory** in the build. FMA contraction at
  `-O2` produces non-deterministic state in HPF / saturation / detector
  accumulators.
- All buffers used in computation are 16-byte aligned (`ALIGN16`).
- No `rand()` / `random()` — CNG noise uses a deterministic RNG seeded
  per-instance (set via `aec_reset` for reproducibility).

### 8.4 Streaming contract

See [c_user_and_integration_guide.md §10](c_user_and_integration_guide.md#10-streaming-contract--sample-rate-scaling)
for the full streaming + SR scaling specification. Briefly: 1 hop in →
1 hop out; latency = 10 ms (one hop, from RES OLA); SR-derived sizes
are automatic.

---

## Appendix A — v3.8.3 baseline metrics

AEC Challenge 2021 Interspeech blind test set, 800 cases, AECMOS
(scale 1–5, higher better).

FS Echo and DT Echo are the means of static + movement sub-buckets.

| Method | FS Echo | DT Echo | DT Degradation | NE Degradation |
|---|---|---|---|---|
| Linear only          | 2.71 | 3.16 | 3.23 | — |
| Speex                | 2.81 | 3.37 | 3.23 | 4.13 |
| WebRTC AEC2          | 3.48 | 4.26 | 2.39 | 4.10 |
| WebRTC AEC3          | 3.88 | **4.54** | 1.85 | 3.45 |
| Ours MILD (新)       | 3.42 | 3.87 | **2.56** | **4.02** |
| Ours SOFT            | 3.64 | 4.05 | 2.38 | 4.02 |
| **Ours BALANCED ★**  | 3.80 | 4.19 | 2.27 | 4.01 |
| Ours AGGRESSIVE      | 3.85 | 4.23 | 2.23 | 4.00 |
| Ours MAXIMUM         | 3.90 | 4.26 | 2.19 | 3.99 |

> Numbers above: MILD = v3.8.3 800-case AECMOS bench (2026-05-05,
> preset=mild, fl=52 ms, CNG on). SOFT = former v3.8.2 MILD numbers
> verbatim (params unchanged, only slot moved). BALANCED+ = v3.8.1
> baseline (filter code unchanged). **For the current release see
> Appendix A2 (v3.10.4 metrics) below.**

NE Degradation ≈ 4.0 is a binding floor under the current
architecture (every preset clusters in 3.99–4.02 regardless of
suppression level). See Appendix C.

---

## Appendix A2 — v3.10.4 release metrics (current)

AEC Challenge 2021 Interspeech blind test set, 800 cases, AECMOS
(scale 1–5, higher better). Bench config: `--filter 832 --cng`,
preset as labeled, j4 parallel.

FS Echo and DT Echo are the n-weighted means of static (169 / 186 cases)
+ movement (131 / 114 cases) sub-buckets; NE = NE bucket only.

| Preset | FS Echo | DT Echo | DT Degradation | NE Degradation |
|---|---|---|---|---|
| Ours MILD            | 3.397 | 3.855 | **2.623** | **4.022** |
| Ours SOFT            | 3.565 | 4.015 | 2.461 | 4.020 |
| **Ours BALANCED ★**  | 3.668 | 4.154 | 2.344 | 4.010 |
| Ours AGGRESSIVE      | 3.686 | 4.178 | 2.319 | 4.006 |
| Ours MAXIMUM         | 3.746 | 4.218 | 2.265 | 3.993 |

Per-bucket breakdown:

| Preset | FS_st | FS_mv | DT_st echo/deg | DT_mv echo/deg | NE echo/deg |
|---|---|---|---|---|---|
| MILD       | 3.332 | 3.480 | 3.888 / 2.632 | 3.802 / 2.608 | 4.998 / 4.022 |
| SOFT       | 3.504 | 3.643 | 4.069 / 2.453 | 3.926 / 2.474 | 4.998 / 4.020 |
| BALANCED ★ | 3.641 | 3.704 | 4.217 / 2.328 | 4.051 / 2.370 | 4.998 / 4.010 |
| AGGRESSIVE | 3.676 | 3.699 | 4.242 / 2.297 | 4.073 / 2.355 | 4.998 / 4.006 |
| MAXIMUM    | 3.748 | 3.743 | 4.285 / 2.240 | 4.109 / 2.307 | 4.998 / 3.993 |

vs v3.8.3 baseline (BALANCED): FS 3.798 → 3.668 (−0.130, locked-in
cost of v3.8.4 Plan A's `[0.1, 0.8, 0.1]` smoothing kernel — see
[CHANGELOG.md](CHANGELOG.md) v3.8.4 / v3.10.4 entries); NE 4.007 →
4.010 (Δ ≈ 0); DT echo 4.186 → 4.154 (−0.032); DT deg 2.272 →
2.344 (+0.072 = Plan A DT-deg gain carried through).

### Resource estimates (16 kHz, 52 ms filter, BALANCED, all features on)

| Item | Value |
|---|---|
| Memory (heap path)              | ~280 KB |
| Memory (static-memory pool)     | ~700 KB (includes scratch) |
| Compute / frame                 | 4 × 512-FFT + Kalman (257 bins × 6 partitions) |
| Convergence time (cold start)   | 0.5–2 s (depends on far-end activity) |
| EPC recovery time               | ~200 ms |
| Algorithmic latency             | 10 ms |

---

## Appendix B — Investigation log

The R1–R16 algorithm investigations (April–May 2026) are recorded in
[SUMMARY.md](SUMMARY.md). High-level outcomes:

| Round | Topic | Verdict |
|---|---|---|
| R1–R9   | Multiple v3.x adjustment axes | mostly null on 800-case |
| R10–R12 | Reverb-tail / stationary-DT refinements | null |
| R13–R14 | Movement-DT capacity / state | confirmed plateau |
| R15     | Feasible-region grid search (8892 configs) | 0 pass — current preset on Pareto front |
| R16     | WebRTC AEC3 missing-state discriminator audit | 0/35 pass |

Conclusion as of v3.8.1: under the current classical-DSP architecture
the score plateau at AEC Challenge 2021 has been reached. Further gains
likely require an NN component (joint NR + RES + dereverb) — see
[~/.claude/plans/jazzy-brewing-castle.md](in user notes).

---

## Appendix C — Known limitations

### C.1 DT-from-frame-0

When near-end speech starts at sample 0 (no far-end-only warmup
period), the filter never enters a clean adaptation window:

- DTD fires from frame 0 → mu_scale ≈ 0
- Filter cannot learn the echo path
- Residual echo persists for the entire utterance

Asymmetry with FS-from-frame-0: there the filter learns the echo path
within 0.5 s and stays converged. The DT-from-frame-0 case never
recovers because there is no "FS-only" window to bootstrap.

**Mitigations explored (v3.6 attempts; null verdict)**:

- Forced WARMUP override of DT freeze → caused FS regressions
- Lowered `dtd_coh_low` threshold → caused echo bleed-through
- Per-frequency DT carve-out → caused musical noise

**Practical advice**: at the application layer, ensure ≥ 0.5 s of
far-end signal precedes near-end speech (e.g., dial tone, "connecting"
cue, ringtone). This is consistent with standard telephony UX.

### C.2 Movement-DT cooccurrence

Mic moving + double-talk is the hardest combined case in the blind
set. Three rounds of ablation (filter capacity, filter state reset,
RES gain) confirmed the bottleneck is the DT gate suppressing
adaptation while the path is changing — downstream RES adjustments
cannot recover it. See `archive/movement_dt_ablation_report.md`.

### C.3 NE Degradation floor (~4.006)

All 5 presets cluster at NE deg ≈ 4.0 on the 800-case blind set. This
is a floor of the current architecture's near-end protection mechanism
(spectral floor + per-bin gate + CNG). Pushing ENR threshold higher
preserves NE further but at unacceptable FS cost. Likely needs an
NN-based per-bin near-end protector to break.

### C.4 Speaker hard-clipping

The linear filter cannot model non-linear distortion. RES + saturation
boost compensates partially, but sustained hard-clipping (e.g.,
overdriven IoT speakers) leaves audible residual. Hardware-side
solution preferred; software cannot fully recover.

### C.5 Sample-rate / channel limits

Hardcoded to 8 / 16 / 48 kHz. Anything else needs upstream resample.
Mono only — no stereo / multi-mic AEC. Negative delay (mic earlier
than ref) is not supported; align upstream.

---

## Appendix D — Removed / legacy paths

### D.1 RES v1 (EER + Wiener)

`res_echo_method = "coherence"` + `res_gain_type = "spectral_sub" |
"wiener"` strings still exist in `AecConfig` but all production presets
force `direct + enr`. The legacy paths are kept only as runtime options
for ablation experiments. They are **not** in the v3.8.1 production
flow.

### D.2 NLMS / FDAF in C

The Python `aec.py` implements all five filter modes for educational
comparison. The C port (`c_impl/`) only ships PBFDKF — there is no
`nlms_filter.c`. Mode selection in C is fixed to PBFDKF.

### D.3 Older preset ranges

Pre-v3 presets had `filter_length = 32 ms` and `kalman_q_low = 1e-5`.
Bumped in v3.6.0 (PR-D1: 32 → 52 ms) and v3 D6 (Q_low: 1e-5 → 1e-6)
respectively. CHANGELOG.md has the full history.

### D.4 `nlms_filter.c`

Earlier docs referenced this file; it never shipped. NLMS is
implemented inline in `pbfdkf.c` for cross-mode debugging only.

### D.5 v1.x / v2.x feature labels

The original methods doc tagged sections with version stamps like
"v1.27.0", "v2.0.0", "v13+". These have been removed from the spec
text — version history belongs in [CHANGELOG.md](CHANGELOG.md) and
[aec_v3_evolution.md](aec_v3_evolution.md).

---

## References

1. Haykin, S. *Adaptive Filter Theory*, 5th ed. Pearson, 2014.
2. Benesty, J., Sondhi, M.M., Huang, Y. *Springer Handbook of Speech
   Processing*. Springer, 2008. (Chapter on AEC)
3. Malik, S. & Enzner, G. "State-Space Frequency-Domain Adaptive
   Filtering for Nonlinear Acoustic Echo Cancellation". *IEEE TASLP*,
   20(7), 2012. (FDKF derivation)
4. WebRTC AEC3 source code:
   `modules/audio_processing/aec3/` (reference for RES design,
   PostprocessGains, render-activity).
5. AEC Challenge Interspeech 2021 Blind Test Set, dataset.
   AECMOS objective metric for evaluation.
6. Soo, J.-S. & Pang, K.K. "Multidelay block frequency domain adaptive
   filter". *IEEE T-ASSP*, 38(2), 1990. (PBFDAF foundation)
