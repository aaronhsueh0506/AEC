# AEC Algorithm Methods

This document is the deep algorithm specification for the production pipeline:
a PBFDKF linear stage followed by an AEC3-aligned post-filter chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + CNG). The Python
reference (fp64) is the **algorithm spec**; the C port is the float32
**implementation** and agrees with Python to a tolerance (§7). The C structure
mirrors Python class boundaries.

[C integration](c_user_manual_zh_TW.md) · [Linear front end](reference/linear_frontend.md) · [Post-filter](reference/postfilter.md) · [NN integration](nn_integration_interface.md)

## 1. Pipeline overview

```
mic ─► HPF ────────────────────────────────────────────────────────────────────────►
ref ──────► Saturation ─► EchoPathDelayEstimator ─► reference ring ─────┐
                          (matched filter +                             │
                           lag aggregator +                             ▼
                           clock-drift detector)                   PBFDKF refined
                                                                   filter (Kalman)
                                                                          │
                                                                  Shadow filter (PBFDAF/NLMS,
                                                                  μ = 0.5) + PathChange-
                                                                  RegimeHandler (load-bearing
                                                                  on cohort tail)
                                                                  + RenderSignalAnalyzer
                                                                          │
                                                                       error (e[k])
                                                                          │
                                                                          ▼
                                                                   AEC3 _aec3_post
                                                                   ├─ StationarityEstimator
                                                                   ├─ AecState (read-only ADT
                                                                   │     over its sub-analyzers)
                                                                   ├─ ResidualEchoEstimator
                                                                   │     (per-bin echo PSD)
                                                                   ├─ SuppressionGain
                                                                   │     (Wiener + over-est)
                                                                   ├─ Comfort noise (per-bin
                                                                   │     PSD EMA)
                                                                   └─ sqrt-Hann synthesis OLA
                                                                          │
                                                                          ▼
                                                                         out
```

### Presets — one production point + three operating points

`AecPreset.BALANCED` is the production preset. It shares a narrow tuned base
with the other presets:

```python
AecConfig.from_preset(AecPreset.BALANCED)  # base, equivalent to:
AecConfig(
    enable_cng=True,        # comfort-noise generator on
    shadow_mu_min=0.5,      # shadow / DT mu floor
    warmup_frames=100,      # forced-high-mu startup window
    kalman_q_high=1e-3,     # PBFDKF Q_high convergence speed
)
```

`mild` and `aggressive` are deliberate **Pareto operating points** on the same
strength axis as `balanced`. `mild`/`balanced` differ in exactly one field —
`min_gain_floor_far_active_db`, the far-active min-gain floor. `aggressive`
additionally deepens `min_gain_floor_dt_db` (the floor used in its place
during double-talk, see [§3.4](reference/postfilter.md#34-suppressiongain)): without that second field, `dt_protect_active`
hard-overrides the far-active floor with the SAME shared −20 dB for every
preset whenever it fires (it also false-arms in far-end single talk), so the
presets would be indistinguishable on exactly the hops where a user would
expect to hear the strength difference. `no_floor` is a fourth tier that
removes the floor mechanism (`min_gain_split_floor_enabled=False`) entirely
instead of deepening it further — the deepest point on the axis, with the most
near-end cost.

| preset | far-active floor | dt floor | character |
|---|---|---|---|
| `mild` | −20 dB | −20 dB | near-priority — more near kept, more echo leak |
| `balanced` | −28 dB | −20 dB | production |
| `aggressive` | −38 dB | −30 dB | echo-priority — dev-pool (47-case) DT_static echo +0.17/+0.14 MOS vs the shared-floor point, at a bounded near-end cost |
| `no_floor` | n/a (floor off) | n/a (floor off) | deepest point on the axis — purely audibility/R²-derived gain, no fixed-dB clamp at all; most near-end loss, and it exposes the cold-start R² behaviour the floor otherwise masks |

The DT-deg-vs-echo trade is a single-channel DSP Pareto wall; the strength
axis exposes it.

Everything else uses `AecConfig` dataclass defaults; the AEC3 chain has
its own tuning embedded in `AecStateConfig` / `SuppressorConfig`
(`modules/state/aec_state.py`, `modules/residual/suppression_gain.py`).

### Determinism

Comfort noise uses an internal LCG (`seed = seed·69069 + 1`) seeded with 42 at
construction and at reset, in both Python and C; the algorithm consumes no
global RNG state, so a render is reproducible per instance.

### Tunable parameters (`AecConfig`)

`AecConfig` (`modules/config.py`) is the full release surface; the table lists
its principal fields. mild/balanced differ **only** in
`min_gain_floor_far_active_db` (−20 / −28); aggressive additionally sets
`min_gain_floor_dt_db` to −30 (shared −20 default otherwise); `no_floor`
instead sets `min_gain_split_floor_enabled=False`. `from_preset` additionally
pins a shared base (`enable_cng=True`, `shadow_mu_min=0.5`, `warmup_frames=100`,
`kalman_q_high=1e-3`). Everything below uses dataclass defaults. Fields are
grouped by subsystem; **ON**/**OFF** marks the default state of a feature flag.
Frame-count fields marked as a 10 ms authored basis are converted at
construction to preserve wall-clock duration on the selected grid; the table
shows their public configuration values, not necessarily the internal hop count.

| Subsystem | Field | Default | Notes |
|---|---|---|---|
| System / framing | `sample_rate` | 16000 | 8000/16000/48000 |
| | `frame_size` / `hop_size` | auto | frame=FFT、hop=frame/2；8k 256/128、16k 256/128（可選512/256）、48k 1024/512 |
| | `filter_length` | auto | 52 ms (<44.1 kHz), 64 ms (≥44.1 kHz) |
| | `mu` / `delta` | 0.3 / 1e-8 | step size / regularization |
| Residual / CNG | `enable_res` | **ON** | post-filter master gate ([§3.7](reference/postfilter.md#37-enable_res--false-bypass)) |
| | `enable_cng` | OFF (preset **ON**) | comfort noise ([§3.5](reference/postfilter.md#35-comfort-noise-cng)) |
| | `cng_n2_windowed_psd` | 0.0 (**OFF**) | feeds the N2 tracker a windowed capture PSD instead of a rectangular one |
| | `comfort_noise_floor_dbfs` | −96.034 | |
| | `enable_td_constraint` | **ON** | |
| C′ ERLE coherence gate | `erle_coh_gate_enabled` | **ON** | Γ²(Ŷ,Y) per-bin ERLE freeze |
| | `erle_coh_gate_threshold` / `_alpha` | 0.5 / 0.05 | |
| Split min-gain floor ([§3.4](reference/postfilter.md#34-suppressiongain)) | `min_gain_split_floor_enabled` | **ON** | OFF for the `no_floor` preset |
| | `min_gain_floor_far_active_db` | −28 | **preset strength knob** (mild −20 / balanced −28 / aggressive −38) |
| | `min_gain_floor_far_silent_db` | −12 | pure-NE floor |
| | `min_gain_far_latch_power` | 1e6 | far-active latch |
| **DT-aware recovery group** | `dt_aware_recovery_soft` | **ON** | soft (non-destructive) delay/EPC recovery ([§2.3](reference/linear_frontend.md#23-delay-estimator-aec3-aligned)) |
| | `dt_aware_res_floor_enabled` | **ON** | RES floor lift ([§3.4](reference/postfilter.md#34-suppressiongain)); default source is the near-recent latch alone |
| | `min_gain_floor_dt_db` | −20 | floor used in double-talk; **−30 for aggressive**, shared −20 for mild/balanced |
| | `ne_recent_threshold` | 0.3 | near-recent gate arm threshold |
| | `ne_recent_hold` | 150 | 10 ms authored basis; approximately 1.5 s after grid retiming |
| | `ne_recent_sustain` | 3 | frames above threshold to arm |
| Soft near-end blend | `soft_nearend_blend_enabled` | **ON** | sigmoid ENR tuning blend |
| | `soft_nearend_blend_enr_threshold` / `_softness` | 0.25 / 0.25 | |
| | `soft_nearend_blend_per_bin` | **ON** | per-bin frequency-selective near protection |
| Nonlinear residual | `nl_r2_enabled` | **ON** | Kuech-Kellermann 2nd-order R² (L1) |
| | `nl_r2_alpha` | 0.1 | |
| | `nl_r2_cap_multiple` | 300 | caps r2_nl at 300x the direct term; the raw formula is unbounded (fixed -20dBFS reference, not signal-level-relative) and reaches 1e16-1e18 on loud cold-start render |
| ERLE coordinate | `erle_render_x2_psd_scale` | **ON** | render PSD scale feeding the reverb model |
| | `reverb_tail_strength` | 1.0 | reverb-tail R² scale |
| | `erle_windowed_capture_psd` | **ON** | Y2/E2 sqrt-Hann coordinate consistency |
| | `output_capture_when_linear_unusable` | **ON** | output guard: when the linear estimate is unusable this hop and the residual energy exceeds the capture energy, the rendering base is the capture Y instead of E; decided once per hop in every API mode, never alters the analysis seam ([§3.8](reference/postfilter.md#38-output-guard-rendering-base-and-post-reset-serial)) |
| Delay-acquire guard | `delay_acquire_protect_converged` | **ON** | reject spurious late FIRST acquisition (Path A only) |
| Cold-start deadlock (OFF) | `h_error_refresh_erl_floor` | 0.0 (**OFF**) | Kalman-gain cold-start breakers |
| | `h_error_floor_override` | 0.0 (**OFF**) | |
| Shadow filter | `enable_shadow` | **ON** | PBFDAF/NLMS coarse filter |
| | `shadow_mu_min` / `shadow_mu_nlms` | 0.5 / 0.5 | |
| | `shadow_copy_threshold` / `_hysteresis` | 0.65 / 3 | |
| | `shadow_err_alpha` | 0.80 | |
| | `shadow_dtd_advantage_scale` / `shadow_dtd_offset` | 3.0 / 1.5 | |
| Misadjustment estimator | `filter_misadjustment_*` | 100 / 30 / 0.5 / 2.0 | hangover/stable use 10 ms authored basis; scale min/max are dimensionless |
| PBFDKF (Kalman) | `use_kalman` | **ON** | |
| | `kalman_q_high` / `kalman_q_low` | 1e-3 / 1e-6 | Python only (the C filter keeps no `Q`) |
| | `warmup_frames` | 80 (preset 100) | 10 ms authored basis; preset remains approximately 1.0 s after grid retiming |
| EPC | `epc_delta_threshold` / `epc_total_rise` / `epc_hangover` | 0.3 / 1.5 / 20 | |
| Delay estimation | `enable_delay_est` | **ON** | deprecated mirror of `delay_mode` |
| | `max_delay_ms` / `delay_buffer_ms` | 1024 / 2048 | `delay_buffer_ms` is Python only; in C `max_delay_ms` is an internal value no ring reads (the C ring, single channel and array, = estimator reach + hop) |
| | `delay_est_period_s` / `delay_est_init_s` | 0.5 / 0.3 | |
| | `delay_par_low_threshold` / `_solid_threshold` | 5.0 / 8.0 | |
| HPF | `enable_highpass` | **ON** | mic-path only |
| | `highpass_cutoff_hz` | 80.0 | |
| Saturation | `enable_saturation_detect` | **ON** | |
| | `saturation_threshold` | 0.95 | |
| | `saturation_softclip_ref` | **ON** | |
| Mode / output | `mode` | `PBFDKF` | |
| | `return_res_context` | OFF | switches `process()` return type (§4); with `enable_res=False` it selects the context-only seam, which publishes the same analysis seam and rendering base as the full mode ([§3.8](reference/postfilter.md#38-output-guard-rendering-base-and-post-reset-serial)) |
| | `clear_filter_history` | OFF | |

Defaults above are the Python `AecConfig` dataclass values. The C library keeps them in an internal `AecTuning` (all of them except the Python-only
`kalman_q_high` / `kalman_q_low`, `delay_buffer_ms` and `epc_mu_floor`, which C does not have) apart from the public fields
(`c_impl/include/internal/aec_tuning.h`, built by `aec_resolve()` from the sample rate, the grid, `n_mics` and the level; the public `AecConfig` has 12 fields
and callers cannot set the others). The C defaults differ from the Python dataclass in three values: `enable_cng` (C: 1, a public field), `warmup_frames` (C: 100) and
`delay_est_period_s` (C: 0.0, i.e. the matched filter analyses every hop). The C `AecConfig` does not have the Python flag pair
`enable_res` / `return_res_context` either: its single field `residual` (`AecResidualMode`, default `AEC_RESIDUAL_FULL`)
covers them (Python `enable_res=True` = C `FULL` driven by `aec_process()`; `enable_res=False, return_res_context=True` =
C `FULL` driven by `aec_process_linear()` + `aec_process_residual()`; both false = C `OFF`), and the C `AEC_RESIDUAL_STATE` /
`AEC_RESIDUAL_ESTIMATE` modes skip the residual-echo half or the gain of the post chain for multi-lane callers.
The Python names in the rest of this section are the Python twin's own.

The **DT-aware recovery group** has two halves:
`dt_aware_recovery_soft` softens the delay/EPC recovery ([§2.3](reference/linear_frontend.md#23-delay-estimator-aec3-aligned)), gated on the
`ne_recent_*` near-recent latch, and `dt_aware_res_floor_enabled` lifts the RES
floor ([§3.4](reference/postfilter.md#34-suppressiongain)), gated on the same `ne_recent_*` latch. The exact per-hop floor
decision is exported to fused consumers as `res_floor_protect`.

---

## 4. Diagnostic surfaces (load-bearing — do not remove)

### `AecStats` / `get_stats()`
Per-frame audio-passive trace (`modules/dataclasses.py` + the
`AEC.get_stats` method in `modules/orchestrator.py`). Consumed by
`run_one_case.py` plots and external research tooling.

### `AecResContext`
Exposes `echo_spec` / per-bin Kalman state so the linear stage can feed
an external (or NN) post-filter. Enabled via
`AecConfig.return_res_context = True`; switches `aec.process()` return
type to `(out, AecResContext)`.

The context carries two separate groups of spectra
([§3.8](reference/postfilter.md#38-output-guard-rendering-base-and-post-reset-serial)):

* **Analysis seam** — `formed_output`, `error_spec`, `echo_spec`, `near_spec`
  (`= error_spec + echo_spec`). The real selected linear residual and its echo
  estimate, a matched set, in every mode; the output guard never alters them,
  and a unity IFFT with overlap-add of `error_spec` reproduces the previous
  formed hop on every hop.
* **Rendering base** — `output_base_spec` and `guard_active`. The guard is
  evaluated once per hop in every mode; a consumer that renders audio itself
  multiplies `output_base_spec` (not `error_spec`) by its gain(s), adds comfort
  noise and synthesizes. The base does not pair with `formed_output`.

`post_reset_serial` is the non-consuming hard-reset counter
(`AEC.get_post_reset_serial()`): on a change the consumer clears its OLA tail and
reseeds its CNG to `AEC_CNG_SEED` before synthesizing that hop. NR and NN state
are the application's to reset. `AecConfig.return_formed_output` requires this
flag or `enable_res` and raises at construction otherwise — the formed seam is
produced by the post chain and cannot be requested without it.

### Regime classifier
`modules/p52_regime_classifier.py` is analysis-only; the
`AntiLoopholeTests` in `python/tests/test_p52_regime.py` enforce that the
classifier never modifies audio.

---

## 7. Python ↔ C precision model

The `c_impl/` port is the production C implementation of the **same algorithm**
as `python/aec.py`. All production C is float32 end-to-end: delay chain,
orchestrator scalars, post/state modules, `residual_echo_estimator`, and the
mic-path HPF. The relationship between the two implementations is:

* **Python (fp64) is the algorithm spec.**
* **C (float32) is the implementation.**
* **Python↔C comparison is tolerance-based**, not 0/0.

The delay-estimator matched filter is float32 fast-math and duty-cycle capable
(see [§2.3](reference/linear_frontend.md#23-delay-estimator-aec3-aligned) and `c_impl/include/internal/delay_aec.h`).

The **FFT backend is NE10 (float32, ARM NEON)** (via the shared
`audiolib` layer), which is not bit-exact to numpy's fp64 `np.fft`. Because
PBFDKF and the post-filter are frequency-domain, this float32 FFT difference
propagates through the whole pipeline, so the **end-to-end C output aligns
with Python to ~float32 precision, not 0/0**. `parity_aec_e2e.c` asserts the
linear `raw_output` and final `out[hop]` stay within a **2e-2 float32
tolerance** at 16 kHz (1e-1 at 48 kHz; measured maxima are tabulated in the
file header) and is the **authoritative C regression gate**.

### Math backends

The shared `audiolib/common/utility/fast_math.h` (sibling repo, header-only)
provides two implementations selected at compile time (orthogonal to the
FFT-backend choice above):

* **Default (production): `fast_math.h` approximations.** `fast_exp` (LUT +
  Taylor), `fast_log`/`fast_log10` (IEEE-754 range reduction + degree-4
  minimax), and `fast_sqrt` (correctly-rounded ARM sqrt: AArch64 hardware
  FSQRT or ARM32 `sqrtf` through hardware/libm; Newton-Raphson is the portable
  non-ARM fallback). `fast_exp` has a
  worst relative error of **3.9e-3** (about 0.034 dB on a gain), which reaches
  the stages that call `exp` (SuppressionGain near-end sigmoid, CNG gain, dB
  conversions) on top of the float32-FFT tolerance above; the ARM
  `fast_sqrt` is correctly rounded and adds none.
* **`-DUSE_STANDARD_MATH`: libm.** Swaps the approximations for `expf`/`logf`/
  `sqrtf` (the `USE_STANDARD_MATH` block at the top of `fast_math.h`); a
  precision/perf toggle only.

`-ffp-contract=off -fno-math-errno` is mandatory in `CFLAGS` on both backends
and on both math modes — the first flag keeps builds deterministic (no fused
multiply-add reassociation), the second makes guarded AArch64 `sqrtf` lower to
`FSQRT` without an errno fallback. The DSP pipeline does not read math
`errno`. The C structure mirrors the Python class boundaries one-to-one
(`PBFDKF`, `AecState`, `SuppressionGain`, `EchoPathDelayEstimator`, …), so a
module-level regression isolates any divergence to a single stage. See
`docs/c_user_manual_zh_TW.md` (build + integration).
