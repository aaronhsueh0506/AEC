# C Port Spec — Python aec.py v3.8.1 → C v3.8.1

Compiled mapping from `python/aec.py` (v3.8.1, commit 3d46fb3+a28a93c) for porting
to `c_impl/`. Companion to `c_rewrite_plan.md` (phase split + parity gates).

This document is the **authoritative reference** for what each Python class /
method must produce in C. When in doubt, refer to specific Python line numbers
quoted here, then read the source.

**Numerical contract**: float32 throughout. `rtol=1e-5 atol=1e-7` per-sample
parity (drift attributable to float32-vs-float64 only, not algorithmic).

---

## Top-level class map

| Python class | Lines | Status | C target |
|---|---|---|---|
| `PBFDAF` | 584-739 | mapped | `c_impl/src/pbfdkf.c` (parent of PBFDKF) |
| `PBFDKF` | 741-910 | **Phase 1 shipped** (`d48b244`) | `c_impl/src/pbfdkf.c` (G1 KX blended) |
| `FilterErleEstimator` | 916-957 | mapped | NEW `c_impl/src/multi_erle.c` |
| `FullbandErleEstimator` | 959-976 | mapped | NEW `c_impl/src/multi_erle.c` |
| `compute_erle_confidence` | 978-986 | mapped | NEW `c_impl/src/multi_erle.c` (function) |
| `HighPassFilter` | 989-1032 | mapped | EXISTS `c_impl/src/hpf.c` (verify parity) |
| `SaturationDetector` | 1034-1093 | mapped | NEW (or inline in `aec.c`) |
| `ResFilter` | 1096-2079 | mapped | REWRITE `c_impl/src/res_filter.c` |
| `ResidualEchoEstimator` | 2507-2636 | mapped | NEW (separate class in `res_filter.c`) |
| `DtdEstimator` | 2080-2160+ | mapped (coherence mode) | NEW `c_impl/src/dtd.c` |
| `RenderActivityDetector` | 2299-2358 | mapped | NEW `c_impl/src/render_activity.c` |
| `FilterConvergenceAnalyzer` | 2369-2435 | mapped | NEW `c_impl/src/convergence.c` |
| `DoubleTalkAnalyzer` | 2437-2505 | mapped | NEW `c_impl/src/dt_analyzer.c` |
| `EchoPathChangeDetector` | 2726-2837 | mapped | NEW `c_impl/src/epc_detector.c` |
| `ShadowCopyController` | 2852-3005+ | mapped | NEW `c_impl/src/shadow_copy.c` |
| `DelayEstimator` | 394-521 | mapped | EXISTS (verify parity) |
| `AEC` | 3009-4358 | mapped | REWRITE `c_impl/src/aec.c` |

---

## Phase 1 status (PBFDKF G1 KX blended) — SHIPPED `d48b244`

Surgical port of v3.7.0 G1 fix on top of v2.5 c_impl base:

```c
/* Python aec.py lines 856-868 → c_impl/src/pbfdkf.c P-update */
float mag2 = X.r * X.r + X.i * X.i;
float KX_optimal = K_scale * mag2;          /* un-mu (= v2.5 behavior) */
float KX_scaled  = K_scale_mu * mag2;       /* per-bin mu */
float KX = mu_mean * KX_optimal + (1.0f - mu_mean) * KX_scaled;
```

Plus `p_max_override` mechanism (matches Python `_p_max_override` /
`_p_max_override_frames`) for EPC events to transiently raise to 1.0.

**Parity status**: build clean. Per-frame parity test infrastructure pending.

---

## Phase 2 plan — Multi-ERLE + DT analyzer + Render activity

### FilterErleEstimator (per-bin)

```python
# Python lines 916-957
self.erle = np.ones(n_freqs, dtype=np.float32)
self._alpha_rise = 0.95
self._alpha_drop = 0.7

def update(echo_spec, error_spec, far_active, dt_indicator):
    if not far_active:
        return
    inst_erle = clip(|echo_spec|² / (|error_spec|² + 1e-10), 0.1, 1000.0)
    dt_factor = clip(dt_indicator, 0, 1)
    a_rise = 0.95 + (1 - 0.95) * dt_factor   # freezes rise on DT
    alpha = where(inst_erle < self.erle, 0.7, a_rise)
    self.erle = alpha * self.erle + (1-alpha) * inst_erle
    self.erle = convolve(self.erle, [0.25, 0.5, 0.25], 'same')  # 3-bin smooth
    self.erle = clip(self.erle, 0.5, 200)
```

C: `multi_erle.c` with `FilterErleEstimator* fe_create(int n_freqs)` and
`fe_update(echo_spec, error_spec, far_active, dt_indicator)`.

### FullbandErleEstimator (scalar)

```python
# Python lines 959-976
self.fb_erle = 1.0
self._alpha = 0.97

def update(near_power, error_power, far_active, dt_indicator):
    if not far_active or dt_indicator > 0.3 or near_power < 1e-8:
        return  # CRITICAL: dt threshold = 0.3 hard gate
    inst = clip(near_power / (error_power + 1e-10), 0.5, 100.0)
    self.fb_erle = 0.97 * self.fb_erle + 0.03 * inst
```

### compute_erle_confidence (scalar return)

```python
# Python lines 978-986
def compute_erle_confidence(erle, fb_erle):
    l1 = mean(erle)
    if l1 < 0.5 or fb_erle < 0.5:
        return 0.0
    log_diff = abs(log(l1 + 1e-10) - log(fb_erle + 1e-10))
    return exp(-log_diff / 2.0)
```

### RenderActivityDetector

```python
# Python lines 2299-2358
ALPHA_CV = 0.99
STATIONARY_CV2 = 0.02

def update(far_end):
    far_pwr = mean(far_end²) + 1e-10
    if far_pwr > 1e-6:
        if not _active_prev:
            _env_mean = far_pwr
            _env_var = 0.0
            _active_prev = True
        else:
            old_mean = _env_mean
            _env_mean = 0.99 * _env_mean + 0.01 * far_pwr
            _env_var = 0.99 * _env_var + 0.01 * (far_pwr - old_mean)²
        cv2 = _env_var / (_env_mean² + 1e-10)
        is_stationary = cv2 < 0.02
    else:
        _active_prev = False
        is_stationary = False
    return RenderActivityState(far_pwr, is_active=_active_prev, is_stationary, warmup_active)
```

### DoubleTalkAnalyzer (energy + shadow DT)

```python
# Python lines 2437-2505
SHADOW_FRAME_GATE = 50

def update_energy_dt(far_active, far_pwr, mic_pwr, erl_estimate):
    if far_active and far_pwr > 1e-4:
        erl_ceiling = 1.0 / max(erl_estimate, 0.01)
        max_echo = far_pwr * erl_ceiling * 2.0   # safety_margin=2.0
        inst = max(0, (mic_pwr - max_echo) / mic_pwr)
    else:
        inst = 0
    if inst > _dt_from_energy:
        _dt_from_energy = 0.3 * _dt_from_energy + 0.7 * inst   # fast rise
    else:
        _dt_from_energy = 0.9 * _dt_from_energy + 0.1 * inst   # slow decay

def update_shadow_dt(shadow_frame_count, far_excited, main_err, shadow_err):
    if shadow_frame_count >= 50 and far_excited:
        _shadow_advantage = main_err / (shadow_err + 1e-10)
        raw = clip((_shadow_advantage - dtd_offset) / dtd_advantage_scale, 0, 1)
        _dt_from_shadow = 0.7 * _dt_from_shadow + 0.3 * raw
    else:
        _dt_from_shadow *= 0.95
```

### FilterConvergenceAnalyzer

```python
# Python lines 2369-2435
CONV_ERLE_DB = 5.0
CONV_FRAMES = 10
DIV_ERLE_LIN = 0.63   # ↔ ERLE < -2 dB
DIV_ALPHA = 0.9
DIV_DECAY = 0.95

def update_convergence(near_pwr, raw_err_pwr, far_active, warmup_done):
    if not far_active or not warmup_done or near_pwr <= 1e-8:
        return False
    inst_erle_db = 10 * log10(near_pwr / (raw_err_pwr + 1e-10))
    if inst_erle_db > 5.0:
        _conv_counter += 1
    else:
        _conv_counter = 0
    if _conv_counter >= 10:
        _converged = True
        _once_converged = True
        return True

def update_divergence(near_pwr, raw_err_pwr):
    if _converged and near_pwr > 1e-8:
        inst = near_pwr / (raw_err_pwr + 1e-10)
        is_div = float(inst < 0.63)
        _divergence = 0.9 * _divergence + 0.1 * is_div
    else:
        _divergence *= 0.95
```

---

## Phase 3 plan — ResFilter v3.8.1

The largest single class (~1000 LOC). Critical structural points:

### 10-step process flow

1. **WOLA analysis** (input_buf shift, sqrt-Hann window, FFT block_size; line 1398-1415)
2. **PSD smoothing** (echo_psd, error_psd via alpha_*; coherence S_fe/S_ff/S_ee; multi-ERLE updates; line 1417-1470)
3. **Dynamic g_min + noise gating** (far_activity EMA, dt_for_fs, effective_dt, epc_dt; line 1472-1504)
4. **Residual echo PSD** via `_residual_est.attribute()` + nonlinear/saturation; cap cascade (echo×2 / error×err_cap_mult / dt_suppress / render_ceil); reverb tail; echo_boost; line 1505-1641
5. **Per-bin gain** (ENR path: dt_per_bin → dt_shaped_per_bin **`** 1.1**`**, soft gate with BUG-3 min_gate_width=0.2, EMR noise-masking lift; lines 1704-1833)
6. **Frequency postprocessing** (3-bin smooth, DC consistency, HF cap; line 1835-1860)
7. **Temporal smoothing** (split attack/release with ERLE-dependent alpha, rate limit, render_dt_gain_ceil=0.6 hard cap when render_based+far_activity>0.3; line 1862-1947)
8. **Spectral floor + NE protect** (computed before gain, used as ceiling; line 1664-1703)
9. **Noise floor gain + CNG** (min-statistics tracker, sqrt-PSD floor, comfort noise injection; line 1952-2001)
10. **WOLA synthesis** (gain × spec_synth, IFFT, sqrt-Hann, OLA; line 1990-2010)

### Critical v3.8 NEGATIVE invariants — DO NOT PORT

These three branches **MUST NOT** appear in C (architectural ablations):

```python
# ABL-1 (line 2609-2612): error_based_floor REMOVED
# DO NOT add: residual_echo_psd = max(residual_echo_psd, error_psd × far_conf × 0.7)

# ABL-2 (line 1531-1535): mic_psd × 0.5 floor REMOVED
# DO NOT add: residual_echo_psd = max(residual_echo_psd, mic_psd × 0.5)

# ABL-4 (line 1643-1650): erl > 1.2 dead branch REMOVED
# DO NOT add: residual_echo_psd = max(residual_echo_psd, error_psd × 0.9)
```

All three were structurally similar errors using NE-contaminated signals as
echo proxy. C-side reviewers must reject any patch reintroducing these.

### Critical POSITIVE invariants — must be present

```python
# v3.8 dt_per_bin ** 1.1 (was 2.0): line 1747
dt_shaped = dt_per_bin ** 1.1

# v3.2 render_dt_gain_ceil = 0.6 (line 1955-1957): KEEP, ablation confirmed load-bearing
if _residual_est.using_render_based and far_activity > 0.3:
    smoothed = min(smoothed, 0.6)

# BUG-3 min_gate_width = 0.2 (line 1769-1773): protects FS soft gate
enr_s_safe = max(enr_s, enr_t + 0.2)

# EMR noise masking (line 1793-1798): raise gain where echo masked by noise
if sum(noise_psd) > 0:
    emr = residual_echo_psd / (noise_psd + 1e-10)
    g_emr = clip(0.3 / (emr + 1e-10), 0, 1)
    g = max(g, g_emr)
```

### ResidualEchoEstimator (separate class, lines 2507-2636)

State: `_using_render_based`, `_render_based_hold` (5-frame hangover).

`attribute()` does ERLE-blended linear (Stage 1) + render-based switch (Stage 2)
with hysteresis (0.05) + force conditions (EPC / saturation / not converged).
Render path: `far_psd × erl_estimate` ONLY (no error-based floor — ABL-1).

---

## Phase 4 plan — Shadow + EPC + delay

### EchoPathChangeDetector (3-trigger state machine)

```python
# Python lines 2726-2837
EPV_FAST_TC = 0.98
EPV_SLOW_TC = 0.999
EPV_LOW = 0.25
EPV_HIGH = 4.0

# Three triggers all return EpcEvent with `fired` and `source` ('delay'/'epv'/'shadow_rise')
def force_delay()  → fired=True, source='delay'
def update_epv(far_pwr, filter_converged, main_paused) → check EPV_LOW < ratio < EPV_HIGH
def update_shadow_rise(main_err, shadow_err, is_stationary)
    → check delta_ratio + total rise; white-noise guard
def tick_hangover()  → caller must gate on (shadow_filter AND filter_converged
                        AND NOT shadow_rise_fired_this_frame)
```

### ShadowCopyController (5-state)

Lines 2852-3005+. 4 gate modes (energy / coherence / coherence_delay / streak_only).
Shadow advantage threshold + hysteresis (5+10 frames). Returns
`ShadowCopyDecision(pause_main, boost_q, reverse_copy)`.

### DelayEstimator

Lines 394-521. GCC-PHAT with EMA cross-spectrum (alpha=0.6). PAR (peak-to-average
ratio) confidence > 5.0 gates first-acquisition. Two-estimate consistency check
(<16 samples diff) gates delay-shift. **Already exists in C** — verify parity.

---

## Phase 5 plan — AEC orchestration

`class AEC` (lines 3009-4358) is wiring, not algorithm.

### process() flow (lines 3625-4266)

1. HPF (optional)
2. Saturation detection
3. Render activity update
4. Saturation emergency mu_scale=0
5. mu_scale (DTD via `_compute_mu_scale()`; non-DTD via `_get_simple_mu_scale()`)
6. startup_dt mu floor
7. Mic clip emergency (mu_scale=0, RES gain → g_min)
8. Delay estimator update + ring buffer
9. PBFDKF main filter process
10. Shadow filter process
11. Shadow advantage + DT update + EPC update + ShadowCopyController
12. EPV / shadow_rise event handling (Q boost, p_max=1.0 30f, p_floor_beta=1.0
    30f, mark_diverged, erl_estimate ≤ 0.3)
13. Multi-ERLE update (FilterErle + FullbandErle)
14. Convergence update
15. ResFilter call with full kwargs
16. Per-bin mu_scale update (post-RES)
17. Output limiting (smoothed gain to prevent peaks)
18. Power tracking (raw vs final)
19. Convergence detection (10 consecutive far-active + ERLE > 5dB)
20. DTD updates for next block
21. Diagnostics dict update
22. Return final_output (or tuple with ctx if return_res_context=True)

### EPC trigger pattern (4 sites)

Same shape, slightly different scope:

```c
/* delay_first (line 3658), delay_shift (line 3675), epv (line 3857), shadow_rise (line 3876) */
pbfdkf_set_q_high(filter);                          /* All */
pbfdkf_set_q_high(shadow_filter);                   /* All */
pbfdkf_set_p_max_override(filter, 1.0f, 30);        /* delay_shift, epv, shadow_rise */
pbfdkf_set_p_max_override(shadow_filter, 1.0f, 30); /* delay_shift, epv, shadow_rise */
pbfdkf_set_p_floor_epc(filter, 1.0f, 30);           /* epv, shadow_rise */
pbfdkf_set_p_floor_epc(shadow_filter, 1.0f, 30);    /* epv, shadow_rise */
maybe_mark_diverged(SOURCE);                        /* All */
self->_erl_estimate = MIN(self->_erl_estimate, 0.3f); /* epv, shadow_rise */
self->_epc_render_forced_remaining = config.epc_hangover; /* epv, shadow_rise */
```

### AecResContext fields (extended in Phase 0)

Already declared in `c_impl/include/aec_types.h` (Phase 0 commit `bcf0f99`):
- echo/far/near_spec re/im
- far_power, filter_converged, filter_once_converged, erle_factor, dt_indicator
- divergence, over_sub, n_freqs, is_stationary_dt, saturation_level, erl_estimate
- shadow_dt, e2_main, e2_shadow, y2, epc_active

NR insertion contract: caller multiplies `echo_spec` by per-bin NR gain before
passing to RES. Pipeline pattern verified in `Audio_ALG/pipelines/aec_nr_pipeline.c`.

---

## Parity test infrastructure (Phase 1+ requirement)

### Python side (`python/parity_pbfdkf.py`)

Generate deterministic synthetic input, run Python PBFDKF, save state per-frame:

```python
np.random.seed(42)
mic = np.random.randn(N_FRAMES * HOP).astype(np.float32) * 0.1
ref = np.random.randn(N_FRAMES * HOP).astype(np.float32) * 0.1
filt = PBFDKF(block_size=320, n_partitions=5, mu=0.3, delta=1e-8, hop_size=160)
output = []
P_states = []  # per-frame copy of P
W_states = []  # per-frame copy of W
err_specs = [] # error_spec (for direct comparison)
for f in range(N_FRAMES):
    o = filt.process(mic[f*HOP:(f+1)*HOP], ref[f*HOP:(f+1)*HOP], mu_scale=0.7)
    output.append(o)
    P_states.append(filt.P.copy())
    W_states.append(filt.W.copy())
    err_specs.append(filt.error_spec.copy())
np.savez('parity_pbfdkf_python.npz',
         mic=mic, ref=ref, output=np.array(output),
         P_states=np.array(P_states), W_states=np.array(W_states),
         err_specs=np.array(err_specs))
```

### C side (`c_impl/example/parity_pbfdkf_test.c`)

Load same input, run, dump matching state, write to .npz-compatible binary.

### Compare

```python
py = np.load('parity_pbfdkf_python.npz')
c  = np.load('parity_pbfdkf_c.npz')
assert np.allclose(py['output'], c['output'], rtol=1e-5, atol=1e-7), \
       f"output max diff: {np.max(np.abs(py['output'] - c['output']))}"
# similarly P, W, err_specs
```

If any phase exceeds tolerance: halt + root cause before proceeding.

---

## Status snapshot

- ✅ Phase 0: scaffolding + plan + integration guide + aec_types.h (`bcf0f99`)
- ✅ Phase 1: PBFDKF G1 KX blended P-update + p_max_override (`d48b244`)
- ⏸️ Phase 1 parity test: framework documented, implementation pending
- ⏸️ Phase 2-5: design specs above, implementation pending

Each subsequent phase:
1. Implement per spec
2. Build clean
3. Per-component parity test rtol=1e-5 atol=1e-7
4. Commit with phase-prefixed message
5. Halt if parity exceeds tolerance — root cause first
