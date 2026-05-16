# v3.14 Arc-P Sprint 02: Adaptive Per-Band ERL EMA — Design + Scaffold

**Date**: 2026-05-14
**Branch**: `feature/v3.14-arc-p`
**Sprint**: P.S2 — Design + wire adaptive per-band ERL EMA + post-EPC cap fix
**Status**: SCAFFOLD COMPLETE — byte-equal verified (5/5 PASS, atol=0.0)

---

## 1. Motivation (from P.S1 findings)

P.S1 audit on 8 worst-FS cases established that the original hypothesis was
**reversed**: scalar `erl_estimate=0.3` (post-EPC cap) **over-estimates** ERL
in low-coupling rooms rather than under-estimating it.

Key evidence (case 04, 742 converged frames):

| Band   | True ERL | Scalar 0.3 | Over-estimate factor |
|--------|----------|------------|----------------------|
| LF 0-1k| 0.043    | 0.300      | 7×                   |
| MF 1-4k| 0.191    | 0.300      | 1.6×                 |
| HF 4-8k| 0.111    | 0.300      | 2.7×                 |

**Consequence**: `far_lw × 0.3 >> far_lw × 0.043` → `excess = error - far_lw × 0.3 ≈ 0`
→ `dt_per_bin ≈ 0` even though there is real near-end residual. The F3.1-v3
excess formula under-reports NE evidence in FS frames because the echo mask
(ERL floor) is too conservative.

**Inter-room variance** is 11× (LF: 0.043 vs 0.489 across two rooms), which
falsifies a fixed per-band ERL table approach. Adaptive per-band EMA is the
canonical solution.

---

## 2. Design

### 2.1 Adaptive per-band ERL EMA

**Source signal**: `|echo_spec[k]|² / |far_spec[k]|²` per bin, aggregated per band.

- `echo_spec` comes from the PBFDKF main filter output (W·X) — the predicted
  echo spectrum in the far-end reference frame.
- `far_spec` is `|X_buf[0]|²` — the current far-end spectrum.
- In a converged filter, `|W·X[k]|² / |X[k]|² ≈ |Ĥ(k)|²` where `Ĥ(k)` is
  the estimated room transfer function — a direct proxy for per-band ERL.

**3 bands**:

| Band | Hz range  | Bin range (16 kHz, fft=640) |
|------|-----------|-----------------------------|
| LF   | 0–1000    | bins 0–39                   |
| MF   | 1000–4000 | bins 40–159                 |
| HF   | 4000–8000 | bins 160–256                |

**EMA time constant**: α = 0.99 → TC ≈ 100 frames ≈ 1 second at hop=160.

**Justification**: Jung (2011) §IV.A uses similar EMA TC (~1 s) for room
impulse response estimation in acoustic echo cancellers. ERL changes only on
room/position changes, so TC = 1 s is conservative (fast relative to room
change) while avoiding over-sensitivity to DT contamination. P.S3 tune sprint
will sweep α ∈ {0.99, 0.995, 0.999} on the 800-case bench.

**Update gating** (conservative, avoids polluting EMA with bad filter states):

```
update fires iff:
  (1) raw_dt_ratio < 2.0           (NE-corruption guard, same as scalar _erl_estimate)
  (2) inst_erl_raw < 1.5           (physical plausibility, same as scalar)
  (3) far_pwr > 1e-4               (far-end active, same gate or hysteresis if f_e1_enabled)
  (4) _prev_filter_state in        (filter output reliable — avoids startup noise
      ('refined_usable',            and diverged-state echo_spec artefacts)
       'coarse_learning')
  (5) filter has echo_spec and     (PBFDKF required)
      far_spec attributes
```

Note: gate (4) uses `_prev_filter_state` (previous frame) to avoid reading the
current frame's state before it is computed (same pattern as `shadow_mu_state_aware`).

**Per-bin safety clipping**: `[0.005, 1.5]` — prevents extreme outliers from
anchoring the EMA. Clip_lo = 0.005 (below P.S1 converged p10 minimum) ensures
the EMA can reach real low-coupling values. Clip_hi = 1.5 is above the 1.0
inst_erl_raw rejection guard and provides a safety net for numerical noise.

### 2.2 Post-EPC per-band cap

**Problem**: The existing post-EPC cap `min(self._erl_estimate, 0.3)` (aec.py
lines 6265 and 6304, the EPV and shadow_rise paths) holds the scalar `_erl_estimate`
at 0.3 when true ERL is 0.04-0.11 in low-coupling rooms. This persists the
over-estimation for the EPC hangover window (~20 frames = 200 ms).

**Fix (flag-ON only)**: Replace the scalar `0.3` cap for the per-band EMA with
wider per-band conservative caps derived from P.S1 clip_hi recommendations:

| Band | cap value | Rationale |
|------|-----------|-----------|
| LF   | 0.6       | P.S1 converged max LF ~0.489 (case 08) + margin |
| MF   | 0.8       | P.S1 converged max MF ~0.799 (case 08) + margin |
| HF   | 1.0       | P.S1 converged max HF ~1.383 (case 08); ERL can exceed 1.0 in HF |

The scalar `_erl_estimate` cap at `min(current, 0.3)` is **unchanged** when
flag is OFF (byte-equal guarantee). When flag is ON, only `_per_band_erl` is
capped per-band; `_erl_estimate` scalar still gets `min(current, 0.3)` (it is
used by the render-based echo path and other consumers not part of P.S2).

### 2.3 F3.1-v3 formula wiring

When `f3_1_per_band_erl_adaptive=True`, the per-bin ERL array `_erl_pb[k]`
is built from the 3-band EMA values and passed as `erl_estimate` to
`ResFilter.process()`:

```python
_erl_pb = np.empty(n_freqs, dtype=np.float32)
_erl_pb[:bin_1k]      = per_band_erl[0]  # LF
_erl_pb[bin_1k:bin_4k] = per_band_erl[1]  # MF
_erl_pb[bin_4k:]      = per_band_erl[2]  # HF
```

Inside `_stage_gain_compute`, the existing `isinstance(erl_estimate, np.ndarray)`
guard routes to per-bin mode in the F3.1-v3 excess formula:

```python
excess = error_psd - far_lw * erl_e   # erl_e: scalar or per-bin array
```

This is mathematically identical to the scalar path when erl_e is constant —
no approximation introduced by vectorisation.

### 2.4 Mathematical byte-equal proof

**When `f3_1_per_band_erl_adaptive=False` (default)**:

1. `_per_band_erl` is initialised to `[0.1, 0.1, 0.1]` but the update gate
   at `if self.config.f3_1_per_band_erl_adaptive:` is never entered.
   → `_per_band_erl` stays at initial values, never read by any hot path.

2. `_erl_arg = self._erl_estimate` (scalar float).
   → `res.process(..., erl_estimate=self._erl_estimate)` — identical to pre-P.S2.

3. Inside `_stage_gain_compute`: `isinstance(erl_estimate, np.ndarray)` is
   False → `erl_e = float(erl_estimate)` → same as pre-P.S2 path.

4. Post-EPC caps: `if self.config.f3_1_per_band_erl_adaptive:` block is skipped.
   → `_erl_estimate = min(self._erl_estimate, 0.3)` unchanged.

5. `_stage_residual_model` Cap 4: `_erl_scalar = float(erl_estimate)` (scalar
   float path) → identical to pre-P.S2.

**Conclusion**: Every conditional added in P.S2 is guarded by
`self.config.f3_1_per_band_erl_adaptive` or `isinstance(erl_estimate, np.ndarray)`.
When flag=False, no new code path is entered and no state is modified.
Output is bit-for-bit identical.

---

## 3. Implementation Summary

### 3.1 Config flags added (`AecConfig`, ~line 665)

```python
f3_1_per_band_erl_adaptive: bool = False   # master enable (default OFF)
per_band_erl_alpha: float = 0.99           # EMA TC ≈ 100 hops
per_band_erl_cap_lf: float = 0.6          # post-EPC cap [LF]
per_band_erl_cap_mf: float = 0.8          # post-EPC cap [MF]
per_band_erl_cap_hf: float = 1.0          # post-EPC cap [HF]
per_band_erl_clip_lo: float = 0.005       # EMA safety floor
per_band_erl_clip_hi: float = 1.5         # EMA safety ceiling
```

### 3.2 State added (`AEC.__init__`)

```python
self._per_band_erl = np.array([0.1, 0.1, 0.1], dtype=np.float64)
```

Initialised to 0.1 (same as scalar `_erl_estimate`). Reset in both
`AEC.reset()` and `_reset_filter_derived_state()`.

### 3.3 Update logic (`AEC.process()`, inside `if erl_update_gate:` block)

Fires only when:
- Flag ON
- `_prev_filter_state in ('refined_usable', 'coarse_learning')`
- Filter has `echo_spec` and `far_spec` attributes (PBFDKF)
- Same raw_dt_ratio and inst_erl_raw guards as scalar ERL

Per-band mean of `|echo_spec[k]|² / |far_spec[k]|²` → clipped → EMA updated.

### 3.4 Post-EPC caps (EPV path line 6317; shadow_rise path line 6356)

Both paths: when flag ON, `_per_band_erl[b] = min(_per_band_erl[b], cap[b])`.
Scalar `_erl_estimate = min(current, 0.3)` **unchanged**.

### 3.5 `res.process()` call

When flag ON: builds `_erl_pb` (per-bin float32 array) from `_per_band_erl`.
Passes it as `erl_estimate=_erl_pb`.
When flag OFF: passes `erl_estimate=self._erl_estimate` (scalar float). Identical
to pre-P.S2.

### 3.6 `_stage_gain_compute` (F3.1-v3 excess formula)

Added `isinstance(erl_estimate, np.ndarray)` guard:
- Array: `erl_e = erl_estimate` (per-bin, shape n_freqs)
- Scalar: `erl_e = float(erl_estimate)` (legacy path)

### 3.7 Backward-compat fixes (scalar users of `erl_estimate`)

- `_stage_residual_model` Cap 4: extracts `_erl_scalar = mean(erl_estimate)`
  for the broadband `render_ceil = far_psd × min(erl_scalar×2, 1.0)` ceiling.
- `_stage_gain_postprocess` HF cap conditional: extracts scalar from HF-band
  slice of per-bin array.
- `_stage_gain_compute` `_unified_gain_floor` path: uses per-bin array directly
  (same `isinstance` guard as F3.1 path).

### 3.8 Diagnostics

Per-frame diag fields added (always present, zero-cost when flag OFF):
```python
self._diag['per_band_erl_lf'] = float(self._per_band_erl[0])
self._diag['per_band_erl_mf'] = float(self._per_band_erl[1])
self._diag['per_band_erl_hf'] = float(self._per_band_erl[2])
```

---

## 4. 5-Case Byte-Equal Verification Results

**Script**: `tools/research/v3_14_p_s2_byte_equal.py`
**Config**: preset=balanced, fl=832, cng=True, seed=42

| Case       | Stem (short)       | flag-OFF max|Δ| | Result |
|------------|--------------------|-----------------|--------|
| NE         | 014AzuqPZku...     | 0.000000e+00    | PASS   |
| FS_static  | 0KjzXA3g20q...     | 0.000000e+00    | PASS   |
| FS_mvmt    | 0I0XMl3M0EC...     | 0.000000e+00    | PASS   |
| DT_static  | 0I0XMl3M0EC...     | 0.000000e+00    | PASS   |
| DT_mvmt    | 49IIo03GZ0C...     | 0.000000e+00    | PASS   |

**BYTE-EQUAL RESULT: ALL 5 CASES PASS (atol=0.0)**

Flag-ON diffs (informational, confirming per-band path fires):

| Case       | flag-ON max|Δ|  |
|------------|-----------------|
| NE         | 0.000000e+00    |
| FS_static  | 0.000000e+00    |
| FS_mvmt    | 1.413e-02       |
| DT_static  | 1.490e-02       |
| DT_mvmt    | 9.171e-03       |

NE and FS_static show zero flag-ON diff because the update gate
(`_prev_filter_state in ('refined_usable', 'coarse_learning')` + PBFDKF echo_spec
available) may not fire in these cases, or the per-band ERL converges to values
that produce the same excess as the scalar path for these specific clips.

### Case 04 per-band ERL convergence (P.S1 primary target)

- Converged frames: 581 / 2376 (24.5%)
- Per-band ERL (all converged frames):
  - LF mean = 0.567 (P.S1 oracle truth: 0.043 — see note below)
  - MF mean = 0.603 (P.S1 oracle truth: 0.191)
  - HF mean = 0.726 (P.S1 oracle truth: 0.111)
  - scalar mean = 0.356 (P.S1 oracle truth: ~0.3 cap)

**Note on source signal discrepancy**: P.S1 measured `error_psd / far_psd` in
converged frames. P.S2 uses `|echo_spec|² / |far_spec|²` from PBFDKF
(`W·X·conj(X) / |X|²`). In a perfectly converged room with stationary path,
these should match (`echo_spec ≈ true_echo ≈ far × H(k)` → `|echo|² / |far|² ≈
ERL`). The discrepancy (P.S2: 0.57–0.73 vs P.S1: 0.04–0.19) suggests the PBFDKF
filter W for case 04 over-estimates the echo transfer function magnitude relative
to the true room coupling measured from `error_psd`. This is a **P.S3 investigation
item**: the source signal selection (echo_spec vs error_psd vs long_window_far_psd)
needs to be re-validated on the 800-case corpus.

---

## 5. Recommendations for P.S3 Tune Sprint

### 5.1 Source signal re-evaluation (HIGH PRIORITY)

P.S2 uses `|echo_spec|² / |far_spec|²` from PBFDKF. The P.S1 audit used
`error_psd / far_psd` (converged-frame residual). The discrepancy suggests:

- Option A: Use `self.res.echo_psd / far_psd` (smoothed echo PSD from ResFilter)
  — more robust to single-frame PBFDKF noise, slower but more stable.
- Option B: Use `self.res.error_psd / far_psd` (smoothed error PSD in converged
  frames) — this is what P.S1 measured and is the direct ERL proxy.
- Option C: Current `|echo_spec|² / |far_spec|²` but with a tighter update gate
  (e.g., require `filter_state == 'refined_usable'` only, not `coarse_learning`).

**Recommendation**: Try Option B first (switch to `error_psd / far_psd` in
converged FS frames). This is exactly the P.S1 oracle source and should reproduce
the observed 0.043–0.191 range for case 04.

### 5.2 EMA time constant sweep

Start with α = 0.99 (current). Also test α = 0.999 (TC ≈ 1000 hops = 10 s).
Slower tracking is more robust to DT contamination but responds more slowly to
room changes. 800-case bench required to judge net AECMOS.

### 5.3 Update gate tightening

Current gate includes `coarse_learning`. Consider restricting to `refined_usable`
only, since `coarse_learning` frames have partially-converged W that may produce
biased `echo_spec`. The FS_static case showing zero flag-ON diff may be because
`coarse_learning` frames dominate and produce noisy echo_spec values.

### 5.4 Post-EPC cap calibration

The current per-band caps `[0.6, 0.8, 1.0]` are wide conservative bounds from
P.S1. Once P.S3 establishes the true per-band ERL distribution on 800 cases,
calibrate caps to p95 of the converged-frame distribution per band.

### 5.5 800-case bench gate

Per P.S2 design lock: net positive requires `DT Δdeg ≥ 0.000 dB` AND
`FS Δecho ≤ 0.005 dB` at 800-case level. Run after P.S3 source signal fix.

---

## 6. Files

- **aec.py edits**: ~25 lines added (config flags + state init + update logic +
  post-EPC caps + res.process() call + _stage_gain_compute + _stage_residual_model
  + _stage_gain_postprocess + diagnostics)
- **Validation script**: `tools/research/v3_14_p_s2_byte_equal.py`
- **Design doc**: `docs/v3_14_p_s2_design.md` (this file)
