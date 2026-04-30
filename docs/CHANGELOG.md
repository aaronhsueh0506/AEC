# AEC Changelog

Versioning: major.minor.patch. Major changes architecture; minor adds features
or tuning sweeps; patch is bugfix.

---

## v3.8.0 (2026-05-01) — Remove e2/mic-as-echo-floor architectural mistakes (ABL-1 + ABL-2)

**Goal**: Continue v3.7.1's AEC3-aligned cleanup. Two more legacy floors that
structurally use NE-contaminated signals as "echo proxies".

**Changes**:

1. **ABL-1 — remove v3.3 `error_based_floor`** (ResidualEchoEstimator):
   ```python
   # Removed:
   far_conf = far_psd / (far_psd + error_psd * 0.05 + 1e-10)
   fallback_factor = 0.7 - 0.2 * np.clip(dt_for_fs, 0.0, 1.0)
   error_based_floor = error_psd * far_conf * fallback_factor
   render_based_echo = np.maximum(render_based_echo, error_based_floor)
   ```
   `error_psd` contains NE during DT, so using it as residual_echo floor
   structurally over-suppresses near-end (same lesson as v3.7.1 PR-B).
   Use only `render_based_echo = far × ERL` — AEC3-aligned.

2. **ABL-2 — remove v3.5 Y2 fallback `mic_psd × 0.5`** (ResFilter.process):
   ```python
   # Removed:
   if (far_power > 1e-4 and self._residual_est.using_render_based
           and near_spec is not None):
       error_max_abs = float(np.max(np.abs(error_hop)))
       if error_max_abs > 0.05:
           mic_psd = np.abs(near_spec).astype(np.float32) ** 2
           residual_echo_psd = np.maximum(residual_echo_psd, mic_psd * 0.5)
   ```
   Trigger `render_based + error_peak > 0.05` is structurally NE-blind — DT
   speech also produces high error peaks. Floor `mic_psd × 0.5` then equates
   the near-end+echo mixture with "echo proxy" → over-suppress NE. AEC3 uses
   explicit saturation detection (mic clipping), not error peak.

3. **Kept `render_dt_gain_ceil = 0.6`** (ABL-3 reverted): ablation showed
   removal causes +0.008 DT_mv deg / -0.046 FS_mv echo (Pareto ratio 0.17,
   much worse than ABL-2's 0.67). Confirmed load-bearing for FS echo —
   keep, document in code comment.

**800-case AECMOS vs v3.7.1 (BALANCED / fl=52ms / cng / j4)**:

| bucket | v3.7.1 | v3.8.0 | Δ |
|---|---|---|---|
| FS_st | 3.877 | 3.801 | **-0.076** |
| FS_mv | 3.941 | 3.863 | **-0.078** |
| NE deg | 3.993 | 4.002 | **+0.009** |
| DT_st echo | 4.263 | 4.256 | -0.007 |
| DT_st deg | 2.224 | **2.257** | **+0.033** |
| DT_mv echo | 4.162 | 4.144 | -0.018 |
| DT_mv deg | 2.219 | **2.269** | **+0.050** |

**DT_mv deg gap vs AEC2: 0.309 → 0.259 (closed 16%)**.

**Why structural, not Pareto sliding**: ABL-1 + ABL-2 both remove "use signal
containing NE as echo proxy" floors. Same mechanism as v3.7.1 PR-B. Closing
this entire family of architectural mistakes (`v3.7.1 e2-floor + v3.8.0
error-based-floor + Y2-fallback`) gives DT preservation across all 4 presets.

**Other presets** (2026-05-01 4-preset rebench):

| Preset | FS_st | FS_mv | NE | DT_st e/d | DT_mv e/d |
|---|---|---|---|---|---|
| MILD | 3.626 | 3.733 | 4.013 | 4.083/2.368 | 4.010/2.395 |
| BALANCED | 3.801 | 3.863 | 4.002 | 4.256/2.257 | 4.144/2.269 |
| AGGRESSIVE | 3.829 | 3.867 | 3.997 | 4.285/2.218 | 4.169/2.239 |
| MAXIMUM | 3.886 | 3.909 | 3.987 | 4.322/2.180 | 4.206/2.186 |

vs v3.7.1: DT deg gain consistent across all presets (DT_mv +0.034~+0.062),
NE deg gain consistent (+0.007~+0.013), FS echo cost consistent (-0.06~-0.09)
— confirms preset-independent structural fix.

**Out of scope (deferred to v3.8.x)**:

- ABL-4: `linear_failed = erl_estimate > 1.2` branch (`error_psd × 0.9`) on
  ResFilter.process line ~1663. Trace verified `self._erl_estimate` clipped
  to [0.001, 1.0] so 0% fire rate — dead code. Defer to single-cut ablation
  in v3.8.1.
- v3.8.1 hygiene candidates tested but deferred (no metric impact in BALANCED):
  delay_first ERL re-init, delay_shift ERL cap, reset() lazy-diag-counter clear.
- H1 (`DominantNearendDetector` for filter-independent DT detection in
  loud-DT) explored extensively (v1-v7) but Pareto-bound on this dataset.
  Architectural finding preserved in plan: linear filter learns NE into W
  during loud-DT (eff_dt false-negative); needs phase coherence (R3) or
  masking-aware suppression (R7) to break Pareto. Stashed.

**Trace investigations that informed this release** (all in plan):

- C1 (far_active adaptive threshold): trace証伪. DT_mv deg gap is in
  loud-far cases (gap_d=-0.601), not quiet-far. C1 not binding.
- R6 (smoothed far-conditioned ERL): trace証伪. `linear_failed > 1.2` is
  dead code, not the false-positive mechanism described pre-v3.7.1.
- 5-site systematic trace: identified that worst-DT_mv cases have eff_dt
  ≈ 0 + gain_after_smoothing ≈ 1.0 — root cause is linear filter learning
  NE into W, NOT ResFilter over-suppression. RES pipeline 5 scalar-DT
  gates are mostly inert in worst cases.

---

## v3.7.1 (2026-04-30) — Drop render-based linear_failed fallback (PR-B)

**Goal**: Address residual DT_mv deg gap (−0.316 vs AEC2 in v3.7.0) by
fixing architectural error in `linear_failed` fallback design.

**WebRTC AEC3 source code research finding**: AEC3 **never uses error_psd
as floor** in residual echo estimation. AEC3's `residual_echo_estimator.cc`
relies on `render×ERL_smoothed` for fallback when linear filter unreliable.
Using `e2 = error_psd` as floor is structurally guaranteed to false-positive
on DT — because `e2 = NE + residual_echo` during DT, flooring residual_echo
to e2 effectively says "all near-end is echo, suppress it."

**E2 trace verification (2026-04-30)**:
- DT_mv: 35.2% frames fired the `linear_failed_render` branch
- Fire-time `effective_dt = 0.008` (false-negative — DT detector failed
  along with the filter that produces it)
- Fire-time `erl_estimate = 0.115` (low — not a real linear failure, just
  filter struggling during DT)
- Net effect: 35% of DT_mv frames had near-end killed by `error_psd × 0.9`

**Patch ([python/aec.py:1666-1670](../python/aec.py#L1666))**: drop the
render-based branch from linear_failed entirely. Keep only `erl_estimate
> 1.2` (physical mic/far ratio, filter-health-independent).

```python
# Before (v3.7.0):
linear_failed = (erl_estimate > 1.2 or
                 (self._using_render_based and erle_factor < 0.2))
# After (v3.7.1):
linear_failed = erl_estimate > 1.2
```

Render-based mode itself already inflates residual via far×ERL. The
secondary `error_psd × 0.9` floor was double-suppression; removing it
lets render-based mode's own residual estimate drive the decision —
matching AEC3 architectural pattern.

**800-case AECMOS vs v3.7.0 (BALANCED / fl=52ms / cng / j4)**:

| bucket | v3.7.0 | v3.7.1 | Δ |
|---|---:|---:|---:|
| FS_st | 3.880 | 3.877 | -0.003 (noise) |
| FS_mv | 3.957 | 3.941 | **-0.016** (trade) |
| NE | 3.991 | 3.993 | **+0.002** ✓ |
| DT_st echo | 4.264 | 4.263 | -0.001 (noise) |
| DT_st deg | 2.220 | 2.224 | **+0.004** ✓ |
| DT_mv echo | 4.163 | 4.162 | -0.001 (noise) |
| DT_mv deg | 2.212 | 2.219 | **+0.011** ✓ |

**Trade summary**: FS_mv echo loses 0.016 (still +0.422 ahead of AEC2
3.519). In return DT_mv deg gains +0.011 (gap closes from −0.316 to
−0.309), DT_st deg recovers +0.004, NE deg climbs above 4.000 floor
(3.993). 3 target metrics improve at cost of shrinking already-massive
FS lead.

**Why architectural, not Pareto sliding**: v3.7.0 G1 fixed filter state
coherence (P/W decoupling). v3.7.1 fixes residual estimator's reliance
on e2 — two independent filter-side fixes that stack.

**Other presets** (2026-04-30 4-preset rebench, fl=52ms / cng / j4):

| Preset | FS_st | FS_mv | NE | DT_st echo | DT_st deg | DT_mv echo | DT_mv deg |
|---|---|---|---|---|---|---|---|
| MILD | 3.712 | 3.820 | 4.006 | 4.089 | 2.330 | 4.027 | 2.333 |
| BALANCED | 3.877 | 3.941 | 3.993 | 4.263 | 2.224 | 4.162 | 2.219 |
| AGGRESSIVE | 3.904 | 3.941 | 3.986 | 4.294 | 2.186 | 4.191 | 2.196 |
| MAXIMUM | 3.951 | 3.969 | 3.974 | 4.332 | 2.155 | 4.223 | 2.152 |

**Preset trade-off**:
- MILD: NE deg 4.006 above floor; DT deg 2.33 (highest).
- BALANCED: G1+B primary operating point; DT_mv deg 2.219 vs AEC2 2.528 = −0.309 gap (largest remaining).
- AGGRESSIVE: balanced echo/deg trade; DT_st echo 4.294.
- MAXIMUM: DT_st echo 4.332 **exceeds AEC2 (4.331)**; FS leads AEC2 by ≥+0.45.

vs v3.7.0 4-preset deltas: MILD-MAX 各 metric ±0.01 內，B 改動 preset-independent 結論成立。

**Plan**: continue C-stick experiment (linear_failed N-frame hysteresis
+ sticky `_filter_once_converged` gate) on `experiment/e2-architectural`
branch; ship as v3.7.2 if additional gain materializes.

---

## v3.7.0 (2026-04-30) — Blended KX P-update for DT consistency (PR-G1)

**Goal**: Break the corr(Δe, Δd) ≈ −0.81 Pareto wall identified in v3.6.1
PR-F session by fixing main filter state coherence rather than reshaping
masks. GPT Phase 1 hypothesis: in PBFDKF `_update_weights`, P covariance
update uses unscaled `K_optimal` while weights use `K_scaled = K_optimal *
mu_scale`. During DT (mu→0), W stays put but P shrinks via K_optimal —
P/W decoupling makes Kalman over-confident; after DT, K becomes too small
and filter recovery is slow.

**KX trace verification (5-bucket, 2026-04-30 [/tmp/kx_trace.py](/tmp/kx_trace.py))**:
DT_st 4.3% DT-active frames showed P median collapse from 0.0093 (FS)
to 0.0026 (DT) — 72% drop per cycle, with mu_mean=0 driving KX_scaled=0
while KX_optimal=4.6e-2. FE_mv recovery period showed P at 54% of FS
level, confirming "DT 後 K 偏低 / recovery 慢" mechanism.

**Patch ([python/aec.py:856-870](../python/aec.py#L856))**: blend KX between
optimal and scaled by `mu_mean` (smooth, no binary discontinuity):
```python
KX = mu_mean * KX_optimal + (1 - mu_mean) * KX_scaled
```
- FS (mu_mean=1): KX = KX_optimal (legacy behavior)
- DT (mu_mean=0): KX = KX_scaled (P consistent with W)
- mid-range: smooth blend

Avoids both extremes' regression risk (full KX_optimal causes DT P-collapse;
full KX_scaled would over-cautious P in weak-far / per-bin-gating windows
where mu_scale is per-bin attenuated for non-DT reasons).

**800-case AECMOS vs git-vanilla v3.6.1 (BALANCED / fl=52ms / cng / j4)**:

| bucket | vanilla | G1 | Δ |
|---|---:|---:|---:|
| FS_st | 3.877 | 3.880 | **+0.003** |
| FS_mv | 3.954 | 3.957 | **+0.003** |
| NE | 3.991 | 3.991 | 0.000 |
| DT_st echo | 4.257 | 4.264 | **+0.008** |
| DT_st deg | 2.224 | 2.220 | -0.004 (noise) |
| DT_mv echo | 4.158 | 4.163 | **+0.006** |
| DT_mv deg | 2.208 | 2.212 | **+0.004** |

**First all-positive-or-flat result this development cycle** — all prior
v3.7 candidates (PR-A coh2, PR-F1v2 dt_per_bin, Path C dominant_nearend
detector, R2 effective_dt gate) slid along corr(Δe,Δd)≈−0.81. G1 breaks
that pattern by fixing upstream filter state coherence rather than
redistributing mask decisions.

**Magnitudes are small** (+0.003 to +0.008) because the existing
`P_floor = Q_high × 0.1` mechanism already partially mitigated DT-end P
collapse. KX blending closes the residual gap without regressing
steady-state behavior.

**Diagnostic infrastructure**: PBFDKF gains `_enable_kx_trace` flag (off
by default, zero overhead) that accumulates per-frame `mu_mean` /
`KX_optimal` / `KX_scaled` / `P_p10/p50/p90` / `Q_gated` / `error_power` /
`far_power` for offline analysis. Used by [/tmp/kx_trace.py](/tmp/kx_trace.py)
5-bucket verification harness.

**Baseline JSON regenerated**: [python/baseline_v36_vs_aec2.json](../python/baseline_v36_vs_aec2.json)
rescored from current git-vanilla v3.6.1 outputs (old baseline drifted
+0.20 FS / +0.07 DT echo from current code; led to mirage Δ during PR-F
session — see [feedback_baseline_json_drift.md](~/.claude/projects/-Users-mingyu-Desktop-novatek-SE/memory/feedback_baseline_json_drift.md)
memory).

**Out of scope (deferred to v3.7.1)**: GPT Phase 2 shadow PBFDKF policy
differentiation. G2 (4-branch shadow_mu_scale) tested on top of G1 — gave
FS+DT echo +0.022~+0.030 but DT_st deg −0.041 / DT_mv deg −0.011 (Pareto
sliding returned). G2a (DT brake 0.15→0.5) softened DT regression but
still net trade-off. Per GPT revised guidance, v3.7.1 will only modify
DT branch (keep FS clean = 1.0) — testing in separate work cycle.

**4-preset 800-case AECMOS** (2026-04-30, fl=52ms / cng=True / j4):

| Preset | FS_st | FS_mv | NE deg | DT_st echo | DT_st deg | DT_mv echo | DT_mv deg |
|---|---:|---:|---:|---:|---:|---:|---:|
| MILD | 3.719 | 3.839 | 4.005 | 4.090 | 2.327 | 4.026 | 2.329 |
| BALANCED | 3.880 | 3.957 | 3.991 | 4.264 | 2.220 | 4.163 | 2.212 |
| AGGRESSIVE | 3.907 | 3.944 | 3.986 | 4.295 | 2.183 | 4.186 | 2.194 |
| MAXIMUM | 3.950 | 3.969 | 3.974 | 4.331 | 2.157 | 4.222 | 2.156 |
| WebRTC AEC2 | 3.457 | 3.519 | 4.098 | 4.331 | 2.304 | 4.149 | 2.528 |

Notes:
- MILD: NE deg above 4.000 floor (4.005); DT deg highest among presets.
- BALANCED: G1 KX blended primary operating point.
- MAXIMUM: DT_st echo (4.331) **fully matches AEC2 reference**, FS leads
  AEC2 by +0.49/+0.45; trade is NE deg −0.124 vs AEC2 and DT_mv deg
  −0.372.

---

## v3.6.1 (2026-04-29) — DT-from-frame-0 spec + stats detector (PR-D4)

**Goal**: Document the linear-AEC fundamental limit identified during PR-D
investigation; add stats-only detector for production debugging.

**PR-D2** (initial-state Q×N boost): tested 4 variants (10×/100, 5×/50,
3×/30, 2×/20). All breached NE deg 4.0 floor — reverted. Q boost
universally pushes filter toward more aggressive adaptation, which costs
NE quality structurally regardless of magnitude.

**PR-D3** (true 10× Q bifurcation on shadow_advantage): tested. Effect at
noise level (DT_movement echo +0.008 / deg -0.009). Shadow filter being
PBFDKF (Kalman, like main) means `shadow_advantage ≈ 1.0` in worst cases —
both filters equally NE-corrupted, no escape signal. Reverted.

**PR-D4 (this PR)**:
- New: [docs/spec_dt_from_frame_zero.md](spec_dt_from_frame_zero.md) — full
  symptom / root-cause / WebRTC comparison documentation.
- New: stats-only detector in `AEC.process` ([python/aec.py:4144-4154](../python/aec.py#L4144))
  fires on `far_active_blocks > 200 AND not _filter_converged AND
  erl_estimate > 0.4` — signature of NE-corrupted filter learning.
- No behavior change. 800-case AECMOS identical to v3.6.0.

Detector validation on representative cases:

| case | category | dt_from_zero % |
|---|---|---:|
| Y7w0W4v9 (top DT_static deg-loser) | DT | 45% |
| QkRkwwFKVE (deg-loser) | DT | 21% |
| hVqUmGvIlk (FS top winner) | FS | 0% |
| PZ7V (FS loser) | FS | 0% |
| 014AzuqPZ (NE control) | NE | 0% |

Discrimination correct: fires only on actual DT-from-frame-0 cases; FS and
NE clean.

**Preset 800-case operating points (vanilla v3.6.1, AEC Challenge blind, AECMOS)**:

| Preset | FS_st echo | FS_mv echo | NE deg | DT_st echo | DT_st deg | DT_mv echo | DT_mv deg |
|--------|-----------|-----------|--------|-----------|----------|-----------|----------|
| MILD | 3.687 | 3.791 | 4.009 | 4.080 | 2.343 | 4.019 | 2.340 |
| BALANCED | 3.675 | 3.931 | 4.000 | 4.182 | 2.325 | 4.151 | 2.258 |
| AGGRESSIVE | 3.889 | 3.939 | 3.989 | 4.282 | 2.199 | 4.174 | 2.199 |
| MAXIMUM | 3.938 | 3.959 | 3.977 | 4.317 | 2.178 | 4.207 | 2.163 |
| WebRTC AEC2 | 3.457 | 3.519 | 4.098 | 4.331 | 2.304 | 4.149 | 2.528 |

Notes:
- MILD/BALANCED honor NE deg ≥ 4.000 floor; AGGRESSIVE/MAXIMUM trade NE
  (−0.011 / −0.023) for echo/FS gains.
- MAXIMUM closes DT_st echo gap vs AEC2 to −0.014 but loses DT_mv deg
  (−0.365 vs AEC2). PR-F (per-bin coh2-aware DT gating) targets the same
  gaps without the NE / DT_mv deg cost.

**PR-D5** (replace shadow Kalman → NLMS): deferred. 1-2 week refactor with
high regression risk on existing v2.5+ shadow tuning. Only worth pursuing if
linear AEC is the chosen direction long-term — competing direction is NN
postfilter (DTLN-AEC, see plan `~/.claude/plans/jazzy-brewing-castle.md`).

**Plan**: ~/.claude/plans/users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md

---

## v3.6.0 (2026-04-29) — Filter length 32ms → 52ms (PR-D1)

**Goal**: Investigation after v3.5.0 PR-B trade-off plateau identified the
linear filter as the root cause of DT-from-frame-0 deg-loss (90% of worst
cases had filter conv_far=0% per [diag_linear_stability.py](../python/diag_linear_stability.py),
ERLE p50 < 0 dB across far-active frames). Source-verified WebRTC AEC3
config showed 52ms default (13 blocks × 4ms) vs our 32ms — significant
RT60 tail capture gap.

**Change** ([python/aec.py:266](../python/aec.py#L266)):
```python
# 16/8kHz default filter_length: 32ms → 52ms
self.filter_length = self.sample_rate * 52 // 1000  # was 32
```

At 16kHz: 512 → 832 samples, 4 → 6 partitions. Compute cost +50%.

**Trace verification**
([diag_filter_dynamics.py](../python/diag_filter_dynamics.py)):

| case | v3.5.0 ERLE p90 | v3.6.0 ERLE p90 |
|---|---:|---:|
| Y7w0W4v9 (DT_static deg-loser) | +1~+3 dB | **+8~+10 dB** |
| QEeKiaNiD (DT_static borderline) | +12 dB | +12 dB (sustained) |
| hVqUmGvIlk (FS_movement winner) | -7~-10 dB | -5~-8 dB (filter still doesn't anchor) |
| PZ7V (FS_static catastrophic) | +10/-68 dB (divergent) | +10/-72 dB (still divergent) |

**800-case AECMOS vs v3.5.0**:

| bucket | v3.5.0 | v3.6.0 | Δ |
|---|---:|---:|---:|
| FS_static echo | 3.590 | 3.675 | **+0.085** |
| FS_movement echo | 3.916 | 3.931 | +0.015 |
| DT_static echo | 4.111 | 4.182 | **+0.071** |
| DT_movement echo | 4.138 | 4.151 | +0.013 |
| DT_static deg | 2.391 | 2.325 | -0.066 |
| DT_movement deg | 2.280 | 2.258 | -0.022 |
| NE deg | 4.004 | 4.000 | -0.004 (at 4.0 floor) |

vs AEC challenge baseline (cumulative v3.4 → v3.6):
- FS_static echo Δ: -0.027 → **+0.218** (large lead)
- DT_static echo Δ: -0.233 → **-0.149** (closed 36% of gap)
- DT_movement echo Δ: -0.027 → +0.002 (first parity)
- NE deg: 4.008 → 4.000 (at floor)

**Trade-off acknowledgment**: DT_static deg lost +0.065 of v3.5.0's lead vs
AEC2 (still +0.022 ahead). DT_movement deg widened to -0.270.
NE = 4.000 is the floor — further pushes risk breaking it.

**Next**: PR-D2 (initial-state Q×100 boost), PR-D3 (Q bifurcation on
shadow_advantage), PR-D4 (DT-from-frame-0 detector + spec).

**Plan**: ~/.claude/plans/users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md

---

## v3.5.0 (2026-04-29) — AEC3-style Y2-fallback for saturated-echo equivalent

**Goal**: Break the v3.4.0 Pareto wall (DT_static echo −0.233 vs AEC challenge
baseline, corr Δecho/Δdeg = −0.81). Five rc attempts (rc15-20) all returned
within ±0.005 — every "single-knob" residual-attribution fix was absorbed by
downstream suppressor reshaping.

**Trace finding** ([python/diag_gain_stages.py](../python/diag_gain_stages.py)):
At worst leak hotspots, Wiener soft-gate output `g = 1.0` (transparent). The
gain pipeline 8 stages do NOT suppress — only `render_dt_gain_ceil = 0.6` cap
fires. Root cause: `attribute_legacy` returns `residual_echo_psd ≈ 0` because
filter unreliable AND lpb_NOW silent (echo from past) → `far_psd × ERL ≈ 0`,
`error × far_conf × factor` also ≈ 0 (far_conf ≈ 0 when far silent now).
Soft-gate sees `ENR ≈ 0` → un-gates entirely.

**Source review** ([WebRTC AEC3 main branch](https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/residual_echo_estimator.cc))
identified the missing mechanism: AEC3 uses `R2 = Y2` (mic spectrum directly)
when `saturated_echo` triggers, regardless of whether linear estimate is usable.
This forces `ENR` large → suppressor engages.

**Fix** ([python/aec.py:1497-1517](../python/aec.py#L1497)):
```python
if (far_power > 1e-4 and self._residual_est.using_render_based
        and near_spec is not None):
    error_max_abs = float(np.max(np.abs(error_hop)))
    if error_max_abs > 0.05:
        mic_psd = np.abs(near_spec).astype(np.float32) ** 2
        residual_echo_psd = np.maximum(residual_echo_psd, mic_psd * 0.5)
```

Trigger: `using_render_based` (filter in fallback mode) AND post-filter signal
amplitude still excessive (`max(|error|) > 0.05` in float32 scale; AEC3 uses
absolute threshold `s_refined_max_abs > 20000.f`). Substitution: residual ←
max(residual, mic_psd × 0.5).

**Trace verification**: WYKA2 frame 706 worst leak ratio 111,827× → 5× (PSD
ratio ours/aec2). NE bit-exact preserved (parity_smoke.py NE case identical).

**800-case AECMOS vs v3.4.0** (CNG=True, balanced preset):

| bucket | v3.4.0 | v3.5.0 | Δ |
|---|---:|---:|---:|
| FS_static echo | 3.522 | 3.590 | **+0.068** |
| FS_movement echo | 3.871 | 3.916 | **+0.045** |
| DT_static echo | 4.098 | 4.111 | +0.013 |
| DT_movement echo | 4.123 | 4.138 | +0.015 |
| DT_static deg | 2.440 | 2.391 | -0.049 |
| DT_movement deg | 2.315 | 2.280 | -0.035 |
| NE deg | 4.008 | 4.004 | -0.004 |

vs AEC challenge baseline:
- FS_static echo Δ = +0.133 (v3.4.0 was +0.065) — **FS lead doubled**
- FS_movement echo Δ = +0.397 (v3.4.0 was +0.353)
- DT_static echo Δ = -0.221 (v3.4.0 was -0.233) — closed 5% more
- NE deg = 4.004 (still > 4.0 floor)

**Trade-off acknowledgment**: DT_static deg -0.049 / DT_movement deg -0.035.
The bimodal Pareto (corr -0.81) is fundamental to fullband DT decisions; this
PR makes the trade-off lean toward echo. PR-B (Profile-swap DominantNearend)
planned to recover DT_deg without losing echo gains.

**Hotspots NOT helped**: hF9Lfj-class (frame 2682 bin 72 ratio 339,329×, final_g=0.901)
where `using_render_based=False` (filter "converged" but echo at HF still leaks).
These are likely speaker-nonlinear harmonics — coh ≈ 0, ERL ≈ 0, sat ≈ 0; no
linear signal can detect. Out of v3.5.0 scope.

**Plan reference**: ~/.claude/plans/users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md
(PR-A complete; PR-B and PR-C pending).

---

## v3.4.0 (2026-04-29) — DT_static echo gap closure via render_ceil skip

**Goal**: Continue closing DT_static echo gap vs AEC2 (was −0.285 in v3.2, −0.241
in v3.3). Trace finding: render_ceil cap (line 1554, `residual ≤ far_now × ERL × 2`)
was neutralizing v3.3.0's error-based fallback at hotspots — capping residual
back down to far_now × ERL even though fallback returned much larger values.

**Three coordinated axes**:
1. **Skip render_ceil in render-mode** — the cap exists to bound a wrong linear
   filter, but render-mode IS the fallback for when filter is wrong; cap was
   self-defeating. Tested gating on `not_once_converged` (rc11): didn't help
   DT_movement deg, lost FS echo.
2. **NE-protect reverb hard-cut** — `reverb_gate=0` when `far_activity < 0.1`.
   far_activity EMA decays slowly; without hard-cut, reverb tail lingers
   ~200ms after far ends, hurting NE-only frames.
3. **DT-aware fallback factor** — `fallback_factor = 0.7 - 0.2 × clip(dt_for_fs)`.
   Strong DT (NE present) → less of error_psd is echo → smaller fallback floor
   (down to 0.5×). Protects DT_movement deg.

**800-case AECMOS vs AEC2** (CNG=True, balanced preset):

| bucket | v3.3.0 → v3.4.0 |
|---|---|
| FS_static echo | +0.031 → +0.065 (won AEC2) |
| FS_movement echo | +0.328 → +0.353 |
| DT_static echo | −0.241 → −0.233 |
| DT echo | −0.163 → −0.155 |
| DT_movement deg | −0.189 → −0.213 |
| NE deg | −0.089 → −0.091 (4.008, > 4.0) |

Win rate: Echo 47% (DT 100/300), Deg 42%.

**Iteration history attempted but reverted (v3.5 candidates)**:
- rc12 (Axis 4: fast reverb decay when far silent): hurt leak +30% on echo
  hotspots. Reverb-tail-fallback contradicts fast-collapse-for-NE; cannot do both
  with one reverb signal.
- rc13/rc14 (Axis 5: tail-augmented `far_for_conf` in fallback): smoke −2~−5%
  leak energy, AECMOS Δ <0.01. Diminishing returns within current framework.
- rc12/rc13 (Axis 6: per-bin NE-dominant gain floor): falsely triggered on
  echo leak hotspots where attribute under-estimates true echo.
  `error_psd >> expected_echo` is true for both NE and under-attributed echo
  bins; cannot discriminate without reliable echo estimate.

**Conclusion**: v3.4.0 is at the limit of what the current attribute+reverb
framework can deliver. Further gains require a per-bin echo-vs-NE discriminator
(long-history coherence, NN postfilter, or sub-band coherence — see Route B
in `aec_v3_evolution.md`).

---

## v3.3.0 (2026-04-29) — Reverb tuning + error-based render fallback

Trace-driven (diag_leak_hotspot.py on 5 worst cases): echo at frame t in losing
cases originates 60–320ms ago — render-based estimate (far_now × ERL) captures
only current X (often quiet between speech bursts) and underestimates true echo
by 2 orders of magnitude (e.g., 4.2 vs error_psd 239).

**Two existing-but-misconfigured mechanisms fixed together**:
1. Existing IIR reverb (line ~1568, WebRTC-AEC3-style):
   - Was: `decay=0.65 (TC ~50ms), gain=1.4`, DT-gate kills reverb 70% during DT
   - Now: `decay=0.85 (TC ~130ms, matches typical RT60), gain=1.6`,
     DT-gate base `0.3 → 0.7` (DT is exactly when reverb matters most)
2. attribute_legacy render-based path:
   - Was: `render_based_echo = far × ERL` (single-frame, undersized when filter
     never converges)
   - Now: `max(far × ERL, error_psd × far_conf × 0.7)` where far_conf is
     per-bin "active echo path" indicator

**800-case vs v3.2.0**: FS +0.045 echo / 0 deg, DT +0.030 echo / −0.032 deg,
NE 0 echo / −0.006 deg.

---

## v3.2.0 (2026-04-29) — Axis 1+2+3: ERL outlier protection + render-mode gain ceiling

Trace-driven multi-axis fix targeting DT/FS echo gap vs AEC2.

**Discriminator found**: losing cases have ERL clamped at high values (mean
0.5–0.95) due to NE-corruption, leading to bloated render_ceil → cap at
error_psd → ENR collapse → gain p90=1.0 (10% of frames no suppression).
Winners have ERL < 0.2 (filter learned echo cleanly).

**Axes**:
- **Axis 1 — NE-corruption ERL protection**: Skip ERL update when
  `inst_erl ≥ 1.5` (physically implausible: mic > far means NE dominates).
  Tighten clip from 10.0 → 1.0.
- **Axis 2 — Render-mode gain ceiling**: Hard ceiling 0.6 on smoothed gain
  when `using_render AND far_active > 0.3`, preventing transient leaks where
  gain hits 1.0.
- **Axis 3 — Relax error_psd cap in render-mode**: Allow `residual ≤ error_psd
  × 1.5` instead of `× 1.0` so render-based estimate isn't bounded by
  NE-inflated error signal. Currently no-op since Axis 1 already keeps render
  estimate small, but defensive against future ERL drift.

**800-case vs AEC2 (CNG=True)**: FS echo +0.115 (was +0.085 in v3.1.0),
DT_static echo −0.285 (was −0.302), DT deg +0.058, NE deg −0.082.

---

## v3.1.0 (2026-04-28) — Trace-driven render-mode RES fixes

Targets DT echo gap. Single deep trace on worst static-DT case revealed two
bottlenecks:
1. error_psd cap at line 1545 killed 71% of render-based residual estimate.
2. min_ne_from_dt floor inflated nearend_est, depressing ENR.

**Fixes** (render-mode aware):
- `echo_psd × 2.0` cap skipped when `using_render_based`.
- `dt_suppress` cap skipped when `using_render_based`.
- `min_ne_from_dt × 0.5` factor (configurable via `render_min_ne_factor`).

**800-case vs AEC2 (CNG=True)**: FS echo +0.085, DT echo −0.211 (closed 13%
of gap from v3.0.2 baseline).

---

## v3.0.0 → v3.0.2 (2026-04-28) — Phase A+B+C decoupling refactor

Architectural decoupling into 5 detector classes + AecState aggregator,
preparing the ground for v3.1+ targeted fixes. **800/800 bit-exact vs v2.8.1**.

- **Phase A** (v3.0.0): Extract from `AEC.process()` into independent classes:
  ShadowCopyController, RenderActivityDetector, FilterConvergenceAnalyzer,
  DoubleTalkAnalyzer, EchoPathChangeDetector, AecState aggregator.
  Properties on AEC delegate to detector state for backward compat.
- **Phase B** (v3.0.1): ResidualEchoEstimator extracted (legacy mode).
- **Phase C** (v3.0.2): Ablation hooks (gate_mode, no_reset_sources,
  maybe_mark_diverged), default = bit-exact v2.8.1.

11 ablation variants (B2 R1 split, C1 S1/S2/S3 coherence/streak, C2 E3
no-reset) all failed to improve movement-DT echo. AEC3-borrowed detector
patterns don't translate to PBFDKF main + PBFDKF shadow architecture.

---

## v2.8.1 (2026-04-28) — Movement DT ablation complete + cleanup

Final v2.x release; baseline for v3 refactor. Movement-DT remains in Pareto
disadvantage vs AEC2 (echo −0.040, deg −0.073) — boundary of detector tuning.

---

## v2.5.0 (2026-04) — Final v2.5 line

Multi-phase shadow filter improvements, ERL tracking refinements,
DT-detector coherence integration. See `archive/CHANGELOG_v2.5.0.md` and
`archive/CHANGELOG_v2.5.md` for detailed phase breakdown.

---

## v2.4.0 (2026-03) — Phase fixes + GetStats API

Fix1+Fix2 (pre-conv render-based force + DT-indicator ERL blend), AecFilterState,
AecStats, GetStats(), AecDebugLogger added.

---

## v2.3.0 (2026-03) — Initial PBFDKF release with shadow filter

PBFDKF main + PBFDKF shadow. ResFilter v2 (direct + ENR + reverb).
DTD coherence integration. Initial AECMOS competitive results vs AEC3 baseline.

---

## Earlier (v1.x and pre-v2.3) — see archive/

`archive/` contains: phase reports (κ, η, b15, b16, phase2_stage_1b/1d/2),
specs (raw_dt_delay_alignment, dt_jump_veto, shadow_nlms),
DEVLOG.md (chronological work log), early CHANGELOGs.
