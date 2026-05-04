# F2 — ENR Main-Path Knob Audit (BALANCED preset)

**Branch**: `algo/f1-f4-investigation`  •  **Date**: 2026-05-02  •  **Source**: `python/aec.py`

Goal of this audit: enumerate every preset / config field that the BALANCED preset ENR main path actually reads, mark each as **live / weak / dead**, and surface the tunable knobs that F1 / F4 can route to without inventing new fields.

The ENR main path is `ResFilter._compute_gain` line ≈1706 with `gain_type == "enr" and residual_echo_psd is not None`. Outside that block (`wiener` / `spectral_sub`) is unreachable in production presets — all four presets force `res_gain_type="enr"`.

## ENR-path knob inventory

| Knob | Type | Where read | Status under ENR | Notes |
|------|------|------------|------------------|-------|
| `res_g_min_db` | preset | `aec.py:3241` → `spectral_g_min` | **LIVE** | hard floor on output gain (BALANCED -55 dB) |
| `res_spectral_floor_db` | preset | `aec.py` ResFilter init | **LIVE** | spectral floor base; folds into `spectral_g_min` |
| `res_ne_protect_db` | preset | ResFilter init | **LIVE** (cap) | NE protection floor; raises gain near NE bins |
| `res_enr_scale` | preset | `aec.py:1751` `scale = self.enr_scale` | **LIVE** | drives FS thresholds `enr_t_fs / enr_s_fs` (BALANCED 0.85) |
| `res_alpha_echo_psd` | preset | echo_psd EMA | **LIVE** | smoothing for `residual_echo_psd` (BALANCED 0.4) |
| `res_alpha_error_psd` | preset | error_psd EMA | **LIVE** | smoothing for `error_psd` (BALANCED 0.5) |
| `res_reverb_decay` | preset | reverb tail | **LIVE** | per-frame decay coeff (BALANCED 0.85) |
| `res_reverb_gain` | preset | reverb tail | **LIVE** | reverb-bin gain (BALANCED 1.6) |
| `res_enable_reverb` | preset | reverb branch gate | **LIVE** | True for all 4 presets |
| `res_echo_method` | preset | residual echo estimator | **LIVE** | `"direct"` for all 4 |
| `enable_cng` | preset | CNG branch | **LIVE** | True for BALANCED (now matches CLI `--cng`) |
| `over_sub` (instance attr) | runtime | `aec.py:4051` set by AEC.process | **DEAD for ENR** | only `wiener` / `spectral_sub` branch reads `self.over_sub` (lines 1813,1819,1822) |
| `res_over_sub_base` | preset | feeds `base_over_sub` | **DEAD for ENR** | end-state is `over_sub` which ENR ignores |
| `res_over_sub_scale` | preset | feeds `base_over_sub` (× erle_factor) | **DEAD for ENR** | same fate |
| `res_dt_reduction` | preset | `aec.py:4039` `dt_reduction = res_dt_reduction × dt_indicator` | **DEAD for ENR** | feeds `effective_over_sub` which is `over_sub` (dead) — explicit comment at L4035-4038 |
| `saturation_over_sub_boost` | preset | `aec.py:3925` `base_over_sub += saturation_level × this` | **DEAD for ENR** | same chain (`base_over_sub` → `over_sub`) |
| `res_over_sub` (legacy field) | config init | passed to ResFilter ctor | **DEAD for ENR** | initial value of `over_sub`; replaced per-frame by `effective_over_sub` (also dead) |

## DT-shaping knobs that ARE reachable in ENR (live or live-conditional)

These are the live levers F1 / F4 can tune without inventing new config fields.

| Lever | Where | Trigger | Hardcoded? | Comment |
|-------|-------|---------|------------|---------|
| `dt_per_bin` | `aec.py:1711-1714` | always | derived from `effective_dt`, `coh2` | per-bin DT mask; foundation of dt-shaping |
| `dt_shaped_per_bin` exponent `1.1` | `aec.py:1722` | always | **hardcoded** | DT shape exponent — no preset hook |
| `min_ne_from_dt = error_psd × dt_shaped_per_bin` | `aec.py:1725` | always | — | NE floor that rises with DT |
| `startup_dt_min_ne_scale` | `aec.py:1729-1730` instance attr | `effective_dt > 0.35 AND not filter_once_converged AND scale != 1.0` | gated | scales `min_ne_from_dt` during startup-DT — **direct hook for F1** |
| `startup_dt_gain_floor` | `aec.py:1701` | `_startup_dt_curr AND floor < 1.0` | gated | clamps `spectral_g_min` ceiling during startup-DT — **direct hook for F1** |
| `render_min_ne_factor` | `aec.py:1738` instance attr (default 0.5) | `_residual_est.using_render_based` | live | `min_ne_from_dt × this` during render-mode — **direct hook for F1/F4 cold-start** |
| `dt_enr_relax = 1 + (dt-0.4)/0.6 × 0.5` | `aec.py:1763-1766` | `effective_dt > 0.4` | **hardcoded** | relaxes `enr_t_ne / enr_s_ne` thresholds for high-DT |
| `dt_residual_scale = 1.0 - 0.5×dt/0.8` | `aec.py:4053-4055` | always (DT-driven) | **hardcoded** | scales echo_spec passed to ResFilter — **direct hook for F1** |
| `enr_t_fs / enr_s_fs` constants `0.07 / 0.1` | `aec.py:1759-1760` | always | **hardcoded** | gate width × `enr_scale` |
| `min_gate_width = 0.2` | `aec.py:1776` | always | **hardcoded** | enforces gate softness |
| `emr_transparent = 0.3` | `aec.py:1788` | `noise_psd > 0` | **hardcoded** | echo-masked-by-noise transparency cap |
| `ne_physical_floor = error_psd × 0.05` | `aec.py:1743` | always | **hardcoded** | absolute NE floor |
| `noise_floor_psd = error_psd × 0.01` | `aec.py:1708` | always | **hardcoded** | clip floor for `nearend_est` |

## Findings (action-relevant)

1. **F2 confirmed**: BALANCED preset's `res_dt_reduction=2.5` is dead. So is `res_over_sub_base=5.0`, `res_over_sub_scale=9.0`, `saturation_over_sub_boost=3.0`, `res_over_sub=3.0` (legacy default). All four are kept "for backward compat if `gain_type` ever changes" per the inline note at L4035-4038. **Recommendation**: leave the fields alone (avoid noisy rename diff), but treat them as dead config in F1 / F4 — do not touch them expecting an effect.

2. **Direct hooks F1 / F4 can use** (no new config field needed):
   - `startup_dt_min_ne_scale` (instance attr) — currently 1.0 (no-op) for all presets unless overridden via `**config_overrides`. F1 can set this during InitialDtGuard window to e.g. 0.7 to scale the NE floor down (giving suppressor more freedom on cold-start NE).
   - `render_min_ne_factor` (instance attr, default 0.5) — already live during render-based mode. F1 can lower further (e.g. 0.35) when InitialDtGuard fires AND `using_render_based`.
   - `dt_residual_scale` at `aec.py:4054` — already live; F1 can override the `0.5` slope for the cold-start window (e.g. push to 0.7 — extra echo-spec attenuation when NE-corrupted filter shouldn't be trusted).

3. **Hardcoded knobs that *would* matter but have no preset / config hook today**:
   - DT shape exponent `1.1` (L1722).
   - DT ENR relax slope `0.5` and threshold `0.4` (L1763-1766).
   - NE physical floor 5% (L1743).
   - EMR transparent 0.3 (L1788).
   - These are out of scope for F1–F4 (would need new config fields and per-preset tuning).

4. **F4 placement confirmed**: the inst-ERLE correction at `aec.py:4015-4027` produces `raw_dt /= erle_for_dt` BEFORE `dt_indicator = clip(raw_dt, 0, 0.8)`. Downstream `dt_indicator` flows into:
   - `dt_residual_scale` (L4054) — direct
   - `effective_dt` parameter into ResFilter._compute_gain (via `dt_indicator` kwarg) — drives `dt_per_bin` floor at L1712, `dt_enr_relax` gate at L1763, `_startup_dt_cond` at L1727
   So F4 (loosen cap during cold-start) recovers DT awareness across all four downstream consumers without touching ResFilter internals.

5. **Per-bin DT routing already in place**: `dt_per_bin = max(effective_dt, 1.0 - coh2)` is live (L1711-1714). The review-suggested "dt_per_bin" knob already exists as a live signal. New levers (F1) can ride on the existing DT chain rather than introducing a parallel path.

## Acceptance for Step 1

- ☑ BALANCED preset ENR-path knob inventory complete.
- ☑ Dead-knob list confirms F2 finding (`res_dt_reduction` and 3 siblings).
- ☑ Live levers identified for F1 / F4: `startup_dt_min_ne_scale`, `render_min_ne_factor`, `dt_residual_scale`. No new preset fields required.
- ☑ Hardcoded gaps documented for future tuning sub-branches (out of scope for F1–F4).

**Decision**: F1 + F4 will route through `startup_dt_min_ne_scale`, `render_min_ne_factor`, `dt_residual_scale` and the existing `_far_active_blocks` / `_filter_once_converged` / `_erl_estimate` triggers. No config-schema changes; sub-branches stay surgical.
