"""temporal_smoother — P52 Phase B Module 4.

Verbatim extraction of `ResFilter._stage_temporal_smoothing` (aec.py:2103-2158)
into a free function operating on a ResFilter-shaped state container `rf`.
Logic copied byte-for-byte per anti-loophole §5.5.
"""

from __future__ import annotations

import numpy as np


def temporal_smoother(rf, *, g_in, dt_indicator, effective_dt,
                      is_stationary_dt, erle_factor, fs_confidence,
                      spectral_g_min, effective_g_min):
    """Stage 4: split attack/release EMA + rate limiting + LF protect + render ceil.

    Mutates: `rf.gain_smooth`, `rf._diag_round5_stages[7]`,
             `rf._stats_last_gain_after_smoothing`. Returns None (legacy parity).
    """
    g = g_in
    dt_temporal = 0.8 if is_stationary_dt else max(dt_indicator, effective_dt * 0.5)

    alpha_fast = 0.3 + 0.2 * (1.0 - erle_factor)
    alpha_slow = 0.85 + 0.1 * (1.0 - erle_factor)
    alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence
    if is_stationary_dt:
        alpha_attack = alpha_attack * np.clip(1.0 - dt_temporal**2, 0.1, 1.0)
    alpha_release_light = 0.5 - 0.2 * dt_temporal
    smoothed = np.where(g < rf.gain_smooth,
                        alpha_attack * rf.gain_smooth + (1 - alpha_attack) * g,
                        alpha_release_light * rf.gain_smooth + (1 - alpha_release_light) * g)

    activity_scale = 0.5 + 0.5 * rf.far_activity
    eff_drop = rf.max_drop_ratio ** activity_scale
    rise_exp = 0.5 + 0.5 * (1.0 - rf.far_activity)
    if dt_temporal > 0.3:
        dt_rise_boost = 1.0 + dt_temporal
        rise_exp = rise_exp / dt_rise_boost
    eff_rise = rf.max_rise_ratio ** rise_exp
    gain_floor = rf.gain_smooth / eff_drop
    gain_ceil = rf.gain_smooth * eff_rise
    if fs_confidence < 0.9:
        lf_limit = min(8, rf.n_freqs)
        lf_factor = 0.25 * max(1.0 - fs_confidence * 2.0, 0.0)
        if lf_factor > 0.01:
            gain_floor[:lf_limit] = np.maximum(
                gain_floor[:lf_limit], rf.gain_smooth[:lf_limit] * lf_factor)
    smoothed = np.maximum(smoothed, gain_floor)
    smoothed = np.minimum(smoothed, gain_ceil)
    if isinstance(spectral_g_min, np.ndarray):
        smoothed = np.maximum(smoothed, spectral_g_min)
    else:
        smoothed = np.maximum(smoothed, effective_g_min)
    smoothed = np.minimum(smoothed, 1.0)
    if rf._residual_est.using_render_based and rf.far_activity > 0.3:
        ceil = getattr(rf, 'render_dt_gain_ceil', 0.6)
        smoothed = np.minimum(smoothed, ceil)
    if getattr(rf, '_capture_stages', False):
        rf._stage_gains['08_post_temporal'] = smoothed.copy()
    rf._diag_round5_stages[7] = float(np.mean(smoothed[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0
    rf.gain_smooth = smoothed
    if rf._stats is not None:
        rf._stats_last_gain_after_smoothing = float(np.mean(rf.gain_smooth))
