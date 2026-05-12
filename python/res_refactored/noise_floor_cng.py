"""noise_floor_cng — P52 Phase B Module 5.

Verbatim extraction of `ResFilter._stage_noise_floor_and_cng` (aec.py:2160-2228)
into a free function operating on a ResFilter-shaped state container `rf`.
Logic copied byte-for-byte per anti-loophole §5.5.
"""

from __future__ import annotations

import numpy as np


def noise_floor_cng(rf, *, spec_synth, far_power, effective_dt,
                    dt_indicator, filter_once_converged,
                    effective_g_min, hop):
    """Stage 5: noise_psd tracker + noise-floor lift + CNG + IFFT/OLA.

    Mutates: `rf.noise_psd`, `rf.gain_smooth`, `rf._smooth_cn_gain`,
             `rf.ola_buf`, `rf._diag_round5_stages[8]`,
             `rf._stats_last_*` (when stats enabled).
    Returns: output[hop_size] (float32).
    """
    if not rf._noise_initialized:
        rf.noise_psd = rf.error_psd.copy() + 1e-8
        rf._noise_initialized = True
        rf._smooth_cn_gain = np.zeros(rf.n_freqs, dtype=np.float32)
    is_learning_safe = (rf.far_activity < 0.01) and (dt_indicator < 0.1)
    alpha_down = 0.98
    alpha_up = 0.998 if is_learning_safe else 1.0
    alpha_n = np.where(rf.error_psd > rf.noise_psd, alpha_up, alpha_down)
    rf.noise_psd = alpha_n * rf.noise_psd + (1 - alpha_n) * rf.error_psd

    spec_pwr_synth = np.abs(spec_synth) ** 2 + 1e-10
    noise_floor_gain = np.sqrt(rf.noise_psd / spec_pwr_synth)
    noise_floor_gain = np.clip(noise_floor_gain, effective_g_min, 1.0)

    _startup_dt_nfl = effective_dt > 0.35 and far_power > 1e-4 and not filter_once_converged
    if rf._stats is not None:
        _nfl_mean = float(np.mean(noise_floor_gain))
        rf._stats_last_noise_floor_gain = _nfl_mean
        rf._stats_last_noise_psd = float(np.mean(rf.noise_psd))
        rf._stats_last_spec_pwr = float(np.mean(spec_pwr_synth))
        rf._stats_last_nfl_lifted = _nfl_mean > float(np.mean(rf.gain_smooth))

    if _startup_dt_nfl and rf.startup_dt_noise_floor_scale < 1.0:
        if rf.startup_dt_noise_floor_scale > 0.0:
            rf.gain_smooth = np.maximum(rf.gain_smooth,
                                        noise_floor_gain * rf.startup_dt_noise_floor_scale)
    else:
        rf.gain_smooth = np.maximum(rf.gain_smooth, noise_floor_gain)
    rf._diag_round5_stages[8] = float(np.mean(rf.gain_smooth[rf._voice_band_idx])) if rf._voice_band_idx.size > 0 else 0.0

    enhanced_spec = rf.gain_smooth * spec_synth

    if rf.enable_cng and np.sum(rf.noise_psd) > 1e-7:
        target_cn_gain = np.sqrt(np.maximum(1.0 - rf.gain_smooth ** 2, 0.0)) * 0.4
        rf._smooth_cn_gain = 0.8 * rf._smooth_cn_gain + 0.2 * target_cn_gain
        noise_std = np.sqrt(rf.noise_psd / 2.0).astype(np.float32)
        cng_real = np.random.randn(rf.n_freqs).astype(np.float32) * noise_std
        cng_imag = np.random.randn(rf.n_freqs).astype(np.float32) * noise_std
        cng_spec = (rf._smooth_cn_gain * (cng_real + 1j * cng_imag)).astype(np.complex64)
        enhanced_spec = enhanced_spec + cng_spec

    enhanced_time = np.fft.irfft(enhanced_spec, rf.block_size)[:rf.frame_size]
    enhanced_time *= rf.window

    rf.ola_buf += enhanced_time
    output = rf.ola_buf[:hop].copy()
    rf.ola_buf[:-hop] = rf.ola_buf[hop:]
    rf.ola_buf[-hop:] = 0.0
    return output
