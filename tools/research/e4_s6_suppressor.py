#!/usr/bin/env python3
"""E4.S6 — subtractive NLP suppressor (offline OLA for isolated bench).

Per E4.S5 design lock (docs/v3_13_e4_s5_suppressor_design.md):
- Fixed attenuation, harmonic-pinned mask
- Floor g_min_e4_db = -12 dB; upper bound 1.0
- Harmonic pinning: Gaussian at H_k = k*f0, sigma = 50 Hz, up to 6 kHz
- Time smoothing alpha = 0.7; 3-bin freq smoothing
- 5-frame ramp-up on nl_confidence onset

Offline OLA implementation: takes a baseline output WAV + detector
trace (per-detector-hop nl_confidence + pitch_lag), produces
suppressed output WAV via 50%-overlap STFT/iSTFT.

This is the standalone for S6 isolated bench. S7 will port to aec.py
with per-hop streaming.
"""
from __future__ import annotations

import numpy as np


def _harmonic_weights(pitch_lag: int, freqs: np.ndarray, sigma_hz: float,
                      max_harm_hz: float) -> np.ndarray:
    """Per-bin Gaussian sum at harmonics H_k = k * f0 (sr/pitch_lag)."""
    if pitch_lag <= 0:
        return np.zeros_like(freqs, dtype=np.float32)
    sr = freqs[1] * 2 * (len(freqs) - 1)   # not used; assume sample_rate known
    # Note: caller provides sr separately. We just need f0 from pitch_lag.
    return None  # placeholder — see _compute_mask


def _compute_mask(nl_confidence: float, pitch_lag: int,
                  freqs: np.ndarray, sample_rate: int,
                  g_min_lin: float, sigma_hz: float, max_harm_hz: float,
                  ramp_factor: float) -> np.ndarray:
    """Per-bin attenuation mask for one frame."""
    n_freqs = len(freqs)
    if nl_confidence <= 0.0 or pitch_lag <= 0:
        return np.ones(n_freqs, dtype=np.float32)
    f0 = sample_rate / pitch_lag
    weights = np.zeros(n_freqs, dtype=np.float64)
    k = 1
    sigma2_inv = 1.0 / (2.0 * sigma_hz ** 2)
    while True:
        h_k = k * f0
        if h_k > max_harm_hz:
            break
        d = freqs - h_k
        weights += np.exp(-(d ** 2) * sigma2_inv)
        k += 1
    weights = np.clip(weights, 0.0, 1.0)
    eff_conf = nl_confidence * ramp_factor
    atten_db = (np.log10(g_min_lin) * 20.0) * weights * eff_conf
    mask_raw = 10.0 ** (atten_db / 20.0)
    mask_raw = np.clip(mask_raw, g_min_lin, 1.0)
    return mask_raw.astype(np.float32)


def _smooth_freq_3bin(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:-1] = 0.25 * mask[:-2] + 0.5 * mask[1:-1] + 0.25 * mask[2:]
    return out


def apply_suppressor(audio: np.ndarray,
                     detector_trace: list,
                     sample_rate: int = 16000,
                     detector_hop: int = 160,
                     fft_size: int = 512,
                     supp_hop: int = 256,
                     g_min_e4_db: float = -12.0,
                     time_alpha: float = 0.7,
                     sigma_hz: float = 50.0,
                     ramp_frames: int = 5,
                     max_harm_hz: float = 6000.0) -> np.ndarray:
    """Apply harmonic-pinned suppressor to audio.

    audio: 1-D float array
    detector_trace: list of dicts {'nl_confidence': float, 'pitch_lag': int}
                    one per detector hop (10 ms by default)
    sample_rate: 16000
    detector_hop: 160 (10 ms) — detector trace hop in samples
    fft_size: 512 (32 ms) — suppressor FFT size
    supp_hop: 256 (16 ms) — suppressor hop (50% overlap → Hann OLA)

    Returns: suppressed audio (same length as input)
    """
    n = len(audio)
    if n < fft_size:
        return audio.copy()
    audio = audio.astype(np.float64)
    g_min_lin = 10.0 ** (g_min_e4_db / 20.0)
    window = np.hanning(fft_size)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    n_freqs = len(freqs)

    output = np.zeros(n, dtype=np.float64)
    win_sum = np.zeros(n, dtype=np.float64)

    mask_time_prev = np.ones(n_freqs, dtype=np.float64)
    was_zero_prev = True
    ramp_counter = 0

    n_frames = (n - fft_size) // supp_hop + 1
    for i in range(n_frames):
        t_center = i * supp_hop + fft_size // 2
        det_idx = min(t_center // detector_hop, len(detector_trace) - 1)
        det_idx = max(det_idx, 0)
        trace_entry = detector_trace[det_idx] if det_idx < len(detector_trace) else {}
        nl_conf = float(trace_entry.get('nl_confidence', 0.0))
        pitch_lag = int(trace_entry.get('pitch_lag', 0))

        # Ramp-up
        if nl_conf > 0.0 and was_zero_prev:
            ramp_counter = ramp_frames
        if ramp_counter > 0:
            ramp_factor = 1.0 - (ramp_counter / max(1, ramp_frames))
            ramp_counter -= 1
        else:
            ramp_factor = 1.0
        was_zero_prev = (nl_conf <= 0.0)

        mask_raw = _compute_mask(nl_conf, pitch_lag, freqs, sample_rate,
                                  g_min_lin, sigma_hz, max_harm_hz,
                                  ramp_factor)
        mask_time = time_alpha * mask_time_prev + (1.0 - time_alpha) * mask_raw.astype(np.float64)
        mask_smooth = _smooth_freq_3bin(mask_time)
        mask_time_prev = mask_time

        start = i * supp_hop
        seg = audio[start:start + fft_size] * window
        spec = np.fft.rfft(seg, n=fft_size)
        spec_masked = spec * mask_smooth
        seg_out = np.fft.irfft(spec_masked, n=fft_size) * window
        output[start:start + fft_size] += seg_out
        win_sum[start:start + fft_size] += window ** 2

    win_sum = np.maximum(win_sum, 1e-10)
    output = output / win_sum
    # Where win_sum is too small (start/end edges), preserve original audio
    output = np.where(win_sum < 1e-6, audio, output)
    return output.astype(np.float32)
