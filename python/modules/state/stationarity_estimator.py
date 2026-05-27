"""Stationarity estimator for the render path.

Float-scale port of `docs/aec3_extracts/src/aec3/stationarity_estimator.{cc,h}`
and `echo_audibility.{cc,h}` (only the bits we need for residual scaling).

When the render signal is stationary (e.g. a constant background hum on the
far-end), the residual echo estimator should report ~0 echo contribution for
those bins so the suppression gain doesn't spuriously damp nearend speech.

NE-outlier root cause (v3.21 Phase C debug): cases like
`E0l0WVPQjEi6AmtbvfSYLA_nearend_singletalk` and
`wJVPo4lexUK40x0nuK0KWg_nearend_singletalk` have a constant ~0.009 RMS hum
on the lpb track; without stationarity awareness, the AEC3-port residual
chain treats nonzero render PSD as active echo and pushes the suppression
gain down on every bin → cuts 9-18% of nearend RMS.

This module wires into ``_aec3_post`` and zeros out R²/R²_unbounded on bands
flagged as stationary (after filter convergence) — mirrors AEC3
`residual_echo_estimator.cc:303-313`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..aec3_scale import (
    STATIONARITY_ALPHA,
    STATIONARITY_ALPHA_INIT,
    STATIONARITY_AVG_INIT_HOPS_DEFAULT,
    STATIONARITY_BLOCK_FRACTION,
    STATIONARITY_HANGOVER_HOPS_DEFAULT,
    STATIONARITY_INITIAL_PHASE_HOPS_DEFAULT,
    STATIONARITY_MIN_NOISE_POWER_FLOAT,
    STATIONARITY_THR_RATIO,
    STATIONARITY_WINDOW_HOPS_DEFAULT,
)


class _NoiseSpectrum:
    """Per-bin slow noise-floor tracker (stationarity_estimator.cc:159-242)."""

    def __init__(self, n_freqs: int, min_noise_power: float,
                 avg_init_hops: int, initial_phase_hops: int,
                 alpha: float, alpha_init: float) -> None:
        self.n_freqs = int(n_freqs)
        self._min = float(min_noise_power)
        self._avg_init = int(max(1, avg_init_hops))
        self._init_phase = int(initial_phase_hops)
        self._alpha = float(alpha)
        self._alpha_init = float(alpha_init)
        self._tilt = (self._alpha_init - self._alpha) / max(1, self._init_phase)
        self.noise = np.full(self.n_freqs, self._min, dtype=np.float32)
        self.block_counter = 0

    def reset(self) -> None:
        self.noise.fill(self._min)
        self.block_counter = 0

    def _alpha_now(self) -> float:
        if self.block_counter > (self._init_phase + self._avg_init):
            return self._alpha
        return self._alpha_init - self._tilt * (self.block_counter - self._avg_init)

    def update(self, spectrum: np.ndarray) -> None:
        """Update noise floor with a single-frame PSD vector."""
        self.block_counter += 1
        if self.block_counter <= self._avg_init:
            self.noise += (1.0 / self._avg_init) * spectrum.astype(np.float32)
            return
        alpha = self._alpha_now()
        # Per-bin update following NoiseSpectrum::UpdateBandBySmoothing.
        rising = spectrum > self.noise
        falling = ~rising
        if np.any(rising):
            pb = spectrum[rising].astype(np.float32)
            pn = self.noise[rising]
            alpha_inc = alpha * (pn / np.maximum(pb, 1e-30))
            if self.block_counter > self._init_phase:
                mask10 = (10.0 * pn) < pb
                alpha_inc = np.where(mask10, alpha_inc * 0.1, alpha_inc)
            self.noise[rising] = pn + alpha_inc * (pb - pn)
        if np.any(falling):
            pb = spectrum[falling].astype(np.float32)
            pn = self.noise[falling]
            self.noise[falling] = np.maximum(pn + alpha * (pb - pn), self._min)

    def power(self, k: int) -> float:
        return float(self.noise[k])


class StationarityEstimator:
    """Render-path per-bin stationarity flag.

    Mirrors AEC3 ``StationarityEstimator`` collapsed for single-channel use.
    The history ring of past render PSDs is held externally (we feed a
    pre-stacked window into ``update_stationarity_flags``).
    """

    def __init__(self, n_freqs: int, hop_samples: int = 160,
                 sample_rate: int = 16000) -> None:
        self.n_freqs = int(n_freqs)
        # Stationarity tunables auto-rescaled from AEC3 block counts to our
        # hop count via aec3_scale helpers so any hop/sr stays AEC3 strict.
        from .. import aec3_scale as _aec3_scale
        self._window_hops = _aec3_scale.blocks_to_hops(13, hop_samples, sample_rate)
        self._hangover_hops = _aec3_scale.blocks_to_hops(12, hop_samples, sample_rate)
        self.noise = _NoiseSpectrum(
            n_freqs=self.n_freqs,
            min_noise_power=STATIONARITY_MIN_NOISE_POWER_FLOAT,
            avg_init_hops=STATIONARITY_AVG_INIT_HOPS_DEFAULT,
            initial_phase_hops=STATIONARITY_INITIAL_PHASE_HOPS_DEFAULT,
            alpha=STATIONARITY_ALPHA,
            alpha_init=STATIONARITY_ALPHA_INIT,
        )
        self.hangovers = np.zeros(self.n_freqs, dtype=np.int32)
        self.stationarity_flags = np.zeros(self.n_freqs, dtype=bool)
        # Internal ring buffer of past render PSDs, length = window_hops.
        self._history = np.zeros((self._window_hops, self.n_freqs), dtype=np.float32)
        self._history_filled = 0
        self._history_write = 0

    def reset(self) -> None:
        self.noise.reset()
        self.hangovers.fill(0)
        self.stationarity_flags.fill(False)
        self._history.fill(0.0)
        self._history_filled = 0
        self._history_write = 0

    def update_noise_estimator(self, render_psd: np.ndarray) -> None:
        """Feed the noise floor tracker with the latest render PSD."""
        self.noise.update(render_psd)

    def update_stationarity_flags(self, render_psd: np.ndarray,
                                  average_reverb: np.ndarray | None = None) -> None:
        """Refresh per-bin stationarity flags using the latest render PSD.

        Internally maintains a ``window_hops``-length history of render PSDs;
        ``average_reverb`` is added to the accumulated power per AEC3 (pass
        zeros if reverb estimate is not yet available).
        """
        psd32 = render_psd.astype(np.float32)
        self._history[self._history_write] = psd32
        self._history_write = (self._history_write + 1) % self._window_hops
        self._history_filled = min(self._history_filled + 1, self._window_hops)

        rev = (np.zeros(self.n_freqs, dtype=np.float32)
               if average_reverb is None
               else average_reverb.astype(np.float32))
        acum_power = self._history[:self._history_filled].sum(axis=0) + rev
        noise = self._window_hops * np.maximum(self.noise.noise, 1e-30)
        self.stationarity_flags = acum_power < STATIONARITY_THR_RATIO * noise
        self._update_hangover()
        self._smooth_per_freq()

    def _update_hangover(self) -> None:
        all_stationary = bool(np.all(self.stationarity_flags))
        not_stationary = ~self.stationarity_flags
        self.hangovers[not_stationary] = self._hangover_hops
        if all_stationary:
            np.maximum(self.hangovers - 1, 0, out=self.hangovers)

    def _smooth_per_freq(self) -> None:
        f = self.stationarity_flags
        smoothed = np.zeros_like(f)
        if self.n_freqs >= 3:
            smoothed[1:-1] = f[:-2] & f[1:-1] & f[2:]
            smoothed[0] = smoothed[1]
            smoothed[-1] = smoothed[-2]
        self.stationarity_flags = smoothed

    def is_band_stationary(self, k: int) -> bool:
        return bool(self.stationarity_flags[k]) and (self.hangovers[k] == 0)

    def band_stationary_mask(self) -> np.ndarray:
        """Per-bin bool mask (length n_freqs) of currently stationary bands."""
        return self.stationarity_flags & (self.hangovers == 0)

    def is_block_stationary(self) -> bool:
        return float(np.mean(self.band_stationary_mask())) > STATIONARITY_BLOCK_FRACTION


def get_residual_echo_scaling(stationarity: StationarityEstimator,
                              filter_has_had_time_to_converge: bool) -> np.ndarray:
    """Per-bin residual scaling (echo_audibility.h:40-51).

    Returns 0.0 for stationary bands once the filter has converged enough that
    we can trust the stationarity flag (avoids zeroing R² before the noise
    floor is reliably learned), 1.0 otherwise.
    """
    scaling = np.ones(stationarity.n_freqs, dtype=np.float32)
    if filter_has_had_time_to_converge:
        scaling[stationarity.band_stationary_mask()] = 0.0
    return scaling


__all__ = ["StationarityEstimator", "get_residual_echo_scaling"]
