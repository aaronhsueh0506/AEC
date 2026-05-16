"""ERLE estimators + cross-validation confidence.

Extracted from ``aec.py`` during refactor R.4. Self-contained: depends
only on numpy.
"""
import numpy as np


class FilterErleEstimator:
    """Per-bin ERLE from adaptive filter echo estimate vs error.
    erle[k] = |echo_spec[k]|² / |error_spec[k]|²

    Key difference from erle_per_bin:
    - Does NOT use near_psd → breaks circular dependency
    - DT: erle naturally drops (speech increases error, echo_spec stays)
    """
    def __init__(self, n_freqs: int):
        self.n_freqs = n_freqs
        self.erle = np.ones(n_freqs, dtype=np.float32)
        self._alpha_rise = 0.95   # slow rise (stable convergence)
        self._alpha_drop = 0.7    # fast drop (DT protection)

    def update(self, echo_spec: np.ndarray, error_spec: np.ndarray,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active:
            return
        echo_pwr = np.abs(echo_spec) ** 2
        error_pwr = np.abs(error_spec) ** 2 + 1e-10
        inst_erle = np.clip(echo_pwr / error_pwr, 0.1, 1000.0)

        # Asymmetric EMA: fast drop (DT protection), slow rise (FS convergence).
        # Bug 2 fix: dt_indicator now freezes rise (alpha_rise→1), instead of
        # inverted dt_weight that accelerated rise during DT (echo leak).
        dt_factor = float(np.clip(dt_indicator, 0.0, 1.0))
        alpha_rise_eff = self._alpha_rise + (1.0 - self._alpha_rise) * dt_factor
        alpha = np.where(
            inst_erle < self.erle,
            self._alpha_drop,
            alpha_rise_eff,
        )
        self.erle = alpha * self.erle + (1.0 - alpha) * inst_erle

        # 3-bin smoothing + cap
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        self.erle = np.convolve(self.erle, kernel, mode='same').astype(np.float32)
        self.erle = np.clip(self.erle, 0.5, 200.0)

    def reset(self):
        self.erle.fill(1.0)


class FullbandErleEstimator:
    """Broadband ERLE for cross-validation confidence.
    Uses near_psd/error_psd broadband mean — stable but slow, FS-only update.
    """
    def __init__(self):
        self.fb_erle = 1.0
        self._alpha = 0.97

    def update(self, near_power: float, error_power: float,
               far_active: bool, dt_indicator: float) -> None:
        if not far_active or dt_indicator > 0.3 or near_power < 1e-8:
            return
        inst = np.clip(near_power / (error_power + 1e-10), 0.5, 100.0)
        self.fb_erle = self._alpha * self.fb_erle + (1.0 - self._alpha) * inst

    def reset(self):
        self.fb_erle = 1.0


def compute_erle_confidence(erle_l1: np.ndarray, fb_erle: float) -> float:
    """Compare FilterErle mean (L1) with FullbandErle (L2).
    Returns confidence in [0, 1]: 1 = consistent, 0 = divergent.
    """
    l1_mean = float(np.mean(erle_l1))
    if l1_mean < 0.5 or fb_erle < 0.5:
        return 0.0
    log_diff = abs(np.log(l1_mean + 1e-10) - np.log(fb_erle + 1e-10))
    return float(np.exp(-log_diff / 2.0))
