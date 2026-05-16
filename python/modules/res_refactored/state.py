"""ResState container + module result value-types.

Anti-pattern guard: every module reads `state.*` and `ResidualResult.*` /
`GainResult.*` etc. Modules MUST NOT touch external `self` of a `ResFilter`.
This makes byte-equal comparison testable per-module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ResState:
    """All per-frame mutable state owned by the RES post-filter."""

    # --- Pre-stage / common state (driven by orchestrator before Module 1) ---
    echo_psd: np.ndarray = None              # (n_freqs,) EMA of |echo_spec|^2
    error_psd: np.ndarray = None             # (n_freqs,) EMA of |error|^2
    coh2_smooth: np.ndarray = None           # (n_freqs,) asymmetric EMA coh^2
    S_fe: np.ndarray = None                  # complex cross-PSD
    S_ff: np.ndarray = None
    S_ee: np.ndarray = None
    far_activity: float = 0.0                # [0, 1] far-end activity scalar

    # --- Module 1: ResidualEstimator state ---
    reverb_psd: np.ndarray = None            # (n_freqs,) reverb tail
    residual_estimator_state: Any = None     # opaque sub-state for ResidualEchoEstimator
    using_render_based: bool = False

    # --- Module 4: TemporalSmoother state ---
    gain_smooth: np.ndarray = None           # (n_freqs,) EMA gain
    prev_gain_smooth: np.ndarray = None
    rate_limit_state: Dict[str, Any] = field(default_factory=dict)

    # --- Module 5: NoiseFloorAndCng state ---
    noise_psd: np.ndarray = None             # (n_freqs,)
    smooth_cn_gain: np.ndarray = None
    ola_buf: np.ndarray = None

    # --- Frame ingestion (orchestrator-owned) ---
    input_buf: np.ndarray = None             # frame_size buffer

    # --- Diagnostic write-back surface (audio-passive) ---
    diag_round5_stages: np.ndarray = None    # (9,) per-stage voice-band gain mean
    diag_coh2_last: np.ndarray = None
    diag_nearend_est_last: np.ndarray = None
    diag_residual_echo_psd_last: np.ndarray = None
    diag_dt_per_bin_last: np.ndarray = None


@dataclass
class ResidualResult:
    """Module 1 output."""
    residual_echo_psd: np.ndarray            # (n_freqs,)
    residual_path_used: str = 'linear'        # P49 M3 enum; one of 'linear','render','blend','reverb'


@dataclass
class GainResult:
    """Module 2 / 3 / 4 cumulative output (per-bin amplitude gain)."""
    gain: np.ndarray                         # (n_freqs,) per-bin amplitude


@dataclass
class NoiseFloorResult:
    """Module 5 output — final synthesised hop."""
    output: np.ndarray                       # (hop_size,) time-domain output
    gain_final: np.ndarray                   # (n_freqs,) final per-bin gain post-floor+CNG
