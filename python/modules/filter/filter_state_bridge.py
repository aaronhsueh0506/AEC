"""Read-only view of the linear-filter family for the AEC3 state machine.

The AEC3 ``AecState`` (Phase 3) reads several signals from the linear
filter to make per-frame decisions (UsableLinearEstimate gates,
HandleEchoPathChange triggers, reverb_decay_estimator filter taps,
filter_quality gating). Those signals live across three of our modules:

  - PBFDKF (``python/modules/filters.py``): filter weights / Kalman P /
    error spectrum / far-end power.
  - ShadowFilter (``python/modules/filters.py``, instantiated as a
    second PBFDKF or NLMS): divergence indicator via shadow err ratio.
  - PathChangeRegimeHandler (``python/modules/epc.py``): regime classifier
    (POS / CTL_FS / CTL_TAIL), main_paused flag, copy_counter.

This bridge gives AecState a stable, typed surface to read from
without coupling it to PBFDKF / Shadow / Regime internals. The
orchestrator builds one per hop with ``build_filter_state_bridge`` and
hands the snapshot to ``aec_state.update``.

Notes:
  - ``filter_taps`` is recovered from ``PBFDKF.W`` via inverse FFT of
    partition 0 (the most recent block's impulse response). Used by
    ``reverb_decay_estimator`` to extract reverb tail slope. Other
    partitions are NOT exposed here — reverb decay only needs the
    head response.
  - ``divergence_indicator`` is a unitless [0, ∞) scalar; high values
    mean the filter has likely diverged.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FilterStateBridge:
    """Per-frame snapshot consumed by AecState."""

    filter_converged: bool
    """True once the main filter has crossed the convergence detector's
    threshold (set by orchestrator's FilterConvergenceAnalyzer)."""

    filter_taps: np.ndarray
    """Time-domain impulse response (first partition of W, IFFT'd).
    Shape: (filter_length,) float32. Used by ReverbDecayEstimator."""

    divergence_indicator: float
    """Scalar in [0, ∞); higher = more likely diverged. Computed from
    PBFDKF Kalman P_trace + Shadow E_ratio. Zero when the filter is
    inactive / cold-started."""

    regime: int
    """PathChangeRegimeHandler regime as an integer code. See
    ``epc.PathChangeRegimeHandler`` for the enum. Used by AecState to
    decide whether to treat the current frame as a path-change window
    (favouring non-linear path)."""

    main_paused: bool
    """True when the regime handler has frozen main-filter updates
    (typically right after a detected path change). Equivalent to
    AEC3 ``filter_quality_state == kBadFilter`` for our cohort."""

    mu_final: float
    """Latest effective mu_scale applied to the main filter (post all
    gating / DTD / EPC / regime modulation). Useful for state-machine
    tracking of "is the filter actively learning now?"."""

    external_delay_samples: int
    """Current orchestrator-reported delay in raw samples (the value
    feeding render alignment). -1 if no delay estimate yet. AecState
    routes this into AEC3 ``FilterDelay`` state."""

    any_coarse_filter_converged: bool = False
    """SubtractorOutputAnalyzer relaxed coarse predicate
    (subtractor_output_analyzer.cc:50-51):
        e2_coarse < 0.3 * y2 AND y2 > kConvergenceThresholdLowLevel (20²·hop).
    Computed by orchestrator; consumed only when TransparentMode HMM is
    active (currently retired in production). Additive: zero-impact when
    no consumer is wired."""

    all_filters_diverged: bool = False
    """SubtractorOutputAnalyzer strict divergence predicate
    (subtractor_output_analyzer.cc:53):
        min(e2_refined, e2_coarse) > 1.5 * y2 AND y2 > 30²·hop.
    Replaces the legacy ``divergence_indicator > 1.0`` heuristic for
    TransparentMode consumers. Additive: zero-impact when no consumer is
    wired (TransparentMode HMM is retired in production)."""


def build_filter_state_bridge(
    *,
    filter_converged: bool,
    pbfdkf: object,
    regime_handler: object,
    mu_final: float,
    external_delay_samples: int,
    shadow_filter: Optional[object] = None,
    any_coarse_filter_converged: bool = False,
    all_filters_diverged: bool = False,
) -> FilterStateBridge:
    """Snapshot the linear-filter family into a FilterStateBridge.

    ``pbfdkf`` is the main filter; ``regime_handler`` is a
    PathChangeRegimeHandler instance; ``shadow_filter`` is optional. Reads
    only — no mutation of any source state.
    """
    # Time-domain filter taps from W[0]. Partition 0 carries the most
    # recent impulse response in PBFDKF/PBFDAF. Cast to float32 so the
    # reverb_decay_estimator can run real-only.
    W = getattr(pbfdkf, "W", None)
    fft_size = getattr(pbfdkf, "fft_size", None)
    if W is not None and fft_size is not None and W.shape[0] > 0:
        w_time = np.fft.irfft(W[0], fft_size).astype(np.float32)
    else:
        w_time = np.zeros(1, dtype=np.float32)

    # Divergence indicator: mean Kalman P_trace * (shadow_err / main_err).
    # Reasonable default 0 if any input is missing.
    div = 0.0
    P = getattr(pbfdkf, "P", None)
    if P is not None and P.size > 0:
        try:
            p_trace = float(np.mean(P))
        except Exception:
            p_trace = 0.0
        e_ratio = 1.0
        if shadow_filter is not None:
            try:
                main_e = float(pbfdkf.get_error_energy())
                shadow_e = float(shadow_filter.get_error_energy())
                if main_e > 1e-12:
                    e_ratio = shadow_e / main_e
            except Exception:
                e_ratio = 1.0
        div = p_trace * max(0.0, e_ratio - 1.0)

    # Regime: handler exposes implementation-specific state. Compress to
    # an int code (0=stable, 1=main_paused, 2=any active state). AecState
    # only needs the coarse classification.
    if getattr(regime_handler, "main_paused", False):
        regime_code = 1
    else:
        regime_code = 0

    return FilterStateBridge(
        filter_converged=bool(filter_converged),
        filter_taps=w_time,
        divergence_indicator=float(div),
        regime=int(regime_code),
        main_paused=bool(getattr(regime_handler, "main_paused", False)),
        mu_final=float(mu_final),
        external_delay_samples=int(external_delay_samples),
        any_coarse_filter_converged=bool(any_coarse_filter_converged),
        all_filters_diverged=bool(all_filters_diverged),
    )
