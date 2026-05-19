"""SubtractorOutputAnalyzer (SOA) — canonical AEC3 per-frame convergence detection.

Direct port of `docs/aec3_extracts/src/aec3/subtractor_output_analyzer.{cc,h}`.
Pure functional, single-channel (our pipeline has one mic + one ref).

Used in `_aec3_post()` to replace the bespoke per-frame convergence proxy
(orchestrator.py STRUCTURAL FIX commit 5f43c34). Output `any_filter_converged`
feeds `FilterStateBridge.filter_converged` → `FilteringQualityAnalyzer` →
`usable_linear_estimate()`.

TransparentMode currently stays disabled. The primary TM trigger is
`strong_not_saturated_render_blocks > 6 * kNumBlocksPerSecond` (6 s,
transparent_mode.cc:205-214), NOT the 60 s active-non-converged timer
(that 60 s is the secondary reset for `recent_convergence_during_activity`,
not the firing condition). For short corpus cases (10-30 s) TM DOES fire
once the 6 s window passes — and on FS converging cases it mis-fires
during the 0-6 s window before the filter is recognised as `sane`. Phase 2
800-case A/B confirmed -0.06 dB FS_static when TM was enabled together
with SOA. Real fix: port `FilterAnalyzer.ConsistentFilterDetector`
(canonical impulse-response shape detector, 1.5 s sustained pattern)
so `any_filter_consistent` is sourced canonically rather than via the
loose orchestrator proxy `any_filter_converged && external_delay
is not None`.

Threshold scaling. Canonical AEC3 evaluates over kBlockSize=64 sample
blocks of int16-equivalent audio; thresholds:

    kConvergenceThreshold         = 50² × 64 = 160000   (int16²·samples)
    kConvergenceThresholdLowLevel = 20² × 64 = 25600
    Divergence y² floor           = 30² × 64 = 57600

Our pipeline uses hop_size = 160 samples of float[-1, 1]. The float
equivalent of (peak_int16)² × kBlockSize is `(peak/32768)² × hop_size`.
At hop_size=160:
    50-peak threshold ≈ 3.73e-4
    20-peak threshold ≈ 5.96e-5
    30-peak threshold ≈ 1.34e-4

Caller contract: `update()` receives `y², e²_refined, e²_coarse` all
summed over the same number of time-domain samples (typically hop_size).
"""
from dataclasses import dataclass, field
from typing import Tuple


_INT16_PEAK_FULL_SCALE = 32768.0


def _scale_int16_threshold(int16_peak: float, hop_size: int) -> float:
    """(peak / 32768)² × hop_size — float[-1,1]² hop-energy unit."""
    return (int16_peak / _INT16_PEAK_FULL_SCALE) ** 2 * hop_size


@dataclass
class SubtractorOutputAnalyzer:
    """Canonical AEC3 SOA, single-channel.

    Tracks one per-channel `filters_converged` latched bool (the public
    `ConvergedFilters()` accessor in canonical). `update()` recomputes
    three transient flags (`any_filter_converged`, `any_coarse_filter_
    converged`, `all_filters_diverged`) and updates the latched per-channel
    flag. `handle_echo_path_change()` clears the latched flag (echo path
    changed → discard convergence belief).
    """

    hop_size: int = 160
    filters_converged: bool = False
    _y2_high: float = field(init=False, default=0.0)
    _y2_low: float = field(init=False, default=0.0)
    _y2_div: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._y2_high = _scale_int16_threshold(50.0, self.hop_size)
        self._y2_low = _scale_int16_threshold(20.0, self.hop_size)
        self._y2_div = _scale_int16_threshold(30.0, self.hop_size)

    def update(
        self,
        *,
        y2: float,
        e2_refined: float,
        e2_coarse: float,
    ) -> Tuple[bool, bool, bool]:
        """Per-frame SOA evaluation. Direct port of canonical Update().

        Returns (any_filter_converged, any_coarse_filter_converged,
        all_filters_diverged).
        """
        refined_filter_converged = e2_refined < 0.5 * y2 and y2 > self._y2_high
        coarse_filter_converged_strict = e2_coarse < 0.05 * y2 and y2 > self._y2_high
        coarse_filter_converged_relaxed = e2_coarse < 0.3 * y2 and y2 > self._y2_low
        min_e2 = e2_refined if e2_refined < e2_coarse else e2_coarse
        filter_diverged = min_e2 > 1.5 * y2 and y2 > self._y2_div

        self.filters_converged = (
            refined_filter_converged or coarse_filter_converged_strict
        )

        any_filter_converged = self.filters_converged
        any_coarse_filter_converged = coarse_filter_converged_relaxed
        all_filters_diverged = filter_diverged
        return any_filter_converged, any_coarse_filter_converged, all_filters_diverged

    def handle_echo_path_change(self) -> None:
        """Reset latched per-channel convergence flag — path-change event."""
        self.filters_converged = False
