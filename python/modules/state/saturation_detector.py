"""AecState::SaturationDetector — flags echo saturation (filter or render path).

Mirrors docs/aec3_extracts/src/aec3/aec_state.cc:446-478 verbatim.

Threshold ``kSaturationThreshold = 20000`` is in raw int16 amplitude
units; works on filter-path subtractor outputs when the linear estimate
is usable, otherwise on the render block scaled by ``echo_path_gain`` *
margin.
"""
import numpy as np


_SATURATION_THRESHOLD = 20000.0
_RENDER_MARGIN = 10.0
_INT16_SATURATION = 32000.0


class SaturationDetector:
    def __init__(self) -> None:
        self._saturated_echo = False

    def update(
        self,
        *,
        render_block: np.ndarray,
        saturated_capture: bool,
        usable_linear_estimate: bool,
        subtractor_s_refined_max_abs: float,
        subtractor_s_coarse_max_abs: float,
        echo_path_gain: float,
    ) -> None:
        self._saturated_echo = False
        if not saturated_capture:
            return
        if usable_linear_estimate:
            self._saturated_echo = (
                subtractor_s_refined_max_abs > _SATURATION_THRESHOLD
                or subtractor_s_coarse_max_abs > _SATURATION_THRESHOLD
            )
        else:
            max_sample = float(np.abs(render_block).max()) if render_block.size > 0 else 0.0
            peak_echo_amplitude = max_sample * echo_path_gain * _RENDER_MARGIN
            self._saturated_echo = peak_echo_amplitude > _INT16_SATURATION

    def saturated_echo(self) -> bool:
        return self._saturated_echo
