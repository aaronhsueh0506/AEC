"""AecDebugLogger: periodic console dump of AEC internal state (read-only)."""
import numpy as np


class AecDebugLogger:
    """Logs AEC internal state to the console every ``log_interval_s`` seconds.

    Call ``update()`` once per processed frame.
    """

    _HEADER = (
        "  t(s)  | state          |fltr| epc |dt_conf|ERLE_i|ERLE_w|  ERL |"
        "far_dB|mic_dB|mu_scl|shd_adv|sat | delay"
    )
    _SEP = "-" * len(_HEADER)

    def __init__(self, aec: 'AEC', log_interval_s: float = 1.0):
        self._aec = aec
        self._interval_frames = max(1, int(log_interval_s / (aec._hop_size / aec.config.sample_rate)))
        self._tick = 0
        self._header_interval = 20

    def update(self) -> None:
        self._tick += 1
        if self._tick % self._interval_frames != 0:
            return
        s = self._aec.get_stats()
        line_no = self._tick // self._interval_frames
        if line_no % self._header_interval == 1:
            print(self._SEP)
            print(self._HEADER)
            print(self._SEP)
        conv = 'Y' if s.filter_converged else 'N'
        epc  = 'Y' if s.epc_active else 'N'
        print(
            f"{s.time_s:7.2f}  | {s.filter_state.value:<14s} | {conv}  | {epc}  |"
            f" {s.dt_confidence:5.3f} |{s.erle_inst_db:6.1f}|{s.erle_windowed_db:6.1f}|"
            f"{s.erl_db:6.1f}|{s.far_power_db:6.1f}|{s.mic_power_db:6.1f}|"
            f" {s.mu_scale:5.3f}| {s.shadow_advantage:5.3f} |{s.saturation_level:4.2f}|"
            f" {s.delay_ms:.1f}ms"
        )
