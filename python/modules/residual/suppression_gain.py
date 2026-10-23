"""SuppressionGain: per-bin suppression gain from the residual echo estimate (single channel, single band).

Per hop: DominantNearendDetector, LowNoiseRenderDetector, then the lower-band gain:
max-gain step limit, nearend smoothing, audibility weighting of the echo, minimum-gain
clamp, ENR/EMR mask gate, [min_gain, max_gain] clamp, LF/HF limiters, sqrt to amplitude.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..freq_utils import hz_to_bin


def _db_to_power_f32(db: float) -> float:
    """``10 ** (db / 10)`` evaluated in float32 (powf semantics)."""
    db32 = np.float32(db)
    return float(np.float32(10.0) ** (db32 / np.float32(10.0)))


@dataclass(frozen=True)
class MaskTuning:
    enr_transparent: float
    enr_suppress: float
    emr_transparent: float


@dataclass(frozen=True)
class SuppressorTuning:
    mask_lf: MaskTuning
    mask_hf: MaskTuning
    max_inc_factor: float = 2.0
    max_dec_factor_lf: float = 0.25


# Mask tuning defaults as (enr_transparent, enr_suppress, emr_transparent).
# Normal: lf (0.3, 0.4, 0.3), hf (0.07, 0.1, 0.3); nearend: lf (1.09, 1.1, 0.3), hf (0.1, 0.3, 0.3).
_DEFAULT_NEAREND_TUNING = SuppressorTuning(
    mask_lf=MaskTuning(1.09, 1.1, 0.3),
    mask_hf=MaskTuning(0.1, 0.3, 0.3),
)
_DEFAULT_NORMAL_TUNING = SuppressorTuning(
    mask_lf=MaskTuning(0.3, 0.4, 0.3),
    mask_hf=MaskTuning(0.07, 0.1, 0.3),
)


@dataclass(frozen=True)
class HighFrequencySuppressionConfig:
    # Limiting-gain anchor in Hz and a one-bin width (31.25 Hz at fft 512), so the cap scales with fft_size.
    limiting_gain_freq_hz: float = 2000.0
    limiting_gain_width_hz: float = 31.25


@dataclass(frozen=True)
class EchoAudibilityConfig:
    floor_power: float = 2.0 * 64.0          # AEC3 default
    audibility_threshold_lf: float = 10.0
    audibility_threshold_mf: float = 10.0
    audibility_threshold_hf: float = 10.0
    low_render_limit: float = 4.0 * 64.0     # min_echo_power when render quiet
    normal_render_limit: float = 64.0
    use_stationarity_properties: bool = False
    # Audibility-weighting band boundaries in Hz (bin spans 0-3 / 3-7 / 7-end at fft 128).
    lf_band_end_hz: float = 375.0
    mf_band_end_hz: float = 875.0


@dataclass(frozen=True)
class DominantNearendConfig:
    enr_threshold: float = 0.25
    enr_exit_threshold: float = 10.0
    snr_threshold: float = 30.0
    # hold_duration_ms: minimum near-end-state dwell in wall-clock ms (converted to hops).
    hold_duration_ms: int = 500
    # trigger_threshold: evidence depth (+1/-1 random walk) before the near-end state triggers,
    # in hops. With use_wallclock_trigger_threshold, trigger_threshold_ms is converted to hops instead.
    trigger_threshold: int = 12
    trigger_threshold_ms: int = 48
    use_wallclock_trigger_threshold: bool = False
    use_during_initial_phase: bool = True
    use_unbounded_echo_spectrum: bool = True
    # Upper edge in Hz of the low-frequency sum used for near-end detection.
    lf_endpoint_hz: float = 2000.0


@dataclass
class SuppressorConfig:
    # LF/HF mask interpolation boundaries in Hz: speech F0 sits in the LF-mask zone, F1 in the
    # interpolation zone, F2 and above in the HF-mask zone.
    last_lf_freq_hz: float = 625.0
    first_hf_freq_hz: float = 1000.0
    nearend_tuning: SuppressorTuning = field(default_factory=lambda: _DEFAULT_NEAREND_TUNING)
    normal_tuning: SuppressorTuning = field(default_factory=lambda: _DEFAULT_NORMAL_TUNING)
    last_lf_smoothing_freq_hz: float = 625.0
    last_permanent_lf_smoothing_band: int = 0
    lf_smoothing_during_initial_phase: bool = True
    conservative_hf_suppression: bool = False
    nearend_average_blocks: int = 4
    floor_first_increase: float = 0.00001
    high_frequency_suppression: HighFrequencySuppressionConfig = field(
        default_factory=HighFrequencySuppressionConfig
    )
    dominant_nearend_detection: DominantNearendConfig = field(
        default_factory=DominantNearendConfig
    )
    stat_aware_ne_proxy_enabled: bool = False
    stat_aware_ne_proxy_threshold: float = 0.10
    # Echo-audibility knobs consumed by ``_weight_echo_for_audibility``; the orchestrator
    # overrides ``use_stationarity_properties``.
    echo_audibility: EchoAudibilityConfig = field(default_factory=EchoAudibilityConfig)


class _MovingAverageSpectrum:
    """Sliding-window per-bin mean over the N most recent inputs."""

    def __init__(self, n_bins: int, n_blocks: int) -> None:
        self._n = max(1, int(n_blocks))
        self._buf: deque = deque(maxlen=self._n)
        self._n_bins = int(n_bins)

    def update_memory_length(self, n_blocks: int) -> None:
        n = max(1, int(n_blocks))
        if n != self._n:
            self._n = n
            self._buf = deque(self._buf, maxlen=n)

    def average(self, spectrum: np.ndarray) -> np.ndarray:
        self._buf.append(np.asarray(spectrum, dtype=np.float32))
        return np.mean(self._buf, axis=0).astype(np.float32)


class _LowNoiseRenderDetector:
    """Low-noise render detector: average render power below a threshold with no strong peak.

    The threshold is 50^2 * 64; ``use_wallclock_block_energy_threshold`` rescales it to the hop size.
    """

    def __init__(self, *, hop_samples: int = 64, sample_rate: int = 16000,
                 use_wallclock_block_energy_threshold: bool = False,
                 use_wallclock_ema_alpha: bool = False) -> None:
        self._average_power = 32768.0 * 32768.0
        self._hop_samples = hop_samples
        if use_wallclock_block_energy_threshold:
            from .. import aec3_scale as _aec3_scale
            self._threshold = float(
                _aec3_scale.block_energy_scale(50.0 * 50.0 * 64.0, hop_samples)
            )
        else:
            self._threshold = 50.0 * 50.0 * 64.0
        # Render power IIR: 0.9 * avg + 0.1 * x2.
        if use_wallclock_ema_alpha:
            from .. import aec3_scale as _aec3_scale
            _alpha = _aec3_scale.per_block_ema_alpha_to_per_hop(
                0.1, hop_samples, sample_rate
            )
            self._iir_decay = 1.0 - _alpha
            self._iir_weight = _alpha
        else:
            self._iir_decay = 0.9
            self._iir_weight = 0.1

    def detect(self, render_block: np.ndarray) -> bool:
        if render_block.size == 0:
            return False
        x2 = render_block.astype(np.float64) ** 2
        x2_sum = float(np.sum(x2))
        x2_max = float(np.max(x2))
        # Normalise average power to a 64-sample block before the peak comparison.
        avg_per_sample = self._average_power * (64.0 / self._hop_samples)
        low_noise = self._average_power < self._threshold and x2_max < 3.0 * avg_per_sample
        self._average_power = self._average_power * self._iir_decay + x2_sum * self._iir_weight
        return bool(low_noise)


def _build_gain_params(
    n_bins: int, last_lf_band: int, first_hf_band: int, tuning: SuppressorTuning
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin enr_transparent / enr_suppress / emr_transparent: linear interpolation from the LF to the HF tuning."""
    lf, hf = tuning.mask_lf, tuning.mask_hf
    enr_tr = np.empty(n_bins, dtype=np.float32)
    enr_su = np.empty(n_bins, dtype=np.float32)
    emr_tr = np.empty(n_bins, dtype=np.float32)
    for k in range(n_bins):
        if k <= last_lf_band:
            a = 0.0
        elif k < first_hf_band:
            a = (k - last_lf_band) / float(first_hf_band - last_lf_band)
        else:
            a = 1.0
        enr_tr[k] = (1 - a) * lf.enr_transparent + a * hf.enr_transparent
        enr_su[k] = (1 - a) * lf.enr_suppress + a * hf.enr_suppress
        emr_tr[k] = (1 - a) * lf.emr_transparent + a * hf.emr_transparent
    return enr_tr, enr_su, emr_tr


def _weight_echo_for_audibility(
    cfg: EchoAudibilityConfig, echo: np.ndarray, out: np.ndarray,
    sr: int = 16000,
) -> None:
    """Down-weight echo below the audibility threshold per band (LF / MF / HF):
    ``echo * max(0, 1 - ((thr - echo) / (thr - floor_power))^2)`` where ``thr = floor_power * audibility_threshold``."""

    def weigh(threshold: float, begin: int, end: int) -> None:
        if begin >= end:
            return
        normalizer = 1.0 / (threshold - cfg.floor_power)
        seg = echo[begin:end]
        result = seg.copy()
        below = seg < threshold
        if below.any():
            tmp = (threshold - seg[below]) * normalizer
            result[below] = seg[below] * np.maximum(0.0, 1.0 - tmp * tmp)
        out[begin:end] = result

    n = out.size
    lf_end = min(hz_to_bin(cfg.lf_band_end_hz, n, sr), n)
    mf_end = min(hz_to_bin(cfg.mf_band_end_hz, n, sr), n)
    weigh(cfg.floor_power * cfg.audibility_threshold_lf, 0, lf_end)
    weigh(cfg.floor_power * cfg.audibility_threshold_mf, lf_end, mf_end)
    weigh(cfg.floor_power * cfg.audibility_threshold_hf, mf_end, n)


def _limit_lf_gains(gain: np.ndarray, sr: int = 16000) -> None:
    """Clamp all bins below 250 Hz to the gain at 250 Hz."""
    n = gain.size
    clamp_to = hz_to_bin(250, n, sr)
    if clamp_to < n:
        anchor = float(gain[clamp_to])
        np.minimum(gain[:clamp_to], anchor, out=gain[:clamp_to])


def _limit_hf_gains(
    cfg: HighFrequencySuppressionConfig,
    conservative_hf: bool,
    gain: np.ndarray,
    sr: int = 16000,
) -> None:
    """Limit gains above the limiting-gain anchor to the anchor-band minimum; optional conservative cap at 2500-3625 Hz."""
    n_bins = gain.size
    lgb = hz_to_bin(cfg.limiting_gain_freq_hz, n_bins, sr)
    biq = max(1, hz_to_bin(cfg.limiting_gain_width_hz, n_bins, sr))
    if biq > 0 and lgb + biq <= n_bins:
        min_upper_gain = float(np.min(gain[lgb : lgb + biq]))
        np.minimum(gain[lgb + 1 :], min_upper_gain, out=gain[lgb + 1 :])
    if n_bins >= 2:
        gain[-1] = gain[-2]
    if conservative_hf:
        # Conservative cap band 2500-3625 Hz.
        cons_lo = hz_to_bin(2500.0, n_bins, sr)
        cons_hi = hz_to_bin(3625.0, n_bins, sr)
        if n_bins > cons_hi and cons_hi > cons_lo:
            hf_gain_bound = float(np.mean(gain[cons_lo:cons_hi]))
            np.minimum(gain[cons_hi:], hf_gain_bound, out=gain[cons_hi:])


class _DominantNearendDetector:
    """Dominant-near-end detector (single channel)."""

    def __init__(
        self,
        cfg: DominantNearendConfig,
        sr: int = 16000,
        hop_size: int = 160,
    ) -> None:
        self._cfg = cfg
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        # Hold and trigger durations in hops.
        from .. import aec3_scale as _aec3_scale
        self._hold_duration_hops = _aec3_scale.ms_to_hops(
            cfg.hold_duration_ms, self._hop_size, self._sr
        )
        self._trigger_threshold_hops = self._derive_trigger_threshold(cfg)
        self._trigger_counter = 0
        self._hold_counter = 0
        self._nearend_state = False
        # Per-frame diagnostic snapshot (read-only).
        self._last_update_snap: dict = {}

    def _derive_trigger_threshold(self, cfg: DominantNearendConfig) -> int:
        """Trigger threshold in hops (wall-clock ms if enabled, else the hop count)."""
        if cfg.use_wallclock_trigger_threshold:
            from .. import aec3_scale as _aec3_scale
            return _aec3_scale.ms_to_hops(
                cfg.trigger_threshold_ms, self._hop_size, self._sr
            )
        return int(cfg.trigger_threshold)

    def set_config(self, cfg: DominantNearendConfig) -> None:
        self._cfg = cfg
        from .. import aec3_scale as _aec3_scale
        self._hold_duration_hops = _aec3_scale.ms_to_hops(
            cfg.hold_duration_ms, self._hop_size, self._sr
        )
        self._trigger_threshold_hops = self._derive_trigger_threshold(cfg)

    def is_nearend_state(self) -> bool:
        return self._nearend_state

    def update(
        self,
        nearend_spectrum: np.ndarray,
        residual_echo: np.ndarray,
        comfort_noise: np.ndarray,
        initial_state: bool,
    ) -> None:
        """Update the near-end state: three independent blocks per frame (trigger, early exit, hold
        decrement). The LF sums skip DC; the trigger is gated inline on the initial state so the hold
        counter always decrements; the exit also requires echo above ``snr_threshold`` times noise."""
        c = self._cfg
        # LF sum endpoint from lf_endpoint_hz.
        n_bins = nearend_spectrum.size
        lf_end = min(hz_to_bin(c.lf_endpoint_hz, n_bins, self._sr), n_bins)
        ne_sum = float(np.sum(nearend_spectrum[1:lf_end]))
        echo_sum = float(np.sum(residual_echo[1:lf_end]))
        noise_sum = float(np.sum(comfort_noise[1:lf_end]))

        # Block 1: trigger (multiplicative thresholds).
        trigger_initial_gate = (not initial_state or c.use_during_initial_phase)
        trigger_enr_pass = echo_sum < c.enr_threshold * ne_sum
        trigger_snr_pass = ne_sum > c.snr_threshold * noise_sum
        trigger_active = trigger_initial_gate and trigger_enr_pass and trigger_snr_pass
        if trigger_active:
            self._trigger_counter += 1
            if self._trigger_counter >= self._trigger_threshold_hops:
                self._hold_counter = self._hold_duration_hops
                self._trigger_counter = self._trigger_threshold_hops
        else:
            self._trigger_counter = max(0, self._trigger_counter - 1)

        early_exit = (echo_sum > c.enr_exit_threshold * ne_sum
                      and echo_sum > c.snr_threshold * noise_sum)
        if early_exit:
            self._hold_counter = 0

        self._hold_counter = max(0, self._hold_counter - 1)
        self._nearend_state = self._hold_counter > 0

        # Diagnostic snapshot (ratios use a +1.0 denominator floor for logging).
        self._last_update_snap = {
            "ne_sum_lf": ne_sum,
            "echo_sum_lf": echo_sum,
            "noise_sum_lf": noise_sum,
            "enr": echo_sum / (ne_sum + 1.0),
            "snr": ne_sum / (noise_sum + 1.0),
            "enr_threshold": float(c.enr_threshold),
            "enr_exit_threshold": float(c.enr_exit_threshold),
            "snr_threshold": float(c.snr_threshold),
            "trigger_enr_pass": bool(trigger_enr_pass),
            "trigger_snr_pass": bool(trigger_snr_pass),
            "trigger_initial_gate": bool(trigger_initial_gate),
            "trigger_active": bool(trigger_active),
            "trigger_counter": int(self._trigger_counter),
            "early_exit_fired": bool(early_exit),
            "hold_counter": int(self._hold_counter),
            "hold_duration_hops": int(self._hold_duration_hops),
            "trigger_threshold_hops": int(self._trigger_threshold_hops),
            "use_wallclock_trigger_threshold": bool(c.use_wallclock_trigger_threshold),
            "nearend_state": bool(self._nearend_state),
            "initial_state": bool(initial_state),
        }


class SuppressionGain:
    """Single-channel single-band suppression gain."""

    def __init__(self, *, n_bins: int = 257, config: Optional[SuppressorConfig] = None,
                 sr: int = 16000, hop_size: int = 160,
                 use_wallclock_block_energy_threshold: bool = False,
                 use_wallclock_ema_alpha: bool = False,
                 use_wallclock_gain_ratchet: bool = False,
                 soft_nearend_blend_enabled: bool = False,
                 soft_nearend_blend_enr_threshold: float = 0.25,
                 soft_nearend_blend_softness: float = 0.25,
                 soft_nearend_blend_per_bin: bool = False,
                 split_floor_enabled: bool = True,
                 split_floor_far_active_db: float = -22.0,
                 split_floor_far_silent_db: float = -12.0,
                 split_floor_dt_db: float = -20.0,
                 split_floor_latch_power: float = 1.0e6,
                 per_bin_floor_enabled: bool = False) -> None:
        self._n_bins = int(n_bins)
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        self._config = config or SuppressorConfig()
        # Split min-gain floor (see AecConfig.min_gain_split_floor_*); power-domain floors from amplitude dB.
        self._split_floor_enabled = bool(split_floor_enabled)
        # ``_split_floor_far_active`` is the target and ``_live`` the applied value; they stay equal
        # unless set_split_floor_far_active_db requests a ramp. Float32 powf.
        self._split_floor_far_active = _db_to_power_f32(
            split_floor_far_active_db)
        self._split_floor_far_active_live = self._split_floor_far_active
        self._split_floor_ramp_ratio = 1.0
        self._split_floor_far_silent = _db_to_power_f32(
            split_floor_far_silent_db)
        # Floor used in place of far_active while the orchestrator sets _dt_protect_active (double-talk).
        self._split_floor_dt = _db_to_power_f32(split_floor_dt_db)
        self._dt_protect_active = False
        self._split_floor_latch_power = float(split_floor_latch_power)
        self._far_active_latched = False
        # Per-bin floor blend (see AecConfig.per_bin_floor_enabled): reuses the soft_nearend_blend_per_bin
        # sigmoid parameters as the per-bin near/echo weight.
        self._per_bin_floor_enabled = bool(per_bin_floor_enabled)
        # echo_audibility lives on SuppressorConfig so the orchestrator can override use_stationarity_properties.
        self._echo_audibility = self._config.echo_audibility
        self._last_gain = np.ones(self._n_bins, dtype=np.float32)
        self._last_nearend = np.zeros(self._n_bins, dtype=np.float32)
        self._last_echo = np.zeros(self._n_bins, dtype=np.float32)
        # Fraction of stationary bins from the orchestrator-supplied mask; used only by the stat-aware proxy.
        self._stat_mask_frac: float = 0.0
        self._low_render = _LowNoiseRenderDetector(
            hop_samples=self._hop_size,
            sample_rate=self._sr,
            use_wallclock_block_energy_threshold=bool(
                use_wallclock_block_energy_threshold
            ),
            use_wallclock_ema_alpha=bool(use_wallclock_ema_alpha),
        )
        # Near-end average window in hops (config is in 4 ms blocks).
        from .. import aec3_scale as _aec3_scale
        _n_smooth_hops = _aec3_scale.blocks_to_hops(
            self._config.nearend_average_blocks, self._hop_size, self._sr
        )
        self._nearend_smoother = _MovingAverageSpectrum(
            n_bins=self._n_bins, n_blocks=_n_smooth_hops
        )
        self._dominant_nearend = _DominantNearendDetector(
            self._config.dominant_nearend_detection,
            sr=self._sr,
            hop_size=self._hop_size,
        )
        self._initial_state = True
        self._last_lower_band_snap: dict = {}
        self._last_g_lin = np.zeros(self._n_bins, dtype=np.float32)
        self._last_g_emr = np.zeros(self._n_bins, dtype=np.float32)
        self._last_fire_mask = np.zeros(self._n_bins, dtype=bool)
        self._last_enr_raw = np.zeros(self._n_bins, dtype=np.float32)
        self._last_emr_raw = np.zeros(self._n_bins, dtype=np.float32)
        # Soft near-end tuning blend: sigmoid((enr_threshold - enr_lf) / softness) weights the
        # near-end tuning against the normal tuning instead of a binary switch.
        self._soft_ne_blend_enabled = bool(soft_nearend_blend_enabled)
        self._soft_ne_blend_per_bin = bool(soft_nearend_blend_per_bin)
        self._soft_ne_blend_enr_thr = float(soft_nearend_blend_enr_threshold)
        self._soft_ne_blend_softness = max(float(soft_nearend_blend_softness), 1e-6)
        self._dne_lf_end = min(hz_to_bin(2000.0, self._n_bins, self._sr), self._n_bins)
        self._last_lf_band = hz_to_bin(self._config.last_lf_freq_hz, self._n_bins, self._sr)
        self._first_hf_band = hz_to_bin(self._config.first_hf_freq_hz, self._n_bins, self._sr)
        self._last_lf_smoothing_band = hz_to_bin(
            self._config.last_lf_smoothing_freq_hz, self._n_bins, self._sr
        )
        self._nearend_enr_tr, self._nearend_enr_su, self._nearend_emr_tr = _build_gain_params(
            self._n_bins, self._last_lf_band, self._first_hf_band,
            self._config.nearend_tuning,
        )
        self._normal_enr_tr, self._normal_enr_su, self._normal_emr_tr = _build_gain_params(
            self._n_bins, self._last_lf_band, self._first_hf_band,
            self._config.normal_tuning,
        )
        # The max-gain ratchet factors are applied once per call; with the wall-clock flag they
        # are converted from per-block to per-hop.
        if use_wallclock_gain_ratchet:
            _hop = self._hop_size
            _sr = self._sr
            self._max_inc_nearend = float(_aec3_scale.per_block_growth_to_per_hop(
                self._config.nearend_tuning.max_inc_factor, _hop, _sr))
            self._max_inc_normal = float(_aec3_scale.per_block_growth_to_per_hop(
                self._config.normal_tuning.max_inc_factor, _hop, _sr))
            self._max_dec_lf_nearend = float(_aec3_scale.per_block_growth_to_per_hop(
                self._config.nearend_tuning.max_dec_factor_lf, _hop, _sr))
            self._max_dec_lf_normal = float(_aec3_scale.per_block_growth_to_per_hop(
                self._config.normal_tuning.max_dec_factor_lf, _hop, _sr))
        else:
            self._max_inc_nearend = float(self._config.nearend_tuning.max_inc_factor)
            self._max_inc_normal = float(self._config.normal_tuning.max_inc_factor)
            self._max_dec_lf_nearend = float(self._config.nearend_tuning.max_dec_factor_lf)
            self._max_dec_lf_normal = float(self._config.normal_tuning.max_dec_factor_lf)

    # Runtime far-active floor retarget. The far-active split floor is the only field the shipped
    # presets differ in, so retargeting it is a runtime preset change. The target is
    # ``_split_floor_far_active`` and ``_live`` walks toward it geometrically (linear in dB).
    # Contract: call between hops only; ramp_ms == 0 applies immediately; a call during a ramp
    # restarts from the current live value; nothing is written unless all arguments validate.
    SPLIT_FLOOR_DB_MIN = -300.0
    SPLIT_FLOOR_DB_MAX = 50.0
    SPLIT_FLOOR_RAMP_MS_MAX = 60000.0

    def _check_split_floor_db(self, db: float, name: str) -> None:
        if not np.isfinite(db) or not (self.SPLIT_FLOOR_DB_MIN <= db
                                       <= self.SPLIT_FLOOR_DB_MAX):
            raise ValueError(
                f"{name} must be finite and within "
                f"[{self.SPLIT_FLOOR_DB_MIN}, {self.SPLIT_FLOOR_DB_MAX}]: {db}")

    def set_split_floor_far_active_db(self, db: float,
                                      ramp_ms: float = 0.0) -> None:
        db = float(db)
        ramp_ms = float(ramp_ms)
        # Validate everything before writing anything.
        self._check_split_floor_db(db, "split_floor_far_active_db")
        if not np.isfinite(ramp_ms) or not (0.0 <= ramp_ms
                                            <= self.SPLIT_FLOOR_RAMP_MS_MAX):
            raise ValueError(
                "ramp_ms must be finite and within "
                f"[0, {self.SPLIT_FLOOR_RAMP_MS_MAX}]: {ramp_ms}")
        target = _db_to_power_f32(db)
        live = self._split_floor_far_active_live
        hops = 0
        if ramp_ms > 0.0 and target > 0.0 and live > 0.0 and live != target:
            from .. import aec3_scale as _aec3_scale
            hops = _aec3_scale.ms_to_hops_f32(
                ramp_ms, self._hop_size, self._sr)
        self._split_floor_far_active = target
        if hops == 0:
            self._split_floor_far_active_live = target
            self._split_floor_ramp_ratio = 1.0
            return
        # Float32 walk, so the target is reached on a deterministic hop.
        ratio = float(
            np.float32(target / live)
            ** (np.float32(1.0) / np.float32(hops))
        )
        if ratio == 1.0:
            # Per-hop factor rounds to identity: land now rather than stall.
            self._split_floor_far_active_live = target
            self._split_floor_ramp_ratio = 1.0
            return
        self._split_floor_ramp_ratio = ratio

    def _advance_split_floor_ramp(self) -> None:
        live = self._split_floor_far_active_live
        target = self._split_floor_far_active
        if live == target:
            return
        ratio = self._split_floor_ramp_ratio
        value = float(np.float32(live) * np.float32(ratio))
        # Land exactly on the target, leaving no residue.
        if ratio == 1.0 or (value >= target if ratio > 1.0 else value <= target):
            value = target
        self._split_floor_far_active_live = value

    # Runtime DT-floor retarget: the DT-lifted floor is a hard per-hop override, so it steps
    # immediately (no ramp).
    def set_split_floor_dt_db(self, db: float) -> None:
        db = float(db)
        self._check_split_floor_db(db, "split_floor_dt_db")
        self._split_floor_dt = _db_to_power_f32(db)

    # Runtime floor-gate retarget: a boolean gate steps immediately; any truthy/falsy value coerces.
    def set_split_floor_enabled(self, enabled: bool) -> None:
        self._split_floor_enabled = bool(enabled)

    def set_initial_state(self, state: bool) -> None:
        self._initial_state = bool(state)

    def is_dominant_nearend(self) -> bool:
        # Raw detector state; the stat-aware proxy applies to gain-policy decisions only.
        return self._dominant_nearend.is_nearend_state()

    def _ne_state_for_gain_rules(self) -> bool:
        """Near-end presence for gain rules: the detector state, or the stat-aware proxy when enabled."""
        ne = self._dominant_nearend.is_nearend_state()
        if not self._config.stat_aware_ne_proxy_enabled:
            return ne
        if ne:
            return True
        return self._stat_mask_frac > self._config.stat_aware_ne_proxy_threshold


    def get_gain(
        self,
        *,
        aec_state,
        nearend_spectrum: np.ndarray,        # Y² (or E² when usable_linear)
        residual_echo_spectrum: np.ndarray,  # R²
        residual_echo_spectrum_unbounded: np.ndarray,  # R²_unbounded
        comfort_noise_spectrum: np.ndarray,  # CNG power
        render_block: np.ndarray,            # time-domain render (for LowNoiseRender)
        clock_drift: bool,
        stationary_mask: Optional[np.ndarray] = None,  # per-bin bool from band_stationary_mask()
    ) -> np.ndarray:
        """Returns low-band suppression GAIN (amplitude domain, sqrt'd; per-bin)."""
        # Stationary-mask fraction for the stat-aware proxy.
        if stationary_mask is not None:
            sm = np.asarray(stationary_mask, dtype=bool)
            self._stat_mask_frac = float(sm.mean()) if sm.size > 0 else 0.0
        else:
            self._stat_mask_frac = 0.0
        # Dominant near-end update (unbounded echo spectrum when configured).
        echo_for_det = (
            residual_echo_spectrum_unbounded
            if self._config.dominant_nearend_detection.use_unbounded_echo_spectrum
            else residual_echo_spectrum
        )
        self._dominant_nearend.update(
            nearend_spectrum, echo_for_det, comfort_noise_spectrum, self._initial_state
        )
        low_noise_render = self._low_render.detect(render_block)
        self._advance_split_floor_ramp()
        # Split min-gain floor: per-recording far-active latch on instantaneous render energy
        # (not aec_state.active_render(), which needs ~800 ms and would miss the cold start).
        # render_block is int16-scaled; once latched it stays until reset.
        if self._split_floor_enabled and not self._far_active_latched:
            _rb = np.asarray(render_block, dtype=np.float64)
            if float(np.mean(_rb * _rb)) > self._split_floor_latch_power:
                self._far_active_latched = True
        gain = self._lower_band_gain(
            aec_state=aec_state,
            low_noise_render=low_noise_render,
            suppressor_input=nearend_spectrum,
            residual_echo=residual_echo_spectrum,
            comfort_noise=comfort_noise_spectrum,
            clock_drift=clock_drift,
        )
        return gain


    def _lower_band_gain(
        self,
        *,
        aec_state,
        low_noise_render: bool,
        suppressor_input: np.ndarray,
        residual_echo: np.ndarray,
        comfort_noise: np.ndarray,
        clock_drift: bool,
    ) -> np.ndarray:
        max_gain = self._get_max_gain(self._config.floor_first_increase)
        nearend = self._nearend_smoother.average(suppressor_input)
        weighted_residual = np.empty(self._n_bins, dtype=np.float32)
        _weight_echo_for_audibility(self._echo_audibility, residual_echo, weighted_residual, self._sr)
        # Per-bin near/echo weight for the min-gain floor blend (same inputs and sigmoid as
        # the soft_nearend_blend_per_bin gain-rule blend, computed independently of whether that is enabled).
        ne_wb_floor = None
        if self._per_bin_floor_enabled:
            _enr_bin_floor = weighted_residual / (nearend + 1.0)
            ne_wb_floor = self._sigmoid_blend_weight(_enr_bin_floor).astype(np.float32)
        min_gain = self._get_min_gain(
            weighted_residual, self._last_nearend, self._last_echo,
            low_noise_render, aec_state.saturated_echo(), ne_wb_floor,
        )
        G_raw = self._gain_to_no_audible_echo(nearend, weighted_residual, comfort_noise)
        # Clip into [min, max].
        # Elementwise min/max so min_gain wins when it exceeds max_gain (np.clip would invert the priority).
        G = np.maximum(np.minimum(G_raw, max_gain), min_gain)
        # Pre-limiter gain and ENR/EMR for the diagnostic snapshot.
        with np.errstate(divide='ignore', invalid='ignore'):
            _enr_diag = np.divide(weighted_residual, nearend + 1.0)
            _emr_diag = np.divide(weighted_residual, comfort_noise + 1.0)
        _G_pre_hf_lim = G.copy()
        _limit_lf_gains(G, self._sr)
        _hf_lim_applied = (
            (not self._ne_state_for_gain_rules())
            or clock_drift
            or self._config.conservative_hf_suppression
        )
        if _hf_lim_applied:
            _limit_hf_gains(
                self._config.high_frequency_suppression,
                self._config.conservative_hf_suppression,
                G,
                self._sr,
            )
        self._last_gain[:] = G
        self._last_nearend[:] = nearend
        self._last_echo[:] = weighted_residual
        def _sb(a, k): return float(a[k]) if a.size > k else 0.0
        def _reason(raw, mn, mx, final):
            if raw < mn: return 'min'
            if raw > mx: return 'max'
            clipped = min(max(raw, mn), mx)
            return 'lim' if abs(final - clipped) > 1e-6 else 'G'
        # Per-band reason histogram (LF=0:7, MF=7:65, HF=65+).
        _r_min = G_raw < min_gain
        _r_max = G_raw > max_gain
        _G_cl  = np.clip(G_raw, min_gain, max_gain)
        _r_lim = np.abs(G - _G_cl) > 1e-6
        _r_G   = ~_r_min & ~_r_max & ~_r_lim
        def _bf(m, sl): return float(m[sl].mean()) if m[sl].size > 0 else 0.0
        _LF, _MF, _HF = slice(0, 7), slice(7, 65), slice(65, None)
        # Attribution: winning term inside _gain_to_no_audible_echo, min_gain clip and audibility downweight.
        _g_lin = self._last_g_lin
        _g_emr = self._last_g_emr
        _fire = self._last_fire_mask
        _g_emr_wins = _fire & (_g_emr > _g_lin)
        _min_clip_fired = G_raw < min_gain
        # Audibility weight = weighted / residual_echo where residual_echo > 0; < 1 means downweighted.
        _ea_cfg = self._echo_audibility
        _audibility_threshold_hf_eff = float(
            _ea_cfg.floor_power * _ea_cfg.audibility_threshold_hf
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            _aud_ratio = np.where(
                residual_echo > 1e-30,
                weighted_residual / residual_echo,
                1.0,
            )
        _aud_lt1 = (residual_echo > 1e-30) & (_aud_ratio < 0.999)
        with np.errstate(divide='ignore', invalid='ignore'):
            _aud_db_hf = np.where(
                (residual_echo[_HF] > 1e-30) & (weighted_residual[_HF] > 1e-30),
                10.0 * np.log10(
                    np.maximum(weighted_residual[_HF], 1e-30)
                    / np.maximum(residual_echo[_HF], 1e-30)
                ),
                0.0,
            )
        self._last_lower_band_snap = {
            'min_gain_5': _sb(min_gain, 5), 'min_gain_100': _sb(min_gain, 100),
            'max_gain_5': _sb(max_gain, 5), 'max_gain_100': _sb(max_gain, 100),
            'G_pre_clip_5': _sb(G_raw, 5), 'G_pre_clip_100': _sb(G_raw, 100),
            'gain_reason_5': _reason(_sb(G_raw, 5), _sb(min_gain, 5), _sb(max_gain, 5), _sb(G, 5)),
            'gain_reason_100': _reason(_sb(G_raw, 100), _sb(min_gain, 100), _sb(max_gain, 100), _sb(G, 100)),
            'gain_reason_200': _reason(_sb(G_raw, 200), _sb(min_gain, 200), _sb(max_gain, 200), _sb(G, 200)),
            'hf_lim_applied': _hf_lim_applied,
            'reason_G_lf':   _bf(_r_G,   _LF), 'reason_G_mf':   _bf(_r_G,   _MF), 'reason_G_hf':   _bf(_r_G,   _HF),
            'reason_min_lf': _bf(_r_min, _LF), 'reason_min_mf': _bf(_r_min, _MF), 'reason_min_hf': _bf(_r_min, _HF),
            'reason_lim_lf': _bf(_r_lim, _LF), 'reason_lim_mf': _bf(_r_lim, _MF), 'reason_lim_hf': _bf(_r_lim, _HF),
            'r2_lf_mean': float(np.mean(residual_echo[_LF])),
            'r2_mf_mean': float(np.mean(residual_echo[_MF])),
            'r2_hf_mean': float(np.mean(residual_echo[_HF])),
            # Per-band gain distribution after the HF cap.
            'gain_lf_median': float(np.median(G[_LF])),
            'gain_mf_median': float(np.median(G[_MF])),
            'gain_hf_median': float(np.median(G[_HF])),
            'gain_hf_p5':     float(np.percentile(G[_HF], 5)),
            'gain_hf_min':    float(np.min(G[_HF])),
            # Pre-HF-cap snapshot, to separate the HF anchor from R^2 inflation.
            'gain_hf_median_pre_hf_lim': float(np.median(_G_pre_hf_lim[_HF])),
            'gain_hf_p5_pre_hf_lim':     float(np.percentile(_G_pre_hf_lim[_HF], 5)),
            # HF anchor bin and its pre-limiter value.
            'hf_anchor_lgb_bin': int(
                round(
                    self._config.high_frequency_suppression.limiting_gain_freq_hz
                    * 2 * (self._n_bins - 1)
                    / float(self._sr)
                )
            ),
            'hf_anchor_value_pre_hf_lim': float(_G_pre_hf_lim[
                min(
                    self._n_bins - 1,
                    int(round(
                        self._config.high_frequency_suppression.limiting_gain_freq_hz
                        * 2 * (self._n_bins - 1) / float(self._sr)
                    ))
                )
            ]),
            # ENR = R^2 / (Y^2 + 1), EMR = R^2 / (CN + 1), post-audibility weight, pre-clip.
            'enr_hf_median': float(np.median(_enr_diag[_HF])),
            'emr_hf_median': float(np.median(_emr_diag[_HF])),
            'enr_tr_hf_median': float(np.median(
                (self._nearend_enr_tr if self._ne_state_for_gain_rules() else self._normal_enr_tr)[_HF])),
            'emr_tr_hf_median': float(np.median(
                (self._nearend_emr_tr if self._ne_state_for_gain_rules() else self._normal_emr_tr)[_HF])),
            # Per-term gain medians and EMR-win fractions.
            'g_lin_hf_median': float(np.median(_g_lin[_HF])),
            'g_emr_hf_median': float(np.median(_g_emr[_HF])),
            'g_lin_mf_median': float(np.median(_g_lin[_MF])),
            'g_emr_mf_median': float(np.median(_g_emr[_MF])),
            'g_emr_wins_frac_hf': float(_g_emr_wins[_HF].mean()) if _HF.start < self._n_bins else 0.0,
            'g_emr_wins_frac_mf': float(_g_emr_wins[_MF].mean()),
            # Fraction of HF bins where the outer gate (enr > enr_tr and emr > emr_tr) fires.
            'gate_fire_frac_hf': float(_fire[_HF].mean()) if _HF.start < self._n_bins else 0.0,
            # min_gain at HF and how often it clips G_raw.
            'min_gain_hf_median': float(np.median(min_gain[_HF])),
            'min_gain_hf_p95': float(np.percentile(min_gain[_HF], 95)),
            'min_gain_clipped_frac_hf': float(_min_clip_fired[_HF].mean()),
            # low_noise_render selects low_render_limit vs normal_render_limit.
            'low_noise_render_active': bool(low_noise_render),
            # Audibility downweight fraction and mean reduction.
            'audibility_weight_lt1_frac_hf': float(_aud_lt1[_HF].mean()),
            'audibility_threshold_eff_hf': _audibility_threshold_hf_eff,
            'weighted_residual_reduction_db_hf_median': float(np.median(_aud_db_hf)),
        }
        return np.sqrt(np.maximum(G, 0.0)).astype(np.float32)

    def _get_max_gain(self, floor_first_increase: float) -> np.ndarray:
        is_ne = self._ne_state_for_gain_rules()
        inc = self._max_inc_nearend if is_ne else self._max_inc_normal
        max_gain = np.minimum(
            np.maximum(self._last_gain * inc, floor_first_increase), 1.0
        )
        return max_gain.astype(np.float32)

    def _get_min_gain(
        self,
        weighted_residual: np.ndarray,
        last_nearend: np.ndarray,
        last_echo: np.ndarray,
        low_noise_render: bool,
        saturated_echo: bool,
        ne_wb_floor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if saturated_echo:
            return np.zeros(self._n_bins, dtype=np.float32)
        min_echo_power = (
            self._echo_audibility.low_render_limit
            if low_noise_render
            else self._echo_audibility.normal_render_limit
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            min_gain = np.where(
                weighted_residual > 0,
                min_echo_power / np.maximum(weighted_residual, 1e-30),
                1.0,
            )
        np.minimum(min_gain, 1.0, out=min_gain)
        # LF smoothing: low frequencies must not drop too fast after strong near-end.
        if not self._initial_state or self._config.lf_smoothing_during_initial_phase:
            is_ne = self._ne_state_for_gain_rules()
            dec = self._max_dec_lf_nearend if is_ne else self._max_dec_lf_normal
            end = min(self._last_lf_smoothing_band + 1, self._n_bins)
            permanent = self._config.last_permanent_lf_smoothing_band
            for k in range(end):
                if last_nearend[k] > last_echo[k] or k <= permanent:
                    min_gain[k] = max(min_gain[k], self._last_gain[k] * dec)
                    min_gain[k] = min(min_gain[k], 1.0)
        # Split min-gain floor (default on): bounds the deepest suppression so the AEC3 floor
        # (min_echo_power / R^2) cannot collapse to ~0 in double-talk. Far-active uses the gentler
        # floor, pure near-end (far never latched) the stronger one.
        if self._split_floor_enabled:
            if self._far_active_latched:
                # Double-talk lifts the floor toward near-end protection; far-end-only keeps the far-active floor.
                if self._dt_protect_active:
                    # max() keeps the floor at the shallower of the two while the far-active floor
                    # ramps (the DT floor steps instantly), so a DT hop is never suppressed more
                    # than a far-end-only hop.
                    base_floor = max(self._split_floor_dt,
                                     self._split_floor_far_active_live)
                elif ne_wb_floor is not None:
                    # Per-bin blend toward the DT floor by near-dominance; echo-dominant bins keep the far-active floor.
                    base_floor = (
                        ne_wb_floor * self._split_floor_dt
                        + (1.0 - ne_wb_floor) * self._split_floor_far_active_live
                    ).astype(np.float32)
                else:
                    base_floor = self._split_floor_far_active_live
            else:
                base_floor = self._split_floor_far_silent
            np.maximum(min_gain, base_floor, out=min_gain)
            np.minimum(min_gain, 1.0, out=min_gain)
        return min_gain.astype(np.float32)

    def _sigmoid_blend_weight(self, enr):
        """ENR -> near/echo sigmoid blend weight (1 when near-end dominates, 0 when echo dominates).
        Accepts a scalar or an array; shared by the soft-blend and floor-blend consumers."""
        _sig_arg = np.clip(
            (enr - self._soft_ne_blend_enr_thr) / self._soft_ne_blend_softness,
            -50.0, 50.0,
        )
        return 1.0 / (1.0 + np.exp(_sig_arg))

    def _gain_to_no_audible_echo(
        self, nearend: np.ndarray, echo: np.ndarray, masker: np.ndarray
    ) -> np.ndarray:
        is_ne = self._ne_state_for_gain_rules()

        ne_w = 0.0
        if self._soft_ne_blend_enabled:
            ne_lf = float(np.sum(nearend[1:self._dne_lf_end]))
            echo_lf = float(np.sum(echo[1:self._dne_lf_end]))
            enr_lf = echo_lf / (ne_lf + 1.0)
            ne_w = float(self._sigmoid_blend_weight(enr_lf))

        if self._soft_ne_blend_enabled:
            # Blend the near-end and normal tunings by ne_w; per-bin ne_w from per-bin ENR when enabled.
            if self._soft_ne_blend_per_bin:
                _enr_bin = echo / (nearend + 1.0)
                ne_wb = self._sigmoid_blend_weight(_enr_bin).astype(np.float32)
            else:
                ne_wb = ne_w
            enr_tr = (ne_wb * self._nearend_enr_tr
                      + (1.0 - ne_wb) * self._normal_enr_tr).astype(np.float32)
            enr_su = (ne_wb * self._nearend_enr_su
                      + (1.0 - ne_wb) * self._normal_enr_su).astype(np.float32)
            emr_tr = (ne_wb * self._nearend_emr_tr
                      + (1.0 - ne_wb) * self._normal_emr_tr).astype(np.float32)
        else:
            enr_tr = self._nearend_enr_tr if is_ne else self._normal_enr_tr
            enr_su = self._nearend_enr_su if is_ne else self._normal_enr_su
            emr_tr = self._nearend_emr_tr if is_ne else self._normal_emr_tr
        enr = echo / (nearend + 1.0)
        emr = echo / (masker + 1.0)
        g = np.ones(self._n_bins, dtype=np.float32)
        fire = (enr > enr_tr) & (emr > emr_tr)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_lin = (enr_su - enr) / np.maximum(enr_su - enr_tr, 1e-30)
            g_emr = emr_tr / np.maximum(emr, 1e-30)
            g_eff = np.maximum(g_lin, g_emr)
        g = np.where(fire, g_eff, g)
        # Stash the competing terms, fire mask and raw ENR/EMR for the diagnostic snapshot.
        self._last_g_lin = g_lin.astype(np.float32, copy=True)
        self._last_g_emr = g_emr.astype(np.float32, copy=True)
        self._last_fire_mask = fire.copy()
        self._last_enr_raw = enr.astype(np.float32, copy=True)
        self._last_emr_raw = emr.astype(np.float32, copy=True)
        return g.astype(np.float32)
