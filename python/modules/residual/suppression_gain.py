"""SuppressionGain — port of AEC3 suppression_gain.cc.

Mirrors docs/aec3_extracts/src/aec3/suppression_gain.{cc,h}.

Single-channel + single-band port (we run 16 kHz single-band; no
UpperBandsGain). DominantNearendDetector is an inner instance ported
inline here (the legacy port at python/modules/res_filter.py:DominantNearendDetector
will be retired in Phase 5).

Pipeline (per hop):
  1. DominantNearendDetector.update(nearend, residual_echo, comfort_noise).
  2. LowNoiseRenderDetector.detect(render).
  3. LowerBandGain (the 6-stage interior):
     a. GetMaxGain  : limit upward step from last_gain * max_inc_factor.
     b. nearend_smoother (MovingAverageSpectrum over input).
     c. WeightEchoForAudibility (per-band downweight if echo < floor).
     d. GetMinGain : `min_echo_power / weighted_residual_echo`, clamped 1.
     e. GainToNoAudibleEcho : ENR/EMR mask gate (per-bin clip 0..1).
     f. Clamp g into [min_gain, max_gain], LF / HF limiters, sqrt to amplitude.

AEC3 source-file:line references inline.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..freq_utils import hz_to_bin


# --------------------------------------------------------------- AEC3 defaults

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


# AEC3 production defaults pulled from echo_canceller3_config.h:
#   normal_tuning   = (mask_lf=(0.3, 0.4, 0.3),  mask_hf=(0.07, 0.1, 0.3))
#   nearend_tuning  = (mask_lf=(1.09, 1.1, 0.3), mask_hf=(0.1, 0.3, 0.3))
# Tuple order: (enr_transparent, enr_suppress, emr_transparent).
# Earlier port had nearend_tuning hardcoded to (0.4,0.4,0.4) placeholder
# values — 2.7-4× looser than AEC3 → nearend-state gain rules suppressed
# much less aggressively than design intent. Aligned 2026-05-27.
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
    # AEC3 echo_canceller3_config.h:HighFrequencySuppression production defaults:
    #   limiting_gain_band = 16   → 2000 Hz @ kFftLength=128 (125 Hz/bin)
    #   bands_in_limiting_gain = 1 → single resolution cell (1 bin count)
    # Anchor freq: expressed in Hz so it scales correctly to any fft_size.
    # Anchor width: AEC3's `bands_in_limiting_gain` is a BIN COUNT (single
    # spectral cell), not a freq width. At higher fft resolution the same
    # 125 Hz freq band covers 4 bins, and MIN over 4 bins finds deeper
    # spectral nulls than MIN over 1 bin → "HF black block" propagation.
    # Use 1-bin width (= 31.25 Hz @ our fft=512) to match AEC3's strict
    # single-cell semantic. Prior 125 Hz freq-width interpretation made
    # the cap 4× more aggressive at our fft resolution.
    # Restored to AEC3 strict single-cell 2026-05-27.
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
    # Audibility-weighting band boundaries — AEC3 strict.
    # AEC3 `WeightEchoForAudibility` (suppression_gain.cc:108-120) hardcodes
    # bin spans 0-3 / 3-7 / 7-end at fft=128 → 0/375/875 Hz. Scaling by
    # frequency keeps the same physical bands at our fft=512.
    # All three audibility_threshold_* default to 10.0, so band boundaries
    # are behaviour-neutral at default but become load-bearing as soon as
    # per-band thresholds diverge.
    lf_band_end_hz: float = 375.0
    mf_band_end_hz: float = 875.0


@dataclass(frozen=True)
class DominantNearendConfig:
    enr_threshold: float = 0.25
    enr_exit_threshold: float = 10.0
    snr_threshold: float = 30.0
    # hold_duration_ms — minimum NE-state dwell, wall-clock milliseconds.
    # PHYSICAL MEANING: speech-phoneme stability (wall-clock) + downstream
    # NE-vs-non-NE gain-rule coupling.
    # 500 ms = ~2.5 phonemes; empirically co-tuned on our 800-case cohort
    # with the v3.21 SuppressionGain mask shapes. Stored as ms so the
    # derived hop count auto-scales with hop_size at SuppressionGain
    # construction (ms_to_hops(500, 160, 16000) = 50 hops at our default).
    # SCALING: wall-clock derivation captures the phoneme part; minor
    # re-tune may still be needed at very different hop sizes due to
    # behavioural coupling with downstream gain rule.
    # See docs/v3_21_4_time_domain_audit_verdict.md.
    hold_duration_ms: int = 500
    # trigger_threshold — net-positive evidence depth (+1/-1 random walk)
    # before NE state triggers. DIMENSIONLESS sample count.
    # PHYSICAL MEANING: statistical hysteresis depth — NOT wall-clock.
    # Depends on per-sample ENR estimator noise floor (set by our PBFDKF +
    # ENR pipeline, not by AEC3's matched-filter + refined). Wall-clock
    # derivation (blocks_to_hops or ms_to_hops) is WRONG here.
    # Empirical: v3.21.4 V4.1 tested 12 -> 5 (= blocks_to_hops(12,160,16k))
    # and BOTH FS+DT regressed — too few samples to reject estimator noise.
    # SCALING: do NOT auto-derive from hop_size or wall-clock. Re-tune
    # empirically if upstream filter / ENR estimator noise profile changes.
    # 12 samples = 120 ms only at hop=160/sr=16k.
    # See docs/v3_21_4_time_domain_audit_verdict.md.
    trigger_threshold: int = 12
    use_during_initial_phase: bool = True
    use_unbounded_echo_spectrum: bool = True
    # LF-only sum endpoint for nearend detection. AEC3 canonical 2000 Hz
    # (= bin 16 exclusive @ fft=128 = `spectrum.begin()+16` in
    # dominant_nearend_detector.cc:43-44). Covers F0+F1+F2 — speech
    # formant peak band.
    # NOTE on v3.21.2 regression: prior 500→2000 Hz test regressed on
    # 800-case (DT_static deg −0.016, FS_static echo −0.012) but that
    # was BEFORE the bin 0 (DC) skip fix (Phase 2 2026-05-27). DC
    # contamination inflated ne_sum at LF, so widening to 2000 Hz pushed
    # ENR in the wrong direction. With bin 1+ slice (AEC3 parity), the
    # regression mechanism is gone.
    # Default flipped 500 → 2000 Hz 2026-05-27 for AEC3 strict alignment.
    lf_endpoint_hz: float = 2000.0


@dataclass(frozen=True)
class _SubbandRegion:
    low: int = 1
    high: int = 1


@dataclass(frozen=True)
class SubbandNearendConfig:
    """AEC3 SubbandNearendDetection — mirrors
    docs/aec3_extracts/src/aec3/subband_nearend_detector.h:47 +
    .cc:73-83. AEC3 ships no-op defaults (all 1s); production callers
    must override subband bounds + thresholds to make this useful.

    The detector triggers nearend state when, in any channel:
       nearend_power[subband1] < nearend_threshold * nearend_power[subband2]
       AND nearend_power[subband1] > snr_threshold * noise_power[subband1]

    Intuition: subband1 is a "baseline" region (typically LF below pitch);
    subband2 is a "target" region (e.g., MF/HF where speech formants sit).
    When the baseline is much weaker than the target AND above noise,
    that's a speech-like spectral signature with formants present.
    """
    nearend_average_blocks: int = 1
    subband1: _SubbandRegion = field(default_factory=_SubbandRegion)
    subband2: _SubbandRegion = field(default_factory=_SubbandRegion)
    nearend_threshold: float = 1.0
    snr_threshold: float = 1.0


@dataclass
class SuppressorConfig:
    # LF<->HF mask coefficient interpolation boundaries. AEC3 canonical
    # values @ fft_size=128 are last_lf=5 (625 Hz) / first_hf=8 (1000 Hz),
    # meaning F0 lives in the LF-mask zone, F1 lives in the interpolation
    # zone, and F2+ live in the HF-mask zone. The pre-refactor port had
    # bin 5/8 hardcoded which silently landed at 156/250 Hz @ fft_size=512,
    # mis-applying aggressive mask_hf to every formant.
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
    subband_nearend_detection: SubbandNearendConfig = field(
        default_factory=SubbandNearendConfig
    )
    use_subband_nearend_detection: bool = False
    # v3.22 Sprint E.1 — stationary-mask-aware NE-presence proxy at
    # gain-policy consumer sites. See AecConfig.e_stat_aware_ne_proxy_*
    # for full context; propagated from AecConfig at orchestrator init.
    stat_aware_ne_proxy_enabled: bool = False
    stat_aware_ne_proxy_threshold: float = 0.10
    # v3.21.6 Sprint P3 — AEC3 EchoAudibility config wiring. The
    # EchoAudibilityConfig dataclass (declared above this class) already
    # carries audibility_threshold_lf/mf/hf + render-floor knobs consumed
    # by SuppressionGain's ``_weight_echo_for_audibility``. P3 promotes
    # it to a SuppressorConfig field so external code (orchestrator) can
    # override ``use_stationarity_properties``. The orchestrator's
    # stationarity zeroing block reads this field — top-level
    # ``AecConfig.aec3_post_stationarity_zero_enabled`` is a deprecated
    # alias propagated into ``echo_audibility.use_stationarity_properties``
    # at init time.
    echo_audibility: EchoAudibilityConfig = field(default_factory=EchoAudibilityConfig)


# --------------------------------------------------------- helper components

class _MovingAverageSpectrum:
    """Sliding-window mean over per-bin spectra. Mirrors
    moving_average_spectrum.cc's per-bin average over N most recent inputs."""

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
    """Mirrors LowNoiseRenderDetector (suppression_gain.cc:461-478)."""

    def __init__(self) -> None:
        self._average_power = 32768.0 * 32768.0

    def detect(self, render_block: np.ndarray) -> bool:
        if render_block.size == 0:
            return False
        x2 = render_block.astype(np.float64) ** 2
        x2_sum = float(np.sum(x2))
        x2_max = float(np.max(x2))
        threshold = 50.0 * 50.0 * 64.0
        low_noise = self._average_power < threshold and x2_max < 3.0 * self._average_power
        self._average_power = self._average_power * 0.9 + x2_sum * 0.1
        return bool(low_noise)


def _build_gain_params(
    n_bins: int, last_lf_band: int, first_hf_band: int, tuning: SuppressorTuning
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin enr_transparent / enr_suppress / emr_transparent from LF/HF
    tuning via linear interpolation (suppression_gain.cc:487-512)."""
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
    """Mirrors WeightEchoForAudibility (suppression_gain.cc:88-121).

    For each band (LF: 0-2, MF: 3-6, HF: 7-end), bins with echo below
    ``threshold = floor_power * audibility_threshold_*`` get scaled by
    ``max(0, 1 - ((threshold - echo) / (threshold - floor_power))²)``.
    Bins at or above threshold pass through unchanged.
    """

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


def _limit_lf_gains(gain: np.ndarray) -> None:
    """Mirrors LimitLowFrequencyGains (suppression_gain.cc:38-42)."""
    if gain.size >= 3:
        gain[0] = gain[1] = min(gain[1], gain[2])


def _limit_hf_gains(
    cfg: HighFrequencySuppressionConfig,
    conservative_hf: bool,
    gain: np.ndarray,
    sr: int = 16000,
) -> None:
    """Mirrors LimitHighFrequencyGains (suppression_gain.cc:44-85)."""
    n_bins = gain.size
    lgb = hz_to_bin(cfg.limiting_gain_freq_hz, n_bins, sr)
    biq = max(1, hz_to_bin(cfg.limiting_gain_width_hz, n_bins, sr))
    if biq > 0 and lgb + biq <= n_bins:
        min_upper_gain = float(np.min(gain[lgb : lgb + biq]))
        np.minimum(gain[lgb + 1 :], min_upper_gain, out=gain[lgb + 1 :])
    if n_bins >= 2:
        gain[-1] = gain[-2]
    if conservative_hf:
        # AEC3 conservative_hf path. Previous Python port hardcoded bins
        # 20/29 from AEC3 (fft=128, 2500-3625 Hz) without unit conversion,
        # silently landing at 625-906 Hz @ fft=512. Express the AEC3
        # canonical 2500-3625 Hz so the band scales with fft_size.
        # Note: conservative_hf_suppression defaults False, so the flag-OFF
        # path is byte-equal regardless.
        cons_lo = hz_to_bin(2500.0, n_bins, sr)
        cons_hi = hz_to_bin(3625.0, n_bins, sr)
        if n_bins > cons_hi and cons_hi > cons_lo:
            hf_gain_bound = float(np.mean(gain[cons_lo:cons_hi]))
            np.minimum(gain[cons_hi:], hf_gain_bound, out=gain[cons_hi:])


# ----------------------------------------- Dominant nearend (ported here)

class _DominantNearendDetector:
    """AEC3 DominantNearendDetector (single-channel collapse).
    Mirrors docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc."""

    def __init__(
        self,
        cfg: DominantNearendConfig,
        sr: int = 16000,
        hop_size: int = 160,
    ) -> None:
        self._cfg = cfg
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        # Derive wall-clock hops from cfg.hold_duration_ms at construction;
        # value is read every update() call so per-frame cost is O(1).
        from .. import aec3_scale as _aec3_scale
        self._hold_duration_hops = _aec3_scale.ms_to_hops(
            cfg.hold_duration_ms, self._hop_size, self._sr
        )
        self._trigger_counter = 0
        self._hold_counter = 0
        self._nearend_state = False

    def set_config(self, cfg: DominantNearendConfig) -> None:
        self._cfg = cfg
        from .. import aec3_scale as _aec3_scale
        self._hold_duration_hops = _aec3_scale.ms_to_hops(
            cfg.hold_duration_ms, self._hop_size, self._sr
        )

    def is_nearend_state(self) -> bool:
        return self._nearend_state

    def update(
        self,
        nearend_spectrum: np.ndarray,
        residual_echo: np.ndarray,
        comfort_noise: np.ndarray,
        initial_state: bool,
    ) -> None:
        """AEC3 port — dominant_nearend_detector.cc:32-76.

        Three independent blocks per frame (trigger / early-exit / hold-decrement);
        do NOT fold them into an if/elif chain. AEC3-parity highlights:
          - LF energy sum skips bin 0 (DC) — AEC3 `begin()+1` (cc:43).
          - Trigger gated by `(!initial_state || use_during_initial_phase)`
            inline; do NOT early-return, hold counter must still decrement.
          - Exit check has noise floor clause: `echo > exit_thr*ne AND
            echo > snr_thr*noise` (cc:67-68) — Python prior version omitted
            the noise clause.
          - Threshold form is multiplicative (`echo < thr*ne`), not divisive
            with `+1.0` floor (which biased ENR/SNR at low signal levels).
        """
        c = self._cfg
        # LF-only sum endpoint from cfg.lf_endpoint_hz (AEC3 canonical 2000 Hz,
        # bin 16 exclusive at fft=128 — see dominant_nearend_detector.cc:43-44).
        n_bins = nearend_spectrum.size
        lf_end = min(hz_to_bin(c.lf_endpoint_hz, n_bins, self._sr), n_bins)
        # AEC3 cc:43 — `begin()+1` skips DC (bin 0). Python prior version
        # included bin 0 → DC contamination biased echo_sum / ne_sum.
        ne_sum = float(np.sum(nearend_spectrum[1:lf_end]))
        echo_sum = float(np.sum(residual_echo[1:lf_end]))
        noise_sum = float(np.sum(comfort_noise[1:lf_end]))

        # Block 1 — Trigger (AEC3 cc:51-64). Multiplicative form, no `+1.0`
        # division floor. initial_state gates the trigger inline, NOT via
        # early-return (hold counter must still decrement below).
        trigger_active = (
            (not initial_state or c.use_during_initial_phase)
            and echo_sum < c.enr_threshold * ne_sum
            and ne_sum > c.snr_threshold * noise_sum
        )
        if trigger_active:
            self._trigger_counter += 1
            if self._trigger_counter >= c.trigger_threshold:
                self._hold_counter = self._hold_duration_hops
                self._trigger_counter = c.trigger_threshold
        else:
            self._trigger_counter = max(0, self._trigger_counter - 1)

        # Block 2 — Early exit at strong echo (AEC3 cc:67-70). Both clauses
        # required; prior Python version omitted the noise-floor clause.
        if (echo_sum > c.enr_exit_threshold * ne_sum
                and echo_sum > c.snr_threshold * noise_sum):
            self._hold_counter = 0

        # Block 3 — Unconditional hold decrement + state (AEC3 cc:72-74).
        self._hold_counter = max(0, self._hold_counter - 1)
        self._nearend_state = self._hold_counter > 0


class _SubbandNearendDetector:
    """AEC3 SubbandNearendDetector (single-channel collapse).

    Mirrors docs/aec3_extracts/src/aec3/subband_nearend_detector.cc.
    Polymorphic alternative to _DominantNearendDetector — same
    is_nearend_state() interface, different detection algorithm.

    Algorithm:
       smoothed_nearend = MovingAverageSpectrum(nearend_spectrum)
       subband1_pwr = mean(smoothed_nearend[subband1.low : subband1.high+1])
       subband2_pwr = mean(smoothed_nearend[subband2.low : subband2.high+1])
       noise_pwr    = mean(comfort_noise[subband1.low : subband1.high+1])
       nearend_state = (subband1_pwr < threshold * subband2_pwr) AND
                       (subband1_pwr > snr * noise_pwr)

    Detects speech-like spectral signature: when subband1 (typically LF
    baseline) is weaker than subband2 (typically formant region) AND
    above noise floor, NE state triggers. AEC3 cc:50-72 stateless per
    frame (no hold/trigger counters — the smoothing IS the temporal
    integration).
    """

    def __init__(self, cfg: SubbandNearendConfig, n_bins: int, sr: int = 16000) -> None:
        self._cfg = cfg
        self._n_bins = int(n_bins)
        self._sr = int(sr)
        self._smoother = _MovingAverageSpectrum(
            n_bins=self._n_bins, n_blocks=cfg.nearend_average_blocks
        )
        self._nearend_state = False
        self._one_over_subband1_len = 1.0 / max(1, cfg.subband1.high - cfg.subband1.low + 1)
        self._one_over_subband2_len = 1.0 / max(1, cfg.subband2.high - cfg.subband2.low + 1)

    def set_config(self, cfg: SubbandNearendConfig) -> None:
        self._cfg = cfg
        self._smoother.update_memory_length(cfg.nearend_average_blocks)
        self._one_over_subband1_len = 1.0 / max(1, cfg.subband1.high - cfg.subband1.low + 1)
        self._one_over_subband2_len = 1.0 / max(1, cfg.subband2.high - cfg.subband2.low + 1)

    def is_nearend_state(self) -> bool:
        return self._nearend_state

    def update(
        self,
        nearend_spectrum: np.ndarray,
        residual_echo: np.ndarray,  # unused (AEC3 cc:36 marks it /*unused*/)
        comfort_noise: np.ndarray,
        initial_state: bool,        # unused (AEC3 cc:39 marks it /*unused*/)
    ) -> None:
        c = self._cfg
        smoothed = self._smoother.average(nearend_spectrum)
        s1_low, s1_high = c.subband1.low, c.subband1.high
        s2_low, s2_high = c.subband2.low, c.subband2.high
        noise_pwr = float(np.sum(comfort_noise[s1_low:s1_high + 1])) * self._one_over_subband1_len
        ne_s1_pwr = float(np.sum(smoothed[s1_low:s1_high + 1])) * self._one_over_subband1_len
        ne_s2_pwr = float(np.sum(smoothed[s2_low:s2_high + 1])) * self._one_over_subband2_len
        self._nearend_state = (
            ne_s1_pwr < c.nearend_threshold * ne_s2_pwr
            and ne_s1_pwr > c.snr_threshold * noise_pwr
        )


# -------------------------------------------------------- top-level class

class SuppressionGain:
    """Single-channel single-band SuppressionGain."""

    def __init__(self, *, n_bins: int = 257, config: Optional[SuppressorConfig] = None,
                 sr: int = 16000, hop_size: int = 160) -> None:
        self._n_bins = int(n_bins)
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        self._config = config or SuppressorConfig()
        # v3.21.6 Sprint P3 — read echo_audibility from SuppressorConfig
        # (was a hardcoded local default instance pre-P3).
        self._echo_audibility = self._config.echo_audibility
        self._last_gain = np.ones(self._n_bins, dtype=np.float32)
        self._last_nearend = np.zeros(self._n_bins, dtype=np.float32)
        self._last_echo = np.zeros(self._n_bins, dtype=np.float32)
        # v3.22 Sprint E.1 — stationary-mask fraction (0.0..1.0); updated
        # by ``get_gain`` from the orchestrator-supplied per-bin mask.
        # Read only via _ne_state_for_gain_rules; no-op when proxy flag OFF.
        self._stat_mask_frac: float = 0.0
        self._low_render = _LowNoiseRenderDetector()
        self._nearend_smoother = _MovingAverageSpectrum(
            n_bins=self._n_bins, n_blocks=self._config.nearend_average_blocks
        )
        # Polymorphic NearendDetector — mirrors AEC3
        # suppression_gain.cc:373-378 (use_subband_nearend_detection flag
        # picks ONE detector at construction; both expose identical
        # is_nearend_state() interface to the gain compute path).
        if self._config.use_subband_nearend_detection:
            self._dominant_nearend = _SubbandNearendDetector(
                self._config.subband_nearend_detection, n_bins=self._n_bins, sr=self._sr
            )
        else:
            self._dominant_nearend = _DominantNearendDetector(
                self._config.dominant_nearend_detection,
                sr=self._sr,
                hop_size=self._hop_size,
            )
        self._initial_state = True
        # Gain attribution snapshot — populated each frame in _lower_band_gain.
        # Read by orchestrator trace_hf_chain; no audio path effect.
        self._last_lower_band_snap: dict = {}
        # Resolve freq-based config to bin indices once at construction.
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

    def set_initial_state(self, state: bool) -> None:
        self._initial_state = bool(state)

    def is_dominant_nearend(self) -> bool:
        # Public API to orchestrator: returns RAW detector state.
        # The Sprint E.1 stat-aware proxy is INTERNAL to gain-policy
        # decisions and intentionally does NOT propagate here (per directive
        # "proxy 僅限 SuppressionGain gain-policy consumer sites").
        return self._dominant_nearend.is_nearend_state()

    def _ne_state_for_gain_rules(self) -> bool:
        """v3.22 Sprint E.1 — augmented NE-presence used by gain-policy
        consumer sites. Returns ``is_nearend_state()`` unmodified when
        the proxy flag is OFF, preserving byte-equal behavior."""
        ne = self._dominant_nearend.is_nearend_state()
        if not self._config.stat_aware_ne_proxy_enabled:
            return ne
        if ne:
            return True
        return self._stat_mask_frac > self._config.stat_aware_ne_proxy_threshold

    # --- API consumed by orchestrator -----------------------------------------

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
        stationary_mask: Optional[np.ndarray] = None,  # E.1: per-bin bool from band_stationary_mask()
    ) -> np.ndarray:
        """Returns low-band suppression GAIN (amplitude domain, sqrt'd; per-bin)."""
        # Sprint E.1 — capture stationary-mask fraction for the
        # _ne_state_for_gain_rules() proxy. No-op when flag is OFF.
        if stationary_mask is not None:
            sm = np.asarray(stationary_mask, dtype=bool)
            self._stat_mask_frac = float(sm.mean()) if sm.size > 0 else 0.0
        else:
            self._stat_mask_frac = 0.0
        # Dominant nearend update using unbounded-echo spectrum when configured
        # (AEC3 cc:410-413).
        echo_for_det = (
            residual_echo_spectrum_unbounded
            if self._config.dominant_nearend_detection.use_unbounded_echo_spectrum
            else residual_echo_spectrum
        )
        self._dominant_nearend.update(
            nearend_spectrum, echo_for_det, comfort_noise_spectrum, self._initial_state
        )
        low_noise_render = self._low_render.detect(render_block)
        gain = self._lower_band_gain(
            aec_state=aec_state,
            low_noise_render=low_noise_render,
            suppressor_input=nearend_spectrum,
            residual_echo=residual_echo_spectrum,
            comfort_noise=comfort_noise_spectrum,
            clock_drift=clock_drift,
        )
        return gain

    # --- internals ------------------------------------------------------------

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
        # Step 1: max gain envelope from last_gain.
        max_gain = self._get_max_gain(self._config.floor_first_increase)
        # Step 2: smoothed nearend.
        nearend = self._nearend_smoother.average(suppressor_input)
        # Step 3: weighted residual (audibility downweight).
        weighted_residual = np.empty(self._n_bins, dtype=np.float32)
        _weight_echo_for_audibility(self._echo_audibility, residual_echo, weighted_residual, self._sr)
        # Step 4: min gain envelope.
        min_gain = self._get_min_gain(
            weighted_residual, self._last_nearend, self._last_echo,
            low_noise_render, aec_state.saturated_echo(),
        )
        # Step 5: GainToNoAudibleEcho (the heart).
        G_raw = self._gain_to_no_audible_echo(nearend, weighted_residual, comfort_noise)
        # Step 6: clip into [min, max].
        G = np.clip(G_raw, min_gain, max_gain)
        # Snapshot pre-HF-limiter G + ENR/EMR for paint-black diagnostic.
        # No audio effect; consumers via _last_lower_band_snap.
        with np.errstate(divide='ignore', invalid='ignore'):
            _enr_diag = np.divide(weighted_residual, nearend + 1.0)
            _emr_diag = np.divide(weighted_residual, comfort_noise + 1.0)
        _G_pre_hf_lim = G.copy()
        # Step 7: LF + HF limiters.
        _limit_lf_gains(G)
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
        # Stash for next hop.
        self._last_gain[:] = G
        self._last_nearend[:] = nearend
        self._last_echo[:] = weighted_residual
        # D — gain attribution snap (trace_hf_chain reads; no audio effect).
        def _sb(a, k): return float(a[k]) if a.size > k else 0.0
        def _reason(raw, mn, mx, final):
            if raw < mn: return 'min'
            if raw > mx: return 'max'
            clipped = min(max(raw, mn), mx)
            return 'lim' if abs(final - clipped) > 1e-6 else 'G'
        # R0.5 per-band reason histogram (diagnostic; no audio effect).
        # Bands match R0 audit defs: LF=0:7, MF=7:65, HF=65+
        _r_min = G_raw < min_gain
        _r_max = G_raw > max_gain
        _G_cl  = np.clip(G_raw, min_gain, max_gain)
        _r_lim = np.abs(G - _G_cl) > 1e-6
        _r_G   = ~_r_min & ~_r_max & ~_r_lim
        def _bf(m, sl): return float(m[sl].mean()) if m[sl].size > 0 else 0.0
        _LF, _MF, _HF = slice(0, 7), slice(7, 65), slice(65, None)
        self._last_lower_band_snap = {
            'min_gain_5': _sb(min_gain, 5), 'min_gain_100': _sb(min_gain, 100),
            'max_gain_5': _sb(max_gain, 5), 'max_gain_100': _sb(max_gain, 100),
            'G_pre_clip_5': _sb(G_raw, 5), 'G_pre_clip_100': _sb(G_raw, 100),
            'gain_reason_5': _reason(_sb(G_raw, 5), _sb(min_gain, 5), _sb(max_gain, 5), _sb(G, 5)),
            'gain_reason_100': _reason(_sb(G_raw, 100), _sb(min_gain, 100), _sb(max_gain, 100), _sb(G, 100)),
            'gain_reason_200': _reason(_sb(G_raw, 200), _sb(min_gain, 200), _sb(max_gain, 200), _sb(G, 200)),
            'hf_lim_applied': _hf_lim_applied,
            # R0.5 per-band fractions
            'reason_G_lf':   _bf(_r_G,   _LF), 'reason_G_mf':   _bf(_r_G,   _MF), 'reason_G_hf':   _bf(_r_G,   _HF),
            'reason_min_lf': _bf(_r_min, _LF), 'reason_min_mf': _bf(_r_min, _MF), 'reason_min_hf': _bf(_r_min, _HF),
            'reason_lim_lf': _bf(_r_lim, _LF), 'reason_lim_mf': _bf(_r_lim, _MF), 'reason_lim_hf': _bf(_r_lim, _HF),
            # per-band raw R² (before audibility weighting; for regression root-cause)
            'r2_lf_mean': float(np.mean(residual_echo[_LF])),
            'r2_mf_mean': float(np.mean(residual_echo[_MF])),
            'r2_hf_mean': float(np.mean(residual_echo[_HF])),
            # === HF paint-black diagnostic (no audio effect) ===
            # Per-band gain distribution AFTER HF cap propagation.
            'gain_lf_median': float(np.median(G[_LF])),
            'gain_mf_median': float(np.median(G[_MF])),
            'gain_hf_median': float(np.median(G[_HF])),
            'gain_hf_p5':     float(np.percentile(G[_HF], 5)),
            'gain_hf_min':    float(np.min(G[_HF])),
            # Pre-HF-cap snapshot — distinguishes (a) HF cap anchor crushing
            # everything above 2 kHz from (b) underlying R² inflation. Equal
            # pre+post in HF → not the HF anchor. Big gap → anchor is the cause.
            'gain_hf_median_pre_hf_lim': float(np.median(_G_pre_hf_lim[_HF])),
            'gain_hf_p5_pre_hf_lim':     float(np.percentile(_G_pre_hf_lim[_HF], 5)),
            # HF anchor location + value. Used to identify which single bin
            # crushed the rest (lgb = freq → bin at SR/2 / N).
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
            # ENR/EMR HF medians (post-audibility-weight, pre-clip).
            # ENR = R² / (Y² + 1). Large → 'echo dominates' branch of
            # gain_to_no_audible_echo fires → G drops linearly with ENR.
            # EMR = R² / (CN + 1). Large → 'echo above masker' → G ≈ emr_tr/emr.
            'enr_hf_median': float(np.median(_enr_diag[_HF])),
            'emr_hf_median': float(np.median(_emr_diag[_HF])),
            'enr_tr_hf_median': float(np.median(
                (self._nearend_enr_tr if self._ne_state_for_gain_rules() else self._normal_enr_tr)[_HF])),
            'emr_tr_hf_median': float(np.median(
                (self._nearend_emr_tr if self._ne_state_for_gain_rules() else self._normal_emr_tr)[_HF])),
        }
        # Step 8: sqrt to amplitude domain.
        return np.sqrt(np.maximum(G, 0.0)).astype(np.float32)

    def _get_max_gain(self, floor_first_increase: float) -> np.ndarray:
        is_ne = self._ne_state_for_gain_rules()
        inc = (
            self._config.nearend_tuning.max_inc_factor
            if is_ne
            else self._config.normal_tuning.max_inc_factor
        )
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
        # LF smoothing band — make sure low frequencies don't drop too quickly
        # after strong nearend.
        if not self._initial_state or self._config.lf_smoothing_during_initial_phase:
            is_ne = self._ne_state_for_gain_rules()
            dec = (
                self._config.nearend_tuning.max_dec_factor_lf
                if is_ne
                else self._config.normal_tuning.max_dec_factor_lf
            )
            end = min(self._last_lf_smoothing_band + 1, self._n_bins)
            permanent = self._config.last_permanent_lf_smoothing_band
            for k in range(end):
                if last_nearend[k] > last_echo[k] or k <= permanent:
                    min_gain[k] = max(min_gain[k], self._last_gain[k] * dec)
                    min_gain[k] = min(min_gain[k], 1.0)
        return min_gain.astype(np.float32)

    def _gain_to_no_audible_echo(
        self, nearend: np.ndarray, echo: np.ndarray, masker: np.ndarray
    ) -> np.ndarray:
        is_ne = self._ne_state_for_gain_rules()
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
        return g.astype(np.float32)
