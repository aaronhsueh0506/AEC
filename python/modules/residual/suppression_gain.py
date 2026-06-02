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
    # with the SuppressionGain mask shapes. Stored as ms so the derived
    # hop count auto-scales with hop_size at SuppressionGain construction
    # (ms_to_hops(500, 160, 16000) = 50 hops at our default). SCALING:
    # wall-clock derivation captures the phoneme part; minor re-tune may
    # still be needed at very different hop sizes due to behavioural
    # coupling with downstream gain rule.
    # v3.22 candidate: AEC3 strict 200 ms (= 50 blocks @ 4 ms = 20 hops
    # @ 10 ms) may be re-evaluable once use_aec3_wallclock_gain_ratchet
    # is shipped ON, since the original 500 ms co-tune compensated in
    # part for the 2.5× slower wall-clock gain recovery. Not v3.21.x
    # alignment (the 500 ms is an intentional divergence, not a port
    # bug); listed here so the dependency is visible to future tuners.
    hold_duration_ms: int = 500
    # trigger_threshold — net-positive evidence depth (+1/-1 random walk)
    # before NE state triggers.
    # AEC3 (echo_canceller3_config.h:234) stores this as a BLOCK COUNT
    # at kBlockSize=64 (= 4 ms wall-clock @ 16 kHz); the value 12 there
    # means 48 ms of consecutive +1 evidence. The legacy Python comment
    # claimed this was dimensionless and that 12→5 regressed both FS+DT,
    # but the regression context did NOT also shorten hold_duration
    # proportionally, so trigger-fast + hold-still-2.5×-AEC3 created
    # sticky false-positives. Strict AEC3 wall-clock alignment uses
    # `trigger_threshold_ms = 48` at hop=160/sr=16k → 5 hops (via
    # ms_to_hops), gated by `use_wallclock_trigger_threshold` to keep
    # byte-equal default. `trigger_threshold` (12) remains the legacy
    # hop-count path used when the flag is OFF.
    trigger_threshold: int = 12
    trigger_threshold_ms: int = 48
    use_wallclock_trigger_threshold: bool = False
    use_during_initial_phase: bool = True
    use_unbounded_echo_spectrum: bool = True
    # v3.22 W4 — relax the ENR trigger when the near-end overwhelmingly
    # dominates the noise floor. During DT the NE-inflated error keeps ERLE
    # low so R² (echo_sum) ≈ ne_sum (enr 0.64–1.39) → the standard ENR test
    # (echo < enr_threshold·ne, 0.25) vetoes NE-state even though SNR passes
    # 240×–100000×, so the near-end is wiped. When ON and the near-end is
    # `loud` (ne_sum > loud_nearend_snr_factor · snr_threshold · noise_sum),
    # the ENR trigger uses `loud_nearend_enr_threshold` (0.75) instead. The
    # early_exit (echo > enr_exit_threshold·ne) guard is untouched. FS is
    # self-guarded: there the "near-end" estimate IS residual echo so enr ≈ 1.
    # Default OFF for byte-equal. See AecConfig.dne_loud_nearend_* for spec.
    loud_nearend_enr_relax_enabled: bool = False
    loud_nearend_snr_factor: float = 3.0
    loud_nearend_enr_threshold: float = 0.75
    # LF-only sum endpoint for nearend detection. AEC3 canonical 2000 Hz
    # (= bin 16 exclusive @ fft=128 = `spectrum.begin()+16` in
    # dominant_nearend_detector.cc:43-44). Covers F0+F1+F2 — speech
    # formant peak band.
    # Earlier widening to 2000 Hz regressed bench because of DC
    # contamination inflating ne_sum at LF; with bin 1+ slice (AEC3
    # parity) the regression mechanism is gone, and 2000 Hz matches
    # AEC3 strict alignment.
    lf_endpoint_hz: float = 2000.0


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
    # Stationary-mask-aware NE-presence proxy at gain-policy consumer sites.
    stat_aware_ne_proxy_enabled: bool = False
    stat_aware_ne_proxy_threshold: float = 0.10
    # AEC3 EchoAudibility config wiring. EchoAudibilityConfig carries
    # audibility_threshold_lf/mf/hf + render-floor knobs consumed by
    # SuppressionGain's ``_weight_echo_for_audibility``. The orchestrator
    # overrides ``use_stationarity_properties`` so its zeroing block fires
    # (load-bearing safety net on cohort tail).
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
    """Mirrors LowNoiseRenderDetector (suppression_gain.cc:461-478).

    AEC3 uses ``threshold = 50² × kBlockSize = 160000`` for a 4 ms (64
    sample) block. At our 10 ms hop the same physical RMS produces 2.5×
    more energy per frame; ``use_wallclock_block_energy_threshold=True``
    rescales the threshold via ``block_energy_scale`` so the detector
    fires at the same wall-clock render level as AEC3."""

    def __init__(self, *, hop_samples: int = 64, sample_rate: int = 16000,
                 use_wallclock_block_energy_threshold: bool = False,
                 use_wallclock_iir: bool = False) -> None:
        self._average_power = 32768.0 * 32768.0
        if use_wallclock_block_energy_threshold:
            from .. import aec3_scale as _aec3_scale
            self._threshold = float(
                _aec3_scale.block_energy_scale(50.0 * 50.0 * 64.0, hop_samples)
            )
        else:
            self._threshold = 50.0 * 50.0 * 64.0
        # AEC3 power IIR (average_power = 0.9·avg + 0.1·x2) is per-4ms-block;
        # verbatim per-hop the detector reacts 2.5× slower. Convert the decay
        # when ON; weight = 1 − decay preserves the convex combination.
        if use_wallclock_iir:
            from .. import aec3_scale as _aec3_scale
            self._iir_decay = float(
                _aec3_scale.per_block_growth_to_per_hop(0.9, hop_samples, sample_rate)
            )
            self._iir_weight = 1.0 - self._iir_decay
        else:
            # Literal 0.1 (NOT 1.0-0.9, which is 0.0999…9) to keep the OFF
            # path bit-identical to the original `* 0.9 + x2_sum * 0.1`.
            self._iir_decay = 0.9
            self._iir_weight = 0.1

    def detect(self, render_block: np.ndarray) -> bool:
        if render_block.size == 0:
            return False
        x2 = render_block.astype(np.float64) ** 2
        x2_sum = float(np.sum(x2))
        x2_max = float(np.max(x2))
        low_noise = self._average_power < self._threshold and x2_max < 3.0 * self._average_power
        self._average_power = self._average_power * self._iir_decay + x2_sum * self._iir_weight
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
        self._trigger_threshold_hops = self._derive_trigger_threshold(cfg)
        self._trigger_counter = 0
        self._hold_counter = 0
        self._nearend_state = False
        # Per-frame diagnostic snapshot — populated in update(); read by
        # tracer to attribute DNE blind-spots. No audio path effect.
        self._last_update_snap: dict = {}

    def _derive_trigger_threshold(self, cfg: DominantNearendConfig) -> int:
        """AEC3-strict wall-clock alignment is opt-in via
        ``cfg.use_wallclock_trigger_threshold``. OFF → legacy hop-count
        path (``cfg.trigger_threshold``). ON → ``ms_to_hops`` of the
        wall-clock spec, matching AEC3's `trigger_threshold * kBlockSize / sr`.
        """
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
        trigger_initial_gate = (not initial_state or c.use_during_initial_phase)
        # W4: relax the ENR trigger threshold when the near-end overwhelmingly
        # dominates the noise floor (loud, clearly-present NE). No-op when the
        # flag is OFF (eff_enr_thr == enr_threshold → byte-equal).
        loud_nearend = (
            c.loud_nearend_enr_relax_enabled
            and ne_sum > c.loud_nearend_snr_factor * c.snr_threshold * noise_sum
        )
        eff_enr_thr = c.loud_nearend_enr_threshold if loud_nearend else c.enr_threshold
        trigger_enr_pass = echo_sum < eff_enr_thr * ne_sum
        trigger_snr_pass = ne_sum > c.snr_threshold * noise_sum
        trigger_active = trigger_initial_gate and trigger_enr_pass and trigger_snr_pass
        if trigger_active:
            self._trigger_counter += 1
            if self._trigger_counter >= self._trigger_threshold_hops:
                self._hold_counter = self._hold_duration_hops
                self._trigger_counter = self._trigger_threshold_hops
        else:
            self._trigger_counter = max(0, self._trigger_counter - 1)

        # Block 2 — Early exit at strong echo (AEC3 cc:67-70). Both clauses
        # required; prior Python version omitted the noise-floor clause.
        early_exit = (echo_sum > c.enr_exit_threshold * ne_sum
                      and echo_sum > c.snr_threshold * noise_sum)
        if early_exit:
            self._hold_counter = 0

        # Block 3 — Unconditional hold decrement + state (AEC3 cc:72-74).
        self._hold_counter = max(0, self._hold_counter - 1)
        self._nearend_state = self._hold_counter > 0

        # Diagnostic snapshot (no audio effect). Ratios use +1.0 floor on
        # denominator for stable logging even when ne_sum/noise_sum=0; the
        # actual trigger uses raw multiplicative form (no floor).
        self._last_update_snap = {
            "ne_sum_lf": ne_sum,
            "echo_sum_lf": echo_sum,
            "noise_sum_lf": noise_sum,
            "enr": echo_sum / (ne_sum + 1.0),
            "snr": ne_sum / (noise_sum + 1.0),
            "enr_threshold": float(c.enr_threshold),
            "loud_nearend": bool(loud_nearend),
            "eff_enr_thr": float(eff_enr_thr),
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


# -------------------------------------------------------- top-level class

class SuppressionGain:
    """Single-channel single-band SuppressionGain."""

    def __init__(self, *, n_bins: int = 257, config: Optional[SuppressorConfig] = None,
                 sr: int = 16000, hop_size: int = 160,
                 use_wallclock_block_energy_threshold: bool = False,
                 use_wallclock_gain_ratchet: bool = False,
                 use_wallclock_low_noise_render_iir: bool = False,
                 hf_min_gain_floor_during_dne_enabled: bool = False,
                 hf_min_gain_floor_during_dne_db: float = -15.0,
                 ser_floor_enabled: bool = False,
                 ser_floor_strength: float = 0.5,
                 soft_nearend_blend_enabled: bool = False,
                 soft_nearend_blend_enr_threshold: float = 0.25,
                 soft_nearend_blend_softness: float = 0.25,
                 soft_nearend_blend_per_bin: bool = False,
                 d5_ne_floor_enabled: bool = False,
                 d5_ne_floor_strength: float = 0.3,
                 coh_gain_floor_enabled: bool = False,
                 coh_gain_floor_strength: float = 0.5,
                 split_floor_enabled: bool = True,
                 split_floor_far_active_db: float = -22.0,
                 split_floor_far_silent_db: float = -12.0,
                 split_floor_latch_power: float = 1.0e6,
                 cohxd_floor_release_enabled: bool = False,
                 cohxd_floor_release_db: float = -45.0,
                 cohxd_gamma_lo: float = 0.5,
                 cohxd_gamma_hi: float = 0.85) -> None:
        self._n_bins = int(n_bins)
        self._sr = int(sr)
        self._hop_size = int(hop_size)
        self._config = config or SuppressorConfig()
        # v3.22 split min-gain floor (default ON). Precompute power-domain
        # floors from amplitude-dB; see AecConfig.min_gain_split_floor_* .
        self._split_floor_enabled = bool(split_floor_enabled)
        self._split_floor_far_active = float(10.0 ** (split_floor_far_active_db / 10.0))
        self._split_floor_far_silent = float(10.0 ** (split_floor_far_silent_db / 10.0))
        self._split_floor_latch_power = float(split_floor_latch_power)
        self._far_active_latched = False
        # v3.22 cohxd selective floor release (delay-aligned reference Γ²(X,Y)).
        # Per-bin RELEASE of the split floor on confidently-echo bins; uses the
        # AEC3 R²-adaptive min_gain underneath where Γ² is high. ASYMMETRIC:
        # only lowers the floor, never raises it. See AecConfig.cohxd_*.
        self._cohxd_floor_release_enabled = bool(cohxd_floor_release_enabled)
        self._cohxd_release_floor = float(10.0 ** (cohxd_floor_release_db / 10.0))
        self._cohxd_gamma_lo = float(cohxd_gamma_lo)
        self._cohxd_gamma_hi = float(cohxd_gamma_hi)
        # Set per-hop by get_gain from the orchestrator-supplied Γ²(X,Y).
        self._coh_xy_gamma2: Optional[np.ndarray] = None
        # echo_audibility lives on SuppressorConfig so orchestrator can
        # override use_stationarity_properties.
        self._echo_audibility = self._config.echo_audibility
        self._last_gain = np.ones(self._n_bins, dtype=np.float32)
        self._last_nearend = np.zeros(self._n_bins, dtype=np.float32)
        self._last_echo = np.zeros(self._n_bins, dtype=np.float32)
        # Stationary-mask fraction (0.0..1.0); updated by ``get_gain``
        # from the orchestrator-supplied per-bin mask. Read only via
        # _ne_state_for_gain_rules; no-op when proxy flag OFF.
        self._stat_mask_frac: float = 0.0
        self._low_render = _LowNoiseRenderDetector(
            hop_samples=self._hop_size,
            sample_rate=self._sr,
            use_wallclock_block_energy_threshold=bool(
                use_wallclock_block_energy_threshold
            ),
            use_wallclock_iir=bool(use_wallclock_low_noise_render_iir),
        )
        # AEC3 `nearend_average_blocks` is in 4 ms blocks; wall-clock rescale
        # to our hop_size so the moving-average window physically matches.
        from .. import aec3_scale as _aec3_scale
        _n_smooth_hops = _aec3_scale.blocks_to_hops(
            self._config.nearend_average_blocks, self._hop_size, self._sr
        )
        self._nearend_smoother = _MovingAverageSpectrum(
            n_bins=self._n_bins, n_blocks=_n_smooth_hops
        )
        # NearendDetector — mirrors AEC3 suppression_gain.cc:373-378.
        self._dominant_nearend = _DominantNearendDetector(
            self._config.dominant_nearend_detection,
            sr=self._sr,
            hop_size=self._hop_size,
        )
        self._initial_state = True
        # Gain attribution snapshot — populated each frame in _lower_band_gain.
        # Read by orchestrator trace_hf_chain; no audio path effect.
        self._last_lower_band_snap: dict = {}
        # Kill-stage diag stashes — set in _gain_to_no_audible_echo each hop.
        # Initialised to zeros so first-frame snap access is safe before the
        # first call. No audio effect.
        self._last_g_lin = np.zeros(self._n_bins, dtype=np.float32)
        self._last_g_emr = np.zeros(self._n_bins, dtype=np.float32)
        self._last_fire_mask = np.zeros(self._n_bins, dtype=bool)
        self._last_enr_raw = np.zeros(self._n_bins, dtype=np.float32)
        self._last_emr_raw = np.zeros(self._n_bins, dtype=np.float32)
        # v3.22 candidate (default OFF): HF minimum-gain floor during DNE.
        # See AecConfig.hf_min_gain_floor_during_dne_* for full spec.
        # Precomputed power-domain floor = 10^(threshold_db / 10).
        self._hf_min_gain_floor_during_dne_enabled = bool(
            hf_min_gain_floor_during_dne_enabled
        )
        self._hf_min_gain_floor_during_dne_power = float(
            10.0 ** (float(hf_min_gain_floor_during_dne_db) / 10.0)
        )
        # v3.22 D2: SER-based gain floor (default OFF).
        # See AecConfig.ser_floor_* for full spec.
        self._ser_floor_enabled = bool(ser_floor_enabled)
        self._ser_floor_strength = float(ser_floor_strength)
        # v3.22 D3: Soft nearend tuning blend (default OFF).
        # Replaces binary DNE is_ne switch with sigmoid LF-ENR weight.
        # ne_weight = sigmoid((enr_threshold - enr_lf) / softness)
        # enr_tr = ne_weight * nearend_enr_tr + (1-ne_weight) * normal_enr_tr
        self._soft_ne_blend_enabled = bool(soft_nearend_blend_enabled)
        self._soft_ne_blend_per_bin = bool(soft_nearend_blend_per_bin)
        self._soft_ne_blend_enr_thr = float(soft_nearend_blend_enr_threshold)
        self._soft_ne_blend_softness = max(float(soft_nearend_blend_softness), 1e-6)
        # LF endpoint bin for D3/D5 ENR sum (AEC3-canonical 2000 Hz, same as DNE).
        self._dne_lf_end = min(hz_to_bin(2000.0, self._n_bins, self._sr), self._n_bins)
        # v3.22 D5: ne_weight gain floor (Speex SPP-proxy, default OFF).
        # G_floor = ne_weight × floor_strength; G = max(G_wiener, G_floor).
        # FS (ENR high → ne_weight→0): floor→0, full echo suppression preserved.
        # DT (ENR low → ne_weight→1): floor=floor_strength, nearend protected.
        # Shares ne_weight computation with D3 (same sigmoid, same LF endpoint).
        self._d5_ne_floor_enabled = bool(d5_ne_floor_enabled)
        self._d5_ne_floor_strength = float(d5_ne_floor_strength)
        self._coh_gain_floor_enabled = bool(coh_gain_floor_enabled)
        self._coh_gain_floor_strength = float(coh_gain_floor_strength)
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
        # AEC3 GetMaxGain / GetMinGain LF-smoothing block apply the tuning
        # ratchet multipliers once per LowerBandGain call (= once per 4 ms
        # block in AEC3). At our 10 ms hop, applying the raw constant once
        # per call recovers gain 2.5× slower wall-clock. When the flag is
        # ON, scale each tuning's inc/dec by per_block_growth_to_per_hop
        # so a single per-hop application matches the per-block AEC3 rate.
        # Flag OFF: cached values == raw config values (byte-equal).
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

    def set_initial_state(self, state: bool) -> None:
        self._initial_state = bool(state)

    def is_dominant_nearend(self) -> bool:
        # Public API to orchestrator: returns RAW detector state. The
        # stat-aware proxy is INTERNAL to gain-policy decisions and
        # intentionally does NOT propagate here.
        return self._dominant_nearend.is_nearend_state()

    def _ne_state_for_gain_rules(self) -> bool:
        """Augmented NE-presence used by gain-policy consumer sites.
        Returns ``is_nearend_state()`` unmodified when the proxy flag is
        OFF."""
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
        coh_gamma2: Optional[np.ndarray] = None,  # Layer1: per-bin Γ²_ŶY for coh gain floor
        coh_xy_gamma2: Optional[np.ndarray] = None,  # cohxd: per-bin Γ²(X,Y) for floor release
    ) -> np.ndarray:
        """Returns low-band suppression GAIN (amplitude domain, sqrt'd; per-bin)."""
        # Sprint E.1 — capture stationary-mask fraction for the
        # _ne_state_for_gain_rules() proxy. No-op when flag is OFF.
        if stationary_mask is not None:
            sm = np.asarray(stationary_mask, dtype=bool)
            self._stat_mask_frac = float(sm.mean()) if sm.size > 0 else 0.0
        else:
            self._stat_mask_frac = 0.0
        # cohxd: stash per-bin Γ²(X,Y) for the selective floor release in
        # _get_min_gain (only consumed when cohxd_floor_release_enabled).
        self._coh_xy_gamma2 = coh_xy_gamma2
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
        # Split min-gain floor: per-recording far-active LATCH (applied in
        # _get_min_gain; see AecConfig.min_gain_split_floor_*). Latch fires from
        # the FIRST far-active frame on instantaneous render energy — NOT
        # aec_state.active_render(), which needs ~800ms to assert and would leak
        # the strong NE floor through the FS/DT cold-start where echo is highest.
        # render_block is int16-scaled (×32768): FS far p99 ≥2e7 vs NE far max
        # ≤~7e4, so latch_power separates with ~10× margin. Once latched it
        # stays for the recording (reset per AEC instance), so FS/DT use the
        # gentler floor throughout; only pure-NE (far never active) keeps the
        # strong floor.
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
            coh_gamma2=coh_gamma2,
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
        coh_gamma2: Optional[np.ndarray] = None,
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
        # D2 v2: SER-based gain floor (power domain, per-bin).
        # nearend_true = max(0, Y² − R²) isolates true speech from echo.
        # G_ser_floor = nearend_true / (nearend + 1.0)
        # FS (Y² ≈ R²): nearend_true ≈ 0 → floor ≈ 0 → echo suppression unaffected.
        # DT (speech >> echo): nearend_true ≈ Y² → floor high → nearend preserved.
        if self._ser_floor_enabled:
            nearend_true = np.maximum(0.0, nearend - weighted_residual)
            G_ser_floor = nearend_true / (nearend + 1.0)
            np.maximum(G, G_ser_floor * self._ser_floor_strength, out=G)
        # Layer1: coherence gain floor (AEC2 NLP-inspired).
        # G_floor = sqrt(max(0, 1-Γ²_ŶY))·strength = nearend-amplitude fraction.
        # FS (Ŷ,Y both ∝ X): Γ²→1 → floor→0 → echo suppression unaffected.
        # DT (nearend N ⊥ X adds to Y): Γ² drops → floor rises → nearend preserved.
        # No FS-unconverged false positive: filter error doesn't decorrelate Ŷ,Y.
        if (self._coh_gain_floor_enabled and coh_gamma2 is not None
                and coh_gamma2.shape[0] == G.shape[0]):
            _g2 = np.clip(coh_gamma2.astype(np.float32), 0.0, 1.0)
            G_coh_floor = np.sqrt(1.0 - _g2) * self._coh_gain_floor_strength
            np.maximum(G, G_coh_floor, out=G)
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
        # Kill-stage attribution: which term won inside _gain_to_no_audible_echo,
        # how often min_gain clip / audibility downweight actually fired, what
        # the effective audibility threshold was. Stashed values come from
        # _gain_to_no_audible_echo / _weight_echo_for_audibility (recomputed
        # from residual_echo vs weighted_residual). No audio effect.
        _g_lin = self._last_g_lin
        _g_emr = self._last_g_emr
        _fire = self._last_fire_mask
        _g_emr_wins = _fire & (_g_emr > _g_lin)
        _min_clip_fired = G_raw < min_gain
        # Audibility weight ratio: weight = weighted/residual_echo where
        # residual_echo > 0. Bins with weight < 1 were downweighted by
        # WeightEchoForAudibility. Use a strict-positive guard to keep
        # silent bins from skewing the fraction.
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
        # 10·log10 of the per-bin reduction; clip empty/silent bins.
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
            # === Kill-stage attribution (HF + MF cross-check) ===
            # g_lin / g_emr per-bin medians at HF — which term inside
            # GainToNoAudibleEcho dominates. g_emr_wins_frac tells whether
            # the EMR-bypass clause (emr_tr/emr) is the killer at HF.
            'g_lin_hf_median': float(np.median(_g_lin[_HF])),
            'g_emr_hf_median': float(np.median(_g_emr[_HF])),
            'g_lin_mf_median': float(np.median(_g_lin[_MF])),
            'g_emr_mf_median': float(np.median(_g_emr[_MF])),
            'g_emr_wins_frac_hf': float(_g_emr_wins[_HF].mean()) if _HF.start < self._n_bins else 0.0,
            'g_emr_wins_frac_mf': float(_g_emr_wins[_MF].mean()),
            # Outer gate fire fraction — fraction of HF bins where
            # (enr > enr_tr AND emr > emr_tr); when 0, GainToNoAudibleEcho
            # leaves G=1.0 untouched at HF (cap / smoothing must be culprit).
            'gate_fire_frac_hf': float(_fire[_HF].mean()) if _HF.start < self._n_bins else 0.0,
            # min_gain actual values at HF (NOT post-clip; the protection
            # floor itself) and how often G_raw < min_gain (clip fired).
            'min_gain_hf_median': float(np.median(min_gain[_HF])),
            'min_gain_hf_p95': float(np.percentile(min_gain[_HF], 95)),
            'min_gain_clipped_frac_hf': float(_min_clip_fired[_HF].mean()),
            # low_noise_render bool consumed this hop (selects
            # low_render_limit vs normal_render_limit in _get_min_gain).
            'low_noise_render_active': bool(low_noise_render),
            # Audibility downweight effectiveness — how often weight < 1
            # and the band-mean dB reduction.
            'audibility_weight_lt1_frac_hf': float(_aud_lt1[_HF].mean()),
            'audibility_threshold_eff_hf': _audibility_threshold_hf_eff,
            'weighted_residual_reduction_db_hf_median': float(np.median(_aud_db_hf)),
        }
        # Step 8: sqrt to amplitude domain.
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
            dec = self._max_dec_lf_nearend if is_ne else self._max_dec_lf_normal
            end = min(self._last_lf_smoothing_band + 1, self._n_bins)
            permanent = self._config.last_permanent_lf_smoothing_band
            for k in range(end):
                if last_nearend[k] > last_echo[k] or k <= permanent:
                    min_gain[k] = max(min_gain[k], self._last_gain[k] * dec)
                    min_gain[k] = min(min_gain[k], 1.0)
        # v3.22 candidate (default OFF): HF minimum-gain floor during DNE.
        # When the dominant-nearend detector indicates NE-state, force a
        # power-domain floor on all HF bins (k >= first_hf_band). The
        # AEC3-strict `render_limit / R²` formula collapses min_gain → 0
        # when R² is huge at HF bins where the filter spuriously outputs
        # large S²_linear during NE-only periods, causing audible
        # painted-black HF (formant valleys + fricatives). This explicit
        # power floor caps total HF suppression at the configured dB-floor.
        # Gated on DNE so SG's full dynamic range is preserved when echo
        # is genuinely dominant (FS / DT-echo-loud) — no impact on echo
        # cancellation aggressiveness outside NE-dominant moments.
        if (self._hf_min_gain_floor_during_dne_enabled
                and self._ne_state_for_gain_rules()
                and self._first_hf_band < self._n_bins):
            hf_slice = slice(self._first_hf_band, self._n_bins)
            np.maximum(
                min_gain[hf_slice],
                self._hf_min_gain_floor_during_dne_power,
                out=min_gain[hf_slice],
            )
            np.minimum(min_gain, 1.0, out=min_gain)
        # Split min-gain floor (default ON): cap the deepest suppression to
        # stop the AEC3 floor (min_echo_power / R²) collapsing to ~0 in DT when
        # ERLE contamination spikes R². far-active (FS/DT) uses the gentler
        # floor; pure-NE (far never latched active) uses the stronger floor —
        # which lifts NE nearend at zero echo cost. See AecConfig and the
        # far-active latch in get_gain.
        if self._split_floor_enabled:
            base_floor = (self._split_floor_far_active if self._far_active_latched
                          else self._split_floor_far_silent)
            if (self._cohxd_floor_release_enabled and self._far_active_latched
                    and self._coh_xy_gamma2 is not None):
                # cohxd per-bin floor RELEASE: where the residual is confidently
                # echo by reference coherence Γ²(X,Y), lower the floor toward
                # release_floor so the AEC3 R²-adaptive min_gain (min_echo_power
                # / R², computed above) can suppress deep like AEC3 (DT echo↑).
                # Low Γ² (nearend, uncorrelated with X) keeps base_floor
                # (deg held). Log-domain lerp over [gamma_lo, gamma_hi].
                # ASYMMETRIC: release_floor < base_floor, so this only LOWERS the
                # floor (never raises) → worst case = current behaviour.
                g2 = np.asarray(self._coh_xy_gamma2, dtype=np.float64)
                t = np.clip(
                    (g2 - self._cohxd_gamma_lo)
                    / max(self._cohxd_gamma_hi - self._cohxd_gamma_lo, 1e-6),
                    0.0, 1.0,
                )
                _log_base = np.log(base_floor)
                _log_rel = np.log(self._cohxd_release_floor)
                floor_perbin = np.exp(
                    _log_base + t * (_log_rel - _log_base)
                ).astype(np.float32)
                np.maximum(min_gain, floor_perbin, out=min_gain)
            else:
                np.maximum(min_gain, base_floor, out=min_gain)
            np.minimum(min_gain, 1.0, out=min_gain)
        return min_gain.astype(np.float32)

    def _gain_to_no_audible_echo(
        self, nearend: np.ndarray, echo: np.ndarray, masker: np.ndarray
    ) -> np.ndarray:
        is_ne = self._ne_state_for_gain_rules()

        # Compute sigmoid LF-ENR weight if needed by D3 or D5.
        # ne_weight → 1 when nearend dominates (low ENR), → 0 when echo dominates.
        ne_w = 0.0
        if self._soft_ne_blend_enabled or self._d5_ne_floor_enabled:
            ne_lf = float(np.sum(nearend[1:self._dne_lf_end]))
            echo_lf = float(np.sum(echo[1:self._dne_lf_end]))
            enr_lf = echo_lf / (ne_lf + 1.0)
            _sig_arg = np.clip(
                (enr_lf - self._soft_ne_blend_enr_thr) / self._soft_ne_blend_softness,
                -50.0, 50.0,
            )
            ne_w = float(1.0 / (1.0 + np.exp(_sig_arg)))

        if self._soft_ne_blend_enabled:
            # D3: blend nearend_tuning ↔ normal_tuning via ne_w.
            # P5: per-bin ne_w from per-bin ENR (echo[k]/nearend[k]) →
            # frequency-selective near-end protection. Falls back to the scalar
            # broadband-LF ne_w when off (byte-equal).
            if self._soft_ne_blend_per_bin:
                _enr_bin = echo / (nearend + 1.0)
                _sig_bin = np.clip(
                    (_enr_bin - self._soft_ne_blend_enr_thr)
                    / self._soft_ne_blend_softness,
                    -50.0, 50.0,
                )
                ne_wb = (1.0 / (1.0 + np.exp(_sig_bin))).astype(np.float32)
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
        # D5: ne_weight gain floor (Speex SPP-proxy).
        # G_floor = ne_w × floor_strength; G = max(G_wiener, G_floor).
        # FS (ne_w→0): floor→0, echo suppression unaffected.
        # DT (ne_w→1): floor=floor_strength, nearend preserved.
        if self._d5_ne_floor_enabled and ne_w > 0.0:
            np.maximum(g, ne_w * self._d5_ne_floor_strength, out=g)
        # Kill-stage diag: stash the two competing terms + fire mask + raw
        # ENR/EMR so _lower_band_gain can attribute which term won per bin.
        # No audio effect (read-only consumer via _last_lower_band_snap).
        self._last_g_lin = g_lin.astype(np.float32, copy=True)
        self._last_g_emr = g_emr.astype(np.float32, copy=True)
        self._last_fire_mask = fire.copy()
        self._last_enr_raw = enr.astype(np.float32, copy=True)
        self._last_emr_raw = emr.astype(np.float32, copy=True)
        return g.astype(np.float32)
