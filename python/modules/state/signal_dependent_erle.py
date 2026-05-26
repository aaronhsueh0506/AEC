"""SignalDependentErleEstimator — per-section ERLE refinement (single-channel).

Verbatim port of `docs/aec3_extracts/src/aec3/signal_dependent_erle_estimator.{cc,h}`
adapted to single-channel and our `n_freqs=257` (FFT=512, 16 kHz pipeline)
instead of AEC3's fixed 65-bin spectrum.

This module is a SUBSTRATE — activated only when
`config.signal_dependent_erle_sections > 0`. When 0 or unset, ErleEstimator
does NOT instantiate it (preserves v3.21.6 byte-equal).

When `num_sections=1`: degenerate case — should produce ERLE equal to the
average_erle input (single section, no refinement). This is the
**Gate 3 degeneracy proof** for the C-protocol — byte-equal vs baseline.

When `num_sections>1`: divides the partitioned linear filter into sections
(non-linear partitioning where lower sections have finer resolution),
tracks per-section per-subband ERLE refinement.

Six subbands (band-boundaries scaled from AEC3 65-bin → our 257-bin spectrum,
×4 ratio):
- subband 0: bins 4-31    (~125-1000 Hz, LF)
- subband 1: bins 32-63   (~1000-2000 Hz)
- subband 2: bins 64-95   (~2000-3000 Hz)
- subband 3: bins 96-127  (~3000-4000 Hz)
- subband 4: bins 128-191 (~4000-6000 Hz, HF)
- subband 5: bins 192-256 (~6000-8000 Hz, HF)

Reference: AEC3 cc lines 32-95 (DefineFilterSectionSizes / SetSectionsBoundaries),
cc 188-227 (Update), cc 256-353 (UpdateCorrectionFactors), cc 355-412
(ComputeEchoEstimatePerFilterSection), cc 414-425 (ComputeActiveFilterSections).
"""
from typing import Optional

import numpy as np


# AEC3 65-bin band boundaries scaled ×4 to our 257-bin spectrum.
# AEC3 kBandBoundaries = {1, 8, 16, 24, 32, 48, kFftLengthBy2Plus1=65}
_AEC3_BAND_BOUNDARIES_65 = [1, 8, 16, 24, 32, 48, 65]
_OUR_N_FREQS_DEFAULT = 257  # fft_size=512 / 2 + 1
_SCALE = (_OUR_N_FREQS_DEFAULT - 1) // 64  # = 4 (we have 4× bins per AEC3 bin)
_BAND_BOUNDARIES_257 = [b * _SCALE for b in _AEC3_BAND_BOUNDARIES_65[:-1]] + [_OUR_N_FREQS_DEFAULT]
# Final: [4, 32, 64, 96, 128, 192, 257]

K_SUBBANDS = 6
_KX2_BAND_ENERGY_THRESHOLD = 44015068.0  # AEC3 cc:263 constant
_KSMTH_DECREASES = 0.1                    # AEC3 cc:264
_KSMTH_INCREASES = _KSMTH_DECREASES / 2.0  # AEC3 cc:265
_KNUM_UPDATE_THR = 50                      # AEC3 cc:335


def _form_subband_map(n_freqs: int, boundaries) -> np.ndarray:
    """Map bin index → subband index, mirroring AEC3 FormSubbandMap (cc:37-49).

    Returns array shape (n_freqs,) of int subband indices in [0, kSubbands).
    """
    m = np.zeros(n_freqs, dtype=np.int32)
    subband = 1
    for k in range(n_freqs):
        if subband < len(boundaries) and k >= boundaries[subband]:
            subband += 1
        m[k] = subband - 1
    return np.clip(m, 0, K_SUBBANDS - 1)


def _define_filter_section_sizes(delay_headroom_blocks: int,
                                 num_blocks: int,
                                 num_sections: int) -> list:
    """AEC3 cc:56-82 — non-linear partitioning, exponential growth.

    First sections have small block count (high resolution near direct path),
    later sections grow exponentially.
    """
    filter_length_blocks = num_blocks - delay_headroom_blocks
    section_sizes = [0] * num_sections
    remaining_blocks = filter_length_blocks
    remaining_sections = num_sections
    estimator_size = 2
    idx = 0
    while (remaining_sections > 1
           and remaining_blocks > estimator_size * remaining_sections):
        section_sizes[idx] = estimator_size
        remaining_blocks -= estimator_size
        remaining_sections -= 1
        estimator_size *= 2
        idx += 1
    if remaining_sections > 0:
        last_groups_size = remaining_blocks // remaining_sections
        for i in range(idx, num_sections):
            section_sizes[i] = last_groups_size
        section_sizes[num_sections - 1] += (
            remaining_blocks - last_groups_size * remaining_sections
        )
    return section_sizes


def _set_sections_boundaries(delay_headroom_blocks: int,
                             num_blocks: int,
                             num_sections: int) -> list:
    """AEC3 cc:87-118 — derive block boundaries per section."""
    boundaries = [0] * (num_sections + 1)
    if len(boundaries) == 2:
        boundaries[0] = 0
        boundaries[1] = num_blocks
        return boundaries
    section_sizes = _define_filter_section_sizes(
        delay_headroom_blocks, num_blocks, num_sections
    )
    idx = 0
    current_size_block = 0
    boundaries[0] = delay_headroom_blocks
    for k in range(delay_headroom_blocks, num_blocks):
        current_size_block += 1
        if current_size_block >= section_sizes[idx]:
            idx += 1
            if idx == len(section_sizes):
                break
            boundaries[idx] = k + 1
            current_size_block = 0
    boundaries[len(section_sizes)] = num_blocks
    return boundaries


def _set_max_erle_subbands(max_erle_l: float,
                           max_erle_h: float,
                           limit_subband_l: int) -> np.ndarray:
    """AEC3 cc:120-126 — LF subbands use max_erle_l; HF use max_erle_h."""
    arr = np.empty(K_SUBBANDS, dtype=np.float32)
    arr[:limit_subband_l] = max_erle_l
    arr[limit_subband_l:] = max_erle_h
    return arr


class SignalDependentErleEstimator:
    """Single-channel signal-dependent ERLE refinement.

    Activated only when num_sections > 0. num_sections=1 is the degenerate
    case (single section = no refinement, byte-equal to baseline).
    """

    def __init__(
        self,
        *,
        num_sections: int,
        num_blocks: int,
        delay_headroom_blocks: int = 0,
        n_freqs: int = _OUR_N_FREQS_DEFAULT,
        min_erle: float = 1.0,
        max_erle_l: float = 4.0,
        max_erle_h: float = 1.5,
        use_onset_detection: bool = True,
    ) -> None:
        if num_sections < 1:
            raise ValueError(f'num_sections must be >= 1, got {num_sections}')
        if num_sections > num_blocks:
            raise ValueError(
                f'num_sections ({num_sections}) must be <= num_blocks ({num_blocks})'
            )
        self._min_erle = float(min_erle)
        self._num_sections = int(num_sections)
        self._num_blocks = int(num_blocks)
        self._delay_headroom_blocks = int(delay_headroom_blocks)
        self._n_freqs = int(n_freqs)
        self._use_onset_detection = bool(use_onset_detection)
        # Subband boundaries scaled to our n_freqs
        scale = (self._n_freqs - 1) // 64
        self._band_boundaries = (
            [b * scale for b in _AEC3_BAND_BOUNDARIES_65[:-1]]
            + [self._n_freqs]
        )
        self._band_to_subband = _form_subband_map(self._n_freqs, self._band_boundaries)
        limit_subband_l = int(self._band_to_subband[(self._n_freqs - 1) // 2])
        self._max_erle = _set_max_erle_subbands(max_erle_l, max_erle_h, limit_subband_l)
        self._section_boundaries_blocks = _set_sections_boundaries(
            self._delay_headroom_blocks, self._num_blocks, self._num_sections
        )
        # Per-section per-subband state (single capture channel)
        self._erle = np.full(self._n_freqs, self._min_erle, dtype=np.float32)
        self._erle_onset_compensated = np.full(self._n_freqs, self._min_erle, dtype=np.float32)
        self._S2_section_accum = np.zeros((self._num_sections, self._n_freqs), dtype=np.float32)
        self._erle_estimators = np.full((self._num_sections, K_SUBBANDS), self._min_erle, dtype=np.float32)
        self._erle_ref = np.full(K_SUBBANDS, self._min_erle, dtype=np.float32)
        self._correction_factors = np.ones((self._num_sections, K_SUBBANDS), dtype=np.float32)
        self._num_updates = np.zeros(K_SUBBANDS, dtype=np.int32)
        self._n_active_sections = np.zeros(self._n_freqs, dtype=np.int32)

    def reset(self) -> None:
        """Reset state to initial values (AEC3 cc:166-180)."""
        self._erle.fill(self._min_erle)
        self._erle_onset_compensated.fill(self._min_erle)
        self._erle_estimators.fill(self._min_erle)
        self._erle_ref.fill(self._min_erle)
        self._correction_factors.fill(1.0)
        self._num_updates.fill(0)
        self._n_active_sections.fill(0)

    def update(
        self,
        *,
        x2: np.ndarray,                       # render PSD per bin [n_freqs]
        y2: np.ndarray,                       # capture PSD per bin [n_freqs]
        e2: np.ndarray,                       # filter error PSD per bin [n_freqs]
        average_erle: np.ndarray,             # input scalar ERLE [n_freqs]
        average_erle_onset_compensated: np.ndarray,  # onset-comp variant [n_freqs]
        filter_freq_response: np.ndarray,     # |W_partition|² per partition per bin [num_blocks, n_freqs]
        x2_history: np.ndarray,               # X² per partition per bin [num_blocks, n_freqs]
        converged_filter: bool,
    ) -> None:
        """Mirrors AEC3 Update (cc:188-227). num_sections must be > 1 for
        meaningful refinement; if num_sections=1 the output equals input."""
        if self._num_sections > 1:
            self._compute_number_of_active_filter_sections(
                x2_history, filter_freq_response
            )
            self._update_correction_factors(x2, y2, e2, converged_filter)
        # Apply correction (cc:212-226). For num_sections=1, all
        # n_active_sections=0, correction_factor=1.0 → erle = average_erle.
        # AEC3 loops over [0, kFftLengthBy2) — bin kFftLengthBy2 (=last)
        # never written by SDE. Our consumer expects all n_freqs bins
        # populated, so we mirror bin (n_freqs-1) from bin (n_freqs-2)
        # to match our baseline subband_erle convention (subband_erle.py:89).
        n_freqs_minus_1 = self._n_freqs - 1
        for k in range(n_freqs_minus_1):
            section_idx = int(self._n_active_sections[k])
            subband = int(self._band_to_subband[k])
            corr = self._correction_factors[section_idx, subband]
            sb_max = self._max_erle[subband]
            self._erle[k] = float(np.clip(
                average_erle[k] * corr, self._min_erle, sb_max
            ))
            if self._use_onset_detection:
                self._erle_onset_compensated[k] = float(np.clip(
                    average_erle_onset_compensated[k] * corr,
                    self._min_erle, sb_max
                ))
        # Mirror last bin (matches baseline subband_erle convention).
        self._erle[-1] = self._erle[-2]
        if self._use_onset_detection:
            self._erle_onset_compensated[-1] = self._erle_onset_compensated[-2]

    def erle(self, onset_compensated: bool) -> np.ndarray:
        if onset_compensated and self._use_onset_detection:
            return self._erle_onset_compensated
        return self._erle

    # ─── private helpers (mirror AEC3 cc) ──────────────────────────────

    def _compute_number_of_active_filter_sections(
        self,
        x2_history: np.ndarray,           # [num_blocks, n_freqs]
        filter_freq_response: np.ndarray,  # [num_blocks, n_freqs]
    ) -> None:
        """AEC3 cc:242-254."""
        self._compute_echo_estimate_per_filter_section(x2_history, filter_freq_response)
        self._compute_active_filter_sections()

    def _compute_echo_estimate_per_filter_section(
        self,
        x2_history: np.ndarray,
        filter_freq_response: np.ndarray,
    ) -> None:
        """AEC3 cc:355-412 — accumulate X²·H² per section, then cumulative sum."""
        for section in range(self._num_sections):
            block_lo = self._section_boundaries_blocks[section]
            block_hi = min(self._section_boundaries_blocks[section + 1],
                           filter_freq_response.shape[0])
            if block_lo >= block_hi:
                self._S2_section_accum[section, :] = 0.0
                continue
            x2_section = x2_history[block_lo:block_hi].sum(axis=0)
            h2_section = filter_freq_response[block_lo:block_hi].sum(axis=0)
            self._S2_section_accum[section] = x2_section * h2_section
        # Cumulative sum across sections (cc:404-410)
        for section in range(1, self._num_sections):
            self._S2_section_accum[section] += self._S2_section_accum[section - 1]

    def _compute_active_filter_sections(self) -> None:
        """AEC3 cc:414-425 — find min #sections containing 90% energy per bin."""
        self._n_active_sections.fill(0)
        last = self._num_sections - 1
        for k in range(self._n_freqs):
            section = self._num_sections
            target = 0.9 * self._S2_section_accum[last, k]
            while section > 0 and self._S2_section_accum[section - 1, k] >= target:
                section -= 1
                self._n_active_sections[k] = section

    def _update_correction_factors(
        self,
        x2: np.ndarray,
        y2: np.ndarray,
        e2: np.ndarray,
        converged_filter: bool,
    ) -> None:
        """AEC3 cc:256-353 — per-subband EMA of Y²/E² ratio per active section."""
        if not converged_filter:
            return
        def subband_powers(arr):
            return np.array([
                arr[self._band_boundaries[i]:self._band_boundaries[i + 1]].sum()
                for i in range(K_SUBBANDS)
            ], dtype=np.float32)
        X2_sb = subband_powers(x2)
        E2_sb = subband_powers(e2)
        Y2_sb = subband_powers(y2)

        # Per-subband: aggregate active-section index by MIN across the
        # bins in the subband (cc:280-294).
        idx_subbands = np.zeros(K_SUBBANDS, dtype=np.int32)
        for sb in range(K_SUBBANDS):
            lo = self._band_boundaries[sb]
            hi = self._band_boundaries[sb + 1]
            idx_subbands[sb] = int(np.min(self._n_active_sections[lo:hi]))

        new_erle = np.zeros(K_SUBBANDS, dtype=np.float32)
        is_updated = np.zeros(K_SUBBANDS, dtype=bool)
        for sb in range(K_SUBBANDS):
            if X2_sb[sb] > _KX2_BAND_ENERGY_THRESHOLD and E2_sb[sb] > 0:
                new_erle[sb] = Y2_sb[sb] / E2_sb[sb]
                is_updated[sb] = True
                self._num_updates[sb] += 1

        # Update per-section per-subband ERLE estimators (cc:310-321)
        for sb in range(K_SUBBANDS):
            idx = int(idx_subbands[sb])
            curr = self._erle_estimators[idx, sb]
            if new_erle[sb] > curr:
                alpha = _KSMTH_INCREASES
            else:
                alpha = _KSMTH_DECREASES
            if not is_updated[sb]:
                alpha = 0.0
            curr_new = curr + alpha * (new_erle[sb] - curr)
            self._erle_estimators[idx, sb] = float(np.clip(
                curr_new, self._min_erle, self._max_erle[sb]
            ))

        # Update reference (cross-section average) ERLE (cc:323-332)
        for sb in range(K_SUBBANDS):
            curr_ref = self._erle_ref[sb]
            if new_erle[sb] > curr_ref:
                alpha = _KSMTH_INCREASES
            else:
                alpha = _KSMTH_DECREASES
            if not is_updated[sb]:
                alpha = 0.0
            ref_new = curr_ref + alpha * (new_erle[sb] - curr_ref)
            self._erle_ref[sb] = float(np.clip(
                ref_new, self._min_erle, self._max_erle[sb]
            ))

        # Update correction factors (cc:334-350)
        for sb in range(K_SUBBANDS):
            if is_updated[sb] and self._num_updates[sb] > _KNUM_UPDATE_THR:
                idx = int(idx_subbands[sb])
                ref = self._erle_ref[sb]
                if ref > 0.0:
                    new_corr = self._erle_estimators[idx, sb] / ref
                    self._correction_factors[idx, sb] += 0.1 * (
                        new_corr - self._correction_factors[idx, sb]
                    )
