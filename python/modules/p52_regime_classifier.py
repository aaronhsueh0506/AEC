"""AcousticRegimeClassifier: labels a recording's echo-path nonstationarity.

Regime = {stable, mildly_nonstationary, wildly_nonstationary} from the std of the
per-decile ERL (dB): stable below 9.43, mild below 21.04, wild above.

Analysis-only: the output must not feed any production decision. It consumes a
complete (mic, loopback) recording and is meant for offline tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


# Thresholds in dB (std of per-decile ERL).
_STABLE_MAX_DB = 9.43  # stable below this
_MILD_MAX_DB = 21.04  # mild below this, wild above


class AcousticRegime(str, Enum):
    """Regime label; string-valued for CSV / JSON."""
    STABLE = 'stable'
    MILDLY_NONSTATIONARY = 'mildly_nonstationary'
    WILDLY_NONSTATIONARY = 'wildly_nonstationary'


@dataclass(frozen=True)
class RegimeClassification:
    """Output of AcousticRegimeClassifier.classify."""

    regime: AcousticRegime
    erl_decile_std_db: float
    erl_decile_ptp_db: float
    deciles_used: int  # deciles with sufficient far-end energy
    erl_per_decile_db: Optional[Sequence[float]] = None  # per-decile ERL, for plotting


class AcousticRegimeClassifier:
    """ERL-decile-std regime label for a (mic, lpb) recording; stateless and reusable."""

    FAR_ACTIVE_PWR_THRESHOLD = 1e-5

    # Fewer far-active deciles than this returns STABLE with ``deciles_used`` set.
    MIN_DECILES_WITH_FAR = 4

    NUM_DECILES = 10

    def __init__(self,
                 stable_max_db: float = _STABLE_MAX_DB,
                 mild_max_db: float = _MILD_MAX_DB):
        if stable_max_db >= mild_max_db:
            raise ValueError(
                f'stable_max_db ({stable_max_db}) must be < mild_max_db ({mild_max_db})')
        self.stable_max_db = float(stable_max_db)
        self.mild_max_db = float(mild_max_db)

    @staticmethod
    def _label_for(value_db: float, stable_max: float, mild_max: float) -> AcousticRegime:
        if value_db < stable_max:
            return AcousticRegime.STABLE
        if value_db < mild_max:
            return AcousticRegime.MILDLY_NONSTATIONARY
        return AcousticRegime.WILDLY_NONSTATIONARY

    def classify(self, mic: np.ndarray, lpb: np.ndarray,
                 sample_rate: int = 16000) -> RegimeClassification:
        """Classify a complete recording.

        ``mic`` and ``lpb`` are mono float waveforms in [-1, 1]. ``sample_rate`` is
        unused (deciles are length-relative). Returns the regime, decile std / peak-to-peak
        (dB) and the per-decile ERL trace.
        """
        del sample_rate  # unused
        mic = np.asarray(mic, dtype=np.float32)
        lpb = np.asarray(lpb, dtype=np.float32)
        if mic.ndim != 1 or lpb.ndim != 1:
            raise ValueError('mic and lpb must be 1-D mono waveforms')
        n = min(len(mic), len(lpb))
        if n < self.NUM_DECILES:
            return RegimeClassification(
                regime=AcousticRegime.STABLE,
                erl_decile_std_db=0.0,
                erl_decile_ptp_db=0.0,
                deciles_used=0,
                erl_per_decile_db=tuple(),
            )
        mic = mic[:n]
        lpb = lpb[:n]
        erls = []
        for i in range(self.NUM_DECILES):
            s = int(i * n / self.NUM_DECILES)
            e = int((i + 1) * n / self.NUM_DECILES)
            if e <= s:
                continue
            l_pwr = float(np.mean(lpb[s:e] ** 2))
            if l_pwr < self.FAR_ACTIVE_PWR_THRESHOLD:
                continue
            m_pwr = float(np.mean(mic[s:e] ** 2))
            erls.append(10.0 * np.log10((m_pwr + 1e-12) / (l_pwr + 1e-12)))

        if len(erls) < self.MIN_DECILES_WITH_FAR:
            return RegimeClassification(
                regime=AcousticRegime.STABLE,
                erl_decile_std_db=0.0,
                erl_decile_ptp_db=0.0,
                deciles_used=len(erls),
                erl_per_decile_db=tuple(erls),
            )

        std_db = float(np.std(erls))
        ptp_db = float(np.ptp(erls))
        regime = self._label_for(std_db, self.stable_max_db, self.mild_max_db)
        return RegimeClassification(
            regime=regime,
            erl_decile_std_db=std_db,
            erl_decile_ptp_db=ptp_db,
            deciles_used=len(erls),
            erl_per_decile_db=tuple(erls),
        )


__all__ = [
    'AcousticRegime', 'AcousticRegimeClassifier', 'RegimeClassification',
]
