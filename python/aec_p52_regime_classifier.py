"""P52 A.0R.3 — AcousticRegimeClassifier.

Classifies a recording's echo-path nonstationarity into a regime label
{stable, mildly_nonstationary, wildly_nonstationary} based on the
ERL-decile-std metric introduced by the post-A.0 post-mortem
(docs/p52_a0_postmortem.md §2 Step 2). Thresholds are anchored to the
cohort distribution measured on the 800-case AEC Challenge corpus.

**Design contract** (anti-loophole, P52 v1.1 §6.4 + Path 3):
- This classifier is **analysis-only**. Its output **must NOT** feed any
  production decision (filter mu gating, RES suppression, shadow copy
  state, AecState flags, P50 preset selection, …).
- The intended consumer is offline tooling: a `tools/research/` driver
  scripts that classify cohort cases for stratified evaluation.
- Anyone wiring this into the live AEC `process()` path is violating
  the v1.1 anti-loophole rules; review must reject such a PR.

**Why a separate module instead of dropping the helper into `aec.py`:**
the classifier consumes a complete recording (mic + lpb wav), not a
per-frame state. Embedding it in `aec.py`'s online state machine would
either require streaming the metric (different definition; not anchored
to post-mortem evidence) or buffering the full recording (memory cost
on long sessions, no production use case). Keeping it in its own module
with a clean function signature makes the analysis-only role obvious.

Thresholds (anchored to docs/p52_a0_postmortem.md §2 Step 2 distribution
on 800 cases, ERL_decile_std, n=660 with sufficient far-active deciles):
   stable                 : value <   9.43 dB  (= cohort p90)
   mildly_nonstationary   : 9.43 ≤ value <  21.04 dB  (= cohort p99)
   wildly_nonstationary   : value ≥  21.04 dB        (top ≤ 1 % outliers)

The target post-mortem case `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`
measured std = 23.39 dB and lands cleanly in wildly_nonstationary under
this scheme (≈ p99.2 in the cohort; rank 655/660).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


# Cohort-anchored thresholds (see module docstring).
_STABLE_MAX_DB = 9.43       # cohort p90 on 800-case ERL_decile_std (n=660)
_MILD_MAX_DB = 21.04        # cohort p99; target case 23.39 ≥ this → wildly


class AcousticRegime(str, Enum):
    """Three-way regime label. String-valued so it serializes cleanly to CSV / JSON."""
    STABLE = 'stable'
    MILDLY_NONSTATIONARY = 'mildly_nonstationary'
    WILDLY_NONSTATIONARY = 'wildly_nonstationary'


@dataclass(frozen=True)
class RegimeClassification:
    """Output of AcousticRegimeClassifier.classify."""

    regime: AcousticRegime
    erl_decile_std_db: float
    erl_decile_ptp_db: float
    deciles_used: int               # number of deciles with sufficient far energy
    erl_per_decile_db: Optional[Sequence[float]] = None  # for plotting / debug


class AcousticRegimeClassifier:
    """Compute the ERL-decile-std regime label for a (mic, lpb) recording.

    Usage:
        clf = AcousticRegimeClassifier()
        result = clf.classify(mic, lpb, sample_rate=16000)
        result.regime  # AcousticRegime.WILDLY_NONSTATIONARY etc.

    The classifier is stateless — instances can be reused freely.
    """

    # FS-active threshold for per-decile ERL: skip deciles where lpb is silent.
    FAR_ACTIVE_PWR_THRESHOLD = 1e-5

    # Minimum number of deciles that must have far-active content for the
    # classification to be considered meaningful. Below this we return STABLE
    # with a flag (deciles_used < 4) so callers can skip / re-bin if desired.
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
        """Compute regime label from a complete (mic, lpb) recording.

        Args:
            mic: mono mic-side waveform, float in [-1, 1]. Length must match lpb.
            lpb: mono loopback (far-end reference) waveform, same length / sr.
            sample_rate: required for documentation only (deciles are length-
                relative).

        Returns:
            RegimeClassification with the regime, the underlying decile-std
            and decile-peak-to-peak values, and the per-decile ERL trace for
            debug / plotting.
        """
        del sample_rate  # not used directly; reserved for API compat
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
