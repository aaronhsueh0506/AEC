"""AEC enums (mode / preset / filter state).

Pure value types — no runtime dependencies on other modules. Imported
by ``aec.py`` and by sibling modules under ``modules/``.
"""
from enum import Enum


class AecMode(Enum):
    FDAF = "fdaf"       # Frequency-domain Adaptive Filter (single block, n_partitions=1)
    PBFDAF = "pbfdaf"   # Partitioned Block FDAF (NLMS adaptation)
    PBFDKF = "pbfdkf"   # Partitioned Block FDKF (Kalman adaptation, recommended)
    SUBBAND = "subband"  # alias of PBFDKF


# Frequency-domain modes (partitioned or single block)
_FREQ_MODES = (AecMode.FDAF, AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)
# Partitioned block modes
_PB_MODES = (AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)


class AecPreset(Enum):
    MILD     = "mild"       # near-priority: more near-end kept, more echo leak
    BALANCED   = "balanced"  # default: balanced echo suppression / near-end preservation
    AGGRESSIVE = "aggressive"   # echo-priority: deeper suppression, more near-end loss
    NO_FLOOR   = "no_floor"  # floor off: gain purely R2-derived, deepest suppression


class AecDelayMode(Enum):
    """Far-end to mic alignment mode; integer values match the C ``AecDelayMode``.

    MATCHED: online matched-filter estimate (``delay_num_filters`` in [1, 5], ``fixed_delay_samples == -1``).
    FIXED: caller-measured delay applied by the reference ring (``fixed_delay_samples`` >= 0).
    EXTERNAL_ALIGNED: far-end aligned by the caller, no estimator or ring (``fixed_delay_samples == -1``).
    Illegal mode/field combinations raise in ``AecConfig.__post_init__``.
    """

    MATCHED = 0
    FIXED = 1
    EXTERNAL_ALIGNED = 2

    @classmethod
    def coerce(cls, value: object) -> 'AecDelayMode':
        """Accept the enum, its int value, or its name (any case)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            # bool is an int subclass; True==1 would silently mean FIXED.
            raise ValueError(f"invalid delay_mode: {value!r}")
        if isinstance(value, int):
            return cls(value)
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                raise ValueError(f"invalid delay_mode: {value!r}") from None
        raise ValueError(f"invalid delay_mode: {value!r}")


class AecFilterState(Enum):
    """Algorithmic state of the adaptive filter, in priority order."""
    WARMUP         = "warmup"          # Pre-convergence: warmup frames not yet exhausted
    DIVERGED       = "diverged"        # Filter diverged (output >> mic), adaptation unreliable
    EPC_RECOVERY   = "epc_recovery"    # Echo path changed, filter re-converging (P_MAX raised)
    DT_ACTIVE      = "dt_active"       # Double-talk detected, adaptation frozen / slowed
    STATIONARY_FAR = "stationary_far"  # Stationary far-end (white noise / fan), special gating
    CONVERGED      = "converged"       # Stable convergence, good echo cancellation
    CONVERGING     = "converging"      # Adapting, not yet converged
