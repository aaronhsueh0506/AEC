"""AEC enums (mode / preset / filter state).

Pure value types — no runtime dependencies on other modules. Imported
by ``aec.py`` and by sibling modules under ``python.modules``.
"""
from enum import Enum


class AecMode(Enum):
    FDAF = "fdaf"       # Frequency-domain Adaptive Filter (single block, n_partitions=1)
    PBFDAF = "pbfdaf"   # Partitioned Block FDAF (NLMS adaptation)
    PBFDKF = "pbfdkf"   # Partitioned Block FDKF (Kalman adaptation, recommended)
    SUBBAND = "subband" # Backward compat alias (= PBFDKF)


# Frequency-domain modes (partitioned or single block)
_FREQ_MODES = (AecMode.FDAF, AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)
# Partitioned block modes
_PB_MODES = (AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)


class AecPreset(Enum):
    MILD     = "mild"       # near-priority: more near-end kept, more echo leak
    BALANCED   = "balanced"     # production: all four ship bars met (AEC3 residual chain)
    AGGRESSIVE = "aggressive"   # echo-priority: deeper suppression, more near-end loss


class AecDelayMode(Enum):
    """How the far-end is aligned to the mic — the single source of truth.

    Integer values are pinned to the C ``AecDelayMode`` enumerators
    (``c_impl/include/aec.h``) so a config can cross the language boundary
    (and land in a descriptor / behaviour hash) unambiguously.

    MATCHED
        The AEC3 matched-filter bank estimates the delay online and a
        reference ring applies it. Requires ``delay_num_filters`` in [1, 5]
        and ``fixed_delay_samples == -1``.
    FIXED
        No estimator. The caller MEASURED the system delay at bring-up and
        the reference ring applies exactly ``fixed_delay_samples`` (>= 0).
    EXTERNAL_ALIGNED
        No estimator and no ring. The caller guarantees the far-end it
        passes is ALREADY aligned to the mic. Requires
        ``fixed_delay_samples == -1``.

    Illegal mode/field combinations raise (``AecConfig.__post_init__``);
    they are never silently normalised, on either side of the port.
    """

    MATCHED = 0
    FIXED = 1
    EXTERNAL_ALIGNED = 2

    @classmethod
    def coerce(cls, value: object) -> 'AecDelayMode':
        """Accept the enum itself, its int value, or its name (any case).

        Mirrors ``AecPreset``'s string tolerance in ``AecConfig.from_preset``
        so config files / CLI flags do not have to import the enum.
        """
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
