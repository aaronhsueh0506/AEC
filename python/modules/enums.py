"""AEC enums (mode / preset / filter state).

Extracted from `aec.py` during refactor R.3. Pure value types — no
runtime dependencies on other modules. Imported by ``aec.py`` and by
sibling modules under ``python.modules``.
"""
from enum import Enum


class AecMode(Enum):
    LMS = "lms"         # Time-domain LMS (no normalization, simplest)
    NLMS = "nlms"       # Time-domain NLMS (sample-by-sample)
    FDAF = "fdaf"       # Frequency-domain Adaptive Filter (single block, n_partitions=1)
    PBFDAF = "pbfdaf"   # Partitioned Block FDAF (NLMS adaptation)
    PBFDKF = "pbfdkf"   # Partitioned Block FDKF (Kalman adaptation, recommended)
    SUBBAND = "subband" # Backward compat alias (= PBFDKF)


# Frequency-domain modes (partitioned or single block)
_FREQ_MODES = (AecMode.FDAF, AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)
# Partitioned block modes
_PB_MODES = (AecMode.PBFDAF, AecMode.PBFDKF, AecMode.SUBBAND)


class AecPreset(Enum):
    MILD = "mild"               # Lightest echo suppression, best near-end preservation
    SOFT = "soft"               # Between MILD and BALANCED — gentler RES for music / sensitive NE
    BALANCED = "balanced"       # Balanced echo suppression and near-end quality (default)
    BALANCED_AEC3 = "balanced_aec3"  # v3.21 Phase C.5: BALANCED + AEC3 residual chain
    AGGRESSIVE = "aggressive"   # Stronger echo suppression, moderate near-end impact
    MAXIMUM = "maximum"         # Maximum echo suppression, significant near-end impact


class AecFilterState(Enum):
    """Algorithmic state of the adaptive filter, in priority order."""
    WARMUP         = "warmup"          # Pre-convergence: warmup frames not yet exhausted
    DIVERGED       = "diverged"        # Filter diverged (output >> mic), adaptation unreliable
    EPC_RECOVERY   = "epc_recovery"    # Echo path changed, filter re-converging (P_MAX raised)
    DT_ACTIVE      = "dt_active"       # Double-talk detected, adaptation frozen / slowed
    STATIONARY_FAR = "stationary_far"  # Stationary far-end (white noise / fan), special gating
    CONVERGED      = "converged"       # Stable convergence, good echo cancellation
    CONVERGING     = "converging"      # Adapting, not yet converged
