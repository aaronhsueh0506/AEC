"""P52 Phase B — RES modular refactor (byte-equal output).

Public surface:
  ResState               state container; replaces scattered ResFilter.self fields
  ResidualResult         output of Module 1
  GainResult             output of Modules 2-4 (per-bin amplitude gain + diag idx writes)
  NoiseFloorResult       output of Module 5 (final synth output)

  residual_estimator     Module 1 — residual echo PSD + reverb tail
  gain_computer          Module 2 — softgate_emr + spectral_floor + epc_dt_cap
  spectral_shaper        Module 3 — quiet_mask + 3bin_smooth + hf_cap + pre_temporal trace
  temporal_smoother      Module 4 — EMA + drop/rise rate limit
  noise_floor_cng        Module 5 — adaptive noise PSD + lift + CNG + OLA synth

  ResFilterRefactored    orchestrator; mirrors ResFilter.process line-for-line

All modules are pure: `(input, state) -> (result, new_state)`. Phase B
preserves byte-equality vs the legacy ResFilter; no logic change permitted
(v1.1 §5.5).
"""

from .state import ResState, ResidualResult, GainResult, NoiseFloorResult
from .residual_estimator import residual_estimator
from .gain_computer import gain_computer
from .spectral_shaper import spectral_shaper
from .temporal_smoother import temporal_smoother
from .noise_floor_cng import noise_floor_cng
from .res_filter_refactored import ResFilterRefactored

__all__ = [
    'ResState', 'ResidualResult', 'GainResult', 'NoiseFloorResult',
    'residual_estimator', 'gain_computer', 'spectral_shaper',
    'temporal_smoother', 'noise_floor_cng', 'ResFilterRefactored',
]
