"""gain_computer — P52 Phase B Module stub.

Task B.2 deliverable: signature only. Logic migration in Task B.3, strict byte-equal
against legacy ResFilter._stage_* per case before commit.

See docs/research_log_p52_phase_b_inventory.md for the mapping of legacy stages
to this module.
"""

from __future__ import annotations

from typing import Tuple

from .state import ResState, ResidualResult, GainResult, NoiseFloorResult


def gain_computer(*args, **kwargs) -> Tuple:  # noqa: D401
    """Module stub. Real signature follows once B.3 migration begins for this module.

    Phase B Task B.2: signature-stub-only; raises NotImplementedError so any
    accidental call surfaces immediately and we cannot accidentally ship a
    half-migrated orchestrator.
    """
    raise NotImplementedError(
        'gain_computer not yet migrated — P52 Phase B Task B.3 pending'
    )
