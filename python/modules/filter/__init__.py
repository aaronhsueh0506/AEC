"""Linear-filter family bridge for the AEC3 state machine.

This package KEEPs the PBFDKF + ShadowFilter + PathChangeRegimeHandler
trio that has shipped GREEN cohort-tail evidence vs AEC3. The classes
themselves stay at ``python/modules/filters.py`` and ``python/modules/epc.py``
through Phase 4; only the AEC3-state-facing bridge is materialised here.

(File-splitting of filters.py / epc.py is deferred to Phase 5 cleanup —
both are under 500 LOC and cohesive; splitting now is churn that doesn't
unblock Phase 3/4 work.)
"""
from .filter_state_bridge import FilterStateBridge, build_filter_state_bridge

__all__ = ["FilterStateBridge", "build_filter_state_bridge"]
