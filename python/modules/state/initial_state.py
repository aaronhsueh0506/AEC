"""AecState::InitialState — controls the initial-phase filter parameter set.

Mirrors docs/aec3_extracts/src/aec3/aec_state.cc:327-353 verbatim with
hop-rescaled thresholds.
"""
from ._constants import HOPS_PER_SECOND


class InitialState:
    """Tracks whether the filter is still in the initial-state regime.

    Transitions out of the initial state once enough "strong, not
    saturated" render frames have accumulated:
      - conservative_initial_phase=True : 5 seconds  (AEC3-default
        for cases sensitive to early misadaptation)
      - conservative_initial_phase=False: ``initial_state_seconds``
        from config (AEC3 default 2.5 s).

    ``TransitionTriggered()`` fires for ONE update on the transition
    edge — used by ERLE reset and other consumers that need to clear
    initial-state-only filtering on entry to steady state.
    """

    def __init__(
        self,
        *,
        conservative_initial_phase: bool = False,
        initial_state_seconds: float = 2.5,
    ) -> None:
        self._conservative = bool(conservative_initial_phase)
        self._initial_state_hops = int(initial_state_seconds * HOPS_PER_SECOND)
        # AEC3 5 s conservative threshold (~5*250=1250 blocks) -> our 5*100=500 hops
        self._conservative_hops = 5 * HOPS_PER_SECOND
        self._initial_state = True
        self._transition_triggered = False
        self._strong_not_saturated_render_blocks = 0

    def reset(self) -> None:
        self._initial_state = True
        self._transition_triggered = False
        self._strong_not_saturated_render_blocks = 0

    def update(self, active_render: bool, saturated_capture: bool) -> None:
        if active_render and not saturated_capture:
            self._strong_not_saturated_render_blocks += 1
        prev_initial = self._initial_state
        if self._conservative:
            self._initial_state = (
                self._strong_not_saturated_render_blocks < self._conservative_hops
            )
        else:
            self._initial_state = (
                self._strong_not_saturated_render_blocks < self._initial_state_hops
            )
        self._transition_triggered = (not self._initial_state) and prev_initial

    def initial_state_active(self) -> bool:
        return self._initial_state

    def transition_triggered(self) -> bool:
        return self._transition_triggered
