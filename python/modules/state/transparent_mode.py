"""TransparentMode (Legacy variant) — detects "no echo path" cases.

Mirrors docs/aec3_extracts/src/aec3/transparent_mode.{cc,h} Legacy impl.
HMM variant NOT ported (it's behind a WebRTC field trial).

Trigger: linear filter has not converged after a long stretch of strong
render → infer "no acoustic coupling" (e.g. headset, mic far from
speaker) → return Active()=True. Downstream consumers use
``echo_path_gain ≈ 0.01`` so R² shrinks to ~0 and the suppression gain
stays ~1 (pass-through).

Rate notes (AEC3 N * kNumBlocksPerSecond -> our N * HOPS_PER_SECOND):
  - 5  s sane-filter recently-seen window  -> 500  hops
  - 30 s sane-filter active blocks bound   -> 3000 hops
  - 20 s non-converged seq bound           -> 2000 hops
  - 60 s active-non-converged seq bound    -> 6000 hops
  - 6  s strong-render-without-converge    -> 600  hops
"""
from ._constants import HOPS_PER_SECOND


_CONVERGED_INIT = 10_000              # large init counter (AEC3 kBlocksSinceConvergencedFilterInit)
_CONSISTENT_INIT = 10_000             # AEC3 kBlocksSinceConsistentEstimateInit
_SANE_FILTER_DELAY_BLOCKS = 5         # AEC3 "filter_delay_blocks < 5"; blocks not hops -> stays 5
_SANE_FILTER_RECENT_INIT_HOPS = 5 * HOPS_PER_SECOND   # 5 s -> 500 hops
_SANE_FILTER_RECENT_ACTIVE_HOPS = 30 * HOPS_PER_SECOND  # 30 s -> 3000 hops
_NON_CONVERGED_SEQ_BOUND_HOPS = 20 * HOPS_PER_SECOND  # 20 s -> 2000 hops
_ACTIVE_NON_CONVERGED_HOPS = 60 * HOPS_PER_SECOND     # 60 s -> 6000 hops
_STRONG_RENDER_HOPS = 6 * HOPS_PER_SECOND             # 6 s -> 600 hops
_DIVERGED_SEQ_BOUND = 60                              # blocks not hops -> stays 60
_NUM_CONVERGED_BLOCKS_HIGH = 50                       # blocks not hops -> stays 50


class TransparentMode:
    """Legacy TransparentMode classifier. Active() returns True when
    the model believes no echo path is present.

    AecState passes ``filter_delay_blocks``, the 3 filter convergence
    flags, ``all_filters_diverged``, ``active_render``,
    ``saturated_capture`` per frame.
    """

    def __init__(self, *, linear_and_stable_echo_path: bool = False) -> None:
        self._linear_and_stable = bool(linear_and_stable_echo_path)
        self._capture_block_counter = 0
        self._transparency_activated = False
        self._active_blocks_since_sane_filter = _CONSISTENT_INIT
        self._sane_filter_observed = False
        self._finite_erl_recently_detected = False
        self._non_converged_sequence_size = _CONVERGED_INIT
        self._diverged_sequence_size = 0
        self._active_non_converged_sequence_size = 0
        self._num_converged_blocks = 0
        self._recent_convergence_during_activity = False
        self._strong_not_saturated_render_blocks = 0

    def reset(self) -> None:
        self._non_converged_sequence_size = _CONVERGED_INIT
        self._diverged_sequence_size = 0
        self._strong_not_saturated_render_blocks = 0
        if self._linear_and_stable:
            self._recent_convergence_during_activity = False

    def active(self) -> bool:
        return self._transparency_activated

    def update(
        self,
        *,
        filter_delay_blocks: int,
        any_filter_consistent: bool,
        any_filter_converged: bool,
        all_filters_diverged: bool,
        active_render: bool,
        saturated_capture: bool,
    ) -> None:
        self._capture_block_counter += 1
        if active_render and not saturated_capture:
            self._strong_not_saturated_render_blocks += 1

        if any_filter_consistent and filter_delay_blocks < _SANE_FILTER_DELAY_BLOCKS:
            self._sane_filter_observed = True
            self._active_blocks_since_sane_filter = 0
        elif active_render:
            self._active_blocks_since_sane_filter += 1

        if not self._sane_filter_observed:
            sane_recent = self._capture_block_counter <= _SANE_FILTER_RECENT_INIT_HOPS
        else:
            sane_recent = (
                self._active_blocks_since_sane_filter <= _SANE_FILTER_RECENT_ACTIVE_HOPS
            )

        if any_filter_converged:
            self._recent_convergence_during_activity = True
            self._active_non_converged_sequence_size = 0
            self._non_converged_sequence_size = 0
            self._num_converged_blocks += 1
        else:
            self._non_converged_sequence_size += 1
            if self._non_converged_sequence_size > _NON_CONVERGED_SEQ_BOUND_HOPS:
                self._num_converged_blocks = 0
            if active_render:
                self._active_non_converged_sequence_size += 1
                if self._active_non_converged_sequence_size > _ACTIVE_NON_CONVERGED_HOPS:
                    self._recent_convergence_during_activity = False

        if not all_filters_diverged:
            self._diverged_sequence_size = 0
        else:
            self._diverged_sequence_size += 1
            if self._diverged_sequence_size >= _DIVERGED_SEQ_BOUND:
                self._non_converged_sequence_size = _CONVERGED_INIT

        if self._active_non_converged_sequence_size > _ACTIVE_NON_CONVERGED_HOPS:
            self._finite_erl_recently_detected = False
        if self._num_converged_blocks > _NUM_CONVERGED_BLOCKS_HIGH:
            self._finite_erl_recently_detected = True

        if self._finite_erl_recently_detected:
            self._transparency_activated = False
        elif sane_recent and self._recent_convergence_during_activity:
            self._transparency_activated = False
        else:
            self._transparency_activated = (
                self._strong_not_saturated_render_blocks > _STRONG_RENDER_HOPS
            )
