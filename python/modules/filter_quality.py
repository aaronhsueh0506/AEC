"""v3.18 Phase C.B — FilteringQualityAnalyzer (audit-only port of AEC3
filtering_quality_analyzer multi-gate usable_linear_estimate).

Combines 4 gates: startup_timer / reset_timer / convergence_seen / not_transparent.
Output: `usable` (bool) + per-gate diagnostic fields.

Design lock: docs/v3_18_c_b1_filtering_quality_design.md.
"""


class FilteringQualityAnalyzer:
    STARTUP_FRAMES = 40        # 0.4 s @ 10ms frame
    RESET_FRAMES = 20          # 0.2 s
    TRANSPARENT_WINDOW = 100   # 1.0 s

    def __init__(self):
        self.startup_timer = 0
        self.reset_timer = 0
        self.frames_since_far_active = self.TRANSPARENT_WINDOW + 1
        self.convergence_seen = False
        self.usable = False
        self.startup_done = False
        self.reset_done = False
        self.far_active_recent = False

    def update(self, *, far_active: bool, epc_reset_fired: bool,
               convergence_signal: bool) -> None:
        """Update FQA gates.

        `convergence_signal` is the caller's choice of "has the filter
        shown stable behaviour at any point in stream?" — AEC3-aligned
        derives this from FilterAnalyzer.consistent_estimate; legacy
        callers can pass `_filter_converged` as a fallback.
        """
        if far_active or self.startup_timer > 0:
            self.startup_timer += 1
        if epc_reset_fired:
            self.reset_timer = 0
        else:
            self.reset_timer += 1
        if convergence_signal:
            self.convergence_seen = True
        if far_active:
            self.frames_since_far_active = 0
        else:
            self.frames_since_far_active += 1
        self.startup_done = self.startup_timer >= self.STARTUP_FRAMES
        self.reset_done = self.reset_timer >= self.RESET_FRAMES
        self.far_active_recent = (
            self.frames_since_far_active < self.TRANSPARENT_WINDOW)
        self.usable = (
            self.startup_done and self.reset_done
            and self.convergence_seen and self.far_active_recent)

    def reset(self) -> None:
        self.startup_timer = 0
        self.reset_timer = 0
        self.frames_since_far_active = self.TRANSPARENT_WINDOW + 1
        self.convergence_seen = False
        self.usable = False
        self.startup_done = False
        self.reset_done = False
        self.far_active_recent = False
