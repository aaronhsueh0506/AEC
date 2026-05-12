"""ResFilterRefactored — Phase B orchestrator (Task B.3: Module 1 migrated).

Subclass of legacy `ResFilter` that overrides `_stage_residual_model` to delegate
to the extracted `residual_estimator()` function in this package. All other
stages (Module 2-5) still use the legacy `ResFilter._stage_*` methods unchanged;
they will be migrated incrementally in B.3 (Module 2 onwards) and B.4.

Once all five Modules are migrated and B.4 confirms 800-case byte-equality
(atol=1e-6, rtol=1e-5; ≥99.99% (frame,bin) within tolerance), the legacy
methods will be retired and this orchestrator will mirror `ResFilter.process()`
line-for-line via explicit module calls.

The eventual config flag (B.5) is `use_res_refactored: bool = False`.
"""

from __future__ import annotations

from aec import ResFilter  # python/ is on sys.path; sibling-module import

from .residual_estimator import residual_estimator


class ResFilterRefactored(ResFilter):
    """Drop-in subclass of ResFilter; Module 1 delegated to residual_estimator()."""

    def _stage_residual_model(self, **kwargs):  # noqa: D401
        return residual_estimator(self, **kwargs)
