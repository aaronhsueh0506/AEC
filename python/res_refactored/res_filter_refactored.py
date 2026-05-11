"""ResFilterRefactored — orchestrator placeholder.

Task B.2 deliverable. Will mirror ResFilter.process() (aec.py:2221-2480)
line-for-line once Task B.3 migrates Modules 1-5.

Until B.3 completes for all five modules and B.4 confirms 800-case byte
equality (atol=1e-6, rtol=1e-5; ≥99.99% (frame,bin) within tolerance),
ResFilterRefactored remains an explicit NotImplementedError so it cannot
accidentally be wired into production behind a half-migrated flag.

The eventual config flag (B.5) is `use_res_refactored: bool = False`.
"""

from __future__ import annotations


class ResFilterRefactored:
    """Drop-in replacement for ResFilter once B.3 + B.4 complete."""

    def __init__(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(
            'ResFilterRefactored is a Phase B Task B.3 deliverable; not migrated yet'
        )

    def process(self, *args, **kwargs):
        raise NotImplementedError
