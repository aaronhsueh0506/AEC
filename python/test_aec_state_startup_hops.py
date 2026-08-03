"""Regression test for the erle_startup_hops/erl_startup_hops sentinel bug.

Bug (Codex review finding): AecStateConfig.erle_startup_hops/erl_startup_hops
used to default to the literal 200, and AecState._resolve_startup_hops read
"field still equals 200" as "caller left it at default -> auto-compute the
grid-correct hop count", overriding ANY caller-supplied value that happened
to equal 200 -- including a caller who deliberately wanted exactly 200
startup hops (e.g. because 200 is the grid-correct value at their own
sample rate/hop size). That request was silently discarded and replaced
with the auto-computed value instead.

Fix: the sentinel is now `None` (AecStateConfig.erle_startup_hops/
erl_startup_hops: Optional[int] = None), which cannot collide with any real,
reachable hop count -- so "not set" and "explicitly set to 200" are now
distinguishable, and 200 is honored like any other explicit value.

Run:
    python3 -m pytest python/test_aec_state_startup_hops.py
    python3 -m unittest python/test_aec_state_startup_hops.py
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.state.aec_state import AecState, AecStateConfig  # noqa: E402


class StartupHopsSentinelTests(unittest.TestCase):

    def test_default_is_none_not_a_reachable_int(self) -> None:
        """The dataclass default must be the sentinel None, not a literal
        int -- otherwise it can collide with a legitimate explicit value."""
        cfg = AecStateConfig()
        self.assertIsNone(cfg.erle_startup_hops)
        self.assertIsNone(cfg.erl_startup_hops)

    def test_unset_auto_resolves_at_legacy_grid(self) -> None:
        """Left at the default (None), AecState computes AEC3's 2.0 s
        startup phase for the configured grid; at hop=160/sr=16000 that is
        200 hops (10 ms/hop)."""
        state = AecState(AecStateConfig(hop_size=160, sample_rate=16000))
        self.assertEqual(state._erle_estimator._startup_hops, 200)
        self.assertEqual(state._erl_estimator._startup_hops, 200)

    def test_unset_auto_resolves_at_a_different_grid(self) -> None:
        """At a grid where 2.0 s is NOT 200 hops (hop=256/sr=16000 -> 16 ms/
        hop -> 125 hops), leaving the field unset must track the grid, not
        freeze at the legacy-grid coincidental value of 200."""
        state = AecState(AecStateConfig(hop_size=256, sample_rate=16000))
        self.assertEqual(state._erle_estimator._startup_hops, 125)
        self.assertEqual(state._erl_estimator._startup_hops, 125)

    def test_explicit_200_honored_verbatim_at_non_default_grid(self) -> None:
        """THE regression case the bug prevented: a caller who explicitly
        sets erle_startup_hops/erl_startup_hops to exactly 200 -- at a grid
        (hop=256/sr=16000) where the auto-computed value is 125, NOT 200 --
        must get exactly 200 honored, not silently replaced by 125.

        Under the old `== 200` sentinel this would have been
        misinterpreted as "left at default" and overridden to 125.
        """
        state = AecState(AecStateConfig(
            hop_size=256, sample_rate=16000,
            erle_startup_hops=200, erl_startup_hops=200,
        ))
        self.assertEqual(state._erle_estimator._startup_hops, 200)
        self.assertEqual(state._erl_estimator._startup_hops, 200)

    def test_explicit_200_honored_verbatim_at_default_grid_too(self) -> None:
        """At the legacy grid, explicit 200 coincides with the auto value --
        still must resolve to 200 (sanity check; not diagnostic on its own
        since auto would also produce 200 here)."""
        state = AecState(AecStateConfig(
            hop_size=160, sample_rate=16000, erle_startup_hops=200,
        ))
        self.assertEqual(state._erle_estimator._startup_hops, 200)

    def test_explicit_non_200_still_honored(self) -> None:
        """Non-default explicit values (already worked pre-fix) keep
        working, and ERLE/ERL resolve independently of each other."""
        state = AecState(AecStateConfig(
            hop_size=256, sample_rate=16000,
            erle_startup_hops=7, erl_startup_hops=9,
        ))
        self.assertEqual(state._erle_estimator._startup_hops, 7)
        self.assertEqual(state._erl_estimator._startup_hops, 9)

    def test_resolve_startup_hops_helper_directly(self) -> None:
        """Unit-level check on the resolver itself: None -> auto, any int
        (including 200) -> verbatim."""
        state = AecState(AecStateConfig(hop_size=256, sample_rate=16000))
        self.assertEqual(state._resolve_startup_hops(None), 125)
        self.assertEqual(state._resolve_startup_hops(200), 200)
        self.assertEqual(state._resolve_startup_hops(0), 0)
        self.assertEqual(state._resolve_startup_hops(1), 1)


if __name__ == '__main__':
    unittest.main()
