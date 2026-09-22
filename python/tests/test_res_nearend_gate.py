"""Product RES floor policy and exported per-hop decision.

The energy-based near-recent latch is the only source of the DT floor. The
same decision is exported as ``AecResContext.res_floor_protect`` so a fused
consumer can reuse it instead of inventing another detector.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AEC, AecConfig  # noqa: E402


class ResNearendGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aec = AEC(AecConfig(enable_res=False, return_res_context=True))
        self.rng = np.random.default_rng(7)
        self.hop = int(self.aec.config.hop_size)
        for _ in range(200):
            self._hop()

    def _hop(self, *, dominant=None, latch=None, usable=None):
        if dominant is not None:
            self.aec._aec3_sg.is_dominant_nearend = lambda: dominant
        if latch is not None:
            self.aec._ne_recent_frames = 10 ** 6 if latch else 0
            self.aec._ne_above = 0
        if usable is not None:
            self.aec._aec3_state.usable_linear_estimate = lambda: usable
        mic = self.rng.standard_normal(self.hop).astype(np.float32) * 0.2
        ref = self.rng.standard_normal(self.hop).astype(np.float32) * 0.2
        return self.aec.process(mic, ref)

    def test_selected_floor_is_minus_20_db(self) -> None:
        self.assertEqual(AecConfig().min_gain_floor_dt_db, -20.0)

    def test_latch_is_the_only_floor_source_and_context_matches(self) -> None:
        for usable in (True, False):
            out = self._hop(dominant=False, latch=True, usable=usable)
            self.assertTrue(self.aec._aec3_sg._dt_protect_active)
            self.assertTrue(out[1].res_floor_protect)
        out = self._hop(dominant=True, latch=False, usable=True)
        self.assertFalse(self.aec._aec3_sg._dt_protect_active)
        self.assertFalse(out[1].res_floor_protect)

    def test_disable_switch_wins(self) -> None:
        self.aec.config.dt_aware_res_floor_enabled = False
        out = self._hop(dominant=True, latch=True, usable=False)
        self.assertFalse(self.aec._aec3_sg._dt_protect_active)
        self.assertFalse(out[1].res_floor_protect)

    def test_stationarity_zeroing_keeps_existing_policy(self) -> None:
        est = self.aec._aec3_stationarity
        n = len(est.band_stationary_mask())
        est.band_stationary_mask = lambda: np.ones(n, dtype=bool)
        self.aec._aec3_stationarity_active_hops = (
            self.aec._aec3_stationarity_converge_hops + 1)
        out = self._hop(dominant=False, latch=False, usable=True)
        self.assertEqual(np.count_nonzero(np.asarray(out[1].r2) > 0.0), 0)


if __name__ == '__main__':
    unittest.main()
