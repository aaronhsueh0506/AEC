"""F2.2 — diverged-streak EMA + P3h reset enable (unit tests).

Validates the small additive change in `aec.py`:

  • new `AecConfig` fields: `use_diverged_streak_ema`,
    `diverged_streak_ema_alpha`, `diverged_streak_ema_threshold`
  • new state `_p3f_diverged_streak_ema` initialised to 0.0
  • EMA update formula (independent of full AEC pipeline)
  • P3h reset gate selector — legacy hard counter when flag OFF,
    EMA gate when flag ON (verified by direct state injection inside
    AEC().process_block flow is too expensive to set up here; we test the
    selector logic in isolation by mirroring the gate predicate).

Integration validation deferred to 800-case bench (different harness).

Run:
    PYTHONPATH=python python3 -m unittest python.test_f2_2_diverged_streak_ema
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import AecConfig, AEC


class ConfigDefaults(unittest.TestCase):
    """Flag must default OFF; EMA params at plan-prescribed values."""

    def test_flag_default_off(self):
        cfg = AecConfig()
        self.assertFalse(cfg.use_diverged_streak_ema)

    def test_alpha_default_095(self):
        cfg = AecConfig()
        self.assertEqual(cfg.diverged_streak_ema_alpha, 0.95)

    def test_threshold_default_07(self):
        cfg = AecConfig()
        self.assertEqual(cfg.diverged_streak_ema_threshold, 0.7)

    def test_overrides_accepted(self):
        cfg = AecConfig(
            use_diverged_streak_ema=True,
            diverged_streak_ema_alpha=0.9,
            diverged_streak_ema_threshold=0.5,
        )
        self.assertTrue(cfg.use_diverged_streak_ema)
        self.assertEqual(cfg.diverged_streak_ema_alpha, 0.9)
        self.assertEqual(cfg.diverged_streak_ema_threshold, 0.5)


class StateInitialisation(unittest.TestCase):
    """`_p3f_diverged_streak_ema` must exist and start at 0.0 on both
    construction and `_reset_filter_derived_state`."""

    def test_init_sets_ema_zero(self):
        aec = AEC(AecConfig(use_diverged_streak_ema=True))
        self.assertTrue(hasattr(aec, '_p3f_diverged_streak_ema'))
        self.assertEqual(aec._p3f_diverged_streak_ema, 0.0)

    def test_reset_clears_ema(self):
        aec = AEC(AecConfig(use_diverged_streak_ema=True))
        # Inject non-zero state then reset.
        aec._p3f_diverged_streak_ema = 0.85
        aec._reset_filter_derived_state(reason='unit_test')
        self.assertEqual(aec._p3f_diverged_streak_ema, 0.0)


class EmaMathStandalone(unittest.TestCase):
    """Replicate the EMA update formula in isolation to verify its
    saturation / decay behaviour matches plan-prescribed TC ≈ 20 frames."""

    @staticmethod
    def _update(ema: float, alpha: float, hit: bool) -> float:
        return alpha * ema + (1.0 - alpha) * (1.0 if hit else 0.0)

    def test_saturates_toward_1_with_continuous_hits(self):
        ema = 0.0
        for _ in range(200):
            ema = self._update(ema, 0.95, True)
        # With α=0.95 the steady-state is 1.0; after 200 frames we should
        # be very close (1 - 0.95^200 ≈ 1.0).
        self.assertGreater(ema, 0.999)

    def test_threshold_crossed_around_24_frames(self):
        """α=0.95 with constant hit=1 reaches 0.7 at n where
        1 - 0.95^n = 0.7 → n ≈ 24."""
        ema = 0.0
        crossing_frame = None
        for n in range(1, 100):
            ema = self._update(ema, 0.95, True)
            if ema > 0.7 and crossing_frame is None:
                crossing_frame = n
                break
        self.assertIsNotNone(crossing_frame)
        self.assertGreaterEqual(crossing_frame, 22)
        self.assertLessEqual(crossing_frame, 26)

    def test_decays_toward_0_with_no_hits(self):
        ema = 0.95
        for _ in range(100):
            ema = self._update(ema, 0.95, False)
        # 0.95 × 0.95^100 ≈ 5.4e-3, well below 0.7 threshold.
        self.assertLess(ema, 1e-2)

    def test_single_frame_dip_does_not_reset(self):
        """The key F2.2 property: a single low-error frame must NOT zero
        the evidence (legacy hard counter does; EMA preserves history)."""
        ema = 0.0
        for _ in range(30):
            ema = self._update(ema, 0.95, True)
        self.assertGreater(ema, 0.7)
        # One frame of no-hit (legacy hard counter would now be 0).
        ema_after_dip = self._update(ema, 0.95, False)
        self.assertGreater(ema_after_dip, 0.7)  # EMA preserves evidence


class GatePredicateLogic(unittest.TestCase):
    """Mirror the in-method gate selector to verify branch semantics.

    The in-method code is:
        if cfg.use_diverged_streak_ema:
            ok = ema > cfg.diverged_streak_ema_threshold
        else:
            ok = streak >= cfg.diverged_reset_streak_frames

    We test that flag OFF uses legacy hard counter and flag ON uses EMA."""

    @staticmethod
    def _streak_ok(cfg, streak_counter: int, streak_ema: float) -> bool:
        if cfg.use_diverged_streak_ema:
            return streak_ema > float(cfg.diverged_streak_ema_threshold)
        return streak_counter >= int(cfg.diverged_reset_streak_frames)

    def test_flag_off_uses_counter(self):
        cfg = AecConfig()
        # Counter below the 50-frame bar → not OK
        self.assertFalse(self._streak_ok(cfg, streak_counter=49, streak_ema=0.95))
        # Counter at/above the bar → OK
        self.assertTrue(self._streak_ok(cfg, streak_counter=50, streak_ema=0.0))

    def test_flag_on_uses_ema(self):
        cfg = AecConfig(use_diverged_streak_ema=True)
        # EMA at 0.5 (below 0.7) → not OK, even with high counter
        self.assertFalse(self._streak_ok(cfg, streak_counter=100, streak_ema=0.5))
        # EMA at 0.8 → OK
        self.assertTrue(self._streak_ok(cfg, streak_counter=0, streak_ema=0.8))


if __name__ == '__main__':
    unittest.main()
