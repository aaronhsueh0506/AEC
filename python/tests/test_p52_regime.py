"""P52 A.0R.4 — unit tests for PathChangeRegimeHandler + AcousticRegimeClassifier.

Tests each handler action (boost_q / reverse_copy / main_paused) with a
synthetic input that triggers it; verifies the state transition.

Tests the regime classifier against synthetic regimes matching the
post-mortem distribution.

Run:
    python -m pytest python/tests/test_p52_regime.py
"""
from __future__ import annotations

import unittest

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AecConfig, PathChangeRegimeHandler, RegimeHandlerDecision, PBFDKF
from modules.p52_regime_classifier import (
    AcousticRegime, AcousticRegimeClassifier, RegimeClassification,
)


# --- Handler tests ---------------------------------------------------------

def _mk(**over) -> PathChangeRegimeHandler:
    """Construct a handler with default AecConfig + optional overrides."""
    cfg = AecConfig()
    for k, v in over.items():
        setattr(cfg, k, v)
    return PathChangeRegimeHandler(cfg)


def _stable_baseline(h: PathChangeRegimeHandler):
    """Seed _copy_err_baseline EMA with many stable-FS frames so subsequent
    abnormal-error frames register correctly. Returns the seeded handler."""
    for i in range(200):
        h.update(shadow_frame_count=200 + i,
                 far_pwr=1e-2,
                 main_err_smooth=1e-3,
                 shadow_err_smooth=1e-3,  # err_balance ~= 0 → is_stable_fs True
                 epc_active=False,
                 saturation_level=0.0,
                 dt_from_energy=0.0)
    return h


class HandlerActionTests(unittest.TestCase):

    def test_warmup_returns_no_decision(self):
        """First 50 frames: handler returns all-False decision regardless of input."""
        h = _mk()
        for i in range(50):
            d = h.update(shadow_frame_count=i, far_pwr=1e-2,
                         main_err_smooth=1.0, shadow_err_smooth=1e-6,
                         epc_active=False, saturation_level=0.0,
                         dt_from_energy=0.0)
            self.assertFalse(d.boost_q)
            self.assertFalse(d.reverse_copy)
            self.assertFalse(d.pause_main)

    def test_boost_q_fires_after_hysteresis_streak(self):
        """Shadow << main for shadow_copy_hysteresis + HYS_STREAK_MIN frames
        in a row triggers boost_q + pause_main."""
        h = _stable_baseline(_mk(shadow_copy_hysteresis=5))
        fired = False
        boost_frame = None
        for i in range(60):
            d = h.update(shadow_frame_count=400 + i,
                         far_pwr=1e-2,
                         main_err_smooth=1e-3,
                         shadow_err_smooth=1e-7,  # << main, copy gate triggers
                         epc_active=False, saturation_level=0.0,
                         dt_from_energy=0.0)
            if d.boost_q:
                fired = True
                boost_frame = i
                break
        self.assertTrue(fired, 'boost_q never fired on sustained shadow-better streak')
        # By the time boost_q fires, main should be paused
        self.assertTrue(h.main_paused, 'pause not set after boost_q')
        # Hysteresis + streak min — must take at least HYS_STREAK_MIN frames
        self.assertGreaterEqual(boost_frame, h.HYS_STREAK_MIN - 1)

    def test_reverse_copy_fires_when_main_beats_shadow(self):
        """main_err << shadow_err while copy_allowed → decision.reverse_copy."""
        h = _stable_baseline(_mk(shadow_copy_threshold=0.5))
        # Now drive main << shadow
        d = h.update(shadow_frame_count=500,
                     far_pwr=1e-2,
                     main_err_smooth=1e-7,         # main is great
                     shadow_err_smooth=1e-3,        # shadow worse
                     epc_active=False, saturation_level=0.0,
                     dt_from_energy=0.0)
        self.assertTrue(d.reverse_copy, 'reverse_copy did not fire on main-much-better case')

    def test_no_fire_when_dt_active(self):
        """dt_from_energy ≥ 0.3 in default gate_mode='energy' suppresses copy_allowed."""
        h = _stable_baseline(_mk())
        for i in range(60):
            d = h.update(shadow_frame_count=600 + i, far_pwr=1e-2,
                         main_err_smooth=1e-3,
                         shadow_err_smooth=1e-7,
                         epc_active=False, saturation_level=0.0,
                         dt_from_energy=0.5)  # DT active → gate closed
            self.assertFalse(d.boost_q, f'frame {i}: boost_q fired during DT')
            self.assertFalse(d.reverse_copy, f'frame {i}: reverse_copy fired during DT')

    def test_no_fire_when_far_inactive(self):
        """far_pwr below 1e-4 disables all handler actions."""
        h = _stable_baseline(_mk())
        for i in range(60):
            d = h.update(shadow_frame_count=700 + i, far_pwr=1e-6,
                         main_err_smooth=1e-3, shadow_err_smooth=1e-7,
                         epc_active=False, saturation_level=0.0,
                         dt_from_energy=0.0)
            self.assertFalse(d.boost_q)
            self.assertFalse(d.reverse_copy)
            # When far inactive in legacy/coherence branch, counters reset and
            # _main_paused goes False.
        self.assertFalse(h.main_paused)

    def test_pause_releases_after_hangover(self):
        """After pause, _main_paused stays True for epc_hangover frames then releases."""
        h = _stable_baseline(_mk(shadow_copy_hysteresis=5, epc_hangover=10))
        # Trigger pause
        for i in range(60):
            d = h.update(shadow_frame_count=800 + i, far_pwr=1e-2,
                         main_err_smooth=1e-3, shadow_err_smooth=1e-7,
                         epc_active=False, saturation_level=0.0,
                         dt_from_energy=0.0)
            if d.boost_q:
                break
        self.assertTrue(h.main_paused)
        # Now stop the trigger condition (shadow not better)
        for i in range(20):
            h.update(shadow_frame_count=900 + i, far_pwr=1e-6,  # far inactive
                     main_err_smooth=1e-3, shadow_err_smooth=1e-3,
                     epc_active=False, saturation_level=0.0,
                     dt_from_energy=0.0)
        self.assertFalse(h.main_paused, 'pause did not release after far inactive')

    def test_reset_clears_state(self):
        """reset() returns handler to BASELINE_INIT, zero counters, unpaused."""
        h = _stable_baseline(_mk())
        # Trigger something
        for i in range(60):
            h.update(shadow_frame_count=1000 + i, far_pwr=1e-2,
                     main_err_smooth=1e-3, shadow_err_smooth=1e-7,
                     epc_active=False, saturation_level=0.0, dt_from_energy=0.0)
        h.reset()
        self.assertEqual(h.copy_counter, 0)
        self.assertFalse(h.main_paused)
        self.assertAlmostEqual(h.copy_err_baseline, h.BASELINE_INIT)


# --- Classifier tests ------------------------------------------------------

class RegimeClassifierTests(unittest.TestCase):

    def test_stable_synthetic(self):
        """Constant-ratio mic/lpb across the recording → stable."""
        clf = AcousticRegimeClassifier()
        rng = np.random.default_rng(0)
        lpb = rng.standard_normal(160_000).astype(np.float32) * 0.1
        mic = 0.5 * lpb
        r = clf.classify(mic, lpb)
        self.assertEqual(r.regime, AcousticRegime.STABLE)
        self.assertLess(r.erl_decile_std_db, 1.0)

    def test_mildly_synthetic(self):
        """Decile ratios spanning ~30 dB land between p90 and p99 bands."""
        clf = AcousticRegimeClassifier()
        rng = np.random.default_rng(1)
        n = 160_000
        lpb = rng.standard_normal(n).astype(np.float32) * 0.1
        ratios = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8])
        chunks = [r * lpb[i*n//10:(i+1)*n//10] for i, r in enumerate(ratios)]
        mic = np.concatenate(chunks).astype(np.float32)
        out = clf.classify(mic, lpb)
        self.assertEqual(out.regime, AcousticRegime.MILDLY_NONSTATIONARY,
                         f'expected mildly, got {out.regime.value} (std={out.erl_decile_std_db:.2f})')

    def test_wildly_synthetic(self):
        """Alternating ratio 0.01 / 1.0 per decile yields high std."""
        clf = AcousticRegimeClassifier()
        rng = np.random.default_rng(2)
        n = 160_000
        lpb = rng.standard_normal(n).astype(np.float32) * 0.1
        ratios = np.tile([0.001, 1.0], 5)
        chunks = []
        for i, r in enumerate(ratios):
            chunks.append(r * lpb[i*n//10:(i+1)*n//10])
        mic = np.concatenate(chunks).astype(np.float32)
        out = clf.classify(mic, lpb)
        self.assertEqual(out.regime, AcousticRegime.WILDLY_NONSTATIONARY,
                         f'std={out.erl_decile_std_db}')

    def test_threshold_invariant(self):
        """Custom thresholds: stable below; wildly above."""
        clf = AcousticRegimeClassifier(stable_max_db=5.0, mild_max_db=12.0)
        self.assertEqual(clf.stable_max_db, 5.0)
        self.assertEqual(clf.mild_max_db, 12.0)
        with self.assertRaises(ValueError):
            AcousticRegimeClassifier(stable_max_db=20.0, mild_max_db=10.0)

    def test_insufficient_far_returns_stable(self):
        """When fewer than MIN_DECILES_WITH_FAR deciles have far energy,
        return STABLE with deciles_used flag."""
        clf = AcousticRegimeClassifier()
        n = 160_000
        # Far-end silent everywhere
        lpb = np.zeros(n, dtype=np.float32)
        mic = np.random.default_rng(3).standard_normal(n).astype(np.float32) * 0.1
        r = clf.classify(mic, lpb)
        self.assertEqual(r.regime, AcousticRegime.STABLE)
        self.assertLess(r.deciles_used, clf.MIN_DECILES_WITH_FAR)


# --- B1: PBFDKF.reset() P-override cleanup (v3.14 housekeeping) -----------

class PBFDKFResetTests(unittest.TestCase):
    """B1 fix verification: reset() must unconditionally clear dynamic
    P-override attrs regardless of countdown state (B1, LOW-MED severity).

    The _p_max_override / _p_max_override_frames / _p_floor_beta /
    _p_floor_beta_frames attrs are dynamically injected as instance attributes
    when an EPC or regime event arms the P-override.  The countdown logic
    decrements _frames and deletes it (plus resets base attr to default) when
    the countdown expires.  reset() must clear all four attrs unconditionally
    so that a reset mid-countdown leaves no stale state that a subsequent
    process() would inherit.
    """

    def _make_filter(self):
        """Minimal PBFDKF: block_size=64 (hop=32), 4 partitions."""
        return PBFDKF(block_size=64, n_partitions=4)

    def test_reset_on_fresh_filter_is_clean(self):
        """reset() on a just-constructed filter: attrs absent before and after."""
        filt = self._make_filter()
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            self.assertFalse(hasattr(filt, attr), f'{attr} present on fresh filter')
        filt.reset()
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            self.assertFalse(hasattr(filt, attr), f'{attr} present after reset on fresh filter')

    def test_reset_during_active_p_override_clears_all_four_attrs(self):
        """B1 core: reset() called mid-countdown removes all 4 dynamic attrs."""
        filt = self._make_filter()
        # Arm a P-override exactly as the EPC trigger does (see aec.py lines 5633-5635)
        filt._p_max_override = 1.0
        filt._p_max_override_frames = 30
        filt._p_floor_beta = 1.0
        filt._p_floor_beta_frames = 30
        # Sanity: all four present
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            self.assertTrue(hasattr(filt, attr), f'{attr} not set before reset')
        # reset() must clear unconditionally
        filt.reset()
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            self.assertFalse(hasattr(filt, attr),
                             f'B1: {attr} still present after reset during active countdown')

    def test_reset_after_countdown_expires_base_attr_also_cleared(self):
        """When countdown expires, base attr is set to default (0.5) but still
        exists as an instance attribute; reset() must remove it."""
        filt = self._make_filter()
        # Simulate post-countdown state: _frames deleted, base attr at default
        filt._p_max_override = 0.5   # countdown expired → reset to default value
        # _p_max_override_frames intentionally absent (expired countdown)
        self.assertTrue(hasattr(filt, '_p_max_override'))
        self.assertFalse(hasattr(filt, '_p_max_override_frames'))
        filt.reset()
        self.assertFalse(hasattr(filt, '_p_max_override'),
                         'B1: _p_max_override still present after reset (post-countdown residue)')

    def test_second_reset_is_idempotent(self):
        """Two consecutive reset() calls must not raise and must leave no attrs."""
        filt = self._make_filter()
        filt._p_max_override = 1.0
        filt._p_max_override_frames = 15
        filt._p_floor_beta = 1.0
        filt._p_floor_beta_frames = 15
        filt.reset()
        filt.reset()  # second reset — must not raise AttributeError
        for attr in ('_p_max_override', '_p_max_override_frames',
                     '_p_floor_beta', '_p_floor_beta_frames'):
            self.assertFalse(hasattr(filt, attr),
                             f'{attr} present after double reset')

    def test_getattr_fallback_works_after_reset(self):
        """After reset() removes dynamic attrs, getattr(..., default) returns default."""
        filt = self._make_filter()
        filt._p_max_override = 1.0
        filt._p_max_override_frames = 20
        filt.reset()
        p_max = getattr(filt, '_p_max_override', 0.5)
        self.assertAlmostEqual(p_max, 0.5,
                               msg='getattr fallback should return 0.5 after reset cleared attr')
        p_frames = getattr(filt, '_p_max_override_frames', 0)
        self.assertEqual(p_frames, 0,
                         msg='getattr fallback should return 0 after reset cleared attr')


# --- Anti-loophole guard ---------------------------------------------------

class AntiLoopholeTests(unittest.TestCase):

    def test_classifier_not_imported_into_aec_module(self):
        """The classifier module must NOT be referenced from aec.py — its
        output is analysis-only and must not feed production decisions."""
        with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'aec.py')) as f:
            src = f.read()
        for forbidden in (
            'aec_p52_regime_classifier',
            'modules.p52_regime_classifier',
            'AcousticRegimeClassifier',
            'AcousticRegime',
            'RegimeClassification',
        ):
            self.assertNotIn(
                forbidden, src,
                f'aec.py references {forbidden} — classifier must stay '
                f'analysis-only per P52 A.0R.3 design contract.')


if __name__ == '__main__':
    unittest.main()
