"""F2.1 — reset stale upstream state on EPC fire (unit tests).

Verifies `AecConfig.use_epc_state_reset` gating of
`AEC._apply_epc_state_reset`. The reset path is invoked inside the
EPV-fired and shadow-rise-fired branches at aec.py:~5129/~5152 when
the flag is True; flag-OFF behaviour is byte-identical to the
pre-F2.1 implementation.

Direct invocation tests (no full bench): mutate the four reset
targets, call `_apply_epc_state_reset`, verify they return to init.
Flag-OFF parity: instantiate AEC with flag OFF and confirm no helper
attribute (`_f2_1_reset_counts`) appears even after EPC firings would
have been possible. Flag-ON path: invoke the helper directly and
inspect the counter increment.

Run:
    PYTHONPATH=python python3 -m unittest python.test_f2_1_epc_state_reset
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import AEC, AecConfig, AecMode, AecPreset


def _make_aec(*, use_epc_reset: bool) -> AEC:
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=16000, mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True, enable_delay_est=False,
        use_epc_state_reset=use_epc_reset,
    )
    np.random.seed(0)
    return AEC(cfg)


class TestF21FlagOff(unittest.TestCase):
    """Flag-OFF: helper is never called by the EPC branches, no telemetry."""

    def test_flag_off_no_telemetry(self):
        aec = _make_aec(use_epc_reset=False)
        # Synthetic 1s far burst into mic to make AEC do real work and
        # exercise the EPC update paths (they are read every frame).
        sr = aec.config.sample_rate
        n = sr
        far = (np.random.default_rng(0).standard_normal(n) * 0.1).astype(np.float32)
        mic = (np.random.default_rng(1).standard_normal(n) * 0.05).astype(np.float32)
        hop = aec.hop_size
        for pos in range(0, n - hop, hop):
            aec.process(mic[pos: pos + hop], far[pos: pos + hop])
        # Even with EPV / shadow-rise possibly firing, flag-OFF must not have
        # touched the F2.1 counter attribute.
        self.assertFalse(hasattr(aec, '_f2_1_reset_counts'),
                          'flag-OFF must not create F2.1 telemetry attribute')


class TestF21Reset(unittest.TestCase):
    """Direct invocation of the reset helper sets the four state values
    back to their post-__init__ values."""

    def test_reset_restores_init_values(self):
        aec = _make_aec(use_epc_reset=True)
        # Mutate the four targets to non-init values (mimics post-conv state).
        aec._erl_estimate = 0.42
        aec._erle_window_near = 17.0
        aec._erle_window_err = 3.5
        aec._wn_err_baseline = 1.0

        aec._apply_epc_state_reset('epv')

        self.assertAlmostEqual(aec._erl_estimate, 0.1, places=6)
        self.assertAlmostEqual(aec._erle_window_near, 1e-10, places=12)
        self.assertAlmostEqual(aec._erle_window_err, 1e-10, places=12)
        self.assertAlmostEqual(aec._wn_err_baseline, 1e-8, places=12)
        self.assertEqual(aec._f2_1_reset_counts['epv'], 1)
        self.assertEqual(aec._f2_1_reset_counts['shadow_rise'], 0)

    def test_per_source_counters_independent(self):
        aec = _make_aec(use_epc_reset=True)
        aec._apply_epc_state_reset('epv')
        aec._apply_epc_state_reset('shadow_rise')
        aec._apply_epc_state_reset('epv')
        self.assertEqual(aec._f2_1_reset_counts['epv'], 2)
        self.assertEqual(aec._f2_1_reset_counts['shadow_rise'], 1)


class TestF21Integration(unittest.TestCase):
    """End-to-end smoke: flag-ON does not crash; produces sensible output."""

    def test_flag_on_runs_without_crash(self):
        aec_off = _make_aec(use_epc_reset=False)
        aec_on = _make_aec(use_epc_reset=True)

        sr = aec_on.config.sample_rate
        n = 2 * sr  # 2s — long enough for shadow filter to start tracking
        rng_far = np.random.default_rng(0)
        rng_mic = np.random.default_rng(1)
        far = (rng_far.standard_normal(n) * 0.1).astype(np.float32)
        mic = (rng_mic.standard_normal(n) * 0.05).astype(np.float32)
        hop = aec_on.hop_size

        out_off = np.zeros(n, dtype=np.float32)
        out_on  = np.zeros(n, dtype=np.float32)
        for pos in range(0, n - hop, hop):
            out_off[pos: pos + hop] = aec_off.process(
                mic[pos: pos + hop], far[pos: pos + hop])
            out_on[pos: pos + hop]  = aec_on.process(
                mic[pos: pos + hop], far[pos: pos + hop])

        # Synthetic noise rarely triggers EPC; the two outputs should be
        # very close (flag-ON is byte-identical unless EPC fires).
        self.assertFalse(np.any(np.isnan(out_on)))
        self.assertFalse(np.any(np.isnan(out_off)))


if __name__ == '__main__':
    unittest.main()
