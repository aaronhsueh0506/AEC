"""Codex review regression test (2026-08-03) — EchoPathDelayEstimator.reset()
must clear its ENTIRE 48kHz signal chain consistently: both the OUTER
_Resample48 anti-alias sidechain (filter state + decimation phase) AND the
INNER decimators / ring buffer / pending edge-chunker samples.

Bug this guards against: reset() cleared only the outer sidechain (added
alongside the sidechain itself), leaving the inner chain's biquad memory /
ring-buffer audio history / pending partial-block samples stale — a mixed
fresh/stale reset. Call-site audit: the ONLY caller of
EchoPathDelayEstimator.reset() is LegacyDelayShim.reset() <- AEC.reset()
(orchestrator.py:839), the top-level, cold-start-style full reset (see
test_aec_reset.py — it recreates AecState/ResidualEchoEstimator/
SuppressionGain, clears CNG state, zeroes the OLA buffer, resets every other
subsystem). It is NOT a lightweight per-echo-path-change nudge — that is the
separate internal ``_reset(reset_lag_aggregator=False, ...)`` soft-reset path
inside ``_process_inner_block`` (fires on the consistent-estimate stability
counter, ~every 500 ms in normal steady state), which correctly never
touches the sidechain or the inner buffers. So the fix makes the public
``reset()`` clear the whole chain together.

Run:
    python3 -m unittest python/test_delay_reset.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.delay.echo_path_delay_estimator import EchoPathDelayEstimator


def _feed_delayed_noise(est, seed, n_hops, hop, delay_samples):
    """Feed (render=noise, capture=noise delayed by delay_samples) through
    est in `hop`-sample chunks for n_hops hops. Returns the LAST emitted
    DelayEstimate, or None."""
    rng = np.random.default_rng(seed)
    total = n_hops * hop + delay_samples
    far = (rng.standard_normal(total).astype(np.float32)) * 0.2
    near = np.zeros(total, dtype=np.float32)
    if delay_samples:
        near[delay_samples:] = far[:-delay_samples]
    else:
        near[:] = far
    last = None
    for i in range(n_hops):
        r = far[i * hop:(i + 1) * hop]
        c = near[i * hop:(i + 1) * hop]
        got = est.estimate_delay(r, c)
        if got is not None:
            last = got
    return last


class Resample48ResetConsistencyTests(unittest.TestCase):
    """Whitebox: reset() must zero the outer sidechain AND the inner
    decimator/ring/pending state together (the exact assertion the pre-fix
    code fails)."""

    def test_reset_clears_inner_signal_chain_state(self):
        est = EchoPathDelayEstimator(sample_rate=48000)
        self.assertIsNotNone(est._resample48_render)
        self.assertIsNotNone(est._resample48_capture)

        # One hop is enough to dirty the inner decimator/ring/pending state
        # without needing to wait for a delay lock.
        hop = 480  # 10 ms @ 48 kHz
        rng = np.random.default_rng(0)
        render_hop = (rng.standard_normal(hop).astype(np.float32)) * 0.2
        capture_hop = (rng.standard_normal(hop).astype(np.float32)) * 0.2
        est.estimate_delay(render_hop, capture_hop)

        # Pre-reset sanity: state actually built up in the inner chain
        # (else the post-reset zero-checks below would pass trivially).
        built_up = (
            np.any(est._render_ring.buffer != 0.0)
            or est._render_pending.size > 0
            or est._capture_pending.size > 0
            or np.any(est._render_decimator._anti_alias._z != 0.0)
            or np.any(est._capture_decimator._anti_alias._z != 0.0)
        )
        self.assertTrue(built_up, "pre-reset sanity: inner chain should hold state")

        est.reset(reset_delay_confidence=True)

        # Outer sidechain cleared (pre-existing behaviour, unaffected by
        # this fix).
        self.assertTrue(np.all(est._resample48_render._zi == 0.0))
        self.assertTrue(np.all(est._resample48_capture._zi == 0.0))
        self.assertEqual(est._resample48_render._phase, 0)
        self.assertEqual(est._resample48_capture._phase, 0)

        # Inner chain must NOW also be fully cleared -- the fix.
        self.assertTrue(np.all(est._render_ring.buffer == 0.0),
                         "render_ring buffer left stale pre-fix")
        self.assertEqual(est._render_ring.write, 0,
                          "render_ring write cursor left stale pre-fix")
        self.assertEqual(est._render_pending.size, 0,
                          "render_pending left stale pre-fix")
        self.assertEqual(est._capture_pending.size, 0,
                          "capture_pending left stale pre-fix")
        self.assertTrue(np.all(est._render_decimator._anti_alias._z == 0.0),
                         "render_decimator anti_alias state left stale pre-fix")
        self.assertTrue(np.all(est._render_decimator._noise_reduction._z == 0.0),
                         "render_decimator noise_reduction state left stale pre-fix")
        self.assertTrue(np.all(est._capture_decimator._anti_alias._z == 0.0),
                         "capture_decimator anti_alias state left stale pre-fix")
        self.assertTrue(np.all(est._capture_decimator._noise_reduction._z == 0.0),
                         "capture_decimator noise_reduction state left stale pre-fix")

    def test_reset_at_16k_still_works(self):
        """Non-48kHz configs have no sidechain (_resample48_* is None) --
        reset() must not crash and must still clear the inner chain."""
        est = EchoPathDelayEstimator(sample_rate=16000)
        self.assertIsNone(est._resample48_render)
        hop = 160
        rng = np.random.default_rng(1)
        render_hop = (rng.standard_normal(hop).astype(np.float32)) * 0.2
        capture_hop = (rng.standard_normal(hop).astype(np.float32)) * 0.2
        est.estimate_delay(render_hop, capture_hop)

        est.reset(reset_delay_confidence=True)

        self.assertTrue(np.all(est._render_ring.buffer == 0.0))
        self.assertEqual(est._render_pending.size, 0)
        self.assertEqual(est._capture_pending.size, 0)


class Resample48ResetReacquisitionTests(unittest.TestCase):
    """Functional: reset mid-stream at 48kHz, then verify delay
    re-acquisition works correctly -- the post-reset estimator must behave
    like a freshly-constructed one for whatever stream follows."""

    def test_reset_mid_stream_reacquires_like_fresh(self):
        hop = 480           # 10 ms @ 48 kHz
        n_hops_d1 = 400      # 4 s: lock onto D1 before reset
        n_hops_d2 = 60       # 600 ms post-reset: short window maximises
                             # sensitivity to any leftover contamination
        d1_samples = 4800    # 100 ms
        d2_samples = 14400   # 300 ms

        est = EchoPathDelayEstimator(sample_rate=48000)
        got1 = _feed_delayed_noise(est, 11, n_hops_d1, hop, d1_samples)
        self.assertIsNotNone(got1, "expected a delay estimate before reset")

        est.reset(reset_delay_confidence=True)

        got_after_reset = _feed_delayed_noise(est, 22, n_hops_d2, hop, d2_samples)

        fresh = EchoPathDelayEstimator(sample_rate=48000)
        got_fresh = _feed_delayed_noise(fresh, 22, n_hops_d2, hop, d2_samples)

        self.assertIsNotNone(got_after_reset, "expected re-acquisition after reset")
        self.assertIsNotNone(got_fresh, "expected fresh-instance baseline estimate")
        self.assertEqual(int(got_after_reset.delay), int(got_fresh.delay))
        self.assertEqual(got_after_reset.quality, got_fresh.quality)


if __name__ == "__main__":
    unittest.main()
