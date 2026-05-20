"""v3.21.3 Codex #1 — AEC.reset() AEC3 post-state clearing.

Pre-v3.21.3 bug: AEC.reset() body initialised the AEC3-aligned post-stage
fields (`_aec3_state` / `_aec3_ree` / `_aec3_sg` / `_aec3_ola_buf` /
`_aec3_noise_psd` / `_aec3_noise_initialized` / `_aec3_smooth_cn_gain` /
`_aec3_pending_*` / `_aec3_stationarity` + counters) in `__init__` but
never touched them on reset(). Re-using an AEC instance across utterances
carried previous-stream post-filter state into the next stream.

Run:
    python -m unittest python/test_aec_reset.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import AEC, AecConfig


def _run_some_frames(aec: AEC, n_frames: int = 20) -> None:
    """Push synthetic mic+ref frames through process() so AEC3 post-state
    accumulates."""
    rng = np.random.default_rng(0)
    block = int(aec.config.hop_size)
    for _ in range(n_frames):
        mic = rng.standard_normal(block).astype(np.float32) * 0.1
        ref = rng.standard_normal(block).astype(np.float32) * 0.1
        aec.process(mic, ref)


class Aec3PostResetTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = AecConfig()
        np.random.seed(42)  # CNG determinism
        self.aec = AEC(self.cfg)

    def test_aec3_post_state_exists_after_init(self) -> None:
        """Pre-condition: __init__ populates AEC3 post-state."""
        self.assertIsNotNone(self.aec._aec3_state)
        self.assertIsNotNone(self.aec._aec3_ree)
        self.assertIsNotNone(self.aec._aec3_sg)
        self.assertIsNotNone(self.aec._aec3_ola_buf)
        self.assertFalse(self.aec._aec3_noise_initialized)

    def test_aec3_noise_initialized_flips_after_frames(self) -> None:
        """Run frames -> noise PSD initialises -> flag True."""
        _run_some_frames(self.aec, n_frames=30)
        # CNG init typically fires within tens of frames once render is
        # non-zero; even if not, OLA buf will hold residual samples.
        ola_has_history = np.any(self.aec._aec3_ola_buf != 0)
        psd_has_history = np.any(self.aec._aec3_noise_psd != 0)
        self.assertTrue(
            self.aec._aec3_noise_initialized
            or ola_has_history
            or psd_has_history,
            'Expected AEC3 post-state to accumulate some history after frames',
        )

    def test_reset_clears_aec3_post_state(self) -> None:
        """After reset(), every _aec3_* mutable field is cleared."""
        _run_some_frames(self.aec, n_frames=30)
        # Snapshot the object identities so we can detect recreation.
        old_state = self.aec._aec3_state
        old_ree = self.aec._aec3_ree
        old_sg = self.aec._aec3_sg

        self.aec.reset()

        # AecState + ResidualEchoEstimator + SuppressionGain recreated
        # (no in-place reset() on AecState / SuppressionGain).
        self.assertIsNot(self.aec._aec3_state, old_state)
        self.assertIsNot(self.aec._aec3_ree, old_ree)
        self.assertIsNot(self.aec._aec3_sg, old_sg)

        # CNG state cleared.
        self.assertFalse(self.aec._aec3_noise_initialized)
        np.testing.assert_array_equal(
            self.aec._aec3_noise_psd, np.zeros_like(self.aec._aec3_noise_psd)
        )
        np.testing.assert_array_equal(
            self.aec._aec3_smooth_cn_gain,
            np.zeros_like(self.aec._aec3_smooth_cn_gain),
        )

        # OLA buffer cleared.
        np.testing.assert_array_equal(
            self.aec._aec3_ola_buf, np.zeros_like(self.aec._aec3_ola_buf)
        )

        # Stationarity counters cleared.
        self.assertFalse(self.aec._aec3_non_zero_render_seen)
        self.assertEqual(self.aec._aec3_stationarity_active_hops, 0)

        # EPV / delay pending events cleared.
        self.assertFalse(self.aec._aec3_pending_gain_change)
        self.assertIsNone(self.aec._aec3_pending_delay_change)

    def test_reset_is_idempotent(self) -> None:
        """Calling reset() twice doesn't raise + state still cleared."""
        _run_some_frames(self.aec, n_frames=5)
        self.aec.reset()
        self.aec.reset()
        self.assertFalse(self.aec._aec3_noise_initialized)
        np.testing.assert_array_equal(
            self.aec._aec3_ola_buf, np.zeros_like(self.aec._aec3_ola_buf)
        )

    def test_post_reset_processing_doesnt_raise(self) -> None:
        """Process some frames after reset() — verify AEC3 chain runs
        cleanly on the recreated state."""
        _run_some_frames(self.aec, n_frames=10)
        self.aec.reset()
        # Should run without exceptions.
        _run_some_frames(self.aec, n_frames=10)


class ReturnResContextTests(unittest.TestCase):
    """v3.21.3 Codex #3 — `return_res_context=True` contract.

    Before v3.21.3: `_res_context` was always None so the documented
    `(output, AecResContext)` return type never fired — only `ndarray`
    came back regardless of the config flag.
    """

    def setUp(self) -> None:
        self.cfg = AecConfig()
        self.cfg.return_res_context = True
        np.random.seed(42)
        self.aec = AEC(self.cfg)

    def test_default_returns_ndarray_only(self) -> None:
        """With return_res_context=False, process() returns ndarray
        (regression guard)."""
        cfg = AecConfig()
        np.random.seed(42)
        aec = AEC(cfg)
        rng = np.random.default_rng(0)
        block = int(aec.config.hop_size)
        out = aec.process(
            rng.standard_normal(block).astype(np.float32) * 0.1,
            rng.standard_normal(block).astype(np.float32) * 0.1,
        )
        self.assertIsInstance(out, np.ndarray)

    def test_flag_returns_tuple(self) -> None:
        """With return_res_context=True, process() returns (output,
        AecResContext)."""
        from aec import AecResContext
        rng = np.random.default_rng(0)
        block = int(self.aec.config.hop_size)
        # First frame may not trigger _aec3_post (warmup); run a few.
        result = None
        for _ in range(5):
            result = self.aec.process(
                rng.standard_normal(block).astype(np.float32) * 0.1,
                rng.standard_normal(block).astype(np.float32) * 0.1,
            )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        out, ctx = result
        self.assertIsInstance(out, np.ndarray)
        self.assertIsInstance(ctx, AecResContext)
        # echo_spec / far_spec / near_spec are complex arrays.
        self.assertTrue(np.iscomplexobj(ctx.echo_spec))
        self.assertTrue(np.iscomplexobj(ctx.far_spec))
        self.assertTrue(np.iscomplexobj(ctx.near_spec))
        # Scalars are typed as Python float / bool.
        self.assertIsInstance(ctx.far_power, float)
        self.assertIsInstance(ctx.filter_converged, bool)
        self.assertIsInstance(ctx.erle_factor, float)


if __name__ == '__main__':
    unittest.main()
