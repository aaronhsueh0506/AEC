"""v3.21.3 reset-state regression — AEC.reset() AEC3 post-state clearing.

Pre-v3.21.3 bug: AEC.reset() body initialised the AEC3-aligned post-stage
fields (`_aec3_state` / `_aec3_ree` / `_aec3_sg` / `_aec3_ola_buf` /
CNG state (`_aec3_n2` / `_aec3_n2_initial` / `_aec3_y2_smoothed` /
`_aec3_n2_counter` / `_aec3_cng_seed` / `_aec3_noise_initialized`) /
`_aec3_pending_*` / `_aec3_stationarity` + counters) in `__init__` but
never touched them on reset(). Re-using an AEC instance across utterances
carried previous-stream post-filter state into the next stream.

Run:
    python -m pytest python/tests/test_aec_reset.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        """Run frames -> Y2_smoothed initialises -> flag True."""
        _run_some_frames(self.aec, n_frames=30)
        # CNG init typically fires within tens of frames once render is
        # non-zero; even if not, OLA buf will hold residual samples.
        ola_has_history = np.any(self.aec._aec3_ola_buf != 0)
        y2_has_history = np.any(self.aec._aec3_y2_smoothed != 0)
        self.assertTrue(
            self.aec._aec3_noise_initialized
            or ola_has_history
            or y2_has_history,
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

        # CNG state cleared (AEC3-strict — mirrors ComfortNoiseGenerator ctor).
        self.assertFalse(self.aec._aec3_noise_initialized)
        self.assertEqual(self.aec._aec3_n2_counter, 0)
        self.assertEqual(self.aec._aec3_cng_seed, 42)
        np.testing.assert_array_equal(
            self.aec._aec3_y2_smoothed,
            np.zeros_like(self.aec._aec3_y2_smoothed),
        )
        np.testing.assert_array_equal(
            self.aec._aec3_n2_initial,
            np.zeros_like(self.aec._aec3_n2_initial),
        )
        np.testing.assert_array_equal(
            self.aec._aec3_n2,
            np.full_like(self.aec._aec3_n2, 1.0e6),
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
    """v3.21.3 return-context regression — `return_res_context=True` contract.

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


def _echo_scene(rng, hop, n_hops, delay, erl):
    """Far-end bursts plus a mic that is a delayed, attenuated copy with
    near-end on top -- an echo path the delay estimator can lock onto, and a
    near-end talker on a slower cycle so the run sees single- and
    double-talk."""
    history = np.zeros(4096, dtype=np.float32)
    for h in range(n_hops):
        gain = 0.4 if (h % 40) < 30 else 0.0
        ref = (gain * rng.standard_normal(hop)).astype(np.float32)
        back = delay + np.arange(hop - 1, -1, -1)
        echo = np.where(back < history.size, history[np.minimum(back, history.size - 1)], 0.0)
        near = (0.25 * rng.standard_normal(hop)).astype(np.float32) if (h % 130) < 35 else 0.0
        mic = (erl * echo + near + 0.01 * rng.standard_normal(hop)).astype(np.float32)
        history = np.concatenate((ref[::-1], history[:-hop]))
        yield mic, ref


class ResetEqualsFreshInstanceTests(unittest.TestCase):
    """``AEC.reset()`` owes the caller a fresh instance.

    The C twin gate is ``make test-reset-parity``
    (c_impl/test/test_reset_parity.c); this is the same property on this port,
    so the two resets cannot drift apart on what they clear. Before the fix
    both ports carried the filters' AEC3 startup gates, the Kalman H_error and
    the inst-ERLE slope ring across the call, and the first post-reset hop
    already diverged.
    """

    WARM_HOPS = 600
    TEST_HOPS = 300
    # The three product grids, same set the C twin runs. The grid matters:
    # the Kalman H_error residue only steers a decision at the two larger
    # transforms, so a 16k/256-only gate reports green with it left standing.
    GRIDS = ((16000, 256), (16000, 512), (48000, 1024))

    def test_warmed_then_reset_matches_a_never_warmed_instance(self) -> None:
        for sample_rate, frame_size in self.GRIDS:
            with self.subTest(sample_rate=sample_rate, frame_size=frame_size):
                self._one_grid(sample_rate, frame_size)

    def _one_grid(self, sample_rate: int, frame_size: int) -> None:
        cfg = AecConfig(sample_rate=sample_rate, frame_size=frame_size)
        np.random.seed(42)
        fresh = AEC(cfg)
        np.random.seed(42)
        warmed = AEC(cfg)
        hop = int(cfg.hop_size)

        # An echo path the compare phase does NOT reuse, so anything the
        # subject remembers is wrong for what follows.
        for mic, ref in _echo_scene(np.random.default_rng(1234567), hop,
                                    self.WARM_HOPS, 611, 0.65):
            warmed.process(mic, ref)
        warmed.reset()

        first_bad = None
        diverged = 0
        scene_a = _echo_scene(np.random.default_rng(89), hop, self.TEST_HOPS, 293, 0.5)
        scene_b = _echo_scene(np.random.default_rng(89), hop, self.TEST_HOPS, 293, 0.5)
        for index, ((mic, ref), (mic_b, ref_b)) in enumerate(zip(scene_a, scene_b)):
            out_a = fresh.process(mic, ref)
            out_b = warmed.process(mic_b, ref_b)
            if not np.array_equal(np.asarray(out_a), np.asarray(out_b)):
                diverged += 1
                if first_bad is None:
                    first_bad = index
        self.assertEqual(
            diverged, 0,
            f'{diverged}/{self.TEST_HOPS} hops differ, first at {first_bad}')


if __name__ == '__main__':
    unittest.main()
