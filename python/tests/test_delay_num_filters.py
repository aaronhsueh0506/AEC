"""Config pass-through for the matched-filter bank size
(``AecConfig.delay_num_filters``).

Why this test exists: ``EchoPathDelayEstimator`` has always taken a
``num_filters`` argument and has always sized its downsampled ring and its
aggregator histograms from it — but nothing could SET it. The orchestrator
built ``LegacyDelayShim`` without it and the shim swallowed every kwarg it
did not recognise, so the whole chain was pinned at the AEC3 default of 5.
That is the right value for bench / dataset generation (it is what the
published scores were measured at), but the embedded target compensates the
measured system delay with ``fixed_delay_samples`` and only needs the
matched filter to track the residual, where a smaller bank buys a large
MAC saving (n=1 is -73% of the full-rate search cost).

What is asserted:
  1. DEFAULT IS UNCHANGED — ``AecConfig()`` and
     ``AecConfig(delay_num_filters=5)`` produce bit-identical output on the
     same input, and the estimator built at the default still has the exact
     ring capacity (2064) and highest-peak histogram size (2433) it had
     before this knob existed.
  2. THE KNOB REALLY MOVES THE GEOMETRY (the can-fail core) — the bank's
     reliable reach is ``(n-1)*384 + 501`` downsampled samples (0.25 ms
     each), i.e. 125 / 221 / 317 / 413 / 509 ms for n = 1..5. So a 350 ms
     echo is INSIDE reach at n=5 and OUTSIDE it at n=2, while a 150 ms echo
     is inside reach at both. Driving the FULL stack (AecConfig -> AEC ->
     orchestrator -> LegacyDelayShim -> EchoPathDelayEstimator) means the
     n=2 "must not lock 350 ms" assertion fails the moment either wiring
     hop stops forwarding the value, because both regressions land on the
     same fallback of 5. ``test_mutation_*`` proves exactly that by
     re-running the n=2 case with the shim's forwarding monkeypatched away.
  3. RANGE IS ENFORCED — 0 and 6 raise, fail-fast like the signal-grid
     check next to it in ``__post_init__``.

Run:
    python3 -m pytest python/tests/test_delay_num_filters.py
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AEC, AecConfig
from modules.delay import legacy_compat
from modules.delay.echo_path_delay_estimator import EchoPathDelayEstimator


# Ring capacity / histogram size the AEC3-default geometry has always had.
# Hard-coded rather than recomputed from the same formula the source uses --
# a recomputed expectation would move together with a broken formula.
_DEFAULT_RING_CAPACITY = 2064          # (32 + 24*(5-1) + 1) * 16
_DEFAULT_HP_HISTOGRAM_SIZE = 2433      # 5*384 + 512 + 1
# One downsampled sample is 0.25 ms, so the tolerance below (160 raw samples
# at 16 kHz = 10 ms) is far wider than the estimator's own quantisation
# (headroom 32 raw + ds granularity) yet far tighter than the ~180 ms gap
# between the true delay and the deepest lag an out-of-reach bank can report.
_LOCK_TOLERANCE_SAMPLES = 160
# 350 ms, not 400: on a pure single-tap echo the PRE-FIX pre-echo
# aggregator (the behaviour main carries while 700994b is branch-pending)
# deterministically reports the 400 ms case one alignment shift early
# (4800 vs 6400) -- same quirk test_delay_num_filters.c documents. 350 ms
# behaves on both pre-echo variants, keeps the geometry point intact
# (350 > n=2's 221 ms reach, < n=5's 509 ms), and lets this file pass on
# main and on the branch alike.


def _synth_echo(sample_rate: int, seconds: float, delay_samples: int,
                seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """White far-end plus a pure delayed/attenuated copy as the near-end."""
    rng = np.random.default_rng(seed)
    total = int(seconds * sample_rate)
    far = rng.standard_normal(total + delay_samples).astype(np.float32) * 0.2
    near = np.zeros_like(far)
    near[delay_samples:] = far[:-delay_samples] * 0.5
    return near, far


def _run_aec(cfg: AecConfig, near: np.ndarray, far: np.ndarray,
             seconds: float) -> tuple[np.ndarray, AEC]:
    np.random.seed(42)   # CNG determinism
    aec = AEC(cfg)
    hop = int(cfg.hop_size)
    n_hops = int(seconds * cfg.sample_rate) // hop
    out = [aec.process(near[i * hop:(i + 1) * hop], far[i * hop:(i + 1) * hop])
           for i in range(n_hops)]
    return np.concatenate(out), aec


def _lock_delay(num_filters: int, delay_ms: float,
                seconds: float = 3.0) -> tuple[int, int, AEC]:
    """Drive the full stack at ``num_filters``; return (true, reported, aec).
    Reported is -1 when the estimator never produced an estimate."""
    cfg = AecConfig(delay_num_filters=num_filters)
    delay = int(round(delay_ms * cfg.sample_rate / 1000.0))
    near, far = _synth_echo(cfg.sample_rate, seconds, delay)
    _, aec = _run_aec(cfg, near, far, seconds)
    return delay, int(aec.delay_est.estimated_delay), aec


class DelayNumFiltersDefaultTests(unittest.TestCase):
    """The default path must be indistinguishable from before the knob."""

    def test_explicit_five_is_bit_identical_to_the_default(self) -> None:
        seconds = 2.0
        base = AecConfig()
        self.assertEqual(base.delay_num_filters, 5)
        near, far = _synth_echo(base.sample_rate, seconds,
                                delay_samples=int(0.12 * base.sample_rate))
        out_default, _ = _run_aec(AecConfig(), near, far, seconds)
        out_explicit, _ = _run_aec(AecConfig(delay_num_filters=5), near, far, seconds)
        self.assertTrue(np.array_equal(out_default, out_explicit),
                        "delay_num_filters=5 must be byte-equal to the default")

    def test_default_geometry_sizes_are_unchanged(self) -> None:
        est = EchoPathDelayEstimator(sample_rate=16000)
        self.assertEqual(est._matched_filter._num_filters, 5)
        self.assertEqual(est._render_ring.size, _DEFAULT_RING_CAPACITY)
        self.assertEqual(est._aggregator._highest_peak._histogram.size,
                         _DEFAULT_HP_HISTOGRAM_SIZE)

    def test_default_geometry_survives_the_full_config_path(self) -> None:
        np.random.seed(42)
        aec = AEC(AecConfig())
        est = aec.delay_est._estimator
        self.assertEqual(est._matched_filter._num_filters, 5)
        self.assertEqual(est._render_ring.size, _DEFAULT_RING_CAPACITY)
        self.assertEqual(est._aggregator._highest_peak._histogram.size,
                         _DEFAULT_HP_HISTOGRAM_SIZE)


class ShimNumFiltersRangeTests(unittest.TestCase):
    """Direct LegacyDelayShim callers face the same [1, 5] range the
    orchestrated AecConfig path and the C low-level API enforce -- accepting
    any int silently here would bypass every other validation layer."""

    def test_direct_shim_rejects_out_of_range(self) -> None:
        from modules.delay.legacy_compat import LegacyDelayShim
        for bad in (0, -1, 6, 99):
            with self.assertRaises(ValueError):
                LegacyDelayShim(sample_rate=16000, hop_size=128,
                                num_filters=bad)

    def test_direct_shim_accepts_bounds(self) -> None:
        from modules.delay.legacy_compat import LegacyDelayShim
        for ok in (1, 5):
            LegacyDelayShim(sample_rate=16000, hop_size=128, num_filters=ok)


class DelayNumFiltersGeometryTests(unittest.TestCase):
    """The knob must actually shrink the bank's reach end to end."""

    def test_five_filters_lock_a_350ms_echo(self) -> None:
        true_delay, got, _ = _lock_delay(num_filters=5, delay_ms=350.0)
        self.assertNotEqual(got, -1, "n=5 must reach a 350 ms echo (509 ms bank)")
        self.assertLessEqual(abs(got - true_delay), _LOCK_TOLERANCE_SAMPLES,
                             f"n=5 reported {got}, true {true_delay}")

    def test_two_filters_do_not_lock_a_350ms_echo(self) -> None:
        """The can-fail assertion: 350 ms is beyond the n=2 bank's 221 ms
        reliable reach, so a correctly-shrunk bank cannot report it."""
        true_delay, got, aec = _lock_delay(num_filters=2, delay_ms=350.0)
        if got != -1:
            self.assertGreater(abs(got - true_delay), _LOCK_TOLERANCE_SAMPLES,
                               f"n=2 must not lock a 350 ms echo, reported {got}")
        self.assertEqual(aec.delay_est._estimator._matched_filter._num_filters, 2)

    def test_two_filters_lock_a_150ms_echo(self) -> None:
        """The complement: shrinking the bank must not break what is still
        inside reach (150 ms < 221 ms), so the test above is measuring reach
        and not simply a broken estimator."""
        true_delay, got, _ = _lock_delay(num_filters=2, delay_ms=150.0)
        self.assertNotEqual(got, -1, "n=2 must still reach a 150 ms echo")
        self.assertLessEqual(abs(got - true_delay), _LOCK_TOLERANCE_SAMPLES,
                             f"n=2 reported {got}, true {true_delay}")

    def test_two_filters_shrink_ring_and_histogram(self) -> None:
        np.random.seed(42)
        aec = AEC(AecConfig(delay_num_filters=2))
        est = aec.delay_est._estimator
        self.assertEqual(est._render_ring.size, (32 + 24 * 1 + 1) * 16)   # 912
        self.assertEqual(est._aggregator._highest_peak._histogram.size,
                         2 * 384 + 512 + 1)                               # 1281

    def test_mutation_dropping_the_shim_forwarding_makes_n2_lock_350ms(self) -> None:
        """Mutation control. Re-run the n=2 / 350 ms case with the shim's
        num_filters forwarding removed (exactly the pre-fix behaviour, and
        also what a dropped orchestrator kwarg degrades to -- both fall back
        to 5). The bank silently becomes a 509 ms one, the echo comes into
        reach, and the assertion in
        ``test_two_filters_do_not_lock_a_350ms_echo`` would go red. If this
        test ever fails, that assertion is passing for some reason other
        than the geometry."""

        class _SwallowNumFilters(EchoPathDelayEstimator):
            def __init__(self, *, num_filters: int = 5, **kwargs) -> None:
                super().__init__(**kwargs)   # num_filters dropped -> default 5

        original = legacy_compat.EchoPathDelayEstimator
        legacy_compat.EchoPathDelayEstimator = _SwallowNumFilters
        try:
            true_delay, got, aec = _lock_delay(num_filters=2, delay_ms=350.0)
            self.assertEqual(
                aec.delay_est._estimator._matched_filter._num_filters, 5,
                "mutation should have reverted the bank to the 5-filter default")
        finally:
            legacy_compat.EchoPathDelayEstimator = original
        self.assertNotEqual(got, -1)
        self.assertLessEqual(
            abs(got - true_delay), _LOCK_TOLERANCE_SAMPLES,
            "un-wired num_filters must let a 350 ms echo lock again "
            f"(reported {got}, true {true_delay}) -- otherwise the n=2 "
            "no-lock assertion is not actually testing the wiring")


class DelayNumFiltersValidationTests(unittest.TestCase):

    def test_zero_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AecConfig(delay_num_filters=0)

    def test_six_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AecConfig(delay_num_filters=6)

    def test_one_through_five_are_accepted(self) -> None:
        for n in (1, 2, 3, 4, 5):
            self.assertEqual(AecConfig(delay_num_filters=n).delay_num_filters, n)


if __name__ == "__main__":
    unittest.main()
