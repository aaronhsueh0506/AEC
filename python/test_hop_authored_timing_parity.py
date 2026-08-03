"""Regression test for the top-level (non-AEC3) hop-authored constant audit
(2026-08 gap-fix, follow-up to the AEC3-internal per-block/hop-count
constant audit -- see CHANGELOG.md).

Bug: ``shadow_err_alpha`` / ``warmup_frames`` / ``epc_hangover`` /
``ne_recent_hold`` / ``filter_misadjustment_stable_frames`` /
``filter_misadjustment_hangover_frames`` (AecConfig) and
``EchoPathChangeDetector.EPV_FAST_TC``/``EPV_SLOW_TC`` (epc.py) are
project-native (NOT AEC3-sourced) constants authored as literal hop counts
/ per-hop EMA constants against the legacy hop=160/sample_rate=16000
(10 ms) grid that predates this repo's own multi-rate history, with zero
rate conversion -- so the SAME literal covered wildly different wall-clock
durations depending on grid (e.g. warmup_frames=100 covered 0.8 s / 1.6 s /
1.067 s at 16k/256, 16k/512, 48k/1024 respectively before this fix).

Fix: retimed via ``aec3_scale.ms_to_hops``/``aec3_scale.growth_rehop`` in
``AecConfig.__post_init__`` (hop-count fields) and
``EchoPathChangeDetector.__init__`` (EPV_FAST_TC/EPV_SLOW_TC, now real
constructor parameters -- ``hop_size``/``sample_rate`` were previously not
parameters at all). ``ne_recent_sustain`` is a genuine consecutive-event
count (NOT a duration) and is intentionally left unretimed.

Run:
    python3 -m pytest python/test_hop_authored_timing_parity.py
    python3 -m unittest python/test_hop_authored_timing_parity.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.config import AecConfig  # noqa: E402
from modules.enums import AecPreset  # noqa: E402
from modules.epc import EchoPathChangeDetector  # noqa: E402

# The three grids the task calls out explicitly, plus 8 kHz (this repo's
# fourth whitelisted grid, and same hop_ms as 16k/512 -- a useful extra
# data point) for parity with the C side's test_rate_structural.c GRIDS[].
GRIDS = [
    (8000, 256),
    (16000, 256),
    (16000, 512),
    (48000, 1024),
]

# Legacy targets: the wall-clock duration / EMA time-constant each field
# was authored to cover at the legacy hop=160/sample_rate=16000 (10 ms)
# grid. TC = -hop_seconds / ln(retention) for the RETENTION-convention EMA
# constants (shadow_err_alpha, EPV_FAST_TC, EPV_SLOW_TC).
HOP_MS_TOL = 0.15   # integer-hop-count rounding budget (see C test's comment)
ALPHA_TOL = 0.01    # growth_rehop is continuous -- should be near-exact


def _hop_seconds(cfg: AecConfig) -> float:
    return cfg.hop_size / float(cfg.sample_rate)


class TopLevelConstantRetimingTests(unittest.TestCase):

    def _cfg(self, sr: int, frame_size: int) -> AecConfig:
        return AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=sr, frame_size=frame_size)

    def test_warmup_frames_wall_clock_matches_across_grids(self) -> None:
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_ms = _hop_seconds(cfg) * 1000.0
            actual_ms = cfg.warmup_frames * hop_ms
            self.assertAlmostEqual(
                actual_ms, 1000.0, delta=1000.0 * HOP_MS_TOL,
                msg=f"sr={sr} fs={fs}: warmup_frames={cfg.warmup_frames} "
                    f"-> {actual_ms:.1f} ms")

    def test_epc_hangover_wall_clock_matches_across_grids(self) -> None:
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_ms = _hop_seconds(cfg) * 1000.0
            actual_ms = cfg.epc_hangover * hop_ms
            self.assertAlmostEqual(
                actual_ms, 200.0, delta=200.0 * HOP_MS_TOL,
                msg=f"sr={sr} fs={fs}: epc_hangover={cfg.epc_hangover} "
                    f"-> {actual_ms:.1f} ms")

    def test_ne_recent_hold_wall_clock_matches_across_grids(self) -> None:
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_ms = _hop_seconds(cfg) * 1000.0
            actual_ms = cfg.ne_recent_hold * hop_ms
            self.assertAlmostEqual(
                actual_ms, 1500.0, delta=1500.0 * HOP_MS_TOL,
                msg=f"sr={sr} fs={fs}: ne_recent_hold={cfg.ne_recent_hold} "
                    f"-> {actual_ms:.1f} ms")

    def test_filter_misadjustment_frames_wall_clock_matches_across_grids(self) -> None:
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_ms = _hop_seconds(cfg) * 1000.0
            stable_ms = cfg.filter_misadjustment_stable_frames * hop_ms
            hangover_ms = cfg.filter_misadjustment_hangover_frames * hop_ms
            self.assertAlmostEqual(
                stable_ms, 300.0, delta=300.0 * HOP_MS_TOL,
                msg=f"sr={sr} fs={fs}: filter_misadjustment_stable_frames="
                    f"{cfg.filter_misadjustment_stable_frames} -> {stable_ms:.1f} ms")
            self.assertAlmostEqual(
                hangover_ms, 1000.0, delta=1000.0 * HOP_MS_TOL,
                msg=f"sr={sr} fs={fs}: filter_misadjustment_hangover_frames="
                    f"{cfg.filter_misadjustment_hangover_frames} -> {hangover_ms:.1f} ms")

    def test_shadow_err_alpha_time_constant_matches_across_grids(self) -> None:
        # shadow_err_alpha is the RETENTION-convention constant
        # (main_err_smooth = alpha*old + (1-alpha)*new -- see
        # orchestrator.py), so TC = -hop_seconds / ln(alpha).
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_s = _hop_seconds(cfg)
            tc_ms = -hop_s / math.log(cfg.shadow_err_alpha) * 1000.0
            self.assertAlmostEqual(
                tc_ms, 44.82, delta=44.82 * ALPHA_TOL,
                msg=f"sr={sr} fs={fs}: shadow_err_alpha={cfg.shadow_err_alpha:.6f} "
                    f"-> TC={tc_ms:.2f} ms")

    def test_epv_time_constants_match_across_grids(self) -> None:
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            hop_s = _hop_seconds(cfg)
            det = EchoPathChangeDetector(
                cfg, hop_size=cfg.hop_size, sample_rate=cfg.sample_rate)
            tc_fast = -hop_s / math.log(det._epv_fast_tc) * 1000.0
            tc_slow = -hop_s / math.log(det._epv_slow_tc) * 1000.0
            self.assertAlmostEqual(
                tc_fast, 495.0, delta=495.0 * ALPHA_TOL,
                msg=f"sr={sr} fs={fs}: EPV_FAST_TC={det._epv_fast_tc:.6f} "
                    f"-> TC={tc_fast:.1f} ms")
            self.assertAlmostEqual(
                tc_slow, 9995.0, delta=9995.0 * ALPHA_TOL,
                msg=f"sr={sr} fs={fs}: EPV_SLOW_TC={det._epv_slow_tc:.6f} "
                    f"-> TC={tc_slow:.1f} ms")

    def test_epc_detector_defaults_hop_size_sample_rate_from_config(self) -> None:
        """hop_size/sample_rate are optional kwargs -- omitting them must
        fall back to config.hop_size/config.sample_rate (the real call site
        in orchestrator.py passes them explicitly, but this guards any
        future/test call site that doesn't)."""
        cfg = self._cfg(16000, 512)
        det_explicit = EchoPathChangeDetector(
            cfg, hop_size=cfg.hop_size, sample_rate=cfg.sample_rate)
        det_implicit = EchoPathChangeDetector(cfg)
        self.assertEqual(det_explicit._epv_fast_tc, det_implicit._epv_fast_tc)
        self.assertEqual(det_explicit._epv_slow_tc, det_implicit._epv_slow_tc)

    def test_ne_recent_sustain_is_not_retimed(self) -> None:
        """Negative-space check: ne_recent_sustain is a genuine consecutive
        -event count, not a duration -- must stay the literal 3 at every
        grid (proving it was correctly left OUT of the retiming batch)."""
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            self.assertEqual(
                cfg.ne_recent_sustain, 3,
                msg=f"sr={sr} fs={fs}: ne_recent_sustain must stay literal 3")

    def test_pre_fix_regression_would_have_failed(self) -> None:
        """Sanity-check the tolerance band itself: the OLD (frozen-literal,
        never-retimed) behavior must fall clearly OUTSIDE the 15% band this
        test enforces -- otherwise the test wouldn't actually catch the bug
        it's named for. Mirrors the C test's own header comment."""
        # warmup_frames=100 hops frozen at every grid (pre-fix behavior):
        old_ms_16k_256 = 100 * (128 / 16000.0) * 1000.0   # 800 ms
        old_ms_16k_512 = 100 * (256 / 16000.0) * 1000.0   # 1600 ms
        rel_err = abs(old_ms_16k_512 - old_ms_16k_256) / old_ms_16k_256
        self.assertGreater(
            rel_err, 0.15,
            msg="the pre-fix frozen-literal behavior should differ by "
                "far more than the 15% tolerance band (it does not -- "
                "the tolerance band is too loose to catch a regression)")


if __name__ == '__main__':
    unittest.main()
