"""Tests for the top-level (non-AEC3) hop-authored constant retiming.

``shadow_err_alpha`` / ``warmup_frames`` / ``epc_hangover`` / ``ne_recent_hold`` /
``filter_misadjustment_stable_frames`` / ``filter_misadjustment_hangover_frames``
(AecConfig) and ``EchoPathChangeDetector.EPV_FAST_TC``/``EPV_SLOW_TC`` (epc.py)
are project-native (NOT AEC3-sourced) constants authored as hop counts / per-hop
EMA constants for the hop=160/sample_rate=16000 (10 ms) grid. They are retimed
via ``aec3_scale.ms_to_hops``/``aec3_scale.growth_rehop`` in
``AecConfig.__post_init__`` (hop-count fields) and
``EchoPathChangeDetector.__init__`` (EPV_FAST_TC/EPV_SLOW_TC, from its
``hop_size``/``sample_rate`` parameters), so each covers the same wall-clock
duration on every grid. ``ne_recent_sustain`` is deliberately left unretimed --
see config.py's field comment.

Run:
    python3 -m pytest python/tests/test_hop_authored_timing_parity.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import AecConfig  # noqa: E402
from modules.enums import AecPreset  # noqa: E402
from modules.epc import EchoPathChangeDetector  # noqa: E402
from modules.state.filter_analyzer import FilterAnalyzer  # noqa: E402

# The three product grids plus 8 kHz (the fourth whitelisted grid, same hop_ms
# as 16k/512), matching the C side's test_rate_structural.c GRIDS[].
GRIDS = [
    (8000, 256),
    (16000, 256),
    (16000, 512),
    (48000, 1024),
]

# Targets: the wall-clock duration / EMA time-constant each field was authored to
# cover at the hop=160/sample_rate=16000 (10 ms) grid. TC = -hop_seconds /
# ln(retention) for the RETENTION-convention EMA constants (shadow_err_alpha,
# EPV_FAST_TC, EPV_SLOW_TC).
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
        """Negative-space check: ne_recent_sustain stays the literal 3 at
        every grid, deliberately left unretimed. See config.py's field
        comment for why."""
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            self.assertEqual(
                cfg.ne_recent_sustain, 3,
                msg=f"sr={sr} fs={fs}: ne_recent_sustain must stay literal 3")

    def test_filter_analyzer_ir_exclusion_window_scales_with_rate(self) -> None:
        """The AEC3 64/128-sample window means 4/8 ms at 16 kHz.

        It indexes the time-domain impulse response, so changing the FFT/hop at the
        same rate must not move it, while 48 kHz needs 192/384 samples.
        """
        expected = {
            (8000, 256): (64, 128),
            (16000, 256): (64, 128),
            (16000, 512): (64, 128),
            (48000, 1024): (192, 384),
        }
        for sr, fs in GRIDS:
            cfg = self._cfg(sr, fs)
            analyzer = FilterAnalyzer(
                hop_size=cfg.hop_size, sample_rate=cfg.sample_rate)
            actual = (
                analyzer._consistent._floor_low_offset,
                analyzer._consistent._floor_high_offset,
            )
            self.assertEqual(actual, expected[(sr, fs)])

    def test_pre_fix_regression_would_have_failed(self) -> None:
        """Sanity-check the tolerance band itself: a frozen (never-retimed) literal must
        fall clearly OUTSIDE the 15% band this test enforces, otherwise the test could
        not catch a missing retime. Mirrors the C test's own header comment.
        """
        # warmup_frames=100 hops frozen at every grid:
        old_ms_16k_256 = 100 * (128 / 16000.0) * 1000.0   # 800 ms
        old_ms_16k_512 = 100 * (256 / 16000.0) * 1000.0   # 1600 ms
        rel_err = abs(old_ms_16k_512 - old_ms_16k_256) / old_ms_16k_256
        self.assertGreater(
            rel_err, 0.15,
            msg="the pre-fix frozen-literal behavior should differ by "
                "far more than the 15% tolerance band (it does not -- "
                "the tolerance band is too loose to catch a regression)")


class ConfigReconstructionIdempotencyTests(unittest.TestCase):
    """Rebuilding an already-resolved config must not re-retime it.

    ``dataclasses.replace()`` / ``AecConfig(**asdict(cfg))`` (e.g.
    process_wav_files()'s sample-rate-mismatch re-resolve path in
    modules/orchestrator.py) must leave the six wall-clock fields unchanged on the
    same grid, and rederive them from the pinned ``_canonical_*`` values on a
    different grid, so repeated or chained reconstruction never compounds drift.
    """

    RETIMED_FIELDS = (
        'warmup_frames', 'epc_hangover', 'ne_recent_hold',
        'filter_misadjustment_stable_frames',
        'filter_misadjustment_hangover_frames', 'shadow_err_alpha',
    )

    def _values(self, cfg: AecConfig) -> tuple:
        return tuple(getattr(cfg, f) for f in self.RETIMED_FIELDS)

    def test_replace_is_idempotent(self) -> None:
        from dataclasses import replace
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        original = self._values(cfg)
        once = self._values(replace(cfg))
        thrice = self._values(replace(replace(replace(cfg))))
        self.assertEqual(
            once, original,
            msg="dataclasses.replace(cfg) must not re-retime an "
                "already-resolved config")
        self.assertEqual(
            thrice, original,
            msg="repeated replace() must not compound drift")

    def test_asdict_roundtrip_is_idempotent(self) -> None:
        from dataclasses import asdict
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        original = self._values(cfg)
        rebuilt = self._values(AecConfig(**asdict(cfg)))
        self.assertEqual(
            rebuilt, original,
            msg="AecConfig(**asdict(cfg)) must not re-retime an "
                "already-resolved config")

    def test_pre_fix_double_retime_would_have_failed(self) -> None:
        """Mutation check: an unconditional retiming (rebase from a hardcoded 10 ms on
        every call, regardless of provenance) applied to an already-resolved config
        must visibly drift from the original, otherwise this test would not catch a
        regression.
        """
        from modules import aec3_scale as _aec3_scale
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        naive_replay = _aec3_scale.ms_to_hops(
            cfg.warmup_frames * 10.0, cfg.hop_size, cfg.sample_rate)
        self.assertNotEqual(
            naive_replay, cfg.warmup_frames,
            msg="the naive (provenance-unaware) retime should visibly "
                "drift from the correct already-resolved value -- if it "
                "doesn't, this test's tolerance is too loose to catch the "
                "bug it's named for")

    def test_sample_rate_change_via_replace_matches_fresh_construction(
            self) -> None:
        """The production path: process_wav_files() calls ``dataclasses.replace(config,
        sample_rate=mic_sr, frame_size=-1, hop_size=-1, filter_length=-1)`` when the
        caller's config was built for a different sample rate than the file being
        processed. The reconstructed config must match a config built fresh for the
        new rate, not a double-retimed drift.
        """
        from dataclasses import replace
        cfg_16k = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        reconstructed = replace(
            cfg_16k, sample_rate=48000, frame_size=-1, hop_size=-1,
            filter_length=-1)
        fresh_48k = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=48000, frame_size=1024)
        self.assertEqual(
            self._values(reconstructed), self._values(fresh_48k),
            msg="replace()-driven sample-rate change must retime "
                "identically to a fresh construction at the new rate")

    def test_fresh_construction_unaffected_by_provenance_tracking(
            self) -> None:
        """Fresh construction (a bare AecConfig() and AecConfig.from_preset()) retimes
        from the 10 ms reference, since the _canonical_* fields capture from that grid
        on first construction.
        """
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        self.assertEqual(cfg._canonical_ms_warmup_frames, 1000.0)
        # warmup_frames=100 authored hops @ 10 ms grid -> retimed to
        # the 8 ms (128-sample hop) grid: round(100*10/8) = 125.
        self.assertEqual(cfg.warmup_frames, 125)

    def test_chained_grid_changes_do_not_drift(self) -> None:
        """A long chain of reconstructions through different grids must return exactly to
        the starting values, not a rounding-drifted neighbour: ms_to_hops() rounds to
        the nearest integer hop, so each step must rederive from the pinned canonical ms
        value rather than the previous step's already-rounded hop count.
        """
        from dataclasses import replace
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=512)
        original = self._values(cfg)
        chain = [(48000, 1024), (16000, 256), (8000, 256), (16000, 512),
                 (48000, 1024), (16000, 512)]
        c = cfg
        for sr, fs in chain:
            c = replace(c, sample_rate=sr, frame_size=fs, hop_size=-1,
                        filter_length=-1)
        self.assertEqual(
            self._values(c), original,
            msg="a chain through every whitelisted grid and back must "
                "return exactly to the original values, not a "
                "rounding-drifted neighbour")

    def test_full_grid_pair_matrix_matches_fresh_construction(self) -> None:
        """Every (from-grid, to-grid) pair among the four whitelisted grids
        (16 pairs including same-grid) must retime a replace()-reconstructed
        config identically to a fresh construction at the to-grid -- not
        just the one 16k->48k pair spot-checked above."""
        from dataclasses import replace
        for sr_a, fs_a in GRIDS:
            cfg_a = AecConfig.from_preset(
                AecPreset.BALANCED, sample_rate=sr_a, frame_size=fs_a)
            for sr_b, fs_b in GRIDS:
                reconstructed = replace(
                    cfg_a, sample_rate=sr_b, frame_size=fs_b, hop_size=-1,
                    filter_length=-1)
                fresh_b = AecConfig.from_preset(
                    AecPreset.BALANCED, sample_rate=sr_b, frame_size=fs_b)
                self.assertEqual(
                    self._values(reconstructed), self._values(fresh_b),
                    msg=f"{sr_a}/{fs_a} -> {sr_b}/{fs_b}: reconstructed "
                        "config must match a fresh construction at the "
                        "target grid")

    def test_pre_fix_chained_drift_would_have_failed(self) -> None:
        """Mutation check: rebasing from the PREVIOUS grid's already-rounded hop count at
        each chain step (instead of a pinned canonical ms value) must visibly drift
        from the correct result, otherwise the chain test would not exercise anything.
        Chain: 16k/512 -> 48k/1024 -> 16k/256 -> 8k/256 -> 16k/512 on
        filter_misadjustment_stable_frames.
        """
        from modules import aec3_scale as _aec3_scale

        def naive_chain_rebase(authored_value, hop_rate_chain):
            ref_hop, ref_rate = 160, 16000  # 10 ms authoring grid
            value = authored_value
            for hop, rate in hop_rate_chain:
                ref_ms_per_hop = 1000.0 * ref_hop / ref_rate
                value = _aec3_scale.ms_to_hops(value * ref_ms_per_hop, hop, rate)
                ref_hop, ref_rate = hop, rate
            return value

        chain = [(256, 16000), (512, 48000), (128, 16000), (128, 8000),
                 (256, 16000)]
        naive_result = naive_chain_rebase(30, chain)  # dataclass default
        correct_result = _aec3_scale.ms_to_hops(30 * 10.0, 256, 16000)
        self.assertNotEqual(
            naive_result, correct_result,
            msg="the naive previous-grid-rebase chain should visibly "
                "drift from the correct canonical-rebase result -- if it "
                "doesn't, this mutation check's numbers don't exercise "
                "the bug it's named for")


if __name__ == '__main__':
    unittest.main()
