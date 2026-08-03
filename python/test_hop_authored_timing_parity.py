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


class ConfigReconstructionIdempotencyTests(unittest.TestCase):
    """Regression test for the reconstruction-idempotency bug (2026-08-04
    gap-fix, follow-up to the retiming fix above): __post_init__ used to
    treat WHATEVER value sat in a retimed field as still "authored at the
    legacy 10 ms grid" on every run, so rebuilding an already-resolved
    config via ``dataclasses.replace()``/``AecConfig(**asdict(cfg))`` (e.g.
    process_wav_files()'s sample-rate-mismatch re-resolve path at
    modules/orchestrator.py) silently double-retimed the six wall-clock
    fields, compounding further on every additional reconstruction.

    Fix: timing_reference_hop_size/_sample_rate track which grid the
    current field values are calibrated for; retiming is skipped when that
    already matches the resolved grid, and rebased from the tracked
    reference (not a hardcoded 10 ms) when it doesn't.
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
        """Sanity-check: mutation test. Simulating the OLD unconditional
        retiming (rebase from a hardcoded 10 ms every call, regardless of
        provenance) on an already-resolved config must visibly drift from
        the original -- otherwise this test wouldn't catch a regression."""
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
        """The actual production path this bug lives on:
        process_wav_files() calls ``dataclasses.replace(config,
        sample_rate=mic_sr, frame_size=-1, hop_size=-1, filter_length=-1)``
        when the caller's config was built for a different sample rate than
        the file being processed. The reconstructed config must match a
        config built fresh for the new rate, not a double-retimed drift."""
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
        """The idempotency fix must be behavior-preserving for the common
        (fresh-construction) path: a bare AecConfig() and an
        AecConfig.from_preset() must retime exactly as before (10 ms
        reference), since the _canonical_* fields capture from the legacy
        grid on first construction."""
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED, sample_rate=16000, frame_size=256)
        self.assertEqual(cfg._canonical_ms_warmup_frames, 1000.0)
        # warmup_frames=100 authored hops @ legacy 10 ms grid -> retimed to
        # the 8 ms (128-sample hop) grid: round(100*10/8) = 125.
        self.assertEqual(cfg.warmup_frames, 125)

    def test_chained_grid_changes_do_not_drift(self) -> None:
        """Regression test (2026-08-04, same-day follow-up): the FIRST
        version of the idempotency fix tracked "which grid is the CURRENT
        value calibrated for" and rebased from THAT -- correct for repeated
        same-grid reconstruction, but still lossy across a CHAIN of
        different grids, because ms_to_hops() rounds to the nearest integer
        hop and each chain step rebased from the previous step's
        already-rounded hop count instead of the original canonical ms
        value (confirmed by direct measurement: 16k/512 -> 48k/1024 ->
        16k/512 landed one hop off the true original in that version).
        Fixed by pinning the canonical value once and always rederiving
        from it. A long chain through every whitelisted grid must return
        exactly to the starting values, not a rounding-drifted neighbour."""
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
        """Mutation check: simulate the FIRST version's bug (rebase from
        the PREVIOUS grid's already-rounded hop count at each chain step,
        instead of a pinned canonical ms value) directly and confirm it
        visibly drifts from the correct (canonical) result -- otherwise the
        chain test above wouldn't actually be exercising anything. Uses the
        exact chain confirmed (by direct measurement) to drift:
        filter_misadjustment_stable_frames 19 -> 18 through
        16k/512 -> 48k/1024 -> 16k/256 -> 8k/256 -> 16k/512."""
        from modules import aec3_scale as _aec3_scale

        def naive_chain_rebase(authored_value, hop_rate_chain):
            ref_hop, ref_rate = 160, 16000  # legacy 10 ms authoring grid
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
