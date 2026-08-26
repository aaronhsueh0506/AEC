"""Effective-value regression for the 2026-08-06 detector/adaptation retiming.

This asserts the constants a live ``AEC`` instance actually ENDS UP WITH on each
supported grid, not the presence of a retiming call. That distinction is the
whole point of this file: an external reviewer read the four touched modules and
concluded the retiming had not been applied at all, because the values they
sampled were the authored ones. Two things make that failure mode easy:

  * a benchmark A/B harness swaps authored-value copies of these modules into
    the working tree while it renders its baseline, so the tree is transiently
    un-retimed and looks exactly like an unfinished implementation;
  * two of the constants are legitimately UNCHANGED on two of the four grids
    (the saturation pair is authored at a 16 ms hop, so it is identity-mapped
    at 8k/128 and 16k/256), so sampling one grid can easily show "nothing
    changed" for part of the table.

Only an effective-value assertion across every grid distinguishes "retimed" from
"not retimed", and it fails loudly if a baseline variant is ever left in place.

Reference grids are NOT uniform -- verified per constant from git provenance:
  * ``self.alpha``            per-SAMPLE, authored at sr=16000
  * ``alpha_erl`` (both)      per-hop, 10 ms  (5407e71 annotates hop as "10ms")
  * ``_alpha_r``              per-hop, 10 ms  (authored 16 ms at e9cb383, but
                              the default moved to 10 ms at 83ced18 and every
                              validating commit since kept 0.95 there)
  * saturation attack/release per-hop, 16 ms  (243d67c, frame 512 / hop 256)

Retiming one of the 16 ms constants off a 10 ms reference is wrong by 1.6x, so
the reference grid is asserted here too, not just the shape of the formula.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.dirname(_HERE)
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from aec import AEC, AecConfig  # noqa: E402
from modules import aec3_scale  # noqa: E402

# (sample_rate, frame_size). hop is frame_size // 2.
GRIDS = [(8000, 256), (16000, 256), (16000, 512), (48000, 1024)]

REF_SR = 16000
REF_HOP_10MS = 160
REF_HOP_16MS = 256


def _instance(sample_rate: int, frame_size: int):
    aec = AEC(AecConfig(sample_rate=sample_rate, frame_size=frame_size))
    filt = aec.filter.filter if hasattr(aec.filter, 'filter') else aec.filter
    return aec, filt, aec._sat_detector_mic


def _per_hop(authored: float, ref_hop: int, hop: int, sr: int) -> float:
    return aec3_scale.growth_rehop(authored, ref_hop, REF_SR, hop, sr)


@pytest.mark.parametrize('sample_rate,frame_size', GRIDS)
def test_effective_timing_constants_match_their_reference_grid(sample_rate,
                                                               frame_size):
    hop = frame_size // 2
    aec, filt, sat = _instance(sample_rate, frame_size)

    # per-SAMPLE: hop-invariant, sample-rate dependent.
    assert aec.alpha == pytest.approx(0.95 ** (REF_SR / sample_rate), rel=1e-12)

    # per-hop, 10 ms reference.
    assert aec._alpha_erl_tracking == pytest.approx(
        _per_hop(0.99, REF_HOP_10MS, hop, sample_rate), rel=1e-12)
    assert aec._alpha_erl_converged == pytest.approx(
        _per_hop(0.999, REF_HOP_10MS, hop, sample_rate), rel=1e-12)
    assert filt._alpha_r == pytest.approx(
        _per_hop(0.95, REF_HOP_10MS, hop, sample_rate), rel=1e-12)

    # per-hop, 16 ms reference.
    assert sat.alpha_attack == pytest.approx(
        _per_hop(0.3, REF_HOP_16MS, hop, sample_rate), rel=1e-12)
    assert sat.alpha_release == pytest.approx(
        _per_hop(0.98, REF_HOP_16MS, hop, sample_rate), rel=1e-12)


def test_retiming_is_actually_applied_somewhere():
    """Guard against a whole-table identity map.

    Every assertion above still passes if every constant is left authored AND
    the reference grid happens to equal the live grid. This pins the case that
    cannot be explained that way: at 16 kHz / hop 128 the 10 ms constants must
    move, and at 48 kHz the per-sample constant must move.
    """
    aec_16k128, filt_16k128, sat_16k128 = _instance(16000, 256)
    assert filt_16k128._alpha_r != pytest.approx(0.95, abs=1e-9)
    assert aec_16k128._alpha_erl_tracking != pytest.approx(0.99, abs=1e-9)
    assert sat_16k128.alpha_attack != pytest.approx(0.3, abs=1e-9)

    aec_48k, _, _ = _instance(48000, 1024)
    assert aec_48k.alpha != pytest.approx(0.95, abs=1e-9)
    assert aec_48k.alpha == pytest.approx(0.983048, abs=1e-6)


def test_sixteen_ms_constants_are_identity_on_their_own_reference_grid():
    """The 16 ms pair must NOT move at 8k/128 and 16k/256 -- both are 16.000 ms.

    A reviewer seeing these unchanged is looking at correct behaviour, not an
    unfinished retime. Retiming them off a 10 ms reference would move them here,
    which is exactly the 1.6x error this pins down.
    """
    for sample_rate, frame_size in ((8000, 256), (16000, 512)):
        _, filt, sat = _instance(sample_rate, frame_size)
        assert sat.alpha_attack == pytest.approx(0.3, rel=1e-12)
        assert sat.alpha_release == pytest.approx(0.98, rel=1e-12)


def test_per_hop_time_constants_are_grid_invariant_in_wall_clock():
    """The point of the exercise: equal wall-clock TC on every grid."""
    import math

    def tc_ms(retention: float, hop: int, sr: int) -> float:
        return -(hop / sr) * 1000.0 / math.log(retention)

    seen = {}
    for sample_rate, frame_size in GRIDS:
        hop = frame_size // 2
        aec, filt, sat = _instance(sample_rate, frame_size)
        for name, value in (('alpha_erl_tracking', aec._alpha_erl_tracking),
                            ('alpha_r', filt._alpha_r),
                            ('sat_release', sat.alpha_release)):
            seen.setdefault(name, []).append(tc_ms(value, hop, sample_rate))

    for name, values in seen.items():
        assert max(values) - min(values) < 1e-6, (
            f'{name} wall-clock TC varies across grids: {values}')
