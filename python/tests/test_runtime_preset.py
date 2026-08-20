"""Runtime retarget of the residual-echo strength axis (``AEC.set_preset``).

Why this test exists: the three shipped presets differ in exactly one field,
``min_gain_floor_far_active_db`` (``AecConfig.from_preset``), and that field
reaches the suppressor as a single scalar clamp read once per hop. So a preset
change has never actually needed a rebuild — but until now the only way to get
one was to construct a new AEC, which throws away the filter, the noise
estimate and the whole smoothing history. An integrator exposing a strength
control had no honest option.

The floor is a HARD clamp with nothing downstream to smooth it, and
mild -> aggressive is an 18 dB step, so the retarget also has to be able to
walk there instead of jumping. That walk is geometric in the power domain,
which is linear in dB, so no logarithm is evaluated per hop.

What is asserted:
  1. THE DEFAULT PATH IS UNTOUCHED — an instance whose setter is never called
     produces bit-identical output to one built before the ramp state existed.
     Covered structurally here by ``test_untouched_instance_matches_target``:
     the live floor and the target stay equal, which is the condition under
     which ``_get_min_gain`` reads exactly what it read before.
  2. ``ramp_ms=0`` LANDS ON THE CONSTRUCTED VALUE — bit-identical to a fresh
     instance built with that preset. This is the property that makes the
     setter a true preset change rather than an approximation of one.
  3. ``ramp_ms>0`` IS MONOTONIC AND LANDS EXACTLY — no overshoot, and no
     1-ULP residue that would leave the ramp nominally live forever.
  4. A CALL DURING A RAMP RESTARTS FROM THE CURRENT LIVE VALUE — it does not
     snap back to the old target first, and does not inherit the old ratio.
  5. REJECTION IS TOTAL — an out-of-range argument leaves the instance
     bit-identical, so a caller cannot half-apply a strength change.

``test_mutation_*`` proves the landing assertion can fail: with the exact-
landing branch removed, the geometric walk leaves a residue and never reports
equality with the target.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules import aec3_scale
from modules.config import AecConfig
from modules.enums import AecPreset
from modules.orchestrator import AEC
from modules.residual.suppression_gain import SuppressionGain


PRESET_DB = {
    AecPreset.MILD: -20.0,
    AecPreset.BALANCED: -28.0,
    AecPreset.AGGRESSIVE: -38.0,
}


def _sg(db: float = -28.0, sr: int = 16000, hop: int = 128) -> SuppressionGain:
    return SuppressionGain(n_bins=129, sr=sr, hop_size=hop,
                           split_floor_far_active_db=db)


def _live(sg: SuppressionGain) -> float:
    return sg._split_floor_far_active_live


def _target(sg: SuppressionGain) -> float:
    return sg._split_floor_far_active


# ── 1. default path ──────────────────────────────────────────────────────

def test_untouched_instance_matches_target():
    sg = _sg()
    assert _live(sg) == _target(sg)
    # Advancing the ramp on an untouched instance must be a no-op, hop after
    # hop -- this is what keeps every pre-existing golden valid.
    for _ in range(64):
        sg._advance_split_floor_ramp()
        assert _live(sg) == _target(sg)


# ── 2. ramp_ms = 0 lands on the constructed value ────────────────────────

@pytest.mark.parametrize("preset", list(PRESET_DB))
def test_immediate_matches_fresh_construction(preset):
    db = PRESET_DB[preset]
    fresh = _sg(db)
    sg = _sg(-28.0)
    sg.set_split_floor_far_active_db(db, 0.0)
    assert _live(sg) == _live(fresh)
    assert _target(sg) == _target(fresh)


def test_set_preset_mirrors_config_and_floor():
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    aec.set_preset(AecPreset.AGGRESSIVE)
    assert aec.config.min_gain_floor_far_active_db == PRESET_DB[AecPreset.AGGRESSIVE]
    assert _live(aec._aec3_sg) == _live(_sg(PRESET_DB[AecPreset.AGGRESSIVE]))


# ── 3. ramp_ms > 0 is monotonic and lands exactly ────────────────────────

def test_ramp_is_monotonic_and_lands_exactly():
    sr, hop, ramp_ms = 16000, 128, 100.0
    hops = int(round(ramp_ms * 1e-3 * sr / hop))
    sg = _sg(-20.0, sr=sr, hop=hop)
    start = _live(sg)
    sg.set_split_floor_far_active_db(-38.0, ramp_ms)
    target = _target(sg)
    assert target < start, "aggressive must be a deeper floor than mild"
    assert _live(sg) == start, "the setter itself must not move the live value"

    prev = start
    for _ in range(hops):
        sg._advance_split_floor_ramp()
        cur = _live(sg)
        assert cur <= prev, "descending ramp must never rise"
        assert cur >= target, "ramp must never overshoot the target"
        prev = cur
    assert _live(sg) == target, "ramp must land EXACTLY, with no residue"

    # Once landed it must stay landed.
    for _ in range(16):
        sg._advance_split_floor_ramp()
        assert _live(sg) == target


def test_ramp_upward_is_monotonic_and_lands_exactly():
    sg = _sg(-38.0)
    start = _live(sg)
    sg.set_split_floor_far_active_db(-20.0, 100.0)
    target = _target(sg)
    assert target > start
    prev = start
    for _ in range(200):
        sg._advance_split_floor_ramp()
        cur = _live(sg)
        assert cur >= prev
        assert cur <= target
        prev = cur
    assert _live(sg) == target


# ── 4. a call during a ramp restarts from the live value ─────────────────

def test_reset_during_ramp_restarts_from_live():
    """A retarget mid-ramp continues from where the walk actually got to.

    The second leg is chosen to REVERSE direction: three hops of a
    -20 -> -38 dB ramp only reach about -24.5 dB, so retargeting to -20 dB
    means the walk has to climb back. A restart that snapped to the old
    target, or that inherited the old descending ratio, would fail here.
    """
    sg = _sg(-20.0)
    start = _live(sg)
    sg.set_split_floor_far_active_db(-38.0, 100.0)
    for _ in range(3):
        sg._advance_split_floor_ramp()
    mid = _live(sg)
    assert mid < start, "the ramp must have moved before we retarget"

    sg.set_split_floor_far_active_db(-20.0, 100.0)
    assert _live(sg) == mid, "retarget must start from the CURRENT live value"
    assert _target(sg) > mid, "this leg must genuinely reverse direction"
    prev = mid
    for _ in range(200):
        sg._advance_split_floor_ramp()
        cur = _live(sg)
        assert cur >= prev, "the reversed leg must ascend"
        assert cur <= _target(sg), "ramp must never overshoot the target"
        prev = cur
    assert _live(sg) == _target(sg) == start


def test_immediate_call_during_ramp_lands_now():
    sg = _sg(-20.0)
    sg.set_split_floor_far_active_db(-38.0, 100.0)
    sg._advance_split_floor_ramp()
    sg.set_split_floor_far_active_db(-28.0, 0.0)
    assert _live(sg) == _target(sg) == _live(_sg(-28.0))


# ── 5. rejection is total ────────────────────────────────────────────────

@pytest.mark.parametrize("db,ramp_ms", [
    (float("nan"), 0.0),
    (float("inf"), 0.0),
    (-301.0, 0.0),
    (51.0, 0.0),
    (-28.0, float("nan")),
    (-28.0, -1.0),
    (-28.0, 60001.0),
])
def test_rejected_arguments_leave_state_untouched(db, ramp_ms):
    sg = _sg(-20.0)
    before = (_live(sg), _target(sg), sg._split_floor_ramp_ratio)
    with pytest.raises(ValueError):
        sg.set_split_floor_far_active_db(db, ramp_ms)
    assert (_live(sg), _target(sg), sg._split_floor_ramp_ratio) == before


def test_set_preset_rejects_unknown_preset():
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    before = aec.config.min_gain_floor_far_active_db
    with pytest.raises(ValueError):
        aec.set_preset("not-a-preset")
    assert aec.config.min_gain_floor_far_active_db == before


# ── mutation: the exact-landing branch must be load-bearing ──────────────

def test_mutation_without_exact_landing_the_ramp_never_lands(monkeypatch):
    """Remove the snap-to-target branch and the landing assertion must fail.

    Without it the geometric walk approaches the target asymptotically in
    floating point and ``_live == _target`` is not reached, so the ramp stays
    nominally active forever. This proves assertion 3 can fail.
    """
    def _advance_without_landing(self):
        live = self._split_floor_far_active_live
        target = self._split_floor_far_active
        if live == target:
            return
        self._split_floor_far_active_live = live * self._split_floor_ramp_ratio

    monkeypatch.setattr(SuppressionGain, "_advance_split_floor_ramp",
                        _advance_without_landing)
    sg = _sg(-20.0)
    sg.set_split_floor_far_active_db(-38.0, 100.0)
    hops = int(round(100.0 * 1e-3 * 16000 / 128))
    for _ in range(hops):
        sg._advance_split_floor_ramp()
    assert _live(sg) != _target(sg), (
        "mutation did not take effect -- the exact-landing branch is not "
        "actually what makes the ramp terminate")


# ── the setter must not disturb anything else ────────────────────────────

def test_setter_leaves_smoothing_and_latch_untouched():
    sg = _sg(-20.0)
    sg._far_active_latched = True
    sg._last_gain = np.full(129, 0.5, dtype=np.float32)
    last_gain = sg._last_gain.copy()
    sg.set_split_floor_far_active_db(-38.0, 100.0)
    for _ in range(8):
        sg._advance_split_floor_ramp()
    assert sg._far_active_latched is True
    assert np.array_equal(sg._last_gain, last_gain)


# ── C parity: the ms -> hops conversion ──────────────────────────────────

# The SAME pairs the C test carries (c_impl/test/test_runtime_preset.c,
# run_half_hop_rounding_parity). Authored at 16 kHz / hop 128. These are
# measured, not derived: the float expression does not land on the clean half
# these ms values look like, which is exactly what the table has to pin.
C_RAMP_HOPS = [
    (12.0, 2),   # float expr is exactly 1.5   -> 2   (truncation would say 1)
    (20.0, 3),   # float expr is 2.5000002     -> 3   (a DOUBLE would say 2)
    (28.0, 4),   # float expr is exactly 3.5   -> 4   (truncation would say 3)
    (24.0, 3),   # whole hop, control
    (40.0, 5),   # float expr is 5.0000005
    (56.0, 7),   # whole hop, control
]


@pytest.mark.parametrize("ramp_ms,hops", C_RAMP_HOPS)
def test_ramp_hops_matches_the_c_conversion(ramp_ms, hops):
    """The reference must agree with the shipping C on the hop count.

    Both round half to even, so the rule was never the risk -- the precision
    was. Computing this in double, as the rest of this file naturally would,
    disagrees with the C on every ramp_ms landing near a half hop (20 ms is
    the case in this table: 2.5 in double, 2.5000002 in float).
    """
    assert aec3_scale.ms_to_hops_f32(ramp_ms, 128, 16000) == hops


def test_ramp_hops_never_returns_zero():
    assert aec3_scale.ms_to_hops_f32(0.001, 128, 16000) == 1
    assert aec3_scale.ms_to_hops_f32(0.0, 128, 16000) == 0


# ── the setter must reject what the C setter rejects ─────────────────────

@pytest.mark.parametrize("bad", [99, None, -1, "gentle", object(), 3.5])
def test_set_preset_rejects_non_presets(bad):
    """AecConfig.from_preset falls back to balanced for anything unknown --
    correct for a config factory, wrong for a setter. Without an explicit
    coercion at the entry point, 99 and None were both silently accepted."""
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    before = aec.config.min_gain_floor_far_active_db
    live_before = _live(aec._aec3_sg)
    with pytest.raises(ValueError):
        aec.set_preset(bad)
    assert aec.config.min_gain_floor_far_active_db == before
    assert _live(aec._aec3_sg) == live_before


@pytest.mark.parametrize("good", [AecPreset.MILD, "mild", "aggressive"])
def test_set_preset_accepts_enum_and_its_string_spelling(good):
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    aec.set_preset(good)
    assert aec.config.min_gain_floor_far_active_db == PRESET_DB[AecPreset(good)]
