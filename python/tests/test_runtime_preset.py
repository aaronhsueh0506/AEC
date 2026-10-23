"""Runtime retarget of the residual-echo strength axis (``AEC.set_preset``).

mild/balanced differ in exactly one field, ``min_gain_floor_far_active_db``;
aggressive also differs in ``min_gain_floor_dt_db`` (``AecConfig.from_preset``).
Both reach the suppressor as a single scalar clamp each, read once per hop, so a
preset change needs no rebuild. The floor is a HARD clamp and mild -> aggressive
is an 18 dB step, so the retarget walks there geometrically in the power domain
(linear in dB, no logarithm per hop) instead of jumping.

Asserted: an instance whose setter is never called keeps live floor == target
(``test_untouched_instance_matches_target``); ``ramp_ms=0`` is bit-identical to
a fresh instance built with that preset; ``ramp_ms>0`` is monotonic and lands
exactly (no overshoot, no 1-ULP residue); a call during a ramp restarts from the
current live value; an out-of-range argument leaves the instance bit-identical.
``test_mutation_*`` proves the landing assertion can fail.
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


def test_direct_suppressor_dt_floor_matches_aec_default():
    """Direct users and the top-level AEC must not silently get two floors."""
    sg = SuppressionGain(n_bins=129, sr=16000, hop_size=128)
    db = np.float32(AecConfig().min_gain_floor_dt_db)
    expected = float(np.float32(10.0) ** (db / np.float32(10.0)))
    assert sg._split_floor_dt == expected


# ── 1. default path ──────────────────────────────────────────────────────

def test_untouched_instance_matches_target():
    sg = _sg()
    assert _live(sg) == _target(sg)
    # Advancing the ramp on an untouched instance must be a no-op, hop after hop.
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


def _dt_live(sg) -> float:
    return sg._split_floor_dt


def test_set_preset_steps_dt_floor_immediately_even_with_a_ramp():
    """aggressive is the only preset whose dt floor differs from the shared
    -20 dB default. The far-active floor ramps over ramp_ms; the dt floor has
    no ramp state at all and must already be at its target the instant
    set_preset returns, regardless of ramp_ms."""
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    assert aec.config.min_gain_floor_dt_db == -20.0
    aec.set_preset(AecPreset.AGGRESSIVE, ramp_ms=5000.0)
    assert aec.config.min_gain_floor_dt_db == -30.0
    expected = float(np.float32(10.0) ** (np.float32(-30.0) / np.float32(10.0)))
    assert _dt_live(aec._aec3_sg) == expected
    # far-active is still mid-ramp (5 s requested) while dt already landed.
    assert _live(aec._aec3_sg) != _live(_sg(PRESET_DB[AecPreset.AGGRESSIVE]))

    aec.set_preset(AecPreset.MILD)
    assert aec.config.min_gain_floor_dt_db == -20.0


def test_set_split_floor_dt_db_rejects_out_of_range():
    sg = _sg(-20.0)
    before = _dt_live(sg)
    with pytest.raises(ValueError):
        sg.set_split_floor_dt_db(float("nan"))
    assert _dt_live(sg) == before
    with pytest.raises(ValueError):
        sg.set_split_floor_dt_db(-301.0)
    assert _dt_live(sg) == before


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


def test_set_preset_allows_no_floor():
    """NO_FLOOR changes min_gain_split_floor_enabled -- a boolean the ramped scalar
    setters cannot represent on their own, so the gate has its own explicit setter
    (set_split_floor_enabled), stepped immediately rather than inferred from the dB
    setters. The retarget lands with the gate False and the far-active dB value
    whatever NO_FLOOR's own config carries (inert once the gate is off, but still
    tracked in self.config so a later retarget AWAY from NO_FLOOR has a consistent
    value to read).
    """
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    no_floor_cfg = AecConfig.from_preset(AecPreset.NO_FLOOR)
    aec.set_preset(AecPreset.NO_FLOOR)
    assert aec.config.min_gain_split_floor_enabled is False
    assert aec.config.min_gain_floor_far_active_db == no_floor_cfg.min_gain_floor_far_active_db
    assert aec.config.min_gain_floor_dt_db == no_floor_cfg.min_gain_floor_dt_db
    assert aec._aec3_sg._split_floor_enabled is False


def test_set_preset_allows_retargeting_away_from_no_floor():
    """The symmetric direction: an instance CONSTRUCTED with NO_FLOOR can be
    retargeted back to a ramped preset, and must land on BOTH that preset's gate
    (True) AND its two dB fields -- never a mix of the two presets' behavior.
    """
    aec = AEC(AecConfig.from_preset(AecPreset.NO_FLOOR))
    assert aec.config.min_gain_split_floor_enabled is False
    balanced_cfg = AecConfig.from_preset(AecPreset.BALANCED)
    aec.set_preset(AecPreset.BALANCED)
    assert aec.config.min_gain_split_floor_enabled is True
    assert aec.config.min_gain_floor_far_active_db == balanced_cfg.min_gain_floor_far_active_db
    assert aec.config.min_gain_floor_dt_db == balanced_cfg.min_gain_floor_dt_db
    assert aec._aec3_sg._split_floor_enabled is True


def test_set_preset_no_floor_round_trip():
    """BALANCED -> NO_FLOOR -> AGGRESSIVE, each step checked against a fresh instance
    of the target preset (gate and scalars never disagree).
    """
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))

    def _assert_matches(preset):
        fresh = AecConfig.from_preset(preset)
        assert aec.config.min_gain_split_floor_enabled == fresh.min_gain_split_floor_enabled
        assert aec.config.min_gain_floor_far_active_db == fresh.min_gain_floor_far_active_db
        assert aec.config.min_gain_floor_dt_db == fresh.min_gain_floor_dt_db
        assert aec._aec3_sg._split_floor_enabled == fresh.min_gain_split_floor_enabled

    aec.set_preset(AecPreset.NO_FLOOR)
    _assert_matches(AecPreset.NO_FLOOR)
    aec.set_preset(AecPreset.AGGRESSIVE)
    _assert_matches(AecPreset.AGGRESSIVE)


def test_set_preset_no_floor_gate_steps_without_ramp():
    """The gate is written through its OWN setter, not inferred: if set_preset forgot
    to call set_split_floor_enabled, the gate would silently stay at whatever it was
    before. ramp_ms must not affect the gate at all -- it is not a scalar.
    """
    aec = AEC(AecConfig.from_preset(AecPreset.BALANCED))
    aec.set_preset(AecPreset.NO_FLOOR, ramp_ms=5000.0)
    assert aec.config.min_gain_split_floor_enabled is False
    assert aec._aec3_sg._split_floor_enabled is False


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
    """AecConfig.from_preset falls back to balanced for anything unknown -- correct
    for a config factory, wrong for a setter, which must coerce at the entry point
    so that 99 and None are rejected.
    """
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
