"""F2.4 regression: the mu holdoff counter must arm only on a FRESH DT onset.

This test exists because the behaviour it pins was silently reverted and stayed
reverted in shipped code for ten weeks (2f3699f, 2026-05-27 -> 2026-08-06).

What happened: F2.4 (7b2cf04, an 800-case ablation, CONDITIONAL PASS ratio 0.9x
worst -0.204) added a guard so that ongoing double-talk does not re-arm the
holdoff counter -- without it, marginal-DT oscillation re-arms every hop and the
adaptation step size never releases. It shipped behind
`mu_holdoff_no_reset`, which lived in `AecConfig._LEGACY_HARDCODE_TRUE`, so the
live expression was:

    if not True or self._simple_mu_holdoff == 0:   ==   if self._simple_mu_holdoff == 0:

A later "remove dead flag branches" cleanup collapsed that line to an
UNCONDITIONAL assignment -- i.e. it kept the arm the flag had disabled. The
comment above the line went on describing the guard, so reading the source
looked correct; only the executed branch was wrong. The C port mirrored the
same collapse.

Nothing caught it: the counter is internal state, no test asserted on it, and
the AECMOS effect of a mu freeze under marginal DT is not large enough to break
a bucket average. Hence an explicit state-level assertion here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.dirname(_HERE)
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from aec import AEC, AecConfig  # noqa: E402

FROZEN_HOLDOFF = 20
FROZEN_ALPHA = {"attack": 0.3, "hold": 0.99, "release": 0.95}


def _aec(sample_rate=16000, frame_size=None):
    kw = {"sample_rate": sample_rate}
    if frame_size is not None:
        kw["frame_size"] = frame_size
    return AEC(AecConfig(**kw))


def _runout(aec):
    """Long enough to observe arm, drain, and release."""
    return 2 * FROZEN_HOLDOFF + 10


def _drive(aec, n_hops, far_scale, err_scale, rng):
    """Feed n_hops of (output, far) straight into the holdoff update.

    Driving the private method rather than full frames is deliberate: the whole
    defect is one branch inside it, and a full-pipeline test would have to get
    the AEC into sustained marginal double-talk to reach that branch at all --
    which is exactly why the regression survived a full test suite.
    """
    hop = aec.config.hop_size
    seen = []
    for _ in range(n_hops):
        far = far_scale * rng.standard_normal(hop).astype(np.float64)
        out = err_scale * rng.standard_normal(hop).astype(np.float64)
        aec._update_simple_mu_ratio(out, far)
        seen.append(aec._simple_mu_holdoff)
    return seen


def _force_attack(aec):
    """Put the instance into a state where the next update takes the attack
    branch: ratio must come out below the current _simple_mu_ratio."""
    aec._simple_mu_ratio = 1.0
    aec._simple_mu_holdoff = 0


def test_holdoff_arms_on_a_fresh_onset():
    """Baseline: the mechanism still works at all."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(1)
    # error >> far  ->  ratio low  ->  attack branch
    _drive(aec, 1, far_scale=0.001, err_scale=0.5, rng=rng)
    assert aec._simple_mu_holdoff > 0, "a fresh DT onset must arm the holdoff"


def test_ongoing_doubletalk_does_not_rearm_the_holdoff():
    """THE regression. Sustained attack-branch hops must let the counter run
    down, not pin it at its armed value."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(2)

    seen = _drive(aec, _runout(aec), far_scale=0.001, err_scale=0.5, rng=rng)
    armed = seen[0]
    assert armed > 0, "setup failed: never armed"

    # With the guard, the counter is armed once and then decrements every hop.
    # Without it, every hop re-arms and the sequence is constant at `armed`.
    assert min(seen) < armed, (
        f"holdoff never decremented across {len(seen)} sustained DT hops "
        f"(stuck at {armed}) -- the F2.4 guard is missing, so ongoing "
        f"double-talk re-arms the counter and mu can never release"
    )
    # Reaching zero AT SOME POINT is the claim; `seen[-1] == 0` was the old
    # form and depended on where in an arm/run-down cycle the trace happened to
    # stop -- luck that held while the counter was 20 and stopped holding when
    # it became 25.
    assert 0 in seen, (
        f"holdoff never reached 0 across {len(seen)} hops: {seen[-8:]}"
    )
    assert armed == FROZEN_HOLDOFF


def test_holdoff_can_rearm_after_it_has_expired():
    """The guard must not weld the counter off: once it reaches 0, a new onset
    arms it again. A fix that simply deleted the assignment would pass the test
    above and fail this one."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(3)

    # Drive until the counter is actually spent, rather than for a fixed number
    # of hops and hoping the trace ends at the bottom of a cycle.
    for _ in range(_runout(aec)):
        _drive(aec, 1, far_scale=0.001, err_scale=0.5, rng=rng)
        if aec._simple_mu_holdoff == 0:
            break
    assert aec._simple_mu_holdoff == 0, "setup failed: never ran down"

    aec._simple_mu_ratio = 1.0          # fresh onset
    again = _drive(aec, 1, far_scale=0.001, err_scale=0.5, rng=rng)
    assert again[0] == FROZEN_HOLDOFF, (
        "a new onset after expiry must re-arm the frozen holdoff")


def test_holdoff_never_rearms_before_the_first_expiry():
    """Pins one arm/run-down cycle, not a later independent onset."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(4)
    seen = _drive(aec, _runout(aec), far_scale=0.001, err_scale=0.5, rng=rng)
    first_zero = seen.index(0)
    for i in range(1, first_zero + 1):
        assert seen[i] <= seen[i - 1], (
            f"holdoff increased at hop {i}: {seen[i - 1]} -> {seen[i]}; "
            f"full trace {seen}"
        )

# ── deterministic state transitions ──────────────────────────────────────────
# The tests above drive the update with pseudo-random frames and assert on the
# SHAPE of the resulting trace. That is real integration coverage, but it is
# not a specification: it depends on the stimulus happening to hold the attack
# branch. These three pin the invariant directly, with the branch forced.


def _force(aec, *, attack):
    """Put the instance one call away from a chosen branch.

    The branch is `ratio < self._simple_mu_ratio`. `ratio` is derived from the
    far/error power ratio, so pinning _simple_mu_ratio to an extreme makes the
    comparison deterministic regardless of what the frame contains.
    """
    aec._simple_mu_ratio = 1.0 if attack else 0.0


def _step(aec, rng):
    hop = aec.config.hop_size
    far = 0.001 * rng.standard_normal(hop)
    out = 0.500 * rng.standard_normal(hop)
    aec._update_simple_mu_ratio(out, far)


def test_attack_while_holdoff_nonzero_leaves_it_unchanged():
    """THE invariant. An ongoing attack must not restart the counter."""
    aec = _aec()
    rng = np.random.default_rng(11)
    aec._simple_mu_holdoff = 7
    _force(aec, attack=True)
    _step(aec, rng)
    assert aec._simple_mu_holdoff == 7, (
        f"attack with holdoff=7 changed it to {aec._simple_mu_holdoff}; "
        f"an ongoing attack must neither re-arm nor decrement"
    )


def test_non_attack_decrements_the_holdoff():
    aec = _aec()
    rng = np.random.default_rng(12)
    aec._simple_mu_holdoff = 7
    _force(aec, attack=False)
    _step(aec, rng)
    assert aec._simple_mu_holdoff == 6, (
        f"non-attack hop left holdoff at {aec._simple_mu_holdoff}, expected 6"
    )


def test_attack_after_expiry_rearms():
    aec = _aec()
    rng = np.random.default_rng(13)
    aec._simple_mu_holdoff = 0
    _force(aec, attack=True)
    _step(aec, rng)
    assert aec._simple_mu_holdoff > 0, (
        "a fresh attack with holdoff=0 must re-arm; a guard that simply "
        "deleted the assignment would pass the first test and fail here"
    )


# ── rejected retime: lock the frozen mechanism on every grid ──────────
# The holdoff and three alphas form one state machine. Their wall-clock retime
# failed the two-grid blind A/B gate, so production intentionally keeps the
# validated hop-authored literals together.

GRIDS = [(8000, 256), (16000, 256), (16000, 512), (48000, 1024)]
@pytest.mark.parametrize("sample_rate,frame_size", GRIDS)
def test_simple_mu_holdoff_stays_frozen_on_every_grid(sample_rate, frame_size):
    aec = _aec(sample_rate, frame_size)
    _force_attack(aec)
    seen = _drive(aec, 1, 0.001, 0.5, np.random.default_rng(21))
    assert seen == [FROZEN_HOLDOFF]


def test_default_grid_distinguishes_frozen_from_retimed_policy():
    from modules import aec3_scale

    aec = _aec(16000, 256)
    hop = aec.config.hop_size
    assert aec3_scale.ms_to_hops(200.0, hop, 16000) != FROZEN_HOLDOFF
    for authored in FROZEN_ALPHA.values():
        retimed = aec3_scale.growth_rehop(authored, 160, 16000, hop, 16000)
        assert retimed != pytest.approx(authored, abs=1e-9)


@pytest.mark.parametrize("branch,holdoff", [("release", 0), ("hold", 5)])
def test_hold_and_release_apply_the_frozen_coefficients(branch, holdoff):
    """Recover the coefficient actually applied at the use site. With far >>
    error the incoming ratio saturates at 1.0, making it invertible:
    ratio_new = a*ratio_old + (1-a)*1.0  ->  a = (ratio_new - 1) / (ratio_old - 1).
    """
    aec = _aec(16000, 256)
    want = FROZEN_ALPHA[branch]
    start = 0.4                      # below 1.0, so the non-attack side is taken
    aec._simple_mu_holdoff = holdoff
    aec._simple_mu_ratio = start

    rng = np.random.default_rng(22)
    hop = aec.config.hop_size
    far = 0.5 * rng.standard_normal(hop)
    out = 0.001 * rng.standard_normal(hop)
    aec._update_simple_mu_ratio(out, far)

    applied = (aec._simple_mu_ratio - 1.0) / (start - 1.0)
    assert applied == pytest.approx(want, abs=2e-6), (
        f"{branch} branch applied {applied:.6f}, expected the frozen "
        f"{want:.6f}")


def test_reset_clears_simple_mu_runtime_state():
    aec = _aec(16000, 256)
    aec._simple_mu_holdoff = 9
    aec._simple_mu_ratio = 0.123
    aec.reset()
    assert aec._simple_mu_holdoff == 0 and aec._simple_mu_ratio == 1.0


def test_the_attack_branch_applies_the_frozen_coefficient():
    """The attack branch cannot use the saturation trick above -- its incoming
    ratio is small and shaped by the echo/near lift -- so recover the
    coefficient from TWO runs of the SAME stimulus that differ only in the
    starting ratio. The incoming ratio depends on the frame and the filter
    state, not on `_simple_mu_ratio`, so it cancels:

        new(s) = a*s + (1-a)*r   ->   a = (new2 - new1) / (s2 - s1)
    """
    def step_from(start):
        aec = _aec(16000, 256)
        aec._simple_mu_holdoff = 0
        aec._simple_mu_ratio = start
        rng = np.random.default_rng(23)
        hop = aec.config.hop_size
        far = 0.001 * rng.standard_normal(hop)
        out = 0.500 * rng.standard_normal(hop)
        aec._update_simple_mu_ratio(out, far)
        return aec._simple_mu_ratio, aec._simple_mu_holdoff, aec

    new1, arm1, aec = step_from(0.90)
    new2, arm2, _ = step_from(0.95)
    assert arm1 == arm2 == FROZEN_HOLDOFF, (
        "both runs must have taken the attack branch and armed the holdoff")

    applied = (new2 - new1) / (0.95 - 0.90)
    assert applied == pytest.approx(FROZEN_ALPHA["attack"], abs=2e-6), (
        f"attack branch applied {applied:.6f}, expected the frozen "
        f"{FROZEN_ALPHA['attack']:.6f}")


def test_all_three_branches_are_reachable_and_distinguishable():
    """The holdoff counter's transition identifies the branch with no test-only
    instrumentation, which is what makes branch coverage checkable on a real
    corpus without adding diagnostic fields to the public struct:

        holdoff went UP        attack, fresh onset (armed to the limit)
        holdoff went DOWN      hold
        holdoff stayed nonzero attack, ongoing
        holdoff stayed 0       release (an attack at 0 would arm it, and the
                               limit is always >= 1)
    """
    aec = _aec(16000, 256)

    rng = np.random.default_rng(24)
    seen = set()
    prev = aec._simple_mu_holdoff
    for i in range(400):
        # Alternate DT-ish and far-only bursts so every branch gets a turn.
        far_scale, err_scale = (0.001, 0.5) if (i // 7) % 2 == 0 else (0.5, 0.001)
        _drive(aec, 1, far_scale=far_scale, err_scale=err_scale, rng=rng)
        now = aec._simple_mu_holdoff
        if now > prev:
            seen.add("attack")
        elif now < prev:
            seen.add("hold")
        elif now == 0:
            seen.add("release")
        else:
            seen.add("attack")
        prev = now
    assert seen == {"attack", "hold", "release"}, (
        f"only reached {sorted(seen)} -- frozen branch coverage incomplete")
