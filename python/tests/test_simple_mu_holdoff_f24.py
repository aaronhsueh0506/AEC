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


def _aec():
    return AEC(AecConfig(sample_rate=16000))


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

    seen = _drive(aec, 40, far_scale=0.001, err_scale=0.5, rng=rng)
    armed = seen[0]
    assert armed > 0, "setup failed: never armed"

    # With the guard, the counter is armed once and then decrements every hop.
    # Without it, every hop re-arms and the sequence is constant at `armed`.
    assert min(seen) < armed, (
        f"holdoff never decremented across {len(seen)} sustained DT hops "
        f"(stuck at {armed}) -- the F2.4 guard is missing, so ongoing "
        f"double-talk re-arms the counter and mu can never release"
    )
    assert seen[-1] == 0, (
        f"holdoff did not run down to 0 within {len(seen)} hops: {seen[-8:]}"
    )


def test_holdoff_can_rearm_after_it_has_expired():
    """The guard must not weld the counter off: once it reaches 0, a new onset
    arms it again. A fix that simply deleted the assignment would pass the test
    above and fail this one."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(3)

    seen = _drive(aec, 40, far_scale=0.001, err_scale=0.5, rng=rng)
    assert seen[-1] == 0

    aec._simple_mu_ratio = 1.0          # fresh onset
    again = _drive(aec, 1, far_scale=0.001, err_scale=0.5, rng=rng)
    assert again[0] > 0, "a new onset after expiry must re-arm the holdoff"


def test_holdoff_decrements_monotonically_while_it_runs():
    """Pins the shape, not just the endpoints: no hop may increase it."""
    aec = _aec()
    _force_attack(aec)
    rng = np.random.default_rng(4)
    seen = _drive(aec, 40, far_scale=0.001, err_scale=0.5, rng=rng)
    for i in range(1, len(seen)):
        assert seen[i] <= seen[i - 1], (
            f"holdoff increased at hop {i}: {seen[i - 1]} -> {seen[i]}; "
            f"full trace {seen}"
        )
