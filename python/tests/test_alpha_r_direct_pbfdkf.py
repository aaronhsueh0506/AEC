"""`_alpha_r` is a live constant of the DIRECT PBFDKF API, and only of that.

This is the Python half of `test_rate_structural` check (d5). It exists because
`_alpha_r` cannot be pinned by anything that drives the AEC through its normal
entry point, and establishing that took two wrong readings:

  ``AEC``, shadow ON (default)
      ``_error_psd`` is never read. The orchestrator publishes
      ``_e2_coarse_per_bin`` every hop and the live per-bin branch of the
      H_error refresh reads ``error_spec`` directly.
  ``AEC``, shadow OFF
      the scalar fallback DOES run and DOES read the smoothed ``_error_psd`` --
      but ``_e2_coarse_for_refresh`` is likewise only set on the shadow path, so
      it stays ``0.0`` and ``e2_ref_sum <= e2_coa_sum`` is constant-false
      whatever the coefficient is. Confirmed empirically in C: 90 cases x 2
      grids with ``--no-shadow`` are byte-identical
      (``eval/ab_evidence/2026-08-07-alpha-r/``).
  direct ``PBFDKF.process()``
      the caller supplies its own ``_e2_coarse_for_refresh``, both sides of the
      comparison vary, and ``_alpha_r`` selects the leakage branch and the
      adaptation that follows.

So a zero delta from a wrapper-level A/B is not evidence that this constant is
inert, and the constant is neither removable nor safe to leave un-retimed: the
10 ms anchor is what a direct caller gets.

What is asserted is the coefficient the EMA ACTUALLY APPLIED, recovered from two
runs differing only in ``_error_psd``'s starting value -- not the stored
attribute, which is what the pre-fix tests asserted while the C loop ran on a
hardcoded literal.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.dirname(_HERE)
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from modules import aec3_scale  # noqa: E402
from modules.filters import PBFDKF  # noqa: E402

GRIDS = [(8000, 256), (16000, 256), (16000, 512), (48000, 1024)]
N_PART = 4
WARM = 8
TC_TARGET_MS = 194.957302      # 0.95 at a 10 ms hop
REF_HOP_10MS = 160
REF_SR = 16000


def _probe(sample_rate, fft_size, seed, old, e2_coa):
    """One measured hop on a fresh PBFDKF.

    `old` seeds `_error_psd`; `e2_coa` is the scalar the refresh compares
    against. Deterministic, so two probes differ only in those two arguments.
    """
    hop = fft_size // 2
    filt = PBFDKF(fft_size, N_PART, mu=0.3, delta=1e-6,
                  hop_size=hop, sample_rate=sample_rate)
    rng = np.random.default_rng(seed)
    far = (0.25 * (2.0 * rng.random(hop) - 1.0)).astype(np.float32)
    mic = (0.10 * (2.0 * rng.random(hop) - 1.0)).astype(np.float32)

    for _ in range(WARM):
        filt.process(mic, far)

    # Open every gate that would divert the hop to refresh-only, so the measured
    # hop takes the full update path (EMA, then refresh). Set directly rather
    # than driven into place: the gates are not what this pins, and driving them
    # would make the probe stimulus-dependent.
    filt._call_counter = 10 * N_PART
    filt._poor_excitation_counter = 10 * N_PART
    filt._saturated_capture = False
    filt._block_stationary = False
    filt._initial_state_active = False      # steady leakage pair, not transient

    filt._e2_coarse_per_bin = None          # the scalar fallback under test
    filt._disallow_leakage_diverged = False  # else use_converged is forced True
    filt._h_error_refresh_erl_floor = np.float32(0.0)
    filt._e2_coarse_for_refresh = e2_coa
    filt._leakage_converged = np.float32(1.25e-4)
    filt._leakage_diverged = np.float32(0.125)
    filt._error_psd = np.full(filt.n_freqs, old, dtype=np.float32)
    filt.H_error_per_bin = np.ones(filt.n_freqs, dtype=np.float32)
    filt._erl_per_bin = np.full(filt.n_freqs, 0.5, dtype=np.float32)

    filt.process(mic, far)

    return {
        "psd_sum": float(np.sum(filt._error_psd, dtype=np.float64)),
        "div_frac": filt._last_leakage_div_frac,
        "h_error_mean": float(np.mean(filt.H_error_per_bin, dtype=np.float64)),
        "n_freqs": filt.n_freqs,
        "alpha_field": float(filt._alpha_r),
    }


@pytest.fixture(scope="module", params=GRIDS, ids=lambda g: f"{g[0]}_{g[1]}")
def measured(request):
    """Recover the applied coefficient once per grid; every test below reads
    it."""
    sample_rate, fft_size = request.param
    hop = fft_size // 2
    seed = 0x5A17 + fft_size + sample_rate

    p1 = _probe(sample_rate, fft_size, seed, 0.0, 0.0)
    assert p1["psd_sum"] > 0.0, "the EMA never ran -- probe is not measuring"

    k = p1["n_freqs"]
    # Large enough that k*old dominates sum(e2); otherwise the two candidate
    # coefficients give nearly the same sum and the branch test below is
    # ill-conditioned.
    old2 = 1.0 + 30.0 * p1["psd_sum"] / k
    p2 = _probe(sample_rate, fft_size, seed, old2, 0.0)

    # sum(old) = a*k*old + (1-a)*sum(e2), so the stimulus term cancels and `a`
    # comes out of the two sums alone -- no assumption about the signal, the
    # filter state, or how many bins converged.
    a_meas = (p2["psd_sum"] - p1["psd_sum"]) / (k * old2)
    return {
        "sample_rate": sample_rate, "fft_size": fft_size, "hop": hop,
        "seed": seed, "n_freqs": k, "old2": old2,
        "a_meas": a_meas, "alpha_field": p1["alpha_field"],
        "p1": p1, "p2": p2,
    }


def test_the_applied_coefficient_is_the_retimed_field(measured):
    expected = aec3_scale.growth_rehop(0.95, REF_HOP_10MS, REF_SR,
                                       measured["hop"], measured["sample_rate"])
    assert measured["a_meas"] == pytest.approx(expected, abs=5e-4), (
        f"error_psd EMA applied alpha={measured['a_meas']:.6f}, expected the "
        f"10 ms-anchored {expected:.6f}")
    assert measured["alpha_field"] == pytest.approx(expected, rel=1e-12)


def test_the_applied_coefficient_covers_the_anchored_wall_clock_span(measured):
    """The field-vs-applied check alone still passes if BOTH move together,
    which is exactly what reverting the reference hop to 256 would do. Pin the
    span itself."""
    hop, sr = measured["hop"], measured["sample_rate"]
    tc_ms = -(hop / sr) * 1000.0 / math.log(measured["a_meas"])
    assert tc_ms == pytest.approx(TC_TARGET_MS, abs=1.0), (
        f"applied alpha_r covers TC {tc_ms:.3f} ms; the 10 ms anchor is "
        f"{TC_TARGET_MS:.3f} ms and a 16 ms reference would give "
        f"{TC_TARGET_MS * 1.6:.3f} ms")


def test_the_applied_coefficient_is_not_the_authored_literal(measured):
    """No shipped grid has a 10 ms hop, so the retimed value must differ from
    0.95 everywhere -- i.e. these assertions cannot be satisfied by a use site
    that ignores the field."""
    assert abs(measured["a_meas"] - 0.95) > 1e-3


def test_the_coefficient_selects_the_leakage_branch(measured):
    """Downstream effect, not just a value: with the threshold set between the
    two candidate sums, the retimed coefficient and the 0.95 literal resolve
    `use_converged` in opposite directions."""
    k, old2 = measured["n_freqs"], measured["old2"]
    a = measured["a_meas"]
    s_e2 = measured["p1"]["psd_sum"] / (1.0 - a)
    sum_retimed = measured["p2"]["psd_sum"]
    sum_literal = 0.95 * k * old2 + 0.05 * s_e2
    mid = 0.5 * (sum_retimed + sum_literal)

    assert abs(sum_retimed - sum_literal) > 1e-3 * abs(mid), (
        "the two coefficients do not give separable sums -- setup is "
        "ill-conditioned, not passing")

    expect_retimed = 0.0 if sum_retimed <= mid else 1.0
    expect_literal = 0.0 if sum_literal <= mid else 1.0
    assert expect_retimed != expect_literal, "setup does not separate branches"

    p3 = _probe(measured["sample_rate"], measured["fft_size"],
                measured["seed"], old2, mid)
    assert p3["div_frac"] == expect_retimed, (
        f"leakage branch followed div_frac={p3['div_frac']}, expected "
        f"{expect_retimed} for the retimed coefficient; the 0.95 literal would "
        f"give {expect_literal}")


def test_the_branch_moves_h_error(measured):
    """...and the branch is not a diagnostic counter: forcing the other side
    moves H_error by exactly (leakage_diverged - leakage_converged) * erl."""
    k, old2 = measured["n_freqs"], measured["old2"]
    a = measured["a_meas"]
    s_e2 = measured["p1"]["psd_sum"] / (1.0 - a)
    sum_retimed = measured["p2"]["psd_sum"]
    sum_literal = 0.95 * k * old2 + 0.05 * s_e2
    mid = 0.5 * (sum_retimed + sum_literal)
    expect_retimed = 0.0 if sum_retimed <= mid else 1.0

    p3 = _probe(measured["sample_rate"], measured["fft_size"],
                measured["seed"], old2, mid)
    far_side = mid * 1e6 if expect_retimed == 1.0 else -1.0
    p4 = _probe(measured["sample_rate"], measured["fft_size"],
                measured["seed"], old2, far_side)

    assert p4["div_frac"] != p3["div_frac"], (
        "forcing the other threshold did not flip the branch")
    gap = abs(p3["h_error_mean"] - p4["h_error_mean"])
    assert gap == pytest.approx((0.125 - 1.25e-4) * 0.5, abs=1e-3)


@pytest.mark.parametrize("enable_shadow", [True, False])
def test_the_aec_wrapper_output_does_not_depend_on_alpha_r(enable_shadow):
    """The scope claim itself, measured rather than described.

    Drive the whole AEC twice with wildly different coefficients and require
    bit-identical output in BOTH shadow modes -- the two halves of the table in
    this module's docstring, which reach the same result for different reasons.

    This is the assertion that fails if somebody later wires
    ``_e2_coarse_for_refresh`` on a non-shadow path: at that point the wrapper
    WOULD start observing ``_alpha_r``, this file's scope note would be wrong,
    and the constant would need a wrapper-level A/B it has never had. Failing
    here is the correct outcome then, not a regression.
    """
    from modules.orchestrator import AEC
    from modules.config import AecConfig

    def render(alpha_r):
        cfg = AecConfig(sample_rate=16000, enable_shadow=enable_shadow)
        aec = AEC(cfg)
        aec.filter._alpha_r = alpha_r
        rng = np.random.default_rng(7)
        hop = cfg.hop_size
        out = []
        for _ in range(40):
            far = (0.3 * (2.0 * rng.random(hop) - 1.0)).astype(np.float64)
            mic = (0.2 * far + 0.05 *
                   (2.0 * rng.random(hop) - 1.0)).astype(np.float64)
            out.append(np.asarray(aec.process(mic, far), dtype=np.float64).copy())
        return np.concatenate(out)

    baseline = render(0.95980)     # the retimed value at this grid
    perturbed = render(0.10)       # nothing like it
    assert np.array_equal(baseline, perturbed), (
        f"AEC output changed with enable_shadow={enable_shadow} when _alpha_r "
        f"moved 0.9598 -> 0.10; the wrapper now observes this constant and the "
        f"scope note in this module is out of date")
