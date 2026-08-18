"""``AEC.apply_external_realign()`` — the caller-side realign of an
``EXTERNAL_ALIGNED`` instance, and the "vertical line" it exists to prevent.

An ``EXTERNAL_ALIGNED`` instance converges while its caller serves one far
alignment; the caller then changes that alignment (a shared delay estimator
re-locked, a wrapper re-measured). The only response available before this
entry point existed was ``reset()``, which wipes the converged filter — the
echo is fully re-exposed for dozens of hops (the bright half of the
spectrogram line) — and restarts the WOLA sequence, which costs a near-zero
output hop (the dark half).

What is asserted:

  1. THE WARM PATH HOLDS CANCELLATION across an advance (delta > 0, the
     raw -> aligned move a first acquisition makes): the residual stays below
     half the echo on every following hop.
  2. THE DEFECT REPRODUCES — the same scene with ``reset()`` instead rebounds
     to >= 0.8x the echo. Without this row, row 1 could pass on a scene where
     the filter never cancelled in the first place.
  3. THE SHIFT IS SIGNED — a retard (delta < 0) has a bounded refill transient
     and then cancels outright, and it moves the taps in the OPPOSITE
     direction from an advance (the control that would catch a retard
     implemented as an advance).
  4. THE CONTRACT — 0 on delta == 0, -1 outside ``EXTERNAL_ALIGNED``, and a
     gate rejection returns 0 (the reset outcome) rather than raising.
  5. THE GATE SURVIVES FAR SILENCE. The inst-ERLE ring holds ~15 entries and
     every far-silent hop pushes 0.0 into it, so a realign arriving >= 16 hops
     after the last far burst — a wrapper that re-measures its alignment
     between bursts, i.e. the ordinary case — would read "not cancelling" on a
     filter cancelling at ~13 dB windowed and throw it away. The windowed half
     of ``linear_is_cancelling()`` is what still knows.
  6. THE RESET BRANCH RECONVERGES rather than stalling: pinned against a
     ``reset()`` twin's trajectory at six checkpoints and settled well under
     the echo. The branch is only correct because it is FILTER-ONLY, which is
     row 7.
  7. FRAMING IS UNTOUCHED. On a near-end-only scene (no far, so the filter and
     its latches are inert and the mic/synthesis path is all that is left to
     observe) the filter-only reset leaves the formed output stream
     BIT-IDENTICAL to an instance that was never called, while a ``reset()``
     twin does not — it drops a near-zero hop at the boundary. That difference
     is the entire reason this entry point is not ``reset()``.

Mutation checks (each breaks one line and must go red here):
  - drop the signed arm of ``PBFDAF.warm_shift_ir`` (return on s <= 0 as
    before) -> row 3's direction control fails;
  - gate on the inst-ERLE ring alone instead of ``linear_is_cancelling()``
    -> row 5 fails;
  - make the reject branch a counters-only ``handle_echo_path_change`` (the
    taps stay at the abandoned alignment) -> row 6 fails;
  - make the reject branch a full ``reset()`` -> row 7 fails;
  - skip the ``X_buf`` clear on a retard -> row 3's post-refill row fails;
  - skip the SHADOW's ``far_buffer`` clear in ``_reset_filter_for_realign``
    -> row 6 fails (settled 0.209 vs 0.007 of the echo). The main filter's own
    ``far_buffer`` clear is inert on this scene — both are cleared for the same
    reason, but it is the shadow's spliced analysis frame that this scene
    measures, through the shadow-advantage/DT signals it feeds.
"""

import numpy as np
import pytest

from modules.config import AecConfig
from modules.enums import AecDelayMode
from modules.orchestrator import AEC


SR = 16000
FFT = 512
HOP = 256
D = 300                 # true bulk delay, samples
HOPS = 250              # convergence hops before the realign
TAIL = 6                # hops watched after it
GAP = 24                # far-silent hops (> the 15-entry inst-ERLE ring)
OBS = 12                # far-active hops watched after the gap
TRAJ = 140              # reconvergence hops compared against the reset() twin
PAD = 4 * D


def _scene(n_hops):
    """One deterministic far-end history, long enough for `n_hops` served hops.

    ``mic[n] = 0.5 * far[n - D]`` is read from the same array, so the echo is
    exactly what a filter aligned at lag ``D - align`` can cancel.
    """
    rng = np.random.default_rng(0x5EED)
    total = PAD + (n_hops + 2) * HOP
    return (rng.standard_normal(total) * 0.25).astype(np.float32)


def _make(warm_transfer=True, res_context=True):
    """The seam configuration a 4-lane wrapper runs.

    ``enable_res=False`` makes the emitted samples the linear residual itself;
    ``return_res_context=True`` keeps the AEC3 post chain running in
    context-only mode, which is what maintains the windowed ERLE the realign
    gate reads (and what produces ``formed_output``).
    """
    cfg = AecConfig(
        sample_rate=SR, frame_size=FFT, hop_size=HOP,
        enable_res=False, return_res_context=res_context,
        enable_delay_est=False, fixed_delay_samples=-1,
        delay_acquire_warm_transfer=warm_transfer,
    )
    aec = AEC(cfg)
    assert aec.config.delay_mode is AecDelayMode.EXTERNAL_ALIGNED
    return aec


def _emitted(result):
    """``process()`` returns ``(audio, context)`` when the context is on."""
    return result[0] if isinstance(result, tuple) else result


def _rms(hop):
    return float(np.sqrt(np.mean(np.square(np.asarray(hop, dtype=np.float64)))))


def _hop(aec, far_hist, h, align):
    """Serve one hop at `align` and return the residual RMS."""
    base = PAD + h * HOP
    mic = 0.5 * far_hist[base - D:base - D + HOP]
    far = far_hist[base - align:base - align + HOP]
    return _rms(_emitted(
        aec.process(np.ascontiguousarray(mic), np.ascontiguousarray(far))))


def _silent_hop(aec):
    zero = np.zeros(HOP, dtype=np.float32)
    aec.process(zero, zero)


def _echo_rms(far_hist, h):
    base = PAD + h * HOP - D
    return _rms(0.5 * far_hist[base:base + HOP])


@pytest.fixture(scope="module")
def far_hist():
    return _scene(HOPS + GAP + max(TAIL, OBS, TRAJ) + 4)


# --- rows 1 + 2: the warm path holds cancellation, the defect reproduces ----

def test_warm_advance_keeps_cancellation_where_reset_re_exposes_the_echo(far_hist):
    aec = _make()
    resid = 0.0
    for h in range(HOPS):
        resid = _hop(aec, far_hist, h, 0)
    echo = _echo_rms(far_hist, HOPS - 1)
    assert resid < 0.5 * echo, "scene must converge at the raw alignment first"

    assert aec.apply_external_realign(D - 0) == 1, \
        "a cancelling filter and an in-span delta take the warm path"
    warm_worst = max(_hop(aec, far_hist, h, D) for h in range(HOPS, HOPS + TAIL))
    assert warm_worst < 0.5 * _echo_rms(far_hist, HOPS), \
        "no echo re-exposure after the warm realign"

    twin = _make()
    for h in range(HOPS):
        _hop(twin, far_hist, h, 0)
    twin.reset()
    reset_worst = max(_hop(twin, far_hist, h, D) for h in range(HOPS, HOPS + TAIL))
    assert reset_worst >= 0.8 * _echo_rms(far_hist, HOPS), \
        "reset() control must re-expose the echo (the defect reproduces)"


# --- row 3: the shift is signed ---------------------------------------------

def test_retard_shifts_the_other_way_and_cancels_after_the_refill(far_hist):
    aec = _make()
    for h in range(HOPS + 1):
        _hop(aec, far_hist, h, D)
    assert _hop(aec, far_hist, HOPS + 1, D) < 0.5 * _echo_rms(far_hist, HOPS), \
        "scene must converge at the aligned lag first"

    peak_before = int(np.argmax(np.abs(aec.filter.get_time_domain_filter())))
    assert aec.apply_external_realign(0 - D) == 1
    peak_after = int(np.argmax(np.abs(aec.filter.get_time_domain_filter())))
    # The direction control: the response must move toward LATER taps by the
    # magnitude of the retard. An advance-only implementation moves it the
    # other way (or, from tap ~0, off the front of the filter entirely).
    assert peak_after > peak_before
    assert abs((peak_after - peak_before) - D) <= 1

    # The retard clears the far history, so the echo is exposed until the
    # partition ring refills past the shifted response. This scene is the
    # mildest corner of that bound (|delta| = 300 with the response near tap
    # 0): what is pinned is that the transient stays at the echo's own scale
    # and that cancellation then resumes OUTRIGHT, against a reset control
    # that stays exposed for dozens of hops.
    transient = max(_hop(aec, far_hist, h, 0) for h in range(HOPS + 2, HOPS + 4))
    assert transient <= 1.4 * _echo_rms(far_hist, HOPS)
    settled = max(_hop(aec, far_hist, h, 0) for h in range(HOPS + 4, HOPS + TAIL + 2))
    assert settled < 0.2 * _echo_rms(far_hist, HOPS), \
        "the retard realign cancels outright once the ring has refilled"


# --- row 4: the contract -----------------------------------------------------

def test_contract_zero_delta_foreign_mode_and_gate_rejection(far_hist):
    aec = _make()
    assert aec.apply_external_realign(0) == 0, "delta 0 is a no-op"

    matched = AEC(AecConfig(sample_rate=SR, frame_size=FFT, hop_size=HOP))
    assert matched.config.delay_mode is AecDelayMode.MATCHED
    assert matched.apply_external_realign(D) == -1, \
        "only EXTERNAL_ALIGNED may have its alignment moved from outside"

    # A cold instance has no cancellation evidence, so the gate rejects and the
    # reset outcome is reported rather than a warm success.
    assert _make().apply_external_realign(D) == 0

    # An in-span delta on a converged filter still rejects when the warm
    # transfer is configured off — the caller gets the reset outcome, not -1.
    off = _make(warm_transfer=False)
    for h in range(HOPS):
        _hop(off, far_hist, h, 0)
    assert off.apply_external_realign(64) == 0


# --- row 5: the gate survives a far-silent gap -------------------------------

def test_gate_survives_a_far_silent_gap(far_hist):
    aec = _make()
    resid = 0.0
    for h in range(HOPS):
        resid = _hop(aec, far_hist, h, 0)
    assert resid < 0.5 * _echo_rms(far_hist, HOPS - 1)

    for _ in range(GAP):
        _silent_hop(aec)
    # The ring reading alone is now 0: every silent hop pushed a 0.0 into it.
    assert max(list(aec._erle_slope_buf)[-15:] or [0.0]) <= float(
        aec.config.delay_acquire_inst_erle_db), \
        "the gap must actually empty the inst-ERLE ring, or this row proves nothing"
    assert aec.apply_external_realign(D - 0) == 1, \
        "the windowed reading still knows the filter is cancelling"

    worst = max(
        _hop(aec, far_hist, h, D) / _echo_rms(far_hist, h)
        for h in range(HOPS + GAP, HOPS + GAP + OBS)
    )
    assert worst < 0.5


# --- row 6: the reset branch reconverges -------------------------------------

def test_reset_branch_tracks_a_reset_twin_and_settles(far_hist):
    """Warm transfer off forces the branch on a genuinely converged filter, so
    the taps really are parked at the alignment the caller abandoned. The twin
    gets the identical scene and a ``reset()`` at the identical hop: that is
    the trajectory a full restart achieves, and the filter-only reset has to
    match it without the twin's synthesis restart.
    """
    aec = _make(warm_transfer=False)
    twin = _make(warm_transfer=False)
    for h in range(HOPS + 1):
        _hop(aec, far_hist, h, 0)
        _hop(twin, far_hist, h, 0)

    assert aec.apply_external_realign(D - 0) == 0
    twin.reset()

    ours, theirs = [], []
    for h in range(HOPS + 1, HOPS + 1 + TRAJ):
        echo = _echo_rms(far_hist, h)
        ours.append(_hop(aec, far_hist, h, D) / echo)
        theirs.append(_hop(twin, far_hist, h, D) / echo)

    # The band is wide on purpose. Both trajectories are already at ~1% of the
    # echo a handful of hops in, where hop-to-hop scene noise — not the
    # restart — decides the ratio (it crosses below 1.0 at several
    # checkpoints). What the comparison is here to catch is a branch that does
    # NOT reconverge: the counters-only echo-path-change this used to be left
    # the residual at 1.3-1.5x the echo indefinitely, i.e. two orders of
    # magnitude outside this band. The absolute settled bound below is the
    # tight half of the row.
    for k in (5, 10, 20, 40, 80, 120):
        assert ours[k] <= 2.5 * theirs[k], (
            f"reset-trajectory +{k}: {ours[k]:.4f} vs twin {theirs[k]:.4f}"
        )
    assert max(ours[-20:]) < 0.02, "the reset branch must settle, not stall"


# --- row 7: framing is untouched ---------------------------------------------

def _near_only_stream(aec, near, first_hop, n_hops):
    """Run a near-end-only stream and return the emitted audio, hop by hop."""
    zero = np.zeros(HOP, dtype=np.float32)
    return [
        np.asarray(_emitted(aec.process(
            np.ascontiguousarray(near[h * HOP:(h + 1) * HOP]), zero)),
            dtype=np.float32).copy()
        for h in range(first_hop, first_hop + n_hops)
    ]


def test_filter_only_reset_leaves_the_synthesis_and_mic_frame_alone():
    """No far end at all, so the filter and every latch it feeds are inert and
    the mic/synthesis path is the only thing left to observe. The filter-only
    reset must be invisible in the output; ``reset()`` must not be.

    Run with the RES stage ON, which is what owns the synthesis overlap-add:
    this is the configuration in which the dark half of the "vertical line" is
    an actual zero hop rather than a perturbed one.
    """
    cfg = dict(sample_rate=SR, frame_size=FFT, hop_size=HOP,
               enable_res=True, return_res_context=True,
               enable_delay_est=False, fixed_delay_samples=-1)
    rng = np.random.default_rng(0xC0FFEE)
    near = (rng.standard_normal(30 * HOP) * 0.2).astype(np.float32)
    boundary, watch = 20, 5

    control, realigned, twin = (AEC(AecConfig(**cfg)) for _ in range(3))
    for aec in (control, realigned, twin):
        _near_only_stream(aec, near, 0, boundary)

    assert realigned.apply_external_realign(D) == 0, \
        "a silent far end leaves no cancellation evidence, so the gate rejects"
    twin.reset()

    ref = _near_only_stream(control, near, boundary, watch)
    ours = _near_only_stream(realigned, near, boundary, watch)
    theirs = _near_only_stream(twin, near, boundary, watch)

    for h, (a, b) in enumerate(zip(ref, ours)):
        assert np.array_equal(a, b), (
            f"the filter-only reset changed the output stream at hop +{h}"
        )
    # The can-fail control: reset() emits a hop of silence at the boundary,
    # which is exactly the discontinuity this entry point exists to avoid.
    assert _rms(ref[0]) > 0.1, "the scene must carry audio across the boundary"
    assert _rms(theirs[0]) < 1e-6 * _rms(ref[0]), \
        "reset() must restart the WOLA sequence (the near-zero output hop)"
