"""Path-B backward-jump quarantine (``delay_backward_quarantine_enabled`` /
``delay_backward_quarantine_s``).

THE DEFECT IT TARGETS, reproduced FIRST with the mechanism off so the rows
below cannot pass against a scene that never mis-locks: with a white-noise far
end at a true bulk delay of 6400 samples (16 kHz, fft 512 / hop 256) the
matched filter acquires CORRECTLY at 6336 on hop 32, then on hop 49 re-locks
to 4800 -- exactly 1600 samples (100 ms) EARLY, the pre-echo mis-attribution
signature -- and holds that wrong answer for the rest of the run. Because the
wrong candidate is SUSTAINED, no K-consecutive-confirmation gate can reject
it: every window it is offered agrees with itself.

WHAT THE MECHANISM DOES: a candidate strictly EARLIER than the delay in force,
offered while the linear filter still cancels at the applied alignment, is
held for a bounded window and then ACCEPTED. It DELAYS a mis-lock by the
window; it does not cure one. Curing pre-echo mis-attribution is estimator
work and is not what this guard is for.

WHY BOUNDED AND DIRECTIONAL: the unbounded, direction-blind predicate this
replaced -- and the multipath measurement that killed it -- are written up
once, on ``delay_backward_quarantine_enabled`` in ``c_impl/include/aec.h``.
The multipath row below is the permanent guard against that regression.

What is asserted:
  1. THE DEFECT REPRODUCES with the mechanism off (the shipped default), and
     the wrong delay is sustained to the end of the run.
  2. THE QUARANTINE DELAYS IT BY EXACTLY THE WINDOW and serves the correct
     6336 meanwhile: 62 hops at the 1.0 s default on this grid, so acceptance
     moves from hop 49 to hop 111 -- and to hop 174 when the window is
     doubled, which is what makes "the expiry is the release" falsifiable
     rather than asserted.
  3. FIRST ACQUISITION IS UNAFFECTED: same acquisition hop, same value, on or
     off. This is a sibling of ``delay_acquire_protect_converged``, not a
     replacement.
  4. A FORWARD MOVE IS NOT QUARANTINED AT ALL: the whole accepted-delay
     trajectory is identical on and off.
  5. MULTIPATH / PATH ADDITION re-locks in ALL FOUR gain rows, never later
     than unguarded by more than one window.
  6. BOTH ERLE ARMS ARE LOAD-BEARING, each pinned by the scene where it is
     the deciding one:
       - the inst-ERLE peak decides row 2. At the acceptance hop windowed
         ERLE is 1.660 dB -- under its own 2.5 dB threshold, because the
         re-lock lands inside the lag ``delay_acquire_inst_erle_db``'s doc
         comment describes -- while the inst peak is 7.269 dB. A
         windowed-only predicate cannot see this defect at all.
       - the windowed arm decides the BACKWARD-move row. At the unguarded
         acceptance hop the inst ring has aged out (peak -0.934 dB) while
         windowed ERLE is still 3.728 dB, so the slow arm alone engages the
         quarantine there.
  7. EARLY RELEASE ON COLLAPSE IS REAL, not just the expiry wearing off: the
     backward MOVE (old path removed, so cancellation genuinely collapses) is
     accepted 37 hops late -- strictly INSIDE the 62-hop window, so what
     released it was the collapse.

Mutation checks (each breaks one line and must go red here):
  - drop the direction test (quarantine every differing candidate) -> row 4
    fails: the forward move is held and the trajectories stop matching;
  - drop the expiry -> rows 2 and 5 fail (nothing is ever accepted);
  - drop the cancellation test from the predicate -> row 7 fails: the
    backward move is held for the full window instead of 37 hops;
  - drop the inst-ERLE-peak arm -> row 2 fails;
  - drop the windowed-ERLE arm -> row 7's "the quarantine engages at all"
    assertion fails (guarded == unguarded);
  - ignore ``delay_backward_quarantine_s`` (hardcode the window) -> row 2's
    doubled-window assertion fails;
  - change the enable default to True -> the default row fails (rows 1-2
    build their configs explicitly, which is why the default needs its own
    row);
  - move the quarantine onto Path A as well -> row 3 fails.

The windowed-arm row is the one the C mirror cannot produce, which is why it
is pinned here.

The C mirror is ``c_impl/test/test_delay_backward_quarantine.c``. It pins the
same rows plus a four-point window sweep, but NOT the windowed arm: the C
matched filter is duty-cycled, so its post-move candidate arrives long after
windowed ERLE has decayed to 0 and the slow arm never decides. That is why
rows 6b and 7 are pinned here and not there.

Run:
    python3 -m pytest python/tests/test_delay_backward_quarantine.py
"""
from __future__ import annotations

import functools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AEC, AecConfig


_SR = 16000
_FFT = 512
_HOP = 256

# The library's own seconds -> cycles conversion, restated (not imported) so a
# silent change to it fails here instead of being absorbed.
def _window_hops(seconds: float) -> int:
    return max(1, int(round(seconds / (_HOP / _SR))))


_WINDOW_S = 1.0              # the config default
_WINDOW_HOPS = _window_hops(_WINDOW_S)          # 62 on this grid
_WIDE_S = 2.0
_WIDE_HOPS = _window_hops(_WIDE_S)              # 125

# --- the pre-echo mis-lock scene -----------------------------------------
# Each variant is only as long as the deepest hop the rows that read it look
# at, because every hop costs a real AEC pass. The unguarded row is the long
# one: its "and it stays there" claim is about the tail, so it keeps 146 hops
# past the widest swept acceptance. The guarded rows stop reading at their own
# acceptance hop (111 at the 1.0 s window, 174 at 2.0 s) and are sized for it.
_MIS_HOPS = 320              # OFF: sustained-tail row
_MIS_ON_HOPS = 120           # ON at the 1.0 s default: reads reach hop 111
_MIS_WIDE_HOPS = 180         # ON at 2.0 s: reads reach hop 174
_MIS_DELAY = 6400            # true bulk delay, 400 ms
_MIS_LOCK_HOP = 32           # measured: where the CORRECT delay is acquired
_MIS_LOCK = 6336             # measured: the correct lock, on the 64-grid
# Retained as the value that must NEVER be applied: this scene used to re-lock
# to it on hop 49 and hold it. The selection-seam fix removed that, and
# test_the_pre_echo_mis_lock_no_longer_happens is what keeps it removed.
_MIS_WRONG = 4800            # _MIS_DELAY - 1600: the pre-echo answer

# --- the FORWARD-move scene (never quarantined) ---------------------------
_FWD_HOPS = 500
_FWD_AT = 375                # the echo moves at hop 375 (6 s)
_FWD_D0 = 64                 # pre-move delay: inside the tap reach
_FWD_D1 = 3000               # post-move delay: LARGER, so unguarded

# --- the BACKWARD-move scene (quarantined, but released by the collapse) --
_BWD_HOPS = 600
_BWD_AT = 375
_BWD_D0 = 3000               # pre-move delay
_BWD_D1 = 64                 # post-move: EARLIER, so the direction test fires
# Measured: guarded acceptance lands 36 hops after unguarded, strictly inside
# the 62-hop window, because the old path is GONE and cancellation really
# collapses. Pinned as the strict "engaged at all" plus "released before
# expiry" pair. (Was 37 while the aggregator reported the pre-echo candidate;
# reporting the dominant peak moves acquisition by one 32-sample grid step in
# this scene, and the hold with it.)
_BWD_EXPECTED_DELTA = 36

# --- the multipath / path-addition scene ---------------------------------
# The old reflection at _MP_OLD survives the whole run at gain g_old; a
# STRONGER new path appears EARLIER, at _MP_NEW / 0.5, from hop _MP_AT. Path
# ADDITION, not movement: ERLE at the applied alignment never collapses, so
# only the expiry can release this scene.
_MP_HOPS = 620
_MP_AT = 250
_MP_OLD = 2400
_MP_NEW = 1600
_MP_GAINS = (0.2, 0.3, 0.4, 0.5)


def _cfg(enabled: bool, window_s: float = _WINDOW_S) -> AecConfig:
    """The NN-seam configuration -- ``enable_res=0`` with
    ``return_res_context=1``, which is what both ULCNet pipelines and the 4ch
    lanes run. It is also the configuration in which C used to cache no
    windowed ERLE at all, so a guard verified only under ``enable_res=1``
    would have been verified in the one place it is not deployed."""
    return AecConfig(sample_rate=_SR, frame_size=_FFT, hop_size=_HOP,
                     enable_res=0, return_res_context=1,
                     delay_backward_quarantine_enabled=enabled,
                     delay_backward_quarantine_s=window_s)


def _static_scene(hops: int, delay: int):
    """White-noise far end -- the stimulus the matched filter is happiest
    with, so a mis-lock here is not an artefact of degenerate excitation."""
    rng = np.random.default_rng(3)
    n = hops * _HOP
    far = rng.standard_normal(n).astype(np.float32) * 0.1
    near = np.zeros_like(far)
    near[delay:] = far[:-delay] * 0.5
    return far, near


def _moving_scene(hops: int, at: int, d0: int, d1: int):
    """The echo path genuinely MOVES at hop ``at`` and nothing remains at
    ``d0``, so cancellation really does collapse -- the early-release
    condition."""
    rng = np.random.default_rng(5)
    n = hops * _HOP
    far = rng.standard_normal(n).astype(np.float32) * 0.1
    near = np.zeros_like(far)
    m = at * _HOP
    near[d0:m] = far[:m - d0] * 0.5
    near[m + d1:] = far[m:n - d1] * 0.5
    return far, near


def _multipath_scene(hops: int, at: int, d_old: int, g_old: float,
                     d_new: int, g_new: float):
    """Path ADDITION: the old reflection is never removed, so the applied
    alignment keeps cancelling and the early-release arm cannot carry this
    scene."""
    rng = np.random.default_rng(7)
    n = hops * _HOP
    far = rng.standard_normal(n).astype(np.float32) * 0.1
    near = np.zeros(n, dtype=np.float32)
    near[d_old:] += far[:n - d_old] * g_old
    new = np.zeros(n, dtype=np.float32)
    new[d_new:] = far[:n - d_new] * g_new
    m = at * _HOP
    near[m:] += new[m:]
    return far, near


def _run(far, near, enabled: bool, window_s: float = _WINDOW_S):
    """Returns (accepted delay per hop, windowed ERLE per hop, inst-ERLE
    15-frame peak per hop). Both ERLE series are sampled BEFORE the hop is
    processed, i.e. they are exactly the readings Path B judges that hop on
    -- sampling after would report the post-acceptance reset instead of the
    decision input."""
    np.random.seed(1)
    aec = AEC(_cfg(enabled, window_s))
    delays, wins, peaks = [], [], []
    for h in range(len(far) // _HOP):
        wins.append(float(aec._diag.get('erle_windowed', 0.0)))
        peaks.append(max(list(aec._erle_slope_buf)[-15:] or [0.0]))
        aec.process(near[h * _HOP:(h + 1) * _HOP],
                    far[h * _HOP:(h + 1) * _HOP])
        delays.append(aec._current_delay)
    return delays, wins, peaks


def _first_change_after(delays, at):
    for h in range(at + 1, len(delays)):
        if delays[h] != delays[h - 1]:
            return h
    return -1


# --- lazily-cached fixtures ----------------------------------------------
# Cached at module level rather than built eagerly in ``setUpClass``: every
# run here is a real AEC pass over hundreds of hops, and a class-level fixture
# makes a selective run (``-k off_by_default``) pay for all of them. With the
# cache, each test pulls only the scenes it reads and a full-suite run still
# computes each one exactly once. The returned lists are shared between
# callers, so no row may mutate them.

@functools.lru_cache(maxsize=None)
def _mis_run(enabled: bool, window_s: float, hops: int):
    far, near = _static_scene(hops, _MIS_DELAY)
    return _run(far, near, enabled, window_s)


@functools.lru_cache(maxsize=None)
def _move_run(enabled: bool, hops: int, at: int, d0: int, d1: int):
    far, near = _moving_scene(hops, at, d0, d1)
    return _run(far, near, enabled)


@functools.lru_cache(maxsize=None)
def _mp_delays(enabled: bool, g_old: float):
    far, near = _multipath_scene(_MP_HOPS, _MP_AT, _MP_OLD, g_old,
                                 _MP_NEW, 0.5)
    return _run(far, near, enabled)[0]


def _mis_off():
    return _mis_run(False, _WINDOW_S, _MIS_HOPS)


def _mis_on():
    return _mis_run(True, _WINDOW_S, _MIS_ON_HOPS)


def _mis_wide():
    return _mis_run(True, _WIDE_S, _MIS_WIDE_HOPS)


def _fwd(enabled: bool):
    return _move_run(enabled, _FWD_HOPS, _FWD_AT, _FWD_D0, _FWD_D1)


def _bwd(enabled: bool):
    return _move_run(enabled, _BWD_HOPS, _BWD_AT, _BWD_D0, _BWD_D1)


class DelayBackwardQuarantineTests(unittest.TestCase):

    def test_the_mechanism_is_off_by_default(self) -> None:
        """The shipped path must be untouched: the default-OFF enable is what
        lets this land without a bench obligation, and the byte-exactness it
        buys was verified out of band (seam-config render, 512000 bytes
        identical before and after). The window carries a real default so the
        knob is usable by setting one field, but it is inert until enabled."""
        self.assertFalse(AecConfig().delay_backward_quarantine_enabled)
        self.assertEqual(AecConfig().delay_backward_quarantine_s, 1.0)

    def test_the_pre_echo_mis_lock_no_longer_happens(self) -> None:
        """This scene used to be the defect; it is now the guard that the
        estimator fix stays.

        It previously acquired correctly and then re-locked to the pre-echo
        answer, holding it for the rest of the run -- which is what this
        quarantine was built to DELAY, never to cure. Curing it was estimator
        work, done at the selection seam: the aggregator now reports the
        dominant peak (see test_delay_dominant_selection). The scene is kept
        because an end-to-end assertion through AEC.process() is what would
        catch the substitution coming back by another route, which the
        aggregator-level test cannot see.

        Asserted with the mechanism OFF -- the shipped default -- so it is the
        estimator being tested and not the guard.
        """
        delays, _, _ = _mis_off()
        acquired = delays[_MIS_LOCK_HOP]
        self.assertGreater(acquired, 0, "a delay is acquired")
        self.assertEqual(set(delays[_MIS_LOCK_HOP:]), {acquired},
                         "the acquired delay is held for the rest of the run")
        self.assertNotIn(_MIS_WRONG, delays,
                         "the pre-echo answer (true - 1600) is never applied")
        # On the 64-sample grid, and closer to the truth than the value this
        # scene used to settle on before the seam was fixed.
        self.assertLessEqual(abs(acquired - _MIS_DELAY), 64,
                             "the applied delay is within one grid step of "
                             "the true bulk delay")

    def test_this_scene_never_engages_the_quarantine_at_all(self) -> None:
        """Stronger than the acquisition-prefix claim it replaces: with the
        selection seam fixed there is no backward candidate here to hold, so
        the whole trajectory must be identical on and off. Path A stays
        governed by delay_acquire_protect_converged, which this mechanism does
        not touch."""
        off, _, _ = _mis_off()
        on, _, _ = _mis_on()
        shared = min(len(off), len(on))
        self.assertEqual(off[:shared], on[:shared],
                         "enabling the guard changes nothing in this scene")

    def test_windowed_erle_is_the_arm_that_decides_a_real_backward_move(
            self) -> None:
        """Which arm is load-bearing, made falsifiable, at the hop the
        decision is actually taken.

        ⚠ The inst-ERLE arm no longer has a scene in this file that shows it
        deciding. The scene that used to -- the pre-echo mis-lock, where
        windowed ERLE read under its threshold while the inst peak read over
        delay_acquire_inst_erle_db -- does not occur any more now that the
        aggregator reports the dominant peak. That arm is therefore currently
        UNPROVEN here rather than shown unnecessary; finding or building a
        scene for it is follow-up work, not something to assert without one.
        """
        delays, wins, peaks = _bwd(False)
        hop = _first_change_after(delays, _BWD_AT)
        self.assertGreater(hop, 0, "the unguarded build accepts the move")
        self.assertGreater(wins[hop], 2.5,
                           f"windowed ERLE at the acceptance hop is "
                           f"{wins[hop]:.3f} dB -- over its own protection "
                           "threshold, so it is the arm that engages here")
        self.assertLess(peaks[hop], 4.0,
                        f"the inst-ERLE peak is {peaks[hop]:.3f} dB -- under "
                        "delay_acquire_inst_erle_db, so it is not")

    def test_a_forward_move_is_never_quarantined(self) -> None:
        off, _, _ = _fwd(False)
        on, _, _ = _fwd(True)
        self.assertGreater(_first_change_after(off, _FWD_AT), 0,
                           "control: the unguarded build accepts the move")
        self.assertEqual(off, on,
                         "a LARGER delay is not the backward direction, so "
                         "the whole trajectory is identical")
        self.assertTrue(_FWD_D1 - 256 < on[-1] < _FWD_D1 + 256,
                        "and the run ends on the moved delay")

    def test_a_backward_move_is_released_by_the_collapse_not_the_expiry(self) -> None:
        """The early-release arm's own row, and the windowed arm's. The old
        path is REMOVED here, so cancellation genuinely collapses: the
        quarantine engages (proving the direction test fires and the slow arm
        is what engaged it -- the inst ring has aged out by then) and then
        releases strictly INSIDE the window."""
        off, wins, peaks = _bwd(False)
        on, _, _ = _bwd(True)
        off_hop = _first_change_after(off, _BWD_AT)
        on_hop = _first_change_after(on, _BWD_AT)
        self.assertGreater(off_hop, 0, "control: the unguarded build accepts")
        self.assertLess(peaks[off_hop], 4.0,
                        "the fast arm has already released by then")
        self.assertGreater(wins[off_hop], 2.5,
                           "so anything still engaging is the slow arm")
        # The two semantic claims first, so each can fail on its own; the
        # exact hold is the tighter pin over the top of them, and would
        # otherwise be the only thing anyone ever saw go red.
        self.assertGreater(on_hop, off_hop,
                           "the quarantine engaged on this backward candidate")
        self.assertLess(on_hop - off_hop, _WINDOW_HOPS,
                        "and released strictly INSIDE the window, so what "
                        "released it was the cancellation collapse")
        self.assertEqual(on_hop - off_hop, _BWD_EXPECTED_DELTA,
                         "the measured hold")

    def test_multipath_path_addition_relocks_in_all_four_gain_rows(self) -> None:
        """The regression guard for the predicate this replaced: unbounded,
        the surviving old reflection vetoed the genuine new path forever;
        here every row re-locks, none later than unguarded by more than a
        window."""
        for gain in _MP_GAINS:
            off, on = _mp_delays(False, gain), _mp_delays(True, gain)
            off_hop = _first_change_after(off, _MP_AT)
            on_hop = _first_change_after(on, _MP_AT)
            with self.subTest(g_old=gain):
                self.assertGreater(off_hop, 0,
                                   "control: the unguarded build re-locks")
                self.assertGreater(on_hop, 0,
                                   "the guarded build re-locks too -- there "
                                   "is no permanent veto")
                self.assertGreaterEqual(on_hop, off_hop)
                self.assertLessEqual(on_hop - off_hop, _WINDOW_HOPS,
                                     "held by at most one window")
                self.assertTrue(_MP_NEW - 128 <= on[-1] <= _MP_NEW + 128,
                                "and ends on the new path")


if __name__ == '__main__':
    unittest.main()
