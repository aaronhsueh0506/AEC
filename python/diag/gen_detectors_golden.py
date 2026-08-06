"""Generate the binary golden for the C detectors port.

Runs the Python RenderActivityDetector / FilterConvergenceAnalyzer /
DoubleTalkAnalyzer over a deterministic stimulus and writes inputs + expected
per-frame state to a raw little-endian file that
``c_impl/test/modules/parity_detectors.c`` replays.

The harness existed with no generator and no Makefile target, so nothing ever
ran it. That is how ``SHADOW_FRAME_GATE`` drifted: commit 6cd995e (2026-06-12)
changed the Python gate 50 -> 20 and never mirrored it to C, and no gate
noticed for eight weeks.

The stimulus is engineered to exercise every branch, in particular the ones a
naive random signal would miss:
  * far silence -> ``_active_prev`` False, ``_is_stationary`` False
  * the FIRST audible far hop -> variance is undefined from one observation, so
    stationarity must NOT be declared (the branch C got wrong until 2026-08-06)
  * a long constant-amplitude far stretch -> CV^2 collapses -> stationary True
  * an amplitude-modulated stretch -> CV^2 rises -> stationary False
  * far returning to silence -> the latch drops, then re-arms
  * ERLE above CONV_ERLE_DB sustained past CONV_FRAMES -> converged latches
  * non-qualifying hops interleaved mid-run: these must SKIP the counter, not
    reset it (this is what makes CONV_FRAMES an evidence count, not a duration)
  * a qualifying-but-failing hop -> counter resets to 0
  * post-convergence divergence EMA rise and its else-branch decay
  * ``shadow_frame_count`` swept slowly across [0, 60) so it passes through the
    whole [20, 50) window where a Python gate of 20 and a C gate of 50 disagree.
    Any future divergence in that constant fails this golden on those frames.

The header carries ``sample_rate`` as well as ``hop``: every wall-clock constant
in detectors.py/.c is now retimed from (hop, sample_rate), and hop alone cannot
disambiguate the grids -- hop=128 is both 8 kHz (16.000 ms) and 16 kHz
(8.000 ms), which retime to different coefficients.

Layout (LE), matching parity_detectors.c exactly:
  int32 hop, int32 sample_rate, int32 n_frames
  n_frames x [ far[hop] f32
             | near_pwr f64 | raw_err_pwr f64 | mic_pwr f64
             | erl_est f64 | main_err f64 | shadow_err f64
             | far_active i32 | warmup_done i32 | far_excited i32
             | shadow_count i32
             | exp_far_pwr f64 | exp_active i32 | exp_stat i32 | exp_warm i32
             | exp_conv i32 | exp_once i32 | exp_just i32 | exp_div f64
             | exp_dte f64 | exp_dts f64 | exp_adv f64 ]

Run: python3 python/diag/gen_detectors_golden.py /tmp/det_golden.bin [hop]
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.detectors import (  # noqa: E402
    DoubleTalkAnalyzer,
    FilterConvergenceAnalyzer,
    RenderActivityDetector,
)

# doubletalk_init(&dt, 1.5, 3.0) in the C harness -- these are the AecConfig
# BALANCED values for the two fields DoubleTalkAnalyzer actually reads.
SHADOW_DTD_OFFSET = 1.5
SHADOW_DTD_ADVANTAGE_SCALE = 3.0


class _Cfg:
    """Minimal stand-in for AecConfig: the analyzer reads only these four."""

    shadow_dtd_offset = SHADOW_DTD_OFFSET
    shadow_dtd_advantage_scale = SHADOW_DTD_ADVANTAGE_SCALE

    def __init__(self, hop_size, sample_rate):
        self.hop_size = hop_size
        self.sample_rate = sample_rate


def build_stimulus(hop, n_frames, rng):
    """Yield one (far, scalars) tuple per frame.

    Segment plan (n_frames = 260):
       0- 19  silence                      far latch off
      20- 99  constant-amplitude tone      first-active hop, then stationary
     100-139  amplitude-modulated          CV^2 rises, stationary drops
     140-159  silence                      latch drops
     160-259  constant again               latch re-arms from scratch
    """
    for fi in range(n_frames):
        if fi < 20 or 140 <= fi < 160:
            far = np.zeros(hop, dtype=np.float32)
        elif 100 <= fi < 140:
            # Modulated: envelope swings hop-to-hop so CV^2 climbs.
            amp = 0.05 + 0.45 * ((fi - 100) % 4) / 3.0
            far = (amp * rng.standard_normal(hop)).astype(np.float32)
        else:
            far = (0.25 * rng.standard_normal(hop)).astype(np.float32)

        far_active = int(fi >= 20 and not (140 <= fi < 160))
        # warmup_done False for the first 30 frames: update_convergence must
        # SKIP (not reset) the counter on those hops.
        warmup_done = int(fi >= 30)

        near_pwr = 1.0e-2
        if fi < 40:
            # Below CONV_ERLE_DB -> counter cannot advance.
            raw_err_pwr = near_pwr / (10.0 ** (2.0 / 10.0))
        elif fi == 70:
            # One qualifying-but-failing hop: this RESETS the counter.
            raw_err_pwr = near_pwr / (10.0 ** (1.0 / 10.0))
        elif 200 <= fi < 240:
            # Post-convergence collapse -> inst ERLE < DIV_ERLE_LIN -> the
            # divergence EMA rises; after 240 it decays on the else-branch.
            raw_err_pwr = near_pwr / 0.4
        else:
            raw_err_pwr = near_pwr / (10.0 ** (9.0 / 10.0))

        if 55 <= fi < 60:
            # near_power <= 1e-8 -> early return, counter SKIPPED not reset.
            near_pwr_eff = 1.0e-12
        else:
            near_pwr_eff = near_pwr

        # erl_est must be LARGE (small erl_ceiling) and mic_pwr comparable to
        # far_pwr, or update_energy_dt's `inst` pins to 0 for the whole run and
        # both DTE branches just multiply zero -- which silently leaves
        # DTE_RISE_*/DTE_DECAY_* untested. A mutation test caught exactly that.
        # mic_pwr swings so `inst` crosses dt_from_energy in both directions,
        # exercising the rise branch and the decay branch.
        erl_est = 0.8
        mic_pwr = 0.20 + 0.80 * ((fi % 10) / 9.0)
        main_err = 1.0e-3 * (1.0 + 0.5 * ((fi % 11) / 10.0))
        shadow_err = 1.0e-3 * (0.4 + 0.9 * ((fi % 13) / 12.0))
        far_excited = int(far_active and (fi % 9) != 0)
        # Sweep slowly through [0, 60) so every value in the [20, 50) window
        # where a 20-gate and a 50-gate disagree is visited on several frames.
        shadow_count = min(60, fi // 4)

        yield (far, near_pwr_eff, raw_err_pwr, mic_pwr, erl_est,
               main_err, shadow_err, far_active, warmup_done,
               far_excited, shadow_count)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/det_golden.bin'
    hop = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    sample_rate = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    n_frames = 260

    rng = np.random.default_rng(20260806)
    ra = RenderActivityDetector(hop, sample_rate)
    fc = FilterConvergenceAnalyzer(hop, sample_rate)
    dt = DoubleTalkAnalyzer(_Cfg(hop, sample_rate))

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<iii', hop, sample_rate, n_frames))
        for (far, near_pwr, raw_err_pwr, mic_pwr, erl_est, main_err,
             shadow_err, far_active, warmup_done, far_excited,
             shadow_count) in build_stimulus(hop, n_frames, rng):

            f.write(far.astype('<f4').tobytes())
            f.write(struct.pack('<6d', near_pwr, raw_err_pwr, mic_pwr,
                                erl_est, main_err, shadow_err))
            f.write(struct.pack('<4i', far_active, warmup_done,
                                far_excited, shadow_count))

            # Call order MUST match parity_detectors.c frame body exactly.
            ra_state = ra.update(far)
            fc.update_divergence(near_pwr, raw_err_pwr)
            just = fc.update_convergence(
                near_power=near_pwr, raw_error_power=raw_err_pwr,
                far_active=bool(far_active), warmup_done=bool(warmup_done))
            dt.update_shadow_dt(shadow_frame_count=shadow_count,
                                far_excited=bool(far_excited),
                                main_err_smooth=main_err,
                                shadow_err_smooth=shadow_err)
            dt.update_energy_dt(far_active=bool(far_active),
                                far_pwr=ra_state.far_pwr,
                                mic_pwr=mic_pwr, erl_estimate=erl_est)

            f.write(struct.pack('<d', float(ra_state.far_pwr)))
            f.write(struct.pack('<3i', int(ra_state.is_active),
                                int(ra_state.is_stationary),
                                int(ra_state.warmup_active)))
            f.write(struct.pack('<3i', int(fc.converged),
                                int(fc.once_converged), int(just)))
            f.write(struct.pack('<d', float(fc.divergence)))
            f.write(struct.pack('<3d', float(dt.dt_from_energy),
                                float(dt.dt_from_shadow),
                                float(dt.shadow_advantage)))

    print(f'wrote {out_path}: hop={hop} sr={sample_rate} n_frames={n_frames}')


if __name__ == '__main__':
    main()
