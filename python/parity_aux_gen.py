#!/usr/bin/env python3
"""Phase 2bcd parity baseline: RenderActivityDetector + DoubleTalkAnalyzer +
FilterConvergenceAnalyzer."""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aec import (RenderActivityDetector, DoubleTalkAnalyzer,
                 FilterConvergenceAnalyzer, AecConfig)

HOP = 160
N_FRAMES = 200
SEED = 42
SHADOW_DTD_OFFSET = 1.5
SHADOW_DTD_ADVANTAGE_SCALE = 3.0

OUT_FILE   = Path(__file__).parent / 'parity_aux_python.npz'
INPUT_FILE = Path(__file__).parent / 'parity_aux_input.bin'


def main():
    rng = np.random.default_rng(SEED)
    far_end = (rng.standard_normal(N_FRAMES * HOP) * 0.1).astype(np.float32)
    # Periodically silence to test active toggling
    for f in range(40, 50):
        far_end[f*HOP:(f+1)*HOP] = 0.0
    for f in range(120, 140):
        far_end[f*HOP:(f+1)*HOP] = 0.0

    # DTAnalyzer inputs (per frame)
    far_pwr_arr = (rng.uniform(0, 0.05, N_FRAMES)).astype(np.float32)
    mic_pwr_arr = (rng.uniform(0, 0.1, N_FRAMES)).astype(np.float32)
    erl_arr     = np.full(N_FRAMES, 0.1, dtype=np.float32)
    far_active  = (far_pwr_arr > 1e-4).astype(np.int32)
    main_err    = (rng.uniform(1e-4, 1e-2, N_FRAMES)).astype(np.float32)
    shadow_err  = (rng.uniform(1e-4, 1e-2, N_FRAMES)).astype(np.float32)
    shadow_far  = (rng.standard_normal(N_FRAMES) > -0.5).astype(np.int32)

    # FilterConvergence inputs
    near_pwr = (rng.uniform(1e-7, 1e-2, N_FRAMES)).astype(np.float32)
    raw_err  = (rng.uniform(1e-9, 1e-3, N_FRAMES)).astype(np.float32)
    warmup_done_arr = np.array([1] * N_FRAMES, dtype=np.int32)
    warmup_done_arr[:30] = 0  # warmup first 30 frames

    # Run Python
    ra = RenderActivityDetector()
    dta = DoubleTalkAnalyzer(AecConfig(
        sample_rate=16000,
        shadow_dtd_offset=SHADOW_DTD_OFFSET,
        shadow_dtd_advantage_scale=SHADOW_DTD_ADVANTAGE_SCALE))
    fc = FilterConvergenceAnalyzer()

    ra_active = np.zeros(N_FRAMES, dtype=np.int32)
    ra_stationary = np.zeros(N_FRAMES, dtype=np.int32)
    ra_far_pwr = np.zeros(N_FRAMES, dtype=np.float32)

    dt_energy = np.zeros(N_FRAMES, dtype=np.float32)
    dt_shadow = np.zeros(N_FRAMES, dtype=np.float32)
    dt_advantage = np.zeros(N_FRAMES, dtype=np.float32)

    fc_converged = np.zeros(N_FRAMES, dtype=np.int32)
    fc_once = np.zeros(N_FRAMES, dtype=np.int32)
    fc_div = np.zeros(N_FRAMES, dtype=np.float32)

    for f in range(N_FRAMES):
        st = ra.update(far_end[f*HOP:(f+1)*HOP])
        ra_active[f]     = int(st.is_active)
        ra_stationary[f] = int(st.is_stationary)
        ra_far_pwr[f]    = st.far_pwr

        # DTAnalyzer
        # shadow_frame_count: f (1-indexed-ish, approximate)
        dta.update_shadow_dt(shadow_frame_count=f, far_excited=bool(shadow_far[f]),
                             main_err_smooth=float(main_err[f]),
                             shadow_err_smooth=float(shadow_err[f]))
        dta.update_energy_dt(far_active=bool(far_active[f]),
                             far_pwr=float(far_pwr_arr[f]),
                             mic_pwr=float(mic_pwr_arr[f]),
                             erl_estimate=float(erl_arr[f]))
        dt_energy[f] = dta.dt_from_energy
        dt_shadow[f] = dta.dt_from_shadow
        dt_advantage[f] = dta.shadow_advantage

        # FilterConvergence
        fc.update_divergence(float(near_pwr[f]), float(raw_err[f]))
        fc.update_convergence(near_power=float(near_pwr[f]),
                              raw_error_power=float(raw_err[f]),
                              far_active=bool(far_active[f]),
                              warmup_done=bool(warmup_done_arr[f]))
        fc_converged[f] = int(fc.converged)
        fc_once[f] = int(fc.once_converged)
        fc_div[f] = fc.divergence

    np.savez(OUT_FILE,
             ra_active=ra_active, ra_stationary=ra_stationary,
             ra_far_pwr=ra_far_pwr,
             dt_energy=dt_energy, dt_shadow=dt_shadow,
             dt_advantage=dt_advantage,
             fc_converged=fc_converged, fc_once=fc_once, fc_div=fc_div,
             config=np.array([HOP, N_FRAMES], dtype=np.int32))

    with open(INPUT_FILE, 'wb') as fp:
        fp.write(far_end.tobytes())
        fp.write(far_pwr_arr.tobytes())
        fp.write(mic_pwr_arr.tobytes())
        fp.write(erl_arr.tobytes())
        fp.write(far_active.tobytes())
        fp.write(main_err.tobytes())
        fp.write(shadow_err.tobytes())
        fp.write(shadow_far.tobytes())
        fp.write(near_pwr.tobytes())
        fp.write(raw_err.tobytes())
        fp.write(warmup_done_arr.tobytes())

    print(f"Saved Python baseline → {OUT_FILE}")
    print(f"Saved C input bin     → {INPUT_FILE}")
    print(f"  ra final: active={ra_active[-1]} stat={ra_stationary[-1]} pwr={ra_far_pwr[-1]:.3e}")
    print(f"  dt final: energy={dt_energy[-1]:.3e} shadow={dt_shadow[-1]:.3e} adv={dt_advantage[-1]:.3e}")
    print(f"  fc final: converged={fc_converged[-1]} once={fc_once[-1]} div={fc_div[-1]:.3e}")


if __name__ == '__main__':
    main()
