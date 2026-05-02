#!/usr/bin/env python3
"""Synthetic parity test for RenderActivityDetector + FilterConvergenceAnalyzer
+ DoubleTalkAnalyzer (Wave 1.5/1.6/1.7).

Layout (LE):
  header: i32 hop, i32 n_frames
  per frame:
    f32 far_end[hop]
    f64 near_power, raw_error_power, mic_power, erl_estimate,
        main_err_smooth, shadow_err_smooth
    i32 far_active, warmup_done, far_excited, shadow_frame_count
    --- expected outputs after this frame ---
    f64 ra_far_pwr; i32 ra_is_active, ra_is_stationary, ra_warmup_active
    i32 fc_converged, fc_once_converged, fc_just_converged
    f64 fc_divergence
    f64 dt_from_energy, dt_from_shadow, shadow_advantage
"""
import sys, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from aec import (RenderActivityDetector, FilterConvergenceAnalyzer,
                 DoubleTalkAnalyzer, AecConfig, AecPreset)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/detectors.bin'
    hop = 160
    n_frames = 500
    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000)
    rng = np.random.default_rng(0xCAFE)

    ra = RenderActivityDetector()
    fc = FilterConvergenceAnalyzer()
    dt = DoubleTalkAnalyzer(cfg)

    with open(out, 'wb') as f:
        f.write(struct.pack('<ii', hop, n_frames))
        for fi in range(n_frames):
            # mix of silence, white noise, stationary noise, transient
            phase = fi // 100
            if phase == 0:                # silence
                far = np.zeros(hop, dtype=np.float32)
            elif phase == 1:              # ramp up
                far = (0.01 * fi * rng.standard_normal(hop)).astype(np.float32)
            elif phase == 2:              # stationary white noise
                far = (0.1 * rng.standard_normal(hop)).astype(np.float32)
            elif phase == 3:              # transient bursts
                far = (0.5 * np.sin(np.arange(hop) * 0.05 + fi * 0.3)
                       * rng.standard_normal(hop)).astype(np.float32)
            else:                          # mixed
                far = (0.3 * rng.standard_normal(hop)).astype(np.float32)

            near_power       = 1e-3 * (1.0 + 0.5 * np.sin(fi * 0.05))
            raw_error_power  = near_power * (0.1 + 0.5 * np.exp(-fi / 100.0))
            mic_power        = near_power * 1.5
            erl_estimate     = 0.1 + 0.05 * np.cos(fi * 0.02)
            main_err_smooth  = 1e-4 * (1.0 + np.sin(fi * 0.07))
            shadow_err_smooth= 1e-4 * (0.5 + 0.3 * np.cos(fi * 0.03))
            far_active        = bool(fi > 50 and phase != 0)
            warmup_done       = bool(fi > 30)
            far_excited       = bool(fi > 60 and fi < 400)
            shadow_frame_count= max(0, fi - 20)

            f.write(far.tobytes())
            f.write(struct.pack('<dddddd',
                                near_power, raw_error_power, mic_power,
                                erl_estimate, main_err_smooth, shadow_err_smooth))
            f.write(struct.pack('<iiii',
                                int(far_active), int(warmup_done),
                                int(far_excited), shadow_frame_count))

            # Run Python in same order as C will:
            ra_state = ra.update(far)
            fc.update_divergence(near_power, raw_error_power)
            just = fc.update_convergence(near_power=near_power,
                                          raw_error_power=raw_error_power,
                                          far_active=far_active,
                                          warmup_done=warmup_done)
            dt.update_shadow_dt(shadow_frame_count=shadow_frame_count,
                                far_excited=far_excited,
                                main_err_smooth=main_err_smooth,
                                shadow_err_smooth=shadow_err_smooth)
            dt.update_energy_dt(far_active=far_active, far_pwr=ra_state.far_pwr,
                                mic_pwr=mic_power, erl_estimate=erl_estimate)

            f.write(struct.pack('<diii',
                                ra_state.far_pwr, int(ra_state.is_active),
                                int(ra_state.is_stationary),
                                int(ra_state.warmup_active)))
            f.write(struct.pack('<iiid',
                                int(fc.converged), int(fc.once_converged),
                                int(just), float(fc.divergence)))
            f.write(struct.pack('<ddd',
                                float(dt.dt_from_energy),
                                float(dt.dt_from_shadow),
                                float(dt.shadow_advantage)))

    print(f'Wrote {n_frames} frames → {out}')


if __name__ == '__main__':
    main()
