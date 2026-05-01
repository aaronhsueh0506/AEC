#!/usr/bin/env python3
"""Run Python AEC and dump per-frame state to binary file for C parity diff.

Format (per frame, 80 bytes):
  u32 frame_idx
  f32 mu_scale, erle_factor, dt_indicator, raw_dt
  f32 far_activity, far_power, mic_power, raw_err_power
  f32 erl_estimate, main_err_smooth, shadow_err_smooth
  f32 simple_mu_ratio, dt_from_energy, dt_from_shadow
  f32 gain_mean, output_energy
  u32 filter_converged, filter_once_converged, using_render_based, epc_active
"""
import sys, struct, numpy as np, soundfile as sf
sys.path.insert(0, '/Users/mingyu/Desktop/novatek/SE/AEC/python')
from aec import AEC, AecConfig, AecPreset


def main():
    if len(sys.argv) < 4:
        print("Usage: dump_state.py <mic.wav> <ref.wav> <out.bin>"); sys.exit(1)
    mic_path, ref_path, out_path = sys.argv[1:4]
    mic, _ = sf.read(mic_path); ref, _ = sf.read(ref_path)
    n = min(len(mic), len(ref))
    mic = mic[:n].astype(np.float32); ref = ref[:n].astype(np.float32)

    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=16000, enable_cng=True)
    aec = AEC(cfg)
    hop = aec._hop_size

    fp = open(out_path, 'wb')
    # header: u32 hop, n_frames
    n_frames = (n - hop) // hop + 1
    fp.write(struct.pack('<II', hop, n_frames))
    for i_frame, i in enumerate(range(0, n - hop, hop)):
        m = mic[i:i+hop]; r = ref[i:i+hop]
        far_pwr = float(np.mean(r ** 2))
        mic_pwr = float(np.mean(m ** 2))
        out = aec.process(m, r)
        d = aec._diag

        raw_err_pwr = float(aec.raw_error_power) if hasattr(aec, 'raw_error_power') else 0.0
        record = struct.pack('<I' + 'f' * 16 + 'I' * 4,
            i_frame,
            float(d.get('mu_scale', 1.0)),
            float(d.get('erle_factor', 0.0)),
            float(d.get('dt_indicator', 0.0)),
            0.0,  # raw_dt not exported
            float(d.get('far_activity', 0.0)),
            far_pwr, mic_pwr, raw_err_pwr,
            float(d.get('erl_estimate', 0.1)),
            float(d.get('main_err_smooth', 0.0)),
            float(d.get('shadow_err_smooth', 0.0)),
            float(getattr(aec, '_simple_mu_ratio', 1.0)),
            float(d.get('dt_from_energy', 0.0)),
            float(d.get('dt_from_shadow', 0.0)),
            float(d.get('res_gain_mean', 1.0)),
            float(np.mean(out ** 2)),
            int(bool(d.get('converged', False))),
            int(bool(d.get('filter_once_converged', False))),
            int(bool(d.get('using_render_based', False))),
            int(bool(d.get('epc_active', False))),
        )
        fp.write(record)
    fp.close()
    print(f"Dumped {n_frames} frames → {out_path}")


if __name__ == '__main__':
    main()
