#!/usr/bin/env python3
"""Analyze worst FS echo and DT deg cases for improvement opportunities."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
from aec import AEC, AecConfig, AecMode

base = "/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"

def analyze_case(lpb_path, mic_path, label, scenario):
    far, sr = sf.read(lpb_path, dtype='float32')
    near, _ = sf.read(mic_path, dtype='float32')
    if far.ndim > 1: far = far[:, 0]
    if near.ndim > 1: near = near[:, 0]
    min_len = min(len(far), len(near))
    far, near = far[:min_len], near[:min_len]

    config = AecConfig.from_preset('balanced', filter_length=512,
                                    enable_res=True, mode=AecMode.PBFDKF)
    aec = AEC(config)
    hop = config.hop_size
    n_frames = min_len // hop

    dt_ind = []
    gain = []
    erle_inst = []
    filter_erle = []
    enr_vals = []
    residual_ratio = []

    for i in range(n_frames):
        f = far[i*hop:(i+1)*hop]
        n = near[i*hop:(i+1)*hop]
        out = aec.process(n, f)

        fp = np.mean(f**2)
        if fp > 1e-4:
            mp = np.mean(n**2) + 1e-10
            ep = np.mean(out**2) + 1e-10
            raw_dt = 1.0 - fp / (mp + fp)
            ie = mp / ep
            if aec._inst_erle_smooth > 2.0:
                corrected = raw_dt / aec._inst_erle_smooth
            else:
                corrected = raw_dt
            dt_ind.append(np.clip(corrected, 0, 0.8))
            gain.append(float(np.mean(aec.res.gain_smooth)))
            erle_inst.append(float(ie))
            filter_erle.append(float(np.mean(aec.res._filter_erle_est.erle)))

            # ENR: reconstruct what the system computed
            res = aec.res
            if hasattr(res, 'error_psd') and np.mean(res.error_psd) > 1e-10:
                raw_ne = np.maximum(res.error_psd - res.echo_psd, 0.0)
                dt_val = dt_ind[-1]
                ne_est = np.maximum(raw_ne * dt_val, np.mean(res.error_psd) * 0.01 + 1e-10)
                enr_val = float(np.mean(res.echo_psd / ne_est))
                enr_vals.append(enr_val)
            else:
                enr_vals.append(0.0)

    dt_arr = np.array(dt_ind)
    gain_arr = np.array(gain)
    erle_arr = np.array(erle_inst)
    ferle_arr = np.array(filter_erle)
    enr_arr = np.array(enr_vals)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {os.path.basename(lpb_path)}")
    print(f"{'='*70}")
    print(f"  Active frames: {len(dt_arr)}/{n_frames}")

    if len(dt_arr) > 0:
        print(f"\n  {'Metric':<25} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
        print(f"  {'-'*57}")
        for name, arr in [('dt_indicator', dt_arr), ('inst_erle', erle_arr),
                           ('filter_erle', ferle_arr), ('ENR', enr_arr),
                           ('gain_smooth', gain_arr)]:
            print(f"  {name:<25} {np.mean(arr):8.3f} {np.percentile(arr,10):8.3f} "
                  f"{np.percentile(arr,50):8.3f} {np.percentile(arr,90):8.3f}")

        # Categorize frames
        low_gain = gain_arr < 0.1
        mid_gain = (gain_arr >= 0.1) & (gain_arr < 0.5)
        high_gain = gain_arr >= 0.5
        print(f"\n  Gain distribution: low(<0.1)={np.mean(low_gain)*100:.0f}%, "
              f"mid(0.1-0.5)={np.mean(mid_gain)*100:.0f}%, "
              f"high(>0.5)={np.mean(high_gain)*100:.0f}%")

        if scenario == 'fs':
            # For FS: high gain frames are bad (echo leaking)
            if np.sum(high_gain) > 0:
                print(f"\n  [Echo leak frames (gain>0.5): {np.sum(high_gain)} frames]")
                print(f"  dt_indicator: {np.mean(dt_arr[high_gain]):.3f}")
                print(f"  inst_erle:    {np.mean(erle_arr[high_gain]):.3f}")
                print(f"  filter_erle:  {np.mean(ferle_arr[high_gain]):.3f}")
                print(f"  ENR:          {np.mean(enr_arr[high_gain]):.3f}")
        elif scenario == 'dt':
            # For DT: low gain frames are bad (speech suppressed)
            if np.sum(low_gain) > 0:
                print(f"\n  [Over-suppression frames (gain<0.1): {np.sum(low_gain)} frames]")
                print(f"  dt_indicator: {np.mean(dt_arr[low_gain]):.3f}")
                print(f"  inst_erle:    {np.mean(erle_arr[low_gain]):.3f}")
                print(f"  filter_erle:  {np.mean(ferle_arr[low_gain]):.3f}")
                print(f"  ENR:          {np.mean(enr_arr[low_gain]):.3f}")


# FS worst cases (low echo MOS)
print("\n" + "=" * 70)
print("  FS WORST CASES (low echo suppression)")
print("=" * 70)
fs_files = sorted(os.listdir(f"{base}/farend_singletalk/"))
fs_lpbs = [f for f in fs_files if f.endswith('_lpb.wav')]
fs_mics = [f for f in fs_files if f.endswith('_mic.wav')]

for idx in [11, 31, 95]:  # Known worst FS cases
    if idx < len(fs_lpbs):
        uuid = fs_lpbs[idx].split('_farend')[0]
        mic = [f for f in fs_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/farend_singletalk/{fs_lpbs[idx]}",
                         f"{base}/farend_singletalk/{mic[0]}",
                         f"FS WORST (case {idx})", 'fs')

# FS good for comparison
for idx in [0, 5]:
    if idx < len(fs_lpbs):
        uuid = fs_lpbs[idx].split('_farend')[0]
        mic = [f for f in fs_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/farend_singletalk/{fs_lpbs[idx]}",
                         f"{base}/farend_singletalk/{mic[0]}",
                         f"FS GOOD (case {idx})", 'fs')

# DT worst cases (low deg MOS)
print("\n" + "=" * 70)
print("  DT WORST CASES (high speech distortion)")
print("=" * 70)
dt_files = sorted(os.listdir(f"{base}/doubletalk/"))
dt_lpbs = [f for f in dt_files if f.endswith('_lpb.wav')]
dt_mics = [f for f in dt_files if f.endswith('_mic.wav')]

for idx in [56, 189, 244]:  # Known worst DT cases
    if idx < len(dt_lpbs):
        uuid = dt_lpbs[idx].split('_doubletalk')[0]
        mic = [f for f in dt_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/doubletalk/{dt_lpbs[idx]}",
                         f"{base}/doubletalk/{mic[0]}",
                         f"DT WORST (case {idx})", 'dt')

# DT good for comparison
for idx in [0, 10]:
    if idx < len(dt_lpbs):
        uuid = dt_lpbs[idx].split('_doubletalk')[0]
        mic = [f for f in dt_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/doubletalk/{dt_lpbs[idx]}",
                         f"{base}/doubletalk/{mic[0]}",
                         f"DT GOOD (case {idx})", 'dt')
