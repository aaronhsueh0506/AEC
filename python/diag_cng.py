#!/usr/bin/env python3
"""Diagnostic: trace CNG behavior — noise_psd, cn_gain, gain_smooth on FS cases."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import soundfile as sf
from aec import AEC, AecConfig, AecMode

base = "/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"

def analyze_cng(lpb_path, mic_path, label):
    far, sr = sf.read(lpb_path, dtype='float32')
    near, _ = sf.read(mic_path, dtype='float32')
    if far.ndim > 1: far = far[:, 0]
    if near.ndim > 1: near = near[:, 0]
    min_len = min(len(far), len(near))
    far, near = far[:min_len], near[:min_len]

    config = AecConfig.from_preset('balanced', filter_length=512,
                                    enable_res=True, enable_cng=True, mode=AecMode.PBFDKF)
    aec = AEC(config)
    hop = config.hop_size
    n_frames = min_len // hop

    noise_psd_trace = []
    error_psd_trace = []
    gain_trace = []
    cn_gain_trace = []

    for i in range(n_frames):
        f = far[i*hop:(i+1)*hop]
        n = near[i*hop:(i+1)*hop]
        aec.process(n, f)

        noise_mean = float(np.mean(aec.res.noise_psd))
        error_mean = float(np.mean(aec.res.error_psd))
        gain_mean = float(np.mean(aec.res.gain_smooth))
        cn_gain_val = float(np.mean(np.sqrt(np.maximum(1.0 - aec.res.gain_smooth**2, 0.0))))

        noise_psd_trace.append(noise_mean)
        error_psd_trace.append(error_mean)
        gain_trace.append(gain_mean)
        cn_gain_trace.append(cn_gain_val)

    noise = np.array(noise_psd_trace)
    error = np.array(error_psd_trace)
    gain = np.array(gain_trace)
    cn_gain = np.array(cn_gain_trace)

    far_pwr = np.array([np.mean(far[i*hop:(i+1)*hop]**2) for i in range(n_frames)])
    active = far_pwr > 1e-4

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {os.path.basename(lpb_path)}")
    print(f"{'='*70}")
    print(f"  Total frames: {n_frames}, Active: {np.sum(active)}")

    if np.sum(active) > 0:
        print(f"\n  [Active frames]")
        print(f"  {'Metric':<25} {'Mean':>12} {'P10':>12} {'P50':>12} {'P90':>12}")
        print(f"  {'-'*65}")
        for name, arr in [('noise_psd_mean', noise),
                           ('error_psd_mean', error),
                           ('noise/error ratio', noise / (error + 1e-10)),
                           ('gain_smooth', gain),
                           ('cn_gain (sqrt(1-G²))', cn_gain)]:
            v = arr[active]
            print(f"  {name:<25} {np.mean(v):12.6f} {np.percentile(v,10):12.6f} {np.percentile(v,50):12.6f} {np.percentile(v,90):12.6f}")

        # Key question: is noise_psd tracking echo or actual noise?
        silence = ~active
        if np.sum(silence) > 5:
            print(f"\n  [Silent frames ({np.sum(silence)} frames)]")
            print(f"  noise_psd_mean:  {np.mean(noise[silence]):12.6f}")
            print(f"  error_psd_mean:  {np.mean(error[silence]):12.6f}")
            print(f"  ratio:           {np.mean(noise[silence] / (error[silence] + 1e-10)):12.6f}")

        # How much CNG energy vs suppressed energy?
        # When gain is low (strong suppression), cn_gain should be high
        deep_suppress = gain[active] < 0.3
        if np.sum(deep_suppress) > 0:
            print(f"\n  [Deep suppression frames (gain<0.3): {np.sum(deep_suppress)} frames]")
            print(f"  gain_mean:       {np.mean(gain[active][deep_suppress]):12.6f}")
            print(f"  cn_gain_mean:    {np.mean(cn_gain[active][deep_suppress]):12.6f}")
            print(f"  noise_psd_mean:  {np.mean(noise[active][deep_suppress]):12.6f}")
            print(f"  error_psd_mean:  {np.mean(error[active][deep_suppress]):12.6f}")
            print(f"  noise/error:     {np.mean(noise[active][deep_suppress] / (error[active][deep_suppress] + 1e-10)):12.6f}")


# FS cases: good (case 0, 5) and worst (case 11, 14)
fs_files = sorted(os.listdir(f"{base}/farend_singletalk/"))
fs_lpbs = [f for f in fs_files if f.endswith('_lpb.wav')]
fs_mics = [f for f in fs_files if f.endswith('_mic.wav')]

for idx in [0, 5, 11, 14]:
    if idx < len(fs_lpbs):
        uuid = fs_lpbs[idx].split('_farend')[0]
        mic = [f for f in fs_mics if uuid in f]
        if mic:
            label = "FS GOOD" if idx in [0, 5] else "FS WORST"
            analyze_cng(f"{base}/farend_singletalk/{fs_lpbs[idx]}",
                        f"{base}/farend_singletalk/{mic[0]}",
                        f"{label} (case {idx})")

# DT case for comparison
dt_files = sorted(os.listdir(f"{base}/doubletalk/"))
dt_lpbs = [f for f in dt_files if f.endswith('_lpb.wav')]
dt_mics = [f for f in dt_files if f.endswith('_mic.wav')]
if len(dt_lpbs) > 0:
    uuid = dt_lpbs[0].split('_doubletalk')[0]
    mic = [f for f in dt_mics if uuid in f]
    if mic:
        analyze_cng(f"{base}/doubletalk/{dt_lpbs[0]}",
                    f"{base}/doubletalk/{mic[0]}",
                    "DT (case 0)")
