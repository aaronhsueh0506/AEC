#!/usr/bin/env python3
"""Quick diagnostic: check FS worst cases ne_confidence with EMA smoothed ERLE."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from aec import AEC, AecConfig, AecMode

base = "/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"
fs_files = sorted(os.listdir(f"{base}/farend_singletalk/"))
fs_lpbs = [f for f in fs_files if f.endswith('_lpb.wav')]
fs_mics = [f for f in fs_files if f.endswith('_mic.wav')]

import soundfile as sf

print(f"{'Case':<8} {'ERLE_corr%':>10} {'Uncorr_dt':>10} {'ne_conf':>10} {'gain':>8}")
print("-" * 50)

for idx in [11, 14, 31, 94, 95, 0, 5]:
    if idx >= len(fs_lpbs):
        continue
    uuid = fs_lpbs[idx].split('_farend')[0]
    mic = [f for f in fs_mics if uuid in f]
    if not mic:
        continue

    far, sr = sf.read(f"{base}/farend_singletalk/{fs_lpbs[idx]}", dtype='float32')
    near, _ = sf.read(f"{base}/farend_singletalk/{mic[0]}", dtype='float32')
    if far.ndim > 1: far = far[:, 0]
    if near.ndim > 1: near = near[:, 0]
    min_len = min(len(far), len(near))
    far, near = far[:min_len], near[:min_len]

    config = AecConfig.from_preset('balanced', filter_length=512, enable_res=True, mode=AecMode.PBFDKF)
    aec = AEC(config)
    hop = config.hop_size
    n_frames = min_len // hop

    dt_inds = []
    gains = []
    for i in range(n_frames):
        f = far[i*hop:(i+1)*hop]
        n = near[i*hop:(i+1)*hop]
        out = aec.process(n, f)
        if np.mean(f**2) > 1e-4:
            # Reconstruct dt_indicator
            fp = np.mean(f**2) + 1e-10
            mp = np.mean(n**2) + 1e-10
            raw = 1.0 - fp / (mp + fp)
            ep = np.mean(out**2) + 1e-10
            ie = mp / ep
            # Use smoothed version from AEC
            if aec._inst_erle_smooth > 2.0:
                corrected = raw / aec._inst_erle_smooth
            else:
                corrected = raw
            dt_inds.append(np.clip(corrected, 0, 0.8))
            gains.append(float(np.mean(aec.res.gain_smooth)))

    dt_arr = np.array(dt_inds)
    g_arr = np.array(gains)
    fs_conf = aec.res.far_activity * (1.0 - dt_arr)**2
    ne_conf = 1.0 - fs_conf

    label = "WORST" if idx in [11,14,31,94,95] else "GOOD "
    print(f"{label} {idx:<3} {np.mean(dt_arr < 0.3)*100:>8.0f}%  {np.mean(dt_arr):>10.3f} {np.mean(ne_conf):>10.3f} {np.mean(g_arr):>8.3f}")
