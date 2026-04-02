#!/usr/bin/env python3
"""Diagnostic: trace dt_indicator, fs_confidence, ne_confidence, gain, filter_erle for FS worst cases."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from aec import AEC, AecConfig, AecMode


def analyze_case(lpb_path, mic_path, label):
    import soundfile as sf

    far, sr = sf.read(lpb_path, dtype='float32')
    near, _ = sf.read(mic_path, dtype='float32')
    if far.ndim > 1: far = far[:, 0]
    if near.ndim > 1: near = near[:, 0]

    min_len = min(len(far), len(near))
    far = far[:min_len]
    near = near[:min_len]

    config = AecConfig.from_preset('balanced', filter_length=512,
                                    enable_res=True, mode=AecMode.PBFDKF)
    aec = AEC(config)
    hop = config.hop_size

    n_frames = min_len // hop
    dt_ind_trace = []
    inst_erle_trace = []
    raw_dt_trace = []
    gain_trace = []
    filter_erle_trace = []
    confidence_trace = []
    enr_trace = []

    for i in range(n_frames):
        f = far[i*hop:(i+1)*hop]
        n = near[i*hop:(i+1)*hop]
        output = aec.process(n, f)

        # Capture dt_indicator from AEC (computed inside process)
        far_pwr = np.mean(f ** 2) + 1e-10
        mic_pwr = np.mean(n ** 2) + 1e-10
        raw_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
        raw_err_pwr = np.mean(output ** 2) + 1e-10
        inst_erle = mic_pwr / raw_err_pwr
        if inst_erle > 2.0:
            corrected_dt = raw_dt / inst_erle
        else:
            corrected_dt = raw_dt
        dt_ind = np.clip(corrected_dt, 0.0, 0.8)

        dt_ind_trace.append(float(dt_ind))
        inst_erle_trace.append(float(inst_erle))
        raw_dt_trace.append(float(raw_dt))
        gain_trace.append(float(np.mean(aec.res.gain_smooth)))
        filter_erle_trace.append(float(np.mean(aec.res._filter_erle_est.erle)))
        from aec import compute_erle_confidence
        conf = compute_erle_confidence(aec.res._filter_erle_est.erle, aec.res._fb_erle_est.fb_erle)
        confidence_trace.append(conf)

    dt_ind = np.array(dt_ind_trace)
    inst_erle = np.array(inst_erle_trace)
    raw_dt = np.array(raw_dt_trace)
    gain = np.array(gain_trace)
    filter_erle = np.array(filter_erle_trace)
    confidence = np.array(confidence_trace)

    far_pwr_per_frame = np.array([np.mean(far[i*hop:(i+1)*hop]**2) for i in range(n_frames)])
    active = far_pwr_per_frame > 1e-4

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {os.path.basename(lpb_path)}")
    print(f"{'='*70}")
    print(f"  Total frames: {n_frames}, Active (far>1e-4): {np.sum(active)}")

    if np.sum(active) > 0:
        a = active
        print(f"\n  [Active frames only]")
        print(f"  {'Metric':<25} {'Mean':>8} {'P10':>8} {'P50':>8} {'P90':>8}")
        print(f"  {'-'*57}")
        for name, arr in [('raw_dt', raw_dt), ('inst_erle_fast', inst_erle),
                           ('dt_indicator', dt_ind),
                           ('fs_confidence', aec.res.far_activity * (1.0 - dt_ind[a])**2),
                           ('ne_confidence', 1.0 - aec.res.far_activity * (1.0 - dt_ind[a])**2),
                           ('gain_mean', gain), ('filter_erle_mean', filter_erle),
                           ('erle_confidence', confidence)]:
            if len(arr) == np.sum(a):
                v = arr
            else:
                v = arr[a]
            print(f"  {name:<25} {np.mean(v):8.3f} {np.percentile(v,10):8.3f} {np.percentile(v,50):8.3f} {np.percentile(v,90):8.3f}")

        # ERLE correction effectiveness
        corrected = inst_erle[a] > 2.0
        pct_corrected = np.mean(corrected) * 100
        print(f"\n  ERLE correction triggered: {pct_corrected:.0f}% of active frames")
        if pct_corrected > 0:
            print(f"  When triggered: inst_erle mean={np.mean(inst_erle[a][corrected]):.1f}, "
                  f"raw_dt mean={np.mean(raw_dt[a][corrected]):.3f} → dt_ind mean={np.mean(dt_ind[a][corrected]):.3f}")
        not_corrected = ~corrected
        if np.sum(not_corrected) > 0:
            print(f"  When NOT triggered: inst_erle mean={np.mean(inst_erle[a][not_corrected]):.2f}, "
                  f"raw_dt={np.mean(raw_dt[a][not_corrected]):.3f} → dt_ind={np.mean(dt_ind[a][not_corrected]):.3f}")


base = "/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"
fs_files = sorted(os.listdir(f"{base}/farend_singletalk/"))
fs_lpbs = [f for f in fs_files if f.endswith('_lpb.wav')]
fs_mics = [f for f in fs_files if f.endswith('_mic.wav')]

# FS worst cases from AECMOS: cases 11, 14, 31, 94, 95, 30
worst_indices = [11, 14, 31, 94, 95]
# Also a good FS case for comparison
good_indices = [0, 5]

for idx in worst_indices:
    if idx < len(fs_lpbs):
        uuid = fs_lpbs[idx].split('_farend')[0]
        mic = [f for f in fs_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/farend_singletalk/{fs_lpbs[idx]}",
                         f"{base}/farend_singletalk/{mic[0]}",
                         f"FS WORST (case {idx})")

for idx in good_indices:
    if idx < len(fs_lpbs):
        uuid = fs_lpbs[idx].split('_farend')[0]
        mic = [f for f in fs_mics if uuid in f]
        if mic:
            analyze_case(f"{base}/farend_singletalk/{fs_lpbs[idx]}",
                         f"{base}/farend_singletalk/{mic[0]}",
                         f"FS GOOD (case {idx})")
