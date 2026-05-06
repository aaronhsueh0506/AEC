#!/usr/bin/env python3
"""Score the 7GT P3h dry-run pair (baseline vs reset) with AECMOS."""
import sys, os
sys.path.insert(0, '/Users/mingyu/Desktop/novatek/SE/AEC/python')
import soundfile as sf
import numpy as np
from bench_aecmos import FastAECMOS, _MODEL_DIR

DATA = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind/doubletalk'
mic, _ = sf.read(f'{DATA}/7GTxyTksSUqCnP5y0ILG4A_doubletalk_mic.wav')
lpb, _ = sf.read(f'{DATA}/7GTxyTksSUqCnP5y0ILG4A_doubletalk_lpb.wav')
mic = np.asarray(mic, dtype=np.float32)
lpb = np.asarray(lpb, dtype=np.float32)

estimator = FastAECMOS(os.path.join(_MODEL_DIR, 'Run_1663915512_Stage_0.onnx'))

for label, path in [('baseline (v3.10.4)', '/tmp/p3h_baseline_7gt.wav'),
                    ('p3h diverged-reset', '/tmp/p3h_on_7gt.wav')]:
    enh, _ = sf.read(path)
    enh = np.asarray(enh, dtype=np.float32)
    n = min(len(mic), len(lpb), len(enh))
    e, d = estimator.score('dt', lpb[:n], mic[:n], enh[:n])
    print(f'{label:<22}  echo={e:.3f}  deg={d:.3f}')
