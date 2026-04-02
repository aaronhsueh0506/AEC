#!/usr/bin/env python3
"""Frame-by-frame comparison of C vs Python AEC output to find divergence."""
import numpy as np
import soundfile as sf
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, AecPreset

# Use synthetic test
mic, sr = sf.read('/tmp/synth_mic.wav', dtype='float32')
ref, _ = sf.read('/tmp/synth_far.wav', dtype='float32')
n = min(len(mic), len(ref))
mic, ref = mic[:n], ref[:n]

# Python: match C config exactly (shadow on, no RES, HPF on)
config = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
    filter_length=512, enable_shadow=True, enable_res=False, enable_highpass=True)
aec = AEC(config)
hop = aec.hop_size

# Read C output
c_out, _ = sf.read('/tmp/c_hpf.wav', dtype='float32')

py_out = np.zeros(n, dtype=np.float32)
for i in range(n // hop):
    s = i * hop
    py_out[s:s+hop] = aec.process(mic[s:s+hop], ref[s:s+hop])

    c_frame = c_out[s:s+hop] if s + hop <= len(c_out) else np.zeros(hop)
    p_frame = py_out[s:s+hop]

    c_rms = np.sqrt(np.mean(c_frame**2))
    p_rms = np.sqrt(np.mean(p_frame**2))
    diff = np.max(np.abs(c_frame - p_frame))

    if i < 30 or (diff > 0.001 and i < 100):
        corr = np.corrcoef(c_frame, p_frame)[0,1] if c_rms > 1e-8 and p_rms > 1e-8 else 0
        print(f'f={i:4d} C_rms={c_rms:.6f} Py_rms={p_rms:.6f} diff={diff:.6f} corr={corr:.4f}')
