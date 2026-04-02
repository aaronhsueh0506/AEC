#!/usr/bin/env python3
"""Dump RES internal values per frame for C/Python comparison."""
import sys, os, types
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, AecPreset

mic, sr = sf.read('/tmp/synth_mic.wav', dtype='float32')
ref, _ = sf.read('/tmp/synth_far.wav', dtype='float32')
n = min(len(mic), len(ref)); mic, ref = mic[:n], ref[:n]

config = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
    filter_length=512, enable_shadow=True, enable_res=True, enable_highpass=True)
aec = AEC(config)
hop = aec.hop_size

# Monkey-patch RES to capture internals
res = aec.res
_diag = {}

orig_process = res.process.__func__
def patched_process(self, *args, **kwargs):
    result = orig_process(self, *args, **kwargs)
    # Capture key values at bins 10, 50, 128
    for bk in [10, 50, 128]:
        _diag[f'gain_{bk}'] = float(self.gain_smooth[bk])
        _diag[f'echo_psd_{bk}'] = float(self.echo_psd[bk])
        _diag[f'error_psd_{bk}'] = float(self.error_psd[bk])
    return result
res.process = types.MethodType(patched_process, res)

for i in range(50):
    s = i * hop
    out = aec.process(mic[s:s+hop], ref[s:s+hop])
    out_rms = np.sqrt(np.mean(out**2))
    if i >= 15 and i < 30:
        g10 = _diag.get('gain_10', 0)
        g50 = _diag.get('gain_50', 0)
        g128 = _diag.get('gain_128', 0)
        ep10 = _diag.get('echo_psd_10', 0)
        er10 = _diag.get('error_psd_10', 0)
        print(f'f={i:3d} out={out_rms:.6f} g10={g10:.6f} g50={g50:.6f} g128={g128:.6f} echo10={ep10:.2e} err10={er10:.2e}')
