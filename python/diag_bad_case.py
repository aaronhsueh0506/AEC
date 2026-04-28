#!/usr/bin/env python3
"""Diagnostic: per-frame trace of worst FS case to find why suppression fails."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))

import numpy as np
import soundfile as sf
from aec import AEC, AecConfig, AecPreset

MIC  = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind/farend_singletalk/7GTxyTksSUqCnP5y0ILG4A_farend_singletalk_mic.wav'
LPB  = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind/farend_singletalk/7GTxyTksSUqCnP5y0ILG4A_farend_singletalk_lpb.wav'

mic, sr = sf.read(MIC)
lpb, _  = sf.read(LPB)
mic = mic.astype(np.float32)
lpb = lpb.astype(np.float32)

config = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr)
aec = AEC(config=config)

frame_size = config.frame_size
hop_size   = config.hop_size

print(f"SR={sr} frame={frame_size} hop={hop_size} total_frames={len(mic)//hop_size}")
print()
print(f"{'t':>6} {'lpb_dB':>7} {'mic_dB':>7} {'out_dB':>7} {'ERLE':>6} "
      f"{'eff_dt':>7} {'linfail':>8} {'rb':>4} {'coh2':>6} "
      f"{'fs_hi':>6} {'cnt':>4} {'fa':>5} {'dte':>6}")
print("-" * 100)

out_buf = np.zeros_like(mic)
for i in range(0, len(mic) - hop_size + 1, hop_size):
    m_frame = mic[i:i+hop_size]
    r_frame = lpb[i:i+hop_size]
    o_frame = aec.process(m_frame, r_frame)
    out_buf[i:i+hop_size] = o_frame

    t = i / sr
    if 5.4 <= t <= 7.0:
        lpb_db = 20*np.log10(np.sqrt(np.mean(r_frame**2)) + 1e-10)
        mic_db = 20*np.log10(np.sqrt(np.mean(m_frame**2)) + 1e-10)
        out_db = 20*np.log10(np.sqrt(np.mean(o_frame**2)) + 1e-10)
        erle   = mic_db - out_db

        d = aec._diag
        eff_dt   = d.get('effective_dt', float('nan'))
        linfail  = d.get('linear_failed', '?')
        rb       = d.get('using_render_based', '?')
        fa       = d.get('far_activity', float('nan'))
        dte      = d.get('dt_from_energy', float('nan'))

        # Get ResFilter internals
        res = aec.res
        if res:
            coh2_mean = float(np.mean(getattr(res, '_last_coh2', np.array([float('nan')]))))
            hi_cnt    = getattr(res, '_fs_hi_erl_counter', -1)
            hi_state  = getattr(res, 'hard_override_active', '?')
        else:
            coh2_mean = float('nan')
            hi_cnt = -1
            hi_state = '?'

        print(f"{t:6.2f} {lpb_db:7.1f} {mic_db:7.1f} {out_db:7.1f} {erle:6.1f} "
              f"{eff_dt:7.3f} {str(linfail):>8} {str(rb):>4} {coh2_mean:6.3f} "
              f"{str(hi_state):>6} {hi_cnt:4d} {fa:5.3f} {dte:6.3f}")
