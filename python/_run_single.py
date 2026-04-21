"""One-off: process single mic/lpb pair with preset=balanced. Args: mic lpb out"""
import sys, os
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_aec_challenge import run_ours

mic_p, lpb_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
mic, sr = sf.read(mic_p, dtype='float32')
lpb, _  = sf.read(lpb_p, dtype='float32')
out = run_ours(mic, lpb, sr, fl=512, preset='balanced',
               is_movement=False, enable_res=True)
sf.write(out_p, out, sr)
print(f"wrote {out_p} ({len(out)/sr:.2f}s)")
