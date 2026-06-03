"""Generate the END-TO-END binary golden for the C aec_process cutover.

Constructs the REAL balanced AEC (AecConfig.from_preset('balanced', 16000, 832))
with np.random.seed(0) (CNG determinism), feeds RAW mic/lpb hops (NO pre-align —
the C runs its own delay alignment), and captures per-hop:
    [ mic_hop[hop], ref_hop[hop], expected_out[hop] ]  (raw LE float32)

The C parity_aec_e2e.c constructs Aec via aec_create(balanced) and replays per
hop asserting out[hop] BIT-EXACT.

Run: python3 python/diag/gen_aec_e2e_golden.py /tmp/aec_e2e_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import AecConfig                          # noqa: E402
from modules.orchestrator import AEC                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASE = os.path.join(WAV, 'doubletalk', '0I0XMl3M0ECO0U1N0cJvpg_doubletalk')

HOP = 160
SR = 16000
FL = 832
MAX_HOPS = 4200


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/aec_e2e_golden.bin'
    preset = sys.argv[2] if len(sys.argv) > 2 else 'balanced'

    cfg = AecConfig.from_preset(preset, sample_rate=SR, filter_length=FL)
    np.random.seed(0)
    aec = AEC(cfg)

    mic, _ = sf.read(CASE + '_mic.wav', dtype='float32')
    lpb, _ = sf.read(CASE + '_lpb.wav', dtype='float32')
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    ref = lpb[:n].astype(np.float32)   # RAW reference — no pre-alignment.

    rows = []
    for i in range(0, n - HOP, HOP):
        m = mic[i:i + HOP].copy()
        r = ref[i:i + HOP].copy()
        o = aec.process(m.copy(), r.copy())
        raw = np.asarray(aec._last_raw_output, np.float32).copy()
        rows.append((m, r, np.asarray(o, np.float32).copy(), raw))
        if len(rows) >= MAX_HOPS:
            break

    n_hops = len(rows)
    with open(out, 'wb') as f:
        np.array([HOP, SR, n_hops], dtype=np.int32).tofile(f)
        for m, r, o, raw in rows:
            m.astype(np.float32).tofile(f)
            r.astype(np.float32).tofile(f)
            o.astype(np.float32).tofile(f)
            raw.astype(np.float32).tofile(f)
    print('wrote %s  (%d hops, hop=%d)' % (out, n_hops, HOP))


if __name__ == '__main__':
    main()
