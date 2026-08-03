"""Generate the END-TO-END binary golden for the C aec_process cutover.

Constructs the REAL balanced AEC (AecConfig.from_preset('balanced', sr, FL))
with np.random.seed(0) (CNG determinism), feeds RAW mic/lpb hops (NO pre-align —
the C runs its own delay alignment), and captures per-hop:
    [ mic_hop[hop], ref_hop[hop], expected_out[hop] ]  (raw LE float32)

The C parity_aec_e2e.c constructs Aec via aec_create(balanced) and replays per
hop asserting out[hop] matches within the documented float32-FFT tolerance
(TOL_E2E in parity_aec_e2e.c). Reads sr/hop straight back out of the golden's
own header, so it needs no changes to run this at a new rate.

Source material: the SAME 16 kHz doubletalk case at every rate.
--sr 16000 (default) reads wav/aec_challenge_blind/doubletalk/..._{mic,lpb}.wav
directly. --sr 8000 / --sr 48000 (M5, multi-rate campaign, review F01) resample
that same case with scipy.signal.resample_poly (1:2 down to 8k, 3:1 up to 48k)
-- the identical resampling this repo's gen_aec_wav_pairs.py-equivalent scratch
script (m5/resample_pairs.py) uses for the other per-rate checks, so the same
underlying audio drives every rate's golden. FL (filter_length) defaults to
None, which lets AecConfig.__post_init__ derive it from sr via the real ms
policy (52ms <44.1kHz / 64ms >=44.1kHz) -- the same policy aec_config_defaults
mirrors in C (aec.c).

Run:
    python3 python/diag/gen_aec_e2e_golden.py /tmp/aec_e2e_golden.bin balanced --sr 16000
    python3 python/diag/gen_aec_e2e_golden.py /tmp/aec_e2e_golden_8k.bin balanced --sr 8000
    python3 python/diag/gen_aec_e2e_golden.py /tmp/aec_e2e_golden_48k.bin balanced --sr 48000
"""
import argparse
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
NATIVE_SR = 16000

MAX_HOPS = 4200

# M5: 1:2 -> 8 kHz, 3:1 -> 48 kHz. Matches m5/resample_pairs.py exactly (same
# up/down factors), so the 8k/48k goldens are driven by the identical
# resampled audio the other M5 per-rate checks (test_static_aec 3-rate loop,
# UBSan sweep, Python<->C waveform check) use.
_RESAMPLE_UP_DOWN = {8000: (1, 2), 48000: (3, 1)}


def _load_case(sr):
    mic, native_sr = sf.read(CASE + '_mic.wav', dtype='float32')
    lpb, native_sr2 = sf.read(CASE + '_lpb.wav', dtype='float32')
    assert native_sr == native_sr2 == NATIVE_SR, (native_sr, native_sr2)
    if sr == NATIVE_SR:
        return mic, lpb
    if sr not in _RESAMPLE_UP_DOWN:
        raise ValueError('no resampling policy for sr=%d (only %s + native %d)'
                          % (sr, sorted(_RESAMPLE_UP_DOWN), NATIVE_SR))
    from scipy.signal import resample_poly
    up, down = _RESAMPLE_UP_DOWN[sr]
    mic_r = resample_poly(mic, up, down).astype(np.float32)
    lpb_r = resample_poly(lpb, up, down).astype(np.float32)
    return mic_r, lpb_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='/tmp/aec_e2e_golden.bin')
    ap.add_argument('preset', nargs='?', default='balanced')
    ap.add_argument('--sr', type=int, default=16000,
                     help='sample rate (8000 / 16000 / 48000); default 16000')
    args = ap.parse_args()

    sr = args.sr
    cfg = AecConfig.from_preset(args.preset, sample_rate=sr)
    hop = cfg.hop_size   # was hardcoded "sr*10//1000" (10ms) -- diverged from
                          # AecConfig's real hop once 16kHz's default grid
                          # moved to 256/128 (8ms), crashing aec.process() on
                          # a mismatched hop size.
    np.random.seed(0)
    aec = AEC(cfg)

    mic, ref = _load_case(sr)
    n = min(len(mic), len(ref))
    mic = mic[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)   # RAW reference — no pre-alignment.

    rows = []
    for i in range(0, n - hop, hop):
        m = mic[i:i + hop].copy()
        r = ref[i:i + hop].copy()
        o = aec.process(m.copy(), r.copy())
        raw = np.asarray(aec._last_raw_output, np.float32).copy()
        rows.append((m, r, np.asarray(o, np.float32).copy(), raw))
        if len(rows) >= MAX_HOPS:
            break

    n_hops = len(rows)
    with open(args.out, 'wb') as f:
        np.array([hop, sr, n_hops], dtype=np.int32).tofile(f)
        for m, r, o, raw in rows:
            m.astype(np.float32).tofile(f)
            r.astype(np.float32).tofile(f)
            o.astype(np.float32).tofile(f)
            raw.astype(np.float32).tofile(f)
    print('wrote %s  (%d hops, hop=%d, sr=%d)' % (args.out, n_hops, hop, sr))


if __name__ == '__main__':
    main()
