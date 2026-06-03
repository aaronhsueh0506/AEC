"""Generate a binary golden for the C AEC3 matched-filter delay port.

This is the DELAY-ESTIMATOR golden for the v3.22 C port. It drives the REAL
production object -- LegacyDelayShim (the adapter the orchestrator uses) -- on
a doubletalk case WITH a known nonzero echo-path delay, capturing per hop the
exact near/far the shim consumes and the resulting public outputs.

WHY RAW (un-pre-aligned) signals:
  The orchestrator calls ``self.delay_est.accumulate(near_end, far_end)`` where
  near_end is the HPF'd mic and far_end is the raw reference BEFORE the ring
  buffer applies the delay compensation (orchestrator.py:1552, the ring write
  happens later at :1665+). On the bench, ``eval_aec_challenge`` PRE-aligns the
  reference (estimate_delay + shift) before feeding the pipeline, so the delay
  estimator sees ~0 delay and never crosses threshold -- a trivial all-(-1)
  trace. To exercise the full chain (COARSE -> REFINED, pre-echo aggregator,
  clockdrift, the consistent-estimate partial reset) the way a real nonzero
  delay would, we feed the shim the RAW unaligned mic/lpb directly. This is the
  identical LegacyDelayShim object the orchestrator instantiates; only the
  input alignment differs (the case carries a ~320-sample echo-path delay).

Per hop we capture:
  - input near[hop] (the mic hop) and far[hop] (the lpb hop)
  - estimated_delay (int), n_updates (int), is_solid (int bool),
    confidence (double 0.0/0.5/1.0) AFTER the accumulate.

Covers the whole case (~4186 hops). Output is raw little-endian binary.

Layout (LE):
  int32   hop, n_hops
  per hop:
    float32 near[hop]
    float32 far[hop]
    int32   estimated_delay
    int32   n_updates
    int32   is_solid
    float64 confidence

Run (from AEC repo root):  python3 python/diag/gen_delay_golden.py /tmp/delay_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.delay.legacy_compat import LegacyDelayShim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASE = os.path.join(WAV, 'doubletalk', '0I0XMl3M0ECO0U1N0cJvpg_doubletalk')

HOP = 160
SR = 16000


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/delay_golden.bin'
    # Optional 2nd arg: alternate case stem (absolute or relative to wav root)
    # for robustness checks. Defaults to the canonical DT case.
    case = CASE
    if len(sys.argv) > 2:
        case = sys.argv[2]
        if not os.path.isabs(case):
            case = os.path.join(WAV, case)

    mic, _ = sf.read(case + '_mic.wav')
    lpb, _ = sf.read(case + '_lpb.wav')
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    n = min(len(mic), len(lpb))
    mic = mic[:n]
    lpb = lpb[:n]

    # The production shim, exactly as the orchestrator constructs it (sr=16k,
    # hop=160; the legacy kwargs are no-op compat). Feed RAW unaligned signals
    # so the matched filter sees the case's true echo-path delay.
    shim = LegacyDelayShim(sample_rate=SR, hop_size=HOP)

    rows = []
    for i in range(0, n - HOP, HOP):
        near = mic[i:i + HOP]
        far = lpb[i:i + HOP]
        shim.accumulate(near, far)
        rows.append({
            'near': np.asarray(near, np.float32).copy(),
            'far': np.asarray(far, np.float32).copy(),
            'estimated_delay': int(shim.estimated_delay),
            'n_updates': int(shim._n_updates),
            'is_solid': int(bool(shim.is_solid)),
            'confidence': float(shim.confidence),
        })

    n_hops = len(rows)
    delays = np.array([r['estimated_delay'] for r in rows])
    confs = np.array([r['confidence'] for r in rows])
    print(f"captured {n_hops} hops")
    print(f"  estimated_delay: min={int(delays.min())} max={int(delays.max())} "
          f"distinct={sorted(set(int(x) for x in delays))[:8]}...")
    print(f"  confidence dist: 0.0={int((confs == 0.0).sum())} "
          f"0.5={int((confs == 0.5).sum())} 1.0={int((confs == 1.0).sum())}")
    print(f"  solid hops={int(sum(r['is_solid'] for r in rows))}  "
          f"max n_updates={int(delays.size and max(r['n_updates'] for r in rows))}")

    with open(out, 'wb') as f:
        np.array([HOP, n_hops], dtype=np.int32).tofile(f)
        for r in rows:
            r['near'].tofile(f)
            r['far'].tofile(f)
            np.array([r['estimated_delay'], r['n_updates'], r['is_solid']],
                     dtype=np.int32).tofile(f)
            np.array([r['confidence']], dtype=np.float64).tofile(f)

    print(f"wrote {out}  ({n_hops} hops, hop={HOP})")


if __name__ == '__main__':
    main()
