#!/usr/bin/env python3
"""E4.S6b — aggression sweep for listen verification.

S6a verdict was NO AUDIBLE CHANGE with g_min_e4_db=-12 dB. Hypothesis
H1: aggression too low (mask shape correct). Test by sweeping g_min
at -18 / -24 / -30 dB on the same 3 NL cases and re-listening.

If H1 holds → some g_min becomes audibly meaningful → ship that value.
If H1 fails → shape problem; move to mask-shape redesign (S6c).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))
sys.path.insert(0, _HERE)
from e4_s6_isolated_bench import load_case, render_with_detector, linear_erle
from e4_s6_suppressor import apply_suppressor


CASES = [
    ('Gsy0lC5QSUi540hiax9XtA_farend_singletalk', 'farend_singletalk'),
    ('9xjhiFbGo06hdQIsHTS6qA_farend_singletalk', 'farend_singletalk'),
    ('IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',
     'farend_singletalk'),
]
G_MIN_SWEEP = [-18.0, -24.0, -30.0]
DATASET = os.path.join(_REPO, 'wav', 'aec_challenge_blind')
LISTEN_DIR = os.path.join(_REPO, 'listen', 'v3_13_e4_s6_listen')


def main() -> int:
    os.makedirs(LISTEN_DIR, exist_ok=True)
    print(f'[s6b] sweeping g_min = {G_MIN_SWEEP} dB on {len(CASES)} cases')
    summary = []
    for stem, sub in CASES:
        print(f'[s6b] render {stem}')
        mic, lpb, sr = load_case(stem, sub, DATASET)
        out_baseline, trace, n = render_with_detector(
            mic, lpb, sr, e4_enabled=True)
        erle_base = linear_erle(mic, out_baseline, sr)
        fires = sum(1 for t in trace if t['nl_confidence'] > 0)
        n_active = len(trace)
        print(f'        baseline ERLE {erle_base:.2f} dB; '
              f'fires {fires}/{n_active} = {100*fires/n_active:.1f}%')
        for g_min in G_MIN_SWEEP:
            suppressed = apply_suppressor(
                out_baseline, trace, sample_rate=sr,
                g_min_e4_db=g_min,
            )
            erle_supp = linear_erle(mic, suppressed, sr)
            tag = f'g{int(-g_min):02d}'
            out_path = os.path.join(
                LISTEN_DIR, f'{stem}_ours_supp_{tag}.wav')
            sf.write(out_path, suppressed.astype(np.float32), sr)
            summary.append({
                'stem': stem, 'g_min_db': g_min,
                'erle_base_db': erle_base,
                'erle_supp_db': erle_supp,
                'delta_db': erle_supp - erle_base,
                'fires_pct': 100.0 * fires / max(1, n_active),
                'out_path': out_path,
            })
            print(f'        g_min={g_min:>5.1f} dB  ERLE {erle_supp:.2f} '
                  f'(Δ {erle_supp - erle_base:+.2f}) → {Path(out_path).name}')
    print()
    print('## S6b sweep summary')
    print(f'{"stem":<55s} {"g_min":>7s} {"ΔERLE":>8s}')
    for r in summary:
        print(f'{r["stem"][:55]:<55s} {r["g_min_db"]:>5.0f}dB '
              f'{r["delta_db"]:>+6.2f} dB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
