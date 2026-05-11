"""P52 A.0R.7 — 800-case AcousticRegimeClassifier distribution + handler fires.

Per case:
  1. Classify regime via AcousticRegimeClassifier (mic + lpb wav, no AEC).
  2. Run AEC with trace_p52_regime_handler=True; collect boost_q /
     reverse_copy / main_paused fire counts + first-fire frames.

Output: JSON + human-readable Markdown table.

Usage:
  python tools/research/p52_a0r7_regime_distribution.py \\
      --out /tmp/p52_a0r7/dist.json -j 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from aec_p52_regime_classifier import (  # noqa: E402
    AcousticRegime, AcousticRegimeClassifier,
)
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SEED = 42


def discover():
    cases = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir(): continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                cases.append((stem, subset, str(mic), str(lpb)))
    return cases


def _one(args):
    stem, subset, mic_p, lpb_p = args
    np.random.seed(SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    clf = AcousticRegimeClassifier()
    cls = clf.classify(mic, lpb, sample_rate=sr)

    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=sr, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True, trace_p52_regime_handler=True,
    )
    aec = AEC(cfg)
    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    rows = aec._regime_trace_rows
    boost_frames = [r['frame'] for r in rows if r['boost_q_fired']]
    rev_frames = [r['frame'] for r in rows if r['reverse_copy_fired']]
    paused_frames = [r['frame'] for r in rows if r['main_paused_fired']]
    return {
        'stem': stem,
        'subset': subset,
        'duration_s': n / sr,
        'regime': cls.regime.value,
        'erl_decile_std_db': cls.erl_decile_std_db,
        'erl_decile_ptp_db': cls.erl_decile_ptp_db,
        'deciles_used': cls.deciles_used,
        'frames_total': len(rows),
        'boost_q_count': len(boost_frames),
        'reverse_copy_count': len(rev_frames),
        'main_paused_frames': len(paused_frames),
        'first_boost_q_frame': boost_frames[0] if boost_frames else None,
        'first_reverse_copy_frame': rev_frames[0] if rev_frames else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cases = discover()
    if args.limit:
        cases = cases[:args.limit]
    print(f'[A.0R.7] {len(cases)} cases j={args.jobs}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    with Pool(args.jobs) as p:
        for i, r in enumerate(p.imap_unordered(_one, cases)):
            results.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(cases):
                dt = time.time() - t0
                print(f'  {i+1}/{len(cases)} ({dt:.0f}s)', flush=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[A.0R.7] wrote {args.out} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
