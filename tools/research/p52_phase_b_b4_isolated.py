"""P52 B.4 supplementary — per-module isolated byte-equal test.

For each subset of overrides M1..M{k}, construct a dynamic ResFilter
subclass with only those overrides active, then run the standard B.3
100-case sample and diff against legacy.

Usage:
  python tools/research/p52_phase_b_b4_isolated.py --up-to {2,3,4,5} --out json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import random

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
sys.path.insert(0, str(_REPO / 'python'))

import aec  # noqa: E402
from aec import AEC, AecConfig, AecMode, ResFilter  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

from res_refactored.residual_estimator import residual_estimator  # noqa: E402
from res_refactored.gain_computer import gain_computer  # noqa: E402
from res_refactored.spectral_shaper import spectral_shaper  # noqa: E402
from res_refactored.temporal_smoother import temporal_smoother  # noqa: E402
from res_refactored.noise_floor_cng import noise_floor_cng  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SAMPLE_SIZE = 100
SAMPLE_SEED = 42
CNG_SEED = 42


def discover():
    out = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir(): continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                out.append((stem, subset, str(mic), str(lpb)))
    return out


def sample_cases():
    return random.Random(SAMPLE_SEED).sample(discover(), SAMPLE_SIZE)


def _build_class(up_to: int):
    """Build a ResFilter subclass overriding stages 1..up_to."""
    overrides = {}
    if up_to >= 1:
        overrides['_stage_residual_model'] = lambda self, **kw: residual_estimator(self, **kw)
    if up_to >= 2:
        overrides['_stage_gain_compute'] = lambda self, **kw: gain_computer(self, **kw)
    if up_to >= 3:
        overrides['_stage_gain_postprocess'] = lambda self, **kw: spectral_shaper(self, **kw)
    if up_to >= 4:
        overrides['_stage_temporal_smoothing'] = lambda self, **kw: temporal_smoother(self, **kw)
    if up_to >= 5:
        overrides['_stage_noise_floor_and_cng'] = lambda self, **kw: noise_floor_cng(self, **kw)
    return type(f'ResFilter_UpTo{up_to}', (ResFilter,), overrides)


def _run_one(args):
    stem, subset, mic_p, lpb_p, up_to = args
    if up_to > 0:
        aec.ResFilter = _build_class(up_to)
    np.random.seed(CNG_SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=sr, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True,
    )
    a = AEC(cfg)
    hop = a.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = a.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    return stem, out[:pos]


def snapshot(up_to: int, jobs: int = 4):
    cases = sample_cases()
    print(f'[up_to=M{up_to}] {len(cases)} cases j={jobs}', flush=True)
    t0 = time.time()
    results = {}
    jobs_args = [(s, sub, m, l, up_to) for (s, sub, m, l) in cases]
    with Pool(jobs) as p:
        for i, (stem, out) in enumerate(p.imap_unordered(_run_one, jobs_args)):
            results[stem] = out
            if (i + 1) % 25 == 0 or (i + 1) == len(cases):
                print(f'  {i+1}/{len(cases)} ({time.time()-t0:.0f}s)', flush=True)
    return results


def diff(pre: dict, post: dict):
    common = sorted(set(pre) & set(post))
    total = exact = close = ident = 0
    max_d = 0.0
    for s in common:
        a = pre[s].astype(np.float32); b = post[s].astype(np.float32)
        m = min(len(a), len(b))
        a = a[:m]; b = b[:m]
        d = np.abs(a - b)
        total += m
        exact += int(np.sum(a == b))
        close += int(np.sum(np.isclose(a, b, atol=1e-6, rtol=1e-5)))
        if np.array_equal(a, b): ident += 1
        max_d = max(max_d, float(d.max()))
    return {
        'cases_total': len(common), 'cases_byte_identical': ident,
        'total_samples': total, 'fraction_exact': exact / max(1, total),
        'fraction_close': close / max(1, total), 'max_abs_delta': max_d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('-j', type=int, default=4)
    args = ap.parse_args()

    # Baseline (no overrides — pure legacy)
    print('=== Snapshot: legacy (no overrides) ===')
    legacy = snapshot(up_to=0, jobs=args.j)

    results = {}
    for k in (2, 3, 4, 5):
        print(f'\n=== Snapshot: M1..M{k} ===')
        snap = snapshot(up_to=k, jobs=args.j)
        d = diff(legacy, snap)
        d['up_to_module'] = k
        results[f'M1_through_M{k}'] = d
        print(f'  result: cases_byte_identical={d["cases_byte_identical"]}/{d["cases_total"]}, '
              f'max_delta={d["max_abs_delta"]:.3e}')

    print('\n=== Summary ===')
    print(json.dumps(results, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
