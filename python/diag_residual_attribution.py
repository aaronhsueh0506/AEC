"""Phase B2 ablation: R0 (legacy) vs R1 (split) two-path R2 model.

Runs AEC on all 800 AEC Challenge blind cases under two ResidualEchoEstimator
modes, scores with local AECMOS, and prints a per-scenario delta table.

Success bar (per plan):
  R1 — movement-DT echo Δ ≥ +0.02 AND
       FS echo regression < 0.01 AND
       NE deg regression < 0.01

Usage:
  python3 diag_residual_attribution.py [-j 4] [-n 50] [--variant R0,R1]
"""
import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
MODEL = REPO / 'model/Run_1663915512_Stage_0.onnx'
sys.path.insert(0, str(REPO / 'model'))

VARIANTS = {
    'R0_legacy': {'mode': 'legacy'},
    'R1_split':  {'mode': 'split'},
}

TALK_TYPE = {
    'farend_singletalk':  'st',
    'nearend_singletalk': 'nst',
    'doubletalk':         'dt',
}


def _all_cases():
    out = []
    for sub in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        d = WAV_BASE / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.name.endswith('_mic.wav'):
                stem = f.name[:-len('_mic.wav')]
                out.append({
                    'subdir': sub,
                    'stem': stem,
                    'mic': str(d / f.name),
                    'lpb': str(d / f'{stem}_lpb.wav'),
                    'talk': TALK_TYPE[sub],
                    'movement': '_with_movement' in stem,
                })
    return out


_aecmos = None  # per-worker singleton


def _init_worker(model_path):
    global _aecmos
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _run_one(case, mode):
    mic, sr = sf.read(case['mic'], dtype='float32')
    lpb, _  = sf.read(case['lpb'], dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    is_mv = case['movement']
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, enable_cng=False, **delay_kw)
    aec = AEC(cfg)
    aec.res._residual_est.mode = mode

    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    out = out[:pos]

    pad = max(0, len(mic) - len(out))
    enh = np.concatenate([out, np.zeros(pad, dtype=np.float32)])

    echo_mos, deg_mos = _aecmos.run(case['talk'], lpb, mic, enh)
    return {
        'subdir': case['subdir'],
        'stem': case['stem'],
        'movement': case['movement'],
        'echo': float(echo_mos),
        'deg': float(deg_mos),
    }


def _job(args):
    case, mode = args
    return _run_one(case, mode)


def _aggregate(rows):
    """Bucketed mean: scenario × movement."""
    buckets = {}
    for r in rows:
        scenario = r['subdir']
        for tag in (scenario, f'{scenario}_movement' if r['movement'] else f'{scenario}_static'):
            d = buckets.setdefault(tag, {'echo': [], 'deg': []})
            d['echo'].append(r['echo'])
            d['deg'].append(r['deg'])
    out = {}
    for k, d in buckets.items():
        out[k] = {
            'n':    len(d['echo']),
            'echo': float(np.mean(d['echo'])),
            'deg':  float(np.mean(d['deg'])),
        }
    return out


def _print_table(name, agg):
    print(f'\n=== {name} ===')
    order = [
        'farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
        'doubletalk', 'doubletalk_static', 'doubletalk_movement',
        'nearend_singletalk',
    ]
    print(f'{"bucket":<30s} {"n":>4s} {"echo":>7s} {"deg":>7s}')
    for k in order:
        if k not in agg:
            continue
        d = agg[k]
        print(f'{k:<30s} {d["n"]:>4d} {d["echo"]:>7.3f} {d["deg"]:>7.3f}')


def _print_delta(base, alt, base_name, alt_name):
    print(f'\n=== Δ ({alt_name} − {base_name}) ===')
    order = [
        'farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
        'doubletalk', 'doubletalk_static', 'doubletalk_movement',
        'nearend_singletalk',
    ]
    print(f'{"bucket":<30s} {"n":>4s} {"Δecho":>8s} {"Δdeg":>8s}')
    for k in order:
        if k not in base or k not in alt:
            continue
        de = alt[k]['echo'] - base[k]['echo']
        dd = alt[k]['deg']  - base[k]['deg']
        flag = ''
        if k == 'doubletalk_movement':
            if de >= 0.02:
                flag = '  ← target met'
            elif de >= 0.01:
                flag = '  ← marginal'
            else:
                flag = '  ← below target'
        print(f'{k:<30s} {base[k]["n"]:>4d} {de:>+8.3f} {dd:>+8.3f}{flag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=4, help='parallel workers (max 4)')
    ap.add_argument('-n', type=int, default=0, help='limit cases (0 = all 800)')
    ap.add_argument('--variant', default='R0_legacy,R1_split',
                    help='comma-separated variant labels')
    args = ap.parse_args()

    n_jobs = max(1, min(args.j, 4))

    cases = _all_cases()
    if args.n > 0:
        # take a stratified sample: half movement, half static; spread across scenarios
        cases = cases[:args.n]
    print(f'cases: {len(cases)}')

    variants = [v.strip() for v in args.variant.split(',')]
    aggregates = {}

    for v in variants:
        if v not in VARIANTS:
            print(f'unknown variant: {v}'); return 2
        mode = VARIANTS[v]['mode']
        t0 = time.time()
        print(f'\n--- variant {v} (mode={mode}) ---')
        rows = []
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_init_worker,
                                 initargs=(str(MODEL),)) as ex:
            futs = [ex.submit(_job, (c, mode)) for c in cases]
            done = 0
            for fu in as_completed(futs):
                rows.append(fu.result())
                done += 1
                if done % 50 == 0:
                    print(f'  [{done}/{len(cases)}] processed', flush=True)
        agg = _aggregate(rows)
        aggregates[v] = agg
        _print_table(v, agg)
        print(f'  elapsed: {time.time() - t0:.1f}s')

    if len(variants) >= 2:
        _print_delta(aggregates[variants[0]], aggregates[variants[1]],
                     variants[0], variants[1])
    return 0


if __name__ == '__main__':
    sys.exit(main())
