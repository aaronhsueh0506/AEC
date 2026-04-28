"""Phase C2 ablation: EPC-in-DT behavior.

Variants test what happens when path-change triggers do NOT reset filter convergence
(per WebRTC AEC3: Q-boost without convergence reset, since reset cascades to RES
erle_factor / startup mu_min / shadow copy gate).

Variants:
    E0 baseline      All triggers reset convergence (v2.8.1).
    E1 no_delay      Skip mark_diverged for delay-shift triggers only.
    E3 no_reset_all  Skip mark_diverged for all sources (Q-boost without reset).

Success bar (per plan):
    movement-DT echo Δ ≥ +0.03  AND
    FS echo regression < 0.02   AND
    DT deg regression  < 0.02

Usage:
    python3 diag_epc_dt_ablation.py [-j 4] [-n 50] [--variant E0,E1,E3]
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
    'E0_baseline':     {'no_reset': frozenset()},
    'E1_no_delay':     {'no_reset': frozenset({'delay_first', 'delay_shift'})},
    'E3_no_reset_all': {'no_reset': frozenset({'delay_first', 'delay_shift', 'epv', 'shadow_rise'})},
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


_aecmos = None


def _init_worker(model_path):
    global _aecmos
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _run_one(case, no_reset):
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
    aec._epc_no_reset_sources = no_reset

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
    case, no_reset = args
    return _run_one(case, no_reset)


def _aggregate(rows):
    buckets = {}
    for r in rows:
        scenario = r['subdir']
        for tag in (scenario, f'{scenario}_movement' if r['movement'] else f'{scenario}_static'):
            d = buckets.setdefault(tag, {'echo': [], 'deg': []})
            d['echo'].append(r['echo'])
            d['deg'].append(r['deg'])
    return {k: {'n': len(d['echo']), 'echo': float(np.mean(d['echo'])), 'deg': float(np.mean(d['deg']))}
            for k, d in buckets.items()}


def _print_table(name, agg):
    print(f'\n=== {name} ===')
    order = [
        'farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
        'doubletalk', 'doubletalk_static', 'doubletalk_movement',
        'nearend_singletalk',
    ]
    print(f'{"bucket":<30s} {"n":>4s} {"echo":>7s} {"deg":>7s}')
    for k in order:
        if k not in agg: continue
        d = agg[k]
        print(f'{k:<30s} {d["n"]:>4d} {d["echo"]:>7.3f} {d["deg"]:>7.3f}')


def _print_delta(base_label, base, agg_dict):
    order = [
        'farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
        'doubletalk', 'doubletalk_static', 'doubletalk_movement',
        'nearend_singletalk',
    ]
    for v_name, agg in agg_dict.items():
        if v_name == base_label: continue
        print(f'\n=== Δ ({v_name} − {base_label}) ===')
        print(f'{"bucket":<30s} {"n":>4s} {"Δecho":>8s} {"Δdeg":>8s}  flag')
        for k in order:
            if k not in base or k not in agg: continue
            de = agg[k]['echo'] - base[k]['echo']
            dd = agg[k]['deg']  - base[k]['deg']
            flag = ''
            if k == 'doubletalk_movement':
                if de >= 0.03 and dd >= -0.02:
                    flag = '  ✓ TARGET'
                elif de >= 0.01:
                    flag = '  marginal'
                else:
                    flag = '  below'
            elif k == 'farend_singletalk' and de < -0.02:
                flag = '  REGRESSION'
            elif k == 'nearend_singletalk' and dd < -0.02:
                flag = '  REGRESSION'
            print(f'{k:<30s} {base[k]["n"]:>4d} {de:>+8.3f} {dd:>+8.3f}{flag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=4)
    ap.add_argument('-n', type=int, default=0)
    ap.add_argument('--variant', default='E0_baseline,E1_no_delay,E3_no_reset_all')
    args = ap.parse_args()

    n_jobs = max(1, min(args.j, 4))
    cases = _all_cases()
    if args.n > 0:
        cases = cases[:args.n]
    print(f'cases: {len(cases)}')

    variants = [v.strip() for v in args.variant.split(',')]
    aggregates = {}

    for v in variants:
        if v not in VARIANTS:
            print(f'unknown variant: {v}'); return 2
        no_reset = VARIANTS[v]['no_reset']
        t0 = time.time()
        print(f'\n--- variant {v} (no_reset={sorted(no_reset) or "[]"}) ---')
        rows = []
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_init_worker,
                                 initargs=(str(MODEL),)) as ex:
            futs = [ex.submit(_job, (c, no_reset)) for c in cases]
            done = 0
            for fu in as_completed(futs):
                rows.append(fu.result())
                done += 1
                if done % 100 == 0:
                    print(f'  [{done}/{len(cases)}] processed', flush=True)
        agg = _aggregate(rows)
        aggregates[v] = agg
        _print_table(v, agg)
        print(f'  elapsed: {time.time() - t0:.1f}s')

    if 'E0_baseline' in aggregates:
        _print_delta('E0_baseline', aggregates['E0_baseline'], aggregates)
    return 0


if __name__ == '__main__':
    sys.exit(main())
