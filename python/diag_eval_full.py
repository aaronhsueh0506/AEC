"""v3.1.0 vs AEC2 full blind test AECMOS comparison (800 cases, CNG=True).

Scores all 800 cases for both ours (current aec.py = v3.1.0, balanced preset
with default CNG=True) and AEC2 (pre-computed outputs in python/output_ref/).
Per-bucket comparison + per-case JSON for any drill-down.

Note: previous evaluations used enable_cng=False for parity-determinism.
This run uses production defaults (CNG=True per BALANCED preset).
"""
import os, sys, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
AEC2_DIR = REPO / 'python/output_ref'
MODEL = REPO / 'model/Run_1663915512_Stage_0.onnx'
sys.path.insert(0, str(REPO / 'model'))
OUT = Path(__file__).parent / 'output_eval'

TALK_TYPE = {'farend_singletalk': 'st', 'nearend_singletalk': 'nst', 'doubletalk': 'dt'}


def _all_cases():
    out = []
    for sub in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        d = WAV_BASE / sub
        for f in sorted(d.iterdir()):
            if f.name.endswith('_mic.wav'):
                stem = f.name[:-len('_mic.wav')]
                aec2 = AEC2_DIR / f'{stem}_aec2.wav'
                if not aec2.is_file():
                    continue
                out.append({'subdir': sub, 'stem': stem,
                            'mic': str(d / f.name),
                            'lpb': str(d / f'{stem}_lpb.wav'),
                            'aec2': str(aec2),
                            'talk': TALK_TYPE[sub],
                            'movement': '_with_movement' in stem})
    return out


_aecmos = None


def _init_worker(model_path):
    global _aecmos
    np.random.seed(20260428)  # CNG uses np.random; seed for repeatability across runs
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _run_one(case):
    np.random.seed(case['idx'] * 7919)  # per-case deterministic seed
    mic, sr = sf.read(case['mic'], dtype='float32')
    lpb, _  = sf.read(case['lpb'], dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]
    is_mv = case['movement']
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    # CNG default per balanced preset = True
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, **delay_kw)
    aec = AEC(cfg)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32); pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    out = out[:pos]
    enh = np.concatenate([out, np.zeros(max(0, len(mic) - len(out)), dtype=np.float32)])

    aec2_sig, _ = sf.read(case['aec2'], dtype='float32')
    if aec2_sig.ndim > 1: aec2_sig = aec2_sig[:, 0]
    aec2_pad_n = min(len(mic), len(aec2_sig))
    aec2_pad = np.concatenate([
        aec2_sig[:aec2_pad_n],
        np.zeros(max(0, len(mic) - aec2_pad_n), dtype=np.float32),
    ])

    ours_e, ours_d = _aecmos.run(case['talk'], lpb, mic, enh)
    aec2_e, aec2_d = _aecmos.run(case['talk'], lpb, mic, aec2_pad)
    return {'subdir': case['subdir'], 'stem': case['stem'],
            'movement': case['movement'],
            'ours_echo': float(ours_e), 'ours_deg': float(ours_d),
            'aec2_echo': float(aec2_e), 'aec2_deg': float(aec2_d)}


def _aggregate(rows):
    buckets = {}
    for r in rows:
        sc = r['subdir']
        for tag in (sc, f'{sc}_movement' if r['movement'] else f'{sc}_static'):
            d = buckets.setdefault(tag, {'oe': [], 'od': [], 'ae': [], 'ad': []})
            d['oe'].append(r['ours_echo']); d['od'].append(r['ours_deg'])
            d['ae'].append(r['aec2_echo']); d['ad'].append(r['aec2_deg'])
    return {k: {'n': len(d['oe']),
                'ours_echo': float(np.mean(d['oe'])), 'ours_deg': float(np.mean(d['od'])),
                'aec2_echo': float(np.mean(d['ae'])), 'aec2_deg': float(np.mean(d['ad']))}
            for k, d in buckets.items()}


def main():
    OUT.mkdir(exist_ok=True)
    cases = _all_cases()
    for i, c in enumerate(cases): c['idx'] = i
    print(f'cases: {len(cases)}, CNG=True (balanced default), v3.1.0')
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init_worker,
                             initargs=(str(MODEL),)) as ex:
        futs = [ex.submit(_run_one, c) for c in cases]
        done = 0
        for fu in as_completed(futs):
            rows.append(fu.result()); done += 1
            if done % 100 == 0: print(f'  [{done}/{len(cases)}]', flush=True)
    print(f'elapsed: {time.time()-t0:.1f}s')

    Path(OUT / 'per_case.json').write_text(json.dumps(rows, indent=2))
    agg = _aggregate(rows)

    order = ['farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
             'doubletalk', 'doubletalk_static', 'doubletalk_movement',
             'nearend_singletalk']

    print(f'\n{"="*100}')
    print(f'  v3.1.0 (CNG=True) vs AEC2  —  AEC Challenge blind test')
    print(f'{"="*100}')
    print(f'{"bucket":<30s} {"n":>4s} {"ours_e":>7s} {"aec2_e":>7s} {"Δecho":>7s}'
          f'  {"ours_d":>7s} {"aec2_d":>7s} {"Δdeg":>7s}')
    for k in order:
        if k not in agg: continue
        a = agg[k]
        print(f'{k:<30s} {a["n"]:>4d} {a["ours_echo"]:>7.3f} {a["aec2_echo"]:>7.3f}'
              f' {a["ours_echo"]-a["aec2_echo"]:>+7.3f}'
              f'  {a["ours_deg"]:>7.3f} {a["aec2_deg"]:>7.3f}'
              f' {a["ours_deg"]-a["aec2_deg"]:>+7.3f}')

    # Win rate
    print(f'\n=== win rate (ours vs aec2, n={len(rows)}) ===')
    n = len(rows)
    we = sum(1 for r in rows if r['ours_echo'] > r['aec2_echo'])
    wd = sum(1 for r in rows if r['ours_deg']  > r['aec2_deg'])
    we_dt = sum(1 for r in rows if r['subdir']=='doubletalk' and r['ours_echo'] > r['aec2_echo'])
    wd_dt = sum(1 for r in rows if r['subdir']=='doubletalk' and r['ours_deg']  > r['aec2_deg'])
    n_dt = sum(1 for r in rows if r['subdir']=='doubletalk')
    print(f'  Echo win  overall: {we}/{n} ({100*we/n:.0f}%)   DT only: {we_dt}/{n_dt} ({100*we_dt/n_dt:.0f}%)')
    print(f'  Deg  win  overall: {wd}/{n} ({100*wd/n:.0f}%)   DT only: {wd_dt}/{n_dt} ({100*wd_dt/n_dt:.0f}%)')


if __name__ == '__main__':
    main()
