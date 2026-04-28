"""v3.1.0-rc1 Phase 1 evaluation: 800-case AECMOS vs v2.8.1 baseline.

Phase 1 changes (in current aec.py):
  1. CONV_ERLE_DB: 5.0 → 3.0
  3. EPC tick_hangover unconditional + ERL cap decay-back (30 frames)
  6. ERL outlier guard (inst_erl < 1.5)

Baseline numbers from v2.8.1 / v3.0.2 (bit-exact).
"""
import os, sys, time
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

TALK_TYPE = {'farend_singletalk': 'st', 'nearend_singletalk': 'nst', 'doubletalk': 'dt'}

BASELINE = {
    'farend_singletalk':           {'echo': 3.461, 'deg': 4.999},
    'farend_singletalk_static':    {'echo': 3.304, 'deg': 5.000},
    'farend_singletalk_movement':  {'echo': 3.664, 'deg': 4.999},
    'doubletalk':                  {'echo': 4.010, 'deg': 2.573},
    'doubletalk_static':           {'echo': 3.980, 'deg': 2.653},
    'doubletalk_movement':         {'echo': 4.059, 'deg': 2.441},
    'nearend_singletalk':          {'echo': 4.998, 'deg': 4.016},
}


def _all_cases():
    out = []
    for sub in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        d = WAV_BASE / sub
        for f in sorted(d.iterdir()):
            if f.name.endswith('_mic.wav'):
                stem = f.name[:-len('_mic.wav')]
                out.append({'subdir': sub, 'stem': stem,
                            'mic': str(d / f.name),
                            'lpb': str(d / f'{stem}_lpb.wav'),
                            'talk': TALK_TYPE[sub],
                            'movement': '_with_movement' in stem})
    return out


_aecmos = None


def _init_worker(model_path):
    global _aecmos
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _run_one(case):
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
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32); pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    out = out[:pos]
    enh = np.concatenate([out, np.zeros(max(0, len(mic) - len(out)), dtype=np.float32)])
    e, d = _aecmos.run(case['talk'], lpb, mic, enh)
    return {'subdir': case['subdir'], 'stem': case['stem'],
            'movement': case['movement'], 'echo': float(e), 'deg': float(d)}


def _aggregate(rows):
    buckets = {}
    for r in rows:
        sc = r['subdir']
        for tag in (sc, f'{sc}_movement' if r['movement'] else f'{sc}_static'):
            d = buckets.setdefault(tag, {'echo': [], 'deg': []})
            d['echo'].append(r['echo']); d['deg'].append(r['deg'])
    return {k: {'n': len(d['echo']), 'echo': float(np.mean(d['echo'])), 'deg': float(np.mean(d['deg']))}
            for k, d in buckets.items()}


def main():
    cases = _all_cases()
    print(f'cases: {len(cases)}')
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

    agg = _aggregate(rows)
    order = ['farend_singletalk', 'farend_singletalk_static', 'farend_singletalk_movement',
             'doubletalk', 'doubletalk_static', 'doubletalk_movement', 'nearend_singletalk']
    print(f'\n=== v3.1.0-rc1 Phase 1 vs v2.8.1 baseline ===')
    print(f'{"bucket":<30s} {"n":>4s} {"echo":>7s} {"deg":>7s} {"Δecho":>8s} {"Δdeg":>8s}  flag')
    for k in order:
        if k not in agg: continue
        b = BASELINE[k]; f = agg[k]
        de = f['echo'] - b['echo']; dd = f['deg'] - b['deg']
        flag = ''
        if k == 'doubletalk_movement':
            if de >= 0.05 and dd >= -0.02: flag = '  ✓ TARGET'
            elif de >= 0.02: flag = '  marginal+'
            elif de <= -0.02: flag = '  REGRESS'
        elif k == 'doubletalk_static':
            if de >= 0.05: flag = '  ✓ improved'
            elif de <= -0.02: flag = '  REGRESS'
        elif k == 'farend_singletalk' and de < -0.02: flag = '  REGRESS'
        elif k == 'nearend_singletalk' and dd < -0.02: flag = '  REGRESS'
        print(f'{k:<30s} {f["n"]:>4d} {f["echo"]:>7.3f} {f["deg"]:>7.3f} {de:>+8.3f} {dd:>+8.3f}{flag}')


if __name__ == '__main__':
    main()
