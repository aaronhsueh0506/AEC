"""DT-specific gap analysis: ours vs AEC2 across all 300 DT cases.

For each of static-DT (186) + movement-DT (114) cases:
  1. Run our v3.0.2 AEC, capture per-frame state trajectory.
  2. Score AECMOS for ours.
  3. Score AECMOS for pre-existing AEC2 baseline.
  4. Compute per-case stats:
       - convergence time (frame where filter_converged first becomes True)
       - EPC fire count + stuck %
       - ERL trajectory (final value, mean, hit-cap %)
       - main_paused %
       - render_based %
  5. Bucket by convergence time and movement flag.
  6. Per-bucket ours vs aec2 echo + deg comparison.

Output:
  python/output_dt_gap/per_case.json     (full per-case data)
  Console:
    overall stats
    bucket table (by convergence time × movement)
    worst-N where ours-vs-aec2 echo gap largest
"""
import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind/doubletalk'
AEC2_DIR = REPO / 'python/output_ref'
MODEL = REPO / 'model/Run_1663915512_Stage_0.onnx'
sys.path.insert(0, str(REPO / 'model'))
OUT = Path(__file__).parent / 'output_dt_gap'


def _all_dt_cases():
    out = []
    for f in sorted(WAV_BASE.iterdir()):
        if f.name.endswith('_mic.wav'):
            stem = f.name[:-len('_mic.wav')]
            aec2 = AEC2_DIR / f'{stem}_aec2.wav'
            if not aec2.is_file():
                continue
            out.append({
                'stem': stem,
                'mic': str(WAV_BASE / f.name),
                'lpb': str(WAV_BASE / f'{stem}_lpb.wav'),
                'aec2': str(aec2),
                'movement': '_with_movement' in stem,
            })
    return out


_aecmos = None


def _init_worker(model_path):
    global _aecmos
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _energy_db(x):
    return 10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


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

    # Per-frame trajectory tracking
    n_frames = (n // hop)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    f_idx = 0
    first_conv_frame = -1
    epc_fires = 0
    epc_active_frames = 0
    main_paused_frames = 0
    render_based_frames = 0
    erl_history = []
    prev_epc = False

    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        if first_conv_frame < 0 and aec._convergence.converged:
            first_conv_frame = f_idx
        cur_epc = aec.epc_active
        if cur_epc and not prev_epc:
            epc_fires += 1
        prev_epc = cur_epc
        if cur_epc:
            epc_active_frames += 1
        if aec._shadow_copy_ctrl.main_paused:
            main_paused_frames += 1
        if aec.res and aec.res._using_render_based:
            render_based_frames += 1
        # Sample ERL every 10 frames to limit memory
        if f_idx % 10 == 0:
            erl_history.append(float(aec._erl_estimate))
        pos += hop
        f_idx += 1

    out = out[:pos]
    enh = np.concatenate([out, np.zeros(max(0, len(mic) - len(out)), dtype=np.float32)])

    # Score ours
    ours_e, ours_d = _aecmos.run('dt', lpb, mic, enh)

    # Score aec2 baseline
    aec2_sig, _ = sf.read(case['aec2'], dtype='float32')
    if aec2_sig.ndim > 1: aec2_sig = aec2_sig[:, 0]
    aec2_pad_n = min(len(mic), len(aec2_sig))
    aec2_pad = np.concatenate([
        aec2_sig[:aec2_pad_n],
        np.zeros(max(0, len(mic) - aec2_pad_n), dtype=np.float32),
    ])
    aec2_e, aec2_d = _aecmos.run('dt', lpb, mic, aec2_pad)

    return {
        'stem': case['stem'],
        'movement': case['movement'],
        'ours_echo': float(ours_e),
        'ours_deg':  float(ours_d),
        'aec2_echo': float(aec2_e),
        'aec2_deg':  float(aec2_d),
        'd_echo':    float(aec2_e - ours_e),  # >0 means ours worse
        'd_deg':     float(aec2_d - ours_d),
        'first_conv_frame': first_conv_frame,
        'first_conv_s': float(first_conv_frame * hop / sr) if first_conv_frame >= 0 else -1.0,
        'never_converged': first_conv_frame < 0,
        'n_frames': n_frames,
        'epc_fires': epc_fires,
        'epc_pct': float(epc_active_frames / max(n_frames, 1)),
        'main_paused_pct': float(main_paused_frames / max(n_frames, 1)),
        'render_based_pct': float(render_based_frames / max(n_frames, 1)),
        'erl_final': erl_history[-1] if erl_history else 0.1,
        'erl_mean':  float(np.mean(erl_history)) if erl_history else 0.1,
        'erl_max':   float(np.max(erl_history)) if erl_history else 0.1,
        'ours_db':  _energy_db(out),
        'aec2_db':  _energy_db(aec2_pad[:len(out)]),
        'mic_db':   _energy_db(mic[:len(out)]),
    }


def _bucket_by_conv(rows):
    """Bucket by convergence-time × movement."""
    buckets = {}
    for r in rows:
        if r['never_converged']:
            ct = 'never'
        elif r['first_conv_s'] < 1.0:
            ct = '0-1s'
        elif r['first_conv_s'] < 5.0:
            ct = '1-5s'
        elif r['first_conv_s'] < 15.0:
            ct = '5-15s'
        else:
            ct = '15s+'
        mv = 'mv' if r['movement'] else 'st'
        key = f'{ct}_{mv}'
        buckets.setdefault(key, []).append(r)
    return buckets


def _print_overall(rows):
    n = len(rows)
    de = np.array([r['d_echo'] for r in rows])
    dd = np.array([r['d_deg']  for r in rows])
    print(f'\n=== overall (n={n}) ===')
    print(f'  Δecho:  mean {de.mean():+.3f}  median {np.median(de):+.3f}  std {de.std():.3f}'
          f'  range [{de.min():+.3f}, {de.max():+.3f}]')
    print(f'  Δdeg:   mean {dd.mean():+.3f}  median {np.median(dd):+.3f}  std {dd.std():.3f}'
          f'  range [{dd.min():+.3f}, {dd.max():+.3f}]')

    ours_wins_echo = (de < 0).sum()
    ours_wins_deg  = (dd < 0).sum()
    print(f'  ours wins echo: {ours_wins_echo}/{n} ({100*ours_wins_echo/n:.0f}%)')
    print(f'  ours wins deg:  {ours_wins_deg}/{n} ({100*ours_wins_deg/n:.0f}%)')


def _print_buckets(buckets):
    print(f'\n=== bucket: convergence-time × movement ===')
    print(f'{"bucket":<12s} {"n":>4s} {"ours_e":>7s} {"aec2_e":>7s} {"Δecho":>7s}'
          f'  {"ours_d":>7s} {"aec2_d":>7s} {"Δdeg":>7s}'
          f'  epc% main_p% render%')
    order = ['0-1s_st','0-1s_mv','1-5s_st','1-5s_mv','5-15s_st','5-15s_mv',
             '15s+_st','15s+_mv','never_st','never_mv']
    for k in order:
        if k not in buckets: continue
        rs = buckets[k]
        n = len(rs)
        oe = np.mean([r['ours_echo'] for r in rs])
        ae = np.mean([r['aec2_echo'] for r in rs])
        od = np.mean([r['ours_deg']  for r in rs])
        ad = np.mean([r['aec2_deg']  for r in rs])
        epc_p = np.mean([r['epc_pct']*100 for r in rs])
        mp_p  = np.mean([r['main_paused_pct']*100 for r in rs])
        rb_p  = np.mean([r['render_based_pct']*100 for r in rs])
        print(f'{k:<12s} {n:>4d} {oe:>7.3f} {ae:>7.3f} {ae-oe:>+7.3f}'
              f'  {od:>7.3f} {ad:>7.3f} {ad-od:>+7.3f}'
              f'  {epc_p:>4.0f} {mp_p:>5.0f}  {rb_p:>5.0f}')


def _print_worst(rows, n_worst=15):
    rs = sorted(rows, key=lambda r: r['d_echo'], reverse=True)[:n_worst]
    print(f'\n=== TOP-{n_worst} WORST (ours much worse than aec2) ===')
    print(f'{"rank":>4s} {"stem":<55s} {"mv":>3s} {"Δecho":>7s} {"Δdeg":>7s}'
          f'  {"conv_s":>7s} {"epc%":>5s} {"erl_max":>8s}')
    for i, r in enumerate(rs):
        cs = f'{r["first_conv_s"]:.1f}' if r["first_conv_s"] >= 0 else 'NEVER'
        mv = 'mv' if r['movement'] else 'st'
        print(f'{i+1:>4d} {r["stem"]:<55s} {mv:>3s} {r["d_echo"]:>+7.3f} {r["d_deg"]:>+7.3f}'
              f'  {cs:>7s} {r["epc_pct"]*100:>4.0f}% {r["erl_max"]:>7.3f}')


def main():
    OUT.mkdir(exist_ok=True)
    cases = _all_dt_cases()
    print(f'cases: {len(cases)}')
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=4,
                             initializer=_init_worker,
                             initargs=(str(MODEL),)) as ex:
        futs = [ex.submit(_run_one, c) for c in cases]
        done = 0
        for fu in as_completed(futs):
            rows.append(fu.result())
            done += 1
            if done % 50 == 0:
                print(f'  [{done}/{len(cases)}]', flush=True)
    print(f'elapsed: {time.time()-t0:.1f}s')

    json_out = OUT / 'per_case.json'
    json_out.write_text(json.dumps(rows, indent=2))
    print(f'saved: {json_out}')

    _print_overall(rows)
    buckets = _bucket_by_conv(rows)
    _print_buckets(buckets)
    _print_worst(rows, 15)

    # Sub-analysis: convergence time histogram
    convs = [r['first_conv_s'] for r in rows if not r['never_converged']]
    nevers = sum(1 for r in rows if r['never_converged'])
    print(f'\n=== convergence-time histogram ===')
    print(f'  never converged: {nevers} ({100*nevers/len(rows):.1f}%)')
    if convs:
        print(f'  converged: n={len(convs)}, mean={np.mean(convs):.2f}s, '
              f'median={np.median(convs):.2f}s, '
              f'p25={np.percentile(convs,25):.2f}s, p75={np.percentile(convs,75):.2f}s')


if __name__ == '__main__':
    main()
