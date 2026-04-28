"""Worst movement-DT case analysis (directions 3 + 4).

For each of the 114 movement-DT cases:
  1. Run our v2.8.1 AEC, score with AECMOS.
  2. Score the pre-computed AEC2 baseline output.
  3. Compute Δecho = aec2 − ours (positive = ours is worse).
  4. Sort, identify worst-N cases.
  5. For worst-N, save ours output to wav + dump state trace JSONL.

Also reports the relation between AECMOS Δ and a physical proxy
(echo-region energy ratio), to test whether AECMOS is saturated.

Usage:
    python3 diag_worst_movement_dt.py [-j 4] [-N 10] [--out output_worst]
"""
import os
import sys
import json
import argparse
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


def _movement_dt_cases():
    """All 114 doubletalk_with_movement cases."""
    out = []
    for f in sorted(WAV_BASE.iterdir()):
        if f.name.endswith('_mic.wav') and '_with_movement' in f.name:
            stem = f.name[:-len('_mic.wav')]
            aec2 = AEC2_DIR / f'{stem}_aec2.wav'
            if not aec2.is_file():
                continue
            out.append({
                'stem': stem,
                'mic': str(WAV_BASE / f.name),
                'lpb': str(WAV_BASE / f'{stem}_lpb.wav'),
                'aec2': str(aec2),
            })
    return out


_aecmos = None


def _init_worker(model_path):
    global _aecmos
    from aecmos import AECMOSEstimator
    _aecmos = AECMOSEstimator(model_path)


def _energy_db(x):
    return 10.0 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def _run_one(case, save_dir=None):
    mic, sr = sf.read(case['mic'], dtype='float32')
    lpb, _  = sf.read(case['lpb'], dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, enable_cng=False,
                                enable_delay_est=True, delay_est_period_s=0.25,
                                delay_est_init_s=0.2)
    aec = AEC(cfg)

    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    out = out[:pos]
    pad = max(0, len(mic) - len(out))
    enh = np.concatenate([out, np.zeros(pad, dtype=np.float32)])

    aec2_sig, _ = sf.read(case['aec2'], dtype='float32')
    if aec2_sig.ndim > 1: aec2_sig = aec2_sig[:, 0]
    aec2_pad_n = min(len(mic), len(aec2_sig))
    aec2_pad = np.concatenate([
        aec2_sig[:aec2_pad_n],
        np.zeros(max(0, len(mic) - aec2_pad_n), dtype=np.float32),
    ])

    ours_echo, ours_deg = _aecmos.run('dt', lpb, mic, enh)
    aec2_echo, aec2_deg = _aecmos.run('dt', lpb, mic, aec2_pad)

    # Physical proxies (reference-free, on echo-region energy):
    #   - mic_db    : mic energy
    #   - ours_db   : our output energy
    #   - aec2_db   : AEC2 output energy
    #   - far_db    : far-end energy
    # If both AEC's residual differs meaningfully but AECMOS is flat,
    # metric saturation is plausible.
    ours_db = _energy_db(enh[:aec2_pad_n])
    aec2_db = _energy_db(aec2_pad[:aec2_pad_n])
    mic_db  = _energy_db(mic[:aec2_pad_n])
    far_db  = _energy_db(lpb[:aec2_pad_n])

    result = {
        'stem': case['stem'],
        'ours_echo': float(ours_echo),
        'ours_deg':  float(ours_deg),
        'aec2_echo': float(aec2_echo),
        'aec2_deg':  float(aec2_deg),
        'd_echo':    float(aec2_echo - ours_echo),  # positive = ours worse
        'd_deg':     float(aec2_deg  - ours_deg),
        'ours_db': ours_db,
        'aec2_db': aec2_db,
        'mic_db':  mic_db,
        'far_db':  far_db,
        'd_db':    float(ours_db - aec2_db),  # positive = ours leaks more energy
    }

    if save_dir is not None:
        sf.write(str(Path(save_dir) / f'{case["stem"]}_ours.wav'), enh, sr)
    return result


def _job(args):
    case, save_dir = args
    return _run_one(case, save_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-j', type=int, default=4)
    ap.add_argument('-N', type=int, default=10, help='top-N worst cases to drill into')
    ap.add_argument('--out', default='output_worst', help='output dir for worst-N artifacts')
    args = ap.parse_args()

    n_jobs = max(1, min(args.j, 4))
    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(exist_ok=True)

    cases = _movement_dt_cases()
    print(f'movement-DT cases: {len(cases)}')

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=_init_worker,
                             initargs=(str(MODEL),)) as ex:
        futs = [ex.submit(_job, (c, None)) for c in cases]
        done = 0
        for fu in as_completed(futs):
            results.append(fu.result())
            done += 1
            if done % 20 == 0:
                print(f'  [{done}/{len(cases)}] processed', flush=True)
    print(f'  scoring elapsed: {time.time() - t0:.1f}s')

    # Sort by ours-worse-than-aec2 (descending d_echo)
    results.sort(key=lambda r: r['d_echo'], reverse=True)

    # Aggregate stats
    d_echo_list = [r['d_echo'] for r in results]
    d_deg_list  = [r['d_deg'] for r in results]
    d_db_list   = [r['d_db']  for r in results]
    print(f'\n=== overall (n={len(results)}) ===')
    print(f'  Δecho:  mean {np.mean(d_echo_list):+.3f}  median {np.median(d_echo_list):+.3f}'
          f'  std {np.std(d_echo_list):.3f}  range [{min(d_echo_list):+.3f}, {max(d_echo_list):+.3f}]')
    print(f'  Δdeg:   mean {np.mean(d_deg_list):+.3f}  median {np.median(d_deg_list):+.3f}')
    print(f'  Δdb:    mean {np.mean(d_db_list):+.2f}dB median {np.median(d_db_list):+.2f}dB'
          f'  range [{min(d_db_list):+.2f}, {max(d_db_list):+.2f}]')

    # AECMOS-vs-physical correlation (Direction 4)
    de = np.array(d_echo_list)
    dd = np.array(d_db_list)
    if len(de) >= 3:
        corr = float(np.corrcoef(de, dd)[0, 1])
        print(f'\n  corr(Δecho_AECMOS, Δenergy_dB) = {corr:+.3f}')
        print(f'  → 高正相關 (>0.5) 表示 AECMOS 隨能量單調變化（不飽和）')
        print(f'  → 低相關 (<0.3) 表示 AECMOS 飽和或被其他因素主導')

    # Top-N worst
    print(f'\n=== TOP-{args.N} WORST (ours much worse than aec2) ===')
    print(f'{"rank":>4s} {"stem":<55s} {"Δecho":>7s} {"Δdeg":>7s} {"Δdb":>7s}'
          f'  {"ours_e":>6s} {"aec2_e":>6s}')
    for i, r in enumerate(results[:args.N]):
        print(f'{i+1:>4d} {r["stem"]:<55s}'
              f' {r["d_echo"]:>+7.3f} {r["d_deg"]:>+7.3f} {r["d_db"]:>+7.2f}'
              f'  {r["ours_echo"]:>6.3f} {r["aec2_echo"]:>6.3f}')

    print(f'\n=== TOP-{args.N} BEST (ours better than aec2) ===')
    print(f'{"rank":>4s} {"stem":<55s} {"Δecho":>7s} {"Δdeg":>7s}')
    for i, r in enumerate(results[-args.N:][::-1]):
        print(f'{i+1:>4d} {r["stem"]:<55s} {r["d_echo"]:>+7.3f} {r["d_deg"]:>+7.3f}')

    # Persist ranking
    json_out = out_dir / 'movement_dt_ranking.json'
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  ranking saved: {json_out}')

    # Save worst-N ours wavs (for spectrogram analysis) — re-run AEC only, no scoring
    print(f'\n  saving worst-{args.N} ours.wav for inspection...')
    for r in results[:args.N]:
        case = next(c for c in cases if c['stem'] == r['stem'])
        mic, sr = sf.read(case['mic'], dtype='float32')
        lpb, _  = sf.read(case['lpb'], dtype='float32')
        if mic.ndim > 1: mic = mic[:, 0]
        if lpb.ndim > 1: lpb = lpb[:, 0]
        n = min(len(mic), len(lpb))
        cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                    enable_dtd=False, enable_shadow=True, enable_res=True,
                                    use_kalman=True, enable_cng=False,
                                    enable_delay_est=True, delay_est_period_s=0.25,
                                    delay_est_init_s=0.2)
        aec = AEC(cfg)
        hop = aec.hop_size
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        while pos + hop <= n:
            out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
            pos += hop
        sf.write(str(out_dir / f'{case["stem"]}_ours.wav'), out[:pos], sr)
    print(f'  saved to {out_dir}/')

    return 0


if __name__ == '__main__':
    sys.exit(main())
