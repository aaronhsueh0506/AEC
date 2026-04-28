"""Stage -1.1: Profile _erl_estimate distribution across 800-case set.

Outputs per-case summary stats (p50/p90/p99/max, frame count above
common thresholds) and aggregated distributions per scenario.
"""
import os, glob, json
from collections import defaultdict
import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, '.')
from aec import AEC, AecConfig

DATASET = '../wav/aec_challenge_blind'
SCENARIOS = {
    'farend_singletalk':   ['_mic.wav', '_lpb.wav'],
    'doubletalk':          ['_mic.wav', '_lpb.wav'],
    'nearend_singletalk':  ['_mic.wav', '_lpb.wav'],
}
TOP_LOSERS = set([
    'iOyPaxX11UOaUkcscKhq1A_farend_singletalk_with_movement',
    'JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk',
    'JjCzlhn3gEiBQvfJtPNJ9A_farend_singletalk_with_movement',
    'VJfVUwJs4k25ziMNvJb43A_farend_singletalk',
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',
    'nyT6FUUdu0W8UpvjP1rRgQ_doubletalk_with_movement',
    'wHmBm7VHfkysBOhjoAXkNA_doubletalk_with_movement',
    'yc5bFUGsR0GSfiGwTTpRWg_doubletalk',
    'XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement',
    'QK70KpLuZ0O43BBSWEZvHg_doubletalk',
])


def profile_one(mic_path, lpb_path):
    mic, sr = sf.read(mic_path); far, _ = sf.read(lpb_path)
    n = min(len(mic), len(far))
    cfg = AecConfig.from_preset('balanced'); cfg.filter_length = 448
    aec = AEC(config=cfg)
    hop = aec.hop_size
    erl_trace = []
    for pos in range(0, n - hop, hop):
        aec.process(mic[pos:pos+hop].astype(np.float32),
                    far[pos:pos+hop].astype(np.float32))
        erl_trace.append(float(aec._erl_estimate))
    arr = np.asarray(erl_trace, dtype=np.float32)
    return {
        'frames': int(arr.size),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(arr.max()),
        'min': float(arr.min()),
        'mean': float(arr.mean()),
        'frac_gt_0p1': float((arr > 0.1).mean()),
        'frac_gt_0p3': float((arr > 0.3).mean()),
        'frac_gt_1p0': float((arr > 1.0).mean()),
        'frac_gt_1p2': float((arr > 1.2).mean()),
    }


def main():
    out_dir = '/tmp/erl_profile'
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        scenario_dir = os.path.join(DATASET, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        mic_files = sorted(glob.glob(os.path.join(scenario_dir, '*_mic.wav')))
        for i, mic_path in enumerate(mic_files):
            base = mic_path.replace('_mic.wav', '')
            lpb_path = base + '_lpb.wav'
            if not os.path.exists(lpb_path):
                continue
            case_id = os.path.basename(base)
            try:
                stats = profile_one(mic_path, lpb_path)
            except Exception as e:
                print(f'  ERROR {case_id}: {e}')
                continue
            stats['scenario'] = scenario
            stats['case_id'] = case_id
            stats['is_top_loser'] = case_id in TOP_LOSERS
            rows.append(stats)
            if (i + 1) % 50 == 0:
                print(f'{scenario} {i+1}/{len(mic_files)}')
    with open(os.path.join(out_dir, 'per_case.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print(f'wrote {len(rows)} cases to {out_dir}/per_case.json')

    # Aggregate
    by_sc = defaultdict(list)
    for r in rows:
        by_sc[r['scenario']].append(r)
    print()
    print('=== Aggregate per scenario ===')
    print(f'{"scenario":25s} {"N":>4s} {"p50":>8s} {"p90":>8s} {"p99":>8s} {"max":>8s} {"%>0.1":>7s} {"%>0.3":>7s} {"%>1.2":>7s}')
    for sc, lst in sorted(by_sc.items()):
        p50 = np.median([r['p50'] for r in lst])
        p90 = np.median([r['p90'] for r in lst])
        p99 = np.median([r['p99'] for r in lst])
        mx = np.median([r['max'] for r in lst])
        f1 = np.mean([r['frac_gt_0p1'] for r in lst]) * 100
        f3 = np.mean([r['frac_gt_0p3'] for r in lst]) * 100
        f12 = np.mean([r['frac_gt_1p2'] for r in lst]) * 100
        print(f'{sc:25s} {len(lst):4d} {p50:8.4f} {p90:8.4f} {p99:8.4f} {mx:8.4f} {f1:7.2f} {f3:7.2f} {f12:7.2f}')

    # Top losers
    print()
    print('=== Top losers ===')
    losers = [r for r in rows if r['is_top_loser']]
    print(f'{"case":40s} {"p50":>8s} {"p90":>8s} {"p99":>8s} {"max":>8s} {"%>0.3":>7s} {"%>1.2":>7s}')
    for r in losers:
        print(f'{r["case_id"][:40]:40s} {r["p50"]:8.4f} {r["p90"]:8.4f} {r["p99"]:8.4f} {r["max"]:8.4f} {r["frac_gt_0p3"]*100:7.2f} {r["frac_gt_1p2"]*100:7.2f}')


if __name__ == '__main__':
    main()
