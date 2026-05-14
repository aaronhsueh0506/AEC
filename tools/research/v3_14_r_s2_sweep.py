#!/usr/bin/env python3
"""v3.14 Arc-R Sprint S2 — per-band ENR threshold tuning sweep.

Runs 3-point coarse grid over `enr_t_ne_per_band` (mirror-scaled
`enr_s_ne_per_band` ~1.67×) with both `f3_1_per_band_erl_adaptive=True`
and `res_per_band_enr=True` for end-to-end per-band evaluation.

Each grid point renders the 800-case BALANCED corpus + scores via
local AECMOS ONNX, then diffs against the v3.13.0 baseline at
`/Users/mingyu/Desktop/novatek/SE/AEC/results/v3_14_baseline/scores.json`.

Standard config: preset=balanced, fl=832, cng=True, parallel=True (3-way
scenario fork). Uses ProcessPoolExecutor inside `eval_aec_challenge.py`
for scenario parallelism — we do not spawn multiple concurrent benches.

Usage:
    python3 tools/research/v3_14_r_s2_sweep.py \\
        --dataset /Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind \\
        --baseline /Users/mingyu/Desktop/novatek/SE/AEC/results/v3_14_baseline/scores.json \\
        --output-root /tmp/v3_14_r_s2 \\
        --grid block_lf uniform admit_hf

Optional: --grid uniform → run a single point first as smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_PY = _REPO / 'python'


# Grid: (label, t_ne_pb, s_ne_pb).  s_ne mirrors t_ne with ~1.67× scaling
# to preserve the legacy t:s ratio (1.5:2.5).
GRID = {
    'block_lf':  ((2.0, 1.5, 1.0), (3.33, 2.5, 1.67)),   # block LF, admit HF
    'uniform':   ((1.5, 1.5, 1.5), (2.5,  2.5, 2.5 )),   # uniform default
    'admit_hf':  ((1.0, 1.5, 2.0), (1.67, 2.5, 3.33)),   # admit LF, block HF
}


def _fmt_tuple(t):
    return ','.join(f'{x:.4f}' for x in t)


def run_bench(label: str, t_pb: tuple, s_pb: tuple,
              dataset: str, output_root: str) -> tuple[str, float]:
    """Run a single bench + AECMOS scoring; return (results_dir, elapsed_s)."""
    out_dir = os.path.join(output_root, f'out_python_{label}')
    res_dir = os.path.join(output_root, f'results_{label}')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    env = os.environ.copy()
    env['AEC_F3_1_PER_BAND_ERL'] = '1'
    env['AEC_RES_PER_BAND_ENR'] = '1'
    env['AEC_ENR_T_NE_PB'] = _fmt_tuple(t_pb)
    env['AEC_ENR_S_NE_PB'] = _fmt_tuple(s_pb)

    print(f'\n{"=" * 72}')
    print(f'GRID POINT: {label}')
    print(f'  enr_t_ne_per_band = {t_pb}')
    print(f'  enr_s_ne_per_band = {s_pb}')
    print(f'  env: AEC_F3_1_PER_BAND_ERL=1 AEC_RES_PER_BAND_ENR=1')
    print(f'       AEC_ENR_T_NE_PB={_fmt_tuple(t_pb)}')
    print(f'       AEC_ENR_S_NE_PB={_fmt_tuple(s_pb)}')
    print(f'  output: {out_dir}')
    print(f'  results: {res_dir}')
    print('=' * 72)

    # Step 1: render 800-case BALANCED corpus
    t0 = time.time()
    cmd_render = [
        sys.executable, str(_PY / 'eval_aec_challenge.py'),
        dataset,
        '--preset', 'balanced',
        '--filter', '832',
        '--cng',
        '--parallel',
        '-o', out_dir,
    ]
    print(f'\n[render] {" ".join(cmd_render)}')
    proc = subprocess.run(cmd_render, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print('RENDER STDOUT:', proc.stdout[-2000:])
        print('RENDER STDERR:', proc.stderr[-2000:])
        raise RuntimeError(f'render failed for {label}')
    t_render = time.time() - t0
    print(f'[render] done in {t_render:.1f}s')

    # Step 2: AECMOS score
    t1 = time.time()
    cmd_score = [
        sys.executable, str(_PY / 'bench_aecmos.py'),
        out_dir, res_dir,
    ]
    print(f'[score] {" ".join(cmd_score)}')
    proc = subprocess.run(cmd_score, capture_output=True, text=True)
    if proc.returncode != 0:
        print('SCORE STDOUT:', proc.stdout[-2000:])
        print('SCORE STDERR:', proc.stderr[-2000:])
        raise RuntimeError(f'AECMOS score failed for {label}')
    t_score = time.time() - t1
    elapsed = time.time() - t0
    print(f'[score] done in {t_score:.1f}s  (total {elapsed:.1f}s)')

    return res_dir, elapsed


def compare(results_dir: str, baseline_path: str, label: str) -> dict:
    """Compute Δ vs baseline for buckets + cohort tail; return summary dict."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(os.path.join(results_dir, 'scores.json')) as f:
        current = json.load(f)

    b_sum = baseline['summary']
    c_sum = current['summary']

    buckets = ['NE', 'FS_static', 'FS_movement', 'DT_static', 'DT_movement']
    rows = []
    for b in buckets:
        bm = b_sum.get(b, {})
        cm = c_sum.get(b, {})
        if not bm or not cm:
            continue
        d_echo = cm['echo_mean'] - bm['echo_mean']
        d_deg = cm['deg_mean'] - bm['deg_mean']
        rows.append({
            'bucket': b,
            'n': cm['n'],
            'echo_base': bm['echo_mean'],
            'echo_cur': cm['echo_mean'],
            'd_echo': d_echo,
            'deg_base': bm['deg_mean'],
            'deg_cur': cm['deg_mean'],
            'd_deg': d_deg,
        })

    # cohort tail
    cohort_stem = 'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk'
    base_cohort = baseline['scores'].get(cohort_stem, {})
    cur_cohort = current['scores'].get(cohort_stem, {})
    cohort = None
    if base_cohort and cur_cohort:
        cohort = {
            'stem': cohort_stem,
            'echo_base': base_cohort['echo'],
            'echo_cur': cur_cohort['echo'],
            'd_echo': cur_cohort['echo'] - base_cohort['echo'],
            'deg_base': base_cohort['deg'],
            'deg_cur': cur_cohort['deg'],
            'd_deg': cur_cohort['deg'] - base_cohort['deg'],
        }

    # Hard bar evaluation
    pass_bars = True
    fail_reasons = []
    for r in rows:
        b = r['bucket']
        if b in ('FS_static', 'FS_movement') and r['d_echo'] < -0.02:
            pass_bars = False
            fail_reasons.append(f'{b} Δecho={r["d_echo"]:+.3f} < -0.020')
        if b in ('NE', 'DT_static', 'DT_movement') and r['d_deg'] < -0.005:
            pass_bars = False
            fail_reasons.append(f'{b} Δdeg={r["d_deg"]:+.3f} < -0.005')
    if cohort and cohort['d_echo'] < -0.05:
        pass_bars = False
        fail_reasons.append(f'cohort tail Δecho={cohort["d_echo"]:+.3f} < -0.050')

    # DT recovery (target +0.025)
    dt_static_d_deg = next((r['d_deg'] for r in rows if r['bucket'] == 'DT_static'), 0.0)
    dt_movement_d_deg = next((r['d_deg'] for r in rows if r['bucket'] == 'DT_movement'), 0.0)
    dt_mean = (dt_static_d_deg + dt_movement_d_deg) / 2.0

    return {
        'label': label,
        'rows': rows,
        'cohort': cohort,
        'pass_bars': pass_bars,
        'fail_reasons': fail_reasons,
        'dt_mean_d_deg': dt_mean,
    }


def print_summary(summaries: list):
    print(f'\n\n{"=" * 90}')
    print('GRID SUMMARY  (Δ vs v3_14_baseline)')
    print('=' * 90)
    print(f'{"label":<10} {"NE_dd":>8} {"FSst_de":>9} {"FSmv_de":>9} '
          f'{"DTst_dd":>9} {"DTmv_dd":>9} {"DT_mean":>9} {"cohort":>9} {"bars":>6}')
    print('-' * 90)
    for s in summaries:
        rows = {r['bucket']: r for r in s['rows']}
        ne_dd = rows.get('NE', {}).get('d_deg', float('nan'))
        fsst_de = rows.get('FS_static', {}).get('d_echo', float('nan'))
        fsmv_de = rows.get('FS_movement', {}).get('d_echo', float('nan'))
        dtst_dd = rows.get('DT_static', {}).get('d_deg', float('nan'))
        dtmv_dd = rows.get('DT_movement', {}).get('d_deg', float('nan'))
        cohort_de = s['cohort']['d_echo'] if s['cohort'] else float('nan')
        bars = 'PASS' if s['pass_bars'] else 'FAIL'
        print(f'{s["label"]:<10} {ne_dd:>+8.3f} {fsst_de:>+9.3f} {fsmv_de:>+9.3f} '
              f'{dtst_dd:>+9.3f} {dtmv_dd:>+9.3f} {s["dt_mean_d_deg"]:>+9.3f} '
              f'{cohort_de:>+9.3f} {bars:>6}')
        if not s['pass_bars']:
            for fr in s['fail_reasons']:
                print(f'  ↳ {fr}')
    print('=' * 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',
                   default='/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind')
    p.add_argument('--baseline',
                   default='/Users/mingyu/Desktop/novatek/SE/AEC/results/v3_14_baseline/scores.json')
    p.add_argument('--output-root', default='/tmp/v3_14_r_s2')
    p.add_argument('--grid', nargs='+', default=['block_lf', 'uniform', 'admit_hf'],
                   choices=list(GRID.keys()))
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    summaries = []
    for label in args.grid:
        t_pb, s_pb = GRID[label]
        res_dir, elapsed = run_bench(label, t_pb, s_pb, args.dataset, args.output_root)
        summary = compare(res_dir, args.baseline, label)
        summary['elapsed_s'] = elapsed
        summaries.append(summary)

        # incremental dump
        with open(os.path.join(args.output_root, 'sweep_summary.json'), 'w') as f:
            json.dump(summaries, f, indent=2)
        print_summary(summaries)

    print('\nDone.')


if __name__ == '__main__':
    main()
