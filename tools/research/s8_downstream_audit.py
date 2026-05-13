#!/usr/bin/env python3
"""S8 downstream-clamp audit (Phase 3B v4).

Identifies the FS-output-visible carrier after the S6 / S6b / S7
three-trial null on Q7 V3 RES gain / evidence layers. Audits two
downstream clamp groups in FS bins (coh² < 0.1):

1. Stage 1 4-cap chain on residual_echo_psd (aec.py:2138/2144/2151/2163):
   cap1 echo×2, cap2 error×mult, cap3 dt_suppress, cap4 render_ceil
2. Nearend_est 4-way floor binding (aec.py:2363/2372/2377):
   raw (raw_nearend_est × dt_shaped), noise_floor_psd, min_ne_from_dt,
   ne_physical_floor

Output JSON has per-bucket aggregates + per-case detail.

Design ref: docs/v3_12_s7_verdict.md §9 (post-S7 re-investigation list).

Usage:
    python3 tools/research/s8_downstream_audit.py --out results/v3_12_s8_audit -j 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    """Bit-for-bit port of eval_aec_challenge.estimate_delay (GCC-PHAT)."""
    max_d = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref))
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val_phat = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx_phat = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val_phat / (rms + 1e-10)
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx_phat
    return delay


def _run_one(mic_path, lpb_path, stem, scenario, movement, fl, enable_cng):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    try:
        mic, sr_m = sf.read(mic_path)
        lpb, sr_l = sf.read(lpb_path)
        if sr_m != sr_l:
            return {'stem': stem, 'status': f'sr_mismatch:{sr_m},{sr_l}'}
        sr = int(sr_m)
        mic = mic.astype(np.float32); lpb = lpb.astype(np.float32)
        delay = _estimate_delay(mic, lpb, sr)
        n = min(len(mic), len(lpb))
        if 0 < delay < n:
            lpb_aligned = np.zeros(n, dtype=np.float32)
            lpb_aligned[delay:] = lpb[: n - delay]
        else:
            lpb_aligned = lpb[:n]
        mic = mic[:n]
        if movement:
            delay_est_kw = dict(enable_delay_est=True,
                                delay_est_period_s=0.25, delay_est_init_s=0.2)
        else:
            delay_est_kw = dict(enable_delay_est=False)
        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr, mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False, enable_shadow=True, enable_res=True,
            enable_cng=enable_cng, use_kalman=True,
            **delay_est_kw,
        )
        np.random.seed(0)
        aec = AEC(cfg)
        aec.enable_res_audit()
        hop = aec.hop_size
        pos = 0
        while pos + hop <= n:
            aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
            pos += hop
        ac = aec.get_res_audit()
        return {'stem': stem, 'scenario': scenario, 'movement': movement,
                'status': 'ok', 'elapsed_s': time.time() - t0,
                'counters': ac}
    except Exception as e:
        return {'stem': stem, 'status': f'err:{e}'}


def _collect_cases(dataset_dir):
    cases = []
    for scenario in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        sd = Path(dataset_dir) / scenario
        if not sd.is_dir(): continue
        for mic_f in sorted(sd.glob('*_mic.wav')):
            stem = mic_f.name[: -len('_mic.wav')]
            lpb_f = sd / f'{stem}_lpb.wav'
            if not lpb_f.is_file(): continue
            mvt = '_with_movement' in stem
            cases.append((stem, scenario, mvt, str(mic_f), str(lpb_f)))
    return cases


def _bucket_of(scenario, movement):
    if scenario == 'farend_singletalk':
        return 'FS_movement' if movement else 'FS_static'
    if scenario == 'doubletalk':
        return 'DT_movement' if movement else 'DT_static'
    return 'NE'


def _aggregate(per_case):
    buckets = ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE')
    s8_keys = (
        's8_stage1_fs_bin_total',
        's8_cap1_echo_x2_binding', 's8_cap1_echo_x2_reduction_sum',
        's8_cap2_err_mult_binding', 's8_cap2_err_mult_reduction_sum',
        's8_cap3_dt_suppress_binding', 's8_cap3_dt_suppress_reduction_sum',
        's8_cap4_render_ceil_binding', 's8_cap4_render_ceil_reduction_sum',
        's8_nef_raw_count', 's8_nef_noise_floor_count',
        's8_nef_min_ne_count', 's8_nef_ne_physical_count',
    )
    agg = {b: {k: (0.0 if k.endswith('_sum') else 0) for k in s8_keys}
              | {'cases': 0, 'frames': 0} for b in buckets + ('GLOBAL',)}
    for rec in per_case:
        if rec.get('status') != 'ok': continue
        b = _bucket_of(rec['scenario'], rec['movement'])
        c = rec['counters']
        for k in s8_keys:
            agg[b][k] += c[k]
            agg['GLOBAL'][k] += c[k]
        agg[b]['cases'] += 1
        agg[b]['frames'] += c['total_frames']
        agg['GLOBAL']['cases'] += 1
        agg['GLOBAL']['frames'] += c['total_frames']

    for b, d in agg.items():
        n = d['s8_stage1_fs_bin_total']
        if n > 0:
            d['cap1_pct'] = d['s8_cap1_echo_x2_binding'] / n * 100
            d['cap2_pct'] = d['s8_cap2_err_mult_binding'] / n * 100
            d['cap3_pct'] = d['s8_cap3_dt_suppress_binding'] / n * 100
            d['cap4_pct'] = d['s8_cap4_render_ceil_binding'] / n * 100
            d['cap1_mean_dB_red'] = (
                d['s8_cap1_echo_x2_reduction_sum'] /
                max(d['s8_cap1_echo_x2_binding'], 1))
            d['cap2_mean_dB_red'] = (
                d['s8_cap2_err_mult_reduction_sum'] /
                max(d['s8_cap2_err_mult_binding'], 1))
            d['cap3_mean_dB_red'] = (
                d['s8_cap3_dt_suppress_reduction_sum'] /
                max(d['s8_cap3_dt_suppress_binding'], 1))
            d['cap4_mean_dB_red'] = (
                d['s8_cap4_render_ceil_reduction_sum'] /
                max(d['s8_cap4_render_ceil_binding'], 1))
        else:
            d['cap1_pct'] = d['cap2_pct'] = d['cap3_pct'] = d['cap4_pct'] = 0.0
            d['cap1_mean_dB_red'] = d['cap2_mean_dB_red'] = 0.0
            d['cap3_mean_dB_red'] = d['cap4_mean_dB_red'] = 0.0
        nef_total = (d['s8_nef_raw_count'] + d['s8_nef_noise_floor_count']
                     + d['s8_nef_min_ne_count'] + d['s8_nef_ne_physical_count'])
        d['nef_total'] = nef_total
        if nef_total > 0:
            d['nef_raw_pct'] = d['s8_nef_raw_count'] / nef_total * 100
            d['nef_nf_pct'] = d['s8_nef_noise_floor_count'] / nef_total * 100
            d['nef_min_ne_pct'] = d['s8_nef_min_ne_count'] / nef_total * 100
            d['nef_phys_pct'] = d['s8_nef_ne_physical_count'] / nef_total * 100
        else:
            d['nef_raw_pct'] = d['nef_nf_pct'] = 0.0
            d['nef_min_ne_pct'] = d['nef_phys_pct'] = 0.0
    return agg


def main():
    ap = argparse.ArgumentParser(description='S8 downstream-clamp audit')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    ap.add_argument('--filter', type=int, default=832)
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--no-cng', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = _collect_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]
    enable_cng = not args.no_cng
    print(f'[s8-audit] cases={len(cases)} jobs={args.jobs} cng={enable_cng} '
          f'out={args.out}', flush=True)

    t_start = time.time()
    per_case = []; n_done = 0; n_err = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for stem, scen, mvt, mic_p, lpb_p in cases:
            futures[pool.submit(_run_one, mic_p, lpb_p, stem, scen, mvt,
                                args.filter, enable_cng)] = stem
        for fut in as_completed(futures):
            rec = fut.result()
            per_case.append(rec)
            n_done += 1
            if rec.get('status') != 'ok':
                n_err += 1
                print(f'[s8-audit] ERR {rec.get("stem")}: '
                      f'{rec.get("status")}', flush=True)
            if n_done % 100 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[s8-audit] {n_done}/{len(cases)} errors={n_err} '
                      f'elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    agg = _aggregate(per_case)
    total = time.time() - t_start
    out_json = {
        'design_ref': 'docs/v3_12_s7_verdict.md §9 (post-S7 re-investigation)',
        'preset': 'BALANCED', 'filter_length': args.filter,
        'enable_cng': enable_cng, 'seed': 0,
        'cases_run': n_done, 'errors': n_err, 'elapsed_s': total,
        'per_bucket': agg, 'per_case': per_case,
    }
    out_path = os.path.join(args.out, 'audit.json')
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    print()
    print('=' * 100)
    print('Stage 1 4-cap binding (FS bins coh²<0.1, % of FS bin total)')
    print('-' * 100)
    print(f'{"bucket":<14} {"cases":>5} {"FS bins":>10} '
          f'{"cap1_echo×2":>11} {"cap2_err×":>10} {"cap3_dt":>8} {"cap4_rc":>8}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["cases"]:>5} {d["s8_stage1_fs_bin_total"]:>10} '
              f'{d["cap1_pct"]:>10.2f}% {d["cap2_pct"]:>9.2f}% '
              f'{d["cap3_pct"]:>7.2f}% {d["cap4_pct"]:>7.2f}%')
    print()
    print('Nearend_est 4-way binding (FS bins, % of nef total)')
    print('-' * 100)
    print(f'{"bucket":<14} {"nef total":>11} {"raw NE":>8} {"noise_floor":>12} '
          f'{"min_ne_dt":>11} {"ne_phys":>9}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["nef_total"]:>11} {d["nef_raw_pct"]:>7.2f}% '
              f'{d["nef_nf_pct"]:>11.2f}% {d["nef_min_ne_pct"]:>10.2f}% '
              f'{d["nef_phys_pct"]:>8.2f}%')
    print('=' * 100)
    print(f'[s8-audit] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{n_done}')
    print(f'[s8-audit] Wrote {out_path}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
