#!/usr/bin/env python3
"""S9 noise_floor_psd refinement pre-implementation audit (Phase 3B v5).

S8 identified `noise_floor_psd = mean(error_psd) * 0.01` as the dominant
FS-visible nearend_est carrier (~43% binding global, ~37% min_ne_from_dt
is #2). This audit measures, on the production 800-case corpus, the
fate of baseline-floor bins under two candidate refinements:

  A.1: scalar × 0.001  (10× lower than baseline 0.01)
  A.2: per-bin error_psd × 0.005  (scalar → per-bin, ~half the mean
                                   coefficient)

For each FS bin (coh² < 0.1) where the baseline winner is noise_floor,
A.1 and A.2 yield one of four new outcomes:

  release_to_raw   : raw_nearend_est × dt_shaped now wins (good — FS-
                     honest evidence becomes binding; nearend_est
                     drops to true NE level)
  shift_to_min_ne  : min_ne_from_dt now wins (carrier shifts from #1
                     to #2; nearend_est still drops but only down to
                     min_ne level, not to raw_NE)
  shift_to_phys    : ne_physical_floor wins (dead zone; S8 confirmed
                     0% binding under baseline so this should be tiny)
  stays_floor      : candidate still highest (A.1 mostly when raw_NE
                     is tiny; A.2 can never stay if error_psd was
                     already < 2× mean)

`reduction_db` is the mean 10·log10(baseline_max / candidate_max) over
all FS bins where the candidate strictly lowers nearend_est — magnitude
of the suppression-side push.

`intrudes_outside_floor`: FS bins where baseline winner was NOT floor
but A.1/A.2 would dethrone the current winner. A.1 (always lower) =
should be 0. A.2 can intrude when error_psd[i] is high enough that
error_psd[i] × 0.005 exceeds the current winner. Tracks A.2's
downside (creates new ceiling problems in high-energy bins).

Design ref: docs/v3_12_s8_verdict.md §S9 candidate listing.

Usage:
    python3 tools/research/s9_noise_floor_audit.py \\
        --out results/v3_12_s9_audit -j 4
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
        's8_nef_raw_count', 's8_nef_noise_floor_count',
        's8_nef_min_ne_count', 's8_nef_ne_physical_count',
    )
    s9_keys = (
        's9_a1_release_to_raw', 's9_a1_shift_to_min_ne',
        's9_a1_shift_to_phys', 's9_a1_stays_floor',
        's9_a1_reduction_db_sum', 's9_a1_reduction_count',
        's9_a1_intrudes_outside_floor_baseline',
        's9_a2_release_to_raw', 's9_a2_shift_to_min_ne',
        's9_a2_shift_to_phys', 's9_a2_stays_floor',
        's9_a2_reduction_db_sum', 's9_a2_reduction_count',
        's9_a2_intrudes_outside_floor_baseline',
    )
    s9c_keys = (
        's9c_baseline_any_floor_count',
        's9c_c1_release_to_raw', 's9c_c1_still_floor',
        's9c_c1_still_min_ne', 's9c_c1_still_phys',
        's9c_c1_reduction_db_sum', 's9c_c1_reduction_count',
        's9c_c2_release_to_raw', 's9c_c2_still_floor',
        's9c_c2_still_min_ne', 's9c_c2_still_phys',
        's9c_c2_reduction_db_sum', 's9c_c2_reduction_count',
    )
    s9d_keys = (
        's9d_release_to_raw', 's9d_still_floor',
        's9d_reduction_db_sum', 's9d_reduction_count',
    )
    all_keys = s8_keys + s9_keys + s9c_keys + s9d_keys
    agg = {b: {k: (0.0 if k.endswith('_sum') else 0) for k in all_keys}
              | {'cases': 0, 'frames': 0} for b in buckets + ('GLOBAL',)}
    for rec in per_case:
        if rec.get('status') != 'ok': continue
        b = _bucket_of(rec['scenario'], rec['movement'])
        c = rec['counters']
        for k in all_keys:
            agg[b][k] += c[k]
            agg['GLOBAL'][k] += c[k]
        agg[b]['cases'] += 1
        agg[b]['frames'] += c['total_frames']
        agg['GLOBAL']['cases'] += 1
        agg['GLOBAL']['frames'] += c['total_frames']

    for b, d in agg.items():
        # Baseline nef distribution
        nef_total = (d['s8_nef_raw_count'] + d['s8_nef_noise_floor_count']
                     + d['s8_nef_min_ne_count'] + d['s8_nef_ne_physical_count'])
        d['nef_total'] = nef_total
        d['nef_floor_count'] = d['s8_nef_noise_floor_count']
        if nef_total > 0:
            d['nef_floor_pct'] = d['s8_nef_noise_floor_count'] / nef_total * 100
        else:
            d['nef_floor_pct'] = 0.0

        # S9 percentages — denominator = baseline floor count (the
        # bins these candidates are trying to address)
        floor_n = d['s8_nef_noise_floor_count']
        for cand in ('a1', 'a2'):
            release = d[f's9_{cand}_release_to_raw']
            shift = d[f's9_{cand}_shift_to_min_ne']
            phys = d[f's9_{cand}_shift_to_phys']
            stays = d[f's9_{cand}_stays_floor']
            if floor_n > 0:
                d[f'{cand}_release_pct'] = release / floor_n * 100
                d[f'{cand}_shift_min_ne_pct'] = shift / floor_n * 100
                d[f'{cand}_shift_phys_pct'] = phys / floor_n * 100
                d[f'{cand}_stays_pct'] = stays / floor_n * 100
            else:
                d[f'{cand}_release_pct'] = 0.0
                d[f'{cand}_shift_min_ne_pct'] = 0.0
                d[f'{cand}_shift_phys_pct'] = 0.0
                d[f'{cand}_stays_pct'] = 0.0
            n_red = d[f's9_{cand}_reduction_count']
            d[f'{cand}_reduction_mean_db'] = (
                d[f's9_{cand}_reduction_db_sum'] / n_red if n_red > 0 else 0.0)
            d[f'{cand}_intrudes_pct_of_nef'] = (
                d[f's9_{cand}_intrudes_outside_floor_baseline']
                / nef_total * 100 if nef_total > 0 else 0.0)

        # S9-C: percentages denominator = baseline ANY-floor bins
        any_floor_n = d['s9c_baseline_any_floor_count']
        for cand in ('c1', 'c2'):
            release = d[f's9c_{cand}_release_to_raw']
            sf_ = d[f's9c_{cand}_still_floor']
            sm = d[f's9c_{cand}_still_min_ne']
            sp = d[f's9c_{cand}_still_phys']
            if any_floor_n > 0:
                d[f'{cand}_c_release_pct'] = release / any_floor_n * 100
                d[f'{cand}_c_still_floor_pct'] = sf_ / any_floor_n * 100
                d[f'{cand}_c_still_min_ne_pct'] = sm / any_floor_n * 100
                d[f'{cand}_c_still_phys_pct'] = sp / any_floor_n * 100
            else:
                d[f'{cand}_c_release_pct'] = 0.0
                d[f'{cand}_c_still_floor_pct'] = 0.0
                d[f'{cand}_c_still_min_ne_pct'] = 0.0
                d[f'{cand}_c_still_phys_pct'] = 0.0
            n_red = d[f's9c_{cand}_reduction_count']
            d[f'{cand}_c_reduction_mean_db'] = (
                d[f's9c_{cand}_reduction_db_sum'] / n_red if n_red > 0 else 0.0)

        # S9-D
        if any_floor_n > 0:
            d['d_release_pct'] = d['s9d_release_to_raw'] / any_floor_n * 100
            d['d_still_floor_pct'] = d['s9d_still_floor'] / any_floor_n * 100
        else:
            d['d_release_pct'] = 0.0
            d['d_still_floor_pct'] = 0.0
        n_red_d = d['s9d_reduction_count']
        d['d_reduction_mean_db'] = (
            d['s9d_reduction_db_sum'] / n_red_d if n_red_d > 0 else 0.0)
    return agg


def main():
    ap = argparse.ArgumentParser(
        description='S9 noise_floor_psd refinement pre-audit')
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
    print(f'[s9-audit] cases={len(cases)} jobs={args.jobs} cng={enable_cng} '
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
                print(f'[s9-audit] ERR {rec.get("stem")}: '
                      f'{rec.get("status")}', flush=True)
            if n_done % 100 == 0 or n_done == len(cases):
                elapsed = time.time() - t_start
                eta = elapsed / n_done * (len(cases) - n_done)
                print(f'[s9-audit] {n_done}/{len(cases)} errors={n_err} '
                      f'elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

    agg = _aggregate(per_case)
    total = time.time() - t_start
    out_json = {
        'design_ref': 'docs/v3_12_s8_verdict.md §S9 candidate listing',
        'preset': 'BALANCED', 'filter_length': args.filter,
        'enable_cng': enable_cng, 'seed': 0,
        'cases_run': n_done, 'errors': n_err, 'elapsed_s': total,
        'per_bucket': agg, 'per_case': per_case,
    }
    out_path = os.path.join(args.out, 'audit.json')
    with open(out_path, 'w') as f:
        json.dump(out_json, f, indent=2)

    print()
    print('=' * 110)
    print('S9 pre-audit — fate of baseline noise_floor-bound FS bins under '
          'candidates A.1 / A.2')
    print('  A.1 = scalar × 0.001   (10× lower than baseline)')
    print('  A.2 = per-bin error_psd × 0.005')
    print('-' * 110)
    print(f'{"bucket":<14} {"floor bins":>11} {"A.1 release→raw":>16} '
          f'{"A.1 →min_ne":>13} {"A.1 stays":>11} {"A.2 release→raw":>16} '
          f'{"A.2 →min_ne":>13} {"A.2 stays":>11}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["nef_floor_count"]:>11} '
              f'{d["a1_release_pct"]:>15.2f}% {d["a1_shift_min_ne_pct"]:>12.2f}% '
              f'{d["a1_stays_pct"]:>10.2f}% '
              f'{d["a2_release_pct"]:>15.2f}% {d["a2_shift_min_ne_pct"]:>12.2f}% '
              f'{d["a2_stays_pct"]:>10.2f}%')

    print()
    print('Magnitude — mean dB reduction in nearend_est across changed bins,')
    print('  and A.2 intrusion pct (bins where A.2 dethrones a non-floor winner)')
    print('-' * 110)
    print(f'{"bucket":<14} {"A.1 mean dB":>13} {"A.1 changed":>13} '
          f'{"A.2 mean dB":>13} {"A.2 changed":>13} {"A.2 intrude%":>14}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["a1_reduction_mean_db"]:>12.2f}  '
              f'{d["s9_a1_reduction_count"]:>13} '
              f'{d["a2_reduction_mean_db"]:>12.2f}  '
              f'{d["s9_a2_reduction_count"]:>13} '
              f'{d["a2_intrudes_pct_of_nef"]:>13.2f}%')
    print('=' * 110)

    print()
    print('=' * 110)
    print('S9-C joint pre-audit — fate of ANY-floor-bound FS bins under '
          'joint candidates C.1 / C.2')
    print('  C.1 = noise_floor → error_psd × 0.005  + min_ne_from_dt × 0.1')
    print('  C.2 = noise_floor → error_psd × 0.005  + min_ne_from_dt → 0')
    print('-' * 110)
    print(f'{"bucket":<14} {"any-floor bins":>14} '
          f'{"C.1 release":>12} {"C.1 still_floor":>16} '
          f'{"C.1 still_min_ne":>17} {"C.1 mean dB":>12} '
          f'{"C.2 release":>12} {"C.2 still_min_ne":>17} '
          f'{"C.2 mean dB":>12}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["s9c_baseline_any_floor_count"]:>14} '
              f'{d["c1_c_release_pct"]:>11.2f}% '
              f'{d["c1_c_still_floor_pct"]:>15.2f}% '
              f'{d["c1_c_still_min_ne_pct"]:>16.2f}% '
              f'{d["c1_c_reduction_mean_db"]:>11.2f}  '
              f'{d["c2_c_release_pct"]:>11.2f}% '
              f'{d["c2_c_still_min_ne_pct"]:>16.2f}% '
              f'{d["c2_c_reduction_mean_db"]:>11.2f}')
    print('=' * 110)

    print()
    print('=' * 110)
    print('S9-D sanity pre-audit — attack ALL 3 floors. '
          'Stack = [raw*dt_shaped, error_psd*0.005, 0, 0]')
    print('  Confirms nearend_est stack fully controls FS release '
          '(release ≥90% → no hidden 4th carrier)')
    print('-' * 110)
    print(f'{"bucket":<14} {"any-floor bins":>14} {"D release":>12} '
          f'{"D still_floor":>15} {"D mean dB":>12}')
    for b in ('FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE',
              'GLOBAL'):
        d = agg[b]
        print(f'{b:<14} {d["s9c_baseline_any_floor_count"]:>14} '
              f'{d["d_release_pct"]:>11.2f}% '
              f'{d["d_still_floor_pct"]:>14.2f}% '
              f'{d["d_reduction_mean_db"]:>11.2f}')
    print('=' * 110)

    print(f'[s9-audit] DONE in {total:.0f}s ({total/60:.1f}min) '
          f'errors={n_err}/{n_done}')
    print(f'[s9-audit] Wrote {out_path}')
    return 0 if n_err == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
