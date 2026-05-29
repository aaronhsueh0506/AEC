#!/usr/bin/env python3
"""v3.21 Tier-C validation — isolated 800-case A/B of the 4 questionable shipped flags.

The 2026-05-29 conversion audit found 4 default-ON "AEC3-alignment" flags that are
mis-derived or gray-zone:
  * active_render_threshold  — 100²×64 kept against a per-sample MEAN → 64× too high;
                               mean-correct AEC3 value = 100²/32768² = 9.31e-6.
  * fft_density_scaled_psd_floors — (fft/2)/64=4× is wrong-basis; per-bin |X|² is
                               fft-independent, scales with frame samples (block/64=5×).
                               4×≈5× (within AECMOS noise) → test scaling ON(4×) vs OFF(1×).
  * reverb_smoothing (EMA-α) — gray-zone seconds-vs-count; revert = verbatim 0.2 (flag OFF).
  * dne_trigger_threshold (evidence counter) — gray-zone; revert = verbatim 12 (flag OFF).

This bench compares, ISOLATED (one change at a time vs current production), whether
correcting/reverting each regresses AECMOS — before /simplify inlines them.

  V0_prod          = plain BALANCED (current production; all 4 at shipped values)
  V1_active_render = V0 + active_render corrected to 9.31e-6
  V2_fft_off       = V0 + fft_density OFF (1× — tests whether floor scaling helps at all)
  V3_reverb_off    = V0 + reverb_smoothing OFF (verbatim 0.2)
  V4_dne_off       = V0 + dne_trigger OFF (verbatim 12)
  V5_all_corrected = active_render 9.31e-6 + reverb OFF + dne OFF (fft_density kept ON:
                     4×≈5×, V2 tests scaling separately). The honest-alignment ship config.

Standard bench config: preset=balanced / filter=832 (52ms) / cng / hop=160.
Matched-magnitude AECMOS Pareto: less echo auto-lifts deg — NOT a win.

No production-default change. No merge. Validation only.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from bench_aecmos import FastAECMOS
from v3_21_800case_bench import (
    _estimate_delay, _classify, _lf_energy_db, _discover_cases,
    CORPUS, MODEL, SR, BUCKET_METRIC,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'out_v3_21_tierc'
DOC_PATH = REPO / 'docs' / 'v3_21_tierc_validation_report.md'
JSON_PATH = OUT_DIR / 'scores_tierc.json'

CONFIG_MANIFEST = {
    'V0_prod':          {},
    'V1_active_render': {'active_render_threshold_aec3_corrected': True},
    'V2_fft_off':       {'use_aec3_fft_density_scaled_psd_floors': False},
    'V3_reverb_off':    {'use_aec3_wallclock_reverb_smoothing': False},
    'V4_dne_off':       {'use_aec3_wallclock_dne_trigger_threshold': False},
    'V5_all_corrected': {'active_render_threshold_aec3_corrected': True,
                         'use_aec3_wallclock_reverb_smoothing': False,
                         'use_aec3_wallclock_dne_trigger_threshold': False},
}
LABELS = list(CONFIG_MANIFEST)
BASELINE = 'V0_prod'
NORES_LABELS = ('V0_prod', 'V5_all_corrected')   # nores LF only for these (FS_static)


def _build_cfg(label, is_movement=False, enable_res=True):
    kw = dict(CONFIG_MANIFEST[label])
    kw['enable_res'] = enable_res
    if is_movement:
        kw.update(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
    else:
        kw['enable_delay_est'] = False
    return AecConfig.from_preset(AecPreset.BALANCED, **kw)


def _render(mic_n, ref_a, label, is_movement, enable_res=True):
    cfg = _build_cfg(label, is_movement=is_movement, enable_res=enable_res)
    np.random.seed(42)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n_hops = len(mic_n) // hop
    out = np.zeros(n_hops * hop, dtype=np.float32)
    for i in range(n_hops):
        s = i * hop
        out[s:s + hop] = aec.process(mic_n[s:s + hop], ref_a[s:s + hop])
    return out


def _case_task(args):
    stem, subdir, bucket, wavtype, mic_path, ref_path, model_path, do_nores = args
    mic_raw, _ = sf.read(mic_path)
    ref_raw, _ = sf.read(ref_path)
    if mic_raw.ndim > 1: mic_raw = mic_raw[:, 0]
    if ref_raw.ndim > 1: ref_raw = ref_raw[:, 0]
    n = min(len(mic_raw), len(ref_raw))
    mic = mic_raw[:n].astype(np.float32)
    ref = ref_raw[:n].astype(np.float32)
    is_mvmt = '_with_movement' in stem

    delay = _estimate_delay(mic, ref, SR)
    if 0 < delay < n:
        ref_a = np.zeros(n, dtype=np.float32)
        ref_a[delay:] = ref[:n - delay]
    else:
        ref_a = ref.copy()

    scorer = FastAECMOS(model_path)
    result = {'stem': stem, 'bucket': bucket, 'wavtype': wavtype}

    for label in LABELS:
        out = _render(mic, ref_a, label, is_movement=is_mvmt, enable_res=True)
        m = min(len(ref), len(mic), len(out))
        echo, deg = scorer.score(wavtype, ref[:m], mic[:m], out[:m])
        result[f'{label}_echo'] = float(echo)
        result[f'{label}_deg'] = float(deg)

    if do_nores:
        for label in NORES_LABELS:
            out_nr = _render(mic, ref_a, label, is_movement=is_mvmt, enable_res=False)
            result[f'{label}_nores_lf_db'] = float(_lf_energy_db(out_nr))

    return result


def _write_report(results):
    bucket_order = ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NE']
    # per-config per-bucket score lists
    by = {lab: defaultdict(list) for lab in LABELS}
    for r in results:
        b = r['bucket']
        metric = BUCKET_METRIC[b]
        for lab in LABELS:
            by[lab][b].append(r[f'{lab}_{metric}'])

    lines = ['# v3.21 Tier-C Validation — corrected/reverted shipped flags (isolated)\n']
    lines.append(f'**Cases**: {len(results)}  ')
    lines.append('**Config**: balanced / filter=832 / cng / hop=160  ')
    lines.append('**Baseline**: V0_prod (current production)\n')
    lines.append('Matched-magnitude AECMOS Pareto: less echo auto-lifts deg — NOT a win.\n')

    # Absolute bucket means
    lines.append('## Bucket means (absolute)\n')
    header = '| Bucket | metric | ' + ' | '.join(LABELS) + ' |'
    lines.append(header)
    lines.append('|' + '---|' * (len(LABELS) + 2))
    for b in bucket_order:
        metric = BUCKET_METRIC[b]
        cells = []
        for lab in LABELS:
            v = by[lab][b]
            cells.append(f'{np.mean(v):.3f}' if v else '—')
        lines.append(f'| {b} | {metric} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # Δ vs V0 per config (the decision table)
    lines.append('## Δ vs V0_prod (negative = regression on own metric)\n')
    variant_labels = [l for l in LABELS if l != BASELINE]
    header = '| Bucket | metric | ' + ' | '.join(variant_labels) + ' |'
    lines.append(header)
    lines.append('|' + '---|' * (len(variant_labels) + 2))
    for b in bucket_order:
        metric = BUCKET_METRIC[b]
        base = by[BASELINE][b]
        if not base:
            continue
        base_mean = np.mean(base)
        cells = []
        for lab in variant_labels:
            v = by[lab][b]
            d = (np.mean(v) - base_mean) if v else 0.0
            cells.append(f'{d:+.3f}')
        lines.append(f'| {b} | {metric} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # Catastrophic count per variant (vs V0, |Δ|>0.20 worse on own metric)
    lines.append('## Catastrophic regressions vs V0 (Δ < −0.20 on own metric)\n')
    lines.append('| Variant | n_catastrophic | worst Δ | worst case |')
    lines.append('|---|---:|---:|---|')
    for lab in variant_labels:
        worst_d = 0.0
        worst_stem = ''
        n_cat = 0
        for r in results:
            b = r['bucket']
            metric = BUCKET_METRIC[b]
            d = r[f'{lab}_{metric}'] - r[f'{BASELINE}_{metric}']
            if d < -0.20:
                n_cat += 1
            if d < worst_d:
                worst_d = d
                worst_stem = r['stem']
        lines.append(f'| {lab} | {n_cat} | {worst_d:+.3f} | `{worst_stem[:46]}` |')
    lines.append('')

    # nores LF (V0 vs V5) on FS_static
    lines.append('## nores LF (FS_static): V5_all_corrected vs V0_prod\n')
    nores = [(r['stem'], r['V0_prod_nores_lf_db'], r['V5_all_corrected_nores_lf_db'])
             for r in results
             if 'V0_prod_nores_lf_db' in r and 'V5_all_corrected_nores_lf_db' in r]
    if nores:
        d_arr = np.array([mf - m0 for _, m0, mf in nores])
        lines.append(f'N={len(nores)}  mean Δ={float(np.mean(d_arr)):+.2f} dB  '
                     f'max regression={float(np.max(d_arr)):+.2f} dB  '
                     f'(regressions Δ>+1dB: {int(np.sum(d_arr > 1.0))})\n')
        xjhi = '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'
        for s, m0, mf in nores:
            if s == xjhi:
                lines.append(f'9xjhi: V0={m0:.2f} dB  V5={mf:.2f} dB  Δ={mf-m0:+.2f} dB\n')
    else:
        lines.append('No nores data.\n')

    DOC_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\n[report] {DOC_PATH}')
    print('\n'.join(lines[:60]))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=9)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    cases = _discover_cases()
    print(f'[corpus] {len(cases)} cases; {len(LABELS)} configs each', flush=True)
    tasks = []
    for stem, subdir, bucket, wavtype, mic_path, ref_path in cases:
        do_nores = (bucket == 'FS_static')
        tasks.append((stem, subdir, bucket, wavtype, mic_path, ref_path, MODEL, do_nores))

    results = []
    done = 0
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_case_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                done += 1
                b = r['bucket']
                metric = BUCKET_METRIC[b]
                d5 = r[f'V5_all_corrected_{metric}'] - r[f'V0_prod_{metric}']
                print(f'  [{done:3d}/{total}] [{b:10s}] {r["stem"][:32]:32s} '
                      f'V0={r[f"V0_prod_{metric}"]:.3f} V5Δ={d5:+.4f}', flush=True)
            except Exception as exc:
                print(f'  [ERROR] {str(futs[fut][0])[:32]}: {exc}', flush=True)

    JSON_PATH.write_text(json.dumps({r['stem']: r for r in results}, indent=2),
                         encoding='utf-8')
    print(f'\n[json] {JSON_PATH}  ({len(results)} cases)')
    _write_report(results)


if __name__ == '__main__':
    run()
