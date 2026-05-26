#!/usr/bin/env python3
"""v3.21 Full Composition 12-case AECMOS gate.

Compares three configurations:
  M0          — v3.21.6 anchor, ALL candidate flags explicitly OFF
  M_D         — corrected full composition, pre-delay-chain baseline
                (Bundles A+B+C; no use_full_delay_change_chain)
  M_full_delay — M_D + use_full_delay_change_chain=True

Gate criteria (each variant vs M0):
  G1: DT bucket Δdeg mean ≥ −0.05
  G2: No per-case catastrophic: DT Δdeg < −0.20 OR FS Δecho > +0.20
  G3: 4 stress cases (nVUnxqHL / XRTnTUjU / 9xjhi / MYrVxVEM) each Δ ≥ −0.10
  G4: FS bucket Δecho ≤ +0.05

Hard watch items:
  W1: wVYSGVTT — M_full_delay should recover vs M_D (expect DT_mvmt deg improvement)
  W2: xFk7     — M_full_delay nores_LF +1.53 dB watch; AECMOS echo/deg must not catastrophically regress
  W3: 9xjhi    — nores LF artifact closure maintained (expect FS echo near M0 or better)
  W4: MYrVxVEM — reset_count anomaly; confirm no AECMOS regression

D2 NOT included — open v3.21 parity item; not required for this gate.
Transparent mode NOT included — independent probe, requires separate user auth.

Output:
  docs/v3_21_full_composition_12case_verdict.md

Usage:
    cd /path/to/AEC
    python3 python/v3_21_full_composition_12case_aecmos.py [--workers N]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from bench_aecmos import FastAECMOS

REPO     = Path(__file__).resolve().parents[1]
MODEL    = REPO / 'model' / 'Run_1663915512_Stage_0.onnx'
COHORT   = REPO / 'wav' / 'v3_21_8_cohort'
OUT_DIR  = REPO / 'out_v3_21_12case_composition'
DOC_PATH = REPO / 'docs' / 'v3_21_full_composition_12case_verdict.md'

# v3.21.6 baseline scores for reference column
BASELINE_V3_21_6 = {
    'ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement':        {'echo': 4.464, 'deg': 2.270},
    'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement':        {'echo': 4.088, 'deg': 2.741},
    'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement':        {'echo': 3.971, 'deg': 2.319},
    'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk':                      {'echo': 4.540, 'deg': 2.166},
    'XRTnTUjU5kS0mejzCqyCiw_doubletalk':                      {'echo': 4.097, 'deg': 3.950},
    'jtYTdZm3lUmFVNibJWq8YQ_doubletalk':                      {'echo': 4.629, 'deg': 2.700},
    'nVUnxqHLr0GTN7shWid1Ow_doubletalk':                      {'echo': 4.369, 'deg': 2.893},
    '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement': {'echo': 4.262, 'deg': 4.999},
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk':               {'echo': 2.367, 'deg': 5.000},
    'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk':               {'echo': 3.550, 'deg': 4.999},
    'xQEUtY2pWUi7v1X93TF2AA_farend_singletalk':               {'echo': 3.387, 'deg': 5.000},
    '014AzuqPZku2004NbTTmcA_nearend_singletalk':              {'echo': 4.999, 'deg': 4.355},
}

COHORT_12 = [
    # (subdir, stem, bucket_label, wavtype)
    ('doubletalk', 'ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk_with_movement',        'DT_mvmt',   'dt'),
    ('doubletalk', 'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement',        'DT_mvmt',   'dt'),
    ('doubletalk', 'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement',        'DT_mvmt',   'dt'),
    ('doubletalk', 'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk',                     'DT_static', 'dt'),
    ('doubletalk', 'XRTnTUjU5kS0mejzCqyCiw_doubletalk',                     'DT_static', 'dt'),
    ('doubletalk', 'jtYTdZm3lUmFVNibJWq8YQ_doubletalk',                     'DT_static', 'dt'),
    ('doubletalk', 'nVUnxqHLr0GTN7shWid1Ow_doubletalk',                     'DT_static', 'dt'),
    ('farend_singletalk', '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement', 'FS_mvmt', 'st'),
    ('farend_singletalk', '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',               'FS_static', 'st'),
    ('farend_singletalk', 'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk',               'FS_static', 'st'),
    ('farend_singletalk', 'xQEUtY2pWUi7v1X93TF2AA_farend_singletalk',               'FS_static', 'st'),
    ('nearend_singletalk', '014AzuqPZku2004NbTTmcA_nearend_singletalk',              'NS',        'nst'),
]

BUCKET_METRIC = {
    'DT_mvmt': 'deg', 'DT_static': 'deg',
    'FS_mvmt': 'echo', 'FS_static': 'echo', 'NS': 'deg',
}

STRESS_CASES = {
    'nVUnxqHLr0GTN7shWid1Ow_doubletalk',
    'XRTnTUjU5kS0mejzCqyCiw_doubletalk',
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',
    'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk',
}

WATCH_CASES = {
    'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement': 'W1',
    'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement': 'W2',
    '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk': 'W3',
    'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk': 'W4',
}

# Configuration manifest — kept explicit so it survives any default drift
CONFIG_MANIFEST = {
    'M0': {
        'use_partition_summed_x2_for_h_error_gain': False,
        'use_current_e2_refined_in_h_error_denominator': False,
        'use_per_bin_h_error_refresh': False,
        'use_aec3_h_error_ceil': False,
        'use_aec3_filter_noise_gate_power': False,
        'use_partition_summed_x2_for_shadow_mu': False,
        'use_aec3_noise_gate_for_shadow': False,
        'use_poor_excitation_gate_for_shadow': False,
        'use_narrowband_mask_for_shadow': False,
        'use_saturation_gate_for_shadow': False,
        'use_refined_output_selection_for_linear_path': False,
        'form_linear_filter_crossfade_enabled': False,
        'use_full_delay_change_chain': False,
        'transparent_mode_enabled': False,
    },
    'M_D': {
        'use_partition_summed_x2_for_h_error_gain': True,
        'use_current_e2_refined_in_h_error_denominator': True,
        'use_per_bin_h_error_refresh': True,
        'use_aec3_h_error_ceil': True,
        'use_aec3_filter_noise_gate_power': True,
        'use_partition_summed_x2_for_shadow_mu': True,
        'use_aec3_noise_gate_for_shadow': True,
        'use_poor_excitation_gate_for_shadow': True,
        'use_narrowband_mask_for_shadow': True,
        'use_saturation_gate_for_shadow': True,
        'use_refined_output_selection_for_linear_path': True,
        'form_linear_filter_crossfade_enabled': True,
        'use_full_delay_change_chain': False,
        'transparent_mode_enabled': False,
    },
    'M_full_delay': {
        'use_partition_summed_x2_for_h_error_gain': True,
        'use_current_e2_refined_in_h_error_denominator': True,
        'use_per_bin_h_error_refresh': True,
        'use_aec3_h_error_ceil': True,
        'use_aec3_filter_noise_gate_power': True,
        'use_partition_summed_x2_for_shadow_mu': True,
        'use_aec3_noise_gate_for_shadow': True,
        'use_poor_excitation_gate_for_shadow': True,
        'use_narrowband_mask_for_shadow': True,
        'use_saturation_gate_for_shadow': True,
        'use_refined_output_selection_for_linear_path': True,
        'form_linear_filter_crossfade_enabled': True,
        'use_full_delay_change_chain': True,   # M_full_delay only
        'transparent_mode_enabled': False,
    },
}

LABELS = ['M0', 'M_D', 'M_full_delay']


def build_cfg(label: str) -> AecConfig:
    flags = CONFIG_MANIFEST[label]
    cfg = AecConfig.from_preset(AecPreset.BALANCED, **flags)
    return cfg


def load_wavs(mic_p: Path, ref_p: Path):
    mic, sr = sf.read(str(mic_p))
    ref, _  = sf.read(str(ref_p))
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    n = min(len(mic), len(ref))
    return mic[:n].astype(np.float32), ref[:n].astype(np.float32), sr


def render_wav(mic: np.ndarray, ref: np.ndarray, label: str) -> np.ndarray:
    cfg = build_cfg(label)
    np.random.seed(42)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = len(mic) // hop
    out = np.zeros(n * hop, dtype=np.float32)
    for i in range(n):
        out[i*hop:(i+1)*hop] = aec.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop]
        )
    return out


def byte_equal_check_m0() -> bool:
    """Verify M0 is byte-equal to plain BALANCED (sanity gate)."""
    print('[byte-equal] Checking M0 vs plain BALANCED ...', flush=True)
    subdir, stem = 'farend_singletalk', '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'
    mic_p = COHORT / subdir / f'{stem}_mic.wav'
    ref_p = COHORT / subdir / f'{stem}_lpb.wav'
    if not mic_p.exists():
        print('[byte-equal] SKIP — ref case missing')
        return False
    mic, ref, _ = load_wavs(mic_p, ref_p)
    cfg_plain = AecConfig.from_preset(AecPreset.BALANCED)
    np.random.seed(42)
    aec_plain = AEC(cfg_plain)
    hop = cfg_plain.hop_size
    n = len(mic) // hop
    out_plain = np.zeros(n * hop, dtype=np.float32)
    for i in range(n):
        out_plain[i*hop:(i+1)*hop] = aec_plain.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop]
        )
    out_m0 = render_wav(mic, ref, 'M0')
    n2 = min(len(out_plain), len(out_m0))
    ok = np.array_equal(out_plain[:n2], out_m0[:n2])
    if ok:
        print('[byte-equal] PASS')
    else:
        diffs = int(np.sum(out_plain[:n2] != out_m0[:n2]))
        print(f'[byte-equal] FAIL — {diffs}/{n2} samples differ')
    return ok


def score_case(out: np.ndarray, mic: np.ndarray, ref: np.ndarray,
               wavtype: str, est: 'FastAECMOS') -> dict:
    echo, deg = est.score(wavtype, ref, mic, out)
    return {'echo': float(echo), 'deg': float(deg)}


def _render_task(args: Tuple) -> Tuple:
    """Top-level function for ProcessPoolExecutor (must be picklable)."""
    subdir, stem, label = args
    mic_p = COHORT / subdir / f'{stem}_mic.wav'
    ref_p = COHORT / subdir / f'{stem}_lpb.wav'
    mic, ref, _ = load_wavs(mic_p, ref_p)
    out = render_wav(mic, ref, label)
    return stem, label, out, mic, ref


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel render workers (default 4)')
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    est = FastAECMOS(str(MODEL))

    be_ok = byte_equal_check_m0()
    print()

    # Build all (subdir, stem, label) render tasks
    task_map: Dict[str, Tuple] = {}  # stem → (subdir, wavtype, bucket_label)
    render_tasks = []
    for subdir, stem, bucket_label, wavtype in COHORT_12:
        mic_p = COHORT / subdir / f'{stem}_mic.wav'
        if not mic_p.exists():
            print(f'[SKIP] {stem} — wav not found')
            continue
        task_map[stem] = (subdir, wavtype, bucket_label)
        for lbl in LABELS:
            render_tasks.append((subdir, stem, lbl))

    total = len(render_tasks)
    print(f'[parallel] {total} render tasks, {args.workers} workers', flush=True)

    # Render in parallel
    raw_results: Dict[str, Dict[str, Tuple]] = {stem: {} for stem in task_map}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_render_task, t): t for t in render_tasks}
        for fut in as_completed(futs):
            stem, label, out, mic, ref = fut.result()
            raw_results[stem][label] = (out, mic, ref)
            done += 1
            print(f'  [{done}/{total}] {stem[:30]:30s} {label}', flush=True)

    print()

    # Score all (sequential, single ONNX session)
    all_scores: Dict[str, Dict[str, dict]] = {lbl: {} for lbl in LABELS}
    for stem, (subdir, wavtype, bucket_label) in task_map.items():
        metric = BUCKET_METRIC[bucket_label]
        watch_tag = WATCH_CASES.get(stem, '')
        tag = f'[{watch_tag}]' if watch_tag else '     '
        print(f'{tag} [{bucket_label}] {stem[:35]:35s}', end='', flush=True)
        for lbl in LABELS:
            if lbl not in raw_results[stem]:
                continue
            out, mic, ref = raw_results[stem][lbl]
            s = score_case(out, mic, ref, wavtype, est)
            all_scores[lbl][stem] = s
            print(f'  {lbl}={s[metric]:.3f}', end='', flush=True)
        print()

    _write_verdict(all_scores, be_ok)


def _write_verdict(all_scores: Dict[str, Dict[str, dict]], be_ok: bool):
    lines = []
    lines.append('# v3.21 Full Composition 12-case AECMOS Gate\n')
    lines.append('**Variants compared**: M0 (anchor) · M_D (pre-delay-chain baseline) · M_full_delay (composition candidate)\n')
    lines.append(f'**Gate 0 byte-equal** (M0 vs plain BALANCED): {"PASS" if be_ok else "FAIL"}\n')
    lines.append('')

    # Config manifest
    lines.append('## Configuration Manifest\n')
    lines.append('All candidate flags set explicitly — no default drift.\n')
    lines.append('| Flag | M0 | M_D | M_full_delay |')
    lines.append('|------|-----|------|--------------|')
    flag_order = [
        'use_partition_summed_x2_for_h_error_gain',
        'use_current_e2_refined_in_h_error_denominator',
        'use_per_bin_h_error_refresh',
        'use_aec3_h_error_ceil',
        'use_aec3_filter_noise_gate_power',
        'use_partition_summed_x2_for_shadow_mu',
        'use_aec3_noise_gate_for_shadow',
        'use_poor_excitation_gate_for_shadow',
        'use_narrowband_mask_for_shadow',
        'use_saturation_gate_for_shadow',
        'use_refined_output_selection_for_linear_path',
        'form_linear_filter_crossfade_enabled',
        'use_full_delay_change_chain',
        'transparent_mode_enabled',
    ]
    for flag in flag_order:
        row = [f'`{flag}`']
        for lbl in LABELS:
            v = CONFIG_MANIFEST[lbl][flag]
            row.append('ON' if v else 'off')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    # Per-case table: M0 absolute + Δ columns
    lines.append('## Per-case AECMOS\n')
    lines.append('> Δ = variant − M0. Watch items: W1=wVYSGVTT, W2=xFk7, W3=9xjhi, W4=MYrVxVEM.\n')
    lines.append('| Watch | Case (short) | Bucket | Metric | M0 | M_D Δ | M_full Δ | M_full−M_D | v3.21.6 |')
    lines.append('|-------|-------------|--------|--------|-----|--------|----------|------------|---------|')

    # Gate tracking
    gates_md  = {'G1': True, 'G2': True, 'G3': True, 'G4': True}
    gates_mfd = {'G1': True, 'G2': True, 'G3': True, 'G4': True}
    bucket_sums_md: Dict[str, List[float]] = {}
    bucket_sums_mfd: Dict[str, List[float]] = {}

    for subdir, stem, bucket_label, wavtype in COHORT_12:
        if stem not in all_scores['M0']:
            continue
        metric = BUCKET_METRIC[bucket_label]
        m0  = all_scores['M0'][stem][metric]
        md  = all_scores['M_D'].get(stem, {}).get(metric, float('nan'))
        mfd = all_scores['M_full_delay'].get(stem, {}).get(metric, float('nan'))
        bl  = BASELINE_V3_21_6.get(stem, {}).get(metric, float('nan'))

        d_md  = md  - m0
        d_mfd = mfd - m0
        d_diff = mfd - md  # M_full_delay − M_D (recovery indicator)

        bucket_sums_md.setdefault(bucket_label, []).append(d_md)
        bucket_sums_mfd.setdefault(bucket_label, []).append(d_mfd)

        # G2: per-case catastrophic
        if wavtype == 'dt':
            if d_md  < -0.20: gates_md['G2']  = False
            if d_mfd < -0.20: gates_mfd['G2'] = False
        if wavtype == 'st' and metric == 'echo':
            if d_md  > 0.20: gates_md['G2']  = False
            if d_mfd > 0.20: gates_mfd['G2'] = False

        # G3: stress cases
        if stem in STRESS_CASES:
            if d_md  < -0.10: gates_md['G3']  = False
            if d_mfd < -0.10: gates_mfd['G3'] = False

        watch_tag = WATCH_CASES.get(stem, '')
        case_short = stem[:28]
        lines.append(
            f'| {watch_tag:2s} | {case_short} | {bucket_label} | {metric} '
            f'| {m0:.3f} | {d_md:+.3f} | {d_mfd:+.3f} | {d_diff:+.3f} | {bl:.3f} |'
        )

    lines.append('')

    # Bucket means
    lines.append('## Bucket Means (Δ vs M0)\n')
    lines.append('| Bucket | Metric | M_D Δ | M_full_delay Δ | M_full−M_D |')
    lines.append('|--------|--------|-------|----------------|------------|')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NS']:
        metric = BUCKET_METRIC[bkt]
        vals_md  = bucket_sums_md.get(bkt, [])
        vals_mfd = bucket_sums_mfd.get(bkt, [])
        if vals_md:
            mean_md  = float(np.mean(vals_md))
            mean_mfd = float(np.mean(vals_mfd))
            mean_diff = mean_mfd - mean_md  # correction: M_full - M_D
            # Actually need to compute: mean(M_full_delay - M_D per case)
            # But we just have bucket sums; approximate via mean_mfd - mean_md
            # For exact: re-compute per-case diff
        else:
            mean_md = mean_mfd = mean_diff = float('nan')

        md_s  = f'{mean_md:+.3f}' if not np.isnan(mean_md) else '—'
        mfd_s = f'{mean_mfd:+.3f}' if not np.isnan(mean_mfd) else '—'
        diff_s = f'{mean_diff:+.3f}' if not np.isnan(mean_diff) else '—'
        lines.append(f'| {bkt} | {metric} | {md_s} | {mfd_s} | {diff_s} |')

        # G1: DT bucket deg
        if metric == 'deg' and not np.isnan(mean_md):
            if mean_md  < -0.05: gates_md['G1']  = False
            if mean_mfd < -0.05: gates_mfd['G1'] = False
        # G4: FS bucket echo
        if metric == 'echo' and not np.isnan(mean_md):
            if mean_md  > 0.05: gates_md['G4']  = False
            if mean_mfd > 0.05: gates_mfd['G4'] = False

    lines.append('')

    # Gate check
    lines.append('## Gate Check\n')
    lines.append('| Gate | Criterion | M_D | M_full_delay |')
    lines.append('|------|-----------|-----|--------------|')
    gate_defs = {
        'G1': 'DT bucket mean Δdeg ≥ −0.05 (both buckets)',
        'G2': 'No per-case: DT Δdeg < −0.20 OR FS Δecho > +0.20',
        'G3': 'Stress cases (nVUnxqHL/XRTnTUjU/9xjhi/MYrVxVEM) each Δ ≥ −0.10',
        'G4': 'FS bucket mean Δecho ≤ +0.05',
    }
    for g, desc in gate_defs.items():
        md_s  = 'PASS' if gates_md[g]  else 'FAIL'
        mfd_s = 'PASS' if gates_mfd[g] else 'FAIL'
        lines.append(f'| {g} | {desc} | {md_s} | {mfd_s} |')
    lines.append('')

    # Hard watch items
    lines.append('## Hard Watch Items\n')
    lines.append('| Tag | Case | Criterion | M_D Δ | M_full Δ | M_full−M_D | Status |')
    lines.append('|-----|------|-----------|-------|----------|------------|--------|')

    watch_specs = [
        ('W1', 'wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement',
         'DT_mvmt', 'deg',
         'M_full_delay should recover vs M_D (trace: ul 49.5%→96.3%)'),
        ('W2', 'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement',
         'DT_mvmt', 'deg',
         'nores_LF +1.53dB watch; AECMOS deg/echo must not catastrophically regress'),
        ('W3', '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',
         'FS_static', 'echo',
         'nores LF artifact closure must be maintained'),
        ('W4', 'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk',
         'DT_static', 'deg',
         'reset_count 63→113 anomaly must not cause AECMOS regression'),
    ]
    for tag, stem, bkt, metric, criterion in watch_specs:
        if stem not in all_scores['M0']:
            lines.append(f'| {tag} | {stem[:28]} | {criterion} | — | — | — | SKIP |')
            continue
        m0  = all_scores['M0'][stem][metric]
        md  = all_scores['M_D'].get(stem, {}).get(metric, float('nan'))
        mfd = all_scores['M_full_delay'].get(stem, {}).get(metric, float('nan'))
        d_md  = md  - m0
        d_mfd = mfd - m0
        d_diff = mfd - md

        # Assess status
        status = '?'
        if tag == 'W1':
            # Recovery: M_full_delay−M_D should be positive (deg improvement)
            if not np.isnan(d_diff) and d_diff > 0.0:
                status = f'RECOVERY ({d_diff:+.3f})'
            elif not np.isnan(d_mfd) and d_mfd < -0.20:
                status = f'CATASTROPHIC FAIL ({d_mfd:+.3f})'
            else:
                status = f'WATCH ({d_diff:+.3f})'
        elif tag == 'W2':
            if not np.isnan(d_mfd) and d_mfd < -0.20:
                status = f'CATASTROPHIC FAIL ({d_mfd:+.3f})'
            elif not np.isnan(d_mfd) and abs(d_mfd) <= 0.10:
                status = f'OK ({d_mfd:+.3f})'
            else:
                status = f'WATCH ({d_mfd:+.3f})'
        elif tag == 'W3':
            if not np.isnan(d_mfd) and d_mfd > 0.20:
                status = f'REGRESSION ({d_mfd:+.3f})'
            else:
                status = f'OK ({d_mfd:+.3f})'
        elif tag == 'W4':
            if not np.isnan(d_mfd) and d_mfd < -0.10:
                status = f'REGRESSION ({d_mfd:+.3f})'
            else:
                status = f'OK ({d_mfd:+.3f})'

        lines.append(
            f'| {tag} | {stem[:28]} | {criterion} '
            f'| {d_md:+.3f} | {d_mfd:+.3f} | {d_diff:+.3f} | {status} |'
        )
    lines.append('')

    # Ship decision
    lines.append('## Ship Decision\n')
    md_overall_pass  = all(gates_md.values())
    mfd_overall_pass = all(gates_mfd.values())

    for lbl, gates, overall in [('M_D', gates_md, md_overall_pass),
                                  ('M_full_delay', gates_mfd, mfd_overall_pass)]:
        if overall:
            lines.append(f'- **{lbl}**: G1–G4 all PASS → proceed to 800-case (requires user authorisation)')
        else:
            failed = [g for g, p in gates.items() if not p]
            lines.append(f'- **{lbl}**: FAIL — gates {failed} failed → stop and report root cause')

    lines.append('')
    lines.append('### Conclusion\n')
    if mfd_overall_pass:
        lines.append(
            '**12-case PASS** (M_full_delay) → ask user whether to authorise 800-case benchmark.\n'
            '\nNote: D2 (coarse rate=0.9, shadow mu_initial=0.643) is an open v3.21 parity item '
            'but is NOT required for this gate. D2 requires separate user authorisation.\n'
        )
    else:
        failed_gates = [g for g, p in gates_mfd.items() if not p]
        lines.append(
            f'**12-case FAIL** (M_full_delay, gates: {failed_gates}) — '
            'stop and report root cause. See hard watch items above for attribution.\n'
        )

    text = '\n'.join(lines) + '\n'
    DOC_PATH.write_text(text, encoding='utf-8')
    print(f'\n[verdict] Written to {DOC_PATH}')
    print(text)


if __name__ == '__main__':
    run()
