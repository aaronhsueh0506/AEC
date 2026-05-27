#!/usr/bin/env python3
"""v3.21 poor_coarse rescue copy (Gap A+B+C+D+E) 12-case AECMOS gate.

Compares two configurations on the v3.21 12-case cohort:
  M_full_delay   — full Bundle A+B+C+D + delay chain (current composition candidate)
  M_full_rescue  — M_full_delay + use_aec3_poor_coarse_rescue_copy=True
                   (AEC3-exact rescue: strict cond, 2-hop threshold, 10-hop hangover
                    without shadow freeze, Gap C E_refined override on fire hops)

Gate criteria (M_full_rescue vs M_full_delay):
  Goal of this gate is to confirm Gap C does not regress against the established
  M_full_delay composition. Same 4 gates as the prior 12-case verdict, evaluated
  against M0 anchor:
    G1: DT bucket Δdeg mean ≥ −0.05 vs M0
    G2: No per-case catastrophic: DT Δdeg < −0.20 OR FS Δecho > +0.20 (vs M0)
    G3: 4 stress cases (nVUnxqHL / XRTnTUjU / 9xjhi / MYrVxVEM) each Δ ≥ −0.10
    G4: FS bucket Δecho ≤ +0.05 vs M0

  Additional rescue-specific check:
    R1: M_full_rescue − M_full_delay per-case Δ within ±0.15 on each metric
        (delta against the established composition; not allowed to silently shift)
    R2: 9xjhi FS_static Δecho vs M_full_delay should be ≥ −0.05 (closing extra gap to AEC3)
    R3: 0I0XMl3M FS_mvmt Δecho vs M_full_delay should not regress > +0.10
        (Gap C closed nores LF +1.4 dB regression that occurred without it on Gate0)

Hard watch items inherited from prior verdict:
  W1: wVYSGVTT — M_full_delay should recover vs M_D
  W2: xFk7     — M_full_delay nores_LF +1.53 dB watch
  W3: 9xjhi    — nores LF artifact closure maintained
  W4: MYrVxVEM — reset_count anomaly

Output:
  docs/v3_21_poor_coarse_rescue_12case_verdict.md

Usage:
    cd /path/to/AEC
    python3 python/v3_21_poor_coarse_rescue_12case_aecmos.py [--workers N]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from bench_aecmos import FastAECMOS

REPO     = Path(__file__).resolve().parents[1]
MODEL    = REPO / 'model' / 'Run_1663915512_Stage_0.onnx'
COHORT   = REPO / 'wav' / 'v3_21_8_cohort'
OUT_DIR  = REPO / 'out_v3_21_12case_rescue'
DOC_PATH = REPO / 'docs' / 'v3_21_poor_coarse_rescue_12case_verdict.md'

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

_FULL_DELAY_FLAGS = {
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
    'use_full_delay_change_chain': True,
    'transparent_mode_enabled': False,
}

CONFIG_MANIFEST = {
    'M0': {**{k: False for k in _FULL_DELAY_FLAGS},
           'use_aec3_poor_coarse_rescue_copy': False},
    'M_full_delay': {**_FULL_DELAY_FLAGS,
                     'use_aec3_poor_coarse_rescue_copy': False},
    'M_full_rescue': {**_FULL_DELAY_FLAGS,
                      'use_aec3_poor_coarse_rescue_copy': True},
}

LABELS = ['M0', 'M_full_delay', 'M_full_rescue']


def build_cfg(label: str) -> AecConfig:
    flags = CONFIG_MANIFEST[label]
    return AecConfig.from_preset(AecPreset.BALANCED, **flags)


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
    subdir, stem, label = args
    mic_p = COHORT / subdir / f'{stem}_mic.wav'
    ref_p = COHORT / subdir / f'{stem}_lpb.wav'
    mic, ref, _ = load_wavs(mic_p, ref_p)
    out = render_wav(mic, ref, label)
    return stem, label, out, mic, ref


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    est = FastAECMOS(str(MODEL))

    be_ok = byte_equal_check_m0()
    print()

    task_map: Dict[str, Tuple] = {}
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
    lines.append('# v3.21 poor_coarse rescue (Gap A+B+C+D+E) 12-case AECMOS Verdict\n')
    lines.append('**Variants compared**: M0 (anchor) · M_full_delay (current candidate) · M_full_rescue (M_full_delay + use_aec3_poor_coarse_rescue_copy=True; Gap C E_refined override included)\n')
    lines.append(f'**Gate 0 byte-equal** (M0 vs plain BALANCED): {"PASS" if be_ok else "FAIL"}\n')
    lines.append('')

    # Per-case table
    lines.append('## Per-case AECMOS\n')
    lines.append('> Δ_full = M_full_delay − M0. Δ_rescue = M_full_rescue − M0. Δ_R−FD = M_full_rescue − M_full_delay.\n')
    lines.append('| Watch | Case (short) | Bucket | Metric | M0 | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) | v3.21.6 |')
    lines.append('|-------|-------------|--------|--------|-----|----------------|-----------------|---------|---------|')

    gates_fd = {'G1': True, 'G2': True, 'G3': True, 'G4': True}
    gates_rc = {'G1': True, 'G2': True, 'G3': True, 'G4': True}
    rescue_gates = {'R1': True, 'R2': True, 'R3': True}
    bucket_sums_fd: Dict[str, List[float]] = {}
    bucket_sums_rc: Dict[str, List[float]] = {}

    for subdir, stem, bucket_label, wavtype in COHORT_12:
        if stem not in all_scores['M0']:
            continue
        metric = BUCKET_METRIC[bucket_label]
        m0  = all_scores['M0'][stem][metric]
        mfd = all_scores['M_full_delay'].get(stem, {}).get(metric, float('nan'))
        mrc = all_scores['M_full_rescue'].get(stem, {}).get(metric, float('nan'))
        bl  = BASELINE_V3_21_6.get(stem, {}).get(metric, float('nan'))

        d_fd = mfd - m0
        d_rc = mrc - m0
        d_diff = mrc - mfd

        bucket_sums_fd.setdefault(bucket_label, []).append(d_fd)
        bucket_sums_rc.setdefault(bucket_label, []).append(d_rc)

        # G2 (per-case catastrophic vs M0)
        if wavtype == 'dt':
            if d_fd < -0.20: gates_fd['G2'] = False
            if d_rc < -0.20: gates_rc['G2'] = False
        if wavtype == 'st' and metric == 'echo':
            if d_fd > 0.20: gates_fd['G2'] = False
            if d_rc > 0.20: gates_rc['G2'] = False
        # G3 stress
        if stem in STRESS_CASES:
            if d_fd < -0.10: gates_fd['G3'] = False
            if d_rc < -0.10: gates_rc['G3'] = False
        # R1: rescue vs M_full_delay within ±0.15
        if not np.isnan(d_diff) and abs(d_diff) > 0.15:
            rescue_gates['R1'] = False
        # R2: 9xjhi Δecho vs M_full_delay ≥ -0.05
        if stem == '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk' and metric == 'echo':
            if not np.isnan(d_diff) and d_diff < -0.05:
                rescue_gates['R2'] = False
        # R3: 0I0XMl3M Δecho vs M_full_delay ≤ +0.10
        if stem == '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement' and metric == 'echo':
            if not np.isnan(d_diff) and d_diff > 0.10:
                rescue_gates['R3'] = False

        watch_tag = WATCH_CASES.get(stem, '')
        case_short = stem[:28]
        lines.append(
            f'| {watch_tag:2s} | {case_short} | {bucket_label} | {metric} '
            f'| {m0:.3f} | {d_fd:+.3f} | {d_rc:+.3f} | {d_diff:+.3f} | {bl:.3f} |'
        )

    lines.append('')

    # Bucket means
    lines.append('## Bucket Means (Δ vs M0)\n')
    lines.append('| Bucket | Metric | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) |')
    lines.append('|--------|--------|----------------|-----------------|---------|')
    for bkt in ['DT_mvmt', 'DT_static', 'FS_mvmt', 'FS_static', 'NS']:
        metric = BUCKET_METRIC[bkt]
        vals_fd = bucket_sums_fd.get(bkt, [])
        vals_rc = bucket_sums_rc.get(bkt, [])
        if vals_fd:
            mean_fd = float(np.mean(vals_fd))
            mean_rc = float(np.mean(vals_rc))
            diff = mean_rc - mean_fd
        else:
            mean_fd = mean_rc = diff = float('nan')

        fd_s = f'{mean_fd:+.3f}' if not np.isnan(mean_fd) else '—'
        rc_s = f'{mean_rc:+.3f}' if not np.isnan(mean_rc) else '—'
        diff_s = f'{diff:+.3f}' if not np.isnan(diff) else '—'
        lines.append(f'| {bkt} | {metric} | {fd_s} | {rc_s} | {diff_s} |')

        if metric == 'deg' and not np.isnan(mean_fd):
            if mean_fd < -0.05: gates_fd['G1'] = False
            if mean_rc < -0.05: gates_rc['G1'] = False
        if metric == 'echo' and not np.isnan(mean_fd):
            if mean_fd > 0.05: gates_fd['G4'] = False
            if mean_rc > 0.05: gates_rc['G4'] = False

    lines.append('')

    # Gate check
    lines.append('## Gate Check\n')
    lines.append('### Composition gates (vs M0)\n')
    lines.append('| Gate | Criterion | M_full_delay | M_full_rescue |')
    lines.append('|------|-----------|--------------|---------------|')
    gate_defs = {
        'G1': 'DT bucket mean Δdeg ≥ −0.05',
        'G2': 'No per-case: DT Δdeg < −0.20 OR FS Δecho > +0.20',
        'G3': 'Stress cases each Δ ≥ −0.10',
        'G4': 'FS bucket mean Δecho ≤ +0.05',
    }
    for g, desc in gate_defs.items():
        fd_s = 'PASS' if gates_fd[g] else 'FAIL'
        rc_s = 'PASS' if gates_rc[g] else 'FAIL'
        lines.append(f'| {g} | {desc} | {fd_s} | {rc_s} |')
    lines.append('')

    lines.append('### Rescue-specific gates (M_full_rescue vs M_full_delay)\n')
    lines.append('| Gate | Criterion | Result |')
    lines.append('|------|-----------|--------|')
    rgate_defs = {
        'R1': 'All per-case Δ(R−FD) within ±0.15',
        'R2': '9xjhi FS_static Δecho vs M_full_delay ≥ −0.05 (Cat3 closure)',
        'R3': '0I0XMl3M FS_mvmt Δecho vs M_full_delay ≤ +0.10 (Gate0 carryover)',
    }
    for g, desc in rgate_defs.items():
        s = 'PASS' if rescue_gates[g] else 'FAIL'
        lines.append(f'| {g} | {desc} | {s} |')
    lines.append('')

    # Watch items
    lines.append('## Hard Watch Items\n')
    lines.append('| Tag | Case | Criterion | M_full_delay Δ | M_full_rescue Δ | Δ(R−FD) |')
    lines.append('|-----|------|-----------|----------------|-----------------|---------|')
    for tag in ['W1', 'W2', 'W3', 'W4']:
        stem = next((s for s, t in WATCH_CASES.items() if t == tag), None)
        if not stem or stem not in all_scores['M0']:
            lines.append(f'| {tag} | — | — | — | — | — |')
            continue
        bkt = next((b for sd, st, b, _ in COHORT_12 if st == stem), 'DT_mvmt')
        metric = BUCKET_METRIC[bkt]
        m0  = all_scores['M0'][stem][metric]
        mfd = all_scores['M_full_delay'].get(stem, {}).get(metric, float('nan'))
        mrc = all_scores['M_full_rescue'].get(stem, {}).get(metric, float('nan'))
        d_fd = mfd - m0
        d_rc = mrc - m0
        d_diff = mrc - mfd
        lines.append(
            f'| {tag} | {stem[:32]} | {bkt} {metric} '
            f'| {d_fd:+.3f} | {d_rc:+.3f} | {d_diff:+.3f} |'
        )
    lines.append('')

    # Ship decision
    lines.append('## Ship Decision\n')
    fd_overall = all(gates_fd.values())
    rc_overall = all(gates_rc.values())
    rescue_overall = all(rescue_gates.values())

    lines.append(f'- **M_full_delay** (baseline): G1–G4 {"PASS" if fd_overall else "FAIL"}'
                 f' (gates: {[g for g,p in gates_fd.items() if not p] or "all"})')
    lines.append(f'- **M_full_rescue**: G1–G4 {"PASS" if rc_overall else "FAIL"}'
                 f' (gates: {[g for g,p in gates_rc.items() if not p] or "all"})')
    lines.append(f'- **Rescue-specific R1–R3**: {"PASS" if rescue_overall else "FAIL"}'
                 f' (gates: {[g for g,p in rescue_gates.items() if not p] or "all"})')
    lines.append('')

    lines.append('### Conclusion\n')
    if rc_overall and rescue_overall:
        lines.append(
            '**12-case PASS** (M_full_rescue) → ask user whether to authorise 800-case benchmark.\n'
            '\nGap C wiring: shadow.process(defer_update=True) + complete_update('
            'error_override=self.filter.error_spec) on copy_fired hops. '
            'Off-fire hops complete_update(None) preserves the inline E_coarse path. '
            'AEC3 subtractor.cc:302-304 parity.\n'
        )
    elif rc_overall and not rescue_overall:
        failed = [g for g, p in rescue_gates.items() if not p]
        lines.append(
            f'**RESCUE-SPECIFIC FAIL** (R-gates: {failed}) — '
            'composition gates pass but rescue introduces regression vs M_full_delay. '
            'Investigate per-case Δ(R−FD) before ship.\n'
        )
    else:
        failed = [g for g, p in gates_rc.items() if not p]
        lines.append(
            f'**COMPOSITION FAIL** (M_full_rescue, gates: {failed}) — '
            'stop and report root cause. See per-case table.\n'
        )

    DOC_PATH.parent.mkdir(exist_ok=True)
    DOC_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print()
    print(f'[doc] wrote {DOC_PATH}')
    print()
    print('=' * 72)
    print(f'M_full_delay  composition: G1-G4 {"PASS" if fd_overall else "FAIL"}')
    print(f'M_full_rescue composition: G1-G4 {"PASS" if rc_overall else "FAIL"}')
    print(f'Rescue-specific R1-R3   : {"PASS" if rescue_overall else "FAIL"}')
    print('=' * 72)


if __name__ == '__main__':
    run()
