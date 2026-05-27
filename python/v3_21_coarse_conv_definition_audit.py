#!/usr/bin/env python3
"""Coarse_conv definition audit — answer "why is coarse_conv = 0%?"

ROOT CAUSE: my prior attribution script's `coarse_conv = (e2_coa < 0.5*y2)
AND (y2 > 50*50*hop)` is doubly wrong vs AEC3
(subtractor_output_analyzer.cc:43-51):

  Bug 1 — int16² → float² scale conversion MISSING:
    AEC3:    kConvergenceThreshold = 50*50*kBlockSize = 160000 (int16² scale)
    Wrong:   thr = 50*50*160 = 400000 (raw — no scale conversion)
    Correct: thr = (50² * hop) / 32768² ≈ 3.73e-4 (float-domain per-hop y²)
    The wrong threshold is ~1e9 too large; in float [-1,1] scale our y2
    rarely exceeds 1e-1, so `y2 > 400000` is ALWAYS false → gate never
    fires → coarse_conv = 0% regardless of the actual e2/y2 ratio.

  Bug 2 — wrong ratio:
    refined_converged uses 0.5; coarse strict uses 0.05; coarse relaxed
    uses 0.3. I used 0.5 (refined's bar).

CORRECT CONVERSION FORMULA (matches URO code at orchestrator.py:4085-4087
which already does this for thr_30 / thr_60):

    int16_scale_sq = 32768.0 ** 2
    y2_thr_strict  = (50 ** 2) * hop / int16_scale_sq   # ≈ 3.73e-4
    y2_thr_relaxed = (20 ** 2) * hop / int16_scale_sq   # ≈ 5.96e-5

    coarse_conv_strict  = (e2_coa < 0.05 * y2) AND (y2 > y2_thr_strict)
    coarse_conv_relaxed = (e2_coa < 0.30 * y2) AND (y2 > y2_thr_relaxed)

This script dumps the per-frame e2_coarse/y2 ratio distribution under
both M_full_delay and M_full_rescue on 9xjhi (FS_static — best
convergence opportunity) + MYrVxVEM (DT_static — sustained DT contrast).
Output: docs/v3_21_coarse_conv_definition_audit.md.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset

REPO   = Path(__file__).resolve().parents[1]
COHORT = REPO / 'wav' / 'v3_21_8_cohort'
DOC    = REPO / 'docs' / 'v3_21_coarse_conv_definition_audit.md'

CASES = {
    '9xjhi':    {'subdir': 'farend_singletalk',
                  'stem':   '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk'},
    'MYrVxVEM': {'subdir': 'doubletalk',
                  'stem':   'MYrVxVEMxkaE7OuyTUmI0Q_doubletalk'},
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
}

VARIANTS = {
    'M_full_delay':  {**_FULL_DELAY_FLAGS,
                       'use_aec3_poor_coarse_rescue_copy': False},
    'M_full_rescue': {**_FULL_DELAY_FLAGS,
                       'use_aec3_poor_coarse_rescue_copy': True},
}


def load_wavs(subdir: str, stem: str):
    mic_p = COHORT / subdir / f'{stem}_mic.wav'
    ref_p = COHORT / subdir / f'{stem}_lpb.wav'
    mic, sr = sf.read(str(mic_p))
    ref, _  = sf.read(str(ref_p))
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    n = min(len(mic), len(ref))
    return mic[:n].astype(np.float32), ref[:n].astype(np.float32), int(sr)


def run(case_key: str, vk: str, vflags: Dict) -> Dict:
    case = CASES[case_key]
    mic, ref, sr = load_wavs(case['subdir'], case['stem'])
    cfg = AecConfig.from_preset(AecPreset.BALANCED, **vflags)
    cfg.trace_hf_chain = True
    np.random.seed(42)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = len(mic) // hop
    out = np.zeros(n * hop, dtype=np.float32)
    for i in range(n):
        out[i*hop:(i+1)*hop] = aec.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop]
        )
    trace = aec._hf_chain_trace
    y2     = np.array([r.get('uro_y2', 0.0) for r in trace], dtype=np.float64)
    e2_ref = np.array([r.get('uro_e2_refined', 0.0) for r in trace], dtype=np.float64)
    e2_coa = np.array([r.get('uro_e2_coarse', 0.0) for r in trace], dtype=np.float64)
    return {'y2': y2, 'e2_ref': e2_ref, 'e2_coa': e2_coa, 'hop': hop, 'n_frames': len(trace)}


def main():
    lines = []
    lines.append('# coarse_conv = 0% — definition audit\n')
    lines.append('**Question**: why does the prior attribution show `coarse_conv = 0%` '
                  'in every case × variant? Was it a metric bug or true non-convergence?\n')
    lines.append('## Answer (one line)\n')
    lines.append('**The 0% was a metric BUG** (wrong scale conversion); the actual '
                  'shadow PBFDAF convergence rate vs AEC3-correct bars is reported below.\n')
    lines.append('')
    lines.append('## The bug\n')
    lines.append('Prior attribution script used:\n')
    lines.append('```python\n')
    lines.append('y2_thr = 50 * 50 * hop           # = 50²×160 = 400 000  ← raw int16² number\n')
    lines.append('coarse_conv = (e2_coa < 0.5 * y2) & (y2 > y2_thr)  ← 0.5 is REFINED ratio\n')
    lines.append('```\n')
    lines.append('Both wrong:\n')
    lines.append('  - **Bug 1 (fatal)**: `y2_thr = 400000` lives in AEC3 int16² scale. '
                  'Our `uro_y2` is `sum(near_block**2)` in **float [-1,1]** scale — its '
                  'typical magnitude is 1e-4 to 1e-1. `y2 > 400000` is **always False** → '
                  'gate never fires → 0% regardless of any ratio.\n')
    lines.append('  - **Bug 2**: `0.5` is the AEC3 REFINED bar; coarse uses 0.05 (strict) '
                  'or 0.3 (relaxed).\n')
    lines.append('')
    lines.append('## AEC3 canonical thresholds + the correct conversion for hop=160\n')
    lines.append('AEC3 source: `subtractor_output_analyzer.cc:43-51`. AEC3 native: '
                  '`kBlockSize = 64`, samples in int16 scale (±32768).\n')
    lines.append('')
    lines.append('| predicate | AEC3 ratio | AEC3 y2 floor (int16² × kBlockSize) | Our float-scale y2 floor (hop=160) |')
    lines.append('|-----------|------------|--------------------------------------|--------------------------------------|')
    lines.append('| refined_converged | `e2_ref < 0.5 × y2` | `50² × 64 = 160 000` | `50² × 160 / 32768² ≈ 3.73e-4` |')
    lines.append('| coarse_converged_STRICT | `e2_coa < 0.05 × y2` | `50² × 64 = 160 000` | `50² × 160 / 32768² ≈ 3.73e-4` |')
    lines.append('| coarse_converged_RELAXED | `e2_coa < 0.3 × y2` | `20² × 64 = 25 600` | `20² × 160 / 32768² ≈ 5.96e-5` |')
    lines.append('')
    lines.append('**Conversion formula** (matches the URO thr_30 / thr_60 computation at '
                  '`orchestrator.py:4085-4087`):\n')
    lines.append('```python\n')
    lines.append('# Preserves AEC3 per-sample RMS semantic; rescales to our hop & float scale\n')
    lines.append('int16_scale_sq = 32768.0 ** 2\n')
    lines.append('y2_thr_strict  = (50 ** 2) * hop / int16_scale_sq   # ≈ 3.73e-4 @ hop=160\n')
    lines.append('y2_thr_relaxed = (20 ** 2) * hop / int16_scale_sq   # ≈ 5.96e-5 @ hop=160\n')
    lines.append('```\n')
    lines.append('Semantic: AEC3 `kConvergenceThreshold = 50² × kBlockSize` enforces '
                  '"average per-sample y² > 50² (int16)" over the AEC3 block window. '
                  'Translating to our pipeline:\n')
    lines.append('  1. **scale**: int16² → float² ⇒ divide by `32768² ≈ 1.07e9`\n')
    lines.append('  2. **sample count**: AEC3 sums over `kBlockSize = 64`; we sum over '
                  '`hop = 160` ⇒ multiply by `160 / 64 = 2.5` (i.e. use `hop` directly in '
                  'the formula).\n')
    lines.append('  3. **ratio**: scale-invariant; use AEC3 0.05/0.3/0.5 unchanged.\n')
    lines.append('')

    print('=' * 80)
    print('coarse_conv definition audit — e2_coarse / y2 ratio distribution')
    print('=' * 80)
    print()

    all_results: Dict[str, Dict[str, Dict]] = {}
    for ck in CASES:
        all_results[ck] = {}
        for vk, vflags in VARIANTS.items():
            print(f'  {ck} × {vk} ...', flush=True)
            all_results[ck][vk] = run(ck, vk, vflags)

    # Per-case ratio + AEC3-correct gate distribution
    for ck in CASES:
        lines.append(f'## {ck}\n')
        lines.append('Gate fields:\n')
        lines.append('  - **r_med** / **r_p25** = e2_coarse / y2 ratio (median / 25th percentile)\n')
        lines.append('  - **STRICT** (`r<0.05 AND y2>3.73e-4`) = AEC3 `coarse_filter_converged_strict`\n')
        lines.append('  - **RELAXED** (`r<0.3 AND y2>5.96e-5`) = AEC3 `coarse_filter_converged_relaxed`\n')
        lines.append('  - **REFINED bar** (`r<0.5 AND y2>3.73e-4`) shown for reference (this is the REFINED bar, not coarse)\n')
        lines.append('  - **y2_gate_pass_strict** = % of frames passing y2 floor only (independent of ratio)\n')
        lines.append('')
        lines.append('| variant | n | y2_med | y2_gate_pass_strict | r_p25 | r_med | STRICT (0.05) | RELAXED (0.3) | REFINED bar (0.5) |')
        lines.append('|---------|---|--------|---------------------|-------|-------|---------------|---------------|--------------------|')
        print(f'\n=== {ck} ===')
        for vk in VARIANTS:
            r = all_results[ck][vk]
            y2 = r['y2']; e2_coa = r['e2_coa']; hop = r['hop']
            int16_sq = 32768.0 ** 2
            y2_thr_strict  = (50.0 ** 2) * hop / int16_sq
            y2_thr_relaxed = (20.0 ** 2) * hop / int16_sq
            valid = y2 > 1e-12
            ratio = np.where(valid, e2_coa / np.maximum(y2, 1e-12), np.nan)
            y2_gate_strict  = y2 > y2_thr_strict
            y2_gate_relaxed = y2 > y2_thr_relaxed
            conv_strict   = (ratio < 0.05) & y2_gate_strict
            conv_relaxed  = (ratio < 0.30) & y2_gate_relaxed
            conv_refbar   = (ratio < 0.50) & y2_gate_strict
            ratio_valid = ratio[~np.isnan(ratio)]
            r_p25 = float(np.percentile(ratio_valid, 25)) if len(ratio_valid) else float('nan')
            r_med = float(np.percentile(ratio_valid, 50)) if len(ratio_valid) else float('nan')
            y2_med = float(np.median(y2[valid])) if valid.any() else float('nan')
            f_y2_pass = float(np.mean(y2_gate_strict))
            f_strict = float(np.mean(conv_strict))
            f_relax  = float(np.mean(conv_relaxed))
            f_refbar = float(np.mean(conv_refbar))
            n_total = len(y2)
            print(f'  {vk:14s}  n={n_total:4d}  y2_med={y2_med:.3e}  y2_pass={f_y2_pass:.1%}  '
                  f'r[p25/med]={r_p25:.3f}/{r_med:.3f}  '
                  f'STRICT={f_strict:.1%}  RELAXED={f_relax:.1%}  REF={f_refbar:.1%}')
            lines.append(
                f'| {vk} | {n_total} | {y2_med:.3e} | {f_y2_pass:.1%} '
                f'| {r_p25:.3f} | {r_med:.3f} | {f_strict:.1%} | {f_relax:.1%} | {f_refbar:.1%} |'
            )
        lines.append('')

    # Verdict
    lines.append('## Verdict\n')
    # Use 9xjhi M_full_delay as the cleanest convergence case
    r = all_results['9xjhi']['M_full_delay']
    y2 = r['y2']; e2_coa = r['e2_coa']
    valid = y2 > 1e-12
    ratio_valid = e2_coa[valid] / y2[valid]
    frac_lt_005 = float(np.mean(ratio_valid < 0.05))
    frac_lt_030 = float(np.mean(ratio_valid < 0.30))
    frac_lt_050 = float(np.mean(ratio_valid < 0.50))

    if frac_lt_005 > 0.02 or frac_lt_030 > 0.10:
        lines.append('**Cause = (a) metric mis-definition** (primary).\n')
        lines.append(f'9xjhi M_full_delay e2_coarse/y2 ratio satisfies:\n')
        lines.append(f'  - AEC3 STRICT (r < 0.05): {frac_lt_005:.1%} of frames\n')
        lines.append(f'  - AEC3 RELAXED (r < 0.3): {frac_lt_030:.1%} of frames\n')
        lines.append(f'  - my prior wrong (r < 0.5): {frac_lt_050:.1%} of frames\n')
        lines.append('Shadow DOES converge by AEC3 RELAXED bar; the prior "0%" reading '
                      'was an artifact of the wrong ratio + wrong y2 floor.\n')
    else:
        lines.append('**Cause = (b) algorithm non-convergence** (primary).\n')
        lines.append(f'9xjhi M_full_delay e2_coarse/y2 ratio rarely drops below either bar:\n')
        lines.append(f'  - AEC3 STRICT (r < 0.05): {frac_lt_005:.1%} of frames\n')
        lines.append(f'  - AEC3 RELAXED (r < 0.3): {frac_lt_030:.1%} of frames\n')
        lines.append(f'  - my prior wrong (r < 0.5): {frac_lt_050:.1%} of frames\n')
        lines.append('Shadow PBFDAF does not satisfy AEC3 convergence under any reasonable '
                      'ratio bar. The "0%" reading was real; the wrong metric just happened '
                      'to give the same answer.\n')
    lines.append('')
    lines.append('### Implications for the §G4 / rescue verdict\n')
    if frac_lt_005 > 0.02 or frac_lt_030 > 0.10:
        lines.append('- The "shadow never converges" framing in the rescue attribution doc '
                      'was based on a mis-calibrated metric.\n')
        lines.append('- Re-state: shadow DOES achieve AEC3 RELAXED convergence but at a '
                      'lower rate than refined. The rescue mechanism still has the same '
                      'AECMOS trade-off (xFk7 +0.195 / MYrVxVEM −0.431 / etc.) because the '
                      'URO routing + usable_linear consumer pathway is the dominant driver, '
                      'not the absolute convergence bar.\n')
        lines.append('- The 9xjhi Cat3 architectural-ceiling framing still holds: rescue '
                      'copies do not improve the COARSE path quality enough to close the '
                      '1.088 dB extra echo gap vs AEC3 reference. The mechanism description '
                      'in attribution doc §I.3 should note "shadow does meet AEC3 RELAXED '
                      'bar but does not catch up to refined when refined is genuinely '
                      'better" rather than "shadow never converges at all".\n')
    else:
        lines.append('- The "shadow never converges" framing stands. Both the strict and '
                      'relaxed AEC3 convergence bars are unmet across the cohort.\n')
        lines.append('- The 9xjhi Cat3 architectural-ceiling verdict is reinforced: even with '
                      'rescue + Gap C, shadow does not reach AEC3-grade convergence.\n')
    lines.append('')
    lines.append('Note: this audit does not change the NO-SHIP verdict for '
                  '`use_aec3_poor_coarse_rescue_copy`. The 12-case AECMOS Pareto FAIL '
                  '(MYrVxVEM −0.431 / qNvSMyUS −0.216) is independent of how we define '
                  'coarse_conv — the AECMOS scores were measured on actual audio output.\n')

    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text('\n'.join(lines), encoding='utf-8')
    print()
    print(f'[doc] wrote {DOC}')


if __name__ == '__main__':
    main()
