#!/usr/bin/env python3
"""E2.S1 delay audit — listen 8-case delay tracking trace dump.

Captures per-frame delay tracking state on the 8 worst-FS cases
identified by the v3.12 listen exercise. Outputs per-case JSON
with:

  gcc_phat_initial_delay   - pre-alignment GCC-PHAT estimate
                             (matches eval_aec_challenge.estimate_delay)
  trace_rows               - per-update rows from DelayEstimator
                             (estimated_delay / last_par / confidence /
                              is_solid / top3_lag/height/par)
  n_estimates              - how many delay re-estimations fired
  final_delay              - last estimated_delay
  max_par / max_confidence - peak quality metrics
  case_signal_stats        - mic/lpb peak / RMS / clip-rate
                             (for cross-reference with listen finding)

Audit setup matches the bench harness exactly (preset=BALANCED,
fl=832, cng=True, seed=0). `enable_delay_est=True` forced ON for all
8 cases (not just movement) so we get tracking trace even on static
cases. `delay_est_period_s=0.25` (same as F-DelayTrack S2 default-ON
config for movement cases).

Purpose: identify whether the worst-FS leak comes from
(a) GCC-PHAT pre-alignment failure (initial delay wrong), or
(b) post-alignment drift (alignment OK but movement causes drift),
or
(c) delay genuinely exceeds filter_length=832 coverage.

Output dir: `results/v3_13_e2_s1_audit/`
  per-case JSON: <stem>.json
  summary table: summary.md

Usage:
    python3 tools/research/e2_s1_delay_audit.py \\
        --out results/v3_13_e2_s1_audit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


# 8 listen cases preserved from v3.12 worst-FS analysis (2026-05-13).
# Order matches listen/v3_12_worst_fs/ numbering.
LISTEN_CASES = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                'FS_static',   False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',                'FS_static',   False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',                'FS_static',   False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',                'FS_static',   False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',                'FS_static',   False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',  'FS_movement', True ),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',  'FS_movement', True ),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',  'FS_movement', True ),
]


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
    return delay, float(confidence)


def _signal_stats(name, sig):
    """Quick mic/lpb signal characterization (multi-threshold clip)."""
    s = sig.astype(np.float64)
    peak = float(np.max(np.abs(s)))
    rms = float(np.sqrt(np.mean(s ** 2)))
    abs_s = np.abs(s)
    return {
        f'{name}_peak': peak,
        f'{name}_rms': rms,
        f'{name}_peak_dbfs': 20 * np.log10(peak + 1e-12),
        f'{name}_clip_90': float(np.mean(abs_s > 0.90)),
        f'{name}_clip_95': float(np.mean(abs_s > 0.95)),
        f'{name}_clip_99': float(np.mean(abs_s > 0.99)),
        f'{name}_clip_999': float(np.mean(abs_s > 0.999)),
    }


def audit_one(case_num, stem, scenario, movement, dataset_dir):
    from aec import AEC, AecConfig, AecMode, AecPreset
    t0 = time.time()
    sub = 'farend_singletalk'
    mic_path = os.path.join(dataset_dir, sub, f'{stem}_mic.wav')
    lpb_path = os.path.join(dataset_dir, sub, f'{stem}_lpb.wav')
    mic, sr_m = sf.read(mic_path)
    lpb, sr_l = sf.read(lpb_path)
    assert sr_m == sr_l, f'sr mismatch: {sr_m} vs {sr_l}'
    sr = int(sr_m)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)

    # GCC-PHAT initial delay (matches bench pre-alignment)
    gcc_delay, gcc_conf = _estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < gcc_delay < n:
        lpb_aligned = np.zeros(n, dtype=np.float32)
        lpb_aligned[gcc_delay:] = lpb[: n - gcc_delay]
    else:
        lpb_aligned = lpb[:n]
    mic = mic[:n]

    # Config: matches bench except enable_delay_est forced ON + trace ON
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True,
        enable_cng=True, use_kalman=True,
        enable_delay_est=True,
        delay_est_period_s=0.25,
        delay_est_init_s=0.2,
        trace_delay_est=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos: pos + hop], lpb_aligned[pos: pos + hop])
        pos += hop

    trace_rows = (aec.delay_est._trace_rows
                  if aec.delay_est is not None
                  else [])

    # Summary stats
    summary = {
        'case_num': case_num,
        'stem': stem,
        'scenario': scenario,
        'movement': movement,
        'n_samples': int(n),
        'duration_s': float(n / sr),
        'gcc_phat_initial_delay_samples': int(gcc_delay),
        'gcc_phat_initial_delay_ms': float(gcc_delay * 1000.0 / sr),
        'gcc_phat_initial_confidence': float(gcc_conf),
        'n_trace_rows': len(trace_rows),
        'n_estimates_fired': sum(1 for r in trace_rows if r.get('did_estimate')),
        'elapsed_s': time.time() - t0,
    }
    if trace_rows:
        final = trace_rows[-1]
        valid = [r for r in trace_rows if r['estimated_delay'] >= 0]
        valid_par = [r['last_par'] for r in trace_rows
                     if r['last_par'] > 0]
        valid_conf = [r['confidence'] for r in trace_rows
                      if r['confidence'] > 0]
        summary['final_estimated_delay'] = int(final['estimated_delay'])
        summary['final_last_par'] = float(final['last_par'])
        summary['final_confidence'] = float(final['confidence'])
        summary['delay_estimates_distinct'] = sorted(set(
            r['estimated_delay'] for r in valid))[:20]
        summary['max_par'] = float(max(valid_par)) if valid_par else 0.0
        summary['mean_par'] = float(np.mean(valid_par)) if valid_par else 0.0
        summary['max_confidence'] = float(max(valid_conf)) if valid_conf else 0.0
        summary['mean_confidence'] = float(np.mean(valid_conf)) if valid_conf else 0.0
        summary['solid_fraction'] = float(
            np.mean([1.0 if r.get('is_solid') else 0.0 for r in trace_rows]))

    # Signal stats
    summary.update(_signal_stats('mic', mic))
    summary.update(_signal_stats('lpb', lpb))

    return {
        'summary': summary,
        'trace_rows': trace_rows,
    }


def main():
    ap = argparse.ArgumentParser(description='E2.S1 delay audit (8-case listen)')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f'[e2-s1] {len(LISTEN_CASES)} listen cases, out={args.out}',
          flush=True)

    all_summaries = []
    for case_num, stem, scenario, movement in LISTEN_CASES:
        print(f'[e2-s1] case {case_num} {stem[:24]}...', flush=True)
        result = audit_one(case_num, stem, scenario, movement, args.dataset)
        with open(os.path.join(args.out, f'{case_num}_{stem[:16]}.json'),
                  'w') as f:
            json.dump(result, f, indent=2)
        all_summaries.append(result['summary'])

    # Summary table
    lines = []
    lines.append('# E2.S1 delay audit summary')
    lines.append('')
    lines.append('Per-case delay tracking state on v3.12 worst-FS listen 8-case.')
    lines.append('All cases run with `enable_delay_est=True`, '
                 '`delay_est_period_s=0.25`, `trace_delay_est=True`.')
    lines.append('')
    lines.append('## Per-case summary')
    lines.append('')
    lines.append('| # | Stem | Scenario | mic peak (dBFS) | mic clip% '
                 '(>0.95/0.99) | GCC-PHAT delay (samp / ms) | GCC conf | '
                 'Online final delay (samp) | Online n_est | Online mean '
                 'PAR | Solid frac | Listen note |')
    lines.append('|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|')
    listen_notes = {
        '01': 'delay ~10000 samp + clipping',
        '02': 'delay ~1200 samp + clipping + radio-like',
        '03': 'gain 變化, mic 品質差 (E1)',
        '04': 'delay ~9800 samp',
        '05': 'delay ~6000 samp',
        '06': 'delay ~1800 samp + clipping',
        '07': 'delay ~640 samp + 嚴重 NL',
        '08': 'delay ~10000 samp + occasional sat',
    }
    for s in all_summaries:
        n = s['case_num']
        lines.append(f"| {n} | {s['stem'][:16]} | {s['scenario']} | "
                     f"{s['mic_peak_dbfs']:.2f} | "
                     f"{s['mic_clip_95']*100:.2f}% / {s['mic_clip_99']*100:.2f}% | "
                     f"{s['gcc_phat_initial_delay_samples']} / "
                     f"{s['gcc_phat_initial_delay_ms']:.1f} ms | "
                     f"{s['gcc_phat_initial_confidence']:.2f} | "
                     f"{s.get('final_estimated_delay', -1)} | "
                     f"{s['n_estimates_fired']} | "
                     f"{s.get('mean_par', 0):.2f} | "
                     f"{s.get('solid_fraction', 0)*100:.1f}% | "
                     f"{listen_notes.get(n, '')} |")
    lines.append('')
    lines.append('## Interpretation cues')
    lines.append('')
    lines.append('- **mic clip% > 1%**: clipping audible (matches listen E5 note)')
    lines.append('- **GCC-PHAT initial delay**: bench pre-alignment value; '
                 'if listen reports residual delay >> this, GCC was right but '
                 'something downstream drifted')
    lines.append('- **Online final delay**: where the online tracker landed; '
                 'large vs initial = movement / drift')
    lines.append('- **Online n_est**: how many times the tracker re-estimated; '
                 '0 means tracker never had enough data')
    lines.append('- **Online mean PAR**: < 5 → unreliable; 5-8 → marginal; '
                 '> 8 → solid')
    lines.append('- **Solid fraction**: % of frames the tracker considered '
                 'its estimate solid')
    lines.append('')

    summary_path = os.path.join(args.out, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'[e2-s1] wrote {summary_path}')

    # Also dump aggregated JSON for downstream processing
    with open(os.path.join(args.out, 'all_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
