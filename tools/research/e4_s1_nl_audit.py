#!/usr/bin/env python3
"""E4.S1 — NL-positive cohort identification audit (output-only metrics).

Per the v3.13 E4 NLP arc plan
(`~/.claude/plans/se-aec-aec-v3-13-e4-nlp-arc.md`), E4.S1 must
identify the NL-positive cohort size BEFORE committing to the
12-sprint module build. The hard gate: if N < 3 → close the arc
without FAIL.

This is Pass A (output-only). Three candidate metrics from the plan,
all computed on the *baseline production output* WAV (no AEC re-run):

  M1  HF/LF energy ratio  — harmonic distortion pushes energy into
      HF; ratio in 2-4 kHz / 100-500 Hz on residual output. NL
      cohort should show elevated M1 vs static-FS control.
  M2  Spectral peak prominence (PSD peak / median) in 1-4 kHz —
      harmonic comb leaves discrete spikes; non-harmonic residual
      is broadband.
  M3  Cepstral peak prominence in 80-500 Hz fundamental band on
      voice-active frames of output — pitched harmonic distortion
      leaves a discrete cepstral peak at the pitch period.

Plus E5-merged metrics (per user 2026-05-13 decision to merge E5.S2
into the E4 arc):
  M4  mic peak / mic RMS  — clipping signature on input mic
  M5  mic-lpb peak correlation r — high r + clipping = lpb-driven
      mic saturation (acoustic NL from loudspeaker driven hot)
  M6  multi-threshold mic clip rate (>0.99 ratio)

Output: per-case JSON + summary.md with NL-positive cohort list.

Usage:
  python3 tools/research/e4_s1_nl_audit.py \\
      --baseline-dir results/v3_11_candidate \\
      --sample-buckets FS_static FS_movement \\
      --out results/v3_13_e4_s1_audit
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


# 8-case listen set (v3.12 worst-FS analysis 2026-05-13).
LISTEN_8 = [
    ('01', '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',
        'FS_static', 'delay ~10000 + clip', False),
    ('02', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk',
        'FS_static', 'delay ~1200 + clip + radio', False),
    ('03', 'pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk',
        'FS_static', 'gain变化 mic质量差 (E1)', False),
    ('04', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk',
        'FS_static', 'delay ~9800', False),
    ('05', 'hVqUmGvIlkO0LBUoE06Q3w_farend_singletalk',
        'FS_static', 'delay ~6000', False),
    ('06', '5bJUo1K3uEmMrGa9UhGyVg_farend_singletalk_with_movement',
        'FS_movement', 'delay ~1800 + clip', False),
    ('07', 'IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement',
        'FS_movement', '严重 NL distortion (喇叭推爆)', True),
    ('08', 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement',
        'FS_movement', 'delay ~10000 + occasional full-scale', False),
]


def voice_active_mask(x, sr, win_ms=20.0, hop_ms=10.0, energy_pct=20.0):
    """Boolean mask over samples for above-energy-percentile frames."""
    n = len(x)
    win = int(win_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    if n < win:
        return np.ones(n, dtype=bool)
    n_frames = (n - win) // hop + 1
    en = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + win].astype(np.float64)
        en[i] = np.mean(seg ** 2)
    thr = np.percentile(en, energy_pct)
    mask = np.zeros(n, dtype=bool)
    for i in range(n_frames):
        if en[i] > thr:
            mask[i * hop: i * hop + win] = True
    return mask


def m1_hf_lf_ratio(out_signal, sr):
    """HF (2-4 kHz) / LF (100-500 Hz) energy ratio on voice-active frames."""
    mask = voice_active_mask(out_signal, sr)
    if mask.sum() < int(0.2 * sr):
        return float('nan')
    win = int(0.032 * sr)  # 32 ms FFT
    hop = int(0.010 * sr)
    n = len(out_signal)
    n_frames = (n - win) // hop + 1
    if n_frames <= 0:
        return float('nan')
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    lf_bins = (freqs >= 100) & (freqs <= 500)
    hf_bins = (freqs >= 2000) & (freqs <= 4000)
    lf_e_sum = 0.0
    hf_e_sum = 0.0
    n_voice_frames = 0
    for i in range(n_frames):
        seg = out_signal[i * hop: i * hop + win]
        if not mask[i * hop + win // 2]:
            continue
        seg_w = seg * np.hanning(win)
        spec = np.fft.rfft(seg_w)
        psd = np.abs(spec) ** 2
        lf_e_sum += float(np.sum(psd[lf_bins]))
        hf_e_sum += float(np.sum(psd[hf_bins]))
        n_voice_frames += 1
    if n_voice_frames < 10 or lf_e_sum < 1e-12:
        return float('nan')
    return 10.0 * np.log10(hf_e_sum / lf_e_sum)  # dB


def m2_hf_peak_prominence(out_signal, sr):
    """1-4 kHz PSD peak-to-median ratio averaged over voice frames (dB)."""
    mask = voice_active_mask(out_signal, sr)
    if mask.sum() < int(0.2 * sr):
        return float('nan')
    win = int(0.032 * sr)
    hop = int(0.010 * sr)
    n = len(out_signal)
    n_frames = (n - win) // hop + 1
    if n_frames <= 0:
        return float('nan')
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    band = (freqs >= 1000) & (freqs <= 4000)
    ratios = []
    for i in range(n_frames):
        if not mask[i * hop + win // 2]:
            continue
        seg = out_signal[i * hop: i * hop + win] * np.hanning(win)
        psd = np.abs(np.fft.rfft(seg)) ** 2
        band_psd = psd[band]
        if len(band_psd) == 0:
            continue
        med = float(np.median(band_psd))
        peak = float(np.max(band_psd))
        if med > 1e-15:
            ratios.append(10.0 * np.log10(peak / med))
    if len(ratios) < 10:
        return float('nan')
    return float(np.mean(ratios))


def m3_cepstral_peak(out_signal, sr):
    """Mean cepstral peak prominence in 2-12.5 ms (80-500 Hz fundamental).

    High value = clear pitched component (NE voice OR pitched
    distortion). Combined with M1/M2 to distinguish.
    """
    mask = voice_active_mask(out_signal, sr)
    if mask.sum() < int(0.2 * sr):
        return float('nan')
    win = int(0.032 * sr)
    hop = int(0.010 * sr)
    n = len(out_signal)
    n_frames = (n - win) // hop + 1
    if n_frames <= 0:
        return float('nan')
    qmin = int(0.002 * sr)  # 500 Hz
    qmax = int(0.0125 * sr)  # 80 Hz
    ratios = []
    for i in range(n_frames):
        if not mask[i * hop + win // 2]:
            continue
        seg = out_signal[i * hop: i * hop + win] * np.hanning(win)
        spec = np.fft.rfft(seg)
        log_mag = np.log(np.abs(spec) + 1e-10)
        ceps = np.fft.irfft(log_mag)
        if qmax > len(ceps):
            continue
        slab = ceps[qmin: qmax + 1]
        if len(slab) < 5:
            continue
        med = float(np.median(np.abs(slab)))
        peak = float(np.max(np.abs(slab)))
        if med > 1e-12:
            ratios.append(peak / med)
    if len(ratios) < 10:
        return float('nan')
    return float(np.mean(ratios))


def e5_metrics(mic_signal, lpb_signal):
    """E5-merged metrics: clip ratio + mic-lpb peak r."""
    abs_mic = np.abs(mic_signal)
    abs_lpb = np.abs(lpb_signal)
    n = min(len(abs_mic), len(abs_lpb))
    # Multi-threshold clip rate
    m4 = float(np.max(np.abs(mic_signal)) / (float(np.sqrt(np.mean(mic_signal**2))) + 1e-12))  # peak-to-rms
    m6_99 = float(np.mean(abs_mic > 0.99))
    m6_999 = float(np.mean(abs_mic > 0.999))
    # mic-lpb envelope correlation on short windows (100 ms)
    sr_assumed = 16000
    win = int(0.10 * sr_assumed)
    hop = int(0.05 * sr_assumed)
    if n < win:
        return {'m4_mic_par': m4, 'm5_mic_lpb_r': float('nan'),
                'm6_clip99': m6_99, 'm6_clip999': m6_999}
    n_frames = (n - win) // hop + 1
    mic_env = np.array([np.sqrt(np.mean(abs_mic[i*hop:i*hop+win] ** 2))
                        for i in range(n_frames)])
    lpb_env = np.array([np.sqrt(np.mean(abs_lpb[i*hop:i*hop+win] ** 2))
                        for i in range(n_frames)])
    valid = (mic_env > 1e-5) & (lpb_env > 1e-5)
    if valid.sum() < 5:
        m5 = float('nan')
    else:
        a = mic_env[valid] - mic_env[valid].mean()
        b = lpb_env[valid] - lpb_env[valid].mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        m5 = float(np.sum(a * b) / denom) if denom > 0 else float('nan')
    return {'m4_mic_par': m4, 'm5_mic_lpb_r': m5,
            'm6_clip99': m6_99, 'm6_clip999': m6_999}


def audit_one(stem, scenario, baseline_dir, dataset_dir, sr=16000,
              use_linear=True):
    """Compute metrics for one case from existing baseline output.

    use_linear=True → use `_ours_nores.wav` (linear residual; RES disabled).
    This is the right surface for NL detection because RES suppression
    masks NL evidence in the production output.
    """
    suffix = '_ours_nores.wav' if use_linear else '_ours.wav'
    out_p = os.path.join(baseline_dir, f'{stem}{suffix}')
    mic_p = os.path.join(dataset_dir, scenario.lower().replace('_static', '').replace('_movement', ''),
                          f'{stem}_mic.wav')
    # Subdir mapping based on stem suffix
    if '_doubletalk' in stem:
        sub = 'doubletalk'
    elif '_farend_singletalk' in stem:
        sub = 'farend_singletalk'
    elif '_nearend_singletalk' in stem:
        sub = 'nearend_singletalk'
    else:
        return None
    mic_p = os.path.join(dataset_dir, sub, f'{stem}_mic.wav')
    lpb_p = os.path.join(dataset_dir, sub, f'{stem}_lpb.wav')
    if not (os.path.isfile(out_p) and os.path.isfile(mic_p)
            and os.path.isfile(lpb_p)):
        return None
    out_sig, sr_o = sf.read(out_p)
    mic_sig, _ = sf.read(mic_p)
    lpb_sig, _ = sf.read(lpb_p)
    sr_use = int(sr_o)
    out_sig = out_sig.astype(np.float64)
    mic_sig = mic_sig.astype(np.float64)
    lpb_sig = lpb_sig.astype(np.float64)
    m1 = m1_hf_lf_ratio(out_sig, sr_use)
    m2 = m2_hf_peak_prominence(out_sig, sr_use)
    m3 = m3_cepstral_peak(out_sig, sr_use)
    e5 = e5_metrics(mic_sig, lpb_sig)
    return {
        'stem': stem,
        'm1_hf_lf_db': m1,
        'm2_peak_med_db': m2,
        'm3_cepstral': m3,
        **e5,
    }


def list_bucket_cases(dataset, bucket, limit=None, seed=0):
    """Return list of (stem, scenario) for a bucket."""
    if bucket == 'FS_static':
        sub = 'farend_singletalk'
        filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' not in f
        scen = 'FS_static'
    elif bucket == 'FS_movement':
        sub = 'farend_singletalk'
        filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' in f
        scen = 'FS_movement'
    elif bucket == 'DT_static':
        sub = 'doubletalk'
        filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' not in f
        scen = 'DT_static'
    elif bucket == 'DT_movement':
        sub = 'doubletalk'
        filt = lambda f: f.endswith('_mic.wav') and '_with_movement_' in f
        scen = 'DT_movement'
    elif bucket == 'NE':
        sub = 'nearend_singletalk'
        filt = lambda f: f.endswith('_mic.wav')
        scen = 'NE'
    else:
        return []
    sub_path = os.path.join(dataset, sub)
    if not os.path.isdir(sub_path):
        return []
    stems = []
    for f in sorted(os.listdir(sub_path)):
        if filt(f):
            stems.append((f[:-len('_mic.wav')], scen))
    if limit is not None and limit < len(stems):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(stems), size=limit, replace=False)
        idx.sort()
        stems = [stems[i] for i in idx]
    return stems


def main():
    ap = argparse.ArgumentParser(description='E4.S1 NL audit (output-only)')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--baseline-dir', default='results/v3_11_candidate')
    ap.add_argument('--sample-buckets', nargs='+',
                    default=['FS_static', 'FS_movement'])
    ap.add_argument('--sample-per-bucket', type=int, default=25,
                    help='Number of random cases per bucket (in addition '
                         'to the 8 listen cases)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Listen 8 cases (already in known cohort)
    listen_rows = []
    for case_num, stem, scen, note, is_nl in LISTEN_8:
        r = audit_one(stem, scen, args.baseline_dir, args.dataset)
        if r is None:
            print(f'[e4-s1] listen {case_num} {stem[:24]} MISSING')
            continue
        r.update({'case_num': case_num, 'scenario': scen,
                  'listen_note': note, 'is_nl_listen': is_nl})
        listen_rows.append(r)
        print(f'[e4-s1] L{case_num} M1={r["m1_hf_lf_db"]:.2f} '
              f'M2={r["m2_peak_med_db"]:.2f} M3={r["m3_cepstral"]:.2f} '
              f'M5={r["m5_mic_lpb_r"]:.2f} clip99%={r["m6_clip99"]*100:.2f}',
              flush=True)

    # 2. Random sample from buckets (NL detector should not light up too
    # often on baseline cases; want to know the bucket distribution)
    sample_rows = []
    for bucket in args.sample_buckets:
        cases = list_bucket_cases(args.dataset, bucket,
                                   limit=args.sample_per_bucket)
        print(f'[e4-s1] sample {bucket}: {len(cases)} cases', flush=True)
        for stem, scen in cases:
            r = audit_one(stem, scen, args.baseline_dir, args.dataset)
            if r is None:
                continue
            r.update({'scenario': scen, 'listen_note': '', 'is_nl_listen': False})
            sample_rows.append(r)

    all_rows = listen_rows + sample_rows

    # Aggregate metric ranges
    def stats(metric):
        vals = [r[metric] for r in sample_rows
                if not np.isnan(r.get(metric, float('nan')))]
        if not vals:
            return {}
        return {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'p25': float(np.percentile(vals, 25)),
            'p75': float(np.percentile(vals, 75)),
            'p95': float(np.percentile(vals, 95)),
            'max': float(np.max(vals)),
        }

    summary = {
        'm1_hf_lf_db_sample_stats': stats('m1_hf_lf_db'),
        'm2_peak_med_db_sample_stats': stats('m2_peak_med_db'),
        'm3_cepstral_sample_stats': stats('m3_cepstral'),
        'm4_mic_par_sample_stats': stats('m4_mic_par'),
        'm5_mic_lpb_r_sample_stats': stats('m5_mic_lpb_r'),
        'm6_clip99_sample_stats': stats('m6_clip99'),
    }

    # Compute "NL-positive flag" per case using percentile thresholds from
    # sample distribution. NL signature = elevated M1 AND elevated M2
    # AND elevated M3 (harmonic content) AND elevated M5 (acoustic NL ⇒
    # mic-lpb envelope tracking).
    th_m1 = summary['m1_hf_lf_db_sample_stats'].get('p95', np.inf)
    th_m2 = summary['m2_peak_med_db_sample_stats'].get('p95', np.inf)
    th_m3 = summary['m3_cepstral_sample_stats'].get('p95', np.inf)
    th_m5 = summary['m5_mic_lpb_r_sample_stats'].get('p95', np.inf)

    def is_nl_positive(r):
        # multi-criteria: any 2 of 4 metrics exceed p95
        if any(np.isnan(r.get(k, float('nan'))) for k in
               ('m1_hf_lf_db', 'm2_peak_med_db', 'm3_cepstral', 'm5_mic_lpb_r')):
            return False
        hits = 0
        if r['m1_hf_lf_db'] > th_m1: hits += 1
        if r['m2_peak_med_db'] > th_m2: hits += 1
        if r['m3_cepstral'] > th_m3: hits += 1
        if r['m5_mic_lpb_r'] > th_m5: hits += 1
        return hits >= 2

    for r in all_rows:
        r['detected_nl'] = is_nl_positive(r)

    nl_positive = [r for r in all_rows if r['detected_nl']]
    listen_nl_marked = [r for r in listen_rows if r.get('is_nl_listen')]

    md = [
        '# E4.S1 NL audit — Pass A (output-only metrics)',
        '',
        'Per the v3.13 E4 NLP arc plan. Output-only metrics computed on '
        f'baseline `{args.baseline_dir}` rendered WAVs (no AEC re-run).',
        '',
        'Metric set:',
        '- M1: HF/LF energy ratio (2-4 kHz / 100-500 Hz) on output, voice-active frames, dB',
        '- M2: HF peak-to-median PSD (1-4 kHz) on output, voice-active frames, dB',
        '- M3: cepstral peak prominence (80-500 Hz fundamental) on output',
        '- M4: mic peak-to-RMS ratio (input)',
        '- M5: mic-lpb envelope correlation r (input, 100 ms windows)',
        '- M6: mic clip rate at >0.99 (input)',
        '',
        '## Listen 8-case + listen note',
        '',
        '| # | Stem | Scen | Note | M1 (dB) | M2 (dB) | M3 | M4 | M5 | clip99% | detected NL |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|',
    ]
    for r in listen_rows:
        md.append(
            f"| {r['case_num']} | `{r['stem'][:24]}` | {r['scenario']} | "
            f"{r['listen_note'][:18]} | "
            f"{r['m1_hf_lf_db']:.2f} | {r['m2_peak_med_db']:.2f} | "
            f"{r['m3_cepstral']:.2f} | {r['m4_mic_par']:.1f} | "
            f"{r['m5_mic_lpb_r']:.2f} | {r['m6_clip99']*100:.1f}% | "
            f"{'YES' if r['detected_nl'] else 'no'} |"
        )
    md.append('')

    md.append('## Sample distribution (per-bucket random sample, n='
              f'{args.sample_per_bucket}/bucket)')
    md.append('')
    md.append('| Metric | mean | median | p25 | p75 | p95 | max |')
    md.append('|---|---:|---:|---:|---:|---:|---:|')
    for key, label in [('m1_hf_lf_db_sample_stats', 'M1 HF/LF (dB)'),
                       ('m2_peak_med_db_sample_stats', 'M2 peak/med (dB)'),
                       ('m3_cepstral_sample_stats', 'M3 cepstral'),
                       ('m4_mic_par_sample_stats', 'M4 mic PAR'),
                       ('m5_mic_lpb_r_sample_stats', 'M5 mic-lpb r'),
                       ('m6_clip99_sample_stats', 'M6 clip99')]:
        s = summary[key]
        if not s:
            continue
        md.append(f"| {label} | {s['mean']:.2f} | {s['median']:.2f} | "
                  f"{s['p25']:.2f} | {s['p75']:.2f} | {s['p95']:.2f} | "
                  f"{s['max']:.2f} |")
    md.append('')
    md.append(f'NL-positive threshold (≥2 of 4 metrics > p95): M1>{th_m1:.2f}, '
              f'M2>{th_m2:.2f}, M3>{th_m3:.2f}, M5>{th_m5:.2f}')
    md.append('')
    md.append(f'## Cohort summary')
    md.append(f'- Listen 8-case marked NL: {len(listen_nl_marked)} (case 07)')
    md.append(f'- Detected NL (8-case + sample): {len(nl_positive)}')
    md.append(f'- Detected NL in 8-case listen: '
              f'{sum(1 for r in listen_rows if r["detected_nl"])}')
    md.append('')
    md.append('## E4 arc gate (per plan)')
    md.append('')
    md.append('**Hard gate**: if N_NL < 3 in 800-case (or per-bucket ratio '
              '< 1%), E4 arc closes WITHOUT FAIL. Otherwise → E4.S2 '
              'detector design lock.')
    md.append('')
    if len(nl_positive) >= 3:
        md.append(f'**Gate result (Pass A audit)**: N={len(nl_positive)} '
                  f'detected on (8 + {args.sample_per_bucket*len(args.sample_buckets)} '
                  f'sample) → proceed to detector design (E4.S2) AFTER full '
                  '800-case Pass B render.')
    else:
        md.append(f'**Gate result (Pass A audit)**: N={len(nl_positive)} '
                  f'detected on Pass A sample → likely close E4 arc. '
                  'Confirm with full 800-case Pass B before commit.')

    out_md = os.path.join(args.out, 'summary.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'rows.json'), 'w') as f:
        json.dump(all_rows, f, indent=2)
    with open(os.path.join(args.out, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[e4-s1] wrote {out_md}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
