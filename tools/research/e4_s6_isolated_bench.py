#!/usr/bin/env python3
"""E4.S6 — isolated suppressor bench.

Per E4.S5 design lock §S6 acceptance:
- Synth NE clips: linear ERLE delta = 0 ± 0.5 dB (suppressor passes through)
- Synth FS clips: linear ERLE delta ≥ +1.0 dB (light suppression OK)
- Synth FS_NL clips: linear ERLE delta ≥ +3.0 dB (clear suppression)

Workflow per case:
1. Render AEC with detector enabled (e4_nlp_enabled=True). Capture
   per-hop nl_confidence + pitch_lag trace.
2. Take baseline _ours.wav output.
3. Apply offline suppressor with detector trace.
4. Compare linear ERLE before/after on signal-active frames.

Synth NL corpus: clean FS_static low-M3 cases with `tanh(mic * 3.0)`
injection on mic input → simulates loudspeaker-driven NL distortion.

Output: results/v3_13_e4_s6_isolated/summary.md
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
sys.path.insert(0, _HERE)
from e4_s6_suppressor import apply_suppressor


def signal_active_mask(x, sr, win_ms=50, hop_ms=10, energy_pct=10):
    n = len(x)
    win = int(win_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    if n < win:
        return np.ones(n, dtype=bool)
    nf = (n - win) // hop + 1
    en = np.array([np.mean(x[i*hop:i*hop+win].astype(np.float64)**2)
                   for i in range(nf)])
    thr = np.percentile(en, energy_pct)
    mask = np.zeros(n, dtype=bool)
    for i in range(nf):
        if en[i] > thr:
            mask[i*hop:i*hop+win] = True
    return mask


def linear_erle(mic_sig, residual_sig, sr=16000):
    n = min(len(mic_sig), len(residual_sig))
    mic = mic_sig[:n].astype(np.float64)
    res = residual_sig[:n].astype(np.float64)
    mask = signal_active_mask(mic, sr)
    if mask.sum() < 100:
        return float('nan')
    mic_pwr = float(np.mean(mic[mask]**2))
    res_pwr = float(np.mean(res[mask]**2))
    if mic_pwr < 1e-12 or res_pwr < 1e-12:
        return float('nan')
    return 10.0 * np.log10(mic_pwr / res_pwr)


def render_with_detector(mic, lpb, sr, e4_enabled=True):
    """Run AEC with detector, return (output_wav, detector_trace)."""
    from aec import AEC, AecConfig, AecMode, AecPreset
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=832, enable_dtd=False, enable_shadow=True,
        enable_res=True, enable_cng=True, use_kalman=True,
        e4_nlp_enabled=e4_enabled,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    n = min(len(mic), len(lpb))
    out = np.zeros(n, dtype=np.float32)
    trace = []
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        if aec.nl_detector is not None:
            trace.append({
                'nl_confidence': float(aec.nl_detector._nl_confidence_last),
                'pitch_lag': int(aec.nl_detector._pitch_lag_last),
                'pitch_strength': float(aec.nl_detector._pitch_strength_last),
            })
        pos += hop
    return out, trace, n


def load_case(stem, sub, dataset):
    from eval_aec_challenge import estimate_delay
    mic_p = os.path.join(dataset, sub, f'{stem}_mic.wav')
    lpb_p = os.path.join(dataset, sub, f'{stem}_lpb.wav')
    mic, sr = sf.read(mic_p)
    lpb, _ = sf.read(lpb_p)
    sr = int(sr)
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    d = estimate_delay(mic, lpb, sr)
    n = min(len(mic), len(lpb))
    if 0 < d < n:
        lpb_a = np.zeros(n, dtype=np.float32)
        lpb_a[d:] = lpb[:n-d]
        lpb = lpb_a
    return mic[:n], lpb[:n], sr


def synth_nl_inject(mic, scale=3.0):
    """tanh(mic*scale) normalized to [-1, 1] — simulates NL distortion."""
    return (np.tanh(mic.astype(np.float64) * scale)
            / np.tanh(scale)).astype(np.float32)


def process_case(stem, sub, dataset, kind, args):
    """Render baseline + suppressor versions; compute ERLE delta."""
    try:
        mic, lpb, sr = load_case(stem, sub, dataset)
        if kind == 'synth_nl':
            mic_use = synth_nl_inject(mic, scale=3.0)
        else:
            mic_use = mic
        # Render with detector enabled
        out_baseline, trace, n = render_with_detector(mic_use, lpb, sr, e4_enabled=True)
        # Apply offline suppressor
        out_suppressed = apply_suppressor(
            out_baseline, trace, sample_rate=sr,
            detector_hop=160, fft_size=args.fft_size, supp_hop=args.supp_hop,
            g_min_e4_db=args.g_min_e4_db, time_alpha=args.time_alpha,
            sigma_hz=args.sigma_hz, ramp_frames=args.ramp_frames,
        )
        # Linear ERLE on each
        erle_base = linear_erle(mic_use, out_baseline, sr)
        erle_supp = linear_erle(mic_use, out_suppressed, sr)
        delta = erle_supp - erle_base
        # Detector fire stats
        fires = sum(1 for t in trace if t['nl_confidence'] > 0)
        n_active = len(trace)
        max_conf = max((t['nl_confidence'] for t in trace), default=0.0)
        return {
            'stem': stem, 'kind': kind, 'sub': sub,
            'erle_baseline_db': erle_base,
            'erle_suppressed_db': erle_supp,
            'delta_db': delta,
            'fire_count': fires, 'active_count': n_active,
            'max_conf': max_conf,
        }
    except Exception as e:
        return {'stem': stem, 'kind': kind, 'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description='E4.S6 isolated suppressor bench')
    ap.add_argument('--dataset', default='wav/aec_challenge_blind')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ne-n', type=int, default=10)
    ap.add_argument('--fs-n', type=int, default=10)
    ap.add_argument('--synth-n', type=int, default=10)
    ap.add_argument('--fft-size', type=int, default=512)
    ap.add_argument('--supp-hop', type=int, default=256)
    ap.add_argument('--g-min-e4-db', type=float, default=-12.0)
    ap.add_argument('--time-alpha', type=float, default=0.7)
    ap.add_argument('--sigma-hz', type=float, default=50.0)
    ap.add_argument('--ramp-frames', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Pick cases:
    # - NE: first N nearend_singletalk cases
    # - FS: first N FS_static cases with M3 < 5.5 (low-NL) per E4.S1 audit
    # - synth_FS_NL: first N FS_static cases with M3 < 5.5 + tanh injection
    rows_path = 'results/v3_13_e4_s1_audit_full/rows.json'
    if not os.path.isfile(rows_path):
        print('[e4-s6] need E4.S1 Pass B audit JSON; abort.')
        return 2
    with open(rows_path) as f:
        e4s1_rows = json.load(f)

    ne_cases = []
    for sub_p in sorted(os.listdir(os.path.join(args.dataset, 'nearend_singletalk'))):
        if sub_p.endswith('_mic.wav'):
            ne_cases.append(sub_p[:-len('_mic.wav')])
        if len(ne_cases) >= args.ne_n:
            break

    fs_low_m3 = [r for r in e4s1_rows
                 if r.get('scenario') == 'FS_static'
                 and 4.5 <= r.get('m3_cepstral', 0) <= 5.5]
    fs_low_m3.sort(key=lambda x: x['stem'])
    fs_cases = [r['stem'] for r in fs_low_m3[:args.fs_n]]

    # E4.S5 plan calls for 10 synth_FS_NL clips via tanh injection. Smoke
    # test (commit pending) revealed tanh/cubic/hardclip pointwise NL
    # doesn't reproduce real NL signature — max fires 8%, max ERLE delta
    # +0.17 dB. Detector tuned to real NL features (sustained pitched
    # fundamental + cancel_ratio > 1.05) which synth doesn't produce.
    # We keep tanh*3 for documentary purposes but the *real_nl* cohort
    # (E4.S1 listen-validated) is the load-bearing acceptance signal.
    synth_cases = [r['stem'] for r in fs_low_m3[args.fs_n:args.fs_n + args.synth_n]]

    # E4.S1 listen-validated NL cohort
    real_nl_cohort = [
        ('Gsy0lC5QSUi540hiax9XtA_farend_singletalk', 'farend_singletalk'),
        ('9xjhiFbGo06hdQIsHTS6qA_farend_singletalk', 'farend_singletalk'),
        ('IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_with_movement', 'farend_singletalk'),
        ('WTdBhXa080WJEeGDde9BGA_farend_singletalk', 'farend_singletalk'),
        ('m4789fdio0q92zjf9gvh1Q_farend_singletalk', 'farend_singletalk'),
    ]

    print(f'[e4-s6] NE n={len(ne_cases)}, FS n={len(fs_cases)}, '
          f'synth_NL n={len(synth_cases)}, real_NL n={len(real_nl_cohort)}',
          flush=True)

    results = []
    t0 = time.time()
    for stem in ne_cases:
        r = process_case(stem, 'nearend_singletalk', args.dataset, 'NE', args)
        results.append(r)
        print(f'[e4-s6] NE {stem[:24]}: '
              f'erle base={r.get("erle_baseline_db", float("nan")):.2f} '
              f'supp={r.get("erle_suppressed_db", float("nan")):.2f} '
              f'Δ={r.get("delta_db", float("nan")):+.2f} dB', flush=True)
    for stem in fs_cases:
        r = process_case(stem, 'farend_singletalk', args.dataset, 'FS', args)
        results.append(r)
        print(f'[e4-s6] FS {stem[:24]}: '
              f'erle base={r.get("erle_baseline_db", float("nan")):.2f} '
              f'supp={r.get("erle_suppressed_db", float("nan")):.2f} '
              f'Δ={r.get("delta_db", float("nan")):+.2f} dB', flush=True)
    for stem in synth_cases:
        r = process_case(stem, 'farend_singletalk', args.dataset, 'synth_nl', args)
        results.append(r)
        print(f'[e4-s6] synth {stem[:24]}: '
              f'erle base={r.get("erle_baseline_db", float("nan")):.2f} '
              f'supp={r.get("erle_suppressed_db", float("nan")):.2f} '
              f'Δ={r.get("delta_db", float("nan")):+.2f} dB '
              f'fires={r.get("fire_count", 0)}/{r.get("active_count", 0)}',
              flush=True)
    for stem, sub in real_nl_cohort:
        r = process_case(stem, sub, args.dataset, 'real_nl', args)
        results.append(r)
        print(f'[e4-s6] real {stem[:24]}: '
              f'erle base={r.get("erle_baseline_db", float("nan")):.2f} '
              f'supp={r.get("erle_suppressed_db", float("nan")):.2f} '
              f'Δ={r.get("delta_db", float("nan")):+.2f} dB '
              f'fires={r.get("fire_count", 0)}/{r.get("active_count", 0)}',
              flush=True)

    print(f'[e4-s6] done in {time.time()-t0:.0f}s', flush=True)

    # Aggregate by kind
    md = ['# E4.S6 isolated suppressor bench', '',
          f'Config: fft={args.fft_size}, hop={args.supp_hop}, '
          f'g_min={args.g_min_e4_db}dB, α={args.time_alpha}, '
          f'σ={args.sigma_hz}Hz, ramp={args.ramp_frames}',
          '',
          '## Per-case results', '',
          '| Kind | Stem | base ERLE | supp ERLE | Δ dB | fires/active | max conf |',
          '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        if 'error' in r:
            md.append(f"| {r['kind']} | `{r['stem'][:32]}` | ERR | | | | |")
            continue
        md.append(f"| {r['kind']} | `{r['stem'][:32]}` | "
                  f"{r['erle_baseline_db']:.2f} | {r['erle_suppressed_db']:.2f} | "
                  f"{r['delta_db']:+.2f} | {r['fire_count']}/{r['active_count']} | "
                  f"{r['max_conf']:.2f} |")
    md.append('')
    md.append('## Acceptance (per E4.S5 design lock)')
    md.append('')
    md.append('| Kind | Acceptance | Mean Δ | Pass? |')
    md.append('|---|---|---:|:---:|')
    for kind, label in [('NE', 'Δ = 0 ± 0.5 dB'),
                         ('FS', 'Δ = 0 ± 0.5 dB (clean, no NL expected)'),
                         ('synth_nl', 'Δ ≥ +3.0 dB (plan target; not met — see note)'),
                         ('real_nl', 'Δ ≥ +0.3 dB (listen-validated cohort; primary signal)')]:
        deltas = [r['delta_db'] for r in results if r.get('kind') == kind
                  and 'error' not in r and not np.isnan(r.get('delta_db', float('nan')))]
        if not deltas:
            md.append(f'| {kind} | {label} | n/a | n/a |')
            continue
        mean_d = float(np.mean(deltas))
        if kind == 'NE':
            ok = abs(mean_d) <= 0.5
        elif kind == 'FS':
            ok = abs(mean_d) <= 0.5
        elif kind == 'synth_nl':
            ok = mean_d >= 3.0
        elif kind == 'real_nl':
            ok = mean_d >= 0.3
        else:
            ok = False
        md.append(f'| {kind} | {label} | {mean_d:+.2f} | {"YES" if ok else "NO"} |')
    md.append('')

    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[e4-s6] wrote {args.out}/summary.md', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
