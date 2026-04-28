#!/usr/bin/env python3
"""Per-case gap analysis: v2.8.0 ours vs AEC2/AEC3 from v2.5 reference CSV.

Usage:
    python3 gap_analysis.py <wav_root> <out_dir> <ref_csv> [out_csv]

Example:
    python3 gap_analysis.py ../wav/aec_challenge_blind /tmp/v280_fl512_balanced \
        ../docs/benchmarks/v2_5/per_file_scores.csv /tmp/gap_v280.csv
"""
import os, sys, csv
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from smoke_retune import FastAECMOS

SCENARIOS = [
    ('farend_singletalk', 'st'),
    ('doubletalk',        'dt'),
    ('nearend_singletalk','nst'),
]


def score_wav(est, talk_type, lpb_path, mic_path, enh_path):
    lpb, _ = sf.read(lpb_path, dtype='float32')
    mic, _ = sf.read(mic_path, dtype='float32')
    enh, _ = sf.read(enh_path, dtype='float32')
    n = min(len(lpb), len(mic), len(enh))
    return est.score(talk_type, lpb[:n], mic[:n], enh[:n])


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)

    wav_root = sys.argv[1]
    out_dir  = sys.argv[2]
    ref_csv  = sys.argv[3]
    out_csv  = sys.argv[4] if len(sys.argv) > 4 else None

    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model', 'Run_1663915512_Stage_0.onnx')
    est = FastAECMOS(model)

    # load reference (AEC2/AEC3)
    ref = {}
    with open(ref_csv) as f:
        for row in csv.DictReader(f):
            key = (row['file_id'], row['scenario'])
            ref[key] = row

    rows = []
    for subdir, talk_type in SCENARIOS:
        wav_dir = os.path.join(wav_root, subdir)
        if not os.path.isdir(wav_dir):
            continue
        mic_files = sorted(f for f in os.listdir(wav_dir) if f.endswith('_mic.wav'))
        print(f"\n{subdir}: {len(mic_files)} cases")
        scored = 0
        for mic_fname in mic_files:
            prefix   = mic_fname.replace('_mic.wav', '')
            # file_id in ref CSV strips the scenario suffix from the prefix
            file_id  = prefix.replace(f'_{subdir}', '')
            mic_path = os.path.join(wav_dir, mic_fname)
            lpb_path = os.path.join(wav_dir, prefix + '_lpb.wav')
            enh_path = os.path.join(out_dir, prefix + '_ours.wav')
            if not os.path.exists(lpb_path) or not os.path.exists(enh_path):
                continue
            try:
                echo, deg = score_wav(est, talk_type, lpb_path, mic_path, enh_path)
            except Exception as e:
                print(f"  SKIP {prefix}: {e}")
                continue

            r = ref.get((file_id, subdir), {})
            row = dict(
                file_id=file_id, scenario=subdir,
                ours_echo=round(echo, 4), ours_deg=round(deg, 4),
                aec2_echo=r.get('aec2_echo', ''), aec2_deg=r.get('aec2_deg', ''),
                aec3_echo=r.get('aec3_echo', ''), aec3_deg=r.get('aec3_deg', ''),
                is_movement=int('with_movement' in prefix),
            )
            rows.append(row)
            scored += 1
        print(f"  scored {scored} files")

    if out_csv:
        fields = ['file_id','scenario','ours_echo','ours_deg',
                  'aec2_echo','aec2_deg','aec3_echo','aec3_deg','is_movement']
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved {len(rows)} rows → {out_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SCENARIO MEANS")
    print("="*72)
    for subdir, _ in SCENARIOS:
        sc = [r for r in rows if r['scenario'] == subdir]
        if not sc: continue
        def mean_col(col):
            vals = [float(r[col]) for r in sc if r[col] != '']
            return np.mean(vals) if vals else float('nan')
        print(f"\n{subdir}:")
        print(f"  {'':8s}  echo    deg")
        for m in ['ours','aec2','aec3']:
            print(f"  {m:8s}  {mean_col(m+'_echo'):.3f}   {mean_col(m+'_deg'):.3f}")

    # ── Worst-case gap tables ─────────────────────────────────────────────────
    analysis_targets = [
        ('farend_singletalk', 'echo', 'FS echo'),
        ('doubletalk',        'echo', 'DT echo'),
        ('doubletalk',        'deg',  'DT deg (nearend quality)'),
    ]
    print("\n" + "="*72)
    print("WORST-CASE GAPS (ours < competitor)")
    print("="*72)
    for scenario, metric, label in analysis_targets:
        sc = [r for r in rows if r['scenario'] == scenario]
        print(f"\n{'─'*72}")
        print(f"{label}  (n={len(sc)})")
        for comp in ['aec2','aec3']:
            gaps = []
            for r in sc:
                ours = r.get(f'ours_{metric}', '')
                cv   = r.get(f'{comp}_{metric}', '')
                if ours == '' or cv == '': continue
                gap = float(cv) - float(ours)
                if gap > 0:
                    gaps.append((gap, r))
            if not gaps: continue
            gaps.sort(key=lambda x: -x[0])
            mean_gap = np.mean([g for g, _ in gaps])
            behind_n = len(gaps)
            print(f"\n  vs {comp.upper()}  (behind {behind_n}/{len(sc)} cases, mean_gap={mean_gap:.3f})")
            print(f"  {'file_id':38s}  mv  ours   {comp}  gap")
            for gap, r in gaps[:15]:
                fid = r['file_id'][:36]
                ours_v = float(r[f'ours_{metric}'])
                comp_v = float(r[f'{comp}_{metric}'])
                mv = 'Y' if r['is_movement'] else ' '
                print(f"  {fid:38s}  {mv}   {ours_v:.3f}  {comp_v:.3f}  +{gap:.3f}")

    # ── Movement breakdown ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("MOVEMENT vs STATIC breakdown")
    print("="*72)
    for scenario, metric, label in analysis_targets:
        sc = [r for r in rows if r['scenario'] == scenario]
        for mv_flag, mv_label in [(1,'movement'), (0,'static')]:
            sub = [r for r in sc if r['is_movement'] == mv_flag]
            if not sub: continue
            def mean_col(col):
                vals = [float(r[col]) for r in sub if r[col] != '']
                return np.mean(vals) if vals else float('nan')
            o = mean_col(f'ours_{metric}')
            a2 = mean_col(f'aec2_{metric}')
            a3 = mean_col(f'aec3_{metric}')
            print(f"  {label:30s}  {mv_label:8s} n={len(sub):3d}  "
                  f"ours={o:.3f}  aec2={a2:.3f}(Δ{o-a2:+.3f})  aec3={a3:.3f}(Δ{o-a3:+.3f})")


if __name__ == '__main__':
    main()
