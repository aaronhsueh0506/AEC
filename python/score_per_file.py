#!/usr/bin/env python3
"""Per-file AECMOS scoring: ours vs linear(nores) vs AEC2(old_aec) vs AEC3.

Usage:
    python3 score_per_file.py <wav_root> <output_dir> <out_csv>

Example:
    python3 score_per_file.py ../wav/aec_challenge_blind output_v26_analysis output_v26_analysis/per_file_scores.csv
"""
import os
import sys
import csv
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))

from smoke_retune import FastAECMOS

SCENARIOS = [
    ('farend_singletalk', 'st'),
    ('nearend_singletalk', 'nst'),
    ('doubletalk', 'dt'),
]

METHODS = ['ours', 'ours_nores', 'old_aec', 'aec3']
METHOD_LABELS = ['ours', 'linear', 'aec2', 'aec3']


def score_file(estimator, talk_type, mic_path, lpb_path, enh_path):
    mic, _ = sf.read(mic_path)
    lpb, _ = sf.read(lpb_path)
    enh, _ = sf.read(enh_path)
    n = min(len(mic), len(lpb), len(enh))
    echo, deg = estimator.score(
        talk_type,
        lpb[:n].astype(np.float32),
        mic[:n].astype(np.float32),
        enh[:n].astype(np.float32),
    )
    return echo, deg


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    wav_root, out_dir, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model', 'Run_1663915512_Stage_0.onnx')
    estimator = FastAECMOS(model)

    rows = []
    method_sums = {lbl: {'echo': [], 'deg': []} for lbl in METHOD_LABELS}

    for subdir, talk_type in SCENARIOS:
        wav_dir = os.path.join(wav_root, subdir)
        if not os.path.isdir(wav_dir):
            continue

        mic_files = sorted(f for f in os.listdir(wav_dir) if f.endswith('_mic.wav'))
        print(f"\n{subdir}: {len(mic_files)} cases")

        for mic_fname in mic_files:
            prefix = mic_fname.replace('_mic.wav', '')
            file_id = prefix.replace(f'_{subdir}', '')
            mic_path = os.path.join(wav_dir, mic_fname)
            lpb_path = os.path.join(wav_dir, prefix + '_lpb.wav')
            if not os.path.exists(lpb_path):
                continue

            row = {'file_id': file_id, 'scenario': subdir}
            for method, label in zip(METHODS, METHOD_LABELS):
                enh_path = os.path.join(out_dir, f"{prefix}_{method}.wav")
                if os.path.exists(enh_path):
                    echo, deg = score_file(estimator, talk_type, mic_path, lpb_path, enh_path)
                    row[f'{label}_echo'] = round(echo, 4)
                    row[f'{label}_deg'] = round(deg, 4)
                    method_sums[label]['echo'].append(echo)
                    method_sums[label]['deg'].append(deg)
                else:
                    row[f'{label}_echo'] = ''
                    row[f'{label}_deg'] = ''
            rows.append(row)

        # Per-scenario means
        sc_rows = [r for r in rows if r['scenario'] == subdir]
        print(f"  {'Method':8s}  echo    deg")
        for label in METHOD_LABELS:
            echos = [r[f'{label}_echo'] for r in sc_rows if r[f'{label}_echo'] != '']
            degs  = [r[f'{label}_deg']  for r in sc_rows if r[f'{label}_deg']  != '']
            if echos:
                print(f"  {label:8s}  {np.mean(echos):.3f}   {np.mean(degs):.3f}")

    # Write CSV
    fieldnames = ['file_id', 'scenario']
    for lbl in METHOD_LABELS:
        fieldnames += [f'{lbl}_echo', f'{lbl}_deg']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {out_csv}")

    # Worst gap analysis: ours vs each competitor
    print("\n" + "="*70)
    print("WORST CASES: ours falls behind competitors")
    print("="*70)

    for scenario, metric_key, sort_label in [
        ('doubletalk',        'deg', 'DT deg'),
        ('farend_singletalk', 'echo', 'FS echo'),
    ]:
        sc = [r for r in rows if r['scenario'] == scenario]
        print(f"\n--- {sort_label} — top 10 cases where ours is worst vs each competitor ---")

        for competitor in ['linear', 'aec2', 'aec3']:
            gaps = []
            for r in sc:
                ours_val = r.get(f'ours_{metric_key}', '')
                comp_val = r.get(f'{competitor}_{metric_key}', '')
                if ours_val != '' and comp_val != '':
                    gap = float(comp_val) - float(ours_val)
                    if gap > 0:
                        gaps.append((gap, r))
            gaps.sort(key=lambda x: -x[0])
            worst10 = gaps[:10]
            if not worst10:
                continue
            mean_gap = np.mean([g for g, _ in gaps])
            print(f"\n  vs {competitor.upper()} ({sort_label} gap, mean={mean_gap:.3f}):")
            print(f"  {'file_id':38s}  ours    {competitor}  gap")
            for gap, r in worst10:
                fid = r['file_id']
                ours_v = r[f'ours_{metric_key}']
                comp_v = r[f'{competitor}_{metric_key}']
                mv = '(mv)' if 'with_movement' in fid else '    '
                print(f"  {fid[:36]:36s} {mv}  {float(ours_v):.3f}   {float(comp_v):.3f}  +{gap:.3f}")

    # Cases where ours beats ALL competitors
    print("\n" + "="*70)
    print("CASES where ours BEATS all 3 competitors (DT deg)")
    sc = [r for r in rows if r['scenario'] == 'doubletalk']
    wins = []
    for r in sc:
        ours = r.get('ours_deg', '')
        if ours == '': continue
        ours = float(ours)
        others = [r.get(f'{c}_deg', '') for c in ['linear','aec2','aec3']]
        others = [float(v) for v in others if v != '']
        if others and all(ours > o for o in others):
            min_margin = min(ours - o for o in others)
            wins.append((min_margin, r['file_id'], ours, others))
    wins.sort(key=lambda x: -x[0])
    print(f"  {len(wins)}/300 DT cases ours > all (avg margin top 5)")
    for margin, fid, ours, others in wins[:5]:
        print(f"  {fid[:36]:36s}  ours={ours:.3f}  others={[f'{o:.3f}' for o in others]}")


if __name__ == '__main__':
    main()
