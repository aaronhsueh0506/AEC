#!/usr/bin/env python3
"""Evaluate DNSMOS (SIG/BAK/OVL/P808) on AEC output files."""
import sys, os, argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, '/Users/mingyu/Desktop/novatek/SE/.venv/lib/python3.9/site-packages')
from speechmos import dnsmos


def main():
    parser = argparse.ArgumentParser(description='DNSMOS evaluation')
    parser.add_argument('output_dir', help='Directory with _pipeline.wav or _ours.wav files')
    parser.add_argument('--suffix', default='pipeline', help='File suffix to match (pipeline, ours)')
    args = parser.parse_args()

    out_dir = args.output_dir
    suffix = f"_{args.suffix}.wav"

    files = sorted([f for f in os.listdir(out_dir) if f.endswith(suffix)])
    if not files:
        print(f"No files matching *{suffix} in {out_dir}")
        return

    # Group by scenario
    scenarios = {}
    for f in files:
        if 'farend_singletalk' in f:
            sc = 'FS'
        elif 'nearend_singletalk' in f:
            sc = 'NE'
        elif 'doubletalk' in f:
            sc = 'DT'
        else:
            sc = 'OTHER'
        scenarios.setdefault(sc, []).append(f)

    for sc in ['FS', 'NE', 'DT']:
        if sc not in scenarios:
            continue
        sc_files = scenarios[sc]
        sigs, baks, ovrls, p808s = [], [], [], []

        for i, f in enumerate(sc_files):
            audio, sr = sf.read(os.path.join(out_dir, f), dtype='float32')
            if audio.ndim > 1:
                audio = audio[:, 0]
            result = dnsmos.run(audio, sr=sr)
            sigs.append(result['sig_mos'])
            baks.append(result['bak_mos'])
            ovrls.append(result['ovrl_mos'])
            p808s.append(result['p808_mos'])
            if (i + 1) % 50 == 0:
                print(f"  [{sc}] {i+1}/{len(sc_files)}", file=sys.stderr)

        print(f"\n{'='*60}")
        print(f"  {sc} — DNSMOS ({len(sc_files)} cases)")
        print(f"{'='*60}")
        print(f"  SIG:  {np.mean(sigs):.3f}")
        print(f"  BAK:  {np.mean(baks):.3f}")
        print(f"  OVL:  {np.mean(ovrls):.3f}")
        print(f"  P808: {np.mean(p808s):.3f}")


if __name__ == '__main__':
    main()
