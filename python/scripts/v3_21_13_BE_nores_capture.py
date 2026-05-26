#!/usr/bin/env python3
"""nores LF/MF/HF band-energy + use-capture rate for A/B/E/BE.

For each variant out_dir, computes per-case LF/MF/HF energy ratio of
`*_ours_nores.wav` to mic LF/MF/HF energy. Also re-runs E and BE in-process
to capture the v3_21_13_trace use-capture rate.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import soundfile as sf

PYDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYDIR))
from aec import AEC, AecConfig

# Bands at 16 kHz STFT (n_fft=512): LF 0-1k → bins 0-32, MF 1-4k → 32-128, HF 4-8k → 128-256
LF_HI, MF_HI = 32, 128

def band_energy(sig, hop=160, N=512):
    win = np.hanning(N).astype(np.float32)
    nf = (len(sig) - N) // hop + 1
    if nf <= 0:
        return dict(lf=0.0, mf=0.0, hf=0.0)
    lf = mf = hf = 0.0
    for f in range(nf):
        seg = sig[f*hop:f*hop+N] * win
        S = np.abs(np.fft.rfft(seg)) ** 2
        lf += float(S[:LF_HI].sum())
        mf += float(S[LF_HI:MF_HI].sum())
        hf += float(S[MF_HI:].sum())
    return dict(lf=lf, mf=mf, hf=hf)

# Cases
COHORT_DIR = Path('wav/v3_21_8_cohort')
case_list = []
for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
    for mic_p in sorted((COHORT_DIR / sub).glob('*_mic.wav')):
        stem = mic_p.name[:-len('_mic.wav')]
        ref_p = COHORT_DIR / sub / f'{stem}_lpb.wav'
        case_list.append((stem, sub, mic_p, ref_p))

# Variant render dirs
variants = {
    'A':  Path('out_12_v3_21_12_A'),
    'B':  Path('out_12_v3_21_12_B'),
    'E':  Path('out_12_v3_21_13'),
    'BE': Path('out_12_v3_21_13_BE'),
}

print('=== nores LF/mic per variant per case ===')
print(f'{"case":<42} | {"A":>8} {"B":>8} {"E":>8} {"BE":>8}')
print('-'*95)
agg = {v: {'lf': [], 'mf': [], 'hf': []} for v in variants}
for stem, sub, mic_p, ref_p in case_list:
    mic, _ = sf.read(str(mic_p), dtype='float32')
    mic_e = band_energy(mic)
    if mic_e['lf'] < 1e-20:
        continue
    short = stem[:40]
    row = []
    for v in 'A B E BE'.split():
        nores_path = variants[v] / f'{stem}_ours_nores.wav'
        if not nores_path.exists():
            row.append('  N/A   ')
            continue
        nores, _ = sf.read(str(nores_path), dtype='float32')
        be = band_energy(nores)
        lf_r = be['lf'] / mic_e['lf']
        mf_r = be['mf'] / mic_e['mf']
        hf_r = be['hf'] / mic_e['hf']
        agg[v]['lf'].append(lf_r)
        agg[v]['mf'].append(mf_r)
        agg[v]['hf'].append(hf_r)
        row.append(f'{lf_r:>8.4f}')
    print(f'{short:<42} | ' + ' '.join(row))

print('\n=== Mean nores band ratio per variant ===')
print(f'{"var":<3} | {"mean_lf":>9} {"mean_mf":>9} {"mean_hf":>9}')
for v in 'ABCDE BE'.split():  # placeholder loop labels
    pass
for v in 'A B E BE'.split():
    lf = np.mean(agg[v]['lf']) if agg[v]['lf'] else 0.0
    mf = np.mean(agg[v]['mf']) if agg[v]['mf'] else 0.0
    hf = np.mean(agg[v]['hf']) if agg[v]['hf'] else 0.0
    print(f'{v:<3} | {lf:>9.4f} {mf:>9.4f} {hf:>9.4f}')

# Use-capture rate for E and BE (re-run in-process)
print('\n=== Use-capture rate (E and BE) ===')
print(f'{"case":<42} | {"E_use_cap%":>11} {"BE_use_cap%":>12}')
print('-'*70)
def run_variant(mic, ref, partition, use_linear_select):
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    cfg.enable_res = True; cfg.enable_cng = True
    cfg.use_partition_summed_x2_for_h_error_gain = partition
    cfg.use_linear_filter_output_selection_for_final_output = use_linear_select
    aec = AEC(cfg)
    hop = 160
    n = min(len(mic), len(ref))
    for i in range(0, n - hop, hop):
        aec.process(mic[i:i+hop], ref[i:i+hop])
    return aec._v3_21_13_trace

for stem, sub, mic_p, ref_p in case_list:
    mic, _ = sf.read(str(mic_p), dtype='float32')
    ref, _ = sf.read(str(ref_p), dtype='float32')
    tE = run_variant(mic, ref, False, True)
    tBE = run_variant(mic, ref, True, True)
    E_pct = 100.0 * tE['frames_use_capture'] / max(tE['frames_total'], 1)
    BE_pct = 100.0 * tBE['frames_use_capture'] / max(tBE['frames_total'], 1)
    short = stem[:40]
    print(f'{short:<42} | {E_pct:>10.1f}% {BE_pct:>11.1f}%')
