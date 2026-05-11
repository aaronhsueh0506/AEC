"""Compute acoustic properties for all 800 cases; identify outlier dimension
for the A.0 regressing case `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`.

Per-case stats (mic + lpb only; no AEC required):
  - ERL_db: voice-band 10*log10(mic_pwr / lpb_pwr) over far-active frames
  - sat_frac: fraction of samples with |mic| > 0.95 (near-clipping)
  - mic_peak_db: 20*log10(max(|mic|))
  - mic_rms_db
  - lpb_rms_db
  - coh2_voice: mean voice-band MSC between mic & lpb
  - tail_decay_ratio_db: rough reverb tail proxy — ratio of post-active mic
    energy to active mic energy after the longest far-active segment ends
  - far_active_frac: fraction of frames with lpb_pwr > 1e-4

Output: CSV with per-case row.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path('/Users/mingyu/Desktop/novatek/SE/AEC')
DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
SR = 16000
HOP = 256
NFFT = 512


def discover():
    out = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir(): continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                out.append((stem, subset, str(mic), str(lpb)))
    return out


def stft(x, nfft=NFFT, hop=HOP):
    win = np.hanning(nfft).astype(np.float32)
    n = len(x)
    n_frames = max(0, (n - nfft) // hop + 1)
    S = np.empty((n_frames, nfft // 2 + 1), dtype=np.complex64)
    for i in range(n_frames):
        seg = x[i*hop : i*hop + nfft] * win
        S[i] = np.fft.rfft(seg).astype(np.complex64)
    return S


def _one(args):
    stem, subset, mic_p, lpb_p = args
    mic, _ = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic = mic[:n]
    lpb = lpb[:n]

    sat_frac = float(np.mean(np.abs(mic) > 0.95))
    mic_peak_db = 20 * np.log10(max(1e-8, float(np.max(np.abs(mic)))))
    mic_rms_db = 20 * np.log10(max(1e-8, float(np.sqrt(np.mean(mic ** 2)))))
    lpb_rms_db = 20 * np.log10(max(1e-8, float(np.sqrt(np.mean(lpb ** 2)))))

    M = stft(mic.astype(np.float32))
    L = stft(lpb.astype(np.float32))
    if len(M) == 0 or len(L) == 0:
        return None
    freqs = np.linspace(0, SR/2, NFFT//2 + 1)
    voice = (freqs >= 300) & (freqs <= 3000)
    far_pwr = np.mean(np.abs(L) ** 2, axis=1)
    mic_pwr = np.mean(np.abs(M) ** 2, axis=1)
    far_active = far_pwr > 1e-4
    far_active_frac = float(np.mean(far_active))

    if far_active.any():
        mic_active = np.sum(np.abs(M[far_active][:, voice]) ** 2, axis=1)
        lpb_active = np.sum(np.abs(L[far_active][:, voice]) ** 2, axis=1)
        erl_db = 10 * np.log10(np.mean(mic_active) / max(1e-12, np.mean(lpb_active)))
    else:
        erl_db = float('nan')

    # Voice-band MSC (mic vs lpb) over the recording
    alpha = 0.9
    n_frames = min(len(M), len(L))
    S_xy = np.zeros(NFFT//2 + 1, dtype=np.complex64)
    S_xx = np.zeros(NFFT//2 + 1, dtype=np.float32)
    S_yy = np.zeros(NFFT//2 + 1, dtype=np.float32)
    for i in range(n_frames):
        S_xy = alpha * S_xy + (1 - alpha) * (L[i] * np.conj(M[i]))
        S_xx = alpha * S_xx + (1 - alpha) * (np.abs(L[i]) ** 2)
        S_yy = alpha * S_yy + (1 - alpha) * (np.abs(M[i]) ** 2)
    coh2 = np.abs(S_xy) ** 2 / (S_xx * S_yy + 1e-12)
    coh2 = np.clip(coh2, 0, 1)
    coh2_voice = float(np.mean(coh2[voice]))

    # Tail decay proxy: find longest far-active run, then measure mic energy
    # in 100ms after far drops below threshold (relative to last 100ms active)
    tail_ratio_db = float('nan')
    if far_active.any():
        # Find changes
        diff = np.diff(far_active.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        if len(starts) and len(ends):
            run_lens = ends - starts
            longest = int(np.argmax(run_lens))
            s, e = int(starts[longest]), int(ends[longest])
            # 100ms = ~6.25 frames at hop=256
            win = 7
            if e - s >= win and e + win <= n_frames:
                e_active = float(np.mean(mic_pwr[e - win:e]))
                e_tail = float(np.mean(mic_pwr[e:e + win]))
                if e_active > 1e-12 and e_tail > 1e-12:
                    tail_ratio_db = 10 * np.log10(e_tail / e_active)

    return {
        'stem': stem, 'subset': subset,
        'duration_s': n / SR,
        'erl_db': float(erl_db),
        'sat_frac': sat_frac,
        'mic_peak_db': float(mic_peak_db),
        'mic_rms_db': float(mic_rms_db),
        'lpb_rms_db': float(lpb_rms_db),
        'far_active_frac': far_active_frac,
        'coh2_voice': coh2_voice,
        'tail_decay_db': tail_ratio_db,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('-j', type=int, default=4)
    args = ap.parse_args()

    cases = discover()
    print(f'cases={len(cases)} j={args.j}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.j) as p:
        rows = [r for r in p.imap_unordered(_one, cases) if r]
    if not rows: return
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {args.out} n={len(rows)}', flush=True)

if __name__ == '__main__':
    main()
