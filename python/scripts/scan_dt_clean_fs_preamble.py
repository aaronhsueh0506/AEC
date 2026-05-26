"""Scan blind/doubletalk/ for cases with a clean farend-only preamble.

A "clean FS preamble" frame is one where:
  - lpb has speech-band (LF+MF) energy above a per-utterance threshold
  - mic LF+MF energy is dominated by echo (i.e. mic_energy is similar
    in magnitude to lpb-driven echo, not by an extra NE source).

Heuristic (no NE ground truth available):
  - utterance-level lpb thr = P40 of lpb_lf+mf
  - frame is "lpb-active" if lpb_lf+mf > lpb_thr
  - utterance-level mic-vs-lpb ratio over a sliding window of
    lpb-active frames at the start of the case. We look at the
    FIRST window (frames 0…W) where W = window_sec/0.01 = 500 for 5 s.
  - "Clean FS" if within first 5-15 s:
      * ≥ frac_active fraction of frames are lpb-active
      * AND DT onset (mic_lf+mf > mic_thr while NOT lpb-active) is
        rare (< 0.10) in the same window

Output: a table sorted by FS preamble duration (in seconds) descending,
restricted to cases with FS preamble ≥ 5 s. Top 8 reported.
"""
from __future__ import annotations

import os
import sys
import re
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DT_DIR = os.path.join(REPO, 'wav', 'aec_challenge_blind', 'doubletalk')

BLOCK = 160
SR = 16000
FFT_PSD = 512
N_FREQS = FFT_PSD // 2 + 1
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)


def stft_lm(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    nf = max(0, (n - FFT_PSD) // BLOCK + 1)
    out = np.empty(nf, dtype=np.float32)
    for i in range(nf):
        seg = x[i * BLOCK: i * BLOCK + FFT_PSD] * win
        a = np.abs(np.fft.rfft(seg)) ** 2
        out[i] = a[LF_SLICE].sum() + a[MF_SLICE].sum()
    return out


def scan_case(stem_path: str) -> dict | None:
    mic_p = stem_path + '_mic.wav'
    lpb_p = stem_path + '_lpb.wav'
    if not (os.path.exists(mic_p) and os.path.exists(lpb_p)):
        return None
    mic, sr_m = sf.read(mic_p)
    lpb, sr_l = sf.read(lpb_p)
    if sr_m != SR or sr_l != SR:
        return None
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    n = min(len(mic), len(lpb))
    if n < SR * 6:
        return None
    mic_lm = stft_lm(mic[:n])
    lpb_lm = stft_lm(lpb[:n])
    if len(mic_lm) == 0 or len(lpb_lm) == 0:
        return None

    nf = min(len(mic_lm), len(lpb_lm))
    mic_lm = mic_lm[:nf]
    lpb_lm = lpb_lm[:nf]

    # Per-utterance thresholds (NE+FS mixed → use percentiles on lpb only
    # for far activity; on mic compare relative to lpb)
    lpb_thr = float(np.percentile(lpb_lm[lpb_lm > 0], 40)) if np.any(lpb_lm > 0) else 0.0
    mic_thr = float(np.percentile(mic_lm[mic_lm > 0], 30)) if np.any(mic_lm > 0) else 0.0

    is_lpb = lpb_lm > lpb_thr
    is_mic = mic_lm > mic_thr

    # Sweep window sizes: 5s, 8s, 12s. Find the maximum start-window that's
    # "FS-clean" (≥ 70% lpb-active AND ≤ 10% "NE-only" frames).
    candidate_durations: list[float] = []
    for win_s in (5, 8, 10, 12, 15):
        win_n = int(win_s * SR / BLOCK)
        if win_n >= nf:
            continue
        seg_lpb = is_lpb[:win_n]
        seg_mic = is_mic[:win_n]
        frac_lpb = seg_lpb.mean()
        # NE-only proxy: mic active but lpb not active (over the same window)
        ne_only = (seg_mic & ~seg_lpb).mean()
        if frac_lpb >= 0.70 and ne_only <= 0.10:
            candidate_durations.append(win_s)

    if not candidate_durations:
        return None

    # Also compute median mic/lpb ratio over the lpb-active frames in the
    # best window — for echo-only frames, mic_lm should track lpb-energy
    # at some ratio determined by ERL (typical 0.01-1.0 of lpb energy).
    best_win = max(candidate_durations)
    best_win_n = int(best_win * SR / BLOCK)
    mask = is_lpb[:best_win_n]
    if not mask.any():
        return None
    ratio = (mic_lm[:best_win_n] / np.maximum(lpb_lm[:best_win_n], 1e-9))[mask]
    median_ratio = float(np.median(ratio))
    return {
        'stem_path': stem_path,
        'best_preamble_s': float(best_win),
        'frac_lpb_active': float(is_lpb[:best_win_n].mean()),
        'frac_ne_only': float((is_mic[:best_win_n] & ~is_lpb[:best_win_n]).mean()),
        'mic_over_lpb_median': median_ratio,
        'utterance_s': float(nf * BLOCK / SR),
    }


def main():
    # XRTnTUjU + jtYTdZm3 + 0I0XMl3M for context comparison (the latter is FS
    # not DT — included only when present).
    stems: list[str] = []
    pat = re.compile(r'(.+)_mic\.wav$')
    for f in sorted(os.listdir(DT_DIR)):
        m = pat.match(f)
        if not m:
            continue
        # Only "_doubletalk" (no movement) — DT_static
        stem = m.group(1)
        if not stem.endswith('_doubletalk'):
            continue
        stems.append(os.path.join(DT_DIR, stem))

    print(f'Scanning {len(stems)} DT_static cases under {DT_DIR}')
    results: list[dict] = []
    for i, sp in enumerate(stems):
        r = scan_case(sp)
        if r is not None:
            results.append(r)
        if (i + 1) % 50 == 0:
            print(f'  ... {i + 1}/{len(stems)} scanned, {len(results)} candidates so far')

    results.sort(key=lambda r: (-r['best_preamble_s'],
                                -r['frac_lpb_active'],
                                r['frac_ne_only']))
    print()
    print(f'Found {len(results)} DT_static cases with FS preamble ≥ 5 s. Top 20:')
    print(f'{"stem":56s} {"preamble_s":>10s} {"lpb_frac":>9s} {"ne_only":>8s} {"mic/lpb":>9s} {"utt_s":>7s}')
    for r in results[:20]:
        stem = Path(r['stem_path']).name
        print(f'{stem:56s} {r["best_preamble_s"]:>10.1f} {r["frac_lpb_active"]:>9.3f} '
              f'{r["frac_ne_only"]:>8.3f} {r["mic_over_lpb_median"]:>9.3e} {r["utterance_s"]:>7.1f}')


if __name__ == '__main__':
    main()
