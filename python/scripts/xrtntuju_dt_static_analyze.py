"""Offline analyzer for XRTnTUjU paired trace.

Loads A_off and B_on trace NPZ + WAVs. Re-runs damaged-frame selection
with a NE-preservation–oriented score (NOT the artifact-blow-up score
the first pass used), then produces a refined attribution report.

Damaged-frame score (NE-preservation):
  - Restrict to "NE-active" frames: mic_lf+mf > P50 of utterance
  - Exclude A-artifact-blow-up frames: A_ours_lf+mf < 3 × mic_lf+mf
    (avoids picking the LF-artifact frames where A is already broken)
  - Score: log10(max(mic_lf+mf, eps) / max(B_ours_lf+mf, eps))
    Higher score ⇒ B suppresses NE more strongly relative to mic.
  - Tie-break: same score sorted by (A_ours_lf+mf − B_ours_lf+mf) desc.

For attribution we additionally compute lpb_lf+mf to distinguish:
  - DT-active frames (lpb_lf+mf above its P25): both NE and FS active.
  - NE-only frames (lpb_lf+mf below P25): mostly NE.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'python'))

OUT_DIR = '/tmp/be_partitionsum/xrtntuju_trace'
STEM = 'XRTnTUjU5kS0mejzCqyCiw_doubletalk'

BLOCK = 160
SR = 16000
FFT_PSD = 512
N_FREQS = FFT_PSD // 2 + 1
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)


def stft_psd(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    nf = max(0, (n - FFT_PSD) // BLOCK + 1)
    out = np.empty((nf, N_FREQS), dtype=np.float32)
    for i in range(nf):
        seg = x[i * BLOCK: i * BLOCK + FFT_PSD] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def load_npz(label: str):
    p = os.path.join(OUT_DIR, f'{STEM}_{label}_trace.npz')
    z = dict(np.load(p, allow_pickle=False))
    return z


def main():
    A = load_npz('A_off')
    B = load_npz('B_on')

    # lpb STFT (one-time)
    mic_p = os.path.join(REPO, 'wav', 'aec_challenge_blind', 'doubletalk',
                         f'{STEM}_mic.wav')
    lpb_p = os.path.join(REPO, 'wav', 'aec_challenge_blind', 'doubletalk',
                         f'{STEM}_lpb.wav')
    mic, _ = sf.read(mic_p); mic = (mic if mic.ndim == 1 else mic[:, 0]).astype(np.float32)
    lpb, _ = sf.read(lpb_p); lpb = (lpb if lpb.ndim == 1 else lpb[:, 0]).astype(np.float32)
    lpb_psd = stft_psd(lpb)
    lpb_lm = lpb_psd[:, LF_SLICE].sum(axis=1) + lpb_psd[:, MF_SLICE].sum(axis=1)

    nf = min(len(A['mic_lf']), len(B['mic_lf']), len(lpb_lm))
    mic_lm = A['mic_lf'][:nf] + A['mic_mf'][:nf]
    Ao_lm = A['ours_lf'][:nf] + A['ours_mf'][:nf]
    Bo_lm = B['ours_lf'][:nf] + B['ours_mf'][:nf]
    An_lm = A['nores_lf'][:nf] + A['nores_mf'][:nf]
    Bn_lm = B['nores_lf'][:nf] + B['nores_mf'][:nf]
    lpb_lm = lpb_lm[:nf]

    eps = 1e-9
    mic_thr = float(np.percentile(mic_lm, 50))
    # 1. NE-active: mic_lf+mf > median
    mask_ne_active = mic_lm > mic_thr
    # 2. Exclude A artifact blow-up frames
    mask_no_artifact = Ao_lm < 3.0 * mic_lm
    # 3. Damaged: B suppresses NE more than A does
    sel_mask = mask_ne_active & mask_no_artifact
    score = np.log10(np.maximum(mic_lm, eps) / np.maximum(Bo_lm, eps))
    score = np.where(sel_mask, score, -np.inf)
    top = np.argsort(score)[-30:][::-1].tolist()

    # FS-only vs DT classification per frame
    lpb_thr = float(np.percentile(lpb_lm[lpb_lm > 0], 25)) if np.any(lpb_lm > 0) else 0.0
    is_dt = lpb_lm > lpb_thr

    print('NE-active threshold (mic_lf+mf P50):', mic_thr)
    print('lpb-active threshold (lpb_lf+mf P25):', lpb_thr)
    n_ne_active = int(mask_ne_active.sum())
    n_no_art = int(mask_no_artifact.sum())
    n_dual = int(sel_mask.sum())
    print(f'N frames NE-active: {n_ne_active} / {nf} '
          f'(no-artifact: {n_no_art}, joint: {n_dual})')

    # Build report
    lines: list[str] = []
    lines.append('# XRTnTUjU DT_static — NE-preservation–oriented damaged frame trace')
    lines.append('')
    lines.append('Selector: mic_lf+mf > P50  AND  A_ours_lf+mf < 3× mic_lf+mf')
    lines.append('Score:    log10(mic_lf+mf / B_ours_lf+mf)  — high ⇒ B suppresses NE more')
    lines.append('')
    lines.append(f'Joint mask N = {n_dual} / {nf} frames')
    lines.append('')

    # Per-frame table
    lines.append('## Top 30 NE-suppressed frames (B vs A)')
    lines.append('')
    lines.append('| frame | DT? | mic_lm | A_ours_lm | B_ours_lm | A_nores_lm | B_nores_lm | log10(mic/B_ours) | log10(A_ours/B_ours) |')
    lines.append('|---:|:---:|---:|---:|---:|---:|---:|---:|---:|')
    for i in top:
        d = 'Y' if is_dt[i] else 'n'
        ratio_mb = float(np.log10(max(mic_lm[i], eps) / max(Bo_lm[i], eps)))
        ratio_ab = float(np.log10(max(Ao_lm[i], eps) / max(Bo_lm[i], eps)))
        lines.append(f'| {i} | {d} | {mic_lm[i]:.2e} | {Ao_lm[i]:.2e} | {Bo_lm[i]:.2e} | '
                     f'{An_lm[i]:.2e} | {Bn_lm[i]:.2e} | {ratio_mb:+.2f} | {ratio_ab:+.2f} |')
    lines.append('')

    # Time-window summary
    sel_idx = np.array(top)
    sel_times = sel_idx * (BLOCK / SR)
    lines.append(f'Time spread of damaged frames: '
                 f'min={sel_times.min():.2f}s max={sel_times.max():.2f}s')
    n_dt = int(is_dt[sel_idx].sum())
    n_ne = len(sel_idx) - n_dt
    lines.append(f'DT-active damaged frames: {n_dt} / {len(sel_idx)}  '
                 f'(NE-only damaged frames: {n_ne})')
    lines.append('')

    # Linear vs RES attribution on these refined frames
    ours_d = sum(Ao_lm[i] - Bo_lm[i] for i in top)
    nores_d = sum(An_lm[i] - Bn_lm[i] for i in top)
    lines.append('## Attribution on NE-suppression damaged frames')
    lines.append('')
    lines.append(f'Sum over damaged frames (A_lf+mf − B_lf+mf):')
    lines.append(f'  ours_diff  = {ours_d:.3e}')
    lines.append(f'  nores_diff = {nores_d:.3e}  ({(nores_d/ours_d*100.0 if ours_d!=0 else 0):.1f}% of ours)')
    lines.append('')
    if ours_d == 0:
        verdict = 'No damage to attribute.'
    else:
        pct = nores_d / ours_d * 100.0
        if pct < 25.0:
            verdict = ('Damage primarily downstream of PBFDKF — '
                       'RES / SuppressionGain over-suppresses NE.')
        elif pct >= 75.0:
            verdict = ('Damage primarily inside PBFDKF — '
                       'linear filter eats NE during W update.')
        else:
            verdict = ('Damage mixed: linear residual differs and RES '
                       'amplifies the residual into NE damage.')
    lines.append(f'  → {verdict}')
    lines.append('')

    # PBFDKF-internal at refined damaged frames
    cols = ('mu_lf', 'mu_mf', 'dW_lf_l2', 'dW_mf_l2', 'W_lf_l2', 'W_mf_l2',
            'X2_latest_lf', 'X2_summed_lf', 'H_lf', 'denom_lf', 'denom_mf',
            'noise_gate_zero_frac_lf', 'noise_gate_zero_frac_mf',
            'e2_refined_sum_post', 'e2_coarse_sum')
    lines.append('## PBFDKF-internal — mean over refined damaged frames')
    lines.append('')
    lines.append('| field | A_off | B_on | B/A |')
    lines.append('|---|---:|---:|---:|')
    for c in cols:
        if c not in A: continue
        va = float(np.mean([A[c][i] for i in top]))
        vb = float(np.mean([B[c][i] for i in top]))
        r = vb / max(abs(va), 1e-30) if va != 0 else float('inf')
        lines.append(f'| {c} | {va:.3e} | {vb:.3e} | {r:.3f} |')
    lines.append('')

    # Sanity: how do the ablation outputs look at THESE refined frames vs
    # the original (artifact-blow-up) damaged frames? Quick band-by-band.
    lines.append('## Whole-utterance summary')
    lines.append('')
    for band, n_a, n_b in [('lf', A['ours_lf'], B['ours_lf']),
                            ('mf', A['ours_mf'], B['ours_mf']),
                            ('hf', A['ours_hf'], B['ours_hf'])]:
        ta = float(np.sum(n_a))
        tb = float(np.sum(n_b))
        lines.append(f'  ours total {band.upper()}: A={ta:.3e}  B={tb:.3e}  B/A={tb/max(ta,1e-30):.3f}')
    lines.append('')

    md = '\n'.join(lines)
    md_path = os.path.join(OUT_DIR, 'trace_diff_report_ne.md')
    with open(md_path, 'w') as f:
        f.write(md)
    print(f'\nWrote {md_path}')


if __name__ == '__main__':
    main()
