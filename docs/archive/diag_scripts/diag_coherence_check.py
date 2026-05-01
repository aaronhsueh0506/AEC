"""Verify whether long-history mic-vs-lpb coherence discriminates echo from NE.

Computes per-bin per-frame:
  - cross_spec EMA(mic_spec × conj(lpb_avg_K_ago))
  - mic_psd EMA, lpb_psd EMA
  - coh_history = |cross|² / (mic_pwr × lpb_pwr + ε)

Then at LEAK hotspots (ours >> aec2, far-active) — should be ECHO bins.
At OVER-SUPP hotspots (ours << aec2) — should be NE bins.

Compare distributions: if echo bins have HIGH coh_history and NE bins have
LOW, coherence is a viable discriminator. If overlapping, not viable.

Usage:
  python3 diag_coherence_check.py STEM [STEM ...]
"""
import os, sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
AEC2_DIR = REPO / 'python/output_ref'


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _trace(stem):
    np.random.seed(20260428)
    sub, mic_p, lpb_p = _resolve(stem)
    mic, sr = sf.read(str(mic_p), dtype='float32')
    lpb, _  = sf.read(str(lpb_p), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    is_mv = '_with_movement' in stem
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, **delay_kw)
    aec = AEC(cfg)
    hop = aec.hop_size; n_freqs = aec.res.n_freqs
    nfft = 2 * (n_freqs - 1)
    K = 16   # frames of lpb history (avg lpb over last K to use as 'far past')
    alpha = 0.95  # EMA TC ~50ms - 400ms depending on hop

    # Per-bin EMAs for coherence
    cross = np.zeros(n_freqs, dtype=np.complex64)
    mic_pwr = np.zeros(n_freqs, dtype=np.float32) + 1e-10
    lpb_pwr = np.zeros(n_freqs, dtype=np.float32) + 1e-10

    lpb_history = np.zeros((K, n_freqs), dtype=np.complex64)

    # Track per-frame metrics
    pos = 0; idx = 0
    out = np.zeros(n, dtype=np.float32)
    coh_per_frame = []
    far_active = []
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        # FFT
        mic_spec = np.fft.rfft(mic_f, n=nfft)[:n_freqs].astype(np.complex64)
        lpb_spec = np.fft.rfft(lpb_f, n=nfft)[:n_freqs].astype(np.complex64)
        # lpb history: average over past K frames (proxy for delayed RIR)
        lpb_history[1:] = lpb_history[:-1]
        lpb_history[0] = lpb_spec
        lpb_avg = lpb_history.mean(axis=0)
        # EMA cross-spectra and psds
        cross    = alpha * cross    + (1-alpha) * mic_spec * np.conj(lpb_avg)
        mic_pwr  = alpha * mic_pwr  + (1-alpha) * np.abs(mic_spec) ** 2
        lpb_pwr  = alpha * lpb_pwr  + (1-alpha) * np.abs(lpb_avg) ** 2
        coh = np.abs(cross) ** 2 / (mic_pwr * lpb_pwr + 1e-12)
        coh = np.clip(coh, 0.0, 1.0)
        coh_per_frame.append(coh.copy())
        far_active.append(float(np.mean(lpb_f**2)) > 1e-4)
        # Run AEC normally for output
        o = aec.process(mic_f, lpb_f); out[pos:pos+hop] = o
        pos += hop; idx += 1

    return {
        'stem': stem, 'sub': sub, 'sr': sr, 'hop': hop, 'n_freqs': n_freqs,
        'mic': mic[:pos], 'lpb': lpb[:pos], 'output': out[:pos],
        'coh': np.stack(coh_per_frame),
        'far_active': np.array(far_active),
    }


def _frame_psds(sig, n_frames, hop, n_freqs):
    nfft = 2 * (n_freqs - 1)
    psds = []
    for i in range(n_frames):
        seg = sig[i*hop:(i+1)*hop]
        if len(seg) < hop: break
        psds.append(np.abs(np.fft.rfft(seg, n=nfft)[:n_freqs]) ** 2)
    return np.stack(psds)


def _verify(t, aec2_signal, top_n=300):
    sr = t['sr']; hop = t['hop']; n_freqs = t['n_freqs']
    n_frames = len(t['coh'])
    a2 = aec2_signal[:hop*n_frames]
    mic_psd = _frame_psds(t['mic'], n_frames, hop, n_freqs)
    ours_psd = _frame_psds(t['output'], n_frames, hop, n_freqs)
    aec2_psd = _frame_psds(a2, n_frames, hop, n_freqs)
    n_frames = ours_psd.shape[0]
    coh = t['coh'][:n_frames]
    far_mask = t['far_active'][:n_frames]
    freqs = np.linspace(0, sr/2, n_freqs)

    print(f'\n=== {t["stem"]} ({t["sub"]}, {n_frames} frames) ===')
    print(f'far_active: {int(far_mask.sum())}/{n_frames}, coh stats overall: '
          f'mean={coh.mean():.3f}  p25={np.percentile(coh,25):.3f}  '
          f'p75={np.percentile(coh,75):.3f}')

    # LEAK hotspots: ours >> aec2 in far-active frames
    leak_excess = np.maximum(ours_psd - aec2_psd, 0)
    leak_excess[~far_mask] = 0
    flat = np.argsort(leak_excess.flatten())[-top_n:][::-1]
    leak_frames, leak_bins = np.unravel_index(flat, leak_excess.shape)
    leak_coh = coh[leak_frames, leak_bins]

    # OVER-SUPP hotspots: ours << aec2
    over = np.maximum(aec2_psd - ours_psd, 0)
    flat = np.argsort(over.flatten())[-top_n:][::-1]
    over_frames, over_bins = np.unravel_index(flat, over.shape)
    over_coh = coh[over_frames, over_bins]

    # Distribution comparison
    if leak_excess.sum() > 0:
        print(f'\n  LEAK hotspots (top {top_n}): coh_history distribution')
        print(f'    mean={leak_coh.mean():.3f}  p25={np.percentile(leak_coh,25):.3f}  '
              f'p50={np.percentile(leak_coh,50):.3f}  p75={np.percentile(leak_coh,75):.3f}')
        print(f'    by freq band:')
        for lo, hi in [(0,500),(500,2000),(2000,4000)]:
            bm = (freqs[leak_bins] >= lo) & (freqs[leak_bins] < hi)
            if bm.sum() > 5:
                print(f'      {lo:>5d}-{hi:<5d}Hz: n={bm.sum()} '
                      f'coh mean={leak_coh[bm].mean():.3f}  p50={np.percentile(leak_coh[bm],50):.3f}')

    if over.sum() > 0:
        print(f'\n  OVER-SUPP hotspots (top {top_n}): coh_history distribution')
        print(f'    mean={over_coh.mean():.3f}  p25={np.percentile(over_coh,25):.3f}  '
              f'p50={np.percentile(over_coh,50):.3f}  p75={np.percentile(over_coh,75):.3f}')
        print(f'    by freq band:')
        for lo, hi in [(0,500),(500,2000),(2000,4000)]:
            bm = (freqs[over_bins] >= lo) & (freqs[over_bins] < hi)
            if bm.sum() > 5:
                print(f'      {lo:>5d}-{hi:<5d}Hz: n={bm.sum()} '
                      f'coh mean={over_coh[bm].mean():.3f}  p50={np.percentile(over_coh[bm],50):.3f}')

    # Discriminative power
    if leak_excess.sum() > 0 and over.sum() > 0:
        sep = leak_coh.mean() - over_coh.mean()
        # Overlap of distributions: how many over-supp bins above leak p25?
        leak_p25 = np.percentile(leak_coh, 25)
        over_above = (over_coh > leak_p25).sum() / len(over_coh)
        print(f'\n  → Separation: leak_mean - over_mean = {sep:+.3f}')
        print(f'  → Overlap: {over_above*100:.0f}% of over-supp coh > leak p25 (lower=better)')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    args = ap.parse_args()
    for stem in args.stems:
        t = _trace(stem)
        a2 = AEC2_DIR / f'{stem}_aec2.wav'
        if not a2.is_file():
            print(f'no AEC2 ref for {stem}'); continue
        sig, _ = sf.read(str(a2), dtype='float32')
        a2_sig = sig[:, 0] if sig.ndim > 1 else sig
        _verify(t, a2_sig)


if __name__ == '__main__':
    main()
