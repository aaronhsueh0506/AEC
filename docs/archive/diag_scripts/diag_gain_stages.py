"""Per-stage gain trace at leak hotspots.

For each leak-hotspot (frame, bin) — where ours_psd >> aec2_psd in a
far-active frame — captures the per-bin gain at each downstream stage:

  01 softgate_emr  — Wiener soft-gate + EMR boost
  02 spectral_floor — spectral_g_min applied
  03 epc_dt_cap    — EPC_DT 0.85 cap
  04 quiet_mask    — quiet bins lifted to 1.0
  05 3bin_smooth   — 3-bin cross-frequency convolution
  06 hf_cap        — HF cap to ~500Hz bin gain
  07 pre_temporal  — divergence override applied
  08 post_temporal — temporal attack/release + rate limit + render_dt_ceil

This identifies WHICH stage is the binding constraint at echo leak bins —
the one that holds gain HIGH (not suppressed) despite earlier stages
suppressing it. That's the stage to attack with coh-gating.

Usage:
  python3 diag_gain_stages.py STEM [--top 5]
"""
import os, sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode


class CoherenceTracker:
    def __init__(self, n_freqs, K=16, alpha=0.95):
        self.K, self.alpha = K, alpha
        self._cross = np.zeros(n_freqs, dtype=np.complex64)
        self._mic_pwr = np.full(n_freqs, 1e-10, dtype=np.float32)
        self._lpb_pwr = np.full(n_freqs, 1e-10, dtype=np.float32)
        self._lpb_history = np.zeros((K, n_freqs), dtype=np.complex64)
    def update(self, mic_spec, lpb_spec):
        self._lpb_history[1:] = self._lpb_history[:-1]; self._lpb_history[0] = lpb_spec
        lpb_avg = self._lpb_history.mean(axis=0); a = self.alpha
        self._cross   = a*self._cross   + (1-a)*mic_spec*np.conj(lpb_avg)
        self._mic_pwr = a*self._mic_pwr + (1-a)*(np.abs(mic_spec)**2).astype(np.float32)
        self._lpb_pwr = a*self._lpb_pwr + (1-a)*(np.abs(lpb_avg)**2).astype(np.float32)
        coh = (np.abs(self._cross)**2) / (self._mic_pwr*self._lpb_pwr + 1e-12)
        return np.clip(coh, 0, 1).astype(np.float32)

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
AEC2_DIR = REPO / 'python/output_ref'

STAGE_KEYS = ['01_softgate_emr', '02_spectral_floor', '03_epc_dt_cap',
              '04_quiet_mask',  '05_3bin_smooth',    '06_hf_cap',
              '07_pre_temporal','08_post_temporal']


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _trace(stem, K_list=(16, 32, 64, 96)):
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
    aec.res._capture_stages = True
    hop = aec.hop_size; n_freqs = aec.res.n_freqs
    coh_trackers = {K: CoherenceTracker(n_freqs, K=K) for K in K_list}

    pos = 0; idx = 0
    out = np.zeros(n, dtype=np.float32)
    stages = {k: [] for k in STAGE_KEYS}
    coh_logs = {K: [] for K in K_list}
    far_active = []; using_render = []
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        nfft = 2 * (n_freqs - 1)
        m_sp = np.fft.rfft(mic_f, n=nfft)[:n_freqs].astype(np.complex64)
        l_sp = np.fft.rfft(lpb_f, n=nfft)[:n_freqs].astype(np.complex64)
        for K, ct in coh_trackers.items():
            coh_logs[K].append(ct.update(m_sp, l_sp).copy())
        o = aec.process(mic_f, lpb_f); out[pos:pos+hop] = o
        sg = aec.res._stage_gains
        for k in STAGE_KEYS:
            stages[k].append(sg.get(k, np.full(n_freqs, np.nan)).copy())
        far_active.append(float(np.mean(lpb_f**2)) > 1e-4)
        using_render.append(bool(aec.res._using_render_based))
        if not hasattr(aec, '_sat_log'):
            aec._sat_log = []
        aec._sat_log.append(float(getattr(aec, '_saturation_level', 0.0)))
        pos += hop; idx += 1

    return {
        'stem': stem, 'sub': sub, 'sr': sr, 'hop': hop, 'n_freqs': n_freqs,
        'mic': mic[:pos], 'output': out[:pos],
        'stages': {k: np.stack(v) for k, v in stages.items()},
        'cohs': {K: np.stack(v) for K, v in coh_logs.items()},
        'far_active': np.array(far_active),
        'using_render': np.array(using_render),
        'sat': np.array(aec._sat_log),
    }


def _frame_psds(sig, n_frames, hop, n_freqs):
    nfft = 2 * (n_freqs - 1)
    psds = []
    for i in range(n_frames):
        seg = sig[i*hop:(i+1)*hop]
        if len(seg) < hop: break
        psds.append(np.abs(np.fft.rfft(seg, n=nfft)[:n_freqs]) ** 2)
    return np.stack(psds)


def _analyze(t, aec2_signal, top_n=8):
    sr = t['sr']; hop = t['hop']; n_freqs = t['n_freqs']
    n_frames = t['stages']['01_softgate_emr'].shape[0]
    a2 = aec2_signal[:hop*n_frames]
    ours_psd = _frame_psds(t['output'], n_frames, hop, n_freqs)
    aec2_psd = _frame_psds(a2,           n_frames, hop, n_freqs)
    n_frames = ours_psd.shape[0]
    far_mask = t['far_active'][:n_frames]
    using_render = t['using_render'][:n_frames]
    sat = t['sat'][:n_frames]
    cohs = {K: c[:n_frames] for K, c in t['cohs'].items()}
    freqs = np.linspace(0, sr/2, n_freqs)

    leak = np.maximum(ours_psd - aec2_psd, 0)
    leak[~far_mask] = 0
    leak[:, :2] = 0  # ignore DC
    flat = np.argsort(leak.flatten())[-top_n:][::-1]
    fr, bn = np.unravel_index(flat, leak.shape)

    print(f'\n=== {t["stem"]} ({t["sub"]}, {n_frames} frames) ===')
    print(f'using_render fraction (far-active): '
          f'{using_render[far_mask].mean():.0%}')
    K_keys = sorted(cohs.keys())
    print(f'\n  Saturation level at hotspots: ' +
          ', '.join(f'{sat[f]:.2f}' for f in fr))
    print(f'  Saturation overall: mean={sat.mean():.3f}  '
          f'frac>0.3={(sat>0.3).mean():.0%}  frac>0.5={(sat>0.5).mean():.0%}')
    print(f'\n  Top-{top_n} leak hotspots (coh per K-window):')
    print(f'  {"#":>2}  {"fr":>5}  {"bin":>3}  {"freq":>5}  {"ratio":>9}  '
          + '  '.join(f'coh_K{K:<3d}' for K in K_keys)
          + '   final_g')
    for i, (f, b) in enumerate(zip(fr, bn)):
        ratio = ours_psd[f, b] / (aec2_psd[f, b] + 1e-12)
        coh_vals = [cohs[K][f, b] for K in K_keys]
        final_g = t['stages']['08_post_temporal'][f, b]
        print(f'  {i+1:>2}  {f:>5}  {b:>3}  {freqs[b]:>5.0f}  {ratio:>8.0f}× '
              + '  '.join(f'{c:>6.2f}  ' for c in coh_vals)
              + f' {final_g:>6.3f}')

    # Distribution of coh at hotspots vs random NE-likely frames
    print(f'\n  Coh distribution at hotspots vs random low-leak frames:')
    low_leak_mask = (leak < np.percentile(leak[leak>0], 30) if (leak>0).sum() else np.zeros_like(leak, bool))
    for K in K_keys:
        c = cohs[K]
        hotspot_coh = c[fr, bn]
        # NE-likely: NE bins are where ours_psd is small (we suppressed) and aec2 also small
        ne_mask = (ours_psd < np.percentile(ours_psd, 20)) & far_mask[:,None]
        ne_coh = c[ne_mask]
        print(f'    K={K:<3d}: hotspot coh mean={hotspot_coh.mean():.3f} median={np.median(hotspot_coh):.3f} '
              f'p25={np.percentile(hotspot_coh,25):.3f}    NE-bin coh mean={ne_coh.mean():.3f}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--top', type=int, default=8)
    args = ap.parse_args()
    for stem in args.stems:
        t = _trace(stem)
        a2_p = AEC2_DIR / f'{stem}_aec2.wav'
        if not a2_p.is_file():
            print(f'no AEC2 ref for {stem}'); continue
        sig, _ = sf.read(str(a2_p), dtype='float32')
        a2_sig = sig[:, 0] if sig.ndim > 1 else sig
        _analyze(t, a2_sig, top_n=args.top)


if __name__ == '__main__':
    main()
