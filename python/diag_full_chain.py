"""Full control-chain trace: leak + over-suppression hotspots.

Per frame dumps:
  STAGE 1 INPUT:  mic_psd, lpb_psd, far_psd
  STAGE 2 FILTER: echo_psd, error_psd, raw_err_pwr, erle_inst, filter_converged
  STAGE 3 RESIDUAL: residual_after_attribute, _echo_cap, _error_cap, _dt_cap, _render_ceil
  STAGE 4 NEAREND: nearend_est, min_ne_from_dt, enr
  STAGE 5 GAIN:   raw_g (Wiener), gain_after_smoothing (Axis 2 cap), gain_after_NF (full final)
  STAGE 6 OUTPUT: output_psd

Compare each (frame, bin) vs AEC2:
  - leak_db = +∞: ours_psd >> aec2_psd → echo leaks
  - over_supp_db = −∞: ours_psd << aec2_psd → NE killed

Identifies top-N each, dumps full state at those points.

Usage:
  python3 diag_full_chain.py STEM [--mode leak|over_supp|both]
"""
import os, sys, json
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
AEC2_DIR = REPO / 'python/output_ref'
OUT = Path(__file__).parent / 'output_full_chain'


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
    if aec.res: aec.res.enable_stats()
    hop = aec.hop_size; n_freqs = aec.res.n_freqs

    pos = 0; idx = 0
    out = np.zeros(n, dtype=np.float32)
    # Per-frame metrics (mean over bins)
    far_active = []; using_render = []; once_conv = []; dt_combined = []
    erle_inst = []; erl_est = []
    res_attr = []; res_echo_cap = []; res_error_cap = []; res_dt_cap = []; res_render_ceil = []
    res_final = []; nearend = []; min_ne = []; enr = []
    gain_smooth_mean = []; gain_after_nf_mean = []
    # Per-frame, per-bin gain
    gains_after_smooth = []  # before NF restoration
    gains_final = []         # after NF restoration (final gain_smooth)
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        o = aec.process(mic_f, lpb_f); out[pos:pos+hop] = o
        s = aec._aec_state
        far_active.append(float(np.mean(lpb_f**2)) > 1e-4)
        using_render.append(bool(aec.res._using_render_based))
        once_conv.append(bool(s.filter_once_converged))
        dt_combined.append(float(s.dt_combined))
        erle_inst.append(float(aec._diag.get('erle_inst', 0)))
        erl_est.append(getattr(aec.res, '_stats_last_erl_estimate', 0))
        res_attr.append(getattr(aec.res, '_stats_last_res_after_attribute', 0))
        res_echo_cap.append(getattr(aec.res, '_stats_last_res_after_echo_cap', 0))
        res_error_cap.append(getattr(aec.res, '_stats_last_res_after_error_cap', 0))
        res_dt_cap.append(getattr(aec.res, '_stats_last_res_after_dt_cap', 0))
        res_render_ceil.append(getattr(aec.res, '_stats_last_res_after_render_ceil', 0))
        res_final.append(getattr(aec.res, '_stats_last_res_psd', 0))
        nearend.append(getattr(aec.res, '_stats_last_nearend', 0))
        min_ne.append(getattr(aec.res, '_stats_last_min_ne', 0))
        enr.append(getattr(aec.res, '_stats_last_enr', 0))
        gain_smooth_mean.append(getattr(aec.res, '_stats_last_gain_after_smoothing', 0))
        gain_after_nf_mean.append(float(np.mean(aec.res.gain_smooth)))
        gains_final.append(aec.res.gain_smooth.copy())
        pos += hop; idx += 1

    return {
        'stem': stem, 'sub': sub, 'sr': sr, 'hop': hop, 'n_freqs': n_freqs,
        'mic': mic[:pos], 'lpb': lpb[:pos], 'output': out[:pos],
        'gains_final': np.stack(gains_final),  # (T, F)
        'far_active': np.array(far_active),
        'using_render': np.array(using_render),
        'once_conv': np.array(once_conv),
        'dt_combined': np.array(dt_combined),
        'erle_inst': np.array(erle_inst),
        'erl_est': np.array(erl_est),
        'res_attr': np.array(res_attr),
        'res_error_cap': np.array(res_error_cap),
        'res_render_ceil': np.array(res_render_ceil),
        'res_final': np.array(res_final),
        'nearend': np.array(nearend),
        'min_ne': np.array(min_ne),
        'enr': np.array(enr),
        'gain_smooth_mean': np.array(gain_smooth_mean),
        'gain_after_nf_mean': np.array(gain_after_nf_mean),
    }


def _frame_psds(sig, n_frames, hop, n_freqs):
    nfft = 2 * (n_freqs - 1)
    psds = []
    for i in range(n_frames):
        seg = sig[i*hop:(i+1)*hop]
        if len(seg) < hop: break
        psds.append(np.abs(np.fft.rfft(seg, n=nfft)[:n_freqs]) ** 2)
    return np.stack(psds)


def _analyze(t, aec2_signal, top_n=200):
    sr = t['sr']; hop = t['hop']; n_freqs = t['n_freqs']
    n_frames = len(t['gains_final'])
    a2 = aec2_signal[:hop*n_frames]
    mic_psd = _frame_psds(t['mic'], n_frames, hop, n_freqs)
    ours_psd = _frame_psds(t['output'], n_frames, hop, n_freqs)
    aec2_psd = _frame_psds(a2, n_frames, hop, n_freqs)
    n_frames = ours_psd.shape[0]
    gains = t['gains_final'][:n_frames]
    far_mask = t['far_active'][:n_frames]
    freqs = np.linspace(0, sr/2, n_freqs)
    band_edges = [(0,500), (500,2000), (2000,4000)]

    # log ratio dB
    log_ratio_db = 10*np.log10((ours_psd + 1e-12) / (aec2_psd + 1e-12))

    # Energy-weighted leak (positive) and over-suppression (negative)
    leak_excess = np.maximum(ours_psd - aec2_psd, 0); leak_excess[~far_mask] = 0
    over_supp = np.maximum(aec2_psd - ours_psd, 0)  # NE-killing applies anywhere

    print(f'\n=== {t["stem"]} ({t["sub"]}, {n_frames} frames) ===')
    print(f'far_active: {int(far_mask.sum())}/{n_frames}, '
          f'render-mode%: {100*t["using_render"][:n_frames].mean():.0f}, '
          f'once_conv%: {100*t["once_conv"][:n_frames].mean():.0f}')
    total_leak = leak_excess.sum(); total_over = over_supp.sum()
    print(f'total LEAK energy: {total_leak:.2e}  total OVER-SUPP energy: {total_over:.2e}')

    # ---- LEAK hotspots (ours >> aec2) ----
    if total_leak > 0:
        flat_idx = np.argsort(leak_excess.flatten())[-top_n:][::-1]
        frames, bins = np.unravel_index(flat_idx, leak_excess.shape)
        print(f'\n  LEAK hotspot freq distribution (top {top_n}):')
        for lo, hi in band_edges:
            bm = (freqs[bins] >= lo) & (freqs[bins] < hi)
            n_hits = bm.sum(); e_hits = leak_excess[frames[bm], bins[bm]].sum()
            if n_hits > 0:
                print(f'    {lo:>5d}-{hi:<5d}Hz : {n_hits:>4d}  energy {e_hits:.2e} ({100*e_hits/total_leak:.0f}%)')
        print('  top-3 LEAK (state @ frame, bin):')
        for k in range(min(3, len(frames))):
            f, b = frames[k], bins[k]
            db = 10*np.log10(ours_psd[f,b]/(aec2_psd[f,b]+1e-15))
            print(f'    [{k}] frame={f} bin={b} ({freqs[b]:.0f}Hz)  ours={ours_psd[f,b]:.2e} aec2={aec2_psd[f,b]:.2e}  Δ={db:+.1f}dB')
            print(f'         dt={t["dt_combined"][f]:.2f} render={t["using_render"][f]} '
                  f'once_conv={t["once_conv"][f]} ERL={t["erl_est"][f]:.2f} erle={t["erle_inst"][f]:.1f}dB')
            print(f'         res: attr={t["res_attr"][f]:.2e} '
                  f'after_err_cap={t["res_error_cap"][f]:.2e} '
                  f'after_render_ceil={t["res_render_ceil"][f]:.2e} '
                  f'final={t["res_final"][f]:.2e}')
            print(f'         nearend={t["nearend"][f]:.2e} min_ne={t["min_ne"][f]:.2e} enr={t["enr"][f]:.3f}')
            print(f'         gain_smooth(pre-NF)={t["gain_smooth_mean"][f]:.3f} '
                  f'gain_final[{b}]={gains[f,b]:.3f} '
                  f'gain_final_mean={t["gain_after_nf_mean"][f]:.3f}')

    # ---- OVER-SUPP hotspots (ours << aec2) ----
    if total_over > 0:
        flat_idx = np.argsort(over_supp.flatten())[-top_n:][::-1]
        frames, bins = np.unravel_index(flat_idx, over_supp.shape)
        print(f'\n  OVER-SUPP hotspot freq distribution (top {top_n}):')
        for lo, hi in band_edges:
            bm = (freqs[bins] >= lo) & (freqs[bins] < hi)
            n_hits = bm.sum(); e_hits = over_supp[frames[bm], bins[bm]].sum()
            if n_hits > 0:
                print(f'    {lo:>5d}-{hi:<5d}Hz : {n_hits:>4d}  energy {e_hits:.2e} ({100*e_hits/total_over:.0f}%)')
        print('  top-3 OVER-SUPP (state @ frame, bin):')
        for k in range(min(3, len(frames))):
            f, b = frames[k], bins[k]
            db = 10*np.log10(ours_psd[f,b]/(aec2_psd[f,b]+1e-15))
            print(f'    [{k}] frame={f} bin={b} ({freqs[b]:.0f}Hz)  ours={ours_psd[f,b]:.2e} aec2={aec2_psd[f,b]:.2e}  Δ={db:+.1f}dB')
            print(f'         far_active={far_mask[f]} dt={t["dt_combined"][f]:.2f} '
                  f'render={t["using_render"][f]} once_conv={t["once_conv"][f]}  '
                  f'mic={mic_psd[f,b]:.2e}')
            print(f'         res_attr={t["res_attr"][f]:.2e} res_final={t["res_final"][f]:.2e} '
                  f'nearend={t["nearend"][f]:.2e} enr={t["enr"][f]:.3f}')
            print(f'         gain_final[{b}]={gains[f,b]:.3f} gain_pre_nf={t["gain_smooth_mean"][f]:.3f}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='+')
    ap.add_argument('--top-n', type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for stem in args.stems:
        t = _trace(stem)
        a2 = AEC2_DIR / f'{stem}_aec2.wav'
        if not a2.is_file():
            print(f'no AEC2 ref for {stem}'); continue
        sig, _ = sf.read(str(a2), dtype='float32')
        a2_sig = sig[:, 0] if sig.ndim > 1 else sig
        _analyze(t, a2_sig, top_n=args.top_n)


if __name__ == '__main__':
    main()
