"""Identify (time, bin) leak hotspots vs AEC2 + dump full pipeline state.

For each worst case:
  1. Frame mic / lpb / our output / AEC2 output → per-bin power
  2. Find top-N (frame, bin) where ours_psd / aec2_psd >> 1 (the leak hotspots)
  3. For those specific points, dump:
     - input: mic, lpb, far_psd
     - filter: echo_psd, error_psd, raw_output_psd
     - residual: residual_echo_psd (per bin if possible), nearend_est, min_ne
     - gain pipeline: enr, raw gain, smoothed, after Axis2 cap, after NF, final
     - downstream: enhanced_spec_psd, CNG contrib (if any), output_psd
  4. Compare per-bin gain to (output_psd / spec_synth_psd) — verify gain delivered
  5. Find the FIRST stage where ours diverges from "what AEC2 produces" —
     that is the bottleneck (downstream caps/floors that neutralize)

Usage:
  python3 diag_leak_hotspot.py STEM [--top-n 200] [--band lo|mid|hi]
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
OUT = Path(__file__).parent / 'output_leak_hotspot'


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
                                use_kalman=True,
                                enable_cng=os.environ.get('AEC_CNG','1')!='0',
                                **delay_kw)
    aec = AEC(cfg)
    if aec.res: aec.res.enable_stats()
    hop = aec.hop_size
    n_freqs = aec.res.n_freqs

    # Run; capture per-frame per-bin gain
    pos = 0; idx = 0
    out = np.zeros(n, dtype=np.float32)
    gains = []   # (T, F) per-bin gain_smooth (final)
    far_active = []
    using_render = []
    dt_combined = []
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        o = aec.process(mic_f, lpb_f); out[pos:pos+hop] = o
        gains.append(aec.res.gain_smooth.copy())
        far_active.append(float(np.mean(lpb_f**2)) > 1e-4)
        using_render.append(bool(aec.res._using_render_based))
        dt_combined.append(float(aec._aec_state.dt_combined))
        pos += hop; idx += 1

    return {
        'stem': stem, 'sub': sub, 'sr': sr, 'hop': hop, 'n_freqs': n_freqs,
        'gains': np.stack(gains),
        'far_active': np.array(far_active),
        'using_render': np.array(using_render),
        'dt_combined': np.array(dt_combined),
        'mic': mic[:pos], 'lpb': lpb[:pos], 'output': out[:pos],
    }


def _analyze(t, aec2_signal, top_n=200):
    sr = t['sr']; hop = t['hop']; n_freqs = t['n_freqs']
    nfft = 2 * (n_freqs - 1)
    n_frames = len(t['gains'])
    # Frame all four signals (mic, lpb, ours, aec2)
    a2 = aec2_signal[:hop*n_frames]
    mic = t['mic'][:hop*n_frames]
    lpb = t['lpb'][:hop*n_frames]
    ours = t['output'][:hop*n_frames]

    def fpsd(sig):
        psds = []
        for i in range(n_frames):
            seg = sig[i*hop:(i+1)*hop]
            if len(seg) < hop: break
            psds.append(np.abs(np.fft.rfft(seg, n=nfft)[:n_freqs])**2)
        return np.stack(psds)

    mic_psd = fpsd(mic)
    lpb_psd = fpsd(lpb)
    ours_psd = fpsd(ours)
    aec2_psd = fpsd(a2)
    n_frames = ours_psd.shape[0]
    gains = t['gains'][:n_frames]
    fa = t['far_active'][:n_frames]
    ur = t['using_render'][:n_frames]
    dtc = t['dt_combined'][:n_frames]

    freqs = np.linspace(0, sr/2, n_freqs)

    # Restrict hotspot search to far-active frames (where echo can leak)
    far_mask = fa
    leak_ratio_db = 10*np.log10((ours_psd + 1e-12) / (aec2_psd + 1e-12))
    # Also weight by absolute energy (low-energy bins matter less)
    # leak_score = (ours_psd - aec2_psd) clipped at 0, in linear
    leak_excess = np.maximum(ours_psd - aec2_psd, 0)
    # Mask non-far-active frames out
    leak_excess[~far_mask] = 0

    # Top-N (frame, bin) by leak_excess (energy units)
    flat_idx = np.argsort(leak_excess.flatten())[-top_n:][::-1]
    frames, bins = np.unravel_index(flat_idx, leak_excess.shape)

    print(f'\n=== {t["stem"]} ({t["sub"]}, {n_frames} frames) ===')
    # Total leak energy distribution
    total_leak = leak_excess.sum()
    far_active_n = far_mask.sum()
    print(f'far-active: {far_active_n}/{n_frames}, total leak energy: {total_leak:.2e}')

    # Hotspot summary by band
    band_edges = [(0,500), (500,2000), (2000,4000), (4000, sr//2)]
    print(f'\n  hotspot freq distribution (top {top_n}):')
    for lo, hi in band_edges:
        bm = (freqs[bins] >= lo) & (freqs[bins] < hi)
        n_hits = bm.sum()
        e_hits = leak_excess[frames[bm], bins[bm]].sum()
        print(f'    {lo:>5d}-{hi:<5d} Hz : {n_hits:>4d} hits ({100*n_hits/top_n:>3.0f}%)  '
              f'energy {e_hits:.2e} ({100*e_hits/total_leak:.0f}%)')

    # Time distribution: bucket into 10 segments
    print(f'\n  hotspot time distribution (top {top_n}, by 10% segments):')
    seg_size = n_frames // 10
    for s in range(10):
        s0, s1 = s*seg_size, (s+1)*seg_size if s < 9 else n_frames
        sm = (frames >= s0) & (frames < s1)
        n_hits = sm.sum()
        if n_hits > 0:
            print(f'    {s*10:>3d}-{(s+1)*10:>3d}% (frames {s0}-{s1}) : {n_hits:>4d} hits')

    # For top-5 hotspots, dump full state + past lpb at SAME bin
    print(f'\n  top-5 hotspots (frame, bin, freq, ours/aec2 dB):')
    for k in range(5):
        f = frames[k]; b = bins[k]
        ours_v = ours_psd[f, b]; aec2_v = aec2_psd[f, b]
        if aec2_v < 1e-15: aec2_v = 1e-15
        db = 10*np.log10(ours_v / aec2_v)
        print(f'    [{k}] frame={f:<5d}  bin={b:<3d}  freq={freqs[b]:>5.0f}Hz  '
              f'ours={ours_v:.2e} aec2={aec2_v:.2e}  Δ={db:+.1f}dB')
        print(f'         dt_comb={dtc[f]:.2f}  gain_smooth[{b}]={gains[f, b]:.3f}  '
              f'mic={mic_psd[f,b]:.2e} lpb_now={lpb_psd[f,b]:.2e}')
        # Past 50 frames lpb at SAME bin: peak / max time offset
        past_lpb = lpb_psd[max(0,f-50):f+1, b]
        if len(past_lpb) > 0:
            peak = past_lpb.max(); peak_off = len(past_lpb) - 1 - past_lpb.argmax()
            past_50_mean = past_lpb.mean()
            print(f'         lpb_past_50f[bin]: peak={peak:.2e} @ -{peak_off}f ({peak_off*1000//50}ms ago)  '
                  f'mean={past_50_mean:.2e}  (peak/now={peak/(lpb_psd[f,b]+1e-15):.1f}x)')

    # Effective vs intended gain check.
    # output = gain * (mic_via_OLA + CNG) — we can compute "delivered gain^2" =
    # ours_psd / mic_psd. If gain_smooth^2 ≈ delivered gain^2, no downstream issue.
    # Else downstream (CNG, OLA window, etc.) is changing things.
    print(f'\n  effective gain check (DT × far frames):')
    df_mask = far_mask & (dtc > 0.3)
    if df_mask.sum() > 5:
        eff_gain_sq = ours_psd[df_mask] / (mic_psd[df_mask] + 1e-12)
        intended_sq = gains[df_mask] ** 2
        # Mean per band
        for lo, hi in [(0,500), (500,2000), (2000,4000)]:
            fmask = (freqs >= lo) & (freqs < hi)
            eg = np.sqrt(np.mean(eff_gain_sq[:, fmask]))
            ig = np.sqrt(np.mean(intended_sq[:, fmask]))
            ratio_db = 20*np.log10(eg / max(ig, 1e-6))
            print(f'    {lo:>5d}-{hi:<5d} Hz  intended_gain={ig:.3f}  effective={eg:.3f}  '
                  f'(eff/intended = {ratio_db:+.1f}dB)')


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
