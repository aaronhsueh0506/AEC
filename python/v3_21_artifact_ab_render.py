#!/usr/bin/env python3
"""v3.21 linear-filter artifact A/B render.

Renders `_ours.wav` + `_ours_nores.wav` for a single (mic, ref) pair under
multiple candidate flag combinations so you can visually / aurally compare
linear-filter over-estimation / nores artifact reduction across variants.

Variants:
  v3.21.6        — baseline (all v3.21 alignment flags OFF)
  M_full_delay   — Bundle A+B+C+D + delay chain (all 13 flags ON)
  M_full         — M_full_delay without delay chain (Bundle A+B+C only)

Usage:
    python3 python/v3_21_artifact_ab_render.py <mic.wav> <ref.wav> [-o out_dir]

For each variant, outputs:
    <out_dir>/<stem>_<variant>_ours.wav        — full output (linear + res + CNG)
    <out_dir>/<stem>_<variant>_ours_nores.wav  — linear-only (no res, no CNG)
    <out_dir>/<stem>_<variant>_nores_spec.png  — nores spectrogram

Plus one combined PNG:
    <out_dir>/<stem>_AB_compare.png  — mic / ref / nores(v3.21.6) / nores(candidates)

No 800-case. No commit. No production change. Diagnostic only.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset


_FULL_DELAY_FLAGS = {
    'use_partition_summed_x2_for_h_error_gain': True,
    'use_current_e2_refined_in_h_error_denominator': True,
    'use_per_bin_h_error_refresh': True,
    'use_aec3_h_error_ceil': True,
    'use_aec3_filter_noise_gate_power': True,
    'use_partition_summed_x2_for_shadow_mu': True,
    'use_aec3_noise_gate_for_shadow': True,
    'use_poor_excitation_gate_for_shadow': True,
    'use_narrowband_mask_for_shadow': True,
    'use_saturation_gate_for_shadow': True,
    'use_refined_output_selection_for_linear_path': True,
    'form_linear_filter_crossfade_enabled': True,
    'use_full_delay_change_chain': True,
}
_FULL_FLAGS = dict(_FULL_DELAY_FLAGS)
_FULL_FLAGS['use_full_delay_change_chain'] = False

VARIANTS: Dict[str, Dict[str, bool]] = {
    'v3216':       {},                                       # baseline
    'M_full':      _FULL_FLAGS,                              # 12 flags, no delay chain
    'M_full_delay': _FULL_DELAY_FLAGS,                       # 13 flags inc delay chain
}


def load_pair(mic_p: Path, ref_p: Path):
    mic, sr = sf.read(str(mic_p))
    ref, _  = sf.read(str(ref_p))
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    n = min(len(mic), len(ref))
    return mic[:n].astype(np.float32), ref[:n].astype(np.float32), int(sr)


def render(mic: np.ndarray, ref: np.ndarray, flags: Dict[str, bool],
           enable_res: bool) -> np.ndarray:
    cfg = AecConfig.from_preset(AecPreset.BALANCED, **flags)
    cfg.enable_res = enable_res
    cfg.enable_cng = enable_res  # CNG only when res ON
    np.random.seed(42)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = len(mic) // hop
    out = np.zeros(n * hop, dtype=np.float32)
    for i in range(n):
        out[i*hop:(i+1)*hop] = aec.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop]
        )
    return out


def spectrogram(ax, sig: np.ndarray, sr: int, title: str,
                 nfft: int = 512, noverlap: int = 384,
                 vmin: float = -90.0, vmax: float = -10.0):
    if len(sig) < nfft:
        ax.set_title(f'{title} (too short)')
        ax.axis('off')
        return
    ax.specgram(sig, NFFT=nfft, Fs=sr, noverlap=noverlap,
                cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel('Hz')
    ax.set_ylim(0, sr // 2)


def main():
    parser = argparse.ArgumentParser(description='Linear-filter nores artifact A/B render')
    parser.add_argument('mic_wav', type=Path)
    parser.add_argument('ref_wav', type=Path)
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('out_artifact_ab'))
    parser.add_argument('--no-png', action='store_true',
                        help='Skip spectrogram PNG generation (faster)')
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    mic, ref, sr = load_pair(args.mic_wav, args.ref_wav)
    stem = args.mic_wav.stem.replace('_mic', '')
    print(f'Loaded {stem}: n={len(mic)} samples @ {sr} Hz ({len(mic)/sr:.1f}s)')
    print()

    out_files: Dict[str, Dict[str, np.ndarray]] = {}
    for vk, flags in VARIANTS.items():
        print(f'=== Variant: {vk} ({len(flags)} flags ON) ===')
        out_ours = render(mic, ref, flags, enable_res=True)
        out_nores = render(mic, ref, flags, enable_res=False)
        out_files[vk] = {'ours': out_ours, 'nores': out_nores}
        # Write wavs
        ours_p  = args.output_dir / f'{stem}_{vk}_ours.wav'
        nores_p = args.output_dir / f'{stem}_{vk}_ours_nores.wav'
        sf.write(str(ours_p),  out_ours,  sr, subtype='FLOAT')
        sf.write(str(nores_p), out_nores, sr, subtype='FLOAT')
        # Quick stats
        nores_rms = float(np.sqrt(np.mean(out_nores.astype(np.float64) ** 2)))
        nores_peak = float(np.max(np.abs(out_nores)))
        print(f'  wrote {nores_p.name}  rms={nores_rms:.4f}  peak={nores_peak:.4f}')

    # Comparison PNG
    if not args.no_png:
        n_var = len(VARIANTS)
        fig, axes = plt.subplots(2 + n_var, 1, figsize=(14, 2.2 * (2 + n_var)),
                                  sharex=False)
        spectrogram(axes[0], mic, sr, f'{stem} — MIC (Ch1)')
        spectrogram(axes[1], ref, sr, f'{stem} — REF (Ch2)')
        for i, vk in enumerate(VARIANTS):
            spectrogram(axes[2 + i], out_files[vk]['nores'], sr,
                         f'{stem} — NORES ({vk})')
        png_p = args.output_dir / f'{stem}_AB_compare.png'
        plt.tight_layout()
        plt.savefig(str(png_p), dpi=100)
        plt.close(fig)
        print()
        print(f'Comparison PNG: {png_p}')

    # Pairwise nores delta-RMS (LF / MF / HF)
    print()
    print('Nores band-energy comparison (M_full / M_full_delay vs v3216):')
    for vk in ['M_full', 'M_full_delay']:
        if vk not in out_files: continue
        for band, lo, hi in [('LF', 0, 500), ('MF', 500, 2000), ('HF', 2000, sr // 2)]:
            base = _band_rms(out_files['v3216']['nores'], sr, lo, hi)
            cand = _band_rms(out_files[vk]['nores'], sr, lo, hi)
            db = 20.0 * np.log10(max(cand, 1e-10) / max(base, 1e-10))
            print(f'  {vk:14s}  {band}  v3216={base:.4f}  {vk}={cand:.4f}  Δ={db:+.2f} dB')
        print()


def _band_rms(sig: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """RMS energy within a frequency band via FFT."""
    n = len(sig)
    if n == 0:
        return 0.0
    spec = np.fft.rfft(sig.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mask = (freqs >= lo) & (freqs < hi)
    # Parseval-style band energy → time-RMS equivalent
    band_energy = float(np.sum(np.abs(spec[mask]) ** 2)) / (n / 2)
    return float(np.sqrt(band_energy / max(int(mask.sum()), 1)))


if __name__ == '__main__':
    main()
