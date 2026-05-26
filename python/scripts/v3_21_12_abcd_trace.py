#!/usr/bin/env python3
"""v3.21.12 — RefinedFilterUpdateGain input-parity audit A/B/C/D trace.

Variants (HPF policy intended: mic ON / ref OFF, both at config default):

    A : baseline (all v3.21.x parity flags OFF)
    B : use_partition_summed_x2_for_h_error_gain = True (v3.21.7 paused)
    C : use_current_e2_refined_in_h_error_denominator = True (NEW v3.21.12)
    D : both ON

Per-hop trace inside PBFDKF._update_weights_aec3 captures:
    e2_refined (time-domain block sum of raw_output²)
    y2 (time-domain block sum of near_end²)
    mu_lf / mu_mf / mu_hf (per-band mean of mu_aec3 array before W update)
    H_error_lf / mf / hf
    X2_lf / mf / hf (the X² actually fed to the denom; latest or summed)
    e2_current_lf / mf / hf (|error_spec|², current per-bin)
    error_psd_lf / mf / hf (smoothed EMA)

Per-case aggregates:
    refined_diverged_frames = count(hops where e2_refined > y2)
    refined_diverged_strong = count(hops where e2_refined > 1.5 * y2)
    nores LF/MF/HF energy ratio  (band energy of *_nores.wav vs mic)
"""
from __future__ import annotations
import argparse, os, sys, json, hashlib
from pathlib import Path
import numpy as np

PYDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYDIR))
import soundfile as sf
from aec import AEC, AecConfig
from modules.filters import PBFDKF

# Band boundaries at 16 kHz with rfft length 257 (fft=512):
#   bin_index = freq_hz × 512 / 16000.   LF=0-1k → bins 0-32, MF=1-4k → 32-128, HF=4-8k → 128-256.
LF_HI, MF_HI = 32, 128


_ORIG_UPDATE_WEIGHTS_AEC3 = PBFDKF._update_weights_aec3


def render_variant(mic, ref, partition, raw_e2):
    """Render one variant, capture trace + final output + nores output."""
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    cfg.enable_res = True; cfg.enable_cng = True
    cfg.use_partition_summed_x2_for_h_error_gain = partition
    cfg.use_current_e2_refined_in_h_error_denominator = raw_e2
    aec = AEC(cfg)
    # Mark main filter so the hook only logs main (not shadow if any).
    aec.filter._v3_21_12_is_main = True
    if aec.shadow_filter is not None:
        aec.shadow_filter._v3_21_12_is_main = False

    log = {
        'mu_lf': [], 'mu_mf': [], 'mu_hf': [],
        'H_err_lf': [], 'H_err_mf': [], 'H_err_hf': [],
        'X2_lf': [], 'X2_mf': [], 'X2_hf': [],
        'e2_inst_lf': [], 'e2_inst_mf': [], 'e2_inst_hf': [],
        'e_psd_lf': [], 'e_psd_mf': [], 'e_psd_hf': [],
        'e2_refined_time': [], 'y2_time': [],
        'refined_conv': [], 'coarse_conv': [],
    }
    orig = _ORIG_UPDATE_WEIGHTS_AEC3
    def hooked(self, curr_p, mu_scale_arr, error_psd, *args, **kwargs):
        if getattr(self, '_v3_21_12_is_main', False):
            X_latest = self.X_buf[curr_p]
            if self._use_partition_summed_x2_for_h_error_gain:
                X2 = (np.abs(self.X_buf) ** 2).sum(axis=0).astype(np.float32)
            else:
                X2 = (np.abs(X_latest) ** 2).astype(np.float32)
            e2_inst = (np.abs(self.error_spec) ** 2).astype(np.float32)
            e_psd = self._error_psd.astype(np.float32)
            H_err = self.H_error_per_bin.astype(np.float32)
            if self._use_current_e2_refined_in_h_error_denominator:
                e2_in_denom = e2_inst
            else:
                e2_in_denom = e_psd
            n_part = np.float32(self.n_partitions)
            delta32 = np.float32(self.delta)
            denom = (np.float32(0.5) * H_err * X2 + n_part * e2_in_denom + delta32)
            mu_arr = (H_err / denom).astype(np.float32)
            from modules import aec3_scale as _aec3
            ng = np.float32(_aec3.NOISE_GATE_POWER_FLOAT)
            mu_arr = np.where(X2 >= ng, mu_arr, np.float32(0.0))
            log['mu_lf'].append(float(mu_arr[:LF_HI].mean()))
            log['mu_mf'].append(float(mu_arr[LF_HI:MF_HI].mean()))
            log['mu_hf'].append(float(mu_arr[MF_HI:].mean()))
            log['H_err_lf'].append(float(H_err[:LF_HI].mean()))
            log['H_err_mf'].append(float(H_err[LF_HI:MF_HI].mean()))
            log['H_err_hf'].append(float(H_err[MF_HI:].mean()))
            log['X2_lf'].append(float(X2[:LF_HI].mean()))
            log['X2_mf'].append(float(X2[LF_HI:MF_HI].mean()))
            log['X2_hf'].append(float(X2[MF_HI:].mean()))
            log['e2_inst_lf'].append(float(e2_inst[:LF_HI].mean()))
            log['e2_inst_mf'].append(float(e2_inst[LF_HI:MF_HI].mean()))
            log['e2_inst_hf'].append(float(e2_inst[MF_HI:].mean()))
            log['e_psd_lf'].append(float(e_psd[:LF_HI].mean()))
            log['e_psd_mf'].append(float(e_psd[LF_HI:MF_HI].mean()))
            log['e_psd_hf'].append(float(e_psd[MF_HI:].mean()))
        return orig(self, curr_p, mu_scale_arr, error_psd, *args, **kwargs)
    PBFDKF._update_weights_aec3 = hooked

    hop = 160
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    # Process two ways: (1) full pipeline = ours.wav (with RES + CNG),
    # (2) linear-only = ours_nores.wav (call AEC with enable_res=False).
    # We render once with enable_res=True and capture per-hop raw_output as
    # nores companion (matches eval_aec_challenge.py behaviour).
    out_full = []
    out_nores = []
    for i in range(0, n - hop, hop):
        # Capture y2 and e2_refined per hop. Run AEC and also pull raw_output.
        m = mic[i:i+hop]; r = ref[i:i+hop]
        o = aec.process(m, r)
        out_full.append(o)
        if aec._last_raw_output is not None:
            out_nores.append(aec._last_raw_output.copy())
            log['y2_time'].append(float(np.sum(m.astype(np.float64) ** 2)))
            log['e2_refined_time'].append(float(np.sum(
                aec._last_raw_output.astype(np.float64) ** 2)))
        else:
            out_nores.append(np.zeros(hop, dtype=np.float32))
            log['y2_time'].append(0.0)
            log['e2_refined_time'].append(0.0)
    out_full = np.concatenate(out_full)
    out_nores = np.concatenate(out_nores)

    # Restore original hook
    PBFDKF._update_weights_aec3 = orig
    return out_full, out_nores, log


def band_energy(sig, sr=16000, hop=160):
    """Band energy via STFT magnitude squared, summed."""
    N = 512
    win = np.hanning(N).astype(np.float32)
    nf = (len(sig) - N) // hop + 1
    if nf <= 0:
        return {'lf': 0.0, 'mf': 0.0, 'hf': 0.0}
    lf = mf = hf = 0.0
    for f in range(nf):
        seg = sig[f*hop:f*hop+N] * win
        S = np.abs(np.fft.rfft(seg)) ** 2
        lf += float(S[:LF_HI].sum())
        mf += float(S[LF_HI:MF_HI].sum())
        hf += float(S[MF_HI:].sum())
    return {'lf': lf, 'mf': mf, 'hf': hf}


def summarize(case_name, variant_logs, mic, refmic_energy):
    """Print one row per variant for this case."""
    print(f'\n=== {case_name} ===')
    print(f"{'var':<3} | {'hops':>5} {'div%':>6} {'div_strong%':>11} | "
          f"{'mu_lf':>9} {'H_err_lf':>11} {'X2_lf':>9} {'e2_inst_lf':>11} {'e_psd_lf':>10} | "
          f"{'nores_lf/mic':>12} {'nores_mf/mic':>12} {'nores_hf/mic':>12}")
    for var, (log, _, nores) in variant_logs.items():
        hops = len(log['e2_refined_time'])
        e2 = np.array(log['e2_refined_time']); y2 = np.array(log['y2_time'])
        div_pct = 100.0 * (e2 > y2).sum() / max(hops, 1)
        div_strong = 100.0 * (e2 > 1.5 * y2).sum() / max(hops, 1)
        mu_lf = np.mean(log['mu_lf']) if log['mu_lf'] else 0.0
        H_lf = np.mean(log['H_err_lf']) if log['H_err_lf'] else 0.0
        X2_lf = np.mean(log['X2_lf']) if log['X2_lf'] else 0.0
        e2i_lf = np.mean(log['e2_inst_lf']) if log['e2_inst_lf'] else 0.0
        ep_lf = np.mean(log['e_psd_lf']) if log['e_psd_lf'] else 0.0
        nb = band_energy(nores)
        lf_r = nb['lf'] / max(refmic_energy['lf'], 1e-20)
        mf_r = nb['mf'] / max(refmic_energy['mf'], 1e-20)
        hf_r = nb['hf'] / max(refmic_energy['hf'], 1e-20)
        print(f"{var:<3} | {hops:>5} {div_pct:>5.1f}% {div_strong:>10.1f}% | "
              f"{mu_lf:>9.5f} {H_lf:>11.5f} {X2_lf:>9.5f} {e2i_lf:>11.5f} {ep_lf:>10.5f} | "
              f"{lf_r:>12.4f} {mf_r:>12.4f} {hf_r:>12.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort-dir',
                    default='wav/v3_21_8_cohort',
                    help='12-case cohort dir with doubletalk/farend_singletalk/nearend_singletalk subdirs')
    ap.add_argument('--cases', nargs='*',
                    help='specific stems to run (default: all in cohort)')
    args = ap.parse_args()

    base = Path(args.cohort_dir)
    cases = []
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        for mic_p in sorted((base / sub).glob('*_mic.wav')):
            stem = mic_p.name[:-len('_mic.wav')]
            if args.cases and not any(stem.startswith(c) for c in args.cases):
                continue
            ref_p = base / sub / f'{stem}_lpb.wav'
            cases.append((stem, mic_p, ref_p))
    print(f'Cohort: {len(cases)} cases')

    for stem, mic_p, ref_p in cases:
        mic, _ = sf.read(str(mic_p), dtype='float32')
        ref, _ = sf.read(str(ref_p), dtype='float32')
        # mic energy by band (for nores ratio denominator)
        mic_e = band_energy(mic)
        variants = {}
        for var, (partition, raw_e2) in [('A',(False,False)),('B',(True,False)),
                                          ('C',(False,True)),('D',(True,True))]:
            out_full, out_nores, log = render_variant(mic, ref, partition, raw_e2)
            variants[var] = (log, out_full, out_nores)
        summarize(stem, variants, mic, mic_e)


if __name__ == '__main__':
    main()
