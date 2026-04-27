#!/usr/bin/env python3
"""Phase 1 instrumentation: per-frame diagnostic trace for AEC tail-case analysis.

Runs our AEC on a list of top-loser files, captures get_diagnostics() every
frame, writes CSV per file. Also writes per-frame mic / far / output / aec2
spectral-band energies for downstream compare_to_aec2.py.

Usage:
    python3 gen_trace.py --cases /tmp/top_losers.txt --out-dir /tmp/traces

Each line of --cases: `scenario/prefix` e.g.
    farend_singletalk/JteZUZ4JYkeD4k2rcVbqHg_farend_singletalk
"""
import argparse, csv, os, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import AEC, AecConfig, AecMode, AecPreset
from eval_aec_challenge import estimate_delay

WAV_ROOT = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind'
AEC2_ROOT = '/Users/mingyu/Desktop/novatek/SE/AEC/python/output_v25'


def _band_energies(spec, sr):
    """Return (lf, mf, hf) mean-power in bands 0.5-2k, 2-4k, 4-8k."""
    n = len(spec)
    freq_per_bin = sr / (2.0 * (n - 1))
    b05 = max(1, int(500.0 / freq_per_bin))
    b2 = max(b05 + 1, int(2000.0 / freq_per_bin))
    b4 = max(b2 + 1, int(4000.0 / freq_per_bin))
    b8 = min(n, int(8000.0 / freq_per_bin) + 1)
    pwr = np.abs(spec) ** 2
    return (float(np.mean(pwr[b05:b2])),
            float(np.mean(pwr[b2:b4])),
            float(np.mean(pwr[b4:b8])))


def _stft_mag(sig, n_fft=1024, hop=160):
    """Return per-frame rfft magnitude spectrum (T, n_fft/2+1)."""
    n = len(sig)
    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for i in range(0, n - n_fft + 1, hop):
        f = np.fft.rfft(sig[i:i + n_fft] * win)
        frames.append(f)
    return np.array(frames)


def run_trace(mic_path, lpb_path, out_csv, sr=16000, fl=448,
              preset=AecPreset.BALANCED, aec2_path=None):
    mic, _ = sf.read(mic_path)
    ref, _ = sf.read(lpb_path)
    mic = mic.astype(np.float32)
    ref = ref.astype(np.float32)
    n = min(len(mic), len(ref))

    # Pre-align ref
    delay = estimate_delay(mic, ref, sr)
    if 0 < delay < n:
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
    else:
        ref_aligned = ref[:n]
    mic = mic[:n]

    is_movement = '_with_movement' in os.path.basename(mic_path)
    delay_est_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25,
                         delay_est_init_s=0.2)
                    if is_movement else dict(enable_delay_est=False))

    cfg = AecConfig.from_preset(
        preset, sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=fl, enable_dtd=False,
        enable_shadow=True, enable_res=True,
        use_kalman=True, **delay_est_kw,
    )
    aec = AEC(cfg)
    hop = aec.hop_size

    # Load AEC2 output for band-energy comparison (optional)
    aec2 = None
    if aec2_path and os.path.exists(aec2_path):
        aec2, _ = sf.read(aec2_path)
        aec2 = aec2[:n].astype(np.float32)

    out = np.zeros(n, dtype=np.float32)
    rows = []
    pos = 0
    frame_idx = 0
    while pos + hop <= n:
        m_hop = mic[pos:pos + hop]
        r_hop = ref_aligned[pos:pos + hop]
        o_hop = aec.process(m_hop, r_hop)
        out[pos:pos + hop] = o_hop
        diag = aec.get_diagnostics()

        # Spectral energies for comparison
        mic_spec = np.fft.rfft(m_hop * np.hanning(hop))
        far_spec = np.fft.rfft(r_hop * np.hanning(hop))
        our_spec = np.fft.rfft(o_hop * np.hanning(hop))
        mic_lf, mic_mf, mic_hf = _band_energies(mic_spec, sr)
        far_lf, far_mf, far_hf = _band_energies(far_spec, sr)
        our_lf, our_mf, our_hf = _band_energies(our_spec, sr)
        if aec2 is not None:
            a2_spec = np.fft.rfft(aec2[pos:pos + hop] * np.hanning(hop))
            a2_lf, a2_mf, a2_hf = _band_energies(a2_spec, sr)
        else:
            a2_lf = a2_mf = a2_hf = 0.0

        rows.append({
            'frame': frame_idx, 't_s': pos / sr,
            # Filter estimates
            'erle_inst': diag.get('erle_inst', 0.0),
            'erle_windowed': diag.get('erle_windowed', 0.0),
            'erle_factor': diag.get('erle_factor', 0.0),
            'erl_estimate': diag.get('erl_estimate', 0.0),
            'echo_psd_mean': diag.get('echo_psd_mean', 0.0),
            'error_psd_mean': diag.get('error_psd_mean', 0.0),
            'filter_w_norm': diag.get('filter_w_norm', 0.0),
            'shadow_w_norm': diag.get('shadow_w_norm', 0.0),
            # DT signals
            'dt_indicator': diag.get('dt_indicator', 0.0),
            'dt_from_energy': diag.get('dt_from_energy', 0.0),
            'dt_from_shadow': diag.get('dt_from_shadow', 0.0),
            'mu_scale': diag.get('mu_scale', 1.0),
            'dt_residual_scale': diag.get('dt_residual_scale', 1.0),
            # Shadow filter state
            'shadow_advantage': diag.get('shadow_advantage', 1.0),
            'main_err_smooth': diag.get('main_err_smooth', 0.0),
            'shadow_err_smooth': diag.get('shadow_err_smooth', 0.0),
            'copy_err_baseline': diag.get('copy_err_baseline', 0.0),
            # Echo path variability
            'epv_gain_ratio': diag.get('epv_gain_ratio', 1.0),
            'divergence': diag.get('divergence', 0.0),
            # Mode flags
            'converged': int(diag.get('converged', False)),
            'epc_active': int(diag.get('epc_active', False)),
            'main_paused': int(diag.get('main_paused', False)),
            'using_render': int(diag.get('using_render_based', False)),
            # RES gains
            'gain_mean': diag.get('res_gain_mean', 0.0),
            'gain_min': diag.get('res_gain_min', 0.0),
            'effective_g_min': diag.get('effective_g_min', 0.0),
            # Signal levels
            'far_activity': diag.get('far_activity', 0.0),
            'saturation_level': diag.get('saturation_level', 0.0),
            # Band energies (dB) for spectrum comparison
            'mic_lf_dB': 10 * np.log10(mic_lf + 1e-20),
            'mic_mf_dB': 10 * np.log10(mic_mf + 1e-20),
            'mic_hf_dB': 10 * np.log10(mic_hf + 1e-20),
            'far_lf_dB': 10 * np.log10(far_lf + 1e-20),
            'our_lf_dB': 10 * np.log10(our_lf + 1e-20),
            'our_mf_dB': 10 * np.log10(our_mf + 1e-20),
            'our_hf_dB': 10 * np.log10(our_hf + 1e-20),
            'aec2_lf_dB': 10 * np.log10(a2_lf + 1e-20),
            'aec2_mf_dB': 10 * np.log10(a2_mf + 1e-20),
            'aec2_hf_dB': 10 * np.log10(a2_hf + 1e-20),
        })
        pos += hop
        frame_idx += 1

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', required=True,
                        help='text file, one "scenario/prefix" per line')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--filter', type=int, default=448)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cases) as f:
        cases = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

    for case in cases:
        scenario, prefix = case.split('/', 1)
        mic_path = f'{WAV_ROOT}/{scenario}/{prefix}_mic.wav'
        lpb_path = f'{WAV_ROOT}/{scenario}/{prefix}_lpb.wav'
        aec2_path = f'{AEC2_ROOT}/{prefix}_old_aec.wav'
        out_csv = f'{args.out_dir}/trace_{prefix}.csv'
        if not os.path.exists(mic_path):
            print(f'SKIP missing: {mic_path}')
            continue
        out_wav, n_frames = run_trace(mic_path, lpb_path, out_csv,
                                      fl=args.filter, aec2_path=aec2_path)
        # Also save output wav for listening
        sf.write(f'{args.out_dir}/ours_{prefix}.wav', out_wav, 16000)
        print(f'{prefix}: {n_frames} frames → {out_csv}')


if __name__ == '__main__':
    main()
