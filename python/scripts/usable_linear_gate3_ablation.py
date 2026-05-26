"""Gate-3 ablation runner for usable_linear latch bug.

Runs XRTnTUjU_DT_static (stress) + XRTnTUjU_DT_movement (normal
counterpart) under partition_summed_x2 ON with 5 gate-3 variants:

  baseline_off  : partition_summed OFF (= legacy v3.21.6)
  V0_on_legacy  : partition_summed ON  + legacy gate-3 (the bug surface)
  V1_counter5   : partition_summed ON  + conv_hops_required=5
  V2_counter5_fa: partition_summed ON  + conv_hops_required=5 +
                  filter_analyzer_consistent AND  (+ filter_analyzer_enabled)
  V3_counter5_no_extdelay
                : partition_summed ON  + conv_hops_required=5 +
                  external_delay shortcut DISABLED
  V4_no_extdelay_only
                : partition_summed ON  + only external_delay shortcut
                  DISABLED (debug — does dropping ext_delay alone help?)

For each variant, we render the case and capture per-frame
`hf_chain` trace (usable_linear, refined_conv, dominant_nearend,
gain_30/50/100/200 etc.). The damaged-window aggregates are
reported in tables.

Output dir: /tmp/be_partitionsum/gate3_ablation/
"""
from __future__ import annotations

import os
import sys
import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'python'))

from aec import AEC, AecConfig, __version__   # noqa: E402
from eval_aec_challenge import estimate_delay   # noqa: E402

BLOCK = 160
SR = 16000
FFT_PSD = 512
N_FREQS = FFT_PSD // 2 + 1
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)

CASES = [
    ('XRTnTUjU5kS0mejzCqyCiw_doubletalk', 'doubletalk', False,
     [(239, 263), (763, 765)]),
    ('XRTnTUjU5kS0mejzCqyCiw_doubletalk_with_movement', 'doubletalk', True,
     [(239, 263), (763, 765)]),
]

CONFIGS = [
    # name, partition_summed, conv_hops, fa_and, no_extdelay, filter_analyzer_enabled
    ('baseline_off',          False, 0, False, False, False),
    ('V0_on_legacy',          True,  0, False, False, False),
    ('V1_counter5',           True,  5, False, False, False),
    ('V2_counter5_fa',        True,  5, True,  False, True),
    ('V3_counter5_no_ext',    True,  5, False, True,  False),
    ('V4_no_extdelay_only',   True,  0, False, True,  False),
]

OUT_DIR = '/tmp/be_partitionsum/gate3_ablation'


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


def render(stem: str, bucket: str, movement: bool, cfg_tuple: tuple) -> dict:
    name, partition_summed, conv_hops, fa_and, no_ext, fa_en = cfg_tuple
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    cfg.enable_highpass_ref = False
    cfg.enable_cng = True
    cfg.enable_res = True
    cfg.use_partition_summed_x2_for_h_error_gain = bool(partition_summed)
    cfg.usable_linear_convergence_hops_required = int(conv_hops)
    cfg.usable_linear_require_filter_analyzer_consistent = bool(fa_and)
    cfg.usable_linear_disable_external_delay_shortcut = bool(no_ext)
    cfg.filter_analyzer_enabled = bool(fa_en)
    cfg.trace_hf_chain = True

    case_dir = os.path.join(REPO, 'wav', 'aec_challenge_blind', bucket)
    mic_path = os.path.join(case_dir, f'{stem}_mic.wav')
    ref_path = os.path.join(case_dir, f'{stem}_lpb.wav')
    mic, _ = sf.read(mic_path); mic = (mic if mic.ndim == 1 else mic[:, 0]).astype(np.float32)
    ref, _ = sf.read(ref_path); ref = (ref if ref.ndim == 1 else ref[:, 0]).astype(np.float32)
    n0 = min(len(mic), len(ref))
    delay = estimate_delay(mic[:n0], ref[:n0], SR)
    if 0 < delay < n0:
        ref_a = np.zeros(n0, dtype=np.float32)
        ref_a[delay:] = ref[:n0 - delay]
        ref = ref_a
    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n]; ref = ref[:n]

    aec = AEC(cfg)
    out_ours = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_ours[s:s + BLOCK] = aec.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
    hf_chain = list(aec._hf_chain_trace)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_wav = os.path.join(OUT_DIR, f'{stem}_{name}_ours.wav')
    sf.write(out_wav, out_ours.astype(np.float32), SR)
    return {
        'config': name,
        'partition_summed': partition_summed,
        'conv_hops_required': conv_hops,
        'fa_and': fa_and,
        'no_extdelay': no_ext,
        'fa_enabled': fa_en,
        'hf_chain': hf_chain,
        'wav_path': out_wav,
        'ours_lm': stft_lm(out_ours),
        'mic_lm': stft_lm(mic),
        'lpb_lm': stft_lm(ref),
    }


def agg_window(hf: list[dict], lo: int, hi: int) -> dict:
    n = len(hf)
    lo = max(0, min(lo, n - 1))
    hi = max(lo, min(hi, n - 1))
    sel = list(range(lo, hi + 1))
    def frac(k):
        vals = [bool(hf[i].get(k, False)) for i in sel]
        return sum(vals) / max(len(vals), 1)
    def mean(k):
        vals = [float(hf[i].get(k, 0.0)) for i in sel]
        return sum(vals) / max(len(vals), 1)
    return {
        'window': f'{lo}-{hi}', 'n': hi - lo + 1,
        'usable_linear': frac('usable_linear'),
        'refined_conv': frac('refined_conv'),
        'coarse_conv': frac('coarse_conv'),
        'aec3_converged': frac('aec3_converged'),
        'is_nearend_state': frac('is_nearend_state'),
        'transparent_mode_active': frac('transparent_mode_active'),
        'filter_analyzer_consistent': frac('filter_analyzer_consistent'),
        'gain_30': mean('gain_30'),
        'gain_50': mean('gain_50'),
        'gain_100': mean('gain_100'),
        'gain_200': mean('gain_200'),
        'r2_to_s2': mean('r2_to_s2_ratio'),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report: list[str] = []
    report.append('# usable_linear gate-3 ablation — XRTnTUjU')
    report.append('')
    report.append(f'AEC __version__ = {__version__}')
    report.append('intended HPF policy (mic ON / ref OFF). preset=balanced.')
    report.append('')
    all_runs: dict = {}
    for stem, bucket, mv, windows in CASES:
        report.append(f'## Case `{stem}`  (movement={mv})')
        report.append('')
        # Run all configs
        runs = []
        for cfg in CONFIGS:
            r = render(stem, bucket, mv, cfg)
            runs.append(r)
            print(f'  [{stem}][{cfg[0]}] done — '
                  f'frames={len(r["hf_chain"])}', flush=True)
        all_runs[stem] = runs
        # Whole-utterance summary first
        report.append('### Whole-utterance usable_linear frac True')
        report.append('')
        report.append('| config | usable_linear | refined_conv | gain_100_mean | mean(ours_lm) |')
        report.append('|---|---:|---:|---:|---:|')
        for r in runs:
            agg = agg_window(r['hf_chain'], 0, len(r['hf_chain']) - 1)
            mean_ours = float(np.mean(r['ours_lm']))
            report.append(f'| `{r["config"]}` | {agg["usable_linear"]:.3f} | {agg["refined_conv"]:.3f} | '
                          f'{agg["gain_100"]:.3e} | {mean_ours:.3e} |')
        report.append('')
        # Per-window
        for w_lo, w_hi in windows:
            report.append(f'### Window {w_lo}-{w_hi}')
            report.append('')
            # Audio context (use baseline_off mic/lpb)
            base = runs[0]
            seg_mic = base['mic_lm'][w_lo:w_hi + 1] if w_hi < len(base['mic_lm']) else base['mic_lm'][w_lo:]
            seg_lpb = base['lpb_lm'][w_lo:w_hi + 1] if w_hi < len(base['lpb_lm']) else base['lpb_lm'][w_lo:]
            report.append(f'  audio mean (LF+MF):  mic={seg_mic.mean():.2e}  lpb={seg_lpb.mean():.2e}')
            report.append('')
            report.append('| config | usable_lin | refined_conv | is_NE | gain_30 | gain_100 | ours_lm |')
            report.append('|---|---:|---:|---:|---:|---:|---:|')
            for r in runs:
                agg = agg_window(r['hf_chain'], w_lo, w_hi)
                seg_ours = r['ours_lm'][w_lo:w_hi + 1] if w_hi < len(r['ours_lm']) else r['ours_lm'][w_lo:]
                ours_m = float(np.mean(seg_ours)) if len(seg_ours) else 0.0
                report.append(f'| `{r["config"]}` | {agg["usable_linear"]:.3f} | {agg["refined_conv"]:.3f} | '
                              f'{agg["is_nearend_state"]:.3f} | {agg["gain_30"]:.3e} | {agg["gain_100"]:.3e} | '
                              f'{ours_m:.3e} |')
            report.append('')

    md_path = os.path.join(OUT_DIR, 'gate3_ablation_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(report))
    print(f'\nWrote {md_path}')


if __name__ == '__main__':
    main()
