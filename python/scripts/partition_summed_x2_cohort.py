"""AEC3 `RefinedFilterUpdateGain` partition-summed X² parity — cohort A/B.

User directive 2026-05-22 (post HPF lock):
  - Intended HPF policy = mic ON / ref OFF (see project_aec_hpf_lock.md);
    every render here overrides `enable_highpass_ref=False`.
  - Production default for `use_partition_summed_x2_for_h_error_gain`
    stays False until 800-case AECMOS/Pareto gate clears.
  - This script measures the candidate ON the intended-policy baseline.

Per case the script renders FOUR outputs:
  A_nores  = partition_summed OFF + enable_res OFF  (linear residual)
  A_ours   = partition_summed OFF + enable_res ON   (full pipeline)
  B_nores  = partition_summed ON  + enable_res OFF
  B_ours   = partition_summed ON  + enable_res ON

Per-case metrics JSON captures (under both A and B):
  - extra_psd_lf / mf / hf totals (nores vs mic STFT)
  - W per-band energy (final-frame)
  - mu_lf / dW_lf / x2_lf / denom_lf cluster aggregates

Default cohort (mirrors hpf_intent_baseline_cohort.py):
  - 1× artifact reference (public 0I0XMl3M FS-mv)
  - 2× DT guard (XRTnTUjU + jtYTdZm3)
  - 3× FS echo guard (9xjhi + xQEUtY2 + qNvSMyU)

Usage:
  python3 python/scripts/partition_summed_x2_cohort.py \\
      --out out_partition_summed_x2/
  python3 python/scripts/partition_summed_x2_cohort.py \\
      --mic <internal_mic.wav> --ref <internal_lpb.wav> \\
      --stem internal_case_X --movement \\
      --out out_partition_summed_x2_internal/
"""
from __future__ import annotations

import argparse
import json
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

BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))     # 16
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))    # 22
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))    # 96
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)
HF_SLICE = slice(BIN_MF_HI, N_FREQS)

PUBLIC_BLIND_ROOT = os.path.join(REPO, 'wav', 'aec_challenge_blind')

# (stem_prefix, bucket_dir, movement)
DEFAULT_COHORT = [
    # Artifact reference
    ('0I0XMl3M0ECO0U1N0cJvpg', 'farend_singletalk', True),
    # DT guards
    ('XRTnTUjU5kS0mejzCqyCiw', 'doubletalk', False),
    ('jtYTdZm3lUmFVNibJWq8YQ', 'doubletalk', False),
    # FS echo guards
    ('9xjhiFbGo06hdQIsHTS6qA', 'farend_singletalk', False),
    ('xQEUtY2pWUi7v1X93TF2AA', 'farend_singletalk', False),
    ('qNvSMyUSXUyrDGpOw7s6qg', 'farend_singletalk', False),
]


def _stft_psd(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    nf = max(0, (n - FFT_PSD) // BLOCK + 1)
    out = np.empty((nf, N_FREQS), dtype=np.float32)
    for i in range(nf):
        seg = x[i * BLOCK: i * BLOCK + FFT_PSD] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def _build_cfg(movement: bool, partition_summed: bool) -> AecConfig:
    """v3.21.6 BALANCED + intended HPF policy + flag toggle."""
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    assert cfg.enable_highpass is True, 'mic HPF expected ON by default'
    cfg.enable_highpass_ref = False     # intended-policy override
    cfg.enable_cng = True
    cfg.use_partition_summed_x2_for_h_error_gain = bool(partition_summed)
    if movement:
        cfg.enable_delay_est = True
        cfg.delay_est_period_s = 0.25
        cfg.delay_est_init_s = 0.2
    return cfg


def _render_one(mic: np.ndarray, ref: np.ndarray, stem: str, out_dir: str,
                movement: bool, partition_summed: bool,
                clusters: list[tuple[int, int]], label: str) -> dict:
    """Render one variant; returns metrics dict."""
    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n].astype(np.float32, copy=False)
    ref = ref[:n].astype(np.float32, copy=False)

    # --- Pass 1: enable_res=False → nores tap ---------------------------
    cfg_nr = _build_cfg(movement, partition_summed)
    cfg_nr.enable_res = False
    cfg_nr.enable_cng = False
    aec_nr = AEC(cfg_nr)
    cluster_idx = {f'{lo}-{hi}': (lo, hi) for lo, hi in clusters}
    cluster_data: dict[str, list[dict]] = {k: [] for k in cluster_idx}
    orig_proc = aec_nr.filter.process
    n_part = aec_nr.filter.n_partitions
    delta32 = np.float32(aec_nr.filter.delta)

    def wrapped_proc(near_end, far_end, mu_scale=1.0):
        i = wrapped_proc.frame_idx
        wrapped_proc.frame_idx += 1
        for name, (lo, hi) in cluster_idx.items():
            if lo <= i <= hi:
                W_before = aec_nr.filter.W.copy()
                H_before = aec_nr.filter.H_error_per_bin.copy()
                out = orig_proc(near_end, far_end, mu_scale)
                # Re-derive denom under the SAME formula path actually used.
                if partition_summed:
                    X2 = (np.abs(aec_nr.filter.X_buf) ** 2).sum(axis=0)
                    X2 = X2.astype(np.float32)
                else:
                    X2 = (np.abs(aec_nr.filter.far_spec) ** 2).astype(np.float32)
                err_psd = aec_nr.filter._error_psd
                denom = (np.float32(0.5) * H_before * X2
                         + np.float32(n_part) * err_psd + delta32)
                mu = H_before / np.maximum(denom, np.float32(1e-30))
                W_after = aec_nr.filter.W
                cluster_data[name].append({
                    'frame': i,
                    'W_after_lf_energy': float(
                        np.sum(np.abs(W_after[:, LF_SLICE]) ** 2)),
                    'dW_lf_energy': float(
                        np.sum(np.abs((W_after - W_before)[:, LF_SLICE]) ** 2)),
                    'mu_lf_mean': float(np.mean(mu[LF_SLICE])),
                    'mu_mf_mean': float(np.mean(mu[MF_SLICE])),
                    'h_lf_mean': float(np.mean(H_before[LF_SLICE])),
                    'denom_lf_mean': float(np.mean(denom[LF_SLICE])),
                    'x2_lf_sum': float(np.sum(X2[LF_SLICE])),
                })
                return out
        return orig_proc(near_end, far_end, mu_scale)
    wrapped_proc.frame_idx = 0
    aec_nr.filter.process = wrapped_proc

    out_nores = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_nores[s:s + BLOCK] = aec_nr.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
    aec_nr.filter.process = orig_proc
    sf.write(os.path.join(out_dir, f'{stem}_{label}_nores.wav'),
             out_nores.astype(np.float32), SR)

    mic_psd = _stft_psd(mic)
    nores_psd = _stft_psd(out_nores)
    extra = np.maximum(nores_psd - mic_psd, 0.0)
    extra_lf = float(extra[:, LF_SLICE].sum())
    extra_mf = float(extra[:, MF_SLICE].sum())
    extra_hf = float(extra[:, HF_SLICE].sum())

    # --- Pass 2: enable_res=True → full pipeline ------------------------
    cfg_full = _build_cfg(movement, partition_summed)
    cfg_full.enable_res = True
    cfg_full.enable_cng = True
    np.random.seed(42)
    aec_full = AEC(cfg_full)
    out_ours = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_ours[s:s + BLOCK] = aec_full.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
    sf.write(os.path.join(out_dir, f'{stem}_{label}_ours.wav'),
             out_ours.astype(np.float32), SR)

    final_W = aec_nr.filter.W
    return {
        'partition_summed': partition_summed,
        'extra_psd_lf_total': extra_lf,
        'extra_psd_mf_total': extra_mf,
        'extra_psd_hf_total': extra_hf,
        'final_W_lf_energy': float(np.sum(np.abs(final_W[:, LF_SLICE]) ** 2)),
        'final_W_mf_energy': float(np.sum(np.abs(final_W[:, MF_SLICE]) ** 2)),
        'final_W_hf_energy': float(np.sum(np.abs(final_W[:, HF_SLICE]) ** 2)),
        'cluster_data': cluster_data,
    }


def render_case(mic_path: str, ref_path: str, stem: str, out_dir: str,
                movement: bool, pre_align: bool,
                clusters: list[tuple[int, int]]) -> dict:
    """Render single case under BOTH partition_summed OFF and ON."""
    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    assert sr_mic == SR == sr_ref, f'expected {SR} Hz'

    if pre_align:
        n0 = min(len(mic), len(ref))
        delay = estimate_delay(mic[:n0], ref[:n0], SR)
        if 0 < delay < n0:
            ref_a = np.zeros(n0, dtype=np.float32)
            ref_a[delay:] = ref[:n0 - delay]
            ref = ref_a

    mic = np.asarray(mic, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    out_A = _render_one(mic, ref, stem, out_dir, movement, False, clusters, 'A_off')
    out_B = _render_one(mic, ref, stem, out_dir, movement, True, clusters, 'B_on')
    return {
        'stem': stem,
        'movement': movement,
        'pre_align': pre_align,
        'version': __version__,
        'enable_highpass': True,
        'enable_highpass_ref': False,
        'A_partition_summed_OFF': out_A,
        'B_partition_summed_ON': out_B,
    }


def _parse_cases(spec: str) -> list[tuple[str, str, bool]]:
    out = []
    for s in spec.split(','):
        s = s.strip()
        if not s:
            continue
        parts = s.split(':')
        stem = parts[0]
        bucket = parts[1] if len(parts) > 1 else 'farend_singletalk'
        movement = (len(parts) > 2 and parts[2].lower() in ('1', 'true', 'yes'))
        out.append((stem, bucket, movement))
    return out


def _format_metric(label: str, A: float, B: float) -> str:
    if abs(A) < 1e-30:
        ratio = float('inf') if abs(B) > 0 else 1.0
    else:
        ratio = B / A
    return f'  {label:>26s}: A={A:.4e}  B={B:.4e}  B/A={ratio:.3f}'


def _print_report(per_case: list[dict]):
    print('\n' + '=' * 78)
    print('AEC3 partition-summed X² parity — cohort A/B verdict')
    print('  A = partition_summed OFF (= intended HPF baseline)')
    print('  B = partition_summed ON  (= AEC3 SpectralSum parity candidate)')
    print('  All runs under mic HPF ON / ref HPF OFF.')
    print('=' * 78)
    for rec in per_case:
        A = rec['A_partition_summed_OFF']
        B = rec['B_partition_summed_ON']
        print(f'\n[{rec["stem"]}]  movement={rec["movement"]}')
        for key in ('extra_psd_lf_total', 'extra_psd_mf_total',
                    'extra_psd_hf_total', 'final_W_lf_energy',
                    'final_W_mf_energy', 'final_W_hf_energy'):
            print(_format_metric(key, A[key], B[key]))
        # Cluster aggregates
        for cname in sorted(A['cluster_data']):
            framesA = A['cluster_data'][cname]
            framesB = B['cluster_data'][cname]
            if not framesA or not framesB:
                continue
            avgA = {k: float(np.mean([f[k] for f in framesA]))
                    for k in framesA[0] if k != 'frame'}
            avgB = {k: float(np.mean([f[k] for f in framesB]))
                    for k in framesB[0] if k != 'frame'}
            print(f'  cluster f={cname}:')
            for k in ('W_after_lf_energy', 'dW_lf_energy', 'mu_lf_mean',
                      'denom_lf_mean', 'x2_lf_sum'):
                print(_format_metric('    ' + k, avgA[k], avgB[k]))
    print('=' * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out_partition_summed_x2')
    ap.add_argument('--cases', default=None,
                    help='comma-separated <stem>:<bucket>[:movement]')
    ap.add_argument('--mic', help='single-case mic wav path')
    ap.add_argument('--ref', help='single-case ref wav path')
    ap.add_argument('--stem', help='single-case stem label')
    ap.add_argument('--movement', action='store_true',
                    help='single-case override: enable delay-est')
    ap.add_argument('--no-pre-align', action='store_true')
    ap.add_argument('--clusters', default='175-179,665-677')
    args = ap.parse_args()

    clusters = [(int(a), int(b)) for a, b in
                (c.split('-') for c in args.clusters.split(',') if c)]
    os.makedirs(args.out, exist_ok=True)
    pre_align = not args.no_pre_align

    if args.mic and args.ref and args.stem:
        cases = [(args.stem, args.mic, args.ref, args.movement)]
    else:
        spec = _parse_cases(args.cases) if args.cases else None
        cohort = spec if spec else DEFAULT_COHORT
        cases = []
        for stem, bucket, movement in cohort:
            base = os.path.join(PUBLIC_BLIND_ROOT, bucket)
            suffix = 'farend_singletalk_with_movement' if (
                movement and bucket == 'farend_singletalk') else bucket
            mic_p = os.path.join(base, f'{stem}_{suffix}_mic.wav')
            ref_p = os.path.join(base, f'{stem}_{suffix}_lpb.wav')
            cases.append((f'{stem}_{suffix}', mic_p, ref_p, movement))

    per_case = []
    for stem, mic_p, ref_p, movement in cases:
        if not (os.path.exists(mic_p) and os.path.exists(ref_p)):
            print(f'SKIP {stem}: missing {mic_p} or {ref_p}', file=sys.stderr)
            continue
        print(f'[render] {stem} (movement={movement})', flush=True)
        rec = render_case(mic_p, ref_p, stem, args.out, movement, pre_align,
                          clusters)
        per_case.append(rec)
        with open(os.path.join(args.out, f'{stem}_AB_metrics.json'), 'w') as f:
            json.dump(rec, f, indent=2)

    summary = {
        'version': __version__,
        'cohort_size': len(per_case),
        'enable_highpass': True,
        'enable_highpass_ref': False,
        'flag': 'use_partition_summed_x2_for_h_error_gain',
        'A_label': 'partition_summed OFF (= intended HPF baseline)',
        'B_label': 'partition_summed ON  (= AEC3 SpectralSum parity)',
        'per_case': per_case,
    }
    with open(os.path.join(args.out, 'cohort_AB_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    _print_report(per_case)


if __name__ == '__main__':
    main()
