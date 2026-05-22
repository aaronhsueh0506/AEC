"""Cohort-level baseline measurement under the intended HPF policy.

Per user directive 2026-05-22: intended HPF policy = mic ON / ref OFF.
Production default in code (config.py:212 = True for ref HPF) is
UNCHANGED pending 800-case AECMOS verdict; this script overrides
`enable_highpass_ref=False` at runtime to render every case under the
intended-policy baseline. The cohort output becomes the score anchor
that future candidates (e.g. E_avg_x2 partition-summed X²) compare
against — never against the code-on-main ON/ON state.

Cohort:
  - 1× artifact reference (public 0I0XMl3M FS-mv)
  - 2× DT guard (XRTnTUjU + jtYTdZm3, both with regression-listening
    history per project_xrtntuju_regression_clip + F2.4 monitor list)
  - 3× FS echo guard (9xjhi + xQEUtY2 + qNvSMyU, all cohort tail / P4
    tail / known difficult per v3.23 G.0 cohort)

Each case renders TWO outputs:
  - <stem>_intended_nores.wav  — enable_res=False (PBFDKF linear out)
  - <stem>_intended_ours.wav   — enable_res=True  (full pipeline out;
                                 anchor for downstream AECMOS scoring)

Per-case metrics JSON captures:
  - extra_psd_lf / mf / hf totals (nores vs mic STFT)
  - W per-band energy snapshot (final-frame)
  - mu_lf / dW_lf cluster-aggregate at user-specified frame windows

Usage:
  python3 python/scripts/hpf_intent_baseline_cohort.py \\
      --out out_hpf_intent_baseline/ \\
      [--cases <stem>:<bucket>:<movement>,...] \\
      [--clusters 175-179,665-677]

  --cases default is the 6-case public cohort. To add internal cases,
  point an additional CLI run at them with the same script + their
  paths.

Note: AECMOS scoring is NOT done by this script (no onnxruntime in
default venv). Render WAVs locally + run `python/bench_aecmos.py`
separately if AECMOS scoring is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

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


def _build_intended_cfg(movement: bool) -> AecConfig:
    """v3.21.6 BALANCED with INTENDED HPF policy override.

    Mic HPF ON (config default), ref HPF FORCED OFF (overrides code-on-
    main default which is True). All other knobs at v3.21.6 default.
    """
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    # User-directed policy override (2026-05-22):
    assert cfg.enable_highpass is True, 'mic HPF expected ON by default'
    cfg.enable_highpass_ref = False     # ← intended-policy override
    cfg.enable_cng = True               # ← matches eval_aec_challenge.py default
    if movement:
        cfg.enable_delay_est = True
        cfg.delay_est_period_s = 0.25
        cfg.delay_est_init_s = 0.2
    return cfg


def render_case(mic_path: str, ref_path: str, stem: str, out_dir: str,
                movement: bool, pre_align: bool,
                clusters: list[tuple[int, int]]) -> dict:
    """Render single case, write nores+ours WAV, return metrics dict."""
    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    assert sr_mic == SR == sr_ref, f'expected {SR} Hz, got mic={sr_mic} ref={sr_ref}'

    if pre_align:
        n0 = min(len(mic), len(ref))
        delay = estimate_delay(mic[:n0], ref[:n0], SR)
        if 0 < delay < n0:
            ref_a = np.zeros(n0, dtype=np.float32)
            ref_a[delay:] = ref[:n0 - delay]
            ref = ref_a

    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n].astype(np.float32, copy=False)
    ref = ref[:n].astype(np.float32, copy=False)

    # ---- Pass 1: enable_res=False → nores tap ---------------------------
    cfg_nr = _build_intended_cfg(movement)
    cfg_nr.enable_res = False
    cfg_nr.enable_cng = False
    aec_nr = AEC(cfg_nr)
    # Cluster trace: capture W_after_lf / dW_lf / mu_lf / h_lf per-frame
    # within the cluster windows by wrapping filter.process.
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
                    'W_after_mf_energy': float(
                        np.sum(np.abs(W_after[:, MF_SLICE]) ** 2)),
                    'W_after_hf_energy': float(
                        np.sum(np.abs(W_after[:, HF_SLICE]) ** 2)),
                    'dW_lf_energy': float(
                        np.sum(np.abs((W_after - W_before)[:, LF_SLICE]) ** 2)),
                    'mu_lf_mean': float(np.mean(mu[LF_SLICE])),
                    'mu_mf_mean': float(np.mean(mu[MF_SLICE])),
                    'h_lf_mean': float(np.mean(H_before[LF_SLICE])),
                    'h_mf_mean': float(np.mean(H_before[MF_SLICE])),
                    'denom_lf_mean': float(np.mean(denom[LF_SLICE])),
                    'x2_lf_sum': float(np.sum(X2[LF_SLICE])),
                    'mic_lf_sum': float(np.sum(
                        np.abs(aec_nr.filter.near_spec[LF_SLICE]) ** 2)),
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
    sf.write(os.path.join(out_dir, f'{stem}_intended_nores.wav'),
             out_nores.astype(np.float32), SR)

    # PSD diff (nores vs mic, post-render)
    mic_psd = _stft_psd(mic)
    nores_psd = _stft_psd(out_nores)
    extra = np.maximum(nores_psd - mic_psd, 0.0)
    extra_lf = extra[:, LF_SLICE].sum()
    extra_mf = extra[:, MF_SLICE].sum()
    extra_hf = extra[:, HF_SLICE].sum()

    # ---- Pass 2: enable_res=True → full pipeline output -----------------
    cfg_full = _build_intended_cfg(movement)
    cfg_full.enable_res = True
    cfg_full.enable_cng = True
    np.random.seed(42)
    aec_full = AEC(cfg_full)
    out_ours = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_ours[s:s + BLOCK] = aec_full.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
    sf.write(os.path.join(out_dir, f'{stem}_intended_ours.wav'),
             out_ours.astype(np.float32), SR)

    # ---- Per-case metrics record ----------------------------------------
    final_W = aec_nr.filter.W
    return {
        'stem': stem,
        'movement': movement,
        'pre_align': pre_align,
        'version': __version__,
        'enable_highpass': cfg_nr.enable_highpass,
        'enable_highpass_ref': cfg_nr.enable_highpass_ref,
        'n_samples': int(n),
        'n_blocks': int(n // BLOCK),
        'extra_psd_lf_total': float(extra_lf),
        'extra_psd_mf_total': float(extra_mf),
        'extra_psd_hf_total': float(extra_hf),
        'final_W_lf_energy': float(
            np.sum(np.abs(final_W[:, LF_SLICE]) ** 2)),
        'final_W_mf_energy': float(
            np.sum(np.abs(final_W[:, MF_SLICE]) ** 2)),
        'final_W_hf_energy': float(
            np.sum(np.abs(final_W[:, HF_SLICE]) ** 2)),
        'cluster_data': cluster_data,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out_hpf_intent_baseline')
    ap.add_argument('--cases', default=None,
                    help='comma-separated <stem>:<bucket>[:movement] '
                         '(default: 6-case public cohort)')
    ap.add_argument('--mic', help='single-case override: mic wav path')
    ap.add_argument('--ref', help='single-case override: ref wav path')
    ap.add_argument('--stem', help='single-case override: stem label')
    ap.add_argument('--movement', action='store_true',
                    help='single-case override: enable delay-est')
    ap.add_argument('--no-pre-align', action='store_true',
                    help='disable xcorr pre-alignment (only for tests)')
    ap.add_argument('--clusters', default='175-179,665-677')
    args = ap.parse_args()

    clusters = [(int(a), int(b)) for a, b in
                (c.split('-') for c in args.clusters.split(',') if c)]
    os.makedirs(args.out, exist_ok=True)
    pre_align = not args.no_pre_align

    # Build case list
    if args.mic and args.ref and args.stem:
        cases_to_render = [(args.stem, args.mic, args.ref,
                            args.movement)]
    else:
        cohort = _parse_cases(args.cases) if args.cases else DEFAULT_COHORT
        cases_to_render = []
        for stem, bucket, mv in cohort:
            mic_path = os.path.join(PUBLIC_BLIND_ROOT, bucket,
                                    f'{stem}_{bucket}_mic.wav')
            ref_path = os.path.join(PUBLIC_BLIND_ROOT, bucket,
                                    f'{stem}_{bucket}_lpb.wav')
            if not (os.path.exists(mic_path) and os.path.exists(ref_path)):
                # Try the `_with_movement` variant for FS-mv DT-mv cases
                mic_path = os.path.join(PUBLIC_BLIND_ROOT, bucket,
                                        f'{stem}_{bucket}_with_movement_mic.wav')
                ref_path = os.path.join(PUBLIC_BLIND_ROOT, bucket,
                                        f'{stem}_{bucket}_with_movement_lpb.wav')
                if not os.path.exists(mic_path):
                    print(f'  WARN: skip {stem} ({bucket}) — wav not found')
                    continue
                label = f'{stem}_{bucket}_with_movement'
            else:
                label = f'{stem}_{bucket}'
            cases_to_render.append((label, mic_path, ref_path, mv))

    print(f'Rendering {len(cases_to_render)} case(s) under intended policy '
          '(mic HPF=ON, ref HPF=OFF, BALANCED, CNG=ON)')
    all_metrics = []
    for stem, mic_path, ref_path, mv in cases_to_render:
        print(f'  {stem}  mv={mv}  ', end='', flush=True)
        rec = render_case(mic_path, ref_path, stem, args.out, mv,
                          pre_align, clusters)
        all_metrics.append(rec)
        print(f'extra_lf={rec["extra_psd_lf_total"]:.2f}  '
              f'final_W_lf={rec["final_W_lf_energy"]:.4f}')

    # Cohort summary JSON
    cohort_json = {
        'version': __version__,
        'policy': 'mic_HPF_ON / ref_HPF_OFF (user 2026-05-22 directive)',
        'clusters': args.clusters,
        'cases': all_metrics,
    }
    cohort_json_path = os.path.join(args.out, 'cohort_summary.json')
    with open(cohort_json_path, 'w') as f:
        json.dump(cohort_json, f, indent=2)

    # ---- Terminal transcription table -----------------------------------
    print('\n' + '=' * 88)
    print(f'HPF-INTENT BASELINE COHORT (mic ON / ref OFF) — v{__version__}')
    print('=' * 88)
    print('\n[TABLE 1] per-case extra_psd + final W energy (LF/MF/HF)')
    print(f"{'case':<48s} {'eLF':>9s} {'eMF':>9s} {'eHF':>9s} "
          f"{'WlfFinal':>10s} {'WmfFinal':>10s} {'WhfFinal':>10s}")
    for r in all_metrics:
        print(f"{r['stem']:<48s} "
              f"{r['extra_psd_lf_total']:>9.2f} "
              f"{r['extra_psd_mf_total']:>9.2f} "
              f"{r['extra_psd_hf_total']:>9.2f} "
              f"{r['final_W_lf_energy']:>10.4f} "
              f"{r['final_W_mf_energy']:>10.4f} "
              f"{r['final_W_hf_energy']:>10.4f}")

    for cname in clusters:
        clab = f'{cname[0]}-{cname[1]}'
        print(f'\n[TABLE 2] cluster f={clab} per-case W/H/mu trace (mean)')
        print(f"{'case':<48s} {'WlfAfter':>9s} {'dWlf':>8s} "
              f"{'muLF':>9s} {'hLF':>9s} {'x2LF':>9s} {'micLF':>9s}")
        for r in all_metrics:
            cd = r['cluster_data'].get(clab, [])
            if not cd:
                continue
            row = {k: float(np.mean([c[k] for c in cd])) for k in
                   ('W_after_lf_energy', 'dW_lf_energy', 'mu_lf_mean',
                    'h_lf_mean', 'x2_lf_sum', 'mic_lf_sum')}
            print(f"{r['stem']:<48s} "
                  f"{row['W_after_lf_energy']:>9.4f} "
                  f"{row['dW_lf_energy']:>8.4f} "
                  f"{row['mu_lf_mean']:>9.4f} "
                  f"{row['h_lf_mean']:>9.4f} "
                  f"{row['x2_lf_sum']:>9.2f} "
                  f"{row['mic_lf_sum']:>9.2f}")

    print('\n' + '=' * 88)
    print(f'cohort_summary.json → {cohort_json_path}')
    print(f'rendered WAVs (intended_nores + intended_ours) → {args.out}/')
    print('READ-BACK: TABLE 1 + TABLE 2.* (per cluster). For internal case,'
          ' add the case with --mic/--ref/--stem and re-run.')
    print('=' * 88)


if __name__ == '__main__':
    main()
