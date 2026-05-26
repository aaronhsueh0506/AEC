"""XRTnTUjU DT_static frame-level paired trace.

Goal: explain the Δdeg -1.253 regression at partition_summed_x2 ON vs OFF on
the canonical 6-case AECMOS A/B (intended HPF policy: mic ON / ref OFF).

For each run (OFF, ON) we dump a per-frame NPZ capturing the fields the user
asked for plus enough STFT context to localise the damaged frames:

PBFDKF-internal (wrapped filter.process):
  mu_lf/mf/hf_mean, dW_lf/mf/hf_l2, W_after_lf/mf/hf_energy, W_norm,
  X2_latest_lf/mf/hf, X2_summed_lf/mf/hf,
  H_error_lf/mf/hf_mean,
  denom_lf/mf/hf_mean,
  noise_gate_zero_frac_lf/mf/hf,
  e2_refined_sum, e2_coarse_sum, h_refresh_converged

Orchestrator (via trace_hf_chain ring):
  is_nearend_state, dominant_nearend_active, transparent_mode_active,
  usable_linear_estimate, _filter_converged_enough,
  r2_post_sum, gain_lf/mf/hf_mean,
  ne_lf, echo_lf, noise_lf

Audio frames (post-render STFT):
  mic_psd_band, nores_psd_band, ours_psd_band  (lf/mf/hf)

Outputs:
  trace_OFF.npz, trace_ON.npz, trace_diff_report.md

Usage:
  python3 python/scripts/xrtntuju_dt_static_trace.py
"""
from __future__ import annotations

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

# Band slices (consistent with hpf_intent_baseline_cohort.py / cohort A/B)
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))     # 16
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))    # 22
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))    # 96
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)
HF_SLICE = slice(BIN_MF_HI, N_FREQS)

CASE_DIR = os.path.join(REPO, 'wav', 'aec_challenge_blind', 'doubletalk')
STEM = 'XRTnTUjU5kS0mejzCqyCiw_doubletalk'
MIC_PATH = os.path.join(CASE_DIR, f'{STEM}_mic.wav')
REF_PATH = os.path.join(CASE_DIR, f'{STEM}_lpb.wav')

OUT_DIR = '/tmp/be_partitionsum/xrtntuju_trace'


def _stft_psd(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    nf = max(0, (n - FFT_PSD) // BLOCK + 1)
    out = np.empty((nf, N_FREQS), dtype=np.float32)
    for i in range(nf):
        seg = x[i * BLOCK: i * BLOCK + FFT_PSD] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def _band_sum(psd_row: np.ndarray) -> tuple[float, float, float]:
    return (float(np.sum(psd_row[LF_SLICE])),
            float(np.sum(psd_row[MF_SLICE])),
            float(np.sum(psd_row[HF_SLICE])))


def _band_mean(arr: np.ndarray) -> tuple[float, float, float]:
    return (float(np.mean(arr[LF_SLICE])),
            float(np.mean(arr[MF_SLICE])),
            float(np.mean(arr[HF_SLICE])))


def _band_l2sum(arr: np.ndarray) -> tuple[float, float, float]:
    a2 = np.abs(arr) ** 2
    return (float(np.sum(a2[LF_SLICE])),
            float(np.sum(a2[MF_SLICE])),
            float(np.sum(a2[HF_SLICE])))


def _build_cfg(partition_summed: bool) -> AecConfig:
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    assert cfg.enable_highpass is True
    cfg.enable_highpass_ref = False     # intended policy
    cfg.enable_cng = True
    cfg.use_partition_summed_x2_for_h_error_gain = bool(partition_summed)
    cfg.trace_hf_chain = True           # request orchestrator ring buffer
    return cfg


def _wrap_filter(aec) -> tuple[dict, callable]:
    """Wrap aec.filter.process to capture per-frame PBFDKF-internal trace."""
    filt = aec.filter
    n_part = int(filt.n_partitions)
    delta32 = np.float32(filt.delta)
    from modules import aec3_scale as _aec3_scale
    NG = np.float32(_aec3_scale.NOISE_GATE_POWER_FLOAT)

    rows: list[dict] = []
    orig = filt.process

    def wrapped(near_end, far_end, mu_scale=1.0):
        i = wrapped.frame_idx
        wrapped.frame_idx += 1
        W_before = filt.W.copy()
        H_before = filt.H_error_per_bin.copy()
        e2_refined_before = filt._error_psd.copy()       # smoothed E²_refined (pre-update view)
        e2_coarse_before = float(filt._e2_coarse_for_refresh)
        # partition_idx BEFORE process() — this is the slot that will hold
        # the current frame's far_spec after orig() runs.
        curr_p = int(filt.partition_idx)

        out = orig(near_end, far_end, mu_scale)

        # X² latest and summed
        X2_latest = (np.abs(filt.X_buf[curr_p]) ** 2).astype(np.float32)
        X2_summed = (np.abs(filt.X_buf) ** 2).sum(axis=0).astype(np.float32)

        # Re-derive denom / mu / noise-gate USING the same X² source the
        # actual path used this frame (matches filter._use_partition_summed_x2…).
        X2_used = X2_summed if filt._use_partition_summed_x2_for_h_error_gain else X2_latest
        denom = (np.float32(0.5) * H_before * X2_used
                 + np.float32(n_part) * e2_refined_before + delta32)
        mu = H_before / np.maximum(denom, np.float32(1e-30))
        gate_zero = X2_used < NG    # bins zeroed by noise gate
        mu_gated = np.where(gate_zero, np.float32(0.0), mu)

        W_after = filt.W
        dW = W_after - W_before

        # Bandwise stats
        mu_lf, mu_mf, mu_hf = _band_mean(mu_gated)
        dW_lf, dW_mf, dW_hf = _band_l2sum(dW)
        W_lf, W_mf, W_hf = _band_l2sum(W_after)
        x2l_lf, x2l_mf, x2l_hf = (float(np.sum(X2_latest[LF_SLICE])),
                                  float(np.sum(X2_latest[MF_SLICE])),
                                  float(np.sum(X2_latest[HF_SLICE])))
        x2s_lf, x2s_mf, x2s_hf = (float(np.sum(X2_summed[LF_SLICE])),
                                  float(np.sum(X2_summed[MF_SLICE])),
                                  float(np.sum(X2_summed[HF_SLICE])))
        Hf_lf, Hf_mf, Hf_hf = _band_mean(H_before)
        den_lf, den_mf, den_hf = _band_mean(denom)
        ng_lf = float(np.mean(gate_zero[LF_SLICE]))
        ng_mf = float(np.mean(gate_zero[MF_SLICE]))
        ng_hf = float(np.mean(gate_zero[HF_SLICE]))
        e2r_now = float(np.sum(filt._error_psd))         # post-update smoothed
        e2c_now = float(filt._e2_coarse_for_refresh)
        refresh_converged = bool(e2r_now <= e2c_now)

        rows.append({
            'frame': i,
            'mu_lf': mu_lf, 'mu_mf': mu_mf, 'mu_hf': mu_hf,
            'dW_lf_l2': dW_lf, 'dW_mf_l2': dW_mf, 'dW_hf_l2': dW_hf,
            'W_lf_l2': W_lf, 'W_mf_l2': W_mf, 'W_hf_l2': W_hf,
            'W_l2_total': float(np.sum(np.abs(W_after) ** 2)),
            'X2_latest_lf': x2l_lf, 'X2_latest_mf': x2l_mf, 'X2_latest_hf': x2l_hf,
            'X2_summed_lf': x2s_lf, 'X2_summed_mf': x2s_mf, 'X2_summed_hf': x2s_hf,
            'H_lf': Hf_lf, 'H_mf': Hf_mf, 'H_hf': Hf_hf,
            'denom_lf': den_lf, 'denom_mf': den_mf, 'denom_hf': den_hf,
            'noise_gate_zero_frac_lf': ng_lf,
            'noise_gate_zero_frac_mf': ng_mf,
            'noise_gate_zero_frac_hf': ng_hf,
            'e2_refined_sum_pre': float(np.sum(e2_refined_before)),
            'e2_refined_sum_post': e2r_now,
            'e2_coarse_sum': e2c_now,
            'h_refresh_converged': refresh_converged,
            'mu_scale_in': float(mu_scale),
        })
        return out

    wrapped.frame_idx = 0
    filt.process = wrapped
    return rows, orig


def render_trace(partition_summed: bool, label: str) -> dict:
    print(f'[render] partition_summed={partition_summed} → {label}', flush=True)
    mic, sr_mic = sf.read(MIC_PATH)
    ref, sr_ref = sf.read(REF_PATH)
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    assert sr_mic == SR == sr_ref

    # Pre-align (matches eval_aec_challenge.py path)
    n0 = min(len(mic), len(ref))
    delay = estimate_delay(mic[:n0], ref[:n0], SR)
    if 0 < delay < n0:
        ref_a = np.zeros(n0, dtype=np.float32)
        ref_a[delay:] = ref[:n0 - delay]
        ref = ref_a

    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n].astype(np.float32, copy=False)
    ref = ref[:n].astype(np.float32, copy=False)

    # --- Pass 1: nores tap ---------------------------------------------------
    cfg_nr = _build_cfg(partition_summed)
    cfg_nr.enable_res = False
    cfg_nr.enable_cng = False
    aec_nr = AEC(cfg_nr)
    rows_filter, _orig = _wrap_filter(aec_nr)
    out_nores = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_nores[s:s + BLOCK] = aec_nr.process(mic[s:s + BLOCK], ref[s:s + BLOCK])

    # --- Pass 2: full pipeline (RES + CNG) + trace_hf_chain ring ------------
    cfg_full = _build_cfg(partition_summed)
    cfg_full.enable_res = True
    cfg_full.enable_cng = True
    np.random.seed(42)
    aec_full = AEC(cfg_full)
    out_ours = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_ours[s:s + BLOCK] = aec_full.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
    hf_chain = getattr(aec_full, '_hf_chain_trace', [])

    # STFT bands
    mic_psd = _stft_psd(mic)
    nr_psd = _stft_psd(out_nores)
    ou_psd = _stft_psd(out_ours)
    nf = mic_psd.shape[0]
    audio_bands = {
        'mic_lf': mic_psd[:, LF_SLICE].sum(axis=1),
        'mic_mf': mic_psd[:, MF_SLICE].sum(axis=1),
        'mic_hf': mic_psd[:, HF_SLICE].sum(axis=1),
        'nores_lf': nr_psd[:, LF_SLICE].sum(axis=1),
        'nores_mf': nr_psd[:, MF_SLICE].sum(axis=1),
        'nores_hf': nr_psd[:, HF_SLICE].sum(axis=1),
        'ours_lf': ou_psd[:, LF_SLICE].sum(axis=1),
        'ours_mf': ou_psd[:, MF_SLICE].sum(axis=1),
        'ours_hf': ou_psd[:, HF_SLICE].sum(axis=1),
    }

    # Save WAVs
    os.makedirs(OUT_DIR, exist_ok=True)
    sf.write(os.path.join(OUT_DIR, f'{STEM}_{label}_nores.wav'),
             out_nores.astype(np.float32), SR)
    sf.write(os.path.join(OUT_DIR, f'{STEM}_{label}_ours.wav'),
             out_ours.astype(np.float32), SR)

    # Pack arrays
    arr_filter = {k: np.array([r[k] for r in rows_filter])
                  for k in rows_filter[0]} if rows_filter else {}
    npz_path = os.path.join(OUT_DIR, f'{STEM}_{label}_trace.npz')
    np.savez(npz_path,
             **arr_filter,
             **audio_bands,
             hf_chain_n=len(hf_chain),
             nf_audio=nf)
    print(f'  wrote {npz_path}  filter_frames={len(rows_filter)}  audio_frames={nf}  hf_chain={len(hf_chain)}',
          flush=True)
    return {
        'label': label,
        'filter_frames': len(rows_filter),
        'audio_frames': nf,
        'hf_chain_frames': len(hf_chain),
        'npz_path': npz_path,
        'hf_chain': hf_chain,
        'audio_bands': audio_bands,
        'rows_filter': rows_filter,
    }


def find_damaged_frames(off: dict, on: dict, top_k: int = 20) -> list[int]:
    """Return audio-frame indices where ours_B suppresses energy vs ours_A by
    the largest absolute amount, summed across LF+MF (NE speech bands).
    """
    a_lf = off['audio_bands']['ours_lf']
    a_mf = off['audio_bands']['ours_mf']
    b_lf = on['audio_bands']['ours_lf']
    b_mf = on['audio_bands']['ours_mf']
    n = min(len(a_lf), len(b_lf))
    # Score: A − B on LF+MF (positive ⇒ B over-suppressed)
    score = (a_lf[:n] - b_lf[:n]) + (a_mf[:n] - b_mf[:n])
    # Restrict to frames where mic actually has speech energy in LF+MF
    mic_lf = off['audio_bands']['mic_lf'][:n]
    mic_mf = off['audio_bands']['mic_mf'][:n]
    mic_thr = np.percentile(mic_lf + mic_mf, 50)
    mask = (mic_lf + mic_mf) > mic_thr
    score = np.where(mask, score, -np.inf)
    return np.argsort(score)[-top_k:][::-1].tolist()


def report(off: dict, on: dict, damaged: list[int]) -> str:
    out: list[str] = []
    out.append('# XRTnTUjU DT_static frame-level trace — partition_summed OFF vs ON')
    out.append('')
    out.append(f'AEC __version__ = {__version__}')
    out.append('Policy: mic HPF ON / ref HPF OFF (intended baseline). Preset balanced.')
    out.append('')
    # 6-case AECMOS context
    out.append('Cohort context (6-case AECMOS):')
    out.append('  this case: A_off A_deg=3.929  B_on B_deg=2.676  Δdeg=−1.253')
    out.append('  this case: A_off A_echo=4.081 B_on B_echo=4.487 Δecho=+0.406')
    out.append('')
    # Overall bucket energy diff (whole utterance)
    for band in ('lf', 'mf', 'hf'):
        ai = float(np.sum(off['audio_bands'][f'ours_{band}']))
        bi = float(np.sum(on['audio_bands'][f'ours_{band}']))
        out.append(f'  ours total energy {band.upper()}: A={ai:.3e}  B={bi:.3e}  B/A={bi/max(ai,1e-30):.3f}')
    out.append('')

    # Damaged frames table
    out.append(f'## Top {len(damaged)} damaged frames (B over-suppresses vs A on LF+MF, mic speech-active)')
    out.append('')
    out.append('| frame | mic_lf+mf | A_ours_lf+mf | B_ours_lf+mf | Δ(A−B) | A_nores_lf+mf | B_nores_lf+mf | Δ(A−B)_nores |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    a_ab = off['audio_bands']
    b_ab = on['audio_bands']
    for i in damaged:
        m_lm = a_ab['mic_lf'][i] + a_ab['mic_mf'][i]
        ao_lm = a_ab['ours_lf'][i] + a_ab['ours_mf'][i]
        bo_lm = b_ab['ours_lf'][i] + b_ab['ours_mf'][i]
        an_lm = a_ab['nores_lf'][i] + a_ab['nores_mf'][i]
        bn_lm = b_ab['nores_lf'][i] + b_ab['nores_mf'][i]
        out.append(f'| {i} | {m_lm:.2e} | {ao_lm:.2e} | {bo_lm:.2e} | {ao_lm-bo_lm:+.2e} | '
                   f'{an_lm:.2e} | {bn_lm:.2e} | {an_lm-bn_lm:+.2e} |')
    out.append('')

    # Linear-filter vs RES attribution
    out.append('## Attribution: linear filter vs RES/suppression stage')
    out.append('')
    ours_diff = sum((a_ab['ours_lf'][i] + a_ab['ours_mf'][i] - b_ab['ours_lf'][i] - b_ab['ours_mf'][i])
                    for i in damaged)
    nores_diff = sum((a_ab['nores_lf'][i] + a_ab['nores_mf'][i] - b_ab['nores_lf'][i] - b_ab['nores_mf'][i])
                     for i in damaged)
    pct = (nores_diff / ours_diff * 100.0) if ours_diff > 0 else 0.0
    out.append(f'Sum over damaged frames (A−B, LF+MF):')
    out.append(f'  ours_diff = {ours_diff:.3e}')
    out.append(f'  nores_diff = {nores_diff:.3e}   ({pct:.1f}% of ours_diff)')
    out.append('')
    if abs(pct) < 25.0:
        attr = ('Linear residual already similar between A and B at damaged '
                'frames → damage is dominantly in **RES / SuppressionGain** '
                'reacting to a different linear residual structure.')
    elif pct >= 75.0:
        attr = ('Linear residual already differs strongly → damage is '
                'dominantly in the **PBFDKF linear filter** (W learned NE).')
    else:
        attr = ('Mixed: damage attributable to both linear filter difference '
                'and RES amplification. Inspect per-frame trace below.')
    out.append(f'  → {attr}')
    out.append('')

    # PBFDKF-internal cluster around damaged frames
    out.append('## PBFDKF-internal — mean over damaged frames (A vs B)')
    out.append('')
    cols = ('mu_lf', 'mu_mf', 'mu_hf', 'dW_lf_l2', 'dW_mf_l2', 'dW_hf_l2',
            'W_lf_l2', 'W_mf_l2', 'X2_latest_lf', 'X2_summed_lf',
            'H_lf', 'H_mf', 'denom_lf', 'denom_mf',
            'noise_gate_zero_frac_lf', 'noise_gate_zero_frac_mf',
            'e2_refined_sum_post', 'e2_coarse_sum')
    out.append('| field | A_off | B_on | B/A |')
    out.append('|---|---:|---:|---:|')
    rA = off['rows_filter']
    rB = on['rows_filter']
    if rA and rB:
        n_f = min(len(rA), len(rB))
        sel = [i for i in damaged if i < n_f]
        for c in cols:
            va = float(np.mean([rA[i][c] for i in sel])) if sel else 0.0
            vb = float(np.mean([rB[i][c] for i in sel])) if sel else 0.0
            ratio = vb / max(abs(va), 1e-30) if va != 0 else float('inf')
            out.append(f'| {c} | {va:.3e} | {vb:.3e} | {ratio:.3f} |')
    out.append('')

    # Orchestrator-level (hf_chain) state at damaged frames
    out.append('## AecState / dominant-NE / SuppressionGain at damaged frames')
    out.append('')
    out.append('| frame | A: NE | A: dominant_NE | A: transparent | A: usable_lin | B: NE | B: dominant_NE | B: transparent | B: usable_lin |')
    out.append('|---:|:---|:---|:---|:---|:---|:---|:---|:---|')
    hfA = off['hf_chain']
    hfB = on['hf_chain']
    for i in damaged:
        rowA = hfA[i] if i < len(hfA) else {}
        rowB = hfB[i] if i < len(hfB) else {}
        out.append(f'| {i} | '
                   f'{rowA.get("is_nearend_state", "?")} | '
                   f'{rowA.get("dominant_nearend_active", rowA.get("dominant_ne", "?"))} | '
                   f'{rowA.get("transparent_mode_active", rowA.get("transparent", "?"))} | '
                   f'{rowA.get("usable_linear_now", rowA.get("usable_linear", "?"))} | '
                   f'{rowB.get("is_nearend_state", "?")} | '
                   f'{rowB.get("dominant_nearend_active", rowB.get("dominant_ne", "?"))} | '
                   f'{rowB.get("transparent_mode_active", rowB.get("transparent", "?"))} | '
                   f'{rowB.get("usable_linear_now", rowB.get("usable_linear", "?"))} |')
    out.append('')
    return '\n'.join(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    off = render_trace(False, 'A_off')
    on = render_trace(True,  'B_on')
    # Audio frames slightly differ from filter frames due to STFT framing —
    # we use audio frame indexing for damaged frames (1 per 10 ms) which lines
    # up 1:1 with the filter (block 160 = 10 ms).
    damaged = find_damaged_frames(off, on, top_k=20)
    print('damaged audio-frame indices (top-20):', damaged, flush=True)
    md = report(off, on, damaged)
    md_path = os.path.join(OUT_DIR, 'trace_diff_report.md')
    with open(md_path, 'w') as f:
        f.write(md)
    print(f'\nWrote {md_path}')


if __name__ == '__main__':
    main()
