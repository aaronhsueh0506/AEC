"""XRTnTUjU_DT_static state-protection audit under partition_summed_x2 ON.

Task brief 2026-05-22 (user execution-guard):
  XRTnTUjU is now a stress-only case ([[project-xrtntuju-dt-static-stress]]).
  This script does NOT measure A/B convergence; it audits whether the
  AEC3-aligned state machine PROTECTS NE/speech when the filter has no
  clean convergence window.

Run config (intended baseline, no A/B):
  - mic HPF ON / ref HPF OFF
  - use_partition_summed_x2_for_h_error_gain = True (B_on only)
  - trace_hf_chain = True (orchestrator state ring)
  - extra wrapped captures on AEC.process for fields not in hf_chain

Per-frame trace fields (audit):
  - usable_linear_estimate
  - any_filter_converged / refined_conv / coarse_conv  (3 separate)
  - external_delay present (= self._delay_active)
  - transparent_mode_active
  - is_nearend_state
  - dominant_nearend (sg.is_dominant_nearend())
  - nearend_pwr_inflated  (E²>Y² proxy → pre-clamp nearend_pwr came from E²)
  - r2_sum / r2_unb_sum / s2_sum / erle samples
  - lower_band_gain LF/MF/HF means + gain_100
  - r2_mask_kill_ratio  (stationarity zeroing's effect on R²)

Damaged windows we audit (user-provided):
  frames 239-263 (mid case), 763-765 (later), plus any "processed /
  silenced" early region we auto-detect from mic LF+MF < 1% of utterance
  median.

Output:
  - state_trace_B_on.npz
  - state_audit_report.md
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
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))     # 16
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))    # 22
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))    # 96
LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)
HF_SLICE = slice(BIN_MF_HI, N_FREQS)

STEM = 'XRTnTUjU5kS0mejzCqyCiw_doubletalk'
CASE_DIR = os.path.join(REPO, 'wav', 'aec_challenge_blind', 'doubletalk')
MIC_P = os.path.join(CASE_DIR, f'{STEM}_mic.wav')
REF_P = os.path.join(CASE_DIR, f'{STEM}_lpb.wav')

OUT_DIR = '/tmp/be_partitionsum/xrtntuju_state_audit'
DAMAGED_WINDOWS = [(239, 263), (763, 765)]


def stft_psd(x: np.ndarray) -> np.ndarray:
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    nf = max(0, (n - FFT_PSD) // BLOCK + 1)
    out = np.empty((nf, N_FREQS), dtype=np.float32)
    for i in range(nf):
        seg = x[i * BLOCK: i * BLOCK + FFT_PSD] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def render_with_audit(partition_summed: bool = True) -> dict:
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    assert cfg.enable_highpass is True
    cfg.enable_highpass_ref = False
    cfg.enable_cng = True
    cfg.enable_res = True
    cfg.use_partition_summed_x2_for_h_error_gain = bool(partition_summed)
    cfg.trace_hf_chain = True

    aec = AEC(cfg)

    # Extra wrapped captures (fields the hf_chain doesn't include explicitly):
    #   - dominant_nearend (sg.is_dominant_nearend())
    #   - external_delay present (self._delay_active)
    #   - per-band gain means
    extra_rows: list[dict] = []
    orig_post = aec._aec3_post

    def wrapped_post(raw_output, near_end, far_end, *args, **kwargs):
        out = orig_post(raw_output, near_end, far_end, *args, **kwargs)
        # Pull state RIGHT AFTER _aec3_post; the hf_chain row for THIS frame
        # has just been appended. The orchestrator's _delay_active / sg.is_dominant_nearend
        # is consistent with that row.
        try:
            dom_ne = bool(aec._aec3_sg.is_dominant_nearend())
        except Exception:
            dom_ne = False
        try:
            ext_delay = bool(getattr(aec, '_delay_active', False))
        except Exception:
            ext_delay = False
        try:
            cur_delay = int(getattr(aec, '_current_delay', -1))
        except Exception:
            cur_delay = -1
        # Per-band gain means from the last sg.get_gain output. We can't read
        # gain directly (it's not stored on aec); but the latest hf_chain row
        # captures gain_5/30/100/200 and gain_n_bins. So we don't duplicate
        # here; just add the dom_ne + external_delay + curr delay fields.
        extra_rows.append({
            'frame': len(extra_rows),
            'dominant_nearend': dom_ne,
            'external_delay_present': ext_delay,
            'current_delay_samples': cur_delay,
        })
        return out
    aec._aec3_post = wrapped_post

    mic, _ = sf.read(MIC_P); mic = (mic if mic.ndim == 1 else mic[:, 0]).astype(np.float32)
    ref, _ = sf.read(REF_P); ref = (ref if ref.ndim == 1 else ref[:, 0]).astype(np.float32)
    n0 = min(len(mic), len(ref))
    delay = estimate_delay(mic[:n0], ref[:n0], SR)
    if 0 < delay < n0:
        ref_a = np.zeros(n0, dtype=np.float32)
        ref_a[delay:] = ref[:n0 - delay]
        ref = ref_a
    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n]
    ref = ref[:n]
    out_ours = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        out_ours[s:s + BLOCK] = aec.process(mic[s:s + BLOCK], ref[s:s + BLOCK])

    hf_chain = list(aec._hf_chain_trace)

    # Audio STFT for "processed/silenced" detection
    mic_psd = stft_psd(mic)
    lpb_psd = stft_psd(ref)
    ours_psd = stft_psd(out_ours)
    mic_lm = mic_psd[:, LF_SLICE].sum(axis=1) + mic_psd[:, MF_SLICE].sum(axis=1)
    lpb_lm = lpb_psd[:, LF_SLICE].sum(axis=1) + lpb_psd[:, MF_SLICE].sum(axis=1)
    ours_lm = ours_psd[:, LF_SLICE].sum(axis=1) + ours_psd[:, MF_SLICE].sum(axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    label = 'B_on' if partition_summed else 'A_off'
    sf.write(os.path.join(OUT_DIR, f'{STEM}_{label}_audit.wav'),
             out_ours.astype(np.float32), SR)
    print(f'render {label} done: {n // BLOCK} frames; hf_chain={len(hf_chain)} extras={len(extra_rows)}',
          flush=True)
    return {
        'hf_chain': hf_chain,
        'extra_rows': extra_rows,
        'mic_lm': mic_lm,
        'lpb_lm': lpb_lm,
        'ours_lm': ours_lm,
        'n_audio_frames': len(mic_lm),
    }


def _detect_silenced_region(mic_lm: np.ndarray) -> tuple[int, int] | None:
    """Find earliest contiguous block of frames where mic_lf+mf is << median."""
    med = float(np.median(mic_lm[mic_lm > 0])) if np.any(mic_lm > 0) else 0.0
    thr = max(med * 0.01, 1e-6)
    silent = mic_lm < thr
    runs = []
    i = 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            runs.append((i, j - 1, j - i))
            i = j
        else:
            i += 1
    runs.sort(key=lambda r: -r[2])
    if not runs:
        return None
    lo, hi, ln = runs[0]
    # Restrict to early region (first 5 s)
    if lo > int(5.0 * SR / BLOCK):
        return None
    return (lo, hi)


def _agg_window(rows: list[dict], extras: list[dict], window: tuple[int, int]) -> dict:
    lo, hi = window
    n = min(len(rows), len(extras))
    lo = max(0, min(lo, n - 1))
    hi = max(lo, min(hi, n - 1))
    fields = ('aec3_converged', 'refined_conv', 'coarse_conv', 'usable_linear',
              'is_nearend_state', 'transparent_mode_active', 'nearend_pwr_inflated',
              'stationary_mask_active')
    agg: dict = {'window': f'{lo}-{hi}', 'n': hi - lo + 1}
    for f in fields:
        vals = [bool(rows[i].get(f, False)) for i in range(lo, hi + 1)]
        agg[f + '_frac_true'] = (sum(vals) / max(len(vals), 1))
    # Numeric fields
    num_fields = ('erle_30', 'erle_100', 'r2_to_s2_ratio',
                  'r2_mask_kill_ratio', 'gain_30', 'gain_50', 'gain_100',
                  'gain_200', 'ne_sum_lf', 'echo_sum_lf', 'enr', 'snr')
    for f in num_fields:
        vals = [float(rows[i].get(f, 0.0)) for i in range(lo, hi + 1)]
        agg[f + '_mean'] = sum(vals) / max(len(vals), 1)
        agg[f + '_min'] = min(vals) if vals else 0.0
        agg[f + '_max'] = max(vals) if vals else 0.0
    # Extras
    dom_ne = [bool(extras[i].get('dominant_nearend', False)) for i in range(lo, hi + 1)]
    ext_d = [bool(extras[i].get('external_delay_present', False)) for i in range(lo, hi + 1)]
    cur_d = [int(extras[i].get('current_delay_samples', -1)) for i in range(lo, hi + 1)]
    agg['dominant_nearend_frac_true'] = sum(dom_ne) / max(len(dom_ne), 1)
    agg['external_delay_frac_true'] = sum(ext_d) / max(len(ext_d), 1)
    agg['current_delay_min'] = min(cur_d) if cur_d else -1
    agg['current_delay_max'] = max(cur_d) if cur_d else -1
    return agg


def _diagnose(agg: dict) -> str:
    """Apply user's 3-branch diagnostic:
       1. usable_linear True without convergence evidence → state bug
       2. usable_linear False but gain over-suppresses → RES/NE/SG issue
       3. NE detector always False but mic has DT speech → state/NE bug
    """
    ul = agg['usable_linear_frac_true']
    refc = agg['refined_conv_frac_true']
    coac = agg['coarse_conv_frac_true']
    anyc = max(refc, coac, agg['aec3_converged_frac_true'])
    ne = agg['is_nearend_state_frac_true']
    dom = agg['dominant_nearend_frac_true']
    g100 = agg['gain_100_mean']
    g30 = agg['gain_30_mean']
    notes = []

    if ul > 0.5 and anyc < 0.3:
        notes.append('STATE BUG: usable_linear is True in this window but '
                     'no convergence evidence (refined/coarse/aec3 all <30% true).')
    elif ul < 0.3 and (g100 < 0.5 or g30 < 0.5):
        notes.append('SG/NE OVER-SUPPRESS: usable_linear False but gain is '
                     'aggressive on speech bins — RES/NE/SG path not honouring '
                     'state.')
    if ne < 0.3 and dom < 0.3:
        notes.append('NE DETECTOR SUPPRESSED: both is_nearend_state and '
                     'dominant_nearend are False most of the window.')
    if not notes:
        notes.append('State coherent for this window — no obvious protection gap.')
    return ' '.join(notes)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d_b = render_with_audit(partition_summed=True)
    d_a = render_with_audit(partition_summed=False)
    # Use B_on as the primary audit + A_off whole-utterance counter-reference
    rows = d_b['hf_chain']
    extras = d_b['extra_rows']
    rows_a = d_a['hf_chain']
    extras_a = d_a['extra_rows']
    mic_lm = d_b['mic_lm']
    lpb_lm = d_b['lpb_lm']
    ours_lm = d_b['ours_lm']
    ours_a_lm = d_a['ours_lm']

    silent = _detect_silenced_region(mic_lm)
    if silent is not None:
        DAMAGED_WINDOWS.insert(0, silent)
        print(f'auto-detected silenced/processed early region: frames {silent[0]}-{silent[1]}',
              flush=True)

    # Per-window aggregate (B_on primary, A_off for comparison)
    aggs = [_agg_window(rows, extras, w) for w in DAMAGED_WINDOWS]
    aggs_a = [_agg_window(rows_a, extras_a, w) for w in DAMAGED_WINDOWS]

    # Save NPZ
    flat_keys = sorted({k for r in rows for k in r.keys() if isinstance(r.get(k), (bool, int, float))})
    arrs = {k: np.array([r.get(k, 0) for r in rows]) for k in flat_keys}
    extra_keys = sorted({k for r in extras for k in r.keys() if isinstance(r.get(k), (bool, int, float))})
    for k in extra_keys:
        arrs['extra_' + k] = np.array([r.get(k, 0) for r in extras])
    arrs['mic_lf_mf'] = mic_lm
    arrs['lpb_lf_mf'] = lpb_lm
    arrs['ours_lf_mf'] = ours_lm
    np.savez(os.path.join(OUT_DIR, 'state_trace_B_on.npz'), **arrs)

    # Build report
    lines: list[str] = []
    lines.append('# XRTnTUjU_DT_static state-protection audit (partition_summed_x2 ON)')
    lines.append('')
    lines.append(f'AEC __version__ = {__version__}')
    lines.append('Config: mic HPF ON / ref HPF OFF, partition_summed_x2 ON, preset balanced.')
    lines.append(f'Total frames: hf_chain={len(rows)}  extras={len(extras)}  audio_lf_mf_frames={len(mic_lm)}')
    lines.append('')
    lines.append('## Damaged / audit windows  (B_on vs A_off side-by-side)')
    lines.append('')
    for w, agg, agg_a in zip(DAMAGED_WINDOWS, aggs, aggs_a):
        lines.append(f'### Frames {w[0]}–{w[1]}  (n = {agg["n"]})')
        lines.append('')
        seg_mic = mic_lm[w[0]:w[1] + 1] if w[1] < len(mic_lm) else mic_lm[w[0]:]
        seg_lpb = lpb_lm[w[0]:w[1] + 1] if w[1] < len(lpb_lm) else lpb_lm[w[0]:]
        seg_ours_b = ours_lm[w[0]:w[1] + 1] if w[1] < len(ours_lm) else ours_lm[w[0]:]
        seg_ours_a = ours_a_lm[w[0]:w[1] + 1] if w[1] < len(ours_a_lm) else ours_a_lm[w[0]:]
        lines.append(f'  audio mean (LF+MF):  mic={seg_mic.mean():.2e}  lpb={seg_lpb.mean():.2e}  '
                     f'ours_A_off={seg_ours_a.mean():.2e}  ours_B_on={seg_ours_b.mean():.2e}')
        lines.append('')
        lines.append('| field | A_off | B_on |')
        lines.append('|---|---:|---:|')
        for k in ('aec3_converged_frac_true', 'refined_conv_frac_true', 'coarse_conv_frac_true',
                  'usable_linear_frac_true', 'is_nearend_state_frac_true',
                  'dominant_nearend_frac_true', 'transparent_mode_active_frac_true',
                  'external_delay_frac_true', 'nearend_pwr_inflated_frac_true',
                  'stationary_mask_active_frac_true'):
            lines.append(f'| {k} | {agg_a[k]:.3f} | {agg[k]:.3f} |')
        for k in ('erle_30', 'erle_100', 'r2_to_s2_ratio', 'r2_mask_kill_ratio',
                  'gain_30', 'gain_50', 'gain_100', 'gain_200'):
            lines.append(f'| {k} (mean) | {agg_a[k + "_mean"]:.3e} | {agg[k + "_mean"]:.3e} |')
        lines.append(f'| current_delay_samples (max) | {agg_a["current_delay_max"]} | {agg["current_delay_max"]} |')
        lines.append('')
        lines.append(f'  **B_on diagnosis:** {_diagnose(agg)}')
        lines.append(f'  **A_off diagnosis:** {_diagnose(agg_a)}')
        lines.append('')

    n_full = len(rows)
    full_b = _agg_window(rows, extras, (0, n_full - 1))
    full_a = _agg_window(rows_a, extras_a, (0, n_full - 1))
    lines.append('## Whole-utterance reference  (A_off vs B_on)')
    lines.append('')
    lines.append('| field | A_off | B_on |')
    lines.append('|---|---:|---:|')
    for k in ('aec3_converged_frac_true', 'refined_conv_frac_true', 'coarse_conv_frac_true',
              'usable_linear_frac_true', 'is_nearend_state_frac_true',
              'dominant_nearend_frac_true', 'transparent_mode_active_frac_true',
              'external_delay_frac_true', 'nearend_pwr_inflated_frac_true',
              'stationary_mask_active_frac_true'):
        lines.append(f'| {k} | {full_a[k]:.3f} | {full_b[k]:.3f} |')
    for k in ('erle_30', 'erle_100', 'gain_30', 'gain_50', 'gain_100', 'gain_200'):
        lines.append(f'| {k} (mean) | {full_a[k + "_mean"]:.3e} | {full_b[k + "_mean"]:.3e} |')
    lines.append('')

    md_path = os.path.join(OUT_DIR, 'state_audit_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'wrote {md_path}', flush=True)


if __name__ == '__main__':
    main()
