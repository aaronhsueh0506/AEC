"""Trace + ablation harness for the v3.21.6 nores artifact debug.

Renders a single (mic, ref) case 7 times under different ablations and
captures per-frame nores−mic delta energy + event timestamps. Output is
JSON-only (no audio leakage) so the user can run on an internal case and
return only the summary.

All ablations are READ-ONLY in the sense that they patch the *live* AEC
instance for the duration of the run; no module code is modified. Run
with `enable_res=False` so the output IS the PBFDKF refined error (the
"nores" tap point the artifact lives in).

Usage:
  python3 python/scripts/nores_artifact_trace.py \\
      --mic /path/to/mic.wav --ref /path/to/ref.wav \\
      --stem CASE_NAME --out out_nores_artifact_debug/ \\
      [--ablations A0,A1,A2,A3,A4,A5,A6] [--write-wav]

Output:
  out_nores_artifact_debug/<stem>/summary.json   ← return this to me
  out_nores_artifact_debug/<stem>/<ablation>.json
  out_nores_artifact_debug/<stem>/<ablation>_nores.wav   (only if --write-wav)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'python'))

from aec import AEC, AecConfig, __version__   # noqa: E402

BLOCK = 160          # 10 ms hop @ 16 kHz
FFT_PSD = 512        # 32 ms analysis window for nores/mic PSD diff
HOP_PSD = BLOCK      # frame-aligned with AEC processing
SR = 16000

# Tonal-grid suspect band 700-3000 Hz (audible "電子音/破音"). LF band 0-500 Hz
# tracks the steady-LF red stripe.
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PSD)))     # ~16
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PSD)))    # ~22
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PSD)))    # ~96

TOP_N = 20


@dataclass
class FrameSnapshot:
    """Per-AEC-block snapshot captured immediately after process()."""
    frame: int
    mu_scale: float
    main_paused: int
    boost_q: int
    reverse_copy: int
    saturation_level: float
    epc_active: int
    converged: int
    h_error_mean: float
    h_error_max: float
    p_mean: float
    w_norm: float
    raw_err_pwr: float


# ---------------------------------------------------------------------------
# Ablation hooks
# ---------------------------------------------------------------------------


def _patch_freeze_mu(aec: AEC) -> Callable[[], None]:
    """A1: freeze adaptation — force main_mu = 0 every frame.

    Wraps `aec.filter.process(near, far, mu_scale)` to override the third
    argument to 0.0. Cheaper than chasing the orchestrator's mu_scale
    branches (`_compute_mu_scale` vs `_get_simple_mu_scale` vs P3e advisory
    multiplier vs main_paused). All filter-internal bookkeeping (Kalman
    update path / H_error refresh leakage) still runs but the NLMS step
    contribution is zeroed via mu_scale=0.
    """
    orig = aec.filter.process

    def patched(near_end, far_end, mu_scale=1.0):
        return orig(near_end, far_end, 0.0)
    aec.filter.process = patched
    return lambda: setattr(aec.filter, 'process', orig)


def _patch_no_reset(aec: AEC, log: list) -> Callable[[], None]:
    """A3: drop all _reset_filter_derived_state events.

    Captures the reason + frame each reset would have fired but does NOT
    perform the reset. Use the captured timeline to align ablation diffs.
    """
    orig = aec._reset_filter_derived_state

    def patched(reason: str = 'plateau', preserve_render_ema: bool = True):
        log.append(('skipped_reset', getattr(aec, '_frame_idx_trace', -1), reason))
        return  # no-op
    aec._reset_filter_derived_state = patched
    return lambda: setattr(aec, '_reset_filter_derived_state', orig)


def _patch_no_pbfdkf_update(aec: AEC) -> Callable[[], None]:
    """A5: PBFDKF Kalman/NLMS weight update off entirely.

    Main filter coefficients, H_error, P stay at whatever they were post
    warm-up. Echo estimate from `self.W @ X_buf` still flows. Shadow path
    is untouched (A2 covers that).
    """
    if not hasattr(aec.filter, '_update_weights_aec3'):
        return lambda: None
    orig_aec3 = aec.filter._update_weights_aec3
    orig_nlms = aec.filter._update_weights

    def patched_aec3(*args, **kwargs):
        return
    def patched_nlms(*args, **kwargs):
        return
    aec.filter._update_weights_aec3 = patched_aec3
    aec.filter._update_weights = patched_nlms

    def restore():
        aec.filter._update_weights_aec3 = orig_aec3
        aec.filter._update_weights = orig_nlms
    return restore


def _patch_fixed_delay(aec: AEC, log: list) -> Callable[[], None]:
    """A4: freeze delay estimator — block any delay adjustments.

    delay_first / delay_shift paths in orchestrator call
    `_reset_filter_derived_state(reason='delay_first'|'delay_shift')`. We
    monkey-patch the delay estimator's accumulate() to always report
    no change after the first acquisition.
    """
    if not hasattr(aec, 'delay_est') or aec.delay_est is None:
        return lambda: None
    de = aec.delay_est
    if not hasattr(de, 'accumulate'):
        return lambda: None
    orig = de.accumulate
    state = {'first_done': False, 'frozen_delay': None}

    def patched(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not state['first_done'] and getattr(de, 'is_solid', False):
            state['first_done'] = True
            state['frozen_delay'] = de.estimated_delay
            return out  # let the first solid detection through
        if state['first_done']:
            try:
                de._estimated_delay = state['frozen_delay']
            except Exception:
                pass
            log.append(('suppressed_delay_update',
                        getattr(aec, '_frame_idx_trace', -1)))
        return out
    de.accumulate = patched
    return lambda: setattr(de, 'accumulate', orig)


def _patch_no_sat(aec: AEC) -> Callable[[], None]:
    """A6: short-circuit saturation gating — clamp _saturation_level=0.

    Disables soft-clip + main_mu freeze + shadow_rise EPC false-trip paths
    that key off _saturation_level. Saturation detector still runs but its
    output is overwritten to 0 right after _compute_mu_scale.
    """
    orig = aec.process

    def patched(*args, **kwargs):
        out = orig(*args, **kwargs)
        aec._saturation_level = 0.0
        return out
    aec.process = patched
    return lambda: setattr(aec, 'process', orig)


def _patch_no_shadow_decisions(cfg: AecConfig) -> None:
    """A2: kill shadow path via config (no monkey-patch needed).

    enable_shadow=False removes the shadow filter entirely; regime handler
    boost_q / reverse_copy / pause_main decisions all return False
    (handler.update() short-circuits when shadow_err is the init sentinel).
    """
    cfg.enable_shadow = False


# ---------------------------------------------------------------------------
# Per-frame snapshot helper
# ---------------------------------------------------------------------------


def _snapshot(aec: AEC, frame: int) -> FrameSnapshot:
    d = aec._diag
    H = aec.filter.H_error_per_bin if hasattr(aec.filter, 'H_error_per_bin') else None
    P = aec.filter.P if hasattr(aec.filter, 'P') else None
    return FrameSnapshot(
        frame=frame,
        mu_scale=float(d.get('mu_scale', 0.0)),
        main_paused=int(bool(d.get('main_paused', False))),
        boost_q=int(d.get('boost_q_fired_ema', 0) > 0)
            if 'boost_q_fired_ema' in d else 0,
        reverse_copy=0,  # captured by listener below
        saturation_level=float(d.get('saturation_level', 0.0)),
        epc_active=int(bool(d.get('epc_active', False))),
        converged=int(bool(d.get('converged', False))),
        h_error_mean=float(np.mean(H)) if H is not None else 0.0,
        h_error_max=float(np.max(H)) if H is not None else 0.0,
        p_mean=float(np.mean(P)) if P is not None else 0.0,
        w_norm=float(d.get('filter_w_norm', 0.0)),
        raw_err_pwr=float(getattr(aec, 'raw_error_power', 0.0)),
    )


# ---------------------------------------------------------------------------
# Render driver
# ---------------------------------------------------------------------------


def _stft_psd(x: np.ndarray) -> np.ndarray:
    """Hann-windowed STFT magnitude squared, shape (n_frames, n_bins)."""
    win = np.hanning(FFT_PSD).astype(np.float32)
    n = len(x)
    n_frames = max(0, (n - FFT_PSD) // HOP_PSD + 1)
    out = np.empty((n_frames, FFT_PSD // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * HOP_PSD: i * HOP_PSD + FFT_PSD] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def run_one(name: str, mic: np.ndarray, ref: np.ndarray,
            patch_fn: Callable[[AEC, AecConfig, list], list],
            out_dir: str, write_wav: bool,
            movement: bool = False, no_shadow: bool = False) -> dict:
    """Render `(mic, ref)` once under ablation `name`. Return summary dict."""
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    cfg.enable_res = False
    cfg.enable_cng = False
    cfg.return_res_context = False
    if movement:
        cfg.enable_delay_est = True
        cfg.delay_est_period_s = 0.25
        cfg.delay_est_init_s = 0.2
    if no_shadow:
        cfg.enable_shadow = False

    extra_events: list = []
    aec = AEC(cfg)
    aec._frame_idx_trace = 0
    restores = patch_fn(aec, cfg, extra_events)

    n = min(len(mic), len(ref))
    n = (n // BLOCK) * BLOCK
    mic = mic[:n].astype(np.float32, copy=False)
    ref = ref[:n].astype(np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)

    snaps: list[FrameSnapshot] = []
    reverse_copy_frames: list[int] = []
    boost_q_frames: list[int] = []
    main_paused_frames: list[int] = []

    n_blocks = n // BLOCK
    for i in range(n_blocks):
        s = i * BLOCK
        aec._frame_idx_trace = i
        out[s:s + BLOCK] = aec.process(mic[s:s + BLOCK], ref[s:s + BLOCK])
        snap = _snapshot(aec, i)
        # Catch single-frame regime-handler events from the orchestrator's
        # diag block (line 2109 group). reverse_copy_fired is in the per-
        # frame trace dict only when the trace flag is on; for now read
        # the regime handler directly.
        rh = aec._regime_handler
        if getattr(rh, '_last_decision_reverse_copy', False):
            reverse_copy_frames.append(i)
            snap.reverse_copy = 1
        if snap.main_paused:
            main_paused_frames.append(i)
        snaps.append(snap)

    for restore in (restores if isinstance(restores, list) else [restores]):
        try:
            restore()
        except Exception:
            pass

    # --- PSD diff ---------------------------------------------------------
    mic_psd = _stft_psd(mic)
    nores_psd = _stft_psd(out)
    extra = np.maximum(nores_psd - mic_psd, 0.0)
    n_psd = extra.shape[0]
    extra_sum = extra.sum(axis=1)
    extra_lf = extra[:, :BIN_LF_HI + 1].sum(axis=1)
    extra_mf = extra[:, BIN_MF_LO:BIN_MF_HI + 1].sum(axis=1)
    top_idx = np.argsort(extra_sum)[-TOP_N:][::-1].tolist()
    # Per-frame neighborhood event lookup
    top_frames = []
    for fi in top_idx:
        # Map STFT frame to AEC block (1:1 hop=160)
        ai = min(fi, len(snaps) - 1)
        ev_window = []
        for j in range(max(0, ai - 5), min(len(snaps), ai + 6)):
            s = snaps[j]
            ev_window.append({
                'f': s.frame,
                'mu': round(s.mu_scale, 4),
                'pause': s.main_paused,
                'rev': s.reverse_copy,
                'boost': s.boost_q,
                'sat': round(s.saturation_level, 3),
                'epc': s.epc_active,
                'conv': s.converged,
                'h_mean': round(s.h_error_mean, 6),
                'p_mean': round(s.p_mean, 6),
                'w_norm': round(s.w_norm, 4),
            })
        top_frames.append({
            'psd_frame': int(fi),
            'extra_sum': float(extra_sum[fi]),
            'extra_lf': float(extra_lf[fi]),
            'extra_mf': float(extra_mf[fi]),
            'event_window': ev_window,
        })

    # Reset events (A3 only; otherwise will be empty). Other ablations may
    # log 2-tuples so be defensive about the shape.
    resets = []
    delay_suppressed = []
    for evt in extra_events:
        if not isinstance(evt, tuple) or not evt:
            continue
        if evt[0] == 'skipped_reset' and len(evt) >= 3:
            resets.append({'frame': int(evt[1]), 'reason': evt[2]})
        elif evt[0] == 'suppressed_delay_update' and len(evt) >= 2:
            delay_suppressed.append(int(evt[1]))

    summary = {
        'ablation': name,
        'version': __version__,
        'n_samples': int(n),
        'n_blocks': int(n_blocks),
        'n_psd_frames': int(n_psd),
        'sample_rate': SR,
        'fft_psd': FFT_PSD,
        'hop_psd': HOP_PSD,
        'extra_psd_sum_total': float(extra_sum.sum()),
        'extra_psd_sum_max': float(extra_sum.max() if n_psd else 0.0),
        'extra_psd_sum_p99': float(np.percentile(extra_sum, 99)) if n_psd else 0.0,
        'extra_psd_sum_mean': float(extra_sum.mean()) if n_psd else 0.0,
        'extra_lf_total': float(extra_lf.sum()),
        'extra_mf_total': float(extra_mf.sum()),
        'reverse_copy_count': len(reverse_copy_frames),
        'boost_q_count': sum(s.boost_q for s in snaps),
        'main_paused_frames_count': len(main_paused_frames),
        'epc_active_frames_count': sum(s.epc_active for s in snaps),
        'saturation_max': max((s.saturation_level for s in snaps), default=0.0),
        'sample_w_norm_first': snaps[0].w_norm if snaps else 0.0,
        'sample_w_norm_last': snaps[-1].w_norm if snaps else 0.0,
        'sample_h_error_mean_first': snaps[0].h_error_mean if snaps else 0.0,
        'sample_h_error_mean_last': snaps[-1].h_error_mean if snaps else 0.0,
        'reverse_copy_frames_sample': reverse_copy_frames[:50],
        'main_paused_runs_sample': main_paused_frames[:50],
        'skipped_resets': resets[:50],
        'suppressed_delay_updates_count': len(delay_suppressed),
        'top_frames': top_frames,
    }

    json_path = os.path.join(out_dir, f'{name}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if write_wav:
        sf.write(os.path.join(out_dir, f'{name}_nores.wav'),
                 out.astype(np.float32), SR)
    return summary


ABLATIONS: dict[str, Callable[[AEC, AecConfig, list], list]] = {
    'A0_baseline':
        lambda aec, cfg, log: [],
    'A1_freeze_adapt':
        lambda aec, cfg, log: [_patch_freeze_mu(aec)],
    'A2_no_shadow':
        lambda aec, cfg, log: [],   # cfg already mutated pre-init below
    'A3_no_reset':
        lambda aec, cfg, log: [_patch_no_reset(aec, log)],
    'A4_fixed_delay':
        lambda aec, cfg, log: [_patch_fixed_delay(aec, log)],
    'A5_no_pbfdkf_update':
        lambda aec, cfg, log: [_patch_no_pbfdkf_update(aec)],
    'A6_no_sat':
        lambda aec, cfg, log: [_patch_no_sat(aec)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mic', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--out', default='out_nores_artifact_debug')
    ap.add_argument('--ablations',
                    default='A0_baseline,A1_freeze_adapt,A2_no_shadow,'
                            'A3_no_reset,A4_fixed_delay,A5_no_pbfdkf_update,A6_no_sat')
    ap.add_argument('--write-wav', action='store_true',
                    help='Save nores wav per ablation locally (gitignored).')
    ap.add_argument('--movement', action='store_true',
                    help='Enable online delay estimation (movement cases).')
    ap.add_argument('--pre-align', action='store_true',
                    help='Pre-align ref to mic via xcorr before AEC (matches '
                         'eval_aec_challenge.py default).')
    args = ap.parse_args()

    mic, sr_mic = sf.read(args.mic)
    ref, sr_ref = sf.read(args.ref)
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    assert sr_mic == SR == sr_ref, f'expected {SR} Hz, got mic={sr_mic} ref={sr_ref}'

    if args.pre_align:
        from eval_aec_challenge import estimate_delay
        n0 = min(len(mic), len(ref))
        delay = estimate_delay(mic[:n0], ref[:n0], SR)
        if 0 < delay < n0:
            ref_a = np.zeros(n0, dtype=np.float32)
            ref_a[delay:] = ref[:n0 - delay]
            ref = ref_a
            print(f'  pre-aligned ref by {delay} samples ({delay/SR*1000:.1f} ms)')

    out_dir = os.path.join(args.out, args.stem)
    os.makedirs(out_dir, exist_ok=True)

    all_summaries = {}
    for name in args.ablations.split(','):
        name = name.strip()
        if name not in ABLATIONS:
            print(f'  WARNING: unknown ablation {name!r}, skipped')
            continue
        print(f'  rendering {name} ...')
        # A2 needs the cfg mutated BEFORE AEC() construction; run_one
        # builds a fresh cfg per call. Wrap the patch_fn to also mutate cfg.
        patch_fn = ABLATIONS[name]
        no_shadow = (name == 'A2_no_shadow')
        summary = run_one(name, mic, ref, patch_fn, out_dir, args.write_wav,
                          movement=args.movement, no_shadow=no_shadow)
        all_summaries[name] = summary

    # Diff table vs baseline (if present)
    base = all_summaries.get('A0_baseline')
    table = []
    keys = ('extra_psd_sum_total', 'extra_psd_sum_max',
            'extra_psd_sum_p99', 'extra_lf_total', 'extra_mf_total')
    for name, s in all_summaries.items():
        row = {'ablation': name}
        for k in keys:
            v = s.get(k)
            row[k] = round(v, 6) if isinstance(v, float) else v
            if base and name != 'A0_baseline':
                b = base.get(k, 0.0) or 1e-12
                row[k + '_ratio_vs_base'] = round((v or 0) / b, 4)
        table.append(row)

    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'stem': args.stem,
            'version': __version__,
            'diff_table': table,
        }, f, indent=2)
    print(f'\n  summary.json written → {summary_path}')
    print('  diff table (vs A0_baseline):')
    for row in table:
        print('   ', row)


if __name__ == '__main__':
    main()
