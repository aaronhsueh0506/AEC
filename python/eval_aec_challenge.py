#!/usr/bin/env python3
"""
Evaluate AEC on AEC Challenge dataset.
- farend_singletalk: ERLE metric (echo removal)
- nearend_singletalk: SDR metric (near-end preservation)
- doubletalk: ERLE metric (both)

Usage:
    python3 eval_aec_challenge.py ../wav/aec_challenge/ --aec3 --speex
"""
import numpy as np
import soundfile as sf
import argparse
import json
import os
import sys
import re
import io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

_ENABLE_CNG = False  # global CNG flag, set by --cng CLI arg
_CASES_LIST = None   # Tier-1 subset stem set; None = full 800-case rendering


def _filter_mic_files(mic_files, tag):
    """Restrict mic_files to stems in _CASES_LIST (no-op when None)."""
    if _CASES_LIST is None:
        return mic_files
    keep = [f for f in mic_files if f[:-len('_mic.wav')] in _CASES_LIST]
    print(f'[cases-list] {tag}: {len(keep)}/{len(mic_files)} matched', file=sys.stderr)
    return keep

# Try PESQ
try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

# Try SpeexDSP
try:
    from speexdsp import EchoCanceller
    HAS_SPEEX = True
except ImportError:
    HAS_SPEEX = False

# WebRTC AEC3 CLI
import subprocess, tempfile
_BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin')
AEC3_CLI = os.path.join(_BIN_DIR, 'aec3_cli')
if not os.path.isfile(AEC3_CLI):
    AEC3_CLI = '/tmp/webrtc-ap/aec3_cli'
HAS_AEC3 = os.path.isfile(AEC3_CLI) and os.access(AEC3_CLI, os.X_OK)

# WebRTC AEC3 Linear CLI (outputs both full and linear-only)
AEC3_LINEAR_CLI = os.path.join(_BIN_DIR, 'aec3_linear_cli')
if not os.path.isfile(AEC3_LINEAR_CLI):
    AEC3_LINEAR_CLI = '/tmp/webrtc-ap/aec3_linear_cli'
HAS_AEC3_LINEAR = os.path.isfile(AEC3_LINEAR_CLI) and os.access(AEC3_LINEAR_CLI, os.X_OK)

# WebRTC old AEC (AEC2) CLI
OLD_AEC_CLI = os.path.join(_BIN_DIR, 'old_aec_cli')
HAS_OLD_AEC = os.path.isfile(OLD_AEC_CLI) and os.access(OLD_AEC_CLI, os.X_OK)


def run_old_aec(mic_path, ref_path, sr):
    """Run WebRTC old AEC (AEC2)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        out_path = f.name
    try:
        r = subprocess.run([OLD_AEC_CLI, mic_path, ref_path, out_path],
                           capture_output=True, timeout=30)
        if r.returncode == 0 and os.path.isfile(out_path):
            data, _ = sf.read(out_path)
            return data.astype(np.float32)
    except Exception:
        pass
    finally:
        if os.path.isfile(out_path):
            os.unlink(out_path)
    return None


def estimate_delay(mic, ref, sr, max_delay_ms=1024.0):
    """Pre-compute delay using full-signal cross-correlation.

    Uses the entire signal for maximum accuracy.
    Plain cross-correlation (no whitening) is most reliable for reverberant data.

    max_delay_ms=1024 matches the online F-DelayTrack search window so the
    pre-align never lands further away than the in-pipeline tracker can
    correct. Override via env var AEC_MAX_DELAY_MS for A/B / regression work.
    """
    _env_override = os.environ.get('AEC_MAX_DELAY_MS')
    if _env_override is not None:
        max_delay_ms = float(_env_override)
    max_d = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref))
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)

    # FFT-based cross-correlation (full signal)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)

    # Primary: GCC-PHAT (sharp peak for most cases)
    cross_phat = cross / (np.abs(cross) + 1e-10)
    xcorr_phat = np.fft.irfft(cross_phat, n=fft_size)
    max_search = min(max_d, fft_size // 2)
    peak_val_phat = np.max(np.abs(xcorr_phat[:max_search + 1]))
    peak_idx_phat = int(np.argmax(np.abs(xcorr_phat[:max_search + 1])))

    # Confidence: peak relative to RMS (high = reliable, low = noise)
    rms = np.sqrt(np.mean(xcorr_phat[:max_search + 1] ** 2))
    confidence = peak_val_phat / (rms + 1e-10)

    # Low confidence → fallback to plain xcorr
    if confidence < 5.0:
        xcorr_plain = np.fft.irfft(cross, n=fft_size)
        delay = int(np.argmax(np.abs(xcorr_plain[:max_search + 1])))
    else:
        delay = peak_idx_phat
    return delay


def run_ours(mic, ref, sr, fl, enable_res=True, preset=None,
             is_movement=False, **config_overrides):
    n = min(len(mic), len(ref))

    # Always pre-align with global delay
    delay = estimate_delay(mic, ref, sr)
    if delay > 0 and delay < n:
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
    else:
        ref_aligned = ref[:n]

    # Movement files: also enable online delay estimation for tracking changes
    if is_movement:
        delay_est_kw = dict(enable_delay_est=True,
                            delay_est_period_s=0.25,
                            delay_est_init_s=0.2)
    else:
        delay_est_kw = dict(enable_delay_est=False)

    # Allow CLI override of gain type via env var or config_overrides
    env_gain_type = os.environ.get('AEC_GAIN_TYPE')
    if env_gain_type and 'res_gain_type' not in config_overrides:
        config_overrides['res_gain_type'] = env_gain_type
    # CNG override from global flag
    if _ENABLE_CNG and 'enable_cng' not in config_overrides:
        config_overrides['enable_cng'] = True
    # v3.22: ERLE render-x2 PSD-scale fix (revives reverb model) — DEFAULT ON.
    # AEC_ERLE_X2_SCALE=0 forces OFF to reproduce the pre-revival baseline.
    _x2env = os.environ.get('AEC_ERLE_X2_SCALE')
    if _x2env in ('0', '1') and 'erle_render_x2_psd_scale' not in config_overrides:
        config_overrides['erle_render_x2_psd_scale'] = (_x2env == '1')
    _rts = os.environ.get('AEC_REVERB_TAIL_STRENGTH')
    if _rts is not None and 'reverb_tail_strength' not in config_overrides:
        config_overrides['reverb_tail_strength'] = float(_rts)
    # v3.22 E1: windowed Y2 for ERLE Y2/E2 coordinate consistency (DEFAULT ON).
    # AEC_ERLE_Y2_WIN=0 forces OFF to reproduce the pre-E1 (old) baseline.
    _e1env = os.environ.get('AEC_ERLE_Y2_WIN')
    if _e1env in ('0', '1') and 'erle_windowed_capture_psd' not in config_overrides:
        config_overrides['erle_windowed_capture_psd'] = (_e1env == '1')
    # v3.22 E2: output base = raw capture Y when linear filter unusable (DEFAULT ON).
    # AEC_OUT_CAPTURE_UNUSABLE=0 forces OFF.
    _e2env = os.environ.get('AEC_OUT_CAPTURE_UNUSABLE')
    if _e2env in ('0', '1') and 'output_capture_when_linear_unusable' not in config_overrides:
        config_overrides['output_capture_when_linear_unusable'] = (_e2env == '1')
    # B2 (v3.22): dne_loud_nearend_enr_relax_enabled — W4 re-audit flag.
    # AEC_DNE_LOUD_NE=1 → True (relax ENR threshold for moderate DT).
    _dne_loud_env = os.environ.get('AEC_DNE_LOUD_NE')
    if _dne_loud_env in ('0', '1') and 'dne_loud_nearend_enr_relax_enabled' not in config_overrides:
        config_overrides['dne_loud_nearend_enr_relax_enabled'] = (_dne_loud_env == '1')
    # B5 (v3.22): enable_lf_filter_failure_r2_injection — candidate-B re-audit.
    # AEC_LF_R2_INJECT=1 → True (inject R² for LF filter-failure frames).
    _lf_r2_env = os.environ.get('AEC_LF_R2_INJECT')
    if _lf_r2_env in ('0', '1') and 'enable_lf_filter_failure_r2_injection' not in config_overrides:
        config_overrides['enable_lf_filter_failure_r2_injection'] = (_lf_r2_env == '1')
    # W1' (v3.22): erle_startup_follows_convergence — drop fixed session-startup
    # ERLE gate; update as soon as converged_filter=True. Fixes ERLE/usable_linear
    # desync that causes near-end over-suppression at session start + after EPC.
    # AEC_ERLE_STARTUP_CONV=1 → True.
    _erle_startup_env = os.environ.get('AEC_ERLE_STARTUP_CONV')
    if _erle_startup_env in ('0', '1') and 'erle_startup_follows_convergence' not in config_overrides:
        config_overrides['erle_startup_follows_convergence'] = (_erle_startup_env == '1')
    # hf_min_gain (v3.22): HF gain floor (−15 dB) during DNE active. Protects
    # NE HF formants when filter hasn't converged at HF.
    # AEC_HF_MIN_GAIN=1 → True.
    _hf_min_env = os.environ.get('AEC_HF_MIN_GAIN')
    if _hf_min_env in ('0', '1') and 'hf_min_gain_floor_during_dne_enabled' not in config_overrides:
        config_overrides['hf_min_gain_floor_during_dne_enabled'] = (_hf_min_env == '1')
    # ser_floor (v3.22 D2): SER-based gain floor — nearend preservation without DT detector.
    # AEC_SER_FLOOR=1 → ser_floor_enabled=True (strength defaults to 0.5).
    _ser_floor_env = os.environ.get('AEC_SER_FLOOR')
    if _ser_floor_env in ('0', '1') and 'ser_floor_enabled' not in config_overrides:
        config_overrides['ser_floor_enabled'] = (_ser_floor_env == '1')
    # soft_nearend_blend (v3.22 D3): sigmoid ENR interpolation replaces binary DNE switch.
    # AEC_SOFT_NE_BLEND=1 → soft_nearend_blend_enabled=True.
    _soft_ne_env = os.environ.get('AEC_SOFT_NE_BLEND')
    if _soft_ne_env in ('0', '1') and 'soft_nearend_blend_enabled' not in config_overrides:
        config_overrides['soft_nearend_blend_enabled'] = (_soft_ne_env == '1')
    # AEC_MODE=PBFDAF for filter-class comparison (default PBFDKF)
    _mode_env = os.environ.get('AEC_MODE', 'PBFDKF').upper()
    _mode = AecMode.PBFDAF if _mode_env == 'PBFDAF' else AecMode.PBFDKF
    _use_kalman = (_mode == AecMode.PBFDKF)
    common_kw = dict(sample_rate=sr, mode=_mode,
                     filter_length=fl, enable_dtd=False,
                     enable_shadow=True, enable_res=enable_res,
                     use_kalman=_use_kalman,
                     **delay_est_kw, **config_overrides)
    if preset is not None:
        from aec import AecPreset
        config = AecConfig.from_preset(preset, **common_kw)
    else:
        config = AecConfig(**common_kw)
    # Per-case CNG determinism: seed numpy before each AEC instantiation
    # so CNG noise is identical across runs (run-to-run AECMOS Δ otherwise
    # masks small code-induced changes).
    np.random.seed(0)
    aec = AEC(config)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], ref_aligned[pos:pos+hop])
        pos += hop
    return out[:n]


def run_speex(mic, ref, sr, fl=2048):
    frame_size = 256
    ec = EchoCanceller.create(frame_size, fl, sr)
    n = min(len(mic), len(ref))
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + frame_size <= n:
        mi = (mic[pos:pos+frame_size] * 32767).clip(-32768, 32767).astype(np.int16)
        ri = (ref[pos:pos+frame_size] * 32767).clip(-32768, 32767).astype(np.int16)
        ob = ec.process(mi.tobytes(), ri.tobytes())
        out[pos:pos+frame_size] = np.frombuffer(ob, dtype=np.int16).astype(np.float32) / 32767.0
        pos += frame_size
    return out[:n]


def run_aec3(mic_path, ref_path, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        out_path = tmp.name
    try:
        r = subprocess.run([AEC3_CLI, mic_path, ref_path, out_path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        o, _ = sf.read(out_path)
        return o.astype(np.float32)
    except:
        return None
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def run_aec3_linear(mic_path, ref_path, sr):
    """Run AEC3 with linear output. Returns (full_output, linear_output)."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp2:
        out_path = tmp1.name
        lin_path = tmp2.name
    try:
        r = subprocess.run([AEC3_LINEAR_CLI, mic_path, ref_path, out_path, lin_path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None, None
        full, _ = sf.read(out_path)
        linear, _ = sf.read(lin_path)
        return full.astype(np.float32), linear.astype(np.float32)
    except:
        return None, None
    finally:
        for p in [out_path, lin_path]:
            if os.path.exists(p):
                os.unlink(p)


def compute_erle(mic, output):
    mic_pwr = np.mean(mic ** 2)
    out_pwr = np.mean(output ** 2)
    if out_pwr < 1e-20:
        return 60.0
    return 10.0 * np.log10(mic_pwr / (out_pwr + 1e-20))


def compute_sdr(mic, output):
    """Signal-to-Distortion Ratio: how well near-end is preserved.

    SDR = 10*log10(sum(mic²) / sum((mic - output)²))
    Higher = less distortion. Inf means perfect passthrough.
    """
    n = min(len(mic), len(output))
    mic, output = mic[:n], output[:n]
    sig_pwr = np.mean(mic ** 2)
    dist_pwr = np.mean((mic - output) ** 2)
    if dist_pwr < 1e-20:
        return 60.0
    return 10.0 * np.log10(sig_pwr / (dist_pwr + 1e-20))


def _is_movement(filename):
    """Check if a filename is a movement case."""
    return '_with_movement_' in filename


def compute_pesq(ref, deg, sr):
    if not HAS_PESQ:
        return None
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    mode = 'wb' if sr >= 16000 else 'nb'
    try:
        return pesq_fn(sr, ref, deg, mode)
    except:
        return None


def eval_farend_singletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=None, do_old_aec=False, chunk_idx=0, n_chunks=1):
    """Evaluate farend_singletalk with ERLE."""
    sc_dir = os.path.join(base_dir, 'farend_singletalk')
    if not os.path.isdir(sc_dir):
        print("No farend_singletalk directory found")
        return

    # Find all mic files (including _with_movement_ variants)
    mic_files = sorted([f for f in os.listdir(sc_dir)
                        if '_farend_singletalk' in f and f.endswith('_mic.wav')])
    mic_files = _filter_mic_files(mic_files, 'farend_singletalk')
    if n_chunks > 1:
        mic_files = mic_files[chunk_idx::n_chunks]
    if not mic_files:
        print("No farend_singletalk files found")
        return

    n_move = sum(1 for f in mic_files if '_with_movement_' in f)
    print(f"\n{'='*60}")
    print(f"FAREND SINGLETALK — ERLE ({len(mic_files)} cases, {n_move} with movement)")
    print(f"{'='*60}")

    hdr = f"{'Case':>5} {'Mv':>2} {'Ours':>8}"
    if do_speex: hdr += f" {'Speex':>8}"
    if do_aec3:  hdr += f" {'AEC3':>8}"
    if do_aec3_linear: hdr += f" {'AEC3-Lin':>8}"
    if do_old_aec: hdr += f" {'OldAEC':>8}"
    print(hdr)
    print("-" * len(hdr))

    erles = {'ours': [], 'speex': [], 'aec3': [], 'aec3_linear': [], 'old_aec': []}

    for i, mf in enumerate(mic_files):
        uuid = mf[:mf.index('_farend_singletalk')]
        lpb_f = mf.replace('_mic.wav', '_lpb.wav')
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)

        mic, sr = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic, ref = mic.astype(np.float32), ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        movement = _is_movement(mf)
        mv_tag = 'M' if movement else ' '
        out_suffix = mf.replace('_mic.wav', '')  # uuid_farend_singletalk[_with_movement]
        # Ours
        output = run_ours(mic, ref, sr, fl, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours.wav"), output, sr)
        e_ours = compute_erle(mic, output)
        erles['ours'].append(e_ours)

        # Ours (no RES) — raw PBFDAF output
        output_nores = run_ours(mic, ref, sr, fl, enable_res=False, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours_nores.wav"), output_nores, sr)

        line = f"{i:>5} {mv_tag:>2} {e_ours:>8.1f}"

        # Speex
        if do_speex:
            out_sp = run_speex(mic, ref, sr)
            sf.write(os.path.join(out_dir, f"{out_suffix}_speex.wav"), out_sp, sr)
            e_sp = compute_erle(mic, out_sp)
            erles['speex'].append(e_sp)
            line += f" {e_sp:>8.1f}"

        # AEC3
        if do_aec3:
            out_a3 = run_aec3(mic_path, lpb_path, sr)
            if out_a3 is not None:
                out_a3 = out_a3[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3.wav"), out_a3, sr)
                e_a3 = compute_erle(mic, out_a3)
                erles['aec3'].append(e_a3)
                line += f" {e_a3:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # AEC3 Linear
        if do_aec3_linear:
            _, out_lin = run_aec3_linear(mic_path, lpb_path, sr)
            if out_lin is not None:
                out_lin = out_lin[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3_linear.wav"), out_lin, sr)
                e_lin = compute_erle(mic, out_lin)
                erles['aec3_linear'].append(e_lin)
                line += f" {e_lin:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # Old AEC (AEC2)
        if do_old_aec:
            out_oa = run_old_aec(mic_path, lpb_path, sr)
            if out_oa is not None:
                out_oa = out_oa[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_old_aec.wav"), out_oa, sr)
                e_oa = compute_erle(mic, out_oa)
                erles['old_aec'].append(e_oa)
                line += f" {e_oa:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        print(line)

    # Summary
    print("-" * len(hdr))
    summary = f"{'MEAN':>5} {np.mean(erles['ours']):>8.1f}"
    if do_speex and erles['speex']:
        summary += f" {np.mean(erles['speex']):>8.1f}"
    if do_aec3 and erles['aec3']:
        summary += f" {np.mean(erles['aec3']):>8.1f}"
    if do_aec3_linear and erles['aec3_linear']:
        summary += f" {np.mean(erles['aec3_linear']):>8.1f}"
    if do_old_aec and erles['old_aec']:
        summary += f" {np.mean(erles['old_aec']):>8.1f}"
    print(summary)


def eval_nearend_singletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=None, do_old_aec=False, chunk_idx=0, n_chunks=1):
    """Evaluate nearend_singletalk with SDR (near-end preservation)."""
    sc_dir = os.path.join(base_dir, 'nearend_singletalk')
    if not os.path.isdir(sc_dir):
        print("No nearend_singletalk directory found, skipping")
        return

    mic_files = sorted([f for f in os.listdir(sc_dir)
                        if '_nearend_singletalk' in f and f.endswith('_mic.wav')])
    mic_files = _filter_mic_files(mic_files, 'nearend_singletalk')
    if n_chunks > 1:
        mic_files = mic_files[chunk_idx::n_chunks]
    if not mic_files:
        print("No nearend_singletalk files found")
        return

    n_move = sum(1 for f in mic_files if '_with_movement_' in f)
    print(f"\n{'='*60}")
    print(f"NEAREND SINGLETALK — SDR ({len(mic_files)} cases, {n_move} with movement)")
    print(f"{'='*60}")

    hdr = f"{'Case':>5} {'Mv':>2} {'Ours':>8}"
    if do_speex: hdr += f" {'Speex':>8}"
    if do_aec3:  hdr += f" {'AEC3':>8}"
    if do_aec3_linear: hdr += f" {'AEC3-Lin':>8}"
    if do_old_aec: hdr += f" {'OldAEC':>8}"
    print(hdr)
    print("-" * len(hdr))

    sdrs = {'ours': [], 'speex': [], 'aec3': [], 'aec3_linear': [], 'old_aec': []}

    for i, mf in enumerate(mic_files):
        uuid = mf[:mf.index('_nearend_singletalk')]
        lpb_f = mf.replace('_mic.wav', '_lpb.wav')
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)

        mic, sr = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic, ref = mic.astype(np.float32), ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        movement = _is_movement(mf)
        mv_tag = 'M' if movement else ' '
        out_suffix = mf.replace('_mic.wav', '')
        # Ours
        output = run_ours(mic, ref, sr, fl, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours.wav"), output, sr)
        s_ours = compute_sdr(mic, output)
        sdrs['ours'].append(s_ours)

        # Ours (no RES) — raw PBFDAF output
        output_nores = run_ours(mic, ref, sr, fl, enable_res=False, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours_nores.wav"), output_nores, sr)

        line = f"{i:>5} {mv_tag:>2} {s_ours:>8.1f}"

        # Speex
        if do_speex:
            out_sp = run_speex(mic, ref, sr)
            sf.write(os.path.join(out_dir, f"{out_suffix}_speex.wav"), out_sp, sr)
            s_sp = compute_sdr(mic, out_sp)
            sdrs['speex'].append(s_sp)
            line += f" {s_sp:>8.1f}"

        # AEC3
        if do_aec3:
            out_a3 = run_aec3(mic_path, lpb_path, sr)
            if out_a3 is not None:
                out_a3 = out_a3[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3.wav"), out_a3, sr)
                s_a3 = compute_sdr(mic, out_a3)
                sdrs['aec3'].append(s_a3)
                line += f" {s_a3:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # AEC3 Linear
        if do_aec3_linear:
            _, out_lin = run_aec3_linear(mic_path, lpb_path, sr)
            if out_lin is not None:
                out_lin = out_lin[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3_linear.wav"), out_lin, sr)
                s_lin = compute_sdr(mic, out_lin)
                sdrs['aec3_linear'].append(s_lin)
                line += f" {s_lin:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # Old AEC (AEC2)
        if do_old_aec:
            out_oa = run_old_aec(mic_path, lpb_path, sr)
            if out_oa is not None:
                out_oa = out_oa[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_old_aec.wav"), out_oa, sr)
                s_oa = compute_sdr(mic, out_oa)
                sdrs['old_aec'].append(s_oa)
                line += f" {s_oa:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        print(line)

    # Summary
    print("-" * len(hdr))
    summary = f"{'MEAN':>5} {np.mean(sdrs['ours']):>8.1f}"
    if do_speex and sdrs['speex']:
        summary += f" {np.mean(sdrs['speex']):>8.1f}"
    if do_aec3 and sdrs['aec3']:
        summary += f" {np.mean(sdrs['aec3']):>8.1f}"
    if do_aec3_linear and sdrs['aec3_linear']:
        summary += f" {np.mean(sdrs['aec3_linear']):>8.1f}"
    if do_old_aec and sdrs['old_aec']:
        summary += f" {np.mean(sdrs['old_aec']):>8.1f}"
    print(summary)


def eval_doubletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=None, do_old_aec=False, chunk_idx=0, n_chunks=1):
    """Evaluate doubletalk with ERLE (real recordings from clean test set)."""
    sc_dir = os.path.join(base_dir, 'doubletalk')
    if not os.path.isdir(sc_dir):
        print("No doubletalk directory found")
        return

    # Find all mic files (including _with_movement_ variants)
    mic_files = sorted([f for f in os.listdir(sc_dir)
                        if '_doubletalk' in f and f.endswith('_mic.wav')])
    mic_files = _filter_mic_files(mic_files, 'doubletalk')
    if n_chunks > 1:
        mic_files = mic_files[chunk_idx::n_chunks]
    if not mic_files:
        print("No doubletalk files found")
        return

    n_move = sum(1 for f in mic_files if '_with_movement_' in f)
    print(f"\n{'='*60}")
    print(f"DOUBLETALK (real) — ERLE ({len(mic_files)} cases, {n_move} with movement)")
    print(f"{'='*60}")

    hdr = f"{'Case':>5} {'Mv':>2} {'Ours':>8}"
    if do_speex: hdr += f" {'Speex':>8}"
    if do_aec3:  hdr += f" {'AEC3':>8}"
    if do_aec3_linear: hdr += f" {'AEC3-Lin':>8}"
    if do_old_aec: hdr += f" {'OldAEC':>8}"
    print(hdr)
    print("-" * len(hdr))

    erles = {'ours': [], 'speex': [], 'aec3': [], 'aec3_linear': [], 'old_aec': []}

    for i, mf in enumerate(mic_files):
        uuid = mf[:mf.index('_doubletalk')]
        lpb_f = mf.replace('_mic.wav', '_lpb.wav')
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)

        mic, sr = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic, ref = mic.astype(np.float32), ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        movement = _is_movement(mf)
        mv_tag = 'M' if movement else ' '
        out_suffix = mf.replace('_mic.wav', '')
        # Ours
        output = run_ours(mic, ref, sr, fl, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours.wav"), output, sr)
        e_ours = compute_erle(mic, output)
        erles['ours'].append(e_ours)

        # Ours (no RES) — raw PBFDAF output
        output_nores = run_ours(mic, ref, sr, fl, enable_res=False, preset=preset, is_movement=movement)
        sf.write(os.path.join(out_dir, f"{out_suffix}_ours_nores.wav"), output_nores, sr)

        line = f"{i:>5} {mv_tag:>2} {e_ours:>8.1f}"

        # Speex
        if do_speex:
            out_sp = run_speex(mic, ref, sr)
            sf.write(os.path.join(out_dir, f"{out_suffix}_speex.wav"), out_sp, sr)
            e_sp = compute_erle(mic, out_sp)
            erles['speex'].append(e_sp)
            line += f" {e_sp:>8.1f}"

        # AEC3
        if do_aec3:
            out_a3 = run_aec3(mic_path, lpb_path, sr)
            if out_a3 is not None:
                out_a3 = out_a3[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3.wav"), out_a3, sr)
                e_a3 = compute_erle(mic, out_a3)
                erles['aec3'].append(e_a3)
                line += f" {e_a3:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # AEC3 Linear
        if do_aec3_linear:
            _, out_lin = run_aec3_linear(mic_path, lpb_path, sr)
            if out_lin is not None:
                out_lin = out_lin[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_aec3_linear.wav"), out_lin, sr)
                e_lin = compute_erle(mic, out_lin)
                erles['aec3_linear'].append(e_lin)
                line += f" {e_lin:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        # Old AEC (AEC2)
        if do_old_aec:
            out_oa = run_old_aec(mic_path, lpb_path, sr)
            if out_oa is not None:
                out_oa = out_oa[:n]
                sf.write(os.path.join(out_dir, f"{out_suffix}_old_aec.wav"), out_oa, sr)
                e_oa = compute_erle(mic, out_oa)
                erles['old_aec'].append(e_oa)
                line += f" {e_oa:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        print(line)

    # Summary
    print("-" * len(hdr))
    summary = f"{'MEAN':>5} {np.mean(erles['ours']):>8.1f}"
    if do_speex and erles['speex']:
        summary += f" {np.mean(erles['speex']):>8.1f}"
    if do_aec3 and erles['aec3']:
        summary += f" {np.mean(erles['aec3']):>8.1f}"
    if do_aec3_linear and erles['aec3_linear']:
        summary += f" {np.mean(erles['aec3_linear']):>8.1f}"
    if do_old_aec and erles['old_aec']:
        summary += f" {np.mean(erles['old_aec']):>8.1f}"
    print(summary)


def _run_eval_captured(func, *args, **kwargs):
    """Run an eval function and capture its stdout output."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


def _run_scenario(scenario_args):
    """Worker function for parallel execution (must be top-level for pickling)."""
    func_name, base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset, do_old_aec, enable_cng, cases_list, chunk_idx, n_chunks = scenario_args
    # Propagate flags to subprocess (globals are lost across ProcessPoolExecutor fork)
    global _ENABLE_CNG, _CASES_LIST
    _ENABLE_CNG = enable_cng
    _CASES_LIST = cases_list
    func = {'fs': eval_farend_singletalk,
            'ne': eval_nearend_singletalk,
            'dt': eval_doubletalk}[func_name]
    return (func_name, chunk_idx,
            _run_eval_captured(func, base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir,
                               preset=preset, do_old_aec=do_old_aec,
                               chunk_idx=chunk_idx, n_chunks=n_chunks))


def run_scenarios(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=None, parallel=False, do_old_aec=False, workers=6):
    """Run all three scenarios, optionally in parallel.

    `workers` controls per-case parallelism: when parallel=True, scenarios
    are sliced into n_chunks = max(1, workers // 3) chunks each, giving
    `n_chunks * 3` worker processes. Default workers=6 → 2 chunks per
    scenario → 6 concurrent processes (M-series 6-P-core throughput).
    """
    n_chunks = max(1, workers // 3)
    scenarios = []
    for sc in ('fs', 'ne', 'dt'):
        for ck in range(n_chunks):
            scenarios.append((sc, base_dir, fl, do_speex, do_aec3, do_aec3_linear,
                              out_dir, preset, do_old_aec, _ENABLE_CNG, _CASES_LIST,
                              ck, n_chunks))

    if parallel:
        results = {}
        max_w = max(workers, 3)
        with ProcessPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_run_scenario, s): (s[0], s[12]) for s in scenarios}
            for future in as_completed(futures):
                name, ck, output = future.result()
                results.setdefault(name, {})[ck] = output
        # Print in order: fs → ne → dt, chunk 0 → chunk N-1
        for key in ['fs', 'ne', 'dt']:
            if key in results:
                for ck in sorted(results[key].keys()):
                    if results[key][ck]:
                        print(results[key][ck], end='')
    else:
        eval_farend_singletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=preset, do_old_aec=do_old_aec)
        eval_nearend_singletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=preset, do_old_aec=do_old_aec)
        eval_doubletalk(base_dir, fl, do_speex, do_aec3, do_aec3_linear, out_dir, preset=preset, do_old_aec=do_old_aec)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AEC on AEC Challenge dataset')
    parser.add_argument('dataset_dir', help='aec_challenge/ directory')
    parser.add_argument('--filter', type=int, default=2048, help='Filter length')
    parser.add_argument('--speex', action='store_true', help='Also run SpeexDSP')
    parser.add_argument('--aec3', action='store_true', help='Also run WebRTC AEC3')
    parser.add_argument('--aec3-linear', action='store_true', help='Also run WebRTC AEC3 linear-only')
    parser.add_argument('--old-aec', action='store_true', help='Also run WebRTC old AEC (AEC2)')
    parser.add_argument('--preset', choices=['balanced'],
                        default=None, help='AEC preset (default: no preset)')
    parser.add_argument('--all-presets', action='store_true',
                        help='Run all 3 presets and compare')
    parser.add_argument('--parallel', action='store_true',
                        help='Run FS/NE/DT scenarios in parallel (3 processes)')
    parser.add_argument('-o', '--output-dir', default=None, help='Output directory')
    parser.add_argument('--gain-type', choices=['wiener', 'enr', 'spectral_sub'],
                        default=None, help='Override RES gain type')
    parser.add_argument('--cng', action='store_true', help='Enable comfort noise generation')
    parser.add_argument('--cases-list', default=None,
                        help='Stem-list file (one stem per line, # comments). '
                             'When present, restricts rendering to listed cases; '
                             'omitted = full 800-case rendering.')
    parser.add_argument('--workers', type=int, default=6,
                        help='Number of parallel worker processes (default 6). '
                             'Each scenario is sliced into max(1, workers//3) chunks. '
                             'M-series CPU recommendation: 6 (≈ P-core count) for '
                             'best wall-time without thermal throttling.')
    args = parser.parse_args()

    global _ENABLE_CNG, _CASES_LIST
    _ENABLE_CNG = args.cng
    if args.cases_list:
        with open(args.cases_list) as fh:
            _CASES_LIST = {ln.strip() for ln in fh
                           if ln.strip() and not ln.lstrip().startswith('#')}
        print(f'[cases-list] loaded {len(_CASES_LIST)} stems from {args.cases_list}',
              file=sys.stderr)

    base_dir = os.path.abspath(args.dataset_dir)
    out_dir = args.output_dir or os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)

    do_speex = args.speex and HAS_SPEEX
    do_aec3 = args.aec3 and HAS_AEC3
    do_aec3_linear = args.aec3_linear and HAS_AEC3_LINEAR
    do_old_aec = args.old_aec and HAS_OLD_AEC

    if args.speex and not HAS_SPEEX:
        print("Warning: speexdsp not installed")
    if args.aec3 and not HAS_AEC3:
        print(f"Warning: AEC3 CLI not found at {AEC3_CLI}")
    if args.aec3_linear and not HAS_AEC3_LINEAR:
        print(f"Warning: AEC3 Linear CLI not found at {AEC3_LINEAR_CLI}")
    if args.old_aec and not HAS_OLD_AEC:
        print(f"Warning: Old AEC CLI not found at {OLD_AEC_CLI}")
    if not HAS_PESQ:
        print("Warning: pesq not installed. pip3 install pesq")
    if args.parallel:
        print("Running scenarios in parallel (3 processes)...")

    from aec import AecPreset
    if args.all_presets:
        for p in AecPreset:
            preset_dir = os.path.join(out_dir, p.value)
            os.makedirs(preset_dir, exist_ok=True)
            print(f"\n{'#'*60}")
            print(f"  PRESET: {p.value.upper()}")
            print(f"{'#'*60}")
            run_scenarios(base_dir, args.filter, do_speex, do_aec3, do_aec3_linear, preset_dir,
                          preset=p, parallel=args.parallel, do_old_aec=do_old_aec, workers=args.workers)
    else:
        preset = AecPreset(args.preset) if args.preset else None
        run_scenarios(base_dir, args.filter, do_speex, do_aec3, do_aec3_linear, out_dir,
                      preset=preset, parallel=args.parallel, do_old_aec=do_old_aec, workers=args.workers)

    print(f"\nOutput saved to {out_dir}")


if __name__ == '__main__':
    main()
