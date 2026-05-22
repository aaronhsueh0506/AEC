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
    # P1.0 Plan A attribution: env vars to selectively revert sub-changes
    for _flag, _env in [
        ('plan_a_kernel_tight', 'AEC_PLAN_A_KERNEL'),
        ('plan_a_hf_cap_2k', 'AEC_PLAN_A_HF_CAP'),
        ('plan_a_stat_mask_7k', 'AEC_PLAN_A_STAT_MASK'),
    ]:
        _v = os.environ.get(_env)
        if _v is not None and _flag not in config_overrides:
            config_overrides[_flag] = _v.lower() not in ('0', 'false', 'off', 'no')
    # P4B γ-primary dt_per_bin (default OFF)
    if 'AEC_PLAN_B' in os.environ and 'plan_b_dt_per_bin_gamma' not in config_overrides:
        config_overrides['plan_b_dt_per_bin_gamma'] = (
            os.environ['AEC_PLAN_B'].lower() not in ('0', 'false', 'off', 'no'))
    # F2.1 EPC state reset (default OFF)
    if ('AEC_USE_EPC_RESET' in os.environ
            and 'use_epc_state_reset' not in config_overrides):
        config_overrides['use_epc_state_reset'] = (
            os.environ['AEC_USE_EPC_RESET'].lower() not in ('0', 'false', 'off', 'no'))
    # F2.2 — diverged-streak EMA + P3h reset enable
    if ('AEC_DIVERGED_STREAK_EMA' in os.environ
            and 'use_diverged_streak_ema' not in config_overrides):
        config_overrides['use_diverged_streak_ema'] = (
            os.environ['AEC_DIVERGED_STREAK_EMA'].lower() not in ('0', 'false', 'off', 'no'))
    if ('AEC_DIVERGED_RESET' in os.environ
            and 'diverged_reset_enabled' not in config_overrides):
        config_overrides['diverged_reset_enabled'] = (
            os.environ['AEC_DIVERGED_RESET'].lower() not in ('0', 'false', 'off', 'no'))
    # B5 — shadow R-reset symmetric extension (v3.11 Phase 1 Sprint 1-2)
    if ('AEC_SHADOW_R_RESET' in os.environ
            and 'shadow_r_reset_enabled' not in config_overrides):
        config_overrides['shadow_r_reset_enabled'] = (
            os.environ['AEC_SHADOW_R_RESET'].lower() not in ('0', 'false', 'off', 'no'))
    # B6 — state-aware shadow µ schedule (v3.11 Phase 1 Sprint 3-4)
    if ('AEC_SHADOW_MU_STATE' in os.environ
            and 'shadow_mu_state_aware' not in config_overrides):
        config_overrides['shadow_mu_state_aware'] = (
            os.environ['AEC_SHADOW_MU_STATE'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.21.5 Phase 1 Sprint A — AEC3 echo_remover.cc:495-501 E2=min(E2,Y2) clamp
    if ('AEC_E2_Y2_CLAMP' in os.environ
            and 'e2_y2_clamp_enabled' not in config_overrides):
        config_overrides['e2_y2_clamp_enabled'] = (
            os.environ['AEC_E2_Y2_CLAMP'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.21.6 nores artifact debug 2026-05-22 — ref HPF policy A/B.
    # Code-on-main default is enable_highpass_ref=True (4a41675 revert
    # CANNOT SHIP at OFF per DT_movement Δdeg). User-directed intended
    # policy 2026-05-22 = mic ON / ref OFF; production gate requires
    # 800-case AECMOS A/B (this env var controls the A/B). Default
    # unset → config default → byte-equal preserved.
    if ('AEC_HPF_REF' in os.environ
            and 'enable_highpass_ref' not in config_overrides):
        config_overrides['enable_highpass_ref'] = (
            os.environ['AEC_HPF_REF'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.21.5 Sprint B (CLOSED rejected; default-True restored).
    # See docs/v3_21_5_phase1_b_stationarity_gate_verdict.md and the config
    # field comment in config.py:237 for the load-bearing safety-net evidence
    # (60+ DT cohort-tail cases, xQEUtY2 worst -0.602 deg). Set =0 to opt
    # into AEC3-default-off research; v3.21.6 P4 will re-test after companion
    # mechanisms ship.
    if ('AEC_STATIONARITY_ZERO' in os.environ
            and 'aec3_post_stationarity_zero_enabled' not in config_overrides):
        config_overrides['aec3_post_stationarity_zero_enabled'] = (
            os.environ['AEC_STATIONARITY_ZERO'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.21.5 Sprint C2 — per-bin H_error refresh selector path.
    # Existing AEC3-aligned code at filters.py:625 + use_per_bin_h_error_refresh
    # flag (config.py:169, default False). C2 re-evaluates the selector ON TOP
    # OF Sprint A only baseline; standalone v3.21.4 U4.A retest CLOSED FAIL.
    if ('AEC_PER_BIN_H_ERROR_REFRESH' in os.environ
            and 'use_per_bin_h_error_refresh' not in config_overrides):
        config_overrides['use_per_bin_h_error_refresh'] = (
            os.environ['AEC_PER_BIN_H_ERROR_REFRESH'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.21.6 Sprint P1 — AEC3 FilterAnalyzer port. Drives non-zero
    # min_direct_path_filter_delay in AecState; unblocks reverb tail
    # update (Sprint C diagnose cause) for the cohort.
    if ('AEC_FILTER_ANALYZER' in os.environ
            and 'filter_analyzer_enabled' not in config_overrides):
        config_overrides['filter_analyzer_enabled'] = (
            os.environ['AEC_FILTER_ANALYZER'].lower() not in ('0', 'false', 'off', 'no'))
    # F-E1 — ERL clip + far_active hysteresis (v3.11 Phase 1 Sprint 5-6)
    if ('AEC_F_E1' in os.environ
            and 'f_e1_enabled' not in config_overrides):
        config_overrides['f_e1_enabled'] = (
            os.environ['AEC_F_E1'].lower() not in ('0', 'false', 'off', 'no'))
    # F-DelayTrack — continuous delay variance tracking (v3.11 Phase 1 Sprint 7-8)
    if ('AEC_F_DELAYTRACK' in os.environ
            and 'f_delaytrack_enabled' not in config_overrides):
        config_overrides['f_delaytrack_enabled'] = (
            os.environ['AEC_F_DELAYTRACK'].lower() not in ('0', 'false', 'off', 'no'))
    # F-E3 — consecutive-EPC hangover + W partial reset (v3.11 Phase 1 Sprint 9-10)
    if ('AEC_F_E3' in os.environ
            and 'f_e3_enabled' not in config_overrides):
        config_overrides['f_e3_enabled'] = (
            os.environ['AEC_F_E3'].lower() not in ('0', 'false', 'off', 'no'))
    # F-E5 — saturation handling extensions (v3.11 Phase 1 Sprint 11-12)
    if ('AEC_F_E5' in os.environ
            and 'f_e5_enabled' not in config_overrides):
        config_overrides['f_e5_enabled'] = (
            os.environ['AEC_F_E5'].lower() not in ('0', 'false', 'off', 'no'))
    # E5.S2 — F-E5 main_mu saturation gate threshold (v3.13 E5 acoustic NL
    # protection). Default 0.5 was calibrated for digital clip; E5.S1 audit
    # shows acoustic NL keeps _saturation_level in [0, 0.09] band. Lower the
    # threshold so main_mu freeze + shadow_rise mask fire on acoustic-NL
    # frames. Same threshold gates both downstream actions (line 6017 / 6285).
    if ('AEC_F_E5_MAIN_MU_THRESHOLD' in os.environ
            and 'f_e5_main_mu_sat_threshold' not in config_overrides):
        config_overrides['f_e5_main_mu_sat_threshold'] = float(
            os.environ['AEC_F_E5_MAIN_MU_THRESHOLD'])
    # diverged_reset triple-AND gate (v3.11 Phase 1 Sprint 13-14)
    if ('AEC_DIVERGED_RESET_TRIPLE' in os.environ
            and 'diverged_reset_triple_and' not in config_overrides):
        config_overrides['diverged_reset_triple_and'] = (
            os.environ['AEC_DIVERGED_RESET_TRIPLE'].lower() not in ('0', 'false', 'off', 'no'))
    # S-orth.A: decoupled shadow Kalman state (default OFF)
    if ('AEC_SHADOW_DECOUPLED' in os.environ
            and 'shadow_state_decoupled' not in config_overrides):
        config_overrides['shadow_state_decoupled'] = (
            os.environ['AEC_SHADOW_DECOUPLED'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.15 §1.2: DT-NE compression fix (default OFF)
    if ('AEC_DT_NE_COMPRESSION_FIX' in os.environ
            and 'dt_ne_compression_fix' not in config_overrides):
        config_overrides['dt_ne_compression_fix'] = (
            os.environ['AEC_DT_NE_COMPRESSION_FIX'].lower() not in ('0', 'false', 'off', 'no'))
    # v3.15 §1.6 Arc F: per-band Kalman Q (default OFF)
    if ('AEC_KALMAN_Q_PER_BAND' in os.environ
            and 'kalman_q_per_band' not in config_overrides):
        config_overrides['kalman_q_per_band'] = (
            os.environ['AEC_KALMAN_Q_PER_BAND'].lower() not in ('0', 'false', 'off', 'no'))
    # Per-band Q scales: comma-separated LF,MF,HF (e.g. "0.5,1.0,2.0")
    if ('AEC_KALMAN_Q_BAND_SCALES' in os.environ
            and 'kalman_q_band_scales' not in config_overrides):
        config_overrides['kalman_q_band_scales'] = tuple(
            float(x) for x in os.environ['AEC_KALMAN_Q_BAND_SCALES'].split(','))
    # v3.15 §1.4 Arc M: EPC-gated per-band Q boost (default OFF)
    if ('AEC_ARC_M_EPC_GATED' in os.environ
            and 'arc_m_epc_gated' not in config_overrides):
        config_overrides['arc_m_epc_gated'] = (
            os.environ['AEC_ARC_M_EPC_GATED'].lower() not in ('0', 'false', 'off', 'no'))
    # Remaining env-var hooks for substrate flags still in AecConfig.
    for _flag, _key, _is_bool in (
        # v3.18 Phase D.1 — Subband NE detector substrate (R4).
        ('AEC_SUBBAND_NE_DETECT', 'subband_ne_detect_enabled', True),
        # v3.18 Phase D.3 — Mask shape swap (R2; needs D.1 ON to take effect).
        ('AEC_MASK_PROFILE_SWAP', 'res_mask_profile_swap_enabled', True),
        # D.5 sweep — gate `effective_dt` threshold for NE-profile activation.
        ('AEC_MASK_NE_GATE_DT', 'res_mask_ne_gate_dt', False),
        # D-Path-D — asymmetric per-bin overlay tuning.
        ('AEC_MASK_FS_OVERLAY_COH2', 'res_mask_fs_overlay_coh2_min', False),
        ('AEC_MASK_FS_OVERLAY_DT_MAX', 'res_mask_fs_overlay_dt_max', False),
        # v3.18 Phase B1 — Dominant NE detector
        ('AEC_DOMINANT_NE_DETECT', 'dominant_ne_detect_enabled', True),
        ('AEC_DOMINANT_NE_ENR_THRESHOLD', 'dominant_ne_enr_threshold', False),
        ('AEC_DOMINANT_NE_SNR_THRESHOLD', 'dominant_ne_snr_threshold', False),
        # v3.18 Phase F.1/F.3 — AEC3-aligned event classification + asymmetric reset
        ('AEC_EVENT_CLASSIFICATION', 'aec_event_classification_enabled', True),
        # v3.18 Phase A.2 — shadow class swap (PBFDKF → PBFDAF/NLMS, AEC3-aligned)
        ('AEC_SHADOW_NLMS', 'shadow_class_nlms', True),
        ('AEC_SHADOW_MU_NLMS', 'shadow_mu_nlms', False),
        # v3.18 Phase B.2/B.3 — FilterMisadjustmentEstimator + ScaleFilter
        ('AEC_FILTER_MISADJUSTMENT', 'filter_misadjustment_enabled', True),
        ('AEC_FILTER_MISADJUSTMENT_THRESHOLD', 'filter_misadjustment_threshold', False),
        ('AEC_FILTER_MISADJUSTMENT_SCALE_P', 'filter_misadjustment_scale_p', True),
        # v3.18 Phase C.A — FilterAnalyzer audit-only port
        ('AEC_FILTER_ANALYZER', 'filter_analyzer_enabled', True),
        # v3.21.6 Sprint P2 — AEC3 Legacy TransparentMode re-enable
        ('AEC_TRANSPARENT_MODE', 'transparent_mode_enabled', True),
        # v3.22 Sprint E.1 — stat-aware NE proxy at SuppressionGain consumers
        ('AEC_STAT_NE_PROXY', 'e_stat_aware_ne_proxy_enabled', True),
        ('AEC_STAT_NE_PROXY_THR', 'e_stat_aware_ne_proxy_threshold', False),
        # v3.22 Sprint F — reverb tail dead-fallback
        ('AEC_REVERB_DEAD_FALLBACK', 'reverb_tail_dead_fallback_enabled', True),
        ('AEC_REVERB_DEAD_THR', 'reverb_tail_dead_threshold_frames', False),
        ('AEC_REVERB_DEAD_STRENGTH', 'reverb_tail_dead_fallback_strength', False),
    ):
        if _flag in os.environ and _key not in config_overrides:
            _v = os.environ[_flag]
            if _is_bool:
                config_overrides[_key] = _v.lower() not in ('0', 'false', 'off', 'no')
            else:
                config_overrides[_key] = float(_v)

    # D.5 anchor sweep — comma-separated 3-tuple "enr_t,enr_s,emr_t"
    for _flag, _key in (
        ('AEC_RES_MASK_NORMAL_LF', 'res_mask_normal_lf'),
        ('AEC_RES_MASK_NORMAL_HF', 'res_mask_normal_hf'),
        ('AEC_RES_MASK_NEAREND_LF', 'res_mask_nearend_lf'),
        ('AEC_RES_MASK_NEAREND_HF', 'res_mask_nearend_hf'),
    ):
        if _flag in os.environ and _key not in config_overrides:
            config_overrides[_key] = tuple(
                float(x) for x in os.environ[_flag].split(','))

    # D-Path-D — mask swap mode ('binary' or 'asymmetric')
    if ('AEC_MASK_SWAP_MODE' in os.environ
            and 'res_mask_swap_mode' not in config_overrides):
        config_overrides['res_mask_swap_mode'] = os.environ['AEC_MASK_SWAP_MODE'].lower()

    # State-scale override: comma-separated state=scale pairs, e.g.
    #   "coarse_learning=2.0,refined_usable=1.0"
    if ('AEC_DT_NE_STATE_SCALE' in os.environ
            and 'dt_ne_state_scale' not in config_overrides):
        _ss = {}
        for pair in os.environ['AEC_DT_NE_STATE_SCALE'].split(','):
            k, _, v = pair.partition('=')
            _ss[k.strip()] = float(v)
        config_overrides['dt_ne_state_scale'] = _ss
    if ('AEC_DT_NE_PER_BIN_THRESH' in os.environ
            and 'dt_ne_per_bin_thresh' not in config_overrides):
        config_overrides['dt_ne_per_bin_thresh'] = float(
            os.environ['AEC_DT_NE_PER_BIN_THRESH'])
    if ('AEC_DT_NE_PER_BIN_SCALE' in os.environ
            and 'dt_ne_per_bin_scale' not in config_overrides):
        config_overrides['dt_ne_per_bin_scale'] = float(
            os.environ['AEC_DT_NE_PER_BIN_SCALE'])
    # P1 Phase 2: conditional HF cap + threshold env vars
    if 'AEC_HF_CAP_CONDITIONAL' in os.environ and 'hf_cap_conditional' not in config_overrides:
        config_overrides['hf_cap_conditional'] = (
            os.environ['AEC_HF_CAP_CONDITIONAL'].lower() not in ('0', 'false', 'off', 'no'))
    if 'AEC_HF_CAP_THETA' in os.environ and 'hf_cap_metric_threshold' not in config_overrides:
        config_overrides['hf_cap_metric_threshold'] = float(os.environ['AEC_HF_CAP_THETA'])
    # P3e: DT advisory gate env vars
    if 'AEC_DT_ADVISORY' in os.environ and 'dt_advisory_enabled' not in config_overrides:
        config_overrides['dt_advisory_enabled'] = (
            os.environ['AEC_DT_ADVISORY'].lower() not in ('0', 'false', 'off', 'no'))
    for _flag, _env in [
        ('dt_advisory_shadow_th', 'AEC_DT_ADV_SHADOW'),
        ('dt_advisory_energy_th', 'AEC_DT_ADV_ENERGY'),
        ('dt_advisory_hold_ms', 'AEC_DT_ADV_HOLD_MS'),
        ('dt_advisory_mu_factor', 'AEC_DT_ADV_MU_FACTOR'),
    ]:
        if _env in os.environ and _flag not in config_overrides:
            config_overrides[_flag] = float(os.environ[_env])
    # P3f Phase 3: gate on filter_state == 'suspicious_dt'
    if ('AEC_DT_ADV_P3F' in os.environ
            and 'dt_advisory_use_p3f_state' not in config_overrides):
        config_overrides['dt_advisory_use_p3f_state'] = (
            os.environ['AEC_DT_ADV_P3F'].lower() not in ('0', 'false', 'off', 'no'))
    # P3c Phase 1a: high-PAR fast-path
    if ('AEC_DELAY_FAST_PATH' in os.environ
            and 'delay_fast_path_enabled' not in config_overrides):
        config_overrides['delay_fast_path_enabled'] = (
            os.environ['AEC_DELAY_FAST_PATH'].lower() not in ('0', 'false', 'off', 'no'))
    if ('AEC_DELAY_FAST_PAR' in os.environ
            and 'delay_fast_par_threshold' not in config_overrides):
        config_overrides['delay_fast_par_threshold'] = float(
            os.environ['AEC_DELAY_FAST_PAR'])
    # v3.2 Stage V1: env override AEC_MODE=PBFDAF for filter comparison
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
    # Phase 3 / bench determinism: seed numpy RNG before each AEC instantiation
    # so CNG (np.random.randn at aec.py:2009-2010) produces identical noise on
    # every bench run. Without this, run-to-run AECMOS Δ is ±0.004 on
    # DT_movement (CNG noise floor) which masks genuinely small code-induced
    # changes. Determinism is per-case (each case gets a fresh seed=0 stream).
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
