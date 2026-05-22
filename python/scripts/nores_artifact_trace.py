"""Trace + ablation harness for the v3.21.6 nores LF artifact debug.

Runs the same (mic, ref) case under N ablations, capturing per-frame
per-band (LF/MF/HF) PBFDKF W-update-control state. Designed for
single-case W-update-control audit per the 2026-05-22 mandate, where:
  - nores tap = `aec.filter.process(...)` output (PBFDKF refined-error,
    pre-`_aec3_post`); verified at orchestrator.py:1792
  - artifact concentrates in LF band; suspected mu_lf overshoot when
    `denom_aec3 ≈ delta` because X²_lf / _error_psd_lf are both small

Output is JSON-only + terminal tables. No audio leakage required to
return results.

Ablation matrix (12 by default):
  A0 baseline             — no patch, reference
  A1 freeze_adapt         — main_mu=0 (sanity)
  A2 no_shadow            — enable_shadow=False
  A3 no_reset             — _reset_filter_derived_state → no-op
  A4 fixed_delay          — delay_est freezes after first acquisition
  A5 no_pbfdkf_update     — _update_weights{_aec3} → no-op (sanity)
  A6 no_sat               — _saturation_level clamped to 0

  -- W-update-control audit ablations (2026-05-22 mandate) --

  HPF_ON_OFF              — enable_highpass=True / enable_highpass_ref
                            =False. AEC3-like / docs-aligned state.
                            HPF policy is UNRESOLVED: code-on-main
                            (config.py:212) has ref HPF True; docs
                            (aec_methods.md / CLAUDE.md / architecture
                            HTML) all say ref HPF retired (OFF); the
                            verdict commit 4a41675 says CANNOT SHIP at
                            OFF, but the un-revert 6e273b6 (authored 1h
                            later) was never merged. Not labelling
                            either as "intended"; runs A/B/C decide
                            empirically.
  HPF_OFF_OFF             — DISABLE BOTH HPFs. Control only — not a
                            shipping candidate; isolates HPF chain as
                            a variable.
  (A0_baseline ≡ HPF_ON_ON, the current v3.21.6 code baseline.)
  B_freeze_lf_mu          — per-bin mu, zero LF bins (0-500 Hz); W_lf
                            stays at its current value but no new
                            adaptation
  C_zero_echo_lf          — zero echo_spec[lf_bins] before iFFT; W
                            adapts normally but output has no LF echo
                            subtraction (isolates update problem vs
                            output problem)
  D_raw_e2                — denom_aec3 uses raw |error_spec|² instead
                            of smoothed _error_psd
  E_avg_x2                — denom_aec3 uses partition-summed |X_buf|²
                            instead of X_latest only (probes AEC3
                            divergence #2: X²_latest vs delay-aligned)
  F_perbin_refresh        — cfg.use_per_bin_h_error_refresh=True;
                            switches H_error refresh from scalar
                            E2_ref_sum vs E2_coarse_sum compare to
                            per-bin E2_refined[k] vs E2_coarse[k]
                            compare (probes AEC3 divergence #1)

Per-band bin layout (PBFDKF fft_size=512, fs=16000 → 31.25 Hz/bin):
  LF   0-500 Hz   bins  0-16   (17 bins)
  MF 700-3000 Hz  bins 22-96   (75 bins)
  HF >3000 Hz     bins 96-256  (161 bins)

Usage:
  python3 python/scripts/nores_artifact_trace.py \\
      --mic /path/to/mic.wav --ref /path/to/ref.wav \\
      --stem CASE_NAME \\
      [--ablations A0,B_freeze_lf_mu,...] \\
      [--movement] [--pre-align] [--write-wav] \\
      [--clusters 175-179,665-677]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'python'))

from aec import AEC, AecConfig, __version__   # noqa: E402

BLOCK = 160          # 10 ms hop @ 16 kHz
FFT_PBFDKF = 512     # PBFDKF fft_size (block_size=320, next pow2=512)
N_FREQS = FFT_PBFDKF // 2 + 1   # 257
SR = 16000

# Band bin indices (LF/MF/HF). PBFDKF fft_size matches my STFT FFT_PSD so
# the same bin indices apply both to PBFDKF state and to the nores/mic
# STFT PSD comparison.
BIN_LF_HI = int(np.ceil(500.0 / (SR / FFT_PBFDKF)))     # 16
BIN_MF_LO = int(np.floor(700.0 / (SR / FFT_PBFDKF)))    # 22
BIN_MF_HI = int(np.ceil(3000.0 / (SR / FFT_PBFDKF)))    # 96
BIN_HF_LO = BIN_MF_HI                                    # 96
BIN_HF_HI = N_FREQS - 1                                  # 256

LF_SLICE = slice(0, BIN_LF_HI + 1)
MF_SLICE = slice(BIN_MF_LO, BIN_MF_HI + 1)
HF_SLICE = slice(BIN_HF_LO, BIN_HF_HI + 1)

TOP_N = 20
DEFAULT_CLUSTERS = '175-179,665-677'

# AEC3 noise gate threshold (read from module — keep in sync with code).
from modules.aec3_scale import NOISE_GATE_POWER_FLOAT   # noqa: E402

# ---------------------------------------------------------------------------
# Per-frame snapshot (kept compact)
# ---------------------------------------------------------------------------


@dataclass
class FrameSnap:
    """Per-AEC-block snapshot. All band-aggregate values are sum or mean
    over bins in band (LF/MF/HF) — see _band_aggregate() below."""
    frame: int = 0
    # Orchestrator-level diag
    mu_scale: float = 0.0
    main_paused: int = 0
    reverse_copy: int = 0
    boost_q: int = 0
    saturation_level: float = 0.0
    epc_active: int = 0
    converged: int = 0
    # PBFDKF W-update-control gates (scalar)
    call_counter: int = 0
    poor_exc_counter: int = 0
    saturated_capture: int = 0
    block_stationary: int = 0
    # Per-band fields (LF/MF/HF) — see Per-band schema
    # Naming: <metric>_<band>; metric in
    # {x2, mic, err, echo, h, denom, mu, mu_scale_arr, W_before, W_after,
    #  dW, ng_hit}
    x2_lf: float = 0.0
    x2_mf: float = 0.0
    x2_hf: float = 0.0
    mic_lf: float = 0.0
    mic_mf: float = 0.0
    mic_hf: float = 0.0
    err_lf: float = 0.0
    err_mf: float = 0.0
    err_hf: float = 0.0
    echo_lf: float = 0.0
    echo_mf: float = 0.0
    echo_hf: float = 0.0
    h_lf: float = 0.0
    h_mf: float = 0.0
    h_hf: float = 0.0
    denom_lf: float = 0.0
    denom_mf: float = 0.0
    denom_hf: float = 0.0
    mu_lf: float = 0.0
    mu_mf: float = 0.0
    mu_hf: float = 0.0
    mu_scale_arr_lf: float = 0.0
    mu_scale_arr_mf: float = 0.0
    mu_scale_arr_hf: float = 0.0
    W_before_lf: float = 0.0
    W_before_mf: float = 0.0
    W_before_hf: float = 0.0
    W_after_lf: float = 0.0
    W_after_mf: float = 0.0
    W_after_hf: float = 0.0
    dW_lf: float = 0.0
    dW_mf: float = 0.0
    dW_hf: float = 0.0
    ng_hit_lf: int = 0   # count of bins where X²<noise_gate (gate fires)
    ng_hit_mf: int = 0
    ng_hit_hf: int = 0


def _band_sum(arr: np.ndarray, sl: slice) -> float:
    return float(np.sum(arr[sl]))


def _band_mean(arr: np.ndarray, sl: slice) -> float:
    return float(np.mean(arr[sl]))


def _w_band_energy(W: np.ndarray, sl: slice) -> float:
    """Sum |W|² over all partitions, summed over bins in band."""
    return float(np.sum(np.abs(W[:, sl]) ** 2))


# ---------------------------------------------------------------------------
# Process-time hook factory — wraps filter.process to capture state
# ---------------------------------------------------------------------------


def install_w_probe(aec: AEC, snaps: list[FrameSnap]) -> Callable[[], None]:
    """Wrap aec.filter.process to record FrameSnap entries.

    Captures W BEFORE process (so dW is true delta), and reads PBFDKF
    state AFTER process for H_error / spectra / error_psd. mu_aec3 and
    denom_aec3 are recomputed at snapshot time using H_BEFORE (since
    they were computed PRE-decay inside _update_weights_aec3).
    """
    orig = aec.filter.process
    n_part = aec.filter.n_partitions
    delta32 = np.float32(aec.filter.delta)

    def wrapped(near_end, far_end, mu_scale=1.0):
        # Snapshot W BEFORE adapting this hop
        W_before = aec.filter.W.copy()
        H_before = aec.filter.H_error_per_bin.copy()
        out = orig(near_end, far_end, mu_scale)
        # POST-state: spectra (latest hop), _error_psd (post-EMA update),
        # W (after adaptation), H_error_per_bin (after decay + refresh)
        far_spec = aec.filter.far_spec
        near_spec = aec.filter.near_spec
        error_spec = aec.filter.error_spec
        echo_spec = aec.filter.echo_spec
        X2 = (np.abs(far_spec) ** 2).astype(np.float32)
        mic2 = (np.abs(near_spec) ** 2).astype(np.float32)
        err2 = (np.abs(error_spec) ** 2).astype(np.float32)
        echo2 = (np.abs(echo_spec) ** 2).astype(np.float32)
        err_psd_post = aec.filter._error_psd  # smoothed, post-update
        # Reproduce the mu/denom that drove THIS hop's update:
        # denom uses H_before (pre-decay) and err_psd_post (matches the
        # update site which writes _error_psd then immediately uses it).
        denom = (
            np.float32(0.5) * H_before * X2
            + np.float32(n_part) * err_psd_post
            + delta32
        )
        mu = H_before / np.maximum(denom, np.float32(1e-30))
        # noise gate
        ng_hit_mask = X2 < np.float32(NOISE_GATE_POWER_FLOAT)
        mu_post_gate = np.where(ng_hit_mask, np.float32(0.0), mu)
        # mu_scale_arr — orchestrator-side scalar/array fed to filter.process
        mu_scale_arr = np.asarray(mu_scale, dtype=np.float32)
        if mu_scale_arr.ndim == 0:
            mu_scale_arr = np.full(N_FREQS, float(mu_scale_arr),
                                   dtype=np.float32)

        d = aec._diag
        snap = FrameSnap(
            frame=len(snaps),
            mu_scale=float(d.get('mu_scale', 0.0)),
            main_paused=int(bool(d.get('main_paused', False))),
            reverse_copy=0,   # set below
            boost_q=0,        # set below
            saturation_level=float(d.get('saturation_level', 0.0)),
            epc_active=int(bool(d.get('epc_active', False))),
            converged=int(bool(d.get('converged', False))),
            call_counter=int(getattr(aec.filter, '_call_counter', 0)),
            poor_exc_counter=int(getattr(aec.filter,
                                         '_poor_excitation_counter', 0)),
            saturated_capture=int(bool(getattr(aec.filter,
                                               '_saturated_capture', False))),
            block_stationary=int(bool(getattr(aec.filter,
                                              '_block_stationary', False))),
            x2_lf=_band_sum(X2, LF_SLICE),
            x2_mf=_band_sum(X2, MF_SLICE),
            x2_hf=_band_sum(X2, HF_SLICE),
            mic_lf=_band_sum(mic2, LF_SLICE),
            mic_mf=_band_sum(mic2, MF_SLICE),
            mic_hf=_band_sum(mic2, HF_SLICE),
            err_lf=_band_sum(err2, LF_SLICE),
            err_mf=_band_sum(err2, MF_SLICE),
            err_hf=_band_sum(err2, HF_SLICE),
            echo_lf=_band_sum(echo2, LF_SLICE),
            echo_mf=_band_sum(echo2, MF_SLICE),
            echo_hf=_band_sum(echo2, HF_SLICE),
            h_lf=_band_mean(H_before, LF_SLICE),
            h_mf=_band_mean(H_before, MF_SLICE),
            h_hf=_band_mean(H_before, HF_SLICE),
            denom_lf=_band_mean(denom, LF_SLICE),
            denom_mf=_band_mean(denom, MF_SLICE),
            denom_hf=_band_mean(denom, HF_SLICE),
            mu_lf=_band_mean(mu_post_gate, LF_SLICE),
            mu_mf=_band_mean(mu_post_gate, MF_SLICE),
            mu_hf=_band_mean(mu_post_gate, HF_SLICE),
            mu_scale_arr_lf=_band_mean(mu_scale_arr, LF_SLICE),
            mu_scale_arr_mf=_band_mean(mu_scale_arr, MF_SLICE),
            mu_scale_arr_hf=_band_mean(mu_scale_arr, HF_SLICE),
            W_before_lf=_w_band_energy(W_before, LF_SLICE),
            W_before_mf=_w_band_energy(W_before, MF_SLICE),
            W_before_hf=_w_band_energy(W_before, HF_SLICE),
            W_after_lf=_w_band_energy(aec.filter.W, LF_SLICE),
            W_after_mf=_w_band_energy(aec.filter.W, MF_SLICE),
            W_after_hf=_w_band_energy(aec.filter.W, HF_SLICE),
            dW_lf=_w_band_energy(aec.filter.W - W_before, LF_SLICE),
            dW_mf=_w_band_energy(aec.filter.W - W_before, MF_SLICE),
            dW_hf=_w_band_energy(aec.filter.W - W_before, HF_SLICE),
            ng_hit_lf=int(np.sum(ng_hit_mask[LF_SLICE])),
            ng_hit_mf=int(np.sum(ng_hit_mask[MF_SLICE])),
            ng_hit_hf=int(np.sum(ng_hit_mask[HF_SLICE])),
        )
        # regime handler decisions
        rh = aec._regime_handler
        if getattr(rh, '_last_decision_reverse_copy', False):
            snap.reverse_copy = 1
        snaps.append(snap)
        return out

    aec.filter.process = wrapped
    return lambda: setattr(aec.filter, 'process', orig)


# ---------------------------------------------------------------------------
# Ablation hooks (in addition to W-probe wrap, which always runs)
# ---------------------------------------------------------------------------


def _patch_freeze_mu(aec):
    """A1: filter.process called with mu_scale=0 (override 3rd arg)."""
    orig = aec.filter.process

    def patched(near, far, mu_scale=1.0):
        return orig(near, far, 0.0)
    aec.filter.process = patched
    return lambda: setattr(aec.filter, 'process', orig)


def _patch_no_reset(aec, log):
    orig = aec._reset_filter_derived_state

    def patched(reason='plateau', preserve_render_ema=True):
        log.append(('skipped_reset', getattr(aec, '_frame_idx_trace', -1), reason))
    aec._reset_filter_derived_state = patched
    return lambda: setattr(aec, '_reset_filter_derived_state', orig)


def _patch_no_pbfdkf_update(aec):
    if not hasattr(aec.filter, '_update_weights_aec3'):
        return lambda: None
    orig_a = aec.filter._update_weights_aec3
    orig_n = aec.filter._update_weights
    aec.filter._update_weights_aec3 = lambda *a, **k: None
    aec.filter._update_weights = lambda *a, **k: None

    def restore():
        aec.filter._update_weights_aec3 = orig_a
        aec.filter._update_weights = orig_n
    return restore


def _patch_fixed_delay(aec, log):
    if not hasattr(aec, 'delay_est') or aec.delay_est is None:
        return lambda: None
    de = aec.delay_est
    orig = de.accumulate
    state = {'first_done': False, 'frozen': None}

    def patched(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not state['first_done'] and getattr(de, 'is_solid', False):
            state['first_done'] = True
            state['frozen'] = de.estimated_delay
        elif state['first_done']:
            try:
                de._estimated_delay = state['frozen']
            except Exception:
                pass
            log.append(('suppressed_delay', getattr(aec, '_frame_idx_trace', -1)))
        return out
    de.accumulate = patched
    return lambda: setattr(de, 'accumulate', orig)


def _patch_no_sat(aec):
    orig = aec.process

    def patched(*args, **kwargs):
        out = orig(*args, **kwargs)
        aec._saturation_level = 0.0
        return out
    aec.process = patched
    return lambda: setattr(aec, 'process', orig)


# ---- W-update-control audit ablations -----------------------------------


def _patch_freeze_lf_mu(aec):
    """B: per-bin mu_scale, zero LF bins. Wrap filter.process and
    convert scalar mu_scale to per-bin array with LF zeroed."""
    orig = aec.filter.process

    def patched(near, far, mu_scale=1.0):
        arr = np.asarray(mu_scale, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full(N_FREQS, float(arr), dtype=np.float32)
        else:
            arr = arr.copy()
        arr[LF_SLICE] = 0.0
        return orig(near, far, arr)
    aec.filter.process = patched
    return lambda: setattr(aec.filter, 'process', orig)


def _patch_zero_echo_lf(aec):
    """C: zero echo_spec[lf_bins] before iFFT.

    W is allowed to adapt as usual (so future frames see W_lf updates),
    but output is recomputed with echo_spec[lf]=0 → no LF echo subtraction
    → if artifact is in nores LF, it should reduce.
    """
    orig = aec.filter.process
    flt = aec.filter

    def patched(near, far, mu_scale=1.0):
        # Run process normally — W adapts, echo_spec is populated
        _ = orig(near, far, mu_scale)
        # Recompute output with LF zeroed in echo_spec
        es = flt.echo_spec.copy()
        es[LF_SLICE] = 0.0
        echo_time = np.fft.irfft(es, flt.fft_size).astype(np.float32)
        new_out = flt.near_buffer[-flt.hop_size:] - \
            echo_time[flt.hop_size:flt.block_size]
        # Update error_spec for downstream (so next frame's _error_psd is
        # consistent). The next-frame _update_weights_aec3 reads
        # self.error_spec; if we leave it as the original (W-adapted)
        # error, _error_psd will EMA that, which is fine — we're only
        # changing the OUTPUT sample stream, not the internal state.
        return new_out.astype(np.float32)
    aec.filter.process = patched
    return lambda: setattr(aec.filter, 'process', orig)


def _patch_raw_e2(aec):
    """D: denom_aec3 uses raw |error_spec|² instead of smoothed _error_psd.

    Wrap _update_weights_aec3 to temporarily replace self._error_psd with
    the passed-in `error_psd` (which is raw |error_spec|² computed in
    _update_weights at line 427). Restore after.
    """
    if not hasattr(aec.filter, '_update_weights_aec3'):
        return lambda: None
    orig = aec.filter._update_weights_aec3

    def patched(curr_p, mu_scale_arr, error_psd):
        saved = aec.filter._error_psd
        aec.filter._error_psd = error_psd   # use raw |E|² this call
        try:
            return orig(curr_p, mu_scale_arr, error_psd)
        finally:
            aec.filter._error_psd = saved
    aec.filter._update_weights_aec3 = patched
    return lambda: setattr(aec.filter, '_update_weights_aec3', orig)


def _patch_avg_x2(aec):
    """E: denom + noise_gate use partition-summed |X|² instead of X_latest.

    Replace _update_weights_aec3 with a custom version that uses
    sum(|X_buf|², axis=0) for denom + noise gate, but per-partition X
    for K (W update direction unchanged).
    """
    flt = aec.filter
    if not hasattr(flt, '_update_weights_aec3'):
        return lambda: None
    orig = flt._update_weights_aec3
    delta32 = np.float32(flt.delta)
    n_part = np.float32(flt.n_partitions)
    h_floor = flt._h_error_floor
    h_ceil = flt._h_error_ceil
    noise_gate = np.float32(NOISE_GATE_POWER_FLOAT)

    def patched(curr_p, mu_scale_arr, error_psd):
        X2_for_denom = np.sum(np.abs(flt.X_buf) ** 2, axis=0).astype(np.float32)
        denom = (np.float32(0.5) * flt.H_error_per_bin * X2_for_denom
                 + n_part * flt._error_psd + delta32)
        mu = (flt.H_error_per_bin / np.maximum(denom, np.float32(1e-30))
              ).astype(np.float32)
        mu = np.where(X2_for_denom >= noise_gate, mu, np.float32(0.0))
        for p in range(flt.n_partitions):
            p_idx = (curr_p - p) % flt.n_partitions
            X = flt.X_buf[p_idx]
            K = mu * np.conj(X)
            K_scaled = K * mu_scale_arr
            flt.W[p] += K_scaled * flt.error_spec
            if flt.enable_td_constraint:
                w_time = np.fft.irfft(flt.W[p], flt.fft_size).astype(np.float32)
                w_time *= flt._td_window
                flt.W[p] = np.fft.rfft(w_time).astype(np.complex64)
        flt.H_error_per_bin -= (
            np.float32(0.5) * mu * X2_for_denom * flt.H_error_per_bin
        )
        flt._h_error_refresh()
    flt._update_weights_aec3 = patched
    return lambda: setattr(flt, '_update_weights_aec3', orig)


# Ablation table: name → (cfg_mutate, hook_fn)
# cfg_mutate is applied BEFORE AEC construction; hook_fn after.
ABLATIONS = {
    'A0_baseline':         (None, None),
    'A1_freeze_adapt':     (None, _patch_freeze_mu),
    'A2_no_shadow':        (lambda cfg: setattr(cfg, 'enable_shadow', False),
                            None),
    'A3_no_reset':         (None, lambda a, l: _patch_no_reset(a, l)),
    'A4_fixed_delay':      (None, lambda a, l: _patch_fixed_delay(a, l)),
    'A5_no_pbfdkf_update': (None, _patch_no_pbfdkf_update),
    'A6_no_sat':           (None, _patch_no_sat),
    'HPF_ON_OFF':          (lambda cfg: setattr(
                                cfg, 'enable_highpass_ref', False), None),
    'HPF_OFF_OFF':         (lambda cfg: (
                                setattr(cfg, 'enable_highpass', False),
                                setattr(cfg, 'enable_highpass_ref', False),
                            ), None),
    'B_freeze_lf_mu':      (None, _patch_freeze_lf_mu),
    'C_zero_echo_lf':      (None, _patch_zero_echo_lf),
    'D_raw_e2':            (None, _patch_raw_e2),
    'E_avg_x2':            (None, _patch_avg_x2),
    'F_perbin_refresh':    (lambda cfg: setattr(
                                cfg, 'use_per_bin_h_error_refresh', True), None),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _stft_psd(x: np.ndarray) -> np.ndarray:
    """Hann-windowed STFT |·|², shape (n_frames, n_bins). Frame-aligned
    with AEC hops (hop=BLOCK)."""
    win = np.hanning(FFT_PBFDKF).astype(np.float32)
    n = len(x)
    n_frames = max(0, (n - FFT_PBFDKF) // BLOCK + 1)
    out = np.empty((n_frames, N_FREQS), dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * BLOCK: i * BLOCK + FFT_PBFDKF] * win
        out[i] = np.abs(np.fft.rfft(seg)) ** 2
    return out


def _build_cfg(movement: bool):
    np.random.seed(42)
    cfg = AecConfig.from_preset('balanced')
    cfg.enable_res = False
    cfg.enable_cng = False
    cfg.return_res_context = False
    if movement:
        cfg.enable_delay_est = True
        cfg.delay_est_period_s = 0.25
        cfg.delay_est_init_s = 0.2
    return cfg


def run_one(name: str, mic: np.ndarray, ref: np.ndarray,
            out_dir: str, write_wav: bool, movement: bool,
            clusters: list[tuple[int, int]]) -> dict:
    cfg_mut, hook_fn = ABLATIONS[name]
    cfg = _build_cfg(movement)
    if cfg_mut is not None:
        cfg_mut(cfg)
    aec = AEC(cfg)
    aec._frame_idx_trace = 0

    snaps: list[FrameSnap] = []
    extra_events: list = []
    restores = [install_w_probe(aec, snaps)]
    if hook_fn is not None:
        try:
            restores.append(hook_fn(aec, extra_events))
        except TypeError:
            restores.append(hook_fn(aec))

    n = (min(len(mic), len(ref)) // BLOCK) * BLOCK
    mic = mic[:n].astype(np.float32, copy=False)
    ref = ref[:n].astype(np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n // BLOCK):
        s = i * BLOCK
        aec._frame_idx_trace = i
        out[s:s + BLOCK] = aec.process(mic[s:s + BLOCK], ref[s:s + BLOCK])

    for r in restores:
        try:
            r()
        except Exception:
            pass

    # PSD diff
    mic_psd = _stft_psd(mic)
    nores_psd = _stft_psd(out)
    extra = np.maximum(nores_psd - mic_psd, 0.0)
    n_psd = extra.shape[0]
    extra_lf = extra[:, LF_SLICE].sum(axis=1)
    extra_mf = extra[:, MF_SLICE].sum(axis=1)
    extra_hf = extra[:, HF_SLICE].sum(axis=1)
    extra_sum = extra.sum(axis=1)

    # Cluster aggregates
    cluster_rows = []
    for lo, hi in clusters:
        m_slice = slice(max(0, lo), min(len(snaps), hi + 1))
        if m_slice.start >= m_slice.stop:
            continue
        ps_slice = slice(max(0, lo), min(n_psd, hi + 1))
        # per-band W/H/mu/denom: mean over cluster frames
        cluster_snaps = snaps[m_slice]
        if not cluster_snaps:
            continue
        def avg(field):
            return float(np.mean([getattr(s, field) for s in cluster_snaps]))
        def total(arr):
            return float(arr[ps_slice].sum()) if ps_slice.stop > ps_slice.start else 0.0

        cluster_rows.append({
            'frames': f'{lo}-{hi}',
            'n_frames_in_cluster': len(cluster_snaps),
            # PSD extras
            'extra_lf_sum': total(extra_lf),
            'extra_mf_sum': total(extra_mf),
            'extra_hf_sum': total(extra_hf),
            # Per-band W energy means
            'W_lf_after_mean': avg('W_after_lf'),
            'W_mf_after_mean': avg('W_after_mf'),
            'W_hf_after_mean': avg('W_after_hf'),
            'dW_lf_mean': avg('dW_lf'),
            'dW_mf_mean': avg('dW_mf'),
            'dW_hf_mean': avg('dW_hf'),
            # PBFDKF drive means
            'x2_lf_mean': avg('x2_lf'),
            'x2_mf_mean': avg('x2_mf'),
            'x2_hf_mean': avg('x2_hf'),
            'mic_lf_mean': avg('mic_lf'),
            'mic_mf_mean': avg('mic_mf'),
            'err_lf_mean': avg('err_lf'),
            'err_mf_mean': avg('err_mf'),
            'echo_lf_mean': avg('echo_lf'),
            'echo_mf_mean': avg('echo_mf'),
            'h_lf_mean': avg('h_lf'),
            'h_mf_mean': avg('h_mf'),
            'denom_lf_mean': avg('denom_lf'),
            'denom_mf_mean': avg('denom_mf'),
            'mu_lf_mean': avg('mu_lf'),
            'mu_mf_mean': avg('mu_mf'),
            'mu_scale_arr_lf_mean': avg('mu_scale_arr_lf'),
            'ng_hit_lf_mean': avg('ng_hit_lf'),
            'ng_hit_mf_mean': avg('ng_hit_mf'),
            # Gates
            'main_paused_count': int(sum(s.main_paused for s in cluster_snaps)),
            'reverse_copy_count': int(sum(s.reverse_copy for s in cluster_snaps)),
            'saturated_capture_count':
                int(sum(s.saturated_capture for s in cluster_snaps)),
            'block_stationary_count':
                int(sum(s.block_stationary for s in cluster_snaps)),
        })

    summary = {
        'ablation': name,
        'version': __version__,
        'n_samples': int(n),
        'n_blocks': int(n // BLOCK),
        'n_psd_frames': int(n_psd),
        'sample_rate': SR,
        'fft_size': FFT_PBFDKF,
        'noise_gate_power': float(NOISE_GATE_POWER_FLOAT),
        'extra_sum_total': float(extra_sum.sum()),
        'extra_sum_max': float(extra_sum.max() if n_psd else 0.0),
        'extra_sum_p99': float(np.percentile(extra_sum, 99)) if n_psd else 0.0,
        'extra_lf_total': float(extra_lf.sum()),
        'extra_mf_total': float(extra_mf.sum()),
        'extra_hf_total': float(extra_hf.sum()),
        'cluster_rows': cluster_rows,
        'extra_events_count': len(extra_events),
        'skipped_resets': [
            {'frame': e[1], 'reason': e[2]} for e in extra_events
            if e[0] == 'skipped_reset' and len(e) >= 3
        ][:50],
    }
    with open(os.path.join(out_dir, f'{name}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    if write_wav:
        sf.write(os.path.join(out_dir, f'{name}_nores.wav'),
                 out.astype(np.float32), SR)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mic', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--out', default='out_nores_artifact_debug')
    ap.add_argument('--ablations',
                    default='A0_baseline,A1_freeze_adapt,A2_no_shadow,'
                            'A3_no_reset,A4_fixed_delay,A5_no_pbfdkf_update,'
                            'A6_no_sat,HPF_ON_OFF,HPF_OFF_OFF,'
                            'B_freeze_lf_mu,C_zero_echo_lf,D_raw_e2,'
                            'E_avg_x2,F_perbin_refresh')
    ap.add_argument('--write-wav', action='store_true')
    ap.add_argument('--movement', action='store_true')
    ap.add_argument('--pre-align', action='store_true')
    ap.add_argument('--clusters', default=DEFAULT_CLUSTERS,
                    help='Cluster frame ranges (e.g., "175-179,665-677")')
    args = ap.parse_args()

    clusters: list[tuple[int, int]] = []
    for c in args.clusters.split(','):
        c = c.strip()
        if not c:
            continue
        lo, hi = c.split('-')
        clusters.append((int(lo), int(hi)))

    mic, sr_mic = sf.read(args.mic)
    ref, sr_ref = sf.read(args.ref)
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    assert sr_mic == SR == sr_ref, f'expected {SR} Hz'

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
    all_summ = {}
    for name in args.ablations.split(','):
        name = name.strip()
        if name not in ABLATIONS:
            print(f'  WARNING: unknown ablation {name!r}, skipped'); continue
        print(f'  rendering {name} ...')
        all_summ[name] = run_one(name, mic, ref, out_dir, args.write_wav,
                                 args.movement, clusters)

    base = all_summ.get('A0_baseline')
    # ---- transcription report --------------------------------------------
    print('\n' + '=' * 72)
    print(f'NORES W-UPDATE-CONTROL AUDIT — stem={args.stem}  v{__version__}')
    print('=' * 72)
    print('\n[TABLE 1] extra_psd ratio vs A0_baseline (1.000 = no effect)')
    print(f"{'ablation':<22s} {'total':>8s} {'max':>8s} {'p99':>8s} "
          f"{'lf':>8s} {'mf':>8s} {'hf':>8s}")
    keys = ('extra_sum_total', 'extra_sum_max', 'extra_sum_p99',
            'extra_lf_total', 'extra_mf_total', 'extra_hf_total')
    for name, s in all_summ.items():
        if name == 'A0_baseline':
            print(f"{name:<22s} " + ' '.join(f"{1.000:>8.3f}" for _ in keys)
                  + '   [ref]')
            continue
        row = [s.get(k, 0.0) / (base.get(k, 0.0) or 1e-12) for k in keys]
        print(f"{name:<22s} " + ' '.join(f"{v:>8.3f}" for v in row))
    print('  (A1/A5: sanity — filter inactive → mic passthrough → ~0.003)')
    print('  (HPF policy UNRESOLVED — A0 ≡ HPF_ON_ON is code-on-main state;')
    print('   HPF_ON_OFF is AEC3-like + docs-aligned; HPF_OFF_OFF is control.')
    print('   Compare A0 vs HPF_ON_OFF for ref-HPF flip evidence. Verdict on')
    print('   shipping default requires DT/echo eval beyond this nores trace.)')

    print('\n[TABLE 2] gate counts per ablation '
          '(call=cold-start; pe=poor-exc; sat=saturated_capture; stat=block_stationary)')
    print(f"{'ablation':<22s} {'pause':>7s} {'revcp':>7s} "
          f"{'call_min':>9s} {'pe_min':>7s} {'sat_sum':>8s} {'stat_sum':>9s}")
    # We need to aggregate from snaps for these. Since we discarded snaps
    # after run_one, fall back to event count proxies + per-cluster gates.
    for name, s in all_summ.items():
        # Take aggregate from cluster_rows (sum of gate counts across clusters);
        # for full-run gate stats we'd need to keep snaps — skipped to keep
        # JSON compact. Cluster-level is the relevant audit anyway.
        clrows = s.get('cluster_rows', [])
        pause = sum(r.get('main_paused_count', 0) for r in clrows)
        rev = sum(r.get('reverse_copy_count', 0) for r in clrows)
        sat = sum(r.get('saturated_capture_count', 0) for r in clrows)
        stat = sum(r.get('block_stationary_count', 0) for r in clrows)
        print(f"{name:<22s} {pause:>7d} {rev:>7d} "
              f"{'-':>9s} {'-':>7s} {sat:>8d} {stat:>9d}")

    if clusters:
        for ci, (lo, hi) in enumerate(clusters):
            print(f'\n[TABLE 3.{ci+1}] cluster f={lo}-{hi} per-band aggregates '
                  '(mean across cluster frames; energy = Σ|·|² per band)')
            print(f"{'ablation':<22s} {'eLF':>10s} {'eMF':>10s} {'eHF':>10s} "
                  f"{'x2LF':>10s} {'micLF':>10s} {'errLF':>10s} {'echoLF':>10s} "
                  f"{'hLF':>10s} {'denomLF':>10s} {'muLF':>10s} "
                  f"{'WafterLF':>10s} {'dWLF':>10s} {'ngLF':>6s}")
            for name, s in all_summ.items():
                clrows = s.get('cluster_rows', [])
                if ci >= len(clrows):
                    continue
                r = clrows[ci]
                print(f"{name:<22s} "
                      f"{r['extra_lf_sum']:>10.2f} "
                      f"{r['extra_mf_sum']:>10.2f} "
                      f"{r['extra_hf_sum']:>10.2f} "
                      f"{r['x2_lf_mean']:>10.4f} "
                      f"{r['mic_lf_mean']:>10.4f} "
                      f"{r['err_lf_mean']:>10.4f} "
                      f"{r['echo_lf_mean']:>10.4f} "
                      f"{r['h_lf_mean']:>10.4f} "
                      f"{r['denom_lf_mean']:>10.4f} "
                      f"{r['mu_lf_mean']:>10.4f} "
                      f"{r['W_lf_after_mean']:>10.4f} "
                      f"{r['dW_lf_mean']:>10.4f} "
                      f"{r['ng_hit_lf_mean']:>6.1f}")

    sk = all_summ.get('A3_no_reset', {}).get('skipped_resets', [])
    if sk:
        print('\n[TABLE 4] A3 skipped-reset timeline')
        for evt in sk[:15]:
            print(f"  f={evt['frame']:>5d}  reason={evt['reason']}")

    summ_path = os.path.join(out_dir, 'summary.json')
    with open(summ_path, 'w') as f:
        json.dump({
            'stem': args.stem,
            'version': __version__,
            'clusters': args.clusters,
            'all_summaries': all_summ,
        }, f, indent=2)
    print(f'\n  summary.json → {summ_path}')

    print('\n' + '=' * 72)
    print('READ-BACK INSTRUCTIONS')
    print('  Minimum: TABLE 1 (12 rows × 7 cols).')
    print('  Strongly preferred: TABLE 3.1 + TABLE 3.2 (12 rows × 13 cols each)')
    print('  — these are the per-cluster per-band aggregates that pin')
    print('  the W-update-control mechanism.')
    print('  Optional: TABLE 2 + TABLE 4 if any anomaly is visible.')
    print('=' * 72)


if __name__ == '__main__':
    main()
