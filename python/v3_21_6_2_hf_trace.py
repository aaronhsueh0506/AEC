#!/usr/bin/env python3
"""Per-frame HF-painted-black diagnostic tracer for v3.21.6.2.

Usage:
    python3 python/v3_21_6_2_hf_trace.py MIC.wav LPB.wav OUTPUT_DIR \\
        [--mode pbfdkf] [--preset balanced] [--enable-res] [--cng]

Renders the case once with the current production AEC, captures per-frame
diagnostic taps (per-bin suppression gain, dominant_nearend mode,
usable_linear flag, SubtractorOutputAnalyzer signals, poor_coarse state,
linear-residual / echo / near spectra) and writes:

    OUTPUT_DIR/
        trace.npz       — all per-frame fields, plus the rendered output
        trace.png       — 6-panel summary (mic + out spectrograms,
                          per-bin gain heatmap, per-band gain medians,
                          dominant_nearend / usable_linear flags,
                          per-band residual / near PSD medians)

Designed for the HF-painted-black symptom (gain wiped > 1 kHz during
NE speech). The 6-panel PNG is usually enough; the .npz lets us follow
up offline if a specific frame range needs deeper investigation.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

# Add `python/` to path so the script works whether invoked from the repo
# root or from inside python/ (mirrors run_one_case.py convention).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aec as aec_module  # noqa: E402
from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402


def _hz_to_bin(hz: float, n_bins: int, sr: int) -> int:
    return int(round(hz * (n_bins - 1) * 2 / sr))


def _band_median(spec_power: np.ndarray, lo_bin: int, hi_bin: int) -> float:
    seg = spec_power[lo_bin:hi_bin]
    return float(np.median(seg)) if seg.size > 0 else 0.0


def _snapshot(aec: AEC, n_bins: int, sr: int) -> dict:
    """Pull every diagnostic surface we care about for the current hop."""
    lf_end = _hz_to_bin(500.0, n_bins, sr)
    mf_end = _hz_to_bin(2000.0, n_bins, sr)

    sg = getattr(aec, "_aec3_sg", None)
    state = getattr(aec, "_aec3_state", None)
    filt = aec.filter

    gain = (sg._last_gain.copy() if sg is not None
            else np.ones(n_bins, dtype=np.float32))
    error_psd = (np.abs(filt.error_spec) ** 2).astype(np.float32) \
        if hasattr(filt, "error_spec") else np.zeros(n_bins, dtype=np.float32)
    near_psd = (np.abs(filt.near_spec) ** 2).astype(np.float32) \
        if hasattr(filt, "near_spec") else np.zeros(n_bins, dtype=np.float32)
    echo_psd = (np.abs(filt.echo_spec) ** 2).astype(np.float32) \
        if hasattr(filt, "echo_spec") else np.zeros(n_bins, dtype=np.float32)
    far_psd = (np.abs(filt.far_spec) ** 2).astype(np.float32) \
        if hasattr(filt, "far_spec") else np.zeros(n_bins, dtype=np.float32)

    diag = dict(getattr(aec, "_diag", {}))

    # Pull SuppressionGain's internal `_last_lower_band_snap` if present.
    # These fields distinguish HF cap propagation from underlying R²
    # inflation, and identify which gain rule fired in HF (min_gain /
    # limiting_gain / pass-through). They are the smoking gun for HF
    # painted-black root cause.
    sg_snap = getattr(sg, "_last_lower_band_snap", {}) or {} if sg is not None else {}

    # Pull ResidualEchoEstimator's direct vs reverb R² split. Tells us
    # whether RES inflation comes from the direct path (S²/ERLE) or the
    # AddReverb tail mass accumulator.
    ree = getattr(aec, "_aec3_ree", None)
    r2_direct = (ree._last_r2_direct_component
                 if ree is not None
                 and hasattr(ree, "_last_r2_direct_component")
                 else np.zeros(n_bins, dtype=np.float32))
    r2_reverb = (ree._last_r2_reverb_component
                 if ree is not None
                 and hasattr(ree, "_last_r2_reverb_component")
                 else np.zeros(n_bins, dtype=np.float32))
    r2_path = (ree._last_r2_path if ree is not None
               and hasattr(ree, "_last_r2_path") else "unset")
    # Reverb model state (tail energy proxy)
    reverb_fr = getattr(ree, "_reverb_freq_resp", None) if ree is not None else None
    tail_max = (float(np.max(reverb_fr.tail_response))
                if reverb_fr is not None
                and hasattr(reverb_fr, "tail_response") else 0.0)

    # AEC3-strict continuous filter quality (FullBandErleEstimator). None
    # during warmup or after the hold counter (~400 ms) expires.
    flq = (state.get_inst_linear_quality_estimate()
           if state is not None
           and hasattr(state, "get_inst_linear_quality_estimate") else None)
    snap = {
        "gain": gain,
        "gain_lf_med": float(np.median(gain[:lf_end])),
        "gain_mf_med": float(np.median(gain[lf_end:mf_end])),
        "gain_hf_med": float(np.median(gain[mf_end:])),
        "filter_quality": float(flq) if flq is not None else -1.0,  # -1 sentinel = None
        "dominant_nearend": bool(sg.is_dominant_nearend()) if sg is not None else False,
        "usable_linear": bool(state.usable_linear_estimate()) if state is not None else False,
        "transparent_mode": bool(state.transparent_mode_active()) if state is not None else False,
        "filter_converged": bool(diag.get("converged", False)),
        "poor_coarse_counter": int(getattr(aec, "_poor_coarse_counter", 0)),
        "coarse_reset_hangover": int(getattr(aec, "_coarse_reset_hangover", 0)),
        "res_gain_mean": float(diag.get("res_gain_mean", 1.0)),
        "res_gain_min": float(diag.get("res_gain_min", 1.0)),
        "saturation_level": float(getattr(aec, "_saturation_level", 0.0)),
        "error_psd_lf": _band_median(error_psd, 0, lf_end),
        "error_psd_mf": _band_median(error_psd, lf_end, mf_end),
        "error_psd_hf": _band_median(error_psd, mf_end, n_bins),
        "near_psd_lf": _band_median(near_psd, 0, lf_end),
        "near_psd_mf": _band_median(near_psd, lf_end, mf_end),
        "near_psd_hf": _band_median(near_psd, mf_end, n_bins),
        "echo_psd_lf": _band_median(echo_psd, 0, lf_end),
        "echo_psd_mf": _band_median(echo_psd, lf_end, mf_end),
        "echo_psd_hf": _band_median(echo_psd, mf_end, n_bins),
        "far_psd_mean": float(np.mean(far_psd)),
    }
    # SG internals — HF wipe root cause attribution
    snap.update({
        # gain just before HF cap propagation (distinguishes anchor vs R²)
        "sg_gain_hf_med_pre_hf_lim": float(sg_snap.get("gain_hf_median_pre_hf_lim", 1.0)),
        "sg_gain_hf_med_post":       float(sg_snap.get("gain_hf_median", 1.0)),
        # Per-band fractions of gain bins that landed on min/lim gate
        "sg_reason_min_hf": float(sg_snap.get("reason_min_hf", 0.0)),
        "sg_reason_lim_hf": float(sg_snap.get("reason_lim_hf", 0.0)),
        "sg_reason_min_mf": float(sg_snap.get("reason_min_mf", 0.0)),
        "sg_reason_lim_mf": float(sg_snap.get("reason_lim_mf", 0.0)),
        # ENR/EMR HF medians (post-audibility weight, pre-clip)
        "sg_enr_hf": float(sg_snap.get("enr_hf_median", 0.0)),
        "sg_emr_hf": float(sg_snap.get("emr_hf_median", 0.0)),
        "sg_enr_tr_hf": float(sg_snap.get("enr_tr_hf_median", 1.0)),
        "sg_emr_tr_hf": float(sg_snap.get("emr_tr_hf_median", 1.0)),
        # Raw R² in HF (BEFORE audibility weighting) — is R² inflated?
        "sg_r2_hf_mean": float(sg_snap.get("r2_hf_mean", 0.0)),
        "sg_r2_mf_mean": float(sg_snap.get("r2_mf_mean", 0.0)),
        "sg_r2_lf_mean": float(sg_snap.get("r2_lf_mean", 0.0)),
        # HF cap anchor bin + its pre-cap gain value
        "sg_hf_anchor_value": float(sg_snap.get("hf_anchor_value_pre_hf_lim", 1.0)),
        "sg_hf_lim_applied": float(sg_snap.get("hf_lim_applied", 0.0)),
        # ResidualEchoEstimator R² breakdown
        "r2_direct_hf": _band_median(r2_direct, mf_end, n_bins),
        "r2_reverb_hf": _band_median(r2_reverb, mf_end, n_bins),
        "r2_direct_mf": _band_median(r2_direct, lf_end, mf_end),
        "r2_reverb_mf": _band_median(r2_reverb, lf_end, mf_end),
        "reverb_tail_max": tail_max,
        "r2_path_linear": 1.0 if r2_path == "linear" else 0.0,
        "r2_path_nonlinear": 1.0 if r2_path == "nonlinear" else 0.0,
    })
    return snap


def trace_case(mic_path: str, lpb_path: str, output_dir: str, *,
               mode: str = "pbfdkf", preset: str = "balanced",
               enable_res: bool = True, cng: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)

    mic, sr_mic = sf.read(mic_path, dtype="float32")
    lpb, sr_lpb = sf.read(lpb_path, dtype="float32")
    assert sr_mic == sr_lpb, f"sample-rate mismatch: mic={sr_mic} lpb={sr_lpb}"
    sr = int(sr_mic)
    if mic.ndim == 2:
        mic = mic.mean(axis=1)
    if lpb.ndim == 2:
        lpb = lpb.mean(axis=1)
    n = int(min(len(mic), len(lpb)))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    np.random.seed(0)  # match eval_aec_challenge.py per-case determinism
    cfg = AecConfig.from_preset(AecPreset[preset.upper()])
    cfg.mode = AecMode[mode.upper()]
    cfg.enable_res = bool(enable_res)
    cfg.enable_cng = bool(cng)
    cfg.sample_rate = sr

    aec = AEC(cfg)
    hop = cfg.hop_size
    n_bins = aec.filter.n_freqs

    n_hops = n // hop
    out = np.zeros(n_hops * hop, dtype=np.float32)
    rows = []

    for h in range(n_hops):
        s, e = h * hop, (h + 1) * hop
        out_hop = aec.process(mic[s:e], lpb[s:e])
        out[s:e] = out_hop
        rows.append(_snapshot(aec, n_bins, sr))

    # Stack snapshots
    scalar_keys = [k for k in rows[0].keys() if k != "gain"]
    cols = {k: np.array([r[k] for r in rows]) for k in scalar_keys}
    cols["gain"] = np.stack([r["gain"] for r in rows], axis=0)  # (n_hops, n_bins)

    # Save .npz with audio + trace
    npz_path = os.path.join(output_dir, "trace.npz")
    np.savez_compressed(
        npz_path,
        mic=mic[:n_hops * hop],
        lpb=lpb[:n_hops * hop],
        out=out,
        sr=sr,
        hop=hop,
        n_bins=n_bins,
        version=getattr(aec_module, "__version__", "unknown"),
        **cols,
    )

    # Plot 6-panel diagnostic PNG
    png_path = os.path.join(output_dir, "trace.png")
    _plot(npz_path, png_path, sr=sr, hop=hop, n_bins=n_bins)

    # Re-render output WAV for listening
    out_wav = os.path.join(output_dir, "out.wav")
    sf.write(out_wav, out, sr)

    # Rich stdout report — designed to be copy/pasteable when the trace
    # files cannot be transferred out.
    _print_console_report(cols, mic, out, sr=sr, hop=hop, n_hops=n_hops,
                          mic_path=mic_path, lpb_path=lpb_path)
    print(f"\nLocal files (optional, for your own inspection):")
    print(f"  npz: {npz_path}")
    print(f"  png: {png_path}")
    print(f"  wav: {out_wav}")


def _print_console_report(cols: dict, mic: np.ndarray, out: np.ndarray,
                          *, sr: int, hop: int, n_hops: int,
                          mic_path: str, lpb_path: str) -> None:
    """Stdout-only summary — designed for environments where binary trace
    files cannot leave the host. Everything below is copy/pasteable as
    plain text and is sufficient to diagnose HF-painted-black symptoms
    without the PNG / NPZ outputs."""

    def _pct(x: np.ndarray) -> str:
        return f"{x.astype(bool).mean() * 100:5.1f}%"

    def _stats(x: np.ndarray) -> str:
        return (f"mean={x.mean():.3f}  min={x.min():.3f}  "
                f"p5={np.percentile(x, 5):.3f}  "
                f"p25={np.percentile(x, 25):.3f}  "
                f"p50={np.percentile(x, 50):.3f}  "
                f"p75={np.percentile(x, 75):.3f}")

    def _db(x: float) -> float:
        return 10 * np.log10(max(x, 1e-12))

    print("=" * 72)
    print(f"  v3.21.6.2 HF-painted-black trace — stdout report")
    print("=" * 72)
    print(f"mic       : {mic_path}")
    print(f"lpb       : {lpb_path}")
    print(f"version   : {getattr(aec_module, '__version__', 'unknown')}")
    print(f"sr={sr} Hz  hop={hop}  n_hops={n_hops}  "
          f"duration={n_hops * hop / sr:.2f}s")
    print()

    # ---------- Overall per-band gain stats ----------
    print("--- Overall gain medians ---")
    for band in ("lf", "mf", "hf"):
        print(f"  gain_{band}_med:  " + _stats(cols[f"gain_{band}_med"]))
    print()

    # ---------- Flag fractions ----------
    print("--- Flag fractions (whole file) ---")
    print(f"  filter_converged       : {_pct(cols['filter_converged'])}")
    print(f"  usable_linear          : {_pct(cols['usable_linear'])}")
    print(f"  dominant_nearend       : {_pct(cols['dominant_nearend'])}")
    print(f"  transparent_mode       : {_pct(cols['transparent_mode'])}")
    print(f"  coarse_reset_hangover>0: {_pct(cols['coarse_reset_hangover'] > 0)}")
    fires = int(np.sum(np.diff(
        (cols["coarse_reset_hangover"] > 0).astype(int)) > 0))
    print(f"  poor_coarse rescue fires (rising edges): {fires}")
    # AEC3-strict continuous filter_quality (FullBandErleEstimator hold ~400 ms)
    fq = cols.get("filter_quality")
    if fq is not None:
        fq_none_frac = (fq < 0).mean()
        fq_live = fq[fq >= 0]
        if fq_live.size > 0:
            print(f"  filter_quality (AEC3 continuous): "
                  f"alive {(1 - fq_none_frac) * 100:.1f}% of frames  "
                  f"alive-mean={fq_live.mean():.3f}  alive-max={fq_live.max():.3f}")
        else:
            print(f"  filter_quality (AEC3 continuous): None throughout file")
    print()

    # ---------- NE-active segments ----------
    win = 1600  # 100 ms @ 16 kHz
    sub = 400
    n_samples = n_hops * hop
    mic_trim = mic[:n_samples]
    n_chunks = (n_samples - win) // sub + 1
    chunk_db = np.empty(n_chunks)
    for i in range(n_chunks):
        chunk_db[i] = _db(np.mean(mic_trim[i * sub:i * sub + win] ** 2))
    ne_chunks_mask = chunk_db > -25.0  # mic-power active threshold
    # Map back to hops: a hop is NE-active if any covering chunk fires.
    ne_hop_mask = np.zeros(n_hops, dtype=bool)
    for i in np.where(ne_chunks_mask)[0]:
        sample_start = i * sub
        sample_end = sample_start + win
        hop_start = sample_start // hop
        hop_end = min(n_hops, (sample_end + hop - 1) // hop)
        ne_hop_mask[hop_start:hop_end] = True
    n_ne = int(ne_hop_mask.sum())
    print(f"--- NE-active segments (mic > -25 dBFS / 100 ms) ---")
    print(f"  {n_ne} hops ({n_ne / max(1, n_hops) * 100:.1f}% of file, "
          f"{n_ne * hop / sr:.2f}s)")
    if n_ne > 0:
        for band in ("lf", "mf", "hf"):
            sub_arr = cols[f"gain_{band}_med"][ne_hop_mask]
            print(f"  gain_{band}_med (NE only):  " + _stats(sub_arr))
        print(f"  filter_converged (NE only): "
              f"{_pct(cols['filter_converged'][ne_hop_mask])}")
        print(f"  usable_linear    (NE only): "
              f"{_pct(cols['usable_linear'][ne_hop_mask])}")
        print(f"  dominant_nearend (NE only): "
              f"{_pct(cols['dominant_nearend'][ne_hop_mask])}")
        print(f"  poor_coarse hangover (NE only): "
              f"{_pct(cols['coarse_reset_hangover'][ne_hop_mask] > 0)}")
    print()

    # ---------- HF-wipe events ----------
    wipe_mask = cols["gain_hf_med"] < 0.3
    n_wipe = int(wipe_mask.sum())
    print(f"--- HF wipe events (gain_hf_med < 0.3) ---")
    print(f"  {n_wipe} hops ({n_wipe / max(1, n_hops) * 100:.1f}% of file, "
          f"{n_wipe * hop / sr:.2f}s)")
    if n_wipe > 0:
        print(f"  Of those wipe-active hops:")
        print(f"    NE-active (mic loud): "
              f"{_pct(ne_hop_mask[wipe_mask])}")
        print(f"    dominant_nearend    : "
              f"{_pct(cols['dominant_nearend'][wipe_mask])}")
        print(f"    usable_linear       : "
              f"{_pct(cols['usable_linear'][wipe_mask])}")
        print(f"    filter_converged    : "
              f"{_pct(cols['filter_converged'][wipe_mask])}")
        print(f"    transparent_mode    : "
              f"{_pct(cols['transparent_mode'][wipe_mask])}")
        print(f"    coarse_reset_hangover>0: "
              f"{_pct(cols['coarse_reset_hangover'][wipe_mask] > 0)}")
        # PSD medians at wipe (dB scale)
        for k in ("near_psd_hf", "echo_psd_hf", "error_psd_hf"):
            arr_db = 10 * np.log10(cols[k][wipe_mask] + 1e-12)
            print(f"    {k} (dB): "
                  f"mean={arr_db.mean():+6.2f}  min={arr_db.min():+6.2f}  "
                  f"max={arr_db.max():+6.2f}")
    print()

    # ---------- NE-active AND HF wiped (the actual symptom) ----------
    symptom_mask = ne_hop_mask & wipe_mask
    n_sym = int(symptom_mask.sum())
    print(f"--- SYMPTOM frames: NE-active AND HF wiped ---")
    print(f"  {n_sym} hops ({n_sym / max(1, n_hops) * 100:.1f}% of file, "
          f"{n_sym * hop / sr:.2f}s)")
    if n_sym > 0:
        print(f"  During the symptom:")
        print(f"    dominant_nearend     : "
              f"{_pct(cols['dominant_nearend'][symptom_mask])}  "
              f"← if low, DNE missed the NE → Candidate A or no-detector gap")
        print(f"    usable_linear        : "
              f"{_pct(cols['usable_linear'][symptom_mask])}  "
              f"← if high during DNE, linear residual fed to SG → Candidate B")
        print(f"    filter_converged     : "
              f"{_pct(cols['filter_converged'][symptom_mask])}")
        print(f"    poor_coarse hangover>0: "
              f"{_pct(cols['coarse_reset_hangover'][symptom_mask] > 0)}")
        for k in ("near_psd_hf", "echo_psd_hf", "error_psd_hf"):
            arr_db = 10 * np.log10(cols[k][symptom_mask] + 1e-12)
            print(f"    {k} (dB): "
                  f"mean={arr_db.mean():+6.2f}  median={np.median(arr_db):+6.2f}")
        # --- SG internal attribution at symptom (where does the wipe come from?) ---
        print(f"  SuppressionGain internals at symptom:")
        pre = cols["sg_gain_hf_med_pre_hf_lim"][symptom_mask]
        post = cols["sg_gain_hf_med_post"][symptom_mask]
        print(f"    gain_hf_med pre-HF-cap : "
              f"mean={pre.mean():.3f}  median={np.median(pre):.3f}  "
              f"min={pre.min():.3f}")
        print(f"    gain_hf_med post-HF-cap: "
              f"mean={post.mean():.3f}  median={np.median(post):.3f}  "
              f"min={post.min():.3f}")
        cap_drop = pre - post
        print(f"    HF-cap drop (pre-post) : "
              f"mean={cap_drop.mean():.3f}  max={cap_drop.max():.3f}  "
              f"← if large, HF cap anchor crushed HF; if ~0, R² inflated")
        print(f"    hf_anchor_value pre-cap: "
              f"mean={cols['sg_hf_anchor_value'][symptom_mask].mean():.3f}  "
              f"min={cols['sg_hf_anchor_value'][symptom_mask].min():.3f}  "
              f"← if low, single-bin null at ~2 kHz wipes whole HF")
        print(f"    hf_lim_applied fraction: "
              f"{cols['sg_hf_lim_applied'][symptom_mask].mean() * 100:5.1f}%")
        print(f"    reason_min HF / MF     : "
              f"{cols['sg_reason_min_hf'][symptom_mask].mean() * 100:5.1f}% / "
              f"{cols['sg_reason_min_mf'][symptom_mask].mean() * 100:5.1f}%  "
              f"← high = audibility floor (min_gain) fires; needs EchoAudibility port")
        print(f"    reason_lim HF / MF     : "
              f"{cols['sg_reason_lim_hf'][symptom_mask].mean() * 100:5.1f}% / "
              f"{cols['sg_reason_lim_mf'][symptom_mask].mean() * 100:5.1f}%  "
              f"← high = HF cap / smoothing limited")
        print(f"    ENR HF (R²/Y²)         : "
              f"mean={cols['sg_enr_hf'][symptom_mask].mean():.3f}  "
              f"median={np.median(cols['sg_enr_hf'][symptom_mask]):.3f}  "
              f"← if >>1, R² inflated above near; SG kills HF")
        print(f"    EMR HF (R²/CN)         : "
              f"mean={cols['sg_emr_hf'][symptom_mask].mean():.3f}  "
              f"median={np.median(cols['sg_emr_hf'][symptom_mask]):.3f}")
        print(f"    enr_target HF (tuning) : "
              f"{cols['sg_enr_tr_hf'][symptom_mask].mean():.3f}  "
              f"← nearend_tuning HF target")
        r2_hf_db = 10 * np.log10(cols["sg_r2_hf_mean"][symptom_mask] + 1e-12)
        print(f"    R² HF mean (RES out, dB): "
              f"mean={r2_hf_db.mean():+6.2f}  "
              f"median={np.median(r2_hf_db):+6.2f}  "
              f"← compare to echo_psd_hf for RES inflation magnitude")
        # RES direct vs reverb breakdown
        rd_hf = cols["r2_direct_hf"][symptom_mask]
        rv_hf = cols["r2_reverb_hf"][symptom_mask]
        rd_db = 10 * np.log10(rd_hf + 1e-12)
        rv_db = 10 * np.log10(rv_hf + 1e-12)
        print(f"  RES R² breakdown at symptom (HF):")
        print(f"    R²_direct (S²/ERLE) dB : mean={rd_db.mean():+6.2f}  "
              f"median={np.median(rd_db):+6.2f}")
        print(f"    R²_reverb (AddReverb) dB: mean={rv_db.mean():+6.2f}  "
              f"median={np.median(rv_db):+6.2f}")
        with np.errstate(divide='ignore', invalid='ignore'):
            reverb_share = rv_hf / np.maximum(rd_hf + rv_hf, 1e-30)
        print(f"    reverb / (direct+reverb): "
              f"mean={reverb_share.mean() * 100:5.1f}%  "
              f"← if >50%, AddReverb dominates R² (reverb model inflation)")
        print(f"    reverb tail_max         : mean={cols['reverb_tail_max'][symptom_mask].mean():.3e}")
        print(f"    R² path: linear={cols['r2_path_linear'][symptom_mask].mean() * 100:5.1f}%  "
              f"nonlinear={cols['r2_path_nonlinear'][symptom_mask].mean() * 100:5.1f}%")
    print()

    # ---------- Time-binned timeline (0.5-second windows) ----------
    print("--- 0.5-second windows (gain medians + flag fractions) ---")
    print("    t(s)   |  LF /  MF /  HF  | DNE   UL    CONV | poor_cnt  hov")
    win_sec = 0.5
    win_hops = int(win_sec * sr / hop)
    for w_start in range(0, n_hops, win_hops):
        w_end = min(n_hops, w_start + win_hops)
        sl = slice(w_start, w_end)
        if w_end - w_start < 5:
            continue
        t = w_start * hop / sr
        gl = cols["gain_lf_med"][sl].mean()
        gm = cols["gain_mf_med"][sl].mean()
        gh = cols["gain_hf_med"][sl].mean()
        d  = cols["dominant_nearend"][sl].mean() * 100
        u  = cols["usable_linear"][sl].mean() * 100
        cv = cols["filter_converged"][sl].mean() * 100
        pc_max = int(cols["poor_coarse_counter"][sl].max())
        hov = (cols["coarse_reset_hangover"][sl] > 0).mean() * 100
        print(f"  {t:5.1f}-{t + win_sec:5.1f}  "
              f"| {gl:.2f} / {gm:.2f} / {gh:.2f}  "
              f"| {d:4.0f}%  {u:4.0f}%  {cv:4.0f}%  "
              f"| {pc_max:3d}        {hov:3.0f}%")
    print()

    # ---------- Narrative ----------
    print("--- Narrative (SG-internal attribution) ---")
    if n_sym == 0:
        print("  No HF wipe during NE-active frames in this trace.")
    else:
        pre = cols["sg_gain_hf_med_pre_hf_lim"][symptom_mask]
        post = cols["sg_gain_hf_med_post"][symptom_mask]
        cap_drop_mean = float((pre - post).mean())
        enr_hf_med = float(np.median(cols["sg_enr_hf"][symptom_mask]))
        r2_hf_mean = float(cols["sg_r2_hf_mean"][symptom_mask].mean())
        echo_hf_mean = float(cols["echo_psd_hf"][symptom_mask].mean())
        r2_over_echo = r2_hf_mean / max(echo_hf_mean * (32768.0 ** 2), 1e-12)
        reason_min_hf = float(cols["sg_reason_min_hf"][symptom_mask].mean())
        reason_lim_hf = float(cols["sg_reason_lim_hf"][symptom_mask].mean())
        dne_during_sym = float(cols['dominant_nearend'][symptom_mask].mean())
        ul_during_sym = float(cols['usable_linear'][symptom_mask].mean())
        print(f"  HF wipe at SYMPTOM frames ({n_sym} hops):")
        print(f"    pre-HF-cap gain median  : {np.median(pre):.3f}")
        print(f"    post-HF-cap gain median : {np.median(post):.3f}")
        print(f"    HF-cap drop (pre-post)  : mean={cap_drop_mean:.3f}")
        print(f"    R²_HF / echo_psd_HF     : {r2_over_echo:.2f}  "
              f"(>>1 = RES inflated above filter echo estimate)")
        print(f"    ENR HF median           : {enr_hf_med:.3f}  "
              f"(>1 = R² above near; SG kills)")
        print(f"    reason_min HF fraction  : {reason_min_hf * 100:.0f}%")
        print(f"    reason_lim HF fraction  : {reason_lim_hf * 100:.0f}%")
        print()
        # Attribution
        if cap_drop_mean > 0.3 and float(np.median(pre)) > 0.5:
            print("  ROOT: HF cap anchor at ~2 kHz. Pre-cap HF gain is healthy")
            print("  but a single bin at the anchor is wiped → propagates to all HF.")
            print("  Check sg_hf_anchor_value: if low, that bin's R² is anomalous.")
            print("  Fix vector: anchor bin selection / window broadening.")
        elif enr_hf_med > 1.0 or r2_over_echo > 5.0:
            print("  ROOT: R² inflated FAR above the linear filter's echo estimate.")
            print(f"  R² / echo_psd ratio ≈ {r2_over_echo:.1f}x — RES is generating")
            print("  echo PSD much larger than S²_linear/ERLE alone. Suspect:")
            print("  (a) reverb model AddReverb() inflated by accumulated tail mass,")
            print("  (b) use_stationarity_properties=True scaling boosting R²,")
            print("  (c) ERLE per-bin floor=1 + small s2_linear → R²≈s2 then reverb")
            print("      pile-on inflates it. Diff vs AEC3 here.")
        elif reason_min_hf > 0.5:
            print("  ROOT: audibility min_gain floor fires on >50% of HF bins.")
            print("  EchoAudibility full per-bin JND port (Tier A #2) expected to fix.")
        else:
            if dne_during_sym > 0.5 and ul_during_sym > 0.5:
                print(f"  Mixed: DNE={dne_during_sym * 100:.0f}% / UL={ul_during_sym * 100:.0f}%")
                print("  but SG internals don't point at a single dominant gate.")
                print("  Check the SYMPTOM detail block above — `enr_hf` value tells")
                print("  whether RES is producing reasonable R²; pre/post cap delta")
                print("  tells whether HF cap is the propagator.")
            else:
                print("  No clear single-gate attribution. Possibly upstream filter")
                print("  divergence; share the full block above for case-specific read.")
    print("=" * 72)


def _plot(npz_path: str, png_path: str, *, sr: int, hop: int, n_bins: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data = np.load(npz_path)
    mic = data["mic"]
    out = data["out"]
    gain = data["gain"]  # (n_hops, n_bins)
    n_hops = gain.shape[0]
    t = np.arange(n_hops) * hop / sr

    # Spectrograms via short-time FFT (use the AEC's own block_size for
    # consistency: 320 = 2 * hop_size).
    def stft(x: np.ndarray) -> np.ndarray:
        block = 2 * hop
        fft_size = 1 << (block - 1).bit_length()
        win = np.hanning(block).astype(np.float32)
        n_frames = (len(x) - block) // hop + 1
        out_spec = np.zeros((n_frames, fft_size // 2 + 1), dtype=np.float32)
        for i in range(n_frames):
            chunk = x[i * hop:i * hop + block] * win
            out_spec[i] = np.abs(np.fft.rfft(chunk, fft_size))
        return out_spec

    mic_spec = stft(mic).T
    out_spec = stft(out).T

    fig, axes = plt.subplots(6, 1, figsize=(14, 22),
                             gridspec_kw={"height_ratios": [2, 2, 2, 1.5, 1, 1.5]})

    # Panel 1: mic spectrogram
    ax = axes[0]
    im = ax.imshow(20 * np.log10(mic_spec + 1e-8), aspect="auto",
                   origin="lower", extent=[0, len(mic) / sr, 0, sr / 2],
                   cmap="magma", vmin=-100, vmax=0)
    ax.set_title("mic spectrogram (dB)")
    ax.set_ylabel("Hz")
    plt.colorbar(im, ax=ax, fraction=0.02)

    # Panel 2: output spectrogram
    ax = axes[1]
    im = ax.imshow(20 * np.log10(out_spec + 1e-8), aspect="auto",
                   origin="lower", extent=[0, len(out) / sr, 0, sr / 2],
                   cmap="magma", vmin=-100, vmax=0)
    ax.set_title("output spectrogram (dB)  ← HF wipe shows here")
    ax.set_ylabel("Hz")
    plt.colorbar(im, ax=ax, fraction=0.02)

    # Panel 3: per-bin SuppressionGain heatmap
    ax = axes[2]
    im = ax.imshow(gain.T, aspect="auto", origin="lower",
                   extent=[0, n_hops * hop / sr, 0, sr / 2],
                   cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("SuppressionGain per-bin (1.0 = pass, 0 = wipe)")
    ax.set_ylabel("Hz")
    plt.colorbar(im, ax=ax, fraction=0.02)

    # Panel 4: per-band gain medians timeline
    ax = axes[3]
    ax.plot(t, data["gain_lf_med"], label="LF (<500 Hz)", color="tab:blue")
    ax.plot(t, data["gain_mf_med"], label="MF (500-2000)", color="tab:orange")
    ax.plot(t, data["gain_hf_med"], label="HF (>2000 Hz)", color="tab:red")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("gain median")
    ax.set_title("per-band SuppressionGain median (HF dropping = painted black)")
    ax.legend(loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 5: DNE / usable_linear / filter_converged flags
    ax = axes[4]
    ax.plot(t, data["dominant_nearend"].astype(float) * 0.9 + 2.05,
            color="tab:green", label="dominant_nearend")
    ax.plot(t, data["usable_linear"].astype(float) * 0.9 + 1.05,
            color="tab:purple", label="usable_linear")
    ax.plot(t, data["filter_converged"].astype(float) * 0.9 + 0.05,
            color="tab:gray", label="filter_converged")
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["converged", "usable_lin", "dom_NE"])
    ax.set_title("AEC state flags")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 6: per-band echo/near/error PSD medians (log)
    ax = axes[5]
    eps = 1e-12
    ax.plot(t, 10 * np.log10(data["near_psd_hf"] + eps),
            label="near HF", color="tab:blue")
    ax.plot(t, 10 * np.log10(data["echo_psd_hf"] + eps),
            label="echo HF", color="tab:red")
    ax.plot(t, 10 * np.log10(data["error_psd_hf"] + eps),
            label="error HF (linear residual)", color="tab:gray")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("PSD median (dB)")
    ax.set_title("HF PSD: near vs echo vs linear-residual error")
    ax.legend(loc="lower left", ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=110)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mic", help="mic.wav path")
    p.add_argument("lpb", help="lpb.wav path")
    p.add_argument("output_dir", help="output directory for trace.npz / trace.png / out.wav")
    p.add_argument("--mode", default="pbfdkf", help="aec mode (default: pbfdkf)")
    p.add_argument("--preset", default="balanced", help="aec preset (default: balanced)")
    p.add_argument("--no-res", action="store_true", help="disable residual chain")
    p.add_argument("--no-cng", action="store_true", help="disable CNG")
    args = p.parse_args()

    trace_case(args.mic, args.lpb, args.output_dir,
               mode=args.mode, preset=args.preset,
               enable_res=not args.no_res, cng=not args.no_cng)


if __name__ == "__main__":
    main()
