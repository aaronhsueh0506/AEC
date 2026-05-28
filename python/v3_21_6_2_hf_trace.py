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

    return {
        "gain": gain,
        "gain_lf_med": float(np.median(gain[:lf_end])),
        "gain_mf_med": float(np.median(gain[lf_end:mf_end])),
        "gain_hf_med": float(np.median(gain[mf_end:])),
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
    print()

    # ---------- Time-binned timeline (2-second windows) ----------
    print("--- 2-second windows (gain medians + flag fractions) ---")
    print("    t(s)   |  LF /  MF /  HF  | DNE   UL    CONV | poor_cnt  hov")
    win_hops = int(2.0 * sr / hop)
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
        print(f"  {t:5.1f}-{t + 2.0:5.1f}  "
              f"| {gl:.2f} / {gm:.2f} / {gh:.2f}  "
              f"| {d:4.0f}%  {u:4.0f}%  {cv:4.0f}%  "
              f"| {pc_max:3d}        {hov:3.0f}%")
    print()

    # ---------- Narrative ----------
    print("--- Narrative ---")
    if n_sym == 0:
        print("  No HF wipe during NE-active frames in this trace. The")
        print("  symptom you observed earlier may already be fixed by")
        print("  v3.21.6.2, OR this case does not exercise it. Suggest")
        print("  trying a different case if the user case still shows the")
        print("  symptom on the spectrogram.")
    else:
        dne_during_sym = cols['dominant_nearend'][symptom_mask].mean()
        ul_during_sym = cols['usable_linear'][symptom_mask].mean()
        print(f"  HF-wipe fires on {n_sym}/{n_hops} hops "
              f"({n_sym / n_hops * 100:.1f}%) of which {dne_during_sym * 100:.0f}% "
              f"are flagged dominant_nearend and {ul_during_sym * 100:.0f}% are "
              f"flagged usable_linear.")
        if dne_during_sym < 0.5:
            print("  → Candidate A: DNE FAILS to fire on >50% of symptom")
            print("    frames. Detector not catching this NE pattern; or")
            print("    Phase-2 SubtractorOutputAnalyzer + TransparentMode")
            print("    HMM may help bypass SG.")
        elif ul_during_sym > 0.5:
            print("  → Candidate B: DNE is firing AND usable_linear is")
            print("    True during the wipe — SuppressionGain consumes")
            print("    linear residual (over-aggressive). convergence_seen")
            print("    latch redesign (Tier C #11) or")
            print("    use_linear_filter_output_selection_for_final_output")
            print("    expected to recover HF.")
        else:
            print("  → Candidate C: DNE fires but usable_linear is False,")
            print("    so SG sees the capture spectrum yet still wipes HF.")
            print("    Likely the audibility-threshold path or HF cap;")
            print("    EchoAudibility full per-bin JND port (Tier A #2)")
            print("    expected to recover HF.")
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
