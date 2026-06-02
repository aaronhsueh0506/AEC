#!/usr/bin/env python3
"""Track C Step 0 — audio-proof the per-bin near-end SPP (NearendSpp) mask.

The mask must light up on genuine near-end and stay low on far-only activity
(incl. the decaying reverb tail) — the discrimination single-lag coherence
cannot do. The synthetic self-test in nearend_spp.py proved this on toy power
timelines; this script proves it on real speech.

Two modes:

  synth  (default, quantitative + ground-truth):
      Build a synthetic double-talk case with a KNOWN near-end window:
      base   = a farend_singletalk recording → mic is echo-only, no near-end
      inject = a nearend_singletalk recording's mic (near speech) added to the
               base mic over [t0, t1] at a target SER (near/echo).
      Reference (lpb) is the base's, unchanged → genuine DT (far stays active).
      Run the AEC with nearend_spp_enabled, capture per-frame p_ne, and check
      mean p_ne INSIDE the injected window ≫ mean p_ne OUTSIDE (far-only).

  real  (visual):
      Run a real doubletalk case; plot the p_ne heatmap against the mic and
      reference spectrograms for a human listen/look. No ground truth.

Usage:
  python3 python/spp_step0_diag.py synth --base <stem> --near <stem> \
      --corpus wav/aec_challenge_blind --out /tmp/spp_step0
  python3 python/spp_step0_diag.py real --case <doubletalk_stem> \
      --corpus wav/aec_challenge_blind --out /tmp/spp_step0_real
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402


def _make_aec(sample_rate=16000, filter_length=832, **spp_overrides):
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sample_rate,
        filter_length=filter_length,
        mode=AecMode.PBFDKF,
        enable_res=True,
        enable_cng=True,
        enable_shadow=True,
        nearend_spp_enabled=True,
        **spp_overrides,
    )
    np.random.seed(0)
    return AEC(cfg)


def _run_capture_p_ne(aec, mic, ref):
    """Process mic/ref hop-by-hop; return (out, p_ne_frames[n_frames, n_bins])."""
    hop = aec.hop_size
    n = min(len(mic), len(ref))
    out = np.zeros(n, dtype=np.float32)
    p_ne_frames = []
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        p = getattr(aec, "_nearend_p_ne", None)
        p_ne_frames.append(np.zeros(0) if p is None else np.asarray(p, dtype=np.float32).copy())
        pos += hop
    # Pad ragged (early frames before AEC3 chain warms may be empty).
    width = max((len(p) for p in p_ne_frames), default=0)
    arr = np.zeros((len(p_ne_frames), width), dtype=np.float32)
    for i, p in enumerate(p_ne_frames):
        if len(p):
            arr[i, :len(p)] = p
    return out[:pos], arr, hop


def _load(corpus, stem, sample_rate):
    mic, sr1 = sf.read(os.path.join(corpus, _bucket(stem), stem + "_mic.wav"))
    lpb, sr2 = sf.read(os.path.join(corpus, _bucket(stem), stem + "_lpb.wav"))
    assert sr1 == sample_rate and sr2 == sample_rate, f"sr mismatch {sr1}/{sr2}"
    return np.asarray(mic, np.float32), np.asarray(lpb, np.float32)


def _bucket(stem):
    if "doubletalk" in stem:
        return "doubletalk"
    if "farend_singletalk" in stem:
        return "farend_singletalk"
    if "nearend_singletalk" in stem:
        return "nearend_singletalk"
    raise ValueError(f"cannot infer bucket from {stem}")


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-20))


def run_synth(args):
    sr = args.sample_rate
    base_mic, base_lpb = _load(args.corpus, args.base, sr)   # echo-only mic
    near_mic, _ = _load(args.corpus, args.near, sr)          # near speech

    n = min(len(base_mic), len(base_lpb))
    base_mic, base_lpb = base_mic[:n], base_lpb[:n]
    dur = n / sr
    if args.t0 is not None and args.t1 is not None:
        t0, t1 = args.t0, args.t1
    else:
        # Auto-pick a 3-s window where the far-end (reference) is active, so the
        # injected near-end produces genuine double-talk (not near-over-silence).
        win = args.win_len
        far_rms_s = np.array([_rms(base_lpb[s * sr:(s + 1) * sr])
                              for s in range(int(dur))])
        active = far_rms_s > 0.02
        t0 = None
        for s in range(len(active) - int(win)):
            if active[s:s + int(win)].all():
                t0 = float(s)
                break
        if t0 is None:                       # fallback: loudest far second
            t0 = float(max(0, int(np.argmax(far_rms_s)) - 1))
        t1 = min(t0 + win, dur - 0.5)
    i0, i1 = int(t0 * sr), int(t1 * sr)

    # Window the near clip to [i0,i1] and scale to the target SER (near/echo)
    # measured against the base-mic echo energy inside the window.
    seg_len = i1 - i0
    near_seg = near_mic[:seg_len] if len(near_mic) >= seg_len else \
        np.pad(near_mic, (0, seg_len - len(near_mic)))
    echo_rms = _rms(base_mic[i0:i1])
    near_rms = _rms(near_seg)
    ser_lin = 10.0 ** (args.ser_db / 20.0)
    scale = (ser_lin * echo_rms / (near_rms + 1e-20)) if near_rms > 0 else 0.0
    # Raised-cosine fade to avoid injection-edge transients reading as onsets.
    fade = min(int(0.05 * sr), seg_len // 4)
    env = np.ones(seg_len, dtype=np.float32)
    if fade > 0:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade)))
        env[:fade] = ramp
        env[-fade:] = ramp[::-1]
    mic = base_mic.copy()
    mic[i0:i1] += (scale * env * near_seg).astype(np.float32)

    aec = _make_aec(sample_rate=sr, filter_length=args.filter,
                    nearend_spp_alpha=args.alpha,
                    nearend_spp_minima_subwindow=args.subwindow,
                    nearend_spp_spike_thr_db=args.thr,
                    nearend_spp_spike_soft_db=args.soft)
    out, p_ne, hop = _run_capture_p_ne(aec, mic, base_lpb)

    fps = sr / hop
    f0, f1 = int(t0 * fps), int(t1 * fps)
    p_mean_t = p_ne.mean(axis=1) if p_ne.shape[1] else np.zeros(p_ne.shape[0])
    inside = p_mean_t[f0:f1]
    outside = np.concatenate([p_mean_t[:f0], p_mean_t[f1:]])
    # Drop the AEC3 warm-up region from the "outside" stat.
    warm = int(1.0 * fps)
    outside_warm = p_mean_t[warm:f0]
    m_in = float(inside.mean()) if len(inside) else 0.0
    m_out = float(outside.mean()) if len(outside) else 0.0
    m_out_w = float(outside_warm.mean()) if len(outside_warm) else m_out

    print(f"[synth] base={args.base} near={args.near} SER={args.ser_db:+.0f}dB "
          f"near-window=[{t0:.2f},{t1:.2f}]s of {dur:.2f}s")
    print(f"[synth] mean p_ne  INSIDE near window = {m_in:.3f}")
    print(f"[synth] mean p_ne  OUTSIDE (far-only)  = {m_out:.3f}  "
          f"(post-warmup far-only = {m_out_w:.3f})")
    sep = m_in - m_out_w
    print(f"[synth] separation (in − far-only) = {sep:+.3f}  "
          f"=> {'PASS' if (m_in > 0.5 and sep > 0.3) else 'WEAK/FAIL'}")

    # Coarse 1-second timeline: mean p_ne, far-end RMS, mic RMS — to see WHERE
    # p_ne is high (does it track near-end, or just far-end activity?).
    print("[synth] 1-s timeline (p_ne | far_rms | mic_rms ; * = near-window):")
    sec = int(fps)
    for s in range(int(np.ceil(len(p_mean_t) / sec))):
        a, b = s * sec, min((s + 1) * sec, len(p_mean_t))
        sa, sb = s * sr, min((s + 1) * sr, len(base_lpb))
        mark = "*" if (s >= t0 and s < t1) else " "
        print(f"   t={s:2d}s{mark} p_ne={p_mean_t[a:b].mean():.3f} "
              f"far_rms={_rms(base_lpb[sa:sb]):.4f} mic_rms={_rms(mic[sa:sb]):.4f}")

    _plot_synth(mic, base_lpb, out, p_ne, sr, hop, (t0, t1), args.out + ".png")
    sf.write(args.out + "_synthmic.wav", mic, sr)
    sf.write(args.out + "_out.wav", out, sr)
    print(f"[synth] wrote {args.out}.png + _synthmic.wav + _out.wav")


def run_real(args):
    sr = args.sample_rate
    mic, lpb = _load(args.corpus, args.case, sr)
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]
    aec = _make_aec(sample_rate=sr, filter_length=args.filter)
    out, p_ne, hop = _run_capture_p_ne(aec, mic, lpb)
    print(f"[real] case={args.case} dur={n/sr:.2f}s "
          f"mean p_ne={p_ne.mean():.3f}  frac(p_ne>0.5)={np.mean(p_ne>0.5):.3f}")
    _plot_synth(mic, lpb, out, p_ne, sr, hop, None, args.out + ".png")
    print(f"[real] wrote {args.out}.png")


def _spec_db(x, sr, nfft=512, hop=256):
    from numpy.fft import rfft
    nb = max(0, (len(x) - nfft) // hop + 1)
    win = np.hanning(nfft).astype(np.float32)
    S = np.zeros((nfft // 2 + 1, max(nb, 1)), dtype=np.float32)
    for k in range(nb):
        S[:, k] = np.abs(rfft(x[k * hop:k * hop + nfft] * win)) + 1e-10
    return 20 * np.log10(S)


def _plot_synth(mic, ref, out, p_ne, sr, hop, window, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib unavailable — skipping plot", file=sys.stderr)
        return
    n = len(mic)
    dur = n / sr
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("SPP Step 0: per-bin near-end probability vs spectra", fontsize=12)

    for a, sig, lab in [(ax[0], ref, "reference (far) spec"),
                        (ax[1], mic, "mic (echo + injected near) spec")]:
        S = _spec_db(sig, sr)
        a.imshow(S, aspect="auto", origin="lower",
                 extent=[0, dur, 0, sr / 2], vmin=-80, vmax=0, cmap="magma")
        a.set_ylabel(lab + "\n(Hz)")
        if window:
            a.axvline(window[0], color="cyan", lw=1.0)
            a.axvline(window[1], color="cyan", lw=1.0)

    # p_ne heatmap (bin index × frame → time)
    fps = sr / hop
    t_end = p_ne.shape[0] / fps
    ax[2].imshow(p_ne.T, aspect="auto", origin="lower",
                 extent=[0, t_end, 0, p_ne.shape[1]], vmin=0, vmax=1, cmap="viridis")
    ax[2].set_ylabel("p_ne heatmap\n(bin)")
    if window:
        ax[2].axvline(window[0], color="cyan", lw=1.0)
        ax[2].axvline(window[1], color="cyan", lw=1.0)

    p_mean_t = p_ne.mean(axis=1) if p_ne.shape[1] else np.zeros(p_ne.shape[0])
    t = np.arange(len(p_mean_t)) / fps
    ax[3].plot(t, p_mean_t, lw=0.8, color="#2ca02c")
    ax[3].axhline(0.5, color="#888", lw=0.5, ls="--")
    ax[3].set_ylim(0, 1)
    ax[3].set_ylabel("mean p_ne(t)")
    ax[3].set_xlabel("time (s)")
    if window:
        ax[3].axvspan(window[0], window[1], color="cyan", alpha=0.2,
                      label="injected near-end")
        ax[3].legend(loc="upper right", fontsize=8)
    for a in ax:
        a.set_xlim(0, dur)
    fig.savefig(png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("synth")
    ps.add_argument("--base", required=True, help="farend_singletalk stem (echo-only)")
    ps.add_argument("--near", required=True, help="nearend_singletalk stem (near speech)")
    ps.add_argument("--ser-db", type=float, default=0.0, help="near/echo ratio in window")
    ps.add_argument("--t0", type=float, default=None)
    ps.add_argument("--t1", type=float, default=None)
    ps.add_argument("--win-len", type=float, default=3.0,
                    help="auto-selected far-active window length (s)")
    ps.add_argument("--alpha", type=float, default=0.02)
    ps.add_argument("--subwindow", type=int, default=200)
    ps.add_argument("--thr", type=float, default=5.0)
    ps.add_argument("--soft", type=float, default=2.0)

    pr = sub.add_parser("real")
    pr.add_argument("--case", required=True, help="doubletalk stem")

    for sp in (ps, pr):
        sp.add_argument("--corpus", default="wav/aec_challenge_blind")
        sp.add_argument("--out", default="/tmp/spp_step0")
        sp.add_argument("--sample-rate", type=int, default=16000)
        sp.add_argument("--filter", type=int, default=832)

    args = p.parse_args()
    if args.mode == "synth":
        run_synth(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
