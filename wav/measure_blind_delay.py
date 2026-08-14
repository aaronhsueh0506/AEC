#!/usr/bin/env python3
"""Per-clip mic-vs-lpb bulk-delay measurement for AEC challenge blind sets.

Method (validated on the FST 579.44ms case):
  - trim both channels to common length (kills the zero-pad boundary peak
    that fooled GCC-PHAT: file-length delta shows up as a fake peak)
  - full cross-correlation via FFT, restricted to lag in [-100ms, +1200ms]
  - per-lag normalized cross-correlation (sliding far energy via cumsum)
  - peak + dominance (second peak outside +/-48ms exclusion)
  - undecidable gate: peak_ncc < 0.03 (NST etc.: mic contains no far)
"""
import sys, glob, os
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

LAG_MIN_MS, LAG_MAX_MS = -100.0, 1200.0
EXCL_MS = 48.0
UNDECIDABLE_NCC = 0.03


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x[:, 0]
    return sr, x / 32768.0


def measure_pair(mic_path, lpb_path):
    sr_m, mic = load(mic_path)
    sr_l, lpb = load(lpb_path)
    assert sr_m == sr_l, f"rate mismatch {sr_m} vs {sr_l}"
    sr = sr_m
    n = min(mic.size, lpb.size)
    mic, lpb = mic[:n], lpb[:n]
    mic = mic - mic.mean()
    lpb = lpb - lpb.mean()

    lo = int(LAG_MIN_MS * sr / 1000.0)
    hi = int(LAG_MAX_MS * sr / 1000.0)
    # c[k] = sum_i mic[i] * lpb[i - k]  (k = delay of lpb inside mic)
    c = fftconvolve(mic, lpb[::-1], mode="full")   # index n-1+k <-> lag k
    lags = np.arange(lo, hi + 1)
    lags = lags[(lags > -n + 1) & (lags < n)]
    ck = c[n - 1 + lags]

    # per-lag normalization: energy of the overlapping lpb segment
    e_l = np.concatenate(([0.0], np.cumsum(lpb * lpb)))
    e_m = float(np.dot(mic, mic))
    # overlap for lag k: lpb[max(0,-k) : n-max(0,k)]
    a = np.maximum(0, -lags)
    b = n - np.maximum(0, lags)
    seg_e = np.maximum(e_l[b] - e_l[a], 1e-12)
    ncc = ck / np.sqrt(e_m * seg_e)

    i = int(np.argmax(ncc))
    peak_lag, peak = int(lags[i]), float(ncc[i])
    excl = int(EXCL_MS * sr / 1000.0)
    mask = np.abs(lags - peak_lag) > excl
    second = float(ncc[mask].max()) if mask.any() else 0.0
    dom = peak / max(second, 1e-9)
    return {
        "sr": sr, "delay_ms": 1000.0 * peak_lag / sr, "ncc": peak,
        "second": second, "dom": dom,
        "decided": peak >= UNDECIDABLE_NCC,
    }


def scenario_of(name):
    n = name.replace("-", "_")
    for s in ("farend_singletalk_with_movement", "doubletalk_with_movement",
              "farend_singletalk", "nearend_singletalk", "doubletalk"):
        if s in n:
            return s
    return "unknown"


def main(root):
    rows = []
    for mic_path in sorted(glob.glob(os.path.join(root, "**", "*mic.wav"),
                                     recursive=True)):
        lpb_path = mic_path.replace("mic.wav", "lpb.wav")
        if not os.path.exists(lpb_path):
            continue
        r = measure_pair(mic_path, lpb_path)
        r["scenario"] = scenario_of(os.path.basename(mic_path))
        r["clip"] = os.path.basename(mic_path)[:24]
        rows.append(r)
        flag = "" if r["decided"] else "  [UNDECIDABLE]"
        print(f'{r["scenario"]:34s} {r["clip"]:26s} sr={r["sr"]} '
              f'delay={r["delay_ms"]:8.2f}ms ncc={r["ncc"]:.4f} '
              f'dom={r["dom"]:.2f}{flag}')

    print("\n== per-scenario stats (decided clips only) ==")
    for s in sorted({r["scenario"] for r in rows}):
        d = [r["delay_ms"] for r in rows if r["scenario"] == s and r["decided"]]
        u = sum(1 for r in rows if r["scenario"] == s and not r["decided"])
        if d:
            d = np.array(d)
            print(f"{s:34s} n={d.size:3d} undec={u:3d} "
                  f"p50={np.percentile(d, 50):7.1f} p90={np.percentile(d, 90):7.1f} "
                  f"max={d.max():7.1f} min={d.min():7.1f} "
                  f">509ms: {(d > 509).sum()}  >128ms: {(d > 128).sum()}")
        else:
            print(f"{s:34s} n=  0 undec={u:3d} (no decided clips)")


if __name__ == "__main__":
    main(sys.argv[1])
