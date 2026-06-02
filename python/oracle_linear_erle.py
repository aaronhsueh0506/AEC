#!/usr/bin/env python3
"""Phase 0a — Oracle achievable-linear-ERLE ceiling.

Answers the A/B fork: is the linear filter broken (structural bug) or is the
echo genuinely linear-unlearnable?

For each case we compute the LS-optimal FIR h of length L minimising
||mic - h*ref||^2 (Wiener-Hopf normal equations, solved via Toeplitz).
ERLE_ceiling(L) = 10*log10(||mic||^2 / ||mic - h*ref||^2).

We sweep L over {60ms (our 6-partition span), 250ms, 500ms}. Reading:
  - oracle@60ms >> online get_erle  -> A: online filter under-converges within span
  - oracle@500ms >> oracle@60ms      -> A: delay/length — echo is linear but outside 60ms span
  - oracle@500ms ~= 0                -> B: genuinely non-linear / unlearnable

FS cases are the cleanest probe (no near-end in mic = pure echo+noise).

Usage:
  python3 oracle_linear_erle.py --bucket farend_singletalk -n 30
  python3 oracle_linear_erle.py --stems lH20r2 0I0XMl 0KjzXA --bucket farend_singletalk
  python3 oracle_linear_erle.py --sanity            # synthetic validation only
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy.linalg import solve_toeplitz

SR = 16000
WAV_ROOT = os.path.join(os.path.dirname(__file__), "..", "wav", "aec_challenge_blind")


def _xcorr_fft(a, b, maxlag):
    """Cross-correlation r[k] = sum_n a[n-k]*b[n], k=0..maxlag-1 (a delayed)."""
    n = len(a) + len(b)
    nfft = 1 << int(np.ceil(np.log2(n)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    # r[k] = sum_n a[n-k] b[n] = IFFT(conj(A)*B)[k] for k>=0
    r = np.fft.irfft(np.conj(A) * B, nfft)
    return r[:maxlag].astype(np.float64)


def oracle_erle(mic, ref, L, ridge=1e-6):
    """LS-optimal length-L causal FIR ceiling ERLE in dB."""
    mic = mic.astype(np.float64)
    ref = ref.astype(np.float64)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    # Normal equations: R_xx h = r_xy ; R_xx Toeplitz from autocorr of ref.
    rxx = _xcorr_fft(ref, ref, L)          # autocorr lags 0..L-1
    rxy = _xcorr_fft(ref, mic, L)          # cross r[k]=sum ref[n-k] mic[n]
    rxx[0] += ridge * (rxx[0] + 1e-20)     # ridge for conditioning
    try:
        h = solve_toeplitz((rxx, rxx), rxy)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    # residual = mic - (h * ref)  causal FIR
    est = np.convolve(ref, h)[:n]
    resid = mic - est
    p_mic = float(np.sum(mic * mic))
    p_res = float(np.sum(resid * resid))
    erle = 10.0 * np.log10((p_mic + 1e-20) / (p_res + 1e-20))
    return erle, float(np.argmax(np.abs(h)))  # erle, peak-tap (≈ delay in samples)


def run_sanity():
    print("=== SANITY: synthetic linear echo (white ref, 200-tap decaying path) ===")
    rng = np.random.default_rng(0)
    N = 16000 * 6
    ref = rng.standard_normal(N)
    htrue = (rng.standard_normal(200) * np.exp(-np.arange(200) / 40.0)).astype(np.float64)
    delay = 1200  # 75 ms bulk delay -> echo outside 60ms(960) span
    echo = np.convolve(ref, np.concatenate([np.zeros(delay), htrue]))[:N]
    for snr_db in (40, 20):
        noise = rng.standard_normal(N)
        noise *= np.sqrt(np.sum(echo**2) / np.sum(noise**2)) * 10 ** (-snr_db / 20)
        mic = echo + noise
        for L in (960, 4000, 8000):
            e, pk = oracle_erle(mic, ref, L)
            print(f"  SNR={snr_db}dB L={L:5d}({L*1000//SR:3d}ms): oracle ERLE={e:6.2f}dB  peaktap={pk:.0f}")
        print(f"    (expect: L=960<<; L>=4000 ~= {snr_db}dB; peaktap~={delay+np.argmax(np.abs(htrue))})")
    print()


def find_case(stem_frag, bucket):
    pat = os.path.join(WAV_ROOT, bucket, f"*{stem_frag}*_mic.wav")
    hits = sorted(glob.glob(pat))
    return hits[0][:-8] if hits else None  # strip "_mic.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="farend_singletalk")
    ap.add_argument("-n", type=int, default=20, help="sample N cases from bucket")
    ap.add_argument("--stems", nargs="*", default=None, help="specific stem fragments")
    ap.add_argument("--lengths", nargs="*", type=int, default=[960, 4000, 8000])
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("-o", default="/tmp/oracle_erle.json")
    args = ap.parse_args()

    if args.sanity:
        run_sanity()
        return

    run_sanity()  # always sanity-check the solver first

    if args.stems:
        bases = [find_case(s, args.bucket) for s in args.stems]
        bases = [b for b in bases if b]
    else:
        mics = sorted(glob.glob(os.path.join(WAV_ROOT, args.bucket, "*_mic.wav")))
        step = max(1, len(mics) // args.n)
        bases = [m[:-8] for m in mics[::step][: args.n]]

    print(f"=== ORACLE linear-ERLE  bucket={args.bucket}  {len(bases)} cases  L(ms)={[l*1000//SR for l in args.lengths]} ===")
    hdr = "stem".ljust(26) + "".join(f"L{l*1000//SR}ms".rjust(9) for l in args.lengths) + "  peaktap(ms)@max"
    print(hdr)
    rows = []
    for base in bases:
        mic, _ = sf.read(base + "_mic.wav")
        ref, _ = sf.read(base + "_lpb.wav")
        if mic.ndim > 1:
            mic = mic[:, 0]
        if ref.ndim > 1:
            ref = ref[:, 0]
        erles = {}
        peak = None
        for L in args.lengths:
            e, pk = oracle_erle(mic, ref, L)
            erles[L] = e
            if L == args.lengths[-1]:
                peak = pk
        stem = os.path.basename(base)[:24]
        line = stem.ljust(26) + "".join(f"{erles[l]:9.2f}" for l in args.lengths)
        line += f"   {peak/SR*1000:7.1f}"
        print(line)
        rows.append({"stem": stem, "erle": {str(l): erles[l] for l in args.lengths},
                     "peaktap_ms": peak / SR * 1000})

    arr = {str(l): np.array([r["erle"][str(l)] for r in rows]) for l in args.lengths}
    print("-" * len(hdr))
    meanline = "MEAN".ljust(26) + "".join(f"{np.nanmean(arr[str(l)]):9.2f}" for l in args.lengths)
    print(meanline)
    medline = "MEDIAN".ljust(26) + "".join(f"{np.nanmedian(arr[str(l)]):9.2f}" for l in args.lengths)
    print(medline)
    json.dump(rows, open(args.o, "w"), indent=2)
    print(f"\nsaved -> {args.o}")


if __name__ == "__main__":
    main()
