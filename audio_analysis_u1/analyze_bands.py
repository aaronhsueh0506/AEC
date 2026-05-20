"""U1 audio analysis: quantitative band-energy comparison baseline vs T1.

For each test case, compute per-band power (LF/F1/F2-F3/HF) and the
preservation ratio (output_band_power / mic_band_power). T1 should
preserve more of mic's F2-F3 energy than baseline if the HF damage
hypothesis is correct.

Run from AEC repo root: python3 audio_analysis_u1/analyze_bands.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

CASES = [
    ("QK70KpLuZ0O43BBSWEZvHg", 1.249),
    ("sYQK1rJlwU2XCy20n0Sx9g", 1.272),
    ("yc5bFUGsR0GSfiGwTTpRWg", 1.308),
    ("QkRkwwFKVEar0WtcuvJsZg", 1.309),
    ("SgKY30fjT0G8e3kQL0RHSQ", 1.331),
]

# Voice formant bands per linguistic analysis (Chinese /i/ specifically):
#   F1 ~ 300 Hz, F2 ~ 2300 Hz, F3 ~ 3000 Hz
BANDS = [
    ("LF",       0,   300),
    ("F1",     300,  1000),
    ("F2-F3", 1000,  4000),
    ("HF",    4000,  8000),
]

REPO = Path("/Users/mingyu/Desktop/novatek/SE/AEC")
MIC_DIR = REPO / "wav/aec_challenge_blind/doubletalk"
ANA_DIR = REPO / "audio_analysis_u1"


def load(path: Path) -> tuple[int, np.ndarray]:
    fs, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / (2**31)
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x


def band_power(P: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Sum power across freq bins in [lo, hi) and frames."""
    mask = (freqs >= lo) & (freqs < hi)
    return float(P[mask, :].sum())


def analyze_case(stem: str, deg: float) -> dict:
    fs_m, mic = load(MIC_DIR / f"{stem}_doubletalk_mic.wav")
    fs_b, base = load(ANA_DIR / f"{stem}_baseline.wav")
    fs_t, t1 = load(ANA_DIR / f"{stem}_t1.wav")
    assert fs_m == fs_b == fs_t == 16000, f"Sample rate mismatch on {stem}"
    n = min(len(mic), len(base), len(t1))
    mic, base, t1 = mic[:n], base[:n], t1[:n]

    # STFT (fs=16k, nperseg=512 -> 31.25 Hz/bin matches our system).
    freqs, _, Z_mic = stft(mic, fs=16000, nperseg=512, noverlap=256)
    _,     _, Z_base = stft(base, fs=16000, nperseg=512, noverlap=256)
    _,     _, Z_t1   = stft(t1,   fs=16000, nperseg=512, noverlap=256)
    P_mic, P_base, P_t1 = np.abs(Z_mic) ** 2, np.abs(Z_base) ** 2, np.abs(Z_t1) ** 2

    # Voice-active frame mask: top 50% mic-energy frames (rough VAD proxy).
    total_per_frame = P_mic.sum(axis=0)
    active = total_per_frame >= np.median(total_per_frame)
    Pm = P_mic[:, active]
    Pb = P_base[:, active]
    Pt = P_t1[:, active]

    rows = []
    for name, lo, hi in BANDS:
        em = band_power(Pm, freqs, lo, hi)
        eb = band_power(Pb, freqs, lo, hi)
        et = band_power(Pt, freqs, lo, hi)
        # Preservation ratio: output_power / mic_power (capped at 1.0
        # for display; can exceed 1.0 if echo dominates the output).
        base_ratio = eb / em if em > 0 else 0.0
        t1_ratio = et / em if em > 0 else 0.0
        improvement_db = 10 * np.log10(t1_ratio / base_ratio) if base_ratio > 0 else float('nan')
        rows.append((name, lo, hi, em, eb, et, base_ratio, t1_ratio, improvement_db))
    return {"stem": stem, "baseline_deg": deg, "rows": rows}


def fmt_row(r):
    name, lo, hi, em, eb, et, br, tr, imp = r
    return (
        f"  {name:8s} [{lo:4.0f}-{hi:4.0f} Hz]"
        f"  mic_E={em:.2e}"
        f"  base_ratio={br:.3f}"
        f"  t1_ratio={tr:.3f}"
        f"  T1/base={tr/br if br>0 else float('nan'):+.3f}x"
        f"  delta={imp:+.2f}dB"
    )


def main():
    print("=" * 80)
    print("U1 audio analysis — per-band energy preservation, baseline vs T1")
    print("(active frames = top 50% mic-energy; preservation ratio = output_E / mic_E)")
    print("=" * 80)

    summary_band = {b[0]: [] for b in BANDS}
    for stem, deg in CASES:
        r = analyze_case(stem, deg)
        print(f"\n## {stem}  (baseline AECMOS deg={deg:.3f})")
        for row in r["rows"]:
            print(fmt_row(row))
            name = row[0]
            summary_band[name].append(row[8])  # improvement_db

    print("\n" + "=" * 80)
    print("Summary — mean Δ(preservation, dB) across 5 cases by band")
    print("=" * 80)
    for name, vals in summary_band.items():
        clean = [v for v in vals if not np.isnan(v) and np.isfinite(v)]
        if clean:
            print(f"  {name:8s}: mean = {np.mean(clean):+.2f} dB"
                  f"  (min {min(clean):+.2f} / max {max(clean):+.2f})")
        else:
            print(f"  {name:8s}: no data")


if __name__ == "__main__":
    main()
