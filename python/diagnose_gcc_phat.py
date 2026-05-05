#!/usr/bin/env python3
"""GCC-PHAT diagnostic for 7GT-class skew investigation.

Standalone tool that processes mic/lpb pairs with a wide-search GCC-PHAT
to discriminate three hypotheses about why the production DelayEstimator
fails to align certain BT/mobile cases:

  H1: real skew is within 1024 ms but rejected by peak-selection / PAR
      / hysteresis gates.
  H2: real skew exceeds 1024 ms (out of search range).
  H3: PAR structure is double-peaked / ambiguous (single-lag GCC-PHAT
      fundamentally insufficient).

Replicates the production gate logic (PAR low=5.0 / solid=8.0,
n_updates>=3, max_delay_samples=16384) WITHOUT touching aec.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Production-equivalent constants (mirrored from aec.py DelayEstimator)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
PROD_MAX_DELAY_MS = 1024.0
PROD_MAX_DELAY_SAMPLES = int(PROD_MAX_DELAY_MS * SAMPLE_RATE / 1000)  # 16384
PAR_LOW = 5.0
PAR_SOLID = 8.0
N_UPDATES_GATE = 3
EMA_ALPHA = 0.6  # production smoothing

# Wide search: 2 seconds = 32000 samples; PAR-search range covers 0..32000.
# To accommodate this we need seg_size >= 2 * search_range -> next pow2 >= 64000.
WIDE_SEARCH_SAMPLES = 32000  # 2.0 s
SEG_SIZE = 65536              # next pow2 >= 2 * 32000
SEG_HOP = SEG_SIZE // 2


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path}: sr={w.getframerate()} (expected 16000)"
        nch = w.getnchannels()
        sw = w.getsampwidth()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        # could be int32 or float32 — assume int32 PCM
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sampwidth {sw}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x.astype(np.float64)


# ---------------------------------------------------------------------------
# GCC-PHAT (fp64) — single segment
# ---------------------------------------------------------------------------
def gcc_phat_segment(mic: np.ndarray, ref: np.ndarray, seg_size: int) -> np.ndarray:
    """Return GCC-PHAT correlation function (length seg_size, abs values)."""
    assert len(mic) == seg_size and len(ref) == seg_size
    M = np.fft.rfft(mic, n=seg_size)
    R = np.fft.rfft(ref, n=seg_size)
    cross = M * np.conj(R)
    mag = np.abs(cross) + 1e-12
    phat = cross / mag
    gcc = np.fft.irfft(phat, n=seg_size)
    return gcc.astype(np.float64)


def find_top_peaks(gcc_abs: np.ndarray, search_max: int, k: int = 5,
                   exclusion: int = 16) -> List[Tuple[int, float]]:
    """Return top-k (lag, height) within [0, search_max] using a greedy
    suppression over an exclusion radius (samples) to avoid returning the
    same lobe k times."""
    region = gcc_abs[: search_max + 1].copy()
    out: List[Tuple[int, float]] = []
    for _ in range(k):
        idx = int(np.argmax(region))
        height = float(region[idx])
        if height <= 0:
            break
        out.append((idx, height))
        lo = max(0, idx - exclusion)
        hi = min(len(region), idx + exclusion + 1)
        region[lo:hi] = -1.0  # suppress this lobe
    return out


def par_for_peak(gcc_abs: np.ndarray, search_max: int, peak_lag: int) -> float:
    """Replicate production PAR formula: peak / mean(|gcc[0..max_d]|) excluding peak."""
    region = np.abs(gcc_abs[: search_max + 1])
    peak = float(region[peak_lag])
    mean_excl = (float(region.sum()) - peak) / (len(region) - 1 + 1e-10)
    return float(peak / (mean_excl + 1e-10))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class BlockResult:
    block_idx: int
    t_start_s: float
    top_peaks_wide: List[Tuple[int, float, float]]  # (lag, height, par)
    prod_lag: int                                    # argmax in [0, 16384]
    prod_par: float
    prod_confidence: float
    prod_gate: str  # which gate would reject
    H1: bool
    H2: bool
    H3: bool


def analyze_case(name: str, mic_path: str, ref_path: str, n_updates_total: int) -> dict:
    """Process one case. Returns summary dict and prints stdout progress."""
    mic = load_wav_mono(mic_path)
    ref = load_wav_mono(ref_path)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]

    # Use sliding GCC-PHAT with EMA cross-spectrum (mirrors production), but
    # emit a per-block snapshot AND a final wide-range estimate.
    n_freqs = SEG_SIZE // 2 + 1
    cross_spec = np.zeros(n_freqs, dtype=np.complex128)
    n_updates = 0

    blocks: List[BlockResult] = []
    pos = 0
    block_idx = 0
    while pos + SEG_SIZE <= n:
        m_seg = mic[pos : pos + SEG_SIZE]
        r_seg = ref[pos : pos + SEG_SIZE]
        M = np.fft.rfft(m_seg, n=SEG_SIZE)
        R = np.fft.rfft(r_seg, n=SEG_SIZE)
        cross = M * np.conj(R)
        n_updates += 1
        if n_updates == 1:
            cross_spec = cross.copy()
        else:
            cross_spec = EMA_ALPHA * cross_spec + (1 - EMA_ALPHA) * cross

        mag = np.abs(cross_spec) + 1e-12
        phat = cross_spec / mag
        gcc = np.fft.irfft(phat, n=SEG_SIZE)
        gcc_abs = np.abs(gcc)

        # Top-5 over wide search range (0..WIDE_SEARCH_SAMPLES).
        top5 = find_top_peaks(gcc_abs, WIDE_SEARCH_SAMPLES, k=5)
        top5_with_par = [
            (lag, h, par_for_peak(gcc_abs, WIDE_SEARCH_SAMPLES, lag)) for (lag, h) in top5
        ]

        # What production would do: argmax over [0..PROD_MAX_DELAY_SAMPLES].
        prod_region = gcc_abs[: PROD_MAX_DELAY_SAMPLES + 1]
        prod_lag = int(np.argmax(prod_region))
        prod_peak = float(prod_region[prod_lag])
        prod_mean_excl = (
            float(prod_region.sum()) - prod_peak
        ) / (len(prod_region) - 1 + 1e-10)
        prod_par = float(prod_peak / (prod_mean_excl + 1e-10))

        # confidence
        if n_updates < N_UPDATES_GATE:
            prod_conf = 0.0
        elif prod_par <= PAR_LOW:
            prod_conf = 0.0
        elif prod_par >= PAR_SOLID:
            prod_conf = 1.0
        else:
            prod_conf = (prod_par - PAR_LOW) / (PAR_SOLID - PAR_LOW)

        # gate label
        if n_updates < N_UPDATES_GATE:
            gate = "n_updates<3"
        elif prod_par < PAR_LOW:
            gate = "PAR<low(5.0)"
        elif prod_par < PAR_SOLID:
            gate = "PAR<solid(8.0) [partial]"
        else:
            gate = "PASS"

        # Hypothesis fire (use wide top peak for "true" lag)
        wide_top = top5_with_par[0] if top5_with_par else (0, 0.0, 0.0)
        true_lag = wide_top[0]
        true_par = wide_top[2]

        H1 = (true_lag < PROD_MAX_DELAY_SAMPLES) and (prod_par < PAR_SOLID) and (true_par >= PAR_LOW)
        H2 = true_lag >= PROD_MAX_DELAY_SAMPLES
        # H3: top-2 within 32 samples and both PAR>5 and no peak with PAR>8
        H3 = False
        if len(top5_with_par) >= 2:
            l1, _, p1 = top5_with_par[0]
            l2, _, p2 = top5_with_par[1]
            if abs(l1 - l2) <= 32 and p1 > PAR_LOW and p2 > PAR_LOW and p1 < PAR_SOLID:
                H3 = True

        blocks.append(BlockResult(
            block_idx=block_idx,
            t_start_s=pos / SAMPLE_RATE,
            top_peaks_wide=top5_with_par,
            prod_lag=prod_lag,
            prod_par=prod_par,
            prod_confidence=prod_conf,
            prod_gate=gate,
            H1=H1, H2=H2, H3=H3,
        ))

        block_idx += 1
        pos += SEG_HOP

    # Final wide-search estimate from final EMA cross-spectrum
    if n_updates == 0:
        return {"name": name, "error": "file too short for one segment"}

    mag = np.abs(cross_spec) + 1e-12
    phat = cross_spec / mag
    gcc = np.fft.irfft(phat, n=SEG_SIZE)
    gcc_abs = np.abs(gcc)
    final_top = find_top_peaks(gcc_abs, WIDE_SEARCH_SAMPLES, k=5)
    final_top_with_par = [
        (lag, h, par_for_peak(gcc_abs, WIDE_SEARCH_SAMPLES, lag)) for (lag, h) in final_top
    ]

    if final_top_with_par:
        true_lag, _, true_par = final_top_with_par[0]
    else:
        true_lag, true_par = 0, 0.0

    inside = true_lag < PROD_MAX_DELAY_SAMPLES
    secondary_close = False
    if len(final_top_with_par) >= 2:
        secondary_close = abs(final_top_with_par[0][0] - final_top_with_par[1][0]) <= 32

    # Vote across blocks
    votes_H1 = sum(1 for b in blocks if b.H1)
    votes_H2 = sum(1 for b in blocks if b.H2)
    votes_H3 = sum(1 for b in blocks if b.H3)
    nb = max(1, len(blocks))
    overall = max(
        [("H1", votes_H1), ("H2", votes_H2), ("H3", votes_H3)],
        key=lambda kv: kv[1],
    )
    if overall[1] == 0:
        # No hypothesis fired -> production would have succeeded with PAR>=8 inside range
        overall_label = "PASS_OR_UNKNOWN"
    else:
        overall_label = overall[0]

    return {
        "name": name,
        "n_blocks": len(blocks),
        "blocks": blocks,
        "final_top": final_top_with_par,
        "true_lag": true_lag,
        "true_lag_ms": 1000.0 * true_lag / SAMPLE_RATE,
        "true_par": true_par,
        "inside_prod_range": inside,
        "secondary_within_32": secondary_close,
        "votes": {"H1": votes_H1, "H2": votes_H2, "H3": votes_H3, "n_blocks": nb},
        "verdict": overall_label,
    }


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
CASES = [
    ("7GT_doubletalk",
     "wav/aec_challenge_blind/doubletalk/7GTxyTksSUqCnP5y0ILG4A_doubletalk_mic.wav",
     "wav/aec_challenge_blind/doubletalk/7GTxyTksSUqCnP5y0ILG4A_doubletalk_lpb.wav"),
    ("7GT_farend_singletalk",
     "wav/aec_challenge_blind/farend_singletalk/7GTxyTksSUqCnP5y0ILG4A_farend_singletalk_mic.wav",
     "wav/aec_challenge_blind/farend_singletalk/7GTxyTksSUqCnP5y0ILG4A_farend_singletalk_lpb.wav"),
    ("IrQv_farend_singletalk",
     "wav/aec_challenge_blind/farend_singletalk/IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_mic.wav",
     "wav/aec_challenge_blind/farend_singletalk/IrQvqOTCmEWMXn9k2ICtRQ_farend_singletalk_lpb.wav"),
    ("pcb1_farend_singletalk",
     "wav/aec_challenge_blind/farend_singletalk/pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk_mic.wav",
     "wav/aec_challenge_blind/farend_singletalk/pcb1Nh0Z3k0WS9a7gBEuqg_farend_singletalk_lpb.wav"),
    ("S22F_farend_singletalk",
     "wav/aec_challenge_blind/farend_singletalk/S22FCqKDWUyymN1YbpItIw_farend_singletalk_mic.wav",
     "wav/aec_challenge_blind/farend_singletalk/S22FCqKDWUyymN1YbpItIw_farend_singletalk_lpb.wav"),
]


def fmt_lag(lag: int) -> str:
    return f"{lag:>6d} ({1000.0*lag/SAMPLE_RATE:6.1f}ms)"


def print_case(res: dict) -> None:
    print(f"\n=== {res['name']} ===")
    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return
    print(f"  blocks: {res['n_blocks']}, seg={SEG_SIZE} (4096ms), hop={SEG_HOP} (2048ms)")
    print(f"  Final wide-search top-5:")
    for i, (lag, h, par) in enumerate(res["final_top"]):
        marker = "<<" if i == 0 else "  "
        in_range = "in" if lag < PROD_MAX_DELAY_SAMPLES else "OUT"
        print(f"    #{i+1} {marker} lag={fmt_lag(lag)} [{in_range}]  height={h:.5f}  PAR={par:6.2f}")
    print(f"  TRUE delay (deep search): {fmt_lag(res['true_lag'])} PAR={res['true_par']:.2f} "
          f"  inside_1024ms={res['inside_prod_range']}  2nd_within_32samp={res['secondary_within_32']}")
    v = res["votes"]
    print(f"  Block votes: H1={v['H1']}/{v['n_blocks']}  H2={v['H2']}/{v['n_blocks']}  H3={v['H3']}/{v['n_blocks']}")
    print(f"  Overall verdict: {res['verdict']}")
    # Per-block detail (compressed)
    print(f"  Per-block:")
    print(f"    {'blk':>3} {'t(s)':>6} {'prod_lag':>10} {'prod_PAR':>9} {'prod_conf':>10} {'gate':<25} {'wide_top1_lag':>14} {'wide_top1_PAR':>14} H1 H2 H3")
    for b in res["blocks"]:
        wt = b.top_peaks_wide[0] if b.top_peaks_wide else (0, 0.0, 0.0)
        print(f"    {b.block_idx:>3d} {b.t_start_s:>6.2f} "
              f"{b.prod_lag:>10d} {b.prod_par:>9.2f} {b.prod_confidence:>10.2f} "
              f"{b.prod_gate:<25} {wt[0]:>14d} {wt[2]:>14.2f} "
              f"{int(b.H1)}  {int(b.H2)}  {int(b.H3)}")


def write_report(results: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append("# 7GT-class GCC-PHAT alignment research log\n")
    lines.append("Generated by `python/diagnose_gcc_phat.py`. Wide-search GCC-PHAT (lag 0..2.0s) "
                 "applied to 7GT primary case + three FS_static worst-echo suspects to "
                 "discriminate H1/H2/H3.\n")
    lines.append("## Configuration\n")
    lines.append(f"- sample rate: {SAMPLE_RATE} Hz\n"
                 f"- seg_size: {SEG_SIZE} ({1000.0*SEG_SIZE/SAMPLE_RATE:.0f} ms), hop: {SEG_HOP}\n"
                 f"- wide search range: 0..{WIDE_SEARCH_SAMPLES} samples ({1000.0*WIDE_SEARCH_SAMPLES/SAMPLE_RATE:.0f} ms)\n"
                 f"- production max_delay: {PROD_MAX_DELAY_SAMPLES} samples (1024 ms)\n"
                 f"- PAR thresholds: low={PAR_LOW}, solid={PAR_SOLID}; n_updates gate>={N_UPDATES_GATE}\n"
                 f"- EMA alpha (cross-spec): {EMA_ALPHA}\n")
    lines.append("\n## Per-case verdicts\n")
    lines.append("| case | true_lag (samp / ms) | inside_1024ms | true_PAR | 2nd_within_32 | verdict | block H1/H2/H3 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR | - | - | - | - | - |\n")
            continue
        v = r["votes"]
        lines.append(f"| {r['name']} | {r['true_lag']} / {1000.0*r['true_lag']/SAMPLE_RATE:.1f} ms | "
                     f"{r['inside_prod_range']} | {r['true_par']:.2f} | {r['secondary_within_32']} | "
                     f"**{r['verdict']}** | {v['H1']}/{v['H2']}/{v['H3']} of {v['n_blocks']} |\n")

    lines.append("\n## Per-case top-5 peaks (final EMA wide search)\n")
    for r in results:
        if "error" in r:
            continue
        lines.append(f"\n### {r['name']}\n\n")
        lines.append("| rank | lag (samp) | lag (ms) | inside_1024ms | height | PAR |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for i, (lag, h, par) in enumerate(r["final_top"]):
            inside = lag < PROD_MAX_DELAY_SAMPLES
            lines.append(f"| {i+1} | {lag} | {1000.0*lag/SAMPLE_RATE:.1f} | {inside} | {h:.5f} | {par:.2f} |\n")

    lines.append("\n## Per-case block trace\n")
    for r in results:
        if "error" in r:
            continue
        lines.append(f"\n### {r['name']}\n\n")
        lines.append("| blk | t(s) | prod_lag | prod_PAR | prod_conf | gate | wide_top1_lag | wide_top1_PAR | H1 | H2 | H3 |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for b in r["blocks"]:
            wt = b.top_peaks_wide[0] if b.top_peaks_wide else (0, 0.0, 0.0)
            lines.append(f"| {b.block_idx} | {b.t_start_s:.2f} | {b.prod_lag} | {b.prod_par:.2f} | "
                         f"{b.prod_confidence:.2f} | `{b.prod_gate}` | {wt[0]} | {wt[2]:.2f} | "
                         f"{int(b.H1)} | {int(b.H2)} | {int(b.H3)} |\n")

    # Cross-case pattern + decision tree
    lines.append("\n## Cross-case pattern\n")
    verdicts = [r.get("verdict", "?") for r in results if "error" not in r]
    counts = {k: verdicts.count(k) for k in set(verdicts)}
    lines.append(f"Verdict distribution across {len(verdicts)} cases: {counts}\n")
    true_lags_ms = [1000.0 * r["true_lag"] / SAMPLE_RATE for r in results if "error" not in r]
    if true_lags_ms:
        lines.append(f"\nTrue-lag range (ms): min={min(true_lags_ms):.1f}, "
                     f"max={max(true_lags_ms):.1f}, "
                     f"median={sorted(true_lags_ms)[len(true_lags_ms)//2]:.1f}\n")
    pct_out = sum(1 for r in results if "error" not in r and not r["inside_prod_range"]) / max(1, len(verdicts))
    lines.append(f"\nFraction with true_lag >= 1024 ms: {pct_out*100:.0f}%\n")

    lines.append("\n## Decision tree for next action\n")
    lines.append("- **If majority H1**: tighten peak selection. Options: require lag != 0 if dominant peak is at 0; require secondary peak within search range; raise par_low for `delay==0` lags.\n")
    lines.append("- **If majority H2**: evaluate FFT/CPU cost of pushing `max_delay_ms` to 2048 ms (seg_size 65536, ~4 s window — 2x current memory + FFT cost). Also consider a coarse pre-search at decimated sample rate.\n")
    lines.append("- **If majority H3**: propose per-band delay tracking or hold-off logic (don't accept ambiguous PAR; let filter converge instead).\n")
    lines.append("- **If mixed**: address H2 first (out-of-range can't be fixed by tuning); H1 and H3 can be addressed in subsequent passes.\n")

    with open(out_path, "w") as f:
        f.writelines(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/mingyu/Desktop/novatek/SE/AEC")
    ap.add_argument("--report", default="docs/research_log_7gt_alignment.md")
    args = ap.parse_args(argv)

    results: List[dict] = []
    for name, mic_rel, ref_rel in CASES:
        mic_path = os.path.join(args.root, mic_rel)
        ref_path = os.path.join(args.root, ref_rel)
        if not os.path.exists(mic_path) or not os.path.exists(ref_path):
            print(f"\n=== {name} ===\n  SKIP (missing): {mic_path if not os.path.exists(mic_path) else ref_path}", file=sys.stderr)
            results.append({"name": name, "error": "missing wav"})
            continue
        try:
            res = analyze_case(name, mic_path, ref_path, n_updates_total=0)
        except Exception as e:
            print(f"\n=== {name} ===\n  ERROR: {e}", file=sys.stderr)
            results.append({"name": name, "error": str(e)})
            continue
        print_case(res)
        results.append(res)

    out_path = os.path.join(args.root, args.report)
    write_report(results, out_path)
    print(f"\nReport written: {out_path}")


if __name__ == "__main__":
    main()
