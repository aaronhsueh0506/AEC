#!/usr/bin/env python3
"""End-to-end Python vs C parity driver for AEC v3.10.4 P0.

Runs the same 3 cases through Python AEC (BALANCED preset, CNG on)
and the C bin/aec_wav (BALANCED preset, --cng), then diffs the WAV
outputs sample-by-sample.

Reports peak |delta|, RMSE, and frame indices with |delta| > 1e-4.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
WAV_DIR = REPO / "wav" / "aec_challenge_blind"
C_BIN = REPO / "c_impl" / "bin" / "aec_wav"

CASES = [
    ("DT_7GTxy",
     WAV_DIR / "doubletalk" / "7GTxyTksSUqCnP5y0ILG4A_doubletalk_mic.wav",
     WAV_DIR / "doubletalk" / "7GTxyTksSUqCnP5y0ILG4A_doubletalk_lpb.wav"),
    ("DT_0I0XM",
     WAV_DIR / "doubletalk" / "0I0XMl3M0ECO0U1N0cJvpg_doubletalk_mic.wav",
     WAV_DIR / "doubletalk" / "0I0XMl3M0ECO0U1N0cJvpg_doubletalk_lpb.wav"),
    ("FS_mvmt_XTqo",
     WAV_DIR / "farend_singletalk" / "XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement_mic.wav",
     WAV_DIR / "farend_singletalk" / "XTqo1aOXDEiqyWTFK99I5Q_farend_singletalk_with_movement_lpb.wav"),
]


def run_python(mic_path: Path, ref_path: Path, out_path: Path):
    sys.path.insert(0, str(REPO / "python"))
    from aec import AecConfig, AecPreset, AEC

    mic, sr = sf.read(str(mic_path))
    ref, sr2 = sf.read(str(ref_path))
    assert sr == sr2, f"sr mismatch {sr} vs {sr2}"
    if mic.ndim > 1:
        mic = mic[:, 0]
    if ref.ndim > 1:
        ref = ref[:, 0]
    n = min(len(mic), len(ref))
    mic = mic[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)

    cfg = AecConfig.from_preset(AecPreset.BALANCED, sample_rate=sr,
                                enable_cng=True)
    aec = AEC(cfg)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    p = 0
    while p + hop <= n:
        out[p:p+hop] = aec.process(mic[p:p+hop], ref[p:p+hop])
        p += hop
    sf.write(str(out_path), out[:p], sr, subtype='FLOAT')
    return p, hop, sr


def run_c(mic_path: Path, ref_path: Path, out_path: Path):
    cmd = [str(C_BIN), str(mic_path), str(ref_path), str(out_path),
           "--preset", "balanced", "--cng"]
    # NOTE: aec_wav.c sets AEC_FP32_WAV (typo) but wav_io.h reads AEC_OUT_FLOAT.
    # Force float32 output here for parity-friendly comparison.
    env = dict(os.environ)
    env["AEC_OUT_FLOAT"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"C binary failed:\nstdout={res.stdout}\nstderr={res.stderr}")
    return res.stderr.strip()


def diff_wavs(py_path: Path, c_path: Path, hop: int):
    py, sr_p = sf.read(str(py_path))
    cw, sr_c = sf.read(str(c_path))
    assert sr_p == sr_c
    if py.ndim > 1:
        py = py[:, 0]
    if cw.ndim > 1:
        cw = cw[:, 0]
    n = min(len(py), len(cw))
    py = py[:n].astype(np.float64)
    cw = cw[:n].astype(np.float64)
    delta = py - cw
    peak = float(np.max(np.abs(delta))) if n else 0.0
    rmse = float(np.sqrt(np.mean(delta * delta))) if n else 0.0

    # frame-level
    nframes = n // hop
    bad = []
    if nframes > 0:
        d = delta[:nframes * hop].reshape(nframes, hop)
        frame_peak = np.max(np.abs(d), axis=1)
        bad = np.where(frame_peak > 1e-4)[0].tolist()
    return peak, rmse, len(bad), nframes, bad[:10]


def main():
    if not C_BIN.exists():
        print(f"ERROR: C binary not found: {C_BIN}", file=sys.stderr)
        sys.exit(1)

    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="aec_parity_"))
    print(f"# E2E Parity (Python vs C) — preset=balanced cng=on\n")
    print(f"tmpdir: {tmp}\n")

    for name, mic, ref in CASES:
        if not mic.exists() or not ref.exists():
            print(f"SKIP {name}: missing input")
            continue
        py_out = tmp / f"{name}_py.wav"
        c_out  = tmp / f"{name}_c.wav"

        n_py, hop, sr = run_python(mic, ref, py_out)
        c_log = run_c(mic, ref, c_out)
        peak, rmse, nbad, nframes, bad_head = diff_wavs(py_out, c_out, hop)
        rows.append((name, sr, hop, nframes, peak, rmse, nbad, bad_head))
        print(f"  {name}: peak={peak:.6e} rmse={rmse:.6e} bad_frames={nbad}/{nframes}")

    print("\n## Markdown table\n")
    print("| case | sr | hop | frames | peak |Δ| | RMSE | frames |Δ|>1e-4 | first bad frames |")
    print("|---|---|---|---|---|---|---|---|")
    for name, sr, hop, nframes, peak, rmse, nbad, bad_head in rows:
        print(f"| {name} | {sr} | {hop} | {nframes} | {peak:.3e} | {rmse:.3e} | {nbad} | {bad_head} |")

    # Verdict
    overall_peak = max((r[4] for r in rows), default=0.0)
    if overall_peak == 0.0:
        verdict = "BIT-EXACT"
    elif overall_peak < 1e-4:
        verdict = f"TOLERANT (peak={overall_peak:.2e})"
    else:
        verdict = f"DIVERGENT (peak={overall_peak:.2e})"
    print(f"\n**Verdict: {verdict}**")
    return rows, verdict


if __name__ == "__main__":
    main()
