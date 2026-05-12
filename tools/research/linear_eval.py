"""Linear-only quality evaluation for v3.11 Phase 0 / Phase 1 verification.

Computes per-case + per-bucket linear-stage metrics from `_ours_nores.wav`
outputs (PBFDKF only, RES bypassed):

- ERLE_full  = 10*log10(mean(mic^2) / mean(nores^2))      — overall echo cancel
- ERLE_active = same but on frames where far_pwr > -55 dBFS — active-far ERLE
- NE_preservation = 10*log10(mean(mic^2) / mean((mic-nores)^2)) — distortion proxy
- Active fraction = fraction of frames where far is active

These metrics evaluate the PBFDKF linear stage *without* RES masking.
Compare two runs (e.g., v3.10.6 baseline vs Sprint-N candidate) by diffing
the per-bucket aggregates; Phase 1 hard bars are bucket mean ERLE delta.

Usage:
    python3 tools/research/linear_eval.py \
        --rendered-dir results/v3_10_5_main \
        --input-dir wav/aec_challenge_blind \
        -o results/v3_10_5_main/linear_baseline.json

Compare two runs:
    python3 tools/research/linear_eval.py \
        --rendered-dir results/<candidate>/ \
        --input-dir wav/aec_challenge_blind \
        --baseline results/v3_10_5_main/linear_baseline.json \
        -o results/<candidate>/linear_delta.json
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf


# Bucket suffix → canonical bucket name (matches bench_aecmos.py convention)
_BUCKET_RULES = (
    ("_farend_singletalk_with_movement", "FS_movement"),
    ("_farend_singletalk", "FS_static"),
    ("_doubletalk_with_movement", "DT_movement"),
    ("_doubletalk", "DT_static"),
    ("_nearend_singletalk", "NE"),
)

# Frame-active threshold for ERLE_active: far-end frame power above -55 dBFS
_ACTIVE_FRAME_DB = -55.0
_FRAME_LEN = 320  # 20 ms at 16 kHz
_EPS = 1e-12


@dataclass
class CaseMetrics:
    stem: str
    bucket: str
    erle_full_db: float
    erle_active_db: float
    ne_preservation_db: float
    active_fraction: float


def _bucket_of(stem: str) -> str:
    for suffix, name in _BUCKET_RULES:
        if stem.endswith(suffix):
            return name
    return "unknown"


def _input_dir_for_bucket(bucket: str) -> str:
    if bucket in ("FS_static", "FS_movement"):
        return "farend_singletalk"
    if bucket in ("DT_static", "DT_movement"):
        return "doubletalk"
    if bucket == "NE":
        return "nearend_singletalk"
    raise ValueError(f"unknown bucket: {bucket}")


def _frame_pwr(x: np.ndarray, frame: int = _FRAME_LEN) -> np.ndarray:
    """Frame-wise mean square (energy proxy)."""
    n_frames = len(x) // frame
    if n_frames == 0:
        return np.array([], dtype=np.float32)
    trimmed = x[: n_frames * frame].reshape(n_frames, frame)
    return np.mean(trimmed * trimmed, axis=1).astype(np.float32)


def _safe_db(num: float, den: float) -> float:
    if den < _EPS:
        return float("inf") if num >= _EPS else 0.0
    if num < _EPS:
        return -float("inf")
    return float(10.0 * np.log10(num / den))


def _eval_case(args: tuple) -> CaseMetrics | None:
    stem, rendered_dir, input_dir = args
    bucket = _bucket_of(stem)
    if bucket == "unknown":
        return None

    nores_path = os.path.join(rendered_dir, f"{stem}_ours_nores.wav")
    if not os.path.exists(nores_path):
        return None

    bucket_dir = _input_dir_for_bucket(bucket)
    mic_path = os.path.join(input_dir, bucket_dir, f"{stem}_mic.wav")
    lpb_path = os.path.join(input_dir, bucket_dir, f"{stem}_lpb.wav")
    if not os.path.exists(mic_path) or not os.path.exists(lpb_path):
        return None

    mic, _ = sf.read(mic_path, dtype="float32")
    lpb, _ = sf.read(lpb_path, dtype="float32")
    nores, _ = sf.read(nores_path, dtype="float32")

    n = min(len(mic), len(lpb), len(nores))
    mic = mic[:n]
    lpb = lpb[:n]
    nores = nores[:n]

    mic_pwr = float(np.mean(mic * mic))
    nores_pwr = float(np.mean(nores * nores))
    diff = mic - nores
    diff_pwr = float(np.mean(diff * diff))

    # Frame-wise active mask (far is active)
    lpb_frame = _frame_pwr(lpb)
    mic_frame = _frame_pwr(mic)
    nores_frame = _frame_pwr(nores)
    active_thr = 10.0 ** (_ACTIVE_FRAME_DB / 10.0)
    active = lpb_frame > active_thr
    active_fraction = float(active.mean()) if active.size else 0.0

    if active.any():
        mic_active = float(mic_frame[active].mean())
        nores_active = float(nores_frame[active].mean())
        erle_active_db = _safe_db(mic_active, nores_active)
    else:
        erle_active_db = 0.0

    return CaseMetrics(
        stem=stem,
        bucket=bucket,
        erle_full_db=_safe_db(mic_pwr, nores_pwr),
        erle_active_db=erle_active_db,
        ne_preservation_db=_safe_db(mic_pwr, diff_pwr),
        active_fraction=active_fraction,
    )


def _collect_stems(rendered_dir: str) -> list[str]:
    stems = []
    for entry in os.listdir(rendered_dir):
        if entry.endswith("_ours_nores.wav"):
            stems.append(entry[: -len("_ours_nores.wav")])
    return sorted(stems)


def _summarize(cases: list[CaseMetrics]) -> dict:
    by_bucket: dict[str, list[CaseMetrics]] = {}
    for c in cases:
        by_bucket.setdefault(c.bucket, []).append(c)

    summary = {}
    for bucket, lst in sorted(by_bucket.items()):
        erle_full = np.array([c.erle_full_db for c in lst if np.isfinite(c.erle_full_db)])
        erle_active = np.array([c.erle_active_db for c in lst if np.isfinite(c.erle_active_db)])
        ne_pres = np.array([c.ne_preservation_db for c in lst if np.isfinite(c.ne_preservation_db)])
        summary[bucket] = {
            "n": len(lst),
            "erle_full_mean": float(erle_full.mean()) if erle_full.size else 0.0,
            "erle_full_median": float(np.median(erle_full)) if erle_full.size else 0.0,
            "erle_full_p10": float(np.percentile(erle_full, 10)) if erle_full.size else 0.0,
            "erle_active_mean": float(erle_active.mean()) if erle_active.size else 0.0,
            "erle_active_median": float(np.median(erle_active)) if erle_active.size else 0.0,
            "erle_active_p10": float(np.percentile(erle_active, 10)) if erle_active.size else 0.0,
            "ne_preservation_mean": float(ne_pres.mean()) if ne_pres.size else 0.0,
            "ne_preservation_median": float(np.median(ne_pres)) if ne_pres.size else 0.0,
            "ne_preservation_p10": float(np.percentile(ne_pres, 10)) if ne_pres.size else 0.0,
        }
    return summary


def _diff_summary(curr: dict, baseline: dict) -> dict:
    deltas = {}
    for bucket in sorted(set(curr) | set(baseline)):
        cb = curr.get(bucket, {})
        bb = baseline.get(bucket, {})
        deltas[bucket] = {
            "d_erle_full_mean": cb.get("erle_full_mean", 0.0) - bb.get("erle_full_mean", 0.0),
            "d_erle_active_mean": cb.get("erle_active_mean", 0.0) - bb.get("erle_active_mean", 0.0),
            "d_ne_preservation_mean": cb.get("ne_preservation_mean", 0.0) - bb.get("ne_preservation_mean", 0.0),
            "n_curr": cb.get("n", 0),
            "n_base": bb.get("n", 0),
        }
    return deltas


def main() -> int:
    p = argparse.ArgumentParser(description="Linear-only quality eval (PBFDKF linear residual)")
    p.add_argument("--rendered-dir", required=True, help="Dir containing *_ours_nores.wav")
    p.add_argument("--input-dir", required=True, help="Root of wav/aec_challenge_blind/")
    p.add_argument("-o", "--output", required=True, help="Output JSON path")
    p.add_argument("--baseline", default=None, help="Baseline JSON for delta comparison")
    p.add_argument("--label", default=None, help="Label for this run (default: rendered-dir basename)")
    p.add_argument("-j", "--jobs", type=int, default=4)
    args = p.parse_args()

    label = args.label or os.path.basename(os.path.normpath(args.rendered_dir))
    stems = _collect_stems(args.rendered_dir)
    print(f"Found {len(stems)} cases in {args.rendered_dir}")

    work = [(stem, args.rendered_dir, args.input_dir) for stem in stems]
    cases: list[CaseMetrics] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as exe:
        futs = [exe.submit(_eval_case, w) for w in work]
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            if result is not None:
                cases.append(result)
            if i % 100 == 0:
                print(f"  processed {i}/{len(work)}")

    summary = _summarize(cases)

    out = {
        "label": label,
        "n_cases": len(cases),
        "summary": summary,
        "scores": {c.stem: asdict(c) for c in cases},
    }

    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_summary = baseline_data.get("summary", {})
        out["delta_vs_baseline"] = _diff_summary(summary, baseline_summary)
        out["baseline_label"] = baseline_data.get("label", "<unknown>")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Print bucket summary table
    print(f"\n=== {label} linear eval summary ===")
    print(f"{'bucket':<15s} {'n':>4s} {'erle_full':>10s} {'erle_active':>11s} {'ne_pres':>9s}")
    for bucket in sorted(summary):
        s = summary[bucket]
        print(f"{bucket:<15s} {s['n']:>4d} "
              f"{s['erle_full_mean']:>9.2f}  "
              f"{s['erle_active_mean']:>10.2f}  "
              f"{s['ne_preservation_mean']:>8.2f}")

    if "delta_vs_baseline" in out:
        print(f"\n=== Δ vs {out['baseline_label']} ===")
        print(f"{'bucket':<15s} {'Δerle_full':>11s} {'Δerle_active':>13s} {'Δne_pres':>10s}")
        for bucket in sorted(out["delta_vs_baseline"]):
            d = out["delta_vs_baseline"][bucket]
            print(f"{bucket:<15s} {d['d_erle_full_mean']:>+10.3f}  "
                  f"{d['d_erle_active_mean']:>+12.3f}  "
                  f"{d['d_ne_preservation_mean']:>+9.3f}")

    print(f"\nSaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
