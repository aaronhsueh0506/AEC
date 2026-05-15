"""v3.16 C4 — noise_floor / CNG interaction audit-first (Phase 2, 2026-05-15).

Tests §1.1 H4: "compression downstream of RES — noise_floor / CNG
over-aggressive on DT-NE residual."

Per-frame trace of:
  - `_stats_last_noise_floor_gain` (mean noise_floor lift on voice band)
  - `_stats_last_nfl_lifted` (boolean: did noise_floor actively raise gain?)
  - `_stats_last_noise_psd` (running noise tracker level)
  - mean CNG `_smooth_cn_gain` (additive comfort noise envelope)
  - res_gain_db, dt_active, cohort_tail_T, filter_state

Verdict gate (per §0.4):

  H4 SUPPORTED: noise_floor_lifted ≥ 60% on DT_active frames AND
    noise_floor_gain_mean ≥ res_gain_lin × 1.2 on those frames AND
    res_gain p5 < -10 dB on lifted DT frames.
    → OPEN C4 mechanism arc.

  H4 REFUTED: noise_floor_lifted < 30% on DT_active frames OR
    no clear correlation between lift and compression.
    → CLOSE C4 audit per §0.4.

  MIXED: between thresholds. Refine.

Usage:
    python3 tools/research/v3_16_c4_noise_floor_cng_audit.py \\
        --cases tools/research/v3_15_subset_cases.txt \\
        --dataset wav/aec_challenge_blind \\
        --out /tmp/v3_16_c4_audit/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from aec import AEC, AecConfig, AecMode, AecPreset  # noqa: E402


SCENARIO_FOLDER = {
    "_doubletalk_with_movement": "doubletalk",
    "_farend_singletalk_with_movement": "farend_singletalk",
    "_nearend_singletalk": "nearend_singletalk",
    "_farend_singletalk": "farend_singletalk",
    "_doubletalk": "doubletalk",
}

SCENARIO_SUFFIXES = (
    "_doubletalk_with_movement",
    "_farend_singletalk_with_movement",
    "_nearend_singletalk",
    "_farend_singletalk",
    "_doubletalk",
)


def parse_stem(stem: str) -> str:
    for s in SCENARIO_SUFFIXES:
        if stem.endswith(s):
            return SCENARIO_FOLDER[s]
    raise ValueError(f"unrecognized stem: {stem}")


def bucket_of(stem: str) -> str:
    if stem.endswith("_with_movement"):
        return "DT_movement" if "doubletalk" in stem else "FS_movement"
    if "doubletalk" in stem:
        return "DT_static"
    if "nearend_singletalk" in stem:
        return "NE"
    if "farend_singletalk" in stem:
        return "FS_static"
    return "UNKNOWN"


def run_one(stem: str, dataset_dir: Path, sample_rate: int = 16000) -> dict:
    folder = parse_stem(stem)
    mic_path = dataset_dir / folder / f"{stem}_mic.wav"
    ref_path = dataset_dir / folder / f"{stem}_lpb.wav"
    if not mic_path.exists() or not ref_path.exists():
        raise FileNotFoundError(f"{mic_path} / {ref_path} missing")

    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sample_rate,
        mode=AecMode.PBFDKF,
        filter_length=832,
        enable_res=True,
        enable_cng=True,
        enable_shadow=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)
    # noise_floor / CNG _stats_last_* fields only updated when
    # ResFilter._stats is not None (opt-in via enable_stats()).
    if aec.res is not None:
        aec.res.enable_stats()

    mic = np.asarray(sf.read(mic_path)[0], dtype=np.float32)
    lpb = np.asarray(sf.read(ref_path)[0], dtype=np.float32)
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    hop = aec.hop_size
    frames = []
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        s = aec.get_stats()
        res = aec.res
        if res is None:
            pos += hop
            continue
        cn_gain_mean = float(np.mean(res._smooth_cn_gain)) if hasattr(
            res, "_smooth_cn_gain") else 0.0
        nfl_gain = float(getattr(res, "_stats_last_noise_floor_gain", 0.0))
        nfl_lifted = bool(getattr(res, "_stats_last_nfl_lifted", False))
        noise_psd = float(getattr(res, "_stats_last_noise_psd", 0.0))
        # Convert res_gain_db to linear for direct compare with noise_floor_gain
        # (which is linear amplitude)
        res_gain_lin = 10.0 ** (s.res_gain_mean_db / 20.0)
        # BALANCED has enable_dtd=False so dt_active / dt_confidence are 0.
        # Use dt_from_energy > 0.3 as proxy for DT-active (always-on signal).
        dt_e = float(s.dt_from_energy)
        dt_active_proxy = int(dt_e > 0.3)
        frames.append({
            "frame": int(s.frame_count),
            "time_s": float(s.time_s),
            "res_gain_db": float(s.res_gain_mean_db),
            "res_gain_lin": float(res_gain_lin),
            "noise_floor_gain": nfl_gain,
            "noise_floor_lifted": int(nfl_lifted),
            "noise_psd": noise_psd,
            "cng_smooth_gain": cn_gain_mean,
            "dt_active": dt_active_proxy,
            "dt_from_energy": dt_e,
            "dt_conf": float(s.dt_confidence),
            "filter_converged": int(s.filter_converged),
            "epc_active": int(s.epc_active),
            "cohort_tail_T": int(s.cohort_tail_T),
            "far_db": float(s.far_power_db),
            "mic_db": float(s.mic_power_db),
            "err_db": float(s.error_power_db),
        })
        pos += hop

    return {
        "stem": stem,
        "bucket": bucket_of(stem),
        "n_frames": len(frames),
        "frames": frames,
    }


def analyze(results: list[dict]) -> dict:
    """Per-bucket noise_floor / CNG diagnostics."""
    by_bucket = {}
    for r in results:
        by_bucket.setdefault(r["bucket"], []).append(r)

    summary = {}
    for bk, cases in by_bucket.items():
        all_frames = [fr for r in cases for fr in r["frames"]]
        n = max(1, len(all_frames))
        # Slice 1: all frames
        nfl_lift_pct = sum(1 for f in all_frames if f["noise_floor_lifted"]) / n
        nfl_gain_mean = float(np.mean([f["noise_floor_gain"] for f in all_frames]))
        cng_gain_mean = float(np.mean([f["cng_smooth_gain"] for f in all_frames]))
        # Slice 2: dt_active frames (the H4 target slice)
        dt_frames = [f for f in all_frames if f["dt_active"]]
        if dt_frames:
            dt_n = len(dt_frames)
            dt_nfl_lift = sum(1 for f in dt_frames if f["noise_floor_lifted"]) / dt_n
            dt_nfl_gain = float(np.mean([f["noise_floor_gain"] for f in dt_frames]))
            dt_res_gain_lin = float(np.mean([f["res_gain_lin"] for f in dt_frames]))
            dt_res_gain_p5_db = float(np.percentile(
                [f["res_gain_db"] for f in dt_frames], 5))
            # Lift fires AND res_gain low ("compression" combo)
            lifted_compressed = [f for f in dt_frames
                                 if f["noise_floor_lifted"] and f["res_gain_db"] < -10]
            dt_lift_AND_low = len(lifted_compressed) / dt_n
            # noise_floor_gain dominates when nfl_gain > res_gain_lin
            nfl_dominates = sum(1 for f in dt_frames
                                if f["noise_floor_gain"] > f["res_gain_lin"] * 1.2)
            dt_nfl_dominates = nfl_dominates / dt_n
        else:
            dt_n = 0
            dt_nfl_lift = 0.0
            dt_nfl_gain = 0.0
            dt_res_gain_lin = 0.0
            dt_res_gain_p5_db = 0.0
            dt_lift_AND_low = 0.0
            dt_nfl_dominates = 0.0

        summary[bk] = {
            "n_cases": len(cases),
            "n_frames": n,
            "nfl_lift_pct_all": nfl_lift_pct,
            "nfl_gain_mean_all": nfl_gain_mean,
            "cng_gain_mean_all": cng_gain_mean,
            "dt_n_frames": dt_n,
            "dt_nfl_lift_pct": dt_nfl_lift,
            "dt_nfl_gain_mean": dt_nfl_gain,
            "dt_res_gain_lin_mean": dt_res_gain_lin,
            "dt_res_gain_p5_db": dt_res_gain_p5_db,
            "dt_lift_AND_lowgain_pct": dt_lift_AND_low,
            "dt_nfl_dominates_pct": dt_nfl_dominates,
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stems = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.append(line)
    print(f"loaded {len(stems)} stems")

    results = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        try:
            r = run_one(stem, args.dataset)
        except Exception as e:
            print(f"  [{i}/{len(stems)}] {stem} FAILED: {e}")
            continue
        out_json = args.out / "per_case" / f"{stem}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(r, f, separators=(",", ":"))
        if i % 10 == 0 or i <= 3 or i == len(stems):
            print(f"  [{i}/{len(stems)}] {stem} {r['n_frames']} frames")
        results.append(r)

    agg = analyze(results)
    with open(args.out / "summary.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Print table
    print()
    print(f"=== noise_floor / CNG audit summary ({len(results)} cases) ===")
    print()
    print(f"{'bucket':<14} {'n':>3} {'frames':>8} "
          f"{'nfl_lift_all':>13} {'nfl_g_all':>10} {'cng_g_all':>10}")
    for bk in sorted(agg):
        b = agg[bk]
        print(f"{bk:<14} {b['n_cases']:>3} {b['n_frames']:>8} "
              f"{b['nfl_lift_pct_all']*100:>12.1f}% "
              f"{b['nfl_gain_mean_all']:>10.4f} "
              f"{b['cng_gain_mean_all']:>10.4f}")
    print()
    print(f"=== H4 target slice: dt_active frames ===")
    print()
    print(f"{'bucket':<14} {'dt_frm':>7} {'dt_nfl_lift':>12} "
          f"{'dt_nfl_g':>10} {'dt_res_g_lin':>13} "
          f"{'dt_lift+lowg':>13} {'nfl_dom_p':>11} "
          f"{'res_p5_db':>10}")
    for bk in sorted(agg):
        b = agg[bk]
        print(f"{bk:<14} {b['dt_n_frames']:>7} "
              f"{b['dt_nfl_lift_pct']*100:>11.1f}% "
              f"{b['dt_nfl_gain_mean']:>10.4f} "
              f"{b['dt_res_gain_lin_mean']:>13.4f} "
              f"{b['dt_lift_AND_lowgain_pct']*100:>12.1f}% "
              f"{b['dt_nfl_dominates_pct']*100:>10.1f}% "
              f"{b['dt_res_gain_p5_db']:>+9.2f}dB")

    # H4 verdict per DT bucket
    print()
    print("--- H4 verdict gate (per bucket) ---")
    dt_buckets = ["DT_static", "DT_movement"]
    for bk in dt_buckets:
        if bk not in agg:
            continue
        b = agg[bk]
        nfl_lift = b["dt_nfl_lift_pct"]
        nfl_dom = b["dt_nfl_dominates_pct"]
        res_p5 = b["dt_res_gain_p5_db"]
        if nfl_lift >= 0.60 and nfl_dom >= 0.30 and res_p5 < -10.0:
            v = "H4 SUPPORTED"
        elif nfl_lift < 0.30:
            v = "H4 REFUTED"
        else:
            v = "MIXED"
        print(f"  {bk}: lift={nfl_lift*100:.0f}% nfl_dom={nfl_dom*100:.0f}% "
              f"res_p5={res_p5:.1f}dB → {v}")

    # Overall
    overall_lift = float(np.mean([
        agg[bk]["dt_nfl_lift_pct"] for bk in dt_buckets if bk in agg
    ]))
    overall_dom = float(np.mean([
        agg[bk]["dt_nfl_dominates_pct"] for bk in dt_buckets if bk in agg
    ]))
    print()
    if overall_lift >= 0.60 and overall_dom >= 0.30:
        overall = "H4 SUPPORTED — OPEN C4 mechanism arc"
    elif overall_lift < 0.30:
        overall = "H4 REFUTED — CLOSE C4 audit per §0.4"
    else:
        overall = "MIXED — refine before commit"
    print(f"OVERALL VERDICT: {overall}")

    print(f"\ntotal wall time {time.time() - t0:.1f}s")
    print(f"summary.json -> {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
