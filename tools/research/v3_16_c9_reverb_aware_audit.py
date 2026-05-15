"""v3.16 C9 — Reverb-aware RES override audit-first (Phase 4, 2026-05-15).

Per-frame mic-lpb Pearson cross-correlation (delay-aligned) + AecStats
context. Output gates whether C9 mechanism arc opens:

  TRIGGER CLEAR: pcb1N has persistent r < 0.15 + far_power > thresh in
  > 50% of voiced far frames; healthy FS_static / NE controls have
  r > 0.30 in > 80% of voiced far frames.
  → OPEN C9 mechanism wire (RES override).

  AMBIGUOUS: pcb1N r < 0.15 in 25-50% of frames OR controls fire FP.
  → Refine trigger before commit.

  TRIGGER ABSENT: pcb1N r < 0.15 < 25% of frames OR no separation
  from controls. → CLOSE C9 audit.

Usage:
    python3 tools/research/v3_16_c9_reverb_aware_audit.py \\
        --cases tools/research/v3_16_c6_tier_a_cases.txt \\
        --dataset wav/aec_challenge_blind \\
        --out /tmp/v3_16_c9_audit/
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


# Pearson r threshold for C9 trigger.
R_LOW_THRESHOLD = 0.15
R_HIGH_THRESHOLD = 0.30
# Far power threshold (dB) for "active far" frame.
FAR_POWER_DB_THRESHOLD = -50.0


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Plain Pearson correlation. Returns 0 on degenerate inputs."""
    if a.size < 16 or b.size < 16:
        return 0.0
    sa = float(a.std())
    sb = float(b.std())
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


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
        trace_delay_est=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)

    mic = np.asarray(sf.read(mic_path)[0], dtype=np.float32)
    lpb = np.asarray(sf.read(ref_path)[0], dtype=np.float32)
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    hop = aec.hop_size  # 160 @ 16k = 10 ms
    # Mic-lpb r window: 320 samples = 20 ms (covers 2 hops of context)
    R_WINDOW = 320

    frames = []
    pos = 0
    while pos + hop <= n:
        # Run pipeline (advances internal state including DelayEst)
        aec.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        s = aec.get_stats()
        d = aec._diag

        # Compute per-frame Pearson r between mic and delay-aligned lpb.
        # lpb is "ahead" of mic by `delay_samples` (typical AEC echo path).
        # So mic[pos:pos+W] should correlate with lpb[pos-delay:pos-delay+W].
        delay = int(s.delay_samples)
        w0 = pos
        w1 = pos + R_WINDOW
        # Source for mic: current window
        if w1 <= n:
            mic_w = mic[w0:w1]
        else:
            pos += hop
            continue
        # Source for lpb: aligned by delay
        l0 = max(0, w0 - delay)
        l1 = l0 + R_WINDOW
        if l1 > n or delay <= 0:
            r = 0.0
        else:
            lpb_w = lpb[l0:l1]
            r = pearson_r(mic_w, lpb_w)

        frames.append({
            "frame": int(s.frame_count),
            "time_s": float(s.time_s),
            "mic_db": float(s.mic_power_db),
            "far_db": float(s.far_power_db),
            "err_db": float(s.error_power_db),
            "delay_samples": delay,
            "delay_ms": float(s.delay_ms),
            "filter_converged": int(s.filter_converged),
            "epc_active": int(s.epc_active),
            "cohort_tail_T": int(s.cohort_tail_T),
            "dt_active": int(s.dt_active),
            "dt_from_energy": float(s.dt_from_energy),
            "dt_from_coherence": float(s.dt_from_coherence),
            "res_gain_db": float(s.res_gain_mean_db),
            "using_render": int(s.res_using_render),
            "erle_inst_db": float(s.erle_inst_db),
            "mic_lpb_r": float(r),
        })
        pos += hop

    # Determine bucket via stem suffix
    if stem.endswith("_with_movement"):
        bucket = "DT_movement" if "doubletalk" in stem else "FS_movement"
    elif "doubletalk" in stem:
        bucket = "DT_static"
    elif "nearend_singletalk" in stem:
        bucket = "NE"
    elif "farend_singletalk" in stem:
        bucket = "FS_static"
    else:
        bucket = "UNKNOWN"

    return {
        "stem": stem,
        "bucket": bucket,
        "sample_rate": sample_rate,
        "hop_size": hop,
        "n_frames": len(frames),
        "duration_s": n / sample_rate,
        "frames": frames,
    }


def analyze(results: list[dict]) -> dict:
    """Compute per-case + per-bucket stats."""
    per_case = []
    for r in results:
        frames = r["frames"]
        n = max(1, len(frames))
        # Active-far frames only (far_db > threshold)
        active = [f for f in frames if f["far_db"] > FAR_POWER_DB_THRESHOLD]
        n_act = max(1, len(active))
        rs = np.array([f["mic_lpb_r"] for f in active])
        delay_samps = np.array([f["delay_samples"] for f in frames])
        coh = np.array([f["dt_from_coherence"] for f in frames])
        res_gains = np.array([f["res_gain_db"] for f in frames])
        case = {
            "stem": r["stem"],
            "bucket": r["bucket"],
            "n_frames": n,
            "n_active_far": len(active),
            "active_far_pct": len(active) / n,
            "r_mean": float(rs.mean()) if len(rs) else 0.0,
            "r_median": float(np.median(rs)) if len(rs) else 0.0,
            "r_p10": float(np.percentile(rs, 10)) if len(rs) else 0.0,
            "r_p90": float(np.percentile(rs, 90)) if len(rs) else 0.0,
            "low_r_active_pct": (
                float(np.mean(rs < R_LOW_THRESHOLD)) if len(rs) else 0.0
            ),
            "high_r_active_pct": (
                float(np.mean(rs > R_HIGH_THRESHOLD)) if len(rs) else 0.0
            ),
            "delay_p50_samp": int(np.percentile(delay_samps, 50)) if len(delay_samps) else 0,
            "delay_p95_samp": int(np.percentile(delay_samps, 95)) if len(delay_samps) else 0,
            "delay_gt_0p8_fl_pct": float(np.mean(delay_samps > 0.8 * 832)),
            "coherence_mean": float(coh.mean()) if len(coh) else 0.0,
            "coherence_low_pct": float(np.mean(coh < 0.3)) if len(coh) else 0.0,
            "res_gain_mean_db": float(res_gains.mean()),
            "res_gain_p5_db": float(np.percentile(res_gains, 5)),
        }
        per_case.append(case)

    # Per-bucket aggregation
    by_bucket = {}
    for c in per_case:
        by_bucket.setdefault(c["bucket"], []).append(c)
    bucket_summary = {}
    for bk, cases in by_bucket.items():
        bucket_summary[bk] = {
            "n_cases": len(cases),
            "low_r_pct_mean": float(np.mean([c["low_r_active_pct"] for c in cases])),
            "low_r_pct_median": float(np.median([c["low_r_active_pct"] for c in cases])),
            "delay_gt_0p8fl_pct_mean": float(np.mean([c["delay_gt_0p8_fl_pct"] for c in cases])),
            "r_mean_mean": float(np.mean([c["r_mean"] for c in cases])),
        }
    return {"per_case": per_case, "per_bucket": bucket_summary}


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
        print(f"  [{i}/{len(stems)}] {stem} {r['n_frames']} frames")
        results.append(r)

    agg = analyze(results)
    with open(args.out / "analysis.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Print per-case table
    print()
    print(f"{'stem':<55} {'bucket':<14} "
          f"{'r_mean':>7} {'r_p10':>7} {'low_r':>7} "
          f"{'d_p50':>6} {'d>0.8fl':>8} {'coh_mn':>7} {'res_gp5':>9}")
    for c in agg["per_case"]:
        print(f"{c['stem']:<55} {c['bucket']:<14} "
              f"{c['r_mean']:>+7.3f} {c['r_p10']:>+7.3f} "
              f"{c['low_r_active_pct']*100:>6.1f}% "
              f"{c['delay_p50_samp']:>6} "
              f"{c['delay_gt_0p8_fl_pct']*100:>7.1f}% "
              f"{c['coherence_mean']:>+7.3f} "
              f"{c['res_gain_p5_db']:>+8.2f}dB")

    # Per-bucket
    print()
    print(f"{'bucket':<15} {'n':>3} {'low_r_pct_mean':>15} "
          f"{'r_mean_mean':>13} {'d>0.8fl_pct_mean':>17}")
    for bk, b in agg["per_bucket"].items():
        print(f"{bk:<15} {b['n_cases']:>3} "
              f"{b['low_r_pct_mean']*100:>14.1f}% "
              f"{b['r_mean_mean']:>+12.3f} "
              f"{b['delay_gt_0p8fl_pct_mean']*100:>16.1f}%")

    # Verdict
    print()
    print(f"--- C9 trigger verdict gate ---")
    pcb1n = next((c for c in agg["per_case"] if c["stem"].startswith("pcb1N")), None)
    if pcb1n:
        ctrl_cases = [c for c in agg["per_case"]
                      if c["bucket"] in ("FS_static", "NE")
                      and not c["stem"].startswith("pcb1N")]
        ctrl_low_r_mean = (
            float(np.mean([c["low_r_active_pct"] for c in ctrl_cases]))
            if ctrl_cases else 0.0
        )
        print(f"  pcb1N low-r-active rate: {pcb1n['low_r_active_pct']*100:.1f}%")
        print(f"  control (FS_static + NE, ex pcb1N) low-r mean: {ctrl_low_r_mean*100:.1f}%")
        if pcb1n["low_r_active_pct"] >= 0.50 and ctrl_low_r_mean <= 0.20:
            verdict = "TRIGGER CLEAR — OPEN C9 mechanism wire"
        elif pcb1n["low_r_active_pct"] < 0.25:
            verdict = "TRIGGER ABSENT — CLOSE C9 audit"
        else:
            verdict = "AMBIGUOUS — refine trigger before commit"
        print(f"  VERDICT: {verdict}")
    else:
        print(f"  pcb1N not in case list — cannot adjudicate")

    print(f"\ntotal wall time {time.time() - t0:.1f}s")
    print(f"analysis.json -> {args.out / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
