"""v3.16 C6 — adjudicate H1 (delay upstream) vs H2 (mechanism walls).

Ingests per-case JSONs from `v3_16_c6_delay_est_audit.py` and emits:
  1. Per-case timeline alignment: bad-frame windows (high residual_linear /
     epc / cohort_tail_T / divergence) vs DelayEst state in preceding
     window.
  2. Lead-lag attribution: fraction of bad-frame bursts with a preceding
     DelayEst issue within 250 ms.
  3. Per-case H1 / H2 / mixed verdict.

Usage:

    python3 tools/research/v3_16_c6_analyze.py /tmp/v3_16_c6_audit/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


# Thresholds (per design doc §3.4)
PAR_LOW = 5.0
PAR_SOLID = 8.0
TOP_RATIO_AMBIGUOUS = 0.5
DELAY_JUMP_SAMPLES = 50
RESIDUAL_PCT_BAD = 90       # frames in top 10% of residual_psd_linear
DIVERGENCE_BAD = 0.3
LEAD_WINDOW_MS = 250
H1_STRONG_PCT = 0.60
H2_STRONG_PCT = 0.30


def detect_delay_issues(delay_trace: list[dict],
                        hop_size: int,
                        sample_rate: int) -> np.ndarray:
    """Return one bool per frame: True if DelayEst is issuing.

    Delay-issue indicators:
      - PAR < PAR_LOW (5.0)
      - top1/top2 ambiguous (top2_par/top1_par > 0.5)
      - delay jump > 50 samples between consecutive estimates
      - did_estimate=False streak > 30 frames AFTER init_done
    """
    n = len(delay_trace)
    issues = np.zeros(n, dtype=bool)
    prev_est_delay = -1
    no_est_streak = 0
    init_done_seen = False

    for i, t in enumerate(delay_trace):
        cur_par = float(t.get("last_par", 0.0))
        cur_top1 = float(t.get("top1_par", 0.0))
        cur_top2 = float(t.get("top2_par", 0.0))
        cur_init = bool(t.get("init_done", False))
        cur_est_now = bool(t.get("did_estimate", False))
        cur_estimated = int(t.get("estimated_delay", -1))

        if cur_init:
            init_done_seen = True

        # 1. PAR threshold
        if cur_par > 0 and cur_par < PAR_LOW:
            issues[i] = True
        # 2. multi-peak ambiguity
        if cur_top1 > 1e-6 and (cur_top2 / cur_top1) > TOP_RATIO_AMBIGUOUS:
            issues[i] = True
        # 3. delay jump (only if estimate actually changed this frame)
        if cur_est_now and prev_est_delay > 0 and cur_estimated > 0:
            if abs(cur_estimated - prev_est_delay) > DELAY_JUMP_SAMPLES:
                issues[i] = True
        # 4. did_estimate streak after init_done
        if init_done_seen and not cur_est_now:
            no_est_streak += 1
            if no_est_streak > 200:  # 200 frames = ~2 s @ 16k/160hop
                issues[i] = True
        else:
            no_est_streak = 0
        if cur_est_now:
            prev_est_delay = cur_estimated

    return issues


def detect_filter_issues(frames: list[dict]) -> np.ndarray:
    """Return one bool per frame: True if filter is struggling.

    Indicators (any true):
      - cohort_tail_T fires
      - epc_active fires
      - divergence > 0.3
      - residual_psd_linear in top 10% of case
      - shadow_advantage > 1.5
      - erle_inst_db < -3 (filter making it worse)
    """
    n = len(frames)
    issues = np.zeros(n, dtype=bool)
    residuals = np.array([fr.get("residual_psd_linear", 0.0) for fr in frames])
    res_thr = float(np.percentile(residuals, RESIDUAL_PCT_BAD))
    for i, fr in enumerate(frames):
        if fr["cohort_tail_T"]:
            issues[i] = True
        if fr["epc_active"]:
            issues[i] = True
        if fr["divergence"] > DIVERGENCE_BAD:
            issues[i] = True
        if fr.get("residual_psd_linear", 0.0) > res_thr and res_thr > 0:
            issues[i] = True
        if fr["shadow_advantage"] > 1.5:
            issues[i] = True
        if fr["erle_inst_db"] < -3.0:
            issues[i] = True
    return issues


def burst_starts(mask: np.ndarray, min_gap: int = 10) -> np.ndarray:
    """Find rising-edge frame indices (start of each fire burst).

    Consecutive runs separated by less than `min_gap` quiet frames are
    merged into one burst (single rising edge).
    """
    starts = []
    in_burst = False
    quiet = 0
    for i, m in enumerate(mask):
        if m:
            if not in_burst:
                starts.append(i)
                in_burst = True
            quiet = 0
        else:
            if in_burst:
                quiet += 1
                if quiet >= min_gap:
                    in_burst = False
                    quiet = 0
    return np.asarray(starts, dtype=int)


def attribute(case_path: Path) -> dict:
    with open(case_path) as f:
        data = json.load(f)
    stem = data["stem"]
    hop = data["hop_size"]
    sr = data["sample_rate"]
    n_frames = data["n_frames"]

    delay_issues = detect_delay_issues(data["delay_est_trace"], hop, sr)
    filter_issues = detect_filter_issues(data["frames"])

    n_de = len(delay_issues)
    n_fi = len(filter_issues)
    n_align = min(n_de, n_fi)
    delay_issues = delay_issues[:n_align]
    filter_issues = filter_issues[:n_align]

    filter_bursts = burst_starts(filter_issues)
    delay_bursts = burst_starts(delay_issues)

    lead_window_frames = int(LEAD_WINDOW_MS * sr / 1000 / hop)
    attributed = 0
    lead_times = []
    for fb in filter_bursts:
        # search delay-issue starts within [fb - lead_window_frames, fb]
        cand = delay_bursts[(delay_bursts >= fb - lead_window_frames)
                            & (delay_bursts <= fb)]
        if len(cand) > 0:
            attributed += 1
            lead_times.append((fb - cand.max()) * hop / sr * 1000)

    n_fb = len(filter_bursts)
    n_db = len(delay_bursts)
    attribution_rate = (attributed / n_fb) if n_fb > 0 else 0.0

    # Per-window "delay issue present" rate (alternative metric)
    bad_window = filter_issues
    delay_issue_in_bad_windows = (
        float(np.mean(delay_issues[bad_window])) if bad_window.sum() > 0 else 0.0
    )

    # ERLE summary
    erle = np.array([fr["erle_inst_db"] for fr in data["frames"][:n_align]])
    erle_mean = float(erle.mean())
    erle_p95_bad = (
        float(np.percentile(erle[bad_window], 95))
        if bad_window.sum() > 0 else 0.0
    )
    erle_p5_bad = (
        float(np.percentile(erle[bad_window], 5))
        if bad_window.sum() > 0 else 0.0
    )

    # PAR / top-ratio summary in BAD windows specifically
    par_in_bad = []
    top_ratio_in_bad = []
    for i in range(n_align):
        if not filter_issues[i]:
            continue
        if i < len(data["delay_est_trace"]):
            t = data["delay_est_trace"][i]
            par_in_bad.append(float(t.get("last_par", 0.0)))
            p1 = float(t.get("top1_par", 0.0))
            p2 = float(t.get("top2_par", 0.0))
            if p1 > 1e-6:
                top_ratio_in_bad.append(p2 / p1)
    par_arr = np.array(par_in_bad) if par_in_bad else np.array([0.0])
    rat_arr = np.array(top_ratio_in_bad) if top_ratio_in_bad else np.array([0.0])

    # Verdict per case
    if attribution_rate >= H1_STRONG_PCT and delay_issue_in_bad_windows >= 0.30:
        verdict = "H1_STRONG"
    elif attribution_rate < H2_STRONG_PCT and delay_issue_in_bad_windows < 0.30:
        verdict = "H2"
    else:
        verdict = "MIXED"

    return {
        "stem": stem,
        "n_frames": n_align,
        "n_filter_bursts": n_fb,
        "n_delay_bursts": n_db,
        "attribution_rate": attribution_rate,
        "delay_issue_in_bad_windows_rate": delay_issue_in_bad_windows,
        "mean_lead_ms": float(np.mean(lead_times)) if lead_times else 0.0,
        "filter_bad_rate": float(filter_issues.mean()),
        "delay_bad_rate": float(delay_issues.mean()),
        "erle_mean_db": erle_mean,
        "erle_p5_bad_db": erle_p5_bad,
        "erle_p95_bad_db": erle_p95_bad,
        "par_mean_in_bad": float(par_arr.mean()),
        "par_p10_in_bad": float(np.percentile(par_arr, 10)),
        "top_ratio_mean_in_bad": float(rat_arr.mean()),
        "top_ratio_p95_in_bad": float(np.percentile(rat_arr, 95)),
        "verdict": verdict,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <audit_dir>", file=sys.stderr)
        return 1
    audit_dir = Path(sys.argv[1])
    files = sorted(audit_dir.glob("*.json"))
    if not files:
        print(f"no JSONs in {audit_dir}", file=sys.stderr)
        return 1

    results = []
    for f in files:
        try:
            r = attribute(f)
        except Exception as e:
            print(f"  {f.name} FAILED: {e}")
            continue
        results.append(r)

    # Print table
    print(f"\n{'stem':<55} {'verdict':<10} {'attrib':>6} {'bad_de_in_fi':>12} "
          f"{'lead_ms':>8} {'fi_burst':>9} {'erle_p5_bad':>12} "
          f"{'par_p10_bad':>12} {'top_ratio_p95':>14}")
    for r in results:
        print(f"{r['stem']:<55} {r['verdict']:<10} "
              f"{r['attribution_rate']*100:>5.1f}% "
              f"{r['delay_issue_in_bad_windows_rate']*100:>11.1f}% "
              f"{r['mean_lead_ms']:>7.0f} "
              f"{r['n_filter_bursts']:>9} "
              f"{r['erle_p5_bad_db']:>11.2f} "
              f"{r['par_p10_in_bad']:>11.2f} "
              f"{r['top_ratio_p95_in_bad']:>13.3f}")

    # Aggregate
    n_h1 = sum(1 for r in results if r["verdict"] == "H1_STRONG")
    n_h2 = sum(1 for r in results if r["verdict"] == "H2")
    n_mix = sum(1 for r in results if r["verdict"] == "MIXED")
    mean_attrib = float(np.mean([r["attribution_rate"] for r in results]))
    mean_bad_window_rate = float(np.mean(
        [r["delay_issue_in_bad_windows_rate"] for r in results]))
    print(f"\n--- aggregate over {len(results)} cases ---")
    print(f"  H1_STRONG: {n_h1}/{len(results)}  H2: {n_h2}/{len(results)}  "
          f"MIXED: {n_mix}/{len(results)}")
    print(f"  mean attribution rate: {mean_attrib*100:.1f}%")
    print(f"  mean delay-issue-in-bad-window rate: {mean_bad_window_rate*100:.1f}%")

    # Overall verdict
    if mean_attrib >= H1_STRONG_PCT and mean_bad_window_rate >= 0.30:
        overall = "H1_STRONG — OPEN delay-aware mechanism arc"
    elif mean_attrib < H2_STRONG_PCT and mean_bad_window_rate < 0.30:
        overall = "H2 — CLOSE C6 audit; Phase 3-4 ROI estimates stand"
    else:
        overall = "MIXED — OPEN narrow delay-confidence gate (consume `is_solid`)"
    print(f"\nOVERALL VERDICT: {overall}\n")

    # Save aggregate JSON
    out = {
        "n_cases": len(results),
        "n_h1_strong": n_h1,
        "n_h2": n_h2,
        "n_mixed": n_mix,
        "mean_attribution_rate": mean_attrib,
        "mean_delay_issue_in_bad_window_rate": mean_bad_window_rate,
        "overall_verdict": overall,
        "per_case": results,
    }
    out_path = audit_dir / "attribution.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"attribution.json -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
