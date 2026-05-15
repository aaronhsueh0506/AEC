"""v3.16 C6 — DelayEst audit (Phase 1 audit-only, 2026-05-15).

Per-frame trace dump of DelayEstimator state + AecStats correlator
fields on Tier A cases. Output JSONs feed C6 verdict adjudication:

  H1: DelayEst is upstream root cause for movement / cohort tail
      closures (Arc M V1 FS_movement, Arc F cohort tail, Arc G
      destructive W reset, qNvSMyU class catastrophe). Evidence:
      DelayEst issues (PAR < threshold, top1/top2 ambiguous,
      stale estimated_delay, init mis-lock) PRECEDE filter issues
      (high residual_echo_psd, EPC fires, cohort_tail_T,
      divergence, NE compression).

  H2: DelayEst behaves correctly throughout. Closures are mechanism
      walls (Q/R steady-state, W destructive zero-out, RES gate
      trade-off). DelayEst trace shows normal behavior in bad-frame
      windows; filter issues coincide with mechanism events.

Audit is read-only — uses existing trace surface (`trace_delay_est`
flag + AecStats per-frame). Zero behaviour change vs production
default config.

Usage:

    python3 tools/research/v3_16_c6_delay_est_audit.py \\
        --cases tools/research/v3_16_c6_tier_a_cases.txt \\
        --dataset wav/aec_challenge_blind \\
        --out /tmp/v3_16_c6_audit/
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


SCENARIO_SUFFIXES = (
    "_doubletalk_with_movement",
    "_farend_singletalk_with_movement",
    "_nearend_singletalk",
    "_farend_singletalk",
    "_doubletalk",
)

SCENARIO_FOLDER = {
    "_doubletalk_with_movement": "doubletalk",
    "_farend_singletalk_with_movement": "farend_singletalk",
    "_nearend_singletalk": "nearend_singletalk",
    "_farend_singletalk": "farend_singletalk",
    "_doubletalk": "doubletalk",
}


def parse_stem(stem: str) -> tuple[str, str]:
    for suffix in SCENARIO_SUFFIXES:
        if stem.endswith(suffix):
            return SCENARIO_FOLDER[suffix], suffix
    raise ValueError(f"unrecognized stem suffix: {stem}")


def run_one(stem: str, dataset_dir: Path, sample_rate: int = 16000) -> dict:
    folder, _ = parse_stem(stem)
    mic_path = dataset_dir / folder / f"{stem}_mic.wav"
    ref_path = dataset_dir / folder / f"{stem}_lpb.wav"
    if not mic_path.exists() or not ref_path.exists():
        raise FileNotFoundError(f"{mic_path} or {ref_path} missing")

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

    mic, sr_mic = sf.read(mic_path)
    ref, sr_ref = sf.read(ref_path)
    if sr_mic != sample_rate or sr_ref != sample_rate:
        raise ValueError(
            f"sample rate mismatch on {stem}: mic={sr_mic} ref={sr_ref}"
        )
    mic = np.asarray(mic, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    hop = aec.hop_size
    frames = []
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        s = aec.get_stats()
        d = aec._diag
        frames.append({
            "frame": int(s.frame_count),
            "time_s": float(s.time_s),
            "filter_state": str(d.get("filter_state", "")),
            "filter_converged": bool(s.filter_converged),
            "filter_once_converged": bool(s.filter_once_converged),
            "erle_inst_db": float(s.erle_inst_db),
            "mic_db": float(s.mic_power_db),
            "far_db": float(s.far_power_db),
            "err_db": float(s.error_power_db),
            "echo_psd_db": float(s.echo_psd_mean_db),
            "err_psd_db": float(s.error_psd_mean_db),
            "res_gain_db": float(s.res_gain_mean_db),
            "using_render": bool(s.res_using_render),
            "divergence": float(s.divergence),
            "epc_active": bool(s.epc_active),
            "epv_gain_ratio": float(d.get("epv_gain_ratio", 1.0)),
            "cohort_tail_T": bool(s.cohort_tail_T),
            "dt_conf": float(s.dt_confidence),
            "dt_active": bool(s.dt_active),
            "dt_from_energy": float(s.dt_from_energy),
            "mu_scale": float(s.mu_scale),
            "shadow_advantage": float(s.shadow_advantage),
            "main_paused": bool(s.main_paused),
            "delay_samples": int(s.delay_samples),
            "delay_ms": float(s.delay_ms),
            "far_activity": float(s.far_activity),
            "residual_psd_linear": float(d.get("residual_psd_linear", 0.0)),
            "residual_psd_render": float(d.get("residual_psd_render", 0.0)),
            "residual_render_blend": float(d.get("residual_render_blend", 0.0)),
            "erl_estimate": float(d.get("erl_estimate", 0.0)),
            "erle_slope_db_per_s": float(d.get("erle_slope_db_per_s", 0.0)),
        })
        pos += hop

    delay_trace = []
    if aec.delay_est is not None and aec.delay_est._trace_rows:
        for r in aec.delay_est._trace_rows:
            delay_trace.append({k: (v if isinstance(v, (int, float, bool, str))
                                    else float(v)) for k, v in r.items()})

    folder_label, suffix = parse_stem(stem)
    return {
        "stem": stem,
        "scenario_folder": folder_label,
        "sample_rate": sample_rate,
        "hop_size": hop,
        "n_frames": len(frames),
        "duration_s": n / sample_rate,
        "config": {
            "preset": "balanced",
            "filter_length": 832,
            "trace_delay_est": True,
            "arc_t_cohort_detector_default": True,
        },
        "frames": frames,
        "delay_est_trace": delay_trace,
    }


def write_summary_csv(results: list[dict], out_path: Path) -> None:
    """Per-case summary row: aggregate fire-rates + DelayEst summary."""
    import csv
    cols = [
        "stem", "scenario", "n_frames", "duration_s",
        "epc_fire_rate", "cohort_tail_T_fire_rate",
        "divergence_p95", "shadow_adv_p95",
        "res_using_render_rate", "main_paused_rate",
        "delay_est_n_estimates", "delay_est_par_mean", "delay_est_par_p10",
        "delay_est_top12_par_ratio_mean",
        "estimated_delay_p50_samp", "estimated_delay_p95_samp",
        "filter_converged_rate", "erle_windowed_db_mean",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            frames = r["frames"]
            n = max(1, len(frames))
            epc_rate = sum(1 for fr in frames if fr["epc_active"]) / n
            ct_rate = sum(1 for fr in frames if fr["cohort_tail_T"]) / n
            divs = np.array([fr["divergence"] for fr in frames])
            sh_adv = np.array([fr["shadow_advantage"] for fr in frames])
            using_render = sum(1 for fr in frames if fr["using_render"]) / n
            main_paused = sum(1 for fr in frames if fr["main_paused"]) / n
            d_estimates = [t for t in r["delay_est_trace"] if t.get("did_estimate")]
            par_vals = np.array([float(t.get("last_par", 0.0))
                                 for t in d_estimates]) if d_estimates else np.array([0.0])
            ratio_vals = []
            for t in d_estimates:
                p1 = float(t.get("top1_par", 0.0))
                p2 = float(t.get("top2_par", 0.0))
                if p1 > 1e-6:
                    ratio_vals.append(p2 / p1)
            ratio_vals = np.array(ratio_vals) if ratio_vals else np.array([0.0])
            delay_samp = np.array([fr["delay_samples"] for fr in frames])
            conv_rate = sum(1 for fr in frames if fr["filter_converged"]) / n
            erle = np.array([fr["erle_inst_db"] for fr in frames])
            w.writerow([
                r["stem"], r["scenario_folder"], n, f"{r['duration_s']:.2f}",
                f"{epc_rate:.4f}", f"{ct_rate:.4f}",
                f"{float(np.percentile(divs, 95)):.4f}",
                f"{float(np.percentile(sh_adv, 95)):.4f}",
                f"{using_render:.4f}", f"{main_paused:.4f}",
                len(d_estimates),
                f"{float(par_vals.mean()):.3f}",
                f"{float(np.percentile(par_vals, 10)):.3f}",
                f"{float(ratio_vals.mean()):.3f}",
                int(np.percentile(delay_samp, 50)),
                int(np.percentile(delay_samp, 95)),
                f"{conv_rate:.4f}",
                f"{float(erle.mean()):.3f}",
            ])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path,
                    help="text file with one stem per line (# = comment)")
    ap.add_argument("--dataset", required=True, type=Path,
                    help="path to wav/aec_challenge_blind/")
    ap.add_argument("--out", required=True, type=Path,
                    help="output dir for per-case JSONs + summary.csv")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    stems = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stems.append(line)
    print(f"loaded {len(stems)} stems from {args.cases}")

    results = []
    t_start = time.time()
    for i, stem in enumerate(stems, 1):
        t0 = time.time()
        try:
            data = run_one(stem, args.dataset)
        except Exception as e:
            print(f"  [{i}/{len(stems)}] {stem} FAILED: {e}")
            continue
        out_json = args.out / f"{stem}.json"
        with open(out_json, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        dt = time.time() - t0
        print(f"  [{i}/{len(stems)}] {stem} {data['n_frames']} frames "
              f"({data['duration_s']:.1f}s) in {dt:.1f}s -> {out_json.name}")
        results.append(data)

    write_summary_csv(results, args.out / "summary.csv")
    print(f"\nsummary.csv at {args.out / 'summary.csv'}")
    print(f"total wall time {time.time() - t_start:.1f}s "
          f"({len(results)}/{len(stems)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
