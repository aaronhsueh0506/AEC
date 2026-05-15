"""v3.16 C3 — Stage-1 4-cap chain stacking audit (Phase 2, audit-only).

For each frame, capture which of the 3 active caps (quiet_mask /
3bin_smooth / hf_cap; epc_dt_cap removed v3.16 C1) attenuated the
voice-band gain, count the stack depth, and correlate with bad-frame
indicators (cohort_tail_T / divergence / NE / DT compression).

Verdict gate (decision for C3 mechanism arc + dependent C5 investment):

  STACKING-DRIVEN: stack=2+ frames ≥ 30% on DT bucket AND mean voice-band
  cumulative attenuation > 6 dB on stack=2+ vs stack=0/1. → OPEN C3
  mechanism arc; C5 architectural investment justified.

  NOT STACKING-DRIVEN: stack=2+ frames < 15% OR cumulative attenuation
  delta < 3 dB between stack levels. → CLOSE C3 audit; redirect Phase 2
  effort.

  MIXED: between thresholds. → narrow per-cap analysis before
  committing.

Usage:
    python3 tools/research/v3_16_c3_4cap_stacking_audit.py \\
        --cases tools/research/v3_15_subset_cases.txt \\
        --dataset wav/aec_challenge_blind \\
        --out /tmp/v3_16_c3_audit/
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


def bucket_of(stem: str) -> str:
    if stem.endswith("_with_movement"):
        if "doubletalk" in stem:
            return "DT_movement"
        return "FS_movement"
    if "doubletalk" in stem:
        return "DT_static"
    if "nearend_singletalk" in stem:
        return "NE"
    if "farend_singletalk" in stem:
        return "FS_static"
    return "UNKNOWN"


def parse_stem(stem: str) -> str:
    for suffix in SCENARIO_SUFFIXES:
        if stem.endswith(suffix):
            return SCENARIO_FOLDER[suffix]
    raise ValueError(f"unrecognized stem suffix: {stem}")


# Threshold: a cap "fired" on the voice band if mean voice-band gain
# dropped by at least this in linear amplitude. 0.005 ~ -0.04 dB
# corresponds roughly to the audit fire-rate threshold from v3.15 §1.7.
CAP_FIRE_THRESHOLD = 0.005


def run_one(stem: str, dataset_dir: Path, sample_rate: int = 16000) -> dict:
    folder = parse_stem(stem)
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
        capture_stages=True,
    )
    np.random.seed(0)
    aec = AEC(cfg)

    mic = np.asarray(sf.read(mic_path)[0], dtype=np.float32)
    ref = np.asarray(sf.read(ref_path)[0], dtype=np.float32)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    hop = aec.hop_size
    res = aec.res
    voice_idx = res._voice_band_idx if res is not None else None

    bucket = bucket_of(stem)
    # Per-frame metrics
    frames_data = []
    pos = 0
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        sg = res.get_stage_gains() if res is not None else {}
        s = aec.get_stats()
        # If voice_idx is empty (shouldn't be), skip frame
        if voice_idx is None or voice_idx.size == 0:
            pos += hop
            continue

        # Stage-1 4-cap chain voice-band means:
        # slot 02 spectral_floor (pre-cap), 04 quiet_mask, 05 3bin_smooth,
        # 06 hf_cap, 07 pre_temporal (post-cap). Slot 03 is alias of 02
        # post v3.16 C1.
        g_pre = sg.get('02_spectral_floor')
        g_qm = sg.get('04_quiet_mask')
        g_sm = sg.get('05_3bin_smooth')
        g_hf = sg.get('06_hf_cap')
        g_post = sg.get('07_pre_temporal')

        def vmean(arr):
            if arr is None:
                return float('nan')
            try:
                return float(np.mean(arr[voice_idx]))
            except Exception:
                return float('nan')

        v_pre = vmean(g_pre)
        v_qm = vmean(g_qm)
        v_sm = vmean(g_sm)
        v_hf = vmean(g_hf)
        v_post = vmean(g_post)

        # Did each cap fire on voice band?
        # quiet_mask: gain rose to 1.0 in masked bins (i.e. unmask) → typically
        #             RAISES voice-band mean. We flag "fire" if mean diff > 0
        #             AND any voice-band bin went from sub-1 to 1.
        # 3bin_smooth: simple convolution; can raise or lower. Fire = abs(Δ) > thr
        # hf_cap: caps high-band gain at cap_bin level; voice band is below
        #         that bin so direct effect rare — keep generic Δ check.
        # For 4-cap STACKING, what matters is the cumulative ATTENUATION
        # (drop) from pre to post. Stack count = number of caps with negative Δ.
        qm_fired_neg = (v_qm - v_pre) < -CAP_FIRE_THRESHOLD if np.isfinite(
            v_qm - v_pre) else False
        sm_fired_neg = (v_sm - v_qm) < -CAP_FIRE_THRESHOLD if np.isfinite(
            v_sm - v_qm) else False
        hf_fired_neg = (v_hf - v_sm) < -CAP_FIRE_THRESHOLD if np.isfinite(
            v_hf - v_sm) else False
        stack_neg = int(qm_fired_neg) + int(sm_fired_neg) + int(hf_fired_neg)

        # Cumulative ATTENUATION pre→post on voice band (linear amplitude).
        if np.isfinite(v_pre) and np.isfinite(v_post) and v_pre > 1e-6:
            cum_db = 20.0 * np.log10((v_post + 1e-12) / (v_pre + 1e-12))
        else:
            cum_db = 0.0

        frames_data.append({
            "v_pre": v_pre,
            "v_qm": v_qm,
            "v_sm": v_sm,
            "v_hf": v_hf,
            "v_post": v_post,
            "stack_neg": stack_neg,
            "cum_db": cum_db,
            "qm_fired": int(qm_fired_neg),
            "sm_fired": int(sm_fired_neg),
            "hf_fired": int(hf_fired_neg),
            "cohort_tail_T": int(s.cohort_tail_T),
            "epc_active": int(s.epc_active),
            "divergence": float(s.divergence),
            "dt_active": int(s.dt_active),
            "filter_converged": int(s.filter_converged),
            "res_gain_db": float(s.res_gain_mean_db),
        })
        pos += hop

    return {
        "stem": stem,
        "bucket": bucket,
        "n_frames": len(frames_data),
        "voice_band_n_bins": int(voice_idx.size) if voice_idx is not None else 0,
        "frames": frames_data,
    }


def aggregate(results: list[dict]) -> dict:
    """Cross-case stacking + cumulative-attenuation distribution."""
    by_bucket = {}
    for r in results:
        bk = r["bucket"]
        by_bucket.setdefault(bk, []).append(r)

    summary = {}
    for bk in sorted(by_bucket):
        cases = by_bucket[bk]
        n_frames_total = sum(r["n_frames"] for r in cases)
        stack_hist = np.zeros(4, dtype=int)  # 0/1/2/3
        cum_db_by_stack = {0: [], 1: [], 2: [], 3: []}
        cap_fire_counts = {"qm": 0, "sm": 0, "hf": 0}
        cohort_tail_stack2plus = 0
        cohort_tail_total = 0
        epc_stack2plus = 0
        epc_total = 0
        for r in cases:
            for fr in r["frames"]:
                stack_hist[fr["stack_neg"]] += 1
                cum_db_by_stack[fr["stack_neg"]].append(fr["cum_db"])
                cap_fire_counts["qm"] += fr["qm_fired"]
                cap_fire_counts["sm"] += fr["sm_fired"]
                cap_fire_counts["hf"] += fr["hf_fired"]
                if fr["cohort_tail_T"]:
                    cohort_tail_total += 1
                    if fr["stack_neg"] >= 2:
                        cohort_tail_stack2plus += 1
                if fr["epc_active"]:
                    epc_total += 1
                    if fr["stack_neg"] >= 2:
                        epc_stack2plus += 1

        bk_summary = {
            "n_cases": len(cases),
            "n_frames": n_frames_total,
            "stack_pct": {i: float(stack_hist[i] / max(1, n_frames_total))
                          for i in range(4)},
            "cap_fire_rate": {k: v / max(1, n_frames_total)
                              for k, v in cap_fire_counts.items()},
            "cum_db_mean_by_stack": {
                i: (float(np.mean(cum_db_by_stack[i])) if cum_db_by_stack[i]
                    else 0.0)
                for i in range(4)},
            "cum_db_p95_by_stack": {
                i: (float(np.percentile(cum_db_by_stack[i], 5)) if cum_db_by_stack[i]
                    else 0.0)
                for i in range(4)},
            "stack2plus_cohort_tail_rate": (
                cohort_tail_stack2plus / cohort_tail_total
                if cohort_tail_total > 0 else 0.0),
            "stack2plus_epc_rate": (
                epc_stack2plus / epc_total if epc_total > 0 else 0.0),
        }
        summary[bk] = bk_summary

    # Global stack distribution
    all_stack = np.zeros(4, dtype=int)
    all_cum_by_stack = {0: [], 1: [], 2: [], 3: []}
    for bk in summary:
        bs = summary[bk]
        for i in range(4):
            all_stack[i] += int(bs["stack_pct"][i] * bs["n_frames"])
    total = max(1, int(all_stack.sum()))
    global_summary = {
        "total_frames": int(total),
        "stack_pct": {i: float(all_stack[i] / total) for i in range(4)},
    }

    return {"per_bucket": summary, "global": global_summary}


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
            if not line or line.startswith("#"):
                continue
            stems.append(line)
    print(f"loaded {len(stems)} stems")

    results = []
    t_start = time.time()
    for i, stem in enumerate(stems, 1):
        t0 = time.time()
        try:
            data = run_one(stem, args.dataset)
        except Exception as e:
            print(f"  [{i}/{len(stems)}] {stem} FAILED: {e}")
            continue
        out_json = args.out / "per_case" / f"{stem}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        dt = time.time() - t0
        if i % 10 == 0 or i <= 3 or i == len(stems):
            print(f"  [{i}/{len(stems)}] {stem} {data['n_frames']} frames in {dt:.1f}s")
        results.append(data)

    agg = aggregate(results)
    with open(args.out / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Print summary
    print()
    print(f"=== C3 stacking audit summary ({len(results)} cases, "
          f"{agg['global']['total_frames']} frames) ===")
    print()
    print(f"Global stack distribution:")
    for i in range(4):
        pct = agg["global"]["stack_pct"][i] * 100
        print(f"  stack={i}: {pct:6.2f}%")

    print()
    print(f"{'bucket':<15} {'n_cases':>8} {'frames':>8} "
          f"{'stk=0':>6} {'stk=1':>6} {'stk=2':>6} {'stk=3':>6} "
          f"{'cum_db@0':>9} {'cum_db@1':>9} {'cum_db@2':>9} {'cum_db@3':>9}")
    for bk in sorted(agg["per_bucket"]):
        b = agg["per_bucket"][bk]
        print(f"{bk:<15} {b['n_cases']:>8} {b['n_frames']:>8} "
              f"{b['stack_pct'][0]*100:>5.1f}% {b['stack_pct'][1]*100:>5.1f}% "
              f"{b['stack_pct'][2]*100:>5.1f}% {b['stack_pct'][3]*100:>5.1f}% "
              f"{b['cum_db_mean_by_stack'][0]:>+8.3f}dB "
              f"{b['cum_db_mean_by_stack'][1]:>+8.3f}dB "
              f"{b['cum_db_mean_by_stack'][2]:>+8.3f}dB "
              f"{b['cum_db_mean_by_stack'][3]:>+8.3f}dB")

    print()
    print(f"{'bucket':<15} {'qm_fire':>8} {'sm_fire':>8} {'hf_fire':>8} "
          f"{'stk2+_in_ctail':>15} {'stk2+_in_epc':>14}")
    for bk in sorted(agg["per_bucket"]):
        b = agg["per_bucket"][bk]
        cf = b["cap_fire_rate"]
        print(f"{bk:<15} {cf['qm']*100:>7.1f}% {cf['sm']*100:>7.1f}% "
              f"{cf['hf']*100:>7.1f}% "
              f"{b['stack2plus_cohort_tail_rate']*100:>14.1f}% "
              f"{b['stack2plus_epc_rate']*100:>13.1f}%")

    print()
    print(f"--- verdict gate evaluation ---")
    dt_pct = agg["per_bucket"].get("DT_static", {}).get("stack_pct", {}).get(2, 0) + \
             agg["per_bucket"].get("DT_static", {}).get("stack_pct", {}).get(3, 0)
    dt_mvmt_pct = agg["per_bucket"].get("DT_movement", {}).get("stack_pct", {}).get(2, 0) + \
                  agg["per_bucket"].get("DT_movement", {}).get("stack_pct", {}).get(3, 0)
    dt_avg = (dt_pct + dt_mvmt_pct) / 2
    print(f"  DT bucket stack=2+ rate (avg static/movement): {dt_avg*100:.1f}%")

    # cum_db delta between stack=0 and stack=2+ on DT bucket
    dt_b = agg["per_bucket"].get("DT_static", {})
    cum_0 = dt_b.get("cum_db_mean_by_stack", {}).get(0, 0)
    cum_2 = dt_b.get("cum_db_mean_by_stack", {}).get(2, 0)
    cum_3 = dt_b.get("cum_db_mean_by_stack", {}).get(3, 0)
    cum_2plus_avg = (cum_2 + cum_3) / 2 if cum_3 != 0 else cum_2
    delta_db = abs(cum_2plus_avg - cum_0)
    print(f"  DT_static cum_db Δ (stack=2+ vs stack=0): {delta_db:.2f} dB")

    if dt_avg >= 0.30 and delta_db >= 6.0:
        verdict = "STACKING-DRIVEN — OPEN C3 mechanism + C5 investment justified"
    elif dt_avg < 0.15 or delta_db < 3.0:
        verdict = "NOT STACKING-DRIVEN — CLOSE C3 audit; redirect Phase 2 effort"
    else:
        verdict = "MIXED — narrow per-cap analysis before commit"
    print(f"  VERDICT: {verdict}")

    print()
    print(f"total wall time {time.time() - t_start:.1f}s")
    print(f"aggregate.json -> {args.out / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
