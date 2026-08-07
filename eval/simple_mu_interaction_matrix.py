"""Full 2^4 interaction matrix over the four simple-mu constants.

Six points -- baseline, all-four, and each constant alone -- cannot tell an
interaction from an accident. Four binary knobs have 16 settings, and the six
that were measured first are the two extremes plus four of the fourteen
interior points.

They are also not four independent EMAs. The mechanism is a state machine:

  * the branch condition is re-evaluated every hop, so a change to any
    retention changes WHICH branch later hops take;
  * the holdoff counter only decrements on the hold branch, so changing the
    attack or release rate changes how often the counter is even reached;
  * the holdoff limit changes how long the frozen branch lasts, which changes
    the ratio trajectory, which feeds back into the branch condition.

So "each alpha and the holdoff are individually converted correctly" does not
imply the state machine as a whole is wall-clock invariant. That claim needs the
full matrix, and it is the matrix's real purpose -- not just picking a winner.

Each setting is scored per clip with target-free physical measures (see
`simple_mu_case_probe.py` for why SI-SDR/STOI are not computable on this
corpus): residual-echo RMS split by far-end activity.

    python3 eval/simple_mu_interaction_matrix.py --stems stems.txt \\
        --frame-size 256 --workers 5 --out matrix_g256.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))
sys.path.insert(0, _HERE)

KNOBS = ["holdoff", "attack", "hold", "release"]
CONSTANTS = {
    "holdoff": "_SIMPLE_MU_HOLDOFF_HOPS",
    "attack": "_SIMPLE_MU_ALPHA_ATTACK",
    "hold": "_SIMPLE_MU_ALPHA_HOLD",
    "release": "_SIMPLE_MU_ALPHA_RELEASE",
}


def mask_name(bits):
    """0000 = everything frozen (baseline), 1111 = everything retimed."""
    return "".join("1" if b else "0" for b in bits)


def run_one(job):
    stem, frame_size, bits = job
    from simple_mu_case_probe import (FROZEN, bulk_delay, case_paths,
                                      hop_energy, retimed_values)
    import eval_aec_challenge as E
    from modules import orchestrator as O

    mic_path, lpb_path = case_paths(stem)
    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(lpb_path)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    retimed = retimed_values(sr, frame_size)
    overrides = {
        CONSTANTS[k]: retimed[CONSTANTS[k]] if bit else FROZEN[CONSTANTS[k]]
        for k, bit in zip(KNOBS, bits)
    }
    env_names = ("AEC_CFG_OVERRIDE", "NO_PREALIGN")
    old_env = {name: os.environ.get(name) for name in env_names}
    old_enable_cng = E._ENABLE_CNG
    original = {key: getattr(O, key) for key in overrides}
    try:
        os.environ["AEC_CFG_OVERRIDE"] = f"frame_size={frame_size}"
        os.environ["NO_PREALIGN"] = "1"
        E._ENABLE_CNG = True
        for key, value in overrides.items():
            setattr(O, key, value)
        y = np.asarray(E.run_ours(mic, ref, sr, 52, preset="balanced",
                                  is_movement=False), dtype=np.float64)
    finally:
        for key, value in original.items():
            setattr(O, key, value)
        E._ENABLE_CNG = old_enable_cng
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    hop = frame_size // 2
    n_hops = len(y) // hop
    delay = bulk_delay(mic, ref, sr)
    ref_al = np.zeros(n)
    if 0 < delay < n:
        ref_al[delay:] = ref[:n - delay]
    else:
        ref_al = ref.copy()
    e_ref = hop_energy(ref_al, hop, n_hops)
    e_y = hop_energy(y, hop, n_hops)
    far = 20 * np.log10(e_ref / max(e_ref.max(), 1e-12)) > -45.0

    def rms(mask):
        return None if mask.sum() == 0 else float(
            np.sqrt((e_y[mask] ** 2).mean()))

    return {
        "stem": stem, "frame_size": frame_size, "mask": mask_name(bits),
        "rms_all": rms(np.ones(n_hops, dtype=bool)),
        "rms_far": rms(far), "rms_near": rms(~far),
        "n_far": int(far.sum()), "n_near": int((~far).sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stems", required=True)
    ap.add_argument("--frame-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(args.stems, encoding="utf-8") as fh:
        stems = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    settings = list(itertools.product([0, 1], repeat=4))
    jobs = [(s, args.frame_size, bits) for s in stems for bits in settings]
    print(f"{len(stems)} stems x {len(settings)} settings = {len(jobs)} runs")

    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)

    by = {}
    for r in results:
        by.setdefault(r["stem"], {})[r["mask"]] = r

    rows = []
    for stem, per in by.items():
        base = per["0000"]
        for mask, r in per.items():
            def d(key):
                a, b = r[key], base[key]
                return None if (a is None or b is None) else float(
                    20 * np.log10((a + 1e-12) / (b + 1e-12)))
            rows.append({"stem": stem, "mask": mask,
                         "d_all_db": d("rms_all"), "d_far_db": d("rms_far"),
                         "d_near_db": d("rms_near")})

    per_mask = {}
    for mask in ("".join(m) for m in itertools.product("01", repeat=4)):
        sel = [r for r in rows if r["mask"] == mask]
        far = [r["d_far_db"] for r in sel if r["d_far_db"] is not None]
        near = [r["d_near_db"] for r in sel if r["d_near_db"] is not None]
        per_mask[mask] = {
            "retimed": [k for k, c in zip(KNOBS, mask) if c == "1"],
            "n": len(sel),
            "far_mean_db": float(np.mean(far)) if far else None,
            "far_worst_db": float(np.max(far)) if far else None,
            "near_mean_db": float(np.mean(near)) if near else None,
            "near_worst_db": float(np.min(near)) if near else None,
        }

    report = {"frame_size": args.frame_size, "stems": stems,
              "knobs": KNOBS, "per_mask": per_mask, "rows": rows}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print(f"\n{'mask':6} {'retimed':38} {'far mean':>9} {'far worst':>10} "
          f"{'near mean':>10} {'near worst':>11}")
    print("(dB vs all-frozen. far: FS clips, positive = MORE residual echo. "
          "near: negative = near end removed.)")
    for mask, m in sorted(per_mask.items(),
                          key=lambda kv: (kv[1]["far_worst_db"] is None,
                                          kv[1]["far_worst_db"])):
        print(f"{mask:6} {','.join(m['retimed']) or '(none)':38} "
              f"{m['far_mean_db']:+9.3f} {m['far_worst_db']:+10.3f} "
              f"{m['near_mean_db']:+10.3f} {m['near_worst_db']:+11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
