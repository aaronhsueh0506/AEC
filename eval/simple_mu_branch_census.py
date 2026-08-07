"""Branch census for the F2.4 simple-mu update, over the 90-case blind corpus.

A retimed constant is only evidence if the branch that reads it is exercised by
the corpus the A/B ran on. `_update_simple_mu_ratio` has three branches and each
uses a different retimed retention:

    attack    _simple_mu_alpha_attack    and arms the holdoff on a fresh onset
    hold      _simple_mu_alpha_hold      while the holdoff runs down
    release   _simple_mu_alpha_release   once it is spent

A neutral A/B result means something different depending on which of those the
corpus actually reached. If `release` never fires, a neutral verdict says
nothing about its coefficient.

No instrumentation is needed and none was added: the holdoff counter's
transition identifies the branch unambiguously.

    counter went UP           attack, fresh onset (armed to the limit)
    counter went DOWN         hold
    counter flat and nonzero  attack, ongoing
    counter flat at zero      release

The last line is exact rather than a default: an attack at zero would arm the
counter, and the limit is always >= 1, so a 0 -> 0 transition can only be
release.

    python3 eval/simple_mu_branch_census.py --out census.json [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

DATASET = os.path.join(_REPO, "wav", "aec_challenge_blind")
MANIFEST = os.path.join(_REPO, "eval", "manifest_90case_stems.txt")


def case_paths(stem):
    scenario = stem.split("_", 1)[1]
    if scenario.endswith("_with_movement"):
        scenario = scenario[: -len("_with_movement")]
    base = os.path.join(DATASET, scenario)
    return (os.path.join(base, f"{stem}_mic.wav"),
            os.path.join(base, f"{stem}_lpb.wav"))


def census_one(args):
    stem, frame_size = args
    from aec import AEC, AecConfig

    mic_path, lpb_path = case_paths(stem)
    mic, sr = sf.read(mic_path, dtype="float64", always_2d=False)
    ref, _ = sf.read(lpb_path, dtype="float64", always_2d=False)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    aec = AEC(AecConfig(sample_rate=sr, frame_size=frame_size,
                        enable_cng=True))
    hop = aec.config.hop_size
    counts = collections.Counter()
    prev = aec._simple_mu_holdoff
    hops = 0
    for i in range(0, n - hop + 1, hop):
        aec.process(mic[i:i + hop], ref[i:i + hop])
        now = aec._simple_mu_holdoff
        if now > prev:
            counts["attack_onset"] += 1
        elif now < prev:
            counts["hold"] += 1
        elif now == 0:
            counts["release"] += 1
        else:
            counts["attack_ongoing"] += 1
        prev = now
        hops += 1
    return stem, aec._simple_mu_holdoff_limit, hops, dict(counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frame-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="census only the first N cases (a partial census must "
                         "say so in whatever cites it)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(MANIFEST, encoding="utf-8") as fh:
        stems = [line.strip() for line in fh
                 if line.strip() and not line.startswith("#")]
    if args.limit:
        stems = stems[: args.limit]

    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(census_one,
                           [(s, args.frame_size) for s in stems])

    total = collections.Counter()
    per_case = {}
    limits = set()
    for stem, limit, hops, counts in results:
        total.update(counts)
        per_case[stem] = {"hops": hops, **counts}
        limits.add(limit)

    branches = ("attack_onset", "attack_ongoing", "hold", "release")
    missing = [b for b in branches if total.get(b, 0) == 0]
    report = {
        "frame_size": args.frame_size,
        "cases": len(stems),
        "partial": bool(args.limit),
        "holdoff_limit_hops": sorted(limits),
        "total_hops": sum(v["hops"] for v in per_case.values()),
        "totals": {b: total.get(b, 0) for b in branches},
        "branches_never_taken": missing,
        "per_case": per_case,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")

    print(f"frame_size={args.frame_size}  {len(stems)} cases, "
          f"{report['total_hops']} hops, holdoff limit "
          f"{report['holdoff_limit_hops']}")
    for b in branches:
        print(f"  {b:<16} {total.get(b, 0):>9}")
    if missing:
        print(f"  NOT EXERCISED: {missing} -- a neutral A/B says nothing about "
              f"the coefficient those branches read", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
