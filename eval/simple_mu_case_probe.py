"""Per-case investigation of a simple-mu A/B regression.

For one clip and one grid, runs baseline (all four constants frozen) and
candidate (all four retimed) through the eval driver's own `run_ours()`, records
the full simple-mu state trajectory of each, and measures the output difference
in physical units.

## What can and cannot be measured on this corpus

`wav/aec_challenge_blind` ships **mic and lpb only**. There is no clean
near-end target in any scenario -- not even `nearend_singletalk`, whose lpb sits
only ~3.7 dB below the mic, so the far end is playing there too. **SI-SDR and
STOI are therefore not computable anywhere in this corpus**, and any figure
presented as one would be computed against the mic, i.e. against a reference
that still contains the echo the AEC is supposed to remove. Saying so is the
point: a DT degradation on this corpus cannot be physically decoupled, and the
remaining honest options are the target-free measure below plus listening.

The target-free measure: split the hops by far-end activity.

  far-INACTIVE hops   No echo is arriving, so everything in the mic is near end
                      (plus noise). Any output energy the candidate removes
                      relative to the baseline is near-end damage, full stop --
                      no reference needed. `d_near_db` below.
  far-ACTIVE hops     On an FS clip the output IS residual echo, so lower is
                      better. On a DT clip the two are superimposed and this
                      number alone cannot separate them. `d_far_db`.

Far activity is taken from the lpb, delay-compensated by the bulk delay the
run's own cross-correlation reports, so a hop is only called inactive when the
echo of that far-end segment has already passed.

    python3 eval/simple_mu_case_probe.py --stem <stem> --frame-size 256 \\
        --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "python"))

DATASET = os.path.join(_REPO, "wav", "aec_challenge_blind")

FROZEN = dict(_SIMPLE_MU_HOLDOFF_HOPS=20, _SIMPLE_MU_ALPHA_ATTACK=0.3,
              _SIMPLE_MU_ALPHA_HOLD=0.99, _SIMPLE_MU_ALPHA_RELEASE=0.95)


def retimed_values(sample_rate, frame_size):
    """Rejected candidate values, independent of production's disposition."""
    from modules import aec3_scale
    hop = frame_size // 2
    return {
        "_SIMPLE_MU_HOLDOFF_HOPS":
            aec3_scale.ms_to_hops(200.0, hop, sample_rate),
        "_SIMPLE_MU_ALPHA_ATTACK":
            aec3_scale.growth_rehop(0.3, 160, 16000, hop, sample_rate),
        "_SIMPLE_MU_ALPHA_HOLD":
            aec3_scale.growth_rehop(0.99, 160, 16000, hop, sample_rate),
        "_SIMPLE_MU_ALPHA_RELEASE":
            aec3_scale.growth_rehop(0.95, 160, 16000, hop, sample_rate),
    }


def case_paths(stem):
    scenario = stem.split("_", 1)[1]
    if scenario.endswith("_with_movement"):
        scenario = scenario[: -len("_with_movement")]
    base = os.path.join(DATASET, scenario)
    return (os.path.join(base, f"{stem}_mic.wav"),
            os.path.join(base, f"{stem}_lpb.wav"))


def run_with_trace(mic, ref, sr, overrides):
    """Run the clip through the driver's own entry point, tracing simple-mu.

    Traced by wrapping AEC.process rather than by editing the module: the branch
    is identified from the holdoff counter's transition, which needs no
    instrumentation inside the update (up = attack/fresh onset, down = hold,
    flat at 0 = release, flat nonzero = attack/ongoing).
    """
    import eval_aec_challenge as E
    from modules import orchestrator as O

    orig_enable_cng = E._ENABLE_CNG
    E._ENABLE_CNG = True
    trace = {"branch": [], "holdoff": [], "ratio": [], "mu": []}
    orig_init, orig_process = O.AEC.__init__, O.AEC.process
    orig_mu = O.AEC._get_simple_mu_scale
    orig_constants = {k: getattr(O, k) for k in FROZEN}
    state = {}

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        state["aec"] = self
        state["prev"] = self._simple_mu_holdoff

    def mu_scale(self, mu_min=None):
        """Record what the REAL call returned. The first version of this probe
        called _get_simple_mu_scale() from the process wrapper instead, which
        changed the run it was measuring: that method decrements
        _warmup_frames as a side effect, so every hop consumed two warmup
        frames instead of one and the probe disagreed with the A/B it was
        supposed to explain."""
        v = orig_mu(self, mu_min)
        state["mu"] = float(np.mean(np.asarray(v, dtype=np.float64)))
        return v

    def process(self, near, far):
        state["mu"] = float("nan")
        out = orig_process(self, near, far)
        mu = state["mu"]
        now = self._simple_mu_holdoff
        prev = state["prev"]
        if now > prev:
            branch = "attack_onset"
        elif now < prev:
            branch = "hold"
        elif now == 0:
            branch = "release"
        else:
            branch = "attack_ongoing"
        state["prev"] = now
        trace["branch"].append(branch)
        trace["holdoff"].append(int(now))
        trace["ratio"].append(float(self._simple_mu_ratio))
        trace["mu"].append(mu)
        return out

    for key, value in overrides.items():
        setattr(O, key, value)
    O.AEC.__init__, O.AEC.process = init, process
    O.AEC._get_simple_mu_scale = mu_scale
    try:
        y = E.run_ours(mic, ref, sr, 52, preset="balanced", is_movement=False)
    finally:
        O.AEC.__init__, O.AEC.process = orig_init, orig_process
        O.AEC._get_simple_mu_scale = orig_mu
        E._ENABLE_CNG = orig_enable_cng
        for key, value in orig_constants.items():
            setattr(O, key, value)
    return np.asarray(y, dtype=np.float64), trace, state["aec"].config.hop_size


def bulk_delay(mic, ref, sr, max_ms=500.0):
    """GCC-PHAT bulk delay, used ONLY to align the far-activity mask -- never
    fed to the AEC (that would be the pre-align crutch this bench forbids)."""
    n = 1 << int(np.ceil(np.log2(len(mic) + len(ref))))
    M = np.fft.rfft(mic, n)
    R = np.fft.rfft(ref, n)
    X = M * np.conj(R)
    X /= np.maximum(np.abs(X), 1e-12)
    cc = np.fft.irfft(X, n)
    lim = int(max_ms * 1e-3 * sr)
    return int(np.argmax(cc[:lim]))


def hop_energy(x, hop, n_hops):
    return np.array([np.sqrt((x[i * hop:(i + 1) * hop] ** 2).mean() + 1e-20)
                     for i in range(n_hops)])


def probe(stem, frame_size, far_active_db=-45.0):
    mic_path, lpb_path = case_paths(stem)
    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(lpb_path)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    env_names = ("AEC_CFG_OVERRIDE", "NO_PREALIGN")
    old_env = {name: os.environ.get(name) for name in env_names}
    os.environ["AEC_CFG_OVERRIDE"] = f"frame_size={frame_size}"
    os.environ["NO_PREALIGN"] = "1"
    candidate_values = retimed_values(sr, frame_size)
    try:
        y_base, tr_base, hop = run_with_trace(mic, ref, sr, dict(FROZEN))
        y_cand, tr_cand, _ = run_with_trace(
            mic, ref, sr, candidate_values)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    m = min(len(y_base), len(y_cand))
    y_base, y_cand = y_base[:m], y_cand[:m]
    n_hops = min(m // hop, len(tr_base["ratio"]), len(tr_cand["ratio"]))

    delay = bulk_delay(mic, ref, sr)
    ref_al = np.zeros(n)
    if 0 < delay < n:
        ref_al[delay:] = ref[:n - delay]
    else:
        ref_al = ref.copy()

    e_ref = hop_energy(ref_al, hop, n_hops)
    e_base = hop_energy(y_base, hop, n_hops)
    e_cand = hop_energy(y_cand, hop, n_hops)
    ref_db = 20 * np.log10(e_ref / max(e_ref.max(), 1e-12))
    far_active = ref_db > far_active_db

    def db(ea, eb, mask):
        """Ratio of RMS over the masked HOPS. Takes per-hop energies rather
        than samples so the far-activity mask, which is per hop, cannot be
        applied to a sample-length array by accident."""
        if mask.sum() == 0:
            return None
        return float(20 * np.log10(
            (np.sqrt((ea[mask] ** 2).mean()) + 1e-12) /
            (np.sqrt((eb[mask] ** 2).mean()) + 1e-12)))

    diff = np.abs(y_base - y_cand)
    first_div_hop = None
    for h in range(n_hops):
        if diff[h * hop:(h + 1) * hop].max() > 1e-9:
            first_div_hop = h
            break

    def branch_counts(tr):
        c = {}
        for b in tr["branch"][:n_hops]:
            c[b] = c.get(b, 0) + 1
        return c

    def transitions(tr):
        """Hop indices where the branch changes -- the timings the retiming is
        supposed to move."""
        out = []
        prev = None
        for i, b in enumerate(tr["branch"][:n_hops]):
            if b != prev:
                out.append([i, b])
                prev = b
        return out

    per_sec = []
    hops_per_sec = max(1, int(round(sr / hop)))
    for s in range(0, n_hops, hops_per_sec):
        sl = slice(s, min(s + hops_per_sec, n_hops))
        seg = np.zeros(n_hops, dtype=bool)
        seg[sl] = True
        per_sec.append({
            "t": s * hop / sr,
            "far_active_frac": float(far_active[sl].mean()),
            "base_db": float(20 * np.log10(e_base[sl].mean() + 1e-12)),
            "cand_db": float(20 * np.log10(e_cand[sl].mean() + 1e-12)),
            "d_db": float(20 * np.log10((e_cand[sl].mean() + 1e-12) /
                                        (e_base[sl].mean() + 1e-12))),
            "erle_base_db": float(20 * np.log10(
                (hop_energy(mic, hop, n_hops)[sl].mean() + 1e-12) /
                (e_base[sl].mean() + 1e-12))),
            "erle_cand_db": float(20 * np.log10(
                (hop_energy(mic, hop, n_hops)[sl].mean() + 1e-12) /
                (e_cand[sl].mean() + 1e-12))),
            "mu_base": float(np.nanmean(tr_base["mu"][sl])),
            "mu_cand": float(np.nanmean(tr_cand["mu"][sl])),
            "ratio_base": float(np.mean(tr_base["ratio"][sl])),
            "ratio_cand": float(np.mean(tr_cand["ratio"][sl])),
        })

    return {
        "stem": stem, "frame_size": frame_size, "hop": hop,
        "sample_rate": sr, "n_hops": n_hops, "bulk_delay_samples": delay,
        "first_divergent_hop": first_div_hop,
        "first_divergent_ms": None if first_div_hop is None
        else first_div_hop * hop / sr * 1000.0,
        "far_active_hops": int(far_active.sum()),
        "far_inactive_hops": int((~far_active).sum()),
        # The two numbers the verdict rests on.
        "d_near_db": db(e_cand, e_base, ~far_active),
        "d_far_db": db(e_cand, e_base, far_active),
        "d_overall_db": db(e_cand, e_base, np.ones(n_hops, dtype=bool)),
        "si_sdr": None, "stoi": None,
        "si_sdr_note": "not computable: the corpus ships mic+lpb only, with no "
                       "clean near-end in any scenario, so any SI-SDR/STOI here "
                       "would be scored against a reference that still contains "
                       "the echo. Use d_near_db, and listen.",
        "holdoff_limit": {
            "base": FROZEN["_SIMPLE_MU_HOLDOFF_HOPS"],
            "cand": candidate_values["_SIMPLE_MU_HOLDOFF_HOPS"],
        },
        "branch_counts": {"base": branch_counts(tr_base),
                          "cand": branch_counts(tr_cand)},
        "branch_transitions": {"base": transitions(tr_base),
                               "cand": transitions(tr_cand)},
        "per_second": per_sec,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--frame-size", type=int, default=256)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    r = probe(args.stem, args.frame_size)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1, sort_keys=True)
            fh.write("\n")

    print(f"{r['stem']}  frame={r['frame_size']} hop={r['hop']} "
          f"{r['n_hops']} hops")
    print(f"  first divergence: hop {r['first_divergent_hop']} "
          f"({r['first_divergent_ms']:.0f} ms)"
          if r["first_divergent_hop"] is not None else "  no divergence")
    print(f"  far-inactive hops {r['far_inactive_hops']}, "
          f"far-active {r['far_active_hops']}")
    print(f"  d_near_db {r['d_near_db']:+.2f}   (negative = candidate removed "
          f"near end the baseline kept)")
    print(f"  d_far_db  {r['d_far_db']:+.2f}   (on an FS clip: positive = more "
          f"residual echo)")
    print(f"  branches base {r['branch_counts']['base']}")
    print(f"           cand {r['branch_counts']['cand']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
