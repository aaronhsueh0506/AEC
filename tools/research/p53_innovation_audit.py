"""P53 Step 0 — Innovation orthogonality audit.

Three subcommands:
  snapshot  — run AEC with trace_p53_innovation=True on each listed case,
              dump per-case .npz under /tmp/p53_audit/.
  analyze   — load .npz, compute r_voice trajectories + per-case aggregates.
  verdict   — apply T0 decision tree (docs/p53_design_lock.md §2.4) and
              write docs/p53_innovation_audit.md + summary JSON.

Audit code path is locked: production main HEAD, use_res_refactored=False.
See docs/p53_design_lock.md §2 + plan §"Step 0".
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'python'))

from aec import AEC, AecConfig, AecMode  # noqa: E402
from aec_p52_regime_classifier import AcousticRegime, AcousticRegimeClassifier  # noqa: E402
from run_one_case import PRESET_MAP  # noqa: E402

DATASET = _REPO / 'wav/aec_challenge_blind'
SUBSETS = ('doubletalk', 'farend_singletalk', 'nearend_singletalk')
CNG_SEED = 42
STABLE_SAMPLE_SEED = 42
STABLE_SAMPLE_SIZE = 12

# Voice band 300–3000 Hz @ 16 kHz STFT (512 fft, 257 bins → bin = k * 16000/512)
VOICE_LOW_HZ = 300.0
VOICE_HIGH_HZ = 3000.0
SAMPLE_RATE = 16000
FFT_SIZE = 512  # PBFDKF default for block_size 256

# EMA + ratio constants (locked at design lock §2.2/§2.4 signing).
ALPHA_OBS = 0.95
RATIO_EPS = 1e-10
T0_WILDLY_HIGH = 3.0
T0_STABLE_LOW = 1.5
T0_DISJOINT_MARGIN = 1.0


def voice_band_mask(n_freqs: int) -> np.ndarray:
    freqs = np.arange(n_freqs, dtype=np.float32) * (SAMPLE_RATE / FFT_SIZE)
    return (freqs >= VOICE_LOW_HZ) & (freqs <= VOICE_HIGH_HZ)


def discover_all_stems() -> list[tuple[str, str, str, str]]:
    """Return [(stem, subset, mic_path, lpb_path), ...] over the 800-case dataset."""
    out = []
    for subset in SUBSETS:
        d = DATASET / subset
        if not d.is_dir():
            continue
        for mic in sorted(d.glob('*_mic.wav')):
            stem = mic.name[:-len('_mic.wav')]
            lpb = d / f'{stem}_lpb.wav'
            if lpb.is_file():
                out.append((stem, subset, str(mic), str(lpb)))
    return out


def find_stem(all_stems, stem: str):
    for tup in all_stems:
        if tup[0] == stem:
            return tup
    return None


def parse_case_list(path: Path, all_stems) -> list[tuple[str, str, str, str, str]]:
    """Return [(category, stem, subset, mic_path, lpb_path), ...]."""
    listener, wildly = [], []
    resolve_stable = False
    seen = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        cat, _, stem = line.partition(':')
        if cat == 'stable' and stem == '_RESOLVE_AT_RUNTIME_':
            resolve_stable = True
            continue
        tup = find_stem(all_stems, stem)
        if tup is None:
            print(f'WARN: stem not found in dataset: {stem}', file=sys.stderr)
            continue
        seen.add(stem)
        (listener if cat == 'listener' else wildly).append((cat, *tup))

    stable = []
    if resolve_stable:
        # Stable baseline: sample STABLE_SAMPLE_SIZE stems from full dataset
        # minus already-named non-stable. Random seed locked at design-lock.
        candidates = [t for t in all_stems if t[0] not in seen]
        rng = random.Random(STABLE_SAMPLE_SEED)
        picks = rng.sample(candidates, STABLE_SAMPLE_SIZE)
        stable = [('stable', *t) for t in picks]

    return listener + wildly + stable


def _run_one(args):
    cat, stem, subset, mic_p, lpb_p, out_dir = args
    np.random.seed(CNG_SEED)
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if lpb.ndim > 1:
        lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    cfg = AecConfig.from_preset(
        PRESET_MAP['balanced'], sample_rate=sr, filter_length=832,
        mode=AecMode.PBFDKF, enable_cng=True, enable_res=True,
        enable_shadow=True, use_res_refactored=False,
        trace_p53_innovation=True,
    )
    a = AEC(cfg)
    hop = a.hop_size
    pos = 0
    while pos + hop <= n:
        a.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    out_path = Path(out_dir) / f'{cat}__{stem}.npz'
    n_frames = a.dump_p53_trace(str(out_path))
    return cat, stem, subset, mic_p, lpb_p, n_frames, str(out_path)


def cmd_snapshot(args):
    all_stems = discover_all_stems()
    print(f'Dataset: {len(all_stems)} stems discovered.', flush=True)
    cases = parse_case_list(Path(args.case_list), all_stems)
    print(f'Audit set: {len(cases)} cases '
          f'(listener={sum(1 for c in cases if c[0]=="listener")}, '
          f'wildly={sum(1 for c in cases if c[0]=="wildly")}, '
          f'stable={sum(1 for c in cases if c[0]=="stable")})', flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist resolved stable picks for reproducibility.
    resolved_path = out_dir / 'resolved_case_list.txt'
    with open(resolved_path, 'w') as f:
        for cat, stem, subset, _, _ in cases:
            f.write(f'{cat}:{stem}:{subset}\n')
    print(f'Resolved case list → {resolved_path}', flush=True)

    jobs = [(cat, stem, subset, mic, lpb, str(out_dir))
            for (cat, stem, subset, mic, lpb) in cases]
    t0 = time.time()
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            for i, res in enumerate(p.imap_unordered(_run_one, jobs)):
                cat, stem, subset, _, _, n_frames, _ = res
                print(f'  [{i+1}/{len(jobs)}] {cat}:{stem} → {n_frames} frames '
                      f'({time.time()-t0:.0f}s)', flush=True)
    else:
        for i, j in enumerate(jobs):
            res = _run_one(j)
            cat, stem, subset, _, _, n_frames, _ = res
            print(f'  [{i+1}/{len(jobs)}] {cat}:{stem} → {n_frames} frames '
                  f'({time.time()-t0:.0f}s)', flush=True)
    print(f'snapshot done in {time.time()-t0:.0f}s', flush=True)


def _r_voice_stats(d, vb_mask):
    """Compute r_voice trajectory + aggregates from one case's npz."""
    inn_pow = d['innovation_power']  # (T, F)
    R = d['R']
    far_psd = d['far_psd']
    P_diag = d['P_diag']
    denom = d['denominator']  # partition-aware Kalman expected variance
    T, F = inn_pow.shape

    # C_obs[k,t] = EMA(innovation_power, alpha=0.95)
    C_obs = np.empty_like(inn_pow)
    s = inn_pow[0].copy()
    C_obs[0] = s
    for t in range(1, T):
        s = ALPHA_OBS * s + (1.0 - ALPHA_OBS) * inn_pow[t]
        C_obs[t] = s

    # Two C_exp variants:
    #   (a) design-lock formula §2.2: far_psd * P_diag + R
    #   (b) partition-aware Kalman expected variance: denominator (= total_echo_var + R + delta)
    C_exp_a = far_psd * P_diag + R
    C_exp_b = denom

    r_a = C_obs / np.maximum(C_exp_a, RATIO_EPS)
    r_b = C_obs / np.maximum(C_exp_b, RATIO_EPS)

    # Far-active mask (per-frame): >1e-5 mean far psd, per regime classifier convention.
    far_active = far_psd.mean(axis=1) > 1e-5

    vb_idx = np.where(vb_mask[:F])[0]
    r_voice_a = r_a[:, vb_idx].mean(axis=1)
    r_voice_b = r_b[:, vb_idx].mean(axis=1)

    active = far_active & np.isfinite(r_voice_a) & np.isfinite(r_voice_b)
    if int(active.sum()) == 0:
        return {
            'n_frames': T,
            'n_far_active': 0,
            'r_voice_design': {'mean': float('nan'), 'median': float('nan'),
                                'p90': float('nan'), 'p99': float('nan')},
            'r_voice_partition': {'mean': float('nan'), 'median': float('nan'),
                                   'p90': float('nan'), 'p99': float('nan')},
        }
    a = r_voice_a[active]
    b = r_voice_b[active]
    return {
        'n_frames': T,
        'n_far_active': int(active.sum()),
        'r_voice_design': {
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p99': float(np.percentile(a, 99)),
        },
        'r_voice_partition': {
            'mean': float(b.mean()), 'median': float(np.median(b)),
            'p90': float(np.percentile(b, 90)), 'p99': float(np.percentile(b, 99)),
        },
    }


def _classify_case(mic_p: str, lpb_p: str) -> str:
    """Whole-recording regime classifier (P52 A.0R.3)."""
    mic, sr = sf.read(mic_p, dtype='float32')
    lpb, _ = sf.read(lpb_p, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if lpb.ndim > 1:
        lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    clf = AcousticRegimeClassifier()
    res = clf.classify(mic[:n], lpb[:n], sample_rate=sr)
    return res.regime.value


def cmd_analyze(args):
    in_dir = Path(args.in_dir)
    resolved = (in_dir / 'resolved_case_list.txt').read_text().splitlines()
    cases = [line.split(':') for line in resolved if line.strip()]
    print(f'Analyzing {len(cases)} cases from {in_dir}', flush=True)

    # Need fft size from one case to build voice band mask.
    first = np.load(in_dir / f'{cases[0][0]}__{cases[0][1]}.npz')
    n_freqs = first['R'].shape[1]
    vb_mask = voice_band_mask(n_freqs)
    print(f'n_freqs={n_freqs}, voice band bins={int(vb_mask.sum())}', flush=True)

    all_stems = discover_all_stems()
    summary = {'cases': {}, 'aggregates': {}}
    for cat, stem, subset in cases:
        npz_path = in_dir / f'{cat}__{stem}.npz'
        if not npz_path.is_file():
            print(f'  MISSING: {npz_path}', file=sys.stderr)
            continue
        d = np.load(npz_path)
        stats = _r_voice_stats(d, vb_mask)
        tup = find_stem(all_stems, stem)
        regime = _classify_case(tup[2], tup[3]) if tup else 'unknown'
        stats['category'] = cat
        stats['subset'] = subset
        stats['regime_class'] = regime
        summary['cases'][stem] = stats
        print(f'  {cat:8s} {regime:25s} {stem[:50]:50s} '
              f'r_voice_design mean={stats["r_voice_design"]["mean"]:.3f} '
              f'p99={stats["r_voice_design"]["p99"]:.3f} '
              f'(part mean={stats["r_voice_partition"]["mean"]:.3f})',
              flush=True)

    # Aggregate by category.
    for cat in ('listener', 'wildly', 'stable'):
        vals_d = [s['r_voice_design']['mean'] for s in summary['cases'].values()
                  if s['category'] == cat and np.isfinite(s['r_voice_design']['mean'])]
        vals_p = [s['r_voice_partition']['mean'] for s in summary['cases'].values()
                  if s['category'] == cat and np.isfinite(s['r_voice_partition']['mean'])]
        if vals_d:
            summary['aggregates'][cat] = {
                'n': len(vals_d),
                'r_voice_design_mean_of_means': float(np.mean(vals_d)),
                'r_voice_design_max_of_means': float(np.max(vals_d)),
                'r_voice_partition_mean_of_means': float(np.mean(vals_p)),
                'r_voice_partition_max_of_means': float(np.max(vals_p)),
            }

    out_json = Path(args.summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'\nsummary JSON → {out_json}', flush=True)
    return summary


def _t0_decide(agg: dict) -> tuple[str, str]:
    """Apply T0 decision tree §2.4 on aggregates {category: {...}}."""
    wildly = agg.get('wildly', {})
    stable = agg.get('stable', {})
    if not wildly or not stable:
        return 'T0.E', 'insufficient category coverage (missing wildly or stable aggregate)'

    w_mean = wildly['r_voice_design_mean_of_means']
    s_mean = stable['r_voice_design_mean_of_means']

    if w_mean >= T0_WILDLY_HIGH and s_mean < T0_STABLE_LOW:
        # Discriminate A vs C using mildly tier if listener proxies mildly.
        listener = agg.get('listener', {})
        l_mean = listener.get('r_voice_design_mean_of_means', float('nan')) if listener else float('nan')
        if np.isfinite(l_mean) and T0_STABLE_LOW <= l_mean < T0_WILDLY_HIGH:
            return 'T0.C', (f'regime-graded innovation: stable {s_mean:.2f} < {T0_STABLE_LOW}, '
                            f'listener (mildly proxy) {l_mean:.2f} in [{T0_STABLE_LOW},{T0_WILDLY_HIGH}), '
                            f'wildly {w_mean:.2f} ≥ {T0_WILDLY_HIGH}')
        return 'T0.A', (f'Q-underestimation confirmed on wildly: '
                        f'wildly {w_mean:.2f} ≥ {T0_WILDLY_HIGH}, stable {s_mean:.2f} < {T0_STABLE_LOW}')

    if w_mean >= T0_WILDLY_HIGH:
        return 'T0.B', (f'wildly innovation elevated ({w_mean:.2f}) but stable not low ({s_mean:.2f}); '
                        f'treat as bin-localized path-change signal')

    if abs(w_mean - 1.0) < 0.5 and abs(s_mean - 1.0) < 0.5:
        return 'T0.D', (f'r_voice ≈ 1 everywhere (wildly {w_mean:.2f}, stable {s_mean:.2f}); '
                        f'no Q-underestimation signal')

    return 'T0.E', (f'ambiguous: wildly {w_mean:.2f}, stable {s_mean:.2f} '
                    f'do not fit A/B/C/D thresholds')


_DECISION_MAP = {
    'T0.A': ('Direction 1 (Adaptive Q)', 'R3 (Lee-Kim SPP) [or R1 if reverb evidence]'),
    'T0.B': ('Direction 2 (Strong Tracking)', 'R2 (Faller-Chen envelope)'),
    'T0.C': ('Direction 4 (IMM bank)', 'R1 (Habets joint dereverb+RES)'),
    'T0.D': ('Direction 3 (VB) OR close Phase L', 'R1 (Habets joint dereverb+RES)'),
    'T0.E': ('close Phase L', 'R3 (Lee-Kim SPP, least risky)'),
}


def cmd_verdict(args):
    summary = json.loads(Path(args.summary).read_text())
    t0_outcome, t0_reason = _t0_decide(summary['aggregates'])
    phase_l, phase_r = _DECISION_MAP[t0_outcome]

    lines = []
    lines.append('# P53 Step 0 — Innovation Audit Verdict')
    lines.append('')
    lines.append(f'**Date**: {time.strftime("%Y-%m-%d")}')
    lines.append(f'**Source data**: `{args.summary}`')
    lines.append('**Audit code path**: production main HEAD, `use_res_refactored=False`, '
                 '`trace_p53_innovation=True`.')
    lines.append('')
    lines.append('## T0 outcome')
    lines.append('')
    lines.append(f'**Outcome**: `{t0_outcome}`')
    lines.append('')
    lines.append(f'**Reason**: {t0_reason}')
    lines.append('')
    lines.append('## Phase commitments (per design lock §2.4)')
    lines.append('')
    lines.append(f'- Phase L → **{phase_l}**')
    lines.append(f'- Phase R → **{phase_r}**')
    lines.append('')
    lines.append('## Aggregate r_voice (design-lock formula `C_exp = far_psd * P_diag + R`)')
    lines.append('')
    lines.append('| Category | n cases | r_voice mean (across cases) | r_voice max |')
    lines.append('|---|---|---|---|')
    for cat in ('stable', 'listener', 'wildly'):
        a = summary['aggregates'].get(cat)
        if not a:
            lines.append(f'| {cat} | 0 | — | — |')
        else:
            lines.append(f'| {cat} | {a["n"]} | {a["r_voice_design_mean_of_means"]:.3f} | '
                         f'{a["r_voice_design_max_of_means"]:.3f} |')
    lines.append('')
    lines.append('## Per-case detail')
    lines.append('')
    lines.append('| Category | Regime (A.0R.3) | Stem | n_frames | n_far_active | '
                 'r_voice mean (design) | p99 (design) | mean (partition) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for stem, s in sorted(summary['cases'].items(),
                           key=lambda kv: (kv[1]['category'], -kv[1]['r_voice_design']['mean']
                                            if np.isfinite(kv[1]['r_voice_design']['mean']) else 0)):
        d = s['r_voice_design']; p = s['r_voice_partition']
        lines.append(f'| {s["category"]} | {s["regime_class"]} | `{stem}` | '
                     f'{s["n_frames"]} | {s["n_far_active"]} | '
                     f'{d["mean"]:.3f} | {d["p99"]:.3f} | {p["mean"]:.3f} |')
    lines.append('')
    lines.append('## Constants (locked at design-lock signing)')
    lines.append('')
    lines.append(f'- `ALPHA_OBS` = {ALPHA_OBS}')
    lines.append(f'- Voice band: {VOICE_LOW_HZ}–{VOICE_HIGH_HZ} Hz')
    lines.append(f'- T0 thresholds: wildly ≥ {T0_WILDLY_HIGH}; stable < {T0_STABLE_LOW}; '
                 f'disjoint margin {T0_DISJOINT_MARGIN}')
    lines.append(f'- Stable sample seed: `random.Random({STABLE_SAMPLE_SEED}).sample(..., {STABLE_SAMPLE_SIZE})`')
    lines.append(f'- CNG seed per case: `np.random.seed({CNG_SEED})`')
    lines.append('')
    lines.append('Verdict immutable post-write.')

    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines) + '\n')
    print(f'verdict → {out_md}')
    print(f'\nT0 outcome: {t0_outcome}')
    print(f'  Phase L: {phase_l}')
    print(f'  Phase R: {phase_r}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('snapshot', help='run AEC w/ trace; dump per-case npz')
    s.add_argument('--case-list', required=True)
    s.add_argument('--out', required=True, help='output dir for per-case .npz')
    s.add_argument('-j', '--jobs', type=int, default=4)
    s.set_defaults(func=cmd_snapshot)

    a = sub.add_parser('analyze', help='compute r_voice aggregates → summary JSON')
    a.add_argument('--in-dir', required=True)
    a.add_argument('--summary', required=True)
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser('verdict', help='apply T0 decision tree → markdown verdict')
    v.add_argument('--summary', required=True)
    v.add_argument('--out', required=True)
    v.set_defaults(func=cmd_verdict)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
