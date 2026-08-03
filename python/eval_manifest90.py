#!/usr/bin/env python3
"""Combined ERLE/SDR + AECMOS eval over the fixed, checked-in 90-case manifest.

Background: eval_aec_challenge.py's doubletalk scoring used ONLY an
ERLE-like metric (10*log10(mic_power/output_power)) with no near-end
preservation term -- a run that destroys real near-end speech along with
the echo scores BETTER than one that correctly preserves it. This script
does not remove that ERLE reporting (still useful as a far-end echo-return
measure) but stops treating it as sufficient on its own, per an external
review:

  - far-end singletalk : ERLE (existing)              + AECMOS echo_mos
  - doubletalk          : ERLE (existing)               + AECMOS echo_mos + AECMOS deg_mos
  - near-end singletalk : SDR (existing)                + AECMOS deg_mos

Every category is reported with its static and movement subsets SEPARATE
(never pooled into one mean) -- movement cases have systematically
different delay-tracking behaviour and pooling hides that.

Cases come from a fixed, checked-in manifest (eval/manifest_90case.json,
see eval/build_manifest_90case.py for how it was built) so any future
baseline/candidate run reads the IDENTICAL case list -- 30 far-end / 30
doubletalk / 30 near-end (near-end is all-static; this corpus, upstream,
has no near-end-singletalk movement variant).

Rendering reuses eval_aec_challenge.py's `run_ours` (the exact same AEC
invocation + offline pre-align + online delay-est policy the standard
800-case bench uses) so results are directly comparable to that harness's
`_ours.wav` renders -- this script adds metrics, it does not reimplement
the AEC call.

Usage (standard bench config):
    python3 python/eval_manifest90.py \
        --manifest eval/manifest_90case.json \
        --dataset-dir wav/aec_challenge_blind \
        --preset balanced --filter 832 --cng \
        --parallel --workers 4 \
        -o out_manifest90/ --result-dir results_manifest90/
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_MODEL_DIR = os.path.join(_REPO, 'model')
sys.path.insert(0, _HERE)
sys.path.insert(0, _MODEL_DIR)

import eval_aec_challenge as eac  # run_ours, compute_erle, compute_sdr
from bench_aecmos import FastAECMOS

# ---------------------------------------------------------------------------
# Run-provenance metadata: a run's scores.json is
# only trustworthy/reproducible if it also records EXACTLY what produced the
# numbers -- the resolved signal grid, whether NO_PREALIGN was set, every
# other AEC-relevant env-var override, the fully-resolved AecConfig (not just
# a preset name), and the exact code state (commit + dirty flag). All of this
# is gathered unconditionally in main() below -- never optional/best-effort.
#
# Every env var found (by grep) to gate behavior reachable from run_ours():
#   - eval_aec_challenge.py: NO_PREALIGN + 17 AEC_* campaign/tuning hooks
#     (see run_ours() docstring/body for what each does).
#   - modules/orchestrator.py: R7_EPV_WEAK_THR (EPV weak-filter damping gate,
#     default '0.0' == OFF).
# config.py itself has no os.environ/getenv references.
_AEC_ENV_VARS = [
    'NO_PREALIGN',
    'AEC_MAX_DELAY_MS',
    'AEC_GAIN_TYPE',
    'AEC_ERLE_X2_SCALE',
    'AEC_REVERB_TAIL_STRENGTH',
    'AEC_ERLE_Y2_WIN',
    'AEC_OUT_CAPTURE_UNUSABLE',
    'AEC_DELAY_PROTECT',
    'AEC_SOFT_NE_BLEND',
    'AEC_SOFT_NE_PER_BIN',
    'AEC_ERLE_COH_GATE',
    'AEC_ERLE_COH_GATE_ALPHA',
    'AEC_NL_ALPHA',
    'AEC_FAR_ACTIVE_FLOOR_DB',
    'AEC_ERL_REFRESH_FLOOR',
    'AEC_HERROR_FLOOR',
    'AEC_CFG_OVERRIDE',
    'AEC_MODE',
    'R7_EPV_WEAK_THR',
]


def _config_to_jsonable(config):
    """Every field of a resolved AecConfig, via dataclasses.asdict() (not a
    hand-picked subset) so knobs a preset name alone would hide -- env-var
    overrides, config_overrides, movement-dependent delay-est kwargs -- are
    all captured as the values actually live for this run.
    """
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, Enum):
            d[k] = v.value
    # fft_size / n_partitions / psd_scale are @property, not dataclass
    # fields -- asdict() only walks declared fields, so add them explicitly.
    d['fft_size'] = config.fft_size
    d['n_partitions'] = config.n_partitions
    d['psd_scale'] = config.psd_scale
    return d


def _git_provenance(repo_root):
    """Commit hash + working-tree dirty state for the AEC repo specifically
    (cwd pinned to repo_root regardless of where this script is invoked
    from). Always attempts every command and always returns a populated
    dict -- on any failure (git missing, not a repo, etc.) the 'error' key
    is set instead of the block being silently omitted, so a run never
    ships with unexplained-missing provenance.
    """
    info = {'repo_root': repo_root}

    def _run(args):
        return subprocess.run(args, cwd=repo_root, capture_output=True,
                               text=True, timeout=30)

    try:
        r = _run(['git', 'rev-parse', 'HEAD'])
        info['commit'] = r.stdout.strip() if r.returncode == 0 else None
        r = _run(['git', 'status', '--porcelain'])
        status_lines = [l for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []
        info['dirty'] = bool(status_lines)
        info['changed_file_count'] = len(status_lines)
        r = _run(['git', 'diff', 'HEAD', '--stat'])
        info['diff_stat'] = r.stdout.strip() if r.returncode == 0 else None
        r = _run(['git', 'diff', 'HEAD'])
        diff_text = r.stdout if r.returncode == 0 else ''
        info['diff_sha256'] = (hashlib.sha256(diff_text.encode('utf-8')).hexdigest()
                                if diff_text else None)
    except (OSError, subprocess.SubprocessError) as e:
        info['error'] = f"{type(e).__name__}: {e}"
    return info

TALK_TYPE = {
    'farend_singletalk': 'st',
    'doubletalk': 'dt',
    'nearend_singletalk': 'nst',
}

# Metrics the review's fix mandates per scenario. ERLE/SDR pick one each;
# AECMOS sub-scores can list more than one.
PRIMARY_METRIC = {
    'farend_singletalk': 'erle',
    'doubletalk': 'erle',
    'nearend_singletalk': 'sdr',
}
AECMOS_METRICS = {
    'farend_singletalk': ('echo',),
    'doubletalk': ('echo', 'deg'),
    'nearend_singletalk': ('deg',),
}

_WORKER_ESTIMATOR = {}


def _get_worker_estimator(model_path):
    """Cache the ONNX session per (spawned) worker process."""
    est = _WORKER_ESTIMATOR.get(model_path)
    if est is None:
        est = FastAECMOS(model_path)
        _WORKER_ESTIMATOR[model_path] = est
    return est


def _run_one_case(case_args):
    """Render one case with run_ours, compute ERLE/SDR + AECMOS echo/deg.

    Runs inside a (spawned) worker process -- module-level globals in
    eval_aec_challenge are not inherited from the parent, so _ENABLE_CNG
    is re-set explicitly here (same pattern eval_aec_challenge.py's own
    _run_scenario worker uses).
    """
    (stem, scenario, split, dataset_dir, out_dir, fl, preset_value,
     enable_cng, model_path) = case_args

    eac._ENABLE_CNG = enable_cng

    sc_dir = os.path.join(dataset_dir, scenario)
    mic_path = os.path.join(sc_dir, stem + '_mic.wav')
    lpb_path = os.path.join(sc_dir, stem + '_lpb.wav')

    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(lpb_path)
    mic, ref = mic.astype(np.float32), ref.astype(np.float32)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    movement = (split == 'movement')

    preset_enum = None
    if preset_value is not None:
        from aec import AecPreset
        preset_enum = AecPreset(preset_value)

    _cfg_holder = []
    output = eac.run_ours(mic, ref, sr, fl, preset=preset_enum, is_movement=movement,
                           config_holder=_cfg_holder)
    sf.write(os.path.join(out_dir, f"{stem}_ours.wav"), output, sr)

    result = {'stem': stem, 'scenario': scenario, 'split': split, 'sample_rate': int(sr)}
    # Popped back out in main() to build the deduped run['resolved_config_variants']
    # block -- kept per-case here because _run_one_case runs in a (possibly
    # spawned) worker process and this is the only channel back to main().
    result['_resolved_config'] = _config_to_jsonable(_cfg_holder[0])
    if PRIMARY_METRIC[scenario] == 'erle':
        result['erle'] = eac.compute_erle(mic, output)
    else:
        result['sdr'] = eac.compute_sdr(mic, output)

    estimator = _get_worker_estimator(model_path)
    talk_type = TALK_TYPE[scenario]
    echo_mos, deg_mos = estimator.score(talk_type, ref, mic, output)
    result['echo_mos'] = echo_mos
    result['deg_mos'] = deg_mos
    return result


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    cases = []
    for scenario, splits in manifest.items():
        for split in ('static', 'movement'):
            for stem in splits.get(split, []):
                cases.append((stem, scenario, split))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=os.path.join(_REPO, 'eval', 'manifest_90case.json'))
    ap.add_argument('--dataset-dir', default=os.path.join(_REPO, 'wav', 'aec_challenge_blind'))
    ap.add_argument('--filter', type=int, default=832, help='Filter length (832 = 52ms @16kHz)')
    ap.add_argument('--preset', choices=['mild', 'balanced', 'aggressive'], default='balanced')
    ap.add_argument('--cng', action='store_true', help='Enable comfort noise generation')
    ap.add_argument('--parallel', action='store_true', help='Render cases in parallel')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('-o', '--output-dir', default=os.path.join(_REPO, 'out_manifest90'))
    ap.add_argument('--result-dir', default=os.path.join(_REPO, 'results_manifest90'))
    ap.add_argument('--model', default=os.path.join(_MODEL_DIR, 'Run_1663915512_Stage_0.onnx'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    cases = load_manifest(args.manifest)
    print(f"Loaded {len(cases)} cases from {args.manifest}", file=sys.stderr)

    case_args = [
        (stem, scenario, split, args.dataset_dir, args.output_dir, args.filter,
         args.preset, args.cng, args.model)
        for (stem, scenario, split) in cases
    ]

    results = []
    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one_case, ca): ca[0] for ca in case_args}
            done = 0
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                print(f"  {done}/{len(case_args)}", file=sys.stderr, flush=True)
    else:
        for i, ca in enumerate(case_args):
            results.append(_run_one_case(ca))
            print(f"  {i+1}/{len(case_args)}", file=sys.stderr, flush=True)

    # ---- provenance: dedupe the resolved AecConfig each case actually ran
    # with (movement vs static take different delay-est kwargs inside
    # run_ours, so more than one distinct resolved config can legitimately
    # occur in a single run -- pool by exact field-for-field content, not by
    # assumption, so this never hides a real difference). ----
    config_variants = {}
    for r in results:
        cfg_dict = r.pop('_resolved_config')
        key = json.dumps(cfg_dict, sort_keys=True)
        entry = config_variants.setdefault(key, {'config': cfg_dict, 'used_by_splits': set()})
        entry['used_by_splits'].add(f"{r['scenario']}/{r['split']}")
    resolved_config_variants = [
        {'config': v['config'], 'used_by_splits': sorted(v['used_by_splits'])}
        for v in config_variants.values()
    ]
    grids_seen = sorted({
        (v['config']['sample_rate'], v['config']['fft_size'], v['config']['hop_size'])
        for v in config_variants.values()
    })
    if len(grids_seen) == 1:
        sr0, fft0, hop0 = grids_seen[0]
        grid_summary = {'sample_rate': sr0, 'fft_size': fft0, 'hop_size': hop0,
                         'uniform_across_all_resolved_configs': True}
    else:
        grid_summary = {'uniform_across_all_resolved_configs': False,
                         'grids_seen': [{'sample_rate': a, 'fft_size': b, 'hop_size': c}
                                        for (a, b, c) in grids_seen]}

    run_metadata = {
        'no_prealign': bool(os.environ.get('NO_PREALIGN')),
        'prealign_mode': ('no_prealign_online_delay_est_only'
                           if os.environ.get('NO_PREALIGN')
                           else 'offline_gcc_phat_prealign_plus_movement_online_delay_est'),
        'grid': grid_summary,
        'env_overrides': {k: os.environ.get(k) for k in _AEC_ENV_VARS},
        'resolved_config_variants': resolved_config_variants,
        'git': _git_provenance(_REPO),
    }

    # ---- aggregate: bucket = (scenario, split), never pooled across split ----
    buckets = defaultdict(list)
    for r in results:
        buckets[(r['scenario'], r['split'])].append(r)

    bucket_order = [
        ('farend_singletalk', 'static'), ('farend_singletalk', 'movement'),
        ('doubletalk', 'static'), ('doubletalk', 'movement'),
        ('nearend_singletalk', 'static'), ('nearend_singletalk', 'movement'),
    ]

    summary = {}
    for key in bucket_order:
        rows = buckets.get(key)
        if not rows:
            continue
        scenario, split = key
        entry = {'n': len(rows)}
        if PRIMARY_METRIC[scenario] == 'erle':
            entry['erle_mean'] = float(np.mean([r['erle'] for r in rows]))
        else:
            entry['sdr_mean'] = float(np.mean([r['sdr'] for r in rows]))
        if 'echo' in AECMOS_METRICS[scenario]:
            entry['echo_mos_mean'] = float(np.mean([r['echo_mos'] for r in rows]))
        if 'deg' in AECMOS_METRICS[scenario]:
            entry['deg_mos_mean'] = float(np.mean([r['deg_mos'] for r in rows]))
        summary[f"{scenario}/{split}"] = entry

    out = {
        'config': {
            'preset': args.preset, 'filter': args.filter, 'cng': args.cng,
            'manifest': os.path.abspath(args.manifest),
            'dataset_dir': os.path.abspath(args.dataset_dir),
        },
        # Provenance/reproducibility block -- unconditionally
        # populated every run, never optional/best-effort: exact signal grid,
        # NO_PREALIGN + every other AEC-relevant env override, the fully
        # resolved AecConfig(s) actually live, and the exact code state.
        'run': run_metadata,
        'summary': summary,
        'cases': results,
    }
    scores_path = os.path.join(args.result_dir, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(out, f, indent=2)

    # ---- print report ----
    print()
    print(f"{'='*78}")
    print(f"90-case manifest eval -- preset={args.preset} filter={args.filter} cng={args.cng}")
    print(f"{'='*78}")
    for key in bucket_order:
        label = f"{key[0]}/{key[1]}"
        if label not in summary:
            continue
        s = summary[label]
        parts = [f"n={s['n']}"]
        if 'erle_mean' in s:
            parts.append(f"ERLE={s['erle_mean']:.2f} dB")
        if 'sdr_mean' in s:
            parts.append(f"SDR={s['sdr_mean']:.2f} dB")
        if 'echo_mos_mean' in s:
            parts.append(f"echo_mos={s['echo_mos_mean']:.3f}")
        if 'deg_mos_mean' in s:
            parts.append(f"deg_mos={s['deg_mos_mean']:.3f}")
        print(f"{label:38s} " + '  '.join(parts))
    print(f"{'='*78}")
    print(f"Wrote {scores_path}")
    print(f"Rendered outputs in {args.output_dir}")


if __name__ == '__main__':
    main()
