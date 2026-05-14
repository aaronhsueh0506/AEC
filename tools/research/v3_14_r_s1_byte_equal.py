#!/usr/bin/env python3
"""v3.14 Arc-R Sprint S1 — Byte-equal verification for `res_per_band_enr`.

Verifies the R.S1 wire (per-band ENR threshold) is byte-equal to the
BALANCED baseline when `res_per_band_enr=False` (default).

5-case sample (sanity) + optional full 800-case sweep.

Usage:
    python3 tools/research/v3_14_r_s1_byte_equal.py \\
        --dataset-dir /path/to/wav/aec_challenge_blind
    python3 tools/research/v3_14_r_s1_byte_equal.py \\
        --dataset-dir /path/to/wav/aec_challenge_blind --full

Standard config: preset=balanced fl=832 cng=True seed=42
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

from aec import AEC, AecConfig, AecPreset  # noqa: E402


CASES = [
    ('NE',         '014AzuqPZku2004NbTTmcA_nearend_singletalk',
     'nearend_singletalk'),
    ('FS_static',  '0KjzXA3g20qsd8zmSekADw_farend_singletalk',
     'farend_singletalk'),
    ('FS_mvmt',    '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement',
     'farend_singletalk'),
    ('DT_static',  '0I0XMl3M0ECO0U1N0cJvpg_doubletalk',
     'doubletalk'),
    ('DT_mvmt',    '49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement',
     'doubletalk'),
]


def load_wav(path: str) -> np.ndarray:
    data, _ = sf.read(path, dtype='float32')
    return data


def run_case(mic: np.ndarray, ref: np.ndarray,
             per_band_enr: bool, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    # Tolerate baseline runs at parent commit where `res_per_band_enr`
    # doesn't exist on AecConfig — only pass the kwarg when the field is
    # present.
    kw = dict(filter_length=832, enable_cng=True)
    import dataclasses
    if any(f.name == 'res_per_band_enr' for f in dataclasses.fields(AecConfig)):
        kw['res_per_band_enr'] = per_band_enr
    cfg = AecConfig.from_preset(AecPreset.BALANCED, **kw)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = min(len(mic), len(ref))
    n_hops = n // hop
    outputs = []
    for i in range(n_hops):
        mic_hop = mic[i * hop:(i + 1) * hop]
        ref_hop = ref[i * hop:(i + 1) * hop]
        out_hop = aec.process(mic_hop, ref_hop)
        outputs.append(out_hop)
    return np.concatenate(outputs)


def find_wav(dataset_dir: str, stem: str, suffix: str) -> str:
    for sub in ('nearend_singletalk', 'farend_singletalk', 'doubletalk'):
        p = os.path.join(dataset_dir, sub, f'{stem}_{suffix}.wav')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {stem}_{suffix}.wav under {dataset_dir}')


def run_sample(dataset_dir: str) -> bool:
    print('=' * 72)
    print('v3.14 Arc-R Sprint S1 — Byte-equal verification (5-case sample)')
    print('Config: preset=balanced, fl=832, cng=True, seed=42')
    print('=' * 72)
    print()
    all_pass = True
    for (bucket, stem, subdir) in CASES:
        mic_path = find_wav(dataset_dir, stem, 'mic')
        ref_path = find_wav(dataset_dir, stem, 'lpb')
        mic = load_wav(mic_path)
        ref = load_wav(ref_path)

        t0 = time.time()
        out_baseline = run_case(mic, ref, per_band_enr=False)
        t1 = time.time()
        out_flag_off = run_case(mic, ref, per_band_enr=False)
        t2 = time.time()
        out_flag_on = run_case(mic, ref, per_band_enr=True)
        t3 = time.time()

        n = min(len(out_baseline), len(out_flag_off))
        max_diff_off = float(np.max(np.abs(out_baseline[:n] - out_flag_off[:n])))
        byte_equal = np.array_equal(out_baseline[:n], out_flag_off[:n])

        n_on = min(len(out_baseline), len(out_flag_on))
        max_diff_on = float(np.max(np.abs(out_baseline[:n_on] - out_flag_on[:n_on])))

        status = 'PASS' if byte_equal else 'FAIL'
        if not byte_equal:
            all_pass = False

        print(f'[{bucket}]  stem={stem[:32]}')
        print(f'  OFF vs baseline: max|Δ|={max_diff_off:.6e}  → {status}')
        print(f'  ON  vs baseline: max|Δ|={max_diff_on:.6e}  (flag-ON diff, informational)')
        print(f'  Runtime: baseline={t1-t0:.2f}s  off={t2-t1:.2f}s  on={t3-t2:.2f}s')
        print()

    print('=' * 72)
    if all_pass:
        print('SAMPLE RESULT: 5/5 PASS (atol=0.0)')
    else:
        print('SAMPLE RESULT: FAIL — regression in flag-OFF path')
    print('=' * 72)
    return all_pass


def _run_one_full(args):
    """Compare flag-OFF output to a saved baseline .npy file.

    args is (mic_path, ref_path, baseline_path). The caller must have
    pre-rendered the baseline at the parent commit using --gen-baseline.
    """
    mic_path, ref_path, baseline_path = args
    mic = load_wav(mic_path)
    ref = load_wav(ref_path)
    out_off = run_case(mic, ref, per_band_enr=False)
    baseline = np.load(baseline_path)
    n = min(len(out_off), len(baseline))
    return (mic_path, np.array_equal(out_off[:n], baseline[:n]),
            float(np.max(np.abs(out_off[:n] - baseline[:n]))))


def _baseline_path_for(case_dir: str, mic_path: str) -> str:
    rel = os.path.relpath(mic_path)
    safe = rel.replace('/', '__').replace('_mic.wav', '.npy')
    return os.path.join(case_dir, safe)


def _gen_one_baseline(args):
    """Generate baseline output (flag-OFF on current code, expected to be parent commit)."""
    mic_path, ref_path, baseline_path = args
    if os.path.exists(baseline_path):
        return mic_path, 'cached'
    mic = load_wav(mic_path)
    ref = load_wav(ref_path)
    out = run_case(mic, ref, per_band_enr=False)
    np.save(baseline_path, out)
    return mic_path, 'generated'


def _gather_cases(dataset_dir: str):
    cases = []
    for sub in ('nearend_singletalk', 'farend_singletalk', 'doubletalk'):
        sc_dir = os.path.join(dataset_dir, sub)
        if not os.path.isdir(sc_dir):
            continue
        for f in sorted(os.listdir(sc_dir)):
            if f.endswith('_mic.wav'):
                mic_p = os.path.join(sc_dir, f)
                ref_p = mic_p.replace('_mic.wav', '_lpb.wav')
                if os.path.exists(ref_p):
                    cases.append((mic_p, ref_p))
    return cases


def gen_baseline(dataset_dir: str, baseline_dir: str, max_workers: int = 4):
    """Generate baseline .npy outputs from CURRENT code (run on parent commit)."""
    os.makedirs(baseline_dir, exist_ok=True)
    cases = _gather_cases(dataset_dir)
    print(f'gen-baseline: rendering {len(cases)} cases (flag-OFF) → {baseline_dir}')
    t0 = time.time()
    args_list = [(m, r, _baseline_path_for(baseline_dir, m)) for m, r in cases]
    n_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_gen_one_baseline, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures)):
            _ = fut.result()
            n_done += 1
            if n_done % 50 == 0:
                print(f'  progress: {n_done}/{len(cases)} elapsed={time.time()-t0:.1f}s')
    print(f'gen-baseline done in {time.time()-t0:.1f}s')


def run_full(dataset_dir: str, baseline_dir: str, max_workers: int = 4) -> bool:
    print('=' * 72)
    print('v3.14 Arc-R Sprint S1 — Full 800-case byte-equal verification')
    print(f'Config: preset=balanced fl=832 cng=True seed=42  (j={max_workers})')
    print(f'Baseline dir: {baseline_dir}')
    print('=' * 72)

    cases = _gather_cases(dataset_dir)
    print(f'Total cases: {len(cases)}')

    # Check baseline coverage
    missing = []
    for mic_p, _ in cases:
        bp = _baseline_path_for(baseline_dir, mic_p)
        if not os.path.exists(bp):
            missing.append(bp)
    if missing:
        print(f'ERROR: missing {len(missing)} baseline files')
        print(f'  First 3: {missing[:3]}')
        print(f'  Run with --gen-baseline first (on parent commit code)')
        return False

    n_pass = 0
    n_fail = 0
    fails = []
    t0 = time.time()
    args_list = [(m, r, _baseline_path_for(baseline_dir, m)) for m, r in cases]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one_full, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures)):
            mic_p, ok, max_d = fut.result()
            if ok:
                n_pass += 1
            else:
                n_fail += 1
                fails.append((mic_p, max_d))
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f'  progress: {i+1}/{len(cases)}  pass={n_pass}  fail={n_fail}  '
                      f'elapsed={elapsed:.1f}s')

    elapsed = time.time() - t0
    print()
    print('-' * 72)
    print(f'Full 800-case result: {n_pass}/{len(cases)} PASS, '
          f'{n_fail} FAIL  (elapsed {elapsed:.1f}s)')
    if fails:
        print(f'First 5 failures:')
        for p, d in fails[:5]:
            print(f'  {os.path.basename(p)}  max|Δ|={d:.6e}')
    print('=' * 72)
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='wav/aec_challenge_blind')
    parser.add_argument('--full', action='store_true',
                        help='Run full 800-case byte-equal sweep (slow).')
    parser.add_argument('--gen-baseline', action='store_true',
                        help='Render baseline .npy outputs from CURRENT code '
                             '(run while on parent commit / stashed R.S1 changes).')
    parser.add_argument('--baseline-dir', default='/tmp/v3_14_r_s1_baseline',
                        help='Where to read/write per-case baseline .npy files.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-sample', action='store_true')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(_REPO, dataset_dir)
    if not os.path.isdir(dataset_dir):
        cand = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind'
        if os.path.isdir(cand):
            dataset_dir = cand
            print(f'[note] using fallback dataset dir: {dataset_dir}')
        else:
            print(f'ERROR: dataset dir not found: {args.dataset_dir}')
            sys.exit(1)

    if args.gen_baseline:
        gen_baseline(dataset_dir, args.baseline_dir, max_workers=args.workers)
        sys.exit(0)

    if not args.skip_sample:
        sample_ok = run_sample(dataset_dir)
        if not sample_ok:
            print('Sample failed — skipping full sweep.')
            sys.exit(1)

    if args.full:
        full_ok = run_full(dataset_dir, args.baseline_dir,
                           max_workers=args.workers)
        sys.exit(0 if full_ok else 1)
    sys.exit(0)


if __name__ == '__main__':
    main()
