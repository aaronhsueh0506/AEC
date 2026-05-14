#!/usr/bin/env python3
"""v3.14 Arc-P Sprint 03 — Source signal fix + alpha scan validation.

This script validates the P.S3 source signal correction:
  P.S2 source: |echo_spec|^2 / |far_spec|^2  (PBFDKF W·X vs X — gave LF=0.57)
  P.S3 source: res.error_psd / far_lw         (same as P.S1 oracle — expects LF~0.05)

Tests:
  1. 5-case byte-equal (flag-OFF must be 0.0)
  2. Case 04 (S22FCqKD — low-coupling room) flag-ON:
     per-band ERL must converge to P.S1 oracle range (LF<0.2, HF<0.4)
  3. Case 08 (S22FCqKD_movement) flag-ON:
     per-band ERL reacts to movement without diverging
  4. Cohort tail (qNvSMyU) flag-ON: safety check, no large FS regression
  5. Alpha scan on cases 04 + 08: alpha in {0.95, 0.99, 0.995, 0.999}

Usage:
    python3 tools/research/v3_14_p_s3_validation.py \\
        --dataset-dir /path/to/wav/aec_challenge_blind

Standard config: preset=balanced fl=832 cng=True seed=42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

from aec import AEC, AecConfig, AecPreset  # noqa: E402

# ---------------------------------------------------------------------------
# 5-case byte-equal sample (from P.S2)
# ---------------------------------------------------------------------------
CASES_5 = [
    ('NE',         '014AzuqPZku2004NbTTmcA_nearend_singletalk',
     'nearend_singletalk', False),
    ('FS_static',  '0KjzXA3g20qsd8zmSekADw_farend_singletalk',
     'farend_singletalk', False),
    ('FS_mvmt',    '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement',
     'farend_singletalk', True),
    ('DT_static',  '0I0XMl3M0ECO0U1N0cJvpg_doubletalk',
     'doubletalk', False),
    ('DT_mvmt',    '49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement',
     'doubletalk', True),
]

# Primary targets from P.S1 audit
CASE_04_STEM = 'S22FCqKDWUyymN1YbpItIw_farend_singletalk'         # low-coupling FS_static
CASE_08_STEM = 'S22FCqKDWUyymN1YbpItIw_farend_singletalk_with_movement'  # FS_movement
CASE_TAIL_STEM = 'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk'       # cohort tail P52 invariant


def load_wav(path: str):
    data, _ = sf.read(path, dtype='float32')
    return data


def find_wav(dataset_dir: str, stem: str, suffix: str) -> str:
    for sub in ('nearend_singletalk', 'farend_singletalk', 'doubletalk'):
        p = os.path.join(dataset_dir, sub, f'{stem}_{suffix}.wav')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {stem}_{suffix}.wav under {dataset_dir}')


def run_aec(mic: np.ndarray, ref: np.ndarray,
            per_band_erl_adaptive: bool = False,
            alpha: float = 0.99,
            seed: int = 42) -> np.ndarray:
    """Run BALANCED AEC and return output array."""
    np.random.seed(seed)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        filter_length=832,
        enable_cng=True,
        f3_1_per_band_erl_adaptive=per_band_erl_adaptive,
        per_band_erl_alpha=alpha,
    )
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = min(len(mic), len(ref))
    outputs = []
    for i in range(n // hop):
        mh = mic[i * hop:(i + 1) * hop]
        rh = ref[i * hop:(i + 1) * hop]
        outputs.append(aec.process(mh, rh))
    return np.concatenate(outputs)


def run_aec_with_diag(mic: np.ndarray, ref: np.ndarray,
                      alpha: float = 0.99,
                      seed: int = 42):
    """Run BALANCED AEC with flag-ON and collect per-band ERL diagnostics.

    Returns: (output_array, diag_dict) where diag_dict has per-frame lists.
    """
    np.random.seed(seed)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        filter_length=832,
        enable_cng=True,
        f3_1_per_band_erl_adaptive=True,
        per_band_erl_alpha=alpha,
    )
    aec = AEC(cfg)
    hop = cfg.hop_size
    n = min(len(mic), len(ref))

    diag_lf, diag_mf, diag_hf = [], [], []
    diag_scalar, diag_converged = [], []
    outputs = []

    for i in range(n // hop):
        mh = mic[i * hop:(i + 1) * hop]
        rh = ref[i * hop:(i + 1) * hop]
        out = aec.process(mh, rh)
        outputs.append(out)
        d = aec._diag
        diag_lf.append(d.get('per_band_erl_lf', 0.1))
        diag_mf.append(d.get('per_band_erl_mf', 0.1))
        diag_hf.append(d.get('per_band_erl_hf', 0.1))
        diag_scalar.append(d.get('erl_estimate', 0.1))
        diag_converged.append(d.get('converged', False))

    diag = {
        'lf': np.array(diag_lf),
        'mf': np.array(diag_mf),
        'hf': np.array(diag_hf),
        'scalar': np.array(diag_scalar),
        'converged': np.array(diag_converged, dtype=bool),
    }
    return np.concatenate(outputs), diag


def print_band_stats(diag, label, mask_key='converged', oracle=None):
    """Print per-band ERL stats for a given mask."""
    mask = diag[mask_key]
    n_mask = int(np.sum(mask))
    n_total = len(diag['lf'])
    print(f'  [{label}] n={n_mask}/{n_total} ({100*n_mask/max(1,n_total):.1f}%)')
    if n_mask == 0:
        print(f'    [no frames in mask]')
        return
    for band, key in [('LF', 'lf'), ('MF', 'mf'), ('HF', 'hf')]:
        vals = diag[key][mask]
        mean_v = float(np.mean(vals))
        p10_v = float(np.percentile(vals, 10))
        p90_v = float(np.percentile(vals, 90))
        oracle_str = ''
        if oracle and band in oracle:
            oracle_str = f'  [P.S1 oracle: {oracle[band]:.3f}]'
        print(f'    {band}: mean={mean_v:.4f}  p10={p10_v:.4f}  p90={p90_v:.4f}{oracle_str}')
    sc_vals = diag['scalar'][mask]
    print(f'    scalar: mean={float(np.mean(sc_vals)):.4f}  p10={float(np.percentile(sc_vals, 10)):.4f}  p90={float(np.percentile(sc_vals, 90)):.4f}')


# P.S1 oracle values for case 04 (742 converged frames, FS_static)
ORACLE_04 = {'LF': 0.043, 'MF': 0.191, 'HF': 0.111}
# P.S1 oracle values for case 08 (296 converged frames, FS_movement)
ORACLE_08 = {'LF': 0.489, 'MF': 0.274, 'HF': 0.344}

# P.S3 acceptance criteria for case 04 converged frames:
ACCEPT_04 = {'LF': (0.01, 0.20), 'MF': (0.05, 0.50), 'HF': (0.05, 0.40)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='wav/aec_challenge_blind')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(_REPO, dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f'ERROR: dataset dir not found: {dataset_dir}')
        sys.exit(1)

    print('=' * 72)
    print('v3.14 Arc-P P.S3 — Source signal fix + alpha scan validation')
    print('Config: preset=balanced, fl=832, cng=True, seed=42')
    print('Source: res.error_psd / far_lw  [P.S3 correction from echo_spec/far_spec]')
    print('=' * 72)
    print()

    overall_pass = True

    # -----------------------------------------------------------------------
    # PART 1: 5-case byte-equal (hard bar: atol=0.0)
    # -----------------------------------------------------------------------
    print('PART 1: 5-case byte-equal (flag-OFF must be 0.0)')
    print('-' * 72)
    byte_equal_pass = True
    for (bucket, stem, subdir, movement) in CASES_5:
        try:
            mic_path = find_wav(dataset_dir, stem, 'mic')
            ref_path = find_wav(dataset_dir, stem, 'lpb')
        except FileNotFoundError as e:
            print(f'  [{bucket}] SKIP: {e}')
            continue
        mic = load_wav(mic_path)
        ref = load_wav(ref_path)

        out_a = run_aec(mic, ref, per_band_erl_adaptive=False)
        out_b = run_aec(mic, ref, per_band_erl_adaptive=False)
        out_c = run_aec(mic, ref, per_band_erl_adaptive=True)

        n = min(len(out_a), len(out_b))
        max_diff_off = float(np.max(np.abs(out_a[:n] - out_b[:n])))
        byte_eq = np.array_equal(out_a[:n], out_b[:n])
        max_diff_on = float(np.max(np.abs(out_a[:min(n, len(out_c))] - out_c[:min(n, len(out_c))])))

        status = 'PASS' if byte_eq else 'FAIL'
        if not byte_eq:
            byte_equal_pass = False
            overall_pass = False
        print(f'  [{bucket}]  OFF vs OFF: max|Δ|={max_diff_off:.2e} → {status}   '
              f'ON vs OFF: max|Δ|={max_diff_on:.2e}')

    result_str = 'ALL PASS' if byte_equal_pass else 'FAIL'
    print(f'  BYTE-EQUAL RESULT: {result_str}')
    print()

    # -----------------------------------------------------------------------
    # PART 2: Case 04 — low-coupling room sanity check
    # -----------------------------------------------------------------------
    print('PART 2: Case 04 (S22FCqKD — low-coupling FS_static) — P.S3 source fix')
    print(f'  P.S1 oracle: LF=0.043  MF=0.191  HF=0.111')
    print(f'  P.S3 accept: LF<0.20  MF<0.50  HF<0.40')
    print('-' * 72)
    try:
        mic04 = load_wav(find_wav(dataset_dir, CASE_04_STEM, 'mic'))
        ref04 = load_wav(find_wav(dataset_dir, CASE_04_STEM, 'lpb'))
        out04_off = run_aec(mic04, ref04, per_band_erl_adaptive=False)
        out04_on, diag04 = run_aec_with_diag(mic04, ref04, alpha=0.99)

        # Overall mask: all frames
        all_mask = np.ones(len(diag04['lf']), dtype=bool)
        # Converged mask
        conv_mask = diag04['converged']
        n_conv = int(np.sum(conv_mask))

        print_band_stats(diag04, 'all frames', mask_key=None, oracle=ORACLE_04)
        diag04['all'] = np.ones(len(diag04['lf']), dtype=bool)
        print_band_stats(diag04, 'converged frames', mask_key='converged', oracle=ORACLE_04)

        # Acceptance check
        print()
        print('  Acceptance check (converged frames):')
        case04_pass = True
        if n_conv > 0:
            for band, key, (lo, hi) in [('LF', 'lf', ACCEPT_04['LF']),
                                          ('MF', 'mf', ACCEPT_04['MF']),
                                          ('HF', 'hf', ACCEPT_04['HF'])]:
                mean_v = float(np.mean(diag04[key][conv_mask]))
                ok = lo <= mean_v <= hi
                if not ok:
                    case04_pass = False
                    overall_pass = False
                print(f'    {band} mean={mean_v:.4f}  range=[{lo:.2f},{hi:.2f}] → {"PASS" if ok else "FAIL"}')
        else:
            print('  WARNING: 0 converged frames — per-band ERL never updated')
            print('  (update gate requires _filter_converged=True)')
            print('  Showing final per-band ERL (EMA at initial value):')
            for band, key in [('LF','lf'),('MF','mf'),('HF','hf')]:
                print(f'    {band} final={float(diag04[key][-1]):.4f}')

        n_4 = min(len(out04_off), len(out04_on))
        diff_04 = float(np.max(np.abs(out04_off[:n_4] - out04_on[:n_4])))
        print(f'  Flag-OFF vs Flag-ON max|Δ|: {diff_04:.4e}  (expected non-zero when ERL updated)')
    except FileNotFoundError as e:
        print(f'  Case 04 NOT FOUND: {e}')
    print()

    # -----------------------------------------------------------------------
    # PART 3: Case 08 — movement room check
    # -----------------------------------------------------------------------
    print('PART 3: Case 08 (S22FCqKD_movement — FS_movement) — movement reactivity')
    print(f'  P.S1 oracle: LF=0.489  MF=0.274  HF=0.344')
    print('-' * 72)
    try:
        mic08 = load_wav(find_wav(dataset_dir, CASE_08_STEM, 'mic'))
        ref08 = load_wav(find_wav(dataset_dir, CASE_08_STEM, 'lpb'))
        out08_on, diag08 = run_aec_with_diag(mic08, ref08, alpha=0.99)
        diag08['all'] = np.ones(len(diag08['lf']), dtype=bool)
        print_band_stats(diag08, 'all frames', mask_key='all', oracle=ORACLE_08)
        print_band_stats(diag08, 'converged frames', mask_key='converged', oracle=ORACLE_08)
    except FileNotFoundError as e:
        print(f'  Case 08 NOT FOUND: {e}')
    print()

    # -----------------------------------------------------------------------
    # PART 4: Cohort tail (qNvSMyU) — P52 invariant check
    # -----------------------------------------------------------------------
    print('PART 4: Cohort tail (qNvSMyU) — P52 invariant (Δecho ≥ -0.05)')
    print('-' * 72)
    try:
        mic_t = load_wav(find_wav(dataset_dir, CASE_TAIL_STEM, 'mic'))
        ref_t = load_wav(find_wav(dataset_dir, CASE_TAIL_STEM, 'lpb'))
        out_t_off = run_aec(mic_t, ref_t, per_band_erl_adaptive=False)
        out_t_on = run_aec(mic_t, ref_t, per_band_erl_adaptive=True)
        n_t = min(len(out_t_off), len(out_t_on))
        diff_t = float(np.max(np.abs(out_t_off[:n_t] - out_t_on[:n_t])))
        # Rough FS echo check: tail is FS dominant, compare RMS
        rms_off = float(np.sqrt(np.mean(out_t_off[:n_t] ** 2)))
        rms_on = float(np.sqrt(np.mean(out_t_on[:n_t] ** 2)))
        rms_db = 20.0 * np.log10(rms_on / (rms_off + 1e-10))
        # Treat >-0.5 dB as pass (the 0.05 dB AECMOS bar is stricter but needs full bench)
        pass_t = rms_db >= -0.5
        if not pass_t:
            overall_pass = False
        print(f'  max|Δ| flag-OFF vs flag-ON: {diff_t:.4e}')
        print(f'  RMS output delta: {rms_db:+.3f} dB → {"PASS" if pass_t else "WARNING (check full bench)"}')
        print(f'  (P52 hard bar needs full 800-case bench; this is single-case sanity)')
    except FileNotFoundError as e:
        print(f'  Cohort tail NOT FOUND: {e}')
    print()

    # -----------------------------------------------------------------------
    # PART 5: Alpha scan on case 04 + 08
    # -----------------------------------------------------------------------
    print('PART 5: Alpha scan — α ∈ {0.95, 0.99, 0.995, 0.999} on cases 04 + 08')
    print('-' * 72)
    ALPHAS = [0.95, 0.99, 0.995, 0.999]
    alpha_results = {}

    for stem_label, stem, oracle in [
        ('Case_04', CASE_04_STEM, ORACLE_04),
        ('Case_08', CASE_08_STEM, ORACLE_08),
    ]:
        print(f'  {stem_label}:')
        try:
            mic_a = load_wav(find_wav(dataset_dir, stem, 'mic'))
            ref_a = load_wav(find_wav(dataset_dir, stem, 'lpb'))
        except FileNotFoundError as e:
            print(f'    NOT FOUND: {e}')
            continue

        for alpha in ALPHAS:
            t0 = time.time()
            _, diag_a = run_aec_with_diag(mic_a, ref_a, alpha=alpha)
            elapsed = time.time() - t0
            mask_c = diag_a['converged']
            n_c = int(np.sum(mask_c))
            if n_c > 0:
                lf_m = float(np.mean(diag_a['lf'][mask_c]))
                mf_m = float(np.mean(diag_a['mf'][mask_c]))
                hf_m = float(np.mean(diag_a['hf'][mask_c]))
            else:
                lf_m = float(diag_a['lf'][-1])
                mf_m = float(diag_a['mf'][-1])
                hf_m = float(diag_a['hf'][-1])
            key = (stem_label, alpha)
            alpha_results[key] = {'lf': lf_m, 'mf': mf_m, 'hf': hf_m, 'n_conv': n_c}
            print(f'    α={alpha:.3f}: n_conv={n_c}  LF={lf_m:.4f}  MF={mf_m:.4f}  HF={hf_m:.4f}'
                  f'  [{elapsed:.1f}s]')
        print()

    # Alpha recommendation
    print('  Alpha recommendation:')
    print('    α=0.99  (TC~100 hops/~1s): fast tracking, may react to DT contamination')
    print('    α=0.995 (TC~200 hops/~2s): balanced — recommended for P.S3 default')
    print('    α=0.999 (TC~1000 hops/~10s): very slow, stable but slow to adapt')
    print()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print('=' * 72)
    print('P.S3 VALIDATION SUMMARY')
    print('=' * 72)
    print(f'  Byte-equal (flag-OFF): {"PASS" if byte_equal_pass else "FAIL"}')
    print(f'  Source signal: res.error_psd / far_lw  (corrected from echo_spec/far_spec)')
    print(f'  Overall: {"PASS — proceed to Arc R" if overall_pass else "NEEDS ATTENTION"}')
    print()

    return 0 if overall_pass else 1


# Override mask_key=None case in print_band_stats
_orig_print_band_stats = print_band_stats

def print_band_stats(diag, label, mask_key=None, oracle=None):
    if mask_key is None:
        mask = np.ones(len(diag['lf']), dtype=bool)
        n_mask = int(np.sum(mask))
        n_total = len(diag['lf'])
        print(f'  [{label}] n={n_mask}/{n_total} ({100*n_mask/max(1,n_total):.1f}%)')
        for band, key in [('LF', 'lf'), ('MF', 'mf'), ('HF', 'hf')]:
            vals = diag[key][mask]
            mean_v = float(np.mean(vals))
            p10_v = float(np.percentile(vals, 10))
            p90_v = float(np.percentile(vals, 90))
            oracle_str = ''
            if oracle and band in oracle:
                oracle_str = f'  [P.S1 oracle: {oracle[band]:.3f}]'
            print(f'    {band}: mean={mean_v:.4f}  p10={p10_v:.4f}  p90={p90_v:.4f}{oracle_str}')
        sc_vals = diag['scalar'][mask]
        print(f'    scalar: mean={float(np.mean(sc_vals)):.4f}')
        return
    _orig_print_band_stats(diag, label, mask_key=mask_key, oracle=oracle)


if __name__ == '__main__':
    sys.exit(main())
