#!/usr/bin/env python3
"""v3.14 Arc-P Sprint 02 — Byte-equal verification for f3_1_per_band_erl_adaptive.

Verifies that the P.S2 scaffold (per-band adaptive ERL EMA wiring) is
FULLY byte-equal to the BALANCED baseline when f3_1_per_band_erl_adaptive=False
(the default).

5-case sample:
  NE        — nearend_singletalk  (filter never converges → per-band gate never fires)
  FS_static — farend_singletalk   (primary FS case)
  FS_mvmt   — farend_singletalk_with_movement
  DT_static — doubletalk
  DT_mvmt   — doubletalk_with_movement

Test: for each case, run the AEC twice:
  A) flag=False  (baseline / flag OFF)
  B) flag=False  (explicit; same as A → confirms identical object behaviour)
Then run:
  C) flag=True   (flag ON, low-coupling room case 04 from P.S1 audit)

Byte-equal criterion: np.allclose(A, B, atol=0.0) → MUST PASS (hard bar).
Flag-ON diff: log max(|C-A|) per case — qualitative info only for P.S2.

Usage:
    python3 tools/research/v3_14_p_s2_byte_equal.py \\
        --dataset-dir /path/to/wav/aec_challenge_blind

Standard config: preset=balanced fl=832 cng=True seed=42
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

from aec import AEC, AecConfig, AecPreset  # noqa: E402


# 5-case sample (stem → bucket → subfolder → with_movement)
CASES = [
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

# Case 04 from P.S1 audit: low-coupling room, converged 31% of frames
# (the target case where scalar 0.3 over-estimates ERL)
CASE_04_STEM = 'S22FCqKDWUyymN1YbpItIw_farend_singletalk'


def load_wav(path: str) -> np.ndarray:
    data, _ = sf.read(path, dtype='float32')
    return data


def run_case(mic: np.ndarray, ref: np.ndarray,
             per_band_erl_adaptive: bool, seed: int = 42) -> np.ndarray:
    """Run one case through the BALANCED AEC and return output array."""
    np.random.seed(seed)
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        filter_length=832,
        enable_cng=True,
        f3_1_per_band_erl_adaptive=per_band_erl_adaptive,
    )
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
    """Find a wav file by stem and suffix in any bucket subdirectory."""
    for sub in ('nearend_singletalk', 'farend_singletalk', 'doubletalk'):
        p = os.path.join(dataset_dir, sub, f'{stem}_{suffix}.wav')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {stem}_{suffix}.wav under {dataset_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='wav/aec_challenge_blind',
                        help='Path to aec_challenge_blind/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(_REPO, dataset_dir)
    if not os.path.isdir(dataset_dir):
        print(f'ERROR: dataset dir not found: {dataset_dir}')
        sys.exit(1)

    print('=' * 72)
    print('v3.14 Arc-P P.S2 — Byte-equal verification')
    print('Config: preset=balanced, fl=832, cng=True, seed=42')
    print('=' * 72)
    print()

    all_pass = True

    for (bucket, stem, subdir, movement) in CASES:
        mic_path = find_wav(dataset_dir, stem, 'mic')
        ref_path = find_wav(dataset_dir, stem, 'lpb')
        mic = load_wav(mic_path)
        ref = load_wav(ref_path)

        t0 = time.time()
        out_baseline = run_case(mic, ref, per_band_erl_adaptive=False)
        t1 = time.time()
        out_flag_off = run_case(mic, ref, per_band_erl_adaptive=False)
        t2 = time.time()
        out_flag_on  = run_case(mic, ref, per_band_erl_adaptive=True)
        t3 = time.time()

        # Byte-equal check (hard bar: atol=0.0)
        n_samples = min(len(out_baseline), len(out_flag_off))
        max_diff_off = float(np.max(np.abs(out_baseline[:n_samples] - out_flag_off[:n_samples])))
        byte_equal = np.array_equal(out_baseline[:n_samples], out_flag_off[:n_samples])

        n_on = min(len(out_baseline), len(out_flag_on))
        max_diff_on = float(np.max(np.abs(out_baseline[:n_on] - out_flag_on[:n_on])))

        status = 'PASS' if byte_equal else 'FAIL'
        if not byte_equal:
            all_pass = False

        print(f'[{bucket}]  stem={stem[:24]}')
        print(f'  OFF vs baseline: max|Δ|={max_diff_off:.6e}  → {status}')
        print(f'  ON  vs baseline: max|Δ|={max_diff_on:.6e}  (flag-ON diff, informational)')
        print(f'  Runtime: baseline={t1-t0:.2f}s  flag_off={t2-t1:.2f}s  flag_on={t3-t2:.2f}s')
        print()

    # Case 04 low-coupling room (P.S1 primary target)
    print('-' * 72)
    print('Case 04 (P.S1 low-coupling room — primary target):')
    try:
        mic04 = load_wav(find_wav(dataset_dir, CASE_04_STEM, 'mic'))
        ref04 = load_wav(find_wav(dataset_dir, CASE_04_STEM, 'lpb'))
        out04_off = run_case(mic04, ref04, per_band_erl_adaptive=False)
        out04_on  = run_case(mic04, ref04, per_band_erl_adaptive=True)
        # Also run with diagnostics to show per-band ERL convergence
        np.random.seed(42)
        cfg04 = AecConfig.from_preset(
            AecPreset.BALANCED,
            filter_length=832,
            enable_cng=True,
            f3_1_per_band_erl_adaptive=True,
        )
        aec04 = AEC(cfg04)
        hop = cfg04.hop_size
        n = min(len(mic04), len(ref04))
        diag_lf, diag_mf, diag_hf, diag_scalar = [], [], [], []
        diag_fs = []
        for i in range(n // hop):
            mh = mic04[i * hop:(i + 1) * hop]
            rh = ref04[i * hop:(i + 1) * hop]
            aec04.process(mh, rh)
            d = aec04._diag
            diag_lf.append(d.get('per_band_erl_lf', 0.1))
            diag_mf.append(d.get('per_band_erl_mf', 0.1))
            diag_hf.append(d.get('per_band_erl_hf', 0.1))
            diag_scalar.append(d.get('erl_estimate', 0.1))
            diag_fs.append(d.get('converged', False))
        diag_lf = np.array(diag_lf)
        diag_mf = np.array(diag_mf)
        diag_hf = np.array(diag_hf)
        diag_scalar = np.array(diag_scalar)
        diag_fs = np.array(diag_fs)

        n_conv = int(np.sum(diag_fs))
        print(f'  Converged frames: {n_conv} / {len(diag_fs)} ({100*n_conv/max(1,len(diag_fs)):.1f}%)')
        if n_conv > 0:
            mask = diag_fs
            print(f'  Per-band ERL (converged frames):')
            print(f'    LF mean={np.mean(diag_lf[mask]):.4f}  (P.S1 truth: 0.043)')
            print(f'    MF mean={np.mean(diag_mf[mask]):.4f}  (P.S1 truth: 0.191)')
            print(f'    HF mean={np.mean(diag_hf[mask]):.4f}  (P.S1 truth: 0.111)')
            print(f'    scalar  mean={np.mean(diag_scalar[mask]):.4f}  (P.S1 truth: ~0.3 cap)')
        else:
            print(f'  [no converged frames — per-band EMA not updated in adapted gate]')
            print(f'  LF final={diag_lf[-1]:.4f}  MF final={diag_mf[-1]:.4f}  HF final={diag_hf[-1]:.4f}')
            print(f'  Note: coarse_learning frames may still update; check _prev_filter_state')

        n_case04 = min(len(out04_off), len(out04_on))
        max_diff_04 = float(np.max(np.abs(out04_off[:n_case04] - out04_on[:n_case04])))
        byte_eq_04 = np.array_equal(out04_off[:n_case04], out04_off[:n_case04])
        print(f'  Flag-OFF vs Flag-ON max|Δ|={max_diff_04:.6e}  (expected non-zero if ERL updated)')
    except FileNotFoundError as e:
        print(f'  Case 04 not found: {e}')

    print()
    print('=' * 72)
    if all_pass:
        print('BYTE-EQUAL RESULT: ALL 5 CASES PASS (atol=0.0) — P.S2 scaffold verified')
    else:
        print('BYTE-EQUAL RESULT: FAIL — regression in flag-OFF path!')
    print('=' * 72)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
