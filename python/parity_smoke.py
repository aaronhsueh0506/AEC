"""
Bit-exact parity smoke test for refactor work.

Runs AEC on a fixed 5-case set covering FS / DT / movement-DT / movement-FS / NE,
saves output samples, compares to a stored baseline (numpy npz of int16 samples).

Workflow:
  1. On v2.8.1 (pre-refactor):  python3 parity_smoke.py --gen-baseline
  2. After refactor:            python3 parity_smoke.py
     -> exits 0 if bit-exact, prints first divergent frame index per case otherwise.
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
BASELINE = Path(__file__).parent / 'parity_baseline.npz'
BASELINE_FULL = Path(__file__).parent / 'parity_baseline_full.npz'

# Fixed 5-case smoke set covering the four relevant scenario classes.
CASES = [
    ('farend_singletalk',                '0KjzXA3g20qsd8zmSekADw_farend_singletalk'),
    ('farend_singletalk',                '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement'),
    ('doubletalk',                       '0I0XMl3M0ECO0U1N0cJvpg_doubletalk'),
    ('doubletalk',                       '49IIo03GZ0CYQOmeA3A0BA_doubletalk_with_movement'),
    ('nearend_singletalk',               None),  # auto-pick first
]


def _all_cases():
    """All 800 cases across the three scenario subdirs."""
    out = []
    for subdir in ('farend_singletalk', 'doubletalk', 'nearend_singletalk'):
        d = WAV_BASE / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.name.endswith('_mic.wav'):
                stem = f.name[:-len('_mic.wav')]
                out.append((subdir, stem))
    return out


def _resolve_case(subdir, stem):
    d = WAV_BASE / subdir
    if stem is None:
        # auto-pick: first *_mic.wav minus suffix
        for f in sorted(d.iterdir()):
            if f.name.endswith('_mic.wav'):
                stem = f.name[:-len('_mic.wav')]
                break
    mic = d / f'{stem}_mic.wav'
    lpb = d / f'{stem}_lpb.wav'
    return stem, mic, lpb


def run_one(mic_path, lpb_path):
    mic, sr = sf.read(str(mic_path), dtype='float32')
    lpb, _ = sf.read(str(lpb_path), dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if lpb.ndim > 1:
        lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    is_movement = '_with_movement' in mic_path.name
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_movement else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset(
        'balanced', sample_rate=sr, mode=AecMode.PBFDKF,
        enable_dtd=False, enable_shadow=True, enable_res=True, use_kalman=True,
        enable_cng=False,  # disable comfort noise (np.random) for deterministic parity
        **delay_kw)
    aec = AEC(cfg)

    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    return out[:pos]


def _run_pair(args):
    subdir, stem = args
    d = WAV_BASE / subdir
    out = run_one(d / f'{stem}_mic.wav', d / f'{stem}_lpb.wav')
    return stem, out


def main():
    gen = '--gen-baseline' in sys.argv
    full = '--full' in sys.argv  # all 800 cases (parity_baseline_full.npz)
    n_jobs = 1
    for i, a in enumerate(sys.argv):
        if a == '-j' and i + 1 < len(sys.argv):
            n_jobs = int(sys.argv[i + 1])

    if full:
        cases = _all_cases()
        baseline_path = BASELINE_FULL
    else:
        cases = [(s, h) for (s, h) in CASES]
        baseline_path = BASELINE

    results = {}
    if n_jobs > 1 and full:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = []
            for subdir, stem_hint in cases:
                stem, mic_p, lpb_p = _resolve_case(subdir, stem_hint)
                if not mic_p.exists():
                    continue
                futs.append(ex.submit(_run_pair, (subdir, stem)))
            done = 0
            for fu in as_completed(futs):
                stem, out = fu.result()
                results[stem] = out
                done += 1
                if done % 50 == 0:
                    print(f'  [{done}/{len(futs)}] processed', flush=True)
    else:
        for subdir, stem_hint in cases:
            stem, mic_p, lpb_p = _resolve_case(subdir, stem_hint)
            if not mic_p.exists():
                print(f'SKIP missing: {mic_p}')
                continue
            out = run_one(mic_p, lpb_p)
            results[stem] = out

    if gen:
        np.savez(baseline_path, **{k: v for k, v in results.items()})
        print(f'baseline saved: {baseline_path}  ({len(results)} cases)')
        return 0

    if not baseline_path.exists():
        print(f'ERROR: no baseline at {baseline_path}. Run with --gen-baseline first.')
        return 2

    ref = np.load(baseline_path)
    fail = 0
    ok = 0
    diff_summary = []
    for stem, out in results.items():
        if stem not in ref:
            print(f'NEW (no ref): {stem}')
            continue
        r = ref[stem]
        n = min(len(r), len(out))
        diff = np.abs(out[:n] - r[:n])
        if diff.max() == 0.0:
            ok += 1
            if not full:
                print(f'OK   {stem}  ({n} samples)')
        else:
            fail += 1
            first = int(np.argmax(diff > 0))
            diff_summary.append((stem, first, float(diff.max()), float(diff.mean())))
            if not full:
                print(f'DIFF {stem}  first_idx={first}  max={diff.max():.3e}  mean={diff.mean():.3e}')
    if full:
        print(f'\n{ok}/{len(results)} bit-exact')
        for stem, first, mx, mn in diff_summary[:20]:
            print(f'  DIFF {stem}  first={first}  max={mx:.3e}  mean={mn:.3e}')
        if len(diff_summary) > 20:
            print(f'  ... and {len(diff_summary)-20} more')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
