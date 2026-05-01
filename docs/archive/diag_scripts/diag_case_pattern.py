"""Characterize temporal pattern of worst-gap cases.

For each stem, computes per-second windowed RMS of mic and lpb,
identifies far-active / near-active intervals, and classifies the
case by overlap pattern: continuous-DT-from-start, late-NE-onset,
NE-only-after-far-stops, gain-jump, etc.

Usage:
  python3 diag_case_pattern.py STEM [STEM ...]
"""
import os, sys
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'

WIN_S = 0.5  # 500ms window
NEAR_DB = -45  # mic active threshold
FAR_DB  = -45


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _envelopes(stem):
    sub, mp, lp = _resolve(stem)
    mic, sr = sf.read(str(mp), dtype='float32')
    lpb, _  = sf.read(str(lp), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb)); mic, lpb = mic[:n], lpb[:n]
    w = int(WIN_S * sr); steps = n // w
    if steps == 0: return sub, sr, [], [], 0
    mic_rms = np.array([np.sqrt(np.mean(mic[i*w:(i+1)*w] ** 2) + 1e-12) for i in range(steps)])
    lpb_rms = np.array([np.sqrt(np.mean(lpb[i*w:(i+1)*w] ** 2) + 1e-12) for i in range(steps)])
    mic_db = 20 * np.log10(mic_rms + 1e-12)
    lpb_db = 20 * np.log10(lpb_rms + 1e-12)
    return sub, sr, mic_db, lpb_db, n / sr


def _classify(sub, mic_db, lpb_db, dur):
    near = mic_db > NEAR_DB
    far  = lpb_db > FAR_DB
    # estimate near-only by removing far-correlated speech: rough mic energy when far quiet
    near_only_when_far_off = (near & ~far).sum()
    dt_overlap = (near & far).sum()
    far_only = (far & ~near).sum()
    silent = (~near & ~far).sum()
    # find onset frames
    def first_true(x):
        idxs = np.where(x)[0]; return idxs[0] if len(idxs) else -1
    far_start = first_true(far)
    near_start = first_true(near)
    # gain steps in lpb (large rise across windows)
    if len(lpb_db) > 4:
        d = np.diff(lpb_db)
        max_jump_up = float(np.max(d[far[1:]])) if far[1:].sum() > 0 else 0.0
        max_jump_dn = float(np.min(d[far[1:]])) if far[1:].sum() > 0 else 0.0
    else:
        max_jump_up = max_jump_dn = 0.0
    # continuous DT from start
    early_window = (near & far)[:max(2, len(near)//8)]  # first 12.5%
    cont_dt_start = bool(early_window.all() if len(early_window) else False)
    # late NE onset: near_start - far_start > ~3s
    late_ne = (far_start >= 0 and near_start >= 0
               and (near_start - far_start) * WIN_S > 3.0)
    flags = []
    if cont_dt_start: flags.append('CONT-DT-FROM-START')
    if late_ne:       flags.append(f'LATE-NE(+{(near_start-far_start)*WIN_S:.1f}s)')
    if max_jump_up > 8:  flags.append(f'GAIN+{max_jump_up:.0f}dB')
    if max_jump_dn < -8: flags.append(f'GAIN{max_jump_dn:.0f}dB')
    return {
        'flags': flags,
        'far_start_s': far_start * WIN_S if far_start >= 0 else -1,
        'near_start_s': near_start * WIN_S if near_start >= 0 else -1,
        'dt_overlap_pct': dt_overlap * 100 / max(1, len(near)),
        'far_only_pct':   far_only   * 100 / max(1, len(near)),
        'near_only_pct':  near_only_when_far_off * 100 / max(1, len(near)),
    }


def main():
    stems = sys.argv[1:]
    for s in stems:
        sub, sr, mic_db, lpb_db, dur = _envelopes(s)
        if len(mic_db) == 0:
            print(f'{s}: empty'); continue
        info = _classify(sub, mic_db, lpb_db, dur)
        print(f'\n{s} ({sub}, {dur:.1f}s)')
        print(f'  far_start={info["far_start_s"]:.1f}s  near_start={info["near_start_s"]:.1f}s')
        print(f'  DT-overlap={info["dt_overlap_pct"]:.0f}%  far-only={info["far_only_pct"]:.0f}%  '
              f'near-only={info["near_only_pct"]:.0f}%')
        print(f'  flags: {info["flags"] or ["normal"]}')
        # mini ASCII timeline (32 cells)
        cells = 32
        if len(mic_db) >= cells:
            step = len(mic_db) / cells
            line = ''
            for i in range(cells):
                a, b = int(i*step), max(int(i*step)+1, int((i+1)*step))
                near = (mic_db[a:b] > NEAR_DB).any()
                far  = (lpb_db[a:b] > FAR_DB).any()
                line += 'D' if (near and far) else ('F' if far else ('N' if near else '.'))
            print(f'  timeline: |{line}|  (D=DT  F=far  N=near  .=silent)')


if __name__ == '__main__':
    main()
