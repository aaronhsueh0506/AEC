#!/usr/bin/env python3
"""Action coverage analysis: mode / action distribution per subset.

Processes AEC Challenge blind test files, collecting per-frame diag to measure:
  - recovery mode entry rate and active duration
  - SOFT_ASSIST vs HARD_RECOVERY_COPY breakdown
  - Fix E candidate count and err_ratio block rate
  - Budget exhaustion (session count at end of file)

Reports per subset: FS-static, FS-movement, DT-static, DT-movement, NE

Usage:
    python3 coverage_analysis.py [wav_root] [--quick N]

    wav_root  : path to aec_challenge_blind/ (default: ../wav/aec_challenge_blind)
    --quick N : only process first N cases per scenario (sanity check)
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aec import AEC, AecConfig, AecMode

# ── AEC configuration (must match eval_aec_challenge.py) ────────────────────
DEFAULT_FL = 2048
DEFAULT_SR = 16000


def estimate_delay(mic, ref, max_delay_ms=250.0, sr=16000):
    max_delay = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref), 16000)
    if n < 512:
        return 0
    corr = np.correlate(mic[:n], ref[:n], mode='full')
    center = len(corr) // 2
    lo = max(0, center)
    hi = min(len(corr), center + max_delay)
    peak = np.argmax(np.abs(corr[lo:hi])) + lo
    delay = peak - center
    return max(0, int(delay))


def run_ours_with_diag(mic, ref, sr, fl, is_movement=False):
    """Run AEC and return (output, per_frame_diag_list)."""
    n = min(len(mic), len(ref))
    delay = estimate_delay(mic, ref, sr=sr)
    if delay > 0 and delay < n:
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
    else:
        ref_aligned = ref[:n].copy()

    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25,
                     delay_est_init_s=0.2)
                if is_movement else dict(enable_delay_est=False))

    config = AecConfig(sample_rate=sr, mode=AecMode.PBFDKF,
                       filter_length=fl, enable_dtd=False,
                       enable_shadow=True, enable_res=True,
                       use_kalman=True, **delay_kw)
    aec = AEC(config)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    frames = []
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec.process(mic[pos:pos + hop],
                                          ref_aligned[pos:pos + hop])
        if aec._diag:
            frames.append({
                'converged':        int(aec._diag.get('converged', False)),
                'pc_recovery_mode': int(aec._diag.get('pc_recovery_mode', 0)),
                'copy_s2m':         int(aec._diag.get('copy_s2m', False)),
                'soft_assist':      int(aec._diag.get('soft_assist_fired', False)),
                'fixe_session':     int(aec._diag.get('fixe_session_count', 0)),
                'fixe_candidate':   int(aec._diag.get('fixe_candidate_count', 0)),
                'fixe_blocked':     int(aec._diag.get('fixe_blocked_err_ratio', 0)),
                'epc_level':        aec._diag.get('epc_level', 'none'),
            })
        pos += hop
    return out[:n], frames


def aggregate_file(frames):
    """Collapse per-frame list to per-file stats dict."""
    if not frames:
        return {}
    n = len(frames)
    conv = [f['converged'] for f in frames]
    rec  = [f['pc_recovery_mode'] for f in frames]
    sa   = [f['soft_assist'] for f in frames]
    cs2m = [f['copy_s2m'] for f in frames]

    # convergence frame (first frame where converged flips to 1)
    conv_frame = next((i for i, c in enumerate(conv) if c), n)

    # recovery entry count: 0→1 transitions
    rec_entries = sum(1 for i in range(1, len(rec)) if rec[i] == 1 and rec[i-1] == 0)
    # add entry at frame 0 if already in recovery (shouldn't happen but be safe)
    if rec and rec[0] == 1:
        rec_entries += 1

    frames_converged = sum(conv)
    frames_in_recovery = sum(rec)
    soft_assist_count = sum(sa)
    # hard copy = copy_s2m where soft_assist NOT set
    hard_copy_count = sum(1 for f in frames if f['copy_s2m'] and not f['soft_assist'])
    total_copy_events = soft_assist_count + hard_copy_count

    # final cumulative counters (last frame)
    last = frames[-1]
    fixe_session_final = last['fixe_session']
    fixe_candidate_final = last['fixe_candidate']
    fixe_blocked_final = last['fixe_blocked']

    budget_exhausted = (fixe_session_final >= 3)  # _DT_COPY_MAX_SESSION = 3

    pct_conv = 100 * frames_converged / n
    pct_rec  = 100 * frames_in_recovery / n

    return {
        'n_frames': n,
        'conv_frame': conv_frame,
        'pct_converged': pct_conv,
        'frames_in_recovery': frames_in_recovery,
        'pct_in_recovery': pct_rec,
        'rec_entries': rec_entries,
        'soft_assist_count': soft_assist_count,
        'hard_copy_count': hard_copy_count,
        'total_copy_events': total_copy_events,
        'fixe_session_final': fixe_session_final,
        'fixe_candidate_final': fixe_candidate_final,
        'fixe_blocked_final': fixe_blocked_final,
        'budget_exhausted': int(budget_exhausted),
    }


def process_scenario(sc_dir, sc_tag, fl, sr, quick_n):
    """Process one scenario directory, return list of (file_id, is_movement, stats)."""
    mic_files = sorted(f for f in os.listdir(sc_dir)
                       if f'_{sc_tag}' in f and f.endswith('_mic.wav'))
    if quick_n:
        mic_files = mic_files[:quick_n]
    results = []
    for i, mf in enumerate(mic_files):
        prefix = mf.replace('_mic.wav', '')
        file_id = prefix.replace(f'_{sc_tag}', '')
        is_mv = '_with_movement_' in mf
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, mf.replace('_mic.wav', '_lpb.wav'))
        if not os.path.exists(lpb_path):
            continue
        mic, _ = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        _, frames = run_ours_with_diag(mic, ref, sr, fl, is_movement=is_mv)
        stats = aggregate_file(frames)
        results.append((file_id, is_mv, stats))
        if (i + 1) % 20 == 0 or (i + 1) == len(mic_files):
            print(f"  [{i+1}/{len(mic_files)}] {file_id[:24]}…", flush=True)
    return results


def report_subset(label, rows):
    """Print subset stats table."""
    if not rows:
        print(f"  {label}: no cases")
        return
    n = len(rows)

    def mean_field(key):
        vals = [r[key] for _, _, r in rows if r]
        return np.mean(vals) if vals else float('nan')

    pct_rec   = mean_field('pct_in_recovery')
    rec_entry = mean_field('rec_entries')
    sa_count  = mean_field('soft_assist_count')
    hc_count  = mean_field('hard_copy_count')
    budget_ex = sum(r['budget_exhausted'] for _, _, r in rows if r)
    fixe_cand = mean_field('fixe_candidate_final')
    fixe_blk  = mean_field('fixe_blocked_final')

    # fraction of files with at least 1 soft assist
    has_sa = sum(1 for _, _, r in rows if r and r['soft_assist_count'] > 0)
    has_hc = sum(1 for _, _, r in rows if r and r['hard_copy_count'] > 0)

    pct_conv  = mean_field('pct_converged')

    print(f"\n  ── {label} ({n} cases) ──")
    print(f"    converged          : {pct_conv:5.1f}% of frames avg")
    print(f"    recovery active    : {pct_rec:5.1f}% of frames avg | entries/file: {rec_entry:.2f}")
    print(f"    SOFT_ASSIST        : {sa_count:5.1f}/file avg  | files w/ any: {has_sa}/{n} ({100*has_sa/n:.0f}%)")
    print(f"    HARD_RECOVERY_COPY : {hc_count:5.1f}/file avg  | files w/ any: {has_hc}/{n} ({100*has_hc/n:.0f}%)")
    print(f"    budget exhausted   : {budget_ex}/{n} ({100*budget_ex/n:.0f}%)")
    print(f"    FixE candidates    : {fixe_cand:6.1f}/file avg  | err_ratio blocked: {fixe_blk:.1f}/file avg")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wav_root', nargs='?',
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'wav', 'aec_challenge_blind'))
    parser.add_argument('--quick', type=int, default=0,
                        help='Only process first N cases per scenario (default: all)')
    parser.add_argument('--fl', type=int, default=DEFAULT_FL)
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    args = parser.parse_args()

    wav_root = os.path.realpath(args.wav_root)
    fl, sr = args.fl, args.sr
    quick_n = args.quick or None

    SCENARIOS = [
        ('farend_singletalk', 'farend_singletalk', 'FS'),
        ('nearend_singletalk', 'nearend_singletalk', 'NE'),
        ('doubletalk', 'doubletalk', 'DT'),
    ]

    all_results = {}
    for subdir, sc_tag, sc_label in SCENARIOS:
        sc_dir = os.path.join(wav_root, subdir)
        if not os.path.isdir(sc_dir):
            print(f"  Skip {subdir} (not found)")
            continue
        print(f"\n{'='*60}")
        print(f"Processing {sc_label} ({subdir})")
        print(f"{'='*60}")
        rows = process_scenario(sc_dir, sc_tag, fl, sr, quick_n)
        all_results[sc_label] = rows

    print(f"\n\n{'='*70}")
    print("ACTION COVERAGE REPORT — subset breakdown")
    print(f"{'='*70}")

    for sc_label, rows in all_results.items():
        if sc_label in ('FS', 'DT'):
            static  = [(fid, mv, s) for fid, mv, s in rows if not mv]
            movement = [(fid, mv, s) for fid, mv, s in rows if mv]
            report_subset(f"{sc_label}-static",   static)
            report_subset(f"{sc_label}-movement", movement)
        else:
            report_subset(sc_label, rows)

    # ── DT echo/deg diagnostic: how often does SOFT_ASSIST fire in DT? ──────
    if 'DT' in all_results:
        dt_rows = all_results['DT']
        print(f"\n{'='*70}")
        print("DT gap diagnostic: SOFT_ASSIST budget consumption")
        print(f"{'='*70}")
        for label, rows in [('DT-static', [r for r in dt_rows if not r[1]]),
                             ('DT-movement', [r for r in dt_rows if r[1]])]:
            if not rows:
                continue
            # Cases where budget exhausted without a single hard copy
            sa_only = sum(1 for _, _, s in rows
                          if s and s['budget_exhausted'] and s['hard_copy_count'] == 0)
            total = len(rows)
            print(f"  {label}: budget exhausted={sum(s['budget_exhausted'] for _,_,s in rows if s)}/{total}, "
                  f"SA-only exhaustion={sa_only}/{total}")

    print("\nDone.")


if __name__ == '__main__':
    main()
