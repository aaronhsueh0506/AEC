#!/usr/bin/env python3
"""Round 6 refactor parity harness.

Runs AEC on a fixed 5-case fixture, captures per-frame output + a small
set of _diag values, and writes them to a pickle. Compare two pickles
(before / after refactor) — every byte must match.

Usage:
    # before refactor
    python3 python/parity_round6.py /tmp/parity_golden.pkl
    # after each stage extraction
    python3 python/parity_round6.py /tmp/parity_check.pkl
    python3 python/parity_round6.py --diff /tmp/parity_golden.pkl /tmp/parity_check.pkl
"""
import os
import sys
import pickle
import argparse
import hashlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from eval_aec_challenge import estimate_delay


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_REPO, 'wav', 'aec_challenge_blind')


# Fixture: representative cases with known characteristics
CASES = [
    ('FS_st_worst', 'farend_singletalk',  '7GTxyTksSUqCnP5y0ILG4A_farend_singletalk',                     False),
    ('FS_mv_loss',  'farend_singletalk',  'Ja8OngfthkOCmL8ldcRNyg_farend_singletalk_with_movement',       True),
    ('DT_mv_worst', 'doubletalk',          'XV5L2dn3S06M9GBEu1q3DA_doubletalk_with_movement',             True),
    ('DT_st_typ',   'doubletalk',          '0I0XMl3M0ECO0U1N0cJvpg_doubletalk',                            False),
    ('NE',          'nearend_singletalk',  '014AzuqPZku2004NbTTmcA_nearend_singletalk',                   False),
]

# Diag fields captured per-frame (cover all stages we plan to refactor)
DIAG_KEYS = [
    'res_gain_mean', 'res_gain_min', 'effective_g_min', 'far_activity',
    'echo_psd_mean', 'error_psd_mean', 'erle_inst', 'converged',
    'erle_factor', 'divergence', 'using_render_based',
    'dt_from_energy', 'dt_from_shadow', 'erl_estimate',
    'epc_active_now', 'epc_hangover_count',
    'p_max_override_active', 'p_floor_beta_active',
    'g_voice_mean', 'g_voice_min',
    'g_stage_softgate_emr_voice', 'g_stage_spectral_floor_voice',
    'g_stage_epc_dt_cap_voice', 'g_stage_quiet_mask_voice',
    'g_stage_3bin_smooth_voice', 'g_stage_hf_cap_voice',
    'g_stage_pre_temporal_voice', 'g_stage_post_temporal_voice',
    'g_stage_after_noise_lift_voice',
]


def run_one(stem, scenario, is_movement):
    mp = os.path.join(_DATA, scenario, stem + '_mic.wav')
    lp = os.path.join(_DATA, scenario, stem + '_lpb.wav')
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED, sample_rate=16000, filter_length=832,
        enable_dtd=False, enable_shadow=True, enable_res=True, enable_cng=True,
        enable_delay_est=is_movement,
        delay_est_period_s=0.25 if is_movement else 1.0,
        delay_est_init_s=0.2 if is_movement else 1.0,
    )
    aec = AEC(cfg)
    np.random.seed(0)
    mic, _ = sf.read(mp); ref, _ = sf.read(lp)
    mic = mic.astype(np.float32); ref = ref.astype(np.float32)
    n = min(len(mic), len(ref)); mic = mic[:n]; ref = ref[:n]
    delay = estimate_delay(mic, ref, 16000)
    if 0 < delay < n:
        ra = np.zeros(n, dtype=np.float32); ra[delay:] = ref[:n - delay]
        ref = ra
    hop = aec.hop_size
    pos = 0
    out = []
    diag_log = []
    while pos + hop <= n:
        o = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        out.append(o)
        # Convert to JSON-friendly snapshot
        snap = {k: aec._diag.get(k) for k in DIAG_KEYS}
        diag_log.append(snap)
        pos += hop
    out_arr = np.concatenate(out).astype(np.float32)
    return out_arr, diag_log


def collect():
    results = {}
    for label, sc, stem, is_mv in CASES:
        out, diag = run_one(stem, sc, is_mv)
        h = hashlib.sha256(out.tobytes()).hexdigest()
        results[label] = {
            'out_hash': h,
            'out_first16': out[:16].tolist(),
            'out_last16': out[-16:].tolist(),
            'out_len': len(out),
            'rms': float(np.sqrt(np.mean(out**2))),
            'max_abs': float(np.max(np.abs(out))),
            'frames': len(diag),
            'diag': diag,
        }
        print(f'  {label}: hash={h[:12]}, rms={results[label]["rms"]:.5f}, '
              f'frames={len(diag)}')
    return results


def diff(golden_path, check_path):
    g = pickle.load(open(golden_path, 'rb'))
    c = pickle.load(open(check_path, 'rb'))
    fail = False
    for label in g:
        if g[label]['out_hash'] == c[label]['out_hash']:
            print(f'  {label}: HASH MATCH')
            continue
        fail = True
        print(f'  {label}: HASH DIFFERS')
        print(f'    rms: golden={g[label]["rms"]:.6f}  check={c[label]["rms"]:.6f}')
        print(f'    max_abs: golden={g[label]["max_abs"]:.6f}  check={c[label]["max_abs"]:.6f}')
        # diff first frame's diag
        gd = g[label]['diag']; cd = c[label]['diag']
        if not gd or not cd: continue
        for k in DIAG_KEYS:
            gv = gd[0].get(k); cv = cd[0].get(k)
            if gv != cv:
                try:
                    delta = abs(gv - cv) if isinstance(gv, (int, float)) else 'n/a'
                except Exception:
                    delta = 'n/a'
                print(f'    diag[{k}] frame0: golden={gv}  check={cv}  Δ={delta}')
                break  # one example per case
    if fail:
        print('PARITY FAIL')
        sys.exit(1)
    print('PARITY PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_path', nargs='?', default=None)
    ap.add_argument('--diff', nargs=2, metavar=('GOLDEN', 'CHECK'))
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff)
        return
    if not args.out_path:
        ap.error('out_path required (or --diff GOLDEN CHECK)')
    print(f'Collecting parity data...')
    results = collect()
    pickle.dump(results, open(args.out_path, 'wb'))
    print(f'Wrote {args.out_path}')


if __name__ == '__main__':
    main()
