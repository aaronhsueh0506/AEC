#!/usr/bin/env python3
"""Phase 0 trace runner — dumps per-case AEC3-state activations.

Walks the blind dataset, runs AEC v3.8.1 (with Phase 0 trace fields),
and records per-case statistics:
- initial_state_pct
- dominant_nearend_pct
- usable_linear_v1_pct / v2_pct
- transitions (count)
- erle_reset_dist {0,1,2,3}
- final flags (once_converged, far_active_blocks, erl_estimate)

No audio output; no AECMOS scoring. Output: experiments/aec3_phase0/states.json.

Usage:
    python3 trace_phase0.py wav/aec_challenge_blind experiments/aec3_phase0
"""
import os
import sys
import json
import argparse
import glob

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from eval_aec_challenge import estimate_delay


def _run_case(args):
    mp, lp, is_mv, bucket, stem = args
    np.random.seed(0)
    try:
        stats = run_one(mp, lp, is_movement=is_mv)
    except Exception as e:
        return stem, None, str(e)
    stats['bucket'] = bucket
    stats['is_movement'] = is_mv
    return stem, stats, None


def classify(filename):
    """Map filename → (bucket, scenario_dir). `_with_movement` may or may not
    have a trailing underscore depending on what's been stripped already."""
    if 'farend_singletalk' in filename:
        bucket = 'FS_movement' if 'with_movement' in filename else 'FS_static'
        return bucket, 'farend_singletalk'
    if 'nearend_singletalk' in filename:
        return 'NE', 'nearend_singletalk'
    if 'doubletalk' in filename:
        bucket = 'DT_movement' if 'with_movement' in filename else 'DT_static'
        return bucket, 'doubletalk'
    return None, None


def run_one(mic_path, lpb_path, sr=16000, fl=832, is_movement=False):
    """Run AEC on one case, return state stats dict."""
    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr,
        filter_length=fl,
        enable_dtd=False,
        enable_shadow=True,
        enable_res=True,
        enable_cng=True,
        enable_delay_est=is_movement,  # match eval_aec_challenge.run_ours
        delay_est_period_s=0.25 if is_movement else 1.0,
        delay_est_init_s=0.2 if is_movement else 1.0,
    )
    aec = AEC(cfg)

    mic, _ = sf.read(mic_path)
    ref, _ = sf.read(lpb_path)
    mic = mic.astype(np.float32)
    ref = ref.astype(np.float32)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]

    # eval_aec_challenge.run_ours always does global-delay pre-alignment
    # (movement variant additionally enables online delay tracking).
    delay = estimate_delay(mic, ref, sr)
    if 0 < delay < n:
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
        ref = ref_aligned

    hop = aec.hop_size
    pos = 0
    counts = dict(initial=0, dominant=0, usable_v1=0, usable_v2=0,
                  transitions=0, reset_0=0, reset_1=0, reset_2=0, reset_3=0,
                  epc=0, pmax=0, pfloor=0)
    total = 0
    cur_epc_run = 0
    max_epc_run = 0
    raw_dt_pre_epc_during = []
    scale_ratios = []
    # Round 4 per-bin RES diagnostics (collected on far-active frames)
    R4_KEYS = ('coh2_mean_full', 'coh2_mean_voice',
               'res_over_err_mean_full', 'res_over_err_mean_voice',
               'ne_over_err_mean_full', 'ne_over_err_mean_voice',
               'noise_over_err_mean_full',
               'g_voice_mean', 'g_voice_min', 'g_voice_p10',
               'echo_dominant_bin_pct')
    r4_buf = {k: [] for k in R4_KEYS}
    # Round 5: per-stage gain means (voice-band)
    R5_KEYS = ('g_stage_softgate_emr_voice', 'g_stage_spectral_floor_voice',
               'g_stage_epc_dt_cap_voice', 'g_stage_quiet_mask_voice',
               'g_stage_3bin_smooth_voice', 'g_stage_hf_cap_voice',
               'g_stage_pre_temporal_voice', 'g_stage_post_temporal_voice',
               'g_stage_after_noise_lift_voice')
    r5_buf = {k: [] for k in R5_KEYS}
    # Round 7: filter trajectory + transition events
    # Counters across all frames (regardless of far-active) for transition stats
    r7_event_counts = {'delay_first': 0, 'delay_shift': 0, 'epv': 0, 'shadow_rise': 0}
    r7_epv_raw_count = 0
    r7_epv_suppressed_count = 0
    r7_epv_w_norm_at_fire = []  # filter_w_norm at frames where EPV raw fired
    r7_delay_samples = []
    r7_delay_delta_abs = []
    r7_p_max_active = 0
    r7_p_floor_active = 0
    r7_once_conv_frames = 0
    r7_epc_force_active = 0
    r7_total_frames = 0
    # Per-frame collections on far-active frames (most meaningful for filter trajectory)
    r7_far_active_frames = 0
    r7_nores_echo_proxy = []
    r7_res_required_gain = []
    r7_nores_pwr_db = []
    r7_final_pwr_db = []
    # Adaptation / convergence signals
    r7_shadow_adv = []
    r7_inst_erle_smooth = []
    r7_main_err_smooth = []
    r7_shadow_err_smooth = []
    r7_filter_w_norm = []
    r7_shadow_w_norm = []
    r7_mu_scale = []
    # Transition-window response: count frames-since-last-event per source,
    # then aggregate inst_erle inside the 30-frame post-event window.
    # Lower mean = slower recovery = filter trajectory pathology after transition.
    r7_post_event_window = 30  # frames (~300 ms at 10 ms hop)
    r7_frames_since = {'delay_first': r7_post_event_window + 1,
                       'delay_shift': r7_post_event_window + 1,
                       'epv': r7_post_event_window + 1,
                       'shadow_rise': r7_post_event_window + 1}
    r7_post_event_inst_erle = {'delay_first': [], 'delay_shift': [],
                                'epv': [], 'shadow_rise': []}
    while pos + hop <= n:
        aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        d = aec._diag
        if d.get('initial_state_active'): counts['initial'] += 1
        if d.get('dominant_nearend_like_state'): counts['dominant'] += 1
        if d.get('usable_linear_estimate_v1'): counts['usable_v1'] += 1
        if d.get('usable_linear_estimate_v2'): counts['usable_v2'] += 1
        if d.get('initial_transition_triggered'): counts['transitions'] += 1
        rs = d.get('erle_reset_signal', 0)
        counts[f'reset_{rs}'] += 1
        # Round 3 fields
        if d.get('epc_active_now'):
            counts['epc'] += 1
            cur_epc_run += 1
            v = float(d.get('raw_dt_pre_epc', 0.0))
            if v > 0:
                raw_dt_pre_epc_during.append(v)
        else:
            max_epc_run = max(max_epc_run, cur_epc_run)
            cur_epc_run = 0
        if d.get('p_max_override_active'): counts['pmax'] += 1
        if d.get('p_floor_beta_active'): counts['pfloor'] += 1
        sr = float(d.get('filter_scale_ratio', 1.0))
        if 0 < sr < 1e6:
            scale_ratios.append(sr)
        # Round 4: per-bin RES diagnostics on far-active frames
        if float(d.get('far_activity', 0.0)) > 0.01:
            for k in R4_KEYS:
                v = d.get(k)
                if v is not None:
                    r4_buf[k].append(float(v))
            # Round 5: stage gain means (same far-active frames)
            for k in R5_KEYS:
                v = d.get(k)
                if v is not None:
                    r5_buf[k].append(float(v))
            # Round 7: per-frame collection on far-active frames
            r7_far_active_frames += 1
            r7_nores_echo_proxy.append(float(d.get('nores_echo_proxy', 0.0)))
            r7_res_required_gain.append(float(d.get('res_required_gain', 1.0)))
            np_pwr = float(d.get('nores_output_power', 0.0))
            fp_pwr = float(d.get('final_output_power', 0.0))
            r7_nores_pwr_db.append(10 * np.log10(max(np_pwr, 1e-12)))
            r7_final_pwr_db.append(10 * np.log10(max(fp_pwr, 1e-12)))
            r7_shadow_adv.append(float(d.get('shadow_advantage', 0.0)))
            r7_inst_erle_smooth.append(float(d.get('inst_erle_smooth', 1.0)))
            r7_main_err_smooth.append(float(d.get('main_err_smooth', 0.0)))
            r7_shadow_err_smooth.append(float(d.get('shadow_err_smooth', 0.0)))
            r7_filter_w_norm.append(float(d.get('filter_w_norm', 0.0)))
            r7_shadow_w_norm.append(float(d.get('shadow_w_norm', 0.0)))
            r7_mu_scale.append(float(d.get('mu_scale', 1.0)))
        # Round 7: per-frame collection (all frames, not gated)
        r7_total_frames += 1
        # R7.1a: EPV raw vs suppressed (mechanism trace)
        if d.get('epv_event_raw'):
            r7_epv_raw_count += 1
            r7_epv_w_norm_at_fire.append(float(d.get('filter_w_norm', 0.0)))
        if d.get('epv_event_suppressed'):
            r7_epv_suppressed_count += 1
        # Detect events; reset post-event counter on each fire
        for _src in ('delay_first', 'delay_shift', 'epv', 'shadow_rise'):
            if d.get(f'event_{_src}'):
                r7_event_counts[_src] += 1
                r7_frames_since[_src] = 0
            else:
                r7_frames_since[_src] += 1
            # If still inside the post-event window, accumulate inst_erle (for response analysis)
            if r7_frames_since[_src] < r7_post_event_window:
                _ies = d.get('inst_erle_smooth')
                if _ies is not None:
                    r7_post_event_inst_erle[_src].append(float(_ies))
        _ds = int(d.get('delay_samples', -1))
        if _ds >= 0:
            r7_delay_samples.append(_ds)
        _dd = int(d.get('delay_delta', 0))
        if _dd != 0:
            r7_delay_delta_abs.append(abs(_dd))
        if int(d.get('p_max_override_remaining', 0)) > 0:
            r7_p_max_active += 1
        if int(d.get('p_floor_beta_remaining', 0)) > 0:
            r7_p_floor_active += 1
        if d.get('filter_once_converged'):
            r7_once_conv_frames += 1
        if int(d.get('epc_render_forced_remaining', 0)) > 0:
            r7_epc_force_active += 1
        total += 1
        pos += hop
    max_epc_run = max(max_epc_run, cur_epc_run)
    div_counts = dict(getattr(aec, '_round3_div_counts',
                              {'delay_first': 0, 'delay_shift': 0,
                               'epv': 0, 'shadow_rise': 0}))
    stats = {
        'frames': total,
        'initial_pct': counts['initial'] / total if total else 0.0,
        'dominant_pct': counts['dominant'] / total if total else 0.0,
        'usable_v1_pct': counts['usable_v1'] / total if total else 0.0,
        'usable_v2_pct': counts['usable_v2'] / total if total else 0.0,
        'transitions': counts['transitions'],
        'reset_dist': {k: counts[f'reset_{k}'] for k in [0, 1, 2, 3]},
        'final_once_converged': bool(aec._filter_once_converged),
        'final_far_active_blocks': int(getattr(aec, '_far_active_blocks', 0)),
        'final_erl_estimate': float(aec._erl_estimate),
        'final_dt_from_zero_count': int(getattr(aec, '_dt_from_zero_count', 0)),
        # Round 3 additions
        'epc_pct': counts['epc'] / total if total else 0.0,
        'epc_max_run': int(max_epc_run),
        'pmax_override_pct': counts['pmax'] / total if total else 0.0,
        'pfloor_beta_pct': counts['pfloor'] / total if total else 0.0,
        'div_counts': div_counts,
        'raw_dt_pre_epc_mean': float(np.mean(raw_dt_pre_epc_during)) if raw_dt_pre_epc_during else 0.0,
        'raw_dt_pre_epc_p90': float(np.percentile(raw_dt_pre_epc_during, 90)) if raw_dt_pre_epc_during else 0.0,
        'raw_dt_pre_epc_count': int(len(raw_dt_pre_epc_during)),
        'scale_ratio_mean': float(np.mean(scale_ratios)) if scale_ratios else 1.0,
        'scale_ratio_median': float(np.median(scale_ratios)) if scale_ratios else 1.0,
        'scale_ratio_p10': float(np.percentile(scale_ratios, 10)) if scale_ratios else 1.0,
    }
    # Round 4: per-case mean / p10 / p90 for each per-bin RES diagnostic
    for k in R4_KEYS:
        arr = r4_buf[k]
        if arr:
            stats[f'r4_{k}_mean'] = float(np.mean(arr))
            stats[f'r4_{k}_p10'] = float(np.percentile(arr, 10))
            stats[f'r4_{k}_p90'] = float(np.percentile(arr, 90))
        else:
            stats[f'r4_{k}_mean'] = 0.0
            stats[f'r4_{k}_p10'] = 0.0
            stats[f'r4_{k}_p90'] = 0.0
    # Round 5: per-case mean for each gain stage (voice-band)
    for k in R5_KEYS:
        arr = r5_buf[k]
        stats[f'r5_{k}_mean'] = float(np.mean(arr)) if arr else 0.0
    # Round 7: per-case aggregates
    nf = max(r7_total_frames, 1)
    fa = max(r7_far_active_frames, 1)
    # Transition event counts
    for _src, _c in r7_event_counts.items():
        stats[f'r7_event_{_src}_count'] = int(_c)
    # R7.1a: EPV raw vs suppressed
    stats['r7_epv_raw_count'] = int(r7_epv_raw_count)
    stats['r7_epv_suppressed_count'] = int(r7_epv_suppressed_count)
    stats['r7_epv_w_norm_at_fire_mean'] = (
        float(np.mean(r7_epv_w_norm_at_fire)) if r7_epv_w_norm_at_fire else 0.0)
    # Delay magnitude / dynamics
    stats['r7_delay_samples_mean'] = float(np.mean(r7_delay_samples)) if r7_delay_samples else -1.0
    stats['r7_delay_samples_max'] = int(np.max(r7_delay_samples)) if r7_delay_samples else -1
    stats['r7_delay_delta_max_abs'] = int(np.max(r7_delay_delta_abs)) if r7_delay_delta_abs else 0
    stats['r7_delay_delta_count'] = int(len(r7_delay_delta_abs))
    # Override/state active fractions (over total frames)
    stats['r7_p_max_active_pct'] = r7_p_max_active / nf
    stats['r7_p_floor_active_pct'] = r7_p_floor_active / nf
    stats['r7_once_conv_pct'] = r7_once_conv_frames / nf
    stats['r7_epc_force_active_pct'] = r7_epc_force_active / nf
    stats['r7_far_active_pct'] = r7_far_active_frames / nf
    # Output power signals (far-active subset)
    stats['r7_nores_echo_proxy_mean'] = float(np.mean(r7_nores_echo_proxy)) if r7_nores_echo_proxy else 0.0
    stats['r7_nores_echo_proxy_max'] = float(np.max(r7_nores_echo_proxy)) if r7_nores_echo_proxy else 0.0
    stats['r7_res_required_gain_mean'] = float(np.mean(r7_res_required_gain)) if r7_res_required_gain else 1.0
    stats['r7_nores_pwr_db_mean'] = float(np.mean(r7_nores_pwr_db)) if r7_nores_pwr_db else -120.0
    stats['r7_final_pwr_db_mean'] = float(np.mean(r7_final_pwr_db)) if r7_final_pwr_db else -120.0
    # Adaptation / convergence signals (far-active subset)
    stats['r7_shadow_advantage_mean'] = float(np.mean(r7_shadow_adv)) if r7_shadow_adv else 1.0
    stats['r7_inst_erle_smooth_mean'] = float(np.mean(r7_inst_erle_smooth)) if r7_inst_erle_smooth else 1.0
    stats['r7_main_err_smooth_mean'] = float(np.mean(r7_main_err_smooth)) if r7_main_err_smooth else 0.0
    stats['r7_shadow_err_smooth_mean'] = float(np.mean(r7_shadow_err_smooth)) if r7_shadow_err_smooth else 0.0
    stats['r7_filter_w_norm_mean'] = float(np.mean(r7_filter_w_norm)) if r7_filter_w_norm else 0.0
    stats['r7_shadow_w_norm_mean'] = float(np.mean(r7_shadow_w_norm)) if r7_shadow_w_norm else 0.0
    stats['r7_mu_scale_mean'] = float(np.mean(r7_mu_scale)) if r7_mu_scale else 1.0
    # Post-event inst_erle (transition-window response). Lower = slower recovery
    # after that transition type; high separation worst-vs-best = transition is binding.
    for _src, _arr in r7_post_event_inst_erle.items():
        stats[f'r7_post_{_src}_inst_erle_mean'] = float(np.mean(_arr)) if _arr else 0.0
        stats[f'r7_post_{_src}_inst_erle_count'] = int(len(_arr))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('out_dir')
    ap.add_argument('--limit', type=int, default=None,
                    help='Limit cases per scenario (for smoke testing)')
    ap.add_argument('--jobs', '-j', type=int, default=1,
                    help='Parallel worker processes (default 1)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)  # CNG determinism for trace

    tasks = []
    scenarios = ['farend_singletalk', 'nearend_singletalk', 'doubletalk']
    for sc in scenarios:
        files = sorted(glob.glob(os.path.join(args.dataset, sc, '*_mic.wav')))
        if args.limit:
            files = files[:args.limit]
        print(f'{sc}: {len(files)} cases', flush=True)
        for mp in files:
            lp = mp.replace('_mic.wav', '_lpb.wav')
            if not os.path.isfile(lp):
                continue
            stem = os.path.basename(mp).replace('_mic.wav', '')
            bucket, _ = classify(stem)
            is_mv = '_with_movement_' in stem
            tasks.append((mp, lp, is_mv, bucket, stem))

    results = {}
    total = len(tasks)
    if args.jobs <= 1:
        for i, t in enumerate(tasks):
            stem, stats, err = _run_case(t)
            if err:
                print(f'  ERR {stem}: {err}', flush=True)
                continue
            results[stem] = stats
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{total}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_run_case, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                stem, stats, err = fut.result()
                done += 1
                if err:
                    print(f'  ERR {stem}: {err}', flush=True)
                    continue
                results[stem] = stats
                if done % 50 == 0:
                    print(f'  {done}/{total}', flush=True)

    out_path = os.path.join(args.out_dir, 'states.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {out_path}: {len(results)} cases')


if __name__ == '__main__':
    main()
