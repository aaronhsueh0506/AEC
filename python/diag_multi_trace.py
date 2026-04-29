"""Trace multiple worst-gap DT cases in parallel; identify common bottleneck.

Per case dumps full per-frame JSONL via diag_deep_trace's machinery, then
aggregates DT × render-based × far-active frames across cases to find which
stage (cap / floor / gain) is consistently active.

Usage:
  python3 diag_multi_trace.py STEM1 STEM2 STEM3 ...
  python3 diag_multi_trace.py --worst-dt 5    # auto-pick worst from per_case.json
"""
import os, sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
OUT = Path(__file__).parent / 'output_multi_trace'


def _resolve(stem):
    for sub in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        p = WAV_BASE / sub / f'{stem}_mic.wav'
        if p.is_file():
            return sub, p, p.parent / f'{stem}_lpb.wav'
    raise FileNotFoundError(stem)


def _trace_one(stem):
    np.random.seed(20260428)
    sub, mic_p, lpb_p = _resolve(stem)
    mic, sr = sf.read(str(mic_p), dtype='float32')
    lpb, _  = sf.read(str(lpb_p), dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))

    is_mv = '_with_movement' in stem
    delay_kw = (dict(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
                if is_mv else dict(enable_delay_est=False))
    cfg = AecConfig.from_preset('balanced', sample_rate=sr, mode=AecMode.PBFDKF,
                                enable_dtd=False, enable_shadow=True, enable_res=True,
                                use_kalman=True, **delay_kw)
    aec = AEC(cfg)
    if aec.res: aec.res.enable_stats()

    hop = aec.hop_size; pos = 0; idx = 0
    rows = []
    while pos + hop <= n:
        mic_f = mic[pos:pos+hop]; lpb_f = lpb[pos:pos+hop]
        far_pwr = float(np.mean(lpb_f**2)); mic_pwr = float(np.mean(mic_f**2))
        o = aec.process(mic_f, lpb_f)
        raw_err_pwr = float(np.mean(o**2))
        res = aec.res; s = aec._aec_state
        rows.append({
            'idx': idx, 'far_pwr': far_pwr, 'mic_pwr': mic_pwr,
            'raw_err_pwr': raw_err_pwr,
            'echo_psd': float(np.mean(res.echo_psd)),
            'error_psd': float(np.mean(res.error_psd)),
            'erle_inst': float(aec._diag.get('erle_inst', 0)),
            'erle_factor': float(aec._diag.get('erle_factor', 0)),
            'filter_converged': bool(s.filter_converged),
            'once_converged': bool(s.filter_once_converged),
            'dt_energy': float(s.dt_from_energy),
            'dt_shadow': float(s.dt_from_shadow),
            'dt_combined': float(s.dt_combined),
            'epc_active': bool(s.epc_active),
            'using_render': bool(res._using_render_based),
            'res_after_attribute': float(getattr(res, '_stats_last_res_after_attribute', 0)),
            'res_after_echo_cap': float(getattr(res, '_stats_last_res_after_echo_cap', 0)),
            'res_after_error_cap': float(getattr(res, '_stats_last_res_after_error_cap', 0)),
            'res_after_dt_cap': float(getattr(res, '_stats_last_res_after_dt_cap', 0)),
            'res_after_render_ceil': float(getattr(res, '_stats_last_res_after_render_ceil', 0)),
            'res_psd': float(getattr(res, '_stats_last_res_psd', 0)),
            'nearend_est': float(getattr(res, '_stats_last_nearend', 0)),
            'min_ne': float(getattr(res, '_stats_last_min_ne', 0)),
            'enr': float(getattr(res, '_stats_last_enr', 0)),
            'spectral_g_min': float(getattr(res, '_stats_last_spectral_g_min', 0)),
            'gain_before_floor': float(getattr(res, '_stats_last_gain_before_floor', 0)),
            'gain_after_floor': float(getattr(res, '_stats_last_gain_after_floor', 0)),
            'gain_after_smooth': float(getattr(res, '_stats_last_gain_after_smoothing', 0)),
            'gain_smooth_mean': float(np.mean(res.gain_smooth)),
            'output_pwr': float(np.mean(o**2)),
        })
        pos += hop; idx += 1
    return stem, rows


def _summarize(stem, rows):
    """Per-case summary: focus on DT × render × far-active frames."""
    n_total = len(rows)
    n_far = sum(1 for r in rows if r['far_pwr'] > 1e-4)
    n_dt = sum(1 for r in rows if r['dt_combined'] > 0.3)
    n_render = sum(1 for r in rows if r['using_render'])
    n_conv = sum(1 for r in rows if r['filter_converged'])
    n_once = sum(1 for r in rows if r['once_converged'])

    # Critical subset: DT + far_active (where echo leaks happen)
    dt_far = [r for r in rows if r['dt_combined'] > 0.3 and r['far_pwr'] > 1e-4]
    if not dt_far:
        return {'stem': stem, 'n_dt_far': 0}

    def m(k): return float(np.mean([r[k] for r in dt_far]))
    # Stage-by-stage residual reduction (geometric mean of ratios)
    def reduction(prev_k, cur_k):
        ratios = [r[cur_k]/r[prev_k] for r in dt_far if r[prev_k] > 1e-12]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0

    return {
        'stem': stem,
        'n_total': n_total,
        'far_pct': 100*n_far/n_total,
        'dt_pct': 100*n_dt/n_total,
        'render_pct': 100*n_render/n_total,
        'converged_pct': 100*n_conv/n_total,
        'once_converged_pct': 100*n_once/n_total,
        'n_dt_far': len(dt_far),
        # mean values in DT × far_active
        'dt_far_far_pwr': m('far_pwr'),
        'dt_far_mic_pwr': m('mic_pwr'),
        'dt_far_echo_psd': m('echo_psd'),
        'dt_far_error_psd': m('error_psd'),
        'dt_far_erle_inst': m('erle_inst'),
        'dt_far_dt_combined': m('dt_combined'),
        'dt_far_using_render_pct': 100*sum(1 for r in dt_far if r['using_render'])/len(dt_far),
        'dt_far_res_attribute': m('res_after_attribute'),
        'dt_far_res_echo_cap': m('res_after_echo_cap'),
        'dt_far_res_error_cap': m('res_after_error_cap'),
        'dt_far_res_dt_cap': m('res_after_dt_cap'),
        'dt_far_res_render_ceil': m('res_after_render_ceil'),
        'dt_far_res_final': m('res_psd'),
        'dt_far_nearend_est': m('nearend_est'),
        'dt_far_min_ne': m('min_ne'),
        'dt_far_enr': m('enr'),
        'dt_far_gain_before_floor': m('gain_before_floor'),
        'dt_far_gain_after_floor': m('gain_after_floor'),
        'dt_far_gain_after_smooth': m('gain_after_smooth'),
        'dt_far_output_pwr': m('output_pwr'),
        # Stage reductions
        'reduce_attr_to_echo': reduction('res_after_attribute', 'res_after_echo_cap'),
        'reduce_echo_to_error': reduction('res_after_echo_cap', 'res_after_error_cap'),
        'reduce_error_to_dt': reduction('res_after_error_cap', 'res_after_dt_cap'),
        'reduce_dt_to_render': reduction('res_after_dt_cap', 'res_after_render_ceil'),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('stems', nargs='*')
    ap.add_argument('--worst-dt', type=int, default=0)
    ap.add_argument('--worst-fs', type=int, default=0, help='worst FS_static (echo gap)')
    ap.add_argument('--worst-dtmv', type=int, default=0, help='worst DT_movement (echo gap)')
    ap.add_argument('--worst-ne', type=int, default=0)
    args = ap.parse_args()

    stems = list(args.stems)
    if args.worst_dt:
        rows = json.load(open(Path(__file__).parent / 'output_v31_vs_aec2/per_case.json'))
        dt_st = [r for r in rows if r['subdir']=='doubletalk' and not r['movement']]
        dt_st.sort(key=lambda r: r['ours_echo']-r['aec2_echo'])
        stems += [r['stem'] for r in dt_st[:args.worst_dt]]
    if args.worst_fs:
        rows = json.load(open(Path(__file__).parent / 'output_v31_vs_aec2/per_case.json'))
        fs = [r for r in rows if r['subdir']=='farend_singletalk' and not r['movement']]
        fs.sort(key=lambda r: r['ours_echo']-r['aec2_echo'])
        stems += [r['stem'] for r in fs[:args.worst_fs]]
    if args.worst_dtmv:
        rows = json.load(open(Path(__file__).parent / 'output_v31_vs_aec2/per_case.json'))
        dtmv = [r for r in rows if r['subdir']=='doubletalk' and r['movement']]
        dtmv.sort(key=lambda r: r['ours_echo']-r['aec2_echo'])
        stems += [r['stem'] for r in dtmv[:args.worst_dtmv]]
    if args.worst_ne:
        rows = json.load(open(Path(__file__).parent / 'output_v31_vs_aec2/per_case.json'))
        ne = [r for r in rows if r['subdir']=='nearend_singletalk']
        ne.sort(key=lambda r: r['ours_deg']-r['aec2_deg'])
        stems += [r['stem'] for r in ne[:args.worst_ne]]

    OUT.mkdir(exist_ok=True)
    print(f'tracing {len(stems)} cases in parallel...')
    summaries = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_trace_one, s) for s in stems]
        for fu in as_completed(futs):
            stem, rows = fu.result()
            summ = _summarize(stem, rows)
            summaries.append(summ)
            # Save per-case JSONL for later drill-down
            with open(OUT / f'{stem}.jsonl', 'w') as f:
                for r in rows: f.write(json.dumps(r)+'\n')
            print(f'  {stem}: {len(rows)} frames')

    summaries.sort(key=lambda s: stems.index(s['stem']))

    # Print headline table
    print('\n' + '='*120)
    print(f'{"stem":<50s} {"conv%":>6s} {"once%":>6s} {"render%":>7s} {"DTxFar":>7s}')
    for s in summaries:
        if s.get('n_dt_far',0)==0:
            print(f'{s["stem"][:49]:<50s}  (no DT×far frames)')
            continue
        print(f'{s["stem"][:49]:<50s} {s["converged_pct"]:>6.0f} {s["once_converged_pct"]:>6.0f} '
              f'{s["render_pct"]:>7.0f} {s["n_dt_far"]:>7d}')

    print('\n=== mean values across DT × far-active frames (per case) ===')
    print(f'{"stem":<50s} {"erle":>6s} {"echo_psd":>10s} {"error_psd":>10s} {"final_res":>10s} {"nearend":>10s} {"min_ne":>10s} {"out_pwr":>10s}')
    for s in summaries:
        if s.get('n_dt_far',0)==0: continue
        print(f'{s["stem"][:49]:<50s} {s["dt_far_erle_inst"]:>6.2f} '
              f'{s["dt_far_echo_psd"]:>10.2e} {s["dt_far_error_psd"]:>10.2e} '
              f'{s["dt_far_res_final"]:>10.2e} {s["dt_far_nearend_est"]:>10.2e} '
              f'{s["dt_far_min_ne"]:>10.2e} {s["dt_far_output_pwr"]:>10.2e}')

    print('\n=== stage-by-stage residual reductions (geom mean of cur/prev, < 1 = active cap) ===')
    print(f'{"stem":<50s} {"attr->echo":>11s} {"echo->err":>11s} {"err->dt":>11s} {"dt->ceil":>11s}')
    for s in summaries:
        if s.get('n_dt_far',0)==0: continue
        print(f'{s["stem"][:49]:<50s} {s["reduce_attr_to_echo"]:>11.3f} '
              f'{s["reduce_echo_to_error"]:>11.3f} {s["reduce_error_to_dt"]:>11.3f} '
              f'{s["reduce_dt_to_render"]:>11.3f}')

    print('\n=== gain trajectory (DT × far means) ===')
    print(f'{"stem":<50s} {"g_before":>10s} {"g_floor":>10s} {"g_smooth":>10s} {"enr":>8s}')
    for s in summaries:
        if s.get('n_dt_far',0)==0: continue
        print(f'{s["stem"][:49]:<50s} {s["dt_far_gain_before_floor"]:>10.3f} '
              f'{s["dt_far_gain_after_floor"]:>10.3f} {s["dt_far_gain_after_smooth"]:>10.3f} '
              f'{s["dt_far_enr"]:>8.3f}')

    Path(OUT / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(f'\nsaved per-case JSONL + summary.json -> {OUT}')


if __name__ == '__main__':
    main()
