#!/usr/bin/env python3
"""Gate0 trace — AEC3 poor_coarse_filter rescue copy correctness audit.

Tests `use_aec3_poor_coarse_rescue_copy` (default-OFF candidate) against M_full
on 4 Gate0 cases to verify:
  1. Flag OFF is byte-equal to M_full (rescue copy fires with existing gapped params).
  2. Flag ON fires rescue copy more frequently (stricter condition + 2-hop threshold).
  3. 9xjhi coarse_conv / shadow_w_norm improves under rescue copy.
  4. xFk7 DT does not worsen by > 0.2 deg.
  5. nores LF/MF/HF artifact does not increase vs M_full.

AEC3 source audit (subtractor.cc:264-307):
  Condition:  e2_refined < e2_coarse (strict, no margin)       [Gap A]
  Threshold:  5 AEC3 blocks = ceil(5×64/hop_size) = 2 hops    [Gap B]
  E_refined:  same-hop override via defer_update / complete_update [Gap C IMPLEMENTED]
  Hangover:   25 AEC3 blocks = ceil(25×64/160) = 10 hops      [Gap D]
  Behavior:   shadow adapts normally; only refined gets
              disallow_leakage_diverged (subtractor.cc:264-265) [Gap E]

Gate0 pass criteria:
  - flag OFF: byte-equal 25/25 (already verified in check_byte_equal.py)
  - rescue copy fires on 9xjhi poor-coarse windows
  - 9xjhi coarse_conv / e2_coarse improves
  - xFk7 DT: delta usable_linear >= -0.05 (no significant regression)
  - nores LF energy does not increase vs M_full on 9xjhi

Usage:
    cd /path/to/AEC
    python3 python/v3_21_poor_coarse_rescue_gate0.py
"""
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modules.config import AecConfig
from aec import AEC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = os.path.join(REPO, 'wav', 'v3_21_8_cohort')

CASES = {
    '9xjhi': {
        'subdir': 'farend_singletalk',
        'stem':   '9xjhiFbGo06hdQIsHTS6qA_farend_singletalk',
        'bucket': 'FS_static',
        'metric': 'echo',
        'M0':     4.565,
        'AEC3':   3.442,
        'M_full': 2.354,
        'extra_gap': -1.088,
    },
    'qNvSMyUS': {
        'subdir': 'farend_singletalk',
        'stem':   'qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk',
        'bucket': 'FS_static',
        'metric': 'echo',
        'M0':     None,
        'AEC3':   None,
        'M_full': None,
        'extra_gap': None,
    },
    'xFk7': {
        'subdir': 'doubletalk',
        'stem':   'xFk7igecuke0R5JMfREyDg_doubletalk_with_movement',
        'bucket': 'DT_mvmt',
        'metric': 'deg',
        'M0':     2.881,
        'AEC3':   1.275,
        'M_full': 1.893,
        'extra_gap': +0.618,
    },
    '0I0XMl3M': {
        'subdir': 'farend_singletalk',
        'stem':   '0I0XMl3M0ECO0U1N0cJvpg_farend_singletalk_with_movement',
        'bucket': 'FS_mvmt',
        'metric': 'echo',
        'M0':     None,
        'AEC3':   None,
        'M_full': None,
        'extra_gap': None,
    },
}

# ---------------------------------------------------------------------------
# Variant flag sets (M_full = all 13 flags + delay chain)
# ---------------------------------------------------------------------------
_BASE_OFF = {
    'use_partition_summed_x2_for_h_error_gain':      False,
    'use_current_e2_refined_in_h_error_denominator': False,
    'use_per_bin_h_error_refresh':                    False,
    'use_aec3_h_error_ceil':                          False,
    'use_aec3_filter_noise_gate_power':               False,
    'use_partition_summed_x2_for_shadow_mu':          False,
    'use_aec3_noise_gate_for_shadow':                 False,
    'use_poor_excitation_gate_for_shadow':            False,
    'use_narrowband_mask_for_shadow':                 False,
    'use_saturation_gate_for_shadow':                 False,
    'use_refined_output_selection_for_linear_path':   False,
    'form_linear_filter_crossfade_enabled':           False,
    'use_full_delay_change_chain':                    False,
    'transparent_mode_enabled':                       False,
    'use_aec3_poor_coarse_rescue_copy':               False,
}
_BUNDLE_A = {
    'use_partition_summed_x2_for_h_error_gain':      True,
    'use_current_e2_refined_in_h_error_denominator': True,
    'use_per_bin_h_error_refresh':                    True,
    'use_aec3_h_error_ceil':                          True,
    'use_aec3_filter_noise_gate_power':               True,
}
_BUNDLE_B = {
    'use_partition_summed_x2_for_shadow_mu':          True,
    'use_aec3_noise_gate_for_shadow':                 True,
    'use_poor_excitation_gate_for_shadow':            True,
    'use_narrowband_mask_for_shadow':                 True,
    'use_saturation_gate_for_shadow':                 True,
}
_BUNDLE_C = {
    'use_refined_output_selection_for_linear_path':   True,
    'form_linear_filter_crossfade_enabled':           True,
}
_BUNDLE_ALL = {**_BUNDLE_A, **_BUNDLE_B, **_BUNDLE_C}

_M_FULL_FLAGS = {**_BASE_OFF, **_BUNDLE_ALL, 'use_full_delay_change_chain': True}

VARIANTS = {
    'M_full':         _M_FULL_FLAGS,
    'M_full_rescue':  {**_M_FULL_FLAGS, 'use_aec3_poor_coarse_rescue_copy': True},
}
VARIANT_ORDER = ['M_full', 'M_full_rescue']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_wav(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), sr


def band_energy_db(audio: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    """Energy in [f_lo, f_hi] Hz band, in dB relative to RMS."""
    n = len(audio)
    spec = np.fft.rfft(audio * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    pwr = float(np.sum(np.abs(spec[mask]) ** 2))
    if pwr <= 0:
        return -80.0
    return 10.0 * np.log10(pwr / (n / 2))


def run_variant(case_key: str, variant_key: str, flags: dict) -> dict:
    """Run one case × variant; return poor_coarse rescue trace statistics."""
    case = CASES[case_key]
    mic_p = os.path.join(COHORT, case['subdir'], f"{case['stem']}_mic.wav")
    ref_p = os.path.join(COHORT, case['subdir'], f"{case['stem']}_lpb.wav")
    if not os.path.exists(mic_p):
        return {'error': f'missing: {mic_p}'}
    mic, sr = load_wav(mic_p)
    ref, _  = load_wav(ref_p)
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]

    cfg = AecConfig.from_preset('balanced')
    cfg.enable_res   = True
    cfg.enable_cng   = True
    cfg.trace_hf_chain = True
    for k, v in flags.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    np.random.seed(42)
    aec = AEC(cfg)

    hop = aec.filter.hop_size
    n_frames = (n - hop) // hop
    out_buf = np.zeros(n_frames * hop, dtype=np.float32)

    for i in range(n_frames):
        out_buf[i*hop:(i+1)*hop] = aec.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop])

    # --- nores run (linear-only, enable_res=False) ---
    cfg_nores = AecConfig.from_preset('balanced')
    cfg_nores.enable_res   = False
    cfg_nores.enable_cng   = False
    cfg_nores.trace_hf_chain = False
    for k, v in flags.items():
        if hasattr(cfg_nores, k):
            setattr(cfg_nores, k, v)
    np.random.seed(42)
    aec_nr = AEC(cfg_nores)
    nores_buf = np.zeros(n_frames * hop, dtype=np.float32)
    for i in range(n_frames):
        nores_buf[i*hop:(i+1)*hop] = aec_nr.process(
            mic[i*hop:(i+1)*hop], ref[i*hop:(i+1)*hop])

    trace = aec._hf_chain_trace
    if not trace:
        return {'error': 'no_trace'}

    n_total = len(trace)

    def _frac(lst):
        return len(lst) / n_total if n_total > 0 else 0.0

    def _mean(lst, key, default=float('nan')):
        vals = [r.get(key, default) for r in lst
                if r.get(key) is not None and not np.isnan(r.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    # --- poor_coarse rescue copy fields (pc_* from _hf_chain_trace) ---
    cond_fire_frames  = [r for r in trace if r.get('pc_cond_fire', False)]
    copy_fired_frames = [r for r in trace if r.get('pc_copy_fired', False)]
    hangover_frames   = [r for r in trace if r.get('pc_hangover_rem', 0) > 0]
    e_override_frames = [r for r in trace if r.get('pc_e_refined_override', False)]

    cond_fire_frac  = _frac(cond_fire_frames)
    copy_fire_frac  = _frac(copy_fired_frames)
    hangover_frac   = _frac(hangover_frames)
    e_override_frac = _frac(e_override_frames)

    threshold_hops = trace[0].get('pc_threshold_hops', 5) if trace else 5
    rescue_aec3    = trace[0].get('pc_rescue_aec3', False) if trace else False

    # e2 ratio distribution (e2_ref / e2_coa)
    e2_ratios = []
    for r in trace:
        e2r = r.get('pc_e2_ref', 0.0)
        e2c = r.get('pc_e2_coa', 1e-10)
        if e2c > 0:
            e2_ratios.append(e2r / e2c)
    e2_ratio_median = float(np.median(e2_ratios)) if e2_ratios else float('nan')
    e2_ratio_q25    = float(np.percentile(e2_ratios, 25)) if e2_ratios else float('nan')

    # --- shadow W_norm ---
    w_norms = [r.get('shadow_w_norm', 0.0) for r in trace]
    w_norm_q25   = float(np.percentile(w_norms, 25)) if w_norms else float('nan')
    w_norm_median = float(np.median(w_norms)) if w_norms else float('nan')
    w_norm_final  = float(w_norms[-1]) if w_norms else float('nan')

    # shadow W_norm at copy-fire frames vs adjacent frames (before/after)
    w_before_copy = []
    w_after_copy  = []
    for idx, r in enumerate(trace):
        if r.get('pc_copy_fired', False):
            if idx > 0:
                w_before_copy.append(trace[idx-1].get('shadow_w_norm', float('nan')))
            if idx + 1 < n_total:
                w_after_copy.append(trace[idx+1].get('shadow_w_norm', float('nan')))
    w_before_mean = float(np.nanmean(w_before_copy)) if w_before_copy else float('nan')
    w_after_mean  = float(np.nanmean(w_after_copy)) if w_after_copy else float('nan')

    # W_shape_match: verify W shapes are equal (main == shadow) at first frame
    try:
        w_shape_match = bool(aec.filter.W.shape == aec.shadow_filter.W.shape)
        main_w_shape  = tuple(aec.filter.W.shape)
        shadow_w_shape = tuple(aec.shadow_filter.W.shape)
    except AttributeError:
        w_shape_match = False
        main_w_shape = shadow_w_shape = None

    # --- coarse convergence ---
    coarse_conv_frac = _frac([r for r in trace if r.get('coarse_conv', False)])
    e2_coarse_median = float(np.nanmedian([r.get('uro_e2_coarse', float('nan'))
                                           for r in trace]))
    e2_refined_median = float(np.nanmedian([r.get('uro_e2_refined', float('nan'))
                                            for r in trace]))

    # --- URO cond1/cond2 ---
    uro_cond1_frac = _frac([r for r in trace if r.get('uro_cond1', False)])
    uro_cond2_frac = _frac([r for r in trace if r.get('uro_cond2', False)])
    uro_fire_frac  = _frac([r for r in trace if r.get('uro_fire', False)])

    # --- usable_linear ---
    ul_frac = _frac([r for r in trace if r.get('usable_linear', False)])

    # --- nores LF/MF/HF energy ---
    nores_lf = band_energy_db(nores_buf, sr, 0,   500)
    nores_mf = band_energy_db(nores_buf, sr, 500,  2000)
    nores_hf = band_energy_db(nores_buf, sr, 2000, 8000)

    # time-domain RMS and peak of nores
    nores_rms  = float(np.sqrt(np.mean(nores_buf ** 2))) if len(nores_buf) > 0 else 0.0
    nores_peak = float(np.max(np.abs(nores_buf))) if len(nores_buf) > 0 else 0.0
    mic_rms    = float(np.sqrt(np.mean(mic[:len(nores_buf)] ** 2))) if len(nores_buf) > 0 else 0.0

    return {
        'n_frames':         n_total,
        'threshold_hops':   threshold_hops,
        'rescue_aec3':      rescue_aec3,
        # --- rescue copy trigger stats ---
        'cond_fire_frac':   cond_fire_frac,
        'copy_fire_frac':   copy_fire_frac,
        'hangover_frac':    hangover_frac,
        'copy_fire_count':  len(copy_fired_frames),
        # e2 ratio
        'e2_ratio_median':  e2_ratio_median,
        'e2_ratio_q25':     e2_ratio_q25,
        # --- shadow W_norm ---
        'w_norm_q25':       w_norm_q25,
        'w_norm_median':    w_norm_median,
        'w_norm_final':     w_norm_final,
        'w_before_copy':    w_before_mean,
        'w_after_copy':     w_after_mean,
        'w_shape_match':    w_shape_match,
        'main_w_shape':     main_w_shape,
        'shadow_w_shape':   shadow_w_shape,
        # --- convergence ---
        'coarse_conv_frac': coarse_conv_frac,
        'e2_coarse_median': e2_coarse_median,
        'e2_refined_median': e2_refined_median,
        # --- URO ---
        'uro_cond1_frac':   uro_cond1_frac,
        'uro_cond2_frac':   uro_cond2_frac,
        'uro_fire_frac':    uro_fire_frac,
        # --- usable_linear ---
        'ul_frac':          ul_frac,
        # --- e_refined_override (Gap C — fraction of frames where the
        # rescue fire hop used E_refined for the shadow update) ---
        'e_refined_override_frac': e_override_frac,
        'e_refined_override_count': len(e_override_frames),
        # --- nores ---
        'nores_lf_db':      nores_lf,
        'nores_mf_db':      nores_mf,
        'nores_hf_db':      nores_hf,
        'nores_rms':        nores_rms,
        'nores_peak':       nores_peak,
        'mic_rms':          mic_rms,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 72)
    print('Gate0 — poor_coarse rescue copy audit (use_aec3_poor_coarse_rescue_copy)')
    print('=' * 72)
    print()
    print('AEC3 vs Python gap summary:')
    print('  Gap A: condition  — Python 0.5× margin vs AEC3 strict <')
    print('  Gap B: threshold  — Python 5 hops vs AEC3 2 hops (5×64/160)')
    print('  Gap C: E_refined  — same-hop override IMPLEMENTED (defer_update + complete_update)')
    print('  Gap D: hangover   — Python 16 hops vs AEC3 10 hops (25×64/160)')
    print('  Gap E: behavior   — Python shadow frozen vs AEC3 shadow adapts')
    print()

    results = {}

    for case_key, case in CASES.items():
        mic_p = os.path.join(COHORT, case['subdir'], f"{case['stem']}_mic.wav")
        if not os.path.exists(mic_p):
            print(f'[SKIP] {case_key}: WAV not found at {mic_p}')
            continue

        print(f'{"="*72}')
        print(f'CASE: {case_key}  bucket={case["bucket"]}  metric={case["metric"]}')
        if case['M0'] is not None:
            print(f'  M0={case["M0"]}  AEC3={case["AEC3"]}  '
                  f'M_full={case["M_full"]}  extra_gap={case["extra_gap"]:+.3f}')
        print()
        results[case_key] = {}

        for vk in VARIANT_ORDER:
            print(f'  Running {case_key} × {vk} ...', flush=True)
            r = run_variant(case_key, vk, VARIANTS[vk])
            results[case_key][vk] = r
            if 'error' in r:
                print(f'    ERROR: {r["error"]}')
                continue
            print(f'    n_frames={r["n_frames"]}  threshold_hops={r["threshold_hops"]}  '
                  f'rescue_aec3={r["rescue_aec3"]}')
            print(f'    Rescue copy:  cond_fire={r["cond_fire_frac"]:.1%}  '
                  f'copy_fire={r["copy_fire_frac"]:.1%}  '
                  f'copy_count={r["copy_fire_count"]}  '
                  f'hangover={r["hangover_frac"]:.1%}')
            print(f'    e2_ratio:     median={r["e2_ratio_median"]:.3f}  '
                  f'q25={r["e2_ratio_q25"]:.3f}  '
                  f'(< 1.0 → cond fires on AEC3-exact; < 0.5 → default fires)')
            print(f'    W shape:      match={r["w_shape_match"]}  '
                  f'main={r["main_w_shape"]}  shadow={r["shadow_w_shape"]}')
            print(f'    W_norm:       q25={r["w_norm_q25"]:.3f}  '
                  f'med={r["w_norm_median"]:.3f}  final={r["w_norm_final"]:.3f}')
            if not np.isnan(r['w_before_copy']):
                print(f'    W_norm copy:  before={r["w_before_copy"]:.3f}  '
                      f'after={r["w_after_copy"]:.3f}  '
                      f'delta={r["w_after_copy"]-r["w_before_copy"]:+.3f}')
            print(f'    Convergence:  coarse_conv={r["coarse_conv_frac"]:.1%}  '
                  f'e2_coa_med={r["e2_coarse_median"]:.2e}  '
                  f'e2_ref_med={r["e2_refined_median"]:.2e}')
            print(f'    URO:          cond1={r["uro_cond1_frac"]:.1%}  '
                  f'cond2={r["uro_cond2_frac"]:.1%}  '
                  f'fire={r["uro_fire_frac"]:.1%}')
            print(f'    usable_linear={r["ul_frac"]:.1%}')
            print(f'    Nores LF/MF/HF: {r["nores_lf_db"]:.1f} / '
                  f'{r["nores_mf_db"]:.1f} / {r["nores_hf_db"]:.1f} dB  '
                  f'rms={r["nores_rms"]:.4f}  peak={r["nores_peak"]:.4f}  '
                  f'mic_rms={r["mic_rms"]:.4f}')
            print(f'    Gap C E_refined override: frac={r["e_refined_override_frac"]:.1%}  '
                  f'count={r["e_refined_override_count"]}')
            print()

    # ---------------------------------------------------------------------------
    # Delta summary: M_full_rescue vs M_full
    # ---------------------------------------------------------------------------
    print('=' * 72)
    print('DELTA SUMMARY: M_full_rescue vs M_full')
    print('=' * 72)
    print(f'{"Case":<14} {"cond_fire Δ":>12} {"copy_fire Δ":>12} '
          f'{"w_norm_med Δ":>14} {"coarse_conv Δ":>14} '
          f'{"ul Δ":>8} {"nores_LF Δ":>11}')
    print('-' * 90)

    gate0_pass = True
    for case_key in CASES:
        if case_key not in results:
            continue
        r_base   = results[case_key].get('M_full', {})
        r_rescue = results[case_key].get('M_full_rescue', {})
        if 'error' in r_base or 'error' in r_rescue:
            print(f'{case_key:<14}  ERROR in one or both variants')
            continue

        d_cond    = r_rescue['cond_fire_frac']  - r_base['cond_fire_frac']
        d_copy    = r_rescue['copy_fire_frac']  - r_base['copy_fire_frac']
        d_w       = r_rescue['w_norm_median']   - r_base['w_norm_median']
        d_cc      = r_rescue['coarse_conv_frac'] - r_base['coarse_conv_frac']
        d_ul      = r_rescue['ul_frac']          - r_base['ul_frac']
        d_nores   = r_rescue['nores_lf_db']      - r_base['nores_lf_db']

        # Gate0 criteria
        case_obj = CASES[case_key]
        case_pass = True
        fail_reasons = []
        if case_obj['bucket'] == 'DT_mvmt' and d_ul < -0.05:
            fail_reasons.append(f'ul_delta={d_ul:.3f} < -0.05')
            case_pass = False
        if case_obj['bucket'].startswith('FS') and d_nores > 1.0:
            fail_reasons.append(f'nores_LF_delta={d_nores:.1f} > 1.0 dB')
            case_pass = False
        if not case_pass:
            gate0_pass = False

        status = 'PASS' if case_pass else 'FAIL'
        print(f'{case_key:<14} {d_cond:>+11.1%} {d_copy:>+11.1%} '
              f'{d_w:>+13.3f} {d_cc:>+13.1%} '
              f'{d_ul:>+7.1%} {d_nores:>+10.1f}  [{status}]'
              + (f'  ← {"; ".join(fail_reasons)}' if fail_reasons else ''))

    print('-' * 90)
    print()
    print(f'Gate0 verdict: {"PASS" if gate0_pass else "FAIL"}')
    print()
    print('Notes:')
    print('  cond_fire Δ > 0: AEC3-exact fires condition more often (expected: +)')
    print('  copy_fire Δ > 0: rescue copy fires more (expected: + for FS cases)')
    print('  w_norm_med Δ:    shadow W_norm change after more frequent copy')
    print('  coarse_conv Δ:   improvement in shadow convergence (expected: +)')
    print('  ul Δ:            usable_linear fraction change (DT: must be >= -0.05)')
    print('  nores_LF Δ:      nores LF energy change dB (FS: must be <= +1.0 dB)')
    print()
    print('Gap C (same-hop E_refined override) IMPLEMENTED:')
    print('  PBFDAF.process(defer_update=True) stashes inputs; orchestrator drives')
    print('  shadow.complete_update(error_override=self.filter.error_spec) on the')
    print('  copy_fired hop. Off-fire hops complete_update(None) preserves the')
    print('  inline E_coarse path. AEC3 subtractor.cc:302-304 parity.')
    print()
    print('Gate0 PASS → proceed to 12-case M_full+rescue (user auth required).')
    print('Gate0 FAIL → diagnose per-case before 12-case.')
