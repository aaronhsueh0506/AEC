"""Generate c_impl/include/aec3_balanced_config.h — the baked v3.22 balanced
AEC3 sub-module construction config.

The orchestrator builds AecState / ResidualEchoEstimator / SuppressionGain /
StationarityEstimator with a large set of derived constants (hz_to_bin,
fft_density_scale, wall-clock-rescaled CNG/ratchet constants, per-bin tuning
interpolation). Rather than re-derive all of that in C (and risk a 1-ULP drift),
we capture the LIVE balanced instance's exact values once and emit them as C
literals. aec_create() then bakes them straight into the sub-module init calls,
mirroring c_impl/test/parity_aec3_post_run.c.

mild / aggressive differ from balanced ONLY in split_floor_far_active_db
(mild -20 / balanced -28 / aggressive -38), applied at runtime in
aec_config_from_preset — so only the balanced base is baked here.

M2 (multi-rate campaign) extension: on top of the legacy 16 kHz block (kept
byte-for-byte, produced by the exact same code path as before), this also
captures the live balanced instance at 8 kHz and 48 kHz and emits
``AEC3B_R8K_*`` / ``AEC3B_R48K_*`` blocks + a runtime ``AEC3B_RATE_TABLE``
lookup. Only genuinely rate-varying quantities are duplicated per-rate; a
generator-internal cross-rate invariance assertion (``_assert_cross_rate_
invariance``) diffs the COMPLETE captured value set across 8/16/48 kHz and
aborts loudly if anything outside the declared rate-varying set
(``RATE_VARYING_MACROS`` / ``RATE_VARYING_ARRAYS``) differs — this is what
catches accidental hidden rate dependence instead of silently baking a wrong
"constant" that happens to be right at 16 kHz.

Run: python3 python/diag/gen_aec_balanced_config_h.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modules.config import AecConfig                          # noqa: E402
from modules.orchestrator import AEC                          # noqa: E402
from gen_aec3_post_run_golden import (                        # noqa: E402
    capture_sg_config, capture_state_config, capture_ree_config)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'c_impl', 'include', 'aec3_balanced_config.h')

SR = 16000
DEFAULT_FRAME = 512
FL = 832

# M2: additional rates. filter_length is left at Python's own __post_init__
# auto-policy (None => not passed => dataclass sentinel -1 => 52 ms <44.1kHz
# / 64 ms >=44.1kHz), NOT hardcoded like the legacy 16 kHz FL=832.
EXTRA_GRIDS = (
    (8000, 256, 'R8K', 'AEC3B_ENABLE_8K'),
    (16000, 256, 'R16K256', 'AEC3B_ENABLE_16K_256'),
    (48000, 1024, 'R48K', 'AEC3B_ENABLE_48K'),
)


def f64(x):
    return '%.17g' % float(x)


def fl(x):
    """Format a float32 as a valid C float literal (always has a decimal
    point / exponent so it never degenerates to e.g. `0f`)."""
    s = '%.9g' % float(np.float32(x))
    if ('.' not in s) and ('e' not in s) and ('E' not in s) \
            and ('inf' not in s) and ('nan' not in s):
        s += '.0'
    return s + 'f'


def emit_array(name, arr):
    vals = ', '.join(fl(v) for v in arr)
    return 'static const float %s[%d] = { %s };\n' % (name, len(arr), vals)


def _lf_clamp_comment(n_bins, sr, val):
    """Reproduces the hand-written E8 comment byte-for-byte at 16 kHz
    (n_bins=257, sr=16000 -> 8), generalised so 8k/48k get their own
    correct (n_bins, sr) values in the same format."""
    return ('/* E8: hz_to_bin(250 Hz, n_bins=%d, sr=%d) = round(250*2*(%d-1)/%d) = %d */\n'
            % (n_bins, sr, n_bins, sr, val))


def _capture_all(sr, frame_size, filter_length=None):
    """Instantiate the live Python balanced AEC at `sr` and return every
    value the header emitter needs, keyed by the AEC3B_* macro name it
    feeds (sans prefix/tag). This dict is the SINGLE SOURCE OF TRUTH used by
    both the emitters below and the cross-rate invariance assertion so the
    two can never drift apart.

    filter_length=None means "don't pass filter_length at all" -> the
    AecConfig dataclass sentinel (-1) triggers its own __post_init__ auto
    policy. Only the legacy 16 kHz call passes an explicit filter_length
    (832, matching the historical hardcoded FL).
    """
    kwargs = dict(
        sample_rate=sr,
        frame_size=frame_size,
        hop_size=frame_size // 2,
    )
    if filter_length is not None:
        kwargs['filter_length'] = filter_length
    cfg = AecConfig.from_preset('balanced', **kwargs)
    np.random.seed(0)
    aec = AEC(cfg)
    f = aec.filter
    sg = capture_sg_config(aec._aec3_sg)
    st = capture_state_config(aec._aec3_state)
    rc = capture_ree_config(aec._aec3_ree)

    tdf = f.get_time_domain_filter()
    filter_taps_size = int(len(tdf))

    values = {}
    arrays = {}

    # ── geometry ──
    values['N_BINS'] = int(f.n_freqs)
    values['FFT_SIZE'] = int(f.fft_size)
    values['BLOCK_SIZE'] = int(f.block_size)
    values['HOP_SIZE'] = int(f.hop_size)
    values['N_PARTITIONS'] = int(f.n_partitions)
    values['FILTER_TAPS_SIZE'] = filter_taps_size
    # FILTER_LENGTH is new in M2 (no legacy unsuffixed macro existed before;
    # the 16k row of the rate table uses the literal 832 instead of minting
    # a brand-new unsuffixed macro, so the legacy block stays untouched).
    values['FILTER_LENGTH'] = int(cfg.filter_length)

    # ── driver flags ──
    values['ERLE_COH_GATE_ENABLED'] = int(bool(cfg.erle_coh_gate_enabled))
    values['ERLE_WINDOWED_CAPTURE_PSD'] = int(bool(getattr(cfg, 'erle_windowed_capture_psd', False)))
    values['ERLE_RENDER_X2_PSD_SCALE'] = int(bool(getattr(cfg, 'erle_render_x2_psd_scale', False)))
    values['OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE'] = int(
        bool(getattr(cfg, 'output_capture_when_linear_unusable', False)))
    values['ENABLE_CNG'] = int(bool(cfg.enable_cng))
    values['USE_STATIONARITY_PROPERTIES'] = int(
        bool(aec._aec3_sg_config.echo_audibility.use_stationarity_properties))
    values['STATIONARITY_CONVERGE_HOPS'] = int(aec._aec3_stationarity_converge_hops)
    # Hardcoded literal in orchestrator.py (_ar_thr = 5.96e-4), NOT derived
    # from sample_rate/hop_size -> genuinely rate-invariant (verified by the
    # cross-rate assertion below, not just assumed).
    values['ACTIVE_RENDER_THRESHOLD'] = 5.96e-4

    # ── CNG constants ──
    values['CNG_Y2_ALPHA'] = float(aec._aec3_cng_y2_alpha)
    values['CNG_N2_TRACK_FRESHNESS'] = float(aec._aec3_cng_n2_track_freshness)
    values['CNG_N2_TRACK_RETENTION'] = float(aec._aec3_cng_n2_track_retention)
    values['CNG_N2_SLOW_UP'] = float(aec._aec3_cng_n2_slow_up)
    values['CNG_N2_INITIAL_ALPHA'] = float(aec._aec3_cng_n2_initial_alpha)
    values['CNG_N2_UPDATE_ONSET_HOPS'] = int(aec._aec3_cng_n2_update_onset_hops)
    values['CNG_N2_INITIAL_DURATION_HOPS'] = int(aec._aec3_cng_n2_initial_duration_hops)
    # NOTE: genuinely rate-varying (fft_size/2 density scale in
    # orchestrator.py's GetNoiseFloorFactor port) -- see RATE_VARYING_MACROS.
    values['NOISE_FLOOR_INT16SQ'] = float(aec._aec3_noise_floor_int16sq)
    values['ERLE_COH_GATE_ALPHA'] = float(cfg.erle_coh_gate_alpha)
    values['ERLE_COH_GATE_THRESHOLD'] = float(cfg.erle_coh_gate_threshold)

    arrays['SYNTH_WINDOW'] = np.asarray(aec._aec3_synth_window, np.float32)
    arrays['SQRT2_SIN_LUT'] = np.asarray(aec._aec3_sqrt2_sin_lut, np.float32)

    # ── AecState config ──
    values['ST_NUM_CAPTURE_CHANNELS'] = int(st['num_capture_channels'])
    values['ST_ENABLE_FILTER_ANALYZER'] = int(st['enable_filter_analyzer'])
    values['ST_ERLE_STARTUP_HOPS'] = int(st['erle_startup_hops'])
    values['ST_ERL_STARTUP_HOPS'] = int(st['erl_startup_hops'])
    values['ST_ECHO_CAN_SATURATE'] = int(st['echo_can_saturate'])
    values['ST_USE_LINEAR_FILTER'] = int(st['use_linear_filter'])
    values['ST_CONSERVATIVE_INITIAL_PHASE'] = int(st['conservative_initial_phase'])
    values['ST_DELAY_HEADROOM_SAMPLES'] = int(st['delay_headroom_samples'])
    values['ST_INITIAL_STATE_SECONDS'] = float(st['initial_state_seconds'])
    values['ST_ERLE_MIN'] = float(st['erle_min'])
    values['ST_ERLE_MAX_L'] = float(st['erle_max_l'])
    values['ST_ERLE_MAX_H'] = float(st['erle_max_h'])

    # ── ResidualEchoEstimator config ──
    values['REE_HOP_SIZE'] = int(rc['hop_size'])
    values['REE_MODEL_REVERB_IN_NL'] = int(rc['model_reverb_in_nl'])
    values['REE_ERLE_ONSET_COMP'] = int(rc['erle_onset_comp'])
    values['REE_REVERB_ENABLED'] = int(rc['reverb_enabled'])
    values['REE_USE_AEC3_RESIDUAL_NOISE_GATE'] = int(rc['use_aec3_residual_noise_gate'])
    values['REE_USE_AEC3_ECHO_GEN_WINDOW'] = int(rc['use_aec3_echo_gen_window'])
    values['REE_NL_R2_ENABLED'] = int(rc['nl_r2_enabled'])
    values['REE_NOISE_FLOOR_HOLD_HOPS'] = int(rc['noise_floor_hold_hops'])
    values['REE_USE_FREQ_RESPONSE'] = int(rc['use_freq_response'])
    values['REE_REVERB_USE_CONSERVATIVE'] = int(rc['reverb_use_conservative'])
    # NOTE: genuinely rate-varying (fft_density_scale of EchoModelConfig's
    # default) -- see RATE_VARYING_MACROS.
    values['REE_MIN_NOISE_FLOOR_POWER'] = float(rc['min_noise_floor_power'])
    values['REE_NOISE_GATE_POWER_LEGACY'] = float(rc['noise_gate_power_legacy'])
    values['REE_NOISE_GATE_SLOPE'] = float(rc['noise_gate_slope'])
    values['REE_STATIONARY_GATE_SLOPE'] = float(rc['stationary_gate_slope'])
    values['REE_DEFAULT_GAIN'] = float(rc['default_gain'])
    values['REE_TM_GAIN'] = float(rc['tm_gain'])
    values['REE_REVERB_DECAY'] = float(rc['reverb_decay'])
    values['REE_REVERB_MILD_SCALE'] = float(rc['reverb_mild_scale'])
    values['REE_REVERB_TAIL_STRENGTH'] = float(rc['reverb_tail_strength'])
    values['REE_NL_R2_ALPHA'] = float(rc['nl_r2_alpha'])
    # NOTE: genuinely rate-varying (scales with frame_size=2*hop_size, see
    # aec3_scale.nl_r2_norm_power) -- see RATE_VARYING_MACROS.
    values['REE_NL_NORM_POWER'] = float(rc['nl_norm_power'])
    # NOTE: genuinely rate-varying (derived from the same density scale as
    # REE_MIN_NOISE_FLOOR_POWER) -- see RATE_VARYING_MACROS.
    values['REE_RESIDUAL_NOISE_GATE_POWER'] = float(rc['residual_noise_gate_power'])
    values['REE_REVERB_SMOOTHING_BASE'] = float(rc['reverb_smoothing_base'])

    # ── SuppressionGain config ──
    values['SG_SPLIT_FLOOR_FAR_ACTIVE'] = float(sg['split_floor_far_active'])
    values['SG_SPLIT_FLOOR_FAR_SILENT'] = float(sg['split_floor_far_silent'])
    values['SG_SPLIT_FLOOR_LATCH_POWER'] = float(sg['split_floor_latch_power'])
    # NOTE: genuinely rate-varying (fft_density_scale) -- see RATE_VARYING_MACROS.
    values['SG_LOW_RENDER_THRESHOLD'] = float(sg['low_render_threshold'])
    values['SG_MAX_INC'] = float(sg['max_inc'])
    values['SG_MAX_DEC_LF'] = float(sg['max_dec_lf'])
    values['SG_SOFT_BLEND_ENABLED'] = int(sg['soft_blend_enabled'])
    values['SG_SOFT_BLEND_PER_BIN'] = int(sg['soft_blend_per_bin'])
    values['SG_SOFT_BLEND_ENR_THR'] = float(sg['soft_blend_enr_thr'])
    values['SG_SOFT_BLEND_SOFTNESS'] = float(sg['soft_blend_softness'])
    values['SG_LAST_LF_BAND'] = int(sg['last_lf_band'])
    values['SG_FIRST_HF_BAND'] = int(sg['first_hf_band'])
    values['SG_LAST_LF_SMOOTHING_BAND'] = int(sg['last_lf_smoothing_band'])
    values['SG_LAST_PERMANENT'] = int(sg['last_permanent'])
    values['SG_LF_SMOOTHING_INITIAL'] = int(sg['lf_smoothing_initial'])
    values['SG_LF_CLAMP_BIN'] = int(sg['lf_clamp_bin'])
    values['SG_DNE_LF_END'] = int(sg['dne_lf_end'])
    values['SG_TRIGGER_THRESHOLD_HOPS'] = int(sg['trigger_threshold_hops'])
    values['SG_HOLD_DURATION_HOPS'] = int(sg['hold_duration_hops'])
    values['SG_NEAREND_SMOOTHER_N'] = int(sg['nearend_smoother_n'])
    values['SG_AUD_LF_END_BIN'] = int(sg['aud_lf_end_bin'])
    values['SG_AUD_MF_END_BIN'] = int(sg['aud_mf_end_bin'])
    # NOTE: genuinely rate-varying (fft_density_scale) -- see RATE_VARYING_MACROS.
    values['SG_FLOOR_POWER'] = float(sg['floor_power'])
    values['SG_AUD_THR_LF'] = float(sg['aud_thr_lf'])
    values['SG_AUD_THR_MF'] = float(sg['aud_thr_mf'])
    values['SG_AUD_THR_HF'] = float(sg['aud_thr_hf'])
    # NOTE: genuinely rate-varying (fft_density_scale) -- see RATE_VARYING_MACROS.
    values['SG_LOW_RENDER_LIMIT'] = float(sg['low_render_limit'])
    # NOTE: genuinely rate-varying (fft_density_scale) -- see RATE_VARYING_MACROS.
    values['SG_NORMAL_RENDER_LIMIT'] = float(sg['normal_render_limit'])
    values['SG_HF_LGB'] = int(sg['hf_lgb'])
    values['SG_HF_BIQ'] = int(sg['hf_biq'])
    values['SG_CONSERVATIVE_HF'] = int(sg['conservative_hf'])
    values['SG_DNE_ENR_THRESHOLD'] = float(sg['dne_enr_threshold'])
    values['SG_DNE_ENR_EXIT_THRESHOLD'] = float(sg['dne_enr_exit_threshold'])
    values['SG_DNE_SNR_THRESHOLD'] = float(sg['dne_snr_threshold'])
    values['SG_DNE_USE_DURING_INITIAL_PHASE'] = int(sg['dne_use_during_initial_phase'])
    values['SG_DNE_USE_UNBOUNDED_ECHO'] = int(sg['dne_use_unbounded_echo'])
    values['SG_DNE_LF_ENDPOINT_BIN'] = int(sg['dne_lf_endpoint_bin'])

    arrays['SG_NEAREND_ENR_TR'] = sg['nearend_enr_tr']
    arrays['SG_NEAREND_ENR_SU'] = sg['nearend_enr_su']
    arrays['SG_NEAREND_EMR_TR'] = sg['nearend_emr_tr']
    arrays['SG_NORMAL_ENR_TR'] = sg['normal_enr_tr']
    arrays['SG_NORMAL_ENR_SU'] = sg['normal_enr_su']
    arrays['SG_NORMAL_EMR_TR'] = sg['normal_emr_tr']

    return dict(
        sr=sr, cfg=cfg, aec=aec, f=f, sg=sg, st=st, rc=rc,
        filter_taps_size=filter_taps_size, values=values, arrays=arrays,
        lf_clamp_comment=_lf_clamp_comment(int(f.n_freqs), sr, int(sg['lf_clamp_bin'])),
    )


# Declared rate-varying macro names -- values legitimately differ 8k/16k/48k.
# Starts from the M2 task's declared list (geometry + Hz-anchored bin
# indices), EXTENDED with quantities this generator's own cross-rate
# invariance assertion caught during implementation (fft/hop density-scaled
# absolute-power constants the original task spec did not anticipate).
# Reported prominently: these 8 fields are the "found" set --
#   SG_LOW_RENDER_THRESHOLD, SG_FLOOR_POWER, SG_LOW_RENDER_LIMIT,
#   SG_NORMAL_RENDER_LIMIT, REE_MIN_NOISE_FLOOR_POWER, REE_NL_NORM_POWER,
#   REE_RESIDUAL_NOISE_GATE_POWER, NOISE_FLOOR_INT16SQ
# -- all AEC3 EchoAudibility / ResidualEchoEstimator / CNG absolute-power
# constants that are density-scaled by fft_size/2 (or frame_size) relative
# to AEC3's fixed 64-sample/128-bin reference, so they move with sample
# rate exactly like the bin geometry does.
RATE_VARYING_MACROS = {
    'N_BINS', 'FFT_SIZE', 'BLOCK_SIZE', 'HOP_SIZE', 'N_PARTITIONS',
    'FILTER_TAPS_SIZE', 'FILTER_LENGTH',
    'REE_HOP_SIZE',
    'SG_LAST_LF_BAND', 'SG_FIRST_HF_BAND', 'SG_LAST_LF_SMOOTHING_BAND',
    'SG_LF_CLAMP_BIN', 'SG_DNE_LF_END', 'SG_DNE_LF_ENDPOINT_BIN',
    'SG_AUD_LF_END_BIN', 'SG_AUD_MF_END_BIN', 'SG_HF_LGB',
    # found by the invariance assertion, not in the original task list:
    'SG_LOW_RENDER_THRESHOLD', 'SG_FLOOR_POWER', 'SG_LOW_RENDER_LIMIT',
    'SG_NORMAL_RENDER_LIMIT',
    'REE_MIN_NOISE_FLOOR_POWER', 'REE_NL_NORM_POWER',
    'REE_RESIDUAL_NOISE_GATE_POWER',
    'NOISE_FLOOR_INT16SQ',
    # Wall-clock-rescaled values vary when hop/sample-rate is no longer
    # locked to exactly 10 ms.
    'STATIONARITY_CONVERGE_HOPS',
    'CNG_Y2_ALPHA', 'CNG_N2_TRACK_FRESHNESS', 'CNG_N2_TRACK_RETENTION',
    'CNG_N2_SLOW_UP', 'CNG_N2_INITIAL_ALPHA',
    'CNG_N2_UPDATE_ONSET_HOPS', 'CNG_N2_INITIAL_DURATION_HOPS',
    'REE_NOISE_FLOOR_HOLD_HOPS', 'REE_REVERB_SMOOTHING_BASE',
    'SG_MAX_INC', 'SG_MAX_DEC_LF', 'SG_TRIGGER_THRESHOLD_HOPS',
    'SG_HOLD_DURATION_HOPS', 'SG_NEAREND_SMOOTHER_N',
}
# The per-bin tuning arrays always vary (length == n_bins).
RATE_VARYING_ARRAYS = {
    'SYNTH_WINDOW', 'SG_NEAREND_ENR_TR', 'SG_NEAREND_ENR_SU',
    'SG_NEAREND_EMR_TR', 'SG_NORMAL_ENR_TR', 'SG_NORMAL_ENR_SU',
    'SG_NORMAL_EMR_TR',
}


def _assert_cross_rate_invariance(captures):
    """`captures`: {sample_rate: _capture_all(...) result}.

    Aborts loudly (SystemExit) if any macro/array NOT declared rate-varying
    differs across sample rates -- this is the mechanism that catches
    accidental hidden rate dependence rather than silently baking a
    16 kHz-only value into what looks like a rate-invariant constant.
    """
    rates = sorted(captures)
    ref_sr = rates[0]
    ref = captures[ref_sr]
    problems = []

    for name in ref['values']:
        if name in RATE_VARYING_MACROS:
            continue
        for sr in rates[1:]:
            v0 = ref['values'][name]
            v1 = captures[sr]['values'][name]
            if v0 != v1:
                problems.append((name, ref_sr, v0, sr, v1))

    for name in ref['arrays']:
        if name in RATE_VARYING_ARRAYS:
            continue
        for sr in rates[1:]:
            a0 = ref['arrays'][name]
            a1 = captures[sr]['arrays'][name]
            if not np.array_equal(a0, a1):
                problems.append((name, ref_sr, '<array>', sr, '<differs>'))

    if problems:
        msg = [
            'CROSS-RATE INVARIANCE ASSERTION FAILED in '
            'gen_aec_balanced_config_h.py -- the following fields are NOT '
            'declared rate-varying (RATE_VARYING_MACROS/RATE_VARYING_ARRAYS) '
            'but differ across sample rates. A genuinely rate-varying field '
            'must be MOVED into the per-rate table (added to those sets and '
            'emitted per-rate), never silently ignored:',
        ]
        for name, r0, v0, r1, v1 in problems:
            msg.append('  %s: grid=%r -> %r   vs   grid=%r -> %r' % (name, r0, v0, r1, v1))
        raise SystemExit('\n'.join(msg))


def emit_legacy_block(w, cap):
    """Emits the ORIGINAL unsuffixed (16 kHz) block. Byte-for-byte identical
    to the pre-M2 generator's output (including the hand-patched E8 LF_CLAMP
    comment/define that predated this generator change -- see cdd7f6b)."""
    v = cap['values']

    # ── geometry ──
    w('/* geometry */\n')
    w('#define AEC3B_N_BINS        %d\n' % v['N_BINS'])
    w('#define AEC3B_FFT_SIZE      %d\n' % v['FFT_SIZE'])
    w('#define AEC3B_BLOCK_SIZE    %d\n' % v['BLOCK_SIZE'])
    w('#define AEC3B_HOP_SIZE      %d\n' % v['HOP_SIZE'])
    w('#define AEC3B_N_PARTITIONS  %d\n' % v['N_PARTITIONS'])
    w('#define AEC3B_FILTER_TAPS_SIZE %d\n\n' % v['FILTER_TAPS_SIZE'])

    # ── driver flags ──
    w('/* aec3_post driver flags */\n')
    w('#define AEC3B_ERLE_COH_GATE_ENABLED  %d\n' % v['ERLE_COH_GATE_ENABLED'])
    w('#define AEC3B_ERLE_WINDOWED_CAPTURE_PSD %d\n' % v['ERLE_WINDOWED_CAPTURE_PSD'])
    w('#define AEC3B_ERLE_RENDER_X2_PSD_SCALE %d\n' % v['ERLE_RENDER_X2_PSD_SCALE'])
    w('#define AEC3B_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE %d\n' % v['OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE'])
    w('#define AEC3B_ENABLE_CNG %d\n' % v['ENABLE_CNG'])
    w('#define AEC3B_USE_STATIONARITY_PROPERTIES %d\n' % v['USE_STATIONARITY_PROPERTIES'])
    w('#define AEC3B_STATIONARITY_CONVERGE_HOPS %d\n' % v['STATIONARITY_CONVERGE_HOPS'])
    w('#define AEC3B_ACTIVE_RENDER_THRESHOLD (%s)\n\n' % f64(v['ACTIVE_RENDER_THRESHOLD']))

    # ── CNG constants ──
    w('/* CNG wall-clock-rescaled constants */\n')
    w('#define AEC3B_CNG_Y2_ALPHA (%s)\n' % f64(v['CNG_Y2_ALPHA']))
    w('#define AEC3B_CNG_N2_TRACK_FRESHNESS (%s)\n' % f64(v['CNG_N2_TRACK_FRESHNESS']))
    w('#define AEC3B_CNG_N2_TRACK_RETENTION (%s)\n' % f64(v['CNG_N2_TRACK_RETENTION']))
    w('#define AEC3B_CNG_N2_SLOW_UP (%s)\n' % f64(v['CNG_N2_SLOW_UP']))
    w('#define AEC3B_CNG_N2_INITIAL_ALPHA (%s)\n' % f64(v['CNG_N2_INITIAL_ALPHA']))
    w('#define AEC3B_CNG_N2_UPDATE_ONSET_HOPS %d\n' % v['CNG_N2_UPDATE_ONSET_HOPS'])
    w('#define AEC3B_CNG_N2_INITIAL_DURATION_HOPS %d\n' % v['CNG_N2_INITIAL_DURATION_HOPS'])
    w('#define AEC3B_NOISE_FLOOR_INT16SQ (%s)\n' % f64(v['NOISE_FLOOR_INT16SQ']))
    w('#define AEC3B_ERLE_COH_GATE_ALPHA (%s)\n' % f64(v['ERLE_COH_GATE_ALPHA']))
    w('#define AEC3B_ERLE_COH_GATE_THRESHOLD (%s)\n\n' % f64(v['ERLE_COH_GATE_THRESHOLD']))

    # ── synth window + sqrt2 sin LUT ──
    w(emit_array('AEC3B_SYNTH_WINDOW', cap['arrays']['SYNTH_WINDOW']))
    w(emit_array('AEC3B_SQRT2_SIN_LUT', cap['arrays']['SQRT2_SIN_LUT']))
    w('\n')

    # ── AecState config ──
    w('/* AecState config */\n')
    w('#define AEC3B_ST_NUM_CAPTURE_CHANNELS %d\n' % v['ST_NUM_CAPTURE_CHANNELS'])
    w('#define AEC3B_ST_ENABLE_FILTER_ANALYZER %d\n' % v['ST_ENABLE_FILTER_ANALYZER'])
    # -1 == AEC_STATE_STARTUP_HOPS_AUTO (aec_state.c): the BALANCED preset
    # never overrides erle_startup_hops/erl_startup_hops, so this is always
    # the "auto" sentinel, resolved live per-grid at aec_state_init() time
    # -- NOT a literal hop count. Must stay the sentinel value (not a real
    # hop count like 200) so aec_create()'s multi-rate (8k/16k/48k) call
    # sites keep auto-scaling instead of freezing at a 16 kHz-only constant.
    w('#define AEC3B_ST_ERLE_STARTUP_HOPS %d\n' % v['ST_ERLE_STARTUP_HOPS'])
    w('#define AEC3B_ST_ERL_STARTUP_HOPS %d\n' % v['ST_ERL_STARTUP_HOPS'])
    w('#define AEC3B_ST_ECHO_CAN_SATURATE %d\n' % v['ST_ECHO_CAN_SATURATE'])
    w('#define AEC3B_ST_USE_LINEAR_FILTER %d\n' % v['ST_USE_LINEAR_FILTER'])
    w('#define AEC3B_ST_CONSERVATIVE_INITIAL_PHASE %d\n' % v['ST_CONSERVATIVE_INITIAL_PHASE'])
    w('#define AEC3B_ST_DELAY_HEADROOM_SAMPLES %d\n' % v['ST_DELAY_HEADROOM_SAMPLES'])
    w('#define AEC3B_ST_INITIAL_STATE_SECONDS (%s)\n' % f64(v['ST_INITIAL_STATE_SECONDS']))
    w('#define AEC3B_ST_ERLE_MIN (%s)\n' % f64(v['ST_ERLE_MIN']))
    w('#define AEC3B_ST_ERLE_MAX_L (%s)\n' % f64(v['ST_ERLE_MAX_L']))
    w('#define AEC3B_ST_ERLE_MAX_H (%s)\n\n' % f64(v['ST_ERLE_MAX_H']))

    # ── ResidualEchoEstimator config ──
    w('/* ResidualEchoEstimator config */\n')
    w('#define AEC3B_REE_HOP_SIZE %d\n' % v['REE_HOP_SIZE'])
    w('#define AEC3B_REE_MODEL_REVERB_IN_NL %d\n' % v['REE_MODEL_REVERB_IN_NL'])
    w('#define AEC3B_REE_ERLE_ONSET_COMP %d\n' % v['REE_ERLE_ONSET_COMP'])
    w('#define AEC3B_REE_REVERB_ENABLED %d\n' % v['REE_REVERB_ENABLED'])
    w('#define AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE %d\n' % v['REE_USE_AEC3_RESIDUAL_NOISE_GATE'])
    w('#define AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW %d\n' % v['REE_USE_AEC3_ECHO_GEN_WINDOW'])
    w('#define AEC3B_REE_NL_R2_ENABLED %d\n' % v['REE_NL_R2_ENABLED'])
    w('#define AEC3B_REE_NOISE_FLOOR_HOLD_HOPS %d\n' % v['REE_NOISE_FLOOR_HOLD_HOPS'])
    w('#define AEC3B_REE_USE_FREQ_RESPONSE %d\n' % v['REE_USE_FREQ_RESPONSE'])
    w('#define AEC3B_REE_REVERB_USE_CONSERVATIVE %d\n' % v['REE_REVERB_USE_CONSERVATIVE'])
    w('#define AEC3B_REE_MIN_NOISE_FLOOR_POWER (%s)\n' % f64(v['REE_MIN_NOISE_FLOOR_POWER']))
    w('#define AEC3B_REE_NOISE_GATE_POWER_LEGACY (%s)\n' % f64(v['REE_NOISE_GATE_POWER_LEGACY']))
    w('#define AEC3B_REE_NOISE_GATE_SLOPE (%s)\n' % f64(v['REE_NOISE_GATE_SLOPE']))
    w('#define AEC3B_REE_STATIONARY_GATE_SLOPE (%s)\n' % f64(v['REE_STATIONARY_GATE_SLOPE']))
    w('#define AEC3B_REE_DEFAULT_GAIN (%s)\n' % f64(v['REE_DEFAULT_GAIN']))
    w('#define AEC3B_REE_TM_GAIN (%s)\n' % f64(v['REE_TM_GAIN']))
    w('#define AEC3B_REE_REVERB_DECAY (%s)\n' % f64(v['REE_REVERB_DECAY']))
    w('#define AEC3B_REE_REVERB_MILD_SCALE (%s)\n' % f64(v['REE_REVERB_MILD_SCALE']))
    w('#define AEC3B_REE_REVERB_TAIL_STRENGTH (%s)\n' % f64(v['REE_REVERB_TAIL_STRENGTH']))
    w('#define AEC3B_REE_NL_R2_ALPHA (%s)\n' % f64(v['REE_NL_R2_ALPHA']))
    w('#define AEC3B_REE_NL_NORM_POWER (%s)\n' % f64(v['REE_NL_NORM_POWER']))
    w('#define AEC3B_REE_RESIDUAL_NOISE_GATE_POWER (%s)\n' % f64(v['REE_RESIDUAL_NOISE_GATE_POWER']))
    w('#define AEC3B_REE_REVERB_SMOOTHING_BASE (%s)\n\n' % f64(v['REE_REVERB_SMOOTHING_BASE']))

    # ── SuppressionGain config ──
    w('/* SuppressionGain config (split_floor_far_active is the preset axis,\n')
    w(' * recomputed at runtime; the baked value is the balanced -28 dB one). */\n')
    w('#define AEC3B_SG_SPLIT_FLOOR_FAR_ACTIVE (%s)\n' % f64(v['SG_SPLIT_FLOOR_FAR_ACTIVE']))
    w('#define AEC3B_SG_SPLIT_FLOOR_FAR_SILENT (%s)\n' % f64(v['SG_SPLIT_FLOOR_FAR_SILENT']))
    w('#define AEC3B_SG_SPLIT_FLOOR_LATCH_POWER (%s)\n' % f64(v['SG_SPLIT_FLOOR_LATCH_POWER']))
    w('#define AEC3B_SG_LOW_RENDER_THRESHOLD (%s)\n' % f64(v['SG_LOW_RENDER_THRESHOLD']))
    w('#define AEC3B_SG_MAX_INC (%s)\n' % f64(v['SG_MAX_INC']))
    w('#define AEC3B_SG_MAX_DEC_LF (%s)\n' % f64(v['SG_MAX_DEC_LF']))
    w('#define AEC3B_SG_SOFT_BLEND_ENABLED %d\n' % v['SG_SOFT_BLEND_ENABLED'])
    w('#define AEC3B_SG_SOFT_BLEND_PER_BIN %d\n' % v['SG_SOFT_BLEND_PER_BIN'])
    w('#define AEC3B_SG_SOFT_BLEND_ENR_THR (%s)\n' % f64(v['SG_SOFT_BLEND_ENR_THR']))
    w('#define AEC3B_SG_SOFT_BLEND_SOFTNESS (%s)\n' % f64(v['SG_SOFT_BLEND_SOFTNESS']))
    w('#define AEC3B_SG_LAST_LF_BAND %d\n' % v['SG_LAST_LF_BAND'])
    w('#define AEC3B_SG_FIRST_HF_BAND %d\n' % v['SG_FIRST_HF_BAND'])
    w('#define AEC3B_SG_LAST_LF_SMOOTHING_BAND %d\n' % v['SG_LAST_LF_SMOOTHING_BAND'])
    w('#define AEC3B_SG_LAST_PERMANENT %d\n' % v['SG_LAST_PERMANENT'])
    w('#define AEC3B_SG_LF_SMOOTHING_INITIAL %d\n' % v['SG_LF_SMOOTHING_INITIAL'])
    w(cap['lf_clamp_comment'])
    w('#define AEC3B_SG_LF_CLAMP_BIN %d\n' % v['SG_LF_CLAMP_BIN'])
    w('#define AEC3B_SG_DNE_LF_END %d\n' % v['SG_DNE_LF_END'])
    w('#define AEC3B_SG_TRIGGER_THRESHOLD_HOPS %d\n' % v['SG_TRIGGER_THRESHOLD_HOPS'])
    w('#define AEC3B_SG_HOLD_DURATION_HOPS %d\n' % v['SG_HOLD_DURATION_HOPS'])
    w('#define AEC3B_SG_NEAREND_SMOOTHER_N %d\n' % v['SG_NEAREND_SMOOTHER_N'])
    w('#define AEC3B_SG_AUD_LF_END_BIN %d\n' % v['SG_AUD_LF_END_BIN'])
    w('#define AEC3B_SG_AUD_MF_END_BIN %d\n' % v['SG_AUD_MF_END_BIN'])
    w('#define AEC3B_SG_FLOOR_POWER (%s)\n' % f64(v['SG_FLOOR_POWER']))
    w('#define AEC3B_SG_AUD_THR_LF (%s)\n' % f64(v['SG_AUD_THR_LF']))
    w('#define AEC3B_SG_AUD_THR_MF (%s)\n' % f64(v['SG_AUD_THR_MF']))
    w('#define AEC3B_SG_AUD_THR_HF (%s)\n' % f64(v['SG_AUD_THR_HF']))
    w('#define AEC3B_SG_LOW_RENDER_LIMIT (%s)\n' % f64(v['SG_LOW_RENDER_LIMIT']))
    w('#define AEC3B_SG_NORMAL_RENDER_LIMIT (%s)\n' % f64(v['SG_NORMAL_RENDER_LIMIT']))
    w('#define AEC3B_SG_HF_LGB %d\n' % v['SG_HF_LGB'])
    w('#define AEC3B_SG_HF_BIQ %d\n' % v['SG_HF_BIQ'])
    w('#define AEC3B_SG_CONSERVATIVE_HF %d\n' % v['SG_CONSERVATIVE_HF'])
    w('#define AEC3B_SG_DNE_ENR_THRESHOLD (%s)\n' % f64(v['SG_DNE_ENR_THRESHOLD']))
    w('#define AEC3B_SG_DNE_ENR_EXIT_THRESHOLD (%s)\n' % f64(v['SG_DNE_ENR_EXIT_THRESHOLD']))
    w('#define AEC3B_SG_DNE_SNR_THRESHOLD (%s)\n' % f64(v['SG_DNE_SNR_THRESHOLD']))
    w('#define AEC3B_SG_DNE_USE_DURING_INITIAL_PHASE %d\n' % v['SG_DNE_USE_DURING_INITIAL_PHASE'])
    w('#define AEC3B_SG_DNE_USE_UNBOUNDED_ECHO %d\n' % v['SG_DNE_USE_UNBOUNDED_ECHO'])
    w('#define AEC3B_SG_DNE_LF_ENDPOINT_BIN %d\n\n' % v['SG_DNE_LF_ENDPOINT_BIN'])

    # ── per-bin tuning arrays ──
    w('/* SuppressionGain per-bin tuning arrays (LF/HF interpolated). */\n')
    w(emit_array('AEC3B_SG_NEAREND_ENR_TR', cap['arrays']['SG_NEAREND_ENR_TR']))
    w(emit_array('AEC3B_SG_NEAREND_ENR_SU', cap['arrays']['SG_NEAREND_ENR_SU']))
    w(emit_array('AEC3B_SG_NEAREND_EMR_TR', cap['arrays']['SG_NEAREND_EMR_TR']))
    w(emit_array('AEC3B_SG_NORMAL_ENR_TR', cap['arrays']['SG_NORMAL_ENR_TR']))
    w(emit_array('AEC3B_SG_NORMAL_ENR_SU', cap['arrays']['SG_NORMAL_ENR_SU']))
    w(emit_array('AEC3B_SG_NORMAL_EMR_TR', cap['arrays']['SG_NORMAL_EMR_TR']))


def emit_rate_block(w, tag, enable_flag, cap):
    """Emits the AEC3B_<tag>_* block for one extra rate (8k/48k): ONLY the
    rate-varying subset (geometry, synth window, REE hop, Hz-anchored SG bin
    indices, and the extra density-scaled constants the invariance
    assertion found), guarded by `enable_flag` so firmware can strip it."""
    v = cap['values']
    sr = cap['sr']

    w('#ifndef %s\n#define %s 1\n#endif\n' % (enable_flag, enable_flag))
    w('#if %s\n' % enable_flag)
    w('/* ---- %d Hz rate block (AUTO-GENERATED multi-rate M2 extension) ----\n'
      % sr)
    w(' * Same live-capture mechanism as the legacy 16 kHz block above, at\n')
    w(' * sample_rate=%d (filter_length via AecConfig.__post_init__\'s own\n'
      % sr)
    w(' * auto policy, not hardcoded). Only the rate-varying subset is\n')
    w(' * duplicated here; everything else is shared with the 16 kHz block\n')
    w(' * (verified equal by this generator\'s cross-rate invariance\n')
    w(' * assertion -- see RATE_VARYING_MACROS/RATE_VARYING_ARRAYS). */\n')

    w('#define AEC3B_%s_N_BINS        %d\n' % (tag, v['N_BINS']))
    w('#define AEC3B_%s_FFT_SIZE      %d\n' % (tag, v['FFT_SIZE']))
    w('#define AEC3B_%s_BLOCK_SIZE    %d\n' % (tag, v['BLOCK_SIZE']))
    w('#define AEC3B_%s_HOP_SIZE      %d\n' % (tag, v['HOP_SIZE']))
    w('#define AEC3B_%s_N_PARTITIONS  %d\n' % (tag, v['N_PARTITIONS']))
    w('#define AEC3B_%s_FILTER_TAPS_SIZE %d\n' % (tag, v['FILTER_TAPS_SIZE']))
    w('#define AEC3B_%s_FILTER_LENGTH %d\n\n' % (tag, v['FILTER_LENGTH']))

    w(emit_array('AEC3B_%s_SYNTH_WINDOW' % tag, cap['arrays']['SYNTH_WINDOW']))
    w('#define AEC3B_%s_SYNTH_WINDOW_LEN %d\n\n' % (tag, len(cap['arrays']['SYNTH_WINDOW'])))

    w('#define AEC3B_%s_REE_HOP_SIZE %d\n\n' % (tag, v['REE_HOP_SIZE']))

    w('/* Hz-anchored SuppressionGain bin indices (rate-varying). */\n')
    w('#define AEC3B_%s_SG_LAST_LF_BAND %d\n' % (tag, v['SG_LAST_LF_BAND']))
    w('#define AEC3B_%s_SG_FIRST_HF_BAND %d\n' % (tag, v['SG_FIRST_HF_BAND']))
    w('#define AEC3B_%s_SG_LAST_LF_SMOOTHING_BAND %d\n' % (tag, v['SG_LAST_LF_SMOOTHING_BAND']))
    w(_lf_clamp_comment(v['N_BINS'], sr, v['SG_LF_CLAMP_BIN']))
    w('#define AEC3B_%s_SG_LF_CLAMP_BIN %d\n' % (tag, v['SG_LF_CLAMP_BIN']))
    w('#define AEC3B_%s_SG_DNE_LF_END %d\n' % (tag, v['SG_DNE_LF_END']))
    w('#define AEC3B_%s_SG_DNE_LF_ENDPOINT_BIN %d\n' % (tag, v['SG_DNE_LF_ENDPOINT_BIN']))
    w('#define AEC3B_%s_SG_AUD_LF_END_BIN %d\n' % (tag, v['SG_AUD_LF_END_BIN']))
    w('#define AEC3B_%s_SG_AUD_MF_END_BIN %d\n' % (tag, v['SG_AUD_MF_END_BIN']))
    w('#define AEC3B_%s_SG_HF_LGB %d\n\n' % (tag, v['SG_HF_LGB']))

    w('/* Density-scaled absolute-power constants -- found genuinely\n')
    w(' * rate-varying by the cross-rate invariance assertion (not in the\n')
    w(' * original M2 task\'s declared rate-varying list). */\n')
    w('#define AEC3B_%s_SG_LOW_RENDER_THRESHOLD (%s)\n' % (tag, f64(v['SG_LOW_RENDER_THRESHOLD'])))
    w('#define AEC3B_%s_SG_FLOOR_POWER (%s)\n' % (tag, f64(v['SG_FLOOR_POWER'])))
    w('#define AEC3B_%s_SG_LOW_RENDER_LIMIT (%s)\n' % (tag, f64(v['SG_LOW_RENDER_LIMIT'])))
    w('#define AEC3B_%s_SG_NORMAL_RENDER_LIMIT (%s)\n' % (tag, f64(v['SG_NORMAL_RENDER_LIMIT'])))
    w('#define AEC3B_%s_REE_MIN_NOISE_FLOOR_POWER (%s)\n' % (tag, f64(v['REE_MIN_NOISE_FLOOR_POWER'])))
    w('#define AEC3B_%s_REE_NL_NORM_POWER (%s)\n' % (tag, f64(v['REE_NL_NORM_POWER'])))
    w('#define AEC3B_%s_REE_RESIDUAL_NOISE_GATE_POWER (%s)\n' % (tag, f64(v['REE_RESIDUAL_NOISE_GATE_POWER'])))
    w('#define AEC3B_%s_NOISE_FLOOR_INT16SQ (%s)\n\n' % (tag, f64(v['NOISE_FLOOR_INT16SQ'])))

    w('/* Wall-clock-rescaled constants for this exact hop duration. */\n')
    for name in ('STATIONARITY_CONVERGE_HOPS',
                 'CNG_N2_UPDATE_ONSET_HOPS', 'CNG_N2_INITIAL_DURATION_HOPS',
                 'REE_NOISE_FLOOR_HOLD_HOPS', 'SG_TRIGGER_THRESHOLD_HOPS',
                 'SG_HOLD_DURATION_HOPS', 'SG_NEAREND_SMOOTHER_N'):
        w('#define AEC3B_%s_%s %d\n' % (tag, name, v[name]))
    for name in ('CNG_Y2_ALPHA', 'CNG_N2_TRACK_FRESHNESS',
                 'CNG_N2_TRACK_RETENTION', 'CNG_N2_SLOW_UP',
                 'CNG_N2_INITIAL_ALPHA', 'REE_REVERB_SMOOTHING_BASE',
                 'SG_MAX_INC', 'SG_MAX_DEC_LF'):
        w('#define AEC3B_%s_%s (%s)\n' % (tag, name, f64(v[name])))
    w('\n')

    w('/* SuppressionGain per-bin tuning arrays at %d Hz (length = n_bins). */\n' % sr)
    w(emit_array('AEC3B_%s_SG_NEAREND_ENR_TR' % tag, cap['arrays']['SG_NEAREND_ENR_TR']))
    w(emit_array('AEC3B_%s_SG_NEAREND_ENR_SU' % tag, cap['arrays']['SG_NEAREND_ENR_SU']))
    w(emit_array('AEC3B_%s_SG_NEAREND_EMR_TR' % tag, cap['arrays']['SG_NEAREND_EMR_TR']))
    w(emit_array('AEC3B_%s_SG_NORMAL_ENR_TR' % tag, cap['arrays']['SG_NORMAL_ENR_TR']))
    w(emit_array('AEC3B_%s_SG_NORMAL_ENR_SU' % tag, cap['arrays']['SG_NORMAL_ENR_SU']))
    w(emit_array('AEC3B_%s_SG_NORMAL_EMR_TR' % tag, cap['arrays']['SG_NORMAL_EMR_TR']))
    w('#define AEC3B_%s_SG_TABLE_LEN %d\n' % (tag, v['N_BINS']))
    w('#endif /* %s */\n\n' % enable_flag)


def emit_rate_table(w):
    """Emits the Aec3BalancedRateDims struct + AEC3B_RATE_TABLE lookup array
    + aec3b_rate_cfg() accessor. The 16 kHz row references the LEGACY
    unsuffixed macros/arrays directly (pointer/value identity with the data
    already baked above -- nothing new is derived for 16 kHz)."""
    w('/* ------------------------------------------------------------------\n')
    w(' * Multi-rate lookup (M2). Static/header-only, mirroring the existing\n')
    w(' * pattern of this header (this file is included by exactly one TU\n')
    w(' * today, c_impl/src/aec.c, so no cross-TU duplication concern yet;\n')
    w(' * if a second TU ever includes this header, follow the same\n')
    w(' * static-const-per-TU pattern already used for the arrays above).\n')
    w(' * ------------------------------------------------------------------ */\n')
    w('typedef struct {\n')
    w('    int sample_rate;\n')
    w('    int n_bins;\n')
    w('    int fft_size;\n')
    w('    int block_size;\n')
    w('    int hop_size;\n')
    w('    int n_partitions;\n')
    w('    int filter_taps_size;\n')
    w('    int filter_length;\n')
    w('    const float* synth_window;\n')
    w('    int synth_window_len;\n')
    w('    const float* sg_nearend_enr_tr;\n')
    w('    const float* sg_nearend_enr_su;\n')
    w('    const float* sg_nearend_emr_tr;\n')
    w('    const float* sg_normal_enr_tr;\n')
    w('    const float* sg_normal_enr_su;\n')
    w('    const float* sg_normal_emr_tr;\n')
    w('    int sg_table_len;\n')
    w('    int sg_last_lf_band;\n')
    w('    int sg_first_hf_band;\n')
    w('    int sg_last_lf_smoothing_band;\n')
    w('    int sg_lf_clamp_bin;\n')
    w('    int sg_dne_lf_end;\n')
    w('    int sg_dne_lf_endpoint_bin;\n')
    w('    int sg_aud_lf_end_bin;\n')
    w('    int sg_aud_mf_end_bin;\n')
    w('    int sg_hf_lgb;\n')
    w('    int ree_hop_size;\n')
    w('    /* Found genuinely rate-varying by the cross-rate invariance\n')
    w('     * assertion (gen_aec_balanced_config_h.py RATE_VARYING_MACROS) --\n')
    w('     * moved here rather than left as a single "constant". */\n')
    w('    float sg_low_render_threshold;\n')
    w('    float sg_floor_power;\n')
    w('    float sg_low_render_limit;\n')
    w('    float sg_normal_render_limit;\n')
    w('    float ree_min_noise_floor_power;\n')
    w('    float ree_nl_norm_power;\n')
    w('    float ree_residual_noise_gate_power;\n')
    w('    float noise_floor_int16sq;\n')
    w('    int stationarity_converge_hops;\n')
    w('    float cng_y2_alpha;\n')
    w('    float cng_n2_track_freshness;\n')
    w('    float cng_n2_track_retention;\n')
    w('    float cng_n2_slow_up;\n')
    w('    float cng_n2_initial_alpha;\n')
    w('    int cng_n2_update_onset_hops;\n')
    w('    int cng_n2_initial_duration_hops;\n')
    w('    int ree_noise_floor_hold_hops;\n')
    w('    float ree_reverb_smoothing_base;\n')
    w('    float sg_max_inc;\n')
    w('    float sg_max_dec_lf;\n')
    w('    int sg_trigger_threshold_hops;\n')
    w('    int sg_hold_duration_hops;\n')
    w('    int sg_nearend_smoother_n;\n')
    w('} Aec3BalancedRateDims;\n\n')

    w('static const Aec3BalancedRateDims AEC3B_RATE_TABLE[] __attribute__((unused)) = {\n')
    w('    { /* 16000 Hz -- the legacy unsuffixed block above (pointer/value\n')
    w('       * identity with today\'s data; nothing re-derived). */\n')
    w('      16000,\n')
    w('      AEC3B_N_BINS, AEC3B_FFT_SIZE, AEC3B_BLOCK_SIZE, AEC3B_HOP_SIZE,\n')
    w('      AEC3B_N_PARTITIONS, AEC3B_FILTER_TAPS_SIZE, 832,\n')
    w('      AEC3B_SYNTH_WINDOW, AEC3B_BLOCK_SIZE,\n')
    w('      AEC3B_SG_NEAREND_ENR_TR, AEC3B_SG_NEAREND_ENR_SU, AEC3B_SG_NEAREND_EMR_TR,\n')
    w('      AEC3B_SG_NORMAL_ENR_TR, AEC3B_SG_NORMAL_ENR_SU, AEC3B_SG_NORMAL_EMR_TR,\n')
    w('      AEC3B_N_BINS,\n')
    w('      AEC3B_SG_LAST_LF_BAND, AEC3B_SG_FIRST_HF_BAND, AEC3B_SG_LAST_LF_SMOOTHING_BAND,\n')
    w('      AEC3B_SG_LF_CLAMP_BIN, AEC3B_SG_DNE_LF_END, AEC3B_SG_DNE_LF_ENDPOINT_BIN,\n')
    w('      AEC3B_SG_AUD_LF_END_BIN, AEC3B_SG_AUD_MF_END_BIN, AEC3B_SG_HF_LGB,\n')
    w('      AEC3B_REE_HOP_SIZE,\n')
    w('      AEC3B_SG_LOW_RENDER_THRESHOLD, AEC3B_SG_FLOOR_POWER, AEC3B_SG_LOW_RENDER_LIMIT,\n')
    w('      AEC3B_SG_NORMAL_RENDER_LIMIT, AEC3B_REE_MIN_NOISE_FLOOR_POWER,\n')
    w('      AEC3B_REE_NL_NORM_POWER, AEC3B_REE_RESIDUAL_NOISE_GATE_POWER,\n')
    w('      AEC3B_NOISE_FLOOR_INT16SQ,\n')
    w('      AEC3B_STATIONARITY_CONVERGE_HOPS, AEC3B_CNG_Y2_ALPHA,\n')
    w('      AEC3B_CNG_N2_TRACK_FRESHNESS, AEC3B_CNG_N2_TRACK_RETENTION,\n')
    w('      AEC3B_CNG_N2_SLOW_UP, AEC3B_CNG_N2_INITIAL_ALPHA,\n')
    w('      AEC3B_CNG_N2_UPDATE_ONSET_HOPS, AEC3B_CNG_N2_INITIAL_DURATION_HOPS,\n')
    w('      AEC3B_REE_NOISE_FLOOR_HOLD_HOPS, AEC3B_REE_REVERB_SMOOTHING_BASE,\n')
    w('      AEC3B_SG_MAX_INC, AEC3B_SG_MAX_DEC_LF,\n')
    w('      AEC3B_SG_TRIGGER_THRESHOLD_HOPS, AEC3B_SG_HOLD_DURATION_HOPS,\n')
    w('      AEC3B_SG_NEAREND_SMOOTHER_N },\n')

    for sr, fft_size, tag, flag in EXTRA_GRIDS:
        w('#if %s\n' % flag)
        w('    { /* %d Hz */\n' % sr)
        w('      %d,\n' % sr)
        w('      AEC3B_%s_N_BINS, AEC3B_%s_FFT_SIZE, AEC3B_%s_BLOCK_SIZE, AEC3B_%s_HOP_SIZE,\n'
          % (tag, tag, tag, tag))
        w('      AEC3B_%s_N_PARTITIONS, AEC3B_%s_FILTER_TAPS_SIZE, AEC3B_%s_FILTER_LENGTH,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SYNTH_WINDOW, AEC3B_%s_SYNTH_WINDOW_LEN,\n' % (tag, tag))
        w('      AEC3B_%s_SG_NEAREND_ENR_TR, AEC3B_%s_SG_NEAREND_ENR_SU, AEC3B_%s_SG_NEAREND_EMR_TR,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SG_NORMAL_ENR_TR, AEC3B_%s_SG_NORMAL_ENR_SU, AEC3B_%s_SG_NORMAL_EMR_TR,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SG_TABLE_LEN,\n' % tag)
        w('      AEC3B_%s_SG_LAST_LF_BAND, AEC3B_%s_SG_FIRST_HF_BAND, AEC3B_%s_SG_LAST_LF_SMOOTHING_BAND,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SG_LF_CLAMP_BIN, AEC3B_%s_SG_DNE_LF_END, AEC3B_%s_SG_DNE_LF_ENDPOINT_BIN,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SG_AUD_LF_END_BIN, AEC3B_%s_SG_AUD_MF_END_BIN, AEC3B_%s_SG_HF_LGB,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_REE_HOP_SIZE,\n' % tag)
        w('      AEC3B_%s_SG_LOW_RENDER_THRESHOLD, AEC3B_%s_SG_FLOOR_POWER, AEC3B_%s_SG_LOW_RENDER_LIMIT,\n'
          % (tag, tag, tag))
        w('      AEC3B_%s_SG_NORMAL_RENDER_LIMIT, AEC3B_%s_REE_MIN_NOISE_FLOOR_POWER,\n' % (tag, tag))
        w('      AEC3B_%s_REE_NL_NORM_POWER, AEC3B_%s_REE_RESIDUAL_NOISE_GATE_POWER,\n' % (tag, tag))
        w('      AEC3B_%s_NOISE_FLOOR_INT16SQ,\n' % tag)
        w('      AEC3B_%s_STATIONARITY_CONVERGE_HOPS, AEC3B_%s_CNG_Y2_ALPHA,\n' % (tag, tag))
        w('      AEC3B_%s_CNG_N2_TRACK_FRESHNESS, AEC3B_%s_CNG_N2_TRACK_RETENTION,\n' % (tag, tag))
        w('      AEC3B_%s_CNG_N2_SLOW_UP, AEC3B_%s_CNG_N2_INITIAL_ALPHA,\n' % (tag, tag))
        w('      AEC3B_%s_CNG_N2_UPDATE_ONSET_HOPS, AEC3B_%s_CNG_N2_INITIAL_DURATION_HOPS,\n' % (tag, tag))
        w('      AEC3B_%s_REE_NOISE_FLOOR_HOLD_HOPS, AEC3B_%s_REE_REVERB_SMOOTHING_BASE,\n' % (tag, tag))
        w('      AEC3B_%s_SG_MAX_INC, AEC3B_%s_SG_MAX_DEC_LF,\n' % (tag, tag))
        w('      AEC3B_%s_SG_TRIGGER_THRESHOLD_HOPS, AEC3B_%s_SG_HOLD_DURATION_HOPS,\n' % (tag, tag))
        w('      AEC3B_%s_SG_NEAREND_SMOOTHER_N },\n' % tag)
        w('#endif /* %s */\n' % flag)

    w('};\n\n')

    w('static inline const Aec3BalancedRateDims* aec3b_rate_cfg(int sample_rate, int fft_size) {\n')
    w('    int i;\n')
    w('    int n = (int)(sizeof(AEC3B_RATE_TABLE) / sizeof(AEC3B_RATE_TABLE[0]));\n')
    w('    for (i = 0; i < n; ++i) {\n')
    w('        if (AEC3B_RATE_TABLE[i].sample_rate == sample_rate &&\n')
    w('            AEC3B_RATE_TABLE[i].fft_size == fft_size) {\n')
    w('            return &AEC3B_RATE_TABLE[i];\n')
    w('        }\n')
    w('    }\n')
    w('    return (const Aec3BalancedRateDims*)0;\n')
    w('}\n')


def main():
    cap16 = _capture_all(SR, DEFAULT_FRAME, filter_length=FL)
    extra_caps = {
        (sr, fft_size): _capture_all(sr, fft_size, filter_length=None)
        for sr, fft_size, _tag, _flag in EXTRA_GRIDS
    }

    all_caps = dict(extra_caps)
    all_caps[(SR, DEFAULT_FRAME)] = cap16
    _assert_cross_rate_invariance(all_caps)

    lines = []
    w = lines.append
    w('/* aec3_balanced_config.h — AUTO-GENERATED by\n')
    w(' * python/diag/gen_aec_balanced_config_h.py. DO NOT EDIT BY HAND.\n')
    w(' *\n')
    w(' * Baked v3.22 BALANCED AEC3 sub-module construction config, captured\n')
    w(' * from the live Python balanced instance (bit-exact source of truth).\n')
    w(' *\n')
    w(' * M2 (multi-rate campaign): the legacy block below is 16 kHz only\n')
    w(' * (unsuffixed macros/arrays, unchanged mechanism). The R8K- and\n')
    w(' * R48K-prefixed blocks + the AEC3B_RATE_TABLE lookup extend this to\n')
    w(' * 8 kHz / 48 kHz without touching any 16 kHz byte or any C\n')
    w(' * consumption code -- see the generator\'s cross-rate invariance\n')
    w(' * assertion for the complete rate-varying field list.\n')
    w(' */\n')
    w('#ifndef AEC3_BALANCED_CONFIG_H\n#define AEC3_BALANCED_CONFIG_H\n\n')

    emit_legacy_block(w, cap16)
    w('\n')

    for sr, fft_size, tag, flag in EXTRA_GRIDS:
        emit_rate_block(w, tag, flag, extra_caps[(sr, fft_size)])

    emit_rate_table(w)

    w('\n#endif /* AEC3_BALANCED_CONFIG_H */\n')

    with open(OUT, 'w') as fh:
        fh.writelines(lines)
    print('wrote %s (%d lines, n_bins=%d filter_taps=%d; +8k=%d +16k/256=%d +48k=%d)'
          % (OUT, len(lines), cap16['values']['N_BINS'], cap16['filter_taps_size'],
             extra_caps[(8000, 256)]['values']['N_BINS'],
             extra_caps[(16000, 256)]['values']['N_BINS'],
             extra_caps[(48000, 1024)]['values']['N_BINS']))


if __name__ == '__main__':
    main()
