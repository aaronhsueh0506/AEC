"""Generate c_impl/include/internal/aec_balanced_config.h — the baked balanced AEC3
sub-module construction config.

The orchestrator builds AecState / ResidualEchoEstimator / SuppressionGain /
StationarityEstimator with a large set of derived constants (hz_to_bin,
fft_density_scale, wall-clock-rescaled CNG/ratchet constants, per-bin tuning
interpolation). Rather than re-derive all of that in C (and risk a 1-ULP drift),
the LIVE balanced instance's exact values are captured once and emitted as C
literals; aec_init() bakes them straight into the sub-module init calls.

The levels differ from balanced only in the split-floor far-active and DT
floors; they are applied at runtime (aec_config_from_level, aec_set_level),
so only the balanced base is baked here.

The unsuffixed block is the 16 kHz capture; the live instance is also captured
at 8 kHz and 48 kHz and emitted as ``AECB_R8K_*`` / ``AECB_R48K_*`` blocks + a
runtime ``AECB_RATE_TABLE`` lookup. Only rate-varying quantities are duplicated
per rate: ``_assert_cross_rate_invariance`` diffs the complete captured value
set across 8/16/48 kHz and aborts if anything outside the declared
(``RATE_VARYING_MACROS`` / ``RATE_VARYING_ARRAYS``) set differs.

Two tables are stored compressed and rebuilt by aec_init() into the instance pool, bit for bit:
the synthesis window (one half table of the longest grid; every shorter window is a stride subsample, the second half
mirrors the first) and the SuppressionGain ENR/SU thresholds (run-length coded). Both encodings are checked here against
the captured Python tables (``decode(encode(x)) == x``), and the original full tables are written to the test-only header
c_impl/test/aec_ref_tables.h, which gives the unit tests an oracle independent of the decoders.

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
OUT = os.path.join(ROOT, 'c_impl', 'include', 'internal', 'aec_balanced_config.h')
OUT_REF = os.path.join(ROOT, 'c_impl', 'test', 'aec_ref_tables.h')

SR = 16000
DEFAULT_FRAME = 512
FL = 832

# Additional rates. filter_length is left to AecConfig.__post_init__ auto-policy
# (52 ms <44.1kHz / 64 ms >=44.1kHz), unlike the 16 kHz FL=832.
EXTRA_GRIDS = (
    (8000, 256, 'R8K'),
    (16000, 256, 'R16K256'),
    (48000, 1024, 'R48K'),
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


def _f32(arr):
    return np.ascontiguousarray(arr, dtype=np.float32)


def _is_prefix(short, long):
    """True when `short` is bytewise the leading len(short) floats of `long`."""
    a, b = _f32(short), _f32(long)
    return len(a) <= len(b) and a.tobytes() == b[:len(a)].tobytes()


EMR_COMMON = 'AECB_SG_EMR_TR_COMMON'


def emit_array_or_alias(w, name, arr, bases):
    """Emit `arr` as `name`, or as a macro naming an always-emitted table in `bases`
    ([(macro, array)]) that starts with exactly these floats. A consumer reads only
    the first len(arr) entries (table_len / block_size), so a longer table is safe."""
    for base_name, base_arr in bases:
        if _is_prefix(arr, base_arr):
            w('#define %s %s /* first %d floats */\n' % (name, base_name, len(arr)))
            return
    w(emit_array(name, arr))


def _plan_shared_tables(cap16, extra_caps):
    """Tables emitted once and referenced by name from several grids.

    emr: every *_EMR_TR table of every grid is a prefix of the longest one (they are the
    constant 0.3); emitted once, ahead of every rate block.
    """
    all_caps = [cap16] + [extra_caps[(sr, fs)] for sr, fs, _t in EXTRA_GRIDS]
    emr = [c['arrays'][n] for c in all_caps for n in ('SG_NEAREND_EMR_TR', 'SG_NORMAL_EMR_TR')]
    emr_base = _f32(max(emr, key=len))
    shared_emr = emr_base if all(_is_prefix(a, emr_base) for a in emr) else None
    return dict(emr=shared_emr)


# ---- compressed tables ------------------------------------------------------------------------------------------------

SG_RLE_TABLES = ('SG_NEAREND_ENR_TR', 'SG_NEAREND_ENR_SU', 'SG_NORMAL_ENR_TR', 'SG_NORMAL_ENR_SU')


def window_build(half, n):
    """The C builder (aec_synth_window_build): w[i] = half[i * stride] for i <= n/2, w[n - i] = w[i] above."""
    half = _f32(half)
    n_full = 2 * (len(half) - 1)
    assert n_full % n == 0, (n_full, n)
    stride = n_full // n
    w = np.zeros(n, dtype=np.float32)
    for i in range(n // 2 + 1):
        w[i] = half[i * stride]
    for i in range(n // 2 + 1, n):
        w[i] = w[n - i]
    return w


def window_half(win_by_n):
    """Half table of the longest captured window; asserts that every captured window is rebuilt bit for bit from it
    (symmetry and stride subsampling), so the C builder needs no per-grid table."""
    n_max = max(win_by_n)
    half = _f32(win_by_n[n_max])[:n_max // 2 + 1].copy()
    for n, win in sorted(win_by_n.items()):
        built = window_build(half, n)
        assert built.tobytes() == _f32(win).tobytes(), 'synthesis window n=%d is not rebuilt from the half table' % n
    return half


def rle_encode(tables):
    """Run-length code of equally long tables with shared run boundaries: (ends, vals) where run r covers bins
    [ends[r-1], ends[r]) (ends[-1] = 0), ends[-1] == n, vals[t][r] is table t's value on run r. Compared on float32 bits."""
    tabs = [_f32(t) for t in tables]
    n = len(tabs[0])
    assert all(len(t) == n for t in tabs)
    bits = [t.view(np.uint32) for t in tabs]
    ends = [k for k in range(1, n) if any(b[k] != b[k - 1] for b in bits)] + [n]
    assert ends[-1] < 65536 and n < 65536
    starts = [0] + ends[:-1]
    vals = [[float(t[s]) for s in starts] for t in tabs]
    return ends, vals


def rle_decode(ends, vals, n):
    """The C decoder (aec_sg_tables_decode): fills bins [0, n) run by run, stopping at n."""
    out = [np.zeros(n, dtype=np.float32) for _ in vals]
    for t, v in enumerate(vals):
        k = 0
        for r, e in enumerate(ends):
            if k >= n:
                break
            e = min(e, n)
            while k < e:
                out[t][k] = np.float32(v[r])
                k += 1
    return out


def rle_checked(cap):
    """(ends, vals) for one grid's four ENR/SU tables, asserting decode(encode(x)) == x."""
    tabs = [cap['arrays'][n] for n in SG_RLE_TABLES]
    ends, vals = rle_encode(tabs)
    for t, got in zip(tabs, rle_decode(ends, vals, len(tabs[0]))):
        assert got.tobytes() == _f32(t).tobytes(), 'ENR/SU run-length code does not round-trip'
    return ends, vals


def rle_is_alias(base, cap):
    """True when decoding `base` (ends, vals) clipped to this grid's length reproduces its four ENR/SU tables."""
    ends, vals = base
    tabs = [cap['arrays'][n] for n in SG_RLE_TABLES]
    n = len(tabs[0])
    return all(g.tobytes() == _f32(t).tobytes() for t, g in zip(tabs, rle_decode(ends, vals, n)))


def emit_rle(w, prefix, rle):
    """Emits <prefix>SG_RUN_END (uint16 run ends), <prefix>SG_RUN_VAL (4 x runs floats, table-major) and <prefix>SG_RUNS."""
    ends, vals = rle
    w('static const uint16_t %sSG_RUN_END[%d] = { %s };\n' % (prefix, len(ends), ', '.join(str(e) for e in ends)))
    flat = [v for row in vals for v in row]
    w(emit_array('%sSG_RUN_VAL' % prefix, flat))
    w('#define %sSG_RUNS %d\n' % (prefix, len(ends)))


def _lf_clamp_comment(n_bins, sr, val):
    """lf_clamp_bin comment text for (n_bins, sr), e.g. n_bins=257, sr=16000 -> 8."""
    return ('/* hz_to_bin(250 Hz, n_bins=%d, sr=%d) = round(250*2*(%d-1)/%d) = %d */\n'
            % (n_bins, sr, n_bins, sr, val))


def _capture_all(sr, frame_size, filter_length=None):
    """Instantiate the live Python balanced AEC at `sr` and return every value the
    header emitter needs, keyed by the AECB_* macro name it feeds (sans
    prefix/tag). This dict is the single source of truth for both the emitters and
    the cross-rate invariance assertion.

    filter_length=None means "don't pass filter_length", so AecConfig's
    __post_init__ auto policy applies; only the 16 kHz call passes an explicit 832.
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
    # FILTER_LENGTH has no unsuffixed macro; the 16k row of the rate table uses the
    # literal 832.
    values['FILTER_LENGTH'] = int(cfg.filter_length)

    # ── driver flags ──
    values['ERLE_COH_GATE_ENABLED'] = int(bool(cfg.erle_coh_gate_enabled))
    values['ERLE_WINDOWED_CAPTURE_PSD'] = int(bool(getattr(cfg, 'erle_windowed_capture_psd', False)))
    values['ERLE_RENDER_X2_PSD_SCALE'] = int(bool(getattr(cfg, 'erle_render_x2_psd_scale', False)))
    values['OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE'] = int(
        bool(getattr(cfg, 'output_capture_when_linear_unusable', False)))
    values['USE_STATIONARITY_PROPERTIES'] = int(
        bool(aec._aec3_sg_config.echo_audibility.use_stationarity_properties))
    values['STATIONARITY_CONVERGE_HOPS'] = int(aec._aec3_stationarity_converge_hops)
    # Hardcoded literal in orchestrator.py (_ar_thr = 5.96e-4), not derived from
    # sample_rate/hop_size, so rate-invariant (checked by the cross-rate assertion).
    values['ACTIVE_RENDER_THRESHOLD'] = 5.96e-4

    # ── CNG constants ──
    values['CNG_Y2_ALPHA'] = float(aec._aec3_cng_y2_alpha)
    values['CNG_N2_TRACK_FRESHNESS'] = float(aec._aec3_cng_n2_track_freshness)
    values['CNG_N2_TRACK_RETENTION'] = float(aec._aec3_cng_n2_track_retention)
    values['CNG_N2_SLOW_UP'] = float(aec._aec3_cng_n2_slow_up)
    values['CNG_N2_INITIAL_ALPHA'] = float(aec._aec3_cng_n2_initial_alpha)
    values['CNG_N2_UPDATE_ONSET_HOPS'] = int(aec._aec3_cng_n2_update_onset_hops)
    values['CNG_N2_INITIAL_DURATION_HOPS'] = int(aec._aec3_cng_n2_initial_duration_hops)
    # genuinely rate-varying (fft_size/2 density scale of orchestrator.py's
    # _aec3_noise_floor_int16sq) -- see RATE_VARYING_MACROS.
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
    values['REE_MODEL_REVERB_IN_NL'] = int(rc['model_reverb_in_nl'])
    values['REE_ERLE_ONSET_COMP'] = int(rc['erle_onset_comp'])
    values['REE_REVERB_ENABLED'] = int(rc['reverb_enabled'])
    values['REE_USE_AEC_RESIDUAL_NOISE_GATE'] = int(rc['use_aec3_residual_noise_gate'])
    values['REE_USE_AEC_ECHO_GEN_WINDOW'] = int(rc['use_aec3_echo_gen_window'])
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
    # Dimensionless multiplier on the already-computed direct term -- a fixed
    # AecConfig constant, not rate-varying (unlike REE_NL_NORM_POWER above).
    values['REE_NL_R2_CAP_MULTIPLE'] = float(rc['nl_r2_cap_multiple'])
    # NOTE: genuinely rate-varying (derived from the same density scale as
    # REE_MIN_NOISE_FLOOR_POWER) -- see RATE_VARYING_MACROS.
    values['REE_RESIDUAL_NOISE_GATE_POWER'] = float(rc['residual_noise_gate_power'])
    values['REE_REVERB_SMOOTHING_BASE'] = float(rc['reverb_smoothing_base'])

    # ── SuppressionGain config ──
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


# Declared rate-varying macro names -- values legitimately differ across
# 8k/16k/48k: geometry, Hz-anchored bin indices, and the absolute-power constants
# density-scaled by fft_size/2 (or frame_size) relative to AEC3's fixed
# 64-sample/128-bin reference (the EchoAudibility / ResidualEchoEstimator / CNG
# constants listed below), which move with sample rate like the bin geometry.
RATE_VARYING_MACROS = {
    'N_BINS', 'FFT_SIZE', 'BLOCK_SIZE', 'HOP_SIZE', 'N_PARTITIONS',
    'FILTER_TAPS_SIZE', 'FILTER_LENGTH',
    'SG_LAST_LF_SMOOTHING_BAND',
    'SG_LF_CLAMP_BIN', 'SG_DNE_LF_END', 'SG_DNE_LF_ENDPOINT_BIN',
    'SG_AUD_LF_END_BIN', 'SG_AUD_MF_END_BIN', 'SG_HF_LGB',
    # density-scaled absolute-power constants:
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

    Aborts (SystemExit) if any macro/array NOT declared rate-varying differs across
    sample rates, so a 16 kHz-only value is never baked in as a constant.
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


def _emr_bases(shared):
    """The shared EMR table as an alias base ([] when the grids do not share one)."""
    return [(EMR_COMMON, shared['emr'])] if shared['emr'] is not None else []


def emit_legacy_block(w, cap, shared, rle):
    """Emits the unsuffixed (16 kHz) block."""
    v = cap['values']

    # ── geometry ──
    w('/* geometry */\n')
    w('#define AECB_N_BINS        %d\n' % v['N_BINS'])
    w('#define AECB_FFT_SIZE      %d\n' % v['FFT_SIZE'])
    w('#define AECB_BLOCK_SIZE    %d\n' % v['BLOCK_SIZE'])
    w('#define AECB_HOP_SIZE      %d\n' % v['HOP_SIZE'])
    w('#define AECB_N_PARTITIONS  %d\n' % v['N_PARTITIONS'])
    w('#define AECB_FILTER_TAPS_SIZE %d\n\n' % v['FILTER_TAPS_SIZE'])

    # ── driver flags ──
    w('/* aec_post driver flags */\n')
    w('#define AECB_ERLE_COH_GATE_ENABLED  %d\n' % v['ERLE_COH_GATE_ENABLED'])
    w('#define AECB_ERLE_WINDOWED_CAPTURE_PSD %d\n' % v['ERLE_WINDOWED_CAPTURE_PSD'])
    w('#define AECB_ERLE_RENDER_X2_PSD_SCALE %d\n' % v['ERLE_RENDER_X2_PSD_SCALE'])
    w('#define AECB_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE %d\n' % v['OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE'])
    w('#define AECB_USE_STATIONARITY_PROPERTIES %d\n' % v['USE_STATIONARITY_PROPERTIES'])
    w('#define AECB_STATIONARITY_CONVERGE_HOPS %d\n' % v['STATIONARITY_CONVERGE_HOPS'])
    w('#define AECB_ACTIVE_RENDER_THRESHOLD (%s)\n\n' % f64(v['ACTIVE_RENDER_THRESHOLD']))

    # ── CNG constants ──
    w('/* CNG wall-clock-rescaled constants */\n')
    w('#define AECB_CNG_Y2_ALPHA (%s)\n' % f64(v['CNG_Y2_ALPHA']))
    w('#define AECB_CNG_N2_TRACK_FRESHNESS (%s)\n' % f64(v['CNG_N2_TRACK_FRESHNESS']))
    w('#define AECB_CNG_N2_TRACK_RETENTION (%s)\n' % f64(v['CNG_N2_TRACK_RETENTION']))
    w('#define AECB_CNG_N2_SLOW_UP (%s)\n' % f64(v['CNG_N2_SLOW_UP']))
    w('#define AECB_CNG_N2_INITIAL_ALPHA (%s)\n' % f64(v['CNG_N2_INITIAL_ALPHA']))
    w('#define AECB_CNG_N2_UPDATE_ONSET_HOPS %d\n' % v['CNG_N2_UPDATE_ONSET_HOPS'])
    w('#define AECB_CNG_N2_INITIAL_DURATION_HOPS %d\n' % v['CNG_N2_INITIAL_DURATION_HOPS'])
    w('#define AECB_NOISE_FLOOR_INT16SQ (%s)\n' % f64(v['NOISE_FLOOR_INT16SQ']))
    w('#define AECB_ERLE_COH_GATE_ALPHA (%s)\n' % f64(v['ERLE_COH_GATE_ALPHA']))
    w('#define AECB_ERLE_COH_GATE_THRESHOLD (%s)\n\n' % f64(v['ERLE_COH_GATE_THRESHOLD']))

    # ── sqrt2 sin LUT (the synthesis window is rebuilt at init: see emit_window_half) ──
    w(emit_array('AECB_SQRT2_SIN_LUT', cap['arrays']['SQRT2_SIN_LUT']))
    w('\n')

    # ── AecState config ──
    w('/* AecState config */\n')
    w('#define AECB_ST_NUM_CAPTURE_CHANNELS %d\n' % v['ST_NUM_CAPTURE_CHANNELS'])
    w('#define AECB_ST_ENABLE_FILTER_ANALYZER %d\n' % v['ST_ENABLE_FILTER_ANALYZER'])
    # -1 == AEC_STATE_STARTUP_HOPS_AUTO (aec_state.c): the BALANCED preset never
    # overrides erle_startup_hops/erl_startup_hops, so this stays the 'auto' sentinel
    # resolved per grid at aec_state_init() time, not a literal hop count (keeps the
    # 8k/16k/48k call sites auto-scaling).
    w('#define AECB_ST_ERLE_STARTUP_HOPS %d\n' % v['ST_ERLE_STARTUP_HOPS'])
    w('#define AECB_ST_ERL_STARTUP_HOPS %d\n' % v['ST_ERL_STARTUP_HOPS'])
    w('#define AECB_ST_ECHO_CAN_SATURATE %d\n' % v['ST_ECHO_CAN_SATURATE'])
    w('#define AECB_ST_USE_LINEAR_FILTER %d\n' % v['ST_USE_LINEAR_FILTER'])
    w('#define AECB_ST_CONSERVATIVE_INITIAL_PHASE %d\n' % v['ST_CONSERVATIVE_INITIAL_PHASE'])
    w('#define AECB_ST_DELAY_HEADROOM_SAMPLES %d\n' % v['ST_DELAY_HEADROOM_SAMPLES'])
    w('#define AECB_ST_INITIAL_STATE_SECONDS (%s)\n' % f64(v['ST_INITIAL_STATE_SECONDS']))
    w('#define AECB_ST_ERLE_MIN (%s)\n' % f64(v['ST_ERLE_MIN']))
    w('#define AECB_ST_ERLE_MAX_L (%s)\n' % f64(v['ST_ERLE_MAX_L']))
    w('#define AECB_ST_ERLE_MAX_H (%s)\n\n' % f64(v['ST_ERLE_MAX_H']))

    # ── ResidualEchoEstimator config ──
    w('/* ResidualEchoEstimator config */\n')
    w('#define AECB_REE_MODEL_REVERB_IN_NL %d\n' % v['REE_MODEL_REVERB_IN_NL'])
    w('#define AECB_REE_ERLE_ONSET_COMP %d\n' % v['REE_ERLE_ONSET_COMP'])
    w('#define AECB_REE_REVERB_ENABLED %d\n' % v['REE_REVERB_ENABLED'])
    w('#define AECB_REE_USE_AEC_RESIDUAL_NOISE_GATE %d\n' % v['REE_USE_AEC_RESIDUAL_NOISE_GATE'])
    w('#define AECB_REE_USE_AEC_ECHO_GEN_WINDOW %d\n' % v['REE_USE_AEC_ECHO_GEN_WINDOW'])
    w('#define AECB_REE_NL_R2_ENABLED %d\n' % v['REE_NL_R2_ENABLED'])
    w('#define AECB_REE_NOISE_FLOOR_HOLD_HOPS %d\n' % v['REE_NOISE_FLOOR_HOLD_HOPS'])
    w('#define AECB_REE_USE_FREQ_RESPONSE %d\n' % v['REE_USE_FREQ_RESPONSE'])
    w('#define AECB_REE_REVERB_USE_CONSERVATIVE %d\n' % v['REE_REVERB_USE_CONSERVATIVE'])
    w('#define AECB_REE_MIN_NOISE_FLOOR_POWER (%s)\n' % f64(v['REE_MIN_NOISE_FLOOR_POWER']))
    w('#define AECB_REE_NOISE_GATE_POWER_LEGACY (%s)\n' % f64(v['REE_NOISE_GATE_POWER_LEGACY']))
    w('#define AECB_REE_NOISE_GATE_SLOPE (%s)\n' % f64(v['REE_NOISE_GATE_SLOPE']))
    w('#define AECB_REE_STATIONARY_GATE_SLOPE (%s)\n' % f64(v['REE_STATIONARY_GATE_SLOPE']))
    w('#define AECB_REE_DEFAULT_GAIN (%s)\n' % f64(v['REE_DEFAULT_GAIN']))
    w('#define AECB_REE_TM_GAIN (%s)\n' % f64(v['REE_TM_GAIN']))
    w('#define AECB_REE_REVERB_DECAY (%s)\n' % f64(v['REE_REVERB_DECAY']))
    w('#define AECB_REE_REVERB_MILD_SCALE (%s)\n' % f64(v['REE_REVERB_MILD_SCALE']))
    w('#define AECB_REE_REVERB_TAIL_STRENGTH (%s)\n' % f64(v['REE_REVERB_TAIL_STRENGTH']))
    w('#define AECB_REE_NL_R2_ALPHA (%s)\n' % f64(v['REE_NL_R2_ALPHA']))
    w('#define AECB_REE_NL_NORM_POWER (%s)\n' % f64(v['REE_NL_NORM_POWER']))
    w('#define AECB_REE_NL_R2_CAP_MULTIPLE (%s)\n' % f64(v['REE_NL_R2_CAP_MULTIPLE']))
    w('#define AECB_REE_RESIDUAL_NOISE_GATE_POWER (%s)\n' % f64(v['REE_RESIDUAL_NOISE_GATE_POWER']))
    w('#define AECB_REE_REVERB_SMOOTHING_BASE (%s)\n\n' % f64(v['REE_REVERB_SMOOTHING_BASE']))

    # ── SuppressionGain config ──
    w('/* SuppressionGain config (split_floor_far_active is the preset axis,\n')
    w(' * recomputed at runtime; the baked value is the balanced -28 dB one). */\n')
    w('#define AECB_SG_SPLIT_FLOOR_FAR_SILENT (%s)\n' % f64(v['SG_SPLIT_FLOOR_FAR_SILENT']))
    w('#define AECB_SG_SPLIT_FLOOR_LATCH_POWER (%s)\n' % f64(v['SG_SPLIT_FLOOR_LATCH_POWER']))
    w('#define AECB_SG_LOW_RENDER_THRESHOLD (%s)\n' % f64(v['SG_LOW_RENDER_THRESHOLD']))
    w('#define AECB_SG_MAX_INC (%s)\n' % f64(v['SG_MAX_INC']))
    w('#define AECB_SG_MAX_DEC_LF (%s)\n' % f64(v['SG_MAX_DEC_LF']))
    w('#define AECB_SG_SOFT_BLEND_ENABLED %d\n' % v['SG_SOFT_BLEND_ENABLED'])
    w('#define AECB_SG_SOFT_BLEND_PER_BIN %d\n' % v['SG_SOFT_BLEND_PER_BIN'])
    w('#define AECB_SG_SOFT_BLEND_ENR_THR (%s)\n' % f64(v['SG_SOFT_BLEND_ENR_THR']))
    w('#define AECB_SG_SOFT_BLEND_SOFTNESS (%s)\n' % f64(v['SG_SOFT_BLEND_SOFTNESS']))
    w('#define AECB_SG_LAST_LF_SMOOTHING_BAND %d\n' % v['SG_LAST_LF_SMOOTHING_BAND'])
    w('#define AECB_SG_LAST_PERMANENT %d\n' % v['SG_LAST_PERMANENT'])
    w('#define AECB_SG_LF_SMOOTHING_INITIAL %d\n' % v['SG_LF_SMOOTHING_INITIAL'])
    w(cap['lf_clamp_comment'])
    w('#define AECB_SG_LF_CLAMP_BIN %d\n' % v['SG_LF_CLAMP_BIN'])
    w('#define AECB_SG_DNE_LF_END %d\n' % v['SG_DNE_LF_END'])
    w('#define AECB_SG_TRIGGER_THRESHOLD_HOPS %d\n' % v['SG_TRIGGER_THRESHOLD_HOPS'])
    w('#define AECB_SG_HOLD_DURATION_HOPS %d\n' % v['SG_HOLD_DURATION_HOPS'])
    w('#define AECB_SG_NEAREND_SMOOTHER_N %d\n' % v['SG_NEAREND_SMOOTHER_N'])
    w('#define AECB_SG_AUD_LF_END_BIN %d\n' % v['SG_AUD_LF_END_BIN'])
    w('#define AECB_SG_AUD_MF_END_BIN %d\n' % v['SG_AUD_MF_END_BIN'])
    w('#define AECB_SG_FLOOR_POWER (%s)\n' % f64(v['SG_FLOOR_POWER']))
    w('#define AECB_SG_AUD_THR_LF (%s)\n' % f64(v['SG_AUD_THR_LF']))
    w('#define AECB_SG_AUD_THR_MF (%s)\n' % f64(v['SG_AUD_THR_MF']))
    w('#define AECB_SG_AUD_THR_HF (%s)\n' % f64(v['SG_AUD_THR_HF']))
    w('#define AECB_SG_LOW_RENDER_LIMIT (%s)\n' % f64(v['SG_LOW_RENDER_LIMIT']))
    w('#define AECB_SG_NORMAL_RENDER_LIMIT (%s)\n' % f64(v['SG_NORMAL_RENDER_LIMIT']))
    w('#define AECB_SG_HF_LGB %d\n' % v['SG_HF_LGB'])
    w('#define AECB_SG_HF_BIQ %d\n' % v['SG_HF_BIQ'])
    w('#define AECB_SG_CONSERVATIVE_HF %d\n' % v['SG_CONSERVATIVE_HF'])
    w('#define AECB_SG_DNE_ENR_THRESHOLD (%s)\n' % f64(v['SG_DNE_ENR_THRESHOLD']))
    w('#define AECB_SG_DNE_ENR_EXIT_THRESHOLD (%s)\n' % f64(v['SG_DNE_ENR_EXIT_THRESHOLD']))
    w('#define AECB_SG_DNE_SNR_THRESHOLD (%s)\n' % f64(v['SG_DNE_SNR_THRESHOLD']))
    w('#define AECB_SG_DNE_USE_DURING_INITIAL_PHASE %d\n' % v['SG_DNE_USE_DURING_INITIAL_PHASE'])
    w('#define AECB_SG_DNE_USE_UNBOUNDED_ECHO %d\n' % v['SG_DNE_USE_UNBOUNDED_ECHO'])
    w('#define AECB_SG_DNE_LF_ENDPOINT_BIN %d\n\n' % v['SG_DNE_LF_ENDPOINT_BIN'])

    # ── per-bin tuning arrays ──
    w('/* SuppressionGain per-bin tuning arrays (LF/HF interpolated). The EMR threshold is the\n')
    w(' * same constant for every table of every grid: one shared table sized for the longest grid. */\n')
    emr_bases = _emr_bases(shared)
    if emr_bases:
        w(emit_array(EMR_COMMON, shared['emr']))
    for n in ('SG_NEAREND_EMR_TR', 'SG_NORMAL_EMR_TR'):
        emit_array_or_alias(w, 'AECB_' + n, cap['arrays'][n], emr_bases)
    w('/* ENR/SU thresholds, run-length coded (aec_sg_tables_decode() expands them into the pool): run r covers bins\n')
    w(' * [SG_RUN_END[r-1], SG_RUN_END[r]) with SG_RUN_END[-1] = 0; SG_RUN_VAL holds SG_RUNS floats for each of\n')
    w(' * nearend_enr_tr, nearend_enr_su, normal_enr_tr, normal_enr_su, in that order. */\n')
    emit_rle(w, 'AECB_', rle)


def emit_rate_block(w, tag, cap, emr_bases, base_rle, own_rle):
    """Emits the AECB_<tag>_* block for one extra rate (8k/48k): only the
    rate-varying subset.
    """
    v = cap['values']
    sr = cap['sr']

    w('/* ---- %d Hz rate block (AUTO-GENERATED) ----\n'
      % sr)
    w(' * Same live-capture mechanism as the 16 kHz block above, at\n')
    w(' * sample_rate=%d (filter_length via AecConfig.__post_init__\'s own\n'
      % sr)
    w(' * auto policy, not hardcoded). Only the rate-varying subset is\n')
    w(' * duplicated here; everything else is shared with the 16 kHz block\n')
    w(' * (verified equal by this generator\'s cross-rate invariance\n')
    w(' * assertion -- see RATE_VARYING_MACROS/RATE_VARYING_ARRAYS). */\n')

    w('#define AECB_%s_N_BINS        %d\n' % (tag, v['N_BINS']))
    w('#define AECB_%s_FFT_SIZE      %d\n' % (tag, v['FFT_SIZE']))
    w('#define AECB_%s_BLOCK_SIZE    %d\n' % (tag, v['BLOCK_SIZE']))
    w('#define AECB_%s_HOP_SIZE      %d\n' % (tag, v['HOP_SIZE']))
    w('#define AECB_%s_N_PARTITIONS  %d\n' % (tag, v['N_PARTITIONS']))
    w('#define AECB_%s_FILTER_TAPS_SIZE %d\n' % (tag, v['FILTER_TAPS_SIZE']))
    w('#define AECB_%s_FILTER_LENGTH %d\n\n' % (tag, v['FILTER_LENGTH']))

    w('#define AECB_%s_SYNTH_WINDOW_LEN %d\n\n' % (tag, len(cap['arrays']['SYNTH_WINDOW'])))

    w('/* Hz-anchored SuppressionGain bin indices (rate-varying). */\n')
    w('#define AECB_%s_SG_LAST_LF_SMOOTHING_BAND %d\n' % (tag, v['SG_LAST_LF_SMOOTHING_BAND']))
    w(_lf_clamp_comment(v['N_BINS'], sr, v['SG_LF_CLAMP_BIN']))
    w('#define AECB_%s_SG_LF_CLAMP_BIN %d\n' % (tag, v['SG_LF_CLAMP_BIN']))
    w('#define AECB_%s_SG_DNE_LF_END %d\n' % (tag, v['SG_DNE_LF_END']))
    w('#define AECB_%s_SG_DNE_LF_ENDPOINT_BIN %d\n' % (tag, v['SG_DNE_LF_ENDPOINT_BIN']))
    w('#define AECB_%s_SG_AUD_LF_END_BIN %d\n' % (tag, v['SG_AUD_LF_END_BIN']))
    w('#define AECB_%s_SG_AUD_MF_END_BIN %d\n' % (tag, v['SG_AUD_MF_END_BIN']))
    w('#define AECB_%s_SG_HF_LGB %d\n\n' % (tag, v['SG_HF_LGB']))

    w('/* Density-scaled absolute-power constants -- genuinely rate-varying, found\n')
    w(' * by the cross-rate invariance assertion. */\n')
    w('#define AECB_%s_SG_LOW_RENDER_THRESHOLD (%s)\n' % (tag, f64(v['SG_LOW_RENDER_THRESHOLD'])))
    w('#define AECB_%s_SG_FLOOR_POWER (%s)\n' % (tag, f64(v['SG_FLOOR_POWER'])))
    w('#define AECB_%s_SG_LOW_RENDER_LIMIT (%s)\n' % (tag, f64(v['SG_LOW_RENDER_LIMIT'])))
    w('#define AECB_%s_SG_NORMAL_RENDER_LIMIT (%s)\n' % (tag, f64(v['SG_NORMAL_RENDER_LIMIT'])))
    w('#define AECB_%s_REE_MIN_NOISE_FLOOR_POWER (%s)\n' % (tag, f64(v['REE_MIN_NOISE_FLOOR_POWER'])))
    w('#define AECB_%s_REE_NL_NORM_POWER (%s)\n' % (tag, f64(v['REE_NL_NORM_POWER'])))
    w('#define AECB_%s_REE_RESIDUAL_NOISE_GATE_POWER (%s)\n' % (tag, f64(v['REE_RESIDUAL_NOISE_GATE_POWER'])))
    w('#define AECB_%s_NOISE_FLOOR_INT16SQ (%s)\n\n' % (tag, f64(v['NOISE_FLOOR_INT16SQ'])))

    w('/* Wall-clock-rescaled constants for this exact hop duration. */\n')
    for name in ('STATIONARITY_CONVERGE_HOPS',
                 'CNG_N2_UPDATE_ONSET_HOPS', 'CNG_N2_INITIAL_DURATION_HOPS',
                 'REE_NOISE_FLOOR_HOLD_HOPS', 'SG_TRIGGER_THRESHOLD_HOPS',
                 'SG_HOLD_DURATION_HOPS', 'SG_NEAREND_SMOOTHER_N'):
        w('#define AECB_%s_%s %d\n' % (tag, name, v[name]))
    for name in ('CNG_Y2_ALPHA', 'CNG_N2_TRACK_FRESHNESS',
                 'CNG_N2_TRACK_RETENTION', 'CNG_N2_SLOW_UP',
                 'CNG_N2_INITIAL_ALPHA', 'REE_REVERB_SMOOTHING_BASE',
                 'SG_MAX_INC', 'SG_MAX_DEC_LF'):
        w('#define AECB_%s_%s (%s)\n' % (tag, name, f64(v[name])))
    w('\n')

    w('/* SuppressionGain per-bin tuning at %d Hz (length = n_bins). The EMR tables are #defines naming the shared\n' % sr)
    w(' * table; the ENR/SU run-length code is either its own or the 16 kHz one, which decodes to these tables when\n')
    w(' * clipped to this length. */\n')
    for n in ('SG_NEAREND_EMR_TR', 'SG_NORMAL_EMR_TR'):
        emit_array_or_alias(w, 'AECB_%s_%s' % (tag, n), cap['arrays'][n], emr_bases)
    if own_rle is not None:
        emit_rle(w, 'AECB_%s_' % tag, own_rle)
    else:
        for part in ('RUN_END', 'RUN_VAL', 'RUNS'):
            w('#define AECB_%s_SG_%s AECB_SG_%s\n' % (tag, part, part))
    w('#define AECB_%s_SG_TABLE_LEN %d\n' % (tag, v['N_BINS']))
    w('\n')


def window_sets(caps):
    """{n: window} over every captured grid; grids of equal length must have bitwise equal windows."""
    by_n = {}
    for cap in caps:
        win = _f32(cap['arrays']['SYNTH_WINDOW'])
        n = len(win)
        assert by_n.setdefault(n, win).tobytes() == win.tobytes(), 'windows of length %d differ across grids' % n
    return by_n


def emit_window_half(w, by_n):
    """The half table of the longest window (the 48 kHz grid's)."""
    full_half = window_half(by_n)
    w('/* Synthesis window: the first n/2 + 1 floats of the longest grid\'s window. aec_synth_window_build() rebuilds\n')
    w(' * the window of any grid length n (a power of two dividing AECB_SYNTH_HALF_N) bit for bit: w[i] = half[i * (HALF_N / n)]\n')
    w(' * for i <= n/2, w[n - i] = w[i] above. This generator checks every captured window against that rule. */\n')
    w('#define AECB_SYNTH_HALF_N %d\n' % max(by_n))
    w(emit_array('AECB_SYNTH_WINDOW_HALF', full_half))
    w('\n')


def ref_header_lines(grids):
    """grids: [(tag, cap)]. The original full tables, for tests only (never compiled into the library)."""
    out = []
    w = out.append
    w('/* aec_ref_tables.h -- AUTO-GENERATED by python/diag/gen_aec_balanced_config_h.py. DO NOT EDIT BY HAND.\n')
    w(' *\n')
    w(' * TEST ONLY, not part of libaec: the full synthesis windows and ENR/SU thresholds captured from the Python twin, i.e. what\n')
    w(' * aec_init() must rebuild from the compressed tables in include/internal/aec_balanced_config.h. The unit tests compare against these, so\n')
    w(' * they do not check the decoders with themselves. */\n')
    w('#ifndef AEC_REF_TABLES_H\n#define AEC_REF_TABLES_H\n\n')
    for tag, cap in grids:
        w(emit_array('AECREF_%s_WINDOW' % tag, cap['arrays']['SYNTH_WINDOW']))
        for n in SG_RLE_TABLES:
            w(emit_array('AECREF_%s_%s' % (tag, n), cap['arrays'][n]))
    w('\ntypedef struct {\n')
    w('    int sample_rate, fft_size, n_bins;\n')
    w('    const float* window;   /* fft_size floats */\n')
    w('    const float *nearend_enr_tr, *nearend_enr_su, *normal_enr_tr, *normal_enr_su;   /* n_bins floats each */\n')
    w('} AecRefTables;\n\n')
    w('static const AecRefTables AECREF_GRIDS[%d] __attribute__((unused)) = {\n' % len(grids))
    for tag, cap in grids:
        w('    { %d, %d, %d, AECREF_%s_WINDOW, AECREF_%s_SG_NEAREND_ENR_TR, AECREF_%s_SG_NEAREND_ENR_SU,\n'
          '      AECREF_%s_SG_NORMAL_ENR_TR, AECREF_%s_SG_NORMAL_ENR_SU },\n'
          % (cap['sr'], int(cap['values']['FFT_SIZE']), int(cap['values']['N_BINS']), tag, tag, tag, tag, tag))
    w('};\n#define AECREF_N_GRIDS %d\n\n#endif /* AEC_REF_TABLES_H */\n' % len(grids))
    return out


def emit_rate_table(w):
    """Emits the AecBalancedRateDims struct + AECB_RATE_TABLE lookup array
    + aecb_rate_cfg() accessor. The 16 kHz row references the unsuffixed
    macros/arrays directly.
    """
    w('/* ------------------------------------------------------------------\n')
    w(' * Multi-rate lookup. Static const per including TU, like the arrays\n')
    w(' * above; what a TU does not use is discarded.\n')
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
    w('    int synth_window_len;\n')
    w('    const uint16_t* sg_run_end;\n')
    w('    const float* sg_run_val;\n')
    w('    int sg_runs;\n')
    w('    const float* sg_nearend_emr_tr;\n')
    w('    const float* sg_normal_emr_tr;\n')
    w('    int sg_table_len;\n')
    w('    int sg_last_lf_smoothing_band;\n')
    w('    int sg_lf_clamp_bin;\n')
    w('    int sg_dne_lf_end;\n')
    w('    int sg_dne_lf_endpoint_bin;\n')
    w('    int sg_aud_lf_end_bin;\n')
    w('    int sg_aud_mf_end_bin;\n')
    w('    int sg_hf_lgb;\n')
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
    # double, not float: ree_init()'s smoothing_base flows straight into a
    # float64 EMA (ReverbFrequencyResponse.average_decay) that must match
    # Python's plain-float _smoothing_base bit-for-bit; narrowing it to
    # float32 here would re-round an already-correct double and bias every
    # EMA step. See reverb_frequency_response.h.
    w('    double ree_reverb_smoothing_base;\n')
    w('    float sg_max_inc;\n')
    w('    float sg_max_dec_lf;\n')
    w('    int sg_trigger_threshold_hops;\n')
    w('    int sg_hold_duration_hops;\n')
    w('    int sg_nearend_smoother_n;\n')
    w('} AecBalancedRateDims;\n\n')

    w('static const AecBalancedRateDims AECB_RATE_TABLE[] __attribute__((unused)) = {\n')
    w('    { /* 16000 Hz -- the unsuffixed block above. */\n')
    w('      16000,\n')
    w('      AECB_N_BINS, AECB_FFT_SIZE, AECB_BLOCK_SIZE, AECB_HOP_SIZE,\n')
    w('      AECB_N_PARTITIONS, AECB_FILTER_TAPS_SIZE, 832,\n')
    w('      AECB_BLOCK_SIZE,\n')
    w('      AECB_SG_RUN_END, AECB_SG_RUN_VAL, AECB_SG_RUNS,\n')
    w('      AECB_SG_NEAREND_EMR_TR, AECB_SG_NORMAL_EMR_TR,\n')
    w('      AECB_N_BINS,\n')
    w('      AECB_SG_LAST_LF_SMOOTHING_BAND,\n')
    w('      AECB_SG_LF_CLAMP_BIN, AECB_SG_DNE_LF_END, AECB_SG_DNE_LF_ENDPOINT_BIN,\n')
    w('      AECB_SG_AUD_LF_END_BIN, AECB_SG_AUD_MF_END_BIN, AECB_SG_HF_LGB,\n')
    w('      AECB_SG_LOW_RENDER_THRESHOLD, AECB_SG_FLOOR_POWER, AECB_SG_LOW_RENDER_LIMIT,\n')
    w('      AECB_SG_NORMAL_RENDER_LIMIT, AECB_REE_MIN_NOISE_FLOOR_POWER,\n')
    w('      AECB_REE_NL_NORM_POWER, AECB_REE_RESIDUAL_NOISE_GATE_POWER,\n')
    w('      AECB_NOISE_FLOOR_INT16SQ,\n')
    w('      AECB_STATIONARITY_CONVERGE_HOPS, AECB_CNG_Y2_ALPHA,\n')
    w('      AECB_CNG_N2_TRACK_FRESHNESS, AECB_CNG_N2_TRACK_RETENTION,\n')
    w('      AECB_CNG_N2_SLOW_UP, AECB_CNG_N2_INITIAL_ALPHA,\n')
    w('      AECB_CNG_N2_UPDATE_ONSET_HOPS, AECB_CNG_N2_INITIAL_DURATION_HOPS,\n')
    w('      AECB_REE_NOISE_FLOOR_HOLD_HOPS, AECB_REE_REVERB_SMOOTHING_BASE,\n')
    w('      AECB_SG_MAX_INC, AECB_SG_MAX_DEC_LF,\n')
    w('      AECB_SG_TRIGGER_THRESHOLD_HOPS, AECB_SG_HOLD_DURATION_HOPS,\n')
    w('      AECB_SG_NEAREND_SMOOTHER_N },\n')

    for sr, fft_size, tag in EXTRA_GRIDS:
        w('    { /* %d Hz */\n' % sr)
        w('      %d,\n' % sr)
        w('      AECB_%s_N_BINS, AECB_%s_FFT_SIZE, AECB_%s_BLOCK_SIZE, AECB_%s_HOP_SIZE,\n'
          % (tag, tag, tag, tag))
        w('      AECB_%s_N_PARTITIONS, AECB_%s_FILTER_TAPS_SIZE, AECB_%s_FILTER_LENGTH,\n'
          % (tag, tag, tag))
        w('      AECB_%s_SYNTH_WINDOW_LEN,\n' % tag)
        w('      AECB_%s_SG_RUN_END, AECB_%s_SG_RUN_VAL, AECB_%s_SG_RUNS,\n' % (tag, tag, tag))
        w('      AECB_%s_SG_NEAREND_EMR_TR, AECB_%s_SG_NORMAL_EMR_TR,\n' % (tag, tag))
        w('      AECB_%s_SG_TABLE_LEN,\n' % tag)
        w('      AECB_%s_SG_LAST_LF_SMOOTHING_BAND,\n' % tag)
        w('      AECB_%s_SG_LF_CLAMP_BIN, AECB_%s_SG_DNE_LF_END, AECB_%s_SG_DNE_LF_ENDPOINT_BIN,\n'
          % (tag, tag, tag))
        w('      AECB_%s_SG_AUD_LF_END_BIN, AECB_%s_SG_AUD_MF_END_BIN, AECB_%s_SG_HF_LGB,\n'
          % (tag, tag, tag))
        w('      AECB_%s_SG_LOW_RENDER_THRESHOLD, AECB_%s_SG_FLOOR_POWER, AECB_%s_SG_LOW_RENDER_LIMIT,\n'
          % (tag, tag, tag))
        w('      AECB_%s_SG_NORMAL_RENDER_LIMIT, AECB_%s_REE_MIN_NOISE_FLOOR_POWER,\n' % (tag, tag))
        w('      AECB_%s_REE_NL_NORM_POWER, AECB_%s_REE_RESIDUAL_NOISE_GATE_POWER,\n' % (tag, tag))
        w('      AECB_%s_NOISE_FLOOR_INT16SQ,\n' % tag)
        w('      AECB_%s_STATIONARITY_CONVERGE_HOPS, AECB_%s_CNG_Y2_ALPHA,\n' % (tag, tag))
        w('      AECB_%s_CNG_N2_TRACK_FRESHNESS, AECB_%s_CNG_N2_TRACK_RETENTION,\n' % (tag, tag))
        w('      AECB_%s_CNG_N2_SLOW_UP, AECB_%s_CNG_N2_INITIAL_ALPHA,\n' % (tag, tag))
        w('      AECB_%s_CNG_N2_UPDATE_ONSET_HOPS, AECB_%s_CNG_N2_INITIAL_DURATION_HOPS,\n' % (tag, tag))
        w('      AECB_%s_REE_NOISE_FLOOR_HOLD_HOPS, AECB_%s_REE_REVERB_SMOOTHING_BASE,\n' % (tag, tag))
        w('      AECB_%s_SG_MAX_INC, AECB_%s_SG_MAX_DEC_LF,\n' % (tag, tag))
        w('      AECB_%s_SG_TRIGGER_THRESHOLD_HOPS, AECB_%s_SG_HOLD_DURATION_HOPS,\n' % (tag, tag))
        w('      AECB_%s_SG_NEAREND_SMOOTHER_N },\n' % tag)

    w('};\n\n')

    w('static inline const AecBalancedRateDims* aecb_rate_cfg(int sample_rate, int fft_size) {\n')
    w('    int i;\n')
    w('    int n = (int)(sizeof(AECB_RATE_TABLE) / sizeof(AECB_RATE_TABLE[0]));\n')
    w('    for (i = 0; i < n; ++i) {\n')
    w('        if (AECB_RATE_TABLE[i].sample_rate == sample_rate &&\n')
    w('            AECB_RATE_TABLE[i].fft_size == fft_size) {\n')
    w('            return &AECB_RATE_TABLE[i];\n')
    w('        }\n')
    w('    }\n')
    w('    return (const AecBalancedRateDims*)0;\n')
    w('}\n')


def main():
    cap16 = _capture_all(SR, DEFAULT_FRAME, filter_length=FL)
    extra_caps = {
        (sr, fft_size): _capture_all(sr, fft_size, filter_length=None)
        for sr, fft_size, _tag in EXTRA_GRIDS
    }

    all_caps = dict(extra_caps)
    all_caps[(SR, DEFAULT_FRAME)] = cap16
    _assert_cross_rate_invariance(all_caps)

    lines = []
    w = lines.append
    w('/* aec_balanced_config.h — AUTO-GENERATED by\n')
    w(' * python/diag/gen_aec_balanced_config_h.py. DO NOT EDIT BY HAND.\n')
    w(' *\n')
    w(' * Baked BALANCED post-filter sub-module construction config, captured from the\n')
    w(' * live Python balanced instance (bit-exact source of truth).\n')
    w(' *\n')
    w(' * Multi-rate: the block below is 16 kHz only (unsuffixed macros/arrays).\n')
    w(' * The R8K- and R48K-prefixed blocks + the AECB_RATE_TABLE lookup extend\n')
    w(' * this to 8 kHz / 48 kHz -- see the generator\'s cross-rate invariance\n')
    w(' * assertion for the complete rate-varying field list.\n')
    w(' */\n')
    w('#ifndef AEC_BALANCED_CONFIG_H\n#define AEC_BALANCED_CONFIG_H\n\n#include <stdint.h>\n\n')

    shared = _plan_shared_tables(cap16, extra_caps)
    rle16 = rle_checked(cap16)
    emit_legacy_block(w, cap16, shared, rle16)
    w('\n')

    emit_window_half(w, window_sets(list(all_caps.values())))

    emr_bases = _emr_bases(shared)
    for sr, fft_size, tag in EXTRA_GRIDS:
        cap = extra_caps[(sr, fft_size)]
        own = None if rle_is_alias(rle16, cap) else rle_checked(cap)
        emit_rate_block(w, tag, cap, emr_bases, rle16, own)

    emit_rate_table(w)

    w('\n#endif /* AEC_BALANCED_CONFIG_H */\n')

    with open(OUT, 'w') as fh:
        fh.writelines(lines)
    os.makedirs(os.path.dirname(OUT_REF), exist_ok=True)
    with open(OUT_REF, 'w') as fh:
        fh.writelines(ref_header_lines([('R16K512', cap16)] + [(tag, extra_caps[(sr, fs)])
                                                               for sr, fs, tag in EXTRA_GRIDS]))
    print('wrote %s (%d lines, n_bins=%d filter_taps=%d; +8k=%d +16k/256=%d +48k=%d)'
          % (OUT, sum(chunk.count('\n') for chunk in lines), cap16['values']['N_BINS'], cap16['filter_taps_size'],
             extra_caps[(8000, 256)]['values']['N_BINS'],
             extra_caps[(16000, 256)]['values']['N_BINS'],
             extra_caps[(48000, 1024)]['values']['N_BINS']))


if __name__ == '__main__':
    main()
