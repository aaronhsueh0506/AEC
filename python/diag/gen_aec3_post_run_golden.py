"""Generate a binary golden for the C aec3_post_run FULL-ORCHESTRATION port.

Unlike gen_aec3_post_golden.py (the DRIVER golden, which injects the sub-module
outputs gain / usable_linear / r2 as inputs), this golden exercises the WHOLE
``AEC._aec3_post`` end-to-end: aec3_post_run drives AecState +
ResidualEchoEstimator + SuppressionGain + StationarityEstimator +
LinearFilterSelect ITSELF and must reproduce ``out[hop]`` bit-exactly.

Approach: hook the REAL balanced pipeline on the doubletalk case, monkeypatch
``AEC._aec3_post`` to capture, PER HOP, every input the method reads from
``self`` (the linear-filter spectra, W, X_buf, partition_idx, the shadow
error_spec / _last_shadow_output_time / _last_s_max_abs, _current_delay /
_delay_active, _saturation_level, the pending EPV flags PRE-call, the
filter_taps via get_time_domain_filter, the stationarity read-state) + the args
(raw_output / near / far) + the returned ``out[hop]``. We capture the WHOLE case
(~4186 hops). The construction config of every sub-module is captured live off
the orchestrator's instances (same extraction the per-module goldens use), so
the C parity test rebuilds them faithfully.

Run: python3 python/diag/gen_aec3_post_run_golden.py /tmp/aec3_post_run_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import AecConfig                          # noqa: E402
from modules.orchestrator import AEC                          # noqa: E402
from eval_aec_challenge import estimate_delay                  # noqa: E402
from modules.freq_utils import hz_to_bin                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASE = os.path.join(WAV, 'doubletalk', '0I0XMl3M0ECO0U1N0cJvpg_doubletalk')

HOP = 160
SR = 16000
FL = 832
MAX_HOPS = 4200   # whole DT case (~4186 hops)


def _w_i32(f, *vals):
    np.array(vals, dtype=np.int32).tofile(f)


def _w_f64(f, *vals):
    np.array(vals, dtype=np.float64).tofile(f)


def _w_f32(f, a):
    np.asarray(a, dtype=np.float32).ravel().tofile(f)


def _w_c64(f, a):
    """complex64 array → interleaved (re, im) float32."""
    a = np.asarray(a, dtype=np.complex64).ravel()
    out = np.empty(a.size * 2, dtype=np.float32)
    out[0::2] = a.real
    out[1::2] = a.imag
    out.tofile(f)


def capture_sg_config(sg):
    """Live-instance SuppressionGain config (mirrors gen_suppression_gain_golden)."""
    N, sr = sg._n_bins, sg._sr
    ea = sg._echo_audibility
    dn = sg._dominant_nearend
    dc = dn._cfg
    hf = sg._config.high_frequency_suppression
    return dict(
        split_floor_far_active=float(sg._split_floor_far_active),
        split_floor_far_silent=float(sg._split_floor_far_silent),
        split_floor_latch_power=float(sg._split_floor_latch_power),
        split_floor_dt=float(sg._split_floor_dt),
        low_render_threshold=float(sg._low_render._threshold),
        max_inc=float(sg._max_inc_normal),
        max_dec_lf=float(sg._max_dec_lf_normal),
        soft_blend_enabled=int(sg._soft_ne_blend_enabled),
        soft_blend_per_bin=int(sg._soft_ne_blend_per_bin),
        soft_blend_enr_thr=float(sg._soft_ne_blend_enr_thr),
        soft_blend_softness=float(sg._soft_ne_blend_softness),
        last_lf_band=int(sg._last_lf_band),
        first_hf_band=int(sg._first_hf_band),
        last_lf_smoothing_band=int(sg._last_lf_smoothing_band),
        last_permanent=int(sg._config.last_permanent_lf_smoothing_band),
        lf_smoothing_initial=int(sg._config.lf_smoothing_during_initial_phase),
        dne_lf_end=int(sg._dne_lf_end),
        trigger_threshold_hops=int(dn._trigger_threshold_hops),
        hold_duration_hops=int(dn._hold_duration_hops),
        nearend_smoother_n=int(sg._nearend_smoother._n),
        aud_lf_end_bin=int(min(hz_to_bin(ea.lf_band_end_hz, N, sr), N)),
        aud_mf_end_bin=int(min(hz_to_bin(ea.mf_band_end_hz, N, sr), N)),
        floor_power=float(ea.floor_power),
        aud_thr_lf=float(ea.audibility_threshold_lf),
        aud_thr_mf=float(ea.audibility_threshold_mf),
        aud_thr_hf=float(ea.audibility_threshold_hf),
        low_render_limit=float(ea.low_render_limit),
        normal_render_limit=float(ea.normal_render_limit),
        hf_lgb=int(hz_to_bin(hf.limiting_gain_freq_hz, N, sr)),
        hf_biq=int(max(1, hz_to_bin(hf.limiting_gain_width_hz, N, sr))),
        lf_clamp_bin=int(hz_to_bin(250.0, N, sr)),
        dne_enr_threshold=float(dc.enr_threshold),
        dne_enr_exit_threshold=float(dc.enr_exit_threshold),
        dne_snr_threshold=float(dc.snr_threshold),
        dne_use_during_initial_phase=int(dc.use_during_initial_phase),
        dne_use_unbounded_echo=int(dc.use_unbounded_echo_spectrum),
        dne_lf_endpoint_bin=int(min(hz_to_bin(dc.lf_endpoint_hz, N, sr), N)),
        conservative_hf=int(sg._config.conservative_hf_suppression),
        nearend_enr_tr=sg._nearend_enr_tr.astype(np.float32).copy(),
        nearend_enr_su=sg._nearend_enr_su.astype(np.float32).copy(),
        nearend_emr_tr=sg._nearend_emr_tr.astype(np.float32).copy(),
        normal_enr_tr=sg._normal_enr_tr.astype(np.float32).copy(),
        normal_enr_su=sg._normal_enr_su.astype(np.float32).copy(),
        normal_emr_tr=sg._normal_emr_tr.astype(np.float32).copy(),
    )


def _hops_or_auto(v):
    """AecStateConfig.erle_startup_hops/erl_startup_hops are Optional[int]
    ("None" == auto-resolve for the live grid). The int32 golden wire format
    has no None, so mirror the C-side AEC_STATE_STARTUP_HOPS_AUTO sentinel
    (-1, aec_state.c) here instead."""
    return -1 if v is None else int(v)


def capture_state_config(state):
    sc = state._config
    fa = state._filter_analyzer
    return dict(
        n_bins=int(sc.n_bins),
        num_capture_channels=int(sc.num_capture_channels),
        hop_size=int(sc.hop_size),
        enable_filter_analyzer=int(fa is not None),
        erle_startup_hops=_hops_or_auto(sc.erle_startup_hops),
        erl_startup_hops=_hops_or_auto(sc.erl_startup_hops),
        echo_can_saturate=int(sc.echo_can_saturate),
        use_linear_filter=int(sc.use_linear_filter),
        conservative_initial_phase=int(sc.conservative_initial_phase),
        initial_state_seconds=float(sc.initial_state_seconds),
        delay_headroom_samples=int(sc.delay_headroom_samples),
        erle_min=float(sc.erle_min),
        erle_max_l=float(sc.erle_max_l),
        erle_max_h=float(sc.erle_max_h),
    )


def capture_ree_config(ree):
    frr = ree._reverb_freq_resp
    return dict(
        hop_size=int(ree._hop_size),
        min_noise_floor_power=float(ree._echo_model.min_noise_floor_power),
        noise_gate_power_legacy=float(ree._echo_model.noise_gate_power),
        noise_gate_slope=float(ree._echo_model.noise_gate_slope),
        stationary_gate_slope=float(ree._echo_model.stationary_gate_slope),
        model_reverb_in_nl=int(ree._echo_model.model_reverb_in_nonlinear_mode),
        default_gain=float(ree._default_gain_early),
        tm_gain=float(ree._tm_gain_early),
        erle_onset_comp=int(ree._erle_onset_compensation_in_dominant),
        reverb_decay=float(ree._reverb_cfg.decay),
        reverb_mild_scale=float(ree._reverb_cfg.mild_decay_scale),
        reverb_enabled=int(ree._reverb_cfg.enabled),
        reverb_tail_strength=float(ree._reverb_tail_strength),
        use_aec3_residual_noise_gate=int(ree._use_aec3_residual_noise_gate),
        use_stationarity_properties=int(ree._use_stationarity_properties),
        use_aec3_echo_gen_window=int(ree._use_aec3_echo_gen_window),
        nl_r2_enabled=int(ree._nl_r2_enabled),
        nl_r2_alpha=float(ree._nl_r2_alpha),
        nl_norm_power=float(ree._nl_norm_power),
        residual_noise_gate_power=float(ree._noise_gate_power),
        noise_floor_hold_hops=int(ree._noise_floor_hold_hops),
        use_freq_response=int(frr is not None),
        reverb_use_conservative=int(frr is not None and frr._use_conservative),
        reverb_smoothing_base=(float(frr._smoothing_base) if frr is not None
                               else 0.2),
    )


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/aec3_post_run_golden.bin'

    cfg = AecConfig.from_preset('balanced', sample_rate=SR, filter_length=FL)
    np.random.seed(0)
    aec = AEC(cfg)
    flt = aec.filter
    K = flt.n_freqs
    N = flt.n_partitions

    mic, _ = sf.read(CASE + '_mic.wav', dtype='float32')
    lpb, _ = sf.read(CASE + '_lpb.wav', dtype='float32')
    n = min(len(mic), len(lpb))
    delay = estimate_delay(mic, lpb, SR)
    ref = np.zeros(n, dtype=np.float32)
    if 0 < delay < n:
        ref[delay:] = lpb[:n - delay]
    else:
        ref = lpb[:n].copy()
    mic = mic[:n]

    consts = {}
    rows = []

    from modules.residual.residual_echo_estimator import ResidualEchoEstimator
    orig_ree_reset = ResidualEchoEstimator.reset

    # The orchestrator's coarse-rescue path (process loop, BEFORE _aec3_post)
    # may call ree.reset(). aec3_post_run does not model that (it is upstream);
    # the C test replays it on the captured rising-edge flag so the REE reverb
    # state stays in lockstep.
    reset_pending = {'v': 0}

    def patched_reset(self):
        reset_pending['v'] = 1
        return orig_ree_reset(self)

    orig_post = AEC._aec3_post

    def patched_post(self, raw_output, near_end, far_end):
        f = self.filter
        sh = self.shadow_filter
        # snapshot construction config once
        if not consts:
            consts['n_bins'] = int(f.n_freqs)
            consts['fft_size'] = int(f.fft_size)
            consts['block_size'] = int(f.block_size)
            consts['hop_size'] = int(f.hop_size)
            consts['n_partitions'] = int(f.n_partitions)
            # CNG + driver constants
            consts['cng_y2_alpha'] = float(self._aec3_cng_y2_alpha)
            consts['cng_n2_track_freshness'] = float(
                self._aec3_cng_n2_track_freshness)
            consts['cng_n2_track_retention'] = float(
                self._aec3_cng_n2_track_retention)
            consts['cng_n2_slow_up'] = float(self._aec3_cng_n2_slow_up)
            consts['cng_n2_initial_alpha'] = float(self._aec3_cng_n2_initial_alpha)
            consts['cng_n2_update_onset_hops'] = int(
                self._aec3_cng_n2_update_onset_hops)
            consts['cng_n2_initial_duration_hops'] = int(
                self._aec3_cng_n2_initial_duration_hops)
            consts['noise_floor_int16sq'] = float(self._aec3_noise_floor_int16sq)
            consts['erle_coh_gate_alpha'] = float(self.config.erle_coh_gate_alpha)
            consts['erle_coh_gate_threshold'] = float(
                self.config.erle_coh_gate_threshold)
            consts['erle_coh_gate_enabled'] = int(
                bool(self.config.erle_coh_gate_enabled))
            consts['erle_windowed_capture_psd'] = int(
                bool(getattr(self.config, 'erle_windowed_capture_psd', False)))
            consts['erle_render_x2_psd_scale'] = int(
                bool(getattr(self.config, 'erle_render_x2_psd_scale', False)))
            consts['output_capture_when_linear_unusable'] = int(
                bool(getattr(self.config,
                             'output_capture_when_linear_unusable', False)))
            consts['enable_cng'] = int(bool(self.config.enable_cng))
            consts['use_stationarity_properties'] = int(bool(
                self._aec3_sg_config.echo_audibility.use_stationarity_properties))
            consts['stationarity_converge_hops'] = int(
                self._aec3_stationarity_converge_hops)
            consts['active_render_threshold'] = 5.96e-4
            consts['synth_window'] = np.asarray(
                self._aec3_synth_window, dtype=np.float32).copy()
            consts['sqrt2_sin_lut'] = np.asarray(
                self._aec3_sqrt2_sin_lut, dtype=np.float32).copy()
            consts['state'] = capture_state_config(self._aec3_state)
            consts['ree'] = capture_ree_config(self._aec3_ree)
            consts['sg'] = capture_sg_config(self._aec3_sg)
            # FilterAnalyzer HPF taps length (from get_time_domain_filter len)
            tdf = (f.get_time_domain_filter()
                   if hasattr(f, 'get_time_domain_filter') else None)
            consts['filter_taps_size'] = int(len(tdf)) if tdf is not None else 0

        # ── per-hop inputs the method reads ──────────────────────────────
        W = np.asarray(f.W, np.complex64)                       # n_part × K
        X_buf = np.asarray(f.X_buf, np.complex64)               # n_part × K
        tdf = (f.get_time_domain_filter()
               if hasattr(f, 'get_time_domain_filter') else None)
        sh_err = (np.asarray(sh.error_spec, np.complex64).copy()
                  if (sh is not None and hasattr(sh, 'error_spec'))
                  else np.zeros(K, np.complex64))
        lso = getattr(self, '_last_shadow_output_time', None)
        s_ref = float(getattr(f, '_last_s_max_abs', 0.0))
        s_coa = (float(getattr(sh, '_last_s_max_abs', 0.0))
                 if sh is not None else 0.0)
        # filter_state_bridge error energies (dead for out[hop] but captured
        # for fidelity).
        main_e = float(np.sum(np.abs(f.error_spec) ** 2)) if hasattr(
            f, 'error_spec') else 0.0
        shadow_e = (float(np.sum(np.abs(sh.error_spec) ** 2))
                    if (sh is not None and hasattr(sh, 'error_spec')) else 0.0)

        ree_reset_before = reset_pending['v']
        reset_pending['v'] = 0
        rec = dict(
            ree_reset_before=ree_reset_before,
            near_spec=np.asarray(f.near_spec, np.complex64).copy(),
            far_spec=np.asarray(f.far_spec, np.complex64).copy(),
            echo_spec=np.asarray(f.echo_spec, np.complex64).copy(),
            esw=np.asarray(f.error_spec_windowed, np.complex64).copy(),
            W0=np.asarray(W[0], np.complex64).copy(),
            W_all=W.copy(),
            X_buf=X_buf.copy(),
            sqrt_hann=np.asarray(f._sqrt_hann_analysis, np.float32).copy(),
            partition_idx=int(f.partition_idx),
            tdf_present=int(tdf is not None),
            tdf=(np.asarray(tdf, np.float32).copy() if tdf is not None
                 else np.zeros(0, np.float32)),
            shadow_present=int(sh is not None),
            shadow_error_spec=sh_err,
            lso_present=int(lso is not None),
            lso=(np.asarray(lso, np.float32).copy() if lso is not None
                 else np.zeros(self.filter.hop_size, np.float32)),
            s_ref=s_ref, s_coa=s_coa,
            main_e=main_e, shadow_e=shadow_e,
            raw_output=np.asarray(raw_output, np.float32).copy(),
            near_end=np.asarray(near_end, np.float32).copy(),
            far_end=np.asarray(far_end, np.float32).copy(),
            current_delay=int(self._current_delay),
            delay_active=int(bool(self._delay_active)),
            saturation_level=float(self._saturation_level),
            pending_gain=int(bool(self._aec3_pending_gain_change)),
            pending_delay=(-1 if self._aec3_pending_delay_change is None
                           else int(self._aec3_pending_delay_change.value)),
            stat_active_hops=int(self._aec3_stationarity_active_hops),
            # Whether the upstream process loop fired the StationarityEstimator
            # update this hop (non_zero_render_seen latched). The C replays the
            # update on the SAME flag so is_block_stationary / band_mask agree.
            stat_update_fired=int(bool(self._aec3_non_zero_render_seen)),
        )
        ret = orig_post(self, raw_output, near_end, far_end)
        # _dt_protect_active is set INSIDE _aec3_post (orchestrator 3444) just
        # before get_gain, from dt_aware_res_floor_enabled & _ne_recent_frames.
        # aec3_post_run does NOT compute it (the production caller aec.c sets it
        # on the SG before the call); capture the value this hop actually used
        # so the C test can replay it onto sg.dt_protect_active.
        rec['dt_protect_active'] = int(bool(self._aec3_sg._dt_protect_active))
        rec['out'] = np.asarray(ret, np.float32).copy()
        # usable_linear after update (localisation assertion)
        rec['usable'] = int(bool(self._aec3_state.usable_linear_estimate()))
        rows.append(rec)
        return ret

    AEC._aec3_post = patched_post
    ResidualEchoEstimator.reset = patched_reset
    try:
        cnt = 0
        for i in range(0, n - HOP, HOP):
            aec.process(mic[i:i + HOP], ref[i:i + HOP])
            cnt += 1
            if cnt >= MAX_HOPS:
                break
    finally:
        AEC._aec3_post = orig_post
        ResidualEchoEstimator.reset = orig_ree_reset

    n_hops = len(rows)
    print(f"captured {n_hops} hops; "
          f"shadow_present={rows[0]['shadow_present']} "
          f"lso_present(any)={int(any(r['lso_present'] for r in rows))} "
          f"usable_true={sum(r['usable'] for r in rows)} "
          f"ree_resets={sum(r['ree_reset_before'] for r in rows)}")

    sgc = consts['sg']
    stc = consts['state']
    rc = consts['ree']

    with open(out, 'wb') as f:
        # ── header geometry + flags ──
        _w_i32(f, consts['n_bins'], consts['fft_size'], consts['block_size'],
               consts['hop_size'], consts['n_partitions'])
        _w_i32(f, consts['erle_coh_gate_enabled'],
               consts['erle_windowed_capture_psd'],
               consts['erle_render_x2_psd_scale'],
               consts['output_capture_when_linear_unusable'],
               consts['enable_cng'],
               consts['use_stationarity_properties'])
        _w_i32(f, consts['cng_n2_update_onset_hops'],
               consts['cng_n2_initial_duration_hops'],
               consts['stationarity_converge_hops'],
               consts['filter_taps_size'])
        _w_f64(f, consts['cng_y2_alpha'], consts['cng_n2_track_freshness'],
               consts['cng_n2_track_retention'], consts['cng_n2_slow_up'],
               consts['cng_n2_initial_alpha'], consts['noise_floor_int16sq'],
               consts['erle_coh_gate_alpha'], consts['erle_coh_gate_threshold'],
               consts['active_render_threshold'])
        _w_f32(f, consts['synth_window'])     # block_size
        _w_f32(f, consts['sqrt2_sin_lut'])    # 32

        # ── AecState config ──
        _w_i32(f, stc['n_bins'], stc['num_capture_channels'], stc['hop_size'],
               stc['enable_filter_analyzer'], stc['erle_startup_hops'],
               stc['erl_startup_hops'], stc['echo_can_saturate'],
               stc['use_linear_filter'], stc['conservative_initial_phase'],
               stc['delay_headroom_samples'])
        _w_f64(f, stc['initial_state_seconds'], stc['erle_min'],
               stc['erle_max_l'], stc['erle_max_h'])

        # ── REE config ──
        _w_i32(f, rc['hop_size'], rc['model_reverb_in_nl'], rc['erle_onset_comp'],
               rc['reverb_enabled'], rc['use_aec3_residual_noise_gate'],
               rc['use_aec3_echo_gen_window'], rc['nl_r2_enabled'],
               rc['noise_floor_hold_hops'], rc['use_freq_response'],
               rc['reverb_use_conservative'], rc['use_stationarity_properties'])
        _w_f64(f, rc['min_noise_floor_power'], rc['noise_gate_power_legacy'],
               rc['noise_gate_slope'], rc['stationary_gate_slope'],
               rc['default_gain'], rc['tm_gain'], rc['reverb_decay'],
               rc['reverb_mild_scale'], rc['reverb_tail_strength'],
               rc['nl_r2_alpha'], rc['nl_norm_power'],
               rc['residual_noise_gate_power'], rc['reverb_smoothing_base'])

        # ── SuppressionGain config ──
        _w_f64(f, sgc['split_floor_far_active'], sgc['split_floor_far_silent'],
               sgc['split_floor_latch_power'], sgc['low_render_threshold'],
               sgc['split_floor_dt'])
        _w_f32(f, np.array([sgc['max_inc'], sgc['max_dec_lf']], np.float32))
        _w_i32(f, sgc['soft_blend_enabled'], sgc['soft_blend_per_bin'])
        _w_f32(f, np.array([sgc['soft_blend_enr_thr'], sgc['soft_blend_softness']],
                           np.float32))
        _w_i32(f, sgc['last_lf_band'], sgc['first_hf_band'],
               sgc['last_lf_smoothing_band'], sgc['last_permanent'],
               sgc['lf_smoothing_initial'], sgc['dne_lf_end'],
               sgc['trigger_threshold_hops'], sgc['hold_duration_hops'],
               sgc['nearend_smoother_n'], sgc['aud_lf_end_bin'],
               sgc['aud_mf_end_bin'])
        _w_f64(f, sgc['floor_power'], sgc['aud_thr_lf'], sgc['aud_thr_mf'],
               sgc['aud_thr_hf'], sgc['low_render_limit'],
               sgc['normal_render_limit'])
        _w_i32(f, sgc['hf_lgb'], sgc['hf_biq'], sgc['conservative_hf'],
               sgc['lf_clamp_bin'])
        _w_f64(f, sgc['dne_enr_threshold'], sgc['dne_enr_exit_threshold'],
               sgc['dne_snr_threshold'])
        _w_i32(f, sgc['dne_use_during_initial_phase'],
               sgc['dne_use_unbounded_echo'], sgc['dne_lf_endpoint_bin'])
        _w_f32(f, sgc['nearend_enr_tr'])
        _w_f32(f, sgc['nearend_enr_su'])
        _w_f32(f, sgc['nearend_emr_tr'])
        _w_f32(f, sgc['normal_enr_tr'])
        _w_f32(f, sgc['normal_enr_su'])
        _w_f32(f, sgc['normal_emr_tr'])

        # ── per-hop rows ──
        _w_i32(f, n_hops)
        for r in rows:
            _w_i32(f, r['ree_reset_before'])
            _w_i32(f, r['dt_protect_active'])
            _w_c64(f, r['near_spec'])
            _w_c64(f, r['far_spec'])
            _w_c64(f, r['echo_spec'])
            _w_c64(f, r['esw'])
            _w_c64(f, r['W0'])
            _w_c64(f, r['W_all'])         # n_part × K
            _w_c64(f, r['X_buf'])         # n_part × K
            _w_f32(f, r['sqrt_hann'])     # block_size
            _w_i32(f, r['partition_idx'], r['tdf_present'], len(r['tdf']))
            if r['tdf_present']:
                _w_f32(f, r['tdf'])
            _w_i32(f, r['shadow_present'])
            _w_c64(f, r['shadow_error_spec'])
            _w_i32(f, r['lso_present'])
            _w_f32(f, r['lso'])           # hop
            _w_f64(f, r['s_ref'], r['s_coa'], r['main_e'], r['shadow_e'])
            _w_f32(f, r['raw_output'])    # hop
            _w_f32(f, r['near_end'])      # hop
            _w_f32(f, r['far_end'])       # hop
            _w_i32(f, r['current_delay'], r['delay_active'])
            _w_f64(f, r['saturation_level'])
            _w_i32(f, r['pending_gain'], r['pending_delay'],
                   r['stat_active_hops'])
            # expected
            _w_i32(f, r['usable'])
            _w_f32(f, r['out'])           # hop
            _w_i32(f, r['stat_update_fired'])

    print(f"wrote {out}  ({n_hops} hops, {K} bins, {N} partitions, "
          f"filter_taps_size={consts['filter_taps_size']})")


if __name__ == '__main__':
    main()
