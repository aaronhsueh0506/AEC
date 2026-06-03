"""Generate a binary golden for the C suppression_gain port (WS5).

Hooks the REAL AEC pipeline (orchestrator `_aec3_post`) by monkeypatching
`SuppressionGain.get_gain` and captures, per hop, EVERY input (with real
dtypes / None-ness) plus the output gain[257]. We run THREE real cases —
one each from doubletalk / farend_singletalk / nearend_singletalk — so the
C parity test exercises the echo-dominant FS, near-dominant NE, and mixed
DT gain/R² branches (a branch left uncovered by a single case would be
silently un-tested).

Captured real dtypes (balanced, 16 kHz, hop=160, filter_length=832):
  nearend / R² / R²_unbounded / comfort_noise : float32 (257,)
  render_block                                : float32 (160,)  (int16-scaled)
  saturated_echo / clock_drift                : bool
  stationary_mask / coh_gamma2 / coh_xy_gamma2 / nearend_p_ne : None
  output gain                                 : float32 (257,)

The golden does NOT replay through a real AecState in C — the only thing
`_lower_band_gain` reads off `aec_state` is `saturated_echo()` (a bool), so
we capture that bool per frame and feed it directly.

Config for all three cases is the balanced active path:
  split_floor_enabled=True (far_active -28 dB / far_silent -12 dB, latch 1e6)
  soft_nearend_blend_enabled=True, per_bin=True, enr_thr=0.25, softness=0.25
  use_wallclock_block_energy_threshold=True, use_wallclock_gain_ratchet=True
  (all other levers OFF)

Layout (LE):
  int32   n_cases
  float64 split_floor_far_active_pow, split_floor_far_silent_pow
  float64 split_floor_latch_power
  float64 low_render_threshold              (block_energy_scale 50*50*64)
  float32 max_inc, max_dec_lf               (wallclock ratchet, ne==normal here)
  int32   soft_blend_enabled, soft_blend_per_bin
  float32 soft_blend_enr_thr, soft_blend_softness
  int32   last_lf_band, first_hf_band, last_lf_smoothing_band
  int32   last_permanent_lf_smoothing_band
  int32   lf_smoothing_during_initial_phase
  int32   dne_lf_end
  int32   trigger_threshold_hops, hold_duration_hops
  int32   nearend_smoother_hops
  int32   lf_band_end_bin, mf_band_end_bin       (audibility)
  float64 floor_power, audibility_thr_lf/mf/hf
  float64 low_render_limit, normal_render_limit
  int32   hf_lgb_bin, hf_biq                      (HF limiter)
  -- per-bin tuning tables (n_bins each, float32) --
  nearend_enr_tr/su, nearend_emr_tr, normal_enr_tr/su, normal_emr_tr
  per case:
    int32 n_bins, int32 n_frames
    n_frames x [
      nearend[n_bins] f32 | R2[n_bins] f32 | R2_unb[n_bins] f32
      | CN[n_bins] f32 | render_block[160] f32
      | saturated_echo u8 | clock_drift u8
      | gain[n_bins] f32   (expected output)
    ]

Run: python3 python/diag/gen_suppression_gain_golden.py /tmp/sg_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import modules.residual.suppression_gain as SG  # noqa: E402
from modules.config import AecConfig  # noqa: E402

N_BINS = 257
HOP = 160
FILTER_LENGTH = 832
MAX_FRAMES = 240   # cap per case (enough to latch split-floor + cross DNE state)

CASES = [
    ('doubletalk',
     'wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_mic.wav',
     'wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_lpb.wav'),
    ('farend_singletalk',
     'wav/aec_challenge_blind/farend_singletalk/0KjzXA3g20qsd8zmSekADw_farend_singletalk_mic.wav',
     'wav/aec_challenge_blind/farend_singletalk/0KjzXA3g20qsd8zmSekADw_farend_singletalk_lpb.wav'),
    ('nearend_singletalk',
     'wav/aec_challenge_blind/nearend_singletalk/014AzuqPZku2004NbTTmcA_nearend_singletalk_mic.wav',
     'wav/aec_challenge_blind/nearend_singletalk/014AzuqPZku2004NbTTmcA_nearend_singletalk_lpb.wav'),
]


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def capture_config(self):
    """Read the FULL derived config off the live SuppressionGain instance.

    The orchestrator overrides several fields on the SuppressorConfig before
    constructing the SuppressionGain (fft-density-scaled audibility floors,
    use_wallclock_trigger_threshold=True, etc.), so reconstructing the config
    from defaults is WRONG. Read the live instance instead — this is the same
    object the production pipeline uses.
    """
    from modules.freq_utils import hz_to_bin
    N, sr = self._n_bins, self._sr
    ea = self._echo_audibility
    dn = self._dominant_nearend
    dc = dn._cfg
    hf = self._config.high_frequency_suppression
    return dict(
        n_bins=N,
        split_floor_far_active=float(self._split_floor_far_active),
        split_floor_far_silent=float(self._split_floor_far_silent),
        split_floor_latch_power=float(self._split_floor_latch_power),
        low_render_threshold=float(self._low_render._threshold),
        max_inc=float(self._max_inc_normal),         # ne == normal under ratchet
        max_dec_lf=float(self._max_dec_lf_normal),
        max_inc_nearend=float(self._max_inc_nearend),
        max_dec_lf_nearend=float(self._max_dec_lf_nearend),
        soft_blend_enabled=int(self._soft_ne_blend_enabled),
        soft_blend_per_bin=int(self._soft_ne_blend_per_bin),
        soft_blend_enr_thr=float(self._soft_ne_blend_enr_thr),
        soft_blend_softness=float(self._soft_ne_blend_softness),
        last_lf_band=int(self._last_lf_band),
        first_hf_band=int(self._first_hf_band),
        last_lf_smoothing_band=int(self._last_lf_smoothing_band),
        last_permanent=int(self._config.last_permanent_lf_smoothing_band),
        lf_smoothing_initial=int(self._config.lf_smoothing_during_initial_phase),
        dne_lf_end=int(self._dne_lf_end),
        trigger_threshold_hops=int(dn._trigger_threshold_hops),
        hold_duration_hops=int(dn._hold_duration_hops),
        nearend_smoother_n=int(self._nearend_smoother._n),
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
        dne_enr_threshold=float(dc.enr_threshold),
        dne_enr_exit_threshold=float(dc.enr_exit_threshold),
        dne_snr_threshold=float(dc.snr_threshold),
        dne_use_during_initial_phase=int(dc.use_during_initial_phase),
        dne_use_unbounded_echo=int(dc.use_unbounded_echo_spectrum),
        dne_lf_endpoint_bin=int(min(hz_to_bin(dc.lf_endpoint_hz, N, sr), N)),
        dne_loud_relax_enabled=int(dc.loud_nearend_enr_relax_enabled),
        dne_loud_snr_factor=float(dc.loud_nearend_snr_factor),
        dne_loud_enr_threshold=float(dc.loud_nearend_enr_threshold),
        conservative_hf=int(self._config.conservative_hf_suppression),
        stat_aware_proxy=int(self._config.stat_aware_ne_proxy_enabled),
        nearend_enr_tr=self._nearend_enr_tr.astype(np.float32).copy(),
        nearend_enr_su=self._nearend_enr_su.astype(np.float32).copy(),
        nearend_emr_tr=self._nearend_emr_tr.astype(np.float32).copy(),
        normal_enr_tr=self._normal_enr_tr.astype(np.float32).copy(),
        normal_enr_su=self._normal_enr_su.astype(np.float32).copy(),
        normal_emr_tr=self._normal_emr_tr.astype(np.float32).copy(),
    )


def run_case(repo, mic_path, ref_path, frames, cfg_out):
    """Drive the real pipeline; record per-frame inputs/output into `frames`.

    cfg_out: dict updated in-place with the live-instance config (first frame).
    """
    import soundfile as sf
    from aec import AEC

    captured = []
    prev_id = [None]

    orig = SG.SuppressionGain.get_gain

    def patched(self, *, aec_state, nearend_spectrum, residual_echo_spectrum,
                residual_echo_spectrum_unbounded, comfort_noise_spectrum,
                render_block, clock_drift, stationary_mask=None,
                coh_gamma2=None, coh_xy_gamma2=None, nearend_p_ne=None):
        if not cfg_out:
            cfg_out.update(capture_config(self))
        # The orchestrator RECREATES the SuppressionGain instance on a
        # mid-stream reset (delay-acquire / filter-recovery -> _reset_aec3_post),
        # which clears the smoother / split-floor latch / DNE counters /
        # last_gain. Detect that by instance identity so the C replay can
        # re-init its state on the same frame. We also capture initial_state
        # (used by the LF-smoothing gate + DNE trigger gate) at call time
        # (BEFORE the call mutates nothing — get_gain reads but doesn't change
        # initial_state; it is set externally by the orchestrator).
        is_reset = (prev_id[0] is not None and id(self) != prev_id[0])
        prev_id[0] = id(self)
        cur_initial_state = bool(self._initial_state)
        gain = orig(
            self, aec_state=aec_state, nearend_spectrum=nearend_spectrum,
            residual_echo_spectrum=residual_echo_spectrum,
            residual_echo_spectrum_unbounded=residual_echo_spectrum_unbounded,
            comfort_noise_spectrum=comfort_noise_spectrum,
            render_block=render_block, clock_drift=clock_drift,
            stationary_mask=stationary_mask, coh_gamma2=coh_gamma2,
            coh_xy_gamma2=coh_xy_gamma2, nearend_p_ne=nearend_p_ne,
        )
        # Sanity: confirm balanced None-ness so the C port can hardwire it.
        # stationary_mask may be bool(257,) on some frames, but it ONLY feeds
        # `_stat_mask_frac`, which is read by `_ne_state_for_gain_rules` ONLY
        # when stat_aware_ne_proxy_enabled (OFF in balanced) -> zero effect on
        # the gain output. coh_gamma2 / coh_xy_gamma2 / nearend_p_ne gate the
        # default-OFF coh / cohxd levers and are always None in balanced.
        assert coh_gamma2 is None and coh_xy_gamma2 is None
        assert nearend_p_ne is None
        if len(captured) < MAX_FRAMES:
            captured.append((
                _f32(nearend_spectrum).copy(),
                _f32(residual_echo_spectrum).copy(),
                _f32(residual_echo_spectrum_unbounded).copy(),
                _f32(comfort_noise_spectrum).copy(),
                _f32(render_block).copy(),
                bool(aec_state.saturated_echo()),
                bool(clock_drift),
                bool(is_reset),
                cur_initial_state,
                _f32(gain).copy(),
            ))
        return gain

    SG.SuppressionGain.get_gain = patched
    try:
        cfg = AecConfig.from_preset('balanced', sample_rate=16000,
                                    filter_length=FILTER_LENGTH)
        cfg.enable_res = True
        cfg.enable_cng = True
        mic, _ = sf.read(os.path.join(repo, mic_path))
        ref, _ = sf.read(os.path.join(repo, ref_path))
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        n = min(len(mic), len(ref))
        np.random.seed(0)
        aec = AEC(cfg)
        nh = n // HOP
        for i in range(nh):
            if len(captured) >= MAX_FRAMES:
                break
            aec.process(mic[i * HOP:(i + 1) * HOP], ref[i * HOP:(i + 1) * HOP])
    finally:
        SG.SuppressionGain.get_gain = orig
    frames.extend(captured)


def write_header(f, C):
    """Write the shared static config block + per-bin tuning tables.

    C is the dict captured off the live SuppressionGain instance
    (capture_config). All three cases share the same balanced config, so we
    use the first case's capture.
    """
    np.array([len(CASES)], dtype=np.int32).tofile(f)
    np.array([C['split_floor_far_active'], C['split_floor_far_silent'],
              C['split_floor_latch_power'], C['low_render_threshold']],
             dtype=np.float64).tofile(f)
    np.array([C['max_inc'], C['max_dec_lf']], dtype=np.float32).tofile(f)
    np.array([C['soft_blend_enabled'], C['soft_blend_per_bin']],
             dtype=np.int32).tofile(f)
    np.array([C['soft_blend_enr_thr'], C['soft_blend_softness']],
             dtype=np.float32).tofile(f)
    np.array([C['last_lf_band'], C['first_hf_band'], C['last_lf_smoothing_band'],
              C['last_permanent'], C['lf_smoothing_initial'], C['dne_lf_end'],
              C['trigger_threshold_hops'], C['hold_duration_hops'],
              C['nearend_smoother_n'], C['aud_lf_end_bin'], C['aud_mf_end_bin']],
             dtype=np.int32).tofile(f)
    np.array([C['floor_power'], C['aud_thr_lf'], C['aud_thr_mf'],
              C['aud_thr_hf'], C['low_render_limit'], C['normal_render_limit']],
             dtype=np.float64).tofile(f)
    np.array([C['hf_lgb'], C['hf_biq']], dtype=np.int32).tofile(f)
    np.array([C['dne_enr_threshold'], C['dne_enr_exit_threshold'],
              C['dne_snr_threshold'], 0.0], dtype=np.float64).tofile(f)
    np.array([C['dne_use_during_initial_phase'], C['dne_use_unbounded_echo'],
              C['dne_lf_endpoint_bin']], dtype=np.int32).tofile(f)

    C['nearend_enr_tr'].tofile(f)
    C['nearend_enr_su'].tofile(f)
    C['nearend_emr_tr'].tofile(f)
    C['normal_enr_tr'].tofile(f)
    C['normal_enr_su'].tofile(f)
    C['normal_emr_tr'].tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/sg_golden.bin'
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    case_frames = []
    captured_cfg = {}
    for name, mic, ref in CASES:
        frames = []
        cfg_out = {}
        run_case(repo, mic, ref, frames, cfg_out)
        if not captured_cfg:
            captured_cfg = cfg_out
        case_frames.append((name, frames))
        print(f"  {name}: {len(frames)} frames")

    with open(out, 'wb') as f:
        write_header(f, captured_cfg)
        for name, frames in case_frames:
            np.array([N_BINS, len(frames)], dtype=np.int32).tofile(f)
            for (ne, r2, r2u, cn, rb, sat, cd, rst, ist, gain) in frames:
                ne.tofile(f)
                r2.tofile(f)
                r2u.tofile(f)
                cn.tofile(f)
                rb.tofile(f)
                np.array([1 if sat else 0], dtype=np.uint8).tofile(f)
                np.array([1 if cd else 0], dtype=np.uint8).tofile(f)
                np.array([1 if rst else 0], dtype=np.uint8).tofile(f)
                np.array([1 if ist else 0], dtype=np.uint8).tofile(f)
                gain.tofile(f)
    total = sum(len(fr) for _, fr in case_frames)
    print(f"wrote {out}  ({len(CASES)} cases, {total} total frames, "
          f"{N_BINS} bins)")


if __name__ == '__main__':
    main()
