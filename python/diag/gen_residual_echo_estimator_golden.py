"""Generate a binary golden for the C residual_echo_estimator port (WS5).

Hooks the REAL balanced pipeline (orchestrator) by monkeypatching
ResidualEchoEstimator.update_reverb_models + .estimate, capturing every input
(with real dtypes / None-ness) and every output per hop, across THREE real
AEC-Challenge cases (one DT / FS / NE) so all gain/R² branches are exercised:
  - doubletalk        : mixed, dominant_nearend transitions, both paths
  - farend_singletalk : echo-dominant, linear path R² = S²/ERLE
  - nearend_singletalk: near-dominant, nonlinear X²·g² + nl_r2 path

Writes raw little-endian binary; c_impl/test/historical/parity_residual_echo_estimator.c
replays the full estimator (update_reverb_models → estimate) and asserts every
output bit-exact, INCLUDING the _last_r2_direct_component /
_last_r2_reverb_component decomposition (so a reverb-tail bug can't hide behind
a direct-path match) and the bound ReverbFrequencyResponse tail_response /
average_decay.

Per-frame record layout (all LE):
  int32 reset_before   ; 1 = ResidualEchoEstimator.reset() fired before this hop
                         (coarse-rescue rising edge; clears noise floor +
                          reverb_model + freq_resp, NOT the render deques)
  [update_reverb_models]
    int32 n_partitions
    int32 filter_delay_blocks_urm
    int32 fq_is_none ; float64 filter_quality
    int32 stationary_block
    float32 frequency_response[n_partitions * n_bins]
  [estimate inputs]
    int32 dominant_nearend
    int32 usable ; int32 saturated ; int32 transparent_mode
    int32 filter_delay_blocks ; int32 filter_length_blocks
    int32 force_nonlinear_path
    float32 render_psd[n_bins]
    float32 capture_psd[n_bins]
    float32 s2_linear[n_bins]
    float32 erle[n_bins]
    float32 erle_unbounded[n_bins]
  [expected outputs]
    float32 r2[n_bins]
    float32 r2_unbounded[n_bins]
    float32 last_r2_direct[n_bins]
    float32 last_r2_reverb[n_bins]
    float32 tail_response[n_bins]
    float64 average_decay

Header (LE):
  int32 n_bins
  int32 n_cases
  per case: int32 n_frames, int32 max_partitions, then the per-frame records.

Config constants (written ONCE in a preamble, captured from the live estimator
on the first frame): see CFG_FIELDS below.

Run: python3 python/diag/gen_residual_echo_estimator_golden.py /tmp/ree_golden.bin
"""
import os
import sys
import struct

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import soundfile as sf  # noqa: E402
from modules.config import AecConfig  # noqa: E402
from aec import AecMode  # noqa: E402
from modules.orchestrator import AEC  # noqa: E402
from modules.residual.residual_echo_estimator import ResidualEchoEstimator  # noqa: E402
from eval_aec_challenge import estimate_delay  # noqa: E402

import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASES = [
    os.path.join(WAV, 'doubletalk', '0I0XMl3M0ECO0U1N0cJvpg_doubletalk'),
    os.path.join(WAV, 'farend_singletalk', '0KjzXA3g20qsd8zmSekADw_farend_singletalk'),
    os.path.join(WAV, 'nearend_singletalk', '014AzuqPZku2004NbTTmcA_nearend_singletalk'),
]

N_BINS = 257
# Cap frames per case to keep the golden small but cover branch transitions.
MAX_FRAMES = 600


def f32(a):
    return np.asarray(a, dtype=np.float32)


def run_case(stem, frames):
    """Run one real case; append per-frame records to ``frames`` list.
    Returns (n_frames, max_partitions, cfg_snapshot or None)."""
    mic, sr = sf.read(stem + '_mic.wav', dtype='float32')
    ref, _ = sf.read(stem + '_lpb.wav', dtype='float32')
    n = min(len(mic), len(ref))
    delay = estimate_delay(mic, ref, sr)
    if 0 < delay < n:
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
    else:
        ref_aligned = ref[:n]

    orig_urm = ResidualEchoEstimator.update_reverb_models
    orig_est = ResidualEchoEstimator.estimate
    orig_reset = ResidualEchoEstimator.reset

    # pending holds the update_reverb_models record until the matching estimate.
    pending = {}
    cfg_holder = {}
    # AEC3-strict mid-stream reset (coarse-rescue rising edge, orchestrator
    # ~line 1921). Clears noise floor + counter + reverb_model + freq_resp
    # (NOT the render deques). Recorded per-frame so the C test replays it.
    reset_state = {'pending': 0}

    def patched_reset(self):
        reset_state['pending'] = 1
        return orig_reset(self)

    def patched_urm(self, *, frequency_response, filter_delay_blocks,
                    filter_quality, usable_linear_filter, stationary_block,
                    time_domain_filter=None):
        fr = np.ascontiguousarray(f32(frequency_response))
        pending['reset_before'] = reset_state['pending']
        reset_state['pending'] = 0
        pending['fr'] = fr
        pending['n_partitions'] = int(fr.shape[0])
        pending['fdb_urm'] = int(filter_delay_blocks)
        pending['fq_is_none'] = 1 if filter_quality is None else 0
        pending['fq'] = 0.0 if filter_quality is None else float(filter_quality)
        pending['stationary'] = 1 if stationary_block else 0
        if not cfg_holder:
            frr = self._reverb_freq_resp
            cfg_holder.update(dict(
                hop_size=int(self._hop_size),
                min_noise_floor_power=float(self._echo_model.min_noise_floor_power),
                noise_gate_power_legacy=float(self._echo_model.noise_gate_power),
                noise_gate_slope=float(self._echo_model.noise_gate_slope),
                stationary_gate_slope=float(self._echo_model.stationary_gate_slope),
                model_reverb_in_nl=1 if self._echo_model.model_reverb_in_nonlinear_mode else 0,
                default_gain=float(self._default_gain_early),
                tm_gain=float(self._tm_gain_early),
                erle_onset_comp=1 if self._erle_onset_compensation_in_dominant else 0,
                reverb_decay=float(self._reverb_cfg.decay),
                reverb_mild_scale=float(self._reverb_cfg.mild_decay_scale),
                reverb_enabled=1 if self._reverb_cfg.enabled else 0,
                reverb_tail_strength=float(self._reverb_tail_strength),
                use_aec3_residual_noise_gate=1 if self._use_aec3_residual_noise_gate else 0,
                use_stationarity_properties=1 if self._use_stationarity_properties else 0,
                use_aec3_echo_gen_window=1 if self._use_aec3_echo_gen_window else 0,
                nl_r2_enabled=1 if self._nl_r2_enabled else 0,
                nl_r2_alpha=float(self._nl_r2_alpha),
                nl_norm_power=float(self._nl_norm_power),
                residual_noise_gate_power=float(self._noise_gate_power),
                noise_floor_hold_hops=int(self._noise_floor_hold_hops),
                use_freq_response=1 if frr is not None else 0,
                reverb_use_conservative=(1 if (frr is not None and frr._use_conservative) else 0),
                reverb_smoothing_base=(float(frr._smoothing_base) if frr is not None else 0.2),
            ))
        return orig_urm(self, frequency_response=frequency_response,
                        filter_delay_blocks=filter_delay_blocks,
                        filter_quality=filter_quality,
                        usable_linear_filter=usable_linear_filter,
                        stationary_block=stationary_block,
                        time_domain_filter=time_domain_filter)

    def patched_est(self, *, aec_state, render_psd, capture_psd, s2_linear,
                    dominant_nearend,
                    filter_delay_blocks=0, filter_length_blocks=0,
                    force_nonlinear_path=False):
        # Capture aec_state-derived values the estimator will read.
        usable = bool(aec_state.usable_linear_estimate())
        if force_nonlinear_path:
            usable_eff = False
        else:
            usable_eff = usable
        saturated = bool(aec_state.saturated_echo())
        transparent = bool(aec_state.transparent_mode_active())
        onset = (self._erle_onset_compensation_in_dominant
                 or not dominant_nearend)
        erle = f32(aec_state.erle(onset)).copy()
        erle_unb = f32(aec_state.erle_unbounded()).copy()

        rp = f32(render_psd).copy()
        cp = f32(capture_psd).copy()
        s2 = f32(s2_linear).copy()

        r2, r2_unb = orig_est(
            self, aec_state=aec_state, render_psd=render_psd,
            capture_psd=capture_psd, s2_linear=s2_linear,
            dominant_nearend=dominant_nearend,
            filter_delay_blocks=filter_delay_blocks,
            filter_length_blocks=filter_length_blocks,
            force_nonlinear_path=force_nonlinear_path)

        if len(frames) >= MAX_FRAMES or not pending:
            pending.clear()
            return r2, r2_unb

        frr = self._reverb_freq_resp
        tail = (f32(frr.tail_response).copy() if frr is not None
                else np.zeros(N_BINS, dtype=np.float32))
        avg_decay = float(frr.average_decay) if frr is not None else 0.0

        rec = dict(
            reset_before=pending['reset_before'],
            n_partitions=pending['n_partitions'],
            fdb_urm=pending['fdb_urm'],
            fq_is_none=pending['fq_is_none'],
            fq=pending['fq'],
            stationary=pending['stationary'],
            fr=pending['fr'],
            dominant_nearend=1 if dominant_nearend else 0,
            usable=1 if usable_eff else 0,
            saturated=1 if saturated else 0,
            transparent=1 if transparent else 0,
            fdb=int(filter_delay_blocks),
            flb=int(filter_length_blocks),
            force_nl=1 if force_nonlinear_path else 0,
            render_psd=rp, capture_psd=cp, s2_linear=s2,
            erle=erle, erle_unb=erle_unb,
            r2=f32(r2).copy(), r2_unb=f32(r2_unb).copy(),
            last_direct=f32(self._last_r2_direct_component).copy(),
            last_reverb=f32(self._last_r2_reverb_component).copy(),
            tail=tail, avg_decay=avg_decay,
        )
        frames.append(rec)
        pending.clear()
        return r2, r2_unb

    ResidualEchoEstimator.update_reverb_models = patched_urm
    ResidualEchoEstimator.estimate = patched_est
    ResidualEchoEstimator.reset = patched_reset
    try:
        cfg = AecConfig.from_preset(
            'balanced', sample_rate=16000, filter_length=832,
            mode=AecMode.PBFDKF, enable_shadow=True,
            enable_res=True, use_kalman=True, enable_delay_est=False)
        np.random.seed(0)
        aec = AEC(cfg)
        hop = aec.hop_size
        pos = 0
        while pos + hop <= n and len(frames) < MAX_FRAMES:
            aec.process(mic[pos:pos + hop], ref_aligned[pos:pos + hop])
            pos += hop
    finally:
        ResidualEchoEstimator.update_reverb_models = orig_urm
        ResidualEchoEstimator.estimate = orig_est
        ResidualEchoEstimator.reset = orig_reset

    max_part = max((r['n_partitions'] for r in frames), default=0)
    return len(frames), max_part, cfg_holder


class _FakeState:
    """Minimal AecState surface ResidualEchoEstimator.estimate consumes —
    used ONLY to drive the saturated-echo branch, which the AEC-Challenge
    corpus never reaches (its signals never clip). All 5 accessors return
    pre-set values; erle/erle_unbounded are fixed float32[257] arrays."""

    def __init__(self, usable, saturated, transparent, erle, erle_unb):
        self._usable = usable
        self._saturated = saturated
        self._transparent = transparent
        self._erle = erle
        self._erle_unb = erle_unb

    def usable_linear_estimate(self):
        return self._usable

    def saturated_echo(self):
        return self._saturated

    def transparent_mode_active(self):
        return self._transparent

    def erle(self, onset_compensated=False):
        return self._erle

    def erle_unbounded(self):
        return self._erle_unb


def build_synthetic_saturated(cfg):
    """Drive the saturated-echo branches (saturated-linear + saturated-nonlinear)
    that real cases never reach. Builds a fresh ResidualEchoEstimator with the
    SAME config the orchestrator uses, then feeds frames with saturated=True in
    both usable_linear=True and =False states. Returns (frames, max_partitions).
    """
    from modules.residual.residual_echo_estimator import (
        ResidualEchoEstimator, EchoModelConfig)
    em = EchoModelConfig(min_noise_floor_power=cfg['min_noise_floor_power'])
    ree = ResidualEchoEstimator(
        n_bins=N_BINS, echo_model=em, sr=16000, hop_size=cfg['hop_size'],
        use_aec3_residual_noise_gate=bool(cfg['use_aec3_residual_noise_gate']),
        use_stationarity_properties=bool(cfg['use_stationarity_properties']),
        use_aec3_echo_gen_window=bool(cfg['use_aec3_echo_gen_window']),
        use_aec3_wallclock_reverb_smoothing=True,
        nl_r2_enabled=bool(cfg['nl_r2_enabled']),
        nl_r2_alpha=cfg['nl_r2_alpha'])
    ree._reverb_tail_strength = cfg['reverb_tail_strength']

    rng = np.random.RandomState(99)
    n_part = 6
    frames = []
    # Interleave: saturated-linear, saturated-nonlinear, normal-linear,
    # normal-nonlinear so the reverb deque warms and both saturated paths
    # (which still run the trailing reverb block in nonlinear mode) are hit.
    plan = [
        (True, True), (True, True), (False, False), (True, True),
        (False, False), (True, False), (False, True), (True, False),
        (False, False), (True, True), (True, False), (False, False),
    ]
    for i, (usable, saturated) in enumerate(plan):
        fr = (rng.rand(n_part, N_BINS).astype(np.float32) * 1.0e3).astype(np.float32)
        fdb = 0
        flb = n_part
        fq = float(0.5 + 0.4 * rng.rand())
        rp = (rng.rand(N_BINS).astype(np.float32) * 5.0e8).astype(np.float32)
        cp = (rng.rand(N_BINS).astype(np.float32) * 8.0e8).astype(np.float32)
        s2 = (rng.rand(N_BINS).astype(np.float32) * 3.0e7).astype(np.float32)
        erle = (1.0 + rng.rand(N_BINS).astype(np.float32) * 3.0).astype(np.float32)
        erle_unb = (1.0 + rng.rand(N_BINS).astype(np.float32) * 3.0).astype(np.float32)
        st = _FakeState(usable, saturated, False, erle, erle_unb)

        ree.update_reverb_models(
            frequency_response=fr, filter_delay_blocks=fdb,
            filter_quality=fq, usable_linear_filter=usable,
            stationary_block=False, time_domain_filter=None)
        r2, r2_unb = ree.estimate(
            aec_state=st, render_psd=rp, capture_psd=cp, s2_linear=s2,
            dominant_nearend=False, filter_delay_blocks=fdb,
            filter_length_blocks=flb, force_nonlinear_path=False)
        frr = ree._reverb_freq_resp
        frames.append(dict(
            reset_before=0, n_partitions=n_part, fdb_urm=fdb,
            fq_is_none=0, fq=fq, stationary=0, fr=f32(fr),
            dominant_nearend=0, usable=1 if usable else 0,
            saturated=1 if saturated else 0, transparent=0,
            fdb=fdb, flb=flb, force_nl=0,
            render_psd=rp, capture_psd=cp, s2_linear=s2,
            erle=erle, erle_unb=erle_unb,
            r2=f32(r2).copy(), r2_unb=f32(r2_unb).copy(),
            last_direct=f32(ree._last_r2_direct_component).copy(),
            last_reverb=f32(ree._last_r2_reverb_component).copy(),
            tail=f32(frr.tail_response).copy(),
            avg_decay=float(frr.average_decay)))
    return frames, n_part


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ree_golden.bin'
    cfg = None
    case_blobs = []
    for stem in CASES:
        frames = []
        nf, mp, cfg_snap = run_case(stem, frames)
        if cfg is None and cfg_snap:
            cfg = cfg_snap
        case_blobs.append((frames, mp))
        # Count branch coverage for the report.
        n_lin = sum(1 for r in frames if r['usable'] and not r['saturated'])
        n_nl = sum(1 for r in frames if (not r['usable']) and not r['saturated'])
        n_sat = sum(1 for r in frames if r['saturated'])
        print(f"{os.path.basename(stem)}: {nf} frames  "
              f"linear={n_lin} nonlinear={n_nl} saturated={n_sat}")

    # 4th case: synthetic saturated-echo coverage (corpus never clips).
    sat_frames, sat_mp = build_synthetic_saturated(cfg)
    case_blobs.append((sat_frames, sat_mp))
    n_sl = sum(1 for r in sat_frames if r['usable'] and r['saturated'])
    n_snl = sum(1 for r in sat_frames if (not r['usable']) and r['saturated'])
    print(f"synthetic_saturated: {len(sat_frames)} frames  "
          f"saturated_linear={n_sl} saturated_nonlinear={n_snl}")

    with open(out, 'wb') as f:
        # ── config preamble (the C test feeds these into ree_init) ──
        f.write(struct.pack('<i', N_BINS))
        f.write(struct.pack('<i', cfg['hop_size']))
        f.write(struct.pack('<d', cfg['min_noise_floor_power']))
        f.write(struct.pack('<d', cfg['noise_gate_power_legacy']))
        f.write(struct.pack('<d', cfg['noise_gate_slope']))
        f.write(struct.pack('<d', cfg['stationary_gate_slope']))
        f.write(struct.pack('<i', cfg['model_reverb_in_nl']))
        f.write(struct.pack('<d', cfg['default_gain']))
        f.write(struct.pack('<d', cfg['tm_gain']))
        f.write(struct.pack('<i', cfg['erle_onset_comp']))
        f.write(struct.pack('<d', cfg['reverb_decay']))
        f.write(struct.pack('<d', cfg['reverb_mild_scale']))
        f.write(struct.pack('<i', cfg['reverb_enabled']))
        f.write(struct.pack('<d', cfg['reverb_tail_strength']))
        f.write(struct.pack('<i', cfg['use_aec3_residual_noise_gate']))
        f.write(struct.pack('<i', cfg['use_stationarity_properties']))
        f.write(struct.pack('<i', cfg['use_aec3_echo_gen_window']))
        f.write(struct.pack('<i', cfg['nl_r2_enabled']))
        f.write(struct.pack('<d', cfg['nl_r2_alpha']))
        f.write(struct.pack('<d', cfg['nl_norm_power']))
        f.write(struct.pack('<d', cfg['residual_noise_gate_power']))
        f.write(struct.pack('<i', cfg['noise_floor_hold_hops']))
        f.write(struct.pack('<i', cfg['use_freq_response']))
        f.write(struct.pack('<i', cfg['reverb_use_conservative']))
        f.write(struct.pack('<d', cfg['reverb_smoothing_base']))

        f.write(struct.pack('<i', len(case_blobs)))
        for frames, mp in case_blobs:
            f.write(struct.pack('<i', len(frames)))
            f.write(struct.pack('<i', mp))
            for r in frames:
                # mid-stream reset marker (clears noise floor + reverb model +
                # freq_resp before this frame's update_reverb_models)
                f.write(struct.pack('<i', r['reset_before']))
                # update_reverb_models record
                f.write(struct.pack('<i', r['n_partitions']))
                f.write(struct.pack('<i', r['fdb_urm']))
                f.write(struct.pack('<i', r['fq_is_none']))
                f.write(struct.pack('<d', r['fq']))
                f.write(struct.pack('<i', r['stationary']))
                f32(r['fr']).ravel().tofile(f)
                # estimate inputs
                f.write(struct.pack('<i', r['dominant_nearend']))
                f.write(struct.pack('<i', r['usable']))
                f.write(struct.pack('<i', r['saturated']))
                f.write(struct.pack('<i', r['transparent']))
                f.write(struct.pack('<i', r['fdb']))
                f.write(struct.pack('<i', r['flb']))
                f.write(struct.pack('<i', r['force_nl']))
                r['render_psd'].tofile(f)
                r['capture_psd'].tofile(f)
                r['s2_linear'].tofile(f)
                r['erle'].tofile(f)
                r['erle_unb'].tofile(f)
                # expected outputs
                r['r2'].tofile(f)
                r['r2_unb'].tofile(f)
                r['last_direct'].tofile(f)
                r['last_reverb'].tofile(f)
                r['tail'].tofile(f)
                f.write(struct.pack('<d', r['avg_decay']))

    total = sum(len(fr) for fr, _ in case_blobs)
    print(f"wrote {out}  ({N_BINS} bins, {len(case_blobs)} cases, "
          f"{total} frames)")


if __name__ == '__main__':
    main()
