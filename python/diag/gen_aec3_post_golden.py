"""Generate a binary golden for the C aec3_post DRIVER port (WS5 Phase 5.5).

This is the INTEGRATION/orchestration golden for the AEC3 post-filter DRIVER —
``AEC._aec3_post`` (orchestrator.py). The per-bin float math of the sub-modules
(AecState / ResidualEchoEstimator / SuppressionGain / reverb models / ERLE /
stationarity / ...) is already bit-exact in their own parity tests; this golden
exercises the DRIVER's ASSEMBLY LOGIC + CNG + OLA stages that the .py owns:

  * PSD derivation from the fp32 spectra (|c64|² · _PSD_SCALE, all-f32)
  * E1 windowed-capture Y² (erle_windowed_capture_psd)
  * avg-render-reverb x2_reverb_for_erle (+ erle_render_x2_psd_scale)
  * coherence Γ²(Ŷ,Y) EMA + ERLE coh gate mask (erle_coh_gate, default ON)
  * CNG N2 tracking (y2_smoothed EMA, n2 track/slow-up, n2_initial transient,
    noise-floor clamp) → comfort_noise
  * E2 output-base select (output_capture_when_linear_unusable, |E|>|Y| guard)
  * gain apply + CNG injection (LCG sin-table indices + sqrt(2)sin LUT, DC/Nyq
    zeroed) + irfft + sqrt-Hann synth window + OLA

Approach: monkeypatch ``AEC._aec3_post`` on the REAL balanced pipeline.
Per hop we capture EVERY input the method reads (the raw filter spectra after
the shadow-output selection — i.e. ``error_spec`` / ``_sel_echo_spec`` — plus
``near_spec`` / ``far_spec`` for the PSDs and the coherence Γ², the X_buf slices
+ delay + steady decay for the avg-reverb, the raw time-domain blocks, and the
saturation level), the SUB-MODULE OUTPUTS we inject (``usable_linear`` /
``saturated_echo`` / ``r2`` / ``r2_unb`` / the final ``gain``), and the returned
``out[hop]``.  We ALSO capture the driver-derived intermediates
(far/near/echo/error PSD, capture_psd_erle, x2_reverb_for_erle, coh_gate_mask,
render_block_scaled, comfort_noise, pre-CNG e_out_spec) so the C side can assert
each driver stage bit-exact (bisection), not just the final out.

The C parity test re-derives all driver quantities from the captured raw inputs,
runs its own CNG/avg-reverb/coherence/OLA state in lockstep, injects the
sub-module outputs, and asserts every stage + the final out[hop] bit-exact.

Run THREE real cases (DT / FS / NE) for CNG/gate branch coverage; the DT case
is long enough to exercise the n2_initial transient release + the per-hop EMA.
A reset marker is emitted whenever ``_reset_aec3_post`` fired before a hop so the
C side re-inits its driver state (OLA / CNG / coherence / avg-reverb / form).

Layout (LE), all sizes in elements; see parity_aec3_post.c for the reader.

Run: python3 python/diag/gen_aec3_post_golden.py /tmp/aec3_post_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import AecConfig          # noqa: E402
from modules.orchestrator import AEC          # noqa: E402
from modules.state.aec_state import AecState  # noqa: E402
from modules.residual.residual_echo_estimator import ResidualEchoEstimator  # noqa: E402
from modules.residual.suppression_gain import SuppressionGain               # noqa: E402
from eval_aec_challenge import estimate_delay  # noqa: E402

# diag lives at python/diag → repo root is two levels up.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASES = [
    os.path.join(WAV, 'doubletalk',
                 '0I0XMl3M0ECO0U1N0cJvpg_doubletalk'),
    os.path.join(WAV, 'farend_singletalk',
                 '0KjzXA3g20qsd8zmSekADw_farend_singletalk'),
    os.path.join(WAV, 'nearend_singletalk',
                 '014AzuqPZku2004NbTTmcA_nearend_singletalk'),
]

HOP = 160
SR = 16000
FL = 832
MAX_FRAMES = 1200  # cap per case to bound the golden size (DT runs ~4187)
_PSD_SCALE = (32768.0) ** 2


def _w_i32(f, *vals):
    np.array(vals, dtype=np.int32).tofile(f)


def _w_f64(f, *vals):
    np.array(vals, dtype=np.float64).tofile(f)


def _w_f32(f, a):
    np.asarray(a, dtype=np.float32).ravel().tofile(f)


def _w_c64(f, a):
    """Write a complex64 array as interleaved (re,im) float32."""
    a = np.asarray(a, dtype=np.complex64).ravel()
    out = np.empty(a.size * 2, dtype=np.float32)
    out[0::2] = a.real
    out[1::2] = a.imag
    out.tofile(f)


def main():
    out = (sys.argv[1] if len(sys.argv) > 1
           else '/tmp/aec3_post_golden.bin')

    cfg = AecConfig.from_preset('balanced', sample_rate=SR, filter_length=FL)
    n_bins = 257
    fft_size = 512
    block_size = 320

    # ---- captured driver-construction constants (written once) ----
    consts = {}

    # ---- per-case captured rows ----
    case_blobs = []  # list of (n_frames, list-of-row-dicts)

    orig_post = AEC._aec3_post
    orig_ree_estimate = ResidualEchoEstimator.estimate
    orig_sg_get_gain = SuppressionGain.get_gain
    orig_reset_post = AEC._reset_aec3_post

    # capture the pre-irfft e_out_spec (post gain-apply + CNG injection) so the
    # C side can bisect the output stage (CNG / gain-apply vs OLA / irfft).
    _real_irfft = np.fft.irfft

    def _wrap_irfft(*a, **kw):
        spec = a[0]
        n = a[1] if len(a) > 1 else kw.get('n', None)
        if n == fft_size:
            grab['e_out_spec'] = np.asarray(spec, np.complex64).copy()
        return _real_irfft(*a, **kw)

    # cross-call capture between the sub-module patches and the post wrapper.
    grab = {}
    reset_pending = {'v': 0}

    def patched_ree_estimate(self, **kw):
        r2, r2u = orig_ree_estimate(self, **kw)
        grab['r2'] = np.asarray(r2, dtype=np.float32).copy()
        grab['r2u'] = np.asarray(r2u, dtype=np.float32).copy()
        return r2, r2u

    def patched_sg_get_gain(self, **kw):
        g = orig_sg_get_gain(self, **kw)
        grab['gain'] = np.asarray(g, dtype=np.float32).copy()
        grab['cn'] = np.asarray(kw['comfort_noise_spectrum'],
                                dtype=np.float32).copy()
        return g

    orig_state_update = AecState.update

    def patched_state_update(self, **kw):
        # capture the EXACT driver-assembled ERLE inputs (x2_reverb_for_erle,
        # capture_psd_erle, coh_gate_mask) for the per-stage C assertion.
        x2r = kw.get('x2_reverb_for_erle', None)
        cpe = kw.get('capture_psd_erle', None)
        cgm = kw.get('erle_coh_gate_mask', None)
        grab['x2r'] = (np.asarray(x2r, np.float32).copy()
                       if x2r is not None else None)
        grab['cpe'] = (np.asarray(cpe, np.float32).copy()
                       if cpe is not None else None)
        grab['cgm'] = (np.asarray(cgm, bool).astype(np.uint8).copy()
                       if cgm is not None else None)
        return orig_state_update(self, **kw)

    def patched_reset_post(self, *args, **kwargs):
        reset_pending['v'] = 1
        return orig_reset_post(self, *args, **kwargs)

    # rows are collected into this mutable container (reset per case).
    _ctx = {'rows': []}

    # Patch the shadow-output selection to capture its result (the actual
    # error_spec / echo_spec the driver consumes), then run the real method.
    orig_select = AEC._aec3_select_linear_filter_output

    def patched_select(self, *, e_refined_time, near_end_block):
        esw, echo = orig_select(self, e_refined_time=e_refined_time,
                                near_end_block=near_end_block)
        grab['sel_esw'] = np.asarray(esw, dtype=np.complex64).copy()
        grab['sel_echo'] = np.asarray(echo, dtype=np.complex64).copy()
        return esw, echo

    def real_patched_post(self, raw_output, near_end, far_end):
        flt = self.filter
        grab.clear()
        rst = reset_pending['v']
        reset_pending['v'] = 0

        # snapshot driver constants once
        if not consts:
            consts['n_bins'] = int(flt.n_freqs)
            consts['fft_size'] = int(flt.fft_size)
            consts['block_size'] = int(flt.block_size)
            consts['hop_size'] = int(flt.hop_size)
            consts['n_partitions'] = int(flt.n_partitions)
            consts['cng_y2_alpha'] = float(self._aec3_cng_y2_alpha)
            consts['cng_n2_track_freshness'] = float(
                self._aec3_cng_n2_track_freshness)
            consts['cng_n2_track_retention'] = float(
                self._aec3_cng_n2_track_retention)
            consts['cng_n2_slow_up'] = float(self._aec3_cng_n2_slow_up)
            consts['cng_n2_initial_alpha'] = float(
                self._aec3_cng_n2_initial_alpha)
            consts['cng_n2_update_onset_hops'] = int(
                self._aec3_cng_n2_update_onset_hops)
            consts['cng_n2_initial_duration_hops'] = int(
                self._aec3_cng_n2_initial_duration_hops)
            consts['noise_floor_int16sq'] = float(
                self._aec3_noise_floor_int16sq)
            consts['erle_coh_gate_alpha'] = float(
                self.config.erle_coh_gate_alpha)
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
            consts['synth_window'] = np.asarray(
                self._aec3_synth_window, dtype=np.float32).copy()
            consts['sqrt2_sin_lut'] = np.asarray(
                self._aec3_sqrt2_sin_lut, dtype=np.float32).copy()

        # --- pre-state snapshot of the avg-render-reverb inputs ---
        x2_at_delay = None
        x2_past = None
        decay_steady = 0.0
        x2_present = 0
        _ree = getattr(self, '_aec3_ree', None)
        _avg = getattr(self, '_aec3_avg_render_reverb', None)
        if (_ree is not None and _avg is not None
                and hasattr(flt, 'X_buf') and hasattr(flt, 'partition_idx')):
            _n_part = flt.n_partitions
            _curr_p = (flt.partition_idx - 1) % _n_part
            _delay = int(self._aec3_state.min_direct_path_filter_delay())
            # MUST match orchestrator's clamp (_n_part - 2); the real method
            # keeps _past_idx from wrapping to current. Using _n_part - 1 here
            # made x2_at_delay/x2_past diverge from the captured x2r golden
            # whenever the clamp was active (delay >= _n_part - 1).
            _delay = max(0, min(_delay, _n_part - 2))
            _delay_idx = (_curr_p - _delay) % _n_part
            _past_idx = (_curr_p - _delay - 1) % _n_part
            x2_at_delay = (np.abs(flt.X_buf[_delay_idx]) ** 2).astype(
                np.float32).copy()
            x2_past = (np.abs(flt.X_buf[_past_idx]) ** 2).astype(
                np.float32).copy()
            decay_steady = float(_ree._reverb_decay(dominant_nearend=False))
            x2_present = 1

        # --- run the REAL method (advances all state, returns out) ---
        np.fft.irfft = _wrap_irfft
        try:
            ret = orig_post(self, raw_output, near_end, far_end)
        finally:
            np.fft.irfft = _real_irfft

        # --- post-state reads (stable until next update) ---
        usable = int(bool(self._aec3_state.usable_linear_estimate()))
        sat_echo = int(bool(self._aec3_state.saturated_echo()))
        sat_level_gt = int(self._saturation_level > 0.5)

        sel_esw = np.asarray(
            grab.get('sel_esw', flt.error_spec_windowed), np.complex64)
        sel_echo = np.asarray(grab.get('sel_echo', flt.echo_spec), np.complex64)
        near_spec = np.asarray(flt.near_spec, np.complex64)
        far_spec = np.asarray(flt.far_spec, np.complex64)
        echo_coh = np.asarray(flt.echo_spec, np.complex64)
        # E1 near_spec_win uses the ORIGINAL esw + echo (orchestrator line ~3049)
        esw_orig = np.asarray(flt.error_spec_windowed, np.complex64)
        echo_orig = np.asarray(flt.echo_spec, np.complex64)
        near_spec_win_e1 = (esw_orig + echo_orig).astype(np.complex64)
        # E2 y_base uses the SELECTED esw + sel_echo (orchestrator line ~3609)
        y_base = (sel_esw + sel_echo).astype(sel_esw.dtype, copy=False)

        # numpy-internal abs(c64) is a SIMD path not portably reproducible in C;
        # capture the f32 magnitudes so the C derives the PSDs by squaring them
        # (the (a*a)*PSD arithmetic IS the parity-relevant driver math).
        def _absf(x):
            return np.abs(np.asarray(x, np.complex64)).astype(np.float32)

        # driver intermediates (recomputed here exactly as the method does, so
        # the C side can assert each stage; PSD = |c|^2 * _PSD_SCALE in f32).
        near_psd = (_absf(near_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        far_psd = (_absf(far_spec) ** 2 * _PSD_SCALE).astype(np.float32)
        echo_psd = (_absf(sel_echo) ** 2 * _PSD_SCALE).astype(np.float32)
        error_psd = (_absf(sel_esw) ** 2 * _PSD_SCALE).astype(np.float32)
        capture_psd_erle = (_absf(near_spec_win_e1) ** 2
                            * _PSD_SCALE).astype(np.float32)

        row = {
            'reset_before': int(rst),
            # complex spectra (reproducible complex ops: sye EMA, E2 base,
            # gain-apply, irfft)
            'near_spec': near_spec.copy(),
            'echo_coh': echo_coh.copy(),
            'error_spec': sel_esw.copy(),      # = _sel_esw, drives e_out_spec
            'sel_echo': sel_echo.copy(),
            # f32 magnitudes (non-reproducible numpy abs → captured, squared in C)
            'abs_near': _absf(near_spec),
            'abs_far': _absf(far_spec),
            'abs_sel_echo': _absf(sel_echo),
            'abs_error': _absf(sel_esw),       # |error_spec|
            'abs_echo_coh': _absf(echo_coh),
            'abs_nsw_e1': _absf(near_spec_win_e1),
            'abs_ybase': _absf(y_base),        # E2 guard |y_base|
            # avg-render-reverb inputs (x2 already squared in f32)
            'x2_present': x2_present,
            'x2_at_delay': (x2_at_delay if x2_at_delay is not None
                            else np.zeros(n_bins, np.float32)),
            'x2_past': (x2_past if x2_past is not None
                        else np.zeros(n_bins, np.float32)),
            'decay_steady': decay_steady,
            # scalars / injected sub-module outputs
            'sat_level_gt': sat_level_gt,
            'usable': usable,
            'sat_echo': sat_echo,
            'gain': grab['gain'].copy(),
            # captured driver intermediates (per-stage assertion)
            'near_psd': near_psd,
            'far_psd': far_psd,
            'echo_psd': echo_psd,
            'error_psd': error_psd,
            'capture_psd_erle': capture_psd_erle,
            # the EXACT driver-assembled ERLE inputs the method fed AecState
            'x2r_present': int(grab.get('x2r') is not None),
            'x2r': (grab['x2r'] if grab.get('x2r') is not None
                    else np.zeros(n_bins, np.float32)),
            'cpe_present': int(grab.get('cpe') is not None),
            'cgm_present': int(grab.get('cgm') is not None),
            'cgm': (grab['cgm'] if grab.get('cgm') is not None
                    else np.zeros(n_bins, np.uint8)),
            'comfort_noise': grab['cn'].copy(),
            # pre-irfft e_out_spec (post gain-apply + CNG injection) for bisect
            'e_out_spec': np.asarray(
                grab['e_out_spec'], np.complex64).copy(),
            # final output
            'out': np.asarray(ret, np.float32).copy(),
        }
        _ctx['rows'].append(row)
        return ret

    AEC._aec3_post = real_patched_post
    AEC._aec3_select_linear_filter_output = patched_select
    AecState.update = patched_state_update
    ResidualEchoEstimator.estimate = patched_ree_estimate
    SuppressionGain.get_gain = patched_sg_get_gain
    AEC._reset_aec3_post = patched_reset_post

    try:
        for stem in CASES:
            mic, sr = sf.read(stem + '_mic.wav', dtype='float32')
            ref, _ = sf.read(stem + '_lpb.wav', dtype='float32')
            n = min(len(mic), len(ref))
            mic = mic[:n]
            ref = ref[:n]
            delay = estimate_delay(mic, ref, sr)
            if 0 < delay < n:
                ra = np.zeros(n, dtype=np.float32)
                ra[delay:] = ref[:n - delay]
            else:
                ra = ref[:n]

            np.random.seed(0)
            aec = AEC(cfg)
            _ctx['rows'] = []

            pos = 0
            nf = 0
            while pos + HOP <= n and nf < MAX_FRAMES:
                aec.process(mic[pos:pos + HOP], ra[pos:pos + HOP])
                pos += HOP
                nf += 1
            case_blobs.append((len(_ctx['rows']), list(_ctx['rows'])))
    finally:
        AEC._aec3_post = orig_post
        AEC._aec3_select_linear_filter_output = orig_select
        AecState.update = orig_state_update
        ResidualEchoEstimator.estimate = orig_ree_estimate
        SuppressionGain.get_gain = orig_sg_get_gain
        AEC._reset_aec3_post = orig_reset_post

    # ---- write the binary ----
    with open(out, 'wb') as f:
        # header
        _w_i32(f, consts['n_bins'], consts['fft_size'], consts['block_size'],
               consts['hop_size'], consts['n_partitions'])
        _w_i32(f, consts['erle_coh_gate_enabled'],
               consts['erle_windowed_capture_psd'],
               consts['erle_render_x2_psd_scale'],
               consts['output_capture_when_linear_unusable'],
               consts['enable_cng'])
        _w_i32(f, consts['cng_n2_update_onset_hops'],
               consts['cng_n2_initial_duration_hops'])
        _w_f64(f, consts['cng_y2_alpha'], consts['cng_n2_track_freshness'],
               consts['cng_n2_track_retention'], consts['cng_n2_slow_up'],
               consts['cng_n2_initial_alpha'], consts['noise_floor_int16sq'],
               consts['erle_coh_gate_alpha'], consts['erle_coh_gate_threshold'])
        _w_f32(f, consts['synth_window'])      # block_size
        _w_f32(f, consts['sqrt2_sin_lut'])     # 32
        _w_i32(f, len(case_blobs))

        for n_frames, rows in case_blobs:
            _w_i32(f, n_frames)
            for r in rows:
                _w_i32(f, r['reset_before'])
                # complex spectra (reproducible complex ops)
                _w_c64(f, r['near_spec'])
                _w_c64(f, r['echo_coh'])
                _w_c64(f, r['error_spec'])
                _w_c64(f, r['sel_echo'])
                # f32 magnitudes (numpy abs → captured, squared in C)
                _w_f32(f, r['abs_near'])
                _w_f32(f, r['abs_far'])
                _w_f32(f, r['abs_sel_echo'])
                _w_f32(f, r['abs_error'])
                _w_f32(f, r['abs_echo_coh'])
                _w_f32(f, r['abs_nsw_e1'])
                _w_f32(f, r['abs_ybase'])
                # avg-render-reverb inputs
                _w_i32(f, r['x2_present'])
                _w_f32(f, r['x2_at_delay'])
                _w_f32(f, r['x2_past'])
                _w_f64(f, r['decay_steady'])
                # scalars / injected sub-module outputs
                _w_i32(f, r['sat_level_gt'], r['usable'], r['sat_echo'])
                _w_f32(f, r['gain'])
                # captured driver intermediates (per-stage assertion)
                _w_f32(f, r['near_psd'])
                _w_f32(f, r['far_psd'])
                _w_f32(f, r['echo_psd'])
                _w_f32(f, r['error_psd'])
                _w_f32(f, r['capture_psd_erle'])
                _w_i32(f, r['x2r_present'])
                _w_f32(f, r['x2r'])
                _w_i32(f, r['cpe_present'])
                _w_i32(f, r['cgm_present'])
                r['cgm'].astype(np.uint8).tofile(f)
                _w_f32(f, r['comfort_noise'])
                _w_c64(f, r['e_out_spec'])      # pre-irfft (bisect)
                # final output
                _w_f32(f, r['out'])             # hop_size
    # stats
    tot = sum(nf for nf, _ in case_blobs)
    n_reset = sum(r['reset_before'] for _, rows in case_blobs for r in rows)
    n_usable = sum(r['usable'] for _, rows in case_blobs for r in rows)
    n_satecho = sum(r['sat_echo'] for _, rows in case_blobs for r in rows)
    print(f"wrote {out}")
    print(f"  cases={len(case_blobs)} total_hops={tot} resets={n_reset}")
    print(f"  usable_linear True: {n_usable}/{tot}  saturated_echo: "
          f"{n_satecho}/{tot}")
    print(f"  n_bins={consts['n_bins']} fft={consts['fft_size']} "
          f"block={consts['block_size']} hop={consts['hop_size']} "
          f"n_part={consts['n_partitions']}")
    print(f"  enable_cng={consts['enable_cng']} "
          f"coh_gate={consts['erle_coh_gate_enabled']} "
          f"E1={consts['erle_windowed_capture_psd']} "
          f"x2scale={consts['erle_render_x2_psd_scale']} "
          f"E2={consts['output_capture_when_linear_unusable']}")


if __name__ == '__main__':
    main()
