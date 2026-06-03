"""Generate a binary golden for the C PBFDKF v3.22 port (WS5 Phase 5.6).

This is the LINEAR-FILTER golden — the crux of the C port. It hooks the REAL
balanced pipeline (NOT synthetic frames) and monkeypatches PBFDKF.process on a
doubletalk case, capturing per hop EVERYTHING the C needs to replay the v3.22
per-bin H_error Kalman update + leakage refresh in lockstep:

  External per-hop inputs (set by the orchestrator BEFORE process runs; the C
  sets these on its filter before calling pbfdkf_process):
    * near[hop], far[hop]                     (the delay-aligned hop blocks)
    * mu_scale_arr[n_bins]  (post-RSA-mask)   — the array _update_weights_aec3
        actually consumes. Capturing it post-mask means the C never models the
        RenderSignalAnalyzer object; the mask is already folded in. For paths
        that don't reach _update_weights_aec3 (far-inactive / gated) we still
        emit the broadcast mu_scale so the layout is uniform (C ignores it).
    * poor_excitation_counter, saturated_capture, block_stationary,
      disallow_leakage_diverged                (orchestrator-driven gates)
    * e2_coarse_per_bin[n_bins] (+ valid flag) (shadow E² for the refresh)
    * erl_per_bin[n_bins]  = Σ_p|W_p|²         (orchestrator-set each hop)

  Internal state the C EVOLVES and we assert against (post-hop):
    * output[hop]                              (FFT-derived → rtol)
    * error_spec, echo_spec  (complex64)       (FFT-derived → rtol)
    * W[n_partitions][n_bins] (complex64)      (FFT-derived after TD-constraint
                                                round-trip → rtol)
    * H_error_per_bin[n_bins]  (float32)       (NON-FFT → bit-exact)
    * error_psd[n_bins], R[n_bins] (float32)   (NON-FFT → bit-exact)
    * partition_idx, call_counter, initial_state_active, init_render_hops

The C parity test (parity_pbfdkf.c) constructs a PBFDKF with the captured
config, then per hop: sets the external inputs, calls pbfdkf_process, and
asserts the non-FFT state bit-exact + the FFT-derived arrays within rtol<1e-4
(atol 1e-6). Covers startup (gated), convergence, far-inactive, block-
stationary, and full-update paths over the whole DT case (~4186 hops).

Layout (LE), all sizes in elements:
  int32 n_bins, n_partitions, hop, fft_size, block_size, n_hops
  float32 mu, delta
  float32 h_error_init, h_error_floor, h_error_ceil
  float32 leak_conv, leak_div, leak_conv_tr, leak_div_tr
  float32 q_high (scalar; all bins equal), q_low
  int32 poor_exc_init, init_state_thr
  per hop:
    float32 near[hop], far[hop]
    int32   path           (0=far_inactive, 1=gated_refresh, 2=full_update)
    int32   poor_exc, sat, block_stat, disallow
    int32   e2cpb_valid
    float32 e2_coarse_per_bin[n_bins]
    float32 erl_per_bin[n_bins]
    float32 mu_scale_arr[n_bins]
    int32   ext_call_counter        (call_counter BEFORE this hop's ++)
    int32   ext_init_state_active   (initial_state_active BEFORE this hop)
    int32   ext_init_render_hops    (BEFORE this hop)
    -- expected post-hop --
    float32 out[hop]
    float32 error_re[n_bins], error_im[n_bins]
    float32 echo_re[n_bins], echo_im[n_bins]
    float32 W_re[n_part*n_bins], W_im[n_part*n_bins]
    float32 H_error[n_bins]
    float32 error_psd[n_bins], R[n_bins]
    int32   partition_idx, call_counter, init_state_active, init_render_hops

Run: python3 python/diag/gen_pbfdkf_golden.py /tmp/pbfdkf_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import AecConfig          # noqa: E402
from modules.orchestrator import AEC          # noqa: E402
from modules.filters import PBFDKF            # noqa: E402
from modules import aec3_scale                # noqa: E402
from eval_aec_challenge import estimate_delay  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAV = os.path.join(ROOT, 'wav', 'aec_challenge_blind')
CASE = os.path.join(WAV, 'doubletalk', '0I0XMl3M0ECO0U1N0cJvpg_doubletalk')

HOP = 160
SR = 16000
FL = 832
MAX_HOPS = 4200   # whole DT case (~4186 hops)


def _w_i32(f, *vals):
    np.array(vals, dtype=np.int32).tofile(f)


def _w_f32(f, *vals):
    np.array(vals, dtype=np.float32).tofile(f)


def _w_f32a(f, a):
    np.asarray(a, dtype=np.float32).ravel().tofile(f)


def _w_c64(f, a):
    """complex64 array → re[n] then im[n] (not interleaved)."""
    a = np.asarray(a, dtype=np.complex64).ravel()
    a.real.astype(np.float32).tofile(f)
    a.imag.astype(np.float32).tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pbfdkf_golden.bin'

    cfg = AecConfig.from_preset('balanced', sample_rate=SR, filter_length=FL)
    np.random.seed(0)
    aec = AEC(cfg)
    flt = aec.filter
    assert isinstance(flt, PBFDKF), type(flt)

    K = flt.n_freqs
    N = flt.n_partitions

    mic, _ = sf.read(CASE + '_mic.wav')
    lpb, _ = sf.read(CASE + '_lpb.wav')
    mic = mic.astype(np.float32)
    lpb = lpb.astype(np.float32)
    n = min(len(mic), len(lpb))
    delay = estimate_delay(mic, lpb, SR)
    ref = np.zeros(n, dtype=np.float32)
    if 0 < delay < n:
        ref[delay:] = lpb[:n - delay]
    else:
        ref = lpb[:n].copy()
    mic = mic[:n]

    orig_proc = PBFDKF.process
    orig_uw = PBFDKF._update_weights
    orig_aec3 = PBFDKF._update_weights_aec3

    grab = {}

    def p_uw(self, curr_p, mu_scale, error_override=None):
        # capture the post-RSA mu_scale_arr the aec3 path will use. Reproduce
        # the same broadcast the real method does (scalar→full, then RSA mask).
        msa = np.asarray(mu_scale, dtype=np.float32)
        if msa.ndim == 0:
            msa = np.full(K, float(msa), dtype=np.float32)
        if self._render_signal_analyzer is not None:
            mask = np.ones(K, dtype=np.float32)
            self._render_signal_analyzer.mask_regions_around_narrow_bands(mask)
            msa = (msa * mask).astype(np.float32)
        grab['mu_arr'] = msa.copy()
        return orig_uw(self, curr_p, mu_scale, error_override)

    def p_aec3(self, curr_p, mu_scale_arr, error_psd, error_override=None):
        grab['aec3_fired'] = True
        return orig_aec3(self, curr_p, mu_scale_arr, error_psd, error_override)

    rows = []

    def p_proc(self, near_end, far_end, mu_scale=1.0, defer_update=False):
        # snapshot external inputs + pre-hop internal counters. The orchestrator
        # may have fired handle_echo_path_change BEFORE this call (resets
        # call_counter→0, poor_exc→400, H_error→10000). We capture the pre-hop
        # values AFTER any such reset; the C SETS the counters from these and
        # re-applies the H_error reset when epc_h_reset fires. (W is untouched
        # by EPC — zero_filter defaults False — so it evolves continuously.)
        pre_he0 = float(self.H_error_per_bin[0])
        ext = {
            'near': np.asarray(near_end, np.float32).copy(),
            'far': np.asarray(far_end, np.float32).copy(),
            'poor_exc': int(self._poor_excitation_counter),
            'sat': int(bool(self._saturated_capture)),
            'block_stat': int(bool(getattr(self, '_block_stationary', False))),
            'disallow': int(bool(self._disallow_leakage_diverged)),
            'e2cpb_valid': int(self._e2_coarse_per_bin is not None),
            'e2cpb': (np.asarray(self._e2_coarse_per_bin, np.float32).copy()
                      if self._e2_coarse_per_bin is not None
                      else np.zeros(K, np.float32)),
            'erl': np.asarray(self._erl_per_bin, np.float32).copy(),
            'pre_call': int(self._call_counter),
            'pre_init': int(bool(self._initial_state_active)),
            'pre_render': int(self._initial_state_active_render_hops),
            # EPC H_error reset detector: evolved H_error is clamped to ceil=2.0,
            # so a pre-hop value >> ceil means handle_echo_path_change reset it.
            'epc_h_reset': int(pre_he0 > 100.0),
        }
        grab.clear()
        far_e = float(np.sum(np.asarray(far_end, np.float32) ** 2) / self.hop_size)
        ret = orig_proc(self, near_end, far_end, mu_scale, defer_update)

        if far_e <= 1e-4:
            path = 0
            mu_arr = np.ones(K, np.float32)
        elif grab.get('aec3_fired', False):
            path = 2
            mu_arr = grab['mu_arr']
        else:
            path = 1
            mu_arr = grab.get('mu_arr', np.ones(K, np.float32))

        rows.append({
            **ext, 'path': path, 'mu_arr': mu_arr,
            'out': np.asarray(ret, np.float32).copy(),
            'error_spec': self.error_spec.copy(),
            'echo_spec': self.echo_spec.copy(),
            'W': self.W.copy(),
            'H_error': self.H_error_per_bin.copy(),
            'error_psd': self._error_psd.copy(),
            'R': self.R.copy(),
            'partition_idx': int(self.partition_idx),
            'call_counter': int(self._call_counter),
            'init_state_active': int(bool(self._initial_state_active)),
            'init_render_hops': int(self._initial_state_active_render_hops),
        })
        return ret

    PBFDKF.process = p_proc
    PBFDKF._update_weights = p_uw
    PBFDKF._update_weights_aec3 = p_aec3
    try:
        cnt = 0
        for i in range(0, n - HOP, HOP):
            aec.process(mic[i:i + HOP], ref[i:i + HOP])
            cnt += 1
            if cnt >= MAX_HOPS:
                break
    finally:
        PBFDKF.process = orig_proc
        PBFDKF._update_weights = orig_uw
        PBFDKF._update_weights_aec3 = orig_aec3

    n_hops = len(rows)
    # path distribution sanity
    paths = np.array([r['path'] for r in rows])
    print(f"captured {n_hops} hops  far_inactive={int((paths==0).sum())} "
          f"gated={int((paths==1).sum())} full={int((paths==2).sum())}")

    with open(out, 'wb') as f:
        _w_i32(f, K, N, HOP, flt.fft_size, flt.block_size, n_hops)
        _w_f32(f, float(flt.mu), float(flt.delta))
        _w_f32(f, float(aec3_scale.H_ERROR_INIT_FLOAT),
               float(flt._h_error_floor), float(flt._h_error_ceil))
        _w_f32(f, float(flt._leakage_converged), float(flt._leakage_diverged),
               float(aec3_scale.LEAKAGE_CONVERGED_TRANSIENT_PER_HOP),
               float(aec3_scale.LEAKAGE_DIVERGED_TRANSIENT_PER_HOP))
        _w_f32(f, float(flt.Q_high[0]), float(flt.Q_low[0]))
        _w_i32(f, int(flt._initial_state_threshold_hops), 0)

        for r in rows:
            _w_f32a(f, r['near'])
            _w_f32a(f, r['far'])
            _w_i32(f, r['path'], r['poor_exc'], r['sat'], r['block_stat'],
                   r['disallow'], r['e2cpb_valid'])
            _w_f32a(f, r['e2cpb'])
            _w_f32a(f, r['erl'])
            _w_f32a(f, r['mu_arr'])
            _w_i32(f, r['pre_call'], r['pre_init'], r['pre_render'],
                   r['epc_h_reset'])
            # expected
            _w_f32a(f, r['out'])
            _w_c64(f, r['error_spec'])
            _w_c64(f, r['echo_spec'])
            _w_c64(f, r['W'])
            _w_f32a(f, r['H_error'])
            _w_f32a(f, r['error_psd'])
            _w_f32a(f, r['R'])
            _w_i32(f, r['partition_idx'], r['call_counter'],
                   r['init_state_active'], r['init_render_hops'])

    print(f"wrote {out}  ({n_hops} hops, {K} bins, {N} partitions)")


if __name__ == '__main__':
    main()
