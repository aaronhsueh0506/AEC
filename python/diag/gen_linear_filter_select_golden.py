"""Generate a binary golden for the C linear-filter-output selection port
(AEC._aec3_select_linear_filter_output → UseRefinedOutput + FormLinearFilterOutput).

Hooks the REAL balanced pipeline (NOT synthetic frames) on the same doubletalk
case + delay-align harness as gen_pbfdkf_golden.py, monkeypatches
AEC._aec3_select_linear_filter_output to capture, per hop where it actually
fires (i.e. self._last_shadow_output_time is not None — skipped on frame 1):

  Inputs the C replays:
    * e_refined_time[hop]            (= raw_output, the kwarg)
    * near_end[hop]                  (= near_end_block, the kwarg)
    * e_coarse_time[hop]             (= self._last_shadow_output_time, pre-call)
    * error_spec_windowed[n_freqs]   (filter.error_spec_windowed, pre-call)
    * echo_spec[n_freqs]             (filter.echo_spec, pre-call)
  Plus once: sqrt_hann[block_size] (filter._sqrt_hann_analysis), and the
  config sizes (hop, block_size, fft_size, n_freqs).

  Expected outputs the C asserts BIT-EXACT (uint32 bit pattern):
    * selected_esw[n_freqs]   (complex64)
    * selected_echo_spec[n_freqs] (complex64)

The C parity_linear_filter_select.c constructs a LinearFilterSelect (init →
_form_prev_output_time=None, _form_last_selection=True), then per captured hop
calls linear_filter_select with the stored inputs and asserts both complex
spectra match bit-for-bit. State (_form_prev_output_time, _form_last_selection,
_refined_filter_output_last_selected) evolves continuously in the C exactly as
in Python across all captured hops.

Layout (LE), sizes in elements:
  int32 hop, block_size, fft_size, n_freqs, n_hops
  float32 sqrt_hann[block_size]
  per hop:
    float32 e_refined_time[hop]
    float32 near_end[hop]
    float32 e_coarse_time[hop]
    float32 error_spec_windowed_re[n_freqs], error_spec_windowed_im[n_freqs]
    float32 echo_spec_re[n_freqs], echo_spec_im[n_freqs]
    -- expected --
    float32 selected_esw_re[n_freqs], selected_esw_im[n_freqs]
    float32 selected_echo_re[n_freqs], selected_echo_im[n_freqs]

Run (from the AEC repo root, NOT c_impl):
  python3 python/diag/gen_linear_filter_select_golden.py /tmp/lfs_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.config import AecConfig          # noqa: E402
from modules.orchestrator import AEC          # noqa: E402
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


def _w_f32a(f, a):
    np.asarray(a, dtype=np.float32).ravel().tofile(f)


def _w_c64(f, a):
    """complex64 array → re[n] then im[n] (not interleaved)."""
    a = np.asarray(a, dtype=np.complex64).ravel()
    a.real.astype(np.float32).tofile(f)
    a.imag.astype(np.float32).tofile(f)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/lfs_golden.bin'

    cfg = AecConfig.from_preset('balanced', sample_rate=SR, filter_length=FL)
    np.random.seed(0)
    aec = AEC(cfg)
    flt = aec.filter
    K = flt.n_freqs
    block_size = flt.block_size
    fft_size = flt.fft_size

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

    orig_sel = AEC._aec3_select_linear_filter_output
    rows = []

    def p_sel(self, *, e_refined_time, near_end_block):
        # Capture inputs BEFORE the call (e_coarse_time / spectra are read
        # inside the real method from current state).
        ext = {
            'e_refined': np.asarray(e_refined_time, np.float32).copy(),
            'near': np.asarray(near_end_block, np.float32).copy(),
            'e_coarse': np.asarray(self._last_shadow_output_time,
                                   np.float32).copy(),
            'esw': self.filter.error_spec_windowed.copy(),
            'echo': self.filter.echo_spec.copy(),
        }
        sel_esw, sel_echo = orig_sel(
            self, e_refined_time=e_refined_time, near_end_block=near_end_block)
        rows.append({
            **ext,
            'sel_esw': np.asarray(sel_esw, np.complex64).copy(),
            'sel_echo': np.asarray(sel_echo, np.complex64).copy(),
        })
        return sel_esw, sel_echo

    AEC._aec3_select_linear_filter_output = p_sel
    try:
        cnt = 0
        for i in range(0, n - HOP, HOP):
            aec.process(mic[i:i + HOP], ref[i:i + HOP])
            cnt += 1
            if cnt >= MAX_HOPS:
                break
    finally:
        AEC._aec3_select_linear_filter_output = orig_sel

    n_hops = len(rows)
    sqrt_hann = np.asarray(flt._sqrt_hann_analysis, np.float32)
    assert sqrt_hann.shape[0] == block_size, (sqrt_hann.shape, block_size)
    print(f"captured {n_hops} selection hops  K={K} block={block_size} "
          f"fft={fft_size}")

    with open(out, 'wb') as f:
        _w_i32(f, HOP, block_size, fft_size, K, n_hops)
        _w_f32a(f, sqrt_hann)
        for r in rows:
            _w_f32a(f, r['e_refined'])
            _w_f32a(f, r['near'])
            _w_f32a(f, r['e_coarse'])
            _w_c64(f, r['esw'])
            _w_c64(f, r['echo'])
            _w_c64(f, r['sel_esw'])
            _w_c64(f, r['sel_echo'])

    print(f"wrote {out}  ({n_hops} hops, {K} bins)")


if __name__ == '__main__':
    main()
