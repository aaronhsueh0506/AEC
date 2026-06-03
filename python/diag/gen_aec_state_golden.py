"""Generate a binary golden for the C aec_state port (WS5 Phase 5.2 capstone).

This is the INTEGRATION/orchestration golden. Rather than synthesise inputs, it
hooks the REAL balanced pipeline: it monkeypatches ``AecState.update`` (and
``handle_echo_path_change``) on a real doubletalk case, dumps every ``update()``
input + every public-query output AFTER each update, and writes a raw LE binary.
The C parity test (parity_aec_state.c) rebuilds a fresh AecState from the same
AecStateConfig, replays the SAME inputs frame-by-frame, and asserts every output
bit-exact across the full case (startup -> first-converged transition -> evolved
ERLE/ERL/quality).

A SECOND segment after the real case injects synthetic
handle_echo_path_change events (gain_change, then a delay_change full-reset
cascade) so the reset paths are exercised even though the real DT case has no
EPV events.

Layout (LE), all sizes in elements:
  header:
    int32  n_bins
    int32  filter_taps_size            (FilterAnalyzer full length; 0 if analyzer off)
    int32  num_capture_channels
    int32  hop_size
    int32  enable_filter_analyzer
    int32  erle_startup_hops
    int32  erl_startup_hops
    int32  echo_can_saturate
    int32  use_linear_filter
    int32  conservative_initial_phase
    f64    initial_state_seconds
    int32  delay_headroom_samples
    f64    erle_min, erle_max_l, erle_max_h
    int32  n_events                    (EPV events before frame replay; the
                                        events run in REPLAY order interleaved
                                        with frames via the per-frame tag)
    int32  n_frames
  n_frames * frame:
    -- per-frame control tag --
    int32  kind                        (0 = update, 1 = handle_echo_path_change)
    -- if kind == 1 (handle_echo_path_change) --
       int32 gain_change
       int32 delay_change              (DelayAdjustment: 0 NONE / 1 FLUSH / 2 NEW)
       (no further payload; the C side runs handle_echo_path_change and emits
        NO output row — it is a pure mutator before the next update)
    -- if kind == 0 (update) --
       inputs:
         int32  bridge_filter_converged
         int32  ext_delay_reported     (0 == None)
         int32  ext_delay_quality
         int32  ext_delay_samples
         f32    render_psd[n_bins]
         f32    capture_psd[n_bins]
         f32    error_psd[n_bins]
         f32    echo_psd[n_bins]
         int32  active_render
         f64    subtractor_s_refined_max_abs
         f64    subtractor_s_coarse_max_abs
         f64    echo_path_gain
         int32  render_block_present
         int32  render_block_len
         f32    render_block[render_block_len]    (only if present)
         int32  filter_taps_present
         int32  filter_taps_len
         f32    filter_taps_full[filter_taps_len] (only if present)
         int32  x2_reverb_present
         f32    x2_reverb_for_erle[n_bins]        (only if present)
         int32  capture_psd_erle_present
         f32    capture_psd_erle[n_bins]          (only if present)
         int32  coh_gate_mask_present
         uint8  erle_coh_gate_mask[n_bins]        (only if present)
         int32  capture_saturation_pre            (capture_signal_saturation set
                                                   by update_capture_saturation
                                                   BEFORE this update, in the
                                                   real orchestrator)
       outputs (queried AFTER update):
         int32  usable_linear_estimate
         int32  active_render_q
         int32  saturated_echo
         int32  min_direct_path_filter_delay
         f64    fullband_erle_log2
         int32  inst_quality_valid
         f64    inst_quality                       (only meaningful if valid)
         f32    erle[n_bins]                        (onset_compensated=False)
         f32    erle_unbounded[n_bins]
         f32    erl[n_bins]
         f64    erl_time_domain

Run: python3 python/diag/gen_aec_state_golden.py /tmp/aec_state_golden.bin
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aec import AEC, AecConfig                                  # noqa: E402
from modules.state.aec_state import AecState                    # noqa: E402
from modules.delay.delay_types import (                         # noqa: E402
    DelayAdjustment, EchoPathVariability,
)

CASE_MIC = ('wav/aec_challenge_blind/doubletalk/'
            '0I0XMl3M0ECO0U1N0cJvpg_doubletalk_mic.wav')
CASE_LPB = ('wav/aec_challenge_blind/doubletalk/'
            '0I0XMl3M0ECO0U1N0cJvpg_doubletalk_lpb.wav')

HOP = 160
SR = 16000
FL = 832

# Captured (kind, payload) rows, in replay order.
_ROWS = []


def _w_i32(f, *vals):
    np.array(vals, dtype=np.int32).tofile(f)


def _w_f64(f, *vals):
    np.array(vals, dtype=np.float64).tofile(f)


def _w_f32arr(f, a):
    np.asarray(a, dtype=np.float32).ravel().tofile(f)


def _capture_inputs_outputs(state, kwargs):
    """Snapshot inputs (the kwargs dict) + post-update query outputs."""
    n_bins = state._config.n_bins
    ext = kwargs.get('external_delay', None)
    rb = kwargs.get('render_block', None)
    ft = kwargs.get('filter_taps_full', None)
    x2r = kwargs.get('x2_reverb_for_erle', None)
    cpe = kwargs.get('capture_psd_erle', None)
    cgm = kwargs.get('erle_coh_gate_mask', None)
    bridge = kwargs['bridge']

    row = {
        'kind': 0,
        'bridge_filter_converged': int(bool(bridge.filter_converged)),
        'ext_reported': int(ext is not None),
        'ext_quality': int(ext.quality.value) if ext is not None else 0,
        'ext_samples': int(ext.delay) if ext is not None else 0,
        'render_psd': np.asarray(kwargs['render_psd'], dtype=np.float32).copy(),
        'capture_psd': np.asarray(kwargs['capture_psd'], dtype=np.float32).copy(),
        'error_psd': np.asarray(kwargs['error_psd'], dtype=np.float32).copy(),
        'echo_psd': np.asarray(kwargs['echo_psd'], dtype=np.float32).copy(),
        'active_render': int(bool(kwargs['active_render'])),
        's_ref': float(kwargs.get('subtractor_s_refined_max_abs', 0.0)),
        's_coa': float(kwargs.get('subtractor_s_coarse_max_abs', 0.0)),
        'epg': float(kwargs.get('echo_path_gain', 1.0)),
        'rb_present': int(rb is not None),
        'rb': (np.asarray(rb, dtype=np.float32).copy()
               if rb is not None else np.zeros(0, np.float32)),
        'ft_present': int(ft is not None),
        'ft': (np.asarray(ft, dtype=np.float32).copy()
               if ft is not None else np.zeros(0, np.float32)),
        'x2r_present': int(x2r is not None),
        'x2r': (np.asarray(x2r, dtype=np.float32).copy()
                if x2r is not None else np.zeros(0, np.float32)),
        'cpe_present': int(cpe is not None),
        'cpe': (np.asarray(cpe, dtype=np.float32).copy()
                if cpe is not None else np.zeros(0, np.float32)),
        'cgm_present': int(cgm is not None),
        'cgm': (np.asarray(cgm, dtype=bool).astype(np.uint8).copy()
                if cgm is not None else np.zeros(0, np.uint8)),
        # capture_signal_saturation as the orchestrator set it before update.
        'cap_sat_pre': int(bool(state._capture_signal_saturation)),
    }
    # --- post-update query outputs ---
    iq = state.get_inst_linear_quality_estimate()
    row['out_usable'] = int(bool(state.usable_linear_estimate()))
    row['out_active'] = int(bool(state.active_render()))
    row['out_sat_echo'] = int(bool(state.saturated_echo()))
    row['out_min_delay'] = int(state.min_direct_path_filter_delay())
    row['out_fb_log2'] = float(state.fullband_erle_log2())
    row['out_iq_valid'] = int(iq is not None)
    row['out_iq'] = float(iq) if iq is not None else 0.0
    row['out_erle'] = np.asarray(state.erle(False), dtype=np.float32).copy()
    row['out_erle_unb'] = np.asarray(state.erle_unbounded(),
                                     dtype=np.float32).copy()
    row['out_erl'] = np.asarray(state.erl(), dtype=np.float32).copy()
    row['out_erl_td'] = float(state.erl_time_domain())
    assert len(row['render_psd']) == n_bins
    _ROWS.append(row)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/aec_state_golden.bin'

    mic, _ = sf.read(CASE_MIC, dtype='float32')
    lpb, _ = sf.read(CASE_LPB, dtype='float32')
    n = min(len(mic), len(lpb))
    mic = mic[:n]
    lpb = lpb[:n]

    cfg = AecConfig.from_preset('balanced', sample_rate=SR, filter_length=FL)

    np.random.seed(0)
    aec = AEC(cfg)

    # --- monkeypatch AecState.update + handle_echo_path_change to capture ---
    _orig_update = AecState.update
    _orig_hepc = AecState.handle_echo_path_change

    captured_cfg = {}
    last_state_id = {'id': None}

    def _maybe_emit_rebuild(self):
        # Detect a mid-stream AecState REBUILD (orchestrator _reset_aec3_post
        # creates a brand-new AecState — a full reset that, unlike
        # handle_echo_path_change/_full_reset, ALSO zeroes
        # filter_update_blocks_since_start). Emit a 'rebuild' control row so the
        # C side re-inits its AecState from scratch and stays in lockstep. Both
        # hepc and update see the new object; whichever runs first emits the
        # marker (hepc precedes update inside one _aec3_post).
        cur = id(self)
        if last_state_id['id'] is not None and cur != last_state_id['id']:
            _ROWS.append({'kind': 2})  # rebuild marker
        last_state_id['id'] = cur

    def _patched_update(self, **kwargs):
        _maybe_emit_rebuild(self)
        if not captured_cfg:
            captured_cfg['state'] = self
        _orig_update(self, **kwargs)   # run the real update first
        _capture_inputs_outputs(self, kwargs)  # then snapshot post-update state

    def _patched_hepc(self, variability):
        _maybe_emit_rebuild(self)
        # Capture as a control row, then run the real reset cascade.
        _ROWS.append({
            'kind': 1,
            'gain_change': int(bool(variability.gain_change)),
            'delay_change': int(variability.delay_change.value),
        })
        _orig_hepc(self, variability)

    AecState.update = _patched_update
    AecState.handle_echo_path_change = _patched_hepc

    try:
        pos = 0
        while pos + HOP <= n:
            aec.process(mic[pos:pos + HOP], lpb[pos:pos + HOP])
            pos += HOP

        # --- Segment 2: exercise handle_echo_path_change + reset cascade ---
        # The real DT case carries no EPV events, so inject them directly on the
        # live AecState, then run a few more update() frames so the C side
        # validates the post-reset evolution (startup re-gating, counter zeroing).
        state = aec._aec3_state
        # 2a. gain_change only -> ERLE-only reset.
        state.handle_echo_path_change(EchoPathVariability(
            gain_change=True, delay_change=DelayAdjustment.NONE,
            clock_drift=False))
        # A few real frames after the gain-change reset.
        for _ in range(50):
            if pos + HOP > n:
                break
            aec.process(mic[pos:pos + HOP], lpb[pos:pos + HOP])
            pos += HOP
        # 2b. delay_change -> FULL reset cascade.
        state.handle_echo_path_change(EchoPathVariability(
            gain_change=False, delay_change=DelayAdjustment.NEW_DETECTED_DELAY,
            clock_drift=False))
        for _ in range(120):
            if pos + HOP > n:
                break
            aec.process(mic[pos:pos + HOP], lpb[pos:pos + HOP])
            pos += HOP
    finally:
        AecState.update = _orig_update
        AecState.handle_echo_path_change = _orig_hepc

    state = aec._aec3_state
    sc = state._config
    n_bins = sc.n_bins
    fa = state._filter_analyzer
    enable_fa = int(fa is not None)

    # Sanity: the real balanced path feeds the non-None ERLE inputs.
    upd_rows = [r for r in _ROWS if r['kind'] == 0]
    # FilterAnalyzer sizes lazily from the first non-empty filter_taps_full.
    # The C side preallocates its HPF taps to this length.
    filter_taps_size = 0
    for r in upd_rows:
        if r['ft_present'] and len(r['ft']) > 0:
            filter_taps_size = int(len(r['ft']))
            break
    n_x2r = sum(r['x2r_present'] for r in upd_rows)
    n_cpe = sum(r['cpe_present'] for r in upd_rows)
    n_cgm = sum(r['cgm_present'] for r in upd_rows)
    n_conv = sum(r['bridge_filter_converged'] for r in upd_rows)
    n_usable = sum(r['out_usable'] for r in upd_rows)
    n_iq_valid = sum(r['out_iq_valid'] for r in upd_rows)
    n_hepc = sum(1 for r in _ROWS if r['kind'] == 1)
    n_rebuild = sum(1 for r in _ROWS if r['kind'] == 2)
    # first frame where bridge_filter_converged flips True (startup transition)
    first_conv = next((i for i, r in enumerate(upd_rows)
                       if r['bridge_filter_converged']), -1)
    first_usable = next((i for i, r in enumerate(upd_rows)
                         if r['out_usable']), -1)

    with open(out, 'wb') as f:
        _w_i32(f, n_bins, filter_taps_size, sc.num_capture_channels, sc.hop_size,
               enable_fa, sc.erle_startup_hops, sc.erl_startup_hops,
               int(sc.echo_can_saturate), int(sc.use_linear_filter),
               int(sc.conservative_initial_phase))
        _w_f64(f, sc.initial_state_seconds)
        _w_i32(f, sc.delay_headroom_samples)
        _w_f64(f, sc.erle_min, sc.erle_max_l, sc.erle_max_h)
        _w_i32(f, n_hepc, len(_ROWS))

        for r in _ROWS:
            _w_i32(f, r['kind'])
            if r['kind'] == 2:
                # rebuild marker — no payload (C re-inits from the header cfg)
                continue
            if r['kind'] == 1:
                _w_i32(f, r['gain_change'], r['delay_change'])
                continue
            # inputs
            _w_i32(f, r['bridge_filter_converged'], r['ext_reported'],
                   r['ext_quality'], r['ext_samples'])
            _w_f32arr(f, r['render_psd'])
            _w_f32arr(f, r['capture_psd'])
            _w_f32arr(f, r['error_psd'])
            _w_f32arr(f, r['echo_psd'])
            _w_i32(f, r['active_render'])
            _w_f64(f, r['s_ref'], r['s_coa'], r['epg'])
            _w_i32(f, r['rb_present'], len(r['rb']))
            if r['rb_present']:
                _w_f32arr(f, r['rb'])
            _w_i32(f, r['ft_present'], len(r['ft']))
            if r['ft_present']:
                _w_f32arr(f, r['ft'])
            _w_i32(f, r['x2r_present'])
            if r['x2r_present']:
                _w_f32arr(f, r['x2r'])
            _w_i32(f, r['cpe_present'])
            if r['cpe_present']:
                _w_f32arr(f, r['cpe'])
            _w_i32(f, r['cgm_present'])
            if r['cgm_present']:
                r['cgm'].astype(np.uint8).tofile(f)
            _w_i32(f, r['cap_sat_pre'])
            # outputs
            _w_i32(f, r['out_usable'], r['out_active'], r['out_sat_echo'],
                   r['out_min_delay'])
            _w_f64(f, r['out_fb_log2'])
            _w_i32(f, r['out_iq_valid'])
            _w_f64(f, r['out_iq'])
            _w_f32arr(f, r['out_erle'])
            _w_f32arr(f, r['out_erle_unb'])
            _w_f32arr(f, r['out_erl'])
            _w_f64(f, r['out_erl_td'])

    print(f"wrote {out}")
    print(f"  n_bins={n_bins} filter_taps_size={filter_taps_size} "
          f"enable_filter_analyzer={enable_fa} hop={sc.hop_size}")
    print(f"  total rows={len(_ROWS)}  update frames={len(upd_rows)}  "
          f"hepc events={n_hepc}  rebuilds={n_rebuild}")
    print(f"  x2_reverb non-None: {n_x2r}/{len(upd_rows)}  "
          f"capture_psd_erle non-None: {n_cpe}/{len(upd_rows)}  "
          f"coh_gate_mask non-None: {n_cgm}/{len(upd_rows)}")
    print(f"  bridge_filter_converged True: {n_conv}/{len(upd_rows)}  "
          f"(first at frame {first_conv})")
    print(f"  usable_linear True: {n_usable}/{len(upd_rows)}  "
          f"(first at frame {first_usable})")
    print(f"  inst_quality valid: {n_iq_valid}/{len(upd_rows)}")
    if n_x2r == 0 or n_cpe == 0 or n_cgm == 0:
        print("  WARNING: an ERLE input was None for all frames — check preset!")


if __name__ == '__main__':
    main()
