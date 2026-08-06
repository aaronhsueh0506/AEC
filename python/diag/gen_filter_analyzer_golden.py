"""Generate a binary golden for the C filter_analyzer port (WS5 5.2).

Runs the Python FilterAnalyzer over a deterministic float32 input sequence and
writes, for each per-hop step, the inputs (filter_taps + render_block) and the
full observable state AFTER the update (the high-pass-filtered taps, peak_index,
delay_blocks, gain, consistent flag, consistent-detector internal accumulators),
to a raw little-endian file replayed by c_impl/test/historical/parity_filter_analyzer.c.

This is a STATEFUL module: the region cursor sweeps the full filter incrementally
and the ConsistentFilterDetector accumulates floor/secondary peaks across one
sweep, so the golden drives a MULTI-FRAME sequence (>= 2 full sweeps over the
960-tap filter, plus a reset) and snapshots the entire state per frame.

REAL input dtypes (captured by monkeypatching FilterAnalyzer.update on the
balanced 16 kHz / fl=832 pipeline,
wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_*.wav):
    filter_taps  : float32, shape (960,)   (= get_time_domain_filter())
    render_block : float32, shape (160,)    (= render_block_scaled, int16 amp)
    _h_highpass  : float32
    _gain        : python float ; _peak_index / _delay_blocks : python int

Layout (LE):
  int32 size, int32 hop, int32 n_steps
  n_steps x [
    int32  reset_before            (1 == fa.reset() was called before update)
    filter_taps[size]   f32
    render_block[hop]   f32
    h_highpass[size]    f32        (state after update)
    int32  peak_index
    int32  delay_blocks
    int32  consistent_estimate     (0/1)
    int32  min_filter_delay_blocks
    int32  any_filter_consistent   (0/1)
    float64 gain
    float64 max_echo_path_gain
    -- ConsistentFilterDetector internals --
    int32   cd_significant_peak    (0/1)
    float64 cd_floor_accum
    float64 cd_secondary_peak
    int32   cd_floor_low_limit
    int32   cd_floor_high_limit
    int64   cd_counter
    int32   cd_delay_ref
  ]

Run: python3 python/diag/gen_filter_analyzer_golden.py /tmp/fa_golden.bin
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.state.filter_analyzer import FilterAnalyzer  # noqa: E402

SIZE = 960   # 6 partitions * 160 (real fl=832 / hop=160 layout)
HOP = 160


def make_taps(rng, peak_pos, peak_amp, floor_scale):
    """Build a float32 impulse response with a dominant peak.

    A localized peak (so the analyzer's significance test can fire) over a low
    random floor; values span both signs like a real adaptive filter.
    """
    taps = ((rng.rand(SIZE).astype(np.float32) * 2.0 - 1.0)
            * floor_scale).astype(np.float32)
    if 0 <= peak_pos < SIZE:
        taps[peak_pos] = np.float32(peak_amp)
        # a couple of decaying side lobes so HPF + neighbour structure is real
        if peak_pos + 1 < SIZE:
            taps[peak_pos + 1] = np.float32(peak_amp * 0.4)
        if peak_pos - 1 >= 0:
            taps[peak_pos - 1] = np.float32(peak_amp * -0.3)
    return taps.astype(np.float32)


def make_block(rng, amp):
    # render_block_scaled is int16-amplitude in the real pipeline; emulate large
    # amplitudes so the active-render power test (> limit^2 * hop) can fire.
    return ((rng.rand(HOP).astype(np.float32) * 2.0 - 1.0) * amp).astype(np.float32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/fa_golden.bin'
    rng = np.random.RandomState(20)

    fa = FilterAnalyzer()  # defaults: active_render_limit=100, bounded_erl=False

    steps = []  # each is (filter_taps, render_block)

    # The region sweeps HOP=160 taps per hop -> SIZE/HOP = 6 hops per full sweep.
    # Drive >2 full sweeps with a stable dominant peak at tap 200 and a loud
    # active render block so the consistent detector accumulates and the counter
    # eventually crosses (we keep the same delay_ref so counter persists).
    HOPS_PER_SWEEP = -(-SIZE // HOP)  # ceil = 6

    # Phase A: 3 full sweeps, persistent peak @200, loud render (active true).
    for _ in range(3 * HOPS_PER_SWEEP):
        taps = make_taps(rng, peak_pos=200, peak_amp=1000.0, floor_scale=1.0)
        blk = make_block(rng, amp=2000.0)   # power ~ (2000^2/3)*160 >> 1.6e6 thr
        steps.append((taps, blk))

    # Phase B: peak moves to 520 (different delay_blocks) for 2 sweeps -> the
    # detector resets its counter when delay_ref changes.
    for _ in range(2 * HOPS_PER_SWEEP):
        taps = make_taps(rng, peak_pos=520, peak_amp=800.0, floor_scale=2.0)
        blk = make_block(rng, amp=50.0)     # quiet -> active false even if sig.
        steps.append((taps, blk))

    # Phase C: a single mid-sweep reset() (exercise reset path) then resume with
    # a fresh peak @50 (edge-ish) for a couple of sweeps.
    steps.append(('RESET', None))
    for _ in range(2 * HOPS_PER_SWEEP):
        taps = make_taps(rng, peak_pos=50, peak_amp=1500.0, floor_scale=0.5)
        blk = make_block(rng, amp=3000.0)
        steps.append((taps, blk))

    # Phase D: near-zero filter (peak insignificant vs floor) so significance
    # gate stays false; verify gain/state still tracks.
    for _ in range(HOPS_PER_SWEEP):
        taps = (rng.rand(SIZE).astype(np.float32) * 0.001).astype(np.float32)
        blk = make_block(rng, amp=10.0)
        steps.append((taps, blk))

    # Phase E: hold a stable strong peak @300 with a loud active render for
    # well over CONSISTENT_HOLD_HOPS (150) frames so the consistent counter
    # crosses the hold threshold and any_filter_consistent() flips True. This
    # exercises the True branch of the consistency verdict + the post-5 s
    # convergence gain assignment (blocks_since_reset > 500 is NOT reached here,
    # but the consistent-gain branch coverage is via cd_counter > 150).
    for _ in range(170):
        taps = make_taps(rng, peak_pos=300, peak_amp=2000.0, floor_scale=0.3)
        blk = make_block(rng, amp=4000.0)
        steps.append((taps, blk))

    records = []
    pending_reset = 0
    for entry in steps:
        if isinstance(entry[0], str) and entry[0] == 'RESET':
            fa.reset()
            pending_reset = 1
            continue
        taps, blk = entry
        fa.update(taps, blk)

        cd = fa._consistent
        rec = {
            'reset_before': pending_reset,
            'taps': np.array(taps, dtype=np.float32),  # copy
            'blk': np.array(blk, dtype=np.float32),    # copy
            # _h_highpass is mutated in place each frame: snapshot a copy.
            'h': np.array(fa._h_highpass, dtype=np.float32),
            'peak_index': int(fa._peak_index),
            'delay_blocks': int(fa._delay_blocks),
            'consistent_estimate': 1 if fa._consistent_estimate else 0,
            'min_delay': int(fa.min_filter_delay_blocks()),
            'any_consistent': 1 if fa.any_filter_consistent() else 0,
            'gain': float(fa._gain),
            'max_epg': float(fa.max_echo_path_gain()),
            'cd_sig': 1 if cd._significant_peak else 0,
            'cd_floor_accum': float(cd._floor_accum),
            'cd_secondary_peak': float(cd._secondary_peak),
            'cd_floor_low': int(cd._floor_low_limit),
            'cd_floor_high': int(cd._floor_high_limit),
            'cd_counter': int(cd._counter),
            'cd_delay_ref': int(cd._delay_ref),
        }
        records.append(rec)
        pending_reset = 0

    n_steps = len(records)
    with open(out, 'wb') as f:
        f.write(struct.pack('<iii', SIZE, HOP, n_steps))
        for r in records:
            f.write(struct.pack('<i', r['reset_before']))
            r['taps'].tofile(f)
            r['blk'].tofile(f)
            r['h'].tofile(f)
            f.write(struct.pack('<iiiii',
                                r['peak_index'], r['delay_blocks'],
                                r['consistent_estimate'], r['min_delay'],
                                r['any_consistent']))
            f.write(struct.pack('<dd', r['gain'], r['max_epg']))
            f.write(struct.pack('<i', r['cd_sig']))
            f.write(struct.pack('<dd', r['cd_floor_accum'], r['cd_secondary_peak']))
            f.write(struct.pack('<ii', r['cd_floor_low'], r['cd_floor_high']))
            f.write(struct.pack('<q', r['cd_counter']))
            f.write(struct.pack('<i', r['cd_delay_ref']))

    print(f"wrote {out}  (size {SIZE}, hop {HOP}, {n_steps} steps)")


if __name__ == '__main__':
    main()
