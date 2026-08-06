"""Generate a binary golden for the C render_signal_analyzer port (WS5 5.1).

Runs the Python RenderSignalAnalyzer over a deterministic float32 input
sequence and writes, for each per-hop step, the inputs + the full observable
state (counters, narrow_peak_band, poor_signal_excitation) and the masked-mu
output, to a raw little-endian file replayed by
c_impl/test/historical/parity_render_signal_analyzer.c.

REAL input dtypes (captured from the balanced 16 kHz / fl=832 pipeline on
wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_*.wav):
    render_psd   : float32, shape (n_freqs,)   (= |far_spec|² .astype(f32))
    render_block : float32, shape (hop=160,)   (= far_end hop)
    mu           : float32, shape (n_freqs,)   (mask input, ones)

The input sequence deliberately drives narrow peaks past the counter > 5 mask
threshold and the counter > 10 poor-excitation threshold, exercises the strong
narrow-band detector (peak_level > 100 × non_peak AND max_abs > 100/32768), and
includes render_psd=None / render_block=None steps (counter reset / skip-strong)
and edge-bin peaks (bin 1 and bin n_freqs-2).

Layout (LE):
  int32 n_freqs, int32 hop, int32 n_steps, int32 n_counters
  n_steps × [
    int32  has_psd, int32 has_block
    render_psd[n_freqs] f32   (present only if has_psd)
    render_block[hop]   f32   (present only if has_block)
    counters[n_counters] int64                 (expected state after update)
    int32  narrow_peak_band                    (-1 == None)
    int32  poor_signal_excitation              (0/1)
    masked_mu[n_freqs] f32                      (after mask on a ones() mu)
  ]

Run: python3 python/diag/gen_render_signal_analyzer_golden.py /tmp/rsa_golden.bin
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.render.render_signal_analyzer import RenderSignalAnalyzer  # noqa: E402

N_FREQS = 257
HOP = 160
STRONG_FREEZE = 6


def make_psd(rng, peak_bins, peak_mul, base_scale, block_scale):
    """Build a float32 render_psd with optional strong narrow peaks.

    peak_bins : list of bin indices to spike.
    peak_mul  : how many × the neighbour floor each peak rises to (>3 → narrow).
    """
    psd = (rng.rand(N_FREQS).astype(np.float32) * base_scale).astype(np.float32)
    # ensure a smooth-ish floor so peaks stand out against immediate neighbours
    for b in peak_bins:
        if 0 <= b < N_FREQS:
            neigh = max(float(psd[b - 1]) if b - 1 >= 0 else 0.0,
                        float(psd[b + 1]) if b + 1 < N_FREQS else 0.0)
            psd[b] = np.float32(neigh * peak_mul + base_scale * 0.01)
    return psd.astype(np.float32)


def make_block(rng, amp):
    return ((rng.rand(HOP).astype(np.float32) * 2.0 - 1.0) * amp).astype(np.float32)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rsa_golden.bin'
    rng = np.random.RandomState(13)

    rsa = RenderSignalAnalyzer(n_freqs=N_FREQS,
                               strong_peak_freeze_duration=STRONG_FREEZE)
    n_counters = rsa._counters.size

    # Build a deterministic step list: each entry is (render_psd, render_block).
    steps = []

    # 1) 16 steps with a persistent strong narrow peak at bin 40 → counter climbs
    #    past 5 (mask) and 10 (poor_excitation). Strong amplitude block so the
    #    strong-peak detector also fires.
    for _ in range(16):
        psd = make_psd(rng, [40], peak_mul=200.0, base_scale=500.0,
                       block_scale=500.0)
        blk = make_block(rng, amp=0.5)   # max_abs ~0.5 > 100/32768
        steps.append((psd, blk))

    # 2) render_psd=None (delay lost) → counters reset; block present.
    steps.append((None, make_block(rng, amp=0.5)))

    # 3) two peaks: one at edge bin 1, one at interior bin 128; rebuild counters.
    for _ in range(8):
        psd = make_psd(rng, [1, 128], peak_mul=50.0, base_scale=300.0,
                       block_scale=300.0)
        blk = make_block(rng, amp=0.2)
        steps.append((psd, blk))

    # 4) peak at edge bin n_freqs-2 (=255) for several steps.
    for _ in range(8):
        psd = make_psd(rng, [N_FREQS - 2], peak_mul=80.0, base_scale=200.0,
                       block_scale=200.0)
        blk = make_block(rng, amp=0.001)   # max_abs tiny → strong-peak gate off
        steps.append((psd, blk))

    # 5) render_block=None (strong-peak skipped) but psd present.
    steps.append((make_psd(rng, [64], peak_mul=10.0, base_scale=100.0,
                           block_scale=0.0), None))

    # 6) flat-ish psd (no narrow peaks) → counters decay to 0.
    for _ in range(4):
        psd = (rng.rand(N_FREQS).astype(np.float32) * 100.0).astype(np.float32)
        blk = make_block(rng, amp=0.05)
        steps.append((psd, blk))

    # 7) strong peak revives narrow_peak_band, then let freeze tick down with
    #    flat psd (band stays until counter > strong_freeze).
    steps.append((make_psd(rng, [90], peak_mul=300.0, base_scale=400.0,
                           block_scale=400.0), make_block(rng, amp=0.8)))
    for _ in range(STRONG_FREEZE + 2):
        psd = (rng.rand(N_FREQS).astype(np.float32) * 10.0).astype(np.float32)
        blk = make_block(rng, amp=0.4)
        steps.append((psd, blk))

    # 8) both None.
    steps.append((None, None))

    n_steps = len(steps)
    with open(out, 'wb') as f:
        np.array([N_FREQS, HOP, n_steps, n_counters], dtype=np.int32).tofile(f)
        for psd, blk in steps:
            has_psd = 0 if psd is None else 1
            has_block = 0 if blk is None else 1
            np.array([has_psd, has_block], dtype=np.int32).tofile(f)
            if has_psd:
                np.asarray(psd, dtype=np.float32).tofile(f)
            if has_block:
                np.asarray(blk, dtype=np.float32).tofile(f)

            rsa.update(psd, blk)

            # expected observable state after the update
            rsa._counters.astype(np.int64).tofile(f)
            npb = rsa.narrow_peak_band()
            np.array([-1 if npb is None else int(npb)], dtype=np.int32).tofile(f)
            np.array([1 if rsa.poor_signal_excitation() else 0],
                     dtype=np.int32).tofile(f)
            mu = np.ones(N_FREQS, dtype=np.float32)
            rsa.mask_regions_around_narrow_bands(mu)
            mu.astype(np.float32).tofile(f)

    print(f"wrote {out}  ({N_FREQS} freqs, {n_steps} steps, "
          f"{n_counters} counters, hop {HOP})")


if __name__ == '__main__':
    main()
