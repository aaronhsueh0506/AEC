"""Shared signal-grid resolver (``modules.config.resolve_signal_grid``).

The frame geometry is derived from ONE table by ONE resolver, and
``__post_init__`` is its only caller; the C port carries its own copy in
``aec_resolve_tuned``/``aec_derive_dims``, kept in lockstep by the C-parity
row. Asserted:
  - frame/hop/n_freqs on all four supported grids (three product grids plus
    8 kHz), hard-coded rather than recomputed from the source's own ``// 2``;
    ``is_legacy`` is set for 8 kHz only;
  - the table matches ``AEC_GRID_TABLE`` in c_impl/src/aec.c row for row, read
    out of the C source;
  - an unspecified / zero / negative fft_size RAISES instead of being filled in
    (16 kHz has two production grids), and every mismatched pair raises;
  - ``default_fft_size()`` (what the ``frame_size = -1`` sentinel expands to)
    agrees with the first table row per rate and is immediately resolvable;
  - ``AEC()`` rejects a config whose grid was poked back into an inconsistent
    state after construction.

Run:
    python3 -m pytest python/tests/test_signal_grid.py
"""
from __future__ import annotations

import os
import re
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aec import AEC, AecConfig
from modules.config import (
    SignalGrid, _GRID_TABLE, default_fft_size, resolve_signal_grid,
)


_C_AEC_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'c_impl', 'src', 'aec.c')

# The four supported grids, spelled out rather than recomputed from the
# formula the source uses — a recomputed expectation would move together
# with a broken formula.
_EXPECTED = (
    #  sr,  fft, frame,  hop, n_freqs, legacy
    (16000,  256,  256,  128,     129, False),
    (16000,  512,  512,  256,     257, False),
    (48000, 1024, 1024,  512,     513, False),
    ( 8000,  256,  256,  128,     129, True),
)


class SignalGridResolverTests(unittest.TestCase):

    def test_resolver_content(self) -> None:
        for sr, fft, frame, hop, k, legacy in _EXPECTED:
            with self.subTest(sr=sr, fft=fft):
                g = resolve_signal_grid(sr, fft)
                self.assertEqual(
                    g, SignalGrid(sr, fft, frame, hop, k, legacy))

    def test_only_8k_is_flagged_legacy(self) -> None:
        legacy = {(g.sample_rate, g.fft_size)
                  for g in _GRID_TABLE if g.is_legacy}
        self.assertEqual(legacy, {(8000, 256)})

    def test_table_matches_the_c_grid_table_row_for_row(self) -> None:
        """AEC_GRID_TABLE in c_impl/src/aec.c must be the same four rows.

        Read out of the C source instead of restated here, so an edit to
        either side that is not mirrored on the other fails this test.
        """
        src = open(_C_AEC_SRC, encoding='utf-8').read()
        m = re.search(
            r'static const AecSignalGrid AEC_GRID_TABLE\[\]\s*=\s*\{(.*?)\n\};',
            src, re.S)
        self.assertIsNotNone(m, "AEC_GRID_TABLE not found in c_impl/src/aec.c")
        rows = re.findall(
            r'\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
            r'\s*(\d+)\s*\}', m.group(1))
        c_rows = [SignalGrid(int(a), int(b), int(c), int(d), int(e), bool(int(f)))
                  for a, b, c, d, e, f in rows]
        self.assertEqual(c_rows, list(_GRID_TABLE))

    def test_unspecified_fft_size_raises_instead_of_guessing(self) -> None:
        for sr in (8000, 16000, 48000):
            for bad in (0, -1, -256, None):
                with self.subTest(sr=sr, fft=bad):
                    with self.assertRaises(ValueError):
                        resolve_signal_grid(sr, bad)

    def test_mismatched_pairs_raise(self) -> None:
        bad_pairs = [(16000, 1024), (48000, 256), (48000, 512), (8000, 512),
                     (8000, 1024), (16000, 128), (44100, 1024), (0, 256)]
        for sr, fft in bad_pairs:
            with self.subTest(sr=sr, fft=fft):
                with self.assertRaises(ValueError):
                    resolve_signal_grid(sr, fft)

    def test_non_power_of_two_raises(self) -> None:
        for fft in (255, 300, 1000):
            with self.subTest(fft=fft):
                with self.assertRaises(ValueError):
                    resolve_signal_grid(16000, fft)

    def test_explicit_mismatched_hop_raises(self) -> None:
        resolve_signal_grid(16000, 256, hop_size=128)      # the true hop
        for bad_hop in (64, 129, 256):
            with self.subTest(hop=bad_hop):
                with self.assertRaises(ValueError):
                    resolve_signal_grid(16000, 256, hop_size=bad_hop)

    def test_default_fft_size_is_the_only_place_a_grid_is_chosen(self) -> None:
        for sr, expected in ((8000, 256), (16000, 256), (48000, 1024)):
            with self.subTest(sr=sr):
                self.assertEqual(default_fft_size(sr), expected)
                # Whatever it returns must resolve immediately.
                self.assertEqual(
                    resolve_signal_grid(sr, default_fft_size(sr)).fft_size,
                    expected)
        self.assertEqual(default_fft_size(44100), 0)


class AecConfigGridTests(unittest.TestCase):
    """__post_init__ is the sole caller of the resolver."""

    def test_sentinel_expands_to_the_product_default(self) -> None:
        for sr, fft, hop in ((8000, 256, 128), (16000, 256, 128),
                             (48000, 1024, 512)):
            with self.subTest(sr=sr):
                cfg = AecConfig(sample_rate=sr)
                self.assertEqual((cfg.frame_size, cfg.hop_size, cfg.fft_size),
                                 (fft, hop, fft))

    def test_explicit_alternate_grid_is_honoured(self) -> None:
        cfg = AecConfig(sample_rate=16000, frame_size=512)
        self.assertEqual((cfg.frame_size, cfg.hop_size), (512, 256))

    def test_unsupported_grid_raises_at_construction(self) -> None:
        for sr, frame in ((16000, 1024), (48000, 256), (44100, 1024),
                          (8000, 512)):
            with self.subTest(sr=sr, frame=frame):
                with self.assertRaises(ValueError):
                    AecConfig(sample_rate=sr, frame_size=frame)

    def test_explicit_inconsistent_hop_raises(self) -> None:
        with self.assertRaises(ValueError):
            AecConfig(sample_rate=16000, frame_size=256, hop_size=64)

    def test_n_freqs_matches_the_resolved_grid(self) -> None:
        for sr, fft, k in ((16000, 256, 129), (16000, 512, 257),
                           (48000, 1024, 513), (8000, 256, 129)):
            with self.subTest(sr=sr, fft=fft):
                cfg = AecConfig(sample_rate=sr, frame_size=fft)
                self.assertEqual(
                    resolve_signal_grid(sr, cfg.fft_size).n_freqs, k)


class AecCoreDemandsResolvedGridTests(unittest.TestCase):
    """The core asserts the config is resolved; it does not re-derive."""

    def test_core_rejects_a_config_poked_back_to_a_sentinel(self) -> None:
        cfg = AecConfig()
        cfg.frame_size = -1
        with self.assertRaises(ValueError):
            AEC(cfg)

    def test_core_rejects_an_inconsistent_hop(self) -> None:
        cfg = AecConfig()
        cfg.hop_size = 64          # frame_size is still 256
        with self.assertRaises(ValueError):
            AEC(cfg)

    def test_core_accepts_every_supported_grid(self) -> None:
        """The guard is not blanket: one hop runs on each grid."""
        for sr, fft in ((16000, 256), (16000, 512), (48000, 1024),
                        (8000, 256)):
            with self.subTest(sr=sr, fft=fft):
                cfg = AecConfig(sample_rate=sr, frame_size=fft)
                aec = AEC(cfg)
                hop = cfg.hop_size
                out = aec.process(np.zeros(hop, dtype=np.float32),
                                  np.zeros(hop, dtype=np.float32))
                self.assertEqual(len(out), hop)


if __name__ == '__main__':
    unittest.main()
