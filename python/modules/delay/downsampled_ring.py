"""Decimator (cascaded biquad LP + HP) + downsampled-render ring buffer.

Mirrors:
  - docs/aec3_extracts/src/aec3/decimator.{cc,h}
  - docs/aec3_extracts/src/aec3/downsampled_render_buffer.{cc,h}
  - utility/cascaded_biquad_filter.{cc,h}  (inlined here; trivial direct-form)

Rate / size invariants:
  - Decimator input block size = 64 samples (AEC3 ``kBlockSize``); output =
    block / down_sampling_factor (16 samples at factor=4).
  - The matched_filter pipeline is called with these 16-sample sub-blocks
    at AEC3's native 1 ms rate. Our 10 ms hop carries 160 samples = 10
    inner blocks of 64 samples = 10 sub-blocks after decimation. The
    edge-chunker lives in echo_path_delay_estimator.py.
  - Down-sampling factor 8 path (BandPass filter) is NOT ported — AEC3
    uses it only at sample rates we don't run (48 kHz). Add only if
    we ever support a non-16 kHz config.

Biquad form: Direct Form II transposed; coefficients pulled verbatim
from decimator.cc; state per-section.
"""
import numpy as np


# AEC3 kLowPassFilterDs4 — 6th-order elliptic LP at 1800 Hz (16 kHz / 8 = 2 kHz cutoff after decimation).
# decimator.cc lines 23-29. Coefficients order is {b0,b1,b2} and {a1,a2} (a0 normalised).
_LP_DS4_SECTIONS = (
    ((0.0180919877, 0.00320961363, 0.0180919877), (-1.5183195, 0.633165865)),
    ((1.0, -1.24550459, 1.0),                       (-1.49784254, 0.853586692)),
    ((1.0, -1.4221681, 1.0),                        (-1.49791282, 0.969572384)),
)

# AEC3 kHighPassFilter — 2nd-order Butterworth HP at 1000 Hz; reduces near-end noise.
# decimator.cc lines 47-52.
_HP_SECTIONS = (
    ((0.757076375, -1.51415275, 0.757076375), (-1.45424359, 0.574061915)),
)


class _CascadedBiquad:
    """Cascaded biquad filter (direct form II transposed, per-section state)."""

    def __init__(self, sections):
        self._b = np.array([list(b) for b, _ in sections], dtype=np.float64)
        self._a = np.array([list(a) for _, a in sections], dtype=np.float64)
        self._z = np.zeros((len(sections), 2), dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a 1-D float array in-place equivalent; returns filtered copy."""
        y = np.empty_like(x, dtype=np.float64)
        z = self._z
        b = self._b
        a = self._a
        for n in range(x.size):
            xn = float(x[n])
            yn = xn
            for s in range(b.shape[0]):
                # DF-II transposed:
                #   out = b0*x + z0
                #   z0  = b1*x - a1*out + z1
                #   z1  = b2*x - a2*out
                inp = yn
                out = b[s, 0] * inp + z[s, 0]
                z[s, 0] = b[s, 1] * inp - a[s, 0] * out + z[s, 1]
                z[s, 1] = b[s, 2] * inp - a[s, 1] * out
                yn = out
            y[n] = yn
        return y.astype(x.dtype, copy=False)

    def reset(self) -> None:
        self._z.fill(0.0)


class Decimator:
    """4x decimator with AEC3 anti-alias + noise-reduction biquad cascades.

    Input: 64-sample block (AEC3 kBlockSize).
    Output: 16-sample sub-block.
    """

    def __init__(self, down_sampling_factor: int = 4) -> None:
        if down_sampling_factor != 4:
            raise NotImplementedError("Only down_sampling_factor=4 is ported (16 kHz path).")
        self._factor = down_sampling_factor
        self._anti_alias = _CascadedBiquad(_LP_DS4_SECTIONS)
        self._noise_reduction = _CascadedBiquad(_HP_SECTIONS)

    def decimate(self, in_block: np.ndarray) -> np.ndarray:
        if in_block.size % self._factor != 0:
            raise ValueError(f"input size {in_block.size} not divisible by factor {self._factor}")
        x = self._anti_alias.process(in_block)
        x = self._noise_reduction.process(x)
        return x[:: self._factor].astype(np.float32, copy=False)

    def reset(self) -> None:
        self._anti_alias.reset()
        self._noise_reduction.reset()


class DownsampledRenderBuffer:
    """Circular buffer for downsampled render samples — forward-stored.

    Convention divergence from AEC3 ``downsampled_render_buffer.h`` (which
    stores reversed with ``write`` decrementing on push). We use the natural
    forward storage: samples written in temporal order at increasing
    indices, ``write`` points one past the newest sample. Conceptually
    equivalent; downstream ``MatchedFilter`` walks BACKWARD in index when
    iterating over filter taps so the access pattern still gives
    ``h[0]`` = newest, ``h[k]`` = k-samples-older.

    ``newest_index()`` returns the position of the most recently written
    sample. ``gather_back(start_offset, length)`` returns ``length`` samples
    going backward in time starting ``start_offset`` samples before
    ``newest_index()`` (returns a contiguous numpy array, copying across
    wraparound — cheap at the size we use).
    """

    def __init__(self, capacity: int) -> None:
        self.size = int(capacity)
        self.buffer = np.zeros(self.size, dtype=np.float32)
        self.write = 0

    def push_sub_block(self, sub_block: np.ndarray) -> None:
        n = sub_block.size
        if n > self.size:
            raise ValueError("sub_block larger than ring capacity")
        end = self.write + n
        if end <= self.size:
            self.buffer[self.write:end] = sub_block
        else:
            tail = self.size - self.write
            self.buffer[self.write:] = sub_block[:tail]
            self.buffer[: n - tail] = sub_block[tail:]
        self.write = (self.write + n) % self.size

    def newest_index(self) -> int:
        """Index of most recently written sample."""
        return (self.write - 1 + self.size) % self.size

    def gather_back(self, start_offset: int, length: int) -> np.ndarray:
        """Return ``length`` samples in time-reversed order starting at
        ``newest_index() - start_offset``. Index 0 of the returned array is
        the sample ``start_offset`` ago; index ``length-1`` is
        ``start_offset+length-1`` ago.

        Uses two-chunk circular copy so the returned array is contiguous.
        """
        if length > self.size:
            raise ValueError("length larger than ring capacity")
        end = (self.newest_index() - start_offset + self.size) % self.size
        # We want indices end, end-1, ..., end-length+1 (with wrap).
        # Equivalently: a slice [start, end+1) in REVERSED order, where
        # start = end - length + 1.
        start = (end - length + 1 + self.size) % self.size
        if start <= end:
            return self.buffer[start : end + 1][::-1].copy()
        # Wrapped case: two slices [start, size) ++ [0, end+1).
        first = self.buffer[start:]
        second = self.buffer[: end + 1]
        return np.concatenate((first, second))[::-1].copy()

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.write = 0


def required_capacity(
    num_filters: int,
    window_size_sub_blocks: int,
    alignment_shift_sub_blocks: int,
    sub_block_size: int,
) -> int:
    """Minimum ring capacity to cover the matched_filter span + safety margin.

    Mirrors GetDownSampledBufferSize(down_sampling_factor, filter_length_blocks)
    in spirit; our matched_filter geometry is direct so compute it explicitly.
    Capacity must hold the deepest filter's window AND alignment shifts of
    all preceding filters. +1 sub_block of headroom for the write cursor.
    """
    span_sub_blocks = window_size_sub_blocks + alignment_shift_sub_blocks * (num_filters - 1)
    return (span_sub_blocks + 1) * sub_block_size
