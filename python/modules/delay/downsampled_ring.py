"""Decimator (biquad low-pass + high-pass cascade) and downsampled render ring buffer.

The decimator maps a 64-sample block to a 16-sample sub-block (factor 4 only, 16 kHz path).
Biquads are direct form II transposed with per-section state.
"""
import numpy as np


# 6th-order elliptic low-pass for factor-4 decimation; per section {b0,b1,b2},{a1,a2} with a0 normalised.
_LP_DS4_SECTIONS = (
    ((0.0180919877, 0.00320961363, 0.0180919877), (-1.5183195, 0.633165865)),
    ((1.0, -1.24550459, 1.0),                       (-1.49784254, 0.853586692)),
    ((1.0, -1.4221681, 1.0),                        (-1.49791282, 0.969572384)),
)

# 2nd-order Butterworth high-pass at 1 kHz (near-end noise reduction).
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
    """Circular buffer of downsampled render samples, stored in forward time order.

    ``write`` points one past the newest sample; ``newest_index()`` is the newest
    sample. ``gather_back(start_offset, length)`` returns a contiguous copy of
    ``length`` samples going backward in time from ``start_offset`` samples before
    the newest.
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
        """Return ``length`` samples in newest-first order starting ``start_offset``
        samples before the newest; index 0 is the sample ``start_offset`` ago.
        """
        if length > self.size:
            raise ValueError("length larger than ring capacity")
        end = (self.newest_index() - start_offset + self.size) % self.size
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
    """Minimum ring capacity covering the deepest filter window plus the alignment
    shifts of the preceding filters, plus one sub-block of headroom."""
    span_sub_blocks = window_size_sub_blocks + alignment_shift_sub_blocks * (num_filters - 1)
    return (span_sub_blocks + 1) * sub_block_size
