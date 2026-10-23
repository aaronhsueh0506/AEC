"""Frequency <-> FFT bin index conversion.

Express frequency constants in Hz and convert here so they stay correct when
fft_size or sample_rate changes.
"""


def hz_to_bin(hz: float, n_bins: int, sr: int = 16000) -> int:
    """Frequency in Hz -> nearest FFT bin index; n_bins = fft_size // 2 + 1."""
    fft_size = (n_bins - 1) * 2
    return int(round(hz * fft_size / sr))


def bin_to_hz(bin_idx: int, n_bins: int, sr: int = 16000) -> float:
    """Convert FFT bin index to its center frequency in Hz."""
    fft_size = (n_bins - 1) * 2
    return bin_idx * sr / fft_size
