"""Frequency <-> FFT bin index conversion utilities.

Central place to convert frequency-canonical AEC config values to FFT
bin indices. Use this instead of hardcoded bin numbers so the codebase
stays correct when fft_size / sample_rate changes.

Context: the AEC3 reference (WebRTC) uses kFftLength=128 (125 Hz/bin).
Our standard config uses fft_size=512 (31.25 Hz/bin, 4x finer). When
config constants are expressed as bin indices, any port from AEC3
silently lands at the wrong frequency. Express constants in Hz and
convert here.
"""


def hz_to_bin(hz: float, n_bins: int, sr: int = 16000) -> int:
    """Convert frequency in Hz to nearest FFT bin index.

    n_bins is the spectrum size (= fft_size // 2 + 1). For the standard
    fft_size=512 at sr=16000 this is 257 bins, 31.25 Hz per bin.
    """
    fft_size = (n_bins - 1) * 2
    return int(round(hz * fft_size / sr))


def bin_to_hz(bin_idx: int, n_bins: int, sr: int = 16000) -> float:
    """Convert FFT bin index to its center frequency in Hz."""
    fft_size = (n_bins - 1) * 2
    return bin_idx * sr / fft_size
