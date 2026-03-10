#!/usr/bin/env python3
"""
gen_sim_data.py - Generate simulated AEC test data

Produces two sets of test files with known echo paths:
  fileid_1: Speech-like signals (chirp + harmonics)
  fileid_2: White noise far-end

Each set has 4 files:
  farend_speech_fileid_{N}.wav   — far-end (loudspeaker) signal
  nearend_speech_fileid_{N}.wav  — clean near-end speech
  echo_fileid_{N}.wav            — echo = farend * RIR
  nearend_mic_fileid_{N}.wav     — mic = echo + nearend_speech (with DT schedule)

Usage:
  python3 gen_sim_data.py [output_dir]
  python3 gen_sim_data.py ../wav/
"""

import sys
import os
import numpy as np
import soundfile as sf

SR = 16000
DURATION = 10.0  # seconds
N_SAMPLES = int(SR * DURATION)


def make_rir(delay=200, gain=0.8, n_taps=512):
    """Create a simple room impulse response.

    Main tap at `delay` with a few reflections.
    """
    rir = np.zeros(n_taps, dtype=np.float32)
    rir[delay] = gain
    # A few reflections (weaker, later)
    if delay + 50 < n_taps:
        rir[delay + 50] = gain * 0.3
    if delay + 120 < n_taps:
        rir[delay + 120] = -gain * 0.15
    if delay + 200 < n_taps:
        rir[delay + 200] = gain * 0.08
    return rir


def make_speech_signal(duration_s, sr, seed=42):
    """Generate a speech-like signal using chirps and harmonics with pauses."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)

    # Several "utterances" with pauses
    utterances = [
        (0.5, 3.5),   # speak
        (4.5, 7.0),   # speak
        (7.5, 9.5),   # speak
    ]

    for start, end in utterances:
        s = int(start * sr)
        e = min(int(end * sr), n)
        seg_t = t[s:e] - start

        # Fundamental + harmonics
        f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * seg_t)  # varying pitch
        phase = np.cumsum(f0) / sr
        seg = np.zeros(e - s, dtype=np.float32)
        for harmonic in [1, 2, 3, 4]:
            amp = 1.0 / harmonic
            seg += amp * np.sin(2 * np.pi * harmonic * phase).astype(np.float32)

        # Apply envelope (attack/decay)
        env_len = min(int(0.05 * sr), len(seg) // 4)
        env = np.ones(len(seg), dtype=np.float32)
        env[:env_len] = np.linspace(0, 1, env_len)
        env[-env_len:] = np.linspace(1, 0, env_len)
        seg *= env

        signal[s:e] = seg

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.5
    return signal


def make_different_speech(duration_s, sr, seed=99):
    """Generate a different speech-like signal for near-end."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float32)

    # Different timing (partially overlapping with far-end = double-talk)
    utterances = [
        (2.0, 4.0),   # overlap with far-end utterance 1 → double-talk
        (5.0, 6.5),   # overlap with far-end utterance 2 → double-talk
        (8.5, 10.0),  # overlap with far-end utterance 3 → double-talk
    ]

    for start, end in utterances:
        s = int(start * sr)
        e = min(int(end * sr), n)
        seg_t = t[s:e] - start

        # Higher pitch, different timbre
        f0 = 250 + 30 * np.sin(2 * np.pi * 0.8 * seg_t)
        phase = np.cumsum(f0) / sr
        seg = np.zeros(e - s, dtype=np.float32)
        for harmonic in [1, 2, 3]:
            amp = 1.0 / (harmonic ** 1.5)
            seg += amp * np.sin(2 * np.pi * harmonic * phase).astype(np.float32)

        env_len = min(int(0.05 * sr), len(seg) // 4)
        env = np.ones(len(seg), dtype=np.float32)
        env[:env_len] = np.linspace(0, 1, env_len)
        env[-env_len:] = np.linspace(1, 0, env_len)
        seg *= env

        signal[s:e] = seg

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.3  # Slightly quieter than far-end
    return signal


def make_white_noise(duration_s, sr, seed=123):
    """Generate white noise signal with on/off pattern."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sr)
    noise = rng.randn(n).astype(np.float32)

    # Apply same utterance schedule as speech for consistency
    mask = np.zeros(n, dtype=np.float32)
    utterances = [
        (0.5, 3.5),
        (4.5, 7.0),
        (7.5, 9.5),
    ]
    for start, end in utterances:
        s = int(start * sr)
        e = min(int(end * sr), n)
        env_len = min(int(0.02 * sr), (e - s) // 4)
        env = np.ones(e - s, dtype=np.float32)
        env[:env_len] = np.linspace(0, 1, env_len)
        env[-env_len:] = np.linspace(1, 0, env_len)
        mask[s:e] = env

    noise *= mask
    # Normalize
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * 0.5
    return noise


def generate_set(fileid, farend, nearend_speech, rir, output_dir):
    """Generate a complete set of 4 AEC test files."""
    # Echo = farend convolved with RIR
    echo = np.convolve(farend, rir)[:len(farend)].astype(np.float32)

    # Mic = echo + nearend_speech
    nearend_mic = (echo + nearend_speech).astype(np.float32)

    # Clip prevention
    max_val = np.max(np.abs(nearend_mic))
    if max_val > 0.95:
        scale = 0.9 / max_val
        nearend_mic *= scale
        echo *= scale
        farend = farend * scale
        nearend_speech = nearend_speech * scale

    # Write files
    sf.write(os.path.join(output_dir, f'farend_speech_fileid_{fileid}.wav'),
             farend, SR, subtype='PCM_16')
    sf.write(os.path.join(output_dir, f'nearend_speech_fileid_{fileid}.wav'),
             nearend_speech, SR, subtype='PCM_16')
    sf.write(os.path.join(output_dir, f'echo_fileid_{fileid}.wav'),
             echo, SR, subtype='PCM_16')
    sf.write(os.path.join(output_dir, f'nearend_mic_fileid_{fileid}.wav'),
             nearend_mic, SR, subtype='PCM_16')

    # Print info
    echo_energy = np.mean(echo[echo != 0] ** 2) if np.any(echo != 0) else 0
    speech_energy = np.mean(nearend_speech[nearend_speech != 0] ** 2) if np.any(nearend_speech != 0) else 0
    ser = 10 * np.log10(speech_energy / (echo_energy + 1e-10)) if echo_energy > 0 else float('inf')
    print(f"  fileid_{fileid}: delay={np.argmax(rir)} samples "
          f"({np.argmax(rir)/SR*1000:.1f}ms), "
          f"gain={rir[np.argmax(rir)]:.2f}, SER={ser:.1f}dB")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wav')

    os.makedirs(output_dir, exist_ok=True)

    rir = make_rir(delay=200, gain=0.8, n_taps=512)
    nearend_speech = make_different_speech(DURATION, SR)

    # Set 1: Speech-like far-end
    print("Generating fileid_1 (speech far-end)...")
    farend_speech = make_speech_signal(DURATION, SR)
    generate_set(1, farend_speech, nearend_speech, rir, output_dir)

    # Set 2: White noise far-end
    print("Generating fileid_2 (white noise far-end)...")
    farend_noise = make_white_noise(DURATION, SR)
    generate_set(2, farend_noise, nearend_speech, rir, output_dir)

    print(f"\nFiles written to: {output_dir}")
    print(f"RIR: {len(rir)} taps, main tap at {np.argmax(rir)}")


if __name__ == '__main__':
    main()
