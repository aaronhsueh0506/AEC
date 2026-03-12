"""
Generate two sets of AEC test signals (16 kHz, mono, float32 WAV).

Set 1 — Far-end single talk:
    ref1.wav : white noise (far-end playback)
    mic1.wav : echo only = ref convolved with a simple room impulse response

Set 2 — Double talk:
    ref2.wav : white noise (far-end playback)
    mic2.wav : echo + near-end speech (400 Hz sine tone, fades in at 1 s)

Echo path: a decaying exponential impulse response (~60 ms, 960 taps).
"""

import numpy as np
import soundfile as sf
import os

# ── parameters ──────────────────────────────────────────────────────
FS = 16000
DURATION_S = 5          # 5 seconds per signal
N = FS * DURATION_S

REF_LEVEL = 0.3         # white noise amplitude
ECHO_DECAY_MS = 60      # echo tail length
ECHO_DELAY_MS = 10      # direct-path delay
NEAR_LEVEL = 0.15       # near-end sine amplitude
NEAR_FREQ = 400         # Hz
DT_ONSET_S = 1.0        # double-talk starts at 1 s

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── helpers ─────────────────────────────────────────────────────────

def make_rir(fs, delay_ms, decay_ms):
    """Simple synthetic room impulse response (exponential decay)."""
    delay_samples = int(fs * delay_ms / 1000)
    decay_samples = int(fs * decay_ms / 1000)
    total = delay_samples + decay_samples
    h = np.zeros(total)
    # direct path + decaying tail
    t = np.arange(decay_samples) / fs
    decay_env = np.exp(-6.9 * t / (decay_ms / 1000))   # ~60 dB decay
    h[delay_samples:] = decay_env
    # normalise so peak echo ≈ 0.7× reference
    h /= np.sum(np.abs(h)) / 0.7
    return h


def generate():
    np.random.seed(42)

    # ── room impulse response ───────────────────────────────────────
    rir = make_rir(FS, ECHO_DELAY_MS, ECHO_DECAY_MS)

    # ── far-end reference (same white noise for both sets) ──────────
    ref = np.random.randn(N) * REF_LEVEL

    # ── echo signal ─────────────────────────────────────────────────
    echo = np.convolve(ref, rir)[:N]

    # ── Set 1: far-end single talk ──────────────────────────────────
    mic1 = echo.copy()
    # add tiny sensor noise so it's not perfectly clean
    mic1 += np.random.randn(N) * 1e-4

    ref1_path = os.path.join(OUT_DIR, "ref1.wav")
    mic1_path = os.path.join(OUT_DIR, "mic1.wav")
    sf.write(ref1_path, ref.astype(np.float32), FS)
    sf.write(mic1_path, mic1.astype(np.float32), FS)
    print(f"Set 1 (single talk): {ref1_path}, {mic1_path}")

    # ── Set 2: double talk ──────────────────────────────────────────
    t = np.arange(N) / FS
    near_end = NEAR_LEVEL * np.sin(2 * np.pi * NEAR_FREQ * t)
    # fade in near-end at DT_ONSET_S with 100 ms ramp
    onset = int(DT_ONSET_S * FS)
    ramp = int(0.1 * FS)
    fade = np.zeros(N)
    fade[onset:onset + ramp] = np.linspace(0, 1, ramp)
    fade[onset + ramp:] = 1.0
    near_end *= fade

    mic2 = echo + near_end
    mic2 += np.random.randn(N) * 1e-4

    ref2_path = os.path.join(OUT_DIR, "ref2.wav")
    mic2_path = os.path.join(OUT_DIR, "mic2.wav")
    sf.write(ref2_path, ref.astype(np.float32), FS)
    sf.write(mic2_path, mic2.astype(np.float32), FS)
    print(f"Set 2 (double talk): {ref2_path}, {mic2_path}")

    print("Done.")


if __name__ == "__main__":
    generate()
