"""Deterministic synthetic AEC scenes for the Python<->C parity harness (no dataset needed).

Everything is generated from fixed seeds (numpy PCG64 default_rng + scipy lfilter), at the requested
sample rate, as float32. The harness pins the sha256 of every generated array
(test/pyc_parity/inputs.sha256), so a numpy/scipy change that alters the audio is reported as input
drift instead of silently moving the baseline.

make(name, sr, seconds) -> (mic float32, ref float32)
  echo      far-end white noise through a short decaying RIR (+200 ms @16k bulk delay) + faint noise:
            the converge-and-hold case (no near-end talker).
  speechdt  speech-like far end, echo through the RIR, speech-like near end gated on/off at 0.11 Hz:
            sustained double talk.
  dtb       the same echo scene with SHORT near-end bursts (100 ms every 700 ms from t=3 s), i.e. abrupt
            onsets: this is what makes the live-observer injection guard (audit S-02) actually fire.
  preecho   a dominant echo path at 25 ms plus a weaker low-frequency-heavy EARLIER path 18.75 ms before it
            (outside the main filter's window head): the case the live observer exists for -- the pre-echo
            path only reaches the observer filter's window, so its decision is the one the counter ordering
            fix (S-01) used to desync from Python's on (call_counter was incremented after, not before, the
            zero-mu early return in pbfdaf_process).
"""
import numpy as np
from scipy.signal import lfilter


def _speechlike(rng, n, sr):
    w = rng.standard_normal(n)
    x = lfilter([1.0], [1.0, -0.9], w)
    t = np.arange(n) / sr
    env = np.abs(np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 6))) ** 0.7
    gate = (np.sin(2 * np.pi * 0.17 * t) > -0.3).astype(float)
    x = x * env * gate
    return 0.3 * x / np.max(np.abs(x))


def _rir(rng, sr, rt_ms=12.0, length_ms=60.0, peak_ms=2.0):
    rl = int(length_ms * 1e-3 * sr)
    t = np.arange(rl)
    h = rng.standard_normal(rl) * np.exp(-t / (rt_ms * 1e-3 * sr))
    h[0] = 0.0
    h[int(peak_ms * 1e-3 * sr)] = 1.5
    return h / np.sqrt(np.sum(h ** 2))


def make(name, sr, seconds):
    n = int(seconds * sr)
    D = int(round(200 * sr / 16000))
    if name == 'echo':
        rng = np.random.default_rng(11)
        h = _rir(rng, sr)
        far = 0.15 * rng.standard_normal(n)
        echo = np.convolve(far, h)[:n]
        echo = np.concatenate([np.zeros(D), echo])[:n]
        mic = echo + 0.003 * rng.standard_normal(n)
    elif name == 'speechdt':
        rng = np.random.default_rng(12)
        h = _rir(rng, sr)
        far = _speechlike(rng, n, sr)
        echo = np.convolve(far, h)[:n]
        echo = np.concatenate([np.zeros(D), echo])[:n]
        t = np.arange(n) / sr
        ne = _speechlike(rng, n, sr) * (np.sin(2 * np.pi * 0.11 * t + 1.0) > 0.0)
        mic = 0.8 * echo + 0.7 * ne + 0.002 * rng.standard_normal(n)
    elif name == 'dtb':
        # ne gain 2.0, echo gain 1.0, seed 2, 100 ms bursts every 700 ms from t = 3 s
        rng = np.random.default_rng(202)
        b = int(0.100 * sr)
        per = int(0.700 * sr)
        far = _speechlike(rng, n, sr)
        h = _rir(rng, sr)
        echo = np.concatenate([np.zeros(D), np.convolve(far, h)[:n]])[:n]
        ne = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
        ne = 0.3 * ne / np.max(np.abs(ne))
        gate = np.zeros(n)
        for s0 in range(int(3 * sr), n - b, per):
            gate[s0:s0 + b] = 1.0
        mic = 1.0 * echo + 2.0 * ne * gate + 0.002 * rng.standard_normal(n)
    elif name == 'preecho':
        rng = np.random.default_rng(19)
        h = _rir(rng, sr)
        far = _speechlike(rng, n, sr)
        D1 = int(round(400 * sr / 16000))
        D0 = int(round(100 * sr / 16000))
        e1 = np.convolve(far, h)[:n]
        e1 = np.concatenate([np.zeros(D1), e1])[:n]
        lp = lfilter([0.05], [1.0, -0.95], far)
        e0 = np.convolve(lp, _rir(rng, sr, rt_ms=25.0, length_ms=80.0))[:n]
        e0 = np.concatenate([np.zeros(D0), e0])[:n]
        mic = e1 + 1.2 * e0 + 0.003 * rng.standard_normal(n)
    else:
        raise ValueError(name)
    return mic.astype(np.float32), far.astype(np.float32)
