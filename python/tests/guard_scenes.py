"""Shared scenes and an independent external renderer for the output-guard tests.

Everything here is written from the documented contract (AecResContext field meanings, the C-side
sk_pairwise_sum_f32 algorithm, the sqrt-Hann OLA synthesis), not by calling the code under test.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from modules.cng import AEC_CNG_SEED, cng_fill

WAV_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wav")
GRIDS = [(16000, 256), (16000, 512), (48000, 1024)]


def sqrt_hann(n: int) -> np.ndarray:
    """Periodic sqrt-Hann (denominator n)."""
    idx = np.arange(n, dtype=np.float64)
    return np.sqrt(0.5 * (1.0 - np.cos(2.0 * np.pi * idx / float(n)))).astype(np.float32)


# ---------------------------------------------------------------- scenes: lists of (near, far) hops

def synthetic_scene(name: str, hop: int, sr: int, n_hops: int, seed: int = 11):
    rng = np.random.RandomState(seed)
    delay = int(0.030 * sr)
    far_all = (0.3 * rng.uniform(-1.0, 1.0, (n_hops + 2) * hop)).astype(np.float32)
    hops = []
    for i in range(n_hops):
        far = far_all[i * hop:(i + 1) * hop].copy()
        if name == "uncorrelated":      # never usable, the guard fires on almost every hop
            near = 0.1 * rng.uniform(-1.0, 1.0, hop)
        elif name == "echo":            # becomes usable, the guard stays quiet
            near = 0.5 * far + 0.01 * rng.uniform(-1.0, 1.0, hop)
        elif name == "burst":           # alternating far bursts: the guard switches on and off
            on = (i // 20) % 2 == 0
            far = far if on else np.zeros(hop, np.float32)
            near = 0.5 * far + (0.0 if on else 0.1) * rng.uniform(-1.0, 1.0, hop)
        elif name == "delayed_echo":    # first delay acquisition hard-resets the post chain
            lo = i * hop - delay
            seg = np.zeros(hop, np.float32)
            if lo + hop > 0:
                src = np.arange(lo, lo + hop)
                ok = src >= 0
                seg[ok] = far_all[src[ok]]
            near = 0.5 * seg + 0.005 * rng.uniform(-1.0, 1.0, hop)
        else:
            raise ValueError(name)
        hops.append((np.asarray(near, np.float32), far))
    return hops


class _XorShift32:
    """The deterministic source of c_impl/test/test_guard_unify.c (``rnd()``): xorshift32, uniform in [-1, 1)."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def __call__(self) -> float:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s
        return float(s & 0xFFFFFF) / float(0x800000) - 1.0


def delay_branch_stream(name: str, sr: int, fft: int, seconds: int, seed: int = 0x2545F491):
    """The delay-branch streams of test_guard_unify.c, same construction (double arithmetic here, so not bit-identical
    to the C stream; the branches reached are checked by the tests, not assumed). The echo path delay changes at 1/3
    and 2/3 of the stream (480 / 1100 / 240 samples at 16 kHz, scaled with the rate).

    'relock'  gen_echo_stream(speech_far=1, echo=1, near_burst=0): speech-like far end, no double talk, so delay
              acquisition and shifts take the hard branches; on the 16 kHz grids the shift to the third delay is
              judged bad and soft-reverted.
    'dt_soft' gen_dt_soft_stream(0.8): every 0.8 s a 0.15 s window with a quiet far end and a loud near end keeps the
              DT-aware near-end hold armed, so every acquisition and shift takes the soft branch (run it with
              delay_acquire_warm_transfer off, as C does, so Path A is soft or hard)."""
    rnd = _XorShift32(seed)
    hop = fft // 2
    n = (seconds * sr // hop) * hop
    k = sr // 16000
    d1, d2, d3 = 480 * k, 1100 * k, 240 * k
    far = np.zeros(n, np.float64)
    near = np.zeros(n, np.float64)
    lp = nlp = 0.0
    if name == "relock":
        seg, env = sr // 4, 0.2
        for i in range(n):
            if i % seg == 0:
                r = rnd()
                env = 0.35 if r > 0.3 else (0.12 if r > -0.4 else 0.0)
            lp = 0.6 * lp + 0.4 * rnd()
            far[i] = env * lp * 2.0
            nlp = 0.7 * nlp + 0.3 * rnd()
            near[i] = 0.004 * nlp
        g1, g2 = 0.45, 0.2
    elif name == "dt_soft":
        per, win = int(0.8 * sr), int(0.15 * sr)
        for i in range(n):
            lp = 0.6 * lp + 0.4 * rnd()
            far[i] = 0.05 * rnd() if (i % per) < win else lp
        for i in range(n):
            nlp = 0.7 * nlp + 0.3 * rnd()
            near[i] = 0.8 * nlp if (i % per) < win else 0.004 * nlp
        g1, g2 = 0.8, 0.44 * 0.8
    else:
        raise ValueError(name)
    idx = np.arange(n)
    d = np.where(idx < n // 3, d1, np.where(idx < 2 * n // 3, d2, d3))
    echo = np.zeros(n, np.float64)
    ok1 = idx - d >= 0
    echo[ok1] += g1 * far[(idx - d)[ok1]]
    ok2 = idx - d - 37 >= 0
    echo[ok2] += g2 * far[(idx - d - 37)[ok2]]
    mic = (near + echo).astype(np.float32)
    far = far.astype(np.float32)
    return [(mic[i:i + hop].copy(), far[i:i + hop].copy()) for i in range(0, n, hop)]


def blind_clip(rel_mic: str, sr: int, hop: int):
    """A whole blind-set clip by path relative to ``wav/`` (int16 / 32768, truncated to the shorter channel, whole hops).
    Skips loudly when the gitignored corpus is absent."""
    import soundfile as sf
    mic_path = os.path.join(WAV_ROOT, rel_mic)
    ref_path = mic_path.replace("_mic.wav", "_lpb.wav")
    if not (os.path.exists(mic_path) and os.path.exists(ref_path)):
        pytest.skip("SKIP (blind-set corpus absent, wav/download_blind_test.py): %s" % rel_mic)
    mic, sr1 = sf.read(mic_path, dtype="int16")
    ref, sr2 = sf.read(ref_path, dtype="int16")
    assert sr1 == sr2 == sr
    n = min(len(mic), len(ref))
    f = lambda a: (a[:n].astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    mic, ref = f(mic), f(ref)
    return [(mic[i:i + hop].copy(), ref[i:i + hop].copy()) for i in range(0, n - hop, hop)]


# Clips chosen because they fire the guard: a far-end single talk that becomes usable after a cold start, and a
# near-end single talk (with some far-end leakage) whose filter never becomes usable. Most 16 kHz near-end
# clips have a silent far end, where E == Y and the guard (correctly) never fires.
_PREFERRED = {"aec_challenge_blind/nearend_singletalk": "0bFTCgvGGUaLnVt5LZHXaA_nearend_singletalk_mic.wav"}


def _first_clip(root: str, sub: str):
    paths = sorted(glob.glob(os.path.join(WAV_ROOT, root, sub, "*_mic.wav")))
    if not paths:
        pytest.skip("blind-set clips not present: %s/%s" % (root, sub))
    want = _PREFERRED.get(root + "/" + sub)
    if want:
        hit = [p for p in paths if os.path.basename(p) == want]
        if not hit:
            pytest.skip("blind-set clip %s not present" % want)
        return hit[0]
    return paths[0]


def clip_scene(kind: str, sr: int, hop: int, n_hops: int):
    """First ``n_hops`` hops of a blind-set clip. kind: 'farend' (guard fires in the cold start, then the
    filter becomes usable) or 'nearend' (the filter never becomes usable)."""
    import soundfile as sf
    root = "aec_challenge_blind" if sr == 16000 else "aec_challenge_blind_icassp2023"
    mic_path = _first_clip(root, {"farend": "farend_singletalk", "nearend": "nearend_singletalk"}[kind])
    mic, sr1 = sf.read(mic_path, dtype="int16")
    ref, sr2 = sf.read(mic_path.replace("_mic.wav", "_lpb.wav"), dtype="int16")
    assert sr1 == sr2 == sr
    n = min(len(mic), len(ref), n_hops * hop)
    f = lambda a: (a[:n].astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    mic, ref = f(mic), f(ref)
    return [(mic[i:i + hop].copy(), ref[i:i + hop].copy()) for i in range(0, (n // hop) * hop, hop)]


# ---------------------------------------------------------------- independent external renderer

class ExternalRenderer:
    """What an application does with the context seam: base * gain (+ CNG), IFFT, sqrt-Hann OLA.

    Tracks its own last-handled ``post_reset_serial``; on a change it clears the OLA tail and reseeds the CNG
    BEFORE the first draw and BEFORE the IFFT/OLA push of that hop.
    """

    def __init__(self, fft_size: int, hop: int, enable_cng: bool, serial0: int = 0):
        self.fft_size, self.hop, self.enable_cng = fft_size, hop, enable_cng
        self.window = sqrt_hann(fft_size)
        self.ola = np.zeros(fft_size, np.float32)
        self.rng = AEC_CNG_SEED
        self.serial = serial0
        self.resets = 0

    def render(self, ctx) -> np.ndarray:
        if ctx.post_reset_serial != self.serial:
            self.ola[:] = 0.0
            self.rng = AEC_CNG_SEED
            self.serial = ctx.post_reset_serial
            self.resets += 1
        g = np.asarray(ctx.res_gain, np.float32)
        base = np.asarray(ctx.output_base_spec, np.complex64)
        spec = np.empty(base.shape, np.complex64)
        spec.real = base.real * g
        spec.imag = base.imag * g
        if self.enable_cng:
            self.rng = cng_fill(self.rng, spec, ctx.comfort_noise, ctx.res_gain)
        frame = np.fft.irfft(spec, n=self.fft_size).astype(np.float32)
        self.ola += frame * self.window
        out = self.ola[:self.hop].copy()
        self.ola[:-self.hop] = self.ola[self.hop:]
        self.ola[-self.hop:] = 0.0
        return out


# ---------------------------------------------------------------- independent guard-energy reference

def pairwise_sum_f32(a) -> np.float32:
    """float32 pairwise sum written from the documented algorithm of ``sk_pairwise_sum_f32`` (the numpy float32
    pairwise reduction): n <= 8 serial; n <= 128 eight-accumulator leaf, then the remainder serially, combined
    as ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)) + remainder; otherwise split at half - half % 8.

    Equal to numpy's float32 ``np.sum`` for the leaf sizes the legal grids produce (n_freqs 129 / 257 / 513 split
    into leaves with at most one remainder element); with a longer remainder numpy adds the tail element by
    element to the accumulated sum, which this twin of the C helper does not."""
    a = np.asarray(a, dtype=np.float32)
    n = a.shape[0]
    f = np.float32
    if n <= 8:
        s = f(0.0)
        for x in a:
            s = f(s + x)
        return s
    if n <= 128:
        acc = a[:8].copy()
        i = 8
        while i + 8 <= n:
            acc = (acc + a[i:i + 8]).astype(np.float32)
            i += 8
        s = f(0.0)
        while i < n:
            s = f(s + a[i])
            i += 1
        r = f(f(f(acc[0] + acc[1]) + f(acc[2] + acc[3])) + f(f(acc[4] + acc[5]) + f(acc[6] + acc[7])))
        return f(r + s)
    half = n // 2
    half -= half % 8
    return f(pairwise_sum_f32(a[:half]) + pairwise_sum_f32(a[half:]))


def cabs_np_f32(spec) -> np.ndarray:
    """|z| as the C kernel ``sk__cabs_np_elem`` computes it (scaled hypot with an explicit fused multiply-add):
    larger * sqrt(fma(r, r, 1)), r = smaller / larger. The fma is emulated in float64 (r * r is exact there)."""
    spec = np.asarray(spec, dtype=np.complex64)
    ar, ai = np.abs(spec.real), np.abs(spec.imag)
    larger, smaller = np.maximum(ar, ai), np.minimum(ar, ai)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (smaller / larger).astype(np.float32)
    root = np.sqrt((ratio.astype(np.float64) * ratio.astype(np.float64) + 1.0).astype(np.float32))
    return np.where(larger == 0.0, np.float32(0.0), (larger * root).astype(np.float32)).astype(np.float32)


def spectral_energy_f32(spec) -> np.float32:
    mag = cabs_np_f32(spec)
    return pairwise_sum_f32((mag * mag).astype(np.float32))


def expected_guard(ctx) -> bool:
    """The rule from the contract: unusable this hop and sum|E|^2 > sum|Y|^2 (strict), Y = E + echo."""
    if ctx.usable_linear:
        return False
    y = (np.asarray(ctx.error_spec, np.complex64) + np.asarray(ctx.echo_spec, np.complex64)).astype(np.complex64)
    return bool(spectral_energy_f32(ctx.error_spec) > spectral_energy_f32(y))
