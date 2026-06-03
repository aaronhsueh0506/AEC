"""Generate a binary golden for the C FFT bit-exact parity test.

Dumps, for many input vectors at n=512 (the production fft_size):

  forward (np.fft.rfft):
    x   : float32[n]                  (the time-domain input the C feeds in)
    X   : complex64[n/2+1]            np.fft.rfft(x.astype(float64)).astype(complex64)

  inverse (np.fft.irfft):
    Xin : complex64[n/2+1]            (the spectrum the C feeds in)
    y   : float32[n]                  np.fft.irfft(Xin.astype(complex128), n).astype(float32)

The C side (parity_fft.c) replays fft_forward(x) vs X and fft_inverse(Xin) vs y
and reports the BIT-EXACT mismatch count (compared as the stored 32-bit values).

The two paths mirror exactly what the AEC pipeline does: the spectra are held
as complex64 (the Complex {float r,i} struct), and np.fft.rfft upcasts the
float32 input to float64 while np.fft.irfft upcasts complex64 -> complex128 and
normalises by 1/n. The downcasts back to float32 match the callers' .astype.

Binary layout (little-endian), header then records:
  int32  n
  int32  n_fwd            number of forward records
  int32  n_inv            number of inverse records
  --- n_fwd forward records ---
  float32[n]              x
  float32[2*(n/2+1)]      X interleaved (re,im) as complex64
  --- n_inv inverse records ---
  float32[2*(n/2+1)]      Xin interleaved (re,im) as complex64
  float32[n]              y

Run: python3 python/diag/gen_fft_golden.py /tmp/fft_golden.bin
"""
import struct
import sys

import numpy as np

N = 512
NF = N // 2 + 1


def make_forward_inputs():
    """A diverse battery of float32 time-domain vectors of length N."""
    rng = np.random.default_rng(12345)
    cases = []

    # broadband random, several magnitudes (incl. near-zero ~1e-11 — the regime
    # where the legacy radix-2 fft diverged from pocketfft by <=1 ULP)
    for scale in (1.0, 0.1, 1e-3, 1e-6, 1e-9, 1e-11, 1e-15):
        cases.append((rng.standard_normal(N).astype(np.float32) * scale))

    # all-zero / DC / Nyquist / impulses
    cases.append(np.zeros(N, np.float32))
    cases.append(np.full(N, 1.0, np.float32))
    cases.append(np.full(N, -3.5, np.float32))
    imp = np.zeros(N, np.float32); imp[0] = 1.0;  cases.append(imp.copy())
    imp = np.zeros(N, np.float32); imp[1] = -2.0; cases.append(imp.copy())
    imp = np.zeros(N, np.float32); imp[N // 2] = 7.0; cases.append(imp.copy())
    imp = np.zeros(N, np.float32); imp[N - 1] = 0.5;  cases.append(imp.copy())
    nyq = np.array([(-1.0) ** i for i in range(N)], np.float32); cases.append(nyq)

    # tones at a spread of bins
    t = np.arange(N)
    for k in (1, 2, 5, 17, 64, 200, 255, 256):
        cases.append(np.cos(2 * np.pi * k * t / N).astype(np.float32))
        cases.append((1e-7 * np.sin(2 * np.pi * k * t / N)).astype(np.float32))

    # windowed near-silence (sqrt-Hann * tiny noise) — mimics the cold-start
    # near-silent blocks that triggered the aec3_post out[hop] ULP residual
    win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(N) + 0.5) / N))
    for scale in (1e-9, 1e-11, 1e-13):
        cases.append((win * rng.standard_normal(N) * scale).astype(np.float32))

    # many more random near-zero vectors to stress the divergence regime
    for _ in range(64):
        s = 10.0 ** rng.uniform(-13, -8)
        cases.append((rng.standard_normal(N) * s).astype(np.float32))

    # plenty of full-scale random vectors
    for _ in range(64):
        cases.append(rng.standard_normal(N).astype(np.float32))

    return cases


def make_inverse_inputs(fwd_cases):
    """Spectra to feed irfft. Derive most from the forward outputs (so they are
    physically-valid half-complex spectra), plus a few synthetic ones."""
    rng = np.random.default_rng(999)
    specs = []

    # the rfft of every forward case (already complex64) — round-trip coverage
    for x in fwd_cases:
        X = np.fft.rfft(x.astype(np.float64)).astype(np.complex64)
        specs.append(X)

    # synthetic spectra: random complex (incl. nonzero DC/Nyquist imag, which
    # irfft must DISCARD — exercises the numpy repack drop-the-imag behaviour)
    for scale in (1.0, 1e-3, 1e-7, 1e-11):
        re = (rng.standard_normal(NF) * scale).astype(np.float32)
        im = (rng.standard_normal(NF) * scale).astype(np.float32)
        specs.append((re + 1j * im).astype(np.complex64))

    # pure DC, pure Nyquist, single-bin
    s = np.zeros(NF, np.complex64); s[0] = 3.0 + 5.0j; specs.append(s.copy())
    s = np.zeros(NF, np.complex64); s[NF - 1] = -2.0 + 1.0j; specs.append(s.copy())
    s = np.zeros(NF, np.complex64); s[7] = 1.0 - 0.5j; specs.append(s.copy())

    return specs


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fft_golden.bin"

    fwd = make_forward_inputs()
    inv = make_inverse_inputs(fwd)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", N, len(fwd), len(inv)))

        for x in fwd:
            x = np.asarray(x, np.float32)
            assert x.shape == (N,)
            X = np.fft.rfft(x.astype(np.float64)).astype(np.complex64)
            f.write(x.tobytes())
            f.write(X.view(np.float32).tobytes())  # interleaved re,im

        for Xin in inv:
            Xin = np.asarray(Xin, np.complex64)
            assert Xin.shape == (NF,)
            y = np.fft.irfft(Xin.astype(np.complex128), n=N).astype(np.float32)
            f.write(Xin.view(np.float32).tobytes())
            f.write(y.tobytes())

    print(f"wrote {out_path}: n={N} fwd={len(fwd)} inv={len(inv)}")


if __name__ == "__main__":
    main()
