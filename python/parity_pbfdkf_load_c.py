#!/usr/bin/env python3
"""Convert parity_pbfdkf_c.bin (raw C dump) → parity_pbfdkf_c.npz.

Layout (matches c_impl/example/parity_pbfdkf_test.c writer):
  Header: 7 int32 [hop, block, fft, n_freqs, n_partitions, n_frames, sr]
  Per frame (× n_frames):
    output[hop]                                   float32
    err_spec[n_freqs]                             complex64 (real,imag)
    echo_spec[n_freqs]                            complex64
    P[n_partitions][n_freqs]                      float32
    W[n_partitions][n_freqs]                      complex64
    power[n_freqs]                                float32
"""
import numpy as np
import sys
from pathlib import Path

IN_FILE  = Path(__file__).parent / 'parity_pbfdkf_c.bin'
OUT_FILE = Path(__file__).parent / 'parity_pbfdkf_c.npz'


def main():
    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found. Run ./c_impl/bin/parity_pbfdkf_test first.")
        sys.exit(2)

    with open(IN_FILE, 'rb') as fp:
        data = fp.read()

    # Parse header
    hdr = np.frombuffer(data[:28], dtype=np.int32)
    HOP, BLOCK, FFT, NF, NP, NFR, SR = hdr.tolist()
    print(f"C header: hop={HOP} block={BLOCK} fft={FFT} n_freqs={NF} "
          f"partitions={NP} frames={NFR} sr={SR}")

    # Per-frame layout sizes (in float32 units; complex64 = 2 floats)
    out_n   = HOP
    err_n   = NF * 2
    echo_n  = NF * 2
    P_n     = NP * NF
    W_n     = NP * NF * 2
    pwr_n   = NF
    per_frame_floats = out_n + err_n + echo_n + P_n + W_n + pwr_n

    body = np.frombuffer(data[28:], dtype=np.float32)
    expected = NFR * per_frame_floats
    if body.size != expected:
        print(f"ERROR: body size {body.size} != expected {expected}")
        sys.exit(3)

    body = body.reshape(NFR, per_frame_floats)

    # Slice per-frame
    output    = body[:, 0:out_n].copy()
    err_re_im = body[:, out_n:out_n+err_n].reshape(NFR, NF, 2)
    err_specs = (err_re_im[..., 0] + 1j * err_re_im[..., 1]).astype(np.complex64)
    s = out_n + err_n
    echo_re_im = body[:, s:s+echo_n].reshape(NFR, NF, 2)
    echo_specs = (echo_re_im[..., 0] + 1j * echo_re_im[..., 1]).astype(np.complex64)
    s += echo_n
    P_states = body[:, s:s+P_n].reshape(NFR, NP, NF).astype(np.float32)
    s += P_n
    W_re_im  = body[:, s:s+W_n].reshape(NFR, NP, NF, 2)
    W_states = (W_re_im[..., 0] + 1j * W_re_im[..., 1]).astype(np.complex64)
    s += W_n
    power_states = body[:, s:s+pwr_n].reshape(NFR, NF).astype(np.float32)

    # Output as flat n_samples to match Python writer
    output_flat = output.reshape(-1)

    np.savez(OUT_FILE,
             output=output_flat, err_specs=err_specs, echo_specs=echo_specs,
             P_states=P_states, W_states=W_states, power_states=power_states,
             config=hdr)
    print(f"Wrote {OUT_FILE}")
    print(f"  output energy: {np.mean(output_flat**2):.4e}")
    print(f"  P[final] mean: {np.mean(P_states[-1]):.4e}")
    print(f"  W[final] |max|: {np.max(np.abs(W_states[-1])):.4e}")


if __name__ == '__main__':
    main()
