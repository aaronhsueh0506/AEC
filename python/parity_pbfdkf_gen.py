#!/usr/bin/env python3
"""Generate Python PBFDKF parity baseline for C v3.8.1 port verification.

Runs Python PBFDKF on deterministic synthetic input, captures per-frame
state (output, error_spec, P, W) and saves to .npz.

The C parity driver loads the SAME input, runs C PBFDKF, dumps matching
state. Compare with `parity_pbfdkf_check.py` (rtol=1e-5 atol=1e-7).

Reference: docs/c_port_spec.md "Parity test infrastructure" section.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from aec import PBFDKF

# Config (matches default 16kHz / 52ms / 5 partitions)
SAMPLE_RATE = 16000
HOP = 160                      # 10 ms hop
BLOCK = 2 * HOP                # 320 (overlap-save 50%)
FFT_SIZE = 512                 # next pow2 ≥ block
N_FREQS = FFT_SIZE // 2 + 1    # 257
FILTER_LEN_SAMPLES = SAMPLE_RATE * 52 // 1000  # 832
N_PARTITIONS = (FILTER_LEN_SAMPLES + HOP - 1) // HOP  # 6
N_FRAMES = 200
SEED = 42

OUT_FILE = Path(__file__).parent / 'parity_pbfdkf_python.npz'
INPUT_FILE = Path(__file__).parent / 'parity_pbfdkf_input.bin'  # for C consumer


def main():
    print(f"Config: hop={HOP} block={BLOCK} fft={FFT_SIZE} n_freqs={N_FREQS} "
          f"partitions={N_PARTITIONS} frames={N_FRAMES}")

    rng = np.random.default_rng(SEED)
    # Synthetic input: white noise scaled to typical speech level (~ -20 dBFS)
    n_samples = N_FRAMES * HOP
    mic = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    ref = (rng.standard_normal(n_samples) * 0.1).astype(np.float32)

    # Deterministic mu_scale schedule: 0.7 first 100 frames (FS-like),
    # then 0.2 next 50 (DT-like), then 0.7 again
    mu_schedule = np.full(N_FRAMES, 0.7, dtype=np.float32)
    mu_schedule[100:150] = 0.2

    # Run PBFDKF
    filt = PBFDKF(BLOCK, N_PARTITIONS, mu=0.3, delta=1e-8, hop_size=HOP)

    output = np.zeros(n_samples, dtype=np.float32)
    err_specs = np.zeros((N_FRAMES, N_FREQS), dtype=np.complex64)
    echo_specs = np.zeros((N_FRAMES, N_FREQS), dtype=np.complex64)
    P_states = np.zeros((N_FRAMES, N_PARTITIONS, N_FREQS), dtype=np.float32)
    W_states = np.zeros((N_FRAMES, N_PARTITIONS, N_FREQS), dtype=np.complex64)
    power_states = np.zeros((N_FRAMES, N_FREQS), dtype=np.float32)

    for f in range(N_FRAMES):
        s = f * HOP
        e = s + HOP
        o = filt.process(mic[s:e], ref[s:e], mu_scale=float(mu_schedule[f]))
        output[s:e] = o
        err_specs[f] = filt.error_spec.copy()
        echo_specs[f] = filt.echo_spec.copy()
        P_states[f] = filt.P.copy()
        W_states[f] = filt.W.copy()
        power_states[f] = filt.power.copy()

    np.savez(OUT_FILE,
             mic=mic, ref=ref, mu_schedule=mu_schedule,
             output=output, err_specs=err_specs, echo_specs=echo_specs,
             P_states=P_states, W_states=W_states, power_states=power_states,
             config=np.array([HOP, BLOCK, FFT_SIZE, N_FREQS, N_PARTITIONS,
                              N_FRAMES, SAMPLE_RATE], dtype=np.int32))

    # Also write input as raw float32 binary for C parity driver to read
    with open(INPUT_FILE, 'wb') as fp:
        fp.write(mic.tobytes())
        fp.write(ref.tobytes())
        fp.write(mu_schedule.tobytes())

    print(f"Saved Python parity baseline → {OUT_FILE}")
    print(f"Saved C input bin           → {INPUT_FILE}")
    print(f"  mic energy : {np.mean(mic**2):.4e}")
    print(f"  output energy: {np.mean(output**2):.4e}")
    print(f"  P[final] mean: {np.mean(P_states[-1]):.4e}")
    print(f"  W[final] |max|: {np.max(np.abs(W_states[-1])):.4e}")


if __name__ == '__main__':
    main()
