"""
Run AEC on both test sets and plot:
  - Waveforms (mic, ref, output)
  - Estimated impulse response vs true IR

Usage:
    python3 plot_aec_results.py [--mode time|freq|subband]
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Import from sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode
from gen_test_signals import make_rir, FS, ECHO_DELAY_MS, ECHO_DECAY_MS


def run_aec(mic_path, ref_path, mode):
    """Run AEC and return output + filter object."""
    mic, sr = sf.read(mic_path)
    ref, _ = sf.read(ref_path)
    mic = mic.astype(np.float32)
    ref = ref.astype(np.float32)

    config = AecConfig(
        sample_rate=sr,
        mode=mode,
        enable_dtd=True,
        mu=0.3,
        filter_length_ms=250,
    )
    aec = AEC(config)
    hop = aec.hop_size
    n = min(len(mic), len(ref))
    output = np.zeros(n, dtype=np.float32)

    processed = 0
    while processed + hop <= n:
        out_block = aec.process(mic[processed:processed+hop],
                                ref[processed:processed+hop])
        output[processed:processed+hop] = out_block
        processed += hop

    return mic[:n], ref[:n], output, aec, sr


def get_estimated_ir(aec):
    """Extract estimated impulse response from AEC filter."""
    if isinstance(aec.filter, __import__('aec', fromlist=['NlmsFilter']).NlmsFilter):
        # TIME mode: weights are the IR directly
        return aec.filter.weights.copy()
    else:
        # FREQ / SUBBAND mode: IFFT each partition and concatenate
        W = aec.filter.W  # [n_partitions, n_freqs]
        block_size = aec.filter.block_size
        hop = aec.filter.hop_size
        ir = np.zeros(W.shape[0] * hop, dtype=np.float32)
        for p in range(W.shape[0]):
            w_time = np.fft.irfft(W[p], block_size)
            ir[p * hop:(p + 1) * hop] = w_time[:hop]
        return ir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['time', 'freq', 'subband'],
                        default='time')
    args = parser.parse_args()

    mode_map = {'time': AecMode.TIME, 'freq': AecMode.FREQ,
                'subband': AecMode.SUBBAND}
    mode = mode_map[args.mode]

    base = os.path.dirname(os.path.abspath(__file__))

    # ── True IR ─────────────────────────────────────────────────────
    true_ir = make_rir(FS, ECHO_DELAY_MS, ECHO_DECAY_MS)

    # ── Run AEC on both sets ────────────────────────────────────────
    sets = [
        ("Set 1: Single Talk", "mic1.wav", "ref1.wav"),
        ("Set 2: Double Talk", "mic2.wav", "ref2.wav"),
    ]

    fig, axes = plt.subplots(len(sets), 3, figsize=(16, 8))

    for row, (title, mic_f, ref_f) in enumerate(sets):
        mic_path = os.path.join(base, mic_f)
        ref_path = os.path.join(base, ref_f)

        mic, ref, out, aec, sr = run_aec(mic_path, ref_path, mode)
        est_ir = get_estimated_ir(aec)

        t = np.arange(len(mic)) / sr

        # ── Col 0: waveforms ────────────────────────────────────────
        ax = axes[row, 0]
        ax.plot(t, mic, alpha=0.6, label='mic (near+echo)')
        ax.plot(t, out, alpha=0.8, label='AEC output')
        ax.set_title(f'{title} — Waveforms')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(0, t[-1])

        # ── Col 1: error energy ─────────────────────────────────────
        ax = axes[row, 1]
        frame_len = 160
        n_frames = len(mic) // frame_len
        mic_energy = np.array([np.mean(mic[i*frame_len:(i+1)*frame_len]**2)
                               for i in range(n_frames)])
        out_energy = np.array([np.mean(out[i*frame_len:(i+1)*frame_len]**2)
                               for i in range(n_frames)])
        t_frames = np.arange(n_frames) * frame_len / sr
        eps = 1e-10
        erle = 10 * np.log10(mic_energy / (out_energy + eps) + eps)
        ax.plot(t_frames, erle)
        ax.set_title(f'{title} — ERLE (dB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ERLE (dB)')
        ax.set_xlim(0, t_frames[-1])
        ax.axhline(0, color='gray', ls='--', lw=0.5)

        # ── Col 2: estimated vs true IR ────────────────────────────
        ax = axes[row, 2]
        t_true = np.arange(len(true_ir)) / sr * 1000  # ms
        t_est = np.arange(len(est_ir)) / sr * 1000
        ax.plot(t_true, true_ir, label='True IR', lw=1.5)
        ax.plot(t_est, est_ir, label='Estimated IR', lw=1.0, alpha=0.8)
        ax.set_title(f'{title} — Impulse Response')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.set_xlim(0, max(t_true[-1], t_est[-1]) * 0.5)

    fig.suptitle(f'AEC Results (mode={args.mode})', fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(base, f'aec_results_{args.mode}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
