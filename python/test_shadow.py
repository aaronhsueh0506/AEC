#!/usr/bin/env python3
"""
test_shadow.py - Demonstrate shadow filter effectiveness

Creates a scenario with echo path change at t=5s:
  Phase 1 (0-5s):  RIR with delay=200, gain=0.8
  Phase 2 (5-10s): RIR changes to delay=150, gain=0.6 (e.g. someone moved)

The main filter (mu=0.5) has already converged to the old RIR.
When RIR changes, the main filter's old weights produce large error,
causing divergence. Shadow filter (mu=0.25) is more conservative and
recovers faster because it doesn't overshoot as much.

Usage:
  python3 test_shadow.py
"""

import numpy as np
import soundfile as sf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

SR = 16000
DURATION = 10.0
N = int(SR * DURATION)
HOP = 256
CHANGE_TIME = 5.0
CHANGE_SAMPLE = int(CHANGE_TIME * SR)

def make_rir(delay, gain, n_taps=512):
    rir = np.zeros(n_taps, dtype=np.float32)
    rir[delay] = gain
    if delay + 50 < n_taps:
        rir[delay + 50] = gain * 0.3
    if delay + 120 < n_taps:
        rir[delay + 120] = -gain * 0.15
    if delay + 200 < n_taps:
        rir[delay + 200] = gain * 0.08
    return rir

def run_aec(mic, ref, enable_shadow, enable_dtd=True):
    config = AecConfig(
        mode=AecMode.SUBBAND,
        enable_shadow=enable_shadow,
        enable_dtd=enable_dtd,
    )
    aec = AEC(config)
    hop = aec.hop_size
    out = []
    erle_history = []
    copy_events = []
    frame_idx = 0

    for i in range(0, len(mic) - hop, hop):
        o = aec.process(mic[i:i+hop], ref[i:i+hop])
        out.append(o)

        # Track ERLE per frame
        mic_pwr = np.mean(mic[i:i+hop] ** 2)
        out_pwr = np.mean(o ** 2)
        if mic_pwr > 1e-10:
            erle = 10 * np.log10(mic_pwr / (out_pwr + 1e-10))
        else:
            erle = 0.0
        erle_history.append(erle)

        # Track shadow copy events
        if enable_shadow and hasattr(aec, 'shadow_filter') and aec.shadow_filter is not None:
            if (hasattr(aec, '_last_main_err') and
                aec.main_err_smooth != aec._last_main_err):
                if aec.main_err_smooth < aec._last_main_err * 0.5:
                    copy_events.append(frame_idx)
            aec._last_main_err = aec.main_err_smooth

        frame_idx += 1

    return np.concatenate(out), erle_history


def main():
    wav_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wav')

    # Generate test signal: white noise far-end
    rng = np.random.RandomState(42)
    farend = (rng.randn(N) * 0.3).astype(np.float32)

    # Two different RIRs
    rir1 = make_rir(delay=200, gain=0.8)  # Phase 1
    rir2 = make_rir(delay=100, gain=0.6)  # Phase 2 (changed path)

    # Generate mic signal with RIR change at CHANGE_TIME
    echo = np.zeros(N, dtype=np.float32)
    # Phase 1: convolve with rir1
    echo1 = np.convolve(farend[:CHANGE_SAMPLE], rir1)[:CHANGE_SAMPLE]
    echo[: CHANGE_SAMPLE] = echo1
    # Phase 2: convolve with rir2
    echo2_full = np.convolve(farend, rir2)[:N]
    echo[CHANGE_SAMPLE:] = echo2_full[CHANGE_SAMPLE:]

    mic = echo.copy()  # No near-end speech, pure echo

    # Add slight noise
    mic += (rng.randn(N) * 0.001).astype(np.float32)

    print(f"Test signal: {DURATION}s, echo path change at {CHANGE_TIME}s")
    print(f"  Phase 1: delay=200, gain=0.8")
    print(f"  Phase 2: delay=100, gain=0.6")
    print()

    # Run without shadow
    print("Running AEC without shadow filter...")
    out_no_shadow, erle_no_shadow = run_aec(mic, farend, enable_shadow=False)

    # Run with shadow
    print("Running AEC with shadow filter...")
    out_shadow, erle_shadow = run_aec(mic, farend, enable_shadow=True)

    # Compute metrics around the path change
    frames_per_sec = SR // HOP
    change_frame = int(CHANGE_TIME * frames_per_sec)

    # Recovery analysis: find frame where ERLE > 10dB after path change
    def find_recovery(erle_list, start_frame, threshold=10.0, window=5):
        for i in range(start_frame, len(erle_list) - window):
            if all(e > threshold for e in erle_list[i:i+window]):
                return i - start_frame
        return len(erle_list) - start_frame  # didn't recover

    recovery_no = find_recovery(erle_no_shadow, change_frame)
    recovery_sh = find_recovery(erle_shadow, change_frame)

    # Average ERLE in windows
    def avg_erle(erle_list, start_s, end_s):
        s = int(start_s * frames_per_sec)
        e = min(int(end_s * frames_per_sec), len(erle_list))
        vals = [x for x in erle_list[s:e] if x > -50]
        return np.mean(vals) if vals else 0.0

    print()
    print("=== Results ===")
    print(f"{'Metric':<40} {'No Shadow':>12} {'Shadow':>12}")
    print("-" * 66)
    print(f"{'ERLE Phase 1 (2-5s, converged)':<40} {avg_erle(erle_no_shadow, 2, 5):>10.1f} dB {avg_erle(erle_shadow, 2, 5):>10.1f} dB")
    print(f"{'ERLE after change (5-6s)':<40} {avg_erle(erle_no_shadow, 5, 6):>10.1f} dB {avg_erle(erle_shadow, 5, 6):>10.1f} dB")
    print(f"{'ERLE after change (6-7s)':<40} {avg_erle(erle_no_shadow, 6, 7):>10.1f} dB {avg_erle(erle_shadow, 6, 7):>10.1f} dB")
    print(f"{'ERLE Phase 2 (8-10s, re-converged)':<40} {avg_erle(erle_no_shadow, 8, 10):>10.1f} dB {avg_erle(erle_shadow, 8, 10):>10.1f} dB")
    print(f"{'Recovery time (frames to ERLE>10dB)':<40} {recovery_no:>10d} fr  {recovery_sh:>10d} fr")
    print(f"{'Recovery time (ms)':<40} {recovery_no * HOP / SR * 1000:>10.0f} ms {recovery_sh * HOP / SR * 1000:>10.0f} ms")

    # Output energy comparison in the danger zone (5.0-5.5s)
    s = int(5.0 * SR)
    e = int(5.5 * SR)
    mic_pwr = np.mean(mic[s:e] ** 2)
    no_sh_pwr = np.mean(out_no_shadow[s:min(e, len(out_no_shadow))] ** 2)
    sh_pwr = np.mean(out_shadow[s:min(e, len(out_shadow))] ** 2)
    print(f"{'Output power 5.0-5.5s (lower=better)':<40} {10*np.log10(no_sh_pwr+1e-10):>10.1f} dB {10*np.log10(sh_pwr+1e-10):>10.1f} dB")
    print(f"{'Mic power 5.0-5.5s (reference)':<40} {10*np.log10(mic_pwr+1e-10):>10.1f} dB")

    # Save test signals for manual inspection
    out_dir = os.path.join(wav_dir, 'shadow_test')
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, 'mic.wav'), mic, SR, subtype='PCM_16')
    sf.write(os.path.join(out_dir, 'ref.wav'), farend, SR, subtype='PCM_16')
    sf.write(os.path.join(out_dir, 'out_no_shadow.wav'), out_no_shadow, SR, subtype='PCM_16')
    sf.write(os.path.join(out_dir, 'out_shadow.wav'), out_shadow, SR, subtype='PCM_16')
    print(f"\nWAV files saved to: {out_dir}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        t_frames = np.arange(len(erle_no_shadow)) * HOP / SR

        # ERLE comparison
        axes[0].plot(t_frames, erle_no_shadow, alpha=0.7, label='No Shadow', linewidth=0.8)
        axes[0].plot(t_frames, erle_shadow, alpha=0.7, label='Shadow', linewidth=0.8)
        axes[0].axvline(CHANGE_TIME, color='red', linestyle='--', label='Echo Path Change')
        axes[0].set_ylabel('ERLE (dB)')
        axes[0].set_title('Shadow Filter Effectiveness: Echo Path Change at 5s')
        axes[0].legend()
        axes[0].set_ylim(-10, 60)
        axes[0].grid(True, alpha=0.3)

        # Mic + output waveforms
        t_samples = np.arange(len(mic)) / SR
        axes[1].plot(t_samples, mic, alpha=0.5, label='Mic (echo)', linewidth=0.5)
        axes[1].axvline(CHANGE_TIME, color='red', linestyle='--')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Mic Signal (echo only, path change at 5s)')
        axes[1].legend()

        t_out = np.arange(len(out_no_shadow)) / SR
        axes[2].plot(t_out, out_no_shadow, alpha=0.7, label='No Shadow', linewidth=0.5)
        axes[2].axvline(CHANGE_TIME, color='red', linestyle='--')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title('Output: No Shadow')
        axes[2].legend()

        axes[3].plot(t_out[:len(out_shadow)], out_shadow, alpha=0.7, label='Shadow', linewidth=0.5, color='green')
        axes[3].axvline(CHANGE_TIME, color='red', linestyle='--')
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Output: With Shadow')
        axes[3].legend()

        # Match y-axis scales for output comparison
        y_max = max(np.max(np.abs(out_no_shadow)), np.max(np.abs(out_shadow))) * 1.1
        axes[2].set_ylim(-y_max, y_max)
        axes[3].set_ylim(-y_max, y_max)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'shadow_test_results.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
