"""
Signal quality diagnostic tool for AEC input files.

Analyzes mic and ref wav files for common real-world issues:
  - DC offset
  - Clipping / saturation
  - Level profile (RMS over time, transient detection)
  - Mic/Ref delay estimation (GCC-PHAT)
  - Low-frequency energy anomalies

Usage:
    python3 analyze_signal.py mic.wav ref.wav
    python3 analyze_signal.py mic.wav ref.wav --plot
    python3 analyze_signal.py mic.wav              # single file analysis
"""

import numpy as np
import soundfile as sf
import argparse
import sys


def analyze_dc_offset(signal, sr, block_ms=500):
    """Analyze DC offset per block and overall."""
    block_size = int(sr * block_ms / 1000)
    n_blocks = len(signal) // block_size
    dc_values = []
    for i in range(n_blocks):
        block = signal[i * block_size:(i + 1) * block_size]
        dc_values.append(np.mean(block))
    dc_values = np.array(dc_values)
    overall_dc = np.mean(signal)
    max_dc = np.max(np.abs(dc_values)) if len(dc_values) > 0 else 0.0
    dc_drift = np.max(dc_values) - np.min(dc_values) if len(dc_values) > 0 else 0.0
    return {
        'overall_dc': overall_dc,
        'max_block_dc': max_dc,
        'dc_drift': dc_drift,
        'dc_per_block': dc_values,
        'block_ms': block_ms,
    }


def analyze_clipping(signal, sr, threshold=0.95, block_ms=100):
    """Detect clipping events and statistics."""
    n = len(signal)
    abs_signal = np.abs(signal)
    clipped_mask = abs_signal > threshold

    # Overall stats
    total_clipped = np.sum(clipped_mask)
    clip_ratio = total_clipped / n if n > 0 else 0.0
    peak = np.max(abs_signal)

    # Find clipping regions (consecutive clipped samples)
    consec_events = []
    in_clip = False
    clip_start = 0
    for i in range(n):
        if clipped_mask[i]:
            if not in_clip:
                clip_start = i
                in_clip = True
        else:
            if in_clip:
                length = i - clip_start
                if length >= 2:  # at least 2 consecutive
                    consec_events.append({
                        'start_sample': clip_start,
                        'start_time': clip_start / sr,
                        'length_samples': length,
                        'peak': float(np.max(abs_signal[clip_start:i])),
                    })
                in_clip = False
    if in_clip:
        length = n - clip_start
        if length >= 2:
            consec_events.append({
                'start_sample': clip_start,
                'start_time': clip_start / sr,
                'length_samples': length,
                'peak': float(np.max(abs_signal[clip_start:n])),
            })

    # Per-block clipping ratio
    block_size = int(sr * block_ms / 1000)
    n_blocks = n // block_size
    clip_per_block = []
    for i in range(n_blocks):
        block_mask = clipped_mask[i * block_size:(i + 1) * block_size]
        clip_per_block.append(np.sum(block_mask) / block_size)

    return {
        'peak': peak,
        'total_clipped_samples': int(total_clipped),
        'clip_ratio': clip_ratio,
        'consecutive_events': consec_events,
        'clip_per_block': np.array(clip_per_block),
        'block_ms': block_ms,
    }


def analyze_level(signal, sr, block_ms=100):
    """RMS level profile and transient detection."""
    block_size = int(sr * block_ms / 1000)
    n_blocks = len(signal) // block_size
    rms_values = []
    peak_values = []
    for i in range(n_blocks):
        block = signal[i * block_size:(i + 1) * block_size]
        rms_values.append(np.sqrt(np.mean(block ** 2)))
        peak_values.append(np.max(np.abs(block)))
    rms_values = np.array(rms_values)
    peak_values = np.array(peak_values)

    # Detect transients: blocks with RMS > 10x median
    median_rms = np.median(rms_values[rms_values > 1e-6]) if np.any(rms_values > 1e-6) else 1e-6
    transients = []
    for i, rms in enumerate(rms_values):
        if rms > median_rms * 10:
            transients.append({
                'block': i,
                'time': i * block_ms / 1000,
                'rms': float(rms),
                'peak': float(peak_values[i]),
                'ratio_to_median': float(rms / median_rms),
            })

    # Overall stats
    overall_rms = np.sqrt(np.mean(signal ** 2))
    rms_db = 20 * np.log10(overall_rms + 1e-10)

    # Activity ratio (blocks with RMS > -60 dB)
    active_threshold = 10 ** (-60 / 20)
    active_blocks = np.sum(rms_values > active_threshold)
    activity_ratio = active_blocks / len(rms_values) if len(rms_values) > 0 else 0.0

    return {
        'overall_rms': overall_rms,
        'overall_rms_db': rms_db,
        'median_rms': float(median_rms),
        'activity_ratio': activity_ratio,
        'transients': transients,
        'rms_per_block': rms_values,
        'peak_per_block': peak_values,
        'block_ms': block_ms,
    }


def analyze_low_freq_energy(signal, sr, fft_size=2048, cutoff_hz=50):
    """Check for abnormal low-frequency energy (DC offset symptom)."""
    n = len(signal)
    n_frames = n // fft_size
    low_energy_total = 0.0
    total_energy = 0.0
    cutoff_bin = int(cutoff_hz * fft_size / sr)

    for i in range(n_frames):
        frame = signal[i * fft_size:(i + 1) * fft_size]
        spec = np.abs(np.fft.rfft(frame)) ** 2
        low_energy_total += np.sum(spec[:cutoff_bin + 1])
        total_energy += np.sum(spec)

    low_ratio = low_energy_total / (total_energy + 1e-20)
    return {
        'low_freq_energy_ratio': low_ratio,
        'cutoff_hz': cutoff_hz,
        'low_freq_energy_db': 10 * np.log10(low_ratio + 1e-20),
    }


def estimate_delay(mic, ref, sr, max_delay_ms=200):
    """Estimate mic-ref delay using GCC-PHAT."""
    n = min(len(mic), len(ref))
    # Use middle portion to avoid initial transients
    start = n // 4
    end = 3 * n // 4
    mic_seg = mic[start:end]
    ref_seg = ref[start:end]

    fft_size = 1
    while fft_size < len(mic_seg):
        fft_size *= 2

    M = np.fft.rfft(mic_seg, fft_size)
    R = np.fft.rfft(ref_seg, fft_size)
    cross = M * np.conj(R)
    magnitude = np.abs(cross) + 1e-10
    gcc_phat = np.fft.irfft(cross / magnitude)

    max_delay_samples = int(sr * max_delay_ms / 1000)
    # Search positive delays (mic lags ref)
    search = np.concatenate([gcc_phat[:max_delay_samples],
                             gcc_phat[-max_delay_samples:]])
    delays = np.concatenate([np.arange(max_delay_samples),
                             np.arange(-max_delay_samples, 0)])
    best_idx = np.argmax(np.abs(search))
    best_delay = int(delays[best_idx])
    confidence = float(np.abs(search[best_idx]))

    return {
        'delay_samples': best_delay,
        'delay_ms': best_delay / sr * 1000,
        'confidence': confidence,
    }


def print_report(name, dc, clip, level, low_freq, delay=None):
    """Print formatted diagnostic report."""
    print(f"\n{'=' * 60}")
    print(f"  Signal Analysis: {name}")
    print(f"{'=' * 60}")

    # Level
    print(f"\n--- Level ---")
    print(f"  Overall RMS:      {level['overall_rms']:.6f} ({level['overall_rms_db']:.1f} dB)")
    print(f"  Median block RMS: {level['median_rms']:.6f}")
    print(f"  Activity ratio:   {level['activity_ratio']:.1%}")

    # DC Offset
    print(f"\n--- DC Offset ---")
    print(f"  Overall DC:       {dc['overall_dc']:.6f}")
    print(f"  Max block DC:     {dc['max_block_dc']:.6f}")
    print(f"  DC drift:         {dc['dc_drift']:.6f}")
    severity = "OK"
    if dc['max_block_dc'] > 0.01:
        severity = "WARNING - significant DC offset"
    elif dc['max_block_dc'] > 0.001:
        severity = "MINOR - small DC offset"
    print(f"  Assessment:       {severity}")

    # Clipping
    print(f"\n--- Clipping ---")
    print(f"  Peak amplitude:   {clip['peak']:.6f}")
    print(f"  Clipped samples:  {clip['total_clipped_samples']} ({clip['clip_ratio']:.4%})")
    print(f"  Consecutive events (>= 2 samples):")
    if clip['consecutive_events']:
        for evt in clip['consecutive_events'][:10]:
            print(f"    t={evt['start_time']:.3f}s: {evt['length_samples']} samples, peak={evt['peak']:.4f}")
        if len(clip['consecutive_events']) > 10:
            print(f"    ... and {len(clip['consecutive_events']) - 10} more events")
    else:
        print(f"    None detected")

    # Low frequency
    print(f"\n--- Low Frequency Energy (< {low_freq['cutoff_hz']} Hz) ---")
    print(f"  Energy ratio:     {low_freq['low_freq_energy_ratio']:.6f} ({low_freq['low_freq_energy_db']:.1f} dB)")
    severity = "OK"
    if low_freq['low_freq_energy_ratio'] > 0.1:
        severity = "WARNING - abnormal low-freq energy (possible DC offset)"
    elif low_freq['low_freq_energy_ratio'] > 0.01:
        severity = "MINOR - elevated low-freq energy"
    print(f"  Assessment:       {severity}")

    # Transients
    print(f"\n--- Transients (> 10x median RMS) ---")
    if level['transients']:
        for t in level['transients'][:10]:
            print(f"    t={t['time']:.2f}s: RMS={t['rms']:.4f}, peak={t['peak']:.4f}, "
                  f"{t['ratio_to_median']:.1f}x median")
    else:
        print(f"    None detected")

    # Delay
    if delay is not None:
        print(f"\n--- Mic/Ref Delay (GCC-PHAT) ---")
        print(f"  Estimated delay:  {delay['delay_samples']} samples ({delay['delay_ms']:.1f} ms)")
        print(f"  Confidence:       {delay['confidence']:.4f}")

    print()


def plot_results(mic, ref, sr, mic_dc, mic_clip, mic_level, ref_dc, ref_clip, ref_level):
    """Generate diagnostic plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    t = np.arange(len(mic)) / sr

    # Row 0: Waveforms
    ax = axes[0]
    ax.plot(t, mic, color='royalblue', linewidth=0.3, alpha=0.7, label='Mic')
    if ref is not None:
        t_ref = np.arange(len(ref)) / sr
        ax.plot(t_ref, ref, color='green', linewidth=0.3, alpha=0.5, label='Ref')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveforms', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 1: RMS level profile
    ax = axes[1]
    block_ms = mic_level['block_ms']
    t_blocks = np.arange(len(mic_level['rms_per_block'])) * block_ms / 1000
    rms_db = 20 * np.log10(mic_level['rms_per_block'] + 1e-10)
    ax.plot(t_blocks, rms_db, color='royalblue', linewidth=0.8, label='Mic RMS')
    if ref is not None:
        t_blocks_ref = np.arange(len(ref_level['rms_per_block'])) * block_ms / 1000
        rms_db_ref = 20 * np.log10(ref_level['rms_per_block'] + 1e-10)
        ax.plot(t_blocks_ref, rms_db_ref, color='green', linewidth=0.8, alpha=0.7, label='Ref RMS')
    ax.set_ylabel('RMS (dB)')
    ax.set_title('Level Profile', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-80, 0)

    # Row 2: DC offset per block
    ax = axes[2]
    t_dc = np.arange(len(mic_dc['dc_per_block'])) * mic_dc['block_ms'] / 1000
    ax.plot(t_dc, mic_dc['dc_per_block'], color='royalblue', linewidth=0.8, label='Mic DC')
    if ref is not None:
        t_dc_ref = np.arange(len(ref_dc['dc_per_block'])) * ref_dc['block_ms'] / 1000
        ax.plot(t_dc_ref, ref_dc['dc_per_block'], color='green', linewidth=0.8, alpha=0.7, label='Ref DC')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('DC Offset')
    ax.set_title('DC Offset per Block', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Row 3: Clipping ratio per block
    ax = axes[3]
    t_clip = np.arange(len(mic_clip['clip_per_block'])) * mic_clip['block_ms'] / 1000
    ax.plot(t_clip, mic_clip['clip_per_block'] * 100, color='red', linewidth=0.8, label='Mic clip %')
    if ref is not None:
        t_clip_ref = np.arange(len(ref_clip['clip_per_block'])) * ref_clip['block_ms'] / 1000
        ax.plot(t_clip_ref, ref_clip['clip_per_block'] * 100, color='orange', linewidth=0.8, alpha=0.7,
                label='Ref clip %')
    ax.set_ylabel('Clip %')
    ax.set_xlabel('Time (s)')
    ax.set_title('Clipping Ratio per Block', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = 'signal_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze signal quality for AEC input files')
    parser.add_argument('mic', help='Microphone (near-end) WAV file')
    parser.add_argument('ref', nargs='?', default=None, help='Reference (far-end) WAV file (optional)')
    parser.add_argument('--plot', action='store_true', help='Generate diagnostic plot')
    parser.add_argument('--threshold', type=float, default=0.95, help='Clipping threshold (default: 0.95)')
    args = parser.parse_args()

    # Read mic
    mic, sr = sf.read(args.mic)
    mic = mic.astype(np.float32)
    if mic.ndim > 1:
        print(f"Mic has {mic.shape[1]} channels, using channel 0")
        mic = mic[:, 0]
    print(f"Mic: {args.mic}")
    print(f"  Sample rate: {sr} Hz, Duration: {len(mic)/sr:.2f}s, Samples: {len(mic)}")

    # Analyze mic
    mic_dc = analyze_dc_offset(mic, sr)
    mic_clip = analyze_clipping(mic, sr, threshold=args.threshold)
    mic_level = analyze_level(mic, sr)
    mic_low_freq = analyze_low_freq_energy(mic, sr)

    # Read ref (optional)
    ref = None
    ref_dc = ref_clip = ref_level = ref_low_freq = None
    delay = None
    if args.ref:
        ref, sr_ref = sf.read(args.ref)
        ref = ref.astype(np.float32)
        if ref.ndim > 1:
            print(f"Ref has {ref.shape[1]} channels, using channel 0")
            ref = ref[:, 0]
        print(f"Ref: {args.ref}")
        print(f"  Sample rate: {sr_ref} Hz, Duration: {len(ref)/sr_ref:.2f}s, Samples: {len(ref)}")
        if sr_ref != sr:
            print(f"  WARNING: sample rate mismatch (mic={sr}, ref={sr_ref})")

        ref_dc = analyze_dc_offset(ref, sr)
        ref_clip = analyze_clipping(ref, sr, threshold=args.threshold)
        ref_level = analyze_level(ref, sr)
        ref_low_freq = analyze_low_freq_energy(ref, sr)
        delay = estimate_delay(mic, ref, sr)

    # Print reports
    print_report("Microphone (near-end)", mic_dc, mic_clip, mic_level, mic_low_freq, delay)
    if ref is not None:
        print_report("Reference (far-end)", ref_dc, ref_clip, ref_level, ref_low_freq)

    # Length check
    if ref is not None:
        diff = abs(len(mic) - len(ref))
        if diff > 0:
            print(f"WARNING: Length mismatch: mic={len(mic)}, ref={len(ref)}, diff={diff} samples ({diff/sr*1000:.1f} ms)")

    # Summary
    print(f"{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    issues = []
    if mic_dc['max_block_dc'] > 0.01:
        issues.append(f"Mic DC offset: {mic_dc['max_block_dc']:.4f}")
    if ref_dc is not None and ref_dc['max_block_dc'] > 0.01:
        issues.append(f"Ref DC offset: {ref_dc['max_block_dc']:.4f}")
    if mic_clip['total_clipped_samples'] > 0:
        issues.append(f"Mic clipping: {mic_clip['total_clipped_samples']} samples ({mic_clip['clip_ratio']:.4%})")
    if ref_clip is not None and ref_clip['total_clipped_samples'] > 0:
        issues.append(f"Ref clipping: {ref_clip['total_clipped_samples']} samples ({ref_clip['clip_ratio']:.4%})")
    if mic_level['transients']:
        issues.append(f"Mic transients: {len(mic_level['transients'])} detected")
    if ref_level is not None and ref_level['transients']:
        issues.append(f"Ref transients: {len(ref_level['transients'])} detected")
    if mic_low_freq['low_freq_energy_ratio'] > 0.01:
        issues.append(f"Mic low-freq energy: {mic_low_freq['low_freq_energy_ratio']:.4f}")

    if issues:
        print(f"\n  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  No significant issues detected.")
    print()

    # Plot
    if args.plot:
        plot_results(mic, ref, sr, mic_dc, mic_clip, mic_level,
                     ref_dc, ref_clip, ref_level)


if __name__ == '__main__':
    main()
