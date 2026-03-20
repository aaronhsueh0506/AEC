"""
Batch AEC processing: scan dataset directory, run AEC on all fileids,
save output wavs (ours + optional Speex/AEC3) and print ERLE/NE-Ret summary.

Each fileid also produces a 4-channel wav:
  ch0=nearend_mic, ch1=aec_output, ch2=farend, ch3=nearend_speech (or silence)

Usage:
    python3 batch_aec.py ../wav/                          # default: subband Shadow+RES, FL=2048
    python3 batch_aec.py ../wav/ -o ../wav/output/        # save to custom dir
    python3 batch_aec.py ../wav/ --speex --aec3           # also run Speex and WebRTC AEC3
    python3 batch_aec.py ../wav/ --files 0,20,800         # specific fileids only
"""

import numpy as np
import soundfile as sf
import argparse
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

# Try SpeexDSP
try:
    from speexdsp import EchoCanceller
    HAS_SPEEX = True
except ImportError:
    HAS_SPEEX = False

# WebRTC AEC3 CLI
AEC3_CLI = '/tmp/webrtc-ap/aec3_cli'
HAS_AEC3 = os.path.isfile(AEC3_CLI) and os.access(AEC3_CLI, os.X_OK)


def scan_fileids(dataset_dir):
    p = Path(dataset_dir)
    mic_files = sorted(p.glob('nearend_mic_fileid_*.wav'))
    groups = []
    for mic_file in mic_files:
        match = re.search(r'fileid_(\d+)', mic_file.stem)
        if not match:
            continue
        fid = match.group(1)
        farend = p / f'farend_speech_fileid_{fid}.wav'
        if not farend.exists():
            continue
        nearend_speech = p / f'nearend_speech_fileid_{fid}.wav'
        groups.append({
            'fileid': fid,
            'mic': str(mic_file),
            'ref': str(farend),
            'nearend_speech': str(nearend_speech) if nearend_speech.exists() else None,
        })
    return groups


def compute_erle(mic, output):
    mic_pwr = np.mean(mic ** 2)
    out_pwr = np.mean(output ** 2)
    if out_pwr < 1e-20:
        return 60.0
    return 10.0 * np.log10(mic_pwr / (out_pwr + 1e-20))


def compute_nearend_retention(mic, ref, output, hop=256):
    n = min(len(mic), len(ref), len(output))
    gain_sum, count = 0.0, 0
    processed = 0
    while processed + hop <= n:
        ref_pwr = np.mean(ref[processed:processed+hop] ** 2)
        mic_pwr = np.mean(mic[processed:processed+hop] ** 2)
        if ref_pwr < 1e-6 and mic_pwr > 1e-6:
            out_pwr = np.mean(output[processed:processed+hop] ** 2)
            gain_sum += out_pwr / (mic_pwr + 1e-10)
            count += 1
        processed += hop
    if count == 0:
        return None
    return 10.0 * np.log10(gain_sum / count + 1e-10)


def run_ours(mic, ref, sr, filter_length, mode, enable_dtd, enable_shadow, enable_res, mu):
    mode_map = {'nlms': AecMode.NLMS, 'freq': AecMode.FREQ,
                'subband': AecMode.SUBBAND, 'lms': AecMode.LMS}
    config = AecConfig(
        sample_rate=sr,
        mode=mode_map[mode],
        filter_length=filter_length,
        enable_dtd=enable_dtd,
        enable_shadow=enable_shadow,
        enable_res=enable_res,
    )
    if mu is not None:
        config.mu = mu
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
    return output[:n]


def run_speexdsp(mic, ref, sr, filter_length=2048):
    frame_size = 256
    ec = EchoCanceller.create(frame_size, filter_length, sr)
    n = min(len(mic), len(ref))
    output = np.zeros(n, dtype=np.float32)
    processed = 0
    while processed + frame_size <= n:
        mic_frame = mic[processed:processed+frame_size]
        ref_frame = ref[processed:processed+frame_size]
        mic_i16 = (mic_frame * 32767).clip(-32768, 32767).astype(np.int16)
        ref_i16 = (ref_frame * 32767).clip(-32768, 32767).astype(np.int16)
        out_bytes = ec.process(mic_i16.tobytes(), ref_i16.tobytes())
        out_i16 = np.frombuffer(out_bytes, dtype=np.int16)
        output[processed:processed+frame_size] = out_i16.astype(np.float32) / 32767.0
        processed += frame_size
    return output[:n]


def run_webrtc_aec3(mic_path, ref_path, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            [AEC3_CLI, mic_path, ref_path, out_path],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        output, _ = sf.read(out_path)
        return output.astype(np.float32)
    except Exception:
        return None
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def save_4ch(output_dir, fid, sr, mic, output, ref, nearend_speech_path):
    """Save 4-channel wav: ch0=mic, ch1=output, ch2=ref, ch3=nearend_speech."""
    n = len(mic)
    if nearend_speech_path and os.path.exists(nearend_speech_path):
        ns, _ = sf.read(nearend_speech_path)
        ns = ns.astype(np.float32)[:n]
        if len(ns) < n:
            ns = np.pad(ns, (0, n - len(ns)))
    else:
        ns = np.zeros(n, dtype=np.float32)
    multi = np.column_stack([mic[:n], output[:n], ref[:n], ns[:n]])
    path = os.path.join(output_dir, f"aec_4ch_fileid_{fid}.wav")
    sf.write(path, multi, sr)
    return path


def main():
    parser = argparse.ArgumentParser(description='Batch AEC processing')
    parser.add_argument('dataset_dir', help='Directory containing wav files')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: <dataset_dir>/aec_output/)')
    parser.add_argument('--filter', type=int, default=2048,
                        help='Filter length (default: 2048)')
    parser.add_argument('--speex-filter', type=int, default=2048,
                        help='SpeexDSP filter length (default: 2048)')
    parser.add_argument('--mode', choices=['nlms', 'freq', 'subband', 'lms'],
                        default='subband', help='Filter mode (default: subband)')
    parser.add_argument('--mu', type=float, default=None, help='Step size override')
    parser.add_argument('--enable-dtd', action='store_true', help='Enable DTD')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
    parser.add_argument('--no-res', action='store_true', help='Disable RES')
    parser.add_argument('--speex', action='store_true', help='Also run SpeexDSP')
    parser.add_argument('--aec3', action='store_true', help='Also run WebRTC AEC3')
    parser.add_argument('--files', type=str, default=None,
                        help='Comma-separated fileid list')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated fileids to exclude')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = args.output_dir or os.path.join(dataset_dir, 'aec_output')
    os.makedirs(output_dir, exist_ok=True)

    groups = scan_fileids(dataset_dir)
    if args.files:
        selected = set(args.files.split(','))
        groups = [g for g in groups if g['fileid'] in selected]
    if args.exclude:
        excluded = set(args.exclude.split(','))
        groups = [g for g in groups if g['fileid'] not in excluded]
    if not groups:
        print(f"No files found in {dataset_dir}")
        sys.exit(1)

    enable_shadow = not args.no_shadow
    enable_res = not args.no_res
    do_speex = args.speex and HAS_SPEEX
    do_aec3 = args.aec3 and HAS_AEC3

    if args.speex and not HAS_SPEEX:
        print("Warning: speexdsp not installed. pip3 install speexdsp")
    if args.aec3 and not HAS_AEC3:
        print(f"Warning: WebRTC AEC3 CLI not found at {AEC3_CLI}")

    dtd_str = "DTD" if args.enable_dtd else "no-DTD"
    shadow_str = "Shadow" if enable_shadow else "no-Shadow"
    res_str = "RES" if enable_res else "no-RES"
    print(f"Batch AEC: {args.mode} {dtd_str}+{shadow_str}+{res_str}, FL={args.filter}")
    if do_speex:
        print(f"  + SpeexDSP FL={args.speex_filter}")
    if do_aec3:
        print(f"  + WebRTC AEC3")
    print(f"Input:  {dataset_dir} ({len(groups)} files)")
    print(f"Output: {output_dir}")
    print()

    # Header
    cols = f"{'fileid':>8} {'Ours':>8} {'Ret':>7}"
    if do_speex:
        cols += f" {'Speex':>8} {'Ret':>7}"
    if do_aec3:
        cols += f" {'AEC3':>8} {'Ret':>7}"
    print(cols)
    print("-" * len(cols))

    results = []
    for group in sorted(groups, key=lambda g: int(g['fileid'])):
        fid = group['fileid']
        mic, sr = sf.read(group['mic'])
        ref, _ = sf.read(group['ref'])
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        # Ours
        output = run_ours(mic, ref, sr, args.filter, args.mode,
                          args.enable_dtd, enable_shadow, enable_res, args.mu)
        sf.write(os.path.join(output_dir, f"aec_output_fileid_{fid}.wav"), output, sr)

        # 4-channel: mic, output, ref, nearend_speech
        save_4ch(output_dir, fid, sr, mic, output, ref, group['nearend_speech'])

        erle_ours = compute_erle(mic, output)
        ret_ours = compute_nearend_retention(mic, ref, output)
        row = {'fileid': fid, 'ours': erle_ours, 'retention': ret_ours}

        # Speex
        if do_speex:
            out_speex = run_speexdsp(mic, ref, sr, filter_length=args.speex_filter)
            sf.write(os.path.join(output_dir, f"speex_output_fileid_{fid}.wav"), out_speex, sr)
            row['speex'] = compute_erle(mic, out_speex)
            row['speex_ret'] = compute_nearend_retention(mic, ref, out_speex)

        # AEC3
        if do_aec3:
            out_aec3 = run_webrtc_aec3(group['mic'], group['ref'], sr)
            if out_aec3 is not None:
                out_aec3 = out_aec3[:n]
                sf.write(os.path.join(output_dir, f"aec3_output_fileid_{fid}.wav"), out_aec3, sr)
                row['aec3'] = compute_erle(mic, out_aec3)
                row['aec3_ret'] = compute_nearend_retention(mic, ref, out_aec3)

        results.append(row)

        # Print row
        ret_str = f"{ret_ours:>6.1f}" if ret_ours is not None else f"{'N/A':>6}"
        line = f"{fid:>8} {erle_ours:>7.1f}  {ret_str}"
        if do_speex:
            speex_ret = f"{row.get('speex_ret', None):>6.1f}" if row.get('speex_ret') is not None else f"{'N/A':>6}"
            line += f" {row.get('speex', 0):>7.1f}  {speex_ret}"
        if do_aec3:
            if 'aec3' in row:
                aec3_ret = f"{row['aec3_ret']:>6.1f}" if row.get('aec3_ret') is not None else f"{'N/A':>6}"
                line += f" {row['aec3']:>7.1f}  {aec3_ret}"
            else:
                line += f" {'N/A':>7}  {'N/A':>6}"
        print(line)

    # Summary
    print("-" * len(cols))

    def _stats(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return None, None
        return np.mean(vals), np.median(vals)

    ours_mean, ours_med = np.mean([r['ours'] for r in results]), np.median([r['ours'] for r in results])
    ret_mean, ret_med = _stats('retention')

    ret_m = f"{ret_mean:>6.1f}" if ret_mean is not None else f"{'N/A':>6}"
    line = f"{'MEAN':>8} {ours_mean:>7.1f}  {ret_m}"
    if do_speex:
        sm, _ = _stats('speex')
        srm, _ = _stats('speex_ret')
        line += f" {sm:>7.1f}  {srm:>6.1f}" if sm is not None else ""
    if do_aec3:
        am, _ = _stats('aec3')
        arm, _ = _stats('aec3_ret')
        line += f" {am:>7.1f}  {arm:>6.1f}" if am is not None else ""
    print(line)

    ret_md = f"{ret_med:>6.1f}" if ret_med is not None else f"{'N/A':>6}"
    line = f"{'MEDIAN':>8} {ours_med:>7.1f}  {ret_md}"
    if do_speex:
        _, smd = _stats('speex')
        _, srmd = _stats('speex_ret')
        line += f" {smd:>7.1f}  {srmd:>6.1f}" if smd is not None else ""
    if do_aec3:
        _, amd = _stats('aec3')
        _, armd = _stats('aec3_ret')
        line += f" {amd:>7.1f}  {armd:>6.1f}" if amd is not None else ""
    print(line)

    # Win/Loss
    if do_speex or do_aec3:
        print()
    for comp_name, comp_key in [('Speex', 'speex'), ('AEC3', 'aec3')]:
        comp_results = [r for r in results if comp_key in r]
        if not comp_results:
            continue
        wins = sum(1 for r in comp_results if r['ours'] > r[comp_key])
        losses = sum(1 for r in comp_results if r[comp_key] > r['ours'])
        ties = len(comp_results) - wins - losses
        print(f"vs {comp_name}: Ours {wins}W / {comp_name} {losses}W / {ties} ties")

    print(f"\n{len(results)} files saved to {output_dir}")
    print(f"  Output files: aec_output_fileid_*.wav, aec_4ch_fileid_*.wav")
    if do_speex:
        print(f"  Speex files:  speex_output_fileid_*.wav")
    if do_aec3:
        print(f"  AEC3 files:   aec3_output_fileid_*.wav")


if __name__ == '__main__':
    main()
