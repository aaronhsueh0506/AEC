"""
Batch AEC processing: scan dataset directory, run AEC on all fileids,
save output wavs and print ERLE/NE-Ret summary.

Usage:
    python3 batch_aec.py ../wav/                          # default: subband Shadow+RES, FL=2048
    python3 batch_aec.py ../wav/ -o ../wav/output/        # save to custom dir
    python3 batch_aec.py ../wav/ --filter 1024 --enable-dtd
    python3 batch_aec.py ../wav/ --files 0,20,800         # specific fileids only
"""

import numpy as np
import soundfile as sf
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode


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
        groups.append({
            'fileid': fid,
            'mic': str(mic_file),
            'ref': str(farend),
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


def main():
    parser = argparse.ArgumentParser(description='Batch AEC processing')
    parser.add_argument('dataset_dir', help='Directory containing wav files')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory (default: <dataset_dir>/aec_output/)')
    parser.add_argument('--filter', type=int, default=2048,
                        help='Filter length (default: 2048)')
    parser.add_argument('--mode', choices=['nlms', 'freq', 'subband', 'lms'],
                        default='subband', help='Filter mode (default: subband)')
    parser.add_argument('--mu', type=float, default=None, help='Step size override')
    parser.add_argument('--enable-dtd', action='store_true', help='Enable DTD')
    parser.add_argument('--no-shadow', action='store_true', help='Disable shadow filter')
    parser.add_argument('--no-res', action='store_true', help='Disable RES')
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

    mode_map = {'nlms': AecMode.NLMS, 'freq': AecMode.FREQ,
                'subband': AecMode.SUBBAND, 'lms': AecMode.LMS}

    enable_shadow = not args.no_shadow
    enable_res = not args.no_res
    dtd_str = "DTD" if args.enable_dtd else "no-DTD"
    shadow_str = "Shadow" if enable_shadow else "no-Shadow"
    res_str = "RES" if enable_res else "no-RES"
    print(f"Batch AEC: {args.mode} {dtd_str}+{shadow_str}+{res_str}, FL={args.filter}")
    print(f"Input:  {dataset_dir} ({len(groups)} files)")
    print(f"Output: {output_dir}")
    print()

    header = f"{'fileid':>8} {'ERLE':>8} {'NE-Ret':>8} {'File':>s}"
    print(header)
    print("-" * 60)

    results = []
    for group in sorted(groups, key=lambda g: int(g['fileid'])):
        fid = group['fileid']
        mic, sr = sf.read(group['mic'])
        ref, _ = sf.read(group['ref'])
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        config = AecConfig(
            sample_rate=sr,
            mode=mode_map[args.mode],
            filter_length=args.filter,
            enable_dtd=args.enable_dtd,
            enable_shadow=enable_shadow,
            enable_res=enable_res,
        )
        if args.mu is not None:
            config.mu = args.mu

        aec = AEC(config)
        hop = aec.hop_size
        output = np.zeros(n, dtype=np.float32)
        processed = 0
        while processed + hop <= n:
            out_block = aec.process(mic[processed:processed+hop],
                                    ref[processed:processed+hop])
            output[processed:processed+hop] = out_block
            processed += hop

        # Save output
        out_filename = f"aec_output_fileid_{fid}.wav"
        out_path = os.path.join(output_dir, out_filename)
        sf.write(out_path, output, sr)

        # Metrics
        erle = compute_erle(mic, output)
        ret = compute_nearend_retention(mic, ref, output)
        results.append({'fileid': fid, 'erle': erle, 'retention': ret})

        ret_str = f"{ret:>7.1f}" if ret is not None else f"{'N/A':>7}"
        print(f"{fid:>8} {erle:>7.1f}  {ret_str}  {out_filename}")

    # Summary
    print("-" * 60)
    erles = [r['erle'] for r in results]
    rets = [r['retention'] for r in results if r['retention'] is not None]
    ret_mean_str = f"{np.mean(rets):>7.1f}" if rets else f"{'N/A':>7}"
    ret_med_str = f"{np.median(rets):>7.1f}" if rets else f"{'N/A':>7}"
    print(f"{'MEAN':>8} {np.mean(erles):>7.1f}  {ret_mean_str}")
    print(f"{'MEDIAN':>8} {np.median(erles):>7.1f}  {ret_med_str}")
    print(f"\n{len(results)} files saved to {output_dir}")


if __name__ == '__main__':
    main()
