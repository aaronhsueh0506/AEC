#!/usr/bin/env python3
"""
Evaluate AEC on AEC Challenge dataset.
- farend_singletalk: ERLE metric
- doubletalk (synthetic): PESQ metric (using nearend_speech * nearend_scale as ref)

Usage:
    python3 eval_aec_challenge.py ../wav/aec_challenge/ --aec3 --speex
"""
import numpy as np
import soundfile as sf
import argparse
import json
import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecMode

# Try PESQ
try:
    from pesq import pesq as pesq_fn
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

# Try SpeexDSP
try:
    from speexdsp import EchoCanceller
    HAS_SPEEX = True
except ImportError:
    HAS_SPEEX = False

# WebRTC AEC3 CLI
import subprocess, tempfile
AEC3_CLI = '/tmp/webrtc-ap/aec3_cli'
HAS_AEC3 = os.path.isfile(AEC3_CLI) and os.access(AEC3_CLI, os.X_OK)


def estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    """Pre-compute delay using full-signal cross-correlation.

    Uses the entire signal for maximum accuracy.
    Plain cross-correlation (no whitening) is most reliable for reverberant data.
    """
    max_d = int(max_delay_ms * sr / 1000)
    n = min(len(mic), len(ref))
    m = mic[:n].astype(np.float64)
    r = ref[:n].astype(np.float64)

    # FFT-based cross-correlation (full signal)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    mic_spec = np.fft.rfft(m, n=fft_size)
    ref_spec = np.fft.rfft(r, n=fft_size)
    cross = mic_spec * np.conj(ref_spec)
    xcorr = np.fft.irfft(cross, n=fft_size)

    # Search positive delays only (mic lags ref)
    max_search = min(max_d, fft_size // 2)
    delay = int(np.argmax(xcorr[:max_search + 1]))
    return delay


def run_ours(mic, ref, sr, fl):
    # Pre-compute delay and align reference signal
    delay = estimate_delay(mic, ref, sr)
    n = min(len(mic), len(ref))
    if delay > 0 and delay < n:
        # Delay ref: ref_aligned[t] = ref[t - delay], so filter sees aligned echo
        ref_aligned = np.zeros(n, dtype=np.float32)
        ref_aligned[delay:] = ref[:n - delay]
    else:
        ref_aligned = ref[:n]

    config = AecConfig(sample_rate=sr, mode=AecMode.SUBBAND,
                       filter_length=fl, enable_dtd=False,
                       enable_shadow=True, enable_res=True,
                       enable_delay_est=False, use_kalman=True)
    aec = AEC(config)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], ref_aligned[pos:pos+hop])
        pos += hop
    return out[:n]


def run_speex(mic, ref, sr, fl=2048):
    frame_size = 256
    ec = EchoCanceller.create(frame_size, fl, sr)
    n = min(len(mic), len(ref))
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + frame_size <= n:
        mi = (mic[pos:pos+frame_size] * 32767).clip(-32768, 32767).astype(np.int16)
        ri = (ref[pos:pos+frame_size] * 32767).clip(-32768, 32767).astype(np.int16)
        ob = ec.process(mi.tobytes(), ri.tobytes())
        out[pos:pos+frame_size] = np.frombuffer(ob, dtype=np.int16).astype(np.float32) / 32767.0
        pos += frame_size
    return out[:n]


def run_aec3(mic_path, ref_path, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        out_path = tmp.name
    try:
        r = subprocess.run([AEC3_CLI, mic_path, ref_path, out_path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        o, _ = sf.read(out_path)
        return o.astype(np.float32)
    except:
        return None
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def compute_erle(mic, output):
    mic_pwr = np.mean(mic ** 2)
    out_pwr = np.mean(output ** 2)
    if out_pwr < 1e-20:
        return 60.0
    return 10.0 * np.log10(mic_pwr / (out_pwr + 1e-20))


def compute_pesq(ref, deg, sr):
    if not HAS_PESQ:
        return None
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    mode = 'wb' if sr >= 16000 else 'nb'
    try:
        return pesq_fn(sr, ref, deg, mode)
    except:
        return None


def eval_farend_singletalk(base_dir, fl, do_speex, do_aec3, out_dir):
    """Evaluate farend_singletalk with ERLE."""
    sc_dir = os.path.join(base_dir, 'farend_singletalk')
    if not os.path.isdir(sc_dir):
        print("No farend_singletalk directory found")
        return

    # Find all mic files
    mic_files = sorted([f for f in os.listdir(sc_dir) if '_farend_singletalk_mic.wav' in f])
    if not mic_files:
        print("No farend_singletalk files found")
        return

    print(f"\n{'='*60}")
    print(f"FAREND SINGLETALK — ERLE ({len(mic_files)} cases)")
    print(f"{'='*60}")

    hdr = f"{'Case':>5} {'Ours':>8}"
    if do_speex: hdr += f" {'Speex':>8}"
    if do_aec3:  hdr += f" {'AEC3':>8}"
    print(hdr)
    print("-" * len(hdr))

    erles = {'ours': [], 'speex': [], 'aec3': []}

    for i, mf in enumerate(mic_files):
        uuid = mf.replace('_farend_singletalk_mic.wav', '')
        lpb_f = f'{uuid}_farend_singletalk_lpb.wav'
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)

        mic, sr = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic, ref = mic.astype(np.float32), ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        # Ours
        output = run_ours(mic, ref, sr, fl)
        sf.write(os.path.join(out_dir, f"fs_{i}_ours.wav"), output, sr)
        e_ours = compute_erle(mic, output)
        erles['ours'].append(e_ours)

        line = f"{i:>5} {e_ours:>8.1f}"

        # Speex
        if do_speex:
            out_sp = run_speex(mic, ref, sr)
            sf.write(os.path.join(out_dir, f"fs_{i}_speex.wav"), out_sp, sr)
            e_sp = compute_erle(mic, out_sp)
            erles['speex'].append(e_sp)
            line += f" {e_sp:>8.1f}"

        # AEC3
        if do_aec3:
            out_a3 = run_aec3(mic_path, lpb_path, sr)
            if out_a3 is not None:
                out_a3 = out_a3[:n]
                sf.write(os.path.join(out_dir, f"fs_{i}_aec3.wav"), out_a3, sr)
                e_a3 = compute_erle(mic, out_a3)
                erles['aec3'].append(e_a3)
                line += f" {e_a3:>8.1f}"
            else:
                line += f" {'N/A':>8}"

        print(line)

    # Summary
    print("-" * len(hdr))
    summary = f"{'MEAN':>5} {np.mean(erles['ours']):>8.1f}"
    if do_speex and erles['speex']:
        summary += f" {np.mean(erles['speex']):>8.1f}"
    if do_aec3 and erles['aec3']:
        summary += f" {np.mean(erles['aec3']):>8.1f}"
    print(summary)


def eval_doubletalk(base_dir, fl, do_speex, do_aec3, out_dir):
    """Evaluate doubletalk with PESQ (synthetic data with nearend_speech)."""
    sc_dir = os.path.join(base_dir, 'doubletalk')
    if not os.path.isdir(sc_dir):
        print("No doubletalk directory found")
        return

    # Load metadata
    meta_path = os.path.join(sc_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    # Find all mic files (synthetic naming: nearend_mic_fileid_N.wav)
    mic_files = sorted([f for f in os.listdir(sc_dir) if f.startswith('nearend_mic_fileid_') and f.endswith('.wav')],
                       key=lambda f: int(re.search(r'fileid_(\d+)', f).group(1)))
    if not mic_files:
        print("No doubletalk files found")
        return

    if not HAS_PESQ:
        print("Warning: pesq not installed. pip3 install pesq")

    print(f"\n{'='*60}")
    print(f"DOUBLETALK (synthetic) — PESQ + ERLE ({len(mic_files)} cases)")
    print(f"{'='*60}")

    hdr = f"{'FID':>6} {'SER':>4}"
    hdr += f" {'ERLE':>6} {'PESQ':>6}"
    if do_speex: hdr += f" {'spERLE':>7} {'spPESQ':>7}"
    if do_aec3:  hdr += f" {'a3ERLE':>7} {'a3PESQ':>7}"
    print(hdr)
    print("-" * len(hdr))

    results = {'ours_erle': [], 'ours_pesq': [], 'speex_erle': [], 'speex_pesq': [],
               'aec3_erle': [], 'aec3_pesq': []}

    for mf in mic_files:
        fid = re.search(r'fileid_(\d+)', mf).group(1)
        lpb_f = f'farend_speech_fileid_{fid}.wav'
        ne_f = f'nearend_speech_fileid_{fid}.wav'

        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)
        ne_path = os.path.join(sc_dir, ne_f)

        mic, sr = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic, ref = mic.astype(np.float32), ref.astype(np.float32)
        n = min(len(mic), len(ref))
        mic, ref = mic[:n], ref[:n]

        # PESQ reference: nearend_speech * nearend_scale
        scale = meta.get(fid, {}).get('nearend_scale', 1.0)
        ser = meta.get(fid, {}).get('ser', '?')
        pesq_ref = None
        if os.path.exists(ne_path):
            ne, _ = sf.read(ne_path)
            ne = ne.astype(np.float32)[:n] * scale
            if len(ne) < n:
                ne = np.pad(ne, (0, n - len(ne)))
            pesq_ref = ne

        # Ours
        output = run_ours(mic, ref, sr, fl)
        sf.write(os.path.join(out_dir, f"dt_{fid}_ours.wav"), output, sr)
        e_ours = compute_erle(mic, output)
        results['ours_erle'].append(e_ours)

        p_ours = compute_pesq(pesq_ref, output, sr) if pesq_ref is not None else None
        if p_ours is not None:
            results['ours_pesq'].append(p_ours)

        line = f"{fid:>6} {ser:>4} {e_ours:>6.1f}"
        line += f" {p_ours:>6.2f}" if p_ours is not None else f" {'N/A':>6}"

        # Speex
        if do_speex:
            out_sp = run_speex(mic, ref, sr)
            sf.write(os.path.join(out_dir, f"dt_{fid}_speex.wav"), out_sp, sr)
            e_sp = compute_erle(mic, out_sp)
            results['speex_erle'].append(e_sp)
            p_sp = compute_pesq(pesq_ref, out_sp, sr) if pesq_ref is not None else None
            if p_sp is not None:
                results['speex_pesq'].append(p_sp)
            line += f" {e_sp:>7.1f}"
            line += f" {p_sp:>7.2f}" if p_sp is not None else f" {'N/A':>7}"

        # AEC3
        if do_aec3:
            out_a3 = run_aec3(mic_path, lpb_path, sr)
            if out_a3 is not None:
                out_a3 = out_a3[:n]
                sf.write(os.path.join(out_dir, f"dt_{fid}_aec3.wav"), out_a3, sr)
                e_a3 = compute_erle(mic, out_a3)
                results['aec3_erle'].append(e_a3)
                p_a3 = compute_pesq(pesq_ref, out_a3, sr) if pesq_ref is not None else None
                if p_a3 is not None:
                    results['aec3_pesq'].append(p_a3)
                line += f" {e_a3:>7.1f}"
                line += f" {p_a3:>7.2f}" if p_a3 is not None else f" {'N/A':>7}"
            else:
                line += f" {'N/A':>7} {'N/A':>7}"

        print(line)

    # Summary
    print("-" * len(hdr))
    summary = f"{'MEAN':>6} {'':>4}"
    summary += f" {np.mean(results['ours_erle']):>6.1f}"
    summary += f" {np.mean(results['ours_pesq']):>6.2f}" if results['ours_pesq'] else f" {'N/A':>6}"
    if do_speex:
        summary += f" {np.mean(results['speex_erle']):>7.1f}" if results['speex_erle'] else f" {'N/A':>7}"
        summary += f" {np.mean(results['speex_pesq']):>7.2f}" if results['speex_pesq'] else f" {'N/A':>7}"
    if do_aec3:
        summary += f" {np.mean(results['aec3_erle']):>7.1f}" if results['aec3_erle'] else f" {'N/A':>7}"
        summary += f" {np.mean(results['aec3_pesq']):>7.2f}" if results['aec3_pesq'] else f" {'N/A':>7}"
    print(summary)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AEC on AEC Challenge dataset')
    parser.add_argument('dataset_dir', help='aec_challenge/ directory')
    parser.add_argument('--filter', type=int, default=2048, help='Filter length')
    parser.add_argument('--speex', action='store_true', help='Also run SpeexDSP')
    parser.add_argument('--aec3', action='store_true', help='Also run WebRTC AEC3')
    parser.add_argument('-o', '--output-dir', default=None, help='Output directory')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset_dir)
    out_dir = args.output_dir or os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)

    do_speex = args.speex and HAS_SPEEX
    do_aec3 = args.aec3 and HAS_AEC3

    if args.speex and not HAS_SPEEX:
        print("Warning: speexdsp not installed")
    if args.aec3 and not HAS_AEC3:
        print(f"Warning: AEC3 CLI not found at {AEC3_CLI}")
    if not HAS_PESQ:
        print("Warning: pesq not installed. pip3 install pesq")

    eval_farend_singletalk(base_dir, args.filter, do_speex, do_aec3, out_dir)
    eval_doubletalk(base_dir, args.filter, do_speex, do_aec3, out_dir)

    print(f"\nOutput saved to {out_dir}")


if __name__ == '__main__':
    main()
