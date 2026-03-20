#!/usr/bin/env python3
"""Download synthetic doubletalk cases from AEC Challenge."""
import csv
import json
import os
import random
import subprocess
import urllib.parse
import numpy as np
import soundfile as sf

REPO = "microsoft/AEC-Challenge"
BRANCH = "main"
BASE_URL = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/datasets/synthetic"
META_CSV = "/tmp/aec_challenge_repo/datasets/synthetic/meta.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aec_challenge", "doubletalk")
MAX_DELAY_MS = 128.0
N = 10

def download(subdir, fname, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    encoded = urllib.parse.quote(fname)
    url = f"{BASE_URL}/{subdir}/{encoded}"
    ret = subprocess.run(
        ["curl", "-sL", "-o", dest, "-w", "%{http_code}", url],
        capture_output=True, text=True, timeout=60
    )
    code = ret.stdout.strip()
    if code == "200" and os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    if os.path.exists(dest):
        os.remove(dest)
    return False

def measure_delay(mic_path, lpb_path):
    mic, sr = sf.read(mic_path)
    lpb, _ = sf.read(lpb_path)
    n = min(len(mic), len(lpb))
    seg = min(n, sr * 2)
    corr = np.correlate(mic[:seg].astype(np.float64), lpb[:seg].astype(np.float64), mode='full')
    delay = np.argmax(np.abs(corr)) - (seg - 1)
    return delay, delay / sr * 1000

def main():
    # Read meta.csv, filter clean DT cases
    with open(META_CSV) as f:
        rows = list(csv.DictReader(f))

    clean_dt = [r for r in rows if r['is_nearend_noisy'] == '0' and r['is_farend_noisy'] == '0']
    print(f"Clean DT candidates: {len(clean_dt)}")

    # Sort by SER, pick spread: 1 each from SER -5 to +4
    by_ser = {}
    for r in clean_dt:
        ser = int(r['ser'])
        if -5 <= ser <= 4:
            by_ser.setdefault(ser, []).append(r)

    random.seed(99)
    candidates = []
    for ser in range(-5, 5):
        if ser in by_ser:
            random.shuffle(by_ser[ser])
            candidates.extend(by_ser[ser])  # all candidates, prioritize variety

    # Try to get N valid cases with delay < 128ms
    kept = []
    meta = {}

    for r in candidates:
        if len(kept) >= N:
            break
        fid = r['fileid']
        scale = float(r['nearend_scale'])
        ser = int(r['ser'])

        mic_f = f"nearend_mic_fileid_{fid}.wav"
        lpb_f = f"farend_speech_fileid_{fid}.wav"
        ne_f = f"nearend_speech_fileid_{fid}.wav"
        echo_f = f"echo_fileid_{fid}.wav"

        mic_path = os.path.join(OUT_DIR, mic_f)
        lpb_path = os.path.join(OUT_DIR, lpb_f)

        # Download mic + lpb first to check delay
        print(f"[{len(kept)+1}/{N}] fileid={fid} SER={ser} scale={scale:.4f}")
        if not download("nearend_mic_signal", mic_f, mic_path):
            print(f"  FAIL download mic")
            continue
        if not download("farend_speech", lpb_f, lpb_path):
            print(f"  FAIL download lpb")
            os.remove(mic_path)
            continue

        # Check delay
        ds, dms = measure_delay(mic_path, lpb_path)
        if abs(dms) > MAX_DELAY_MS:
            print(f"  SKIP: delay={dms:.1f}ms > {MAX_DELAY_MS}ms")
            os.remove(mic_path)
            os.remove(lpb_path)
            continue

        # Download nearend_speech + echo
        ne_path = os.path.join(OUT_DIR, ne_f)
        echo_path = os.path.join(OUT_DIR, echo_f)
        if not download("nearend_speech", ne_f, ne_path):
            print(f"  FAIL download nearend_speech")
            os.remove(mic_path); os.remove(lpb_path)
            continue
        if not download("echo_signal", echo_f, echo_path):
            print(f"  FAIL download echo")
            os.remove(mic_path); os.remove(lpb_path); os.remove(ne_path)
            continue

        print(f"  OK: delay={dms:.1f}ms")
        kept.append(fid)
        meta[fid] = {"nearend_scale": scale, "ser": ser, "delay_ms": round(dms, 1)}

    # Save metadata
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    n_files = len([f for f in os.listdir(OUT_DIR) if f.endswith('.wav')])
    print(f"\nDone! {len(kept)} cases, {n_files} wav files")
    print(f"Metadata saved to {meta_path}")
    for fid, info in meta.items():
        print(f"  fileid={fid}: SER={info['ser']}, scale={info['nearend_scale']:.4f}, delay={info['delay_ms']}ms")

if __name__ == "__main__":
    main()
