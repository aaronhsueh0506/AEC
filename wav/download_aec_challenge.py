#!/usr/bin/env python3
"""Download AEC Challenge real dataset files from GitHub LFS."""
import os
import random
import subprocess
import urllib.parse

REPO = "microsoft/AEC-Challenge"
BRANCH = "main"
BASE_URL = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/datasets/real"

LFS_DIR = "/tmp/aec_challenge_repo/datasets/real"
OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aec_challenge")

SCENARIOS = {
    "farend_singletalk": {"suffix_mic": "_farend_singletalk_mic.wav", "has_lpb": True, "suffix_lpb": "_farend_singletalk_lpb.wav"},
    "nearend_singletalk": {"suffix_mic": "_nearend_singletalk_mic.wav", "has_lpb": False},
    "doubletalk": {"suffix_mic": "_doubletalk_mic.wav", "has_lpb": True, "suffix_lpb": "_doubletalk_lpb.wav"},
}

N = 10

def get_uuids(scenario_info):
    suffix = scenario_info["suffix_mic"]
    uuids = []
    for f in os.listdir(LFS_DIR):
        if f.endswith(suffix):
            uuid = f[:-len(suffix)]
            uuids.append(uuid)
    return sorted(uuids)

def download(fname, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    encoded = urllib.parse.quote(fname)
    url = f"{BASE_URL}/{encoded}"
    ret = subprocess.run(
        ["curl", "-sL", "-o", dest, "-w", "%{http_code}", url],
        capture_output=True, text=True, timeout=60
    )
    code = ret.stdout.strip()
    if code == "200" and os.path.getsize(dest) > 1000:
        return True
    else:
        print(f"  FAIL ({code}): {fname}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

def main():
    random.seed(42)

    for sc_name, sc_info in SCENARIOS.items():
        uuids = get_uuids(sc_info)
        print(f"\n{sc_name}: {len(uuids)} available UUIDs")
        selected = random.sample(uuids, min(N, len(uuids)))

        sc_dir = os.path.join(OUT_BASE, sc_name)
        os.makedirs(sc_dir, exist_ok=True)

        ok = 0
        for uuid in selected:
            # Download mic
            mic_f = f"{uuid}{sc_info['suffix_mic']}"
            print(f"  [{ok+1}/{N}] {mic_f}")
            if download(mic_f, os.path.join(sc_dir, mic_f)):
                ok_mic = True
            else:
                ok_mic = False

            # Download lpb if exists
            if sc_info.get("has_lpb"):
                lpb_f = f"{uuid}{sc_info['suffix_lpb']}"
                print(f"         {lpb_f}")
                download(lpb_f, os.path.join(sc_dir, lpb_f))

            if ok_mic:
                ok += 1

        n_files = len([f for f in os.listdir(sc_dir) if f.endswith('.wav')])
        print(f"  => {n_files} files downloaded")

    print("\nDone!")

if __name__ == "__main__":
    main()
