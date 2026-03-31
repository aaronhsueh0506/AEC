#!/usr/bin/env python3
"""Download AEC Challenge blind test sets from GitHub LFS.

Supports:
  - Interspeech 2021 blind test set (default)
  - ICASSP 2023 blind test set

Usage:
    python3 download_blind_test.py [--dataset interspeech2021|icassp2023] [--max N] [--no-movement] [--movement-only]

Downloads to:
    interspeech2021 -> wav/aec_challenge_blind/
    icassp2023      -> wav/aec_challenge_blind_icassp2023/
"""
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
import argparse

REPO = "microsoft/AEC-Challenge"
BRANCH = "main"

# --- Interspeech 2021 configuration ---

INTERSPEECH2021_DATASET_PATH = "datasets/blind_test_set_interspeech2021"
INTERSPEECH2021_OUT_DIR = "aec_challenge_blind"

# GitHub directory naming uses hyphens, but file naming uses underscores
SCENARIOS = {
    "farend_singletalk": {
        "gh_dir": "farend-singletalk",
        "suffixes": ["_farend_singletalk_mic.wav", "_farend_singletalk_lpb.wav"],
    },
    "nearend_singletalk": {
        "gh_dir": "nearend-singletalk",
        "suffixes": ["_nearend_singletalk_mic.wav", "_nearend_singletalk_lpb.wav"],
    },
    "doubletalk": {
        "gh_dir": "doubletalk",
        "suffixes": ["_doubletalk_mic.wav", "_doubletalk_lpb.wav"],
    },
}

# Movement scenarios: same GitHub directories, different file suffixes
MOVEMENT_SCENARIOS = {
    "farend_singletalk": {
        "gh_dir": "farend-singletalk",
        "suffixes": ["_farend_singletalk_with_movement_mic.wav",
                     "_farend_singletalk_with_movement_lpb.wav"],
    },
    "doubletalk": {
        "gh_dir": "doubletalk",
        "suffixes": ["_doubletalk_with_movement_mic.wav",
                     "_doubletalk_with_movement_lpb.wav"],
    },
    # nearend_singletalk has no movement files
}

# Known UUIDs from the GitHub listing (non-movement cases)
UUIDS = {
    "farend_singletalk": [
        "0KjzXA3g20qsd8zmSekADw", "1fvt8ajGxk2OhS7UglBjoA",
        "49DamGOwmUWGCn23bmI8xw", "4BX_05GVzEiPc4TJp4OwzA",
        "7GTxyTksSUqCnP5y0ILG4A", "HAxmF7v4dE0itSp5R5B3Dw",
        "J2JmN8YYhEWF1J6a9NMBgQ", "KjYjc9Ri3E6wVcTzW5K27w",
        "L4Xnh1PbOkWpNdG5tBqJQg", "M7rplKmGZk6y_Zk5Dt2yrQ",
    ],
    "doubletalk": [
        "0I0XMl3M0ECO0U1N0cJvpg", "49IIo03GZ0CYQOmeA3A0BA",
        "7GTxyTksSUqCnP5y0ILG4A", "HAxmF7v4dE0itSp5R5B3Dw",
        "J2JmN8YYhEWF1J6a9NMBgQ", "KjYjc9Ri3E6wVcTzW5K27w",
        "L4Xnh1PbOkWpNdG5tBqJQg", "M7rplKmGZk6y_Zk5Dt2yrQ",
    ],
    "nearend_singletalk": [
        "014AzuqPZku2004NbTTmcA", "0I0XMl3M0ECO0U1N0cJvpg",
        "49IIo03GZ0CYQOmeA3A0BA", "7GTxyTksSUqCnP5y0ILG4A",
        "HAxmF7v4dE0itSp5R5B3Dw", "J2JmN8YYhEWF1J6a9NMBgQ",
        "KjYjc9Ri3E6wVcTzW5K27w", "L4Xnh1PbOkWpNdG5tBqJQg",
        "M7rplKmGZk6y_Zk5Dt2yrQ",
    ],
}

# Movement UUIDs — complete list from GitHub API (131 FS + 114 DT)
MOVEMENT_UUIDS = {
    "farend_singletalk": [
        "0I0XMl3M0ECO0U1N0cJvpg", "0luXwWjGEEC9G5nf0yTVXw",
        "3UAwzzOa40aCXQAmEdpwww", "4pN9yn7mhEa5iDiKnr5jlw",
        "5bJUo1K3uEmMrGa9UhGyVg", "BYRb7rMHZUOVwHO90KRg9Q",
        "Fi80N5kW9U6nwaoS04O3vQ", "Hp5g1asacUCt5rJVLO1FuQ",
        "Hq00pd6Ey0mGtuMFRoF79w", "I2bme08keUmAnyJRKNYDGQ",
        "IqtJR4tjJkWrwUjYorz0Og", "IrQvqOTCmEWMXn9k2ICtRQ",
        "Ixf70mgKwkCoFYq32586cw", "Ja8OngfthkOCmL8ldcRNyg",
        "Je6gJ7y1PECStwxnrOe9aA", "JjCzlhn3gEiBQvfJtPNJ9A",
        "JtodX3Ug6Eu5TYu0HN5IOw", "KSN5Jrzo7kaixP0z8xfr4Q",
        "KgCF0xsN8EibTdJ2Yac2dw", "Khk1qeMXFUuvFhw3YRSm0w",
        "LHsrJBRGnUKiMC2mihEr0g", "LeV1uF4j10Whm0FPG80tmw",
        "Lsa5WpwTpUeb7C9dc9RXuQ", "ML4MF3Mea0yurjceNQPfNA",
        "MYrVxVEMxkaE7OuyTUmI0Q", "MkSLte0FTkqybGcLTwA3Tw",
        "N2rQLbnp2UOg2QFRaggbDw", "N94NwopiZEyNnraWHLMDcg",
        "Nhk0I6UHKkaUguACs1mHNg", "OX2l6zV7nkmmSkVA3ETLKg",
        "OXCtw0FVhUWGUBil6Uucdw", "OmB0Ht0hmE2crVnftAEtsw",
        "OwkV685H2Em8jjTOAbhbow", "PXfMWCKVykukw7Se9Aq7wQ",
        "QEeKiaNiDECfqXTRrDFWWw", "QHbI00BzQUeUGwd1ohan2g",
        "QK70KpLuZ0O43BBSWEZvHg", "QkRkwwFKVEar0WtcuvJsZg",
        "S22FCqKDWUyymN1YbpItIw", "SgKY30fjT0G8e3kQL0RHSQ",
        "SwfEwuGDlkWYy9pb4H00eQ", "TGZ5Wq0SCUCOXPsfee3uMQ",
        "TRSNunEou0aqmBCGIC8B7A", "Tgtk8jp1zkqmKzsmdrKt0g",
        "Uc4dmejgWUCTvn0XZbMTBw", "V0JqgjlrB0Ke9y91r0rxNw",
        "V6Mw0Ti8RUSkzvMGB4WGiw", "VJfVUwJs4k25ziMNvJb43A",
        "VNgRsWxMdkaUx1gKV9W1Zw", "VNkNShj97UajHDVbSmIG0g",
        "VgSXlJJEI02dytkMm5UTzA", "WH0jN3PY40es2S0LsxmkkQ",
        "WJC7Ri8s0E2qIrgvcXtoiQ", "WYKA2zSbcE2gRBPHvMLQZw",
        "XTqo1aOXDEiqyWTFK99I5Q", "XXz0qkUSd0GT4dsywxpfJg",
        "XuguA1uJAE0bWT0xXRDdeA", "Xv7jH2KcBEWqdpbT000HQA",
        "Y91uE2tRg0SUB2a9XjT30w", "ZBc1WgCwmEa2M5gKDuVWkw",
        "ZJYUt0O0AEKSQ9LJ8z7t0A", "ZWq0X5sPiUe0lQjZdCPSeQ",
        "afHuFvflAkaH7Pr85kheUQ", "hF9Lfjcn9kGQ4430uAbINA",
        "hVqUmGvIlkO0LBUoE06Q3w", "hvY1v0viv0yMdAXKa2y1aw",
        "i2BU43nmM00qbI1MKr2njQ", "iOyPaxX11UOaUkcscKhq1A",
        "it28X10JfUOAHjBiv6JIwA", "iyuYIcszXku7BWYOOwqh5Q",
        "jZfooJf57k24ANUbn1Vv0Q", "jtYTdZm3lUmFVNibJWq8YQ",
        "kGFOHthwrUWqHCLkBYIQnA", "kHsrUmyfT0O0RYtusGuQyQ",
        "kOGPX6kHskOaKSZdLGNz8A", "kZogUfYct0qMwSqvRTwOVg",
        "kg9YJVP17k2YTFuPQTOsdA", "khqZY41lNEyIvMf2ZNJuVA",
        "kwolfjBXWEOJmdbDdFoTVQ", "kz23X4pDSEiPmWtw2Qx00Q",
        "lH20r2skzU02a647xYoFoA", "lV0kQN0hR0ySmE0bQhuYbw",
        "lxLsvT1rY0mdtZuRogM06Q", "lzEZpNXmy0KWtSGT6td00g",
        "m4789fdio0q92zjf9gvh1Q", "m6ciKvH6AEe7Yi2ptKjj1g",
        "mXuYaMbcZka0TpdHDdTlWA", "mljUO9k4gUiYCHgXSsfo5A",
        "n1ikw2tG8k08ElWNxbamcw", "nV9v63E5CUKtKTjha8dtdQ",
        "nlSSRl4k50Gq2mIRYlMBCg", "nyT6FUUdu0W8UpvjP1rRgQ",
        "oQK3bVihI0qel9As840Zzw", "ogDXK0NV0EuImVvsG9kG9w",
        "oxSdYr0mzESqEpSyHlztug", "pG9Bikvr40Ct1kUtch95kw",
        "pmzLFdKTzEixfU0l0furvA", "qVd1gtwQ0k2lVRqPVp1NKQ",
        "qkGW9Frbs0Gq5gdfsztA2g", "ql7yTcebJU20VE5qpW0kCA",
        "s0oJqM6Y1UCHSVmHmgsx4Q", "sKXucFp4FUCJKo5d0G54Og",
        "sRCs6SKo6kC0xire475q0A", "sYQK1rJlwU2XCy20n0Sx9g",
        "sZ9Egg0YjkuD87ykt0O0ng", "sx6mxKBQpkq520m64BwUdQ",
        "tXQY0lmGekumoTRzAbpt4w", "tl5UFRCXZkyL6EoWVl09xA",
        "u0X5XB2KzEGduXtfWfjGDw", "uGr0ksRjB0aB0kPJw0ugQA",
        "uLl640xveUuHp2kEtOCTeQ", "ukBE9mlUikGHCbHsIuflsA",
        "urm5FZsuoEGEayow6ckb0w", "vjW8NP6JgUC3ved1NRJwbQ",
        "w0QrMwsZ5kGoJjRWvP0iKg", "w5XDRNfB2Ei2UoUDtrTkzg",
        "wlAXM0iDgkm06i7UdRww1w", "wr54weKzNkOcZ07hB04kzA",
        "xFk7igecuke0R5JMfREyDg", "xSh5yXWiP02K0UkYdkZ0cA",
        "xYuPW7feGkyc8a1rfcDv9w", "xb7eJJF0Vki6Yl3y4B7oJA",
        "yS7NgOlleU600sZ80T9bng", "yxKXWLezBUeCpdYkIQOT0A",
        "yyvS0Ljh1k0AHMx6cxtNyg", "z4PqfBhq2E01IDBkTH0gnw",
        "zONvcX0qYkuaAViV5PXcYg", "zOiK6oSHp0ib3nHvzLKbRQ",
        "zddPqpp1a06xttKdc0iNTA", "zpiSOkxpHkCs5SqdOo5ZIQ",
        "zzCIhneJ8UKTWZ48U0kRXw",
    ],
    "doubletalk": [
        "49IIo03GZ0CYQOmeA3A0BA", "7GTxyTksSUqCnP5y0ILG4A",
        "Hp5g1asacUCt5rJVLO1FuQ", "I2bme08keUmAnyJRKNYDGQ",
        "IrQvqOTCmEWMXn9k2ICtRQ", "Ixf70mgKwkCoFYq32586cw",
        "Je6gJ7y1PECStwxnrOe9aA", "JjCzlhn3gEiBQvfJtPNJ9A",
        "KSN5Jrzo7kaixP0z8xfr4Q", "LHsrJBRGnUKiMC2mihEr0g",
        "LN18k5r8t00C9DulUd809A", "Lsa5WpwTpUeb7C9dc9RXuQ",
        "N2rQLbnp2UOg2QFRaggbDw", "N94NwopiZEyNnraWHLMDcg",
        "OXCtw0FVhUWGUBil6Uucdw", "OmB0Ht0hmE2crVnftAEtsw",
        "QEeKiaNiDECfqXTRrDFWWw", "QK70KpLuZ0O43BBSWEZvHg",
        "QkRkwwFKVEar0WtcuvJsZg", "S22FCqKDWUyymN1YbpItIw",
        "SgKY30fjT0G8e3kQL0RHSQ", "SwfEwuGDlkWYy9pb4H00eQ",
        "TGZ5Wq0SCUCOXPsfee3uMQ", "Tgtk8jp1zkqmKzsmdrKt0g",
        "V0JqgjlrB0Ke9y91r0rxNw", "V6Mw0Ti8RUSkzvMGB4WGiw",
        "VNkNShj97UajHDVbSmIG0g", "VgSXlJJEI02dytkMm5UTzA",
        "W0J6iZv7ZkmHOobCToob4A", "W0zK3dv0QE2YckPArTGXCg",
        "W4r0UCjieEuM0u930spvug", "W6eXdzuIPkaPtxI04uwsVA",
        "WAx9ADn1O00xxkqYq0hPlg", "WH0jN3PY40es2S0LsxmkkQ",
        "WH7rA6R2zkyopKUrcq9p3A", "WLeLQyPtWk2uOefcjBdZmw",
        "WcK0OrF6ukW03fViPXTQjQ", "WnDjVFWmC0m0WhVq22mRlQ",
        "WqEYNwalSUebZxaeYVay2g", "Wv6yp6N1L0WqQ6ZLn6nD8g",
        "X7Ua9txMj0aws848JPEbOg", "XCXcCwUPY0GmrtqtJ6xY2g",
        "XGDaZuEkE0WU4IN0Yi4XtA", "XRTnTUjU5kS0mejzCqyCiw",
        "XTqo1aOXDEiqyWTFK99I5Q", "XV5L2dn3S06M9GBEu1q3DA",
        "XnfMDZLl0U2WvLRphiGJ6A", "XqvGR01tJkan17zltLs38Q",
        "XuiheB7eUkyJA2XzFIovHQ", "Xvyiz1o0cEijZQ8DT9mB2w",
        "XzmU8Bbs70WtSj0koaWIzw", "ZJYUt0O0AEKSQ9LJ8z7t0A",
        "afHuFvflAkaH7Pr85kheUQ", "hF9Lfjcn9kGQ4430uAbINA",
        "hvY1v0viv0yMdAXKa2y1aw", "i2BU43nmM00qbI1MKr2njQ",
        "iOyPaxX11UOaUkcscKhq1A", "it28X10JfUOAHjBiv6JIwA",
        "iyuYIcszXku7BWYOOwqh5Q", "jZfooJf57k24ANUbn1Vv0Q",
        "jtYTdZm3lUmFVNibJWq8YQ", "kGFOHthwrUWqHCLkBYIQnA",
        "kOGPX6kHskOaKSZdLGNz8A", "kZogUfYct0qMwSqvRTwOVg",
        "kg9YJVP17k2YTFuPQTOsdA", "khqZY41lNEyIvMf2ZNJuVA",
        "kwolfjBXWEOJmdbDdFoTVQ", "kz23X4pDSEiPmWtw2Qx00Q",
        "lH20r2skzU02a647xYoFoA", "lV0kQN0hR0ySmE0bQhuYbw",
        "m6ciKvH6AEe7Yi2ptKjj1g", "mljUO9k4gUiYCHgXSsfo5A",
        "nlSSRl4k50Gq2mIRYlMBCg", "nyT6FUUdu0W8UpvjP1rRgQ",
        "oQK3bVihI0qel9As840Zzw", "qVd1gtwQ0k2lVRqPVp1NKQ",
        "qkGW9Frbs0Gq5gdfsztA2g", "ql7yTcebJU20VE5qpW0kCA",
        "s0oJqM6Y1UCHSVmHmgsx4Q", "sKXucFp4FUCJKo5d0G54Og",
        "sRCs6SKo6kC0xire475q0A", "sYQK1rJlwU2XCy20n0Sx9g",
        "sx6mxKBQpkq520m64BwUdQ", "tl5UFRCXZkyL6EoWVl09xA",
        "u0X5XB2KzEGduXtfWfjGDw", "uLl640xveUuHp2kEtOCTeQ",
        "urm5FZsuoEGEayow6ckb0w", "vjW8NP6JgUC3ved1NRJwbQ",
        "w0QrMwsZ5kGoJjRWvP0iKg", "w0ogzwvJ7EmiHTCzx7sgwA",
        "w5XDRNfB2Ei2UoUDtrTkzg", "wHmBm7VHfkysBOhjoAXkNA",
        "wVYSGVTTakih9twI4xlDWQ", "wWeNtFK0dEG9Wub40bB15A",
        "wY00iJ3cE0aQsjt0m1tC0g", "wZvBJ5R4REKr7IitKv9FIw",
        "waxU019BVEacr7vK6v00mQ", "wlAXM0iDgkm06i7UdRww1w",
        "wr2CEenGL0Oec4b1GW5Zww", "xFk7igecuke0R5JMfREyDg",
        "xNr7L0xsLUG4B9oUqW0V4Q", "xSh5yXWiP02K0UkYdkZ0cA",
        "xYuPW7feGkyc8a1rfcDv9w", "xb7eJJF0Vki6Yl3y4B7oJA",
        "xnpFE06ShUea4Jn1Wu7EzQ", "xofDX004bkqiOv9YOxmGVQ",
        "xuKL15aeq0CZpaYTrP9V4w", "xvACDxradUuKNYImFSd1ww",
        "y2ZCo1jA6kGdWZ0MgoaZ5w", "yc5bFUGsR0GSfiGwTTpRWg",
        "zONvcX0qYkuaAViV5PXcYg", "zOiK6oSHp0ib3nHvzLKbRQ",
        "zpiSOkxpHkCs5SqdOo5ZIQ", "zzCIhneJ8UKTWZ48U0kRXw",
    ],
}

# --- ICASSP 2023 configuration ---

ICASSP2023_DATASET_PATH = "datasets/blind_test_set_icassp2023"
ICASSP2023_OUT_DIR = "aec_challenge_blind_icassp2023"

ICASSP2023_SCENARIOS = {
    "farend_singletalk": {
        "gh_dir": "farend-singletalk",
    },
    "nearend_singletalk": {
        "gh_dir": "nearend-singletalk",
    },
    "doubletalk": {
        "gh_dir": "doubletalk",
    },
}


def fetch_uuids_from_github(base_api_url, gh_dir, scenario_name):
    """Fetch file list from GitHub API and extract UUIDs for a scenario.

    Calls the GitHub Contents API to list files in the given scenario directory.
    Filters out enrollment (enrl) files used for personal AEC tasks, keeps only
    _mic.wav and _lpb.wav files, and extracts unique UUIDs from the filenames.

    Args:
        base_api_url: GitHub API base URL, e.g.
            "https://api.github.com/repos/microsoft/AEC-Challenge/contents/datasets/blind_test_set_icassp2023"
        gh_dir: Subdirectory name on GitHub (e.g. "farend-singletalk")
        scenario_name: Scenario key with underscores (e.g. "farend_singletalk")

    Returns:
        (uuids, suffixes): tuple of
            - sorted list of unique UUID strings
            - list of suffix strings (e.g. ["_farend_singletalk_mic.wav", ...])
    """
    api_url = f"{base_api_url}/{gh_dir}"
    print(f"  Fetching file list from GitHub API: {api_url}")

    req = urllib.request.Request(api_url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "AEC-Challenge-Downloader")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: Failed to fetch file list from GitHub API: {e}")
        return [], []

    # Collect filenames, skip enrl files
    filenames = []
    for entry in data:
        name = entry.get("name", "")
        if not name.endswith(".wav"):
            continue
        if "enrl" in name:
            continue
        filenames.append(name)

    # Extract (uuid, suffixes) pairs from mic files
    # Handles both normal and movement patterns:
    #   {UUID}_{gh_dir}_mic.wav
    #   {UUID}_{gh_dir}-with-movement_mic.wav
    mic_files = [f for f in filenames if f.endswith("_mic.wav")]
    uuid_suffix_map = {}  # uuid → [mic_suffix, lpb_suffix]
    for mf in mic_files:
        # Find the suffix: everything from the first _ that matches the scenario
        # e.g. "ABC_farend-singletalk_mic.wav" → suffix = "_farend-singletalk_mic.wav"
        #      "ABC_farend-singletalk-with-movement_mic.wav" → suffix = "_farend-singletalk-with-movement_mic.wav"
        idx = mf.find(f"_{gh_dir}")
        if idx < 0:
            continue
        uuid = mf[:idx]
        mic_suffix = mf[idx:]
        lpb_suffix = mic_suffix.replace("_mic.wav", "_lpb.wav")
        uuid_suffix_map[uuid] = [mic_suffix, lpb_suffix]

    # Return as list of (uuid, suffixes) sorted by uuid
    items = sorted(uuid_suffix_map.items())
    uuids = [u for u, _ in items]
    # For download_scenario compatibility, group by suffix pattern
    # Return all items as a flat list with per-uuid suffixes
    print(f"  Found {len(uuids)} UUIDs for {scenario_name} (excluding enrl files)")
    return items  # list of (uuid, [mic_suffix, lpb_suffix])


def download(url, dest):
    """Download a single file via curl. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    ret = subprocess.run(
        ["curl", "-sL", "-o", dest, "-w", "%{http_code}", url],
        capture_output=True, text=True, timeout=120
    )
    code = ret.stdout.strip()
    if code == "200" and os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    else:
        print(f"  FAIL ({code}): {os.path.basename(dest)}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_scenario(sc_name, sc_info, uuids, base_url, out_base, max_uuids=None):
    """Download files for a scenario."""
    if max_uuids:
        uuids = uuids[:max_uuids]

    sc_dir = os.path.join(out_base, sc_name)
    os.makedirs(sc_dir, exist_ok=True)

    print(f"\n{sc_name}: downloading {len(uuids)} cases")
    ok = 0
    for uuid in uuids:
        all_ok = True
        for suffix in sc_info["suffixes"]:
            fname = f"{uuid}{suffix}"
            encoded = urllib.parse.quote(fname)
            url = f"{base_url}/{sc_info['gh_dir']}/{encoded}"
            dest = os.path.join(sc_dir, fname)
            print(f"  {fname} ... ", end="", flush=True)
            if download(url, dest):
                print("OK")
            else:
                all_ok = False
        if all_ok:
            ok += 1
    print(f"  => {ok}/{len(uuids)} cases downloaded")
    return ok


def run_interspeech2021(args):
    """Download the Interspeech 2021 blind test set."""
    base_url = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{INTERSPEECH2021_DATASET_PATH}"
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), INTERSPEECH2021_OUT_DIR)

    # Download non-movement files
    if not args.movement_only:
        for sc_name, sc_info in SCENARIOS.items():
            uuids = UUIDS.get(sc_name, [])
            download_scenario(sc_name, sc_info, uuids, base_url, out_base, args.max)

    # Download movement files
    if not args.no_movement:
        for sc_name, sc_info in MOVEMENT_SCENARIOS.items():
            uuids = MOVEMENT_UUIDS.get(sc_name, [])
            label = f"{sc_name} (movement)"
            if args.max:
                uuids = uuids[:args.max]

            sc_dir = os.path.join(out_base, sc_name)
            os.makedirs(sc_dir, exist_ok=True)

            print(f"\n{label}: downloading {len(uuids)} cases")
            ok = 0
            for uuid in uuids:
                all_ok = True
                for suffix in sc_info["suffixes"]:
                    fname = f"{uuid}{suffix}"
                    encoded = urllib.parse.quote(fname)
                    url = f"{base_url}/{sc_info['gh_dir']}/{encoded}"
                    dest = os.path.join(sc_dir, fname)
                    print(f"  {fname} ... ", end="", flush=True)
                    if download(url, dest):
                        print("OK")
                    else:
                        all_ok = False
                if all_ok:
                    ok += 1
            print(f"  => {ok}/{len(uuids)} movement cases downloaded")

    print(f"\nDone! Files in {out_base}")


def run_icassp2023(args):
    """Download the ICASSP 2023 blind test set."""
    base_api_url = f"https://api.github.com/repos/{REPO}/contents/{ICASSP2023_DATASET_PATH}"
    base_url = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}/{ICASSP2023_DATASET_PATH}"
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), ICASSP2023_OUT_DIR)

    print("ICASSP 2023 blind test set — querying GitHub API for file lists...\n")

    for sc_name, sc_cfg in ICASSP2023_SCENARIOS.items():
        gh_dir = sc_cfg["gh_dir"]
        items = fetch_uuids_from_github(base_api_url, gh_dir, sc_name)
        if not items:
            print(f"  Skipping {sc_name}: no UUIDs found")
            continue

        if args.max:
            items = items[:args.max]

        sc_dir = os.path.join(out_base, sc_name)
        os.makedirs(sc_dir, exist_ok=True)

        print(f"\n{sc_name}: downloading {len(items)} cases")
        ok = 0
        for uuid, suffixes in items:
            all_ok = True
            for suffix in suffixes:
                fname = f"{uuid}{suffix}"
                encoded = urllib.parse.quote(fname)
                url = f"{base_url}/{gh_dir}/{encoded}"
                dest = os.path.join(sc_dir, fname)
                print(f"  {fname} ... ", end="", flush=True)
                if download(url, dest):
                    print("OK")
                else:
                    all_ok = False
            if all_ok:
                ok += 1
        print(f"  => {ok}/{len(items)} cases downloaded")

    print(f"\nDone! Files in {out_base}")


def main():
    parser = argparse.ArgumentParser(
        description='Download AEC Challenge blind test sets from GitHub LFS')
    parser.add_argument('--dataset', choices=['interspeech2021', 'icassp2023'],
                        default='interspeech2021',
                        help='Which blind test set to download (default: interspeech2021)')
    parser.add_argument('--max', type=int, default=None,
                        help='Max UUIDs per scenario')
    parser.add_argument('--no-movement', action='store_true',
                        help='Skip movement files (Interspeech 2021 only)')
    parser.add_argument('--movement-only', action='store_true',
                        help='Only download movement files (Interspeech 2021 only)')
    args = parser.parse_args()

    if args.dataset == 'interspeech2021':
        run_interspeech2021(args)
    elif args.dataset == 'icassp2023':
        run_icassp2023(args)


if __name__ == "__main__":
    main()
