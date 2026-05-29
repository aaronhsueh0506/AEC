#!/usr/bin/env python3
"""Minimal byte-equal gate for the v3.21 CLOSE cleanup.

CLAUDE.md references a check_byte_equal.py + docs/bench anchor that are not
present in the v3_21_release branch. This is a self-contained substitute: it
renders a fixed, representative case set with the standard BALANCED config
(filter=832 / cng / hop=160) and md5s the float32 output of both the full
(`_ours`) and linear-only (`_ours_nores`) paths.

Usage:
    python3 python/v3_21_byte_equal_check.py --save /tmp/be_baseline.json
    # ... make cleanup edits ...
    python3 python/v3_21_byte_equal_check.py --check /tmp/be_baseline.json

A clean cleanup (inline default-ON flags, drop default-OFF flags) MUST print
`=== N/N PASS ===`. Any md5 drift means a flag disposition changed behaviour.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig, AecPreset
from v3_21_800case_bench import _estimate_delay, _discover_cases, SR

# Fixed representative set (covers every bucket + a movement case for the
# delay path + the 9xjhi nores LF artifact case). Resolved against the corpus
# by substring; falls back to the first sorted stems per bucket if absent.
PREFERRED = {
    'doubletalk': ['xFk7igec', 'wVYSGVTT', 'nVUnxqHLr', 'XRTnTUjU'],   # 2 movement + 2 static
    'farend_singletalk': ['9xjhiFbGo', 'qNvSMyUS', '0I0XMl3M'],         # +1 movement-ish
    'nearend_singletalk': [],   # first-N fallback
}
N_PER_BUCKET = 4
SUBDIR_OF = {'doubletalk': 'doubletalk', 'farend_singletalk': 'farend_singletalk',
             'nearend_singletalk': 'nearend_singletalk'}


def _select_cases():
    cases = _discover_cases()
    by_sub = {}
    for stem, subdir, bucket, wavtype, mic_path, ref_path in cases:
        by_sub.setdefault(subdir, []).append((stem, subdir, wavtype, mic_path, ref_path))
    selected = []
    for subdir in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        pool = sorted(by_sub.get(subdir, []), key=lambda t: t[0])
        picked, seen = [], set()
        for pref in PREFERRED.get(subdir, []):
            for t in pool:
                if pref in t[0] and t[0] not in seen:
                    picked.append(t); seen.add(t[0]); break
        for t in pool:
            if len(picked) >= N_PER_BUCKET:
                break
            if t[0] not in seen:
                picked.append(t); seen.add(t[0])
        selected.extend(picked[:N_PER_BUCKET])
    return selected


def _render(mic, ref_a, is_mvmt, enable_res):
    kw = dict(enable_res=enable_res)
    if is_mvmt:
        kw.update(enable_delay_est=True, delay_est_period_s=0.25, delay_est_init_s=0.2)
    else:
        kw['enable_delay_est'] = False
    cfg = AecConfig.from_preset(AecPreset.BALANCED, **kw)
    np.random.seed(0)
    aec = AEC(cfg)
    hop = cfg.hop_size
    n_hops = len(mic) // hop
    out = np.zeros(n_hops * hop, dtype=np.float32)
    for i in range(n_hops):
        s = i * hop
        out[s:s + hop] = aec.process(mic[s:s + hop], ref_a[s:s + hop])
    return out


def _md5(x):
    return hashlib.md5(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def _snapshot():
    out = {}
    for stem, subdir, wavtype, mic_path, ref_path in _select_cases():
        mic_raw, _ = sf.read(mic_path)
        ref_raw, _ = sf.read(ref_path)
        if mic_raw.ndim > 1:
            mic_raw = mic_raw[:, 0]
        if ref_raw.ndim > 1:
            ref_raw = ref_raw[:, 0]
        n = min(len(mic_raw), len(ref_raw))
        mic = mic_raw[:n].astype(np.float32)
        ref = ref_raw[:n].astype(np.float32)
        is_mvmt = '_with_movement' in stem
        delay = _estimate_delay(mic, ref, SR)
        if 0 < delay < n:
            ref_a = np.zeros(n, dtype=np.float32)
            ref_a[delay:] = ref[:n - delay]
        else:
            ref_a = ref.copy()
        m_ours = _md5(_render(mic, ref_a, is_mvmt, enable_res=True))
        m_nores = _md5(_render(mic, ref_a, is_mvmt, enable_res=False))
        out[stem] = {'ours': m_ours, 'nores': m_nores}
        print(f"  {stem[:42]:42s} ours={m_ours[:12]} nores={m_nores[:12]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save')
    ap.add_argument('--check')
    args = ap.parse_args()
    print(f"[byte-equal] rendering representative set ...")
    snap = _snapshot()
    if args.save:
        json.dump(snap, open(args.save, 'w'), indent=2)
        print(f"[saved] {args.save}  ({len(snap)} cases)")
    if args.check:
        base = json.load(open(args.check))
        n_ok = n_fail = 0
        for stem, md in snap.items():
            b = base.get(stem)
            if b and b['ours'] == md['ours'] and b['nores'] == md['nores']:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL {stem}: ours {b and b['ours'][:10]}->{md['ours'][:10]} "
                      f"nores {b and b['nores'][:10]}->{md['nores'][:10]}")
        total = n_ok + n_fail
        print(f"\n=== {n_ok}/{total} PASS, {n_fail} FAIL ===")
        sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
