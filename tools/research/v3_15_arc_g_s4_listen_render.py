#!/usr/bin/env python3
"""v3.15 §1.4.S4 Arc G — listen pack render: 5 candidates + 5 controls × ON/OFF.

Renders nores + full output for 10 cases × 2 configurations.  Per §0.6
linear-filter rule, the nores listen on these renders is the PRIMARY
metric channel for Arc G acceptance.  Output lands at
listen/v3_15_arc_g_s4/<config>/<stem>_{ours,ours_nores}.wav.

Configs:
  off — arc_g_per_band_w_reset=False (default; baseline)
  on  — arc_g_per_band_w_reset=True (default threshold 4.0)

Usage:
    python3 tools/research/v3_15_arc_g_s4_listen_render.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))

# Top 5 candidates from v3_15_arc_g_fire_audit (drift_ratio=4.0, 30 cases).
# Mix: 2 DT_static (highest fires), 1 FS_movement, 1 FS_static, 1 DT_movement.
CANDIDATES = [
    ('MeQ3WL4hykKuT2761h0xFg', 'doubletalk',          False),  # 20 fires
    ('QkRkwwFKVEar0WtcuvJsZg', 'doubletalk',          False),  # 19 fires
    ('OX2l6zV7nkmmSkVA3ETLKg', 'farend_singletalk',   True),   # 15 fires (movement)
    ('Y91uE2tRg0SUB2a9XjT30w', 'farend_singletalk',   False),  # 13 fires
    ('WH0jN3PY40es2S0LsxmkkQ', 'doubletalk',          True),   # 10 fires (movement)
]
# Control 5: zero-fire from same audit.
CONTROLS = [
    ('SgKY30fjT0G8e3kQL0RHSQ', 'farend_singletalk',   True),   # FS movement, 0 fires
    ('zOiK6oSHp0ib3nHvzLKbRQ', 'farend_singletalk',   True),   # FS movement, 0 fires
    ('NN7yhG2XTEqq46X8X0yLfA', 'doubletalk',          False),  # DT, 0 fires
    ('ql7yTcebJU20VE5qpW0kCA', 'doubletalk',          False),  # DT, 0 fires
    ('s90M7MOTBkqaV4nQPLhKbA', 'doubletalk',          False),  # DT, 0 fires
]

CONFIGS = [
    ('off', False),
    ('on',  True),
]


def stem_to_full(stem: str, bucket: str, is_movement: bool) -> str:
    if is_movement:
        return f"{stem}_{bucket}_with_movement"
    return f"{stem}_{bucket}"


def render(stem_short: str, bucket: str, is_movement: bool,
           cfg_label: str, arc_g_on: bool, dataset_dir: Path, out_dir: Path):
    from aec import AEC, AecConfig, AecPreset

    full_stem = stem_to_full(stem_short, bucket, is_movement)
    mic_path = dataset_dir / bucket / f"{full_stem}_mic.wav"
    lpb_path = dataset_dir / bucket / f"{full_stem}_lpb.wav"
    if not mic_path.exists():
        print(f"  MISSING: {mic_path}")
        return None

    mic, sr = sf.read(str(mic_path))
    lpb, _ = sf.read(str(lpb_path))
    n = min(len(mic), len(lpb))
    mic = mic[:n].astype(np.float32)
    lpb = lpb[:n].astype(np.float32)

    cfg = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr,
        filter_length=832,
        enable_res=True,
        enable_cng=True,
    )
    cfg.arc_g_per_band_w_reset = arc_g_on

    np.random.seed(0)
    aec = AEC(cfg)
    hop = aec.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos+hop] = aec.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop

    cfg_dir = out_dir / cfg_label
    cfg_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(cfg_dir / f"{full_stem}_ours.wav"), out, sr)

    # nores render — separate AEC instance
    cfg2 = AecConfig.from_preset(
        AecPreset.BALANCED,
        sample_rate=sr,
        filter_length=832,
        enable_res=False,
        enable_cng=False,
    )
    cfg2.arc_g_per_band_w_reset = arc_g_on
    np.random.seed(0)
    aec2 = AEC(cfg2)
    out2 = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out2[pos:pos+hop] = aec2.process(mic[pos:pos+hop], lpb[pos:pos+hop])
        pos += hop
    sf.write(str(cfg_dir / f"{full_stem}_ours_nores.wav"), out2, sr)

    fires = aec._arc_g_fire_count.copy() if arc_g_on else np.zeros(3)
    return {
        'stem': full_stem,
        'cfg': cfg_label,
        'fires': [int(x) for x in fires],
    }


def main():
    dataset_dir = Path('wav/aec_challenge_blind')
    out_dir = Path('listen/v3_15_arc_g_s4')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases = [(*c, 'CAND') for c in CANDIDATES] + [(*c, 'CTRL') for c in CONTROLS]
    print(f"Rendering {len(all_cases)} cases × {len(CONFIGS)} configs = "
          f"{len(all_cases) * len(CONFIGS)} renders × 2 (ours + nores) = "
          f"{len(all_cases) * len(CONFIGS) * 2} files")

    summary = []
    t0 = time.time()
    for i, (stem, bucket, is_mvmt, role) in enumerate(all_cases):
        for cfg_label, arc_g_on in CONFIGS:
            r = render(stem, bucket, is_mvmt, cfg_label, arc_g_on,
                       dataset_dir, out_dir)
            if r is not None:
                r['role'] = role
                summary.append(r)
                fires_str = '/'.join(str(x) for x in r['fires'])
                print(f"  [{i+1:>2}/{len(all_cases)}] {role} {cfg_label:>3}  "
                      f"{r['stem'][:50]:<50} fires=L/M/H={fires_str}")
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s.')

    # Save summary
    import json
    (out_dir / 'render_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Listen pack at {out_dir}/<config>/, summary at {out_dir}/render_summary.json')


if __name__ == '__main__':
    main()
