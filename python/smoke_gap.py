#!/usr/bin/env python3
"""Targeted smoke test: runs AEC only on the gap cases listed in a TSV file.

Gap case TSV format (from gap_analysis.py output):
    scenario<TAB>file_id[<TAB>gap_score]
where file_id already includes _with_movement suffix if applicable.

Usage:
    python3 smoke_gap.py <wav_root> <gap_cases_tsv> -o <out_dir> [--preset balanced] [--filter 512]
"""
import os, sys, argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aec import AEC, AecConfig

SCENARIO_MAP = {
    'farend_singletalk': 'st',
    'doubletalk':        'dt',
    'nearend_singletalk':'nst',
}

PRESET_PARAMS = {
    'balanced': dict(
        res_g_min_db=-55.0,
        res_echo_method='direct',
        res_gain_type='enr',
    ),
    'aggressive': dict(
        res_g_min_db=-80.0,
        res_echo_method='direct',
        res_gain_type='enr',
    ),
}


def build_config(sr, fl, preset):
    kw = dict(sample_rate=sr, filter_length=fl,
              enable_res=True, enable_shadow=True, enable_dtd=False)
    kw.update(PRESET_PARAMS.get(preset, {}))
    fields = set(AecConfig.__dataclass_fields__.keys())
    return AecConfig(**{k: v for k, v in kw.items() if k in fields})


def run_file(aec_inst, mic_path, lpb_path):
    mic, sr = sf.read(mic_path, dtype='float32')
    lpb, _  = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]
    hop = aec_inst.hop_size
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec_inst.process(mic[pos:pos + hop], lpb[pos:pos + hop])
        pos += hop
    return out, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav_root')
    ap.add_argument('gap_tsv')
    ap.add_argument('-o', '--out_dir', default='/tmp/gap_smoke')
    ap.add_argument('--preset', default='balanced')
    ap.add_argument('--filter', type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load gap cases
    cases = []
    with open(args.gap_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                cases.append((parts[0], parts[1]))  # (scenario, file_id)

    print(f"Gap cases: {len(cases)}")
    counts = {'farend_singletalk': 0, 'doubletalk': 0, 'nearend_singletalk': 0}
    ok = 0

    for scenario, file_id in cases:
        wav_dir  = os.path.join(args.wav_root, scenario)
        # file_id may have _with_movement suffix; actual filename is
        # {base_id}_{scenario}[_with_movement] not {file_id}_{scenario}
        if file_id.endswith('_with_movement'):
            base_id = file_id[:-len('_with_movement')]
            stem = f"{base_id}_{scenario}_with_movement"
        else:
            stem = f"{file_id}_{scenario}"
        mic_path = os.path.join(wav_dir, f"{stem}_mic.wav")
        lpb_path = os.path.join(wav_dir, f"{stem}_lpb.wav")
        out_path = os.path.join(args.out_dir, f"{stem}_ours.wav")

        if not os.path.exists(mic_path) or not os.path.exists(lpb_path):
            print(f"  MISSING: {stem}")
            continue

        sr = sf.info(mic_path).samplerate
        config = build_config(sr, args.filter, args.preset)
        aec = AEC(config)

        try:
            out, sr_out = run_file(aec, mic_path, lpb_path)
            sf.write(out_path, out, sr_out)
            counts[scenario] += 1
            ok += 1
        except Exception as e:
            print(f"  ERROR {stem}: {e}")

    print(f"Wrote {ok} files → {args.out_dir}")
    for s, n in counts.items():
        if n: print(f"  {s}: {n}")


if __name__ == '__main__':
    main()
