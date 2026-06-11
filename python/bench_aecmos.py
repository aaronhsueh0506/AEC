#!/usr/bin/env python3
"""AECMOS bench analyzer — score AEC outputs against blind dataset.

Usage:
    python3 bench_aecmos.py <output_dir> <result_dir> [--baseline <baseline.json>]

Walks <output_dir> for `<stem>_ours.wav` files, finds the matching
`<stem>_mic.wav` / `<stem>_lpb.wav` under wav/aec_challenge_blind/<scenario>/,
runs the local AECMOS ONNX (model/Run_1663915512_Stage_0.onnx), aggregates
per-bucket means, and dumps per-case scores + worst-N lists.

Buckets (matches review acceptance criteria):
  FS_static   — farend_singletalk, no _with_movement_
  FS_movement — farend_singletalk, _with_movement_
  NE          — nearend_singletalk
  DT_static   — doubletalk, no _with_movement_
  DT_movement — doubletalk, _with_movement_

Outputs (in <result_dir>):
  scores.json — per-case dict {stem: {bucket, echo, deg}}
  result.md   — bucket means + worst-20 per bucket; if --baseline given,
                also Δ vs baseline + Pareto verdict

Speed: single ONNX session reused across cases (FastAECMOS pattern).
"""
import os
import sys
import json
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_MODEL_DIR = os.path.join(_REPO, 'model')
sys.path.insert(0, _MODEL_DIR)

import onnxruntime as ort
import librosa


class FastAECMOS:
    """Reusable AECMOS estimator (single ONNX session)."""

    def __init__(self, model_path):
        self.sampling_rate = 16000
        self.dft_size = 512
        self.hop_fraction = 0.5
        self.hidden_size = (4, 1, 64)
        self.max_len = 20
        # Cap ONNX to 1 thread/instance so N bench workers map 1:1 onto cores
        # (uncapped, each session spawns ~ncores intra-op threads → heavy
        # oversubscription — the reason plain --workers 9 doesn't speed up).
        # Thread count does NOT affect deterministic inference output.
        _so = ort.SessionOptions()
        _so.intra_op_num_threads = 1
        _so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options=_so)
        self.input_name = self.session.get_inputs()[0].name

    def _mel(self, x):
        m = librosa.feature.melspectrogram(
            y=x, sr=self.sampling_rate,
            n_fft=self.dft_size + 1,
            hop_length=int(self.hop_fraction * self.dft_size),
            n_mels=160,
        )
        return ((librosa.power_to_db(m, ref=np.max) + 40) / 40).T

    def score(self, talk_type, lpb, mic, enh):
        seg = self.max_len * self.sampling_rate
        if len(lpb) >= seg:
            lpb, mic, enh = lpb[:seg], mic[:seg], enh[:seg]
        L, M, E = self._mel(lpb), self._mel(mic), self._mel(enh)
        if talk_type == 'nst':
            ne_st, fe_st = 1, 0
        elif talk_type == 'st':
            ne_st, fe_st = 0, 1
        else:  # dt
            ne_st, fe_st = 0, 0
        M = np.concatenate(
            (M, np.ones((20, M.shape[1])) * (1 - fe_st), np.zeros((20, M.shape[1]))), axis=0
        )
        L = np.concatenate(
            (L, np.ones((20, L.shape[1])) * (1 - ne_st), np.zeros((20, L.shape[1]))), axis=0
        )
        E = np.concatenate(
            (E, np.ones((20, E.shape[1])), np.zeros((20, E.shape[1]))), axis=0
        )
        feats = np.expand_dims(np.stack((L, M, E)).astype(np.float32), 0)
        h0 = np.zeros(self.hidden_size, dtype=np.float32)
        out = self.session.run([], {self.input_name: feats, 'h0': h0})[0]
        return float(out[0]), float(out[1])


_BUCKET_RE = re.compile(
    r'^(?P<stem>.+?)_(?P<scenario>farend_singletalk|nearend_singletalk|doubletalk)'
    r'(?P<mv>_with_movement)?_ours\.wav$'
)


def classify(filename):
    """Return (bucket, talk_type, full_stem) or None."""
    m = _BUCKET_RE.match(filename)
    if not m:
        return None
    scenario = m.group('scenario')
    is_mv = bool(m.group('mv'))
    stem = m.group('stem') + '_' + scenario + (m.group('mv') or '')
    if scenario == 'farend_singletalk':
        bucket = 'FS_movement' if is_mv else 'FS_static'
        talk_type = 'st'
    elif scenario == 'nearend_singletalk':
        bucket = 'NE'
        talk_type = 'nst'
    else:
        bucket = 'DT_movement' if is_mv else 'DT_static'
        talk_type = 'dt'
    return bucket, talk_type, stem


def find_dataset_paths(stem, dataset_root):
    """Locate <stem>_mic.wav / <stem>_lpb.wav under dataset_root."""
    if 'farend_singletalk' in stem:
        scenario_dir = 'farend_singletalk'
    elif 'nearend_singletalk' in stem:
        scenario_dir = 'nearend_singletalk'
    else:
        scenario_dir = 'doubletalk'
    mic = os.path.join(dataset_root, scenario_dir, stem + '_mic.wav')
    lpb = os.path.join(dataset_root, scenario_dir, stem + '_lpb.wav')
    return mic, lpb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir', help='Directory containing <stem>_ours.wav files')
    ap.add_argument('result_dir', help='Where to write scores.json + result.md')
    ap.add_argument('--dataset', default=os.path.join(_REPO, 'wav/aec_challenge_blind'))
    ap.add_argument('--model', default=os.path.join(_MODEL_DIR, 'Run_1663915512_Stage_0.onnx'))
    ap.add_argument('--baseline', default=None,
                    help='baseline scores.json to compute Δ against')
    ap.add_argument('--label', default='current',
                    help='label for this run (used in result.md headers)')
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    estimator = FastAECMOS(args.model)

    cases = []
    for f in sorted(os.listdir(args.output_dir)):
        if not f.endswith('_ours.wav'):
            continue
        c = classify(f)
        if not c:
            continue
        bucket, talk_type, stem = c
        mic_p, lpb_p = find_dataset_paths(stem, args.dataset)
        if not (os.path.isfile(mic_p) and os.path.isfile(lpb_p)):
            continue
        cases.append({
            'stem': stem, 'bucket': bucket, 'talk_type': talk_type,
            'mic': mic_p, 'lpb': lpb_p,
            'enh': os.path.join(args.output_dir, f),
        })

    print(f"Scoring {len(cases)} cases...", flush=True)
    scores = {}
    bucket_acc = defaultdict(lambda: {'echo': [], 'deg': []})
    for i, c in enumerate(cases):
        mic, _ = sf.read(c['mic'])
        lpb, _ = sf.read(c['lpb'])
        enh, _ = sf.read(c['enh'])
        n = min(len(mic), len(lpb), len(enh))
        mic = mic[:n].astype(np.float32)
        lpb = lpb[:n].astype(np.float32)
        enh = enh[:n].astype(np.float32)
        echo, deg = estimator.score(c['talk_type'], lpb, mic, enh)
        scores[c['stem']] = {'bucket': c['bucket'], 'echo': echo, 'deg': deg}
        bucket_acc[c['bucket']]['echo'].append((c['stem'], echo))
        bucket_acc[c['bucket']]['deg'].append((c['stem'], deg))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}", flush=True)

    # Aggregate
    summary = {}
    for bucket, d in bucket_acc.items():
        echo_vals = [v for _, v in d['echo']]
        deg_vals = [v for _, v in d['deg']]
        summary[bucket] = {
            'n': len(echo_vals),
            'echo_mean': float(np.mean(echo_vals)),
            'deg_mean': float(np.mean(deg_vals)),
        }

    out = {'label': args.label, 'summary': summary, 'scores': scores}
    json.dump(out, open(os.path.join(args.result_dir, 'scores.json'), 'w'),
              indent=2)

    # result.md
    md = [f"# Bench result — {args.label}", '']
    md.append(f"Dataset: {args.dataset}")
    md.append(f"Output dir: {args.output_dir}")
    md.append(f"Cases scored: {len(cases)}")
    md.append('')
    md.append('## Bucket means')
    md.append('')
    md.append('| Bucket | n | echo (↑) | deg (↑) |')
    md.append('|---|---:|---:|---:|')
    bucket_order = ['FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE']
    for b in bucket_order:
        if b not in summary:
            continue
        s = summary[b]
        md.append(f"| {b} | {s['n']} | {s['echo_mean']:.3f} | {s['deg_mean']:.3f} |")
    md.append('')

    if args.baseline:
        bl = json.load(open(args.baseline))
        bl_scores = bl.get('scores', {})
        md.append(f"## Δ vs baseline ({bl.get('label', args.baseline)})")
        md.append('')
        md.append('| Bucket | n | n_bl | n_common | Δecho | Δdeg | verdict |')
        md.append('|---|---:|---:|---:|---:|---:|---|')
        for b in bucket_order:
            if b not in summary:
                continue
            cur_stems = {s: d for s, d in scores.items() if d['bucket'] == b}
            bl_stems = {s: d for s, d in bl_scores.items() if d.get('bucket') == b}
            common = sorted(set(cur_stems) & set(bl_stems))
            n_cur, n_bl = len(cur_stems), len(bl_stems)
            if not common:
                md.append(f"| {b} | {n_cur} | {n_bl} | 0 | n/a | n/a | NO COMMON STEMS |")
                continue
            de = float(np.mean([cur_stems[s]['echo'] - bl_stems[s]['echo'] for s in common]))
            dd = float(np.mean([cur_stems[s]['deg'] - bl_stems[s]['deg'] for s in common]))
            verdict = []
            if b in ('FS_static', 'FS_movement') and de < -0.02:
                verdict.append('FS echo regress')
            if b == 'NE' and dd < -0.01:
                verdict.append('NE deg regress')
            if n_cur != n_bl:
                verdict.append(f'n mismatch {n_cur}≠{n_bl}')
            md.append(f"| {b} | {n_cur} | {n_bl} | {len(common)} | "
                      f"{de:+.3f} | {dd:+.3f} | {' ; '.join(verdict) or 'ok'} |")
        md.append('')

    # Worst-20 per bucket (sort by deg ascending for DT/NE; echo ascending for FS)
    md.append('## Worst-20 per bucket')
    md.append('')
    sort_metric = {
        'FS_static': 'echo', 'FS_movement': 'echo',
        'DT_static': 'deg', 'DT_movement': 'deg', 'NE': 'deg',
    }
    for b in bucket_order:
        if b not in bucket_acc:
            continue
        metric = sort_metric[b]
        worst = sorted(bucket_acc[b][metric], key=lambda kv: kv[1])[:20]
        md.append(f"### {b} (sorted by {metric} ascending)")
        md.append('')
        md.append(f"| stem | {metric} |")
        md.append('|---|---:|')
        for stem, v in worst:
            md.append(f"| `{stem}` | {v:.3f} |")
        md.append('')

    with open(os.path.join(args.result_dir, 'result.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f"Wrote {args.result_dir}/scores.json + result.md")


if __name__ == '__main__':
    main()
