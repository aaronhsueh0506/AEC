#!/usr/bin/env python3
"""Smoke test for BUG-1+2 retune experiments.

Processes 20 FS + 20 NE + 20 DT cases with balanced preset, scores with AECMOS.
Reports FS echo, NE deg, DT echo, DT deg means.

Uses deterministic selection: sorted mic filenames, first 20 non-movement cases
per category (matches v2.3.1 smoke baseline).
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))

import torch
import librosa
import onnxruntime as ort

# ---- Custom AECMOS estimator (reuses a single ONNX session) ----
class FastAECMOS:
    def __init__(self, model_path):
        self.model_path = model_path
        self.max_len = 20
        self.hop_fraction = 0.5
        self.sampling_rate = 16000
        self.dft_size = 512
        self.hidden_size = (4, 1, 64)
        self.need_scenario_marker = True  # Run_1663915512
        # Reuse a single session (big speedup)
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def _mel_transform(self, sample, sr):
        mel_spec = librosa.feature.melspectrogram(
            y=sample, sr=sr, n_fft=self.dft_size + 1,
            hop_length=int(self.hop_fraction * self.dft_size), n_mels=160)
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def score(self, talk_type, lpb_sig, mic_sig, enh_sig):
        # truncate to max_len
        seg = self.max_len * self.sampling_rate
        if len(lpb_sig) >= seg:
            lpb_sig = lpb_sig[:seg]
            mic_sig = mic_sig[:seg]
            enh_sig = enh_sig[:seg]
        L = self._mel_transform(lpb_sig, self.sampling_rate)
        M = self._mel_transform(mic_sig, self.sampling_rate)
        E = self._mel_transform(enh_sig, self.sampling_rate)
        if talk_type == 'nst':
            ne_st, fe_st = 1, 0
        elif talk_type == 'st':
            ne_st, fe_st = 0, 1
        else:
            ne_st, fe_st = 0, 0
        M = np.concatenate((M, np.ones((20, M.shape[1])) * (1 - fe_st), np.zeros((20, M.shape[1]))), axis=0)
        L = np.concatenate((L, np.ones((20, L.shape[1])) * (1 - ne_st), np.zeros((20, L.shape[1]))), axis=0)
        E = np.concatenate((E, np.ones((20, E.shape[1])), np.zeros((20, E.shape[1]))), axis=0)
        feats = np.stack((L, M, E)).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)
        h0 = np.zeros(self.hidden_size, dtype=np.float32)
        result = self.session.run([], {self.input_name: feats, 'h0': h0})[0]
        return float(result[0]), float(result[1])


def pick_cases(root, subdir, n, exclude_movement=True):
    d = os.path.join(root, subdir)
    files = sorted(f for f in os.listdir(d) if f.endswith('_mic.wav'))
    if exclude_movement:
        files = [f for f in files if '_with_movement_' not in f]
    files = files[:n]
    return [(os.path.join(d, f), os.path.join(d, f.replace('_mic.wav', '_lpb.wav'))) for f in files]


def run_category(label, cases, talk_type, estimator, run_ours_fn, sr=16000, fl=2048, preset='balanced'):
    echo_list, deg_list = [], []
    for mic_path, lpb_path in cases:
        mic, _ = sf.read(mic_path)
        ref, _ = sf.read(lpb_path)
        mic = mic.astype(np.float32)
        ref = ref.astype(np.float32)
        is_mv = '_with_movement_' in os.path.basename(mic_path)
        out = run_ours_fn(mic, ref, sr, fl, preset=preset, is_movement=is_mv)
        # Score
        n = min(len(mic), len(ref), len(out))
        lpb_sig = ref[:n].astype(np.float32)
        mic_sig = mic[:n].astype(np.float32)
        enh_sig = out[:n].astype(np.float32)
        echo, deg = estimator.score(talk_type, lpb_sig, mic_sig, enh_sig)
        echo_list.append(echo)
        deg_list.append(deg)
    return np.mean(echo_list), np.mean(deg_list)


def main():
    from eval_aec_challenge import run_ours

    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'model', 'Run_1663915512_Stage_0.onnx')
    estimator = FastAECMOS(model)

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'wav', 'aec_challenge_blind')

    n_cases = 20
    # Include _with_movement_ cases (matches eval_aecmos.py which sorts all)
    fs_cases = pick_cases(root, 'farend_singletalk', n_cases, exclude_movement=False)
    ne_cases = pick_cases(root, 'nearend_singletalk', n_cases, exclude_movement=False)
    dt_cases = pick_cases(root, 'doubletalk', n_cases, exclude_movement=False)

    print(f"FS cases: {len(fs_cases)}  NE cases: {len(ne_cases)}  DT cases: {len(dt_cases)}")

    t0 = time.time()
    fs_echo, fs_deg = run_category('FS', fs_cases, 'st', estimator, run_ours)
    t1 = time.time()
    print(f"FS: echo={fs_echo:.3f} deg={fs_deg:.3f}  ({t1-t0:.1f}s)")

    ne_echo, ne_deg = run_category('NE', ne_cases, 'nst', estimator, run_ours)
    t2 = time.time()
    print(f"NE: echo={ne_echo:.3f} deg={ne_deg:.3f}  ({t2-t1:.1f}s)")

    dt_echo, dt_deg = run_category('DT', dt_cases, 'dt', estimator, run_ours)
    t3 = time.time()
    print(f"DT: echo={dt_echo:.3f} deg={dt_deg:.3f}  ({t3-t2:.1f}s)")

    print(f"\nSUMMARY: FS_echo={fs_echo:.3f} NE_deg={ne_deg:.3f} DT_echo={dt_echo:.3f} DT_deg={dt_deg:.3f}")
    print(f"Total: {t3-t0:.1f}s")


if __name__ == '__main__':
    main()
