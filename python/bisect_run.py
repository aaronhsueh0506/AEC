#!/usr/bin/env python3
"""git bisect run script — FS_echo regression test.

Returns 0 (good) if FS_echo >= THRESHOLD, 1 (bad) otherwise.
Exit 125 = skip this commit (import error / incompatible).

Standalone: does NOT depend on eval_aec_challenge.py.
Handles both v2.8.0 (use_kalman) and v3.0.0+ (mode=AecMode) APIs.

Usage (automatic):
    git bisect run python3 bisect_run.py
"""
import sys, os, argparse
import numpy as np
import soundfile as sf

DATASET  = '/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind'
OUT_DIR  = '/tmp/bisect_step'
N_CASES  = 30          # fewer cases = faster per-step
THRESHOLD = 3.0        # FS_echo ≥ this → good

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from aec import AEC, AecConfig
except Exception as e:
    print(f"[bisect] import error: {e} — skip", file=sys.stderr)
    sys.exit(125)

# ── AecConfig compat ─────────────────────────────────────────────────────────
def _fields():
    return set(AecConfig.__dataclass_fields__.keys())

def make_config(sr, fl):
    fields = _fields()
    kw = dict(sample_rate=sr, filter_length=fl,
              enable_res=True, enable_shadow=True, enable_dtd=False)

    if 'mode' in fields:
        try:
            from aec import AecMode
            kw['mode'] = AecMode.PBFDKF
        except ImportError:
            pass
    if 'use_kalman' in fields:
        kw['use_kalman'] = True

    # keep only known fields to avoid TypeError on unknown kwargs
    return AecConfig(**{k: v for k, v in kw.items() if k in fields})

# ── Delay estimation (simple GCC) ────────────────────────────────────────────
def estimate_delay(mic, ref, max_lag=800):
    n = min(len(mic), len(ref), max_lag * 8)
    corr = np.correlate(mic[:n], ref[:n], mode='full')
    lag  = int(np.argmax(np.abs(corr))) - (n - 1)
    return max(0, min(lag, max_lag))

# ── Process one file ──────────────────────────────────────────────────────────
def run_file(mic_path, lpb_path):
    mic, sr = sf.read(mic_path, dtype='float32')
    lpb, _  = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if lpb.ndim > 1: lpb = lpb[:, 0]
    n = min(len(mic), len(lpb))
    mic, lpb = mic[:n], lpb[:n]

    delay = estimate_delay(mic, lpb)
    ref   = np.zeros(n, dtype=np.float32)
    if delay > 0:
        ref[delay:] = lpb[:n - delay]
    else:
        ref[:] = lpb

    config = make_config(sr, 2048)
    aec    = AEC(config)
    hop    = aec.hop_size
    out    = np.zeros(n, dtype=np.float32)
    pos    = 0
    while pos + hop <= n:
        out[pos:pos + hop] = aec.process(mic[pos:pos + hop], ref[pos:pos + hop])
        pos += hop
    return out, sr

# ── Score with local ONNX AECMOS ─────────────────────────────────────────────
def score_fs_echo(out_dir, dataset):
    model_dir = os.path.join(script_dir, '..', 'model')
    sys.path.insert(0, model_dir)
    try:
        from aecmos import AECMOSEstimator
        onnx = os.path.join(model_dir, 'Run_1663915512_Stage_0.onnx')
        est  = AECMOSEstimator(onnx)
    except Exception as e:
        print(f"[bisect] AECMOS load error: {e}", file=sys.stderr)
        sys.exit(125)

    fs_dir = os.path.join(dataset, 'farend_singletalk')
    stems  = [f[:-len('_mic.wav')] for f in sorted(os.listdir(fs_dir))
              if f.endswith('_mic.wav')][:N_CASES]

    echo_scores = []
    for stem in stems:
        enh = os.path.join(out_dir, f'{stem}_ours.wav')
        if not os.path.isfile(enh):
            continue
        try:
            lpb_sig, mic_sig, enh_sig = est.read_and_process_audio_files(
                os.path.join(fs_dir, f'{stem}_lpb.wav'),
                os.path.join(fs_dir, f'{stem}_mic.wav'),
                enh,
            )
            echo_mos, _ = est.run('st', lpb_sig, mic_sig, enh_sig)
            echo_scores.append(echo_mos)
        except Exception:
            pass

    return float(np.mean(echo_scores)) if echo_scores else 0.0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fs_dir = os.path.join(DATASET, 'farend_singletalk')
    mic_files = sorted(f for f in os.listdir(fs_dir) if f.endswith('_mic.wav'))[:N_CASES]

    ok = 0
    for mic_f in mic_files:
        stem    = mic_f[:-len('_mic.wav')]
        mic_p   = os.path.join(fs_dir, mic_f)
        lpb_p   = os.path.join(fs_dir, f'{stem}_lpb.wav')
        if not os.path.isfile(lpb_p):
            continue
        try:
            out, sr = run_file(mic_p, lpb_p)
            sf.write(os.path.join(OUT_DIR, f'{stem}_ours.wav'), out, sr)
            ok += 1
        except Exception as e:
            print(f"[bisect] error on {stem}: {e}", file=sys.stderr)

    if ok == 0:
        print("[bisect] no files processed — skip", file=sys.stderr)
        sys.exit(125)

    fs_echo = score_fs_echo(OUT_DIR, DATASET)
    commit  = os.popen('git rev-parse --short HEAD').read().strip()
    result  = 'GOOD' if fs_echo >= THRESHOLD else 'BAD'
    print(f"[bisect] {commit}  FS_echo={fs_echo:.3f}  → {result}", flush=True)

    sys.exit(0 if fs_echo >= THRESHOLD else 1)

if __name__ == '__main__':
    main()
