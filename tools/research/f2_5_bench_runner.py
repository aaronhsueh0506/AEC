#!/usr/bin/env python3
"""F2.5 bench runner — prev_dtd_conf two-stage hangover ablation.

Current ×0.9 per-frame decay gives TC ≈ 10 frames (100ms), shorter than R
EMA recovery (α=0.95, TC ≈ 20 frames). Fix: attack fast (1 frame), hold 10
frames, then ×0.9 decay. Toggle via AEC_DTD_CONF_TWO_STAGE=1.

Usage:
    # Ablation
    AEC_DTD_CONF_TWO_STAGE=1 \
        python3 tools/research/f2_5_bench_runner.py --out results/f2_5_on -j 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, 'python'))


def _estimate_delay(mic, ref, sr, max_delay_ms=250.0):
    max_d = int(sr * max_delay_ms / 1000)
    n = min(len(mic), len(ref))
    mic = mic[:n]
    ref = ref[:n]
    nfft = 1 << int(np.ceil(np.log2(n + max_d)))
    M = np.fft.rfft(mic, n=nfft)
    R = np.fft.rfft(ref, n=nfft)
    xc = np.fft.irfft(M * np.conj(R))
    return int(np.argmax(xc[:max_d]))


def _run_one(mic_path: str, lpb_path: str, out_path: str,
             fl: int, enable_cng: bool,
             dtd_conf_two_stage: bool = False) -> tuple[str, float, str]:
    from aec import AEC, AecConfig, AecMode, AecPreset

    t0 = time.time()
    try:
        mic, sr_m = sf.read(mic_path)
        lpb, sr_l = sf.read(lpb_path)
        if sr_m != sr_l:
            return (os.path.basename(out_path), 0.0, f'sr_mismatch:{sr_m},{sr_l}')
        sr = int(sr_m)
        mic = mic.astype(np.float32)
        lpb = lpb.astype(np.float32)

        delay = _estimate_delay(mic, lpb, sr)
        n = min(len(mic), len(lpb))
        if delay > 0 and delay < n:
            lpb_aligned = np.zeros(n, dtype=np.float32)
            lpb_aligned[delay:] = lpb[: n - delay]
        else:
            lpb_aligned = lpb[:n]
        mic = mic[:n]

        cfg = AecConfig.from_preset(
            AecPreset.BALANCED,
            sample_rate=sr, mode=AecMode.PBFDKF,
            filter_length=fl,
            enable_dtd=False, enable_shadow=True, enable_res=True,
            enable_cng=enable_cng,
            use_kalman=True,
            enable_delay_est=False,
            dtd_conf_two_stage=dtd_conf_two_stage,
        )
        np.random.seed(0)
        aec = AEC(cfg)
        hop = aec.hop_size
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        while pos + hop <= n:
            out[pos: pos + hop] = aec.process(mic[pos: pos + hop],
                                              lpb_aligned[pos: pos + hop])
            pos += hop

        sf.write(out_path, out, sr, subtype='PCM_16')
        return (os.path.basename(out_path), time.time() - t0, 'ok')
    except Exception as e:
        return (os.path.basename(out_path), time.time() - t0, f'err:{e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', default='wav/aec_challenge_blind')
    parser.add_argument('--out', required=True)
    parser.add_argument('--filter', type=int, default=832)
    parser.add_argument('--cng', action='store_true', default=True)
    parser.add_argument('-j', '--workers', type=int, default=4)
    args = parser.parse_args()

    dtd_conf_two_stage = os.environ.get('AEC_DTD_CONF_TWO_STAGE', '0').lower() not in (
        '0', 'false', 'no', '')
    enable_cng = args.cng
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_root = Path(args.wav)
    tasks = []
    for bucket in ('doubletalk', 'farend_singletalk', 'nearend_singletalk'):
        bdir = wav_root / bucket
        if not bdir.exists():
            continue
        mic_files = sorted(bdir.glob('*_mic.wav'))
        for mic_path in mic_files:
            stem = mic_path.name.replace('_mic.wav', '')
            lpb_path = bdir / f'{stem}_lpb.wav'
            if not lpb_path.exists():
                continue
            out_path = out_dir / f'{stem}_ours.wav'
            tasks.append((str(mic_path), str(lpb_path), str(out_path),
                           args.filter, enable_cng, dtd_conf_two_stage))

    print(f'F2.5: {len(tasks)} cases, dtd_conf_two_stage={dtd_conf_two_stage} '
          f'cng={enable_cng} out={args.out}')
    t_start = time.time()
    ok = err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            stem, dt, status = fut.result()
            if status == 'ok':
                ok += 1
            else:
                err += 1
                print(f'  ERR {stem}: {status}')
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                print(f'  {i}/{len(tasks)} ok={ok} err={err} '
                      f'elapsed={elapsed/60:.1f}min')

    print(f'Done: {ok}/{len(tasks)} ok, {err} errors, '
          f'{(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
