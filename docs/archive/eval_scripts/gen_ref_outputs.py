"""
Generate AEC2 (old_aec) and AEC3 reference output WAVs for all files in aec_challenge_blind.
Saves to python/output_ref/{stem}_aec2.wav and {stem}_aec3.wav.
Skips files that already exist (safe to re-run).
Usage: python3 gen_ref_outputs.py [--workers N]
"""
import os, sys, subprocess, tempfile, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf
import numpy as np

REPO = Path(__file__).parent.parent
WAV_BASE = REPO / 'wav/aec_challenge_blind'
BIN_DIR  = REPO / 'bin'
OUT_DIR  = Path(__file__).parent / 'output_ref'
OUT_DIR.mkdir(exist_ok=True)

AEC3_CLI    = str(BIN_DIR / 'aec3_cli')
OLD_AEC_CLI = str(BIN_DIR / 'old_aec_cli')

SCENARIOS = ['farend_singletalk', 'nearend_singletalk', 'doubletalk']


def _run_cli(cli, mic_path, lpb_path, out_path):
    try:
        r = subprocess.run([cli, mic_path, lpb_path, out_path],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def process_file(args):
    mic_path, lpb_path, stem = args
    results = []

    out_aec2 = OUT_DIR / f'{stem}_aec2.wav'
    out_aec3 = OUT_DIR / f'{stem}_aec3.wav'

    for cli, out_path, tag in [
        (OLD_AEC_CLI, out_aec2, 'aec2'),
        (AEC3_CLI,    out_aec3, 'aec3'),
    ]:
        if out_path.exists():
            results.append(f'skip {tag} {stem[:40]}')
            continue
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            ok = _run_cli(cli, mic_path, lpb_path, tmp)
            if ok and os.path.isfile(tmp):
                data, sr = sf.read(tmp)
                sf.write(str(out_path), data.astype(np.float32), sr)
                results.append(f'ok   {tag} {stem[:40]}')
            else:
                results.append(f'FAIL {tag} {stem[:40]}')
        finally:
            if os.path.isfile(tmp):
                os.unlink(tmp)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    tasks = []
    for scenario in SCENARIOS:
        wav_dir = WAV_BASE / scenario
        if not wav_dir.exists():
            print(f'WARNING: {wav_dir} not found, skipping')
            continue
        for f in sorted(os.listdir(str(wav_dir))):
            if not f.endswith('_mic.wav'):
                continue
            stem = f.replace('_mic.wav', '')
            mic_path = str(wav_dir / f)
            lpb_path = str(wav_dir / f.replace('_mic.wav', '_lpb.wav'))
            if not os.path.isfile(lpb_path):
                continue
            tasks.append((mic_path, lpb_path, stem))

    total = len(tasks)
    print(f'Total files: {total}  workers: {args.workers}')
    print(f'Output dir:  {OUT_DIR}')

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_file, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            for msg in fut.result():
                if not msg.startswith('skip'):
                    print(f'[{done}/{total}] {msg}')
            if done % 50 == 0:
                print(f'  --- {done}/{total} done ---')

    print('\nDone.')


if __name__ == '__main__':
    main()
