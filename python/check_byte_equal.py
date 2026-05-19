"""Byte-equal check against 3aadd2d baseline (balanced_aec3 path).

Renders 25 representative cases (5 per bucket: FS_static / FS_movement /
DT_static / DT_movement / NE) via ``eval_aec_challenge.py`` -- the same
render pipeline that produced the baseline -- and compares md5 of each
output WAV against the baseline md5 recorded in
``docs/bench/v3_21_3aadd2d_baseline/byte_equal_reference.json``.

Used after each cleanup round to confirm no behavioural drift from the
baseline shipped at commit 3aadd2d.

Usage:
    python3 python/check_byte_equal.py [--preset balanced_aec3]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, 'docs/bench/v3_21_3aadd2d_baseline/byte_equal_reference.json')
CORPUS = os.path.join(ROOT, 'wav/aec_challenge_blind')
EVAL_SCRIPT = os.path.join(ROOT, 'python/eval_aec_challenge.py')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='balanced_aec3',
                   help='Preset to use (switch to "balanced" once renamed)')
    args = p.parse_args()

    with open(REFERENCE) as f:
        ref = json.load(f)

    with tempfile.TemporaryDirectory(prefix='be_check_') as tmpdir:
        stem_file = os.path.join(tmpdir, 'stems.txt')
        with open(stem_file, 'w') as f:
            for stem in ref:
                f.write(stem + '\n')
        out_dir = os.path.join(tmpdir, 'out')
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            'python3', EVAL_SCRIPT, CORPUS,
            '--preset', args.preset,
            '--filter', '832',
            '--cng', '--parallel',
            '--cases-list', stem_file,
            '-o', out_dir,
            '--workers', '6',
        ]
        print(f'Rendering {len(ref)} cases via eval_aec_challenge.py ...')
        subprocess.run(cmd, check=True)

        pass_n = fail_n = 0
        failures = []
        for stem, expected in ref.items():
            ours_path = os.path.join(out_dir, stem + '_ours.wav')
            nores_path = os.path.join(out_dir, stem + '_ours_nores.wav')
            got_ours = (hashlib.md5(open(ours_path, 'rb').read()).hexdigest()[:16]
                        if os.path.exists(ours_path) else 'MISSING')
            got_nores = (hashlib.md5(open(nores_path, 'rb').read()).hexdigest()[:16]
                         if os.path.exists(nores_path) else 'MISSING')
            ok = (got_ours == expected['md5_ours']
                  and got_nores == expected['md5_ours_nores'])
            if ok:
                pass_n += 1
            else:
                fail_n += 1
                failures.append((stem, expected, got_ours, got_nores))
            marker = 'OK ' if ok else 'FAIL'
            print(f'  [{marker}] {expected["bucket"]:12s} {stem[:50]:50s} '
                  f'ours={got_ours} (exp {expected["md5_ours"]})')

        print(f'\n=== {pass_n}/{len(ref)} PASS, {fail_n} FAIL ===')
        if failures:
            print('\nFAILURES:')
            for stem, exp, got_o, got_n in failures:
                print(f'  {stem}')
                print(f'    expected ours={exp["md5_ours"]} nores={exp["md5_ours_nores"]}')
                print(f'    got      ours={got_o} nores={got_n}')
            sys.exit(1)


if __name__ == '__main__':
    main()
