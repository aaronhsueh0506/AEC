#!/usr/bin/env python3
"""Resolve and pin the 90-case manifest against the git-ignored corpus.

The corpus under wav/aec_challenge_blind/ is local data. A stem resolves to
<dataset>/<scenario>/<stem>_{mic,lpb}.wav, or, when the clip was moved aside,
to <dataset>/<scenario>/_quarantine_non_official/. The same 90 clips are
compared either way.

  prepare  resolve every stem, check manifest_90case.sha256, and link the
           resolved clips into a scratch view <view>/<scenario>/<stem>_*.wav.
           The runners hand that view to the renderers and the scorer.
  write    regenerate manifest_90case.sha256 from the current corpus.

The pin lists the sha256 of each clip (keyed by its scenario-relative name,
not by where it currently sits), the stems list and the manifest JSON. Only
the standard library is used.
"""
import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
STEMS = os.path.join(_HERE, 'manifest_90case_stems.txt')
MANIFEST = os.path.join(_HERE, 'manifest_90case.json')
PIN = os.path.join(_HERE, 'manifest_90case.sha256')
QUARANTINE = '_quarantine_non_official'
SCENARIOS = ('farend_singletalk', 'nearend_singletalk', 'doubletalk')
KINDS = ('mic', 'lpb')
STEMS_KEY = 'eval/manifest_90case_stems.txt'
JSON_KEY = 'eval/manifest_90case.json'


class PinError(Exception):
    pass


def read_stems(path=STEMS):
    with open(path, encoding='utf-8') as fh:
        stems = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
    if not stems:
        raise PinError('%s lists no cases' % path)
    return stems


def scenario_of(stem):
    for sc in SCENARIOS:
        if '_' + sc in stem:
            return sc
    raise PinError('stem names no known scenario: %s' % stem)


def logical_name(stem, kind):
    return '%s/%s_%s.wav' % (scenario_of(stem), stem, kind)


def locate(dataset, rel):
    """Scenario directory first, then that scenario's quarantine directory."""
    sc, name = rel.split('/', 1)
    for cand in (os.path.join(dataset, sc, name),
                 os.path.join(dataset, sc, QUARANTINE, name)):
        if os.path.isfile(cand):
            return cand
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def resolve_all(dataset, stems):
    """Return {logical name: path}; raise naming every unresolved stem."""
    found, missing = {}, []
    for stem in stems:
        gone = []
        for kind in KINDS:
            rel = logical_name(stem, kind)
            path = locate(dataset, rel)
            if path is None:
                gone.append(kind)
            else:
                found[rel] = path
        if gone:
            missing.append('%s (%s)' % (stem, '+'.join(gone)))
    if missing:
        raise PinError(
            'cannot resolve %d of %d stems under %s (tried <scenario>/ and '
            '<scenario>/%s/):\n  %s'
            % (len(missing), len(stems), dataset, QUARANTINE,
               '\n  '.join(missing)))
    return found


def check_manifest_consistent(stems):
    with open(MANIFEST, encoding='utf-8') as fh:
        doc = json.load(fh)
    in_json = [s for sp in doc.values() for k in ('static', 'movement')
               for s in sp.get(k, [])]
    if sorted(in_json) != sorted(stems):
        a, b = set(in_json), set(stems)
        raise PinError(
            'manifest_90case.json and manifest_90case_stems.txt differ: '
            'only in json %s, only in stems %s'
            % (sorted(a - b)[:5], sorted(b - a)[:5]))


def parse_pin(path=PIN):
    pins = {}
    with open(path, encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.rstrip('\n')
            if not ln or ln.startswith('#'):
                continue
            digest, name = ln.split('  ', 1)
            pins[name] = digest
    return pins


def verify(found):
    if not os.path.isfile(PIN):
        raise PinError('missing pin file %s (create it with: manifest_pin.py '
                       'write --dataset DIR)' % PIN)
    pins = parse_pin()
    actual = {k: sha256_file(v) for k, v in found.items()}
    actual[STEMS_KEY] = sha256_file(STEMS)
    actual[JSON_KEY] = sha256_file(MANIFEST)
    bad = ['%s: %s' % (k, 'not pinned' if k not in pins else
                       'sha256 differs (pinned %s..., now %s...)'
                       % (pins[k][:12], actual[k][:12]))
           for k in sorted(actual) if pins.get(k) != actual[k]]
    bad += ['%s: pinned but not in the manifest' % k
            for k in sorted(set(pins) - set(actual))]
    if bad:
        raise PinError('manifest pin mismatch (%d):\n  %s\n'
                       'The corpus or the manifest drifted since the pin was '
                       'written; fix the data, or re-pin on purpose with '
                       'manifest_pin.py write.' % (len(bad), '\n  '.join(bad)))
    return len(actual)


def link_view(found, view):
    for rel, path in found.items():
        dst = os.path.join(view, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(path), dst)


def cmd_prepare(args):
    stems = read_stems()
    check_manifest_consistent(stems)
    found = resolve_all(args.dataset, stems)
    n_q = sum(1 for p in found.values() if QUARANTINE in p.split(os.sep))
    n = verify(found)
    link_view(found, args.view)
    print('manifest: %d stems resolved (%d clips, %d from %s), %d sha256 pins '
          'verified' % (len(stems), len(found), n_q, QUARANTINE, n))


def cmd_write(args):
    stems = read_stems()
    check_manifest_consistent(stems)
    found = resolve_all(args.dataset, stems)
    rows = sorted((k, sha256_file(v)) for k, v in found.items())
    rows += [(STEMS_KEY, sha256_file(STEMS)), (JSON_KEY, sha256_file(MANIFEST))]
    with open(PIN, 'w', encoding='utf-8') as fh:
        fh.write('# sha256 of the %d clips of the 90-case manifest (scenario-'
                 'relative names; a clip may sit in\n# <scenario>/ or '
                 '<scenario>/%s/), the stems list and the '
                 'manifest JSON.\n# Checked by eval/manifest_pin.py prepare '
                 'before every A/B render.\n' % (len(found), QUARANTINE))
        for k, d in rows:
            fh.write('%s  %s\n' % (d, k))
    print('wrote %s (%d entries)' % (PIN, len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', required=True)
    p.add_argument('--view', required=True)
    p.set_defaults(fn=cmd_prepare)
    w = sub.add_parser('write')
    w.add_argument('--dataset', required=True)
    w.set_defaults(fn=cmd_write)
    args = ap.parse_args()
    try:
        args.fn(args)
    except PinError as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
