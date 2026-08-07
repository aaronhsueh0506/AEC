"""Completeness and byte-equality gate for a blind A/B run.

An A/B is only evidence if it covered the corpus it claims to have covered, and
"byte-identical" is only a finding if somebody compared the bytes. Both had been
taken on trust:

  * the first version of the C harness reported "rendered 90" while all 90
    renders had exited 2 (an invalid CLI flag), because it counted loop
    iterations rather than successes;
  * `bench_aecmos.py` writes a well-formed `scores.json` for "Scoring 0 cases",
    so a scorer that saw nothing still produced a file that reads like a result;
  * the follow-up guard only rejected *zero* scored cases, so 1..89 still
    passed as a complete run;
  * the byte-identical conclusion in two evidence READMEs was inferred from
    every AECMOS delta being exactly +0.0000, which is what a broken harness
    produces too.

This module answers those directly: the rendered set, the scored set and the
manifest must be the SAME SET (not the same size), and every output pair is
compared sample by sample with the result written to `wav_comparison.json`.

Usable as a CLI from the harness or as functions from a test.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import struct
import sys

SUFFIX = "_ours.wav"


class AbError(Exception):
    """A completeness failure. Always fatal: the run is not evidence."""


def read_manifest(path):
    stems = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                stems.append(line)
    if not stems:
        raise AbError(f"manifest {path} lists no cases")
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        raise AbError(f"manifest {path} repeats stems: {dupes}")
    return stems


def rendered_stems(directory):
    if not os.path.isdir(directory):
        raise AbError(f"render directory missing: {directory}")
    return {name[:-len(SUFFIX)] for name in os.listdir(directory)
            if name.endswith(SUFFIX)}


def scored_stems(scores_path):
    try:
        with open(scores_path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError:
        raise AbError(f"scores file missing: {scores_path}")
    except ValueError as exc:
        raise AbError(f"scores file unreadable: {scores_path}: {exc}")
    scores = doc.get("scores")
    if not isinstance(scores, dict):
        raise AbError(f"{scores_path}: no 'scores' mapping")
    return set(scores)


def require_same_set(what, actual, expected):
    """Set equality, not cardinality.

    Counting alone accepts a run that rendered case X twice and case Y never,
    which is the failure a count check is least likely to survive contact with.
    """
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise AbError(
            f"{what}: {len(actual)} of {len(expected)} manifest cases"
            + (f"; MISSING {missing[:8]}"
               f"{' (+%d more)' % (len(missing) - 8) if len(missing) > 8 else ''}"
               if missing else "")
            + (f"; UNEXPECTED {extra[:8]}"
               f"{' (+%d more)' % (len(extra) - 8) if len(extra) > 8 else ''}"
               if extra else ""))


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Wav:
    __slots__ = ("sample_rate", "channels", "format", "samples")

    def __init__(self, sample_rate, channels, fmt, samples):
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = fmt        # "float32" | "pcm16"
        self.samples = samples


def _read_wav(path):
    """Minimal RIFF reader.

    `aec_wav` writes WAVE_FORMAT_IEEE_FLOAT (tag 3); the stdlib `wave` module
    only accepts PCM and raises "unknown format: 3", so it cannot be used to
    check the outputs this harness actually produces. Both tags are handled
    here so the same gate works on the CLI's float output and on any PCM
    reference.
    """
    with open(path, "rb") as fh:
        blob = fh.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise AbError(f"{path}: not a RIFF/WAVE file")
    pos, fmt = 12, None
    data = None
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack_from("<I", blob, pos + 4)[0]
        body = blob[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise AbError(f"{path}: missing fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 3 and bits == 32:
        samples = array.array("f")
        kind = "float32"
    elif tag == 1 and bits == 16:
        samples = array.array("h")
        kind = "pcm16"
    else:
        raise AbError(f"{path}: unsupported format tag {tag} / {bits}-bit")
    samples.frombytes(data[:len(data) - (len(data) % samples.itemsize)])
    if sys.byteorder == "big":
        samples.byteswap()
    return Wav(rate, channels, kind, samples)


def compare_pair(base_path, cand_path):
    """Per-case record. Byte equality is decided by the file digests; the
    sample statistics are what distinguish 'identical' from 'different but
    scores the same'."""
    bw, cw = _read_wav(base_path), _read_wav(cand_path)
    if bw.sample_rate != cw.sample_rate:
        raise AbError(f"{os.path.basename(base_path)}: sample rate "
                      f"{bw.sample_rate} vs {cw.sample_rate}")
    if bw.channels != cw.channels:
        raise AbError(f"{os.path.basename(base_path)}: channel count "
                      f"{bw.channels} vs {cw.channels}")
    if bw.format != cw.format:
        raise AbError(f"{os.path.basename(base_path)}: sample format "
                      f"{bw.format} vs {cw.format}")

    b, c = bw.samples, cw.samples
    n = min(len(b), len(c))
    max_abs, sq = 0.0, 0.0
    for i in range(n):
        d = abs(b[i] - c[i])
        if d > max_abs:
            max_abs = d
        sq += d * d
    # A NaN or Inf in the candidate makes every delta comparison vacuously
    # true, so it is reported rather than folded into the statistics.
    finite = all(math.isfinite(v) for v in c) if cw.format == "float32" \
        else True
    peak = max((abs(v) for v in c if math.isfinite(v)), default=0.0)
    base_digest, cand_digest = _digest(base_path), _digest(cand_path)
    return {
        "sample_rate": cw.sample_rate,
        "channels": cw.channels,
        "format": cw.format,
        "base_sample_count": len(b),
        "cand_sample_count": len(c),
        "base_sha256": base_digest,
        "cand_sha256": cand_digest,
        "byte_equal": base_digest == cand_digest,
        "max_abs_diff": max_abs,
        "rms_diff": math.sqrt(sq / n) if n else 0.0,
        "finite": finite,
        "peak": peak,
    }


def compare_run(manifest, base_dir, cand_dir, scores_base=None,
                scores_cand=None):
    """Full gate. Raises AbError on any completeness failure."""
    stems = read_manifest(manifest)
    expected = set(stems)
    require_same_set(f"rendered {base_dir}", rendered_stems(base_dir), expected)
    require_same_set(f"rendered {cand_dir}", rendered_stems(cand_dir), expected)
    if scores_base:
        require_same_set(f"scored {scores_base}", scored_stems(scores_base),
                         expected)
    if scores_cand:
        require_same_set(f"scored {scores_cand}", scored_stems(scores_cand),
                         expected)

    cases = {}
    for stem in stems:
        cases[stem] = compare_pair(os.path.join(base_dir, stem + SUFFIX),
                                   os.path.join(cand_dir, stem + SUFFIX))

    n_equal = sum(1 for c in cases.values() if c["byte_equal"])
    nonfinite = sorted(s for s, c in cases.items() if not c["finite"])
    if nonfinite:
        raise AbError(f"non-finite samples in candidate output: {nonfinite[:8]}")
    return {
        "manifest": os.path.basename(manifest),
        "case_count": len(cases),
        "byte_equal_count": n_equal,
        "all_byte_equal": n_equal == len(cases),
        "worst_max_abs_diff": max((c["max_abs_diff"] for c in cases.values()),
                                  default=0.0),
        "worst_rms_diff": max((c["rms_diff"] for c in cases.values()),
                              default=0.0),
        "cases": cases,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--cand-dir", required=True)
    ap.add_argument("--scores-base")
    ap.add_argument("--scores-cand")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        report = compare_run(args.manifest, args.base_dir, args.cand_dir,
                             args.scores_base, args.scores_cand)
    except AbError as exc:
        print(f"AB GATE FAILED: {exc}", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"  compared {report['case_count']} cases -> {args.out}: "
          f"{report['byte_equal_count']} byte-equal, worst |diff| "
          f"{report['worst_max_abs_diff']} LSB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
