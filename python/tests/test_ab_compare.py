"""Mutation tests for the A/B completeness gate (`eval/ab_compare.py`).

Every guard here exists because its absence already let a broken run be reported
as a result:

  * the first C harness printed "rendered 90" while all 90 renders had exited 2
    on an invalid CLI flag -- it counted loop iterations, not successes;
  * `bench_aecmos.py` writes a well-formed `scores.json` for "Scoring 0 cases",
    so an empty scoring pass still produces a file that reads like a result;
  * the follow-up guard rejected only ZERO scored cases, so 1..89 still passed;
  * two evidence READMEs concluded "byte-identical" from every AECMOS delta
    being exactly +0.0000 -- which is also what a broken harness produces.

So each guard is mutation-tested: break the run in one specific way and require
the gate to fail. A gate that cannot fail is the defect it was written to catch.
"""
from __future__ import annotations

import array
import json
import math
import os
import struct
import sys
import wave

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "eval"))

import ab_compare  # noqa: E402

STEMS = [f"case{i:02d}_farend_singletalk" for i in range(6)]


def _write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array("h", samples).tobytes())


def _write_wav_f32(path, samples, rate=16000):
    """WAVE_FORMAT_IEEE_FLOAT, which is what `aec_wav` actually writes.

    The stdlib `wave` module cannot write (or read) it -- that is why the gate
    carries its own RIFF reader, and why these cases exist: the first run of the
    hardened harness died on `wave.Error: unknown format: 3` against real
    output, having passed every PCM16 test here.
    """
    data = array.array("f", samples).tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(data), b"WAVE", b"fmt ", 16,
        3, 1, rate, rate * 4, 4, 32, b"data", len(data))
    with open(str(path), "wb") as fh:
        fh.write(header)
        fh.write(data)


@pytest.fixture
def run(tmp_path):
    """A complete, passing A/B run: manifest, two render dirs, two score
    files."""
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# header comment\n" + "\n".join(STEMS) + "\n",
                        encoding="utf-8")
    base, cand = tmp_path / "out_base", tmp_path / "out_cand"
    base.mkdir()
    cand.mkdir()
    for i, stem in enumerate(STEMS):
        samples = [((j * 37 + i * 11) % 2000) - 1000 for j in range(256)]
        _write_wav(base / f"{stem}{ab_compare.SUFFIX}", samples)
        _write_wav(cand / f"{stem}{ab_compare.SUFFIX}", samples)
    scores = {}
    for name in ("scores_base.json", "scores_cand.json"):
        path = tmp_path / name
        path.write_text(json.dumps({
            "label": name,
            "scores": {s: {"bucket": "FS", "echo": 4.0, "deg": 3.0}
                       for s in STEMS}}), encoding="utf-8")
        scores[name] = path
    return {
        "manifest": str(manifest), "base": str(base), "cand": str(cand),
        "scores_base": str(scores["scores_base.json"]),
        "scores_cand": str(scores["scores_cand.json"]),
        "tmp": tmp_path,
    }


def _compare(run):
    return ab_compare.compare_run(run["manifest"], run["base"], run["cand"],
                                  run["scores_base"], run["scores_cand"])


# ── the happy path, which must actually be a pass ───────────────────────────

def test_a_complete_run_passes_and_reports_byte_equality(run):
    report = _compare(run)
    assert report["case_count"] == len(STEMS)
    assert report["all_byte_equal"] is True
    assert report["worst_max_abs_diff"] == 0
    for stem, case in report["cases"].items():
        assert case["byte_equal"] is True
        assert case["base_sha256"] == case["cand_sha256"]
        assert len(case["base_sha256"]) == 64
        assert case["sample_rate"] == 16000
        assert case["base_sample_count"] == 256


# ── MUTATION: the run is incomplete ─────────────────────────────────────────

def test_a_missing_output_wav_fails(run):
    """The failure the original harness reported as 'rendered 90'."""
    os.remove(os.path.join(run["cand"], STEMS[2] + ab_compare.SUFFIX))
    with pytest.raises(ab_compare.AbError) as exc:
        _compare(run)
    assert STEMS[2] in str(exc.value)


def test_a_missing_score_entry_fails(run):
    """1..89 scored cases used to pass; only 0 was rejected."""
    doc = json.loads(open(run["scores_cand"]).read())
    del doc["scores"][STEMS[4]]
    open(run["scores_cand"], "w").write(json.dumps(doc))
    with pytest.raises(ab_compare.AbError) as exc:
        _compare(run)
    assert STEMS[4] in str(exc.value)


def test_an_empty_scores_file_fails(run):
    """bench_aecmos.py writes this file happily for 'Scoring 0 cases'."""
    open(run["scores_base"], "w").write(json.dumps({"label": "x",
                                                    "scores": {}}))
    with pytest.raises(ab_compare.AbError):
        _compare(run)


def test_a_substituted_stem_fails_even_though_the_count_is_right(run):
    """THE case a count check cannot catch: render one case twice and another
    never, and the total is still exactly right."""
    src = os.path.join(run["cand"], STEMS[0] + ab_compare.SUFFIX)
    victim = os.path.join(run["cand"], STEMS[1] + ab_compare.SUFFIX)
    os.remove(victim)
    with open(src, "rb") as fh:
        blob = fh.read()
    with open(os.path.join(run["cand"], "case99_extra" + ab_compare.SUFFIX),
              "wb") as fh:
        fh.write(blob)
    assert len(ab_compare.rendered_stems(run["cand"])) == len(STEMS)
    with pytest.raises(ab_compare.AbError) as exc:
        _compare(run)
    message = str(exc.value)
    assert STEMS[1] in message and "case99_extra" in message


def test_a_repeated_manifest_stem_fails(run):
    """A manifest that lists a case twice makes every count check meaningless
    before the run even starts."""
    with open(run["manifest"], "a", encoding="utf-8") as fh:
        fh.write(STEMS[0] + "\n")
    with pytest.raises(ab_compare.AbError) as exc:
        _compare(run)
    assert "repeats" in str(exc.value)


def test_a_missing_render_directory_fails(run):
    os.rename(run["cand"], run["cand"] + ".moved")
    with pytest.raises(ab_compare.AbError):
        _compare(run)


def test_a_corrupt_scores_file_fails_closed(run):
    open(run["scores_cand"], "w").write("{not json")
    with pytest.raises(ab_compare.AbError):
        _compare(run)


# ── MUTATION: the run is complete but the outputs differ ────────────────────

def test_one_changed_sample_is_reported_as_not_byte_equal(run):
    """The byte-identical claim has to be decidable, in both directions."""
    path = os.path.join(run["cand"], STEMS[3] + ab_compare.SUFFIX)
    with wave.open(path, "rb") as w:
        params, raw = w.getparams(), w.readframes(256)
    samples = array.array("h")
    samples.frombytes(raw)
    samples[100] += 7
    _write_wav(path, samples, params.framerate)

    report = _compare(run)
    assert report["all_byte_equal"] is False
    assert report["byte_equal_count"] == len(STEMS) - 1
    case = report["cases"][STEMS[3]]
    assert case["byte_equal"] is False
    assert case["base_sha256"] != case["cand_sha256"]
    assert case["max_abs_diff"] == 7
    assert case["rms_diff"] > 0.0
    # every other case is untouched
    assert all(report["cases"][s]["byte_equal"] for s in STEMS if s != STEMS[3])


def test_a_sample_rate_mismatch_fails_rather_than_comparing(run):
    """Comparing 8 kHz output against 16 kHz output sample-by-sample would
    produce a number, and the number would be meaningless."""
    path = os.path.join(run["cand"], STEMS[0] + ab_compare.SUFFIX)
    _write_wav(path, [0] * 256, rate=8000)
    with pytest.raises(ab_compare.AbError) as exc:
        _compare(run)
    assert "sample rate" in str(exc.value)


def test_a_truncated_output_is_visible_in_the_record(run):
    path = os.path.join(run["cand"], STEMS[0] + ab_compare.SUFFIX)
    _write_wav(path, [0] * 128)
    report = _compare(run)
    case = report["cases"][STEMS[0]]
    assert case["base_sample_count"] == 256
    assert case["cand_sample_count"] == 128
    assert case["byte_equal"] is False


# ── the CLI, which is what the harness actually calls ───────────────────────

def test_cli_writes_the_report_and_returns_zero(run, capsys):
    out = str(run["tmp"] / "wav_comparison.json")
    rc = ab_compare.main(["--manifest", run["manifest"],
                          "--base-dir", run["base"], "--cand-dir", run["cand"],
                          "--scores-base", run["scores_base"],
                          "--scores-cand", run["scores_cand"], "--out", out])
    assert rc == 0
    report = json.loads(open(out).read())
    assert report["all_byte_equal"] is True
    assert set(report["cases"]) == set(STEMS)


def test_cli_returns_nonzero_and_writes_nothing_on_an_incomplete_run(run):
    os.remove(os.path.join(run["base"], STEMS[0] + ab_compare.SUFFIX))
    out = str(run["tmp"] / "wav_comparison.json")
    rc = ab_compare.main(["--manifest", run["manifest"],
                          "--base-dir", run["base"], "--cand-dir", run["cand"],
                          "--scores-base", run["scores_base"],
                          "--scores-cand", run["scores_cand"], "--out", out])
    assert rc == 1
    assert not os.path.exists(out), (
        "a failed gate must not leave a report behind for someone to cite")


# ── the format the CLI actually writes ──────────────────────────────────────

@pytest.fixture
def f32_run(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(STEMS) + "\n", encoding="utf-8")
    base, cand = tmp_path / "out_base", tmp_path / "out_cand"
    base.mkdir()
    cand.mkdir()
    for i, stem in enumerate(STEMS):
        samples = [math.sin(0.01 * j + i) * 0.5 for j in range(256)]
        _write_wav_f32(base / f"{stem}{ab_compare.SUFFIX}", samples)
        _write_wav_f32(cand / f"{stem}{ab_compare.SUFFIX}", samples)
    return {"manifest": str(manifest), "base": str(base), "cand": str(cand),
            "tmp": tmp_path}


def test_float32_output_is_read_and_compared(f32_run):
    report = ab_compare.compare_run(f32_run["manifest"], f32_run["base"],
                                    f32_run["cand"])
    assert report["all_byte_equal"] is True
    case = report["cases"][STEMS[0]]
    assert case["format"] == "float32"
    assert case["base_sample_count"] == 256
    assert 0.4 < case["peak"] <= 0.5


def test_float32_difference_is_measured_not_rounded_away(f32_run):
    """A 1e-6 difference is invisible to a 16-bit comparison and is exactly the
    size of difference a retiming change produces."""
    path = os.path.join(f32_run["cand"], STEMS[2] + ab_compare.SUFFIX)
    wav = ab_compare._read_wav(path)
    samples = list(wav.samples)
    samples[50] += 1e-6
    _write_wav_f32(path, samples)
    report = ab_compare.compare_run(f32_run["manifest"], f32_run["base"],
                                    f32_run["cand"])
    assert report["all_byte_equal"] is False
    case = report["cases"][STEMS[2]]
    assert case["byte_equal"] is False
    assert 0 < case["max_abs_diff"] < 1e-5
    assert case["rms_diff"] > 0.0


def test_a_nan_in_the_candidate_fails_rather_than_scoring(f32_run):
    """Every |base - NaN| > tol comparison is false, so a NaN run reads as
    byte-identical to any tolerance-based check."""
    path = os.path.join(f32_run["cand"], STEMS[1] + ab_compare.SUFFIX)
    wav = ab_compare._read_wav(path)
    samples = list(wav.samples)
    samples[10] = float("nan")
    _write_wav_f32(path, samples)
    with pytest.raises(ab_compare.AbError) as exc:
        ab_compare.compare_run(f32_run["manifest"], f32_run["base"],
                               f32_run["cand"])
    assert "non-finite" in str(exc.value)


def test_a_pcm16_baseline_against_float32_candidate_fails(f32_run):
    path = os.path.join(f32_run["cand"], STEMS[0] + ab_compare.SUFFIX)
    _write_wav(path, [0] * 256)
    with pytest.raises(ab_compare.AbError) as exc:
        ab_compare.compare_run(f32_run["manifest"], f32_run["base"],
                               f32_run["cand"])
    assert "sample format" in str(exc.value)
