"""Trust gates for the per-case A/B report."""
from __future__ import annotations

import importlib.util
import json
import math
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PATH = os.path.join(_REPO, "eval", "ab_report.py")
_SPEC = importlib.util.spec_from_file_location("ab_report", _PATH)
ab_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ab_report)


def _scores(n=6):
    return {
        f"case-{i}-with-a-full-identifier": {
            "bucket": "FS_static" if i < 3 else "DT_movement",
            "echo": 4.0 + i / 10,
            "deg": 3.0 + i / 10,
        }
        for i in range(n)
    }


def test_report_has_per_case_tail_statistics_and_full_identifiers():
    base = _scores()
    cand = {k: dict(v) for k, v in base.items()}
    cand["case-0-with-a-full-identifier"]["echo"] -= 0.2
    cand["case-4-with-a-full-identifier"]["deg"] -= 0.15

    report = ab_report.analyse(base, cand, "test")
    assert report["overall"]["echo"]["n_below_deep"] == 1
    assert report["overall"]["deg"]["n_below_deep"] == 1
    assert len(report["overall"]["echo"]["worst5"]) == 5
    rendered = ab_report.render_md([report])
    assert "case-0-with-a-full-identifier" in rendered
    assert "case-4-with-a-full-identifier" in rendered


def test_nearest_rank_percentile_is_not_interpolated():
    values = list(range(1, 11))
    assert ab_report.percentile(values, 10) == 1
    assert ab_report.percentile(values, 50) == 5


def test_missing_case_is_fatal():
    base, cand = _scores(), _scores()
    cand.pop(next(iter(cand)))
    with pytest.raises(ab_report.ReportError, match="score sets differ"):
        ab_report.analyse(base, cand, "bad")


def test_bucket_drift_is_fatal():
    base, cand = _scores(), _scores()
    cand[next(iter(cand))]["bucket"] = "NE"
    with pytest.raises(ab_report.ReportError, match="bucket mismatch"):
        ab_report.analyse(base, cand, "bad")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, None, True])
def test_nonfinite_or_nonnumeric_score_is_fatal(value):
    base, cand = _scores(), _scores()
    cand[next(iter(cand))]["echo"] = value
    with pytest.raises(ab_report.ReportError, match="not finite"):
        ab_report.analyse(base, cand, "bad")


def test_a_small_bucket_cannot_hide_a_deep_case():
    base, cand = _scores(5), _scores(5)
    first = next(iter(cand))
    cand[first]["echo"] -= 0.25
    for key in list(cand)[1:]:
        cand[key]["echo"] += 0.07
    report = ab_report.analyse(base, cand, "mutation")
    assert report["overall"]["echo"]["mean"] > 0
    assert report["overall"]["echo"]["n_below_deep"] == 1
    assert report["flagged"][0]["stem"] == first


def test_summary_rejects_threshold_or_nonfinite_drift(tmp_path):
    report = ab_report.analyse(_scores(), _scores(), "test")
    path = tmp_path / "report.json"

    report["band"] = -0.06
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ab_report.ReportError, match="threshold mismatch"):
        ab_report.load_report(path)

    report["band"] = ab_report.BAND
    report["overall"]["echo"]["mean"] = math.nan
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ab_report.ReportError, match="non-finite"):
        ab_report.load_report(path)
