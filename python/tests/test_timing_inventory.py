"""Structural gate for docs/timing_constant_inventory.md.

The inventory is release evidence: it is the document that answers "is the
wall-clock timing audit complete?". Its first version could not answer that,
because it was internally inconsistent -- the category counts summed to 83
against a stated total of 82, two of the six categories had no section at all,
one candidate carried two mutually exclusive verdicts, five entries were
truncated mid-sentence, and a trailing space broke `git diff --check`. None of
that was visible without adding up the table by hand.

So the document is generated, and this test asserts the properties a reader
relies on when treating it as evidence. A hand-edit that breaks any of them
fails here rather than in someone's release review.
"""
from __future__ import annotations

import collections
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AEC = os.path.dirname(os.path.dirname(_HERE)) if os.path.basename(
    os.path.dirname(_HERE)) == "python" else None
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DOC = os.path.join(_REPO, "docs", "timing_constant_inventory.md")

# Every category the generator may emit. A candidate lands in exactly one.
EXPECTED_SECTIONS = [
    "Retime",
    "Already retimed",
    "Keep: event count",
    "Keep: fixed internal cadence",
    "Keep: dominated default",
    "Dead / unreachable",
    "Not a timing constant",
]

REQUIRED_FIELDS = ["Verdict", "Anchor", "Effective today", "Consumer",
                   "Provenance", "Reasoning"]


@pytest.fixture(scope="module")
def doc():
    assert os.path.isfile(_DOC), f"inventory missing: {_DOC}"
    with open(_DOC, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def sections(doc):
    """{title: [entry-name, ...]} in document order."""
    out = collections.OrderedDict()
    current = None
    for line in doc.splitlines():
        h2 = re.match(r"^## (.+?)(?: \((\d+)\))?$", line)
        if h2:
            title = h2.group(1)
            current = title if title in EXPECTED_SECTIONS else None
            if current:
                out[current] = {"declared": int(h2.group(2) or -1),
                                "entries": []}
            continue
        h3 = re.match(r"^### `(.+)`$", line)
        if h3 and current:
            out[current]["entries"].append(h3.group(1))
    return out


def test_every_category_has_a_section(sections):
    missing = [s for s in EXPECTED_SECTIONS if s not in sections]
    assert not missing, (
        f"declared categories with no section: {missing}. The first version of "
        f"this document claimed 14 'already retimed' and 6 'not a timing "
        f"constant' in its summary table and had neither section."
    )


def test_declared_count_matches_entries(sections):
    for title, body in sections.items():
        assert body["declared"] == len(body["entries"]), (
            f"section {title!r} declares {body['declared']} but contains "
            f"{len(body['entries'])} entries"
        )


def test_candidate_ids_are_unique_across_the_whole_document(sections):
    """One candidate, one verdict. This is the check that would have caught
    shadow_copy_hysteresis being filed under both retime and dead-code."""
    seen = collections.defaultdict(list)
    for title, body in sections.items():
        for name in body["entries"]:
            seen[name].append(title)
    dupes = {n: cats for n, cats in seen.items() if len(cats) > 1}
    assert not dupes, f"candidates in more than one category: {dupes}"


def test_summary_table_matches_the_sections(doc, sections):
    """The summary table is what a reader adds up; it must equal reality."""
    table = {}
    for line in doc.splitlines():
        m = re.match(r"^\| (.+?) \| (\d+) \|$", line)
        if m and m.group(1) in EXPECTED_SECTIONS:
            table[m.group(1)] = int(m.group(2))
        m_tot = re.match(r"^\| \*\*total\*\* \| \*\*(\d+)\*\* \|$", line)
        if m_tot:
            table["__total__"] = int(m_tot.group(1))

    for title, body in sections.items():
        assert table.get(title) == len(body["entries"]), (
            f"summary says {table.get(title)} for {title!r}, sections have "
            f"{len(body['entries'])}"
        )
    total = sum(len(b["entries"]) for b in sections.values())
    assert table.get("__total__") == total, (
        f"summary total {table.get('__total__')} != sum of sections {total} "
        f"-- the original document said 82 while its own categories summed "
        f"to 83"
    )


def test_no_entry_has_an_empty_or_missing_field(doc):
    entries = re.split(r"^### ", doc, flags=re.M)[1:]
    problems = []
    for chunk in entries:
        name = chunk.splitlines()[0].strip("` ")
        for field in REQUIRED_FIELDS:
            m = re.search(rf"^- \*\*{re.escape(field)}\*\*: (.*)$", chunk,
                          flags=re.M)
            if not m:
                problems.append(f"{name}: missing {field}")
            elif not m.group(1).strip():
                problems.append(f"{name}: empty {field}")
    assert not problems, problems


def test_no_trailing_whitespace(doc):
    """`git diff --check` failed on the first version because of exactly one
    trailing space, which is enough to block a release gate."""
    bad = [i + 1 for i, line in enumerate(doc.splitlines())
           if line != line.rstrip()]
    assert not bad, f"trailing whitespace on lines {bad}"


def test_cited_source_paths_exist(doc):
    """A path cited as evidence must be resolvable, or the entry is not
    evidence."""
    cited = set(re.findall(
        r"\b((?:python|c_impl)/[A-Za-z0-9_./-]+\.(?:py|c|h))\b", doc))
    assert cited, "no source paths cited at all -- parser broken?"
    missing = sorted(p for p in cited
                     if not os.path.isfile(os.path.join(_REPO, p)))
    # test/historical/ paths are deliberately retired; everything else must
    # exist in the working tree.
    missing = [p for p in missing if "/historical/" not in p]
    assert not missing, f"cited but absent: {missing}"


def test_retime_section_is_not_silently_empty(sections):
    """Guards the degenerate pass: a generator bug that emitted zero entries
    everywhere would satisfy every count check above."""
    assert len(sections["Retime"]["entries"]) > 0
    total = sum(len(b["entries"]) for b in sections.values())
    assert total >= 50, f"only {total} candidates -- generator likely truncated"
