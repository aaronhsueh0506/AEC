"""Render docs/timing_constant_inventory.md from docs/timing_constant_inventory.json.

The Markdown is derived, never hand-edited. Regenerating is how the counts, the
category exclusivity and the absence of truncation stay true; editing in place
would reintroduce exactly the drift that made the first version of that document
unusable as release evidence.

Both inputs live in this repository and every path is derived from this file's
own location, so the document rebuilds in a clean clone with no local state.

    python3 python/diag/gen_timing_inventory.py --write   # regenerate
    python3 python/diag/gen_timing_inventory.py --check    # verify, never writes

--check exits 0 when the committed Markdown matches, 1 when it diverges (and
prints a unified diff), and 2 when the JSON itself fails an integrity rule.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "timing_constant_inventory.json"
DOC = REPO / "docs" / "timing_constant_inventory.md"

SCHEMA_VERSION = 1

# Exclusive and ordered. A candidate lands in exactly one.
CATEGORIES = [
    ("retime", "Retime",
     "Live on the default-ON audio path, semantically a wall-clock duration, "
     "and not currently routed through a retiming helper."),
    ("already-retimed", "Already retimed",
     "Verified to route through a retiming helper. Listed so that a future "
     "audit can tell 'checked and correct' from 'never looked at'."),
    ("keep-event-count", "Keep: event count",
     "The counter is SKIPPED (early-returned past) on non-qualifying hops, so "
     "its realised span is duty-cycle dependent and N means N qualifying "
     "observations, not N hops of time. Retiming these would be wrong."),
    ("keep-fixed-cadence", "Keep: fixed internal cadence",
     "Runs on an internal cadence that does not change with the external hop "
     "(AEC3's 4 ms block, a matched-filter sub-block, a per-window update). "
     "The external hop is not its clock, so external-hop retiming does not "
     "apply."),
    ("keep-dominated-default", "Keep: dominated default",
     "A configurable threshold that is not the binding term at the shipped "
     "default, so retiming it is a provable no-op -- but it is public, and a "
     "caller can make it binding, so it cannot be deleted either."),
    ("dead-code", "Dead / unreachable",
     "Written and never read, or gated behind a condition that cannot hold on "
     "the production path. Listed rather than silently dropped: dead state in "
     "a tightly coupled 4500-line file reads as live invariant maintenance."),
    ("not-a-timing-constant", "Not a timing constant",
     "Dimensionless, or a size/offset in samples/bins/blocks that carries no "
     "wall-clock meaning."),
]
KNOWN = {key for key, _, _ in CATEGORIES}
STATUSES = {"open", "closed"}
TEXT_FIELDS = ("anchor", "effective", "provenance", "reasoning")

PATH_RE = re.compile(r"\b((?:python|c_impl|docs|eval)/[A-Za-z0-9_./-]+"
                     r"\.(?:py|c|h|md|json))\b")

# This document is read by people outside the work that produced it. Words that
# only make sense from inside that work are a defect, not a style preference.
# Matched on word boundaries: `TransparentMode` and `enable_transparent_mode`
# are identifiers, and a naive substring check on "parent" made an earlier pass
# elide them out of a quoted commit subject.
BANNED = (r"\bparent\b", r"\bbrief\w*\b", r"\bintake\b", r"\bsubagent\b",
          r"\badjudicat\w*\b")

# An absolute path in the data is the same defect in a different costume: it
# pins the document to one machine.
ABSOLUTE_PATH = re.compile(r"(?<![\w.])/[A-Za-z][\w.-]*/")


def load():
    with DATA.open(encoding="utf-8") as fh:
        return json.load(fh)


def validate(data):
    """Return a list of integrity failures. Empty means the data is usable."""
    problems = []
    if data.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {data.get('schema_version')!r} != "
                        f"{SCHEMA_VERSION}")
    entries = data.get("entries") or []
    if not entries:
        problems.append("no entries")

    ids = collections.Counter(e.get("id") for e in entries)
    for cid, n in sorted(ids.items()):
        if n > 1:
            cats = sorted({e["category"] for e in entries if e.get("id") == cid})
            problems.append(f"duplicate id {cid!r} x{n}: {', '.join(cats)}")

    for e in entries:
        name = e.get("name", "<unnamed>")
        if e.get("category") not in KNOWN:
            problems.append(f"unknown category {e.get('category')!r} for {name!r}")
        if e.get("status") not in STATUSES:
            problems.append(f"unknown status {e.get('status')!r} for {name!r}")
        if (e.get("category") == "retime") != (e.get("status") == "open"):
            problems.append(f"{name!r}: status {e.get('status')!r} contradicts "
                            f"category {e.get('category')!r} -- a retime is the "
                            f"only open verdict")
        for field in ("id", "name", "confidence") + TEXT_FIELDS:
            if not str(e.get(field, "")).strip():
                problems.append(f"empty {field} for {name!r}")
        consumers = e.get("consumers") or []
        if not consumers or not all(str(c).strip() for c in consumers):
            problems.append(f"empty consumers for {name!r}")

    for e in entries:
        for field, value in _values(e):
            for pattern in BANNED:
                m = re.search(pattern, value, flags=re.I)
                if m:
                    problems.append(
                        f"{e.get('name')!r} {field}: contains {m.group(0)!r}, "
                        f"which describes how the audit was run rather than "
                        f"what the constant does")
            m = ABSOLUTE_PATH.search(value)
            if m:
                problems.append(
                    f"{e.get('name')!r} {field}: absolute path {m.group(0)!r} "
                    f"-- cite repository-relative paths only")
    return problems


def _values(entry):
    for field in TEXT_FIELDS:
        yield field, str(entry.get(field, ""))
    for i, c in enumerate(entry.get("consumers") or []):
        yield f"consumers[{i}]", str(c)


def cited_paths(entries):
    cited, missing = set(), set()
    for e in entries:
        for _, value in _values(e):
            for m in PATH_RE.finditer(value):
                p = m.group(1)
                cited.add(p)
                if not (REPO / p).is_file():
                    missing.add(p)
    return cited, missing


def clean(text):
    return " ".join(str(text).split())


def render(data):
    entries = data["entries"]
    by_cat = collections.OrderedDict(
        (key, [e for e in entries if e["category"] == key])
        for key, _, _ in CATEGORIES)
    total = sum(len(v) for v in by_cat.values())
    assert total == len(entries), (total, len(entries))
    cited, missing = cited_paths(entries)
    open_count = sum(1 for e in entries if e["status"] == "open")

    L = []
    A = L.append
    A("# Hop-authored timing-constant inventory")
    A("")
    A(f"Generated from `docs/timing_constant_inventory.json` by "
      f"`python/diag/gen_timing_inventory.py`. **{total} candidates**, each in "
      f"exactly one category; **{open_count} still open**.")
    A("")
    A("This file is GENERATED, not hand-edited. Regenerating is how the counts, "
      "the category exclusivity and the absence of truncation stay true; an "
      "edit-in-place would reintroduce exactly the drift that made the first "
      "version of this document unusable as release evidence. "
      "`gen_timing_inventory.py --check` rebuilds it in memory and fails if the "
      "committed text has drifted.")
    A("")
    A("## Conventions")
    A("")
    A("**One entry per semantic constant.** A constant that exists in both "
      "ports is one candidate with one verdict and one anchor; its Consumer "
      "field lists every consumption site, one per port. Counting the two "
      "ports separately would make a single unretimed value look like two "
      "findings, and would let one port be closed while the other stayed open "
      "under a different name.")
    A("")
    A("**Anchor rule.** The wall-clock span is taken from the commit that last "
      "*empirically validated* the constant -- one that ran a benchmark and "
      "accepted the result -- not the commit that introduced it, and not the "
      "in-code comment. These disagree often enough to matter: "
      "`simple_mu_holdoff` was introduced on a 16 ms grid and validated on a "
      "10 ms grid, so anchoring on the introduction commit is a 1.6x error. "
      "Where no commit ever measured the constant, the entry says so and stays "
      "unresolved rather than guessing.")
    A("")
    A("**EMA definitions are exact-pole throughout.** The `1/(1-alpha)` "
      "approximation is not used as a time constant anywhere in this document; "
      "where an approximate span is quoted it is labelled as such.")
    A("")
    A("```text")
    A("retention convention (state = a*state + (1-a)*x):")
    A("    tau      = -dt / ln(a)")
    A("    a_new    = a_ref ** (dt_new / dt_ref)")
    A("")
    A("new-weight convention (state += w*(x - state)), w = 1 - a:")
    A("    w_new    = 1 - (1 - w_ref) ** (dt_new / dt_ref)")
    A("")
    A("additive per-hop leak (dB/hop): scales LINEARLY, not by a power law:")
    A("    leak_new = leak_ref * (dt_new / dt_ref)")
    A("```")
    A("")
    A("**Shipped grids.** 8k 256/128, 16k 256/128 (default), 16k 512/256, "
      "48k 1024/512.")
    A("")
    A("## Summary")
    A("")
    A("| category | count |")
    A("|---|---:|")
    for key, title, _ in CATEGORIES:
        A(f"| {title} | {len(by_cat[key])} |")
    A(f"| **total** | **{total}** |")
    A("")
    if missing:
        A("### Source paths cited but not present")
        A("")
        A("These are cited by an entry below and do not exist in the tree. Each "
          "is a path retired by the release cleanup; the entry's reasoning "
          "still stands but the file must be recovered from git history.")
        A("")
        for p in sorted(missing):
            A(f"- `{p}`")
        A("")
    else:
        A(f"All {len(cited)} distinct source paths cited below exist in the "
          f"tree.")
        A("")

    for key, title, blurb in CATEGORIES:
        group = sorted(by_cat[key], key=lambda e: (e["confidence"] != "high",
                                                   e["name"].lower()))
        A("---")
        A("")
        A(f"## {title} ({len(group)})")
        A("")
        A(clean(blurb))
        A("")
        if not group:
            A("_None._")
            A("")
            continue
        if key == "retime":
            A("| # | constant | anchor | confidence |")
            A("|---:|---|---|---|")
            for i, e in enumerate(group, 1):
                A(f"| {i} | `{clean(e['name'])}` | {clean(e['anchor'])} "
                  f"| {e['confidence']} |")
            A("")
        for e in group:
            A(f"### `{clean(e['name'])}`")
            A("")
            A(f"- **Verdict**: {title.lower()} ({e['confidence']} confidence, "
              f"{e['status']})")
            A(f"- **Anchor**: {clean(e['anchor'])}")
            A(f"- **Effective today**: {clean(e['effective'])}")
            consumers = e["consumers"]
            if len(consumers) == 1:
                A(f"- **Consumer**: {clean(consumers[0])}")
            else:
                A("- **Consumer**:")
                for c in consumers:
                    A(f"  - {clean(c)}")
            A(f"- **Provenance**: {clean(e['provenance'])}")
            A(f"- **Reasoning**: {clean(e['reasoning'])}")
            A("")

    text = "\n".join(line.rstrip() for line in L).rstrip() + "\n"
    assert "  \n" not in text and " \n" not in text
    return text


def write_atomic(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                              suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true",
                      help="regenerate the Markdown from the JSON")
    mode.add_argument("--check", action="store_true",
                      help="verify the committed Markdown; never writes")
    args = ap.parse_args(argv)

    data = load()
    problems = validate(data)
    if problems:
        print("INTEGRITY FAILURES:", *problems, sep="\n  ", file=sys.stderr)
        return 2

    text = render(data)

    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
        if current == text:
            print(f"OK: {DOC.relative_to(REPO)} matches "
                  f"{DATA.relative_to(REPO)} ({len(data['entries'])} entries)")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile=f"a/{DOC.relative_to(REPO)} (committed)",
            tofile=f"b/{DOC.relative_to(REPO)} (regenerated)")
        sys.stdout.writelines(diff)
        print(f"\nDIVERGED: run "
              f"`python3 python/diag/gen_timing_inventory.py --write`",
              file=sys.stderr)
        return 1

    write_atomic(DOC, text)
    entries = data["entries"]
    print(f"wrote {DOC.relative_to(REPO)}")
    print(f"  {len(entries)} candidates in {len(CATEGORIES)} categories")
    for key, title, _ in CATEGORIES:
        print(f"    {title:<34} "
              f"{sum(1 for e in entries if e['category'] == key)}")
    cited, missing = cited_paths(entries)
    print(f"  cited paths: {len(cited)}, missing: {len(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
