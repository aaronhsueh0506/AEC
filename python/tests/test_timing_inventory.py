"""Gate for the timing-constant inventory: the data, the generator, the document.

The inventory is release evidence -- it is what answers "is the wall-clock
timing audit complete?". Its first version could not answer that, because it was
internally inconsistent: the category counts summed to 83 against a stated total
of 82, two of the six categories had no section at all, one candidate carried two
mutually exclusive verdicts, five entries were truncated mid-sentence, and a
trailing space broke `git diff --check`. None of that was visible without adding
up the table by hand.

So the document is generated, and this file asserts three separate things:

1. the rendered document has the properties a reader relies on when treating it
   as evidence (the original failure mode);
2. `--check` really verifies, really refuses to write, and really fails when the
   committed text has drifted -- proved by mutating a sandbox copy, not by
   trusting the exit code of the happy path;
3. the whole artifact rebuilds from repository state alone. The second version
   of the generator read an absolute path under a developer's home directory, so
   the document was reproducible on exactly one machine.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DOC = os.path.join(_REPO, "docs", "timing_constant_inventory.md")
_DATA = os.path.join(_REPO, "docs", "timing_constant_inventory.json")
_GEN = os.path.join(_REPO, "python", "diag", "gen_timing_inventory.py")

# Every category the generator may emit. A candidate lands in exactly one.
EXPECTED_SECTIONS = [
    "Retime",
    "Already retimed",
    "Keep: rejected retime",
    "Keep: event count",
    "Keep: fixed internal cadence",
    "Keep: dominated default",
    "Dead / unreachable",
    "Not a timing constant",
]

REQUIRED_FIELDS = ["Verdict", "Anchor", "Effective today", "Consumer",
                   "Provenance", "Reasoning"]

# Words that only mean something from inside the work that produced the
# document. A release reader has no referent for them.
BANNED = ["parent", "brief", "intake", "subagent", "adjudication"]


@pytest.fixture(scope="module")
def doc():
    assert os.path.isfile(_DOC), f"inventory missing: {_DOC}"
    with open(_DOC, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def data():
    with open(_DATA, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def sections(doc):
    """{title: {declared, entries}} in document order."""
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


# ── the rendered document ───────────────────────────────────────────────────

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
            m = re.search(rf"^- \*\*{re.escape(field)}\*\*:(.*)$", chunk,
                          flags=re.M)
            if not m:
                problems.append(f"{name}: missing {field}")
            elif not m.group(1).strip() and field != "Consumer":
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


# ── the data ────────────────────────────────────────────────────────────────

def test_one_entry_per_semantic_constant(data):
    """alpha_power and alpha_r were each filed twice, once per port, so a single
    unretimed value read as two findings and one port could be closed while the
    other stayed open under a different name. Both ports now share one entry,
    and the Consumer list is what carries the per-port detail."""
    both = [e for e in data["entries"]
            if any("Python:" in c for c in e["consumers"])
            or any(c.startswith("C:") for c in e["consumers"])]
    assert both, "no per-port consumer lists at all -- convention lost?"
    for e in both:
        ports = {c.split(":", 1)[0] for c in e["consumers"]}
        assert len(e["consumers"]) == len(ports), (
            f"{e['name']}: repeated port label in {ports}")


def test_status_and_category_cannot_disagree(data):
    for e in data["entries"]:
        assert (e["category"] == "retime") == (e["status"] == "open"), (
            f"{e['name']}: status {e['status']!r} with category "
            f"{e['category']!r}")


def test_the_two_alpha_bypasses_are_closed(data):
    """Both were fixed at HEAD (5232ab6, d7e94f7). An inventory that still calls
    them hardcoded would send a reader to re-fix a fixed bug."""
    by_id = {e["id"]: e for e in data["entries"]}
    for cid in ("pbfdaf-alpha-power", "pbfdkf-alpha-r"):
        e = by_id[cid]
        assert e["category"] == "already-retimed" and e["status"] == "closed", (
            f"{cid} is {e['category']}/{e['status']}")
        c_side = [c for c in e["consumers"] if c.startswith("C:")]
        assert len(c_side) == 1, f"{cid}: expected exactly one C consumer"
        assert "never read" not in c_side[0], (
            f"{cid}: the C consumer still describes the pre-fix state")


@pytest.mark.parametrize("path", [_DOC, _DATA])
def test_no_process_vocabulary(path):
    """The prose was full of words that only mean something inside the work that
    produced the document. Matched on word boundaries, because `TransparentMode`
    is an identifier -- a naive substring check on "parent" made an earlier pass
    elide it out of a quoted commit subject."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    hits = {}
    for word in BANNED:
        found = re.findall(rf"\b{re.escape(word)}\w*\b", text, flags=re.I)
        if found:
            hits[word] = sorted(set(found))
    assert not hits, f"{os.path.basename(path)}: {hits}"


@pytest.mark.parametrize("path", [_DOC, _DATA])
def test_no_absolute_paths_in_the_document_or_the_data(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    bad = sorted(set(re.findall(r"(?<![\w.])/[A-Za-z][\w.-]*/[\w./-]*", text)))
    assert not bad, f"{os.path.basename(path)}: {bad[:5]}"


def test_the_generator_holds_no_absolute_path_literal():
    """THE reproducibility rule. The previous generator read the adjudication
    data from an absolute path under a developer's home directory, so the
    document could be rebuilt on exactly one machine. Checking string literals
    rather than grepping for one particular home prefix means a different
    machine layout cannot slip past."""
    import ast
    with open(_GEN, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=_GEN)
    bad = [node.value for node in ast.walk(tree)
           if isinstance(node, ast.Constant) and isinstance(node.value, str)
           and node.value.startswith("/")]
    assert not bad, f"absolute path literals in the generator: {bad}"


# ── the generator ───────────────────────────────────────────────────────────

def _run(*args, cwd=None):
    return subprocess.run([sys.executable, _GEN, *args], cwd=cwd or _REPO,
                          capture_output=True, text=True)


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_check_passes_against_the_committed_document():
    r = _run("--check")
    assert r.returncode == 0, (
        f"--check failed; regenerate with --write\n{r.stdout}\n{r.stderr}")


def test_check_does_not_write():
    """--check is used in review and in CI; it must be safe to run against a
    clean tree."""
    before_digest, before_stat = _digest(_DOC), os.stat(_DOC)
    r = _run("--check")
    assert r.returncode == 0
    assert _digest(_DOC) == before_digest, "--check rewrote the document"
    after_stat = os.stat(_DOC)
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns, (
        "--check touched the document's mtime")
    assert after_stat.st_size == before_stat.st_size


def test_a_mode_is_required_and_the_two_modes_are_exclusive():
    assert _run().returncode != 0, "running with no mode must fail"
    assert _run("--write", "--check").returncode != 0, (
        "--write and --check must be mutually exclusive")


@pytest.fixture
def sandbox(tmp_path):
    """A throwaway repo root: real docs/ and python/diag/, everything else
    symlinked, so cited source paths still resolve and the generator's
    Path(__file__)-derived root lands inside the sandbox."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "python" / "diag").mkdir(parents=True)
    shutil.copy2(_DATA, root / "docs" / "timing_constant_inventory.json")
    shutil.copy2(_DOC, root / "docs" / "timing_constant_inventory.md")
    shutil.copy2(_GEN, root / "python" / "diag" / "gen_timing_inventory.py")
    for name in os.listdir(os.path.join(_REPO, "docs")):
        if not name.startswith("timing_constant_inventory."):
            os.symlink(os.path.join(_REPO, "docs", name), root / "docs" / name)
    for name in os.listdir(os.path.join(_REPO, "python")):
        if name != "diag":
            os.symlink(os.path.join(_REPO, "python", name),
                       root / "python" / name)
    for name in os.listdir(os.path.join(_REPO, "python", "diag")):
        if name != "gen_timing_inventory.py":
            os.symlink(os.path.join(_REPO, "python", "diag", name),
                       root / "python" / "diag" / name)
    for name in ("c_impl", "eval"):
        os.symlink(os.path.join(_REPO, name), root / name)
    return root


def _sandbox_run(root, *args):
    gen = str(root / "python" / "diag" / "gen_timing_inventory.py")
    return subprocess.run([sys.executable, gen, *args], cwd=str(root),
                          capture_output=True, text=True)


def test_sandbox_reproduces_the_committed_document(sandbox):
    """The reproducibility claim itself: a tree containing only repository
    state rebuilds the document byte for byte."""
    assert _sandbox_run(sandbox, "--check").returncode == 0
    md = sandbox / "docs" / "timing_constant_inventory.md"
    md.write_text("clobbered\n", encoding="utf-8")
    assert _sandbox_run(sandbox, "--write").returncode == 0
    assert _digest(str(md)) == _digest(_DOC), (
        "--write did not reproduce the committed document byte for byte")


def test_check_detects_a_hand_edit_and_prints_a_diff(sandbox):
    """MUTATION: the happy path returning 0 proves nothing on its own."""
    md = sandbox / "docs" / "timing_constant_inventory.md"
    text = md.read_text(encoding="utf-8")
    md.write_text(text.replace("| **total** |", "| **grand total** |", 1),
                  encoding="utf-8")
    r = _sandbox_run(sandbox, "--check")
    assert r.returncode == 1, "a hand-edited document must fail --check"
    assert "grand total" in r.stdout, "no unified diff was printed"
    assert md.read_text(encoding="utf-8") != r.stdout, "--check wrote the file"


def test_check_detects_data_that_diverges_from_the_document(sandbox):
    """The other direction: editing the JSON without regenerating."""
    js = sandbox / "docs" / "timing_constant_inventory.json"
    d = json.loads(js.read_text(encoding="utf-8"))
    d["entries"][0]["confidence"] = "medium" \
        if d["entries"][0]["confidence"] == "high" else "high"
    js.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    assert _sandbox_run(sandbox, "--check").returncode == 1


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d["entries"][0].update(status="open", category="dead-code"),
     "status"),
    (lambda d: d["entries"][0].update(reasoning="handed to the parent"),
     "parent"),
    (lambda d: d["entries"][0].update(id=d["entries"][1]["id"]),
     "duplicate id"),
    (lambda d: d["entries"][0].update(consumers=[]),
     "empty consumers"),
    (lambda d: d.update(schema_version=99),
     "schema_version"),
])
def test_integrity_rules_actually_fail(sandbox, mutate, expect):
    """MUTATION per rule. A validator that never rejects anything is the same
    defect as a test that cannot fail."""
    js = sandbox / "docs" / "timing_constant_inventory.json"
    d = json.loads(js.read_text(encoding="utf-8"))
    mutate(d)
    js.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    r = _sandbox_run(sandbox, "--check")
    assert r.returncode == 2, (
        f"integrity rule did not fire (rc={r.returncode})\n{r.stdout}\n"
        f"{r.stderr}")
    assert expect in r.stderr, f"{expect!r} not reported:\n{r.stderr}"
