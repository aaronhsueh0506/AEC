"""Guards for docs/c_python_function_map.md (the C <-> Python function map).

The map is a hand-maintained Markdown document; these tests keep it honest
without importing any algorithm module:

* every Python symbol the map names still exists (parsed with ``ast``);
* every C function is listed exactly once and no listed C function is stale;
* every Python symbol is either mapped or listed as Python-only;
* the coverage statistics in the document header match the tables;
* the gap ids, tool references and path hygiene hold.

Paths are derived from this file's own location (``<repo>/python/tests``), so
the same file works unchanged in ``AEC`` and in its mirror
``Audio_ALG/lib/aec``: each checkout is checked against its own ``python/``
and ``c_impl/``.  The two checkouts carry identical Python symbol sets and
identical C function sets (2026-10-20: the one prior textual difference, a
redundant duplicate NO_FLOOR rejection guard in this repo's
``AEC.set_preset()`` that ``lib/aec`` never carried, was removed here as part
of allowing the runtime retarget on both sides -- the two ``set_preset()``
bodies are now identical too).
"""

import ast
import collections
import os
import re

import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(_TESTS)
REPO = os.path.dirname(PY_ROOT)
C_ROOT = os.path.join(REPO, "c_impl")
DOC = os.path.join(REPO, "docs", "c_python_function_map.md")

COLUMNS = ["C 檔:函式", "Python 檔:符號", "關係", "已知落差", "對拍工具／測試"]
RELATIONS = ("mirror", "partial", "C-only", "Python-only", "未能判斷")
HEADERS = ("aec_balanced_config.h", "pbfdkf.h", "aec_debug.h")

_PY_TOKEN = re.compile(
    r"^(?P<path>[\w/]+\.py)(?::(?P<sym>[\w.]+))?(?:#(?P<anchor>.+))?$")
_C_TOKEN = re.compile(r"^(?P<dir>src|include)/(?P<file>[\w]+\.[ch]):(?P<fn>\w+)$")
_GAP = re.compile(r"^[SDT]-\d\d$")


# --------------------------------------------------------------------------
# document parsing
# --------------------------------------------------------------------------
def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


@pytest.fixture(scope="module")
def doc_text():
    assert os.path.isfile(DOC), f"missing {DOC}"
    with open(DOC, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def rows(doc_text):
    """Every data row of every map table, located by header text."""
    out = []
    lines = doc_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and _cells(lines[i]) == COLUMNS:
            j = i + 2                      # skip the |---| separator
            while j < len(lines) and lines[j].startswith("|"):
                c = _cells(lines[j])
                assert len(c) == 5, f"line {j + 1}: expected 5 cells, got {len(c)}"
                out.append(dict(line=j + 1, c=c[0], py=c[1], rel=c[2],
                                gap=c[3], tools=c[4]))
                j += 1
            i = j
        else:
            i += 1
    return out


def _ticks(cell):
    return [] if cell.strip() == "—" else re.findall(r"`([^`]+)`", cell)


def _rel_kind(rel):
    for k in RELATIONS:
        if rel.startswith(k):
            return k
    return None


# --------------------------------------------------------------------------
# Python side: ast only, never import
# --------------------------------------------------------------------------
def _py_files():
    files = [os.path.join(PY_ROOT, "aec.py")]
    for d, _, names in os.walk(os.path.join(PY_ROOT, "modules")):
        files += [os.path.join(d, n) for n in names if n.endswith(".py")]
    return sorted(files)


def _parse(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    return src, ast.parse(src)


def _py_universe():
    """{ 'modules/x.py:Class.method', ... } -- classes, functions, methods;
    nested closures inside functions are not symbols."""
    out = set()
    for p in _py_files():
        rel = os.path.relpath(p, PY_ROOT).replace(os.sep, "/")
        _, tree = _parse(p)

        def visit(body, prefix):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(f"{rel}:{prefix}{n.name}")
                elif isinstance(n, ast.ClassDef):
                    out.add(f"{rel}:{prefix}{n.name}")
                    visit(n.body, prefix + n.name + ".")
        visit(tree.body, "")
    return out


def _resolve(src, tree, sym):
    body, node = tree.body, None
    for part in sym.split("."):
        node = next((n for n in body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        body = node.body
    return node


# --------------------------------------------------------------------------
# C side: a small depth-0 function-definition scanner
# --------------------------------------------------------------------------
def _strip_c(src):
    src = re.sub(r'extern\s+"C"\s*\{', lambda m: " " * len(m.group(0)), src)
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n - 2 if j < 0 else j
            out.append(re.sub(r"[^\n]", " ", src[i:j + 2]))
            i = j + 2
        elif src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src[i] in "\"'":
            q, j = src[i], i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            seg = src[i:j + 1]
            out.append(q + re.sub(r"[^\n]", " ", seg[1:-1]) + q)
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    code = "".join(out).split("\n")
    cont = False
    for k, line in enumerate(code):
        if cont or line.lstrip().startswith("#"):
            cont = line.rstrip().endswith("\\")
            code[k] = " " * len(line)
    return "\n".join(code)


def _c_function_names(path):
    with open(path, encoding="utf-8") as f:
        code = _strip_c(f.read())
    names, depth, start = set(), 0, 0
    for i, ch in enumerate(code):
        if ch == "{":
            if depth == 0:
                head = code[start:i].strip()
                if head.endswith(")") and not re.match(
                        r"^(typedef|struct|enum|union)\b", head):
                    d, k = 0, len(head) - 1
                    while k >= 0:
                        d += head[k] == ")"
                        if head[k] == "(":
                            d -= 1
                            if d == 0:
                                break
                        k -= 1
                    m = re.search(r"([A-Za-z_]\w*)$", head[:k].rstrip())
                    if m and m.group(1) not in (
                            "if", "for", "while", "switch", "return", "sizeof"):
                        names.add(m.group(1))
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
            if depth == 0:
                start = i + 1
        elif ch == ";" and depth == 0:
            start = i + 1
    return names


def _c_universe():
    out = set()
    src_dir = os.path.join(C_ROOT, "src")
    for n in sorted(os.listdir(src_dir)):
        if n.endswith(".c"):
            out |= {f"src/{n}:{f}" for f in _c_function_names(os.path.join(src_dir, n))}
    for h in HEADERS:
        names = _c_function_names(os.path.join(C_ROOT, "include", h))
        out |= {f"include/{h}:{f}" for f in names}
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_document_has_the_expected_tables(rows):
    assert len(rows) > 400, f"only {len(rows)} rows parsed -- header text changed?"


def test_row_shape_and_relation_rules(rows, doc_text):
    headers = " ".join(
        open(os.path.join(C_ROOT, "include", h), encoding="utf-8").read()
        for h in sorted(os.listdir(os.path.join(C_ROOT, "include")))
        if h.endswith(".h"))
    defined_gaps = set(re.findall(r"^\| ([SDT]-\d\d) \|", doc_text, re.M))
    bad = []
    seen_c = collections.Counter()
    for r in rows:
        kind = _rel_kind(r["rel"])
        py = _ticks(r["py"])
        if kind is None:
            bad.append((r["line"], "unknown relation", r["rel"][:30]))
            continue
        if kind == "Python-only":
            cc = _ticks(r["c"])
            if r["c"] != "—":
                m = re.match(r"^struct:(\w+)$", cc[0] if cc else "")
                if not m or not re.search(r"\b%s\b" % m.group(1), headers):
                    bad.append((r["line"], "bad struct reference", r["c"]))
            if not py:
                bad.append((r["line"], "Python-only row without python symbol", ""))
        else:
            cc = _ticks(r["c"])
            if len(cc) != 1 or not _C_TOKEN.match(cc[0]):
                bad.append((r["line"], "bad C cell", r["c"]))
            else:
                seen_c[cc[0]] += 1
            if kind in ("mirror", "partial") and not py:
                bad.append((r["line"], f"{kind} without python symbol", r["c"]))
            if kind in ("C-only", "未能判斷") and py:
                bad.append((r["line"], f"{kind} must not name a python symbol", r["c"]))
        for tok in py:
            if not _PY_TOKEN.match(tok):
                bad.append((r["line"], "malformed python token", tok))
        if r["gap"] != "—":
            for g in r["gap"].split("；"):
                if not _GAP.match(g) or g not in defined_gaps:
                    bad.append((r["line"], "gap id not defined in appendix B", g))
    bad += [(0, "C function listed twice", k) for k, v in seen_c.items() if v > 1]
    assert not bad, "\n".join(map(str, bad[:40]))


def test_every_python_symbol_named_in_the_map_exists(rows):
    """The check this file exists for: a renamed or deleted Python symbol
    (or a vanished anchor string) breaks the map and fails here."""
    missing = []
    cache = {}
    for r in rows:
        for tok in _ticks(r["py"]):
            m = _PY_TOKEN.match(tok)
            if not m:
                continue                       # reported by the shape test
            path = os.path.join(PY_ROOT, m.group("path"))
            if not os.path.isfile(path):
                missing.append((r["line"], tok, "file missing"))
                continue
            if path not in cache:
                cache[path] = _parse(path)
            src, tree = cache[path]
            sym = m.group("sym")
            if not sym:
                continue                       # bare file path: existence only
            node = _resolve(src, tree, sym)
            if node is None:
                missing.append((r["line"], tok, "symbol missing"))
            elif m.group("anchor") and m.group("anchor") not in (
                    ast.get_source_segment(src, node) or ""):
                missing.append((r["line"], tok, "anchor not in symbol body"))
    assert not missing, "\n".join(map(str, missing[:40]))


def test_every_python_symbol_is_mapped_or_listed_python_only(rows):
    mapped, only = set(), collections.Counter()
    for r in rows:
        toks = [_PY_TOKEN.match(t) for t in _ticks(r["py"])]
        keys = [f"{m.group('path')}:{m.group('sym')}" for m in toks if m and m.group("sym")]
        if _rel_kind(r["rel"]) == "Python-only":
            only.update(keys)
        else:
            mapped.update(keys)
    universe = _py_universe()
    problems = []
    problems += [f"not in map: {k}" for k in sorted(universe - mapped - set(only))]
    problems += [f"both mapped and Python-only: {k}" for k in sorted(mapped & set(only))]
    problems += [f"Python-only listed twice: {k}" for k, v in only.items() if v > 1]
    assert not problems, "\n".join(problems[:40])


def test_every_c_function_is_listed_exactly_once(rows):
    listed = collections.Counter()
    for r in rows:
        if _rel_kind(r["rel"]) == "Python-only":
            continue
        for tok in _ticks(r["c"]):
            listed[tok] += 1
    universe = _c_universe()
    problems = []
    problems += [f"C function without a row: {k}" for k in sorted(universe - set(listed))]
    problems += [f"row names a C function that no longer exists: {k}"
                 for k in sorted(set(listed) - universe)]
    problems += [f"listed twice: {k}" for k, v in listed.items() if v > 1]
    assert not problems, "\n".join(problems[:40])


def test_stats_block_matches_tables(rows, doc_text):
    m = re.search(r"<!-- stats:begin -->(.*?)<!-- stats:end -->", doc_text, re.S)
    assert m, "stats block markers missing"
    stated = {k: int(v) for k, v in re.findall(
        r"^\| `(\w+)` \| (\d+) \|", m.group(1), re.M)}
    src, hdr = collections.Counter(), collections.Counter()
    inferred = 0
    for r in rows:
        kind = _rel_kind(r["rel"])
        if kind == "Python-only":
            continue
        (src if r["c"].startswith("`src/") else hdr)[kind] += 1
        inferred += "推定" in r["rel"]
    only = collections.Counter()
    mapped = set()
    for r in rows:
        keys = [t.split("#")[0] for t in _ticks(r["py"]) if ":" in t]
        if _rel_kind(r["rel"]) == "Python-only":
            kind = ("struct" if r["rel"].startswith("Python-only（C 以資料結構") else
                    "field" if r["rel"].startswith("Python-only（C 以結構欄位") else
                    "unknown" if r["rel"].startswith("Python-only（未能判斷") else "none")
            only[kind] += len(keys)
        else:
            mapped.update(keys)
    actual = {
        "c_src_total": sum(src.values()), "c_src_mirror": src["mirror"],
        "c_src_partial": src["partial"], "c_src_c_only": src["C-only"],
        "c_src_unknown": src["未能判斷"], "c_hdr_total": sum(hdr.values()),
        "c_hdr_c_only": hdr["C-only"], "c_hdr_mapped": hdr["mirror"] + hdr["partial"],
        "c_total": sum(src.values()) + sum(hdr.values()),
        "c_mapped": src["mirror"] + src["partial"] + hdr["mirror"] + hdr["partial"],
        "c_c_only": src["C-only"] + hdr["C-only"],
        "c_unknown": src["未能判斷"] + hdr["未能判斷"], "inferred": inferred,
        "py_total": len(_py_universe()), "py_mapped": len(mapped),
        "py_only_struct": only["struct"], "py_only_field": only["field"],
        "py_only_none": only["none"], "py_unknown": only["unknown"],
    }
    diff = {k: (stated.get(k), v) for k, v in actual.items() if stated.get(k) != v}
    assert not diff, f"stats block out of date (stated, actual): {diff}"


def test_tool_references_point_at_existing_files(rows):
    test_dir = os.path.join(C_ROOT, "test")
    names = {n for d, dirs, files in os.walk(test_dir)
             if os.sep + "historical" not in d + os.sep for n in files}
    makefile = open(os.path.join(C_ROOT, "Makefile"), encoding="utf-8").read()
    bad = []
    for r in rows:
        for tag in _ticks(r["tools"]):
            kind, _, arg = tag.partition(":")
            if kind == "live":
                ok = re.search(r"^%s:" % re.escape(arg), makefile, re.M)
            elif kind == "e2e":
                ok = (os.path.isfile(os.path.join(test_dir, arg + ".c")) and
                      os.path.isfile(os.path.join(PY_ROOT, "diag", "gen_aec_e2e_golden.py")))
            elif kind == "hist":
                m = re.match(r"^(\w+)\((PASS|FAIL|NOGEN|NOBUILD|NOTRUN)"
                             r"(\+(NOGEN|NOBUILD))?\)$", arg)
                ok = m and os.path.isfile(os.path.join(
                    test_dir, "historical", f"parity_{m.group(1)}.c"))
            elif kind in ("manual", "ctest"):
                ok = arg in names
            elif kind == "pytest":
                ok = os.path.isfile(os.path.join(_TESTS, arg))
            else:
                ok = False
            if not ok:
                bad.append((r["line"], tag))
    assert not bad, "\n".join(map(str, bad[:40]))


def test_document_contains_no_absolute_paths(doc_text):
    """Machine-specific absolute paths do not belong in a shipped document."""
    roots = "|".join("/%s/" % d for d in ("Users", "private", "home", "tmp"))
    hits = [(n + 1, m) for n, line in enumerate(doc_text.splitlines())
            for m in re.findall(r"(?:%s|[A-Za-z]:\\)\S*" % roots, line)]
    assert not hits, hits[:10]
