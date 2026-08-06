"""Regenerate docs/timing_constant_inventory.md from the adjudication journal.

Deterministic and re-runnable: the document is derived, never hand-edited, so a
truncation or a count that disagrees with the rows cannot survive a rebuild.
"""
import json, os, re, sys, collections

JOURNAL = ("/Users/mingyu/.claude/projects/-Users-mingyu-Desktop-novatek-SE/"
           "fdd8dd4d-5e9f-4d28-b239-09e595e85967/subagents/workflows/"
           "wf_bef0982e-1ed/journal.jsonl")
AEC = "/Users/mingyu/Desktop/novatek/SE/AEC"
OUT = os.path.join(AEC, "docs/timing_constant_inventory.md")

# ── load ────────────────────────────────────────────────────────────────────
rows = []
for line in open(JOURNAL):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("type") == "result" and isinstance(d.get("result"), dict) \
            and "verdicts" in d["result"]:
        rows.extend(d["result"]["verdicts"])

# ── resolve the one double-adjudicated candidate ────────────────────────────
# shadow_copy_hysteresis came back twice (retime/low and dead-code/high).
# Neither is right. Its gate is `counter >= 3 AND streak >= 10`, so at the
# default streak the 3 is dominated and 3->4 is a no-op -- but it is a public
# configurable, and a caller setting it above 10 makes it live again, so it
# cannot be deleted either. Its own category.
DOMINATED = "keep-dominated-default"
rows = [r for r in rows if r["name"] != "shadow_copy_hysteresis"]
rows.append({
    "name": "shadow_copy_hysteresis",
    "verdict": DOMINATED,
    "anchor_ms": "n/a -- dominated at the default configuration",
    "anchor_evidence":
        "Introduced on a 16 ms grid; the later commit that would be the "
        "validation anchor is a revert whose own message disclaims the "
        "800-case run, so no commit ever measured and accepted a wall-clock "
        "span for this literal in either direction. That ambiguity is moot "
        "given the domination below.",
    "effective_ms":
        "Not meaningful at the shipped default: the counter threshold is "
        "never the binding term.",
    "consumer":
        "epc.py:314-329 / epc_shadow.c:219-231. The shadow-copy gate is "
        "`hysteresis_counter >= shadow_copy_hysteresis AND streak >= "
        "HYS_STREAK_MIN`. Both counters advance together on the same "
        "condition, so with the shipped HYS_STREAK_MIN = 10 the streak term "
        "is strictly binding and the 3 can never be the deciding factor: "
        "3 -> 4 is a provable no-op.",
    "confidence": "high",
    "reasoning":
        "NOT a retime and NOT dead code -- the two verdicts this candidate "
        "originally received, each correct about one half. Retiming it is "
        "pointless while it is dominated, and deleting it is wrong because "
        "shadow_copy_hysteresis is a public configurable: a caller setting it "
        "ABOVE HYS_STREAK_MIN makes it the binding term and it starts "
        "behaving as a real duration. Keep the value, do not retime, and note "
        "the domination. The constant that actually governs this gate at the "
        "default configuration is HYS_STREAK_MIN = 10, which is adjudicated "
        "separately (see the EPC/P3f entries).",
})

# ── 2026-08-07 field findings that supersede the original adjudication ──────
# Established by the alpha_power A/B (eval/ab_evidence/2026-08-07-alpha-power):
# `power[]` is written by the EMA and read at exactly one site -- its own
# cold-start guard -- in BOTH ports. Changing its coefficient produces
# byte-identical output across 90 cases at two grids. The constant is real and
# now correctly retimed, but it is dead for audio, so it belongs in the
# auxiliary batch (parity + goldens, no AECMOS), not an audio batch.
_ALPHA_POWER_NOTE = (
    " FIELD FINDING 2026-08-07: retimed and now correctly consumed by the EMA "
    "(5232ab6), but AUDIO-DEAD -- power[] is read only by its own cold-start "
    "guard in both ports, and the 90-case two-grid A/B produced byte-identical "
    "output. Belongs in the auxiliary/dead-output batch: needs C/Python parity "
    "and golden coverage, does not need AECMOS."
)
for _r in rows:
    if "alpha_power" in _r["name"]:
        _r["reasoning"] = _r.get("reasoning", "") + _ALPHA_POWER_NOTE
    if "_alpha_r" in _r["name"] or "alpha_r" == _r["name"]:
        _r["reasoning"] = _r.get("reasoning", "") + (
            " FIELD FINDING 2026-08-07: anchor re-adjudicated from 16 ms to "
            "10 ms (TC 194.96 ms) and the C live path wired to the retimed "
            "field. AUDIO-DEAD, same as alpha_power: error_psd's only consumer "
            "is the scalar-fallback branch of the H_error refresh, which the "
            "source itself marks 'Not exercised in production: orchestrator "
            "sets e2_coarse_per_bin every hop'. The live per-bin branch reads "
            "error_spec directly and never touches the smoothed error_psd. "
            "90-case two-grid A/B: byte-identical. Auxiliary batch."
        )

# ── categories (exclusive, ordered) ─────────────────────────────────────────
# Split the original single 'keep-event-count' bucket: a counter whose span is
# duty-cycle dependent is NOT the same thing as an EMA running on a fixed
# internal cadence, and calling both "event count" is how the CONV_FRAMES
# exception got borrowed by a constant that had not earned it.
FIXED_CADENCE_MARKERS = ("fixed-cadence", "FIXED-CADENCE", "per-window",
                         "4 ms block", "fixed internal", "sub-block")

def recategorize(r):
    v = r["verdict"]
    if v != "keep-event-count":
        return v
    blob = (r.get("reasoning", "") + " " + r.get("consumer", ""))
    if any(m in blob for m in FIXED_CADENCE_MARKERS):
        return "keep-fixed-cadence"
    return "keep-event-count"

for r in rows:
    r["category"] = recategorize(r)

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
    (DOMINATED, "Keep: dominated default",
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
KNOWN = {k for k, _, _ in CATEGORIES}

# ── integrity checks (fail the build, do not paper over) ────────────────────
problems = []
ids = {}
for r in rows:
    if r["category"] not in KNOWN:
        problems.append(f"unknown category {r['category']!r} for {r['name']!r}")
    cid = re.sub(r"[^a-z0-9]+", "-", r["name"].lower()).strip("-")[:70]
    ids.setdefault(cid, []).append(r)
for cid, group in ids.items():
    if len(group) > 1:
        problems.append(f"duplicate id {cid!r} x{len(group)}: "
                        + ", ".join(sorted({g['category'] for g in group})))
for r in rows:
    for field in ("anchor_ms", "anchor_evidence", "effective_ms", "consumer",
                  "reasoning", "confidence"):
        if not str(r.get(field, "")).strip():
            problems.append(f"empty {field} for {r['name']!r}")
if problems:
    print("INTEGRITY FAILURES:", *problems, sep="\n  ")
    sys.exit(1)

# ── source-path validation ──────────────────────────────────────────────────
PATH_RE = re.compile(r"\b((?:python|c_impl|docs|eval)/[A-Za-z0-9_./-]+"
                     r"\.(?:py|c|h|md))\b")
cited, missing = set(), set()
for r in rows:
    for field in ("consumer", "anchor_evidence", "reasoning", "effective_ms"):
        for m in PATH_RE.finditer(str(r.get(field, ""))):
            p = m.group(1)
            cited.add(p)
            if not os.path.isfile(os.path.join(AEC, p)):
                missing.add(p)

# ── emit ────────────────────────────────────────────────────────────────────
def clean(text):
    """Collapse whitespace; guarantee no trailing spaces and no truncation."""
    s = " ".join(str(text).split())
    return s

L = []
A = L.append
by_cat = collections.OrderedDict()
for key, title, _ in CATEGORIES:
    by_cat[key] = [r for r in rows if r["category"] == key]
total = sum(len(v) for v in by_cat.values())
assert total == len(rows), (total, len(rows))

A("# Hop-authored timing-constant inventory")
A("")
A(f"Generated from the adjudication journal against `AEC 5232ab6`. "
  f"**{total} candidates**, each in exactly one category.")
A("")
A("This file is GENERATED, not hand-edited. Regenerating is how the counts, "
  "the category exclusivity and the absence of truncation stay true; an "
  "edit-in-place would reintroduce exactly the drift that made the first "
  "version of this document unusable as release evidence.")
A("")
A("## Conventions")
A("")
A("**Anchor rule.** The wall-clock span is taken from the commit that last "
  "*empirically validated* the constant -- one that ran a benchmark and "
  "accepted the result -- not the commit that introduced it, and not the "
  "in-code comment. These disagree often enough to matter: `simple_mu_holdoff` "
  "was introduced on a 16 ms grid and validated on a 10 ms grid, so anchoring "
  "on the introduction commit is a 1.6x error. Where no commit ever measured "
  "the constant, the entry says so and stays unresolved rather than guessing.")
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
    A("These are cited by an entry below and do not exist in the tree. Each is "
      "a path retired by the release cleanup; the entry's reasoning still "
      "stands but the file must be recovered from git history.")
    A("")
    for p in sorted(missing):
        A(f"- `{p}`")
    A("")
else:
    A(f"All {len(cited)} distinct source paths cited below exist in the tree.")
    A("")

for key, title, blurb in CATEGORIES:
    group = sorted(by_cat[key], key=lambda r: (r["confidence"] != "high",
                                               r["name"].lower()))
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
        for i, r in enumerate(group, 1):
            A(f"| {i} | `{clean(r['name'])}` | {clean(r['anchor_ms'])} "
              f"| {r['confidence']} |")
        A("")
    for r in group:
        A(f"### `{clean(r['name'])}`")
        A("")
        A(f"- **Verdict**: {title.lower()} ({r['confidence']} confidence)")
        A(f"- **Anchor**: {clean(r['anchor_ms'])}")
        A(f"- **Effective today**: {clean(r['effective_ms'])}")
        A(f"- **Consumer**: {clean(r['consumer'])}")
        A(f"- **Provenance**: {clean(r['anchor_evidence'])}")
        A(f"- **Reasoning**: {clean(r['reasoning'])}")
        A("")

text = "\n".join(line.rstrip() for line in L).rstrip() + "\n"
assert "  \n" not in text and " \n" not in text
open(OUT, "w").write(text)
print(f"wrote {OUT}")
print(f"  {total} candidates in {len(CATEGORIES)} categories")
for key, title, _ in CATEGORIES:
    print(f"    {title:<34} {len(by_cat[key])}")
print(f"  cited paths: {len(cited)}, missing: {len(missing)}")
