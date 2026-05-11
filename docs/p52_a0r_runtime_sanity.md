# P52 A.0R.8 — Runtime sanity for trace flag default-OFF

**Date**: 2026-05-11
**Branch**: `feature/p52-phase-a-shadow`
**Comparison**:
- **Pre-Path-3** = commit `eac5325^` (= `8c20f6d`), pre-rename `ShadowCopyController` source
- **Post-Path-3** = commit `09e2ecd` HEAD with `trace_p52_regime_handler = False` (default)

**Sample**: 50 cases sampled with `random.Random(42).sample(stems, 50)` from the
800-case AEC Challenge corpus.
**Per-case run config**: balanced preset / fl = 832 / cng = True /
sequential (j = 1) / per-case `np.random.seed(42)`.
**Warmup**: one short zero-frame `aec.process` call before timing loop.
**Tool**: `tools/research/p52_a0r8_runtime_sanity.py`

## Verdict at a glance

| Bar | Pre-Path-3 vs Post-Path-3 | Result |
|---|---|---|
| Total wall time delta < 1 % | **−0.43 %** (post = 121.13 s, pre = 121.66 s — post slightly **faster**) | **PASS** |
| Per-case `\|Δ%\|` p95 < 2 % | **6.56 %** | **FAIL by spec** |

**Spec disposition**: per A.0R.8 task spec, FAIL → do NOT merge automatically;
investigate, document the responsible code path, await user decision.

**Substantive disposition (after noise characterization)**: the FAIL is
**measurement noise**, not Path 3 overhead. See §3 below. No code path has
been identified that adds runtime cost; total wall time even improves slightly.
User decision required: accept the substantive PASS or hold the strict
literal FAIL.

## Per-case timing summary (n = 50)

| Statistic | Pre-Path-3 (s) | Post-Path-3 (s) | Δ (s) | Δ (%) |
|---|---:|---:|---:|---:|
| Total | 121.66 | 121.13 | **−0.52** | **−0.43** |
| Mean per case | 2.433 | 2.423 | −0.010 | −0.43 |

Per-case Δ% distribution:

```
min  = −6.39 %
p5   = −2.81 %
p25  = −1.47 %
med  = −0.15 %
p75  = +1.22 %
p95  = +6.22 %
p99  = +8.47 %
max  = +10.01 %

|Δ%| p95 = 6.56 %
```

Distribution is **symmetric around zero** (median −0.15 %), consistent with
random measurement noise rather than a systematic overhead.

## Top-5 cases by |Δ%|

| Δ% | Pre (s) | Post (s) | Subset | Case |
|---:|---:|---:|---|---|
| +10.01 | 0.786 | 0.865 | nearend_singletalk | `3tlTJTX8GEmlUQgJJf1y6A_nearend_singletalk` |
|  +6.87 | 0.790 | 0.844 | nearend_singletalk | `NL23aL0w3E6huGfizi8xeg_nearend_singletalk` |
|  +6.70 | 0.810 | 0.864 | nearend_singletalk | `06Q90a0wkkulvuJBJGQqzQ_nearend_singletalk` |
|  −6.39 | 0.998 | 0.935 | nearend_singletalk | `kz23X4pDSEiPmWtw2Qx00Q_nearend_singletalk` |
|  −6.03 | 0.967 | 0.908 | nearend_singletalk | `qkGW9Frbs0Gq5gdfsztA2g_nearend_singletalk` |

**Pattern**: all 5 outliers are `nearend_singletalk` with sub-1 s wall time —
any OS scheduling / page-cache / GC blip is a large fractional share of the
total. There is no `farend_singletalk` or `doubletalk` case in the top 5.

## §3 — Noise floor characterisation

To distinguish Path 3 overhead from measurement noise, the post-Path-3 run
was repeated under identical conditions (same commit, same code, same seed)
and compared against itself:

| Comparison | Total Δ% | Per-case `\|Δ%\|` p95 |
|---|---:|---:|
| Pre-Path-3 vs Post-Path-3 (real) | −0.43 % | 6.56 % |
| Post-Path-3 vs Post-Path-3 (same code, two runs) | +0.67 % | **6.36 %** |
| Bar | < 1 % | < 2 % |

The same-code-twice measurement floor (6.36 % p95) is the intrinsic noise of
this measurement setup. The pre-vs-post 6.56 % is **statistically
indistinguishable** from the noise floor; the 0.20 percentage-point excess is
well within run-to-run variation.

**Conclusion**: Path 3 (rename + default-OFF trace flag + classifier module)
introduces **no measurable runtime overhead**. The strict literal FAIL on
the 2 % p95 bar reflects that the bar is below the intrinsic noise floor of
single-run sequential timing on sub-1 s cases, not a real overhead in the
code path.

## §4 — What would actually identify overhead

If Path 3 added overhead, the expected signature would be:
- Per-case Δ% skewed positive (post slower than pre), not symmetric.
- Total wall time delta > 0 by more than noise margin.
- Concentration on cases with shadow-filter active (FS / DT), not on the
  short NE cases.

None of these are observed:
- Skew is symmetric (median −0.15 %, p5 and p95 nearly mirror at ±6 %).
- Total Δ is **negative** (post faster).
- Outliers cluster on short NE cases (where the shadow filter has the
  *least* relative work) — i.e. exactly where noise dominates signal.

The trace flag (`config.trace_p52_regime_handler = False`) is checked with
`getattr(..., False)` once per frame at one site, then short-circuits; this
is a single attribute read and dictionary append-list-skip per frame. Path 3
also added a one-line `self._regime_trace_rows = []` in `__init__` and a
no-op `dump_regime_trace` that early-returns on empty list. None of these
operations consume measurable CPU when the flag is off.

## §5 — Disposition options for the user

1. **Accept PASS-on-substance** — recognise that the 2 % p95 bar is below
   the noise floor of this measurement setup; Path 3 has no measurable
   overhead; merge can proceed.
2. **Strict literal FAIL** — keep the bar as written; do not merge until a
   tighter measurement methodology (averaged over N runs, or excluding
   sub-1 s cases) reproduces PASS.
3. **Re-spec the bar** — adopt a more realistic per-case bar (e.g. 7 % p95
   given the 6.4 % noise floor) and re-evaluate.

A.0R.8 cannot self-resolve — per task spec anti-loophole "此 task 不修改
任何 Path 3 code" and "If fail, document finding, 不 merge, 等 user decision".

## Cross-references

- Tool: [tools/research/p52_a0r8_runtime_sanity.py](../tools/research/p52_a0r8_runtime_sanity.py)
- CSV artefacts: `/tmp/p52_a0r8/{pre,post,post2}.csv`
- Verdict JSONs: `/tmp/p52_a0r8/{verdict,noise}.json`
- A.0R.7 distribution: [p52_a0r_regime_distribution.md](p52_a0r_regime_distribution.md)
- A.0R.6 byte-equal: see [p52_phase_a_verdict.md](p52_phase_a_verdict.md) §A.0R.6
- Phase A verdict: [p52_phase_a_verdict.md](p52_phase_a_verdict.md)
