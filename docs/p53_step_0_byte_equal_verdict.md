# P53 Step 0 byte-equal sanity verdict

**Date**: 2026-05-12
**Branch under test**: `p53-step-0-audit` HEAD `331c36b`
**Baseline**: `p52-phase-b-closed` (`5ad573a`) — code-equivalent to `main` HEAD `4f1f9a7` (doc-only Phase C closure descendant)
**Config**: production path; `use_res_refactored=False`; `trace_p53_innovation=False` (default)
**Corpus**: 800-case full `wav/aec_challenge_blind/`, `preset=balanced / fl=832 / cng=True / j=4`
**CNG seed**: 42 per case

## Result: PASS

```json
{
  "cases_total": 800,
  "cases_byte_identical": 800,
  "total_samples": 325123520,
  "samples_exact_match": 325123520,
  "samples_close_atol_1e-6_rtol_1e-5": 325123520,
  "fraction_exact": 1.0,
  "fraction_close": 1.0,
  "pass_99_99": true,
  "pass_100_exact": true,
  "cases_over_0.1dB_mean_drift": 0,
  "pass_zero_over_0.1dB": true
}
```

- **800/800 cases byte-identical**
- **325,123,520 / 325,123,520 samples exact match (100 %)**
- Zero cases with > 0.1 dB mean drift
- Internal target 100 % exact: met
- Hard bar (A.0R.6 / B.4 equivalent: ≥ 99.99 % within `atol=1e-6, rtol=1e-5`): met with margin

## Interpretation

The `trace_p53_innovation` flag and its `PBFDKF._enable_p53_trace` /
`AEC.dump_p53_trace` plumbing introduce **zero numerical drift** on the
default-OFF path. The hook is guarded by `if self._enable_p53_trace:` at
[aec.py:1135](../python/aec.py#L1135); the guard's branch-not-taken cost
is non-observable at sample level.

This confirms the Step 0 audit code path satisfies the §6.4 isolation
discipline carried from P52: research instrumentation must be
trace-flag gated and trace-flag default-OFF must be byte-equal to the
pre-instrumentation baseline.

## Snapshots

- `/tmp/p53_be/baseline.npz` — 1,300,746,762 bytes, written 527 s
- `/tmp/p53_be/p53_head.npz` — 1,300,746,762 bytes, written 595 s
- `/tmp/p53_be/diff.json` — diff verdict (this doc cites it verbatim)

## Conclusion

Step 0 audit code path safe to merge into `main`. Proceeding to
Action 2 (merge + tag `p53-step-0-closed-T0E` + push origin).
