# P52 errata

Post-execution clarifications and methodology learning collected during
Phase A Path 3 closure. Entries are documentation only — they do **not**
modify the P52 v1.1 design lock or any hard test bar. Constants remain
locked per anti-loophole rule 5.1.

## A.0R.8 spec design defect — 2 % p95 bar below measurement noise floor

**Symptom**: A.0R.8 task spec set a per-case `|Δ%|` p95 acceptance bar
of < 2 % for trace-flag-default-OFF runtime sanity. Execution produced
6.56 % p95 (literal FAIL by spec) but with total wall time −0.43 %
(post slightly faster than pre) and a Δ% distribution symmetric around
zero (median −0.15 %).

**Root cause** (characterised in
[`p52_a0r_runtime_sanity.md`](p52_a0r_runtime_sanity.md) §3): the 2 %
bar was set without first measuring the intrinsic single-run timing
noise floor. Empirically, post-vs-post2 same-code re-runs under
identical conditions yield p95 = 6.36 %. The bar was placed
**below** the noise floor of the measurement methodology, making a
literal PASS infeasible regardless of whether Path 3 introduced any
overhead.

**Resolution**: A.0R.8 closed with **PASS-on-substance** per user
decision (2026-05-11). Substantive evidence — total runtime faster
post-Path-3, symmetric Δ% distribution, outliers concentrated on
sub-1 s `nearend_singletalk` cases where any OS-level blip is a large
fractional share — supports the conclusion that Path 3 introduces no
measurable runtime overhead. The literal p95 6.56 % vs the same-code
6.36 % floor differs by 0.20 percentage points, well within run-to-run
variation.

**Recommendation for future timing tests**: when authoring a per-case
p95 wall-time bar, either

- (a) average across N independent runs to reduce the effective noise
  floor below the intended bar, OR
- (b) measure the intrinsic same-code noise floor first and set the bar
  relative to that floor with explicit margin, OR
- (c) exclude sub-1 s wall-time cases from the per-case statistic
  (where absolute scheduling jitter dominates relative metrics) and
  state the exclusion in the bar definition, OR
- (d) state the bar against total wall time only (which behaves like an
  average across cases and has a tighter empirical distribution).

**Scope**: P52 v1.1 §2.7 / §3.6 / §4.4 hard test bars are **not
modified**. This errata applies only to A.0R.8, which is a
non-listed-in-design-lock supplementary sanity check authored as part
of Path 3 closure (not in the v1.1 § T1–T8 hard-test list).
Future Phase A / B / C timing tests, if any, may reference this
errata when authoring acceptance bars.

---

## T0 threshold methodology scope limitation (P53 finding)

P52 design lock §2.4 T0 thresholds (`r_voice mean ≥ 3.0`, etc.) implicitly
assumed whole-case mean would detect Q-related signatures. P53 forensic
review ([docs/p53_audit_forensic_review.md](p53_audit_forensic_review.md)
§3) found this scope limitation: transient divergence events are < 1 % of
frames, washed out by quiescent mass.

Future audit-style methodologies should either:

(a) condition on handler-fire-aligned frame subsets, or
(b) use a different statistical scope (max, p99, or event-conditional
    means), avoiding whole-case mean for transient detection.

**P52 spec NOT modified.** This errata documents methodology learning
carried forward to future design locks.
