# S6b verdict — `epc_dt_cap` state-driven gate: legacy gate is dead code

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a`
**Design**: [v3_12_s6b_epc_dt_cap_design.md](v3_12_s6b_epc_dt_cap_design.md) (d28e0a9)
**Status**: **S6b CLOSED, retarget Q7 V3 carrier**

## TL;DR

The S6b 800-case fire-rate audit (Step 2 of design §5.3) found that the
legacy `epc_dt_cap` gate fires **0 / 2,032,022 frames** at production
thresholds. The cap action `g = min(g, 0.85)` is dead code in BALANCED
preset. The Q7 V3 verdict that identified `epc_dt_cap` as the FS leak
carrier is wrong — a dead cap cannot leak. Re-investigation of the
real carrier is required before S8-S9 (Phase 3C).

Flag-ON A/B (design §5.4) **was not executed**. BALANCED preset is
unchanged. `res_state_driven_epc_dt_cap` is retained default-OFF as a
research substrate for any future Phase 3C state-set revision.

## Audit results

Run: `tools/research/s6b_fire_rate_audit.py` (deleted post-verdict),
preset BALANCED, `res_consume_filter_state=True` (forced),
`res_state_driven_epc_dt_cap=False`. Counts what each gate *would* fire
under identical inputs.

| bucket | cases | frames | legacy% | state% | overlap% | state/legacy |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 169 | 387,980 | **0.00%** | 4.94% | 0.00% | inf |
| FS_movement | 131 | 295,122 | **0.00%** | 4.12% | 0.00% | inf |
| DT_static | 186 | 701,736 | **0.00%** | 1.73% | 0.00% | inf |
| DT_movement | 114 | 424,351 | **0.00%** | 2.12% | 0.00% | inf |
| NE | 200 | 222,833 | **0.00%** | 0.46% | 0.00% | inf |
| **TOTAL** | **800** | **2,032,022** | **0.00%** | **2.63%** | **0.00%** | **inf** |

Raw data: [results/v3_12_s6b_audit/fire_rate.json](../results/v3_12_s6b_audit/fire_rate.json)

## Mechanism: why legacy gate is dead

Legacy gate ([aec.py:2710](../python/aec.py#L2710), pre-S6b):
```python
epc_dt = epc_active and effective_dt > 0.35
```

- `epc_active`: latched flag from `EchoPathChangeDetector` ([aec.py:3495](../python/aec.py#L3495))
- `effective_dt = max(dt_for_fs, shadow_dt)` ([aec.py:2705](../python/aec.py#L2705))

The audit's 0.00% across all buckets means there is no frame in the
800-case corpus where **both** conditions hold simultaneously.
Plausible explanations (not investigated for this verdict; flagged
for future Phase 3C work):

1. **`epc_active` is rarer than design assumed**. EPC detection in
   BALANCED requires either delay-shift events or shadow_rise streaks
   ≥3 frames — relatively conservative.
2. **`effective_dt > 0.35` is rarer in EPC frames**. When EPC fires
   the filter regime is typically `transient` or `recovering`, where
   `dt_for_fs` may already be reset / dampened (F2.3 R-reset path).
3. **Both are required**. With each individual gate at ~few-percent
   fire-rate, intersection drops to numerical 0% on this corpus size.

The legacy cap is therefore **not** load-bearing on the 800-case bench
under BALANCED. Removing the cap entirely (`epc_dt = False`) would be
byte-equal to the current state.

## What this overturns

Design doc §1-2 framed `epc_dt_cap` as the actual Q7 V3 leak carrier
because:
- It has no `(1 - fs_confidence)` neutraliser (unlike `ne_g_floor`).
- The scalar `effective_dt` ingests `(1 - coh²)` evidence that can
  spike in FS post-cancellation.

Both of those are still **true at the level of mechanism**, but they
are **moot** because the gate's intersection condition (`AND`) never
holds in practice. The cap is gated to a frame regime that does not
exist in the corpus.

The S6 verdict (ne_g_floor swap byte-equal — neutralised by
`(1-fs_confidence)`) + S6b verdict (epc_dt_cap dead code) together
**falsify the Q7 V3 attribution** to gain-postprocess caps as the
canonical-coherence breakers. The real carrier must be sought elsewhere.

## What the new (state-driven) gate would do

If `res_state_driven_epc_dt_cap=True` were flipped on, the gate would
fire 2.63% of all frames globally, with bucket distribution per the
table. This is **additive** to current behaviour (not retargeting an
active cap) — we would be activating a previously-dormant cap with a
new gate condition. Risk profile changes substantially:

- **FS (4.94% / 4.12%)**: probable net-positive. `diverged` and
  `suspicious_dt` states in FS indicate filter unreliability →
  capping gain at 0.85 reduces residual echo.
- **DT (1.73% / 2.12%)**: risk zone. 2% of DT frames would have
  gain force-capped at 0.85, potentially over-suppressing NE speech
  when the state classifier mis-fires.
- **NE (0.46%)**: low risk by fire-rate, but per design doc §3.4
  `startup` / `coarse_learning` were intentionally excluded — the
  0.46% comes from `diverged` / `suspicious_dt` firing in NE, which
  would be a state-classifier bug if anything.

Flag-ON A/B is **not** proceeding because:
1. The S6b design framed this as a retarget, not an addition.
2. Activating dead-code caps is closer in spirit to P58 (added new
   cap structure → FS Δecho −0.674) than to Phase 3B (retarget broken
   evidence).
3. The Q7 V3 carrier remains unidentified — adding more cap-side
   intervention before locating the real evidence leak risks adding
   yet another patch on broken evidence (Q7 V3 verdict §1.5).

## Anti-trap reflection

| Trap | Original mitigation | New reading |
|---|---|---|
| **P50** (FS Δecho −1.328) | "New gate is state-membership, removes `effective_dt` from gate, FS frames no longer fire" | **Moot** — legacy gate already never fires in FS, so P50 path-replication concern is unfounded |
| **P58** (FS Δecho −0.674) | "Cap action preserved 0.85, gate only changes" | **Inverted risk** — we would be *activating* a dormant cap with new gate, closer to P58's *adding new cap behaviour* than to gate-swap |

Conclusion: design doc anti-trap audit was written under the
assumption that the gate change is retargeting an active cap. With
the audit finding that the cap is dead, the anti-trap framing flips —
we'd be in an *additive* regime, which is the P58 failure mode.

## Action taken

1. **Counters scrubbed**: `_epc_dt_legacy_count` / `_state_count` /
   `_overlap_count` / `_total_frames` removed from `ResFilter`
   (per design doc §7 question 3).
2. **Audit script deleted**: `tools/research/s6b_fire_rate_audit.py`
   removed (depended on scrubbed counters; raw data preserved in
   `results/v3_12_s6b_audit/fire_rate.json`).
3. **Flag retained default-OFF**: `res_state_driven_epc_dt_cap` stays
   in `AecConfig` as a dormant research substrate, consistent with
   other Phase-1 verdict-FAIL flags (F1.2 `streak_only`, F2.2 EMA).
4. **Gate code retained default-OFF**: `if ... res_state_driven_epc_dt_cap`
   branch in `_stage_gain_postprocess` simplified to direct membership
   check; legacy branch is the default.
5. **BALANCED preset unchanged**: no flag flip, no preset modification.
6. **Byte-equal verification (single-case, cohort tail `qNvSMyU`)**:
   pre-scrub MD5 == post-scrub MD5 (sample-level identical, flag OFF
   takes legacy branch). Pre-existing 3e-5 max-diff vs
   `v3_12_s3_candidate` baseline is from S6 commit a60768f
   (documented in [v3_12_res_audit.md](v3_12_res_audit.md) Addendum),
   well below AECMOS noise floor.

## Next step: Q7 V3 re-investigation

The Q7 V3 verdict identified 5 gain-floor / cap paths as evidence
patches that fragment the canonical RES gain. Two are now eliminated
as the FS leak carrier:

| Path | Status after S6/S6b |
|---|---|
| `ne_g_floor` | Eliminated (S6 byte-equal; `(1-fs_confidence)` neutraliser already gates FS) |
| `epc_dt_cap` | Eliminated (S6b: legacy gate is dead code, fires 0%) |
| `spectral_floor` | **Candidate** — never investigated in detail |
| `quiet_mask` | **Candidate** — never investigated in detail |
| `divergence_floor` | **Candidate** — never investigated in detail |
| 3-bin smooth (kernel [0.1, 0.8, 0.1]) | **Candidate** — Q7 V3 §7.1 flagged sidelobe-leakage |
| `dt_per_bin` ENR usage in `_stage_gain_compute` | **Candidate** — Q7 V3 §7.2 flagged scalar/per-bin mismatch |

Recommended Phase 3B v3 sequencing (deferred to user-authorized sprint):

1. **Fire-rate audit of all 4 remaining caps** under BALANCED on 800
   cases. Identify which caps are dead code, which are infrequent,
   which fire on every frame. Same pattern as S6b counters but
   targeting full 4-cap chain.
2. **Identify the high-fire-rate cap in FS**. That is the candidate
   for the Q7 V3 evidence breaker. Likely candidates by frame coverage:
   - `spectral_floor` (frame-global, every frame): physical fallback
     but may have evidence-driven internal gate
   - `quiet_mask` (gated): runs only when noise floor estimate is low
3. **Or alternatively retarget to `dt_per_bin` ENR**: Q7 V3 §7.2
   flagged the scalar-vs-per-bin mixing as a key fragmentation point.
   `dt_per_bin` is consumed by `_stage_gain_compute` ENR weighting
   *before* the 4-cap chain. If `dt_per_bin` is the leak source, no
   amount of cap-side intervention will fix it.

## Open questions for next user-authorized sprint

1. **Reproducibility infrastructure**: the audit script was a one-shot
   diagnostic with diagnostic counters in `aec.py`. For systematic
   re-audit (e.g. of the other 4 caps), should we build a more durable
   counter infrastructure (e.g. AecStats fields) or keep one-shot
   counter substrates?
2. **Define "dead code" threshold**: a cap firing 0 / 2M frames is
   clearly dead. But where do we draw the line for "rare enough to
   retire"? <0.1% global? <0.01%?
3. **Should `epc_dt_cap` legacy gate be retired** (not just bypassed)?
   The audit shows it never fires; the code is dead. Retiring it would
   remove the substrate for any future re-enable. Recommend keeping it
   for now (substrate cheap, removal has no productivity benefit).
4. **Should `res_state_driven_epc_dt_cap` flag itself be retired**?
   Same trade-off — substrate is cheap to retain but adds cruft. Keep
   for now; revisit if Phase 3C lands a different state set or
   alternative gate target.

---

## Appendix — Why this is "find the bug" not "design failure"

The S6b design was internally consistent: it correctly identified
that the `effective_dt` saturation problem (Q7 V3 §3.1) lacks a
downstream neutraliser at `epc_dt_cap` (unlike `ne_g_floor` which has
`(1-fs_confidence)`). The fix proposed (state-driven gate) would
correctly mechanism-replace the saturating scalar with a robust
membership check.

What the design missed was an empirical check on **whether the gate
ever fires in production**. The byte-equal scaffold (design §5.2) was
trivially passing — the design noted "this is sanity that the code-path
branch is inert when flag is OFF" — but the fire-rate audit (§5.3) was
intended to characterise the new gate's coverage, not to verify the
legacy gate was alive. With hindsight, the audit's first finding should
have been to **confirm the legacy gate fires at all** before measuring
the new gate's calibration. That step is now standard for any future
gate-swap design.

This was a design-validation gap, not a mechanism error. The S6 + S6b
joint outcome (`ne_g_floor` neutralised; `epc_dt_cap` dead) is a
stronger negative result than either taken alone — they jointly
falsify the Q7 V3 attribution of FS echo leak to *post-process gain
caps* and redirect the investigation to *pre-cap evidence paths*
(specifically `dt_per_bin` ENR usage in `_stage_gain_compute`, or
the 3 remaining caps yet to be audited).
