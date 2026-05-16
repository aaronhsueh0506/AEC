# Phase 3B v3 design — Q7 V3 carrier re-investigation after S6/S6b joint negative

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a` (HEAD = 43cb886 = S6b verdict)
**Predecessors**:
- [v3_12_res_audit.md](v3_12_res_audit.md) — Phase 3A static 5-path audit
- [v3_12_phase3b_v1 = S6 commit a60768f] — `ne_g_floor` swap byte-equal
- [v3_12_s6b_epc_dt_cap_design.md](v3_12_s6b_epc_dt_cap_design.md) — S6b design lock
- [v3_12_s6b_verdict.md](v3_12_s6b_verdict.md) — S6b legacy gate is dead code

**Status**: **DESIGN-ONLY**. Implementation gated on user review.

---

## 1. Why we are here

Phase 3A static audit identified 5 gain-floor / cap paths in RES. Phase 3B v1
targeted `ne_g_floor`; Phase 3B v2 (S6b) targeted `epc_dt_cap`. Both came
back with negative outcomes on the same root mechanism:

| Phase | Target | Outcome | Reason |
|---|---|---|---|
| S6 (v1) | `ne_g_floor` swap to mic-excess evidence | 800-case **byte-equal** vs v3.11.2 | `(1-fs_confidence)` already neutralises ne_g_floor in FS; evidence swap has no FS-visible effect |
| S6b (v2) | `epc_dt_cap` gate swap to `filter_state` | Legacy gate fires **0 / 2.03 M frames** | Cap is dead code in BALANCED; gate cannot leak echo if it never fires |

**Joint implication**: the Q7 V3 attribution of the FS echo leak to
*post-process gain caps* (Paths 2-3-5 of the Phase 3A audit) is
empirically falsified. The remaining 2 paths in Phase 3A are physical
fallbacks (Paths 1 `spectral_floor` + 4 `quiet_mask`) — not evidence
patches and not the carrier of any FS leak.

The Q7 V3 critique was correct at the **mechanism** level (RES has 8
patch generations layered on broken evidence) but mis-located the
**dominant FS-visible carrier** within RES. This document re-locates
the carrier and proposes a new sprint sequence.

---

## 2. Mechanism findings from Explore trace (2026-05-13)

### 2.1 Stage 1 vs Stage 3 cap terminology clarification

P58 verdict states "retire 4-cap chain → FS Δecho -0.674". This refers
to **Stage 1 residual attribution caps** in `_stage_residual_model`
([aec.py:2074-2103](../python/aec.py#L2074-L2103)), NOT the
gain-postprocess caps in `_stage_gain_postprocess`. The 4 are:

1. `residual ≤ echo × 2.0` (physical: residual can't exceed 2× filter estimate)
2. `residual ≤ error × 1.5` render / `× 1.0` else (error-based ceiling)
3. `residual ≤ error × dt_suppress` where `dt_suppress = clip(1-dt², 0.1, 1.0)` (DT-weakening)
4. `residual ≤ far × min(2×erl_estimate, 1.0)` (ERL-anchored ceiling)

These are pre-canonical caps on the residual echo PSD estimate, applied
BEFORE Wiener / ENR gain compute. They are load-bearing on FS per P58
post-mortem. **Phase 3B v3 must preserve these unchanged** (anti-P58
trap, same as Phase 3B v1 / v2).

The 5 gain-postprocess paths (Phase 3A audit) are downstream of Stage 1
and operate on the gain `g` directly. Phase 3B v3's carrier search
focuses here.

### 2.2 The actual FS leak candidates after S6/S6b elimination

| Path | File:line | Phase 3A verdict | Post-S6/S6b status |
|---|---|---|---|
| `spectral_floor` | [aec.py:2733-2744](../python/aec.py#L2733-L2744) | Physical fallback (keep) | Confirmed not the FS carrier |
| `ne_g_floor` | [aec.py:2755-2759](../python/aec.py#L2755-L2759) | Evidence patch (merge) | **Eliminated by S6** — `(1-fs_confidence)` neutraliser |
| `epc_dt_cap` | [aec.py:2710](../python/aec.py#L2710) | Evidence patch (lift) | **Eliminated by S6b** — legacy gate fires 0% |
| `quiet_mask` | [aec.py:2676-2678](../python/aec.py#L2676-L2678) | Physical fallback (keep) | Confirmed not the FS carrier |
| `divergence_floor` | [aec.py:2433-2436](../python/aec.py#L2433-L2436) | Hybrid (generalise) | **Candidate** — frame-scalar, unconditional, no FS neutraliser |

And two paths flagged in Q7 V3 §7 but NOT enumerated in Phase 3A audit:

| Path | File:line | Mechanism | Status |
|---|---|---|---|
| **3-bin smooth** kernel `[0.1, 0.8, 0.1]` | [aec.py:2348-2365](../python/aec.py#L2348-L2365) | Per-bin convolve gain along freq axis; v3.8.4 tightened kernel from `[0.25, 0.5, 0.25]` | **Top candidate** — code comment at [aec.py:2377-2383](../python/aec.py#L2377-L2383) explicitly says "Plan A's actual FS cost lives in the smoothing kernel change, not [HF cap]" |
| **`dt_per_bin` ENR usage in `_stage_gain_compute`** | [aec.py:2201-2219](../python/aec.py#L2201-L2219), consumed at [2236-2237](../python/aec.py#L2236-L2237) | Per-bin DT indicator from `(1-coh²)` shapes nearend NE estimate via `dt_shaped = dt_per_bin ** 1.1`; in FS post-cancellation `coh² → 0` so `dt_per_bin → 1` → NE estimate over-shapes in FS, raising `g` (leakage) | **Top candidate** — upstream of all caps; if dt_per_bin saturates in FS, downstream Wiener gain has no chance |

### 2.3 What the code comments tell us

The 2026-05-13 explore trace surfaced two key comments:

**Comment 1** ([aec.py:2377-2383](../python/aec.py#L2377-L2383)):
> NOTE (v3.10.5 investigation): cap is largely dead code in FS post-
> cancellation. ... Plan A's actual FS cost lives in the smoothing
> kernel change, not here.

This is an explicit code-level confession that the hf_cap path (Path D
in the Explore report) does not carry the FS Δecho cost — the **3-bin
smoothing kernel** does. v3.8.4 tightened it precisely because the
v3.8.3 wider kernel was leaking gain from low-NE bins into high-NE
neighbours via sidelobe convolution. The fix is in place but the
mechanism (per-bin gain bleed via convolution) is still the dominant
FS sensitivity surface.

**Comment 2** (Q7 V3 §3.1, already in the Phase 1 plan):
> dt_per_bin = max(effective_dt, 1-coh²) saturates ~1 in FS post-
> cancellation (echo cancelled → low coh² → "NE-like"), so the
> high_ne_conf < 0.3 gate rarely fires in FS — cap is largely dead
> code there.

This locates the saturation at `dt_per_bin` (in `_stage_gain_compute`
where the ENR uses it for per-bin shaping). The downstream consumers
of `dt_per_bin` include both the hf_cap gate (dead per Comment 1) AND
the nearend estimate shaping (`dt_shaped = dt_per_bin ** 1.1` at
line 2236-2237). The shaping path is what makes the per-bin NE
estimate over-trigger in FS, and it is upstream of all caps / floors.

### 2.4 Hypothesised carrier hierarchy

Synthesis: the FS echo leak in v3.11.2 (and any v3 ancestor) flows
through TWO parallel mechanisms, both upstream of the gain-postprocess
caps that Phase 3B v1/v2 targeted:

```
                              FS post-cancellation
                                       │
                                       ▼
                         coh² → 0    (true: echo cancelled, no NE)
                                       │
                            ┌──────────┴──────────┐
                            ▼                      ▼
                  dt_per_bin → 1            (1 - coh²) → 1
                  (line 2201-2219)          (legacy NE evidence)
                            │                      │
                            ▼                      │ (neutralised by
              dt_shaped = dt_per_bin^1.1            │  `(1-fs_confidence)`
              shapes nearend_est                    │   in ne_g_floor —
              upward in FS (line 2236-2237)         │   S6 confirmed)
                            │                      ▼
                            ▼               ne_g_floor (S6 byte-equal)
              ENR / Wiener gain inflates
              (no `(1-fs_confidence)`
               neutraliser here)
                            │
                            ▼
                  g[k] elevated in FS
                            │
                            ▼
                  3-bin smooth [0.1, 0.8, 0.1]
                  bleeds elevated g into
                  neighbour bins (small but
                  persistent across FS)
                            │
                            ▼
                  output: residual echo leakage
```

This explains why:
- S6 was byte-equal: the right-hand path (`ne_g_floor`) is already
  zeroed in FS.
- S6b's legacy gate is dead: `effective_dt > 0.35` AND `epc_active`
  doesn't co-occur in production because by the time effective_dt
  is high, EPC has already de-latched.
- v3.8.4 had to tighten the smooth kernel: it was the visible leak,
  but the upstream cause (`dt_per_bin` saturation) was never fixed.
- F3.1 v3 mic-excess (in v3.10.6 production via
  `use_mic_excess_evidence=True`) helps but doesn't fully fix:
  it replaces evidence for `dt_per_bin` per-bin construction (line
  2188-2205) but only when the gate `filter_converged AND
  long_window_ready AND not epc_active` is satisfied. In the
  legacy path (line 2212-2215), `dt_per_bin = max(effective_dt,
  1-coh²)` still feeds the per-bin NE shape, so the saturation is
  only partially mitigated.

---

## 3. Proposed Phase 3B v3 work

Three candidate sprint targets, ordered by mechanism centrality:

### 3.1 Option α — fix `dt_per_bin` legacy-path saturation (upstream evidence)

**Mechanism change** ([aec.py:2212-2215](../python/aec.py#L2212-L2215)):

```python
# OLD legacy path (when F3.1 v3 gate fails)
dt_per_bin = np.maximum(effective_dt, 1.0 - coh2)
```

The fallback uses `(1-coh²)` directly, so when `coh² → 0` in FS
post-cancellation, `dt_per_bin → 1`. Proposal: extend the F3.1 v3
mic-excess blend to the fallback path as well, gated on a relaxed
condition (e.g. just `filter_converged AND _long_window_n_updates > 0`,
drop the `NOT epc_active` constraint for the fallback specifically).

**Why this is upstream of all caps**: `dt_per_bin` shapes
`nearend_est` at line 2236-2237 via `dt_shaped = dt_per_bin ** 1.1`.
A NE estimate inflated in FS will propagate through Wiener / softgate /
ENR before any cap or floor sees the signal.

**Flag**: `res_dt_per_bin_unified` (default OFF).

**Risk**: MED. The legacy path is the FS-bypass for F3.1 v3 when EPC
is active. Removing the EPC gate may regress EPC-recovery cases.
Anti-trap: gate the change on `epc_streak_count` so post-EPC frames
have the legacy `(1-coh²)` path for N frames (e.g. 30) before unified
takes over.

**ROI**: HIGH. Mechanism-critical; addresses the root saturation point
that all S6/S6b downstream verdicts confirm is upstream.

### 3.2 Option β — re-design 3-bin smooth (sidelobe leakage)

**Mechanism change** ([aec.py:2348-2365](../python/aec.py#L2348-L2365)):

Current kernel `[0.1, 0.8, 0.1]` has 10 % sidelobe per side. v3.8.4
already tightened from `[0.25, 0.5, 0.25]` (25 % sidelobes regressed by
10 dB HF cut on case 7GTxyT). Two sub-options:

- **β.1** — replace symmetric convolve with **per-bin spectral median**
  (3-tap median across frequency). Median is non-linear, no sidelobe
  leakage; preserves edges. Risk: median is computationally heavier
  but per-frame cost is trivial (~33 ops × 257 bins = 8.5K ops).
- **β.2** — replace symmetric convolve with **gain-aware asymmetric
  smooth**: smooth only when neighbouring bins agree in direction
  (both above or both below 0.5). Stop smoothing across sign edges.
  Preserves spectral shape transitions.

**Flag**: `res_gain_smooth_v2` (default OFF).

**Risk**: LOW-MED. Smoothing changes are byte-disrupting but
empirically validated through 800-case bench. v3.8.3 → v3.8.4 already
established that smoother changes work.

**ROI**: MED. Directly addresses the code-confessed "actual FS cost"
per Comment 1 at line 2377-2383. But fixes the symptom; if upstream
`dt_per_bin` saturation is fixed (Option α) the smoothing leakage
may auto-resolve.

### 3.3 Option γ — `divergence_floor` generalisation to `filter_state`

**Mechanism change** ([aec.py:2433-2436](../python/aec.py#L2433-L2436)):

```python
# OLD
if divergence > 0.3:
    divergence_gain = 0.01 + (1.0 - 0.01) × (1.0 - divergence)
    g = min(g, divergence_gain)
```

Phase 3A Path 5 verdict: this is the hybrid cap that already consumes
filter-state-like signal (`divergence` scalar). Generalisation: replace
with `filter_state` lookup table from S3 wiring.

**Per-state divergence cap proposal**:
| state | divergence_cap |
|---|---|
| `idle` | 1.0 (no cap) |
| `startup` | 0.5 (conservative) |
| `diverged` | 0.1 (heavy cap, matches current divergence > 0.7 region) |
| `suspicious_dt` | 0.3 (light cap) |
| `refined_usable` | 1.0 (no cap) |
| `coarse_learning` | 0.5 |

**Flag**: `res_state_driven_divergence_floor` (default OFF).

**Risk**: MED. Cohort tail cases (qNvSMyU etc.) rely on the divergence
> 0.3 cap firing during the transient. Per-state tuple must preserve
this. Anti-trap: byte-equal scaffold (always-return-current-cap
mapping) → tuple swap → A/B.

**ROI**: LOW-MED. Cohort tail is already protected by current code;
this is more of a "tidying" sprint than a leak-fix.

---

## 4. Sprint sequencing

Recommended order (highest mechanism centrality first):

1. **S7 (re-numbered, was S6 in original plan)** — Option α
   `dt_per_bin` legacy-path unification. Upstream of all caps,
   addresses confirmed saturation point. Single flag, ~10 lines.
2. **S8 (was S6b)** — Option β `gain_smooth_v2` (only if α doesn't
   close the FS gap). Adds median or asymmetric smoother. Could be
   merged with S7 in a single 800-case A/B if α alone underdelivers.
3. **S9 (was S8-S9 in original plan)** — Option γ `divergence_floor`
   per-state tuple + filter_state generalisation. Tidying sprint;
   prepares foundation for full per-state ENR tuple work.
4. **S10** — original S8-S9: 4-cap ranked priority + per-state ENR
   tuple full implementation. Builds on S7-S9 substrate.
5. **S11** — v3.12.0 release commit + tag.

**Total**: 5 sprints (was 8 in original plan; deleted S6/S6b which
are now closed as negatives).

---

## 5. Anti-trap audit

| Trap | Mechanism | How v3 design avoids it |
|---|---|---|
| **P50** (FS Δecho −1.328 from `nearend_protect_dt`) | Single scalar `effective_dt` gate fires in FS-pretending-DT | Option α removes `effective_dt` from legacy `dt_per_bin` construction in FS frames. Per-bin construction is the OPPOSITE of P50's single scalar gate. |
| **P52** (cohort tail −0.56 from PathChangeRegimeHandler retirement) | `qNvSMyU` lost catastrophe defence | PathChangeRegimeHandler **untouched**. Option α touches `dt_per_bin` construction only; γ touches divergence_floor only. Handler unchanged. |
| **P55** (DT-FS only +7.01 vs 20 dB bar) | New Wiener discriminator weaker than legacy ENR | Wiener / softgate / ENR computation **unchanged**. Option α changes input evidence to existing ENR path; β changes post-smooth kernel; γ changes cap policy. No new discriminator. |
| **P58** (FS Δecho −0.674 from 4-cap retire) | Stage 1 residual caps removed → no minimum suppression | Stage 1 residual caps (lines 2074-2103) **preserved unchanged** in all 3 options. v3 design targets *gain-postprocess* and *evidence-input*, never Stage 1. |
| **S6 false-positive** (ne_g_floor swap byte-equal) | `(1-fs_confidence)` neutraliser pre-empts evidence change | Option α is upstream of `ne_g_floor`, before any FS-gating multiplier. Test bypass via post-cancellation FS frames where `effective_dt` is moderate (0.2-0.4) — should produce visible per-bin `dt_per_bin` reduction. |
| **S6b dead-code surprise** | Legacy gate AND-condition never met in production | Pre-implementation fire-rate audit MANDATORY for any new gate. v3 design audit step (§6.1) requires verifying every proposed gate fires at meaningful rate before flag-flip. |

---

## 6. Verification framework

### 6.1 Pre-implementation fire-rate audit (NEW requirement)

Before any flag-flip, instrument the proposed gate / mechanism with a
counter. Run 800-case audit. **Require firing > 0.5 % in target
bucket** before proceeding to flag-ON A/B. This was the gap that S6b
exposed (legacy gate at 0 %).

For Option α: counter on per-bin `dt_per_bin` after blend, measure
mean reduction vs legacy across FS frames where `coh² < 0.1`. Target:
≥ 20 % reduction (i.e. dt_per_bin shifts from 0.9 to 0.7 in FS).

For Option β: counter on per-bin gain bleed (`|g_smoothed[k] -
g_raw[k]|` per bin). Target: median reduction with new smoother vs
legacy of ≥ 10 % on FS post-cancellation frames.

For Option γ: fire-rate of `divergence > 0.3` vs `filter_state ==
'diverged'` (already partially done via Phase 3A trace; refresh with
800-case audit).

### 6.2 Flag-OFF byte-equal scaffold

Same as S3 / S6 / S6b: any new flag must produce 99.99 % byte-equal
output on 800-case when default OFF.

### 6.3 Flag-ON 800-case AECMOS A/B

Hard abort thresholds (same as prior sprints, unchanged):

- FS_static Δecho < −0.02
- FS_movement Δecho < −0.02
- DT_static / DT_movement Δdeg < −0.005
- NE Δdeg < −0.005
- Cohort tail `qNvSMyU` Δecho < −0.05

Positive criteria (any one to qualify as net improvement):

- FS_static or FS_movement Δecho ≥ +0.02 (echo leakage relief, larger
  bar than prior sprints because we are targeting the *real* carrier
  not patches)
- DT bucket Δdeg ≥ +0.005

### 6.4 Listen check

- `xrtntuju` 5-clip DT regression (NE preservation)
- `qNvSMyU` cohort tail FS (catastrophe defence not weakened)
- Add **post-cancellation FS** listen for Option α specifically — any
  case in FS bucket where current output has audible "decorrelated
  residue" character (low-pass coloured hiss). Hypothesise α reduces
  this; verify subjectively.

---

## 7. Critical invariants (don't touch list)

The S6 + S6b joint outcome strengthens our confidence that the following
paths are **physical / load-bearing / well-calibrated**:

- **Stage 1 residual caps** (lines 2074-2103) — P58-confirmed
  load-bearing on FS. **NEVER retire**.
- **PathChangeRegimeHandler** (aec.py:3866-3952 6-gate AND) — P52
  cohort tail defence. **Untouched** by Phase 3B v3.
- **`spectral_floor`** (Path 1, lines 2733-2744) — physical
  shape-preservation. **Keep as final clamp**.
- **`quiet_mask`** (Path 4, lines 2676-2678) — physical noise-floor
  preservation. **Keep as last-resort override**.
- **`fs_confidence` neutraliser** in `ne_g_floor` — S6-confirmed
  load-bearing for FS NE protection elsewhere; the fact that it
  zeroes `ne_g_floor` in FS is a *feature* not a bug.
- **3-bin smooth tight kernel `[0.1, 0.8, 0.1]`** — v3.8.4-validated.
  Option β can propose a *replacement* smoother but must preserve
  the spectral-shape constraint that v3.8.4 established.

---

## 8. Open questions for review

1. **Option α vs β ordering**: α is more upstream and likely to
   subsume β's symptom. But if α requires more design iteration,
   β can ship as a quick FS Δecho fix in parallel. Run α first, or
   parallel?
2. **Counter substrate durability**: S6b's diagnostic counters were
   one-shot in aec.py. For systematic re-audits (this sprint + future
   Phase 3C), should we build a durable counter infrastructure in
   `AecStats` rather than per-sprint additions?
3. **Cohort tail risk band**: Option γ replaces a working
   divergence > 0.3 cap with a state-lookup. Cohort tail (`qNvSMyU`)
   relies on this cap firing — should γ go BEFORE α (so we have a
   strong state-driven divergence cap before touching evidence) or
   AFTER (so evidence is clean before retargeting policy)?
4. **Sprint S7 vs S6c naming**: prior plan had S6b followed by
   S8-S9. With S6 and S6b both closed, renumber to S6c (Option α),
   S6d (Option β), S6e (Option γ) to keep S6-family branding? Or
   skip to S7 to indicate "Phase 3B v3 is its own iteration"?
5. **Listen subjective bar**: §6.4 proposes adding "post-cancellation
   FS decorrelated residue" as a new listen criterion. Should this
   be formalised or kept ad-hoc?

---

## 9. Risk summary

| Aspect | Assessment |
|---|---|
| LOE | 5 sprints (Option α ~1 sprint, β ~1 sprint, γ ~1 sprint, S10 ranked priority ~2 sprints) |
| Architectural risk | LOW. Same surgical-flag pattern as B5 / B6 / S3 / S6 / S6b. No new structural paths. |
| P50/P52/P55/P58 trap risk | LOW per §5 audit |
| ROI uncertainty | MED for Option α (mechanism-critical, but bench effect TBD). LOW for β / γ. |
| Rollback cost | LOW. Each option flag-gated default-OFF. |
| Pre-audit cost | LOW. §6.1 audit step is ~1 day per option (single 800-case run with counter). |

---

## 10. Decision points for user

Before authorising implementation, please confirm:

1. **Do we want to proceed with Phase 3B v3 at all?** Two prior
   iterations (S6 / S6b) both closed as either byte-equal or
   dead-code. It is reasonable to question whether further
   gain-postprocess restructuring is the highest ROI sprint.
   Alternative: skip Phase 3B entirely, jump to Phase 4 (NLP arc /
   F-HFR / F-Harm) or close v3.12 with v3.11.2 as the production
   floor + Phase 3A audit as the documentation deliverable.
2. **If proceeding**, which option first? α (upstream evidence,
   recommended) / β (smoother kernel) / γ (divergence policy).
3. **Counter substrate decision**: per-sprint ad-hoc (current
   pattern) or durable `AecStats` extension (one-time investment).
4. **Sprint numbering**: S7+ (clean break) or S6c+ (S6-family
   continuation).

This document does NOT propose code changes pending review of the
above decision points.
