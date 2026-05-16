# v3.18 Cycle Closeout — WebRTC AEC3 alignment cycle (2026-05-16)

**Branch**: `feature/v3.18-aec3-fetch`
**Algorithm changes shipped**: **0** (per §0.4 negative-result protocol)
**Default-OFF substrate shipped**: **8 distinct arcs** ready for v3.19+
**Last commit before closeout**: `6200ec4`
**Awaiting**: §0.7 user merge authorisation

This pack closes the v3.18 cycle initiated 2026-05-15 per user
directive 「我要你檢查 webrtc, 如果他有做我們就做」. The cycle attempted
to import AEC3 mechanisms wholesale; it learned that **AEC3 mechanisms
depend on AEC3 co-design context that our pipeline lacks**. The
co-design infrastructure (Phase C audit-only ports) shipped as
substrate; the mechanisms shipped as substrate; the cycle's user-facing
algorithm change count remains 0, mirroring v3.17.

## 1. Cycle ledger

| Phase | Sprints | Verdict | Hard bar miss | Doc |
|---|---:|---|---|---|
| 0 (fetch + audit) | 6 | COMPLETE | n/a | v3_18_phase0_decision.md |
| D-γ (per-band NE + dominant detector) | ~12 | CLOSED | all 4 configs fail bucket guards | v3_18_d_gamma_closeout.md |
| F (EchoPathVariability classification) | 3 | CLOSED | FS_static Δecho -0.048; sx6mxKBQ -0.081 | v3_18_f_closeout.md |
| A (shadow NLMS swap) | 8 | CLOSED | qNvSMyU Δecho -0.001; 4 worst-case breaches | v3_18_a_closeout.md |
| B (FilterMisadjustment + ScaleFilter) | 4 | CLOSED | 0/61 fires on B.4 pre-bench | v3_18_b_closeout.md |
| **C.A** (FilterAnalyzer audit-only) | 5 | **PASS** | n/a | v3_18_c_a_verdict.md |
| **C.B** (FilteringQualityAnalyzer audit-only) | 5 | **PASS** | n/a | v3_18_c_b_verdict.md |
| **C.C** (AecState extension audit-only) | 3 | **PASS** | n/a | (in plan revision §1) |
| C.D-α (leakage_diverged Q-bifurcation) | 4 | CLOSED | PRIMARY DT_static Δdeg +0.000 (need +0.010) | v3_18_c_d_alpha_closeout.md |
| C.E (RES filter_converged → fq_usable) | 3 | CLOSED | FS_static Δecho -0.151 (15× over); 18 worst-case breaches | v3_18_c_e_closeout.md |

Total: **~53 sprints**, 3 PASSes (audit-only), 7 CANNOT SHIP closures.

## 2. The single most important finding: C.E Pareto

C.E is the largest AECMOS movement in v3.18 — and the **only**
behaviour change that produced a clear signal rather than silence.

| Metric | Magnitude |
|---|---|
| Single-case Δdeg max (sUQrHEPA, DT_static) | **+1.200** |
| Single-case Δdeg ≥ +0.1 | 10 cases |
| Single-case Δecho min (N2rQLbnp, FS_static) | -0.490 |
| Single-case Δecho < -0.05 (worst-case breaches) | 18 cases |
| Bucket DT_static Δdeg | +0.214 (21× PRIMARY bar) |
| Bucket FS_static Δecho | -0.151 (15× over guard) |

Interpretation: RES branches gated on `_filter_converged` (5% coverage)
were doing aggressive echo suppression unconditionally. When RES gets
`fq_usable` (52-86% coverage), those branches activate frequently →
RES relaxes echo suppression in confident-filter regions → big DT NE
preserve win, big FS echo regression.

The trade-off is the v3.13 E5 / Phase A Pareto wall, but at a previously-
unseen magnitude. The mechanism is real; the engineering challenge is
walking the Pareto frontier (selective per-RES-branch migration —
out of v3.18 scope).

## 3. Substrate inventory (8 arcs, all default-OFF)

| Arc | Files added/modified | Hooks for v3.19+ |
|---|---|---|
| D-γ subband + dominant NE | aec.py (residual_estimator gates) | Phase D-γ-prime retry |
| F event classification | aec.py (`classify_epc_event` / `AecEvent`) | v3.20 matched-filter delay |
| A shadow NLMS | aec.py + AecConfig 2 fields + PBFDAF construction branch | C.E-ordered retry |
| B FilterMisadjustment + ScaleFilter | aec.py + AecConfig 9 fields + `PBFDAF.scale_filter` / `PBFDKF.scale_filter` | Phase C.B-derived fq_usable hookup |
| C.A FilterAnalyzer | aec_filter_analyzer.py + 1 config + lazy-init + diag | C.B consumer |
| C.B FilteringQualityAnalyzer | aec_filter_quality.py + 1 config + lazy-init + diag | C.E/C.D consumer |
| C.C AecState extension | aec.py back-ref pattern on existing AecState class | C.E consumer |
| C.D-α leakage_diverged | aec.py + AecConfig 3 fields + check/apply methods | follow-up after RES per-branch migration |
| C.E surgical RES migration | aec.py 1-line guarded switch + AecConfig flag | Pareto-front-walking ablation |

**Total lines added (estimate)**: ~700 lines code + ~3500 lines docs.
All flag-OFF byte-equal verified at every sprint boundary.

## 4. v3.19+ priority backlog (ordered)

### Tier 1 (high-confidence + high-leverage)
1. **C.E Pareto-front-walking** (6-10 sprints, MEDIUM RISK)
   Per-RES-branch ablation: identify branches inside
   `ResFilter._stage_*` that benefit from `fq_usable` (DT NE preserve)
   vs hurt from it (FS echo suppression). Migrate only the helpful
   ones. Highest theoretical ceiling per C.E single-case +1.2 Δdeg.

2. **C.D-α + C.E combined retry** (3-5 sprints, MEDIUM RISK)
   After Tier 1 lands, re-bench C.D-α with RES that responds to
   leakage_diverged Q-boosts. The C.D-α silent failure mode may flip
   once RES consumes fq_usable per-branch.

### Tier 2 (substrate-ready, lower-confidence)
3. **B FilterMisadjustment retry** (3-5 sprints, MEDIUM RISK)
   Substrate from Phase B + C.B's fq_usable + C.A's consistent_estimate
   together solve B's structural silence (refined_usable coverage gap).

4. **F event classification retry** (3-5 sprints, MEDIUM RISK)
   Substrate from Phase F + (future) matched-filter delay arc → asymmetric
   reset becomes safe to enable.

5. **A shadow NLMS retry** (5-7 sprints, MEDIUM-HIGH RISK)
   Substrate from Phase A + C.E-derived RES per-branch migration may
   recover the FS regression while keeping the DT win.

### Tier 3 (new substrate needed)
6. **Matched-filter NLMS + histogram delay detector** (8-12 sprints, HIGH LOE)
   v3.20 gating arc per F + A closeouts. Replaces GCC-PHAT DelayEst.
   Unlocks Phase F's asymmetric reset + Phase D-γ retry.

7. **Reverb-port** (5-8 sprints, MEDIUM RISK)
   AEC3 reverb_model.cc / reverb_decay_estimator.cc port. Out of v3.18
   scope; substrate from Phase 0 fetch in `docs/aec3_extracts/`.

### Tier 4 (architecture)
8. **InitialState + TransparentMode lifecycle** (4-6 sprints)
   AEC3 has lifecycle states (initial / transition / steady). Our
   pipeline has implicit warmup but no formal lifecycle. Builds on
   C.C AecState extension.

## 5. Cycle-level lessons learned

1. **AEC3 mechanisms need AEC3 co-design context.** Every AEC3-port
   arc (D-γ / F / A / B / C.D-α / C.E) closed because some upstream or
   downstream piece of AEC3's architecture wasn't there. The audit-only
   port pattern (C.A / C.B / C.C) is the right shape for AEC3 alignment
   — port the observability, then port the mechanism.

2. **Cheap pre-bench gates save wall-clock.** Phase B's B.4 fire-rate
   gate (0/61 fires) saved a 60-case bench. C.A's coverage-delta
   pre-bench gate (61/61 cases ≥ 10pp) confirmed value before bench.
   Apply this discipline to every future arc.

3. **§0.4 substrate-retained protocol is right.** v3.18 closes with
   8 substrate arcs all default-OFF, all available for v3.19+. v3.17
   pioneered this; v3.18 confirms it scales. Closures aren't sunk cost
   if they leave reusable substrate.

4. **The Pareto wall is real and shaped consistently.** v3.13 E5 (saturation
   detector), Phase A (shadow NLMS), C.E (RES migration) all hit
   roughly the same FS-DT trade-off shape: every +1 dB DT gain costs
   ~0.5-1 dB FS. v3.19+ Pareto-walking arcs need this prior baked in.

5. **Audit-only ports are real engineering value.** C.A / C.B / C.C
   shipped no algorithm change but enable everything downstream. Future
   cycles can include audit-only sprints as substrate creation, not as
   "weak deliverables."

## 6. Branch state

```
feature/v3.18-aec3-fetch (HEAD: 6200ec4 + uncommitted changes)

Uncommitted (this closeout):
  docs/v3_18_c_d_alpha_closeout.md  [new]
  docs/v3_18_c_e1_consumer_migration_design.md  [new]
  docs/v3_18_c_e_closeout.md  [new]
  docs/v3_18_cycle_closeout.md  [new, this file]
  docs/v3_18_plan_revision_2026_05_15.md  [updated]
  python/aec.py  [C.D-α + C.E flags + methods + wiring]
  python/eval_aec_challenge.py  [env-var bridges]
```

## 7. §0.7 merge authorisation package

Per CLAUDE.md branch model, v3.18 closeout should ship via one of two
paths:

### Option A (recommended): tag `v3.18.0` as substrate-only cycle
- Merge `feature/v3.18-aec3-fetch` → `main`
- Tag `v3.18.0` with this closeout as release notes
- All 8 substrate arcs ship default-OFF (zero user-facing change)
- v3.19+ branch from `v3.18.0`

### Option B: defer merge until C.E Pareto-walking lands
- Keep `feature/v3.18-aec3-fetch` open
- v3.19 work on same branch (rename `feature/v3.19`)
- Single merge when first algorithm change actually ships

Recommendation: **Option A**. Substrate is real value; the cycle is
genuinely complete on its own terms; v3.19+ benefits from a clean
baseline.

## 8. User decision points for §0.7

1. **Approve closeout?** — Sign off on 0-algorithm-change cycle outcome
2. **Option A or B?** — Merge to main now, or defer
3. **v3.19+ priority?** — Tier 1.1 (C.E Pareto-walking) is highest leverage
   but medium risk; alternatives at Tiers 2-4
4. **C-port debt?** — All 8 substrate arcs need C-port follow-up per
   CLAUDE.md byte-equal invariant. Suggest deferring until v3.19+ ships
   the first algorithm change

## 9. Cross-references

- [docs/v3_18_plan_revision_2026_05_15.md](v3_18_plan_revision_2026_05_15.md) — cycle plan + per-phase status
- All `docs/v3_18_*_closeout.md` + `*_verdict.md` files — per-phase detail
- [docs/v3_17_closeout.md](v3_17_closeout.md) — predecessor cycle (also 0 algorithm changes)
- [CLAUDE.md](../CLAUDE.md) — branch model + §0.7 protocol
- `~/.claude/memory/` — feedback / project memories driving this cycle
