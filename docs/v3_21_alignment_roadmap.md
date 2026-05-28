# v3.21 AEC3 Alignment Roadmap (post-3.21.6.2)

**Scope rule (HARD)**: v3.21.x = AEC3-strict alignment work only. Any
work that does not have an AEC3-default-production equivalent goes to
v3.22 (which is reserved for genuinely beyond-AEC3 optimisation).

## Audit summary (2026-05-28, v3.21.6.2)

A full pass over the "still missing" list below found that 80 % of the
items were either already aligned, no-consumer additive surfaces, or
no-op under our `current_size = max_size = 13` steady state. Four items
were shipped in v3.21.6.2 (see [CHANGELOG.md](../CHANGELOG.md) §3.21.6.2);
the remainder either require an architecture redesign (Tier C #11) or
carry too much behavioural risk without a dedicated design lock (Tier A #2
`IsRenderTooLow`).

Items shipped in v3.21.6.2: **Tier A #3** (SubtractorOutputAnalyzer
signals — additive), **Tier A #5** (dedup of inline R0.4 dead code),
**Tier B #8** (poor_coarse hangover physical-meaning correction —
behavioural), **Tier C #9** (`PBFDAF.zero_filter_partitions` strict
semantics — no-op in steady state).

Items deferred:
- **Tier A #2** `IsRenderTooLow` + `non_zero_render_seen_` latch — would
  change stationarity-noise-floor update timing on every frame; too
  high-risk to bundle without isolated bench. Phase-2 design item.
- **Tier A #4** TransparentMode HMM — Phase 2; consumes the new
  `any_coarse_filter_converged` / `all_filters_diverged` signals shipped
  this cycle.
- **Tier A #6** ScaleFilter exact port — retiring
  `FilterMisadjustmentEstimator` needs its own ablation arc.
- **Tier B #7** 4-mode HF tuning — needs separate per-band re-tune cycle.
- **Tier C #11** `convergence_seen` latch redesign — architecture work.

## Status at v3.21.6.1 (current production)

- **22 / 22** AEC3-strict alignment flags that have config presence are
  shipped ON and hard-coded into call sites.
- **6** alignment flags exist but did not ship — each documented with
  reason and re-evaluation gate (see `CHANGELOG.md` v3.21.6.1 entry).
- **2 production-blocking bugs fixed** on internal cohort:
  - nores LF artifact — closed
  - painted-black HF — 24.3 % → 14.0 % (residual 14 % is AEC3-inherent;
    `bin/aec3_cli` fails the same threshold on the same cases).
- **800-case AECMOS Pareto**: FS echo +0.19 / DT echo +0.27 vs v3.21.0,
  but DT deg −0.46 / NE deg −0.30 — strong cancellation, weaker
  preservation. Root cause: mode-switched gain shape not in production
  because the upstream NE detectors are still not ported.

## What is still missing for v3.21 to be "AEC3 alignment complete"

The 6 OFF flags are only part of the picture. The bigger gap is
**entire AEC3 modules that we never started porting** — these are not
listed as OFF flags because they have no config flag, just absence.

### Tier A — entire modules not yet ported (AEC3 production has)

> **2026-05-28 correction**: Earlier draft listed `DominantNearendDetector` as a Tier A gap. **That was wrong** — it is already ported (lives inline at `python/modules/residual/suppression_gain.py:311` as `_DominantNearendDetector`), AEC3-strict parameters (enr=0.25 / exit=10 / snr=30 / trigger=12 / nearend_tuning lf=(1.09,1.1,0.3) hf=(0.1,0.3,0.3)), owned by `SuppressionGain`, drives 2-mode (normal/nearend) gain shape per frame. The only deviation from AEC3 strict is `hold_duration_ms=500` (vs AEC3 200) — intentional v3.21 tuning per [[feedback-aec-target-aec2-aec3]]. v3.21.7 Phase 1 attempt extracted the detector to a separate module (architecture refactor) but 12-case bench showed near-zero behaviour change vs v3.21.6.1, confirming the detector was already doing its job. **The −0.46 DT deg / −0.30 NE deg debt is therefore NOT a "detector absent" gap — it lives downstream.**

| # | Module | AEC3 source | What it does | Why we don't have it |
|---|---|---|---|---|
| 2 | `EchoAudibility` (full) | `echo_audibility.{cc,h}` | JND-weighted per-bin audibility; gates SuppressionGain per-bin floor | only `aec3_post_stationarity_zero_enabled` (binary) ported; full JND-weighted scaling missing |
| 3 | `SubtractorOutputAnalyzer` (full signal set) | `subtractor_output_analyzer.cc` | Emits `saturated / diverged / converged / no_excitation` to AecState + TransparentMode | only `converged` ported |
| 4 | `TransparentMode` HMM variant | `transparent_mode.cc:53-100` | 2-state HMM detects "echo absent" → bypass SuppressionGain | only `LegacyTransparentMode` substrate; default OFF; HMM variant never started |
| 5 | `FullBandErleEstimator` | `fullband_erle_estimator.cc` | Continuous-quality ERLE (vs binary converged); feeds reverb decay update | only `SubbandErleEstimator` ported |
| 6 | `ScaleFilter` (exact) | `scale_filter.cc` + per-block scale ratios | AEC3's production filter-rescaling mechanism | we use `FilterMisadjustmentEstimator` (single fullband ratio + asymmetric EMA) as functional substitute |

### Tier B — partial port, AEC3 production has more

| # | Component | AEC3 | Ours | Gap |
|---|---|---|---|---|
| 7 | `HighFrequencySuppressionConfig` + `DominantNearendConfig` (4-mode) | normal / nearend / dominant_nearend / convergence per-band tuning | normal + subset of nearend | HF / nearend / dominant_nearend three modes lack strict tuning |
| 8 | `poor_coarse_filter_counter` hangover | full 25-block hangover + state machine | simplified version | coarse-filter freeze timing not exactly aligned |

### Tier C — flag exists, architecture blocker prevents ship

| # | Flag | Blocker | What needs to happen |
|---|---|---|---|
| 9 | `use_aec3_zero_filter_on_epc` | Earlier `W.fill(0)` impl was over-aggressive (NOT strict AEC3 — AEC3's `ZeroFilter(current, max)` is a no-op when current==max==13 in steady state). Need PBFDKF-aware `ZeroFilter` that respects current/max semantics. | Port AEC3 `AdaptiveFirFilter::ZeroFilter` exactly + redesign or co-exist with `PathChangeRegimeHandler` |
| 10 | `use_aec3_epc_classification` | Depends on #9 | Ship jointly with #9 |
| 11 | `use_linear_filter_output_selection_for_final_output` | `convergence_seen` is binary latch; AEC3 is counter-based gate. Latch contamination on cohort tail. | Redesign `convergence_seen` to counter-based; this is itself an AEC3 alignment item ([[project-usable-linear-gate3-latch-bug]]) |

### Already aligned to AEC3 production default (NOT a gap)

- `ReverbDecayEstimator` adaptive path — AEC3 production sets
  `ep_strength.default_len = 0.83 > 0` which **disables** the
  estimator. Our `use_adaptive_decay = False` matches.
- `use_coarse_e2_time_domain_parity` — confirmed no-op (threshold-bound
  dormant); documentation-only divergence.

### Genuinely v3.22 (not alignment)

- `use_aec3_poor_coarse_rescue_copy` strict port — 12-case Pareto FAIL
  (`xFk7 +0.195 / MYrVxVEM −0.431 / qNvSMyUS −0.216 / 9xjhi −0.112`).
  PBFDKF architecture cannot satisfy AEC3 strict rescue semantics
  (`coarse_conv = 0 %` in every case × variant). **Retest gate
  attached**: re-evaluate after Phase 3 of this roadmap completes —
  EchoAudibility + DominantNearendDetector + 4-mode HF tuning may
  change the Pareto picture enough that conditional rescue becomes
  viable. If still FAIL after retest → conditional / FS-only / DT_mvmt-
  only gating belongs in v3.22.

## Phased Execution Plan

Ordered by **deg-debt recovery × implementation cost**. Each phase has
its own bench gate (12-case + 800-case) before next phase starts.

### Phase 1 — `v3.21.7` — **REPLAN PENDING** (original DominantNearendDetector plan invalidated 2026-05-28)

**Original plan**: port `DominantNearendDetector` + mode-switched
SuppressionGain. **Invalidated** by 12-case dry-run showing the
detector is already ported and firing — near-zero behaviour change
from architecture refactor.

**Replan options pending diagnostic evidence**:

A. **`SuppressionGain` mask shape tuning** — `nearend_tuning` is
   AEC3-strict (lf=(1.09,1.1,0.3) hf=(0.1,0.3,0.3)), but our PBFDKF
   refined-filter chain is MORE aggressive than AEC3's NLMS refined
   in steady state. Same mask shape against stronger residual could
   produce harder suppression. Candidate: per-band re-tuning of
   `nearend_tuning.mask_hf` to recover HF preservation, OR `mask_hf`
   conditional on `is_dominant_nearend()`.

B. **`use_linear_filter_output_selection_for_final_output`** (Tier C
   #11 in this doc) — currently OFF because `convergence_seen` is
   binary latch. If contaminated `usable_linear` causes us to feed
   over-aggressive linear residual to SuppressionGain instead of
   capture spectrum Y, that explains the HF wipe during NE. Fix:
   redesign `convergence_seen` to counter-based gate, ship the flag.

C. **EchoAudibility full port** (Tier A #2) — JND-weighted per-bin
   floor. HF psychoacoustic threshold is looser than LF; full
   audibility port would naturally protect HF gain shape. This is a
   bigger module port but directly targets the painted-black-HF
   symptom.

**Diagnostic gate** (REQUIRED before any v3.21.7 implementation):
trace the user-supplied internal HF-painted-black case with current
v3.21.6.1 to identify which mechanism kills HF. Candidate signals to
record per frame: `is_dominant_nearend()` fire rate, `gain_hf` median,
`r2 / r2_unbounded` LF/MF/HF, `usable_linear_estimate`, HF cap fire
rate, stationarity flag fraction in HF bins. Without trace evidence
the next implementation pick is guessing.

### Phase 2 — `v3.21.8` — SubtractorOutputAnalyzer + TransparentMode HMM (unchanged)

### Phase 2 — `v3.21.8` — SubtractorOutputAnalyzer + TransparentMode HMM

**Items**: Tier A #3 + Tier A #4

**Why now**: TransparentMode HMM needs the full SubtractorOutputAnalyzer
signal set to safely fire; both ship together. Adds another deg-debt
recovery lever (TransparentMode bypasses SuppressionGain entirely when
echo is absent).

**Tasks**:
1. Port full `SubtractorOutputAnalyzer` (saturated / diverged /
   converged / no_excitation signals).
2. Port `TransparentModeImpl` HMM variant from
   `transparent_mode.cc:53-100`; replace `LegacyTransparentMode`.
3. Wire `any_coarse_filter_converged` from `coarse_filter_converged_relaxed`
   (Tier B #9-like — flag already exists but no consumer until HMM ships).
4. Default ON if Phase 1 + Phase 2 800-case is Pareto-positive vs
   v3.21.7.

**Bench gate**: 12-case + 800-case Pareto-positive vs v3.21.7.

**Expected delta**: NE deg **+0.1~0.3** further (clean NE bypasses
suppression entirely).

**Estimated**: 3–4 weeks.

### Phase 3 — `v3.21.9` — EchoAudibility + FullBandErleEstimator + ScaleFilter + poor_coarse hangover

**Items**: Tier A #2 + #5 + #6 + Tier B #8

**Why now**: completes the residual-side alignment. Each is independent
and can ship together if individually Pareto-neutral.

**Tasks**:
1. Port full `EchoAudibility` module (JND-weighted per-bin floor).
2. Port `FullBandErleEstimator` → enables Tier C-adjacent flag
   `use_aec3_erle_reverb_quality`.
3. Port `ScaleFilter` exactly; retire our
   `FilterMisadjustmentEstimator` (or keep as fallback under a
   bench-only flag for one cycle).
4. Complete `poor_coarse_filter_counter` hangover state machine to
   match AEC3 25-block timing.

**Bench gate**: each item individually + combined 800-case must be
Pareto-neutral-or-better vs v3.21.8.

**Expected delta**: HF over-suppression rescue +0.05~0.15 deg; echo
neutral or +.

**Estimated**: 4–6 weeks.

### Phase 3.5 — `use_aec3_poor_coarse_rescue_copy` RETEST (per user 2026-05-28)

After Phase 3 completes, **re-run the 12-case rescue test with the
upstream alignment now in place**. Hypothesis (untested): with
DominantNearendDetector + EchoAudibility + 4-mode HF tuning live,
the Pareto picture for rescue copy may change. If 12-case is no longer
catastrophic on MYrVxVEM / qNvSMyUS / 9xjhi → ship as part of v3.21.9.x
patch. If still FAIL → graduates to v3.22 as conditional rescue.

### Phase 4 — `v3.21.10` — Architecture redesigns for Tier C blockers

**Items**: Tier C #9 + #10 + #11

**Why last**: requires upstream redesign work, not pure port. Risk is
higher; isolated until earlier phases stabilise.

**Tasks**:
1. Port `AdaptiveFirFilter::ZeroFilter` exactly (current_size /
   max_size semantics → no-op in steady state, only zeros tail
   partitions during startup growth).
2. Decide: redesign `PathChangeRegimeHandler` to AEC3-aligned
   `Subtractor::HandleEchoPathChange` + `AecState::full_reset` cascade,
   OR co-exist with `ZeroFilter` (PathChangeRegimeHandler stays for
   cohort-tail safety net, ZeroFilter handles AEC3-spec EPC).
3. Ship `use_aec3_zero_filter_on_epc` + `use_aec3_epc_classification`
   together.
4. Redesign `convergence_seen` from binary latch to counter-based gate
   (matches AEC3 `LinearFilterUsable` semantics).
5. Ship `use_linear_filter_output_selection_for_final_output`.

**Bench gate**: 12-case + 800-case Pareto-positive vs v3.21.9 + no
cohort-tail destruction.

**Estimated**: 4–6 weeks.

## v3.22 — Beyond AEC3 (no alignment work)

Reserved exclusively for work that has NO AEC3-default-production
equivalent. Tier-1 candidates:

1. **Customer-facing RES strength API** — per user directive
   (residual layer is where user-tunable AEC strength lives).
   Proposed: `res_aggressiveness` / `res_ne_protect` / `cng_level_db`.
   Requires Phase 1–4 done first (otherwise no meaningful trade space).
2. **Conditional `poor_coarse_rescue_copy`** — if Phase 3.5 retest
   confirms strict AEC3 still FAILs, ship conditional gating (DT_mvmt
   only / convergence-qualified / etc.).
3. **Volterra non-linear inverse filter** — multi-month arc; designed
   in v3.14 design lock.
4. **C port catch-up** — `c_impl/` mirror of v3.21.x Python alignment
   work. Currently lags significantly.

NOT allowed in v3.22 (per [[feedback-no-residual-fallback-or-hf-cap-tuning]]):
- ❌ Additional HF cap tuning
- ❌ Residual-layer fallback safety nets
- ❌ Stat-aware NE proxy at SuppressionGain gain-policy level

## Versioning Summary

| Version | Status | Scope |
|---|---|---|
| **v3.21.6.1** | **shipped (HEAD)** | 22 flag alignment + cleanup |
| v3.21.7 | planned | Phase 1: DominantNearendDetector + 4-mode HF |
| v3.21.8 | planned | Phase 2: SubtractorOutputAnalyzer + TransparentMode HMM |
| v3.21.9 | planned | Phase 3: EchoAudibility + FullBandErleEstimator + ScaleFilter + poor_coarse hangover |
| v3.21.9.x | conditional | Phase 3.5 retest result for poor_coarse_rescue_copy |
| v3.21.10 | planned | Phase 4: ZeroFilter + EPC classification + convergence_seen redesign |
| v3.22 | gated on v3.21 complete | Customer RES API + Volterra + C port + conditional rescue (if 3.5 FAILs) |

## Estimated total to v3.21 "alignment complete"

**8–12 weeks** of focused work across Phase 1 → Phase 4, including bench
verification rounds. Each phase has its own ship gate; bail-out points
are at every Pareto-FAIL bench result.
