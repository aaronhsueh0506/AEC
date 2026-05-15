# v3.17 plan — Stability + Mechanism + Tunable Strength (2026-05-15)

**Branch**: `feature/v3.17` (off `4cc3b3b` v3.16.0 merge).
**Scope** (per user 2026-05-15): three-phase cycle bundling all
identified v3.17 directions in one cycle.
**Predecessors**:
[`docs/v3_16_closeout.md`](v3_16_closeout.md) (v3.16 outcome),
[`docs/v3_16_c6_delay_est_audit_verdict.md`](v3_16_c6_delay_est_audit_verdict.md) (movement-rate backlog),
[`docs/v3_16_c9_reverb_aware_audit_closure.md`](v3_16_c9_reverb_aware_audit_closure.md) (C9.v2 scope).

---

## Strategic framing

v3.16 cycle (1 production change + 5 audit closures) revealed the
DSP-only AEC at fl=832/balanced is approaching its perceptual ceiling
from the OUTSIDE — most cohort-tail-targeted RES policies subsume to
existing logic. v3.17 redirects effort to:

1. **Phase A — Stability baseline**: code-health audit + dead code
   removal + bug-fix sprint to deliver a known-clean substrate.
   "首先要找出一版穩定" per user 2026-05-15.
2. **Phase B — Audible-target mechanism arcs**: 3 mechanism arcs
   targeting concrete listen-validated debt cases (0I0XMl3M movement,
   pcb1N reverb, DT-NE compression). Each independently sized so any
   subset can ship.
3. **Phase C — Tunable strength interface**: preset gradient design
   (MILD → MAXIMUM with monotone aggressiveness) so downstream users
   can dial suppression strength without per-knob expertise.

Estimated total LOE: **16-23 sprints**.

---

## Phase A — Stability baseline (3-4 sprints)

### A.1 Dead-code inventory + targeted removal

Confirmed dead code candidates from initial scan:

| ID | Location | Why dead | Removal LOE |
|---|---|---|---|
| **A.1.1** | `_stage_dt_indicator` flow line 7384-7389 (`dt_reduction → effective_over_sub → self.res.over_sub`) | `over_sub` only consumed by `gain_type=='wiener'` / `'spectral_sub'` paths (lines 3244-3251); ALL 5 BALANCED presets use `gain_type='enr'`. Same closure family as v3.16-A H1 | S |
| **A.1.2** | `_shadow_mu_holdoff` init/reset (5831/5988/6267) | Initialized + reset in 3 sites, NEVER READ. Comment says "kept harmless for future wiring" but actually adds maintenance noise | S |
| **A.1.3** | P3h reset action (line 420 area) | Per inline comment "P3h reset action becomes dead code" post F2.2 — verify and remove | S-M (verify-first) |
| **A.1.4** | line 3680 dead branch retained | Per inline comment "Branch was dead code retained" — verify and either remove or document why | S |
| **A.1.5** | line 3310 cap-in-FS dead code | Per inline comment "fires in FS — cap is largely dead code" — quantify and remove if 0/800 | M (audit-first) |
| **A.1.6** | line 1133 diverged_reset triple-AND gate | Per comment "v3.11 diverged_reset: triple-AND gate unblocks dead code safely" — investigate gate state | M (verify) |

**Hard bar (per removal)**: 5-case byte-equal sanity PASS on
(FS_static, FS_movement, DT_static, NE, pcb1Nh0Z) for both
`_ours.wav` and `_ours_nores.wav`. Same protocol as v3.16 C1.

### A.2 Default-OFF substrate flag triage

42 default-False bool fields in `AecConfig` (counted via grep).
Categorize each into:
- **ACTIVE substrate** — flag intentionally OFF for production but ON in test/research; KEEP
- **DEAD substrate** — code path no longer reachable / mechanism shipped elsewhere; REMOVE
- **BROKEN substrate** — flag exists but enable path produces unexpected behavior; FIX or REMOVE

Output: `docs/v3_17_a2_substrate_triage.md` per-flag verdict.

### A.3 Code review for logic bugs

Scan for:
- State machine inconsistencies (filter_state enum vs string mismatches)
- Flag combinations producing unexpected behavior (e.g. C5 architecture
  problem v3.16 deferred — Arc D × Arc R style write conflicts)
- Edge-case handling (init/reset paths, post-state-restore behavior)
- Numerical issues (denominator zeros, division by tiny values)
- B-list outstanding items from prior plans (B4 already fixed; B5/B6
  status verify; B7-B11 known-substrate)

### A.4 Stability gate

After Phase A.1-A.3 land:
- 5-case byte-equal sanity PASS (each individual fix already verified;
  this is end-of-Phase-A consolidation)
- 60-case subset bench Δ vs `4cc3b3b` (v3.16.0) expected ≤ 0.001 dB on
  all bucket means
- Cohort tail (`qNvSMyU`) Δecho ≥ -0.005 (no regression)

**Output**: `docs/v3_17_a_stability_baseline.md` declaring v3.17 Phase A
done; HEAD now considered "stable substrate" for Phase B.

---

## Phase B — Audible-target mechanism arcs (10-15 sprints)

### B.1 Movement-rate DelayEst (2-3 sprints, 0I0XMl3M target)

**Origin**: v3.16 C6 audit
([`docs/v3_16_c6_delay_est_audit_verdict.md`](v3_16_c6_delay_est_audit_verdict.md) §6).
0I0XMl3M_FS_movement showed estimated_delay jumps 1230→4132→4369→2 +
ERLE p5_bad −49 dB. Period_seconds=2.0 lags fast movement.

**Mechanism**: shorten DelayEst `period_seconds` 2.0 → 0.25 s when
motion detector indicates fast change. Motion detector substrate
exists in v3.15 Arc M closure (variance EMA proxy).

**Sprints**:
- B.1.S1: design + 5-case byte-equal sanity (flag default OFF)
- B.1.S2: tune motion detector threshold + 0I0XMl3M listen + 8-case
  trace
- B.1.S3 (gated on PASS): 60-case bench + cohort tail regression check

**Acceptance**: 0I0XMl3M ERLE p5_bad ≥ −20 dB AND 60-case AECMOS no
regression (FS Δecho ≥ −0.020, DT Δdeg ≥ −0.005, cohort tail Δecho
≥ −0.05).

### B.2 C9.v2 multi-feature reverb-aware RES override (5-8 sprints, pcb1N target)

**Origin**: v3.16 C9 closure
([`docs/v3_16_c9_reverb_aware_audit_closure.md`](v3_16_c9_reverb_aware_audit_closure.md))
identified plain Pearson r insufficient. Multi-feature classifier
needed.

**Mechanism**: combine 3-4 features:
- `delay > 0.8 × fl_samples` (fl-undercoverage signal)
- coherence-band statistics (per-band coh² mean low across voice band)
- `top1/top2 par_ratio` (DelayEst multi-peak ambiguity)
- envelope cross-correlation magnitude (alternative to sample-level r)

**Sprints**:
- B.2.S1: feature engineering + pcb1N + 5+ similar reverb-heavy cohort
  case discovery (audit on 800-case)
- B.2.S2: classifier design + threshold calibration on positive +
  negative cohort
- B.2.S3: RES override design (which knobs to flip when trigger fires)
- B.2.S4: wire flag `reverb_aware_res_override=False` (default OFF) +
  byte-equal sanity
- B.2.S5: tune override values + pcb1N listen + 60-case A/B
- B.2.S6 (gated): 800-case FP regression guard
- B.2.S7-S8 (if needed): per-band override refinement

**Acceptance**: pcb1N audible improvement (subjective) + 800-case
AECMOS no regression (FS / DT / NE all bucket-mean Δ ≥ −0.005).

### B.3 C2.v2 narrow per-state ENR (3-4 sprints, DT-NE debt)

**Origin**: v3.13 E2 Path 3 ship left DT_static Δdeg −0.050 /
DT_movement Δdeg −0.025 ACCEPTED but unfixed. v3.15 §1.2 attempted
recovery and CLOSED CANNOT SHIP (FS-vs-DT wall). v3.16 plan §C2 noted
narrow scope variant might escape wall.

**Mechanism**: per-state × per-band ENR but ONLY on `coarse_learning +
suspicious_dt` states (NOT `refined_usable` which is byte-equal locked).
Hypothesis: DT-NE compression in non-converged states might be
recoverable without inheriting §1.2's `coarse_learning` FS damage if
gating is tighter.

**Sprints**:
- B.3.S1: §1.1 audit data re-mining — confirm `coarse_learning +
  suspicious_dt` is the actual compression locus (vs `refined_usable`
  from §1.1 H1)
- B.3.S2: design narrow per-state × per-band ENR table (table values
  for non-converged states only) + flag `enr_narrow_per_state=False`
  default OFF
- B.3.S3: 5-case byte-equal sanity (`refined_usable` byte-equal proof)
- B.3.S4: 60-case bench + cohort tail regression guard

**Acceptance**: DT bucket Δdeg ≥ +0.015 AND cohort tail Δecho ≥ −0.05
AND FS bucket Δecho ≥ −0.020. Hard bar acknowledges higher RISK
(§1.2 wall).

**Kill criterion**: if DT bucket recovery < 0.005 after 2 tuning sweeps
OR FS Δecho violates bar, CLOSE per §0.4 (same family as §1.2).

### B.4 (optional, gated on user auth) NL detector revival

Per v3.13 E5 saturation detector substrate preserved. Combine with
phase-aware NL research-track if user authorises (per
`feedback_dsp_only_until_completion`). Out of v3.17 default scope.

### B.5 v3.17 audible-debt closure list

After Phase B.1-B.3 (and optional B.4) land:

| Original debt | Phase B candidate | Expected outcome |
|---|---|---|
| v3.13 E2 DT_static -0.050 / DT_movement -0.025 | B.3 C2.v2 | partial recovery if escape wall |
| 0I0XMl3M extreme movement -49 dB ERLE | B.1 movement-rate DelayEst | recover to ≥ -20 dB |
| pcb1N reverb / fl-undercoverage | B.2 C9.v2 | audible improvement |
| qNvSMyU cohort tail catastrophe | (none — DSP wall, see v3.16-A closure) | UNADDRESSABLE in v3.17 |
| HF cap removal during DT (v3.13 unmask) | B.3 C2.v2 narrow path | partial |

---

## Phase C — Tunable strength interface (3-5 sprints)

### C.1 Strength knob inventory (1 sprint)

Identify all "strength" knobs across the pipeline:
- `res_g_min_db` (-20 dB default) — lower = more aggressive
- `res_over_sub_base / res_over_sub_scale` — higher = more aggressive
- `enr_t_ne / enr_s_ne` (or per-band variants from B.3 if shipped) —
  lower thresholds = more aggressive
- `res_max_drop_db_per_frame` — higher = faster gain drop
- `res_spectral_floor_db` — lower = deeper notches
- (post-B.1) DelayEst period_seconds + motion threshold
- (post-B.2) reverb-aware override aggressiveness scalar

Output: `docs/v3_17_c_strength_knob_inventory.md`.

### C.2 Preset gradient validation (2-3 sprints)

Audit current 5 presets (MILD/SOFT/BALANCED/AGGRESSIVE/MAXIMUM) on
60-case subset:
- Verify monotone strength: AECMOS echo metric should be monotone
  non-decreasing MILD → MAXIMUM
- Verify monotone NE preservation: AECMOS deg metric should be monotone
  non-increasing
- Identify outlier knobs that don't move monotonically with preset

Output: `docs/v3_17_c_preset_gradient_audit.md`.

### C.3 (gated on C.2 outliers) Preset re-tuning

If C.2 finds non-monotone knob behavior (e.g. an "MILD" preset
accidentally more aggressive than "BALANCED" on some metric),
re-tune to enforce monotone gradient.

**Acceptance**: monotone gradient on FS Δecho, DT Δdeg, NE Δdeg
across all 5 presets on 60-case + xrtntuju 5-clip listen.

---

## Phase sequencing

```
Phase A — Stability (3-4 sprints, sequential)
  A.1 dead code inventory
  A.2 substrate flag triage
  A.3 logic bug review
  A.4 stability gate (byte-equal vs v3.16.0)

Phase B — Mechanism arcs (10-15 sprints, can parallelise some)
  B.1 movement-rate DelayEst (2-3)  — independent, smallest LOE
  B.2 C9.v2 reverb-aware (5-8)      — independent, biggest LOE
  B.3 C2.v2 narrow ENR (3-4)        — independent, highest risk

Phase C — Tunable strength (3-5 sprints)
  C.1 knob inventory (1)            — gated on Phase B closure
  C.2 preset gradient audit (2-3)
  C.3 retune if outliers (1-2)
```

Total: **16-23 sprints**.

---

## Verification rules per sprint (carried from v3.15/v3.16 §5)

1. §0.1 per-sprint hard ordering (trace before edit, baseline JSON drift,
   byte-equal flag-OFF, standard config, listen, root cause, one-arc
   per sprint)
2. §0.2 state-mutation disjointness for parallel arcs
3. §0.3 oracle validation before design lock
4. §0.4 negative-result acceptance (lever moves < 0.002 dB → ship as
   substrate or close)
5. §0.6 metric channel rules (linear-filter arcs use nores listen
   PRIMARY; RES-output arcs use AECMOS PRIMARY; hybrid both)
6. §0.7 branch isolation + user-gated merge — `feature/v3.17` not
   pushed/merged until full Phase A+B+C verdict pack reviewed by user

Standard 60-case subset bench config: `preset=balanced / fl=832 /
cng=True / -j 6` (v3.15 §10.S0c subset bench infra).

---

## Critical files (v3.17 modification targets)

**Will be modified (Phase A)**:
- `python/aec.py` — dead code removal sites (lines 420 / 1133 / 3310 /
  3680 / 7384-7389); _shadow_mu_holdoff init/reset cleanup; possibly
  AecConfig flag pruning
- (Phase B) `python/aec.py` PBFDKF + DelayEstimator + ResFilter per arc
- `python/eval_aec_challenge.py` — env overrides for new flags
- `docs/v3_17_*.md` — phase verdict docs

**Read-only references**:
- `docs/v3_16_*.md` — v3.16 closure verdict pack
- `python/res_refactored/` — v3.16 substrate (don't modify per design lock)
- `c_impl/` — C port follows once Python algorithm merged

---

## Out-of-scope

- NN any module (`feedback_no_nn_mention`)
- C5 architectural modular interface (deferred; only worth investing
  if multiple Phase B arcs all need it)
- WebRTC AEC3 port / replacement
- `fl > 832` for production C
- C7 Arc M.v3 retry (was CLOSED v3.15 §1.5b; would need new detector
  primitive not on v3.17 critical path)
- C8 Arc G partial decay (LOW priority; Arc G already CLOSED v3.15)
