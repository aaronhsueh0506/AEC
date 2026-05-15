# v3.18 Phase 0.6 — Decision Sprint (2026-05-15)

**Phase 0 outcome**: AEC3 source fetched + 6 RES modules mapped + RES-side
gap table compiled. Decisions captured below; this doc closes Phase 0.

## 1. Phase D — verdict: **OPEN at D-γ scope**

**Rationale**:
- AEC3 has 2 distinct mechanisms for NE HF protection that we lack:
  - **R2** (per-bin mask profile swap) — primary mechanism, addresses
    user's main concern "webrtc 怎麼做 residual 可以學的"
  - **R4** (subband NE detector) — secondary, addresses cohort tail
    misclassification
- Phase 0.3 mapping confirms the user-observed structural difference:
  AEC3's per-bin masking thresholds are themselves per-bin (LF→HF
  interpolation) AND get swapped based on NE detector; ours have per-bin
  gain but scalar-modulated thresholds.
- v3.17 B.2 closure's "5-8 sprint mechanism R&D" estimate matches D-γ's
  7-10 sprint LOE.

**Scope D-γ** (final):
1. **D.1** — Subband NE detector port (R4), audit-only flag default OFF (1-2 sprints)
2. **D.2** — Pre-tune two per-bin mask profile tables (R2 substrate) (1 sprint)
3. **D.3** — Rewrite `_stage_gain_compute` per-bin masking using profile lookup (R2 + R14) (2-3 sprints)
4. **D.4** — Migrate 35+ legacy `effective_dt` floor-lift consumers (2 sprints, per-batch byte-equal)
5. **D.5** — Tune profile sets on 60-case grid + nores listen (1 sprint)
6. **D.6** — 800-case AECMOS + ship gate (1 sprint)
7. **D-aux1** — `X2_noise_floor` minimum-statistics tracker (R6) (1 sprint, optional)

**Total LOE**: 7-10 sprints (D-aux1 optional, +1 sprint if included)

**Hard bar (D-γ)** — final:
| Bucket | Metric | Bar |
|---|---|---|
| NE | Δdeg | ≥ +0.010 (HF protection — primary target) |
| DT | Δdeg | ≥ +0.005 (v3.13 E2 Path 3 debt recovery) |
| FS | Δecho | ≥ -0.010 |
| cohort tail (qNvSMyU + 0I0XMl3M) | Δecho | ≥ -0.05 |

**Kill criterion (D-γ)**: NE Δdeg < +0.003 OR DT Δdeg < +0.003 →
close per §0.4, retain substrate (D.1 subband detector + D.2 profile
tables remain default-OFF).

**Branch**: `feature/v3.18-res-per-band-ne`

## 2. Phase A — hard bar unchanged

Per v3.18 plan Phase A §"Hard bar":
- cohort tail (qNvSMyU) Δecho ≥ +0.030 dB
-護欄 unchanged

**Phase 0 evidence relevant to Phase A**:
- AEC3 SuppressionGain uses `aec_state.Erle()` (per-bin) — does NOT
  require shadow NLMS to function. Phase A's primary gain target stays
  the cohort-tail EPC trigger signal (Gap #1+#2), which is upstream of RES.
- Phase 0.2 §2.1 item 1 confirms `R2 = S2_linear / Erle` is the linear-mode
  path. Phase A improves S2 quality (better filter → better Erle), which
  indirectly helps R2 quality. No Phase A hard-bar adjustment needed.

## 3. Phase B — hard bar unchanged

Per v3.18 plan Phase B §"Hard bar":
- cohort tail Δecho ≥ +0.020 dB
- DT Δdeg ≥ -0.010, FS Δecho ≥ -0.010

**Phase 0 evidence relevant to Phase B**:
- AEC3 `ReverbDecayEstimator::Update` (Phase 0.4 §2.1) reads filter
  impulse response → would benefit if W is well-scaled. This is
  consistent with Phase B improving filter quality → eventual v3.19
  reverb arc value.
- No Phase B hard-bar adjustment needed.

## 4. Phase C — hard bar refined

Per v3.18 plan Phase C §"Hard bar":
- DT bucket Δdeg ≥ +0.015

**Phase 0 evidence relevant to Phase C**:
- Phase 0.3 §6 found that v3.13 E2 Path 3 DT debt is BETTER targeted
  by Phase D-β than by Phase C. Phase C tightens `usable_linear_estimate`
  but DT debt is masking, not gating.

**Refinement**: Phase C DT hard bar **lowered to ≥ +0.010** (Phase D-β
takes the +0.005 share). The +0.015 target is preserved for "Phase C
+ Phase D-β combined" 800-case bench at v3.18 closeout.

**Combined v3.18 cycle target**: DT Δdeg ≥ +0.015 across full
v3.18 stack (A + B + C + D-γ). Per-phase shares:
- Phase A: 0 (FS gain target)
- Phase B: 0 (FS gain target)
- Phase C: +0.010 (usable_linear gating)
- Phase D-γ: +0.005 (per-bin mask swap)

## 5. Phase E — unchanged

Per v3.18 plan Phase E. Phase 0 evidence doesn't change the 14-flag
substrate inventory. Phase E migration of new D-γ flags (added to
inventory after Phase D ships) handled in E.6 update sprint.

## 6. Cross-phase coordination

| Dependency | Resolved? |
|---|---|
| Phase A method-collision with Phase B at PBFDKF L1916-1920 | Yes — A adds copy semantics, B adds scale_filter, both method-additions |
| Phase C `usable_linear_estimate` substrate ready | Yes — half-built at `aec.py:4809` |
| Phase D `R2_unbounded` second residual path | New — D.3 introduces dual residual outputs |
| Reverb arc deferred to v3.19 | Confirmed in Phase 0.4 §5 |

## 7. Branches state

```
feature/v3.17 HEAD 87730e4 (v3.17 closeout, unmerged)
  └── feature/v3.18-aec3-fetch (Phase 0.1-0.5 docs)
           ├── feature/v3.18-shadow-nlms (Phase A, pending)
           ├── feature/v3.18-misadj-scaling (Phase B, pending, parallel A)
           ├── feature/v3.18-leakage-quality (Phase C, pending, post A+B)
           ├── feature/v3.18-res-per-band-ne (Phase D-γ, pending, post C)
           └── feature/v3.18-preset-promotion (Phase E, pending, partial post D)
```

§0.2 file-disjointness verified: A/B parallel-safe (method-additions
in same class group, mergeable in either order); C/D depend on A+B
land; E migration sprints depend on D land.

## 8. Open questions deferred to within-phase sprints

- Phase A: post-copy hangover (Gap #4) folding into A.4 — pending
  exact NLMS rate vs Kalman rate gap measurement (A.5 grid)
- Phase B: `ScaleFilter` `P *= scale²` (Option A) vs P-untouched (Option B
  per AEC3) — B.3 sprint
- Phase C: 35+ `_filter_converged` consumer migration order (C.5
  per-batch) — needs grep audit before C.5 starts
- Phase D-γ: subband detector `subband1.low/high` / `subband2.low/high`
  default values — port from AEC3 default config OR re-tune from
  cohort tail spectra (D.5 listen confirms)

## 9. Closeout

Phase 0 status: **CLOSED 2026-05-15**.

Deliverables:
- ✓ Phase 0.1 — full AEC3 source @ `docs/aec3_extracts/src/aec3/`
- ✓ Phase 0.2 — `docs/aec3_residual_pipeline_mapping.md`
- ✓ Phase 0.3 — `docs/aec3_subband_ne_detector_mapping.md`
- ✓ Phase 0.4 — `docs/aec3_reverb_mapping.md`
- ✓ Phase 0.5 — `docs/aec3_reference.md` §9.1 + §11 RES gap table updated
- ✓ Phase 0.6 — this doc

**Next sprint**: Phase A.1 — design `shadow_kwargs` PBFDAF-safe
configuration (move `alpha_r_scale` / `leak` Kalman-only attrs out,
add `shadow_mu` 0.3-0.5 NLMS rate). Branch:
`feature/v3.18-shadow-nlms`.

Phase B.1 starts in parallel on `feature/v3.18-misadj-scaling`.
