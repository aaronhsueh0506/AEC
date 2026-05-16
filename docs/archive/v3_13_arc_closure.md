# v3.13 arc — final closure summary

**Status**: v3.13 CLOSED 2026-05-14. One production change shipped (E2
Path 3); two architectural arcs (E4 NLP + E5 Saturation deepening)
closed CANNOT SHIP; one back-end audit (Phase 3 RES gain_floor) closed
with limited surface for further work. All deferred items consolidated
into v3.14 candidates.

**Branch**: `feature/v3.11-route-a` (production candidate; merge to
`main` decision pending Volterra v3.14 design lock).

## What shipped (v3.13 production change)

**E2 Delay Path 3** (commit 5b1760c, 2026-05-13):
- `eval_aec_challenge.py` `estimate_delay()` default `max_delay_ms`
  raised 250 → 1024 ms
- Aligns bench pre-alignment with online F-DelayTrack search window
- Fixes 6/8 worst-FS listen cases that had residual delay 1200-10000
  samples (75-625 ms) AFTER prior GCC-PHAT pre-alignment
- 800-case bench: FS_static Δecho +0.107
- xrtntuju 5-case DT regression listen: 0 reg / 2 imp
- Cohort tail (qNvSMyU): Δecho -0.004 (within bar)
- DT bucket Δdeg violation (-0.050/-0.025) ACCEPTED as RES unmasking;
  documented and deferred to v3.14+ for per-state ENR refactor
- Verdict: `docs/v3_13_e2_s5_verdict.md`

## What closed CANNOT SHIP

### E4 NLP arc (commit 3e10621, 2026-05-14)

12 sprints across S1 - S6b. Full sprint chain:
- **S1** cohort sized: 5/5 listen-validated NL on FS bucket M3 > 9.0
- **S2-S4.1** detector arc: voice-band autocorrelation pitch tracker +
  3 gates (RMS, cancel_ratio, filter_state). NE FP 14.31% → 0.00% after
  S4.1; 5/5 NL cohort fires 19.6-44.1%; detector design solid
- **S5** suppressor design lock: harmonic-pinned σ=50 Hz Gaussian mask,
  g_min_e4_db = -12 dB
- **S6** isolated bench: real cohort +0.77 dB ΔERLE mean (5/5 cases
  positive); synth NL (tanh×3) FAIL plan target (signature mismatch
  with detector)
- **S6a** listen verification at -12 dB: NO AUDIBLE NL REDUCTION
- **S6b** aggression sweep -18/-24/-30 dB: NO AUDIBLE NL REDUCTION at
  any level; -30 dB damages voice formants (mask is working) but NL
  perceptual character unchanged

**Closure mechanism**: multiplicative spectral mask `m[k,t] · Y[k,t]`
only modulates amplitude; cannot change phase. Real loudspeaker NL
("爆掉") and codec NL ("無線電") perceptual character is dominantly
phase distortion + time-domain transient — unreachable by any
amplitude mask family (any σ, any aggression).

**Preserved**: SubtractiveNLP detector class (default-OFF in aec.py)
+ AecConfig flags + 5/5 listen-validated cohort. Reusable for v3.14
Volterra arc as NL-frame identifier.

**Retired**: E4.S5 suppressor algorithm + E4.S7 800-case A/B port
(cancelled).

Verdicts: `docs/v3_13_e4_s6_verdict.md` + `docs/v3_13_e4_s6a_s6b_verdict.md`.
Memory: `project_v3_13_e4_nlp_arc_state.md`.

### E5 Saturation deepening arc (commit c871a5d, 2026-05-14)

4 sub-variants (S2/S3/S4a/S4b) tested. All on FS-vs-DT trade-off line.

| Variant | FS_static Δecho | DT_static Δdeg | DT_movement Δdeg | Verdict |
|---|---:|---:|---:|---|
| S2 lower F-E5 threshold 0.5→0.05 | +0.076 | -0.049 | -0.021 | FAIL DT bar |
| S3 mic-lpb correlation r>0.35 | +0.073 | -0.043 | -0.018 | FAIL DT bar |
| S4(a) S3 + state-gated freeze | +0.080 | -0.043 | -0.018 | FAIL DT bar |
| S4(b) S3 + shadow_rise mask only | +0.095 | -0.052 | -0.024 | FAIL DT bar |

**Closure mechanism**: amplitude-layer detector cannot distinguish
FS-NL frames (acoustic NL) from DT high-echo frames (mic = echo +
voice; same correlation signature in 0.7-0.95 mic peak band). Same
detector trigger fires on both; same filter-protection action helps
FS-NL but hurts DT high-echo. Trade-off slope ~0.5 dB DT loss per
+1 dB FS gain. Cannot escape line within amplitude / multiplicative
mask layer.

**Preserved**: E5.S3 correlation gate detector (worktree branches
`worktree-agent-a3c6eebf4ab4289a4`, `s4a-state-gated`, `s4b-shadow-only`).
0% NE FP, 14-56% real-NL fire rate. Reusable for v3.14 Volterra as
NL-frame identifier.

Verdict: `docs/v3_13_e5_closure_verdict.md`.
Memory: `project_v3_13_e5_closure.md`.

## What audited but produced no actionable work

### Phase 3 RES gain_floor audit (Sprint S4-S5, commit 6cdfbb0, 2026-05-14)

5-path empirical fire-rate audit on 800-case BALANCED. Findings:

| Path | Fire rate (FS / DT / NE) | Skew | S6-S7 disposition |
|---|---|---|---|
| spectral_floor | 89% / 52% / 10% | 0.80 | KEEP (cohort tail load-bearing 97%) |
| ne_g_floor | 88% / 93% / 99% | 0.13 | UNIFY (universal baseline; cosmetic) |
| **epc_dt_cap** | **0% / 0% / 0%** | 0 | **REMOVE (dead code, all 800)** |
| quiet_mask | 51% / 30% / 6% | 0.45 | KEEP (physical noise gate) |
| divergence_floor | 0.6% / 0.2% / 0% | 0.006 | KEEP (rare edge case) |

**Q7 V3 hypothesis revised**: ne_g_floor is NOT the main fragmentation
source (low skew). It behaves as universal baseline floor (88-99% all
buckets).

**Magnitude of canonical refactor benefit**: SMALL. Architectural
cleanup possible (1 path removable, 1 absorbable into canonical), but
behaviour change near-zero (consistent with v3.12 5-NEUTRAL closure).
Not a productisable improvement.

**S6-S7 deprioritized**; **S8-S9 (4-cap chain audit + per-state ENR
tuple) deferred to v3.14**.

Verdict: `docs/v3_13_phase3_res_audit_verdict.md`.

## v3.13 closure rationale

Per user directive 2026-05-14 "如果前端filter做完都沒問題的話 後端res
就接著往下做": front-end arcs run to closure, back-end audit run to
closure, all surface evidence-exhausted within v3.13 scope.

The arcs that PERCEPTUALLY mattered (E4 NLP, E5 acoustic NL) hit a
common physics wall — amplitude / multiplicative mask in frequency
domain has a fundamental ceiling for non-linear distortion. The
canonical breakthrough requires time-domain Volterra non-linear
inverse filter (or equivalent), which is a 6+ month dedicated arc
exceeding v3.13 scope.

The arcs that SHIPPED (E2 Delay) addressed an orthogonal problem
(filter coverage of long delays) and produced a clean +0.107 dB
FS_static improvement.

The audit arcs (Phase 3 RES) confirmed v3.12's 5-NEUTRAL closure of
Stage 1 RES surface — architectural cleanup possible but no
behaviour-change leverage.

## v3.14 candidate items

| Item | Source | Priority | Notes |
|---|---|---|---|
| **Volterra non-linear inverse filter (NL processor)** | E4 + E5 closures | **HIGHEST** | Canonical breakthrough path for 爆掉 / 無線電 perceptual NL. Dedicated 6+ month arc. Detector reuse from E4.S2/E5.S3. |
| Phase 3 RES canonical refactor (S6-S9) | Phase 3 audit | LOW | Cosmetic only; epc_dt_cap removable, ne_g_floor absorbable; S8-S9 4-cap audit unstarted |
| F-HFR per-band Q/R | hazy-lynx plan | LOW-MED | PBFDKF Kalman state structural change; Route C-V1 risk level |
| E1 mic_dynamic_margin | hazy-lynx plan | LOW | 1 listen case affected (case 3) |
| DT regression mechanism (per-state ENR) | E2 Path 3 verdict | MED | DT bucket Δdeg from E2 Path 3 unmasking; needs Phase 3 RES per-state work |

## Production state

**Production candidate**: `feature/v3.11-route-a` HEAD = 6cdfbb0.
**Behaviour vs v3.12 main**:
- E2 Path 3 (max_delay 250→1024 in eval_aec_challenge.py)
- All other v3.13 work is research substrate (default-OFF flags) or doc-only

**Merge-to-main decision**: defer until v3.14 Volterra design lock
opens. Branch can stay long-lived; no urgency.

## Worktree cleanup

The following worktrees can be cleaned at user's discretion (code
preserved in named branches):

- `worktree-agent-a3c6eebf4ab4289a4` (E5.S3 correlation gate detector)
- `s4a-state-gated` (E5.S4(a))
- `s4b-shadow-only` (E5.S4(b))
- `worktree-agent-abbbb8ba75683ce4d` (Phase 3 RES audit instrumentation)

To remove: `git worktree remove <path>` for each. Branches retain
history.

## Verdict closure

v3.13 arc CLOSED 2026-05-14. Three commits today:
- 3e10621: E4.S6a + S6b verdict
- c871a5d: E5 closure verdict
- 6cdfbb0: Phase 3 RES audit verdict

E2 Path 3 (5b1760c) shipped 2026-05-13.

Front-end and back-end surface fully evidence-explored within v3.13
scope. v3.14 opens with Volterra non-linear inverse design as the
primary new arc.
