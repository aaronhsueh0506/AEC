# v3.18 Phase A.1 — Shadow NLMS conversion (design lock, doc-only) — 2026-05-16

**Status**: A.1 DESIGN SPRINT — doc only, no code edits.
**Branch**: `feature/v3.18-aec3-fetch` (HEAD `26ae39f`).
**Authorisation scope**: A.1 only. **A.2 requires user re-auth** per
[v3_18_plan_revision_2026_05_15.md §11](v3_18_plan_revision_2026_05_15.md#L244).

## 0. One-paragraph framing

AEC3 ships a Kalman-refined / NLMS-coarse dual filter; we ship Kalman
main + Q-scaled Kalman shadow. In the high-signal regime the steady-state
Kalman gain converges to `1/X*` regardless of Q, so the Q×3.5 shadow
follows main's update direction — that is why v3.15 Arc M (V1/V2/V3) /
Arc G / Arc F / Arc T S2 all closed with the same shadow-correlation
failure mode. Phase A converts shadow → PBFDAF (NLMS) to restore the
structural orthogonality AEC3 relies on, while keeping
PathChangeRegimeHandler (which is **load-bearing** on the cohort tail
per [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md) /
[docs/p52_a0_postmortem.md](p52_a0_postmortem.md)) intact.

## 1. Audit answer — current code state (D.0 refresh)

### 1.1 Class inheritance + construction site

| Item | Location | State |
|---|---|---|
| `PBFDAF` base (NLMS) | [aec.py:1679](../python/aec.py#L1679) | ✓ exists, full NLMS pipeline |
| `PBFDKF(PBFDAF)` (Kalman extension) | [aec.py:1836](../python/aec.py#L1836) | ✓ exists |
| `PBFDAF.copy_weights_from(src)` | [aec.py:1832](../python/aec.py#L1832) | `self.W[:] = src.W` (W only) |
| `PBFDKF.copy_weights_from(src)` | [aec.py:2029](../python/aec.py#L2029) | `self.W[:] = src.W` (W only — P/Q/R intentionally not copied) |
| Shadow construction site | [aec.py:6200-6216](../python/aec.py#L6200) | Uses `FilterClass` (same as main) → `PBFDKF` in BALANCED |
| `enable_shadow` / `shadow_mu_ratio` / `shadow_q_ratio` | [aec.py:204-211](../python/aec.py#L204) | ✓ ready as scaffolding |
| `_arc_m_q_boost(filt)` | [aec.py:5732](../python/aec.py#L5732) | Sets `filt.Q = Q_high.copy()` (PBFDKF-only attr) |
| `filt._p_max_override` / `_p_floor_beta` | [aec.py:1885-1890](../python/aec.py#L1885) | PBFDKF-only attrs (delay_change full reset path) |

### 1.2 Shadow consumers that must keep working

| Consumer | Reads from shadow | Today's semantics |
|---|---|---|
| `_shadow_advantage` / DoubleTalkAnalyzer | `shadow_err_smooth` vs `main_err_smooth` | error-energy ratio (filter-type agnostic) |
| `PathChangeRegimeHandler.update()` | `shadow_err_smooth`, `main_err_smooth`, `epc_active`, etc. | error-energy ratio + counters |
| `_dt_from_shadow` (DTD via shadow) | `shadow_err_smooth` | error-energy ratio |
| EPC `shadow_rise` source | shadow error rising while main error rising | error-energy ratio |
| `shadow_filter.copy_weights_from(main)` (reverse_copy) | W only | `self.W[:] = src.W` ✓ filter-type agnostic |
| Arc T cohort-tail proxy | doesn't read shadow directly | — |
| Arc S-orth.A (decoupled shadow Kalman state, v3.14) | shadow's own `P` / `Q` / `error_psd` | **PBFDKF-only** ⚠️ |

**Verdict**: all *signal* consumers read error-energy, filter-type
agnostic. Only Arc S-orth.A reads PBFDKF-internal state — see §3.5 for
its handling.

### 1.3 What changes structurally

Only one line decides "Kalman or NLMS shadow":

```python
# aec.py:6204 (current)
self.shadow_filter = FilterClass(...)   # FilterClass = PBFDKF when use_kalman=True

# A.2 target (flag ON)
ShadowClass = PBFDAF if cfg.shadow_class_nlms else FilterClass
self.shadow_filter = ShadowClass(...)
```

Everything else is downstream consequences of that swap.

## 2. Design lock — flag, kwargs, defaults

### 2.1 New `AecConfig` field

```python
# v3.18 Phase A.2 — shadow filter adaptation class.
# False (default): shadow uses same class as main (PBFDKF in BALANCED).
# True: shadow uses PBFDAF (NLMS), AEC3-aligned. Restores orthogonality
#       between shadow and main update direction in high-signal regime
#       where Kalman gain saturates (X·P·X* >> R → K → 1/X*).
shadow_class_nlms: bool = False

# NLMS step-size for shadow. Only consumed when shadow_class_nlms=True.
# AEC3 coarse-filter μ default is 0.5. We start at 0.5 with A.5 grid.
shadow_mu_nlms: float = 0.5
```

**Naming check** (per `feedback_no_version_in_var_names.md`): both names
are mechanism descriptors (`_class_nlms` / `_mu_nlms`), not iteration
suffixes. ✓

### 2.2 Construction (one new branch in `AEC.__init__`)

```python
# v3.18 Phase A.2 — shadow class selection.
if self.config.shadow_class_nlms:
    ShadowClass = PBFDAF
    shadow_kwargs = dict(
        block_size=self.filter.block_size,
        n_partitions=self.filter.n_partitions,
        mu=self.config.shadow_mu_nlms,   # AEC3-aligned NLMS rate
        delta=self.config.delta,
        hop_size=self.filter.hop_size,
    )
else:
    ShadowClass = FilterClass
    shadow_kwargs = dict(
        block_size=self.filter.block_size,
        n_partitions=self.filter.n_partitions,
        mu=self.config.mu * self.config.shadow_mu_ratio,
        delta=self.config.delta,
        hop_size=self.filter.hop_size,
    )
self.shadow_filter = ShadowClass(**shadow_kwargs)
self.shadow_filter.enable_td_constraint = self.config.enable_td_constraint
# Q-scaling only meaningful for PBFDKF shadow.
if isinstance(self.shadow_filter, PBFDKF):
    self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
    self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
    self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()
```

Note: legacy branch is byte-identical to current `aec.py:6204-6216`.
Flag-OFF byte-equal is enforced by construction.

### 2.3 PBFDKF-only attribute callsites (must guard with `isinstance` or `hasattr`)

The following sites currently assume shadow is PBFDKF; under flag ON
the shadow is PBFDAF and these attrs do not exist:

| Site | Today | Required guard |
|---|---|---|
| `_arc_m_q_boost(self.shadow_filter)` | `filt.Q = Q_high.copy()` | Already gated by `hasattr(filt, 'Q')` ✓ |
| `_handle_delay_change_full` (F.2) — `filt._p_max_override = 1.0`, etc. | sets PBFDKF P-override attrs on both filters | wrap each in `if isinstance(filt, PBFDKF):` |
| Arc S-orth.A — shadow's own Kalman state read | reads `shadow_filter.P` / `_error_psd` | wrap reader in `isinstance(self.shadow_filter, PBFDKF)` — when shadow is NLMS, Arc S signals fall back to legacy path |

All guards are conservative (preserve PBFDKF path; bypass for NLMS).
None of the guards change PBFDKF-shadow behaviour ⇒ flag-OFF byte-equal
holds.

## 3. Design decisions

### 3.1 NLMS step-size `shadow_mu_nlms`

AEC3 coarse-filter μ = 0.5 (single global value; refined uses Kalman
P-based adaptation). Grid for A.5 tuning: **{0.3, 0.5, 0.7}** on
0I0XMl3M + qNvSMyU smoke. Default 0.5 baseline (AEC3-matched). A.5
selects per cohort-tail Δecho proxy.

Rationale for AEC3 μ being safe at 0.5: NLMS step is normalised by
`(power + delta) × n_partitions`, so the effective per-bin update
magnitude is dimensionless — μ=0.5 means 50% innovation injection per
block. PBFDAF's existing `_update_weights` ([aec.py:1807](../python/aec.py#L1807))
already normalises this way.

### 3.2 Copy gate semantics — flag-ON skips `reverse_copy` (CORRECTED 2026-05-16)

**Original §3.2 reasoning had two factual errors**, flagged by user
2026-05-16:

1. AEC3 has **no** W-copy mechanism (neither refined→coarse nor
   coarse→refined). The two filters run independently; coarse signals
   `HandleEchoPathChange` to refined which boosts the refined
   *step-size* — no W transfer.
2. Our `reverse_copy` is **shadow ← main** (one direction), not
   bidirectional. Triggers when main_err < shadow_err × threshold
   ([aec.py:5535](../python/aec.py#L5535)), then
   [aec.py:7476](../python/aec.py#L7476) copies main's W into shadow.
   This rescues a Kalman shadow stuck in a bad basin (PBFDKF P-memory
   issue), preventing it from emitting wrong `shadow_advantage`.

The qNvSMyU cohort tail defence travels a **different** regime-handler
path: shadow << main (shadow tracking better than main) → `pause_main`
+ `boost_q` ([aec.py:5497-5499](../python/aec.py#L5497)). That path
does **not** involve `reverse_copy`. Both `pause_main` and `boost_q`
are filter-type agnostic and stay intact under NLMS shadow.

**Corrected decision (AEC3-aligned)**: when `shadow_class_nlms=True`,
**skip the `reverse_copy` action entirely**. NLMS shadow has no
P-memory to get stuck; it re-adapts from its own residual stream.
Skipping copy under flag-ON mirrors AEC3 design exactly.

Implementation (A.3 wiring): at [aec.py:7474](../python/aec.py#L7474)
guard the reverse_copy block with `if not config.shadow_class_nlms`.
The `boost_q` / `pause_main` paths and reverse_copy_p_reset branch
(NLMS shadow doesn't enter it anyway via `hasattr(filt, 'P')` —
already filter-type-safe) are unchanged.

Flag-OFF byte-equal: preserved (the new guard adds a False branch only
when flag is True).

### 3.3 Post-copy hangover — obsolete under corrected §3.2 (CORRECTED 2026-05-16)

Original §3.3 required a post-copy hangover (`_shadow_update_holdoff_remaining`
counter passing `mu_scale=0` for 30 frames) to prevent NLMS-shadow
oscillation after `shadow ← main` W copy. With §3.2 corrected so
**flag-ON skips reverse_copy entirely**, there is no copy event for
the hangover to gate — the hangover concept becomes moot.

What AEC3 actually has as "hangover": a guard preventing
`refined_gains_->HandleEchoPathChange()` from re-firing too rapidly
(would compound the step-size boost into divergence). Our equivalent:
`PathChangeRegimeHandler._pause_resume` countdown
([aec.py:5500-5504](../python/aec.py#L5500)) which holds `main_paused`
for `epc_hangover` frames after `boost_q` fires, naturally preventing
back-to-back boost_q. **This already exists and is filter-type
agnostic — no change needed.**

A.4 is now **descope** to a confirmation sprint: verify
`_pause_resume` countdown still works as expected under NLMS shadow
(trace-only, no code edit). Sprint A.4 in §8 retitled accordingly.

### 3.4 PathChangeRegimeHandler — needs retune, scope deferred to A.6

The handler's thresholds ([aec.py:5468-5479](../python/aec.py#L5468)) were
calibrated against PBFDKF-shadow error magnitudes:
- `shadow_copy_threshold` = 0.65 (shadow_err < main_err × 0.65 → copy)
- `_copy_err_baseline` decay = 0.995 / 0.005 EMA
- `shadow_copy_hysteresis` = 3 frames

NLMS shadow steady-state error differs from Kalman shadow steady-state
error by a factor that depends on `mu_nlms` vs `Q_high` × signal
condition. Without retune, fire-rate of regime handler will drift.

**Decision for A.1**: do not retune thresholds in A.2-A.5. Land NLMS
shadow first, run A.6 trace-only A/B (capture per-frame
`shadow_err_smooth / main_err_smooth` distributions under both flags),
then propose threshold adjustments. **A.6 outputs another doc, not a
code edit yet** — threshold edits go in A.6.1.

### 3.5 Arc S-orth.A interaction

Arc S-orth.A (v3.14 shipped — decoupled shadow Kalman state) gives RES
access to shadow's `P`, `_error_psd`, etc. When shadow is NLMS those
attrs don't exist.

**Decision**: gate Arc S-orth.A reader with
`isinstance(self.shadow_filter, PBFDKF)`. When False, Arc S-orth.A
signals fall back to scalar / legacy substitute (already implemented as
the v3.14 default-OFF branch). This means when `shadow_class_nlms=True`,
**Arc S-orth.A is implicitly disabled**. Document in A.7 closeout — the
60-case bench captures combined (NLMS + Arc-S-off) vs (PBFDKF + Arc-S-on)
delta; if NLMS wins net, accept Arc S as cost-of-port; if not, kill A
per §0.4.

### 3.6 Per-preset enablement plan

A.2 lands flag default OFF globally. A.7 60-case + A.8 800-case run
under BALANCED only. If A.8 ships, only `BALANCED.shadow_class_nlms=True`;
MILD/SOFT/AGGRESSIVE/MAXIMUM stay False until Phase E substrate
promotion (post Phase B + C).

## 4. Reverse-evidence risk register (per §11)

| # | Risk | Evidence | Mitigation in A.1-A.8 |
|---|---|---|---|
| R1 | v3.14 Arc S-orth.A shipped decoupled state — coupling problem may be in **signal interpretation**, not filter type | Arc S-orth.A shipped 2026-05-14 yet v3.15 Arc M/G/F/T still closed (post-v3.14 baseline) | A.7 captures `shadow_advantage` distribution under NLMS — if NLMS shadow still shows >0.7 correlation with main error envelope on EPC events, coupling is signal-side, **kill** A per §0.4 |
| R2 | qNvSMyU cohort tail defence — NLMS shadow may worsen the −0.56 dB outlier | P52 audit found PathChangeRegimeHandler load-bearing; NLMS without Kalman P doesn't give same defence shape | A.7 hard bar adds: **no new qNvSMyU outlier worse than P52 baseline floor (−0.56 dB)**. A.7 fail → kill A per §0.4 |
| R3 | F closeout points to DelayEst as real bottleneck, not shadow filter | F.4 60-case showed FS_static −0.048 driven by missing delay_change events (GCC-PHAT gap); Phase A doesn't add a delay detector | A.7 explicitly reports cohort-tail Δecho separately for sx6mxKBQ-class samples (DelayEst-relevant) vs qNvSMyU-class (path-stationary outlier). If only the latter improves, NLMS shadow is solving a different problem than F was trying to solve — acceptable but narrow win |
| R4 | D + F pattern (AEC3-port arcs hit our co-tuning wall) | Both AEC3-aligned ports closed CANNOT SHIP on the same DT-FS Pareto wall | A.7 60-case bucket bar enforced: any of FS Δecho < −0.010 / DT Δdeg < −0.005 / NE Δdeg < −0.005 → kill per §0.4 |
| R5 | NLMS post-copy cycle/divergence (Gap #4) | AEC3 has explicit post-copy hangover; without it shadow may oscillate | §3.3 fold-in mandatory; A.5 smoke includes 0I0XMl3M (EPC-heavy) to surface cycles |
| R6 | μ_nlms wrongly tuned — too low ⇒ no orthogonal signal, too high ⇒ shadow itself diverges | NLMS step-size unfamiliar in this codebase | A.5 grid {0.3, 0.5, 0.7}; bench wall = qNvSMyU + 0I0XMl3M Δerle; reject grid if any rate gives shadow_err > main_err × 5.0 sustained |

## 5. A.1 design confidence read

Per §11 kill criterion: **if design confidence < 60% → close before
A.2 and pivot to Phase B (lower-risk, substrate-ready)**.

**Reading: 65%** (proceed, but narrowly).

Components:
- (+) Construction change is genuinely 1-line; substrate (PBFDAF class,
  copy_weights_from W-only semantics, regime handler's filter-type-
  agnostic inputs) is all in place. **Implementation risk: low.**
- (+) AEC3 design choice is structural, not just engineering convenience.
  K_ratio math + 4 v3.15 closures with same root cause is convergent
  evidence. **Mechanism diagnosis: high confidence.**
- (−) Reverse evidence R1 (Arc S-orth.A shipped + still failed) is the
  strongest counter-signal; it suggests shadow correlation may be
  upstream of "Kalman vs NLMS" — at the signal-interpretation layer.
  We mitigate with A.7 distribution capture, but cannot rule out before
  A.7.
- (−) Reverse evidence R2 (qNvSMyU defence) is asymmetric risk — gain
  comes from middle-cohort cases, loss comes from cohort tail. AECMOS
  geomean averages this out, but tail audibility is what we shipped
  PathChangeRegimeHandler to protect.
- (−) F closeout (R3) sets prior that AEC3-port arcs are hitting a
  pipeline-co-tuning wall, not a mechanism gap. Phase A is the next
  AEC3 port in sequence; same wall may apply.

Decision: confidence sits **above** the 60% kill threshold but
**below** the threshold for unconditional A.8 ship. We proceed to A.2
with the explicit understanding that A.7 60-case is the next decision
point; A.8 800-case is not authorised by this design — it requires a
fresh user §0.7 review after A.7.

## 6. Updated Phase A hard bar (anchored from §11)

### A.7 60-case (must pass to authorise A.8 800-case)
- **PRIMARY**: cohort tail (qNvSMyU) Δecho ≥ **+0.030 dB**
- **HARD GUARD**: qNvSMyU outlier no worse than P52 baseline (−0.56 dB floor)
- DT_static Δdeg ≥ −0.005
- DT_movement Δdeg ≥ −0.005
- FS_static Δecho ≥ −0.010
- FS_movement Δecho ≥ −0.010
- NE Δdeg ≥ −0.005

Worst-case (single sample) bar: no sample Δecho worse than **−0.05 dB**
(adopted from F closeout sx6mxKBQ-class learning).

### A.8 800-case (productisation, requires post-A.7 user re-auth)
- Same per-bucket bars as A.7, evaluated on full 800-case
- BALANCED preset only; other presets stay default OFF
- nores listen verification on qNvSMyU + 0I0XMl3M + xrtntuju 5-clip
- Cohort tail catastrophe defence verified: no new −0.05 dB outlier on
  full cohort

### Kill criterion (per §0.4)
Any failure of A.7 PRIMARY OR HARD GUARD → close A per §0.4.
Substrate (`shadow_class_nlms` flag, PBFDAF construction branch,
`shadow_mu_nlms` config, isinstance guards on PBFDKF-only sites)
retained as default-OFF for future re-enable when DelayEst /
co-tuning blockers lift.

## 7. A.1 → A.2 hand-off checklist

A.1 design complete when:
- [x] D.0 audit refreshed (§1 above)
- [x] Construction change designed (§2)
- [x] Hangover / copy gate / regime handler interactions classified (§3)
- [x] Reverse-evidence risk register (§4) with explicit mitigations
- [x] Confidence read with kill threshold (§5)
- [x] Hard bar updated for A.7/A.8 (§6)

A.2 commit gate (user must re-authorise):
- [ ] User confirms §5 confidence read is acceptable risk
- [ ] User confirms §6 hard bar is the right adjudication line
- [ ] User authorises code edits in flag-OFF byte-equal mode
- [ ] A.2 deliverable: flag + construction branch only, 5-case byte-equal
  flag-OFF md5 PASS, no flag-ON bench

## 8. A.2-A.8 sprint preview (for re-auth context)

| Sprint | Action | Output |
|---|---|---|
| A.2 | Add `shadow_class_nlms` / `shadow_mu_nlms` config; construction branch; isinstance guards on `_handle_delay_change_full` / Arc S-orth.A reader | 5-case byte-equal flag-OFF md5 PASS |
| A.3 | Copy gate (CORRECTED): under flag-ON, skip `reverse_copy` to mirror AEC3 (no W transfer between independent filters). 5-case byte-equal flag-OFF re-verify | byte-equal flag-OFF md5 PASS + trace confirms boost_q/pause_main still fire flag-ON |
| A.4 | (descope) confirm `_pause_resume` boost_q hangover still works under NLMS shadow (trace-only; no code edit) | trace inspection |
| A.5 | NLMS rate grid {0.3, 0.5, 0.7} on 0I0XMl3M + qNvSMyU smoke | rate selection doc |
| A.6 | PathChangeRegimeHandler trace-only A/B (capture distributions) | retune proposal doc (A.6.1 separate sprint) |
| A.7 | 60-case AECMOS + nores listen + per-§6 hard bar adjudication | A.7 verdict doc |
| A.8 | 800-case + ship gate (requires post-A.7 user re-auth) | A.8 verdict + §0.7 merge package |

## 9. Cross-references

- [docs/v3_18_plan_revision_2026_05_15.md §11](v3_18_plan_revision_2026_05_15.md#L244) — design-confidence framing + decision authorisation
- [docs/v3_18_f_closeout.md](v3_18_f_closeout.md) — predecessor closeout, motivates Phase A
- [docs/v3_18_d_gamma_closeout.md](v3_18_d_gamma_closeout.md) — AEC3-port co-tuning wall evidence
- [docs/aec3_reference.md §6.1](aec3_reference.md#L750) — AEC3 shadow design + K_ratio math
- [docs/p52_phase_a_verdict.md](p52_phase_a_verdict.md) — PathChangeRegimeHandler post-mortem (load-bearing rationale)
- [docs/p52_a0_postmortem.md](p52_a0_postmortem.md) — qNvSMyU −0.56 dB outlier evidence
- [~/.claude/memory/project_v3_15_closeout_outcome.md](~/.claude/memory/project_v3_15_closeout_outcome.md) — 4-arc shadow-correlation failure history
