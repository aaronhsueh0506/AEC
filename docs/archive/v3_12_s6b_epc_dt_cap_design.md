# S6b design lock — `epc_dt_cap` state-driven retarget

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a`
**Predecessor**: S6 verdict commit `a60768f` (`ne_g_floor` swap byte-equal,
rolled back to research substrate). See [v3_12_res_audit.md](v3_12_res_audit.md)
Addendum for the mechanism finding that retargets Phase 3B to `epc_dt_cap`.

This document is **design-only**. No code changes accompany this commit.
Implementation gated on user review.

---

## 1. Current behaviour

Two firing sites in `aec.py`:

- **Gate construction** ([aec.py:2710](../python/aec.py#L2710)):
  ```python
  epc_dt = epc_active and effective_dt > 0.35
  ```
- **Cap action** ([aec.py:2322-2324](../python/aec.py#L2322-L2324)):
  ```python
  if epc_dt:
      EPC_DT_GAIN_CAP = 0.85
      g = np.minimum(g, EPC_DT_GAIN_CAP)
  ```

`effective_dt = max(float(dt_for_fs), float(shadow_dt))`
([aec.py:2705](../python/aec.py#L2705)). `dt_for_fs` ingests `(1-coh²)`-style
evidence and `shadow_dt` reflects shadow-vs-main energy ratio, both of which
can spike in FS post-cancellation per Q7 V3 (low coh² → false NE flag;
shadow lagging main → spurious shadow_dt).

## 2. Why this is the real `effective_dt` leak path

S6 verdict established that `ne_g_floor` is gated downstream by
`(1 - fs_confidence)`, where `fs_confidence ≈ 1` in FS, so its evidence
swap is byte-equal. `epc_dt_cap` has **no** `(1 - fs_confidence)` neutraliser:
the cap fires whenever `epc_active AND effective_dt > 0.35` regardless of
whether the frame is truly DT or FS-post-cancellation pretending to be DT.

The cap value `0.85` corresponds to ~1.4 dB extra echo suppression. If
applied falsely in FS, the gain is force-capped and prevents the Wiener /
softgate path from passing FS bins through at ≈ 1.0 — i.e. residual echo
gets over-protected through misclassification.

Path 3 of [v3_12_res_audit.md](v3_12_res_audit.md) already lists this as
**EVIDENCE PATCH — fold into per-state ENR tuple in S8-S9**. S6b promotes
it from S8-S9 architecture into S6-S7 mechanism, with cap action unchanged.

## 3. Proposed change

### 3.1 New gate (Option A, recommended)

Replace the scalar gate with `filter_state` membership:

```python
# OLD
epc_dt = epc_active and effective_dt > 0.35

# NEW (when flag ON)
epc_dt = filter_state in {'diverged', 'suspicious_dt'}
```

Cap action `g = min(g, 0.85)` and constant `EPC_DT_GAIN_CAP = 0.85`
**unchanged**. Per P58 anti-trap: only the gate condition changes.

### 3.2 Why this set of states

The existing 6-state enum from `FilterConvergenceAnalyzer`
([aec.py:6102-6122](../python/aec.py#L6102-L6122)):

| State | Should cap? | Reason |
|---|---|---|
| `idle` | No | No far activity → cap meaningless |
| `startup` | No (debatable — see 3.4) | New stream warmup; ENR conservative path already handles |
| `diverged` | **Yes** | Filter divergent → residual cannot be trusted, force suppression |
| `suspicious_dt` | **Yes** | Main_err jumped + shadow leads → likely path change in DT, mirror current `epc_active AND DT` |
| `refined_usable` | No | Filter converged + ERLE healthy → Wiener gain reliable |
| `coarse_learning` | No (debatable — see 3.4) | Default "not yet converged" state; over-firing here regresses long warmup |

Selecting `{diverged, suspicious_dt}` mirrors the **intent** of the
existing scalar gate (filter unreliable + DT-like behaviour) but sources
the signal from `FilterConvergenceAnalyzer` instead of
`epc_active AND effective_dt`. This decouples the cap from `effective_dt`
saturation, which is the Q7 V3 fix.

### 3.3 Two safety variants (Option B / C — for fall-back if A fails)

**Option B — OR'd (super-safe, additive)**:
```python
epc_dt = (filter_state in {'diverged', 'suspicious_dt'}) \
         or (epc_active and effective_dt > 0.35)
```
Adds new false-positives but never removes existing protection. Does NOT
fix the Q7 V3 leak (legacy gate still fires on FS-saturated `effective_dt`).
Treat as roll-forward option if A regresses.

**Option C — AND'd (restrictive)**:
```python
epc_dt = (filter_state in {'diverged', 'suspicious_dt'}) \
         and epc_active
```
Requires both state-driven AND EPC-active. Cap fires less than current.
Use if A over-fires and we want a more selective gate.

### 3.4 What about `startup` / `coarse_learning`?

These states are "filter not yet converged". The 0.85 cap might help echo
suppression during warmup. However:

- Including them would make the cap fire on every new stream's first
  20-50 frames — affecting NE bucket and FS_static when filter is still
  learning. Risk of NE Δdeg regression.
- The existing scalar gate did NOT cap during pure warmup (no EPC + low DT).
- Including them is a scope expansion outside Q7 V3.

**Recommendation**: NOT include in S6b. If S6b passes and we want to
extend, add as a separate Sprint S6c with its own 800-case A/B.

## 4. Anti-trap audit

| Trap | Mechanism | How S6b avoids it |
|---|---|---|
| **P50** (FS Δecho −1.328 from `nearend_protect_dt`) | Single `effective_dt` trigger fired in both DT and FS, raising NE floor in FS | New gate is filter_state membership, removes `effective_dt` from the gate. FS frames with high `effective_dt` (Q7 V3 saturation) no longer fire the cap as long as filter_state is `refined_usable` (which it should be in genuine FS). |
| **P52** (cohort tail −0.56 from PathChangeRegimeHandler retirement) | `qNvSMyU…` lost catastrophe defence | PathChangeRegimeHandler untouched. `filter_state` enum comes from `FilterConvergenceAnalyzer` ([aec.py:6090+](../python/aec.py#L6090)), which is downstream of the regime handler — pause_main / boost_q / reverse_copy decisions remain authoritative. Cap is downstream consumer, not blocker. |
| **P55** (DT-FS +7.01 from Enzner-Vary Wiener) | New Wiener discriminator was weaker than legacy ENR | Wiener / softgate / ENR path unchanged. Only the cap **gate** changes. Wiener gain is computed and applied **before** `epc_dt_cap` so the new gate cannot mis-shape Wiener output. |
| **P58** (FS Δecho −0.674 from 4-cap retire) | Cap action removed entirely → no minimum suppression | Cap action preserved: 0.85 retained, applied only when gate fires. Anti-loophole: cap value is `EPC_DT_GAIN_CAP = 0.85` constant, NOT a function of state. |

## 5. Verification plan

### 5.1 Flag wiring

```python
# AecConfig
res_state_driven_epc_dt_cap: bool = False  # v3.12 S6b

# ResFilter.__init__
state_driven_epc_dt_cap: bool = False  # → self._state_driven_epc_dt_cap

# ResFilter.process() at line ~2710:
if self._state_driven_epc_dt_cap and self._consume_filter_state:
    epc_dt = filter_state in {'diverged', 'suspicious_dt'}
else:
    epc_dt = epc_active and effective_dt > 0.35  # legacy
```

Note `AND self._consume_filter_state`: the new gate depends on Phase 2's
C2 wiring being ON. This is intentional — Phase 3 mechanism flips
shouldn't be possible without Phase 2 wiring being live.

### 5.2 Step 1 — byte-equal scaffold

`res_state_driven_epc_dt_cap=False` (default OFF) + run 100-case sample
through `run_one_case.py` or 800-case j4 → expect 0.000 Δ vs v3.11.2.
Sanity that the code-path branch is inert when flag is OFF.

### 5.3 Step 2 — fire-rate audit (without flag flip)

Add temporary diagnostic counter (deleted after audit):
- count of frames where legacy gate fires (`epc_active AND effective_dt > 0.35`)
- count of frames where new gate fires (`filter_state in {diverged, suspicious_dt}`)
- count of overlap (both fire)
- per-bucket breakdown

Run 800-case j4 once, dump counters, **delete diagnostic counter code
before flag flip**. Decision:
- If new gate fires < 10% of legacy gate's rate → new gate is too narrow,
  consider Option B (OR) or expanding state set.
- If new gate fires > 5× legacy gate's rate → too broad, narrow with
  Option C (AND `epc_active`) or remove `coarse_learning` if it was included.
- If new gate fires within 0.5-2× of legacy rate → A is calibrated.

### 5.4 Step 3 — flag-ON A/B

Promote `res_state_driven_epc_dt_cap=True` to BALANCED preset, run
800-case j4, compare to v3.11.2 baseline (`results/v3_12_s3_candidate/scores.json`):

**Hard abort thresholds** (any single failure → rollback BALANCED flag,
keep code default-OFF, file verdict):
- FS_static Δecho < -0.02
- FS_movement Δecho < -0.02
- DT_static Δdeg < -0.005
- DT_movement Δdeg < -0.005
- NE Δdeg < -0.005
- Cohort tail `qNvSMyU…` Δecho < -0.05

**Positive criteria** (any one to qualify as net improvement):
- FS_static or FS_movement Δecho ≥ +0.01 (echo leakage relief)
- DT bucket Δdeg ≥ +0.005 (NE preservation when state correctly identifies DT)

If neither hard-abort nor positive — file as NEUTRAL (similar to S6),
roll back, retarget further.

### 5.5 Step 4 — listen check

- `xrtntuju` 5-clip DT regression (NE preservation)
- `qNvSMyU` cohort tail (catastrophe defence not weakened)

## 6. Per-state ENR tuple (deferred to S8-S9)

S6b only swaps the `epc_dt_cap` **gate**. The full per-state ENR tuple
work from the audit doc Section 3 (gain_cap / divergence_cap / ENR
aggression per state) stays in S8-S9 scope. S6b is the bridge between
Phase 2 wiring (S3) and Phase 3C policy (S8-S9): proves filter_state
can drive an existing cap gate without architectural perturbation.

## 7. Open questions for review

1. **State set**: is `{diverged, suspicious_dt}` right? Should we audit
   `coarse_learning` empirically before excluding it?
2. **`_consume_filter_state` dependency**: should S6b require it as
   pre-condition, or carry its own flag-OR-flag dependency? (Currently
   designed as required pre-condition.)
3. **Diagnostic counter**: keep behind a `_trace_*` flag for ongoing
   audit, or scrub entirely after Step 2? (My recommendation: scrub
   after audit so production code stays clean.)
4. **Fire-rate calibration target**: is the 0.5-2× legacy rate band
   too narrow / too wide? Should we sample qNvSMyU specifically?
5. **Option B fall-back**: if A regresses, do we want Option B as
   a separate verdict commit (additive safety) or call S6b CLOSED?

## 8. Risk summary

| Aspect | Assessment |
|---|---|
| LOE | 1 sprint (~1-2 days). Code change is ~20 lines including flag plumbing |
| Architectural risk | LOW. Same pattern as B6 (state-aware shadow_mu, S1) — gate swap, no new code path |
| P50/P52/P55/P58 trap risk | LOW per Section 4 audit |
| ROI uncertainty | MED — same shape as S6 (Phase 3B v1). Hypothesis is stronger than S6 because no downstream neutraliser, but Q7 V3 fire-rate is empirical-unknown |
| Rollback cost | LOW. Flag-gated, default-OFF if A fails |

---

## Appendix — Why not just delete `epc_dt_cap` entirely?

Tempting given P58 verdict shows 4-cap chain is load-bearing on FS but
P58 also showed retiring caps regresses output. The Q7 V3 thesis is
that the cap **action** is fine, the **gate** is broken (false fires in
FS, missed fires in true DT). Keeping the cap action and fixing the
gate is the minimum-risk lever.

If S6b passes, future work (Phase 3C / S8-S9) can refine the cap value
per-state (transient = 0.85, recovering = 0.95 ramp, etc.). Until then,
keep cap = 0.85 across the new gate set for byte-budget conservation.
