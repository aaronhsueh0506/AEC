# Mode/Action Path Review — κ-4 (v2.8.0)

Date: 2026-04-23. Code: `python/aec.py` v2.8.0, commit `7235499`.

## §1 Design intent vs. actual control flow

### Intended three-tier action hierarchy

| Condition | Intended action |
|---|---|
| Pre-convergence (filter_converged=False) | NO_ACTION or SOFT_ASSIST (gentle blend, no harm) |
| Post-convergence NORMAL mode (no EPC event) | NO_ACTION |
| Post-convergence + PATH_CHANGE_RECOVERY active | HARD_RECOVERY_COPY (full copy + Kalman reset) |

### Actual Fix E gate control flow (lines 3542–3572)

```
if sustain_count >= SUSTAIN_FRAMES:
    _can_hard_copy = filter_converged AND pc_recovery_mode AND not_saturating
    if _can_hard_copy:
        HARD_RECOVERY_COPY          ← only if ALL THREE guards true
    else:
        SOFT_ASSIST                 ← fires for EVERY OTHER CASE
    # both paths: session_count += 1, cooldown = 200, sustain_count = 0
```

**The `else` branch catches three distinct cases as a single fallback:**
1. Pre-convergence (filter_converged=False) — *intended: SOFT_ASSIST*
2. Post-convergence NORMAL mode (pc_recovery_mode=False) — *intended: NO_ACTION*
3. Post-convergence saturating (saturation≥0.3) — *intended: NO_ACTION*

Cases 2 and 3 should not fire SOFT_ASSIST but currently do.

---

## §2 Structural concern confirmations

### 2.1 SOFT_ASSIST scope too wide — CONFIRMED

Post-convergence NORMAL mode falls to SOFT_ASSIST via the `else` branch.  
This is the primary source of unnecessary interventions in stable FS scenarios
where shadow has a brief advantage but pc_recovery_mode was never entered.

**Impact**: SOFT_ASSIST fires in stable FS sessions where echo is already low,
partially misaligning the main filter with no corresponding recovery benefit.

### 2.2 PATH_CHANGE_RECOVERY entry — REQUIRES STATS

Entry conditions (lines 3776–3781):
```python
if (filter_converged
        and epc_level == 'small'
        and err_baseline_ratio >= 1.5
        and saturation < 0.3):
    pc_recovery_mode = True
    pc_recovery_hangover = 100
```

Quick 3-case test: **recovery active = 0.0% in ALL subsets.**
This suggests one of the four conditions is almost never jointly satisfied.
Most likely suspect: `epc_level == 'small'` requires the EPC detector to fire,
which requires `dt_signal < 0.3` (DT guard) and confirmed echo-path change.
In typical DT-movement and FS scenarios, EPC may be gating out correctly or
the threshold alignment may not produce 'small' level.

Full 800-case stats needed to confirm which condition is the bottleneck.

### 2.3 SOFT_ASSIST has no state handling — CONFIRMED

SOFT_ASSIST path (lines 3563–3566):
```python
self.filter.W[:] = (1-α)*W + α*shadow.W
self.main_err_smooth = (1-α)*main_err_smooth + α*shadow_err_smooth
```

Only `W` (filter weights) and `main_err_smooth` are updated.  
HARD_RECOVERY_COPY calls `soft_reset_after_copy()` which adjusts Kalman state (P/Q/R).  
SOFT_ASSIST leaves P/Q/R unchanged — Kalman state now inconsistent with new W.  
Effect: filter adapts from new starting W with stale P/Q/R; potential lag or tail artifact.

### 2.4 Shared session budget — CONFIRMED

Both SOFT_ASSIST and HARD_RECOVERY_COPY execute:
```python
self._dt_copy_cooldown = self._DT_COPY_COOLDOWN_FRAMES   # 200
self._dt_copy_session_count += 1
```
`_DT_COPY_MAX_SESSION = 3`. Three SOFT_ASSIST events during pre-convergence
exhaust the session budget entirely, making HARD_RECOVERY_COPY impossible
for the rest of the session even if pc_recovery_mode eventually activates.

**Quick 3-case test result**: budget exhausted = 100% of FS and DT files,
HARD_RECOVERY_COPY fires = 0 times. All 3 budget slots consumed by SOFT_ASSIST
before filter converges.

### 2.5 Division of labor — CONFIRMED

Mode, gate, floor, and budget are entangled in a single code block.
Current responsibility assignments:

- `pc_recovery_mode`: determines if hard copy is allowed (hard-gate)
- Fix E gate conditions: determines if any action fires (sustain, cooldown, cap, adv)
- `_dt_err_floor`: varies with both `pc_recovery_mode` and `filter_converged`
- `_dt_copy_session_count`: shared count for conceptually distinct actions

No clean separation between: "are we in a recovery state?" vs "what action tier
is authorized?" vs "has this action been used recently?".

---

## §3 Quick 3-case coverage stats (early signal)

From `--quick 3` run on farend_singletalk, doubletalk, nearend_singletalk:

| Subset | recovery active | SOFT_ASSIST/file | HARD_COPY/file | budget_exhausted |
|---|---:|---:|---:|---:|
| FS-static (1) | 0.0% | 3.0 | 0 | 100% |
| FS-movement (2) | 0.0% | 3.0 | 0 | 100% |
| DT-static (2) | 0.0% | 3.0 | 0 | 100% |
| DT-movement (1) | 0.0% | 3.0 | 0 | 100% |
| NE (3) | 0.0% | 0.0 | 0 | 0% |

**Key observations:**
1. `pc_recovery_mode` activates 0% of frames — PATH_CHANGE_RECOVERY never enters
2. Every FS/DT file gets exactly 3 SOFT_ASSIST events (the session cap)
3. Budget is 100% exhausted before any HARD_RECOVERY_COPY can fire
4. FixE candidates (12–19/file) fire during pre-convergence (err_floor=0.0)
   → sustain triggers SOFT_ASSIST, not HARD_RECOVERY_COPY
5. NE: no far-end activity → Fix E gate never qualifies → no actions, no exhaustion

Full 800-case stats in progress (`coverage_analysis.log`).

---

## §4 Root cause hypothesis

### FS_echo gap (vs AEC2/AEC3)

**Hypothesis**: SOFT_ASSIST fires 3 times during pre-convergence in nearly every FS file.
Each SOFT_ASSIST slightly misaligns the main filter toward the shadow filter's state.
After convergence, the budget is exhausted → no further correction possible.
In stable FS, the main filter is worse than AEC2/AEC3 not because shadow can't help,
but because (a) the 3 pre-convergence assists cause misalignment, and
(b) post-convergence shadow advantage (if any) is locked out by exhausted budget.

**Alternative**: Standard gate also requires pc_recovery_mode to fire hard copy.
Since pc_recovery_mode never activates, standard gate hard copy ALSO never fires.
In stable FS, shadow occasionally has a clear advantage (EPC-like event or
gain change) but main filter cannot receive a copy from either gate.

### DT_echo gap (vs AEC2/AEC3)

**Hypothesis**: DT-aware shadow gating (κ-1-A) intentionally keeps shadow adaptation
rate low during DT to protect near-end — this is the DT_deg improvement trade-off.
The DT_echo gap is structural, not a control flow bug.

However: even if shadow did converge during DT, HARD_RECOVERY_COPY cannot fire
because pc_recovery_mode is always False. The "recovery" path is completely blocked.

---

## §5 Actionable findings for modification planning

Priority 1 (high confidence, clean fix):
**Separate SOFT_ASSIST from HARD_RECOVERY_COPY budget.**
SOFT_ASSIST should have its own session cap (or no cap — it's low-risk).
HARD_RECOVERY_COPY budget should be reserved for post-convergence recovery events.

Priority 2 (medium confidence, needs full stats):
**Restrict SOFT_ASSIST to pre-convergence only.**
Post-convergence NORMAL mode should fall through to NO_ACTION.
This prevents 3 pre-convergence assists from consuming the hard-copy budget.

Priority 3 (pending recovery-entry stats from full run):
**Investigate why pc_recovery_mode never activates.**
If epc_level=='small' never fires → investigate EPC detector thresholds.
If err_baseline_ratio < 1.5 → lower threshold or add alternative entry condition.

Priority 4 (low urgency):
**Add minimal state handling after SOFT_ASSIST.**
At minimum: re-set `main_err_smooth` (already done).
Consider partial Q adjustment to stabilize adaptation after W blend.
