# Subset Action Coverage Review — κ-4 (v2.8.0)

Date: 2026-04-23. Full 800-case AEC Challenge blind test.
Script: `python/coverage_analysis.py`. Code: `python/aec.py` v2.8.0 + `_diag_soft_assist_fired` instrumentation.

---

## §1 Full coverage table

| Subset | N | conv% | rec% | rec_entry/f | SA/f | SA_files | HC/f | HC_files | budget_ex | cand/f | blk/f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FS-static | 169 | 36.3 | 2.1 | 0.54 | 2.9 | 169/169 (100%) | 0.3 | 43/169 (25%) | 168/169 (99%) | 24.3 | 7.4 |
| FS-movement | 131 | 22.4 | 2.2 | 0.49 | 2.9 | 131/131 (100%) | 0.2 | 20/131 (15%) | 130/131 (99%) | 20.4 | 3.7 |
| DT-static | 186 | 46.5 | 1.8 | 0.63 | 2.9 | 186/186 (100%) | 0.2 | 35/186 (19%) | 186/186 (100%) | 27.4 | 10.7 |
| DT-movement | 114 | 17.1 | 0.8 | 0.32 | 2.9 | 114/114 (100%) | 0.1 | 16/114 (14%) | 114/114 (100%) | 22.7 | 5.8 |
| NE | 200 | 0.0 | 0.0 | 0.00 | 0.1 | 16/200 (8%) | 0.0 | 0/200 (0%) | 2/200 (1%) | 1.2 | 0.0 |

**Columns**: conv% = frames converged, rec% = frames in pc_recovery_mode, rec_entry/f = 0→1 recovery transitions per file,
SA/f = SOFT_ASSIST events per file, HC/f = HARD_RECOVERY_COPY events per file, budget_ex = session cap exhausted (≥3 actions),
cand/f = Fix E candidate frames, blk/f = frames blocked by err_ratio floor.

### DT budget-exhaustion detail

| Subset | budget_ex | SA-only (no HC) |
|---|---:|---:|
| DT-static | 186/186 (100%) | 151/186 (81%) |
| DT-movement | 114/114 (100%) | 98/114 (86%) |

---

## §2 Key findings

### Finding 1: SOFT_ASSIST fires in 100% of FS/DT files, exhausts budget

Every FS and DT file receives exactly ~2.9 SOFT_ASSIST events (≈3, the session cap `_DT_COPY_MAX_SESSION`).
The Fix E gate fires during **pre-convergence** (filter_converged=False → err_floor=0.0 → err_ratio_ok always True).
Three SOFT_ASSIST events → session_count=3 → **budget exhausted before convergence**.

After convergence, Fix E's HARD_RECOVERY_COPY path is inaccessible because:
- `_can_hard_copy` requires pc_recovery_mode=True (rarely active)
- Even if it were, session_count ≥ 3 → `_dt_cap_ok = False` → Fix E gate blocked

### Finding 2: HARD_RECOVERY_COPY fires only from the STANDARD GATE

The 43 FS-static hard copies and 35 DT-static hard copies do NOT come from Fix E.
They come from the **standard gate** (lines 3462–3479), which checks
`filter_converged AND pc_recovery_mode` WITHOUT the Fix E session budget.

Standard gate hard copy rate: 25% FS-static, 19% DT-static, 15% FS-movement, 14% DT-movement.
These are the cases where an EPC event fires → pc_recovery_mode enters → shadow builds sustained advantage.

**Fix E's HARD_RECOVERY_COPY path fires 0 times across all 800 cases.**

### Finding 3: PATH_CHANGE_RECOVERY enters in ~50–63% of FS/DT files but is active briefly

Recovery entries/file: 0.54 FS-static, 0.49 FS-movement, 0.63 DT-static, 0.32 DT-movement.

Only ~49–63% of FS/DT files ever enter recovery mode at all.
The other 37–51% have no EPC events (stable echo path throughout) → pc_recovery_mode never True
→ neither Fix E hard copy nor standard gate hard copy can fire for these files.

Recovery is active only 0.8–2.2% of frames average because:
a) Only ~50% of files enter it
b) When entered, hard copy auto-exits it (pc_recovery_mode=False on copy), or hangover drains in 100 frames

### Finding 4: DT-movement has lowest recovery rate

DT-movement: 0.32 recovery entries/file (32% of files), 0.8% recovery active.
This is the subset with the largest expected benefit from a copy mechanism
(path changes in DT), but has the fewest recovery entries.

Likely cause: movement path changes happen during heavy DT frames where
`dt_signal ≥ 0.3` blocks EPC detection → epc_level stays 'none' → pc_recovery_mode never enters.

### Finding 5: err_ratio floor blocks substantial candidates in DT

DT-static blocked: 10.7 candidate frames/file by err_ratio floor.
DT-movement blocked: 5.8/file.

In post-convergence NORMAL mode, err_floor=1.0. Candidates blocked here are
"backwards-copy protection" working as designed. But since budget is already
exhausted by pre-conv SOFT_ASSIST, these blocked frames are moot.

---

## §3 Control flow reconstruction

### What actually happens in a typical FS file:

1. **Frames 0–~50**: `shadow_frame_count < 50` — no gate activity
2. **Frames ~50–conv_frame**: pre-convergence (filter_converged=False)
   - Fix E can fire: err_floor=0.0, shadow has natural init advantage
   - Fix E fires SOFT_ASSIST × 3 (sustain=4, then cooldown=200, then again)
   - Budget exhausted: session_count=3
3. **After conv_frame**: post-convergence
   - Fix E gate: `_dt_cap_ok = False` → Fix E permanently blocked for this session
   - Standard gate: if pc_recovery_mode activates (EPC event), can fire hard copy
   - ~54% of FS files: EPC occurs → 25% eventually get standard gate hard copy
   - ~46% of FS files: no EPC → no hard copy, no Fix E → only natural PBFDKF tracking

### What actually happens in a typical DT file:

1. Pre-convergence SOFT_ASSIST × 3 (same as FS)
2. Budget exhausted
3. Post-convergence: DT-aware gating slows shadow adaptation during DT → shadow rarely has large echo advantage
4. Standard gate rarely fires in DT: shadow advantage in DT is near-end bias, not echo-path advantage
5. pc_recovery_mode rarely enters in DT-movement because DT blocks EPC detection
6. Result: 81% DT-static and 86% DT-movement files get ONLY the 3 pre-conv SOFT_ASSIST events

---

## §4 Root cause per gap

### FS_echo gap (−0.370 vs AEC2)

**Primary causes**:
1. **Pre-convergence SOFT_ASSIST wastes budget**: 3 assists during pre-conv period
   misalign main filter W without proportional benefit, and prevent post-conv hard copies via Fix E.
2. **46% of FS files never get a hard copy** (no EPC → no recovery mode → no copy).
   These files depend entirely on natural PBFDKF tracking, which is slower than AEC2/AEC3.
3. **Fix E HARD_RECOVERY_COPY path is dead**: Budget always exhausted; even when Fix E would
   qualify for hard copy, it can't fire.

**Secondary causes**:
- Standard gate fires in only 25% of FS-static files (requires EPC + recovery + shadow sustained advantage simultaneously).

### DT_echo gap (−0.671 vs AEC2)

**Primary cause**:
- DT-aware shadow gating (κ-1-A) intentionally slows shadow adaptation during DT to protect near-end.
  This is the **structural trade-off** that produces DT_deg +0.506 gain but DT_echo −0.671 cost.
  Not a control-flow bug — the shadow simply doesn't track echo during DT.

**Contributing cause** (would not close the gap, but confirms mechanism is broken):
- 81–86% of DT files get only SOFT_ASSIST (3 pre-conv events), never a hard copy.
- Even if shadow tracked echo perfectly in DT, Fix E hard copy can't fire (budget gone).
- Standard gate fires in only 14–19% of DT files.

---

## §5 Fix priority assessment

| Fix | Plan §7 | Confidence | Expected impact |
|---|---|---|---|
| **Separate SOFT_ASSIST / HARD_RECOVERY_COPY budget** | Priority 4 | Very high | Unblocks Fix E hard copy path post-convergence |
| **Restrict SOFT_ASSIST to pre-convergence only** | Priority 1 | High | Eliminates post-conv NORMAL firing; reduces budget waste |
| **Add state handling for SOFT_ASSIST** | Priority 3 | Medium | W/Kalman consistency; secondary effect |
| **Adjust PATH_CHANGE_RECOVERY entry** | Priority 2 | Medium | Needs more data; DT-movement gap may not be recoverable |

**Note**: The DT_echo gap is primarily a structural trade-off (κ-1-A design), not a bug in the
copy/recovery mechanism. Fixing the budget issue will not close DT_echo gap significantly.
The FS_echo gap has more potential to close via mechanism fix.
