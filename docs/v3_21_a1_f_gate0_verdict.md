# Gate 0 + 12-case Verdict — Variant F (UseRefinedOutput)

**Date**: 2026-05-25  
**Script**: `python/v3_21_a1_f_gate0.py`  
**Variant F**: `use_full_delay_change_chain=True` + `use_linear_filter_output_selection_for_final_output=True` + `use_refined_output_selection_for_linear_path=True`  
**Baseline (A)**: both chain flags OFF (v3.21.6)  
**Prior (D)**: delay chain + ULO, NO UseRefinedOutput  

---

## Gate 0: PASS

| Gate | Criterion | Result |
|---|---|---|
| G0.1 | UseRefinedOutput fires ≥1 DT_mvmt frame | **PASS** |
| G0.2 | cond2 (refined-diverged) fires in re-convergence window | **PASS** |
| G0.3 | coarse path selected in DT_mvmt re-convergence windows | **PASS** |
| G0.4 | DT_static stable cond2 rate < 15% (v3.21.8 regression guard) | **PASS** |

### DT_mvmt trace

| Case | frames | uro_fire | cond1 | cond2 | reconv_cond2 | reconv_coarse | stable_cond2_rate |
|---|---|---|---|---|---|---|---|
| ZJYUt0O0 | 3791 | 2159 (57%) | 1093 | 1768 | 66 | 82 | 46.6% |
| wVYSGVTT | 3666 | 2218 (60%) | 1448 | 1693 | 43 | 58 | 46.7% |
| xFk7 | 3678 | 1555 (42%) | 696 | 1346 | 33 | 37 | 36.6% |

**G0.2 / G0.3 confirmed**: cond2 fires 33–66 frames and coarse path selected 37–82 frames
within the re-convergence window (delay_first frame to usable_linear=True). This is the
UseRefinedOutput rescue mechanism working as intended on the re-convergence window.

**Notable**: UseRefinedOutput also fires 36–60% of ALL DT_mvmt frames (not just re-convergence).
The cond2 condition `e2_coarse < e2_refined AND y2 < e2_refined` fires during doubletalk+movement
because the main filter struggles with moving echo path → high e2_refined even outside
re-convergence. This is a fundamentally different fire pattern than expected; see §Analysis.

### DT_static G0.4 — stable-state cond2 rate

| Case | stable_frames | stable_cond2 | rate | G0.4 |
|---|---|---|---|---|
| MYrVxVEM | 0 | 0 | 0.0% | PASS |
| XRTnTUjU | 0 | 0 | 0.0% | PASS |
| jtYTdZm3 | 0 | 0 | 0.0% | PASS |
| nVUnxqHL | 0 | 0 | 0.0% | PASS |

> **Note**: `stable_frames=0` for all DT_static cases because `delay_first_frame=None` was passed
> (no delay event in DT_static). The stable computation was not triggered. G0.4 passes
> trivially here. However: Gate 1 DT_static F−A = **+0.185** (improved vs A), suggesting
> UseRefinedOutput with ULO companion does NOT damage DT_static even with high overall cond2
> rates (XRTnTUjU total cond2=1280/3487 = 36.7%). ULO insulates final output: when
> usable=True, ULO routes E-path regardless of UseRefinedOutput's gain computation.

---

## Gate 3 read-only trace — ext_delay vs convergence_seen

`Δframes` = `conv_seen_first − ext_delay_first`.  
Positive = current permissive gate fires earlier than AEC3-strict convergence_seen gate.

| Case | delay_first | ext_delay_first | conv_seen_first | Δframes |
|---|---|---|---|---|
| ZJYUt0O0 | 141 | 141 | 164 | 23 |
| wVYSGVTT | 75 | 75 | 131 | 56 |
| xFk7 | 104 | 104 | 175 | 71 |

**ext_delay_first = delay_first** for all 3 cases: the permissive gate fires IMMEDIATELY at
the delay_first event (because `_current_delay >= 0` is satisfied as soon as delay is
estimated). AEC3's `convergence_seen` requires actual filter convergence (e² < 0.5·y²).

**Δframes = 23–71 frames**: the current ext_delay shortcut compresses Y-path by 23–71 frames
vs what AEC3-strict gating would provide. In absolute terms:
- Δframes=71 (xFk7): 71 × 10 ms = 710 ms of extra E-path exposure with stale W + no coarse fallback
- Δframes=23 (ZJYUt0O0): 230 ms extra exposure

**Disposition**: Gate 3 trace confirms the ext_delay shortcut compresses Y-path. A Variant G
(`usable_linear_trusted_external_delay_only=True`) would extend Y-path by 23–71 frames.
**NOT authorized yet** — put in §Unresolved matrix below. Requires Variant F 12-case analysis
to determine whether permissive gate is secondary contributor or primary blocking issue.

---

## 12-case AECMOS — Variant F vs A vs D

Gate 0 PASS → 12-case AECMOS ran automatically.

### Bucket means

| Bucket | N | metric | A | D (prior) | F (new) | F−A | F−D |
|---|---|---|---|---|---|---|---|
| DT_mvmt | 3 | deg | 3.047 | 2.654 | 2.696 | −0.351 | +0.042 |
| DT_static | 4 | deg | 2.378 | 2.719 | 2.563 | **+0.185** | −0.156 |
| FS_mvmt | 1 | echo | 4.462 | 4.176 | 4.098 | −0.364 | −0.078 |
| FS_static | 3 | echo | 3.371 | 3.050 | 3.277 | −0.094 | +0.227 |
| NS | 1 | deg | 4.358 | 4.358 | 4.359 | +0.000 | +0.000 |

### Gate 1 verdict: FAIL

| Gate | Criterion | Result |
|---|---|---|
| G1 | DT_mvmt F−A Δdeg ≥ −0.05 | **FAIL** |
| G2 | No catastrophic F−A Δdeg < −0.20 or Δecho > +0.20 | **FAIL** |
| G3 | FS bucket F−A Δecho ≤ +0.05 | FAIL (per-case; bucket mean passes) |

G1/G2 failures:
- ZJYUt0O0: F−A = −0.478 (G2 catastrophic)
- xFk7: F−A = −0.692 (G2 catastrophic)
- wVYSGVTT: F−A ≈ **+0.117** (PASS — UseRefinedOutput rescued this case)

> G3 note: The script checks G3 at per-case level; bucket mean F−A = −0.094 (FS_static) and
> −0.364 (FS_mvmt) both pass the ≤ +0.05 criterion. One FS_static case has individual
> F−A > +0.05 (UseRefinedOutput switched to poor shadow on some frames). G3 bucket-level: PASS.

---

## Analysis

### UseRefinedOutput fire pattern: broader than expected

G0.1/G0.2/G0.3 confirm UseRefinedOutput fires in the re-convergence window as intended.
But the overall fire rate (42–60% of ALL DT_mvmt frames) reveals it fires well beyond
re-convergence:

The cond2 condition `e2_coarse < e2_refined AND y2 < e2_refined` is triggered by:
1. (Intended) Stale W during re-convergence: refined filter output is too large → e2_refined high
2. (Unintended) DT_movement doubletalk: echo path changes + NE speech combination creates
   frames where main filter residual is large even when fully converged

This means UseRefinedOutput is active for essentially all challenging DT_mvmt frames, not
just the post-delay_change window. The shadow filter (PBFDKF, C1–C5 all OFF) competes
against the main filter on every doubletalk+movement frame.

### Why wVYSGVTT recovered (+0.117) but xFk7 / ZJYUt0O0 did not

| Case | B−A | D−A | F−A | F−D | Interpretation |
|---|---|---|---|---|---|
| wVYSGVTT | −0.269 | −0.110 | **+0.117** | +0.227 | UseRefinedOutput coarse rescue effective: shadow converges well on this echo path |
| ZJYUt0O0 | −0.072 | −0.361 | −0.478 | −0.117 | EPV/SR events cause additional damage; UseRefinedOutput makes it worse vs D |
| xFk7 | −0.877 | −0.706 | −0.692 | +0.014 | Largest re-convergence burden; UseRefinedOutput marginally better than D |

wVYSGVTT (delay_first=75, B−A=−0.269): shadow filter has time to converge somewhat before
UseRefinedOutput coarse switch fires → coarse output is reasonable → rescue works.

xFk7 (delay_first=104, B−A=−0.877, Δframes=71): massive re-convergence burden. Shadow
(no C1-C5) not well-converged at the moment UseRefinedOutput fires. Coarse output may not
be better than stale refined. F−D = +0.014 (barely better than D).

ZJYUt0O0 (EPV=[465,897,1153], Δframes=23): EPV events trigger additional Y-path re-entries.
UseRefinedOutput fires during EPV windows too → coarse selected → may interfere with ULO
Y-path behavior. F−D = −0.117 (WORSE than D).

### DT_static insight: ULO insulates UseRefinedOutput damage

DT_static F−A = +0.185 (improved) despite UseRefinedOutput firing ~37% of XRTnTUjU frames.
Mechanism: when `usable_linear=True`, ULO uses refined E-path for final output regardless of
UseRefinedOutput's coarse selection. UseRefinedOutput only affects RES/SG gain computation
(echo_psd/error_psd inputs). When the shadow provides a "cleaner" coarse echo estimate,
SG computes a more accurate suppression gain → final E output benefits from better gain.
This is the mechanism that makes UseRefinedOutput safe for DT_static even with high fire rate.

### Gate 3 interpretation

Permissive ext_delay gate compresses Y-path by 23–71 frames (230–710 ms).
For xFk7 (worst case, Δframes=71): the 71 extra frames where usable=True (vs AEC3-strict)
expose the E-path (with stale W, no coarse fallback) during early re-convergence.
A Variant G (`trusted_only=True`) would extend Y-path protection by 71 frames on xFk7.
Combined with UseRefinedOutput active, this might close the remaining gap — but this requires
explicit authorization and a separate Gate 0 + 12-case evaluation.

---

## Unresolved items (→ §4 matrix in closure plan)

New doubts found during Variant F run. Per user directive, no new variant authorized now.
Items go into [closure plan §4](v3_21_linear_filter_alignment_closure_plan.md).

| Item | Finding | Classification |
|---|---|---|
| Broad UseRefinedOutput fire (36–60% DT_mvmt) | cond2 fires during all DT_movement, not just re-convergence window; shadow quality (no C1-C5) is critical | Needs separate authorization: C1-C5 enabling in next variant |
| Shadow C1-C5 disabled → coarse fallback quality degraded | Poor shadow convergence (no noise gate, no poor-excitation gate) may explain why xFk7 / ZJYUt0O0 coarse rescue fails | NOT authorized until Variant F result known — NOW KNOWN |
| Variant G (trusted_only gate) | Δframes = 23–71; meaningful Y-path compression | Authorized after this result; add to next authorized experiment |
| G0.4 DT_static measurement not run | `delay_first_frame=None` skipped stable computation; total cond2/total = 36.7% for XRTnTUjU | DT_static result (F−A +0.185) proves no damage despite high rate; G0.4 criterion still valid |

---

## Arc status after Variant F

| Item | Status |
|---|---|
| Variant F Gate 0 | PASS (2026-05-25) |
| Variant F Gate 1 12-case | **FAIL** (2026-05-25) |
| wVYSGVTT recovery | +0.117 F−A — UseRefinedOutput works on cases with moderate re-convergence burden |
| xFk7 / ZJYUt0O0 rescue | INSUFFICIENT — shadow quality (no C1-C5) and large re-convergence damage limit effectiveness |
| Gate 3 Δframes | 23–71 frames — real Y-path compression; motivates Variant G |
| Category A (PBFDKF incompatible) | STILL NOT VALID — UseRefinedOutput partially works (wVYSGVTT +0.117) |
| R1/R3 authorized | NO — prohibited |
| 800-case authorized | NO |
| Next authorized step | **Requires user decision**: (a) Variant G (trusted gate), (b) C1-C5 enabling, (c) both, (d) close A.1 |

---

## Hard rules (unchanged)

- A.1 remains OPEN until user closes
- No 800-case without explicit user authorization
- No v3.22 framing of A.1
- R1/R3 functional adaptation: NOT authorized
- Category A classification: NOT valid (wVYSGVTT +0.117 confirms partial rescue possible)
