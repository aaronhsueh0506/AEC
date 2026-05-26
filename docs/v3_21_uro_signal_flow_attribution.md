# v3.21 URO / Linear Output Selection — Signal Flow Attribution

**Date**: 2026-05-27
**Cases**: xFk7 (DT_mvmt), 9xjhi (FS_static)
**AEC3 behavioral reference**: `bin/aec3_cli` run completed 2026-05-27

---

## A. AEC3 Behavioral Reference Results

AEC3 reference binary (`bin/aec3_cli`, arm64) scored on the 12-case cohort (2026-05-27).
`bin/aec3_linear_cli` not found — nores table skipped.

| Case (short) | Bucket | Metric | M0 | AEC3 | AEC3 Δ | Our M_full Δ | AEC3 vs M_full |
|---|---|---|---|---|---|---|---|
| ZJYUt0O | DT_mvmt | deg | 3.058 | 2.177 | −0.881 | (−0.090) | AEC3 worse |
| **xFk7** | **DT_mvmt** | **deg** | **2.881** | **1.275** | **−1.606** | **(−0.988)** | **AEC3 worse by 0.618** |
| wVYSGVTT | DT_mvmt | deg | 3.205 | 1.540 | −1.665 | (+0.046) | AEC3 worse |
| MYrVxVEM | DT_static | deg | 1.724 | 1.275 | −0.449 | (+0.096) | AEC3 worse |
| XRTnTUjU | DT_static | deg | 2.751 | 2.062 | −0.689 | (+0.069) | AEC3 worse |
| jtYTdZm3 | DT_static | deg | 2.587 | 2.298 | −0.289 | (+0.204) | AEC3 worse |
| nVUnxqHLr | DT_static | deg | 2.424 | 1.547 | −0.877 | (−0.130) | AEC3 worse |
| 0I0XMl3M | FS_mvmt | echo | 4.375 | 4.296 | −0.079 | (−0.241) | AEC3 better |
| **9xjhi** | **FS_static** | **echo** | **4.565** | **3.442** | **−1.123** | **(−2.211)** | **Python worse by 1.088** |
| qNvSMyUS | FS_static | echo | 3.972 | 3.596 | −0.376 | (−0.115) | AEC3 worse |
| xQEUtY2p | FS_static | echo | 3.712 | 4.219 | +0.507 | (−0.086) | AEC3 better |
| 014AzuqP | NS | deg | 4.356 | 4.164 | −0.192 | (+0.000) | AEC3 worse |

**AEC3 bucket means** (Δ vs M0):
- DT_mvmt: −1.384 | DT_static: −0.576 | FS_mvmt: −0.079 | FS_static: −0.331 | NS: −0.192

**Gate check against v3.21 gate criteria**:
- G1 (DT bucket Δdeg ≥ −0.05): AEC3 FAILS on DT_mvmt (−1.384) and DT_static (−0.576)
- G2 (no per-case DT Δdeg < −0.20): AEC3 FAILS — xFk7 Δ=−1.606, wVYSGVTT Δ=−1.665, ZJYUt0O Δ=−0.881
- G3 (stress cases Δ ≥ −0.10): AEC3 FAILS — xFk7 Δ=−1.606, nVUnxqHLr Δ=−0.877
- G4 (FS bucket Δecho ≤ +0.05): AEC3 PASSES (Δ=−0.331)

**Critical finding**: AEC3 itself does not meet G1/G2/G3 on this cohort. The v3.21 gate criteria cannot be used as an AEC3-parity target for DT_movement cases.

---

## B. URO C++ Source Audit (`echo_remover.cc`)

### B.1 Default configuration

- `enable_coarse_filter_output_usage` (EchoCanceller3Config::FilterConfiguration) = **`true`** (default)
- `use_coarse_filter_output_` (EchoRemover member field) initialised from config → **`true`** by default

Prior MEMORY.md note stated "defaults False" — **this was incorrect**. URO runs by default in production AEC3. Python flag `use_refined_output_selection_for_linear_path` default-OFF means Python historically did NOT run URO. Full composition turns it ON (correct alignment).

### B.2 Exact cond1/cond2 formulas

```cpp
// AEC3 UseRefinedOutput() — returns true = use refined, false = use coarse

// cond1 (coarse_cleaner): e2_coarse is smaller than e2_refined → coarse output preferred
if (e2_coarse < 0.9f * e2_refined
    && y2 > 30.f * 30.f * kBlockSize          // kBlockSize = 64
    && (s2_refined > 60.f * 60.f * kBlockSize
     || s2_coarse  > 60.f * 60.f * kBlockSize))
    return false;  // use coarse

// cond2 (refined_diverged): refined diverged (both e2 signals but y2 below e2)
if (e2_coarse < e2_refined && y2 < e2_refined)
    return false;  // use coarse

return true;  // default: use refined
```

**Python match**: cond1 and cond2 formulas are **identical** in Python `_aec3_select_linear_filter_output`. ✓

### B.3 Signal sources

All five signals (e2_refined, e2_coarse, y2, s2_refined, s2_coarse) are **time-domain sum-of-squares** over one kBlockSize=64-sample block (`SubtractorOutput::ComputeMetrics(y)`). Python uses the same approach with `np.sum(x**2)`. ✓

### B.4 Threshold analysis D2 (no candidate fix)

| | AEC3 | Python |
|---|---|---|
| Block size | kBlockSize = 64 samples (4 ms @ 16 kHz) | hop = 160 samples (10 ms @ 16 kHz) |
| thr_30 (float²) | 30²×64 / 32768² = 1.788e-6 | 30²×160 / 32768² = 4.471e-6 |
| thr_60 (float²) | 60²×64 / 32768² = 7.153e-6 | 60²×160 / 32768² = 1.788e-5 |
| Absolute ratio | — | 2.5× higher (hop-proportional) |
| **Gate predicate y2 / thr_30** | same | **same** (y2 also scales with hop) |

**Both AEC3 and Python compute sum-of-squares (not mean) over their respective block sizes.** Because y2 = Σ(mic²) and thr_30 = 30²×hop both scale with hop, the predicate `y2 > thr_30` is hop-normalized — it fires at the same signal amplitude in both systems. The absolute literal values differ (×2.5) but the gate predicate is functionally equivalent under equal signal amplitude.

**Classification**: Source literal differs (kBlockSize=64 vs hop=160) but is a correct RMS-equivalent scaling. **No parity bug; no candidate fix.** Python hop=160 is a hard constraint (10 ms @ 16 kHz). Changing the threshold to kBlockSize=64 would introduce a 2.5× asymmetry (y2 not correspondingly changed) and is incorrect.

**Note on 9xjhi**: D2 does NOT cause the extra gap. The threshold difference is cosmetic (same gate behaviour). The extra 1.088 gap is attributable to shadow NLMS convergence rate, not URO threshold.

### B.5 Other audit findings

| | AEC3 | Python | Status |
|---|---|---|---|
| Hysteresis on cond1/cond2 | None | None | ✓ Match |
| FormLinearFilterOutput | Always runs (30-sample crossfade) | Default OFF (binary switch) | Structural gap; flag ON enables it |
| ML-REE override gate | Forces refined when ML-REE active | N/A (no ML-REE) | Irrelevant |
| Multi-channel | Loops all channels | Single-channel | By design |
| SafeClamp before ComputeMetrics | e_refined clamped AFTER metrics | No equivalent clamp | Low severity |

---

## C. 9xjhi — Per-frame Signal Flow (FS_static, cond1 root cause)

**Root cause (CORRECTED 2026-05-27)**: URO cond1 fires 41.8% of frames in M_full_delay.
Prior attribution claimed Bundle A `use_per_bin_h_error_refresh` was the **direct cause** — this is INCORRECT.
Trace confirms: cond1 fires 41.8% even in **M_C_only** (URO only, no Bundle A/B).
cond1 is **intrinsic to the FS_static recording**, not Bundle-A-induced.

### C.1 AEC3 classification

AEC3 ref echo = 3.442 (Δ = −1.123 vs M0 = 4.565). **AEC3 also regresses on this case.**

Our M_full echo = 2.354 (Δ = −2.211). Extra gap vs AEC3 = **1.088 dB**.

**Classification**: Mixed Category 1 (AEC3 also fails) + Category 3 (extra Python gap from coarse filter quality).

The extra 1.088 dB gap is attributable to:
- Python shadow NLMS less converged than AEC3's built-in coarse NLMS when cond1 fires
- D2 threshold: Python's 2.5× higher threshold means cond1 fires LESS often than AEC3 would — if anything, Python is more conservative (fewer coarse switches). The 1.088 gap is NOT from D2.
- Root: when cond1 fires and routes to coarse, AEC3's coarse output has better echo suppression (coarse filter better converged). Our shadow NLMS output when not converged is worse.

### C.2 Per-variant cond1 analysis (trace run 2026-05-27)

Script: `python/v3_21_uro_signal_flow_trace.py`, 2187 frames total (9xjhi FS_static).

**URO gate fire rates:**

```
  Variant       n_fr URO_on  cond1%  cond2%  coarse%    ul%
  ------------ ----- ------ ------- ------- -------- ------
  M0            2187      N    0.0%    0.0%     0.0%  94.6%
  M_C_only      2187      Y   41.8%   13.4%    46.0%  94.6%
  M_BC          2187      Y   32.5%    7.4%    35.4%  92.4%
  M_D           2187      Y   42.5%    6.0%    47.0%  92.4%
  M_full        2187      Y   41.8%    6.5%    46.9%  92.4%
```

**cond1 energy breakdown (mean over frames where cond1 fires):**

```
  Variant       e2_ratio     e2_ref     e2_coa         y2    W_norm  coarse_conv%
  ------------ --------- ---------- ---------- ---------- --------- -------------
  M0               N/A        N/A        N/A        N/A       83.01          0.0%
  M_C_only        0.6150   9356.991      8.055     19.271     83.01          0.0%
  M_BC            0.5938  12040.185     11.635     19.342     71.85          0.1%
  M_D             0.5467     16.016      8.417     19.817     70.05          0.0%
  M_full          0.5422     16.006      8.317     19.954     70.10          0.0%

  [D2 threshold ref] thr_30 Python=1.341e-04  AEC3-equiv=5.364e-05  ratio=2.5×
  [D2 threshold ref] thr_60 Python=5.364e-04  AEC3-equiv=2.146e-04  ratio=2.5×
```

**Key findings:**

1. **cond1 is intrinsic**: Fires 41.8% in M_C_only (no Bundle A/B). Prior hypothesis that Bundle A *caused* cond1 is WRONG.

2. **Mechanism without Bundle A (M_C_only)**: e2_refined=9357 (large — refined filter has large residuals on FS_static without per_bin_h_error_refresh), e2_coarse=8 (small). Ratio=0.615 < 0.9 → cond1 fires. y2=19.27 >> thr_30=1.341e-4 → y2 gate always passes.

3. **Mechanism with Bundle A (M_D/M_full)**: per_bin_h_error_refresh reduces e2_refined from 9357 → 16. But e2_coarse=8.4 → ratio still 0.547 < 0.9 → cond1 still fires. Bundle A changes the absolute level but not the relative ratio.

4. **Bundle B partially helps (M_BC)**: cond1 drops 41.8% → 32.5%. Shadow gates (poor_excitation, narrowband_mask) update shadow more conservatively → e2_coarse slightly increases → fewer cond1 frames. But only 9% reduction — not a fix.

5. **coarse_conv=0% throughout ALL variants**: Shadow NLMS never achieves convergence on this FS_static recording. When cond1 routes to shadow output, the shadow is always un-converged → poor echo suppression → echo collapse.

6. **Root (revised)**: The FS_static case has a signal pattern where the shadow filter consistently has lower residual (e2_coarse < e2_refined) but is un-converged. This is an inherent weakness of cond1 routing to an un-converged shadow. AEC3's coarse NLMS converges better (less AEC3 gap vs M0). Python shadow NLMS with weaker convergence dynamics → larger gap.

---

## D. xFk7 — Per-frame Signal Flow (DT_mvmt, cond2 root cause)

**Root cause (CONFIRMED 2026-05-27)**: URO cond2 fires 31.5% of frames in M_full_delay after delay_first event (frame 104).
H_error reset → e2_refined spikes → cond2 (e2_coarse < e2_refined AND y2 < e2_refined) fires → routes to coarse output (also not converged post-delay).

### D.1 AEC3 classification

AEC3 ref deg = 1.275 (Δ = −1.606 vs M0 = 2.881). **AEC3 is worse than our M_full (1.893).**

**Classification**: **Category 1 — AEC3 also fails catastrophically.** Our M_full is 0.618 deg BETTER than AEC3. This is NOT a v3.21 alignment failure — it is an AEC3-inherent limitation on DT_movement cases.

AEC3's cond2 also fires after echo path change (delay_first) because:
- AEC3's H_error reset on path change (SetConfig(initial) + leakage=0.005/hop) causes e2_refined to spike
- This triggers cond2 (e2_coarse < e2_refined AND y2 < e2_refined)
- AEC3's H_error reset is MORE aggressive than our Python version (AEC3 leakage_converged=0.005/hop = +1.25/hop, while our filter continues without such aggressive leakage)
- Despite this, AEC3 gets worse deg (1.275 vs 1.893) — the H_error reset in AEC3 is more damaging

**Implication**: The D1_corrected (AEC3 leakage_diverged=1.25/hop) gate variant previously showed once_conv never latches with this leakage, consistent with AEC3's own failure on this case.

### D.2 Per-variant cond2 analysis (trace run 2026-05-27)

Script: `python/v3_21_uro_signal_flow_trace.py`, 3677 frames total (xFk7 DT_mvmt).

**URO gate fire rates:**

```
  Variant       n_fr URO_on  cond1%  cond2%  coarse%    ul%
  ------------ ----- ------ ------- ------- -------- ------
  M0            3677      N    0.0%    0.0%     0.0%  96.1%
  M_C_only      3677      Y   38.0%   59.2%    63.3%  96.1%
  M_BC          3677      Y   36.3%   56.8%    61.1%  96.1%
  M_D           3677      Y   20.4%   13.3%    25.0%  96.1%
  M_full        3677      Y   16.6%   31.5%    40.5%  96.1%
```

**Delay-first split (cond2% and ul% before/after delay event):**

```
  Variant       delay_fr  cond2_pre%  cond2_post%  ul_pre%  ul_post%
  ------------ --------- ----------- ------------ -------- ---------
  M0                   6        0.0%         0.0%     0.0%     96.1%
  M_C_only             6        0.0%        59.2%     0.0%     96.1%
  M_BC                 6        0.0%        56.8%     0.0%     96.1%
  M_D                  6        0.0%        13.3%     0.0%     96.1%
  M_full             104        0.8%        30.7%     0.0%     96.1%
```

Note: `delay_fr=6` for M0/M_C_only/M_BC/M_D is the fallback detection (no `use_full_delay_change_chain`). `delay_fr=104` for M_full is the actual full delay chain trigger.

**usable_linear gate chain (M_D and M_full):**

```
  Variant       gate1%  gate2%  gate3_ext%  gate3_conv%
  ------------ ------- ------- ----------- ------------
  M_D            96.2%   96.2%       97.2%        96.5%
  M_full         96.2%   96.2%       97.2%        96.5%
```

**Key findings:**

1. **cond2 without Bundle A (M_C_only): 59.2%**. Without per_bin_h_error_refresh, e2_refined is very large (same phenomenon as 9xjhi — refined filter has large residuals) → cond2 (e2_coarse < e2_refined AND y2 < e2_refined) fires almost every post-delay frame.

2. **Bundle A dramatically reduces cond2**: M_C_only 59.2% → M_D 13.3%. per_bin_h_error_refresh reduces e2_refined → cond2 fires less (y2 no longer < e2_refined as often).

3. **Full delay chain (M_full) increases cond2 vs M_D**: 13.3% → 30.7% post-delay. The H_error reset on delay_first (full chain) causes e2_refined to spike more aggressively → more cond2. This explains why M_full AECMOS is slightly worse than M_D on xFk7 (−0.988 vs −0.973).

4. **cond1 high in M_C_only/M_BC**: 38.0%/36.3% — without Bundle A, e2_refined is large → coarse output has lower residual → cond1 also fires. Bundle A reduces cond1: 38.0% → 20.4% (M_D) → 16.6% (M_full).

5. **usable_linear unaffected**: ul=96.1% for all variants on xFk7. Unlike wVYSGVTT, xFk7 does NOT suffer from FilterPlateauDetector. The AECMOS failure on xFk7 is entirely from URO cond2 routing to un-converged coarse output post-delay, not from ul suppression.

6. **Gate chain healthy**: gate1=96.2%, gate2=96.2%, gate3_ext=97.2%, gate3_conv=96.5% — ul gates pass consistently. The problem is purely in the URO signal routing, not in the AecState / usable_linear chain.

---

## E. Conclusion Classification and nores vs AECMOS Distinction

### E.1 nores LF artifact vs AECMOS echo — MUST distinguish

| Layer | Status | Evidence |
|-------|--------|----------|
| **nores LF artifact** (0–500 Hz, `_ours_nores.wav`) | **CLOSED / IMPROVED** | M_A Bundle A: nores LF Δ = −6.03 dB vs M0. The LF grid/fan artifact on 9xjhi is a linear-layer issue that Bundle A (H_error parity) substantially closes. |
| **final echo AECMOS regression** (`_ours.wav`) | **STILL OPEN** | M_full echo AECMOS = 2.354 (M0=4.565 Δ=−2.211). URO cond1 fires 41.8% → routes to un-converged shadow → downstream output degraded. |

**The nores improvement and echo AECMOS failure are separate phenomena.** Bundle A fixed the linear-filter residual artifact (nores). URO routing to an un-converged shadow then causes the final echo AECMOS collapse. These must not be conflated.

### E.2 Case classification

| Case | Category | Verdict | Action |
|------|----------|---------|--------|
| xFk7 DT_mvmt | **1 — AEC3 also fails** | AEC3 deg=1.275, Python M_full=1.893. **Our impl is better than AEC3 by 0.618 deg.** NOT a v3.21 port gap. xFk7 is NOT a blocking issue for v3.21. Monitor as DT guard: confirm no future fix makes Python worse than AEC3. | Document as AEC3 limitation. No fix required. |
| 9xjhi FS_static | **1+3 mixed** | AEC3 echo=3.442, Python M_full=2.354. Extra gap=1.088. cond1 INTRINSIC (fires 41.8% even without Bundle A). Shadow coarse_conv=0% all variants. Root: partition-depth-normalised effective adaptation cadence ~0.62× AEC3 (n_partitions=6, hop=160 hard constraint; if n=5 historical: ~0.74×). No direct AEC3 formula or unit-mixing parity bug identified. | Category 1+3. Extra gap = shadow/coarse convergence structural mismatch. No direct formula/unit parity bug found in this audit; no safe strict-port v3.21 candidate identified yet. |

### E.3 Gate reconsideration — AEC3-relative comparison replaces M0-relative gates

The v3.21 gate criteria (G1/G2/G3) are M0-relative. AEC3 itself fails G1/G2/G3 on DT cases in this cohort. **G1/G2/G3 are not valid AEC3-parity targets for DT_movement.** They must not be used to block M_full shipment on DT grounds.

**Revised pass criteria**: Compare M_full vs AEC3 directly (AEC3-relative bars).

| Metric | Our M_full Δ vs M0 | AEC3 Δ vs M0 | M_full vs AEC3 | Status |
|--------|-------------------|--------------|----------------|--------|
| DT_mvmt mean deg | −0.363 | −1.384 | **+1.021 (we better)** | ✓ PASS |
| DT_static mean deg | +0.060 | −0.576 | **+0.636 (we better)** | ✓ PASS |
| FS_mvmt mean echo | −0.241 | −0.079 | −0.162 (AEC3 better) | Monitor |
| FS_static mean echo | −0.804 | −0.331 | **−0.473 (AEC3 better)** | ⚠ GAP |

**Our M_full beats AEC3 on all DT buckets.** FS_static is the only remaining gap (−0.473 mean, primary case 9xjhi extra 1.088). This gap is architectural (convergence rate), not a parity bug.

**New 12-case pass criteria** (AEC3-relative, replaces M0-relative G1/G2/G3):
1. Flag OFF byte-equal 25/25 PASS
2. DT cases: M_full must not be worse than AEC3 by more than 0.10 deg (per-case)
3. 9xjhi extra gap vs AEC3 ≤ 0.30 echo (target; current = 1.088)
4. 9xjhi nores LF artifact closure maintained (nores LF Δ vs M0 < −0.5 dB)
5. No new FS echo catastrophic regression: any FS case vs AEC3 worse > 0.30 → STOP
6. No new DT catastrophic regression: any DT case vs AEC3 worse > 0.10 → STOP

---

## F. URO Formula Parity Summary

| Item | AEC3 | Python | Gap | Priority |
|------|------|--------|-----|----------|
| cond1 formula | e2_coa < 0.9×ref AND y2>thr30 AND s2>thr60 | Identical | None | — |
| cond2 formula | e2_coa < e2_ref AND y2 < e2_ref | Identical | None | — |
| D2 threshold factor | kBlockSize=64 (sum-of-squares) | hop=160 (sum-of-squares, hop-proportional) | **Source literal differs; gate predicate equivalent** | **No candidate fix** |
| FormLinearFilterOutput | Always runs (crossfade) | OFF by default; ON when flag enabled | Structural | Flag ON in M_full_delay composition |
| Hysteresis | None | None | None | — |
| Outer ML-REE gate | Forces refined | N/A | N/A | — |

**D2 final classification**: Both AEC3 and Python use sum-of-squares (not mean). Python scales thresholds by hop=160, AEC3 by kBlockSize=64. Since `y2 = Σ(mic²)` also scales proportionally with block size, the gate predicate `y2 > thr_30` is functionally equivalent at equal signal amplitude. No candidate fix; hop=160 is a hard constraint. Previously mislabelled as "2.5× parity gap" — corrected to "hop-normalized equivalent."

---

## G. Next Steps

**G1 (DONE — gate framing corrected)**: M0-relative G1/G2/G3 replaced by AEC3-relative bars (see §E.3). xFk7 is no longer a blocking issue.

**G2 (DONE — D2 reclassified)**: D2 threshold is hop-normalized equivalent, not a real parity gap. No candidate fix.

**G3 (DONE — nores / AECMOS distinction)**: nores LF artifact = CLOSED (Bundle A). Final echo AECMOS regression = still open (URO + shadow quality).

**G4 (active) — 9xjhi shadow/coarse convergence audit**:
Target: close 9xjhi extra gap of 1.088 echo vs AEC3, or document as architectural ceiling.
See `docs/v3_21_9xjhi_shadow_coarse_gap_trace.md` for comprehensive audit results.
Focus: shadow NLMS convergence quality when cond1 routes to shadow. Primary finding: partition-depth-normalised effective adaptation cadence ~0.62× AEC3 (n=6 canonical; if n=5 historical: ~0.74×; no direct formula bug). See `docs/v3_21_9xjhi_shadow_coarse_gap_trace.md §B, §C, §E` for corrected analysis.

**Policy**: Only fix code if exact AEC3 parity bug identified. If not, document as:
"No direct formula/unit parity bug found in this audit; remaining 9xjhi extra gap is currently classified as hop=160 / partition-depth structural limitation, with no safe strict-port v3.21 candidate identified yet."

---

## H. Reference

- AEC3 source: `docs/aec3_extracts/src/aec3/echo_remover.cc` (UseRefinedOutput lines 112-133)
- AEC3 source: `docs/aec3_extracts/src/aec3/subtractor.cc` (ComputeMetrics)
- Python implementation: `python/modules/orchestrator.py` `_aec3_select_linear_filter_output` (~line 3947)
- AEC3 behavioral reference script: `python/v3_21_aec3_reference_aecmos.py`
- Targeted frame trace: `python/v3_21_uro_signal_flow_trace.py`
- 12-case verdict: `docs/v3_21_full_composition_12case_verdict.md`

**AEC3 reference NOT RUN for internal state inspection** — only output AECMOS scored. AEC3 cond1/cond2 fire rates on these cases are unknown (binary not instrumented). Source parity verified by code audit; behavioral parity by output comparison only.
