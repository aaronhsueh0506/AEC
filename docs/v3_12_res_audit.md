# v3.12 Phase 3A — RES gain_floor 5-path audit

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a` (v3.11.2, HEAD = f0ca5db)
**Inputs**: v3.11.2 main code (byte-equal vs v3.11.1)
**Status**: STATIC AUDIT — read-only; gates Sprint S6-S7 (gain_floor unification)
**Sprint**: S4-S5 (Phase 3A)

This doc enumerates the 5 gain-floor / gain-cap paths in `ResFilter`,
maps each to its trigger, source signal, and effect, then categorises
each as **physical fallback** (preserve as discrete clamp) or
**evidence patch** (merge into the canonical 6-step Beroutti / Ephraim-
Malah / Hänsler-Schmidt floor in Sprint S6-S7).

The Q7 V3 verdict (RES coherence broken — 5 paths apply independently
+ in parallel) is what Phase 3B is supposed to fix; this doc decides
which of the 5 are real architecture features and which are accumulated
patches.

---

## Path-by-path enumeration

### Path 1: `spectral_floor` — `enable_spectral_floor` shape-preserving floor

- **Location**: [aec.py:2733-2744](../python/aec.py#L2733-L2744), applied at [aec.py:2262](../python/aec.py#L2262), [2288](../python/aec.py#L2288), [2291](../python/aec.py#L2291), [2293](../python/aec.py#L2293)
- **Mechanism**:
  ```
  env_normalized = error_envelope / max(error_envelope)
  spectral_g_min[k] = effective_g_min + (1 - effective_g_min) ×
                      env_normalized[k] × spectral_floor_ratio
  ```
  Then `g = max(g, spectral_g_min)` at each gain-compute branch.
- **Source**: `self.error_envelope` (EMA of |error_spec|, α = `alpha_envelope`)
- **Gating**: `enable_spectral_floor AND far_power > 1e-4` (effectively always on when far-end active)
- **Per-bin or scalar**: **per-bin** (one floor per frequency bin, shape-dependent)
- **Fire-rate estimate**: ≈ 100 % of far-active frames; per-bin floor magnitude depends on local envelope shape
- **What it solves**: prevents flat over-suppression — a bin with high error envelope mass cannot be cut below `effective_g_min + (1-effective_g_min) × env_normalized × ratio`. Preserves perceptual spectral shape (avoids "musical hole" artefacts).
- **Verdict**: **PHYSICAL FALLBACK — preserve as discrete pre-canonical clamp**
  - This is shape-preservation, not echo-evidence-driven. Canonical Wiener / LSA gain alone has no spectral-shape constraint; this floor is what makes single-bin overshoot bounded.
  - Cohort tail uses this floor to avoid the "never-stationary path" cliff where Wiener gain alone goes to `g_min` uniformly.
  - Phase 3B action: **keep as a final clamp** (`g = max(g_canonical, spectral_floor)`).

### Path 2: `ne_g_floor` — `(1-coh²)` near-end protection floor

- **Location**: [aec.py:2755-2759](../python/aec.py#L2755-L2759); merged into `spectral_g_min` at line 2759
- **Mechanism**:
  ```
  ne_erle_gate = max(erle_factor, 0.3)
  ne_protection = (1 - coh²) × ne_erle_gate × (1 - fs_confidence)
  ne_g_min_ceil = 10 ** (ne_protect_db / 20)           # e.g. -10 dB → 0.316
  ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) × ne_protection
  spectral_g_min = max(spectral_g_min, ne_g_floor)
  ```
- **Source**: `coh²` (per-bin), `fs_confidence` (per-frame), `erle_factor` (per-frame)
- **Gating**: always computed; protection scales by `(1-fs_confidence)` so FS → 0
- **Per-bin or scalar**: **per-bin** (via coh² and the merged-spectral path)
- **Fire-rate estimate**: ≥ 95 % of frames (non-zero floor whenever coh² < 1 anywhere)
- **What it solves**: protect NE-only bins from over-suppression when coh² is low
- **Q7 V3 known defect** (already documented in [aec.py:2347-2353](../python/aec.py#L2347-L2353)):
  > NOTE (v3.10.5 investigation): dt_per_bin = max(effective_dt, 1-coh²) saturates ~1 in FS post-cancellation (echo cancelled → low coh² → "NE-like"), so the high_ne_conf < 0.3 gate rarely fires in FS — cap is largely dead code there. ... Left as-is pending a redesigned evidence metric that distinguishes DT-NE from FS-decoupling.
- The same root issue applies to `ne_g_floor`: post-cancellation FS frames have coh² → 0 → `(1-coh²) → 1` → `ne_g_floor` raises spectral_g_min → echo leakage **in FS** because we mistake decorrelated residual for NE.
- **Verdict**: **EVIDENCE PATCH — merge into canonical floor in S6-S7**
  - Replace `(1-coh²)`-driven NE evidence with the F3.1 mic-excess metric (`use_mic_excess_evidence`). The mic-excess metric is physical (mic energy exceeds ERL × X), does NOT saturate post-cancellation, and was AUROC-validated (0.871) in P1 Phase 1.
  - In the unified floor: `ne_protection ← ne_protection_from_mic_excess(near_psd, far_psd, erl_estimate)`.
  - This unblocks the Q7 V3 verdict: FS post-cancellation no longer raises ne_g_floor.

### Path 3: `epc_dt_cap` — EPC + DT scalar gain cap

- **Location**: [aec.py:2304-2307](../python/aec.py#L2304-L2307); applied in `_stage_gain_postprocess`
- **Mechanism**:
  ```
  if epc_active AND effective_dt > 0.35:
      g = min(g, 0.85)
  ```
- **Source**: scalar `epc_active` (bool, frame-level), scalar `effective_dt` (frame-level)
- **Gating**: 2-AND scalar gate (no per-bin signal)
- **Per-bin or scalar**: **frame-scalar** (uniform cap across all bins)
- **Fire-rate estimate**: ≈ 1-5 % of frames (EPC events × DT-active overlap)
- **What it solves**: during echo-path-change + double-talk, the filter is unreliable → cap gain to force ≥ 1 - 0.85 = 15 % minimum echo suppression
- **Architectural critique**:
  - Frame-scalar gate applied per-bin uniformly: ignores that some bins may be true NE (should NOT be capped at 0.85) while others may be true echo (should be capped harder than 0.85).
  - The 0.85 cap exists because: (1) filter W is mis-converged after EPC; (2) DT prevents the filter from re-converging during the event. So we need an upstream-of-Wiener clamp.
  - This is fundamentally a **filter_state hook**: `transient` or `recovering` state should drive this. Phase 2 wiring (`res_consume_filter_state`) makes this connection possible.
- **Verdict**: **EVIDENCE PATCH — fold into per-state ENR tuple in S8-S9**
  - Replace scalar `if epc_dt: g = min(g, 0.85)` with `if filter_state in {'transient', 'recovering'}: g = min(g, state_tuple.gain_cap)`.
  - This generalises beyond EPC to all path-change / divergence events, AND lets per-state tuples calibrate the cap (transient = 0.85, recovering = 0.95 ramp).

### Path 4: `quiet_mask` — sub-noise-floor bin pass-through

- **Location**: [aec.py:2676-2678](../python/aec.py#L2676-L2678); applied at [aec.py:2312](../python/aec.py#L2312)
- **Mechanism**:
  ```
  signal_floor = mean(error_psd) × 0.001 + 1e-8
  quiet_mask[k] = (echo_psd[k] < signal_floor) AND (error_psd[k] < signal_floor)
  g[quiet_mask] = 1.0       # full pass-through
  ```
- **Source**: per-bin `echo_psd` AND `error_psd` (both below 1/1000 of mean error)
- **Gating**: 2-AND per-bin gate, magnitude-based
- **Per-bin or scalar**: **per-bin** (true mask)
- **Fire-rate estimate**: depends on dynamic range — dominantly fires on bins between speech formants (silent gaps in spectrum)
- **What it solves**: physical noise floor preservation — if a bin has no echo AND no audible error energy, do not multiply by `g < 1` (that would inject coloured noise from quantisation / numerical noise into a silent bin).
- **Verdict**: **PHYSICAL FALLBACK — preserve as discrete pre-canonical clamp**
  - This is not echo-evidence-driven. The AND condition on `echo_psd` + `error_psd` (both sub-floor) means the bin contributes nothing audibly regardless of what `g` is.
  - Canonical formulation doesn't need this in steady state, but for cohort tail and weak-signal frames it is the only thing preventing the post-filter from sculpting numerical noise.
  - Phase 3B action: **keep as a hard override after the canonical floor**: `g[quiet_mask] = 1.0` last (post-canonical-floor, post-spectral-floor).

### Path 5: `divergence_floor` — divergence indicator gain cap (hybrid)

- **Location**: [aec.py:2403-2406](../python/aec.py#L2403-L2406); applied in `_stage_gain_postprocess`
- **Mechanism**:
  ```
  if divergence > 0.3:
      divergence_gain = 0.01 + (1 - 0.01) × (1 - divergence)
      g = min(g, divergence_gain)
  ```
- **Source**: scalar `divergence` ∈ [0, 1] from `FilterConvergenceAnalyzer.divergence` (EMA of `error / near` ratio)
- **Gating**: scalar threshold > 0.3
- **Per-bin or scalar**: **frame-scalar** (uniform cap, divergence-dependent)
- **Fire-rate estimate**: rare in steady state; concentrated on divergence-prone outlier cases (cohort tail, post-EPC, etc.)
- **What it solves**: when filter diverges (`error >> near`), the residual_echo estimate cannot be trusted → cap gain to force ≥ (1 - divergence_gain) minimum suppression
- **Hybrid status**:
  - This path **already** consumes a filter-state signal (`divergence` scalar from convergence analyser).
  - Phase 2 wiring (`res_consume_filter_state=True`) lets Phase 3 generalise: instead of one scalar `divergence > 0.3` threshold, use the full `filter_state` enum (`diverged` state explicitly + `suspicious_dt` mid-cap + `transient` light-cap, etc.).
- **Verdict**: **HYBRID — generalise into per-state ENR tuple in S8-S9**
  - Replace `if divergence > 0.3: g = min(g, 0.01 + 0.99×(1-divergence))` with `state_tuple[filter_state].gain_cap`.
  - The existing `divergence` scalar can stay as a redundant safety net (or be retired if state classifier is authoritative — decided at S8-S9 design time).

---

## Categorisation summary

| Path | Category | S6-S7 action | S8-S9 action |
|---|---|---|---|
| `spectral_floor` | Physical | Keep as final clamp | (no change) |
| `ne_g_floor` | Evidence patch | **MERGE INTO CANONICAL** (use mic-excess for NE evidence) | (consumed by canonical) |
| `epc_dt_cap` | Evidence patch | Lift out of cap chain | **FOLD INTO per-state tuple** (`transient` / `recovering`) |
| `quiet_mask` | Physical | Keep as last-resort override | (no change) |
| `divergence_floor` | Hybrid | (no change) | **GENERALISE** into per-state tuple, retire scalar threshold |

---

## Phase 3B (S6-S7) execution plan implications

**S6 — Canonical 6-step assembly**:
1. Noise PSD estimate (unify signal_floor + CNG floor)
2. A priori SNR (residual_echo_psd from `_stage_residual_model`)
3. A posteriori SNR (ENR from gain compute)
4. Wiener / LSA gain (`softgate_emr` ENR path)
5. Temporal smooth (unify gain_smooth + CNG temporal)
6. **Unified gain floor**:
   - `canonical_floor = mic_excess_protection(near_psd, far_psd, erl_estimate)` — replaces `ne_g_floor`'s `(1-coh²)` evidence
   - `g = max(g_wiener, canonical_floor)`
   - `g = max(g, spectral_floor)` — discrete shape-preservation clamp (PRESERVED)
   - `g[quiet_mask] = 1.0` — discrete physical override (PRESERVED)

**S7 — Per-state ENR tuple wiring**:
- Replace `if epc_dt: g = min(g, 0.85)` with `state_tuple[filter_state].gain_cap`
- Replace `if divergence > 0.3: g = min(g, …)` with `state_tuple[filter_state].divergence_cap`
- Tuples (steady = `1.0` cap → preserves v3.11.x byte-equal):
  | state | gain_cap | divergence_cap | ENR aggression |
  |---|---|---|---|
  | idle | 1.0 | 1.0 | (bypass) |
  | startup | 1.0 | 0.5 | conservative |
  | transient | 0.85 | 0.3 | conservative |
  | recovering | 0.95 (ramp) | 0.5 | medium |
  | **steady** | **1.0** | **1.0 (effectively off)** | **= v3.11.x** |
  | diverged | 0.5 | 0.1 | conservative |

**Critical invariant** (anti-P58 trap): `steady` state must be byte-equal v3.11.x. The 4-cap chain action is preserved on `steady`; Phase 3 only adds per-state perturbation on non-steady states.

---

## Anti-trap checks (P50 / P52 / P55 / P58)

| Trap | How v3.12 S6-S9 avoids it |
|---|---|
| P50 (`nearend_protect_dt` → FS Δecho -1.328) | Evidence change is to mic-excess (AUROC 0.871), not to a new `effective_dt`-based gate. FS-vs-DT discrimination is moved upstream of the floor (via per-state filter_state). |
| P52 (cohort tail catastrophe via removing PathChangeRegimeHandler) | PathChangeRegimeHandler **untouched** in Phase 3. `transient` state's `gain_cap` mirrors the pause_main + boost_q effective behaviour (capping gain when filter is unreliable). |
| P55 (Enzner-Vary Wiener: DT-FS only +7.01 dB) | Wiener gain is unchanged in Phase 3; only the **floor** is unified. The discriminator (mic-excess) replaces only `(1-coh²)` evidence, not the gain compute itself. |
| P58 (retire 4-cap → FS Δecho -0.674) | 4-cap **action** preserved: `gain_cap` value 0.85 retained for `transient`. Phase 3 changes the **gate** (from scalar `epc_dt` to state-driven), not the cap value. |

---

## Fire-rate empirical follow-up (deferred to S6 if needed)

This audit is static (code reading + comment archaeology). If S6 design
decisions hinge on per-bucket fire-rate distributions, add diagnostic
counters (per-path `np.sum(mask)` increment per frame) to `AecStats`
and run 800-case in flag-ON mode. Decision deferred: do NOT instrument
unless S6 ambiguity surfaces; the categorisation above is already
sufficient to write S6 design code.

---

## Sources

- v3.10.5 dt_per_bin saturation comment: [aec.py:2347-2353](../python/aec.py#L2347-L2353)
- Q7 V3 verdict: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (Q7 section)
- F3.1 mic-excess AUROC 0.871: [docs/research_log_p55_phase1_verdict.md](research_log_p55_phase1_verdict.md)
- Beroutti 1984 / Ephraim-Malah 1984 / Hänsler-Schmidt canonical RES — see `~/.claude/plans/...hazy-lynx.md` Sources section
