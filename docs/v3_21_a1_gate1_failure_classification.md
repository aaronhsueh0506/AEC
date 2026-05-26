# v3.21 A.1 Gate 1 Failure Classification

**Date**: 2026-05-25  
**Source**: Gate 1 12-case AECMOS verdict (`docs/v3_21_a1_gate1_12case_verdict.md`)  
**Scope**: Read-only / trace-only analysis of structural failure causes. No algorithm patch, no 800-case, no v3.22 framing.  
**Status**: Gate 1 12-case AECMOS COMPLETE 2026-05-25 for **variant D (full AEC3 composition)**. Gate 0 wiring PASS. Gate 1 FAIL: all 3 DT_mvmt cases regress vs A baseline (xFk7 D−A=−0.706, ZJYUt0O0 D−A=−0.361, wVYSGVTT D−A=−0.110); 2 catastrophic. DT_static preserved (+0.339 mean). FS echo improved (−0.326/−0.351). **Category A classification now valid** — full AEC3 composition (delay-chain + UseLinearFilterOutput) insufficient to prevent DT_mvmt regression. R1/R3 functional adaptation requires user authorization. Verdict docs: `docs/v3_21_a1_combo_gate0_trace_verdict.md` (wiring) + `docs/v3_21_a1_d_12case_verdict.md` (behavioral).

---

## 0. Observed failure pattern

| Bucket | Δdeg/Δecho | Verdict |
|---|---|---|
| DT_static (4) | Δdeg = **+0.396** | Strong improvement |
| FS (4) | Δecho = **−0.092 / −0.112** | Slight improvement |
| DT_mvmt (3) | Δdeg = **−0.435 mean; worst −0.965** | Catastrophic regression |

DT_static and FS both benefit from the chain. DT_mvmt alone fails — and it fails
by a structural asymmetry, not noise. All 3 DT_mvmt cases regress; 0 DT_static cases
regress. The asymmetry is deterministic (trigger scope PASS, G5 byte-equal PASS, no
harness artifact).

---

## 1. Sub-component decomposition

The `use_full_delay_change_chain=True` implementation comprises 5 distinct sub-components.
Each is analyzed below for: strict AEC3 parity? / port semantic correctness? / plausible
DT_mvmt damage cause? / required for parity?

### 1.1 H_error reset + filter counter reset (AEC3 Steps 3–4)

**What fires (Sites 1 and 2):**
```
PBFDKF.handle_echo_path_change(delay_change=True):
    H_error_per_bin.fill(10000)          → _h_error_refresh clamps to 100 next frame
    call_counter = 0                      → filter freeze for 6 blocks (24 ms)
    poor_excitation_counter = 400         → skip-update gate arms (vs AEC3's 1000)

PBFDAF.handle_echo_path_change(delay_change=True):
    similar counter resets (no H_error; shadow is NLMS)
```

**Is it strict AEC3?** Yes. Mirrors `refined_gains_[ch]->HandleEchoPathChange()` in
`subtractor.cc`. H_error fill + call_counter=0 + poor_excitation=kPoorExcitationCounterInitial.
Our poor_excitation=400 vs AEC3's 1000 is a PBFDKF-adapted value, not a port error.

**Port semantically correct?** Yes. Gate 0 confirmed H_error ON=100.0 vs OFF=0.4 at
delay_first+1. W ratio at SR+10 = 0.9766 (< 1.05) — H_error reset alone does NOT
systematically diverge filter convergence vs OFF.

**Plausibly explains DT_mvmt damage?** Partially. H_error=100 (ceiling) = maximum Kalman
gain. During the post-freeze re-adaptation phase, the filter greedily adapts to whatever
is in the microphone. If active near-end speech is present, the filter starts tracking the
NE path instead of the echo path. However this effect is bounded by the 24 ms freeze
(during which the filter does not update). After the freeze, H_error drives more aggressive
adaptation than the OFF variant, but the primary output-domain damage requires ERLE to be
simultaneously blind (see §1.3).

**Required for v3.21 parity?** Yes. Core of `subtractor.cc:HandleEchoPathChange`. Cannot
be omitted from the parity chain.

**Classification: A — correct strict AEC3, contributes to DT_mvmt via max Kalman gain;
secondary damage vector.**

---

### 1.2 Leakage config split — Steps 5–6, Q1 (NOT IMPLEMENTED)

**What fires:** Nothing. Steps 5–6 (`SetConfig(refined_initial, true)` = leakage_converged
100×, leakage_diverged 10× for 2.5 s; `SetConfig(coarse_initial, true)` = rate 1.29×) are
explicitly deferred as the Q1 design decision. The design doc §3.3 documents Q1-B (always-elevated
steady leakage) as the current functional adaptation.

**Is it strict AEC3?** Yes — these steps exist in AEC3's full chain.

**Port semantically correct?** N/A — not implemented. Our always-elevated steady leakage
(leakage_converged ≈ 1e-3 = 20× AEC3 steady, but 5× below AEC3 transient) is intentionally
different.

**Plausibly explains DT_mvmt damage?** No — and in the wrong direction. If Q1-A (strict port)
were implemented, the leakage_converged 100× boost would cause *more* aggressive adaptation
post-delay_change, not less. The absence of Steps 5–6 makes DT_mvmt damage *smaller*, not larger.
Q1 absence does not explain the Gate 1 failure.

**Required for v3.21 parity?** Design doc defers to Q1 decision. Not needed to explain or fix
the current failure.

**Classification: D — optional/Q1-deferred, zero behavioral surface in current code. Not
a cause of Gate 1 failure.**

---

### 1.3 AecState._full_reset (AEC3 §1.3 steps 6a–6i) ← PRIMARY CAUSE

**What fires (via `_aec3_pending_delay_change = NEW_DETECTED_DELAY` queued in Site 1/2,
consumed in `_aec3_post` at line 4042):**

```
AecState._full_reset():
    _strong_not_saturated_render_blocks = 0     ← counter restart
    _blocks_with_active_render = 0
    _initial_state.reset():
        initial_state = True                    ← re-enter initial state
        _transition_triggered = False
        _strong_not_saturated_render_blocks = 0
    _filter_quality.reset()
    _capture_signal_saturation = False
    _erle_estimator.reset(delay_change=True)    ← ERLE full reset to min_erle
    _erl_estimator.reset()
    transparent_mode.reset()   (dormant under TM=OFF)
    filter_analyzer.reset()
```

**Is it strict AEC3?** Yes. Mirrors `aec_state.cc:143-171` exactly. All 9 sub-steps mapped
in design doc §1.3.

**Port semantically correct?** Yes. The delay_first AecState gap (previously _aec3_pending was
only set on delay_shift) was fixed by A.1 Site 1. `_full_reset` is correctly implemented and
the sub-step mapping is verified.

**Plausibly explains DT_mvmt damage?** YES — this is the primary structural cause.

The critical damage chain is:

```
delay_first fires during active DT speech (DT_mvmt case)
    ↓
AecState._full_reset():
    _erle_estimator.reset(delay_change=True) → ERLE = min_erle (≈ 1.0) for all bins
    _strong_not_saturated_render_blocks = 0  → re-enter initial_state, count up 625 hops
    ↓
ResidualEchoEstimator sees ERLE ≈ 1.0:
    suppression_ratio = linear_residual² / (ERLE × erl_estimate × …)
    ERLE=1.0 → denominator tiny → residual_estimate ≈ near_end + echo → no echo detected
    ↓
SuppressionGain:
    with no echo detected and RES seeing everything as near-end, gain approaches 1.0
    BUT: on real DT+echo: residual echo PSD is high (filter hasn't re-converged);
    ERLE-divide with ERLE=1.0 → echo_estimate blows up → suppression gain crushes everything
    ↓
DT speech during 625-hop (2.5 s) re-convergence window is strongly suppressed
```

The exact ERLE→RES path depends on pipeline configuration, but the core is:
`erle_estimator.reset()` sets per-subband ERLE to `min_erle` (1.0 = no cancellation).
When ERLE is 1.0, the SubbandErleEstimator provides no cancellation signal to
ResidualEchoEstimator. The RES then cannot distinguish echo from near-end during the
re-convergence window → applies aggressive suppression → DT speech damaged.

**Why DT_static avoids this:** In DT_static cases, delay_first fires at room start (frame ~100),
before doubletalk speech begins (DT speaker typically enters several seconds later). During the
2.5 s re-convergence window, the only signal is far-end (reference echo). ERLE rebuilds from
far-end content. By the time DT speaker starts talking, ERLE has partially recovered → suppression
is not blind → DT speech preserved.

**Why DT_mvmt is damaged:** The `with_movement` variant fires delay changes during active
doubletalk. `delay_first` at frame ~104 with DT speech already present. ERLE reset fires
immediately into active DT → suppression blind for up to 2.5 s → speech destroyed.

**Required for v3.21 parity?** Yes — the full AecState reset is strict parity. Cannot be
omitted without documenting a functional adaptation.

**Missing companion safety valve in our implementation:** In AEC3, the full chain is designed
to operate alongside `UseLinearFilterOutput` (v3.21.13 parity item, currently PAUSED CANDIDATE).
When `initial_state_active=True` (re-convergence window post-delay_change), AEC3's
`echo_remover.cc:475-480` checks `usable_linear_estimate`. If the filter is poorly adapted
(as it always is after `_full_reset`), `usable_linear_estimate=False` → output path selects
Y (mic + echo) instead of E (linear residual). This Y-path selection preserves DT speech
because Y contains both echo AND near-end signal, but SuppressionGain applied to Y correctly
attenuates only the echo component. Without `UseLinearFilterOutput`, suppression sees only E
(which is highly distorted after the blind ERLE period) → DT speech damaged.

**Classification: A — correct strict AEC3, primary structural cause of DT_mvmt regression.
Damage is not from a wrong port but from applying AEC3's chain without AEC3's companion
output-path safety valve (UseLinearFilterOutput).**

---

### 1.4 SuppressionGain.set_initial_state(True) — Step 9 ON

**What fires (Site 3):**
```
if delay_change was pending and use_full_delay_change_chain:
    _aec3_sg.set_initial_state(True)    → sets initial_state_ = True
    _diag['a1_set_initial_state'] = 'on'
```

**Is it strict AEC3?** Yes. `echo_remover.cc:404-407`.

**Port semantically correct?** Yes.

**Plausibly explains DT_mvmt damage?** No. A.2 closure explicitly verified: zero behavioral
effect under default config (`lf_smoothing_during_initial_phase=True`, `use_during_initial_phase=True`
— both consumers are no-ops under these defaults). This call is a parity placeholder.

**Required for v3.21 parity?** Yes, mechanically — the call site is required for correct
parity. Zero impact on audio output.

**Classification: B — mechanically correct, zero behavioral surface at default config.
Not a cause of Gate 1 failure.**

---

### 1.5 set_initial_state(False) on TransitionTriggered — Step 9 OFF

**What fires (Site 4):**
```
if use_full_delay_change_chain and aec3_state.transition_triggered():
    _aec3_sg.set_initial_state(False)
    _diag['a1_set_initial_state'] = 'off'
```

**Is it strict AEC3?** Yes. `echo_remover.cc:418-421`.

**Port semantically correct?** Yes. `transition_triggered()` fires on the edge when
`_strong_not_saturated_render_blocks` crosses the initial_state_hops threshold. Exactly
one frame edge = exactly one `set_initial_state(False)` call.

**Plausibly explains DT_mvmt damage?** No. Same A.2 closure — zero behavioral effect.
The ExitInitialState call also does not revert leakage config (Steps 5–6 not implemented;
Q3-B selected).

**Required for v3.21 parity?** Yes, mechanically. Zero audio impact.

**Classification: B — mechanically correct, zero behavioral surface. Not a cause of Gate 1
failure.**

---

## 2. Attribution summary

| Sub-component | Category | Port correct? | DT_mvmt cause? | Parity required? |
|---|---|---|---|---|
| H_error reset / counter reset (Steps 3–4) | A | Yes | Secondary (max Kalman gain during active DT) | Yes |
| Leakage config split (Steps 5–6, Q1) | D | N/A (missing, deferred) | No (absence reduces damage) | Q1-decision |
| AecState._full_reset (Steps 6a–6i) | A | Yes | **PRIMARY** (ERLE reset → blind suppression) | Yes |
| set_initial_state(True) (Step 9 ON) | B | Yes | No (A.2 zero effect) | Yes (parity placeholder) |
| set_initial_state(False) + TransitionTriggered | B | Yes | No (A.2 zero effect) | Yes (parity placeholder) |

**Gate 1 failure = Incomplete AEC3 composition.** Gate 1 tested:
- delay-change reset/state chain (`use_full_delay_change_chain=True`) — sub-chain 1
- WITHOUT `use_linear_filter_output_selection_for_final_output` — sub-chain 2 absent

AEC3 production path in `echo_remover.cc:475` couples both:
`Y_fft = aec_state_.UseLinearFilterOutput() ? E : Y`

When `usable_linear_estimate=False` (always true during initial_state re-convergence window),
AEC3 applies suppression to Y, not E. Y preserves DT speech. Gate 1 never activated this
companion → classifying A.1 as "Category A structurally incompatible" is premature.
**A.1 cannot close. Combo D = A.1 + UseLinearFilterOutput must be traced first.**

---

## 3. Structural asymmetry: why DT_static improves and DT_mvmt regresses

The delay-change chain resets the ERLE estimator to baseline (ERLE=1.0) and re-enters
initial_state (re-convergence window of 2.5 s / 625 hops).

**DT_static timeline:**
```
t=0.4s   delay_first fires (room start; no DT speech yet)
         → H_error reset, AecState._full_reset, ERLE=1.0
t=0.4–2.9s  re-convergence window
         → only far-end signal present; ERLE rebuilds from FS content
         → H_error re-converges to steady state
t=~3s    DT speaker begins
         → ERLE is partially/fully recovered; suppression not blind
         → DT speech preserved; filter well-adapted to new delay path
```

**DT_mvmt (with_movement) timeline:**
```
t=0.4s   delay_first fires
         → H_error reset, AecState._full_reset, ERLE=1.0
t=0.4s   DT speech ALREADY ACTIVE (movement scenarios have DT+movement)
         → ERLE=1.0: suppression blind immediately
         → H_error=100: max Kalman gain during active DT speech
         → filter adapts greedily to NE speech in mic
         → RES/SuppressionGain with blind ERLE crushes both echo and DT speech
t=0.4–2.9s  re-convergence window under active DT
         → damage accumulates for up to 2.5 s
```

**FS benefit:** Delay path changes affect echo-only signal. No near-end speaker.
H_error reset + ERLE reset → faster re-adaptation to new echo path → echo improves.
No DT speech to damage.

---

## 4. The missing companion safety valve

AEC3 uses the full chain together with `UseLinearFilterOutput` (AEC3 default: `coarse_filter_output`
= `use_coarse_filter_output_=true` in `echo_remover.cc:430-444`, which provides a Y-path
bypass when linear filter is not usable).

In AEC3:
```
if usable_linear_estimate:
    apply gain to E (linear residual)     ← converged path
else:
    apply gain to Y (mic - echo_path)     ← fallback path during re-convergence
```

When `initial_state_active=True` after `_full_reset`, the filter is poorly adapted →
`usable_linear_estimate=False` → output uses Y-path → DT speech in Y is preserved
(suppression gain applied to Y attenuates echo component while NE speech passes through).

Our implementation:
- `UseLinearFilterOutput` = v3.21.13 PAUSED CANDIDATE (not default-enabled)
- Without it: suppression always uses E. During re-convergence with ERLE=1.0, E contains
  distorted echo + NE speech, and the suppression gain is applied to the wrong signal.

This explains why AEC3 can tolerate the full chain while our implementation cannot:
AEC3 uses the Y-path fallback during the re-convergence window. We always use the E-path.

---

## 5. Remediation candidates (analysis only — not authorized)

No implementation is authorized until user direction is given. The following candidates
are identified purely to scope the problem space:

| # | Approach | Mechanism | DT_static preserved? | FS preserved? | AEC3 parity? |
|---|---|---|---|---|---|
| R1 | Gate H_error + ERLE reset on NE energy at delay_first | Skip AecState._full_reset (or skip ERLE sub-step) if NE energy > threshold at delay_first | YES (fires normally when no NE) | YES | Functional adaptation (not strict) |
| R2 | Limit full chain to delay_shift only | delay_first fires chain only for H_error reset; AecState._full_reset only on delay_shift | NO (loses DT_static benefit) | PARTIAL | Partial parity; delay_first AecState gap documented |
| R3 | Skip ERLE/ERL reset sub-steps only (keep counter reset + initial_state reset) | Remove `_erle_estimator.reset()` + `_erl_estimator.reset()` from _full_reset on delay_first; keep `_strong_not_saturated_render_blocks=0` + `_initial_state.reset()` | YES (filter adapts; suppression not blind) | YES | Functional adaptation; split §1.3 steps |
| R4 | Enable UseLinearFilterOutput (v3.21.13) as companion | Port AEC3's Y-path fallback when `usable_linear=False`; preserves DT speech during re-convergence | YES | YES | Strict AEC3 parity path |
| R5 | Close A.1 no-ship — document as AEC3-incompatible without companion UseLinearFilterOutput | No code change; document the structural dependency | N/A | N/A | Closure with rationale |

**Note on R4 (priority action — required before A.1 verdict):** v3.21.13 isolation results
(PAUSED, DT_static Δdeg −0.335 for E-alone vs OFF) cannot be applied to the A.1 companion
scenario. In isolation, UseLinearFilterOutput fires on all frames where `usable_linear=False`
— including non-delay-change frames. In the A.1 combo (variant D), the Y-path companion is
specifically relevant during the `initial_state_active` re-convergence window after `delay_first`.
The interaction must be tested as D = A.1 + UseLinearFilterOutput via Gate 0 trace first.
PAUSED status of v3.21.13 does not apply to the combo — it must be re-evaluated in this context.

**Note on R1/R3:** Both require a NE energy signal at the delay_first frame. The orchestrator
has access to near-end energy estimates via the DT detector. This is a targeted functional
adaptation, not a parity gap.

---

## 6. Failure category verdict

**REVISED 2026-05-25** — Gate 1 tested an incomplete AEC3 composition.

| Category | Description | Status |
|---|---|---|
| A | Correct strict AEC3 behavior, incompatible with production objective | **PENDING** — requires full composition (D) test before determination |
| B | Wrong or incomplete port of one sub-component | **No** — all A.1 sub-components semantically correct |
| C | delay_first semantics differ from AEC3 design intent | **No** — trigger scope PASS; timing matches AEC3 design |
| Incomplete composition | Gate 1 activated sub-chain 1 (delay-change cascade) without sub-chain 2 (UseLinearFilterOutput Y-fallback) | **YES — correct characterization of Gate 1 test** |

**Current verdict: Gate 1 tested an incomplete AEC3 composition.**

AEC3 production `echo_remover.cc` couples two inseparable sub-chains:

```
Sub-chain 1 (delay-change cascade):
  subtractor.HandleEchoPathChange → H_error reset + counter reset
  aec_state.HandleEchoPathChange  → AecState._full_reset
  suppression_gain.SetInitialState → initial_state placeholder

Sub-chain 2 (output-path Y/E selection):
  Y_fft = UseLinearFilterOutput() ? E : Y   ← echo_remover.cc:475
  UseLinearFilterOutput() = LinearFilterUsable() && config.filter.use_linear_filter
```

Gate 1 enabled sub-chain 1. Sub-chain 2 was absent. During `initial_state_active=True`
(re-convergence window), AEC3's `usable_linear_estimate=False` → AEC3 selects Y-path → DT
speech passes through. Without sub-chain 2: suppression always uses E. ERLE=1.0 + E-path →
DT speech damaged. This is the Gate 1 failure mechanism.

**Sub-component attribution (not changed by revision):**
- Damage vector: AecState._full_reset → ERLE=1.0 → blind suppression → DT damaged
- Missing companion: UseLinearFilterOutput Y-path during `initial_state_active` window
- Port errors: None — all A.1 sub-components are semantically correct ports

**A.1 cannot close. Combo D = A.1 + `use_linear_filter_output_selection_for_final_output`
must be traced via Gate 0 before any verdict on Category A is issued.**

---

## 7. AEC3 full composition requirement

AEC3 `echo_remover.cc` couples two sub-chains that are not independently valid:

```
AEC3 delay-change behavior         Current implementation status
─────────────────────────────      ──────────────────────────────
subtractor.HandleEchoPathChange    ✓ A.1: H_error reset + counter reset
aec_state.HandleEchoPathChange     ✓ A.1: AecState._full_reset (ERLE/ERL/counters)
suppression_gain.SetInitialState   ✓ A.1: set_initial_state (zero effect per A.2)
UseLinearFilterOutput Y-fallback   ✗ NOT in A.1; use_linear_filter_output_selection_for_final_output
                                     = v3.21.13 PAUSED (isolation verdict does not apply to combo)
```

The Y-fallback (`Y_fft = aec_state_.UseLinearFilterOutput() ? E : Y`) is AEC3's built-in
safety mechanism for delay-change re-convergence windows. AEC3's aggressive delay-change
cascade was designed with this companion in the same processing unit. Activating sub-chain
1 alone is not a valid test of AEC3's production intent.

**Required next step:** Variant D = `use_full_delay_change_chain=True` +
`use_linear_filter_output_selection_for_final_output=True`. Gate 0 trace must confirm that
D actually selects Y-path during `initial_state_active` window on DT_mvmt damage cases before
12-case AECMOS is authorized.

Functional adaptations R1/R3 (NE-energy gate on ERLE reset, skip ERLE sub-steps) are NOT
authorized until D is traced and evaluated. Only if D fails Gate 0 or 12-case does
functional adaptation become authorized.

---

## 8. Next test plan — Gate 0 trace for full AEC3 composition (Variant D)

### 8.1 Variants

| Variant | Config | Purpose |
|---|---|---|
| A | baseline OFF (both flags False) | Control; byte-equal reference |
| B | `use_full_delay_change_chain=True` only | Gate 1 result — reuse existing data |
| C | `use_linear_filter_output_selection_for_final_output=True` only | v3.21.13 isolation — reference only; do not re-run unless data gap |
| D | `use_full_delay_change_chain=True` + `use_linear_filter_output_selection_for_final_output=True` | **Full AEC3 composition — primary Gate 0 target** |

### 8.2 Cases

Must include all 3 Gate 1 DT_mvmt fail cases:

| Case | Gate 1 Δdeg | Priority |
|---|---|---|
| `xFk7igecuke0R5` | −0.965 | P1 (worst) |
| `wVYSGVTTakih9t` | −0.268 | P2 (G2 catastrophic) |
| `ZJYUt0O0AEKSQ9` | −0.071 | P3 (G1 fail only) |

### 8.3 Required trace fields (per frame, all variants)

| Field | Purpose |
|---|---|
| `delay_first` frame index | ERLE reset onset — zero-frame reference |
| `delay_shift` frame index(es) | Secondary reset events |
| `aec3_initial_state_active` | Re-convergence window flag |
| `usable_linear_estimate` | FilterQuality gate signal |
| `a1_ulo_output_base` | Final output path selected: `E` or `Y` |
| `aec3_erle` (mean or per-subband) | ERLE recovery trace post-reset |
| `aec3_erl` | ERL state |
| `suppression_gain` mean | RES attenuation level |
| Near-end energy | DT speech activity indicator |
| E energy | Linear residual energy |
| Output energy | Post-suppression output level |

### 8.4 Gate 0 acceptance criteria for Variant D

**PASS → proceed to 12-case AECMOS:**
1. `a1_ulo_output_base=Y` fires on at least 1 frame during `aec3_initial_state_active=True` window for each DT_mvmt case.
2. Output energy in D is higher than B during the DT speech window (Y-path reduces over-suppression).
3. `aec3_initial_state_active` onset aligns with `delay_first` frame.
4. D byte-equal to A when both flags OFF.

**FAIL actions:**
- If D never selects Y (`a1_ulo_output_base=E` always): fix `use_linear_filter_output_selection_for_final_output` wiring for the A.1 combo before 12-case.
- If D selects Y but output still damaged: reclassify A.1 as true Category A (full AEC3 composition structurally incompatible with PBFDKF; then authorize functional adaptation R1/R3).

### 8.5 Forbidden in this round

- No 800-case bench.
- No v3.22 framing.
- No R1/R3 functional adaptation (NE gate on ERLE reset, skip ERLE sub-steps) — not authorized until D verdict known.
- Do not close A.1.
- Do not apply v3.21.13 isolation PAUSED verdict to D — combo must be re-evaluated independently.

---

## 9. Source references

| Reference | Location |
|---|---|
| AEC3 full chain dispatch | `docs/aec3_extracts/src/aec3/echo_remover.cc:385-421` |
| AecState._full_reset | `python/modules/state/aec_state.py:360-375` |
| handle_echo_path_change consumer | `python/modules/state/aec_state.py:219-224` |
| InitialState.reset() + transition_triggered() | `python/modules/state/initial_state.py:38-61` |
| A.1 Site 1 (delay_first H_error + _aec3_pending) | `python/modules/orchestrator.py:1902-1913` |
| A.1 Site 2 (delay_shift H_error) | `python/modules/orchestrator.py:1991-2000` |
| A.1 Site 3 (set_initial_state True) | `python/modules/orchestrator.py:4046-4050` |
| A.1 Site 4 (set_initial_state False on TT) | `python/modules/orchestrator.py:4093-4097` |
| Gate 1 AECMOS results | `docs/v3_21_a1_gate1_12case_verdict.md` |
| Gate 0 trace verdict | `docs/v3_21_a1_gate0_trace_verdict.md` |
| Design doc (Q1, remediation options) | `docs/v3_21_a1_delay_change_chain_design.md` |
| UseLinearFilterOutput verdict (paused) | `docs/v3_21_13_use_linear_filter_output_verdict.md` |
| A.2 closure (set_initial_state zero effect) | `docs/v3_21_a2_suppression_gain_initial_state_audit.md` |
