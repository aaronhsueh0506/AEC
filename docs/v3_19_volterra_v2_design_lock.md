# v3.19 Volterra V.2 — design lock (2026-05-16)

**Status**: Doc-only deliverable for v3.19 Phase 0.1 §5 auto-trigger
Option (B) fold-in. Locks v3.20 Volterra arc design parameters.
Consumes V.1 literature recommendation.

**Hard rule**: This is **doc-only** for v3.19. Code changes happen in
v3.20 dedicated cycle.

## 1. Mechanism (chosen approach from V.1 §9)

**Approach**: Stenger-Kellermann adaptive memoryless polynomial
preprocessor + parallel Hammerstein 3-channel linear NLMS (joint
identification).

### 1.1 Architecture

```
ref ─► HPF ─► [Volterra Preprocessor] ─► DelayEst+RingBuf ─► PBFDKF ─► error
                       │                                         │
                       │ adaptive polynomial f_NL(x)              │
                       │ + parallel h_2, h_3 channels             │
                       │                                          │
                       └──── error feedback for joint update ─────┘
```

The Volterra Preprocessor replaces the static `Saturation` block in
the existing pipeline (CLAUDE.md `mic → HPF → Saturation → DelayEst → PBFDKF`).

### 1.2 Kernel parameters

| Parameter | Value | Rationale |
|---|---|---|
| Polynomial order p | **3** | Captures odd (clipping) + even (asymmetry) NL components per V.1 §1.2 |
| Memory length M | **128 samples** | Matches one hop, fits one PBFDKF block boundary, V.1 §1.3 anchor |
| Sample rate | 16 kHz | Project standard |
| Effective NL window | 8 ms | M / fs = 128 / 16000 |
| State size (kernel) | 3 × M = 384 floats | 3 Hammerstein channels, each M-tap linear filter |
| State size (polynomial coeffs) | 3 floats `(a_1, a_2, a_3)` | Static memoryless polynomial |

### 1.3 Update equations

Build extended regressor at sample n:

```
x_p(n) = (a_1 x(n) + a_2 x(n)² + a_3 x(n)³)   # polynomial preprocessor output
y_hat(n) = Σ_{i=0}^{M-1} h_1(i) x_p(n-i)        # linear filter on preprocessed
e(n) = d(n) − y_hat(n)                          # error vs mic
```

Joint NLMS update (one step per sample, computed once per hop):

```
# Linear filter update (existing PBFDKF mechanism — unchanged)
h_1 ← h_1 + μ_linear · x_p(n-0..M-1) · e(n) / (||x_p||² + δ)

# Polynomial coeff update (NEW — parallel-channel stacked NLMS)
gradient_a_p = Σ_{i=0}^{M-1} h_1(i) · x(n-i)^p · e(n)
a_p ← a_p + μ_NL · gradient_a_p / (||x_p||² + δ + ε_a)
```

`δ` = existing PBFDKF regulariser; `ε_a` = additional polynomial
regulariser (suggested 1e-6 to prevent runaway when far-end is silent).

### 1.4 Step size

```
μ_linear = existing PBFDKF μ (unchanged; preset-dependent)
μ_NL = μ_linear / 10
```

Per V.1 §2.3 universal consensus. Conservative ratio 1/10 anchors
inside the stable operating regime regardless of far-end signal
statistics. AecConfig should expose `volterra_mu_scale_factor = 0.1`
to allow ablation.

### 1.5 Initialisation

```
a_1 = 1.0       # Pass-through linear at startup
a_2 = 0.0       # Zero NL contribution
a_3 = 0.0
h_1 = zeros(M)  # Cold-start linear filter (same as existing PBFDKF)
```

When `volterra_enabled = False` (default), `a_1 = 1.0` constant and
`a_2 = a_3 = 0` constant — preprocessor is identity. This is the
byte-equal guarantee for ship-OFF builds.

### 1.6 Convergence indicator

Per V.1 §2.4:

```
volterra_converged = (||a_2||² + ||a_3||²) / a_1² > 1e-3
                     AND streak_length(volterra_active) > 32 hops
                     AND error_spectrum_flatness > 0.85
```

Three-condition AND: meaningful NL energy captured, sustained over ~250 ms,
residual spectrum approximately white.

## 2. Integration with existing architecture

**Chosen insertion point**: **Option (a) preprocessor — replace existing
Saturation block in reference path**.

### 2.1 Why option (a) over (b) / (c)

Option (a) — replace Saturation block (reference path, pre-PBFDKF):

- **PRO**: Stenger-canonical position. Matches V.1 §4 directly.
- **PRO**: The static Saturation block is currently a fixed `tanh`
  curve; replacing with adaptive polynomial generalises it without
  adding a new pipeline stage.
- **PRO**: Effect propagates through ENTIRE downstream pipeline
  (PBFDKF, ShadowFilter, ResFilter, CNG) — single insertion point
  benefits all consumers.
- **PRO**: C port path is straightforward — Saturation block already
  exists in `c_impl/`; making it adaptive is well-scoped.
- **CON**: Adapts on FS-confident frames only (NE contamination if
  adapted during DT). Gating burden falls on the preprocessor.

Option (b) — pre-RES, post-PBFDKF (NL inverse on error signal):
- **CON**: Operates on residual after linear cancellation; signal SNR
  is much lower. NLMS convergence is poor at low SNR.
- **CON**: Cannot remove NL added at the reference (loudspeaker NL is
  *upstream* of the mic; post-PBFDKF NL inverse can only model the
  remaining mic-side codec NL).
- **CON**: Requires brand-new pipeline stage; not a drop-in.

Option (c) — inside ResFilter as a new `_stage_nl_inverse`:
- **CON**: ResFilter operates on frequency-domain magnitudes; Volterra
  is fundamentally time-domain.
- **CON**: ResFilter's 9-stage pipeline is already tightly coupled
  (per CLAUDE.md `feedback_holistic_trace_before_change.md`); adding
  Volterra inside RES violates the v3.16 RES refactor anti-loophole.

**Decision**: Option (a). Aligns with Stenger 2000 canonical placement
and matches the existing Saturation block surface.

### 2.2 Exact `aec.py` insertion points

Based on inspection (the AEC pipeline diagram in CLAUDE.md and existing
references):

| File | Line range (approx) | Change kind |
|---|---|---|
| `python/aec.py` | new class `VolterraPreprocessor` near `SubtractiveNLP` (~L5764) | New class definition |
| `python/aec.py` | `AEC.__init__` (~L7368 area near other detector wiring) | Instantiate VolterraPreprocessor if `volterra_enabled` |
| `python/aec.py` | reference saturation step inside `AEC.process()` | Replace `static_saturation(ref)` with `volterra_preprocessor.process(ref, error_feedback)` when enabled |
| `python/aec.py` | `AecState` class (~L5245) | Add 3 new getter methods (see §3) |
| `python/aec.py` | `AecConfig` (~L146) | Add 6 new config fields (see §5) |

C port (`c_impl/`) follows once Python algorithm change has merged per
CLAUDE.md branch model.

## 3. State integration via AecState back-ref

Per CLAUDE.md `feedback_holistic_trace_before_change.md` ("trace upstream
+ downstream before changing"), Volterra state must be readable by
C.A FilterAnalyzer / C.B FilteringQualityAnalyzer / detectors. Spec
the new AecState methods:

### 3.1 New AecState methods (additive only)

```python
def volterra_active(self) -> bool:
    """Returns True if VolterraPreprocessor is enabled AND currently adapting."""
    a = self._aec_ref
    if a is None or getattr(a, '_volterra_preprocessor', None) is None:
        return False
    return bool(a._volterra_preprocessor.is_active)

def volterra_converged(self) -> bool:
    """Returns True when 3-condition convergence (§1.6) is met for current frame."""
    a = self._aec_ref
    if a is None or getattr(a, '_volterra_preprocessor', None) is None:
        return False
    return bool(a._volterra_preprocessor.converged)

def volterra_state_dump(self) -> dict:
    """Per-frame snapshot for trace + audit.

    Returns {
      'active': bool,
      'converged': bool,
      'a_1': float, 'a_2': float, 'a_3': float,
      'h_1_norm': float,
      'nl_energy_ratio': float,           # (||a_2||² + ||a_3||²) / a_1²
      'error_spec_flatness': float,
      'mu_NL_effective': float,
      'fire_streak': int,                  # hops since last adapt-gate fire
    }
    """
    a = self._aec_ref
    if a is None or getattr(a, '_volterra_preprocessor', None) is None:
        return {}
    return a._volterra_preprocessor.state_dump()
```

### 3.2 Aec3-snapshot extension

Extend `aec3_snapshot()` (aec.py:5359-5363) to include:

```python
'volterra_active': self.volterra_active(),
'volterra_converged': self.volterra_converged(),
```

This is **additive**; existing snapshot consumers continue to work.

## 4. Detector reuse — when does Volterra adapt vs freeze?

### 4.1 Adapt-gate (preprocessor coefficients ARE updated)

Three-condition AND for stable adaptation:

```
adapt_gate = aec_state.fq_usable()
             AND aec_state.consistent_estimate()
             AND (e5_s3_mic_lpb_correlation_r > 0.35
                  OR subtractive_nlp.nl_confidence > 0.4)
```

Rationale:
1. `fq_usable()` (C.B): main filter has produced trustworthy estimate
   (52-86% bucket coverage per v3.18 C.E closeout). NL adaptation
   requires a usable linear baseline.
2. `consistent_estimate()` (C.A): filter impulse-response shape is
   stable. Volterra adaptation will diverge if linear filter is still
   chasing a moving target.
3. E5.S3 correlation OR E4.S2 pitch tracker: at least one NL detector
   is firing — there IS something for Volterra to identify. Avoids
   wasting adaptation cycles on NL-free FS-clean frames.

### 4.2 Freeze-gate (preprocessor coefficients are FROZEN)

```
freeze_gate = NOT adapt_gate
              OR aec_state.dt_combined > 0.3
              OR aec_state.saturated_capture()
              OR aec_state.transparent_mode_active()
```

Rationale:
1. NOT adapt_gate: default freeze when prerequisites unmet.
2. DT detected: NE speech contaminates error signal → polynomial coeffs
   would drift toward modelling NE voice as NL. CRITICAL freeze.
3. Saturated capture: existing F-E5 protection signal; capture-side
   clipping makes the error signal unreliable for any adaptive update.
4. Transparent mode: no echo to cancel → no NL to identify.

### 4.3 Pass-through always (even when frozen)

When frozen, the preprocessor still **applies** the current coefficients
(`x_p = a_1 x + a_2 x² + a_3 x³`) — it just doesn't update them. This
preserves whatever NL inverse was learned during prior adaptation
windows.

## 5. AecConfig surface — new flags

| Flag | Default | Description |
|---|---|---|
| `volterra_enabled` | `False` | Master enable; OFF preserves byte-equal with current production |
| `volterra_kernel_order` | `2` | Polynomial order p; 2 = up to `x²`, 3 = up to `x³`. v3.20 lock = 3 per V.1 §1.2 and §1.3 |
| `volterra_memory` | `128` | Memory length M in samples; locked at 128 per V.1 §1.3 |
| `volterra_mu_scale_factor` | `0.1` | μ_NL = μ_linear × this factor (V.1 §2.3 / §1.4) |
| `volterra_adapt_fq_usable` | `True` | Require `aec_state.fq_usable()` in adapt-gate (§4.1) |
| `volterra_adapt_consistent_estimate` | `True` | Require `aec_state.consistent_estimate()` in adapt-gate (§4.1) |
| `volterra_adapt_nl_detector_threshold` | `0.4` | E4.S2 nl_confidence threshold for adapt-gate |
| `volterra_freeze_dt_threshold` | `0.3` | `dt_combined` threshold for freeze-gate (§4.2) |
| `volterra_convergence_min_streak` | `32` | Min hop streak for converged flag (§1.6) |
| `volterra_convergence_flatness_threshold` | `0.85` | Min error-spectrum flatness for converged flag (§1.6) |

Env-var override pattern (consistent with v3.18 C.E):
`AEC_VOLTERRA_ENABLED`, `AEC_VOLTERRA_MU_SCALE`, etc.

## 6. Hard bar pre-statement for v3.20 Volterra ship

Per v3.13 E4.S6 closure pattern: subjective listening gate is **load-bearing
acceptance signal** (linear ERLE can be metric-positive but
perceptually neutral). Hard bars:

### 6.1 Listen verification (load-bearing — primary acceptance)

**5/5 v3.13 NL cohort produces audible NL reduction** when rendered
with `volterra_enabled = True` (preset BALANCED, all gates default).

Specifically:
- Type 1 case (Gsy0lC5): user verdict shifts from "喇叭推爆" to either
  "輕微推爆" or "正常"
- 4 × Type 2 cases (9xjhiFb, IrQvqOTC_mvmt, WTdBhX, m4789f): user
  verdict shifts from "失真/非線性/無線電" to "正常" or "可接受"

This is the gate that v3.13 E4.S6a/S6b FAILED at; passing it is the
canonical breakthrough validation.

### 6.2 800-case AECMOS (secondary, must-not-regress)

| Bucket | Hard bar |
|---|---|
| FS_static | Δecho ≥ -0.010 AND Δdeg ≥ -0.005 |
| FS_movement | Δecho ≥ -0.010 AND Δdeg ≥ -0.005 |
| DT_static | Δecho ≥ -0.010 AND Δdeg ≥ -0.005 |
| DT_movement | Δecho ≥ -0.010 AND Δdeg ≥ -0.005 |
| NE | Δdeg ≥ -0.005 |

Tighter than typical v3.18 hard bars because Volterra's primary defence
is preprocessor adaptation, which can corrupt the linear path if
gating fails.

### 6.3 Cohort tail invariant

`qNvSMyU` (FS_static cohort tail): Δecho ≥ -0.020 (existing P52
PathChangeRegimeHandler tail-defence invariant). Volterra cannot
regress this case beyond noise.

### 6.4 xrtntuju 5-clip invariant

DT RES NE-damage 5-clip regression listening (per CLAUDE.md memory
`project_xrtntuju_regression_clip.md`): Volterra cannot regress any of
the 5 positive DT windows.

## 7. Sprint plan for v3.20

Total estimated LOE: 30-50 sprints across 4 phases. Detailed plan-doc
to be written at v3.20 cycle kickoff; this section enumerates the
gate-bearing milestones.

### Phase 0 (3-5 sprints) — Substrate prep + design lock validation

| Sprint | Deliverable | Gate |
|---|---|---|
| 0.1 | VolterraPreprocessor skeleton class (default-OFF; identity pass-through) | byte-equal 5-case |
| 0.2 | AecConfig flags + AecState getter methods | byte-equal 5-case |
| 0.3 | Hammerstein math unit test (synthesised toy NL, ideal SNR) | unit pass |
| 0.4 | Adapt-gate + freeze-gate plumbing (no coefficient updates yet) | byte-equal 800-case (flag-OFF) |

**§0.7 user auth gate**: Phase 0 → Phase 1 transition (per CLAUDE.md
plan-doc conventions, user authorises move from substrate prep to live
adaptation).

### Phase 1 (8-12 sprints) — Adaptation on synthesised cohort

| Sprint | Deliverable | Gate |
|---|---|---|
| 1.1 | Stenger NLMS coefficient update wired (synthesised tanh×3 corpus) | Convergence proof: μ_NL stable; a_2/a_3 ≠ 0 within 5 sec |
| 1.2 | Joint update with PBFDKF (synthesised corpus) | Combined ERLE > linear-only baseline + 5 dB |
| 1.3 | NLMS-Volterra direct-sum 2nd-order variant (Yang 2014) — comparison vs Hammerstein | Pick winner: Hammerstein OR full Volterra |
| 1.4 | Numerical stability sweep (μ_NL ablation 0.01 / 0.1 / 0.5) | Stable across full range; no divergence on long runs |
| 1.5 | Full 5/5 cohort isolated bench (real NL, not synth) | ERLE > +3 dB / 5 cases ON real cohort |

**§0.7 user auth gate**: Phase 1 → Phase 2 (move from synth + isolated
bench to real-cohort listen verification). Listen failure here closes
v3.20 cycle.

### Phase 2 (5-8 sprints) — Cohort listen verification

| Sprint | Deliverable | Gate |
|---|---|---|
| 2.1 | Render listen materials (5/5 cohort + v3.18 candidates per V.3) | All 5 + new candidates have `_volterra_on.wav` files |
| 2.2 | User listen verdict v1 (baseline μ_NL_scale = 0.1) | ≥ 3/5 cohort audibly improved |
| 2.3 | μ_NL sweep listen (0.05 / 0.1 / 0.2 / 0.5) | Identify production μ_NL |
| 2.4 | Edge-case listen (NE + xrtntuju 5-clip) | No regressions |
| 2.5 | Listen verdict consolidation | 5/5 cohort improved OR fall-back to V.1 §9 alternative (Yang 2014 2nd-order full Volterra) |

**§0.7 user auth gate**: Phase 2 → Phase 3 (move from listen to
800-case bench). Listen failure here triggers fallback path.

### Phase 3 (5-8 sprints) — 800-case bench + tuning

| Sprint | Deliverable | Gate |
|---|---|---|
| 3.1 | 800-case AECMOS A/B (BALANCED, volterra ON vs OFF) | All hard bars §6.2 pass |
| 3.2 | Cohort-tail (qNvSMyU) check | §6.3 pass |
| 3.3 | xrtntuju 5-clip regression listen | §6.4 pass |
| 3.4 | Per-bucket gain ratios + adapt-gate fire-rate audit | Fire-rate sane: 5-30% on FS, 0-2% on NE |
| 3.5 | Preset wiring (BALANCED default ON; other presets opt-in) | Preset-wise A/B for SOFT / AGGRESSIVE |

**§0.7 user auth gate**: Phase 3 → Phase 4 (move from research bench
to production merge). Any hard-bar failure here closes v3.20.

### Phase 4 (4-6 sprints) — C port + final acceptance

| Sprint | Deliverable | Gate |
|---|---|---|
| 4.1 | C `volterra_preprocessor.c` skeleton | Python↔C byte-equal at flag-OFF (5-case sample) |
| 4.2 | C adaptive update (NLMS-stacked) | Python↔C byte-equal at flag-ON (5-case sample, within atol=1e-6) |
| 4.3 | C 800-case parity check | byte-equal full 800 |
| 4.4 | Documentation + verdict doc | docs/v3_20_volterra_ship_verdict.md |
| 4.5 | __version__ bump + CHANGELOG | v3.20.0 tag candidate |

**§0.7 user auth gate**: Phase 4 → merge-to-main (production
authorisation per CLAUDE.md branch model).

### Cycle-level kill criteria

Any of these triggers v3.20 close-without-ship (substrate retained
default-OFF):

1. Phase 1.5 isolated bench < +3 dB ERLE on real cohort
2. Phase 2.2 listen verdict < 3/5 cohort improved (even after fallback
   to Yang 2014 full Volterra)
3. Phase 3.1 any FS bucket Δecho < -0.020 (2× hard bar)
4. Phase 3.3 any xrtntuju 5-clip regression
5. Phase 4.3 C port byte-equality fails repeatedly

## 8. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Adapt-gate fires on DT → polynomial drift | MED | HIGH | Conservative dt_combined > 0.3 freeze; AecState.dt_combined is multi-source max |
| Adaptation diverges on long FS runs | LOW-MED | HIGH | Convergence indicator §1.6 gates downstream consumption; preprocessor reverts to identity if `nl_energy_ratio` exceeds sanity bound |
| Hammerstein insufficient for codec NL (Type 2) | MED | MED | V.1 §9 fallback to Yang 2014 2nd-order full Volterra |
| C port numerical drift | LOW | MED | `-ffp-contract=off` already mandatory per CLAUDE.md; Volterra terms are products of HPF'd-bounded signals |
| Listen-gate fails on subset of cohort | MED | LOW (research only) | Acceptable per §6.1 acceptance threshold (5/5 required only for SHIP; 3/5 triggers fallback path) |
| Preset BALANCED regression from preprocessor unconditional load | LOW | HIGH | Default-OFF preserves byte-equal; preset wiring happens only after §6.2 passes |

## 9. Non-goals for v3.20

Explicit OUT OF SCOPE for v3.20 Volterra arc:
1. **NN-based NL processor** — per CLAUDE.md `feedback_dsp_only_until_completion.md`
2. **Per-bin Volterra (frequency-domain)** — different mathematical framework, V.1 surveys time-domain only
3. **4th+ order kernels** — V.1 §1.2 noted diminishing returns; deferred to v3.21+ if v3.20 ships
4. **Cross-coupling between Volterra preprocessor and ShadowFilter** — too tight; v3.21+
5. **Adaptive μ_NL** — fixed `volterra_mu_scale_factor` for v3.20; adaptive scaling deferred
6. **Volterra inside ResFilter** (Option c in V.1) — explicitly rejected in §2.1

## 10. Cross-references

- [docs/v3_19_volterra_v1_literature_survey.md](v3_19_volterra_v1_literature_survey.md) — V.1 (this doc consumes V.1 §9 recommendation)
- [docs/v3_19_volterra_v3_cohort_sizing.md](v3_19_volterra_v3_cohort_sizing.md) — V.3 (companion doc for cohort)
- [docs/v3_19_phase0_1_volterra_eval.md](v3_19_phase0_1_volterra_eval.md) — auto-trigger conditions met by Phase 1 early closure
- [docs/v3_18_c_e_closeout.md](v3_18_c_e_closeout.md) — fq_usable arrival evidence + Pareto wall context
- [docs/v3_18_c_b1_filtering_quality_design.md](v3_18_c_b1_filtering_quality_design.md) — C.B substrate consumed by Volterra adapt-gate
- [docs/v3_18_c_a1_filter_analyzer_design.md](v3_18_c_a1_filter_analyzer_design.md) — C.A substrate consumed by Volterra adapt-gate
- `python/aec.py` L5245-5350 — AecState class (insertion point for §3 methods)
- `python/aec.py` L5764-5860 — SubtractiveNLP class (Volterra peer class location)
- `python/aec_filter_quality.py` — C.B fq_usable producer
- `python/aec_filter_analyzer.py` — C.A consistent_estimate producer
- CLAUDE.md — pipeline diagram + branch model + DSP-only directive

## 11. Verification rules followed

1. Mechanism derived from V.1 §9 recommendation (Stenger-Kellermann + Hammerstein p=3)
2. Insertion point selected via 3-option comparison §2.1 (drop-in Saturation replacement)
3. State integration follows v3.18 C.C AecState back-ref pattern (additive only)
4. Detector reuse chains C.A + C.B + E4.S2 + E5.S3 substrate (all default-OFF substrate already on disk)
5. Hard bars match v3.18 cycle precedent (Δecho ≥ -0.010, Δdeg ≥ -0.005)
6. Cohort invariants (qNvSMyU tail + xrtntuju 5-clip) preserved
7. Sprint plan has 5 §0.7 user-auth gates (Phase 0→1, 1→2, 2→3, 3→4, 4→merge)
8. NN approaches explicitly excluded per CLAUDE.md feedback memory
