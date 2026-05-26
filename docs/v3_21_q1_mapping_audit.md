# v3.21 Q1 mapping audit — AEC3 transient RefinedFilterUpdateGain SetConfig(initial) → PBFDKF (2026-05-24)

**Status**: read-only paper audit. No code. No benchmark. No cohort.

**Companions**:
- [docs/v3_21_audit_b_m1_m4_parity.md](v3_21_audit_b_m1_m4_parity.md) — full
  EPC chain audit.
- [docs/v3_21_delay_change_chain_design_note.md](v3_21_delay_change_chain_design_note.md) rev 3 — Q1–Q3 design framing.
- AEC3 source: [docs/aec3_extracts/src/aec3/refined_filter_update_gain.cc](aec3_extracts/src/aec3/refined_filter_update_gain.cc) + [docs/aec3_extracts/api/audio/echo_canceller3_config.h](aec3_extracts/api/audio/echo_canceller3_config.h).
- Our source: [python/modules/filters.py](../python/modules/filters.py) `PBFDKF` class + [python/modules/aec3_scale.py](../python/modules/aec3_scale.py).

**Goal**: decide how AEC3 `RefinedFilterUpdateGain::SetConfig(refined_initial, true)`
should map onto our PBFDKF architecture by tracing each `RefinedConfiguration`
field through both Compute() pipelines.

---

## 0. TL;DR

| Question | Answer (paper analysis only) |
|---|---|
| Is `_p_max_override` a valid adapter for AEC3 transient leakage config? | **NO**. `_p_max_override` operates on our **legacy `P` matrix** (`filters.py:384`, demoted to "backwards-compat / diagnostic" per `filters.py:411-414`); AEC3 transient leakage operates on the **primary `H_error` per-bin state**. Different state quantity, different mechanism, different effect. The rev 2/3 framing of "documented equivalence" should NOT use `_p_max_override` as the analogue. |
| Does PBFDKF need AEC3's 2.5 s transient + 1 s smoothing, or is `H_error.fill(10000)` enough? | **UNKNOWN from source alone**. `H_error` reset gives a sharp initial mu spike that decays fast (governed by `H_error_new = H_error × (1 − 0.5·mu·X²)`, which is near-immediate when `H_error` dominates the denom); AEC3 transient maintains an **elevated H_error STEADY STATE** for 2.5 s via 100× larger `leakage_converged`. The two mechanisms address different time-scales of recovery. Needs empirical trace to know which matters. |
| Is there a default-OFF trace that could show transient-leakage surface? | **YES** — instrument `H_error_per_bin` decay trajectory + per-frame `mu` magnitude on a delay-event cohort case (when one exists). Compare time-to-steady for our current PBFDKF vs hypothetical 100× leakage. No code change required; just inspect existing `_diag` fields or a simple post-hoc analysis script. |
| Q2 — can dynamic partition be documented as fixed-size functional adaptation? | **YES**. AEC3 default shrinks by **1 partition** (12 vs 13) for 2.5 s with 1-second smoothing — small magnitude, brief. We have 6 partitions throughout (already structurally different filter); "shrink to 5.5 partitions" doesn't map cleanly. H_error reset provides AEC3's fast-adaptation benefit independently. Q2 closes as **documented fixed-size functional adaptation (Q2=(b))** with no code change. |

---

## 1. AEC3 RefinedFilterUpdateGain field-by-field semantics

Source: `refined_filter_update_gain.cc:104-138` (Compute body) + `echo_canceller3_config.h:78-100` (struct + defaults).

```cpp
// Lines 104-111: noise_gate
for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
  if (X2[k] >= current_config_.noise_gate) {
    mu[k] = H_error_[k] /
            (0.5f * H_error_[k] * X2[k] + size_partitions * E2_refined[k]);
  } else {
    mu[k] = 0.f;
  }
}

// Line 114: narrow-band mask
render_signal_analyzer.MaskRegionsAroundNarrowBands(&mu);

// Lines 117-119: H_error decay
for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
  H_error_[k] -= 0.5f * mu[k] * X2[k] * H_error_[k];
}

// Lines 122-125: W gain
for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
  G->re[k] = mu[k] * E_refined.re[k];
  G->im[k] = mu[k] * E_refined.im[k];
}

// Lines 128-138: leakage refresh + clamp (ALWAYS runs)
for (size_t k = 0; k < kFftLengthBy2Plus1; ++k) {
  if (E2_refined[k] <= E2_coarse[k] || disallow_leakage_diverged) {
    H_error_[k] += current_config_.leakage_converged * erl[k];
  } else {
    H_error_[k] += current_config_.leakage_diverged * erl[k];
  }
  H_error_[k] = std::max(H_error_[k], current_config_.error_floor);
  H_error_[k] = std::min(H_error_[k], current_config_.error_ceil);
}
```

| Field | Role in Compute() | Effect on H_error trajectory |
|---|---|---|
| `noise_gate` | Pre-gate at line 105; if `X²[k] < noise_gate` then `mu[k] = 0` for THIS frame | Stops both W update and H_error decay on weak-far bins (mu=0 → H_error refresh dominates → H_error trends toward `leakage × erl / 1` since no decay). When refresh > decay, H_error grows. |
| `leakage_converged` | Refresh additive at line 131 when `E²_refined ≤ E²_coarse` (refined "winning") | Per-block injection rate. Higher = H_error settles higher → mu higher → faster adaptation. Steady value at AEC3 default = `5e-5`; transient = `5e-3` (100× higher). |
| `leakage_diverged` | Refresh additive at line 133 when `E²_refined > E²_coarse` (refined "losing") | Same shape as converged; engages when shadow beating refined. Steady value = `5e-2`; transient = `5e-1` (10× higher). |
| `error_floor` | Lower clamp at line 136 | Prevents H_error from collapsing to 0 (which would zero mu). Default `0.001` (same in steady + transient). |
| `error_ceil` | Upper clamp at line 137 | Prevents H_error from running away during transient. Default `2.0` (same in steady + transient). |
| `length_blocks` | NOT used inside Compute(); used by AdaptiveFirFilter SetSizePartitions | Affects size_partitions which appears in mu denom (`n × E²`) and in poor_excitation gate. Different filter length. |

**Field-vs-field initial-vs-steady delta**:

| Field | steady | initial (transient) | initial/steady ratio | Effect direction |
|---|---:|---:|---:|---|
| `length_blocks` | 13 | 12 | 0.92× | Smaller `n` in mu denom → mu slightly larger during transient |
| `leakage_converged` | 5e-5 | 5e-3 | **100×** | H_error refreshes 100× faster → settles ~100× higher → mu correspondingly higher |
| `leakage_diverged` | 5e-2 | 5e-1 | **10×** | Same shape, diverged-side |
| `error_floor` | 1e-3 | 1e-3 | 1× | unchanged |
| `error_ceil` | 2.0 | 2.0 | 1× | unchanged |
| `noise_gate` | 2.01e7 | 2.01e7 | 1× | unchanged |

**Key insight**: the transient profile changes ONE qualitative variable (the
**H_error refresh rate**) by a large factor. Everything else is identical
or near-identical. So "transient config" in AEC3 means: **maintain H_error
at an elevated equilibrium for 2.5 s after HEPC, then smoothly relax back
over 1 s**.

---

## 2. Our PBFDKF equivalents

Source: `filters.py:378-825` (`PBFDKF` class + `_update_weights_aec3` + `_h_error_refresh`) + `aec3_scale.py:71-99` (constants).

### 2.1 H_error per-bin (primary Kalman state)

`PBFDKF.H_error_per_bin` shape `(n_freqs,)` initialised to `H_ERROR_INIT_FLOAT = 10000.0` (matches AEC3 `kHErrorInitial`).

Compute path: `_update_weights_aec3` at `filters.py:714-824` implements the AEC3 formula:
- mu denom: `0.5 × H × X² + n × E² + delta` (matches AEC3)
- Decay: `H -= 0.5 × mu × X² × H` (matches AEC3)
- Refresh: `H += leakage_{conv,div} × erl` (`_h_error_refresh` at `filters.py:826`)
- Clamp: `[H_ERROR_FLOOR, H_ERROR_CEIL]`

### 2.2 Leakage values (`aec3_scale.py:87-88`)

```python
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-3, 160, 16000)  # = 2.5e-3
LEAKAGE_DIVERGED_PER_HOP_DEFAULT  = per_block_rate_to_per_hop(1e-1, 160, 16000)  # = 2.5e-1
```

`per_block_rate_to_per_hop(x, 160, 16000)` = `x × (10ms / 4ms)` = `x × 2.5`. So our per-BLOCK base values are `1e-3` and `1e-1` respectively.

**Comparison to AEC3 (per-block scale)**:

| Quantity | AEC3 `refined` | AEC3 `refined_initial` | Ours (per-block base) | Our position |
|---|---:|---:|---:|---|
| `leakage_converged` | 5e-5 | 5e-3 | 1e-3 | 20× higher than AEC3 steady; 5× lower than AEC3 transient |
| `leakage_diverged` | 5e-2 | 5e-1 | 1e-1 | 2× higher than AEC3 steady; 5× lower than AEC3 transient |

**Implication**: our PBFDKF default leakage values are **biased toward AEC3
transient** (not equal to either). We do not currently switch between
steady and transient values; the same biased-toward-transient values are
used at all times.

### 2.3 Clamp values (`aec3_scale.py:82-84`)

| Field | Our value | AEC3 value | Ratio (ours / AEC3) |
|---|---:|---:|---:|
| `H_ERROR_INIT_FLOAT` | 10000.0 | 10000 | 1× ✓ |
| `H_ERROR_FLOOR_FLOAT` | 1e-3 | 1e-3 | 1× ✓ |
| `H_ERROR_CEIL_FLOAT` | 1e2 = 100.0 | 2.0 | **50× higher** ⚠ |

**Divergence**: our `H_error_ceil` is 50× higher than AEC3 default. With
our higher leakage values + higher ceil, our H_error can grow much larger
than AEC3 allows. This is an **independent divergence** from the
transient-config question — not a "transient port" gap but a separate
tuning choice. Documenting here for completeness; out of Q1 scope.

### 2.4 `_p_max_override` / `_p_floor_beta` (legacy P matrix)

`PBFDKF.P` shape `(n_partitions, n_freqs)` initialised at `0.01`. Per
`filters.py:411-414`:

> "H_error replaces our P matrix as the primary Kalman-like state in the
> K compute (refined_filter_update_gain.cc:104-138). P stays as a
> parallel field for backwards-compat (PathChangeRegimeHandler overrides
> / diagnostic), but the K applied to W comes from H_error when
> `use_aec3_h_error=True` (default)."

`_p_max_override` (set in `orchestrator.py:1933, 2417, 2456, 2566, 2572,
2650, 2660`) operates on this **secondary P matrix** — NOT on H_error.

**Implication**: `_p_max_override` does not affect the AEC3-aligned
mu / W-update path. It only affects the legacy P-based diagnostic /
PathChangeRegimeHandler internal state. Therefore it **cannot be a
functional adapter for AEC3's transient leakage config**, which operates
on H_error.

This corrects the rev 2/3 design-note framing ("documented per-event
override as parity port") — `_p_max_override` is the WRONG mechanism
to point at for this purpose.

### 2.5 Counter resets

`_call_counter = 0`, `_poor_excitation_counter = 400` (= 4 s wall) on
HEPC `!gain_change` (per Audit B §3.2). These affect the startup gate
(60 ms freeze post-HEPC). NOT a transient-config analogue; just AEC3
parity for `kPoorExcitationCounterInitial`.

---

## 3. Mapping table — AEC3 transient `RefinedConfiguration` → PBFDKF

Six columns per user request.

| AEC3 field / mechanism | Our current equivalent | Same semantics? | Strict port — what changes | Functional-equivalence closure — what evidence needed | v3.22 divergence if not ported? |
|---|---|---|---|---|---|
| `SetConfig(refined_initial, true)` on HEPC: switch `leakage_converged` 5e-5 → 5e-3 (100×) | None. Our `_leakage_converged` is fixed at construction to `2.5e-3` per-hop (≈ 1e-3 per-block, which is between AEC3 steady 5e-5 and transient 5e-3, biased toward transient). | **NO** — we do not switch on HEPC; fixed value used at all times. | Add a transient-vs-steady leakage tuple to `PBFDKF.__init__`; `handle_echo_path_change` switches to transient values; introduce a `_in_transient_phase` flag with 2.5 s countdown; `_h_error_refresh` reads current value (transient vs steady). | Show that our H_error already settles "high enough" after HEPC such that AEC3's elevated transient `H_error` would not produce meaningfully larger mu. Either (a) trace `H_error_per_bin` post-HEPC, measure time-to-steady-state and steady-state value vs AEC3 hypothetical, or (b) demonstrate via dynamics analysis that the H_error reset's initial spike provides enough fast-adaptation to make sustained 100× leakage unnecessary. | If user explicitly chooses to keep our fixed leakage with no transient switch (Q1=(c)), this is v3.22 divergence. Document why our fixed-near-transient value is sufficient for our PBFDKF Kalman dynamics. |
| `SetConfig(refined_initial, true)` switches `leakage_diverged` 5e-2 → 5e-1 (10×) | None; fixed at `2.5e-1` per-hop (1e-1 per-block, between AEC3 steady 5e-2 and transient 5e-1). | **NO** — same as above. | Paired with leakage_converged switch; same touch point. | Same as above. Note our default is 2× larger than AEC3 steady but 5× smaller than AEC3 transient — argument similar. | Same as above. |
| `SetConfig(refined_initial, true)` switches `error_floor` 1e-3 → 1e-3 | `H_ERROR_FLOOR_FLOAT = 1e-3` (fixed). | **YES** ✓ (same value, no switch needed because AEC3 default doesn't change this field) | n/a — already at parity. | n/a. | n/a. |
| `SetConfig(refined_initial, true)` switches `error_ceil` 2.0 → 2.0 | `H_ERROR_CEIL_FLOAT = 1e2 = 100.0` (fixed). | **NO** — different absolute value (50× higher), but neither value switches at HEPC. | Lower `H_ERROR_CEIL_FLOAT` from 100 to 2.0. Separate from transient switching — this is an INDEPENDENT clamp tuning. | Show that our higher ceil never hits the clamp in practice (i.e. `H_error` rarely exceeds 2.0 anyway), so the difference is theoretical. Trace `H_error.max()` per case. | If kept at 100 by design choice, v3.22 divergence. NOT a transient-config question — separate audit. |
| `SetConfig(refined_initial, true)` switches `noise_gate` (no change) | `NOISE_GATE_POWER_FLOAT = psd_int16_to_float(27509562)` ≈ 0.0256 (after FFT-scale conversion). AEC3 native = 20075344 (different value). | Different absolute value but AEC3 doesn't switch this field either; no transient delta. | n/a for transient switching. Independent question of noise_gate value calibration. | Out of Q1 scope. | Out of Q1 scope. |
| `SetSizePartitions(refined_initial.length_blocks, true)` — shrink 13 → 12 partitions for 2.5 s | None. Our `n_partitions = 6` fixed. | **NO** — no analogue; no dynamic sizing. | Add `_active_size_partitions` runtime var + zero-on-shrink semantics; gate W/X_buf/P loops on `_active`. HIGH cost (~200 lines). | AEC3 default shrinks by 1 of 13 partitions (8%); we run 6 partitions throughout. The "shrink to 5.5 partitions" doesn't map (fractional). H_error reset already provides AEC3's fast-adaptation benefit via the elevated H_error spike. Document fixed-size as functional adaptation. | If kept fixed by design choice, NOT v3.22 — closes as Q2=(b) functional adaptation with documented justification. Only v3.22 if user explicitly chooses to deviate beyond AEC3 with separate motivation. |
| Smooth transition over `config_change_duration_blocks = 250` blocks (1 s) on `SetConfig(_, immediate=false)` from `ExitInitialState` | None. We have no `ExitInitialState` hook and no smoothing window. | **NO** — no analogue. | Add `_config_change_counter` to PBFDKF; `set_config` method takes `immediate=true/false`; per-frame interpolation in `_h_error_refresh`. Couples to transient-vs-steady config split. | Only needed if strict transient-config port is chosen. If documented equivalence (b) is chosen, smoothing is moot. | Same as above — if Q1=(c), v3.22 divergence with documented justification. |
| `_p_max_override` ↔ AEC3 transient? | (the rev 2/3 design note suggested this mapping) | **NO** — `_p_max_override` operates on legacy `P` matrix (demoted to backwards-compat / diagnostic per `filters.py:411-414`). AEC3 transient operates on PRIMARY `H_error` state. Different state quantity, different mechanism. | n/a — wrong analogue. | n/a — wrong analogue. | n/a — wrong analogue. **The "documented equivalence" closure for Q1 should NOT cite `_p_max_override` as the adapter.** |

---

## 4. Answers to four specific questions

### 4.1 Is `_p_max_override` really a valid adapter for AEC3 transient leakage config?

**NO.** Detailed reasoning:

1. **State quantity mismatch**: `_p_max_override` overrides the ceiling
   of our `P` matrix (legacy Kalman covariance, `filters.py:384`,
   initialised to 0.01). AEC3 transient config modifies `leakage_*`
   parameters that affect `H_error` (the primary AEC3 Kalman-like state).
   These are different state quantities, not equivalent overrides.
2. **Code architecture has explicitly demoted `P`**: per `filters.py:411-414`,
   "H_error replaces our P matrix as the primary Kalman-like state in
   the K compute. P stays as a parallel field for backwards-compat
   (PathChangeRegimeHandler overrides / diagnostic)". `_p_max_override`
   acts on the now-secondary diagnostic state. It does NOT affect the
   `_update_weights_aec3` path that computes mu from H_error.
3. **Effect direction differs**: AEC3 transient leakage causes H_error
   to settle HIGHER → mu HIGHER → faster adaptation. `_p_max_override`
   forces P to a max — this doesn't translate into a mu boost in the
   H_error path (mu depends on H_error, X², E², not P).
4. **Time-constant differs**: 30 frames (300 ms) vs AEC3's 2.5 s — and
   even this comparison is irrelevant given (1)-(3).

**Implication for design note**: the Q1=(b) "documented equivalence /
adaptation" stance in [docs/v3_21_delay_change_chain_design_note.md](v3_21_delay_change_chain_design_note.md)
rev 3 §3.5 and §0 TL;DR should NOT cite `_p_max_override` as the adapter.
The true v3.21.x closure under Q1=(b) is: **our PBFDKF leakage values
are pre-tuned biased toward AEC3 transient (1e-3 vs 5e-5 steady; 5e-3
transient) — equivalent to "always running close to AEC3 transient
config" without the explicit switch + smoothing machinery.** This is a
DIFFERENT analogue from `_p_max_override` and weaker because:
- We don't reach AEC3 transient's 100× converged elevation;
- We don't get the 1 s smooth transition;
- We pay the cost of "always-transient-ish" leakage in steady-state.

### 4.2 AEC3 2.5 s transient + 1 s smoothing — necessary for our PBFDKF, or is `H_error.fill(10000)` enough?

**Unknown from source alone.** Mechanistic analysis:

- **H_error reset**: spike from steady → 10000 instantly on HEPC. Decay
  rate: `H_new = H × (1 − 0.5·mu·X²)`. When H dominates the denom, this
  is `1 − [0.5·H·X² / (0.5·H·X² + n·E²)]` ≈ `n·E² / (0.5·H·X²)`. With
  large H (post-reset), the bracket is near 1, so `H_new ≈ H × small`.
  Decay is fast in the H-dominated regime — H drops back near steady
  within a few Compute() calls if leakage rate is low.
- **AEC3 transient leakage**: maintains H_error at an elevated steady
  state for 2.5 s by refreshing `H += 5e-3 × erl` per call (vs steady
  `H += 5e-5 × erl`). The elevated steady supports a sustained higher
  mu for the entire transient phase.

**The two mechanisms target different timescales**:
- H_error reset: spike on frame 0, recovers fast adaptation for ~10-30
  frames (100-300 ms) until H_error settles back to steady leakage value.
- AEC3 transient: sustained elevation for 2.5 s.

**Whether the difference matters in our pipeline depends on**:
- (a) How long does our H_error actually take to decay back to steady
  after the reset? — Source analysis cannot answer (depends on X², E²
  trajectories specific to the stress case).
- (b) Does our pre-tuned-near-transient leakage steady (1e-3 base, 20×
  AEC3 steady) substitute for AEC3's transient elevation? — Likely
  partial, since our 1e-3 ≠ AEC3 transient 5e-3.
- (c) For delay-change recovery (the scenario AEC3 designed the
  transient phase for), how much of total cancellation improvement
  comes from the initial spike vs the sustained 2.5 s elevation? —
  Requires trace evidence.

**Conclusion**: cannot be answered from source. Surfaces a trace
candidate (§4.3).

### 4.3 Is there a default-OFF trace that could show transient-leakage surface on stress / delay-change cases?

**YES — and it is feasible without code changes.** Trace design:

**Instrumentation**: existing `_diag['H_error_mean']` and `_diag['H_error_max']`
in trace scripts (per `v3_21_20_phase_e_trace.py:134-135`) already
provide per-frame H_error trajectory. No new code needed.

**Cases to scan**: any case with a real `delay_first` / `delay_shift`
event (Audit B §5.4 shows 12-case has zero such events; would need to
identify a cohort with delay events — see §8.1 of design note).

**Analysis** (post-hoc, no code change to algorithm):

1. Render the case with current production main; capture trace.
2. From trace JSON: identify HEPC fire frame, extract H_error trajectory
   for next 250 hops (= 2.5 s, AEC3's transient duration).
3. Measure:
   - Initial H_error spike value (= 10000 after reset).
   - Time-to-steady (frames until `|H_error_mean - long_term_mean| < threshold`).
   - Steady-state value after spike decays.
4. Compute hypothetical AEC3-transient-equivalent H_error:
   - Same spike (= 10000).
   - Same decay rate (governed by same mu formula).
   - But 100× refresh rate during transient → analytically project the
     elevated steady value.
5. Compute hypothetical mu for both H_error trajectories.
6. Compare: how different is mu sustained over 2.5 s?

**What this trace would surface**:
- If our H_error settles to a value close to AEC3 transient's elevated
  steady within ~10 frames, then **the sustained transient elevation is
  largely captured by our pre-tuned-high leakage** → Q1=(b) stance is
  validated → no code change needed.
- If our H_error settles much lower than AEC3 transient's would, then
  **AEC3 transient gives a sustained mu boost we lack** → Q1=(a) becomes
  more attractive (strict port) OR Q1=(c) becomes a documented divergence.
- If our H_error settles ABOVE AEC3 transient's would (because our
  leakage is biased high and clamp is 50× higher), then **AEC3 transient
  would actually be MORE CONSERVATIVE than our current behaviour** →
  porting it would REDUCE mu → arguably worse for fast tracking →
  Q1=(b) or (c) clearly preferred.

**Cost**: ~30 lines of analysis script. No flag, no benchmark, no
behaviour change.

**Prerequisite**: a delay-event cohort. This same prerequisite is
already gating §8.1 of the design note (true-delay-event audit).

### 4.4 Q2 — can dynamic partition be documented as fixed-size functional adaptation, no code?

**YES.** Justification (paper-only):

1. **AEC3 default magnitude is small**: 12 vs 13 partitions = 8% shrink,
   for 2.5 s, with 1 s smoothing on the way back. Even on AEC3's own
   pipeline, this is a tertiary effect — the dominant transient mechanism
   is the 100× leakage_converged boost, not the 1-partition shrink.
2. **Our filter is already structurally different**: 6 partitions vs
   AEC3's 13. "Shrink by 1 of 6" would be a 17% shrink (~5.5 partitions);
   "shrink to AEC3-initial proportionate" would be 6 × (12/13) ≈ 5.54 —
   fractional, doesn't map.
3. **H_error reset already provides the fast-adaptation benefit**: AEC3
   ships the 1-partition shrink because their NLMS path can't benefit
   from H_error reset (PBFDAF coarse). For their refined (PBFDKF-style),
   the shrink is a secondary refinement on top of leakage modulation.
   Our refined PBFDKF gets the H_error reset; we don't have the analogous
   "NLMS-only refinement" need.
4. **Cost of port is HIGH**: ~200 lines touching innermost W/X_buf/P
   loops + P-matrix shrink/restore semantics (§5 of design note).
   Disproportionate vs the 8% transient-phase magnitude on a tertiary
   mechanism.

**Documented justification for Q2=(b)** (paper-only, no code):

> Our PBFDKF uses 6 partitions throughout (52 ms filter / 10 ms hop).
> AEC3 default shrinks refined from 13 to 12 partitions for 2.5 s after
> HEPC, then smoothly restores. The shrink magnitude is 8% on AEC3's own
> filter; the dominant transient mechanism in AEC3 is the 100×
> `leakage_converged` increase, not the partition shrink. Our PBFDKF
> H_error reset (which AEC3's NLMS coarse cannot use) already provides
> the fast-adaptation benefit that AEC3's partition shrink targets, via
> a different mechanism. Therefore we retain fixed `n_partitions = 6`
> as **documented functional adaptation** of AEC3's transient sizing —
> not v3.21.x parity violation, not v3.22 divergence.

**This closes Q2 with no code change required.** The remaining
architectural question collapses to Q1 (transient leakage config) and
Q3 (exit-transient global vs per-event), which couple together.

---

## 5. Implications for design note rev 3 framing

After this audit, the design note's Q1=(b) stance needs revision:

- **Current rev 3 framing** (§3.5 Q1 row): "(b) Document our existing
  per-event `_p_max_override` machinery as the parity port"
- **Should read**: "(b) Document our pre-tuned-near-transient PBFDKF
  leakage values (per-hop base 1e-3 / 1e-1, vs AEC3 steady 5e-5 / 5e-2
  and AEC3 transient 5e-3 / 5e-1) as a static-equivalent of AEC3's
  transient-vs-steady switch. We accept the 5× gap below AEC3 transient
  AND the 20× elevation above AEC3 steady as the implicit equivalent of
  'always running close to AEC3 transient leakage'. Pros: no code, no
  state machine. Cons: pays elevated leakage cost in steady state and
  doesn't reach AEC3 transient peak. **`_p_max_override` is NOT the
  adapter** — different state quantity (legacy P, not H_error)."

Q2=(b) framing should be strengthened per §4.4 above with the explicit
justification.

Q3 framing is unchanged — couples to Q1 + Q2.

**This is a documentation correction, not a new architectural choice.**

---

## 6. Conclusion

The user's four questions have factual answers from this read-only audit:

1. **`_p_max_override` is NOT a valid adapter** for AEC3's transient
   leakage config — it operates on legacy P (now diagnostic), not on
   H_error (the primary state AEC3 transient modulates).
2. **Whether AEC3's 2.5 s transient + 1 s smoothing is necessary** —
   cannot answer from source; requires a small post-hoc trace analysis
   on a delay-event cohort case (§4.3 outlines the cheap design).
3. **A default-OFF trace IS feasible** — existing trace fields suffice;
   ~30 lines of analysis script; no algorithm change; gating only on
   identifying a delay-event cohort case.
4. **Q2 closes as documented fixed-size functional adaptation** with no
   code change — AEC3's 8% partition shrink magnitude is disproportionate
   to porting cost; H_error reset provides the analogous fast-adaptation
   benefit.

**Recommendations** (no implementation; review and approval gates):

- Patch design note rev 3 to correct the Q1=(b) "documented equivalence"
  framing per §5 above (the adapter is the pre-tuned leakage values,
  not `_p_max_override`).
- Close Q2 as documented adaptation in the design note's verdict section.
- For Q1, defer the (a) / (b) / (c) decision until a delay-event cohort
  trace is available (per §4.3) — the trace will inform whether sustained
  transient elevation is materially absent from our pipeline.

No code change in this round. No benchmark. No cohort run.
