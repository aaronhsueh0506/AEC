# v3.21 Design note — AEC3 delay-change reset chain in our PBFDKF architecture (2026-05-24, rev 4)

**Status**: design note, NOT a patch plan. Read-only exploration.
No code in this round. No benchmark.

**rev 4 changes** (2026-05-24, Q1 framing correction after Q1 mapping audit):
- Retracted earlier framing of `_p_max_override` / `_p_floor_beta` /
  `_arc_m_q_boost` as the parity adapter for AEC3 transient
  `RefinedFilterUpdateGain::SetConfig(refined_initial, …)`. Per
  [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md), these are
  **legacy P-matrix overrides** (operate on the diagnostic `P` field,
  not the primary `H_error_per_bin` Kalman state — see
  `filters.py:411-414` comment block). They are orthogonal to AEC3
  `RefinedFilterUpdateGain` and do not adapt the leakage / H_error
  update path.
- Q1 stances reframed:
  - **(a) strict AEC3 port** — `SetConfig(refined_initial, true)` on
    HEPC switching `leakage_converged` / `leakage_diverged` to the
    transient profile for `initial_state_seconds = 2.5 s`, then
    `SetConfig(refined, false)` on `TransitionTriggered` smoothing
    back to steady over `config_change_duration_blocks = 1 s`.
  - **(b) functional adaptation (documented; current code state)** —
    PBFDKF already runs **elevated steady-state leakage values**
    (`LEAKAGE_CONVERGED_PER_HOP_DEFAULT` / `LEAKAGE_DIVERGED_PER_HOP_DEFAULT`
    in `python/modules/aec3_scale.py`; consumed by `PBFDKF.__init__`
    in `python/modules/filters.py`). These per-block values are **20×
    AEC3 steady `leakage_converged`** and **2× AEC3 steady
    `leakage_diverged`**, i.e. biased toward AEC3 transient profile
    without the two-state machine. Documented equivalence requires
    accepting (i) the time-constant gap (no 2.5 s transient bound,
    no 1 s smoothing) and (ii) the missing transient-vs-steady
    separation.
  - **(c) intentional v3.22 divergence** — explicitly choose to
    deviate beyond AEC3 with separate justification.
- Q2 (dynamic partition sizing) **CLOSED as documented fixed-size
  functional adaptation, no code**. AEC3 default only shrinks by 1
  partition (12 → 13) with 1 s smoothing; PBFDKF Kalman P-matrix
  shrink/restore would be lossy; H_error reset already provides
  per-bin fast-adaptation independently of partition count. Closure
  is doc-only; no code change.
- **Q1 remains OPEN v3.21.x parity question**. Next step is **NOT
  implementation** — it is identification of a delay-event /
  EPC-rich cohort + trace of `H_error_per_bin` post-HEPC trajectory
  vs AEC3 trace, to determine whether stance (a) [strict port] or
  stance (b) [documented equivalence with current leakage tuning]
  best closes the parity gap. Per user direction: no code, no
  benchmark, no cohort run in this round.

**rev 3 changes** (2026-05-24, AEC3 config defaults fetched):
- Fetched `echo_canceller3_config.{h,cc}` from upstream WebRTC mirror
  (`webrtc-mirror/webrtc/main/api/audio/`) and saved to
  `docs/aec3_extracts/api/audio/`.
- §3.1 table now populated with confirmed default values for
  `refined / refined_initial / coarse / coarse_initial`,
  `config_change_duration_blocks`, `initial_state_seconds`,
  `coarse_reset_hangover_blocks`, `conservative_initial_phase`.
- **Three of the four unknowns in old §7 are now answered.**
- The §1.5 design-intent "more permissive transient" framing, previously
  marked HYPOTHESIS in rev 2, is now CONFIRMED FACT for refined +
  coarse leakage / rate values.
- §2.5 (dynamic partitions) now notes the AEC3 default shrink is only
  1 partition (12 vs 13) — machinery active but small magnitude.
- §3.5 + §0 TL;DR rewritten with fact-aware Q1–Q3 stances.

**rev 2 changes** (2026-05-24, earlier fact-gathering pass):
- Added §3 "Source-derived facts before user decision" — direct read of
  AEC3 extracts + our PBFDKF code answers many previously-open questions
  (block sizes, partition count, override durations, tail-zero semantics,
  shadow PBFDAF override absence).
- Narrowed §4 (was §3) Q1-Q4 — Q4 resolved (collapses into Q2); Q1-Q3
  re-stated with facts established.
- §7 listed only the remaining unknowns (now mostly resolved in rev 3).

**Companion**: [docs/v3_21_audit_b_m1_m4_parity.md](v3_21_audit_b_m1_m4_parity.md)
identified four OPEN v3.21.x structural parity gaps (M2 exact semantics:
rows 1+2+6+7+8+9). This note maps each onto our PBFDKF architecture and
surfaces the architectural questions the user needs to decide BEFORE any
code is written.

**Scope**:
- AEC3 `delay_change` chain only. `gain_change` handler is parity per
  Audit B; not revisited.
- Filter-side reset chain + paired `ExitInitialState` transition.
- `SuppressionGain::SetInitialState` excluded (separate subsystem;
  catalogued in Audit B §5.1).

**Non-goals**:
- This note does NOT pick winners between options.
- It does NOT lay out an implementation sequence.
- It does NOT estimate AECMOS impact.
- It does NOT propose adopting any particular option.

---

## 0. The decision the user has to make (one-page summary, rev 4)

After §3 fact-gathering + Q1 mapping audit:
- **Q2** (dynamic partition sizing) — **CLOSED** as documented
  fixed-size functional adaptation, no code.
- **Q4** (`W.fill(0)` vs tail-zero) — **CLOSED** following Q2;
  `W.fill(0)` remains non-AEC3 default-OFF substrate.
- **Q3** (TransitionTriggered hook) — **DEFERRED**, fate follows Q1.
- **Q1** (transient → steady leakage config split) — **OPEN**
  v3.21.x parity question.

The single remaining architectural question is Q1, with three stances:
- **(a) Strict AEC3 mechanism port** — implement the AEC3 mechanism in our
  pipeline as faithfully as the architecture allows.
- **(b) v3.21.x closure by documented functional equivalence/adaptation** —
  do not introduce the AEC3 mechanism literally; close the parity gap by
  documenting the **elevated steady leakage values** in
  `python/modules/aec3_scale.py` / `python/modules/filters.py` as the
  functional adapter and recording the equivalence caveats (no transient
  bound, no smoothing window).
- **(c) Intentional v3.22 divergence** — explicitly choose to deviate
  beyond AEC3 with separate justification.

| Architectural question | Stances | Facts (§3) | Status |
|---|---|---|---|
| **Q1** — does PBFDKF need a transient → steady leakage config split (like AEC3's `refined_initial` → `refined`)? | (a) **Strict AEC3 port** — `SetConfig(refined_initial, true)` on HEPC for `initial_state_seconds = 2.5 s`, `SetConfig(refined, false)` on `TransitionTriggered` smoothing back over `config_change_duration_blocks = 1 s`; (b) **Functional adaptation (current code state)** — elevated steady leakage values in `aec3_scale.py` / `filters.py` (20× AEC3 steady `leakage_converged`, 2× AEC3 steady `leakage_diverged`) accepted as documented equivalence, with time-constant + transient-bound caveats; (c) **Intentional v3.22 divergence** with separate justification. **`_p_max_override` is NOT a Q1 adapter** — it operates on the legacy P matrix (orthogonal diagnostic state), not on `H_error_per_bin` (the primary Kalman state). See [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md). | **(rev 3+4 facts)** AEC3 default transient phase **2.5 s** (`initial_state_seconds`); smoothing window **1 s** (`config_change_duration_blocks = 250` blocks); transient `leakage_converged` is **100×** steady, `leakage_diverged` is **10×** steady. Our per-block leakage values: `leakage_converged_per_block ≈ 1e-3` = 20× AEC3 steady (5e-5), 5× lower than AEC3 transient (5e-3); `leakage_diverged_per_block ≈ 1e-1` = 2× AEC3 steady (5e-2), 5× lower than AEC3 transient (5e-1). PBFDKF runs on always-elevated leakage with no two-state machine. | **OPEN v3.21.x parity question.** Next step = identify delay-event / EPC-rich cohort + trace `H_error_per_bin` trajectory post-HEPC vs AEC3 trace. **NOT implementation.** |
| **Q2** — should PBFDKF support dynamic partition sizing? | **CLOSED** as documented fixed-size functional adaptation (no code). PBFDKF Kalman P-matrix shrink/restore would be lossy; AEC3 default only shrinks by 1 partition (8%) with 1 s smoothing — magnitude small; H_error reset already provides per-bin fast-adaptation independently of partition count. | **(rev 3 facts)** Our partitions fixed at `n_partitions=6` (52 ms / 10 ms hop). AEC3 default `refined_initial.length_blocks=12` vs `refined.length_blocks=13` (8% shrink, **1 partition**). | **CLOSED — documented fixed-size functional adaptation, no code.** Closure recorded here + §2.5 + §3.5 + §4.2. |
| **Q3** — exit-transient global vs per-event? | (a) Global TransitionTriggered hook (only if Q1=(a)); (b) No transition hook needed (Q2 closed; Q1 stance (b)/(c) needs no exit). | Couples to Q1. With Q2 closed, Q3 reduces to a single-axis sub-question of Q1. | **DEFERRED** — fate follows Q1. |
| ~~**Q4** — `W.fill(0)` vs `ZeroFilter(current, max)`?~~ | **RESOLVED in §3.4** — collapses into Q2. With Q2 closed (fixed-size), AEC3 tail-zero has no direct surface; current `W.fill(0)` remains a non-AEC3 default-OFF substrate, do not ship unless separately justified. | — | **CLOSED — answer follows Q2 (CLOSED).** |

**This note's job is to surface Q1, not pre-make the decision.** Q2 / Q3 / Q4 are now resolved or coupled-to-Q1.

---

## 1. AEC3 reference: full delay-change chain

### 1.1 Top-level dispatch (`echo_remover.cc:385-424`)

```
1. if (AudioPathChanged):
     subtractor_.HandleEchoPathChange(variability)
     aec_state_.HandleEchoPathChange(variability)
     if (delay_change != kNone): suppression_gain_.SetInitialState(true)
2. if (aec_state_.TransitionTriggered()):
     subtractor_.ExitInitialState()
     suppression_gain_.SetInitialState(false)
3. subtractor_.Process(...)
4. ... residual_estimator + suppression_gain ...
5. aec_state_.Update(...)
```

### 1.2 `Subtractor::HandleEchoPathChange::full_reset` (`subtractor.cc:150-163`)

Per capture channel, in order:

| Order | Call | Effect |
|---:|---|---|
| 1 | `refined_filters_[ch]->HandleEchoPathChange()` | `ZeroFilter(current_size_partitions_, max_size_partitions_, &H_)` — zero the unused tail only |
| 2 | `coarse_filter_[ch]->HandleEchoPathChange()` | Same for coarse |
| 3 | `refined_gains_[ch]->HandleEchoPathChange(variability)` | `H_error_.fill(10000)` (delay branch) + `poor_excitation_counter=1000` + `call_counter=0` (`!gain_change` branch) |
| 4 | `coarse_gains_[ch]->HandleEchoPathChange()` | `poor_signal_excitation_counter=0` + `call_counter=0` |
| 5 | `refined_gains_[ch]->SetConfig(refined_initial, true)` | Switch refined gain config to transient profile; `immediate=true` means no smoothing |
| 6 | `coarse_gains_[ch]->SetConfig(coarse_initial, true)` | Switch coarse gain config to transient profile |
| 7 | `refined_filters_[ch]->SetSizePartitions(refined_initial.length_blocks, true)` | Shrink filter to initial length + zero displaced partitions; `immediate=true` |
| 8 | `coarse_filter_[ch]->SetSizePartitions(coarse_initial.length_blocks, true)` | Same for coarse |

### 1.3 `Subtractor::ExitInitialState` (`subtractor.cc:177-186`)

Paired return path. Per capture channel:

| Order | Call | Effect |
|---:|---|---|
| 1 | `refined_gains_[ch]->SetConfig(refined, false)` | Switch to steady refined config; `immediate=false` → smooth transition over `config_change_duration_blocks` |
| 2 | `coarse_gains_[ch]->SetConfig(coarse, false)` | Same for coarse |
| 3 | `refined_filters_[ch]->SetSizePartitions(refined.length_blocks, false)` | Grow filter to steady length; `immediate=false` |
| 4 | `coarse_filter_[ch]->SetSizePartitions(coarse.length_blocks, false)` | Same for coarse |

Trigger: `aec_state_.TransitionTriggered()` — fires once when `strong_not_saturated_render_blocks_` crosses the initial-state threshold (`5 * kNumBlocksPerSecond` in conservative mode, else `initial_state_seconds * kNumBlocksPerSecond`).

### 1.4 `AecState::HandleEchoPathChange::full_reset` (`aec_state.cc:143-167`)

Already ported in `state/aec_state.py:219-224`. Listed for completeness:

- `filter_analyzer_.Reset()`
- `capture_signal_saturation_ = false`
- `strong_not_saturated_render_blocks_ = 0`
- `blocks_with_active_render_ = 0`
- `initial_state_.Reset()` ← resets the gate that drives `TransitionTriggered`
- `transparent_state_->Reset()`
- `erle_estimator_.Reset(true)`
- `erl_estimator_.Reset()`
- `filter_quality_state_.Reset()`

**Note**: AecState's `initial_state_.Reset()` is what makes
`TransitionTriggered` fire again after the next delay change. Without
this reset, ExitInitialState would only fire once per stream (at the
initial startup).

### 1.5 Why AEC3 does this — design intent

AEC3 treats every delay change as a **mini-startup**:
1. **Filter knowledge is invalidated** (delay changed → past taps are wrong).
2. **Recovery requires a transient phase** — slightly shorter filter
   (refined: 12 vs 13 blocks at default; 8% shrink), a transient gain
   profile that **is** materially more permissive than steady
   (`leakage_converged` 100×, `leakage_diverged` 10×, NLMS `rate` 1.29×
   higher — confirmed rev 3 from `api/audio/echo_canceller3_config.h`),
   and Wiener gain in conservative initial mode.
3. **Once enough render energy accumulates** (`TransitionTriggered`),
   filter grows back to steady length, gain config relaxes, Wiener gain
   exits initial mode.
4. The pair (HEPC, ExitInitialState) is symmetric: HEPC enters
   transient phase; ExitInitialState exits it.

Splitting any one piece out of the pair breaks symmetry. E.g.,
M2 W=0 without `SetConfig(initial)` + `SetSizePartitions(initial)`
removes filter knowledge but leaves the gain/size at steady values,
which is what Phase E experienced as catastrophic recovery.

---

## 2. Mapping AEC3 mechanisms onto our PBFDKF architecture

### 2.1 H_error reset + counter reset (already ported)

| AEC3 mechanism | Our equivalent | Status |
|---|---|---|
| `refined_gains_[ch]->HandleEchoPathChange(delay_change)` → `H_error_.fill(10000)` | `PBFDKF.handle_echo_path_change(delay_change=True)` → `H_error_per_bin.fill(10000)` | Ported |
| `refined_gains_[ch]->HandleEchoPathChange(!gain_change)` → counter reset | `PBFDKF.handle_echo_path_change(!gain_change)` → counter reset | Ported |
| `coarse_gains_[ch]->HandleEchoPathChange()` → counter reset | `PBFDAF.handle_echo_path_change(!gain_change)` → counter reset | Ported |

No design work needed.

### 2.2 Filter W tail-zero (`AdaptiveFirFilter::HandleEchoPathChange`)

**AEC3 mechanism**: `ZeroFilter(current_size_partitions_, max_size_partitions_, &H_)` — zero indices `[current, max)`. If `current == max` (no dynamic sizing), this is a no-op.

**Our current state**: `PBFDKF.handle_echo_path_change(zero_filter=True)` calls `W.fill(0)` (entire array). This is over-aggressive.

**Architectural relationship**: tail-zero is meaningful only with dynamic
sizing (§2.5). If we keep fixed partitions, the "tail" beyond `current_size_partitions`
doesn't exist; tail-zero would be a no-op.

**Options**:

| Option | Description | Touch points |
|---|---|---|
| 2.2-A | Replace `W.fill(0)` with `W[current:].fill(0)` (tail-only). Requires `_active_size_partitions` runtime var to exist (§2.5). Paired with 2.5-A. | `PBFDAF.handle_echo_path_change`, `PBFDKF.handle_echo_path_change` — ~2-line change each, dependent on §2.5 |
| 2.2-B | Keep `W.fill(0)` semantic. Document as PBFDKF divergence WITH SEPARATE JUSTIFICATION explaining why our zero is full-W when AEC3's is tail-only. (Would need to characterise PBFDKF Kalman state and explain why zeroing the active prefix is desirable for our filter.) | No code change. Doc-only. |
| 2.2-C | Don't fire W=0 at all on delay change. Rely on H_error reset alone to re-arm Kalman. (Phase D mechanism, currently load-bearing.) | Already production behaviour. Document explicitly as our chosen interpretation of "W reset". |

**Open question**: does our PBFDKF Kalman P matrix track "filter trust"
in a way that would re-converge well from a full W=0, given the H_error
reset? Or would the W=0 + H_error=10000 combination over-trust the
random initial direction the Kalman picks?

### 2.3 Transient refined gain config (`SetConfig(refined_initial, true)`)

**AEC3 mechanism**: switch `RefinedFilterUpdateGain` to a config with
different `leakage_converged` (100×), `leakage_diverged` (10×) values;
`error_floor`, `error_ceil`, `noise_gate` stay identical at default
(confirmed rev 3, §3.1). The transient profile is materially more
permissive: H_error rebuilds faster after each Compute() call (`H_error
+= leakage × erl`), so Kalman gain stays large longer in the transient
phase. The smoothing window via `config_change_duration_blocks=250`
blocks = 1 s decays the transient profile back to steady linearly over
1 wall-second after `SetConfig(_, immediate=false)` is invoked from
`ExitInitialState`. Transient phase total duration =
`initial_state_seconds = 2.5 s` (default).

**Our current state (corrected rev 4)**: PBFDKF runs a **single
elevated steady-state leakage configuration** — no two-state machine.
Per-block leakage values (computed in `aec3_scale.py:LEAKAGE_*_PER_HOP_DEFAULT`,
consumed in `PBFDKF.__init__` `filters.py:424-429`):

- `leakage_converged_per_block ≈ 1e-3` = **20× AEC3 steady (5e-5)**,
  **5× lower than AEC3 transient (5e-3)**
- `leakage_diverged_per_block ≈ 1e-1` = **2× AEC3 steady (5e-2)**,
  **5× lower than AEC3 transient (5e-1)**

These values are **biased toward AEC3 transient** but always-on — they
do not switch with HEPC, and there is no smoothing window. This is the
**current functional adaptation candidate for Q1 stance (b)**.

**`_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost` are NOT the
Q1 adapter.** Per [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md):

- They operate on the **legacy `P` matrix** (now demoted to diagnostic
  per `filters.py:411-414` comment block: "H_error replaces our P matrix
  as the primary Kalman-like state in the K compute. P stays as a
  parallel field for backwards-compat (PathChangeRegimeHandler overrides
  / diagnostic)").
- They do NOT modulate `H_error_per_bin`, `leakage_converged`, or
  `leakage_diverged` — the three state quantities AEC3's
  `RefinedFilterUpdateGain::SetConfig` does modulate.
- They are orthogonal legacy/diagnostic mechanisms inherited from the
  pre-H_error PBFDKF, retained for the PathChangeRegimeHandler regime
  classifier; they are not adapters for the AEC3 transient gain config
  concept.

**Architectural relationship**: AEC3's `SetConfig(initial)` is a
**global state transition** (per-channel, on `H_error_per_bin` rebuild
rate via leakage) that lasts `initial_state_seconds = 2.5 s`. PBFDKF's
elevated-steady-leakage adaptation is **always-on, no transition
mechanism**. Functionally:

| Aspect | AEC3 `SetConfig(refined_initial)` | PBFDKF elevated-steady-leakage (Q1 (b)) |
|---|---|---|
| State quantity modulated | `leakage_converged`, `leakage_diverged` | `leakage_converged`, `leakage_diverged` — same state quantity |
| Transient bound | `initial_state_seconds = 2.5 s` (default) | None — always-on |
| Steady value | `leakage_converged = 5e-5`, `leakage_diverged = 5e-2` per block | `leakage_converged ≈ 1e-3`, `leakage_diverged ≈ 1e-1` per block (20× / 2× steady) |
| Transient value | `leakage_converged = 5e-3`, `leakage_diverged = 5e-1` per block (100× / 10× steady) | n/a — single-state |
| Smoothing on exit | Linear over `config_change_duration_blocks = 1 s` | n/a — no transition |
| Re-trigger on subsequent EPC | Re-fires; resets transient window | No transient window to reset |

**Options**:

| Option | Description | Touch points |
|---|---|---|
| 2.3-A — **Q1 stance (b)**, current code state, doc-only | Document the **elevated steady-state leakage values** (`aec3_scale.py:LEAKAGE_*_PER_HOP_DEFAULT`, consumed in `filters.py:424-429`) as the PBFDKF functional adaptation for the AEC3 transient gain config concept. State the documented equivalence caveats: (i) no two-state machine, no transient bound; (ii) no smoothing window; (iii) per-block values are biased toward AEC3 transient (20× steady on `leakage_converged`, 2× steady on `leakage_diverged`) but still 5× below AEC3 transient. | Doc-only. Validation: trace `H_error_per_bin` trajectory post-HEPC on a delay-event cohort vs an AEC3 trace (if available) to confirm the always-on elevated-leakage profile matches or comes within tolerance of AEC3's transient profile. |
| 2.3-B — **Q1 stance (a)**, strict AEC3 port | Introduce explicit `refined_initial` vs `refined` PBFDKF leakage configs. `SetConfig(refined_initial, immediate=true)` on HEPC (switch `_leakage_converged` / `_leakage_diverged` to transient values for `initial_state_seconds = 2.5 s`). `SetConfig(refined, immediate=false)` on `TransitionTriggered` (smooth linear decay back to steady over `config_change_duration_blocks = 1 s`). Requires §2.6 (TransitionTriggered hook). | New config fields; `PBFDKF.set_leakage_config(transient/steady, immediate)` method; orchestrator hook; smoothing-window state. ~50-line surface. Requires §2.6 for the smoothed revert. |
| 2.3-C — **Q1 stance (c)**, intentional v3.22 divergence | Document elevated-steady as deliberate divergence beyond AEC3 with separate justification (e.g. "PBFDKF Kalman-state per-bin H_error decays faster than AEC3's per-block H_error, requiring continuously higher leakage to maintain comparable Kalman gain magnitude"). Choose only if user explicitly authorises deviation beyond AEC3. | Doc-only. v3.22-class change. |

**Note on `_p_max_override` (corrected rev 4)**: not listed in this
options table. It is an **orthogonal legacy/P diagnostic mechanism**
operating on a different state quantity (legacy P matrix), not on the
AEC3 transient-config state quantity (leakage on `H_error_per_bin`).
The fact that `_p_max_override` fires on the same trigger sites
(EPV / shadow_rise / delay_shift) does not make it an AEC3 transient
config adapter. Future audits of `_p_max_override` / `_p_floor_beta` /
`_arc_m_q_boost` belong to the PathChangeRegimeHandler legacy-P arc,
not the AEC3 `RefinedFilterUpdateGain` parity arc.

### 2.4 Transient coarse gain config (`SetConfig(coarse_initial, true)`)

Same shape as §2.3 for the coarse / shadow side. AEC3 default
(confirmed rev 3, §3.1): `coarse_initial.rate = 0.9` vs
`coarse.rate = 0.7` (1.29× larger NLMS step in transient).
`noise_gate` identical.

**Our shadow PBFDAF current state (rev 4)**: PBFDAF has a single
fixed NLMS `mu` configuration; no transient-vs-steady rate split.
The legacy-P `_p_max_override` mechanism is **PBFDKF-only**
(`isinstance(filt, PBFDKF)` guard at `orchestrator.py:1932,
2565-2569, 2649-2653`); shadow PBFDAF has no analogous override of
any kind. Adopting AEC3's coarse transient config would require
introducing a coarse-side rate boost (e.g. a transient `mu` scale on
PBFDAF for ~1 s after HEPC).

**Coupling to Q1**: this question couples to Q1 stance. Under Q1=(a)
[strict AEC3 port], the symmetric coarse-side rate split would also
need to be ported. Under Q1=(b) [documented functional equivalence],
the coarse-side question becomes "does our shadow always-on `mu`
provide functional equivalence to AEC3's 0.7 → 0.9 → 0.7 rate
profile?" Empirically open; not resolvable from source.

**Open question — partially resolved rev 3**: PBFDAF is NLMS — step
size is already fresh every frame regardless of "filter age". AEC3
nevertheless boosts `coarse.rate` by 1.29× in transient. Whether the
benefit transfers to our PBFDAF is empirical, not knowable from
source. Defer as a coarse-side sub-question of Q1.

### 2.5 Dynamic `SetSizePartitions`

**AEC3 mechanism**: filter grows/shrinks across the EPC ↔ ExitInitialState
boundary. Initial length = `refined_initial.length_blocks` (default 12 in
WebRTC defaults; varies by stage tuning). Steady length =
`refined.length_blocks` (default 12-100+). Shrink on EPC (`immediate=true`),
grow on ExitInitialState (`immediate=false`, smooth over
`config_change_duration_blocks`).

**Our current state**: PBFDKF allocates `W`, `P`, `X_buf` at construction
to `n_partitions` (fixed, derived from `filter_length_ms`). No runtime
resize.

**Why AEC3 does this**:
- Shorter filter during transient = fewer partitions to fit; faster per-partition convergence (mu effective per partition is unchanged, but the total energy distributed over fewer partitions converges faster per partition).
- Once enough render energy is observed (`TransitionTriggered`), grow back to capture longer reverb tails.

**Options**:

| Option | Description | Touch points |
|---|---|---|
| 2.5-A | Implement runtime `_active_size_partitions` ≤ `n_partitions`. `W`, `P`, `X_buf` allocated at max size; computations restricted to `[0:_active]` slice. Zero-on-shrink semantics for tail. | LARGE surface: every PBFDKF compute step that loops over partitions needs to gate on `_active`. P-matrix block-diagonal shrink + restore. ~200-line surface + careful testing. |
| 2.5-B | Keep fixed partitions; document the design choice with PBFDKF-specific justification. Justification candidates: (a) our Kalman state in `P` would be lossy across shrink/grow; (b) PBFDKF's per-bin H_error is independent of partition count and provides the "fast adaptation" benefit directly without size change. | Doc-only. Needs separate sub-note explaining PBFDKF vs NLMS Kalman state preservation. |
| 2.5-C | Mark as intentional v3.22 divergence with separate justification (would require user explicit decision to deviate beyond AEC3). | Doc-only. Same justification work as 2.5-B but framed as divergence. |

**Q2 CLOSURE (rev 4) — documented fixed-size functional adaptation, no code.**

Closure rationale:
1. AEC3 default `refined_initial.length_blocks = 12` vs
   `refined.length_blocks = 13` (8% shrink, **1 partition** difference).
   The mechanism IS active at defaults but the magnitude is small.
2. Combined with the 1-second smoothing window, the effective
   transient-vs-steady filter length difference is modest.
3. PBFDKF Kalman state in `P` would be lossy across shrink/grow
   (block-diagonal shrink + restore not trivially invertible).
4. PBFDKF's per-bin `H_error_per_bin` reset provides per-bin
   fast-adaptation independently of partition count — i.e. the
   "fast adaptation" benefit AEC3 obtains via shrink is already
   covered by our `H_error_per_bin.fill(10000)` on HEPC.
5. Porting a HIGH-cost mechanism (~200 lines, touches innermost
   loops + Kalman state) for a 1-partition shrink that AEC3 itself
   smooths over 1 second is **disproportionate** to expected benefit.

**Closure form**: Q2 is closed as a **documented fixed-size functional
adaptation** — fixed `n_partitions=6` is deliberate, with the
PBFDKF-specific justifications (3) + (4) above. No code is written.
This closure is recorded in §0 TL;DR, §3.5, §4.2, and here. **The
v3.21.x parity arc for Q2 is complete at the doc level.**

Empirical confirmation of (4) (whether PBFDKF Kalman re-converges
fully on H_error reset without partition shrink) belongs to the same
delay-event cohort trace that Q1 needs — not a Q2 blocker.

### 2.6 `ExitInitialState` paired transition

**AEC3 mechanism**: fires once on `aec_state_.TransitionTriggered()` —
a single-edge signal when `strong_not_saturated_render_blocks_` crosses
the initial-state threshold (default 2.5 s; 5 s in conservative mode).

**Our current state (rev 4)**: no `TransitionTriggered` hook. AecState
has `initial_state_active` per `_initial_state.update`, but no consumer
for its transition edge. **No analogous mechanism exists in PBFDKF**:
- `_p_max_override` countdowns are **NOT** an ExitInitialState analogue
  — they operate on the legacy P matrix (orthogonal to AEC3's
  leakage / H_error transient mechanism) and fire on per-event
  trigger sites, not on a global render-energy threshold.
- PBFDKF's elevated steady leakage values (Q1 (b) candidate, §2.3) have
  no exit transition — they are always-on.

**Architectural relationship (rev 4)**: ExitInitialState only matters
if we adopt §2.3-B (strict AEC3 port — Q1 stance (a)). Q2 is CLOSED
(no dynamic sizing); §2.5-A is off the table. Without Q1=(a), there
is no transient bound to "exit" from.

**Options**:

| Option | Description | Touch points |
|---|---|---|
| 2.6-A | Add `AecState.TransitionTriggered` accessor + orchestrator hook that calls a new `PBFDKF.exit_initial_state()` callback invoking `SetConfig(refined, immediate=false)`. Implementation depends on Q1=(a) [§2.3-B strict port]. | Conditional on §2.3-B. |
| 2.6-B | Don't introduce. PBFDKF runs always-elevated steady leakage with no transition; nothing to exit. Q1=(b) or Q1=(c) closure makes this the chosen path. | No code. |
| 2.6-C | Add `AecState.TransitionTriggered` accessor only (no consumer yet) for trace + future use. | Audit-only: ~5-line accessor + diag dump. |

### 2.7 Coarse filter reset hangover

**AEC3 mechanism** (`subtractor.cc:264-265, 296-307`): when coarse is
copied from refined (after `poor_coarse_filter_counters_[ch] >= 5`),
arms `coarse_filter_reset_hangover_[ch] = config_.filter.coarse_reset_hangover_blocks`
which suppresses `disallow_leakage_diverged` during the hangover.
NOT part of `HandleEchoPathChange`. Internal coarse-only machinery.

**Our current state**: not relevant to delay-change chain. Out of scope
for this design note.

---

## 3. Source-derived facts before user decision (2026-05-24 fact-gathering pass)

Read-only pass over AEC3 extracts (`docs/aec3_extracts/src/aec3/`) and
our code (`python/modules/{config,filters,orchestrator,aec3_scale}.py`).
Purpose: replace earlier "open questions" with the facts that source
inspection can establish, so the user only decides genuine architecture
tradeoffs (§4) rather than facts.

### 3.1 AEC3 default config values

Source: `docs/aec3_extracts/api/audio/echo_canceller3_config.h` (fetched
2026-05-24 from `webrtc-mirror/webrtc/main`).

#### Block / counter constants

| Field | Default | Source | Wall time |
|---|---|---|---|
| `kBlockSize` | 64 samples | `aec3_common.h:47` | 4 ms @ 16 kHz |
| `kNumBlocksPerSecond` | 250 | `aec3_common.h:28` | — |
| `kHErrorInitial` | 10000 | `refined_filter_update_gain.cc:31` | — |
| `kPoorExcitationCounterInitial` | 1000 blocks | `refined_filter_update_gain.cc:32` | 4 s @ 4 ms/block |
| `kMaxBlocksPerFrame` (gain_change rate-limit) | 3 blocks | `echo_remover.cc:389` | 12 ms |

#### Filter `RefinedConfiguration` — steady (`refined`) vs transient (`refined_initial`)

Source: `api/audio/echo_canceller3_config.h:88-100`.

| Field | `refined` (steady) | `refined_initial` (transient) | Ratio (initial / steady) |
|---|---:|---:|---|
| `length_blocks` | 13 | 12 | 0.92× (shrink by 1 partition) |
| `leakage_converged` | 0.00005f | 0.005f | **100×** (faster H_error rebuild when converged) |
| `leakage_diverged` | 0.05f | 0.5f | **10×** (faster H_error rebuild when diverged) |
| `error_floor` | 0.001f | 0.001f | 1× (same) |
| `error_ceil` | 2.f | 2.f | 1× (same) |
| `noise_gate` | 20075344.f | 20075344.f | 1× (same) |

#### Filter `CoarseConfiguration` — steady (`coarse`) vs transient (`coarse_initial`)

Source: `api/audio/echo_canceller3_config.h:102-110`.

| Field | `coarse` (steady) | `coarse_initial` (transient) | Ratio |
|---|---:|---:|---|
| `length_blocks` | 13 | 12 | 0.92× (shrink by 1 partition) |
| `rate` | 0.7f | 0.9f | **1.29×** (larger NLMS step in transient) |
| `noise_gate` | 20075344.f | 20075344.f | 1× (same) |

#### Top-level filter timing

Source: `api/audio/echo_canceller3_config.h:112-118`.

| Field | Default | Wall time | Notes |
|---|---|---|---|
| `config_change_duration_blocks` | **250** | **1000 ms** (1 s) @ 4 ms/block | Smoothing window for `SetConfig(_, immediate=false)` |
| `initial_state_seconds` | **2.5f** | 2.5 s | Transient-phase length when `conservative_initial_phase=false` |
| `conservative_initial_phase` | **false** | — | Default ⇒ transient is 2.5 s, not 5 s |
| `coarse_reset_hangover_blocks` | 25 | 100 ms | NOT EPC chain — coarse-only mechanism |
| `enable_coarse_filter_output_usage` | true | — | UseRefinedOutput selector default |
| `use_linear_filter` | true | — | |

#### Confirmed facts (rev 3, previously unknown in rev 2)

1. **`refined_initial.length_blocks` < `refined.length_blocks`** at default (12 < 13). AEC3 actively shrinks-then-grows by ONE partition at default settings — machinery is active, magnitude is small (8% size change).
2. **`refined_initial` DOES differ from `refined` beyond `length_blocks`**: `leakage_converged` is 100× larger, `leakage_diverged` is 10× larger. Other fields identical. The transient gain profile **is** materially more permissive (rebuilds H_error faster, keeping Kalman gain large longer post-HEPC).
3. **`coarse_initial` DOES differ from `coarse` in `rate`**: 0.9 vs 0.7 → 1.29× larger NLMS step in transient. Shadow filter adapts faster during transient.
4. **`config_change_duration_blocks = 250`** = 1 second smoothing window. **No PBFDKF analogue** — we have no smoothing window on `leakage_*` (always-on elevated steady values; see §2.3 rev 4). (Note rev 4: `_p_max_override_frames = 30` is a legacy-P diagnostic countdown — NOT a leakage smoothing window analogue. The earlier rev 3 "~3.3× shorter" comparison was misleading and is retracted.)
5. **`initial_state_seconds = 2.5f`** (`conservative_initial_phase=false` default) = 2.5 second transient phase. **No PBFDKF analogue** — we have no transient-vs-steady leakage state machine (always-on elevated steady values; see §2.3 rev 4). (Note rev 4: `_p_max_override` is not a transient-phase analogue — see §2.3 + Q1 mapping audit.)

### 3.2 Our PBFDKF current state

This table catalogues PBFDKF / orchestrator internal fields. **Rev 4
clarification**: the `_p_max_override` / `_p_floor_beta` /
`_arc_m_q_boost` rows are recorded here for completeness but are
**NOT AEC3 `RefinedFilterUpdateGain` equivalents** — they operate on
the legacy P matrix (demoted diagnostic state per
`filters.py:411-414`), not on `H_error_per_bin` or `leakage_*` (the
primary Kalman state). For Q1 parity mapping, the relevant rows are
`H_error_per_bin` + `leakage_*` (which appear here and in §3.1's AEC3
table). For the legacy-P arc audit, see [PathChangeRegimeHandler / regime classifier].

| Field | Value (BALANCED default) | Source |
|---|---|---|
| `sample_rate` | 16000 | `config.py:1306-1308` (default branch) |
| `hop_size` | 160 samples = 10 ms | `config.py:1299-1300` (default 160 at 16 kHz) |
| `filter_length` | 832 samples = 52 ms | `config.py:1308` |
| `n_partitions` (computed) | **6** | `config.py:1325-1329`: `(832 + 160 − 1) // 160 = 6` (comment line 1328 explicitly states "6 partitions × 160 = 960-sample filter") |
| `W` shape | `(6, n_freqs)` fixed at construction | `filters.py:109` |
| `X_buf` shape | `(6, n_freqs)` fixed | `filters.py:112` |
| `P` shape (PBFDKF, legacy diagnostic) | `(6, n_freqs)` fixed | `filters.py:384` |
| **Active-partition runtime var** | **None — no `_active_size_partitions` analogue exists** | direct read of `filters.py` PBFDKF class |
| `H_error_per_bin` shape | `(n_freqs,)` per-bin (primary Kalman state) | `filters.py:419-421` |
| `H_error_per_bin` reset value (HEPC) | `10000.0` | `aec3_scale.py:82` `H_ERROR_INIT_FLOAT = 10000.0` |
| `H_error_per_bin` floor (clamp) | `1e-3` | `aec3_scale.py:83` `H_ERROR_FLOOR_FLOAT` (matches AEC3) |
| `H_error_per_bin` ceil (clamp) | `100.0` | `aec3_scale.py:84` `H_ERROR_CEIL_FLOAT` (**50× AEC3 default of 2.0** — out of Q1 scope; recorded for separate clamp audit) |
| `_leakage_converged` per-block | `≈ 1e-3` | `aec3_scale.py:87` `LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-3, 160, 16000)` (20× AEC3 steady) |
| `_leakage_diverged` per-block | `≈ 1e-1` | `aec3_scale.py:88` `LEAKAGE_DIVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-1, 160, 16000)` (2× AEC3 steady) |
| Transient-vs-steady leakage state machine | **None** — always-on elevated steady values | direct read; Q1 (b) functional adaptation candidate |
| `_poor_excitation_counter` HEPC reset value | `400` hops = 4 s wall time | `aec3_scale.py:115` `POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT = blocks_to_hops(1000, 160, 16000) = 400` |
| `_call_counter` HEPC reset value | `0` | `filters.py:153, 193` |
| `_call_counter` increment site | each filter `.process()` call | `filters.py:276, 543` |
| `_poor_excitation_counter` increment site | orchestrator (not filter); `+= 1` per frame, `= 0` when `RenderSignalAnalyzer.poor_signal_excitation()` fires | `orchestrator.py:2111-2114` (main) + 2121-2124 (shadow) |
| Skip-update gate | `if _call_counter <= n_partitions OR _poor_excitation_counter < n_partitions: skip` | `filters.py:277-278` (PBFDAF), 544-545 (PBFDKF) |
| Effective freeze on fresh HEPC (assuming render keeps being active) | `_call_counter=0` → ~6 hops = 60 ms | inferred from above; `_poor_excitation_counter=400` ≫ 6, doesn't gate by itself |
| **Legacy P diagnostic mechanisms (NOT Q1 adapters; for catalogue / future legacy-P arc only)** | | |
| `_p_max_override` standard EPC fire (EPV / shadow_rise / delay_shift) | `_p_max_override=1.0`, `_p_max_override_frames=30` (300 ms) | `orchestrator.py:1933-1934, 2572-2573, 2659-2660` |
| `_p_max_override` saturation handler | `_p_max_override_frames=20` (200 ms) | `orchestrator.py:2417-2418` |
| `_p_max_override` one specialised site | `_p_max_override_frames=15` (150 ms) | `orchestrator.py:2457-2459` |
| `_p_floor_beta` standard EPC fire | `_p_floor_beta=1.0`, `_p_floor_beta_frames=30` (300 ms) | `orchestrator.py:2574-2575, 2661-2662` |
| `_arc_m_q_boost` fire | invoked on EPV / shadow_rise / delay_shift | `orchestrator.py:1929, 2569, 2644, 2653` etc. |
| Shadow PBFDAF legacy-P override | **NONE** — `_p_max_override` is guarded by `isinstance(filt, PBFDKF)` everywhere it fires; shadow PBFDAF has no `_p_max_override` attribute | `orchestrator.py:1932, 2565-2569, 2649-2653` (the `isinstance(filt, PBFDKF)` guard skips PBFDAF) |
| Shadow PBFDAF post-HEPC state | counter resets only (via `PBFDAF.handle_echo_path_change` at `filters.py:188-193`); no gain override | direct read |

**Interaction summary** of H_error reset + counter reset + adaptation
gating:
- After HEPC: `H_error_per_bin = 10000` (high → Kalman mu large) + `_call_counter=0` (gate fires for first 6 frames = 60 ms) + `_poor_excitation_counter=400` (does not gate by itself).
- During the 60 ms freeze: filter does NOT adapt. Output uses pre-HEPC W as starting point (unless M2 W=0 ON, in which case output is mic-passthrough).
- After the 60 ms: filter resumes. Large `H_error` drives large Kalman gain for ~10-20 frames before decaying via `H_error -= 0.5 mu X² H_error`.
- `_p_max_override` (when armed) holds the **legacy P matrix** ceiling
  at 1.0 for the next 30 frames (300 ms), suppressing P-driven
  diagnostic shrinkage. **NOTE rev 4**: this affects the demoted
  diagnostic P state only, not `H_error_per_bin` (the primary Kalman
  state used in the K compute per `filters.py:411-414`). It is an
  orthogonal legacy mechanism, not a transient leakage analogue.
- The 60 ms freeze + ~100 ms H_error-driven Kalman re-arm =
  effective filter-side transient recovery of ~160 ms per EPC event.
  Always-elevated steady leakage (Q1 (b) candidate) continues to
  drive higher Kalman gain than AEC3 steady values for the rest of
  the case, with no transition mechanism.

### 3.3 Time-constant comparison (rev 4)

| Time-constant | AEC3 (default, confirmed rev 3) | Ours | Comparable? |
|---|---|---|---|
| Block / hop duration | 4 ms (`kBlockSize=64`) | 10 ms (`hop_size=160`) | Different time-base; convert blocks ↔ hops via 2.5× ratio |
| H_error reset value | 10000 | 10000 | Same (intentional parity, `aec3_scale.py:82`) |
| poor_excitation counter HEPC reset value | 1000 blocks = 4 s wall | 400 hops = 4 s wall | **Same wall-time** (intentional parity via `blocks_to_hops`) |
| Skip-gate freeze (`call_counter ≤ size_partitions`) | 13 blocks = 52 ms (refined; `length_blocks=13` confirmed) | 6 hops = 60 ms | **Comparable wall-time** (52 ms vs 60 ms — within ~15%) |
| `leakage_converged` per-block (steady profile, post-transient) | 5e-5 | ~1e-3 (always-on; rev 4) | **NOT comparable** — Ours is **20× higher** than AEC3 steady (functional adaptation, Q1 (b) candidate). We have no two-state profile. |
| `leakage_diverged` per-block (steady profile, post-transient) | 5e-2 | ~1e-1 (always-on; rev 4) | **NOT comparable** — Ours is **2× higher** than AEC3 steady. We have no two-state profile. |
| `leakage_converged` per-block (transient profile, HEPC → 2.5 s) | 5e-3 | n/a (always-on steady) | **NOT comparable** — Ours is **5× lower** than AEC3 transient. No transient bound. |
| `leakage_diverged` per-block (transient profile, HEPC → 2.5 s) | 5e-1 | n/a (always-on steady) | **NOT comparable** — Ours is **5× lower** than AEC3 transient. No transient bound. |
| `config_change_duration_blocks` smoothing window | **250 blocks = 1000 ms** (default, confirmed rev 3) | n/a — no smoothing window on leakage values | **NOT comparable** — we have no two-state profile to smooth between. |
| Transient phase total duration (HEPC → ExitInitialState) | **`initial_state_seconds = 2.5 s`** (default, `conservative_initial_phase=false`). Conservative mode = 5 s. Field-trial overrides 0–2 s. | n/a — we have no `TransitionTriggered`; always-on steady leakage instead | **NOT comparable** — fundamentally different architecture. |
| `_p_max_override` per-event countdown (legacy P diagnostic) | n/a (AEC3 has no legacy P override) | 30 frames = 300 ms | **NOT a Q1 comparison row.** Legacy P diagnostic mechanism, orthogonal to AEC3 `RefinedFilterUpdateGain`. Belongs to PathChangeRegimeHandler legacy-P arc. |
| `gain_change_hangover` (gain_change rate-limit) | 3 blocks = 12 ms | n/a — EPV/shadow_rise can fire every hop | **NOT comparable** — Audit B §5.4 showed our cohort has 251-frame min gap → no observable surface anyway |

**Verdict (rev 4)**: H_error reset value, poor_excitation reset
value, and ~52 ms freeze are wall-time comparable to AEC3 default
(intentional parity). The transient-phase architecture differs in
three orthogonal ways:
(1) AEC3 has a 1-second leakage smoothing window; we have none.
(2) AEC3's transient phase lasts 2.5 s (default); ours is **not
present** — we run always-on elevated steady leakage instead (Q1 (b)
candidate).
(3) AEC3 modulates `leakage_converged` (100×) and `leakage_diverged`
(10×) on a two-state machine; we modulate the **same state quantity**
(`leakage_converged` / `leakage_diverged`) but **only with elevated
steady values** (20× / 2× AEC3 steady), no transient profile and no
transition. **`_p_max_override` is NOT in this comparison set** —
it operates on the legacy P matrix (demoted diagnostic state per
`filters.py:411-414`), not on the leakage-driven H_error update path
AEC3 `RefinedFilterUpdateGain` modulates.

### 3.4 Tail-zero semantics

**Factual answer**: AEC3-style `ZeroFilter(current, max)` would be a
**structural no-op in our current architecture**.

Evidence:
- AEC3 allocates `H_` at `max(refined_initial.length_blocks, refined.length_blocks)` (`subtractor.cc:96-97`). `current_size_partitions_` is a runtime variable that can be ≤ `max_size_partitions_` (set via `SetSizePartitions`). The "tail" is indices `[current, max)`.
- Our PBFDKF allocates `W` at `n_partitions` (computed once, fixed). There is no `_active_size_partitions` runtime variable. The "tail" beyond `n_partitions` does not exist.
- Therefore `ZeroFilter(n_partitions, n_partitions)` would zero an empty range.

**Implication**: tail-zero cannot be meaningfully ported as a standalone
change. It is only meaningful as PART OF the dynamic-partition machinery
(which would introduce `_active_size_partitions` and make the tail
non-empty when shrunk). The two changes are inseparable.

This also means the "M2 over-aggressive" framing (Audit B §0 TL;DR + §2
rows 1-2) has an extra dimension:
- **If fixed partitions are retained**: AEC3 tail-zero has no direct
  surface in our architecture (`ZeroFilter(n_partitions, n_partitions)`
  zeros an empty range). Current `W.fill(0)` remains a **non-AEC3
  default-OFF substrate and should NOT ship unless separately
  justified** (it is more aggressive than the AEC3 equivalent
  no-op, with no compensating benefit established).
- **If dynamic partitions are added**: AEC3 tail-zero becomes correct
  (preserves active prefix); our `W.fill(0)` would be a **bug**
  (destroys active state AEC3 deliberately preserves).

So **Q4 (W.fill(0) vs tail-zero)** is not a standalone question. It
collapses into Q2 (dynamic partitions yes/no). In **both** Q2 stances,
the current `W.fill(0)` is unfit to ship as production default:
- Q2=(a) → `W.fill(0)` is a bug; replace with tail-zero.
- Q2=(b)/(c) → `W.fill(0)` is a non-AEC3 substrate with no surface
  benefit; keep default-OFF substrate; do not ship without separate
  justification.

### 3.5 Reclassification of Q1–Q4 post-facts

| Q | Original framing | Facts known (§3.1-3.4 + Q1 mapping audit) | Unknowns requiring experiment | Genuine user decision | v3.21.x parity vs v3.22? | Status |
|---|---|---|---|---|---|---|
| Q1 transient → steady leakage config split | Open question | (rev 4) AEC3 default transient phase is **2.5 s** (`initial_state_seconds`), smoothed back via **1 s** `config_change_duration_blocks`. AEC3 modulates `leakage_converged` 100× + `leakage_diverged` 10× + coarse `rate` 1.29× during transient (confirmed §3.1). Our PBFDKF runs **elevated steady-state leakage** (`leakage_converged_per_block ≈ 1e-3` = 20× AEC3 steady, 5× lower than AEC3 transient; `leakage_diverged_per_block ≈ 1e-1` = 2× AEC3 steady, 5× lower than AEC3 transient) with **no two-state machine, no smoothing window**. `_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost` are **NOT Q1 adapters** — they operate on the legacy P matrix (diagnostic state), not on `H_error_per_bin` or `leakage_*` (the primary Kalman state). See [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md). | Empirical question: does PBFDKF's always-elevated-steady-leakage profile track AEC3's 2.5 s transient → 1 s smoothing → steady trajectory closely enough to be a documented functional adaptation, or does the missing transient bound materially degrade post-HEPC recovery? Requires delay-event / EPC-rich cohort trace. | (a) **Strict AEC3 port** = transient leakage config: `refined_initial` `leakage_converged` / `leakage_diverged` for 2.5 s, then `refined` `leakage_converged` / `leakage_diverged` smoothed back over 1 s. (b) **Current PBFDKF functional adaptation candidate** = elevated steady leakage values in `aec3_scale.py` / `filters.py` accepted as documented equivalence. (c) Intentional v3.22 divergence. | Closes v3.21.x parity gap under either (a) [strict mechanism port] or (b) [documented functional equivalence]. v3.22 divergence ONLY if (c) is explicitly chosen with separate justification. | **OPEN v3.21.x parity question.** Next step = delay-event cohort identification + `H_error_per_bin` post-HEPC trajectory trace. NOT implementation. |
| Q2 dynamic partition sizing | Open question | (rev 3) Our partitions fixed at `n_partitions=6`. AEC3 default `refined_initial.length_blocks=12` vs `refined.length_blocks=13` (8% shrink, **1 partition** difference). Tail-zero only meaningful with `_active_size_partitions`. Machinery IS active at AEC3 defaults but magnitude is small. | None. PBFDKF Kalman P-matrix shrink/restore is lossy; H_error reset covers per-bin fast-adaptation independently of partition count; AEC3 default magnitude (1 partition + 1 s smoothing) is small relative to porting cost (~200 lines, innermost loops). | None — Q2 is closed at the doc level. | **N/A — closed as documented fixed-size functional adaptation, no code.** | **CLOSED — documented fixed-size functional adaptation, no code.** Closure recorded in §0 TL;DR, §2.5 (rev 4), §3.5 here, §4.2. |
| Q3 exit-transient global vs per-event | Open question | With Q2 closed, Q3 reduces to a single-axis sub-question of Q1: only meaningful if Q1=(a) [strict port]. If Q1=(b) [documented equivalence] or Q1=(c) [v3.22 divergence], there is no transient bound to "exit". | None — fully coupled to Q1. | **DEFERRED — fate follows Q1.** | Same as Q1. | **DEFERRED — couples to Q1.** |
| Q4 `W.fill(0)` vs tail-zero | Standalone question | **ANSWERED in §3.4**: tail-zero is structural no-op without dynamic partitions; collapses into Q2. With Q2 closed (fixed-size), AEC3 tail-zero has no direct surface. | None. | **No standalone decision** — answer follows Q2 (CLOSED). | Current `W.fill(0)` remains a **non-AEC3 default-OFF substrate**, do not ship unless separately justified (no AEC3-equivalent benefit established). | **CLOSED — follows Q2 (CLOSED).** |

---

## 4. Architectural questions (narrowed by §3 facts + Q1 mapping audit)

### Q1. Does PBFDKF need a transient → steady leakage config split?

**Status**: **OPEN v3.21.x parity question.**

**Facts** (§3.1-3.3 + [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md)):
- AEC3's transient phase lasts **2.5 s globally** at default
  (`initial_state_seconds=2.5`, `conservative_initial_phase=false`).
  Conservative mode would extend to 5 s; field-trial overrides span 0–2 s.
- AEC3's `config_change_duration_blocks` is **250** (default; 1 s
  smoothing wall). Short field-trial sets to 10 blocks (40 ms).
- AEC3's `refined_initial` differs from `refined` on:
  `leakage_converged` 100× larger (5e-3 vs 5e-5 per block),
  `leakage_diverged` 10× larger (5e-1 vs 5e-2 per block),
  `length_blocks` 12 vs 13. Other fields identical.
- PBFDKF's per-block leakage values:
  `leakage_converged ≈ 1e-3` = **20× AEC3 steady**, **5× lower than
  AEC3 transient**; `leakage_diverged ≈ 1e-1` = **2× AEC3 steady**,
  **5× lower than AEC3 transient**. Always-on, no two-state machine,
  no smoothing window.
- **`_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost` are NOT
  Q1 adapters.** They operate on the legacy P matrix (now demoted to
  diagnostic per `filters.py:411-414`), not on `H_error_per_bin` or
  the leakage values. They are orthogonal legacy/diagnostic
  mechanisms; future audits of them belong to the
  PathChangeRegimeHandler legacy-P arc, not the AEC3
  `RefinedFilterUpdateGain` parity arc.

**Genuine user decision** (only after the above fact set is accepted):

- **(a) Strict AEC3 mechanism port** — Implement
  `SetConfig(refined_initial, immediate=true)` on HEPC:
  switch `_leakage_converged` and `_leakage_diverged` to AEC3
  transient values (5e-3 / 5e-1 per block) for
  `initial_state_seconds = 2.5 s`. Then on `TransitionTriggered`,
  `SetConfig(refined, immediate=false)`: smooth linear decay back to
  AEC3 steady values (5e-5 / 5e-2 per block) over
  `config_change_duration_blocks = 1 s`. v3.21.x parity. Code surface
  medium (~50 lines). Couples to Q3=(a) [TransitionTriggered hook].

- **(b) Documented functional equivalence (current code state)** —
  Document PBFDKF's **elevated steady-state leakage values**
  (`LEAKAGE_CONVERGED_PER_HOP_DEFAULT` / `LEAKAGE_DIVERGED_PER_HOP_DEFAULT`
  in `python/modules/aec3_scale.py`; consumed in
  `python/modules/filters.py:PBFDKF.__init__`) as the functional
  adaptation candidate for the AEC3 transient gain config concept.
  Doc-only. Required disclosures: (i) no two-state machine, no
  transient bound, no smoothing window; (ii) per-block values are
  biased toward AEC3 transient (20× steady on `leakage_converged`,
  2× steady on `leakage_diverged`) but still 5× below AEC3 transient;
  (iii) the time-constant gap and missing transition mechanism are
  accepted as PBFDKF-specific functional adaptation. v3.21.x parity-by-adaptation.

- **(c) Intentional v3.22 divergence** — Explicitly document
  elevated-steady leakage as deliberate deviation beyond AEC3 with
  separate justification (e.g. PBFDKF Kalman-state dynamics that
  require continuously higher leakage). Only choose if user
  explicitly authorises deviation beyond AEC3.

**Next step**: NOT implementation. **Identify a delay-event /
EPC-rich cohort** (12-case has zero `delay_first` / `delay_shift`;
need either a subset of the 800-case AEC challenge dataset or a
synthetic controlled-delay cohort). Trace per-frame `H_error_per_bin`
trajectory post-HEPC against AEC3-style reference behaviour (or
mathematical reference: simulated AEC3 transient leakage profile).
Determine whether stance (b) is empirically close enough to AEC3 to
close the parity gap, or whether stance (a) [strict port] is required.
Per user direction: no code, no benchmark, no cohort run in this round.

### Q2. Does PBFDKF need dynamic partition sizing?

**Status**: **CLOSED — documented fixed-size functional adaptation, no code.**

**Facts** (§3.1-3.4):
- AEC3 has `_active_size_partitions` runtime variable; we don't.
- AEC3 default shrinks by 1 partition (12 vs 13) with 1 s smoothing
  — magnitude small.
- Tail-zero is structural no-op in our fixed-size architecture (§3.4).
- PBFDKF Kalman P-matrix shrink/restore would be lossy.
- PBFDKF's per-bin `H_error_per_bin.fill(10000)` on HEPC already
  provides per-bin fast-adaptation independently of partition count.

**Closure rationale** (§2.5 rev 4): porting a HIGH-cost mechanism
(~200 lines, innermost loops + Kalman state) for a 1-partition shrink
that AEC3 itself smooths over 1 second is disproportionate to expected
benefit. PBFDKF's H_error reset already covers the "fast adaptation"
benefit AEC3 obtains via shrink.

**Closure form**: fixed `n_partitions=6` is deliberate, documented as
PBFDKF-specific functional adaptation. **No code.** Closure recorded
in §0 TL;DR, §2.5 (rev 4), §3.5, here.

### Q3. Should "exit transient phase" be global or per-event?

**Status**: **DEFERRED — fate follows Q1.**

**Facts** (§3.2):
- We have NO `TransitionTriggered` accessor or hook.
- With Q2 closed, Q3 reduces to a single-axis sub-question of Q1.

**Decision**:
- Only meaningful if Q1=(a) — otherwise nothing global to "exit".
- If Q1=(a): implement `AecState.TransitionTriggered` accessor +
  orchestrator hook + `PBFDKF.exit_initial_state()` callback that
  invokes `SetConfig(refined, immediate=false)` to start the 1 s
  smoothed revert. v3.21.x parity. ~30-line surface.
- If Q1=(b) or Q1=(c): no transient bound to exit. No change needed.

### Q4. `W.fill(0)` vs `ZeroFilter(current, max)` — CLOSED

**Status**: **CLOSED — follows Q2 (CLOSED).**

**Facts** (§3.4):
- Tail-zero is no-op without dynamic partitions.
- Q2 closed as documented fixed-size functional adaptation.

**Disposition**: AEC3 tail-zero has no direct surface in our
architecture. Current `W.fill(0)` remains a **non-AEC3 default-OFF
substrate** and **should not ship unless separately justified** (no
AEC3-equivalent benefit established).

---

## 5. Cost / risk (rev 4)

| Option | Code surface | Validation prerequisite | Risk to production |
|---|---:|---|---|
| §2.3-A — Q1 (b) doc-only (elevated steady leakage as functional adaptation) | 0 lines | Trace per-frame `H_error_per_bin` trajectory post-HEPC on a delay-event cohort, compare vs simulated AEC3 transient leakage profile (read-only) | None |
| §2.3-B — Q1 (a) strict AEC3 port (transient leakage config + 1 s smoothing + 2.5 s state machine) | ~50 lines (`refined_initial` / `refined` config fields; `PBFDKF.set_leakage_config(profile, immediate)` method; smoothing-window state; orchestrator HEPC + TransitionTriggered hooks) | Trace + delay-event cohort + 12-case + 800-case bench | Medium (changes PBFDKF leakage rebuild rate post-HEPC) |
| §2.3-C — Q1 (c) intentional v3.22 divergence | 0 lines | PBFDKF Kalman-state vs AEC3 H_error rebuild-rate justification note | None (doc) |
| ~~§2.5-A (dynamic partitions)~~ | ~~CLOSED — Q2 not implementing~~ | n/a | n/a |
| §2.5-B (Q2 CLOSED — documented fixed-size functional adaptation) | 0 lines | None — closed at doc level (§2.5 rev 4 + §4.2) | None |
| §2.6-A (TransitionTriggered hook) | ~30 lines (`AecState.TransitionTriggered` accessor + orchestrator dispatch + `PBFDKF.exit_initial_state()` callback) | **Conditional on §2.3-B only** (Q2 is CLOSED). | Low if no consumer wired |
| §2.6-C (audit-only accessor) | ~5 lines | None | None |

**Note rev 4**: `_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost`
are not in this options table — they are legacy P diagnostic mechanisms
on an orthogonal state quantity. Audits of them belong to the
PathChangeRegimeHandler legacy-P arc, not the AEC3
`RefinedFilterUpdateGain` parity arc covered here.

---

## 6. Validation prerequisite (before any implementation)

**No design choice in this note is testable without a cohort that has
real delay-change events.** The current 12-case cohort has zero
`delay_first` / `delay_shift` fires (Audit B §5.4 + round-2 audit §3.1).
Phase E catastrophe was driven by EPV / shadow_rise mis-classified as
delay_change — not by real delay events.

**Pre-implementation steps in order**:
1. Identify a cohort with real delay events. Options:
   - (a) Render the 800-case AEC challenge dataset with current
     production main + trace `delay_first` / `delay_shift` per case;
     identify subset that fires events.
   - (b) Construct a synthetic dataset with known delay shifts at known
     times (controlled cohort).
2. Run that cohort with current production main and verify delay events
   actually fire on the captured per-frame diag.
3. Only AFTER (1) and (2) succeed, evaluate which options in §3 are
   testable; the rest stay theoretical.

**No 800-case render is in scope for this design note** — that's a
separate execution step the user must authorise.

---

## 7. Remaining unknowns (status after rev 4)

§3 (rev 2) closed: hop-size, n_partitions, H_error reset value,
poor_excitation counter value, freeze duration, override durations,
shadow PBFDAF override presence, and tail-zero semantics.

§3.1 (rev 3) closed three of the four open questions previously listed
in rev 2 §7 by fetching `api/audio/echo_canceller3_config.{h,cc}` from
the upstream WebRTC mirror:

1. ~~**AEC3 default `refined_initial.length_blocks` vs `refined.length_blocks`**~~ — **ANSWERED rev 3**: 12 vs 13 (8% shrink, 1 partition). Machinery active at default, small magnitude.
2. ~~**AEC3 default `config_change_duration_blocks`**~~ — **ANSWERED rev 3**: 250 blocks = 1 s wall.
3. ~~**Whether `refined_initial` and `refined` differ beyond `length_blocks`**~~ — **ANSWERED rev 3**: YES. `leakage_converged` 100× larger, `leakage_diverged` 10× larger; `error_floor` / `error_ceil` / `noise_gate` identical. Coarse `rate` 1.29× larger in transient.

**Rev 4 status update — Q1 mapping audit findings**:

- The earlier framing "Q1 stance (b) = document `_p_max_override` as the
  parity adapter" was **wrong**. `_p_max_override` operates on the
  legacy P matrix (diagnostic state), not on `H_error_per_bin` or the
  `leakage_*` values. The corrected Q1 (b) framing — elevated steady
  leakage values in `aec3_scale.py` / `filters.py` — is now reflected
  in §0, §2.3, §3.5, §4.1. The Q1 mapping audit is at
  [docs/v3_21_q1_mapping_audit.md](v3_21_q1_mapping_audit.md).
- Q2 is **CLOSED** as documented fixed-size functional adaptation
  (no code). See §0 TL;DR + §2.5 (rev 4) + §3.5 + §4.2.

**Remaining unknowns** (empirical PBFDKF questions; not in source extracts):

4. **Does PBFDKF Kalman gain self-correct from full `W=0` + `H_error=10000`
   start?** — affects whether option 2.2-B (keep `W.fill(0)` semantic
   under Q2=CLOSED) is safe in isolation. (Q4 ruling: current
   `W.fill(0)` stays default-OFF substrate; do not ship unless
   separately justified.)
   - **Resolution path**: PBFDKF mathematical analysis or controlled
     simulation. NOT a benchmark. Deferred per "no implementation" rule.

5. **Does PBFDKF's elevated-steady-leakage profile (Q1 (b)) track
   AEC3's 2.5 s transient → 1 s smoothing → steady trajectory closely
   enough to close the Q1 parity gap, or does the missing transient
   bound materially degrade post-HEPC recovery?** — the central
   v3.21.x Q1 question that the rev 4 reframe surfaces.
   - **Resolution path**: identify a delay-event / EPC-rich cohort
     (12-case has zero such events). Options: (a) trace the 800-case
     AEC challenge dataset for cases with `delay_first` / `delay_shift`
     fires; (b) construct synthetic controlled-delay cohort. Trace
     per-frame `H_error_per_bin` trajectory post-HEPC vs an AEC3-style
     reference behaviour or simulated transient leakage profile.
     NOT a benchmark run — a targeted trace audit.
   - Per user direction (rev 4): **no code, no benchmark, no cohort
     run in this round**. This is the next step the user must authorise.

After rev 4: Q2 / Q3 / Q4 closed or deferred-to-Q1. Q1 remains the
single OPEN v3.21.x parity question. Unknown #4 gates only the
default-OFF `W.fill(0)` substrate fate. Unknown #5 gates the Q1
stance decision and is the natural next step after this design note.

---

## 8. Recommended next step (after design note user-review, rev 4)

**Q2, Q3 (fate-couples-to-Q1), and Q4 are closed at the doc level
(rev 4).** No code is involved. The single OPEN v3.21.x parity question
is Q1.

The recommended next step is **NOT implementation**. It is **a Q1 trace
audit on a delay-event / EPC-rich cohort** to determine whether stance
(b) [documented functional equivalence via elevated steady leakage] is
empirically close enough to AEC3's transient leakage profile, or
whether stance (a) [strict port: 2.5 s transient + 1 s smoothing] is
required to close the parity gap.

**Trace audit prerequisites** (the user must authorise these before any
code or benchmark run):
1. **Cohort identification** — 12-case has zero `delay_first` /
   `delay_shift` fires (Audit B §5.4). Need either:
   - (a) subset of the 800-case AEC challenge dataset that fires
     delay events (requires read-only trace pass over current
     production main on the 800-case, no code change);
   - (b) synthetic controlled-delay cohort with known delay shifts
     at known times.
2. **Trace fields to capture per frame** post-HEPC:
   `H_error_per_bin` mean / max / min, `_leakage_converged` /
   `_leakage_diverged` in effect, frames-since-last-HEPC counter,
   Kalman gain magnitude trajectory, output ERLE.
3. **Reference for comparison**:
   - Mathematical simulation of AEC3 transient leakage trajectory
     (5e-3 → 5e-5 per block linear over 1 s, after 2.5 s transient
     hold), no code in our pipeline.
   - Optional: real AEC3 reference render if available.

After the trace audit, Q1 stance (a) / (b) / (c) can be answered with
empirical evidence. **Only then** would an implementation proposal
(stance (a)) or a documentation-only closure (stance (b) or (c)) be
authored.

**This note still does NOT propose code, benchmark, or cohort run.**
The above trace audit prerequisites are the next step the user must
explicitly authorise — they are **not** part of this rev.

---

## 9. What this design note does NOT do

Per user direction:
- Does not implement.
- Does not benchmark.
- Does not propose a patch sequence.
- Does not pick winners between options.
- Does not estimate AECMOS impact (no rendering done).
- Does not authorise any cohort rendering.

If the user wants to move toward implementation:
- **Q2 / Q3 / Q4** are closed or deferred (no implementation
  authorisation needed; doc-level closure already recorded).
- **Q1** is the single remaining open decision. The next discrete
  step is **not** implementation but the **delay-event cohort trace
  audit** described in §8 + Unknown #5 of §7 — identify a
  delay-event / EPC-rich cohort, trace `H_error_per_bin` trajectory
  post-HEPC, compare to a reference AEC3 transient leakage profile,
  then decide between Q1 stance (a) / (b) / (c). Per user direction:
  no code, no benchmark, no cohort run in this round. None of that
  happens in this note.
