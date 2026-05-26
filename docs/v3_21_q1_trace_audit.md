# v3.21 Q1 trace audit — PBFDKF elevated steady leakage vs AEC3 transient leakage (2026-05-24)

**Status**: read-only paper + trace audit. No algorithm code change, no
benchmark, no cohort render. Per user direction (2026-05-24) under the
rev 4 design-note framing
([docs/v3_21_delay_change_chain_design_note.md](v3_21_delay_change_chain_design_note.md)).

**Question**: Is PBFDKF's elevated steady-state leakage in
`python/modules/aec3_scale.py` / `python/modules/filters.py` close
enough to AEC3's transient `refined_initial.leakage_*` profile to
**close Q1 as documented functional equivalence (stance (b))**, or do
we need a strict AEC3-style transient → steady leakage port
(stance (a))?

**Scope guard**:
- Does NOT use `_p_max_override` as a Q1 adapter (it operates on the
  demoted legacy P matrix per `filters.py:411-414`).
- Does NOT touch Phase D, Phase G, M2/M3 W=0, usable_linear,
  PathChangeRegimeHandler, RES tuning.
- Does NOT re-open Q2 / Q3 / Q4 (rev 4 closures stand).
- Does NOT propose code, version bump, or 800-case.

---

## 0. TL;DR

**Verdict: C — insufficient evidence in current cohort to close Q1
either way.** Tentative lean: stance (b) [documented functional
equivalence] looks defensible *for the trigger sites we actually
observe in the existing trace cohort* (EPV / shadow_rise under Phase D
wiring) but **CANNOT be confirmed without a real-delay-event cohort
trace**, which the 12-case stress cohort does not contain.

What we learned from this round:
1. **AEC3 leakage math is fully characterised** (§1). `H_error` is
   rebuilt per block via `H_error += leakage × erl`; AEC3's transient
   profile is 100× steady on `leakage_converged` (5e-3 → 5e-5 per
   block) and 10× steady on `leakage_diverged` (5e-1 → 5e-2 per block).
2. **PBFDKF leakage math is fully characterised** (§2). Always-on
   elevated steady values: `1e-3` per-block-equivalent `leakage_converged`
   (20× AEC3 steady, 5× lower than AEC3 transient) and `1e-1`
   per-block-equivalent `leakage_diverged` (2× AEC3 steady, 5× lower
   than AEC3 transient).
3. **Observed H_error trajectory post-HEPC** (§3, 3 cases) recovers
   from `H_init=10000` → clamped `100` (gate window) → working
   equilibrium (1–30) within 50–100 frames (0.5–1.0 s wall).
   Out_rms recovers within 0.5–1.5 s in all 3 cases. No evidence of
   pathological recovery from leakage choice.
4. **Per-bin synthetic math** (§4) confirms post-HEPC `H_error` MEAN is
   dominated by the **`H_ERROR_CEIL` choice** (ours `100` vs AEC3
   `2.0`) and the gate-window holding period, not by the leakage value.
   In active-far-end regime both AEC3 and PBFDKF settle to similar
   per-bin `mu` magnitudes via the `mu = 2/X²` upper bound.
5. **All 3 traced events are EPV / shadow_rise**, not real
   `delay_first` / `delay_shift` events. Under production Phase D
   wiring they ARE dispatched as `delay_change=True` so the H_error
   reset path fires — but the underlying physics (filter trained
   against a now-wrong delay alignment) is NOT exercised. AEC3's
   transient leakage is designed for the **delay-change physics**;
   our trace cannot test that.

**No code change is recommended in this round.** Q1 stays OPEN as a
v3.21.x parity question. The next step the user must authorise (if
they want to close Q1) is either:
- a real-delay-event cohort identification pass (read-only trace over
  the 800-case AEC challenge dataset to find cases that fire
  `delay_first` / `delay_shift` under production main), **or**
- a synthetic delay-shift cohort construction (controlled
  delay-jumps in known input audio), **or**
- accepting stance (b) on the basis of §3 + §4 evidence and writing
  the equivalence-with-caveats note, leaving Q1 closed-by-documentation
  with no code.

The user should pick one of these three paths explicitly. **No path
requires implementation in this round.**

---

## 1. AEC3 reference: leakage → H_error trajectory

### 1.1 Per-block math (`refined_filter_update_gain.cc:70-141`)

```
Compute() each call (per block, after gate clears):
  μ[k] = H_error[k] / (0.5 × H_error[k] × X²[k] + n × E²_refined[k])
            if X²[k] >= noise_gate, else μ[k] = 0
  H_error[k] -= 0.5 × μ[k] × X²[k] × H_error[k]    # decay
  G[k] = μ[k] × E[k]                                # filter update gain

  # Always-on H_error refresh + clamp (runs every Compute even when
  # decay+update gates skip):
  if E²_refined[k] <= E²_coarse[k] or disallow_leakage_diverged:
      H_error[k] += current_config_.leakage_converged × erl[k]
  else:
      H_error[k] += current_config_.leakage_diverged × erl[k]
  H_error[k] = clamp(H_error[k], current_config_.error_floor,
                     current_config_.error_ceil)
```

`current_config_` is per-block-smoothed from `old_target_config_` →
`target_config_` over `config_change_duration_blocks_` blocks (§1.3).

Gates that skip the decay+G branch (`refined_filter_update_gain.cc:96-99`):
```
if ++poor_excitation_counter_ < size_partitions
   or saturated_capture_signal
   or call_counter_ <= size_partitions:
    G.re/im.fill(0)   # no filter update this call
    # falls through to leakage refresh + clamp
```

So leakage refresh runs every block; decay+update is skipped during the
gate window.

### 1.2 Reset on `delay_change` (`refined_filter_update_gain.cc:53-68`)

```
HandleEchoPathChange(echo_path_variability):
  if echo_path_variability.gain_change:
    # TODO(bugs.webrtc.org/9526) Handle gain changes.
    pass
  if echo_path_variability.delay_change != kNone:
    H_error_.fill(kHErrorInitial)   # 10000.f
  if not echo_path_variability.gain_change:
    poor_excitation_counter_ = kPoorExcitationCounterInitial  # 1000
    call_counter_ = 0
```

**delay_change is the ONLY trigger for H_error reset.** gain_change
fires nothing on the refined gain side. (Compare: our PBFDKF dispatch
table in §2.4 — Phase D fires `delay_change=True` even on
EPV/shadow_rise, so H_error DOES reset on those sites in our pipeline.)

### 1.3 SetConfig switching (`refined_filter_update_gain.h:59-70`)

```
SetConfig(config, immediate_effect):
  if immediate_effect:
    old_target_config_ = current_config_ = target_config_ = config
    config_change_counter_ = 0       # no smoothing
  else:
    old_target_config_ = current_config_
    target_config_ = config
    config_change_counter_ = config_change_duration_blocks_   # 250 blocks

UpdateCurrentConfig() each Compute:
  if config_change_counter_ > 0:
    --config_change_counter_
    factor = config_change_counter_ / config_change_duration_blocks_
    current_config_.leakage_converged = lerp(old → target, 1-factor)
    # ... and leakage_diverged, error_floor, error_ceil, noise_gate
```

So during ExitInitialState's smoothed revert, leakage drops linearly
from transient → steady over 250 blocks = 1 s wall.

### 1.4 Dispatch in `subtractor.cc:148-186` (already covered in design note)

Summary:
- `delay_change` triggers `full_reset`:
  - `refined_gains_->HandleEchoPathChange()` → `H_error.fill(10000)` + counter reset
  - `refined_gains_->SetConfig(refined_initial, immediate=true)` → **switches leakage to TRANSIENT profile immediately**
  - (plus coarse-side equivalents and `SetSizePartitions(refined_initial.length_blocks, immediate=true)`)
- `ExitInitialState()` (triggered by `AecState::TransitionTriggered()` after
  `initial_state_seconds=2.5 s` of strong active render post-HEPC):
  - `refined_gains_->SetConfig(refined, immediate=false)` → 1-second
    smoothed revert leakage transient → steady.

### 1.5 Confirmed AEC3 default values

Source: [docs/aec3_extracts/api/audio/echo_canceller3_config.h:88-118](aec3_extracts/api/audio/echo_canceller3_config.h).

| Profile | `leakage_converged` | `leakage_diverged` | `error_floor` | `error_ceil` | `length_blocks` |
|---|---:|---:|---:|---:|---:|
| `refined` (steady) | **5e-5** | **5e-2** | 1e-3 | **2.0** | 13 |
| `refined_initial` (transient) | **5e-3** | **5e-1** | 1e-3 | **2.0** | 12 |
| Ratio transient/steady | **100×** | **10×** | 1× | 1× | 0.92× |

Coarse side: `coarse.rate=0.7`, `coarse_initial.rate=0.9` (1.29×).

Smoothing: `config_change_duration_blocks = 250` (1 s).
Transient bound: `initial_state_seconds = 2.5`.

---

## 2. PBFDKF reference: leakage → H_error trajectory

### 2.1 Per-hop math (`python/modules/filters.py:716-871`)

The PBFDKF `_update_weights` mirrors AEC3 `Compute`, with per-hop
units. Compute per hop (after gate clears):

```
mu[k]            = H_error_per_bin[k] / (0.5 × H_error_per_bin[k] × X²[k] + n × E²[k])
H_error_per_bin -= 0.5 × mu × X² × H_error_per_bin       # filters.py:817-819
_h_error_refresh()                                        # filters.py:824
  # inside _h_error_refresh (filters.py:826-871):
  use_converged = (E²_ref_sum <= E²_coa_sum) or _disallow_leakage_diverged
  leakage = _leakage_converged if use_converged else _leakage_diverged
  H_error_per_bin += leakage × _erl_per_bin
  np.clip(H_error_per_bin, _h_error_floor, _h_error_ceil, out=H_error_per_bin)
```

Gates that skip decay+update (`filters.py:543-549`):
```
self._call_counter += 1
if (call_counter <= n_partitions             # n_partitions = 6 by default
    or _poor_excitation_counter < n_partitions
    or _saturated_capture):
    if _use_aec3_h_error: _h_error_refresh()    # leakage + clamp still runs
    return                                       # no W update
```

### 2.2 Reset on `delay_change=True` (`filters.py:482-506`)

```
def handle_echo_path_change(self, delay_change=True, gain_change=False, zero_filter=False):
    super().handle_echo_path_change(delay_change=delay_change,
                                    gain_change=gain_change,
                                    zero_filter=zero_filter)
    # PBFDAF super: _call_counter=0 if not gain_change;
    # _poor_excitation_counter = POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT (400)
    if delay_change:
        self.H_error_per_bin.fill(np.float32(_aec3_scale.H_ERROR_INIT_FLOAT))  # 10000
```

**Same trigger semantic as AEC3**: only `delay_change=True` resets
H_error. `gain_change` alone does NOT reset H_error. (NB: under
production Phase D wiring, EPV / shadow_rise are dispatched as
`delay_change=True` rather than `gain_change=True`. See §2.4.)

### 2.3 Leakage values (`python/modules/aec3_scale.py:82-88`)

```
H_ERROR_INIT_FLOAT             = 10000.0       # matches AEC3
H_ERROR_FLOOR_FLOAT            = 1e-3          # matches AEC3
H_ERROR_CEIL_FLOAT             = 1e2 = 100.0   # 50× AEC3 default of 2.0 — OUT OF Q1 SCOPE
LEAKAGE_CONVERGED_PER_HOP_DEFAULT = per_block_rate_to_per_hop(1e-3, 160, 16000) = 2.5e-3 per hop
LEAKAGE_DIVERGED_PER_HOP_DEFAULT  = per_block_rate_to_per_hop(1e-1, 160, 16000) = 2.5e-1 per hop
```

`per_block_rate_to_per_hop` (line 53) multiplies by
`hop_seconds/block_seconds = 10/4 = 2.5`. So **per-block-equivalent**
values are:

| Quantity | PBFDKF per-block-equivalent | AEC3 steady | AEC3 transient |
|---|---:|---:|---:|
| `leakage_converged` | **1e-3** | 5e-5 | 5e-3 |
| Ratio to AEC3 steady | **20× higher** | 1× | 100× |
| Ratio to AEC3 transient | **5× lower** | 0.01× | 1× |
| `leakage_diverged` | **1e-1** | 5e-2 | 5e-1 |
| Ratio to AEC3 steady | **2× higher** | 1× | 10× |
| Ratio to AEC3 transient | **5× lower** | 0.1× | 1× |

PBFDKF leakage values are biased toward AEC3 transient (consistently
~5× below) and well above AEC3 steady (20× / 2×). They are
**always-on**: no two-state machine, no smoothing window.

### 2.4 Orchestrator dispatch — production config

Per `orchestrator.py:1875-1890, 1945-1959, 2548-2559, 2626-2659` +
`config.py:287, 302, 317, 331`:

| Trigger | Production default flags | `filter.handle_echo_path_change` arguments | H_error reset fires? |
|---|---|---|---|
| `delay_first` | `use_aec3_handle_echo_path_change=True` + `use_aec3_epc_classification=False` | `delay_change=True, gain_change=False` | **YES** |
| `delay_shift` | same | `delay_change=True, gain_change=False` | **YES** |
| `EPV` (echo-path-variability detector) | same | `delay_change=True, gain_change=False` (`_delay_change = not _epc_cls`) | **YES** (under Phase D wiring) |
| `shadow_rise` | same | `delay_change=True, gain_change=False` (same) | **YES** (under Phase D wiring) |

So under production main, ALL four trigger sites fire the H_error
reset path. This matters for §3 — our trace cohort contains EPV /
shadow_rise events that DO exercise the H_error reset → leakage
recovery path, but do NOT exercise the underlying delay-change physics
that AEC3's transient leakage profile was designed for.

---

## 3. Per-case trace observations

### 3.1 Cohort identification

**Available existing trace data** (read-only, from a previous Phase G
isolation render — not re-rendered for this audit):

| Path | Cases | Variants |
|---|---|---|
| [out_v3_21_20_phase_g_trace/](../out_v3_21_20_phase_g_trace/) | nVUnxqHLr (DT_static), jtYTdZm (DT_static), wVYSGVTTakih (DT_movement) | `A_off` (production wiring, control) + `G` (Phase G isolation) |

Per-frame diag fields captured in `*_A_off_diag.json`:
- `frame`, `t_s`, `mic_rms`, `out_rms`
- `epv_event_raw`, `epv_event_suppressed`, `epc_active`, `epc_active_now`
- `filter_w_norm`, `shadow_w_norm`
- `erle_reset_signal`
- **`H_error_mean`** (the scalar mean over 257 bins per frame — the
  primary Q1 trace field)
- `call_counter_refined`, `poor_exc_counter_refined`

**Per-bin H_error is NOT in the existing trace.** Adding it would
require either a re-render with an extended trace script or post-hoc
recomputation — both deferred per the no-render rule.

### 3.2 EPC event inventory in available cohort

| Case | Frames | `epv_fires` | `shadow_rise_fires_est` | `h_error_resets` | Real `delay_first` / `delay_shift` |
|---|---:|---:|---:|---:|---:|
| nVUnxqHLr (DT_static) | 4298 | 0 | 2 | 2 | **0** |
| jtYTdZm (DT_static) | 3676 | 1 | 2 | 3 | **0** |
| wVYS_mvmt (DT_movement) | 3666 | 0 | 3 | 3 | **0** |
| Total | — | 1 | 7 | 8 | **0** |

**No real delay-change events in any of the 3 cases.** All 8 H_error
resets fire via EPV / shadow_rise classified as `delay_change=True`
under Phase D production wiring.

### 3.3 H_error post-HEPC trajectory (production code, A_off variant)

Trajectory at relative frames after each reset (frame 0 = reset frame).
Each value is `H_error_mean` (scalar mean of per-bin H_error over 257
bins). Hop = 10 ms wall.

| Case | Event | +0 (reset) | +1 | +5 | +10 | +20 | +50 | +100 | +250 | +500 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| nVUnxqHLr | 1 (t=11.12s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 1.7 | 0.29 | 0.20 | 0.17 | 0.35 | (next reset) |
| nVUnxqHLr | 2 (t=15.67s) | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 | 1.0e+04 |
| jtYTdZm | 1 (t=2.89s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 78 | 4.6 | 5.1 | 4.5 | 3.2 | 12 |
| jtYTdZm | 2 (t=5.40s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 61 | 23 | 21 | 18 | 12 | 6.6 |
| jtYTdZm | 3 (t=31.15s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 88 | 65 | 20 | 19 | 12 | 7.6 |
| wVYS_mvmt | 1 (t=2.61s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 46 | 21 | 13 | 12 | 2.0 | 32 |
| wVYS_mvmt | 2 (t=6.42s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | 90 | 59 | 39 | 32 | 23 | 7.5 |
| wVYS_mvmt | 3 (t=36.57s) | 1.0e+04 | 1.0e+02 | 1.0e+02 | — | — | — | — | — | — |

**Pattern**:
- `+0`: H_error = 10000 (just set by `handle_echo_path_change`).
- `+1` to `+5`: H_error = 100 (= `H_ERROR_CEIL`) — `_call_counter ≤ 6`
  → W update gate fires; only `_h_error_refresh` runs; clamp drops
  10000 → 100 instantly.
- `+10`: gate released (`call_counter` past `n_partitions=6`); decay
  begins; H_error drops below 100.
- `+10` to `+50`: rapid decay to working equilibrium.
- `+50` to `+500`: oscillates around case-specific equilibrium (0.17 –
  35) depending on signal dynamics.

**nVUnxqHLr Event 2 is an outlier**: H stays at 10000 indefinitely
post-reset. Inspection of `call_counter_refined` shows it stays at 0
for hundreds of frames — meaning the filter is NOT being called at
all. This is the **PathChangeRegimeHandler's `main_paused` mode**
firing (the legacy P-based regime classifier deciding to freeze the
main filter on the cohort tail); it is NOT a leakage / H_error
mechanism and is **out of Q1 scope** (PathChangeRegimeHandler is its
own arc, design note rev 4 §2.3 closing note). Excluded from §3.4
recovery-time summary.

### 3.4 Output recovery (out_rms) ±2.5 s around reset

| Case | Event | mic pre (50 frames) | mic post[0,0.5)s | mic [0.5,1.5)s | mic [1.5,2.5)s | out pre | out post[0,0.5)s | out [0.5,1.5)s | out [1.5,2.5)s |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nVUnxqHLr | 1 | 8.5e-3 | 2.2e-2 | 3.2e-2 | 2.0e-2 | 4.2e-3 | 1.1e-3 | 1.2e-3 | 2.9e-3 |
| jtYTdZm | 1 | 1.1e-1 | 9.2e-2 | 6.1e-2 | 4.9e-2 | 1.7e-3 | 4.8e-3 | 3.3e-3 | 2.5e-3 |
| jtYTdZm | 2 | 5.9e-2 | 7.0e-2 | 3.3e-2 | 5.1e-2 | 2.5e-3 | 1.8e-2 | 1.4e-2 | 6.2e-3 |
| wVYS_mvmt | 1 | 1.4e-1 | 1.5e-1 | 7.0e-2 | 8.9e-2 | 7.0e-3 | 2.1e-2 | 1.3e-2 | 8.0e-3 |
| wVYS_mvmt | 2 | 2.3e-1 | 1.5e-1 | 1.6e-2 | 5.2e-2 | 1.0e-2 | 1.8e-2 | 6.5e-3 | 1.1e-2 |

**Pattern**:
- Output `out_rms` ratio (out / mic) recovers within 0.5–1.5 s in all
  cases. nVUnxqHLr Event 1 actually *improves* immediately (filter
  re-converges faster than the pre-reset state). The other events show
  a ~1.5× – 3× rise in `out_rms` during the first 0.5 s, settling
  back to within 2× of pre by 1.5–2.5 s.
- No evidence of pathological recovery in the available 5 events.

### 3.5 H_error ceiling / floor hit fraction post-event

For each event's first 2.5 s post-reset (250 frames):

| Case | Event | Ceil hits (H=100) | Floor hits (H=1e-3) | Comment |
|---|:---:|---:|---:|---|
| nVUnxqHLr | 1 | 6 / 249 (2.4%) | 0 | Only first 6 frames (gate window) |
| jtYTdZm | 1 | 6 / 249 (2.4%) | 0 | Only gate window |
| jtYTdZm | 2 | 6 / 249 (2.4%) | 0 | Only gate window |
| wVYS_mvmt | 1 | 6 / 249 (2.4%) | 0 | Only gate window |
| wVYS_mvmt | 2 | 7 / 249 (2.8%) | 0 | Gate window |

**MEAN H_error never sits at floor (1e-3) in any of these traces** —
indicating that across all 257 bins, at least some bins always have
substantial H_error contribution (the quiet bins reach ceiling = 100
under leakage growth with no decay).

**The ceiling clamp is hit ONLY during the 6-7-frame gate window**
right after reset. Post-gate, the mean drops below ceiling within
~3-4 frames.

---

## 4. Synthetic math: AEC3 vs PBFDKF per-bin H_error trajectory

A standalone per-bin simulation comparing AEC3 transient leakage, AEC3
steady leakage, and PBFDKF elevated steady leakage under three regimes.
Computed in Python; no rendering involved. Inputs: `n_partitions = 13`
for AEC3, `6` for PBFDKF; `erl_per_bin = 0.1`; `H_init = 10000`;
clamps as documented. Gate skips `decay+G` for first `n_partitions`
calls but still runs leakage refresh + clamp.

### 4.1 Scenario A — sustained active far-end (X² = 1e-2, E² = 1e-3)

The "normal" post-HEPC condition: far-end resumes immediately after
reset and stays active.

| Block | AEC3 transient | AEC3 steady | PBFDKF |
|---:|---:|---:|---:|
| 0 | 2.000 | 2.000 | 100.000 |
| 1 | 2.000 | 2.000 | 100.000 |
| 5 | 2.000 | 2.000 | 100.000 |
| 10 | 2.000 | 2.000 | 0.240 |
| 13 | 1.131 | 1.130 | 0.150 |
| 20 | 0.281 | 0.280 | 0.080 |
| 50 | 0.073 | 0.066 | 0.028 |
| 100 | 0.043 | 0.029 | 0.016 |
| 249 | 0.036 | 0.011 | 0.011 |
| **Mean 0-249** | **0.170** | **0.155** | **2.433** |

**Observation**: PBFDKF mean is **14× higher** than AEC3 transient
mean, dominated entirely by the gate-window holding period (first 6
frames at ceiling = 100, vs AEC3's 13 blocks at ceiling = 2.0).
Post-gate equilibrium is actually **lower** for PBFDKF than AEC3
transient (mu-balance with `H_ERROR_CEIL` and `n_partitions=6` puts
us in a slightly different per-block decay regime).

**Implication for Q1**: in the normal active-far-end regime, the
**leakage value choice has minimal effect on the post-gate
equilibrium**. The mean-H gap is driven by the `H_ERROR_CEIL` choice
(100 vs 2.0), which is OUT OF Q1 SCOPE (it is a separate clamp-axis
divergence noted in design note rev 4 §3.2).

### 4.2 Scenario B — quiet far-end for 100 blocks then active

Tests leakage's role on quiet bins (no decay → only leakage growth +
clamp determines trajectory).

| Block | AEC3 transient | AEC3 steady | PBFDKF |
|---:|---:|---:|---:|
| 0 | 2.000 | 2.000 | 100.000 |
| 50 (quiet) | 2.000 | 2.000 | 100.000 |
| 99 (last quiet) | 2.000 | 2.000 | 100.000 |
| 100 (first active) | 1.131 | 1.130 | 1.186 |
| 110 | 0.214 | 0.211 | 0.109 |
| 150 | 0.058 | 0.050 | 0.025 |
| 200 | 0.041 | 0.026 | 0.015 |
| 249 | 0.037 | 0.017 | 0.013 |
| **Mean 0-249** | **0.853** | **0.846** | **40.028** |

**Observation**: during the 100 quiet blocks, all three configs sit at
ceiling — but PBFDKF ceiling is 100 while AEC3 is 2.0. The 50× ceiling
gap drives the 47× mean gap. Post-quiet, decay reasserts and all three
converge to similar equilibrium magnitudes within 10-20 blocks.

**Implication**: PBFDKF's high `H_ERROR_CEIL` is the dominant factor
in the mean during silent-far-end periods. The leakage value sets the
RATE at which H_error climbs from floor to ceiling — but our elevated
steady leakage (1e-3 per block) is fast enough to keep H_error at
ceiling during typical quiet periods. AEC3 steady (5e-5 per block)
would take ~40 000 blocks to reach AEC3 ceiling (2.0) from floor —
i.e. AEC3 steady essentially keeps H_error at floor during sustained
quiet, while AEC3 transient (5e-3) reaches AEC3 ceiling in ~400 blocks
(1.6 s).

### 4.3 Scenario C — divergence regime (E² ≈ X² mismatch)

Tests `leakage_diverged` path: refined error is large (filter not
tracking).

| Block | AEC3 transient | AEC3 steady | PBFDKF |
|---:|---:|---:|---:|
| 0 | 2.000 | 2.000 | 100.000 |
| 10 | 2.000 | 2.000 | 2.366 |
| 13 | 1.907 | 1.862 | 1.510 |
| 20 | 1.515 | 1.266 | 0.849 |
| 50 | 1.188 | 0.593 | 0.407 |
| 100 | 1.166 | 0.409 | 0.354 |
| 249 | 1.165 | 0.364 | 0.351 |
| **Mean 0-249** | **1.240** | **0.568** | **2.900** |

**Observation**: in divergence regime, the differences in equilibrium
become more apparent because leakage_diverged dominates the rebuild
rate. AEC3 transient (5e-1 per block) keeps H_error elevated near
ceiling (~1.17). AEC3 steady (5e-2) settles to ~0.36. **PBFDKF settles
to ~0.35 — virtually identical to AEC3 steady, not transient.**

**Implication for Q1**: in the divergence regime where the leakage
profile choice matters most (it determines how aggressively the filter
can re-converge from a bad state), **PBFDKF's elevated steady
leakage_diverged (2× AEC3 steady) behaves much more like AEC3 steady
than AEC3 transient**. This is the strongest mathematical case against
"PBFDKF elevated steady ≈ AEC3 transient functional equivalence" — at
least on `leakage_diverged`, we are NOT close to AEC3 transient.

### 4.4 Summary of synthetic math

| Regime | PBFDKF mean | AEC3 transient mean | AEC3 steady mean | PBFDKF closer to … |
|---|---:|---:|---:|---|
| Active far-end (A) | 2.43 | 0.17 | 0.16 | Neither (dominated by ceiling) |
| Quiet far-end → active (B) | 40.0 | 0.85 | 0.85 | Neither (dominated by ceiling) |
| Divergence (C, post-gate equilibrium) | 0.35 | 1.17 | 0.36 | **AEC3 STEADY** |

**Key finding**: per-bin post-gate equilibrium analysis suggests
PBFDKF is functionally closer to **AEC3 steady** than to AEC3
transient, NOT the other way around. The "20× AEC3 steady" leakage
value (which is what design note rev 4 cites as evidence for being
"biased toward AEC3 transient") **does not translate into a transient-
like post-gate H_error trajectory** because the per-block mu-balance
equilibrium is the dominant determinant, not the absolute leakage
value.

**The mean-H magnitude difference (PBFDKF vs AEC3) is dominated by the
`H_ERROR_CEIL` choice (100 vs 2.0) AND the gate-window length**, both
of which are out of Q1 scope.

---

## 5. Verdict and recommendation

### 5.1 Why not (a) [strict port]

No evidence in §3 or §4 that the strict AEC3 transient leakage profile
would improve our pipeline. Output recovery in the 3 traced cases is
fast (out_rms recovers within 0.5–1.5 s post-EPC). The
post-gate H_error decay+equilibrium in our pipeline is governed by
mu-balance with X² / E², not by the leakage profile.

Implementing stance (a) would require:
- ~50 lines of code (refined_initial vs refined config, SetConfig
  method, smoothing-window state, orchestrator HEPC + TransitionTriggered
  hooks).
- Adding `AecState.TransitionTriggered` accessor (Q3, currently
  DEFERRED).
- Re-running 800-case + stress cohort to validate no regression on
  existing production behaviour.

**Not justified by current evidence.** Deferred until a real-delay-event
cohort can show stance (b) is insufficient.

### 5.2 Why not (b) [documented functional equivalence — close now]

Two reasons:

1. **§4 synthetic math contradicts the design-note rev 4 framing that
   PBFDKF leakage is "biased toward AEC3 transient"**. In the
   divergence regime (where leakage profile matters most for
   re-convergence dynamics), PBFDKF behaves like AEC3 STEADY, not
   transient. The "20× AEC3 steady" magnitude on `leakage_converged`
   does not survive the per-block mu-balance — equilibrium ends up at
   the steady-like value because decay dominates the rebuild rate.

2. **No real-delay-event trace data in the cohort**. All 8 H_error
   reset events traced are EPV / shadow_rise events. Real
   delay_first / delay_shift would have additional physics (filter
   trained against now-wrong delay alignment) that the available
   cohort cannot test.

**Closing Q1 by documentation now would lock in an inaccurate
"transient-like" framing.** A more accurate framing — "PBFDKF runs
steady-like leakage with higher absolute values that primarily affect
ceiling-saturation dynamics on quiet bins" — needs the real-delay-event
trace to confirm it doesn't break under the use case AEC3's transient
leakage targets.

### 5.3 Why (c) [insufficient evidence — next step is cohort identification]

**This is the honest verdict.**

The available trace (3 cases, 8 reset events, all EPV/shadow_rise)
shows no obvious functional deficiency from the elevated-steady
leakage choice — but it ALSO doesn't validate the equivalence claim
on the actual delay-change use case AEC3's transient leakage is
designed for.

### 5.4 Recommended next step (the user must authorise)

Pick one of three paths. **No path is implementation, all three are
read-only / analysis tasks:**

1. **Real-delay-event cohort identification pass** *(recommended for
   completeness)*. Read-only trace over the 800-case AEC challenge
   dataset using current production main + the same per-frame
   `H_error_mean` / event-count diag (no rendering — the existing
   per-case `*_diag.json` from prior bench runs should already
   capture event counts via `epc_active` flag transitions). Identify
   subset of cases that fire real `delay_first` / `delay_shift`.
   If any exist, re-run the trace-only render on the identified subset
   with the existing per-frame trace script (no code change to the
   algorithm). If none exist, conclude that the production cohort
   genuinely doesn't exercise delay_change physics — making Q1
   moot in practice, with stance (b) defensible by "no observable
   surface".

2. **Synthetic delay-shift cohort construction**. Build a 3–6-case
   controlled cohort: take an existing clean DT case, insert known
   `lpb`-channel delay shifts at known times (e.g. +200 sample shift
   at t=10s, +500 sample shift at t=20s), render with current
   production main, trace H_error trajectory post-shift. This
   directly tests the delay-change physics. Construction is mechanical
   (numpy roll on the lpb wav) and pure data prep, no algorithm code
   change. Verifies whether elevated steady leakage handles real delay
   shifts adequately.

3. **Close Q1 as stance (b) with corrected framing**. Accept that the
   evidence here is sufficient to close, but use the framing from §4:
   "PBFDKF runs always-on leakage closer to AEC3 steady than AEC3
   transient in terms of post-gate equilibrium. AEC3-equivalent behaviour
   on delay-change-driven scenarios is presumed adequate from the
   trace evidence on EPV/shadow_rise reset cases, with the caveat that
   we have no direct delay-change trace." This is a documentation
   patch only (update design note rev 4 §0 + §2.3 + §4 to reflect the
   §4 math finding), no code, no benchmark.

### 5.5 Disposition

| Choice | What ships | Risk to production | What gets documented |
|---|---|---|---|
| Path 1 above | Trace audit appendix listing delay-event cases (if any) | None (read-only) | Append §6 to this doc listing identified cases + per-case findings |
| Path 2 above | New synthetic cohort + audit appendix | None (synthetic data prep + read-only trace) | Append §6 with synthetic trace results |
| Path 3 above | Design note rev 5 with corrected framing | None (doc only) | Update design note + close Q1 in MEMORY |

**Default recommendation: Path 1 first** (cheapest, may make Q1 moot
without further work).

---

## 6. What this audit does NOT do

- Does NOT touch algorithm code.
- Does NOT run benchmarks.
- Does NOT render new cohorts.
- Does NOT propose a patch sequence.
- Does NOT bump version.
- Does NOT re-open Q2 / Q3 / Q4 (rev 4 closures stand).
- Does NOT use `_p_max_override` as a Q1 adapter.
- Does NOT recommend stance (a) [strict port] — current evidence does
  not support it.
- Does NOT recommend stance (b) [close now as functional equivalence]
  — current cohort doesn't include real delay events.
- Recommends stance (c) [verdict insufficient evidence] + one of three
  next-step paths the user must authorise.
