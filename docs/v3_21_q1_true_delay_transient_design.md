# v3.21 Q1 True-Delay Transient Leakage — Design Note

**Date:** 2026-05-25  
**Status:** DESIGN ONLY — no code change yet  
**Depends on:** Path C mini-bench ([v3_21_q1_real_delay_minibench.md](v3_21_q1_real_delay_minibench.md))  
**Config flag:** `use_q1_true_delay_transient_leakage` (default OFF)

---

## §0 TL;DR

Path C confirmed that real `delay_first` / `delay_shift` events fire in 8/9 production
cases, and that the current production code does **not** apply any H_error response to
these events (only EPV/shadow_rise get the 10000 reset). AEC3 applies
`SetConfig(refined_initial, immediate=true)` uniformly to all four EPC trigger classes,
giving a 2.5 s transient leakage burst after any delay event.

This design adds a minimal default-OFF response: on `delay_first` / `delay_shift` only,
temporarily elevate `_leakage_converged` / `_leakage_diverged` to AEC3 transient values
for 2.5 s (250 hops), then smoothly return to PBFDKF elevated steady over 1 s (100
hops). No H_error hard reset. EPV / shadow_rise wiring unchanged.

---

## §1 Event sources in orchestrator.py

Both trigger sites are in `python/modules/orchestrator.py` inside the online delay
estimation block (processed every hop when `delay_est is not None`):

### §1.1 delay_first — Path A (first delay acquisition)

```
orchestrator.py:1854-1889
```

```python
# Path A: first delay acquisition
if (_delay_eligible
        and self._current_delay < 0
        and self.delay_est.is_solid):
    self._current_delay = new_delay
    self._reset_filter_derived_state(reason='delay_first',
                                     preserve_render_ema=True)
    self._maybe_mark_diverged('delay_first')          # ← arm point A
    if bool(getattr(self.config, 'use_aec3_epc_classification', False)):
        ...  # handle_echo_path_change (not active in production)
```

Arm point A = line **1868**, immediately after `_maybe_mark_diverged('delay_first')`.  
Guard: Path A fires once per case (when `_current_delay` transitions from −1 → ≥0).

### §1.2 delay_shift — Path B (mid-case delay change)

```
orchestrator.py:1906-1959
```

```python
# Path B: delay shift
if (_delay_eligible
        and self._current_delay >= 0
        and self.delay_est.confidence >= 0.5
        and abs(new_delay - self._current_delay) > 32):
    if (hasattr(self, '_pending_delay')
            and abs(new_delay - self._pending_delay) < 16):
        self._current_delay = new_delay
        ...
        self._reset_filter_derived_state(reason='delay_shift', ...)
        ...
        self._maybe_mark_diverged('delay_shift')      # ← arm point B
        ...
```

Arm point B = line **1935**, immediately after `_maybe_mark_diverged('delay_shift')`.  
Guard: Path B fires only when two consecutive consistent estimates agree to a shift
> 32 samples with confidence ≥ 0.5. wlAXM0iDgkm fired 3 in Path C.

### §1.3 What is NOT triggered

EPV and shadow_rise dispatch sites **do not arm** the transient counter — their
H_error reset (10000 → clipped to 100) continues unchanged via existing Phase D wiring.

---

## §2 AEC3 leakage mapping

AEC3 source operates on 64-sample 4 ms blocks. Our hop = 160 samples at 16 kHz = 10 ms.  
Conversion: `per_hop = per_block × (10 ms / 4 ms) = per_block × 2.5`

(All values below are per-hop for our hop=160 at sr=16000.)

| Profile | AEC3 per-block | Our per-hop (×2.5) |
|---|---|---|
| AEC3 transient `refined_initial` leakage_converged | 5e-3 | **1.25e-2** |
| AEC3 transient `refined_initial` leakage_diverged | 5e-1 | **1.25** |
| AEC3 steady `refined` leakage_converged | 5e-5 | 1.25e-4 |
| AEC3 steady `refined` leakage_diverged | 5e-2 | 1.25e-1 |
| Our elevated steady (current production) leakage_converged | — | **2.5e-3** (20× AEC3 steady) |
| Our elevated steady (current production) leakage_diverged | — | **2.5e-1** (2× AEC3 steady) |

**Ratios for the transient mode vs our elevated steady:**
- leakage_converged: 1.25e-2 / 2.5e-3 = **5×** (matches AEC3 transient / PBFDKF elevated ratio)
- leakage_diverged: 1.25 / 2.5e-1 = **5×** (same ratio)

So the transient bump in our system is 5× above our elevated steady, identical to the ratio between AEC3 transient and PBFDKF elevated steady.

**Two new constants to add to `aec3_scale.py`:**
```python
# AEC3 transient-profile leakage rates (refined_initial config), per 10ms hop
LEAKAGE_CONVERGED_TRANSIENT_PER_HOP = per_block_rate_to_per_hop(5e-3, 160, 16000)  # 1.25e-2
LEAKAGE_DIVERGED_TRANSIENT_PER_HOP  = per_block_rate_to_per_hop(5e-1, 160, 16000)  # 1.25
```

For non-default hop sizes, the orchestrator should compute these at runtime via
`per_block_rate_to_per_hop(5e-3, config.hop_size, config.sample_rate)`.

---

## §3 Mechanism — orchestrator-driven leakage override

The filter already stores `_leakage_converged` and `_leakage_diverged` as mutable
`np.float32` scalars (set once at `PBFDKF.__init__`; read every hop by
`_h_error_refresh`). The orchestrator already writes other per-hop filter attributes
at the same code location (lines 2235–2281: `_e2_coarse_for_refresh`, `_erl_per_bin`,
`_disallow_leakage_diverged`).

**Adding to the existing per-hop filter-attribute block:**

```python
# --- Q1 true-delay transient leakage override (after _disallow_leakage_diverged write)
if (getattr(self.config, 'use_q1_true_delay_transient_leakage', False)
        and self.filter is not None
        and isinstance(self.filter, PBFDKF)):
    _rem = getattr(self, '_q1_tdt_rem', 0)
    if _rem > 0:
        self._q1_tdt_rem = _rem - 1
        # Transient values (AEC3 refined_initial profile)
        _tc = _aec3_scale.per_block_rate_to_per_hop(
            5e-3, self.config.hop_size, self.config.sample_rate)
        _td = _aec3_scale.per_block_rate_to_per_hop(
            5e-1, self.config.hop_size, self.config.sample_rate)
        # Our elevated steady baseline — cached from filter init to avoid config drift
        _sc = self._q1_tdt_lc_steady
        _sd = self._q1_tdt_ld_steady
        _smooth = getattr(self.config, 'q1_tdt_smoothing_hops', 100)
        if _rem <= _smooth:
            # Linear interpolation: transient → elevated steady
            # alpha 1.0 at start of smoothing window, 0.0 at end
            alpha = _rem / float(_smooth)
            _lc = alpha * _tc + (1.0 - alpha) * _sc
            _ld = alpha * _td + (1.0 - alpha) * _sd
        else:
            _lc, _ld = _tc, _td
        self.filter._leakage_converged = np.float32(_lc)
        self.filter._leakage_diverged = np.float32(_ld)
    else:
        # Restore to cached filter init values (NOT AEC3 true-steady)
        self.filter._leakage_converged = np.float32(self._q1_tdt_lc_steady)
        self.filter._leakage_diverged = np.float32(self._q1_tdt_ld_steady)
```

The arm code at each event site (lines 1868 and 1935):
```python
if getattr(self.config, 'use_q1_true_delay_transient_leakage', False):
    _t = getattr(self.config, 'q1_tdt_transient_hops', 250)
    _s = getattr(self.config, 'q1_tdt_smoothing_hops', 100)
    self._q1_tdt_rem = _t + _s   # full transient then smooth; total 350 hops
```

**Counter and cache initialisation (in `__init__`, after filter creation):**
```python
self._q1_tdt_rem = 0
self._q1_tdt_lc_steady = float(self.filter._leakage_converged) if self.filter else 2.5e-3
self._q1_tdt_ld_steady = float(self.filter._leakage_diverged) if self.filter else 2.5e-1
```
Cache the filter's actual init values so the smooth-back target is immune to
hop-size or config drift vs the module-level defaults.

**No changes to `filters.py`**. The filter's `_h_error_refresh` remains unchanged —
it reads `self._leakage_converged` / `self._leakage_diverged` which the orchestrator
now manages dynamically when the flag is ON.

**Flag-OFF byte-equality:** When `use_q1_true_delay_transient_leakage=False` (default),
no counter is armed, no leakage writes happen, filter's leakage stays at its init
values → production output is bit-identical.

---

## §4 Duration and smoothing

### §4.1 AEC3 reference duration

- AEC3 `initial_state_seconds = 2.5` → transient window = 2.5 s
- AEC3 counts only **active render blocks** (`strong_not_saturated_render_blocks_`).
  Silent frames do NOT count against the 2.5 s budget.
- AEC3 `config_change_duration_blocks = 250` AEC3-blocks = 1 s of smoothing.

### §4.2 Our implementation choices

| Parameter | AEC3 exact | Our approximation | Notes |
|---|---|---|---|
| Transient window | 2.5 s active render | `q1_tdt_transient_hops = 250` wall-clock hops | Conservative: counts all hops including silent → transient ends ≤ 2.5 s. Simpler. |
| Smoothing window | 1 s (100 × 10 ms hops) | `q1_tdt_smoothing_hops = 100` | Matches AEC3 duration at our hop scale. |
| **Total counter at arm** | 3.5 s combined | `_q1_tdt_rem = 350` (= 250 + 100) | Full transient runs while `rem > smoothing_hops`; smoothing gate: `rem ≤ smoothing_hops` (100). |
| Smoothing shape | AEC3 uses `config_change_duration_blocks` exponential | Linear alpha `= rem / smoothing_hops` | alpha=1.0 at start of smoothing (full transient values); alpha→0.0 at end (steady values). |

**Active-render gating deviation:** Counting all hops (not just active-render hops) is a
known departure from AEC3 semantics. On movement cases with active far-end throughout,
this doesn't matter. On quiet far-end cases, the transient ends sooner than AEC3 would.
This is conservative (shorter transient = smaller perturbation).

### §4.3 Re-arm on repeat events

If `delay_shift` fires while a transient window is still running (counter > 0), the counter
resets to 250. This is consistent with AEC3 semantics: each delay event restarts the
`initial_state_seconds` countdown. The wlAXM0iDgkm case (3 delay_shift events at t=7, 14,
18 s) would trigger 3 separate 2.5 s windows — each separated by > 2.5 s, so they don't
overlap.

---

## §5 Implementation plan

### §5.1 Files and changes

**`python/modules/aec3_scale.py`** (~2 lines):
```python
# AEC3 transient-profile (refined_initial) leakage rates, per 10ms hop
LEAKAGE_CONVERGED_TRANSIENT_PER_HOP = per_block_rate_to_per_hop(5e-3, 160, 16000)  # 1.25e-2
LEAKAGE_DIVERGED_TRANSIENT_PER_HOP  = per_block_rate_to_per_hop(5e-1, 160, 16000)  # 1.25
```

**`python/modules/config.py`** (~3 lines in dataclass body):
```python
use_q1_true_delay_transient_leakage: bool = False  # Q1 parity: transient leakage on delay_first/shift
q1_tdt_transient_hops: int = 250   # 2.5 s at hop=160/sr=16000; AEC3 initial_state_seconds
q1_tdt_smoothing_hops: int = 100   # 1 s smooth back to elevated steady; AEC3 config_change_duration
```

**`python/modules/orchestrator.py`** (~20 lines in 3 locations):
1. `__init__` (after filter creation): `self._q1_tdt_rem = 0` + cache `_q1_tdt_lc_steady` / `_q1_tdt_ld_steady` from filter init values
2. At delay_first arm point (after line 1868): 4-line counter arm block
3. At delay_shift arm point (after line 1935): 4-line counter arm block (same code)
4. In per-hop filter attribute write section (after line 2281): ~12-line override block

**Total: ~27 lines across 3 files.** No `filters.py` changes.

### §5.2 What stays unchanged

- `handle_echo_path_change` path (EPV / shadow_rise → H_error=10000 → 100). Unchanged.
- Q2 fixed-size partition closure. Unchanged (no `n_partitions` changes).
- `W.fill(0)` (M2/M3). Unchanged.
- `_p_max_override` / `_p_floor_beta` / `_arc_m_q_boost`. Unchanged.
- Shadow filter leakage. Shadow is PBFDAF (NLMS, no `_leakage_converged`). No change needed.
- All other `use_aec3_*` flags. Unchanged.

---

## §6 Risk analysis

### §6.1 FS echo loss (HIGH attention, low expected severity)

**Mechanism:** Elevated transient leakage_converged (5× our steady) makes H_error
rise faster after delay acquisition. Higher H_error → larger mu → filter adapts faster.
On FS cases, faster adaptation is generally beneficial (tracks the new delay path
quickly). Risk: in pure FS cases where there's no actual path change after delay_first
(case opens with a fixed delay, delay_first just acquires the initial alignment), the
transient leakage bump runs for 2.5 s while the filter is adapting to the CORRECT
alignment from the start. This is harmless if H_error is already low (0.2–0.8 as
observed in 6/8 cases in Path C). The leakage push is mild: from ~0.3 equilibrium, adding
1.25e-2 × erl (~0.01 × 0.1 = 0.001 per hop) for 250 hops would raise H_error by ~0.25
before leakage balances with the K·E decay term. This is within the normal equilibrium
range (well below 100 ceiling).

**Worst case:** FS case with low far-end power (small erl numerator) → leakage effect
small anyway. FS case with high far-end power (erl close to 1.0): H_error rises faster →
mu larger → possible echo leakage if filter over-tracks. Monitor `9xjhiFbGo` (FS_static)
and `wlAXM0iDgkm` (FS_mvmt) in validation.

### §6.2 DT speech damage (MEDIUM attention)

**Mechanism:** If delay_first fires during DT speech (Path C: 6/8 movement cases fired at
t=0.7–1.5 s, which may coincide with DT speech), the transient leakage is 5× elevated
for 2.5 s. During DT, the `_disallow_leakage_diverged` flag (from coarse reset hangover)
keeps the filter in the converged leakage branch. Transient converged leakage = 1.25e-2
vs steady 2.5e-3 (5× increase). Effect: H_error rises faster → mu larger → filter more
aggressively adapts during DT → potential DT speech damage if the filter "chases" DT
speech frames as echo.

**Protection in BALANCED config (`enable_dtd=False`):** The DoubleTalkDetector is NOT
active in BALANCED — DTD is not the mitigation. Actual protections are: (1)
`_disallow_leakage_diverged` (coarse-reset hangover gate) forces the converged leakage
branch during DT-like frames — only the converged rate (1.25e-2) applies, not the diverged
rate (1.25); (2) PathChangeRegimeHandler may engage `main_paused` if the shadow filter
diverges; (3) H_error affects mu magnitude but the W-update stability depends on far-end
power more than H_error alone. **This is a REAL validation risk** — DT speech damage
from 5× elevated converged leakage must be empirically confirmed via Phase 1 trace + Phase
2 AECMOS on DT_mvmt cases. Do not rely on theoretical bounds.

Monitor: `nVUnxqHLr` (DT_static, delay_first at t=1.19 s with H_error=2.6) and
`XRTnTUjU_with_movement` (DT_mvmt stress, delay_first at t=1.54 s).

### §6.3 H_error ceiling interaction (LOW risk)

**Mechanism:** Transient leakage_diverged = 1.25 per hop. At H_error_ceil = 100 and
erl = 0.1: each hop adds 1.25 × 0.1 = 0.125. From H_error = 50: would reach 100 in
400 hops without K·E decay. In practice, K·E decay dominates for active-far cases and
H_error stays well below 100. If EPV fires simultaneously and H_error = 100 (at ceiling),
transient leakage adds nothing (clips). Safe.

### §6.4 Q2 isolation

Q2 (dynamic partition sizing) is CLOSED. This candidate does NOT touch `n_partitions`.
The transient leakage applies to the existing fixed-size PBFDKF — no conflict with Q2.

### §6.5 Interaction with existing EPV/shadow_rise wiring

EPV/shadow_rise continue to reset H_error to 10000 → 100. If EPV fires AFTER
delay_first while transient window is running, EPV resets H_error to 10000 → 100
while leakage is still at transient rate (1.25e-2 converged). At H_error=100, leakage
adds 1.25e-2 × erl per hop → clips to 100. Neutral effect. Safe.

---

## §7 Validation gate

### §7.1 Phase 1 — 9-case trace check (same cohort as Path C)

Run the Path C script (`python/v3_21_q1_minibench.py`) with the new flag ON. Verify:
1. Transient counter arms on delay_first / delay_shift (check `_q1_tdt_rem` in diag)
2. Filter's `_leakage_converged` = 1.25e-2 during transient window (can add to diag dump)
3. H_error_mean rises faster vs baseline in the 350 hops after delay_first
4. **Timing validation** — for each delay event, report:
   - the exact frame (hop index) of the delay_first / delay_shift event
   - the first frame where `_leakage_converged` changes from the steady value (2.5e-3) to the transient value (1.25e-2) — expected: same hop as the event or next hop; confirm which
   - the first frame where H_error_mean departs upward from the pre-event baseline — expected: event-hop+1 (because PBFDKF.process reads the leakage written at the end of the previous hop); confirm empirically
   - whether any off-by-one exists between arm, leakage write, and H_error effect
5. Counter expires after 350 hops (`_q1_tdt_rem = 0`); leakage restores to cached steady (2.5e-3)
6. Smoothing: leakage interpolates from 1.25e-2 → 2.5e-3 linearly over hops 100 → 1
7. EPV/shadow_rise behavior unchanged (verify on nVUnxqHLr EPV at t=4.26 s; H_error must still jump to 100)
8. Byte-equality with flag OFF vs baseline production (must be bit-identical when OFF)

### §7.2 Phase 2 — 9-case AECMOS quick check (if Phase 1 passes)

AECMOS on Path C cohort only. Gate: no DT_mvmt case regression > 0.05 dB vs baseline.
This is a quick sanity check before the 12-case.

### §7.3 Phase 3 — 12-case AECMOS bench

Standard 12-case bench (`eval_aec_challenge.py` on the 12-case subset).
Ship gate: ≥ 4/5 criteria PASS (same gate as all prior v3.21.x candidates).
Must not add new Cat C stress regressions.

### §7.4 Phase 4 — 800-case bench

Only after Phase 3 passes. Standard 800-case bench.
Not authorised until Phase 3 gate is met.

---

## §8 Open questions (NOT blocking implementation)

1. **Should the counter decrement only on active-render hops?** (Matches AEC3 semantics;
   slightly more complex — needs to check `_render_activity_state` or `_far_power_high`
   per hop.) Recommendation: use wall-clock hops for v3.21.x (simpler, conservative).

2. **Should shadow filter get a parallel transient bump?** Shadow is PBFDAF (NLMS, no
   `_leakage_converged`). Shadow doesn't use H_error. No direct equivalent. Skip for now.

3. **Should delay_first on STATIC cases get the transient bump?** Path C found that
   static guard cases (`nVUnxqHLr`, `9xjhiFbGo`) also fire delay_first at t=0.7–1.2 s.
   Including these is correct AEC3 parity (AEC3 doesn't distinguish movement from static
   in the delay-first response). The transient bump on static cases is short (~2.5 s at
   case start) and low-risk given the H_error values observed.

4. **Interaction with `use_aec3_epc_classification=True`**: If that flag is ever enabled
   (not production), `handle_echo_path_change(delay_change=True)` would also fire on
   delay_first/shift, resetting H_error to 10000. With the transient leakage ON
   simultaneously, the leakage bump would run from the 10000/100 ceiling — effectively
   identical to EPV/SR behaviour. The two flags can coexist safely (leakage override
   applies regardless of H_error initial value).

---

## §9 Summary of proposed changes

| Change | Location | Lines |
|---|---|---|
| Add 2 transient leakage constants | `aec3_scale.py` | +2 |
| Add 3 config fields | `config.py` | +3 |
| Init counter + cache steady leakage | `orchestrator.py:__init__` | +3 |
| Arm counter at delay_first | `orchestrator.py:~1869` | +3 |
| Arm counter at delay_shift | `orchestrator.py:~1936` | +3 |
| Per-hop leakage override | `orchestrator.py:~2282` | +18 |
| **Total** | 3 files | **~30 lines** |

No `filters.py` changes. Flag OFF = byte-equal. No benchmark authorised until
Phase 1 + Phase 2 + Phase 3 pass in sequence.
