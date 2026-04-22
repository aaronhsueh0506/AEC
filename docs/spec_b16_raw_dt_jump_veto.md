# Spec: B-16 raw_dt jump veto

## 1. Background

Live diagnostic on PZ7V t=9.9-11.0s (Task 3 probe) confirmed the
cascade behind the FS#84 −13.83 dB leak:

```
1. Physical event: echo path gain jump (speaker volume +20 dB at t=9.95)
2. mic_pwr jumps 1e-4 → 0.18 (1700×); filter W still for OLD path
3. raw_dt = 1 - far_pwr / (mic_pwr + far_pwr)
          = 1 - tiny / big → 0.96 (single-frame jump 0.05 → 0.96)
   ← raw_dt energy-mismatch bug
4. effective_dt = max(raw_dt, shadow_dt) → saturates at 0.80
5. mu_scale crushed by effective_dt → K_scaled = K · mu_scale ≈ 0
6. Filter frozen: W_delta collapses 100000× (t=9.94: 1.26e-1
                                             t=10.00: 1.85e-5)
7. ERLE stuck at −21.5 dB (error > mic, filter output WORSE than nothing
   because W is wrong for the new path)
8. RES sees effective_dt = 0.80 → g_min rises to 0.9 → mic passes through
9. Leak = mic ≈ output → −13.83 dB at 10-11s
```

The earlier `docs/PZ7V_FS84_root_cause.md` correctly identified the
raw_dt bug. The earlier B-15 attempted to replace the formula with
delay-aligned echo_est_pwr — that failed because `filter.echo_spec`
magnitude lags onset (the filter is the same one being frozen;
circular dependency). B-15 is superseded.

Shadow / EPC layer was investigated orthogonally
(`experiment/pbfdkf-no-epc-baseline` branch): `AEC_EXP_NO_EPC=1`
produces bit-exact −13.83 dB leak at PZ7V t=10-11s. **EPC is
irrelevant** — it never fires at the onset anyway because
shadow_adv stays ≤ 1.05 under Kalman high-signal K-saturation.

## 2. Goal

Break the cascade `raw_dt jump → mu_scale crush → filter freeze → leak`
without touching:

- the `raw_dt` formula itself (legacy energy-mismatch formula preserved)
- `mu_scale` computation or `effective_dt` derivation
- Kalman internal math (P, Q, R, K, denominator)
- RES logic (gain curves, g_min, ENR, ne_protection)
- EPC trigger system (shadow-based or delay-shift)
- DTD path (`enable_dtd=True` uses `get_dtd_confidence()`, untouched)

Introduce a **veto layer**: detect `raw_dt` rate-of-change anomaly
characteristic of echo-path-gain jumps, and during such events fall
back to a slow-EMA `raw_dt` so downstream (mu_scale, RES) see the
correct FS classification.

## 3. Design: sustained 2-frame veto

### 3.1 Trigger rationale (from Task 4 + Task 5 probes)

7 cases probed. `raw_dt_jump_fast = raw_dt - raw_dt_ema_fast`
(EMA α=0.7, TC ≈ 50 ms). far_active frames only:

| case | type | max jump | 1-frame > 0.6 | **2-frame sustained > 0.6** |
|---|---|---|---|---|
| PZ7V (target) | FS onset | **+0.906** | 2 | **1** ✓ |
| W0zK3d | DT | +0.513 | 0 | 0 |
| nyT6 | DT | +0.725 | 4 | 0 |
| Tgtk | DT | +0.691 | 3 | 0 |
| XTqo_mv | DT-movement | **+0.823** | 6 | 0 |
| r7U6 | FS non-onset | +0.233 | 0 | 0 |
| HIMq | FS non-onset | +0.577 | 0 | 0 |

Single-frame threshold 0.6 has 3-6 false positives per DT case
(movement DT sees single-frame 0.82). **Sustained 2-frame threshold
0.6 cleanly separates** PZ7V onset (1 event) from DT / FS non-onset
(0 events across 4 DT + 2 FS cases).

### 3.2 New state in `AEC.__init__` and `reset()`

```python
self._raw_dt_ema_fast = 0.0           # ~50ms EMA (α=0.7)
self._raw_dt_ema_slow = 0.0           # ~500ms EMA (α=0.98)
self._raw_dt_jump_prev = 0.0          # previous frame jump_fast (sustained check)
self._diag_b16_veto_active = False    # diagnostic
```

4 scalar states added. EMAs reset to 0 on `AEC.__init__` and
`reset()`. No per-frame dependency on filter state.

### 3.3 Veto logic

**Placement**: inside the `else:` branch of
`if self.config.enable_dtd:` at `aec.py:3369`, **after** the
`raw_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)` assignment (or the
inst_erle correction at line ~3420 if that still runs for non-DTD),
**before** the EPC physical gate (`if self.epc_active: raw_dt = 0.0`
at line ~3423) and the `dt_indicator = np.clip(raw_dt, 0.0, 0.8)`
(line ~3427).

Exact insertion: just before the EPC gate, so veto modifies raw_dt
before EPC override and final clipping. The inst_erle correction
(line 3409-3420) is `if not self.config.enable_dtd:` which is the
same branch — leave it unchanged; run after it.

```python
# B-16: raw_dt jump veto (sustained 2-frame). Only in non-DTD branch
# (§5 invariant 1). See docs/spec_b16_raw_dt_jump_veto.md.
raw_dt_jump_fast = raw_dt - self._raw_dt_ema_fast
far_pwr_frame = float(np.mean(far_end ** 2) + 1e-10)
far_active = far_pwr_frame > 1e-4

# Diagnostic: always compute (independent of AEC_FIX_B16 flag, §5 inv 3)
self._diag_raw_dt_jump_fast = float(raw_dt_jump_fast)
self._diag_raw_dt_ema_fast  = float(self._raw_dt_ema_fast)
self._diag_raw_dt_ema_slow  = float(self._raw_dt_ema_slow)

veto_cond = (
    raw_dt_jump_fast > 0.6
    and self._raw_dt_jump_prev > 0.6
    and far_active
    and not self._is_stationary_far
)
self._diag_b16_veto_active = bool(veto_cond)

# Apply veto (gated by flag, §5 inv 4)
if AEC_FIX_B16 and veto_cond:
    raw_dt = self._raw_dt_ema_slow

# EMAs update using POST-veto raw_dt, so slow EMA is not
# contaminated by the jump event itself. prev-jump uses the
# pre-veto jump value (since veto decision is about the raw
# formula's output, not the vetoed value).
self._raw_dt_ema_fast = 0.7 * self._raw_dt_ema_fast + 0.3 * raw_dt
self._raw_dt_ema_slow = 0.98 * self._raw_dt_ema_slow + 0.02 * raw_dt
self._raw_dt_jump_prev = raw_dt_jump_fast
```

**Important subtlety (EMAs update order)**: `_raw_dt_ema_fast` must
be updated AFTER `jump_fast` computation (so jump is `current −
previous_ema`), matching the Task 4/5 probe definition exactly. EMAs
feed on post-veto `raw_dt` so that during a sustained veto event the
slow EMA tracks the safe fall-back value rather than the anomalous
0.96.

**`_raw_dt_jump_prev` stores the pre-veto jump** (the detector's
own raw output), so the next frame's sustained check uses the
detector signal, not the outcome.

### 3.4 Feature flag

```python
AEC_FIX_B16 = int(os.environ.get('AEC_FIX_B16', '0'))
```

Default OFF. Flag gates only the `raw_dt` replacement line; the
detector, EMAs, and diagnostic run unconditionally (so pre-release
comparison is free).

## 4. Trade-off: 1-frame veto latency

Sustained-2-frame design means PZ7V onset's **first** frame
(t=9.95s, jump=0.906) does NOT veto — only t=9.96s (second
consecutive, jump=0.652) onward vetoes. Filter freezes for ~10 ms
before veto kicks in.

Acceptance:

- Task 5 showed single-frame threshold 0.6 false-triggers 3 of 4
  DT cases (incl. movement DT single-frame 0.823). Unacceptable.
- PZ7V leak window is ~1 s (10-11 s). 10 ms latency costs
  ≤ 0.5 dB of the total leak reduction potential.
- 2-frame sustained is the coarsest clean separator across 7 probed
  cases. 3-frame would miss PZ7V (only 2 consecutive frames > 0.6).

## 5. Invariants (mandatory)

1. **else-branch only**: all B-16 code is inside the `else:` branch
   of `if self.config.enable_dtd:` at `aec.py:3369`. DTD path
   (`raw_dt = self.get_dtd_confidence()`) is untouched, including
   diagnostic. grep-verify in Stage 1B: the diff must not touch
   any line above the `else:` in that block.

2. **Legacy raw_dt formula preserved**: line `raw_dt = 1.0 - far_pwr
   / (mic_pwr + far_pwr)` is unchanged. B-16 reads raw_dt and
   conditionally overwrites only when veto fires.

3. **Diagnostic unconditional**: `_diag_raw_dt_jump_fast`,
   `_diag_raw_dt_ema_fast`, `_diag_raw_dt_ema_slow`,
   `_diag_b16_veto_active` populated every frame regardless of
   `AEC_FIX_B16`. EMAs also update unconditionally.

4. **Flag gates swap only**: `AEC_FIX_B16` controls ONLY the
   `raw_dt = self._raw_dt_ema_slow` replacement line. Detector
   computation, EMAs, and diagnostic run both ON and OFF.

5. **NE / stationary guards**: `far_active` and `not
   self._is_stationary_far` conditions ensure NE-only (far silent)
   and stationary-far (WN, tones) never trigger veto. Both are
   existing AEC attributes; no new NE / stationarity machinery.

## 6. Diagnostic placement

Inside else-branch, immediately before `if AEC_FIX_B16 and
veto_cond:`. Allows post-hoc plotting of jump_fast / EMAs / veto
active timeline on PZ7V and top-20 worst cases without re-running.

## 7. Smoke test criteria (Stage 1C)

Cases (6):
- PZ7V (target; FS onset)
- nyT6, W0zK3d, Tgtk (DT no-movement, from worst_dt_gap top)
- XTqo_mv (DT movement; max single-frame jump 0.823 — boundary case)
- HIMq, r7U6 (FS non-onset)

Metrics (per case, B16=0 vs B16=1):

| flag | pass criterion |
|---|---|
| B16=0 | bit-exact to baseline (Δ ≤ 0.01 dB on any window, pre-release sanity) |
| B16=1 PZ7V 10-11s | leak reduced from −13.83 dB to ≤ −20 dB (acceptable), target ≤ −25 dB |
| B16=1 DT full-file | no regression > 0.5 dB, XTqo_mv especially |
| B16=1 FS non-onset | delta < 0.1 dB |

Fail modes and responses:

- **PZ7V leak unchanged** (≥ −15 dB): veto triggered but downstream
  didn't respond. Probe: is `raw_dt_ema_slow` actually low at t=9.96
  (should be < 0.1)? If yes, downstream has additional DT signal
  source (shadow-derived) keeping effective_dt high — escalate.
- **DT regression > 0.5 dB**: veto false-triggered somewhere. Probe:
  find frames where `_diag_b16_veto_active=True` in DT case. If
  < 3 events → tune threshold to 0.65 or require 3-frame sustained.
  If many → root-cause the false positive.
- **FS non-onset regression**: should not happen (probe showed 0
  events). If does, raw_dt behaves differently under real audio
  than our `_diag_raw_dt_legacy` probe.

## 8. Stage 1D: 800-case benchmark

After Stage 1C passes:

- `AEC_FIX_B16=1` eval on 800-case blind set
- Target: ΔFS_echo better than current `full` baseline
  (was −0.147 dB; B-16 should recover ≥ 0.05 dB of this, proportional
  to Cat-A share in the worst-case distribution)
- Hard floor: no per-scenario regression > 0.05 dB (FS / DT_echo /
  DT_deg / NE_deg). B-16 must be a strict Pareto improvement.

## 9. Rollback plan

If Stage 1D regresses:

- `AEC_FIX_B16=0` reverts (detector and EMAs still populate, still
  gated off, zero behavioural change)
- Investigate via `_diag_b16_veto_active` per-frame trace on
  regressing cases
- Candidate mitigations (in order):
  1. Raise threshold to 0.65 (lose PZ7V onset margin; check Task 5
     data if still sustained 2-frame separable)
  2. Require 3-frame sustained (misses PZ7V by Task 5 data; need to
     redesign detector if chosen)
  3. Add `not self.epc_active` to veto guard (if EPC-triggered
     raw_dt=0 transitions produce spurious jumps)
- If none work: abandon B-16, keep spec + diagnostic for future
  attempt

**Not acceptable**: trading DT/NE/non-PZ7V FS for PZ7V recovery
when the net Pareto is negative.

## 10. Stage 1B completion checklist

- [ ] `AEC_FIX_B16` env flag declared alongside other `AEC_FIX_*`
- [ ] 4 new state vars in `AEC.__init__` and `reset()`
- [ ] Veto block inserted inside else-branch, position as §3.3
- [ ] Diagnostic unconditional (not in `if AEC_FIX_B16:`)
- [ ] EMAs update after veto-gated raw_dt assignment
- [ ] `_raw_dt_jump_prev` stores pre-veto jump
- [ ] Invariants §5 1-5 all verified by grep / diff inspection:
  - [ ] diff shows only `else:` branch region at line 3369+
  - [ ] `raw_dt = 1.0 - far_pwr / ...` unchanged
  - [ ] No `if AEC_FIX_B16:` wrapping the diagnostic block
  - [ ] `if AEC_FIX_B16:` wraps only one line (the replacement)
  - [ ] `far_active` and `not self._is_stationary_far` present in
        veto condition

## 11. Not in scope

- raw_dt formula redesign (B-15 scope, superseded; `b15_superseded.md`
  draft on main tree, un-committed)
- EPC system overhaul (investigated orthogonally in
  `experiment/pbfdkf-no-epc-baseline`, no action)
- Kalman shadow differentiation (Kalman K-saturation documented in
  `b15_superseded.md`, independent issue)
- `P_max` / `P_floor` re-tuning (independent, after B-16)
- Coherence-based guards (Task 1 archaeology confirmed coherence
  still live, not revisited in B-16)
