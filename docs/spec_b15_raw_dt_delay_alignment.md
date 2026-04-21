# Spec: B-15 raw_dt delay-alignment fix

## 1. Background

Per `docs/PZ7V_FS84_root_cause.md`:

`AEC.process()` at `python/aec.py:3365`:
```python
raw_dt = 1.0 - far_pwr / (mic_pwr + far_pwr)
```
compares **mic at frame t** against **far at frame t** (same-time window).
Mic at t contains echo from `far[t−delay]`, so at echo onset the ratio
mis-fires:

| t      | mic_pwr | far_pwr | raw_dt | truth   |
|--------|---------|---------|--------|---------|
| 9.96s  | —       | —       | 0.241  | FS      |
| 10.04s | 0.0995  | 0.0026  | 0.974  | FS (echo onset) |
| 10.10s | —       | —       | 0.885  | FS      |

At PZ7V t=10.04s, mic/far = 38× because echo arrival energy is compared
against a quiet far-end sample. `raw_dt` → 0.974 → `effective_dt = 0.8`
→ ENR gate stops suppressing echo → **+15 dB leak**.

**This is a pre-existing latent bug.** `OLD` `aec.py` has bit-exact the
same formula; OLD's `epc_active=True` hangover zeroed `raw_dt` every
frame, masking the bug. Phase 1+2 NEW EPC guards (B-4a/4b/12) are
correct but no longer mask the bug → **exposure, not introduction**.

Evidence:
- 800-case bisect: no single flag explains −0.147 FS regression
  (`docs/bisect_analysis_and_plan.md`)
- Top-20 worst FS-gap cases: 12/20 FS no-movement (Category A),
  8/20 FS with_movement (Category D) — symptom matches PZ7V.

## 2. Goal

Replace the same-time `mic_pwr / far_pwr` comparison with a
**delay-aligned** echo-power reference so that:

- Echo-onset frames no longer mis-fire `raw_dt` → 0.97
- True double-talk (near-end speech during far-end activity) still
  raises `raw_dt` → > 0.4
- No regression on pure FS (low-coupling) or pure near-end singletalk

## 3. Design options

### Option A — Delay-aligned echo estimate as far-reference

```python
echo_est_pwr = float(np.sum(np.abs(self.filter.echo_spec) ** 2)) + 1e-10
other_pwr   = max(mic_pwr - echo_est_pwr, 1e-10)
raw_dt = other_pwr / (echo_est_pwr + other_pwr)
```

Physical intent: `echo_spec` is the filter's delay-aligned echo estimate
for the **current** frame. If mic is dominated by echo_est, raw_dt low.
If mic has significant non-echo energy (near speech), raw_dt high.

- **Pros**: directly treats root cause (delay mismatch); reuses existing
  `filter.echo_spec` already computed at `aec.py:699`; no new
  hyper-parameter.
- **Cons**: depends on filter convergence. Before filter converges,
  `echo_est_pwr` underestimates → `raw_dt` inflates → false DT.
  Mitigation: gate by `self._filter_converged` (already available in
  AEC.process), fall back to legacy formula until converged.
- **Complexity**: low (~10 lines).

### Option B — Mic-minus-echo residual ratio

```python
residual_pwr = max(mic_pwr - echo_est_pwr, 0.0)
raw_dt = np.clip(residual_pwr / (mic_pwr + 1e-10), 0.0, 0.8)
```

Physical intent: large residual → near-end speech present.

- **Pros**: same philosophy as A but symmetric (residual / mic); no
  `other_pwr` denominator with potential 0.
- **Cons**:
  - Depends on `echo_est_pwr` accuracy, same filter-convergence risk
    as A.
  - At steady-state FS the residual is already small (filter cancels
    echo), so raw_dt stays low — good. But during **divergence** or
    **ERL spikes**, residual briefly large → false DT. More fragile
    than A around transient filter errors.
  - Clipping to 0.8 mirrors existing post-clip at line 3427 —
    redundant.
- **Complexity**: low (~8 lines).

### Option C — Echo-onset guard on top of legacy formula

```python
raw_dt_legacy = 1.0 - far_pwr / (mic_pwr + far_pwr)
echo_onset = (echo_est_pwr > 2.0 * self._echo_est_pwr_prev)
if echo_onset:
    raw_dt = min(raw_dt_legacy, self._raw_dt_prev * 0.5)
else:
    raw_dt = raw_dt_legacy
self._echo_est_pwr_prev = echo_est_pwr
self._raw_dt_prev = raw_dt
```

Physical intent: detect echo onset spike in `echo_est_pwr`, suppress
raw_dt rise during that single frame.

- **Pros**: conservative — keeps legacy formula for non-onset frames,
  minimal downstream cascade.
- **Cons**: band-aid on top of a known broken formula; 2.0× threshold
  is a new magic number; misses delayed-onset cases where `echo_spec`
  rises slowly over 2-3 frames (no single-frame spike).
- **Complexity**: medium (new state `_echo_est_pwr_prev`,
  `_raw_dt_prev`; threshold tuning).

### Recommendation

**Option A (default ON behind `AEC_FIX_B15` flag, with convergence
gate).** Reasons:

1. Treats root cause (delay mismatch) directly instead of patching
   symptoms.
2. Reuses already-computed `self.filter.echo_spec`; no new state.
3. Convergence gate (`self._filter_converged`) is already a tested
   signal in the codebase.
4. Option C doesn't generalise to slow-rise onset (Category D
   `with_movement` cases).
5. Option B is similar intent but more fragile during filter
   transients.

Fallback: if Option A regresses 800-case, fall back to Option C as a
safer incremental fix, keeping Option A code behind a nested flag.

## 4. Feature flag

```python
# near top of AEC class (alongside existing _FIX flags)
AEC_FIX_B15 = int(os.environ.get('AEC_FIX_B15', '0'))  # default OFF
```

- `AEC_FIX_B15=0` (default): keep legacy `raw_dt = 1 - far_pwr/(mic_pwr+far_pwr)`
- `AEC_FIX_B15=1`: Option A formula, with convergence fallback

Starts OFF because cascade risk (Section 7) is non-trivial; user flips
ON for verification runs.

## 5. Diagnostic

Add inside `AEC.process()` around line 3365, unconditional (not gated
by flag) so both branches populate:

```python
# Always compute both for observability
raw_dt_legacy_diag = 1.0 - far_pwr / (mic_pwr + far_pwr)
if hasattr(self.filter, 'echo_spec'):
    echo_est_pwr = float(np.sum(np.abs(self.filter.echo_spec) ** 2)) + 1e-10
    other_pwr = max(mic_pwr - echo_est_pwr, 1e-10)
    raw_dt_new_diag = other_pwr / (echo_est_pwr + other_pwr)
else:
    raw_dt_new_diag = raw_dt_legacy_diag

self._diag_raw_dt_legacy = float(raw_dt_legacy_diag)
self._diag_raw_dt_new    = float(raw_dt_new_diag)
self._diag_raw_dt_delta  = float(raw_dt_new_diag - raw_dt_legacy_diag)
self._diag_echo_est_pwr  = float(echo_est_pwr)
```

Allows post-hoc plotting of legacy vs new on PZ7V and top-20 worst
cases without re-running.

## 6. Smoke test plan

Single-file PZ7V:
```bash
cd python
AEC_MODE=PBFDKF AEC_EPC_MULTI_LEVEL=1 AEC_FIX_B15=0 \
  python3 _run_single.py PZ7V0SfxUkem4IalTp1YgA farend_singletalk
AEC_MODE=PBFDKF AEC_EPC_MULTI_LEVEL=1 AEC_FIX_B15=1 \
  python3 _run_single.py PZ7V0SfxUkem4IalTp1YgA farend_singletalk
```

Pass criteria (AEC_FIX_B15=1):
- t=10.0–10.4s: `dt_indicator` stays < 0.3 (vs ≥ 0.8 on legacy)
- 10-11s output: ≤ −25 dB (vs −13.83 dB on legacy full)
- NE singletalk: no audible speech degradation (spot-check waveform)

Pass criteria (AEC_FIX_B15=0):
- bit-exact to current `full` run (−13.83 dB leak reproduces)
  — confirms flag gating correct.

Then sample 3 of each category before 800-case:
- Cat A: JteZUZ4JYkeD4k2rcVbqHg, VGlWeOPC6UiXSq4SYPiKpw, JLNgGcvTNEqbTDbc28wLkg
- Cat D: iOyPaxX11UOaUkcscKhq1A_with_movement, s0oJqM6Y1UCHSVmHmgsx4Q_with_movement, JjCzlhn3gEiBQvfJtPNJ9A_with_movement

Expect ≥ 1.0 dB FS_echo improvement on Cat A; Cat D uncertain
(movement may still need delay-tracker fix in 組 7).

## 7. Risk analysis — downstream cascade

`raw_dt` feeds (through `dt_indicator` post-clip at line 3427):

### 7.1 `dt_indicator` consumers (direct)

| Consumer | Line | Behaviour change risk |
|----------|------|-----------------------|
| `FilterErleEstimator.update(dt_indicator)` | 1516 | `dt_factor` freezes ERLE rise. Lower raw_dt at onset → ERLE rises earlier → shadow_dt can grow. Generally safe (closer to FS-truth). |
| `FbErleEstimator.update(dt_indicator)` | 1519 | Gate `dt_indicator > 0.3` skips update. Lower raw_dt at onset → fb_erle tracks more. Small improvement expected. |
| `dt_for_fs` / `effective_dt` | 1543, 1550 | **Primary intended effect.** At PZ7V onset dt_for_fs drops 0.8→~0.2, effective_dt driven by shadow_dt instead. This is the fix. |
| `dt_temporal` | 1940 | Uses `max(dt_indicator, effective_dt*0.5)`. Lower raw_dt → dt_temporal lower → gain rise faster in true-FS onset (desired). |
| `is_learning_safe` | 2017 | `dt_indicator < 0.1` allows learning. Lower raw_dt at onset → more learning allowed → potentially faster filter update but riskier if still DT. Monitor via PZ7V divergence trace. |

### 7.2 `effective_dt` downstream

`effective_dt = max(dt_for_fs, shadow_dt)` — fix lowers dt_for_fs at
onset, so `effective_dt` will mostly equal `shadow_dt` at onset.
Shadow_dt (per-bin EER) is orthogonal to raw_dt, so overall
`effective_dt` behaviour depends on shadow filter state:

- **FS onset (true)**: shadow_dt low → effective_dt low → fs_confidence
  high → aggressive suppression. **Fixes leak.**
- **DT onset (true)**: shadow_dt should still rise (near speech in
  residual) → effective_dt correct. **No regression.**
- **DT with high coupling**: shadow_dt alone carries DT signal;
  raw_dt was previously over-estimating in this case too. Needs
  verification on NE_deg.

### 7.3 Shadow filter `mu_scale`

Shadow filter decisions gated on `dt_indicator` (e.g. shadow-adv
checks). Lower raw_dt at FS onset → shadow_adv more likely to trigger →
EPC can fire → **extra safety**, not regression.

### 7.4 EPC decision

Guards at line ~3171: `dt_signal < 0.3` required for EPC. Lower raw_dt
at onset means `dt_signal` easier to stay < 0.3 → EPC can fire during
true FS onset. This is exactly the pre-Phase1 behaviour EPC was
designed for, so **aligned with B-4a/4b/12 intent**, not a regression.

### 7.5 `mu_eff` (filter step-size)

PBFDKF/PBFDAF `mu_eff` paths read `dt_indicator` to freeze learning
during DT. Lower raw_dt at FS onset → filter continues learning through
onset → faster convergence on FS clips → potentially better ERL
tracking. Positive.

### 7.6 `dt_reduction` → `effective_over_sub`

Line 3432. Dead code for `gain_type="enr"` (all presets). No effect.

### 7.7 Unknown unknowns

`enable_dtd=True` path (line 3363, `get_dtd_confidence`) is **not
touched by this spec**. If any preset sets `enable_dtd=True`, spec
needs extension. Verify by grep `enable_dtd` across configs during
Stage 1B implementation.

## 8. Rollback plan

- `AEC_FIX_B15=0` (env) → legacy formula restored, bit-exact to
  current `full` run.
- If 800-case FS improves but DT_echo / NE_deg regresses > 0.02:
  - Add convergence-gate threshold tuning (e.g. also require
    `self._filter_erle_est.erle_db > 10` before applying new formula).
  - Or retreat to Option C (echo-onset guard only).
- If PZ7V t=10s still leaks with Option A:
  - Likely `echo_est_pwr` underestimating (filter not tracking). Add
    fallback: when `echo_est_pwr < 0.1 * mic_pwr` AND
    `self._filter_converged`, treat as divergence and use legacy.

## 9. Completion checklist

- [x] Spec committed
- [x] Option selected: **Option A** (with `_filter_converged`
      fallback)
- [x] Implementation spec precise enough for direct coding:
  - File: `python/aec.py`
  - Flag declaration: add near other `AEC_FIX_*` (line 27-55 area)
  - Diagnostic block: insert at line ~3362 (before `raw_dt = ...`)
  - Formula swap: replace line 3365 with gated Option A
  - No changes to DTD path (line 3363)
  - No changes to EPC gate (line 3423-3425)

## 10. Not in scope (Stage 1A)

- Implementation (Stage 1B)
- 800-case benchmark (Stage 1C)
- Tuning convergence threshold (Stage 1B if needed)
- Category D movement cases — may need 組 7 delay tracker orthogonally
- Revert of any existing `_FIX` flag
