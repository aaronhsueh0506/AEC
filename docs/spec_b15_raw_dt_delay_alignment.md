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
# Reuse filter's existing near_spec (same rfft convention and fft_size
# as echo_spec, so subtraction is unit-consistent by construction).
# Filter populates self.filter.near_spec at line 674-676 before
# returning from process(); AEC.process sees the same bin layout as
# self.filter.echo_spec (line 699).
mic_pwr_fft   = float(np.mean(np.abs(self.filter.near_spec) ** 2)) + 1e-10
echo_est_pwr  = float(np.mean(np.abs(self.filter.echo_spec) ** 2)) + 1e-10

# Note: mic_pwr_fft may differ from existing time-domain mic_pwr
# (line 2893) due to FFT window / overlap. Use mic_pwr_fft here
# for unit-consistent subtraction. Leave existing time-domain
# mic_pwr untouched (other consumers depend on it).

other_pwr  = max(mic_pwr_fft - echo_est_pwr, 1e-10)
raw_dt_new = other_pwr / (echo_est_pwr + other_pwr)
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

#### Unit consistency note

Both `mic_pwr_fft` and `echo_est_pwr` use `np.mean` over FFT bin
magnitudes of the filter's internal spectra, giving mean-square
power in frequency domain with **identical bin layout** (same
`fft_size`, same rfft convention, same source buffer scaling).

Why reuse `self.filter.near_spec` instead of recomputing `rfft`
from `near_end` at AEC.process level:

- `near_end` passed into AEC.process is hop-sized; recomputing
  `np.fft.rfft(near_end, n=fft_size)` would zero-pad and produce a
  **different bin distribution** from `self.filter.echo_spec`
  (which is computed from the filter's `near_buffer`, a
  fft_size-long circular buffer at `aec.py:674`). Their
  subtraction would be ill-defined.
- The filter already publishes `self.near_spec` as a public
  attribute (declared at `aec.py:92`, initialised at line 640,
  populated at line 674-676 by
  `np.fft.rfft(self.near_buffer, self.fft_size)`). Reuse is free
  and guarantees bin alignment.

Parseval note: with the codebase's rfft convention (no explicit
normalization), `Σ|X|²` gives N× time-domain mean; using
`np.mean()` over bins aligns units.

This creates a NEW computation for mic power, parallel to the
existing time-domain self-computed `mic_pwr` at line 2893.
**Do NOT modify the existing `mic_pwr`** — other consumers rely on
its current semantics.

#### Convergence fallback with EMA smoothing

Raw Boolean fallback (converged: use new; not converged: use legacy)
causes `raw_dt` to jump ~0.7 in a single frame when
`self._filter_converged` flips, producing audible click artifacts
downstream.

Mitigation: blend new and legacy formulas with a convergence-weighted
EMA:

```python
# Compute both formulas unconditionally (cheap)
raw_dt_new = other_pwr / (echo_est_pwr + other_pwr)
raw_dt_legacy = 1.0 - far_pwr / (mic_pwr + far_pwr + 1e-10)

# Smooth transition using _conv_counter (existing state in AEC)
# Assumes _conv_counter increments per frame while converging,
# resets to 0 on EPC reset.
CONV_BLEND_FRAMES = 20   # ~200ms @ 10ms hop
conv_weight = min(self._conv_counter / CONV_BLEND_FRAMES, 1.0)

raw_dt = conv_weight * raw_dt_new + (1.0 - conv_weight) * raw_dt_legacy
```

This gradually shifts `raw_dt` from legacy to new formula over
~200ms after convergence, avoiding click artifact.

Edge cases:
- `_conv_counter` reset on EPC: `raw_dt` smoothly returns to legacy
  during filter re-convergence (correct behavior — filter output
  unreliable during re-learning)
- `_conv_counter` not yet incremented (first frames):
  `conv_weight=0`, `raw_dt = raw_dt_legacy` (safe)

If `self._conv_counter` doesn't exist or behaves differently than
assumed, Stage 1B must verify before implementing.

#### _conv_counter fallback policy

The EMA formula relies on `self._conv_counter` existing and
incrementing once per frame (reset on EPC). Stage 1B MUST first
verify this assumption:

```bash
grep -n "_conv_counter" aec.py
```

Three outcomes and their required handling:

**Case 1: `_conv_counter` exists and behaves as assumed**
(increments per frame, resets on EPC)
→ Implement EMA blend as specified. Normal path.

**Case 2: `_conv_counter` exists but semantics differ**
(e.g. only counts frames after EPC, doesn't count during warmup;
or only increments under specific conditions)
→ Implement with observed semantics, document the divergence in
Stage 1B commit message.
→ If the divergence makes `conv_weight` stuck at 0 or 1, escalate:
STOP Stage 1B and report back; do NOT implement with broken blend.

**Case 3: `_conv_counter` does not exist**
→ Introduce it as a NEW state in AEC class, alongside
`self._filter_converged`. Semantics:

```python
# In AEC.__init__
self._conv_counter = 0

# In AEC.process, after self._filter_converged update:
if self._filter_converged:
    self._conv_counter += 1
else:
    self._conv_counter = 0   # reset on divergence or EPC
```

→ This adds minimal new state (one integer). Document in Stage 1B
commit message.

**Note on semantics**: `self._filter_converged` flips False→True
when the filter converges post-warmup, and True→False on EPC
trigger or divergence. `_conv_counter` is thus precisely "frames
since last convergence". Do NOT search for a separate EPC-reset
path for `_conv_counter` — the `_filter_converged` semantics
already provide it. EMA blend behavior:

- EPC triggered → `_filter_converged=False` → counter=0 →
  `conv_weight=0` → `raw_dt = legacy` (safe during re-learning)
- EPC hangover → counter stays 0 → still legacy
- Filter re-converges → counter starts incrementing → gradually
  shifts to new formula over 20 frames (~200ms)

**Not acceptable fallbacks (do NOT do):**

- Degrade to Boolean `_filter_converged` (causes 0.7 jump at
  convergence, defeating the purpose of EMA blend)
- Disable B-15 entirely (defeats the purpose of this spec)
- Skip the EMA blend and use pure new formula without convergence
  gate (may regress on unconverged frames)

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

**Placement**: inside the `else:` branch of `if self.config.enable_dtd:`
at `aec.py:3362`, immediately before the (possibly new) `raw_dt`
assignment. **Never run in the DTD branch** (protects Invariant
§8.5-1).

Formula MUST match §3 Option A exactly (same `np.mean`, same
`self.filter.near_spec` / `self.filter.echo_spec` reuse), so
`_diag_raw_dt_new` equals what the flag-ON path would produce:

```python
# Always compute both for observability (else-branch only)
raw_dt_legacy_diag = 1.0 - far_pwr / (mic_pwr + far_pwr + 1e-10)
if (hasattr(self.filter, 'echo_spec')
        and hasattr(self.filter, 'near_spec')):
    mic_pwr_fft_d  = float(np.mean(np.abs(self.filter.near_spec) ** 2)) + 1e-10
    echo_est_pwr_d = float(np.mean(np.abs(self.filter.echo_spec) ** 2)) + 1e-10
    other_pwr_d    = max(mic_pwr_fft_d - echo_est_pwr_d, 1e-10)
    raw_dt_new_diag = other_pwr_d / (echo_est_pwr_d + other_pwr_d)
else:
    mic_pwr_fft_d   = 0.0
    echo_est_pwr_d  = 0.0
    raw_dt_new_diag = raw_dt_legacy_diag

self._diag_raw_dt_legacy = float(raw_dt_legacy_diag)
self._diag_raw_dt_new    = float(raw_dt_new_diag)
self._diag_raw_dt_delta  = float(raw_dt_new_diag - raw_dt_legacy_diag)
self._diag_echo_est_pwr  = float(echo_est_pwr_d)
self._diag_mic_pwr_fft   = float(mic_pwr_fft_d)
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

## 8.5 Implementation invariants

The following must hold in Stage 1B implementation:

1. **else-branch only**: All B-15 formula changes must be inside the
   `else:` branch of `if self.config.enable_dtd:` at line 3363.
   The DTD branch (`raw_dt = self.get_dtd_confidence()`) MUST be
   untouched, even in diagnostic output paths.
2. **Legacy `mic_pwr` untouched**: The existing time-domain `mic_pwr`
   at `aec.py:2893` must not be modified. B-15 introduces new
   `mic_pwr_fft` for the formula only; downstream consumers of line
   2893's `mic_pwr` (grep to verify: ENR gate, saturation detector,
   etc.) keep their current semantics.
3. **Diagnostic unconditional**: §5 diagnostic block runs regardless
   of `AEC_FIX_B15` flag (always populates `_diag_raw_dt_legacy`,
   `_diag_raw_dt_new`, `_diag_raw_dt_delta`), so post-hoc comparison
   is always available.
4. **Flag gates swap only**: `AEC_FIX_B15` controls which formula's
   result becomes `raw_dt`, not which formula is COMPUTED. Both are
   always computed for diagnostic (performance cost ~1 FFT per frame,
   acceptable).

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

### Stage 1B implementation verification

- [ ] §3 公式用 Parseval-aligned `mic_pwr_fft` (not line 2893 `mic_pwr`)
- [ ] EMA blend 用 `self._conv_counter / CONV_BLEND_FRAMES`
- [ ] DTD if-branch 在 diff 中未出現任何變更 (grep 驗證)
- [ ] diagnostic 無條件執行 (不在 `if AEC_FIX_B15` block 內)

## 10. Not in scope (Stage 1A)

- Implementation (Stage 1B)
- 800-case benchmark (Stage 1C)
- Tuning convergence threshold (Stage 1B if needed)
- Category D movement cases — may need 組 7 delay tracker orthogonally
- Revert of any existing `_FIX` flag
