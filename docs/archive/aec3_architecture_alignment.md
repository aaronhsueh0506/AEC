# AEC3 architecture alignment — shadow algorithm mismatch

Date: 2026-04-22
Status: forward-looking note. **Not scheduled work.** Follow-up after
B-16 merge decision.

## 1. WebRTC AEC3 mainline dual-filter architecture

AEC3 uses **two filters of different algorithmic classes**:

### 1.1 Refined filter (the output-producing main filter)

- Source:
  https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/refined_filter_update_gain.cc
- **Algorithm: Kalman-like gain update**
  ```
  mu[k] = H_error[k] / (H_error[k]·X²[k] + n·E²[k])
  P_update → H_error[k] evolves per-bin
  ```
- Slow, high-quality adaptation. Corresponds to the `refined` role:
  stable output, careful learning.

### 1.2 Coarse filter (the shadow / backup tracker)

- Source:
  https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/aec3/coarse_filter_update_gain.cc
- **Algorithm: NLMS with fixed rate**
  ```
  mu = rate / X²
  ```
- Fast, simpler adaptation. Corresponds to the `coarse` role:
  responsive to changes (echo path shifts, gain jumps), copy W to
  refined when outperforming.

**Key design point**: the two filters are NOT the same class with
different hyper-parameters. They use **fundamentally different
update mechanisms** — Kalman for stability, NLMS for responsiveness.

## 2. Current system (this repo, PBFDKF mode)

Code reference: `python/aec.py:2548-2554`:

```python
self.shadow_filter = FilterClass(**shadow_kwargs)   # same class as main
...
if self.config.use_kalman:
    self.shadow_filter.Q_high = self.filter.Q_high * self.config.shadow_q_ratio
    self.shadow_filter.Q_low  = self.filter.Q_low  * self.config.shadow_q_ratio
    self.shadow_filter.Q      = self.shadow_filter.Q_high.copy()
```

- When `AEC_MODE=PBFDKF`, `FilterClass = PBFDKF` and shadow is also
  PBFDKF.
- Only differentiation: `shadow_q_ratio` (default 3.5) scales
  `Q_high`/`Q_low`, i.e. Kalman process-noise variance.
- Both filters follow the **same update equation** (line 877-971);
  they share the same denominator-saturation math.

## 3. Mismatch vs AEC3 (PBFDKF mode)

| Aspect | AEC3 mainline | This repo (PBFDKF mode) |
|---|---|---|
| Main algo | Kalman (refined) | Kalman (PBFDKF) |
| Shadow algo | **NLMS (coarse)** | **Kalman (PBFDKF)** ← mismatch |
| Shadow differentiation | Algorithm class | Q ratio only |
| Onset behaviour | Coarse NLMS responsive | Shadow K-saturates |
| Copy direction | refined → coarse (reset on re-convergence) | shadow → main (when shadow outperforms) |

### 3.1 Why Q-ratio alone fails (Task diagnostic confirmation)

Probed on PZ7V onset (Task 3-4, this branch):

- `Q_ratio = 3.50` exactly (config honoured)
- `P_ratio = 3.50 → 3.71` (grows faithfully with Q)
- **`K_ratio → 1.0`** at echo event: Kalman denominator
  `Σ P·|X|² + R + δ` is dominated by `Σ P·|X|²` term when signal is
  high; that term scales with P in both filters, so ratio cancels.
- Downstream: `shadow_err / main_err ≈ 1.00-1.05` throughout
  PZ7V t=9.9-10.5s (EMA probe result)
- `shadow_adv` never crosses EPC guard `1.3` → EPC never fires at
  onset (confirmed separately: `experiment/pbfdkf-no-epc-baseline`
  shows `AEC_EXP_NO_EPC` yields bit-exact −13.83 dB leak at PZ7V)

The root cause of shadow ineffectiveness in PBFDKF mode is
**structural, not parametric**: Kalman self-normalization neutralises
any P-based differentiation during high-signal regimes, precisely
the regime where EPC-trigger signal is needed.

## 4. Proposed fix (future, separate scope)

**Direction A (minimal)**: make PBFDKF-mode shadow use the existing
PBFDAF (NLMS) class, not PBFDKF:

```python
# Pseudo — not implemented
if self.config.use_kalman:
    # Main is PBFDKF; make shadow PBFDAF to mimic AEC3 coarse
    self.shadow_filter = PBFDAF(**shadow_kwargs_without_kalman)
    # shadow_mu ≈ config.mu (NLMS fixed rate), not Q ratio
```

- Main PBFDKF stays as-is (output quality unchanged)
- Shadow PBFDAF responds to echo-path-gain jumps via NLMS dynamics
  (no K-saturation)
- `shadow_adv` becomes genuine differentiator → EPC guards work
  as originally intended

**Direction B (symmetric)**: apply to PBFDAF mode too — shadow as
something orthogonal (e.g. a lower-mu PBFDAF with different
step-size rule). Lower priority.

## 5. Expected impact

- Shadow actually differentiates at onsets; `shadow_adv > 1.3`
  naturally triggers EPC during PZ7V-class events
- EPC guards (B-4a/4b/12) work as originally designed; no need to
  band-aid raw_dt via B-16
- B-16 still has defensive value (raw_dt bug is structural, EPC
  may not always fire fast enough); keep as belt-and-suspenders
- Potential PZ7V recovery: if shadow-driven EPC fires at onset and
  resets filter Q_high, filter may re-converge faster than B-16's
  460ms veto window → room for > −25 dB target (currently B-16
  alone reaches −20.42 dB)

## 6. Implementation risks

### 6.1 Copy gate compatibility

Current copy gate at `aec.py:3136-3145` uses
`self.filter.copy_weights_from(self.shadow_filter)` which on PBFDKF
copies W only (per line 979-981 `copy_weights_from` comment: "P/Q
are confidence states … copying them contaminates the Kalman gain").

If shadow becomes PBFDAF:
- PBFDAF W is the same FD complex-weight shape as PBFDKF W — copy
  is structurally compatible
- PBFDAF has no Q/P state — nothing to copy, nothing to contaminate
- PBFDKF main's P is learned from its own error → already decoupled
  from shadow state, so no change needed
- PBFDAF `_R` (measurement-noise tracker) is separate from PBFDKF
  `R` which is derived from `_error_psd × R_scale_smooth`; copy
  semantics need explicit review

### 6.2 Shadow step-size tuning

- Current: `shadow_q_ratio=3.5` (irrelevant once shadow is NLMS)
- New: need `shadow_mu` for NLMS shadow. Plausible starting point
  from AEC3 source: `mu = rate / X²` with `rate ≈ 0.3-0.5`
- Requires smoke test on several FS / DT / onset cases before
  800-case

### 6.3 Invariants that must hold

- PBFDAF class is a drop-in instance; main remains PBFDKF
- Main filter output (`self.filter`) is the only one producing
  audio output; shadow is EPC-trigger / copy-gate only
- `enable_dtd=False` path (legacy formula, used by eval) must not
  regress

### 6.4 Benchmark risk

- 800-case Pareto may shift meaningfully. This is a **larger scope**
  than B-16 (architectural, not a veto layer).
- Possible downside: DT/NE cases where current shadow-PBFDKF's
  smoothness actually helped copy-gate stability; PBFDAF shadow
  might over-trigger copies → output artefacts.
- Must be gated behind a flag with full bit-exact fallback path.

## 7. Suggested stage plan (not scheduled)

1. **B-16 Stage 1D** (in progress): finish 800-case benchmark. If
   PASS, merge `feature/b16-raw-dt-jump-veto` → main. If FAIL,
   diagnose / abandon B-16 first.
2. **After B-16 merge decision**: open `experiment/shadow-as-nlms`
   branch off main. Do not combine with B-16 scope.
3. Write `docs/spec_shadow_nlms_refactor.md`. Stage 1A audit must be
   **strict** — this touches core architectural invariants.
4. Smoke test on same 7 cases used in B-16 Stage 1C.
5. Full 800-case benchmark, same methodology as B-16 Stage 1D.
6. Merge decision: SAFE / MARGINAL / UNSAFE per same rubric.

## 8. Confidence and risk posture

**Confidence (+)**: this is explicitly aligning to WebRTC AEC3
mainline, which is a mature open-source reference. Shadow-as-NLMS is
**not an invention**, it is the upstream design. Deviation cost
(current Kalman-shadow) was identified as structural root cause by
two independent probes (shadow_adv plateau, Kalman K-saturation
math). Design direction carries higher prior than typical new-feature
work.

**Risk (−)**: scope is bigger than B-16. Multiple coupling points
(copy gate, tuning, benchmark regression risk) need careful
sequencing. Must not ship under time pressure; Stage 1A audit is
load-bearing.

## 9. References

- `docs/spec_b15_raw_dt_delay_alignment.md` + `docs/b15_superseded.md`
  — historical raw_dt fix attempt, superseded by B-16
- `docs/spec_b16_raw_dt_jump_veto.md` — current B-16 veto layer
- `docs/aec_methods.md §6.4` — internal dual-filter design intent
  (at introduction time, 2026-03-13, shadow was conservative
  mu×0.5 for divergence recovery; role inverted by v1.12.0 to
  aggressive Q×3 for copy-gate trigger)
- `python/aec.py:2525-2554` — shadow filter instantiation
- `python/aec.py:877-971` — PBFDKF `_update_weights` (shared by
  main and current shadow, the saturating denominator is here)
- `experiment/pbfdkf-no-epc-baseline` branch — empirical evidence
  that EPC never fires at PZ7V onset (bit-exact leak with EPC off)
