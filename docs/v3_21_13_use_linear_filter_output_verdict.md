# v3.21.13 — AEC3 UseLinearFilterOutput final-output selection parity verdict (PAUSED CANDIDATE)

**Date**: 2026-05-22
**Verdict**: **PAUSED CANDIDATE.** Mixed Pareto vs A baseline (modest stress regression preserved); but **substantial recovery of v3.21.7 Cat C stress damage vs B** (+0.65 to +1.20 deg per stress case). First v3.21.x candidate to materially address the downstream Cat C mechanism. Awaiting user decision on combination with partition_summed_x2 + production-default disposition. Code default-OFF preserved.

## AEC3 source reference

`docs/aec3_extracts/src/aec3/echo_remover.cc:445-475`:

```cpp
// Compute the FFTs.
WindowedPaddedFft(fft_, y->View(/*band=*/0, ch), y_old_[ch], &Y[ch]);
WindowedPaddedFft(fft_, e[ch], e_old_[ch], &E[ch]);
...
const auto& Y_fft = aec_state_.UseLinearFilterOutput() ? E : Y;
...
suppression_filter_.ApplyGain(comfort_noise, comfort_noise_high_bands,
                              high_bands_gain, G, Y_fft, e);
```

`aec_state.h:60-63`:
```cpp
bool UseLinearFilterOutput() const {
  return filter_quality_state_.LinearFilterUsable() &&
         config_.filter.use_linear_filter;
}
```

AEC3 applies the suppression gain to the **capture-windowed FFT Y** when the linear filter isn't usable, falling back to gain-suppressed mic instead of gain-suppressed linear residual. Our v3.21.6 default ALWAYS applies the gain to E (the residual), regardless of `usable_linear_estimate()`. This is the v3.21.x production parity gap.

## Implementation

- Flag: `use_linear_filter_output_selection_for_final_output: bool = False` ([config.py](AEC/python/modules/config.py))
- Env: `AEC_USE_LINEAR_OUTPUT_SELECT=1`
- Code: branch at the `e_out_spec = error_spec * gain` line in `_aec3_post` ([orchestrator.py](AEC/python/modules/orchestrator.py))
- Y derivation: `near_spec_win = filter.error_spec_windowed + filter.echo_spec`
  - PBFDKF.process line 195 sets `error_spec_windowed = near_spec_win - echo_spec`
  - Reversing the subtraction yields `near_spec_win`, AEC3-equivalent of `WindowedPaddedFft(y_post_hpf, y_old, sqrt-Hann)`
  - Independent of v3.21.8 UseRefinedOutput (which selects refined vs coarse within the linear branch)
- Trace counter: `self._v3_21_13_trace = {'frames_total', 'frames_use_capture'}` (zero overhead when flag OFF)

## Byte-equal verification

Smoke test on 5 cases (XRTnTUjU + nVUnxqHLr + 9xjhi + qNvSMyU + jtYTdZm3):

| case | OFF md5 | ON md5 | byte_eq | frames | use_capture | %_capture |
|---|---|---|---|---:|---:|---:|
| XRTnTUjU (stress) | d12cc234 | d06e8ec9 | False | 3486 | 2071 | **59.4 %** |
| nVUnxqHLr (stress) | fcf8dd2e | 6c6761fe | False | 4297 | 177 | 4.1 % |
| 9xjhi (FS) | 000f44cb | 27a4f80a | False | 2187 | 119 | 5.4 % |
| qNvSMyU (FS) | 3ffb3cfe | 18a0d1da | False | 2685 | 469 | 17.5 % |
| jtYTdZm3 (normal DT) | f7859e5f | 0aed95c1 | False | 3675 | 81 | 2.2 % |

**A baseline OFF md5 matches v3.21.12 A md5** (e.g. XRTnTUjU `d12cc234`) — byte-equal preserved against the existing baseline anchor.

**XRTnTUjU 59.4 % use_capture** confirms the user's hypothesis: this stress case has `usable_linear=False` for the majority of frames (because the latch never engages cleanly), so the AEC3 fallback path activates the majority of the time.

## 12-case AECMOS — Δ vs A baseline

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | -0.138 | +0.000 |
| FS_movement | -0.007 | +0.000 |
| **DT_static** | **-0.145** | **-0.335** |
| **DT_movement** | -0.142 | **+0.298** |
| NE | ±0 | ±0 |

vs A: small FS echo regression (-0.138 on FS_static), modest DT_static deg regression (-0.335), DT_movement deg IMPROVEMENT (+0.298). FS_movement and NE unchanged.

## Per-case DT — head-to-head with v3.21.7 B (partition_summed_x2) and v3.21.12 D

| case | A_deg | B_deg | D_deg | **E_deg (v3.21.13)** | E-A | **E-B** |
|---|---:|---:|---:|---:|---:|---:|
| MYrVxVEM_DT_static (stress) | 2.869 | 1.523 | 1.520 | **2.718** | -0.151 | **+1.195** |
| XRTnTUjU_DT_static (stress) | 3.929 | 2.676 | 2.674 | **3.326** | -0.603 | **+0.650** |
| nVUnxqHLr_DT_static (stress, worst) | 3.223 | 1.748 | 1.694 | **2.663** | -0.561 | **+0.914** |
| jtYTdZm3_DT_static (normal) | 2.347 | 2.670 | 2.343 | 2.323 | -0.024 | -0.347 |
| ZJYUt0O0A_DT_mvmt | 1.991 | 3.146 | 3.126 | 2.417 | **+0.427** | -0.729 |
| wVYSGV_DT_mvmt | 2.203 | 3.192 | 3.826 | 2.497 | +0.294 | -0.695 |
| xFk7igec_DT_mvmt (stress) | 2.882 | 1.825 | 2.001 | 3.054 | **+0.173** | **+1.229** |

**Stress recovery vs B**: v3.21.13 recovers +0.65 to +1.20 deg per stress case. This is the first v3.21.x candidate to MATERIALLY reduce the Cat C downstream regression.

**Normal DT trade-off**: v3.21.13 hurts jtYTdZm3 (-0.347 vs B), ZJYUt0O0A (-0.729 vs B), wVYSGV (-0.695 vs B). These are normal-convergence cases where B's cleaner residual benefited DT_deg; v3.21.13 swaps that for Y-fallback in a few frames and gives up some of that gain.

## FS_static per-case echo

| case | A | B | D | E (v3.21.13) | E-A |
|---|---:|---:|---:|---:|---:|
| 9xjhi (canonical nores-artifact target) | 2.194 | 2.100 | 1.875 | 1.944 | -0.249 |
| qNvSMyU | 3.673 | 3.568 | 3.576 | 3.539 | -0.134 |
| xQEUtY2 | 3.346 | 3.710 | 3.697 | 3.315 | -0.031 |

FS_static echo regression on 9xjhi (-0.249) is the largest; mean -0.138. On 9xjhi, ~5.4 % of frames pre-convergence use Y (uncancelled echo present) — that fraction shows up as an echo metric hit.

## Mechanism — why this works on stress but trade-off on FS

**Stress case (XRTnTUjU, 59.4 % use_capture)**:
- `usable_linear=False` for majority of frames → output = Y × gain (suppressed mic)
- Y contains both NE speech and uncancelled echo
- SuppressionGain in `usable_linear=False` regime is conservative (gain close to 1.0 on NE-dominated bins)
- Result: NE speech preserved, echo present but suppressed → +0.65 to +0.92 deg vs B's catastrophic over-suppression

**FS_static (9xjhi, 5.4 % use_capture)**:
- `usable_linear` latches True after convergence → most frames use E (same as baseline)
- The 5.4 % pre-convergence frames now use Y (uncancelled echo) → -0.249 echo regression
- Trade-off: lose a bit of pre-convergence echo cancellation to keep stress DT preservation

**Normal DT (jtYTdZm3, 2.2 % use_capture)**:
- Mostly E path → behaves like A baseline
- The 2.2 % Y frames are early-utterance / pre-convergence → small effect

## Comparison with all v3.21.x candidates

| Patch | DT_static Δdeg vs A | FS_static Δecho vs A | Note |
|---|---:|---:|---|
| B (partition_summed_x2 v3.21.7) | -0.938 | +0.055 | Cat C BLOCKED-STRESS |
| C (raw E2 v3.21.12) | -0.061 | -0.016 | No-op neutral |
| D (B + C v3.21.12) | -1.034 | -0.022 | REJECTED — worse than B |
| **E (v3.21.13 UseLinearFilterOutput)** | **-0.335** | -0.138 | **PAUSED CANDIDATE — first material stress recovery** |

v3.21.13 (E) is the first non-no-op v3.21.x candidate that:
- materially reduces stress DT regression (E-B per case: +0.65 to +1.20)
- preserves byte-equal default-OFF
- is true AEC3 production parity (verbatim port of `echo_remover.cc:475` semantics)

## Decision per Step 5 gate

| Outcome | Decision |
|---|---|
| Improves stress DT vs B + non-blocking vs A | **PAUSED CANDIDATE** — present to user for ship decision |
| Improves stress DT vs B but new normal regression vs A | PAUSED CANDIDATE (current state — modest DT_static Δdeg -0.335 vs A) |

## What this confirms vs prior closures

1. **The v3.21.7 Cat C mechanism IS downstream of the filter** — confirmed by v3.21.12 verdict. v3.21.13 demonstrates this by intervening at the final-output selection (downstream of the linear filter), which is exactly where AEC3 has the safety valve we were missing.

2. **`usable_linear_estimate()` is the proximate control variable**. v3.21.13 uses it correctly per AEC3. The remaining DT_static regression (-0.335 vs A) comes from frames where `usable_linear=True` (latched) — those still use E path. To fully fix Cat C would require either:
   - The latch logic itself (out of v3.21.x scope per memory)
   - A different `usable_linear` definition (also state-layer)

3. **v3.21.13 IS AEC3 production parity**, not v3.22 divergence. AEC3 production runs with this branch active by default. We were missing it.

## Next-step options for user

1. **SHIP v3.21.13 as production default** — accept the modest DT_static Δdeg -0.335 / FS_static Δecho -0.138 vs A baseline in exchange for substantial stress DT recovery. Run 800-case to confirm. (Per user CLAUDE.md `[feedback_aec_target_aec2_aec3]`, target is AEC2/AEC3 absolute scores, not yesterday's baseline; the absolute deg numbers of v3.21.13 are likely competitive.)

2. **TEST COMBO** v3.21.13 + partition_summed_x2 (B+E). Trace evidence suggests this could combine B's nores improvement with E's DT-stress recovery. User's current rule says "Do not combine yet" — explicit approval needed.

3. **PAUSED SUBSTRATE** — leave v3.21.13 default-OFF, flag retained for future research. Move to #1e or RES audit.

4. **REJECT v3.21.13** — DT_static Δdeg -0.335 vs A baseline is non-trivial; some users may prefer the v3.21.6 baseline behaviour. Close as paused-substrate.

## Forbidden actions (still in force)

- Do NOT combine with partition_summed_x2 / UseRefinedOutput / state gate-3 counters / FA-AND / shadow-converged precondition — explicitly out of scope for this round
- Do NOT enable production-default until user reviews the per-case trade-off and AECMOS magnitude
- Do NOT delete code (flag default-OFF preserves byte-equal)
- Do NOT run 800-case until user authorises ship-candidate evaluation
