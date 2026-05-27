# coarse_conv = 0% — definition audit

**Question**: why does the prior attribution show `coarse_conv = 0%` in every case × variant? Was it a metric bug or true non-convergence?

## Answer (one line)

**The 0% was a metric BUG** (wrong scale conversion); the actual shadow PBFDAF convergence rate vs AEC3-correct bars is reported below.


## The bug

Prior attribution script used:

```python

y2_thr = 50 * 50 * hop           # = 50²×160 = 400 000  ← raw int16² number

coarse_conv = (e2_coa < 0.5 * y2) & (y2 > y2_thr)  ← 0.5 is REFINED ratio

```

Both wrong:

  - **Bug 1 (fatal)**: `y2_thr = 400000` lives in AEC3 int16² scale. Our `uro_y2` is `sum(near_block**2)` in **float [-1,1]** scale — its typical magnitude is 1e-4 to 1e-1. `y2 > 400000` is **always False** → gate never fires → 0% regardless of any ratio.

  - **Bug 2**: `0.5` is the AEC3 REFINED bar; coarse uses 0.05 (strict) or 0.3 (relaxed).


## AEC3 canonical thresholds + the correct conversion for hop=160

AEC3 source: `subtractor_output_analyzer.cc:43-51`. AEC3 native: `kBlockSize = 64`, samples in int16 scale (±32768).


| predicate | AEC3 ratio | AEC3 y2 floor (int16² × kBlockSize) | Our float-scale y2 floor (hop=160) |
|-----------|------------|--------------------------------------|--------------------------------------|
| refined_converged | `e2_ref < 0.5 × y2` | `50² × 64 = 160 000` | `50² × 160 / 32768² ≈ 3.73e-4` |
| coarse_converged_STRICT | `e2_coa < 0.05 × y2` | `50² × 64 = 160 000` | `50² × 160 / 32768² ≈ 3.73e-4` |
| coarse_converged_RELAXED | `e2_coa < 0.3 × y2` | `20² × 64 = 25 600` | `20² × 160 / 32768² ≈ 5.96e-5` |

**Conversion formula** (matches the URO thr_30 / thr_60 computation at `orchestrator.py:4085-4087`):

```python

# Preserves AEC3 per-sample RMS semantic; rescales to our hop & float scale

int16_scale_sq = 32768.0 ** 2

y2_thr_strict  = (50 ** 2) * hop / int16_scale_sq   # ≈ 3.73e-4 @ hop=160

y2_thr_relaxed = (20 ** 2) * hop / int16_scale_sq   # ≈ 5.96e-5 @ hop=160

```

Semantic: AEC3 `kConvergenceThreshold = 50² × kBlockSize` enforces "average per-sample y² > 50² (int16)" over the AEC3 block window. Translating to our pipeline:

  1. **scale**: int16² → float² ⇒ divide by `32768² ≈ 1.07e9`

  2. **sample count**: AEC3 sums over `kBlockSize = 64`; we sum over `hop = 160` ⇒ multiply by `160 / 64 = 2.5` (i.e. use `hop` directly in the formula).

  3. **ratio**: scale-invariant; use AEC3 0.05/0.3/0.5 unchanged.


## 9xjhi

Gate fields:

  - **r_med** / **r_p25** = e2_coarse / y2 ratio (median / 25th percentile)

  - **STRICT** (`r<0.05 AND y2>3.73e-4`) = AEC3 `coarse_filter_converged_strict`

  - **RELAXED** (`r<0.3 AND y2>5.96e-5`) = AEC3 `coarse_filter_converged_relaxed`

  - **REFINED bar** (`r<0.5 AND y2>3.73e-4`) shown for reference (this is the REFINED bar, not coarse)

  - **y2_gate_pass_strict** = % of frames passing y2 floor only (independent of ratio)


| variant | n | y2_med | y2_gate_pass_strict | r_p25 | r_med | STRICT (0.05) | RELAXED (0.3) | REFINED bar (0.5) |
|---------|---|--------|---------------------|-------|-------|---------------|---------------|--------------------|
| M_full_delay | 2188 | 1.632e+01 | 100.0% | 0.444 | 0.782 | 0.0% | 11.2% | 30.2% |
| M_full_rescue | 2188 | 1.632e+01 | 100.0% | 0.402 | 0.671 | 0.0% | 12.2% | 35.2% |

## MYrVxVEM

Gate fields:

  - **r_med** / **r_p25** = e2_coarse / y2 ratio (median / 25th percentile)

  - **STRICT** (`r<0.05 AND y2>3.73e-4`) = AEC3 `coarse_filter_converged_strict`

  - **RELAXED** (`r<0.3 AND y2>5.96e-5`) = AEC3 `coarse_filter_converged_relaxed`

  - **REFINED bar** (`r<0.5 AND y2>3.73e-4`) shown for reference (this is the REFINED bar, not coarse)

  - **y2_gate_pass_strict** = % of frames passing y2 floor only (independent of ratio)


| variant | n | y2_med | y2_gate_pass_strict | r_p25 | r_med | STRICT (0.05) | RELAXED (0.3) | REFINED bar (0.5) |
|---------|---|--------|---------------------|-------|-------|---------------|---------------|--------------------|
| M_full_delay | 3947 | 6.035e-03 | 85.5% | 0.857 | 1.000 | 1.0% | 12.4% | 16.8% |
| M_full_rescue | 3947 | 6.035e-03 | 85.5% | 0.861 | 1.000 | 1.1% | 11.5% | 16.1% |

## Verdict

**Cause = (a) metric mis-definition** (primary).

9xjhi M_full_delay e2_coarse/y2 ratio satisfies:

  - AEC3 STRICT (r < 0.05): 0.0% of frames

  - AEC3 RELAXED (r < 0.3): 11.2% of frames

  - my prior wrong (r < 0.5): 30.2% of frames

Shadow DOES converge by AEC3 RELAXED bar; the prior "0%" reading was an artifact of the wrong ratio + wrong y2 floor.


### Implications for the §G4 / rescue verdict

- The "shadow never converges" framing in the rescue attribution doc was based on a mis-calibrated metric.

- Re-state: shadow DOES achieve AEC3 RELAXED convergence but at a lower rate than refined. The rescue mechanism still has the same AECMOS trade-off (xFk7 +0.195 / MYrVxVEM −0.431 / etc.) because the URO routing + usable_linear consumer pathway is the dominant driver, not the absolute convergence bar.

- The 9xjhi Cat3 architectural-ceiling framing still holds: rescue copies do not improve the COARSE path quality enough to close the 1.088 dB extra echo gap vs AEC3 reference. The mechanism description in attribution doc §I.3 should note "shadow does meet AEC3 RELAXED bar but does not catch up to refined when refined is genuinely better" rather than "shadow never converges at all".


Note: this audit does not change the NO-SHIP verdict for `use_aec3_poor_coarse_rescue_copy`. The 12-case AECMOS Pareto FAIL (MYrVxVEM −0.431 / qNvSMyUS −0.216) is independent of how we define coarse_conv — the AECMOS scores were measured on actual audio output.
