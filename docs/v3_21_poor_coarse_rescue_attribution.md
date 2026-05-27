# v3.21 poor_coarse rescue copy regression attribution

> **CORRECTION 2026-05-27**: this doc's `coarse_conv = 0%` reading was a
> diagnostic-script bug, not real non-convergence. The script's gate
> `e2_coa < 0.5*y2 AND y2 > 50*50*hop` missed the int16²→float² scale
> conversion AND used the REFINED-filter ratio (0.5) instead of the
> coarse strict (0.05) or relaxed (0.3) bar. **AEC3-correct re-audit**
> (`docs/v3_21_coarse_conv_definition_audit.md`) shows the real picture:
> shadow PBFDAF achieves AEC3 RELAXED bar (`r < 0.3 AND y2 > 5.96e-5`)
> on **9xjhi @ 11.2%** (M_full_delay) / **12.2%** (M_full_rescue) and
> MYrVxVEM @ 12.4% / 11.5% of frames — non-zero but well below
> refined-filter rates. AEC3 STRICT bar (`r < 0.05`) stays at 0% / 1%.
> The NO-SHIP verdict below is UNCHANGED — AECMOS scores are measured
> on actual audio output, independent of any convergence-metric
> calibration. The "shadow architecturally cannot converge" framing
> should read "shadow meets AEC3 RELAXED bar but does not catch up to
> refined when refined is genuinely better".

**Purpose**: post-FAIL sanity audit. Confirm the M_full_rescue vs M_full_delay regression on FS_static (qNvSMyUS / 9xjhi) and MYrVxVEM DT_static is the result of strict AEC3 rescue composition under the current shadow PBFDAF, not an implementation bug.

**Pair**: `M_full_delay` (Bundle A+B+C+D + delay chain, no rescue) vs `M_full_rescue` (M_full_delay + `use_aec3_poor_coarse_rescue_copy=True` including Gap C E_refined override on copy_fired hops).


## Attribution Table

| Case | bucket | metric | M_full_delay | M_full_rescue | AECMOS Δ |
|------|--------|--------|--------------|---------------|----------|
| MYrVxVEM | DT_static | deg | 1.820 | 1.389 | -0.431 |
| qNvSMyUS | FS_static | echo | 3.856 | 3.640 | -0.216 |
| 9xjhi | FS_static | echo | 2.354 | 2.242 | -0.112 |
| xFk7 | DT_mvmt | deg | 1.893 | 2.088 | +0.195 |

### MYrVxVEM  (DT_static · deg; AECMOS Δ -0.431)

| field | M_full_delay | M_full_rescue | Δ |
|------|--------------|---------------|---|
| pc_cond_fire frac | 2.5% | 8.8% | +6.3% |
| pc_copy_fire frac | 0.2% | 2.8% | +2.7% |
| E_refined_override frac | 0.0% | 2.8% | +2.8% |
| URO cond1 frac (coarse cleaner) | 23.8% | 22.6% | -1.3% |
| URO cond2 frac (refined diverged) | 23.3% | 16.5% | -6.8% |
| use_refined_frac | 65.7% | 73.1% | +7.4% |
| use_coarse_frac | 34.3% | 26.9% | -7.4% |
| usable_linear frac | 95.6% | 96.8% | +1.3% |
| coarse_conv frac | 0.0% | 0.0% | +0.0% |
| e2_refined mean (all) | 7.325e-02 | 7.256e-02 | — |
| e2_coarse mean (all) | 5.108e-02 | 5.431e-02 | — |
| e2_refined mean (uro=coarse) | 1.441e-01 | 1.745e-01 | — |
| e2_coarse mean (uro=coarse) | 6.861e-02 | 1.046e-01 | — |
| selected residual mean (uro=coarse) | 6.861e-02 | 1.046e-01 | — |
| output RMS (time-domain) | 0.0138 | 0.0137 | -0.0001 (-0.5%) |

**Routing direction (URO)**: cond1 Δ -1.3%, cond2 Δ -6.8%, use_coarse Δ -7.4%, usable_linear Δ +1.3%.

**Convergence (coarse_conv)**: 0.0% → 0.0% (shadow NLMS structural: AEC3 rescue copies W from refined, but shadow then re-adapts under the current PBFDAF gain family — coarse_conv does not improve).


### qNvSMyUS  (FS_static · echo; AECMOS Δ -0.216)

| field | M_full_delay | M_full_rescue | Δ |
|------|--------------|---------------|---|
| pc_cond_fire frac | 9.9% | 23.5% | +13.6% |
| pc_copy_fire frac | 1.0% | 9.6% | +8.6% |
| E_refined_override frac | 0.0% | 9.6% | +9.6% |
| URO cond1 frac (coarse cleaner) | 3.7% | 4.5% | +0.7% |
| URO cond2 frac (refined diverged) | 53.4% | 50.0% | -3.5% |
| use_refined_frac | 45.3% | 48.6% | +3.2% |
| use_coarse_frac | 54.7% | 51.4% | -3.2% |
| usable_linear frac | 82.5% | 82.5% | +0.0% |
| coarse_conv frac | 0.0% | 0.0% | +0.0% |
| e2_refined mean (all) | 2.381e-01 | 2.117e-01 | — |
| e2_coarse mean (all) | 1.737e-01 | 1.586e-01 | — |
| e2_refined mean (uro=coarse) | 2.849e-01 | 2.850e-01 | — |
| e2_coarse mean (uro=coarse) | 1.114e-01 | 1.672e-01 | — |
| selected residual mean (uro=coarse) | 1.114e-01 | 1.672e-01 | — |
| output RMS (time-domain) | 0.0219 | 0.0217 | -0.0002 (-0.9%) |

**Routing direction (URO)**: cond1 Δ +0.7%, cond2 Δ -3.5%, use_coarse Δ -3.2%, usable_linear Δ +0.0%.

**Convergence (coarse_conv)**: 0.0% → 0.0% (shadow NLMS structural: AEC3 rescue copies W from refined, but shadow then re-adapts under the current PBFDAF gain family — coarse_conv does not improve).


### 9xjhi  (FS_static · echo; AECMOS Δ -0.112)

| field | M_full_delay | M_full_rescue | Δ |
|------|--------------|---------------|---|
| pc_cond_fire frac | 8.4% | 11.9% | +3.5% |
| pc_copy_fire frac | 0.9% | 4.3% | +3.4% |
| E_refined_override frac | 0.0% | 4.3% | +4.3% |
| URO cond1 frac (coarse cleaner) | 41.9% | 54.8% | +12.9% |
| URO cond2 frac (refined diverged) | 6.5% | 4.1% | -2.4% |
| use_refined_frac | 53.1% | 43.4% | -9.7% |
| use_coarse_frac | 46.9% | 56.6% | +9.7% |
| usable_linear frac | 92.4% | 92.4% | +0.0% |
| coarse_conv frac | 0.0% | 0.0% | +0.0% |
| e2_refined mean (all) | 1.233e+01 | 1.260e+01 | — |
| e2_coarse mean (all) | 1.854e+01 | 1.030e+01 | — |
| e2_refined mean (uro=coarse) | 1.453e+01 | 1.740e+01 | — |
| e2_coarse mean (uro=coarse) | 7.662e+00 | 9.879e+00 | — |
| selected residual mean (uro=coarse) | 7.662e+00 | 9.879e+00 | — |
| output RMS (time-domain) | 0.1230 | 0.1220 | -0.0010 (-0.8%) |

**Routing direction (URO)**: cond1 Δ +12.9%, cond2 Δ -2.4%, use_coarse Δ +9.7%, usable_linear Δ +0.0%.

**Convergence (coarse_conv)**: 0.0% → 0.0% (shadow NLMS structural: AEC3 rescue copies W from refined, but shadow then re-adapts under the current PBFDAF gain family — coarse_conv does not improve).


### xFk7  (DT_mvmt · deg; AECMOS Δ +0.195)

| field | M_full_delay | M_full_rescue | Δ |
|------|--------------|---------------|---|
| pc_cond_fire frac | 4.3% | 16.2% | +11.9% |
| pc_copy_fire frac | 0.3% | 5.7% | +5.4% |
| E_refined_override frac | 0.0% | 5.7% | +5.7% |
| URO cond1 frac (coarse cleaner) | 16.6% | 21.5% | +4.9% |
| URO cond2 frac (refined diverged) | 31.5% | 12.3% | -19.2% |
| use_refined_frac | 59.5% | 75.2% | +15.7% |
| use_coarse_frac | 40.5% | 24.8% | -15.7% |
| usable_linear frac | 96.1% | 96.1% | +0.0% |
| coarse_conv frac | 0.0% | 0.0% | +0.0% |
| e2_refined mean (all) | 3.773e-01 | 3.746e-01 | — |
| e2_coarse mean (all) | 5.043e-01 | 2.999e-01 | — |
| e2_refined mean (uro=coarse) | 4.880e-01 | 7.517e-01 | — |
| e2_coarse mean (uro=coarse) | 2.818e-01 | 3.974e-01 | — |
| selected residual mean (uro=coarse) | 2.818e-01 | 3.974e-01 | — |
| output RMS (time-domain) | 0.0301 | 0.0307 | +0.0005 (+1.8%) |

**Routing direction (URO)**: cond1 Δ +4.9%, cond2 Δ -19.2%, use_coarse Δ -15.7%, usable_linear Δ +0.0%.

**Convergence (coarse_conv)**: 0.0% → 0.0% (shadow NLMS structural: AEC3 rescue copies W from refined, but shadow then re-adapts under the current PBFDAF gain family — coarse_conv does not improve).


## Verdict

### Sanity gates

- **Task 1 — threshold/time semantics**: PASS

    - AEC3 5 blocks × 64 samples = 320 samples = **20 ms**

    - Python `round(5*64/160) = 2` hops = 320 samples = **20 ms** — match.

    - AEC3 hangover 25 blocks × 64 = 1600 samples = **100 ms**

    - Python `round(25*64/160) = 10` hops = 1600 samples = **100 ms** — match.

    - No 64-sample-block / 160-sample-hop mixing (verified).

    - Sole quantization gap: AEC3 has 5 independent decision points per 20 ms; Python has 2 over the same window. Irreducible 10ms-hop vs 4ms-block limit; not a bug.


- **Task 2 — E_refined override correctness**: PASS

    - Single `complete_update()` call per frame in the rescue block (`if _copy_fired: override=E_refined.copy() else override=None`); mutually exclusive branches → no double-update.

    - `complete_update` clears `_deferred_update_pending=False` and `partition_idx` is always advanced exactly once per frame (matches inline behaviour).

    - Safety flush at orchestrator:2489-2492 only fires when the inner hasattr-rescue block did not run; cannot double-call in normal flow.

    - Byte-equal flag OFF: 25/25 PASS confirmed.


### Conclusion

The 12-case FAIL is **NOT** an implementation bug. Threshold time, hangover time and override correctness are all verified.


**Mechanism (consistent across all 4 cases)**: rescue copies W from refined into shadow on fire hops; the same hop's shadow update then uses E_refined (Gap C). But `coarse_conv` stays at 0% in every variant — shadow PBFDAF NLMS does not converge under the current gain family, so the copied W immediately re-diverges. The behavioural impact reaches AECMOS via the URO routing layer + `usable_linear` consumer:

  - URO `cond2 = (e2_refined > e2_coarse AND y2 < e2_refined)` falls because rescue keeps refined `e2` near coarse `e2` (e.g. xFk7 cond2 31.5%→12.3%, −19.2pp).

  - URO `cond1 = (e2_coarse < 0.9·e2_refined AND y2 > thr30 AND s2 > thr60)` rises when rescue temporarily makes coarse cleaner-looking (e.g. 9xjhi cond1 41.9%→54.8%, +12.9pp) even though coarse is still un-converged.

  - `usable_linear` ticks UP on MYrVxVEM (95.6%→96.8%) because rescue reduces the cond2-driven "shadow looks bad" signal; the suppression-gain consumer then receives an aggressive `error_psd` input that over-suppresses nearend speech in DT (AECMOS deg −0.431).


Per-case attribution:

  - **xFk7 (+0.195, DT_mvmt deg)**: WIN. Rescue + Gap C lets shadow track refined during DT movement → cond2 drops 31.5%→12.3% → use_coarse drops 40.5%→24.8% → less unnecessary suppression of nearend during movement. This is the *only* directional win and is exactly AEC3's design intent.

  - **9xjhi (−0.112, FS_static echo)**: Cat3 gap to AEC3 (1.088 dB on M_full_delay) NOT closed. Rescue raises use_coarse 46.9%→56.6% via cond1 (+12.9pp) but the coarse path is no closer to truth (e2_coarse on coarse-selected frames 7.66 → 9.88, +29%) — URO routes more often to a worse path.

  - **qNvSMyUS (−0.216, FS_static echo)**: rescue fires aggressively (9.6%); frequent W resets perturb FS-static refined convergence. e2_coarse on coarse-selected frames rises 1.11e-01 → 1.67e-01 (+50%). Output RMS actually *drops* −0.9% but the residual's spectral signature regresses on AECMOS echo.

  - **MYrVxVEM (−0.431, DT_static deg)**: catastrophic. Rescue raises `usable_linear` 95.6%→96.8% via the cond2 reduction → SuppressionGain consumes a more aggressive `error_psd` clamp → nearend speech over-suppressed in sustained DT. The −0.431 is a DT-deg over-suppression artifact, not an echo-leakage artifact.


**Verdict**: Gap C strict AEC3 poor-coarse rescue = **NO-SHIP default-OFF substrate**; no 800-case. Conditional gating / FS-only rescue / convergence-qualified rescue belong in v3.22 beyond-AEC3 optimization, not v3.21 alignment.


### Nores artifact guard

- 9xjhi nores LF improvement attributed to Bundle A in earlier verdicts **does not** mean the internal farend-singletalk nores artifact is closed.

- Internal nores artifact closure requires a separate diagnostic (cohort + listen-test). This 12-case verdict does **not** close that issue.
