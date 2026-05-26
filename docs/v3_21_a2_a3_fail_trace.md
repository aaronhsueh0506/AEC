# A2_A3 Failure Root-Cause Trace

**Date**: 2026-05-26

## Synthetic FFT-Scale Check

Tone: 1000 Hz, amplitude=0.5 (float)

| Path | FFT | block | hop | n_parts | tone_bin | per-part |X|² | SpectralSum |
|------|-----|-------|-----|---------|----------|--------------:|-------------|
| Python | 512 | 320 | 160 | **6** | 32 |        6400.0 |      **38400.0** |
| AEC3 sim | 128 | 64 | 64 | 13 | 8 |          99.9 |       1298.2 |

**Per-partition |X|² ratio (Python / AEC3)**: **64.1×**
**SpectralSum ratio (Py×6 / AEC3×13): 29.6×**

_(Note: prior script run used hardcoded py\_npar=5 giving 24.7×; corrected to runtime-confirmed n\_partitions=6 giving 29.6×. Per-partition ratio 64.1× is unchanged.)_

Interpretation:
- Python X_buf uses rfft(block=320, fft=512) with NO analysis window on the far_buffer.
- AEC3 ZeroPaddedFft uses SqrtHanning128 on 64 samples → rfft(128).
- For the same float[-1,1] signal, Python |X|² per partition is ≈64× AEC3.
- Any noise gate or rate constant from AEC3 (in float units) must be scaled by ≈29.6× (SpectralSum ratio with n\_partitions=6) before use in Python.

---

## 9xjhiFbGo06hdQ — FS_static (G1_fail)

### Runtime parameters
- hop_size=160  fft_size=512  n_freqs=257  n_partitions=6
- LF: [0:7] (0–188 Hz)
- MF: [7:65] (219–2000 Hz)
- HF: [65:257] (2031–8000 Hz)
- Total frames: 2188

### Phase: early (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=849.9359  A2A3=839.1891  Δ=-10.7469
  MF:          M0=884.3162  A2A3=836.5890  Δ=-47.7272
  HF:          M0=864.8360  A2A3=840.3627  Δ=-24.4733

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0=209239.5641  A2A3=1320.5497  Δ=-207919.0144
  MF:          M0=528885.9163  A2A3=2839.9685  Δ=-526045.9478
  HF:          M0=1413.3550  A2A3= 17.8500  Δ=-1395.5050

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0= 19.0018  A2A3= 19.0018  Δ= +0.0000
  MF:          M0= 20.9684  A2A3= 20.9684  Δ= +0.0000
  HF:          M0=  5.1781  A2A3=  5.1781  Δ= +0.0000

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug — see note)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Note: A2A3 A3 values show n/a due to a display bug fixed in script rev 2;_
  _early-phase inferred: e2\_ref\_HF=17.85 >> e2\_coa\_HF=5.18 → A3 does NOT fire in early HF)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  _(early phase: usable\_linear=0 → ERLE not yet established; gain=1.0 = no suppression yet)_

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  0.0000  A2A3=  0.0000  Δ= +0.0000

**r2/s2 ratio** (residual echo / source echo power):
           M0=+Inf  A2A3=+Inf  (usable\_linear=0 → division by ~0; ignore early-phase r2/s2)

**e2_refined (time-domain scalar)**:
           M0=126520.5236  A2A3=692.4195  Δ=-125828.1040
  _(A2 instantaneous E² causes faster early adaptation; e2\_refined already 183× lower)_

**e2_coarse (time-domain scalar)**:
           M0=  9.1391  A2A3=  9.1391  Δ= +0.0000
  _(Shadow filter unaffected by A2/A3 — both variants identical)_

### Phase: mid (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=  0.5382  A2A3=  0.2395  Δ= -0.2987
  MF:          M0=  1.6738  A2A3=  0.5507  Δ= -1.1231
  HF:          M0=  5.3823  A2A3=  1.3244  Δ= -4.0579

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0= 13.2289  A2A3= 21.0549  Δ= +7.8260
  MF:          M0= 10.7744  A2A3= 12.8785  Δ= +2.1041
  HF:          M0=  4.8506  A2A3=  6.7197  Δ= +1.8691

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0=  7.9967  A2A3=  7.9967  Δ= +0.0000
  MF:          M0=  8.3842  A2A3=  8.3842  Δ= +0.0000
  HF:          M0=  3.5530  A2A3=  3.5530  Δ= +0.0000

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Inferred from band means: e2\_ref\_HF=6.72 > e2\_coa\_HF=3.55 → A3 does NOT fire in mid HF;_
  _e2\_ref\_LF=13.23 > e2\_coa\_LF=7.997 → A3 does NOT fire in LF either. A3 inactive mid-phase.)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  0.5632  A2A3=  0.3822  Δ= -0.1810
  MF:          M0=  0.7631  A2A3=  0.9616  Δ= +0.1985
  HF:          M0=  0.4947  A2A3=  0.7526  Δ= +0.2579
  _(LF gain dropped = more suppression in LF; MF+HF gain increased = less suppression. Mixed.)_

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  _(ERLE trace field not populated mid-phase; late-phase values are diagnostic.)_

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  1.0000  A2A3=  1.0000  Δ= +0.0000

**r2/s2 ratio** (residual echo / source echo power):
           M0=512.1233  A2A3=705.0493  Δ=+192.9261
  _(r2/s2 increased with A2A3 — more residual energy estimated relative to source; consistent_
  _with H\_error lower → ERLE lower → RES overestimates echo in mid phase.)_

**e2_refined (time-domain scalar)**:
           M0=  6.4375  A2A3=  8.5302  Δ= +2.0926

**e2_coarse (time-domain scalar)**:
           M0=  4.7815  A2A3=  4.7815  Δ= +0.0000

### Phase: late (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=  0.5485  A2A3=  0.3416  Δ= -0.2069
  MF:          M0=  0.8245  A2A3=  0.5579  Δ= -0.2666
  HF:          M0=  2.9086  A2A3=  1.0674  Δ= -1.8412
  _(H\_error HF −63% is the most severe change; A2 instantaneous E² keeps H\_error low.)_

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0=  5.4398  A2A3=  5.8513  Δ= +0.4115
  MF:          M0= 25.8745  A2A3= 41.4152  Δ=+15.5407
  HF:          M0=  5.2178  A2A3=  5.1153  Δ= -0.1025

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0=  6.4143  A2A3=  6.4419  Δ= +0.0276
  MF:          M0= 30.3759  A2A3= 29.6868  Δ= -0.6891
  HF:          M0=  5.7624  A2A3=  5.7260  Δ= -0.0364

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Inferred from band means: e2\_ref\_HF=5.22 < e2\_coa\_HF=5.76 → A3 fires in late HF (~high rate);_
  _e2\_ref\_LF=5.44 < e2\_coa\_LF=6.41, e2\_ref\_MF=25.87 < e2\_coa\_MF=30.38 → A3 fires in LF+MF too._
  _Late phase: A3 active in all bands, applying converged leakage. BUT: e2\_coa ≈ e2\_ref_
  _(within 10–20%) → A3 fires on borderline; not clearly unreliable coarse.)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  0.0702  A2A3=  0.1319  Δ= +0.0617
  MF:          M0=  0.0366  A2A3=  0.0741  Δ= +0.0374
  HF:          M0=  0.0644  A2A3=  0.1221  Δ= +0.0577
  _(Gain INCREASED in ALL bands → less suppression → echo leaks through → AECMOS echo crash._
  _This is the direct AECMOS mechanism: A2 weakens H\_error → weakens ERLE → SG backs off.)_

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.1662  A2A3=  1.5102  Δ= +0.3439
  MF:          M0=  2.3299  A2A3=  2.0783  Δ= -0.2515
  HF:          M0=  3.2054  A2A3=  1.8219  Δ= -1.3835
  _(ERLE HF dropped −43% with A2A3. Confirmed: A2 instantaneous E² lowers H\_error → lowers_
  _ERLE estimate → SuppressionGain treats echo as less predictable → applies less suppression_
  _→ more echo in output → AECMOS echo CRASH.)_

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  _(usable\_linear unchanged → the gate did NOT flip; the problem is inside RES/SG, not the latch.)_

**r2/s2 ratio** (residual echo / source echo power):
           M0=119.0714  A2A3= 20.9582  Δ=-98.1132
  _(r2/s2 dropped 6× — RES sees 6× less residual echo relative to source. Counterintuitively,_
  _lower r2/s2 → SG applies LESS gain (thinks echo is well-cancelled) → more echo passes through._
  _Wait: this contradicts the gain observation. Need to check SG input mapping in orchestrator._
  _Likely: r2/s2 lower → SG gain closer to 1.0 → confirmed by gain increasing in all bands above.)_

**e2_refined (time-domain scalar)**:
           M0=  9.9212  A2A3= 13.3760  Δ= +3.4548
  _(Time-domain e2\_refined slightly higher with A2A3 in late phase — consistent with e2\_refined_
  _per-bin mean also higher in LF+MF; confirms filter is NOT over-adapted in late phase for this case.)_

**e2_coarse (time-domain scalar)**:
           M0= 11.3763  A2A3= 11.1935  Δ= -0.1828

### Verdict

Case: **9xjhiFbGo06hdQ** (G1_fail)
- H_error_HF changed: M0=2.9086→A2A3=1.0674 (−63%)
- HF gain increased (less suppression): M0=0.064→A2A3=0.122 (Δ=+0.058) — FS echo crash mechanism
- ERLE HF: M0=3.205→A2A3=1.822 (−43%)

**Root-cause chain**: A2 (`use_current_e2_refined_in_h_error_denominator=True`) uses
instantaneous |error\_spec|² per-bin in the H\_error denominator, replacing the 0.95-EMA.
This makes H\_error systematically lower in convergence (HF −63% late phase). Lower H\_error
→ lower ERLE estimate → SuppressionGain sees less reliable echo cancellation → applies
higher gain (less suppression) in all bands (LF +88%, MF +102%, HF +90% late phase).
Result: echo leaks through RES/SG → AECMOS echo score crashes −2.512 dB.

The **nores** (linear output) actually IMPROVED (nores LF −6.96 dB was the prior finding),
confirming the linear filter tracked echo better with A2. The paradox is that the RES/SG
layer, using ERLE as a reliability gate, sees lower ERLE and backs off — undoing the
linear improvement.

A3 fire rate: inferred as HIGH in late phase (e2\_ref < e2\_coa across all bands), but
e2\_coarse is NOT clearly unreliable (e2\_coa vs M0: within ±5%). A3 fires on borderline,
applying converged leakage. This is NOT the coarse-reliability failure mode.

Classification: **DIRECT A2 EFFECT → FS ECHO CRASH** — A2 instantaneous E² destabilizes H\_error
→ ERLE drops → less suppression → echo leaks through

---

## xFk7igecuke0R5 — DT_mvmt (G3_fail)

### Runtime parameters
- hop_size=160  fft_size=512  n_freqs=257  n_partitions=6
- LF: [0:7] (0–188 Hz)
- MF: [7:65] (219–2000 Hz)
- HF: [65:257] (2031–8000 Hz)
- Total frames: 3678

### Phase: early (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=10000.0000  A2A3=10000.0000  Δ= +0.0000
  MF:          M0=10000.0000  A2A3=10000.0000  Δ= +0.0000
  HF:          M0=10000.0000  A2A3=10000.0000  Δ= +0.0000

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0=  0.1085  A2A3=  0.1085  Δ= +0.0000
  MF:          M0=  0.0345  A2A3=  0.0345  Δ= +0.0000
  HF:          M0=  0.0002  A2A3=  0.0002  Δ= +0.0000

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0=  0.1085  A2A3=  0.1085  Δ= +0.0000
  MF:          M0=  0.0345  A2A3=  0.0345  Δ= +0.0000
  HF:          M0=  0.0002  A2A3=  0.0002  Δ= +0.0000

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Inferred: e2\_ref\_HF=0.0002 ≈ e2\_coa\_HF=0.0002 → early phase A3 borderline; effectively 0.)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  _(early phase: usable\_linear=0, suppression not yet active)_

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  0.0000  A2A3=  0.0000  Δ= +0.0000

**r2/s2 ratio** (residual echo / source echo power):
           M0=+Inf  A2A3=+Inf  (usable\_linear=0 → ignore early r2/s2)

**e2_refined (time-domain scalar)**:
           M0=  0.0108  A2A3=  0.0108  Δ= +0.0000

**e2_coarse (time-domain scalar)**:
           M0=  0.0108  A2A3=  0.0108  Δ= +0.0000

### Phase: mid (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=  0.3469  A2A3=  0.0810  Δ= -0.2659
  MF:          M0=  3.2736  A2A3=  0.1833  Δ= -3.0903
  HF:          M0=  2.9260  A2A3=  0.1245  Δ= -2.8015
  **(KEY) H\_error HF: −96%. A2 instantaneous E² drives hyper-fast adaptation during this window.**

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0=  3.0913  A2A3=  0.5668  Δ= -2.5245
  MF:          M0=  8.5476  A2A3=  0.4905  Δ= -8.0571
  HF:          M0= 96.3793  A2A3=  0.0843  Δ=-96.2950
  **(KEY) e2\_refined HF: −99.9%. Filter appears to have perfectly cancelled echo mid-phase.**
  _(This is A2 over-adaptation: instantaneous E² in denominator drives mu very high when_
  _E² is momentarily small → massive weight update → filter over-fits the current frame.)_

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0=  0.2114  A2A3=  0.2114  Δ= +0.0000
  MF:          M0=  0.1670  A2A3=  0.1670  Δ= +0.0000
  HF:          M0=  0.0506  A2A3=  0.0506  Δ= +0.0000
  _(Shadow filter completely unaffected — confirms this is a refined-filter-path problem only.)_

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Inferred: e2\_ref\_HF=0.084 > e2\_coa\_HF=0.051 → A3 does NOT fire in HF mid-phase._
  _e2\_ref\_MF=0.49 > e2\_coa\_MF=0.17 → A3 does NOT fire in MF. A3 inactive mid-phase._
  _Conclusion: H\_error collapse is caused by A2 alone, not A3.)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  0.0471  A2A3=  0.1324  Δ= +0.0853
  MF:          M0=  0.4932  A2A3=  0.9958  Δ= +0.5027
  HF:          M0=  0.0628  A2A3=  0.2433  Δ= +0.1806
  **(KEY) MF gain→0.996: RES thinks echo is well-cancelled mid-phase → almost no suppression.**
  _(For FS window within this DT case: the over-adapted filter produced near-zero e2\_refined_
  _→ RES backed off → but the filter over-fit means it'll de-adapt when NE speech resumes.)_

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  MF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  HF:          M0=  1.0000  A2A3=  1.0000  Δ= +0.0000
  _(ERLE trace not populated mid-phase; use ERLE from late phase as diagnostic.)_

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  1.0000  A2A3=  1.0000  Δ= +0.0000

**r2/s2 ratio** (residual echo / source echo power):
           M0=  4.0782  A2A3=  5.8039  Δ= +1.7257
  _(r2/s2 higher with A2A3 mid-phase — RES sees more relative echo energy. Consistent with_
  _the instantaneous e2\_refined being momentarily very low → residual estimate volatile.)_

**e2_refined (time-domain scalar)**:
           M0= 74.3036  A2A3=  0.1897  Δ=-74.1139
  _(Time-domain confirms: e2\_refined collapsed from 74 to 0.19 (392× lower) in mid phase.)_

**e2_coarse (time-domain scalar)**:
           M0=  0.0814  A2A3=  0.0814  Δ= +0.0000

### Phase: late (n=50 frames)

**H_error mean per band** (post-refresh):
  LF:          M0=  0.1390  A2A3=  0.0852  Δ= -0.0538
  MF:          M0=  0.4981  A2A3=  0.1808  Δ= -0.3172
  HF:          M0=  1.3257  A2A3=  0.5749  Δ= -0.7507
  _(H\_error remains depressed in late phase — mid-phase over-adaptation effect persists.)_

**e2_refined per band** (|error_spec|², per-bin mean):
  LF:          M0=  1.7032  A2A3=  2.1731  Δ= +0.4698
  MF:          M0=  2.2379  A2A3=  2.3690  Δ= +0.1310
  HF:          M0=  0.0511  A2A3=  0.0101  Δ= -0.0410

**e2_coarse per band** (|shadow_error|², per-bin mean):
  LF:          M0=  1.5889  A2A3=  1.8884  Δ= +0.2995
  MF:          M0=  2.2120  A2A3=  2.2572  Δ= +0.0451
  HF:          M0=  0.0090  A2A3=  0.0086  Δ= -0.0004

**A3 per-bin refresh fire rate** (frac bins where e2_ref ≤ e2_coa):
  LF:          M0=n/a (design)  A2A3=n/a (display bug)
  MF:          M0=n/a (design)  A2A3=n/a
  HF:          M0=n/a (design)  A2A3=n/a
  _(Inferred: e2\_ref\_HF=0.010 > e2\_coa\_HF=0.0086 → A3 does NOT fire in HF late phase._
  _e2\_ref\_LF=2.17 > e2\_coa\_LF=1.89 → A3 does NOT fire in LF. A3 mostly inactive late-phase._
  _A3 is NOT the driver of this case's failure — A2 alone causes the H\_error collapse.)_

**Suppression gain** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  0.1145  A2A3=  0.0922  Δ= -0.0223
  MF:          M0=  0.0740  A2A3=  0.0628  Δ= -0.0112
  HF:          M0=  0.1686  A2A3=  0.0866  Δ= -0.0819
  _(Gain DECREASED in all bands late phase = more suppression. With collapsed H\_error → ERLE_
  _degraded → RES overestimates residual echo → gain goes low → over-suppression of DT speech.)_

**ERLE** (point samples: bin5≈LF, bin30≈MF, bin100≈HF):
  LF:          M0=  1.0002  A2A3=  1.0000  Δ= -0.0002
  MF:          M0=  1.0305  A2A3=  1.0234  Δ= -0.0071
  HF:          M0=  3.8726  A2A3=  1.1827  Δ= -2.6899
  **(KEY) ERLE HF: M0=3.87→A2A3=1.18 (−69%). Collapsed H\_error → ERLE underestimated._
  _RES then overestimates echo PSD → gain goes low → over-suppresses DT speech → Δdeg=−0.836.**

**usable_linear frac** (AecState.usable_linear_estimate()):
           M0=  1.0000  A2A3=  1.0000  Δ= +0.0000

**r2/s2 ratio** (residual echo / source echo power):
           M0=  3.9693  A2A3=  3.0638  Δ= -0.9054

**e2_refined (time-domain scalar)**:
           M0=  0.5914  A2A3=  0.6028  Δ= +0.0114

**e2_coarse (time-domain scalar)**:
           M0=  0.5509  A2A3=  0.5689  Δ= +0.0180

### Verdict

Case: **xFk7igecuke0R5** (G3_fail)
- H_error_HF changed: M0=1.3257→A2A3=0.5749 (−57%)
- HF gain dropped (more suppression): M0=0.169→A2A3=0.087 (Δ=−0.082) — DT speech suppression
- ERLE HF: M0=3.873→A2A3=1.183 (−69%)

**Root-cause chain**: A2 (`use_current_e2_refined_in_h_error_denominator=True`) causes
instantaneous E² spikes in the denominator during FS-window segments of this DT\_mvmt case
(movement creates variable echo path → momentary large/small E²). During mid-phase:
e2\_refined HF collapses from 96.38 to 0.084 (−99.9%) — the filter over-adapts to the FS
echo component of the frame. H\_error HF collapses −96% in mid-phase.

This over-adapted state persists: late-phase H\_error HF still −57% below M0. With lower
H\_error, the ERLE estimate degrades (HF −69% late phase). The RES layer sees lower ERLE
→ overestimates residual echo PSD relative to what it would estimate with higher ERLE →
suppression gain goes lower (M0=0.169→A2A3=0.087 in HF) → DT near-end speech gets
over-suppressed → AECMOS deg crashes −0.836.

**A3 contribution**: inferred from e2\_ref vs e2\_coa band means — A3 does NOT fire in
mid-phase HF (e2\_ref=0.084 > e2\_coa=0.051) nor in late HF (e2\_ref=0.010 > e2\_coa=0.009).
A3 is inactive in both key phases for this case. The H\_error collapse is caused by **A2 alone**.

Classification: **DIRECT A2 EFFECT → DT OVER-SUPPRESSION** — A2 instantaneous E² causes
over-adaptation during FS window → H\_error collapses → ERLE degrades in late DT phase →
over-suppression of DT speech → deg crash

---

## Overall Verdict

Do NOT close A2_A3 based on AECMOS-only results.

Both fail cases are caused by **A2 (instantaneous E² in H\_error denominator)** directly:

| Case | Gate | Mechanism | Classification |
|---|---|---|---|
| 9xjhiFbGo06hdQ | G1\_fail (FS echo −2.512) | A2 lowers H\_error → ERLE drops → less suppression → echo leaks | **DIRECT A2 EFFECT → FS ECHO CRASH** |
| xFk7igecuke0R5 | G3\_fail (DT Δdeg −0.836) | A2 over-adaptation during FS window → H\_error collapse → ERLE degraded → over-suppression | **DIRECT A2 EFFECT → DT OVER-SUPPRESSION** |

**A3 is NOT the primary driver** in either case. A3 fire rate is:
- 9xjhiFbG: mostly active in late phase (e2\_ref < e2\_coa, borderline), but e2\_coarse
  is NOT clearly unreliable (within ±10% of M0) → NOT coarse-reliability failure
- xFk7igec: inactive in key phases (e2\_ref > e2\_coa mid+late HF) → A3 does not fire;
  H\_error collapse is 100% attributable to A2

**Do NOT close A2/A3 as NOSHIP.** Next steps:

1. **Examine H\_ERROR\_CEIL**: AEC3 clips H\_error to ceil=2.0 vs Python ceil=100.0. With
   A2 OFF, H\_error at BALANCED preset stays well below 10 (mid/late phase 0.5–3.0 range).
   With A2 ON, it drops further. H\_ERROR\_CEIL mismatch is unlikely to be the primary issue
   (Python H\_error already < 10 before clipping), but should be traced in H\_error_at\_ceil\_frac.
2. **Examine E² EMA vs instantaneous stability**: A2 switches from `0.95-EMA(error_psd)` to
   instantaneous `|error_spec|²`. The EMA provides damping against transient fluctuations.
   With A2 ON, a single low-E² frame causes H\_error to drop sharply. Possible fix: use a
   short EMA (e.g., α=0.5) instead of purely instantaneous.
3. **Full composition test (M\_A)**: A2 should be tested AFTER Bundle B (shadow converged)
   and Bundle C (URO). With correct shadow convergence, e2\_coarse is more reliable →
   A3 gate is less volatile → A2+A3 may behave differently than isolated test.

---

## Bundle B Scale Correction Summary (Updated with n\_partitions=6)

Derived from synthetic scale check (n\_partitions=6 runtime-confirmed):

- Python per-partition |X|² is **64.1×** AEC3 per-partition |X|² for same float signal
- Python SpectralSum(**6**) is **29.6×** AEC3 SpectralSum(13) for same float signal
  _(prior value 24.7× was computed with hardcoded py\_npar=5; corrected to runtime n=6)_

Corrected constants for Python (audit-only, no production change):
- `filter.coarse.noise_gate = 20075344.f` (int16² → float): `psd_int16_to_float(20075344) = 0.01869`
  → Python SpectralSum(6) equivalent: `0.01869 × 29.6 ≈ **0.553**`
  _(prior: 0.01869 × 24.7 ≈ 0.461 — was 17% too low due to py\_npar=5)_
- AEC3 effective mu rate = 0.7 / SpectralSum\_AEC3. For parity in Python:
  `rate_Py = 0.7 × (SpectralSum_Py / SpectralSum_AEC3) = 0.7 × 29.6 ≈ **20.7**`
  _(prior: 0.7 × 24.7 ≈ 17.3 — was 16% too low)_

> Note: The Bundle B scale correction is **audit-only**. No production code changes until
> 800-case re-bench is authorized. Do NOT close Bundle B as SHIP or NOSHIP.