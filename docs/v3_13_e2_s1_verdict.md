# E2.S1 verdict — delay tracking audit on listen 8-case

**Date**: 2026-05-13
**Branch**: `feature/v3.11-route-a` (post-substrate merge `60d91f5`)
**Predecessors**: [v3_12_s6_s11_stage1_locked.md](v3_12_s6_s11_stage1_locked.md)
**Status**: **E2.S1 closed — delay-failure bimodal pattern identified**

## TL;DR

Two distinct failure modes split the worst-FS 8-case pool:

**Group A — Delay-broken (50%, cases 01 / 04 / 05 / 08)**:
- GCC-PHAT pre-alignment confidence LOW (3.6–8.7) → falls below
  `_estimate_delay` confidence=5.0 trust threshold
- Online tracker reports HUGE residual delay after pre-alignment:
  2647 → 10467 samples (165–654 ms)
- True total delay = pre-alignment + residual ≈ 3000–10800 samples
  (188–680 ms) — **7–13× over filter_length=832 (52 ms)**

**Group B — Delay-OK, leak from saturation / NL / E1 (50%, cases 02 / 03 / 06 / 07)**:
- GCC-PHAT confidence HIGH (19.5–28.5) → trusted
- Online residual delay = 0 (no drift detected)
- All listen-reported delay numbers match GCC-PHAT to within ~100
  samples (cases 02 / 06 / 07 perfect match)
- Listen issues attributable to E5 (clipping) / E4 (NL distortion) /
  E1 (mic quality), not delay

## Per-case audit table

| # | Stem | Scenario | GCC-PHAT (samp / ms / conf) | Online residual (samp) | TOTAL delay (samp) | Group | Listen note |
|---|---|---|---|---:|---:|---|---|
| 01 | 7GTxyTks | FS_static | 415 / 25.9 ms / **3.62** | **10467** | **10882** | **A** | delay ~10000 + clipping |
| 02 | IrQvqOTC | FS_static | 1475 / 92.2 ms / 28.47 | 0 | 1475 | B | delay ~1200 + clipping + radio |
| 03 | pcb1Nh0Z | FS_static | 2523 / 157.7 ms / 19.53 | 0 | 2523 | B | gain變化, mic品質差 (E1) |
| 04 | S22FCqKD | FS_static | 0 / 0.0 ms / **8.74** | **9855** | **9855** | **A** | delay ~9800 |
| 05 | hVqUmGvI | FS_static | 3006 / 187.9 ms / **4.76** | **2647** | **5653** | **A** | delay ~6000 |
| 06 | 5bJUo1K3 | FS_movement | 1701 / 106.3 ms / 25.07 | 0 | 1701 | B | delay ~1800 + clipping |
| 07 | IrQvqOTC | FS_movement | 644 / 40.2 ms / 21.15 | 0 | 644 | B | delay ~640 + 嚴重 NL |
| 08 | S22FCqKD | FS_movement | 3056 / 191.0 ms / **4.11** | **7634** | **10690** | **A** | delay ~10000 + occasional sat |

PAR cutoff: `_estimate_delay` confidence threshold = 5.0. Cases 01 /
05 / 08 fall below → bench fallback to non-PHAT cross-correlation,
but the fallback ALSO produces wrong initial estimate. Case 04 sits
just above the threshold (8.74) but PHAT returns 0 samples (no peak
distinguishable) — also broken.

## Mechanism diagnosis

### Group A (delay-broken) root cause

GCC-PHAT (and its non-PHAT fallback) operates on linear cross-
correlation between mic and lpb. When **lpb is clipped / speaker-
driven into non-linearity** (typical for these cases per listen),
mic ≠ linear function of lpb. The cross-correlation peak is weak
and unstable, producing low PAR + wrong delay estimate.

Then the AEC's online `DelayEstimator` (also GCC-PHAT-based) can
re-detect the residual delay using its segment-accumulated cross-
spectrum (mean PAR 31–73 for these cases — solid online estimates).
**But the AEC has no mechanism to re-align ref based on the online
estimate** — the online value feeds only RES gating (`delay_reliable`
check at line 5690), not lpb buffer alignment.

### Group B (delay-OK)

GCC-PHAT pre-alignment correctly identifies delay (high PAR). Online
tracker correctly reports no further drift. AEC processing is on
clean-aligned ref. The remaining FS leak comes from elsewhere
(saturation / NL / E1), not delay.

## Clip detection note

Multi-threshold mic clipping check across 8 cases:
- `|x| > 0.95` rate: 0.00% on all cases
- `|x| > 0.99` rate: 0.00% on all (case 08 = 0.01%)

User's "+1/-1 平頂" listen descriptions likely refer to **perceived
saturation distortion** rather than literal full-scale clip rate.
Mic peaks are near 0 dBFS (cases 01/04/07/08 = −0.00 dBFS) but the
percentage of samples at full scale is < 0.02%. The audible
"clipping" sound likely comes from speaker-side limiting / NL
distortion that gets recorded into mic with audible artifacts but
not at literal-clip threshold.

This refines the listen findings:
- "Clipping" symptom ≈ NL distortion + brief full-scale moments
- True clip-rate is low; bulk of distortion is upstream (speaker / acoustic path)

## Implications for E2.S2 (filter_length scan)

If pre-alignment were FIXED for Group A (4 cases), the residual
delays in those cases would shrink dramatically — possibly into
filter coverage. So **filter_length scan alone won't fix Group A
unless we also fix the pre-alignment**.

For E2.S2, two paths:

### Path 1 — Brute-force fl extension
Scan fl = 832 / 2080 / 4160 / 6400 / 12288.
- fl=832: covers nothing in Group A (residuals 2647–10467)
- fl=2080 (130 ms = 2080 samp): covers nothing
- fl=4160 (260 ms): covers case 05 residual (2647), partial case 08 (7634)
- fl=6400 (400 ms): covers cases 05 + (partial) 08; misses 01 (10467), 04 (9855)
- fl=12288 (768 ms): covers ALL Group A
- Tradeoff: fl=12288 is 15× current; FFT cost 4× per frame

### Path 2 — Pre-alignment improvement
Insert a more robust delay estimator before AEC: e.g., 2-pass
envelope-based GCC-PHAT (energy envelope cross-correlation is
robust to NL distortion), then sub-sample refinement via PHAT.
- If pre-alignment finds true delay, online tracker reports
  residual ≈ 0, filter_length=832 sufficient
- C-port: a new pre-alignment module, smaller memory cost than fl
  extension

**Recommendation for E2.S2**: Run **both** paths in parallel:
- Path 1 fl scan (Python only) — gives dataset characterization
- Path 2 pre-alignment improvement (Python prototype) — likely the
  C-deployable fix

## Critical files reference

- `python/aec.py:845` — `DelayEstimator` class
- `python/aec.py:5650` — online delay accumulate / new_delay capture
- `python/aec.py:5690` — `delay_reliable` consumer (RES gate, not ref align)
- `python/eval_aec_challenge.py:_estimate_delay` — bench pre-alignment GCC-PHAT
- `tools/research/e2_s1_delay_audit.py` — this audit harness
- `results/v3_13_e2_s1_audit/` — per-case JSON + summary.md

## Sources

- v3.12 worst-FS listen findings (`docs/v3_12_s6_s11_stage1_locked.md`)
- DelayEstimator implementation (`python/aec.py:845-1042`)
- Switchboard AEC3 continuous delay (referenced in hazy-lynx plan E2 arc)
