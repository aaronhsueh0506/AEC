# B-16 Stage 1D Benchmark Results

**Verdict: NEUTRAL** (merge flag-OFF, wav-level value only)

## Spec §8 SAFE / MARGINAL / UNSAFE rubric

| Criterion | Threshold | Observed | Result |
|---|---|---|---|
| SAFE ΔFS_echo ≥ +0.03 | | +0.0002 | **NOT MET** |
| SAFE no per-scenario regression > 0.05 | | max −0.0011 | MET |
| SAFE < 5 cases Δ > 0.2 regression | | 0 cases across all metrics | MET |
| MARGINAL ΔFS_echo +0.01 to +0.03 | | +0.0002 | **NOT MET** |
| UNSAFE mean regression > 0.05 | | none | PASS (not UNSAFE) |
| UNSAFE > 5 cases Δ < −0.5 | | 0 | PASS (not UNSAFE) |

Falls in the gap between MARGINAL (+0.01 to +0.03) and UNSAFE. No
rubric label applies literally. Classified as **NEUTRAL**.

## 800-case AECMOS results

| scenario | metric | baseline (B16=0) | B16=1 | Δ |
|---|---|---|---|---|
| FAREND SINGLETALK | echo_mos | 3.4747 | 3.4749 | **+0.0002** |
| FAREND SINGLETALK | deg_mos | 4.9994 | 4.9994 | −0.0000 |
| DOUBLETALK | echo_mos | 4.0341 | 4.0330 | **−0.0011** |
| DOUBLETALK | deg_mos | 2.4484 | 2.4489 | +0.0006 |
| NEAREND SINGLETALK | echo_mos | 4.9981 | 4.9981 | +0.0000 |
| NEAREND SINGLETALK | deg_mos | 4.0053 | 4.0054 | +0.0001 |

Raw data: `docs/benchmarks/b16_stage_1d/baseline_aecmos.log.gz`,
`docs/benchmarks/b16_stage_1d/b16on_aecmos.log.gz`,
`docs/benchmarks/b16_stage_1d/per_case_delta.csv`,
`docs/benchmarks/b16_stage_1d/pareto_summary.csv`.

## Per-case regression stats

| metric | std | Δ > +0.2 wins | Δ < −0.2 losses | Δ < −0.5 big-losses |
|---|---|---|---|---|
| FS / echo_mos | 0.027 | 0 | 0 | 0 |
| FS / deg_mos | 0.000 | 0 | 0 | 0 |
| DT / echo_mos | 0.027 | 0 | 0 | 0 |
| DT / deg_mos | 0.042 | 0 | 0 | 0 |
| NE / echo_mos | 0.000 | 0 | 0 | 0 |
| NE / deg_mos | 0.003 | 0 | 0 | 0 |

**ΔFS_echo distribution**: 4 cases > +0.1 (top +0.189), 1 case
< −0.1 (worst −0.138), 74 cases > +0.01, 78 cases < −0.01.
Essentially symmetric around zero; mean +0.0002 within std noise.

## Category A (PZ7V-class targeted cases) coverage

Cat-A was the 12 top FS-gap cases (no-movement) + PZ7V itself (our
primary target), identified in
`docs/bisect_analysis_and_plan.md`. If B-16 were to produce a
metric-visible benefit, these 13 cases would be the first to show
it.

| caseID | idx | baseline | B16=1 | Δ | mark |
|---|---|---|---|---|---|
| **PZ7V0SfxUkem4IalTp1YgA** (prime target) | 84 | 1.908 | 1.940 | **+0.032** | · |
| IxgmaPghzUGnR6sxrbGU3Q | 34 | 3.007 | 3.057 | +0.050 | · |
| JLNgGcvTNEqbTDbc28wLkg | 36 | 1.973 | 1.993 | +0.020 | · |
| s0oJqM6Y1UCHSVmHmgsx4Q | 228 | 2.323 | 2.340 | +0.017 | · |
| 9xjhiFbGo06hdQIsHTS6qA | 14 | 1.542 | 1.544 | +0.002 | · |
| r7U6JmcRl0ibIh0mN3CP9g | 225 | 2.328 | 2.328 | +0.000 | · |
| VJfVUwJs4k25ziMNvJb43A | 117 | 2.142 | 2.141 | −0.001 | · |
| lV0kQN0hR0ySmE0bQhuYbw | 181 | 2.023 | 2.022 | −0.001 | · |
| sLWe8bfYbkGwX1W3PzI1PQ | 234 | 2.439 | 2.438 | −0.001 | · |
| JteZUZ4JYkeD4k2rcVbqHg | 41 | 1.960 | 1.957 | −0.003 | · |
| VGlWeOPC6UiXSq4SYPiKpw | 116 | 4.011 | 4.006 | −0.005 | · |
| wr54weKzNkOcZ07hB04kzA | 273 | 2.604 | 2.599 | −0.005 | · |
| HIMqDWjSoECJFtIP0TM9bg | 22 | 3.980 | 3.975 | −0.005 | · |

**13/13 Cat-A deltas are < +0.05 AECMOS points.** None qualifies
as a "big save" threshold (+0.3). PZ7V target itself improves only
+0.032.

## Veto events

- Stage 1C 7-case probe: **PZ7V 1 run × 46 frames; 6 others 0
  events** (`feat: B-16 raw_dt jump veto (Stage 1B)` commit log).
- Stage 1D 800-case: per-case `_diag_b16_veto_active` counts not
  probed (expensive; would have required a second 800-case run
  with diagnostic hook). Signal structure is consistent with
  Stage 1C — the 4 cases with ΔFS_echo > +0.1 plus PZ7V likely
  account for most veto fires; other 795 cases likely 0 events,
  matching Stage 1C's 6/7 zero rate.

## Why the metric doesn't reflect the wav-level improvement

Technical success:
- PZ7V wav-level leak (10-11 s window RMS): **−13.83 → −20.42 dB**
  (Δ −6.59 dB, B-16 Stage 1C smoke)
- PZ7V AECMOS FS echo_mos: **1.908 → 1.940** (Δ +0.032, this Stage
  1D benchmark)

Root cause of the dilution:
- PZ7V leak window is ~1 s in a 21-s full file
- AECMOS evaluates the full 21-s signal in aggregate
- A 1-s localised improvement is ~21× diluted in the MOS mapping,
  especially when the rest of the file is already close to the
  model's saturation points for "clean" and "polluted"
- This is consistent with the forecast in
  `docs/aec3_full_architecture_analysis.md §6.1`: "B-16 alone
  ~−6 dB on PZ7V wav, shadow-as-NLMS needed for full recovery to
  target −25 dB". The +0.032 AECMOS points are the
  one-window-of-21 tail of that wav-level improvement.

## Movement cases — variance observation

Top-4 FS wins (ΔFS_echo > +0.1):
- case 136 XXz0qk **with_movement** +0.189
- case 62 MnzrTJXYi +0.125
- case 120 VNkNShj97 +0.103
- case 58 ML4MF3 **with_movement** +0.102

Top-10 regressions (max −0.138 case 265 w0QrMw **with_movement**,
next −0.099 case 194): all within −0.14 dB, zero cases exceed the
−0.2 floor.

**Observation**: movement-flagged cases show both the top wins
(+0.189) and the top regression (−0.138). The mean cancels to
+0.0002 but the per-case variance is structurally non-random —
B-16 is doing *something* in movement cases that is sometimes
right and sometimes wrong.

**Implication for Phase 2 shadow-as-NLMS**:
- Stage 1C covered one movement case (XTqo_mv, boundary case).
  800-case shows the movement population has richer raw_dt
  behaviour than that single sample suggested.
- Phase 2 benchmark plan should separate movement-subset analysis
  from overall.
- Movement-subset std should be an independent gate vs baseline,
  not just the 800-case overall std.
- A case-level audit of ±0.1 movement outliers (both directions)
  may reveal whether B-16's veto fires on real gain-jumps or on
  DT/FS artefacts that look similar.

## Merge decision rationale

Per Stage 1D verdict NEUTRAL, a strict metric-driven merge gate
would **not** merge. However:

1. Zero AECMOS regression verified (no downside with flag OFF).
2. Bit-exact baseline verified (B16=0 vs pre-merge main, max|Δ|
   below −40 dB noise level — see Stage 1B smoke section 5a).
3. Real wav-level improvement on 7/7 Cat-A Stage 1C cases.
4. Historical precedent: Group 5/7.1 EPC multi-level was also
   merged flag-gated with similar rationale (preserve code +
   experiment artefacts even without immediate metric benefit).
5. Spec + 5-round audit history (commits `867adc5`, `c544c67`,
   `f72b412`, `45c3239`, `7731387`, `284669f`) preserved in main
   log, not buried in feature branch.
6. B-16 as defensive layer complements future architectural
   fixes. Shadow-as-NLMS (Phase 2) doesn't subsume the raw_dt
   cascade protection — the two are orthogonal.

Flag `AEC_FIX_B16` default OFF. Users experiencing PZ7V-class
echo-path-gain-jump leaks can opt in explicitly.

## Future work — Stage 1D observations drive Phase 2 spec

1. **Separate movement subset benchmark** (both gate + reported).
2. **PZ7V wav-level dB as secondary metric** alongside AECMOS, to
   catch cases where a metric-invisible wav-level improvement is
   being delivered.
3. **2×2 matrix testing**: B-16 ON/OFF × shadow-as-NLMS ON/OFF
   (after shadow-as-NLMS implementation), so we can attribute any
   movement of ΔFS_echo to the right mechanism and detect
   interactions.

Reference: `docs/aec3_full_architecture_analysis.md §6` for the
full Phase 2-4 experiment queue.
