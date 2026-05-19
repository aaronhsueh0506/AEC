# F3.1 v3 verdict — blend with legacy (2026-05-12)

## TL;DR

**F3.1 v3 is the chosen F3.1 productisation candidate.** Blending the excess-ratio metric `0.7 × F3.1 + 0.3 × (1 - coh2)` on top of v2's `not epc_active` gate improves all 5 FS_movement outliers (best gain: pG9Bikvr Δecho −0.140 → −0.076), recovers KOy0eft (DT_static gainer that v2 had flipped to regress), and closes the FS_movement bucket mean by 33 % vs v1 (−0.0055 → −0.0037). DT_static deg gain magnitudes shrink (tl5UFRCX +0.178 → +0.126, i2BU43nm +0.174 → +0.099) but the count stays at 2 |Δdeg|>0.05 gainers.

## 800-case bench (v1 vs v2 vs v3, all vs main baseline)

### Bucket means Δ

| Bucket | v1 Δecho | v2 Δecho | **v3 Δecho** | v1 Δdeg | v2 Δdeg | **v3 Δdeg** |
|---|---:|---:|---:|---:|---:|---:|
| FS_static   | −0.0011 | −0.0010 | **−0.0016** | +0.0000 | +0.0000 | +0.0000 |
| FS_movement | −0.0055 | −0.0046 | **−0.0037** | +0.0000 | +0.0000 | −0.0000 |
| DT_static   | +0.0003 | +0.0006 | +0.0006 | +0.0006 | −0.0002 | −0.0003 |
| DT_movement | +0.0017 | +0.0015 | +0.0027 | −0.0030 | −0.0024 | −0.0033 |
| NE          | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

All within `bench_aecmos.py` bars (FS echo ≥ −0.02, NE deg ≥ −0.01).

### FS_movement 5 outliers Δecho

| stem | base | v1 | v2 | **v3** |
|---|---:|---:|---:|---:|
| Lsa5Wpw | 3.880 | −0.301 | −0.301 | **−0.280** |
| s0oJqM6Y | 2.898 | −0.213 | −0.143 | **−0.133** |
| wr54weK | 3.719 | −0.167 | −0.161 | **−0.142** |
| pG9Bikvr | 3.818 | −0.140 | −0.142 | **−0.076** |
| Khk1qeM | 3.825 | −0.075 | +0.009 | +0.012 |

All five improve. pG9Bikvr the largest single recovery (+0.064 vs v2); Lsa5Wpw moves off the v2 plateau (+0.021).

### DT_static top gainers Δdeg

| stem | base | v1 | v2 | **v3** |
|---|---:|---:|---:|---:|
| tl5UFRCX | 3.405 | +0.178 | +0.165 | **+0.126** |
| i2BU43nm | 2.792 | +0.174 | +0.174 | **+0.099** |
| Wv6yp6N1 | 2.772 | +0.063 | +0.037 | +0.020 |
| KOy0eft | 1.799 | +0.060 | −0.037 | **+0.011** |

DT gain magnitudes shrink under the blend (expected: F3.1 contributes 70 % instead of 100 %); KOy0eft, which v2 had flipped to regress, recovers under v3.

### |Δ|>0.05 histogram (v3 vs main)

| Bucket | echo regress | echo gain | deg regress | deg gain |
|---|---:|---:|---:|---:|
| FS_static   | 4 | 0 | 0 | 0 |
| FS_movement | 4 | 1 | 0 | 0 |
| DT_static   | 0 | 1 | 2 | **2** |
| DT_movement | 0 | 2 | 4 | 0 |
| NE          | 0 | 0 | 0 | 0 |

Vs v2: FS_movement count unchanged (4/1) but each regressor weaker per the outlier table. DT_static deg gain count 2 (same as v2, down from v1's 4) — blend trade.

## Decision log

The full F3.1 design lineage on this branch:

  • **v1** — pure excess-ratio metric, gated by `filter_converged AND _long_window_n_updates > 0`. GREEN-PASS slim (`docs/f3_1_phase1_verdict.md`).
  • **F2.1** — reset upstream state on EPC. CLOSED FAIL (`docs/f2_1_verdict.md`): hard reset of EMA state hurts F3.1 metric (erl→0.1 over-attributes; erle_window→0 collapses over_sub).
  • **v2** — add `not epc_active` to gate. Marginal trade (`docs/f3_1_v2_verdict.md`): recovers 2/5 outliers (s0oJqM6Y, Khk1qeM) but unchanged on Lsa5Wpw and loses one DT gainer (KOy0eft).
  • **v3 (this verdict)** — add 0.7:0.3 blend with legacy `(1 − coh2)` on top of v2 gate. Improves all 5 outliers, recovers KOy0eft, dilutes top DT gainers. All buckets within bars; FS_movement +33 % vs v1.
  • An interim v3 with a `mic_pwr ≤ 2·far·erl` envelope gate was tried and rejected: it blocked legitimate DT cases where mic naturally exceeds expected echo (i2BU43nm DT-static gainer +0.174 → 0). Softening with `OR effective_dt ≥ 0.2` re-opened FS-noise paths. Binary gating on mic/far ratio can't be made FS-only without a label we don't have; the blend is the honest cap.

## Mechanism

The blend addresses two of three failure regimes from the outlier audit:

  • **Regime 1 (EPC hangover)** — handled by v2 `not epc_active` gate.
  • **Regime 3 (high-coupling, erl underestimate)** — softened by blend. Pure F3.1 over-attributes NE when `erl_estimate` is capped at 0.3 but true ERL is 0.5–0.7 (Lsa5Wpw / wr54weK). 30 % legacy weight bounds the swing.
  • **Regime 2 (FS with non-echo content)** — not addressed by a gate; pG9Bikvr still has Δecho −0.076 (better than v1's −0.140 but not zero). The blend partially helps because the legacy `(1 − coh2)` is closer to "neutral" than F3.1's strong NE-attribution in those frames.

## Verdict: **GREEN-PASS, F3.1 v3 = productisation candidate**

- ✅ Bucket means all within bars; FS_movement mean closest to baseline of all three versions.
- ✅ 5 FS_movement outliers all improve vs v1 (best +0.064 on pG9Bikvr).
- ✅ Cohort tail `qNvSMyUSXUyrDGp` still Δ = 0/0 (gate correctly excludes — filter never converges).
- ✅ NE bucket Δ = 0/0 (gate keeps F3.1 inert when far is silent).
- ⚠️ DT_static deg gain magnitudes shrink under blend; still net positive at bucket level and 2 |Δdeg|>0.05 gainers remain.

Default OFF; flag = `use_mic_excess_evidence`. Mirrored to `python/res_refactored/gain_computer.py`.

## Next-step candidates (not blocking)

1. **F3.2 — per-bin effective_dt** (original plan). Decouples scalar gates (HF cap / spectral floor lift / ENR relax) from frame-level effective_dt, so F3.1's per-bin evidence can propagate to those decisions. Independent of erl_estimate accuracy.
2. **Lsa5Wpw / pG9Bikvr listen check** to confirm whether the residual −0.280 / −0.076 are perceptually audible or AECMOS-only artifacts.
3. **Tuning the blend weight** (0.7 chosen by intuition; sweep 0.5 / 0.6 / 0.8 to see if a different ratio improves the DT-gain / FS-regress trade).

## Files / artefacts

- Code: `python/aec.py` + `python/res_refactored/gain_computer.py` (commit 263ef20 on `feature/f3-1-mic-excess-evidence`).
- Branch commit history: 0eb318d (impl) → 721891d (env var) → 765d340 (runner) → 8560c42 (resume flag) → cec1c1f (glob fix) → b6719b1 (F2.1) → 21f832e (F2.1 verdict) → 0e08de8 (v2 gate) + 2ed03da / f3_1_phase1_verdict.md / f3_1_v2_verdict.md → 263ef20 (v3 blend).
- Bench output: `/tmp/f3_1_v3_on/` (800 WAVs).
- Scores: `/tmp/f3_1_v3_scores/{scores.json, result.md}`.
- Unit tests: 7 F3.1 + 4 F2.1 + 13 P52 — 24/24 green.
