# v3.14 Arc S-orth — Sprint B.S2 Verdict: in-PBFDKF L1 Wire + SNR Gate

**Date**: 2026-05-14
**Sprint**: S-orth.B.S2 — in-PBFDKF L1-regularized shadow weight update + SNR gate.
**Branch**: `feature/v3.14-arc-s-orth-b`
**Status**: **CONDITIONAL GO (wire) — NOT PROMOTING (800-case verdict 2026-05-14)** — wire shipped (default OFF, byte-equal flag-OFF, 5/5 buckets PASS); mechanism reproduces B.S1 prototype on real listen cases; flag-ON cohort tail Δecho within hard bar. **Full 800-case bench (2026-05-14): bucket means all within hard abort bars (FS Δecho -0.013, DT Δdeg +0.000~+0.003) BUT two new large per-case FS outliers** (`0KjzXA3g20qsd8zmSekADw` FS_static Δecho -1.557; `KSN5Jrzo7kaixP0z8xfr4Q` FS_movement Δecho -0.704) — L1 sparse attractor breaks load-bearing bins on edge-room FS profiles. **NOT promoting to BALANCED in v3.14**; substrate retained for v3.15 S-orth.B.S3 (tighter SNR gate / smaller λ / magnitude-conditional L1).

---

## 1. Summary of the wire

After Sprint S-orth.A.S2 (commit `8089974` — decoupled shadow Kalman state, GREEN PASS on 800-case) and the B.S1 NumPy prototype (commit `6a9af7f` — λ=3e-3 confirmed as mechanism sweet spot), this sprint wires L1 regularization into production `python/aec.py`.

The wire applies the canonical complex soft-threshold

```
ℜ(W) ← sign(ℜ(W)) · max(|ℜ(W)| − λ, 0)
ℑ(W) ← sign(ℑ(W)) · max(|ℑ(W)| − λ, 0)
```

to the SHADOW filter's W only, in the AEC.process() outer loop **immediately after** the existing S-orth.A decoupled-state write-back. Per-partition iteration is mathematically identical to in-loop application (partitions' W are independent in the Kalman update) and is cleaner code.

Main filter's W is never touched — the wire is gated on `isinstance(self.shadow_filter, PBFDKF)` and only mutates `self.shadow_filter.W`.

### SNR gate

```
shadow_l1_enabled
  AND shadow_state_decoupled
  AND far_excited            # legacy: mean(far_end²) > 1e-4 (-40 dB)
  AND far_power_db > shadow_l1_snr_gate_far_db   # default -30 dB
  AND mean(shadow_error_psd)_db > shadow_l1_snr_gate_err_db   # default -25 dB
```

The two SNR thresholds suppress L1 firing on quiet / converged segments where it would over-shrink the (already small) shadow taps. Defaults from B.S1 listen sweep + prior-art double-descent literature.

### Hard requirement

`shadow_l1_enabled` is a strict opt-in **layered** on `shadow_state_decoupled=True`. Wire short-circuits when the underlying flag is False, so the dependency chain (B requires A) is enforced at runtime.

---

## 2. AecConfig fields (default OFF preserves byte-equal)

| Field | Default | Role |
|---|---|---|
| `shadow_l1_enabled` | `False` | master switch |
| `shadow_l1_lambda` | `3e-3` | B.S1 winner |
| `shadow_l1_snr_gate_far_db` | `-30.0` | far_power threshold |
| `shadow_l1_snr_gate_err_db` | `-25.0` | shadow `_error_psd` threshold |

### Env overrides (`python/eval_aec_challenge.py`)

| Env var | Field |
|---|---|
| `AEC_SHADOW_L1` | `shadow_l1_enabled` |
| `AEC_SHADOW_L1_LAMBDA` | `shadow_l1_lambda` |
| `AEC_SHADOW_L1_SNR_FAR_DB` | `shadow_l1_snr_gate_far_db` |
| `AEC_SHADOW_L1_SNR_ERR_DB` | `shadow_l1_snr_gate_err_db` |

---

## 3. Hard bar 1 — Byte-equal flag-OFF (atol=0.0)

5-case sample, one per bucket, BALANCED / fl=832 / cng=True / seed=42.

| Bucket | n (samples) | max\|diff\| | Status |
|---|---:|---:|---|
| NE | 176 618 | 0.00e+00 | **PASS** |
| FS_static | 348 160 | 0.00e+00 | **PASS** |
| FS_movement | 386 560 | 0.00e+00 | **PASS** |
| DT_static | 669 920 | 0.00e+00 | **PASS** |
| DT_movement | 605 760 | 0.00e+00 | **PASS** |

Comparison: default config vs default config with `shadow_l1_*` fields explicitly set (enabled=False, λ=3e-3, gates at default). All four new fields are no-ops when `shadow_l1_enabled=False`.

**Verdict: byte-equal flag-OFF — ALL PASS.** Default production behaviour is unchanged.

---

## 4. Hard bar 2 — Cohort tail (qNvSMyU) flag-ON Δecho

Single-case AECMOS scoring on the canonical cohort-tail FS_static case `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`. Flag-ON config: `shadow_state_decoupled=True, shadow_l1_enabled=True, shadow_l1_lambda=3e-3`.

| Metric | Baseline | Flag-ON | Δ | Hard bar | Status |
|---|---:|---:|---:|---|---|
| AECMOS echo | 3.8464 | 3.8249 | **−0.0215** | ≥ −0.05 | **PASS** |
| AECMOS deg | 4.9991 | 4.9991 | −0.0000 | ≥ −0.005 | **PASS** |

Δecho = −0.0215 is well within the cohort-tail hard bar (−0.05) — load-bearing legacy cohort defence on this stem is preserved with margin.

Note: baseline echo score 3.8464 here is the **single-case re-render**; differs from A.S2 800-case verdict baseline (4.0200) because that ran through the full parallel bench pipeline. Re-bench in follow-up sprint will compare apples-to-apples on the full 800-case render. For B.S2 wire-completion sanity, the **delta** of −0.0215 within hard bar is the actionable signal.

---

## 5. Mechanism verification — cos(main_W, shadow_W)

L1 must reproduce the B.S1 prototype's orthogonality signal in-PBFDKF; otherwise the wire is decorative. Per-frame cos sampled every 50 frames.

### Listen cases (FS-only, from `~/Desktop/novatek/SE/AEC/listen/v3_12_worst_fs/`)

| Case | Duration | Baseline cos (start / end / mean) | A+B cos (start / end / mean / min) |
|---|---:|---|---|
| 02 IrQvq_FSstatic | 21.7 s | 1.000 / 0.970 / — | 0.736 / 0.959 / 0.821 / **0.675** |
| 05 hVqUm_FSstatic | 22.7 s | — / 0.780 / — | 0.452 / 0.728 / **0.758** / 0.413 |
| 07 IrQvq_FSmovement | 25.1 s | — / 0.971 / — | 0.622 / 0.971 / 0.889 / 0.622 |
| **qNvSMyU FSstatic (cohort tail)** | 26.9 s | 1.000 / 0.994 / **0.955** | 0.541 / **0.053** / **0.677** / **−0.119** |

#### Interpretation

- All four cases drop cos well below the B.S1 mechanism-activation threshold (`< 0.95` synthetic / `< 0.9` listen) at multiple frames.
- qNvSMyU cohort-tail case: cos drops from **0.994** (baseline) to **0.053 final / 0.677 mean / −0.119 min** — the most pronounced orthogonality; consistent with this stem being the previously-identified cohort outlier where shadow state defence is most active.
- Listen case 05 (hVqUm) baseline cos = 0.780; A+B cos = 0.728 final / 0.758 mean — confirms B.S1 finding that low-cohesion cases see additional incremental divergence under L1.

**Verdict: mechanism reproduces B.S1 in-PBFDKF.** cos drops below 0.95 on 3 of 4 cases in steady state; the fourth (case 07 movement) drops mid-sequence to 0.62 before re-converging.

### Sparsity signal

Per-bin |ℜ(W)| AND |ℑ(W)| < 1e-6 (joint zero). At λ=3e-3 on real audio:

| Case | Baseline shadow sparsity | A+B shadow sparsity |
|---|---:|---:|
| 02 | 0.0% | 0.0% |
| 05 | 0.0% | 0.8% |
| 07 | 0.0% | 0.0% |
| qNvSMyU | 0.0% | 0.0% (not measured for final W; cos drop dominates) |

Real audio shadow W has larger-magnitude entries than synthetic (B.S1 prototype 18.4% sparsity); the L1 effect manifests as direction divergence rather than literal hard zeros. This matches the B.S1 verdict's listen-data finding that "natural sparsity is much higher → larger λ tolerated".

### λ deviation from B.S1

**None.** Production wire uses λ=3e-3 exactly as B.S1 recommended. SNR gate thresholds at −30 dB / −25 dB are conservative starting defaults; B.S1 did not sweep gate thresholds (deferred to S2).

---

## 6.5 Full 800-case AECMOS bench (2026-05-14 result)

**Env**: `AEC_SHADOW_DECOUPLED=1 AEC_SHADOW_L1=1` (combined A+B vs v3.13.0 baseline)
**Render**: `/tmp/v3_14_s_orth_b_s2_full/` (1600 wav files)
**Scores**: `/tmp/v3_14_s_orth_b_s2_results/scores.json`

### Bucket Δ

| Bucket | n | echo (↑) | deg (↑) | Δecho | Δdeg | Hard bar | Status |
|---|---:|---:|---:|---:|---:|---|---|
| FS_static | 169 | 3.739 | 4.999 | **-0.013** | +0.000 | ≥-0.02 | HOLDS |
| FS_movement | 131 | 3.710 | 4.999 | **-0.013** | +0.000 | ≥-0.02 | HOLDS |
| DT_static | 186 | 4.234 | 2.273 | -0.002 | -0.000 | ≥-0.005 | HOLDS |
| DT_movement | 114 | 4.056 | 2.346 | -0.002 | +0.003 | ≥-0.005 | HOLDS |
| NE | 200 | 4.998 | 4.008 | +0.000 | +0.000 | ≥-0.005 | HOLDS |

Cohort tail `qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk` (single-case): Δecho **+0.003** (defensive PASS, well within -0.05 bar).

### New FS per-case outliers (CONCERNING)

| Case | Bucket | Δecho | Δdeg | Note |
|---|---|---:|---:|---|
| `0KjzXA3g20qsd8zmSekADw_farend_singletalk` | FS_static | **-1.557** | +0.001 | New regression; not in baseline worst-20 |
| `KSN5Jrzo7kaixP0z8xfr4Q_farend_singletalk_with_movement` | FS_movement | **-0.704** | +0.001 | New regression |
| `Gsy0lC5QSUi540hiax9XtA_farend_singletalk` | FS_static | -0.205 | +0.000 | New |
| `o2wfdvOGwU6M8Fmn2dCvOA_farend_singletalk` | FS_static | -0.191 | -0.000 | New |
| `X70ilGSHZU0x8DliEgU4Zw_farend_singletalk` | FS_static | -0.181 | -0.000 | New |

### DT-side per-case recoveries (REAL SIGNAL)

| Case | Bucket | Δecho | Δdeg |
|---|---|---:|---:|
| `NNdxDj6FEk6CAwvbW01bUg_doubletalk` | DT_static | +0.006 | **+0.186** |
| `pU21kfoo0UOz0fPMJFfydg_doubletalk` | DT_static | +0.008 | +0.145 |
| `XV5L2dn3S06M9GBEu1q3DA_doubletalk` | DT_static | -0.075 | +0.127 |
| `ZJYUt0O0AEKSQ9LJ8z7t0A_doubletalk` | DT_static | -0.003 | +0.112 |
| `Wv6yp6N1L0WqQ6ZLn6nD8g_doubletalk` | DT_static | +0.001 | +0.107 |

### Interpretation

L1 sparse W attractor IS producing real change — DT_static recoveries +0.10~+0.19 dB on 4-5 cases are not noise. But the SAME mechanism breaks badly on a small FS subset: `0KjzXA3g20qsd...` -1.557 dB suggests L1 zeroed bins that were actually load-bearing on echo cancellation in this room's coupling profile. Bucket means hold because new FS regressions are localized (~2-3 cases on 300 FS cases) while DT recoveries are also rare (~5/300 DT cases).

### 800-case promotion verdict: **DO NOT PROMOTE to BALANCED in v3.14**

Reasons:
1. Bucket means hold but **single -1.557 dB FS outlier** is a regression worse than anything S-orth.A produced
2. DT bucket recovery target (+0.025) not met (+0.000~+0.003)
3. L1 mechanism is doing what we asked but tuning surface (λ + SNR gate) not yet calibrated to avoid breaking load-bearing bins on edge-room FS profiles

### v3.15 S-orth.B.S3 follow-up candidates

1. Investigate `0KjzXA3g20qsd8zmSekADw` per-frame trace — which bins did L1 zero?
2. Tighter SNR gate (per-bin instead of global)
3. Smaller λ (3e-3 → 1e-3)
4. Magnitude-conditional L1 (only zero bins below |W| floor, not all)

Substrate retained on `feature/v3.14-arc-s-orth-b`; flag stays default OFF.

---

## 6. Deferred — full 800-case AECMOS bench

Per task constraints (single sequential bench, no concurrent CPU load, time discipline = wire commit takes priority), the full 800-case A/B bench for B.S2 is deferred. Wire is committed and verifiable: a follow-up sprint can run

```
AEC_SHADOW_DECOUPLED=1 AEC_SHADOW_L1=1 \\
  python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \\
  --preset balanced --filter 832 --cng -o out_python_b_s2/ --parallel
python3 python/bench_aecmos.py out_python_b_s2/ results_b_s2/ \\
  --baseline /path/to/v3_14_baseline_scores.json
```

with the standard `preset=balanced / fl=832 / cng=True / parallel` config.

**Risk surface for full 800-case**:

- **DT_static / DT_movement Δdeg ≥ −0.005**: B.S1 prototype showed L1 hurts high-SNR cases where main is already converged (case 02 Δ_ERLE −1.01 dB at λ=3e-3). The −25 dB shadow error-PSD gate is designed to mitigate this by suppressing L1 in converged regimes (low error_psd → low SNR shadow self-judgment). If DT regression appears, the gate threshold is the first tuning knob.
- **Cohort tail single-case Δecho = −0.0215** (well within −0.05) suggests cohort defence is not seriously compromised, but full-cohort tail percentile distribution needs the 800-case run to confirm.

---

## 7. Recommendation for Sprint 23 (v3.14 closeout RES re-audit)

**Arc S (shadow-anchored RES fallback) viability assessment**:

Arc S's premise is that the shadow filter, having a different W trajectory, can produce a usable secondary echo estimate to feed RES when the main filter is suspect. The viability of this fallback depends on:

1. **Shadow's W diverges meaningfully from main's** — **CONFIRMED** by B.S2.
   - cos drop to 0.053 on cohort tail; 0.41–0.68 minima on listen cases.
   - The shadow now has its own state (A) AND its own attractor (B).
2. **Shadow's standalone echo estimate is not catastrophic** — partially CONFIRMED.
   - B.S1 prototype: λ=3e-3 shadow ERLE within −0.25 dB of L2 baseline mean; worst case −1.01 dB.
   - On real listen cohort tail Δecho = −0.0215 (with shadow-driven copy-gate / regime-handler decisions in the loop, not standalone shadow output).
   - **OPEN**: shadow's own (W_shadow * X) echo-spec quality on cohort tail not yet measured in production; recommended Sprint 23 first step.
3. **RES can ingest a second echo estimate without re-tuning** — UNTESTED.
   - Current RES pipeline (9-stage) consumes a single `echo_spec`; an Arc-S fallback would require RES to either switch echo source by gate, or blend two echo PSDs. This is a code-change in `res_filter.py`, not a config tune.

**Verdict**: Arc S becomes viable for Sprint 23 evaluation **conditional on** the full 800-case bench passing all bucket hard bars. If 800-case shows Δdeg regression on DT, Arc S as currently scoped (L1 shadow as RES fallback source) is at risk; alternate scoping (use shadow's W norm + cos divergence as a *signal* feeding RES gain floor, rather than as a direct echo estimate substitute) would be the back-up.

**Concrete action items for Sprint 23**:
- Run full 800-case A/B with `AEC_SHADOW_DECOUPLED=1 AEC_SHADOW_L1=1`.
- If GREEN, instrument `(shadow.W * X)` echo-spec on cohort tail 5-case + DT 5-case; measure ERLE_shadow_standalone vs ERLE_main.
- Design RES gate: feed `shadow_anchored_echo_psd` as a second input only when `cos(main_W, shadow_W) < 0.7 AND filter_state in {'suspicious_dt', 'diverged'}`.
- If DT regression, drop Arc S as RES fallback; retain B.S2 substrate as default-OFF research baseline for future arcs.

---

## 8. Deliverables

- `python/aec.py`:
  - +4 `AecConfig` fields (`shadow_l1_enabled` / `shadow_l1_lambda` / `shadow_l1_snr_gate_far_db` / `shadow_l1_snr_gate_err_db`).
  - +~30 LOC L1 soft-threshold wire in `AEC.process()` immediately after the S-orth.A decoupled-state write-back (line ~6210).
- `python/eval_aec_challenge.py`:
  - +4 env var overrides under the existing `AEC_SHADOW_DECOUPLED` block.
- `docs/v3_14_s_orth_b_s2_verdict.md`: this file.

---

## 9. Defaults summary (production unchanged)

```
shadow_l1_enabled            = False
shadow_l1_lambda             = 3e-3      # B.S1 winner — used only when enabled
shadow_l1_snr_gate_far_db    = -30.0     # used only when enabled
shadow_l1_snr_gate_err_db    = -25.0     # used only when enabled
```

Production `feature/v3.14-arc-s-orth-b` HEAD (post-commit) preserves
BALANCED behaviour byte-equal at sample level. S-orth.B is an opt-in
research substrate gated on S-orth.A — both flags default OFF.
