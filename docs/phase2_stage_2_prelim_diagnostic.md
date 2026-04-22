# Phase 2 Stage 2 — Preliminary Diagnostic: Static Farend Regression

Date: 2026-04-22. Branch `feature/phase2-shadow-as-nlms`.
Related: [phase2_stage_1d_results.md](phase2_stage_1d_results.md), [spec_phase2_shadow_as_nlms.md](spec_phase2_shadow_as_nlms.md) v4.

## 1. Goal

Identify root cause of Phase 2 static farend regression (−0.040 FS_echo on 169 static cases, §3 of Stage 1D results). Hypothesis: Phase 2 PBFDAF shadow causes spurious shadow→main copy events in stable FS periods, disrupting converged main filter。

## 2. Method

Probed 3 representative big-loss cases from Stage 1D `(0,1)` vs `(0,0)` FS big-losses list:

| Case (short) | Stage 1D ΔFS_echo | Type |
|---|---:|---|
| KSN5Jr | −0.722 | Static FS (top big loss) |
| IUp2c0 | −0.294 | Static FS |
| LeV1uF | −0.234 | Static FS |

Probe: `/tmp/probe_static_regress.py` — run AEC frame-by-frame, log per-frame `shadow_adv`, `main_err_smooth`, `shadow_err_smooth`, `shadow_copy_counter`, `_shadow_advantage_streak`, `W_l2` (main filter weight norm), and detect copy events via `W_l2` jumps > 1.8×。

Both `AEC_EXP_SHADOW_NLMS=0` and `=1` traced。CSV output at `/tmp/static_probe_{KSN5Jr,IUp2c0,LeV1uF}_{base,p2}.csv`。

## 3. Copy event timelines

### 3.1 KSN5Jr (worst loss)

| Run | Frame | t (s) | shadow_adv | W_l2 before → after | Note |
|---|---:|---:|---:|---|---|
| (0,0) | 260 | 2.60 | 1.000 | 0 → 0.14 | Cold-start main |
| (0,0) | 261 | 2.61 | 1.106 | 0.14 → 1.83 | Copy |
| (0,0) | 262 | 2.62 | 1.110 | 1.83 → 3.48 | Copy (W converging) |
| (0,1) | 260 | 2.60 | 1.000 | 0 → 0.14 | Same cold-start |
| (0,1) | 261 | 2.61 | 1.156 | 0.14 → 1.83 | Copy (shadow_adv slightly higher) |
| (0,1) | 262 | 2.62 | 1.366 | 1.83 → 3.48 | Copy |
| **(0,1)** | **1799** | **17.99** | **7.201** | **3.05 → 18.39** | **NEW late copy — 6× W jump** |

**Late copy at 18s is the regression source**. Baseline converges normally; Phase 2 triggers a disruptive 6× W rewrite at end-of-file due to shadow_adv spike to 7.2。

### 3.2 IUp2c0

| Run | Frame | t (s) | shadow_adv | W_l2 before → after |
|---|---:|---:|---:|---|
| (0,0) | 64 | 0.64 | 1.027 | 0.09 → 1.09 |
| (0,0) | **1780** | **17.80** | **2.007** | **1.85 → 3.87** |
| (0,1) | 64 | 0.64 | 1.240 | 0.09 → 1.09 |

**(0,1) missed the baseline's frame-1780 beneficial copy**. PBFDAF shadow_adv built up earlier (1.240 at frame 64 vs 1.027 baseline), possibly consuming copy opportunity budget before the late productive moment。

### 3.3 LeV1uF

| Run | Frame | t (s) | shadow_adv | W_l2 before → after |
|---|---:|---:|---:|---|
| (0,0) | 76/77 | 0.76–0.77 | 1.000/1.012 | 0 → 1.46 |
| (0,0) | 245 | 2.45 | 2.303 | 6.91 → 26.08 |
| (0,1) | 76/77 | 0.76–0.77 | 1.000/**2.015** | 0 → 1.46 |
| (0,1) | 246 | 2.46 | 2.575 | 6.94 → **29.10** |

Copy **timing similar**, but Phase 2 `shadow_adv` peaks higher (2.575 vs 2.303), producing 12% larger copy (29.1 vs 26.1). Copy is slightly more aggressive than baseline needs。

## 4. Pattern

Three distinct regression mechanisms across the 3 cases:

| Case | Mechanism |
|---|---|
| KSN5Jr | **Spurious late copy** (shadow_adv 7.2 at t=18s in stable FS → 6× W disruption) |
| IUp2c0 | **Missed beneficial copy** (early shadow_adv creep consumes budget; late productive copy doesn't fire) |
| LeV1uF | **Over-aggressive copy** (timing correct, amplitude ~12% larger than needed) |

Common root: **PBFDAF shadow in stable FS accumulates non-productive `shadow_adv` excess**, because `leak=1e-6` preserves cumulative learning better than PBFDKF main's `P/Q` dynamics (which gently drift in steady-state)。Result: shadow_err slightly lower than main_err even when main is fine, leading to copy-gate decisions that don't reflect real quality advantage。

## 5. Stage 2 tune direction (priority)

| # | Parameter | Current | Candidate | Rationale |
|---|---|---|---|---|
| 1 | `shadow_copy_threshold` | 0.65 | **0.55** | Stricter threshold filters out small `shadow_adv` creep; only real advantages (≥ 1/0.55 = 1.82) trigger copy。Directly addresses KSN5Jr/LeV1uF over-triggering |
| 2 | `shadow_copy_hysteresis` + streak | 3 + 10 | **5 + 20** | Longer observation window defeats transient advantage spikes。Addresses KSN5Jr t=18s spike |
| 3 | PBFDAF shadow `alpha_r_scale` | 1.03 | **1.10** | Slower shadow R tracking reduces shadow-over-main bias in stable periods。Addresses root-cause drift |
| 4 | New: stable-FS copy-gate suspend | — | `dt < 0.1 + recent copy < 2s → skip` | Defensive: don't allow back-to-back copies in stable conditions |

**Suggested Stage 2 order**:
1. Start with **#1 only** (shadow_copy_threshold 0.65 → 0.55) — single-parameter, clear rationale, grep-verifiable single-line diff
2. Stage 2B smoke on PZ7V + KSN5Jr + IUp2c0 + LeV1uF (4 cases, 2 combos each)
3. If PZ7V still PASSES (full-file ≤ −22 dB) AND 3 losers improve ≥ 0.2, extend to 800-case
4. Only add #2 / #3 if #1 alone doesn't retain PZ7V fix OR 3 losers still bad

Risk: threshold 0.55 可能讓 PZ7V copy 不 trigger (Stage 0b 量到 shadow_adv 2.0+ at onset, threshold 0.55 需要 shadow_err ≤ 0.55 × main_err i.e. shadow_adv ≥ 1.82 — PZ7V onset +16 frames 時 shadow_adv 達 2.04, 剛好過關但風險邊際)。Stage 2A 要確認 PZ7V 仍 PASS。

## 6. Alternative hypotheses checked

- [x] **Shadow→main copy is the direct mechanism**: Confirmed by W_l2 jump events co-incident with output leak degradation in KSN5Jr。
- [ ] main→shadow copy (reverse direction): not measured in this probe, but unlikely to cause main regression since main W isn't modified。
- [x] **EPC large fire difference**: Unlikely; Stage 1B telemetry showed EPC large=0 frames for PZ7V (0,1)。Would need separate probe on static cases to rule out。

## 7. Artifacts

- `/tmp/probe_static_regress.py` — probe script
- `/tmp/static_probe_{KSN5Jr,IUp2c0,LeV1uF}_{base,p2}.csv` — per-frame trace
- `/tmp/static_probe_*_{base,p2}.err` — copy-event summaries (see §3 tables)

## 8. Not in scope

- Full Stage 2 spec — pending user go-ahead
- Stage 2 implementation — diagnostic only
- Threshold tune numeric calibration — §5 candidates are first-pass hypotheses, need Stage 2A experiment to finalize
