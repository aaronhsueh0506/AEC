# v3.21.5 Sprint C2 — Per-bin H_error refresh selector re-evaluation CLOSED NO-LEVERAGE

**Date**: 2026-05-21
**Branch**: `feature/v3_21_5_phase1_aec3_parity` (HEAD `f15cf23`)
**Status**: **CLOSED at C2.0 gate — no-leverage; single-case memory evidence stale**
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (Round 7 v3.21.5 Sprint C2)
**Code impact**: ZERO new code (env hook scaffolding already in commit `2b98c13`); flag `use_per_bin_h_error_refresh` stays default-False; selector path remains dormant research code

## Context (Round 6 reframe — what Sprint C2 actually was)

Sprint C2 was NOT "add new code". The AEC3 `H_error += factor * erl` refresh ALREADY exists at [`filters.py:625`](../python/modules/filters.py#L625) (`_h_error_refresh()`), called always-on at [`filters.py:623`](../python/modules/filters.py#L623). What `use_per_bin_h_error_refresh: bool = False` ([`config.py:169`](../python/modules/config.py#L169)) gates is a **per-bin REFINED/COARSE selector path** — feeds `_e2_coarse_per_bin` (per-bin instantaneous coarse error PSD) vs the legacy scalar `_e2_coarse_for_refresh` to the refresh formula.

C2 was therefore a re-evaluation of an existing dormant code path under the new v3.21.5 baseline (with Sprint A's E2=min(E2,Y2) clamp ON, Sprint B rejected, default-True legacy stationarity zeroing preserved).

## Prior evidence

- **Memory `project_v3_21_pbfdkf_incomplete_port.md` (2026-05-18)**: single-case tracer on `9xjhiFbGo06hdQIsHTS6qA` showed ERLE 10.22 → **18.64 dB** (= **+8.42 dB**) when `use_per_bin_h_error_refresh=True`. Cited as evidence the per-bin path was load-bearing PARITY that was wrongly rejected.
- **U4.A standalone retest verdict `docs/v3_21_4_u4a_per_bin_h_error_retest_verdict.md` (2026-05-20)**: 800-case bench at flag=True on v3.21.3 baseline → CLOSED FAIL with 17% per-case regression (82 Δecho < -0.05; 54 Δdeg < -0.05; worst Δdeg -0.437 DT_static).

C2's hypothesis was that the disagreement (single-case +8.4 dB win vs 800-case 17% per-case regression) might be reconciled by Sprint A's E2 clamp — A correctly bounds `nearend_pwr` → `dominant_nearend` detector fires correctly on DT segments → SuppressionGain uses conservative `nearend_tuning` on DT → per-bin tracking damage is masked. If hypothesis holds: A + C2 cumulative on 800-case might be net Pareto-positive even though C2 alone was rejected.

## C2.0 — Cohort tail trace re-verify (5-case, ON TOP OF A only)

**Setup**: 5 FS_static worst cases (Sprint A.0 cohort) rendered twice each:
- **A only baseline**: `AEC_E2_Y2_CLAMP=1` (stationarity legacy default-True via post-Round-7 revert in commit `2b98c13`)
- **A + C2**: `AEC_E2_Y2_CLAMP=1 AEC_PER_BIN_H_ERROR_REFRESH=1`

Reproduce:
```bash
git checkout f15cf23
for stem in pcb1Nh0Z3k0WS9a7gBEuqg LN18k5r8t00C9DulUd809A s90M7MOTBkqaV4nQPLhKbA 9xjhiFbGo06hdQIsHTS6qA lV0kQN0hR0ySmE0bQhuYbw; do
  mic=wav/aec_challenge_blind/farend_singletalk/${stem}_farend_singletalk_mic.wav
  ref=wav/aec_challenge_blind/farend_singletalk/${stem}_farend_singletalk_lpb.wav
  AEC_E2_Y2_CLAMP=1                              python3 python/run_one_case.py "$mic" "$ref" /tmp/A_only/${stem}.wav   --preset balanced --trace-hf-chain /tmp/A_only/${stem}.csv
  AEC_E2_Y2_CLAMP=1 AEC_PER_BIN_H_ERROR_REFRESH=1 python3 python/run_one_case.py "$mic" "$ref" /tmp/A_plus_C2/${stem}.wav --preset balanced --trace-hf-chain /tmp/A_plus_C2/${stem}.csv
done
```

### Per-case ERLE summary (A only vs A + C2)

ERLE windowed (mean over case, converted log2 → dB via ×3.0103). NE = `is_nearend_state` fire rate; UL = `usable_linear` fire rate; e2>y2 = Sprint A clamp activity proxy.

| Case | Config | erle_5 dB | erle_30 dB | erle_100 dB | NE % | UL % | e2>y2 | e2>y2_HF | e2_excess dB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pcb1Nh | A only | 3.53 | 3.49 | 4.18 | 2.8 | 88.5 | 0.284 | 0.272 | +3.82 |
| | A+C2 | 3.57 | 3.52 | 3.99 | 5.1 | 86.8 | 0.273 | 0.263 | +3.72 |
| | **Δ** | +0.04 | +0.03 | **-0.19** | +2.3 | -1.8 | -0.010 | -0.009 | -0.10 |
| LN18k5r8 | A only | 3.01 | 3.01 | 3.01 | 0.0 | 0.0 | 0.603 | 0.614 | +10.39 |
| | A+C2 | 3.01 | 3.01 | 3.01 | 0.0 | 0.9 | 0.597 | 0.608 | +10.26 |
| | **Δ** | 0.00 | 0.00 | **0.00** | 0.0 | +0.9 | -0.006 | -0.006 | -0.13 |
| s90M7MOT | A only | 3.01 | 3.12 | 3.10 | 12.2 | 76.7 | 0.337 | 0.328 | +10.80 |
| | A+C2 | 3.06 | 3.07 | 3.06 | 12.7 | 77.9 | 0.337 | 0.329 | +10.75 |
| | **Δ** | +0.05 | -0.05 | **-0.04** | +0.5 | +1.3 | +0.001 | 0.000 | -0.05 |
| **9xjhi** | A only | 3.36 | 5.09 | 5.42 | 0.0 | 89.5 | 0.240 | 0.228 | +4.11 |
| | A+C2 | 3.43 | 5.47 | 5.40 | 0.0 | 92.4 | 0.234 | 0.222 | +4.05 |
| | **Δ** | +0.08 | **+0.38** | **-0.01** | 0.0 | +2.9 | -0.006 | -0.006 | -0.07 |
| lV0kQN | A only | 3.60 | 4.18 | 7.89 | 3.3 | 97.0 | 0.254 | 0.250 | +3.27 |
| | A+C2 | 3.48 | 4.11 | 7.22 | 3.3 | 97.0 | 0.253 | 0.249 | +3.26 |
| | **Δ** | -0.12 | -0.07 | **-0.67** | 0.0 | 0.0 | -0.001 | -0.001 | -0.01 |

## C2.0 gate verdict

Plan gate (verbatim):
> - 9xjhi ERLE doesn't reproduce ≥ +5 dB improvement → "single-case evidence stale; canonical state changed"; close C2 as no-leverage
> - Reproduces AND DT segments inside FS cases show conservative-gain pattern (Sprint A working) → proceed to C2.1
> - Reproduces but `dominant_nearend` fire rate unchanged from baseline → hypothesis weak; still proceed to C2.1 but mark "no synergy expected"

**9xjhi Δ erle_100 = -0.01 dB** (memory predicted +8.42 dB). The single-case win that the memory cited does NOT reproduce on the v3.21.5 baseline. Plus 4/5 cohort cases show neutral-to-negative Δ on erle_100 (the long-window indicator of cumulative cancellation). The hypothesis test fails at the first gate.

**Gate decision: CLOSE C2 as no-leverage.**

### Why the memory result no longer reproduces

The 2026-05-18 memory note's 9xjhi tracer was taken on a code state that has since changed:
- v3.21.2 fixed FFT-scale bin-index unit-conversion bug (`docs/v3_21_2_*.md`)
- v3.21.3 Codex hygiene (AEC.reset / dead contracts / etc.)
- v3.21.4 ms-based config refactor + 4 carry-over audits (per-bin H_error retest CLOSED FAIL; B3 intermediate values CLOSED FAIL; etc.)
- v3.21.5 Sprint A E2=min(E2,Y2) clamp added (the C2 baseline)

Any one of these could shift the canonical state enough that the per-bin REFINED/COARSE selector no longer produces the single-case win. The U4.A 800-case bench (one day before v3.21.5 started) already showed the path doesn't generalize. C2's "Sprint A masking" hypothesis would require both the single-case win AND the masking effect to compose positively; with the single-case win already absent on the new baseline, the hypothesis has no leg to stand on.

## C2.1 / C2.2 / C2.3 not executed

Plan gate said close at C2.0 if 9xjhi doesn't reproduce ≥ +5 dB. Gate fail → skip C2.1 implementation (env hook scaffolding already exists in commit `2b98c13` for future re-evaluation), skip C2.2 visual check (no per-bin path active to compare), skip C2.3 full 800-case bench (no reason to bench a path that fails on the cohort tail's most-favorable case).

## Final state

- **Config flag `use_per_bin_h_error_refresh: bool = False`** unchanged at [`config.py:169`](../python/modules/config.py#L169); per-bin REFINED/COARSE selector path stays dormant research code at [`filters.py:625`](../python/modules/filters.py#L625) + downstream
- **Env hook `AEC_PER_BIN_H_ERROR_REFRESH`** in [`eval_aec_challenge.py`](../python/eval_aec_challenge.py) + [`run_one_case.py`](../python/run_one_case.py) (committed in `2b98c13`) — kept for future re-evaluation after v3.21.6 P1 (FilterAnalyzer port) or v3.21.6 P3 (EchoAudibilityConfig wiring) may again shift the canonical state
- **v3.21.5 cumulative ship path collapses to A only** (Sprint A E2 clamp). Bucket FS_static +0.033 / FS_movement +0.035; DT deg metric drops are AECMOS-sensitive but not audible (user spectrogram check confirms); ship as v3.21.5 candidate
- **Memory `project_v3_21_pbfdkf_incomplete_port.md`**: needs update — single-case 9xjhi 18.64 dB result no longer reproduces on v3.21.5 baseline; the U4.A + C2 evidence chain now consistently shows the per-bin selector path is not load-bearing PARITY but rather dormant code that may re-enter consideration after companion mechanisms ship in v3.21.6

## Triage policy reference

Per `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` "AEC3 Parity Gap Triage Policy", `use_per_bin_h_error_refresh` is **Bucket 1 (Must fix / evaluate before v3.22)** with cycle home v3.21.5 Sprint C2. Closure status now: **closed as no-leverage with trace evidence** — satisfies v3.22 entry gate requirement (no parity item in deferred state).

The path is NOT marked "intentionally incompatible with PBFDKF" because the U4.A failure mode is "per-bin leakage over-protects converged bins during echo-path movement → tracking lag" which is fixable in principle (with AEC3's ScaleFilter + FilterMisadjustment companions). If v3.21.6 P1+P3 lands those companions, a future cycle MAY re-evaluate the per-bin selector through the canonical control surface. Until then, the path stays dormant.

## Reproduction

```bash
git checkout f15cf23
mkdir -p /tmp/sprintC2_audio/{A_only,A_plus_C2}
# (Render the 5 cases per the C2.0 setup block above.)
# Analysis script:
cat > /tmp/sprintC2_audio/analyze_C2_0.py <<'PYEOF'
# (See attached /tmp/sprintC2_audio/analyze_C2_0.py for full script.)
PYEOF
python3 /tmp/sprintC2_audio/analyze_C2_0.py
# Expect: 9xjhi Δ erle_100 ≈ -0.01 dB → GATE FAIL → CLOSE.
```
