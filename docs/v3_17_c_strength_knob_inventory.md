# v3.17 Phase C.1 — Strength knob inventory (2026-05-15)

**Branch**: `feature/v3.17` HEAD `cb677b4`.
**Sprint**: Phase C.1 (per [`docs/v3_17_plan.md`](v3_17_plan.md) §C.1).
**Source**: `python/aec.py:1042-1228` (preset definitions in
`AecConfig.from_preset()`).

---

## 1. Strength gradient by knob (5 presets)

| Knob | MILD | SOFT | BALANCED | AGGRESSIVE | MAXIMUM | Aggressive direction |
|---|---:|---:|---:|---:|---:|---|
| `res_g_min_db` | −25 | −35 | −55 | −65 | −72 | LOWER (deeper notch) ✓ monotone |
| `res_over_sub_base` | 1.5 | 2.5 | 5.0 | 7.0 | 10.0 | HIGHER ✓ monotone |
| `res_over_sub_scale` | 2.5 | 4.0 | 9.0 | 12.0 | 15.0 | HIGHER ✓ monotone |
| `res_dt_reduction` | 4.5 | 3.5 | 2.5 | 1.5 | 0.5 | LOWER ✓ monotone |
| `res_spectral_floor_db` | −18 | −25 | −38 | −45 | −55 | LOWER ✓ monotone |
| `res_ne_protect_db` | −7 | −10 | −16 | −22 | −30 | LOWER ✓ monotone |
| `res_enr_scale` | 1.15 | 1.0 | 0.85 | 0.7 | 0.5 | LOWER ✓ monotone |
| `res_alpha_echo_psd` | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | LOWER (faster EMA) ✓ monotone |
| `res_alpha_error_psd` | 0.6 | 0.6 | 0.5 | 0.4 | 0.3 | LOWER ✓ weakly monotone |
| `shadow_q_ratio` | 3.0 | 3.0 | 3.5 | 4.0 | 5.0 | HIGHER ✓ weakly monotone |
| `shadow_mu_min` | 0.5 | 0.5 | 0.6 | 0.7 | 0.9 | HIGHER ✓ weakly monotone |
| `kalman_q_high` | 1.5e-3 | 1.5e-3 | 1e-3 | 7e-4 | 7e-4 | LOWER ✓ weakly monotone |

**Conclusion**: all 12 strength knobs are monotone or weakly monotone
in the intended aggressiveness direction. The knob design is correct.

## 2. Substrate-flag asymmetry (CRITICAL FINDING)

BALANCED includes **14 default-OFF substrate flags** that are flipped
ON for BALANCED but NOT for MILD / SOFT / AGGRESSIVE / MAXIMUM:

| Flag | Origin |
|---|---|
| `use_mic_excess_evidence` | F3.1-v3 (v3.10.6) |
| `epc_r_reset_enabled` | F2.3 (v3.10.6) |
| `mu_holdoff_no_reset` | F2.4 (v3.10.6) |
| `shadow_r_reset_enabled` | v3.11 B5 |
| `shadow_state_decoupled` | v3.14 S-orth.A |
| `f3_1_per_band_erl_adaptive` | v3.14 Arc P |
| `res_per_band_enr` | v3.14 Arc R |
| `f_e5_enabled` | v3.11 F-E5 (saturation) |
| `diverged_reset_enabled` | v3.11 |
| `diverged_reset_triple_and` | v3.11 |
| `shadow_mu_state_aware` | v3.12 S1 |
| `f_e1_enabled` | v3.12 S2 |
| `f_delaytrack_enabled` | v3.12 S2 |
| `arc_t_cohort_detector` | v3.15 §10.S0b |

**Implication**: MILD / SOFT / AGGRESSIVE / MAXIMUM are running on a
v3.9 vintage substrate while BALANCED is running on v3.15.0 substrate.
The "strength gradient" assumption (presets differ only in strength
knobs, share substrate) is FALSE.

**Why this matters for tunability**:
- Users selecting "SOFT" expect a less-aggressive BALANCED, but they
  actually get a v3.9-era algorithm.
- Comparison of MILD vs MAXIMUM AECMOS scores doesn't isolate
  strength-knob contribution; it confounds with substrate-flag flips.
- Any future "user-dial-able strength" interface needs substrate
  parity across operating points.

## 3. v3.17 Phase C.3 candidate

Promote the 14 BALANCED-only substrate flags to ALL 5 presets, then
re-bench gradient. Each substrate flag was individually validated on
800-case AECMOS at promotion time. Combined effect on non-BALANCED
presets is unverified — could regress (because each flag was tuned
against BALANCED-knob context).

**Risk**: high. Substrate-flag promotion across 4 presets requires
4 × 800-case A/B + listen verification. Multi-sprint LOE.

**Alternative (safer)**: Document the asymmetry; recommend users stay
on BALANCED unless they explicitly need MILD/SOFT for ultra-light
mode or AGGRESSIVE/MAXIMUM for stub-output need. Defer substrate
parity to v3.18+.

## 4. Phase C.2 plan (next sprint)

Run 60-case bench on each of 5 presets, measure:
- AECMOS echo per bucket
- AECMOS deg per bucket

Verify monotone gradient on (FS Δecho, DT Δdeg, NE Δdeg) across
MILD → MAXIMUM. Output: `docs/v3_17_c_preset_gradient_audit.md`.

If monotone (despite substrate asymmetry): document as user-facing
gradient. If non-monotone: identify where (which preset, which
metric) and decide: re-tune (Phase C.3) OR document caveat OR defer
to v3.18+.
