# v3.17 Phase C.2 — Preset gradient audit (2026-05-15)

**Branch**: `feature/v3.17` HEAD `ab4cf2c`.
**Sprint**: Phase C.2 (per [`docs/v3_17_plan.md`](v3_17_plan.md) §C.2).
**Verdict**: ✓ ALL 5 PRESETS MONOTONE on 60-case AECMOS — gradient
interface is correctly tuned; **no Phase C.3 re-tuning needed**.

---

## 1. Bench config

- 60-case Tier 1 subset (`tools/research/v3_15_subset_cases.txt`)
- Each preset rendered separately: `--preset {mild,soft,balanced,aggressive,maximum}`
- All other defaults: `fl=832 / cng=True / parallel=True`
- AECMOS scored with `model/Run_1663915512_Stage_0.onnx`

## 2. Bucket means by preset

### echo_mean (higher = more echo suppression)

| Bucket | MILD | SOFT | BALANCED | AGGRESSIVE | MAXIMUM | Δ MILD→MAX |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 3.225 | 3.383 | 3.507 | 3.562 | 3.645 | **+0.420** |
| FS_movement | 3.460 | 3.640 | 3.719 | 3.723 | 3.752 | **+0.292** |
| DT_static | 4.109 | 4.325 | 4.439 | 4.483 | 4.480 | +0.371 |
| DT_movement | 3.770 | 3.856 | 4.003 | 4.031 | 4.065 | +0.295 |
| NE | 4.999 | 4.999 | 4.999 | 4.999 | 4.999 | +0.000 (saturated) |

### deg_mean (higher = better NE preservation)

| Bucket | MILD | SOFT | BALANCED | AGGRESSIVE | MAXIMUM | Δ MILD→MAX |
|---|---:|---:|---:|---:|---:|---:|
| FS_static | 5.000 | 4.999 | 4.999 | 4.999 | 4.999 | -0.001 (saturated NE-empty) |
| FS_movement | 5.000 | 4.999 | 4.999 | 4.999 | 4.999 | -0.001 (saturated) |
| DT_static | 2.551 | 2.369 | 2.377 | 2.307 | 2.257 | **−0.294** |
| DT_movement | 2.655 | 2.491 | 2.421 | 2.395 | 2.349 | **−0.306** |
| NE | 3.874 | 3.833 | 3.747 | 3.734 | 3.687 | **−0.187** |

## 3. Monotonicity verdict

ALL 10 bucket-metric pairs PASS monotonicity check (tolerance 0.01 dB
to ignore sub-noise jitter):

| Bucket | echo monotone ↑? | deg monotone ↓? |
|---|---|---|
| FS_static | ✓ | ✓ |
| FS_movement | ✓ | ✓ |
| DT_static | ✓ | ✓ |
| DT_movement | ✓ | ✓ |
| NE | ✓ (saturated) | ✓ |

## 4. Gradient characterisation

The preset gradient delivers the expected echo / NE trade-off:
- **MILD → MAXIMUM**: gains ~0.4 dB echo suppression on FS_static
- **MILD → MAXIMUM**: loses ~0.3 dB NE preservation on DT
- DT_static AECMOS deg is the most sensitive metric to preset
  selection (Δ = -0.294 dB across the gradient)
- NE bucket loses 0.187 dB deg moving aggressive — NE-only cases
  still get processed by RES even when no echo is present (gain
  attenuation reduces NE intelligibility)

## 5. Substrate asymmetry doesn't break user-facing gradient

Per Phase C.1 (`docs/v3_17_c_strength_knob_inventory.md`), 14
default-OFF substrate flags are flipped ON for BALANCED only,
NOT for MILD / SOFT / AGGRESSIVE / MAXIMUM. This means the
non-BALANCED presets run on v3.9-vintage substrate.

Despite this asymmetry, the OUTPUT gradient is monotone:
- v3.10+ substrate improvements (F3.1-v3, F2.3, F2.4, Arc P, Arc R,
  S-orth.A, etc.) are most impactful at the BALANCED operating
  point where ENR is in its sensitive range
- MILD's ultra-light suppression doesn't need them as much
- MAXIMUM's heavy suppression already saturates the metrics
- The strength knobs (12 monotone parameters) dominate the gradient

**Implication**: substrate parity (Phase C.3 candidate) is NOT
required for v3.17 Phase C deliverable. The user-facing tunable
strength interface is already functional.

## 6. Phase C.3 disposition

Per v3.17 plan §C.3: "If C.2 finds non-monotone knob behavior,
re-tune to enforce monotone gradient." C.2 found 0 non-monotone
metrics → Phase C.3 NOT triggered.

Optional v3.18+ candidate: substrate flag promotion across all 5
presets (4 × 800-case A/B + listen verify). Risk-bounded refactor;
defer until user explicitly needs non-BALANCED presets to inherit
v3.10+ improvements.

## 7. v3.17 Phase C — overall verdict

✓ Phase C.1 — strength knob inventory complete; 12 monotone knobs
  identified; substrate asymmetry documented
✓ Phase C.2 — preset gradient bench PASS; 10/10 bucket-metrics
  monotone on 60-case AECMOS
✓ Phase C.3 — NOT NEEDED (gradient already correct)

Phase C delivers a v3.17 substrate confirming the preset interface
is ready for downstream user-facing tunable strength UI. No code
changes; documentation-only deliverables.

## 8. Next: v3.17 closeout

Per v3.17 plan, after Phase C lands:
- Compile v3.17 verdict pack: B.1 + B.3 + B.2 closures + Phase A +
  Phase C deliverables
- Present to user for §0.7 merge authorisation
- If user authorises: `feature/v3.17` → main + tag v3.17.0
- If not: branch remains as research substrate, main stays at v3.16.0
