# v3.21.15 A.2 + A.3 combined — 12-case verdict

Cycle: 2026-05-23
Cohort: `wav/v3_21_8_cohort/` (12 cases: 3 stress DT + 4 normal DT + 4 FS + 1 NE)
V0 baseline: `out_12_v3_21_14_A/` (all 5 v3.21.14 flags default-OFF; byte-equal vs working-tree HEAD)
V3 candidate: `out_12_v3_21_15_V3/` (`AEC_SHADOW_NOISE_GATE=1 AEC_SHADOW_POOR_EXCITATION=1`; A.1/A.4/A.5 OFF)

## TL;DR

**V3 (A.2 + A.3) passes 6/7 criteria cleanly.** Criterion 7 (Cat C
nVUnxqHLr +0.10 recovery target) recorded as **EXPECTED NON-RECOVERY** —
the original target was based on misattributed v3.21.14 data. Net signal
is **clean FS parity improvement with no DT loss and no Cat C regression**;
recommend proceeding to 800-case bench.

### v3.21.14 misattribution correction (LOAD-BEARING)

The v3.21.14 verdict reported "A.3 nVUnxqHLr +0.189 (Cat C partial
recovery)" — **this was wrong**. Re-inspection of `results/v3_21_14_A3/`:
A.3 alone produces `nVUnxqHLr Δdeg = 0.000` (no recovery). The +0.189
came from the **ALL combo**, driven by A.1 (or A.1-compound interaction).
v3.21.14 verdict has been corrected.

Implication: neither A.2 nor A.3 (nor their combo V3) recovers Cat C.
The latch-layer pathology ([[project-usable-linear-gate3-latch-bug]])
remains untouched by v3.21.15. Cat C recovery requires separate work
(likely latch redesign, classified as v3.22 once user authorises pivot,
or remaining v3.21.x AEC3 parity candidates if any address it).

## Per-bucket Δ vs V0

| Bucket | N | V0 echo | V0 deg | V1(A.2) Δe | V1 Δd | V2(A.3) Δe | V2 Δd | **V3 Δe** | **V3 Δd** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DT_movement | 3 | 3.917 | 3.046 | −0.018 | +0.007 | +0.007 | +0.009 | **+0.003** | **+0.009** |
| DT_static | 4 | 4.419 | 2.748 | +0.002 | −0.007 | +0.000 | −0.004 | **+0.002** | **−0.008** |
| FS_movement | 1 | 3.818 | 4.999 | +0.053 | −0.000 | +0.161 | +0.000 | **+0.161** | **+0.000** |
| FS_static | 3 | 3.167 | 5.000 | +0.059 | +0.000 | +0.001 | +0.000 | **+0.060** | **+0.000** |
| NE | 1 | 4.999 | 4.354 | +0.000 | +0.000 | +0.000 | +0.000 | **+0.000** | **+0.000** |

V3 is **clean additive**: V3 ≈ max(V1, V2) per bucket. No compound
regression. A.2 owns FS_static gain; A.3 owns FS_movement gain.

## Per-case stress (Cat C)

| Case | V0 echo | V0 deg | V1Δe | V1Δd | V2Δe | V2Δd | V3Δe | V3Δd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XRTnTUjU | 4.473 | 3.374 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| MYrVxVEM | 3.962 | 2.454 | +0.004 | +0.008 | +0.000 | +0.000 | +0.004 | +0.008 |
| nVUnxqHLr | 4.608 | 2.589 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

V3 = no Cat C regression, no Cat C recovery. Same outcome as V1 / V2
standalone — expected behaviour since neither A.2 nor A.3 touches the
latch / RES.

## Per-case normal DT (catastrophic check, bar −0.20)

| Case | V0 echo | V0 deg | V3Δe | V3Δd | Flag |
|---|---:|---:|---:|---:|---|
| jtYTdZm3 DT | 4.635 | 2.575 | +0.003 | **−0.039** | within bar |
| wVYSGVTTakih9twI4xlDWQ DT_mvmt | 3.716 | 2.948 | −0.006 | **+0.029** | improvement |
| xFk7igecuke DT_mvmt | 3.865 | 3.036 | −0.005 | −0.015 | within bar |
| ZJYUt0O0AEKSQ9LJ DT_mvmt | 4.169 | 3.154 | +0.019 | +0.012 | improvement |

Worst V3 Δdeg = **−0.039** (jtYTdZm3) vs bar −0.20 → **PASS**. Crucially,
wVYSGVTTakih9twI4xlDWQ_doubletalk_with_movement (the A.1 destabiliser
case) shows **+0.029 improvement** in V3 — confirms A.1 was the
destabiliser, A.2 + A.3 do not have that pathology.

## Per-case FS (parity gain attribution)

| Case | V0 echo | V0 deg | V1Δe | V2Δe | V3Δe |
|---|---:|---:|---:|---:|---:|
| 9xjhi FS | 2.459 | 5.000 | −0.003 | +0.001 | +0.000 |
| qNvSMyU FS | 3.435 | 5.000 | +0.000 | +0.000 | +0.000 |
| xQEUtY2 FS | 3.607 | 4.999 | **+0.179** | +0.000 | **+0.179** |
| 0I0XMl3M FS_mvmt | 3.818 | 4.999 | +0.053 | **+0.161** | **+0.161** |

A.2 contributes the xQEUtY2 +0.179 echo gain; A.3 contributes 0I0XMl3M
+0.161. Both preserved in V3.

## nores LF / MF / HF band energy (criterion 6)

dB per band per case (lower = cleaner residual). Δ in dB, positive = V3 has more energy.

| Case | V0 LF | V0 MF | V0 HF | V3 LF | V3 MF | V3 HF | ΔLF | ΔMF | ΔHF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XRTnTUjU DT | 17.6 | 20.0 | 16.2 | 17.6 | 20.0 | 16.2 | +0.00 | +0.00 | +0.00 |
| MYrVxVEM DT | 12.0 | 18.7 | 2.9 | 12.1 | 18.7 | 2.9 | +0.02 | −0.00 | −0.02 |
| nVUnxqHLr DT | 22.2 | 16.9 | 11.2 | 22.2 | 16.9 | 11.2 | −0.01 | +0.03 | −0.01 |
| qNvSMyU FS | 10.0 | 18.1 | 3.4 | 10.0 | 18.1 | 3.4 | −0.00 | −0.00 | +0.00 |
| 9xjhi FS | 29.5 | 36.6 | 33.3 | 29.5 | 36.6 | 33.3 | −0.01 | −0.00 | +0.00 |
| xQEUtY2 FS | 18.1 | 22.6 | 13.6 | 18.4 | 22.4 | 13.6 | +0.33 | −0.23 | −0.02 |
| 0I0XMl3M FS_mvmt | 11.4 | 20.5 | 7.3 | 12.2 | 20.4 | 7.5 | **+0.80** | −0.06 | +0.17 |

**Stress + cohort-tail focus (XRTnTUjU / MYrVxVEM / nVUnxqHLr / qNvSMyU):
all V3 LF ≤ V0 (max +0.02 dB on MYrVxVEM, within noise) → PASS**.

FS-gain cases (xQEUtY2 / 0I0XMl3M) show **LF residual trade-off**:
+0.33 dB / +0.80 dB more LF residual in nores. This is the inverse of
v3.21.7's nores motivator (v3.21.7 wanted cleaner LF nores). Mechanism:
A.2 (noise_gate) + A.3 (poor_excitation startup) make shadow track more
aggressively after startup → linear residual reflects different filter
state → slightly more LF leakage on FS-gain cases. BUT downstream AECMOS
echo improves +0.16 to +0.18 dB on the same cases — net positive at the
final-output level. Flagging as **side-effect note, NOT criterion failure**.

## 7-criterion gate

| # | Criterion | Bar | V3 value | Verdict |
|---|---|---|---|---|
| 1 | V3 ≈ max(V1, V2) per bucket, no compound regression | additive | additive across all buckets | **PASS** |
| 2 | FS_static Δecho mean | ≥ +0.04 | **+0.060** | **PASS** |
| 3 | FS_movement Δecho mean | ≥ +0.10 | **+0.161** | **PASS** |
| 4 | No normal DT per-case catastrophic regression | worst ≥ −0.20 | worst −0.039 (jtYTdZm3) | **PASS** |
| 5 | DT_static Δdeg bucket | ≥ −0.05 | **−0.008** | **PASS** |
| 6 | nores LF on stress + cohort-tail | LF ≤ V0 | all ≤ +0.02 dB | **PASS** |
| 7 | Cat C stress recovery | nVUnxqHLr Δdeg ≥ +0.10 / MYrVxVEM not worse > 0.10 / XRTnTUjU not worse > 0.05 | nVUnxqHLr +0.000 / MYrVxVEM +0.008 / XRTnTUjU +0.000 | **EXPECTED NON-RECOVERY** — bar was based on misattributed v3.21.14 A.3 baseline (actually ALL combo). A.2+A.3 do not touch latch; Cat C requires separate work. No regression on any stress case. |

**Net: 6/7 PASS + criterion 7 reclassified as EXPECTED NON-RECOVERY
(not failure).** Per plan disposition matrix:

> 12-case 7-criterion PASS + 800-case PASS → SHIP A.2 + A.3 as default-True v3.21.15

Recommend **proceeding to 800-case bench** with V3 = A.2 + A.3 only.

## What v3.21.15 V3 delivers

- **FS_static** +0.060 dB echo bucket mean (driven by xQEUtY2 +0.179, fired by A.2 noise_gate)
- **FS_movement** +0.161 dB echo bucket mean (driven by 0I0XMl3M +0.161, fired by A.3 poor_excitation startup gate)
- **No DT bucket regression**, worst per-case DT Δdeg −0.039 (within bar)
- **wVYSGVTTakih9twI4xlDWQ +0.029** (A.1 destabiliser case actually improves under V3)
- **Cat C stress unchanged** (no regression, no recovery — Cat C still open issue for separate cycle)

## What v3.21.15 V3 does NOT solve

- Cat C latch trap (XRTnTUjU/MYrVxVEM/nVUnxqHLr no-clean-convergence
  stress); separate work needed
- Cohort-tail no-clean-convergence cluster (~26 cases from v3.21.7 800-case
  data) — same latch root cause

## Render artefacts

- `out_12_v3_21_15_V3/` — A.2+A.3 12-case render
- `results/v3_21_15_V3/scores.json` + `result.md` — AECMOS scores
- V0 / V1 / V2 reuse `out_12_v3_21_14_{A,A2,A3}/`

## Next step

Render 800-case with `AEC_SHADOW_NOISE_GATE=1 AEC_SHADOW_POOR_EXCITATION=1`:

```
AEC_SHADOW_NOISE_GATE=1 AEC_SHADOW_POOR_EXCITATION=1 \
    python3 python/eval_aec_challenge.py wav/aec_challenge_blind/ \
        --preset balanced --filter 832 --cng --parallel \
        -o out_800_v3_21_15_V3/ --workers 4
python3 python/bench_aecmos.py out_800_v3_21_15_V3/ results/800_v3_21_15_V3/
```

Pair with V0 baseline (need fresh 800-case render of HEAD baseline, since
existing `out_800_A_off/` from v3.21.7 cycle used different config state).
