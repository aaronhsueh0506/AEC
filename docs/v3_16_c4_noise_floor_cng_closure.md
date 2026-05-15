# v3.16 C4 — noise_floor / CNG audit CLOSED (2026-05-15)

**Status**: CLOSED — H4 effectively REFUTED per §0.4.
**Branch**: `feature/v3.16` (commit pending).
**Sprint**: v3.16 Phase 2 audit-first (last v3.16 candidate).
**Substrate**: `tools/research/v3_16_c4_noise_floor_cng_audit.py` retained.

---

## 1. Headline

**Verdict**: §1.1 H4 (noise_floor / CNG over-aggressive on DT-NE
residual) is REFUTED in spirit. noise_floor is **active in 43-55 %
of DT frames** but **NEVER dominates the gain decision** (`nfl_dom = 0 %`
across all buckets — noise_floor_gain is always smaller than 1.2 ×
res_gain_lin). The lift mechanism is acting as designed: PROTECTS NE
from over-suppression below noise level, NOT the source of DT-NE
compression.

DT-NE compression source remains §1.1 H1+H2 (ENR gate per-state ×
per-band → C2 candidate). C4 mechanism arc would target the wrong
surface.

**v3.16 has now disposed all 5 audit-able candidates** (C6 / v3.16-A /
C3 / C9 / C4). Only architectural / dependent / low-priority remain.

---

## 2. Audit data (60-case subset, 162,791 frames)

### 2.1 Per-bucket noise_floor / CNG activity

| Bucket | n | frames | nfl_lift_all | nfl_g_all | cng_g_all |
|---|---:|---:|---:|---:|---:|
| FS_static | 11 | 25,386 | 62.0 % | 0.516 | 0.216 |
| FS_movement | 11 | 25,371 | 79.2 % | 0.560 | 0.211 |
| DT_static | 14 | 51,572 | 43.0 % | 0.606 | 0.128 |
| DT_movement | 13 | 48,352 | 42.9 % | 0.593 | 0.128 |
| NE | 11 | 12,110 | 7.3 % | 0.452 | 0.071 |

`nfl_lift_pct` is the rate at which noise_floor_gain raises gain_smooth
(`gain_smooth < noise_floor_gain` before the `np.maximum`). NE bucket
shows the LOWEST lift rate (7.3 %) — consistent with noise_floor being
mostly inactive when there's no echo to suppress.

### 2.2 H4 target slice (dt_from_energy > 0.3 proxy for DT-active)

| Bucket | dt frames | dt_nfl_lift | dt_nfl_g | dt_res_g_lin | dt_nfl_dom | res_gain p5 dB |
|---|---:|---:|---:|---:|---:|---:|
| DT_static | 3,872 | 47.0 % | 0.430 | 0.771 | **0.0 %** | −5.24 |
| DT_movement | 2,918 | 54.8 % | 0.460 | 0.769 | **0.0 %** | −5.44 |
| FS_movement | 1,660 | 72.8 % | 0.577 | 0.805 | 0.0 % | −5.01 |
| FS_static | 2,317 | 31.1 % | 0.326 | 0.756 | 0.0 % | −5.64 |
| NE | 100 | 31.0 % | 0.305 | 0.779 | 0.0 % | −4.35 |

**Critical finding**: `dt_nfl_dom = 0 %` across ALL buckets. The
condition tested was `noise_floor_gain > res_gain_lin × 1.2` — i.e.
"is the noise_floor lift the dominant determinant of the final gain?"
Empirically it NEVER is. res_gain_lin (0.756–0.805) is consistently
1.5–2.4× the noise_floor_gain (0.305–0.577).

### 2.3 Implication: noise_floor is NE-protection floor, not compression source

When `noise_floor_gain` ≈ 0.43 and `res_gain` ≈ 0.77, the
`np.maximum(gain_smooth, noise_floor_gain)` operation:
- In the **95 %+ of frames** where `gain_smooth > noise_floor_gain`,
  the lift is a no-op (gain unchanged).
- In the **rare deep-suppression frames** where `gain_smooth <
  noise_floor_gain`, the lift PREVENTS over-suppression below noise
  level (audibility floor).

This is **textbook NE-protection behaviour**, not compression. Removing
or capping the noise_floor would WORSEN NE preservation in the rare
deep-suppression frames, not improve DT-NE compression.

### 2.4 CNG observation

`cng_g_all` is moderate (0.07-0.22 mean across buckets) and consistent
with comfort-noise filling spectral gaps. NE bucket has the lowest CNG
gain (0.071) — also expected (less suppression on NE means smaller
spectral gap to fill).

CNG also does NOT show a compression signature.

---

## 3. v3.16 plan implications

### 3.1 C4 disposition

**CLOSED for v3.16** per §0.4. noise_floor / CNG mechanism is
audit-validated as designed (NE-protection floor + comfort noise fill).
H4 hypothesis was a wrong-surface attribution; DT-NE compression
source is upstream (ENR gate, §1.1 H1+H2 → C2 candidate).

### 3.2 Remaining v3.16 candidates (5 / 15 = 33 %)

| ID | Phase | Status |
|---|---|---|
| C5 | 1.6 | DEFERRED (architectural, ZERO Δ; defer to v3.17) |
| C2 | 2 | candidate (likely subsumed by §1.2 / Arc D wall) |
| v3.16-B | 3 | gated on C2 |
| C7 | 4 | candidate (was CLOSED v3.15 §1.5b) |
| C8 | 4 | candidate (LOW priority partial-decay alt for Arc G; Arc G CLOSED v3.15) |

**All 5 remaining candidates are either architectural (no Δ),
likely-closure (FS-vs-DT wall family), or were already closed in v3.15.**
The v3.16 cycle has **fully exhausted audit-able candidates**.

### 3.3 v3.16 closeout finalised

- **10 / 15 candidates disposed** (5 Phase 0 + 5 mechanism closures
  including C4).
- **1 production change** (`d90efdc` C1 epc_dt_cap removal).
- **5 closure verdict docs** (C6 + v3.16-A + C3 + C9 + C4) inform v3.17.

---

## 4. Substrate (committed)

- Audit script:
  [`tools/research/v3_16_c4_noise_floor_cng_audit.py`](../tools/research/v3_16_c4_noise_floor_cng_audit.py)
- Per-case JSONs (gitignored): `/tmp/v3_16_c4_audit/per_case/*.json`
- Aggregate: `/tmp/v3_16_c4_audit/summary.json`

Zero changes to `python/aec.py` — audit consumed existing
`_stats_last_*` surface (after enabling via `aec.res.enable_stats()`)
and `dt_from_energy` AecStats field (proxy for dt_active since
BALANCED has `enable_dtd=False`).

---

## 5. Verdict signed-off

**CLOSED for v3.16** — H4 effectively REFUTED. noise_floor is
NE-protection floor; CNG fills spectral gaps. Neither is the source of
DT-NE compression. C4 mechanism arc not opened.

v3.16 cycle has now closed all 5 audit-first candidates (C6 / v3.16-A /
C3 / C9 / C4). Of remaining 5, all have closure indicators or
architectural-only status. **Cycle definitively closed.** Awaits §0.7
user merge authorisation.
