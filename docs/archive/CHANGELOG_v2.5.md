# CHANGELOG — v2.5.x Series

> Version format: `v2.5.X` — algorithm and RES tuning on PBFDKF baseline.
> All changes are in `python/aec.py` unless noted.

---

## v2.5.5 — NE Bypass (Route B v3)  2026-04

**Score vs AEC2:**

| Metric | v2.5.5 | AEC2 | Δ |
|--------|--------|------|---|
| NE deg | **4.104** | 4.098 | **+0.006 ✓** |
| DT deg | **2.418** | 2.388 | **+0.030 ✓** |
| FS echo | 3.393 | 3.484 | −0.091 (ceiling) |
| DT echo | 4.059 | 4.262 | −0.203 (ceiling) |

**Change:** When `res.far_activity < 0.15 AND far_power < 1e-4`, bypass RES entirely and return `raw_output` directly. Eliminates STFT/OLA round-trip cost of −0.116 NE deg.

```python
# aec.py ~4212
if self.res.far_activity < 0.15 and far_power < 1e-4:
    final_output = raw_output
```

**Root cause diagnosed:** NE deg loss was 100% from STFT/OLA path overhead, not gain suppression. `far_power < 1e-4` guard prevents DT files from triggering bypass (DT startup has `far_activity=0` before first far-end block).

**Reverted in this cycle (Route A/A-2):** FS ENR override via `ne_confidence` scaling — DT deg cost too high (−0.030). FS echo gap confirmed as acoustic ceiling (146/300 files echo_l=2.093).

---

## v2.5.4 — Route A/A-2 (reverted)  2026-04

**Status:** Fully reverted. No net code change from v2.5.3.

- Route A: global `ne_confidence` scaling → DT deg −0.030, unacceptable
- Route A-2: FS-specific ENR override with `effective_dt < 0.45` gate → DT quiet frames also triggered
- EMR diagnostic: `g_emr_mean=0.065`, `emr_mean=21M` → EMR completely inactive in FS, not the bottleneck
- FS echo gap confirmed structural: only 9/146 below-AEC2 files are addressable, impact +0.004

---

## v2.5.3 — RES DT Gain Floor + Gate Fix  2026-04

**Changes:**

1. **ENR slope** (`dt_enr_relax`): slope 0.5 → 0.8. Effect: +0.004 DT deg (noise level).

2. **DT gain floor** (Stage 2B): `dt_gain_floor = 0.3 + ...` → floor binding, +0.017 DT deg.

3. **DT gain floor** (Stage 2B-2): `dt_gain_floor = 0.5 + ...` → FS echo regressed −0.017 (FS eff_dt≈0.403 incorrectly triggered floor=0.501).

4. **Gate fix:** `if effective_dt > 0.4` → `if effective_dt > 0.45`. FS eff_dt=0.403 no longer triggers floor. FS echo recovered.

```python
# aec.py ~2005
if effective_dt > 0.45:
    dt_gain_floor = 0.5 + (effective_dt - 0.4) / 0.6 * 0.2  # max 0.7
    g = np.maximum(g, dt_gain_floor)
```

**Net result from v2.5.2:** FS echo 3.408→3.557, DT deg 2.424→2.437.

---

## v2.5.2 — KF-1 R Calibration (not executed)  2026-04

**Hypothesis:** R_mean high → K degraded → filter divergence → echo leak.
**Diagnostic:** `r(K, echo_linear) = 0.022` (near-zero correlation). Root cause is acoustic ceiling, not Kalman state.
**Decision:** No change made.

---

## v2.5.1 — HR Gate Relaxation (HR-2)  2026-04

**Problem:** 74–76% of `blk_adv` events concentrated in `adv_raw` 1.05–1.20. Shadow filter outperforms but copy is blocked.

**Change:** `_HC_MIN_ADV_RATIO` reduced in two stages:
- Stage 1: 1.20 → 1.15 (FS +14.7%, DT +14.1% copy rate)
- Stage 2: 1.15 → **1.10** (FS +13.2%, DT +14.6%, blk_adv −69%)

```python
# aec.py ~2691
self._HC_MIN_ADV_RATIO = 1.10
```

**Safety check:** harmful rate 13.2%/14.6% < 15% threshold. Remaining blk_adv events are at noise floor; further relaxation not safe.

**Learning:** HR gate relaxation alone cannot improve `echo_linear`. Linear filter performance is the upstream bottleneck.

---

## v2.5.0 — Safety Gates  2026-04

**Problem:** DT catastrophic tail events; `harmful` HR copy rate too high.

**Changes:**

```python
# aec.py ~2690
self._HC_DT_SIGNAL_MAX = 0.85   # block copy during heavy DT (dt_signal > 0.85)
self._HC_MIN_ADV_RATIO = 1.20   # minimum shadow advantage for HR copy (initial)
```

**Result:** DT catastrophic 2→1, harmful 4→2. Baseline established for v2.5.x tuning.

---

## Architecture Notes (v2.5.x era)

- **Shadow filter**: hardcoded to PBFDAF (NLMS) since Phase ι (pre-v2.5). `AEC_EXP_SHADOW_NLMS` flag removed. See `aec.py:2669-2676`.
- **Remaining open gaps**: Filter Misadjustment W-Scaling (#2), Usable Linear Gate (#3), Reset granularity (#4). See `docs/aec3_full_architecture_analysis.md`.
- **Version naming**: v2.5.x replaces prior Phase κ/ι naming. Greek letters retired.
