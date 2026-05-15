# v3.18 Phase D.2 — Mask Profile Substrate (2026-05-15)

**Goal**: build per-bin `enr_transparent / enr_suppress / emr_transparent`
tables for `normal` and `nearend` mask profiles. Phase D.3 consumes
them in the gain compute stage.

**Verdict**: ✓ SUBSTRATE READY

## 1. What landed

### 1.1 AecConfig — 7 new fields

| Field | Default | Notes |
|---|---|---|
| `res_mask_profile_swap_enabled` | `False` | D.3 master switch |
| `res_mask_last_lf_band` | `20` | bin 20 = 625 Hz @ fs=16k, fft=512 (AEC3 last_lf=5 @ fft=128) |
| `res_mask_first_hf_band` | `32` | bin 32 = 1000 Hz (AEC3 first_hf=8) |
| `res_mask_normal_lf` | `(0.3, 0.4, 0.3)` | AEC3 verbatim `normal.mask_lf` |
| `res_mask_normal_hf` | `(0.07, 0.1, 0.3)` | AEC3 verbatim `normal.mask_hf` |
| `res_mask_nearend_lf` | `(1.09, 1.1, 0.3)` | AEC3 verbatim `nearend.mask_lf` |
| `res_mask_nearend_hf` | `(0.1, 0.3, 0.3)` | AEC3 verbatim `nearend.mask_hf` |

Tuple ordering: `(enr_transparent, enr_suppress, emr_transparent)`.

Source for verbatim values:
`api/audio/echo_canceller3_config.h` @ commit `9310b29acd`
(via WebFetch — file not in local extract trim).

### 1.2 ResFilter — profile builder

New method [`_build_mask_profiles()`](../python/aec.py) at L2495.
Called once after `self.n_freqs` is set in `__init__`. Builds
`self._normal_mask_profile` + `self._nearend_mask_profile`, each a
3-tuple of `(n_freqs,) np.float32` arrays.

Interpolation formula mirrors AEC3
[suppression_gain.cc:487 `GainParameters::SetConfig`](aec3_extracts/src/aec3/suppression_gain.cc):

```
for k in [0..n_freqs):
    if k <= last_lf:      a = 0
    elif k < first_hf:    a = (k - last_lf) / (first_hf - last_lf)
    else:                  a = 1
    enr_t[k] = (1-a)*lf.enr_t + a*hf.enr_t
    enr_s[k] = (1-a)*lf.enr_s + a*hf.enr_s
    emr_t[k] = (1-a)*lf.emr_t + a*hf.emr_t
```

Built unconditionally (flag-state independent) so D.3 can toggle the
consumer at runtime without re-init. Profile cost ≈ 6 KB total (6 × 257
× 4 bytes for the standard BALANCED + fl=832 config).

### 1.3 AEC class — 7 kwargs pass-through

[python/aec.py:5708-5715](../python/aec.py#L5708) — forwards
`config.res_mask_*` fields into `ResFilter()`.

## 2. Verification

### 2.1 Profile build correctness

Smoke test with BALANCED preset (n_freqs=257, bin_spacing=31.25 Hz):

| Bin | Freq (Hz) | norm_enr_t | norm_enr_s | ne_enr_t | ne_enr_s | HF NE lift |
|---|---|---|---|---|---|---|
| 10 | 312 | 0.3000 | 0.4000 | 1.0900 | 1.1000 | **+0.7000** |
| 20 | 625 | 0.3000 | 0.4000 | 1.0900 | 1.1000 | **+0.7000** |
| 25 | 781 | 0.2042 | 0.2750 | 0.6775 | 0.7667 | +0.4917 |
| 32 | 1000 | 0.0700 | 0.1000 | 0.1000 | 0.3000 | **+0.2000** |
| 64 | 2000 | 0.0700 | 0.1000 | 0.1000 | 0.3000 | +0.2000 |
| 128 | 4000 | 0.0700 | 0.1000 | 0.1000 | 0.3000 | +0.2000 |
| 200 | 6250 | 0.0700 | 0.1000 | 0.1000 | 0.3000 | +0.2000 |
| 256 | 8000 | 0.0700 | 0.1000 | 0.1000 | 0.3000 | +0.2000 |

**HF NE lift = 0.2 (= 0.3 − 0.1) — exact AEC3 default ✓**

LF region (bins 0-20): nearend `enr_suppress` is **1.1 vs FS 0.4 (2.75×
higher)** — needs much louder echo before suppression engages.

HF region (bins 32-256): nearend `enr_suppress` is **0.3 vs FS 0.1 (3×
higher)** — this is the primary HF NE protection mechanism (R2 in
[docs/aec3_reference.md §11](aec3_reference.md)).

### 2.2 Byte-equal flag-OFF (default config)

Same 5-case `eval_aec_challenge.py` run as Phase D.1:

| Set | Hash count | Sorted diff |
|---|---|---|
| D.2 flag-OFF | 10 files | — |
| Baseline (87730e4) | 10 files | identical |

`diff` of sorted hash columns: **empty → 10/10 md5 PASS** → default-OFF
byte-equal verified.

(Sanity: build-but-don't-consume is byte-equal-safe because profile
arrays are only referenced under `if self._mask_profile_swap_enabled:`
guards introduced in D.3.)

## 3. Substrate readiness checklist

- [x] 7 config fields with AEC3-verbatim defaults
- [x] ResFilter accepts + stores anchor params + band boundaries
- [x] `_build_mask_profiles()` runs after `self.n_freqs` known
- [x] LF→HF interpolation correct (linear, mirroring AEC3 §3.1)
- [x] HF NE lift = AEC3 default 0.2 (numerical parity verified)
- [x] Default-OFF byte-equal (10/10 md5 PASS on 5-case eval)
- [x] AEC class instantiation pass-through
- [ ] D.3 wiring — pending next sprint
- [ ] D.4 effective_dt migration — pending
- [ ] D.5 tune — pending (sub1/sub2 bins + thresholds + anchor values)

## 4. Files modified

- [python/aec.py](../python/aec.py)
  - L1029-1064 — AecConfig 7 new fields with AEC3-verbatim defaults + comment block
  - L2178-2185 — ResFilter __init__ kwargs
  - L2253-2266 — `_mask_*` attr storage + profile placeholders
  - L2473-2491 — `_build_mask_profiles()` method + call after n_freqs
  - L2493+ — `_build_mask_profiles` definition (helper, ~20 lines)
  - L5708-5715 — AEC class ResFilter instantiation kwargs

## 5. Cross-references

- AEC3 source: [docs/aec3_extracts/src/aec3/suppression_gain.cc:487](aec3_extracts/src/aec3/suppression_gain.cc) (`GainParameters::SetConfig`)
- AEC3 config defaults: `api/audio/echo_canceller3_config.h` @
  `9310b29acd` (not in local extract — fetched via WebFetch)
- Phase D.1 substrate: [docs/v3_18_d1_subband_ne_detector.md](v3_18_d1_subband_ne_detector.md)
- Phase 0.2 mapping: [docs/aec3_residual_pipeline_mapping.md §3.1](aec3_residual_pipeline_mapping.md)
- Phase 0.6 decision: [docs/v3_18_phase0_decision.md](v3_18_phase0_decision.md)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
