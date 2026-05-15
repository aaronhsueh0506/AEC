# v3.18 Phase D.3 — Mask Shape Swap (2026-05-15)

**Goal**: rewrite `_stage_gain_compute` per-bin masking to use D.2 mask
profile lookup tables with AEC3-style binary swap (R2). Gate by D.1
subband NE state AND echo-aware `effective_dt > 0.3` (R4 + DT-aware
combine).

**Verdict**: ✓ MECHANISM WIRED, awaits Phase D.5 tuning + 800-case ship gate

## 1. What landed

### 1.1 `_stage_gain_compute` — AEC3 path branch

[python/aec.py:3340-3409](../python/aec.py#L3340) — added an
`if self._mask_profile_swap_enabled:` branch BEFORE the legacy
continuous interpolation block:

```python
if self._mask_profile_swap_enabled:
    _use_nearend = bool(self._subband_ne_state
                        and float(effective_dt) > 0.3)
    _profile = (self._nearend_mask_profile if _use_nearend
                else self._normal_mask_profile)
    enr_t = _profile[0]                  # per-bin, from D.2 substrate
    enr_s = _profile[1]
    _emr_transparent_pb = _profile[2]    # per-bin emr_transparent
    self._diag_mask_profile_nearend = _use_nearend
else:
    # ... legacy continuous interpolation path (unchanged) ...
```

The legacy code path (continuous `ne_confidence × enr_t_ne + (1 -
ne_confidence) × enr_t_fs` + `per_band_enr` Arc R + `dt_ne_compression_fix`
v3.15 §1.2) stays intact under the `else:` branch.

### 1.2 EMR section — per-bin `emr_transparent` when AEC3 path active

[python/aec.py:3415-3422](../python/aec.py#L3415):

```python
if np.sum(self.noise_psd) > 0:
    emr = residual_echo_psd / (self.noise_psd + 1e-10)
    if _emr_transparent_pb is not None:
        g_emr = np.clip(_emr_transparent_pb / (emr + 1e-10), 0.0, 1.0)
    else:
        g_emr = np.clip(0.3 / (emr + 1e-10), 0.0, 1.0)
    g = np.maximum(g, g_emr)
```

When flag-OFF (default), `_emr_transparent_pb` stays `None` (set at top
of the gain compute block) → scalar `0.3` path = byte-equal to legacy.

### 1.3 Diagnostic state

- `self._diag_mask_profile_nearend` — last-frame bool: did NE profile
  fire?

## 2. Verification

### 2.1 Byte-equal flag-OFF (default config)

5-case `eval_aec_challenge.py` run with all D.1+D.2+D.3 changes,
flags default OFF:

| Verdict | Result |
|---|---|
| md5 hashes vs baseline (87730e4) sorted-diff | **empty** |
| Pass count | 10 / 10 |

**Default-OFF byte-equal PASS** → no observable effect when flag off.

### 2.2 Flag-ON behavioural smoke

`subband_ne_detect_enabled=True` + `res_mask_profile_swap_enabled=True`:

| Scenario | NE-profile fire % | ERLE Δ | Interpretation |
|---|---|---|---|
| NE (014Azuq...) | 18.7% | unchanged | Voiced NE segments correctly use nearend profile |
| FS (0KjzXA3g...) | 0.0% | unchanged | Echo-aware AND-gate filters out D.1 false-trigger (which was 19.9% pre-gate) |
| DT cohort tail (0I0XMl3M...) | 2.4% | -8.9 → -8.7 dB | Small NE-protect engagement, small expected ERLE softness |

**Key finding**: the AND-gate `_subband_ne_state AND effective_dt >
0.3` successfully prevents FS misfire. The 19.9% raw subband detector
fire on FS gets reduced to 0% actual profile swap (because FS frames
have effective_dt ≈ 0). NE case correctly engages the relaxed-HF
profile 18.7% of the time.

DT cohort tail (0I0XMl3M) ERLE softens by 0.2 dB — within the
"trade-off direction is correct" range. The 2.4% fire rate suggests
the cohort tail residual still doesn't pass the structural NE cue
(HF quiet, voice-band loud); Phase D.5 may need to relax the gate or
tune sub1/sub2 bins for cohort tail recovery.

## 3. What's actually changed in the gain compute

### Legacy path (flag OFF, default)

```
ne_confidence = dt_per_bin              # per-bin, from F3.1 mic-excess + (1-coh²) blend
enr_t = ne_confidence × enr_t_ne + (1 - ne_confidence) × enr_t_fs
enr_s = ne_confidence × enr_s_ne + (1 - ne_confidence) × enr_s_fs
```

- Per-bin continuous interpolation between NE and FS anchors
- NE anchors are scalar (or 3-band tilt via Arc R)
- FS anchors are scalar (with `enr_scale` config multiplier)

### AEC3 path (flag ON, new)

```
use_nearend = subband_ne_state AND effective_dt > 0.3
enr_t = nearend_profile[k] if use_nearend else normal_profile[k]
enr_s = ...
emr_t = ...
```

- Per-bin LF→HF interpolated profile (D.2 substrate)
- Binary atomic swap (matches AEC3 `LowerBandGain` two-way logic)
- HF NE protection: nearend HF `enr_s = 0.3` vs normal HF `enr_s = 0.1`
  → 3× more permissive HF when NE detected

## 4. Open question — interaction with legacy `effective_dt` consumers

35+ other consumers of `effective_dt` still operate (spectral floor,
temporal smoothing, noise floor + CNG, postprocess HF cap, etc.).
These were tuned under the continuous-interpolation regime; under the
AEC3 binary swap regime they may need re-tuning.

**Decision per v3.18 plan §D.4**: defer formal migration. The mask
profile swap mechanism is independently testable — Phase D.5 60-case
bench will reveal whether the binary swap alone improves NE/DT
buckets, even without downstream consumer harmonisation. If yes, D.4
migration becomes mandatory; if no, the substrate stays default-OFF
and we close D-γ per §0.4.

## 5. Substrate readiness checklist

- [x] AEC3-style binary mask profile swap wired
- [x] Echo-aware AND-gate (`subband_ne_state AND effective_dt > 0.3`)
- [x] EMR section uses per-bin `emr_transparent` when AEC3 path active
- [x] Default-OFF byte-equal (5-case eval, 10/10 md5)
- [x] Flag-ON behavioural sanity (FS misfire 0%, NE engages, DT slight
      ERLE softness)
- [x] Diagnostic `_diag_mask_profile_nearend` per-frame state
- [ ] 60-case AECMOS sweep — pending D.5
- [ ] Profile tuning (LF/HF anchors + sub1/sub2 bins) — pending D.5
- [ ] 800-case ship gate — pending D.6

## 6. Files modified

- [python/aec.py](../python/aec.py)
  - L2265 — `_diag_mask_profile_nearend` init (false default)
  - L3340-3409 — `_stage_gain_compute` AEC3 path branch +
    per-bin `emr_transparent_pb`

## 7. Cross-references

- D.1 substrate (subband NE detector): [docs/v3_18_d1_subband_ne_detector.md](v3_18_d1_subband_ne_detector.md)
- D.2 substrate (mask profile tables): [docs/v3_18_d2_mask_profile_substrate.md](v3_18_d2_mask_profile_substrate.md)
- AEC3 source: [docs/aec3_extracts/src/aec3/suppression_gain.cc:215-233](aec3_extracts/src/aec3/suppression_gain.cc) `GainToNoAudibleEcho`
- Phase 0.2 mapping §3.2: [docs/aec3_residual_pipeline_mapping.md](aec3_residual_pipeline_mapping.md)
- Phase 0.6 decision: [docs/v3_18_phase0_decision.md](v3_18_phase0_decision.md)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
