# v3.18 Phase D.1 — Subband NE Detector Substrate (2026-05-15)

**Goal**: port AEC3 `SubbandNearendDetector` (88-line module) as
default-OFF audit-only substrate. Wires `self.res._subband_ne_state`
for Phase D.3 consumer.

**Verdict**: ✓ SUBSTRATE READY

## 1. What landed

### 1.1 AecConfig — 7 new fields (default OFF / safe seed values)

| Field | Default | Notes |
|---|---|---|
| `subband_ne_detect_enabled` | `False` | Master switch (default-OFF byte-equal) |
| `subband_ne_sub1_low` | `192` | ~3 kHz (HF "breath" region start, fs=16k n_freqs=513) |
| `subband_ne_sub1_high` | `320` | ~5 kHz (HF region end) |
| `subband_ne_sub2_low` | `32` | ~500 Hz (voice band start) |
| `subband_ne_sub2_high` | `128` | ~2 kHz (voice band end / 1st formant) |
| `subband_ne_threshold` | `0.5` | sub1 < threshold × sub2 |
| `subband_ne_snr_threshold` | `30.0` | sub1 > snr × noise floor |

Default bin indices to be re-tuned in Phase D.5 from AEC3 default
config (`echo_canceller3_config.cc`, not yet read).

### 1.2 ResFilter — 8 new state fields, 1 detector update block, 1 reset hook

- `__init__` accepts 7 new kwargs, stores as `_subband_ne_*` attrs +
  initialises `_subband_ne_state = False` ([python/aec.py:2200-2252](../python/aec.py#L2200))
- `_stage_residual_model` adds 16-line detector block under
  `if self._subband_ne_detect_enabled:` guard, after `near_psd`
  4-frame smoother update ([python/aec.py:2730-2754](../python/aec.py#L2730))
- `reset()` adds `self._subband_ne_state = False`
  ([python/aec.py:2453](../python/aec.py#L2453))

### 1.3 AEC class — 7 kwargs wired into `ResFilter()`

[python/aec.py:5694-5702](../python/aec.py#L5694)

## 2. Verification

### 2.1 Byte-equal flag-OFF (default config)

5-case `eval_aec_challenge.py` smoke (seed=0, BALANCED, fl=832, --cng,
1 case per scenario + 1 extra DT):

| Case | md5 (D.1 OFF) | md5 (baseline 87730e4) | Match |
|---|---|---|---|
| 014Azuq...nearend_singletalk_ours | e046f6eb... | e046f6eb... | ✓ |
| 014Azuq...nearend_singletalk_ours_nores | aa18c245... | aa18c245... | ✓ |
| 0I0XMl3M...doubletalk_ours | 079d9ee9... | 079d9ee9... | ✓ |
| 0I0XMl3M...doubletalk_ours_nores | 56cf8210... | 56cf8210... | ✓ |
| 0I0XMl3M...farend_singletalk_with_movement_ours | 80c130e9... | 80c130e9... | ✓ |
| 0I0XMl3M...farend_singletalk_with_movement_ours_nores | 0875d8ae... | 0875d8ae... | ✓ |
| 0KjzXA3g...farend_singletalk_ours | 7bf2b81e... | 7bf2b81e... | ✓ |
| 0KjzXA3g...farend_singletalk_ours_nores | 2b706f9d... | 2b706f9d... | ✓ |
| 49IIo03G...doubletalk_ours | 882559ce... | 882559ce... | ✓ |
| 49IIo03G...doubletalk_ours_nores | c7e8618f... | c7e8618f... | ✓ |

**10/10 md5 identical** → default-OFF byte-equal PASS.

### 2.2 Flag-ON cross-scenario fire-rate audit

Detector enabled, BALANCED preset, 4 cases:

| Scenario | Case | Fire rate |
|---|---|---|
| NE | `014AzuqPZku2004NbTTmcA_nearend_singletalk` | 26.2% |
| FS | `0KjzXA3g20qsd8zmSekADw_farend_singletalk` | 19.9% |
| DT | `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` (cohort tail) | 6.6% |
| DT | `49IIo03GZ0CYQOmeA3A0BA_doubletalk` | 35.5% |

**Observations**:
- NE 26.2% — reasonable (voiced speech 30-40% of frames, HF-quiet
  pattern detected on voiced segments)
- FS 19.9% — **too high for pure far-end**. Echo IS speech (voice from
  speaker → structural cue matches). This is exactly why AEC3 pairs
  `SubbandNearendDetector` with `DominantNearendDetector` (echo-aware)
  — substrate alone is not enough for production decision.
- DT 0I0XMl3M 6.6% — low (cohort tail residual messy; HF not quiet)
- DT 49IIo03G... 35.5% — high NE content

### 2.3 Quality implication for Phase D.3 / D.5

The 19.9% FS misfire is **expected substrate behaviour, not a bug**.
Phase D.3 wiring options:
1. **Direct gate**: `mask_profile = nearend if _subband_ne_state else normal`
   → too aggressive on FS (would relax HF suppression during pure echo)
2. **AND combine with echo-aware signal**: `mask_profile = nearend if
   _subband_ne_state AND (effective_dt > 0.3 OR shadow_dt > 0.3)`
   → echo-aware gate filters out FS misfire
3. **Soft blend**: `mask_profile = blend(nearend, normal,
   _subband_ne_state × ne_confidence)`
   → continuous interpolation

Recommendation for D.3: **option 2** — combine with existing
`effective_dt` as logical-AND gate. Matches AEC3 architecture (both
detectors must agree for nearend state).

## 3. Substrate readiness checklist

- [x] Config fields added with safe defaults
- [x] ResFilter accepts + stores params
- [x] Detector compute wired in residual stage (post-smoother, pre-gain)
- [x] Default-OFF byte-equal (5-case eval_aec_challenge, 10/10 md5 PASS)
- [x] Flag-ON state actually flips (audit fire rates non-zero)
- [x] reset() hook added
- [x] AEC class instantiation pass-through
- [ ] D.3 wiring (consumer of `_subband_ne_state`) — pending D.2 + D.3
- [ ] D.5 tuning of sub1/sub2 bins + thresholds — pending Phase D.5

## 4. Files modified

- [python/aec.py](../python/aec.py)
  - L1001-1027 — AecConfig 7 new fields
  - L2171-2178 — ResFilter __init__ kwargs
  - L2243-2252 — `_subband_ne_*` attr storage + state init
  - L2453 — reset() state reset
  - L2731-2754 — detector compute in `_stage_residual_model`
  - L5694-5702 — AEC class ResFilter instantiation kwargs

## 5. Cross-references

- Phase 0.3 mapping: [docs/aec3_subband_ne_detector_mapping.md](aec3_subband_ne_detector_mapping.md)
- AEC3 source: [docs/aec3_extracts/src/aec3/subband_nearend_detector.cc](aec3_extracts/src/aec3/subband_nearend_detector.cc)
- Phase 0.6 decision: [docs/v3_18_phase0_decision.md](v3_18_phase0_decision.md)
- v3.18 plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md`
