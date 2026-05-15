# v3.18 Phase F.1 — AEC3 event taxonomy + classifier (2026-05-16)

**Status**: SUBSTRATE LANDED. Hard bar met (5-case byte-equal flag-OFF).
F.2+ wires asymmetric cascade reset.

## What landed

- `AecEventType` constants (`DELAY_NONE` / `DELAY_BUFFER_FLUSH` /
  `DELAY_NEW_DETECTED`) — mirrors `webrtc::EchoPathVariability::DelayAdjustment`.
- `AecEvent` dataclass — 3-tuple `(gain_change, delay_change, clock_drift)`
  + `audio_path_changed` property. Mirrors `webrtc::EchoPathVariability`.
- `classify_epc_event(EpcEvent) → AecEvent` — pure classifier. Mapping:
  - `EpcEvent.source='delay'`       → `delay_change=DELAY_NEW_DETECTED`
  - `EpcEvent.source='epv'`         → `gain_change=True`
  - `EpcEvent.source='shadow_rise'` → `gain_change=True`
  - `clock_drift` always `False` in F.1 (no detector yet).
- `AecConfig.aec_event_classification_enabled: bool = False` — gate.
- `AEC._classified_event: AecEvent` — latest classified event (default empty).
- Classifier call sites added at the 3 `EchoPathChangeDetector` fire sites
  (force_delay / EPV / shadow_rise), each gated on the new config flag.

## Why this is the right scope for F.1

AEC3's reset cascade is asymmetric: `gain_change` only resets ERLE,
`delay_change` does a full filter+state reset (per
`docs/aec3_extracts/src/aec3/echo_path_variability.{h,cc}` + AEC3
subtractor / aec_state mapping). Our pipeline collapses all 3 trigger
sources into a single `_epc_active` boolean, so:
- shadow_rise fires the same reset path as delay shift → over-resets on
  gain glitches
- delay shifts fire the same path as gain glitches → under-resets on
  alignment changes

F.1 lands the type system + classifier without changing any consumer
logic. F.2 splits `Subtractor.HandleEchoPathChange` semantics. F.3 wires
asymmetric reset cascade behind the same flag.

## Hard bar

- 5-case byte-equal flag-OFF: PASS (md5 match vs clean-env baseline).

## Operational caveats discovered

The original `/tmp/d_eval/pathDf/` baseline (00:18 render) was
contaminated by leaked env vars during the broken D-γ 4-config bench
launch. Going forward use `/tmp/f_baseline/` (clean-env 60-case render
at 01:17) for Phase F deltas. D-γ closeout deltas remain valid because
the bench env vars were the SAME for baseline and test (both
contaminated equally → cancellation in Δ).

## Files modified

- `python/aec.py`:
  - L5163 onwards — added `AecEventType` / `AecEvent` / `classify_epc_event`
  - L1112 — added `aec_event_classification_enabled` AecConfig field
  - L6160 — added `_classified_event` init
  - L7033 — capture force_delay return + classify when flag ON
  - L7480 — classify EPV event when flag ON (post-mask)
  - L7540 — classify shadow_rise event when flag ON (post-mask)

## Next sprint (F.2)

Split `Subtractor.HandleEchoPathChange` equivalent in our code:
- Identify which state mutations in our EPC chain belong to "gain_change
  soft reset" semantics (ERLE only)
- Identify which belong to "delay_change full reset" semantics
- Add `_handle_gain_change()` / `_handle_delay_change()` helpers
- Wire behind `aec_event_classification_enabled` flag with default
  behaviour = current (do both for any source)

## Cross-references

- `docs/aec3_extracts/src/aec3/echo_path_variability.h` — AEC3 struct
- `docs/aec3_extracts/src/aec3/echo_path_variability.cc` — constructor
- `docs/aec3_extracts/src/aec3/subtractor.cc` — HandleEchoPathChange
- `docs/v3_18_plan_revision_2026_05_15.md` §3 — Phase F sprint plan
- `docs/v3_18_d_gamma_closeout.md` — pivot rationale (D-γ → F)
