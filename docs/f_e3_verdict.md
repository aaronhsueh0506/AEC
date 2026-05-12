# F-E3 verdict — consecutive-EPC hangover + W partial reset

**Phase**: v3.11 Phase 1 Sprint 9-10
**Flag**: `f_e3_enabled` (default OFF)
**Result**: **FAIL — too aggressive on FS_static; flag stays OFF**

## Mechanism

Targets edge case **E3: echo gain change (blind test pattern)**:
- E3-1: extend EPC hangover to ≥1s on consecutive fires
- E3-3: W partial reset on 2nd consecutive EPC (gap-guarded against
  cohort-tail spam)

Detects "consecutive" when `_frames_since_last_epc < 100` (1s at hop=160 /
sr=16k). W reset has `min_gap_frames=1000` (10s) cohort guard.

## Verification

**Baseline**: v3.10.6 (`results/v3_10_5_main/`)
**Candidate**: v3.10.6 + `AEC_F_E3=1` (`results/sprint_f_e3_on/`)

### Bucket-mean Δ vs baseline

| Bucket | n | Δerle_full | Δerle_active | Δne_pres |
|---|---|---|---|---|
| FS_static    | 169 | -0.108 | **-0.110** | +0.132 |
| FS_movement  | 131 | -0.066 | **-0.068** | +0.111 |
| DT_static    | 186 | -0.034 | **-0.060** | +0.116 |
| DT_movement  | 114 | -0.022 | **-0.034** | +0.074 |
| NE           | 200 | +0.000 | +0.000 | +0.000 |

### Hard abort criteria

| Criterion | Threshold | Actual | Pass? |
|---|---|---|---|
| FS_static bucket mean | ≥ -0.02 | **-0.110** | **✗** |
| FS_movement bucket mean | ≥ -0.02 | **-0.068** | **✗** |
| DT_static bucket mean | ≥ -0.02 | **-0.060** | **✗** |
| DT_movement bucket mean | ≥ -0.02 | **-0.034** | **✗** |
| Linear ERLE bucket Δ ≥ -0.5 | OK | min: -0.110 | ✓ |
| Cohort tail qNvSMyU | ≥ -0.05 | +0.000 | ✓ |

**4 / 6 bucket-mean criteria FAIL.** Flag must not be promoted.

### Per-bucket regression count (|ΔerleA| > 0.1)

| Bucket | Improve | Regress | Δ < -0.3 | Δ < -0.5 |
|---|---|---|---|---|
| FS_static | 4 | **52** | 23 | 13 |
| FS_movement | 1 | 26 | 10 | 4 |
| DT_static | 1 | 33 | 13 | 5 |
| DT_movement | 0 | 16 | 2 | 2 |

### Trade-off pattern

All FS/DT regressions are paired with NE_pres improvements (mean Δne
+0.074 to +0.132, individual cases up to +1.935). F-E3 is making linear
stage more conservative everywhere — better NE pres at cost of weak
linear cancellation.

### Cohort-tail guard worked

`qNvSMyUSXUyrDGpOw7s6qg_farend_singletalk`: ΔerleA = 0.000. The
`f_e3_w_reset_min_gap_frames=1000` (10s) gap-guard successfully
prevented W reset from firing on this case despite frequent
shadow_rise events.

## Root cause analysis

The design assumption — that "consecutive EPC fires within 1s" indicates
a back-to-back gain change — is **wrong on this corpus**.

EPC detector's `shadow_rise` trigger fires whenever main+shadow error
rises with low delta_ratio. On normal speech with intermittent silence,
this happens frequently as a function of speech vs silence transitions,
NOT as a function of physical room/path discontinuity. So
"frames_since_last_epc < 100" is satisfied frequently in normal FS
audio.

Worst regressions are `MeQ3WL4hykKuT2761h0xFg_farend_singletalk` (-1.037)
and similar FS_static cases — no movement, no gain change, just normal
speech where shadow_rise periodically fires. F-E3 misidentifies these
as "consecutive gain changes" and applies its mitigations.

Effects:
1. Hangover extension to 100 frames keeps `_erl_estimate` capped at 0.3
   and `_epc_render_forced_remaining` armed for full second → linear
   filter cannot fully converge → ERLE stays low.
2. W partial reset (every 10s on FS_static due to gap-guard release)
   halves W; the re-learning is hindered by the simultaneously-extended
   hangover. Compounding losses across 10+ such events in a 30s file.

The cohort-tail guard (min_gap) prevented disaster on the
ERL_decile_std cases but did not save FS_static — the gap-guard is
absolute (10s) which is short enough to fire repeatedly in normal FS
where shadow_rise events are spaced 1-5s apart.

## What would need to change for F-E3 to work

1. **Distinguish path-change EPC from steady-state EPC**. The current
   detector fires on error rise, which happens for many reasons. Need a
   signature specific to gain step (e.g., `_epv_gain_fast / _epv_gain_slow`
   ratio outside [0.25, 4.0] for >5 consecutive frames, indicating
   sustained gain shift rather than instantaneous spike).
2. **Stricter consecutive window**. Currently 1s; a true mid-clip gain
   reversal pattern would be 5-15s apart, not 1s. Shorten to require
   "second fire 3-10s after first" instead of "within 1s".
3. **Gap-guard tied to f_e3 fire count, not frame count**. Cap at e.g.
   max 3 W resets per stream rather than every 10s.

These are non-trivial design changes that need their own ablation
sprint. F-E3 in its current form is incompatible with this codebase's
EPC detector.

## Decision

**Flag state**: keep `f_e3_enabled = False` (default OFF). **Do NOT
promote**. Mechanism is incorrect for production use as-implemented.

**Final review recommendation**: discard. The edge case (mid-clip
gain step then revert) is real but the detection signal needs to be
re-designed before any consecutive-EPC logic can be safely added.

Sprint outcome: ablation result is valuable — it confirms the EPC
detector fires too frequently to use as a consecutive-event signal
without additional discrimination. This is a learning, not a fix.
