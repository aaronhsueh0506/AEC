# v3.19 Phase 0.5 — C-port debt audit (2026-05-16)

**Status**: AUDIT COMPLETE. **C-port debt: 14 production-promoted
mechanisms** between v3.10.5 and v3.16, none ported.
**Verdict**: C-port stays a **v3.20+ DEDICATED batch cycle**. Phase 4
v3.19 closeout adds any v3.19 ship to the queue; no per-cycle C-port
sub-sprints during v3.19.

## 1. C-port baseline

Most recent commit touching `c_impl/`:

```
437005c fix(p3c-phase1a-c): C parity for DelayEstimator high-PAR fast-path
2026-05-06 13:09:58 +0800
```

Anchor: C port at **v3.10.4** + `p3c-phase1a` parity fix.

Confirmed by code inspection:
- [c_impl/include/aec.h](../c_impl/include/aec.h) anchor:
  `max_delay_ms /* v3.10.4: 1024 (was 250 → 512 → 1024) */`
- [c_impl/src/aec.c](../c_impl/src/aec.c) anchor:
  `delay_buffer_ms 1024 → 2048` comment

## 2. Python ↔ C parity scan (since 2026-05-06)

Searched `c_impl/src/*.c` and `c_impl/include/*.h` for v3.10+ flag
names:

| Flag pattern | Match count in c_impl/ |
|---|---:|
| `f3_1_per_band_erl_adaptive` | 0 |
| `res_per_band_enr` | 0 |
| `shadow_state_decoupled` | 0 |
| `arc_t_cohort_detector` | 0 |
| `f_e5_enabled` | 0 |
| `diverged_reset` | 0 |
| `f_e1_enabled` | 0 |
| `f_delaytrack` | 0 |
| `shadow_mu_state_aware` | 0 |
| `use_mic_excess_evidence` | 0 |
| `epc_r_reset_enabled` | 0 |
| `mu_holdoff_no_reset` | 0 |
| `shadow_r_reset_enabled` | 0 |

**All zero matches.** No v3.10.5+ ship has been ported to C.

## 3. Outstanding ship-to-C debt (by version)

### v3.10.5 (commit `1a5a2ba`) — 3 flags
- `use_mic_excess_evidence` (F3.1 v3 per-bin mic-energy excess NE evidence)
- `epc_r_reset_enabled` (F2.3 Yang 2017 R-reset on EPC)
- `mu_holdoff_no_reset` (F2.4 mu holdoff only on fresh onset)

### v3.11.0 (commit `ff0f2c8`) — 3 flags
- `shadow_r_reset_enabled` (B5 symmetric Yang R-reset on shadow)
- `f_e5_enabled` (saturation handling extensions)
- `diverged_reset_enabled` + `diverged_reset_triple_and` (triple-AND
  gate)

### v3.11.1 (commit `2341080`) — 3 flags
- `shadow_mu_state_aware` (B6 state-aware shadow mu)
- `f_e1_enabled` (ERL clip range + far_active hysteresis)
- `f_delaytrack_enabled` (continuous delay variance tracking)

### v3.11.2 (commit `f0ca5db`) — 1 wire
- `filter_state` thread through `ResFilter` (byte-equal on the wire,
  but adds new public field that C must mirror)

### v3.13 (commit `5b1760c`) — 0 C-port debt
- E2 Path 3 raised `eval_aec_challenge.py` `max_delay_ms` 250→1024 ms.
  This is in the **bench harness**, not `aec.py`. The C port already
  ships v3.10.4 `max_delay_ms = 1024` (per c_impl anchor §1).
  **No C debt** for this ship.

### v3.14 (commits `0352ae2` + `169df3c` + `b3273de` + `f08ddbf`) — 3 flags
- `f3_1_per_band_erl_adaptive` (Arc P per-band ERL EMA)
- `res_per_band_enr` (Arc R per-band ENR with block_lf tilt)
- `shadow_state_decoupled` (Arc S-orth.A decoupled shadow Kalman state)

### v3.15 (commit `5bb2fa8`) — 1 flag
- `arc_t_cohort_detector` (Arc T cohort tail real-time detector;
  default ON, byte-equal on audio because RES preempt path stays OFF)
- C-port LOE smaller — only the detector compute path needs porting;
  no audio-output behaviour change.

### v3.16 (commit `d90efdc`) — 1 removal
- `epc_dt_cap` dead-code removal in Python. C port also has the
  legacy mechanism — removal mirrors deletion in C, **net negative
  LOE** (cleanup, not feature port).

### v3.17 — 0 ship (cycle closed CANNOT SHIP per
  [docs/v3_17_closeout.md](v3_17_closeout.md))
### v3.18 — 0 ship (cycle closed 0 algorithm changes per
  [docs/v3_18_cycle_closeout.md](v3_18_cycle_closeout.md))

### Total v3.10.5 → v3.18: **14 mechanisms** + 1 removal

| Category | Count |
|---|---:|
| Per-frame algorithm flags | 12 |
| Wire-only (filter_state thread) | 1 |
| Detector-compute-only (arc_t_cohort_detector) | 1 |
| Removals (epc_dt_cap dead code) | 1 |
| **Net mechanism delta to port** | **14** |

## 4. v3.19 implication

Per CLAUDE.md branch model:

> The C port (`c_impl/`) follows Python once an algorithm change has
> merged.

Strict reading: every v3.10.5+ Python ship should have a follow-up
C parity commit. Practical reading: cycle 3.13 / 3.14 / 3.15 each
queued multiple ships in rapid succession; C-port follow-ups got
deferred and the queue accumulated.

**v3.19 disposition options**:

### Option (A) — fold C-port into v3.19 cycle (REJECTED)

- Per-flag C port LOE: 1-3 days each (PBFDKF/ResFilter modifications,
  byte-equal verification at each).
- 14 flags × 1-3 days = 14-42 days = 3-9 weeks of dedicated C work.
- Plus any v3.19-shipped flag adds 1-3 more days each.
- **Doubles v3.19 wall-clock** without advancing v3.19 ship goal.

### Option (B) — v3.20+ DEDICATED C-port batch cycle (RECOMMENDED)

- Single dedicated cycle to port v3.10.5 → v3.19 in one batch.
- LOE: 14-17 mechanisms × 1-3 days = 3-10 weeks (same as Option A).
- BUT: batched 60-case A/B per port + single cohort 800-case at end =
  amortised bench cost.
- Allows full Python algorithm freeze for the cycle (no rebase
  thrash).
- Aligns with §0.7 merge auth — single user gate after batch
  completes.

### Option (C) — per-cycle C-port sub-sprint (REJECTED)

- Add C-port sub-sprint to every Python ship cycle going forward.
- v3.19 ship → 1-3 more C-port sprints in v3.19.
- Doesn't address the 14-mechanism backlog.

## 5. Decision

**Option (B): v3.20+ DEDICATED C-port batch cycle.**

v3.19 actions:
1. Phase 4 v3.19 closeout adds any v3.19-shipped flag to the
   C-port queue (table above gets a row per ship).
2. No C-port work during v3.19 sprints.
3. v3.19 closeout doc states: "C-port debt now N+M flags (M = v3.19
   ships)."

v3.20+ backlog gets new entry:

> **C-port batch cycle (v3.10.5 → v3.19 ships, 14+M mechanisms)** |
> 4-10 weeks dedicated | trigger: v3.19 closeout + user §0.7 auth on
> batch scope | source: v3.19 Phase 0.5 audit

## 6. Per-mechanism queue (input to v3.20+ batch cycle plan)

| # | Source ship | Flag | C-port LOE | Notes |
|---:|---|---|---|---|
| 1 | v3.10.5 | use_mic_excess_evidence | 2-3d | F3.1 v3 per-bin formula |
| 2 | v3.10.5 | epc_r_reset_enabled | 1-2d | EPC R reset symmetric |
| 3 | v3.10.5 | mu_holdoff_no_reset | 1-2d | mu schedule edit |
| 4 | v3.11.0 | shadow_r_reset_enabled | 1-2d | mirror #2 on shadow |
| 5 | v3.11.0 | f_e5_enabled | 2-3d | saturation extensions |
| 6 | v3.11.0 | diverged_reset_enabled+triple_and | 2-3d | triple-AND gate |
| 7 | v3.11.1 | shadow_mu_state_aware | 1-2d | state-aware mu |
| 8 | v3.11.1 | f_e1_enabled | 1-2d | ERL clip + hysteresis |
| 9 | v3.11.1 | f_delaytrack_enabled | 2-3d | delay variance tracker |
| 10 | v3.11.2 | filter_state thread | 1d | wire-only |
| 11 | v3.14 | f3_1_per_band_erl_adaptive | 3-4d | per-band ERL EMA (Arc P) |
| 12 | v3.14 | res_per_band_enr | 2-3d | per-band ENR (Arc R) |
| 13 | v3.14 | shadow_state_decoupled | 2-3d | decoupled shadow state |
| 14 | v3.15 | arc_t_cohort_detector | 1-2d | detector-only (byte-equal audio) |
| 15 | v3.16 | epc_dt_cap removal | 0.5d | net negative |

**Subtotal (v3.10.5 → v3.16)**: 22-37 dev-days = 4-8 weeks dedicated.

Plus v3.19 ship (Phase 1/2/3 winning combo if any): +1-3 mechanisms,
3-9 dev-days.

**Total v3.20+ batch LOE**: ~5-10 weeks.

## 7. Risk tracking

C-port debt in this volume creates two risks:

1. **Production users on C path are at v3.10.4 algorithm level.**
   Anyone running `c_impl/bin/aec_wav` lacks Phase 1+ improvements
   (BALANCED-only flags 11+12+13 above — the per-band ERL/ENR Arc
   P+R that are the v3.14 ship). Document this in v3.19 closeout.
2. **Each new Python ship widens the gap.** v3.20+ batch cycle has
   "Python algorithm freeze" precondition; if v3.20 starts a new
   Python arc before the C-port batch, the queue grows again.
   Recommend: v3.20 cycle is **C-port batch ONLY**; v3.21 picks up
   next Python arc (Volterra per Phase 0.1 / Pareto Phase 1 retry
   per Phase 1 outcome).

## 8. Cross-references

- [CLAUDE.md](../CLAUDE.md) — branch model + byte-equal Python ↔ C
  invariant
- [c_impl/include/aec.h](../c_impl/include/aec.h) — current C anchor
  v3.10.4
- v3.10.5+ commit history (per §3 above) — debt source
- [docs/v3_18_cycle_closeout.md](v3_18_cycle_closeout.md) — confirms
  v3.17/v3.18 add no debt
- `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` — v3.19 cycle plan
  (Phase 0.5 doc-only audit; no plan body change)
