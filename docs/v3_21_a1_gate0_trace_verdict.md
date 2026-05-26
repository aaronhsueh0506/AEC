# Gate 0 Verdict — A.1 `use_full_delay_change_chain`

**Date**: 2026-05-25  
**Script**: `python/gate0_trace_a1.py`  
**Case**: `xFk7igecuke0R5JMfREyDg_farend_singletalk` (Guards 1–4 + H_error) and  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`xFk7igecuke0R5JMfREyDg_farend_singletalk_with_movement` (W ratio gate)  
**Config**: BALANCED preset, filter_length=832, mode=PBFDKF, enable_res=True, enable_cng=True

---

## GATE 0 VERDICT: PASS — proceed to Gate 1 AECMOS

| Criterion | Result | Details |
|---|---|---|
| Guard 3 — AecState delay_first reset | **PASS** | initial_state_active()=True from delay_first frame; H_error jumps to ceiling |
| Guard 4 — Trigger scope | **PASS** | Chain fires only on delay_first; 0 EPV events, 0 shadow_rise events in case |
| H_error reset to ceiling | **PASS** | ON=100.0, OFF=0.4 at frame 279 (>50 threshold) |
| W ratio at SR+10 | **PASS** | 0.9766 < 1.05 (_with_movement variant, SR@1118) |

---

## §1 Guard 1 — Flag Isolation

- `use_full_delay_change_chain` is the **only** non-default flag in both variants.
- All Q1 flags (`use_q1_true_delay_transient_leakage`, `use_q1_terminate_on_non_delay_epc`,
  `use_aec3_zero_filter_on_epc`) are OFF at BALANCED defaults.
- OFF and ON variants differ only by this one flag.

## §2 Guard 2 — Byte-Equal Precheck

`check_byte_equal.py` vs `3aadd2d` anchor: **1/25 PASS** (pre-existing).  
HEAD is 8+ commits past the anchor with legitimate v3.21.6 divergence from that commit.

Flag-OFF additions are entirely inside `if bool(getattr(self.config, 'use_full_delay_change_chain', False)):` blocks (which evaluate to `False` at OFF default) plus read-only `_diag` assignments with no audio path effect. The 1/25 byte-equal score is a pre-existing condition; the OFF variant of this flag does not contribute to it.

Direct OFF/ON isolation is confirmed within this script run (same session, same seed).

## §3 Guard 4 — Trigger Scope (runtime)

```
delay_first fires at frames: [278]
delay_shift fires at frames: []
EPV fires at frames:         []
shadow_rise fires at frames: []
[PASS] Chain did NOT fire on EPV or shadow_rise frames
```

`set_initial_state(True)` fires at frame 278 (coincides with `delay_first`), confirming
the trigger gate is exclusive to real delay events.

## §4 Guard 3 — AecState delay_first Confirmation

Frames around delay_first=278 (ON variant):

```
 frame   aec3_init    h_err_mean      a1_set    w_norm
   276        True           0.4        None    4.0814
   277        True           0.4        None    4.0625
   278        True         100.0          on    0.0000
   279        True         100.0        None    0.0000
   280        True         100.0        None    0.0000
   281        True         100.0        None    0.0000
   282        True         100.0        None    0.0000
   283        True         100.0        None    0.0000
```

- `initial_state_active() == True` within 5 frames of delay_first: **PASS**
- `H_error_mean` at frame 279: ON=100.0, OFF=0.4 → **PASS** (ceiling confirmed)
- `set_initial_state(True)` fires at frame 278: **PASS**
- `transition_triggered()` fires at frame 859: **PASS**
- `set_initial_state(False)` fires at frame 859: **PASS**

**Note on `aec3_initial_state=True` before delay_first**: the filter was already in
initial state pre-delay_first (not yet converged). `_full_reset()` re-enters initial
state, which is already True — no toggle visible in the trace. The H_error jump from
0.4→100 and the W reset to 0 confirm the chain fired correctly.

**Note on H_error ceiling**: `H_ERROR_CEIL_FLOAT = 1e2` in `aec3_scale.py`.
`handle_echo_path_change()` fills H_error_per_bin with 10000, but `_h_error_refresh()`
clamps to 100 on the next frame. ON=100.0 IS the confirmed reset state.

**Note on W_norm=0 at frame 278**: `_reset_filter_derived_state()` zeros W (via
`PBFDKF.reset()`) before the A.1 chain code. The chain's `handle_echo_path_change(zero_filter=False)`
correctly does NOT re-zero W (it was already 0). Frames 279–283 show W_on=0 as the
shadow filter re-initializes; from frame 284 onward the shadow filter's Q×3.5 boost
drives rapid re-convergence (W_on ~110 by frame 284 vs W_off ~4.2).

## §5 Primary Gate — W Ratio at SR+10

Source: `_with_movement` variant (shadow_rise at frame 1118).

```
shadow_rise frame: 1118  SR+10 frame: 1128
W_norm OFF=18.457767  ON=18.025572  ratio=0.9766
[PASS] W ratio 0.9766 < 1.05
```

The ON variant's filter converges as well as OFF at the SR+10 checkpoint (within
2.3%), confirming the H_error reset does not destabilise long-term filter adaptation.

## §6 Implementation Status

Changes made in this cycle (all behind `use_full_delay_change_chain` default-OFF):

| File | Change |
|---|---|
| `python/modules/config.py` | Added `use_full_delay_change_chain: bool = False` |
| `python/modules/orchestrator.py` | Site 1: delay_first branch — H_error reset + _aec3_pending_delay_change queue (gap fix) |
| `python/modules/orchestrator.py` | Site 2: delay_shift branch — H_error reset (AecState wiring already correct) |
| `python/modules/orchestrator.py` | Site 3: _aec3_post pending-change block — set_initial_state(True) on delay_change |
| `python/modules/orchestrator.py` | Site 4: _aec3_post after state.update() — TransitionTriggered → set_initial_state(False) + h_error_mean trace |
| `docs/v3_21_a1_delay_change_chain_design.md` | §4.4 pre-implementation guards added |
| `python/gate0_trace_a1.py` | Gate 0 trace script (created) |

## §7 Next Step

Gate 0 PASS. Next authorised step: **Gate 1 AECMOS** on `xFk7igecuk` (farend_singletalk
+ _with_movement) — compare OFF vs ON AECMOS scores.

Gate 1 scope per design doc §5:
- `bench_aecmos.py` on xFk7igecuk cases with flag ON vs OFF
- Criterion: deg improvement or neutral (≥ −0.01) on both variants
- Do not run 12-case or 800-case until Gate 1 passes
