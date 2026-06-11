# Track F: DT H_error Three-Layer Root Cause — Probe Results (2026-06-11)

## File Probed
`AEC/wav/aec_record/aec_record_mic_10s.wav` / `aec_record_ref_10s.wav`
Focus window: 6.2–7.5s (DT + post-DT transition)

## Layer 1+2 (Main cause): leakage_diverged pumps during DT — CONFIRMED

**Trace summary (key segment 6.27–7.10s):**

| Period | leakage_div_frac | H_error mean | disallow | dt_energy |
|--------|-----------------|-------------|---------|-----------|
| 6.27–6.42s (DT start) | 0.549–0.685 | 2.00 | False | 0.000 |
| 6.43–6.52s (rescue hangover) | 0.000 | 0.087–0.021 | **True** | 0.000 |
| 6.53–6.59s (DT mid, unprotected) | 0.323–0.619 | 0.022–0.022 | False | 0.000–0.291 |
| 6.60–6.69s (rescue hangover) | 0.000–0.117 | 0.021–0.016 | **True** | 0.262–0.113 |
| 6.70–6.82s (DT late, unprotected) | 0.089–0.743 | 0.014–0.021 | False | 0.074–0.023 |
| 6.83–7.00s (rescue hangover) | 0.000 | 0.022–0.009 | **True** | 0.019–0.000 |
| 7.00–7.50s (post-DT) | 0.000 | 0.001 (floor) | False | 0.000 |

**Key findings:**
1. `leakage_div_frac` reaches 0.55–0.74 during unprotected DT windows (55–74% of bins use diverged leakage = 1000× faster than converged)
2. `_disallow_leakage_diverged = True` fires intermittently via rescue/`_coarse_reset_hangover` path (~10-frame windows), NOT from any DT signal
3. `dt_from_energy` max = 0.291 in this recording (below the 0.3 gate threshold) — energy-based DT never trips the disallow gate directly
4. Between rescue windows (6.53–6.59s, 6.70–6.82s), diverged leakage runs freely

## Layer 3 (Secondary): Post-DT mu surge — CONFIRMED

After DT ends (7.0s+):
- H_error decays to floor: 0.001
- `leakage_div_frac → 0.000` (all converged path)
- `mu_p95 max = 619.8` at 7.x seconds

Mechanism: `mu[k] = H_error[k] / (0.5·H_error·X² + n·E²)`. With H_error = 0.001 and E² → 0 (no nearend), denominator → 0.5 × 0.001 × X² → mu = 2/X². Low-X² bins get mu >> 1 → uncontrolled W update → echo rebound.

## Root Cause

`_disallow_leakage_diverged` is ONLY armed by `_coarse_reset_hangover` (rescue path = "shadow W copied from main W after ≥N poor-coarse frames"). It is **not** connected to any DT detection signal. The rescue fires at most 3× during a 0.8s DT segment, leaving 60–70% of the DT window unprotected.

## Proposed Fix

**Location:** `orchestrator.py` lines 1725–1729 (the `_coarse_reset_hangover` / `_disallow_leakage_diverged` block)

**Design:** Add sustained-leakage detection as secondary gate for `_disallow_leakage_diverged`. When `_last_leakage_div_frac > 0.5` for ≥5 consecutive hops, this signals sustained DT (nearend falsely inflating E²_refined above shadow E²_coarse). Force converged path until the pattern clears.

```python
# Track F: sustained diverged-leakage = DT indicator → gate leakage_diverged
_ld_frac = float(getattr(self.filter, '_last_leakage_div_frac', 0.0))
_ld_ctr = getattr(self, '_leakage_div_sustained_counter', 0)
_ld_ctr = min(_ld_ctr + 1, 10) if _ld_frac > 0.5 else max(_ld_ctr - 1, 0)
self._leakage_div_sustained_counter = _ld_ctr
_dt_leakage_gate = (_ld_ctr >= 5)

if getattr(self, '_coarse_reset_hangover', 0) > 0:
    self._coarse_reset_hangover -= 1
    self.filter._disallow_leakage_diverged = True
elif _dt_leakage_gate:
    self.filter._disallow_leakage_diverged = True
else:
    self.filter._disallow_leakage_diverged = False
```

**Reset hook:** `_leakage_div_sustained_counter` must be reset in both `reset()` and `_reset_filter_derived_state()`.

## Validation Protocol (before implementing fix)
- 800-case (preset=balanced/fl=52ms/cng/j4)
- Primary: DT bucket — DT echo and DT deg
- Secondary: FS bucket — must not regress
- Audio: aec_record.wav 6.2–7.5s segment before/after

## Status
- Probe: **DONE** (2026-06-11)
- Fix: design complete, **needs 800-case validation before implementation**
- Dependency: E15 (same-hop E²_coarse timing) already shipped
