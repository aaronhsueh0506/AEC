# v3.18 Phase C.D.1 — leakage_diverged + RES per-state ENR (design lock, doc-only) — 2026-05-16

**Status**: C.D sub-design under [docs/v3_18_c1_aec_state_design.md](v3_18_c1_aec_state_design.md) §2 C.D.
**THIS IS THE FIRST BEHAVIOUR-CHANGING PHASE C SUB-PHASE.**

Prior C.A / C.B / C.C all audit-only (read-only mirrors). C.D **changes
how the AEC behaves under flag-ON** — so it carries the same risk
profile as A.2-A.7 (which closed CANNOT SHIP) and B.2-B.4 (silent gate).

## 0. C.D scope split — 2 sprints, each its own kill gate

Original C.1 §2.C.D bundled two ideas. Splitting per AEC discipline:

**C.D-α — leakage_diverged Q bifurcation** (smaller surface, this design)
- Use `fq_usable` (C.B) + shadow_advantage to gate WHEN to inject Q-boost
- File-disjoint from RES: touches `aec.py` Q-injection block only
- Cheap pre-bench gate: fire-rate count vs baseline

**C.D-β — RES per-state ENR refactor** (larger surface, separate sub-design)
- Per-filter-state ENR thresholds in `ResFilter._stage_gain_compute`
- File-disjoint from α: touches `python/aec.py` ResFilter stages
  (or `python/res_refactored/gain_computer.py`)
- Deferred to its own design lock if α PASSes

**This design covers only C.D-α.** C.D-β is its own sprint.

## 1. C.D-α specific scope

### 1.1 Today's Q-injection sites (audit)

| Site | Trigger | Action |
|---|---|---|
| `delay_shift` (L7090ish) | DelayEst force_delay event | `_arc_m_q_boost` + P-override |
| `EPV` (L7730ish) | EchoPathVariability fires | `_arc_m_q_boost` (+ legacy P-override) |
| `shadow_rise` (L7790ish) | shadow err > main err sustained | `_arc_m_q_boost` (+ legacy P-override) |
| `PathChangeRegimeHandler.boost_q` (L7497) | shadow << main * threshold for ≥10 frames | Decision passed to AEC; AEC fires `_arc_m_q_boost` |

All 4 sites fire when something says "filter is wrong." C.D-α adds a
5th trigger: **"refined filter SAYS it's usable (fq_usable=True), but
shadow STILL outperforms refined by large margin → refined is wrong
about being usable → re-learn."**

This is AEC3 leakage_diverged: refined says it knows; coarse contradicts.

### 1.2 New trigger

```python
def _check_leakage_diverged(self) -> bool:
    """C.D-α: AEC3 leakage_diverged trigger.

    Fires when:
      - fq_usable == True (refined claims it's trustworthy)
      - shadow_advantage > LEAKAGE_THRESHOLD (coarse contradicts)
      - Not already in EPC recovery
      - Hangover countdown == 0

    On fire: invokes _arc_m_q_boost on refined filter; arms hangover.
    """
    if not self.config.leakage_diverged_enabled:
        return False
    if self._leakage_diverged_hangover > 0:
        self._leakage_diverged_hangover -= 1
        return False
    if self.epc_active or self._regime_handler.main_paused:
        return False
    if self._aec_state is None or self._aec_state._aec_ref is None:
        return False
    if not self._aec_state.fq_usable():
        return False
    sh_adv = getattr(self, 'shadow_advantage', 1.0)
    if sh_adv < self.config.leakage_diverged_threshold:
        return False
    return True

def _apply_leakage_diverged(self) -> None:
    self._arc_m_q_boost(self.filter)
    self._leakage_diverged_hangover = (
        self.config.leakage_diverged_hangover_frames)
    self._leakage_diverged_fire_count += 1
```

### 1.3 Config

```python
leakage_diverged_enabled: bool = False
# Shadow-advantage threshold: shadow_err must be < main_err / threshold
# for the trigger to consider "shadow is better." 2.0 = shadow needs to
# be 2× better than main. AEC3 uses similar conservative ratio.
leakage_diverged_threshold: float = 2.0
# Hangover frames between consecutive fires (avoid Q-boost cascade).
leakage_diverged_hangover_frames: int = 100
```

### 1.4 Wiring

Callsite: `AEC.process()`, **after** all 4 existing EPC fire sites have
been evaluated this frame, **before** the C.A / C.B / C.C audit blocks.
This makes leakage_diverged a 5th independent trigger.

```python
# After shadow_rise block ends (around L7820), before the C.A audit block:
if self._check_leakage_diverged():
    self._apply_leakage_diverged()
    self._epc_reset_fired_this_frame = True   # propagate to FQA reset_timer
```

### 1.5 State

```python
# AEC.__init__ — lazy-init when flag enabled
if self.config.leakage_diverged_enabled:
    self._leakage_diverged_hangover = 0
    self._leakage_diverged_fire_count = 0
```

`shadow_advantage` already exists as `self._dt_analyzer.shadow_advantage`
(per existing AecState.shadow_advantage property). We can read via the
property.

## 2. Pre-bench gate (C.D-α.3)

Per cheap-gate discipline (Phase B lesson):
- Fire count ≥ 5 across 5-case smoke (less means trigger is silent;
  close cheap)
- Fire count ≤ 200 across 5-case smoke (much more means trigger is
  too aggressive; tune threshold up before bench)
- Average shadow_advantage at fire ≥ 2.5 (sanity: fires happen on real
  leakage_diverged, not random noise)

## 3. 60-case bench gate (C.D-α.4)

Hard bar (anchored from prior failures + design intent):
- **PRIMARY**: DT_static Δdeg ≥ +0.010 (DT NE preserve target — fires
  during DT when refined wrongly claims usable should unlock RES better)
- Guards:
  - DT_movement Δdeg ≥ -0.005
  - FS_static Δecho ≥ -0.010
  - FS_movement Δecho ≥ -0.010
  - NE Δdeg ≥ -0.005
- Worst-case: no single sample Δecho < -0.05 dB

**Kill criterion** (per §0.4 + 4-of-4 closure precedent):
- PRIMARY FAIL OR worst-case breach → close C.D-α; substrate retained
- If fire count is 0 (silent on 60-case) → close per Phase B precedent

## 4. Reverse-evidence risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Fires too aggressively → Q-boost cascade destabilises main | hangover=100; threshold=2.0 conservative; pre-bench fire-rate cap=200 |
| R2 | Silent gate (Phase B precedent — never fires) | pre-bench fire-rate floor=5 catches this cheap |
| R3 | Fires correctly but RES doesn't react (RES still gated by legacy `_filter_converged`) | C.D-α only does Q-boost. RES gating migration is C.E scope. Q-boost alone speeds filter re-learn → better suppression on subsequent frames → indirect RES benefit |
| R4 | qNvSMyU cohort tail interaction — qNvSMyU has 96.3% fq_usable (per C.B) but is a true non-converging case. Fires would cascade. | hangover=100 frames (~1 sec) limits cascade frequency; threshold=2.0 may not trip on qNvSMyU's high-noise regime |
| R5 | Co-tuning wall (4-of-4 precedent) | Behaviour change is contained (Q-boost only); RES + scattered consumers unchanged. Risk surface smaller than A or B |

## 5. C.D-α design confidence

**Reading: 60%** (lowest of the C-series so far).

Components:
- (+) Behaviour change is contained (Q-boost on refined only)
- (+) C.A + C.B substrate validated PASS gives this trigger meaningful inputs
- (+) Pre-bench fire-rate gate catches silent/cascade failures cheap
- (−) Lowest confidence in C-series because **first behaviour change**
- (−) PRIMARY bar (DT_static Δdeg ≥ +0.010) is aggressive — Phase A.7
  showed even with PASS guards, primary often fails
- (−) qNvSMyU interaction unknown (R4)

At the §0.4 60% kill threshold — proceed with explicit recognition
that close is plausible.

## 6. Sprint sequence

| Sprint | Action | Output |
|---|---|---|
| C.D-α.1 | This design doc | doc only |
| C.D-α.2 | Config + state + `_check_leakage_diverged` + `_apply_leakage_diverged` + wiring + 5-case byte-equal flag-OFF | md5 PASS |
| C.D-α.3 | 5-case smoke fire-rate gate | PASS/FAIL |
| C.D-α.4 | 60-case AECMOS A/B (if C.D-α.3 PASS) | verdict |
| C.D-α.5 | C.D-α verdict + decision on C.D-β | verdict doc |

## 7. Cross-references

- [docs/v3_18_c1_aec_state_design.md §2.C.D](v3_18_c1_aec_state_design.md) — parent design
- [docs/v3_18_c_b_verdict.md](v3_18_c_b_verdict.md) — C.B PASS (provides fq_usable)
- [docs/v3_18_b_closeout.md](v3_18_b_closeout.md) — Phase B closeout (fire-rate gate precedent)
- [docs/v3_18_a_closeout.md](v3_18_a_closeout.md) — Phase A closeout (worst-case bar precedent)
- [aec.py:5497](../python/aec.py#L5497) — existing PathChangeRegimeHandler.boost_q (related trigger)
