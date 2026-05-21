# v3.22 Sprint F — Reverb tail dead fallback CLOSED no-leverage

**Date**: 2026-05-21
**Branch**: `feature/v3_22_optimization`
**Status**: **CLOSED no-leverage**. Code substrate stays in tree under default-OFF flag `reverb_tail_dead_fallback_enabled`; production behaviour unchanged.
**Prior**:
- [G.0 unified triage verdict](v3_22_g0_triage_verdict.md) — F was the only PROCEED item (2/8 cohort, LN18k5r8 100% tail-dead + s90M7MOT 73.5%)
- [v3.21.6 P1 verdict](v3_21_6_p1_filter_analyzer_verdict.md) — FilterAnalyzer port revives reverb tail on 4/5 cohort cases but NOT on these 2 catalyst cases (analyzer preconditions fail)

**Headline**: The fallback mechanism works as designed on the FS catalyst (LN18k5r8 Δg100 = -0.16, s90M7MOT Δg100 = -0.18 on tail-dead frames) but **introduces DT damage that cannot be tuned away**. Even at 5× reduced strength (0.05 instead of 0.25), WcK0OrF retains 61 new Δg100 < −0.3 catastrophic frames — 12× the user-set strict ≤ 5 gate. The mechanism is fundamentally render-power-scaled: on DT cases, render is active and F adds R² mass to bins where NE speech coexists with echo, damaging NE-speech preservation.

## Implementation (substrate kept dormant)

3 AecConfig fields added (`reverb_tail_dead_fallback_enabled / threshold_frames / strength`); env hooks `AEC_REVERB_DEAD_FALLBACK / AEC_REVERB_DEAD_THR / AEC_REVERB_DEAD_STRENGTH` plumbed.

`self._reverb_tail_dead_counter` state in AEC.__init__ + always-on update in `_aec3_post` (single source of truth for both the production fallback path and the trace_hf_chain audit field).

Fallback injection (before stationarity zeroing block so zeroing can protect stationary NE-presence):
```python
if (self.config.reverb_tail_dead_fallback_enabled
        and self._reverb_tail_dead_counter
        >= int(self.config.reverb_tail_dead_threshold_frames)):
    _fallback_mass = (far_psd * float(self.config.reverb_tail_dead_fallback_strength))
    r2 = (r2 + _fallback_mass).astype(np.float32)
    r2_unb = (r2_unb + _fallback_mass).astype(np.float32)
```

### F v1 → v2 bug fix

The initial implementation injected only into `r2_unb` (matching the AEC3 pseudocode in the plan). Cohort sanity v1 showed **Δg100 mean = 0.0000** on LN18k5r8 catalyst (no measurable change at all). Root cause: `SuppressionGain.get_gain` consumes `r2_unb` only for the `DominantNearendDetector.update()` call (ENR comparison); the gain rule `_lower_band_gain` reads `r2` (bounded). AEC3's own reverb_tail mass goes to both bounded and unbounded R² via `residual_echo_estimator.cc`. Fixed to inject into BOTH `r2` and `r2_unb` (v2). Byte-equal preserved at default-OFF in both versions.

## Cohort sanity (strength=0.25, default)

8-case cohort = G.0 catalyst (LN18k5r8 + s90M7MOT) + 3 FS_static guards (pcb1Nh / 9xjhi / lV0kQN) + 3 DT guards (WcK0OrF / wVYSGV / xQEUtY2).

### Section 1 — gain_100 distribution OFF vs ON

| Case | Role | Frames | g100 OFF | g100 ON | Δ>+0.3 | **Δ<-0.3** | Δg mean |
|---|---|---:|---:|---:|---:|---:|---:|
| pcb1Nh | guard | 2318 | 0.4368 | 0.4157 | 0 | 54 | −0.0210 |
| **LN18k5r8** | catalyst | 2358 | 0.2791 | 0.1265 | 0 | **386** | **−0.1526** |
| **s90M7MOT** | catalyst | 2307 | 0.7038 | 0.5830 | 0 | **384** | **−0.1208** |
| 9xjhi | guard | 2188 | 0.4760 | 0.4752 | 0 | 3 | −0.0008 |
| lV0kQN | guard | 2167 | 0.8342 | 0.8339 | 0 | 0 | −0.0003 |
| WcK0OrF | guard DT_mov | 3916 | 0.7085 | 0.6931 | 0 | **69** | −0.0155 |
| wVYSGV | guard DT_mov | 3666 | 0.8474 | 0.8401 | 0 | **33** | −0.0073 |
| xQEUtY2 | guard DT_st | 3988 | 0.5559 | 0.5538 | 0 | **10** | −0.0020 |

### Section 2 — catalyst tail-dead-frame focus

| Case | Tail-dead frames | g100 OFF mean | g100 ON mean | Δg100 mean |
|---|---:|---:|---:|---:|
| LN18k5r8 | 2309 (98%) | 0.2645 | 0.1087 | **−0.1558** |
| s90M7MOT | 1548 (67%) | 0.7603 | 0.5831 | **−0.1771** |

The catalyst mechanism works: on tail-dead frames, F adds R² mass → SuppressionGain produces lower gain → echo is suppressed more. Both catalyst cases show ~0.15 absolute drop in g100 on their dead-streak frames.

### Section 3 — FS guard non-tail-dead frames (no spurious fire)

| Case | Non-dead frames | Δg100 mean | Δ<-0.3 | Δ<-0.1 |
|---|---:|---:|---:|---:|
| pcb1Nh | 2242 | −0.0003 | 1 | 2 |
| 9xjhi | 2063 | +0.0000 | 0 | 0 |
| lV0kQN | 2068 | +0.0000 | 0 | 0 |

Gate (c) PASS. F doesn't over-fire on non-dead frames.

### Section 4 — DT guard catastrophic check

| Case | Frames | Δ<-0.3 | Δ<-0.1 | Δg100 mean |
|---|---:|---:|---:|---:|
| WcK0OrF | 3916 | **69** | 70 | −0.0155 |
| wVYSGV | 3666 | **33** | 34 | −0.0073 |
| xQEUtY2 | 3988 | **10** | 19 | −0.0020 |

Gate (b) FAIL. All 3 DT cases over the strict ≤ 5 frame bar; WcK0OrF 13.8× over.

### Strength reduction test (0.05, 5× lower)

| Case | Δ<-0.3 @ 0.25 | Δ<-0.3 @ 0.05 | Catalyst dead Δg100 @ 0.25 | @ 0.05 |
|---|---:|---:|---:|---:|
| LN18k5r8 (cata) | 386 | 212 | −0.1558 | −0.0994 |
| s90M7MOT (cata) | 384 | 206 | −0.1771 | −0.0929 |
| WcK0OrF (DT_mov) | 69 | **61** | n/a | n/a |
| wVYSGV (DT_mov) | 33 | 17 | n/a | n/a |
| xQEUtY2 (DT_st) | 10 | 4 ✓ | n/a | n/a |

Reducing strength to 0.05 halves the FS catalyst gain reduction (−0.16 → −0.10) but only marginally reduces DT damage on WcK0OrF (69 → 61, still 12× over the strict gate). The mechanism damage on WcK0OrF saturates at the strength level where any meaningful FS effect remains.

## Why the DT damage can't be tuned away

F's mechanism is `r2 += far_psd * strength`. On DT cases:
- Render is active throughout (both speakers talking)
- NE-speech bins co-occur with render-energy bins
- F injects mass scaled by render → mass appears on bins where NE speech is present
- SuppressionGain sees R² inflated on those bins → gain drops → NE speech damaged

Stationarity zeroing (the load-bearing safety net per v3.21.6 P4) was designed to protect bins where far-end is stationary AND filter converged. NE speech itself is non-stationary → mask doesn't fire on NE-speech bins → F's injection survives → damage.

Possible additional gates considered and rejected:
- **Gate F by `not is_nearend_state()`**: DominantNearend mis-classifies on stationary-far conditions (P4 root cause) — fires only 0.4-15% on cohort tail DT cases → doesn't reliably prevent firing on NE speech
- **Gate F by `_stationary_block AND _filter_converged_enough`**: F's catalyst cases (LN18k5r8) don't satisfy `_filter_converged_enough` because the FilterAnalyzer (whose convergence drives this gate) is the same component whose preconditions fail on these cases. Adding this gate would make F never fire on the catalyst cases.
- **HF-band-only restriction**: catalyst Δg100 was measured at bin 100 (~3.1 kHz, mid-band). Restricting F to HF (> 4 kHz) would lose the catalyst's main effect (the FS gain drop is broad-band, not HF-localized).
- **Lower strength + tighter threshold**: as shown above, strength=0.05 still produces 61 catastrophic frames on WcK0OrF. The gate threshold (50 frames) could be raised, but raising it would fail to fire on s90M7MOT-style episodic dead-streaks.

The mechanism trades **FS echo on the 2/8 catalyst cases for DT NE-speech damage on the cohort tail DT cases**. The trade is structurally similar to Sprint E's reverse-direction failure: any residual-echo-PSD injection that operates on cohort-tail-likely frames cannot be made safe by gating because the protective mechanisms (stationarity zeroing, NE detection) themselves have known mis-classification patterns on cohort tail.

## Verdict

**Sprint F CLOSED no-leverage** for v3.22 ship set. Substrate stays in tree:
- `reverb_tail_dead_fallback_enabled: bool = False` (research toggle; mirrors P2/P4/E precedent)
- `reverb_tail_dead_threshold_frames: int = 50` / `reverb_tail_dead_fallback_strength: float = 0.25` (defaults retained for any future re-evaluation)
- Env hooks `AEC_REVERB_DEAD_FALLBACK / _THR / _STRENGTH` retained
- `self._reverb_tail_dead_counter` state retained (used by trace audit field and harmless when fallback disabled)

The G.0 triage already closed D and G.1/H.3 as no-leverage / marginal-no-leverage. With F now also closed, **v3.22 has no shippable algorithm change**.

## v3.22 ship outcome

v3.22 becomes **post-parity optimization audit** — no algorithm change, only:
- Dormant research substrate from P2 (`transparent_mode_enabled`) + E (`e_stat_aware_ne_proxy_enabled` + threshold) + F (3 fallback knobs)
- 6 new trace_hf_chain fields (D / F / G.1 / H.3 gate evidence; default-OFF preserves byte-equal)
- Sprint I config cleanup pass (inventory flags; decide subsume vs research-toggle)

Recommended cycle close: bump `__version__` 3.21.6 → 3.22, write `docs/v3_22_cycle_close.md` summarising all closures (E + D + F + G.1/H.3 marginal + G.2/3/4/5 deferred), tag v3.22 (needs user approval).

## Files

- Implementation: `python/modules/orchestrator.py` (init + counter + fallback block) + `python/modules/config.py` (3 fields) + `python/run_one_case.py` + `python/eval_aec_challenge.py` (env hooks)
- Cohort trace data: `/tmp/f_cohort/*.csv` + `analyze.py` + `analyze_lo.py` + `render.sh` + `render_lo.sh`
- Byte-equal: verified at default-OFF (single-case pcb1Nh md5 identical pre/post: `6143992ee53a9866f6fda068137603b4`)
- G.0 verdict: [`docs/v3_22_g0_triage_verdict.md`](v3_22_g0_triage_verdict.md)
