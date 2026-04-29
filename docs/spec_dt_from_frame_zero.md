# Spec: DT-from-frame-0 Limit

**Status**: Documented limitation as of v3.6.0 (2026-04-29).

## Definition

A "DT-from-frame-0" case is a recording where:
- Mic signal contains near-end (NE) speech / activity from sample index 0
- Far-end loopback also active from start
- **No clean far-only window** for the linear filter to learn the echo path
  before being exposed to NE corruption

Common in real conversations where remote party starts speaking before local
mic has been silent.

## Symptoms

From [diag_linear_stability.py](../python/diag_linear_stability.py) trace on
12 worst-gap cases (post-v3.5.0):

| pattern | n / 12 | shared signature |
|---|---:|---|
| A_never_conv | 6 | `conv_far=0%`, `once_converged=False`, ERLE p50 < 0 dB |
| A_mostly_unconv | 4 | `conv_far ≤ 14%`, brief `once_converged=True` then drops |
| B_partial_conv | 2 | `conv_far` 16-31%, ERLE p50 +5~+8 dB |

In Pattern A cases (~80% of worst), `using_render_based` runs at 89-100% of
far-active frames — meaning the linear filter contributes almost nothing to
suppression; downstream PR-A (Y2-fallback) and `render_dt_gain_ceil=0.6` cap
do all the work.

## Root cause chain

Filter dynamics trace ([diag_filter_dynamics.py](../python/diag_filter_dynamics.py))
shows W (filter weights) DOES grow (W_main norm 2-15), but in wrong direction:

1. **DTD `dt_from_energy` ≈ 0 in startup**: the energy-based DTD compares
   pre-filter mic vs post-filter error. Before filter converges, error ≈ mic,
   ratio looks like FS → DTD reports `dt_from_energy ≈ 0`.

2. **mu_scale stays high (~0.7)**: filter assumes FS, full adaptation.

3. **Filter learns NE-corrupted error**: NLMS / Kalman both minimize
   `|mic - W·far|²`. With NE present, optimum W is mixture of (echo path,
   NE-correlated noise) → wrong W learned.

4. **ERL drifts upward**: trace `Y7w0W4v9` ERL: 0.26 → 0.61 over 36 seconds.
   `inst_erl > 1.5` outlier protection (v3.2 axis 1) prevents *spikes* but
   allows *drift*.

5. **Self-reinforcing**: wrong W → ERLE stays < 0 dB → `_filter_converged =
   False` → `using_render_based = True` → render-mode runs forever.

## Asymmetry: FS vs DT trade-off

Same Pattern A behavior produces opposite outcomes per scenario:

| case | pattern | filter ERLE | PR-A result |
|---|---|---:|---|
| `hVqUmGvIlk` (FS_movement) | A_never_conv | -7.8 dB | echo **+0.777** (massive win) |
| `Y7w0W4v9` (DT_static) | A_never_conv | -32 dB | echo +0.323 / **deg -0.830** |

In FS, `mic_psd × 0.5` substitution suppresses echo without NE collateral. In
DT, mic_psd contains nearend speech → suppression kills NE.

## WebRTC AEC2 / AEC3 comparison

Source-verified ([echo_canceller3_config.h](https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/echo_canceller3_config.h)):

| mechanism | v3.6.0 | AEC2 (legacy) | AEC3 (modern) |
|---|---|---|---|
| filter length (16kHz) | 52ms (PR-D1, was 32ms) | ~32-50ms NLMS | 52ms (13 blocks × 4ms) |
| initial-state Q boost | none | none | **100×** during startup |
| Q bifurcation | none | none | leakage_diverged > leakage_converged 10× |
| coarse / shadow algorithm | PBFDKF (Kalman, high-Q ratio) | none | NLMS (rate=0.7, stateless) |
| dom_nearend detector | none | none | active in initial phase |
| filter quality gate | binary `_filter_converged` | none | startup ≥ 0.4s + reset ≥ 0.2s + convergence_seen |

**AEC2** has the same DT-from-frame-0 problem but its NLP doesn't depend on
filter convergence — `EER = echo_psd / error_psd` works regardless. Pays the
cost in NE quality (DT_static deg 2.304 vs ours 2.391 at v3.5.0).

**AEC3** has multiple architectural mitigations. Doesn't eliminate the
problem but provides escape paths:
- Initial Q×100 boost → faster anchor before NE corruption
- Coarse NLMS as independent escape signal
- Reset taxonomy preserves ERL across gain changes
- DominantNearend protects suppressor in initial phase

## v3.6 attempts at mitigation

PR-D series in plan
[users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md](../../../.claude/plans/users-mingyu-desktop-novatek-se-aec-pyr-tranquil-scroll.md):

| PR | mechanism | result |
|---|---|---|
| **D1** | filter 32→52ms | **shipped v3.6.0**: DT_static echo +0.071, FS_static echo +0.085 |
| D2 | initial-state Q boost (×2 to ×10) | reverted: ALL variants breach NE deg 4.0 floor |
| D3 | true 10× Q bifurcation on shadow_advantage | reverted: shadow_advantage ≈ 1.0 in worst cases (shadow Kalman = same confusion as main); trigger rarely fires |
| D4 | this spec + stats detector | this document |
| D5 | replace shadow Kalman → NLMS | deferred; 1-2 week refactor |

## Detector (stats-only, no behavior change)

Production logging for ops debug. Fires when filter has had >2s of far-active
frames, never converged, AND ERL has drifted upward — strong signal of
DT-from-frame-0:

```python
dt_from_frame_zero = (
    far_active_blocks > 200       # 2s at 10ms hop
    and not _filter_once_converged
    and erl_estimate > 0.4        # NE-inflated ERL
)
```

Stats counter increments on each fire; surfaced via `AecStats.get_stats()`.
No behavior change — engineers / production can detect the case but no
intervention applied (existing v3.6.0 PR-A Y2 fallback handles the symptom
downstream).

## Practical advice for v3.6.x and beyond

1. **Hard ceiling on linear AEC**: per current 800-case eval, DT_static echo
   gap to AEC2 is -0.149 (was -0.233 in v3.4.0). Closing further requires:
   - Better filter algorithm (e.g., PR-D5 NLMS shadow), OR
   - NN postfilter (DTLN-AEC, TEACAEC) — see plan
     `~/.claude/plans/jazzy-brewing-castle.md`

2. **NE deg floor (4.0) is binding constraint**: at v3.6.0 NE = 4.000.
   Any future mechanism that pushes NE below this is unshippable per project
   spec. PR-D2's universal Q boost couldn't avoid this.

3. **Per-case trade-off correlation**: corr(Δecho, Δdeg) = -0.81 within DT
   bucket. Bimodal Pareto means single-knob tuning shifts trade-off curve
   without breaking it.

4. **Movement and static behave similarly**: DT_movement and DT_static show
   identical Pattern A distribution. Movement isn't the discriminator.

5. **Real fix likely structural**: replace shadow with NLMS (D5), or add a
   per-bin echo-vs-NE discriminator from a learned model.
