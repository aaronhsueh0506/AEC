# v3.21 int16² / hop-vs-block scale codebase audit

**Trigger**: 2026-05-27 discovery that the diagnostic script
`v3_21_poor_coarse_rescue_attribution.py` had a faulty `coarse_conv`
metric — used `y2 > 50*50*hop_size = 400000` directly, missing the
int16² → float² scale conversion (factor 1/32768²). This made the gate
**always False** in our float [-1, 1] pipeline, producing a misleading
"coarse_conv = 0%" reading independent of actual convergence.

**Question**: are there other sites with the same scale or
hop-vs-kBlockSize confusion?

## Two correct conversion patterns in production

### Pattern P1 — outer / "translate at point of use"
Used by code that lives in float [-1,1] sample/PSD scale and needs to
test against an AEC3 constant. Conversion formula:

```python
# AEC3 native: threshold = RMS² × kBlockSize  (int16² × samples)
# Our pipeline @ hop=160, float[-1,1]:
int16_scale_sq = 32768.0 ** 2
threshold_float = (RMS ** 2) * hop / int16_scale_sq
```

The two scale corrections are independent and both required:
- **int16² → float²**: divide by `32768² ≈ 1.07e9`
- **kBlockSize=64 → hop=160 sample count**: multiply by `hop / kBlockSize` (i.e. use `hop` directly in the formula instead of `64`)

These collapse to `(RMS² × hop) / 32768²` — the canonical form.

### Pattern P2 — inner / "pre-scale inputs, use AEC3-native constants"
Used by AEC3-pattern modules (residual/, state/) that were ported as-is
from AEC3 source. The orchestrator pre-scales spectra at the
`_aec3_post` entry:

```python
_PSD_SCALE = (32768.0) ** 2          # orchestrator.py:4211
near_psd  = abs(near_spec) ** 2 * _PSD_SCALE
far_psd   = abs(far_spec)  ** 2 * _PSD_SCALE
echo_psd  = abs(echo_spec) ** 2 * _PSD_SCALE
error_psd = abs(error_spec)** 2 * _PSD_SCALE
render_block_scaled = far_end * 32768.0   # int16-amplitude scale
```

All consumers downstream (suppression_gain.py, residual_echo_estimator.py,
state/*) work in AEC3 int16² scale and use AEC3-native constants
unchanged. The boundary is documented at
[orchestrator.py:4202-4211](../python/modules/orchestrator.py#L4202-L4211).

## Audit findings

### A. Production code — CORRECT

| Site | Pattern | Constant | Conversion | Status |
|------|---------|----------|------------|--------|
| [orchestrator.py:254-256](../python/modules/orchestrator.py#L254-L256) — overhang detector | P1 | `(200² × n) / 32768²`, `(7500² × n) / 32768²` | yes | ✓ |
| [orchestrator.py:4085-4087](../python/modules/orchestrator.py#L4085-L4087) — URO thr_30 / thr_60 | P1 | `(30² × hop) / 32768²`, `(60² × hop) / 32768²` | yes | ✓ |
| [orchestrator.py:4259](../python/modules/orchestrator.py#L4259) — `_y2_threshold = 3.73e-4` for `_coarse_conv` | P1 | `50² × 64 / 32768² × (160/64)` | yes (numeric literal annotated) | ✓ |
| [orchestrator.py:4301-4302](../python/modules/orchestrator.py#L4301-L4302) — `_y2_relaxed_threshold` | P1 | `(20² × 64 / 32768²) × (160/64)` | yes | ✓ |
| [orchestrator.py:4238](../python/modules/orchestrator.py#L4238) — `render_block_scaled = far_end * 32768.0` | P2 boundary | — | yes (boundary rescale) | ✓ |
| [orchestrator.py:4211, 4230-4234](../python/modules/orchestrator.py#L4211-L4234) — `_PSD_SCALE` upscale | P2 boundary | `_PSD_SCALE = 32768²` | yes | ✓ |
| [orchestrator.py:2424, 2446](../python/modules/orchestrator.py#L2424-L2446) — rescue threshold / hangover | time-only (no scale) | `round(5×64/hop)`, `round(25×64/hop)` | n/a (sample-count only) | ✓ |
| [suppression_gain.py:75, 79, 80, 241](../python/modules/residual/suppression_gain.py#L75-L241) — `EchoAudibilityConfig`, `_LowNoiseRenderDetector.threshold` | P2 inner | `128`, `256`, `64`, `50²×64=160000` | n/a (works in int16² space) | ✓ |
| [suppression_gain.py:233](../python/modules/residual/suppression_gain.py#L233) — `_average_power = 32768²` init | P2 inner | `32768²` initial | matches `render_block_scaled` | ✓ |
| [residual_echo_estimator.py:52](../python/modules/residual/residual_echo_estimator.py#L52) — `min_noise_floor_power = 1638400` | P2 inner | `16²×6400 = 1.64e6` | n/a (int16²) | ✓ |
| [state/erl_estimator.py:18, 50, 67](../python/modules/state/erl_estimator.py#L18) — `_X2_MIN = 44015068` | P2 inner | per-bin compare + `sum > X×n_bins` | n/a | ✓ |
| [state/subband_erle.py:17, 130](../python/modules/state/subband_erle.py#L17) — `_X2_BAND_ENERGY_THRESHOLD` | P2 inner | per-bin compare | n/a | ✓ |
| [state/fullband_erle.py:22, 141](../python/modules/state/fullband_erle.py#L22) — `_X2_BAND_ENERGY_THRESHOLD` | P2 inner | `sum > X × x2.size` | n/a | ✓ |
| [state/signal_dependent_erle.py:46, 318](../python/modules/state/signal_dependent_erle.py#L46) — `_KX2_BAND_ENERGY_THRESHOLD` | P2 inner | per-subband compare | n/a | ✓ |
| [orchestrator.py:3846](../python/modules/orchestrator.py#L3846) — `_X2_THRESHOLD = 44015068.0 * 257` (R0.4 ErleInstantaneous) | P2 inner | sum vs threshold | n/a (called inside `_aec3_post`) | ✓ |
| [aec3_scale.py](../python/modules/aec3_scale.py) — central helpers (`psd_int16_to_float`, `blocks_to_hops`, `ms_to_hops`, `per_block_rate_to_per_hop`) | both | all canonical constants | n/a (it IS the helper module) | ✓ |

### B. Production code — KNOWN BUG, FLAG-GATED, NO ACTION

| Site | Issue | Default behaviour | Fix |
|------|-------|-------------------|-----|
| [residual_echo_estimator.py:53](../python/modules/residual/residual_echo_estimator.py#L53) — `noise_gate_power = 27509562.0` | Off by 1000× (AEC3 actual = 27509.42) | default uses wrong value | gated by `use_aec3_residual_noise_gate` (default-OFF). When `True`, uses `aec3_scale.RESIDUAL_NOISE_GATE_POWER = 27509.42`. Documented as R0.2 in plan + aec3_scale.py. No action required. |

### C. DIAGNOSTIC SCRIPTS — BUG (mine, 2026-05-27)

| Site | Bug | Impact | Fix |
|------|-----|--------|-----|
| `python/v3_21_poor_coarse_rescue_attribution.py` (prior version) and `v3_21_poor_coarse_rescue_gate0.py` | `y2_thr = 50*50*hop` — missing `/32768²` AND used `0.5` (refined ratio) instead of `0.05` (coarse strict) or `0.3` (coarse relaxed) | `coarse_conv` reported as 0% always (gate never fires in float scale because `y2 > 400000` is never true with float samples in [-1,1]) | corrected in `v3_21_coarse_conv_definition_audit.py` (added: `y2_thr = (RMS² × hop) / 32768²`; ratio per AEC3 0.05 strict / 0.3 relaxed / 0.5 refined) |

### D. SEMANTIC NOTE — FFT-size port quantization (NOT a scale bug)

| Site | AEC3 reference | Our pipeline | Note |
|------|----------------|--------------|------|
| Pattern `sum > _X2_BAND_ENERGY_THRESHOLD * x2.size` (multiple state/ modules) | AEC3 n_bins = 65 (FFT = 128) | n_bins = 257 (FFT = 512) | If AEC3 constant is *per-bin*, multiplying by our `n_bins=257` is correct (which the code does). If AEC3 intended *total*, we'd over-threshold by 257/65 ≈ 3.95×. Per-bin interpretation matches AEC3 source `signal_dependent_erle_estimator.cc:263` (`kX2BandEnergyThreshold`). No action — flagged for awareness. |
| Rescue threshold: `5 blocks × 64 / 160 = 2 hops` | 5 independent decision points in 20 ms | 2 independent decision points in 20 ms | Time window matches (20 ms); decision-point granularity differs. Irreducible 10 ms hop vs 4 ms block limit. Documented as part of Task 1 in `docs/v3_21_poor_coarse_rescue_attribution.md`. |

## Conclusion

**Production code = clean.** All sites that consume AEC3-derived constants use the correct
conversion pattern (P1 outside `_aec3_post`, P2 inside). The `_PSD_SCALE = 32768²` pre-scale
at the `_aec3_post` boundary ([orchestrator.py:4211, 4230-4238](../python/modules/orchestrator.py#L4211))
correctly bridges the float-domain spectra to the AEC3-native int16² constants used in
suppression_gain / residual_echo_estimator / state/ modules.

**Only diagnostic-script bug.** The Gap C attribution script's faulty `coarse_conv` metric
was a script-only issue — it caused a misleading "0%" reading in audit output, but did not
affect any production audio. Re-running with the corrected formula will give the real
shadow PBFDAF convergence rate vs AEC3 strict / relaxed bars (in progress at
`docs/v3_21_coarse_conv_definition_audit.md`).

**Known issue R0.2 stands.** `residual_echo_estimator.py:53`'s 1000× too-large noise gate
default is flag-gated, documented, and left as-is per prior plan decision.

**FFT-size quantization (D) is awareness-only**, not a fixable parity bug — different hop +
different FFT size are deliberate v3.21 architecture choices, not strict-port targets.
