# v3.21.6 Sprint P3 — EchoAudibilityConfig structural wiring VERDICT

**Date**: 2026-05-21
**Branch**: `feature/v3_21_6_parity_completion`
**Status**: ✅ PASS — structural parity shipped; default-True behavior preserved (byte-equal)
**Plan**: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (v3.21.6 Sprint P3)
**AEC3 reference**: [`docs/aec3_extracts/src/aec3/aec_state.cc:236-241`](aec3_extracts/src/aec3/aec_state.cc#L236), [`echo_audibility.cc`](aec3_extracts/src/aec3/echo_audibility.cc), `EchoCanceller3Config::EchoAudibility`

## P3.0 — Inventory

Pre-P3 state:
- [`SuppressorConfig`](../python/modules/residual/suppression_gain.py) already had a `EchoAudibilityConfig` dataclass declared **but it was instantiated locally inside `SuppressionGain.__init__`** (`self._echo_audibility = EchoAudibilityConfig()`), not exposed at SuppressorConfig level. External code couldn't override it.
- The orchestrator's stationarity zeroing block at `_aec3_post:3500-3520` consumed the top-level [`AecConfig.aec3_post_stationarity_zero_enabled`](../python/modules/config.py#L272) flag (default `True` — load-bearing safety net per [v3.21.5 Sprint B verdict](v3_21_5_phase1_b_stationarity_gate_verdict.md)).
- Two parallel surfaces controlling closely related semantics → architectural drift.

## P3.1 — Wire EchoAudibilityConfig into SuppressorConfig

**Change**: promote the existing `EchoAudibilityConfig` (already with rich AEC3 fields: `audibility_threshold_lf/mf/hf`, `low_render_limit`, `normal_render_limit`, `use_stationarity_properties`, `lf_band_end_hz`, `mf_band_end_hz`, `floor_power`) from a SuppressionGain-internal local to a SuppressorConfig field.

Diff in [`python/modules/residual/suppression_gain.py`](../python/modules/residual/suppression_gain.py):

```python
@dataclass
class SuppressorConfig:
    ...
    use_subband_nearend_detection: bool = False
    # v3.21.6 Sprint P3 — exposed echo_audibility.
    echo_audibility: EchoAudibilityConfig = field(default_factory=EchoAudibilityConfig)
```

And the SuppressionGain `__init__` now reads from the passed-in config instead of constructing a local default:

```python
# Was: self._echo_audibility = EchoAudibilityConfig()
# Now:
self._echo_audibility = self._config.echo_audibility
```

NOTE: I initially mis-added a duplicate `EchoAudibilityConfig` class (clobbering the existing one's rich fields). Caught immediately by a smoke-render `AttributeError: 'EchoAudibilityConfig' object has no attribute 'lf_band_end_hz'` and reverted to use the existing dataclass.

## P3.2 — Refactor stationarity zeroing to read echo_audibility

Diff in [`python/modules/orchestrator.py`](../python/modules/orchestrator.py) (`_aec3_post`, two consumer sites at lines 3500 + 3511):

```python
# Was:
_need_stationary_mask = (
    self.config.trace_hf_chain
    or (self.config.aec3_post_stationarity_zero_enabled
        and _filter_converged_enough))
...
if (self.config.aec3_post_stationarity_zero_enabled and ...):

# Now:
_use_stationarity = bool(
    self._aec3_sg_config.echo_audibility.use_stationarity_properties)
_need_stationary_mask = (
    self.config.trace_hf_chain
    or (_use_stationarity and _filter_converged_enough))
...
if (_use_stationarity and ...):
```

## P3.3 — Deprecate top-level alias

[`AecConfig.aec3_post_stationarity_zero_enabled`](../python/modules/config.py#L272) stays at default `True` (preserves v3.21.5 Sprint B load-bearing safety net). Comment block updated to mark it as a **deprecated alias** propagated into `SuppressorConfig.echo_audibility.use_stationarity_properties` at orchestrator init via `dataclasses.replace` (EchoAudibilityConfig is `frozen=True`):

```python
import dataclasses as _dc
_sg_config.echo_audibility = _dc.replace(
    _sg_config.echo_audibility,
    use_stationarity_properties=bool(
        self.config.aec3_post_stationarity_zero_enabled),
)
```

Env hook `AEC_STATIONARITY_ZERO` continues to work via the alias (it sets `cfg.aec3_post_stationarity_zero_enabled`, which propagates into the nested field on init). Removal of the alias is scheduled for **v3.22 Sprint I cleanup** AFTER P4 verdict ships.

## P3.4 — Byte-equal regression

Single-case md5 comparison on FS_static `0KjzXA3g20qsd8zmSekADw_farend_singletalk`:

| State | md5 |
|---|---|
| Pre-P3 (P1-shipped, P2-shipped baseline) | `25f3098af71cac1b89590cfb0dd6ec29` |
| Post-P3 default (alias = True → echo_audibility.use_stationarity_properties = True) | `25f3098af71cac1b89590cfb0dd6ec29` |
| Post-P3 + `AEC_STATIONARITY_ZERO=0` (alias = False → propagates to nested = False) | `9d8406238eb5f0eb3da3934964c60980` (different — env override path works) |

Default-True equivalence preserved. Env override path still produces AEC3-default-off behavior through the alias.

`check_byte_equal.py` against `docs/bench/v3_21_3aadd2d_baseline/` continues to differ because of P1's algorithm change (filter_analyzer_enabled default-True ship) — that's the v3.21.5 anchor mismatch, not P3.

## Final verdict

✅ **PASS — structural parity shipped, default behavior preserved.**

- Canonical control point for stationarity-driven R² zeroing is now `SuppressorConfig.echo_audibility.use_stationarity_properties`.
- `AecConfig.aec3_post_stationarity_zero_enabled` retained as **deprecated alias** for backwards compatibility; propagated at init. Default `True` (load-bearing).
- SuppressionGain now reads its `echo_audibility` slice from the passed-in config (was hardcoded default-instance), opening the door for AEC3-canonical audibility threshold tuning in future cycles without touching SuppressionGain.

### Ready for P4

The structural surface for `use_stationarity_properties` flip experiment is now clean:
- Set `cfg.aec3_post_stationarity_zero_enabled = False` (or `AEC_STATIONARITY_ZERO=0`) → propagates to `echo_audibility.use_stationarity_properties = False`
- Compare 800-case bench vs current default-True state (post P1+P2+P3 baseline)
- If cohort-tail formant damage from v3.21.5 Sprint B has subsided (P1's FilterAnalyzer may have rescued `dominant_nearend` detection on stationary-far frames): flip default; ship as v3.21.6 candidate.
- If damage persists: close P4 as "intentionally incompatible with current PBFDKF + cohort tail; AEC3 default-off retired" (mirror of P2's verdict shape — successor in v3.22 would be intentional divergence, not parity).

### v3.22 Sprint I follow-up

Remove `AecConfig.aec3_post_stationarity_zero_enabled` entirely after P4 ships. Migrate env hook to read directly into the nested field (or retire if AecConfig field is removed and presets don't need it). Update `run_one_case.py` + `eval_aec_challenge.py` env hook to set `cfg.suppressor_xxx.echo_audibility.use_stationarity_properties` directly (will need either a SuppressorConfig field on AecConfig or a helper).
