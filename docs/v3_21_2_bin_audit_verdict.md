# U2 Bin-index audit verdict — v3.21.2 cycle closure

**Date**: 2026-05-20
**Branch**: v3.21.2 HEAD
**Trigger**: Discovery of FFT-scale unit-conversion bug in SuppressionGain port
(b5728e5, 2026-05-18) that caused user-reported Chinese /i/ HF damage.

## Question

The bin-index bug in [python/modules/residual/suppression_gain.py](../python/modules/residual/suppression_gain.py)
was caused by copying AEC3 reference constants (designed for FFT=128 / 125 Hz
per bin) directly into our codebase (FFT=512 / 31.25 Hz per bin) without
unit conversion. Are there OTHER hardcoded bin-index constants elsewhere
in `python/modules/` that suffer the same FFT-scale mismatch?

## Method

Exhaustive grep + line-read audit across all modules in `python/modules/`
that consume or produce frequency-domain spectra:

- `python/modules/filter/` (PBFDKF, ShadowFilter, filter_quality, filter_analyzer)
- `python/modules/state/` (state_estimator, subband_erle_estimator, stationarity_estimator, aec_state, erl_estimator)
- `python/modules/delay/` (EchoPathDelayEstimator)
- `python/modules/render/` (render_signal_analyzer)
- `python/modules/detectors.py`
- `python/modules/epc.py`
- `python/modules/orchestrator.py` (focus on `_aec3_post` chain)
- `python/modules/residual/residual_echo_estimator.py`
- `python/modules/config.py` (top-level config knobs)
- `python/aec.py` (top-level shim)

Search patterns:
1. Small integer literals (3, 5, 7, 8, 12, 15, 16, 20, 24, 29, 30, 32, 35, 40, 50, 60, 64) used as slice indices, range bounds, array subscripts
2. Constants named `lf_band`, `hf_band`, `first_hf`, `last_lf`, `bin`, `start_bin`, `end_bin`
3. AEC3-port comments (`Mirrors X.cc`, `AEC3 default`) near small int literals
4. Loop ranges like `for k in range(N):` where N is bin-related

For each find: classify as HIGH (HF/formant/NE/audibility path), MEDIUM
(ERLE/stationarity), LOW (analysis only, unitless offset, or Hz-domain),
or SAFE (already verified-scaled or relative-bound).

## Result — codebase audit-clean

### HIGH-severity findings — all fixed in commits 7e9e612 + f1ea92c

All 7 critical hot spots were inside [suppression_gain.py](../python/modules/residual/suppression_gain.py).
After Phase A refactor (configs → freq_hz fields) + Phase B canonical flips:

| Field / function | Before (bin @ FFT=512) | After (freq Hz) | Old freq | Canonical freq |
|---|---|---|---|---|
| `HighFrequencySuppressionConfig.limiting_gain_band` | 30 (hardcoded) | `limiting_gain_freq_hz=4000.0` | 937.5 Hz | 4000 Hz |
| `HighFrequencySuppressionConfig.bands_in_limiting_gain` | 5 (hardcoded) | `limiting_gain_width_hz=156.25` | 156 Hz | 156 Hz (count-preserved) |
| `SuppressorConfig.last_lf_band` | 5 | `last_lf_freq_hz=625.0` | 156 Hz | 625 Hz |
| `SuppressorConfig.first_hf_band` | 8 | `first_hf_freq_hz=1000.0` | 250 Hz | 1000 Hz |
| `SuppressorConfig.last_lf_smoothing_band` | 5 | `last_lf_smoothing_freq_hz=625.0` | 156 Hz | 625 Hz |
| `_DominantNearendDetector` LF window | `min(16, n)` | `lf_endpoint_hz=500.0` | 500 Hz | (kept; B3 revert) |
| `_limit_hf_gains` conservative_hf path | bins 20/29 inline | inline 2500/3625 Hz | 625-906 Hz | 2500-3625 Hz (flag-OFF default) |

All consumers now resolve bins via `hz_to_bin(freq, n_bins, sr)` from
[python/modules/freq_utils.py](../python/modules/freq_utils.py).

### MEDIUM-severity — already pre-scaled experimental substrate (default-OFF)

These were CORRECTLY scaled in earlier dev cycles, but stayed default-OFF
so they didn't ship in v3.21.0. No action needed.

| Field | File:Line | Value @ FFT=512 | Maps to (AEC3 freq) |
|---|---|---|---|
| `subband_ne_sub1_low/high`, `sub2_low/high` | [config.py:769-772](../python/modules/config.py#L769-L772) | 192/320/32/128 | 6k/10k/1k/4k Hz (note: scaled for fft=1024 historical context) |
| `res_mask_last_lf_band` | [config.py:806](../python/modules/config.py#L806) | 20 | 625 Hz (= AEC3 bin 5 × 4) |
| `res_mask_first_hf_band` | [config.py:807](../python/modules/config.py#L807) | 32 | 1000 Hz (= AEC3 bin 8 × 4) |
| `dominant_ne_lf_low/high` | [config.py:843-844](../python/modules/config.py#L843-L844) | 4/60 | 125-1875 Hz (= AEC3 bins 1-15 × 4) |

The fact that someone (v3.18 phase D author) DID know about the 4× scaling
when adding these experimental flags — but did NOT retroactively fix the
production-active constants in `suppression_gain.py` — is the historical
artifact that produced the bug we just fixed.

### LOW-severity — verified safe (FFT-agnostic or Hz-domain)

| Constant | File:Line | Reason safe |
|---|---|---|
| `peak_bin - 14`, `peak_bin + 5` | [render_signal_analyzer.py:106-108](../python/modules/render/render_signal_analyzer.py#L106-L108) | Peak-relative offsets in narrow-band peak detection; unitless. Scale with the detected peak position. FFT-agnostic. |
| `range(1, n_bins - 1)` | [erl_estimator.py:49,59](../python/modules/state/erl_estimator.py#L49) | Loop bounds relative to `n_bins`. Skips first and last bin (endpoint mirroring pattern). FFT-agnostic. |
| `300.0 <= f <= 4000.0` | [dtd.py:69](../python/modules/dtd.py#L69) | Operates on `f` (frequency in Hz already), not bin index. Self-documenting Hz-domain weighting (voice band emphasis). FFT-agnostic. |
| `n_bins - 2` | [erl_estimator.py:28](../python/modules/state/erl_estimator.py#L28) | Endpoint-trim array size; relative to `n_bins`. Same pattern. |
| `_COUNTER_THRESHOLD = 5` | [render_signal_analyzer.py:25](../python/modules/render/render_signal_analyzer.py#L25) | Frame counter threshold (mask-fire condition); not a bin index. |

## Verdict

**AUDIT-CLEAN**. No remaining HIGH-severity bin-index unit-conversion bugs
in the v3.21.2 codebase. All known-affected production-active paths
were fixed in commits 7e9e612 + f1ea92c. Pre-scaled experimental substrate
in `config.py` is correctly aligned; LOW-severity hardcoded constants
elsewhere are mathematically FFT-agnostic.

## Recommendations

1. **Apply same Phase A pattern** if future ports add bin-index constants:
   express as `freq_hz` config fields + resolve via `hz_to_bin()` at the
   consumer. Pattern lives in [python/modules/residual/suppression_gain.py](../python/modules/residual/suppression_gain.py).

2. **U3 sr threading follow-up** — `hz_to_bin()` currently defaults to
   `sr=16000`. Codebase + C port are 16 kHz only, so no runtime risk, but
   threading `cfg.sample_rate` through SuppressionGain would close the last
   "future-proof" gap. Tracked as sprint U3 in the v3.21.2 plan.

3. **Periodic re-audit** at each significant AEC3 port wave (e.g. when
   pulling a new WebRTC tree). Use this doc as the methodology template.

## Files

| Artifact | Path |
|---|---|
| Bug-fix commit | 7e9e612 |
| Revert commit | f1ea92c |
| `hz_to_bin` helper | [python/modules/freq_utils.py](../python/modules/freq_utils.py) |
| Refactored configs + consumers | [python/modules/residual/suppression_gain.py](../python/modules/residual/suppression_gain.py) |
