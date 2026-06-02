# Changelog

All notable changes to this AEC implementation. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) but adapted for the
research-arc workflow used here. Each version entry links to the canonical
verdict / closure doc under [docs/](docs/) for full evidence.

Versioning: `__version__` in [python/aec.py](python/aec.py) tracks the
production-graded BALANCED preset. `v3.x.y` jumps when a new production change
ships into BALANCED; `v3.x` arc closure (collection of NEUTRAL closures + arc
documentation) bumps `x`.

Bench standard for every entry below: 800-case AEC Challenge corpus,
`preset=balanced / fl=832 (52 ms) / cng=True / -j 4`. Listen evidence cited
when verdict requires it.

---

## [Unreleased] — code hygiene (byte-equal, no algorithm change)

Behaviour-neutral cleanup on top of 3.22.2. `__version__` stays **3.22.2**
(no production behaviour change). Each commit byte-equal-verified
(`_ours` + `_ours_nores`, incl. movement bucket) before landing.

- **orchestrator dedup** (Track A): extracted `_build_sg_kwargs()` so the
  `SuppressionGain` ctor — previously copied verbatim across `__init__` and
  `_reset_aec3_post` (the two blocks were identical except the
  `SuppressorConfig`) — is built in one place (~140 duplicated lines → 2 call
  sites). Collapsed the always-true `erle_coh_gate_enabled or True` Γ²(Ŷ,Y) EMA
  guard to its real consumer set (`erle_coh_gate_enabled or
  coh_gain_floor_enabled`); the accumulators feed nothing else, so the EMA is
  skipped when neither consumer is on. 14/14 byte-identical.

- **dead-flag retire** (Track B): removed 6 closed/dud default-OFF flags + their
  gated code, plus 2 inert/dead items. Production stays byte-equal (all
  default-OFF; the OFF/default code path was preserved verbatim in every case)
  — 14/14 byte-identical. Removed flags + footprint:
  - `emura_r2_enabled` / `_alpha` / `_blend` — the cross-PSD R² EMA block +
    the `r2_emura`/`emura_r2_blend` blend branch in `ResidualEchoEstimator`.
  - `subband_wallclock_smoothing` / `fullband_wallclock_smoothing` — the per-hop
    ERLE-alpha rescaling branches in `subband_erle` / `fullband_erle` (kept the
    AEC3 per-block default alphas); also dropped the now-unused `sample_rate`
    plumbing into those estimators.
  - `use_wallclock_low_noise_render_iir` — the `_LowNoiseRenderDetector` IIR
    rescaling branch in `suppression_gain` (kept the literal 0.9 decay).
  - `erle_startup_follows_convergence` (W1') — the convergence-following startup
    branch in `erle_estimator` (kept the fixed-200-hop AEC3-strict gate).
  - `enable_lf_filter_failure_r2_injection` + 5 companions (lfinj) — the
    post-residual LF R²-injection block.
  - `filter_misadjustment_scale_p` — the `scale_p` P-scaling branch in `PBFDKF`
    (verified no caller passes True; the orchestrator `isinstance` if/else
    collapsed to one `scale_filter` call).
  Inert/dead: the two `_per_band_erl[:] = 0.1` resets (array is only ever
  `[0.1,0.1,0.1]`; init + 3 diagnostic reads kept), and the unused
  `AecState.reverb_decay()` / `get_reverb_frequency_response()` stubs. Their
  `AEC_*` env hooks in `eval_aec_challenge.py` were removed too. AecConfig: 103
  fields.

## [3.22.2] — 2026-06-02 — BALANCED: per-bin near-end blend + far-active floor −28

**Headline**: BALANCED ships `soft_nearend_blend_per_bin=True` + lowers
`min_gain_floor_far_active_db` −22 → −28 dB. The per-bin near-end blend derives
a PER-BIN `ne_w` from per-bin ENR (echo[k]/nearend[k]) so the suppression-gain
tuning is frequency-selective: near-dominant bins keep `nearend_tuning`
(gentle), echo-dominant bins use `normal_tuning` (aggressive). This lets the
deeper −28 floor cancel more echo without the broadband DT near-end cost.

**800-case (movement-agnostic) vs the −22 baseline (A)**:
- DT echo 4.042 → **4.155** (+0.113); DT deg 2.227 → 2.183 (−0.044, in-bar).
- FS echo 3.529 → 3.549; NE deg 4.047 (flat). All four ship bars met.
- per-bin recovers ~+0.058 DT deg vs the floor-alone −28 point (frequency-
  selective near protection); verified byte-identical to the
  `AEC_SOFT_NE_PER_BIN=1 AEC_FAR_ACTIVE_FLOOR_DB=-28` A/B render.
- vs refs: DT echo now 4.155 (AEC2 4.262 / AEC3 4.538); DT deg 2.183 beats
  AEC3 (1.850), below AEC2 (2.389).

`far_active_floor_db` is the documented single-knob preset axis (weak −18 /
strong −28+); per-bin blend is the base. Full default-OFF flag-campaign
evidence (incl. the cohxd reference-coherence echo lever, deferred for its
DT-deg root cause) in [docs/v3_22.md](docs/v3_22.md).

Repo hygiene this version: consolidated the v3.21/v3.22 docs into single
`docs/v3_21.md` / `docs/v3_22.md`; removed unused tooling
(`ab_compare.py`, `v3_21_6_2_hf_trace.py`, `oracle_linear_erle.py`,
`v3_21_800case_bench.py`, `v3_21_800case_report_from_json.py`,
`v3_21_byte_equal_check.py`); byte-equal check is now eval + `cmp` (see
CLAUDE.md).

---

## [3.22.1] — 2026-06-02 — P4: delay-acquire phantom-mislock guard (default ON)

**Headline**: ships `delay_acquire_protect_converged` into BALANCED (default
ON). Guards the Path-A first delay acquisition: when the linear filter is
already cancelling (>2.5 dB windowed ERLE) at the current alignment, a "solid"
late acquisition is rejected (it would reset the filter + apply a spurious large
shift). Verdict: [docs/v3_22_1_p4_delay_protect_verdict.md](docs/v3_22_1_p4_delay_protect_verdict.md).

**Root cause (audio-localised)**: the bench pre-aligns the reference (GCC-PHAT)
*and then* the in-pipeline AEC3 matched filter runs — a double-alignment. On
weak/nonlinear-echo FS+DT-movement cases the matched-filter correlation surface
is flat, so it latches a noise peak (e.g. kZogUfYc +96 ms, 9xjhi +188 ms) at
confidence 1.0, double-shifts the reference, and ERLE collapses to ~0 for the
rest of the case. Production (raw ref, matched-filter-only) does not hit this
(it locks the true 16–32 ms delay); the guard is therefore harmless in
production and recovers a bench-measurement artifact.

**800-case (vs splitcfg baseline)**: 26 cases changed, net **Σecho +2.85 /
Σdeg +0.28**; biggest wins kZogUfYc +2.06, Xv7jH2 +1.51 (FS_movement echo);
genuine DT_movement deg wins waxU01 +0.236, w0QrMw +0.193. The FS/DT echo
"casualties" (wlAXM0iD −0.36, W0zK3dv0 −0.32, JtodX −0.17) are **audio-verified
AECMOS movement-quirks** — coherence(mic,far)=0.6 echo-dominant regions where
the fix removes *echo* (not near-end); near-only frames are byte-identical. Net
positive, no genuine regression. Bucket-level effect is small (FS_movement echo
3.480→3.502) — a clean win but not an AEC3-gap lever.

**Also landed (default-OFF research substrate, byte-equal — for v3.23)**:
re-exposed `subband_wallclock_smoothing` / `fullband_wallclock_smoothing` /
`use_wallclock_low_noise_render_iir` (M_full_delay bundle isolation knobs),
`h_error_refresh_erl_floor` / `h_error_floor_override` (cold-start deadlock
probes), and the `cohxd_*` selective floor-release scaffolding. All gated on
default-False/0.0 → output unchanged with flags off (verified vs baseline to
within int16 storage rounding). New diagnostic tool `oracle_linear_erle.py`
(LS-optimal achievable-linear-ERLE ceiling, sanity-validated).

**Phase 0 re-audit conclusion (motivating context)**: the linear filter is *not*
under-converging — on the worst FS cases it reaches/beats its LS-optimal 60 ms
ceiling (median under-convergence −1.0 dB); the linear→ERLE→RES handoff is
correct (usable_linear 87–97%) and RES does not over-suppress near-end. The
genuine FS echo gap to AEC3 (~0.32) is 47% nonlinear loudspeaker distortion
(LS-optimal linear leaves it at 0.09 dB — audio-verified) + ~43% RES suppression
depth (a gain-floor Pareto trade). Our back-end is a faithful AEC3 port (AEC3 is
itself pure-DSP NLRES with no nonlinear filter). The only beat-AEC3 lever is a
Hammerstein/power-filter PSD term (v3.23 candidate).

## [3.22.0] — 2026-06-01 — v3.22: split min-gain floor (DT/NE nearend preservation)

**Headline**: ships the split min-gain floor into BALANCED (default ON), the
first production change of the v3.22 arc. Closes the DT/NE nearend
over-suppression gap while holding echo above AEC2 — the only configuration
that meets all four ship thresholds simultaneously.

**Root cause (audio-localised)**: in doubletalk the RES Wiener gain over-
suppresses near-end by up to −8 dB (the linear filter only cuts −1.4..−2.9 dB).
Mechanism: near-end inflates the error → ERLE drops → R² spikes → the AEC3
min-gain floor `min_echo_power / R²` collapses to ~0, exactly when near-end is
present. A coherence-based double-talk discriminator was proven unworkable
(single-lag X–Y coherence cannot separate reverberant FS from DT — offline
oracle + audio). The working lever is a Pareto floor on the deepest suppression.

**Mechanism**: power-domain min-gain floor split by far-end activity —
* far-active (FS/DT): −22 dB. Caps the DT gain-collapse; the only FS cost is
  deep echo suppression below ~−40 dB (already inaudible), so FS echo stays
  above AEC2.
* far-silent (pure NE): −12 dB. Lifts NE near-end at zero echo cost (no echo
  present to leak).
Routing uses a per-recording latch on instantaneous far energy
(`mean(render_block²) > 1e6`, render int16-scaled; ≈ the orchestrator's own
far-active criterion). Latches from the first far frame so FS/DT use the gentler
floor throughout (no cold-start leak); only recordings where far is never active
keep the strong floor. Config: `min_gain_split_floor_enabled` (default True),
`min_gain_floor_far_active_db` (−22), `min_gain_floor_far_silent_db` (−12),
`min_gain_far_latch_power` (1e6) in [config.py](python/modules/config.py);
applied in [suppression_gain.py](python/modules/residual/suppression_gain.py)
`_get_min_gain` / `get_gain`.

**800-case (combined FS=300 / DT=300 / NE=200, vs cp_05 default-ON stack
E1+x2+E2+D3+L1+C′)**:

| metric | cp_05 | v3.22.0 | AEC2 | AEC3 | threshold |
|---|---|---|---|---|---|
| FS echo  | 3.726 | 3.520 | 3.484 | 3.875 | > 3.5 ✓ |
| DT echo  | 4.461 | 4.042 | 4.262 | 4.538 | > 4.0 ✓ |
| DT deg   | 1.952 | 2.226 | 2.389 | 1.850 | > 2.2 ✓ |
| NE deg   | 3.906 | 4.047 | 4.098 | 3.454 | ≥ 4.0 ✓ |

Only configuration to meet all four (AEC2 3/4, AEC3 2/4). vs v3.21 baseline:
FS echo +0.09, DT deg +0.13, NE deg +0.11. Audio-validated: DT near-end
recovered +1.9..7.8 dB with no echo leak in far-silent gaps; NE +1..4 dB.

**Audit by-products** (no production change): refuted via data two "critical"
claims — PBFDKF H_error step-size collapse (H_error spans 1e-3..2.0 during
movement = correct Kalman behaviour) and a delay-histogram movement latency
(the worst FS_movement case has zero mid-case delay shifts). The one verified
unused lever is the reverb accumulator's ~2.9× steady-state under-weight,
deferred as an approach-AEC3-echo stretch (echo↑/deg↓).

---

## [Unreleased] — 2026-05-29 — v3.21 CLOSE: conversion audit + Tier-C verdict + alignment-flag inlining (byte-equal)

**Headline**: closes the v3.21 AEC3-alignment arc. A deep audit of every
hop/fft "physical-meaning conversion" flag found two shipped `default-True`
flags were mislabelled "AEC3-strict" but actually mis-derived, and two were
unvalidated gray-zone conversions. The 800-case **Tier-C validation**
(`docs/v3_21_tierc_validation_report.md`) adjudicated each via the
matched-magnitude AECMOS Pareto, then all surviving alignment flags were
inlined into their call sites and all NOSHIP substrate removed. **No
production behaviour change — byte-equal preserved** (12-case `_ours` +
`_ours_nores` md5 gate, all buckets incl. movement).

### Tier-C validation verdict (800-case, isolated per flag)

- **`active_render` correction → REVERTED.** The arithmetically-correct AEC3
  value (`100²/32768² = 9.31e-6`, mean-correct vs the shipped `5.96e-4` which
  kept AEC3's block-SUM ×64 against our per-sample mean) **regressed** FS echo
  −0.033 mean + 25 catastrophic cases (worst −1.431). Production keeps
  `5.96e-4`, **relabelled as an empirically-tuned threshold, NOT strict
  alignment**; re-tuning for our hop/fft deferred to v3.22 Arc 4.
- **`fft_density` floor scaling → KEPT (inert).** 4× vs 1× ≈ 0.000 across all
  buckets. The shipped factor is `(fft/2)/64 = 4×`; the true per-bin energy
  basis is the real frame `2×hop = 320 → (2×hop)/64 = 5×` (broadband; 25×
  tonal). The 4× is a single-constant approximation that is AECMOS-insensitive;
  derivation comment corrected, signal-adaptive floor deferred to v3.22 Arc 3.
- **`reverb_smoothing` (EMA-α 0.2→0.428) → KEPT converted.** Reverb is an
  envelope/decay tracker (preserve wall-clock τ is the aligned choice), same
  class as `dne_trigger` — not a ratio estimator like the C1-C3 ERLE EMAs.
  Inert on AECMOS (0.000); keeping it preserves byte-equal.
- **`dne_trigger` (evidence count 12→5 hops) → KEPT converted.** Reverting
  tanked DT deg (−0.065/−0.075 + 51 catastrophic, 47 of them DT) for an FS
  echo gain — a losing matched-magnitude trade. The conversion is a correct
  genuine-temporal (trigger-latency) alignment.
- **Combined "honest-alignment" config (V5) → REJECTED** (DT deg −0.062 + 49
  catastrophic) — validation prevented shipping a DT regression.
- Earlier **C1-C6 wall-clock-EMA bundle → NO-SHIP** (FS echo down / illusory DT
  lift); root cause: ratio estimators (ERLE) want preserve-count, not
  preserve-seconds.

### Release cleanup (byte-equal)

- **`config.py` 412 → 230 lines.** Removed all 16 AEC3-alignment / NOSHIP /
  temp config flags. The 9 shipped `default-True` alignment flags are now
  hard-coded into their `orchestrator.py` / `filters.py` call sites; the
  C1/C3/C4/C6 + `just_reset` (CLOSED −8 dB) + `block_energy` (dormant) + the
  temp `active_render_threshold_aec3_corrected` validation flag are deleted.
- **`filters.py` 1057 → 863 lines.** Collapsed PBFDKF `_update_weights`: the
  legacy P-denominator Kalman body (default-OFF, 131 lines) is removed and the
  10 always-True refined/shadow AEC3-parity flags (`_use_aec3_h_error`,
  partition-summed X², current-E²-refined, per-bin H_error refresh, filter
  noise gate; the five PBFDAF coarse-filter gates) are inlined. Orphaned
  kx/p53 trace scaffolds removed.
- **`orchestrator.py`**: init + reset construction paths collapsed (fft-density
  if/else, dne `_dc.replace`, gain-ratchet, reverb), `_aec3_post` flag branches
  inlined (x2-reverb-for-ERLE, subtractor-max-abs, active_render → single
  value, just_reset retired, block_energy dropped). `epc.py` C6 gate → legacy
  1e-4.
- Satellite estimators (`state/subband_erle`, `state/fullband_erle`,
  `residual/suppression_gain`, `residual/reverb_frequency_response`) retain
  their internal parameterisation, now driven by hard-coded orchestrator
  literals (config-level cleanup complete; deeper leaf-branch removal is an
  optional follow-up).

### Canonical 800-case scorecard (shipped version, this entry)

Rendered via `eval_aec_challenge` (per-case `np.random.seed(0)`, ref pre-aligned
by `estimate_delay`, max_delay 1024 ms) and scored with
`model/Run_1663915512_Stage_0.onnx` (FastAECMOS). Cross-checked against the
in-memory Tier-C `V0_prod` to within ±0.011 echo / ±0.002 deg (CNG-seed margin
only — confirms the two render paths agree). Full per-bucket worst-20:
`results/v3_21_close/result.md`.

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.491 | 4.999 |
| FS_movement | 131 | 3.351 | 5.000 |
| DT_static | 186 | 4.371 | 2.059 |
| DT_movement | 114 | 4.247 | 2.159 |
| NE | 200 | 4.998 | 3.941 |

Collapsed across movement (FS/DT each = static + movement, case-weighted from
the same `scores.json`):

| Bucket | n | echo (↑) | deg (↑) |
|---|--:|--:|--:|
| FS | 300 | 3.429 | 4.999 |
| DT | 300 | 4.324 | 2.097 |
| NE | 200 | 4.998 | 3.941 |

### Repo housekeeping (this entry)

Removed closed-arc cruft: `docs/v3_21_6_2_trace.md` (superseded 3-case trace),
`docs/v3_21_close_handoff.md` (AI-continuity, folded into project memory + this
entry), `python/v3_21_tierc_validation_bench.py` (its V1–V5 variants toggle
now-inlined flags → no longer runnable; verdict captured in the validation
report). Also pruned ~1.2 GB of stale render output (`out_python/`,
`out_v3_21_800case/`, stale `results/` subdirs, `listen_ab/`). Kept
`python/v3_21_6_2_hf_trace.py` as the painted-black HF reference tracer for v3.22
Arc 2 (per the new diagnostics-discipline directive in `docs/v3_22_plan.md`:
future diagnostics fold into modules as flag-gated instrumentation, not
standalone trace scripts). `run_one_case.py` re-verified on current code
(deterministic; differs from the 800-bench only by the bench's ref pre-align).

Evidence: `docs/v3_21_tierc_validation_report.md`,
`docs/v3_21_800case_bench_report.md`, `docs/v3_21_alignment_roadmap.md`.
v3.22 roadmap: `docs/v3_22_plan.md`.

---

## [3.21.6.4] — 2026-05-28 — physical-meaning fix: ReverbModel decay wall-clock alignment

**Headline**: v3.21.6.3 trace on user's case (568_EVB_online) reduced
reverb share 64.4 % → 51.2 % via the FullBandErleEstimator hold wire-up,
but the 6.2-7 s severe-wipe segment still showed gain_hf=0.49 unchanged
because the gap between convergence events exceeded the 400 ms hold and
quality dropped to None. Re-audit of `ReverbModel::UpdateReverb` revealed
the actual physical-meaning bug: `decay = 0.83` is the AEC3 per-block
(4 ms / 64-sample) multiplier, but our pipeline calls the same update
once per 10 ms / 160-sample hop. Applied verbatim, our wall-clock T_60
was 371 ms vs AEC3's 148 ms — **2.5× too long**. When the filter is
unconverged and `tail_response` is held stale, this inflated the steady-
state reverb mass by ~2.2× via `reverb_ss = injection / (1 - decay)`
(5.88 × at 0.83 per hop vs 2.66 × at the corrected 0.624 per hop).

### Fix

`python/modules/residual/residual_echo_estimator.py` `_reverb_decay()`:
apply wall-clock alignment to both static-config decay and adaptive
estimator output:
```python
_AEC3_BLOCK_SAMPLES = 64
if self._hop_size != _AEC3_BLOCK_SAMPLES:
    d = d ** (self._hop_size / _AEC3_BLOCK_SAMPLES)
```
At our hop=160 this is `0.83 ** 2.5 ≈ 0.624`. The ratio is computed
from `_hop_size` at runtime so the conversion auto-scales if hop_size
ever changes.

`python/modules/residual/residual_echo_estimator.py` `ReverbConfig`
docstring updated: `decay = 0.83` is now correctly noted as the AEC3
per-block constant with per-hop derivation documented at the call site.

### Verification (demo case 0I0XMl3M DT_static)

v3.21.6.3 (hold wire-up only) → v3.21.6.4 (decay wall-clock fix):
- `ENR HF median` (R²/Y²): **0.945 → 0.741** (−21.6 %)
- `ENR HF mean`:           **1.457 → 1.073** (−26.4 %)
- `R²_reverb median`:      **+58.13 dB → +52.32 dB** (−5.8 dB)
- `reverb share`:          **51.2 % → 21.7 %** (−29.5 pp)
- `hf_lim_applied`:        **30.6 % → 8.4 %**
- `gain_hf_med` post-cap:  **0.036 → 0.075** (+0.039)

### What this doesn't fix

The wall-clock-aligned decay reduces R²_reverb steady-state but FS
echo cancellation strength is also reduced proportionally (faster
decay = less tail mass at steady-state to suppress real reverb).
Trade-off needs 800-case bench to confirm cohort-level Pareto. The
shipped change is AEC3-strict physical alignment — not tuning —
so it stays default-ON without flag.

### Files

- `python/aec.py`: `__version__ = "3.21.6.4"`
- `python/modules/residual/residual_echo_estimator.py`: `_reverb_decay()`
  wall-clock alignment + `ReverbConfig` docstring
- `CLAUDE.md`: version bump
- `CHANGELOG.md`: this entry

---

## [3.21.6.3] — 2026-05-28 — strict AEC3 wire-up: FullBandErleEstimator → ReverbFrequencyResponse

**Headline**: HF-painted-black per-frame trace on the user's internal case
(568_EVB_online) attributed the symptom to inflated R²_reverb (64.4 %
share, +56.6 dB int16²) — the reverb tail held STALE from a brief early
convergence and never refreshed during sustained NE. Root cause is a
v3.21 port wire-up gap: `ReverbFrequencyResponse.Update()` was fed a
binary `1.0 if converged else None` filter quality proxy instead of the
AEC3-strict continuous estimate from `FullBandErleEstimator`. AEC3 keeps
quality alive for ~400 ms past per-frame convergence via the hold
counter, extending the reverb refresh window so the tail tracks the
current filter state instead of freezing.

### Fix

`python/modules/state/aec_state.py`: add
`get_inst_linear_quality_estimate()` accessor sourcing per-frame quality
from `ErleEstimator._fullband.get_inst_linear_quality_estimate()`. AEC3
strict path: `aec_state.cc:286-289` →
`reverb_model_estimator.cc:58-66` → `reverb_frequency_response.cc:88`.

`python/modules/orchestrator.py` (`_aec3_post`): replace
```python
_converged_for_reverb = bool(_aec3_converged and _filter_converged_enough)
_filter_q = 1.0 if _converged_for_reverb else None
```
with
```python
_filter_q = self._aec3_state.get_inst_linear_quality_estimate()
```

`python/v3_21_6_2_hf_trace.py`: expose continuous `filter_quality` in the
per-frame snapshot + Flag-fractions block so the 400 ms hold can be
verified directly.

### Verification (single-case smoke, 0I0XMl3M DT_static)

- `filter_quality` alive **13.6 %** of frames vs `filter_converged` only
  **2.0 %** — hold extends live-window 6.8×, exactly the AEC3 behaviour.
- `R²_reverb (HF)` at SYMPTOM: **+58.00 dB → +52.11 dB** (−5.9 dB).
- Per-band gain at 4-6 s window: LF/MF/HF **0.81 / 0.56 / 0.66 → 0.88 /
  0.63 / 0.68**.

### What this doesn't fix

Changes WHEN reverb update fires (extends 400 ms hold) but not HOW the
tail is recomputed during sustained no-convergence. On cases where the
PBFDKF never converges for > 400 ms quality goes None and the tail
freezes. If the symptom persists, escalation candidates:
- shorten reverb decay via `ep_strength.nearend_len < default_len`
  (AEC3 has both at 0.83 by default);
- explicit stale-reverb suppression gate at the `AddReverb` call site.

---

## [3.21.6.2] — 2026-05-28 — AEC3 alignment audit + 4 shipped items

**Headline**: Full audit of the `docs/v3_21_alignment_roadmap.md` "still missing"
list against the actual codebase. 80 % of the listed items turned out to be
already-aligned, no-consumer additions, or no-op in our `current=max=13`
steady state. The audit closes 4 items that genuinely needed shipping; the
remaining gaps either require an architecture redesign (Tier C #11
`convergence_seen` latch) or carry too much risk to ship inside v3.21.x
without a dedicated design lock (Tier A #2 `IsRenderTooLow` /
`use_stationarity_properties = false`). `__version__` bumped to 3.21.6.2.

### Audit findings

| Roadmap item | Status after audit |
|---|---|
| Tier A #2 EchoAudibility config defaults | Already AEC3-strict (`floor_power = 2·64`, `audibility_threshold_{lf,mf,hf} = 10.0`, `low_render_limit = 4·64`, `normal_render_limit = 64.0`). Orchestrator override `use_stationarity_properties = True` retained as load-bearing safety net (AEC3 default is `false`; flipping is a Phase-2 design item, not shipped). `IsRenderTooLow` + `non_zero_render_seen_` latch NOT shipped — would change stationarity-noise-floor update timing on every frame; too high-risk for an unbenched bundle. |
| Tier A #3 SubtractorOutputAnalyzer | Strict AEC3 surface has only 3 signals (`any_filter_converged`, `any_coarse_filter_converged`, `all_filters_diverged`). `any_filter_converged` already wired (`bridge.filter_converged`). **Shipped**: added `any_coarse_filter_converged` (relaxed predicate `e²_c < 0.3·y²` with `kConvergenceThresholdLowLevel = 20²·hop`) and `all_filters_diverged` (`min(e²_r, e²_c) > 1.5·y²` with `30²·hop` threshold) to `FilterStateBridge`. No production consumer today (TransparentMode HMM is retired); zero-impact additive surface for Phase 2 HMM port. |
| Tier A #5 FullBandErleEstimator | Already fully ported (`python/modules/state/fullband_erle.py`) and wired via `ErleEstimator.update()`. **Shipped**: removed duplicate inline `_update_erle_inst_quality` from `orchestrator.py` (dead code — the `use_aec3_erle_reverb_quality` flag was retired in v3.21.6.1 cleanup). |
| Tier B #8 `poor_coarse_filter_counter` hangover | **Shipped, behavioural change**. Two physical-meaning corrections vs the pre-audit code: (a) trigger threshold `_threshold_hops` was 5 hops = 50 ms wall-clock; AEC3 strict `< 5` blocks = 20 ms → `blocks_to_hops(5, 160, 16k) = 2 hops`; (b) hangover was 40 AEC3 blocks (16 hops, 160 ms); AEC3 strict `coarse_reset_hangover_blocks = 25` → 10 hops, 100 ms. Also dropped the non-AEC3 `shadow_mu_scale = 0.0` freeze during hangover (AEC3 keeps coarse adapting, only the refined filter disallows `leakage_diverged` — see `subtractor.cc:264-307`). 0.5× safety margin on the trigger predicate retained (the strict `e²_r < e²_c` rule was 12-case Pareto-FAIL — `docs/v3_21_poor_coarse_rescue_12case_verdict.md`). |
| Tier C #9 `use_aec3_zero_filter_on_epc` | **Shipped (documentation port)**. Added `PBFDAF.zero_filter_partitions(old, new)` mirroring AEC3 `AdaptiveFirFilter::ZeroFilter(old, new, &H_)` (`adaptive_fir_filter.cc:460-472`). Steady-state semantics: `current_size = max_size = 13` → call is a no-op. Existing call sites (all `zero_filter=False`) unchanged. Documented `W.fill(0)` branch as PBFDKF-specific divergence not parity. |
| Tier C #10 `use_aec3_epc_classification` | No-op (depended on #9 redesign). |
| Tier A #1 DominantNearendDetector | (rescinded 2026-05-28) — already ported inline at `suppression_gain.py:311`. |
| Tier A #4 TransparentMode HMM | Deferred to Phase 2 with `SubtractorOutputAnalyzer` (consumer of new bridge signals). |
| Tier A #6 ScaleFilter (exact) | Out of scope for this audit — retiring `FilterMisadjustmentEstimator` requires its own ablation arc. |
| Tier B #7 4-mode HF tuning | Out of scope — needs separate per-band re-tune cycle. |
| Tier C #11 `convergence_seen` latch redesign | Architecture work; not in this audit. |

### Byte-equal verification (single-case smoke)

Cohort case `0I0XMl3M0ECO0U1N0cJvpg_doubletalk` (~42 s DT_static).
- Tier A #5 dedup + Tier A #3 additive + Tier C #9 doc port (alone): md5 `b898fc57f094db6da10891b0f606240a` — **byte-equal vs v3.21.6.1 baseline**.
- All four shipped items: md5 changes (Tier B #8 hangover is the only behavioural delta; expected).

### Physical-meaning alignment (user directive 2026-05-28)

All AEC3 wall-clock conversions audited under hop=160 / block=320 / fft=512:
- `coarse_reset_hangover_blocks = 25` → 100 ms → `blocks_to_hops(25, 160, 16k) = 10 hops`.
- `poor_coarse_filter_counter < 5` blocks → 20 ms → `blocks_to_hops(5, 160, 16k) = 2 hops`.
- `kBlocksToHoldErle = 100` → 400 ms → `int(0.4 × HOPS_PER_SECOND) = 40 hops`.
- y² thresholds (`50² / 20² / 30²` × kBlockSize) → scaled by `hop/block = 160/64 = 2.5×` for our hop sums.

### Files

- `python/aec.py`: `__version__ = "3.21.6.2"`
- `python/modules/filter/filter_state_bridge.py`: `FilterStateBridge` adds `any_coarse_filter_converged` + `all_filters_diverged`; `build_filter_state_bridge()` accepts both.
- `python/modules/orchestrator.py`: computes relaxed coarse / strict diverged signals next to the existing strict-converged block; threads into bridge; removes inline R0.4 dead code; corrects poor_coarse hangover physical-meaning + drops shadow-freeze divergence.
- `python/modules/state/aec_state.py`: TransparentMode branch sources `all_filters_diverged` from bridge (falls back to legacy heuristic for back-compat).
- `python/modules/filters.py`: adds `PBFDAF.zero_filter_partitions(old, new)`; documents `W.fill(0)` ablation surface.
- `CLAUDE.md`: version bumped.
- `docs/v3_21_alignment_roadmap.md`: updated with audit verdict per item.

---

## [3.21.6.1] — 2026-05-27 — AEC3 alignment completion (nores artifact + HF deficit fixes) + release cleanup

**Headline**: Two production-blocking defects fixed on the v3.21.6 baseline, then full release cleanup (config / orchestrator / dev-trace removal). `__version__` bumped to 3.21.6.1.

### Bug 1 — nores LF artifact (linear-filter over-estimation)

PBFDKF refined-filter weight-update parity gaps caused LF (0–500 Hz) over-modelling: residual carried inverted-phase echo instead of cancelling cleanly. Five AEC3-strict fixes shipped, all now hard-coded into the call sites (no flags):

- `RefinedFilterUpdateGain` denominator uses `SpectralSum = Σₚ|X_buf[p]|²` (was: `X²_latest` of current hop only). AEC3 `refined_filter_update_gain.cc:103-107`.
- Per-bin H_error refresh from `|error_spec|²` per bin, gated `e2_refined_per_bin ≤ e2_coarse_per_bin` (was: scalar fullband leakage compare). AEC3 `cc:128-138`.
- H_error ceiling `= 2.0` in float (was: `= 100.0` legacy int16-derived). AEC3 `kHErrorCeiling`.
- Refined filter noise gate constant = `20075344` (was: `27509562` borrowed from suppression path — 1.37× too tight). AEC3 `refined.noise_gate` + `coarse.noise_gate`.
- μ denominator `n_part·E²` term uses current-block `|error_spec|²` (was: smoothed `_error_psd` α=0.95 with ~200 ms lag). AEC3 `SubtractorOutput.E²_refined`.

Matching PBFDAF shadow protection (A.1–A.5: partition-summed X² for shadow μ / noise gate / poor-excitation gate / narrowband mask / saturation gate) also hard-coded ON. AEC3 `coarse_filter_update_gain.cc:34-82`.

### Bug 2 — painted-black HF (residual chain over-suppression)

Internal DT cohort case showed > 4 kHz bins painted black (gain_hf_median < 0.1) on 24.3% of frames; reverb-to-direct R² ratio median 6.2 / max ~10⁴. Ten alignment commits brought reverb chain + back-end to AEC3 strict:

- **Reverb-render-history bug**: `ReverbModel` was fed current render PSD instead of the render block from N+1 partitions ago. Added `_reverb_render_history` deque (`maxlen=16`); read `[n_partitions + 1]` for linear mode, `[filter_delay_blocks + 1]` for nonlinear mode. AEC3 `residual_echo_estimator.cc:367-376`.
- **RFR neighbour window**: reverted from ±4-bin expansion back to AEC3-strict ±1-bin point-wise max. Wider window inflated sparse HF energy.
- **ReverbConfig defaults**: `decay = 0.83` (was 0.85), `mild_decay_scale = 1.0` (was 0.5), `use_adaptive_decay = False` (was True). AEC3 `ep_strength.default_len = 0.83`.
- **Audibility band boundaries**: `lf_band_end_hz = 375.0`, `mf_band_end_hz = 875.0` (was: 94 / 219 Hz from legacy 65-bin port). AEC3 `WeightEchoForAudibility` at fft=128.
- **HF cap anchor**: 2000 Hz (was: 4000 Hz) with 1-bin physical width 31.25 Hz (was: 125 Hz). AEC3 `limiting_gain_freq_hz`.
- **CNG strict alignment**: AEC3 3-state N2 estimator (Y² EMA → N2 update with 50-block onset / 1.0002 slow-up / 0.9·0.1 track-down → N2_initial 1000-block transient → noise floor clamp) + `sqrt(1 − G²)` injection (no 0.4 scale) + LCG random + `sqrt(2)·sin` LUT phase. AEC3 `ComfortNoiseGenerator`.
- **OLA synthesis window**: MATLAB-canonical sqrt-Hann (denom `N`) replacing numpy `np.hanning` (denom `N − 1`).
- **Stationarity window**: auto-rescales via `blocks_to_hops(13/12, hop, sr)`. Was: literal 13/12 raw hops treating AEC3 4 ms blocks as 10 ms hops (130 ms vs intended 52 ms).
- **Delay estimator**: `detect_pre_echo = True` (AEC3 strict per `echo_canceller3_config.h:73`).
- **FilterPlateauDetector**: moved to opt-in flag (Python-only safety net, no AEC3 equivalent). Default off for strict alignment.

### Cohort DT case progression (internal test)

| Metric | v3.21.6 baseline | + reverb fix | + full alignment |
|---|---:|---:|---:|
| Painted-black HF (`gain_hf_median < 0.1`) | 24.3% | 15.1% | **14.0%** |
| Healthy frames | 60% | 75% | **76.3%** |
| Rev/direct R² ratio (median) | 6.2 | 0.13 | **0.081** |
| Rev/direct R² ratio (p95) | ~10⁴ | 170 | 170 |

Residual 14% painted-black is in the AEC3-inherent regime: `bin/aec3_cli` itself fails the same threshold on the cohort DT_movement cases. Further improvement requires beyond-AEC3 work (deferred to v3.22).

### Release cleanup (12 commits, audio-zero)

After alignment shipped, dev-time substrate was removed. **Byte-equal vs `cd73f4e` (alignment HEAD) verified at every commit (10/10 MD5 PASS).**

| Area | Before | After | Reduction |
|---|---:|---:|---:|
| `config.py` | 1547 | 160 | −90% |
| `orchestrator.py` | 5487 | 3632 | −34% |
| `filters.py` | 1062 | 1020 | −4% |
| `eval_aec_challenge.py` | 1169 | 779 | −33% |

- **Config**: linear-filter parametric tuning (`mu`, `kalman_*`, `shadow_*`, `filter_misadjustment_*`) kept as INTERNAL for preset backward-compat; AEC3 alignment flags hard-coded into call sites; NOSHIP / ablation / trace flags removed entirely. Customer surface is now residual-side only (`enable_res`, `enable_cng`, `comfort_noise_floor_dbfs`, HPF, saturation, delay-est).
- **Orchestrator**: removed Q1 transient leakage, F-E1/E3/E5/DelayTrack helpers, DT advisory + diverged-reset chains, kalman_q_per_band + Arc M tilts, all per-frame trace dicts (`_hf_chain_trace`, `_uro_trace` etc.), `__getattr__` shim that the first cleanup agent inserted.
- **Modules deleted**: `state/transparent_mode.py`, `state/signal_dependent_erle.py`, `nlp.py` (SubtractiveNLP closed CANNOT-SHIP per its own docstring).
- **Eval driver**: stripped ~50 dev env-var hooks (`AEC_PLAN_A_*`, `AEC_F_E*`, `AEC_SHADOW_*` etc.). Surviving env vars: `AEC_GAIN_TYPE`, `AEC_MODE`, `AEC_MAX_DELAY_MS`.
- **Docs**: 110 v3.21/v3.22 trace + verdict files removed. Kept: `aec_methods.md`, `aec_algorithm_guide.html`, `architecture_v3_10_5_vs_v3_21_vs_aec3.html`, `c_user_and_integration_guide.md`, `dtd_design.md`, `pbfdkf_shadow_intro.md`, `linear_filter_evolution.svg`, `aec3_extracts/`.
- **Misc**: renamed `aec_filter_evolution.svg` → `linear_filter_evolution.svg` (clearer scope vs the residual / post-filter SVGs). Removed `python/check_byte_equal.py` (anchored to deleted baseline JSON).

### AEC3 alignment audit — what shipped, what didn't, why

**22 / 22 should-be-ON flags shipped (hard-coded into call sites)**: `use_aec3_filter_misadjustment_parity`, `use_per_bin_h_error_refresh`, `use_aec3_h_error_ceil`, `use_partition_summed_x2_for_h_error_gain`, `use_aec3_filter_noise_gate_power`, `use_aec3_residual_noise_gate`, `use_aec3_echo_gen_power_window`, `use_aec3_handle_echo_path_change`, `use_full_delay_change_chain`, `use_current_e2_refined_in_h_error_denominator`, `use_refined_output_selection_for_linear_path`, `form_linear_filter_crossfade_enabled`, `use_partition_summed_x2_for_shadow_mu`, `use_aec3_noise_gate_for_shadow`, `use_poor_excitation_gate_for_shadow`, `use_narrowband_mask_for_shadow`, `use_saturation_gate_for_shadow`, `filter_misadjustment_enabled`, `filter_analyzer_enabled`, `e2_y2_clamp_enabled`, `aec3_post_stationarity_zero_enabled`, `shadow_class_nlms`.

**6 NOT shipped** (each with documented reason for future re-evaluation):

| AEC3 flag | Reason NOT shipped | Next-step gate |
|---|---|---|
| `use_aec3_erle_reverb_quality` | Requires `FullBandErleEstimator` port (continuous quality 0→1 for reverb model). Substrate gap. | Port FullBandErleEstimator first, then re-test. |
| `use_aec3_zero_filter_on_epc` | `W.fill(0)` on EPC destructive on cohort tail (~7/800 cases); `PathChangeRegimeHandler` is load-bearing replacement. | Needs PBFDKF-aware EPC handler redesign before any flip. |
| `use_aec3_epc_classification` | Depends on `use_aec3_zero_filter_on_epc`; both fail Pareto together. | Re-evaluate jointly after zero_filter dependency clears. |
| `use_coarse_e2_time_domain_parity` | Confirmed 24/24 byte-equal no-op (threshold-bound dormant). | Document-only; not a real divergence. |
| `use_aec3_poor_coarse_rescue_copy` | 12-case Pareto FAIL: xFk7 +0.195 / MYrVxVEM −0.431 / qNvSMyUS −0.216 / 9xjhi −0.112. Conditional gating belongs in v3.22 beyond-AEC3, not strict alignment. | Defer to v3.22. |
| `use_linear_filter_output_selection_for_final_output` | `usable_linear` latch contaminates Y-vs-E selection on cohort tail. | Needs `convergence_seen` latch redesign first. |

### Commit ledger

**Alignment phase** (10 commits — `219ee2a..cd73f4e` — audio intentionally changed):

| Hash | Subject |
|---|---|
| `219ee2a` | HF cap anchor 4000 Hz → 2000 Hz (root cause of DT HF black block) |
| `795549b` | HF cap anchor width 125 Hz → 31.25 Hz (strict 1-bin semantic) |
| `8aafe61` | reverb_freq_response fft-resolution-aware smoothing (reverted in `a44dd6d`) |
| `9d77c3b` | audibility band boundaries 94/219 Hz → 375/875 Hz |
| `1650153` | strict CNG + OLA window port (back-end alignment) |
| `e7db416` | CNG constants auto-rescale by hop/sr (wall-clock parity) |
| `1209fa0` | HF paint-black attribution fields in `_hf_chain_trace` (diag only) |
| `a44dd6d` | reverb chain alignment (paint-black HF root cause — 3 reverb bugs) |
| `d311b24` | `nearend_average_blocks` wall-clock rescale |
| `cd73f4e` | 3 alignment fixes from Bundle/Plateau/Stat/Delay audit |

**Cleanup phase** (13 commits — `1b11933..14b3327` — audio zero-impact, byte-equal verified each step):

| Hash | Subject |
|---|---|
| `1b11933` | remove v3.21/v3.22 dev trace docs + scripts (140 files, −55 416 lines) |
| `55872db` | collapse AecConfig dev-time substrate flags (~85 fields) |
| `11d7525` | remove orchestrator dead flag branches |
| `88e84fa` | hard-code shipped flags in filters.py + final orchestrator pass |
| `5cd93bc` | remove SubtractiveNLP module (closed CANNOT-SHIP substrate) |
| `2b59d05` | remove SubbandNearendDetector linear-filter mode |
| `4e945fd` | remove plateau detector + DT advisory dead branches |
| `092da0a` | delete F-E1/E3/E5/DelayTrack dead state + helpers |
| `8139418` | remove AecConfig `__getattr__` legacy-flag shim |
| `587955b` | strip version-history archaeology comments |
| `53759e1` | drop write-only diagnostic state + dead trace dicts |
| `0035dd8` | strip dev-time env-var hooks in eval_aec_challenge + rename evolution svg |
| `14b3327` | v3.21.6.1 release: docs + version bump |

### Verification

- **Byte-equal**: 10/10 MD5 match `cd73f4e` baseline on 5-case sample (NE / FS_static / FS_movement / DT_static / DT_movement) after the full 13-commit cleanup phase.
- **Sanity tests**: 25/25 pass (`test_p52_regime.py` 18 + `test_aec_reset.py` 7).
- **800-case AECMOS** (preset `BALANCED`, fl=832, `--cng`, 4 workers; raw data in `results/v3_21_6_1_release/scores.json`):

  | Bucket | n | echo (↑) | deg (↑) |
  |---|---:|---:|---:|
  | FS_static | 169 | 3.917 | 4.999 |
  | FS_movement | 131 | 3.814 | 4.999 |
  | DT_static | 186 | 4.543 | 1.916 |
  | DT_movement | 114 | 4.437 | 1.942 |
  | NE | 200 | 4.998 | 3.750 |

  vs v3.21.0 baseline (architecture doc reference, `3aadd2d`): FS echo +0.19 / DT echo +0.27 / DT deg −0.46 / NE deg −0.30. Pareto trade — stronger echo cancellation, weaker preservation. Per `feedback_aecmos_pareto_comparison.md`: deg-only comparison invalid; matched-magnitude or full-Pareto evaluation required before any reference comparison. Internal listen-test verification pending.

### Branch / remote operations

- `cleanup/v3_21_6_release` work branch deleted locally after merge into target.
- `debug/v3_21_6_nores_artifact` renamed → `v3_21_release` (force-pushed to origin, old remote ref deleted).
- `feature/v3_23` deleted locally (v3.23 cycle closed, no production output).
- Remote `refs/heads` final state: `main` (unchanged), `v3.16` (legacy), `v3_21_release` (HEAD = `14b3327`).

---

## [3.21.6] — 2026-05-21 — AEC3 Parity Completion (Sprint P1 ships; Sprints P2 / P4 closed intentionally-incompatible; Sprint P3 byte-equal structural)

**Headline**: 1 production change shipped — **Sprint P1 AEC3 FilterAnalyzer port** (single-channel verbatim port of [`docs/aec3_extracts/src/aec3/filter_analyzer.cc`](docs/aec3_extracts/src/aec3/filter_analyzer.cc), `~250 LOC`, owned by `AecState`; default-True). The port produces a non-zero direct-path delay scalar that AEC3's reverb-tail update consumes — indirectly closes v3.21.5 Sprint C's reverb-tail blocker. Cumulative 800-case bench Pareto-positive vs v3.21.5: FS_static **+0.059** / FS_movement **+0.036** / DT buckets within ±0.01 / NE flat.

Sprints P2 / P4 closed as **intentionally-incompatible with our PBFDKF architecture** — both AEC3 parity items are permanently retired (TransparentMode + AEC3-default-off stationarity); any v3.22+ revisit must be labelled as PBFDKF-specific divergence, NOT AEC3 parity restoration. Sprint P3 ships byte-equal structural parity (canonical control surface for `use_stationarity_properties` now lives at `SuppressorConfig.echo_audibility`, with top-level `aec3_post_stationarity_zero_enabled` retained as deprecated alias).

Cycle close: [`docs/v3_21_6_cycle_close.md`](docs/v3_21_6_cycle_close.md). v3.22 entry gate (AEC3 parity baseline locked): **MET** — every Bucket-1 item has a closed verdict.

### Sprint P1 — FilterAnalyzer port SHIPPED (Pareto-positive)

AEC3 [`filter_analyzer.cc`](docs/aec3_extracts/src/aec3/filter_analyzer.cc) produces `FilterDelaysBlocks()` (per-channel direct-path delay) + `ConsistentFilterDetector` (peak-stability gate for `any_filter_consistent`). Pre-v3.21.6, our [`python/modules/state/filter_delay.py:57-60`](python/modules/state/filter_delay.py#L57) received `analyzer_filter_delay_estimates_blocks=None` always (FilterAnalyzer was a v3.18 audit-only stub at `python/modules/filter_analyzer.py`, since deleted) → `min_direct_path_filter_delay()` returned 0 → reverb-tail update never fired on cohort-tail cases. This was the root cause v3.21.5 Sprint C diagnosed but couldn't fix in the v3.21.5 narrow scope.

P1 ships a full single-channel port (block units translated AEC3 `kBlockSize=64` 4ms → our `HOP_SAMPLES=160` 10ms; convergence hold 5s = 500 hops; consistency hold 1.5s = 150 hops). The new `state/filter_analyzer.py` (~250 LOC) covers `ConsistentFilterDetector` + 3-tap 600Hz HPF + region-sweep peak finder + state machine, verbatim against the AEC3 source. `AecState` owns the analyzer; `PBFDAF.get_time_domain_filter()` (new ~10 LOC IFFT concat helper) feeds the time-domain impulse response per hop. Reverb-update `_delay_blocks` switches from legacy `_current_delay // hop_size` to `aec_state.min_direct_path_filter_delay()`. The v3.18 Phase C.A audit-only stub (incompatible API) is deleted.

Verified default-OFF byte-equal preserved (25/25 PASS vs v3.21.5 anchor) before flipping default True. 800-case bench Pareto-positive on FS without DT damage. Verdict: [`docs/v3_21_6_p1_filter_analyzer_verdict.md`](docs/v3_21_6_p1_filter_analyzer_verdict.md).

Known limitation (not blocking ship): `fa_consistent=0%` on the LN18k5r8 cohort case — PBFDKF's Kalman peak position is noisier than AEC3's NLMS-stable envelope, so the 1.5s peak-stability detector rarely fires. Effect: `UpdateFilterGain` falls back to running-max path; `TransparentMode.any_filter_consistent` stays False (irrelevant — TM disabled by P2). Does not affect `filter_delays_blocks()` output (P1's primary deliverable). Documented as the PBFDKF-vs-AEC3 architectural-incompatibility note that motivated P2's parity closure.

### Sprint P2 — TransparentMode audit CLOSED intentionally-incompatible

4 mismatch findings vs AEC3 source ([`transparent_mode.cc`](docs/aec3_extracts/src/aec3/transparent_mode.cc) Legacy variant + [`aec_state.cc:189-325`](docs/aec3_extracts/src/aec3/aec_state.cc#L189) Update flow):

- (A) `enable_transparent_mode=False` hard-coded in orchestrator with stale rationale citing the v3.20 legacy 10-frame ERLE latch (already retired in v3.21 by the per-frame e²<0.5·y² gate in `_aec3_post`)
- (B) 3 block-unit constants in `transparent_mode.py` left at AEC3 4ms-block values with a misleading "blocks not hops -> stays N" comment; actually wall-clock durations
- (C) `any_coarse_filter_converged` not threaded into `TransparentMode.update` (Legacy ignores; HMM variant not ported)
- (D) `all_filters_diverged` derived from `bridge.divergence_indicator > 1.0` proxy (vs AEC3 SubtractorOutputAnalyzer)

P2.0 cohort 3-case trace (LN18k5r8 / s90M7MOT / 9xjhi + 2 others) with `AEC_TRANSPARENT_MODE=1` showed LN18k5r8 fires TM 23.1% @ fa_consistent=0% — the exact PBFDKF-vs-AEC3 cohort-tail false-activation pattern P1's verdict had already documented for FilterAnalyzer. Per [plan's strict P2.1 protocol option 3](`~/.claude/plans/se-aec-aec-main-hazy-lynx.md`), this is sufficient cohort evidence to close as intentionally-incompatible without a full 800-case bench.

Per-mismatch verdicts:
- A → intentionally-incompatible (production stays `transparent_mode_enabled=False`)
- B → **fixed dormant** (`_SANE_FILTER_DELAY_BLOCKS=5` → `_SANE_FILTER_DELAY_HOPS=2`; `_DIVERGED_SEQ_BOUND=60` → `_DIVERGED_SEQ_BOUND_HOPS=24`; `_NUM_CONVERGED_BLOCKS_HIGH=50` → `_NUM_CONVERGED_BLOCKS_HIGH_HOPS=20`; parity correctness with zero current behavior impact)
- C → aligned no-op
- D → aligned via different source signal

Parity substrate (config flag, `AEC_TRANSPARENT_MODE` env hook, 3 corrected constants, trace field) shipped dormant as v3.22 G.2 substrate. **v3.22 G.2 must be PBFDKF-specific divergence (e.g., Kalman-state-derived "no echo path" criterion / delete subsystem / keep dormant) — must NOT claim AEC3 parity restoration.** Verdict: [`docs/v3_21_6_p2_transparent_mode_audit_verdict.md`](docs/v3_21_6_p2_transparent_mode_audit_verdict.md). Discipline rule recorded as feedback memory.

### Sprint P3 — EchoAudibilityConfig structural wiring SHIPPED (byte-equal)

Promoted existing `EchoAudibilityConfig` dataclass (already had AEC3 audibility thresholds + render-floor knobs + use_stationarity_properties + band boundaries) from `SuppressionGain.__init__`-internal local instance to `SuppressorConfig.echo_audibility` field. Orchestrator's stationarity zeroing block at `_aec3_post:3500-3520` (two consumer sites) now reads canonical `self._aec3_sg_config.echo_audibility.use_stationarity_properties`; top-level `AecConfig.aec3_post_stationarity_zero_enabled` retained as DEPRECATED ALIAS propagated via `dataclasses.replace` at orchestrator init (the existing dataclass is `frozen=True`).

Mid-implementation pitfall caught immediately: an initial duplicate `EchoAudibilityConfig` definition clobbered the existing rich one's fields; AttributeError on smoke-render flagged it → reverted to use the existing dataclass. Single-case md5 identical pre/post at default-True; env override `AEC_STATIONARITY_ZERO=0` still produces differing output (alias path verified working).

Removal of the deprecated alias is deferred to v3.22 Sprint I cleanup (after P4 verdict). Per P4 outcome (below), the recommendation is to **keep** the alias as a research toggle indefinitely. Verdict: [`docs/v3_21_6_p3_echo_audibility_wiring_verdict.md`](docs/v3_21_6_p3_echo_audibility_wiring_verdict.md).

### Sprint P4 — Stationarity default-off re-test CLOSED intentionally-incompatible

Re-tested the v3.21.5 Sprint B hypothesis: that P1 FilterAnalyzer + P2 TransparentMode audit + P3 EchoAudibilityConfig wiring may have rescued `_DominantNearendDetector.is_nearend_state()` firing under stationary-far conditions, making AEC3-default-off (`use_stationarity_properties=False`) safe to flip on our cohort.

P4.0 cohort 3-case re-trace (Sprint B's worst 3: WcK0OrF / wVYSGV / xQEUtY2) on post-P1+P2+P3 baseline with user-set strict 3-criterion gate. **All 3 criteria FAIL**:

| Criterion | Result |
|---|---|
| Catastrophic gain drops (Δgain_100 < -0.3) disappear | ✗ 233 / 235 / 424 frames per case (Sprint B baseline was ~280 on xQEUtY2) |
| `is_nearend_state` rate notably improves | ✗ **ΔNE = +0.0 on all 3 cases** — xQEUtY2 stays at 7.0% vs Sprint B's 7.2% baseline |
| Formant damage (1-4 kHz Δ dB) eliminated | ✗ HF Δ -0.94 dB on xQEUtY2 (still audible) |

Root cause: P1 / P2 / P3 paths don't feed into the `_DominantNearendDetector` ENR/SNR decision. The Sprint B safety-net evidence (stationarity zeroing compensates for the incomplete detector port — AEC3 has ScaleFilter / FilterMisadjustmentEstimator companions we don't port) holds on the post-P1+P2+P3 baseline. Hypothesis falsified by direct trace evidence.

Per user directive ("若 P4.0 fail，直接 close P4 intentionally-incompatible，v3.21.6 保留 zeroing default True"), **no 800-case bench run**. Production stays `aec3_post_stationarity_zero_enabled = True` permanently for our PBFDKF + RES port. AEC3-default-off `use_stationarity_properties=False` retired as Bucket-3 closed-DSP decision. Any v3.22+ revisit (e.g., port the missing AEC3 ScaleFilter / FilterMisadjustmentEstimator companions, or replace the detector with a PBFDKF-Kalman-aware NE detector) must be labelled as PBFDKF-specific divergence, NOT AEC3 parity restoration. Verdict: [`docs/v3_21_6_p4_stationarity_retest_verdict.md`](docs/v3_21_6_p4_stationarity_retest_verdict.md).

### Cumulative bench

Standard 800-case render (j9, no env overrides — exercise defaults) against `docs/bench/v3_21_5_baseline/scores.json`:

| Bucket | n | echo Δ | deg Δ | verdict |
|---|---:|---:|---:|---|
| FS_static | 169 | **+0.059** | -0.000 | ok |
| FS_movement | 131 | **+0.036** | -0.000 | ok |
| DT_static | 186 | +0.029 | -0.009 | ok |
| DT_movement | 114 | +0.016 | +0.008 | ok |
| NE | 200 | +0.000 | -0.001 | ok |

Per-case distribution (Δ < -0.05 / Δ > +0.05 threshold): FS_static 17r/68i echo; FS_movement 15r/39i echo; DT_static 11r/48i echo + 51r/56i deg (balanced); DT_movement 7r/23i echo + 17r/29i deg (net positive); NE flat. Identical to per-sprint P1.3 results (P2/P3/P4 don't change algorithmic ship state).

### vs AEC2 / AEC3 reference scores (post-v3.21.6)

Per `docs/aec_methods.md`: v3.21.5 already beat AEC2 by +1.12 FS and beat AEC3 by +0.52 DT_deg / +0.60 NE. v3.21.6 widens the AEC2 FS_static advantage to ~+1.18 (+0.059 on top of +1.12); DT_deg / NE advantage over AEC3 unchanged. AEC3 parity is **structurally complete**: every Bucket-1 item has a closed verdict (shipped / no-leverage / intentionally-incompatible). The two intentionally-incompatible closures (TransparentMode + stationarity-default-off) document permanent PBFDKF-architecture-specific deviations that v3.22+ would only re-open as labeled divergence designs.

### Discipline rules established

- [`feedback_no_parity_claim_for_divergence`](../../../.claude/projects/-Users-mingyu-Desktop-novatek-SE/memory/feedback_no_parity_claim_for_divergence.md): when AEC3 parity closes as intentionally-incompatible, successor design in a later cycle (e.g., v3.22 G.2 after v3.21.6 P2) must be labelled as intentional divergence with PBFDKF-specific rationale — must NOT claim AEC3 parity restoration. Mirror-image of the Round-7 "no parity smuggling into v3.22" anti-pattern.

---

## [3.21.5] — 2026-05-21 — Safe AEC3 Parity (Sprint A ships; Sprints B / C / C2 closed)

**Headline**: 1 production change shipped — Sprint A E2 = min(E2, Y2)
clamp (AEC3 `echo_remover.cc:495-501` port-fidelity fix; default-True).
Sprints B / C / C2 all closed without shipping. Cumulative bench
(A only, since B+C+C2 closed) Pareto-positive vs v3.21.4: FS_static
+0.033 / FS_movement +0.035 dB. DT deg AECMOS-sensitive but not audible
per user spectrogram check.

Plan: `~/.claude/plans/se-aec-aec-main-hazy-lynx.md` (3-cycle Round 7
split: v3.21.5 safe parity / v3.21.6 parity completion / v3.22 intentional
divergence; full plan-review evolution Rounds 1-9 documented in plan
appendix). Triage policy and Bucket-1 closure status for each item below.

### Sprint A — E2 = min(E2, Y2) clamp SHIPPED (Pareto-positive)

AEC3 [`echo_remover.cc:495-501`](docs/aec3_extracts/src/aec3/echo_remover.cc#L495)
specifies `E2 = min(E2, Y2)` when `UsableLinearEstimate()` is True
(bounds residual PSD by mic PSD). Our pre-v3.21.5 [`orchestrator.py:3479-3481`](python/modules/orchestrator.py#L3479)
cited the AEC3 contract in a comment but the clamp itself was absent.
When `error_psd > near_psd` on some bins, unclamped `nearend_pwr` was
inflated → `DominantNearendDetector` ENR (= echo / nearend) biased low
→ detector mis-triggered nearend → `SuppressionGain` used conservative
`nearend_tuning` → echo leaked through HF bands.

Verdict: [docs/v3_21_5_phase1_a_e2_y2_clamp_verdict.md](docs/v3_21_5_phase1_a_e2_y2_clamp_verdict.md).
Action: `e2_y2_clamp_enabled: bool = True` (default-True) in
[`config.py:222`](python/modules/config.py#L222); flag retained for A/B (set
False for byte-equal vs v3.21.4).

### Sprint B — Stationarity AEC3-default-off CLOSED REJECTED (load-bearing safety net)

AEC3 [`echo_audibility.h:40-51`](docs/aec3_extracts/src/aec3/echo_audibility.h#L40)
+ [`residual_echo_estimator.cc:303-313`](docs/aec3_extracts/src/aec3/residual_echo_estimator.cc#L303)
gates stationarity-driven R² scaling by `EchoCanceller3Config::EchoAudibility.use_stationarity_properties`
(AEC3 default = False). Our pre-v3.21.5 [`orchestrator.py:3471`](python/modules/orchestrator.py#L3471)
unconditionally zeroed R² on stationary bins. Sprint B introduced
`aec3_post_stationarity_zero_enabled` flag with default = False
(AEC3-default-off) attempting port fidelity restoration.

800-case bench: bucket means Pareto-acceptable (FS +0.032/+0.048) BUT
**62 DT cohort-tail cases with Δdeg < −0.05** (>> strict halt 30) and
audio listen showed **xQEUtY2 worst formant Δ -2.12 dB F1** (6-10× worse
than Sprint A; audible-grade attenuation, not metric noise). xQEUtY2
trace deep-dive: 6 catastrophic segments totalling ~2.9 s in 40-s case
where `gain_100` drops from ~1 → ~0 when `stationary_mask_active=100%`
AND `is_nearend_state=0%` (detector mis-fires under stationary-far,
SuppressionGain uses aggressive far-tuning → NE-speech destroyed).

Root cause: the legacy stationarity zeroing is a **load-bearing safety
net** compensating for our incomplete AEC3 detector port (missing
companion `ScaleFilter` + `FilterMisadjustment` that keep
`is_nearend_state` correctly firing on stationary-far).

Verdict: [docs/v3_21_5_phase1_b_stationarity_gate_verdict.md](docs/v3_21_5_phase1_b_stationarity_gate_verdict.md).
Action: `aec3_post_stationarity_zero_enabled: bool = True` (default-True
restored — load-bearing legacy zeroing kept). Byte-equal 25/25 vs
v3.21.4 70e7f96 verified. Re-test scheduled for **v3.21.6 Sprint P4**
after companion mechanisms ported (P1 FilterAnalyzer + P2
transparent_mode audit + P3 EchoAudibilityConfig structural wiring).

### Sprint C — Reverb AEC3 semantic audit CLOSED diagnose-only

3 early-return paths in [`reverb_frequency_response.py:61-65`](python/modules/residual/reverb_frequency_response.py#L61)
investigated via Sprint 0 trace fields. Root cause: upstream FilterAnalyzer
stub (`aec3_min_direct_path_blocks=0` always; `linear_filter_quality=None`
90.5%). Both v3.21.5 fix candidates degenerate; cannot ship in v3.21.5
narrow scope. Verdict: [docs/v3_21_5_phase1_c_reverb_semantic_audit_verdict.md](docs/v3_21_5_phase1_c_reverb_semantic_audit_verdict.md).
Moved to **v3.21.6 Sprint P1** (FilterAnalyzer port — parity) +
**v3.22 Sprint F** (RES-internal dead-tail fallback — divergence).

### Sprint C2 — Per-bin H_error refresh selector re-evaluation CLOSED no-leverage

AEC3 `H_error += factor * erl` refresh already exists at [`filters.py:625`](python/modules/filters.py#L625);
`use_per_bin_h_error_refresh: bool = False` flag gates a per-bin REFINED/COARSE
selector path. v3.21.4 U4.A standalone 800-case retest closed FAIL
(17% per-case regression). Sprint C2 hypothesis: Sprint A's E2 clamp
might mask the per-bin tracking damage on DT cases. C2.0 gate test
(5 FS_static worst cases, A only vs A+C2): **9xjhi Δ erle_100 = -0.01 dB**
(memory predicted +8.42 dB single-case win from 2026-05-18 tracer; no
longer reproduces on v3.21.5 baseline). Gate fail → close as no-leverage.
Verdict: [docs/v3_21_5_phase1_c2_per_bin_h_error_refresh_verdict.md](docs/v3_21_5_phase1_c2_per_bin_h_error_refresh_verdict.md).
Path stays dormant research code; env hook `AEC_PER_BIN_H_ERROR_REFRESH`
retained for re-evaluation after v3.21.6 P1/P3 may again shift canonical state.

### Sprint 0 — Trace instrumentation extension (byte-equal preserving)

Extended [`orchestrator.py:3514-3564`](python/modules/orchestrator.py#L3514)
`trace_hf_chain` schema with 22 new per-frame fields covering Sprint A
E2 clamp evidence, Sprint B stationarity mask activity, Sprint C reverb
update paths, NE state staleness, and v3.21.6 / v3.22 reservation fields.
Default-OFF, byte-equal preserving (25/25 PASS at default `trace_hf_chain=False`).

### Cumulative bench (v3.21.5 = A only) vs v3.21.4 baseline

| Bucket | n | Δecho | Δdeg | direction |
|---|---:|---:|---:|---|
| FS_static | 169 | **+0.033** | -0.000 | target direction ✓ |
| FS_movement | 131 | **+0.035** | -0.000 | target direction ✓ |
| DT_static | 186 | +0.013 | -0.014 | echo↑ deg↓ small (AECMOS-only, not audible) |
| DT_movement | 114 | +0.013 | -0.019 | echo↑ deg↓ small (AECMOS-only, not audible) |
| NE | 200 | +0.000 | +0.002 | neutral |

Per-case Pareto: FS 3.7:1 improvement:regression; DT echo 2.3:1; NE
neutral. Audio listen on 5 DT worst-dreg cases (qiQL0BUP / Je6gJ7y1 /
y2ZCo1jA / hF9Lfjcn / I2bme08k) confirms formant-band Δ ≤ 0.4 dB —
AECMOS metric over-reacts to micro-artifacts that don't manifest as
audible damage. User spectrogram check: "看起來差不多".

Cumulative FS recovery vs v3.21.0 baseline (where FS regression was
−0.218 / −0.181): A recovers ~1/3 of the gap. Remaining FS gap to be
addressed in v3.21.6 (FilterAnalyzer port may revive reverb tail update
naturally) + v3.22 (intentional non-AEC3 divergence: hybrid residual /
HF cap-NE decoupling / etc.).

### vs AEC2 / AEC3 reference scores (post-v3.21.5)

Per `docs/aec_methods.md` reference table — v3.21.5 still beats AEC2 by
+1.12 FS (echo bucket) and beats AEC3 by +0.52 DT_deg / +0.60 NE.
Sprint A's small DT deg regression does not threaten the reference
advantage. AEC3 parity is now PARTIAL (E2 clamp shipped; stationarity
gate / FilterAnalyzer port / EchoAudibilityConfig wiring deferred to
v3.21.6).

---

## [3.21.4] — 2026-05-21 — Audit cycle (4 v3.21.2 carry-overs closed; structural ms-based refactor)

**Headline**: research / audit cycle. All 4 v3.21.2 carry-overs from
the original plan adjudicated; 0 production code changes shipped. One
structural refactor (ms-based time-domain config). Byte-equal at
default config vs v3.21.3.

### V4 — Time-domain unit-conversion "bugs" CLOSED NOT-A-BUG

The v3.21.2 plan flagged 3 HIGH-severity "time-domain unit-conversion
bugs" parallel to the bin-index bug fixed in v3.21.2:
`DominantNearendConfig.trigger_threshold=12`, `hold_duration=50`,
`EchoModelConfig.noise_floor_hold=50` — bare-value ports from AEC3
(4 ms blocks) into our 10 ms hops, giving 2.5× longer wall-clock than
AEC3 intended.

V4.1 (`trigger_threshold` 12 → 5 = `blocks_to_hops` canonical) tested
empirically: bench result was **both directions worse** (FS_static
echo −0.021 / FS_movement −0.027 / DT_static deg −0.027 / DT_movement
deg −0.007). Strict regression, not Pareto.

User redirect: physical-meaning analysis (not wall-clock) is the
correct yardstick. AEC3 source inspection
([dominant_nearend_detector.cc](docs/aec3_extracts/src/aec3/dominant_nearend_detector.cc) +
[residual_echo_estimator.cc:340-358](docs/aec3_extracts/src/aec3/residual_echo_estimator.cc#L340))
confirmed each counter measures a different physical quantity:

- `trigger_threshold`: statistical hysteresis depth (+1/−1 random walk)
  — depends on per-sample ENR estimator noise floor, NOT wall-clock.
- `hold_duration`: NE-state dwell — part wall-clock (phoneme) + part
  downstream NE-gain behaviour coupling.
- `noise_floor_hold`: room-noise adapt rate — wall-clock, but quiet-
  room cohort favours slower adapt (= less false-positive on speech
  transients).

Existing values (12 / 50 / 50) kept as empirically-validated cohort
tuning. Verdict: [docs/v3_21_4_time_domain_audit_verdict.md](docs/v3_21_4_time_domain_audit_verdict.md).

#### Companion structural refactor: ms-based time-domain config

Renamed bare-int counter fields to ms-based fields so wall-clock
semantics auto-scale with hop_size at construction:
- `DominantNearendConfig.hold_duration_ms: int = 500` (was
  `hold_duration: int = 50` hops)
- `EchoModelConfig.noise_floor_hold_ms: int = 500` (was
  `noise_floor_hold: int = 50` hops)
- `trigger_threshold` kept as samples (dimensionless statistical
  hysteresis, NOT wall-clock-anchored).

`_DominantNearendDetector` + `ResidualEchoEstimator` now derive their
hop counts via `ms_to_hops()` at construction. `SuppressionGain` +
`ResidualEchoEstimator` accept `hop_size` alongside `sr` (the existing
v3.21.2 U3 sr threading). At 16k/10ms default, derived values are
50 / 50 — **byte-equal preserved**.

### U4.A — Per-bin H_error refresh retest CLOSED FAIL

Cherry-picked v3.21.1's per-bin H_error refresh substrate (commit
`d4e266e` → `12297ed`) onto canonical state and flipped flag to True.

Bucket means within ±0.007 dB (essentially flat), BUT cohort tail is
bad:
- 82 / 800 (10%) cases Δecho < −0.05
- 54 / 800 (7%) cases Δdeg < −0.05
- Worst single-case Δdeg **−0.437** (`LHsrJBRGnUKiMC2m` DT_static)

Same Pareto-damage pattern as v3.21.1 original verdict — bucket means
hide per-case damage. Root cause unchanged: AEC3 per-bin leakage
formula needs companion `ScaleFilter` + `FilterMisadjustment`
stabilisers we don't have aligned.

`use_per_bin_h_error_refresh: bool = False` (default OFF restored).
Substrate code retained as dormant research path for v3.22+.
Verdict: [docs/v3_21_4_u4a_per_bin_h_error_retest_verdict.md](docs/v3_21_4_u4a_per_bin_h_error_retest_verdict.md).

### U4.B — B3 `lf_endpoint_hz` intermediate values CLOSED FAIL

Tested intermediate values between baseline (500 Hz) and B3-failed
(2000 Hz): 1000 Hz (U4.B1) and 1500 Hz (U4.B2). Both regress
monotonically:

| Variant | DT_st Δdeg | FS_st Δecho | Cohort tail |
|---|---:|---:|---:|
| 500 Hz baseline | 0 | 0 | 0 |
| **1000 Hz** | **−0.012** | **−0.007** | 21 echo + 19 deg |
| **1500 Hz** | **−0.015** | **−0.008** | 26 echo + 22 deg |
| 2000 Hz (B3) | −0.016 | −0.012 | (similar) |

Mechanism re-confirmed: 500-2000 Hz band on this cohort carries more
echo than voice on average. Wider LF sum → ENR rises → fewer NE
triggers → DT speech sees less protection.

`lf_endpoint_hz = 500.0` confirmed cohort-empirical sweet spot. B3
fully closed across all tested intermediate values.
Verdict: [docs/v3_21_4_u4b_b3_intermediate_verdict.md](docs/v3_21_4_u4b_b3_intermediate_verdict.md).

### ReverbDecayEstimator — CLOSED NOT-PORTING

Audit across 4 representative cases shows our simpler 139-LOC port
NEVER adapts the decay value: stays at `default_decay=0.85` on all
2188-4307 frame cases. Four contributing factors:

1. **Architectural granularity mismatch** — `n_partitions=6` with
   `K_EARLY_REVERB_MIN_SIZE_BLOCKS=3` leaves only 0-2 data points for
   the slope regression on typical delays. AEC3 has 13 (64-sample)
   blocks for the same filter; partition-level port is fundamentally
   too coarse.
2. Upstream gates (`FilteringQualityAnalyzer` 4-gate AND +
   `StationarityEstimator`) intermittently block.
3. Codex #2 recreate-on-recovery (v3.21.3) wipes estimator state on
   `_reset_filter_derived_state` events.
4. Even when gates open, regression has too few data points to produce
   stable slope.

Full AEC3 port (270 LOC: `AnalyzeFilter` + `EarlyReverbLengthEstimator`
+ `LateReverbLinearRegressor` + validation gates) wouldn't change
observable behavior without first addressing factors (2) + (3). The
constant `0.85` IS the operating reverb value (= non-adaptive fallback)
across all v3.21.x cycles.

v3.22+ prerequisites for re-attempting port: loosen upstream gates +
move estimator state to render-side preservation. Verdict:
[docs/v3_21_4_reverb_decay_audit_verdict.md](docs/v3_21_4_reverb_decay_audit_verdict.md).

### 800-case AECMOS

No production change vs v3.21.3. Default-config byte-equal preserved.
Cumulative numbers vs v3.21.0 baseline unchanged from v3.21.3.

### Commits

- `d7698cd` — V4 time-domain audit CLOSED NOT-A-BUG.
- `12297ed` — v3.21.1 per-bin H_error substrate cherry-pick (default OFF).
- `0446287` — U4.A retest CLOSED FAIL + ms-based config refactor.
- `a80f5a0` — U4.B B3 intermediate values CLOSED FAIL.
- `136ee4c` — ReverbDecayEstimator audit CLOSED NOT-PORTING.

### Carry-over status after v3.21.4

All 4 v3.21.2 plan carry-overs now adjudicated:

| Item | Status |
|---|---|
| Time-domain unit-conversion bugs | CLOSED NOT-A-BUG (kept empirical values + ms refactor) |
| Per-bin H_error refresh retest | CLOSED FAIL (substrate dormant for v3.22+) |
| B3 intermediate values | CLOSED FAIL (500 Hz confirmed) |
| ReverbDecayEstimator full port | CLOSED NOT-PORTING (dormant in our pipeline; v3.22+ prerequisites needed) |

v3.22+ open work documented in each verdict; not blocking current
production.

---

## [3.21.3] — 2026-05-20 — Codex hygiene cycle (reset() AEC3 post-state + return_res_context + dead-knob removal)

**Headline**: 4 hygiene findings from Codex review of v3.21.2, all
fixed and confirmed correct. Three are pure correctness improvements
(reset path completeness + documented contract implementation); one
is dead-code removal. One fix (Codex #2) produces a measurable Pareto
shift on the 800-case bench — accepted as honest correction of a
previously-illusory FS_echo advantage.

### Codex #1 (HIGH) — `AEC.reset()` clears AEC3 post-state

Pre-fix: `AEC.reset()` body initialised the v3.21 AEC3-aligned
post-stage fields (`_aec3_state` / `_aec3_ree` / `_aec3_sg` / OLA buf /
noise PSD / CN gain / pending events / stationarity tracker) in
`__init__` but never touched them on reset(). Re-using an AEC instance
across utterances carried previous-stream post-filter state into the
next stream.

Fix:
- Store `n_bins` + the env-var-driven `_sg_config` as instance
  attributes so reset can rebuild the post chain with the same config.
- Add `_reset_aec3_post()` helper. AecState + SuppressionGain don't
  expose in-place `reset()` so they're recreated; ResidualEchoEstimator
  and StationarityEstimator do, so they're called. Numpy buffers
  zero-filled, counters cleared.
- `reset()` body invokes `self._reset_aec3_post()` at the end.

Test coverage: 5 new unit tests in [python/test_aec_reset.py](python/test_aec_reset.py),
all PASS.

### Codex #2 (MED) — `_reset_filter_derived_state()` clears AEC3 post chain

Pre-fix: helper cleared the legacy ResFilter post-state (gain_smooth /
echo_psd / noise_psd / gates) but v3.21.0 retired ResFilter. The
helper was never updated to clear the AEC3 post chain. Result: on
delay_first / delay_shift / p3h_diverged recovery, the filter taps
reset to zero but the AEC3 post chain kept its prior ERLE / R² / ERL
estimates — applying confident suppression logic on the noisy output
of a freshly-reset (un-trained) filter.

Fix:
- Update docblock to drop the stale ResFilter reference and add the
  AEC3 post chain to the CLEARED list (`_aec3_stationarity` +
  render-side counters in PRESERVED — render activity is input-side).
- Add `preserve_render_side` kwarg to `_reset_aec3_post()` so the same
  helper covers both AEC.reset() (full clear) and
  `_reset_filter_derived_state()` (preserve render-side).
- Invoke `self._reset_aec3_post(preserve_render_side=True)` in the
  helper body.

### Codex #3 (MED) — implement `return_res_context=True` contract

Pre-fix: `AecConfig.return_res_context=True` was documented (CLAUDE.md
"Diagnostic surfaces") to switch `process()` return type from
`ndarray` to `(output, AecResContext)`, but `_res_context` was always
`None` so the documented contract never fired. Dead surface.

Fix: when `config.return_res_context=True` and the AEC3 chain ran,
populate `_res_context = AecResContext(...)` from in-scope state
(raw_output / echo_spec / far_spec / near_spec / far_power / converged
flag / erle_factor / dt_indicator / divergence / over_sub /
saturation / erl_estimate). The end-of-`process()` existing branch
`if _res_context is not None: return (result, _res_context)` now
fires, satisfying the documented contract.

Default path (`return_res_context=False`) byte-equal to v3.21.2 HEAD
on 25-case byte-equal sample.

Test coverage: 2 new unit tests, both PASS.

### Codex #4 (MED) — remove dead legacy delay knobs

Two AecConfig fields became silent no-ops when v3.21 replaced the
legacy DelayEstimator with `LegacyDelayShim` wrapping the AEC3
estimator:

(1) `mov_rate_delay_est_enabled` + `delay_est_period_s_fast` +
    `delay_est_alpha_fast` — orchestrator wrote to
    `LegacyDelayShim._period_samples` / `_alpha` under EPC motion. The
    shim documents these as "no-op compat attributes"; its
    `accumulate()` doesn't read them. Pure dead writes.

(2) `trace_delay_est` + `trace_delay_est_path` — passed as `trace=`
    kwarg into `LegacyDelayShim`, which collected it into
    `_legacy_kwargs` and never consumed it. Documented to populate
    `aec.delay_est._trace_rows`, but the AEC3 estimator doesn't expose
    any such surface; the `--trace-delay-est` CLI flag was a silent
    no-op.

Removed: 5 config fields, the 17-line dead conditional in
orchestrator, 2 env var hooks in eval, 5 reference sites in
run_one_case. Byte-equal preserved (removed code was provably dead).

### 800-case AECMOS vs v3.21.2 baseline

| Bucket | Δecho | Δdeg | Note |
|---|---:|---:|---|
| FS_static | **−0.050** | +0.000 | Pareto cost of Codex #2 (was illusion) |
| FS_movement | **−0.037** | +0.000 | Pareto cost of Codex #2 (was illusion) |
| DT_static | −0.028 | **+0.025** | Pareto gain of Codex #2 (speech recovered) |
| DT_movement | −0.032 | **+0.026** | Pareto gain of Codex #2 (speech recovered) |
| NE | +0.000 | +0.000 | flat |

Mechanism (Pareto attribution): Codex #1 / #3 / #4 are not exercised
on a fresh-instance-per-case bench so contribute no delta. Codex #2
is the source. Pre-Codex #2 (buggy): on filter recovery, AEC3 post
chain held stale ERLE / R² → applied confident-suppression logic to
noisy untrained-filter output → over-suppressed FS_echo (good metric)
but also over-suppressed DT speech (bad metric). Post-Codex #2
(correct): post chain resets alongside filter → no fake confidence →
suppression backs off until filter retrains → less FS_echo gain, more
DT speech preserved. Pareto shift reveals the bug was extracting
illusory FS_echo at DT_deg cost.

### Cumulative 800-case AECMOS vs v3.21.0 baseline (a537b65)

| Bucket | Δecho | Δdeg |
|---|---:|---:|
| FS_static | −0.197 | +0.000 |
| FS_movement | −0.154 | +0.000 |
| DT_static | −0.077 | **+0.119** |
| DT_movement | −0.080 | **+0.141** |
| NE | +0.000 | +0.003 |

### Commits

- `81a5103` — Codex #1 AEC.reset() AEC3 post-state.
- `b2491a8` — Codex #2 _reset_filter_derived_state() AEC3 post chain.
- `80da109` — Codex #3 return_res_context contract.
- `fd2cfcd` — Codex #4 dead legacy delay knobs removed.

---

## [3.21.2] — 2026-05-20 — Frequency-canonical bin-index alignment (HF damage Pareto step) + FS recovery

**Headline**: the v3.21 SuppressionGain port (b5728e5) copied AEC3
bin-index constants directly without converting for our 4× finer FFT
(AEC3 uses fft=128 / 125 Hz per bin; we run fft=512 / 31.25 Hz per
bin). Every HF-processing knob therefore landed at 1/4 of the intended
frequency — most damaging, the HF cap (`limiting_gain_band`) started
at **937 Hz instead of 3750 Hz**, slicing F2/F3 of voiced speech and
producing the user-reported Chinese /i/-vowel distortion ("低頻還在,
400 Hz 以上就被砍").

### Phase A — refactor (mechanism only)

Refactor 4 `SuppressionGain` config dataclasses from bin-index `int`
fields to frequency `float` fields, derive bins at use-site so values
auto-scale with `fft_size`:

- New [`python/modules/freq_utils.py`](python/modules/freq_utils.py):
  `hz_to_bin(hz, n_bins, sr=16000)` / `bin_to_hz(bin, n_bins, sr=16000)`.
  `fft_size` derived from `n_bins` so callers only thread the spectrum
  array, not FFT size separately.
- `HighFrequencySuppressionConfig`: `limiting_gain_band` →
  `limiting_gain_freq_hz`; `bands_in_limiting_gain` →
  `limiting_gain_width_hz`.
- `SuppressorConfig`: `last_lf_band` / `first_hf_band` /
  `last_lf_smoothing_band` → `*_freq_hz`.
- `EchoAudibilityConfig`: add `lf_band_end_hz` / `mf_band_end_hz`
  (audibility weighting band split; previously hardcoded bin 3 / 7).
- `DominantNearendConfig`: add `lf_endpoint_hz` (LF sum window for
  nearend detection; previously hardcoded `min(16, n)`).
- Consumers (`_limit_hf_gains` / `_weight_echo_for_audibility` /
  `_DominantNearendDetector.update` / `SuppressionGain.__init__`)
  resolve bins via `hz_to_bin()` against the input spectrum size.

Smoke-test confirmed all freq defaults reverse-compute to the
pre-refactor bin values (Phase A is mechanically byte-equal-at-init).
[P52 regime tests](python/test_p52_regime.py) 18/18 PASS.

### Phase B — flip to AEC3 frequency-canonical (ship candidate)

After Pareto sweep across each unit-conversion knob, ship candidate
applies four of five flips; one was reverted as cohort-pareto-regressing:

| Knob | Old (bin / freq @ fft=512) | New (freq / bin) | Status |
|---|---|---|---|
| HF cap `lgb` | bin 30 / 937 Hz | **4000 Hz / bin 128** | SHIP |
| HF cap `biq` | 5 bins / 156 Hz | 156 Hz / 5 bins | SHIP (count-preserved; biq=625 Hz tested wash) |
| Mask `last_lf` | bin 5 / 156 Hz | **625 Hz / bin 20** | SHIP |
| Mask `first_hf` | bin 8 / 250 Hz | **1000 Hz / bin 32** | SHIP |
| Mask `last_lf_smoothing` | bin 5 / 156 Hz | **625 Hz / bin 20** | SHIP |
| NE detector `lf_endpoint` | bin 16 / 500 Hz | (kept 500 Hz) | **REVERT** — see below |
| `conservative_hf` inline | bins 20/29 / 625-906 Hz | 2500-3625 Hz | inline (flag-OFF, no-op) |

NE detector LF endpoint flip (500 → 2000 Hz, = AEC3 canonical bin 64)
was tested as T2 and regressed both DT and FS on the 800-case cohort
(DT_static deg −0.016 vs T1, FS_static echo −0.012). Cause: on this
cohort, the 500-2000 Hz band carries more echo than voice energy on
average, so widening the sum pushes `enr` higher and reduces nearend
triggers → cap fires more often → DT damage. AEC3 canonical alignment
does not always translate to cohort improvement.

### Phase C — FS recovery (U5.3)

Bumped `EpStrengthConfig.default_gain` from 0.014 → 0.020 in
[python/modules/residual/residual_echo_estimator.py](python/modules/residual/residual_echo_estimator.py).
`R²` in the nonlinear path = `X² × default_gain²`, so this scales R²
by `(0.020/0.014)² ≈ 2.04×` in nonlinear-mode frames.

S1 trace shows the 800-case cohort runs the nonlinear path 66-92% of
frames (linear ERLE not yet converged on FS), so this knob has high
population leverage. AEC3 precedent: `WebRTC-Aec3EchoPathGain` Aggressive
field-trial profile uses 0.02 — within AEC3-documented range, not an
invention.

Asymmetric Pareto-positive: every bucket non-negative vs Phase B T1.
See [docs/v3_21_2_u5_fs_recovery_verdict.md](docs/v3_21_2_u5_fs_recovery_verdict.md)
for full mechanism + U5.1 (mask_hf.enr_transparent) and U5.2
(normal_render_limit) closed-no-effect results.

### sr threading (U3)

`SuppressionGain.__init__` now accepts `sr=16000`; threads through
`_DominantNearendDetector` / `_weight_echo_for_audibility` /
`_limit_hf_gains` and all `hz_to_bin()` call sites. Orchestrator passes
`self.config.sample_rate` at construction. 16 kHz behaviour byte-equal;
verified sr=48000 now resolves lgb=4000 Hz to bin 43 (vs bin 128 @
16 kHz). See [docs/v3_21_2_bin_audit_verdict.md](docs/v3_21_2_bin_audit_verdict.md)
for the broader audit-clean verdict across `filter/`, `state/`, `delay/`,
`render/`, `epc`, `orchestrator`.

### 800-case AECMOS vs v3.21.0 (a537b65) baseline — final v3.21.2 (T1 + U5.3)

| Bucket | n | baseline echo / deg | new echo / deg | Δecho | Δdeg |
|---|---:|---|---|---:|---:|
| FS_static | 169 | 3.729 / 4.999 | **3.582** / 4.999 | −0.147 | +0.000 |
| FS_movement | 131 | 3.626 / 4.999 | **3.509** / 4.999 | −0.117 | +0.000 |
| DT_static | 186 | 4.237 / 2.387 | 4.188 / **2.481** | −0.049 | **+0.094** |
| DT_movement | 114 | 4.215 / 2.371 | 4.166 / **2.485** | −0.048 | **+0.115** |
| NE | 200 | 4.998 / 4.052 | 4.998 / 4.054 | +0.000 | +0.003 |

DT formant fidelity recovered (matches user HF damage report) at the
cost of HF echo cap relaxation in FS. FS regression remains net negative
but ~5% improved via U5.3 vs unmitigated T1.

### Audit verdicts

- [docs/v3_21_2_audio_analysis_verdict.md](docs/v3_21_2_audio_analysis_verdict.md) —
  U1 quantitative band-energy analysis on 5 worst-deg DT_static cases.
  F2-F3 preservation **+0.48 dB mean** (all 5 cases positive +0.18–+0.91 dB);
  F1 +0.32 dB mean. PASS — the AECMOS deg gain corresponds to real voice
  formant preservation, not a spurious metric move.
- [docs/v3_21_2_bin_audit_verdict.md](docs/v3_21_2_bin_audit_verdict.md) —
  U2 exhaustive grep + line-read audit of all `python/modules/` for
  FFT-scale unit-conversion bugs. AUDIT-CLEAN: no other HIGH-severity
  bin-index bugs in production-active code.
- [docs/v3_21_2_u5_fs_recovery_verdict.md](docs/v3_21_2_u5_fs_recovery_verdict.md) —
  U5 sweep verdict + ship-candidate selection.

### Commits

- `7e9e612` — Phase A refactor + all 5 canonical flips (T2 state).
- `f1ea92c` — Revert B3 NE detector flip (T1 ship candidate).
- `5b7bf1c` — U1 audio analysis verdict (F2-F3 +0.48 dB).
- `8b7de5c` — U2 bin-index audit closure: codebase audit-clean.
- `6a071c1` — U3 sr threading through SuppressionGain consumers.
- `c7481a4` — U5.3 default_gain 0.014 → 0.020 + U5 verdict doc.

### Known carry-overs (v3.21.3+; same AEC3-alignment arc)

- **Time-domain unit-conversion bugs** (3 HIGH severity, parallel pattern
  to the bin-index bug fixed in this version): `trigger_threshold`,
  `hold_duration`, `noise_floor_hold` ported as bare ints from AEC3
  4 ms blocks into our 10 ms hops → 2.5× longer time-equivalent than
  AEC3 intended. Direction of fix opposes FS recovery so deferred.
- **ReverbDecayEstimator partial port** (1/3 of AEC3 size; missing
  `AnalyzeFilter` + `EarlyReverbLengthEstimator` + validation gates).
- **Codex hygiene findings** (4 items, all verified): `AEC.reset()`
  doesn't clear AEC3 post-state; `_reset_filter_derived_state()`
  docblock stale; `return_res_context=True` dead contract; legacy
  delay knobs (`mov_rate_delay_est_enabled`, `trace_delay_est`) no-op.
- **Conservative_hf inline path** — semantics changed from 625-906 Hz
  to AEC3 canonical 2500-3625 Hz; `conservative_hf_suppression=False`
  default means flag-OFF byte-equal.

---

## [3.21.0] — 2026-05-19 — Retire legacy ResFilter; AEC3 chain becomes the production post-filter

**Headline**: v3.21 ships the AEC3-aligned `_aec3_post` chain
(`AecState` + `ResidualEchoEstimator` + `SuppressionGain` + per-bin
comfort noise + sqrt-Hann OLA synthesis) as the single production
post-filter. The legacy 9-stage `ResFilter` chain (~2 200 LOC) is
deleted. The 5-preset menu collapses to a single `BALANCED` preset
(other 4 deleted in R1; the legacy BALANCED was retired in R2).
Cumulative cleanup: −5 565 Python LOC + −32 341 docs lines across
16 commits, byte-equal verified at every step (25-case representative
sample, 5 per bucket at echo percentiles 0/25/50/75/100).

### Architecture change

Production post-filter migration:

```
v3.10.5 — v3.20:                     v3.21:
  ResFilter 9-stage chain ─►           _aec3_post() chain ─►
    stage 1 residual_echo_psd            StationarityEstimator
    stage 2 softgate_emr                 AecState (read-only ADT over
    stage 3 epc_dt_cap                     12 sub-analyzers)
    stage 4 quiet_mask                   ResidualEchoEstimator
    stage 5 3bin_smooth                    (linear / render-based
    stage 6 hf_cap                          + ReverbModel tail)
    stage 7 pre_temporal                 SuppressionGain
    stage 8 temporal smoothing             (Wiener + over-estimation)
    stage 9 noise floor + CNG            Comfort noise generator
                                         sqrt-Hann synthesis OLA
```

The AEC3 chain was developed in stages from v3.18 (Phase C.C AecState
substrate) through v3.20 (Phase A.1 delay subsystem + Phase B PBFDKF
wiring + Phase C residual). v3.21 promotes it from substrate to
production by retiring ResFilter and the `use_aec3_residual` flag.

Reference comparison: [docs/architecture_v3_10_5_vs_v3_21_vs_aec3.html](docs/architecture_v3_10_5_vs_v3_21_vs_aec3.html).

### Bench scores (800-case AEC Challenge, BALANCED)

| Bucket       |    n |  echo (↑) |  deg (↑) | vs AEC3 ref deg | vs AEC2 ref deg |
|--------------|-----:|----------:|---------:|----------------:|----------------:|
| FS_static    |  169 |     3.729 |    4.999 | — | — |
| FS_movement  |  131 |     3.626 |    4.999 | — | — |
| DT_static    |  186 |     4.237 |    2.387 | **+0.537** | −0.003 |
| DT_movement  |  114 |     4.215 |    2.371 | **+0.521** | −0.019 |
| NE           |  200 |     4.998 |    4.052 | **+0.602** | −0.048 |

Anchor scores at [docs/bench/v3_21_3aadd2d_baseline/](docs/bench/v3_21_3aadd2d_baseline/README.md).

### Cleanup rounds (in order)

| Round | Commit | Summary | Net LOC |
|---|---|---|---:|
| Phase A | a24d154 | Baseline + 25-case byte-equal harness + 800-case anchor | +8 684 (test infra) |
| R1 | c07d428 | Delete MILD / SOFT / AGGRESSIVE / MAXIMUM presets | −117 |
| R2 | 6267de0 | Drop legacy BALANCED → rename `BALANCED_AEC3` → `BALANCED` | −89 |
| R3 | 97509c3 | Remove `use_aec3_residual` flag + 2 runtime gates + env hook | −18 |
| R4 | 28ef604 | Collapse if-AEC3 / else-ResFilter to single `_aec3_post` call site | −20 |
| R5 | ceb9ead | Prune dead local-var prep + dead `self.res` state writes | −98 |
| R6 | b63dcbd | Drop dead `_residual_est` readers + Arc G + Arc T blocks | −194 |
| R7 | 0532c57 + 651ccdd | Delete ResFilter chain | −3 302 |
| R8 | 8b51007 | Retire `legacy_state.py` + delete `diagnose_gcc_phat.py` | −624 |
| R9 | c677725 | Delete `legacy_delay.py` | −282 |
| R10a | 1f0bb7f | Drop legacy ResFilter config knobs | −63 |
| R10b-1 | 2176a11 | arc_g / arc_t dead state init + reset paths | −54 |
| R10b-2 + R10c | 9d92334 | Drop dead substrate flags + readers + env hooks | −357 |
| R11 | 09ad7a9 | Archive sweep | (renames) |
| R12 | df793ce | Drop python module orphans | −347 |
| R13 | 75dddf7 | Aggressive `docs/` + `docs/archive/` prune | −32 341 |
| Phase D-1 / D-3 | f60b6a5 + (this commit) | Doc + version bump + v3.21 rewrites | — |

### Closed substrate retired

- v3.14 Arc P + Arc R + Arc S-orth.A.
- v3.15 Arc M v1+v2+v3 / Arc G / Arc T.
- v3.18 Phase C.C AecState facade / C.D-α leakage_diverged / C.E + C.E
  branch ablations / D-γ retried mask shape swap.
- P52 Phase B subclass-and-delegate ResFilter refactor.
- P53 / P55 / P58 dual-filter / dual-PBFDKF / AEC3-pattern RES
  restructure (closed CANNOT-SHIP on 800-case during their respective
  cycles; substrate retained as research log until R13 doc cleanup).

Shipped substrate retained:

- v3.18 Phase A.2 shadow NLMS coarse filter (default ON).
- v3.18 Phase B.2 / B.3 FilterMisadjustmentEstimator + ScaleFilter
  (default ON v3.21).
- v3.18 Phase C.A FilterAnalyzer (audit-only).
- v3.18 Phase F.1 / F.3 AEC3 event classification + asymmetric reset.

### Tests + tooling

- `python/check_byte_equal.py` — 25-case representative byte-equal
  harness. Reference at `docs/bench/v3_21_3aadd2d_baseline/byte_equal_
  reference.json`. Must report `=== 25/25 PASS, 0 FAIL ===` before any
  commit that touches Python outside docs.
- `python/test_f3_1_mic_excess.py` retired with ResFilter (R7).
- `python/test_p52_regime.py` retained — enforces the
  `AcousticRegimeClassifier` anti-loophole contract.
- `python/diagnose_gcc_phat.py` retired (R8) — research-only.

### Docs

Canonical doc set at `docs/` root collapsed from 66 → 11:

- `aec_methods.md` (v3.21 rewrite — algorithm spec).
- `aec_algorithm_guide.html` (v3.21 rewrite — presentation overview).
- `architecture_v3_10_5_vs_v3_21_vs_aec3.html` (NEW — comparison).
- `pbfdkf_shadow_intro.md` / `dtd_design.md` (canonical algorithm refs).
- `c_user_and_integration_guide.md` (C API + integration).
- `refactor_modules_layout.md` (current module map; v3.21 rewrite).
`docs/archive/` retired entirely — 130+ per-arc verdict / design docs
deleted across R11 / R13 / the docs-trim that landed alongside Phase
D-2. The historical record lives in this CHANGELOG and in git history.

---

## [3.15.0] — 2026-05-15 — v3.15 arc closeout (Arc T detector default ON)

**Headline**: Zero ship-able algorithm changes; one preset default flip
(Arc T cohort tail real-time detector → BALANCED default ON, byte-equal
on audio output). Six candidate arcs CLOSED CANNOT SHIP after exhausting
their structural ceilings. Six default-OFF substrate flags retained for
v3.16 retry. v3.16 RES refactor plan authored with 13 ranked candidates
(5 with predicted Δ ≥ +0.005); v3.16 cycle authorised pending phase
kickoff.

### Production-affecting (BALANCED preset behaviour)

- **§10.S0b** (`5bb2fa8`): `arc_t_cohort_detector=True` in BALANCED.
  Cohort tail real-time detector populates `AecStats.cohort_tail_T`
  per-frame and writes `self._arc_t_cohort_tail_signal` field. All
  consumers (5 `arc_m_t_gated` gates + 1 RES preempt path) require
  additional default-OFF flags, so detector ON is **byte-equal on audio
  output** — only diagnostic state changes. 5/5 sanity case byte-equal
  PASS (NE / DT / DT_movement / FS / FS_movement, atol=0.0).
  - Why: enables v3.16 RES refactor consumers (Phase 3 candidates
    v3.16-A force_render OR-in / v3.16-B ENR-path lift) to read the
    signal without per-bench env-flag flipping.
  - Verdict: [docs/v3_15_arc_t_s1_design_and_verdict.md](docs/v3_15_arc_t_s1_design_and_verdict.md)

### Bug fixes shipped

- **§1.0.S1 B4** (`3860335`): drop dead `'converged'` branch in
  quiescent re-sync (`_prev_filter_state` checks). The string belonged
  to `AecFilterState` enum vocabulary, not the internal P3f state
  machine — the branch was structurally unreachable. Cleanup removes a
  code-clarity hazard; behaviour byte-equal on production paths.
  - Verdict: [docs/v3_15_b4_verdict.md](docs/v3_15_b4_verdict.md)
- **§1.0.S2 B5** (`bb9076f`): `_shadow_copy_err_baseline` doc aligned
  with actual implementation as RESERVED (declared but not wired —
  future arc scope). Doc-only change.
  - Verdict: [docs/v3_15_b5_verdict.md](docs/v3_15_b5_verdict.md)
- **§10.S0c B9** (`1323f92`): bench tooling `--workers` CLI flag +
  per-scenario chunk-split (`n_chunks = workers // 3`); 800-case bench
  ~2× speedup over hardcoded `max_workers=3`. Byte-equal sanity 120/120
  between j=3 and j=6 outputs.
- **§1.5b naming** (`03e311b`): renamed `arc_m_v3_t_gated_enabled` →
  `arc_m_t_gated_enabled` per project naming convention (drop numeric
  version suffix from live config field names; keep arc-codename
  prefix as identifier).

### Closed CANNOT SHIP (no production change; default-OFF substrate retained)

- **§1.2 DT-NE compression fix** (`81f59bf`): per-state ENR + per-bin
  override candidates (full + per-bin only). Both fail FS Δecho bars
  3.8–10× over. Same family as v3.13 E5: filter-protection mechanism is
  trade-off-bound. Substrate `dt_ne_compression_fix=False` retained.
  - Verdict: [docs/v3_15_dt_ne_compression_fix_closure.md](docs/v3_15_dt_ne_compression_fix_closure.md)
- **§1.4 Arc M V1+V2** (`92f264b`): EPC-gated per-band Kalman Q boost.
  V1 (0.5/1.0/2.0) FS_movement −0.027; V2 (0.7/1.0/1.5) cohort tail
  −0.053. EPC ⊃ cohort tail catastrophe windows — boosting Q during
  EPC-active windows boosts Q during catastrophe windows. Substrate
  `arc_m_epc_gated` retained.
  - Verdict: [docs/v3_15_arc_m_closure.md](docs/v3_15_arc_m_closure.md)
- **§1.4 Arc G** (`acd2f2d`): per-band W reset on detected gain-change
  drift. ERLE Δ=−1.48 dB / 0/5 audible improvement on listen cohort.
  Destructive zero-out; v3.16 candidate C8 considers non-destructive
  partial decay. Substrate `arc_g_per_band_w_reset` retained.
  - Verdict: [docs/v3_15_arc_g_closure.md](docs/v3_15_arc_g_closure.md)
- **§1.5 Arc T S2 RES preempt wiring** (`3d77486`): two independent
  no-op bugs proven by single-case smoke test on `qNvSMyU` (output
  bit-equal ON vs OFF):
    - **H1** (`over_sub × 1.3`): DEAD CODE in BALANCED — `over_sub`
      only read by `gain_type='wiener'`; all 5 presets use `'enr'`.
    - **H2** (`_using_render_based = True`): OVERWRITTEN 1 line later
      by `_residual_est.compute_residual_echo()` state machine.
  Substrate `arc_t_res_preempt_mode` retained for code symmetry; v3.16
  candidates v3.16-A / v3.16-B fix the integration patterns.
  - Verdict: [docs/v3_15_arc_t_s2_wiring_closure.md](docs/v3_15_arc_t_s2_wiring_closure.md)
- **§1.5b Arc M.v3 T-gated rescue** (`03e311b`): wraps 5 `_arc_m_q_boost`
  call sites with `(arc_m_t_gated_enabled AND _arc_t_cohort_tail_signal)`
  gate. Subset 60-case bench: V1 ΔERLE_lin == M.v3 ΔERLE_lin in EVERY
  bucket EVERY decimal (linear filter byte-equal C1 vs C2). Per-case
  MD5 verified on `qNvSMyU` — ours.wav identical between V1 and M.v3.
  Trace: 4/5 q_boost fires at signal=False (rising-edge events fire AT
  boundary of signal assertion); 1/5 at signal=True was on shadow
  filter (S-orth.A decoupled, no main-output path). Structural
  timing/scope mismatch — discrete-event signals don't pair with
  persistent-state signals without designed temporal alignment.
  Substrate `arc_m_t_gated_enabled` retained. v3.16 candidate C7
  documents 3 retry options (α predictive signal / β post-assertion
  hysteresis / γ per-filter dispatch).
  - Verdict: [docs/v3_15_arc_m_v3_closure.md](docs/v3_15_arc_m_v3_closure.md)
- **§1.6 Arc F per-band Kalman Q schedule** (`415e8ec`): cohort tail
  damage. Substrate `kalman_q_per_band` + `kalman_q_band_scales`
  retained — paired with Arc M V1 substrate (V1 reproduction needs
  THREE flags atomically).
  - Verdict: [docs/v3_15_arc_f_closure.md](docs/v3_15_arc_f_closure.md)

### Audited but produced no actionable work

- **§1.7 RES audit** (`04c1dfe`): 60-case directional audit on the
  v3.15 closeout substrate. Headline finding: `ne_g_floor` fire-rate
  0.93 → **0.000 on DT** — v3.14 Arc P + R raise `spectral_g_min`
  enough that the `max(spectral_g_min, ne_g_floor)` comparison never
  picks `ne_g_floor`. v3.13 verdict's "universal baseline floor" no
  longer holds on v3.15 substrate. Adds NEW v3.16 candidate **C1b**
  (`ne_g_floor` removal) alongside C1 (`epc_dt_cap` removal — still
  0/all-buckets, doubly dead).
  - Sample bias: `--n-cases 60` enumerated alphabetical-first cases,
    all in `doubletalk/` scenario (40 DT_static + 20 DT_movement);
    0 FS / 0 NE / 0 cohort_tail. 800-case re-audit at v3.16 phase
    entry mandatory.
  - Audit + plan: [docs/v3_15_res_audit_and_refactor_plan.md](docs/v3_15_res_audit_and_refactor_plan.md)

### v3.16 candidate plan (13 candidates, 5 phases, 21–30 sprints)

| Phase | Candidates | Sprints |
|---|---|---|
| 0 housekeeping | HK-1 (B3 CNG seed), HK-2 (pcb1N patch), C1 (epc_dt_cap removal), **C1b (ne_g_floor removal — substrate-shift)** | 3 – 4 |
| 1 foundation | C5 (per-state RES interface), C6 (DelayEst audit ⭐ critical gate) | 4 – 5 |
| 2 RES refactor | C2 (ENR per-state × per-band), C3 (4-cap reorder), C4 (noise_floor / CNG) | 6 – 9 |
| 3 Arc T consumers | v3.16-A (force_render OR-in), v3.16-B (ENR-path lift) | 4 – 6 |
| 4 Arc M / G retry | C7 (Arc M.v3 α/β/γ), C8 (Arc G non-destructive decay) | 4 – 6 |

**C6 DelayEst audit** is a critical gate — 5 movement-related v3.15
closures (cohort tail, Arc M V1 FS_movement, Arc F cohort tail, Arc G
destructive W reset, §1.1 H5 DT-NE hypothesis) share echo-path-changing
substrate where DelayEst tracks. If audit confirms DelayEst is the
upstream cause for ≥ 30 % of those wall magnitudes, Phase 3-4 ROI
estimates change.

### Inherited debt (carried to v3.16)

- **v3.13 E2 Path 3 DT debt** (DT_static −0.050, DT_movement −0.025):
  remains unrecoverable in v3.15 production. Closure target moves to
  v3.16 RES refactor (C2 / C4 / C3 totalling +0.005 to +0.040
  predicted DT bucket recovery).

### References

- Top-level closeout: [docs/v3_15_closeout_verdict_pack.md](docs/v3_15_closeout_verdict_pack.md)
- v3.16 plan: [docs/v3_15_res_audit_and_refactor_plan.md](docs/v3_15_res_audit_and_refactor_plan.md)
- All v3.15 closure / verdict docs: [docs/v3_15_*.md](docs/)

---

## [3.14.0] — 2026-05-14 — v3.14 arc (per-band ERL/ENR + decoupled shadow)

**Headline**: Three production changes ship to BALANCED — Arc P
(adaptive per-band ERL EMA), Arc R (per-band ENR thresholds with
`block_lf` tilt), Arc S-orth.A (decoupled shadow Kalman state). First
mechanism in 5+ shadow-retirement attempts that produces genuinely
independent shadow Kalman state. Arc H (Huber loss) closed CANNOT
SHIP after H.S1 — real listen mic saturation is bounded NL residual
floor, not impulsive gradient spike. Arc D (filter-state-aware RES
policy) substrate shipped on `feature/v3.14-arc-d` but not merged
(deferred to v3.15 then v3.16).

### Production-affecting (BALANCED preset behaviour)

- **Arc P P.S3** (`9162d78`): adaptive per-band ERL EMA driven by
  `error_psd / far_lw` (Option B source signal). Replaces scalar
  `erl_estimate=0.3` (7× over-estimate in low-coupling rooms) with
  3-band LF/MF/HF EMA (α=0.99). Flag `f3_1_per_band_erl_adaptive=True`.
  - Verdict: [docs/v3_14_p_s3_verdict.md](docs/v3_14_p_s3_verdict.md)
- **Arc R R.S2** (`5e3e96b`): per-band ENR thresholds with `block_lf`
  tilt (raise LF, lower HF). DT bucket +0.007 dB mean Δdeg on
  800-case; FS regression within −0.02 bar. 7-case xrtntuju listen
  verification: NE not damaged, FS not audibly leaking. Paired with
  `f3_1_per_band_erl_adaptive` for end-to-end per-band gate. Flag
  `res_per_band_enr=True`. R.S2.1 admit_hf control later confirmed
  block_lf winner direction; FS_static intrinsic cost is per-band ENR
  mechanism overhead, not direction-dependent.
  - Verdict: [docs/v3_14_r_s2_verdict.md](docs/v3_14_r_s2_verdict.md)
- **Arc S-orth.A** (`8089974` + `f08ddbf`): decouple shadow's Kalman
  `_error_psd` + `R` from main's. 800-case GREEN PASS — all 5 buckets
  within bar; cohort tail `qNvSMyU` Δecho +0.0036; state correlation
  drops main vs shadow 0.99 → 0.47 on DT_static (target 0.5–0.7 hit).
  Includes Option B quiescent re-sync safety regularization (10% blend
  toward main when 3× drift in steady FS). Flag
  `shadow_state_decoupled=True`.
  - Verdict: [docs/v3_14_s_orth_a_s2_verdict.md](docs/v3_14_s_orth_a_s2_verdict.md)
- **Housekeeping B1 + B2** (`5fbceb0`): `PBFDKF.reset()` cleanup
  (unconditional `delattr` of `_p_max_override_frames`); `AecStats`
  `filter_state` enum/string contract aligned at API boundary.

### Closed CANNOT SHIP (substrate retained)

- **Arc H Huber loss** (`feature/v3.14-arc-h` HEAD): synthetic
  clipping (19.8% bursts) Huber δ ≥ 0.30 identical to L2 (no clipping
  trigger), smaller δ degrades. Real listen cases (01/02/07): Huber
  strictly worse than L2 for every δ. Impulse spike test confirms
  Huber works for true impulsive outliers — but real listen mic
  saturation = bounded NL residual floor (model mismatch), NOT
  impulsive gradient spike. Same physics wall as v3.13 E4/E5
  amplitude-domain closures. Substrate
  [`tools/research/v3_14_h_s1_huber_proto.py`](tools/research/v3_14_h_s1_huber_proto.py)
  preserved.
  - Verdict: [docs/v3_14_h_s1_verdict.md](docs/v3_14_h_s1_verdict.md)

### Substrate shipped but not merged to BALANCED

- **Arc D filter-state-aware RES policy** (`feature/v3.14-arc-d`
  HEAD `0218906`): per-state ENR tuples + 4-cap on/off. 800-case
  bench Δ ≈ 0 on aggregate (only `suspicious_dt + diverged` states
  differentiate — rarely fire in production). Deferred to v3.15
  (which deferred it to v3.16 C2 candidate that subsumes Arc D's
  `coarse_learning` tuple into per-state × per-band ENR refactor).

- **Arc S-orth.B** L1-regularized shadow weight update
  (`feature/v3.14-arc-s-orth-b`): bucket means within hard abort bars
  (FS Δecho −0.013, DT Δdeg +0.000~+0.003) BUT two new large per-case
  FS outliers (`0KjzXA3g…` FS_static Δecho −1.557; `KSN5Jrzo…`
  FS_movement Δecho −0.704). NOT promoted; substrate retained for
  potential v3.15 / v3.16 S-orth.B.S3 retry.
  - Verdict: [docs/v3_14_s_orth_b_s2_verdict.md](docs/v3_14_s_orth_b_s2_verdict.md)

### Volterra arc (research substrate, not what shipped as v3.14)

`feature/v3.14-volterra` carried the Volterra non-linear inverse arc
(S1 cohort baseline + S2 detector wiring + S3.0 joint Hammerstein
feasibility PASS, +2.99 dB mean ERLE on 5/5 NL). Branch was deleted
in v3.15 closeout cleanup; design lock + S2 audit + S3.0 verdict docs
preserved under [docs/v3_14_volterra_*.md](docs/). Volterra arc remains
listed as v3.16 Track 2 in the v3.15 plan §9 roadmap (re-authorisation
required if reopened).

### References

- Per-version evolution: [docs/aec_v3_evolution.md](docs/aec_v3_evolution.md) §v3.14
- v3.14 plan archive: [docs/v3_14_plan.md](docs/v3_14_plan.md) (if preserved)

---

## [3.13.0] — 2026-05-14 — v3.13 arc closure

**Headline**: Single production change shipped (E2 Path 3); two architectural
arcs (E4 NLP + E5 Saturation deepening) closed CANNOT SHIP after exhausting
their physics ceiling; back-end RES audit closed with limited refactor
surface. v3.14 Volterra design lock opens as the canonical breakthrough path.

### Production-affecting (BALANCED preset behaviour)

- **E2.S5 Path 3** (`5b1760c`): `eval_aec_challenge.py` `estimate_delay()`
  default `max_delay_ms` raised 250 → 1024 ms. Aligns bench pre-alignment
  with online F-DelayTrack search window. Closes 6/8 worst-FS listen cases
  that had residual delay 1200–10000 samples (75–625 ms) AFTER prior
  GCC-PHAT pre-alignment.
  - 800-case Δ vs v3.11.x baseline:
    - FS_static Δecho **+0.107**
    - FS_movement Δecho +0.018
    - DT_static Δdeg **−0.050** (accepted "RES unmasking" trade-off)
    - DT_movement Δdeg **−0.025** (accepted)
    - NE Δdeg −0.002 (within bar)
  - Listen: xrtntuju 5-clip DT regression 0 reg / 2 imp; cohort tail
    (qNvSMyU FS_static) Δecho −0.004 (within bar).
  - Trade-off deferred to v3.14+ per-state ENR refactor.
  - Verdict: [docs/v3_13_e2_s5_verdict.md](docs/v3_13_e2_s5_verdict.md)

### Closed CANNOT SHIP (no production change; default-OFF substrate retained)

- **E4 NLP arc** (`3e10621`): 12 sprints S1–S6b. SubtractiveNLP detector
  validated (5/5 NL cohort listen, 0% NE FP after S4.1 cancellation-ratio
  gate). Suppressor (harmonic-pinned σ=50 Hz Gaussian mask) PROVABLY
  ATTENUATES (voice formants disappear at g_min=−30 dB) but **NO AUDIBLE
  NL REDUCTION** at any aggression level (S6a/S6b listen). Closure
  mechanism: multiplicative spectral mask `m[k,t] · Y[k,t]` only modulates
  amplitude; real NL is dominantly phase distortion + time-domain
  transients — unreachable by any amplitude mask family.
  - Detector preserved as default-OFF (`e4_nlp_enabled`); reused in v3.14
    as NL-frame identifier component of ensemble.
  - Verdict: [docs/v3_13_e4_s6_verdict.md](docs/v3_13_e4_s6_verdict.md) +
    [docs/v3_13_e4_s6a_s6b_verdict.md](docs/v3_13_e4_s6a_s6b_verdict.md)

- **E5 Saturation deepening arc** (`c871a5d`): 4 sub-variants (S2/S3/S4a/S4b).
  All on FS-vs-DT trade-off line, slope ~0.5 dB DT loss per +1 dB FS gain.
  All FAIL DT Δdeg ≥ −0.005 hard bar by 4–10×. Mechanism: amplitude-layer
  detector cannot distinguish FS-NL frames from DT high-echo frames — same
  correlation signature in [0.7, 0.95] mic-peak band fires on both.
  - Detector (E5.S3 mic-lpb correlation gate) preserved; reused in v3.14.
  - Verdict: [docs/v3_13_e5_closure_verdict.md](docs/v3_13_e5_closure_verdict.md)

### Audited but produced no actionable work

- **Phase 3 RES gain_floor 5-path audit** (`6cdfbb0`): Empirical fire-rate
  audit on 800-case BALANCED. Findings:
    - `epc_dt_cap`: 0/800 fires (DEAD CODE confirmed, removable)
    - `spectral_floor`: 97% on cohort tail qNvSMyU (LOAD-BEARING)
    - `ne_g_floor`: 88–99% all buckets, low skew 0.13 (Q7 V3 fragmentation
      hypothesis FALSIFIED — universal baseline floor, NOT main FS leak
      carrier)
    - `quiet_mask` / `divergence_floor`: physical fallback, KEEP
  - Canonical refactor surface SMALL (1 path removable, 1 absorbable);
    expected AECMOS delta ~ 0 (consistent with v3.12 5-NEUTRAL closure).
  - S6–S7 (canonical refactor) deprioritized; S8–S9 (4-cap audit + per-state
    ENR) deferred to v3.14+.
  - Verdict: [docs/v3_13_phase3_res_audit_verdict.md](docs/v3_13_phase3_res_audit_verdict.md)

### v3.14 candidate items (deferred)

- **Volterra non-linear inverse filter (HIGHEST priority)**: 6+ month
  dedicated arc. Detector reuse from E4.S2 + E5.S3.
- Phase 3 RES canonical refactor (LOW, cosmetic)
- F-HFR per-band Q/R (LOW-MED, structural)
- E1 mic_dynamic_margin (LOW, 1 listen case)
- DT regression mechanism per-state ENR (MED)

### References

- Top-level closure: [docs/v3_13_arc_closure.md](docs/v3_13_arc_closure.md)
- v3.14 design lock: [docs/v3_14_volterra_design_lock.md](docs/v3_14_volterra_design_lock.md)

---

## [3.12.x] — 2026-05-13 — Stage 1 RES exhaustion (NEUTRAL closure)

**Headline**: 5 NEUTRAL sprints (S6 / S6b / S7 / S10 / S11) targeting every
meaningful gate on ENR denominator and numerator. Stage 1 RES surface is at
local optimum — Δ ≈ ±0.001 on every bucket. No production change. Worst-FS
8-case listen redirected work to filter-side arcs (E1/E2/E4/E5), opening
the v3.13 plan.

### Notable

- Q3 / Q6 / Q7 RES architectural hypotheses fully falsified by 5-NEUTRAL +
  listen.
- v3.11.x retained as production ceiling.
- Verdict: [docs/v3_12_s6_s11_stage1_locked.md](docs/v3_12_s6_s11_stage1_locked.md)

### Sprints

- S6 / S6b: nearend_floor refinement variants — NEUTRAL.
- S7: dt_per_bin unified (third Q7 V3 carrier) — NEUTRAL ([docs/v3_12_s7_verdict.md](docs/v3_12_s7_verdict.md)).
- S8: noise_floor_psd dominant carrier diagnostic.
- S9: noise_floor_refine triple-trial null.
- S10: res_noise_floor_refined NEUTRAL ([docs/v3_12_s10_*.md](docs/)).
- S11: Cap2 FS-loosen NEUTRAL.

---

## [3.11.2] — v3.11 Phase 1 promotions, third tranche

### Production-affecting (BALANCED preset)

- `f_e1_enabled = True`: F-E1 ERL clip range extension + far_active hysteresis.
  - 800-case: NEUTRAL bench mean (Δ < 0.001), addresses extreme-ERL listen
    edge cases.
- `f_delaytrack_enabled = True`: F-DelayTrack continuous delay variance
  (replaces hard cut at confidence ≥ 0.5).
  - 800-case: NEUTRAL bench mean.

### References

- Phase 1 final review: [docs/v3_11_phase1_final_review.md](docs/v3_11_phase1_final_review.md)

---

## [3.11.1] — v3.11 Phase 1 promotions, second tranche

### Production-affecting (BALANCED preset)

- `shadow_mu_state_aware = True` (B6): 4-band shadow µ schedule with
  `suspicious_dt → 0.5` band; binary cut → state-aware.
  - 800-case bucket-mean +0.007 ΔERLE; wlAXM0i listen verified
    indistinguishable from baseline.

### References

- B6 listen verdict: [docs/v3_11_phase1_b6_listen_verdict.md](docs/) (see
  [Phase 1 final review](docs/v3_11_phase1_final_review.md))

---

## [3.11.0] — v3.11 Phase 1 promotions, first tranche

### Production-affecting (BALANCED preset)

- `shadow_r_reset_enabled = True` (B5, Yang 2017 R-reset): symmetric R-reset
  on EPC (extends F2.3 to shadow filter's `_error_psd` + `R`).
- `f_e5_enabled = True` (F-E5 saturation 4-fix bundle):
  - mic soft-clip when sat_mic > 0.3
  - main mu sat-gate (freezes at sat_level > 0.5)
  - error_psd fast-attack reset on sat → clean transition
  - shadow_rise mask during saturation
  - sKXucFp4 single-case top: +0.348 dB Δecho
- `diverged_reset_enabled = True` + `diverged_reset_triple_and = True`:
  triple-AND gate (streak + shadow_advantage > 2.0 + filter_state == diverged)
  to avoid F2.2 EMA trap (which closed FAIL with 17 reg / 8 imp).

### Bench

- 5 buckets verdict OK; Δ < 0.001 dB vs v3.10.6 baseline; cohort tail
  qNvSMyU +0.010 linear preserved.

### References

- [docs/v3_11_phase1_final_review.md](docs/v3_11_phase1_final_review.md)
- F2.3 R-reset verdict: [docs/f2_3_verdict.md](docs/f2_3_verdict.md)
- F2.4 mu holdoff verdict: [docs/f2_4_verdict.md](docs/f2_4_verdict.md)

---

## [3.10.6] — three v3.10.6 fix promotes (2026-05-12)

### Production-affecting (BALANCED preset)

- **F3.1 v3** (mic-excess gate + dt_per_bin blend): per-bin NE evidence,
  AUROC 0.871. Closes xrtntuju 5-clip DT NE-damage regression cohort.
- **F2.3** (epc_r_reset_enabled): EPC R-reset for main filter (Yang 2017
  pattern, single-filter scope).
- **F2.4** (mu_holdoff_no_reset): release-counter form of `_simple_mu_holdoff`;
  prevents marginal-DT counter resets.

### References

- Plan closure: [project_plan_hazy_lynx_closure.md](memory/project_plan_hazy_lynx_closure.md)
  (memory)
- F3.1 / F2.1 verdicts: [project_f3_1_f2_1_results.md](memory/project_f3_1_f2_1_results.md)
  (memory)

---

## [3.10.5] — baseline reference (pre-v3.11 era)

The 800-case AECMOS reference snapshot used as the comparison baseline for
all v3.11+ work. Captured in `results/v3_10_5_main/scores.json`.

### Bucket means (800-case BALANCED)

| Bucket | n | echo (↑) | deg (↑) |
|---|---:|---:|---:|
| FS_static | 169 | 3.646 | 4.999 |
| FS_movement | 131 | 3.705 | 4.999 |
| DT_static | 186 | 4.221 | 2.323 |
| DT_movement | 114 | 4.053 | 2.368 |
| NE | 200 | 4.998 | 4.011 |

---

## Aggregate v3.10.5 → v3.13.0 (this release vs pre-v3.11 baseline)

Computed from `results/v3_10_5_main/scores.json` vs
`results/v3_14_baseline/scores.json` (rendered today on v3.13 closure HEAD;
v3.14 detector substrate is default-OFF so render = pure v3.13 behaviour).

| Bucket | Δecho | Δdeg | Source |
|---|---:|---:|---|
| FS_static | **+0.107** | 0 | E2 Path 3 |
| FS_movement | +0.018 | 0 | E2 Path 3 + Phase 1 micro-effects |
| DT_static | +0.014 | **−0.050** | E2 Path 3 (RES unmasking, accepted) |
| DT_movement | +0.005 | **−0.025** | E2 Path 3 (accepted) |
| NE | 0 | −0.002 | NE invariant preserved |

**Net**: FS bucket improved (Δecho +0.107 / +0.018), DT bucket trade-off
(echo micro-up, deg micro-down within bar), NE unchanged. Cohort tail
listen materially improved (E2 Path 3 closes 6/8 worst-FS listen edge
cases; xrtntuju 5-clip 0 reg / 2 imp).

---

## Earlier history

For v3.10.4 and earlier (v3.7 → v3.10.4), see canonical research log
[docs/SUMMARY.md](docs/SUMMARY.md). v3.7.1 is the most recent git tag
prior to v3.13.0; tags between v3.7.1 and v3.13.0 are P52/P53 milestone
tags rather than product versions:

- `p52-phase-a-closed-path3` (2026-05-12)
- `p52-phase-b-closed`
- `p53-design-locked`
- `p53-step-0-closed-T0E`
