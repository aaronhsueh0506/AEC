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

### Known carry-overs (v3.22)

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
