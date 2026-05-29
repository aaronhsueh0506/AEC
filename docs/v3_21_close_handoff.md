# v3.21 CLOSE — Session Handoff (UPDATED 2026-05-29, pre-2nd-compact)

> Read fully after `/compact` to restore state. v3.21 = AEC3 *alignment only*; we are
> CLOSING it, now via an "honest-alignment" path uncovered by a deep conversion audit.
> Supersedes the original handoff (which only covered C1-C6 + the first 800-case bench).

---

## 0. Where we are (one paragraph)

The first 800-case bench closed **C1-C6 (wall-clock EMA) = NO-SHIP**. That triggered a
**deep audit of every AEC3-conversion flag**, which found **2 shipped (default-ON)
"AEC3-alignment" flags are mis-derived** (`active_render` 64× too high; `fft_density`
wrong-basis) and **2 are gray-zone unvalidated** (`reverb_smoothing`, `dne_trigger`).
User chose to **correct to true AEC3 + validate**. The **Tier-C validation bench is
RUNNING now** (§1). When it finishes → per-flag verdict → `/simplify` → close. A
**v3.22 plan** is already written (§7).

---

## 1. RUNNING NOW: Tier-C validation bench (task `b5dsi9npm`)

- **Script**: `python/v3_21_tierc_validation_bench.py` (NEW this session; self-contained `_write_report`, no stale-flag-table crash risk).
- **Cmd**: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 python/v3_21_tierc_validation_bench.py --workers 9`
- **Log**: `/tmp/tierc_val.log` · **JSON**: `out_v3_21_tierc/scores_tierc.json` (saved BEFORE report) · **Report**: `docs/v3_21_tierc_validation_report.md`
- **6 configs** (each isolated vs V0 = current production = plain BALANCED):

  | config | tests |
  |---|---|
  | `V0_prod` | current production baseline |
  | `V1_active_render` | active_render corrected 5.96e-4 → **9.31e-6** (the 64× fix) |
  | `V2_fft_off` | fft_density OFF (1×) — does floor scaling help at all (4× vs 1×) |
  | `V3_reverb_off` | reverb_smoothing revert to verbatim 0.2 |
  | `V4_dne_off` | dne_trigger revert to verbatim 12 |
  | `V5_all_corrected` | active_render 9.31e-6 + reverb OFF + dne OFF (honest-alignment ship candidate; fft kept ON) |

- **~303/800 when last checked, healthy (16 Python workers); just entered FS bucket.** Order = DT(300)→FS(300)→NE(200). FS/NE are the decision-relevant buckets. This run also **completes NE** (the length bug is fixed).
- **On completion**: read the report → per-flag verdict via **matched-magnitude AECMOS Pareto** (less echo auto-lifts deg ≠ win). If report missing (crash), reconstruct from the JSON. Decide each flag: keep-corrected / keep-shipped / revert.

---

## 2. THE DEEP AUDIT (core result — the 4-category framework)

When frame-rate changes (AEC3 4ms-block → our 10ms-hop), **what invariant survives** depends on the quantity:

| Category | Convert? | Flags |
|---|---|---|
| **Genuine temporal** (counter/duration/slew-rate) | convert OK | `gain_ratchet` (slew); hold/hangover counters |
| **Structural / formula port** (no frame dep) | inline, safe | `filter_erl_for_h_error_refresh`, `x2_reverb_for_erle`, `reset_res_on_rescue_edge`, `subtractor_max_abs_for_saturation` |
| **EMA-α / evidence-count gray-zone** (seconds-vs-count) | **questionable** | `reverb_smoothing`, `dne_trigger`; C1/C3/C4 (dropped); CNG (low-pri) |
| **Threshold / per-bin density** (does the compared quantity scale?) | **verify** | `active_render` (WRONG), `fft_density` (WRONG-basis) |

### Confirmed mis-derivations (both shipped default-ON):
- **`active_render` = 64× too high (CONFIRMED).** Compares `far_pwr = mean(far²)` (per-sample mean, float[-1,1]). AEC3's `100²×64` is `limit²×kBlockSize` for a block **SUM** ⇔ `mean > 100²`. Mean-correct value = `100²/32768² = **9.31e-6**`, NOT `5.96e-4` (=100²×64). Proof-by-contrast: `low_noise_render` correctly uses `block_energy_scale` for its sum-threshold; active_render kept the ×64 against a mean. Real impact: on loud FS, fires 63% vs AEC3-equiv 79% (16pp; wider on quiet far). It's a *tuning* (less-eager) mislabeled "AEC3-strict" — true AEC3 is MORE eager than legacy 1e-4.
- **`fft_density` = wrong-basis (CONFIRMED).** White-noise demo: per-bin |X|² is **fft_size-INDEPENDENT** ((64,128)==(64,512)=63.93) and scales with the **real frame = block_size = 2×hop = 320** ((320,512)=319.66 = 5.000×). Tonal → 25× ((2×hop/64)²). So the true factor is **(2×hop)/64 = 5×**, NOT the shipped `(fft/2)/64 = 256/64 = 4×`. BUT 4× vs 5× = 1.25× ≈ **0.97 dB, sub-AECMOS-noise** → v3.21 performance NOT affected; only the derivation comment is wrong.

### Gray-zone (single-case-motivated, never 800-case-validated):
- `reverb_smoothing` (EMA-α 0.2→0.428) and `dne_trigger` (evidence-count 12→5-hops) both chose "preserve seconds." C1-C3 (same premise) empirically regressed → "preserve count" (revert to verbatim) is the better-supported choice.

### Other EMA-sweep findings:
- **CNG** (`orch 620-624`): 4 converted EMAs (Y²-α, N²-freshness, N²-slow-up growth, N²-initial), **unconditional (no flag)**, but power-envelope (defensible) + only affects comfort noise → **low-pri watch** (v3.22 Arc 5).
- **legacy `erle.py`** (`FilterErleEstimator`/`FullbandErleEstimator`): only re-exported by `aec.py`; pipeline uses `state/subband_erle`+`state/fullband_erle`. → **dead/back-compat-only, /simplify cleanup candidate.**
- **our-own EMAs** (preprocessing sat-envelope, dtd coherence, filters `alpha_power`/`_alpha_r`, orch diagnostic/mu/erle-window/limiter): tuned for our hop directly → **no porting bug, out-of-scope.**

---

## 3. Flag dispositions for /simplify

- **5 SOLID alignment (default-ON) → INLINE (remove flag, make unconditional):**
  `use_aec3_filter_erl_for_h_error_refresh`, `use_aec3_x2_reverb_for_erle`,
  `use_aec3_reset_res_on_rescue_edge`, `use_aec3_subtractor_max_abs_for_saturation`,
  `use_aec3_wallclock_gain_ratchet`.
- **4 QUESTIONABLE (default-ON) → per Tier-C verdict:** `use_aec3_active_render_threshold`
  (→9.31e-6 if V1 OK), `use_aec3_fft_density_scaled_psd_floors` (keep ON if V2 shows scaling helps; value 4× fine), `use_aec3_wallclock_reverb_smoothing` (→revert if V3 OK), `use_aec3_wallclock_dne_trigger_threshold` (→revert if V4 OK).
- **C1/C3/C4/C6 (default-OFF) → DROP** (NO-SHIP bundle): `use_aec3_wallclock_subband_erle_smoothing`, `use_aec3_wallclock_fullband_erle_smoothing`, `use_aec3_wallclock_low_noise_render_iir`, `use_aec3_active_render_threshold_shadow_epc`.
- **Closed/dormant → DROP:** `use_aec3_just_reset_gate_on_linear_path` (CLOSED −8dB), `use_aec3_block_energy_for_reverb_decay` (dormant; AEC3 default static).
- **Temp validation flag → REMOVE after decision:** `active_render_threshold_aec3_corrected` (fold its value into active_render's inlined behavior).
- **v3.22 substrate → KEEP default-OFF:** `hf_min_gain_floor_during_dne_enabled`, `enable_lf_filter_failure_r2_injection`.

---

## 4. Tier-C implementation DONE (byte-equal verified)

- `aec3_scale.py`: added `ACTIVE_RENDER_THR_AEC3_FLOAT = psd_int16_to_float(100.0**2)` (=9.31e-6) + comment explaining the 64× mis-derivation.
- `config.py`: added `active_render_threshold_aec3_corrected: bool = False` (temp validation flag).
- `orchestrator.py:~3197`: `_ar_thr` now 3-way: corrected(9.31e-6) / shipped(5.96e-4) / legacy(1e-4). **NOTE: I wrote it inline as `(100.0**2)/(32768.0**2)` — flagged for cleanup (use the named constant) in §6.**
- reverb/dne reverts need NO code (existing flag-OFF). fft 4≈5× tested via existing ON/OFF.
- **Byte-equal**: toggle test on 9xjhi (~3s): plain BALANCED md5 `3c9a1465a83a4a63314a5b0702dbae06` == corrected=False (default); corrected=True differs `c2d43fbc...` (flag live). ✓

---

## 5. 2×hop architecture finding (verified `filters.py:88-90`)

`block_size = 2×hop = 320` is the **real energy frame**; `fft_size = 512` is just next-pow2 **zero-pad** of 320. **Per-bin |X|² energy ∝ 2×hop = 320, NOT fft_size = 512.**
- **Only `fft_density` is affected** among v3.21 constants (it's the per-bin spectral floor). Time-domain detectors are correctly hop-based: `low_noise_render` sums the hop(160) with `block_energy_scale(·,160)=×2.5` ✓; `active_render` means the hop (its 64× is the separate mean-vs-sum issue).
- **v3.21 impact: performance NONE** (fft 4× vs true 5× = sub-noise). **Only fix the comment** at /simplify (energy ∝ 2×hop; 4× is a single-constant approx of the signal-dependent 5×-broadband/25×-tonal true value; adaptive version = v3.22 Arc 3).

---

## 6. int16→float code-style review conclusion

**KEEP** `psd_int16_to_float(<int16 literal>)` named constants — they are exact (byte-equal-safe vs rounded floats), preserve AEC3-source provenance (greppable, e.g. `20075344`), and centralize the `32768²` convention; the literal is only in the *definition*, use-sites already see a float. Do NOT hardcode rounded floats.
**The real cleanup = scattered inline `32768` magic at USE-sites** → route through `_PSD_SCALE`/`psd_int16_to_float`/named constants: `orch:659` (`10/32768`), `orch:3020` (`*32768`), `orch:132`/`2924` (`32768²`), `config:395`, `suppression_gain:224`, `echo_path_delay:55`, **and my new inline `(100²)/(32768²)` at orch:3197.** No bare `32768` at use-sites.

---

## 7. v3.22 plan — WRITTEN at `docs/v3_22_plan.md`

Beyond-AEC3, DSP-only. Arc 1 = shadow convergence quality (~2-3 dB movable; coarse_conv≈0% root). Arc 2 = painted-black/non-linear (B0 Speex-study → B1/B2 stub stopgaps → B3 Volterra). Arc 3 = signal-adaptive per-bin floor (the (2×hop)/64 signal-dependent fix). Arc 4 = pipeline-tuned thresholds (gated on Tier-C). Arc 5 = per-estimator EMA invariant (incl. CNG). Arc 6 = Python↔C shadow reconcile. **Reframed with the 2×hop note**: most of the "6-8 dB FFT ceiling" was a frame-energy units artifact → movable budget is LARGER than first thought.

---

## 8. PENDING — the v3.21 CLOSE sequence (gated on Tier-C finishing)

1. Read Tier-C report → per-flag verdict (matched-magnitude Pareto) → finalize honest-alignment config.
2. `/simplify` Python: inline 5 solid; apply Tier-C verdicts; drop C1/C3/C4/C6 + closed/dormant; remove temp `active_render_threshold_aec3_corrected` flag; **LINEAR always max strength** (no down-gating on linear/filter side); keep only **RESIDUAL control-strength** knobs adjustable; inline-`32768` cleanup (§6); remove dead `erle.py` (verify first). **FOCUS: orchestrator.py (3811 L) + filters.py (1057 L)** — the two big files; user explicitly asked to make cleanup effective there. (Cleanup map: orch `__init__` 712 L / `process` 1254 L / `_aec3_post` 620 L; filters PBFDKF has legacy `_update_weights` 672-870 + AEC3 `_update_weights_aec3` 870-976 — collapse legacy, byte-equal-CRITICAL, do last.)
3. Docs: CHANGELOG + fix derivation comments (active_render 64×, fft_density 2×hop-basis); tidy diff to release style.
4. Clean dev cruft (AEC/ + /tmp); preserve test data (`aec_record.wav`) + load-bearing.
5. Present for user review → close v3.21.

**Open decisions to surface at review:** Python↔C shadow divergence (C reconcile deferred to v3.22 Arc 6); `reverse_copy` (unit-tested, keep or remove+update test); the 2 v3.22 stubs (keep default-OFF); `NlmsFilter`+`AecMode.LMS/TIME` (alternate modes BALANCED never uses — drop to PBFDKF-only? API decision).

---

## 9. Files changed this session (since first handoff)

1. `python/v3_21_800case_bench.py` — `_write_report` KeyError fix (iterate manifest) + header date/workers + `_case_task` NE length-equalize fix.
2. `python/v3_21_800case_report_from_json.py` — NEW (regen helper for the C1-C6 report).
3. `python/v3_21_tierc_validation_bench.py` — NEW (6-config Tier-C validation, RUNNING).
4. `python/modules/aec3_scale.py` — `ACTIVE_RENDER_THR_AEC3_FLOAT` constant.
5. `python/modules/config.py` — `active_render_threshold_aec3_corrected` temp flag.
6. `python/modules/orchestrator.py` — `_ar_thr` 3-way corrected branch (~3197; inline 32768 to clean).
7. `docs/v3_22_plan.md` — NEW.
8. `docs/v3_21_close_handoff.md` — this update.

(Earlier-session, already in tree: C1/C3/C4/C6 flags + state/subband_erle + state/fullband_erle + suppression_gain + erle_estimator + aec_state + epc + CLAUDE.md + bench_aecmos ONNX cap + shadow_q_ratio removal. C1-C6 bench artifacts: `/tmp/c1c6_800.log`, `out_v3_21_800case/scores_800case.json`, `docs/v3_21_800case_bench_report.md`.)

---

## 10. CONSTRAINTS (permanent)

- hop/block/fft = 160/320/512 HARD; mic-HPF ON / ref-HPF OFF locked; **DSP-only** (no NN; don't even raise it).
- v3.21.x = AEC3-alignment only; v3.22 = beyond-AEC3.
- Matched-magnitude AECMOS Pareto (less echo auto-lifts deg ≠ win); absolute target = beat AEC2 + beat AEC3 deg.
- Byte-equal before any commit; isolate via 800-case; never batch.
- No version numbers in variable/config names (mechanism descriptors).
- Language: Chinese or English only.
- 800-case standard: BALANCED / fl=832(52ms) / cng / capped workers OK (scores identical).

---

## 11. Quick resume checklist

1. Read this doc.
2. Check Tier-C bench: `tail /tmp/tierc_val.log`; if done, read `docs/v3_21_tierc_validation_report.md` (or regen from `out_v3_21_tierc/scores_tierc.json`).
3. Per-flag verdict → finalize config → proceed to `/simplify` (gated on user review of the verdict).
