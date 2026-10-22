/* aec_scale.h — reference constant -> our-scale conversion helpers (C port of
 * the Python scale-conversion module under python/modules/).
 *
 * The reference constants are authored on an int16 PSD scale (PSD ~ int16^2 up
 * to 32768^2) and 4 ms blocks (64 samples @ 16 kHz). Our pipeline runs on float[-1,1] PSD
 * scale and the live config hop_size (50% overlap, frame=2*hop). Constants
 * explicitly labelled as project-native below keep their reference 16 kHz /
 * hop=160 authoring grid; that is not the current production default.
 * NEVER paste a raw int16 PSD threshold or 4 ms-block count into ported code —
 * route it through here so the conversion is explicit and uniform.
 *
 * Parity note: Python's int(round(x)) is round-half-to-EVEN; the *_to_hops
 * helpers below use lrintf() (default FE_TONEAREST = round-half-to-even), NOT
 * round() (half-away-from-zero), so bin/hop counts match the reference exactly.
 *
 * These are init-only conversion helpers (called once at setup, not per-hop);
 * every parameter and return value is float32 (drift versus Python
 * accepted). The #define constants below are plain (untyped) literals,
 * consumed verbatim by other modules across the codebase.
 */
#ifndef AEC_SCALE_H
#define AEC_SCALE_H

/* ── Core scale constants ──────────────────────────────────────────────── */
#define AEC_PSD_SCALE        (32768.0 * 32768.0)   /* 1.0737e9  */
#define AEC_PSD_SCALE_INV    (1.0 / AEC_PSD_SCALE) /* 9.313e-10 */
#define AEC_BLOCK_SAMPLES_16K 64                    /* kBlockSize @ 16 kHz */
#define AEC_BLOCK_MS          4.0

/* ── Conversion functions (pure; match the Python module's semantics) ───── */
int    aec_blocks_to_hops(int blocks, int hop_samples, int sample_rate);
int    aec_ms_to_hops(float ms, int hop_samples, int sample_rate);
float  aec_per_block_rate_to_per_hop(float per_block_rate, int hop_samples, int sample_rate);
float  aec_per_block_ema_alpha_to_per_hop(float per_block_alpha, int hop_samples, int sample_rate);
float  aec_per_bin_psd_threshold(float calibrated_value, int hop_size, int ref_hop);
float  aec_block_energy_scale(float value_int16sq, int hop_samples);
/* per_block_multiplier is double, NOT float like this file's other helpers:
 * Python's residual_echo_estimator.py keeps the reverb-decay config
 * constant (0.83) as a python float (float64) and exponentiates it in
 * float64 before the result is ever narrowed to float32 (the single
 * narrowing numpy does when the scalar multiplies a float32 array). Taking
 * a float32 base here and calling powf() double-rounds: the base is
 * narrowed to float32 BEFORE exponentiating, which is a different (and
 * less accurate) float32 result than exponentiating in float64 and
 * narrowing once at the end -- by exactly 1 ULP for the reverb-decay case
 * (0.83 at the default 512-fft grid), which an unbounded IIR state
 * (ResidualEchoEstimator's reverb accumulator) carries forward forever.
 * Callers must pass the full-precision config constant (the
 * AECB_REE_REVERB_DECAY/_MILD_SCALE macros already are double literals). */
float  aec_per_block_growth_to_per_hop(double per_block_multiplier, int hop_samples, int sample_rate);
/* Generalizes aec_per_block_growth_to_per_hop's fixed 4 ms-block
 * reference to an ARBITRARY reference grid. For PROJECT-NATIVE (non-reference)
 * per-hop RETENTION-convention EMA constants -- i.e. ones whose update is
 * literally ``x <- retention * x + (1 - retention) * new`` (the constant
 * multiplies the OLD state directly; this is the OPPOSITE convention from
 * aec_per_block_ema_alpha_to_per_hop's reference ``x <- (1-a)x + a*new``, where a
 * multiplies the NEW sample) -- authored against this repo's own reference
 * hop=160/sample_rate=16000 (10 ms) grid with no rate conversion (e.g.
 * aec.c's shadow_err_alpha, epc_shadow.c's EPV_FAST_TC/EPV_SLOW_TC). Same
 * power-law derivation as aec_per_block_growth_to_per_hop (``g_per_hop =
 * g_per_ref ^ (hop_seconds/ref_seconds)``), with a parameterized reference
 * period instead of the hardcoded 4 ms block. */
float  aec_growth_rehop(float retention_at_ref, int ref_hop_samples, int ref_sample_rate,
                        int hop_samples, int sample_rate);

/* ── Double-precision exceptions ─────────────────────────────────────────
 * UNLIKE every function above (float32, init-only, drift vs Python
 * accepted), these two must stay double: they derive the live-observer
 * corroboration gate's EMA alpha / margin, which feeds a discrete per-hop
 * inject/don't-inject decision that Python evaluates in double throughout
 * (see AecPost.observer_obs_err_ema's own comment in aec_post.h for why
 * that must match). A float32 version here would silently reintroduce
 * that precision-parity bug. Still init-only (one call per instance, at
 * carve time) -- the double precision is for cross-language bit-parity,
 * not runtime cost. */
double aec_ema_alpha_from_tau_ms(double tau_ms, double hop_ms);
double aec_db_to_linear(double db);

/* ── Pre-converted constants (16 kHz, hop=160 reference) ───────────────────
 * Defined by their computing expression so they fold bit-identically to the
 * Python module-level constants. */
#define AEC_H_ERROR_INIT_FLOAT   10000.0
#define AEC_H_ERROR_FLOOR_FLOAT  1.0e-3
#define AEC_H_ERROR_CEIL_FLOAT   2.0

/* Refined-filter leakage rates, per-hop (per_block_rate_to_per_hop(rate,160,16k)
 * = rate × (160/16000)/(64/16000) = rate × 2.5). Steady + transient (initial-
 * state) profiles. Defined by their computing expression so they fold bit-
 * identically to the Python scale module's module-level constants. */
#define AEC_LEAKAGE_CONVERGED_PER_HOP            (5.0e-5 * 2.5)   /* 1.25e-4 */
#define AEC_LEAKAGE_DIVERGED_PER_HOP             (5.0e-2 * 2.5)   /* 0.125   */
#define AEC_LEAKAGE_CONVERGED_TRANSIENT_PER_HOP  (5.0e-3 * 2.5)   /* 0.0125  */
#define AEC_LEAKAGE_DIVERGED_TRANSIENT_PER_HOP   (5.0e-1 * 2.5)   /* 1.25    */

/* POOR_EXCITATION_COUNTER_INITIAL_HOPS_DEFAULT = blocks_to_hops(1000,160,16k) = 400 */
#define AEC_POOR_EXC_COUNTER_INITIAL_HOPS  400

/* Unitless (same on both sides) */
#define AEC_MAX_ERLE_LF  4.0
#define AEC_MAX_ERLE_HF  1.5
#define AEC_MIN_ERLE     1.0

#define AEC_STATIONARITY_THR_RATIO       10.0
#define AEC_STATIONARITY_BLOCK_FRACTION  0.75
#define AEC_STATIONARITY_ALPHA           0.004
#define AEC_STATIONARITY_ALPHA_INIT      0.04

/* Matched-filter amplitude limits (/32768, NOT /32768^2 — amplitude not PSD) */
#define AEC_MATCHED_FILTER_EXCITATION_LIMIT_FLOAT (150.0 / 32768.0)
#define AEC_MATCHED_FILTER_SATURATION_LIMIT_FLOAT (32000.0 / 32768.0)

/* PSD-scale render/floor constants (echo_canceller3_config.h) */
#define AEC_NORMAL_RENDER_LIMIT_FLOAT (64.0  * AEC_PSD_SCALE_INV)  /* 5.96e-8 */
#define AEC_LOW_RENDER_LIMIT_FLOAT    (256.0 * AEC_PSD_SCALE_INV)  /* 2.38e-7 */
#define AEC_FLOOR_POWER_FLOAT         (128.0 * AEC_PSD_SCALE_INV)  /* 1.19e-7 */
#define AEC_FILTER_NOISE_GATE_POWER_FLOAT (20075344.0 * AEC_PSD_SCALE_INV) /* 0.018697 */
#define AEC_STATIONARITY_MIN_NOISE_POWER_FLOAT (10.0 * AEC_PSD_SCALE_INV)  /* 9.31e-9 */
/* echo_model.noise_gate_power — int16^2 scale, used verbatim (no conversion) */
#define AEC_RESIDUAL_NOISE_GATE_POWER 27509.42

#endif /* AEC_SCALE_H */
