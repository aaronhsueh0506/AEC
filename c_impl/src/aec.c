/* Top-level PBFDKF AEC orchestration with the AEC3-style post-filter chain.
 * Python defines the algorithm; this file is the float32 C implementation. */
#include "aec.h"
#include "aec_debug.h"
#include "aec3_balanced_config.h"
#include "aec3_scale.h"
#include "aec_simd_kernels.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DelayAdjustment enum mirror (delay/delay_types.py). */
#define AEC_DA_NONE          0
#define AEC_DA_BUFFER_FLUSH  1
#define AEC_DA_NEW_DETECTED  2

/* Streaming render/capture call-jitter FIFO budget (milliseconds). Converted
 * to a whole number of hops at create time. 320 ms comfortably covers audio
 * callback scheduling jitter / block-size mismatches without wasting memory
 * (320 ms @ 16 kHz = 5120 samples ≈ 20 KB). NOT the echo delay (that is the
 * downstream ref_ring, delay_buffer_ms). */
#define AEC_STREAM_FIFO_MS 320

/* ───────────────────────── config ──────────────────────────────────────── */

/* Convert fields authored on the legacy 16 kHz/10 ms grid to the final grid.
 * This runs in aec_carve(), after a caller may have overridden fft_size.
 * Retention alphas use a direct power-law conversion. */
static int aec_legacy10ms_hops(float legacy_ms, int hop, int sample_rate) {
    return (hop > 0 && sample_rate > 0)
        ? aec3_ms_to_hops(legacy_ms, hop, sample_rate)
        : (int)lrintf(legacy_ms / 10.0f);
}
static float aec_legacy10ms_alpha(float legacy_alpha, int hop, int sample_rate) {
    return (hop > 0 && sample_rate > 0)
        ? aec3_growth_rehop(legacy_alpha, 160, 16000, hop, sample_rate)
        : legacy_alpha;
}

void aec_config_defaults(AecConfig* cfg, int sr) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    /* CONVENIENCE default only. This factory is the ONE place a default
     * fft_size is applied -- it returns a complete config with a concrete
     * grid, and the core (aec_validate_config / aec_get_mem_size /
     * aec_create / aec_init) never guesses again: fft_size == 0 is rejected
     * there. A caller wanting the 16 kHz alternate 512/256 grid overrides
     * this field afterwards. aec_default_fft_size() reads the same grid
     * table aec_resolve_signal_grid() does, so there is no second copy of
     * the mapping (it returns 0 for an unsupported rate, which the
     * validator then rejects, exactly as the old inline expression did). */
    cfg->fft_size = aec_default_fft_size(sr);
    /* M4 (multi-rate consumption switch): filter_length used to be the fixed
     * 16 kHz bake (832 = 52 ms @ 16000 Hz). Replaced with the actual Python
     * ms policy (AecConfig.__post_init__): 64 ms at sr>=44100 (the high-rate
     * tap budget), 52 ms otherwise. sr<=0 leaves filter_length at 0, so
     * aec_validate_config's existing `filter_length <= 0` check rejects it
     * exactly as it always has (a garbage/zero sr was never a valid config
     * either way). At 16 kHz: 16000*52/1000 = 832 exactly -- byte-identical. */
    cfg->filter_length = (sr <= 0) ? 0
                        : (sr >= 44100) ? (sr * 64 / 1000)
                                        : (sr * 52 / 1000);
    cfg->n_partitions = 0;
    cfg->mu = 0.3f;
    cfg->delta = 1e-8f;
    cfg->enable_cng = 1;
    /* Delay mode: MATCHED + n=5, the shipped default. enable_delay_est is
     * the deprecated mirror of it (see aec.h) and is filled consistently so
     * a config produced here reads the same whichever field a caller looks
     * at; fixed_delay_samples MUST be spelled out (-1) rather than left at
     * the memset 0, since 0 is a legal FIXED delay and would otherwise make
     * the default config an illegal MATCHED+fixed combination. */
    cfg->delay_mode = AEC_DELAY_MATCHED;
    cfg->fixed_delay_samples = -1;
    cfg->enable_delay_est = 1;
    cfg->enable_highpass = 1;
    cfg->enable_saturation = 1;
    cfg->enable_shadow = 1;
    cfg->enable_res = 1;
    cfg->saturation_softclip_ref = 1;
    /* shadow_err_alpha/warmup_frames/epc_hangover/ne_recent_hold/
     * filter_misadjustment_{stable,hangover}_frames below are set here to
     * their RAW legacy-hop=160/sample_rate=16000 (10 ms) literals -- NOT
     * retimed at this point. aec_config_defaults() only knows the DEFAULT
     * fft_size for `sr`; a caller MAY still override cfg.fft_size afterward
     * (e.g. aec_wav.c's --fft-size flag, test_rate_structural.c's alternate
     * 16000/512 grid selection) before ever calling aec_create()/aec_init(),
     * so retiming here against the (possibly stale) default fft_size would
     * silently freeze the wrong grid. The actual retiming happens exactly
     * once, in aec_carve() (2026-08 gap-fix), against the FINAL resolved
     * hop derived from cfg->fft_size at construction time -- see that
     * function's own comment. Must match config.py's AecConfig.__post_init__
     * (Python side has no equivalent post-construction grid-override path;
     * see that file's own comment for why the two sides differ here). */
    cfg->shadow_err_alpha = 0.80f;
    cfg->shadow_mu_min = 0.5f;
    cfg->shadow_mu_nlms = 0.5f;
    cfg->warmup_frames = 100;
    cfg->epc_hangover = 20;
    cfg->epc_total_rise = 1.5f;
    cfg->epc_delta_threshold = 0.3f;
    cfg->epc_mu_floor = 0.5f;
    cfg->max_delay_ms = 1024.0f;
    cfg->delay_buffer_ms = 2048.0f;
    cfg->delay_est_init_s = 0.3f;
    cfg->delay_est_period_s = 0.5f;
    /* Matched-filter bank size (mirrors config.py's delay_num_filters). 5 =
     * unchanged AEC3 geometry; see aec.h for the embedded compute knob. */
    cfg->delay_num_filters = DA_NUM_FILTERS;
    cfg->highpass_cutoff_hz = 80.0f;
    cfg->saturation_threshold = 0.95f;
    cfg->kalman_q_high = 1e-3f;
    cfg->kalman_q_low = 1e-6f;
    cfg->delay_acquire_protect_converged = 1;
    /* Warm tap-transfer (v3.24.1, default ON) + its gate threshold; option-A
     * (block realign) default-OFF. Mirrors config.py:266/271/286. */
    cfg->delay_acquire_warm_transfer = 1;
    cfg->delay_acquire_inst_erle_db = 4.0f;
    cfg->delay_acquire_protect_inst_erle = 0;
    cfg->delay_backward_quarantine_enabled = 0;
    cfg->delay_backward_quarantine_s = 1.0f;
    /* DT-deg recovery stack (default ON, mirrors Python 16285fd). */
    cfg->dt_aware_recovery_soft = 1;
    cfg->dt_aware_res_floor_enabled = 1;
    /* -16 (re-tuned from -20 alongside constraint_round_robin): round-robin's
     * deeper linear convergence lifts FS echo, freeing headroom to raise this
     * DT-only floor and neutralise round-robin's DT-deg cost. */
    cfg->min_gain_floor_dt_db = -16.0f;
    cfg->ne_recent_threshold = 0.3f;   /* float32-by-design (Python bit-exact parity retired) */
    cfg->ne_recent_hold = 150;
    /* ne_recent_sustain: genuine event count (consecutive near-end-active
     * hops required to ARM the hold above, not a duration itself) --
     * intentionally left unretimed. Out of scope for this pass regardless
     * (not part of the audited constant batch). */
    cfg->ne_recent_sustain = 3;
    cfg->min_gain_floor_far_active_db = -28.0f;   /* balanced */
    cfg->filter_misadjustment_stable_frames = 30;
    cfg->filter_misadjustment_hangover_frames = 100;
    cfg->filter_misadjustment_scale_min = 0.5f;
    cfg->filter_misadjustment_scale_max = 2.0f;
}

void aec_config_from_preset(AecConfig* cfg, AecPreset p, int sr) {
    aec_config_defaults(cfg, sr);
    /* mild / balanced / aggressive differ ONLY in the far-active min-gain
     * floor (the SuppressionGain split-floor power axis). */
    switch (p) {
        case AEC_PRESET_MILD:     cfg->min_gain_floor_far_active_db = -20.0f; break;
        case AEC_PRESET_BALANCED:   cfg->min_gain_floor_far_active_db = -28.0f; break;
        case AEC_PRESET_AGGRESSIVE: cfg->min_gain_floor_far_active_db = -38.0f; break;
        /* preset enum out of range (F05): no default case existed before —
         * an out-of-enum-range int silently left min_gain_floor_far_active_db
         * at aec_config_defaults' -28 dB (balanced), which is already a safe,
         * defined fallback. Made explicit here rather than left implicit. */
        default: cfg->min_gain_floor_far_active_db = -28.0f; break;
    }
}

/* ── Signal-grid resolver: THE single source of truth ─────────────────────
 * Every (sample_rate, fft_size) admissibility question and every
 * frame/hop/bin derivation in this library goes through the table below,
 * via aec_resolve_signal_grid(). aec_validate_config(), aec_derive_dims()
 * (and therefore aec_get_mem_size() / aec_create() / aec_init()) and
 * aec_is_valid_sample_rate() are all thin readers of it -- there is no
 * second place that decides "is this grid legal" or re-derives hop from
 * fft. See aec.h's AecSignalGrid comment for the public contract.
 *
 * The multi-rate implementation widened the accepted set from {16000} to
 * {8000, 16000, 48000}: the M2 per-rate coefficient/threshold tables
 * (aec3_balanced_config.h's R8K-/R48K- blocks + AEC3B_RATE_TABLE) and the
 * M4 consumption switch (aec_carve / aec3_post_chain_reset resolving every
 * rate-varying dimension through aec3b_rate_cfg()) have landed and been
 * proven 16 kHz byte-identical; the per-rate verification suite
 * (parity_aec_e2e, gen_delay_c_golden/parity_delay, test_static_aec,
 * test_rate_structural) covers 8000/16000/48000 end to end. 44100 and any
 * other rate stay rejected -- no per-rate tables exist for them.
 *
 * AEC3B_RATE_TABLE carries its own copy of {n_bins, fft_size, block_size,
 * hop_size} alongside ~40 tuning constants. It is NOT a second resolver:
 * nothing consults it to decide admissibility, and
 * test_rate_structural.c's grid-lockstep section pins it row-for-row
 * against this table so the two can never disagree.
 *
 * 8 kHz is a LEGACY grid (plan §11.2, decided): supported by this
 * standalone library and by the Audio_ALG MONO pipeline (audio_pipeline.c
 * accepts/tests it as a 4th grid -- see Audio_ALG/pipelines/README.md
 * "Parameter Alignment"), but NOT a product grid and NOT supported by the
 * Audio_ALG 4-CHANNEL pipeline, whose public API contracts to exactly the
 * three product grids (see 4ch_aec_bf_nr_res/README.md and 4aec_nr_res.c's
 * explicit sample_rate check). It is kept because real tests depend on it;
 * it is flagged rather than left to sit anonymously in a "guessed" path. */
static const AecSignalGrid AEC_GRID_TABLE[] = {
    /* sr,    fft, frame,  hop, n_freqs, is_legacy */
    { 16000,  256,   256,  128,     129, 0 },   /* product, 16 kHz default   */
    { 16000,  512,   512,  256,     257, 0 },   /* product, 16 kHz alternate */
    { 48000, 1024,  1024,  512,     513, 0 },   /* product                   */
    {  8000,  256,   256,  128,     129, 1 },   /* LEGACY, not a product grid*/
};

int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out) {
    /* fft_size == 0 ("auto") never resolves: 16 kHz has two production
     * grids and guessing would silently pick one. See aec.h. */
    if (fft_size <= 0) return 0;
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate &&
            AEC_GRID_TABLE[i].fft_size == fft_size) {
            if (out) *out = AEC_GRID_TABLE[i];
            return 1;
        }
    }
    return 0;
}

int aec_is_valid_sample_rate(int sample_rate) {
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate) return 1;
    }
    return 0;
}

int aec_default_fft_size(int sample_rate) {
    /* The convenience-default policy, in ONE place: the first table row for
     * the rate is its product default (16 kHz -> 256, the low-latency/
     * low-compute grid; 48 kHz -> 1024; 8 kHz -> 256). Callers that want the
     * 16 kHz alternate must ask for 512 explicitly -- that is the whole
     * reason the core refuses fft_size == 0. */
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate)
            return AEC_GRID_TABLE[i].fft_size;
    }
    return 0;
}

/* ── delay-mode translation layer ─────────────────────────────────────────
 * See aec.h's aec_config_resolve_delay() doc comment for the mapping table
 * and the rationale. This is the ONE place the deprecated enable_delay_est
 * mirror is read; everything downstream reads cfg->delay_mode.
 *
 * Deliberately NOT a "pick whichever field looks non-default" heuristic:
 * `enable_delay_est == 1` is both its default AND its only informative
 * value for MATCHED, so it can never contradict a delay_mode a caller set
 * -- the two non-MATCHED delay_mode values are exactly the two outcomes
 * clearing the legacy flag can select. Hence no conflict case exists and
 * this function's only failure is an out-of-enum delay_mode. */
int aec_config_resolve_delay(AecConfig* cfg) {
    if (!cfg) return -1;
    if (cfg->delay_mode != AEC_DELAY_MATCHED &&
        cfg->delay_mode != AEC_DELAY_FIXED &&
        cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED) return -1;
    if (!cfg->enable_delay_est && cfg->delay_mode == AEC_DELAY_MATCHED) {
        /* Legacy shape: the flag was cleared and delay_mode is untouched, so
         * the legacy fields still describe the intent. fixed_delay_samples
         * disambiguates "apply a measured delay" from "far is pre-aligned"
         * exactly as the pre-delay_mode Python orchestrator did. */
        cfg->delay_mode = (cfg->fixed_delay_samples >= 0)
                          ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    }
    /* Rewrite the mirror so the config is self-consistent afterwards (also
     * what makes this function idempotent). */
    cfg->enable_delay_est = (cfg->delay_mode == AEC_DELAY_MATCHED) ? 1 : 0;
    return 0;
}

/* Resolve a caller's config ONCE, on the way in. Every public entry point
 * (aec_create / aec_get_mem_size / aec_get_mem_breakdown / aec_init) must
 * work off byte-identical resolved input, or a size query and the carve it
 * budgets for could disagree about the mode; funnelling all four through
 * this one helper is what makes that lockstep structural rather than a
 * convention four call sites have to keep repeating.
 *
 * Cannot fail: every caller runs aec_validate_config() on the same input
 * first, and an out-of-enum delay_mode is aec_config_resolve_delay()'s only
 * failure. Returns by value so the caller owns the copy the rest of its
 * body reads (the caller's own `cfg_in` is never mutated). */
static AecConfig aec_resolved_config(const AecConfig* cfg_in) {
    AecConfig cfg = *cfg_in;
    (void)aec_config_resolve_delay(&cfg);
    return cfg;
}

/* "Does this config have a reference alignment ring?" -- MATCHED and FIXED
 * do, EXTERNAL_ALIGNED does not (the caller pre-aligned; nothing to buffer).
 * Mirrors Python's `_delay_active`, and is a DIFFERENT question from "is
 * there a matched-filter estimator" (that is MATCHED only, Aec::has_delay).
 * The two agreed before FIXED existed, which is exactly why the predicate
 * gets a name here instead of being open-coded at each site. */
static inline int aec_ring_active(const AecConfig* cfg) {
    return cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED;
}

/* Single source of truth for AecConfig bounds-checking (F05: "no sample-
 * rate/config validation anywhere"). aec_create / aec_get_mem_size /
 * aec_init all run this before touching cfg-derived sizes or state; none of
 * them assert on the release path — an invalid cfg is reported through each
 * function's existing failure convention (aec_create: nonzero; aec_get_
 * mem_size: 0; aec_init: NULL).
 *
 * Bounds are "generous but finite" around each field's shipped default (see
 * aec_config_defaults above): wide enough that every preset/tuning value in
 * this repo passes (verified: mild/balanced/aggressive all pass), tight
 * enough that a corrupted or garbage-filled AecConfig cannot reach the size/
 * pool arithmetic below with a value that overflows or loops unboundedly
 * (e.g. a negative/huge filter_length or n_partitions). */
static int aec_validate_config(const AecConfig* cfg_in) {
    if (!cfg_in) return 0;
    /* The deprecated enable_delay_est mirror is bounds-checked HERE, on the
     * caller's own value, before aec_config_resolve_delay() rewrites it --
     * otherwise a garbage non-{0,1} value would slip through as "truthy"
     * and then be silently normalised to 1. */
    if (cfg_in->enable_delay_est != 0 && cfg_in->enable_delay_est != 1) return 0;
    /* Everything below validates the RESOLVED config: delay_mode is the
     * single source of truth once the translation layer has run. */
    AecConfig resolved = *cfg_in;
    if (aec_config_resolve_delay(&resolved) != 0) return 0;   /* bad enum */
    const AecConfig* cfg = &resolved;

    /* Signal grid: ONE resolver call, no inline (sample_rate, fft_size)
     * table of its own. This also rejects fft_size == 0 -- the core never
     * guesses a grid, since 16 kHz has two production ones (see
     * aec_resolve_signal_grid). aec_derive_dims() below resolves the SAME
     * pair through the SAME function, so validation and sizing cannot see
     * different geometry. */
    if (!aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, NULL)) return 0;
    if (cfg->filter_length <= 0 || cfg->filter_length > 4096) return 0;
    if (cfg->n_partitions != 0 &&
        (cfg->n_partitions < 1 || cfg->n_partitions > 256)) return 0;

#define AEC_CK_BOOL(field) \
    do { if (cfg->field != 0 && cfg->field != 1) return 0; } while (0)
#define AEC_CK_RANGE_F(field, lo, hi) \
    do { if (!isfinite(cfg->field) || cfg->field < (lo) || cfg->field > (hi)) return 0; } while (0)
#define AEC_CK_RANGE_I(field, lo, hi) \
    do { if (cfg->field < (lo) || cfg->field > (hi)) return 0; } while (0)

    /* toggles (15) — enable_delay_est is checked above, pre-translation. */
    AEC_CK_BOOL(enable_cng);
    AEC_CK_BOOL(enable_highpass);
    AEC_CK_BOOL(enable_saturation);
    AEC_CK_BOOL(enable_shadow);
    AEC_CK_BOOL(enable_res);
    AEC_CK_BOOL(saturation_softclip_ref);
    AEC_CK_BOOL(return_res_context);
    AEC_CK_BOOL(spatial_linear_context);
    AEC_CK_BOOL(delay_acquire_protect_converged);
    AEC_CK_BOOL(delay_acquire_warm_transfer);
    AEC_CK_BOOL(delay_acquire_protect_inst_erle);
    AEC_CK_BOOL(delay_backward_quarantine_enabled);
    AEC_CK_BOOL(dt_aware_recovery_soft);
    AEC_CK_BOOL(dt_aware_res_floor_enabled);
    /* spatial_linear_context's whole premise is that apply_output() (Step 21)
     * never reads the skipped gain -- true only when context_only holds
     * (enable_res==0 && return_res_context==1). Reject any other
     * combination here rather than relying on the aec3_post_run() debug
     * assert alone. */
    if (cfg->spatial_linear_context &&
        (cfg->enable_res || !cfg->return_res_context)) return 0;

    /* scalar tunables (22 floats) — generous, finite ranges around default */
    AEC_CK_RANGE_F(mu,                          0.0f,    10.0f);
    AEC_CK_RANGE_F(delta,                       0.0f,     1.0f);
    AEC_CK_RANGE_F(shadow_err_alpha,            0.0f,     1.0f);
    AEC_CK_RANGE_F(shadow_mu_min,               0.0f,    10.0f);
    AEC_CK_RANGE_F(shadow_mu_nlms,              0.0f,    10.0f);
    AEC_CK_RANGE_F(epc_total_rise,              0.0f,  1000.0f);
    AEC_CK_RANGE_F(epc_delta_threshold,         0.0f,  1000.0f);
    AEC_CK_RANGE_F(epc_mu_floor,                0.0f,    10.0f);
    AEC_CK_RANGE_F(max_delay_ms,                0.0f, 60000.0f);
    AEC_CK_RANGE_F(delay_buffer_ms,             0.0f,120000.0f);
    AEC_CK_RANGE_F(delay_est_init_s,            0.0f,  3600.0f);
    AEC_CK_RANGE_F(delay_est_period_s,          0.0f,  3600.0f);
    AEC_CK_RANGE_F(delay_backward_quarantine_s, 0.0f,  3600.0f);
    /* When enable_highpass
     * is set, this field is fed straight into hpf_init() (aec_carve, below)
     * which silently returns NULL — dropping the HPF entirely, with no error
     * surfaced — whenever audio_common's hpf_params_valid() (hpf.c) rejects
     * the (cutoff_hz, sample_rate) pair. hpf_params_valid is *rate-relative*
     * (cutoff must stay < 0.45*sample_rate, matching hpf_compute_coeffs's
     * tan(wc/2) margin from the pi/2 singularity), while this validator's
     * old bound was a flat, rate-blind [0, 20000] Hz range: e.g.
     * {sample_rate=8000, highpass_cutoff_hz=4000} used to pass here but
     * made hpf_init() return NULL underneath, silently constructing an AEC
     * instance with no mic-path HPF even though the caller asked for one.
     * Mirror hpf_params_valid() exactly — same isfinite/>0/0.45-margin
     * checks, same double-vs-float comparison shape — so this validator and
     * hpf_init can never disagree. When enable_highpass is 0 the field is
     * inert (aec_carve's `if (cfg->enable_highpass)` guard means it never
     * reaches hpf_init), so it keeps the old flat generous range — the same
     * treatment every other enable_*-gated tunable in this function gets
     * (e.g. shadow_mu_min/shadow_mu_nlms stay unconditionally range-checked
     * whether or not enable_shadow is set): inert fields are still bounded,
     * just not rate-relative. */
    if (cfg->enable_highpass) {
        if (!isfinite(cfg->highpass_cutoff_hz) || cfg->highpass_cutoff_hz <= 0.0f) return 0;
        if ((double)cfg->highpass_cutoff_hz >= 0.45 * (double)cfg->sample_rate) return 0;
    } else {
        AEC_CK_RANGE_F(highpass_cutoff_hz,      0.0f, 20000.0f);
    }
    AEC_CK_RANGE_F(saturation_threshold,        0.0f,    10.0f);
    AEC_CK_RANGE_F(kalman_q_high,               0.0f,     1.0f);
    AEC_CK_RANGE_F(kalman_q_low,                0.0f,     1.0f);
    AEC_CK_RANGE_F(delay_acquire_inst_erle_db,-100.0f,   100.0f);
    AEC_CK_RANGE_F(min_gain_floor_dt_db,     -300.0f,    50.0f);
    AEC_CK_RANGE_F(ne_recent_threshold,          0.0f,  1000.0f);
    AEC_CK_RANGE_F(min_gain_floor_far_active_db,-300.0f,  50.0f);
    AEC_CK_RANGE_F(filter_misadjustment_scale_min, 0.0f, 1000.0f);
    AEC_CK_RANGE_F(filter_misadjustment_scale_max, 0.0f, 1000.0f);

    /* scalar tunables (6 ints, non-boolean counts) */
    AEC_CK_RANGE_I(warmup_frames,                       0, 1000000);
    AEC_CK_RANGE_I(epc_hangover,                         0, 1000000);
    AEC_CK_RANGE_I(ne_recent_hold,                       0, 1000000);
    AEC_CK_RANGE_I(ne_recent_sustain,                    0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_stable_frames,   0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_hangover_frames, 0, 1000000);
    /* Bank size: unlike the counts above (whose "generous but finite" bounds
     * only exist to stop garbage reaching the size arithmetic), this one is
     * a HARD bound -- DA_NUM_FILTERS is the static array extent in
     * DaMatchedFilter, and 0 filters is a delay estimator that can never
     * report anything. Rejected rather than clamped, matching the Python
     * side's ValueError (config.py __post_init__). */
    AEC_CK_RANGE_I(delay_num_filters,                    1, DA_NUM_FILTERS);

    /* ── delay mode × field compatibility (productization plan §2.1) ──────
     * STRICT by design: an illegal combination is REJECTED, never
     * "normalised" and never silently ignored. The two rules that are easy
     * to get wrong and therefore spelled out:
     *   - a fixed delay is meaningless outside FIXED, so MATCHED /
     *     EXTERNAL_ALIGNED demand the -1 unset sentinel rather than
     *     quietly dropping a delay the caller measured and passed in;
     *   - delay_num_filters sizes the MATCHED bank only, so outside MATCHED
     *     it must still be the default -- accepting n=2 with no matched
     *     filter in existence would let a caller believe they had bought a
     *     compute saving that mode already gives them in full.
     * Both mirror config.py's __post_init__ ValueErrors exactly. */
    switch (cfg->delay_mode) {
        case AEC_DELAY_MATCHED:
            if (cfg->fixed_delay_samples != -1) return 0;
            break;
        case AEC_DELAY_FIXED:
            if (cfg->fixed_delay_samples < 0) return 0;
            /* Generous but finite, and rate-relative: the reference ring is
             * sized `fixed_delay_samples + hop` (aec_ref_ring_samples), so
             * an unbounded value here would drive an unbounded allocation.
             * 120 s matches delay_buffer_ms's own 120000 ms upper bound. */
            if ((long)cfg->fixed_delay_samples > 120L * (long)cfg->sample_rate) return 0;
            if (cfg->delay_num_filters != DA_NUM_FILTERS) return 0;
            break;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            if (cfg->fixed_delay_samples != -1) return 0;
            if (cfg->delay_num_filters != DA_NUM_FILTERS) return 0;
            break;
        default:
            return 0;   /* unreachable: resolve_delay already rejected it */
    }

#undef AEC_CK_BOOL
#undef AEC_CK_RANGE_F
#undef AEC_CK_RANGE_I

    return 1;
}

/* ───────────────────────── helpers ─────────────────────────────────────── */

/* np.mean(x ** 2): square in fp32, sum via numpy-1.26 pairwise f32, divide.
 * Runs entirely in float32-by-design (f64 widening retired — see aec.c/aec.h
 * Stage-2 conversion note). Uses the bit-exact pairwise from aec3_post.
 * `scratch` must hold >= n floats (n is always hop_size at every call site;
 * callers pass a->scr_sq, sized hop_size -- see aec.h struct comment). */
static float mean_sq(const float* x, int n, float* scratch) {
    /* sk_sq_scale_f32(x, 1.0f, ...) is bit-exact to x[i]*x[i] (scale=1.0f is
     * an exact IEEE no-op multiply) -- same established pattern already
     * shipped at suppression_gain.c:112 and aec3_post.c:270/272. */
    sk_sq_scale_f32(x, 1.0f, scratch, n);
    float s = aec3_post_pairwise_sum_f32(scratch, (size_t)n);
    return s / (float)n;
}

/* (historical note: this used to widen to float64 to match a Python
 * reference; the reference comparison is retired, kept float32.) Not used. */

/* Kept (not deleted/renamed) per task scope even though every call site in
 * this file now goes through simd_kernels.h's sk_cmag2_np_f32/_acc_f32 --
 * marked unused to keep the build warning-clean. */
__attribute__((unused)) static float cmag2_c(float r, float i) {
    /* numpy complex64 |z|² via scaled-hypot FMA (cmag2_np). */
    float ar = r < 0.0f ? -r : r;
    float ai = i < 0.0f ? -i : i;
    float larger = ar > ai ? ar : ai;
    float smaller = ar > ai ? ai : ar;
    float m;
    if (larger == 0.0f) m = 0.0f;
    else { float ratio = smaller / larger; m = larger * sqrtf(fmaf(ratio, ratio, 1.0f)); }
    return m * m;
}

/* set Q on a filter to Q_high (Arc-M boost, no per-band scale in balanced). */
static void filter_q_high(PBFDKF* f) {
    int K = f->base.n_freqs;
    for (int k = 0; k < K; ++k) f->Q[k] = f->Q_high[k];
}

/* ── per-bin tuning array build (LF/HF interpolation, suppression_gain.cc) ─ */
/* (Baked arrays come from the generated header; no runtime build needed.) */

/* ── _reset_filter_derived_state (orchestrator 1210-1383). Used by delay
 *    Path-A (delay_first) + Path-B (delay_shift). preserve_render_ema is a
 *    no-op at this layer (render-side trackers live in the StationarityEstimator
 *    which is preserved per preserve_render_side=True). ───────────────────── */
static void aec3_post_chain_reset(Aec* a);   /* fwd */

static void aec_reset_filter_derived_state(Aec* a) {
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);

    filter_convergence_reset(&a->convergence);
    epc_reset(&a->epc);

    a->main_err_smooth = 0.0f;
    a->shadow_err_smooth = 0.0f;
    a->raw_error_power = 0.0f;
    a->near_power = 0.0f;

    a->erle_window_near = 1e-10f;
    a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f;
    a->inst_erle_smooth = 1.0f;
    a->erle_slope_len = 0;     /* mirror _erle_slope_buf.clear() (orch 1176) */
    a->erle_slope_head = 0;

    a->simple_mu_ratio = 1.0f;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;          /* _per_bin_mu_scale = None */

    a->erl_estimate = 0.1f;
    a->epc_render_forced_remaining = 0;
    doubletalk_reset(&a->dt_analyzer);
    a->stat_dt_hangover = 0;

    a->shadow_frame_count = 0;
    shadow_copy_reset(&a->regime);

    /* AEC3 post chain (filter-output-derived); render-side preserved. */
    aec3_post_chain_reset(a);

    /* Re-arm warmup with high Q. */
    filter_q_high(&a->main_filter);
    /* shadow is PBFDAF (no Q) — nothing to boost. */
    int half = a->cfg.warmup_frames / 2;
    if (a->warmup_frames_remaining < half) a->warmup_frames_remaining = half;
    a->warmup_far_active = 0;

    /* clear pending delay shift state */
    a->pending_delay = -1;
    a->has_pending = 0;
    a->pending_delay_ttl = 0;

    a->last_erle_windowed = 0.0f;
}

/* _get_simple_mu_scale (orchestrator 1456-1476). Writes a per-bin array into
 * a->per_bin_mu_scale when one is active, returns scalar otherwise; sets
 * *out_is_array. */
static float get_simple_mu_scale(Aec* a, int* out_is_array) {
    float mu_min = a->cfg.shadow_mu_min;
    *out_is_array = 0;
    if (a->warmup_frames_remaining > 0) {
        if (a->warmup_far_active) a->warmup_frames_remaining--;
        if (a->simple_mu_ratio < 0.2f) {
            return (a->simple_mu_ratio > 0.2f) ? a->simple_mu_ratio : 0.2f;  /* max(0.2,ratio) */
        }
        float v = a->simple_mu_ratio + 0.2f;       /* min(1.0, max(0.5, .)) */
        if (v < 0.5f) v = 0.5f;
        if (v > 1.0f) v = 1.0f;
        return v;
    }
    if (!a->convergence.converged) { if (mu_min < 0.3f) mu_min = 0.3f; }
    else                          { if (mu_min < 0.2f) mu_min = 0.2f; }
    if (a->has_per_bin_mu) {
        /* np.maximum(_per_bin_mu_scale, mu_min) (f32 array vs f32 scalar). */
        int K = a->n_freqs;
        for (int k = 0; k < K; ++k) {
            float v = a->per_bin_mu_scale[k];
            if (v < mu_min) v = mu_min;
            a->per_bin_mu_scale[k] = v;
        }
        *out_is_array = 1;
        return 0.0f;  /* unused */
    }
    return mu_min + (1.0f - mu_min) * a->simple_mu_ratio;
}

/* _update_simple_mu_ratio (orchestrator 1478-1522). */
/* F2.4 simple-mu is intentionally hop-authored. A wall-clock retime of this
 * four-value state machine failed the two-grid A/B gate; keep the validated
 * values together until the mechanism itself can be made grid-invariant. */
enum { SIMPLE_MU_HOLDOFF_HOPS = 20 };
static const float SIMPLE_MU_ALPHA_ATTACK  = 0.3f;
static const float SIMPLE_MU_ALPHA_HOLD    = 0.99f;
static const float SIMPLE_MU_ALPHA_RELEASE = 0.95f;

static void update_simple_mu_ratio(Aec* a, const float* output,
                                   const float* far_end, int n) {
    float error_power = mean_sq(output, n, a->scr_sq) + 1e-10f;
    float far_power   = mean_sq(far_end, n, a->scr_sq) + 1e-10f;
    if (far_power < 1e-6f && error_power < 1e-6f) return;
    if (far_power > 1e-4f && a->simple_mu_ratio < 0.1f) {
        a->simple_mu_ratio = 0.8f;
        a->simple_mu_holdoff = 0;
        return;
    }
    float ratio = far_power / error_power;
    if (ratio > 1.0f) ratio = 1.0f;
    int K = a->n_freqs;
    /* np.sum(np.abs(spec)**2): cmag2_np per bin → numpy-1.26 pairwise f32 sum;
     * float32 throughout (f64 widening retired). */
    float *e2_echo = a->scr_e2_echo, *e2_near = a->scr_e2_near;
    /* fissioned into two independent per-array kernel fills (elementwise,
     * no data dependency between echo_spec and near_spec) */
    sk_cmag2_np_f32(a->main_filter.base.echo_spec, e2_echo, K);
    sk_cmag2_np_f32(a->main_filter.base.near_spec, e2_near, K);
    float echo_est_pwr = aec3_post_pairwise_sum_f32(e2_echo, (size_t)K) + 1e-10f;
    float near_pwr     = aec3_post_pairwise_sum_f32(e2_near, (size_t)K) + 1e-10f;
    if (near_pwr > 1e-8f) {
        float r2 = echo_est_pwr / near_pwr;
        if (r2 < 0.0f) r2 = 0.0f; if (r2 > 1.0f) r2 = 1.0f;
        float r2_half = r2 * 0.5f;
        if (r2_half > ratio) ratio = r2_half;
    }
    float alpha;
    if (ratio < a->simple_mu_ratio) {
        alpha = SIMPLE_MU_ALPHA_ATTACK;
        /* F2.4 invariant: arm only on a FRESH attack; an ongoing attack must
         * not restart it, or marginal DT re-arms every hop and mu never
         * releases. Mirrors orchestrator.py. */
        if (a->simple_mu_holdoff == 0)
            a->simple_mu_holdoff = SIMPLE_MU_HOLDOFF_HOPS;
    }
    else if (a->simple_mu_holdoff > 0) {
        a->simple_mu_holdoff--;
        alpha = SIMPLE_MU_ALPHA_HOLD;
    }
    else alpha = SIMPLE_MU_ALPHA_RELEASE;
    a->simple_mu_ratio = alpha * a->simple_mu_ratio + (1.0f - alpha) * ratio;
}

#ifdef AEC_TESTING
/* TEST-ONLY hook. Absent from the production library: `make lib` never defines
 * AEC_TESTING, so this symbol is not exported and `make test-no-testing-symbols`
 * asserts that in both directions.
 *
 * It exists because test_rate_structural check (d6) has to drive the update
 * directly. Going through aec_process() cannot substitute: simple_mu_ratio is
 * read EARLIER in the same hop to scale mu, so it feeds back into the very
 * error signal the two-point coefficient recovery has to hold constant, and the
 * two probes would no longer differ in one variable only.
 *
 * Declared in test/aec_test_hooks.h, which is not installed. */
void aec_testing_update_simple_mu_ratio(Aec* a, const float* output,
                                        const float* far_end, int n) {
    update_simple_mu_ratio(a, output, far_end, n);
}
#endif

/* ───────────────────────── construction ────────────────────────────────── */

/* F05: was a signed left shift (`int n = 1; while (n < x) n <<= 1;`) with no
 * bound on x — a corrupted/garbage x approaching INT_MAX loops until n
 * overflows int (UB) rather than terminating. Shift is now done in unsigned
 * arithmetic (well-defined on overflow) and x is capped at 1<<20 (1,048,576 —
 * >2000x any block_size this repo derives from a validated 16 kHz config, so
 * the cap never engages for a real config; it only bounds a pathological
 * input). Byte-identical to the old expression for every x this repo ever
 * passes (block_size <= a few thousand). */
static int next_pow2(int x) {
    if (x <= 1) return 1;
    if (x > (1 << 20)) x = (1 << 20);
    unsigned int n = 1u;
    unsigned int ux = (unsigned int)x;
    while (n < ux) n <<= 1u;
    return (int)n;
}


/* FilterAnalyzer tap materializer (installed with aec_state_set_taps_provider
 * on every AecState init below).
 *
 * The analyzer consumes one hop-sized region of the main filter's impulse
 * response per hop and is the only reader of a->filter_taps_full, so only the
 * partitions overlapping the span it asks for have to be inverse-transformed;
 * the remaining partitions' taps stay at their previous values, unread. The
 * span arrives in tap indices, so the partition mapping is derived from the
 * filter's own layout rather than assuming the region lines up with a
 * partition boundary -- it does not, for the sweep that follows an analyzer
 * reset (that sweep's regions are offset by one sample until the next wrap),
 * and it need not for a grid whose region size differs from the tap extent of
 * one partition. */
static void aec_fill_filter_taps(void* ctx, int first, int last) {
    Aec* a = (Aec*)ctx;
    PBFDAF* f = &a->main_filter.base;
    int hop = f->hop_size;
    int p_first, p_last;
    if (hop <= 0 || f->n_partitions <= 0 || last < first || first < 0) return;
    p_first = first / hop;
    p_last  = last  / hop;
    if (p_last > f->n_partitions - 1) p_last = f->n_partitions - 1;
    if (p_last < p_first) return;
    pbfdaf_get_time_domain_filter_range(f, p_first, p_last - p_first + 1,
                                        a->filter_taps_full);
}

/* Clear + recreate the AEC3 post chain sub-objects (mirrors _reset_aec3_post
 * with preserve_render_side=True: StationarityEstimator + non_zero_render_seen
 * + active_hops are preserved; everything else re-init'd). */
static void aec3_post_chain_reset(Aec* a) {
    /* aec3_post_reset clears OLA / CNG / coherence EMAs / avg-reverb. */
    aec3_post_reset(&a->post);
    /* URO crossfade memory (_form_prev_output_time / _form_last_selection /
     * _refined_filter_output_last_selected) — _reset_aec3_post clears it
     * (orchestrator 1189-1190). */
    linear_filter_select_reset(&a->a3_lfs);
    /* M4 (multi-rate consumption switch): rate-varying REE absolute-power
     * floats below now read through this lookup. cfg.sample_rate never
     * changes across a reset (aec_validate_config already ran once at
     * construction), so this always hits -- the `rd ? rd->x : AEC3B_x`
     * fallback below is a pure defensive no-op for the validated {16000}
     * whitelist (never actually taken). */
    const Aec3BalancedRateDims* rd = aec3b_rate_cfg(
        a->cfg.sample_rate, a->cfg.fft_size);
    /* AecState + REE + SuppressionGain are recreated in Python; here we
     * re-init in place (same backing storage). */
    {
        AecStateConfig acfg;
        aec_state_config_defaults(&acfg);
        acfg.n_bins = a->n_freqs;
        acfg.num_capture_channels = AEC3B_ST_NUM_CAPTURE_CHANNELS;
        acfg.hop_size = a->hop_size;
        /* AecStateConfig.sample_rate was never populated at this call site
         * (stayed at aec_state_config_defaults()'s 16000 regardless of the
         * real cfg.sample_rate) -- every hop/sr-scaled threshold downstream
         * silently behaved as if sample_rate==16000 at every grid. */
        acfg.sample_rate = a->cfg.sample_rate;
        acfg.enable_filter_analyzer = AEC3B_ST_ENABLE_FILTER_ANALYZER;
        /* AEC3B_ST_ERLE_STARTUP_HOPS/ST_ERL_STARTUP_HOPS are always the
         * AEC_STATE_STARTUP_HOPS_AUTO sentinel (-1, aec_state.c), never a
         * literal hop count -- resolve_startup_hops() re-derives the real
         * hop count from acfg.hop_size/acfg.sample_rate above (which DO vary
         * across the 8k/16k/48k M4 grids). Do not replace with a literal. */
        acfg.erle_startup_hops = AEC3B_ST_ERLE_STARTUP_HOPS;
        acfg.erl_startup_hops = AEC3B_ST_ERL_STARTUP_HOPS;
        acfg.echo_can_saturate = AEC3B_ST_ECHO_CAN_SATURATE;
        acfg.use_linear_filter = AEC3B_ST_USE_LINEAR_FILTER;
        acfg.conservative_initial_phase = AEC3B_ST_CONSERVATIVE_INITIAL_PHASE;
        acfg.delay_headroom_samples = AEC3B_ST_DELAY_HEADROOM_SAMPLES;
        acfg.initial_state_seconds = AEC3B_ST_INITIAL_STATE_SECONDS;
        acfg.erle_min = AEC3B_ST_ERLE_MIN;
        acfg.erle_max_l = AEC3B_ST_ERLE_MAX_L;
        acfg.erle_max_h = AEC3B_ST_ERLE_MAX_H;
        /* M4: runtime np*hop, not AEC3B_FILTER_TAPS_SIZE (== rd->
         * filter_taps_size at 16 kHz -- mirrors aec_carve's AecStateConfig
         * wiring). */
        acfg.filter_taps_size = a->n_partitions * a->hop_size;
        /* Re-init reuses the same fa_abs_scratch/fa_render_sq_scratch
         * pointers already bound in a->a3_state_st (carved once in
         * aec_carve); fa_scratch_size must match the np*hop capacity that
         * buffer was originally sized to. */
        acfg.fa_scratch_size = a->n_partitions * a->hop_size;
        aec_state_init(&a->a3_state, &acfg, &a->a3_state_st);
        aec_state_set_taps_provider(&a->a3_state, aec_fill_filter_taps, a);
    }
    {
        ReeEchoModelConfig em;
        memset(&em, 0, sizeof(em));
        /* M4: rate-varying REE absolute-power floats via rd (see the
         * fallback-lookup comment above). */
        em.min_noise_floor_power = rd ? rd->ree_min_noise_floor_power
                                      : AEC3B_REE_MIN_NOISE_FLOOR_POWER;
        em.noise_gate_power = AEC3B_REE_NOISE_GATE_POWER_LEGACY;
        em.noise_gate_slope = AEC3B_REE_NOISE_GATE_SLOPE;
        em.stationary_gate_slope = AEC3B_REE_STATIONARY_GATE_SLOPE;
        em.model_reverb_in_nonlinear_mode = AEC3B_REE_MODEL_REVERB_IN_NL;
        /* re-init reuses the same storage pointers already bound in a3_ree. */
        ree_init(&a->a3_ree, a->n_freqs, a->hop_size, a->cfg.sample_rate, &em,
                 AEC3B_REE_DEFAULT_GAIN, AEC3B_REE_TM_GAIN, AEC3B_REE_ERLE_ONSET_COMP,
                 AEC3B_REE_REVERB_DECAY, AEC3B_REE_REVERB_MILD_SCALE,
                 AEC3B_REE_REVERB_ENABLED, AEC3B_REE_REVERB_TAIL_STRENGTH,
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE,
                 AEC3B_USE_STATIONARITY_PROPERTIES,
                 AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
                 AEC3B_REE_NL_R2_ENABLED, AEC3B_REE_NL_R2_ALPHA,
                 rd ? rd->ree_nl_norm_power : AEC3B_REE_NL_NORM_POWER,
                 rd ? rd->ree_residual_noise_gate_power
                    : AEC3B_REE_RESIDUAL_NOISE_GATE_POWER,
                 rd ? rd->ree_noise_floor_hold_hops : AEC3B_REE_NOISE_FLOOR_HOLD_HOPS,
                 AEC3B_REE_USE_FREQ_RESPONSE, AEC3B_REE_REVERB_USE_CONSERVATIVE,
                 rd ? rd->ree_reverb_smoothing_base : AEC3B_REE_REVERB_SMOOTHING_BASE,
                 a->a3_ree.x2_noise_floor, a->a3_ree.x2_noise_floor_counter,
                 a->a3_ree.reverb_model.reverb,
                 a->a3_ree.reverb_freq_resp.tail_response,
                 a->a3_ree.render_history,
                 a->a3_ree.delay_render_buf, a->a3_ree.reverb_render_history,
                 a->a3_ree.last_r2_direct, a->a3_ree.last_r2_reverb,
                 a->a3_ree.scratch);
    }
    /* SuppressionGain recreate (preserves config; clears persistent state). */
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg = sg->cfg;     /* keep config */
        SuppressionGainTuning stun = sg->tun;
        suppression_gain_init(sg, &scfg, &stun, sg->last_gain, sg->last_nearend,
                              sg->last_echo, sg->ma_buf, sg->nearend,
                              sg->weighted_residual, sg->min_gain, sg->max_gain,
                              sg->g_raw, sg->gain, sg->sum_scratch);
    }
}

/* fwd decls: aec_carve() (F03 arena-fication shared carve) and
 * aec_derive_dims() are defined further down, next to aec_get_mem_size()
 * (whose field layout aec_carve's pointer walk must stay in lockstep with),
 * but aec_create() above that point needs to call both. */
static void aec_derive_dims(const AecConfig* cfg,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp, int* o_fifo_cap);
static int aec_carve(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, int fcap, int is_static);

int aec_create(Aec* a, const AecConfig* cfg_in) {
    /* F05/F07: reject before any state is touched — no release-path assert,
     * just the existing int failure convention (0 = success, checked by
     * every caller as `!= 0` or truthy). */
    if (!a || !cfg_in || !aec_validate_config(cfg_in)) return -1;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;

    int hop, blk, fft, K, np, buf_samp, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf_samp, &fcap);

    /* F03 (arena-fication): the ~87 individual per-array mallocs this
     * function used to make (one per sub-module scratch/state array, plus
     * ~200 lines of config boilerplate hand-duplicated from aec_init) are
     * gone. One arena, sized by the exact same aec_get_mem_size() total the
     * static-pool path budgets against (minus the Aec struct term -- the
     * caller already owns `a`, so the arena only needs the part of the pool
     * that would otherwise sit after the struct), carved by the same
     * aec_carve() aec_init() uses below. Byte-for-byte the same initial
     * state as before this refactor (test_static_aec.c / test_lifecycle.c
     * both exercise this). */
    size_t total = aec_get_mem_size(cfg);
    if (total == 0) return -1;
    size_t arena_sz = total - ALIGN16(sizeof(Aec));

    uint8_t* arena = (uint8_t*)malloc(arena_sz);
    if (!arena) return -1;

    memset(a, 0, sizeof(*a));
    if (aec_carve(a, arena, cfg, hop, blk, fft, K, np, buf_samp, fcap,
                  /*is_static=*/0) != 0) {
        /* F04: the shared FFT allocation failed (OOM -- main filter /
         * shadow filter only ever borrow it, never own one), or (R08
         * belt-and-braces, practically unreachable post-validator-fix)
         * hpf_init() rejected its params, partway through the carve.
         * aec_carve() already tore down the shared handle if it had brought
         * it up before returning, so the arena itself is the only thing left
         * to release. */
        free(arena);
        memset(a, 0, sizeof(*a));
        return -1;
    }
    a->heap_arena = arena;
    return 0;
}

/* ── reference alignment ring sizing (ONE checked helper) ─────────────────── */

/* Legacy hop-boundary headroom carried by the MATCHED search ring.
 *
 * Deliberately NOT applied to FIXED (see aec_ref_ring_samples): under
 * MATCHED it is load-bearing rather than mere slack, because ring size feeds
 * the `new_delay <= ref_ring_size - hop` eligibility gate in aec_process()
 * -- i.e. it also decides which estimates the controller is allowed to
 * accept. Changing it would change the default path's audio, so it stays
 * verbatim. */
#define AEC_REF_RING_MATCHED_HEADROOM 4096

/* Reference alignment ring capacity, in samples, for a DELAY-RESOLVED cfg.
 * The single source of truth for that number: aec_get_mem_size() budgets it
 * and aec_carve() carves it from the same call (via aec_derive_dims), so the
 * two cannot drift.
 *
 *   MATCHED   max(delay_buffer_ms, max_delay_ms + AEC_REF_RING_MATCHED_HEADROOM)
 *             -- a SEARCH ring: the applied delay is unknown at init and can
 *             move at any hop, so it is sized for the whole configured
 *             search budget rather than for one delay.
 *   FIXED     fixed_delay_samples + hop_size       (exact, see derivation)
 *   EXTERNAL  0                                    (nothing to buffer)
 *
 * FIXED derivation. Let T be the total sample count written so far; the ring
 * holds absolute samples [T-rs, T). Each hop aec_process() first writes the
 * newest `hop` samples (advancing T by hop), then the delay-compensating read
 * takes absolute samples [T-d-hop, T-d). Validity is therefore exactly
 *
 *     rs >= d + hop
 *
 * and under FIXED `d` is immutable (== cfg.fixed_delay_samples: no estimator
 * exists, and aec_reset() re-seeds the same value), so this is a tight bound
 * on a constant, not a worst case over a moving quantity -- no headroom term
 * is meaningful. Equality is safe: at rs == d+hop the read starts exactly at
 * ref_ring_write, i.e. on the oldest hop in the ring, which is precisely the
 * one this hop's write did NOT overwrite.
 *
 * Byte-exactness against the old (oversized) ring follows from the fill gate
 * `ref_ring_filled >= d + hop`: ref_ring_filled saturates at rs, so for ANY
 * rs >= d+hop the gate first passes at hop index ceil((d+hop)/hop) and the
 * samples served are the same absolute samples. d == 0 is the degenerate
 * case -- a hop-sized, write-only ring (the read is skipped entirely by the
 * `current_delay > 0` guard), kept rather than special-cased to NULL so that
 * FIXED has exactly one ring code path.
 *
 * Returns 0 for EXTERNAL_ALIGNED (no ring wanted) and, defensively, for a
 * capacity that would not fit an int -- callers treat 0-on-a-ring-mode as a
 * hard config failure, so the two cases never merge in practice.
 * aec_validate_config() caps fixed_delay_samples at 120 s of samples
 * (<= 5,760,000) and hop at 512, so the overflow arm is unreachable today. */
static int aec_ref_ring_samples(const AecConfig* cfg, int hop) {
    if (cfg->delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) return 0;
    if (hop <= 0) return 0;
    if (cfg->delay_mode == AEC_DELAY_FIXED) {
        long long need = (long long)cfg->fixed_delay_samples + (long long)hop;
        if (need < (long long)hop || need > 2147483647LL) return 0;
        return (int)need;
    }
    {
        int max_d = (int)(cfg->max_delay_ms * cfg->sample_rate / 1000.0f);
        int buf   = (int)(cfg->delay_buffer_ms * cfg->sample_rate / 1000.0f);
        if (buf < max_d + AEC_REF_RING_MATCHED_HEADROOM)
            buf = max_d + AEC_REF_RING_MATCHED_HEADROOM;
        return buf;
    }
}

/* ── dimension helper (shared by aec_create() and aec_init()) ─────────────── */

/* Derive all frame-dimension parameters from cfg.
 *
 * The frame geometry (fft/frame/hop/n_freqs) comes from the ONE shared
 * resolver -- this function does NOT re-derive `hop = fft/2` or
 * `K = fft/2+1` itself, which is exactly the duplication that let
 * get-mem-size and init drift apart in principle. Everything else below
 * (partitions, ring, FIFO capacity) is genuinely config-dependent and has
 * always lived here.
 *
 * Every caller (aec_get_mem_size / aec_create / aec_init) has already run
 * aec_validate_config(), which resolves the same pair, so the lookup here
 * cannot miss; the zero-fill fallback exists only so a future caller that
 * forgets cannot walk off with uninitialised dimensions. */
static void aec_derive_dims(const AecConfig* cfg,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp, int* o_fifo_cap) {
    AecSignalGrid g;
    if (!aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &g)) {
        *o_hop = 0; *o_blk = 0; *o_fft = 0; *o_K = 0;
        *o_nparts = 0; *o_buf_samp = 0; *o_fifo_cap = 0;
        return;
    }
    int fft = g.fft_size;
    int blk = g.frame_size;
    int hop = g.hop_size;
    *o_hop = hop; *o_blk = blk; *o_fft = fft; *o_K = g.n_freqs;
    int n = cfg->n_partitions;
    if (n <= 0) { n = (cfg->filter_length + hop - 1) / hop; if (n < 1) n = 1; }
    *o_nparts = n;
    /* Reference alignment ring: ONE mode-aware checked helper, shared with
     * aec_get_mem_size()/aec_carve() through this out-param (0 == no ring,
     * i.e. EXTERNAL_ALIGNED). Mirrors the Python orchestrator's
     * ref_ring_samples(). */
    *o_buf_samp = aec_ref_ring_samples(cfg, hop);
    int cap = (AEC_STREAM_FIFO_MS * cfg->sample_rate / 1000 + hop - 1) / hop;
    if (cap < 2) cap = 2;
    /* F09 Variant A': round up to the next power of two. The SPSC ring
     * indexes with `seq % cap` on ever-increasing unsigned sequence numbers
     * that wrap at 2^32 -- that index stays continuous across the wrap only
     * when cap divides 2^32 exactly (a non-power-of-two cap would alias two
     * different (write - read) occupancies onto the same physical slot right
     * at the wrap boundary). All currently supported rates (8/16/48 kHz)
     * already land on cap=32, itself a power of two, so this is a no-op
     * today; it only guards a future rate whose arithmetic lands elsewhere. */
    cap = next_pow2(cap);
    *o_fifo_cap = cap;
}

/* Checked equivalent of `t += ALIGN16(count*elem_size) * reps` — reps
 * identically-sized fields laid out back-to-back (each individually
 * ALIGN16'd by aec_init's pool-carve, so their sum is exactly
 * reps * ALIGN16(count*elem_size)). Saturates to SIZE_MAX on overflow via
 * mem_align.h's ck_* helpers, same as ck_field_size. */
static size_t ck_field_size_reps(size_t total, size_t count, size_t elem_size,
                                  size_t reps) {
    size_t one = ck_align16_size(ck_mul_size(count, elem_size));
    return ck_add_size(total, ck_mul_size(one, reps));
}

size_t aec_get_mem_size(const AecConfig* cfg_in) {
    if (!cfg_in) return 0;
    if (!aec_validate_config(cfg_in)) return 0;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;
    int hop, blk, fft, K, np, buf, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf, &fcap);

    const int ncc  = AEC3B_ST_NUM_CAPTURE_CHANNELS;
    const int rh   = (AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW ? 1 : 0) + 2;
    const Aec3BalancedRateDims* rd = aec3b_rate_cfg(
        cfg->sample_rate, cfg->fft_size);
    if (!rd) return 0;
    const int sg_n = rd->sg_nearend_smoother_n;
    const size_t Kz = (size_t)K;

    /* Checked size arithmetic (F05): every add/multiply/align below
     * saturates to SIZE_MAX on overflow (mem_align.h ck_* helpers) instead
     * of silently wrapping; MEM_SIZE_INVALID(t) at the end catches an
     * overflow anywhere in the chain and this function reports failure (0)
     * rather than a small wrapped total a later aec_init would carve past.
     * Field ORDER/grouping is unchanged from the previous unchecked
     * `t += ALIGN16(...)` walk — only the arithmetic is wrapped, since
     * get_mem_size/aec_init are a lockstep pool-carve pair. */
    size_t t = 0;
    t = ck_field_size(t, 1, sizeof(Aec));
    /* Matched-filter delay estimator (plan step 2). `Aec` no longer embeds
     * the bank/ring/histogram arrays -- DelayAec3 is a metadata struct and
     * its arrays are carved from THIS pool, sized by the resolved
     * (sample_rate, hop, delay_num_filters) triple, so n=1 really is ~23 KB
     * cheaper than n=5 instead of merely cheaper in MACs. Carved only for
     * MATCHED: that is the only mode that constructs an estimator (the
     * FIXED / EXTERNAL_ALIGNED ring differentiation is plan step 3).
     * Must stay in lockstep with the matching carve in aec_carve(). */
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        size_t da_sz = delay_aec3_get_mem_size(cfg->sample_rate, hop,
                                               cfg->delay_num_filters);
        if (da_sz == 0) return 0;   /* sub-config rejected its own inputs */
        t = ck_add_size(t, da_sz);
    }
    /* Reference alignment ring: present for MATCHED and FIXED, absent only
     * for EXTERNAL_ALIGNED (the caller pre-aligned; nothing to buffer).
     * `buf` already carries the mode (aec_ref_ring_samples via
     * aec_derive_dims): 0 means "this mode wants no ring". A ring mode that
     * came back 0 is an unrepresentable capacity, i.e. a config failure --
     * never a silent no-ring instance. Must stay in lockstep with the
     * matching carve in aec_carve(). */
    if (cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED && buf <= 0) return 0;
    if (buf > 0)
        t = ck_field_size(t, (size_t)buf, sizeof(float));
    t = ck_field_size(t, ck_mul_size((size_t)fcap, (size_t)hop), sizeof(float));   /* render_fifo */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* fifo_zero_ref (F09 Variant A' underrun ref) */
    t = ck_field_size(t, (size_t)(K - 2 > 0 ? K - 2 : 1), sizeof(int64_t));
    /* Shared FFT handle: ONE FftHandle for the whole instance (main filter,
     * shadow filter, and aec3_post all borrow it -- see aec_carve()) instead
     * of each carving/owning its own. Carved here, before the main/shadow
     * filters, since both now require an already-constructed handle to
     * borrow at their own init time; kf_sz/af_sz below no longer include a
     * nested-FFT term (pbfdaf_get_mem_size never does, post-Group-7). */
    t = ck_add_size(t, fft_get_mem_size(fft));
    {
        size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
        if (kf_sz == 0) return 0;   /* sub-config rejected its own inputs */
        t = ck_add_size(t, kf_sz);
    }
    if (cfg->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (af_sz == 0) return 0;
        t = ck_add_size(t, af_sz);
    }
    /* RenderActivityDetector pairwise-sum scratch (M3: de-stacked from a
     * fixed `float[1024]` local; ceil(hop/8) blocks -- see detectors.c). */
    t = ck_field_size(t, (size_t)((hop + 7) / 8), sizeof(float));
    /* aec3_post (21) */
    t = ck_field_size_reps(t, Kz, sizeof(float), 6);   /* avg_rev y2s n2 n2i sye_re sye_im */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* syy see */
    t = ck_field_size(t, (size_t)blk, sizeof(float));  /* ola */
    t = ck_field_size_reps(t, Kz, sizeof(float), 8);   /* np_ fp_ ep_ erp_ cpe x2r cn nf */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* cgm */
    t = ck_field_size(t, Kz, sizeof(Complex));         /* eout */
    t = ck_field_size(t, (size_t)fft, sizeof(float));  /* eout_full */
    /* AecStateStorage (14) */
    t = ck_field_size_reps(t, Kz, sizeof(float), 5);   /* erle_max/erle/oc/unb/during */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* erle_coming_onset */
    t = ck_field_size(t, Kz, sizeof(int32_t));         /* erle_hold */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* erle_y2_acc/e2_acc */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* erle_low_render */
    t = ck_field_size(t, Kz, sizeof(float));           /* erl */
    t = ck_field_size(t, (size_t)(K - 2), sizeof(int)); /* erl_hold */
    t = ck_field_size(t, (size_t)ncc, sizeof(int));    /* filter_delays_blocks */
    /* M4 (multi-rate consumption switch): fa_h_highpass is now sized from
     * the runtime full impulse-response length (n_partitions*hop) instead
     * of the 16 kHz-baked AEC3B_FILTER_TAPS_SIZE macro -- equal to
     * rd->filter_taps_size (960 @ 16 kHz) by construction for any validated
     * config, same runtime dims the fa_abs_scratch term below (M3) already
     * uses. ck_mul_size keeps the same overflow-checked arithmetic as every
     * other field in this function. */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* fa_h_highpass */
    /* FilterAnalyzer de-stacked fa_update() scratch (M3): sized from the
     * same runtime dims (n_partitions*hop / hop) that fa_update actually
     * bounds its writes by. */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* fa_abs_scratch */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* fa_render_sq_scratch */
    /* ResidualEchoEstimator (10) */
    t = ck_field_size(t, Kz, sizeof(float));           /* x2_nf */
    t = ck_field_size(t, Kz, sizeof(int));             /* x2_nf_c */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* rm_st rt_st */
    t = ck_field_size(t, ck_mul_size((size_t)rh, Kz), sizeof(float)); /* rh_st */
    t = ck_field_size_reps(t, ck_mul_size((size_t)REE_DELAY_BUF_SIZE, Kz),
                           sizeof(float), 2);          /* drd rrd */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* ld_st lr_st scr_st */
    /* SuppressionGain (11) */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* last_gain/ne/echo */
    t = ck_field_size(t, ck_mul_size((size_t)sg_n, Kz), sizeof(float)); /* ma */
    t = ck_field_size_reps(t, Kz, sizeof(float), 7);   /* ne wr ming maxg graw gout gsum */
    /* StationarityEstimator (4) */
    t = ck_field_size(t, Kz, sizeof(float));           /* stat_noise */
    t = ck_field_size(t, Kz, sizeof(int32_t));         /* stat_hang */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_flags */
    t = ck_field_size(t, ck_mul_size((size_t)16, Kz), sizeof(float)); /* stat_hist */
    /* LinearFilterSelect (4) */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 2); /* prev_output_time e_form */
    t = ck_field_size(t, (size_t)blk, sizeof(float));  /* block_win */
    t = ck_field_size(t, Kz, sizeof(Complex));         /* lfs sel_esw */
    /* LinearFilterSelect de-stacked scratch (4): were fixed-size stack locals
     * (hop/fft_size entries, up to 8192) in linear_filter_select() —
     * a stack-overflow hazard on small embedded task stacks. */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 3); /* scr_sq scr_sref scr_scoa */
    t = ck_field_size(t, (size_t)fft, sizeof(float));  /* scr_tin */
    /* Aec3PostRunScratch (19) */
    t = ck_field_size_reps(t, Kz, sizeof(Complex), 4); /* sel_esw sel_echo nsw_e1 ybase */
    t = ck_field_size_reps(t, Kz, sizeof(float), 9);   /* abs_near..x2_past */
    t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(float)); /* w_mag2 */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* render_block_scaled */
    /* bridge_taps (formerly here, [fft_size]) removed: it was write-only-by-
     * nobody once the filter_state_bridge IRFFT call was deleted from
     * aec3_post.c -- see that file's Step-5 comment. */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* r2 r2_unb nearend_pwr */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_mask */
    /* hop scratch (9) */
    t = ck_field_size(t, Kz, sizeof(float));           /* per_bin_mu_scale */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 5); /* near_hop far_hop raw shadow final */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* filter_taps_full */
    /* per-hop freq-bin scratch (12; see aec.h struct comment) */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* scr_sq */
    t = ck_field_size_reps(t, Kz, sizeof(float), 11);  /* scr_e2_echo .. scr_erl_arr */
    if (cfg->enable_highpass) t = ck_field_size(t, 1, hpf_get_mem_size()); /* mic-path HPF */

    if (MEM_SIZE_INVALID(t)) return 0;
    return t;
}

/* Static-pool memory breakdown (plan §3.4.6 test 6; see AecMemBreakdown's
 * doc comment in aec.h). Deliberately does NOT re-sum the fields
 * aec_get_mem_size() walks above -- total_bytes is that exact call, so a
 * caller (aec_wav.c's --print-mem-size) reads the SAME number
 * aec_get_mem_size() would give it, not a second, driftable computation of
 * it. estimator_bytes/ring_bytes reuse the identical helpers
 * aec_get_mem_size() itself carves against (delay_aec3_get_mem_size(),
 * aec_derive_dims()'s buf out-param), so a query here can never disagree
 * with what aec_carve() actually lays out. */
int aec_get_mem_breakdown(const AecConfig* cfg_in, AecMemBreakdown* out) {
    if (!out) return 0;
    memset(out, 0, sizeof(*out));
    if (!cfg_in) return 0;

    size_t total = aec_get_mem_size(cfg_in);
    if (total == 0) return 0;

    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;

    int hop, blk, fft, K, np, buf, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf, &fcap);

    size_t est = 0;
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        est = delay_aec3_get_mem_size(cfg->sample_rate, hop,
                                       cfg->delay_num_filters);
    }
    size_t ring = (buf > 0) ? ck_field_size(0, (size_t)buf, sizeof(float)) : 0;

    out->total_bytes     = total;
    out->estimator_bytes = est;
    out->ring_bytes      = ring;
    return 1;
}

/* Place Aec + all backing arrays in the provided pool; no malloc called.
 * Returns (Aec*)mem on success, NULL on invalid inputs, misaligned base, or
 * undersized pool (F05/F07). */
/* ── shared arena carve (F03) ──────────────────────────────────────────────
 * Both aec_init() (caller-owned pool, Aec placed at mem[0]) and aec_create()
 * (caller-owned Aec, followed by a single malloc'd arena) carve the
 * identical set of sub-module buffers in the identical order from a raw
 * byte cursor -- this is that shared body (formerly hand-duplicated: ~200
 * lines of matching config boilerplate plus, on the aec_create side, ~400
 * lines of individual per-array mallocs where this carve does one pointer
 * bump per array instead). `a` must already be zeroed by the caller; `ptr`
 * is the first byte of the arena (aec_init: mem + ALIGN16(sizeof(Aec));
 * aec_create: the base of the freshly malloc'd arena). `is_static` records
 * the two paths' only remaining difference, read by aec_destroy(): 1 means
 * the caller owns the memory and aec_destroy() must not free it.
 *
 * Returns 0 on success, -1 iff the shared FFT allocation failed (one
 * FftHandle for the whole instance -- main filter and shadow filter only
 * ever borrow it, they no longer carve/own a private one), or iff the
 * mic-path HPF's hpf_init()
 * rejected its (cutoff_hz, sample_rate) pair (R08 belt-and-braces: with
 * aec_validate_config's highpass_cutoff_hz check now mirroring
 * hpf_params_valid exactly, every {enable_highpass, cutoff_hz, sample_rate}
 * combination that reaches this carve has already been proven acceptable to
 * hpf_init, so this branch is unreachable in practice -- kept as a defined
 * internal-error path rather than silently constructing an instance with no
 * HPF). Those are the only sub-inits below that can fail post-arena-
 * fication: every other sub-init here is a pure pointer-bump against a
 * buffer aec_get_mem_size() already proved big enough, so it cannot fail.
 * On failure, any nested FFT handle that DID come up earlier in this same
 * carve (and therefore owns a real allocation outside the pool/arena -- the
 * NE10 backend's R2C/C2R twiddle config) is torn down before returning, so
 * the caller has only the pool/arena bytes themselves left to deal with. */

/* Wall-clock-preserving thresholds for poor_coarse/coarse_reset/
 * leakage_div_sustain/stat_dt_hangover -- shared by aec_carve() (construction)
 * and aec_reset() (mirrors orchestrator.py's live blocks_to_hops(5)/(25) and
 * ms_to_hops(50.0)/(800.0) calls; see aec.h field comments). Single source of
 * truth so the two call sites cannot drift the way they briefly did before
 * this helper existed. */
static void aec_recompute_wallclock_thresholds(Aec* a, int hop, int sample_rate) {
    a->poor_coarse_threshold_hops = aec3_blocks_to_hops(5, hop, sample_rate);
    a->coarse_reset_hangover_hops = aec3_blocks_to_hops(25, hop, sample_rate);
    a->leakage_div_sustain_hops = aec3_ms_to_hops(50.0f, hop, sample_rate);
    a->stat_dt_hangover_hops = aec3_ms_to_hops(800.0f, hop, sample_rate);
}

static int aec_carve(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, int fcap, int is_static) {
    /* M4 (multi-rate consumption switch): per-rate lookup for every
     * constant below that AEC3B_RATE_TABLE carries. aec_validate_config's
     * {16000} whitelist guarantees a hit here (both aec_create and
     * aec_init run it before ever reaching aec_carve), so this never
     * returns NULL in practice -- guarded anyway (F05 house style: never
     * trust a lookup silently). At 16 kHz this row is pointer/value-
     * identical to the legacy unsuffixed macros/arrays it replaces below
     * (see aec3_balanced_config.h), so every switch in this function is a
     * no-op change in the bytes produced. */
    const Aec3BalancedRateDims* rd = aec3b_rate_cfg(
        cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;

    a->cfg = *cfg;
    a->hop_size = hop; a->block_size = blk; a->fft_size = fft;
    a->n_freqs = K; a->n_partitions = np;

    /* Retime the top-level (non-AEC3) wall-clock-authored constants against
     * the FINAL resolved (hop, cfg->sample_rate) grid (2026-08 gap-fix) --
     * see aec_legacy10ms_hops()/aec_legacy10ms_alpha()'s doc comment for why
     * this must happen here (aec_carve, construction time) rather than in
     * aec_config_defaults(). Mutates a->cfg (the carved instance's own
     * copy) only -- the caller's original *cfg is left untouched. Every
     * downstream reader of these six fields in this file already reads
     * a->cfg.* except the three call sites immediately below (epc_init/
     * shadow_copy_init/warmup_frames_remaining init), updated alongside
     * this to read a->cfg.* too so they see the retimed value. */
    a->cfg.shadow_err_alpha = aec_legacy10ms_alpha(cfg->shadow_err_alpha, hop, cfg->sample_rate);
    a->cfg.warmup_frames = aec_legacy10ms_hops((float)cfg->warmup_frames * 10.0f, hop, cfg->sample_rate);
    a->cfg.epc_hangover = aec_legacy10ms_hops((float)cfg->epc_hangover * 10.0f, hop, cfg->sample_rate);
    a->cfg.ne_recent_hold = aec_legacy10ms_hops((float)cfg->ne_recent_hold * 10.0f, hop, cfg->sample_rate);
    a->cfg.filter_misadjustment_stable_frames = aec_legacy10ms_hops(
        (float)cfg->filter_misadjustment_stable_frames * 10.0f, hop, cfg->sample_rate);
    a->cfg.filter_misadjustment_hangover_frames = aec_legacy10ms_hops(
        (float)cfg->filter_misadjustment_hangover_frames * 10.0f, hop, cfg->sample_rate);

    /* Backward-jump quarantine window: seconds -> estimation cycles ONCE,
     * here, against the resolved grid -- same rule as delay_est_init_s /
     * delay_est_period_s, which Path A converts with this identical
     * hop_s expression. Floor of 1: a window shorter than one cycle must
     * still quarantine for exactly one cycle, never collapse into "off"
     * (off is delay_backward_quarantine_enabled == 0 and nothing else).
     * The window lives in a->delay_quarantine_hops rather than being written
     * back into a->cfg, because a->cfg's copy stays in the SECONDS unit the
     * caller supplied -- unlike the retimes above, whose fields are already
     * hop-denominated. */
    {
        float q_hop_s = (float)hop / (float)cfg->sample_rate;
        int q_hops = (int)lrintf(cfg->delay_backward_quarantine_s / q_hop_s);
        a->delay_quarantine_hops = q_hops < 1 ? 1 : q_hops;
        a->delay_quarantine_left = -1;   /* disarmed */
    }

    /* inst-ERLE slope ring cap = Python _slope_n = max(2, int(0.5*sr/hop)),
     * clamped to the static array size. (orchestrator.py:649) */
    {
        int _sn = (int)(0.5f * (float)cfg->sample_rate / (float)(hop > 0 ? hop : 1));
        if (_sn < 2) _sn = 2;
        if (_sn > (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0])))
            _sn = (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0]));
        a->erle_slope_cap = _sn;
        a->erle_slope_len = 0;
        a->erle_slope_head = 0;
    }

    /* HPF: arena/pool-carved at the scratch tail below (audio_common f32 HPF). */
    /* Saturation */
    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold,
                        hop, cfg->sample_rate);
        saturation_init(&a->sat_mic, cfg->saturation_threshold,
                        hop, cfg->sample_rate);
        a->has_sat = 1;
    }
    /* Delay ring. `cfg` is already delay-resolved by every entry point
     * (aec_create / aec_init), so delay_mode is authoritative here.
     *   MATCHED           estimator + ring, delay unknown until acquisition
     *   FIXED             ring only, delay known from the caller's bring-up
     *                     measurement and applied from the first hop the
     *                     ring can serve it
     *   EXTERNAL_ALIGNED  neither; `ref` is already aligned by contract
     *
     * The estimator's own arrays are pool-carved here (plan step 2), in the
     * same position aec_get_mem_size() budgets them, with the SAME
     * (sample_rate, hop, delay_num_filters) triple -- that triple is the
     * whole lockstep contract, so it is read from `cfg`/`hop` in both places
     * and nowhere reconstructed. Only MATCHED carves it.
     *
     * The ring's capacity is likewise mode-dependent (plan step 3): MATCHED
     * gets the full search ring, FIXED only `fixed_delay_samples + hop` --
     * see aec_ref_ring_samples(), which produced the `buf_samp` passed in
     * here and the identical number aec_get_mem_size() budgeted. */
    if (aec_ring_active(cfg)) {
        /* Same rejection aec_get_mem_size() already made for this config;
         * repeated so the carve can never walk a ring it could not size.
         * Nothing has been constructed yet, so there is nothing to unwind. */
        if (buf_samp <= 0) return -1;
        if (cfg->delay_mode == AEC_DELAY_MATCHED) {
            size_t da_sz = delay_aec3_get_mem_size(cfg->sample_rate, hop,
                                                   cfg->delay_num_filters);
            /* Unreachable for a validated config (aec_get_mem_size already
             * queried the identical triple and refused to size the pool if
             * it came back 0), but this carve must never walk past a block
             * it could not size. Nothing has been constructed yet at this
             * point, so there is nothing to unwind. */
            if (da_sz == 0) return -1;
            if (delay_aec3_init(&a->delay, ptr, da_sz, cfg->sample_rate, hop,
                                cfg->delay_num_filters) != 0) return -1;
            ptr += da_sz;
            a->has_delay = 1;
        }
        a->ref_ring_size = buf_samp;
        a->ref_ring = (float*)ptr; ptr += ALIGN16((size_t)buf_samp * sizeof(float));
        memset(a->ref_ring, 0, (size_t)buf_samp * sizeof(float));
        a->current_delay = (cfg->delay_mode == AEC_DELAY_MATCHED)
                           ? -1 : cfg->fixed_delay_samples;
        a->pending_delay = -1; a->has_pending = 0; a->pending_delay_ttl = 0;
    } else {
        a->current_delay = 0;
    }
    a->duty_last_delay = -1;   /* duty-cycle change detect (rest zeroed by memset) */
    /* Streaming FIFO. Plain (non-atomic) init is correct here: this runs
     * before the instance is published to any render/capture thread (F09 —
     * see the SPSC atomics discipline in aec_analyze_render()/
     * aec_process_capture() and the field comments in aec.h). */
    a->fifo_cap_hops = fcap;
    a->render_fifo = (float*)ptr; ptr += ALIGN16((size_t)fcap * hop * sizeof(float));
    memset(a->render_fifo, 0, (size_t)fcap * hop * sizeof(float));
    /* fifo_zero_ref: immutable all-zero hop, the F09 Variant A' underrun
     * reference (see aec.h). Carved immediately after render_fifo — same
     * position in both aec_get_mem_size() and this carve, per the lockstep
     * walk-order requirement. */
    a->fifo_zero_ref = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    memset(a->fifo_zero_ref, 0, (size_t)hop * sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;

    /* RSA counters */
    a->rsa_counters = (int64_t*)ptr;
    size_t rsa_sz = ALIGN16((size_t)(K - 2 > 0 ? K - 2 : 1) * sizeof(int64_t));
    memset(ptr, 0, rsa_sz); ptr += rsa_sz;

    /* Shared FFT handle. ONE FftHandle for the whole instance,
     * carved and constructed first so the main filter, shadow filter, and
     * aec3_post all have something to borrow at their own init time below --
     * they never carve/own a private handle anymore (see pbfdaf_init_static's
     * doc comment). Safe because all three run at the identical fft_size
     * (this same `fft` local, per-instance -- see aec_derive_dims above) and
     * fully sequentially: shadow runs to completion, then main, then
     * aec3_post, never interleaved, on this codebase's single-threaded
     * synchronous call path. This is also now the ONLY nested-FFT allocation
     * in this whole carve, so it is also the only failure this function can
     * unwind for that reason -- nothing has been constructed yet that owns
     * an external (NE10 twiddle-config) allocation, so a failure here needs
     * no unwind at all. */
    size_t fft_sz = fft_get_mem_size(fft);
    a->post_fft = fft_init(ptr, fft_sz, fft);
    if (!a->post_fft) return -1;   /* F04: fft_init failed (OOM) -- nothing to unwind yet */
    ptr += fft_sz;

    /* Main filter (PBFDKF, arena/pool-carved on BOTH heap and static paths
     * since the F03 arena-fication below aec_create()). Borrows a->post_fft
     * (the shared FFT handle above) rather than owning its own handle. */
    size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
    if (pbfdkf_init_static(&a->main_filter, ptr, kf_sz, blk, np, cfg->mu, cfg->delta, hop, cfg->sample_rate, a->post_fft) != 0) {
        fft_destroy(a->post_fft); a->post_fft = NULL;
        return -1;
    }
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }
    a->main_filter.base.poor_excitation_counter = aec3_blocks_to_hops(1000, hop, cfg->sample_rate);
    a->main_filter.base.constraint_round_robin = 1;  /* AEC3 round-robin (main) */
    ptr += kf_sz;

    /* Shadow filter (PBFDAF, arena/pool-carved on BOTH paths since F03).
     * Borrows a->post_fft too -- same shared handle the main filter just
     * borrowed above. */
    if (cfg->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (pbfdaf_init_static(&a->shadow_filter, ptr, af_sz, blk, np, cfg->shadow_mu_nlms, cfg->delta, hop, 1, cfg->sample_rate, a->post_fft) != 0) {
            /* Neither main filter nor the shared handle owns anything
             * needing unwind-order care beyond the usual pbfdkf_free/
             * fft_destroy pair -- main filter's pbfdaf_free() is now a pure
             * pointer-drop (it never owned the fft it borrowed). */
            pbfdkf_free(&a->main_filter);
            fft_destroy(a->post_fft); a->post_fft = NULL;
            return -1;
        }
        a->shadow_filter.poor_excitation_counter = a->main_filter.base.poor_excitation_counter;
        a->shadow_filter.saturated_capture = 0;
        /* FFT dedup: skip the shadow's unread near_spec / error_spec_windowed. */
        a->shadow_filter.lightweight = 1;
        a->shadow_filter.constraint_round_robin = 1;  /* AEC3 round-robin (shadow) */
        a->has_shadow = 1;
        ptr += af_sz;
    }

    /* Detectors / EPC / regime / RSA */
    a->ra_pairwise_scratch = (float*)ptr;
    ptr += ALIGN16((size_t)((hop + 7) / 8) * sizeof(float));
    render_activity_init(&a->render_activity, a->ra_pairwise_scratch, (hop + 7) / 8,
                         hop, cfg->sample_rate);
    filter_convergence_init(&a->convergence, hop, cfg->sample_rate);
    doubletalk_init(&a->dt_analyzer, 1.5, 3.0, hop, cfg->sample_rate);
    epc_init(&a->epc, a->cfg.epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold,
             hop, cfg->sample_rate);
    shadow_copy_init(&a->regime, SC_GATE_ENERGY, 0.65, 3, a->cfg.epc_hangover);
    rsa_init(&a->rsa, a->rsa_counters, K, np);

    /* aec3_post backing (reuses a->post_fft, the shared handle constructed
     * above -- no separate carve/init here anymore). */
    {
        Aec3PostConfig pcfg;
        aec3_post_config_defaults(&pcfg);
        pcfg.n_bins = K; pcfg.fft_size = fft; pcfg.block_size = blk; pcfg.hop_size = hop;
        /* M4: rate-table synth window length (== block_size at every
         * validated rate); aec3_post_apply_output asserts this before
         * reading synth_window. */
        pcfg.synth_window_len            = rd->synth_window_len;
        pcfg.erle_coh_gate_enabled       = AEC3B_ERLE_COH_GATE_ENABLED;
        pcfg.erle_windowed_capture_psd   = AEC3B_ERLE_WINDOWED_CAPTURE_PSD;
        pcfg.erle_render_x2_psd_scale    = AEC3B_ERLE_RENDER_X2_PSD_SCALE;
        pcfg.output_capture_when_linear_unusable = AEC3B_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE;
        pcfg.enable_cng                  = cfg->enable_cng;
        pcfg.cng_n2_update_onset_hops    = rd->cng_n2_update_onset_hops;
        pcfg.cng_n2_initial_duration_hops = rd->cng_n2_initial_duration_hops;
        pcfg.cng_y2_alpha                = rd->cng_y2_alpha;
        pcfg.cng_n2_track_freshness      = rd->cng_n2_track_freshness;
        pcfg.cng_n2_track_retention      = rd->cng_n2_track_retention;
        pcfg.cng_n2_slow_up              = rd->cng_n2_slow_up;
        pcfg.cng_n2_initial_alpha        = rd->cng_n2_initial_alpha;
        pcfg.noise_floor_int16sq         = rd->noise_floor_int16sq;
        pcfg.erle_coh_gate_alpha         = AEC3B_ERLE_COH_GATE_ALPHA;
        pcfg.erle_coh_gate_threshold     = AEC3B_ERLE_COH_GATE_THRESHOLD;

#define P_FSLICE(n) ((float*)(ptr));    ptr += ALIGN16(K * sizeof(float));    (void)(n)
#define P_CSLICE(n) ((Complex*)(ptr));  ptr += ALIGN16(K * sizeof(Complex));  (void)(n)
#define P_BSLICE(n) ((unsigned char*)(ptr)); ptr += ALIGN16(K * sizeof(unsigned char)); (void)(n)
        float        *avg_rev  = P_FSLICE(0); float *y2s    = P_FSLICE(0);
        float        *n2       = P_FSLICE(0); float *n2i    = P_FSLICE(0);
        float        *sye_re   = P_FSLICE(0); float *sye_im = P_FSLICE(0);
        float        *syy      = P_FSLICE(0); float *see    = P_FSLICE(0);
        float        *ola      = (float*)ptr;  ptr += ALIGN16((size_t)blk * sizeof(float));
        float        *np_      = P_FSLICE(0); float *fp_    = P_FSLICE(0);
        float        *ep_      = P_FSLICE(0); float *erp_   = P_FSLICE(0);
        float        *cpe      = P_FSLICE(0); float *x2r    = P_FSLICE(0);
        float        *cn       = P_FSLICE(0); float *nf_    = P_FSLICE(0);
        unsigned char *cgm     = P_BSLICE(0);
        Complex      *eout     = P_CSLICE(0);
        float        *eout_full = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        memset(syy, 0, K * sizeof(float)); memset(see, 0, K * sizeof(float));
        aec3_post_init(&a->post, &pcfg, a->post_fft,
                       rd->synth_window, AEC3B_SQRT2_SIN_LUT, avg_rev,
                       y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                       np_, fp_, ep_, erp_, cpe, x2r, cgm, cn, nf_,
                       eout, eout_full);
    }

    /* AecStateStorage */
    {
        AecStateStorage* s = &a->a3_state_st;
        s->erle_max           = P_FSLICE(0); s->erle            = P_FSLICE(0);
        s->erle_oc            = P_FSLICE(0); s->erle_unb        = P_FSLICE(0);
        s->erle_during        = P_FSLICE(0);
        s->erle_coming_onset  = P_BSLICE(0);
        s->erle_hold          = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        s->erle_y2_acc        = P_FSLICE(0); s->erle_e2_acc     = P_FSLICE(0);
        s->erle_low_render    = P_BSLICE(0);
        s->erl                = P_FSLICE(0);
        s->erl_hold           = (int*)ptr; ptr += ALIGN16((size_t)(K - 2) * sizeof(int));
        s->filter_delays_blocks = (int*)ptr; ptr += ALIGN16((size_t)AEC3B_ST_NUM_CAPTURE_CHANNELS * sizeof(int));
        /* M4: runtime np*hop (== rd->filter_taps_size at 16 kHz), not the
         * AEC3B_FILTER_TAPS_SIZE macro -- must stay in lockstep with the
         * matching aec_get_mem_size term above. */
        s->fa_h_highpass      = (float*)ptr; ptr += ALIGN16(ck_mul_size((size_t)np, (size_t)hop) * sizeof(float));
        /* FilterAnalyzer de-stacked fa_update() scratch (M3): sized from the
         * same runtime dims (np*hop / hop) -- see the matching
         * aec_get_mem_size comment above. */
        s->fa_abs_scratch       = (float*)ptr; ptr += ALIGN16(ck_mul_size((size_t)np, (size_t)hop) * sizeof(float));
        s->fa_render_sq_scratch = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        {
            AecStateConfig acfg; aec_state_config_defaults(&acfg);
            acfg.n_bins = K; acfg.num_capture_channels = AEC3B_ST_NUM_CAPTURE_CHANNELS;
            acfg.hop_size = hop;
            /* Was never populated at this call site either (see the sibling
             * fix's comment above) -- stayed at the config-defaults 16000
             * regardless of the real cfg->sample_rate. */
            acfg.sample_rate = cfg->sample_rate;
            acfg.enable_filter_analyzer = AEC3B_ST_ENABLE_FILTER_ANALYZER;
            /* Sentinel, not a literal hop count -- see the sibling call
             * site's comment above (AEC_STATE_STARTUP_HOPS_AUTO). */
            acfg.erle_startup_hops = AEC3B_ST_ERLE_STARTUP_HOPS;
            acfg.erl_startup_hops  = AEC3B_ST_ERL_STARTUP_HOPS;
            acfg.echo_can_saturate = AEC3B_ST_ECHO_CAN_SATURATE;
            acfg.use_linear_filter = AEC3B_ST_USE_LINEAR_FILTER;
            acfg.conservative_initial_phase = AEC3B_ST_CONSERVATIVE_INITIAL_PHASE;
            acfg.delay_headroom_samples = AEC3B_ST_DELAY_HEADROOM_SAMPLES;
            acfg.initial_state_seconds  = AEC3B_ST_INITIAL_STATE_SECONDS;
            acfg.erle_min = AEC3B_ST_ERLE_MIN; acfg.erle_max_l = AEC3B_ST_ERLE_MAX_L;
            acfg.erle_max_h = AEC3B_ST_ERLE_MAX_H;
            /* M4: runtime np*hop, not AEC3B_FILTER_TAPS_SIZE (== rd->
             * filter_taps_size at 16 kHz -- see the aec_get_mem_size /
             * fa_h_highpass carve comments above for the lockstep pair). */
            acfg.filter_taps_size = np * hop;
            acfg.fa_scratch_size = np * hop;
            aec_state_init(&a->a3_state, &acfg, s);
            aec_state_set_taps_provider(&a->a3_state, aec_fill_filter_taps, a);
        }
    }

    /* ResidualEchoEstimator */
    {
        int rh = (AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW ? 1 : 0) + 2;
        float *x2_nf  = P_FSLICE(0);
        int   *x2_nfc = (int*)ptr;   ptr += ALIGN16(K * sizeof(int));
        float *rm_st  = P_FSLICE(0); float *rt_st = P_FSLICE(0);
        float *rh_st  = (float*)ptr; ptr += ALIGN16((size_t)rh * K * sizeof(float));
        float *drd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *rrd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *ld_st  = P_FSLICE(0); float *lr_st = P_FSLICE(0); float *scr_st = P_FSLICE(0);
        ReeEchoModelConfig em; memset(&em, 0, sizeof(em));
        /* M4: rate-varying REE absolute-power floats now come from rd (== the
         * legacy macros' value at 16 kHz, byte-identical); noise_gate_power/
         * slope/model_reverb_in_nonlinear_mode are rate-invariant (see the
         * residual macro-use list in the M4 report) and stay direct. */
        em.min_noise_floor_power = rd->ree_min_noise_floor_power;
        em.noise_gate_power      = AEC3B_REE_NOISE_GATE_POWER_LEGACY;
        em.noise_gate_slope      = AEC3B_REE_NOISE_GATE_SLOPE;
        em.stationary_gate_slope = AEC3B_REE_STATIONARY_GATE_SLOPE;
        em.model_reverb_in_nonlinear_mode = AEC3B_REE_MODEL_REVERB_IN_NL;
        ree_init(&a->a3_ree, K, hop, cfg->sample_rate, &em,
                 AEC3B_REE_DEFAULT_GAIN, AEC3B_REE_TM_GAIN, AEC3B_REE_ERLE_ONSET_COMP,
                 AEC3B_REE_REVERB_DECAY, AEC3B_REE_REVERB_MILD_SCALE,
                 AEC3B_REE_REVERB_ENABLED, AEC3B_REE_REVERB_TAIL_STRENGTH,
                 AEC3B_REE_USE_AEC3_RESIDUAL_NOISE_GATE,
                 AEC3B_USE_STATIONARITY_PROPERTIES,
                 AEC3B_REE_USE_AEC3_ECHO_GEN_WINDOW,
                 AEC3B_REE_NL_R2_ENABLED, AEC3B_REE_NL_R2_ALPHA, rd->ree_nl_norm_power,
                 rd->ree_residual_noise_gate_power, rd->ree_noise_floor_hold_hops,
                 AEC3B_REE_USE_FREQ_RESPONSE, AEC3B_REE_REVERB_USE_CONSERVATIVE,
                 rd->ree_reverb_smoothing_base,
                 x2_nf, x2_nfc, rm_st, rt_st, rh_st, drd_st, rrd_st,
                 ld_st, lr_st, scr_st);
    }

    /* SuppressionGain */
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg; SuppressionGainTuning stun;
        memset(&scfg, 0, sizeof(scfg));
        scfg.n_bins = K; scfg.sr = cfg->sample_rate; scfg.hop_size = hop;
        /* M4: the 9 Hz-anchored bin scalars now come from rd (byte-
         * identical at 16 kHz -- see the M4 report's replaced-macro
         * inventory). last_permanent_lf_smoothing_band / lf_smoothing_
         * during_initial_phase are rate-invariant fixed constants (0 / 1)
         * and stay direct macro uses. */
        scfg.last_lf_band = rd->sg_last_lf_band;
        scfg.first_hf_band = rd->sg_first_hf_band;
        scfg.last_lf_smoothing_band = rd->sg_last_lf_smoothing_band;
        scfg.last_permanent_lf_smoothing_band = AEC3B_SG_LAST_PERMANENT;
        scfg.lf_smoothing_during_initial_phase = AEC3B_SG_LF_SMOOTHING_INITIAL;
        scfg.lf_clamp_bin = rd->sg_lf_clamp_bin;
        scfg.dne_lf_end = rd->sg_dne_lf_end;
        scfg.nearend_smoother_n = rd->sg_nearend_smoother_n;
        scfg.aud_lf_end_bin = rd->sg_aud_lf_end_bin; scfg.aud_mf_end_bin = rd->sg_aud_mf_end_bin;
        /* M4: rate-varying SG absolute-power floor. */
        scfg.floor_power = rd->sg_floor_power;
        scfg.aud_thr_lf = AEC3B_SG_AUD_THR_LF; scfg.aud_thr_mf = AEC3B_SG_AUD_THR_MF;
        scfg.aud_thr_hf = AEC3B_SG_AUD_THR_HF;
        /* M4: rate-varying SG render-limit floats. */
        scfg.low_render_limit = rd->sg_low_render_limit;
        scfg.normal_render_limit = rd->sg_normal_render_limit;
        scfg.hf_lgb = rd->sg_hf_lgb; scfg.hf_biq = AEC3B_SG_HF_BIQ;
        scfg.conservative_hf = AEC3B_SG_CONSERVATIVE_HF;
        scfg.max_inc_normal = rd->sg_max_inc;
        scfg.max_inc_nearend = rd->sg_max_inc;
        scfg.max_dec_lf_normal = rd->sg_max_dec_lf;
        scfg.max_dec_lf_nearend = rd->sg_max_dec_lf;
        scfg.floor_first_increase = 0.00001f;
        /* M4: rate-varying SG absolute-power threshold. */
        scfg.low_render_threshold = rd->sg_low_render_threshold;
        scfg.split_floor_enabled = 1;
        scfg.split_floor_far_active =
            powf(10.0f, cfg->min_gain_floor_far_active_db / 10.0f);
        scfg.split_floor_far_silent = AEC3B_SG_SPLIT_FLOOR_FAR_SILENT;
        /* DT-gated floor: 10^(min_gain_floor_dt_db/10) (mirrors aec_create). */
        scfg.split_floor_dt =
            powf(10.0f, cfg->min_gain_floor_dt_db / 10.0f);
        scfg.split_floor_latch_power = AEC3B_SG_SPLIT_FLOOR_LATCH_POWER;
        scfg.soft_blend_enabled = AEC3B_SG_SOFT_BLEND_ENABLED;
        scfg.soft_blend_per_bin = AEC3B_SG_SOFT_BLEND_PER_BIN;
        scfg.soft_blend_enr_thr = (float)AEC3B_SG_SOFT_BLEND_ENR_THR;
        scfg.soft_blend_softness = (float)AEC3B_SG_SOFT_BLEND_SOFTNESS;
        scfg.dne_enr_threshold = AEC3B_SG_DNE_ENR_THRESHOLD;
        scfg.dne_enr_exit_threshold = AEC3B_SG_DNE_ENR_EXIT_THRESHOLD;
        scfg.dne_snr_threshold = AEC3B_SG_DNE_SNR_THRESHOLD;
        scfg.dne_use_during_initial_phase = AEC3B_SG_DNE_USE_DURING_INITIAL_PHASE;
        scfg.dne_use_unbounded_echo = AEC3B_SG_DNE_USE_UNBOUNDED_ECHO;
        scfg.dne_lf_endpoint_bin = rd->sg_dne_lf_endpoint_bin;
        scfg.dne_trigger_threshold_hops = rd->sg_trigger_threshold_hops;
        scfg.dne_hold_duration_hops = rd->sg_hold_duration_hops;
        scfg.stat_aware_ne_proxy_enabled = 0; scfg.stat_aware_ne_proxy_threshold = 0.10f;
        /* M4: the six per-bin tuning table pointers now come from rd
         * (pointer-identical to the legacy AEC3B_SG_* arrays at 16 kHz) +
         * their length, asserted == n_bins in suppression_gain_init. */
        stun.nearend_enr_tr = rd->sg_nearend_enr_tr; stun.nearend_enr_su = rd->sg_nearend_enr_su;
        stun.nearend_emr_tr = rd->sg_nearend_emr_tr;
        stun.normal_enr_tr  = rd->sg_normal_enr_tr;  stun.normal_enr_su  = rd->sg_normal_enr_su;
        stun.normal_emr_tr  = rd->sg_normal_emr_tr;
        stun.table_len      = rd->sg_table_len;
        float *last_gain = P_FSLICE(0); float *last_ne = P_FSLICE(0); float *last_echo = P_FSLICE(0);
        float *ma   = (float*)ptr; ptr += ALIGN16((size_t)rd->sg_nearend_smoother_n * K * sizeof(float));
        float *ne_s = P_FSLICE(0); float *wr_s = P_FSLICE(0);
        float *ming = P_FSLICE(0); float *maxg = P_FSLICE(0);
        float *graw = P_FSLICE(0); float *gout = P_FSLICE(0); float *gsum = P_FSLICE(0);
        suppression_gain_init(sg, &scfg, &stun, last_gain, last_ne, last_echo,
                              ma, ne_s, wr_s, ming, maxg, graw, gout, gsum);
    }

    /* StationarityEstimator */
    {
        float        *st_noise = P_FSLICE(0);
        int32_t      *st_hang  = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        unsigned char *st_flg  = P_BSLICE(0);
        float        *st_hist  = (float*)ptr;   ptr += ALIGN16(16 * K * sizeof(float));
        stationarity_estimator_init(&a->a3_stat, K, hop, cfg->sample_rate,
                                    st_noise, st_hang, st_flg, st_hist);
    }

    /* LinearFilterSelect (manual slice — no static API) */
    {
        memset(&a->a3_lfs, 0, sizeof(a->a3_lfs));
        a->a3_lfs.hop        = hop;
        a->a3_lfs.block_size = blk;
        a->a3_lfs.fft_size   = fft;
        a->a3_lfs.n_freqs    = K;
        a->a3_lfs.prev_output_time = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.e_form           = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.block_win        = (float*)ptr; ptr += ALIGN16((size_t)blk * sizeof(float));
        a->a3_lfs.sel_esw          = (Complex*)ptr; ptr += ALIGN16(K * sizeof(Complex));
        /* De-stacked scratch (matching aec_get_mem_size's "LinearFilterSelect
         * de-stacked scratch (4)" block above — keep in lockstep). */
        a->a3_lfs.scr_sq   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_sref = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_scoa = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_tin  = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        linear_filter_select_reset(&a->a3_lfs);
    }

    /* Aec3PostRunScratch */
    {
        Aec3PostRunScratch* sc = &a->a3_sc;
        sc->sel_esw  = P_CSLICE(0); sc->sel_echo = P_CSLICE(0);
        sc->nsw_e1   = P_CSLICE(0); sc->ybase    = P_CSLICE(0);
        sc->abs_near     = P_FSLICE(0); sc->abs_far      = P_FSLICE(0);
        sc->abs_sel_echo = P_FSLICE(0); sc->abs_error    = P_FSLICE(0);
        sc->abs_echo_coh = P_FSLICE(0); sc->abs_nsw_e1   = P_FSLICE(0);
        sc->abs_ybase    = P_FSLICE(0); sc->x2_at_delay  = P_FSLICE(0);
        sc->x2_past      = P_FSLICE(0);
        sc->w_mag2 = (float*)ptr; ptr += ALIGN16((size_t)np * K * sizeof(float));
        sc->render_block_scaled = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        sc->r2          = P_FSLICE(0); sc->r2_unb   = P_FSLICE(0);
        sc->nearend_pwr = P_FSLICE(0);
        sc->stat_mask   = P_BSLICE(0);
    }
#undef P_FSLICE
#undef P_CSLICE
#undef P_BSLICE

    /* hop scratch */
    a->per_bin_mu_scale = (float*)ptr; ptr += ALIGN16(K * sizeof(float));
    a->near_hop   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->far_hop    = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->raw_output = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->shadow_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->final_out  = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->filter_taps_full = (float*)ptr; ptr += ALIGN16((size_t)np * hop * sizeof(float));

    /* per-hop freq-bin scratch (see aec.h struct comment). */
    a->scr_sq        = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->scr_e2_echo   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2_near   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_rsa_psd   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_rsa_mask  = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf_pre= (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2coa_pre = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf    = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_far_psd   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2ref_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2coa_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_erl_arr   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    /* mic-path HPF (audio_common f32; arena/pool-resident, hpf_destroy no-ops).
     * R08: hpf_init() returns NULL if hpf_params_valid() (audio_common
     * hpf.c) rejects (cutoff_hz, sample_rate) -- aec_validate_config now
     * mirrors that exact check whenever enable_highpass is set, so this
     * should never fire in practice. Checked anyway (belt-and-braces, same
     * F04 unwind-and-fail shape as the nested FFT allocations above) rather
     * than silently leaving a->hp_mic NULL and running with no mic-path HPF
     * despite the caller asking for one. */
    if (cfg->enable_highpass) {
        a->hp_mic = hpf_init(ptr, hpf_get_mem_size(),
                             cfg->highpass_cutoff_hz, cfg->sample_rate);
        if (!a->hp_mic) {
            /* Unwind every nested allocation brought up earlier in this
             * carve (post_fft is always non-NULL here -- the fft_init check
             * above already returned -1 if it failed). */
            fft_destroy(a->post_fft); a->post_fft = NULL;
            if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
            pbfdkf_free(&a->main_filter);
            return -1;
        }
        ptr += ALIGN16(hpf_get_mem_size());
    }

    /* scalar state */
    a->pending_gain_change = 0; a->pending_delay_change = -1;
    a->stationarity_active_hops = 0; a->non_zero_render_seen = 0;
    a->render_peak_floor = 10.0f / 32768.0f;
    a->block_stationary_next = 0;
    a->stationarity_converge_hops = rd->stationarity_converge_hops;
    a->saturation_level = 0.0f; a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0; a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->wn_err_baseline = 1e-8f; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->cfg.warmup_frames; a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f; a->alpha_pow = powf(0.95f, 16000.0f / (float)a->cfg.sample_rate);
    a->alpha_erl_tracking  = aec3_growth_rehop(0.99f,  160, 16000,
                                               a->hop_size, a->cfg.sample_rate);
    a->alpha_erl_converged = aec3_growth_rehop(0.999f, 160, 16000,
                                               a->hop_size, a->cfg.sample_rate);
    /* ^ per-SAMPLE ERLE power EMA (applied once per sample, not per hop),
     * so it is retimed off the SAMPLE RATE and is hop-invariant.
     * aec3_growth_rehop() would be wrong here. Mirrors orchestrator.py. */
    a->frame_count = 0; a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    aec_recompute_wallclock_thresholds(a, hop, cfg->sample_rate);
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
    filter_q_high(&a->main_filter);
    a->is_static = is_static;
    return 0;
}

Aec* aec_init(void* mem, size_t mem_size, const AecConfig* cfg_in) {
    if (!mem || !cfg_in) return NULL;
    if (!aec_validate_config(cfg_in)) return NULL;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;
    /* F07: reject a misaligned pool base before any pool write. Every offset
     * aec_carve carves below is an ALIGN16 bump off `mem`, so a misaligned
     * base would misalign every sub-module's SIMD-sensitive buffers too. */
    if (!MEM_IS_ALIGNED16(mem)) return NULL;
    {
        /* aec_get_mem_size(cfg) returning 0 means "invalid" (F05) — a
         * legitimate config's total is always > 0 (sizeof(Aec) alone
         * guarantees that), so 0 is an unambiguous failure sentinel here,
         * unlike a bare `mem_size < aec_get_mem_size(cfg)` which would never
         * reject anything once aec_get_mem_size started returning 0 for
         * invalid input (mem_size < 0 is never true for a size_t). */
        size_t need = aec_get_mem_size(cfg);
        if (need == 0 || mem_size < need) return NULL;
    }

    int hop, blk, fft, K, np, buf_samp, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf_samp, &fcap);

    Aec* a = (Aec*)mem;
    memset(a, 0, sizeof(Aec));
    uint8_t* ptr = (uint8_t*)mem + ALIGN16(sizeof(Aec));
    if (aec_carve(a, ptr, cfg, hop, blk, fft, K, np, buf_samp, fcap,
                  /*is_static=*/1) != 0) {
        /* F04: a nested FFT allocation failed (OOM), or (R08 belt-and-
         * braces, practically unreachable post-validator-fix) hpf_init()
         * rejected its params. aec_carve() already destroyed any nested FFT
         * handles it had brought up before returning; the pool itself
         * belongs to the caller, so there is nothing else for us to
         * release. */
        return NULL;
    }
    a->heap_arena = NULL;
    return a;
}

void aec_destroy(Aec* a) {
    if (!a) return;
    /* Release library-internal FFT allocations on BOTH the heap-arena and
     * static-pool paths. a->post_fft is the SOLE FftHandle for
     * the whole instance -- main_filter / shadow_filter only ever borrow it
     * (see pbfdaf_init_static's doc comment), so their pbfdkf_free/
     * pbfdaf_free calls below are now pure pointer-drops, not destroys.
     * Destroying a->post_fft here is therefore the ONLY place that tears
     * down the nested FFT's NE10 twiddle config (an allocation that lives
     * outside the pool/arena either way); calling it a second time (e.g. via
     * a stray destroy inside main_filter/shadow_filter) would double-free
     * that allocation, which is exactly what borrowing-not-owning avoids.
     * fft_destroy() is a no-op for a NULL handle and (KISS backend) for a
     * pool-owned handle, so this is safe pre- or post- pool/arena teardown.
     * post_fft is NULLed right after so a second aec_destroy() call on the
     * same instance (see test_lifecycle.c) cannot dereference a pointer that
     * now lives in freed memory instead of silently corrupting/crashing. */
    if (a->post_fft) { fft_destroy(a->post_fft); a->post_fft = NULL; }
    if (a->hp_mic) { hpf_destroy(a->hp_mic); a->hp_mic = NULL; }
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdaf_free(&a->shadow_filter);

    if (a->is_static) return;   /* caller owns the pool: nothing else to free */

    /* F03: everything else (ref_ring, render_fifo, rsa_counters, the whole
     * AEC3 post chain's backing arrays, hop / per-hop-freq-bin scratch, ...)
     * is carved out of the single arena below rather than individually
     * malloc'd -- freeing any of those pointers on their own would free an
     * interior address instead of the malloc() base, which is undefined
     * behaviour. One free() reclaims all of it, and is itself idempotent
     * (a second call sees heap_arena already NULLed). */
    free(a->heap_arena);
    a->heap_arena = NULL;
}

int aec_hop_size(const Aec* a) { return a->hop_size; }

/* Far-end-FFT-sharing instrumentation: how many times THIS instance has
 * actually run its own far-end rfft (not borrowed one, whether from its own
 * internal shadow->main dedup or an external aec_process_context_shared_far()
 * caller), summed across whichever of shadow/main filter is the one that
 * can run it fresh this hop. At most one of the two ever increments per
 * hop (shadow's own internal dedup already guaranteed that before this
 * sharing feature existed) -- see pbfdaf_frontend()'s
 * far_fft_real_compute_count comment. */
long aec_far_fft_real_compute_count(const Aec* a) {
    return a->main_filter.base.far_fft_real_compute_count +
           (a->has_shadow ? a->shadow_filter.far_fft_real_compute_count : 0);
}

static float aec_erle_ring_max_last15(const Aec* a);

int aec_apply_external_realign(Aec* a, int delta_samples) {
    if (a == NULL || a->cfg.delay_mode != AEC_DELAY_EXTERNAL_ALIGNED)
        return -1;
    if (delta_samples == 0) return 0;
    /* Same evidence gate as the internal MATCHED warm tap-transfer: the
     * inst-ERLE ring is filled unconditionally every hop, so it is live in
     * EXTERNAL_ALIGNED mode too. The shift must fit the tap span in BOTH
     * directions or the learned response would be pushed off the filter. */
    int warm_ok = 0;
    if (a->cfg.delay_acquire_warm_transfer) {
        float wpk = aec_erle_ring_max_last15(a);
        int reach = a->main_filter.base.n_partitions
                    * a->main_filter.base.hop_size;
        int magnitude = delta_samples > 0 ? delta_samples : -delta_samples;
        warm_ok = (wpk > a->cfg.delay_acquire_inst_erle_db)
                  && (magnitude < reach);
    }
    if (warm_ok) {
        pbfdaf_warm_shift_ir(&a->main_filter.base, delta_samples);
        if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, delta_samples);
        if (delta_samples < 0) {
            /* The alignment retarded, so the far history the filter holds is
             * AHEAD of the new stream: the next hops replay samples X_buf
             * already contains, and convolving that duplicated context
             * against the shifted taps re-exposes echo for dozens of hops
             * (measured on the regression scene). A delta > 0 realign only
             * leaves a gap in the history, which settles quietly, so the
             * history is cleared for the retard direction alone. */
            int Wsz = a->main_filter.base.n_partitions
                      * a->main_filter.base.n_freqs;
            memset(a->main_filter.base.X_buf, 0,
                   (size_t)Wsz * sizeof(Complex));
            if (a->has_shadow) {
                int Ssz = a->shadow_filter.n_partitions
                          * a->shadow_filter.n_freqs;
                memset(a->shadow_filter.X_buf, 0,
                       (size_t)Ssz * sizeof(Complex));
            }
        }
        return 1;
    }
    /* Soft echo-path change, exactly the internal soft-acquisition branch:
     * excitation/convergence counters restart, the taps stay and re-adapt at
     * normal step size. No tap wipe, no WOLA restart, no AecState reset. */
    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
    if (a->has_shadow) {
        a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
        a->shadow_filter.call_counter = 0;
    }
    return 0;
}

void aec_reset(Aec* a) {
    if (a->hp_mic) hpf_reset(a->hp_mic);
    if (a->has_sat) { saturation_reset(&a->sat_ref); saturation_reset(&a->sat_mic); }
    /* Ring exists for MATCHED and FIXED; only the ESTIMATOR is
     * MATCHED-exclusive. Mirrors the Python orchestrator's reset(), which
     * clears the ring under `_delay_active` and re-seeds _current_delay from
     * fixed_delay_samples when there is no estimator to reset. */
    if (aec_ring_active(&a->cfg)) {
        if (a->has_delay) delay_aec3_reset(&a->delay);
        memset(a->ref_ring, 0, (size_t)a->ref_ring_size * sizeof(float));
        a->ref_ring_write = 0; a->ref_ring_filled = 0;
        a->current_delay = a->has_delay ? -1 : a->cfg.fixed_delay_samples;
        a->pending_delay = -1; a->has_pending = 0;
        a->pending_delay_ttl = 0;
        /* A reset abandons the alignment the quarantine was protecting, so a
         * countdown armed against it must not survive into the next lock. */
        a->delay_quarantine_left = -1;
    }
    /* Reset invalidates any alignment context an external consumer cached. */
    if (a->delay_generation != 0xFFFFFFFFu) a->delay_generation++;
    a->far_hop_aligned = 0;
    a->duty_active = 0; a->duty_stable_hops = 0; a->duty_pos = 0;
    a->duty_last_delay = -1; a->duty_erle_peak = 0.0f;
    /* The census measures ONE run of the duty machine, so it restarts with
     * it -- carrying pre-reset hops across would blend two different duty
     * regimes into a single, meaningless ratio. Matches the "cumulative
     * since aec_create/aec_init/aec_reset" contract on AecDebugStatus. */
    a->duty_hops_total = 0; a->duty_hops_run = 0;
    if (a->render_fifo)
        memset(a->render_fifo, 0,
               (size_t)a->fifo_cap_hops * a->hop_size * sizeof(float));
    /* fifo_zero_ref hygiene: it is never written by either streaming thread
     * in steady state, so this re-zero is purely defensive (guards a caller
     * that poked at it, or a future change) — cheap at reset frequency. */
    if (a->fifo_zero_ref)
        memset(a->fifo_zero_ref, 0, (size_t)a->hop_size * sizeof(float));
    /* Plain (non-atomic) reset — contract requires the caller to have
     * quiesced both the render and capture threads before calling
     * aec_reset() on a streaming-mode instance (F09, see aec.h). */
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);
    render_activity_reset(&a->render_activity);
    filter_convergence_reset(&a->convergence);
    doubletalk_reset(&a->dt_analyzer);
    epc_reset(&a->epc);
    shadow_copy_reset(&a->regime);
    rsa_reset(&a->rsa);
    stationarity_estimator_reset(&a->a3_stat);
    a->stationarity_active_hops = 0;
    a->non_zero_render_seen = 0;
    a->block_stationary_next = 0;
    aec3_post_chain_reset(a);
    a->pending_gain_change = 0;
    a->pending_delay_change = -1;
    a->saturation_level = 0.0f;
    a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->wn_err_baseline = 1e-8f; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->cfg.warmup_frames;
    a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f;
    a->frame_count = 0;
    a->ne_above = 0; a->ne_recent_frames = 0;
    a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    aec_recompute_wallclock_thresholds(a, a->hop_size, a->cfg.sample_rate);
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
}

/* ───────────────────────── misadjustment estimator ─────────────────────── */

/* _update_misadjustment_estimator (orchestrator 100-148). float32-by-design
 * (Stage-2 conversion; block-energy sums accumulate in int16²-domain
 * magnitudes, same domain as the power EMAs above). */
static void misadj_update(Aec* a, const float* near_hpf, const float* raw_out, int hop) {
    float e2_block = 0.0f, y2_block = 0.0f;
    for (int i = 0; i < hop; ++i) {
        float r = raw_out[i]; e2_block += r * r;
        float n = near_hpf[i]; y2_block += n * n;
    }
    a->misadj_e2_acum += e2_block;
    a->misadj_y2_acum += y2_block;
    a->misadj_n_acum += 1;
    int n_hops_target = 2;
    if (a->misadj_n_acum < n_hops_target) return;
    float total_samples = (float)(n_hops_target * hop);
    float int16_sq = 32768.0f * 32768.0f;
    float y2_threshold = (200.0f * 200.0f) * total_samples / int16_sq;
    float e2_overhang_threshold = (7500.0f * 7500.0f) * total_samples / int16_sq;
    if (a->misadj_y2_acum > y2_threshold) {
        float denom = a->misadj_y2_acum > 1e-20f ? a->misadj_y2_acum : 1e-20f;
        float update = a->misadj_e2_acum / denom;
        if (a->misadj_e2_acum > e2_overhang_threshold) a->misadj_overhang = 4;
        else { a->misadj_overhang -= 1; if (a->misadj_overhang < 0) a->misadj_overhang = 0; }
        if ((update < a->misadj_inv) || (a->misadj_overhang > 0))
            a->misadj_inv += 0.1f * (update - a->misadj_inv);
    }
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
}

/* _fire_aec3_misadj_scale (orchestrator 150-191). float32-by-design. */
static void misadj_fire(Aec* a) {
    if (a->misadj_hangover_remaining > 0) { a->misadj_hangover_remaining--; return; }
    int stable = a->convergence.converged && !a->epc.active && !a->regime.main_paused;
    if (!stable) { a->misadj_stable_count = 0; return; }
    /* Threshold-gate counter: sole reader is the
     * "< a->cfg.filter_misadjustment_stable_frames" comparison on the next
     * line. "stable" (converged, EPC inactive,
     * main not paused) is the ordinary steady state for a fixed-position
     * device left running -- it can hold continuously for the entire
     * unbounded-overflow timeframe, so the `!stable` reset above does NOT
     * bound this counter (a reset branch existing here does not by itself
     * guarantee periodic resets). Saturate at filter_misadjustment_stable_frames
     * -- once reached, the "<" comparison is permanently false either way,
     * so this is observationally identical to the old unconditional
     * increment for every reachable state, while eliminating the eventual
     * signed-integer-overflow UB on a very long streaming session. */
    if (a->misadj_stable_count < a->cfg.filter_misadjustment_stable_frames)
        a->misadj_stable_count++;
    if (a->misadj_stable_count < a->cfg.filter_misadjustment_stable_frames) return;
    if (a->misadj_inv <= 10.0f) return;
    float base = a->misadj_inv > 1e-6f ? a->misadj_inv : 1e-6f;
    float scale_raw = 2.0f / sqrtf(base);
    float scale = scale_raw;
    if (scale < a->cfg.filter_misadjustment_scale_min) scale = a->cfg.filter_misadjustment_scale_min;
    if (scale > a->cfg.filter_misadjustment_scale_max) scale = a->cfg.filter_misadjustment_scale_max;
    pbfdkf_scale_filter(&a->main_filter, scale);
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_hangover_remaining = a->cfg.filter_misadjustment_hangover_frames;
}

/* max() of the last 15 inst-ERLE ring entries — the warm-transfer gate input.
 * Mirrors Python `max(list(self._erle_slope_buf)[-15:] or [0.0])`
 * (orchestrator.py:1409): 0.0 when the ring is empty, else the true max
 * (which may be negative) of the most recent <=15 appends. */
static float aec_erle_ring_max_last15(const Aec* a) {
    int n = a->erle_slope_len < 15 ? a->erle_slope_len : 15;
    if (n <= 0) return 0.0f;
    int idx = a->erle_slope_head;   /* head = next write = one past newest */
    float m = 0.0f; int got = 0;
    for (int i = 0; i < n; i++) {
        idx = (idx - 1 + a->erle_slope_cap) % a->erle_slope_cap;
        float v = a->erle_slope_buf[idx];
        if (!got || v > m) { m = v; got = 1; }
    }
    return m;
}

/* Public "is the linear filter demonstrably cancelling" test -- see aec.h.
 * TWO readings, because neither alone answers the question over the whole
 * life of a lock: windowed ERLE is the sustained evidence but lags ~0 for a
 * few hundred ms after every realign (the lag documented on
 * cfg.delay_acquire_inst_erle_db), and the inst-ERLE peak is the recent
 * evidence but ages out of its ~15-frame ring between far-active bursts.
 * Both thresholds are the ones Path A's own acquire-time guard already uses;
 * nothing new is introduced here. */
int aec_linear_is_cancelling(const Aec* a) {
    if (!a) return 0;
    return (a->last_erle_windowed > 2.5f)
           || (aec_erle_ring_max_last15(a) > a->cfg.delay_acquire_inst_erle_db);
}

/* ───────────────────────── process ─────────────────────────────────────── */

/* The context-only split's shared core: everything aec_process()/
 * aec_process_context() share -- steps 1-17
 * (linear filter + AEC3 post/RES block, writing a->final_out) and steps 19-20
 * (power EMAs + convergence detection, which the NEXT hop's step-17 logic
 * depends on and therefore can never be skipped by either caller). Does NOT
 * run step 21 (emit into a caller `out` buffer) -- that is caller-specific
 * and lives in the aec_process() wrapper below. Since the custom output
 * limiter was removed, the core IS the whole audio path: the two public
 * entry points differ only by that final copy and share all state, so they
 * may be mixed freely on one instance.
 *
 * Far-end-FFT sharing: shared_far_spec (NULL for the normal aec_process()/
 * aec_process_context() path -- byte-identical to before this sharing
 * feature existed) lets a
 * caller supply an externally-computed far-end spectrum instead of having
 * this instance compute its own via FFT, for multi-instance callers (e.g.
 * a 4-lane wrapper) whose lanes all consume the IDENTICAL far-end signal
 * every hop: one lane computes the real FFT, the rest borrow it. This
 * plugs into the exact same one-shot precomputed_far_spec mechanism
 * aec.c already uses INTERNALLY to let the main filter borrow the shadow
 * filter's far_spec instead of recomputing it (see the step-9 comment
 * below) -- shared_far_spec just gives that mechanism an external source
 * instead of an internal one. ref_in is still required even when
 * shared_far_spec is supplied: pbfdaf_frontend() unconditionally updates
 * far_buffer (the OLA history) from it, and every non-FFT use of the raw
 * time-domain far signal in this function (saturation detection, delay
 * estimation, mu_scale, ...) is unaffected by far-end-FFT sharing and
 * still needs it. */
static void aec_process_core(Aec* a, const float* mic_in, const float* ref_in,
                              const Complex* shared_far_spec) {
    const int hop = a->hop_size, K = a->n_freqs, N = a->n_partitions;
    int stationarity_block_for_post;
    /* AecLinearContext bookkeeping: CHANGED is "generation moved during this
     * hop", so snapshot at entry; far_hop_aligned is re-derived every hop. */
    a->delay_gen_hop_start = a->delay_generation;
    a->far_hop_aligned = 0;
    memcpy(a->near_hop, mic_in, (size_t)hop * sizeof(float));
    memcpy(a->far_hop,  ref_in, (size_t)hop * sizeof(float));

    /* 1. mic HPF (ref HPF OFF). */
    if (a->hp_mic) hpf_process(a->hp_mic, a->near_hop, hop);

    /* Held "near-end seen recently" gate for DT-aware soft recovery (mirrors
     * Python orchestrator 16285fd). Reads the PREVIOUS frame's dt_from_energy
     * (doubletalk_update_energy_dt runs later in this frame, in the RES block),
     * so the value here is the prior hop's — exactly like Python's _dt_from_energy
     * property read at the top of process(). dt_from_energy ONLY: it is ~0 in
     * far-end single-talk (mic ≈ echo), so FS never arms the gate (FS echo depth
     * preserved). Require `sustain` consecutive frames above threshold before
     * arming, then hold for `hold` frames. */
    {
        float ne_ind = a->dt_analyzer.dt_from_energy;
        if (ne_ind > a->cfg.ne_recent_threshold) {
            /* Threshold-gate counter: sole reader is the
             * ">= a->cfg.ne_recent_sustain" comparison two lines below.
             * Confirmed via a live UBSan repro: sustained near-end energy
             * above ne_recent_threshold (e.g. a
             * long nearend-singletalk / continuous-speech stretch) can hold
             * this branch for the entire unbounded-overflow timeframe with
             * no reset in between, so an unconditional `+= 1` would
             * eventually signed-integer-overflow on a very long streaming
             * session. Saturate at ne_recent_sustain -- once reached the
             * ">=" comparison is permanently true either way, so this is
             * observationally identical to the old unconditional increment
             * for every reachable state. */
            if (a->ne_above < a->cfg.ne_recent_sustain)
                a->ne_above += 1;
        } else
            a->ne_above = 0;
        if (a->ne_above >= a->cfg.ne_recent_sustain)
            a->ne_recent_frames = a->cfg.ne_recent_hold;
        else if (a->ne_recent_frames > 0)
            a->ne_recent_frames -= 1;
        else
            a->ne_recent_frames = 0;
    }

    /* 2. saturation. */
    float sat_ref = 0.0f;  /* receive: Saturation (Stage-3) getter returns double */
    if (a->has_sat) {
        sat_ref = saturation_detect(&a->sat_ref, a->far_hop, hop);
        float sat_mic = saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5f) ? sat_ref : sat_mic * 0.5f;
        if (a->cfg.saturation_softclip_ref && sat_ref > 0.1f)
            saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
    }

    /* 3. delay estimation + ring-buffer alignment. Duty-cycled analysis is
     * ALWAYS ON (baked in — see the duty_* field doc in aec.h): the estimator
     * is FED every hop regardless (accumulate_ex keeps decimators + ring
     * gapless); only the matched-filter analysis is decimated to 1-in-K once
     * the estimate has been solid+unchanged for delay_est_init_s. This is an
     * intentional, sampled-cost-free divergence from the Python reference
     * (which always analyses every hop).
     *
     * Structure (mirrors the Python orchestrator exactly): the OUTER gate is
     * "does a reference ring exist" (MATCHED or FIXED -- Python's
     * `_delay_active`); the INNER gate is "is there an estimator to run"
     * (MATCHED only -- Python's `delay_est is not None`). FIXED therefore
     * shares the ring write + delay-compensating read below verbatim and
     * simply never runs the matched filter, since its applied delay came
     * from the caller's bring-up measurement and never changes. */
    if (aec_ring_active(&a->cfg)) {
      if (a->has_delay) {
        float hop_s = (float)hop / (float)a->cfg.sample_rate;
        int hold_hops = (int)lrintf(a->cfg.delay_est_init_s / hop_s);
        int K = (int)lrintf(a->cfg.delay_est_period_s / hop_s) / 5;
        int run_filter = 1;
        if (hold_hops < 1) hold_hops = 1;
        if (K < 2) K = 2;
        if (a->duty_active) {
            a->duty_pos += 1;
            if (a->duty_pos >= K) a->duty_pos = 0;
            run_filter = (a->duty_pos == 0);
        }
        /* Engagement census (diagnostic; see the duty_hops_* doc in aec.h).
         * Counted HERE, at the one call site that consumes run_filter, so
         * the census can never drift from what actually executed. */
        a->duty_hops_total += 1;
        if (run_filter) a->duty_hops_run += 1;
        delay_aec3_accumulate_ex(&a->delay, a->near_hop, a->far_hop, hop,
                                 run_filter);
        {
            int cur = delay_aec3_estimated_delay(&a->delay);
            int solid = delay_aec3_is_solid(&a->delay);
            if (!solid || cur < 0 || cur != a->duty_last_delay) {
                /* estimate moved / lost confidence → full rate + re-arm */
                a->duty_active = 0;
                a->duty_stable_hops = 0;
            } else if (!a->duty_active) {
                a->duty_stable_hops += 1;
                if (a->duty_stable_hops >= hold_hops) {
                    a->duty_active = 1;
                    a->duty_pos = 0;
                    a->duty_erle_peak = a->last_erle_windowed;
                }
            }
            a->duty_last_delay = cur;
        }
        if (a->duty_active) {
            /* ERLE watchdog: leaky peak (~0.1 dB/s at 10 ms hops); resume
             * full-rate analysis on a >6 dB collapse from that peak. Armed
             * only once the peak exceeds 6 dB so it cannot fire before the
             * filter ever converged. */
            if (a->last_erle_windowed > a->duty_erle_peak)
                a->duty_erle_peak = a->last_erle_windowed;
            else
                a->duty_erle_peak -= 0.001f;
            if (a->duty_erle_peak > 6.0f &&
                a->last_erle_windowed < a->duty_erle_peak - 6.0f) {
                a->duty_active = 0;
                a->duty_stable_hops = 0;
                a->duty_erle_peak = 0.0f;
            }
        }
        int new_delay = delay_aec3_estimated_delay(&a->delay);
        int eligible = (new_delay >= 0 && delay_aec3_n_updates(&a->delay) >= 3
                        /* Defensive: a delay the ring cannot hold would alias
                         * through the modulo read below and silently return
                         * wrong (effectively future) far. Unreachable with the
                         * default ring (2048 ms) vs the matched filter's
                         * ~509 ms span; reachable only when a caller shrinks
                         * delay_buffer_ms/max_delay_ms. */
                        && new_delay <= a->ref_ring_size - hop);

        /* Path A — first acquisition. */
        int already_cancelling = a->cfg.delay_acquire_protect_converged
                                 && (a->last_erle_windowed > 2.5f);
        /* option-A (default-OFF): also protect via inst-ERLE peak (orch 1387). */
        if (a->cfg.delay_acquire_protect_inst_erle
                && aec_erle_ring_max_last15(a) > a->cfg.delay_acquire_inst_erle_db)
            already_cancelling = 1;
        if (eligible && a->current_delay < 0 && delay_aec3_is_solid(&a->delay)
                && !already_cancelling) {
            a->current_delay = new_delay;
            /* Single bump site covers warm-transfer, soft and hard branches
             * below alike -- generation tracks the ring offset, not the
             * recovery flavour. */
            if (a->delay_generation != 0xFFFFFFFFu) a->delay_generation++;
            /* Warm tap-transfer (orch 1407-1422): if the filter is already
             * cancelling (inst-ERLE peak > thresh) AND the delay fits the tap
             * reach, shift the learned IR by the delay instead of zeroing — the
             * cold-start cancellation survives the realign (the "line" fix).
             * Outside that, fall through to soft / hard reset unchanged. */
            int warm_ok = 0;
            if (a->cfg.delay_acquire_warm_transfer) {
                float wpk = aec_erle_ring_max_last15(a);
                int reach = a->main_filter.base.n_partitions
                            * a->main_filter.base.hop_size;
                warm_ok = (wpk > a->cfg.delay_acquire_inst_erle_db)
                          && (new_delay > 0) && (new_delay < reach);
            }
            if (warm_ok) {
                pbfdaf_warm_shift_ir(&a->main_filter.base, new_delay);
                if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, new_delay);
            } else if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
                /* Soft acquisition (mirrors Python 16285fd): apply the ring
                 * alignment (current_delay set above) + reset filter excitation
                 * counters ONLY; keep the converged taps + let them re-adapt at
                 * normal step size. No tap wipe, no mark_diverged, no AecState
                 * full reset (no pending_delay_change). Avoids the aggressive-
                 * recovery tail overfitting near-end that arrives after this
                 * (far-only) acquisition. */
                pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
                if (a->has_shadow) {
                    a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                    a->shadow_filter.call_counter = 0;
                }
            } else {
                aec_reset_filter_derived_state(a);
                filter_convergence_mark_diverged(&a->convergence);
                pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
                /* shadow is PBFDAF: handle_echo_path_change has no PBFDAF variant;
                 * the Python shadow.handle_echo_path_change resets the coarse
                 * counter. Mirror via the PBFDAF counters directly. */
                if (a->has_shadow) {
                    a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                    a->shadow_filter.call_counter = 0;
                }
                a->pending_delay_change = AEC_DA_NEW_DETECTED;
            }
        }

        /* pending TTL aging (once per estimation cycle / per hop). */
        if (a->pending_delay_ttl > 0) {
            a->pending_delay_ttl--;
            if (a->pending_delay_ttl <= 0) { a->has_pending = 0; a->pending_delay = -1; }
        }

        float conf = delay_aec3_confidence(&a->delay);

        /* Backward-jump quarantine for Path B. The rule it implements, its
         * evidence test and the defect it replaces are documented once, on
         * delay_backward_quarantine_enabled in aec.h. What only this site
         * knows:
         *   - it is evaluated BEFORE Path B and repeats every one of Path B's
         *     own admission terms, so the window can only ever be spent on a
         *     candidate Path B would otherwise have accepted this cycle;
         *   - `left < 0` is unarmed and `left == 0` is expiry: the countdown
         *     is armed once, on the first refusal, and ticks unconditionally
         *     while armed, so the total veto stays bounded however the
         *     candidate jitters.
         * Mirrors orchestrator.py's _change_blocked. */
        int change_blocked = 0;
        if (a->cfg.delay_backward_quarantine_enabled
                && eligible && a->current_delay >= 0 && conf >= 0.5f
                && new_delay < a->current_delay
                && a->current_delay - new_delay > 32
                && aec_linear_is_cancelling(a)) {
            if (a->delay_quarantine_left < 0)
                a->delay_quarantine_left = a->delay_quarantine_hops;
            if (a->delay_quarantine_left > 0) {
                a->delay_quarantine_left--;
                change_blocked = 1;
            }
        } else {
            a->delay_quarantine_left = -1;
        }

        /* Path B — delay shift. */
        if (eligible && a->current_delay >= 0 && conf >= 0.5f
                && abs(new_delay - a->current_delay) > 32
                && !change_blocked) {
            if (a->has_pending && abs(new_delay - a->pending_delay) < 16) {
                a->current_delay = new_delay;
                /* Same single-site rule as Path A: the soft-realign branch
                 * below deliberately sets no other flag, so this bump is the
                 * ONLY externally visible trace of a soft shift. */
                if (a->delay_generation != 0xFFFFFFFFu) a->delay_generation++;
                a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
                if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
                    /* Soft realign (mirrors Python 16285fd): the ring read
                     * offset is already updated (current_delay above). Keep the
                     * converged filter, re-adapt at normal step size — no tap
                     * wipe, no Kalman P-override, no q-boost, no mark_diverged,
                     * no epc_force_delay, no pending_delay_change. Avoids the
                     * over-cancel + near-end overfit when movement re-locks
                     * through double-talk. (Python only sets the dead
                     * _epc_reset_fired_this_frame FQA flag here — no audio
                     * effect, not modelled in C.) */
                } else {
                    aec_reset_filter_derived_state(a);
                    epc_force_delay(&a->epc);
                    filter_q_high(&a->main_filter);
                    if (a->has_shadow) {
                        a->shadow_filter.poor_excitation_counter = AEC3_POOR_EXC_COUNTER_INITIAL_HOPS;
                        a->shadow_filter.call_counter = 0;
                    }
                    filter_convergence_mark_diverged(&a->convergence);
                    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
                    a->pending_delay_change = AEC_DA_NEW_DETECTED;
                }
            } else {
                a->pending_delay = new_delay; a->has_pending = 1; a->pending_delay_ttl = 3;
            }
        }
      }   /* end MATCHED-only estimator block */

        /* ring write. */
        int w = a->ref_ring_write, rs = a->ref_ring_size;
        if (w + hop <= rs) memcpy(a->ref_ring + w, a->far_hop, (size_t)hop * sizeof(float));
        else {
            int p1 = rs - w;
            memcpy(a->ref_ring + w, a->far_hop, (size_t)p1 * sizeof(float));
            memcpy(a->ref_ring, a->far_hop + p1, (size_t)(hop - p1) * sizeof(float));
        }
        a->ref_ring_write = (w + hop) % rs;
        /* Saturate at ring capacity: every comparison below only needs
         * "filled >= current_delay + hop" and current_delay is capped at
         * rs - hop, so rs is the largest value ever meaningful. Unbounded,
         * this int overflows (UB) after ~37 h of continuous 16 kHz audio.
         * Guard BEFORE adding -- the increment itself must never run near
         * INT_MAX (same freeze-not-wrap rule as the other counters, see
         * test_counter_saturation.c). */
        if (a->ref_ring_filled < rs) {
            a->ref_ring_filled += hop;
            if (a->ref_ring_filled > rs) a->ref_ring_filled = rs;
        }

        /* delay compensation read. */
        if (a->current_delay > 0 && a->ref_ring_filled >= a->current_delay + hop) {
            int d = a->current_delay;
            int read_pos = (a->ref_ring_write - hop - d) % rs;
            if (read_pos < 0) read_pos += rs;
            if (read_pos + hop <= rs)
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)hop * sizeof(float));
            else {
                int p1 = rs - read_pos;
                memcpy(a->far_hop, a->ref_ring + read_pos, (size_t)p1 * sizeof(float));
                memcpy(a->far_hop + p1, a->ref_ring, (size_t)(hop - p1) * sizeof(float));
            }
            a->far_hop_aligned = 1;
        } else if (a->current_delay == 0) {
            /* A zero applied delay needs no ring read: the raw far IS the
             * aligned far. */
            a->far_hop_aligned = 1;
        }
    } else {
        /* EXTERNAL_ALIGNED: the caller CONTRACTED that `ref` is already
         * aligned to `mic`, so the far this hop is aligned by definition --
         * there is nothing to buffer and nothing to shift. Seam bookkeeping
         * only (aec_get_linear_context); no audio path reads this flag, so
         * this is byte-identical to the pre-delay_mode enable_delay_est=0
         * behaviour, which simply had no way to say "aligned by contract". */
        a->far_hop_aligned = 1;
    }

    /* far_hop is fully finalized above (memcpy + optional soft-clip +
     * optional delay-ring re-read) and is never written again for the rest
     * of this hop (every remaining reference below is a read) -- compute
     * mean_sq(far_hop) ONCE here and reuse it at every call site that would
     * otherwise recompute the identical deterministic value from the same
     * unchanged input. */
    float far_hop_mean_sq = mean_sq(a->far_hop, hop, a->scr_sq);

    /* 4. render activity. */
    RenderActivityResult ra = render_activity_update(&a->render_activity, a->far_hop, hop);
    a->warmup_far_active = ra.warmup_active;
    float far_pwr_global = ra.far_pwr;

    /* 5. mu_scale (simple variable mu, Valin 2007 RER-inspired). */
    int mu_is_array = 0;
    float mu_scalar = get_simple_mu_scale(a, &mu_is_array);

    /* 6. mic-clip emergency. */
    if (a->has_sat && a->sat_mic.saturation_level > 0.8f) {
        mu_scalar = 0.0f;
        mu_is_array = 0;
    }

    /* 7. RSA update + poor-excitation counters + saturated_capture. The PSD is
     *    |filter.far_spec|² from the PREVIOUS hop (far_spec set inside step-9
     *    pbfdkf_process). First hop: far_spec is zero. */
    {
        float *rsa_psd = a->scr_rsa_psd;
        sk_cmag2_np_f32(a->main_filter.base.far_spec, rsa_psd, K);
        rsa_update(&a->rsa, rsa_psd, a->far_hop, hop);
        int poor = rsa_poor_signal_excitation(&a->rsa);
        /* Ceilinged at each filter's own n_partitions (UBSan-confirmed
         * signed-overflow fix): poor_excitation_counter's only production
         * consumers are pbfdaf_process's/pbfdkf_process's
         * `poor_excitation_counter < N` gates, each read against THIS
         * SAME filter instance's own n_partitions (N) -- confirmed by grep,
         * no other consumer anywhere in src/include/python. Not read
         * bit-exact by any parity harness (test/historical/parity_pbfdkf.c and
         * test/historical/parity_pbfdkf_loc.c treat it purely as an
         * orchestrator-supplied external input, set on the C filter before
         * calling process(), never asserted afterward). Once the counter
         * reaches n_partitions, further increments can never change the
         * `< n_partitions` outcome (already false, stays false), so
         * gating the increment at n_partitions is observationally
         * identical to the old unconditional `+= 1` for every reachable
         * state, while eliminating the eventual signed overflow. */
        if (poor) a->main_filter.base.poor_excitation_counter = 0;
        else if (a->main_filter.base.poor_excitation_counter <
                 a->main_filter.base.n_partitions)
            a->main_filter.base.poor_excitation_counter += 1;
        a->main_filter.base.saturated_capture = (a->saturation_level > 0.5f);
        if (a->has_shadow) {
            if (poor) a->shadow_filter.poor_excitation_counter = 0;
            else if (a->shadow_filter.poor_excitation_counter <
                     a->shadow_filter.n_partitions)
                a->shadow_filter.poor_excitation_counter += 1;
            a->shadow_filter.saturated_capture = (a->saturation_level > 0.5f);
        }
    }

    /* 8. push PREVIOUS hop's _block_stationary_for_next_hop latch onto BOTH
     *    filters (orchestrator 1787-1790). Main consumes it at step 9, shadow
     *    at step 11 — so it MUST be applied here, before step 10 recomputes the
     *    next-hop latch. */
    a->main_filter.base.block_stationary = a->block_stationary_next;
    if (a->has_shadow) a->shadow_filter.block_stationary = a->block_stationary_next;

    /* The RSA narrowband mask is applied INSIDE Python's _update_weights for
     * BOTH main and shadow (filters.py:338-341 + 627-629); the C pbfdkf/pbfdaf
     * omit it. Compute it ONCE here and apply to main (step 9) + shadow (11). */
    float *rsa_mask = a->scr_rsa_mask;
    for (int k = 0; k < K; ++k) rsa_mask[k] = 1.0f;
    rsa_mask_regions_around_narrow_bands(&a->rsa, rsa_mask);
    int rsa_mask_active = 0;
    for (int k = 0; k < K; ++k) if (rsa_mask[k] < 1.0f) { rsa_mask_active = 1; break; }

    /* 8.5 E15: shadow runs BEFORE main so main's H_error refresh reads
     *     same-hop e2_coarse (mirrors orchestrator.py L1576-1590 E15 fix).
     *     far_excited hoisted here so step 13 (doubletalk_update_shadow_dt) can use it. */
    int far_excited = 0;
    if (a->has_shadow) {
        a->shadow_frame_count++;
        far_excited = (far_hop_mean_sq > 1e-4f);
        int saturation_safe_pre = (a->saturation_level < 0.5f);
        float shadow_mu_pre = (far_excited && saturation_safe_pre) ? 1.0f : 0.1f;
        /* Borrow the caller-supplied spectrum instead of computing
         * this instance's own far FFT. NULL (the normal aec_process()/
         * aec_process_context() path) is a pure no-op here -- shadow's
         * precomputed_far_spec is already NULL from init/reset, so this
         * assignment changes nothing in that case. */
        a->shadow_filter.precomputed_far_spec = shared_far_spec;
        if (rsa_mask_active) {
            float *mu_buf_pre = a->scr_mu_buf_pre;
            for (int k = 0; k < K; ++k) mu_buf_pre[k] = shadow_mu_pre * rsa_mask[k];
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, mu_buf_pre, 0.0f, a->shadow_out);
        } else {
            pbfdaf_process(&a->shadow_filter, a->near_hop, a->far_hop, NULL,
                           shadow_mu_pre, a->shadow_out);
        }
        /* Publish current-hop e2_coarse to main_filter BEFORE main runs. */
        {
            float *e2coa_pre = a->scr_e2coa_pre;
            sk_cmag2_np_f32(a->shadow_filter.error_spec, e2coa_pre, K);
            a->main_filter.e2_coarse_for_refresh =
                (float)aec3_post_pairwise_sum_f32(e2coa_pre, (size_t)K);
            for (int k = 0; k < K; ++k)
                a->main_filter.e2_coarse_per_bin[k] = e2coa_pre[k];
            a->main_filter.e2_coarse_per_bin_valid = 1;
        }
    }

    /* 9. MAIN filter. */
    /* FFT dedup: the shadow ran pre-main on the SAME far_hop with an identical
     * far_buffer (lockstep shift + paired reset), so reuse its far_spec instead
     * of recomputing — byte-equal, saves 1 FFT/hop. One-shot (frontend clears).
     * When there's no shadow to borrow from, fall through to shared_far_spec
     * instead of unconditionally computing a fresh FFT -- NULL in
     * the normal path, so this is unchanged from before far-end-FFT sharing
     * existed whenever no caller-supplied spectrum exists. */
    a->main_filter.base.precomputed_far_spec =
        a->has_shadow ? a->shadow_filter.far_spec : shared_far_spec;
    float main_mu_scalar = a->regime.main_paused ? 0.0f : mu_scalar;
    {
        float *mu_buf = a->scr_mu_buf;
        int use_array = mu_is_array;
        if (mu_is_array) {
            float pause = a->regime.main_paused ? 0.0f : 1.0f;
            for (int k = 0; k < K; ++k) mu_buf[k] = a->per_bin_mu_scale[k] * pause;
        }
        if (rsa_mask_active && !use_array) {
            /* Python: scalar → full(scalar) → × mask. */
            for (int k = 0; k < K; ++k) mu_buf[k] = main_mu_scalar * rsa_mask[k];
            use_array = 1;
        } else if (rsa_mask_active && use_array) {
            for (int k = 0; k < K; ++k) mu_buf[k] = mu_buf[k] * rsa_mask[k];
        }
        if (use_array)
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, mu_buf, 0.0f, a->raw_output);
        else
            pbfdkf_process(&a->main_filter, a->near_hop, a->far_hop, NULL,
                           main_mu_scalar, a->raw_output);
    }

    /* 10. stationarity refresh for NEXT hop (StationarityEstimator on
     *     |filter.far_spec|²; first-render latch; block-stationary push). */
    {
        float far_max = 0.0f;
        for (int i = 0; i < hop; ++i) { float v = fabsf(a->far_hop[i]); if (v > far_max) far_max = v; }
        if (!a->non_zero_render_seen && far_max >= a->render_peak_floor)
            a->non_zero_render_seen = 1;
        if (a->non_zero_render_seen) {
            float *far_psd = a->scr_far_psd;
            sk_cmag2_np_f32(a->main_filter.base.far_spec, far_psd, K);
            stationarity_estimator_update_noise_estimator(&a->a3_stat, far_psd);
            /* E16: pass avg_reverb from previous hop's aec3_post_compute_x2_reverb —
             * same state Python reads from _aec3_avg_render_reverb.reverb (one-hop stale,
             * aec3_post_run fires at step 18, AFTER this stationarity refresh). */
            stationarity_estimator_update_stationarity_flags(&a->a3_stat, far_psd,
                                                              a->post.avg_reverb.reverb);
            /* Threshold-gate counter: sole reader is the
             * ">= a->stationarity_converge_hops" comparison just below (and
             * the same comparison re-derived from the mirrored copy passed
             * into aec3_post_run's inputs). Saturate at EXACTLY
             * stationarity_converge_hops -- once the counter reaches it the
             * ">=" comparison is permanently true either way -- so a very
             * long streaming session can never hit signed-integer-overflow
             * UB on this unconditional increment.
             *
             * Off-by-one fix: the guard used to read
             * "<= stationarity_converge_hops", which lets the counter take
             * one extra step to stationarity_converge_hops + 1 before the
             * guard stops firing -- one more than this comment's own claimed
             * cap, though NOT a behavior change (both
             * stationarity_converge_hops and stationarity_converge_hops + 1
             * satisfy the ">=" comparison identically, forever, so every
             * downstream decision -- here and in aec3_post.c's mirrored
             * "in->stationarity_active_hops >= in->stationarity_converge_hops"
             * copy -- is unaffected either way; see
             * test/test_counter_saturation.c's
             * "stationarity_active_hops settles at converge_hops" case,
             * which pins the exact numeric cap so this can't silently drift
             * again). "<" makes the counter settle at exactly
             * stationarity_converge_hops, matching this comment's claim. */
            if (a->stationarity_active_hops < a->stationarity_converge_hops) {
                a->stationarity_active_hops += 1;
            }
        }
        /* latch flag for the NEXT hop (applied at step 8 of hop+1). Do NOT
         * overwrite this hop's filter.block_stationary — step 11's shadow
         * still reads the value pushed at step 8. */
        int converged_enough = (a->stationarity_active_hops >= a->stationarity_converge_hops);
        /* This raw value is also consumed later by REE in aec3_post_run. No
         * stationarity flags or hangovers change between here and that call,
         * so carry it forward instead of scanning every frequency bin twice
         * after convergence. Keep it separate from block_stationary_next:
         * that latch is convergence-gated, while REE expects the raw result. */
        stationarity_block_for_post =
            stationarity_estimator_is_block_stationary(&a->a3_stat);
        a->block_stationary_next =
            converged_enough && stationarity_block_for_post;
    }

    /* 11. SHADOW filter — already ran pre-main in step 8.5 (E15 fix). */

    /* 12. e2_coarse + erl publish + poor-coarse rescue. */
    if (a->has_shadow) {
        /* _e2_ref = Σ|filter.error_spec|² (cmag2_np, pairwise f32). */
        float *e2ref_arr = a->scr_e2ref_arr, *erl_arr = a->scr_erl_arr;
        sk_cmag2_np_f32(a->main_filter.base.error_spec, e2ref_arr, K);
        float e2_ref = aec3_post_pairwise_sum_f32(e2ref_arr, (size_t)K);
        /* e2_coa: shadow_filter.error_spec did NOT change since step 8.5
         * (the shadow filter runs once/hop, pre-main; nothing between here
         * and there touches it) -- so cmag2(shadow_filter.error_spec) +
         * its pairwise-sum + per-bin publish were already computed and
         * published there. Reuse a->main_filter.e2_coarse_for_refresh /
         * e2_coarse_per_bin[] instead of recomputing the identical values
         * a second time this hop. */
        float e2_coa = a->main_filter.e2_coarse_for_refresh;
        /* erl[k] = Σ_p |W_p[k]|². zero-init once, then accumulate every
         * partition (including p==0) via the acc kernel — matches the
         * original zero-fill-then-`+=`-every-partition shape exactly. */
        for (int k = 0; k < K; ++k) erl_arr[k] = 0.0f;
        for (int part = 0; part < N; ++part) {
            const Complex* Wp = a->main_filter.base.W + (size_t)part * K;
            sk_cmag2_np_acc_f32(Wp, erl_arr, K);
        }
        for (int k = 0; k < K; ++k) a->main_filter.erl_per_bin[k] = erl_arr[k];

        /* rescue: cond_fire = e2_ref < 0.5*e2_coa; threshold_hops live-computed
         * at construction (poor_coarse_threshold_hops), not a frozen literal. */
        int cond_fire = (e2_ref < 0.5f * e2_coa);
        int threshold_hops = a->poor_coarse_threshold_hops;
        if (cond_fire) a->poor_coarse_counter += 1; else a->poor_coarse_counter = 0;
        if (a->poor_coarse_counter >= threshold_hops) {
            pbfdaf_copy_weights_from(&a->shadow_filter, &a->main_filter.base);
            a->coarse_reset_hangover = a->coarse_reset_hangover_hops;
            a->poor_coarse_counter = 0;
            ree_reset(&a->a3_ree);          /* REE reset on rescue rising edge. */
        }
        /* Track F: sustained leakage_div gate. Fires after
         * leakage_div_sustain_hops (~50 ms wall-clock, live-computed) of
         * consecutive hops with >50% bins on the diverged-leakage branch —
         * covers the DT-onset window before coarse_reset_hangover kicks in.
         * Mirrors orchestrator.py _leakage_div_sustained_counter /
         * _dt_leakage_gate. */
        float ld_frac = a->main_filter.last_leakage_div_frac;
        int ld_sustain_hops = a->leakage_div_sustain_hops;
        if (ld_frac > 0.5f) {
            if (a->leakage_div_sustained_counter < 2 * ld_sustain_hops)
                a->leakage_div_sustained_counter++;
        } else {
            if (a->leakage_div_sustained_counter > 0)
                a->leakage_div_sustained_counter--;
        }
        int dt_leakage_gate = (a->leakage_div_sustained_counter >= ld_sustain_hops);
        if (a->coarse_reset_hangover > 0) {
            a->coarse_reset_hangover--;
            a->main_filter.disallow_leakage_diverged = 1;
        } else if (dt_leakage_gate) {
            a->main_filter.disallow_leakage_diverged = 1;
        } else {
            a->main_filter.disallow_leakage_diverged = 0;
        }
    }

    /* 13. smoothed errs + DT analyzer + regime handler. */
    if (a->has_shadow) {
        float main_err   = pbfdkf_get_error_energy(&a->main_filter);
        float shadow_err = pbfdaf_get_error_energy(&a->shadow_filter);
        float as = a->cfg.shadow_err_alpha, oas = 1.0f - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;
        doubletalk_update_shadow_dt(&a->dt_analyzer, a->shadow_frame_count,
                                    far_excited, a->main_err_smooth, a->shadow_err_smooth);
        int delay_reliable = a->has_delay && (delay_aec3_confidence(&a->delay) >= 0.5f);
        ShadowCopyDecision dec = shadow_copy_update(
            &a->regime, a->shadow_frame_count, far_hop_mean_sq,
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
        if (dec.boost_q) filter_q_high(&a->main_filter);
        /* reverse_copy: no-op on NLMS shadow (Python never invokes it). */
    }

    /* 14. EPV trigger. */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                  a->convergence.converged, a->regime.main_paused);
    /* (Python's _epv_suppressed weak-filter damping is env-gated, default OFF,
     * so epv.fired here == fired-and-not-suppressed in the balanced config.) */
    if (epv.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
        /* Soft (mirrors Python 16285fd): EPV gain-change recovery is
         * double-talk-blind (far-power EMA swing only). While near recently
         * present, skip the aggressive Kalman re-adapt + mark_diverged + ERLE
         * reset; the converged filter tracks moderate gain change at its normal
         * step size. (Python only sets the dead _epc_reset_fired_this_frame FQA
         * flag — no audio effect, not modelled in C.) */
    } else if (epv.fired) {
        a->pending_gain_change = 1;
        pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
        filter_q_high(&a->main_filter);
        filter_convergence_mark_diverged(&a->convergence);
        a->epc_render_forced_remaining = a->cfg.epc_hangover;
        if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
    }

    /* 15. shadow_rise (only if shadow + converged); else hangover tick. */
    if (a->has_shadow && a->convergence.converged) {
        EpcEvent rise = epc_update_shadow_rise(&a->epc, a->main_err_smooth,
                                               a->shadow_err_smooth,
                                               a->render_activity.is_stationary);
        if (rise.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
            /* Soft (mirrors Python 16285fd): shadow_rise recovery is
             * double-talk-blind (only a far-stationarity guard). While near
             * recently present, skip the aggressive re-adapt + mark_diverged +
             * ERLE reset; its tail overfits near-end that arrives after the
             * (far-only) trigger. No hangover tick (Python ticks only when rise
             * did NOT fire). (Python only sets the dead _epc_reset_fired_this_frame
             * FQA flag here — no audio effect, not modelled in C.) */
        } else if (rise.fired) {
            a->pending_gain_change = 1;
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            filter_q_high(&a->main_filter);
            filter_convergence_mark_diverged(&a->convergence);
            a->epc_render_forced_remaining = a->cfg.epc_hangover;
            if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
        } else {
            epc_tick_hangover(&a->epc);
        }
    }

    /* final_output starts from raw_output. */
    memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));

    /* 16. misadjustment estimator update + fire. */
    misadj_update(a, a->near_hop, a->raw_output, hop);
    misadj_fire(a);

    /* inst-ERLE slope ring: append get_erle_instant() (orchestrator.py:2382)
     * UNCONDITIONALLY every freq-path hop — Python appends OUTSIDE the
     * enable_res sub-block, so the warm-transfer gate works even with
     * enable_res=0 / return_res_context=0. Value = the SAME formula as the
     * erle_for_factor compute inside the block below (get_erle_instant uses
     * error_power == raw_error_power, alias set at orch 2563). near_power /
     * raw_error_power here are the prior-hop EMAs (updated later, step ~1958),
     * matching Python's read-before-update ordering. Read at Path-A (step 3)
     * is therefore 1-hop delayed, matching Python (read@1412 / append@2382). */
    {
        float _ei;
        if (a->near_power < 1e-10f && a->raw_error_power < 1e-10f) _ei = 0.0f;
        else _ei = 10.0f * log10f((a->near_power + 1e-10f) / (a->raw_error_power + 1e-10f));
        int _h = a->erle_slope_head;
        a->erle_slope_buf[_h] = _ei;
        a->erle_slope_head = (_h + 1) % a->erle_slope_cap;
        if (a->erle_slope_len < a->erle_slope_cap) a->erle_slope_len++;
    }

    /* 17. RES / post block (enable_res). */
    int is_stationary_dt = 0;
    float dt_indicator = 0.0f;
    float erle_windowed = 0.0f;
    float far_power = 0.0f;
    /* Run the AEC3 post block when EITHER the internal RES is on OR the caller
     * asked for the res-context seam (mirrors Python orchestrator.py:2021,
     * `enable_res or return_res_context`). With enable_res=1 this is unchanged
     * (return_res_context is irrelevant) → production cascade byte-exact. */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        far_power = far_hop_mean_sq;
        /* erle_windowed (step 13a). */
        const float erle_decay = 0.999f;
        a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
        a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
        erle_windowed = 10.0f * log10f((a->erle_window_near + 1e-10f)
                                     / (a->erle_window_err + 1e-10f));
        /* erle_for_factor = max(get_erle_instant(), erle_windowed). */
        float erle_inst;
        if (a->near_power < 1e-10f && a->raw_error_power < 1e-10f) erle_inst = 0.0f;
        else erle_inst = 10.0f * log10f((a->near_power + 1e-10f) / (a->raw_error_power + 1e-10f));
        float erle_for_factor = erle_inst > erle_windowed ? erle_inst : erle_windowed;
        float erle_factor = erle_for_factor / 10.0f;
        if (erle_factor < 0.0f) erle_factor = 0.0f;
        if (erle_factor > 1.0f) erle_factor = 1.0f;
        a->erle_factor_prev = erle_factor;

        float far_pwr = far_power + 1e-10f;
        float mic_pwr = mean_sq(a->near_hop, hop, a->scr_sq) + 1e-10f;
        float raw_err_pwr = mean_sq(a->raw_output, hop, a->scr_sq) + 1e-10f;

        /* erl tracking (B4). */
        if (far_pwr > 1e-4f) {
            float raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10f);
            float inst_erl_raw = mic_pwr / far_pwr;
            if (raw_dt_ratio < 2.0f && inst_erl_raw < 1.5f) {
                float inst_erl = inst_erl_raw;
                if (inst_erl < 0.001f) inst_erl = 0.001f; if (inst_erl > 1.0f) inst_erl = 1.0f;
                /* Per-hop retention EMAs authored at the legacy 10 ms hop
                 * grid; mirrors orchestrator.py's _alpha_erl_*. */
                float alpha_erl = a->convergence.converged
                                ? a->alpha_erl_converged
                                : a->alpha_erl_tracking;
                a->erl_estimate = alpha_erl * a->erl_estimate + (1.0f - alpha_erl) * inst_erl;
            }
        }

        doubletalk_update_energy_dt(&a->dt_analyzer, ra.is_active,
                                    far_pwr, mic_pwr, a->erl_estimate);

        /* base DT confidence (simple energy-ratio estimate). */
        float simple_dt = 1.0f - far_pwr / (mic_pwr + far_pwr);
        float raw_dt = a->dt_analyzer.dt_from_energy;
        float sd_half = simple_dt * 0.5f;
        if (sd_half > raw_dt) raw_dt = sd_half;

        /* stationary-DT macro detection. */
        if (a->render_activity.is_stationary && a->convergence.converged) {
            int fft_size = a->main_filter.base.fft_size;
            float freq_per_bin = (float)a->cfg.sample_rate / (float)fft_size;
            int vb_start = (int)(100.0f / freq_per_bin); if (vb_start < 1) vb_start = 1;
            int vb_limit = (int)(3000.0f / freq_per_bin); if (vb_limit > K) vb_limit = K;
            float track_err_pwr = 0.0f;
            {
                /* Scalar serial reduction (not an elementwise fill/acc target):
                 * fill-scratch-then-serial-sum, reusing scr_e2ref_arr, which is
                 * provably dead here (last written/read at step 12 above, not
                 * touched again until next hop's step 12) — the k-order
                 * `track_err_pwr +=` sum below is unchanged from the original
                 * inline-compute-then-accumulate loop, just reading the
                 * precomputed (bit-identical) per-bin magnitude values. */
                float *terr_scr = a->scr_e2ref_arr;
                sk_cmag2_np_f32(a->main_filter.base.error_spec + vb_start,
                                 terr_scr + vb_start, vb_limit - vb_start);
                for (int k = vb_start; k < vb_limit; ++k) track_err_pwr += terr_scr[k];
            }
            track_err_pwr += 1e-10f;
            if (a->wn_err_baseline < 1e-6f) a->wn_err_baseline = track_err_pwr;
            float jump_ratio = track_err_pwr / (a->wn_err_baseline + 1e-10f);
            /* 800ms protection window (covers syllable gaps), live-computed
             * at construction (stat_dt_hangover_hops), not a frozen literal. */
            if (jump_ratio > 1.5f) a->stat_dt_hangover = a->stat_dt_hangover_hops;
            if (a->stat_dt_hangover > 0) {
                is_stationary_dt = 1;
                a->stat_dt_hangover--;
                a->wn_err_baseline = 0.999f * a->wn_err_baseline + 0.001f * track_err_pwr;
            } else {
                is_stationary_dt = 0;
                a->wn_err_baseline = 0.95f * a->wn_err_baseline + 0.05f * track_err_pwr;
            }
        }
        /* D4 baseline slow-track. */
        if (a->convergence.converged && !a->render_activity.is_stationary
                && far_pwr > 1e-4f && a->wn_err_baseline > 1e-6f) {
            a->wn_err_baseline = 0.995f * a->wn_err_baseline + 0.005f * raw_err_pwr;
        }

        /* inst_erle correction. */
        float inst_erle_fast_raw = mic_pwr / raw_err_pwr;
        a->inst_erle_smooth = 0.7f * a->inst_erle_smooth + 0.3f * inst_erle_fast_raw;
        if (a->inst_erle_smooth > 2.0f) {
            float erle_for_dt = a->inst_erle_smooth;
            if (erle_for_dt > 4.0f) erle_for_dt = 4.0f;
            raw_dt /= erle_for_dt;
        }
        /* EPC physical gate. */
        if (a->epc.active) { raw_dt = 0.0f; is_stationary_dt = 0; }
        dt_indicator = raw_dt; if (dt_indicator < 0.0f) dt_indicator = 0.0f;
        if (dt_indicator > 0.8f) dt_indicator = 0.8f;

        /* divergence indicator EMA. */
        filter_convergence_update_divergence(&a->convergence, a->near_power, a->raw_error_power);

        /* EPC render-forced countdown (Python decrements + sets RES flag;
         * the RES flag is consumed inside REE which aec3_post_run drives —
         * the orchestrator no longer wires using_render_based to REE, so this
         * is a counter decrement only). */
        if (a->epc_render_forced_remaining > 0) a->epc_render_forced_remaining--;

        /* shadow_dt stash for RES context. Both operands received from
         * DoubleTalk (Stage-3, double) — cast at the receive/compare site. */
        float shadow_dt_v = a->dt_analyzer.dt_from_energy;
        if (a->dt_analyzer.dt_from_shadow > shadow_dt_v) shadow_dt_v = a->dt_analyzer.dt_from_shadow;
        if (a->epc.active) shadow_dt_v *= 0.08f;
        a->last_far_power = far_power;
        a->last_shadow_dt = shadow_dt_v;
        a->last_is_stationary_dt = is_stationary_dt;
        a->last_dt_indicator = dt_indicator;

        /* ── aec3_post_run → final_output ── */
        {
            /* filter_taps_full is materialized on demand by
             * aec_fill_filter_taps(), driven from inside the FilterAnalyzer
             * once it knows which region it is about to read -- not
             * whole-filter up front here, which cost an inverse FFT per
             * partition every hop to hand the analyzer one region's worth of
             * taps. The buffer and its length are still passed exactly as
             * before; only the partitions covering the analyzer's current
             * region are refreshed inside the call below. */

            Aec3PostRunIn in;
            memset(&in, 0, sizeof(in));
            in.near_spec = a->main_filter.base.near_spec;
            in.far_spec  = a->main_filter.base.far_spec;
            in.echo_spec = a->main_filter.base.echo_spec;
            in.error_spec_windowed = a->main_filter.base.error_spec_windowed;
            in.W0 = a->main_filter.base.W;             /* W[0] */
            in.W_all = a->main_filter.base.W;          /* live, read-only for the call */
            in.X_buf = a->main_filter.base.X_buf;      /* live, read-only for the call */
            in.sqrt_hann = a->main_filter.base.sqrt_hann;
            in.kalman_P = NULL; in.kalman_P_len = 0;   /* divergence_indicator dead */
            in.partition_idx = a->main_filter.base.partition_idx;
            in.n_partitions = N;
            in.filter_taps_full = a->filter_taps_full;
            in.filter_taps_full_len = N * hop;
            in.shadow_present = a->has_shadow;
            in.shadow_error_spec = a->has_shadow ? a->shadow_filter.error_spec : NULL;
            in.last_shadow_output_time = a->has_shadow ? a->shadow_out : NULL;
            in.s_ref_max = a->main_filter.base.last_s_max_abs * 32768.0f;
            in.s_coa_max = a->has_shadow ? a->shadow_filter.last_s_max_abs * 32768.0f : 0.0f;
            /* in.main_error_energy / in.shadow_error_energy intentionally left
             * at their memset(0) default: Aec3PostRunIn.main_error_energy /
             * shadow_error_energy have zero readers anywhere in aec3_post.c
             * (the FilterStateBridge call that used to consume them was
             * removed) -- recomputing them here was a pure-waste O(n_freqs)
             * cmag2 + pairwise-sum pass feeding an unread field. The real,
             * still-needed computation (feeding a->main_err_smooth /
             * a->shadow_err_smooth for double-talk detection) stays at step 13
             * above; error_spec is not mutated between there and here in the
             * same hop, so this was provably a redundant recompute, not a
             * distinct value. */
            in.raw_output = a->raw_output;
            in.near_end = a->near_hop;
            in.far_end = a->far_hop;
            in.current_delay = a->current_delay;
            /* Mirrors Python's `self._delay_active` (orchestrator ~3150),
             * which is "a reference ring exists", NOT "an estimator exists"
             * -- so FIXED must report 1 here or the AEC3 post chain would
             * see no external delay despite one being applied every hop.
             * MATCHED (1) and EXTERNAL_ALIGNED (0) are unchanged from the
             * previous `a->has_delay`, so this is byte-exact for both. */
            in.delay_active = aec_ring_active(&a->cfg);
            in.saturation_level = a->saturation_level;
            in.pending_gain_change = a->pending_gain_change;
            in.pending_delay_change = a->pending_delay_change;
            in.stationarity_active_hops = a->stationarity_active_hops;
            in.stationarity_converge_hops = a->stationarity_converge_hops;
            in.stationary_block = stationarity_block_for_post;
            in.erle_coh_gate_enabled = AEC3B_ERLE_COH_GATE_ENABLED;
            in.use_stationarity_properties = AEC3B_USE_STATIONARITY_PROPERTIES;
            /* In the external AEC→NR/RES seam, Step 19's comfort-noise PSD
             * and Step 20's R² are consumed by the caller (Step 20's gain
             * itself is consumed only when spatial_linear_context is unset --
             * see below), but Step 21's private gain/CNG/IFFT/OLA result is
             * immediately replaced by the linear residual. Skip only that
             * final synthesis stage. */
            in.context_only = (!a->cfg.enable_res && a->cfg.return_res_context);
            /* spatial_linear_context requires context_only (enforced by
             * aec_validate_config()): a caller that sets it has already
             * committed to never reading res_gain (aec_get_res_context()
             * exposes NULL for it in this mode), so Step 20 can skip
             * computing the array Step 21 would otherwise never read here
             * either way. */
            in.spatial_linear_context = a->cfg.spatial_linear_context;
            in.active_render_threshold = AEC3B_ACTIVE_RENDER_THRESHOLD;

            Aec3PostRunObj obj;
            obj.state = &a->a3_state; obj.ree = &a->a3_ree; obj.sg = &a->a3_sg;
            obj.stationarity = &a->a3_stat; obj.lfs = &a->a3_lfs; obj.fft = a->post_fft;

            /* DT-gated min-gain floor lift (mirrors Python 16285fd): during
             * double-talk (near recently present) protect near-end by lifting
             * the RES floor; FS (no near) keeps the aggressive far_active floor.
             * Set on the SG just before get_gain (driven inside aec3_post_run). */
            a->a3_sg.dt_protect_active =
                (a->cfg.dt_aware_res_floor_enabled && a->ne_recent_frames > 0) ? 1 : 0;

            int pgc = a->pending_gain_change, pdc = a->pending_delay_change;
            aec3_post_run(&a->post, &in, &obj, &a->a3_sc, a->final_out, &pgc, &pdc);
            a->pending_gain_change = pgc;
            a->pending_delay_change = pdc;

            /* In context-only mode aec3_post_run emits raw_output directly;
             * the external NR/RES remains the sole suppressor. */
        }

        /* per-bin mu_scale update AFTER RES (orchestrator 2291-2307). */
        if (a->convergence.converged) {
            float mu_min = a->cfg.shadow_mu_min;
            for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
            a->simple_mu_ratio = 0.0f;
            if (is_stationary_dt) {
                for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
                a->simple_mu_ratio = mu_min;
            }
            a->has_per_bin_mu = 1;
        } else {
            a->has_per_bin_mu = 0;
            update_simple_mu_ratio(a, a->raw_output, a->far_hop, hop);
        }
    }
    /* enable_res==False C-parity fallback (orchestrator 2313-2316) is unused
     * here since balanced always has enable_res=True. */

    /* 19. power EMAs (sample loop; read by NEXT frame's step 17). */
    {
        const float ap = a->alpha_pow, oap = 1.0f - a->alpha_pow;
        for (int i = 0; i < hop; ++i) {
            float n = a->near_hop[i], r = a->raw_output[i];
            a->near_power      = ap * a->near_power      + oap * (n * n);
            a->raw_error_power = ap * a->raw_error_power + oap * (r * r);
        }
    }

    /* 20. convergence detection. */
    {
        int far_active = (far_hop_mean_sq > 1e-4f);
        int just_converged = filter_convergence_update_convergence(
            &a->convergence, a->near_power, a->raw_error_power,
            far_active, a->warmup_frames_remaining <= 0);
        if (just_converged) {
            int Kk = a->main_filter.base.n_freqs;
            for (int k = 0; k < Kk; ++k) a->main_filter.Q[k] = a->main_filter.Q_low[k];
            /* shadow is PBFDAF (no Q). */
        }
    }

    /* cache erle_windowed for next frame's Path-A guard and the duty
     * watchdog. Condition mirrors the compute block above (and Python
     * orchestrator's post sub-block): enable_res OR return_res_context.
     * The old enable_res-only gate was a port divergence -- in the
     * context-only seam config Python's already-cancelling guard and ERLE
     * watchdog stayed live while C's went dead (see CHANGELOG). */
    if (a->cfg.enable_res || a->cfg.return_res_context)
        a->last_erle_windowed = erle_windowed;

    a->frame_count++;
}

/* Process exactly hop_size samples — OFFLINE / lockstep path. Byte-exact to
 * Python aec.py. Render and capture supplied together. Thin wrapper over
 * aec_process_core(): the only thing it adds is step 21 (emit into the
 * caller's `out` buffer), which aec_process_context() below does not want.
 * There is no output limiter: `final_out` is delivered exactly as the linear
 * filter + AEC3 post chain produced it. */
void aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    const int hop = a->hop_size;
    aec_process_core(a, mic_in, ref_in, NULL);

    /* ── per-frame structured trace ("logr"). Audio-passive, read-only: only
     *    runs when --debug-trace set a CSV file. Zero hot-path cost otherwise
     *    (single NULL test). Reads the post-filter internals — the three not
     *    otherwise persisted (aec3_converged / far_active / gain_mean) were
     *    stashed on a->post.trace during aec3_post_run; everything else is
     *    re-read here from the AEC3 sub-module accessors.
     *
     * Whole block compiled out under AEC_NO_STDIO:
     * aec_debug_trace_active()/aec_debug_trace_row() are runtime-gated only
     * (a single NULL-FILE* test), which still pulls in aec_debug.o's
     * fprintf/vfprintf/stderr references for board/no-stdio builds even
     * though the trace is never armed there. NDEBUG alone never stripped
     * this — it is release/debug orthogonal (unlike AEC_DEBUG_LOG, this
     * trace runs in ordinary release builds whenever --debug-trace is set).
     * AEC_NO_STDIO removes the call sites entirely so the library carries
     * no stdio reference regardless of runtime state; see aec_debug.h/.c
     * and the Makefile's NO_STDIO switch.
     *
     * Only reachable via aec_process(): aec_process_context() never runs
     * this. That is not a new limitation -- the a->cfg.enable_res guard
     * below already meant this never fired for a context_only instance
     * (enable_res=0 by definition), so no observable behavior changes for
     * any existing caller. */
#ifndef AEC_NO_STDIO
    if (a->cfg.enable_res && aec_debug_trace_active()) {
        int Kk = a->n_freqs;
        AecDebugTraceRow tr;
        float esum = 0.0f, r2sum = 0.0f, cnsum = 0.0f;
        const float *erle = aec_state_erle(&a->a3_state, /*onset=*/1);
        int kk;
        for (kk = 0; kk < Kk; ++kk) {
            esum  += erle[kk];
            r2sum += a->a3_sc.r2[kk];
            cnsum += a->post.comfort_noise[kk];
        }
        tr.delay            = aec_state_min_direct_path_filter_delay(&a->a3_state);
        tr.far_active       = a->post.trace.far_active;
        tr.saturated_echo   = aec_state_saturated_echo(&a->a3_state);
        tr.usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
        tr.dominant_nearend = suppression_gain_is_dominant_nearend(&a->a3_sg);
        tr.filter_converged = a->post.trace.aec3_converged;
        tr.fullband_erle    = aec_state_fullband_erle_log2(&a->a3_state);
        tr.erle_mean        = (Kk > 0) ? esum  / Kk : 0.0f;
        tr.r2_mean          = (Kk > 0) ? r2sum / Kk : 0.0f;
        tr.gain_mean        = a->post.trace.gain_mean;
        tr.comfort_noise_mean = (Kk > 0) ? cnsum / Kk : 0.0f;
        tr.near_pwr         = a->near_power;
        tr.raw_err_pwr      = a->raw_error_power;
        aec_debug_trace_row(&tr);
    }
#endif /* AEC_NO_STDIO */

    /* 21. emit. */
    memcpy(out, a->final_out, (size_t)hop * sizeof(float));
}

/* Context-only entry point: runs aec_process_core() only -- it just skips
 * the final copy into an `out` buffer. For callers that only read
 * aec_get_res_context() (error_spec / res_gain / formed_output / etc via
 * AecResContext) and never touch aec_process()'s own returned audio, e.g.
 * a pipeline running enable_res=0 && return_res_context=1 (context_only)
 * purely for the linear filter's context -- the final copy in aec_process()
 * is pure waste for them.
 *
 * No mixing restriction: aec_process(), aec_process_capture(),
 * aec_process_context() and aec_process_context_shared_far() all advance
 * exactly the same state via aec_process_core(), and differ only by whether
 * they copy the result out. They may be interleaved freely on one instance,
 * in any order, without a reset in between. */
void aec_process_context(Aec* a, const float* mic_in, const float* ref_in) {
    aec_process_core(a, mic_in, ref_in, NULL);
}

/* Far-end-FFT-sharing entry point: like aec_process_context(), but shared_far_spec (non-NULL, length
 * n_freqs) lets this instance skip computing its OWN far-end FFT and borrow
 * one an external caller already computed instead -- for a multi-instance
 * caller (e.g. a 4-lane wrapper) whose lanes all see the IDENTICAL far-end
 * signal every hop: one lane (via plain aec_process_context(), which still
 * computes its own far_spec) computes the real FFT once, the rest borrow it
 * through this entry point instead of each redundantly recomputing an
 * identical transform. Read the lane-0 spectrum to share via
 * aec_get_res_context()'s far_spec field (unconditionally populated,
 * independent of enable_res/return_res_context).
 *
 * PRECONDITION (caller's responsibility): shared_far_spec must be the
 * value THIS hop's aec_get_res_context() on the computing instance would
 * return, i.e. computed from the exact same far-end time-domain signal
 * this call's own ref_in carries -- ref_in is still required (pbfdaf_frontend()
 * unconditionally updates far_buffer's OLA history from it, and every
 * non-FFT use of the raw far signal elsewhere in aec_process_core() --
 * saturation detection, delay estimation, mu_scale -- is unaffected by
 * this sharing). A mismatched or stale spectrum silently produces a wrong
 * (not crashing) linear filter result -- see 4aec_nr_res.c's caller for how
 * the 4-lane wrapper keeps this invariant (identical p->aligned_ref handed
 * to every lane, all lanes reset together on any delay change). */
void aec_process_context_shared_far(
        Aec* a, const float* mic_in, const float* ref_in,
        const Complex* shared_far_spec) {
    aec_process_core(a, mic_in, ref_in, shared_far_spec);
}

/* ── Streaming API ────────────────────────────────────────────────────────
 * Render-hop FIFO wrapper over the bit-exact aec_process() engine. In lockstep
 * (one analyze_render then one process_capture, same thread) the FIFO is
 * pass-through (fifo_write one hop ahead of fifo_read, no event) so the engine
 * sees exactly the same (mic,ref) it would via aec_process() → byte-identical
 * output. Only real async jitter exercises the underrun/overrun paths.
 *
 * F09 Variant A' (drop-new + consumer catch-up): a from-scratch SPSC ring.
 * Render and capture are documented,
 * aec.h, as callable from two different threads — SPSC — and the FIFO
 * bookkeeping was plain, non-atomic `int`/`long` read-modify-write, a data
 * race the instant that documented concurrency was actually used) routed
 * every cross-thread touch of fifo_read/fifo_count through `__atomic_*_n`
 * builtins, but kept the ORIGINAL single-thread overrun semantics: the
 * *producer*, on overrun, advanced fifo_read to drop the OLDEST buffered hop.
 * That gives fifo_read two writers (producer's overrun path, consumer's
 * normal claim), which is why that version needed a full `fetch_add` RMW on
 * both sides instead of the textbook "consumer owns the read cursor,
 * producer owns the write cursor" SPSC shape.
 *
 * This rewrite removes that second writer instead of arbitrating it: on
 * overrun the producer now drops the NEW incoming hop (drop-new) rather than
 * the oldest buffered one, and a symmetric case is added on the consumer side
 * — if the consumer finds the ring completely full (it has fallen a full
 * `cap` hops behind), it catches up by skipping straight to the freshest
 * buffered hop instead of grinding through a `cap`-hop backlog of stale
 * audio. Both responses still report AEC_BUF_RENDER_OVERRUN; only which
 * hop is sacrificed changes. With that, fifo_write is written ONLY by the
 * render thread and fifo_read is written ONLY by the capture thread — each
 * a plain monotonic unsigned cursor, own-thread-loaded as a plain read and
 * cross-thread-published with a single acquire/release pair, no RMW needed
 * anywhere. See the full per-field protocol + invariant proof in aec.h's
 * struct comment above the FIFO fields. Builtins (not `<stdatomic.h>`
 * _Atomic fields) throughout, so the struct's layout/ABI and C89
 * includability are both untouched. */
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref) {
    const int hop = a->hop_size;
    const unsigned cap = (unsigned)a->fifo_cap_hops;
    a->render_call_count++;                      /* producer-private: plain */

    /* fifo_write is this thread's own field (sole writer) -- a plain load
     * cannot race. fifo_read is the consumer's field -- ACQUIRE so this
     * thread observes the consumer's most recent published claim before
     * deciding whether the ring has room. */
    unsigned w = *(unsigned*)&a->fifo_write;     /* own field: plain load   */
    unsigned r = __atomic_load_n((unsigned*)&a->fifo_read, __ATOMIC_ACQUIRE);
    if (w - r >= cap)
        return AEC_BUF_RENDER_OVERRUN;           /* full: drop-new, zero ring bytes touched */

    memcpy(a->render_fifo + (size_t)(w % cap) * (size_t)hop, ref, (size_t)hop * sizeof(float));
    /* Publish AFTER the memcpy: this release pairs with the consumer's
     * acquire-load of fifo_write, so the payload written above is
     * guaranteed visible before any consumer can observe this hop as
     * available (see aec.h's invariant proof for why w-r never exceeds cap
     * at the moment this store fires). */
    __atomic_store_n((unsigned*)&a->fifo_write, w + 1u, __ATOMIC_RELEASE);
    return AEC_BUF_NONE;
}

AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out) {
    const int hop = a->hop_size;
    const unsigned cap = (unsigned)a->fifo_cap_hops;
    a->capture_call_count++;   /* single-writer (capture thread only): plain */
    AecBufferingEvent ev;
    const float* ref;
    int claimed;

    /* fifo_read is this thread's own field (sole writer) -- a plain load
     * cannot race. fifo_write is the producer's field -- ACQUIRE so this
     * thread observes the producer's most recent published hop (and, by the
     * release-before-store pairing in aec_analyze_render(), that hop's
     * payload memcpy) before deciding what to consume. */
    unsigned r = *(unsigned*)&a->fifo_read;      /* own field: plain load   */
    unsigned w = __atomic_load_n((unsigned*)&a->fifo_write, __ATOMIC_ACQUIRE);

    if (w == r) {
        /* Underrun: nothing buffered. Process with the immutable all-zero
         * hop (fifo_zero_ref) -- no echo to cancel this hop -- and signal
         * the caller. fifo_read is not advanced (nothing was claimed), so
         * this ring slot bookkeeping is untouched by this branch. */
        ref = a->fifo_zero_ref;
        ev = AEC_BUF_RENDER_UNDERRUN;
        claimed = 0;
    } else if (w - r == cap) {
        /* Consumer catch-up: the ring is completely full (we have fallen a
         * full `cap` hops behind the producer). Skip the entire stale
         * backlog and jump straight to the freshest buffered hop (index
         * w-1) instead of draining it oldest-first. This is still a
         * consumer-side-only advance of fifo_read (the producer never
         * writes this field in this design) and is monotonic: w-1 =
         * r + cap - 1 > r for any cap >= 2 (aec_derive_dims enforces
         * cap >= 2), so the read cursor strictly increases. */
        r = w - 1u;
        ref = a->render_fifo + (size_t)(r % cap) * (size_t)hop;
        ev = AEC_BUF_RENDER_OVERRUN;
        claimed = 1;
    } else {
        /* Normal case: consume the oldest buffered hop. */
        ref = a->render_fifo + (size_t)(r % cap) * (size_t)hop;
        claimed = 1;
        ev = AEC_BUF_NONE;
    }

    aec_process(a, mic, ref, out);
    /* Publish the advanced read cursor only once aec_process() has
     * returned -- its very first act (aec.c, top of aec_process()) is to
     * memcpy `ref` into a->far_hop, so by the time we get here the ring
     * slot (when claimed==1) has definitely been read out. This is later
     * than strictly necessary (the slot is actually free the instant that
     * first memcpy completes, not after the whole hop's DSP work) but it
     * needs no change inside aec_process() itself, and is still
     * race-free: the producer cannot treat this slot as free until it
     * observes this release store of fifo_read. */
    if (claimed)
        __atomic_store_n((unsigned*)&a->fifo_read, r + 1u, __ATOMIC_RELEASE);
    /* last_buffering_event: written only by this (capture) thread, so the
     * store itself needs no atomics for correctness on this side; relaxed
     * atomic is used purely so a third thread calling
     * aec_last_buffering_event() never sees a torn/undefined read (see
     * aec.h). */
    __atomic_store_n(&a->last_buffering_event, (int)ev, __ATOMIC_RELAXED);
    return ev;
}

AecBufferingEvent aec_last_buffering_event(const Aec* a) {
    return (AecBufferingEvent)__atomic_load_n(&a->last_buffering_event, __ATOMIC_RELAXED);
}

void aec_get_res_context(const Aec* a, AecResContext* ctx) {
    if (!a || !ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->n_freqs = a->n_freqs;
    ctx->hop_size = a->hop_size;
    ctx->linear_hop = a->raw_output;
    ctx->far_spec   = a->main_filter.base.far_spec;
    /* Freq-domain seam (valid when the AEC3 post block ran this hop, i.e.
     * enable_res || return_res_context). These alias the internal per-hop
     * buffers populated by aec3_post_run.  sel_esw is the exact 50%-overlap
     * sqrt-Hann STFT of the formed linear output; sel_echo is its matching
     * echo estimate and ybase = sel_esw + sel_echo is the matching windowed
     * capture spectrum.  Do not expose main_filter.error_spec_windowed here:
     * that is a PBFDKF estimator quantity, not a reconstructing WOLA frame.
     * res_gain is the SuppressionGain output, r2 the residual-echo PSD
     * (int16²), and comfort_noise the CNG N² (int16²). Left NULL when
     * neither RES nor the context seam is enabled -- also left NULL under
     * spatial_linear_context, where a->a3_sg.gain is never written past its
     * zero-init (get_gain() itself never runs), so exposing it would misread
     * as "G_res == 0 everywhere" (full suppression) rather than "not
     * computed". */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        ctx->error_spec    = a->a3_sc.sel_esw;
        ctx->echo_spec     = a->a3_sc.sel_echo;
        ctx->near_spec     = a->a3_sc.ybase;
        ctx->formed_hop    = a->a3_lfs.e_form;
        ctx->res_gain      = a->cfg.spatial_linear_context ? NULL : a->a3_sg.gain;
        ctx->r2            = a->a3_sc.r2;
        ctx->comfort_noise = a->post.comfort_noise;
    } else {
        /* Diagnostic-only raw PBFDKF spectra when the post block did not run.
         * No reconstructing error spectrum is available in this mode. */
        ctx->echo_spec = a->main_filter.base.echo_spec;
        ctx->near_spec = a->main_filter.base.near_spec;
    }
    ctx->far_power = a->last_far_power;
    ctx->erle_factor = a->erle_factor_prev;
    ctx->dt_indicator = a->last_dt_indicator;
    ctx->divergence = a->convergence.divergence;
    ctx->saturation_level = a->saturation_level;
    ctx->erl_estimate = a->erl_estimate;
    ctx->shadow_dt = a->last_shadow_dt;
    ctx->is_stationary_dt = a->last_is_stationary_dt;
    ctx->filter_converged = a->convergence.converged;
    ctx->filter_once_converged = a->convergence.once_converged;
    ctx->epc_active = a->epc.active;
}

/* How much to trust the alignment currently in force, in one place, for
 * every reporting surface (AecDebugStatus and AecLinearContext both read
 * it, so a consumer polling the two can never see them disagree).
 *
 * MATCHED is the only mode with an estimator, so it is the only mode whose
 * confidence is a MEASUREMENT (delay_aec3_confidence's 0.0/0.5/1.0 ladder).
 * FIXED and EXTERNAL_ALIGNED both report 1.0 for the same reason: the
 * alignment is a CALLER CONTRACT, not something this library estimated --
 * FIXED applies a delay the caller measured at bring-up, EXTERNAL_ALIGNED
 * applies none because the caller guarantees `ref` arrives aligned. Neither
 * may read the never-constructed DelayAec3, which would report 0.0 --
 * "estimator present but unconverged" -- for an alignment that carries no
 * uncertainty at all. */
static float aec_delay_confidence(const Aec* a) {
    if (a->cfg.delay_mode == AEC_DELAY_MATCHED)
        return delay_aec3_confidence(&a->delay);
    return 1.0f;
}

/* Read-only status query — see AecDebugStatus (aec.h). No state mutated, no
 * per-frame cost added: every field read here is already maintained by the
 * engine on the hot path regardless of whether this is ever called. */
void aec_debug_status(const Aec* a, AecDebugStatus* out) {
    if (!a || !out) return;
    memset(out, 0, sizeof(*out));

    out->delay_samples    = a->current_delay;
    out->delay_confidence = aec_delay_confidence(a);
    out->delay_updates    = a->has_delay ? delay_aec3_n_updates(&a->delay) : 0;

    out->erle_windowed_db = a->last_erle_windowed;
    out->usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
    out->filter_converged = a->convergence.converged;

    out->near_power = a->near_power;
    out->out_power  = a->raw_error_power;

    out->duty_hops_total = a->duty_hops_total;
    out->duty_hops_run   = a->duty_hops_run;
}

/* Read-only linear-AEC seam view — see AecLinearContext (aec.h). Pointer
 * fields alias per-hop internals; valid until the next process/reset. */
void aec_get_linear_context(const Aec* a, AecLinearContext* ctx) {
    if (!a || !ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->hop_size = a->hop_size;
    ctx->formed_linear_hop = (a->cfg.enable_res || a->cfg.return_res_context)
                             ? a->a3_lfs.e_form : a->raw_output;
    ctx->aligned_far_hop = a->far_hop;
    ctx->generation = a->delay_generation;
    /* Per-mode semantics -- see the AecLinearContext contract in aec.h.
     * Confidence is NOT one of the per-mode branches: it is the same
     * question aec_debug_status() answers, so both read the one helper and
     * only delay_samples/delay_state branch below. */
    ctx->delay_confidence = aec_delay_confidence(a);
    if (a->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        /* far IS the caller's own, already-aligned hop. */
        ctx->delay_samples = 0;
        ctx->delay_state = AEC_LINEAR_DELAY_LOCKED;
        return;
    }
    if (a->cfg.delay_mode == AEC_DELAY_FIXED) {
        ctx->delay_samples = a->current_delay;   /* == cfg.fixed_delay_samples */
        /* far_hop_aligned is still consulted: for the first
         * ceil(fixed/hop) hops the ring cannot yet serve the offset, so
         * far_hop is RAW and claiming LOCKED there would break this seam's
         * central promise. CHANGED is unreachable (nothing bumps generation
         * during processing without an estimator). */
        ctx->delay_state = a->far_hop_aligned ? AEC_LINEAR_DELAY_LOCKED
                                              : AEC_LINEAR_DELAY_UNLOCKED;
        return;
    }
    ctx->delay_samples = a->current_delay;
    if (a->current_delay < 0 || !a->far_hop_aligned)
        ctx->delay_state = AEC_LINEAR_DELAY_UNLOCKED;
    else if (a->delay_generation != a->delay_gen_hop_start)
        ctx->delay_state = AEC_LINEAR_DELAY_CHANGED;
    else
        ctx->delay_state = AEC_LINEAR_DELAY_LOCKED;
}
