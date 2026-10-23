/* Top-level PBFDKF AEC orchestration with the post-filter chain. */
#include "aec.h"
#include "aec_config_build.h"
#include "aec_debug.h"
#include "aec_balanced_config.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"
#include "utility/mem_align.h"   /* ALIGN16 */
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DelayAdjustment value stored in pending_delay_change. */
#define AEC_DA_NEW_DETECTED  2

/* Streaming FIFO budget (ms), converted to whole hops at create time. */
#define AEC_STREAM_FIFO_MS 320

/* ───────────────────────── config ──────────────────────────────────────── */

/* Fields authored on the 16 kHz / 10 ms reference grid are converted to the
 * resolved grid in aec_carve(): hop counts via ms_to_hops, retention alphas
 * by power law, additive per-hop rates linearly. */
#define AEC_LEGACY10MS_HOP  160
#define AEC_LEGACY10MS_SR   16000
static int aec_legacy10ms_hops(float legacy_ms, int hop, int sample_rate) {
    return (hop > 0 && sample_rate > 0)
        ? aec_ms_to_hops(legacy_ms, hop, sample_rate)
        : (int)lrintf(legacy_ms / 10.0f);
}
static float aec_legacy10ms_alpha(float legacy_alpha, int hop, int sample_rate) {
    return (hop > 0 && sample_rate > 0)
        ? aec_growth_rehop(legacy_alpha, AEC_LEGACY10MS_HOP, AEC_LEGACY10MS_SR,
                           hop, sample_rate)
        : legacy_alpha;
}
/* Additive per-hop amount (a leak, a step): scales linearly with the hop
 * period, unlike a retention alpha. */
static float aec_legacy10ms_rate(float legacy_per_hop, int hop,
                                 int sample_rate) {
    const float ref_s = (float)AEC_LEGACY10MS_HOP / (float)AEC_LEGACY10MS_SR;
    return (hop > 0 && sample_rate > 0)
        ? legacy_per_hop * (((float)hop / (float)sample_rate) / ref_s)
        : legacy_per_hop;
}

void aec_config_defaults(AecConfig* cfg, int sr) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    /* Only place a default fft_size is applied (0 for an unsupported rate,
     * which the validator rejects). */
    cfg->fft_size = aec_default_fft_size(sr);
    /* filter_length: 64 ms at sr >= 44100, 52 ms otherwise (0 for sr <= 0). */
    cfg->filter_length = (sr <= 0) ? 0
                        : (sr >= 44100) ? (sr * 64 / 1000)
                                        : (sr * 52 / 1000);
    cfg->n_partitions = 0;
    cfg->mu = 0.3f;
    cfg->delta = 1e-8f;
    cfg->enable_cng = 1;
    /* MATCHED with 5 filters is the default; fixed_delay_samples must be -1
     * (0 is a legal FIXED delay). */
    cfg->delay_mode = AEC_DELAY_MATCHED;
    cfg->fixed_delay_samples = -1;
    cfg->enable_delay_est = 1;
    cfg->enable_highpass = 1;
    cfg->enable_saturation = 1;
    cfg->enable_shadow = 1;
    cfg->enable_res = 1;
    cfg->saturation_softclip_ref = 1;
    /* Frame-count fields are 16 kHz / 10 ms reference-grid values, retimed in
     * aec_carve(). */
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
    cfg->delay_est_period_s = 0.0f;
    cfg->delay_num_filters = DA_NUM_FILTERS;
    cfg->highpass_cutoff_hz = 80.0f;
    cfg->saturation_threshold = 0.95f;
    cfg->kalman_q_high = 1e-3f;
    cfg->kalman_q_low = 1e-6f;
    cfg->delay_acquire_protect_converged = 1;
    cfg->delay_acquire_warm_transfer = 1;
    cfg->delay_acquire_inst_erle_db = 4.0f;
    cfg->dt_aware_recovery_soft = 1;
    cfg->dt_aware_res_floor_enabled = 1;
    /* -20 dB: recovers residual-echo suppression when the near-recent latch
     * false-arms in far-end single talk, at a small double-talk cost. */
    cfg->min_gain_floor_dt_db = -20.0f;
    cfg->ne_recent_threshold = 0.3f;
    cfg->ne_recent_hold = 150;
    cfg->ne_recent_sustain = 3;
    cfg->min_gain_floor_far_active_db = -28.0f;
    cfg->filter_misadjustment_stable_frames = 30;
    cfg->filter_misadjustment_hangover_frames = 100;
    cfg->filter_misadjustment_scale_min = 0.5f;
    cfg->filter_misadjustment_scale_max = 2.0f;
    cfg->live_observer_enabled = 0;
    cfg->live_observer_shift_ms = 24.0f;
    cfg->live_observer_max_hz = 1000.0f;
    cfg->live_observer_mu = 0.3f;
    cfg->live_observer_corrob_ms = 100.0f;
    cfg->live_observer_corrob_margin_db = 0.0f;
    /* Default on, so a config that skips aec_config_from_preset keeps the split floor. */
    cfg->min_gain_split_floor_enabled = 1;
}

/* Preset strength table in AecPreset order; NO_FLOOR repeats balanced's
 * (unused) values. */
typedef struct { float far_active_db, dt_db; int split_floor_enabled; } AecPresetFloors;
static const AecPresetFloors AEC_PRESET_FLOORS[] = {
    [AEC_PRESET_MILD]       = { -20.0f, -20.0f, 1 },
    [AEC_PRESET_BALANCED]   = { -28.0f, -20.0f, 1 },
    [AEC_PRESET_AGGRESSIVE] = { -38.0f, -30.0f, 1 },
    [AEC_PRESET_NO_FLOOR]   = { -28.0f, -20.0f, 0 },
};

/* Row for an in-enum preset, NULL otherwise (every lookup below then fails). */
static const AecPresetFloors* aec_preset_row(AecPreset p) {
    return ((unsigned)p < sizeof(AEC_PRESET_FLOORS) / sizeof(AEC_PRESET_FLOORS[0]))
               ? &AEC_PRESET_FLOORS[p] : NULL;
}

int aec_preset_floor_db(AecPreset p, float* db_out) {
    const AecPresetFloors* r = aec_preset_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->far_active_db;
    return 0;
}

int aec_preset_dt_floor_db(AecPreset p, float* db_out) {
    const AecPresetFloors* r = aec_preset_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->dt_db;
    return 0;
}

int aec_preset_split_floor_enabled(AecPreset p, int* enabled_out) {
    const AecPresetFloors* r = aec_preset_row(p);
    if (enabled_out == NULL || r == NULL) return -1;
    *enabled_out = r->split_floor_enabled;
    return 0;
}

void aec_config_from_preset(AecConfig* cfg, AecPreset p, int sr) {
    float db, dt_db;
    int en;
    aec_config_defaults(cfg, sr);
    cfg->min_gain_floor_far_active_db =
        (aec_preset_floor_db(p, &db) == 0) ? db : -28.0f;
    cfg->min_gain_floor_dt_db =
        (aec_preset_dt_floor_db(p, &dt_db) == 0) ? dt_db : -20.0f;
    cfg->min_gain_split_floor_enabled =
        (aec_preset_split_floor_enabled(p, &en) == 0) ? en : 1;
}

/* Signal-grid table: admissible (sample_rate, fft_size) pairs and their
 * geometry. */
static const AecSignalGrid AEC_GRID_TABLE[] = {
    { 16000,  256,   256,  128,     129, 0 },
    { 16000,  512,   512,  256,     257, 0 },
    { 48000, 1024,  1024,  512,     513, 0 },
    {  8000,  256,   256,  128,     129, 1 },
};

int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out) {
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
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate)
            return AEC_GRID_TABLE[i].fft_size;
    }
    return 0;
}

/* Delay-mode translation: the only reader of the deprecated enable_delay_est;
 * everything else reads cfg->delay_mode. */
int aec_config_resolve_delay(AecConfig* cfg) {
    if (!cfg) return -1;
    if (cfg->delay_mode != AEC_DELAY_MATCHED &&
        cfg->delay_mode != AEC_DELAY_FIXED &&
        cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED) return -1;
    if (!cfg->enable_delay_est && cfg->delay_mode == AEC_DELAY_MATCHED) {
        /* Flag cleared: fixed_delay_samples >= 0 means a measured delay, else
         * pre-aligned far. */
        cfg->delay_mode = (cfg->fixed_delay_samples >= 0)
                          ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    }
    cfg->enable_delay_est = (cfg->delay_mode == AEC_DELAY_MATCHED) ? 1 : 0;
    return 0;
}

/* Resolves the caller's config once so size query and carve agree; cfg_in is
 * not mutated. */
static AecConfig aec_resolved_config(const AecConfig* cfg_in) {
    AecConfig cfg = *cfg_in;
    (void)aec_config_resolve_delay(&cfg);
    return cfg;
}

/* True when a reference ring exists (MATCHED, FIXED); Aec::has_delay is the
 * MATCHED-only estimator. */
static inline int aec_ring_active(const AecConfig* cfg) {
    return cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED;
}

/* Bounds-checks an AecConfig (finite, generous ranges around each default)
 * before any derived size or state is touched. Failure follows each caller's
 * convention (aec_create nonzero, aec_get_mem_size 0, aec_init NULL). */
static int aec_validate_config(const AecConfig* cfg_in) {
    if (!cfg_in) return 0;
    /* Checked on the caller's own value, before resolve_delay rewrites it. */
    if (cfg_in->enable_delay_est != 0 && cfg_in->enable_delay_est != 1) return 0;
    AecConfig resolved = *cfg_in;
    if (aec_config_resolve_delay(&resolved) != 0) return 0;
    const AecConfig* cfg = &resolved;

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

    AEC_CK_BOOL(enable_cng);
    AEC_CK_BOOL(cng_n2_windowed_psd);
    AEC_CK_BOOL(enable_highpass);
    AEC_CK_BOOL(enable_saturation);
    AEC_CK_BOOL(enable_shadow);
    AEC_CK_BOOL(enable_res);
    AEC_CK_BOOL(saturation_softclip_ref);
    AEC_CK_BOOL(return_res_context);
    AEC_CK_RANGE_I(spatial_linear_context, 0, AEC_CONTEXT_LINEAR_ONLY);
    AEC_CK_BOOL(delay_acquire_protect_converged);
    AEC_CK_BOOL(delay_acquire_warm_transfer);
    AEC_CK_BOOL(dt_aware_recovery_soft);
    AEC_CK_BOOL(dt_aware_res_floor_enabled);
    /* Levels 1 and 2 require context-only operation (enable_res == 0 and
     * return_res_context == 1). */
    if (cfg->spatial_linear_context &&
        (cfg->enable_res || !cfg->return_res_context)) return 0;

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
    /* With enable_highpass the cutoff must satisfy hpf_params_valid(): 0 <
     * cutoff < 0.45 * sample_rate. */
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

    AEC_CK_RANGE_I(warmup_frames,                       0, 1000000);
    AEC_CK_RANGE_I(epc_hangover,                         0, 1000000);
    AEC_CK_RANGE_I(ne_recent_hold,                       0, 1000000);
    AEC_CK_RANGE_I(ne_recent_sustain,                    0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_stable_frames,   0, 1000000);
    AEC_CK_RANGE_I(filter_misadjustment_hangover_frames, 0, 1000000);
    /* Hard bound: DA_NUM_FILTERS is the static array extent; 0 filters is invalid. */
    AEC_CK_RANGE_I(delay_num_filters,                    1, DA_NUM_FILTERS);

    AEC_CK_BOOL(live_observer_enabled);
    AEC_CK_BOOL(min_gain_split_floor_enabled);
    AEC_CK_RANGE_F(live_observer_shift_ms,       0.0f, 1.0e9f);
    AEC_CK_RANGE_F(live_observer_max_hz,         0.0f, 1.0e9f);
    AEC_CK_RANGE_F(live_observer_corrob_ms,      0.0f, 1.0e9f);
    if (!(cfg->live_observer_mu > 0.0f) || cfg->live_observer_mu > 2.0f) return 0;
    if (!isfinite(cfg->live_observer_corrob_margin_db)) return 0;

    /* Delay mode x field compatibility: MATCHED and EXTERNAL_ALIGNED need
     * fixed_delay_samples == -1; delay_num_filters must stay at its default
     * outside MATCHED. */
    switch (cfg->delay_mode) {
        case AEC_DELAY_MATCHED:
            if (cfg->fixed_delay_samples != -1) return 0;
            break;
        case AEC_DELAY_FIXED:
            if (cfg->fixed_delay_samples < 0) return 0;
            /* Bound on the reference-ring size (fixed_delay_samples + hop): 120 s. */
            if ((long)cfg->fixed_delay_samples > 120L * (long)cfg->sample_rate) return 0;
            if (cfg->delay_num_filters != DA_NUM_FILTERS) return 0;
            break;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            if (cfg->fixed_delay_samples != -1) return 0;
            if (cfg->delay_num_filters != DA_NUM_FILTERS) return 0;
            break;
        default:
            return 0;
    }

#undef AEC_CK_BOOL
#undef AEC_CK_RANGE_F
#undef AEC_CK_RANGE_I

    return 1;
}

/* ───────────────────────── helpers ─────────────────────────────────────── */

/* Mean of x^2 in float32. scratch holds >= n floats. */
static float mean_sq(const float* x, int n, float* scratch) {
    sk_sq_scale_f32(x, 1.0f, scratch, n);
    float s = sk_pairwise_sum_f32(scratch, (size_t)n);
    return s / (float)n;
}


static void filter_q_high(PBFDKF* f) {
    int K = f->base.n_freqs;
    for (int k = 0; k < K; ++k) f->Q[k] = f->Q_high[k];
}

/* Counters saturate at LONG_MAX (32-bit long on some targets). */
static inline void aec_counter_bump(long* c) {
    if (*c < LONG_MAX) *c += 1;
}

/* Clamps a long hop count to INT_MAX for gate comparisons. */
static inline int aec_count_as_int(long c) {
    return c > (long)INT_MAX ? INT_MAX : (int)c;
}

/* Re-arms a PBFDAF excitation guard after a delay change (shadow or observer). */
static inline void aec_rearm_excitation(PBFDAF* f) {
    f->poor_excitation_counter = AEC_POOR_EXC_COUNTER_INITIAL_HOPS;
    f->call_counter = 0;
}

static void aec_post_chain_reset(Aec* a);

/* Latches, EMAs and detectors downstream of the taps that a restarted filter
 * must not carry forward. */
static void aec_reset_filter_latches(Aec* a) {
    epc_reset(&a->epc);

    a->main_err_smooth = 0.0f;
    a->shadow_err_smooth = 0.0f;
    a->raw_error_power = 0.0f;
    a->near_power = 0.0f;

    a->erle_window_near = 1e-10f;
    a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f;
    a->inst_erle_smooth = 1.0f;
    a->erle_slope_len = 0;
    a->erle_slope_head = 0;

    a->simple_mu_ratio = 1.0f;
    a->simple_mu_holdoff = 0;
    a->has_per_bin_mu = 0;

    a->erl_estimate = 0.1f;
    a->epc_render_forced_remaining = 0;
    doubletalk_reset(&a->dt_analyzer);
    a->stat_dt_hangover = 0;

    a->shadow_frame_count = 0;
    shadow_copy_reset(&a->regime);

    a->poor_coarse_counter = 0;
    a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;

    filter_q_high(&a->main_filter);
    int half = a->cfg.warmup_frames / 2;
    if (a->warmup_frames_remaining < half) a->warmup_frames_remaining = half;
    a->warmup_far_active = 0;

    /* Warm-transfer evidence refers to the abandoned alignment: clear it. */
    a->last_erle_windowed = 0.0f;

    a->block_stationary_next = 0;
}

static void aec_reset_filter_derived_state(Aec* a) {
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);
    if (a->has_observer) pbfdaf_reset(&a->observer_filter);

    filter_convergence_reset(&a->convergence);
    aec_reset_filter_latches(a);

    aec_post_chain_reset(a);

    a->pending_delay = -1;
    a->has_pending = 0;
    a->pending_delay_ttl = 0;
}

static int aec_delay_value_rejected(const Aec* a, int candidate) {
    int i;
    for (i = 0; i < a->delay_reject_count; ++i) {
        if (abs(candidate - a->delay_reject_values[i]) < 16) return 1;
    }
    return 0;
}

/* Commits target_delay as the ring-read alignment and runs the delay-shift
 * recovery chain (Path B hard reset, bad-relock revert). */
static void aec_apply_delay_shift_hard_reset(Aec* a, int target_delay) {
    a->current_delay = target_delay;
    aec_reset_filter_derived_state(a);
    epc_force_delay(&a->epc);
    filter_q_high(&a->main_filter);
    if (a->has_shadow) aec_rearm_excitation(&a->shadow_filter);
    if (a->has_observer) aec_rearm_excitation(&a->observer_filter);
    filter_convergence_mark_diverged(&a->convergence);
    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
    a->pending_delay_change = AEC_DA_NEW_DETECTED;
}

/* Filter-only restart for a far realign (aec_apply_external_realign): clears
 * taps, far spectra, far-power normalizer and far_buffer; keeps the
 * post-filter chain and near_buffer so the output stays continuous. Does not
 * call pbfdkf_handle_echo_path_change() (for filters that keep their taps).
 * Re-arms stationarity so the freeze cannot hold wiped taps. */
static void aec_reset_filter_for_realign(Aec* a) {
    pbfdkf_reset_taps(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset_taps(&a->shadow_filter);
    if (a->has_observer) pbfdaf_reset_taps(&a->observer_filter);

    memset(a->main_filter.base.far_buffer, 0,
           (size_t)a->main_filter.base.block_size * sizeof(float));
    if (a->has_shadow)
        memset(a->shadow_filter.far_buffer, 0,
               (size_t)a->shadow_filter.block_size * sizeof(float));
    if (a->has_observer) {
        memset(a->observer_filter.far_buffer, 0,
               (size_t)a->observer_filter.block_size * sizeof(float));
        /* Post chain is preserved, so clear the corroboration EMAs here. */
        a->post.observer_obs_err_ema = 0.0;
        a->post.observer_main_err_ema = 0.0;
    }

    filter_convergence_mark_diverged(&a->convergence);
    aec_reset_filter_latches(a);

    a->stationarity_active_hops = 0;
}

/* Per-bin mu scale into a->per_bin_mu_scale when active, else a scalar; sets
 * *out_is_array. */
static float get_simple_mu_scale(Aec* a, int* out_is_array) {
    float mu_min = a->cfg.shadow_mu_min;
    *out_is_array = 0;
    if (a->warmup_frames_remaining > 0) {
        if (a->warmup_far_active) a->warmup_frames_remaining--;
        if (a->simple_mu_ratio < 0.2f) {
            return (a->simple_mu_ratio > 0.2f) ? a->simple_mu_ratio : 0.2f;
        }
        float v = a->simple_mu_ratio + 0.2f;
        if (v < 0.5f) v = 0.5f;
        if (v > 1.0f) v = 1.0f;
        return v;
    }
    if (!a->convergence.converged) { if (mu_min < 0.3f) mu_min = 0.3f; }
    else                          { if (mu_min < 0.2f) mu_min = 0.2f; }
    if (a->has_per_bin_mu) {
        int K = a->n_freqs;
        for (int k = 0; k < K; ++k) {
            float v = a->per_bin_mu_scale[k];
            if (v < mu_min) v = mu_min;
            a->per_bin_mu_scale[k] = v;
        }
        *out_is_array = 1;
        return 0.0f;
    }
    return mu_min + (1.0f - mu_min) * a->simple_mu_ratio;
}

/* Simple-mu state machine, hop-authored (not retimed to wall clock). */
enum { SIMPLE_MU_HOLDOFF_HOPS = 20 };
static const float SIMPLE_MU_ALPHA_ATTACK  = 0.3f;
static const float SIMPLE_MU_ALPHA_HOLD    = 0.99f;
static const float SIMPLE_MU_ALPHA_RELEASE = 0.95f;

/* ERL tracking plus the energy-DT update; the powers are passed in. */
static void advance_erl_and_dt(Aec* a, int render_active, float far_pwr,
                               float mic_pwr, float raw_err_pwr) {
    if (far_pwr > 1e-4f) {
        float raw_dt_ratio = raw_err_pwr / (far_pwr + 1e-10f);
        float inst_erl_raw = mic_pwr / far_pwr;
        if (raw_dt_ratio < 2.0f && inst_erl_raw < 1.5f) {
            float inst_erl = inst_erl_raw;
            float alpha_erl;
            if (inst_erl < 0.001f) inst_erl = 0.001f;
            if (inst_erl > 1.0f) inst_erl = 1.0f;
            alpha_erl = a->convergence.converged ? a->alpha_erl_converged
                                                 : a->alpha_erl_tracking;
            a->erl_estimate = alpha_erl * a->erl_estimate
                            + (1.0f - alpha_erl) * inst_erl;
        }
    }
    doubletalk_update_energy_dt(&a->dt_analyzer, render_active,
                                far_pwr, mic_pwr, a->erl_estimate);
}

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
    float *e2_echo = a->scr_e2_echo, *e2_near = a->scr_e2_near;
    sk_cmag2_np_f32(a->main_filter.base.echo_spec, e2_echo, K);
    sk_cmag2_np_f32(a->main_filter.base.near_spec, e2_near, K);
    float echo_est_pwr = sk_pairwise_sum_f32(e2_echo, (size_t)K) + 1e-10f;
    float near_pwr     = sk_pairwise_sum_f32(e2_near, (size_t)K) + 1e-10f;
    if (near_pwr > 1e-8f) {
        float r2 = echo_est_pwr / near_pwr;
        if (r2 < 0.0f) r2 = 0.0f;
        if (r2 > 1.0f) r2 = 1.0f;
        float r2_half = r2 * 0.5f;
        if (r2_half > ratio) ratio = r2_half;
    }
    float alpha;
    if (ratio < a->simple_mu_ratio) {
        alpha = SIMPLE_MU_ALPHA_ATTACK;
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
void aec_testing_update_simple_mu_ratio(Aec* a, const float* output,
                                        const float* far_end, int n) {
    update_simple_mu_ratio(a, output, far_end, n);
}

void aec_testing_reset_filter_latches(Aec* a) {
    aec_reset_filter_latches(a);
}

float aec_testing_legacy10ms_rate(float legacy_per_hop, int hop,
                                  int sample_rate) {
    return aec_legacy10ms_rate(legacy_per_hop, hop, sample_rate);
}
#endif

/* ───────────────────────── construction ────────────────────────────────── */

static int next_pow2(int x) {
    if (x <= 1) return 1;
    if (x > (1 << 20)) x = (1 << 20);
    unsigned int n = 1u;
    unsigned int ux = (unsigned int)x;
    while (n < ux) n <<= 1u;
    return (int)n;
}


/* FilterAnalyzer tap materializer: inverse-transforms only the partitions
 * covering tap span [first, last] into a->filter_taps_full. */
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

/* Shared by aec_carve_state() and aec_post_chain_reset(). */
static void aec_state_bind(Aec* a, int K, int hop, int sample_rate, int np) {
    AecStateConfig acfg;
    aec_state_config_defaults(&acfg);
    acfg.n_bins = K;
    acfg.num_capture_channels = AECB_ST_NUM_CAPTURE_CHANNELS;
    acfg.hop_size = hop;
    /* Real rate, not the 16000 config default. */
    acfg.sample_rate = sample_rate;
    acfg.enable_filter_analyzer = AECB_ST_ENABLE_FILTER_ANALYZER;
    /* Sentinel, not a literal hop count (AEC_STATE_STARTUP_HOPS_AUTO). */
    acfg.erle_startup_hops = AECB_ST_ERLE_STARTUP_HOPS;
    acfg.erl_startup_hops = AECB_ST_ERL_STARTUP_HOPS;
    acfg.echo_can_saturate = AECB_ST_ECHO_CAN_SATURATE;
    acfg.use_linear_filter = AECB_ST_USE_LINEAR_FILTER;
    acfg.conservative_initial_phase = AECB_ST_CONSERVATIVE_INITIAL_PHASE;
    acfg.delay_headroom_samples = AECB_ST_DELAY_HEADROOM_SAMPLES;
    acfg.initial_state_seconds = AECB_ST_INITIAL_STATE_SECONDS;
    acfg.erle_min = AECB_ST_ERLE_MIN;
    acfg.erle_max_l = AECB_ST_ERLE_MAX_L;
    acfg.erle_max_h = AECB_ST_ERLE_MAX_H;
    /* Runtime np*hop, not AECB_FILTER_TAPS_SIZE (equal only at 16 kHz). */
    acfg.filter_taps_size = np * hop;
    acfg.fa_scratch_size = np * hop;
    aec_state_init(&a->a3_state, &acfg, &a->a3_state_st);
    aec_state_set_taps_provider(&a->a3_state, aec_fill_filter_taps, a);
}

/* Shared by aec_carve_residual() and aec_post_chain_reset(). */
static void aec_ree_bind(Aec* a, const AecBalancedRateDims* rd, int K, int hop,
                         int sample_rate, float* x2_nf, int* x2_nfc,
                         float* rm_st, float* rt_st, float* rh_st,
                         float* drd_st, float* rrd_st,
                         float* ld_st, float* lr_st, float* scr_st) {
    ReeEchoModelConfig em;
    memset(&em, 0, sizeof(em));
    /* Rate-varying REE floats come from rd; gate power, slope and
     * reverb-mode fields are rate-invariant. */
    em.min_noise_floor_power = rd->ree_min_noise_floor_power;
    em.noise_gate_power      = AECB_REE_NOISE_GATE_POWER_LEGACY;
    em.noise_gate_slope      = AECB_REE_NOISE_GATE_SLOPE;
    em.stationary_gate_slope = AECB_REE_STATIONARY_GATE_SLOPE;
    em.model_reverb_in_nonlinear_mode = AECB_REE_MODEL_REVERB_IN_NL;
    ree_init(&a->a3_ree, K, hop, sample_rate, &em,
             AECB_REE_DEFAULT_GAIN, AECB_REE_TM_GAIN, AECB_REE_ERLE_ONSET_COMP,
             AECB_REE_REVERB_DECAY, AECB_REE_REVERB_MILD_SCALE,
             AECB_REE_REVERB_ENABLED, AECB_REE_REVERB_TAIL_STRENGTH,
             AECB_REE_USE_AEC_RESIDUAL_NOISE_GATE,
             AECB_USE_STATIONARITY_PROPERTIES,
             AECB_REE_USE_AEC_ECHO_GEN_WINDOW,
             AECB_REE_NL_R2_ENABLED, AECB_REE_NL_R2_ALPHA, rd->ree_nl_norm_power,
             AECB_REE_NL_R2_CAP_MULTIPLE,
             rd->ree_residual_noise_gate_power, rd->ree_noise_floor_hold_hops,
             AECB_REE_USE_FREQ_RESPONSE, AECB_REE_REVERB_USE_CONSERVATIVE,
             rd->ree_reverb_smoothing_base,
             x2_nf, x2_nfc, rm_st, rt_st, rh_st, drd_st, rrd_st,
             ld_st, lr_st, scr_st);
}

/* Re-initialises the post-filter chain sub-objects; StationarityEstimator,
 * non_zero_render_seen and active_hops survive. */
static void aec_post_chain_reset(Aec* a) {
    aec_post_reset(&a->post);
    linear_filter_select_reset(&a->a3_lfs);
    /* Sample rate and fft size are fixed at construction, so the lookup always hits. */
    const AecBalancedRateDims* rd = aecb_rate_cfg(
        a->cfg.sample_rate, a->cfg.fft_size);
    aec_state_bind(a, a->n_freqs, a->hop_size, a->cfg.sample_rate,
                   a->n_partitions);
    aec_ree_bind(a, rd, a->n_freqs, a->hop_size, a->cfg.sample_rate,
                 a->a3_ree.x2_noise_floor, a->a3_ree.x2_noise_floor_counter,
                 a->a3_ree.reverb_model.reverb,
                 a->a3_ree.reverb_freq_resp.tail_response,
                 a->a3_ree.render_history,
                 a->a3_ree.delay_render_buf, a->a3_ree.reverb_render_history,
                 a->a3_ree.last_r2_direct, a->a3_ree.last_r2_reverb,
                 a->a3_ree.scratch);
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainConfig scfg = sg->cfg;
        SuppressionGainTuning stun = sg->tun;
        suppression_gain_init(sg, &scfg, &stun, sg->last_gain, sg->last_nearend,
                              sg->last_echo, sg->ma_buf, sg->nearend,
                              sg->weighted_residual, sg->min_gain, sg->max_gain,
                              sg->g_raw, sg->gain, sg->sum_scratch);
    }
}

static void aec_derive_dims(const AecConfig* cfg,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp, int* o_fifo_cap);
static int aec_carve(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, int fcap, int is_static);

int aec_create(Aec* a, const AecConfig* cfg_in) {
    if (!a || !cfg_in || !aec_validate_config(cfg_in)) return -1;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;

    int hop, blk, fft, K, np, buf_samp, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf_samp, &fcap);

    /* One arena (aec_get_mem_size() minus the Aec struct), carved like
     * aec_init(). */
    size_t total = aec_get_mem_size(cfg);
    if (total == 0) return -1;
    size_t arena_sz = total - ALIGN16(sizeof(Aec));

    /* 16-byte base required by the carved FFT and static sub-objects. */
    void* arena_mem = NULL;
    if (posix_memalign(&arena_mem, 16u, arena_sz) != 0) return -1;
    uint8_t* arena = (uint8_t*)arena_mem;

    memset(a, 0, sizeof(*a));
    if (aec_carve(a, arena, cfg, hop, blk, fft, K, np, buf_samp, fcap,
                  /*is_static=*/0) != 0) {
        /* aec_carve() failed (FFT allocation); it already released the FFT handle. */
        free(arena);
        memset(a, 0, sizeof(*a));
        return -1;
    }
    a->heap_arena = arena;
    return 0;
}

/* ── reference alignment ring sizing ─────────────────────────────────────── */

/* Hop-boundary headroom (samples) of the MATCHED search ring; also bounds
 * which delay estimates are eligible. */
#define AEC_REF_RING_MATCHED_HEADROOM 4096

/* Reference ring capacity in samples (shared by aec_get_mem_size() and
 * aec_carve()): MATCHED max(delay_buffer_ms, max_delay_ms + headroom); FIXED
 * fixed_delay_samples + hop_size (tight); EXTERNAL_ALIGNED 0. Returns 0 for
 * EXTERNAL_ALIGNED or an int overflow. */
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

/* Frame dimensions: geometry from the shared resolver, partitions / ring /
 * FIFO capacity from cfg; zero-filled on an unresolvable grid. */
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
    *o_buf_samp = aec_ref_ring_samples(cfg, hop);
    int cap = (AEC_STREAM_FIFO_MS * cfg->sample_rate / 1000 + hop - 1) / hop;
    if (cap < 2) cap = 2;
    /* FIFO capacity is a power of two so unsigned sequence indexing stays
     * continuous across the 2^32 wrap. */
    cap = next_pow2(cap);
    *o_fifo_cap = cap;
}

size_t aec_get_mem_size(const AecConfig* cfg_in) {
    if (!cfg_in) return 0;
    if (!aec_validate_config(cfg_in)) return 0;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;
    int hop, blk, fft, K, np, buf, fcap;
    aec_derive_dims(cfg, &hop, &blk, &fft, &K, &np, &buf, &fcap);

    const int ncc  = AECB_ST_NUM_CAPTURE_CHANNELS;
    const int rh   = (AECB_REE_USE_AEC_ECHO_GEN_WINDOW ? 1 : 0) + 2;
    const AecBalancedRateDims* rd = aecb_rate_cfg(
        cfg->sample_rate, cfg->fft_size);
    if (!rd) return 0;
    const int sg_n = rd->sg_nearend_smoother_n;
    const size_t Kz = (size_t)K;

    /* Checked size arithmetic: overflow saturates and MEM_SIZE_INVALID(t)
     * returns 0. Field order must match aec_carve. */
    size_t t = 0;
    t = ck_field_size(t, 1, sizeof(Aec));
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        size_t da_sz = delay_aec_get_mem_size(cfg->sample_rate, hop,
                                              cfg->delay_num_filters);
        if (da_sz == 0) return 0;
        t = ck_add_size(t, da_sz);
    }
    /* Reference ring: MATCHED and FIXED only; zero capacity there is a config
     * failure. */
    if (cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED && buf <= 0) return 0;
    if (buf > 0)
        t = ck_field_size(t, (size_t)buf, sizeof(float));
    t = ck_field_size(t, ck_mul_size((size_t)fcap, (size_t)hop), sizeof(float));   /* render_fifo */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* fifo_zero_ref (underrun reference) */
    t = ck_field_size(t, (size_t)(K - 2 > 0 ? K - 2 : 1), sizeof(int64_t));
    t = ck_add_size(t, fft_get_mem_size(fft));
    {
        size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
        if (kf_sz == 0) return 0;
        t = ck_add_size(t, kf_sz);
    }
    if (cfg->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (af_sz == 0) return 0;
        t = ck_add_size(t, af_sz);
    }
    if (cfg->live_observer_enabled) {
        size_t obs_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (obs_sz == 0) return 0;
        t = ck_add_size(t, obs_sz);
        t = ck_field_size(t, (size_t)hop, sizeof(float));   /* observer_far_hop */
        t = ck_field_size(t, (size_t)hop, sizeof(float));   /* a3_sc.observer_out (hop) */
        t = ck_field_size(t, Kz, sizeof(Complex));          /* a3_sc.observer_diff_spec */
        t = ck_field_size_reps(t, Kz, sizeof(float), 2);    /* a3_sc diff/cap magnitude^2 */
    }
    /* Pre-shift W snapshots for the bad-relock revert (shadow's only when
     * enable_shadow). */
    t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(Complex));
    if (cfg->enable_shadow)
        t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(Complex));
    t = ck_field_size(t, (size_t)((hop + 7) / 8), sizeof(float));
    t = ck_field_size_reps(t, Kz, sizeof(float), 6);   /* avg_rev y2s n2 n2i sye_re sye_im */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* syy see */
    t = ck_field_size(t, (size_t)blk, sizeof(float));  /* ola */
    t = ck_field_size_reps(t, Kz, sizeof(float), 8);   /* np_ fp_ ep_ erp_ cpe x2r cn nf */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* cgm */
    t = ck_field_size(t, Kz, sizeof(Complex));         /* eout */
    t = ck_field_size(t, (size_t)fft, sizeof(float));  /* eout_full */
    t = ck_field_size_reps(t, Kz, sizeof(float), 5);   /* erle_max/erle/oc/unb/during */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* erle_coming_onset */
    t = ck_field_size(t, Kz, sizeof(int32_t));         /* erle_hold */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* erle_y2_acc/e2_acc */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* erle_low_render */
    t = ck_field_size(t, Kz, sizeof(float));           /* erl */
    t = ck_field_size(t, (size_t)(K - 2), sizeof(int)); /* erl_hold */
    t = ck_field_size(t, (size_t)ncc, sizeof(int));    /* filter_delays_blocks */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* fa_h_highpass */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* fa_abs_scratch */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* fa_render_sq_scratch */
    t = ck_field_size(t, Kz, sizeof(float));           /* x2_nf */
    t = ck_field_size(t, Kz, sizeof(int));             /* x2_nf_c */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* rm_st rt_st */
    t = ck_field_size(t, ck_mul_size((size_t)rh, Kz), sizeof(float)); /* rh_st */
    t = ck_field_size_reps(t, ck_mul_size((size_t)REE_DELAY_BUF_SIZE, Kz),
                           sizeof(float), 2);          /* drd rrd */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* ld_st lr_st scr_st */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* last_gain/ne/echo */
    t = ck_field_size(t, ck_mul_size((size_t)sg_n, Kz), sizeof(float)); /* ma */
    t = ck_field_size_reps(t, Kz, sizeof(float), 7);   /* ne wr ming maxg graw gout gsum */
    t = ck_field_size(t, Kz, sizeof(float));           /* stat_noise */
    t = ck_field_size(t, Kz, sizeof(int32_t));         /* stat_hang */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_flags */
    t = ck_field_size(t, ck_mul_size((size_t)16, Kz), sizeof(float)); /* stat_hist */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 2); /* prev_output_time e_form */
    t = ck_field_size(t, (size_t)blk, sizeof(float));  /* block_win */
    t = ck_field_size(t, Kz, sizeof(Complex));         /* lfs sel_esw */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 3); /* scr_sq scr_sref scr_scoa */
    t = ck_field_size(t, (size_t)fft, sizeof(float));  /* scr_tin */
    t = ck_field_size_reps(t, Kz, sizeof(Complex), 4); /* sel_esw sel_echo nsw_e1 ybase */
    t = ck_field_size_reps(t, Kz, sizeof(float), 9);   /* abs_near..x2_past */
    t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(float)); /* w_mag2 */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* render_block_scaled */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* r2 r2_unb nearend_pwr */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_mask */
    t = ck_field_size(t, Kz, sizeof(float));           /* per_bin_mu_scale */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 5); /* near_hop far_hop raw shadow final */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* filter_taps_full */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* scr_sq */
    t = ck_field_size_reps(t, Kz, sizeof(float), 8);   /* scr_e2_echo .. scr_erl_arr */
    if (cfg->enable_highpass) t = ck_field_size(t, 1, hpf_get_mem_size()); /* mic-path HPF */

    if (MEM_SIZE_INVALID(t)) return 0;
    return t;
}

/* Static-pool memory breakdown; total_bytes is aec_get_mem_size(). */
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
        est = delay_aec_get_mem_size(cfg->sample_rate, hop,
                                       cfg->delay_num_filters);
    }
    size_t ring = (buf > 0) ? ck_field_size(0, (size_t)buf, sizeof(float)) : 0;

    out->total_bytes     = total;
    out->estimator_bytes = est;
    out->ring_bytes      = ring;
    return 1;
}

/* Shared carve for aec_init() (Aec at mem[0]) and aec_create() (caller-owned
 * Aec plus one arena): same buffers, same order. `a` must be zeroed;
 * is_static tells aec_destroy() the caller owns the memory. Returns 0, or -1
 * if the shared FFT or hpf_init() failed (an FFT handle that came up is
 * released first). */

/* Wall-clock-preserving counter thresholds shared by aec_carve() and
 * aec_reset(). */
static void aec_recompute_wallclock_thresholds(Aec* a, int hop, int sample_rate) {
    a->poor_coarse_threshold_hops = aec_blocks_to_hops(5, hop, sample_rate);
    a->coarse_reset_hangover_hops = aec_blocks_to_hops(25, hop, sample_rate);
    a->leakage_div_sustain_hops = aec_ms_to_hops(50.0f, hop, sample_rate);
    a->stat_dt_hangover_hops = aec_ms_to_hops(800.0f, hop, sample_rate);
}

/* Suppression-gain config and tuning for a validated, resolved cfg; shared by
 * Aec and standalone RES instances. Tuning tables are static per-rate.
 * Returns 0, or -1 without a signal grid or rate row. */
int aec_build_sg_config(const AecConfig* cfg, SuppressionGainConfig* out_cfg,
                        SuppressionGainTuning* out_tun) {
    AecSignalGrid grid;
    const AecBalancedRateDims* rd;
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    int hop, K;
    if (!cfg || !out_cfg || !out_tun) return -1;
    if (!aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &grid)) return -1;
    rd = aecb_rate_cfg(cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;
    hop = grid.hop_size;
    K = grid.n_freqs;
    memset(&scfg, 0, sizeof(scfg));
    scfg.n_bins = K; scfg.sr = cfg->sample_rate; scfg.hop_size = hop;
    scfg.last_lf_band = rd->sg_last_lf_band;
    scfg.first_hf_band = rd->sg_first_hf_band;
    scfg.last_lf_smoothing_band = rd->sg_last_lf_smoothing_band;
    scfg.last_permanent_lf_smoothing_band = AECB_SG_LAST_PERMANENT;
    scfg.lf_smoothing_during_initial_phase = AECB_SG_LF_SMOOTHING_INITIAL;
    scfg.lf_clamp_bin = rd->sg_lf_clamp_bin;
    scfg.dne_lf_end = rd->sg_dne_lf_end;
    scfg.nearend_smoother_n = rd->sg_nearend_smoother_n;
    scfg.aud_lf_end_bin = rd->sg_aud_lf_end_bin; scfg.aud_mf_end_bin = rd->sg_aud_mf_end_bin;
    scfg.floor_power = rd->sg_floor_power;
    scfg.aud_thr_lf = AECB_SG_AUD_THR_LF; scfg.aud_thr_mf = AECB_SG_AUD_THR_MF;
    scfg.aud_thr_hf = AECB_SG_AUD_THR_HF;
    scfg.low_render_limit = rd->sg_low_render_limit;
    scfg.normal_render_limit = rd->sg_normal_render_limit;
    scfg.hf_lgb = rd->sg_hf_lgb; scfg.hf_biq = AECB_SG_HF_BIQ;
    scfg.conservative_hf = AECB_SG_CONSERVATIVE_HF;
    scfg.max_inc_normal = rd->sg_max_inc;
    scfg.max_inc_nearend = rd->sg_max_inc;
    scfg.max_dec_lf_normal = rd->sg_max_dec_lf;
    scfg.max_dec_lf_nearend = rd->sg_max_dec_lf;
    scfg.floor_first_increase = 0.00001f;
    scfg.low_render_threshold = rd->sg_low_render_threshold;
    /* From AecConfig, so NO_FLOOR reaches the suppressor at construction. */
    scfg.split_floor_enabled = cfg->min_gain_split_floor_enabled;
    scfg.split_floor_far_active =
        powf(10.0f, cfg->min_gain_floor_far_active_db / 10.0f);
    scfg.split_floor_far_silent = AECB_SG_SPLIT_FLOOR_FAR_SILENT;
    scfg.split_floor_dt =
        powf(10.0f, cfg->min_gain_floor_dt_db / 10.0f);
    scfg.split_floor_latch_power = AECB_SG_SPLIT_FLOOR_LATCH_POWER;
    scfg.soft_blend_enabled = AECB_SG_SOFT_BLEND_ENABLED;
    scfg.soft_blend_per_bin = AECB_SG_SOFT_BLEND_PER_BIN;
    scfg.soft_blend_enr_thr = (float)AECB_SG_SOFT_BLEND_ENR_THR;
    scfg.soft_blend_softness = (float)AECB_SG_SOFT_BLEND_SOFTNESS;
    scfg.dne_enr_threshold = AECB_SG_DNE_ENR_THRESHOLD;
    scfg.dne_enr_exit_threshold = AECB_SG_DNE_ENR_EXIT_THRESHOLD;
    scfg.dne_snr_threshold = AECB_SG_DNE_SNR_THRESHOLD;
    scfg.dne_use_during_initial_phase = AECB_SG_DNE_USE_DURING_INITIAL_PHASE;
    scfg.dne_use_unbounded_echo = AECB_SG_DNE_USE_UNBOUNDED_ECHO;
    scfg.dne_lf_endpoint_bin = rd->sg_dne_lf_endpoint_bin;
    scfg.dne_trigger_threshold_hops = rd->sg_trigger_threshold_hops;
    scfg.dne_hold_duration_hops = rd->sg_hold_duration_hops;
    scfg.stat_aware_ne_proxy_enabled = 0; scfg.stat_aware_ne_proxy_threshold = 0.10f;
    stun.nearend_enr_tr = rd->sg_nearend_enr_tr; stun.nearend_enr_su = rd->sg_nearend_enr_su;
    stun.nearend_emr_tr = rd->sg_nearend_emr_tr;
    stun.normal_enr_tr  = rd->sg_normal_enr_tr;  stun.normal_enr_su  = rd->sg_normal_enr_su;
    stun.normal_emr_tr  = rd->sg_normal_emr_tr;
    stun.table_len      = rd->sg_table_len;
    *out_cfg = scfg;
    *out_tun = stun;
    return 0;
}

int aec_resolve_valid_config(const AecConfig* in, AecConfig* out) {
    if (!in || !out || !aec_validate_config(in)) return 0;
    *out = aec_resolved_config(in);
    return 1;
}

static void aec_init_config_state(Aec* a, const AecConfig* cfg, int hop, int blk,
                                  int fft, int K, int np) {
    a->cfg = *cfg;
    a->hop_size = hop; a->block_size = blk; a->fft_size = fft;
    a->n_freqs = K; a->n_partitions = np;

    /* Retimes wall-clock-authored constants to the resolved grid (only
     * a->cfg, the instance copy). */
    a->cfg.shadow_err_alpha = aec_legacy10ms_alpha(cfg->shadow_err_alpha, hop, cfg->sample_rate);
    a->cfg.warmup_frames = aec_legacy10ms_hops((float)cfg->warmup_frames * 10.0f, hop, cfg->sample_rate);
    a->cfg.epc_hangover = aec_legacy10ms_hops((float)cfg->epc_hangover * 10.0f, hop, cfg->sample_rate);
    a->cfg.ne_recent_hold = aec_legacy10ms_hops((float)cfg->ne_recent_hold * 10.0f, hop, cfg->sample_rate);
    a->cfg.filter_misadjustment_stable_frames = aec_legacy10ms_hops(
        (float)cfg->filter_misadjustment_stable_frames * 10.0f, hop, cfg->sample_rate);
    a->cfg.filter_misadjustment_hangover_frames = aec_legacy10ms_hops(
        (float)cfg->filter_misadjustment_hangover_frames * 10.0f, hop, cfg->sample_rate);

    {
        int _sn = (int)(0.5f * (float)cfg->sample_rate / (float)(hop > 0 ? hop : 1));
        if (_sn < 2) _sn = 2;
        if (_sn > (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0])))
            _sn = (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0]));
        a->erle_slope_cap = _sn;
        a->erle_slope_len = 0;
        a->erle_slope_head = 0;
    }

    if (cfg->enable_saturation) {
        saturation_init(&a->sat_ref, cfg->saturation_threshold,
                        hop, cfg->sample_rate);
        saturation_init(&a->sat_mic, cfg->saturation_threshold,
                        hop, cfg->sample_rate);
        a->has_sat = 1;
    }
}

static uint8_t* aec_carve_delay_fifo(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                 int hop, int K, int buf_samp, int fcap) {
    /* Delay ring (cfg is delay-resolved): MATCHED estimator + ring; FIXED
     * ring only; EXTERNAL_ALIGNED neither. buf_samp is the capacity
     * aec_get_mem_size() budgeted. */
    if (aec_ring_active(cfg)) {
        if (buf_samp <= 0) return NULL;
        if (cfg->delay_mode == AEC_DELAY_MATCHED) {
            size_t da_sz = delay_aec_get_mem_size(cfg->sample_rate, hop,
                                                  cfg->delay_num_filters);
            if (da_sz == 0) return NULL;
            if (delay_aec_init(&a->delay, ptr, da_sz, cfg->sample_rate, hop,
                               cfg->delay_num_filters) != 0) return NULL;
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
    a->duty_last_delay = -1;
    /* Streaming FIFO: plain init, before the instance is published to any thread. */
    a->fifo_cap_hops = fcap;
    a->render_fifo = (float*)ptr; ptr += ALIGN16((size_t)fcap * hop * sizeof(float));
    memset(a->render_fifo, 0, (size_t)fcap * hop * sizeof(float));
    a->fifo_zero_ref = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    memset(a->fifo_zero_ref, 0, (size_t)hop * sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;

    a->rsa_counters = (int64_t*)ptr;
    size_t rsa_sz = ALIGN16((size_t)(K - 2 > 0 ? K - 2 : 1) * sizeof(int64_t));
    memset(ptr, 0, rsa_sz); ptr += rsa_sz;

    return ptr;
}

static uint8_t* aec_carve_filters(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                              int hop, int blk, int fft, int K, int np) {
    /* Shared FFT handle: one per instance, borrowed by the main filter,
     * shadow filter and aec_post; a failure here needs no unwind. */
    size_t fft_sz = fft_get_mem_size(fft);
    a->post_fft = fft_init(ptr, fft_sz, fft);
    if (!a->post_fft) return NULL;
    ptr += fft_sz;

    size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
    if (pbfdkf_init_static(&a->main_filter, ptr, kf_sz, blk, np, cfg->mu, cfg->delta, hop, cfg->sample_rate, a->post_fft) != 0) {
        fft_destroy(a->post_fft); a->post_fft = NULL;
        return NULL;
    }
    for (int k = 0; k < K; ++k) {
        a->main_filter.Q_high[k] = cfg->kalman_q_high;
        a->main_filter.Q_low[k]  = cfg->kalman_q_low;
        a->main_filter.Q[k]      = cfg->kalman_q_high;
    }
    a->main_filter.base.poor_excitation_counter = aec_blocks_to_hops(1000, hop, cfg->sample_rate);
    a->main_filter.base.constraint_round_robin = 1;
    ptr += kf_sz;

    if (cfg->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (pbfdaf_init_static(&a->shadow_filter, ptr, af_sz, blk, np, cfg->shadow_mu_nlms, cfg->delta, hop, 1, cfg->sample_rate, a->post_fft) != 0) {
            pbfdkf_free(&a->main_filter);
            fft_destroy(a->post_fft); a->post_fft = NULL;
            return NULL;
        }
        a->shadow_filter.poor_excitation_counter = a->main_filter.base.poor_excitation_counter;
        a->shadow_filter.saturated_capture = 0;
        a->shadow_filter.lightweight = 1;
        a->shadow_filter.constraint_round_robin = 1;
        a->has_shadow = 1;
        ptr += af_sz;
    }

    /* Live observer: PBFDAF like the shadow with mu = live_observer_mu and
     * the same excitation seeding; it does not mirror the soft-realign IR
     * shift, the RSA mask or the tap snapshot. */
    if (cfg->live_observer_enabled) {
        size_t obs_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (pbfdaf_init_static(&a->observer_filter, ptr, obs_sz, blk, np,
                               cfg->live_observer_mu, cfg->delta, hop, 1,
                               cfg->sample_rate, a->post_fft) != 0) {
            if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
            pbfdkf_free(&a->main_filter);
            fft_destroy(a->post_fft); a->post_fft = NULL;
            return NULL;
        }
        a->observer_filter.poor_excitation_counter = a->main_filter.base.poor_excitation_counter;
        a->observer_filter.saturated_capture = 0;
        a->observer_filter.lightweight = 1;
        a->observer_filter.constraint_round_robin = 1;
        a->has_observer = 1;
        ptr += obs_sz;
        a->observer_far_hop = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->observer_far_hop_valid = 0;
        a->observer_shift_hops = aec_ms_to_hops(cfg->live_observer_shift_ms, hop, cfg->sample_rate);
    }

    {
        size_t w_snap_sz = ALIGN16(ck_mul_size((size_t)np, (size_t)K) * sizeof(Complex));
        a->delay_shift_judgment_pre_W = (Complex*)ptr; ptr += w_snap_sz;
        if (cfg->enable_shadow) {
            a->delay_shift_judgment_pre_shadow_W = (Complex*)ptr; ptr += w_snap_sz;
        } else {
            a->delay_shift_judgment_pre_shadow_W = NULL;
        }
    }
    a->delay_shift_judgment_active = 0;
    a->delay_shift_judgment_pre_delay = -1;
    a->delay_shift_judgment_pre_erle = 0.0f;
    a->delay_shift_judgment_hops_left = 0;
    a->delay_reject_count = 0;

    return ptr;
}

static uint8_t* aec_carve_detectors(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                    int hop, int K, int np) {
    a->ra_pairwise_scratch = (float*)ptr;
    ptr += ALIGN16((size_t)((hop + 7) / 8) * sizeof(float));
    render_activity_init(&a->render_activity, a->ra_pairwise_scratch, (hop + 7) / 8,
                         hop, cfg->sample_rate);
    filter_convergence_init(&a->convergence, hop, cfg->sample_rate);
    doubletalk_init(&a->dt_analyzer, 1.5, 3.0, hop, cfg->sample_rate);
    epc_init(&a->epc, a->cfg.epc_hangover, cfg->epc_total_rise, cfg->epc_delta_threshold,
             hop, cfg->sample_rate);
    shadow_copy_init_for_grid(&a->regime, SC_GATE_ENERGY, 0.65, 3,
                              a->cfg.epc_hangover, hop, cfg->sample_rate);
    rsa_init(&a->rsa, a->rsa_counters, K, np);

    return ptr;
}

static uint8_t* aec_carve_post_backing(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                       const AecBalancedRateDims* rd,
                                       int hop, int blk, int fft, int K) {
    {
        AecPostConfig pcfg;
        aec_post_config_defaults(&pcfg);
        pcfg.n_bins = K; pcfg.fft_size = fft; pcfg.block_size = blk; pcfg.hop_size = hop;
        pcfg.synth_window_len            = rd->synth_window_len;
        pcfg.erle_coh_gate_enabled       = AECB_ERLE_COH_GATE_ENABLED;
        pcfg.erle_windowed_capture_psd   = AECB_ERLE_WINDOWED_CAPTURE_PSD;
        pcfg.erle_render_x2_psd_scale    = AECB_ERLE_RENDER_X2_PSD_SCALE;
        pcfg.output_capture_when_linear_unusable = AECB_OUTPUT_CAPTURE_WHEN_LINEAR_UNUSABLE;
        pcfg.enable_cng                  = cfg->enable_cng;
        pcfg.cng_n2_windowed_psd         = cfg->cng_n2_windowed_psd;
        pcfg.cng_n2_update_onset_hops    = rd->cng_n2_update_onset_hops;
        pcfg.cng_n2_initial_duration_hops = rd->cng_n2_initial_duration_hops;
        pcfg.cng_y2_alpha                = rd->cng_y2_alpha;
        pcfg.cng_n2_track_freshness      = rd->cng_n2_track_freshness;
        pcfg.cng_n2_track_retention      = rd->cng_n2_track_retention;
        pcfg.cng_n2_slow_up              = rd->cng_n2_slow_up;
        pcfg.cng_n2_initial_alpha        = rd->cng_n2_initial_alpha;
        pcfg.noise_floor_int16sq         = rd->noise_floor_int16sq;
        pcfg.erle_coh_gate_alpha         = AECB_ERLE_COH_GATE_ALPHA;
        pcfg.erle_coh_gate_threshold     = AECB_ERLE_COH_GATE_THRESHOLD;

        /* Live observer scalars: max_bin = ceil(max_hz / bin_hz);
         * err_ema_alpha and corrob_margin stay double (double-precision
         * per-hop recursion). */
        pcfg.observer_enabled = cfg->live_observer_enabled;
        if (cfg->live_observer_enabled) {
            float bin_hz = (float)cfg->sample_rate / (float)fft;
            pcfg.observer_max_bin = (int)ceilf((float)cfg->live_observer_max_hz / bin_hz);
            double hop_ms = 1000.0 * (double)hop / (double)cfg->sample_rate;
            double corrob_ms = (double)cfg->live_observer_corrob_ms;
            if (corrob_ms < 1e-6) corrob_ms = 1e-6;
            pcfg.observer_err_ema_alpha =
                aec_ema_alpha_from_tau_ms(corrob_ms, hop_ms);
            pcfg.observer_corrob_margin =
                aec_db_to_linear((double)cfg->live_observer_corrob_margin_db);
        }

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
        aec_post_init(&a->post, &pcfg, a->post_fft,
                      rd->synth_window, AECB_SQRT2_SIN_LUT, avg_rev,
                      y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                      np_, fp_, ep_, erp_, cpe, x2r, cgm, cn, nf_,
                      eout, eout_full);
    }

    return ptr;
}

static uint8_t* aec_carve_state(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                int hop, int K, int np) {
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
        s->filter_delays_blocks = (int*)ptr; ptr += ALIGN16((size_t)AECB_ST_NUM_CAPTURE_CHANNELS * sizeof(int));
        s->fa_h_highpass      = (float*)ptr; ptr += ALIGN16(ck_mul_size((size_t)np, (size_t)hop) * sizeof(float));
        s->fa_abs_scratch       = (float*)ptr; ptr += ALIGN16(ck_mul_size((size_t)np, (size_t)hop) * sizeof(float));
        s->fa_render_sq_scratch = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        aec_state_bind(a, K, hop, cfg->sample_rate, np);
    }

    return ptr;
}

static uint8_t* aec_carve_residual(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                   const AecBalancedRateDims* rd, int hop, int K) {
    {
        int rh = (AECB_REE_USE_AEC_ECHO_GEN_WINDOW ? 1 : 0) + 2;
        float *x2_nf  = P_FSLICE(0);
        int   *x2_nfc = (int*)ptr;   ptr += ALIGN16(K * sizeof(int));
        float *rm_st  = P_FSLICE(0); float *rt_st = P_FSLICE(0);
        float *rh_st  = (float*)ptr; ptr += ALIGN16((size_t)rh * K * sizeof(float));
        float *drd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *rrd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
        float *ld_st  = P_FSLICE(0); float *lr_st = P_FSLICE(0); float *scr_st = P_FSLICE(0);
        aec_ree_bind(a, rd, K, hop, cfg->sample_rate, x2_nf, x2_nfc, rm_st,
                     rt_st, rh_st, drd_st, rrd_st, ld_st, lr_st, scr_st);
    }

    return ptr;
}

static uint8_t* aec_carve_suppression(Aec* a, uint8_t* ptr, const AecBalancedRateDims* rd,
                                      const SuppressionGainConfig* scfg,
                                      const SuppressionGainTuning* stun, int K) {
    {
        SuppressionGain* sg = &a->a3_sg;
        float *last_gain = P_FSLICE(0); float *last_ne = P_FSLICE(0); float *last_echo = P_FSLICE(0);
        float *ma   = (float*)ptr; ptr += ALIGN16((size_t)rd->sg_nearend_smoother_n * K * sizeof(float));
        float *ne_s = P_FSLICE(0); float *wr_s = P_FSLICE(0);
        float *ming = P_FSLICE(0); float *maxg = P_FSLICE(0);
        float *graw = P_FSLICE(0); float *gout = P_FSLICE(0); float *gsum = P_FSLICE(0);
        suppression_gain_init(sg, scfg, stun, last_gain, last_ne, last_echo,
                              ma, ne_s, wr_s, ming, maxg, graw, gout, gsum);
    }

    return ptr;
}

static uint8_t* aec_carve_stationarity(Aec* a, uint8_t* ptr, const AecConfig* cfg, int hop, int K) {
    {
        float        *st_noise = P_FSLICE(0);
        int32_t      *st_hang  = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        unsigned char *st_flg  = P_BSLICE(0);
        float        *st_hist  = (float*)ptr;   ptr += ALIGN16(16 * K * sizeof(float));
        stationarity_estimator_init(&a->a3_stat, K, hop, cfg->sample_rate,
                                    st_noise, st_hang, st_flg, st_hist);
    }

    return ptr;
}

static uint8_t* aec_carve_linear_select(Aec* a, uint8_t* ptr, int hop, int blk, int fft, int K) {
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
        a->a3_lfs.scr_sq   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_sref = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_scoa = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_tin  = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        linear_filter_select_reset(&a->a3_lfs);
    }

    return ptr;
}

static uint8_t* aec_carve_post_scratch(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                       int hop, int K, int np) {
    {
        AecPostRunScratch* sc = &a->a3_sc;
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
        sc->observer_out = NULL; sc->observer_diff_spec = NULL;
        sc->observer_diff_mag2 = NULL; sc->observer_cap_mag2 = NULL;
        if (cfg->live_observer_enabled) {
            sc->observer_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
            sc->observer_diff_spec = P_CSLICE(0);
            sc->observer_diff_mag2 = P_FSLICE(0);
            sc->observer_cap_mag2  = P_FSLICE(0);
        }
    }
#undef P_FSLICE
#undef P_CSLICE
#undef P_BSLICE

    return ptr;
}

static uint8_t* aec_carve_hop_scratch(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                                  int hop, int K, int np) {
    a->per_bin_mu_scale = (float*)ptr; ptr += ALIGN16(K * sizeof(float));
    a->near_hop   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->far_hop    = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->raw_output = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->shadow_out = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->final_out  = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->filter_taps_full = (float*)ptr; ptr += ALIGN16((size_t)np * hop * sizeof(float));

    a->scr_sq        = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
    a->scr_e2_echo   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2_near   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_rsa_mask  = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf_pre= (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2coa_pre = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_mu_buf    = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2ref_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_erl_arr   = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    /* Mic-path HPF; hpf_init() failure unwinds the earlier allocations. */
    if (cfg->enable_highpass) {
        a->hp_mic = hpf_init(ptr, hpf_get_mem_size(),
                             cfg->highpass_cutoff_hz, cfg->sample_rate);
        if (!a->hp_mic) {
            fft_destroy(a->post_fft); a->post_fft = NULL;
            if (a->has_observer) pbfdaf_free(&a->observer_filter);
            if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
            pbfdkf_free(&a->main_filter);
            return NULL;
        }
        ptr += ALIGN16(hpf_get_mem_size());
    }

    return ptr;
}

static void aec_init_scalar_state(Aec* a, const AecConfig* cfg, const AecBalancedRateDims* rd,
                                  int hop, int is_static) {
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
    a->alpha_erl_tracking  = aec_growth_rehop(0.99f,  160, 16000,
                                              a->hop_size, a->cfg.sample_rate);
    a->alpha_erl_converged = aec_growth_rehop(0.999f, 160, 16000,
                                              a->hop_size, a->cfg.sample_rate);
    /* Duty-cycle ERLE watchdog leak: 0.001 dB per 10 ms hop (0.1 dB/s),
     * converted once. */
    a->duty_erle_leak_db = aec_legacy10ms_rate(0.001f, hop, cfg->sample_rate);
    a->frame_count = 0; a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    aec_recompute_wallclock_thresholds(a, hop, cfg->sample_rate);
    {
        /* Matched-delay duty gate hold / period in hops, cached at carve
         * time. */
        float hop_s = (float)hop / (float)cfg->sample_rate;
        a->duty_hold_hops = (int)lrintf(a->cfg.delay_est_init_s / hop_s);
        a->duty_period_hops = (int)lrintf(a->cfg.delay_est_period_s / hop_s) / 5;
        if (a->duty_hold_hops < 1) a->duty_hold_hops = 1;
        if (a->duty_period_hops < 1) a->duty_period_hops = 1;
    }
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
    filter_q_high(&a->main_filter);
    a->is_static = is_static;
}

static int aec_carve(Aec* a, uint8_t* ptr, const AecConfig* cfg,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, int fcap, int is_static) {
    /* Per-rate row (a hit is guaranteed by aec_validate_config). */
    const AecBalancedRateDims* rd = aecb_rate_cfg(
        cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    if (aec_build_sg_config(cfg, &scfg, &stun) != 0) return -1;

    aec_init_config_state(a, cfg, hop, blk, fft, K, np);
    ptr = aec_carve_delay_fifo(a, ptr, cfg, hop, K, buf_samp, fcap);
    if (!ptr) return -1;
    ptr = aec_carve_filters(a, ptr, cfg, hop, blk, fft, K, np);
    if (!ptr) return -1;
    ptr = aec_carve_detectors(a, ptr, cfg, hop, K, np);
    ptr = aec_carve_post_backing(a, ptr, cfg, rd, hop, blk, fft, K);
    ptr = aec_carve_state(a, ptr, cfg, hop, K, np);
    ptr = aec_carve_residual(a, ptr, cfg, rd, hop, K);
    ptr = aec_carve_suppression(a, ptr, rd, &scfg, &stun, K);
    ptr = aec_carve_stationarity(a, ptr, cfg, hop, K);
    ptr = aec_carve_linear_select(a, ptr, hop, blk, fft, K);
    ptr = aec_carve_post_scratch(a, ptr, cfg, hop, K, np);
    ptr = aec_carve_hop_scratch(a, ptr, cfg, hop, K, np);
    if (!ptr) return -1;
    aec_init_scalar_state(a, cfg, rd, hop, is_static);
    return 0;
}

Aec* aec_init(void* mem, size_t mem_size, const AecConfig* cfg_in) {
    if (!mem || !cfg_in) return NULL;
    if (!aec_validate_config(cfg_in)) return NULL;
    AecConfig cfg_resolved = aec_resolved_config(cfg_in);
    const AecConfig* cfg = &cfg_resolved;
    if (!MEM_IS_ALIGNED16(mem)) return NULL;
    {
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
        /* Nested FFT or hpf_init() failure; aec_carve() already released any
         * FFT handle. */
        return NULL;
    }
    a->heap_arena = NULL;
    return a;
}

void aec_destroy(Aec* a) {
    if (!a) return;
    /* The FFT handle, high-pass filter and filters all live in the arena or
     * pool, so these only drop references. NULLed so a repeated destroy is
     * safe. */
    if (a->post_fft) { fft_destroy(a->post_fft); a->post_fft = NULL; }
    if (a->hp_mic) { hpf_destroy(a->hp_mic); a->hp_mic = NULL; }
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
    if (a->has_observer) pbfdaf_free(&a->observer_filter);

    if (a->is_static) return;

    /* All other storage is in the single arena; one free() reclaims it. */
    free(a->heap_arena);
    a->heap_arena = NULL;
}

int aec_hop_size(const Aec* a) { return a->hop_size; }


/* Saturating bump of the AecLinearContext alignment generation. */
static void aec_bump_delay_generation(Aec* a) {
    if (a->delay_generation != 0xFFFFFFFFu) { a->delay_generation++; }
}
/* Far-end FFT count: own rffts summed over shadow, main and observer (main
 * and shadow are exclusive per hop). */
long aec_far_fft_real_compute_count(const Aec* a) {
    long total = a->main_filter.base.far_fft_real_compute_count;
    if (a->has_shadow) {
        long extra = a->shadow_filter.far_fft_real_compute_count;
        if (extra > LONG_MAX - total) return LONG_MAX;
        total += extra;
    }
    if (a->has_observer) {
        long extra = a->observer_filter.far_fft_real_compute_count;
        if (extra > LONG_MAX - total) return LONG_MAX;
        total += extra;
    }
    return total;
}

int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms) {
    float db, dt_db;
    int en;
    if (a == NULL) return -1;
    if (aec_preset_floor_db(preset, &db) != 0) return -1;
    if (aec_preset_dt_floor_db(preset, &dt_db) != 0) return -1;
    /* Lookups are validated before any write; the suppressor applies first
     * and refuses without writing, leaving a->cfg unchanged. */
    if (aec_preset_split_floor_enabled(preset, &en) != 0) return -1;
    if (suppression_gain_set_split_floor_far_active_db(&a->a3_sg, db,
                                                       ramp_ms) != 0) {
        return -1;
    }
    if (suppression_gain_set_split_floor_dt_db(&a->a3_sg, dt_db) != 0) {
        return -1;
    }
    if (suppression_gain_set_split_floor_enabled(&a->a3_sg, en) != 0) {
        return -1;
    }
    a->cfg.min_gain_floor_far_active_db = db;
    a->cfg.min_gain_floor_dt_db = dt_db;
    a->cfg.min_gain_split_floor_enabled = en;
    return 0;
}

int aec_apply_external_realign(Aec* a, int delta_samples) {
    if (a == NULL || a->cfg.delay_mode != AEC_DELAY_EXTERNAL_ALIGNED)
        return -1;
    if (delta_samples == 0) return 0;
    /* Warm transfer needs aec_linear_is_cancelling() and a shift that fits
     * the tap span. */
    int warm_ok = 0;
    if (a->cfg.delay_acquire_warm_transfer) {
        int reach = pbfdaf_tap_span(&a->main_filter.base);
        int magnitude = delta_samples == INT_MIN ? INT_MAX
                      : (delta_samples > 0 ? delta_samples : -delta_samples);
        warm_ok = aec_linear_is_cancelling(a) && (magnitude < reach);
    }
    /* A refused shift leaves the taps at the abandoned alignment (soft
     * outcome). */
    if (warm_ok && pbfdaf_warm_shift_ir(&a->main_filter.base, delta_samples) == 0) {
        if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, delta_samples);
        aec_bump_delay_generation(a);
        return 1;
    }
    /* Filter reset: taps at the abandoned alignment would cancel against the
     * wrong lag (see aec_reset_filter_for_realign). */
    aec_reset_filter_for_realign(a);
    aec_bump_delay_generation(a);
    return 0;
}

void aec_reset(Aec* a) {
    if (a->hp_mic) hpf_reset(a->hp_mic);
    if (a->has_sat) { saturation_reset(&a->sat_ref); saturation_reset(&a->sat_mic); }
    if (aec_ring_active(&a->cfg)) {
        if (a->has_delay) delay_aec_reset(&a->delay);
        memset(a->ref_ring, 0, (size_t)a->ref_ring_size * sizeof(float));
        a->ref_ring_write = 0; a->ref_ring_filled = 0;
        a->current_delay = a->has_delay ? -1 : a->cfg.fixed_delay_samples;
        a->pending_delay = -1; a->has_pending = 0;
        a->pending_delay_ttl = 0;
        a->delay_shift_judgment_active = 0;
        a->delay_reject_count = 0;
    }
    aec_bump_delay_generation(a);
    a->far_hop_aligned = 0;
    a->duty_active = 0; a->duty_stable_hops = 0; a->duty_pos = 0;
    a->duty_last_delay = -1; a->duty_erle_peak = 0.0f;
    a->duty_hops_total = 0; a->duty_hops_run = 0;
    if (a->render_fifo)
        memset(a->render_fifo, 0,
               (size_t)a->fifo_cap_hops * a->hop_size * sizeof(float));
    if (a->fifo_zero_ref)
        memset(a->fifo_zero_ref, 0, (size_t)a->hop_size * sizeof(float));
    a->fifo_count = 0; a->fifo_read = 0; a->fifo_write = 0;
    a->render_call_count = 0; a->capture_call_count = 0;
    a->last_buffering_event = AEC_BUF_NONE;
    pbfdkf_reset(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset(&a->shadow_filter);
    if (a->has_observer) { pbfdaf_reset(&a->observer_filter); a->observer_far_hop_valid = 0; }
    /* pbfdkf_reset / pbfdaf_reset leave startup/excitation gates standing;
     * aec_reset() clears them. */
    pbfdkf_reset_carried_state(&a->main_filter);
    if (a->has_shadow) pbfdaf_reset_carried_state(&a->shadow_filter);
    if (a->has_observer) pbfdaf_reset_carried_state(&a->observer_filter);
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
    aec_post_chain_reset(a);
    a->pending_gain_change = 0;
    a->pending_delay_change = -1;
    a->saturation_level = 0.0f;
    a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0;
    a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->erle_slope_len = 0; a->erle_slope_head = 0;
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
    /* After a reset the RES-context stash must read what a fresh instance
     * publishes. */
    a->last_far_power = 0.0f; a->last_shadow_dt = 0.0f;
    a->last_dt_indicator = 0.0f; a->last_is_stationary_dt = 0;
}

/* ───────────────────────── misadjustment estimator ─────────────────────── */

/* Filter-misadjustment accumulator update (float32). */
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

/* Filter-misadjustment scale fire (float32). */
static void misadj_fire(Aec* a) {
    if (a->misadj_hangover_remaining > 0) { a->misadj_hangover_remaining--; return; }
    int stable = a->convergence.converged && !a->epc.active && !a->regime.main_paused;
    if (!stable) { a->misadj_stable_count = 0; return; }
    /* stable_count saturates at filter_misadjustment_stable_frames (avoids
     * overflow). */
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

/* Max of the last 15 inst-ERLE ring entries (0.0 when empty): the
 * warm-transfer gate input. */
static float aec_erle_ring_max_last15(const Aec* a) {
    int n = a->erle_slope_len < 15 ? a->erle_slope_len : 15;
    if (n <= 0) return 0.0f;
    int idx = a->erle_slope_head;
    float m = 0.0f; int got = 0;
    for (int i = 0; i < n; i++) {
        idx = (idx - 1 + a->erle_slope_cap) % a->erle_slope_cap;
        float v = a->erle_slope_buf[idx];
        if (!got || v > m) { m = v; got = 1; }
    }
    return m;
}

/* Cancelling test (aec.h): windowed ERLE above 2.5 dB or recent inst-ERLE
 * peak above delay_acquire_inst_erle_db. */
int aec_linear_is_cancelling(const Aec* a) {
    if (!a) return 0;
    return (a->last_erle_windowed > 2.5f)
           || (aec_erle_ring_max_last15(a) > a->cfg.delay_acquire_inst_erle_db);
}

/* ───────────────────────── process ─────────────────────────────────────── */

/* Shared core of the process entry points: linear filter and post-filter/RES
 * block (writing a->final_out), then the power EMAs and convergence
 * detection. shared_far_spec (NULL normally) replaces this instance's far-end
 * FFT; ref_in is still required. */
/* Stage timing is off by default (stamps fold to 0); build with
 * -DAEC_STAGE_TIMING=1. */
#if AEC_STAGE_TIMING
/* 32-bit microsecond monotonic stamp. A target without CLOCK_MONOTONIC
 * defines -DAEC_NOW_US=<fn> (no-argument function returning monotonic
 * uint32_t microseconds). */
#ifndef AEC_NOW_US
#include <time.h>
static uint32_t aec_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull
                      + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static uint32_t aec_now_us(void) { return AEC_NOW_US(); }
#endif
#else
#define aec_now_us() 0u
#endif

void aec_get_last_timing(const Aec* a, AecStageTiming* out) {
    if (!out) return;
    if (!a) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = a->last_timing;
}

static void aec_update_delay_estimator(Aec* a) {
    const int hop = a->hop_size;
    int hold_hops = a->duty_hold_hops;
    int K = a->duty_period_hops;
    int run_filter = 1;
    if (a->duty_active) {
        a->duty_pos += 1;
        if (a->duty_pos >= K) a->duty_pos = 0;
        run_filter = (a->duty_pos == 0);
    }
    /* Engagement census, counted where run_filter is consumed. */
    a->duty_hops_total += 1;
    if (run_filter) a->duty_hops_run += 1;
    delay_aec_accumulate_ex(&a->delay, a->near_hop, a->far_hop, hop,
                            run_filter);
    {
        int cur = delay_aec_estimated_delay(&a->delay);
        int solid = delay_aec_is_solid(&a->delay);
        if (!solid || cur < 0 || cur != a->duty_last_delay) {
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
        /* ERLE watchdog: leaky peak (0.1 dB/s); a drop of more than 6 dB
         * resumes full-rate analysis. Armed once the peak exceeds 6 dB. */
        if (a->last_erle_windowed > a->duty_erle_peak)
            a->duty_erle_peak = a->last_erle_windowed;
        else
            a->duty_erle_peak -= a->duty_erle_leak_db;
        if (a->duty_erle_peak > 6.0f &&
            a->last_erle_windowed < a->duty_erle_peak - 6.0f) {
            a->duty_active = 0;
            a->duty_stable_hops = 0;
            a->duty_erle_peak = 0.0f;
        }
    }
}

static void aec_judge_delay_shift(Aec* a) {
    /* Judges a pending post-shift window before Path A/B: a relock with
     * windowed ERLE more than 1 dB below the pre-shift value is reverted and
     * its candidate stored in delay_reject_values. */
    if (a->delay_shift_judgment_active) {
        a->delay_shift_judgment_hops_left -= 1;
        if (a->delay_shift_judgment_hops_left <= 0) {
            float cur_erle = a->last_erle_windowed;
            /* Plain float32 '<' without epsilon (a float32 ULP exceeds any
             * double slack). */
            int has_estimate = (a->raw_error_power < a->near_power);
            if (has_estimate && cur_erle
                    < a->delay_shift_judgment_pre_erle - 1.0f) {
                int bad_delay = a->current_delay;
                size_t w_elems = (size_t)a->n_partitions * (size_t)a->n_freqs;
                /* Run the reset chain first (it wipes W), then restore the snapshot. */
                aec_apply_delay_shift_hard_reset(
                    a, a->delay_shift_judgment_pre_delay);
                /* The ring read offset moved back; seam consumers must see CHANGED. */
                aec_bump_delay_generation(a);
                memcpy(a->main_filter.base.W,
                       a->delay_shift_judgment_pre_W,
                       w_elems * sizeof(Complex));
                if (a->has_shadow
                        && a->delay_shift_judgment_pre_shadow_W != NULL) {
                    memcpy(a->shadow_filter.W,
                           a->delay_shift_judgment_pre_shadow_W,
                           w_elems * sizeof(Complex));
                }
                if (a->delay_reject_count < AEC_DELAY_REJECT_CAP) {
                    a->delay_reject_values[a->delay_reject_count++] = bad_delay;
                }
            }
            a->delay_shift_judgment_active = 0;
        }
    }

}

static void aec_acquire_delay(Aec* a, int new_delay, int eligible) {
    /* Path A: first acquisition. */
    int already_cancelling = a->cfg.delay_acquire_protect_converged
                             && (a->last_erle_windowed > 2.5f);
    if (eligible && a->current_delay < 0 && delay_aec_is_solid(&a->delay)
            && !already_cancelling) {
        a->current_delay = new_delay;
        aec_bump_delay_generation(a);
        /* Warm tap-transfer: if the inst-ERLE peak exceeds the threshold and
         * the delay fits the tap reach, shift the learned IR instead of
         * zeroing it. */
        int warm_ok = 0;
        if (a->cfg.delay_acquire_warm_transfer) {
            float wpk = aec_erle_ring_max_last15(a);
            int reach = pbfdaf_tap_span(&a->main_filter.base);
            warm_ok = (wpk > a->cfg.delay_acquire_inst_erle_db)
                      && (new_delay > 0) && (new_delay < reach);
        }
        /* A refused shift leaves the taps in place: fall through to the soft
         * / hard branches. */
        if (warm_ok
            && pbfdaf_warm_shift_ir(&a->main_filter.base, new_delay) == 0) {
            if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, new_delay);
        } else if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
            /* Soft acquisition: apply the ring alignment and reset excitation
             * counters only; converged taps are kept. */
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            if (a->has_shadow) aec_rearm_excitation(&a->shadow_filter);
            if (a->has_observer) aec_rearm_excitation(&a->observer_filter);
        } else {
            aec_reset_filter_derived_state(a);
            filter_convergence_mark_diverged(&a->convergence);
            pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
            if (a->has_shadow) aec_rearm_excitation(&a->shadow_filter);
            if (a->has_observer) aec_rearm_excitation(&a->observer_filter);
            a->pending_delay_change = AEC_DA_NEW_DETECTED;
        }
    }

}

static void aec_track_delay_shift(Aec* a, int new_delay, int eligible) {
    if (a->pending_delay_ttl > 0) {
        a->pending_delay_ttl--;
        if (a->pending_delay_ttl <= 0) { a->has_pending = 0; a->pending_delay = -1; }
    }

    float conf = delay_aec_confidence(&a->delay);

    /* Path B: delay shift with a reject-list veto. */
    if (eligible && a->current_delay >= 0 && conf >= 0.5f
            && abs(new_delay - a->current_delay) > 32
            && !aec_delay_value_rejected(a, new_delay)) {
        if (a->has_pending && abs(new_delay - a->pending_delay) < 16) {
            int old_current_delay = a->current_delay;
            /* Snapshot the taps so a confirmed-bad relock can restore them;
             * pre_erle is the reading just before the shift. */
            float pre_shift_erle = a->last_erle_windowed;
            size_t w_elems = (size_t)a->n_partitions * (size_t)a->n_freqs;
            memcpy(a->delay_shift_judgment_pre_W, a->main_filter.base.W,
                   w_elems * sizeof(Complex));
            if (a->has_shadow)
                memcpy(a->delay_shift_judgment_pre_shadow_W,
                       a->shadow_filter.W, w_elems * sizeof(Complex));
            a->current_delay = new_delay;
            aec_bump_delay_generation(a);
            a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
            if (a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
                /* Soft realign: converged filter kept, IR shifted by the
                 * applied delta (pbfdaf_warm_shift_ir) instead of a wipe. */
                int soft_delta = new_delay - old_current_delay;
                if (abs(soft_delta) < pbfdaf_tap_span(&a->main_filter.base)) {
                    pbfdaf_warm_shift_ir(&a->main_filter.base, soft_delta);
                    if (a->has_shadow)
                        pbfdaf_warm_shift_ir(&a->shadow_filter, soft_delta);
                }
            } else {
                /* Taps are matched to the pre-shift alignment: treat as Path
                 * A. */
                aec_apply_delay_shift_hard_reset(a, new_delay);
            }
            /* Bad-relock judgment (both branches): judged after
             * AEC_DELAY_SHIFT_JUDGMENT_HOPS. */
            a->delay_shift_judgment_active = 1;
            a->delay_shift_judgment_pre_delay = old_current_delay;
            a->delay_shift_judgment_pre_erle = pre_shift_erle;
            a->delay_shift_judgment_hops_left = AEC_DELAY_SHIFT_JUDGMENT_HOPS;
        } else {
            a->pending_delay = new_delay; a->has_pending = 1; a->pending_delay_ttl = 3;
        }
    }
}

static void aec_align_reference(Aec* a) {
    const int hop = a->hop_size;
    /* Delay estimation and ring alignment. The estimator is fed every
     * hop; its analysis can be decimated 1-in-K (delay_est_period_s, 0 = full
     * rate). Ring gate: MATCHED or FIXED; estimator gate: MATCHED only. */
    if (aec_ring_active(&a->cfg)) {
      if (a->has_delay) {
        aec_update_delay_estimator(a);
        int new_delay = delay_aec_estimated_delay(&a->delay);
        int eligible = (new_delay >= 0 && delay_aec_n_updates(&a->delay) >= 3
                        && new_delay <= a->ref_ring_size - hop);

        aec_judge_delay_shift(a);

        aec_acquire_delay(a, new_delay, eligible);

        aec_track_delay_shift(a, new_delay, eligible);
      }

        /* Live observer far slice: invalid unless set in the current_delay >
         * 0 branch below. */
        if (a->has_observer) a->observer_far_hop_valid = 0;

        int w = a->ref_ring_write, rs = a->ref_ring_size;
        if (w + hop <= rs) memcpy(a->ref_ring + w, a->far_hop, (size_t)hop * sizeof(float));
        else {
            int p1 = rs - w;
            memcpy(a->ref_ring + w, a->far_hop, (size_t)p1 * sizeof(float));
            memcpy(a->ref_ring, a->far_hop + p1, (size_t)(hop - p1) * sizeof(float));
        }
        a->ref_ring_write = (w + hop) % rs;
        /* Saturate at ring capacity; guard before adding so the int cannot
         * overflow. */
        if (a->ref_ring_filled < rs) {
            a->ref_ring_filled += hop;
            if (a->ref_ring_filled > rs) a->ref_ring_filled = rs;
        }

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

            /* Live observer: same ring read observer_shift_hops hops earlier,
             * clamped at 0. */
            if (a->has_observer) {
                long long obs_back = (long long)a->observer_shift_hops * hop;
                int d_obs = obs_back >= d ? 0 : d - (int)obs_back;
                int read_pos_obs = (a->ref_ring_write - hop - d_obs) % rs;
                if (read_pos_obs < 0) read_pos_obs += rs;
                if (read_pos_obs + hop <= rs)
                    memcpy(a->observer_far_hop, a->ref_ring + read_pos_obs,
                          (size_t)hop * sizeof(float));
                else {
                    int p1o = rs - read_pos_obs;
                    memcpy(a->observer_far_hop, a->ref_ring + read_pos_obs,
                          (size_t)p1o * sizeof(float));
                    memcpy(a->observer_far_hop + p1o, a->ref_ring,
                          (size_t)(hop - p1o) * sizeof(float));
                }
                a->observer_far_hop_valid = 1;
            }
        } else if (a->current_delay == 0) {
            a->far_hop_aligned = 1;
        }
    } else {
        /* EXTERNAL_ALIGNED: ref is aligned by contract. */
        a->far_hop_aligned = 1;
    }

}

static void aec_prepare_capture_hop(Aec* a, const float* mic_in, const float* ref_in) {
    const int hop = a->hop_size;
    memcpy(a->near_hop, mic_in, (size_t)hop * sizeof(float));
    memcpy(a->far_hop,  ref_in, (size_t)hop * sizeof(float));

    if (a->hp_mic) hpf_process(a->hp_mic, a->near_hop, hop);

    /* Held near-end-recent gate for DT-aware soft recovery: arms after
     * ne_recent_sustain frames of the previous dt_from_energy above
     * ne_recent_threshold, holds for ne_recent_hold. Loud residual echo can
     * false-arm it. */
    {
        float ne_ind = a->dt_analyzer.dt_from_energy;
        if (ne_ind > a->cfg.ne_recent_threshold) {
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

    float sat_ref = 0.0f;
    if (a->has_sat) {
        sat_ref = saturation_detect(&a->sat_ref, a->far_hop, hop);
        float sat_mic = saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5f) ? sat_ref : sat_mic * 0.5f;
        if (a->cfg.saturation_softclip_ref && sat_ref > 0.1f)
            saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
    }

}

static float aec_prepare_mu(Aec* a, int* is_array) {
    int mu_is_array = 0;
    float mu_scalar = get_simple_mu_scale(a, &mu_is_array);

    if (a->has_sat && a->sat_mic.saturation_level > 0.8f) {
        mu_scalar = 0.0f;
        mu_is_array = 0;
    }

    *is_array = mu_is_array;
    return mu_scalar;
}

static int aec_prepare_rsa(Aec* a) {
    const int hop = a->hop_size, K = a->n_freqs;
    /* RSA update, poor-excitation counters and saturated_capture (PSD from
     * the previous hop). */
    {
        /* far_cmag2_hold still holds the previous hop's |far_spec|^2 (the
         * frontend has not run). */
        rsa_update(&a->rsa, a->main_filter.base.far_cmag2_hold, a->far_hop, hop);
        int poor = rsa_poor_signal_excitation(&a->rsa);
        /* Counters are capped at the filter's n_partitions (their only
         * consumers are `< n_partitions` gates). */
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
        if (a->has_observer) {
            if (poor) a->observer_filter.poor_excitation_counter = 0;
            else if (a->observer_filter.poor_excitation_counter <
                     a->observer_filter.n_partitions)
                a->observer_filter.poor_excitation_counter += 1;
            a->observer_filter.saturated_capture = (a->saturation_level > 0.5f);
        }
    }

    /* Push the previous hop's block-stationary latch onto all filters
     * before the stationarity refresh below recomputes it. */
    a->main_filter.base.block_stationary = a->block_stationary_next;
    if (a->has_shadow) a->shadow_filter.block_stationary = a->block_stationary_next;
    if (a->has_observer) a->observer_filter.block_stationary = a->block_stationary_next;

    /* RSA narrowband mask: computed once, applied to the main and shadow mu. */
    float *rsa_mask = a->scr_rsa_mask;
    int rsa_mask_active = rsa_any_narrow_band_active(&a->rsa);
    if (rsa_mask_active) {
        for (int k = 0; k < K; ++k) rsa_mask[k] = 1.0f;
        rsa_mask_regions_around_narrow_bands(&a->rsa, rsa_mask);
    }

    return rsa_mask_active;
}

static int aec_run_shadow(Aec* a, float far_hop_mean_sq, int rsa_mask_active,
                          const Complex* shared_far_spec,
                          const float* shared_far_cmag2) {
    const int K = a->n_freqs;
    const float* rsa_mask = a->scr_rsa_mask;
    /* The shadow runs before main so main's H_error refresh reads the
     * same-hop e2_coarse. */
    int far_excited = 0;
    if (a->has_shadow) {
        aec_counter_bump(&a->shadow_frame_count);
        far_excited = (far_hop_mean_sq > 1e-4f);
        int saturation_safe_pre = (a->saturation_level < 0.5f);
        float shadow_mu_pre = (far_excited && saturation_safe_pre) ? 1.0f : 0.1f;
        a->shadow_filter.precomputed_far_spec = shared_far_spec;
        if (rsa_mask_active) {
            float *mu_buf_pre = a->scr_mu_buf_pre;
            sk_scale_f32(mu_buf_pre, rsa_mask, shadow_mu_pre, K);
            pbfdaf_process_with_far_cmag2(&a->shadow_filter, a->near_hop, a->far_hop,
                                           mu_buf_pre, 0.0f, a->shadow_out, shared_far_cmag2);
        } else {
            pbfdaf_process_with_far_cmag2(&a->shadow_filter, a->near_hop, a->far_hop, NULL,
                                           shadow_mu_pre, a->shadow_out, shared_far_cmag2);
        }
        {
            sk_cmag2_np_f32(a->shadow_filter.error_spec,
                            a->main_filter.e2_coarse_per_bin, K);
            a->main_filter.e2_coarse_for_refresh =
                (float)sk_pairwise_sum_f32(a->main_filter.e2_coarse_per_bin,
                                           (size_t)K);
            a->main_filter.e2_coarse_per_bin_valid = 1;
        }
    }

    return far_excited;
}

static int aec_run_main_filter(Aec* a, float mu_scalar, int mu_is_array,
                                int rsa_mask_active, const Complex* shared_far_spec,
                                const float* shared_far_cmag2) {
    const int K = a->n_freqs;
    const float* rsa_mask = a->scr_rsa_mask;
    /* MAIN filter. */
    /* FFT dedup: main reuses the shadow's far_spec (one-shot); without a
     * shadow it uses shared_far_spec. */
    a->main_filter.base.precomputed_far_spec =
        a->has_shadow ? a->shadow_filter.far_spec : shared_far_spec;
    const float* far_cmag2 = a->has_shadow
        ? a->shadow_filter.far_cmag2_hold : shared_far_cmag2;
    float main_mu_scalar = a->regime.main_paused ? 0.0f : mu_scalar;
    {
        float *mu_buf = a->scr_mu_buf;
        int use_array = mu_is_array;
        if (mu_is_array) {
            float pause = a->regime.main_paused ? 0.0f : 1.0f;
            sk_scale_f32(mu_buf, a->per_bin_mu_scale, pause, K);
        }
        if (rsa_mask_active && !use_array) {
            sk_scale_f32(mu_buf, rsa_mask, main_mu_scalar, K);
            use_array = 1;
        } else if (rsa_mask_active && use_array) {
            for (int k = 0; k < K; ++k) mu_buf[k] = mu_buf[k] * rsa_mask[k];
        }
        return pbfdkf_process_with_far_cmag2(
            &a->main_filter, a->near_hop, a->far_hop,
            use_array ? mu_buf : NULL, use_array ? 0.0f : main_mu_scalar,
            a->raw_output, far_cmag2);
    }

}

static int aec_refresh_stationarity(Aec* a) {
    const int hop = a->hop_size;
    int stationarity_block_for_post;
    /* Stationarity refresh for the next hop. */
    {
        float far_max = sk_max_abs_f32(a->far_hop, hop);
        if (!a->non_zero_render_seen && far_max >= a->render_peak_floor)
            a->non_zero_render_seen = 1;
        if (a->non_zero_render_seen) {
            const float *far_psd = a->main_filter.base.far_cmag2_hold;
            stationarity_estimator_update_noise_estimator(&a->a3_stat, far_psd);
            stationarity_estimator_update_stationarity_flags(&a->a3_stat, far_psd,
                                                              a->post.avg_reverb.reverb);
            /* stationarity_active_hops saturates at
             * stationarity_converge_hops. */
            if (a->stationarity_active_hops < a->stationarity_converge_hops) {
                a->stationarity_active_hops += 1;
            }
        }
        /* Latch for the next hop; this hop's shadow still reads the value
         * pushed before the shadow update. */
        int converged_enough = (a->stationarity_active_hops >= a->stationarity_converge_hops);
        /* The raw block-stationary result goes to REE; block_stationary_next
         * is convergence-gated. */
        stationarity_block_for_post =
            stationarity_estimator_is_block_stationary(&a->a3_stat);
        a->block_stationary_next =
            converged_enough && stationarity_block_for_post;
    }

    return stationarity_block_for_post;
}

static void aec_update_shadow_quality(Aec* a, int main_error_cmag2_valid) {
    const int K = a->n_freqs, N = a->n_partitions;
    if (a->has_shadow) {
        float *e2ref_arr = a->scr_e2ref_arr;
        /* PBFDKF fills its cache only on far-active hops; otherwise compute it here. */
        if (main_error_cmag2_valid)
            memcpy(e2ref_arr, a->main_filter.e2_ref_scratch, (size_t)K * sizeof(float));
        else
            sk_cmag2_np_f32(a->main_filter.base.error_spec, e2ref_arr, K);
        float e2_ref = sk_pairwise_sum_f32(e2ref_arr, (size_t)K);
        float e2_coa = a->main_filter.e2_coarse_for_refresh;
        /* erl[k] = sum_p |W_p[k]|^2, recomputed only when w_dirty (its only
         * consumer). */
        if (a->main_filter.base.w_dirty) {
            for (int k = 0; k < K; ++k) a->main_filter.erl_per_bin[k] = 0.0f;
            for (int part = 0; part < N; ++part) {
                const Complex* Wp = a->main_filter.base.W + (size_t)part * K;
                sk_cmag2_np_acc_f32(Wp, a->main_filter.erl_per_bin, K);
            }
            a->main_filter.base.w_dirty = 0;
        }

        /* Rescue: fires when e2_ref < 0.5 * e2_coa for poor_coarse_threshold_hops. */
        int cond_fire = (e2_ref < 0.5f * e2_coa);
        int threshold_hops = a->poor_coarse_threshold_hops;
        if (cond_fire) a->poor_coarse_counter += 1; else a->poor_coarse_counter = 0;
        if (a->poor_coarse_counter >= threshold_hops) {
            pbfdaf_copy_weights_from(&a->shadow_filter, &a->main_filter.base);
            a->coarse_reset_hangover = a->coarse_reset_hangover_hops;
            a->poor_coarse_counter = 0;
            ree_reset(&a->a3_ree);
        }
        /* Sustained leakage_div gate: > 50% of bins diverged for
         * leakage_div_sustain_hops (~50 ms). */
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

}

static void aec_update_shadow_dt(Aec* a, float far_hop_mean_sq, int far_excited) {
    const int hop = a->hop_size, K = a->n_freqs;
    if (a->has_shadow) {
        /* Both |error_spec|^2 arrays already exist (scr_e2ref_arr,
         * e2_coarse_per_bin). Only the arrays are shared: error energy uses a
         * tail-fold sum that rounds differently from the pairwise sums. */
        float main_err   = sk_pairwise_sum_tailfold_f32(a->scr_e2ref_arr, (size_t)K);
        float shadow_err = sk_pairwise_sum_tailfold_f32(a->main_filter.e2_coarse_per_bin, (size_t)K);
        float as = a->cfg.shadow_err_alpha, oas = 1.0f - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;
        doubletalk_update_shadow_dt(&a->dt_analyzer, aec_count_as_int(a->shadow_frame_count),
                                    far_excited, a->main_err_smooth, a->shadow_err_smooth);
        int delay_reliable = a->has_delay && (delay_aec_confidence(&a->delay) >= 0.5f);
        /* Shadow-copy warm-up is a 500 ms wall-clock guard, converted per grid. */
        int shadow_warmup_hops =
            aec_ms_to_hops(500.0f, hop, a->cfg.sample_rate);
        ShadowCopyDecision dec = shadow_copy_update(
            &a->regime, aec_count_as_int(a->shadow_frame_count), shadow_warmup_hops,
            far_hop_mean_sq,
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
        if (dec.boost_q) filter_q_high(&a->main_filter);
    }

}

/* Echo-path-change recovery shared by the EPV and shadow_rise triggers:
 * aggressive Kalman re-adapt, mark_diverged, ERLE-side reset. */
static void aec_epc_gain_change_recovery(Aec* a) {
    a->pending_gain_change = 1;
    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
    filter_q_high(&a->main_filter);
    filter_convergence_mark_diverged(&a->convergence);
    a->epc_render_forced_remaining = a->cfg.epc_hangover;
    if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
}

static void aec_update_echo_path(Aec* a, float far_pwr_global) {
    /* EPV trigger. */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                  a->convergence.converged, a->regime.main_paused);
    if (epv.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
        /* Soft: EPV recovery is double-talk-blind; while near-end is recently
         * present the aggressive re-adapt, mark_diverged and ERLE reset are
         * skipped. */
    } else if (epv.fired) {
        aec_epc_gain_change_recovery(a);
    }

    if (a->has_shadow && a->convergence.converged) {
        EpcEvent rise = epc_update_shadow_rise(&a->epc, a->main_err_smooth,
                                               a->shadow_err_smooth,
                                               a->render_activity.is_stationary);
        if (rise.fired && a->cfg.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
        } else if (rise.fired) {
            aec_epc_gain_change_recovery(a);
        } else {
            epc_tick_hangover(&a->epc);
        }
    }

    /* EPC render-forced countdown, decremented every hop (diagnostic). */
    if (a->epc_render_forced_remaining > 0) a->epc_render_forced_remaining--;

}

static float aec_update_misadjustment_erle(Aec* a) {
    const int hop = a->hop_size;
    /* final_out is seeded from raw_output only when aec_post_run does not set
     * every sample. */
    if (!(a->cfg.enable_res || a->cfg.return_res_context))
        memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));

    misadj_update(a, a->near_hop, a->raw_output, hop);
    misadj_fire(a);

    /* inst-ERLE slope ring: appended every freq-path hop outside the
     * enable_res gate (the warm-transfer gate needs it with RES off); uses
     * the prior hop's EMAs. */
    float _ei;
    if (a->near_power < 1e-10f && a->raw_error_power < 1e-10f) _ei = 0.0f;
    else _ei = 10.0f * log10f((a->near_power + 1e-10f) / (a->raw_error_power + 1e-10f));
    {
        int _h = a->erle_slope_head;
        a->erle_slope_buf[_h] = _ei;
        a->erle_slope_head = (_h + 1) % a->erle_slope_cap;
        if (a->erle_slope_len < a->erle_slope_cap) a->erle_slope_len++;
    }

    return _ei;
}

static float aec_update_windowed_erle(Aec* a) {
    /* Windowed ERLE feeds the judgment, the already-cancelling guard, the
     * duty watchdog and divergence tracking, so it updates regardless of
     * enable_res. */
    const float erle_decay = 0.999f;
    a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
    a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
    float erle_windowed = 10.0f * log10f((a->erle_window_near + 1e-10f)
                                        / (a->erle_window_err + 1e-10f));
    a->last_erle_windowed = erle_windowed;
    filter_convergence_update_divergence(&a->convergence, a->near_power, a->raw_error_power);
    return erle_windowed;
}

static void aec_run_post_chain(Aec* a, int stationarity_block_for_post, int render_active) {
    const int hop = a->hop_size, N = a->n_partitions;
    {
        /* filter_taps_full is filled on demand by aec_fill_filter_taps(). */

        AecPostRunIn in;
        memset(&in, 0, sizeof(in));
        in.near_spec = a->main_filter.base.near_spec;
        in.far_spec  = a->main_filter.base.far_spec;
        in.echo_spec = a->main_filter.base.echo_spec;
        in.error_spec_windowed = a->main_filter.base.error_spec_windowed;
        in.W0 = a->main_filter.base.W;
        in.W_all = a->main_filter.base.W;
        in.X_buf = a->main_filter.base.X_buf;
        in.x2_cache = a->main_filter.base.x2_cache;
        in.sqrt_hann = a->main_filter.base.sqrt_hann;
        in.kalman_P = NULL; in.kalman_P_len = 0;
        in.partition_idx = a->main_filter.base.partition_idx;
        in.n_partitions = N;
        in.filter_taps_full = a->filter_taps_full;
        in.filter_taps_full_len = N * hop;
        in.shadow_present = a->has_shadow;
        in.shadow_error_spec = a->has_shadow ? a->shadow_filter.error_spec : NULL;
        in.last_shadow_output_time = a->has_shadow ? a->shadow_out : NULL;
        in.s_ref_max = a->main_filter.base.last_s_max_abs * 32768.0f;
        in.s_coa_max = a->has_shadow ? a->shadow_filter.last_s_max_abs * 32768.0f : 0.0f;
        /* main_error_energy / shadow_error_energy stay zero (unread). */
        in.raw_output = a->raw_output;
        in.near_end = a->near_hop;
        in.far_end = a->far_hop;
        in.current_delay = a->current_delay;
        /* delay_active means a reference ring exists (MATCHED, FIXED), not an estimator. */
        in.delay_active = aec_ring_active(&a->cfg);
        in.saturation_level = a->saturation_level;
        in.pending_gain_change = a->pending_gain_change;
        in.pending_delay_change = a->pending_delay_change;
        in.stationarity_active_hops = a->stationarity_active_hops;
        in.stationarity_converge_hops = a->stationarity_converge_hops;
        in.stationary_block = stationarity_block_for_post;
        in.erle_coh_gate_enabled = AECB_ERLE_COH_GATE_ENABLED;
        in.use_stationarity_properties = AECB_USE_STATIONARITY_PROPERTIES;
        /* Context-only seam: the caller consumes comfort noise and R^2 (and
         * the gain unless spatial_linear_context); only final synthesis is
         * skipped. */
        in.context_only = (!a->cfg.enable_res && a->cfg.return_res_context);
        /* spatial_linear_context implies context_only (validated): res_gain
         * is never read. */
        in.spatial_linear_context = a->cfg.spatial_linear_context != 0;
        in.linear_context_only =
            a->cfg.spatial_linear_context == AEC_CONTEXT_LINEAR_ONLY;
        in.active_render_threshold = AECB_ACTIVE_RENDER_THRESHOLD;

        AecPostRunObj obj;
        obj.state = &a->a3_state; obj.ree = &a->a3_ree; obj.sg = &a->a3_sg;
        obj.stationarity = &a->a3_stat; obj.lfs = &a->a3_lfs; obj.fft = a->post_fft;

        /* DT-gated min-gain floor lift: aec_post_run owns the final decision. */
        in.dt_floor_enabled = a->cfg.dt_aware_res_floor_enabled;
        in.nearend_recent = a->ne_recent_frames > 0;

        /* Live observer: obj.observer_filter runs inside aec_post_run, which
         * knows this hop's dominant_ne / usable_linear verdicts. */
        in.render_activity_is_active = render_active;
        in.observer_far_end = (a->has_observer && a->observer_far_hop_valid)
                               ? a->observer_far_hop : NULL;
        obj.observer_filter = a->has_observer ? &a->observer_filter : NULL;

        int pgc = a->pending_gain_change, pdc = a->pending_delay_change;
        aec_post_run(&a->post, &in, &obj, &a->a3_sc, a->final_out, &pgc, &pdc);
        a->pending_gain_change = pgc;
        a->pending_delay_change = pdc;

        /* In context-only mode aec_post_run emits raw_output. */
    }

}

static int aec_prepare_post_dt(Aec* a, float far_hop_mean_sq, float _ei,
                               float erle_windowed, int render_active) {
    const int hop = a->hop_size, K = a->n_freqs;
    int is_stationary_dt = 0;
    float dt_indicator = 0.0f, far_power = 0.0f;
    far_power = far_hop_mean_sq;
    /* erle_for_factor = max(inst ERLE, windowed ERLE); _ei is the same
     * expression. */
    float erle_inst = _ei;
    float erle_for_factor = erle_inst > erle_windowed ? erle_inst : erle_windowed;
    float erle_factor = erle_for_factor / 10.0f;
    if (erle_factor < 0.0f) erle_factor = 0.0f;
    if (erle_factor > 1.0f) erle_factor = 1.0f;
    a->erle_factor_prev = erle_factor;

    float far_pwr = far_power + 1e-10f;
    float mic_pwr = mean_sq(a->near_hop, hop, a->scr_sq) + 1e-10f;
    float raw_err_pwr = mean_sq(a->raw_output, hop, a->scr_sq) + 1e-10f;

    advance_erl_and_dt(a, render_active, far_pwr, mic_pwr, raw_err_pwr);

    float simple_dt = 1.0f - far_pwr / (mic_pwr + far_pwr);
    float raw_dt = a->dt_analyzer.dt_from_energy;
    float sd_half = simple_dt * 0.5f;
    if (sd_half > raw_dt) raw_dt = sd_half;

    if (a->render_activity.is_stationary && a->convergence.converged) {
        int fft_size = a->main_filter.base.fft_size;
        float freq_per_bin = (float)a->cfg.sample_rate / (float)fft_size;
        int vb_start = (int)(100.0f / freq_per_bin); if (vb_start < 1) vb_start = 1;
        int vb_limit = (int)(3000.0f / freq_per_bin); if (vb_limit > K) vb_limit = K;
        float track_err_pwr = 0.0f;
        if (a->has_shadow) {
            /* Reuse cmag2 of main error_spec from scr_e2ref_arr (filled in the main-filter update). */
            for (int k = vb_start; k < vb_limit; ++k) track_err_pwr += a->scr_e2ref_arr[k];
        } else {
            /* Serial reduction in k order (scr_e2ref_arr is free without a
             * shadow). */
            float *terr_scr = a->scr_e2ref_arr;
            sk_cmag2_np_f32(a->main_filter.base.error_spec + vb_start,
                             terr_scr + vb_start, vb_limit - vb_start);
            for (int k = vb_start; k < vb_limit; ++k) track_err_pwr += terr_scr[k];
        }
        track_err_pwr += 1e-10f;
        if (a->wn_err_baseline < 1e-6f) a->wn_err_baseline = track_err_pwr;
        float jump_ratio = track_err_pwr / (a->wn_err_baseline + 1e-10f);
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
    if (a->convergence.converged && !a->render_activity.is_stationary
            && far_pwr > 1e-4f && a->wn_err_baseline > 1e-6f) {
        a->wn_err_baseline = 0.995f * a->wn_err_baseline + 0.005f * raw_err_pwr;
    }

    float inst_erle_fast_raw = mic_pwr / raw_err_pwr;
    a->inst_erle_smooth = 0.7f * a->inst_erle_smooth + 0.3f * inst_erle_fast_raw;
    if (a->inst_erle_smooth > 2.0f) {
        float erle_for_dt = a->inst_erle_smooth;
        if (erle_for_dt > 4.0f) erle_for_dt = 4.0f;
        raw_dt /= erle_for_dt;
    }
    if (a->epc.active) { raw_dt = 0.0f; is_stationary_dt = 0; }
    dt_indicator = raw_dt; if (dt_indicator < 0.0f) dt_indicator = 0.0f;
    if (dt_indicator > 0.8f) dt_indicator = 0.8f;

    /* shadow_dt stash for context telemetry (not used by the in-tree RES). */
    float shadow_dt_v = a->dt_analyzer.dt_from_energy;
    if (a->dt_analyzer.dt_from_shadow > shadow_dt_v) shadow_dt_v = a->dt_analyzer.dt_from_shadow;
    if (a->epc.active) shadow_dt_v *= 0.08f;
    a->last_far_power = far_power;
    a->last_shadow_dt = shadow_dt_v;
    a->last_is_stationary_dt = is_stationary_dt;
    a->last_dt_indicator = dt_indicator;

    return is_stationary_dt;
}

static void aec_update_post_mu(Aec* a, int is_stationary_dt) {
    const int hop = a->hop_size, K = a->n_freqs;
    if (a->convergence.converged) {
        float mu_min = a->cfg.shadow_mu_min;
        for (int k = 0; k < K; ++k) a->per_bin_mu_scale[k] = mu_min;
        a->simple_mu_ratio = 0.0f;
        if (is_stationary_dt) {
            a->simple_mu_ratio = mu_min;
        }
        a->has_per_bin_mu = 1;
    } else {
        a->has_per_bin_mu = 0;
        update_simple_mu_ratio(a, a->raw_output, a->far_hop, hop);
    }
}

static void aec_update_linear_only_steering(Aec* a, float far_hop_mean_sq, int render_active) {
    const int hop = a->hop_size;
    /* Mu ratio, converged mu freeze and ERL/DT trackers must advance even
     * without the RES gate; has_per_bin_mu is not cleared here. */
    if (!a->cfg.enable_res && !a->cfg.return_res_context) {
        if (a->convergence.converged) {
            int k;
            float mu_min = a->cfg.shadow_mu_min;
            for (k = 0; k < a->n_freqs; ++k) a->per_bin_mu_scale[k] = mu_min;
            a->has_per_bin_mu = 1;
            a->simple_mu_ratio = 0.0f;
        } else {
            update_simple_mu_ratio(a, a->raw_output, a->far_hop, hop);
        }

        {
            float far_pwr = far_hop_mean_sq + 1e-10f;
            float mic_pwr = mean_sq(a->near_hop, hop, a->scr_sq) + 1e-10f;
            float raw_err_pwr = mean_sq(a->raw_output, hop, a->scr_sq) + 1e-10f;
            advance_erl_and_dt(a, render_active, far_pwr, mic_pwr, raw_err_pwr);
        }
    }

}

static void aec_finish_hop(Aec* a, float far_hop_mean_sq) {
    const int hop = a->hop_size;
    {
        const float ap = a->alpha_pow, oap = 1.0f - a->alpha_pow;
        for (int i = 0; i < hop; ++i) {
            float n = a->near_hop[i], r = a->raw_output[i];
            a->near_power      = ap * a->near_power      + oap * (n * n);
            a->raw_error_power = ap * a->raw_error_power + oap * (r * r);
        }
    }

    {
        int far_active = (far_hop_mean_sq > 1e-4f);
        int just_converged = filter_convergence_update_convergence(
            &a->convergence, a->near_power, a->raw_error_power,
            far_active, a->warmup_frames_remaining <= 0);
        if (just_converged) {
            int Kk = a->main_filter.base.n_freqs;
            for (int k = 0; k < Kk; ++k) a->main_filter.Q[k] = a->main_filter.Q_low[k];
        }
    }

    aec_counter_bump(&a->frame_count);
}

static void aec_process_core(Aec* a, const float* mic_in, const float* ref_in,
                              const Complex* shared_far_spec,
                              const float* shared_far_cmag2) {
    const int hop = a->hop_size;
    /* Stage stamps: five windows, seven clock reads; t_delay is subtracted
     * from the frontend window. */
    uint32_t t_stage, t_mark, t_delay;
    a->delay_gen_hop_start = a->delay_generation;
    a->far_hop_aligned = 0;
    a->last_timing.delay_us = 0;
    a->last_timing.frontend_us = 0;
    a->last_timing.linear_us = 0;
    a->last_timing.res_us = 0;
    a->last_timing.steering_us = 0;
    t_stage = aec_now_us();
    aec_prepare_capture_hop(a, mic_in, ref_in);

    t_delay = aec_now_us();
    aec_align_reference(a);

    t_delay = aec_now_us() - t_delay;
    a->last_timing.delay_us = t_delay;

    float far_hop_mean_sq = mean_sq(a->far_hop, hop, a->scr_sq);

    RenderActivityResult ra = render_activity_update(&a->render_activity, a->far_hop, hop);
    a->warmup_far_active = ra.warmup_active;
    float far_pwr_global = ra.far_pwr;

    int mu_is_array;
    float mu_scalar = aec_prepare_mu(a, &mu_is_array);

    int rsa_mask_active = aec_prepare_rsa(a);

    int far_excited = aec_run_shadow(a, far_hop_mean_sq, rsa_mask_active,
                                     shared_far_spec, shared_far_cmag2);

    t_mark = aec_now_us();
    /* Frontend = pre-filter window - delay_us (t_delay is the local so it
     * folds when timing is off). */
    a->last_timing.frontend_us = (t_mark - t_stage) - t_delay;

    int main_error_cmag2_valid = aec_run_main_filter(
        a, mu_scalar, mu_is_array, rsa_mask_active, shared_far_spec, shared_far_cmag2);

    {
        uint32_t t_lin_end = aec_now_us();
        a->last_timing.linear_us = t_lin_end - t_mark;
        t_mark = t_lin_end;
    }

    int stationarity_block_for_post = aec_refresh_stationarity(a);

    aec_update_shadow_quality(a, main_error_cmag2_valid);

    aec_update_shadow_dt(a, far_hop_mean_sq, far_excited);

    aec_update_echo_path(a, far_pwr_global);

    float _ei = aec_update_misadjustment_erle(a);

    /* RES / post block; one stamp closes the steering window. */
    t_stage = aec_now_us();
    a->last_timing.steering_us = t_stage - t_mark;
    int is_stationary_dt = 0;
    float erle_windowed = aec_update_windowed_erle(a);
    /* Run the post-filter block when enable_res or return_res_context is set. */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        is_stationary_dt = aec_prepare_post_dt(a, far_hop_mean_sq, _ei, erle_windowed, ra.is_active);
        aec_run_post_chain(a, stationarity_block_for_post, ra.is_active);

        aec_update_post_mu(a, is_stationary_dt);
    }
    aec_update_linear_only_steering(a, far_hop_mean_sq, ra.is_active);

    a->last_timing.res_us = aec_now_us() - t_stage;

    aec_finish_hop(a, far_hop_mean_sq);
}

/* Processes exactly hop_size samples: aec_process_core() plus the copy to
 * out; no output limiter. */
void aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    const int hop = a->hop_size;
    aec_process_core(a, mic_in, ref_in, NULL, NULL);

    /* Debug trace: runs only when a trace file is armed; compiled out under
     * AEC_NO_STDIO. */
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
        tr.filter_converged = a->post.trace.aec_converged;
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

    memcpy(out, a->final_out, (size_t)hop * sizeof(float));
}

/* Context-only entry point: aec_process_core() without the output copy. All
 * entry points share state and may be interleaved. */
void aec_process_context(Aec* a, const float* mic_in, const float* ref_in) {
    aec_process_core(a, mic_in, ref_in, NULL, NULL);
}

/* Far-end-FFT-sharing entry point: shared_far_spec (n_freqs) replaces this
 * instance's far-end FFT; it must equal the computing instance's
 * aec_get_res_context() far_spec for the same ref_in, else the result is
 * silently wrong. */
void aec_process_context_shared_far(
        Aec* a, const float* mic_in, const float* ref_in,
        const Complex* shared_far_spec) {
    aec_process_core(a, mic_in, ref_in, shared_far_spec, NULL);
}

void aec_process_context_shared_far_from(
        Aec* a, const float* mic_in, const float* ref_in, const Aec* far_owner) {
    if (!a) return;
    if (!far_owner || far_owner == a || far_owner->n_freqs != a->n_freqs ||
        far_owner->hop_size != a->hop_size ||
        far_owner->cfg.sample_rate != a->cfg.sample_rate) {
        aec_process_core(a, mic_in, ref_in, NULL, NULL);
        return;
    }
    aec_process_core(a, mic_in, ref_in, far_owner->main_filter.base.far_spec,
                     far_owner->main_filter.base.far_cmag2_hold);
}

/* Streaming API: render-hop FIFO around aec_process(); in lockstep it is
 * pass-through and equals aec_process(). SPSC ring with drop-new and consumer
 * catch-up: fifo_write is written only by the render thread, fifo_read only
 * by the capture thread, via acquire/release `__atomic_*_n`. Overrun drops
 * the incoming hop; a full ring makes the consumer skip to the freshest hop;
 * both report AEC_BUF_RENDER_OVERRUN. */
AecBufferingEvent aec_analyze_render(Aec* a, const float* ref) {
    const int hop = a->hop_size;
    const unsigned cap = (unsigned)a->fifo_cap_hops;
    aec_counter_bump(&a->render_call_count);

    /* fifo_write: own field (plain load); fifo_read: acquire, to see the
     * consumer's claim. */
    unsigned w = *(unsigned*)&a->fifo_write;
    unsigned r = __atomic_load_n((unsigned*)&a->fifo_read, __ATOMIC_ACQUIRE);
    if (w - r >= cap)
        return AEC_BUF_RENDER_OVERRUN;

    memcpy(a->render_fifo + (size_t)(w % cap) * (size_t)hop, ref, (size_t)hop * sizeof(float));
    /* Publish after the memcpy (release pairs with the consumer's acquire). */
    __atomic_store_n((unsigned*)&a->fifo_write, w + 1u, __ATOMIC_RELEASE);
    return AEC_BUF_NONE;
}

AecBufferingEvent aec_process_capture(Aec* a, const float* mic, float* out) {
    const int hop = a->hop_size;
    const unsigned cap = (unsigned)a->fifo_cap_hops;
    aec_counter_bump(&a->capture_call_count);
    AecBufferingEvent ev;
    const float* ref;
    int claimed;

    /* fifo_read: own field (plain load); fifo_write: acquire, to see the
     * payload. */
    unsigned r = *(unsigned*)&a->fifo_read;
    unsigned w = __atomic_load_n((unsigned*)&a->fifo_write, __ATOMIC_ACQUIRE);

    if (w == r) {
        ref = a->fifo_zero_ref;
        ev = AEC_BUF_RENDER_UNDERRUN;
        claimed = 0;
    } else if (w - r == cap) {
        r = w - 1u;
        ref = a->render_fifo + (size_t)(r % cap) * (size_t)hop;
        ev = AEC_BUF_RENDER_OVERRUN;
        claimed = 1;
    } else {
        ref = a->render_fifo + (size_t)(r % cap) * (size_t)hop;
        claimed = 1;
        ev = AEC_BUF_NONE;
    }

    aec_process(a, mic, ref, out);
    /* Publish the read cursor after aec_process() returns (it copies `ref`
     * out first). */
    if (claimed)
        __atomic_store_n((unsigned*)&a->fifo_read, r + 1u, __ATOMIC_RELEASE);
    /* Relaxed atomic store so a polling third thread cannot read a torn value. */
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
    /* Valid when the post-filter block ran; pointers alias aec_post_run
     * buffers. res_gain is NULL under spatial_linear_context (never
     * computed); all are NULL when neither RES nor the context seam is on. */
    if (a->cfg.enable_res || a->cfg.return_res_context) {
        ctx->error_spec    = a->a3_sc.sel_esw;
        ctx->echo_spec     = a->a3_sc.sel_echo;
        ctx->near_spec     = a->a3_sc.ybase;
        ctx->formed_hop    = a->a3_lfs.e_form;
        if (a->cfg.spatial_linear_context == AEC_CONTEXT_LINEAR_ONLY) {
            ctx->res_gain      = NULL;
            ctx->r2            = NULL;
            ctx->comfort_noise = NULL;
            ctx->res_floor_protect = 0;
        } else {
            ctx->res_gain      = a->cfg.spatial_linear_context ? NULL : a->a3_sg.gain;
            ctx->r2            = a->a3_sc.r2;
            ctx->comfort_noise = a->post.comfort_noise;
            ctx->res_floor_protect = a->a3_sg.dt_protect_active;
        }
        ctx->usable_linear = aec_state_usable_linear_estimate(&a->a3_state);
    } else {
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

/* Alignment confidence shared by AecDebugStatus and AecLinearContext: MATCHED
 * reports the estimator's value; FIXED and EXTERNAL_ALIGNED report 1.0
 * (caller contract). */
static float aec_delay_confidence(const Aec* a) {
    if (a->cfg.delay_mode == AEC_DELAY_MATCHED)
        return delay_aec_confidence(&a->delay);
    return 1.0f;
}

/* Read-only AecDebugStatus snapshot. */
void aec_debug_status(const Aec* a, AecDebugStatus* out) {
    if (!a || !out) return;
    memset(out, 0, sizeof(*out));

    out->delay_samples    = a->current_delay;
    out->delay_confidence = aec_delay_confidence(a);
    out->delay_updates    = a->has_delay ? delay_aec_n_updates(&a->delay) : 0;

    out->erle_windowed_db = a->last_erle_windowed;
    out->usable_linear    = aec_state_usable_linear_estimate(&a->a3_state);
    out->filter_converged = a->convergence.converged;

    out->near_power = a->near_power;
    out->out_power  = a->raw_error_power;

    out->duty_hops_total = a->duty_hops_total;
    out->duty_hops_run   = a->duty_hops_run;
}

/* Read-only AecLinearContext view; pointers alias per-hop buffers, valid
 * until the next process/reset. */
void aec_get_linear_context(const Aec* a, AecLinearContext* ctx) {
    if (!a || !ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->hop_size = a->hop_size;
    ctx->formed_linear_hop = (a->cfg.enable_res || a->cfg.return_res_context)
                             ? a->a3_lfs.e_form : a->raw_output;
    ctx->aligned_far_hop = a->far_hop;
    ctx->generation = a->delay_generation;
    /* Per-mode semantics: see AecLinearContext in aec.h. */
    ctx->delay_confidence = aec_delay_confidence(a);
    if (a->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        ctx->delay_samples = 0;
        ctx->delay_state = AEC_LINEAR_DELAY_LOCKED;
        return;
    }
    if (a->cfg.delay_mode == AEC_DELAY_FIXED) {
        ctx->delay_samples = a->current_delay;
        /* far_hop is raw until the ring can serve the offset, so report
         * UNLOCKED then; CHANGED cannot occur. */
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
