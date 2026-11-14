/* Top-level PBFDKF AEC orchestration with the post-filter chain. */
#include "aec.h"
#include "aec_impl.h"
#include "aec_internal.h"
#include "aec_config_build.h"
#include "aec_balanced_config.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"
#include "utility/mem_align.h"   /* ALIGN16 */
#include "utility/simd_kernels.h"
#include "utility/simd_kernel_nn.h"  /* skn_all_finite_f32 */
#include "utility/fast_math.h"
#include "aec_array_internal.h"      /* array dispatch */
#include "aec_res_internal.h"
#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* DelayAdjustment value stored in pending_delay_change. */
#define AEC_DA_NEW_DETECTED  2

/* Array dispatch. -DAEC_NO_ARRAY builds the single-channel library: no instance is an array, and src/aec_array.c is not part
 * of it (see aec_array_internal.h). */
#ifdef AEC_NO_ARRAY
#define AEC_IS_ARRAY(a)         0
#define AEC_CFG_IS_ARRAY(cfg)   0
#else
#define AEC_IS_ARRAY(a)         ((a)->array != NULL)
#define AEC_CFG_IS_ARRAY(cfg)   ((cfg)->n_mics == AEC_ARRAY_MICS)
#endif

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

/* Level strength table in AecLevel order; NO_FLOOR repeats balanced's
 * (unused) values. */
typedef struct { float far_active_db, dt_db; int split_floor_enabled; } AecLevelFloors;
static const AecLevelFloors AEC_LEVEL_FLOORS[] = {
    [AEC_LEVEL_MILD]       = { -20.0f, -20.0f, 1 },
    [AEC_LEVEL_BALANCED]   = { -28.0f, -20.0f, 1 },
    [AEC_LEVEL_AGGRESSIVE] = { -38.0f, -30.0f, 1 },
    [AEC_LEVEL_NO_FLOOR]   = { -28.0f, -20.0f, 0 },
};

/* Row for an in-enum level, NULL otherwise (every lookup below then fails). */
static const AecLevelFloors* aec_level_row(AecLevel p) {
    return ((unsigned)p < sizeof(AEC_LEVEL_FLOORS) / sizeof(AEC_LEVEL_FLOORS[0]))
               ? &AEC_LEVEL_FLOORS[p] : NULL;
}

const char* aec_level_name(AecLevel p) {
    static const char* const names[] = {
        [AEC_LEVEL_MILD]       = "mild",
        [AEC_LEVEL_BALANCED]   = "balanced",
        [AEC_LEVEL_AGGRESSIVE] = "aggressive",
        [AEC_LEVEL_NO_FLOOR]   = "no_floor",
    };
    return aec_level_row(p) ? names[p] : "balanced";
}

int aec_level_floor_db(AecLevel p, float* db_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->far_active_db;
    return 0;
}

int aec_level_dt_floor_db(AecLevel p, float* db_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (db_out == NULL || r == NULL) return -1;
    *db_out = r->dt_db;
    return 0;
}

int aec_level_split_floor_enabled(AecLevel p, int* enabled_out) {
    const AecLevelFloors* r = aec_level_row(p);
    if (enabled_out == NULL || r == NULL) return -1;
    *enabled_out = r->split_floor_enabled;
    return 0;
}

void aec_config_from_level(AecConfig* cfg, AecLevel p, int sr) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->sample_rate = sr;
    /* The only place a default fft_size is applied (0 for an unsupported rate, which the validator rejects). */
    cfg->fft_size = aec_default_fft_size(sr);
    cfg->n_mics = 1;
    cfg->level = aec_level_row(p) ? p : AEC_LEVEL_BALANCED;
    cfg->residual = AEC_RESIDUAL_FULL;
    /* MATCHED with 5 filters is the default; fixed_delay_samples must be -1 (0 is a legal FIXED delay). */
    cfg->delay_mode = AEC_DELAY_MATCHED;
    cfg->fixed_delay_samples = -1;
    cfg->delay_num_filters = AEC_DELAY_NUM_FILTERS_MAX;
    cfg->enable_cng = 1;
    cfg->capture_proxy_channel = 0;
    cfg->delay_backward_quarantine_enabled = 0;
    cfg->delay_backward_quarantine_s = 1.0f;
}

/* Signal-grid table: admissible (sample_rate, fft_size) pairs and their
 * geometry. */
static const AecSignalGrid AEC_GRID_TABLE[] = {
    { 16000,  256,   256,  128,     129, 0 },
    { 16000,  512,   512,  256,     257, 0 },
    { 48000, 1024,  1024,  512,     513, 0 },
    {  8000,  256,   256,  128,     129, 1 },
};

/* First table row for sample_rate whose fft_size matches (fft_size <= 0: any), or NULL. */
static const AecSignalGrid* aec_find_grid(int sample_rate, int fft_size) {
    for (size_t i = 0; i < sizeof(AEC_GRID_TABLE) / sizeof(AEC_GRID_TABLE[0]); ++i) {
        if (AEC_GRID_TABLE[i].sample_rate == sample_rate &&
            (fft_size <= 0 || AEC_GRID_TABLE[i].fft_size == fft_size))
            return &AEC_GRID_TABLE[i];
    }
    return NULL;
}

int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out) {
    const AecSignalGrid* g;
    if (fft_size <= 0) return AEC_ERR_ARG;
    g = aec_find_grid(sample_rate, fft_size);
    if (!g) return AEC_ERR_ARG;
    if (out) *out = *g;
    return AEC_OK;
}

int aec_is_valid_sample_rate(int sample_rate) {
    return aec_find_grid(sample_rate, 0) != NULL;
}

int aec_default_fft_size(int sample_rate) {
    const AecSignalGrid* g = aec_find_grid(sample_rate, 0);
    return g ? g->fft_size : 0;
}

/* True when a reference ring exists (MATCHED, FIXED); Aec::has_delay is the
 * MATCHED-only estimator. */
static inline int aec_ring_active(const AecConfig* cfg) {
    return cfg->delay_mode != AEC_DELAY_EXTERNAL_ALIGNED;
}

/* The caller's fields, the tuning, and the delay-mode / array rules. 1 = acceptable. */
static int aec_resolved_valid(const AecResolved* r) {
    const AecConfig* cfg = &r->cfg;
    if (aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, NULL) != AEC_OK) return 0;
    if (cfg->enable_cng != 0 && cfg->enable_cng != 1) return 0;
    if ((unsigned)cfg->level > (unsigned)AEC_LEVEL_NO_FLOOR) return 0;
    if (cfg->residual < AEC_RESIDUAL_OFF || cfg->residual > AEC_RESIDUAL_FULL) return 0;
    /* Hard bound: AEC_DELAY_NUM_FILTERS_MAX is the static array extent; 0 filters is invalid. */
    if (cfg->delay_num_filters < 1 || cfg->delay_num_filters > AEC_DELAY_NUM_FILTERS_MAX) return 0;
    if (!aec_tuning_valid(&r->tun, cfg)) return 0;

    /* Delay mode x field compatibility: MATCHED and EXTERNAL_ALIGNED need fixed_delay_samples == -1; delay_num_filters must stay
     * at its default outside MATCHED. */
    switch (cfg->delay_mode) {
        case AEC_DELAY_MATCHED:
            if (cfg->fixed_delay_samples != -1) return 0;
            break;
        case AEC_DELAY_FIXED:
            if (cfg->fixed_delay_samples < 0) return 0;
            /* Bound on the reference-ring size (fixed_delay_samples + hop): 120 s. */
            if ((long)cfg->fixed_delay_samples > 120L * (long)cfg->sample_rate) return 0;
            if (cfg->delay_num_filters != AEC_DELAY_NUM_FILTERS_MAX) return 0;
            break;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            if (cfg->fixed_delay_samples != -1) return 0;
            if (cfg->delay_num_filters != AEC_DELAY_NUM_FILTERS_MAX) return 0;
            break;
        default:
            return 0;
    }

    /* The quarantine fields are inert on a single-channel instance, which must not request them. */
    if (cfg->delay_backward_quarantine_enabled != 0 && cfg->delay_backward_quarantine_enabled != 1) return 0;
    if (!isfinite(cfg->delay_backward_quarantine_s) || cfg->delay_backward_quarantine_s < 0.0f ||
        cfg->delay_backward_quarantine_s > 3600.0f) return 0;
    /* n_mics, capture_proxy_channel and the array instance's own rules. */
#ifdef AEC_NO_ARRAY
    return cfg->n_mics == 1 && cfg->capture_proxy_channel == 0 && cfg->delay_backward_quarantine_enabled == 0;
#else
    return aec_array_impl_config_ok(r);
#endif
}

int aec_resolve_tuned(const AecConfig* in, const AecTuning* tun, AecResolved* out) {
    AecResolved r;
    if (!in || !out) return 0;
    r.cfg = *in;
    if (tun) r.tun = *tun; else aec_tuning_defaults(&r.tun, in);
    if (!aec_resolved_valid(&r)) return 0;
    *out = r;
    return 1;
}

int aec_resolve(const AecConfig* in, AecResolved* out) {
    return aec_resolve_tuned(in, NULL, out);
}

/* ───────────────────────── helpers ─────────────────────────────────────── */

/* Mean of x^2 in float32. scratch holds >= n floats. */
static float mean_sq(const float* x, int n, float* scratch) {
    sk_sq_scale_f32(x, 1.0f, scratch, n);
    float s = sk_pairwise_sum_f32(scratch, (size_t)n);
    return s / (float)n;
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

    int half = a->tun.warmup_frames / 2;
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
 * recovery chain for a new, unverified delay (Path B hard reset). A bad-relock
 * revert uses aec_apply_delay_shift_soft_revert(). */
static void aec_apply_delay_shift_hard_reset(Aec* a, int target_delay) {
    a->current_delay = target_delay;
    aec_reset_filter_derived_state(a);
    epc_force_delay(&a->epc);
    if (a->has_shadow) aec_rearm_excitation(&a->shadow_filter);
    if (a->has_observer) aec_rearm_excitation(&a->observer_filter);
    filter_convergence_mark_diverged(&a->convergence);
    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
    a->pending_delay_change = AEC_DA_NEW_DETECTED;
}

/* Bad-relock revert to target_delay (the pre-shift delay) with the tap / H_error
 * snapshot taken at the shift, without the hard-reset chain: the post chain
 * (OLA, CNG, AecState, suppressor), EPC and convergence state are kept,
 * so aec_post_reset_serial() does not move. A retard (target < bad) clears the
 * far history, an advance keeps it (the pbfdaf_warm_shift_ir sign rule). The
 * windowed-ERLE accumulators, the main / shadow error EMAs and any pending delay
 * are cleared. */
static void aec_apply_delay_shift_soft_revert(Aec* a, int target_delay,
                                              int bad_delay) {
    size_t w_elems = (size_t)a->n_partitions * (size_t)a->n_freqs;
    a->current_delay = target_delay;
    memcpy(a->main_filter.base.W, a->delay_shift_judgment_pre_W,
           w_elems * sizeof(Complex));
    a->main_filter.base.w_dirty = 1;
    memcpy(a->main_filter.H_error_per_bin, a->delay_shift_judgment_pre_H_error,
           (size_t)a->n_freqs * sizeof(float));
    if (a->has_shadow && a->delay_shift_judgment_pre_shadow_W != NULL) {
        memcpy(a->shadow_filter.W, a->delay_shift_judgment_pre_shadow_W,
               w_elems * sizeof(Complex));
        a->shadow_filter.w_dirty = 1;
    }
    if (target_delay - bad_delay < 0) {
        pbfdaf_clear_far_history(&a->main_filter.base);
        if (a->has_shadow) pbfdaf_clear_far_history(&a->shadow_filter);
    }
    a->erle_window_near = 1e-10f;
    a->erle_window_err = 1e-10f;
    a->main_err_smooth = 0.0f;
    a->shadow_err_smooth = 0.0f;
    a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
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
    float mu_min = a->tun.shadow_mu_min;
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
    if (aec_unusable(a) || AEC_IS_ARRAY(a)) return;
    aec_reset_filter_latches(a);
}

float aec_testing_legacy10ms_rate(float legacy_per_hop, int hop,
                                  int sample_rate) {
    return aec_legacy10ms_rate(legacy_per_hop, hop, sample_rate);
}
#endif

/* ───────────────────────── construction ────────────────────────────────── */


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

static void aec_derive_dims(const AecResolved* rs,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp);
/* Carve options. lane = 1 builds a lane of the array flow (EXTERNAL_ALIGNED, residual below FULL): the pool omits what that
 * contract never runs and the lane borrows shared_fft. Both aec_lane_get_mem_size() and aec_lane_init() set it; the
 * single-channel Aec uses the zero value. */
typedef struct AecCarveOpt {
    int        lane;
    FftHandle* shared_fft;
} AecCarveOpt;

static int aec_carve(Aec* a, uint8_t* ptr, const AecResolved* rs,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, const AecCarveOpt* opt);
static size_t aec_mem_size_impl(const AecResolved* rs, int lane);
/* A STATE lane stops after the AecState update (linear_context_only): nothing reads its residual-echo estimator or its
 * comfort-noise arrays, so the carve omits them. */
static inline int aec_state_lane(const AecConfig* cfg, int lane) {
    return lane && cfg->residual == AEC_RESIDUAL_STATE;
}

static size_t aec_mem_size_resolved(const AecResolved* rs);
static Aec* aec_init_resolved(void* mem, size_t mem_size, const AecResolved* rs);

static Aec* aec_create_resolved(const AecResolved* rs) {
    void* block = NULL;
    Aec*  a;
    size_t size = aec_mem_size_resolved(rs);
    if (size == 0) return NULL;
    /* 16-byte base required by the carved FFT and static sub-objects. */
    if (posix_memalign(&block, 16u, size) != 0 || !block) return NULL;
    a = aec_init_resolved(block, size, rs);
    if (!a) { free(block); return NULL; }
    a->is_static = 0;
    a->heap_arena = block;
    return a;
}

Aec* aec_create(const AecConfig* cfg) {
    AecResolved rs;
    return aec_resolve(cfg, &rs) ? aec_create_resolved(&rs) : NULL;
}

/* ── reference alignment ring sizing ─────────────────────────────────────── */

/* Reference ring capacity in samples (shared by aec_get_mem_size() and
 * aec_carve()): MATCHED the estimator's largest reportable delay + hop_size (a
 * delay is eligible only up to capacity - hop_size); FIXED fixed_delay_samples
 * + hop_size; EXTERNAL_ALIGNED 0. Returns 0 for EXTERNAL_ALIGNED or an int
 * overflow. */
static int aec_ref_ring_samples(const AecResolved* rs, int hop) {
    const AecConfig* cfg = &rs->cfg;
    if (cfg->delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) return 0;
    if (hop <= 0) return 0;
    if (cfg->delay_mode == AEC_DELAY_FIXED) {
        long long need = (long long)cfg->fixed_delay_samples + (long long)hop;
        if (need < (long long)hop || need > 2147483647LL) return 0;
        return (int)need;
    }
    {
        int reach = delay_aec_max_delay_samples(cfg->sample_rate, cfg->delay_num_filters);
        return reach > 0 ? reach + hop : 0;
    }
}

/* ── dimension helper (shared by aec_create() and aec_init()) ─────────────── */

/* Frame dimensions: geometry from the shared resolver, partitions / ring
 * length from cfg; zero-filled on an unresolvable grid. */
static void aec_derive_dims(const AecResolved* rs,
                            int* o_hop, int* o_blk, int* o_fft, int* o_K,
                            int* o_nparts, int* o_buf_samp) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    AecSignalGrid g;
    if (aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &g) != AEC_OK) {
        *o_hop = 0; *o_blk = 0; *o_fft = 0; *o_K = 0;
        *o_nparts = 0; *o_buf_samp = 0;
        return;
    }
    int fft = g.fft_size;
    int blk = g.frame_size;
    int hop = g.hop_size;
    *o_hop = hop; *o_blk = blk; *o_fft = fft; *o_K = g.n_freqs;
    int n = tn->n_partitions;
    if (n <= 0) { n = (tn->filter_length + hop - 1) / hop; if (n < 1) n = 1; }
    *o_nparts = n;
    *o_buf_samp = aec_ref_ring_samples(rs, hop);
}

static size_t aec_mem_size_resolved(const AecResolved* rs) {
    if (AEC_CFG_IS_ARRAY(&rs->cfg)) return aec_array_impl_get_mem_size(rs);
    return aec_mem_size_impl(rs, 0);
}

size_t aec_get_mem_size(const AecConfig* cfg_in) {
    AecResolved rs;
    return aec_resolve(cfg_in, &rs) ? aec_mem_size_resolved(&rs) : 0;
}

#ifndef AEC_NO_ARRAY
/* The lane contract the lean carve relies on (cfg resolved). */
static int aec_lane_contract_ok(const AecConfig* cfg) {
    return cfg->n_mics == 1 && cfg->delay_mode == AEC_DELAY_EXTERNAL_ALIGNED &&
           (cfg->residual == AEC_RESIDUAL_STATE || cfg->residual == AEC_RESIDUAL_ESTIMATE);
}

size_t aec_lane_get_mem_size(const AecResolved* lane) {
    if (!lane || !aec_resolved_valid(lane) || !aec_lane_contract_ok(&lane->cfg)) return 0;
    return aec_mem_size_impl(lane, 1);
}
#endif

/* Pool bytes of a validated, resolved single-channel cfg; lane selects the lean lane carve. Field order must match
 * aec_carve(). */
static size_t aec_mem_size_impl(const AecResolved* rs, int lane) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    int hop, blk, fft, K, np, buf;
    aec_derive_dims(rs, &hop, &blk, &fft, &K, &np, &buf);

    const int ncc  = AECB_ST_NUM_CAPTURE_CHANNELS;
    const int state_lane = aec_state_lane(cfg, lane);
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
    t = ck_field_size(t, (size_t)(K - 2 > 0 ? K - 2 : 1), sizeof(int64_t));
    if (!lane) t = ck_add_size(t, fft_get_mem_size(fft));   /* a lane borrows the array's handle */
    {
        size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
        if (kf_sz == 0) return 0;
        t = ck_add_size(t, kf_sz);
    }
    if (tn->enable_shadow) {
        size_t af_sz = pbfdaf_get_mem_size_follower(blk, np, hop);
        if (af_sz == 0) return 0;
        t = ck_add_size(t, af_sz);
    }
    if (tn->live_observer_enabled) {
        size_t obs_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (obs_sz == 0) return 0;
        t = ck_add_size(t, obs_sz);
        t = ck_field_size(t, (size_t)hop, sizeof(float));   /* observer_far_hop */
        t = ck_field_size(t, (size_t)hop, sizeof(float));   /* a3_sc.observer_out (hop) */
        t = ck_field_size(t, Kz, sizeof(Complex));          /* a3_sc.observer_diff_spec */
        t = ck_field_size_reps(t, Kz, sizeof(float), 2);    /* a3_sc diff/cap magnitude^2 */
    }
    /* Pre-shift W snapshots for the bad-relock revert (shadow's only when
     * enable_shadow) and the main filter's H_error snapshot; only the MATCHED estimator takes them, so a lane has none. */
    if (!lane) {
        t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(Complex));
        if (tn->enable_shadow)
            t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(Complex));
        t = ck_field_size(t, Kz, sizeof(float));            /* pre-shift main H_error snapshot */
    }
    t = ck_field_size(t, (size_t)((hop + 7) / 8), sizeof(float));
    t = ck_field_size_reps(t, Kz, sizeof(float), state_lane ? 3 : 6);   /* avg_rev sye_re sye_im; y2s n2 n2i with the comfort noise */
    t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* syy see */
    if (!lane) t = ck_field_size(t, (size_t)blk, sizeof(float));  /* swin: the synthesis window, built by aec_init */
    if (!lane) t = ck_field_size(t, (size_t)blk, sizeof(float));  /* ola: only aec_post_apply_output() (FULL) uses it */
    t = ck_field_size_reps(t, Kz, sizeof(float), state_lane ? 7 : 8);   /* np_ fp_ ep_ erp_ cpe x2r nf; cn */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* cgm */
    if (!lane) {
        t = ck_field_size(t, Kz, sizeof(Complex));         /* eout */
        t = ck_field_size(t, (size_t)fft, sizeof(float));  /* eout_full */
    }
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
    if (!state_lane) {   /* the residual-echo estimator's arrays */
        t = ck_field_size(t, Kz, sizeof(float));           /* x2_nf */
        t = ck_field_size(t, Kz, sizeof(int));             /* x2_nf_c */
        t = ck_field_size_reps(t, Kz, sizeof(float), 2);   /* rm_st rt_st */
        if (!AECB_REE_USE_AEC_ECHO_GEN_WINDOW)   /* rh_st: read only without the echo-generating window */
            t = ck_field_size(t, ck_mul_size((size_t)rh, Kz), sizeof(float));
        t = ck_field_size_reps(t, ck_mul_size((size_t)REE_DELAY_BUF_SIZE, Kz),
                               sizeof(float), 2);          /* drd rrd */
        t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* ld_st lr_st scr_st */
    }
    if (!lane) {   /* the gain arrays: only suppression_gain_get_gain() (FULL) reads them */
        t = ck_add_size(t, aec_sg_tables_bytes(K));        /* the four decoded ENR/SU tables */
        t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* last_gain/ne/echo */
        t = ck_field_size(t, ck_mul_size((size_t)sg_n, Kz), sizeof(float)); /* ma */
        t = ck_field_size_reps(t, Kz, sizeof(float), 7);   /* ne wr ming maxg graw gout gsum */
    }
    t = ck_field_size(t, Kz, sizeof(float));           /* stat_noise */
    t = ck_field_size(t, Kz, sizeof(int32_t));         /* stat_hang */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_flags */
    {   /* exactly the ring's rows */
        const int rows = stationarity_estimator_window_hops(hop, cfg->sample_rate);
        if (rows < 1) return 0;
        t = ck_field_size(t, ck_mul_size((size_t)rows, Kz), sizeof(float)); /* stat_hist */
    }
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 2); /* prev_output_time e_form */
    t = ck_field_size(t, Kz, sizeof(Complex));         /* lfs sel_esw */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 3); /* scr_sq scr_sref scr_scoa */
    t = ck_field_size(t, (size_t)fft, sizeof(float));  /* scr_tin */
    t = ck_field_size_reps(t, Kz, sizeof(Complex), 5); /* sel_esw sel_echo nsw_e1 ybase output_base */
    t = ck_field_size_reps(t, Kz, sizeof(float), 8);   /* abs_near .. x2_at_delay */
    t = ck_field_size(t, ck_mul_size((size_t)np, Kz), sizeof(float)); /* w_mag2 */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* render_block_scaled */
    t = ck_field_size_reps(t, Kz, sizeof(float), 3);   /* r2 r2_unb nearend_pwr */
    t = ck_field_size(t, Kz, sizeof(unsigned char));   /* stat_mask */
    t = ck_field_size(t, Kz, sizeof(float));           /* per_bin_mu_scale */
    t = ck_field_size_reps(t, (size_t)hop, sizeof(float), 5); /* near_hop far_hop raw shadow final */
    t = ck_field_size(t, ck_mul_size((size_t)np, (size_t)hop), sizeof(float)); /* filter_taps_full */
    t = ck_field_size(t, (size_t)hop, sizeof(float));  /* scr_sq */
    t = ck_field_size_reps(t, Kz, sizeof(float), 6);   /* scr_e2_echo .. scr_e2ref_arr */
    if (tn->enable_highpass) t = ck_field_size(t, 1, hpf_get_mem_size()); /* mic-path HPF */

    if (MEM_SIZE_INVALID(t)) return 0;
    return t;
}

/* Static-pool memory breakdown; total_bytes is aec_get_mem_size(). */
static int aec_mem_breakdown_resolved(const AecResolved* rs, AecMemBreakdown* out) {
    const AecConfig* cfg = &rs->cfg;
    size_t total = aec_mem_size_resolved(rs);
    if (total == 0) return AEC_ERR_ARG;
    if (AEC_CFG_IS_ARRAY(cfg)) return aec_array_impl_get_mem_breakdown(rs, out);

    int hop, blk, fft, K, np, buf;
    aec_derive_dims(rs, &hop, &blk, &fft, &K, &np, &buf);

    size_t est = 0;
    if (cfg->delay_mode == AEC_DELAY_MATCHED) {
        est = delay_aec_get_mem_size(cfg->sample_rate, hop,
                                       cfg->delay_num_filters);
    }
    size_t ring = (buf > 0) ? ck_field_size(0, (size_t)buf, sizeof(float)) : 0;

    out->total_bytes     = total;
    out->estimator_bytes = est;
    out->ring_bytes      = ring;
    return AEC_OK;
}

int aec_get_mem_breakdown(const AecConfig* cfg_in, AecMemBreakdown* out) {
    AecResolved rs;
    if (!out) return AEC_ERR_ARG;
    memset(out, 0, sizeof(*out));
    if (!cfg_in || !aec_resolve(cfg_in, &rs)) return AEC_ERR_ARG;
    return aec_mem_breakdown_resolved(&rs, out);
}

/* Carve of an instance whose Aec sits at the start of its pool (aec_init() and aec_create()). `a` must be zeroed. Returns 0,
 * or -1 if a stage refused (no unwinding: every handle lives in the carved memory). */

/* Wall-clock-preserving counter thresholds shared by aec_carve() and
 * aec_reset(). */
static void aec_recompute_wallclock_thresholds(Aec* a, int hop, int sample_rate) {
    a->poor_coarse_threshold_hops = aec_blocks_to_hops(5, hop, sample_rate);
    a->coarse_reset_hangover_hops = aec_blocks_to_hops(25, hop, sample_rate);
    a->leakage_div_sustain_hops = aec_ms_to_hops(50.0f, hop, sample_rate);
    a->stat_dt_hangover_hops = aec_ms_to_hops(800.0f, hop, sample_rate);
}

/* Synthesis window of length n (a power of two that divides AECB_SYNTH_HALF_N), rebuilt bit for bit from the half table. */
void aec_synth_window_build(float* w, int n) {
    const int stride = AECB_SYNTH_HALF_N / n;
    int i;
    assert(n >= 2 && AECB_SYNTH_HALF_N % n == 0);
    for (i = 0; i < n; ++i) w[i] = AECB_SYNTH_WINDOW_HALF[(i <= n / 2 ? i : n - i) * stride];
}

size_t aec_sg_tables_bytes(int n_bins) {
    return ck_mul_size(4, ck_align16_size((size_t)n_bins * sizeof(float)));
}

/* Expands the row's run-length ENR/SU tables into `mem`: four tables, each padded to 16 bytes and zero in the padding. */
static void aec_sg_tables_expand(const AecBalancedRateDims* rd, void* mem, SuppressionGainTuning* tun) {
    const int n = rd->sg_table_len;
    const size_t stride = ALIGN16((size_t)n * sizeof(float)) / sizeof(float);
    float* const dst = (float*)mem;
    int t, r = 0, k;
    assert(rd->sg_runs >= 1 && rd->sg_run_end[rd->sg_runs - 1] >= n);
    for (k = 0; k < n; ++k) {
        while (k >= rd->sg_run_end[r]) ++r;
        for (t = 0; t < 4; ++t) dst[(size_t)t * stride + (size_t)k] = rd->sg_run_val[(size_t)t * (size_t)rd->sg_runs + (size_t)r];
    }
    for (t = 0; t < 4; ++t)
        for (k = n; (size_t)k < stride; ++k) dst[(size_t)t * stride + (size_t)k] = 0.0f;
    tun->nearend_enr_tr = dst;                tun->nearend_enr_su = dst + stride;
    tun->normal_enr_tr  = dst + 2 * stride;   tun->normal_enr_su  = dst + 3 * stride;
}

int aec_sg_tables_decode(const AecResolved* rs, void* mem, SuppressionGainTuning* tun) {
    const AecBalancedRateDims* rd;
    if (!rs || !mem || !tun) return -1;
    rd = aecb_rate_cfg(rs->cfg.sample_rate, rs->cfg.fft_size);
    if (!rd) return -1;
    aec_sg_tables_expand(rd, mem, tun);
    return 0;
}

/* Suppression-gain config and tuning for a resolved configuration; shared by
 * Aec and standalone RES instances. The ENR/SU tables of the tuning stay NULL (aec_sg_tables_decode() fills them) and the
 * EMR tables point at static data. Returns 0, or -1 without a signal grid or rate row. */
int aec_build_sg_config(const AecResolved* rs, SuppressionGainConfig* out_cfg,
                        SuppressionGainTuning* out_tun) {
    const AecConfig* cfg;
    const AecTuning* tn;
    AecSignalGrid grid;
    const AecBalancedRateDims* rd;
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    int hop, K;
    if (!rs || !out_cfg || !out_tun) return -1;
    cfg = &rs->cfg;
    tn = &rs->tun;
    if (aec_resolve_signal_grid(cfg->sample_rate, cfg->fft_size, &grid) != AEC_OK) return -1;
    rd = aecb_rate_cfg(cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;
    hop = grid.hop_size;
    K = grid.n_freqs;
    memset(&scfg, 0, sizeof(scfg));
    scfg.n_bins = K; scfg.sr = cfg->sample_rate; scfg.hop_size = hop;
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
    /* From the level, so NO_FLOOR reaches the suppressor at construction. */
    scfg.split_floor_enabled = tn->min_gain_split_floor_enabled;
    scfg.split_floor_far_active =
        powf(10.0f, tn->min_gain_floor_far_active_db / 10.0f);
    scfg.split_floor_far_silent = AECB_SG_SPLIT_FLOOR_FAR_SILENT;
    scfg.split_floor_dt =
        powf(10.0f, tn->min_gain_floor_dt_db / 10.0f);
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
    /* The ENR/SU tables are decoded into the pool (aec_sg_tables_decode()); only the EMR tables are static. */
    stun.nearend_emr_tr = rd->sg_nearend_emr_tr;
    stun.normal_emr_tr  = rd->sg_normal_emr_tr;
    stun.table_len      = rd->sg_table_len;
    *out_cfg = scfg;
    *out_tun = stun;
    return 0;
}

static void aec_init_config_state(Aec* a, const AecResolved* rs, int hop, int blk,
                                  int fft, int K, int np) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    a->cfg = *cfg;
    a->tun = *tn;
    a->hop_size = hop; a->block_size = blk; a->fft_size = fft;
    a->n_freqs = K; a->n_partitions = np;

    /* Retimes wall-clock-authored constants to the resolved grid (only
     * a->tun, the instance copy). */
    a->tun.shadow_err_alpha = aec_legacy10ms_alpha(tn->shadow_err_alpha, hop, cfg->sample_rate);
    a->tun.warmup_frames = aec_legacy10ms_hops((float)tn->warmup_frames * 10.0f, hop, cfg->sample_rate);
    a->tun.epc_hangover = aec_legacy10ms_hops((float)tn->epc_hangover * 10.0f, hop, cfg->sample_rate);
    a->tun.ne_recent_hold = aec_legacy10ms_hops((float)tn->ne_recent_hold * 10.0f, hop, cfg->sample_rate);
    a->tun.filter_misadjustment_stable_frames = aec_legacy10ms_hops(
        (float)tn->filter_misadjustment_stable_frames * 10.0f, hop, cfg->sample_rate);
    a->tun.filter_misadjustment_hangover_frames = aec_legacy10ms_hops(
        (float)tn->filter_misadjustment_hangover_frames * 10.0f, hop, cfg->sample_rate);

    {
        int _sn = (int)(0.5f * (float)cfg->sample_rate / (float)(hop > 0 ? hop : 1));
        if (_sn < 2) _sn = 2;
        if (_sn > (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0])))
            _sn = (int)(sizeof(a->erle_slope_buf) / sizeof(a->erle_slope_buf[0]));
        a->erle_slope_cap = _sn;
        a->erle_slope_len = 0;
        a->erle_slope_head = 0;
    }

    if (tn->enable_saturation) {
        saturation_init(&a->sat_ref, tn->saturation_threshold,
                        hop, cfg->sample_rate);
        saturation_init(&a->sat_mic, tn->saturation_threshold,
                        hop, cfg->sample_rate);
        a->has_sat = 1;
    }
}

static uint8_t* aec_carve_delay_ring(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                 int hop, int K, int buf_samp) {
    const AecConfig* cfg = &rs->cfg;
    /* Delay ring: MATCHED estimator + ring; FIXED
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
    a->duty.last_delay = -1;

    a->rsa_counters = (int64_t*)ptr;
    size_t rsa_sz = ALIGN16((size_t)(K - 2 > 0 ? K - 2 : 1) * sizeof(int64_t));
    memset(ptr, 0, rsa_sz); ptr += rsa_sz;

    return ptr;
}

static uint8_t* aec_carve_filters(Aec* a, uint8_t* ptr, const AecResolved* rs,
                              int hop, int blk, int fft, int K, int np, const AecCarveOpt* opt) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    /* Shared FFT handle: one per instance, borrowed by the main filter, shadow filter and aec_post; a lane borrows the
     * array's instead. A failure here needs no unwind. */
    if (opt->lane) {
        if (!opt->shared_fft || fft_get_n_freqs(opt->shared_fft) != K) return NULL;
        a->post_fft = opt->shared_fft;
    } else {
        size_t fft_sz = fft_get_mem_size(fft);
        a->post_fft = fft_init(ptr, fft_sz, fft);
        if (!a->post_fft) return NULL;
        ptr += fft_sz;
    }

    size_t kf_sz = pbfdkf_get_mem_size(blk, np, hop);
    if (pbfdkf_init_static(&a->main_filter, ptr, kf_sz, blk, np, tn->mu, tn->delta, hop, cfg->sample_rate, a->post_fft) != 0) {
        fft_destroy(a->post_fft); a->post_fft = NULL;
        return NULL;
    }
    a->main_filter.base.poor_excitation_counter = aec_blocks_to_hops(1000, hop, cfg->sample_rate);
    a->main_filter.base.constraint_round_robin = 1;
    ptr += kf_sz;

    if (tn->enable_shadow) {
        /* The shadow follows the main filter: same far input, same clears and partition steps, so it shares the main
         * filter's far history and skips the spectra its lightweight frontend never forms. */
        size_t af_sz = pbfdaf_get_mem_size_follower(blk, np, hop);
        if (pbfdaf_init_static_follower(&a->shadow_filter, ptr, af_sz, blk, np, tn->shadow_mu_nlms, tn->delta, hop,
                                        cfg->sample_rate, a->post_fft, &a->main_filter.base) != 0) {
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
    if (tn->live_observer_enabled) {
        size_t obs_sz = pbfdaf_get_mem_size(blk, np, hop, 1);
        if (pbfdaf_init_static(&a->observer_filter, ptr, obs_sz, blk, np,
                               tn->live_observer_mu, tn->delta, hop, 1,
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
        a->observer_shift_hops = aec_ms_to_hops(tn->live_observer_shift_ms, hop, cfg->sample_rate);
    }

    if (opt->lane) {
        /* EXTERNAL_ALIGNED: no estimator, so no delay-shift judgment ever snapshots. */
        a->delay_shift_judgment_pre_W = NULL;
        a->delay_shift_judgment_pre_shadow_W = NULL;
        a->delay_shift_judgment_pre_H_error = NULL;
    } else {
        size_t w_snap_sz = ALIGN16(ck_mul_size((size_t)np, (size_t)K) * sizeof(Complex));
        a->delay_shift_judgment_pre_W = (Complex*)ptr; ptr += w_snap_sz;
        if (tn->enable_shadow) {
            a->delay_shift_judgment_pre_shadow_W = (Complex*)ptr; ptr += w_snap_sz;
        } else {
            a->delay_shift_judgment_pre_shadow_W = NULL;
        }
        a->delay_shift_judgment_pre_H_error = (float*)ptr;
        ptr += ALIGN16((size_t)K * sizeof(float));
    }
    a->delay_shift_judgment_active = 0;
    a->delay_shift_judgment_pre_delay = -1;
    a->delay_shift_judgment_pre_erle = 0.0f;
    a->delay_shift_judgment_hops_left = 0;
    a->delay_reject_count = 0;

    return ptr;
}

static uint8_t* aec_carve_detectors(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                    int hop, int K, int np) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    a->ra_pairwise_scratch = (float*)ptr;
    ptr += ALIGN16((size_t)((hop + 7) / 8) * sizeof(float));
    render_activity_init(&a->render_activity, a->ra_pairwise_scratch, (hop + 7) / 8,
                         hop, cfg->sample_rate);
    filter_convergence_init(&a->convergence, hop, cfg->sample_rate);
    doubletalk_init(&a->dt_analyzer, 1.5, 3.0, hop, cfg->sample_rate);
    epc_init(&a->epc, a->tun.epc_hangover, tn->epc_total_rise, tn->epc_delta_threshold,
             hop, cfg->sample_rate);
    shadow_copy_init_for_grid(&a->regime, SC_GATE_ENERGY, 0.65, 3,
                              a->tun.epc_hangover, hop, cfg->sample_rate);
    rsa_init(&a->rsa, a->rsa_counters, K, np);

    return ptr;
}

static uint8_t* aec_carve_post_backing(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                       const AecBalancedRateDims* rd,
                                       int hop, int blk, int fft, int K, int lane) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
    const int state_lane = aec_state_lane(cfg, lane);
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
        pcfg.cng_n2_windowed_psd         = tn->cng_n2_windowed_psd;
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
        pcfg.observer_enabled = tn->live_observer_enabled;
        if (tn->live_observer_enabled) {
            float bin_hz = (float)cfg->sample_rate / (float)fft;
            pcfg.observer_max_bin = (int)ceilf((float)tn->live_observer_max_hz / bin_hz);
            double hop_ms = 1000.0 * (double)hop / (double)cfg->sample_rate;
            double corrob_ms = (double)tn->live_observer_corrob_ms;
            if (corrob_ms < 1e-6) corrob_ms = 1e-6;
            pcfg.observer_err_ema_alpha =
                aec_ema_alpha_from_tau_ms(corrob_ms, hop_ms);
            pcfg.observer_corrob_margin =
                aec_db_to_linear((double)tn->live_observer_corrob_margin_db);
        }

#define P_FSLICE(n) ((float*)(ptr));    ptr += ALIGN16(K * sizeof(float));    (void)(n)
#define P_CSLICE(n) ((Complex*)(ptr));  ptr += ALIGN16(K * sizeof(Complex));  (void)(n)
#define P_BSLICE(n) ((unsigned char*)(ptr)); ptr += ALIGN16(K * sizeof(unsigned char)); (void)(n)
        float        *avg_rev  = P_FSLICE(0);
        float        *y2s = NULL, *n2 = NULL, *n2i = NULL;   /* the comfort-noise tracker */
        if (!state_lane) { y2s = P_FSLICE(0); n2 = P_FSLICE(0); n2i = P_FSLICE(0); }
        float        *sye_re   = P_FSLICE(0); float *sye_im = P_FSLICE(0);
        float        *syy      = P_FSLICE(0); float *see    = P_FSLICE(0);
        /* ola, eout and eout_full serve aec_post_apply_output() only (FULL); a lane never synthesizes. */
        float        *swin     = NULL;   /* the synthesis window, rebuilt from the half table; a lane never synthesizes */
        float        *ola      = NULL;
        if (!lane) {
            swin = (float*)ptr; ptr += ALIGN16((size_t)blk * sizeof(float));
            aec_synth_window_build(swin, blk);
            ola = (float*)ptr; ptr += ALIGN16((size_t)blk * sizeof(float));
        }
        float        *np_      = P_FSLICE(0); float *fp_    = P_FSLICE(0);
        float        *ep_      = P_FSLICE(0); float *erp_   = P_FSLICE(0);
        float        *cpe      = P_FSLICE(0); float *x2r    = P_FSLICE(0);
        float        *cn = NULL;
        if (!state_lane) { cn = P_FSLICE(0); }
        float        *nf_      = P_FSLICE(0);
        unsigned char *cgm     = P_BSLICE(0);
        Complex      *eout     = NULL;
        float        *eout_full = NULL;
        if (!lane) {
            eout = P_CSLICE(0);
            eout_full = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        }
        memset(syy, 0, K * sizeof(float)); memset(see, 0, K * sizeof(float));
        aec_post_init(&a->post, &pcfg, a->post_fft,
                      swin, avg_rev,
                      y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                      np_, fp_, ep_, erp_, cpe, x2r, cgm, cn, nf_,
                      eout, eout_full);
    }

    return ptr;
}

static uint8_t* aec_carve_state(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                int hop, int K, int np) {
    const AecConfig* cfg = &rs->cfg;
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

static uint8_t* aec_carve_residual(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                   const AecBalancedRateDims* rd, int hop, int K, int lane) {
    const AecConfig* cfg = &rs->cfg;
    {
        int rh = (AECB_REE_USE_AEC_ECHO_GEN_WINDOW ? 1 : 0) + 2;
        float *x2_nf = NULL, *rm_st = NULL, *rt_st = NULL, *rh_st = NULL, *drd_st = NULL, *rrd_st = NULL;
        float *ld_st = NULL, *lr_st = NULL, *scr_st = NULL;
        int   *x2_nfc = NULL;
        if (!aec_state_lane(cfg, lane)) {   /* a STATE lane never runs the estimator: ree_init() takes NULL storage */
            x2_nf  = P_FSLICE(0);
            x2_nfc = (int*)ptr;   ptr += ALIGN16(K * sizeof(int));
            rm_st  = P_FSLICE(0); rt_st = P_FSLICE(0);
            if (!AECB_REE_USE_AEC_ECHO_GEN_WINDOW) { rh_st = (float*)ptr; ptr += ALIGN16((size_t)rh * K * sizeof(float)); }   /* read only without the echo-generating window (see ree_estimate()) */
            drd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
            rrd_st = (float*)ptr; ptr += ALIGN16((size_t)REE_DELAY_BUF_SIZE * K * sizeof(float));
            ld_st  = P_FSLICE(0); lr_st = P_FSLICE(0); scr_st = P_FSLICE(0);
        }
        aec_ree_bind(a, rd, K, hop, cfg->sample_rate, x2_nf, x2_nfc, rm_st,
                     rt_st, rh_st, drd_st, rrd_st, ld_st, lr_st, scr_st);
    }

    return ptr;
}

static uint8_t* aec_carve_suppression(Aec* a, uint8_t* ptr, const AecBalancedRateDims* rd,
                                      const SuppressionGainConfig* scfg,
                                      const SuppressionGainTuning* stun_in, int K, int lane) {
    {
        SuppressionGain* sg = &a->a3_sg;
        SuppressionGainTuning stun = *stun_in;
        /* A lane never computes a gain (residual below FULL): only the DominantNearend state of sg runs, so the arrays stay
         * NULL (suppression_gain_init() accepts that). */
        float *last_gain = NULL, *last_ne = NULL, *last_echo = NULL, *ma = NULL, *ne_s = NULL, *wr_s = NULL;
        float *ming = NULL, *maxg = NULL, *graw = NULL, *gout = NULL, *gsum = NULL;
        if (!lane) {
            aec_sg_tables_expand(rd, ptr, &stun);
            ptr += aec_sg_tables_bytes(K);
            last_gain = P_FSLICE(0); last_ne = P_FSLICE(0); last_echo = P_FSLICE(0);
            ma   = (float*)ptr; ptr += ALIGN16((size_t)rd->sg_nearend_smoother_n * K * sizeof(float));
            ne_s = P_FSLICE(0); wr_s = P_FSLICE(0);
            ming = P_FSLICE(0); maxg = P_FSLICE(0);
            graw = P_FSLICE(0); gout = P_FSLICE(0); gsum = P_FSLICE(0);
        }
        suppression_gain_init(sg, scfg, &stun, last_gain, last_ne, last_echo,
                              ma, ne_s, wr_s, ming, maxg, graw, gout, gsum);
    }

    return ptr;
}

static uint8_t* aec_carve_stationarity(Aec* a, uint8_t* ptr, const AecResolved* rs, int hop, int K) {
    const AecConfig* cfg = &rs->cfg;
    {
        float        *st_noise = P_FSLICE(0);
        int32_t      *st_hang  = (int32_t*)ptr; ptr += ALIGN16(K * sizeof(int32_t));
        unsigned char *st_flg  = P_BSLICE(0);
        const int    rows = stationarity_estimator_window_hops(hop, cfg->sample_rate);
        float        *st_hist  = (float*)ptr;   ptr += ALIGN16((size_t)rows * K * sizeof(float));
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
        a->a3_lfs.sel_esw          = (Complex*)ptr; ptr += ALIGN16(K * sizeof(Complex));
        a->a3_lfs.scr_sq   = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_sref = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_scoa = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        a->a3_lfs.scr_tin  = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
        linear_filter_select_reset(&a->a3_lfs);
    }

    return ptr;
}

static uint8_t* aec_carve_post_scratch(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                       int hop, int K, int np) {
    const AecTuning* tn = &rs->tun;
    {
        AecPostRunScratch* sc = &a->a3_sc;
        sc->sel_esw  = P_CSLICE(0); sc->sel_echo = P_CSLICE(0);
        sc->nsw_e1   = P_CSLICE(0); sc->ybase    = P_CSLICE(0);
        sc->output_base = P_CSLICE(0);
        sc->abs_near     = P_FSLICE(0); sc->abs_far      = P_FSLICE(0);
        sc->abs_sel_echo = P_FSLICE(0); sc->abs_error    = P_FSLICE(0);
        sc->abs_echo_coh = P_FSLICE(0); sc->abs_nsw_e1   = P_FSLICE(0);
        sc->abs_ybase    = P_FSLICE(0); sc->x2_at_delay  = P_FSLICE(0);
        sc->w_mag2 = (float*)ptr; ptr += ALIGN16((size_t)np * K * sizeof(float));
        sc->render_block_scaled = (float*)ptr; ptr += ALIGN16((size_t)hop * sizeof(float));
        sc->r2          = P_FSLICE(0); sc->r2_unb   = P_FSLICE(0);
        sc->nearend_pwr = P_FSLICE(0);
        sc->stat_mask   = P_BSLICE(0);
        sc->observer_out = NULL; sc->observer_diff_spec = NULL;
        sc->observer_diff_mag2 = NULL; sc->observer_cap_mag2 = NULL;
        if (tn->live_observer_enabled) {
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

static uint8_t* aec_carve_hop_scratch(Aec* a, uint8_t* ptr, const AecResolved* rs,
                                  int hop, int K, int np) {
    const AecConfig* cfg = &rs->cfg;
    const AecTuning* tn = &rs->tun;
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
    a->scr_mu_buf    = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    a->scr_e2ref_arr = (float*)ptr; ptr += ALIGN16((size_t)K   * sizeof(float));
    /* Mic-path HPF; hpf_init() failure unwinds the earlier allocations. */
    if (tn->enable_highpass) {
        a->hp_mic = hpf_init(ptr, hpf_get_mem_size(),
                             tn->highpass_cutoff_hz, cfg->sample_rate);
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

/* Per-hop scalar state a fresh instance and aec_reset() both start from. The Aec is zeroed before the carve, so the fields only
 * a reset has to clear store zero again at init. */
static void aec_reset_scalar_state(Aec* a, int hop, int sample_rate) {
    a->pending_gain_change = 0; a->pending_delay_change = -1;
    a->stationarity_active_hops = 0; a->non_zero_render_seen = 0;
    a->block_stationary_next = 0;
    a->saturation_level = 0.0f; a->erl_estimate = 0.1f;
    a->main_err_smooth = 0.0f; a->shadow_err_smooth = 0.0f;
    a->shadow_frame_count = 0; a->epc_render_forced_remaining = 0;
    a->erle_window_near = 1e-10f; a->erle_window_err = 1e-10f;
    a->erle_factor_prev = 0.0f; a->inst_erle_smooth = 1.0f;
    a->erle_slope_len = 0; a->erle_slope_head = 0;
    a->wn_err_baseline = 1e-8f; a->stat_dt_hangover = 0;
    a->warmup_frames_remaining = a->tun.warmup_frames; a->warmup_far_active = 0;
    a->simple_mu_ratio = 1.0f; a->simple_mu_holdoff = 0; a->has_per_bin_mu = 0;
    a->near_power = 0.0f; a->raw_error_power = 0.0f;
    a->frame_count = 0;
    a->ne_above = 0; a->ne_recent_frames = 0;
    a->poor_coarse_counter = 0; a->coarse_reset_hangover = 0;
    a->leakage_div_sustained_counter = 0;
    aec_recompute_wallclock_thresholds(a, hop, sample_rate);
    a->misadj_e2_acum = 0.0f; a->misadj_y2_acum = 0.0f; a->misadj_n_acum = 0;
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_stable_count = 0; a->misadj_hangover_remaining = 0;
    a->last_erle_windowed = 0.0f;
    /* The RES-context stash reads what a fresh instance publishes. */
    a->last_far_power = 0.0f; a->last_shadow_dt = 0.0f;
    a->last_dt_indicator = 0.0f; a->last_is_stationary_dt = 0;
}

static void aec_init_scalar_state(Aec* a, const AecResolved* rs, const AecBalancedRateDims* rd, int hop) {
    const AecConfig* cfg = &rs->cfg;
    aec_reset_scalar_state(a, hop, cfg->sample_rate);
    a->render_peak_floor = 10.0f / 32768.0f;
    a->stationarity_converge_hops = rd->stationarity_converge_hops;
    a->alpha_pow = powf(0.95f, 16000.0f / (float)a->cfg.sample_rate);
    a->alpha_erl_tracking  = aec_growth_rehop(0.99f,  160, 16000,
                                              a->hop_size, a->cfg.sample_rate);
    a->alpha_erl_converged = aec_growth_rehop(0.999f, 160, 16000,
                                              a->hop_size, a->cfg.sample_rate);
    aec_duty_configure(&a->duty, a->tun.delay_est_init_s, a->tun.delay_est_period_s, hop, cfg->sample_rate);
    a->is_static = 1;
}

static int aec_carve(Aec* a, uint8_t* ptr, const AecResolved* rs,
                      int hop, int blk, int fft, int K, int np,
                      int buf_samp, const AecCarveOpt* opt) {
    const AecConfig* cfg = &rs->cfg;
    /* Per-rate row (a hit is guaranteed by aec_resolve). */
    const AecBalancedRateDims* rd = aecb_rate_cfg(
        cfg->sample_rate, cfg->fft_size);
    if (!rd) return -1;
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    if (aec_build_sg_config(rs, &scfg, &stun) != 0) return -1;

    uint8_t* const start = ptr;
    aec_init_config_state(a, rs, hop, blk, fft, K, np);
    ptr = aec_carve_delay_ring(a, ptr, rs, hop, K, buf_samp);
    if (!ptr) return -1;
    ptr = aec_carve_filters(a, ptr, rs, hop, blk, fft, K, np, opt);
    if (!ptr) return -1;
    ptr = aec_carve_detectors(a, ptr, rs, hop, K, np);
    ptr = aec_carve_post_backing(a, ptr, rs, rd, hop, blk, fft, K, opt->lane);
    ptr = aec_carve_state(a, ptr, rs, hop, K, np);
    ptr = aec_carve_residual(a, ptr, rs, rd, hop, K, opt->lane);
    ptr = aec_carve_suppression(a, ptr, rd, &scfg, &stun, K, opt->lane);
    ptr = aec_carve_stationarity(a, ptr, rs, hop, K);
    ptr = aec_carve_linear_select(a, ptr, hop, blk, fft, K);
    ptr = aec_carve_post_scratch(a, ptr, rs, hop, K, np);
    ptr = aec_carve_hop_scratch(a, ptr, rs, hop, K, np);
    if (!ptr) return -1;
    /* The carve must end exactly where aec_mem_size_impl() said (a lane sits back to back with its neighbours; a single-channel
     * pool would otherwise be overrun, or over-sized, without a sign). */
    if ((size_t)(ptr - start) + ALIGN16(sizeof(Aec)) != aec_mem_size_impl(rs, opt->lane)) return -1;
    aec_init_scalar_state(a, rs, rd, hop);
    return 0;
}

/* Pool descriptor: the carve text is hashed, not the tunable values; bump AEC_LAYOUT_VERSION whenever a carved layout changes
 * size or order (single-channel or array). */
static uint32_t aec_build_flags_hash(void) {
    static const char* const parts[] = {
        "ne10",
        "|carve:aec(cfg-v2),single(aec,delay?,ring?,filters(shadow-follows-main),detectors,post(swin,ola),state,residual,suppression(tables),stationarity(window-rows),lfs,scratch,hpf?)",
        "|array(aec,ctl,swin,mic,aligned,fuse?(complex5,float2),ring?,delay?,res?(tables),fft,lane4(shared-fft,lean,state-lane-no-ree-cng))",
        "|align16"
    };
    uint32_t hash = AC_FNV1A_SEED;
    size_t i;
    for (i = 0; i < sizeof(parts) / sizeof(parts[0]); ++i) hash = ac_fnv1a_str(hash, parts[i]);
    return ac_memreq_abi_hash(hash);
}

static int aec_mem_requirements_resolved(const AecResolved* rs, AecMemReq* out) {
    size_t bytes = aec_mem_size_resolved(rs);
    memset(out, 0, sizeof(*out));
    if (bytes == 0) return -1;
    out->descriptor_version = AEC_DESCRIPTOR_VERSION;
    out->layout_version = AEC_LAYOUT_VERSION;
    out->backend_id = AEC_BACKEND_NE10;
    out->build_flags_hash = aec_build_flags_hash();
    out->alignment = 16u;
    out->bytes = (uint64_t)bytes;
    return 0;
}

int aec_get_mem_requirements(const AecConfig* cfg, AecMemReq* out) {
    AecResolved rs;
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!cfg || !aec_resolve(cfg, &rs)) return -1;
    return aec_mem_requirements_resolved(&rs, out);
}

/* Single-channel construction in `mem`; opt->lane builds a lane (see AecCarveOpt). */
static Aec* aec_init_single(void* mem, size_t mem_size, const AecResolved* rs, const AecCarveOpt* opt) {
    {
        size_t need = aec_mem_size_impl(rs, opt->lane);
        if (need == 0 || mem_size < need) return NULL;
    }

    int hop, blk, fft, K, np, buf_samp;
    aec_derive_dims(rs, &hop, &blk, &fft, &K, &np, &buf_samp);

    Aec* a = (Aec*)mem;
    memset(a, 0, sizeof(Aec));
    uint8_t* ptr = (uint8_t*)mem + ALIGN16(sizeof(Aec));
    if (aec_carve(a, ptr, rs, hop, blk, fft, K, np, buf_samp, opt) != 0) {
        /* aec_carve() failed: no unwinding; every handle lives in the caller's memory. */
        return NULL;
    }
    a->heap_arena = NULL;
    a->ready = 1;
    return a;
}

static Aec* aec_init_resolved(void* mem, size_t mem_size, const AecResolved* rs) {
    static const AecCarveOpt plain = { 0, NULL };
    if (!mem || !MEM_IS_ALIGNED16(mem)) return NULL;
    if (AEC_CFG_IS_ARRAY(&rs->cfg)) return aec_array_impl_init(mem, mem_size, rs);
    return aec_init_single(mem, mem_size, rs, &plain);
}

#ifndef AEC_NO_ARRAY
Aec* aec_lane_init(void* mem, size_t mem_size, const AecResolved* lane, FftHandle* shared_fft) {
    AecCarveOpt opt;
    if (!mem || !lane || !shared_fft || !MEM_IS_ALIGNED16(mem)) return NULL;
    if (!aec_resolved_valid(lane) || !aec_lane_contract_ok(&lane->cfg)) return NULL;
    opt.lane = 1;
    opt.shared_fft = shared_fft;
    return aec_init_single(mem, mem_size, lane, &opt);
}
#endif

static Aec* aec_init_ex_resolved(void* mem, size_t mem_size, const AecResolved* rs, const AecMemReq* expected) {
    if (expected) {
        AecMemReq current;
        if (aec_mem_requirements_resolved(rs, &current) != 0) return NULL;
        if (expected->descriptor_version != current.descriptor_version ||
            expected->layout_version != current.layout_version ||
            expected->backend_id != current.backend_id ||
            expected->build_flags_hash != current.build_flags_hash ||
            expected->alignment != current.alignment ||
            expected->reserved != 0u ||
            expected->bytes < current.bytes ||
            (uint64_t)mem_size < current.bytes) return NULL;
    }
    return aec_init_resolved(mem, mem_size, rs);
}

Aec* aec_init_ex(void* mem, size_t mem_size, const AecConfig* cfg, const AecMemReq* expected) {
    AecResolved rs;
    if (!cfg || !aec_resolve(cfg, &rs)) return NULL;
    return aec_init_ex_resolved(mem, mem_size, &rs, expected);
}

Aec* aec_init(void* mem, size_t mem_size, const AecConfig* cfg_in) {
    return aec_init_ex(mem, mem_size, cfg_in, NULL);
}

void aec_destroy(Aec* a) {
    void* heap;
    if (aec_unusable(a)) return;
    if (AEC_IS_ARRAY(a)) { aec_array_impl_destroy(a); return; }
    /* The FFT handle, high-pass filter and filters all live in the instance's pool, so these only drop references. */
    if (a->post_fft) { fft_destroy(a->post_fft); a->post_fft = NULL; }
    if (a->hp_mic) { hpf_destroy(a->hp_mic); a->hp_mic = NULL; }
    pbfdkf_free(&a->main_filter);
    if (a->has_shadow) pbfdaf_free(&a->shadow_filter);
    if (a->has_observer) pbfdaf_free(&a->observer_filter);

    heap = a->is_static ? NULL : a->heap_arena;
    a->ready = 0;
    if (heap) free(heap);       /* aec_create(): the block that holds *a; nothing touches *a after this */
}

int aec_hop_size(const Aec* a) {
    if (aec_unusable(a)) return 0;
    return a->hop_size;
}


/* Saturating bump of the AecLinearView alignment generation. */
static void aec_bump_delay_generation(Aec* a) {
    if (a->delay_generation != 0xFFFFFFFFu) { a->delay_generation++; }
}
/* Far-end FFT count: own rffts summed over shadow, main and observer (main
 * and shadow are exclusive per hop). */
long aec_far_fft_real_compute_count(const Aec* a) {
    if (aec_unusable(a)) return 0;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_far_fft_count(a);
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

int aec_sg_apply_level(SuppressionGain* sg, AecLevel level) {
    float db, dt_db;
    int en;
    if (aec_level_floor_db(level, &db) != 0 || aec_level_dt_floor_db(level, &dt_db) != 0 ||
        aec_level_split_floor_enabled(level, &en) != 0) return -1;
    /* The suppressor validates and applies first and writes nothing on refusal. The ramp governs the far-active floor only; the
     * DT floor and the split-floor gate step immediately. */
    if (suppression_gain_set_split_floor_far_active_db(sg, db, AEC_LEVEL_RAMP_MS) != 0) return -1;
    if (suppression_gain_set_split_floor_dt_db(sg, dt_db) != 0) return -1;
    return suppression_gain_set_split_floor_enabled(sg, en) != 0 ? -1 : 0;
}

int aec_set_level(Aec* a, AecLevel level) {
    if (aec_unusable(a)) return -1;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_set_level(a, level);
    /* The suppressor applies first and refuses without writing, leaving the instance's level unchanged. */
    if (aec_sg_apply_level(&a->a3_sg, level) != 0) return -1;
    return aec_apply_level(&a->cfg, &a->tun, level);
}

int aec_apply_external_realign(Aec* a, int delta_samples) {
    if (aec_unusable(a) || AEC_IS_ARRAY(a) || a->cfg.delay_mode != AEC_DELAY_EXTERNAL_ALIGNED)
        return -1;      /* an array instance realigns its lanes itself, on a shared-alignment change */
    if (delta_samples == 0) return 0;
    /* Warm transfer needs aec_linear_is_cancelling() and a shift that fits
     * the tap span. */
    int warm_ok = 0;
    if (a->tun.delay_acquire_warm_transfer) {
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
    if (aec_unusable(a)) return;
    if (AEC_IS_ARRAY(a)) { aec_array_impl_reset(a); return; }
    a->hop_phase = 0;
    a->residual_valid = 0;
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
    aec_duty_reset(&a->duty);
    a->duty_hops_total = 0;
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
    aec_post_chain_reset(a);
    aec_reset_scalar_state(a, a->hop_size, a->cfg.sample_rate);
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
    if (a->misadj_stable_count < a->tun.filter_misadjustment_stable_frames)
        a->misadj_stable_count++;
    if (a->misadj_stable_count < a->tun.filter_misadjustment_stable_frames) return;
    if (a->misadj_inv <= 10.0f) return;
    float base = a->misadj_inv > 1e-6f ? a->misadj_inv : 1e-6f;
    float scale_raw = 2.0f / sqrtf(base);
    float scale = scale_raw;
    if (scale < a->tun.filter_misadjustment_scale_min) scale = a->tun.filter_misadjustment_scale_min;
    if (scale > a->tun.filter_misadjustment_scale_max) scale = a->tun.filter_misadjustment_scale_max;
    pbfdkf_scale_filter(&a->main_filter, scale);
    a->misadj_inv = 0.0f; a->misadj_overhang = 0;
    a->misadj_hangover_remaining = a->tun.filter_misadjustment_hangover_frames;
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
    if (aec_unusable(a)) return 0;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_is_cancelling(a);
    return (a->last_erle_windowed > 2.5f)
           || (aec_erle_ring_max_last15(a) > a->tun.delay_acquire_inst_erle_db);
}

/* ───────────────────────── process ─────────────────────────────────────── */

void aec_get_last_timing(const Aec* a, AecStageTiming* out) {
    if (!out) return;
    if (aec_unusable(a)) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = a->last_timing;
}

/* Matched-filter duty-cycle gate (AecDutyGate), shared with the array flow. The ERLE watchdog keeps a leaky peak (0.1 dB/s);
 * a drop of more than AEC_DUTY_ERLE_COLLAPSE_DB resumes full-rate analysis, armed once the peak exceeds it. */
float aec_duty_leak_db(int hop_size, int sample_rate) {
    return aec_legacy10ms_rate(0.001f, hop_size, sample_rate);
}

void aec_duty_configure(AecDutyGate* g, float hold_s, float period_s, int hop_size, int sample_rate) {
    const float hop_s = (float)hop_size / (float)sample_rate;
    const int hold = (int)lrintf(hold_s / hop_s);
    const int period = (int)lrintf(period_s / hop_s) / AEC_DUTY_PERIOD_DIVISOR;
    g->hold_hops = hold < 1 ? 1 : hold;
    g->period_hops = period < 1 ? 1 : period;
    g->erle_leak_db = aec_duty_leak_db(hop_size, sample_rate);
}

void aec_duty_reset(AecDutyGate* g) {
    g->active = 0;
    g->stable_hops = 0;
    g->pos = 0;
    g->last_delay = -1;
    g->erle_peak = 0.0f;
    g->hops_run = 0;
}

int aec_duty_begin(AecDutyGate* g) {
    int run_filter = 1;
    if (g->active) {
        g->pos += 1;
        if (g->pos >= g->period_hops) g->pos = 0;
        run_filter = (g->pos == 0);
    }
    if (run_filter) g->hops_run += 1;
    return run_filter;
}

void aec_duty_end(AecDutyGate* g, int cur_delay, int solid, float erle) {
    if (!solid || cur_delay < 0 || cur_delay != g->last_delay) {
        g->active = 0;
        g->stable_hops = 0;
    } else if (!g->active) {
        g->stable_hops += 1;
        if (g->stable_hops >= g->hold_hops) {
            g->active = 1;
            g->pos = 0;
            g->erle_peak = erle;
        }
    }
    g->last_delay = cur_delay;
    if (g->active) {
        if (erle > g->erle_peak)
            g->erle_peak = erle;
        else
            g->erle_peak -= g->erle_leak_db;
        if (g->erle_peak > AEC_DUTY_ERLE_COLLAPSE_DB && erle < g->erle_peak - AEC_DUTY_ERLE_COLLAPSE_DB) {
            g->active = 0;
            g->stable_hops = 0;
            g->erle_peak = 0.0f;
        }
    }
}

static void aec_update_delay_estimator(Aec* a) {
    const int run_filter = aec_duty_begin(&a->duty);
    /* Engagement census, counted where run_filter is consumed. */
    a->duty_hops_total += 1;
    delay_aec_accumulate_ex(&a->delay, a->near_hop, a->far_hop, a->hop_size, run_filter);
    aec_duty_end(&a->duty, delay_aec_estimated_delay(&a->delay), delay_aec_is_solid(&a->delay), a->last_erle_windowed);
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
                aec_apply_delay_shift_soft_revert(
                    a, a->delay_shift_judgment_pre_delay, bad_delay);
                /* The ring read offset moved back; seam consumers must see CHANGED. */
                aec_bump_delay_generation(a);
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
    int already_cancelling = a->tun.delay_acquire_protect_converged
                             && (a->last_erle_windowed > 2.5f);
    if (eligible && a->current_delay < 0 && delay_aec_is_solid(&a->delay)
            && !already_cancelling) {
        a->current_delay = new_delay;
        aec_bump_delay_generation(a);
        /* Warm tap-transfer: if the inst-ERLE peak exceeds the threshold and
         * the delay fits the tap reach, shift the learned IR instead of
         * zeroing it. */
        int warm_ok = 0;
        if (a->tun.delay_acquire_warm_transfer) {
            float wpk = aec_erle_ring_max_last15(a);
            int reach = pbfdaf_tap_span(&a->main_filter.base);
            warm_ok = (wpk > a->tun.delay_acquire_inst_erle_db)
                      && (new_delay > 0) && (new_delay < reach);
        }
        /* A refused shift leaves the taps in place: fall through to the soft
         * / hard branches. */
        if (warm_ok
            && pbfdaf_warm_shift_ir(&a->main_filter.base, new_delay) == 0) {
            if (a->has_shadow) pbfdaf_warm_shift_ir(&a->shadow_filter, new_delay);
        } else if (a->tun.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
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
            && abs(new_delay - a->current_delay) > AEC_ARRAY_DELAY_CHANGE_MIN_SAMPLES
            && !aec_delay_value_rejected(a, new_delay)) {
        if (a->has_pending && abs(new_delay - a->pending_delay) < AEC_ARRAY_DELAY_CHANGE_CONFIRM_SAMPLES) {
            int old_current_delay = a->current_delay;
            /* Snapshot the taps and the main H_error (a statistic of the taps)
             * so a confirmed-bad relock can restore them; pre_erle is the
             * reading just before the shift. */
            float pre_shift_erle = a->last_erle_windowed;
            size_t w_elems = (size_t)a->n_partitions * (size_t)a->n_freqs;
            memcpy(a->delay_shift_judgment_pre_W, a->main_filter.base.W,
                   w_elems * sizeof(Complex));
            if (a->has_shadow)
                memcpy(a->delay_shift_judgment_pre_shadow_W,
                       a->shadow_filter.W, w_elems * sizeof(Complex));
            memcpy(a->delay_shift_judgment_pre_H_error,
                   a->main_filter.H_error_per_bin,
                   (size_t)a->n_freqs * sizeof(float));
            a->current_delay = new_delay;
            aec_bump_delay_generation(a);
            a->has_pending = 0; a->pending_delay = -1; a->pending_delay_ttl = 0;
            if (a->tun.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
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
            a->pending_delay = new_delay; a->has_pending = 1; a->pending_delay_ttl = AEC_ARRAY_DELAY_CHANGE_CANDIDATE_TTL;
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
        aec_ring_write(a->ref_ring, rs, w, a->far_hop, hop);
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
            aec_ring_read(a->ref_ring, rs, read_pos, a->far_hop, hop);
            a->far_hop_aligned = 1;

            /* Live observer: same ring read observer_shift_hops hops earlier,
             * clamped at 0. */
            if (a->has_observer) {
                long long obs_back = (long long)a->observer_shift_hops * hop;
                int d_obs = obs_back >= d ? 0 : d - (int)obs_back;
                int read_pos_obs = (a->ref_ring_write - hop - d_obs) % rs;
                if (read_pos_obs < 0) read_pos_obs += rs;
                aec_ring_read(a->ref_ring, rs, read_pos_obs, a->observer_far_hop, hop);
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

/* `ana` (non-NULL): reuse the donor's results of the stages that read the far hop alone; `out`: publish this lane's. */
static void aec_prepare_capture_hop(Aec* a, const float* mic_in, const float* ref_in,
                                    const AecFarShare* ana, AecFarShare* out) {
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
        if (ne_ind > a->tun.ne_recent_threshold) {
            if (a->ne_above < a->tun.ne_recent_sustain)
                a->ne_above += 1;
        } else
            a->ne_above = 0;
        if (a->ne_above >= a->tun.ne_recent_sustain)
            a->ne_recent_frames = a->tun.ne_recent_hold;
        else if (a->ne_recent_frames > 0)
            a->ne_recent_frames -= 1;
        else
            a->ne_recent_frames = 0;
    }

    float sat_ref = 0.0f;
    if (a->has_sat) {
        if (ana) {
            a->sat_ref = ana->donor->sat_ref;
            sat_ref = ana->sat_ref;
        } else {
            sat_ref = saturation_detect(&a->sat_ref, a->far_hop, hop);
        }
        if (out) out->sat_ref = sat_ref;
        float sat_mic = saturation_detect(&a->sat_mic, a->near_hop, hop);
        a->saturation_level = (sat_ref > sat_mic * 0.5f) ? sat_ref : sat_mic * 0.5f;
        if (a->tun.saturation_softclip_ref && sat_ref > 0.1f) {
            if (ana) memcpy(a->far_hop, ana->donor->far_hop, (size_t)hop * sizeof(float));
            else saturation_soft_clip(a->far_hop, a->far_hop, hop, 0.8);
        }
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

static int aec_prepare_rsa(Aec* a, const AecFarShare* ana, AecFarShare* out) {
    const int hop = a->hop_size, K = a->n_freqs;
    int poor, rsa_mask_active;
    /* RSA update, poor-excitation counters and saturated_capture (PSD from
     * the previous hop). */
    {
        if (ana) {
            poor = ana->rsa_poor;
        } else {
            /* far_cmag2_hold still holds the previous hop's |far_spec|^2 (the
             * frontend has not run). */
            rsa_update(&a->rsa, a->main_filter.base.far_cmag2_hold, a->far_hop, hop);
            poor = rsa_poor_signal_excitation(&a->rsa);
        }
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
    if (ana) {
        rsa_mask_active = ana->rsa_mask_active;
    } else {
        float *rsa_mask = a->scr_rsa_mask;
        rsa_mask_active = rsa_any_narrow_band_active(&a->rsa);
        if (rsa_mask_active) {
            for (int k = 0; k < K; ++k) rsa_mask[k] = 1.0f;
            rsa_mask_regions_around_narrow_bands(&a->rsa, rsa_mask);
        }
    }
    if (out) {
        out->rsa_poor = poor;
        out->rsa_mask_active = rsa_mask_active;
    }

    return rsa_mask_active;
}

static int aec_run_shadow(Aec* a, float far_hop_mean_sq, int rsa_mask_active, const float* rsa_mask,
                          const Complex* shared_far_spec,
                          const float* shared_far_cmag2) {
    const int K = a->n_freqs;
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
                                int rsa_mask_active, const float* rsa_mask, const Complex* shared_far_spec,
                                const float* shared_far_cmag2) {
    const int K = a->n_freqs;
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

static int aec_refresh_stationarity(Aec* a, const AecFarShare* hop_in, const AecFarShare* ana, AecFarShare* out) {
    const int hop = a->hop_size;
    int stationarity_block_for_post;
    /* Stationarity refresh for the next hop. */
    {
        int floor_hit = hop_in ? hop_in->far_floor_hit
                               : (sk_max_abs_f32(a->far_hop, hop) >= a->render_peak_floor);
        if (out) out->far_floor_hit = floor_hit;
        if (!a->non_zero_render_seen && floor_hit)
            a->non_zero_render_seen = 1;
        if (a->non_zero_render_seen) {
            const float *far_psd = a->main_filter.base.far_cmag2_hold;
            if (ana) {
                /* The noise floor is the donor's, already updated with this same PSD; this lane's own copy is not kept. */
                float *own = a->a3_stat.noise.noise;
                a->a3_stat.noise.noise = ana->donor->a3_stat.noise.noise;
                stationarity_estimator_update_stationarity_flags(&a->a3_stat, far_psd,
                                                                  a->post.avg_reverb.reverb);
                a->a3_stat.noise.noise = own;
            } else {
                stationarity_estimator_update_noise_estimator(&a->a3_stat, far_psd);
                stationarity_estimator_update_stationarity_flags(&a->a3_stat, far_psd,
                                                                  a->post.avg_reverb.reverb);
            }
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
        float as = a->tun.shadow_err_alpha, oas = 1.0f - as;
        a->main_err_smooth   = as * a->main_err_smooth   + oas * main_err;
        a->shadow_err_smooth = as * a->shadow_err_smooth + oas * shadow_err;
        doubletalk_update_shadow_dt(&a->dt_analyzer, aec_count_as_int(a->shadow_frame_count),
                                    far_excited, a->main_err_smooth, a->shadow_err_smooth);
        int delay_reliable = a->has_delay && (delay_aec_confidence(&a->delay) >= 0.5f);
        /* Shadow-copy warm-up is a 500 ms wall-clock guard, converted per grid. */
        int shadow_warmup_hops =
            aec_ms_to_hops(500.0f, hop, a->cfg.sample_rate);
        shadow_copy_update(
            &a->regime, aec_count_as_int(a->shadow_frame_count), shadow_warmup_hops,
            far_hop_mean_sq,
            a->main_err_smooth, a->shadow_err_smooth,
            a->epc.active, a->saturation_level,
            a->dt_analyzer.dt_from_energy, 0.0, delay_reliable);
    }

}

/* Echo-path-change recovery shared by the EPV and shadow_rise triggers:
 * aggressive Kalman re-adapt, mark_diverged, ERLE-side reset. */
static void aec_epc_gain_change_recovery(Aec* a) {
    a->pending_gain_change = 1;
    pbfdkf_handle_echo_path_change(&a->main_filter, 1, 0);
    filter_convergence_mark_diverged(&a->convergence);
    a->epc_render_forced_remaining = a->tun.epc_hangover;
    if (a->erl_estimate > 0.3f) a->erl_estimate = 0.3f;
}

static void aec_update_echo_path(Aec* a, float far_pwr_global) {
    /* EPV trigger. */
    EpcEvent epv = epc_update_epv(&a->epc, far_pwr_global,
                                  a->convergence.converged, a->regime.main_paused);
    if (epv.fired && a->tun.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
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
        if (rise.fired && a->tun.dt_aware_recovery_soft && a->ne_recent_frames > 0) {
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
    if (a->cfg.residual == AEC_RESIDUAL_OFF)
        memcpy(a->final_out, a->raw_output, (size_t)hop * sizeof(float));

    misadj_update(a, a->near_hop, a->raw_output, hop);
    misadj_fire(a);

    /* inst-ERLE slope ring: appended every freq-path hop outside the
     * residual stage (the warm-transfer gate needs it with the residual stage off); uses
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
     * the residual mode. */
    const float erle_decay = 0.999f;
    a->erle_window_near = erle_decay * a->erle_window_near + a->near_power;
    a->erle_window_err  = erle_decay * a->erle_window_err  + a->raw_error_power;
    float erle_windowed = 10.0f * log10f((a->erle_window_near + 1e-10f)
                                        / (a->erle_window_err + 1e-10f));
    a->last_erle_windowed = erle_windowed;
    filter_convergence_update_divergence(&a->convergence, a->near_power, a->raw_error_power);
    return erle_windowed;
}

static void aec_run_post_chain(Aec* a, int stationarity_block_for_post, int render_active, int apply,
                               const AecFarShare* share) {
    const int hop = a->hop_size, N = a->n_partitions;
    {
        /* filter_taps_full is filled on demand by aec_fill_filter_taps(). */

        AecPostRunIn in;
        memset(&in, 0, sizeof(in));
        in.near_spec = a->main_filter.base.near_spec;
        in.far_spec  = a->main_filter.base.far_spec;
        in.echo_spec = a->main_filter.base.echo_spec;
        in.error_spec_windowed = a->main_filter.base.error_spec_windowed;
        in.W_all = a->main_filter.base.W;
        in.x2_cache = a->main_filter.base.x2_cache;
        in.sqrt_hann = a->main_filter.base.sqrt_hann;
        in.partition_idx = a->main_filter.base.partition_idx;
        in.n_partitions = N;
        in.filter_taps_full = a->filter_taps_full;
        in.filter_taps_full_len = N * hop;
        in.shadow_present = a->has_shadow;
        in.shadow_error_spec = a->has_shadow ? a->shadow_filter.error_spec : NULL;
        in.last_shadow_output_time = a->has_shadow ? a->shadow_out : NULL;
        in.s_ref_max = a->main_filter.base.last_s_max_abs * 32768.0f;
        in.s_coa_max = a->has_shadow ? a->shadow_filter.last_s_max_abs * 32768.0f : 0.0f;
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
        /* Without `apply` the caller consumes comfort noise, R^2 and the gain (aec_get_residual()); only final
         * synthesis is skipped. Below FULL the gain is never computed (apply is 0 there). */
        in.context_only = !apply;
        in.hold_state_only = (a->cfg.residual == AEC_RESIDUAL_STATE ||
                                     a->cfg.residual == AEC_RESIDUAL_ESTIMATE);
        in.linear_context_only = (a->cfg.residual == AEC_RESIDUAL_STATE);
        in.active_render_threshold = AECB_ACTIVE_RENDER_THRESHOLD;

        AecPostRunObj obj;
        obj.state = &a->a3_state; obj.ree = &a->a3_ree; obj.sg = &a->a3_sg;
        obj.stationarity = &a->a3_stat; obj.lfs = &a->a3_lfs; obj.fft = a->post_fft;

        /* DT-gated min-gain floor lift: aec_post_run owns the final decision. */
        in.dt_floor_enabled = a->tun.dt_aware_res_floor_enabled;
        in.nearend_recent = a->ne_recent_frames > 0;

        /* Live observer: obj.observer_filter runs inside aec_post_run, which
         * knows this hop's dominant_ne / usable_linear verdicts. */
        in.render_activity_is_active = render_active;
        if (share && share->post) {
            /* Same far spectrum and far hop as the donor's, so its PSD, scaled block and verdict are this lane's. */
            in.shared_far_psd = share->donor->post.far_psd;
            in.shared_render_block = share->donor->a3_sc.render_block_scaled;
            in.shared_far_active = share->donor->post.far_active;
        }
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
        float mu_min = a->tun.shadow_mu_min;
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
    if (a->cfg.residual == AEC_RESIDUAL_OFF) {
        if (a->convergence.converged) {
            int k;
            float mu_min = a->tun.shadow_mu_min;
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
        (void)filter_convergence_update_convergence(
            &a->convergence, a->near_power, a->raw_error_power,
            far_active, a->warmup_frames_remaining <= 0);
    }

    aec_counter_bump(&a->frame_count);
}

/* Stage 1 of a hop: front end, delay, shadow, main filter and the estimators that follow them. Leaves the locals the
 * residual stage needs in a->hop_carry. */
static void aec_stage_linear(Aec* a, const float* mic_in, const float* ref_in,
                             const Complex* shared_far_spec,
                             const float* shared_far_cmag2,
                             const AecFarShare* ana, AecFarShare* out) {
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
    const AecFarShare* hop_in = (ana && ana->hop) ? ana : NULL;
    ana = (ana && ana->analysis) ? ana : NULL;
    aec_prepare_capture_hop(a, mic_in, ref_in, hop_in, out);

    t_delay = aec_now_us();
    aec_align_reference(a);

    t_delay = aec_now_us() - t_delay;
    a->last_timing.delay_us = t_delay;

    float far_hop_mean_sq;
    RenderActivityResult ra;
    if (hop_in) {
        const RenderActivity* d = &hop_in->donor->render_activity;
        far_hop_mean_sq = hop_in->far_mean_sq;
        ra = hop_in->ra;
        a->render_activity.env_mean = d->env_mean;
        a->render_activity.env_var = d->env_var;
        a->render_activity.active_prev = d->active_prev;
        a->render_activity.is_stationary = d->is_stationary;
    } else {
        far_hop_mean_sq = mean_sq(a->far_hop, hop, a->scr_sq);
        ra = render_activity_update(&a->render_activity, a->far_hop, hop);
    }
    if (out) {
        out->far_mean_sq = far_hop_mean_sq;
        out->ra = ra;
    }
    a->warmup_far_active = ra.warmup_active;
    float far_pwr_global = ra.far_pwr;

    int mu_is_array;
    float mu_scalar = aec_prepare_mu(a, &mu_is_array);

    int rsa_mask_active = aec_prepare_rsa(a, ana, out);
    const float* rsa_mask = ana ? ana->donor->scr_rsa_mask : a->scr_rsa_mask;

    int far_excited = aec_run_shadow(a, far_hop_mean_sq, rsa_mask_active, rsa_mask,
                                     shared_far_spec, shared_far_cmag2);

    t_mark = aec_now_us();
    /* Frontend = pre-filter window - delay_us (t_delay is the local so it
     * folds when timing is off). */
    a->last_timing.frontend_us = (t_mark - t_stage) - t_delay;

    int main_error_cmag2_valid = aec_run_main_filter(
        a, mu_scalar, mu_is_array, rsa_mask_active, rsa_mask, shared_far_spec, shared_far_cmag2);

    {
        uint32_t t_lin_end = aec_now_us();
        a->last_timing.linear_us = t_lin_end - t_mark;
        t_mark = t_lin_end;
    }

    int stationarity_block_for_post = aec_refresh_stationarity(a, hop_in, ana, out);

    aec_update_shadow_quality(a, main_error_cmag2_valid);

    aec_update_shadow_dt(a, far_hop_mean_sq, far_excited);

    aec_update_echo_path(a, far_pwr_global);

    float _ei = aec_update_misadjustment_erle(a);

    /* The steering window closes here; the residual stage's window opens at the same stamp when it follows at once. */
    t_stage = aec_now_us();
    a->last_timing.steering_us = t_stage - t_mark;
    float erle_windowed = aec_update_windowed_erle(a);

    a->hop_carry.far_hop_mean_sq = far_hop_mean_sq;
    a->hop_carry.erle_inst = _ei;
    a->hop_carry.erle_windowed = erle_windowed;
    a->hop_carry.render_active = ra.is_active;
    a->hop_carry.stationarity_block_for_post = stationarity_block_for_post;
    a->hop_carry.t_stage = t_stage;
}

/* Stage 2 of a hop: the residual block (FORM, AecState, R^2, comfort noise, G_res, as far as AecConfig.residual reaches)
 * and the hop's closing bookkeeping. `apply` also applies the gain, adds the comfort noise and synthesizes into
 * final_out (aec_process() with AEC_RESIDUAL_FULL); `t_stage` opens the residual timing window. */
static void aec_stage_residual(Aec* a, int apply, uint32_t t_stage, const AecFarShare* share) {
    const struct AecHopCarry* c = &a->hop_carry;
    int is_stationary_dt = 0;
    if (a->cfg.residual != AEC_RESIDUAL_OFF) {
        is_stationary_dt = aec_prepare_post_dt(a, c->far_hop_mean_sq, c->erle_inst, c->erle_windowed, c->render_active);
        aec_run_post_chain(a, c->stationarity_block_for_post, c->render_active, apply, share);

        aec_update_post_mu(a, is_stationary_dt);
        a->residual_valid = 1;
    }
    aec_update_linear_only_steering(a, c->far_hop_mean_sq, c->render_active);

    a->last_timing.res_us = aec_now_us() - t_stage;

    aec_finish_hop(a, c->far_hop_mean_sq);
}

/* A non-finite input sample is refused before any state changes (an instance cannot recover from one without a reset). */
static int aec_input_finite(const Aec* a, const float* mic, const float* ref) {
    return skn_all_finite_f32(mic, (size_t)a->hop_size) && skn_all_finite_f32(ref, (size_t)a->hop_size);
}

/* Processes exactly hop_size samples: both stages plus the copy to out; no output limiter. */
int aec_process(Aec* a, const float* mic_in, const float* ref_in, float* out) {
    if (aec_unusable(a) || !mic_in || !ref_in || !out) return AEC_ERR_ARG;
    if (AEC_IS_ARRAY(a)) return AEC_ERR_MODE;   /* an array takes four microphones: aec_process_linear() */
    const int hop = a->hop_size;
    if (a->hop_phase != 0) return AEC_ERR_SEQUENCE;
    if (!aec_input_finite(a, mic_in, ref_in)) return AEC_ERR_ARG;
    const int apply = (a->cfg.residual == AEC_RESIDUAL_FULL);
    aec_stage_linear(a, mic_in, ref_in, NULL, NULL, NULL, NULL);
    aec_stage_residual(a, apply, a->hop_carry.t_stage, NULL);

    memcpy(out, a->final_out, (size_t)hop * sizeof(float));
    return AEC_OK;
}

/* Ends stage 1: with AEC_RESIDUAL_OFF there is no residual stage, so the hop closes here; otherwise it stays open. */
static void aec_close_linear(Aec* a) {
    if (a->cfg.residual == AEC_RESIDUAL_OFF) aec_stage_residual(a, 0, a->hop_carry.t_stage, NULL);
    else a->hop_phase = 1;
}

/* A far-spectrum donor this instance can borrow from: another single-channel instance on the same grid. */
static int aec_far_donor_ok(const Aec* a, const Aec* d) {
    return d && d != a && d->ready && !d->array && d->n_freqs == a->n_freqs && d->hop_size == a->hop_size &&
           d->cfg.sample_rate == a->cfg.sample_rate;
}

/* Stage 1 only. */
int aec_process_linear(Aec* a, const float* mic_in, const float* ref_in) {
    if (aec_unusable(a) || !mic_in || !ref_in) return AEC_ERR_ARG;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_process_linear(a, mic_in, ref_in);
    return aec_process_linear_shared_far_from(a, mic_in, ref_in, NULL);
}

int aec_process_linear_shared_far_from(
        Aec* a, const float* mic_in, const float* ref_in, const Aec* far_owner) {
    if (aec_unusable(a) || !mic_in || !ref_in) return AEC_ERR_ARG;
    if (AEC_IS_ARRAY(a)) return AEC_ERR_MODE;
    if (a->hop_phase != 0) return AEC_ERR_SEQUENCE;
    if (!aec_input_finite(a, mic_in, ref_in)) return AEC_ERR_ARG;
    if (!aec_far_donor_ok(a, far_owner)) {
        aec_stage_linear(a, mic_in, ref_in, NULL, NULL, NULL, NULL);
    } else {
        aec_stage_linear(a, mic_in, ref_in, far_owner->main_filter.base.far_spec,
                         far_owner->main_filter.base.far_cmag2_hold, NULL, NULL);
    }
    aec_close_linear(a);
    return AEC_OK;
}

#ifndef AEC_NO_ARRAY
int aec_lane_process_linear(Aec* a, const float* mic_in, const float* ref_in, AecFarShare* share, int publish) {
    const AecFarShare* in = NULL;
    const Complex* far_spec = NULL;
    const float* far_cmag2 = NULL;
    if (!a || !share || !mic_in || !ref_in || a->array || a->hop_phase != 0) return AEC_ERR_ARG;
    if (!publish) {
        const Aec* d = share->donor;
        if (!aec_far_donor_ok(a, d)) return AEC_ERR_ARG;
        if (share->fft) {
            far_spec = d->main_filter.base.far_spec;
            far_cmag2 = d->main_filter.base.far_cmag2_hold;
        }
        if (share->analysis && !share->fft) return AEC_ERR_ARG;
        if (share->hop || share->analysis) in = share;
    }
    aec_stage_linear(a, mic_in, ref_in, far_spec, far_cmag2, in, publish ? share : NULL);
    aec_close_linear(a);
    return AEC_OK;
}

int aec_lane_process_residual(Aec* a, const AecFarShare* share) {
    if (!a || a->array || a->cfg.residual == AEC_RESIDUAL_OFF) return AEC_ERR_ARG;
    if (a->hop_phase != 1) return AEC_ERR_SEQUENCE;
    a->hop_phase = 0;
    aec_stage_residual(a, 0, aec_now_us(), share);
    return AEC_OK;
}

void aec_lane_adopt_far_analysis(Aec* a, const Aec* d) {
    if (!a || !d || a == d) return;
    memcpy(a->rsa.counters, d->rsa.counters, (size_t)a->rsa.n_counters * sizeof(int64_t));
    a->rsa.narrow_peak_band = d->rsa.narrow_peak_band;
    a->rsa.narrow_peak_counter = d->rsa.narrow_peak_counter;
    memcpy(a->a3_stat.noise.noise, d->a3_stat.noise.noise, (size_t)a->a3_stat.noise.n_freqs * sizeof(float));
    a->a3_stat.noise.block_counter = d->a3_stat.noise.block_counter;
}

int aec_lane_far_history_equal(const Aec* a, const Aec* b) {
    size_t blk, bins;
    if (!a || !b || a->array || b->array || a->has_shadow != b->has_shadow ||
        a->main_filter.base.block_size != b->main_filter.base.block_size || a->n_freqs != b->n_freqs) return 0;
    blk = (size_t)a->main_filter.base.block_size * sizeof(float);
    bins = (size_t)a->n_freqs * sizeof(float);
    if (memcmp(a->main_filter.base.far_buffer, b->main_filter.base.far_buffer, blk) != 0 ||
        memcmp(a->main_filter.base.far_cmag2_hold, b->main_filter.base.far_cmag2_hold, bins) != 0) return 0;
    if (a->has_shadow &&
        (memcmp(a->shadow_filter.far_buffer, b->shadow_filter.far_buffer,
                (size_t)a->shadow_filter.block_size * sizeof(float)) != 0 ||
         memcmp(a->shadow_filter.far_cmag2_hold, b->shadow_filter.far_cmag2_hold, bins) != 0)) return 0;
    return 1;
}
#endif /* !AEC_NO_ARRAY */

/* Stage 2 of a hop begun by aec_process_linear(); the caller applies the gain itself. */
int aec_process_residual(Aec* a, const AecBeam* beam) {
    if (aec_unusable(a)) return AEC_ERR_ARG;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_process_residual(a, beam);
    if (beam) return AEC_ERR_ARG;
    if (a->cfg.residual == AEC_RESIDUAL_OFF) return AEC_ERR_MODE;
    if (a->hop_phase != 1) return AEC_ERR_SEQUENCE;
    a->hop_phase = 0;
    aec_stage_residual(a, 0, aec_now_us(), NULL);
    return AEC_OK;
}

int aec_get_residual(const Aec* a, AecResidualView* out) {
    /* The caller's declared size is validated before anything is written: a short struct is left untouched, a
     * larger one gets only the known prefix. */
    if (!out || out->struct_size < sizeof(*out)) return AEC_ERR_ARG;
    memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    if (aec_unusable(a)) return AEC_ERR_ARG;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_get_residual(a, out);
    if (a->cfg.residual == AEC_RESIDUAL_OFF) return AEC_ERR_MODE;
    if (a->hop_phase != 0 || !a->residual_valid) return AEC_ERR_SEQUENCE;
    out->n_freqs = a->n_freqs;
    out->hop_size = a->hop_size;
    out->formed_hop  = a->a3_lfs.e_form;
    out->error_spec  = a->a3_sc.sel_esw;
    out->echo_spec   = a->a3_sc.sel_echo;
    out->near_spec   = a->a3_sc.ybase;
    out->far_spec    = a->main_filter.base.far_spec;
    out->guard_active = a->post.output_guard_active;
    out->base_spec   = out->guard_active ? a->a3_sc.output_base : a->a3_sc.sel_esw;
    if (a->cfg.residual >= AEC_RESIDUAL_ESTIMATE) {
        out->r2            = a->a3_sc.r2;
        out->comfort_noise = a->post.comfort_noise;
        out->res_floor_protect = a->a3_sg.dt_protect_active;
    }
    if (a->cfg.residual == AEC_RESIDUAL_FULL) out->res_gain = a->a3_sg.gain;
    out->usable_linear = aec_state_usable_linear_estimate(&a->a3_state);
    out->post_reset_serial = a->post.reset_serial;
    /* The gain stage's inputs, from this instance's own linear stage. */
    {
        AecLinearView lv;
        memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
        (void)aec_get_linear(a, &lv);
        out->all_converged = lv.filter_converged;
        out->max_dt = lv.dt_indicator;
        out->max_saturation = lv.saturation_level;
        out->clock_drift = (a->cfg.delay_mode == AEC_DELAY_MATCHED && a->has_delay) ? delay_aec_has_clockdrift(&a->delay) : 0;
        out->delay_changed = (lv.delay_state == AEC_LINEAR_DELAY_CHANGED);
        out->delay_locked = (lv.delay_state != AEC_LINEAR_DELAY_UNLOCKED);
        out->guard_lanes = out->guard_active ? 1u : 0u;
        out->far_hop = a->far_hop;
    }
    return AEC_OK;
}

/* G_res and the rendering base of the last completed hop, or the reason there are none. */
static int aec_gain_ready(const Aec* a) {
    if (aec_unusable(a)) return AEC_ERR_ARG;
    if (a->cfg.residual != AEC_RESIDUAL_FULL) return AEC_ERR_MODE;
    if (a->hop_phase != 0 || !a->residual_valid) return AEC_ERR_SEQUENCE;
    return AEC_OK;
}

/* Bins per chunk of the shared comfort-noise core: the N^2 -> amplitude conversion is vectorised per chunk, the random phase
 * draws stay strictly sequential in bin order. */
#define AEC_CNG_CHUNK 64

void aec_apply_gain_spec(Complex* out, const Complex* base, const float* g_res, const float* g_extra, int n) {
    if (!out || !base || !g_res || n <= 0) return;
    if (!g_extra) {
        sk_capply_gain_f32(out, base, g_res, (size_t)n);
        return;
    }
    {
        /* min then multiply, block by block with the same kernels the samples used (non-finite gains stay visible). */
        float total[AEC_CNG_CHUNK];
        int k = 0;
        while (k < n) {
            int m = (n - k < AEC_CNG_CHUNK) ? (n - k) : AEC_CNG_CHUNK;
            sk_min_f32(total, g_extra + k, g_res + k, (size_t)m);
            sk_capply_gain_f32(out + k, base + k, total, (size_t)m);
            k += m;
        }
    }
}

void aec_cng_fill(uint32_t* rng, Complex* spec, const float* cn,
                  const float* g_res, const float* g_nr, float nr_floor, int n) {
    static const float inv_psd_scale = (float)AEC_PSD_SCALE_INV;     /* comfort_noise is on the int16^2 scale */
    float pw[AEC_CNG_CHUNK], amp[AEC_CNG_CHUNK];
    uint32_t seed;
    int k0;
    if (!rng || !spec || !cn || !g_res || n < 3) return;
    seed = *rng;
    for (k0 = 1; k0 < n - 1; k0 += AEC_CNG_CHUNK) {
        int m = n - 1 - k0, j;
        if (m > AEC_CNG_CHUNK) m = AEC_CNG_CHUNK;
        /* amp[j] = sqrt(max(cn / 32768^2, 0)); the square root reads pw and writes amp (no in-place call) */
        sk_scale_f32(pw, cn + k0, inv_psd_scale, m);
        sk_clip_f32(pw, 0.0f, (float)INFINITY, m);
        sk_fast_sqrt_f32(pw, amp, m);
        for (j = 0; j < m; ++j) {
            int bin = k0 + j;
            uint32_t ix;
            float ng, cn_re, cn_im, t;
            seed = (seed * 69069u + 1u) & 0x7FFFFFFFu;
            ix = seed >> 26;
            cn_re = amp[j] * AECB_SQRT2_SIN_LUT[ix];
            cn_im = amp[j] * AECB_SQRT2_SIN_LUT[(ix + 8u) & 31u];
            t = 1.0f - g_res[bin] * g_res[bin];
            if (t < 0.0f) t = 0.0f;
            ng = fast_sqrt(t);
            if (g_nr) ng = ng * fmaxf(g_nr[bin], nr_floor);
            spec[bin].r += ng * cn_re;
            spec[bin].i += ng * cn_im;
        }
    }
    *rng = seed;
}

void aec_apply_cng_spec(uint32_t* rng, Complex* spec, const float* cn, const float* g_res, const float* g_extra, int n) {
    aec_cng_fill(rng, spec, cn, g_res, g_extra, AEC_CNG_NR_GAIN_FLOOR, n);
}

int aec_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec) {
    if (!aec_unusable(a) && AEC_IS_ARRAY(a)) return aec_array_impl_apply_gain(a, extra_gain, out_spec);
    int rc = aec_gain_ready(a);
    if (rc != AEC_OK) return rc;
    if (!out_spec) return AEC_ERR_ARG;
    const Complex* base = a->post.output_guard_active ? a->a3_sc.output_base : a->a3_sc.sel_esw;
    aec_apply_gain_spec(out_spec, base, a->a3_sg.gain, extra_gain, a->n_freqs);
    return AEC_OK;
}

int aec_apply_cng(Aec* a, const float* extra_gain, Complex* spec) {
    if (!aec_unusable(a) && AEC_IS_ARRAY(a)) return aec_array_impl_apply_cng(a, extra_gain, spec);
    int rc = aec_gain_ready(a);
    if (rc != AEC_OK) return rc;
    if (!spec) return AEC_ERR_ARG;
    aec_apply_cng_spec(&a->post.cng_seed, spec, a->post.comfort_noise, a->a3_sg.gain, extra_gain, a->n_freqs);
    return AEC_OK;
}

uint32_t aec_post_reset_serial(const Aec* a) {
    if (aec_unusable(a)) return 0u;
    if (AEC_IS_ARRAY(a)) return aec_array_impl_post_reset_serial(a);
    return a->post.reset_serial;
}

int aec_synthesis_window(const Aec* a, const float** window, int* length) {
    if (aec_unusable(a)) return -1;
    if (window && length && AEC_IS_ARRAY(a)) return aec_array_impl_synthesis_window(a, window, length);
    if (!window || !length || !a->post.synth_window) return -1;
    *window = a->post.synth_window;
    *length = a->block_size;
    return 0;
}

/* Alignment confidence shared by AecDebugStatus and AecLinearView: MATCHED
 * reports the estimator's value; FIXED and EXTERNAL_ALIGNED report 1.0
 * (caller contract). */
static float aec_delay_confidence(const Aec* a) {
    if (a->cfg.delay_mode == AEC_DELAY_MATCHED)
        return delay_aec_confidence(&a->delay);
    return 1.0f;
}

/* Read-only AecDebugStatus snapshot. */
void aec_debug_status(const Aec* a, AecDebugStatus* out) {
    if (!out) return;
    if (aec_unusable(a)) {
        memset(out, 0, sizeof(*out));
        return;
    }
    if (AEC_IS_ARRAY(a)) { aec_array_impl_debug_status(a, out); return; }
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
    out->duty_hops_run   = a->duty.hops_run;

    out->n_partitions       = a->n_partitions;
    out->delay_period_hops  = a->duty.period_hops;
}

/* Read-only AecLinearView; pointers alias per-hop buffers, valid
 * until the next process/reset. */
int aec_get_linear(const Aec* a, AecLinearView* out) {
    if (!out || out->struct_size < sizeof(*out)) return AEC_ERR_ARG;
    if (!aec_unusable(a) && AEC_IS_ARRAY(a)) return aec_array_impl_get_linear(a, out);
    memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(*out);
    if (aec_unusable(a)) return AEC_ERR_ARG;
    out->hop_size = a->hop_size;
    out->n_freqs = a->n_freqs;
    out->linear_hop = a->raw_output;
    out->echo_spec = a->main_filter.base.echo_spec;
    out->near_spec = a->main_filter.base.near_spec;
    out->far_spec = a->main_filter.base.far_spec;
    out->aligned_far_hop = a->far_hop;
    out->generation = a->delay_generation;
    out->far_power = a->last_far_power;
    out->erle_factor = a->erle_factor_prev;
    out->dt_indicator = a->last_dt_indicator;
    out->divergence = a->convergence.divergence;
    out->saturation_level = a->saturation_level;
    out->erl_estimate = a->erl_estimate;
    out->shadow_dt = a->last_shadow_dt;
    out->is_stationary_dt = a->last_is_stationary_dt;
    out->filter_converged = a->convergence.converged;
    out->filter_once_converged = a->convergence.once_converged;
    out->epc_active = a->epc.active;
    /* Per-mode semantics: see AecLinearDelayState in aec.h. */
    out->delay_confidence = aec_delay_confidence(a);
    if (a->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED) {
        out->delay_samples = 0;
        out->delay_state = AEC_LINEAR_DELAY_LOCKED;
        return AEC_OK;
    }
    if (a->cfg.delay_mode == AEC_DELAY_FIXED) {
        out->delay_samples = a->current_delay;
        /* far_hop is raw until the ring can serve the offset, so report
         * UNLOCKED then; CHANGED cannot occur. */
        out->delay_state = a->far_hop_aligned ? AEC_LINEAR_DELAY_LOCKED
                                              : AEC_LINEAR_DELAY_UNLOCKED;
        return AEC_OK;
    }
    out->delay_samples = a->current_delay;
    if (a->current_delay < 0 || !a->far_hop_aligned)
        out->delay_state = AEC_LINEAR_DELAY_UNLOCKED;
    else if (a->delay_generation != a->delay_gen_hop_start)
        out->delay_state = AEC_LINEAR_DELAY_CHANGED;
    else
        out->delay_state = AEC_LINEAR_DELAY_LOCKED;
    return AEC_OK;
}

/* ---- test-only hooks (test/aec_test_hooks.h; compiled only under -DAEC_TESTING): the entry points with a caller-built tuning -- */
#ifdef AEC_TESTING
void aec_testing_tuning_defaults(AecTuning* tun, const AecConfig* cfg) {
    aec_tuning_defaults(tun, cfg);
}

int aec_testing_tuning_valid(const AecTuning* tun, const AecConfig* cfg) {
    return aec_tuning_valid(tun, cfg);
}

size_t aec_testing_get_mem_size(const AecConfig* cfg, const AecTuning* tun) {
    AecResolved rs;
    return aec_resolve_tuned(cfg, tun, &rs) ? aec_mem_size_resolved(&rs) : 0;
}

int aec_testing_get_mem_requirements(const AecConfig* cfg, const AecTuning* tun, AecMemReq* out) {
    AecResolved rs;
    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!cfg || !aec_resolve_tuned(cfg, tun, &rs)) return -1;
    return aec_mem_requirements_resolved(&rs, out);
}

Aec* aec_testing_init(void* mem, size_t mem_size, const AecConfig* cfg, const AecTuning* tun) {
    AecResolved rs;
    return aec_resolve_tuned(cfg, tun, &rs) ? aec_init_resolved(mem, mem_size, &rs) : NULL;
}

Aec* aec_testing_create(const AecConfig* cfg, const AecTuning* tun) {
    AecResolved rs;
    return aec_resolve_tuned(cfg, tun, &rs) ? aec_create_resolved(&rs) : NULL;
}
#endif
