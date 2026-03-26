/**
 * aec.c - AEC Coordinator
 *
 * Orchestrates: HPF → PBFDKF (main + shadow) → simple variable mu →
 *               convergence detection → RES (WOLA) → output limiter
 *
 * Matches Python AEC v1.21.0 (PBFDKF mode, Kalman always on).
 */

#include "aec.h"
#include "pbfdkf.h"
#include "res_filter.h"
#include "hpf.h"
#include "fast_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

struct Aec {
    AecConfig config;
    int is_static;      /* 1 = placed in external memory, skip free */

    /* HPF (mic + ref) */
    Hpf* hp_mic;
    Hpf* hp_ref;

    /* Main adaptive filter */
    Pbfdkf* filter;

    /* Shadow filter */
    Pbfdkf* shadow_filter;
    float main_err_smooth;
    float shadow_err_smooth;
    int shadow_frame_count;
    int shadow_copy_counter;

    /* RES post-filter */
    ResFilter* res;

    /* Simple variable mu (Valin 2007 RER-inspired) */
    float simple_mu_ratio;
    int simple_mu_holdoff;
    int warmup_frames;

    /* Convergence */
    int filter_converged;
    int conv_counter;

    /* ERLE tracking */
    float near_power;
    float raw_error_power;
    float final_error_power;
    float near_power_sum;
    float raw_error_power_sum;
    float final_error_power_sum;
    float alpha_erle;

    /* Output limiter */
    float limiter_gain;

    /* Temp buffers */
    float* mic_buf;     /* [hop_size] HPF scratch */
    float* ref_buf;     /* [hop_size] HPF scratch */
    float* raw_output;  /* [hop_size] */

    /* Per-bin mu_scale from RES */
    float* per_bin_mu_scale; /* [n_freqs] or NULL */
    int n_freqs;
};

Aec* aec_create(const AecConfig* config) {
    if (!config) return NULL;

    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec) return NULL;

    aec->config = *config;
    int hop = config->hop_size;
    int nf = config->n_freqs;
    aec->n_freqs = nf;

    /* HPF */
    if (config->enable_highpass) {
        aec->hp_mic = hpf_create(config->highpass_cutoff_hz, config->sample_rate);
        aec->hp_ref = hpf_create(config->highpass_cutoff_hz, config->sample_rate);
    }

    /* Main filter (PBFDKF) */
    aec->filter = pbfdkf_create(config->fft_size, hop,
                                 config->n_partitions, config->delta,
                                 config->kalman_q_high, config->kalman_q_low);
    if (!aec->filter) { aec_destroy(aec); return NULL; }

    /* Shadow filter */
    aec->shadow_filter = pbfdkf_create(config->fft_size, hop,
                                        config->n_partitions, config->delta,
                                        config->kalman_q_high, config->kalman_q_low);
    if (!aec->shadow_filter) { aec_destroy(aec); return NULL; }

    /* Set shadow Q = main Q × ratio */
    pbfdkf_set_q_ratio(aec->shadow_filter, config->kalman_q_high,
                        config->kalman_q_low, config->shadow_q_ratio);

    /* RES */
    if (config->enable_res) {
        ResConfig rc;
        rc.block_size = config->fft_size;
        rc.frame_size = config->frame_size;
        rc.hop_size = hop;
        rc.n_freqs = nf;
        rc.g_min_db = config->res_g_min_db;
        rc.max_drop_db_per_frame = config->res_max_drop_db_per_frame;
        rc.max_rise_db_per_frame = config->res_max_rise_db_per_frame;
        rc.spectral_floor_db = config->res_spectral_floor_db;
        rc.ne_protect_db = config->res_ne_protect_db;
        rc.enable_cng = config->enable_cng;
        rc.alpha_echo_psd = config->res_alpha_echo_psd;
        rc.alpha_error_psd = config->res_alpha_error_psd;
        rc.echo_method = config->res_echo_method;
        rc.gain_type = config->res_gain_type;
        rc.enable_reverb = config->res_enable_reverb;
        rc.reverb_decay = config->res_reverb_decay;
        rc.reverb_gain = config->res_reverb_gain;
        rc.enr_scale = config->res_enr_scale;
        aec->res = res_create(&rc);
    }

    /* Variable mu */
    aec->simple_mu_ratio = 1.0f;
    aec->simple_mu_holdoff = 0;
    aec->warmup_frames = config->warmup_frames;

    /* Convergence */
    aec->filter_converged = 0;
    aec->conv_counter = 0;

    /* ERLE */
    aec->alpha_erle = 0.95f;

    /* Limiter */
    aec->limiter_gain = 1.0f;

    /* Buffers */
    aec->mic_buf    = (float*)malloc(hop * sizeof(float));
    aec->ref_buf    = (float*)malloc(hop * sizeof(float));
    aec->raw_output = (float*)malloc(hop * sizeof(float));
    aec->per_bin_mu_scale = NULL;

    if (!aec->mic_buf || !aec->ref_buf || !aec->raw_output) {
        aec_destroy(aec); return NULL;
    }

    /* Shadow init */
    aec->main_err_smooth = 0.0f;
    aec->shadow_err_smooth = 0.0f;
    aec->shadow_frame_count = 0;
    aec->shadow_copy_counter = 0;

    return aec;
}

/* --- Static memory API --- */

size_t aec_get_mem_size(const AecConfig* config) {
    if (!config) return 0;
    int hop = config->hop_size;
    size_t total = 0;

    total += ALIGN16(sizeof(Aec));

    /* HPF (2 instances, conditional) */
    if (config->enable_highpass) {
        total += hpf_get_mem_size() * 2;
    }

    /* Main + shadow PBFDKF */
    size_t pbf_size = pbfdkf_get_mem_size(config->fft_size, hop, config->n_partitions);
    total += pbf_size * 2;

    /* RES (conditional) */
    if (config->enable_res) {
        ResConfig rc = res_config_from_aec(config);
        total += res_get_mem_size(&rc);
    }

    /* Temp buffers */
    total += ALIGN16(hop * sizeof(float)) * 3;  /* mic_buf, ref_buf, raw_output */

    return total;
}

Aec* aec_init(void* mem, size_t mem_size, const AecConfig* config) {
    if (!mem || !config) return NULL;
    if (mem_size < aec_get_mem_size(config)) return NULL;

    int hop = config->hop_size;
    int nf = config->n_freqs;
    uint8_t* ptr = (uint8_t*)mem;

    Aec* aec = (Aec*)ptr;
    ptr += ALIGN16(sizeof(Aec));
    memset(aec, 0, sizeof(Aec));

    aec->config = *config;
    aec->is_static = 1;
    aec->n_freqs = nf;

    /* HPF */
    if (config->enable_highpass) {
        size_t hpf_sz = hpf_get_mem_size();
        aec->hp_mic = hpf_init(ptr, hpf_sz, config->highpass_cutoff_hz, config->sample_rate);
        ptr += hpf_sz;
        aec->hp_ref = hpf_init(ptr, hpf_sz, config->highpass_cutoff_hz, config->sample_rate);
        ptr += hpf_sz;
        if (!aec->hp_mic || !aec->hp_ref) return NULL;
    }

    /* Main PBFDKF */
    size_t pbf_size = pbfdkf_get_mem_size(config->fft_size, hop, config->n_partitions);
    aec->filter = pbfdkf_init(ptr, pbf_size, config->fft_size, hop,
                               config->n_partitions, config->delta,
                               config->kalman_q_high, config->kalman_q_low);
    ptr += pbf_size;
    if (!aec->filter) return NULL;

    /* Shadow PBFDKF */
    aec->shadow_filter = pbfdkf_init(ptr, pbf_size, config->fft_size, hop,
                                      config->n_partitions, config->delta,
                                      config->kalman_q_high, config->kalman_q_low);
    ptr += pbf_size;
    if (!aec->shadow_filter) return NULL;
    pbfdkf_set_q_ratio(aec->shadow_filter, config->kalman_q_high,
                        config->kalman_q_low, config->shadow_q_ratio);

    /* RES */
    if (config->enable_res) {
        ResConfig rc = res_config_from_aec(config);
        size_t res_sz = res_get_mem_size(&rc);
        aec->res = res_init(ptr, res_sz, &rc);
        ptr += res_sz;
        if (!aec->res) return NULL;
    }

    /* Temp buffers */
    aec->mic_buf    = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));
    aec->ref_buf    = (float*)ptr; ptr += ALIGN16(hop * sizeof(float));
    aec->raw_output = (float*)ptr;
    memset(aec->mic_buf, 0, hop * sizeof(float));
    memset(aec->ref_buf, 0, hop * sizeof(float));
    memset(aec->raw_output, 0, hop * sizeof(float));

    /* Init state */
    aec->simple_mu_ratio = 1.0f;
    aec->warmup_frames = config->warmup_frames;
    aec->alpha_erle = 0.95f;
    aec->limiter_gain = 1.0f;
    aec->per_bin_mu_scale = NULL;

    return aec;
}

void aec_destroy(Aec* aec) {
    if (!aec) return;
    if (aec->is_static) return;

    hpf_destroy(aec->hp_mic);
    hpf_destroy(aec->hp_ref);
    pbfdkf_destroy(aec->filter);
    pbfdkf_destroy(aec->shadow_filter);
    res_destroy(aec->res);
    free(aec->mic_buf);
    free(aec->ref_buf);
    free(aec->raw_output);
    free(aec->per_bin_mu_scale);
    free(aec);
}

void aec_reset(Aec* aec) {
    if (!aec) return;
    if (aec->hp_mic) { hpf_reset(aec->hp_mic); hpf_reset(aec->hp_ref); }
    pbfdkf_reset(aec->filter);
    pbfdkf_reset(aec->shadow_filter);
    pbfdkf_set_q_ratio(aec->shadow_filter, aec->config.kalman_q_high,
                        aec->config.kalman_q_low, aec->config.shadow_q_ratio);
    if (aec->res) res_reset(aec->res);
    aec->simple_mu_ratio = 1.0f;
    aec->simple_mu_holdoff = 0;
    aec->warmup_frames = aec->config.warmup_frames;
    aec->filter_converged = 0;
    aec->conv_counter = 0;
    aec->near_power = 0; aec->raw_error_power = 0; aec->final_error_power = 0;
    aec->near_power_sum = 0; aec->raw_error_power_sum = 0; aec->final_error_power_sum = 0;
    aec->limiter_gain = 1.0f;
    aec->main_err_smooth = 0; aec->shadow_err_smooth = 0;
    aec->shadow_frame_count = 0; aec->shadow_copy_counter = 0;
    free(aec->per_bin_mu_scale); aec->per_bin_mu_scale = NULL;
}

/* --- Simple variable mu (Valin 2007 RER-inspired) --- */
static float get_simple_mu_scale(Aec* aec) {
    float mu_min = aec->config.shadow_mu_min;

    /* Warmup: high mu for fast initial convergence */
    if (aec->warmup_frames > 0) {
        aec->warmup_frames--;
        return minf(1.0f, maxf(0.7f, aec->simple_mu_ratio + 0.3f));
    }

    if (!aec->filter_converged)
        mu_min = maxf(mu_min, 0.5f);
    else
        mu_min = maxf(mu_min, 0.2f);

    /* Per-bin mu_scale from RES (post-convergence) */
    /* (returned as scalar fallback when per_bin not available) */
    return mu_min + (1.0f - mu_min) * aec->simple_mu_ratio;
}

static void update_simple_mu_ratio(Aec* aec, const float* output,
                                    const float* far, int hop) {
    float err_pwr = 0.0f, far_pwr = 0.0f;
    for (int i = 0; i < hop; i++) {
        err_pwr += output[i] * output[i];
        far_pwr += far[i] * far[i];
    }
    err_pwr = err_pwr / hop + 1e-10f;
    far_pwr = far_pwr / hop + 1e-10f;
    float ratio = far_pwr / err_pwr;
    if (ratio > 1.0f) ratio = 1.0f;

    /* Echo estimate ratio boost */
    const Complex* echo_spec = pbfdkf_get_echo_spec(aec->filter);
    if (echo_spec) {
        int nf = aec->n_freqs;
        float echo_pwr = 0.0f;
        for (int k = 0; k < nf; k++)
            echo_pwr += echo_spec[k].r * echo_spec[k].r +
                        echo_spec[k].i * echo_spec[k].i;
        echo_pwr += 1e-10f;
        /* Use near_power_sum as approximation */
        float near_pwr = aec->near_power + 1e-10f;
        if (near_pwr > 1e-8f) {
            float ratio_echo = clampf(echo_pwr / (near_pwr * nf), 0.0f, 1.0f);
            ratio = maxf(ratio, ratio_echo * 0.5f);
        }
    }

    /* Asymmetric EMA + holdoff */
    float alpha;
    if (ratio < aec->simple_mu_ratio) {
        alpha = 0.3f;
        aec->simple_mu_holdoff = 20;
    } else if (aec->simple_mu_holdoff > 0) {
        aec->simple_mu_holdoff--;
        alpha = 0.99f;
    } else {
        alpha = 0.95f;
    }
    aec->simple_mu_ratio = alpha * aec->simple_mu_ratio + (1.0f - alpha) * ratio;
}

/* --- Context allocation / destruction --- */

AecResContext* aec_context_create(const Aec* aec) {
    if (!aec) return NULL;
    int nf = aec->n_freqs;

    AecResContext* ctx = (AecResContext*)calloc(1, sizeof(AecResContext));
    if (!ctx) return NULL;
    ctx->n_freqs = nf;

    ctx->echo_spec_re = (float*)calloc(nf, sizeof(float));
    ctx->echo_spec_im = (float*)calloc(nf, sizeof(float));
    ctx->far_spec_re  = (float*)calloc(nf, sizeof(float));
    ctx->far_spec_im  = (float*)calloc(nf, sizeof(float));
    ctx->near_spec_re = (float*)calloc(nf, sizeof(float));
    ctx->near_spec_im = (float*)calloc(nf, sizeof(float));

    if (!ctx->echo_spec_re || !ctx->echo_spec_im ||
        !ctx->far_spec_re  || !ctx->far_spec_im  ||
        !ctx->near_spec_re || !ctx->near_spec_im) {
        aec_context_destroy(ctx);
        return NULL;
    }
    return ctx;
}

/* --- Context static memory API --- */

size_t aec_context_get_mem_size(int n_freqs) {
    size_t total = 0;
    total += ALIGN16(sizeof(AecResContext));
    total += ALIGN16(n_freqs * sizeof(float)) * 6;  /* 6 spectrum arrays */
    return total;
}

AecResContext* aec_context_init(void* mem, size_t mem_size, int n_freqs) {
    if (!mem || n_freqs <= 0) return NULL;
    if (mem_size < aec_context_get_mem_size(n_freqs)) return NULL;

    uint8_t* ptr = (uint8_t*)mem;
    AecResContext* ctx = (AecResContext*)ptr;
    ptr += ALIGN16(sizeof(AecResContext));
    memset(ctx, 0, sizeof(AecResContext));

    ctx->n_freqs = n_freqs;
    ctx->is_static = 1;

    ctx->echo_spec_re = (float*)ptr; ptr += ALIGN16(n_freqs * sizeof(float));
    ctx->echo_spec_im = (float*)ptr; ptr += ALIGN16(n_freqs * sizeof(float));
    ctx->far_spec_re  = (float*)ptr; ptr += ALIGN16(n_freqs * sizeof(float));
    ctx->far_spec_im  = (float*)ptr; ptr += ALIGN16(n_freqs * sizeof(float));
    ctx->near_spec_re = (float*)ptr; ptr += ALIGN16(n_freqs * sizeof(float));
    ctx->near_spec_im = (float*)ptr;

    memset(ctx->echo_spec_re, 0, n_freqs * sizeof(float));
    memset(ctx->echo_spec_im, 0, n_freqs * sizeof(float));
    memset(ctx->far_spec_re,  0, n_freqs * sizeof(float));
    memset(ctx->far_spec_im,  0, n_freqs * sizeof(float));
    memset(ctx->near_spec_re, 0, n_freqs * sizeof(float));
    memset(ctx->near_spec_im, 0, n_freqs * sizeof(float));

    return ctx;
}

void aec_context_destroy(AecResContext* ctx) {
    if (!ctx) return;
    if (ctx->is_static) return;

    free(ctx->echo_spec_re);
    free(ctx->echo_spec_im);
    free(ctx->far_spec_re);
    free(ctx->far_spec_im);
    free(ctx->near_spec_re);
    free(ctx->near_spec_im);
    free(ctx);
}

/* Fill context from current PBFDKF state */
static void fill_context(Aec* aec, const float* mic, const float* ref,
                          int hop, AecResContext* ctx) {
    const AecConfig* cfg = &aec->config;
    int nf = aec->n_freqs;

    /* Copy spectra from PBFDKF (Complex → split re/im) */
    const Complex* echo = pbfdkf_get_echo_spec(aec->filter);
    const Complex* far  = pbfdkf_get_far_spec(aec->filter);
    const Complex* near = pbfdkf_get_near_spec(aec->filter);
    for (int k = 0; k < nf; k++) {
        ctx->echo_spec_re[k] = echo[k].r;
        ctx->echo_spec_im[k] = echo[k].i;
        ctx->far_spec_re[k]  = far[k].r;
        ctx->far_spec_im[k]  = far[k].i;
        ctx->near_spec_re[k] = near[k].r;
        ctx->near_spec_im[k] = near[k].i;
    }

    /* Far-end power */
    float far_power = 0.0f;
    for (int i = 0; i < hop; i++) far_power += ref[i] * ref[i];
    far_power /= hop;
    ctx->far_power = far_power;

    ctx->filter_converged = aec->filter_converged;

    /* ERLE factor */
    float inst_erle = aec_get_erle_instant(aec);
    ctx->erle_factor = clampf((inst_erle - 3.0f) / 15.0f, 0.0f, 1.0f);

    /* DT indicator */
    float mic_pwr = 0.0f;
    for (int i = 0; i < hop; i++) mic_pwr += mic[i] * mic[i];
    mic_pwr = mic_pwr / hop + 1e-10f;
    float far_p = far_power + 1e-10f;
    ctx->dt_indicator = clampf(1.0f - far_p / (mic_pwr + far_p), 0.0f, 0.8f);

    /* Dynamic over_sub */
    float base_over_sub = cfg->res_over_sub_base +
                           cfg->res_over_sub_scale * ctx->erle_factor;
    float dt_reduction = cfg->res_dt_reduction * ctx->dt_indicator;
    ctx->over_sub = maxf(base_over_sub - dt_reduction, 0.5f);

    /* Divergence (simple: error/near ratio) */
    float err_pwr = aec->raw_error_power + 1e-10f;
    float near_pwr = aec->near_power + 1e-10f;
    ctx->divergence = clampf(err_pwr / near_pwr - 0.5f, 0.0f, 1.0f);
}

int aec_process_ex(Aec* aec,
                   const float* near_end,
                   const float* far_end,
                   float* output,
                   AecResContext* ctx) {
    if (!aec || !near_end || !far_end || !output) return -1;

    const int hop = aec->config.hop_size;
    const float alpha = aec->alpha_erle;
    const AecConfig* cfg = &aec->config;

    /* Copy input (we modify via HPF) */
    memcpy(aec->mic_buf, near_end, hop * sizeof(float));
    memcpy(aec->ref_buf, far_end, hop * sizeof(float));

    /* HPF */
    if (aec->hp_mic) {
        hpf_process(aec->hp_mic, aec->mic_buf, hop);
        hpf_process(aec->hp_ref, aec->ref_buf, hop);
    }

    const float* mic = aec->mic_buf;
    const float* ref = aec->ref_buf;

    /* Compute mu_scale */
    float mu_scale = get_simple_mu_scale(aec);

    /* === Main filter === */
    pbfdkf_process(aec->filter, mic, ref, aec->raw_output, mu_scale);

    /* === Shadow filter (always mu_scale=1.0) === */
    float shadow_out[160]; /* stack alloc, hop <= 160 */
    aec->shadow_frame_count++;
    pbfdkf_process(aec->shadow_filter, mic, ref, shadow_out, 1.0f);

    /* Smoothed error tracking */
    float main_err = pbfdkf_get_error_energy(aec->filter);
    float shadow_err = pbfdkf_get_error_energy(aec->shadow_filter);
    float alpha_s = cfg->shadow_err_alpha;
    aec->main_err_smooth   = alpha_s * aec->main_err_smooth   + (1.0f - alpha_s) * main_err;
    aec->shadow_err_smooth = alpha_s * aec->shadow_err_smooth + (1.0f - alpha_s) * shadow_err;

    /* Copy gate (after warmup) */
    if (aec->shadow_frame_count >= 50) {
        float threshold = cfg->shadow_copy_threshold;

        /* Far-end activity check */
        float far_pwr = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr += ref[i] * ref[i];
        int far_active = (far_pwr / hop > 1e-4f);

        /* DT check: far/error ratio */
        float raw_err_pwr = 0.0f;
        for (int i = 0; i < hop; i++)
            raw_err_pwr += aec->raw_output[i] * aec->raw_output[i];
        raw_err_pwr = raw_err_pwr / hop + 1e-10f;
        float far_pwr_avg = far_pwr / hop + 1e-10f;
        int not_dt = (far_pwr_avg / raw_err_pwr > 0.3f);

        int copy_allowed = far_active && not_dt;

        if (copy_allowed) {
            if (aec->shadow_err_smooth < aec->main_err_smooth * threshold) {
                aec->shadow_copy_counter++;
            } else {
                aec->shadow_copy_counter = 0;
            }

            if (aec->shadow_copy_counter >= cfg->shadow_copy_hysteresis) {
                /* Shadow → Main copy */
                pbfdkf_copy_weights(aec->filter, aec->shadow_filter);
                aec->main_err_smooth = aec->shadow_err_smooth;
                aec->shadow_copy_counter = 0;
            } else if (aec->main_err_smooth < aec->shadow_err_smooth * threshold && not_dt) {
                /* Bidirectional: Main → Shadow */
                pbfdkf_copy_weights(aec->shadow_filter, aec->filter);
                aec->shadow_err_smooth = aec->main_err_smooth;
            }
        } else {
            aec->shadow_copy_counter = 0;
        }
    }

    /* Copy raw output for ERLE + RES input */
    memcpy(output, aec->raw_output, hop * sizeof(float));

    /* === RES post-filter === */
    if (aec->res) {
        float far_power = 0.0f;
        for (int i = 0; i < hop; i++) far_power += ref[i] * ref[i];
        far_power /= hop;

        /* Dynamic over_sub */
        float inst_erle = aec_get_erle_instant(aec);
        float erle_factor = clampf((inst_erle - 3.0f) / 15.0f, 0.0f, 1.0f);
        float base_over_sub = cfg->res_over_sub_base +
                               cfg->res_over_sub_scale * erle_factor;

        /* DT indicator */
        float mic_pwr = 0.0f;
        for (int i = 0; i < hop; i++) mic_pwr += mic[i] * mic[i];
        mic_pwr = mic_pwr / hop + 1e-10f;
        float far_p = far_power + 1e-10f;
        float dt_indicator = clampf(1.0f - far_p / (mic_pwr + far_p), 0.0f, 0.8f);

        float dt_reduction = cfg->res_dt_reduction * dt_indicator;
        float over_sub = maxf(base_over_sub - dt_reduction, 0.5f);

        res_process(aec->res, aec->raw_output,
                    pbfdkf_get_echo_spec(aec->filter),
                    pbfdkf_get_far_spec(aec->filter),
                    pbfdkf_get_near_spec(aec->filter),
                    far_power, aec->filter_converged,
                    erle_factor, dt_indicator, over_sub,
                    output);

        /* Update per-bin mu_scale from RES PSDs */
        if (aec->filter_converged) {
            const float* echo_psd = res_get_echo_psd(aec->res);
            const float* error_psd = res_get_error_psd(aec->res);
            if (echo_psd && error_psd) {
                float mean_eer = 0.0f;
                int nf = aec->n_freqs;
                for (int k = 0; k < nf; k++) {
                    float eer_k = clampf(echo_psd[k] / (error_psd[k] + 1e-10f), 0.0f, 1.0f);
                    mean_eer += eer_k;
                }
                aec->simple_mu_ratio = mean_eer / nf;
            }
        } else {
            update_simple_mu_ratio(aec, aec->raw_output, ref, hop);
        }
    } else {
        update_simple_mu_ratio(aec, aec->raw_output, ref, hop);
    }

    /* === Fill external context (if requested) === */
    if (ctx) {
        fill_context(aec, mic, ref, hop, ctx);
    }

    /* === Output limiter (smoothed gain) === */
    {
        float near_peak = 0.0f, out_peak = 0.0f;
        for (int i = 0; i < hop; i++) {
            float an = fabsf(mic[i]);
            float ao = fabsf(output[i]);
            if (an > near_peak) near_peak = an;
            if (ao > out_peak) out_peak = ao;
        }
        float target_gain;
        if (out_peak > near_peak && near_peak > 1e-6f)
            target_gain = near_peak / out_peak;
        else
            target_gain = 1.0f;

        float alpha_lim = (target_gain < aec->limiter_gain) ? 0.3f : 0.8f;
        aec->limiter_gain = alpha_lim * aec->limiter_gain +
                             (1.0f - alpha_lim) * target_gain;

        for (int i = 0; i < hop; i++)
            output[i] *= aec->limiter_gain;
    }

    /* === Noise gate === */
    {
        float out_power = 0.0f, near_power_inst = 0.0f, far_power_inst = 0.0f;
        for (int i = 0; i < hop; i++) {
            out_power += output[i] * output[i];
            near_power_inst += mic[i] * mic[i];
            far_power_inst += ref[i] * ref[i];
        }
        out_power /= hop;
        near_power_inst /= hop;
        far_power_inst /= hop;

        if (far_power_inst > 1e-4f && near_power_inst > 1e-8f) {
            float snr = out_power / near_power_inst;
            if (snr < 0.01f) {
                float scale = snr / 0.01f;
                for (int i = 0; i < hop; i++) output[i] *= scale;
            }
        }
    }

    /* === ERLE tracking === */
    for (int i = 0; i < hop; i++) {
        aec->near_power      = alpha * aec->near_power      + (1.0f - alpha) * mic[i] * mic[i];
        aec->raw_error_power = alpha * aec->raw_error_power  + (1.0f - alpha) * aec->raw_output[i] * aec->raw_output[i];
        aec->final_error_power = alpha * aec->final_error_power + (1.0f - alpha) * output[i] * output[i];
    }
    for (int i = 0; i < hop; i++) {
        aec->near_power_sum += mic[i] * mic[i];
        aec->raw_error_power_sum += aec->raw_output[i] * aec->raw_output[i];
        aec->final_error_power_sum += output[i] * output[i];
    }

    /* === Convergence detection: 10 consecutive frames ERLE > 6dB === */
    if (!aec->filter_converged && aec->near_power > 1e-8f) {
        float inst_erle = 10.0f * fast_log10(aec->near_power /
                                          (aec->raw_error_power + 1e-10f));
        if (inst_erle > 6.0f)
            aec->conv_counter++;
        else
            aec->conv_counter = 0;

        if (aec->conv_counter >= 10) {
            aec->filter_converged = 1;
            pbfdkf_switch_q_low(aec->filter);
            pbfdkf_switch_q_low(aec->shadow_filter);
        }
    }

    return 0;
}

int aec_process(Aec* aec,
                const float* near_end,
                const float* far_end,
                float* output) {
    return aec_process_ex(aec, near_end, far_end, output, NULL);
}

int aec_get_hop_size(const Aec* aec) {
    return aec ? aec->config.hop_size : 0;
}

float aec_get_erle(const Aec* aec) {
    if (!aec) return 0.0f;
    const float eps = 1e-10f;
    if (aec->near_power_sum < eps && aec->raw_error_power_sum < eps)
        return 0.0f;
    return 10.0f * fast_log10((aec->near_power_sum + eps) /
                           (aec->raw_error_power_sum + eps));
}

float aec_get_erle_instant(const Aec* aec) {
    if (!aec) return 0.0f;
    const float eps = 1e-10f;
    if (aec->near_power < eps && aec->raw_error_power < eps)
        return 0.0f;
    return 10.0f * fast_log10((aec->near_power + eps) /
                           (aec->raw_error_power + eps));
}

int aec_is_converged(const Aec* aec) {
    return aec ? aec->filter_converged : 0;
}

const AecConfig* aec_get_config(const Aec* aec) {
    return aec ? &aec->config : NULL;
}
