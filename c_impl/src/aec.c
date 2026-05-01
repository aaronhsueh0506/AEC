/**
 * aec.c - AEC Coordinator
 *
 * Orchestrates: PBFDKF (main + shadow) → simple variable mu →
 *               convergence detection → RES (WOLA) → output limiter
 *
 * Matches Python AEC v1.28.1 (PBFDKF mode, Kalman always on).
 */

#include "aec.h"
#include "pbfdkf.h"
#include "res_filter.h"
#include "fft_wrapper.h"
#include "fast_math.h"
#include "dtd_estimator.h"
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

    /* Main adaptive filter */
    Pbfdkf* filter;

    /* Shadow filter */
    Pbfdkf* shadow_filter;
    float main_err_smooth;
    float shadow_err_smooth;
    int shadow_frame_count;
    int shadow_copy_counter;
    float copy_err_baseline;      /* EMA of best-filter err (v2.3.0 copy gate) */

    /* RES post-filter */
    ResFilter* res;

    /* Simple variable mu (Valin 2007 RER-inspired) */
    float simple_mu_ratio;
    int simple_mu_holdoff;
    int warmup_frames;
    int warmup_far_active;

    /* v3.x DTD-based mu_scale (Python aec.py _compute_mu_scale) */
    DtdEstimator* dtd_coherence;
    DtdEstimator* dtd_divergence;
    float prev_dtd_conf;

    /* Echo Path Change detection */
    float prev_total_err;
    int epc_active;
    int epc_hangover_count;

    /* Stationary far-end DT detection (matches Python v13) */
    float far_env_mean;            /* EMA α=0.99, TC≈1s */
    float far_env_var;
    int   far_active_prev;
    int   is_stationary_far;       /* CV² < 0.02 */
    float wn_err_baseline;         /* voice-band error baseline */
    int   stat_dt_hangover;        /* 800ms (80 frames @10ms) */
    int   is_stationary_dt;

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

    /* dt_indicator ERLE correction */
    float inst_erle_smooth;
    float divergence_indicator;

    /* Output limiter */
    float limiter_gain;

    /* Temp buffers */
    float* mic_buf;     /* [hop_size] working copy */
    float* ref_buf;     /* [hop_size] working copy */
    float* raw_output;  /* [hop_size] */

    /* Per-bin mu_scale from RES */
    float* per_bin_mu_scale; /* [n_freqs] or NULL */
    int n_freqs;

    /* Delay estimation (GCC-PHAT) */
    struct {
        int enabled;
        int seg_size;         /* FFT size for cross-correlation (power of 2) */
        int seg_hop;          /* seg_size / 2 */
        int n_freqs;          /* seg_size / 2 + 1 */
        int max_delay;        /* samples */

        float* mic_buf;       /* [seg_size] accumulation buffer */
        float* ref_buf;       /* [seg_size] accumulation buffer */
        int buf_pos;

        float* cross_re;      /* [n_freqs] smoothed cross-spectrum real */
        float* cross_im;      /* [n_freqs] smoothed cross-spectrum imag */
        int n_updates;
        float alpha;          /* 0.6 EMA smoothing */

        int estimated_delay;  /* -1 = not yet estimated */
        int current_delay;    /* currently applied delay */
        int pending_delay;    /* pending delay for two-consecutive-estimate check */
        int has_pending;      /* whether pending_delay is valid */
        int samples_accumulated;
        int samples_since_est;
        int init_samples;
        int period_samples;
        int init_done;

        FftHandle* fft;       /* dedicated FFT handle (may differ from PBFDKF size) */
        Complex* mic_spec;    /* [n_freqs] scratch */
        Complex* ref_spec;    /* [n_freqs] scratch */
        Complex* phat;        /* [n_freqs] scratch for PHAT normalization */
        float* gcc;           /* [seg_size] scratch for IRFFT output */
    } delay_est;

    /* Reference delay ring buffer */
    float* ref_delay_buf;     /* circular buffer [ref_delay_buf_size] */
    int ref_delay_buf_size;
    int ref_delay_write_pos;
    int ref_delay_filled;     /* total samples written (for warmup) */

    /* v2.3.x: Energy DT signal */
    float dt_from_energy;     /* pre-filter energy-based DT [0, 1] */
    float dt_from_shadow;     /* shadow DTD signal [0, 1] */
    float shadow_advantage;   /* main_err / shadow_err */
    int   shadow_advantage_streak; /* G3: consecutive advantage frames */

    /* v2.3.x: Dynamic ERL (B4) */
    float erl_estimate;       /* tracked ERL for render-based echo */

    /* v2.3.x: EPC render-forced (G1) */
    int epc_render_forced_remaining;

    /* v2.3.x: erle_factor_prev for diagnostics */
    float erle_factor_prev;

    /* v2.3.x: ERLE windowed (v2.1.0) */
    float erle_window_near;
    float erle_window_err;

    /* v2.3.x: saturation level */
    float saturation_level;

    /* v2.3.x: GCC-PHAT PAR (A5) */
    float delay_last_par;
};

Aec* aec_create(const AecConfig* config) {
    if (!config) return NULL;

    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec) return NULL;

    aec->config = *config;
    int hop = config->hop_size;
    int nf = config->n_freqs;
    aec->n_freqs = nf;

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

    /* v3.x DTD: coherence (primary) + divergence (fallback).
     * Match Python AecConfig.enable_dtd default = False — only allocate when
     * config.enable_dtd is explicitly set. BALANCED preset doesn't enable
     * DTD, uses non-DTD path (raw_dt = max(dt_from_energy, simple_dt × 0.5)). */
    aec->dtd_coherence = NULL;
    aec->dtd_divergence = NULL;
    aec->prev_dtd_conf = 0.0f;
    /* AecConfig has no enable_dtd in C struct; default leave DTD off to
     * match Python BALANCED. Future: add config.enable_dtd field. */

    /* Convergence */
    aec->filter_converged = 0;
    aec->conv_counter = 0;

    /* ERLE */
    aec->alpha_erle = 0.95f;

    /* Limiter */
    aec->limiter_gain = 1.0f;
    aec->inst_erle_smooth = 1.0f;
    aec->divergence_indicator = 0.0f;

    /* v2.3.x state */
    aec->dt_from_energy = 0.0f;
    aec->dt_from_shadow = 0.0f;
    aec->shadow_advantage = 1.0f;
    aec->shadow_advantage_streak = 0;
    aec->erl_estimate = 0.1f;
    aec->epc_render_forced_remaining = 0;
    aec->erle_factor_prev = 0.0f;
    aec->erle_window_near = 1e-10f;
    aec->erle_window_err = 1e-10f;
    aec->saturation_level = 0.0f;
    aec->delay_last_par = 0.0f;

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
    aec->copy_err_baseline = 0.0f;

    /* EPC init */
    aec->prev_total_err = 0.0f;
    aec->epc_active = 0;
    aec->epc_hangover_count = 0;

    /* Stationary far-end DT init */
    aec->far_env_mean = 0.0f;
    aec->far_env_var = 0.0f;
    aec->far_active_prev = 0;
    aec->is_stationary_far = 0;
    aec->wn_err_baseline = 1e-8f;
    aec->stat_dt_hangover = 0;
    aec->is_stationary_dt = 0;

    /* Delay estimation */
    aec->delay_est.enabled = 0;
    aec->delay_est.estimated_delay = -1;
    aec->delay_est.current_delay = -1;
    aec->delay_est.fft = NULL;
    aec->delay_est.mic_buf = NULL;
    aec->delay_est.ref_buf = NULL;
    aec->delay_est.cross_re = NULL;
    aec->delay_est.cross_im = NULL;
    aec->delay_est.mic_spec = NULL;
    aec->delay_est.ref_spec = NULL;
    aec->delay_est.phat = NULL;
    aec->delay_est.gcc = NULL;
    aec->ref_delay_buf = NULL;

    if (config->enable_delay_est) {
        int max_delay = (int)(config->delay_est_max_ms * config->sample_rate / 1000.0f);
        int min_seg = 2 * max_delay;
        if (min_seg < 2048) min_seg = 2048;
        int seg_size = 1;
        while (seg_size < min_seg) seg_size *= 2;
        int de_nfreqs = seg_size / 2 + 1;

        aec->delay_est.enabled = 1;
        aec->delay_est.seg_size = seg_size;
        aec->delay_est.seg_hop = seg_size / 2;
        aec->delay_est.n_freqs = de_nfreqs;
        aec->delay_est.max_delay = max_delay;
        aec->delay_est.buf_pos = 0;
        aec->delay_est.n_updates = 0;
        aec->delay_est.alpha = 0.6f;
        aec->delay_est.estimated_delay = -1;
        aec->delay_est.current_delay = -1;
        aec->delay_est.pending_delay = -1;
        aec->delay_est.has_pending = 0;
        aec->delay_est.samples_accumulated = 0;
        aec->delay_est.samples_since_est = 0;
        aec->delay_est.init_samples = (int)(config->delay_est_init_s * config->sample_rate);
        aec->delay_est.period_samples = (int)(config->delay_est_period_s * config->sample_rate);
        aec->delay_est.init_done = 0;

        aec->delay_est.fft = fft_create(seg_size);
        aec->delay_est.mic_buf  = (float*)calloc(seg_size, sizeof(float));
        aec->delay_est.ref_buf  = (float*)calloc(seg_size, sizeof(float));
        aec->delay_est.cross_re = (float*)calloc(de_nfreqs, sizeof(float));
        aec->delay_est.cross_im = (float*)calloc(de_nfreqs, sizeof(float));
        aec->delay_est.mic_spec = (Complex*)calloc(de_nfreqs, sizeof(Complex));
        aec->delay_est.ref_spec = (Complex*)calloc(de_nfreqs, sizeof(Complex));
        aec->delay_est.phat     = (Complex*)calloc(de_nfreqs, sizeof(Complex));
        aec->delay_est.gcc      = (float*)calloc(seg_size, sizeof(float));

        if (!aec->delay_est.fft || !aec->delay_est.mic_buf || !aec->delay_est.ref_buf ||
            !aec->delay_est.cross_re || !aec->delay_est.cross_im ||
            !aec->delay_est.mic_spec || !aec->delay_est.ref_spec ||
            !aec->delay_est.phat || !aec->delay_est.gcc) {
            aec_destroy(aec); return NULL;
        }

        /* Reference delay ring buffer: max_delay + some extra for hop reads */
        int ring_size = max_delay + 4096;
        aec->ref_delay_buf = (float*)calloc(ring_size, sizeof(float));
        aec->ref_delay_buf_size = ring_size;
        aec->ref_delay_write_pos = 0;
        aec->ref_delay_filled = 0;
        if (!aec->ref_delay_buf) { aec_destroy(aec); return NULL; }
    }

    return aec;
}

void aec_destroy(Aec* aec) {
    if (!aec) return;
    pbfdkf_destroy(aec->filter);
    pbfdkf_destroy(aec->shadow_filter);
    res_destroy(aec->res);
    dtd_destroy(aec->dtd_coherence);
    dtd_destroy(aec->dtd_divergence);
    free(aec->mic_buf);
    free(aec->ref_buf);
    free(aec->raw_output);
    free(aec->per_bin_mu_scale);
    /* Delay estimation */
    fft_destroy(aec->delay_est.fft);
    free(aec->delay_est.mic_buf);
    free(aec->delay_est.ref_buf);
    free(aec->delay_est.cross_re);
    free(aec->delay_est.cross_im);
    free(aec->delay_est.mic_spec);
    free(aec->delay_est.ref_spec);
    free(aec->delay_est.phat);
    free(aec->delay_est.gcc);
    free(aec->ref_delay_buf);
    free(aec);
}

void aec_reset(Aec* aec) {
    if (!aec) return;
    pbfdkf_reset(aec->filter);
    pbfdkf_reset(aec->shadow_filter);
    pbfdkf_set_q_ratio(aec->shadow_filter, aec->config.kalman_q_high,
                        aec->config.kalman_q_low, aec->config.shadow_q_ratio);
    if (aec->res) res_reset(aec->res);
    if (aec->dtd_coherence) dtd_reset(aec->dtd_coherence);
    if (aec->dtd_divergence) dtd_reset(aec->dtd_divergence);
    aec->prev_dtd_conf = 0.0f;
    aec->simple_mu_ratio = 1.0f;
    aec->simple_mu_holdoff = 0;
    aec->warmup_frames = aec->config.warmup_frames;
    aec->warmup_far_active = 0;
    aec->filter_converged = 0;
    aec->conv_counter = 0;
    aec->near_power = 0; aec->raw_error_power = 0; aec->final_error_power = 0;
    aec->near_power_sum = 0; aec->raw_error_power_sum = 0; aec->final_error_power_sum = 0;
    aec->limiter_gain = 1.0f;
    aec->inst_erle_smooth = 1.0f;
    aec->divergence_indicator = 0.0f;
    aec->main_err_smooth = 0; aec->shadow_err_smooth = 0;
    aec->shadow_frame_count = 0; aec->shadow_copy_counter = 0;
    aec->copy_err_baseline = 0.0f;
    aec->prev_total_err = 0.0f; aec->epc_active = 0; aec->epc_hangover_count = 0;
    aec->far_env_mean = 0; aec->far_env_var = 0; aec->far_active_prev = 0;
    aec->is_stationary_far = 0; aec->wn_err_baseline = 1e-8f;
    aec->stat_dt_hangover = 0; aec->is_stationary_dt = 0;
    free(aec->per_bin_mu_scale); aec->per_bin_mu_scale = NULL;

    /* v2.3.x state */
    aec->dt_from_energy = 0.0f;
    aec->dt_from_shadow = 0.0f;
    aec->shadow_advantage = 1.0f;
    aec->shadow_advantage_streak = 0;
    aec->erl_estimate = 0.1f;
    aec->epc_render_forced_remaining = 0;
    aec->erle_factor_prev = 0.0f;
    aec->erle_window_near = 1e-10f;
    aec->erle_window_err = 1e-10f;
    aec->saturation_level = 0.0f;
    aec->delay_last_par = 0.0f;

    /* Reset delay estimation */
    if (aec->delay_est.enabled) {
        memset(aec->delay_est.mic_buf, 0, aec->delay_est.seg_size * sizeof(float));
        memset(aec->delay_est.ref_buf, 0, aec->delay_est.seg_size * sizeof(float));
        memset(aec->delay_est.cross_re, 0, aec->delay_est.n_freqs * sizeof(float));
        memset(aec->delay_est.cross_im, 0, aec->delay_est.n_freqs * sizeof(float));
        aec->delay_est.buf_pos = 0;
        aec->delay_est.n_updates = 0;
        aec->delay_est.estimated_delay = -1;
        aec->delay_est.current_delay = -1;
        aec->delay_est.pending_delay = -1;
        aec->delay_est.has_pending = 0;
        aec->delay_est.samples_accumulated = 0;
        aec->delay_est.samples_since_est = 0;
        aec->delay_est.init_done = 0;
        memset(aec->ref_delay_buf, 0, aec->ref_delay_buf_size * sizeof(float));
        aec->ref_delay_write_pos = 0;
        aec->ref_delay_filled = 0;
    }
}

/* --- DTD-based mu_scale (Python aec.py _compute_mu_scale, lines 3526-3558) ---
 * Coherence is primary; divergence is fallback only when coherence inactive.
 * Confidence has memory decay 0.9 to avoid sudden drops.
 * Pre-convergence: mu_min raised to 0.3 so filter can still learn during DT.
 * EPC: mu_scale floor at config.epc_mu_floor. */
static float compute_mu_scale_dtd(Aec* aec) {
    float conf_div = dtd_get_confidence(aec->dtd_divergence);
    float conf_coh = dtd_get_confidence(aec->dtd_coherence);

    float raw_conf;
    if (conf_coh > 0.1f) {
        raw_conf = conf_coh;
    } else {
        raw_conf = (conf_div > conf_coh) ? conf_div : conf_coh;
    }

    float decayed = aec->prev_dtd_conf * 0.9f;
    float conf = (raw_conf > decayed) ? raw_conf : decayed;
    aec->prev_dtd_conf = conf;

    if (conf == 0.0f) return 1.0f;

    /* Use shadow_mu_min as dtd_mu_min_ratio surrogate (Python BALANCED defaults
     * align: dtd_mu_min_ratio=0.05 vs shadow_mu_min=0.6 — these are different
     * but both clamp the floor). Match Python by using a separate min_r=0.05.
     */
    float min_r = 0.05f;
    if (!aec->filter_converged) {
        if (min_r < 0.3f) min_r = 0.3f;
    }
    float mu_scale = 1.0f - conf * (1.0f - min_r);

    if (aec->epc_active) {
        if (mu_scale < aec->config.epc_mu_floor) mu_scale = aec->config.epc_mu_floor;
    }
    return mu_scale;
}

/* --- Simple variable mu (Valin 2007 RER-inspired, used as backup) --- */
static float get_simple_mu_scale(Aec* aec) {
    /* If DTD active and not pre-warmup, prefer DTD-based mu_scale. */
    if (aec->dtd_coherence && aec->warmup_frames <= 0) {
        return compute_mu_scale_dtd(aec);
    }
    float mu_min = aec->config.shadow_mu_min;

    /* Warmup: fast convergence, but respect DT signals to protect near-end */
    if (aec->warmup_frames > 0) {
        /* Only consume warmup when far-end is active (don't waste on silence) */
        if (aec->warmup_far_active)
            aec->warmup_frames--;
        /* Strong DT detected: don't force high mu, protect filter */
        if (aec->simple_mu_ratio < 0.2f)
            return maxf(0.2f, aec->simple_mu_ratio);
        return minf(1.0f, maxf(0.5f, aec->simple_mu_ratio + 0.2f));
    }

    if (!aec->filter_converged)
        mu_min = maxf(mu_min, 0.3f);
    else
        mu_min = maxf(mu_min, 0.2f);

    /* Per-bin mu_scale from RES (post-convergence) */
    /* (returned as scalar fallback when per_bin not available) */
    float mu_scale = mu_min + (1.0f - mu_min) * aec->simple_mu_ratio;

    /* Echo path change: keep mu high so filter can adapt to new path */
    if (aec->epc_active)
        mu_scale = maxf(mu_scale, aec->config.epc_mu_floor);

    return mu_scale;
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

    /* FS-parity fix: match Python aec.py:2392-2399.
     * (1) Don't update during bilateral silence — preserves value for warmup.
     * (2) Quick reset to 0.8 on silence→active transition. */
    if (far_pwr < 1e-6f && err_pwr < 1e-6f) return;
    if (far_pwr > 1e-4f && aec->simple_mu_ratio < 0.1f) {
        aec->simple_mu_ratio = 0.8f;
        aec->simple_mu_holdoff = 0;
        return;
    }

    float ratio = far_pwr / err_pwr;
    if (ratio > 1.0f) ratio = 1.0f;

    /* Echo estimate ratio boost (matches Python: sum(|echo|²) / sum(|near|²)) */
    const Complex* echo_spec = pbfdkf_get_echo_spec(aec->filter);
    const Complex* near_spec = pbfdkf_get_near_spec(aec->filter);
    if (echo_spec && near_spec) {
        int nf = aec->n_freqs;
        float echo_pwr = 1e-10f, near_spec_pwr = 1e-10f;
        for (int k = 0; k < nf; k++) {
            echo_pwr += echo_spec[k].r * echo_spec[k].r +
                        echo_spec[k].i * echo_spec[k].i;
            near_spec_pwr += near_spec[k].r * near_spec[k].r +
                             near_spec[k].i * near_spec[k].i;
        }
        if (near_spec_pwr > 1e-8f) {
            float ratio_echo = clampf(echo_pwr / near_spec_pwr, 0.0f, 1.0f);
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

void aec_context_destroy(AecResContext* ctx) {
    if (!ctx) return;
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
    ctx->erle_factor = clampf(inst_erle / 10.0f, 0.0f, 1.0f);

    /* DT indicator with ERLE correction (v2.0.0) */
    float mic_pwr = 0.0f;
    for (int i = 0; i < hop; i++) mic_pwr += mic[i] * mic[i];
    mic_pwr = mic_pwr / hop + 1e-10f;
    float far_p = far_power + 1e-10f;
    float raw_dt = 1.0f - far_p / (mic_pwr + far_p);

    /* Instant ERLE with 3-frame EMA smoothing */
    float raw_err_pwr = 0.0f;
    for (int i = 0; i < hop; i++) raw_err_pwr += aec->raw_output[i] * aec->raw_output[i];
    raw_err_pwr = raw_err_pwr / hop + 1e-10f;
    float inst_erle_raw = mic_pwr / raw_err_pwr;
    aec->inst_erle_smooth = 0.7f * aec->inst_erle_smooth + 0.3f * inst_erle_raw;
    if (aec->inst_erle_smooth > 2.0f) {
        float erle_for_dt = aec->inst_erle_smooth < 4.0f ? aec->inst_erle_smooth : 4.0f;
        raw_dt /= erle_for_dt;
    }
    ctx->dt_indicator = clampf(raw_dt, 0.0f, 0.8f);

    /* Dynamic over_sub */
    float base_over_sub = cfg->res_over_sub_base +
                           cfg->res_over_sub_scale * ctx->erle_factor;
    float dt_reduction = cfg->res_dt_reduction * ctx->dt_indicator;
    ctx->over_sub = maxf(base_over_sub - dt_reduction, 0.5f);

    /* Divergence (simple: error/near ratio) */
    float err_pwr = aec->raw_error_power + 1e-10f;
    float near_pwr = aec->near_power + 1e-10f;
    ctx->divergence = clampf(err_pwr / near_pwr - 0.5f, 0.0f, 1.0f);

    /* Stationary far-end DT detection result (computed in aec_process_ex) */
    ctx->is_stationary_dt = aec->is_stationary_dt;

    /* v2.3.x context fields */
    ctx->saturation_level = aec->saturation_level;
    ctx->erl_estimate = aec->erl_estimate;
}

/* --- Delay estimation (GCC-PHAT) --- */

/* Update smoothed cross-spectrum from filled segment buffers */
static void delay_est_update_cross(Aec* aec) {
    int nf = aec->delay_est.n_freqs;
    float alpha = aec->delay_est.alpha;

    fft_forward(aec->delay_est.fft, aec->delay_est.mic_buf, aec->delay_est.mic_spec);
    fft_forward(aec->delay_est.fft, aec->delay_est.ref_buf, aec->delay_est.ref_spec);

    /* cross = mic_spec * conj(ref_spec) */
    aec->delay_est.n_updates++;
    if (aec->delay_est.n_updates == 1) {
        for (int k = 0; k < nf; k++) {
            float mr = aec->delay_est.mic_spec[k].r;
            float mi = aec->delay_est.mic_spec[k].i;
            float rr = aec->delay_est.ref_spec[k].r;
            float ri = aec->delay_est.ref_spec[k].i;
            aec->delay_est.cross_re[k] = mr * rr + mi * ri;
            aec->delay_est.cross_im[k] = mi * rr - mr * ri;
        }
    } else {
        for (int k = 0; k < nf; k++) {
            float mr = aec->delay_est.mic_spec[k].r;
            float mi = aec->delay_est.mic_spec[k].i;
            float rr = aec->delay_est.ref_spec[k].r;
            float ri = aec->delay_est.ref_spec[k].i;
            float cr = mr * rr + mi * ri;
            float ci = mi * rr - mr * ri;
            aec->delay_est.cross_re[k] = alpha * aec->delay_est.cross_re[k] + (1.0f - alpha) * cr;
            aec->delay_est.cross_im[k] = alpha * aec->delay_est.cross_im[k] + (1.0f - alpha) * ci;
        }
    }
}

/* Estimate delay from accumulated cross-spectrum */
static void delay_est_estimate(Aec* aec) {
    int nf = aec->delay_est.n_freqs;
    int seg_size = aec->delay_est.seg_size;

    /* PHAT normalization: cross / |cross| */
    for (int k = 0; k < nf; k++) {
        float re = aec->delay_est.cross_re[k];
        float im = aec->delay_est.cross_im[k];
        float mag = sqrtf(re * re + im * im) + 1e-10f;
        aec->delay_est.phat[k].r = re / mag;
        aec->delay_est.phat[k].i = im / mag;
    }

    /* IRFFT */
    fft_inverse(aec->delay_est.fft, aec->delay_est.phat, aec->delay_est.gcc);

    /* Find peak in [0, max_delay] range (positive delays: mic lags ref) */
    int max_d = aec->delay_est.max_delay;
    if (max_d > seg_size / 2) max_d = seg_size / 2;

    int best_pos = 0;
    float best_val = fabsf(aec->delay_est.gcc[0]);
    float sum_gcc = fabsf(aec->delay_est.gcc[0]);
    for (int d = 1; d <= max_d; d++) {
        float v = fabsf(aec->delay_est.gcc[d]);
        sum_gcc += v;
        if (v > best_val) {
            best_val = v;
            best_pos = d;
        }
    }

    /* A5: Compute PAR (Peak-to-Average Ratio) for confidence */
    float avg_gcc = sum_gcc / (float)(max_d + 1) + 1e-10f;
    aec->delay_last_par = best_val / avg_gcc;

    aec->delay_est.estimated_delay = best_pos;
    aec->delay_est.samples_since_est = 0;
}

/* Accumulate mic/ref samples for delay estimation.
   Returns 1 if a new estimate was produced. */
static int delay_est_accumulate(Aec* aec, const float* mic, const float* ref, int n) {
    aec->delay_est.samples_accumulated += n;
    aec->delay_est.samples_since_est += n;

    int seg_size = aec->delay_est.seg_size;
    int seg_hop = aec->delay_est.seg_hop;
    int remaining = n;
    int src_pos = 0;

    while (remaining > 0) {
        int space = seg_size - aec->delay_est.buf_pos;
        int chunk = remaining < space ? remaining : space;
        memcpy(aec->delay_est.mic_buf + aec->delay_est.buf_pos, mic + src_pos, chunk * sizeof(float));
        memcpy(aec->delay_est.ref_buf + aec->delay_est.buf_pos, ref + src_pos, chunk * sizeof(float));
        aec->delay_est.buf_pos += chunk;
        src_pos += chunk;
        remaining -= chunk;

        if (aec->delay_est.buf_pos >= seg_size) {
            delay_est_update_cross(aec);
            /* Shift by seg_hop (50% overlap) */
            memmove(aec->delay_est.mic_buf, aec->delay_est.mic_buf + seg_hop, seg_hop * sizeof(float));
            memmove(aec->delay_est.ref_buf, aec->delay_est.ref_buf + seg_hop, seg_hop * sizeof(float));
            aec->delay_est.buf_pos = seg_hop;
        }
    }

    /* Need at least 2 cross-spectrum updates before estimating */
    if (aec->delay_est.n_updates < 2)
        return 0;

    if (!aec->delay_est.init_done) {
        if (aec->delay_est.samples_accumulated >= aec->delay_est.init_samples) {
            delay_est_estimate(aec);
            aec->delay_est.init_done = 1;
            return 1;
        }
    } else {
        if (aec->delay_est.samples_since_est >= aec->delay_est.period_samples) {
            delay_est_estimate(aec);
            return 1;
        }
    }
    return 0;
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

    /* Copy input (working buffers) */
    memcpy(aec->mic_buf, near_end, hop * sizeof(float));
    memcpy(aec->ref_buf, far_end, hop * sizeof(float));

    float* mic = aec->mic_buf;
    float* ref = aec->ref_buf;

    /* === Stationary far-end detection (CV² gate) ===
     * Matches Python aec.py — α=0.99 EMA on far_pwr, CV² < 0.02 → stationary.
     * α=0.99 (TC≈1s) needed because single-hop WN power std/mean ≈ 11%.
     */
    {
        float far_pwr_global = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_global += far_end[i] * far_end[i];
        far_pwr_global = far_pwr_global / hop + 1e-10f;

        if (far_pwr_global > 1e-6f) {
            if (!aec->far_active_prev) {
                aec->far_env_mean = far_pwr_global;
                aec->far_env_var = 0.0f;
                aec->far_active_prev = 1;
            } else {
                const float alpha_cv = 0.99f;
                float old_mean = aec->far_env_mean;
                aec->far_env_mean = alpha_cv * aec->far_env_mean
                                    + (1.0f - alpha_cv) * far_pwr_global;
                float dev = far_pwr_global - old_mean;
                aec->far_env_var = alpha_cv * aec->far_env_var
                                   + (1.0f - alpha_cv) * (dev * dev);
            }
            float far_cv2 = aec->far_env_var /
                            (aec->far_env_mean * aec->far_env_mean + 1e-10f);
            aec->is_stationary_far = (far_cv2 < 0.02f) ? 1 : 0;
        } else {
            aec->far_active_prev = 0;
            aec->is_stationary_far = 0;
        }
    }

    /* === Delay estimation === */
    if (aec->delay_est.enabled) {
        /* Accumulate into delay estimator */
        delay_est_accumulate(aec, mic, ref, hop);

        int new_delay = aec->delay_est.estimated_delay;
        if (new_delay >= 0) {
            if (aec->delay_est.current_delay < 0) {
                /* First estimate — A5 PAR confidence gate + n_updates >= 3.
                 * n_updates >= 3 ensures enough cross-spectrum averaging
                 * to avoid spurious delay=0 from early weak correlation.
                 * (Matches Python aec.py line 2437.) */
                if (aec->delay_last_par > 5.0f && aec->delay_est.n_updates >= 3) {
                    aec->delay_est.current_delay = new_delay;
                    /* B3: full reset for clean start at correct delay.
                     * Clear X_buf/input buffers too — old spectra from wrong
                     * alignment corrupt early Kalman updates.
                     * (Matches Python self.filter.reset().) */
                    pbfdkf_reset(aec->filter);
                    pbfdkf_reset(aec->shadow_filter);
                    /* Restore shadow Q ratio after full reset */
                    pbfdkf_set_q_ratio(aec->shadow_filter, cfg->kalman_q_high,
                                        cfg->kalman_q_low, cfg->shadow_q_ratio);
                    if (aec->res) res_reset(aec->res);
                    aec->filter_converged = 0;
                    aec->conv_counter = 0;
                }
            } else if (abs(new_delay - aec->delay_est.current_delay) > 32) {
                /* Require two consecutive consistent estimates before switching */
                if (aec->delay_est.has_pending &&
                    abs(new_delay - aec->delay_est.pending_delay) < 16) {
                    aec->delay_est.current_delay = new_delay;
                    aec->delay_est.has_pending = 0;
                    /* Trigger EPC for fast re-convergence */
                    aec->epc_active = 1;
                    aec->epc_hangover_count = cfg->epc_hangover;
                    pbfdkf_set_q_high(aec->filter, cfg->kalman_q_high);
                    pbfdkf_set_q_high(aec->shadow_filter,
                                      cfg->kalman_q_high * cfg->shadow_q_ratio);
                    pbfdkf_set_p_floor_epc(aec->filter, 1.0f, 30);
                    pbfdkf_set_p_floor_epc(aec->shadow_filter, 1.0f, 30);
                    aec->filter_converged = 0;
                    aec->conv_counter = 0;
                } else {
                    aec->delay_est.pending_delay = new_delay;
                    aec->delay_est.has_pending = 1;
                }
            }
        }

        /* Write ref into ring buffer */
        int ring_sz = aec->ref_delay_buf_size;
        int w = aec->ref_delay_write_pos;
        for (int i = 0; i < hop; i++) {
            aec->ref_delay_buf[w] = ref[i];
            w++;
            if (w >= ring_sz) w = 0;
        }
        aec->ref_delay_write_pos = w;
        aec->ref_delay_filled += hop;

        /* Read delayed ref from ring buffer */
        int delay = aec->delay_est.current_delay;
        if (delay < 0) delay = 0;
        if (aec->ref_delay_filled >= delay + hop) {
            /* Read position = write_pos - delay - hop */
            int r = aec->ref_delay_write_pos - delay - hop;
            if (r < 0) r += ring_sz;
            for (int i = 0; i < hop; i++) {
                ref[i] = aec->ref_delay_buf[r];
                r++;
                if (r >= ring_sz) r = 0;
            }
        }
        /* else: not enough data yet, use ref as-is */
    }

    /* FS-parity fix: update warmup_far_active BEFORE computing mu_scale so that
     * the current frame's far activity affects warmup countdown, matching Python
     * aec.py:2512 (_warmup_far_active set before _get_simple_mu_scale). */
    {
        float far_pwr_wup = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_wup += ref[i] * ref[i];
        aec->warmup_far_active = (far_pwr_wup / hop > 1e-6f);
    }

    /* Compute mu_scale */
    float mu_scale = get_simple_mu_scale(aec);

    /* === Main filter (per-bin mu post-convergence for FS-parity with Python).
     * Python aec.py 3578-3579: when per_bin_mu_scale set, return is
     * np.maximum(per_bin_mu_scale, mu_min) — apply floor elementwise. C had
     * NO floor → bins below mu_min over-suppress filter learning. */
    const float* per_bin_mu = NULL;
    static float per_bin_mu_floored[513];  /* enough for n_freqs=257 */
    if (aec->filter_converged && aec->per_bin_mu_scale) {
        float floor = aec->config.shadow_mu_min;
        if (floor < 0.2f) floor = 0.2f;  /* converged: max(shadow_mu_min, 0.2) */
        int nf = aec->n_freqs;
        for (int k = 0; k < nf; k++) {
            float v = aec->per_bin_mu_scale[k];
            per_bin_mu_floored[k] = (v < floor) ? floor : v;
        }
        per_bin_mu = per_bin_mu_floored;
    }
    pbfdkf_process_per_bin(aec->filter, mic, ref, aec->raw_output,
                           per_bin_mu, mu_scale);

    /* === Shadow filter (always mu_scale=1.0) === */
    float shadow_out[160]; /* stack alloc, hop <= 160 */
    aec->shadow_frame_count++;
    pbfdkf_process(aec->shadow_filter, mic, ref, shadow_out, 1.0f);

    /* v3.x DTD update — Python aec.py 4197-4214: gated on filter_converged.
     * Coherence: error_spec + far_spec from main filter.
     * Divergence: near_end + raw_output (post-filter). */
    if (aec->filter_converged && aec->dtd_coherence) {
        const Complex* err_spec = pbfdkf_get_error_spec(aec->filter);
        const Complex* far_spec_p = pbfdkf_get_far_spec(aec->filter);
        if (err_spec && far_spec_p) {
            dtd_detect_coherence(aec->dtd_coherence, err_spec, far_spec_p);
        }
        if (aec->dtd_divergence) {
            dtd_detect_divergence(aec->dtd_divergence, mic, aec->raw_output, hop);
        }
    }

    /* Smoothed error tracking */
    float main_err = pbfdkf_get_error_energy(aec->filter);
    float shadow_err = pbfdkf_get_error_energy(aec->shadow_filter);
    float alpha_s = cfg->shadow_err_alpha;
    aec->main_err_smooth   = alpha_s * aec->main_err_smooth   + (1.0f - alpha_s) * main_err;
    aec->shadow_err_smooth = alpha_s * aec->shadow_err_smooth + (1.0f - alpha_s) * shadow_err;

    /* === Shadow DTD with offset (B2) === */
    {
        float far_pwr_shadow = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_shadow += ref[i] * ref[i];
        int far_active_shadow = (far_pwr_shadow / hop > 1e-4f);

        if (aec->shadow_frame_count >= 50 && far_active_shadow) {
            float shadow_advantage = aec->main_err_smooth / (aec->shadow_err_smooth + 1e-10f);
            aec->shadow_advantage = shadow_advantage;
            float dt_from_shadow = clampf(
                (shadow_advantage - cfg->shadow_dtd_offset) / cfg->shadow_dtd_advantage_scale,
                0.0f, 1.0f);
            aec->dt_from_shadow = 0.7f * aec->dt_from_shadow + 0.3f * dt_from_shadow;
        } else {
            aec->dt_from_shadow *= 0.95f;
        }
    }

    /* Copy gate — aligned with Python aec.py:2616-2662
     * Key fixes vs old C code:
     * (a) baseline-gated copy_allowed (error_is_normal) instead of crude far/err ratio
     * (b) shadow_advantage_streak reset on copy_allowed=False (G3 Fix 1)
     * (c) err_balance stability check for baseline update */
    if (aec->shadow_frame_count >= 50) {
        float threshold = cfg->shadow_copy_threshold;

        float far_pwr = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr += ref[i] * ref[i];
        int far_active = (far_pwr / hop > 1e-4f);

        float err_sum = aec->main_err_smooth + aec->shadow_err_smooth + 1e-10f;
        float err_balance = fabsf(aec->main_err_smooth - aec->shadow_err_smooth) / err_sum;
        int is_stable_fs = far_active && err_balance < 0.3f && !aec->epc_active;

        /* Update copy_err_baseline when filters are stable */
        if (is_stable_fs) {
            float best_err = aec->main_err_smooth < aec->shadow_err_smooth
                           ? aec->main_err_smooth : aec->shadow_err_smooth;
            if (aec->copy_err_baseline < 1e-10f)
                aec->copy_err_baseline = best_err;  /* cold start */
            else
                aec->copy_err_baseline = 0.995f * aec->copy_err_baseline + 0.005f * best_err;
        }

        int error_is_normal = (aec->copy_err_baseline < 1e-10f) ||
                              (aec->main_err_smooth < aec->copy_err_baseline * 4.0f + 1e-10f);
        int not_saturating = (aec->saturation_level < 0.3f);
        int copy_allowed = far_active && error_is_normal && !aec->epc_active && not_saturating;

        if (copy_allowed) {
            if (aec->shadow_err_smooth < aec->main_err_smooth * threshold) {
                aec->shadow_copy_counter++;
                aec->shadow_advantage_streak++;
            } else {
                aec->shadow_copy_counter = 0;
                aec->shadow_advantage_streak = 0;
            }

            if (aec->shadow_copy_counter >= cfg->shadow_copy_hysteresis
                && aec->shadow_advantage_streak >= 10) {
                /* Shadow → Main copy */
                pbfdkf_copy_weights(aec->filter, aec->shadow_filter);
                aec->main_err_smooth = aec->shadow_err_smooth;
                aec->shadow_copy_counter = 0;
                aec->shadow_advantage_streak = 0;
            } else if (aec->main_err_smooth < aec->shadow_err_smooth * threshold
                       && error_is_normal) {
                /* Bidirectional: Main → Shadow */
                pbfdkf_copy_weights(aec->shadow_filter, aec->filter);
                aec->shadow_err_smooth = aec->main_err_smooth;
            }
        } else {
            /* (b) reset both counter AND streak on copy_allowed=False */
            aec->shadow_copy_counter = 0;
            aec->shadow_advantage_streak = 0;
        }
    }

    /* === Echo Path Change detection (only after convergence) === */
    if (aec->shadow_filter && aec->filter_converged) {
        float total_err = aec->main_err_smooth + aec->shadow_err_smooth;
        float delta_ratio = 0.0f;
        if (total_err > 1e-10f)
            delta_ratio = fabsf(aec->main_err_smooth - aec->shadow_err_smooth) / total_err;

        int errors_rising = (total_err > aec->prev_total_err * cfg->epc_total_rise
                             && aec->prev_total_err > 1e-10f);
        int is_echo_change = errors_rising && (delta_ratio < cfg->epc_delta_threshold);
        /* Stationary far-end guard: error rise is DT, not echo path change */
        if (is_echo_change && aec->is_stationary_far) is_echo_change = 0;
        aec->prev_total_err = total_err;

        if (is_echo_change) {
            aec->epc_hangover_count = cfg->epc_hangover;
            aec->epc_active = 1;
            /* Reset Q to Q_high for fast re-convergence */
            pbfdkf_set_q_high(aec->filter, cfg->kalman_q_high);
            pbfdkf_set_q_high(aec->shadow_filter,
                              cfg->kalman_q_high * cfg->shadow_q_ratio);
            pbfdkf_set_p_floor_epc(aec->filter, 1.0f, 30);
            pbfdkf_set_p_floor_epc(aec->shadow_filter, 1.0f, 30);
            aec->filter_converged = 0;
            aec->conv_counter = 0;
            /* G1: render-forced countdown */
            aec->epc_render_forced_remaining = cfg->epc_hangover;
            /* A4: clamp ERL estimate on EPC */
            if (aec->erl_estimate > 0.3f) aec->erl_estimate = 0.3f;
        } else if (aec->epc_hangover_count > 0) {
            aec->epc_hangover_count--;
            aec->epc_active = 1;
        } else {
            aec->epc_active = 0;
        }
    }

    /* Copy raw output for ERLE + RES input */
    memcpy(output, aec->raw_output, hop * sizeof(float));

    /* === Stationary DT detection (jump_ratio + 800ms hangover) ===
     * Always run regardless of internal/external RES, so AecResContext
     * sees a valid is_stationary_dt for pipeline use.
     */
    if (aec->is_stationary_far && aec->filter_converged) {
        const Complex* err_spec = pbfdkf_get_error_spec(aec->filter);
        float track_err_pwr = 1e-10f;
        if (err_spec) {
            int fft_size = pbfdkf_get_fft_size(aec->filter);
            float freq_per_bin = (float)cfg->sample_rate / (float)fft_size;
            int vb_start = (int)(100.0f / freq_per_bin);
            if (vb_start < 1) vb_start = 1;
            int vb_limit = (int)(3000.0f / freq_per_bin);
            if (vb_limit > aec->n_freqs) vb_limit = aec->n_freqs;
            for (int k = vb_start; k < vb_limit; k++) {
                track_err_pwr += err_spec[k].r * err_spec[k].r
                               + err_spec[k].i * err_spec[k].i;
            }
        }
        if (aec->wn_err_baseline < 1e-6f) {
            aec->wn_err_baseline = track_err_pwr;
        }
        float jump_ratio = track_err_pwr / (aec->wn_err_baseline + 1e-10f);

        if (jump_ratio > 1.5f) {
            aec->stat_dt_hangover = 80;  /* 800ms hold */
        }

        if (aec->stat_dt_hangover > 0) {
            aec->is_stationary_dt = 1;
            aec->stat_dt_hangover--;
            aec->wn_err_baseline = 0.999f * aec->wn_err_baseline
                                 + 0.001f * track_err_pwr;
        } else {
            aec->is_stationary_dt = 0;
            aec->wn_err_baseline = 0.95f * aec->wn_err_baseline
                                 + 0.05f * track_err_pwr;
        }
    } else {
        aec->is_stationary_dt = 0;
    }

    /* EPC physical gate */
    if (aec->epc_active) {
        aec->is_stationary_dt = 0;
    }

    /* D4: wn_err_baseline slow tracking during converged non-stationary far */
    {
        float far_pwr_d4 = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_d4 += ref[i] * ref[i];
        float far_pwr_d4_avg = far_pwr_d4 / hop;
        float raw_err_pwr_d4 = 0.0f;
        for (int i = 0; i < hop; i++) raw_err_pwr_d4 += aec->raw_output[i] * aec->raw_output[i];
        raw_err_pwr_d4 = raw_err_pwr_d4 / hop + 1e-10f;
        if (aec->filter_converged && !aec->is_stationary_far
            && far_pwr_d4_avg > 1e-4f && aec->wn_err_baseline > 1e-6f) {
            aec->wn_err_baseline = 0.995f * aec->wn_err_baseline + 0.005f * raw_err_pwr_d4;
        }
    }

    /* === RES post-filter === */
    if (aec->res) {
        /* far_power from ref */
        float far_power = 0.0f;
        for (int i = 0; i < hop; i++) far_power += ref[i] * ref[i];
        far_power /= hop;

        /* mic_pwr for DT + energy DT */
        float mic_pwr = 0.0f;
        for (int i = 0; i < hop; i++) mic_pwr += mic[i] * mic[i];
        mic_pwr = mic_pwr / hop + 1e-10f;

        /* raw error power */
        float raw_err_pwr_dt = 0.0f;
        for (int i = 0; i < hop; i++) raw_err_pwr_dt += aec->raw_output[i] * aec->raw_output[i];
        raw_err_pwr_dt = raw_err_pwr_dt / hop + 1e-10f;

        int far_active = (far_power > 1e-4f);

        /* === E: Windowed ERLE (v2.1.0 style) === */
        float inst_erle = aec_get_erle_instant(aec);
        {
            float erle_decay = 0.999f;
            aec->erle_window_near = erle_decay * aec->erle_window_near + aec->near_power;
            aec->erle_window_err = erle_decay * aec->erle_window_err + aec->raw_error_power;
            float erle_windowed = 10.0f * fast_log10((aec->erle_window_near + 1e-10f) /
                                                      (aec->erle_window_err + 1e-10f));
            float erle_for_factor = inst_erle > erle_windowed ? inst_erle : erle_windowed;

            /* Dynamic over_sub: use max(instant, windowed) ERLE */
            float erle_factor_raw = clampf(erle_for_factor / 10.0f, 0.0f, 1.0f);
            /* Note: erle_factor ramp already applied by previous agent — keep as-is */
            float erle_factor = erle_factor_raw;
            aec->erle_factor_prev = erle_factor;

            float base_over_sub = cfg->res_over_sub_base +
                                   cfg->res_over_sub_scale * erle_factor;

            /* DT indicator — match Python aec.py 3963-3972:
             *   if enable_dtd: raw_dt = get_dtd_confidence()  (coherence > 0.1
             *                                                   else max with div)
             *   else:          raw_dt = max(dt_from_energy, simple_dt * 0.5)
             * Then inst_erle correction (only no-DTD path). */
            float raw_dt;
            int dtd_active = (aec->dtd_coherence != NULL);
            if (dtd_active) {
                /* get_dtd_confidence: coherence primary, divergence fallback */
                float c_coh = dtd_get_confidence(aec->dtd_coherence);
                float c_div = dtd_get_confidence(aec->dtd_divergence);
                if (c_coh > 0.1f) raw_dt = c_coh;
                else raw_dt = (c_div > c_coh) ? c_div : c_coh;
            } else {
                float far_p_dt = far_power + 1e-10f;
                float simple_dt = 1.0f - far_p_dt / (mic_pwr + far_p_dt);
                raw_dt = aec->dt_from_energy > simple_dt * 0.5f
                          ? aec->dt_from_energy
                          : simple_dt * 0.5f;
                float inst_erle_raw_dt = mic_pwr / raw_err_pwr_dt;
                aec->inst_erle_smooth = 0.7f * aec->inst_erle_smooth + 0.3f * inst_erle_raw_dt;
                if (aec->inst_erle_smooth > 2.0f) {
                    float erle_for_dt = aec->inst_erle_smooth < 4.0f ? aec->inst_erle_smooth : 4.0f;
                    raw_dt /= erle_for_dt;
                }
            }

            /* EPC physical gate also zeroes raw_dt for the dt_indicator below */
            if (aec->epc_active) raw_dt = 0.0f;

            float dt_indicator = clampf(raw_dt, 0.0f, 0.8f);

            float dt_reduction = cfg->res_dt_reduction * dt_indicator;
            float over_sub = maxf(base_over_sub - dt_reduction, 0.5f);

            /* === F: Energy DT signal — match Python DoubleTalkAnalyzer.update_energy_dt
             * (aec.py 2483-2497). Gate: far_active AND far_pwr > 1e-4 (NOT
             * far_active_prev AND far_active). Else fade-out via slow decay
             * (NOT hard zero). */
            if (far_active && far_power > 1e-4f) {
                float erl_ceiling = 1.0f / (aec->erl_estimate > 0.01f ? aec->erl_estimate : 0.01f);
                float max_echo = far_power * erl_ceiling * 2.0f;
                float dt_energy = (mic_pwr - max_echo) / mic_pwr;
                if (dt_energy < 0.0f) dt_energy = 0.0f;
                if (dt_energy > aec->dt_from_energy)
                    aec->dt_from_energy = 0.3f * aec->dt_from_energy + 0.7f * dt_energy;
                else
                    aec->dt_from_energy = 0.9f * aec->dt_from_energy + 0.1f * dt_energy;
            } else {
                /* inst = 0; same EMA branch with 0 input → slow decay only */
                aec->dt_from_energy = 0.9f * aec->dt_from_energy;
            }

            /* === G: Dynamic ERL tracking — match Python aec.py 3970-3974
             * Two gates: raw_dt_ratio < 2.0 AND inst_erl_raw < 1.5 (NE-corruption
             * guard). Clip to [0.001, 1.0] (NOT 10.0 — Python physical cap). */
            if (far_active) {
                float raw_dt_ratio = raw_err_pwr_dt / (far_power + 1e-10f);
                float inst_erl_raw = mic_pwr / (far_power + 1e-10f);
                if (raw_dt_ratio < 2.0f && inst_erl_raw < 1.5f) {
                    float inst_erl = inst_erl_raw;
                    if (inst_erl < 0.001f) inst_erl = 0.001f;
                    if (inst_erl > 1.0f)   inst_erl = 1.0f;
                    float alpha_erl = aec->filter_converged ? 0.999f : 0.99f;
                    aec->erl_estimate = alpha_erl * aec->erl_estimate
                                        + (1.0f - alpha_erl) * inst_erl;
                }
            }

            /* Compute divergence for RES (matching Python: only after convergence) */
            if (aec->filter_converged && aec->near_power > 1e-8f) {
                float inst_erle_linear = aec->near_power / (aec->raw_error_power + 1e-10f);
                float is_diverged = (inst_erle_linear < 0.63f) ? 1.0f : 0.0f;
                aec->divergence_indicator = 0.9f * aec->divergence_indicator + 0.1f * is_diverged;
            } else {
                aec->divergence_indicator *= 0.95f;
            }
            float res_divergence = aec->divergence_indicator;

            /* === I: Combined shadow_dt for RES === */
            float shadow_dt_signal = aec->epc_active ? 0.0f
                : maxf(aec->dt_from_energy, aec->dt_from_shadow);

            /* === J: EPC render-forced countdown (G1) === */
            if (aec->epc_render_forced_remaining > 0) {
                aec->epc_render_forced_remaining--;
                res_force_render_based(aec->res);
            }

            /* v2.1.0: pass shadow_dt separately — RES merges via effective_dt */
            res_process(aec->res, aec->raw_output,
                        pbfdkf_get_echo_spec(aec->filter),
                        pbfdkf_get_far_spec(aec->filter),
                        pbfdkf_get_near_spec(aec->filter),
                        far_power, aec->filter_converged,
                        erle_factor, dt_indicator, over_sub,
                        res_divergence, aec->is_stationary_dt,
                        shadow_dt_signal, aec->erl_estimate,
                        aec->epc_active, aec->saturation_level,
                        output);

            /* Update per-bin mu_scale from RES PSDs */
            if (aec->filter_converged) {
                const float* echo_psd = res_get_echo_psd(aec->res);
                const float* error_psd = res_get_error_psd(aec->res);
                if (echo_psd && error_psd) {
                    float mean_eer = 0.0f;
                    int nf = aec->n_freqs;
                    float mu_min = cfg->shadow_mu_min;
                    /* FS-parity fix: populate per-bin mu_scale (matches Python
                     * aec.py:2895-2898). Post-convergence each freq bin gets its
                     * own step size based on residual EER, preventing the 11%
                     * echo_psd divergence that compounded into FS gain bias. */
                    if (!aec->per_bin_mu_scale) {
                        aec->per_bin_mu_scale = (float*)calloc(nf, sizeof(float));
                    }
                    for (int k = 0; k < nf; k++) {
                        float eer_k = clampf(echo_psd[k] / (error_psd[k] + 1e-10f), 0.0f, 1.0f);
                        mean_eer += eer_k;
                        if (aec->per_bin_mu_scale)
                            aec->per_bin_mu_scale[k] = mu_min + (1.0f - mu_min) * eer_k;
                    }
                    aec->simple_mu_ratio = mean_eer / nf;
                    /* Stationary DT: freeze filter at mu_min */
                    if (aec->is_stationary_dt) {
                        aec->simple_mu_ratio = mu_min;
                        if (aec->per_bin_mu_scale) {
                            for (int k = 0; k < nf; k++)
                                aec->per_bin_mu_scale[k] = mu_min;
                        }
                    }
                }
            } else {
                /* Pre-convergence: disable per-bin mu, use scalar from simple ratio */
                if (aec->per_bin_mu_scale) {
                    free(aec->per_bin_mu_scale);
                    aec->per_bin_mu_scale = NULL;
                }
                update_simple_mu_ratio(aec, aec->raw_output, ref, hop);
            }
        } /* end E block */
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

    /* === Track far-end activity for warmup gating === */
    {
        float far_pwr_warmup = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_warmup += ref[i] * ref[i];
        aec->warmup_far_active = (far_pwr_warmup / hop > 1e-6f);
    }

    /* === Convergence detection: 10 consecutive far-active frames ERLE > 5dB === */
    /* Gate on far_active (not _simple_mu_ratio) to avoid deadlock in
     * high-coupling FS where mic ≈ far × strong_coupling pulls
     * _simple_mu_ratio < 0.5 forever, blocking convergence.
     * ERLE > 5 dB sustained 10 frames is essentially impossible during
     * real DT, so we don't need an extra DT exclusion gate.
     * (Matches Python aec.py line 3016-3029.) */
    /* Skip during warmup — let filter learn with Q_high before judging convergence */
    if (!aec->filter_converged && aec->near_power > 1e-8f && aec->warmup_frames <= 0) {
        float inst_erle = 10.0f * fast_log10(aec->near_power /
                                          (aec->raw_error_power + 1e-10f));
        float far_pwr_conv = 0.0f;
        for (int i = 0; i < hop; i++) far_pwr_conv += ref[i] * ref[i];
        int far_active_conv = (far_pwr_conv / hop > 1e-4f);
        if (far_active_conv) {
            if (inst_erle > 5.0f)
                aec->conv_counter++;
            else
                aec->conv_counter = 0;

            if (aec->conv_counter >= 10) {
                aec->filter_converged = 1;
                pbfdkf_switch_q_low(aec->filter);
                pbfdkf_switch_q_low(aec->shadow_filter);
            }
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

struct ResFilter* aec_get_res(const Aec* aec) {
    return aec ? aec->res : NULL;
}

int aec_get_warmup_frames(const Aec* aec) {
    return aec ? aec->warmup_frames : 0;
}

float aec_get_simple_mu_ratio(const Aec* aec) {
    return aec ? aec->simple_mu_ratio : 0.0f;
}

struct Pbfdkf* aec_get_filter(const Aec* aec) {
    return aec ? aec->filter : NULL;
}

int aec_get_delay(const Aec* aec) {
    if (!aec || !aec->delay_est.enabled) return -1;
    return aec->delay_est.current_delay;
}

/* ============================================================
 * Filter training API — TODO stubs
 * ============================================================
 *
 * Future implementation plan:
 *   - AecTrainer: lightweight wrapper around PBFDKF with full mu,
 *     no shadow gating, no DTD, no RES.
 *   - aec_train_process(): runs adaptive filter on (mic, ref) pairs
 *     containing only echo (no near-end speech).
 *   - aec_train_get_filter(): exposes PBFDKF W array for serialization.
 *   - aec_load_filter(): injects pre-trained weights into runtime AEC,
 *     skipping cold-start convergence.
 */

struct AecTrainer {
    int placeholder;
};

AecTrainer* aec_train_create(const AecConfig* config) {
    (void)config;
    /* TODO: allocate trainer with full-mu PBFDKF */
    return NULL;
}

void aec_train_process(AecTrainer* trainer,
                       const float* mic,
                       const float* ref,
                       int n_samples) {
    (void)trainer; (void)mic; (void)ref; (void)n_samples;
    /* TODO: feed (mic, ref) into PBFDKF training loop */
}

const void* aec_train_get_filter(const AecTrainer* trainer) {
    (void)trainer;
    /* TODO: return pointer to PBFDKF W array */
    return NULL;
}

int aec_load_filter(Aec* aec, const void* weights) {
    (void)aec; (void)weights;
    /* TODO: copy weights into aec->filter->W */
    return -1;
}

void aec_train_destroy(AecTrainer* trainer) {
    (void)trainer;
    /* TODO: free trainer */
}
