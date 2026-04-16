/**
 * res_filter.c - Residual Echo Suppressor with WOLA (v2)
 *
 * Matches Python ResFilter v1.28.1:
 * - WOLA analysis/synthesis with sqrt-Hann window
 * - Coherence-based + direct echo PSD estimation
 * - ENR masking gain (default) + Wiener + spectral subtraction
 * - Per-bin ERLE tracking, reverb tail model
 * - Frequency postprocessing (DC consistency, HF cap)
 * - Per-bin near-end gate, dynamic g_min, spectral floor, rate limiting, CNG
 */

#include "res_filter.h"
#include "fast_math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* dB to amplitude: 10^(db/20) = exp(db * ln(10)/20) */
static inline float db_to_amp(float db) {
#ifdef USE_STANDARD_MATH
    return powf(10.0f, db / 20.0f);
#else
    return fast_exp(db * (FM_LN10 / 20.0f));
#endif
}

#define NEAR_PSD_BLOCKS 4   /* ring buffer size for near-end PSD averaging */

struct ResFilter {
    int block_size;       /* 512 */
    int frame_size;       /* 320 */
    int hop_size;         /* 160 */
    int n_freqs;          /* 257 */

    float g_min;          /* linear minimum gain */
    float ne_protect_db;

    /* Rate limiting */
    float max_drop_ratio;
    float max_rise_ratio;

    /* Spectral floor */
    float spectral_floor_ratio;
    float* error_envelope; /* [n_freqs] */
    float alpha_envelope;  /* 0.95 */

    /* PSD smoothing (configurable per-preset) */
    float alpha_echo_psd;
    float alpha_error_psd;
    float* echo_psd;      /* [n_freqs] */
    float* error_psd;     /* [n_freqs] */
    float* gain_smooth;   /* [n_freqs] */

    /* Coherence */
    float alpha_coh;      /* 0.65 */
    float* S_fe_r;        /* [n_freqs] cross-PSD real */
    float* S_fe_i;        /* [n_freqs] cross-PSD imag */
    float* S_ff;          /* [n_freqs] far PSD */
    float* S_ee;          /* [n_freqs] error PSD */
    float* coh2_smooth;   /* [n_freqs] smoothed coherence² */

    /* Far-end activity */
    float far_activity;

    /* CNG */
    int enable_cng;
    int noise_initialized; /* 0 until first CNG frame */
    float* noise_psd;     /* [n_freqs] */
    float alpha_noise;    /* 0.98 */

    /* === v2: direct echo estimation === */
    int echo_method;      /* RES_ECHO_COHERENCE or RES_ECHO_DIRECT */
    int gain_type;        /* RES_GAIN_SPECTRAL_SUB / _WIENER / _ENR */
    float enr_scale;
    float* erle_per_bin;  /* [n_freqs] per-bin ERLE estimate */
    float alpha_erle;     /* 0.95 */
    float* near_psd_buf;  /* [NEAR_PSD_BLOCKS * n_freqs] ring buffer */
    int near_psd_idx;
    float* near_psd;      /* [n_freqs] 4-block average */

    /* Multi-ERLE (v2.0.0) */
    float* filter_erle;   /* [n_freqs] per-bin ERLE from echo_spec²/error_spec² */
    float  fb_erle;       /* fullband ERLE for cross-validation */

    /* v2: reverb tail */
    int enable_reverb;
    float reverb_decay;
    float reverb_gain;
    float* reverb_psd;    /* [n_freqs] */

    /* v2.1.0: AEC3-style echo switching */
    int using_render_based;
    int render_hold;

    /* WOLA */
    float* window;        /* [frame_size] sqrt-Hann */
    float* input_buf;     /* [frame_size] analysis sliding buffer */
    float* ola_buf;       /* [frame_size] synthesis OLA buffer */

    /* FFT */
    FftHandle* fft;
    float* time_buf;      /* [block_size] scratch */
    Complex* spec_buf;    /* [n_freqs] scratch */
};

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

ResFilter* res_create(const ResConfig* cfg) {
    if (!cfg || cfg->n_freqs <= 0) return NULL;

    ResFilter* r = (ResFilter*)calloc(1, sizeof(ResFilter));
    if (!r) return NULL;

    r->block_size = cfg->block_size;
    r->frame_size = cfg->frame_size;
    r->hop_size = cfg->hop_size;
    r->n_freqs = cfg->n_freqs;
    r->g_min = db_to_amp(cfg->g_min_db);
    r->ne_protect_db = cfg->ne_protect_db;

    r->max_drop_ratio = db_to_amp(cfg->max_drop_db_per_frame);
    r->max_rise_ratio = db_to_amp(cfg->max_rise_db_per_frame);
    r->spectral_floor_ratio = db_to_amp(cfg->spectral_floor_db);

    r->alpha_echo_psd = cfg->alpha_echo_psd;
    r->alpha_error_psd = cfg->alpha_error_psd;
    r->alpha_envelope = 0.95f;
    r->alpha_coh = 0.65f;
    r->alpha_noise = 0.98f;
    r->far_activity = 0.0f;
    r->enable_cng = cfg->enable_cng;

    /* v2 config */
    r->echo_method = cfg->echo_method;
    r->gain_type = cfg->gain_type;
    r->enr_scale = cfg->enr_scale;
    r->enable_reverb = cfg->enable_reverb;
    r->reverb_decay = cfg->reverb_decay;
    r->reverb_gain = cfg->reverb_gain;
    r->alpha_erle = 0.95f;
    r->near_psd_idx = 0;

    int nf = cfg->n_freqs;
    int fs = cfg->frame_size;
    int bs = cfg->block_size;

    /* Allocate arrays */
    r->echo_psd       = (float*)calloc(nf, sizeof(float));
    r->error_psd      = (float*)calloc(nf, sizeof(float));
    r->gain_smooth    = (float*)malloc(nf * sizeof(float));
    r->error_envelope = (float*)malloc(nf * sizeof(float));
    r->S_fe_r         = (float*)calloc(nf, sizeof(float));
    r->S_fe_i         = (float*)calloc(nf, sizeof(float));
    r->S_ff           = (float*)calloc(nf, sizeof(float));
    r->S_ee           = (float*)calloc(nf, sizeof(float));
    r->coh2_smooth    = (float*)calloc(nf, sizeof(float));
    r->noise_psd      = (float*)calloc(nf, sizeof(float));

    /* v2 arrays */
    r->erle_per_bin   = (float*)malloc(nf * sizeof(float));
    r->near_psd_buf   = (float*)calloc(NEAR_PSD_BLOCKS * nf, sizeof(float));
    r->near_psd       = (float*)calloc(nf, sizeof(float));
    r->reverb_psd     = (float*)calloc(nf, sizeof(float));

    /* Multi-ERLE arrays */
    r->filter_erle    = (float*)malloc(nf * sizeof(float));
    r->fb_erle        = 1.0f;

    r->window         = (float*)malloc(fs * sizeof(float));
    r->input_buf      = (float*)calloc(fs, sizeof(float));
    r->ola_buf        = (float*)calloc(fs, sizeof(float));
    r->time_buf       = (float*)calloc(bs, sizeof(float));
    r->spec_buf       = (Complex*)calloc(nf, sizeof(Complex));

    if (!r->echo_psd || !r->error_psd || !r->gain_smooth ||
        !r->error_envelope || !r->S_fe_r || !r->S_fe_i ||
        !r->S_ff || !r->S_ee || !r->coh2_smooth || !r->noise_psd ||
        !r->erle_per_bin || !r->near_psd_buf || !r->near_psd ||
        !r->reverb_psd || !r->filter_erle ||
        !r->window || !r->input_buf || !r->ola_buf ||
        !r->time_buf || !r->spec_buf) {
        res_destroy(r);
        return NULL;
    }

    /* Init gains + v2 state */
    for (int k = 0; k < nf; k++) {
        r->gain_smooth[k] = r->g_min;
        r->error_envelope[k] = 1.0f;
        r->erle_per_bin[k] = 1.0f;
        r->filter_erle[k] = 1.0f;
    }

    /* Compute sqrt-Hann window */
    for (int i = 0; i < fs; i++) {
        float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / fs));
        r->window[i] = sqrtf(hann);
    }

    /* FFT handle */
    r->fft = fft_create(bs);
    if (!r->fft) {
        res_destroy(r);
        return NULL;
    }

    return r;
}

void res_destroy(ResFilter* r) {
    if (!r) return;
    free(r->echo_psd);
    free(r->error_psd);
    free(r->gain_smooth);
    free(r->error_envelope);
    free(r->S_fe_r);
    free(r->S_fe_i);
    free(r->S_ff);
    free(r->S_ee);
    free(r->coh2_smooth);
    free(r->noise_psd);
    free(r->erle_per_bin);
    free(r->filter_erle);
    free(r->near_psd_buf);
    free(r->near_psd);
    free(r->reverb_psd);
    free(r->window);
    free(r->input_buf);
    free(r->ola_buf);
    free(r->time_buf);
    free(r->spec_buf);
    fft_destroy(r->fft);
    free(r);
}

void res_reset(ResFilter* r) {
    if (!r) return;
    int nf = r->n_freqs;
    memset(r->echo_psd, 0, nf * sizeof(float));
    memset(r->error_psd, 0, nf * sizeof(float));
    for (int k = 0; k < nf; k++) {
        r->gain_smooth[k] = r->g_min;
        r->error_envelope[k] = 1.0f;
        r->erle_per_bin[k] = 1.0f;
        r->filter_erle[k] = 1.0f;
    }
    r->fb_erle = 1.0f;
    memset(r->S_fe_r, 0, nf * sizeof(float));
    memset(r->S_fe_i, 0, nf * sizeof(float));
    memset(r->S_ff, 0, nf * sizeof(float));
    memset(r->S_ee, 0, nf * sizeof(float));
    memset(r->coh2_smooth, 0, nf * sizeof(float));
    memset(r->noise_psd, 0, nf * sizeof(float));
    r->noise_initialized = 0;
    memset(r->near_psd_buf, 0, NEAR_PSD_BLOCKS * nf * sizeof(float));
    memset(r->near_psd, 0, nf * sizeof(float));
    memset(r->reverb_psd, 0, nf * sizeof(float));
    r->near_psd_idx = 0;
    memset(r->input_buf, 0, r->frame_size * sizeof(float));
    memset(r->ola_buf, 0, r->frame_size * sizeof(float));
    r->far_activity = 0.0f;
    r->using_render_based = 0;
    r->render_hold = 0;
}

void res_force_render_based(ResFilter* r) {
    if (r) r->using_render_based = 1;
}

/* === Multi-ERLE helper functions (v2.0.0) === */

static void update_filter_erle(ResFilter* r, const float* echo_pwr,
                                const float* error_pwr,
                                int far_active, float dt_indicator) {
    if (!far_active) return;
    int nf = r->n_freqs;
    for (int k = 0; k < nf; k++) {
        float inst = echo_pwr[k] / (error_pwr[k] + 1e-10f);
        if (inst < 0.1f) inst = 0.1f;
        if (inst > 1000.0f) inst = 1000.0f;
        float dt_weight = 1.0f - dt_indicator * 1.5f;
        if (dt_weight < 0.1f) dt_weight = 0.1f;
        float alpha = (inst < r->filter_erle[k])
            ? (0.7f * dt_weight + (1.0f - dt_weight) * 0.5f)
            : (0.95f * dt_weight);
        r->filter_erle[k] = alpha * r->filter_erle[k] + (1.0f - alpha) * inst;
    }
    /* 3-bin smoothing */
    float prev = r->filter_erle[0];
    for (int k = 1; k < nf - 1; k++) {
        float cur = r->filter_erle[k];
        r->filter_erle[k] = 0.25f * prev + 0.5f * cur + 0.25f * r->filter_erle[k+1];
        prev = cur;
    }
    /* Clip: frequency-dependent cap (LF 32, HF 16) aligned with Python */
    int lf_bins = nf > 8 ? 8 : nf;
    for (int k = 0; k < nf; k++) {
        if (r->filter_erle[k] < 0.5f) r->filter_erle[k] = 0.5f;
        float erle_max = (k < lf_bins) ? 32.0f : 16.0f;
        if (r->filter_erle[k] > erle_max) r->filter_erle[k] = erle_max;
    }
}

static void update_fb_erle(ResFilter* r, float near_power, float error_power,
                            int far_active, float dt_indicator) {
    if (!far_active || dt_indicator > 0.3f || near_power < 1e-8f) return;
    float inst = near_power / (error_power + 1e-10f);
    if (inst < 0.5f) inst = 0.5f;
    if (inst > 100.0f) inst = 100.0f;
    r->fb_erle = 0.97f * r->fb_erle + 0.03f * inst;
}

static float compute_erle_confidence(const float* filter_erle, int nf,
                                      float fb_erle) {
    float l1_mean = 0.0f;
    for (int k = 0; k < nf; k++) l1_mean += filter_erle[k];
    l1_mean /= nf;
    if (l1_mean < 0.5f || fb_erle < 0.5f) return 0.0f;
    float log_diff = fabsf(logf(l1_mean + 1e-10f) - logf(fb_erle + 1e-10f));
    return expf(-log_diff / 2.0f);
}

void res_process(ResFilter* r,
                 const float* error_hop,
                 const Complex* echo_spec,
                 const Complex* far_spec,
                 const Complex* near_spec,
                 float far_power,
                 int converged,
                 float erle_factor,
                 float dt_indicator_in,
                 float over_sub,
                 float divergence,
                 int is_stationary_dt,
                 float shadow_dt,
                 float erl_estimate,
                 int epc_active,
                 float saturation_level,
                 float* output_hop) {
    if (!r || !error_hop || !output_hop) return;

    /* Stationary DT virtual indicators (matches Python v13)
     * - dt_for_fs / dt_temporal = 0.8 only when stationary DT detected.
     * - For non-stationary scenes both equal dt_indicator_in (zero impact). */
    float dt_indicator = dt_indicator_in;
    float dt_for_fs   = is_stationary_dt ? 0.8f : dt_indicator_in;

    /* v2.1.0: three-line effective_dt drive — merge shadow_dt */
    float effective_dt = maxf(dt_for_fs, shadow_dt);

    /* v2.1.0: light release EMA — dt_temporal uses shadow_dt blend */
    float dt_temporal = is_stationary_dt ? 0.8f : maxf(dt_indicator_in, shadow_dt * 0.5f);

    const int nf = r->n_freqs;
    const int hop = r->hop_size;
    const int fs = r->frame_size;
    const int bs = r->block_size;
    const float eps = 1e-10f;

    /* === (a) WOLA analysis: slide input_buf, window, zero-pad, FFT === */
    memmove(r->input_buf, r->input_buf + hop, (fs - hop) * sizeof(float));
    memcpy(r->input_buf + fs - hop, error_hop, hop * sizeof(float));

    for (int i = 0; i < fs; i++)
        r->time_buf[i] = r->input_buf[i] * r->window[i];
    memset(r->time_buf + fs, 0, (bs - fs) * sizeof(float));

    fft_forward(r->fft, r->time_buf, r->spec_buf);

    /* Power spectra */
    float echo_pwr[257];
    float error_pwr[257];
    for (int k = 0; k < nf; k++) {
        error_pwr[k] = r->spec_buf[k].r * r->spec_buf[k].r +
                        r->spec_buf[k].i * r->spec_buf[k].i;
    }
    if (echo_spec) {
        for (int k = 0; k < nf; k++) {
            echo_pwr[k] = echo_spec[k].r * echo_spec[k].r +
                          echo_spec[k].i * echo_spec[k].i;
        }
    } else {
        memset(echo_pwr, 0, nf * sizeof(float));
    }

    /* === (b) Coherence tracking === */
    float coh2[257];
    memset(coh2, 0, nf * sizeof(float));

    if (far_spec && far_power > 1e-4f) {
        float a = r->alpha_coh;
        for (int k = 0; k < nf; k++) {
            float cr = r->spec_buf[k].r * far_spec[k].r +
                       r->spec_buf[k].i * far_spec[k].i;
            float ci = r->spec_buf[k].i * far_spec[k].r -
                       r->spec_buf[k].r * far_spec[k].i;
            float fp = far_spec[k].r * far_spec[k].r +
                       far_spec[k].i * far_spec[k].i;

            r->S_fe_r[k] = a * r->S_fe_r[k] + (1.0f - a) * cr;
            r->S_fe_i[k] = a * r->S_fe_i[k] + (1.0f - a) * ci;
            r->S_ff[k]   = a * r->S_ff[k]   + (1.0f - a) * fp;
            r->S_ee[k]   = a * r->S_ee[k]   + (1.0f - a) * error_pwr[k];

            float num = r->S_fe_r[k] * r->S_fe_r[k] +
                        r->S_fe_i[k] * r->S_fe_i[k];
            float den = r->S_ff[k] * r->S_ee[k] + 1e-10f;
            float c2_raw = num / den;
            if (c2_raw > 1.0f) c2_raw = 1.0f;

            /* Asymmetric EMA: fast drop / slow rise */
            float a_coh;
            if (converged) {
                a_coh = (c2_raw < r->coh2_smooth[k]) ? 0.50f : 0.90f;
            } else {
                a_coh = (c2_raw < r->coh2_smooth[k]) ? 0.50f : 0.80f;
            }
            r->coh2_smooth[k] = a_coh * r->coh2_smooth[k] +
                                 (1.0f - a_coh) * c2_raw;
            coh2[k] = r->coh2_smooth[k];
        }
    } else {
        if (far_power <= 1e-4f) {
            for (int k = 0; k < nf; k++) {
                r->S_fe_r[k] *= 0.5f;
                r->S_fe_i[k] *= 0.5f;
                r->S_ff[k]   *= 0.5f;
                r->S_ee[k]   *= 0.5f;  /* A1: sync decay */
            }
        }
    }

    /* === (c) Cold start: init PSD on first far-end frame === */
    {
        float sum_echo = 0.0f;
        for (int k = 0; k < nf; k++) sum_echo += r->echo_psd[k];
        if (far_power > 1e-4f && sum_echo < 1e-10f) {
            memcpy(r->echo_psd, echo_pwr, nf * sizeof(float));
            memcpy(r->error_psd, error_pwr, nf * sizeof(float));
        }
    }

    /* === (d) PSD smoothing === */
    for (int k = 0; k < nf; k++) {
        r->echo_psd[k]  = r->alpha_echo_psd * r->echo_psd[k] +
                           (1.0f - r->alpha_echo_psd) * echo_pwr[k];
        r->error_psd[k] = r->alpha_error_psd * r->error_psd[k] +
                           (1.0f - r->alpha_error_psd) * error_pwr[k];
    }

    /* === (e) Multi-ERLE update === */
    {
        int far_active_flag = (far_power > 1e-4f) ? 1 : 0;
        update_filter_erle(r, echo_pwr, error_pwr, far_active_flag, dt_indicator);

        float near_pwr_broad = 0.0f, error_pwr_broad = 0.0f;
        if (near_spec) {
            for (int k = 0; k < nf; k++)
                near_pwr_broad += near_spec[k].r * near_spec[k].r +
                                  near_spec[k].i * near_spec[k].i;
            near_pwr_broad /= nf;
        }
        for (int k = 0; k < nf; k++) error_pwr_broad += error_pwr[k];
        error_pwr_broad /= nf;
        update_fb_erle(r, near_pwr_broad, error_pwr_broad,
                       far_active_flag, dt_indicator);
    }

    /* === (f) Far-end silence handling: echo_psd decay === */
    if (far_power < 1e-4f) {
        for (int k = 0; k < nf; k++) r->echo_psd[k] *= 0.3f;
    }

    /* === (g) Dynamic g_min: far_activity tracking === */
    float is_far_active = (far_power > 1e-4f) ? 1.0f : 0.0f;
    if (is_far_active > r->far_activity) {
        r->far_activity = 0.7f * r->far_activity + 0.3f * is_far_active;
    } else {
        r->far_activity = 0.98f * r->far_activity + 0.02f * is_far_active;
    }
    float eff_g_min = r->g_min;

    /* === (h) Noise gate: quiet_mask === */
    float mean_error = 0.0f;
    for (int k = 0; k < nf; k++) mean_error += r->error_psd[k];
    mean_error /= nf;
    float signal_floor = mean_error * 0.001f + 1e-8f;

    /* === (i) Residual echo PSD estimation === */
    float residual_echo_psd[257];
    int have_residual = 0;

    if (r->echo_method == RES_ECHO_DIRECT && near_spec) {
        float near_pwr[257];
        for (int k = 0; k < nf; k++) {
            near_pwr[k] = near_spec[k].r * near_spec[k].r +
                          near_spec[k].i * near_spec[k].i;
        }

        /* 4-block ring buffer moving average */
        float* ring_slot = r->near_psd_buf + r->near_psd_idx * nf;
        memcpy(ring_slot, near_pwr, nf * sizeof(float));
        r->near_psd_idx = (r->near_psd_idx + 1) % NEAR_PSD_BLOCKS;

        for (int k = 0; k < nf; k++) {
            float sum = 0.0f;
            for (int b = 0; b < NEAR_PSD_BLOCKS; b++)
                sum += r->near_psd_buf[b * nf + k];
            r->near_psd[k] = sum * (1.0f / NEAR_PSD_BLOCKS);
        }

        /* Update per-bin ERLE */
        if (far_power > 1e-4f && dt_indicator < 0.3f && erle_factor > 0.3f) {
            float temp[257];
            for (int k = 0; k < nf; k++) {
                float inst = r->near_psd[k] / (r->error_psd[k] + eps);
                inst = clampf(inst, 1.0f, 1000.0f);
                r->erle_per_bin[k] = r->alpha_erle * r->erle_per_bin[k]
                                     + (1.0f - r->alpha_erle) * inst;
            }
            /* 3-bin frequency smoothing [0.25, 0.5, 0.25] */
            temp[0] = 0.75f * r->erle_per_bin[0] + 0.25f * r->erle_per_bin[1];
            for (int k = 1; k < nf - 1; k++)
                temp[k] = 0.25f * r->erle_per_bin[k-1] + 0.5f * r->erle_per_bin[k]
                          + 0.25f * r->erle_per_bin[k+1];
            temp[nf-1] = 0.25f * r->erle_per_bin[nf-2] + 0.75f * r->erle_per_bin[nf-1];
            /* Frequency-dependent ERLE cap: LF=32, HF=16 */
            {
                int lf_bins = 8 < nf ? 8 : nf;
                for (int k = 0; k < nf; k++) {
                    float erle_max = (k < lf_bins) ? 32.0f : 16.0f;
                    r->erle_per_bin[k] = clampf(temp[k], 1.0f, erle_max);
                }
            }
        }

        /* Multi-ERLE residual estimation */
        float confidence = compute_erle_confidence(r->filter_erle, nf, r->fb_erle);
        for (int k = 0; k < nf; k++) {
            float erle_corrected = confidence * r->filter_erle[k]
                                   + (1.0f - confidence) * 1.0f;
            if (erle_corrected < 0.5f) erle_corrected = 0.5f;
            float erle_est = r->echo_psd[k] / erle_corrected;
            float direct_est = r->echo_psd[k];

            /* Nonlinear echo floor (uses dt_for_fs to bypass on stationary DT) */
            if (far_power > 1e-4f) {
                float dt_weight = 1.0f - dt_for_fs;
                float nonlinear_floor = r->error_psd[k] * coh2[k]
                                        * r->far_activity * dt_weight;
                if (direct_est < nonlinear_floor) direct_est = nonlinear_floor;
                if (erle_est < nonlinear_floor) erle_est = nonlinear_floor;
            }

            residual_echo_psd[k] = (1.0f - erle_factor) * direct_est
                                   + erle_factor * erle_est;
            /* Residual cap at 2x echo_psd */
            float cap = r->echo_psd[k] * 2.0f;
            if (residual_echo_psd[k] > cap) residual_echo_psd[k] = cap;
        }

        /* === v2.1.0: AEC3-style render-based vs filter-based switching === */
        {
            float enr_avg = 0.0f;
            float error_psd_sum = 0.0f;
            for (int k = 0; k < nf; k++) {
                error_psd_sum += r->error_psd[k];
            }
            /* Python uses far_power / mean(error_psd), not echo/error ratio */
            enr_avg = far_power / (error_psd_sum / nf + 1e-10f);
            float switching_threshold = 0.5f * clampf(enr_avg / (enr_avg + 1.0f), 0.3f, 0.7f);

            float hysteresis = 0.05f;
            float eff_threshold = r->using_render_based
                ? switching_threshold + hysteresis : switching_threshold;

            /* G2: explicit overrides */
            int force_render = epc_active || (saturation_level > 0.5f);

            /* G5: minimum hold time */
            int want_render = (erle_factor < eff_threshold) || force_render;
            if (want_render && !r->using_render_based)
                r->render_hold = 5;
            if (r->using_render_based && r->render_hold > 0)
                r->render_hold--;
            int can_exit = !want_render && (r->render_hold == 0);
            r->using_render_based = want_render || (r->using_render_based && !can_exit);

            if (r->using_render_based && far_spec) {
                /* Render-based: far_psd × erl_estimate */
                for (int k = 0; k < nf; k++) {
                    float far_psd_k = far_spec[k].r * far_spec[k].r +
                                      far_spec[k].i * far_spec[k].i;
                    float render_echo = far_psd_k * erl_estimate;
                    float blend = clampf(1.0f - erle_factor / eff_threshold, 0.0f, 1.0f);
                    residual_echo_psd[k] = (1.0f - blend) * residual_echo_psd[k]
                                           + blend * render_echo;
                }
            }
        }

        /* === (j) Reverb tail ===
         * Skip on stationary DT — reverb_psd accumulates massive WN energy
         * that drowns near-end speech in residual_echo_psd. */
        if (r->enable_reverb) {
            for (int k = 0; k < nf; k++) {
                float far_psd_k;
                if (far_spec) {
                    far_psd_k = far_spec[k].r * far_spec[k].r +
                                far_spec[k].i * far_spec[k].i;
                } else {
                    far_psd_k = echo_pwr[k];
                }
                r->reverb_psd[k] = r->reverb_decay * r->reverb_psd[k]
                                   + (1.0f - r->reverb_decay) * far_psd_k;
                if (!is_stationary_dt) {
                    float ne_reverb_factor = 0.3f + 0.7f * r->far_activity
                                             * (1.0f - dt_for_fs);
                    float reverb_gate = r->far_activity * ne_reverb_factor;
                    residual_echo_psd[k] += r->reverb_gain * r->reverb_psd[k]
                                            * reverb_gate;
                }
            }
        }

        have_residual = 1;
    }

    /* === (k) Echo boost: high-coh2 bins → boost residual estimate === */
    if (far_power > 1e-4f && erle_factor > 0.3f && have_residual) {
        for (int k = 0; k < nf; k++) {
            float echo_boost = (dt_for_fs < 0.2f)
                               ? (1.0f + 0.5f * coh2[k]) : 1.0f;
            residual_echo_psd[k] *= echo_boost;
        }
    }

    /* EER computation (for legacy + near-end gate) */
    float eer[257];
    for (int k = 0; k < nf; k++) {
        float eer_linear = r->echo_psd[k] / (r->error_psd[k] + eps);
        float eer_conv = eer_linear * (0.5f + 0.5f * coh2[k]);
        if (far_power > 1e-4f) {
            eer[k] = (1.0f - erle_factor) * coh2[k] + erle_factor * eer_conv;
        } else {
            eer[k] = eer_conv;
        }
    }

    /* === (l) Spectral floor: error_envelope tracking, spectral_g_min === */
    float spectral_g_min[257];
    if (far_power > 1e-4f) {
        float env_max = 0.0f;
        for (int k = 0; k < nf; k++) {
            float err_mag = fast_sqrt(error_pwr[k] + 1e-10f);
            r->error_envelope[k] = r->alpha_envelope * r->error_envelope[k] +
                                    (1.0f - r->alpha_envelope) * err_mag;
            if (r->error_envelope[k] > env_max)
                env_max = r->error_envelope[k];
        }
        env_max += 1e-10f;

        for (int k = 0; k < nf; k++) {
            float env_norm = r->error_envelope[k] / env_max;
            spectral_g_min[k] = eff_g_min + (1.0f - eff_g_min) *
                                env_norm * r->spectral_floor_ratio;
            spectral_g_min[k] = maxf(spectral_g_min[k], eff_g_min);
        }
    } else {
        for (int k = 0; k < nf; k++) spectral_g_min[k] = eff_g_min;
    }

    /* fs_confidence: continuous FS/DT/NE indicator (uses effective_dt) */
    float dt_inv = 1.0f - effective_dt;
    float fs_confidence = r->far_activity * dt_inv * dt_inv;

    /* === (m) Per-bin near-end gate === */
    float ne_erle_gate;
    if (erle_factor < 0.3f) {
        ne_erle_gate = 0.3f;
    } else {
        ne_erle_gate = maxf(erle_factor, 0.2f);
    }
    float ne_g_min_ceil = db_to_amp(r->ne_protect_db);
    for (int k = 0; k < nf; k++) {
        float ne_protection = (1.0f - coh2[k]) * ne_erle_gate
                              * (1.0f - fs_confidence);
        float ne_g_floor = eff_g_min + (ne_g_min_ceil - eff_g_min) * ne_protection;
        ne_g_floor = maxf(ne_g_floor, eff_g_min);
        spectral_g_min[k] = maxf(spectral_g_min[k], ne_g_floor);
    }

    /* === Gain computation === */
    float g[257];

    if (r->gain_type == RES_GAIN_ENR && have_residual) {
        /* === (n) ENR computation === */
        float scale = r->enr_scale;
        float error_mean = 0.0f;
        for (int k = 0; k < nf; k++) error_mean += r->error_psd[k];
        error_mean /= nf;
        float noise_floor_psd = error_mean * 0.01f + 1e-10f;

        /* === Per-bin DT indicator with v3 frequency-shaped mask ===
         * dt_per_bin = max(dt_for_fs, 1-coh2)
         * + on stationary DT, raise speech band (300-3000Hz) to 0.8 */
        float dt_per_bin[257];
        for (int k = 0; k < nf; k++) {
            float v = 1.0f - coh2[k];
            dt_per_bin[k] = (effective_dt > v) ? effective_dt : v;
        }
        if (is_stationary_dt) {
            float freq_per_bin = 16000.0f / (float)bs;
            for (int k = 0; k < nf; k++) {
                float f = (float)k * freq_per_bin;
                float wn_mask = 0.0f;
                if (f >= 300.0f && f <= 3000.0f) {
                    wn_mask = 0.8f;
                } else if (f > 100.0f && f < 300.0f) {
                    wn_mask = 0.8f * (f - 100.0f) / 200.0f;
                } else if (f > 3000.0f && f < 4000.0f) {
                    wn_mask = 0.8f * (4000.0f - f) / 1000.0f;
                }
                if (wn_mask > dt_per_bin[k]) dt_per_bin[k] = wn_mask;
            }
        }

        for (int k = 0; k < nf; k++) {
            float dt_pb = dt_per_bin[k];
            float dt_pb_sq = dt_pb * dt_pb;
            float raw_ne = maxf(r->error_psd[k] - residual_echo_psd[k], 0.0f);
            float nearend_est = maxf(raw_ne * dt_pb_sq, noise_floor_psd);
            float min_ne_from_dt = r->error_psd[k] * dt_pb_sq;
            if (nearend_est < min_ne_from_dt) nearend_est = min_ne_from_dt;
            float ne_confidence = dt_pb;
            float enr = residual_echo_psd[k] / nearend_est;

            float blend = clampf((k - 5.0f) / 5.0f, 0.0f, 1.0f);

            float enr_t_ne = (1.0f - blend) * 2.0f + blend * 1.5f;
            float enr_s_ne = (1.0f - blend) * 3.0f + blend * 2.5f;
            float enr_t_fs = (1.0f - blend) * 0.3f * scale + blend * 0.07f * scale;
            float enr_s_fs = (1.0f - blend) * 0.4f * scale + blend * 0.1f * scale;

            float enr_t = ne_confidence * enr_t_ne + (1.0f - ne_confidence) * enr_t_fs;
            float enr_s = ne_confidence * enr_s_ne + (1.0f - ne_confidence) * enr_s_fs;

            /* === (p) Soft gate: linear interpolation === */
            float min_gate_width = 0.2f;
            if (enr_s < enr_t + min_gate_width) enr_s = enr_t + min_gate_width;
            if (enr > enr_t) {
                g[k] = clampf((enr_s - enr) / (enr_s - enr_t + eps), 0.0f, 1.0f);
            } else {
                g[k] = 1.0f;
            }
        }

        /* === (q) EMR noise masking === */
        {
            float noise_sum = 0.0f;
            for (int k = 0; k < nf; k++) noise_sum += r->noise_psd[k];
            if (noise_sum > 0.0f) {
                for (int k = 0; k < nf; k++) {
                    float emr = residual_echo_psd[k] / (r->noise_psd[k] + 1e-10f);
                    float g_emr = clampf(0.3f / (emr + 1e-10f), 0.0f, 1.0f);
                    g[k] = maxf(g[k], g_emr);
                }
            }
        }

        /* Apply spectral_g_min after EMR (matches Python order) */
        for (int k = 0; k < nf; k++) {
            g[k] = maxf(g[k], spectral_g_min[k]);
        }
    } else if (r->gain_type == RES_GAIN_WIENER && have_residual) {
        float noise_floor_psd = mean_error * 0.01f + eps;
        for (int k = 0; k < nf; k++) {
            float nearend_est = maxf(r->error_psd[k] - residual_echo_psd[k],
                                     noise_floor_psd);
            float beta = over_sub;
            g[k] = nearend_est / (nearend_est + beta * residual_echo_psd[k] + eps);
            g[k] = maxf(g[k], spectral_g_min[k]);
        }
    } else if (have_residual) {
        for (int k = 0; k < nf; k++) {
            float eer_direct = residual_echo_psd[k] / (r->error_psd[k] + eps);
            g[k] = maxf(1.0f - over_sub * eer_direct, spectral_g_min[k]);
        }
    } else {
        for (int k = 0; k < nf; k++) {
            g[k] = maxf(1.0f - over_sub * eer[k], spectral_g_min[k]);
        }
    }

    /* === (t) Quiet mask application === */
    for (int k = 0; k < nf; k++) {
        if (r->echo_psd[k] < signal_floor && r->error_psd[k] < signal_floor)
            g[k] = 1.0f;
    }

    /* === (r) Cross-frequency smoothing: 3-bin kernel [0.25, 0.5, 0.25] === */
    if (far_power > 1e-4f) {
        float temp[257];
        temp[0] = 0.75f * g[0] + 0.25f * g[1];
        for (int k = 1; k < nf - 1; k++)
            temp[k] = 0.25f * g[k-1] + 0.5f * g[k] + 0.25f * g[k+1];
        temp[nf-1] = 0.25f * g[nf-2] + 0.75f * g[nf-1];
        memcpy(g, temp, nf * sizeof(float));

        /* === (s) DC consistency: bins 0-1 follow bin 2 === */
        if (nf > 2) {
            float dc_g = minf(g[1], g[2]);
            g[0] = dc_g;
            g[1] = dc_g;
        }
        /* HF cap: upper bins capped at gain of bin near ~500Hz
         * Bypass during DT: speech harmonics above 500Hz must not be crushed */
        float freq_res = 16000.0f / (float)bs;
        int hf_cap_bin = (int)(500.0f / freq_res);
        if (hf_cap_bin > nf - 1) hf_cap_bin = nf - 1;
        if (nf > hf_cap_bin + 1 && effective_dt < 0.5f && !is_stationary_dt) {
            float hf_cap = g[hf_cap_bin];
            for (int k = hf_cap_bin + 1; k < nf; k++)
                g[k] = minf(g[k], hf_cap);
        }
    }

    /* === (u) Divergence override === */
    if (divergence > 0.3f) {
        float divergence_gain = 0.01f + (1.0f - 0.01f) * (1.0f - divergence);
        for (int k = 0; k < nf; k++)
            g[k] = minf(g[k], divergence_gain);
    }

    /* === (v) Gain temporal smoothing ===
     * v2.1.0: light release EMA driven by dt_temporal */
    float alpha_release = 0.5f - 0.2f * dt_temporal;
    float alpha_fast = 0.3f + 0.2f * (1.0f - erle_factor);
    float alpha_slow = 0.85f + 0.1f * (1.0f - erle_factor);
    float alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence;

    /* === (w) Gain rate limiting === */
    float act_scale = 0.5f + 0.5f * r->far_activity;
    float eff_drop = powf(r->max_drop_ratio, act_scale);
    float eff_rise = powf(r->max_rise_ratio, 0.5f + 0.5f * (1.0f - r->far_activity));
    /* DT rise boost only on stationary DT */
    if (is_stationary_dt) {
        float dt_cu = dt_temporal * dt_temporal * dt_temporal;
        eff_rise *= (1.0f + 19.0f * dt_cu);
    }

    /* LF protection */
    int lf_limit = 8 < nf ? 8 : nf;
    float lf_factor = 0.0f;
    if (fs_confidence < 0.9f) {
        lf_factor = 0.25f * maxf(1.0f - fs_confidence * 2.0f, 0.0f);
    }

    for (int k = 0; k < nf; k++) {
        float alpha_g = (g[k] < r->gain_smooth[k]) ? alpha_attack : alpha_release;
        float smoothed = alpha_g * r->gain_smooth[k] + (1.0f - alpha_g) * g[k];

        float gain_floor = r->gain_smooth[k] / eff_drop;
        float gain_ceil  = r->gain_smooth[k] * eff_rise;

        /* LF gain decrease limiting */
        if (k < lf_limit && lf_factor > 0.01f) {
            float lf_floor = r->gain_smooth[k] * lf_factor;
            gain_floor = maxf(gain_floor, lf_floor);
        }

        smoothed = maxf(smoothed, gain_floor);
        smoothed = minf(smoothed, gain_ceil);

        /* === (x) spectral_g_min clamping === */
        smoothed = maxf(smoothed, spectral_g_min[k]);
        smoothed = minf(smoothed, 1.0f);

        r->gain_smooth[k] = smoothed;
    }

    /* === (y) Apply gain: spec *= gain_smooth === */
    for (int k = 0; k < nf; k++) {
        r->spec_buf[k].r *= r->gain_smooth[k];
        r->spec_buf[k].i *= r->gain_smooth[k];
    }

    /* === (z) CNG: freeze-gated noise tracking + comfort noise injection === */
    if (r->enable_cng) {
        /* Noise PSD initialization on first frame */
        if (!r->noise_initialized) {
            for (int k = 0; k < nf; k++)
                r->noise_psd[k] = r->error_psd[k] + 1e-8f;
            r->noise_initialized = 1;
        }

        /* Freeze-gated minima tracking (uses current frame's gain_smooth) */
        float gain_mean = 0.0f;
        for (int k = 0; k < nf; k++) gain_mean += r->gain_smooth[k];
        gain_mean /= nf;
        int is_suppressing = (gain_mean < 0.9f);
        float alpha_down_cng = 0.90f;
        float alpha_up_cng = is_suppressing ? 1.0f : 0.998f;
        for (int k = 0; k < nf; k++) {
            float alpha_n = (r->error_psd[k] > r->noise_psd[k])
                            ? alpha_up_cng : alpha_down_cng;
            r->noise_psd[k] = alpha_n * r->noise_psd[k]
                               + (1.0f - alpha_n) * r->error_psd[k];
        }

        /* Comfort noise injection */
        float noise_sum = 0.0f;
        for (int k = 0; k < nf; k++) noise_sum += r->noise_psd[k];
        if (noise_sum > 1e-7f) {
            for (int k = 0; k < nf; k++) {
                float cn_gain_k = fast_sqrt(maxf(1.0f - r->gain_smooth[k] * r->gain_smooth[k], 0.0f)) * 0.3f;
                float noise_std = fast_sqrt(r->noise_psd[k] / 2.0f);
                /* Box-Muller for two N(0,1) samples */
                float u1 = (float)(rand() + 1) / ((float)RAND_MAX + 2.0f);
                float u2 = (float)rand() / ((float)RAND_MAX + 1.0f);
                float z0 = fast_sqrt(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
                float z1 = fast_sqrt(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
                r->spec_buf[k].r += cn_gain_k * noise_std * z0;
                r->spec_buf[k].i += cn_gain_k * noise_std * z1;
            }
        }
    }

    /* === (aa) WOLA synthesis: IFFT → truncate → sqrt-Hann → OLA === */
    fft_inverse(r->fft, r->spec_buf, r->time_buf);

    for (int i = 0; i < fs; i++)
        r->time_buf[i] *= r->window[i];

    for (int i = 0; i < fs; i++)
        r->ola_buf[i] += r->time_buf[i];

    memcpy(output_hop, r->ola_buf, hop * sizeof(float));
    memmove(r->ola_buf, r->ola_buf + hop, (fs - hop) * sizeof(float));
    memset(r->ola_buf + fs - hop, 0, hop * sizeof(float));
}

const float* res_get_echo_psd(const ResFilter* r) {
    return r ? r->echo_psd : NULL;
}

const float* res_get_error_psd(const ResFilter* r) {
    return r ? r->error_psd : NULL;
}
