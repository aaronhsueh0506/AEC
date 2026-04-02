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

    /* v2: reverb tail */
    int enable_reverb;
    float reverb_decay;
    float reverb_gain;
    float* reverb_psd;    /* [n_freqs] */

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

    r->window         = (float*)malloc(fs * sizeof(float));
    r->input_buf      = (float*)calloc(fs, sizeof(float));
    r->ola_buf        = (float*)calloc(fs, sizeof(float));
    r->time_buf       = (float*)calloc(bs, sizeof(float));
    r->spec_buf       = (Complex*)calloc(nf, sizeof(Complex));

    if (!r->echo_psd || !r->error_psd || !r->gain_smooth ||
        !r->error_envelope || !r->S_fe_r || !r->S_fe_i ||
        !r->S_ff || !r->S_ee || !r->coh2_smooth || !r->noise_psd ||
        !r->erle_per_bin || !r->near_psd_buf || !r->near_psd ||
        !r->reverb_psd ||
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
    }
    memset(r->S_fe_r, 0, nf * sizeof(float));
    memset(r->S_fe_i, 0, nf * sizeof(float));
    memset(r->S_ff, 0, nf * sizeof(float));
    memset(r->S_ee, 0, nf * sizeof(float));
    memset(r->coh2_smooth, 0, nf * sizeof(float));
    memset(r->noise_psd, 0, nf * sizeof(float));
    memset(r->near_psd_buf, 0, NEAR_PSD_BLOCKS * nf * sizeof(float));
    memset(r->near_psd, 0, nf * sizeof(float));
    memset(r->reverb_psd, 0, nf * sizeof(float));
    r->near_psd_idx = 0;
    memset(r->input_buf, 0, r->frame_size * sizeof(float));
    memset(r->ola_buf, 0, r->frame_size * sizeof(float));
    r->far_activity = 0.0f;
}

void res_process(ResFilter* r,
                 const float* error_hop,
                 const Complex* echo_spec,
                 const Complex* far_spec,
                 const Complex* near_spec,
                 float far_power,
                 int converged,
                 float erle_factor,
                 float dt_indicator,
                 float over_sub,
                 float* output_hop) {
    if (!r || !error_hop || !output_hop) return;

    const int nf = r->n_freqs;
    const int hop = r->hop_size;
    const int fs = r->frame_size;
    const int bs = r->block_size;
    const float eps = 1e-10f;

    /* === Analysis: slide input_buf, window, zero-pad, FFT === */
    memmove(r->input_buf, r->input_buf + hop, (fs - hop) * sizeof(float));
    memcpy(r->input_buf + fs - hop, error_hop, hop * sizeof(float));

    /* Windowed + zero-pad */
    for (int i = 0; i < fs; i++)
        r->time_buf[i] = r->input_buf[i] * r->window[i];
    memset(r->time_buf + fs, 0, (bs - fs) * sizeof(float));

    fft_forward(r->fft, r->time_buf, r->spec_buf);

    /* Power spectra */
    float error_pwr[257]; /* stack alloc, nf <= 257 */
    for (int k = 0; k < nf; k++) {
        error_pwr[k] = r->spec_buf[k].r * r->spec_buf[k].r +
                        r->spec_buf[k].i * r->spec_buf[k].i;
    }

    float echo_pwr[257];
    if (echo_spec) {
        for (int k = 0; k < nf; k++) {
            echo_pwr[k] = echo_spec[k].r * echo_spec[k].r +
                          echo_spec[k].i * echo_spec[k].i;
        }
    } else {
        memset(echo_pwr, 0, nf * sizeof(float));
    }

    /* === Coherence-based echo PSD estimation === */
    float coh2[257];
    memset(coh2, 0, nf * sizeof(float));

    if (far_spec && far_power > 1e-4f) {
        float a = r->alpha_coh;
        for (int k = 0; k < nf; k++) {
            /* Cross-PSD: spec × conj(far_spec) */
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

            /* Asymmetric EMA */
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
            }
        }
    }

    /* Cold start: init PSD on first far-end frame */
    {
        float sum_echo = 0.0f;
        for (int k = 0; k < nf; k++) sum_echo += r->echo_psd[k];
        if (far_power > 1e-4f && sum_echo < 1e-10f) {
            memcpy(r->echo_psd, echo_pwr, nf * sizeof(float));
            memcpy(r->error_psd, error_pwr, nf * sizeof(float));
        }
    }

    /* Smooth PSD */
    for (int k = 0; k < nf; k++) {
        r->echo_psd[k]  = r->alpha_echo_psd * r->echo_psd[k] +
                           (1.0f - r->alpha_echo_psd) * echo_pwr[k];
        r->error_psd[k] = r->alpha_error_psd * r->error_psd[k] +
                           (1.0f - r->alpha_error_psd) * error_pwr[k];
    }

    if (far_power < 1e-4f) {
        for (int k = 0; k < nf; k++) r->echo_psd[k] *= 0.3f;
        /* Track noise for CNG: only when far-end silent for ~800ms */
        if (r->enable_cng && r->far_activity < 0.1f) {
            for (int k = 0; k < nf; k++) {
                float updated = r->alpha_noise * r->noise_psd[k] +
                                (1.0f - r->alpha_noise) * error_pwr[k];
                float ceil_val = error_pwr[k] * 2.0f;
                r->noise_psd[k] = updated < ceil_val ? updated : ceil_val;
            }
        }
    }

    /* === Residual echo PSD estimation (v2: direct method) === */
    float residual_echo_psd[257];
    int have_residual = 0;

    if (r->echo_method == RES_ECHO_DIRECT && near_spec) {
        /* Near-end power */
        float near_pwr[257];
        for (int k = 0; k < nf; k++) {
            near_pwr[k] = near_spec[k].r * near_spec[k].r +
                          near_spec[k].i * near_spec[k].i;
        }

        /* 4-block ring buffer moving average for stable near-end PSD */
        float* ring_slot = r->near_psd_buf + r->near_psd_idx * nf;
        memcpy(ring_slot, near_pwr, nf * sizeof(float));
        r->near_psd_idx = (r->near_psd_idx + 1) % NEAR_PSD_BLOCKS;

        for (int k = 0; k < nf; k++) {
            float sum = 0.0f;
            for (int b = 0; b < NEAR_PSD_BLOCKS; b++)
                sum += r->near_psd_buf[b * nf + k];
            r->near_psd[k] = sum * (1.0f / NEAR_PSD_BLOCKS);
        }

        /* Update per-bin ERLE: only when far active, no DT, converged */
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

        /* Residual echo PSD: blend based on convergence */
        for (int k = 0; k < nf; k++) {
            float erle_est = r->echo_psd[k] / r->erle_per_bin[k];
            float direct_est = r->echo_psd[k];

            /* Pre-convergence echo floor: coh2-weighted error_psd */
            if (erle_factor < 0.5f && far_power > 1e-4f) {
                float ef_fade = 1.0f - erle_factor * 2.0f;
                float echo_floor = r->error_psd[k] * coh2[k] * ef_fade
                                   * r->far_activity;
                if (direct_est < echo_floor) direct_est = echo_floor;
            }

            residual_echo_psd[k] = (1.0f - erle_factor) * direct_est
                                   + erle_factor * erle_est;
        }

        /* Reverb tail: use far_spec power (render signal) */
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
                residual_echo_psd[k] += r->reverb_gain * r->reverb_psd[k]
                                        * r->far_activity;
            }
        }

        /* Echo boost: high-coh2 bins → boost residual estimate */
        if (far_power > 1e-4f && erle_factor > 0.3f) {
            for (int k = 0; k < nf; k++) {
                float echo_boost = (dt_indicator < 0.2f)
                                   ? (1.0f + 0.5f * coh2[k]) : 1.0f;
                residual_echo_psd[k] *= echo_boost;
            }
        }

        have_residual = 1;
    }

    /* === Dynamic g_min: far-end activity tracking === */
    float is_far_active = (far_power > 1e-4f) ? 1.0f : 0.0f;
    if (is_far_active > r->far_activity) {
        r->far_activity = 0.7f * r->far_activity + 0.3f * is_far_active;
    } else {
        r->far_activity = 0.98f * r->far_activity + 0.02f * is_far_active;
    }
    /* Fixed g_min: gain floor is constant (AEC3 style — purely ENR-driven) */
    float eff_g_min = r->g_min;

    /* fs_confidence: continuous FS/DT/NE indicator */
    float dt_inv = 1.0f - dt_indicator;
    float fs_confidence = r->far_activity * dt_inv * dt_inv;

    /* === Noise gate: quiet bins pass through === */
    float mean_error = 0.0f;
    for (int k = 0; k < nf; k++) mean_error += r->error_psd[k];
    mean_error /= nf;
    float signal_floor = mean_error * 0.001f + 1e-8f;

    /* === EER computation (for legacy + near-end gate) === */
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

    /* === Spectral floor + near-end gate === */
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

    /* Per-bin near-end gate with fs_confidence */
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

    /* === Gain computation (v2: three branches) === */
    float g[257];

    if (r->gain_type == RES_GAIN_ENR && have_residual) {
        /* ENR with dt_indicator × nearend_est (v2.0.0) */
        float scale = r->enr_scale;
        /* ne_confidence directly from dt_indicator (ERLE-corrected at source) */
        float ne_confidence = dt_indicator;

        /* Compute noise floor for nearend_est */
        float error_mean = 0.0f;
        for (int k = 0; k < nf; k++) error_mean += r->error_psd[k];
        error_mean /= nf;
        float noise_floor_psd = error_mean * 0.01f + 1e-10f;

        for (int k = 0; k < nf; k++) {
            /* dt × nearend_est: FS(dt≈0) → ENR>>10, DT(dt≈0.5) → ENR low */
            float raw_ne = maxf(r->error_psd[k] - residual_echo_psd[k], 0.0f);
            float nearend_est = maxf(raw_ne * dt_indicator, noise_floor_psd);
            float enr = residual_echo_psd[k] / nearend_est;

            float blend = clampf((k - 5.0f) / 5.0f, 0.0f, 1.0f);

            float enr_t_ne = (1.0f - blend) * 2.0f + blend * 0.5f;
            float enr_s_ne = (1.0f - blend) * 3.0f + blend * 1.0f;
            float enr_t_fs = (1.0f - blend) * 0.3f * scale + blend * 0.07f * scale;
            float enr_s_fs = (1.0f - blend) * 0.4f * scale + blend * 0.1f * scale;

            float enr_t = ne_confidence * enr_t_ne + (1.0f - ne_confidence) * enr_t_fs;
            float enr_s = ne_confidence * enr_s_ne + (1.0f - ne_confidence) * enr_s_fs;

            if (enr > enr_t) {
                g[k] = clampf((enr_s - enr) / (enr_s - enr_t + eps), 0.0f, 1.0f);
            } else {
                g[k] = 1.0f;
            }
            g[k] = maxf(g[k], spectral_g_min[k]);
        }

        /* EMR noise masking: if echo below noise, don't suppress */
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
    } else if (r->gain_type == RES_GAIN_WIENER && have_residual) {
        /* Wiener gain */
        float noise_floor_psd = mean_error * 0.01f + eps;
        for (int k = 0; k < nf; k++) {
            float nearend_est = maxf(r->error_psd[k] - residual_echo_psd[k],
                                     noise_floor_psd);
            float beta = over_sub;
            g[k] = nearend_est / (nearend_est + beta * residual_echo_psd[k] + eps);
            g[k] = maxf(g[k], spectral_g_min[k]);
        }
    } else if (have_residual) {
        /* Spectral subtraction with direct residual echo PSD */
        for (int k = 0; k < nf; k++) {
            float eer_direct = residual_echo_psd[k] / (r->error_psd[k] + eps);
            g[k] = maxf(1.0f - over_sub * eer_direct, spectral_g_min[k]);
        }
    } else {
        /* Legacy coherence-based spectral subtraction */
        for (int k = 0; k < nf; k++) {
            g[k] = maxf(1.0f - over_sub * eer[k], spectral_g_min[k]);
        }
    }

    /* Noise gate: pass through quiet bins */
    for (int k = 0; k < nf; k++) {
        if (r->echo_psd[k] < signal_floor && r->error_psd[k] < signal_floor)
            g[k] = 1.0f;
    }

    /* === Cross-frequency smoothing (3-bin convolution) === */
    if (far_power > 1e-4f) {
        float prev = g[0];
        for (int k = 1; k < nf - 1; k++) {
            float cur = g[k];
            g[k] = 0.25f * prev + 0.5f * cur + 0.25f * g[k+1];
            prev = cur;
        }
    }

    /* === Frequency-domain postprocessing (cf. AEC3 PostprocessGains) === */
    if (far_power > 1e-4f) {
        /* DC consistency: bins 0-1 follow bin 2 */
        if (nf > 2) {
            float dc_g = minf(g[1], g[2]);
            g[0] = dc_g;
            g[1] = dc_g;
        }
        /* HF cap: upper bins capped at ~2kHz gain (bin 16 for 512-FFT) */
        int hf_cap_bin = 16 < (nf - 1) ? 16 : (nf - 1);
        if (nf > hf_cap_bin + 1) {
            float hf_cap = g[hf_cap_bin];
            for (int k = hf_cap_bin + 1; k < nf; k++)
                g[k] = minf(g[k], hf_cap);
        }
    }

    /* === Temporal smoothing === */
    float alpha_release = 0.4f + 0.5f * r->far_activity;
    /* Continuous attack blend: FS→fast, DT/NE→slow */
    float alpha_fast = 0.3f + 0.2f * (1.0f - erle_factor);
    float alpha_slow = 0.85f + 0.1f * (1.0f - erle_factor);
    float alpha_attack = alpha_slow + (alpha_fast - alpha_slow) * fs_confidence;

    /* Pre-compute rate limiting constants outside per-bin loop */
    float act_scale = 0.5f + 0.5f * r->far_activity;
    float eff_drop = fast_exp(act_scale * fast_log(r->max_drop_ratio));
    float eff_rise = fast_exp((0.5f + 0.5f * (1.0f - r->far_activity))
                              * fast_log(r->max_rise_ratio));
    /* LF gain decrease limiting */
    int lf_limit = 8 < nf ? 8 : nf;
    float lf_factor = 0.0f;
    if (fs_confidence < 0.9f) {
        lf_factor = 0.25f * maxf(1.0f - fs_confidence * 2.0f, 0.0f);
    }

    for (int k = 0; k < nf; k++) {
        float alpha_g = (g[k] < r->gain_smooth[k]) ? alpha_attack : alpha_release;
        float smoothed = alpha_g * r->gain_smooth[k] + (1.0f - alpha_g) * g[k];

        /* Rate limiting */
        float gain_floor = r->gain_smooth[k] / eff_drop;
        float gain_ceil  = r->gain_smooth[k] * eff_rise;

        /* LF gain decrease limiting: protect low frequencies during DT/NE */
        if (k < lf_limit && lf_factor > 0.01f) {
            float lf_floor = r->gain_smooth[k] * lf_factor;
            gain_floor = maxf(gain_floor, lf_floor);
        }

        smoothed = maxf(smoothed, gain_floor);
        smoothed = minf(smoothed, gain_ceil);
        smoothed = maxf(smoothed, spectral_g_min[k]);
        smoothed = minf(smoothed, 1.0f);

        r->gain_smooth[k] = smoothed;
    }

    /* === Apply gain to spectrum === */
    for (int k = 0; k < nf; k++) {
        r->spec_buf[k].r *= r->gain_smooth[k];
        r->spec_buf[k].i *= r->gain_smooth[k];
    }

    /* === CNG: complex Gaussian comfort noise === */
    if (r->enable_cng) {
        float noise_sum = 0.0f;
        for (int k = 0; k < nf; k++) noise_sum += r->noise_psd[k];
        if (noise_sum > 0.0f) {
            for (int k = 0; k < nf; k++) {
                float cn_gain_k = fast_sqrt(maxf(1.0f - r->gain_smooth[k] * r->gain_smooth[k], 0.0f));
                float noise_std = fast_sqrt(r->noise_psd[k] / 2.0f + 1e-10f);
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

    /* === Synthesis: IFFT → truncate → sqrt-Hann → OLA === */
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
