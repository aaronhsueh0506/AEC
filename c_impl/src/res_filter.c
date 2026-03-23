/**
 * res_filter.c - Residual Echo Suppressor with WOLA
 *
 * Matches Python ResFilter v1.15.0:
 * - WOLA analysis/synthesis with sqrt-Hann window
 * - Coherence-based nonlinear echo PSD estimation
 * - Per-bin near-end gate (protects near-end during DT)
 * - Dynamic g_min, spectral floor, rate limiting, CNG
 */

#include "res_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    /* PSD smoothing */
    float alpha_echo_psd;  /* 0.7 */
    float alpha_error_psd; /* 0.8 */
    float* echo_psd;      /* [n_freqs] */
    float* error_psd;     /* [n_freqs] */
    float* gain_smooth;   /* [n_freqs] */

    /* Coherence */
    float alpha_coh;      /* 0.3 */
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
/* clampf reserved for future use */

ResFilter* res_create(const ResConfig* cfg) {
    if (!cfg || cfg->n_freqs <= 0) return NULL;

    ResFilter* r = (ResFilter*)calloc(1, sizeof(ResFilter));
    if (!r) return NULL;

    r->block_size = cfg->block_size;
    r->frame_size = cfg->frame_size;
    r->hop_size = cfg->hop_size;
    r->n_freqs = cfg->n_freqs;
    r->g_min = powf(10.0f, cfg->g_min_db / 20.0f);
    r->ne_protect_db = cfg->ne_protect_db;

    r->max_drop_ratio = powf(10.0f, cfg->max_drop_db_per_frame / 20.0f);
    r->max_rise_ratio = powf(10.0f, cfg->max_rise_db_per_frame / 20.0f);
    r->spectral_floor_ratio = powf(10.0f, cfg->spectral_floor_db / 20.0f);

    r->alpha_echo_psd = 0.7f;
    r->alpha_error_psd = 0.8f;
    r->alpha_envelope = 0.95f;
    r->alpha_coh = 0.3f;
    r->alpha_noise = 0.98f;
    r->far_activity = 0.0f;
    r->enable_cng = cfg->enable_cng;

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
    r->window         = (float*)malloc(fs * sizeof(float));
    r->input_buf      = (float*)calloc(fs, sizeof(float));
    r->ola_buf        = (float*)calloc(fs, sizeof(float));
    r->time_buf       = (float*)calloc(bs, sizeof(float));
    r->spec_buf       = (Complex*)calloc(nf, sizeof(Complex));

    if (!r->echo_psd || !r->error_psd || !r->gain_smooth ||
        !r->error_envelope || !r->S_fe_r || !r->S_fe_i ||
        !r->S_ff || !r->S_ee || !r->coh2_smooth || !r->noise_psd ||
        !r->window || !r->input_buf || !r->ola_buf ||
        !r->time_buf || !r->spec_buf) {
        res_destroy(r);
        return NULL;
    }

    /* Init gains */
    for (int k = 0; k < nf; k++) {
        r->gain_smooth[k] = r->g_min;
        r->error_envelope[k] = 1.0f;
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
    }
    memset(r->S_fe_r, 0, nf * sizeof(float));
    memset(r->S_fe_i, 0, nf * sizeof(float));
    memset(r->S_ff, 0, nf * sizeof(float));
    memset(r->S_ee, 0, nf * sizeof(float));
    memset(r->coh2_smooth, 0, nf * sizeof(float));
    memset(r->noise_psd, 0, nf * sizeof(float));
    memset(r->input_buf, 0, r->frame_size * sizeof(float));
    memset(r->ola_buf, 0, r->frame_size * sizeof(float));
    r->far_activity = 0.0f;
}

void res_process(ResFilter* r,
                 const float* error_hop,
                 const Complex* echo_spec,
                 const Complex* far_spec,
                 float far_power,
                 int converged,
                 float erle_factor,
                 float dt_indicator __attribute__((unused)),
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
        /* Track noise for CNG */
        if (r->enable_cng) {
            for (int k = 0; k < nf; k++) {
                float updated = r->alpha_noise * r->noise_psd[k] +
                                (1.0f - r->alpha_noise) * error_pwr[k];
                float ceil_val = error_pwr[k] * 2.0f;
                r->noise_psd[k] = updated < ceil_val ? updated : ceil_val;
            }
        }
    }

    /* === Dynamic g_min: far-end activity tracking === */
    float is_far_active = (far_power > 1e-4f) ? 1.0f : 0.0f;
    if (is_far_active > r->far_activity) {
        r->far_activity = 0.7f * r->far_activity + 0.3f * is_far_active;
    } else {
        r->far_activity = 0.98f * r->far_activity + 0.02f * is_far_active;
    }
    float eff_g_min = r->g_min + (1.0f - r->g_min) * (1.0f - r->far_activity);
    eff_g_min = maxf(eff_g_min, powf(10.0f, -60.0f / 20.0f));

    /* === Noise gate: quiet bins pass through === */
    float mean_error = 0.0f;
    for (int k = 0; k < nf; k++) mean_error += r->error_psd[k];
    mean_error /= nf;
    float signal_floor = mean_error * 0.001f + 1e-8f;

    /* === EER computation with convergence blending === */
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
        /* Update spectral envelope */
        float env_max = 0.0f;
        for (int k = 0; k < nf; k++) {
            float err_mag = sqrtf(error_pwr[k] + 1e-10f);
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

    /* Per-bin near-end gate */
    float ne_g_min_ceil = powf(10.0f, r->ne_protect_db / 20.0f);
    for (int k = 0; k < nf; k++) {
        float ne_protection = (1.0f - coh2[k]) * erle_factor;
        float ne_g_floor = eff_g_min + (ne_g_min_ceil - eff_g_min) * ne_protection;
        ne_g_floor = maxf(ne_g_floor, eff_g_min);
        spectral_g_min[k] = maxf(spectral_g_min[k], ne_g_floor);
    }

    /* === Compute gain === */
    float g[257];
    for (int k = 0; k < nf; k++) {
        g[k] = maxf(1.0f - over_sub * eer[k], spectral_g_min[k]);

        /* Noise gate: pass through quiet bins */
        if (r->echo_psd[k] < signal_floor && r->error_psd[k] < signal_floor) {
            g[k] = 1.0f;
        }
    }

    /* === Temporal smoothing === */
    float alpha_release = 0.4f + 0.5f * r->far_activity;
    float alpha_attack  = 0.60f + 0.25f * (1.0f - erle_factor);

    for (int k = 0; k < nf; k++) {
        float alpha_g = (g[k] < r->gain_smooth[k]) ? alpha_attack : alpha_release;
        float smoothed = alpha_g * r->gain_smooth[k] + (1.0f - alpha_g) * g[k];

        /* Rate limiting */
        float act_scale = 0.5f + 0.5f * r->far_activity;
        float eff_drop = powf(r->max_drop_ratio, act_scale);
        float eff_rise = powf(r->max_rise_ratio, 1.0f / act_scale);
        float gain_floor = r->gain_smooth[k] / eff_drop;
        float gain_ceil  = r->gain_smooth[k] * eff_rise;
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

    /* === CNG === */
    if (r->enable_cng && far_power <= 1e-4f) {
        float noise_sum = 0.0f;
        for (int k = 0; k < nf; k++) noise_sum += r->noise_psd[k];
        if (noise_sum > 0) {
            for (int k = 0; k < nf; k++) {
                float suppressed_pwr = r->gain_smooth[k] * r->gain_smooth[k] * error_pwr[k];
                float target_pwr = r->noise_psd[k] * 0.5f;
                float deficit = target_pwr - suppressed_pwr;
                if (deficit > 0) {
                    float cng_mag = sqrtf(deficit);
                    /* Simple pseudo-random phase using k as seed */
                    float phase = (float)(k * 2654435761u % 1000) / 1000.0f *
                                  2.0f * (float)M_PI;
                    r->spec_buf[k].r += cng_mag * cosf(phase);
                    r->spec_buf[k].i += cng_mag * sinf(phase);
                }
            }
        }
    }

    /* === Synthesis: IFFT → truncate → sqrt-Hann → OLA === */
    fft_inverse(r->fft, r->spec_buf, r->time_buf);

    /* Apply synthesis window (first frame_size samples only) */
    for (int i = 0; i < fs; i++)
        r->time_buf[i] *= r->window[i];

    /* OLA */
    for (int i = 0; i < fs; i++)
        r->ola_buf[i] += r->time_buf[i];

    /* Extract output hop */
    memcpy(output_hop, r->ola_buf, hop * sizeof(float));

    /* Shift OLA buffer */
    memmove(r->ola_buf, r->ola_buf + hop, (fs - hop) * sizeof(float));
    memset(r->ola_buf + fs - hop, 0, hop * sizeof(float));
}

const float* res_get_echo_psd(const ResFilter* r) {
    return r ? r->echo_psd : NULL;
}

const float* res_get_error_psd(const ResFilter* r) {
    return r ? r->error_psd : NULL;
}

void res_set_over_sub(ResFilter* r, float over_sub) {
    (void)r; (void)over_sub;
    /* over_sub is now passed per-frame to res_process */
}
