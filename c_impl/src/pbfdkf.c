/**
 * pbfdkf.c - Partitioned Block Frequency-Domain Kalman Filter
 *
 * Overlap-save with independent hop_size (160) and block_size (512).
 * FDKF Kalman adaptation (always on, no NLMS fallback).
 *
 * v1.17.0 fixes:
 * - P_init=0.01 (was 0.5) to prevent Kalman gain explosion
 * - P_MAX=0.02 clamping
 * - Q_low configurable (was hardcoded 1e-5)
 * - Per-bin Q gating on far-end activity
 * - Far-end energy gate for weight update (skip during silence)
 *
 * Algorithm:
 * 1. Shift buffer, insert hop_size new samples
 * 2. X[k] = FFT(far_buffer[block_size])
 * 3. Y_hat[k] = sum(W[p] * X_buf[p])
 * 4. echo_time = IFFT(Y_hat)
 * 5. output = near_buffer[-hop:] - echo_time[-hop:]  (valid region)
 * 6. error_time = [0...0, output]  (zero-pad for constraint)
 * 7. Kalman update: K, W, P per partition (with Q gating + P clamping)
 * 8. Time-domain constraint: w[hop_size:] = 0
 */

#include "pbfdkf.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define P_INIT  0.01f   /* Initial error covariance */
#define P_MAX   0.5f    /* Max P — let correct Kalman math converge naturally */

struct Pbfdkf {
    int block_size;       /* Input block size (2*hop = 320) */
    int hop_size;         /* Processing hop (160) */
    int fft_size;         /* FFT size = next pow2 >= block_size (512) */
    int n_partitions;
    int n_freqs;          /* fft_size/2 + 1 = 257 */
    float delta;
    float alpha_power;    /* 0.9 */

    FftHandle* fft;

    /* Filter weights [n_partitions][n_freqs] */
    Complex** W;

    /* Reference spectrum history [n_partitions][n_freqs] */
    Complex** X_buf;
    int partition_idx;

    /* FDKF Kalman state */
    float** P;            /* [n_partitions][n_freqs] error covariance */
    float* Q;             /* [n_freqs] current process noise */
    float* Q_high;        /* [n_freqs] fast convergence Q */
    float* Q_low;         /* [n_freqs] stable tracking Q */
    float* R;             /* [n_freqs] measurement noise PSD */
    float* error_psd;     /* [n_freqs] smoothed error PSD */
    float alpha_r;        /* R smoothing (0.95, aligned with Python) */
    float q_low_base;     /* base Q_low value for set_q_ratio */

    /* P-floor clamping (v2.1.0) */
    float p_floor_beta;       /* 0.1 normal, 1.0 during EPC */
    int   p_floor_beta_frames;/* countdown frames for EPC beta */

    /* v3.x: P_max override (matches Python _p_max_override / _p_max_override_frames).
     * EPC events transiently raise to 1.0 for fast re-convergence. */
    float p_max_override;       /* 0.5 default; 1.0 during EPC */
    int   p_max_override_frames;/* countdown frames for override */

    /* Input buffers [block_size] */
    float* near_buffer;
    float* far_buffer;

    /* Spectrum buffers [n_freqs] */
    Complex* near_spec;
    Complex* far_spec;
    Complex* echo_spec;
    Complex* error_spec;

    /* Power estimation [n_freqs] */
    float* power;

    /* TD constraint window [fft_size] */
    float* td_window;

    /* Temporary [fft_size] / [n_freqs] */
    float* temp_time;
    Complex* temp_spec;
};

/* Complex multiply: a * b */
static inline Complex cmul(Complex a, Complex b) {
    Complex r;
    r.r = a.r * b.r - a.i * b.i;
    r.i = a.r * b.i + a.i * b.r;
    return r;
}

/* Next power of 2 >= n */
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

Pbfdkf* pbfdkf_create(int block_size, int hop_size, int n_partitions,
                       float delta, float q_high, float q_low) {
    (void)block_size; /* block_size now derived from hop_size */
    if (hop_size <= 0 || n_partitions <= 0) return NULL;

    Pbfdkf* f = (Pbfdkf*)calloc(1, sizeof(Pbfdkf));
    if (!f) return NULL;

    /* Direction A: block_size = 2*hop, FFT zero-pads to next pow2 */
    f->block_size = 2 * hop_size;
    f->hop_size = hop_size;
    f->fft_size = next_pow2(f->block_size);
    f->n_partitions = n_partitions;
    f->n_freqs = f->fft_size / 2 + 1;
    f->delta = delta;
    f->alpha_power = 0.9f;
    f->alpha_r = 0.95f;
    f->partition_idx = 0;
    f->q_low_base = q_low;
    f->p_floor_beta = 0.1f;
    f->p_floor_beta_frames = 0;
    f->p_max_override = 0.5f;
    f->p_max_override_frames = 0;

    f->fft = fft_create(f->fft_size);
    if (!f->fft) goto error;

    /* Allocate W [n_partitions][n_freqs] */
    f->W = (Complex**)calloc(n_partitions, sizeof(Complex*));
    if (!f->W) goto error;
    for (int p = 0; p < n_partitions; p++) {
        f->W[p] = (Complex*)calloc(f->n_freqs, sizeof(Complex));
        if (!f->W[p]) goto error;
    }

    /* Allocate X_buf */
    f->X_buf = (Complex**)calloc(n_partitions, sizeof(Complex*));
    if (!f->X_buf) goto error;
    for (int p = 0; p < n_partitions; p++) {
        f->X_buf[p] = (Complex*)calloc(f->n_freqs, sizeof(Complex));
        if (!f->X_buf[p]) goto error;
    }

    /* Allocate Kalman state: P [n_partitions][n_freqs] */
    f->P = (float**)calloc(n_partitions, sizeof(float*));
    if (!f->P) goto error;
    for (int p = 0; p < n_partitions; p++) {
        f->P[p] = (float*)malloc(f->n_freqs * sizeof(float));
        if (!f->P[p]) goto error;
        for (int k = 0; k < f->n_freqs; k++) f->P[p][k] = P_INIT;
    }

    /* Q, Q_high, Q_low, R, error_psd */
    f->Q      = (float*)malloc(f->n_freqs * sizeof(float));
    f->Q_high = (float*)malloc(f->n_freqs * sizeof(float));
    f->Q_low  = (float*)malloc(f->n_freqs * sizeof(float));
    f->R      = (float*)malloc(f->n_freqs * sizeof(float));
    f->error_psd = (float*)malloc(f->n_freqs * sizeof(float));
    if (!f->Q || !f->Q_high || !f->Q_low || !f->R || !f->error_psd) goto error;

    for (int k = 0; k < f->n_freqs; k++) {
        f->Q_high[k] = q_high;
        f->Q_low[k]  = q_low;
        f->Q[k]      = q_high;
        f->R[k]      = 1e-2f;
        f->error_psd[k] = 1e-2f;
    }

    /* Input buffers [block_size] (not fft_size) */
    f->near_buffer = (float*)calloc(f->block_size, sizeof(float));
    f->far_buffer  = (float*)calloc(f->block_size, sizeof(float));
    /* Spectrum buffers [n_freqs] */
    f->near_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->far_spec    = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->echo_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->error_spec  = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->power       = (float*)calloc(f->n_freqs, sizeof(float));
    /* Temporary buffers use fft_size */
    f->temp_time   = (float*)calloc(f->fft_size, sizeof(float));
    f->temp_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));

    if (!f->near_buffer || !f->far_buffer || !f->near_spec ||
        !f->far_spec || !f->echo_spec || !f->error_spec ||
        !f->power || !f->temp_time || !f->temp_spec) {
        goto error;
    }

    /* TD constraint window [fft_size]: raised cosine fade */
    f->td_window = (float*)malloc(f->fft_size * sizeof(float));
    if (!f->td_window) goto error;
    {
        int fade_len = hop_size / 4;  /* 40 for hop=160 */
        int fade_start = hop_size - fade_len;
        for (int i = 0; i < fade_start; i++)
            f->td_window[i] = 1.0f;
        for (int i = 0; i < fade_len; i++) {
            /* Python: fade = 0.5*(1-cos(pi*arange(fade_len)/fade_len))[::-1] */
            float t = (float)(fade_len - 1 - i) / (float)fade_len;
            f->td_window[fade_start + i] = 0.5f * (1.0f - cosf((float)M_PI * t));
        }
        for (int i = hop_size; i < f->fft_size; i++)
            f->td_window[i] = 0.0f;
    }

    return f;

error:
    pbfdkf_destroy(f);
    return NULL;
}

void pbfdkf_destroy(Pbfdkf* f) {
    if (!f) return;

    if (f->W) {
        for (int p = 0; p < f->n_partitions; p++) free(f->W[p]);
        free(f->W);
    }
    if (f->X_buf) {
        for (int p = 0; p < f->n_partitions; p++) free(f->X_buf[p]);
        free(f->X_buf);
    }
    if (f->P) {
        for (int p = 0; p < f->n_partitions; p++) free(f->P[p]);
        free(f->P);
    }

    free(f->Q);
    free(f->Q_high);
    free(f->Q_low);
    free(f->R);
    free(f->error_psd);

    fft_destroy(f->fft);
    free(f->td_window);
    free(f->near_buffer);
    free(f->far_buffer);
    free(f->near_spec);
    free(f->far_spec);
    free(f->echo_spec);
    free(f->error_spec);
    free(f->power);
    free(f->temp_time);
    free(f->temp_spec);
    free(f);
}

void pbfdkf_reset(Pbfdkf* f) {
    if (!f) return;

    for (int p = 0; p < f->n_partitions; p++) {
        memset(f->W[p], 0, f->n_freqs * sizeof(Complex));
        memset(f->X_buf[p], 0, f->n_freqs * sizeof(Complex));
        for (int k = 0; k < f->n_freqs; k++) f->P[p][k] = P_INIT;
    }
    for (int k = 0; k < f->n_freqs; k++) {
        f->Q[k] = f->Q_high[k];
        f->R[k] = 1e-2f;
        f->error_psd[k] = 1e-2f;
    }
    memset(f->near_buffer, 0, f->block_size * sizeof(float));
    memset(f->far_buffer, 0, f->block_size * sizeof(float));
    memset(f->power, 0, f->n_freqs * sizeof(float));
    f->partition_idx = 0;
    f->p_floor_beta = 0.1f;
    f->p_floor_beta_frames = 0;
    f->p_max_override = 0.5f;
    f->p_max_override_frames = 0;
}

void pbfdkf_reset_weights(Pbfdkf* f) {
    if (!f) return;
    for (int p = 0; p < f->n_partitions; p++) {
        memset(f->W[p], 0, f->n_freqs * sizeof(Complex));
        for (int k = 0; k < f->n_freqs; k++) f->P[p][k] = P_INIT;
    }
    for (int k = 0; k < f->n_freqs; k++) {
        f->Q[k] = f->Q_high[k];
        f->R[k] = 1e-2f;
        f->error_psd[k] = 1e-2f;
    }
    f->p_floor_beta = 0.1f;
    f->p_floor_beta_frames = 0;
    f->p_max_override = 0.5f;
    f->p_max_override_frames = 0;
}

int pbfdkf_process(Pbfdkf* f,
                   const float* near_end,
                   const float* far_end,
                   float* output,
                   float mu_scale) {
    return pbfdkf_process_per_bin(f, near_end, far_end, output, NULL, mu_scale);
}

int pbfdkf_process_per_bin(Pbfdkf* f,
                           const float* near_end,
                           const float* far_end,
                           float* output,
                           const float* per_bin_mu,
                           float scalar_mu) {
    float mu_scale = scalar_mu;
    if (!f || !near_end || !far_end || !output) return -1;

    const int bs = f->block_size;       /* 2*hop = 320 */
    const int fft_sz = f->fft_size;     /* next pow2 = 512 */
    const int hop = f->hop_size;
    const int nfreq = f->n_freqs;
    const float alpha = f->alpha_power;
    const float delta = f->delta;

    /* Shift buffers left by hop, insert new at end */
    memmove(f->near_buffer, f->near_buffer + hop, (bs - hop) * sizeof(float));
    memcpy(f->near_buffer + bs - hop, near_end, hop * sizeof(float));

    memmove(f->far_buffer, f->far_buffer + hop, (bs - hop) * sizeof(float));
    memcpy(f->far_buffer + bs - hop, far_end, hop * sizeof(float));

    /* FFT: zero-pad block_size to fft_size */
    memcpy(f->temp_time, f->near_buffer, bs * sizeof(float));
    memset(f->temp_time + bs, 0, (fft_sz - bs) * sizeof(float));
    fft_forward(f->fft, f->temp_time, f->near_spec);

    memcpy(f->temp_time, f->far_buffer, bs * sizeof(float));
    memset(f->temp_time + bs, 0, (fft_sz - bs) * sizeof(float));
    fft_forward(f->fft, f->temp_time, f->far_spec);

    /* Store far spec in circular buffer */
    int curr_p = f->partition_idx;
    memcpy(f->X_buf[curr_p], f->far_spec, nfreq * sizeof(Complex));

    /* Update power estimate (Python aec.py PBFDAF.process lines 672-678):
     * cold start — initialize directly on first active frame instead of
     * EMA-warming from zero (otherwise EMA needs ~30 frames to catch up). */
    {
        float power_sum = 0.0f, far_psd_sum = 0.0f;
        for (int k = 0; k < nfreq; k++) {
            power_sum += f->power[k];
            float pwr = f->far_spec[k].r * f->far_spec[k].r +
                        f->far_spec[k].i * f->far_spec[k].i;
            far_psd_sum += pwr;
        }
        if (power_sum < 1e-10f && far_psd_sum > 1e-10f) {
            for (int k = 0; k < nfreq; k++) {
                f->power[k] = f->far_spec[k].r * f->far_spec[k].r +
                              f->far_spec[k].i * f->far_spec[k].i;
            }
        } else {
            for (int k = 0; k < nfreq; k++) {
                float pwr = f->far_spec[k].r * f->far_spec[k].r +
                            f->far_spec[k].i * f->far_spec[k].i;
                f->power[k] = alpha * f->power[k] + (1.0f - alpha) * pwr;
            }
        }
    }

    /* Echo estimate: Y_hat = sum(W[p] * X_buf[p]) */
    memset(f->echo_spec, 0, nfreq * sizeof(Complex));
    for (int p = 0; p < f->n_partitions; p++) {
        int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;
        for (int k = 0; k < nfreq; k++) {
            Complex prod = cmul(f->W[p][k], f->X_buf[p_idx][k]);
            f->echo_spec[k].r += prod.r;
            f->echo_spec[k].i += prod.i;
        }
    }

    /* IFFT echo estimate (fft_size) */
    fft_inverse(f->fft, f->echo_spec, f->temp_time);

    /* Output = valid region [hop, block_size) of overlap-save */
    for (int n = 0; n < hop; n++) {
        output[n] = f->near_buffer[hop + n] - f->temp_time[hop + n];
    }

    /* Error spectrum: zero-pad to fft_size, valid region at [hop, block_size) */
    memset(f->temp_time, 0, fft_sz * sizeof(float));
    memcpy(f->temp_time + hop, output, hop * sizeof(float));
    fft_forward(f->fft, f->temp_time, f->error_spec);

    /* Far-end activity gate: skip update when ref is silent */
    float far_hop_energy = 0.0f;
    for (int n = 0; n < hop; n++)
        far_hop_energy += far_end[n] * far_end[n];
    far_hop_energy /= hop;

    if (mu_scale > 0 && far_hop_energy > 1e-6f) {
        /* Update R (measurement noise) from error PSD */
        for (int k = 0; k < nfreq; k++) {
            float epsd = f->error_spec[k].r * f->error_spec[k].r +
                         f->error_spec[k].i * f->error_spec[k].i;
            f->error_psd[k] = f->alpha_r * f->error_psd[k] +
                               (1.0f - f->alpha_r) * epsd;
            f->R[k] = f->error_psd[k] > delta ? f->error_psd[k] : delta;
        }

        /* Python uses mu_mean (average of per-bin array) for R_scale and q_scale
         * (scalar effects). Match by computing mean if per-bin is provided. */
        float mu_mean = mu_scale;
        if (per_bin_mu) {
            float s = 0.0f;
            for (int k = 0; k < nfreq; k++) s += per_bin_mu[k];
            mu_mean = s / (float)nfreq;
        }

        /* Adaptive R: scale by mu_mean to break R-deadlock */
        float R_scale = 0.1f + 0.9f * (1.0f - mu_mean);
        for (int k = 0; k < nfreq; k++) f->R[k] *= R_scale;

        /* Q modulation: reduce Q during DT */
        float q_scale = 0.1f + 0.9f * mu_mean;

        /* Per-bin Q gating: only add Q where far-end has energy */
        float mean_power = 0.0f;
        for (int k = 0; k < nfreq; k++) mean_power += f->power[k];
        mean_power /= nfreq;
        float q_gate_thresh = mean_power * 0.01f + 1e-6f;

        /* Bug 1 fix: compute global denominator (sum over ALL partitions)
         * Innovation variance = Σ_p P[p] × |X_p|² + R
         * Old code only used one partition → K was ~n_partitions× too large */
        /* Use stack buffer for denominator — temp_time is reused by TD constraint */
        float denom_buf[257];  /* max n_freqs for 512-FFT */
        for (int k = 0; k < nfreq; k++) {
            float total_echo_var = 0.0f;
            for (int p = 0; p < f->n_partitions; p++) {
                int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;
                Complex X = f->X_buf[p_idx][k];
                total_echo_var += f->P[p][k] * (X.r * X.r + X.i * X.i);
            }
            denom_buf[k] = total_echo_var + f->R[k] + delta;
        }

        /* Kalman update per partition */
        for (int p = 0; p < f->n_partitions; p++) {
            int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;

            for (int k = 0; k < nfreq; k++) {
                Complex X = f->X_buf[p_idx][k];

                /* K_optimal = P * conj(X) / global_denominator */
                float K_scale = f->P[p][k] / denom_buf[k];

                /* Bug 2 fix: separate K for weights (scaled) and P update (unscaled)
                 * mu_scale only affects weight update, not covariance.
                 * Per-bin mu (FS-parity): Python applies different step size per
                 * frequency bin post-convergence (mu_min + (1-mu_min)*per_bin_eer). */
                float mu_k = per_bin_mu ? per_bin_mu[k] : mu_scale;
                float K_scale_mu = K_scale * mu_k;
                Complex K_w;  /* for weight update */
                K_w.r = K_scale_mu * X.r;
                K_w.i = K_scale_mu * (-X.i);

                /* W += K_scaled * error_spec */
                Complex Ke = cmul(K_w, f->error_spec[k]);
                f->W[p][k].r += Ke.r;
                f->W[p][k].i += Ke.i;

                /* v3.7.0 G1: blended KX for P update.
                 * Python aec.py lines 856-868: KX = mu_mean*KX_optimal + (1-mu_mean)*KX_scaled
                 * where KX_optimal = K_scale × |X|² (un-mu)
                 *       KX_scaled  = K_scale × mu_k × |X|² (per-bin mu)
                 *       blend factor mu_mean = mean(mu_scale_arr) — scalar.
                 * Decouples P from W during DT-mu-scaling: FS (mu_mean=1) → 100% optimal,
                 * DT (mu_mean=0) → 100% scaled, smooth transition between. Avoids both
                 * extremes' regression risk (full optimal: P over-confident in DT;
                 * full scaled: P over-cautious in steady state with weak far). */
                float mag2 = X.r * X.r + X.i * X.i;
                float KX_optimal = K_scale * mag2;
                float KX_scaled  = K_scale_mu * mag2;
                float KX = mu_mean * KX_optimal + (1.0f - mu_mean) * KX_scaled;
                float Q_mod = f->Q[k] * q_scale;
                float Q_gated = (f->power[k] > q_gate_thresh) ? Q_mod : (Q_mod * 0.05f);
                float P_new = (1.0f - KX) * f->P[p][k] + Q_gated;
                /* v3.x parity: clip(P_new, P_floor, p_max) where p_max may be EPC override */
                float p_max_eff = f->p_max_override;
                if (P_new > p_max_eff) P_new = p_max_eff;
                /* P-floor: clamp P >= Q_high * beta (v2.1.0) */
                float p_floor = f->Q_high[k] * f->p_floor_beta;
                if (P_new < p_floor) P_new = p_floor;
                f->P[p][k] = P_new;
            }

            /* Time-domain constraint: raised cosine fade window */
            fft_inverse(f->fft, f->W[p], f->temp_time);
            for (int k = 0; k < fft_sz; k++)
                f->temp_time[k] *= f->td_window[k];
            fft_forward(f->fft, f->temp_time, f->W[p]);
        }
    }

    /* P-floor beta countdown (v2.1.0) */
    if (f->p_floor_beta_frames > 0) {
        f->p_floor_beta_frames--;
        if (f->p_floor_beta_frames == 0)
            f->p_floor_beta = 0.1f;
    }

    /* v3.x: P_max override countdown */
    if (f->p_max_override_frames > 0) {
        f->p_max_override_frames--;
        if (f->p_max_override_frames == 0)
            f->p_max_override = 0.5f;
    }

    /* Advance partition index */
    f->partition_idx = (f->partition_idx + 1) % f->n_partitions;

    return 0;
}

void pbfdkf_switch_q_low(Pbfdkf* f) {
    if (!f) return;
    memcpy(f->Q, f->Q_low, f->n_freqs * sizeof(float));
}

void pbfdkf_set_q_high(Pbfdkf* f, float q_high) {
    if (!f) return;
    for (int k = 0; k < f->n_freqs; k++) {
        f->Q_high[k] = q_high;
        f->Q[k] = q_high;
    }
}

void pbfdkf_set_q_ratio(Pbfdkf* f, float q_high, float q_low, float ratio) {
    if (!f) return;
    for (int k = 0; k < f->n_freqs; k++) {
        f->Q_high[k] = q_high * ratio;
        f->Q_low[k]  = q_low * ratio;
        f->Q[k]      = f->Q_high[k];
    }
}

void pbfdkf_set_p_floor_epc(Pbfdkf* f, float beta, int frames) {
    if (!f) return;
    f->p_floor_beta = beta;
    f->p_floor_beta_frames = frames;
}

void pbfdkf_set_p_max_override(Pbfdkf* f, float p_max, int frames) {
    if (!f) return;
    f->p_max_override = p_max;
    f->p_max_override_frames = frames;
}

int pbfdkf_copy_weights(Pbfdkf* dst, const Pbfdkf* src) {
    if (!dst || !src) return -1;
    if (dst->n_partitions != src->n_partitions ||
        dst->n_freqs != src->n_freqs) return -1;

    /* Only copy filter coefficients W, NOT Kalman internal state (P/Q/R).
     * P/Q/R are confidence states accumulated from each filter's own
     * learning history — copying them across filters contaminates the
     * destination's Kalman gain. After W copy, P self-corrects within
     * a few frames. Match Python's copy_weights_from() behavior. */
    for (int p = 0; p < src->n_partitions; p++) {
        memcpy(dst->W[p], src->W[p], src->n_freqs * sizeof(Complex));
    }
    return 0;
}

float pbfdkf_get_error_energy(const Pbfdkf* f) {
    if (!f) return 0.0f;
    float energy = 0.0f;
    for (int k = 0; k < f->n_freqs; k++) {
        energy += f->error_spec[k].r * f->error_spec[k].r +
                  f->error_spec[k].i * f->error_spec[k].i;
    }
    return energy;
}

const Complex* pbfdkf_get_echo_spec(const Pbfdkf* f) {
    return f ? f->echo_spec : NULL;
}

float pbfdkf_get_w_mean_mag(const Pbfdkf* f) {
    if (!f || !f->W || !f->W[0]) return 0.0f;
    float sum = 0.0f;
    for (int k = 0; k < f->n_freqs; k++)
        sum += sqrtf(f->W[0][k].r * f->W[0][k].r + f->W[0][k].i * f->W[0][k].i);
    return sum / (float)f->n_freqs;
}

const Complex* pbfdkf_get_far_spec(const Pbfdkf* f) {
    return f ? f->far_spec : NULL;
}

const Complex* pbfdkf_get_near_spec(const Pbfdkf* f) {
    return f ? f->near_spec : NULL;
}

const Complex* pbfdkf_get_error_spec(const Pbfdkf* f) {
    return f ? f->error_spec : NULL;
}

float pbfdkf_get_far_power(const Pbfdkf* f) {
    if (!f) return 0.0f;
    float total = 0.0f;
    for (int k = 0; k < f->n_freqs; k++) total += f->power[k];
    return total / f->n_freqs;
}

int pbfdkf_get_block_size(const Pbfdkf* f)    { return f ? f->block_size : 0; }
int pbfdkf_get_fft_size(const Pbfdkf* f)      { return f ? f->fft_size : 0; }
int pbfdkf_get_hop_size(const Pbfdkf* f)      { return f ? f->hop_size : 0; }
int pbfdkf_get_n_freqs(const Pbfdkf* f)       { return f ? f->n_freqs : 0; }
int pbfdkf_get_n_partitions(const Pbfdkf* f)  { return f ? f->n_partitions : 0; }

const float* pbfdkf_get_P(const Pbfdkf* f, int partition) {
    if (!f || partition < 0 || partition >= f->n_partitions) return NULL;
    return f->P[partition];
}

const Complex* pbfdkf_get_W(const Pbfdkf* f, int partition) {
    if (!f || partition < 0 || partition >= f->n_partitions) return NULL;
    return f->W[partition];
}

const float* pbfdkf_get_power(const Pbfdkf* f) {
    return f ? f->power : NULL;
}
