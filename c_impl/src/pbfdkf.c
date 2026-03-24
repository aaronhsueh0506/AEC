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

#define P_INIT  0.01f   /* Initial error covariance (was 0.5, caused K explosion) */
#define P_MAX   0.02f   /* Max P to prevent covariance growth during silence */

struct Pbfdkf {
    int block_size;       /* FFT size (512) */
    int hop_size;         /* Processing hop (160) */
    int n_partitions;
    int n_freqs;          /* block_size/2 + 1 = 257 */
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
    float alpha_r;        /* R smoothing (0.95) */
    float q_low_base;     /* base Q_low value for set_q_ratio */

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

    /* Temporary [block_size] / [n_freqs] */
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

Pbfdkf* pbfdkf_create(int block_size, int hop_size, int n_partitions,
                       float delta, float q_high, float q_low) {
    if (block_size <= 0 || hop_size <= 0 || n_partitions <= 0) return NULL;
    if (hop_size > block_size / 2) return NULL; /* overlap-save constraint */

    Pbfdkf* f = (Pbfdkf*)calloc(1, sizeof(Pbfdkf));
    if (!f) return NULL;

    f->block_size = block_size;
    f->hop_size = hop_size;
    f->n_partitions = n_partitions;
    f->n_freqs = block_size / 2 + 1;
    f->delta = delta;
    f->alpha_power = 0.9f;
    f->alpha_r = 0.95f;
    f->partition_idx = 0;
    f->q_low_base = q_low;

    f->fft = fft_create(block_size);
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

    /* Buffers */
    f->near_buffer = (float*)calloc(block_size, sizeof(float));
    f->far_buffer  = (float*)calloc(block_size, sizeof(float));
    f->near_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->far_spec    = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->echo_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->error_spec  = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->power       = (float*)calloc(f->n_freqs, sizeof(float));
    f->temp_time   = (float*)calloc(block_size, sizeof(float));
    f->temp_spec   = (Complex*)calloc(f->n_freqs, sizeof(Complex));

    if (!f->near_buffer || !f->far_buffer || !f->near_spec ||
        !f->far_spec || !f->echo_spec || !f->error_spec ||
        !f->power || !f->temp_time || !f->temp_spec) {
        goto error;
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
}

int pbfdkf_process(Pbfdkf* f,
                   const float* near_end,
                   const float* far_end,
                   float* output,
                   float mu_scale) {
    if (!f || !near_end || !far_end || !output) return -1;

    const int bs = f->block_size;
    const int hop = f->hop_size;
    const int nfreq = f->n_freqs;
    const float alpha = f->alpha_power;
    const float delta = f->delta;

    /* Shift buffers left by hop, insert new at end */
    memmove(f->near_buffer, f->near_buffer + hop, (bs - hop) * sizeof(float));
    memcpy(f->near_buffer + bs - hop, near_end, hop * sizeof(float));

    memmove(f->far_buffer, f->far_buffer + hop, (bs - hop) * sizeof(float));
    memcpy(f->far_buffer + bs - hop, far_end, hop * sizeof(float));

    /* FFT */
    fft_forward(f->fft, f->near_buffer, f->near_spec);
    fft_forward(f->fft, f->far_buffer, f->far_spec);

    /* Store far spec in circular buffer */
    int curr_p = f->partition_idx;
    memcpy(f->X_buf[curr_p], f->far_spec, nfreq * sizeof(Complex));

    /* Update power estimate */
    for (int k = 0; k < nfreq; k++) {
        float pwr = f->far_spec[k].r * f->far_spec[k].r +
                    f->far_spec[k].i * f->far_spec[k].i;
        f->power[k] = alpha * f->power[k] + (1.0f - alpha) * pwr;
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

    /* IFFT echo estimate */
    fft_inverse(f->fft, f->echo_spec, f->temp_time);

    /* Output = last hop samples (valid region of overlap-save) */
    for (int n = 0; n < hop; n++) {
        output[n] = f->near_buffer[bs - hop + n] - f->temp_time[bs - hop + n];
    }

    /* Error spectrum: zero-pad, valid region at end */
    memset(f->temp_time, 0, bs * sizeof(float));
    memcpy(f->temp_time + bs - hop, output, hop * sizeof(float));
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

        /* Per-bin Q gating: only add Q where far-end has energy */
        float mean_power = 0.0f;
        for (int k = 0; k < nfreq; k++) mean_power += f->power[k];
        mean_power /= nfreq;
        float q_gate_thresh = mean_power * 0.01f + 1e-6f;

        /* Kalman update per partition */
        for (int p = 0; p < f->n_partitions; p++) {
            int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;

            for (int k = 0; k < nfreq; k++) {
                Complex X = f->X_buf[p_idx][k];
                float X_power = X.r * X.r + X.i * X.i + delta;

                /* K = P * conj(X) / (|X|^2 * P + R + delta) */
                float denom = X_power * f->P[p][k] + f->R[k] + delta;
                float K_scale = f->P[p][k] / denom;

                /* K = K_scale * conj(X) * mu_scale */
                Complex K;
                K.r = K_scale * X.r * mu_scale;
                K.i = K_scale * (-X.i) * mu_scale;

                /* W += K * error_spec */
                Complex Ke = cmul(K, f->error_spec[k]);
                f->W[p][k].r += Ke.r;
                f->W[p][k].i += Ke.i;

                /* P update: P = clamp((1 - Re(K*X)) * P + Q_gated, delta, P_MAX) */
                float KX = K_scale * mu_scale * (X.r * X.r + X.i * X.i);
                float Q_gated = (f->power[k] > q_gate_thresh) ? f->Q[k] : 0.0f;
                float P_new = (1.0f - KX) * f->P[p][k] + Q_gated;
                if (P_new < delta) P_new = delta;
                if (P_new > P_MAX) P_new = P_MAX;
                f->P[p][k] = P_new;
            }

            /* Time-domain constraint: w[hop_size:] = 0 */
            fft_inverse(f->fft, f->W[p], f->temp_time);
            memset(f->temp_time + hop, 0, (bs - hop) * sizeof(float));
            fft_forward(f->fft, f->temp_time, f->W[p]);
        }
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

int pbfdkf_copy_weights(Pbfdkf* dst, const Pbfdkf* src) {
    if (!dst || !src) return -1;
    if (dst->n_partitions != src->n_partitions ||
        dst->n_freqs != src->n_freqs) return -1;

    for (int p = 0; p < src->n_partitions; p++) {
        memcpy(dst->W[p], src->W[p], src->n_freqs * sizeof(Complex));
        memcpy(dst->P[p], src->P[p], src->n_freqs * sizeof(float));
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

const Complex* pbfdkf_get_far_spec(const Pbfdkf* f) {
    return f ? f->far_spec : NULL;
}

const Complex* pbfdkf_get_near_spec(const Pbfdkf* f) {
    return f ? f->near_spec : NULL;
}

float pbfdkf_get_far_power(const Pbfdkf* f) {
    if (!f) return 0.0f;
    float total = 0.0f;
    for (int k = 0; k < f->n_freqs; k++) total += f->power[k];
    return total / f->n_freqs;
}

int pbfdkf_get_block_size(const Pbfdkf* f)    { return f ? f->block_size : 0; }
int pbfdkf_get_hop_size(const Pbfdkf* f)      { return f ? f->hop_size : 0; }
int pbfdkf_get_n_freqs(const Pbfdkf* f)       { return f ? f->n_freqs : 0; }
int pbfdkf_get_n_partitions(const Pbfdkf* f)  { return f ? f->n_partitions : 0; }
