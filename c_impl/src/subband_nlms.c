/**
 * subband_nlms.c - Frequency-domain NLMS implementation
 *
 * Partitioned Block Frequency-Domain Adaptive Filter (PBFDAF)
 *
 * Algorithm (Overlap-Save method):
 * 1. Buffer block_size samples (hop_size new + hop_size old)
 * 2. X[k] = FFT(x_block)
 * 3. Y_hat[k] = sum(W[p][k] * X_buf[p][k]) for all partitions
 * 4. y_hat = IFFT(Y_hat), take last hop_size samples
 * 5. e = d - y_hat
 * 6. E[k] = FFT([zeros, e])  (constrained update)
 * 7. For each partition p:
 *    mu_eff[k] = mu / (power[k] + delta)
 *    W[p][k] += mu_eff[k] * E[k] * conj(X_buf[p][k])
 *    Apply constraint: w = IFFT(W), w[hop_size:] = 0, W = FFT(w)
 */

#include "subband_nlms.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct SubbandNlms {
    int block_size;       // FFT size
    int hop_size;         // block_size / 2
    int n_partitions;     // Number of filter partitions
    int n_freqs;          // block_size / 2 + 1

    float mu;
    float delta;
    float alpha_power;    // Power smoothing (0.9)

    FftHandle* fft;

    // Filter weights [n_partitions][n_freqs]
    Complex** W;

    // Reference spectrum history [n_partitions][n_freqs]
    Complex** X_buf;
    int partition_idx;    // Current partition index (circular)

    // Input buffers
    float* near_buffer;   // [block_size] - overlap buffer for near-end
    float* far_buffer;    // [block_size] - overlap buffer for far-end

    // Spectrum buffers
    Complex* near_spec;   // [n_freqs]
    Complex* far_spec;    // [n_freqs]
    Complex* echo_spec;   // [n_freqs] - estimated echo
    Complex* error_spec;  // [n_freqs]

    // Power estimation per frequency bin
    float* power;         // [n_freqs]

    // Temporary buffers
    float* temp_time;     // [block_size]
    Complex* temp_spec;   // [n_freqs]
};

SubbandNlms* subband_nlms_create(int block_size, int n_partitions,
                                  float mu, float delta) {
    if (block_size <= 0 || n_partitions <= 0 || mu <= 0) {
        return NULL;
    }

    SubbandNlms* f = (SubbandNlms*)calloc(1, sizeof(SubbandNlms));
    if (!f) return NULL;

    f->block_size = block_size;
    f->hop_size = block_size / 2;
    f->n_partitions = n_partitions;
    f->n_freqs = block_size / 2 + 1;
    f->mu = mu;
    f->delta = delta;
    f->alpha_power = 0.9f;
    f->partition_idx = 0;

    // Create FFT handle
    f->fft = fft_create(block_size);
    if (!f->fft) goto error;

    // Allocate filter weights
    f->W = (Complex**)calloc(n_partitions, sizeof(Complex*));
    if (!f->W) goto error;
    for (int p = 0; p < n_partitions; p++) {
        f->W[p] = (Complex*)calloc(f->n_freqs, sizeof(Complex));
        if (!f->W[p]) goto error;
    }

    // Allocate reference history
    f->X_buf = (Complex**)calloc(n_partitions, sizeof(Complex*));
    if (!f->X_buf) goto error;
    for (int p = 0; p < n_partitions; p++) {
        f->X_buf[p] = (Complex*)calloc(f->n_freqs, sizeof(Complex));
        if (!f->X_buf[p]) goto error;
    }

    // Allocate buffers
    f->near_buffer = (float*)calloc(block_size, sizeof(float));
    f->far_buffer = (float*)calloc(block_size, sizeof(float));
    f->near_spec = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->far_spec = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->echo_spec = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->error_spec = (Complex*)calloc(f->n_freqs, sizeof(Complex));
    f->power = (float*)calloc(f->n_freqs, sizeof(float));
    f->temp_time = (float*)calloc(block_size, sizeof(float));
    f->temp_spec = (Complex*)calloc(f->n_freqs, sizeof(Complex));

    if (!f->near_buffer || !f->far_buffer || !f->near_spec ||
        !f->far_spec || !f->echo_spec || !f->error_spec ||
        !f->power || !f->temp_time || !f->temp_spec) {
        goto error;
    }

    return f;

error:
    subband_nlms_destroy(f);
    return NULL;
}

void subband_nlms_destroy(SubbandNlms* f) {
    if (!f) return;

    if (f->W) {
        for (int p = 0; p < f->n_partitions; p++) {
            free(f->W[p]);
        }
        free(f->W);
    }

    if (f->X_buf) {
        for (int p = 0; p < f->n_partitions; p++) {
            free(f->X_buf[p]);
        }
        free(f->X_buf);
    }

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

void subband_nlms_reset(SubbandNlms* f) {
    if (!f) return;

    for (int p = 0; p < f->n_partitions; p++) {
        memset(f->W[p], 0, f->n_freqs * sizeof(Complex));
        memset(f->X_buf[p], 0, f->n_freqs * sizeof(Complex));
    }

    memset(f->near_buffer, 0, f->block_size * sizeof(float));
    memset(f->far_buffer, 0, f->block_size * sizeof(float));
    memset(f->power, 0, f->n_freqs * sizeof(float));
    f->partition_idx = 0;
}

// Complex multiply: a * conj(b)
static inline Complex cmul_conj(Complex a, Complex b) {
    Complex result;
    result.r = a.r * b.r + a.i * b.i;
    result.i = a.i * b.r - a.r * b.i;
    return result;
}

// Complex multiply: a * b
static inline Complex cmul(Complex a, Complex b) {
    Complex result;
    result.r = a.r * b.r - a.i * b.i;
    result.i = a.r * b.i + a.i * b.r;
    return result;
}

int subband_nlms_process(SubbandNlms* f,
                         const float* near_end,
                         const float* far_end,
                         float* output,
                         bool update_weights) {
    if (!f || !near_end || !far_end || !output) {
        return -1;
    }

    const int hop = f->hop_size;
    const int nfreq = f->n_freqs;
    const float alpha = f->alpha_power;

    // Shift buffers and insert new samples (overlap-save)
    memmove(f->near_buffer, f->near_buffer + hop, hop * sizeof(float));
    memcpy(f->near_buffer + hop, near_end, hop * sizeof(float));

    memmove(f->far_buffer, f->far_buffer + hop, hop * sizeof(float));
    memcpy(f->far_buffer + hop, far_end, hop * sizeof(float));

    // FFT of input blocks
    fft_forward(f->fft, f->near_buffer, f->near_spec);
    fft_forward(f->fft, f->far_buffer, f->far_spec);

    // Store current far-end spectrum in circular buffer
    int curr_p = f->partition_idx;
    memcpy(f->X_buf[curr_p], f->far_spec, nfreq * sizeof(Complex));

    // Update power estimate
    for (int k = 0; k < nfreq; k++) {
        float pwr = f->far_spec[k].r * f->far_spec[k].r +
                    f->far_spec[k].i * f->far_spec[k].i;
        f->power[k] = alpha * f->power[k] + (1.0f - alpha) * pwr;
    }

    // Compute echo estimate: Y_hat = sum(W[p] * X_buf[p])
    memset(f->echo_spec, 0, nfreq * sizeof(Complex));
    for (int p = 0; p < f->n_partitions; p++) {
        // Get partition index (circular buffer)
        int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;
        for (int k = 0; k < nfreq; k++) {
            Complex prod = cmul(f->W[p][k], f->X_buf[p_idx][k]);
            f->echo_spec[k].r += prod.r;
            f->echo_spec[k].i += prod.i;
        }
    }

    // IFFT to get echo estimate in time domain
    fft_inverse(f->fft, f->echo_spec, f->temp_time);

    // Error signal in time domain (last hop_size samples due to overlap-save)
    for (int n = 0; n < hop; n++) {
        output[n] = f->near_buffer[hop + n] - f->temp_time[hop + n];
    }

    // Error spectrum for update (zero-pad first half for constraint)
    memset(f->temp_time, 0, hop * sizeof(float));
    memcpy(f->temp_time + hop, output, hop * sizeof(float));
    fft_forward(f->fft, f->temp_time, f->error_spec);

    // Update filter weights if enabled
    // Skip update when reference power is too low to avoid divergence
    float total_power = 0.0f;
    for (int k = 0; k < nfreq; k++) {
        total_power += f->power[k];
    }
    if (update_weights && total_power > f->delta * nfreq) {
        // Find max power for per-bin floor
        float max_power = 0.0f;
        for (int k = 0; k < nfreq; k++) {
            if (f->power[k] > max_power) max_power = f->power[k];
        }
        float power_floor_val = max_power * 1e-4f;

        for (int p = 0; p < f->n_partitions; p++) {
            int p_idx = (curr_p - p + f->n_partitions) % f->n_partitions;

            // Update each frequency bin
            for (int k = 0; k < nfreq; k++) {
                // Per-bin normalization with power floor
                float bin_power = f->power[k] > power_floor_val ? f->power[k] : power_floor_val;
                float mu_eff = f->mu / (bin_power * f->n_partitions + f->delta);

                // Gradient: E * conj(X)
                Complex grad = cmul_conj(f->error_spec[k], f->X_buf[p_idx][k]);

                // Update
                f->W[p][k].r += mu_eff * grad.r;
                f->W[p][k].i += mu_eff * grad.i;
            }

            // Apply constraint: time-domain truncation
            // w = IFFT(W), w[hop_size:] = 0, W = FFT(w)
            fft_inverse(f->fft, f->W[p], f->temp_time);
            memset(f->temp_time + hop, 0, hop * sizeof(float));
            fft_forward(f->fft, f->temp_time, f->W[p]);
        }
    }

    // Advance partition index
    f->partition_idx = (f->partition_idx + 1) % f->n_partitions;

    return 0;
}

void subband_nlms_get_echo_spectrum(const SubbandNlms* f, Complex* echo_spec) {
    if (f && echo_spec) {
        memcpy(echo_spec, f->echo_spec, f->n_freqs * sizeof(Complex));
    }
}

void subband_nlms_get_error_spectrum(const SubbandNlms* f, Complex* error_spec) {
    if (f && error_spec) {
        memcpy(error_spec, f->error_spec, f->n_freqs * sizeof(Complex));
    }
}

int subband_nlms_get_block_size(const SubbandNlms* f) {
    return f ? f->block_size : 0;
}

int subband_nlms_get_hop_size(const SubbandNlms* f) {
    return f ? f->hop_size : 0;
}

int subband_nlms_get_n_freqs(const SubbandNlms* f) {
    return f ? f->n_freqs : 0;
}

int subband_nlms_get_filter_length(const SubbandNlms* f) {
    return f ? f->hop_size * f->n_partitions : 0;
}
