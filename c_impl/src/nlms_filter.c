/**
 * nlms_filter.c - Time-domain NLMS adaptive filter implementation
 *
 * Normalized Least Mean Squares algorithm:
 *   y_hat[n] = w^T * x[n]           (echo estimate)
 *   e[n] = d[n] - y_hat[n]          (error/output)
 *   mu_eff = mu / (||x[n]||^2 + delta)
 *   w = leak * w + mu_eff * e[n] * x[n]  (weight update)
 */

#include "nlms_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct NlmsFilter {
    int filter_length;
    float mu;               // Step size
    float delta;            // Regularization
    float leak;             // Weight leakage

    float* weights;         // Filter weights [filter_length]
    float* ref_buffer;      // Circular buffer for reference [filter_length]
    int buf_idx;            // Current buffer index (newest sample)
    float power_sum;        // Running sum of x^2 for efficiency
};

NlmsFilter* nlms_create(int filter_length, float mu, float delta, float leak) {
    if (filter_length <= 0 || mu <= 0 || mu > 1.0f) {
        return NULL;
    }

    NlmsFilter* filter = (NlmsFilter*)calloc(1, sizeof(NlmsFilter));
    if (!filter) return NULL;

    filter->filter_length = filter_length;
    filter->mu = mu;
    filter->delta = delta;
    filter->leak = leak;

    filter->weights = (float*)calloc(filter_length, sizeof(float));
    filter->ref_buffer = (float*)calloc(filter_length, sizeof(float));

    if (!filter->weights || !filter->ref_buffer) {
        nlms_destroy(filter);
        return NULL;
    }

    filter->buf_idx = 0;
    filter->power_sum = 0.0f;

    return filter;
}

void nlms_destroy(NlmsFilter* filter) {
    if (filter) {
        free(filter->weights);
        free(filter->ref_buffer);
        free(filter);
    }
}

void nlms_reset(NlmsFilter* filter) {
    if (!filter) return;

    memset(filter->weights, 0, filter->filter_length * sizeof(float));
    memset(filter->ref_buffer, 0, filter->filter_length * sizeof(float));
    filter->buf_idx = 0;
    filter->power_sum = 0.0f;
}

float nlms_process_sample(NlmsFilter* filter,
                          float near_end,
                          float far_end,
                          bool update_weights) {
    const int L = filter->filter_length;

    // Update power sum: remove oldest, add newest
    int oldest_idx = (filter->buf_idx + 1) % L;
    float oldest_sample = filter->ref_buffer[oldest_idx];
    filter->power_sum -= oldest_sample * oldest_sample;
    filter->power_sum += far_end * far_end;

    // Prevent negative power due to numerical errors
    if (filter->power_sum < 0.0f) {
        filter->power_sum = 0.0f;
    }

    // Insert new reference sample
    filter->ref_buffer[filter->buf_idx] = far_end;

    // Compute echo estimate: y_hat = w^T * x
    // Reference buffer is organized as circular buffer
    // x[n] is at buf_idx, x[n-1] is at buf_idx-1 (wrapped), etc.
    float echo_est = 0.0f;
    for (int k = 0; k < L; k++) {
        // x[n-k] is at position (buf_idx - k + L) % L
        int idx = (filter->buf_idx - k + L) % L;
        echo_est += filter->weights[k] * filter->ref_buffer[idx];
    }

    // Error signal (echo-cancelled output)
    float error = near_end - echo_est;

    // Update weights if enabled (not during double-talk)
    if (update_weights) {
        // Normalized step size
        float norm_power = filter->power_sum + filter->delta;
        float mu_eff = filter->mu / norm_power;

        // Weight update with leakage
        for (int k = 0; k < L; k++) {
            int idx = (filter->buf_idx - k + L) % L;
            filter->weights[k] = filter->leak * filter->weights[k]
                               + mu_eff * error * filter->ref_buffer[idx];
        }
    }

    // Advance buffer index for next sample
    filter->buf_idx = (filter->buf_idx + 1) % L;

    return error;
}

void nlms_process_block(NlmsFilter* filter,
                        const float* near_end,
                        const float* far_end,
                        float* output,
                        float* echo_est,
                        int num_samples,
                        bool update_weights) {
    for (int n = 0; n < num_samples; n++) {
        float err = nlms_process_sample(filter, near_end[n], far_end[n], update_weights);
        output[n] = err;

        if (echo_est) {
            // Echo estimate = near_end - error
            echo_est[n] = near_end[n] - err;
        }
    }
}

float nlms_get_echo_estimate(const NlmsFilter* filter) {
    const int L = filter->filter_length;
    float echo_est = 0.0f;

    for (int k = 0; k < L; k++) {
        // Current reference buffer state (without advancing)
        int idx = (filter->buf_idx - 1 - k + L) % L;
        echo_est += filter->weights[k] * filter->ref_buffer[idx];
    }

    return echo_est;
}

int nlms_get_filter_length(const NlmsFilter* filter) {
    return filter ? filter->filter_length : 0;
}

float nlms_get_mu(const NlmsFilter* filter) {
    return filter ? filter->mu : 0.0f;
}

void nlms_set_mu(NlmsFilter* filter, float mu) {
    if (filter && mu > 0 && mu <= 1.0f) {
        filter->mu = mu;
    }
}

float nlms_get_ref_power(const NlmsFilter* filter) {
    return filter ? filter->power_sum : 0.0f;
}
