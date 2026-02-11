/**
 * res_filter.c - Residual Echo Suppressor implementation
 *
 * Algorithm:
 * 1. Estimate Echo-to-Error Ratio (EER) per frequency bin
 *    EER[k] = |Y_hat[k]|^2 / (|E[k]|^2 + eps)
 *
 * 2. Compute suppression gain
 *    G[k] = 1 / (1 + alpha * EER[k])
 *    G[k] = max(G[k], G_min)
 *
 * 3. Smooth gain over time
 *    G_smooth[k] = alpha * G_smooth[k] + (1 - alpha) * G[k]
 *
 * 4. Apply gain
 *    E_out[k] = G_smooth[k] * E[k]
 */

#include "res_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct ResFilter {
    int n_freqs;

    // Parameters
    float g_min;          // Minimum gain (linear)
    float over_sub;       // Over-subtraction factor
    float alpha;          // Gain smoothing

    // State
    float* gain;          // Current gain [n_freqs]
    float* gain_smooth;   // Smoothed gain [n_freqs]
    float* echo_psd;      // Echo PSD estimate [n_freqs]
    float* error_psd;     // Error PSD estimate [n_freqs]

    float alpha_psd;      // PSD smoothing factor
};

ResFilter* res_create(int n_freqs, float g_min_db, float over_sub, float alpha) {
    if (n_freqs <= 0) return NULL;

    ResFilter* res = (ResFilter*)calloc(1, sizeof(ResFilter));
    if (!res) return NULL;

    res->n_freqs = n_freqs;
    res->g_min = powf(10.0f, g_min_db / 20.0f);  // dB to linear
    res->over_sub = over_sub;
    res->alpha = alpha;
    res->alpha_psd = 0.9f;

    res->gain = (float*)malloc(n_freqs * sizeof(float));
    res->gain_smooth = (float*)malloc(n_freqs * sizeof(float));
    res->echo_psd = (float*)calloc(n_freqs, sizeof(float));
    res->error_psd = (float*)calloc(n_freqs, sizeof(float));

    if (!res->gain || !res->gain_smooth || !res->echo_psd || !res->error_psd) {
        res_destroy(res);
        return NULL;
    }

    // Initialize gains to 1.0 (no suppression)
    for (int k = 0; k < n_freqs; k++) {
        res->gain[k] = 1.0f;
        res->gain_smooth[k] = 1.0f;
    }

    return res;
}

void res_destroy(ResFilter* res) {
    if (res) {
        free(res->gain);
        free(res->gain_smooth);
        free(res->echo_psd);
        free(res->error_psd);
        free(res);
    }
}

void res_reset(ResFilter* res) {
    if (!res) return;

    for (int k = 0; k < res->n_freqs; k++) {
        res->gain[k] = 1.0f;
        res->gain_smooth[k] = 1.0f;
        res->echo_psd[k] = 0.0f;
        res->error_psd[k] = 0.0f;
    }
}

void res_process(ResFilter* res,
                 const Complex* error_spec,
                 const Complex* echo_spec,
                 float far_power,
                 Complex* output_spec) {
    if (!res || !error_spec || !echo_spec || !output_spec) return;

    const int nfreq = res->n_freqs;
    const float alpha_psd = res->alpha_psd;
    const float eps = 1e-10f;

    // Only apply suppression when far-end is active
    float far_active = (far_power > 1e-6f) ? 1.0f : 0.0f;

    for (int k = 0; k < nfreq; k++) {
        // Compute power spectra
        float echo_pwr = echo_spec[k].r * echo_spec[k].r +
                         echo_spec[k].i * echo_spec[k].i;
        float error_pwr = error_spec[k].r * error_spec[k].r +
                          error_spec[k].i * error_spec[k].i;

        // Smooth PSD estimates
        res->echo_psd[k] = alpha_psd * res->echo_psd[k] +
                           (1.0f - alpha_psd) * echo_pwr;
        res->error_psd[k] = alpha_psd * res->error_psd[k] +
                            (1.0f - alpha_psd) * error_pwr;

        // Echo-to-Error Ratio (EER)
        float eer = res->echo_psd[k] / (res->error_psd[k] + eps);

        // Compute gain based on EER
        // Higher EER = more echo leakage = more suppression needed
        float g = 1.0f / (1.0f + res->over_sub * eer);

        // Apply minimum gain floor
        if (g < res->g_min) {
            g = res->g_min;
        }

        // When far-end is inactive, gradually release suppression
        if (far_active < 0.5f) {
            g = 1.0f;
        }

        res->gain[k] = g;

        // Smooth gain over time (asymmetric: fast attack, slow release)
        float alpha_g = (g < res->gain_smooth[k]) ? 0.3f : res->alpha;
        res->gain_smooth[k] = alpha_g * res->gain_smooth[k] +
                              (1.0f - alpha_g) * g;

        // Apply gain
        output_spec[k].r = res->gain_smooth[k] * error_spec[k].r;
        output_spec[k].i = res->gain_smooth[k] * error_spec[k].i;
    }
}

void res_get_gains(const ResFilter* res, float* gains) {
    if (res && gains) {
        memcpy(gains, res->gain_smooth, res->n_freqs * sizeof(float));
    }
}

void res_set_g_min(ResFilter* res, float g_min_db) {
    if (res) {
        res->g_min = powf(10.0f, g_min_db / 20.0f);
    }
}

void res_set_over_sub(ResFilter* res, float over_sub) {
    if (res && over_sub > 0) {
        res->over_sub = over_sub;
    }
}
