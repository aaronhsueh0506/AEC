/**
 * dtd.c - Double-Talk Detector implementation
 *
 * Hybrid DTD using:
 * 1. Geigel method: |d[n]| > threshold * max(|x[n-k]|)
 * 2. Energy ratio: smoothed_error_energy / smoothed_echo_energy > threshold
 *
 * With hangover to prevent rapid switching.
 */

#include "dtd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct DtdEstimator {
    int window_length;
    float threshold;            // Geigel threshold
    int hangover_max;           // Max hangover frames
    float energy_threshold;     // Energy ratio threshold

    // Geigel: circular buffer for max tracking
    float* far_max_buffer;      // [window_length]
    int buf_idx;
    float current_far_max;      // Running max of far-end

    // Energy-based DTD
    float near_energy;          // Smoothed near-end energy
    float far_energy;           // Smoothed far-end energy
    float error_energy;         // Smoothed error energy
    float echo_energy;          // Smoothed echo estimate energy
    float alpha_energy;         // Smoothing factor (0.95)

    // State
    bool dtd_active;
    int hangover_count;
    float confidence;

    // Sample counter for frame-based hangover
    int samples_in_frame;
    int frame_size;
};

DtdEstimator* dtd_create(int window_length,
                         float threshold,
                         int hangover_frames,
                         float energy_ratio_threshold) {
    if (window_length <= 0 || threshold <= 0) {
        return NULL;
    }

    DtdEstimator* dtd = (DtdEstimator*)calloc(1, sizeof(DtdEstimator));
    if (!dtd) return NULL;

    dtd->window_length = window_length;
    dtd->threshold = threshold;
    dtd->hangover_max = hangover_frames;
    dtd->energy_threshold = energy_ratio_threshold;

    dtd->far_max_buffer = (float*)calloc(window_length, sizeof(float));
    if (!dtd->far_max_buffer) {
        dtd_destroy(dtd);
        return NULL;
    }

    dtd->buf_idx = 0;
    dtd->current_far_max = 0.0f;

    dtd->near_energy = 0.0f;
    dtd->far_energy = 0.0f;
    dtd->error_energy = 0.0f;
    dtd->echo_energy = 0.0f;
    dtd->alpha_energy = 0.95f;

    dtd->dtd_active = false;
    dtd->hangover_count = 0;
    dtd->confidence = 0.0f;

    // Assume 160 samples per frame (10ms @ 16kHz)
    dtd->frame_size = 160;
    dtd->samples_in_frame = 0;

    return dtd;
}

void dtd_destroy(DtdEstimator* dtd) {
    if (dtd) {
        free(dtd->far_max_buffer);
        free(dtd);
    }
}

void dtd_reset(DtdEstimator* dtd) {
    if (!dtd) return;

    memset(dtd->far_max_buffer, 0, dtd->window_length * sizeof(float));
    dtd->buf_idx = 0;
    dtd->current_far_max = 0.0f;

    dtd->near_energy = 0.0f;
    dtd->far_energy = 0.0f;
    dtd->error_energy = 0.0f;
    dtd->echo_energy = 0.0f;

    dtd->dtd_active = false;
    dtd->hangover_count = 0;
    dtd->confidence = 0.0f;
    dtd->samples_in_frame = 0;
}

// Helper: recompute max from buffer
static float compute_max(const float* buffer, int length) {
    float max_val = 0.0f;
    for (int i = 0; i < length; i++) {
        if (buffer[i] > max_val) {
            max_val = buffer[i];
        }
    }
    return max_val;
}

bool dtd_detect_sample(DtdEstimator* dtd,
                       float near_end,
                       float far_end,
                       float error,
                       float echo_est) {
    const float alpha = dtd->alpha_energy;

    // Absolute values
    float abs_near = fabsf(near_end);
    float abs_far = fabsf(far_end);
    (void)error;     // Used for energy calculation below
    (void)echo_est;  // Used for energy calculation below

    // Update far-end max buffer (for Geigel)
    dtd->far_max_buffer[dtd->buf_idx] = abs_far;
    dtd->buf_idx = (dtd->buf_idx + 1) % dtd->window_length;

    // Update current max (efficient incremental update)
    if (abs_far >= dtd->current_far_max) {
        dtd->current_far_max = abs_far;
    } else {
        // Recompute max periodically (when buffer wraps)
        if (dtd->buf_idx == 0) {
            dtd->current_far_max = compute_max(dtd->far_max_buffer, dtd->window_length);
        }
    }

    // Update smoothed energies
    dtd->near_energy = alpha * dtd->near_energy + (1.0f - alpha) * (near_end * near_end);
    dtd->far_energy = alpha * dtd->far_energy + (1.0f - alpha) * (far_end * far_end);
    dtd->error_energy = alpha * dtd->error_energy + (1.0f - alpha) * (error * error);
    dtd->echo_energy = alpha * dtd->echo_energy + (1.0f - alpha) * (echo_est * echo_est);

    // Geigel DTD criterion
    bool geigel_dt = false;
    if (dtd->current_far_max > 1e-10f) {
        geigel_dt = (abs_near > dtd->threshold * dtd->current_far_max);
    }

    // Energy ratio criterion
    // If error energy >> echo energy, likely double-talk
    bool energy_dt = false;
    if (dtd->echo_energy > 1e-10f) {
        float ratio = dtd->error_energy / (dtd->echo_energy + 1e-10f);
        energy_dt = (ratio > dtd->energy_threshold);
    }

    // Combined decision
    bool detected = geigel_dt || energy_dt;

    // Frame-based hangover management
    dtd->samples_in_frame++;
    if (dtd->samples_in_frame >= dtd->frame_size) {
        dtd->samples_in_frame = 0;

        if (detected) {
            dtd->hangover_count = dtd->hangover_max;
            dtd->dtd_active = true;
        } else if (dtd->hangover_count > 0) {
            dtd->hangover_count--;
            dtd->dtd_active = true;
        } else {
            dtd->dtd_active = false;
        }
    }

    // Update confidence
    if (detected) {
        dtd->confidence = fminf(dtd->confidence + 0.1f, 1.0f);
    } else {
        dtd->confidence = fmaxf(dtd->confidence - 0.02f, 0.0f);
    }

    return dtd->dtd_active;
}

bool dtd_detect_block(DtdEstimator* dtd,
                      const float* near_end,
                      const float* far_end,
                      const float* error,
                      const float* echo_est,
                      int num_samples) {
    bool any_detected = false;

    for (int n = 0; n < num_samples; n++) {
        bool det = dtd_detect_sample(dtd, near_end[n], far_end[n],
                                     error[n], echo_est[n]);
        any_detected = any_detected || det;
    }

    return dtd->dtd_active;
}

bool dtd_is_active(const DtdEstimator* dtd) {
    return dtd ? dtd->dtd_active : false;
}

float dtd_get_confidence(const DtdEstimator* dtd) {
    return dtd ? dtd->confidence : 0.0f;
}

void dtd_set_threshold(DtdEstimator* dtd, float threshold) {
    if (dtd && threshold > 0) {
        dtd->threshold = threshold;
    }
}

float dtd_get_far_end_level(const DtdEstimator* dtd) {
    return dtd ? sqrtf(dtd->far_energy) : 0.0f;
}
