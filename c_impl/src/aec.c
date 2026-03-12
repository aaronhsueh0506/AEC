/**
 * aec.c - Acoustic Echo Cancellation main implementation
 *
 * PBFDAF (Partitioned Block Frequency Domain Adaptive Filter) with:
 *   - Error-based DTD with warmup and holdover
 *   - RES post-filter (optional)
 * Streaming architecture with hop-size based processing.
 */

#include "aec.h"
#include "subband_nlms.h"
#include "res_filter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct Aec {
    AecConfig config;
    AecDerivedParams params;

    // PBFDAF adaptive filter
    SubbandNlms* filter;

    // RES post-filter (optional)
    ResFilter* res;

    // Buffers
    float* output_buffer;   // [hop_size]

    // ERLE estimation
    float near_power;
    float error_power;
    float alpha_erle;

    // Error-based DTD state
    bool dtd_active;
    float dtd_ratio_smooth;
    int frame_count;
};

Aec* aec_create(const AecConfig* config) {
    if (!config) return NULL;

    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec) return NULL;

    aec->config = *config;
    aec->params = aec_compute_params(config);

    // Create PBFDAF filter
    aec->filter = subband_nlms_create(
        aec->params.block_size,
        aec->params.n_partitions,
        config->mu,
        config->delta
    );
    if (!aec->filter) {
        aec_destroy(aec);
        return NULL;
    }

    // Create RES post-filter (optional)
    if (config->enable_res) {
        aec->res = res_create(
            aec->params.n_freqs,
            config->res_g_min_db,
            config->res_over_sub,
            config->res_alpha
        );
        // Non-fatal if RES creation fails
    }

    // Allocate buffers
    aec->output_buffer = (float*)calloc(aec->params.hop_size, sizeof(float));
    if (!aec->output_buffer) {
        aec_destroy(aec);
        return NULL;
    }

    // Init ERLE
    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
    aec->alpha_erle = 0.95f;

    // Init DTD
    aec->dtd_active = false;
    aec->dtd_ratio_smooth = 0.0f;
    aec->frame_count = 0;

    return aec;
}

void aec_destroy(Aec* aec) {
    if (aec) {
        subband_nlms_destroy(aec->filter);
        res_destroy(aec->res);
        free(aec->output_buffer);
        free(aec);
    }
}

void aec_reset(Aec* aec) {
    if (!aec) return;

    if (aec->filter) {
        subband_nlms_reset(aec->filter);
    }
    if (aec->res) {
        res_reset(aec->res);
    }

    int hop_size = aec->params.hop_size;
    memset(aec->output_buffer, 0, hop_size * sizeof(float));

    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
    aec->dtd_active = false;
    aec->dtd_ratio_smooth = 0.0f;
    aec->frame_count = 0;
}

void aec_retrain(Aec* aec) {
    if (!aec) return;

    // Reset weights only — keep X_buf, power, overlap buffers
    if (aec->filter) {
        subband_nlms_reset_weights(aec->filter);
    }

    // Restart DTD warmup
    aec->dtd_active = false;
    aec->dtd_ratio_smooth = 0.0f;
    aec->frame_count = 0;

    // Reset ERLE
    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
}

int aec_process(Aec* aec,
                const float* near_end,
                const float* far_end,
                float* output) {
    if (!aec || !near_end || !far_end || !output) {
        return -1;
    }

    const int hop_size = aec->params.hop_size;
    const float alpha = aec->alpha_erle;

    // DTD: freeze weights during double-talk
    bool update_weights = true;
    if (aec->config.enable_dtd && aec->dtd_active) {
        update_weights = false;
    }

    // Process through PBFDAF
    subband_nlms_process(aec->filter, near_end, far_end, output, update_weights);

    // TODO: RES post-filter (requires spectrum access refactoring)

    // Update error-based DTD state for NEXT block
    aec->frame_count++;
    if (aec->config.enable_dtd &&
        aec->frame_count >= aec->config.dtd_warmup_frames) {

        float error_energy = 0.0f;
        float echo_energy = 0.0f;
        float near_energy = 0.0f;

        for (int n = 0; n < hop_size; n++) {
            error_energy += output[n] * output[n];
            near_energy += near_end[n] * near_end[n];
            float echo_est = near_end[n] - output[n];
            echo_energy += echo_est * echo_est;
        }
        error_energy /= hop_size;
        near_energy /= hop_size;
        echo_energy /= hop_size;

        // Convergence guard + holdover
        if (echo_energy > near_energy * 0.1f && echo_energy > 1e-10f) {
            // Normal DTD: filter has converged, check error/echo ratio
            float ratio = error_energy / (echo_energy + 1e-10f);
            aec->dtd_ratio_smooth = 0.95f * aec->dtd_ratio_smooth + 0.05f * ratio;
            aec->dtd_active = (aec->dtd_ratio_smooth > aec->config.dtd_threshold);
        } else if (aec->dtd_active && near_energy > 1e-6f) {
            // Holdover: far-end stopped but near-end still active.
            // Keep DTD active to prevent weight corruption from residual X_buf.
            // (do nothing — dtd_active stays true)
        } else {
            // Filter not converged yet — keep DTD inactive
            aec->dtd_ratio_smooth *= 0.9f;
            aec->dtd_active = false;
        }
    }

    // Update ERLE estimation
    for (int n = 0; n < hop_size; n++) {
        aec->near_power = alpha * aec->near_power +
                         (1.0f - alpha) * (near_end[n] * near_end[n]);
        aec->error_power = alpha * aec->error_power +
                          (1.0f - alpha) * (output[n] * output[n]);
    }

    return 0;
}

int aec_get_hop_size(const Aec* aec) {
    return aec ? aec->params.hop_size : 0;
}

int aec_get_frame_size(const Aec* aec) {
    return aec ? aec->params.frame_size : 0;
}

int aec_get_latency(const Aec* aec) {
    return aec ? aec->params.hop_size : 0;
}

float aec_get_erle(const Aec* aec) {
    if (!aec) return 0.0f;

    const float eps = 1e-10f;
    if (aec->near_power < eps && aec->error_power < eps) {
        return 0.0f;
    }

    return 10.0f * log10f((aec->near_power + eps) / (aec->error_power + eps));
}

bool aec_is_dtd_active(const Aec* aec) {
    if (!aec) return false;
    return aec->dtd_active;
}

const AecConfig* aec_get_config(const Aec* aec) {
    return aec ? &aec->config : NULL;
}
