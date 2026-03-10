/**
 * aec.c - Acoustic Echo Cancellation main implementation
 *
 * Orchestrates adaptive filter with DTD protection.
 * Supports three filter modes:
 *   - NLMS:    Time-domain NLMS (sample-by-sample)
 *   - FREQ:    Frequency-domain NLMS (single FFT block)
 *   - SUBBAND: Partitioned block FDAF (PBFDAF)
 * Streaming architecture with hop-size based processing.
 */

#include "aec.h"
#include "nlms_filter.h"
#include "subband_nlms.h"
#include "dtd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct Aec {
    AecConfig config;
    AecDerivedParams params;

    // Sub-modules (one of these is active based on filter_mode)
    NlmsFilter* nlms;           // Time-domain NLMS (AEC_MODE_NLMS)
    SubbandNlms* subband;       // Frequency-domain NLMS (AEC_MODE_FREQ or AEC_MODE_SUBBAND)
    DtdEstimator* dtd;

    // Buffers for block processing
    float* near_buffer;     // [hop_size]
    float* far_buffer;      // [hop_size]
    float* output_buffer;   // [hop_size]
    float* echo_est_buffer; // [hop_size]

    // ERLE estimation
    float erle_smoothed;
    float near_power;
    float error_power;
    float alpha_erle;

    // Error-based DTD state
    bool dtd_active;            // Current DTD decision (for next block)
    float dtd_ratio_smooth;     // Smoothed error/echo ratio
};

Aec* aec_create(const AecConfig* config) {
    if (!config) return NULL;

    Aec* aec = (Aec*)calloc(1, sizeof(Aec));
    if (!aec) return NULL;

    // Copy configuration
    aec->config = *config;
    aec->params = aec_compute_params(config);

    // Create adaptive filter based on mode
    switch (config->filter_mode) {
        case AEC_MODE_FREQ:
        case AEC_MODE_SUBBAND:
            // Frequency-domain NLMS (FREQ uses n_partitions=1, SUBBAND uses multiple)
            aec->subband = subband_nlms_create(
                config->fft_size,
                aec->params.n_partitions,
                config->mu,
                config->delta
            );
            if (!aec->subband) {
                aec_destroy(aec);
                return NULL;
            }
            aec->nlms = NULL;
            break;

        case AEC_MODE_LMS:
            // Time-domain LMS (no normalization, fixed step size)
            aec->nlms = nlms_create(
                aec->params.filter_length,
                config->mu,
                config->delta,
                1.0f,   // leak=1.0 for LMS (no weight decay)
                false   // normalize=false -> LMS
            );
            if (!aec->nlms) {
                aec_destroy(aec);
                return NULL;
            }
            aec->subband = NULL;
            break;

        case AEC_MODE_NLMS:
        default:
            // Time-domain NLMS (sample-by-sample)
            aec->nlms = nlms_create(
                aec->params.filter_length,
                config->mu,
                config->delta,
                config->leak,
                true    // normalize=true -> NLMS
            );
            if (!aec->nlms) {
                aec_destroy(aec);
                return NULL;
            }
            aec->subband = NULL;
            break;
    }

    // Apply clear_filter_history setting for TIME/LMS
    if (aec->nlms) {
        nlms_set_clear_history(aec->nlms, config->clear_filter_history);
    }

    // Create DTD (if enabled)
    if (config->enable_dtd) {
        // DTD window = filter length for Geigel
        aec->dtd = dtd_create(
            aec->params.filter_length,
            config->dtd_threshold,
            config->dtd_hangover_frames,
            config->dtd_energy_ratio
        );
        if (!aec->dtd) {
            aec_destroy(aec);
            return NULL;
        }
    }

    // Allocate buffers
    int hop_size = aec->params.hop_size;
    aec->near_buffer = (float*)calloc(hop_size, sizeof(float));
    aec->far_buffer = (float*)calloc(hop_size, sizeof(float));
    aec->output_buffer = (float*)calloc(hop_size, sizeof(float));
    aec->echo_est_buffer = (float*)calloc(hop_size, sizeof(float));

    if (!aec->near_buffer || !aec->far_buffer ||
        !aec->output_buffer || !aec->echo_est_buffer) {
        aec_destroy(aec);
        return NULL;
    }

    // ERLE estimation init
    aec->erle_smoothed = 0.0f;
    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
    aec->alpha_erle = 0.95f;

    // Error-based DTD init
    aec->dtd_active = false;
    aec->dtd_ratio_smooth = 0.0f;

    return aec;
}

void aec_destroy(Aec* aec) {
    if (aec) {
        nlms_destroy(aec->nlms);
        subband_nlms_destroy(aec->subband);
        dtd_destroy(aec->dtd);
        free(aec->near_buffer);
        free(aec->far_buffer);
        free(aec->output_buffer);
        free(aec->echo_est_buffer);
        free(aec);
    }
}

void aec_reset(Aec* aec) {
    if (!aec) return;

    if (aec->nlms) {
        nlms_reset(aec->nlms);
    }
    if (aec->subband) {
        subband_nlms_reset(aec->subband);
    }
    if (aec->dtd) {
        dtd_reset(aec->dtd);
    }

    int hop_size = aec->params.hop_size;
    memset(aec->near_buffer, 0, hop_size * sizeof(float));
    memset(aec->far_buffer, 0, hop_size * sizeof(float));
    memset(aec->output_buffer, 0, hop_size * sizeof(float));
    memset(aec->echo_est_buffer, 0, hop_size * sizeof(float));

    aec->erle_smoothed = 0.0f;
    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
    aec->dtd_active = false;
    aec->dtd_ratio_smooth = 0.0f;
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

    // Error-based DTD: use PREVIOUS block's decision for current block.
    // Old Geigel DTD compared mic vs ref, but mic includes echo, causing
    // false triggers when echo_gain > threshold.
    // New approach: compare error energy vs echo estimate energy AFTER processing.
    bool update_weights = true;
    if (aec->config.enable_dtd && aec->dtd_active) {
        if (aec->config.filter_mode == AEC_MODE_FREQ ||
            aec->config.filter_mode == AEC_MODE_SUBBAND) {
            // Subband: freeze weights during double-talk
            update_weights = false;
        }
        // NLMS/LMS: soft DTD — reduce mu (handled below)
    }

    // Process block through adaptive filter (mode-dependent)
    float saved_mu = 0.0f;
    switch (aec->config.filter_mode) {
        case AEC_MODE_FREQ:
        case AEC_MODE_SUBBAND:
            subband_nlms_process(aec->subband, near_end, far_end, output, update_weights);
            break;

        case AEC_MODE_LMS:
        case AEC_MODE_NLMS:
        default:
            // Soft DTD for NLMS/LMS: reduce mu during double-talk
            if (aec->config.enable_dtd && aec->dtd_active && aec->nlms) {
                saved_mu = nlms_get_mu(aec->nlms);
                nlms_set_mu(aec->nlms, saved_mu * 0.1f);
            }
            nlms_process_block(aec->nlms, near_end, far_end, output,
                               aec->echo_est_buffer, hop_size, true);
            // Restore mu
            if (saved_mu > 0.0f && aec->nlms) {
                nlms_set_mu(aec->nlms, saved_mu);
            }
            break;
    }

    // Update error-based DTD state for NEXT block
    if (aec->config.enable_dtd) {
        float error_energy = 0.0f;
        float echo_energy = 0.0f;
        float near_energy = 0.0f;

        for (int n = 0; n < hop_size; n++) {
            error_energy += output[n] * output[n];
            near_energy += near_end[n] * near_end[n];
        }
        error_energy /= hop_size;
        near_energy /= hop_size;

        // Compute echo estimate energy based on mode
        if (aec->config.filter_mode == AEC_MODE_FREQ ||
            aec->config.filter_mode == AEC_MODE_SUBBAND) {
            // Use echo estimate from subband filter
            for (int n = 0; n < hop_size; n++) {
                float echo_est = near_end[n] - output[n];
                echo_energy += echo_est * echo_est;
            }
            echo_energy /= hop_size;
        } else {
            // NLMS/LMS: use echo_est_buffer
            for (int n = 0; n < hop_size; n++) {
                echo_energy += aec->echo_est_buffer[n] * aec->echo_est_buffer[n];
            }
            echo_energy /= hop_size;
        }

        // Only update DTD when filter produces meaningful echo estimates
        if (echo_energy > near_energy * 0.01f && echo_energy > 1e-10f) {
            float ratio = error_energy / (echo_energy + 1e-10f);
            aec->dtd_ratio_smooth = 0.8f * aec->dtd_ratio_smooth + 0.2f * ratio;
            aec->dtd_active = (aec->dtd_ratio_smooth > aec->config.dtd_threshold);
        } else {
            // Filter not converged yet — keep DTD inactive
            aec->dtd_ratio_smooth *= 0.9f;
            aec->dtd_active = false;
        }
    }

    // Update detailed DTD estimator (for query functions)
    if (aec->dtd) {
        dtd_detect_block(aec->dtd, near_end, far_end, output,
                         aec->echo_est_buffer, hop_size);
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
    if (!aec) return 0;
    // Latency = hop_size (mode-dependent):
    // TIME:    hop_size (e.g., 160 @ 16kHz = 10ms)
    // FREQ:    fft_size/2 (e.g., 256 @ 16kHz = 16ms)
    // SUBBAND: fft_size/2 (e.g., 256 @ 16kHz = 16ms)
    return aec->params.hop_size;
}

AecFilterMode aec_get_filter_mode(const Aec* aec) {
    return aec ? aec->config.filter_mode : AEC_MODE_NLMS;
}

float aec_get_erle(const Aec* aec) {
    if (!aec || aec->error_power < 1e-10f) {
        return 0.0f;
    }

    // ERLE = 10 * log10(near_power / error_power)
    float erle_linear = aec->near_power / (aec->error_power + 1e-10f);
    float erle_db = 10.0f * log10f(erle_linear + 1e-10f);

    return erle_db;
}

bool aec_is_dtd_active(const Aec* aec) {
    if (!aec) return false;
    return aec->dtd_active;
}

const AecConfig* aec_get_config(const Aec* aec) {
    return aec ? &aec->config : NULL;
}
