/**
 * aec.c - Acoustic Echo Cancellation main implementation
 *
 * PBFDAF (Partitioned Block Frequency Domain Adaptive Filter) with:
 *   - WebRTC-style divergence detection (confidence-based mu scaling)
 *   - Output limiter (output never exceeds mic amplitude)
 *   - RES post-filter (optional)
 * Streaming architecture with hop-size based processing.
 */

#include "aec.h"
#include "subband_nlms.h"
#include "res_filter.h"
#include "fft_wrapper.h"
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
    FftHandle* res_fft;         // FFT for RES IFFT (pre-allocated)
    Complex* res_echo_spec;     // [n_freqs]
    Complex* res_error_spec;    // [n_freqs]
    Complex* res_output_spec;   // [n_freqs]
    float* res_temp;            // [block_size] IFFT scratch

    // Shadow filter (dual-filter divergence control)
    SubbandNlms* shadow_filter;
    float* shadow_output;       // [hop_size]
    float main_err_smooth;
    float shadow_err_smooth;

    // Buffers
    float* output_buffer;   // [hop_size]

    // ERLE estimation
    float near_power;
    float error_power;
    float alpha_erle;

    // Divergence detection state
    float dtd_confidence;   // [0.0, 1.0]
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

    // Create shadow filter (dual-filter divergence control)
    if (config->enable_shadow) {
        float shadow_mu = config->mu * config->shadow_mu_ratio;
        aec->shadow_filter = subband_nlms_create(
            aec->params.block_size,
            aec->params.n_partitions,
            shadow_mu,
            config->delta
        );
        if (aec->shadow_filter) {
            aec->shadow_output = (float*)calloc(aec->params.hop_size, sizeof(float));
            if (!aec->shadow_output) {
                subband_nlms_destroy(aec->shadow_filter);
                aec->shadow_filter = NULL;
            }
        }
        aec->main_err_smooth = 0.0f;
        aec->shadow_err_smooth = 0.0f;
    }

    // Create RES post-filter (optional)
    if (config->enable_res) {
        aec->res = res_create(
            aec->params.n_freqs,
            config->res_g_min_db,
            config->res_over_sub,
            config->res_alpha
        );
        if (aec->res) {
            int n_freqs = aec->params.n_freqs;
            int block_size = aec->params.block_size;
            aec->res_fft = fft_create(block_size);
            aec->res_echo_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_error_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_output_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_temp = (float*)calloc(block_size, sizeof(float));
            if (!aec->res_fft || !aec->res_echo_spec || !aec->res_error_spec ||
                !aec->res_output_spec || !aec->res_temp) {
                // RES allocation failed — disable RES gracefully
                fft_destroy(aec->res_fft);
                free(aec->res_echo_spec);
                free(aec->res_error_spec);
                free(aec->res_output_spec);
                free(aec->res_temp);
                res_destroy(aec->res);
                aec->res = NULL;
                aec->res_fft = NULL;
                aec->res_echo_spec = NULL;
                aec->res_error_spec = NULL;
                aec->res_output_spec = NULL;
                aec->res_temp = NULL;
            }
        }
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
    aec->dtd_confidence = 0.0f;
    aec->frame_count = 0;

    return aec;
}

void aec_destroy(Aec* aec) {
    if (aec) {
        subband_nlms_destroy(aec->filter);
        subband_nlms_destroy(aec->shadow_filter);
        free(aec->shadow_output);
        res_destroy(aec->res);
        fft_destroy(aec->res_fft);
        free(aec->res_echo_spec);
        free(aec->res_error_spec);
        free(aec->res_output_spec);
        free(aec->res_temp);
        free(aec->output_buffer);
        free(aec);
    }
}

void aec_reset(Aec* aec) {
    if (!aec) return;

    if (aec->filter) {
        subband_nlms_reset(aec->filter);
    }
    if (aec->shadow_filter) {
        subband_nlms_reset(aec->shadow_filter);
        aec->main_err_smooth = 0.0f;
        aec->shadow_err_smooth = 0.0f;
    }
    if (aec->res) {
        res_reset(aec->res);
    }

    int hop_size = aec->params.hop_size;
    memset(aec->output_buffer, 0, hop_size * sizeof(float));
    if (aec->shadow_output) {
        memset(aec->shadow_output, 0, hop_size * sizeof(float));
    }

    aec->near_power = 0.0f;
    aec->error_power = 0.0f;
    aec->dtd_confidence = 0.0f;
    aec->frame_count = 0;
}

void aec_retrain(Aec* aec) {
    if (!aec) return;

    // Reset weights only — keep X_buf, power, overlap buffers
    if (aec->filter) {
        subband_nlms_reset_weights(aec->filter);
    }
    if (aec->shadow_filter) {
        subband_nlms_reset_weights(aec->shadow_filter);
        aec->main_err_smooth = 0.0f;
        aec->shadow_err_smooth = 0.0f;
    }

    // Restart DTD warmup
    aec->dtd_confidence = 0.0f;
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

    // Compute mu_scale from previous block's divergence confidence
    float mu_scale = 1.0f;
    if (aec->config.enable_dtd) {
        float conf = aec->dtd_confidence;
        float min_r = aec->config.dtd_mu_min_ratio;
        mu_scale = 1.0f - conf * (1.0f - min_r);
    }

    // Process through PBFDAF
    subband_nlms_process(aec->filter, near_end, far_end, output, mu_scale);

    // Shadow filter: always adapts with full step, copy to main if better
    if (aec->shadow_filter) {
        subband_nlms_process(aec->shadow_filter, near_end, far_end,
                             aec->shadow_output, 1.0f);

        float main_err = subband_nlms_get_error_energy(aec->filter);
        float shadow_err = subband_nlms_get_error_energy(aec->shadow_filter);

        float alpha_s = aec->config.shadow_err_alpha;
        aec->main_err_smooth = alpha_s * aec->main_err_smooth +
                               (1.0f - alpha_s) * main_err;
        aec->shadow_err_smooth = alpha_s * aec->shadow_err_smooth +
                                 (1.0f - alpha_s) * shadow_err;

        if (aec->shadow_err_smooth <
            aec->main_err_smooth * aec->config.shadow_copy_threshold) {
            // Shadow is better — copy weights to main
            subband_nlms_copy_weights(aec->filter, aec->shadow_filter);
            memcpy(output, aec->shadow_output, hop_size * sizeof(float));
            aec->main_err_smooth = aec->shadow_err_smooth;
        }
    }

    // RES post-filter: suppress residual echo in frequency domain
    if (aec->res) {
        subband_nlms_get_echo_spectrum(aec->filter, aec->res_echo_spec);
        subband_nlms_get_error_spectrum(aec->filter, aec->res_error_spec);

        // Compute far-end power for activity detection
        float far_power = 0.0f;
        for (int n = 0; n < hop_size; n++) {
            far_power += far_end[n] * far_end[n];
        }
        far_power /= hop_size;

        // Apply RES spectral suppression
        res_process(aec->res, aec->res_error_spec, aec->res_echo_spec,
                    far_power, aec->res_output_spec);

        // IFFT back to time domain, overlap-save: take last hop_size samples
        fft_inverse(aec->res_fft, aec->res_output_spec, aec->res_temp);
        for (int n = 0; n < hop_size; n++) {
            output[n] = aec->res_temp[hop_size + n];
        }
    }

    // Update divergence detection BEFORE limiter (so DTD sees true output)
    aec->frame_count++;
    if (aec->config.enable_dtd &&
        aec->frame_count >= aec->config.dtd_warmup_frames) {

        float output_energy = 0.0f;
        float near_energy = 0.0f;
        float near_peak = 0.0f;
        float out_peak = 0.0f;

        for (int n = 0; n < hop_size; n++) {
            output_energy += output[n] * output[n];
            near_energy += near_end[n] * near_end[n];
            float abs_near = fabsf(near_end[n]);
            float abs_out = fabsf(output[n]);
            if (abs_near > near_peak) near_peak = abs_near;
            if (abs_out > out_peak) out_peak = abs_out;
        }
        output_energy /= hop_size;
        near_energy /= hop_size;

        // Check both energy and peak divergence
        float energy_ratio = (near_energy > 1e-10f) ?
            output_energy / (near_energy + 1e-10f) : 0.0f;
        float peak_ratio = (near_peak > 1e-6f) ?
            out_peak / (near_peak + 1e-10f) : 0.0f;
        float ratio = (energy_ratio > peak_ratio) ? energy_ratio : peak_ratio;

        float attack = aec->config.dtd_confidence_attack;
        float release = aec->config.dtd_confidence_release;
        float div_factor = aec->config.dtd_divergence_factor;

        if (near_energy < 1e-10f && near_peak < 1e-6f) {
            // Silence
            aec->dtd_confidence = fmaxf(aec->dtd_confidence - release, 0.0f);
        } else if (ratio > div_factor) {
            // Severe divergence
            aec->dtd_confidence = fminf(aec->dtd_confidence + attack, 1.0f);
        } else if (ratio > 1.2f) {
            // Mild divergence (threshold 1.2: ratio < 1.2 is normal unconverged state)
            aec->dtd_confidence = fminf(
                aec->dtd_confidence + attack * (ratio - 1.2f), 1.0f);
        } else {
            // Normal — faster release when ratio is well below 1.0
            float release_scale = (1.0f - ratio > 0.2f) ? (1.0f - ratio) : 0.2f;
            aec->dtd_confidence = fmaxf(
                aec->dtd_confidence - release * (1.0f + 4.0f * release_scale), 0.0f);
        }
    }

    // Output limiter: output should never exceed mic amplitude
    {
        float near_peak = 0.0f;
        float out_peak = 0.0f;
        for (int n = 0; n < hop_size; n++) {
            float abs_near = fabsf(near_end[n]);
            float abs_out = fabsf(output[n]);
            if (abs_near > near_peak) near_peak = abs_near;
            if (abs_out > out_peak) out_peak = abs_out;
        }
        if (out_peak > near_peak && near_peak > 1e-6f) {
            float scale = near_peak / out_peak;
            for (int n = 0; n < hop_size; n++) {
                output[n] *= scale;
            }
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
    return aec->dtd_confidence > 0.5f;
}

float aec_get_dtd_confidence(const Aec* aec) {
    return aec ? aec->dtd_confidence : 0.0f;
}

const AecConfig* aec_get_config(const Aec* aec) {
    return aec ? &aec->config : NULL;
}
