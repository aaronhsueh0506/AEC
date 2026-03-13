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

    // RES post-filter (optional, uses overlap-save)
    ResFilter* res;
    FftHandle* res_fft;         // FFT for RES IFFT (pre-allocated)
    Complex* res_echo_spec;     // [n_freqs]
    Complex* res_near_spec;     // [n_freqs] near-end spectrum for overlap-save
    Complex* res_residual_spec; // [n_freqs] residual = near - echo
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
    float dtd_confidence;       // [0.0, 1.0] divergence confidence
    int frame_count;

    // Coherence-based DT detection state
    float dtd_coh_confidence;   // [0.0, 1.0] coherence confidence
    float* S_ex_r;              // [n_freqs] smoothed cross-PSD real
    float* S_ex_i;              // [n_freqs] smoothed cross-PSD imag
    float* S_ee;                // [n_freqs] smoothed error auto-PSD
    float* S_xx;                // [n_freqs] smoothed far-end auto-PSD
    Complex* coh_far_spec;      // [n_freqs] temp buffer for far spectrum
    Complex* coh_error_spec;    // [n_freqs] temp buffer for error spectrum
    int coh_hangover_count;
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
            aec->res_near_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_residual_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_output_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
            aec->res_temp = (float*)calloc(block_size, sizeof(float));
            if (!aec->res_fft || !aec->res_echo_spec || !aec->res_near_spec ||
                !aec->res_residual_spec || !aec->res_output_spec || !aec->res_temp) {
                // RES allocation failed — disable RES gracefully
                fft_destroy(aec->res_fft);
                free(aec->res_echo_spec);
                free(aec->res_near_spec);
                free(aec->res_residual_spec);
                free(aec->res_output_spec);
                free(aec->res_temp);
                res_destroy(aec->res);
                aec->res = NULL;
                aec->res_fft = NULL;
                aec->res_echo_spec = NULL;
                aec->res_near_spec = NULL;
                aec->res_residual_spec = NULL;
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

    // Init coherence DTD
    aec->dtd_coh_confidence = 0.0f;
    aec->coh_hangover_count = 0;
    if (config->enable_dtd) {
        int n_freqs = aec->params.n_freqs;
        aec->S_ex_r = (float*)calloc(n_freqs, sizeof(float));
        aec->S_ex_i = (float*)calloc(n_freqs, sizeof(float));
        aec->S_ee = (float*)calloc(n_freqs, sizeof(float));
        aec->S_xx = (float*)calloc(n_freqs, sizeof(float));
        aec->coh_far_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
        aec->coh_error_spec = (Complex*)calloc(n_freqs, sizeof(Complex));
        if (!aec->S_ex_r || !aec->S_ex_i || !aec->S_ee || !aec->S_xx ||
            !aec->coh_far_spec || !aec->coh_error_spec) {
            free(aec->S_ex_r); free(aec->S_ex_i);
            free(aec->S_ee); free(aec->S_xx);
            free(aec->coh_far_spec); free(aec->coh_error_spec);
            aec->S_ex_r = NULL; aec->S_ex_i = NULL;
            aec->S_ee = NULL; aec->S_xx = NULL;
            aec->coh_far_spec = NULL; aec->coh_error_spec = NULL;
        }
    }

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
        free(aec->res_near_spec);
        free(aec->res_residual_spec);
        free(aec->res_output_spec);
        free(aec->res_temp);
        free(aec->output_buffer);
        free(aec->S_ex_r);
        free(aec->S_ex_i);
        free(aec->S_ee);
        free(aec->S_xx);
        free(aec->coh_far_spec);
        free(aec->coh_error_spec);
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
    aec->dtd_coh_confidence = 0.0f;
    aec->coh_hangover_count = 0;
    aec->frame_count = 0;
    if (aec->S_ex_r) {
        int n = aec->params.n_freqs;
        memset(aec->S_ex_r, 0, n * sizeof(float));
        memset(aec->S_ex_i, 0, n * sizeof(float));
        memset(aec->S_ee, 0, n * sizeof(float));
        memset(aec->S_xx, 0, n * sizeof(float));
    }
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

    // Compute mu_scale from combined confidence (divergence + coherence)
    float mu_scale = 1.0f;
    if (aec->config.enable_dtd) {
        float conf_div = aec->dtd_confidence;
        float conf_coh = aec->dtd_coh_confidence;
        float conf = (conf_div > conf_coh) ? conf_div : conf_coh;
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
            // Shadow is better — copy weights and echo spectrum to main
            subband_nlms_copy_weights(aec->filter, aec->shadow_filter);
            subband_nlms_copy_echo_spec(aec->filter, aec->shadow_filter);
            memcpy(output, aec->shadow_output, hop_size * sizeof(float));
            aec->main_err_smooth = aec->shadow_err_smooth;
        }
    }

    // RES post-filter: suppress residual echo using overlap-save
    // Apply RES gain to full-block residual spectrum (near_spec - echo_spec),
    // then IFFT and take last hop_size samples. This inherits the overlap-save
    // framework from PBFDAF, avoiding block boundary artifacts.
    if (aec->res) {
        int nfreq = aec->params.n_freqs;
        subband_nlms_get_echo_spectrum(aec->filter, aec->res_echo_spec);
        subband_nlms_get_near_spectrum(aec->filter, aec->res_near_spec);

        // Compute residual spectrum: near_spec - echo_spec
        for (int k = 0; k < nfreq; k++) {
            aec->res_residual_spec[k].r = aec->res_near_spec[k].r - aec->res_echo_spec[k].r;
            aec->res_residual_spec[k].i = aec->res_near_spec[k].i - aec->res_echo_spec[k].i;
        }

        // Compute far-end power for activity detection
        float far_power = 0.0f;
        for (int n = 0; n < hop_size; n++) {
            far_power += far_end[n] * far_end[n];
        }
        far_power /= hop_size;

        // Apply RES spectral suppression to residual
        res_process(aec->res, aec->res_residual_spec, aec->res_echo_spec,
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

    // Coherence-based double-talk detection
    if (aec->config.enable_dtd && aec->S_ex_r &&
        aec->frame_count >= aec->config.dtd_warmup_frames) {

        int nfreq = aec->params.n_freqs;
        float coh_alpha = aec->config.dtd_coh_alpha;
        float attack = aec->config.dtd_confidence_attack;
        float release = aec->config.dtd_coh_release;

        subband_nlms_get_error_spectrum(aec->filter, aec->coh_error_spec);
        subband_nlms_get_far_spectrum(aec->filter, aec->coh_far_spec);

        float sum_num = 0.0f, sum_den = 0.0f;
        float sum_ee = 0.0f, sum_xx = 0.0f;

        for (int k = 0; k < nfreq; k++) {
            float er = aec->coh_error_spec[k].r, ei = aec->coh_error_spec[k].i;
            float xr = aec->coh_far_spec[k].r, xi = aec->coh_far_spec[k].i;

            // Cross-PSD: error × conj(far)
            float cx_r = er * xr + ei * xi;
            float cx_i = ei * xr - er * xi;

            aec->S_ex_r[k] = coh_alpha * aec->S_ex_r[k] + (1.0f - coh_alpha) * cx_r;
            aec->S_ex_i[k] = coh_alpha * aec->S_ex_i[k] + (1.0f - coh_alpha) * cx_i;
            aec->S_ee[k] = coh_alpha * aec->S_ee[k] + (1.0f - coh_alpha) * (er*er + ei*ei);
            aec->S_xx[k] = coh_alpha * aec->S_xx[k] + (1.0f - coh_alpha) * (xr*xr + xi*xi);

            sum_num += aec->S_ex_r[k] * aec->S_ex_r[k] + aec->S_ex_i[k] * aec->S_ex_i[k];
            sum_den += aec->S_ee[k] * aec->S_xx[k];
            sum_ee += aec->S_ee[k];
            sum_xx += aec->S_xx[k];
        }

        float coherence = sum_num / (sum_den + 1e-10f);
        int has_energy = (sum_ee > aec->config.dtd_coh_energy_floor * sum_xx) && (sum_xx > 1e-10f);

        if (coherence > aec->config.dtd_coh_high) {
            // Correlated → not DT → release
            if (aec->coh_hangover_count > 0) {
                aec->coh_hangover_count--;
                aec->dtd_coh_confidence = fmaxf(aec->dtd_coh_confidence - release * 0.5f, 0.0f);
            } else {
                aec->dtd_coh_confidence = fmaxf(aec->dtd_coh_confidence - release, 0.0f);
            }
        } else if (coherence < aec->config.dtd_coh_low && has_energy) {
            // Uncorrelated + energy → DT
            aec->coh_hangover_count = aec->config.dtd_coh_hangover;
            aec->dtd_coh_confidence = fminf(aec->dtd_coh_confidence + attack, 1.0f);
        } else {
            // Ambiguous → slow release
            aec->dtd_coh_confidence = fmaxf(aec->dtd_coh_confidence - release * 0.5f, 0.0f);
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
