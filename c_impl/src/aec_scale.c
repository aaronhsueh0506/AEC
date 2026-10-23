/* aec_scale.c -- see aec_scale.h. Rounding is round-half-to-even (lrintf). */
#include "aec_scale.h"

#include <math.h>

int aec_blocks_to_hops(int blocks, int hop_samples, int sample_rate) {
    float seconds, hops;
    if (blocks <= 0) {
        return 0;
    }
    seconds = blocks * (float)AEC_BLOCK_SAMPLES_16K / 16000.0f;
    hops = (float)lrintf(seconds * (float)sample_rate / (float)hop_samples);
    return hops < 1.0f ? 1 : (int)hops;
}

int aec_ms_to_hops(float ms, int hop_samples, int sample_rate) {
    float hops;
    if (ms <= 0.0f) {
        return 0;
    }
    hops = (float)lrintf(ms * 1e-3f * (float)sample_rate / (float)hop_samples);
    return hops < 1.0f ? 1 : (int)hops;
}

float aec_per_block_rate_to_per_hop(float per_block_rate, int hop_samples,
                                    int sample_rate) {
    float block_seconds = (float)AEC_BLOCK_SAMPLES_16K / 16000.0f;
    float hop_seconds = (float)hop_samples / (float)sample_rate;
    return per_block_rate * (hop_seconds / block_seconds);
}

float aec_per_block_ema_alpha_to_per_hop(float per_block_alpha,
                                         int hop_samples, int sample_rate) {
    float block_seconds, hop_seconds;
    if (per_block_alpha <= 0.0f) {
        return 0.0f;
    }
    if (per_block_alpha >= 1.0f) {
        return 1.0f;
    }
    block_seconds = (float)AEC_BLOCK_SAMPLES_16K / 16000.0f;
    hop_seconds = (float)hop_samples / (float)sample_rate;
    return 1.0f - powf(1.0f - per_block_alpha, hop_seconds / block_seconds);
}

float aec_per_bin_psd_threshold(float calibrated_value, int hop_size,
                                int ref_hop) {
    return calibrated_value * hop_size / ref_hop;
}

float aec_block_energy_scale(float value_int16sq, int hop_samples) {
    return value_int16sq * ((float)hop_samples / (float)AEC_BLOCK_SAMPLES_16K);
}

float aec_per_block_growth_to_per_hop(double per_block_multiplier,
                                      int hop_samples, int sample_rate) {
    /* double throughout, narrowed once at return (see aec_scale.h). */
    double block_seconds = (double)AEC_BLOCK_SAMPLES_16K / 16000.0;
    double hop_seconds = (double)hop_samples / (double)sample_rate;
    return (float)pow(per_block_multiplier, hop_seconds / block_seconds);
}

double aec_ema_alpha_from_tau_ms(double tau_ms, double hop_ms) {
    return exp(-hop_ms / tau_ms);
}

double aec_db_to_linear(double db) {
    return pow(10.0, db / 10.0);
}

float aec_growth_rehop(float retention_at_ref, int ref_hop_samples,
                       int ref_sample_rate, int hop_samples,
                       int sample_rate) {
    float ref_seconds, hop_seconds;
    /* Single funnel for every retimed constant: a non-positive rate or hop would make powf()
     * return 0 or NaN, so the authoring value is returned (not retimed) instead. */
    if (sample_rate <= 0 || hop_samples <= 0 ||
        ref_sample_rate <= 0 || ref_hop_samples <= 0) {
        return retention_at_ref;
    }
    ref_seconds = (float)ref_hop_samples / (float)ref_sample_rate;
    hop_seconds = (float)hop_samples / (float)sample_rate;
    return powf(retention_at_ref, hop_seconds / ref_seconds);
}
