/* aec3_scale.c — C port of python/modules/aec3_scale.py conversion helpers.
 *
 * Every function mirrors the Python semantics structurally, including the
 * round-half-to-even behaviour of Python's int(round(.)) via lrintf() under
 * the default FE_TONEAREST rounding mode. Float32-by-design (init-only
 * helpers; converted for uniformity as part of the f32 campaign, drift
 * accepted — formerly fp64 to mirror Python exactly). Built with
 * -ffp-contract=off (Makefile), so no fused multiply-add reorders the float
 * ops away from the reference.
 */
#include "aec3_scale.h"

#include <math.h>

float aec3_psd_int16_to_float(float value) {
    return value * (float)AEC3_PSD_SCALE_INV;
}

int aec3_blocks_to_hops(int blocks, int hop_samples, int sample_rate) {
    float seconds, hops;
    if (blocks <= 0) {
        return 0;
    }
    seconds = blocks * (float)AEC3_BLOCK_SAMPLES_16K / 16000.0f;
    hops = (float)lrintf(seconds * (float)sample_rate / (float)hop_samples);
    return hops < 1.0f ? 1 : (int)hops;
}

int aec3_ms_to_hops(float ms, int hop_samples, int sample_rate) {
    float hops;
    if (ms <= 0.0f) {
        return 0;
    }
    hops = (float)lrintf(ms * 1e-3f * (float)sample_rate / (float)hop_samples);
    return hops < 1.0f ? 1 : (int)hops;
}

float aec3_per_block_rate_to_per_hop(float per_block_rate, int hop_samples,
                                     int sample_rate) {
    float block_seconds = (float)AEC3_BLOCK_SAMPLES_16K / 16000.0f;
    float hop_seconds = (float)hop_samples / (float)sample_rate;
    return per_block_rate * (hop_seconds / block_seconds);
}

float aec3_per_block_ema_alpha_to_per_hop(float per_block_alpha,
                                          int hop_samples, int sample_rate) {
    float block_seconds, hop_seconds;
    if (per_block_alpha <= 0.0f) {
        return 0.0f;
    }
    if (per_block_alpha >= 1.0f) {
        return 1.0f;
    }
    block_seconds = (float)AEC3_BLOCK_SAMPLES_16K / 16000.0f;
    hop_seconds = (float)hop_samples / (float)sample_rate;
    return 1.0f - powf(1.0f - per_block_alpha, hop_seconds / block_seconds);
}

float aec3_fft_density_scale(float value_int16sq, int fft_size) {
    /* caller guarantees fft_size > 0 (Python raises; C is a release path) */
    return value_int16sq * (((fft_size / 2)) / (float)AEC3_FFT_LENGTH_BY_2);
}

float aec3_per_bin_psd_threshold(float calibrated_value, int hop_size,
                                 int ref_hop) {
    return calibrated_value * hop_size / ref_hop;
}

float aec3_nl_r2_norm_power(int hop_size, int ref_hop) {
    const float calibrated_at_ref = 1.07e7f; /* tuned at hop=160; do not change */
    return calibrated_at_ref * hop_size / ref_hop;
}

float aec3_block_energy_scale(float value_int16sq, int hop_samples) {
    return value_int16sq * ((float)hop_samples / (float)AEC3_BLOCK_SAMPLES_16K);
}

float aec3_per_block_growth_to_per_hop(float per_block_multiplier,
                                       int hop_samples, int sample_rate) {
    float block_seconds = (float)AEC3_BLOCK_SAMPLES_16K / 16000.0f;
    float hop_seconds = (float)hop_samples / (float)sample_rate;
    return powf(per_block_multiplier, hop_seconds / block_seconds);
}

float aec3_growth_rehop(float retention_at_ref, int ref_hop_samples,
                        int ref_sample_rate, int hop_samples,
                        int sample_rate) {
    float ref_seconds, hop_seconds;
    /* Guard here rather than at each call site: this is the single funnel every
     * retimed constant passes through, and a non-positive rate/hop would make
     * hop_seconds Inf or negative, so powf() returns 0 or NaN and silently
     * poisons an EMA that is only ever read, never validated. Falling back to
     * the authoring value is the one well-defined answer -- it degrades to "not
     * retimed" (the pre-2026-08 behaviour) instead of to a dead filter.
     *
     * The top-level AecConfig validator already rejects these, so this is
     * unreachable through aec_create(); it exists for the direct pbfdaf_init()/
     * pbfdkf_init()/saturation_init() entry points, which are public API and
     * are called by tests and by out-of-tree integrations. saturation_init()
     * in particular returns void and has no other way to refuse. */
    if (sample_rate <= 0 || hop_samples <= 0 ||
        ref_sample_rate <= 0 || ref_hop_samples <= 0) {
        return retention_at_ref;
    }
    ref_seconds = (float)ref_hop_samples / (float)ref_sample_rate;
    hop_seconds = (float)hop_samples / (float)sample_rate;
    return powf(retention_at_ref, hop_seconds / ref_seconds);
}
