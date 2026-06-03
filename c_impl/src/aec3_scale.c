/* aec3_scale.c — C port of python/modules/aec3_scale.py conversion helpers.
 *
 * Every function mirrors the Python semantics bit-for-bit, including the
 * round-half-to-even behaviour of Python's int(round(.)) via lrint() under the
 * default FE_TONEAREST rounding mode. Built with -ffp-contract=off (Makefile),
 * so no fused multiply-add reorders the float ops away from the reference.
 */
#include "aec3_scale.h"

#include <math.h>

double aec3_psd_int16_to_float(double value) {
    return value * AEC3_PSD_SCALE_INV;
}

int aec3_blocks_to_hops(int blocks, int hop_samples, int sample_rate) {
    double seconds, hops;
    if (blocks <= 0) {
        return 0;
    }
    seconds = blocks * (double)AEC3_BLOCK_SAMPLES_16K / 16000.0;
    hops = (double)lrint(seconds * sample_rate / hop_samples);
    return hops < 1.0 ? 1 : (int)hops;
}

int aec3_ms_to_hops(double ms, int hop_samples, int sample_rate) {
    double hops;
    if (ms <= 0.0) {
        return 0;
    }
    hops = (double)lrint(ms * 1e-3 * sample_rate / hop_samples);
    return hops < 1.0 ? 1 : (int)hops;
}

double aec3_per_block_rate_to_per_hop(double per_block_rate, int hop_samples,
                                      int sample_rate) {
    double block_seconds = (double)AEC3_BLOCK_SAMPLES_16K / 16000.0;
    double hop_seconds = hop_samples / (double)sample_rate;
    return per_block_rate * (hop_seconds / block_seconds);
}

double aec3_per_block_ema_alpha_to_per_hop(double per_block_alpha,
                                           int hop_samples, int sample_rate) {
    double block_seconds, hop_seconds;
    if (per_block_alpha <= 0.0) {
        return 0.0;
    }
    if (per_block_alpha >= 1.0) {
        return 1.0;
    }
    block_seconds = (double)AEC3_BLOCK_SAMPLES_16K / 16000.0;
    hop_seconds = hop_samples / (double)sample_rate;
    return 1.0 - pow(1.0 - per_block_alpha, hop_seconds / block_seconds);
}

double aec3_fft_density_scale(double value_int16sq, int fft_size) {
    /* caller guarantees fft_size > 0 (Python raises; C is a release path) */
    return value_int16sq * (((fft_size / 2)) / (double)AEC3_FFT_LENGTH_BY_2);
}

double aec3_per_bin_psd_threshold(double calibrated_value, int hop_size,
                                  int ref_hop) {
    return calibrated_value * hop_size / ref_hop;
}

double aec3_nl_r2_norm_power(int hop_size, int ref_hop) {
    const double calibrated_at_ref = 1.07e7; /* tuned at hop=160; do not change */
    return calibrated_at_ref * hop_size / ref_hop;
}

double aec3_block_energy_scale(double value_int16sq, int hop_samples) {
    return value_int16sq * (hop_samples / (double)AEC3_BLOCK_SAMPLES_16K);
}

double aec3_per_block_growth_to_per_hop(double per_block_multiplier,
                                        int hop_samples, int sample_rate) {
    double block_seconds = (double)AEC3_BLOCK_SAMPLES_16K / 16000.0;
    double hop_seconds = hop_samples / (double)sample_rate;
    return pow(per_block_multiplier, hop_seconds / block_seconds);
}
