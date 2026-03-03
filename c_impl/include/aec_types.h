/**
 * aec_types.h - Common types and configuration for AEC
 *
 * Acoustic Echo Cancellation Implementation
 * Based on NLMS adaptive filter with DTD and RES post-filter
 */

#ifndef AEC_TYPES_H
#define AEC_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * AEC filter mode selection
 *
 * Three modes with different trade-offs:
 *   - TIME:    Time-domain NLMS, sample-by-sample, lowest latency
 *   - FREQ:    Frequency-domain NLMS, single FFT block, no partitions
 *   - SUBBAND: Partitioned block FDAF (PBFDAF), for long echo paths
 */
typedef enum {
    AEC_MODE_TIME,      // Time-domain NLMS (default, lowest latency)
    AEC_MODE_FREQ,      // Frequency-domain NLMS (single block, no partitions)
    AEC_MODE_SUBBAND    // Partitioned block FDAF (PBFDAF, for long echo paths)
} AecFilterMode;

/**
 * Configuration structure for AEC
 */
typedef struct {
    int sample_rate;            // Sample rate (16000 Hz)
    int frame_size_ms;          // Frame length in ms (20)
    int frame_shift_ms;         // Frame shift in ms (10) = hop size for NLMS mode
    int fft_size;               // FFT size (512)

    // Filter mode selection
    AecFilterMode filter_mode;  // AEC_MODE_TIME, AEC_MODE_FREQ, or AEC_MODE_SUBBAND

    // NLMS adaptive filter parameters
    int filter_length_ms;       // Filter length in ms (250)
    float mu;                   // Step size (0.3)
    float delta;                // Regularization (1e-8)
    float leak;                 // Weight leakage factor (0.9999, NLMS only)

    // DTD (Double-Talk Detection) parameters
    bool enable_dtd;            // Enable DTD (true)
    float dtd_threshold;        // Geigel threshold (0.6)
    int dtd_hangover_frames;    // Hangover duration in frames (15)
    float dtd_energy_ratio;     // Energy ratio threshold (0.4)

    // RES (Residual Echo Suppressor) parameters
    bool enable_res;            // Enable post-filter (true)
    float res_g_min_db;         // Minimum gain in dB (-20)
    float res_over_sub;         // Over-subtraction factor (1.5)
    float res_alpha;            // Gain smoothing (0.8)

} AecConfig;

/**
 * Create default configuration for given sample rate
 * FFT size is automatically calculated to be >= frame_size (next power of 2)
 *
 * Note: hop_size depends on filter_mode:
 *   - AEC_MODE_TIME:    hop = sample_rate * frame_shift_ms / 1000 (e.g., 160 @ 16kHz)
 *   - AEC_MODE_FREQ:    hop = fft_size / 2 (e.g., 256 @ fft_size=512)
 *   - AEC_MODE_SUBBAND: hop = fft_size / 2 (same as FREQ)
 */
static inline AecConfig aec_default_config(int sample_rate) {
    AecConfig config;

    config.sample_rate = sample_rate;
    config.frame_size_ms = 20;
    config.frame_shift_ms = 10;

    // Calculate FFT size (next power of 2 >= frame_size)
    int frame_size = sample_rate * config.frame_size_ms / 1000;
    int fft_size = 256;
    while (fft_size < frame_size) {
        fft_size *= 2;
    }
    config.fft_size = fft_size;

    // Filter mode: default to time-domain NLMS (lowest latency)
    config.filter_mode = AEC_MODE_TIME;

    // NLMS parameters (tuned for 200-300ms echo path)
    config.filter_length_ms = 250;
    config.mu = 0.3f;
    config.delta = 1e-8f;
    config.leak = 0.9999f;

    // DTD parameters
    config.enable_dtd = true;
    config.dtd_threshold = 0.6f;
    config.dtd_hangover_frames = 15;
    config.dtd_energy_ratio = 0.4f;

    // RES parameters
    config.enable_res = true;
    config.res_g_min_db = -20.0f;
    config.res_over_sub = 1.5f;
    config.res_alpha = 0.8f;

    return config;
}

/**
 * Derived parameters (computed from config)
 */
typedef struct {
    int frame_size;         // Samples per frame
    int hop_size;           // Samples per hop (mode-dependent)
    int filter_length;      // Filter length in samples
    int n_freqs;            // Number of frequency bins (fft_size/2 + 1)
    int n_partitions;       // Number of filter partitions (subband mode)
} AecDerivedParams;

/**
 * Compute derived parameters from configuration
 *
 * Note: hop_size is mode-dependent:
 *   - TIME mode:    hop = sample_rate * frame_shift_ms / 1000
 *   - FREQ mode:    hop = fft_size / 2, n_partitions = 1
 *   - SUBBAND mode: hop = fft_size / 2, n_partitions = ceil(filter_length / hop)
 */
static inline AecDerivedParams aec_compute_params(const AecConfig* config) {
    AecDerivedParams params;
    params.frame_size = config->sample_rate * config->frame_size_ms / 1000;
    params.filter_length = config->sample_rate * config->filter_length_ms / 1000;
    params.n_freqs = config->fft_size / 2 + 1;

    switch (config->filter_mode) {
        case AEC_MODE_FREQ:
            // Frequency-domain NLMS: single FFT block, no partitioning
            params.hop_size = config->fft_size / 2;
            params.n_partitions = 1;
            break;

        case AEC_MODE_SUBBAND:
            // Partitioned block FDAF: multiple partitions for long echo paths
            params.hop_size = config->fft_size / 2;
            params.n_partitions = (params.filter_length + params.hop_size - 1) / params.hop_size;
            break;

        case AEC_MODE_TIME:
        default:
            // Time-domain NLMS: use configured frame shift
            params.hop_size = config->sample_rate * config->frame_shift_ms / 1000;
            params.n_partitions = 0;  // Not used in time-domain mode
            break;
    }

    return params;
}

#ifdef __cplusplus
}
#endif

#endif // AEC_TYPES_H
