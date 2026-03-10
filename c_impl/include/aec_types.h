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
 * Four modes with different trade-offs:
 *   - NLMS:    Time-domain NLMS, sample-by-sample
 *   - FREQ:    FDAF, single FFT block (buffered if filter > hop)
 *   - SUBBAND: Partitioned block FDAF (PBFDAF), for long echo paths
 *   - LMS:     Time-domain LMS, fixed step size, simplest
 */
typedef enum {
    AEC_MODE_NLMS,      // Time-domain NLMS (default)
    AEC_MODE_FREQ,      // FDAF: single FFT block (buffered if filter > hop)
    AEC_MODE_SUBBAND,   // Partitioned block FDAF (PBFDAF, for long echo paths)
    AEC_MODE_LMS        // Time-domain LMS (no normalization, simplest)
} AecFilterMode;

/**
 * Configuration structure for AEC
 *
 * All sizes are in samples (not ms).
 */
typedef struct {
    int sample_rate;            // Sample rate (16000 Hz)
    int frame_size;             // Frame length in samples (512 @ 16kHz)
    int hop_size;               // Hop size in samples (256 @ 16kHz)
    int fft_size;               // FFT size (512, = frame_size)

    // Filter mode selection
    AecFilterMode filter_mode;  // AEC_MODE_NLMS, AEC_MODE_FREQ, AEC_MODE_SUBBAND, AEC_MODE_LMS

    // Adaptive filter parameters
    int filter_length;          // Filter length in samples (512 @ 16kHz for TIME/LMS)
    float mu;                   // Step size (0.3 for NLMS, 0.01 for LMS)
    float delta;                // Regularization (1e-8)
    float leak;                 // Weight leakage factor (0.9999, NLMS only)

    // DTD (Double-Talk Detection) parameters
    bool enable_dtd;            // Enable DTD (true)
    float dtd_threshold;        // Error-based DTD: error/echo ratio (2.0)
    int dtd_hangover_frames;    // Hangover duration in frames (15)
    float dtd_energy_ratio;     // Energy ratio threshold (0.4)

    // RES (Residual Echo Suppressor) parameters
    bool enable_res;            // Enable post-filter (true)
    float res_g_min_db;         // Minimum gain in dB (-20)
    float res_over_sub;         // Over-subtraction factor (1.5)
    float res_alpha;            // Gain smoothing (0.8)

    // TIME/LMS history control
    bool clear_filter_history;  // Clear ref_buffer each block (default: false)

} AecConfig;

/**
 * Create default configuration for given sample rate
 *
 * frame_size = next power of 2 >= sample_rate * 32ms (= fft_size, no zero-padding)
 * hop_size = frame_size / 2 (50% overlap)
 * filter_length = frame_size (suitable for TIME/LMS; SUBBAND should use larger value)
 */
static inline AecConfig aec_default_config(int sample_rate) {
    AecConfig config;

    config.sample_rate = sample_rate;

    // frame_size = next power of 2 >= ~32ms worth of samples
    int target = sample_rate * 32 / 1000;
    int frame_size = 256;
    while (frame_size < target) {
        frame_size *= 2;
    }
    config.frame_size = frame_size;       // 512 @ 16kHz
    config.hop_size = frame_size / 2;     // 256 @ 16kHz
    config.fft_size = frame_size;         // = frame_size (no zero-padding)

    // Filter mode: default to time-domain NLMS
    config.filter_mode = AEC_MODE_NLMS;

    // Adaptive filter parameters
    config.filter_length = frame_size;    // 512 @ 16kHz (TIME/LMS default)
    config.mu = 0.3f;
    config.delta = 1e-8f;
    config.leak = 0.9999f;

    // DTD parameters
    config.enable_dtd = true;
    config.dtd_threshold = 2.0f;
    config.dtd_hangover_frames = 15;
    config.dtd_energy_ratio = 0.4f;

    // RES parameters
    config.enable_res = true;
    config.res_g_min_db = -20.0f;
    config.res_over_sub = 1.5f;
    config.res_alpha = 0.8f;

    // TIME/LMS: keep 1 hop_size history by default
    config.clear_filter_history = false;

    return config;
}

/**
 * Derived parameters (computed from config)
 */
typedef struct {
    int frame_size;         // = config.frame_size
    int hop_size;           // External hop size (= config.hop_size, e.g. 256)
    int block_size;         // SubbandNlms FFT size (FREQ: 2*filter, SUBBAND: fft_size)
    int internal_hop;       // SubbandNlms hop (= block_size/2)
    int filter_length;      // Effective filter length
    int n_freqs;            // Number of frequency bins (block_size/2 + 1)
    int n_partitions;       // Number of filter partitions (FREQ=1, SUBBAND=N)
} AecDerivedParams;

/**
 * Compute derived parameters from configuration
 *
 * FREQ:    Buffered FDAF — single large FFT, block_size = next_pow2(2*filter_length)
 *          External hop = config.hop_size, internal hop = block_size/2
 * SUBBAND: PBFDAF — multiple partitions, block_size = config.fft_size
 * NLMS/LMS: Time-domain, no SubbandNlms
 */
static inline AecDerivedParams aec_compute_params(const AecConfig* config) {
    AecDerivedParams params;
    params.frame_size = config->frame_size;
    params.hop_size = config->hop_size;

    switch (config->filter_mode) {
        case AEC_MODE_FREQ: {
            // True FDAF: single big FFT block, n_partitions=1
            params.filter_length = config->filter_length;
            params.n_partitions = 1;
            // block_size = next power of 2 >= 2 * filter_length
            int desired = 2 * config->filter_length;
            int bs = 256;
            while (bs < desired) bs *= 2;
            params.block_size = bs;
            params.internal_hop = bs / 2;
            params.n_freqs = bs / 2 + 1;
            break;
        }

        case AEC_MODE_SUBBAND:
            // PBFDAF: partitioned block, configurable filter_length
            params.filter_length = config->filter_length;
            params.n_partitions = (config->filter_length + config->hop_size - 1) / config->hop_size;
            if (params.n_partitions < 1) params.n_partitions = 1;
            params.block_size = config->fft_size;
            params.internal_hop = config->hop_size;
            params.n_freqs = config->fft_size / 2 + 1;
            break;

        case AEC_MODE_LMS:
        case AEC_MODE_NLMS:
        default:
            // Time-domain: no SubbandNlms
            params.filter_length = config->filter_length;
            params.n_partitions = 0;
            params.block_size = 0;
            params.internal_hop = config->hop_size;
            params.n_freqs = 0;
            break;
    }

    return params;
}

#ifdef __cplusplus
}
#endif

#endif // AEC_TYPES_H
