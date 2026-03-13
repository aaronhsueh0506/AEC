/**
 * aec_types.h - Common types and configuration for AEC
 *
 * Acoustic Echo Cancellation Implementation
 * Based on PBFDAF (Partitioned Block Frequency Domain Adaptive Filter)
 * with error-based DTD and RES post-filter
 */

#ifndef AEC_TYPES_H
#define AEC_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration structure for AEC
 *
 * All sizes are in samples (not ms).
 * C implementation only supports PBFDAF (SUBBAND) mode.
 */
typedef struct {
    int sample_rate;            // Sample rate (16000 Hz)
    int frame_size;             // Frame length in samples (512 @ 16kHz)
    int hop_size;               // Hop size in samples (256 @ 16kHz)
    int fft_size;               // FFT size (512, = frame_size)

    // Adaptive filter parameters
    int filter_length;          // Filter length in samples (1024 @ 16kHz)
    float mu;                   // Step size (0.3)
    float delta;                // Regularization (1e-8)

    // DTD: WebRTC-style divergence detection
    bool enable_dtd;                // Enable DTD (true)
    float dtd_divergence_factor;    // output > input × factor → diverged (1.5)
    float dtd_mu_min_ratio;         // Minimum mu scale during divergence (0.05)
    float dtd_confidence_attack;    // Confidence ramp-up rate per block (0.3)
    float dtd_confidence_release;   // Confidence ramp-down rate per block (0.05)
    int dtd_warmup_frames;          // Frames to skip DTD at startup (200)

    // Shadow filter (dual-filter divergence control)
    bool enable_shadow;             // Enable shadow filter (false)
    float shadow_mu_ratio;          // Shadow mu = main mu × ratio (0.5)
    float shadow_copy_threshold;    // Copy when shadow_err < main_err × threshold (0.8)
    float shadow_err_alpha;         // Error energy EMA smoothing (0.95)

    // RES (Residual Echo Suppressor) parameters
    bool enable_res;            // Enable post-filter (true)
    float res_g_min_db;         // Minimum gain in dB (-20)
    float res_over_sub;         // Over-subtraction factor (1.5)
    float res_alpha;            // Gain smoothing (0.8)

} AecConfig;

/**
 * Create default configuration for given sample rate
 *
 * frame_size = next power of 2 >= sample_rate * 32ms (= fft_size)
 * hop_size = frame_size / 2 (50% overlap)
 * filter_length = hop_size * 4 (4 partitions, 64ms @ 16kHz)
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

    // Adaptive filter parameters (PBFDAF)
    config.filter_length = config.hop_size * 4;  // 1024 @ 16kHz (4 partitions)
    config.mu = 0.3f;
    config.delta = 1e-8f;

    // DTD parameters (output-vs-input divergence detection)
    config.enable_dtd = true;
    config.dtd_divergence_factor = 1.5f;
    config.dtd_mu_min_ratio = 0.05f;
    config.dtd_confidence_attack = 0.3f;
    config.dtd_confidence_release = 0.05f;
    config.dtd_warmup_frames = 50;

    // Shadow filter (dual-filter)
    config.enable_shadow = false;
    config.shadow_mu_ratio = 0.5f;
    config.shadow_copy_threshold = 0.8f;
    config.shadow_err_alpha = 0.95f;

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
    int frame_size;         // = config.frame_size
    int hop_size;           // = config.hop_size (e.g. 256)
    int block_size;         // = config.fft_size (SubbandNlms FFT size)
    int filter_length;      // = config.filter_length
    int n_freqs;            // = fft_size / 2 + 1
    int n_partitions;       // = ceil(filter_length / hop_size)
} AecDerivedParams;

/**
 * Compute derived parameters from configuration
 */
static inline AecDerivedParams aec_compute_params(const AecConfig* config) {
    AecDerivedParams params;
    params.frame_size = config->frame_size;
    params.hop_size = config->hop_size;
    params.filter_length = config->filter_length;
    params.block_size = config->fft_size;
    params.n_freqs = config->fft_size / 2 + 1;
    params.n_partitions = (config->filter_length + config->hop_size - 1) / config->hop_size;
    if (params.n_partitions < 1) params.n_partitions = 1;
    return params;
}

#ifdef __cplusplus
}
#endif

#endif // AEC_TYPES_H
