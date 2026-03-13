/**
 * aec.h - Acoustic Echo Cancellation main interface
 *
 * PBFDAF-based AEC with divergence detection and RES post-filter.
 *
 * Usage:
 *   AecConfig cfg = aec_default_config(16000);
 *   Aec* aec = aec_create(&cfg);
 *
 *   int hop = aec_get_hop_size(aec);
 *   while (has_audio) {
 *       aec_process(aec, mic_in, ref_in, output);
 *   }
 *
 *   aec_destroy(aec);
 */

#ifndef AEC_H
#define AEC_H

#include "aec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Opaque AEC handle
typedef struct Aec Aec;

/**
 * Create AEC instance
 *
 * @param config Configuration structure
 * @return AEC handle, or NULL on error
 */
Aec* aec_create(const AecConfig* config);

/**
 * Destroy AEC instance
 */
void aec_destroy(Aec* aec);

/**
 * Reset AEC state (clear filter weights, buffers, DTD)
 * Full reset — equivalent to destroy + create with same config.
 */
void aec_reset(Aec* aec);

/**
 * Retrain AEC filter — clear weights only, keep buffers.
 *
 * Use when echo path changes (e.g., device/room switch).
 * Preserves reference history (X_buf) for faster reconvergence.
 * Restarts DTD warmup.
 */
void aec_retrain(Aec* aec);

/**
 * Process hop_size samples through AEC
 *
 * @param aec AEC handle
 * @param near_end Microphone input [hop_size]
 * @param far_end Reference/loudspeaker signal [hop_size]
 * @param output Echo-cancelled output [hop_size]
 * @return 0 on success, negative on error
 */
int aec_process(Aec* aec,
                const float* near_end,
                const float* far_end,
                float* output);

/**
 * Get hop size (number of samples per process call)
 */
int aec_get_hop_size(const Aec* aec);

/**
 * Get frame size (internal processing frame)
 */
int aec_get_frame_size(const Aec* aec);

/**
 * Get processing latency in samples
 */
int aec_get_latency(const Aec* aec);

/**
 * Get current Echo Return Loss Enhancement (ERLE) estimate in dB
 * Higher is better (typical: 10-30 dB)
 */
float aec_get_erle(const Aec* aec);

/**
 * Check if double-talk/divergence is currently detected
 */
bool aec_is_dtd_active(const Aec* aec);

/**
 * Get DTD confidence (0.0 = normal, 1.0 = fully diverged)
 */
float aec_get_dtd_confidence(const Aec* aec);

/**
 * Get configuration (for inspection)
 */
const AecConfig* aec_get_config(const Aec* aec);

#ifdef __cplusplus
}
#endif

#endif // AEC_H
