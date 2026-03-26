/**
 * aec.h - Acoustic Echo Cancellation main interface
 *
 * PBFDKF (Kalman always on) + Shadow + RES (WOLA) + HPF + Preset
 * Matches Python AEC v1.15.0 (PBFDKF mode).
 *
 * Usage:
 *   AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
 *   Aec* aec = aec_create(&cfg);
 *   int hop = aec_get_hop_size(aec);
 *   while (has_audio) {
 *       aec_process(aec, mic, ref, output);
 *   }
 *   aec_destroy(aec);
 */

#ifndef AEC_H
#define AEC_H

#include "aec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Aec Aec;

/**
 * Create AEC instance from config
 */
Aec* aec_create(const AecConfig* config);

/**
 * Destroy AEC instance
 */
void aec_destroy(Aec* aec);

/**
 * Reset all state
 */
void aec_reset(Aec* aec);

/**
 * Process hop_size samples
 *
 * @param aec      AEC handle
 * @param near_end Microphone input [hop_size]
 * @param far_end  Reference signal [hop_size]
 * @param output   Echo-cancelled output [hop_size]
 * @return 0 on success
 */
int aec_process(Aec* aec,
                const float* near_end,
                const float* far_end,
                float* output);

/* --- Getters --- */
int   aec_get_hop_size(const Aec* aec);
float aec_get_erle(const Aec* aec);
float aec_get_erle_instant(const Aec* aec);
int   aec_is_converged(const Aec* aec);
const AecConfig* aec_get_config(const Aec* aec);

#ifdef __cplusplus
}
#endif

#endif /* AEC_H */
