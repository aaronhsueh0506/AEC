/**
 * aec.h - Acoustic Echo Cancellation main interface
 *
 * PBFDKF (Kalman always on) + Shadow + RES (WOLA) + HPF + Preset
 * Matches Python AEC v1.28.1 (PBFDKF mode).
 *
 * Usage (standalone):
 *   AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
 *   Aec* aec = aec_create(&cfg);
 *   int hop = aec_get_hop_size(aec);
 *   while (has_audio) {
 *       aec_process(aec, mic, ref, output);
 *   }
 *   aec_destroy(aec);
 *
 * Usage (linear pipeline — external RES):
 *   AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, 16000);
 *   cfg.enable_res = 0;
 *   Aec* aec = aec_create(&cfg);
 *   AecResContext* ctx = aec_context_create(aec);
 *   while (has_audio) {
 *       aec_process_ex(aec, mic, ref, output, ctx);
 *       // ctx now contains echo_spec, far_spec, etc. for RES
 *   }
 *   aec_context_destroy(ctx);
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

/**
 * Process hop_size samples with context output for external RES
 *
 * Same as aec_process(), but also fills ctx with per-frame AEC state
 * needed by an external RES post-filter. Pass ctx=NULL for no context
 * (equivalent to aec_process).
 *
 * @param aec      AEC handle
 * @param near_end Microphone input [hop_size]
 * @param far_end  Reference signal [hop_size]
 * @param output   Echo-cancelled output [hop_size]
 * @param ctx      Context output (or NULL to skip)
 * @return 0 on success
 */
int aec_process_ex(Aec* aec,
                   const float* near_end,
                   const float* far_end,
                   float* output,
                   AecResContext* ctx);

/* --- Context allocation --- */

/**
 * Create AEC context (allocates internal buffers based on n_freqs)
 */
AecResContext* aec_context_create(const Aec* aec);

/**
 * Destroy AEC context
 */
void aec_context_destroy(AecResContext* ctx);

/* --- Getters --- */
int   aec_get_hop_size(const Aec* aec);
float aec_get_erle(const Aec* aec);
float aec_get_erle_instant(const Aec* aec);
int   aec_is_converged(const Aec* aec);
const AecConfig* aec_get_config(const Aec* aec);

/**
 * Get estimated delay in samples (-1 if not yet estimated or disabled)
 */
int   aec_get_delay(const Aec* aec);

#ifdef __cplusplus
}
#endif

#endif /* AEC_H */
