/**
 * pbfdkf_v381.h - PBFDKF v3.8.1 (matches Python aec.py PBFDKF, lines 741-910)
 *
 * Partitioned Block Frequency-Domain Kalman Filter with G1 KX-blended P-update.
 * Bit-exact parity to Python within float32 vs float64 numerical drift.
 *
 * Key v3.x fixes vs legacy (c_impl_v25_legacy/include/pbfdkf.h):
 * - v3.6.0: filter_length default 32 → 52ms (caller-side; this header agnostic)
 * - v3.7.0 G1: P-update uses blended KX = mu_mean × KX_optimal +
 *              (1 - mu_mean) × KX_scaled. Decouples P from W during DT-mu-scaling.
 *
 * Float32 enforcement (CRITICAL for parity):
 * - All real arithmetic: float32 (NOT float / double)
 * - Complex: kiss_fft_cpx (interleaved float32 pairs)
 * - delta cast to float32 explicitly before use in denominators
 * - Reduction order: must match np.sum / np.matmul ordering
 */

#ifndef PBFDKF_V381_H
#define PBFDKF_V381_H

#include <stdint.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PbfdkfV381 PbfdkfV381;

/**
 * Create PBFDKF instance.
 *
 * @param block_size      FFT length (e.g., 512 @ 16kHz, 320 frame_size + 192 zero-pad)
 * @param hop_size        Processing hop (e.g., 160 @ 16kHz)
 * @param n_freqs         Number of frequency bins (block_size/2 + 1, e.g., 257)
 * @param n_partitions    Filter length / hop_size (e.g., 5 @ 16kHz/52ms)
 * @param mu              Base learning rate (e.g., 0.3, used by R adaptation only here)
 * @param delta           Regularization (1e-8f) — cast float32 internally
 * @param q_high          Process noise high (1e-4f balanced)
 * @param q_low           Process noise low (1e-6f balanced; reserved)
 * @return Heap-allocated handle, NULL on failure.
 */
PbfdkfV381* pbfdkf_v381_create(int block_size, int hop_size, int n_freqs,
                                int n_partitions,
                                float mu, float delta,
                                float q_high, float q_low);

/**
 * Destroy instance.
 */
void pbfdkf_v381_destroy(PbfdkfV381* f);

/**
 * Reset all state. Equivalent to Python PBFDKF.reset():
 *   parent reset (W, X_buf, buffers, power, partition_idx, error_spec_windowed)
 *   P.fill(0.01), R.fill(1e-2), _error_psd.fill(1e-2), Q[:] = Q_high
 *   delete optional EPC overrides (_p_max_override etc.)
 */
void pbfdkf_v381_reset(PbfdkfV381* f);

/**
 * Process one hop: equivalent to Python PBFDAF.process(near, far, mu_scale).
 *
 * Steps (must match Python lines 650-710):
 *   1. Buffer shift: near_buf, far_buf shift by hop, append new
 *   2. FFT: near_buf → near_spec (n_freqs); far_buf → far_spec
 *   3. Store far_spec into X_buf[partition_idx]; advance partition_idx
 *   4. Update power EMA: power = α × power + (1-α) × |far_spec|², α=0.9
 *   5. Echo estimation (partitioned convolution):
 *        echo_spec = sum over p: W[p] × X_buf[(partition_idx - p) mod N_part]
 *   6. IFFT(echo_spec) → echo_time; output = near_buf[hop:block] - echo_time[hop:block]
 *   7. Compute error_spec from windowed near minus echo_spec
 *   8. If sum(far_end²)/hop > 1e-4 (~ -40 dBFS): call _update_weights(mu_scale)
 *   9. Return output[hop_size]
 *
 * @param mu_scale        Either NULL (broadcast scalar 1.0) or float32[n_freqs]
 * @param mu_scale_scalar Used when mu_scale==NULL; ignored otherwise
 *                        (pass 1.0f for full update, 0.0f to freeze, etc.)
 * @param output          Caller-allocated [hop_size] for echo-cancelled output
 */
void pbfdkf_v381_process(PbfdkfV381* f,
                          const float* near_end,
                          const float* far_end,
                          const float* mu_scale,        /* NULL = use scalar */
                          float mu_scale_scalar,
                          float* output);

/**
 * Get error spectrum (last process call). Pointer into internal state, valid
 * until next process(). Length = n_freqs Complex pairs.
 */
const Complex* pbfdkf_v381_get_error_spec(const PbfdkfV381* f);

/**
 * Get echo spectrum (last process call). For external RES.
 */
const Complex* pbfdkf_v381_get_echo_spec(const PbfdkfV381* f);

/**
 * Get near spectrum (last process call), windowed (sqrt-Hann).
 */
const Complex* pbfdkf_v381_get_near_spec(const PbfdkfV381* f);

/**
 * Get far spectrum (last process call).
 */
const Complex* pbfdkf_v381_get_far_spec(const PbfdkfV381* f);

/**
 * Energy of last error_spec: sum(|error_spec|²). Used for ERLE / shadow advantage.
 */
float pbfdkf_v381_get_error_energy(const PbfdkfV381* f);

/**
 * Get far_power EMA (for AEC orchestrator).
 */
const float* pbfdkf_v381_get_far_power(const PbfdkfV381* f);

/**
 * EPC overrides: set p_max ceiling and p_floor beta for next N frames.
 * (Equivalent to Python: f._p_max_override, f._p_max_override_frames, etc.)
 */
void pbfdkf_v381_set_p_max_override(PbfdkfV381* f, float p_max, int frames);
void pbfdkf_v381_set_p_floor_beta(PbfdkfV381* f, float beta, int frames);

/**
 * Q schedule: set Q[:] = q_high or q_low (called by EPC handler).
 */
void pbfdkf_v381_set_q_high(PbfdkfV381* f);
void pbfdkf_v381_set_q_low(PbfdkfV381* f);

/**
 * Copy weights (W only, NOT P/R) from another PBFDKF — for shadow → main copy.
 * Python equivalent: PBFDKF.copy_weights_from(src) at line 901-909.
 */
void pbfdkf_v381_copy_weights_from(PbfdkfV381* dst, const PbfdkfV381* src);

#ifdef __cplusplus
}
#endif

#endif /* PBFDKF_V381_H */
