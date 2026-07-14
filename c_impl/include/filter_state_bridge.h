/* filter_state_bridge.h — C port of
 * python/modules/filter/filter_state_bridge.py.
 *
 * Thin Kalman->state adapter. The orchestrator builds one snapshot per hop
 * from the linear-filter family (PBFDKF main, optional Shadow filter,
 * PathChangeRegimeHandler) and hands it to AecState. This port carries the
 * LOGIC only: it consumes synthetic filter-state inputs (W partition 0, the
 * Kalman P matrix, the two error energies, a few flags) and produces the
 * derived snapshot fields — it does NOT depend on the real PBFDKF struct.
 *
 * Parity-relevant computations (numpy 1.26 -> C, -ffp-contract=off):
 *   1. filter_taps = np.fft.irfft(W[0], fft_size).astype(np.float32)
 *        - numpy irfft of a complex64 input computes in float64 (pocketfft),
 *          then .astype(float32) truncates. fft_inverse (fft_pocketfft.c) wraps
 *          the vendored numpy-1.26.4 pocketfft, so it is BIT-EXACT vs np.fft.irfft.
 *   2. p_trace = np.mean(P)      [P is float32]
 *        - np.mean over a float32 array = float32 PAIRWISE sum / n (in f32),
 *          kept in float32 (float32-by-design; formerly widened via float()).
 *   3. divergence_indicator:
 *        e_ratio = 1.0 (default); if a shadow is present and main_e > 1e-12,
 *                  e_ratio = shadow_e / main_e   (float32-by-design; main_e /
 *                  shadow_e are already-provided float32 energies).
 *        div = p_trace * max(0.0, e_ratio - 1.0)     (all float32)
 *        div = 0 when P is empty/missing.
 *   4. regime_code = 1 if main_paused else 0.
 *
 * The remaining fields (filter_converged, main_paused, mu_final,
 * external_delay_samples, any_coarse_filter_converged, all_filters_diverged)
 * are pass-through bool/float/int with the same casts as the Python dataclass.
 */
#ifndef FILTER_STATE_BRIDGE_H
#define FILTER_STATE_BRIDGE_H

#include <stddef.h>

#include "fft_wrapper.h"  /* Complex, FftHandle */

/* Snapshot mirroring the Python FilterStateBridge dataclass. The caller owns
 * `filter_taps` storage (length fft_size). */
typedef struct {
    int    filter_converged;       /* bool */
    float *filter_taps;            /* owned by caller; length fft_size (f32)  */
    int    filter_taps_len;        /* fft_size, or 1 when W absent            */
    float  divergence_indicator;   /* float32 scalar                          */
    int    regime;                 /* 0 stable, 1 main_paused                 */
    int    main_paused;            /* bool                                    */
    float  mu_final;               /* float32 scalar                          */
    int    external_delay_samples; /* int                                     */
    int    any_coarse_filter_converged; /* bool                              */
    int    all_filters_diverged;        /* bool                              */
} FilterStateBridge;

/* float32 pairwise sum (numpy 1.26 semantics), accumulated in float32.
 * Exposed for the parity harness; shared with the .c. */
float fsb_f32_pairwise_sum(const float *a, size_t n);

/* np.mean over a flattened float32 array: pairwise sum / n, computed and kept
 * in float32 (float32-by-design; formerly widened to double via
 * float(np.mean(P))). n == 0 -> 0.0 (caller-side: empty P yields div = 0). */
float fsb_f32_mean(const float *a, size_t n);

/* NOTE on error energies: the bridge consumes main_e / shadow_e as float32
 * energies (the np.abs/np.sum over error_spec lives inside the PBFDKF/Shadow
 * modules' get_error_energy(), not in this adapter). The bridge's own
 * arithmetic is only e_ratio = shadow_e / main_e, matched in float32
 * (float32-by-design; formerly fp64). */

/* Build the snapshot.
 *
 *   fft    : FftHandle sized to fft_size (for the irfft); may be NULL only if
 *            W is NULL (W absent path -> filter_taps = {0}, len 1).
 *   W0     : partition-0 weights, complex64, length n_freqs. NULL => W absent.
 *   fft_size, n_freqs : irfft geometry (n_freqs == fft_size/2 + 1).
 *   P      : Kalman P matrix flattened (float32), length p_len. NULL/0 => div 0.
 *   has_shadow : 1 if a shadow filter participates in the e_ratio.
 *   main_e, shadow_e : error energies (float32); used only when has_shadow.
 *   taps_out : caller storage, length >= fft_size (or >= 1 when W absent).
 *
 * Pure read of inputs; writes only `out` and `taps_out`. */
void filter_state_bridge_build(FilterStateBridge *out,
                               FftHandle *fft,
                               const Complex *W0, int fft_size, int n_freqs,
                               const float *P, size_t p_len,
                               int has_shadow, float main_e, float shadow_e,
                               int filter_converged,
                               int main_paused,
                               float mu_final,
                               int external_delay_samples,
                               int any_coarse_filter_converged,
                               int all_filters_diverged,
                               float *taps_out);

#endif /* FILTER_STATE_BRIDGE_H */
