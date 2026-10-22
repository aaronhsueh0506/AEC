/* reverb_frequency_response.h — C port of
 * python/modules/residual/reverb_frequency_response.py
 * (strict line port of the reference reverb frequency response).
 *
 * Builds a per-bin reverb tail frequency response from the per-partition |W|²
 * filter frequency response:
 *
 *   direct_path  = frequency_response[filter_delay_blocks]   (|W|² at direct path)
 *   raw_tail     = frequency_response[-1]                    (last partition |W|²)
 *   average_decay = EMA( Σ_{k>=1} raw_tail[k] / Σ_{k>=1} direct_path[k],
 *                        α = smoothing_base * linear_filter_quality )
 *   tail[k]      = direct_path[k] * average_decay
 *   (conservative) tail[k] = max(tail[k], raw_tail[k])
 *   (neighbour)    tail[k] = max(tail[k], 0.5*(tail[k-1]+tail[k+1])) for 1<=k<N-1
 *
 * FLOAT32/FLOAT64 NOTES (numpy 1.26 -> C):
 *  - frequency_response is float32 (n_partitions x n_freqs); rows are float32.
 *  - The two energy reductions are numpy `np.sum` over a float32 slice, which
 *    uses PAIRWISE summation accumulated in float32 (NOT naive left-to-right).
 *    Replicated bit-exactly by sk_pairwise_sum_tailfold_b_f32
 *    (utility/simd_aec.h).
 *  - average_decay is a DOUBLE (float64), matching Python: `_average_decay`
 *    is a plain python float, and `average_decay = tail_energy /
 *    direct_energy` plus the EMA update (`_average_decay += smoothing *
 *    (average_decay - _average_decay)`) both run in float64 -- only
 *    `float(np.sum(...))` widening the float32 pairwise sums into that
 *    arithmetic, never narrowing. `smoothing` is likewise float64.
 *  - tail = freq_resp_direct(f32) * average_decay: Python multiplies a
 *    float32 array by the float64 `_average_decay` scalar, which numpy<2
 *    narrows ONCE (to the nearest float32) before the float32 multiply --
 *    mirrored here by narrowing average_decay to float32 at the point of
 *    that multiply, not before.
 *  - 0.5*(tail[k-1]+tail[k+1]) is all-float32 (0.5f, exact).
 *  Built with -ffp-contract=off so the f32 roundings are not fused.
 */
#ifndef REVERB_FREQUENCY_RESPONSE_H
#define REVERB_FREQUENCY_RESPONSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    n_freqs;            /* spectrum length (e.g. 257) */
    int    use_conservative;   /* point-wise max with raw_tail */
    double smoothing_base;     /* 0.2, or wall-clock-rescaled value. Double,
                                 * not float: the caller's config constant
                                 * (AECB_REE_REVERB_SMOOTHING_BASE) is a
                                 * full-precision double literal, and
                                 * Python keeps `_smoothing_base` as a plain
                                 * (float64) python float forever -- never
                                 * narrowed until the tail multiply. Storing
                                 * it as float32 here would re-narrow an
                                 * already-correctly-rounded double and bias
                                 * every EMA step afterward. */
    double average_decay;      /* EMA scalar state (float64 -- matches
                                 * Python's plain-float _average_decay) */
    float *tail_response;      /* owned by caller storage; length n_freqs */
} ReverbFrequencyResponse;

/* tail_storage must point to n_freqs floats; ownership stays with caller. */
void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           double smoothing_base);

void reverb_freq_resp_reset(ReverbFrequencyResponse *r);

/* update (cc:74-106).
 *   frequency_response : flat row-major (n_partitions x n_freqs) float32.
 *   n_partitions       : number of rows.
 *   filter_delay_blocks: direct-path partition index.
 *   linear_filter_quality : convergence confidence in [0,1]; pass a value < 0
 *                           to signal Python's `None` (skip, mirrors cc:67).
 *   stationary_block   : if non-zero, skip the update.
 * No-op (state unchanged) when stationary, quality==None(<0), or
 * filter_delay_blocks out of [0, n_partitions). */
void reverb_freq_resp_update(ReverbFrequencyResponse *r,
                             const float *frequency_response,
                             int n_partitions,
                             int filter_delay_blocks,
                             double linear_filter_quality,
                             int linear_filter_quality_is_none,
                             int stationary_block);

#ifdef __cplusplus
}
#endif

#endif /* REVERB_FREQUENCY_RESPONSE_H */
