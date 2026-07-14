/* reverb_frequency_response.h — C port of
 * python/modules/residual/reverb_frequency_response.py
 * (strict line port of AEC3 reverb_frequency_response.cc:74-106).
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
 * FLOAT32-BY-DESIGN NOTES (numpy 1.26 -> C, captured from real balanced/DT case):
 *  - frequency_response is float32 (n_partitions x n_freqs); rows are float32.
 *  - The two energy reductions are numpy `np.sum` over a float32 slice, which
 *    uses PAIRWISE summation accumulated in float32 (NOT naive left-to-right).
 *    Replicated bit-exactly by f32_pairwise_sum().
 *  - average_decay = tail_energy / direct_energy is computed in float32
 *    throughout (formerly widened via python `float()` of each f32 sum, then
 *    f64 divide; converted for uniformity, drift accepted). _average_decay and
 *    `smoothing` are float32.
 *  - tail = freq_resp_direct(f32) * average_decay(f32): each store is
 *    f32_elem * average_decay, both float32.
 *  - 0.5*(tail[k-1]+tail[k+1]) is all-float32 (0.5f, exact).
 *  Built with -ffp-contract=off so the f32 roundings are not fused.
 */
#ifndef REVERB_FREQUENCY_RESPONSE_H
#define REVERB_FREQUENCY_RESPONSE_H

#include <stddef.h>

typedef struct {
    int    n_freqs;            /* spectrum length (e.g. 257) */
    int    use_conservative;   /* point-wise max with raw_tail */
    float  smoothing_base;     /* 0.2, or wall-clock-rescaled value */
    float  average_decay;      /* EMA scalar state (float32) */
    float *tail_response;      /* owned by caller storage; length n_freqs */
} ReverbFrequencyResponse;

/* tail_storage must point to n_freqs floats; ownership stays with caller. */
void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           float smoothing_base);

void reverb_freq_resp_reset(ReverbFrequencyResponse *r);

/* Numpy-1.26-bit-exact pairwise sum of a float32 array (block <=128, unrolled
 * by 8, recursive split). Exposed for the parity test. Returns a float. */
float f32_pairwise_sum(const float *a, size_t n);

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
                             float linear_filter_quality,
                             int linear_filter_quality_is_none,
                             int stationary_block);

#endif /* REVERB_FREQUENCY_RESPONSE_H */
