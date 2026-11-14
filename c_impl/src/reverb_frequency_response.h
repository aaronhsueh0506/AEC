/* Per-bin reverb tail frequency response from the per-partition |W|^2 filter response:
 *  direct_path = frequency_response[filter_delay_blocks], raw_tail = frequency_response[last];
 *  average_decay = EMA of (sum raw_tail[k>=1] / sum direct_path[k>=1]) with weight
 *  smoothing_base * linear_filter_quality;
 *  tail[k] = direct_path[k] * average_decay, optionally raised to raw_tail[k] (conservative) and to
 *  the mean of its neighbours (ascending, in place).
 * average_decay and smoothing are double; energy sums are float32 pairwise sums.
 * Python reference: modules/residual/reverb_frequency_response.py. */
#ifndef REVERB_FREQUENCY_RESPONSE_H
#define REVERB_FREQUENCY_RESPONSE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    n_freqs;
    int    use_conservative;
    double smoothing_base;
    double average_decay;
    float *tail_response;
} ReverbFrequencyResponse;

/* tail_storage points to n_freqs floats, owned by the caller. */
void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           double smoothing_base);

void reverb_freq_resp_reset(ReverbFrequencyResponse *r);

/* frequency_response: row-major (n_partitions x n_freqs) float32. filter_delay_blocks: direct-path
 * partition index. linear_filter_quality: confidence in [0, 1];
 * linear_filter_quality_is_none != 0 means no estimate. No-op when stationary_block != 0, quality
 * is none, or filter_delay_blocks is outside [0, n_partitions). */
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

#endif
