/* reverb_frequency_response.c -- see reverb_frequency_response.h. */
#include "reverb_frequency_response.h"

#include "utility/simd_aec.h"

#include <string.h>

void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           double smoothing_base) {
    r->n_freqs = n_freqs;
    r->use_conservative = use_conservative ? 1 : 0;
    r->smoothing_base = smoothing_base;
    r->average_decay = 0.0;
    r->tail_response = tail_storage;
    if (r->tail_response) memset(r->tail_response, 0, (size_t)n_freqs * sizeof(float));
}

void reverb_freq_resp_reset(ReverbFrequencyResponse *r) {
    r->average_decay = 0.0;
    if (r->tail_response) memset(r->tail_response, 0, (size_t)r->n_freqs * sizeof(float));
}

void reverb_freq_resp_update(ReverbFrequencyResponse *r,
                             const float *frequency_response,
                             int n_partitions,
                             int filter_delay_blocks,
                             double linear_filter_quality,
                             int linear_filter_quality_is_none,
                             int stationary_block) {
    const int N = r->n_freqs;
    const float *direct;
    const float *tail_row;
    float direct_energy_f, tail_energy_f;
    double direct_energy, average_decay, smoothing;
    float decay_f; /* average_decay narrowed to float32 at the tail multiply */
    float *tail = r->tail_response;
    int k;

    /* Skip on a stationary block or no quality estimate. */
    if (stationary_block || linear_filter_quality_is_none) {
        return;
    }
    if (filter_delay_blocks < 0 || filter_delay_blocks >= n_partitions) {
        return;
    }

    direct   = frequency_response + (size_t)filter_delay_blocks * (size_t)N;
    tail_row = frequency_response + (size_t)(n_partitions - 1) * (size_t)N;

    /* Average decay over k >= 1 (float32 pairwise sums widened to double, division and EMA in
     * double). */
    direct_energy_f = sk_pairwise_sum_tailfold_b_f32(direct + 1, (size_t)(N - 1));
    direct_energy = (double)direct_energy_f;
    if (direct_energy == 0.0) {
        average_decay = 0.0;
    } else {
        tail_energy_f = sk_pairwise_sum_tailfold_b_f32(tail_row + 1, (size_t)(N - 1));
        average_decay = (double)tail_energy_f / direct_energy;
    }

    /* EMA: smoothing = smoothing_base * quality, state kept in double. */
    smoothing = r->smoothing_base * linear_filter_quality;
    r->average_decay += smoothing * (average_decay - r->average_decay);

    /* Per-bin tail = direct * average_decay, with the decay narrowed to float32 once. */
    decay_f = (float)r->average_decay;
    sk_scale_f32(tail, direct, decay_f, N);

    /* Conservative variant: point-wise max with the last partition. */
    if (r->use_conservative) {
        sk_raise_to_f32(tail, tail_row, N);
    }

    /* Neighbour-max smoothing for k = 1..N-2, ascending in place so a peak spreads rightward. */
    if (N >= 3) {
        int M = N - 2;
        for (k = 1; k <= M; ++k) {
            float avg = 0.5f * (tail[k - 1] + tail[k + 1]);
            if (avg > tail[k]) tail[k] = avg;
        }
    }

}
