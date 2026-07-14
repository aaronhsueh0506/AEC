/* reverb_frequency_response.c — C port of reverb_frequency_response.py.
 * See the header for the parity contract. Build with -ffp-contract=off. */
#include "reverb_frequency_response.h"

#include "aec_simd_kernels.h"

#include <string.h>

/* numpy 1.26 pairwise_sum over float32, accumulated in float32. Mirrors
 * numpy/core/src/umath/loops_utils.h.src @pairwise_sum@:
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators, combine as a balanced tree, then add
 *                the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded down to a multiple of 8, recurse,
 *                add the two halves.
 * Every operation is a float32 add (no float64 widening).
 * Body delegates to simd_kernels.h's sk_pairwise_sum_tailfold_b_f32 (this
 * tree is byte-identical, modulo whitespace, to filter_analyzer.c's
 * fa_f32_pairwise_sum — that kernel's `_scalar` twin's verbatim source);
 * local symbol/signature kept so call sites (here and in erl_estimator.c /
 * suppression_gain.c, which declare the same extern symbol) are unchanged. */
float f32_pairwise_sum(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_b_f32(a, n);
}

void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           float smoothing_base) {
    r->n_freqs = n_freqs;
    r->use_conservative = use_conservative ? 1 : 0;
    r->smoothing_base = smoothing_base;
    r->average_decay = 0.0f;
    r->tail_response = tail_storage;
    memset(r->tail_response, 0, (size_t)n_freqs * sizeof(float));
}

void reverb_freq_resp_reset(ReverbFrequencyResponse *r) {
    r->average_decay = 0.0f;
    memset(r->tail_response, 0, (size_t)r->n_freqs * sizeof(float));
}

void reverb_freq_resp_update(ReverbFrequencyResponse *r,
                             const float *frequency_response,
                             int n_partitions,
                             int filter_delay_blocks,
                             float linear_filter_quality,
                             int linear_filter_quality_is_none,
                             int stationary_block) {
    const int N = r->n_freqs;
    const float *direct;   /* frequency_response[filter_delay_blocks] */
    const float *tail_row; /* frequency_response[-1] */
    float tail_energy_f;
    float direct_energy, average_decay, smoothing;
    float decay_f; /* average_decay applied per-bin (float32-by-design) */
    float *tail = r->tail_response;
    int k;

    /* cc:67 — skip on stationary block or quality None. */
    if (stationary_block || linear_filter_quality_is_none) {
        return;
    }
    if (filter_delay_blocks < 0 || filter_delay_blocks >= n_partitions) {
        return;
    }

    direct   = frequency_response + (size_t)filter_delay_blocks * (size_t)N;
    tail_row = frequency_response + (size_t)(n_partitions - 1) * (size_t)N;

    /* Average-decay scalar over k>=1 (kSkipBins=1). np.sum -> f32 pairwise
     * (float32-by-design; formerly widened via python float() -> f64). */
    direct_energy = f32_pairwise_sum(direct + 1, (size_t)(N - 1));
    if (direct_energy == 0.0f) {
        average_decay = 0.0f;
    } else {
        tail_energy_f = f32_pairwise_sum(tail_row + 1, (size_t)(N - 1));
        average_decay = tail_energy_f / direct_energy;
    }

    /* EMA: smoothing = smoothing_base * quality (f32); state update in f32. */
    smoothing = r->smoothing_base * linear_filter_quality;
    r->average_decay += smoothing * (average_decay - r->average_decay);

    /* Per-bin tail = direct(f32) * average_decay(f32), both float32. */
    decay_f = r->average_decay;
    for (k = 0; k < N; ++k) {
        tail[k] = direct[k] * decay_f;
    }

    /* Conservative variant: point-wise max with raw last partition. */
    if (r->use_conservative) {
        for (k = 0; k < N; ++k) {
            float t = tail[k];
            float rt = tail_row[k];
            /* np.maximum: NaN-propagating, picks rt on tie? numpy maximum is
             * commutative for non-NaN; for ties either is identical. Here the
             * real data is non-negative |W|², no NaN. Straight max. */
            tail[k] = (rt > t) ? rt : t;
        }
    }

    /* Neighbour-max smoothing (k=1..N-2), ascending in-place — AEC3 cc:101-105.
     * AEC3 runs ascending so tail[k-1] is the ALREADY-RAISED value from
     * iteration k-1; a strong peak propagates rightward as a >=0.5/bin skirt.
     * Mirror Python fix: `tail[k] = max(tail[k], 0.5*(tail[k-1]+tail[k+1]))`. */
    if (N >= 3) {
        int M = N - 2;
        for (k = 1; k <= M; ++k) {
            float avg = 0.5f * (tail[k - 1] + tail[k + 1]);
            if (avg > tail[k]) tail[k] = avg;
        }
    }

    /* tail.astype(np.float32) is a no-op (already f32); tail_response is tail. */
}
