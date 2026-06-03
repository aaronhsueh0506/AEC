/* reverb_frequency_response.c — C port of reverb_frequency_response.py.
 * See the header for the parity contract. Build with -ffp-contract=off. */
#include "reverb_frequency_response.h"

#include <string.h>

/* numpy 1.26 pairwise_sum over float32, accumulated in float32. Mirrors
 * numpy/core/src/umath/loops_utils.h.src @pairwise_sum@:
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators, combine as a balanced tree, then add
 *                the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded down to a multiple of 8, recurse,
 *                add the two halves.
 * Every operation is a float32 add (no float64 widening). */
float f32_pairwise_sum(const float *a, size_t n) {
    if (n == 0) {
        return 0.0f;
    }
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    }
    if (n <= 128) {
        float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        float r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        float res;
        size_t i;
        for (i = 8; i + 8 <= n; i += 8) {
            r0 = r0 + a[i + 0];
            r1 = r1 + a[i + 1];
            r2 = r2 + a[i + 2];
            r3 = r3 + a[i + 3];
            r4 = r4 + a[i + 4];
            r5 = r5 + a[i + 5];
            r6 = r6 + a[i + 6];
            r7 = r7 + a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) {
            res = res + a[i];
        }
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return f32_pairwise_sum(a, n2) + f32_pairwise_sum(a + n2, n - n2);
    }
}

void reverb_freq_resp_init(ReverbFrequencyResponse *r, float *tail_storage,
                           int n_freqs, int use_conservative,
                           double smoothing_base) {
    r->n_freqs = n_freqs;
    r->use_conservative = use_conservative ? 1 : 0;
    r->smoothing_base = smoothing_base;
    r->average_decay = 0.0;
    r->tail_response = tail_storage;
    memset(r->tail_response, 0, (size_t)n_freqs * sizeof(float));
}

void reverb_freq_resp_reset(ReverbFrequencyResponse *r) {
    r->average_decay = 0.0;
    memset(r->tail_response, 0, (size_t)r->n_freqs * sizeof(float));
}

void reverb_freq_resp_update(ReverbFrequencyResponse *r,
                             const float *frequency_response,
                             int n_partitions,
                             int filter_delay_blocks,
                             double linear_filter_quality,
                             int linear_filter_quality_is_none,
                             int stationary_block) {
    const int N = r->n_freqs;
    const float *direct;   /* frequency_response[filter_delay_blocks] */
    const float *tail_row; /* frequency_response[-1] */
    float direct_energy_f, tail_energy_f;
    double direct_energy, average_decay, smoothing;
    float decay_f; /* (float)average_decay applied per-bin (numpy weak scalar) */
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

    /* Average-decay scalar over k>=1 (kSkipBins=1). np.sum -> f32 pairwise. */
    direct_energy_f = f32_pairwise_sum(direct + 1, (size_t)(N - 1));
    /* python: float(np.sum(...)) -> f64. The == 0.0 test is on the f64 value. */
    direct_energy = (double)direct_energy_f;
    if (direct_energy == 0.0) {
        average_decay = 0.0;
    } else {
        tail_energy_f = f32_pairwise_sum(tail_row + 1, (size_t)(N - 1));
        average_decay = (double)tail_energy_f / direct_energy;
    }

    /* EMA: smoothing = smoothing_base * quality (f64); state update in f64. */
    smoothing = r->smoothing_base * linear_filter_quality;
    r->average_decay += smoothing * (average_decay - r->average_decay);

    /* Per-bin tail = direct(f32) * average_decay: numpy casts the f64 scalar to
     * f32 first, so the multiply is f32 * (float)average_decay. */
    decay_f = (float)r->average_decay;
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

    /* Neighbour-max smoothing (k=1..N-2), all float32.
     * avg = 0.5*(tail[k-1]+tail[k+1]) computed from the PRE-update tail values
     * (numpy slices tail[:-2]/tail[2:] before the in-place maximum assign). */
    if (N >= 3) {
        /* tail[1:-1] = np.maximum(tail[1:-1], 0.5*(tail[:-2]+tail[2:]))
         * The RHS reads original tail; the LHS write does not feed forward
         * because numpy evaluates the whole RHS array first. */
        int M = N - 2;
        /* Build avg into a scratch from originals, then assign. To match numpy
         * (RHS fully materialised first) we read neighbours from the original
         * buffer; since we only write indices 1..N-2 and read 0..N-1, and a
         * left-to-right in-place write of index k would corrupt the read of
         * index k+1's left-neighbour (k) for index k+1... so we must snapshot
         * the left neighbour. */
        float left_prev = tail[0]; /* original tail[k-1] for k=1 */
        for (k = 1; k <= M; ++k) {
            float orig_k = tail[k];
            float right = tail[k + 1];
            float avg = 0.5f * (left_prev + right);
            float nv = (avg > orig_k) ? avg : orig_k;
            tail[k] = nv;
            left_prev = orig_k; /* original (pre-write) value for next k-1 */
        }
    }

    /* tail.astype(np.float32) is a no-op (already f32); tail_response is tail. */
}
