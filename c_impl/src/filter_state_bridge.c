/* filter_state_bridge.c — C port of filter_state_bridge.py. See header for the
 * parity contract. Build with -ffp-contract=off. */
#include "filter_state_bridge.h"

/* numpy 1.26 pairwise_sum over float32, accumulated in float32. Mirrors
 * numpy/core/src/umath/loops_utils.h.src @pairwise_sum@:
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators (stride 8, unrolled), combined as a
 *                balanced tree, then the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded down to a multiple of 8, recurse,
 *                add the two halves.
 * Every operation is a float32 add (no float64 widening). */
float fsb_f32_pairwise_sum(const float *a, size_t n) {
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
        return fsb_f32_pairwise_sum(a, n2) + fsb_f32_pairwise_sum(a + n2, n - n2);
    }
}

/* np.mean over a float32 array: float32 pairwise sum / float32(n), widened to
 * double by the float() cast in `p_trace = float(np.mean(P))`. */
double fsb_f32_mean(const float *a, size_t n) {
    float s;
    if (n == 0) {
        return 0.0;
    }
    s = fsb_f32_pairwise_sum(a, n);
    /* numpy mean divides the f32 sum by n in float32, then float() widens. */
    return (double)(s / (float)n);
}

void filter_state_bridge_build(FilterStateBridge *out,
                               FftHandle *fft,
                               const Complex *W0, int fft_size, int n_freqs,
                               const float *P, size_t p_len,
                               int has_shadow, double main_e, double shadow_e,
                               int filter_converged,
                               int main_paused,
                               double mu_final,
                               int external_delay_samples,
                               int any_coarse_filter_converged,
                               int all_filters_diverged,
                               float *taps_out) {
    /* --- filter_taps: irfft(W[0], fft_size).astype(float32) ----------------
     * When W is absent the Python path returns np.zeros(1, float32). */
    if (W0 != NULL && fft != NULL && fft_size > 0 && n_freqs > 0) {
        fft_inverse(fft, W0, taps_out);   /* fp64 internal, float32 store */
        out->filter_taps = taps_out;
        out->filter_taps_len = fft_size;
    } else {
        taps_out[0] = 0.0f;
        out->filter_taps = taps_out;
        out->filter_taps_len = 1;
    }

    /* --- divergence_indicator ----------------------------------------------
     * div = 0 unless P is present and non-empty. */
    {
        double div = 0.0;
        if (P != NULL && p_len > 0) {
            double p_trace = fsb_f32_mean(P, p_len);
            double e_ratio = 1.0;
            if (has_shadow) {
                if (main_e > 1e-12) {
                    e_ratio = shadow_e / main_e;
                }
            }
            {
                double t = e_ratio - 1.0;
                if (t < 0.0) {
                    t = 0.0;        /* max(0.0, e_ratio - 1.0) */
                }
                div = p_trace * t;
            }
        }
        out->divergence_indicator = div;
    }

    /* --- regime ------------------------------------------------------------ */
    out->regime = main_paused ? 1 : 0;

    /* --- pass-through fields ----------------------------------------------- */
    out->filter_converged = filter_converged ? 1 : 0;
    out->main_paused = main_paused ? 1 : 0;
    out->mu_final = mu_final;
    out->external_delay_samples = external_delay_samples;
    out->any_coarse_filter_converged = any_coarse_filter_converged ? 1 : 0;
    out->all_filters_diverged = all_filters_diverged ? 1 : 0;
}
