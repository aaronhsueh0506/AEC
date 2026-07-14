/* filter_state_bridge.c — C port of filter_state_bridge.py. See header for the
 * parity contract. Build with -ffp-contract=off. */
#include "filter_state_bridge.h"

#include "simd_kernels.h"

/* numpy 1.26 pairwise_sum over float32, accumulated in float32. Mirrors
 * numpy/core/src/umath/loops_utils.h.src @pairwise_sum@:
 *   n < 8     -> sequential
 *   n <= 128  -> 8 partial accumulators (stride 8, unrolled), combined as a
 *                balanced tree, then the (n % 8) tail sequentially
 *   n > 128   -> split at n2 = (n/2) rounded down to a multiple of 8, recurse,
 *                add the two halves.
 * Every operation is a float32 add (no float64 widening).
 * Body delegates to simd_kernels.h's sk_pairwise_sum_tailfold_b_f32 (this
 * tree is byte-identical, modulo whitespace, to filter_analyzer.c's
 * fa_f32_pairwise_sum — that kernel's `_scalar` twin's verbatim source);
 * local symbol/signature kept so call sites are unchanged. */
float fsb_f32_pairwise_sum(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_b_f32(a, n);
}

/* np.mean over a float32 array: float32 pairwise sum / float32(n), kept in
 * float32 (float32-by-design; formerly widened via the float() cast in
 * `p_trace = float(np.mean(P))`). */
float fsb_f32_mean(const float *a, size_t n) {
    float s;
    if (n == 0) {
        return 0.0f;
    }
    s = fsb_f32_pairwise_sum(a, n);
    /* numpy mean divides the f32 sum by n in float32; kept in float32. */
    return s / (float)n;
}

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
        float div = 0.0f;
        if (P != NULL && p_len > 0) {
            float p_trace = fsb_f32_mean(P, p_len);
            float e_ratio = 1.0f;
            if (has_shadow) {
                if (main_e > 1e-12f) {
                    e_ratio = shadow_e / main_e;
                }
            }
            {
                float t = e_ratio - 1.0f;
                if (t < 0.0f) {
                    t = 0.0f;       /* max(0.0, e_ratio - 1.0) */
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
