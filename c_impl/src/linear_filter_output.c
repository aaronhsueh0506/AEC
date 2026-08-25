/* linear_filter_output.c — C port of
 * AEC._aec3_select_linear_filter_output (orchestrator.py:2894-2999).
 * See linear_filter_output.h for the parity contract.
 */
#include "linear_filter_output.h"

#include "aec_simd_kernels.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * pairwise float32 sum matching numpy's reduction tree shape (float32-by-
 * design; formerly a float64 accumulator mirroring pbfdkf.c's pairwise_sum_f32
 * twin): an 8-accumulator unrolled leaf for n<=128 (numpy NPY_PW_BLOCKSIZE),
 * then a recursive split at n/2 rounded down to a multiple of 8.
 * Used for the e2/y2/s2 = np.sum(<arr> ** 2) block energies.
 * Body delegates to simd_kernels.h's sk_pairwise_sum_tailfold_f32 (same
 * kernel pbfdkf.c's twin now uses — this file's tree diffed byte-identical
 * to pbfdkf.c's, modulo whitespace); local symbol/signature kept so call
 * sites are unchanged. */
static float pairwise_sum_f32(const float *a, int n) {
    return sk_pairwise_sum_tailfold_f32(a, (size_t)n);
}

/* np.sum(arr ** 2): square and pairwise-sum, both float32 (float32-by-design;
 * formerly upcast each f32 sample to double before squaring/summing). */
static float sum_sq_f32(const float *a, int n, float *scratch) {
    for (int i = 0; i < n; ++i) {
        float v = a[i];
        scratch[i] = v * v;
    }
    return pairwise_sum_f32(scratch, n);
}

int linear_filter_select_init(LinearFilterSelect *s,
                              int hop, int block_size, int fft_size,
                              int n_freqs) {
    memset(s, 0, sizeof(*s));
    s->hop        = hop;
    s->block_size = block_size;
    s->fft_size   = fft_size;
    s->n_freqs    = n_freqs;

    s->prev_output_time = (float *)malloc((size_t)hop * sizeof(float));
    s->e_form           = (float *)malloc((size_t)hop * sizeof(float));
    s->block_win        = (float *)malloc((size_t)block_size * sizeof(float));
    s->sel_esw          = (Complex *)malloc((size_t)n_freqs * sizeof(Complex));
    /* De-stacked per-hop scratch (see LinearFilterSelect struct comment). */
    s->scr_sq           = (float *)malloc((size_t)hop * sizeof(float));
    s->scr_sref         = (float *)malloc((size_t)hop * sizeof(float));
    s->scr_scoa         = (float *)malloc((size_t)hop * sizeof(float));
    s->scr_tin          = (float *)malloc((size_t)fft_size * sizeof(float));
    if (!s->prev_output_time || !s->e_form || !s->block_win || !s->sel_esw ||
        !s->scr_sq || !s->scr_sref || !s->scr_scoa || !s->scr_tin) {
        linear_filter_select_free(s);
        return -1;
    }
    linear_filter_select_reset(s);
    return 0;
}

void linear_filter_select_reset(LinearFilterSelect *s) {
    s->prev_output_valid    = 0;     /* Python _form_prev_output_time = None */
    s->form_last_selection  = LFS_SEL_REFINED;
    s->refined_last_selected = 1;
    if (s->prev_output_time)
        memset(s->prev_output_time, 0, (size_t)s->hop * sizeof(float));
}

void linear_filter_select_free(LinearFilterSelect *s) {
    if (!s) return;
    free(s->prev_output_time); s->prev_output_time = NULL;
    free(s->e_form);           s->e_form = NULL;
    free(s->block_win);        s->block_win = NULL;
    free(s->sel_esw);          s->sel_esw = NULL;
    free(s->scr_sq);           s->scr_sq = NULL;
    free(s->scr_sref);         s->scr_sref = NULL;
    free(s->scr_scoa);         s->scr_scoa = NULL;
    free(s->scr_tin);          s->scr_tin = NULL;
}

void linear_filter_select(LinearFilterSelect *s,
                          const float *e_refined_time,
                          const float *near_end,
                          const float *e_coarse_time,
                          const Complex *error_spec_windowed,
                          const Complex *echo_spec,
                          const float *sqrt_hann,
                          FftHandle *fft,
                          int linear_unusable,
                          int capture_fallback_enabled,
                          Complex *out_sel_esw,
                          Complex *out_sel_echo) {
    int hop = s->hop, blk = s->block_size, fft_size = s->fft_size, K = s->n_freqs;

    /* s_refined_time = near - e_refined; s_coarse_time = near - e_coarse
     * (float32 elementwise). e2/y2/s2 = np.sum(arr ** 2), float32-by-design
     * (formerly np.sum(arr.astype(f64) ** 2)). Instance scratch, not stack
     * (see LinearFilterSelect struct comment) — sq is reused sequentially
     * across the calls below (each sum_sq_f32 call fully consumes it before
     * the next writes it). */
    float *sq = s->scr_sq;      /* length hop */
    /* e2_refined */
    float e2_refined = sum_sq_f32(e_refined_time, hop, sq);
    /* e2_coarse */
    float e2_coarse = sum_sq_f32(e_coarse_time, hop, sq);
    /* y2 */
    float y2 = sum_sq_f32(near_end, hop, sq);
    /* s2_refined: build s_refined_time (f32) then sum-sq in f32 */
    {
        float *sref = s->scr_sref;   /* length hop */
        for (int i = 0; i < hop; ++i) sref[i] = near_end[i] - e_refined_time[i];
        float s2_refined = sum_sq_f32(sref, hop, sq);
        /* s2_coarse */
        float *scoa = s->scr_scoa;   /* length hop */
        for (int i = 0; i < hop; ++i) scoa[i] = near_end[i] - e_coarse_time[i];
        float s2_coarse = sum_sq_f32(scoa, hop, sq);

        /* thresholds (all float32) */
        float int16_scale_sq = 32768.0f * 32768.0f;
        float thr_30 = (30.0f * 30.0f) * (float)hop / int16_scale_sq;
        float thr_60 = (60.0f * 60.0f) * (float)hop / int16_scale_sq;

        int cond_coarse_cleaner =
            (e2_coarse < 0.9f * e2_refined) &&
            (y2 > thr_30) &&
            (s2_refined > thr_60 || s2_coarse > thr_60);
        int cond_refined_diverged =
            (e2_coarse < e2_refined) && (y2 < e2_refined);
        int use_refined = !(cond_coarse_cleaner || cond_refined_diverged);

        s->refined_last_selected = use_refined;

        /* Capture is the third candidate. A linear estimate the quality
         * analyzer has not declared usable, whose residual carries MORE
         * energy than the microphone it was subtracted from, is not a
         * cancellation at all -- it is the filter adding signal. Handing the
         * capture through instead makes capture the steady-state fallback,
         * which is what AEC3 does one stage later (echo_remover.cc:475's
         * Y_fft = UseLinearFilterOutput() ? E : Y).
         *
         * Doing it HERE rather than at the seam is what keeps the published
         * time hop and its spectrum a matched pair: sel_esw below is by
         * definition the FFT of [previous formed hop | this formed hop], and
         * sel_echo is back-solved from it, so both follow the substitution
         * with no second WOLA state machine to keep in sync. The refined/
         * coarse crossfade already in place carries the boundary, so no
         * overlap-add history is discarded and no hop goes to zero. During
         * the 30-sample transition, the mixture of the old residual and
         * capture is not a strict sample-by-sample or hop-energy bound.
         *
         * e2/y2 are the time-domain energies this function already computed;
         * by Parseval this is the same comparison the post chain's own
         * over-output guard spells in the frequency domain.
         *
         * linear_unusable is the analyzer's verdict as of the PREVIOUS hop --
         * this step runs before AecState is updated. The verdict is a latched
         * quality assessment rather than a per-hop signal, so the lag costs a
         * single hop at each edge. */
        float e2_selected = use_refined ? e2_refined : e2_coarse;
        int selection = use_refined ? LFS_SEL_REFINED : LFS_SEL_COARSE;
        if (capture_fallback_enabled && linear_unusable && e2_selected > y2)
            selection = LFS_SEL_CAPTURE;

        const float *candidate[3];
        candidate[LFS_SEL_COARSE]  = e_coarse_time;
        candidate[LFS_SEL_REFINED] = e_refined_time;
        candidate[LFS_SEL_CAPTURE] = near_end;

        const float *from_time = candidate[s->form_last_selection];
        const float *to_time   = candidate[selection];
        int transition_active = (s->form_last_selection != selection);

        if (transition_active) {
            const int kT = 30;
            /* _s = (np.arange(kT,f32)+1.0) / (kT+1.0): f32_array / pyfloat
             * → f32. (1.0 - _s): pyfloat - f32_array → f32. products f32. */
            for (int k = 0; k < kT; ++k) {
                float kf = (float)k + 1.0f;          /* arange(f32)+1.0 → f32 */
                float sc = kf / (float)(kT + 1);      /* f32 divide */
                float one_minus = 1.0f - sc;         /* 1.0(f32) - sc */
                s->e_form[k] = one_minus * from_time[k] + sc * to_time[k];
            }
            for (int k = kT; k < hop; ++k) s->e_form[k] = to_time[k];
        } else {
            memcpy(s->e_form, to_time, (size_t)hop * sizeof(float));
        }

        /* _e_old = prev_output_time (zeros if None); _e_block = [e_old | e_form]
         * _e_block_win = _e_block * sqrt_hann (length block_size, f32) */
        for (int i = 0; i < hop; ++i) {
            float eo = s->prev_output_valid ? s->prev_output_time[i] : 0.0f;
            s->block_win[i] = eo * sqrt_hann[i];
        }
        for (int i = 0; i < hop; ++i)
            s->block_win[hop + i] = s->e_form[i] * sqrt_hann[hop + i];

        /* selected_esw = rfft(_e_block_win, fft_size).astype(complex64):
         * zero-pad block_size samples to fft_size, forward FFT (bit-exact
         * pocketfft). Mirrors pbfdkf.c rfft_padded. Instance scratch, not
         * stack (see LinearFilterSelect struct comment). */
        {
            float *tin = s->scr_tin;   /* length fft_size, dead after the
                                        * call -> clobber-permitted rfft */
            memcpy(tin, s->block_win, (size_t)blk * sizeof(float));
            memset(tin + blk, 0, (size_t)(fft_size - blk) * sizeof(float));
            fft_forward_scratch(fft, tin, s->sel_esw);
        }

        /* _near_spec_win = error_spec_windowed + echo_spec (complex64)
         * selected_echo_spec = _near_spec_win - selected_esw (complex64) */
        for (int k = 0; k < K; ++k) {
            float nr = error_spec_windowed[k].r + echo_spec[k].r;
            float ni = error_spec_windowed[k].i + echo_spec[k].i;
            out_sel_esw[k]  = s->sel_esw[k];
            out_sel_echo[k].r = nr - s->sel_esw[k].r;
            out_sel_echo[k].i = ni - s->sel_esw[k].i;
        }

        /* UPDATE STATE: _form_prev_output_time = e_form; latches = use_refined */
        memcpy(s->prev_output_time, s->e_form, (size_t)hop * sizeof(float));
        s->prev_output_valid   = 1;
        s->form_last_selection = selection;
    }
}
