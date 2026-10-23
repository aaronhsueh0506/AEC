/* linear_filter_output.c -- see linear_filter_output.h. */
#include "linear_filter_output.h"

#include "utility/simd_aec.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Sum of squares via the float32 pairwise reduction. */
static float sum_sq_f32(const float *a, int n, float *scratch) {
    for (int i = 0; i < n; ++i) {
        float v = a[i];
        scratch[i] = v * v;
    }
    return sk_pairwise_sum_tailfold_f32(scratch, (size_t)n);
}

void linear_filter_select_reset(LinearFilterSelect *s) {
    s->prev_output_valid    = 0;
    s->form_last_selection  = LFS_SEL_REFINED;
    s->refined_last_selected = 1;
    if (s->prev_output_time)
        memset(s->prev_output_time, 0, (size_t)s->hop * sizeof(float));
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

    /* Instance scratch, reused sequentially by the sum_sq_f32 calls below. */
    float *sq = s->scr_sq;
    float e2_refined = sum_sq_f32(e_refined_time, hop, sq);
    float e2_coarse = sum_sq_f32(e_coarse_time, hop, sq);
    float y2 = sum_sq_f32(near_end, hop, sq);
    {
        float int16_scale_sq = 32768.0f * 32768.0f;
        float thr_30 = (30.0f * 30.0f) * (float)hop / int16_scale_sq;
        float thr_60 = (60.0f * 60.0f) * (float)hop / int16_scale_sq;

        /* s2_refined/s2_coarse only feed the last term of cond_coarse_cleaner; both are computed
         * lazily. */
        int cond_coarse_cleaner = 0;
        if ((e2_coarse < 0.9f * e2_refined) && (y2 > thr_30)) {
            float *sref = s->scr_sref;
            for (int i = 0; i < hop; ++i) sref[i] = near_end[i] - e_refined_time[i];
            float s2_refined = sum_sq_f32(sref, hop, sq);
            if (s2_refined > thr_60) {
                cond_coarse_cleaner = 1;
            } else {
                float *scoa = s->scr_scoa;
                for (int i = 0; i < hop; ++i) scoa[i] = near_end[i] - e_coarse_time[i];
                float s2_coarse = sum_sq_f32(scoa, hop, sq);
                cond_coarse_cleaner = (s2_coarse > thr_60);
            }
        }
        int cond_refined_diverged =
            (e2_coarse < e2_refined) && (y2 < e2_refined);
        int use_refined = !(cond_coarse_cleaner || cond_refined_diverged);

        s->refined_last_selected = use_refined;

        /* Capture is the third candidate: when the linear estimate is unusable and the residual has
         * more energy than the capture, the filter is adding signal, so the capture is passed
         * through. This is the time-domain hop test, separate from the post chain's windowed
         * over-output guard. */
        float e2_selected = use_refined ? e2_refined : e2_coarse;
        int selection = use_refined ? LFS_SEL_REFINED : LFS_SEL_COARSE;
        if (capture_fallback_enabled && linear_unusable && e2_selected > y2)
            selection = LFS_SEL_CAPTURE;

        const float *candidate[3] = { e_coarse_time, e_refined_time,
                                      near_end };
        const float *from_time = candidate[s->form_last_selection];
        const float *to_time   = candidate[selection];
        int transition_active = (s->form_last_selection != selection);

        if (transition_active) {
            const int kT = 30;
            for (int k = 0; k < kT; ++k) {
                float kf = (float)k + 1.0f;
                float sc = kf / (float)(kT + 1);
                float one_minus = 1.0f - sc;
                s->e_form[k] = one_minus * from_time[k] + sc * to_time[k];
            }
            for (int k = kT; k < hop; ++k) s->e_form[k] = to_time[k];
        } else {
            memcpy(s->e_form, to_time, (size_t)hop * sizeof(float));
        }

        /* Window [previous | current] with sqrt-Hann, zero-pad to fft_size and transform into
         * scr_tin (scratch), which is dead after the FFT. */
        {
            float *tin = s->scr_tin;
            for (int i = 0; i < hop; ++i) {
                float eo = s->prev_output_valid ? s->prev_output_time[i] : 0.0f;
                tin[i] = eo * sqrt_hann[i];
            }
            for (int i = 0; i < hop; ++i)
                tin[hop + i] = s->e_form[i] * sqrt_hann[hop + i];
            memset(tin + blk, 0, (size_t)(fft_size - blk) * sizeof(float));
            fft_forward_scratch(fft, tin, s->sel_esw);
        }

        /* selected echo = (error_spec_windowed + echo_spec) - selected error spectrum, computed in
         * place in out_sel_echo. */
        sk_cadd_f32(out_sel_echo, error_spec_windowed, echo_spec, K);
        memcpy(out_sel_esw, s->sel_esw, (size_t)K * sizeof(Complex));
        sk_csub_f32(out_sel_echo, out_sel_echo, s->sel_esw, K);

        memcpy(s->prev_output_time, s->e_form, (size_t)hop * sizeof(float));
        s->prev_output_valid   = 1;
        s->form_last_selection = selection;
    }
}
