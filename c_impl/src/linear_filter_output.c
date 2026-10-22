/* linear_filter_output.c — C port of
 * the orchestrator's linear-filter output selector (python/modules/orchestrator.py).
 * See linear_filter_output.h for the parity contract.
 */
#include "linear_filter_output.h"

#include "utility/simd_aec.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* np.sum(arr ** 2): square and pairwise-sum (numpy's reduction tree), both
 * float32. Used for the e2/y2/s2 block energies. */
static float sum_sq_f32(const float *a, int n, float *scratch) {
    for (int i = 0; i < n; ++i) {
        float v = a[i];
        scratch[i] = v * v;
    }
    return sk_pairwise_sum_tailfold_f32(scratch, (size_t)n);
}

void linear_filter_select_reset(LinearFilterSelect *s) {
    s->prev_output_valid    = 0;     /* Python _form_prev_output_time = None */
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

    /* s_refined_time = near - e_refined; s_coarse_time = near - e_coarse
     * (float32 elementwise). e2/y2/s2 = np.sum(arr ** 2), float32. Instance
     * scratch, not stack (see LinearFilterSelect struct comment) — sq is reused
     * sequentially across the calls below (each sum_sq_f32 call fully consumes
     * it before the next writes it). */
    float *sq = s->scr_sq;      /* length hop */
    /* e2_refined */
    float e2_refined = sum_sq_f32(e_refined_time, hop, sq);
    /* e2_coarse */
    float e2_coarse = sum_sq_f32(e_coarse_time, hop, sq);
    /* y2 */
    float y2 = sum_sq_f32(near_end, hop, sq);
    /* thresholds (all float32) */
    {
        float int16_scale_sq = 32768.0f * 32768.0f;
        float thr_30 = (30.0f * 30.0f) * (float)hop / int16_scale_sq;
        float thr_60 = (60.0f * 60.0f) * (float)hop / int16_scale_sq;

        /* s2_refined/s2_coarse (near - e_*, summed-squared, f32) only feed
         * cond_coarse_cleaner's final (s2_refined>thr_60 || s2_coarse>thr_60)
         * term -- short-circuit on both the outer (e2_coarse,y2) gate and on
         * s2_refined alone clearing thr_60, so s2_coarse's build+reduction is
         * skipped whenever it can't change the result. */
        int cond_coarse_cleaner = 0;
        if ((e2_coarse < 0.9f * e2_refined) && (y2 > thr_30)) {
            float *sref = s->scr_sref;   /* length hop */
            for (int i = 0; i < hop; ++i) sref[i] = near_end[i] - e_refined_time[i];
            float s2_refined = sum_sq_f32(sref, hop, sq);
            if (s2_refined > thr_60) {
                cond_coarse_cleaner = 1;
            } else {
                float *scoa = s->scr_scoa;   /* length hop */
                for (int i = 0; i < hop; ++i) scoa[i] = near_end[i] - e_coarse_time[i];
                float s2_coarse = sum_sq_f32(scoa, hop, sq);
                cond_coarse_cleaner = (s2_coarse > thr_60);
            }
        }
        int cond_refined_diverged =
            (e2_coarse < e2_refined) && (y2 < e2_refined);
        int use_refined = !(cond_coarse_cleaner || cond_refined_diverged);

        s->refined_last_selected = use_refined;

        /* Capture is the third candidate. A linear estimate the quality
         * analyzer has not declared usable, whose residual carries MORE
         * energy than the capture it was subtracted from, is not a
         * cancellation at all -- it is the filter adding signal. Handing the
         * capture through instead makes capture the steady-state fallback,
         * matching the reference remover's (Y_fft = UseLinearFilterOutput()
         * ? E : Y) one stage later. During the 30-sample transition the
         * mixture of old residual and capture is not a strict bound.
         *
         * Related but NOT the same test as the post chain's own over-output
         * guard (the output-selection step in aec_post.c): that one compares
         * the sqrt-Hann weighted two-hop block in the frequency domain, this one the
         * current unwindowed hop. Neither subsumes the other.
         *
         * linear_unusable is the analyzer's verdict as of the PREVIOUS hop;
         * see the parameter contract in the header. */
        float e2_selected = use_refined ? e2_refined : e2_coarse;
        int selection = use_refined ? LFS_SEL_REFINED : LFS_SEL_COARSE;
        if (capture_fallback_enabled && linear_unusable && e2_selected > y2)
            selection = LFS_SEL_CAPTURE;

        /* Indexed in LFS_SEL_* order. */
        const float *candidate[3] = { e_coarse_time, e_refined_time,
                                      near_end };
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
         * _e_block_win = _e_block * sqrt_hann (length block_size, f32).
         * selected_esw = rfft(_e_block_win, fft_size).astype(complex64):
         * zero-pad block_size samples to fft_size, forward FFT (bit-exact
         * pocketfft). Mirrors pbfdkf.c rfft_padded. Written straight into
         * scr_tin (instance scratch, dead after the FFT call -> clobber-
         * permitted) instead of via block_win -- nothing else reads
         * block_win between the fill and this FFT. */
        {
            float *tin = s->scr_tin;   /* length fft_size */
            for (int i = 0; i < hop; ++i) {
                float eo = s->prev_output_valid ? s->prev_output_time[i] : 0.0f;
                tin[i] = eo * sqrt_hann[i];
            }
            for (int i = 0; i < hop; ++i)
                tin[hop + i] = s->e_form[i] * sqrt_hann[hop + i];
            memset(tin + blk, 0, (size_t)(fft_size - blk) * sizeof(float));
            fft_forward_scratch(fft, tin, s->sel_esw);
        }

        /* _near_spec_win = error_spec_windowed + echo_spec (complex64)
         * selected_echo_spec = _near_spec_win - selected_esw (complex64)
         *
         * Routed through sk_cadd_f32/sk_csub_f32 instead of a hand-rolled
         * per-k loop: out_sel_echo holds _near_spec_win after the add (same
         * two per-component float adds, same rounding, as an open-coded
         * per-component loop would give), then sk_csub_f32 subtracts
         * s->sel_esw from it IN PLACE (out == a, which sk_csub_f32 supports
         * by loading a 4-lane block in full before it is stored, never
         * revisiting an earlier index) -- the identical two separately-
         * rounded operations in the identical order, just via kernel calls.
         * out_sel_esw is copied from s->sel_esw with memcpy, an exact bit
         * copy matching the loop's plain struct assignment. */
        sk_cadd_f32(out_sel_echo, error_spec_windowed, echo_spec, K);
        memcpy(out_sel_esw, s->sel_esw, (size_t)K * sizeof(Complex));
        sk_csub_f32(out_sel_echo, out_sel_echo, s->sel_esw, K);

        /* UPDATE STATE: _form_prev_output_time = e_form; latches = use_refined */
        memcpy(s->prev_output_time, s->e_form, (size_t)hop * sizeof(float));
        s->prev_output_valid   = 1;
        s->form_last_selection = selection;
    }
}
