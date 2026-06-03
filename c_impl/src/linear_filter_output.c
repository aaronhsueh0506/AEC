/* linear_filter_output.c — C port of
 * AEC._aec3_select_linear_filter_output (orchestrator.py:2894-2999).
 * See linear_filter_output.h for the parity contract.
 */
#include "linear_filter_output.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * pairwise float64 sum matching numpy's reduction EXACTLY (the f64 twin of
 * pbfdkf.c's pairwise_sum_f32): an 8-accumulator unrolled leaf for n<=128
 * (numpy NPY_PW_BLOCKSIZE), then a recursive split at n/2 rounded down to a
 * multiple of 8. np.sum over float64 is pairwise the same way as float32.
 * Used for the e2/y2/s2 = np.sum(<arr>.astype(float64)**2) block energies. */
static double pw_leaf_f64(const double *a, int n) {
    if (n < 8) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += a[i];
        return s;
    }
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int i = 8;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
}
static double pairwise_sum_f64(const double *a, int n) {
    if (n <= 128) return pw_leaf_f64(a, n);
    int n2 = n / 2; n2 -= (n2 % 8);
    return pairwise_sum_f64(a, n2) + pairwise_sum_f64(a + n2, n - n2);
}

/* np.sum(arr.astype(float64) ** 2): upcast each f32 sample to double, square in
 * double, pairwise-sum in double. */
static double sum_sq_f64(const float *a, int n, double *scratch) {
    for (int i = 0; i < n; ++i) {
        double v = (double)a[i];
        scratch[i] = v * v;
    }
    return pairwise_sum_f64(scratch, n);
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
    if (!s->prev_output_time || !s->e_form || !s->block_win || !s->sel_esw) {
        linear_filter_select_free(s);
        return -1;
    }
    linear_filter_select_reset(s);
    return 0;
}

void linear_filter_select_reset(LinearFilterSelect *s) {
    s->prev_output_valid    = 0;     /* Python _form_prev_output_time = None */
    s->form_last_selection  = 1;     /* True = refined */
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
}

void linear_filter_select(LinearFilterSelect *s,
                          const float *e_refined_time,
                          const float *near_end,
                          const float *e_coarse_time,
                          const Complex *error_spec_windowed,
                          const Complex *echo_spec,
                          const float *sqrt_hann,
                          FftHandle *fft,
                          Complex *out_sel_esw,
                          Complex *out_sel_echo) {
    int hop = s->hop, blk = s->block_size, fft_size = s->fft_size, K = s->n_freqs;

    /* s_refined_time = near - e_refined; s_coarse_time = near - e_coarse
     * (float32 elementwise). e2/y2/s2 = np.sum(arr.astype(f64) ** 2). */
    double sq[8192];   /* hop <= block_size <= 8192 */
    /* e2_refined */
    double e2_refined = sum_sq_f64(e_refined_time, hop, sq);
    /* e2_coarse */
    double e2_coarse = sum_sq_f64(e_coarse_time, hop, sq);
    /* y2 */
    double y2 = sum_sq_f64(near_end, hop, sq);
    /* s2_refined: build s_refined_time (f32) then sum-sq in f64 */
    {
        float sref[8192];
        for (int i = 0; i < hop; ++i) sref[i] = near_end[i] - e_refined_time[i];
        double s2_refined = sum_sq_f64(sref, hop, sq);
        /* s2_coarse */
        float scoa[8192];
        for (int i = 0; i < hop; ++i) scoa[i] = near_end[i] - e_coarse_time[i];
        double s2_coarse = sum_sq_f64(scoa, hop, sq);

        /* thresholds (all double) */
        double int16_scale_sq = 32768.0 * 32768.0;
        double thr_30 = (30.0 * 30.0) * (double)hop / int16_scale_sq;
        double thr_60 = (60.0 * 60.0) * (double)hop / int16_scale_sq;

        int cond_coarse_cleaner =
            (e2_coarse < 0.9 * e2_refined) &&
            (y2 > thr_30) &&
            (s2_refined > thr_60 || s2_coarse > thr_60);
        int cond_refined_diverged =
            (e2_coarse < e2_refined) && (y2 < e2_refined);
        int use_refined = !(cond_coarse_cleaner || cond_refined_diverged);

        s->refined_last_selected = use_refined;

        /* from_time = refined if _form_last_selection else coarse
         * to_time   = refined if use_refined         else coarse */
        const float *from_time = s->form_last_selection ? e_refined_time
                                                        : e_coarse_time;
        const float *to_time   = use_refined ? e_refined_time : e_coarse_time;
        int transition_active = (s->form_last_selection != use_refined);

        if (transition_active) {
            const int kT = 30;
            /* _s = (np.arange(kT,f32)+1.0) / (kT+1.0): f32_array / pyfloat
             * → f32. (1.0 - _s): pyfloat - f32_array → f32. products f32. */
            for (int k = 0; k < kT; ++k) {
                float kf = (float)k + 1.0f;          /* arange(f32)+1.0 → f32 */
                float sc = kf / (float)(kT + 1.0);   /* / pyfloat-as-f32 */
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
         * pocketfft). Mirrors pbfdkf.c rfft_padded. */
        {
            float tin[8192];
            memcpy(tin, s->block_win, (size_t)blk * sizeof(float));
            memset(tin + blk, 0, (size_t)(fft_size - blk) * sizeof(float));
            fft_forward(fft, tin, s->sel_esw);
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
        s->form_last_selection = use_refined;
    }
}
