/* erle.c — port of aec.py FilterErleEstimator/FullbandErleEstimator/
 *             compute_erle_confidence.
 */
#include "erle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── FilterErleEstimator ───────────────────────────────────────── */

void filter_erle_init(FilterErleEst* f, int n_freqs) {
    f->n_freqs = n_freqs;
    f->erle = (float*)malloc((size_t)n_freqs * sizeof(float));
    for (int k = 0; k < n_freqs; ++k) f->erle[k] = 1.0f;
    f->alpha_rise = 0.95;
    f->alpha_drop = 0.7;
}
void filter_erle_free(FilterErleEst* f) { free(f->erle); f->erle = NULL; }
void filter_erle_reset(FilterErleEst* f) {
    for (int k = 0; k < f->n_freqs; ++k) f->erle[k] = 1.0f;
}

void filter_erle_update(FilterErleEst* f,
                           const float* echo_re, const float* echo_im,
                           const float* err_re,  const float* err_im,
                           int far_active, double dt_indicator) {
    if (!far_active) return;

    const int K = f->n_freqs;
    /* dt_factor = clip(dt_indicator, 0, 1) — fp64 in Python (Python float) */
    double dt_factor = dt_indicator;
    if (dt_factor < 0.0) dt_factor = 0.0;
    else if (dt_factor > 1.0) dt_factor = 1.0;
    double alpha_rise_eff = f->alpha_rise + (1.0 - f->alpha_rise) * dt_factor;

    /* Compute new erle in fp64 buffer (matches Python's promote-to-fp64
     * via mixed scalar/array math), then convolve, then cast back to fp32. */
    double* tmp = (double*)malloc((size_t)K * sizeof(double));
    for (int k = 0; k < K; ++k) {
        /* echo_pwr fp32 (np.abs(complex64)**2 → fp32) */
        float echo_pwr_f = echo_re[k] * echo_re[k] + echo_im[k] * echo_im[k];
        float err_pwr_f  = err_re[k] * err_re[k] + err_im[k] * err_im[k] + 1e-10f;
        float inst_f     = echo_pwr_f / err_pwr_f;
        if (inst_f < 0.1f)    inst_f = 0.1f;
        if (inst_f > 1000.0f) inst_f = 1000.0f;
        /* alpha picked per-bin from fp64 scalars → result is fp64 */
        double alpha = (inst_f < f->erle[k]) ? f->alpha_drop : alpha_rise_eff;
        tmp[k] = alpha * (double)f->erle[k] + (1.0 - alpha) * (double)inst_f;
    }

    /* 3-bin convolution in 'same' mode: kernel = [0.25, 0.5, 0.25].
     * np.convolve(arr, kernel, 'same') keeps len; boundaries are zero-padded.
     * Convolution is reverse-multiply-sum, but for symmetric kernel this
     * equals correlation. For 'same' mode with kernel length 3, output[i] =
     *   k[0]*arr[i+1] + k[1]*arr[i] + k[2]*arr[i-1]      (correlation form)
     * But np.convolve with kernel reversed is equivalent because kernel
     * is symmetric. Verify by:
     *   np.convolve([a,b,c,d], [0.25,0.5,0.25], 'same')
     *     = [0.25*0+0.5*a+0.25*b, 0.25*a+0.5*b+0.25*c,
     *        0.25*b+0.5*c+0.25*d, 0.25*c+0.5*d+0.25*0]
     */
    double* conv = (double*)malloc((size_t)K * sizeof(double));
    for (int k = 0; k < K; ++k) {
        double left  = (k > 0)       ? tmp[k - 1] : 0.0;
        double mid   = tmp[k];
        double right = (k < K - 1)   ? tmp[k + 1] : 0.0;
        conv[k] = 0.25 * left + 0.5 * mid + 0.25 * right;
    }

    /* clip [0.5, 200] then cast to fp32 (matches `.astype(np.float32)`) */
    for (int k = 0; k < K; ++k) {
        double v = conv[k];
        if (v < 0.5)   v = 0.5;
        if (v > 200.0) v = 200.0;
        f->erle[k] = (float)v;
    }
    free(tmp); free(conv);
}

/* ── FullbandErleEstimator ─────────────────────────────────────── */

void fullband_erle_init(FullbandErleEst* f) {
    f->fb_erle = 1.0;
    f->alpha   = 0.97;
}
void fullband_erle_reset(FullbandErleEst* f) { f->fb_erle = 1.0; }

void fullband_erle_update(FullbandErleEst* f,
                             double near_power, double error_power,
                             int far_active, double dt_indicator) {
    if (!far_active || dt_indicator > 0.3 || near_power < 1e-8) return;
    double inst = near_power / (error_power + 1e-10);
    if (inst < 0.5)   inst = 0.5;
    if (inst > 100.0) inst = 100.0;
    f->fb_erle = f->alpha * f->fb_erle + (1.0 - f->alpha) * inst;
}

/* ── compute_erle_confidence ───────────────────────────────────── */

double erle_compute_confidence(const float* erle_l1, int n_freqs,
                                  double fb_erle) {
    /* Python: l1_mean = float(np.mean(erle_l1))
     *   np.mean of fp32 array returns fp64 (np.mean default dtype).
     */
    double sum = 0.0;
    for (int k = 0; k < n_freqs; ++k) sum += (double)erle_l1[k];
    double l1_mean = sum / (double)n_freqs;

    if (l1_mean < 0.5 || fb_erle < 0.5) return 0.0;
    double log_diff = fabs(log(l1_mean + 1e-10) - log(fb_erle + 1e-10));
    return exp(-log_diff / 2.0);
}
