/* saturation.c — 1:1 port of python/aec.py SaturationDetector.
 *
 * Python detect() (aec.py:1048-1073):
 *     n = len(signal)
 *     if n == 0: return saturation_level
 *     abs_sig    = np.abs(signal)
 *     clip_count = np.sum(abs_sig > threshold)
 *     consec_count = 0
 *     high_mask = abs_sig > threshold * 0.8
 *     for i in range(1, n):
 *         if high_mask[i] and high_mask[i-1] and abs(s[i]-s[i-1]) < 1e-6:
 *             consec_count += 1
 *     raw_sat = min((clip + 2*consec)/n, 1.0)
 *     alpha = attack if raw_sat > saturation_level else release
 *     saturation_level = alpha*saturation_level + (1-alpha)*raw_sat
 *
 * Note: np.sum over a bool array is integer; in C we use long sum.
 * abs(...) on Python float is fp64; C uses fabs.
 */
#include "saturation.h"
#include <math.h>
#include <stddef.h>

void saturation_init(Saturation* s, double threshold) {
    s->threshold        = threshold;
    s->saturation_level = 0.0;
    s->alpha_attack     = 0.3;
    s->alpha_release    = 0.98;
}

void saturation_reset(Saturation* s) {
    s->saturation_level = 0.0;
}

double saturation_detect(Saturation* s, const float* signal, size_t n) {
    if (n == 0) return s->saturation_level;

    const double thr      = s->threshold;
    const double thr_high = thr * 0.8;

    long clip_count   = 0;
    long consec_count = 0;

    /* First pass: clip_count + build high_mask implicitly via fabs. */
    for (size_t i = 0; i < n; ++i) {
        double a = fabs((double)signal[i]);
        if (a > thr) clip_count++;
    }

    /* Consecutive-near-peak loop (Python iterates i from 1..n-1) */
    for (size_t i = 1; i < n; ++i) {
        double a_i  = fabs((double)signal[i]);
        double a_im = fabs((double)signal[i - 1]);
        if (a_i > thr_high && a_im > thr_high &&
            fabs((double)signal[i] - (double)signal[i - 1]) < 1e-6) {
            consec_count++;
        }
    }

    double raw_sat = (double)(clip_count + 2 * consec_count) / (double)n;
    if (raw_sat > 1.0) raw_sat = 1.0;

    double alpha = (raw_sat > s->saturation_level) ? s->alpha_attack
                                                   : s->alpha_release;
    s->saturation_level = alpha * s->saturation_level
                        + (1.0 - alpha) * raw_sat;
    return s->saturation_level;
}

void saturation_soft_clip(const float* signal, float* out, size_t n,
                             double knee) {
    /* Python:
     *     out = signal.copy()
     *     mask = abs(signal) >= knee
     *     for masked: out = sign * (knee + tanh(excess/scale)*scale)
     *     scale = max(1.0 - knee, 1e-6)
     */
    double scale = 1.0 - knee;
    if (scale < 1e-6) scale = 1e-6;
    for (size_t i = 0; i < n; ++i) {
        double xi = (double)signal[i];
        double a  = fabs(xi);
        if (a >= knee) {
            double sign     = (xi > 0.0) ? 1.0 : (xi < 0.0 ? -1.0 : 0.0);
            double excess   = a - knee;
            double compr    = knee + tanh(excess / scale) * scale;
            out[i] = (float)(sign * compr);
        } else {
            out[i] = signal[i];
        }
    }
}
