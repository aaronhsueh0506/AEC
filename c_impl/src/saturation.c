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
 * abs(...) runs float32-by-design; C uses fabsf.
 */
#include "saturation.h"
#include "aec3_scale.h"
#include <math.h>
#include <stddef.h>

void saturation_init(Saturation* s, float threshold,
                     int hop_size, int sample_rate) {
    s->threshold        = threshold;
    s->saturation_level = 0.0f;
    /* Asymmetric per-hop EMA. Reference grid is hop=256/16000 (16 ms),
     * NOT 10 ms: authored by commit 243d67c against frame 512/hop 256.
     * Mirrors preprocessing.py SaturationDetector. */
    s->alpha_attack  = aec3_growth_rehop(0.3f,  256, 16000,
                                         hop_size, sample_rate);
    s->alpha_release = aec3_growth_rehop(0.98f, 256, 16000,
                                         hop_size, sample_rate);
}

void saturation_reset(Saturation* s) {
    s->saturation_level = 0.0f;
}

float saturation_detect(Saturation* s, const float* signal, size_t n) {
    if (n == 0) return s->saturation_level;

    const float thr      = s->threshold;
    const float thr_high = thr * 0.8f;

    long clip_count   = 0;
    long consec_count = 0;

    /* First pass: clip_count + build high_mask implicitly via fabsf. */
    for (size_t i = 0; i < n; ++i) {
        float a = fabsf(signal[i]);
        if (a > thr) clip_count++;
    }

    /* Consecutive-near-peak loop (Python iterates i from 1..n-1) */
    for (size_t i = 1; i < n; ++i) {
        float a_i  = fabsf(signal[i]);
        float a_im = fabsf(signal[i - 1]);
        if (a_i > thr_high && a_im > thr_high &&
            fabsf(signal[i] - signal[i - 1]) < 1e-6f) {
            consec_count++;
        }
    }

    float raw_sat = (float)(clip_count + 2 * consec_count) / (float)n;
    if (raw_sat > 1.0f) raw_sat = 1.0f;

    float alpha = (raw_sat > s->saturation_level) ? s->alpha_attack
                                                  : s->alpha_release;
    s->saturation_level = alpha * s->saturation_level
                        + (1.0f - alpha) * raw_sat;
    return s->saturation_level;
}

void saturation_soft_clip(const float* signal, float* out, size_t n,
                             float knee) {
    /* Python:
     *     out = signal.copy()
     *     mask = abs(signal) >= knee
     *     for masked: out = sign * (knee + tanh(excess/scale)*scale)
     *     scale = max(1.0 - knee, 1e-6)
     */
    float scale = 1.0f - knee;
    if (scale < 1e-6f) scale = 1e-6f;
    for (size_t i = 0; i < n; ++i) {
        float xi = signal[i];
        float a  = fabsf(xi);
        if (a >= knee) {
            float sign     = (xi > 0.0f) ? 1.0f : (xi < 0.0f ? -1.0f : 0.0f);
            float excess   = a - knee;
            float compr    = knee + tanhf(excess / scale) * scale;
            out[i] = sign * compr;
        } else {
            out[i] = signal[i];
        }
    }
}
