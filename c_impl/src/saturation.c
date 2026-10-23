/* saturation.c -- clipping detector.
 * Counts samples above the threshold plus 2x the runs of near-threshold flat samples,
 * then smooths the clipped fraction with an asymmetric EMA. */
#include "saturation.h"
#include "aec_scale.h"
#include "utility/simd_aec.h"
#include <math.h>
#include <stddef.h>

void saturation_init(Saturation* s, float threshold,
                     int hop_size, int sample_rate) {
    s->threshold        = threshold;
    s->saturation_level = 0.0f;
    /* Attack/release EMAs authored for hop=256 @ 16 kHz (16 ms). */
    s->alpha_attack  = aec_growth_rehop(0.3f,  256, 16000,
                                        hop_size, sample_rate);
    s->alpha_release = aec_growth_rehop(0.98f, 256, 16000,
                                        hop_size, sample_rate);
}

void saturation_reset(Saturation* s) {
    s->saturation_level = 0.0f;
}

float saturation_detect(Saturation* s, const float* signal, size_t n) {
    if (n == 0) return s->saturation_level;

    const float thr      = s->threshold;
    const float thr_high = thr * 0.8f;

    long clip_count, consec_count;
    sk_saturation_counts_f32(signal, n, thr, thr_high,
                              &clip_count, &consec_count);

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
    /* Soft knee: |x| >= knee is compressed with tanh over scale = max(1 - knee, 1e-6). */
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
