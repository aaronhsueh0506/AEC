/**
 * hpf.h - 2nd-order Butterworth IIR high-pass filter
 *
 * Matches Python HighPassFilter (bilinear transform, Direct Form II).
 * Removes DC offset, 50/60Hz hum, and low-frequency rumble.
 */

#ifndef HPF_H
#define HPF_H

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} Hpf;

static inline Hpf* hpf_create(float cutoff_hz, int sample_rate) {
    Hpf* h = (Hpf*)calloc(1, sizeof(Hpf));
    if (!h) return NULL;

    /* Bilinear transform: pre-warp analog frequency */
    float wc = 2.0f * (float)M_PI * cutoff_hz / (float)sample_rate;
    float wc_w = tanf(wc / 2.0f);
    float k = wc_w * wc_w;
    float sqrt2 = sqrtf(2.0f);
    float norm = 1.0f / (1.0f + sqrt2 * wc_w + k);

    h->b0 = norm;
    h->b1 = -2.0f * norm;
    h->b2 = norm;
    h->a1 = 2.0f * (k - 1.0f) * norm;
    h->a2 = (1.0f - sqrt2 * wc_w + k) * norm;
    h->z1 = 0.0f;
    h->z2 = 0.0f;
    return h;
}

static inline void hpf_process(Hpf* h, float* x, int n) {
    float b0 = h->b0, b1 = h->b1, b2 = h->b2;
    float a1 = h->a1, a2 = h->a2;
    float z1 = h->z1, z2 = h->z2;
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float yi = b0 * xi + z1;
        z1 = b1 * xi - a1 * yi + z2;
        z2 = b2 * xi - a2 * yi;
        x[i] = yi;
    }
    h->z1 = z1;
    h->z2 = z2;
}

static inline void hpf_reset(Hpf* h) {
    if (h) { h->z1 = 0.0f; h->z2 = 0.0f; }
}

static inline void hpf_destroy(Hpf* h) {
    free(h);
}

#ifdef __cplusplus
}
#endif

#endif /* HPF_H */
