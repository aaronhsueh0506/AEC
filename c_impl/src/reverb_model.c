/* reverb_model.c — C port of reverb_model.py. All-float32 arithmetic to match
 * the numpy<2 reference (float32 power_spectrum + scalar/float32 scaling →
 * three separate float32 roundings per bin). */
#include "reverb_model.h"

#include <string.h>

void reverb_model_init(ReverbModel *m, float *reverb_storage, int n_bins) {
    m->n_bins = n_bins;
    m->reverb = reverb_storage;
    memset(m->reverb, 0, (size_t)n_bins * sizeof(float));
}

void reverb_model_reset(ReverbModel *m) {
    memset(m->reverb, 0, (size_t)m->n_bins * sizeof(float));
}

void reverb_model_update_no_freq_shaping(ReverbModel *m,
                                         const float *power_spectrum,
                                         float scaling, float decay) {
    int k;
    if (decay <= 0.0f) {
        return;
    }
    for (k = 0; k < m->n_bins; ++k) {
        /* numpy: tmp = ps*scaling (f32); reverb += tmp (f32); reverb *= decay (f32) */
        float tmp = power_spectrum[k] * scaling;
        m->reverb[k] = m->reverb[k] + tmp;
        m->reverb[k] = m->reverb[k] * decay;
    }
}

void reverb_model_update(ReverbModel *m, const float *power_spectrum,
                         const float *scaling, float decay) {
    int k;
    if (decay <= 0.0f) {
        return;
    }
    for (k = 0; k < m->n_bins; ++k) {
        float tmp = power_spectrum[k] * scaling[k];
        m->reverb[k] = m->reverb[k] + tmp;
        m->reverb[k] = m->reverb[k] * decay;
    }
}
