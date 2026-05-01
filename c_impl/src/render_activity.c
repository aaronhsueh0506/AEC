/**
 * render_activity.c - port of Python RenderActivityDetector (aec.py 2299-2358).
 */

#include "render_activity.h"
#include <stdlib.h>
#include <math.h>

#define ALPHA_CV       0.99f
#define STATIONARY_CV2 0.02f


struct RenderActivityDetector {
    float env_mean;
    float env_var;
    int   active_prev;
    int   is_stationary;
};


RenderActivityDetector* ra_create(void) {
    RenderActivityDetector* ra = (RenderActivityDetector*)calloc(1, sizeof(RenderActivityDetector));
    if (!ra) return NULL;
    ra_reset(ra);
    return ra;
}


void ra_destroy(RenderActivityDetector* ra) { free(ra); }


void ra_reset(RenderActivityDetector* ra) {
    if (!ra) return;
    ra->env_mean = 1e-10f;
    ra->env_var = 0.0f;
    ra->active_prev = 0;
    ra->is_stationary = 0;
}


RenderActivityState ra_update(RenderActivityDetector* ra,
                              const float* far_end, int hop) {
    RenderActivityState s = { 1e-10f, 0, 0, 0 };
    if (!ra || !far_end || hop <= 0) return s;

    /* float far_pwr_raw = float(mean(far_end²)) — Python line 2328 */
    float sum = 0.0f;
    for (int i = 0; i < hop; i++) sum += far_end[i] * far_end[i];
    float far_pwr_raw = sum / (float)hop;
    float far_pwr = far_pwr_raw + 1e-10f;
    int warmup_active = (far_pwr_raw > 1e-6f);

    if (far_pwr > 1e-6f) {
        if (!ra->active_prev) {
            ra->env_mean = far_pwr;
            ra->env_var = 0.0f;
            ra->active_prev = 1;
        } else {
            float old_mean = ra->env_mean;
            ra->env_mean = ALPHA_CV * ra->env_mean + (1.0f - ALPHA_CV) * far_pwr;
            float diff = far_pwr - old_mean;
            ra->env_var = ALPHA_CV * ra->env_var + (1.0f - ALPHA_CV) * diff * diff;
        }
        float cv2 = ra->env_var / (ra->env_mean * ra->env_mean + 1e-10f);
        ra->is_stationary = (cv2 < STATIONARY_CV2) ? 1 : 0;
    } else {
        ra->active_prev = 0;
        ra->is_stationary = 0;
    }
    s.far_pwr = far_pwr;
    s.is_active = ra->active_prev;
    s.is_stationary = ra->is_stationary;
    s.warmup_active = warmup_active;
    return s;
}


int ra_is_active(const RenderActivityDetector* ra) {
    return ra ? ra->active_prev : 0;
}


int ra_is_stationary(const RenderActivityDetector* ra) {
    return ra ? ra->is_stationary : 0;
}
