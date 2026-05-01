/**
 * multi_erle.c - Multi-ERLE estimators
 *
 * Bit-exact port of Python aec.py:
 *   FilterErleEstimator   (lines 916-957)
 *   FullbandErleEstimator (lines 959-976)
 *   compute_erle_confidence (lines 978-986)
 *
 * Float32 throughout. Drift bound: float32-vs-float64 reduction-order only.
 */

#include "multi_erle.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* ========== FilterErleEstimator ========== */

struct FilterErleEstimator {
    int    n_freqs;
    float* erle;          /* [n_freqs] per-bin ERLE */
    float* tmp;           /* [n_freqs] scratch for 3-bin smooth */
    float  alpha_rise;    /* 0.95 — slow rise */
    float  alpha_drop;    /* 0.7 — fast drop */
};


FilterErleEstimator* fe_create(int n_freqs) {
    if (n_freqs <= 0) return NULL;
    FilterErleEstimator* fe = (FilterErleEstimator*)calloc(1, sizeof(FilterErleEstimator));
    if (!fe) return NULL;
    fe->n_freqs = n_freqs;
    fe->erle = (float*)malloc(n_freqs * sizeof(float));
    fe->tmp  = (float*)malloc(n_freqs * sizeof(float));
    if (!fe->erle || !fe->tmp) {
        fe_destroy(fe);
        return NULL;
    }
    fe->alpha_rise = 0.95f;
    fe->alpha_drop = 0.7f;
    fe_reset(fe);
    return fe;
}


void fe_destroy(FilterErleEstimator* fe) {
    if (!fe) return;
    free(fe->erle);
    free(fe->tmp);
    free(fe);
}


void fe_reset(FilterErleEstimator* fe) {
    if (!fe) return;
    for (int k = 0; k < fe->n_freqs; k++) fe->erle[k] = 1.0f;
}


void fe_update(FilterErleEstimator* fe,
               const Complex* echo_spec,
               const Complex* error_spec,
               int far_active,
               float dt_indicator) {
    if (!fe || !echo_spec || !error_spec) return;
    if (!far_active) return;  /* Python line 932-933 */

    const int n = fe->n_freqs;
    /* Python: dt_factor = clip(dt_indicator, 0, 1)
     *         alpha_rise_eff = alpha_rise + (1-alpha_rise) * dt_factor */
    float dt_factor = dt_indicator;
    if (dt_factor < 0.0f) dt_factor = 0.0f;
    if (dt_factor > 1.0f) dt_factor = 1.0f;
    const float alpha_rise_eff = fe->alpha_rise
                                  + (1.0f - fe->alpha_rise) * dt_factor;
    const float alpha_drop = fe->alpha_drop;

    /* Step 1: per-bin EMA update */
    for (int k = 0; k < n; k++) {
        float echo_pwr = echo_spec[k].r * echo_spec[k].r
                       + echo_spec[k].i * echo_spec[k].i;
        float error_pwr = error_spec[k].r * error_spec[k].r
                        + error_spec[k].i * error_spec[k].i + 1e-10f;
        /* Python: clip(echo_pwr / error_pwr, 0.1, 1000) */
        float inst = echo_pwr / error_pwr;
        if (inst < 0.1f)    inst = 0.1f;
        if (inst > 1000.0f) inst = 1000.0f;
        /* asymmetric EMA: drop fast, rise (DT-frozen) slow */
        float a = (inst < fe->erle[k]) ? alpha_drop : alpha_rise_eff;
        fe->erle[k] = a * fe->erle[k] + (1.0f - a) * inst;
    }

    /* Step 2: 3-bin convolution kernel [0.25, 0.5, 0.25] same-mode.
     * numpy convolve([0.25,0.5,0.25], 'same') edge handling:
     *   k=0:    0.25*0     + 0.5*erle[0] + 0.25*erle[1]
     *   k=N-1:  0.25*erle[N-2] + 0.5*erle[N-1] + 0.25*0 */
    fe->tmp[0] = 0.5f * fe->erle[0] + 0.25f * fe->erle[1];
    for (int k = 1; k < n - 1; k++) {
        fe->tmp[k] = 0.25f * fe->erle[k-1]
                   + 0.5f  * fe->erle[k]
                   + 0.25f * fe->erle[k+1];
    }
    fe->tmp[n-1] = 0.25f * fe->erle[n-2] + 0.5f * fe->erle[n-1];

    /* Step 3: clip(erle, 0.5, 200) and copy back */
    for (int k = 0; k < n; k++) {
        float v = fe->tmp[k];
        if (v < 0.5f)   v = 0.5f;
        if (v > 200.0f) v = 200.0f;
        fe->erle[k] = v;
    }
}


const float* fe_get_erle(const FilterErleEstimator* fe) {
    return fe ? fe->erle : NULL;
}


int fe_get_n_freqs(const FilterErleEstimator* fe) {
    return fe ? fe->n_freqs : 0;
}


/* ========== FullbandErleEstimator ========== */

struct FullbandErleEstimator {
    float fb_erle;
    float alpha;        /* 0.97 */
};


FullbandErleEstimator* fbe_create(void) {
    FullbandErleEstimator* fbe = (FullbandErleEstimator*)calloc(1, sizeof(FullbandErleEstimator));
    if (!fbe) return NULL;
    fbe->alpha = 0.97f;
    fbe->fb_erle = 1.0f;
    return fbe;
}


void fbe_destroy(FullbandErleEstimator* fbe) {
    free(fbe);
}


void fbe_reset(FullbandErleEstimator* fbe) {
    if (!fbe) return;
    fbe->fb_erle = 1.0f;
}


void fbe_update(FullbandErleEstimator* fbe,
                float near_power, float error_power,
                int far_active, float dt_indicator) {
    /* Python lines 968-972: gate is far_active AND dt<=0.3 AND near>=1e-8 */
    if (!fbe) return;
    if (!far_active || dt_indicator > 0.3f || near_power < 1e-8f) return;
    float inst = near_power / (error_power + 1e-10f);
    if (inst < 0.5f)   inst = 0.5f;
    if (inst > 100.0f) inst = 100.0f;
    fbe->fb_erle = fbe->alpha * fbe->fb_erle + (1.0f - fbe->alpha) * inst;
}


float fbe_get_fb_erle(const FullbandErleEstimator* fbe) {
    return fbe ? fbe->fb_erle : 1.0f;
}


/* ========== compute_erle_confidence ========== */

float compute_erle_confidence(const float* erle_l1, int n_freqs, float fb_erle) {
    if (!erle_l1 || n_freqs <= 0) return 0.0f;

    /* Python: float(np.mean(erle_l1)) — float64 reduction. We use float32
     * accumulator for parity-friendly behavior; mean of float32 values
     * stays in float32 range. */
    float sum = 0.0f;
    for (int k = 0; k < n_freqs; k++) sum += erle_l1[k];
    float l1_mean = sum / (float)n_freqs;

    if (l1_mean < 0.5f || fb_erle < 0.5f) return 0.0f;

    float log_diff = logf(l1_mean + 1e-10f) - logf(fb_erle + 1e-10f);
    if (log_diff < 0.0f) log_diff = -log_diff;
    return expf(-log_diff / 2.0f);
}
