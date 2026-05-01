/**
 * dtd_estimator.c - port of Python DtdEstimator (aec.py 2080-2280)
 * Coherence + divergence modes only.
 */

#include "dtd_estimator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


struct DtdEstimator {
    DtdMode mode;
    float confidence;
    float attack;
    float release;
    int   hangover_max;
    int   hangover_count;
    int   warmup_frames;
    int   frame_count;

    /* Coherence state */
    int      n_freqs;
    float*   S_ex_re;        /* [n_freqs] cross-spec real */
    float*   S_ex_im;        /* [n_freqs] cross-spec imag */
    float*   S_ee;           /* [n_freqs] error PSD */
    float*   S_xx;           /* [n_freqs] far PSD */
    float*   voice_weight;   /* [n_freqs] precomputed weight */
    float    coh_alpha;
    float    coh_high;
    float    coh_low;
    float    coh_energy_floor;
    float    coh_abs_floor;

    /* Divergence state */
    float    divergence_factor;
};


static DtdEstimator* dtd_alloc(void) {
    DtdEstimator* d = (DtdEstimator*)calloc(1, sizeof(DtdEstimator));
    if (!d) return NULL;
    d->attack = 0.3f;
    d->release = 0.05f;
    d->hangover_max = 15;
    d->warmup_frames = 50;
    d->coh_alpha = 0.85f;
    d->coh_high = 0.6f;
    d->coh_low = 0.3f;
    d->coh_energy_floor = 0.01f;
    d->coh_abs_floor = 1e-6f;
    d->divergence_factor = 1.5f;
    return d;
}


DtdEstimator* dtd_create_coherence(int n_freqs, int sample_rate, int block_size) {
    if (n_freqs <= 0) return NULL;
    DtdEstimator* d = dtd_alloc();
    if (!d) return NULL;
    d->mode = DTD_MODE_COHERENCE;
    d->n_freqs = n_freqs;
    d->S_ex_re = (float*)calloc(n_freqs, sizeof(float));
    d->S_ex_im = (float*)calloc(n_freqs, sizeof(float));
    d->S_ee = (float*)calloc(n_freqs, sizeof(float));
    d->S_xx = (float*)calloc(n_freqs, sizeof(float));
    d->voice_weight = (float*)malloc(n_freqs * sizeof(float));
    if (!d->S_ex_re || !d->S_ex_im || !d->S_ee || !d->S_xx || !d->voice_weight) {
        dtd_destroy(d);
        return NULL;
    }
    /* Voice band weight (Python aec.py 2132-2140) */
    float freq_per_bin = (float)sample_rate / (float)block_size;
    for (int k = 0; k < n_freqs; k++) {
        float f = (float)k * freq_per_bin;
        if (f >= 300.0f && f <= 4000.0f) d->voice_weight[k] = 3.0f;
        else if (f < 100.0f || f > 6000.0f) d->voice_weight[k] = 0.3f;
        else d->voice_weight[k] = 1.0f;
    }
    return d;
}


DtdEstimator* dtd_create_divergence(void) {
    DtdEstimator* d = dtd_alloc();
    if (!d) return NULL;
    d->mode = DTD_MODE_DIVERGENCE;
    return d;
}


void dtd_destroy(DtdEstimator* d) {
    if (!d) return;
    free(d->S_ex_re); free(d->S_ex_im);
    free(d->S_ee); free(d->S_xx); free(d->voice_weight);
    free(d);
}


void dtd_reset(DtdEstimator* d) {
    if (!d) return;
    d->confidence = 0.0f;
    d->frame_count = 0;
    d->hangover_count = 0;
    if (d->S_ex_re) {
        memset(d->S_ex_re, 0, d->n_freqs * sizeof(float));
        memset(d->S_ex_im, 0, d->n_freqs * sizeof(float));
        memset(d->S_ee, 0, d->n_freqs * sizeof(float));
        memset(d->S_xx, 0, d->n_freqs * sizeof(float));
    }
}


/* Match Python aec.py 2158-2167 */
static void update_confidence(DtdEstimator* d, int detected) {
    if (detected) {
        d->hangover_count = d->hangover_max;
        float c = d->confidence + d->attack;
        d->confidence = (c > 1.0f) ? 1.0f : c;
    } else if (d->hangover_count > 0) {
        d->hangover_count--;
        float c = d->confidence - d->release * 0.5f;
        d->confidence = (c < 0.0f) ? 0.0f : c;
    } else {
        float c = d->confidence - d->release;
        d->confidence = (c < 0.0f) ? 0.0f : c;
    }
}


float dtd_detect_coherence(DtdEstimator* d, const Complex* error_spec, const Complex* far_spec) {
    if (!d || d->mode != DTD_MODE_COHERENCE) return 0.0f;
    d->frame_count++;
    if (d->frame_count < d->warmup_frames) return 0.0f;

    const float a = d->coh_alpha;
    const int n = d->n_freqs;
    /* Update smoothed PSDs */
    for (int k = 0; k < n; k++) {
        /* cross = error * conj(far) → real = e.r*f.r + e.i*f.i, imag = e.i*f.r - e.r*f.i */
        float cr = error_spec[k].r * far_spec[k].r + error_spec[k].i * far_spec[k].i;
        float ci = error_spec[k].i * far_spec[k].r - error_spec[k].r * far_spec[k].i;
        d->S_ex_re[k] = a * d->S_ex_re[k] + (1.0f - a) * cr;
        d->S_ex_im[k] = a * d->S_ex_im[k] + (1.0f - a) * ci;
        float ee = error_spec[k].r * error_spec[k].r + error_spec[k].i * error_spec[k].i;
        float xx = far_spec[k].r * far_spec[k].r + far_spec[k].i * far_spec[k].i;
        d->S_ee[k] = a * d->S_ee[k] + (1.0f - a) * ee;
        d->S_xx[k] = a * d->S_xx[k] + (1.0f - a) * xx;
    }
    /* Voice-band weighted coherence (ratio of sums) */
    float num = 0.0f, den = 0.0f;
    float sum_ee = 0.0f, sum_xx = 0.0f;
    for (int k = 0; k < n; k++) {
        float w = d->voice_weight[k];
        float sex_mag2 = d->S_ex_re[k] * d->S_ex_re[k] + d->S_ex_im[k] * d->S_ex_im[k];
        num += w * sex_mag2;
        den += w * d->S_ee[k] * d->S_xx[k];
        sum_ee += d->S_ee[k];
        sum_xx += d->S_xx[k];
    }
    float coherence = num / (den + 1e-10f);
    int has_energy = (sum_ee > d->coh_energy_floor * sum_xx)
                     && (sum_xx > 1e-10f)
                     && (sum_ee > d->coh_abs_floor);

    if (coherence > d->coh_high) {
        update_confidence(d, 0);
    } else if (coherence < d->coh_low && has_energy) {
        update_confidence(d, 1);
    } else {
        float c = d->confidence - d->release * 0.5f;
        d->confidence = (c < 0.0f) ? 0.0f : c;
    }
    return d->confidence;
}


float dtd_detect_divergence(DtdEstimator* d, const float* near_end, const float* output, int hop) {
    if (!d || d->mode != DTD_MODE_DIVERGENCE) return 0.0f;
    d->frame_count++;
    if (d->frame_count < d->warmup_frames) return 0.0f;

    float out_e = 0.0f, near_e = 0.0f, out_p = 0.0f, near_p = 0.0f;
    for (int i = 0; i < hop; i++) {
        out_e += output[i] * output[i];
        near_e += near_end[i] * near_end[i];
        float oa = output[i]; if (oa < 0) oa = -oa;
        float na = near_end[i]; if (na < 0) na = -na;
        if (oa > out_p) out_p = oa;
        if (na > near_p) near_p = na;
    }
    out_e /= hop; near_e /= hop;

    if (near_e < 1e-10f && near_p < 1e-6f) {
        float c = d->confidence - d->release;
        d->confidence = (c < 0.0f) ? 0.0f : c;
        return d->confidence;
    }

    float energy_ratio = (near_e > 1e-10f) ? (out_e / (near_e + 1e-10f)) : 0.0f;
    float peak_ratio = (near_p > 1e-6f) ? (out_p / (near_p + 1e-10f)) : 0.0f;
    float ratio = (energy_ratio > peak_ratio) ? energy_ratio : peak_ratio;

    const float mild = 1.2f;
    if (ratio > d->divergence_factor) {
        float c = d->confidence + d->attack;
        d->confidence = (c > 1.0f) ? 1.0f : c;
    } else if (ratio > mild) {
        float c = d->confidence + d->attack * (ratio - mild);
        d->confidence = (c > 1.0f) ? 1.0f : c;
    } else {
        float rel_scale = 1.0f - ratio;
        if (rel_scale < 0.2f) rel_scale = 0.2f;
        float c = d->confidence - d->release * (1.0f + 4.0f * rel_scale);
        d->confidence = (c < 0.0f) ? 0.0f : c;
    }
    return d->confidence;
}


float dtd_get_confidence(const DtdEstimator* d) { return d ? d->confidence : 0.0f; }


void dtd_scale_confidence(DtdEstimator* d, float factor) {
    if (d) d->confidence *= factor;
}
