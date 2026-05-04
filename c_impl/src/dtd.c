/* dtd.c — port of aec.py DtdEstimator. */
#include "dtd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void common_init(DtdEstimator* d, double attack, double release,
                        int warmup_frames) {
    d->confidence = 0.0;
    d->attack     = attack;
    d->release    = release;
    d->warmup_frames = warmup_frames;
    d->frame_count   = 0;
    d->far_abs_buffer = NULL; d->buf_len = 0; d->buf_idx = 0;
    d->geigel_threshold = 0.5; d->hangover_max = 0; d->hangover_count = 0;
    d->divergence_factor = 1.5;
    d->n_freqs = 0;
    d->S_ex = NULL; d->S_ee = NULL; d->S_xx = NULL; d->voice_weight = NULL;
    d->coh_alpha = 0.85; d->coh_high = 0.6; d->coh_low = 0.3;
    d->coh_energy_floor = 0.01; d->coh_abs_floor = 1e-6;
}

void dtd_init_geigel(DtdEstimator* d, int window_blocks,
                        double geigel_threshold, int hangover_max,
                        double attack, double release, int warmup_frames) {
    common_init(d, attack, release, warmup_frames);
    d->mode = DTD_GEIGEL;
    int wb = window_blocks > 0 ? window_blocks : 1;
    d->far_abs_buffer = (double*)calloc((size_t)wb, sizeof(double));
    d->buf_len = wb;
    d->geigel_threshold = geigel_threshold;
    d->hangover_max = hangover_max;
}

void dtd_init_divergence(DtdEstimator* d, double divergence_factor,
                            double attack, double release, int warmup_frames) {
    common_init(d, attack, release, warmup_frames);
    d->mode = DTD_DIVERGENCE;
    d->divergence_factor = divergence_factor;
}

void dtd_init_coherence(DtdEstimator* d, int n_freqs,
                           int sample_rate, int block_size,
                           double coh_alpha, double coh_high, double coh_low,
                           double coh_energy_floor, double coh_abs_floor,
                           double attack, double release, int warmup_frames) {
    common_init(d, attack, release, warmup_frames);
    d->mode = DTD_COHERENCE;
    d->n_freqs = n_freqs;
    d->coh_alpha = coh_alpha;
    d->coh_high  = coh_high;
    d->coh_low   = coh_low;
    d->coh_energy_floor = coh_energy_floor;
    d->coh_abs_floor    = coh_abs_floor;
    d->S_ex = (Complex*)calloc((size_t)n_freqs, sizeof(Complex));
    d->S_ee = (float*)calloc((size_t)n_freqs, sizeof(float));
    d->S_xx = (float*)calloc((size_t)n_freqs, sizeof(float));
    d->voice_weight = (float*)malloc((size_t)n_freqs * sizeof(float));
    double freq_per_bin = (double)sample_rate / (double)block_size;
    for (int k = 0; k < n_freqs; ++k) {
        double f = k * freq_per_bin;
        if (f >= 300.0 && f <= 4000.0) d->voice_weight[k] = 3.0f;
        else if (f < 100.0 || f > 6000.0) d->voice_weight[k] = 0.3f;
        else d->voice_weight[k] = 1.0f;
    }
}

void dtd_free(DtdEstimator* d) {
    free(d->far_abs_buffer);
    free(d->S_ex); free(d->S_ee); free(d->S_xx); free(d->voice_weight);
    d->far_abs_buffer = NULL;
    d->S_ex = NULL; d->S_ee = NULL; d->S_xx = NULL; d->voice_weight = NULL;
}

void dtd_reset(DtdEstimator* d) {
    d->confidence = 0.0;
    d->frame_count = 0;
    d->buf_idx = 0;
    d->hangover_count = 0;
    if (d->far_abs_buffer) memset(d->far_abs_buffer, 0,
                                  (size_t)d->buf_len * sizeof(double));
    if (d->S_ex) memset(d->S_ex, 0, (size_t)d->n_freqs * sizeof(Complex));
    if (d->S_ee) memset(d->S_ee, 0, (size_t)d->n_freqs * sizeof(float));
    if (d->S_xx) memset(d->S_xx, 0, (size_t)d->n_freqs * sizeof(float));
}

static void update_confidence(DtdEstimator* d, int detected) {
    if (detected) {
        d->hangover_count = d->hangover_max;
        d->confidence += d->attack;
        if (d->confidence > 1.0) d->confidence = 1.0;
    } else if (d->hangover_count > 0) {
        d->hangover_count--;
        d->confidence -= d->release * 0.5;
        if (d->confidence < 0.0) d->confidence = 0.0;
    } else {
        d->confidence -= d->release;
        if (d->confidence < 0.0) d->confidence = 0.0;
    }
}

static void detect_geigel(DtdEstimator* d, const float* near_end, int n,
                          const float* far_end, int nf) {
    double far_abs_max = 0.0;
    for (int i = 0; i < nf; ++i) {
        double a = fabs((double)far_end[i]);
        if (a > far_abs_max) far_abs_max = a;
    }
    d->far_abs_buffer[d->buf_idx] = far_abs_max;
    d->buf_idx = (d->buf_idx + 1) % d->buf_len;
    double far_max = 0.0;
    for (int i = 0; i < d->buf_len; ++i) {
        if (d->far_abs_buffer[i] > far_max) far_max = d->far_abs_buffer[i];
    }
    double near_max = 0.0;
    for (int i = 0; i < n; ++i) {
        double a = fabs((double)near_end[i]);
        if (a > near_max) near_max = a;
    }
    int detected = (far_max > 1e-6) && (near_max > d->geigel_threshold * far_max);
    update_confidence(d, detected);
}

static void detect_divergence(DtdEstimator* d, const float* near_end, int n,
                              const float* output) {
    double out_e = 0.0, near_e = 0.0;
    double out_p = 0.0, near_p = 0.0;
    for (int i = 0; i < n; ++i) {
        double on = output[i], ne = near_end[i];
        out_e += on * on; near_e += ne * ne;
        double ao = fabs(on), an = fabs(ne);
        if (ao > out_p) out_p = ao;
        if (an > near_p) near_p = an;
    }
    out_e /= n; near_e /= n;
    if (near_e < 1e-10 && near_p < 1e-6) {
        d->confidence -= d->release;
        if (d->confidence < 0.0) d->confidence = 0.0;
        return;
    }
    double energy_ratio = (near_e > 1e-10) ? out_e / (near_e + 1e-10) : 0.0;
    double peak_ratio   = (near_p > 1e-6) ? out_p / (near_p + 1e-10) : 0.0;
    double ratio = energy_ratio > peak_ratio ? energy_ratio : peak_ratio;
    const double mild = 1.2;
    if (ratio > d->divergence_factor) {
        d->confidence += d->attack;
        if (d->confidence > 1.0) d->confidence = 1.0;
    } else if (ratio > mild) {
        d->confidence += d->attack * (ratio - mild);
        if (d->confidence > 1.0) d->confidence = 1.0;
    } else {
        double scale = 1.0 - ratio;
        if (scale < 0.2) scale = 0.2;
        d->confidence -= d->release * (1.0 + 4.0 * scale);
        if (d->confidence < 0.0) d->confidence = 0.0;
    }
}

static void detect_coherence(DtdEstimator* d,
                             const Complex* error_spec, const Complex* far_spec) {
    double a = d->coh_alpha, oa = 1.0 - d->coh_alpha;
    /* cross = error * conj(far) */
    for (int k = 0; k < d->n_freqs; ++k) {
        float er = error_spec[k].r, ei = error_spec[k].i;
        float fr = far_spec[k].r,   fi = far_spec[k].i;
        float cr = er * fr + ei * fi;
        float ci = ei * fr - er * fi;
        d->S_ex[k].r = (float)(a * d->S_ex[k].r + oa * cr);
        d->S_ex[k].i = (float)(a * d->S_ex[k].i + oa * ci);
        float epsd = er * er + ei * ei;
        float xpsd = fr * fr + fi * fi;
        d->S_ee[k] = (float)(a * d->S_ee[k] + oa * epsd);
        d->S_xx[k] = (float)(a * d->S_xx[k] + oa * xpsd);
    }
    double num = 0.0, den = 0.0;
    double sum_ee = 0.0, sum_xx = 0.0;
    for (int k = 0; k < d->n_freqs; ++k) {
        double w = d->voice_weight[k];
        double sx_r = d->S_ex[k].r, sx_i = d->S_ex[k].i;
        num += w * (sx_r * sx_r + sx_i * sx_i);
        den += w * (double)d->S_ee[k] * (double)d->S_xx[k];
        sum_ee += d->S_ee[k];
        sum_xx += d->S_xx[k];
    }
    double coherence = num / (den + 1e-10);
    int has_energy = (sum_ee > d->coh_energy_floor * sum_xx)
                  && (sum_xx > 1e-10) && (sum_ee > d->coh_abs_floor);
    if (coherence > d->coh_high) {
        update_confidence(d, 0);
    } else if (coherence < d->coh_low && has_energy) {
        update_confidence(d, 1);
    } else {
        d->confidence -= d->release * 0.5;
        if (d->confidence < 0.0) d->confidence = 0.0;
    }
}

double dtd_detect_block(DtdEstimator* d,
                           const float* near_end, int near_len,
                           const float* far_end, int far_len,
                           const float* output,
                           const Complex* error_spec,
                           const Complex* far_spec) {
    d->frame_count++;
    if (d->frame_count < d->warmup_frames) return 0.0;
    switch (d->mode) {
        case DTD_GEIGEL:
            detect_geigel(d, near_end, near_len, far_end, far_len);
            break;
        case DTD_COHERENCE:
            if (error_spec && far_spec) detect_coherence(d, error_spec, far_spec);
            break;
        case DTD_DIVERGENCE:
            if (output) detect_divergence(d, near_end, near_len, output);
            break;
    }
    return d->confidence;
}
