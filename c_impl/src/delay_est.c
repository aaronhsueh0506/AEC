/* delay_est.c — 1:1 port of python/aec.py DelayEstimator.
 *
 * Differences from Python (intentional, allowed by parity gate):
 *  - Python uses pocketfft (numpy); C uses kiss_fft. FFT outputs differ
 *    in the last few bits — parity gate is rtol < 1e-3 on PAR and exact on
 *    integer delay output / counters.
 *  - Cross-spectrum EMA stays in fp64 (matches Python np.complex128).
 *
 * User Fix 5: PAR mean excludes peak — Python aec.py:515:
 *   mean_excl = (np.sum(np.abs(pos_range)) - peak) / (len(pos_range)-1+1e-10)
 * We match this exactly.
 */
#include "delay_est.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compute seg_size from sample_rate + max_delay_ms. Pure helper used by
 * both init paths and get_mem_size to keep sizing consistent. */
static int delay_est_compute_seg_size(int sample_rate, double max_delay_ms) {
    int max_delay_samples = (int)(max_delay_ms * sample_rate / 1000.0);
    int min_seg = 2 * max_delay_samples;
    if (min_seg < 2048) min_seg = 2048;
    int seg = 1;
    while (seg < min_seg) seg *= 2;
    return seg;
}

/* Common state initialization (post-buffer-placement). */
static void delay_est_init_state(DelayEst* d, int sample_rate,
                                  double max_delay_ms,
                                  double init_seconds,
                                  double period_seconds, int seg) {
    d->sample_rate       = sample_rate;
    d->max_delay_samples = (int)(max_delay_ms * sample_rate / 1000.0);
    d->init_seconds      = init_seconds;
    d->period_seconds    = period_seconds;
    d->par_low_threshold   = 5.0;
    d->par_solid_threshold = 8.0;
    d->seg_size          = seg;
    d->seg_hop           = seg / 2;
    d->n_freqs           = seg / 2 + 1;
    d->alpha             = 0.6;
    d->n_updates         = 0;
    d->buf_pos           = 0;
    d->estimated_delay     = -1;
    d->samples_accumulated = 0;
    d->samples_since_est   = 0;
    d->init_done           = 0;
    d->init_samples        = (int)(init_seconds   * sample_rate);
    d->period_samples      = (int)(period_seconds * sample_rate);
    d->n_estimates         = 0;
    d->last_par            = 0.0;
    /* P3c Phase 1a — high-PAR fast-path. Defaults mirror AecConfig. */
    d->prev_estimated_delay = -1;
    d->fast_path_enabled   = 1;
    d->fast_par_threshold  = 40.0;
}

void delay_est_init(DelayEst* d, int sample_rate,
                       double max_delay_ms,
                       double init_seconds,
                       double period_seconds) {
    int seg = delay_est_compute_seg_size(sample_rate, max_delay_ms);
    delay_est_init_state(d, sample_rate, max_delay_ms,
                         init_seconds, period_seconds, seg);

    d->cross_re = (double*)calloc((size_t)d->n_freqs, sizeof(double));
    d->cross_im = (double*)calloc((size_t)d->n_freqs, sizeof(double));
    d->mic_buf = (float*)calloc((size_t)d->seg_size, sizeof(float));
    d->ref_buf = (float*)calloc((size_t)d->seg_size, sizeof(float));
    d->fft      = fft_create(d->seg_size);
    d->mic_spec = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->ref_spec = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->phat     = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->gcc      = (float*)calloc((size_t)d->seg_size, sizeof(float));
    d->is_static = 0;
}

size_t delay_est_get_mem_size(int sample_rate, double max_delay_ms) {
    int seg = delay_est_compute_seg_size(sample_rate, max_delay_ms);
    int K   = seg / 2 + 1;
    size_t total = 0;
    total += ALIGN16((size_t)K   * sizeof(double));   /* cross_re */
    total += ALIGN16((size_t)K   * sizeof(double));   /* cross_im */
    total += ALIGN16((size_t)seg * sizeof(float));    /* mic_buf */
    total += ALIGN16((size_t)seg * sizeof(float));    /* ref_buf */
    total += fft_get_mem_size(seg);                   /* nested FFT */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* mic_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* ref_spec */
    total += ALIGN16((size_t)K   * sizeof(Complex));  /* phat */
    total += ALIGN16((size_t)seg * sizeof(float));    /* gcc */
    return total;
}

void delay_est_init_static(DelayEst* d, void* mem, size_t mem_size,
                            int sample_rate, double max_delay_ms,
                            double init_seconds, double period_seconds) {
    if (!d || !mem) return;
    if (mem_size < delay_est_get_mem_size(sample_rate, max_delay_ms)) return;

    int seg = delay_est_compute_seg_size(sample_rate, max_delay_ms);
    memset(d, 0, sizeof(*d));
    delay_est_init_state(d, sample_rate, max_delay_ms,
                         init_seconds, period_seconds, seg);

    uint8_t* ptr = (uint8_t*)mem;
    int K = d->n_freqs;

    d->cross_re = (double*)ptr; ptr += ALIGN16((size_t)K * sizeof(double));
    d->cross_im = (double*)ptr; ptr += ALIGN16((size_t)K * sizeof(double));
    d->mic_buf  = (float*)ptr;  ptr += ALIGN16((size_t)seg * sizeof(float));
    d->ref_buf  = (float*)ptr;  ptr += ALIGN16((size_t)seg * sizeof(float));
    size_t fft_sz = fft_get_mem_size(seg);
    d->fft      = fft_init(ptr, fft_sz, seg); ptr += fft_sz;
    d->mic_spec = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    d->ref_spec = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    d->phat     = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    d->gcc      = (float*)ptr;   /* last field */

    memset(d->cross_re, 0, (size_t)K   * sizeof(double));
    memset(d->cross_im, 0, (size_t)K   * sizeof(double));
    memset(d->mic_buf,  0, (size_t)seg * sizeof(float));
    memset(d->ref_buf,  0, (size_t)seg * sizeof(float));
    memset(d->mic_spec, 0, (size_t)K   * sizeof(Complex));
    memset(d->ref_spec, 0, (size_t)K   * sizeof(Complex));
    memset(d->phat,     0, (size_t)K   * sizeof(Complex));
    memset(d->gcc,      0, (size_t)seg * sizeof(float));
    d->is_static = 1;
}

void delay_est_free(DelayEst* d) {
    if (!d) return;
    if (d->is_static) {
        if (d->fft) fft_destroy(d->fft);  /* no-op when FFT also static */
        return;
    }
    free(d->cross_re); free(d->cross_im);
    free(d->mic_buf);  free(d->ref_buf);
    free(d->mic_spec); free(d->ref_spec);
    free(d->phat);     free(d->gcc);
    if (d->fft) fft_destroy(d->fft);
}

void delay_est_reset(DelayEst* d) {
    memset(d->cross_re, 0, (size_t)d->n_freqs * sizeof(double));
    memset(d->cross_im, 0, (size_t)d->n_freqs * sizeof(double));
    memset(d->mic_buf, 0, (size_t)d->seg_size * sizeof(float));
    memset(d->ref_buf, 0, (size_t)d->seg_size * sizeof(float));
    d->buf_pos             = 0;
    d->n_updates           = 0;
    d->estimated_delay     = -1;
    d->prev_estimated_delay = -1;
    d->samples_accumulated = 0;
    d->samples_since_est   = 0;
    d->init_done           = 0;
    d->n_estimates         = 0;
}

static void update_cross_spectrum(DelayEst* d) {
    /* numpy rfft uses unnormalized forward (matches kiss_fft) */
    fft_forward(d->fft, d->mic_buf, d->mic_spec);
    fft_forward(d->fft, d->ref_buf, d->ref_spec);

    d->n_updates++;
    /* cross[k] = mic_spec[k] * conj(ref_spec[k])
     *         = (Mr + jMi) * (Rr - jRi)
     *         = (Mr*Rr + Mi*Ri) + j*(Mi*Rr - Mr*Ri) */
    if (d->n_updates == 1) {
        for (int k = 0; k < d->n_freqs; ++k) {
            double Mr = d->mic_spec[k].r, Mi = d->mic_spec[k].i;
            double Rr = d->ref_spec[k].r, Ri = d->ref_spec[k].i;
            d->cross_re[k] = Mr * Rr + Mi * Ri;
            d->cross_im[k] = Mi * Rr - Mr * Ri;
        }
    } else {
        const double a = d->alpha, b = 1.0 - d->alpha;
        for (int k = 0; k < d->n_freqs; ++k) {
            double Mr = d->mic_spec[k].r, Mi = d->mic_spec[k].i;
            double Rr = d->ref_spec[k].r, Ri = d->ref_spec[k].i;
            double cr = Mr * Rr + Mi * Ri;
            double ci = Mi * Rr - Mr * Ri;
            d->cross_re[k] = a * d->cross_re[k] + b * cr;
            d->cross_im[k] = a * d->cross_im[k] + b * ci;
        }
    }
}

static void estimate(DelayEst* d) {
    /* phat[k] = cross[k] / (|cross[k]| + 1e-10) */
    for (int k = 0; k < d->n_freqs; ++k) {
        double cr  = d->cross_re[k];
        double ci  = d->cross_im[k];
        double mag = sqrt(cr * cr + ci * ci) + 1e-10;
        d->phat[k].r = (float)(cr / mag);
        d->phat[k].i = (float)(ci / mag);
    }
    /* fft_inverse already returns time-domain real samples; check whether
     * it normalises. fft_wrapper does not normalise (kiss_fft inverse just
     * scales by N). numpy irfft normalises by 1/N, so apply 1/N. */
    fft_inverse(d->fft, d->phat, d->gcc);
    {
        const double inv_n = 1.0 / (double)d->seg_size;
        for (int k = 0; k < d->seg_size; ++k) {
            d->gcc[k] = (float)((double)d->gcc[k] * inv_n);
        }
    }

    int max_d = d->max_delay_samples;
    if (max_d > d->seg_size / 2) max_d = d->seg_size / 2;
    int n_pos = max_d + 1;

    /* argmax over |gcc[0..max_d]| */
    int   best_pos = 0;
    double best_abs = fabs((double)d->gcc[0]);
    for (int k = 1; k < n_pos; ++k) {
        double a = fabs((double)d->gcc[k]);
        if (a > best_abs) { best_abs = a; best_pos = k; }
    }

    /* PAR with peak excluded from mean (User Fix 5) */
    double sum_abs = 0.0;
    for (int k = 0; k < n_pos; ++k) sum_abs += fabs((double)d->gcc[k]);
    double mean_excl = (sum_abs - best_abs) / ((double)(n_pos - 1) + 1e-10);
    d->last_par = best_abs / (mean_excl + 1e-10);

    /* P3c Phase 1a: remember the previous estimate's lag before
     * overwriting, so the fast-path's same-lag-twice guard can fire. */
    d->prev_estimated_delay = d->estimated_delay;
    d->estimated_delay   = best_pos;
    d->samples_since_est = 0;
    d->n_estimates++;
}

int delay_est_accumulate(DelayEst* d,
                            const float* mic, const float* ref, size_t n_in) {
    int n = (int)n_in;
    d->samples_accumulated += n;
    d->samples_since_est   += n;

    int remaining = n, src_pos = 0;
    while (remaining > 0) {
        int space = d->seg_size - d->buf_pos;
        int chunk = (remaining < space) ? remaining : space;
        memcpy(d->mic_buf + d->buf_pos, mic + src_pos, (size_t)chunk * sizeof(float));
        memcpy(d->ref_buf + d->buf_pos, ref + src_pos, (size_t)chunk * sizeof(float));
        d->buf_pos += chunk;
        src_pos    += chunk;
        remaining  -= chunk;

        if (d->buf_pos >= d->seg_size) {
            update_cross_spectrum(d);
            /* Shift by seg_hop (50% overlap) */
            memmove(d->mic_buf, d->mic_buf + d->seg_hop,
                    (size_t)d->seg_hop * sizeof(float));
            memmove(d->ref_buf, d->ref_buf + d->seg_hop,
                    (size_t)d->seg_hop * sizeof(float));
            d->buf_pos = d->seg_hop;
        }
    }

    if (d->n_updates < 2) return 0;
    /* fallthrough handled below */

    if (!d->init_done) {
        if (d->samples_accumulated >= d->init_samples) {
            estimate(d);
            d->init_done = 1;
            return 1;
        }
    } else {
        if (d->samples_since_est >= d->period_samples) {
            estimate(d);
            return 1;
        }
    }
    return 0;
}

/* v3.10.0 — confidence + is_solid. Mirrors aec.py confidence() property.
 * P3c Phase 1a: high-PAR fast-path branch precedes the legacy
 * n_updates >= 3 gate (default ON). */
double delay_est_confidence(const DelayEst* d) {
    if (!d) return 0.0;
    if (d->estimated_delay < 0) return 0.0;
    double par = d->last_par;
    if (d->fast_path_enabled
        && d->n_updates >= 2
        && par >= d->fast_par_threshold
        && d->prev_estimated_delay == d->estimated_delay
        && d->prev_estimated_delay >= 0) {
        return 1.0;
    }
    if (d->n_updates < 3) return 0.0;
    double lo  = d->par_low_threshold;
    double hi  = d->par_solid_threshold;
    if (par <= lo) return 0.0;
    if (par >= hi) return 1.0;
    return (par - lo) / (hi - lo);
}

int delay_est_is_solid(const DelayEst* d) {
    return delay_est_confidence(d) >= 1.0 ? 1 : 0;
}
