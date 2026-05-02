/* delay_est_v2.c — 1:1 port of python/aec.py DelayEstimator.
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
#include "delay_est_v2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void delay_est_v2_init(DelayEstV2* d, int sample_rate,
                       double max_delay_ms,
                       double init_seconds,
                       double period_seconds) {
    d->sample_rate       = sample_rate;
    d->max_delay_samples = (int)(max_delay_ms * sample_rate / 1000.0);
    d->init_seconds      = init_seconds;
    d->period_seconds    = period_seconds;

    /* Segment size = next power of 2 ≥ max(2048, 2*max_delay_samples) */
    int min_seg = 2 * d->max_delay_samples;
    if (min_seg < 2048) min_seg = 2048;
    int seg = 1;
    while (seg < min_seg) seg *= 2;
    d->seg_size = seg;
    d->seg_hop  = seg / 2;
    d->n_freqs  = seg / 2 + 1;

    d->cross_re = (double*)calloc((size_t)d->n_freqs, sizeof(double));
    d->cross_im = (double*)calloc((size_t)d->n_freqs, sizeof(double));
    d->alpha     = 0.6;
    d->n_updates = 0;

    d->mic_buf = (float*)calloc((size_t)d->seg_size, sizeof(float));
    d->ref_buf = (float*)calloc((size_t)d->seg_size, sizeof(float));
    d->buf_pos = 0;

    d->estimated_delay     = -1;
    d->samples_accumulated = 0;
    d->samples_since_est   = 0;
    d->init_done           = 0;
    d->init_samples        = (int)(init_seconds   * sample_rate);
    d->period_samples      = (int)(period_seconds * sample_rate);
    d->n_estimates         = 0;
    d->last_par            = 0.0;

    d->fft      = fft_create(d->seg_size);
    d->mic_spec = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->ref_spec = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->phat     = (Complex*)calloc((size_t)d->n_freqs, sizeof(Complex));
    d->gcc      = (float*)calloc((size_t)d->seg_size, sizeof(float));
}

void delay_est_v2_free(DelayEstV2* d) {
    free(d->cross_re); free(d->cross_im);
    free(d->mic_buf);  free(d->ref_buf);
    free(d->mic_spec); free(d->ref_spec);
    free(d->phat);     free(d->gcc);
    if (d->fft) fft_destroy(d->fft);
}

void delay_est_v2_reset(DelayEstV2* d) {
    memset(d->cross_re, 0, (size_t)d->n_freqs * sizeof(double));
    memset(d->cross_im, 0, (size_t)d->n_freqs * sizeof(double));
    memset(d->mic_buf, 0, (size_t)d->seg_size * sizeof(float));
    memset(d->ref_buf, 0, (size_t)d->seg_size * sizeof(float));
    d->buf_pos             = 0;
    d->n_updates           = 0;
    d->estimated_delay     = -1;
    d->samples_accumulated = 0;
    d->samples_since_est   = 0;
    d->init_done           = 0;
    d->n_estimates         = 0;
}

static void update_cross_spectrum(DelayEstV2* d) {
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

static void estimate(DelayEstV2* d) {
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

    d->estimated_delay   = best_pos;
    d->samples_since_est = 0;
    d->n_estimates++;
}

int delay_est_v2_accumulate(DelayEstV2* d,
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
