/* render_signal_analyzer.c -- see render_signal_analyzer.h.
 * Thresholds: mask when counter > 5, poor excitation when counter > 10, strong-peak amplitude floor
 * 100/32768, narrow-peak ratio 3, strong-peak ratio 100. */
#include "render_signal_analyzer.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "utility/simd_aec.h"

#define RSA_COUNTER_THRESHOLD          5
#define RSA_POOR_EXCITATION_THRESHOLD  10
#define RSA_MAX_ABS_THRESHOLD_FLOAT    (100.0f / 32768.0f)
#define RSA_PEAK_RATIO_NARROW          3.0f
#define RSA_PEAK_RATIO_STRONG          100.0f

void rsa_init(RenderSignalAnalyzer *m, int64_t *counters_storage, int n_freqs,
              int strong_peak_freeze_duration) {
    m->n_freqs = n_freqs;
    m->n_freqs_by2 = n_freqs - 1;
    m->n_counters = m->n_freqs_by2 - 1;
    m->counters = counters_storage;
    m->strong_peak_freeze_duration = strong_peak_freeze_duration;
    m->narrow_peak_band = -1;
    m->narrow_peak_counter = 0;
    memset(m->counters, 0, (size_t)m->n_counters * sizeof(int64_t));
}

void rsa_reset(RenderSignalAnalyzer *m) {
    memset(m->counters, 0, (size_t)m->n_counters * sizeof(int64_t));
    m->narrow_peak_band = -1;
    m->narrow_peak_counter = 0;
}

/* Per bin k in [1, n_freqs_by2): counter[k-1] = (X2[k] > 3 * max(X2[k-1], X2[k+1]))
 * ? counter[k-1] + 1 : 0. A NULL render_psd zeroes all counters. */
static void identify_small_narrow_band_regions(RenderSignalAnalyzer *m,
                                               const float *render_psd) {
    if (render_psd == NULL) {
        memset(m->counters, 0, (size_t)m->n_counters * sizeof(int64_t));
        return;
    }
    sk_peak_counters_i64(m->counters, render_psd,
                         RSA_PEAK_RATIO_NARROW, m->n_counters);
}

/* First maximum index (ties resolve to the lowest). */
static int rsa_argmax_first(const float *x, int n) {
    return sk_argmax_f32(x, n);
}

/* Largest |x[i]|. */
static float rsa_max_abs(const float *x, int n) {
    return sk_max_abs_f32(x, n);
}

/* Strong narrow-band component detection. */
static void identify_strong_narrow_band_component(RenderSignalAnalyzer *m,
                                                  const float *render_psd,
                                                  const float *render_block,
                                                  int render_block_len) {
    int peak_bin, lo_lo, lo_hi, hi_lo, hi_hi, i;
    float non_peak_power, max_abs, peak_level;
    float wmax;

    /* Tick down the freeze counter. */
    if (m->narrow_peak_band != -1) {
        m->narrow_peak_counter += 1;
        if (m->narrow_peak_counter > m->strong_peak_freeze_duration) {
            m->narrow_peak_band = -1;
            m->narrow_peak_counter = 0;
        }
    }
    if (render_psd == NULL || render_block == NULL) {
        return;
    }

    peak_bin = rsa_argmax_first(render_psd, m->n_freqs);

    /* Only the peak_bin > 0 branch observes the window scan below, so skip it otherwise. */
    if (peak_bin > 0) {
        /* Non-peak windows, excluding +-5 bins around the peak. */
        lo_lo = peak_bin - 14; if (lo_lo < 0) lo_lo = 0;
        lo_hi = peak_bin - 4;  if (lo_hi < lo_lo) lo_hi = lo_lo;
        hi_lo = peak_bin + 5;  if (hi_lo > m->n_freqs) hi_lo = m->n_freqs;
        hi_hi = peak_bin + 15; if (hi_hi > m->n_freqs) hi_hi = m->n_freqs;

        non_peak_power = 0.0f;
        if (lo_hi > lo_lo) {
            wmax = render_psd[lo_lo];
            for (i = lo_lo + 1; i < lo_hi; ++i) {
                if (render_psd[i] > wmax) wmax = render_psd[i];
            }
            if (wmax > non_peak_power) non_peak_power = wmax;
        }
        if (hi_hi > hi_lo) {
            wmax = render_psd[hi_lo];
            for (i = hi_lo + 1; i < hi_hi; ++i) {
                if (render_psd[i] > wmax) wmax = render_psd[i];
            }
            if (wmax > non_peak_power) non_peak_power = wmax;
        }

        peak_level = render_psd[peak_bin];

        /* The cheaper peak_level check runs first; max_abs is O(render_block_len). */
        if (peak_level > RSA_PEAK_RATIO_STRONG * non_peak_power) {
            max_abs = rsa_max_abs(render_block, render_block_len);
            if (max_abs > RSA_MAX_ABS_THRESHOLD_FLOAT) {
                m->narrow_peak_band = peak_bin;
                m->narrow_peak_counter = 0;
            }
        }
    }
}

void rsa_update(RenderSignalAnalyzer *m, const float *render_psd,
                const float *render_block, int render_block_len) {
    identify_small_narrow_band_regions(m, render_psd);
    identify_strong_narrow_band_component(m, render_psd, render_block,
                                          render_block_len);
}

int rsa_poor_signal_excitation(const RenderSignalAnalyzer *m) {
    int i;
    if (m->n_counters == 0) return 0;
    for (i = 0; i < m->n_counters; ++i) {
        if (m->counters[i] > RSA_POOR_EXCITATION_THRESHOLD) return 1;
    }
    return 0;
}

/* Returns 1 iff the mask function would zero at least one entry. */
int rsa_any_narrow_band_active(const RenderSignalAnalyzer *m) {
    int N = m->n_freqs_by2;
    int k;
    if (m->counters[0] > RSA_COUNTER_THRESHOLD) return 1;
    for (k = 2; k < N - 1; ++k) {
        if (m->counters[k - 1] > RSA_COUNTER_THRESHOLD) return 1;
    }
    if (m->counters[N - 2] > RSA_COUNTER_THRESHOLD) return 1;
    return 0;
}

void rsa_mask_regions_around_narrow_bands(const RenderSignalAnalyzer *m, float *mu) {
    int N = m->n_freqs_by2;
    int k, j;

    if (m->counters[0] > RSA_COUNTER_THRESHOLD) {
        mu[0] = 0.0f;
        mu[1] = 0.0f;
    }
    for (k = 2; k < N - 1; ++k) {
        if (m->counters[k - 1] > RSA_COUNTER_THRESHOLD) {
            for (j = k - 2; j <= k + 2; ++j) mu[j] = 0.0f;
        }
    }
    if (m->counters[N - 2] > RSA_COUNTER_THRESHOLD) {
        mu[N - 1] = 0.0f;
        mu[N] = 0.0f;
    }
}
