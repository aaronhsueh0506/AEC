/* render_signal_analyzer.c — byte-equal C port of
 * python/modules/render/render_signal_analyzer.py. See header for parity notes.
 *
 * Constants (mirror the Python module-level constants):
 *   _COUNTER_THRESHOLD        = 5    (mask fires when counter > 5)
 *   _POOR_EXCITATION_THRESHOLD= 10   (counter > 10 → poor_excitation)
 *   _MAX_ABS_THRESHOLD_FLOAT  = 100.0 / 32768.0  (strong-peak amplitude floor)
 *   _PEAK_RATIO_NARROW        = 3.0  (small narrow band)
 *   _PEAK_RATIO_STRONG        = 100.0 (strong peak)
 */
#include "render_signal_analyzer.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define RSA_COUNTER_THRESHOLD          5
#define RSA_POOR_EXCITATION_THRESHOLD  10
#define RSA_MAX_ABS_THRESHOLD_FLOAT    (100.0f / 32768.0f)   /* 3.0517578125e-3 */
#define RSA_PEAK_RATIO_NARROW          3.0f
#define RSA_PEAK_RATIO_STRONG          100.0f

void rsa_init(RenderSignalAnalyzer *m, int64_t *counters_storage, int n_freqs,
              int strong_peak_freeze_duration) {
    m->n_freqs = n_freqs;
    m->n_freqs_by2 = n_freqs - 1;          /* kFftLengthBy2 */
    m->n_counters = m->n_freqs_by2 - 1;    /* = n_freqs - 2 */
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

/* Mirrors _identify_small_narrow_band_regions (cc:33-59). render_psd NULL →
 * zero all counters. Else per bin k in [1, n_freqs_by2):
 *   is_peak = X2[k] > 3.0f * fmaxf(X2[k-1], X2[k+1])   (ALL float32)
 *   counter[k-1] = is_peak ? counter[k-1] + 1 : 0
 * Counter array index runs 0..n_counters-1 for k = 1..n_freqs_by2-1. */
static void identify_small_narrow_band_regions(RenderSignalAnalyzer *m,
                                               const float *render_psd) {
    int k;
    if (render_psd == NULL) {
        memset(m->counters, 0, (size_t)m->n_counters * sizeof(int64_t));
        return;
    }
    /* k_lo=1, k_hi=n_freqs_by2 (exclusive). Index into counters is k-1. */
    for (k = 1; k < m->n_freqs_by2; ++k) {
        float x_center = render_psd[k];
        float x_left   = render_psd[k - 1];
        float x_right  = render_psd[k + 1];
        float neigh    = x_left > x_right ? x_left : x_right; /* np.maximum */
        float thresh   = RSA_PEAK_RATIO_NARROW * neigh; /* f32*f32 -> f32 */
        int idx = k - 1;
        if (x_center > thresh) {
            m->counters[idx] = m->counters[idx] + 1;
        } else {
            m->counters[idx] = 0;
        }
    }
}

/* Mirrors _identify_strong_narrow_band_component (cc:62-121). */
static void identify_strong_narrow_band_component(RenderSignalAnalyzer *m,
                                                  const float *render_psd,
                                                  const float *render_block,
                                                  int render_block_len) {
    int peak_bin, lo_lo, lo_hi, hi_lo, hi_hi, i;
    float non_peak_power, max_abs, peak_level;
    float wmax;

    /* Tick down freeze counter. */
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

    /* peak_bin = int(np.argmax(X2)) — first index of the maximum. */
    peak_bin = 0;
    for (i = 1; i < m->n_freqs; ++i) {
        if (render_psd[i] > render_psd[peak_bin]) {
            peak_bin = i;
        }
    }

    /* Non-peak power windows (avoiding ±5 bins around peak). */
    lo_lo = peak_bin - 14; if (lo_lo < 0) lo_lo = 0;
    lo_hi = peak_bin - 4;  if (lo_hi < lo_lo) lo_hi = lo_lo;
    hi_lo = peak_bin + 5;  if (hi_lo > m->n_freqs) hi_lo = m->n_freqs;
    hi_hi = peak_bin + 15; if (hi_hi > m->n_freqs) hi_hi = m->n_freqs;

    non_peak_power = 0.0f;
    if (lo_hi > lo_lo) {
        wmax = render_psd[lo_lo];               /* np.max over float32 window */
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

    /* max_abs of render time block: np.max(np.abs(render_block)), kept in
     * float32 (float32-by-design; formerly widened via float() to f64). */
    {
        float amax = (render_block_len > 0) ? fabsf(render_block[0]) : 0.0f;
        for (i = 1; i < render_block_len; ++i) {
            float a = fabsf(render_block[i]);
            if (a > amax) amax = a;
        }
        max_abs = amax;
    }

    peak_level = render_psd[peak_bin];   /* X2[peak_bin], float32-by-design */

    if (peak_bin > 0
            && max_abs > RSA_MAX_ABS_THRESHOLD_FLOAT
            && peak_level > RSA_PEAK_RATIO_STRONG * non_peak_power) {
        m->narrow_peak_band = peak_bin;
        m->narrow_peak_counter = 0;
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

int rsa_narrow_peak_band(const RenderSignalAnalyzer *m) {
    return m->narrow_peak_band;
}

/* In-place mask: zero mu for ±2 bins around any peak with counter > 5.
 *   counter[0]   > thr → mu[0], mu[1]
 *   counter[k-1] > thr → mu[k-2..k+2]   (k in [2, N-1))
 *   counter[N-2] > thr → mu[N-1], mu[N]     where N = n_freqs_by2
 */
void rsa_mask_regions_around_narrow_bands(const RenderSignalAnalyzer *m, float *mu) {
    int N = m->n_freqs_by2;  /* = n_freqs - 1 */
    int k, j;

    if (m->counters[0] > RSA_COUNTER_THRESHOLD) {
        mu[0] = 0.0f;
        mu[1] = 0.0f;
    }
    for (k = 2; k < N - 1; ++k) {
        if (m->counters[k - 1] > RSA_COUNTER_THRESHOLD) {
            /* mu[k-2 : k+3] = 0.0  (k+2 inclusive, clamped to n_freqs-1) */
            for (j = k - 2; j <= k + 2; ++j) {
                if (j >= 0 && j < m->n_freqs) mu[j] = 0.0f;
            }
        }
    }
    if (m->counters[N - 2] > RSA_COUNTER_THRESHOLD) {
        mu[N - 1] = 0.0f;
        mu[N] = 0.0f;
    }
}
