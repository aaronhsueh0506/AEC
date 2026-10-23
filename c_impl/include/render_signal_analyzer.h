/* Per-hop render-signal analyser.
 *  1. Per-bin tonal counters; rsa_mask_regions_around_narrow_bands() zeroes mu for +-2 bins
 *  around any sustained narrow peak (counter > 5).
 *  2. rsa_poor_signal_excitation(): any counter > 10 means the filter update should freeze.
 *  3. narrow_peak_band: dominant strong narrow peak (internal state).
 * Inputs are float32; the peak test runs in float32 (x_center > 3 * max(x_left, x_right)).
 * Python reference: modules/render/render_signal_analyzer.py. */
#ifndef RENDER_SIGNAL_ANALYZER_H
#define RENDER_SIGNAL_ANALYZER_H

#include <stdint.h>

typedef struct {
    int      n_freqs;
    int      n_freqs_by2;
    int      n_counters;
    int64_t *counters;
    int      strong_peak_freeze_duration;
    int      narrow_peak_band;
    int      narrow_peak_counter;
} RenderSignalAnalyzer;

/* counters_storage holds at least n_freqs - 2 int64 elements.
 * strong_peak_freeze_duration is in hops (default 6). */
void rsa_init(RenderSignalAnalyzer *m, int64_t *counters_storage, int n_freqs,
              int strong_peak_freeze_duration);
void rsa_reset(RenderSignalAnalyzer *m);

/* Per-hop update. render_psd (length n_freqs) may be NULL (counters reset); render_block may be
 * NULL (strong-peak detection skipped). */
void rsa_update(RenderSignalAnalyzer *m, const float *render_psd,
                const float *render_block, int render_block_len);

/* True iff any narrow-band counter > 10. */
int  rsa_poor_signal_excitation(const RenderSignalAnalyzer *m);

/* In-place mask: zero mu (float32, length n_freqs) for +-2 bins around any peak with counter > 5.
 * */
void rsa_mask_regions_around_narrow_bands(const RenderSignalAnalyzer *m, float *mu);

/* Returns 1 iff rsa_mask_regions_around_narrow_bands would set at least one entry of mu to 0. */
int rsa_any_narrow_band_active(const RenderSignalAnalyzer *m);

#endif
