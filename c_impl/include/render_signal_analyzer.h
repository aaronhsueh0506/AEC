/* render_signal_analyzer.h — C port of
 * python/modules/render/render_signal_analyzer.py (which mirrors AEC3
 * render_signal_analyzer.{cc,h}).
 *
 * Per-hop render-signal analyser:
 *   1. per-bin tonal narrowband counters → mask_regions_around_narrow_bands()
 *      zeroes mu for ±2 bins around any sustained narrow peak (counter > 5).
 *   2. poor_signal_excitation(): any narrow-band counter > 10 → freeze W.
 *   3. narrow_peak_band(): dominant strong narrow peak (diagnostic).
 *
 * BYTE-EQUAL parity notes (numpy 1.26 → C):
 *   - render_psd (|X|²) and render_block (time samples) arrive as float32 from
 *     the real pipeline; the small-narrow-band peak test runs ALL-float32
 *     (numpy value-based promotion keeps `3.0f * f32_array` in float32):
 *         is_peak = x_center > 3.0f * fmaxf(x_left, x_right)
 *   - the strong-narrow-band test widens float32 window maxima to f64 via
 *     Python float(); those comparisons run in double.
 *   - counters are int64 (length n_freqs - 2).
 *   - np.argmax returns the FIRST max index (strict-greater scan).
 *
 * Build with -ffp-contract=off so the float32 multiply/compare is not fused.
 */
#ifndef RENDER_SIGNAL_ANALYZER_H
#define RENDER_SIGNAL_ANALYZER_H

#include <stdint.h>

typedef struct {
    int      n_freqs;        /* spectrum length = fft_size/2 + 1            */
    int      n_freqs_by2;    /* n_freqs - 1 (AEC3 kFftLengthBy2)            */
    int      n_counters;     /* n_freqs_by2 - 1 = n_freqs - 2               */
    int64_t *counters;       /* owned by caller; length n_counters          */
    int      strong_peak_freeze_duration;
    int      narrow_peak_band;    /* -1 == None                             */
    int      narrow_peak_counter;
} RenderSignalAnalyzer;

/* counters_storage must have at least (n_freqs - 2) int64 elements.
 * strong_peak_freeze_duration matches the Python default (6) unless overridden. */
void rsa_init(RenderSignalAnalyzer *m, int64_t *counters_storage, int n_freqs,
              int strong_peak_freeze_duration);
void rsa_reset(RenderSignalAnalyzer *m);

/* Per-hop update. render_psd may be NULL (→ counters reset); render_block may
 * be NULL (→ skip strong-peak detection). Both float32 (length n_freqs and the
 * hop length respectively). */
void rsa_update(RenderSignalAnalyzer *m, const float *render_psd,
                const float *render_block, int render_block_len);

/* True iff any narrow-band counter > 10. */
int  rsa_poor_signal_excitation(const RenderSignalAnalyzer *m);

/* -1 == None. */
int  rsa_narrow_peak_band(const RenderSignalAnalyzer *m);

/* In-place mask: zero mu (float32, length n_freqs) for ±2 bins around any peak
 * with counter > 5. */
void rsa_mask_regions_around_narrow_bands(const RenderSignalAnalyzer *m, float *mu);

#endif /* RENDER_SIGNAL_ANALYZER_H */
