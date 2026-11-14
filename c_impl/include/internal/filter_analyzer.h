/* Single-channel filter analyzer, float32. Python reference: modules/state/filter_analyzer.py.
 * fa_update(filter_taps, render_block) cycles the peak-finding region through the filter, one hop
 * of taps per call, on a 3-tap minimum-phase ~600 Hz high-pass copy of the taps, and maintains:
 *  - fa_min_filter_delay_blocks(): peak_index / hop_size.
 *  - fa_max_echo_path_gain(): |h[peak]| after the 5 s convergence period.
 * A ConsistentFilterDetector checks that the peak stands out from the tap floor and persists.
 * All time constants are computed in fa_init() from hop_size/sample_rate. */
#ifndef FILTER_ANALYZER_H
#define FILTER_ANALYZER_H

#include <stddef.h>

/* On-demand tap materializer (see fa_set_taps_provider), called from fa_update() after the
 * region advances and before the taps are read, with the inclusive [first, last] index span. */
typedef void (*FaTapsProvider)(void *ctx, int first, int last);

typedef struct {
    float  active_render_threshold;
    int    significant_peak;
    float  floor_accum;
    float  secondary_peak;
    int    floor_low_limit;
    int    floor_high_limit;
    long   counter;
    int    delay_ref;                /* starts at -10 */
    int    consistent_hold_hops;
} FaConsistentDetector;

typedef struct {
    int    sample_rate;              /* scales the 48 kHz IR exclusion window */
    int    bounded_erl;
    float  default_gain;
    FaConsistentDetector consistent;

    int    region_start;
    int    region_end;
    long   blocks_since_reset;
    int    peak_index;
    float  gain;
    int    consistent_estimate;
    int    delay_blocks;

    int    size;
    int    hop_size;
    int    convergence_threshold_hops;
    float *h_highpass;               /* caller-owned; length `size` */

    /* fa_update() scratch, caller-owned (see fa_init). */
    float *abs_scratch;
    int    abs_scratch_len;
    float *render_sq_scratch;
    int    render_sq_scratch_len;

    FaTapsProvider taps_provider;
    void          *taps_provider_ctx;
} FilterAnalyzer;

/* h_highpass_storage holds at least `size` floats (the filter length).
 * active_render_limit, bounded_erl and default_gain default to 100.0, 0 and 1.0.
 * abs_scratch / render_sq_scratch are caller-owned scratch for fa_update; if either is too small
 * for a call, fa_update does nothing. */
void fa_init(FilterAnalyzer *m, float *h_highpass_storage, int size,
             float active_render_limit, int bounded_erl, float default_gain,
             float *abs_scratch, int abs_scratch_len,
             float *render_sq_scratch, int render_sq_scratch_len,
             int hop_size, int sample_rate);

void fa_reset(FilterAnalyzer *m);

/* Installs (fn == NULL clears) the on-demand tap materializer. Each fa_update() reads only a
 * hop-sized slice of taps plus the two preceding ones the HPF needs, so the owner can produce just
 * that span instead of the whole impulse response. Taps outside the span are left as they were.
 * Cleared by fa_init(), kept across fa_reset(). */
void fa_set_taps_provider(FilterAnalyzer *m, FaTapsProvider fn, void *ctx);

/* Per-hop update: filter_taps (float32[size]) and render_block (float32[render_block_len]).
 * Requires abs_scratch_len >= size and render_sq_scratch_len >= render_block_len, else no-op.
 * With a provider installed, filter_taps need only be valid over the span it fills. */
void fa_update(FilterAnalyzer *m, const float *filter_taps,
               const float *render_block, int render_block_len);

int    fa_min_filter_delay_blocks(const FilterAnalyzer *m);
float  fa_max_echo_path_gain(const FilterAnalyzer *m);

#endif
