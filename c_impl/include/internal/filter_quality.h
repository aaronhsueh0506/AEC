/* Linear-filter usability gate (integer counters and flags, no float math).
 * The estimate is usable when all hold: more than startup_hops filter-update blocks since start,
 * more than reset_hops since the last reset, an external delay present or a convergence seen
 * (latched), and not transparent_mode.
 * reset() clears overall_usable and blocks_since_reset only; blocks_since_start and
 * convergence_seen persist. Python reference: modules/state/filter_quality.py. */
#ifndef FILTER_QUALITY_H
#define FILTER_QUALITY_H

typedef struct {
    int use_linear_filter;
    int overall_usable;
    int filter_update_blocks_since_reset;
    int filter_update_blocks_since_start;
    int convergence_seen;
    int convergence_hops_counter;
    int startup_hops;   /* ~400 ms in hops */
    int reset_hops;     /* ~200 ms in hops */
} FilteringQualityAnalyzer;

/* use_linear_filter is 0/1. hop_size/sample_rate set startup_hops/reset_hops (~400/200 ms). */
void fq_init(FilteringQualityAnalyzer *m, int use_linear_filter,
            int hop_size, int sample_rate);

/* Clears overall_usable and blocks_since_reset; the other counters persist. */
void fq_reset(FilteringQualityAnalyzer *m);

/* Per-hop update; all arguments are 0/1 flags.
 * external_delay_present is 1 when an external delay estimate exists. */
void fq_update(FilteringQualityAnalyzer *m,
               int active_render,
               int transparent_mode,
               int saturated_capture,
               int external_delay_present,
               int any_filter_converged);

int fq_linear_filter_usable(const FilteringQualityAnalyzer *m);

#endif
