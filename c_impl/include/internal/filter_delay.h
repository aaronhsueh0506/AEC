/* Caches the external (render-delay-controller) delay and the per-channel direct-path filter
 * delays from the filter analyzer, in hop units (integer only). A reported external delay persists
 * across calls. While unconverged (adaptation counter below filter_adaptation_threshold_hops) with
 * an external delay cached, all channels reset to delay_headroom_blocks; otherwise the analyzer
 * delays are copied when given. min_filter_delay is the minimum over channels.
 * Python reference: modules/state/filter_delay.py. */
#ifndef FILTER_DELAY_H
#define FILTER_DELAY_H

/* DelayEstimate POD. `delay` is in samples; `reported` == 0 means no estimate. */
typedef struct {
    int reported;
    int delay;
} FilterDelayEstimate;

typedef struct {
    int delay_headroom_blocks;
    int num_channels;
    int *filter_delays_blocks;   /* caller-owned; length num_channels */
    int min_filter_delay;
    FilterDelayEstimate external_delay;
    int filter_adaptation_threshold_hops;
} FilterDelay;

/* delays_storage points to num_capture_channels ints, owned by the caller.
 * hop_size/sample_rate set delay_headroom_blocks and filter_adaptation_threshold_hops. */
void filter_delay_init(FilterDelay *fd, int *delays_storage,
                       int delay_headroom_samples, int num_capture_channels,
                       int hop_size, int sample_rate);

/* analyzer: per-channel direct-path delays of length analyzer_len, or NULL to keep the
 * current delays (analyzer_len is then ignored).
 * external: external delay; NULL or external->reported == 0 means none.
 * blocks_with_proper_filter_adaptation: adaptation counter in hops.
 * Leaves state unmodified on an analyzer length mismatch. */
void filter_delay_update(FilterDelay *fd, const int *analyzer, int analyzer_len,
                         const FilterDelayEstimate *external,
                         int blocks_with_proper_filter_adaptation);

int filter_delay_min_direct_path(const FilterDelay *fd);

#endif
