/* filter_delay.c -- see filter_delay.h. Integer state machine. */
#include "filter_delay.h"
#include "aec_scale.h"

#include <stddef.h>

void filter_delay_init(FilterDelay *fd, int *delays_storage,
                       int delay_headroom_samples, int num_capture_channels,
                       int hop_size, int sample_rate) {
    int c;

    /* Headroom in blocks, floored toward -inf for a negative input. */
    int blocks = delay_headroom_samples / hop_size;
    if (delay_headroom_samples < 0 &&
        blocks * hop_size != delay_headroom_samples) {
        blocks -= 1;
    }

    fd->delay_headroom_blocks = blocks;
    /* 2 s filter-adaptation threshold, computed from the live grid. */
    fd->filter_adaptation_threshold_hops =
        aec_ms_to_hops(2000.0f, hop_size, sample_rate);
    fd->num_channels = num_capture_channels;
    fd->filter_delays_blocks = delays_storage;
    for (c = 0; c < num_capture_channels; ++c) {
        fd->filter_delays_blocks[c] = blocks;
    }
    fd->min_filter_delay = blocks;

    fd->external_delay.reported = 0;
}

void filter_delay_update(FilterDelay *fd, const int *analyzer, int analyzer_len,
                         const FilterDelayEstimate *external,
                         int blocks_with_proper_filter_adaptation) {
    int delay_estimator_unconverged;
    int c, mn;

    /* Cache a reported external delay. */
    if (external != NULL && external->reported) {
        fd->external_delay = *external;
    }

    delay_estimator_unconverged =
        blocks_with_proper_filter_adaptation < fd->filter_adaptation_threshold_hops;

    if (delay_estimator_unconverged && fd->external_delay.reported) {
        /* Unconverged with a cached external delay: reset all channels to the headroom. */
        for (c = 0; c < fd->num_channels; ++c) {
            fd->filter_delays_blocks[c] = fd->delay_headroom_blocks;
        }
    } else if (analyzer != NULL) {
        /* The analyzer length must equal num_channels; state stays unmodified on mismatch. */
        if (analyzer_len != fd->num_channels) {
            return;
        }
        for (c = 0; c < fd->num_channels; ++c) {
            fd->filter_delays_blocks[c] = analyzer[c];
        }
    }

    /* min over channels */
    if (fd->num_channels > 0) {
        mn = fd->filter_delays_blocks[0];
        for (c = 1; c < fd->num_channels; ++c) {
            if (fd->filter_delays_blocks[c] < mn) {
                mn = fd->filter_delays_blocks[c];
            }
        }
        fd->min_filter_delay = mn;
    }
}

int filter_delay_min_direct_path(const FilterDelay *fd) {
    return fd->min_filter_delay;
}
