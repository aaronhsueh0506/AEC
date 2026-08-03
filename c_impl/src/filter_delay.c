/* filter_delay.c — C port of python/modules/state/filter_delay.py.
 * Pure integer state machine (no float math). See the header for the parity
 * contract. Built with -ffp-contract=off for consistency with the rest of the
 * port (no float ops here, so it is a no-op for this TU). */
#include "filter_delay.h"
#include "aec3_scale.h"

#include <stddef.h>  /* NULL */

void filter_delay_init(FilterDelay *fd, int *delays_storage,
                       int delay_headroom_samples, int num_capture_channels,
                       int hop_size, int sample_rate) {
    int c;

    /* delay_headroom_blocks = int(delay_headroom_samples) // hop_size.
     * Python `//` is floor division; for the non-negative headroom (always
     * >= 0 in practice) C's `/` on non-negative ints is identical. Clamp to
     * floor semantics defensively in case of a negative input. Was
     * `/ FILTER_DELAY_HOP_SAMPLES` (frozen at 160); uses the live hop_size
     * now (numerically inert either way while delay_headroom_samples stays
     * well under any real hop_size, but no longer silently wrong if it
     * ever isn't). */
    int blocks = delay_headroom_samples / hop_size;
    if (delay_headroom_samples < 0 &&
        blocks * hop_size != delay_headroom_samples) {
        blocks -= 1;  /* floor toward -inf, matching Python `//` */
    }

    fd->delay_headroom_blocks = blocks;
    /* AEC3 2 s (= 500 blocks) filter-adaptation threshold. Was a frozen
     * #define (200, correct only at hop=160/sr=16000); computed live here. */
    fd->filter_adaptation_threshold_hops =
        aec3_ms_to_hops(2000.0f, hop_size, sample_rate);
    fd->num_channels = num_capture_channels;
    fd->filter_delays_blocks = delays_storage;
    for (c = 0; c < num_capture_channels; ++c) {
        fd->filter_delays_blocks[c] = blocks;
    }
    fd->min_filter_delay = blocks;

    fd->external_delay.reported = 0;
    fd->external_delay.quality = 0;
    fd->external_delay.delay = 0;
}

int filter_delay_update(FilterDelay *fd, const int *analyzer, int analyzer_len,
                        const FilterDelayEstimate *external,
                        int blocks_with_proper_filter_adaptation) {
    int delay_estimator_unconverged;
    int c, mn;

    /* if external_delay is not None: self._external_delay = external_delay */
    if (external != NULL && external->reported) {
        fd->external_delay = *external;
    }

    delay_estimator_unconverged =
        blocks_with_proper_filter_adaptation < fd->filter_adaptation_threshold_hops;

    if (delay_estimator_unconverged && fd->external_delay.reported) {
        /* filter_delays_blocks[*] = delay_headroom_blocks */
        for (c = 0; c < fd->num_channels; ++c) {
            fd->filter_delays_blocks[c] = fd->delay_headroom_blocks;
        }
    } else if (analyzer != NULL) {
        /* len(analyzer) must equal num_channels, else Python raises ValueError.
         * The Python raise aborts BEFORE the assignment and the min recompute,
         * so we return -1 with state unmodified. */
        if (analyzer_len != fd->num_channels) {
            return -1;
        }
        for (c = 0; c < fd->num_channels; ++c) {
            fd->filter_delays_blocks[c] = analyzer[c];
        }
    }
    /* else: filter_delays_blocks untouched */

    /* min_filter_delay = min(filter_delays_blocks) */
    if (fd->num_channels > 0) {
        mn = fd->filter_delays_blocks[0];
        for (c = 1; c < fd->num_channels; ++c) {
            if (fd->filter_delays_blocks[c] < mn) {
                mn = fd->filter_delays_blocks[c];
            }
        }
        fd->min_filter_delay = mn;
    }

    return 0;
}

int filter_delay_external_reported(const FilterDelay *fd) {
    return fd->external_delay.reported ? 1 : 0;
}

int filter_delay_min_direct_path(const FilterDelay *fd) {
    return fd->min_filter_delay;
}
