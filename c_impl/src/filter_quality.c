/* filter_quality.c -- see filter_quality.h. Integer counters and flags, no float math. */
#include "filter_quality.h"
#include "aec_scale.h"

#include <limits.h>

void fq_init(FilteringQualityAnalyzer *m, int use_linear_filter,
            int hop_size, int sample_rate) {
    m->use_linear_filter = use_linear_filter ? 1 : 0;
    m->overall_usable = 0;
    m->filter_update_blocks_since_reset = 0;
    m->filter_update_blocks_since_start = 0;
    m->convergence_seen = 0;
    m->convergence_hops_counter = 0;
    /* ~400/200 ms, computed from the live grid. */
    m->startup_hops = aec_ms_to_hops(400.0f, hop_size, sample_rate);
    m->reset_hops = aec_ms_to_hops(200.0f, hop_size, sample_rate);
}

void fq_reset(FilteringQualityAnalyzer *m) {
    m->overall_usable = 0;
    m->filter_update_blocks_since_reset = 0;
}

void fq_update(FilteringQualityAnalyzer *m,
               int active_render,
               int transparent_mode,
               int saturated_capture,
               int external_delay_present,
               int any_filter_converged) {
    int filter_update = active_render && !saturated_capture;
    int startup_ok, reset_ok, usable, gate3;

    /* Counters saturate at INT_MAX; the raw values are observable, so they are not capped at the
     * gate thresholds. */
    if (filter_update) {
        if (m->filter_update_blocks_since_reset < INT_MAX)
            m->filter_update_blocks_since_reset += 1;
        if (m->filter_update_blocks_since_start < INT_MAX)
            m->filter_update_blocks_since_start += 1;
    }
    if (any_filter_converged) {
        m->convergence_seen = 1;
        if (m->convergence_hops_counter < INT_MAX)
            m->convergence_hops_counter += 1;
    }

    startup_ok = m->filter_update_blocks_since_start > m->startup_hops;
    reset_ok = startup_ok &&
               (m->filter_update_blocks_since_reset > m->reset_hops);
    usable = startup_ok && reset_ok;
    /* Gate 3: convergence ever seen or an external delay present. */
    gate3 = external_delay_present || m->convergence_seen;
    usable = usable && gate3 && !transparent_mode;
    m->overall_usable = usable ? 1 : 0;
}

int fq_linear_filter_usable(const FilteringQualityAnalyzer *m) {
    return (m->overall_usable && m->use_linear_filter) ? 1 : 0;
}

