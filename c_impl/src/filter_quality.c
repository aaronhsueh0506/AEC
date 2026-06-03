/* filter_quality.c — byte-equal C port of
 * python/modules/state/filter_quality.py. See header for parity notes.
 *
 * Pure integer counters + bool gates; no float math. Mirrors AEC3
 * aec_state.cc:390-444 with hop-rescaled thresholds (FQ_STARTUP_HOPS=40,
 * FQ_RESET_HOPS=20).
 */
#include "filter_quality.h"

void fq_init(FilteringQualityAnalyzer *m, int use_linear_filter) {
    m->use_linear_filter = use_linear_filter ? 1 : 0;
    m->overall_usable = 0;
    m->filter_update_blocks_since_reset = 0;
    m->filter_update_blocks_since_start = 0;
    m->convergence_seen = 0;
    m->convergence_hops_counter = 0;
}

void fq_reset(FilteringQualityAnalyzer *m) {
    m->overall_usable = 0;
    m->filter_update_blocks_since_reset = 0;
    /* NOTE: filter_update_blocks_since_start is NOT reset (matches AEC3).  */
    /* NOTE: convergence_seen latches, NOT reset (matches AEC3).            */
}

void fq_update(FilteringQualityAnalyzer *m,
               int active_render,
               int transparent_mode,
               int saturated_capture,
               int external_delay_present,
               int any_filter_converged) {
    int filter_update = active_render && !saturated_capture;
    int startup_ok, reset_ok, usable, gate3;

    if (filter_update) {
        m->filter_update_blocks_since_reset += 1;
        m->filter_update_blocks_since_start += 1;
    }
    if (any_filter_converged) {
        m->convergence_seen = 1;
        m->convergence_hops_counter += 1;
    }

    startup_ok = m->filter_update_blocks_since_start > FQ_STARTUP_HOPS;
    reset_ok = startup_ok &&
               (m->filter_update_blocks_since_reset > FQ_RESET_HOPS);
    usable = startup_ok && reset_ok;
    /* Gate 3: convergence ever seen OR external delay present (AEC3 default). */
    gate3 = external_delay_present || m->convergence_seen;
    usable = usable && gate3 && !transparent_mode;
    m->overall_usable = usable ? 1 : 0;
}

int fq_linear_filter_usable(const FilteringQualityAnalyzer *m) {
    return (m->overall_usable && m->use_linear_filter) ? 1 : 0;
}

int fq_startup_blocks(const FilteringQualityAnalyzer *m) {
    return m->filter_update_blocks_since_start;
}

int fq_reset_blocks(const FilteringQualityAnalyzer *m) {
    return m->filter_update_blocks_since_reset;
}

int fq_gate1_pass(const FilteringQualityAnalyzer *m) {
    return (m->filter_update_blocks_since_start > FQ_STARTUP_HOPS) ? 1 : 0;
}

int fq_gate2_pass(const FilteringQualityAnalyzer *m) {
    return ((m->filter_update_blocks_since_start > FQ_STARTUP_HOPS) &&
            (m->filter_update_blocks_since_reset > FQ_RESET_HOPS)) ? 1 : 0;
}
