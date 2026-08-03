/* filter_quality.c — byte-equal C port of
 * python/modules/state/filter_quality.py. See header for parity notes.
 *
 * Pure integer counters + bool gates; no float math. Mirrors AEC3
 * aec_state.cc:390-444 with hop-rescaled thresholds (startup_hops/
 * reset_hops, computed live in fq_init() from hop_size/sample_rate).
 */
#include "filter_quality.h"
#include "aec3_scale.h"

#include <limits.h>   /* INT_MAX */

void fq_init(FilteringQualityAnalyzer *m, int use_linear_filter,
            int hop_size, int sample_rate) {
    m->use_linear_filter = use_linear_filter ? 1 : 0;
    m->overall_usable = 0;
    m->filter_update_blocks_since_reset = 0;
    m->filter_update_blocks_since_start = 0;
    m->convergence_seen = 0;
    m->convergence_hops_counter = 0;
    /* AEC3 ~100/50 blocks (~400/200 ms). Were frozen #defines (40/20,
     * correct only at hop=160/sr=16000); computed live here. */
    m->startup_hops = aec3_ms_to_hops(400.0f, hop_size, sample_rate);
    m->reset_hops = aec3_ms_to_hops(200.0f, hop_size, sample_rate);
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

    /* All three counters below: UBSan-confirmed signed-overflow fix,
     * floored at INT_MAX rather than at either gate threshold
     * (FQ_STARTUP_HOPS / FQ_RESET_HOPS). Unlike a pure boolean-gate
     * counter, all three raw values are read bit-exact by
     * test/parity_filter_quality.c (`got[3..4]` via fq_startup_blocks/
     * fq_reset_blocks, `got[6]` via m.convergence_hops_counter directly),
     * which mirrors this module's Python reference's own unbounded
     * counters over a multi-frame sequence -- capping at the gate
     * threshold would still satisfy the `> FQ_STARTUP_HOPS`/
     * `> FQ_RESET_HOPS` booleans, but would desync the raw value from the
     * golden the very next hop (same trap as erl_estimator.c's
     * hold_counter_time_domain -- see that fix's comment).
     * convergence_hops_counter specifically is also documented in the
     * header as "diagnostic only (no gate consumes it)" -- the textbook
     * case for a saturating diagnostic counter. Capping at INT_MAX instead
     * is a no-op for every practically-reachable hop count while
     * eliminating the UB at the true overflow boundary. */
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
    return (m->filter_update_blocks_since_start > m->startup_hops) ? 1 : 0;
}

int fq_gate2_pass(const FilteringQualityAnalyzer *m) {
    return ((m->filter_update_blocks_since_start > m->startup_hops) &&
            (m->filter_update_blocks_since_reset > m->reset_hops)) ? 1 : 0;
}
