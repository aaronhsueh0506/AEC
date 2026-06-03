/* filter_quality.h — byte-equal C port of
 * python/modules/state/filter_quality.py
 * (AEC3 AecState::FilteringQualityAnalyzer, the 4-gate AND driving
 *  UsableLinearEstimate / linear_filter_usable()).
 *
 * PURE INTEGER COUNTERS + bool gates — no float math at all, so "byte-equal"
 * here means exact int/bool agreement.
 *
 * The 4 gates (all must pass for the linear estimate to be usable):
 *   1. startup_ok : filter_update_blocks_since_start > _STARTUP_HOPS (=40)
 *   2. reset_ok   : startup_ok && filter_update_blocks_since_reset > _RESET_HOPS (=20)
 *   3. gate3      : external_delay present || convergence_seen (latches)
 *   4. NOT transparent_mode
 *
 * Threshold derivation (mirrors state/_constants.py + filter_quality.py):
 *   HOPS_PER_SECOND = 100
 *   _STARTUP_HOPS = int(0.4 * 100) = 40
 *   _RESET_HOPS   = int(0.2 * 100) = 20
 *
 * Reset semantics (must match AEC3 exactly):
 *   - filter_update_blocks_since_start is NOT reset on reset().
 *   - convergence_seen LATCHES (never reset).
 *   - convergence_hops_counter is a diagnostic-only counter (no gate consumes
 *     it); it also latches (not reset).
 *
 * external_delay (Optional[DelayEstimate]) is consumed by presence ONLY
 * (None vs not-None), so the C API takes a plain int flag external_delay_present.
 * filter_analyzer_consistent is accepted as a no-op in the Python source and is
 * therefore not part of the C update signature.
 */
#ifndef FILTER_QUALITY_H
#define FILTER_QUALITY_H

/* AEC3 0.4*250=100 blocks (~400 ms) -> our 40 hops */
#define FQ_STARTUP_HOPS 40
/* AEC3 0.2*250=50 blocks (~200 ms) -> our 20 hops */
#define FQ_RESET_HOPS   20

typedef struct {
    int use_linear_filter;                    /* config (bool) */
    int overall_usable;                       /* bool */
    int filter_update_blocks_since_reset;
    int filter_update_blocks_since_start;
    int convergence_seen;                     /* bool, latches */
    int convergence_hops_counter;             /* diagnostic only */
} FilteringQualityAnalyzer;

/* use_linear_filter mirrors the Python ctor kwarg (default True -> pass 1). */
void fq_init(FilteringQualityAnalyzer *m, int use_linear_filter);

/* reset(): clears overall_usable + blocks_since_reset. blocks_since_start and
 * convergence_seen are deliberately NOT reset (AEC3 parity). */
void fq_reset(FilteringQualityAnalyzer *m);

/* Per-hop update.
 *   active_render, transparent_mode, saturated_capture : bool (0/1)
 *   external_delay_present : 1 iff Optional[DelayEstimate] is not None
 *   any_filter_converged   : bool (0/1)
 */
void fq_update(FilteringQualityAnalyzer *m,
               int active_render,
               int transparent_mode,
               int saturated_capture,
               int external_delay_present,
               int any_filter_converged);

/* overall_usable && use_linear_filter */
int fq_linear_filter_usable(const FilteringQualityAnalyzer *m);

/* diagnostic accessors (trace_hf_chain reads; no audio path) */
int fq_startup_blocks(const FilteringQualityAnalyzer *m);
int fq_reset_blocks(const FilteringQualityAnalyzer *m);
int fq_gate1_pass(const FilteringQualityAnalyzer *m);
int fq_gate2_pass(const FilteringQualityAnalyzer *m);

#endif /* FILTER_QUALITY_H */
