/* filter_delay.h — C port of python/modules/state/filter_delay.py
 * (AecState::FilterDelay; mirrors AEC3 aec_state.cc:355-388 with hop-rescaled
 * threshold).
 *
 * Caches the external (RenderDelayController) delay plus the per-channel
 * direct-path filter delays reported by the filter analyzer, in OUR hop units.
 *
 * This module is PURELY INTEGER (no floating point at all): all quantities are
 * block/hop counts and array indices. There are no numpy-dtype parity concerns
 * here — the only correctness surface is exact branch selection and the
 * integer floor-division `delay_headroom_samples // HOP_SAMPLES`.
 *
 * Behaviour (verbatim from the Python):
 *   ctor:  delay_headroom_blocks = delay_headroom_samples // HOP_SAMPLES
 *          filter_delays_blocks[c] = delay_headroom_blocks (all channels)
 *          min_filter_delay        = delay_headroom_blocks
 *          external_delay          = <unreported>
 *   update(analyzer, external, blocks_with_proper_filter_adaptation):
 *     if external is reported:            external_delay = external   (cached)
 *     unconverged = blocks_with_proper_filter_adaptation
 *                   < FILTER_ADAPTATION_THRESHOLD_HOPS (200)
 *     if unconverged AND external_delay reported:
 *         filter_delays_blocks[*] = delay_headroom_blocks
 *     elif analyzer provided:
 *         require len(analyzer) == num_channels  (else error)
 *         filter_delays_blocks    = analyzer
 *     (else: untouched)
 *     min_filter_delay = min(filter_delays_blocks)
 *
 * The cached external_delay PERSISTS across update calls — it is only replaced
 * when a *reported* external delay is supplied; a subsequent unreported
 * external still drives the unconverged-reset branch because the cache stays
 * set. Verified against the Python on the real balanced/DT pipeline.
 */
#ifndef FILTER_DELAY_H
#define FILTER_DELAY_H

/* delay_headroom_blocks and filter_adaptation_threshold_hops were frozen at
 * a hop=160/sr=16000 assumption (FILTER_DELAY_HOP_SAMPLES=160,
 * FILTER_ADAPTATION_THRESHOLD_HOPS=200); both are now computed live in
 * filter_delay_init() from hop_size/sample_rate -- see the struct fields. */

/* DelayEstimate POD (python modules/delay/delay_types.DelayEstimate).
 * `quality` is DelayQuality (0=COARSE, 1=REFINED); `delay` is in samples.
 * `reported` replaces Python's Optional[...] None sentinel (0 == None). */
typedef struct {
    int reported;  /* 0 == None (unreported), 1 == a DelayEstimate is present */
    int quality;   /* DelayQuality enum value (0=COARSE, 1=REFINED)           */
    int delay;     /* delay in samples                                        */
} FilterDelayEstimate;

typedef struct {
    int delay_headroom_blocks;   /* delay_headroom_samples / hop_size, live   */
    int num_channels;            /* len(filter_delays_blocks)                 */
    int *filter_delays_blocks;   /* caller-owned storage; length num_channels */
    int min_filter_delay;        /* min(filter_delays_blocks)                 */
    FilterDelayEstimate external_delay;  /* cached external delay (.reported)  */
    int filter_adaptation_threshold_hops; /* live-computed, was frozen at 200 */
} FilterDelay;

/* delays_storage must point to num_capture_channels ints; ownership stays with
 * the caller. hop_size/sample_rate drive the live delay_headroom_blocks and
 * filter_adaptation_threshold_hops computation. Mirrors FilterDelay.__init__. */
void filter_delay_init(FilterDelay *fd, int *delays_storage,
                       int delay_headroom_samples, int num_capture_channels,
                       int hop_size, int sample_rate);

/* update (aec_state.cc:355-388 line port).
 *   analyzer            : per-channel direct-path delays, length analyzer_len.
 *                         Pass analyzer == NULL to signal Python's None
 *                         (leave the cached estimate untouched in the else
 *                         branch). analyzer_len is ignored when analyzer==NULL.
 *   external            : external delay; pass external->reported == 0 (or
 *                         external == NULL) to signal None.
 *   blocks_with_proper_filter_adaptation : adaptation counter (hops).
 * Returns 0 on success, -1 on the analyzer length mismatch (Python raises
 * ValueError("analyzer delays length mismatch")). On the -1 path the state is
 * left unmodified up to the point of the check (matching Python, where the
 * raise aborts before the assignment and before min recompute). */
int filter_delay_update(FilterDelay *fd, const int *analyzer, int analyzer_len,
                        const FilterDelayEstimate *external,
                        int blocks_with_proper_filter_adaptation);

/* Accessors (mirror the Python methods). */
int filter_delay_external_reported(const FilterDelay *fd);
int filter_delay_min_direct_path(const FilterDelay *fd);

#endif /* FILTER_DELAY_H */
