/* filter_analyzer.h — C port (float32-by-design; parity retired) of
 * python/modules/state/filter_analyzer.py (single-channel AEC3 FilterAnalyzer,
 * mirrors AEC3 filter_analyzer.{cc,h}).
 *
 * Per-hop ``fa_update(filter_taps, render_block)`` produces:
 *   - fa_min_filter_delay_blocks() -> peak_index // HOP_SAMPLES
 *   - fa_any_filter_consistent()   -> ConsistentFilterDetector verdict
 *   - fa_max_echo_path_gain()      -> |h[peak]| (post 5 s convergence)
 *   - fa_get_adjusted_filters()    -> the high-pass-filtered TD taps
 *
 * The peak-finding region cycles incrementally over the full filter length,
 * processing HOP_SAMPLES samples per call. The HPF is the AEC3 verbatim 3-tap
 * minimum-phase ~600 Hz cutoff filter.
 *
 * FLOAT32-BY-DESIGN notes (originally captured from numpy 1.26 -> C parity on
 * a real balanced/DT case wav/aec_challenge_blind/doubletalk/0I0XMl3M...
 * _doubletalk_*.wav, preset=balanced fl=832 hop=160; the module now runs
 * entirely in float32 — the former fp64 accumulation paths below were
 * converted for uniformity as part of the f32 campaign, drift accepted):
 *   - filter_taps : float32, size 960 (6 partitions * 160).
 *   - render_block: float32, size 160 (= render_block_scaled, int16 amplitude).
 *   - _h_highpass : float32. HPF taps (float32) * filter_taps (float32) all f32.
 *   - _gain : float32; _peak_index / _delay_blocks : int.
 *   - ConsistentFilterDetector floor accumulators:
 *       * np.abs(filter_taps_slice).sum()  -> float32 PAIRWISE sum
 *         (f32_pairwise_sum), accumulated directly into _floor_accum (float32).
 *       * seg.max(initial=0.0)             -> float32 max, stored directly.
 *       * render active power
 *         (render_block**2).sum() -> float32 PAIRWISE sum
 *         (f32_pairwise_sum, formerly f64_pairwise_sum) compared >
 *         _active_render_threshold (float32).
 *   - peak find: h[k]*h[k] in float32; np.argmax returns FIRST max index.
 *
 * Build with -ffp-contract=off so the float32 multiply/add/compare is not fused.
 */
#ifndef FILTER_ANALYZER_H
#define FILTER_ANALYZER_H

#include <stddef.h>

/* hop_size/sample_rate were previously not parameters at all -- every
 * time-based constant below (active_render_threshold's block-length factor,
 * the region-cycling block size, the delay_blocks conversion, the 5s
 * convergence threshold, the 1.5s consistency hold) was hardcoded against
 * the stale hop=160/sr=16000 assumption. All are now computed live in
 * fa_init() from hop_size/sample_rate -- see the struct fields below. */

/* On-demand tap materializer (see fa_set_taps_provider). Called from inside
 * fa_update(), after the region has advanced and before the taps are read,
 * with the inclusive [first, last] filter_taps index span that call will
 * touch. */
typedef void (*FaTapsProvider)(void *ctx, int first, int last);

/* ConsistentFilterDetector mirror state (reset each full-filter sweep). */
typedef struct {
    float  active_render_threshold;  /* (active_render_limit^2) * hop_size      */
    int    significant_peak;         /* bool                                    */
    float  floor_accum;              /* float32                                 */
    float  secondary_peak;           /* float32 (holds the f32 max)             */
    int    floor_low_limit;
    int    floor_high_limit;
    long   counter;
    int    delay_ref;                /* starts at -10                           */
    int    consistent_hold_hops;     /* live-computed, was FA_CONSISTENT_HOLD_HOPS=150 */
} FaConsistentDetector;

typedef struct {
    float  active_render_threshold;  /* (active_render_limit^2) * hop_size      */
    int    bounded_erl;              /* bool                                    */
    float  default_gain;
    FaConsistentDetector consistent;

    int    region_start;
    int    region_end;
    long   blocks_since_reset;
    int    peak_index;
    float  gain;
    int    consistent_estimate;      /* bool                                    */
    int    delay_blocks;

    int    size;                     /* filter length (== filter_taps size)     */
    int    hop_size;                 /* live grid, was FA_HOP_SAMPLES=160       */
    int    convergence_threshold_hops; /* live-computed, was FA_CONVERGENCE_THRESHOLD_HOPS=500 */
    float *h_highpass;               /* owned by caller; length `size`, f32     */

    /* De-stacked fa_update() scratch (formerly `float abs_scratch[1024]` /
     * `float render_sq_scratch[1024]` fixed-size locals -- a stack-overflow
     * hazard once the runtime filter length can exceed 1024, e.g. a
     * higher-sample-rate build). Both owned by caller; see fa_init. */
    float *abs_scratch;              /* length >= size (ConsistentFilterDetector abs-slice sums) */
    int    abs_scratch_len;
    float *render_sq_scratch;        /* length >= render_block_len at every fa_update call */
    int    render_sq_scratch_len;

    FaTapsProvider taps_provider;    /* NULL == caller pre-fills filter_taps  */
    void          *taps_provider_ctx;
} FilterAnalyzer;

/* h_highpass_storage must hold at least `size` floats (filter_taps length).
 * active_render_limit / bounded_erl / default_gain match the Python defaults
 * (100.0 / 0 / 1.0) unless overridden. `size` is the full filter length.
 *
 * abs_scratch/abs_scratch_len and render_sq_scratch/render_sq_scratch_len are
 * caller-owned scratch buffers used internally by fa_update (the abs-slice
 * ConsistentFilterDetector sums and the render-block-squared sum,
 * respectively) -- see the fa_update doc comment below for the exact size
 * each call requires; fa_update early-returns (no-op) if either buffer is
 * too small for the current call, so undersizing is safe but silently
 * disables the analyzer rather than overrunning caller memory. */
void fa_init(FilterAnalyzer *m, float *h_highpass_storage, int size,
             float active_render_limit, int bounded_erl, float default_gain,
             float *abs_scratch, int abs_scratch_len,
             float *render_sq_scratch, int render_sq_scratch_len,
             int hop_size, int sample_rate);

void fa_reset(FilterAnalyzer *m);

/* Install (or clear, with fn == NULL) the on-demand tap materializer.
 *
 * The region cycles one hop-sized block per fa_update() call, so only a
 * hop-sized slice of the impulse response -- plus the two taps preceding it
 * that the 3-tap HPF needs -- is consumed per hop; the rest of the tap array
 * is never looked at. Producing the whole impulse response every hop
 * therefore pays for an inverse FFT per partition to read back one
 * partition's worth of taps. With a provider installed, fa_update() asks for
 * exactly the span it is about to read (after the region has advanced, so the
 * request is the real span and not a prediction) and the owner materializes
 * only that. Taps outside the requested span are left as they were; nothing
 * in this module reads them.
 *
 * Cleared by fa_init(), preserved across fa_reset(): re-install after any
 * re-init of the owning object. */
void fa_set_taps_provider(FilterAnalyzer *m, FaTapsProvider fn, void *ctx);

/* Per-hop update. filter_taps (float32, length == size) and render_block
 * (float32, length render_block_len). Requires m->abs_scratch_len >= size
 * and m->render_sq_scratch_len >= render_block_len (both set at fa_init);
 * no-ops (state unchanged) if either scratch buffer is undersized.
 *
 * With a provider installed, filter_taps need only be VALID over the span the
 * provider fills for this call; without one it must be fully materialized. */
void fa_update(FilterAnalyzer *m, const float *filter_taps,
               const float *render_block, int render_block_len);

/* Public queries. */
int    fa_min_filter_delay_blocks(const FilterAnalyzer *m);
int    fa_any_filter_consistent(const FilterAnalyzer *m);
float  fa_max_echo_path_gain(const FilterAnalyzer *m);
int    fa_peak_index(const FilterAnalyzer *m);
const float *fa_get_adjusted_filters(const FilterAnalyzer *m);

/* float32 pairwise sum (numpy-1.26-shaped reduction tree; exposed for the
 * parity test). The former fa_f64_pairwise_sum twin (a float64 accumulator
 * variant for the render active-power sum) was deleted after the f32
 * campaign made it identical to this one. */
float  fa_f32_pairwise_sum(const float *a, size_t n);

#endif /* FILTER_ANALYZER_H */
