/* delay_aec3.h — C port of python/modules/delay/ (the v3.22 AEC3
 * matched-filter delay-estimation package). ⚠ The ENTIRE C delay chain
 * (decimator biquads, matched-filter dot products / NLMS update,
 * error_sum_anchor / x2_sum_threshold / pre-echo aggregator scalars, the
 * confidence getter) now runs float32 unconditionally — an intentional,
 * sampled-cost-free divergence from the Python float64 reference. This
 * chain's Python bit-exact parity is retired BY DESIGN: test/parity_delay.c
 * + test/gen_delay_c_golden.c form a C-regression golden (catches
 * accidental future changes) rather than a Python-parity gate.
 *
 * REPLACES the stale v3.10 GCC-PHAT estimator in delay_est.{c,h}. This is a
 * pure-additive module; the cutover that wires it into aec.c happens later.
 *
 * Chain (mirrors the Python package):
 *   Decimator x4 (cascaded biquad LP + HP; Python runs float64, C runs
 *   float32 -- see the divergence note above)
 *     -> DownsampledRenderBuffer (forward-stored ring)
 *       -> MatchedFilter bank (num_filters NLMS cross-correlators)
 *         -> MatchedFilterLagAggregator (250-estimate histogram +
 *            PreEchoLagAggregator)
 *           -> ClockdriftDetector
 *   + EchoPathDelayEstimator (the 64-sample inner-block edge-chunker draining
 *     160-sample hops) + LegacyDelayShim (the adapter the orchestrator uses).
 *
 * Public surface mirrors LegacyDelayShim:
 *   delay_aec3_accumulate(d, near[hop], far[hop], hop)  -- per-hop drive
 *   delay_aec3_estimated_delay(d)  -> int    (-1 until first estimate)
 *   delay_aec3_confidence(d)       -> float  (0.0 / 0.5 / 1.0)
 *   delay_aec3_is_solid(d)         -> int    (confidence >= 1.0)
 *   delay_aec3_n_updates(d)        -> int    (estimate_count)
 *
 * PARITY NOTES (numpy 1.26 -> C, -ffp-contract=off mandatory) -- historical,
 * describing the retired Python bit-exact target; the C side below is now
 * float32 throughout by construction, not double:
 *   - Decimator biquads: Python b/a/z are float64, each section direct-form-II
 *     transposed, output cast to float32 per sample then strided [::4]. The C
 *     port (DaBiquad) now runs the whole cascade in float32 instead of
 *     mirroring Python's float64 -- an intentional divergence.
 *   - The NLMS update `h += alpha * x_window` is the numpy ARRAY scalar rule:
 *     alpha (a Python double) is value-cast to float32 FIRST, then the
 *     per-tap product and add are computed in float32 (verified 0/20000 vs
 *     the f64-mul candidate which mismatched 99%). The C port now computes
 *     alpha directly as a float32 expression (no double intermediate).
 *   - `instantaneous * one_over_anchor` and `0.015 * (norm - accumulated)`
 *     are also the array scalar rule (scalar cast to f32, op in f32); the C
 *     port carries one_over_anchor as float32 end to end now.
 *   - The dot products `np.dot(h, x)` / `np.dot(x, x)` are float32 inputs.
 *     numpy routes these through OpenBLAS SDOT (a CPU-dispatched AVX kernel
 *     that is NOT portably bit-reproducible). The matched-filter's two
 *     512-tap dot products (da_dot_f32 in delay_aec3.c) accumulate in
 *     float32 directly (WebRTC NEON-matched-filter style). The smaller
 *     16-sample error_sum_anchor dot (delay_aec3_dot) also now accumulates
 *     in float32 (previously double) -- an intentional divergence (60-case
 *     AECMOS sample: all deltas 0.000). The matched-filter peak (argmax h^2)
 *     and the histogram-voted lag are robust to the sub-ULP/f32 accumulation
 *     difference in practice, but are no longer guaranteed bit-exact to
 *     Python by construction.
 *   - argmax = numpy argmax = FIRST max index on ties.
 *   - int(round())/lrint not needed here: every quantisation is integer
 *     floor (// ) or bit-shift, reproduced verbatim.
 */
#ifndef AEC_DELAY_AEC3_H
#define AEC_DELAY_AEC3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- DelayQuality (delay_types.py) ---- */
typedef enum {
    DELAY_QUALITY_COARSE  = 0,
    DELAY_QUALITY_REFINED = 1
} DelayQuality;

/* ---- ClockdriftLevel (clockdrift_detector.py) ---- */
typedef enum {
    CLOCKDRIFT_NONE     = 0,
    CLOCKDRIFT_PROBABLE = 1,
    CLOCKDRIFT_VERIFIED = 2
} ClockdriftLevel;

/* ===================== AEC3 config (balanced defaults) ====================
 * All values traced from the Python defaults the LegacyDelayShim constructs
 * (legacy_compat.py builds EchoPathDelayEstimator() with no overrides, so the
 * EchoPathDelayEstimator __init__ defaults are authoritative):
 *   num_filters=5, window_size_sub_blocks=32, alignment_shift_sub_blocks=24,
 *   excitation_limit=150/32768, smoothing_fast=0.7, smoothing_slow=0.1,
 *   matching_filter_threshold=0.3, delay_headroom_samples=32,
 *   detect_pre_echo=True, thresholds=DelaySelectionThresholds(5, 20).
 *   _DOWN_SAMPLING_FACTOR=4, _AEC3_BLOCK_SIZE=64, _SUB_BLOCK_SIZE=16,
 *   _CONSISTENT_ESTIMATE_THRESHOLD=125.
 */
/* Down-sampling factor: 4, the WebRTC-style reference decimation (the
 * bit-exact Python-reference path). An 8x option was sampled (60-case
 * stratified AECMOS) and removed — real farend-singletalk regression. */
#define DA_DOWN_SAMPLING_FACTOR 4
#define DA_AEC3_BLOCK_SIZE      64
#define DA_SUB_BLOCK_SIZE       (DA_AEC3_BLOCK_SIZE / DA_DOWN_SAMPLING_FACTOR) /* 16 */
#define DA_NUM_FILTERS          5
#define DA_WINDOW_SIZE_SB       32
#define DA_ALIGNMENT_SHIFT_SB   24
#define DA_FILTER_SIZE          (DA_WINDOW_SIZE_SB * DA_SUB_BLOCK_SIZE)        /* 512 */
#define DA_FILTER_INTRA_SHIFT   (DA_ALIGNMENT_SHIFT_SB * DA_SUB_BLOCK_SIZE)   /* 384 */
#define DA_MAX_FILTER_LAG       (DA_NUM_FILTERS * DA_FILTER_INTRA_SHIFT + DA_FILTER_SIZE) /* 2432 */
#define DA_RING_CAPACITY        ((DA_WINDOW_SIZE_SB + DA_ALIGNMENT_SHIFT_SB * (DA_NUM_FILTERS - 1) + 1) * DA_SUB_BLOCK_SIZE) /* 2064 */
#define DA_ACC_ERR_SIZE         (DA_FILTER_SIZE / 4)   /* 128 ; subsample rate 4 */
#define DA_HIST_WINDOW          250
#define DA_HP_HIST_SIZE         (DA_MAX_FILTER_LAG + 1)             /* 2433 */
#define DA_PE_HIST_SIZE         (((DA_MAX_FILTER_LAG + 1) * DA_DOWN_SAMPLING_FACTOR) >> 6) /* 152 */
#define DA_HEADROOM             (32 / DA_DOWN_SAMPLING_FACTOR)      /* 8 */
#define DA_THRESH_INITIAL       5
#define DA_THRESH_CONVERGED     20
#define DA_CONSISTENT_EST_THR   125
#define DA_PRE_ECHO_UPDATES_TO_REPORT 50
#define DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE 4
#define DA_BLOCK_SIZE_LOG2      6    /* kBlockSizeLog2 (64 = 1<<6) */
#define DA_K_MFW_SUB_BLOCKS     32   /* kMatchedFilterWindowSizeSubBlocks */
#define DA_K_NUM_BLOCKS_PER_SEC 250  /* kNumBlocksPerSecond (16000/64) */
#define DA_STABILITY_RESET_HOPS 3000 /* ms_to_hops(30000) -> 3000 */

/* ---- biquad cascade ----
 * Max sections: 3 (the ds4 LP is elliptic, 3 sections). */
#define DA_BQ_MAX_SECTIONS 3
typedef struct {
    int    n_sections;
    float  b[DA_BQ_MAX_SECTIONS][3];    /* {b0,b1,b2} per section */
    float  a[DA_BQ_MAX_SECTIONS][2];    /* {a1,a2} per section (a0 normalised) */
    float  z[DA_BQ_MAX_SECTIONS][2];    /* per-section state */
} DaBiquad;

typedef struct {
    DaBiquad anti_alias;       /* LP ds4 (3 sections) */
    DaBiquad noise_reduction;  /* HP (1 section) */
} DaDecimator;

typedef struct {
    float buffer[DA_RING_CAPACITY];
    int   write;
} DaRing;

typedef struct {
    float filters[DA_NUM_FILTERS][DA_FILTER_SIZE];
    float accumulated_error[DA_NUM_FILTERS][DA_ACC_ERR_SIZE];
    float instantaneous_error[DA_ACC_ERR_SIZE];   /* per-tap-prefix err, last filter (matches Python) */
    int   last_detected_best_lag_filter;
    int   number_pre_echo_updates;
    /* reported lag */
    int   reported_valid;
    int   reported_lag;
    int   reported_pre_echo_lag;
    /* per-update winner lag (transient) */
    int   winner_lag_valid;
    int   winner_lag;
} DaMatchedFilter;

typedef struct {
    int histogram[DA_HP_HIST_SIZE];
    int ring[DA_HIST_WINDOW];
    int ring_index;
    int candidate;
    int candidate_valid;   /* internal-only: 0 forces a da_argmax_i rescan on
                             * the next aggregate() call (set on every reset();
                             * `candidate` itself deliberately is NOT reset --
                             * see da_highest_peak_reset -- this flag exists
                             * precisely so the incremental-argmax scheme never
                             * trusts that stale value). */
} DaHighestPeak;

typedef struct {
    int histogram[DA_PE_HIST_SIZE];
    int ring[DA_HIST_WINDOW];
    int ring_index;
    int number_updates;
    int pre_echo_candidate;
    int block_size_log2;
    int argmax_idx;        /* incremental-argmax tracking (raw bin index, NOT
                             * shifted by block_size_log2) -- maintained every
                             * call regardless of which branch (windowed
                             * local-max vs steady-state) produces
                             * pre_echo_candidate, so it is already warm by
                             * the time number_updates crosses the steady-
                             * state threshold. */
    int argmax_valid;
} DaPreEcho;

typedef struct {
    DaHighestPeak highest_peak;
    DaPreEcho     pre_echo;
    int           significant_candidate_found;  /* bool */
} DaLagAggregator;

typedef struct {
    int delay_history[3];      /* newest -> oldest */
    ClockdriftLevel level;
    int stability_counter;
} DaClockdrift;

typedef struct {
    DaDecimator     capture_decimator;
    DaDecimator     render_decimator;
    DaRing          render_ring;
    DaMatchedFilter matched_filter;
    DaLagAggregator aggregator;
    DaClockdrift    clockdrift;
    /* old aggregated lag (raw samples) */
    int old_agg_valid;
    int old_agg_delay;
    DelayQuality old_agg_quality;
    int consistent_estimate_counter;
    /* outer->inner edge buffers (raw 16 kHz samples awaiting a 64-sample block) */
    float capture_pending[DA_AEC3_BLOCK_SIZE];
    float render_pending[DA_AEC3_BLOCK_SIZE];
    int   pending_count;
} DaEstimator;

/* ===================== LegacyDelayShim ===================== */
typedef struct {
    DaEstimator est;
    /* latest emitted estimate (raw 16 kHz samples) */
    int          latest_valid;
    int          latest_delay;
    DelayQuality latest_quality;
    int          estimate_count;   /* _n_updates */
} DelayAec3;

/* lifecycle */
void delay_aec3_init(DelayAec3 *d);
void delay_aec3_reset(DelayAec3 *d);

/* per-hop drive. near/far are length `hop` float arrays (near = HPF mic,
 * far = raw reference, both BEFORE ring-buffer alignment). Returns 1 iff a
 * NEW delay value was emitted this hop (mirrors shim accumulate()). */
int  delay_aec3_accumulate(DelayAec3 *d, const float *near, const float *far, int hop);

/* Duty-cycle variant. aec.c's per-hop duty-cycle state machine (always
 * active — see the duty_* fields in aec.h) drives this unconditionally. With
 * run_matched_filter=1: identical to delay_aec3_accumulate (byte-exact).
 * With 0: the hop is FED (decimators run for state continuity, the decimated
 * render sub-blocks are pushed into the ring so the matched filter never
 * sees gapped audio) but not ANALYSED (no matched-filter / aggregator /
 * clockdrift update; latest_* unchanged). Feeding-without-analysing has no
 * Python counterpart — an intentional, sampled-cost-free divergence. */
int  delay_aec3_accumulate_ex(DelayAec3 *d, const float *near, const float *far,
                              int hop, int run_matched_filter);

/* legacy accessors */
int    delay_aec3_estimated_delay(const DelayAec3 *d);   /* -1 until first estimate */
float  delay_aec3_confidence(const DelayAec3 *d);        /* 0.0 / 0.5 / 1.0 (float32, was double) */
int    delay_aec3_is_solid(const DelayAec3 *d);          /* confidence >= 1.0 */
int    delay_aec3_n_updates(const DelayAec3 *d);
int    delay_aec3_has_clockdrift(const DelayAec3 *d);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DELAY_AEC3_H */
