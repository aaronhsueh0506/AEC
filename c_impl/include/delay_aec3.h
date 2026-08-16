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
 *   delay_aec3_get_mem_size(sr, hop, n) -> size_t  (pool bytes for this config)
 *   delay_aec3_init(d, mem, bytes, sr, hop, n) -> 0/-1  (pool-first lifecycle)
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

/* ============================ MEMORY CONTRACT ============================
 * POOL-FIRST, CONFIG-SIZED (productization plan §3.2). Every array whose
 * length depends on the runtime bank size -- the matched-filter coefficient
 * bank, its accumulated-error rows, the down-sampled render ring, and both
 * lag histograms -- is carved from a CALLER-OWNED block by
 * delay_aec3_init(), sized by delay_aec3_get_mem_size() for that exact
 * (sample_rate, hop_size, num_filters) triple. The struct below keeps only
 * scalars, biquad state, the 64-sample edge-chunker buffers, and pointers.
 *
 * This REPLACES the previous "compute shrinks, RAM does not" contract, under
 * which every array was carved at the 5-filter bound so the footprint stayed
 * a single compile-time constant. That trade-off is withdrawn: the pool size
 * is a function of the init config (which it already was for the AEC as a
 * whole -- aec_get_mem_size(cfg)), so a short bank now saves BYTES as well as
 * MACs. n=1 saves ~23 KB against n=5 at every grid.
 *
 * Consequences the caller must respect:
 *   - There is exactly ONE canonical lifecycle: get_mem_size -> init. No
 *     construct-in-place-from-a-bare-struct entry point exists any more; a
 *     `DelayAec3` that was never handed a pool has NULL array pointers.
 *   - num_filters (like sample_rate and hop_size) is init-time IMMUTABLE.
 *     Changing it means re-querying the size and re-initialising, because the
 *     layout differs. Nothing pre-allocates the n=5 maximum "just in case".
 *   - Zero heap: neither entry point allocates.
 *   - Every carved array is ALIGN16 (mem_align.h's project-wide value), which
 *     the NEON matched-filter kernels' 512-tap loads rely on.
 *
 * DA_NUM_FILTERS below is now only the GEOMETRY UPPER BOUND used by
 * validation (and by the reach table in this header) -- it no longer sizes
 * anything. */

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
/* DA_NUM_FILTERS is the matched-filter bank's GEOMETRY UPPER BOUND: the
 * largest n aec_validate_config()/delay_aec3_get_mem_size() accept, and the
 * `n` the AEC3 reference itself uses. It sizes NOTHING -- see the MEMORY
 * CONTRACT block at the top of this header. The bank actually searched is
 * DaMatchedFilter::num_filters, fixed at init from AecConfig::
 * delay_num_filters (range [1, DA_NUM_FILTERS], default 5 = unchanged AEC3
 * geometry), and every array is carved to match it. */
#define DA_NUM_FILTERS          5
#define DA_WINDOW_SIZE_SB       32
#define DA_ALIGNMENT_SHIFT_SB   24
#define DA_FILTER_SIZE          (DA_WINDOW_SIZE_SB * DA_SUB_BLOCK_SIZE)        /* 512 */
#define DA_FILTER_INTRA_SHIFT   (DA_ALIGNMENT_SHIFT_SB * DA_SUB_BLOCK_SIZE)   /* 384 */
#define DA_ACC_ERR_SIZE         (DA_FILTER_SIZE / 4)   /* 128 ; subsample rate 4 */
/* DA_HIST_WINDOW is a TIME window (250 estimates ~= 1 s at the 64-sample
 * inner-block cadence), not a geometry term: it does NOT scale with n. Both
 * aggregator rings stay this length at every bank size. */
#define DA_HIST_WINDOW          250
#define DA_HEADROOM             (32 / DA_DOWN_SAMPLING_FACTOR)      /* 8 */

/* ---- per-bank-size geometry (the ONLY sizing formulas; used by both
 * delay_aec3_get_mem_size() and delay_aec3_init(), and reproduced by the
 * permanent size tests) ---------------------------------------------------
 *
 * DA_RING_CAPACITY_FOR(n) is the exact gather reach: the deepest sample the
 * bank ever touches is (n-1)*DA_FILTER_INTRA_SHIFT + DA_FILTER_SIZE +
 * DA_SUB_BLOCK_SIZE - 2, and this capacity is exactly that plus two.
 *
 * DA_MAX_FILTER_LAG_FOR(n) DELIBERATELY CARRIES ONE FILTER OF HEADROOM: the
 * deepest lag the bank can actually report is (n-1)*SHIFT + (SIZE-1), so the
 * arithmetically tight bound would be (n-1)*SHIFT + SIZE. The n*SHIFT + SIZE
 * form kept here is the pre-pool formula, and it is kept ON PURPOSE, for
 * three measured reasons rather than caution:
 *
 *   0. IT IS THE UPSTREAM FORMULA. WebRTC AEC3's MatchedFilter::
 *      GetMaxFilterLag() -- and this repo's Python port of it,
 *      matched_filter.py's get_max_filter_lag(), whose docstring spells the
 *      over-count out -- computes exactly `num_filters * intra_shift +
 *      filter_size` and uses it for exactly this purpose: SIZING the
 *      aggregator histograms, not describing reach. Diverging here would put
 *      the C port's histogram geometry out of step with both.
 *
 *   1. n=5 reproduces the pre-refactor array sizes EXACTLY (2433 highest-peak
 *      bins, 152 pre-echo bins), so the default configuration is byte-exact
 *      by construction rather than by argument.
 *
 *   2. The pre-echo histogram's SIZE IS BEHAVIOURAL, not just capacity. Its
 *      windowed local-max scan walks fixed 32-bin windows with a 0.7^k
 *      penalty and stops when fewer than 32 bins remain, so the bin count
 *      decides HOW MANY windows are scanned. Under the tight formula, n=2
 *      would yield 56 bins = 1 window covering bins 0..31 -- while that bank
 *      can genuinely report up to bin 55, silently dropping half its own
 *      search range. The headroom form yields 80 bins = 2 windows and covers
 *      it. Checked for every n in [1,5]: the headroom form always scans
 *      exactly the windows that contain the bank's reachable bins, and the
 *      windows it drops are provably all-zero (they sit above the reachable
 *      lag), which can never win the scan because best_value is >= 0 after
 *      the first window. So n=1..5 all keep the behaviour they had when the
 *      arrays were carved at the n=5 bound.
 *
 * The over-sized tail is behaviour-neutral for the highest-peak histogram by
 * the same standing argument as before: bins above the reachable lag are
 * never incremented, and both da_argmax_i and the incremental-argmax tracker
 * return the FIRST maximum, so an all-zero tail can never win a tie. */
#define DA_RING_CAPACITY_FOR(n) \
    ((DA_WINDOW_SIZE_SB + DA_ALIGNMENT_SHIFT_SB * ((n) - 1) + 1) * DA_SUB_BLOCK_SIZE)
#define DA_MAX_FILTER_LAG_FOR(n) ((n) * DA_FILTER_INTRA_SHIFT + DA_FILTER_SIZE)
#define DA_HP_HIST_SIZE_FOR(n)   (DA_MAX_FILTER_LAG_FOR(n) + 1)
#define DA_PE_HIST_SIZE_FOR(n) \
    (((DA_MAX_FILTER_LAG_FOR(n) + 1) * DA_DOWN_SAMPLING_FACTOR) >> 6)
#define DA_THRESH_INITIAL       5
#define DA_THRESH_CONVERGED     20
#define DA_CONSISTENT_EST_THR   125
#define DA_PRE_ECHO_UPDATES_TO_REPORT 50
#define DA_ACCUMULATED_ERROR_SUBSAMPLE_RATE 4
#define DA_BLOCK_SIZE_LOG2      6    /* kBlockSizeLog2 (64 = 1<<6) */
#define DA_K_MFW_SUB_BLOCKS     32   /* kMatchedFilterWindowSizeSubBlocks */
#define DA_K_NUM_BLOCKS_PER_SEC 250  /* kNumBlocksPerSecond (16000/64) -- FIXED
                                      * at the 16kHz-native rate; DA_HIST_WINDOW
                                      * and the ring/histogram arrays it sizes
                                      * are NOT yet rate-parameterised for
                                      * sample rates above 16 kHz (tracked
                                      * alongside the decimator's fixed
                                      * anti-alias biquad coefficients, which
                                      * have the same 16kHz-only limitation --
                                      * see downsampled_ring.py's docstring). */
/* DA_STABILITY_RESET_HOPS was a frozen #define (3000, = ms_to_hops(30000)
 * only at hop=160/sr=16000, wrong even at sr=16000 since this counter ticks
 * once per DA_AEC3_BLOCK_SIZE=64-sample inner block -- not once per outer
 * hop); now computed live in delay_aec3_init() from sample_rate, matching
 * the Python ClockdriftDetector fix (see clockdrift_detector.py). */

/* ---- biquad cascade ----
 * Max sections: 4 (the ds4 LP is elliptic, 3 sections; the 48k->16k
 * pre-decimation anti-alias stage below is elliptic order-7, 4 sections). */
#define DA_BQ_MAX_SECTIONS 4
typedef struct {
    int    n_sections;
    float  b[DA_BQ_MAX_SECTIONS][3];    /* {b0,b1,b2} per section */
    float  a[DA_BQ_MAX_SECTIONS][2];    /* {a1,a2} per section (a0 normalised) */
    float  z[DA_BQ_MAX_SECTIONS][2];    /* per-section state */
} DaBiquad;

/* ---- 48kHz -> 16kHz pre-decimation sidechain ----
 * DelayAec3's own internal decimator/matched-filter/clockdrift constants are
 * all native to a 16kHz feed rate (DA_K_NUM_BLOCKS_PER_SEC etc. above). At
 * 48kHz, this stage anti-alias-filters + decimates-by-3 BEFORE any of that
 * internal machinery ever sees a sample, so DelayAec3 always operates on a
 * genuine 16kHz-equivalent stream regardless of the caller's native rate --
 * exactly mirroring the shared 4ch pipeline's SharedMatchedDelayEstimator
 * (Python pipeline.py), which does the same 48->16 reduction externally
 * before feeding this same estimator. Two independent filter+phase chains
 * (capture/near, render/far) since they're independent signals.
 *
 * The two scratch buffers this stage needs are POOL-CARVED AND 48 kHz-ONLY
 * (plan §3.2.7): sized DA_RESAMPLE48_CAP_FOR(hop) from the resolved hop, and
 * not carved at all at any other rate -- a 16 kHz instance no longer pays for
 * two 48 kHz scratch arrays it can never use. The old fixed
 * DA_RESAMPLE48_SCRATCH_MAX=192 is gone. */
#define DA_RESAMPLE48_FACTOR      3
/* Output samples per call is at most ceil(hop/3) (the phase only ever moves
 * which samples are kept, never how many, beyond that ceiling); hop/3 + 1 is
 * >= ceil(hop/3) for every hop and is exactly 171 at the only 48 kHz grid
 * (hop=512), which is that ceiling. */
#define DA_RESAMPLE48_CAP_FOR(hop) ((hop) / DA_RESAMPLE48_FACTOR + 1)
typedef struct {
    DaBiquad near_lp;    /* capture channel anti-alias LP (order-7, 4 sections) */
    DaBiquad far_lp;     /* render channel anti-alias LP (order-7, 4 sections) */
    int      phase;      /* decimation phase, continuous across hops (like the
                           * 4ch pipeline's own delay_phase) */
} DaResample48;

typedef struct {
    DaBiquad anti_alias;       /* LP ds4 (3 sections) */
    DaBiquad noise_reduction;  /* HP (1 section) */
} DaDecimator;

typedef struct {
    float *buffer;     /* pool-carved [capacity], ALIGN16 */
    int    capacity;   /* DA_RING_CAPACITY_FOR(num_filters) */
    int    write;
} DaRing;

typedef struct {
    /* Runtime bank size, [1, DA_NUM_FILTERS]. EVERY row below exists -- the
     * bank is carved to exactly this size, so there is no inactive tail to
     * initialise for byte-image determinism any more (the pre-pool code
     * zeroed all 5 rows for exactly that reason). */
    int   num_filters;
    /* Pool-carved, row-major, ALIGN16. Row i of `filters` is
     * filters + i*DA_FILTER_SIZE (2048 B, itself 16-aligned, so every row is
     * 16-aligned); row i of `accumulated_error` is
     * accumulated_error + i*DA_ACC_ERR_SIZE (512 B, likewise). */
    float *filters;             /* [num_filters][DA_FILTER_SIZE] */
    float *accumulated_error;   /* [num_filters][DA_ACC_ERR_SIZE] */
    /* per-tap-prefix err, last filter (matches Python's single shared
     * buffer). ONE filter-size buffer at every bank size -- it is
     * recomputed from scratch for the last searched filter on every update,
     * so it never needed to be per-filter (plan §3.2.6). */
    float *instantaneous_error; /* [DA_ACC_ERR_SIZE] */
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
    int *histogram;   /* pool-carved [hist_size], ALIGN16 */
    int *ring;        /* pool-carved [DA_HIST_WINDOW], ALIGN16 */
    int hist_size;    /* DA_HP_HIST_SIZE_FOR(num_filters) */
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
    int *histogram;   /* pool-carved [hist_size], ALIGN16 */
    int *ring;        /* pool-carved [DA_HIST_WINDOW], ALIGN16 */
    int hist_size;    /* DA_PE_HIST_SIZE_FOR(num_filters) -- see the
                        * DA_MAX_FILTER_LAG_FOR headroom note: this count
                        * decides how many windows the local-max scan walks,
                        * so it is a BEHAVIOURAL term, not just a capacity. */
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
    int stability_reset_hops;  /* live-computed, was DA_STABILITY_RESET_HOPS=3000 */
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
    int consistent_estimate_threshold; /* live-computed, was DA_CONSISTENT_EST_THR=125 */
    /* outer->inner edge buffers (raw 16 kHz samples awaiting a 64-sample
     * block). Deliberately EMBEDDED, not pool-carved: DA_AEC3_BLOCK_SIZE is
     * the fixed AEC3 inner-block length, independent of both the bank size
     * and the hop, so these are 256 B each at every config -- pooling them
     * would add two carve steps and buy nothing. */
    float capture_pending[DA_AEC3_BLOCK_SIZE];
    float render_pending[DA_AEC3_BLOCK_SIZE];
    int   pending_count;
} DaEstimator;

/* ===================== LegacyDelayShim ===================== */
typedef struct {
    DaEstimator est;
    /* 48kHz->16kHz sidechain (see DaResample48 above). rate_factor is 1 at
     * every native-16kHz-equivalent grid (est always runs at 16kHz-native
     * regardless), 3 only when this instance was constructed at 48000. */
    int          rate_factor;
    DaResample48 resample48;
    /* Pool-carved [resample_cap] each, ALIGN16 -- both NULL and
     * resample_cap == 0 at every rate other than 48 kHz, where the sidechain
     * does not run at all. */
    float       *near16_scratch;
    float       *far16_scratch;
    int          resample_cap;
    /* latest emitted estimate (raw samples in the ESTIMATOR's own 16kHz-
     * equivalent domain; delay_aec3_estimated_delay() rescales by
     * rate_factor before returning to the caller's native domain) */
    int          latest_valid;
    int          latest_delay;
    DelayQuality latest_quality;
    int          estimate_count;   /* _n_updates */
} DelayAec3;

/* ============================== lifecycle ================================
 * Canonical pool-first pair. Query, then place. Both take the SAME
 * (sample_rate, hop_size, num_filters) triple and derive every size from ONE
 * internal layout helper, so they cannot drift.
 *
 * sample_rate selects the 48kHz->16kHz sidechain (rate_factor=3 at 48000,
 * else 1) -- the underlying DaEstimator always runs at a fixed 16kHz-native
 * cadence (da_estimator_init() is always called with 16000, never the
 * caller's sample_rate) since accumulate() below guarantees it is only ever
 * fed a 16kHz-equivalent stream, either directly (native 16kHz callers) or
 * via the pre-decimation sidechain (48kHz callers). This mirrors the Python
 * SharedMatchedDelayEstimator's identical design.
 *
 * hop_size is the RESOLVED grid hop (aec_resolve_signal_grid). It sizes only
 * the 48 kHz sidechain scratch; DelayAec3 does not need, and deliberately
 * does not take, an fft_size (plan §2.4).
 *
 * num_filters is the number of staggered matched-filter hypotheses actually
 * searched; out-of-range values ([1, DA_NUM_FILTERS] exclusive) are
 * REJECTED here too -- delay_aec3_get_mem_size() returns 0 and
 * delay_aec3_init() returns -1 -- matching the fail-fast rule the
 * top-level aec_validate_config() already applies. A silent clamp at this
 * layer would let a direct low-level caller run a different bank size than
 * requested with no error signal. One shared
 * validity helper feeds both entry points, so a query and an init can never
 * disagree about whether a request is admissible.
 *
 * Reliable reach shrinks with the bank: (n-1)*DA_FILTER_INTRA_SHIFT +
 * (DA_FILTER_SIZE - 11) downsampled samples at 0.25 ms each, i.e.
 * n=1 -> 125 ms, 2 -> 221 ms, 3 -> 317 ms, 4 -> 413 ms, 5 -> 509 ms. Use a
 * short bank only where the bulk system delay is already compensated
 * out-of-band (AEC3's external-delay-hint stance); bench/dataset runs must
 * stay at 5. */

/* Bytes this exact config needs. 0 = invalid input (non-positive rate/hop,
 * or a size computation that overflowed). Always ALIGN16-sized, so callers
 * embedding it in a larger pointer-bump pool stay aligned afterwards. */
size_t delay_aec3_get_mem_size(int sample_rate, int hop_size, int num_filters);

/* Place the estimator in `mem` (>= delay_aec3_get_mem_size() bytes, 16-byte
 * aligned base). Returns 0 on success, -1 on NULL/misaligned/undersized/
 * invalid input. Allocates nothing. The carve is asserted to consume EXACTLY
 * the queried byte count. */
int delay_aec3_init(DelayAec3 *d, void *mem, size_t bytes,
                    int sample_rate, int hop_size, int num_filters);

/* Clears the signal chain (matched filter, aggregator, decimators, ring,
 * pending samples, sidechain). Keeps the geometry, the pool pointers and the
 * bank size -- never re-carves, never allocates. */
void delay_aec3_reset(DelayAec3 *d);

/* per-hop drive. near/far are length `hop` float arrays in the CALLER's
 * native sample-rate domain (near = HPF mic, far = raw reference, both
 * BEFORE ring-buffer alignment) -- at 48kHz they are internally anti-alias
 * filtered + decimated by 3 before reaching the 16kHz-native estimator; at
 * every other supported rate this is a no-op passthrough. Returns 1 iff a
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
int    delay_aec3_estimated_delay(const DelayAec3 *d);   /* -1 until first estimate;
                                                            * else caller-native-rate
                                                            * samples (rescaled by
                                                            * rate_factor internally) */
float  delay_aec3_confidence(const DelayAec3 *d);        /* 0.0 / 0.5 / 1.0 (float32, was double) */
int    delay_aec3_is_solid(const DelayAec3 *d);          /* confidence >= 1.0 */
int    delay_aec3_n_updates(const DelayAec3 *d);
int    delay_aec3_has_clockdrift(const DelayAec3 *d);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DELAY_AEC3_H */
