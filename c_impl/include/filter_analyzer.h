/* filter_analyzer.h — byte-equal C port of
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
 * BYTE-EQUAL parity notes (numpy 1.26 -> C, captured from real balanced/DT
 * case wav/aec_challenge_blind/doubletalk/0I0XMl3M..._doubletalk_*.wav,
 * preset=balanced fl=832 hop=160):
 *   - filter_taps : float32, size 960 (6 partitions * 160).
 *   - render_block: float32, size 160 (= render_block_scaled, int16 amplitude).
 *   - _h_highpass : float32. HPF taps (float32) * filter_taps (float32) all f32.
 *   - _gain : python float (f64); _peak_index / _delay_blocks : python int.
 *   - ConsistentFilterDetector floor accumulators:
 *       * np.abs(filter_taps_slice).sum()  -> float32 PAIRWISE sum
 *         (f32_pairwise_sum), then float()->f64 accumulate into _floor_accum.
 *       * seg.max(initial=0.0)             -> float32 max, widened to f64.
 *       * render active power
 *         (render_block.astype(f64)**2).sum() -> float64 PAIRWISE sum
 *         (f64_pairwise_sum) compared > _active_render_threshold (f64).
 *   - peak find: h[k]*h[k] in float32; np.argmax returns FIRST max index.
 *
 * Build with -ffp-contract=off so the float32 multiply/add/compare is not fused.
 */
#ifndef FILTER_ANALYZER_H
#define FILTER_ANALYZER_H

#include <stddef.h>

/* Project-locked framing (modules/_rates.py): hop=160, sr=16000. */
#define FA_HOP_SAMPLES               160
#define FA_SR_HZ                     16000
#define FA_HOPS_PER_SECOND           (FA_SR_HZ / FA_HOP_SAMPLES)        /* 100 */
#define FA_CONVERGENCE_THRESHOLD_HOPS (5 * FA_HOPS_PER_SECOND)          /* 500 */
/* int(1.5 * 100) = 150. */
#define FA_CONSISTENT_HOLD_HOPS      150

/* ConsistentFilterDetector mirror state (reset each full-filter sweep). */
typedef struct {
    double active_render_threshold;  /* (active_render_limit^2) * HOP_SAMPLES   */
    int    significant_peak;         /* bool                                    */
    double floor_accum;              /* f64                                     */
    double secondary_peak;           /* f64 (holds widened f32 max)             */
    int    floor_low_limit;
    int    floor_high_limit;
    long   counter;
    int    delay_ref;                /* starts at -10                           */
} FaConsistentDetector;

typedef struct {
    double active_render_threshold;  /* (active_render_limit^2) * HOP_SAMPLES   */
    int    bounded_erl;              /* bool                                    */
    double default_gain;
    FaConsistentDetector consistent;

    int    region_start;
    int    region_end;
    long   blocks_since_reset;
    int    peak_index;
    double gain;
    int    consistent_estimate;      /* bool                                    */
    int    delay_blocks;

    int    size;                     /* filter length (== filter_taps size)     */
    float *h_highpass;               /* owned by caller; length `size`, f32     */
} FilterAnalyzer;

/* h_highpass_storage must hold at least `size` floats (filter_taps length).
 * active_render_limit / bounded_erl / default_gain match the Python defaults
 * (100.0 / 0 / 1.0) unless overridden. `size` is the full filter length. */
void fa_init(FilterAnalyzer *m, float *h_highpass_storage, int size,
             double active_render_limit, int bounded_erl, double default_gain);

void fa_reset(FilterAnalyzer *m);

/* Per-hop update. filter_taps (float32, length == size) and render_block
 * (float32, length render_block_len). */
void fa_update(FilterAnalyzer *m, const float *filter_taps,
               const float *render_block, int render_block_len);

/* Public queries. */
int    fa_min_filter_delay_blocks(const FilterAnalyzer *m);
int    fa_any_filter_consistent(const FilterAnalyzer *m);
double fa_max_echo_path_gain(const FilterAnalyzer *m);
int    fa_peak_index(const FilterAnalyzer *m);
const float *fa_get_adjusted_filters(const FilterAnalyzer *m);

/* numpy-1.26-bit-exact pairwise sums (exposed for the parity test). */
float  fa_f32_pairwise_sum(const float *a, size_t n);
double fa_f64_pairwise_sum(const double *a, size_t n);

#endif /* FILTER_ANALYZER_H */
