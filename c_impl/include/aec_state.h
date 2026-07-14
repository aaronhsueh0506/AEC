/* aec_state.h — C port of python/modules/state/aec_state.py.
 *
 * Top-level AEC3 state machine (WS5 Phase 5.2 capstone). INTEGRATION /
 * orchestration layer: the per-bin float math is already bit-exact in the 9
 * sub-modules (subband_erle / fullband_erle / erl_estimator / filter_quality /
 * saturation_detector / filter_delay / initial_state / filter_analyzer /
 * filter_state_bridge). This module wires them in the STRICT aec_state.cc
 * order and adds the few AecState-level integer counters.
 *
 * STRICT update() order (aec_state.cc:189-291 / aec_state.py):
 *   1.  any_filter_converged = bridge.filter_converged
 *   1b. if filter_analyzer && filter_taps_full && render_block:
 *           analyzer.update(taps, render_block)
 *           analyzer_delays = analyzer.filter_delays_blocks()  (per channel)
 *   2.  delay_state.update(analyzer_delays, external_delay,
 *                          strong_not_saturated_render_blocks)
 *   3.  if active_render: blocks_with_active_render++
 *       if active_render && !capture_saturation: strong_not_saturated++
 *   4a. if initial_state.transition_triggered():
 *           erle.reset(delay_change=False)
 *       erle_converged = any_filter_converged
 *       erle.update(x2 = x2_reverb_for_erle or render_psd,
 *                   y2 = capture_psd_erle or capture_psd,
 *                   e2 = error_psd, converged=erle_converged,
 *                   coh_gate_mask=...)
 *   4b. erl.update(render_psd, capture_psd, any_filter_converged)
 *   5.  _epg = analyzer.max_echo_path_gain() if analyzer else echo_path_gain
 *       if echo_can_saturate && render_block:
 *           saturation.update(render_block, capture_saturation,
 *                             usable_linear_estimate(), s_refined_max_abs,
 *                             s_coarse_max_abs, _epg)
 *   6.  initial_state.update(active_render, capture_saturation)
 *   7.  TransparentMode — ALWAYS None in BALANCED → SKIPPED.
 *   8.  filter_quality.update(active_render, transparent_mode_active()=false,
 *                             capture_saturation, external_delay_present,
 *                             any_filter_converged)
 *
 * handle_echo_path_change(gain_change, delay_change):
 *   if delay_change != NONE: _full_reset()
 *   elif gain_change:        erle.reset(delay_change=False)
 *
 * _full_reset (aec_state.cc:145-157):
 *   strong_not_saturated_render_blocks = 0
 *   blocks_with_active_render          = 0
 *   initial_state.reset(); filter_quality.reset()
 *   capture_signal_saturation = False
 *   erle.reset(delay_change=True); erl.reset()
 *   filter_analyzer.reset()
 *
 * BALANCED simplifications (verified against the .py):
 *   - TransparentMode permanently None → transparent_mode_active()=false,
 *     step 7 skipped, filter_quality always sees transparent_mode=0.
 *   - enable_filter_analyzer=True → step 1b runs; analyzer drives step-2
 *     delays AND step-5 echo_path_gain.
 *
 * PRECISION: no new float math at this layer — all counters are
 * integer/bool, and this file performs no floating-point computation of its
 * own. The float32-by-design conversion (f32 campaign; Python bit-exact
 * parity retired, drift accepted) is delegated to the sub-modules:
 *   - erle_estimator / subband_erle / fullband_erle / erl_estimator are
 *     float32-by-design; the getters that surface their scalars
 *     (aec_state_fullband_erle_log2, aec_state_get_inst_linear_quality_estimate,
 *     aec_state_erl_time_domain) are float32 here too.
 *   - erle_min / erle_max_l / erle_max_h in AecStateConfig feed directly into
 *     erle_estimator_init() (float32 params) and are float32 here as well —
 *     the (float) casts that used to bridge double config -> float ctor are
 *     gone.
 *   - initial_state_seconds is float32 end-to-end: its only consumer is
 *     initial_state_init() (initial_state.h/.c, NOT part of this conversion),
 *     which itself takes a float32 param.
 *   - subtractor_s_refined_max_abs / subtractor_s_coarse_max_abs /
 *     echo_path_gain (aec_state_update params) and
 *     aec_state_filter_analyzer_max_echo_path_gain are float32 too: they are
 *     pure pass-through plumbing to saturation_detector.h and
 *     filter_analyzer.h (both outside this conversion's file set); no
 *     arithmetic on them happens in this file, so the value is simply carried
 *     through unchanged.
 */
#ifndef AEC_STATE_H
#define AEC_STATE_H

#include <stddef.h>
#include <stdint.h>

#include "initial_state.h"
#include "filter_delay.h"
#include "filter_quality.h"
#include "saturation_detector.h"
#include "erle_estimator.h"
#include "erl_estimator.h"
#include "filter_analyzer.h"

/* AEC3 _ACTIVE_RENDER_BLOCKS = 80 (aec_state.py:45). */
#define AEC_STATE_ACTIVE_RENDER_BLOCKS 80

/* Config knobs (mirrors AecStateConfig fields the C path uses). */
typedef struct {
    int    use_linear_filter;          /* default 1 */
    int    conservative_initial_phase; /* default 0 */
    float initial_state_seconds;       /* default 2.5 */
    int    delay_headroom_samples;     /* default 32 */
    int    num_capture_channels;       /* default 1 */
    int    echo_can_saturate;          /* default 1 */
    int    n_bins;                     /* default 257 */
    int    erle_startup_hops;          /* default 200 */
    float  erle_min;                   /* default 1.0f */
    float  erle_max_l;                 /* default 4.0f */
    float  erle_max_h;                 /* default 1.5f */
    int    erl_startup_hops;           /* default 200 */
    int    hop_size;                   /* default 160 */
    int    sample_rate;                /* default 16000 (unused at this layer) */
    int    enable_filter_analyzer;     /* default 1 (BALANCED) */
    int    filter_taps_size;           /* full impulse-response length (FilterAnalyzer) */
} AecStateConfig;

/* Caller-owned backing storage for the per-bin sub-estimator arrays + the
 * FilterAnalyzer HPF taps + the FilterDelay channel array. Embed one
 * AecStateStorage alongside the AecState; sizes derive from n_bins /
 * filter_taps_size / num_capture_channels. */
typedef struct {
    /* SubbandErle per-bin arrays (length n_bins). */
    float   *erle_max;
    float   *erle;
    float   *erle_oc;
    float   *erle_unb;
    float   *erle_during;
    unsigned char *erle_coming_onset;
    int32_t *erle_hold;
    float   *erle_y2_acc;
    float   *erle_e2_acc;
    unsigned char *erle_low_render;
    /* ErlEstimator arrays. */
    float   *erl;                 /* length n_bins   */
    int     *erl_hold;            /* length n_bins-2 */
    /* FilterDelay channel delays (length num_capture_channels). */
    int     *filter_delays_blocks;
    /* FilterAnalyzer HPF taps (length filter_taps_size). */
    float   *fa_h_highpass;
} AecStateStorage;

typedef struct {
    AecStateConfig cfg;

    InitialState           initial_state;
    FilterDelay            delay_state;
    FilteringQualityAnalyzer filter_quality;
    SaturationDetector     saturation_detector;
    ErleEstimator          erle_estimator;
    ErlEstimator           erl_estimator;
    FilterAnalyzer         filter_analyzer;
    int                    has_filter_analyzer;   /* enable_filter_analyzer */

    /* AecState-level counters. */
    int strong_not_saturated_render_blocks;
    int blocks_with_active_render;
    int capture_signal_saturation;                /* bool */

    AecStateStorage st;
} AecState;

/* Fill cfg with the AecStateConfig dataclass defaults. */
void aec_state_config_defaults(AecStateConfig *cfg);

/* Construct. The AecStateStorage `st` must already point at backing arrays of
 * the correct sizes (see AecStateStorage doc). Mirrors AecState.__init__. */
void aec_state_init(AecState *s, const AecStateConfig *cfg,
                    const AecStateStorage *st);

/* ── public mutators ───────────────────────────────────────────────────── */

void aec_state_update_capture_saturation(AecState *s, int saturated);

/* handle_echo_path_change. delay_change: a DelayAdjustment enum value
 * (0=NONE,1=BUFFER_FLUSH,2=NEW_DETECTED_DELAY); gain_change: bool. */
void aec_state_handle_echo_path_change(AecState *s, int gain_change,
                                       int delay_change);

/* Per-frame update — strict aec_state.cc order.
 *   bridge_filter_converged   : bridge.filter_converged (bool)
 *   external_delay            : Optional[DelayEstimate]; pass NULL or a
 *                               FilterDelayEstimate with .reported==0 for None.
 *   render_psd/capture_psd/error_psd : float32 arrays, length n_bins.
 *   echo_psd                  : unused at this layer (kept for signature
 *                               parity; may be NULL).
 *   active_render             : bool.
 *   subtractor_s_refined_max_abs / subtractor_s_coarse_max_abs / echo_path_gain
 *                             : float32 (pass-through to saturation_detector.h).
 *   render_block              : float32 array, length render_block_len; NULL
 *                               signals Python None (skips analyzer + sat).
 *   filter_taps_full          : float32, length filter_taps_size; NULL == None.
 *   x2_reverb_for_erle        : float32 length n_bins or NULL (→ render_psd).
 *   capture_psd_erle          : float32 length n_bins or NULL (→ capture_psd).
 *   erle_coh_gate_mask        : uchar  length n_bins or NULL.
 */
void aec_state_update(AecState *s,
                      int bridge_filter_converged,
                      const FilterDelayEstimate *external_delay,
                      const float *render_psd,
                      const float *capture_psd,
                      const float *error_psd,
                      const float *echo_psd,
                      int active_render,
                      float subtractor_s_refined_max_abs,
                      float subtractor_s_coarse_max_abs,
                      float echo_path_gain,
                      const float *render_block, int render_block_len,
                      const float *filter_taps_full, int filter_taps_full_len,
                      const float *x2_reverb_for_erle,
                      const float *capture_psd_erle,
                      const unsigned char *erle_coh_gate_mask);

/* ── public queries (mirror AecState methods) ──────────────────────────── */

int    aec_state_usable_linear_estimate(const AecState *s);
int    aec_state_active_render(const AecState *s);
int    aec_state_saturated_echo(const AecState *s);
int    aec_state_saturated_capture(const AecState *s);
int    aec_state_transparent_mode_active(const AecState *s);   /* always 0 */
int    aec_state_transition_triggered(const AecState *s);
int    aec_state_initial_state_active(const AecState *s);
int    aec_state_min_direct_path_filter_delay(const AecState *s);
const float *aec_state_erle(AecState *s, int onset_compensated);
const float *aec_state_erle_unbounded(const AecState *s);
float aec_state_fullband_erle_log2(const AecState *s);
/* get_inst_linear_quality_estimate: *valid==0 mirrors a Python None. */
float aec_state_get_inst_linear_quality_estimate(const AecState *s, int *valid);
const float *aec_state_erl(const AecState *s);
float aec_state_erl_time_domain(const AecState *s);
int    aec_state_filter_analyzer_consistent(const AecState *s);
int    aec_state_filter_analyzer_peak_index(const AecState *s);
float aec_state_filter_analyzer_max_echo_path_gain(const AecState *s);

#endif /* AEC_STATE_H */
