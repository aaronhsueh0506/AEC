/* Top-level echo-state machine: wires the ERLE, ERL, filter-quality, saturation, filter-delay,
 * initial-state and filter-analyzer sub-modules; integer counters only. Python reference:
 * modules/state/aec_state.py.
 * aec_state_update(): analyzer (taps and render block given) and its delays; filter delay; block
 * counters; ERLE (soft reset when the initial state just ended) then ERL; saturation detector;
 * initial state; filter quality (transparent mode always off).
 * handle_echo_path_change(): a delay change fully resets the state; a gain change resets the ERLE. */
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

typedef struct {
    int    use_linear_filter;
    int    conservative_initial_phase;
    float initial_state_seconds;
    int    delay_headroom_samples;
    int    num_capture_channels;
    int    echo_can_saturate;
    int    n_bins;
    /* -1 (AEC_STATE_STARTUP_HOPS_AUTO) resolves to the grid-correct 2 s hop count; any other value
     * is used verbatim. */
    int    erle_startup_hops;
    float  erle_min;
    float  erle_max_l;
    float  erle_max_h;
    int    erl_startup_hops;
    int    hop_size;
    int    sample_rate;
    int    enable_filter_analyzer;
    int    filter_taps_size;
    /* Capacity in floats of fa_abs_scratch (FilterAnalyzer abs-slice scratch); derived from the
     * true runtime filter length n_partitions * hop. fa_render_sq_scratch holds hop_size floats. */
    int    fa_scratch_size;
} AecStateConfig;

/* Caller-owned backing storage for the sub-estimator arrays, the FilterAnalyzer taps and the
 * FilterDelay channel array; sizes derive from n_bins, filter_taps_size and num_capture_channels. */
typedef struct {
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
    float   *erl;
    int     *erl_hold;
    int     *filter_delays_blocks;
    float   *fa_h_highpass;
    float   *fa_abs_scratch;
    float   *fa_render_sq_scratch;
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
    int                    has_filter_analyzer;

    int strong_not_saturated_render_blocks;
    int blocks_with_active_render;
    int capture_signal_saturation;
    int active_render_blocks;
} AecState;

void aec_state_config_defaults(AecStateConfig *cfg);

/* Construct. `st` must point at backing arrays of the correct sizes (see AecStateStorage). */
void aec_state_init(AecState *s, const AecStateConfig *cfg,
                    const AecStateStorage *st);

/* Forwards the FilterAnalyzer's on-demand tap provider so the tap-buffer owner can produce only
 * the slice step 1 consumes. No-op when the analyzer is disabled; cleared by aec_state_init().
 * Without a provider, filter_taps_full must be fully materialized by the caller. */
void aec_state_set_taps_provider(AecState *s, FaTapsProvider fn, void *ctx);

void aec_state_update_capture_saturation(AecState *s, int saturated);

/* delay_change: DelayAdjustment (0=NONE, 1=BUFFER_FLUSH, 2=NEW_DETECTED_DELAY); gain_change: 0/1. */
void aec_state_handle_echo_path_change(AecState *s, int gain_change,
                                       int delay_change);

/* Per-frame update (order above).
 * bridge_filter_converged: 0/1. external_delay: NULL or .reported == 0 for none.
 * render_psd, capture_psd, error_psd: float32[n_bins].
 * render_block: float32[render_block_len] or NULL (skips analyzer and saturation).
 * filter_taps_full: float32[filter_taps_size] or NULL.
 * x2_reverb_for_erle, capture_psd_erle: float32[n_bins] or NULL (render_psd / capture_psd used).
 * erle_coh_gate_mask: uchar[n_bins] or NULL. */
void aec_state_update(AecState *s,
                      int bridge_filter_converged,
                      const FilterDelayEstimate *external_delay,
                      const float *render_psd,
                      const float *capture_psd,
                      const float *error_psd,
                      int active_render,
                      float subtractor_s_refined_max_abs,
                      float subtractor_s_coarse_max_abs,
                      float echo_path_gain,
                      const float *render_block, int render_block_len,
                      const float *filter_taps_full, int filter_taps_full_len,
                      const float *x2_reverb_for_erle,
                      const float *capture_psd_erle,
                      const unsigned char *erle_coh_gate_mask);

int    aec_state_usable_linear_estimate(const AecState *s);
int    aec_state_saturated_echo(const AecState *s);
int    aec_state_transition_triggered(const AecState *s);
int    aec_state_min_direct_path_filter_delay(const AecState *s);
const float *aec_state_erle(AecState *s, int onset_compensated);
const float *aec_state_erle_unbounded(const AecState *s);
float aec_state_fullband_erle_log2(const AecState *s);
/* *valid == 0 when there is no estimate. */
float aec_state_get_inst_linear_quality_estimate(const AecState *s, int *valid);

#endif
