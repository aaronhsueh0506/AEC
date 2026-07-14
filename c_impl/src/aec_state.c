/* aec_state.c — C port of python/modules/state/aec_state.py.
 * Integration / orchestration layer; see aec_state.h for the full spec and the
 * strict update() order. */
#include "aec_state.h"

#include <assert.h>

/* DelayAdjustment enum (delay_types.py): NONE=0, BUFFER_FLUSH=1,
 * NEW_DETECTED_DELAY=2. */
#define AEC_DELAY_ADJ_NONE 0

void aec_state_config_defaults(AecStateConfig *cfg) {
    cfg->use_linear_filter = 1;
    cfg->conservative_initial_phase = 0;
    cfg->initial_state_seconds = 2.5;
    cfg->delay_headroom_samples = 32;
    cfg->num_capture_channels = 1;
    cfg->echo_can_saturate = 1;
    cfg->n_bins = 257;
    cfg->erle_startup_hops = 200;
    cfg->erle_min = 1.0f;
    cfg->erle_max_l = 4.0f;
    cfg->erle_max_h = 1.5f;
    cfg->erl_startup_hops = 200;
    cfg->hop_size = 160;
    cfg->sample_rate = 16000;
    cfg->enable_filter_analyzer = 1;
    cfg->filter_taps_size = 960;  /* hop=160 -> fft=512 -> 6 part * 160 = 960 */
    cfg->fa_scratch_size = 960;   /* default matches filter_taps_size at 16 kHz */
}

void aec_state_init(AecState *s, const AecStateConfig *cfg,
                    const AecStateStorage *st) {
    s->cfg = *cfg;
    s->st  = *st;

    initial_state_init(&s->initial_state,
                       cfg->conservative_initial_phase,
                       cfg->initial_state_seconds);
    filter_delay_init(&s->delay_state, st->filter_delays_blocks,
                      cfg->delay_headroom_samples, cfg->num_capture_channels);
    fq_init(&s->filter_quality, cfg->use_linear_filter);
    saturation_detector_init(&s->saturation_detector);

    erle_estimator_init(&s->erle_estimator,
                        cfg->erle_startup_hops, cfg->n_bins,
                        cfg->erle_min, cfg->erle_max_l, cfg->erle_max_h,
                        /*use_onset_detection=*/1, cfg->hop_size,
                        st->erle_max, st->erle, st->erle_oc, st->erle_unb,
                        st->erle_during, st->erle_coming_onset, st->erle_hold,
                        st->erle_y2_acc, st->erle_e2_acc, st->erle_low_render);

    erl_estimator_init(&s->erl_estimator, cfg->erl_startup_hops, cfg->n_bins,
                       cfg->hop_size, st->erl, st->erl_hold);

    s->has_filter_analyzer = cfg->enable_filter_analyzer ? 1 : 0;
    if (s->has_filter_analyzer) {
        /* FilterAnalyzer() Python defaults: active_render_limit=100.0,
         * bounded_erl=False, default_gain=1.0. */
        fa_init(&s->filter_analyzer, st->fa_h_highpass, cfg->filter_taps_size,
                100.0, 0, 1.0,
                st->fa_abs_scratch, cfg->fa_scratch_size,
                st->fa_render_sq_scratch, cfg->hop_size);
    }

    s->strong_not_saturated_render_blocks = 0;
    s->blocks_with_active_render = 0;
    s->capture_signal_saturation = 0;
}

/* ── queries ───────────────────────────────────────────────────────────── */

int aec_state_usable_linear_estimate(const AecState *s) {
    return fq_linear_filter_usable(&s->filter_quality)
           && s->cfg.use_linear_filter;
}

int aec_state_active_render(const AecState *s) {
    return s->blocks_with_active_render > AEC_STATE_ACTIVE_RENDER_BLOCKS;
}

int aec_state_saturated_echo(const AecState *s) {
    return saturation_detector_saturated_echo(&s->saturation_detector);
}

int aec_state_saturated_capture(const AecState *s) {
    return s->capture_signal_saturation;
}

int aec_state_transparent_mode_active(const AecState *s) {
    (void)s;
    return 0;   /* TransparentMode permanently None. */
}

int aec_state_transition_triggered(const AecState *s) {
    return initial_state_transition_triggered(&s->initial_state);
}

int aec_state_initial_state_active(const AecState *s) {
    return initial_state_initial_state_active(&s->initial_state);
}

int aec_state_min_direct_path_filter_delay(const AecState *s) {
    return filter_delay_min_direct_path(&s->delay_state);
}

const float *aec_state_erle(AecState *s, int onset_compensated) {
    return erle_estimator_erle(&s->erle_estimator, onset_compensated);
}

const float *aec_state_erle_unbounded(const AecState *s) {
    return erle_estimator_erle_unbounded(&s->erle_estimator);
}

float aec_state_fullband_erle_log2(const AecState *s) {
    return erle_estimator_fullband_erle_log2(&s->erle_estimator);
}

float aec_state_get_inst_linear_quality_estimate(const AecState *s, int *valid) {
    return erle_estimator_get_inst_linear_quality_estimate(&s->erle_estimator,
                                                           valid);
}

const float *aec_state_erl(const AecState *s) {
    return erl_estimator_erl(&s->erl_estimator);
}

float aec_state_erl_time_domain(const AecState *s) {
    return erl_estimator_erl_time_domain(&s->erl_estimator);
}

int aec_state_filter_analyzer_consistent(const AecState *s) {
    if (!s->has_filter_analyzer) return 0;
    return fa_any_filter_consistent(&s->filter_analyzer);
}

int aec_state_filter_analyzer_peak_index(const AecState *s) {
    if (!s->has_filter_analyzer) return -1;
    return fa_peak_index(&s->filter_analyzer);
}

float aec_state_filter_analyzer_max_echo_path_gain(const AecState *s) {
    if (!s->has_filter_analyzer) return 0.0;
    return fa_max_echo_path_gain(&s->filter_analyzer);
}

/* ── mutators ──────────────────────────────────────────────────────────── */

void aec_state_update_capture_saturation(AecState *s, int saturated) {
    s->capture_signal_saturation = saturated ? 1 : 0;
}

static void aec_state_full_reset(AecState *s) {
    s->strong_not_saturated_render_blocks = 0;
    s->blocks_with_active_render = 0;
    initial_state_reset(&s->initial_state);
    fq_reset(&s->filter_quality);
    s->capture_signal_saturation = 0;
    erle_estimator_reset(&s->erle_estimator, /*delay_change=*/1);
    erl_estimator_reset(&s->erl_estimator);
    if (s->has_filter_analyzer) {
        fa_reset(&s->filter_analyzer);
    }
}

void aec_state_handle_echo_path_change(AecState *s, int gain_change,
                                       int delay_change) {
    if (delay_change != AEC_DELAY_ADJ_NONE) {
        aec_state_full_reset(s);
    } else if (gain_change) {
        erle_estimator_reset(&s->erle_estimator, /*delay_change=*/0);
    }
}

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
                      const unsigned char *erle_coh_gate_mask) {
    (void)echo_psd;  /* not consumed at this layer */

    /* 1. any_filter_converged (= bridge.filter_converged). */
    int any_filter_converged = bridge_filter_converged ? 1 : 0;

    /* 1b. FilterAnalyzer (aec_state.cc:199-200). Runs BEFORE FilterDelay so the
     * per-channel delays are fresh. analyzer_delays is None unless the analyzer
     * is present AND both filter_taps_full and render_block are non-None. */
    int  analyzer_delay_storage[1];
    const int *analyzer_delays = NULL;
    int  analyzer_delays_len = 0;
    /* M4 (multi-rate consumption switch): filter_taps_full_len is a live,
     * per-hop value (aec.c recomputes it every hop as n_partitions*hop from
     * the then-current dims), not a construction-time constant like
     * cfg.filter_taps_size -- so, unlike the SuppressionGain tuning-table
     * length check (a pure init-time assert), this gets BOTH a debug assert
     * (catches a real per-hop wiring bug immediately) AND a release-path
     * guard (skips fa_update instead of letting it read filter_taps_full
     * against s->filter_analyzer's h_highpass/abs_scratch buffers, which
     * are sized from cfg.filter_taps_size, with a mismatched length -- an
     * OOB read if filter_taps_full_len is too short). Never fires for the
     * validated {16000} whitelist (both are n_partitions*hop for the same
     * np/hop pair every hop there). */
    assert(filter_taps_full == NULL
           || filter_taps_full_len == s->cfg.filter_taps_size);
    if (s->has_filter_analyzer && filter_taps_full != NULL
            && render_block != NULL
            && filter_taps_full_len == s->cfg.filter_taps_size) {
        fa_update(&s->filter_analyzer, filter_taps_full, render_block,
                  render_block_len);
        /* filter_delays_blocks() -> [delay_blocks] (single channel). */
        analyzer_delay_storage[0] = fa_min_filter_delay_blocks(&s->filter_analyzer);
        analyzer_delays = analyzer_delay_storage;
        analyzer_delays_len = 1;
    }

    /* 2. FilterDelay update (aec_state.cc:203-206). */
    filter_delay_update(&s->delay_state, analyzer_delays, analyzer_delays_len,
                        external_delay,
                        s->strong_not_saturated_render_blocks);

    /* 3. active_render + saturation block counters. */
    if (active_render) {
        s->blocks_with_active_render += 1;
    }
    if (active_render && !s->capture_signal_saturation) {
        s->strong_not_saturated_render_blocks += 1;
    }

    /* 4a. ERLE — transition-triggered soft reset BEFORE update. */
    if (initial_state_transition_triggered(&s->initial_state)) {
        erle_estimator_reset(&s->erle_estimator, /*delay_change=*/0);
    }
    int erle_converged = any_filter_converged;
    {
        const float *x2_in = (x2_reverb_for_erle != NULL)
                             ? x2_reverb_for_erle : render_psd;
        const float *y2_in = (capture_psd_erle != NULL)
                             ? capture_psd_erle : capture_psd;
        erle_estimator_update(&s->erle_estimator, x2_in, y2_in, error_psd,
                              erle_converged, erle_coh_gate_mask);
    }

    /* 4b. ERL update. */
    erl_estimator_update(&s->erl_estimator, render_psd, capture_psd,
                         any_filter_converged);

    /* 5. SaturationDetector. AEC3-strict echo_path_gain from FilterAnalyzer
     * (aec_state.cc:258-260) when the analyzer is present. */
    float epg = echo_path_gain;
    if (s->has_filter_analyzer) {
        epg = fa_max_echo_path_gain(&s->filter_analyzer);
    }
    if (s->cfg.echo_can_saturate && render_block != NULL) {
        saturation_detector_update(&s->saturation_detector,
                                   render_block, render_block_len,
                                   s->capture_signal_saturation,
                                   aec_state_usable_linear_estimate(s),
                                   subtractor_s_refined_max_abs,
                                   subtractor_s_coarse_max_abs,
                                   epg);
    }

    /* 6. InitialState (post-saturation flag). */
    initial_state_update(&s->initial_state, active_render,
                         s->capture_signal_saturation);

    /* 7. TransparentMode — None in BALANCED → SKIPPED. */

    /* 8. FilteringQualityAnalyzer (last). transparent_mode_active()==0;
     * filter_analyzer_consistent is a no-op in the Python fq.update so it is
     * not part of the C update signature (header note). */
    fq_update(&s->filter_quality,
              active_render,
              /*transparent_mode=*/0,
              s->capture_signal_saturation,
              /*external_delay_present=*/(external_delay != NULL
                                          && external_delay->reported) ? 1 : 0,
              any_filter_converged);
}
