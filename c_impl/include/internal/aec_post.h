/* Post-filter driver: assembles the per-hop post-filter step from the sub-modules (AecState,
 * ResidualEchoEstimator, SuppressionGain, stationarity, reverb models) and adds comfort noise (CNG)
 * and overlap-add. Stages: PSDs from the float32 spectra; windowed-capture Y^2 for the ERLE;
 * average-render reverb; coherence EMA and ERLE gate mask; CNG N2 tracking; output-base select
 * (|E| > |Y| guard), gain apply, CNG injection, irfft, sqrt-Hann synthesis and overlap-add.
 * Sub-module results (usable_linear, saturated_echo, r2, r2_unb, gain) are inputs; the caller owns
 * all storage. Arithmetic is float32. Python reference: modules/orchestrator.py. */
#ifndef AEC_POST_H
#define AEC_POST_H

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"
#include "reverb_model.h"

#include "aec_state.h"
#include "residual_echo_estimator.h"
#include "suppression_gain.h"
#include "stationarity_estimator.h"
#include "linear_filter_output.h"
#include "pbfdkf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The comfort-noise generator starts from AEC_CNG_SEED (aec.h) after construction and after every hard post-chain reset. */

/* Driver config and wall-clock-rescaled CNG constants; pass the values computed at construction. */
typedef struct {
    int    n_bins;
    int    fft_size;
    int    block_size;
    int    hop_size;

    int    erle_coh_gate_enabled;
    int    erle_windowed_capture_psd;
    int    erle_render_x2_psd_scale;
    int    output_capture_when_linear_unusable;
    int    enable_cng;
    /* When set (default off), the CNG tracker reads capture_psd_win instead of near_psd. */
    int    cng_n2_windowed_psd;

    float  erle_coh_gate_alpha;
    float  erle_coh_gate_threshold;

    /* CNG wall-clock-rescaled constants */
    float  cng_y2_alpha;
    float  cng_n2_track_freshness;
    float  cng_n2_track_retention;
    float  cng_n2_slow_up;
    float  cng_n2_initial_alpha;
    int    cng_n2_update_onset_hops;
    int    cng_n2_initial_duration_hops;
    float  noise_floor_int16sq;

    int    synth_window_len;

    /* Live observer: a second PBFDAF reads the far-end ring earlier-shifted than the main filter
     * (AecPostRunIn.observer_far_end). Its evidence raises R^2 (r2 = max(r2, ev); never r2_unb)
     * only while its low-frequency residual energy is no worse than the main filter's on the same
     * hop (EMA pair, alpha observer_err_ema_alpha, margin observer_corrob_margin), capped at
     * min(|observer.echo_spec - sel_echo|^2, |sel_esw|^2) and limited to observer_max_bin. Off by
     * default. */
    int    observer_enabled;
    int    observer_max_bin;
    double observer_err_ema_alpha;    /* exp(-hop_ms / live_observer_corrob_ms) */
    double observer_corrob_margin;    /* 10^(live_observer_corrob_margin_db / 10) */
} AecPostConfig;

typedef struct {
    AecPostConfig cfg;

    /* Convergence-flags y2 threshold, scaled to the live hop_size. */
    float y2_thr;

    /* Synthesis window (block_size), caller-owned, read-only. */
    const float *synth_window;

    /* Comfort-noise LCG state: AEC_CNG_SEED after every aec_post_reset(). */
    uint32_t cng_seed;
    /* Number of hard resets since aec_post_init() (wrapping); aec_post_reset_serial() publishes it. */
    uint32_t reset_serial;
    /* This hop's output guard (aec_post_run): 1 when the rendering base is the capture spectrum
     * error + echo instead of the linear residual. Cleared by aec_post_reset() and at the start of
     * every aec_post_run(). */
    int      output_guard_active;
    int      noise_initialized;
    int      n2_counter;
    float   *y2_smoothed;
    float   *n2;
    float   *n2_initial;

    float   *coh_sye_re;
    float   *coh_sye_im;
    float   *coh_syy;
    float   *coh_see;

    ReverbModel avg_reverb;

    float   *ola_buf;          /* NULL when no hop synthesizes (context_only) */

    /* per-hop scratch (caller-owned) */
    float   *near_psd;
    float   *far_psd;
    float   *echo_psd;
    float   *error_psd;
    float   *capture_psd_win;
    float   *x2_reverb_for_erle;
    unsigned char *coh_gate_mask;
    float   *comfort_noise;
    float   *nf;
    Complex *e_out_spec;       /* with e_out_full: NULL as ola_buf */
    float   *e_out_full;

    FftHandle *fft;

    /* double: compared against a hard threshold, so kept in the reference precision. */
    double observer_obs_err_ema;
    double observer_main_err_ema;
} AecPost;

/* Fill cfg with the balanced defaults; the caller overrides the wall-clock constants and flags. */
void aec_post_config_defaults(AecPostConfig *cfg);

/* Initialise. The caller supplies cfg, the read-only window and table, the FFT handle and all
 * backing storage; avg_reverb_storage is the ReverbModel's n_bins floats. */
void aec_post_init(AecPost *p, const AecPostConfig *cfg,
                   FftHandle *fft,
                   const float *synth_window,
                   float *avg_reverb_storage,
                   float *y2_smoothed, float *n2, float *n2_initial,
                   float *coh_sye_re, float *coh_sye_im,
                   float *coh_syy, float *coh_see,
                   float *ola_buf,
                   float *near_psd, float *far_psd, float *echo_psd,
                   float *error_psd, float *capture_psd_win,
                   float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                   float *comfort_noise, float *nf,
                   Complex *e_out_spec, float *e_out_full);

/* Hard reset of the driver-owned state: OLA buffer, CNG (seed AEC_CNG_SEED), coherence EMAs and
 * avg-reverb; clears the output-guard decision and advances reset_serial. aec_post_init() restarts
 * reset_serial at 0 after its own initial reset. */
void aec_post_reset(AecPost *p);

/* Per-hop magnitudes |.| of the spectra, passed in by the caller. */
typedef struct {
    const float *abs_near;
    const float *abs_far;
    const float *abs_sel_echo;
    const float *abs_error;
    const float *abs_echo_coh;
    const float *abs_nsw_e1;
} AecPostAbs;

/* Ring slots the average-render reverb reads: delay_idx is the partition `delay` blocks older
 * than the newest, past_idx the one just older; delay is clamped to [0, n_part - 2]. Requires
 * n_part >= 1. */
void aec_post_avg_reverb_slots(int n_part, int partition_idx, int delay,
                               int *delay_idx, int *past_idx);

/* aec_post_run drives AecState, ResidualEchoEstimator, SuppressionGain, StationarityEstimator
 * and LinearFilterSelect for one hop. The caller must advance the stationarity estimator for this
 * hop first. */

#define AEC_POST_DELAY_ADJ_NONE          0

/* Per-hop inputs and persistent objects for aec_post_run; pointers are caller-owned. */
typedef struct {
    const Complex *near_spec;
    const Complex *far_spec;
    const Complex *echo_spec;
    const Complex *error_spec_windowed;
    const Complex *W_all;
    const float   *x2_cache;             /* filter.x2_cache flat (n_part×nf): |X_buf|² per partition row;
                                          * NULL → no render-reverb term this hop. */
    const float   *sqrt_hann;
    int            partition_idx;
    int            n_partitions;
    const float   *filter_taps_full;
    int            filter_taps_full_len;

    int            shadow_present;
    const Complex *shadow_error_spec;
    const float   *last_shadow_output_time; /* _last_shadow_output_time;
                                             * NULL → not available        */
    float          s_ref_max;
    float          s_coa_max;

    const float   *raw_output;
    const float   *near_end;
    const float   *far_end;

    int            current_delay;
    int            delay_active;
    float          saturation_level;
    int            nearend_recent;        /* self._ne_recent_frames > 0: the energy-
                                             * based near-recent latch (bool) */
    int            pending_gain_change;
    int            pending_delay_change; /* pending echo-path delay change:
                                          *  -1 == None, else DelayAdjustment */

    int            stationarity_active_hops;
    int            stationarity_converge_hops;
    int            stationary_block;       /* raw is_block_stationary result;
                                             * not convergence-gated          */

    int            erle_coh_gate_enabled;
    int            use_stationarity_properties;
    int            dt_floor_enabled;
    int            context_only;          /* expose gain/R²/CNG seam but skip
                                             * unused private time synthesis */
    int            linear_context_only;   /* nothing reads R², comfort noise
                                             * or G_res: stop after the AecState
                                             * update (none of the skipped
                                             * state feeds the linear filter) */
    int            hold_state_only; /* this lane's G_res is unused; keep the DominantNearend hold state advancing (needs context_only) */
    float          active_render_threshold;

    /* RenderActivityDetector verdict, updated by the caller before this call. */
    int            render_activity_is_active;
    /* Non-NULL iff the live observer is enabled and this hop's shifted ring read was valid. */
    const float   *observer_far_end;
    /* Non-NULL: the far PSD and the scaled render block of another instance fed this same far spectrum and far hop; they
     * replace the computation here (and p->far_psd / render_block_scaled stay unwritten). */
    const float   *shared_far_psd;
    const float   *shared_render_block;
    float          far_mean_sq;           /* mean square of far_end over the hop (the active-render test against
                                           * active_render_threshold) */
} AecPostRunIn;

/* Sub-module bundle the driver wires; caller-owned. */
typedef struct {
    AecState              *state;
    ResidualEchoEstimator *ree;
    SuppressionGain       *sg;
    StationarityEstimator *stationarity;
    LinearFilterSelect    *lfs;
    FftHandle             *fft;
    /* Live observer filter (NULL when disabled); driven inside aec_post_run. */
    PBFDAF                *observer_filter;
} AecPostRunObj;

/* Scratch buffers (caller-owned). */
typedef struct {
    Complex *sel_esw;
    Complex *sel_echo;
    Complex *nsw_e1;
    Complex *ybase;
    /* Dedicated rendering-base buffer (n_bins): error + echo, written only on hops where the output
     * guard fires; the exposed aec_get_residual().base_spec so that no other scratch can overwrite
     * it before the caller reads it. */
    Complex *output_base;
    float   *abs_near;
    float   *abs_far;
    float   *abs_sel_echo;
    float   *abs_error;
    float   *abs_echo_coh;
    float   *abs_nsw_e1;
    float   *abs_ybase;
    float   *x2_at_delay;
    float   *w_mag2;
    float   *render_block_scaled;
    float   *r2;
    float   *r2_unb;
    float   *nearend_pwr;
    unsigned char *stat_mask;

    /* Live observer scratch: observer_diff_spec = observer.echo_spec - sel_echo;
     * observer_diff_mag2 / observer_cap_mag2 hold |diff|^2 and |sel_esw|^2. */
    float   *observer_out;
    Complex *observer_diff_spec;
    float   *observer_diff_mag2;
    float   *observer_cap_mag2;
} AecPostRunScratch;

/* Output guard (stage 6a), evaluated by aec_post_run once per hop after the usable verdict of this hop is
 * final and before any context exit: with output_capture_when_linear_unusable set, usable == 0 and
 * sum(abs_error^2) > sum(abs_ybase^2) (strict; float32 magnitudes, float32 pairwise reduction; abs_ybase is
 * computed here from sc->ybase), p->output_guard_active becomes 1 and the capture spectrum sel_esw + sel_echo
 * is written to sc->output_base; otherwise it becomes 0. Needs sc->sel_esw, sel_echo, ybase and abs_error of
 * this hop. Exposed for the threshold fixture test; callers outside aec_post_run need not call it. */
void aec_post_eval_output_guard(AecPost *p, AecPostRunScratch *sc, int usable);

/* Process one hop. out[hop] is always written: the synthesized output, or in->raw_output when
 * context_only. *out_pending_gain_change / *out_pending_delay_change receive the post-call state of
 * the pending EPV flags (0 / -1 when an event fired). */
void aec_post_run(AecPost *p,
                  const AecPostRunIn *in,
                  const AecPostRunObj *obj,
                  AecPostRunScratch *sc,
                  float *out,
                  int *out_pending_gain_change,
                  int *out_pending_delay_change);

#ifdef __cplusplus
}
#endif

#endif
