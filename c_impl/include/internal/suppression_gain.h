/* Single-channel, single-band suppression gain. Produces gain[n_bins] (amplitude, sqrt'd) in
 * float32; the ENR/EMR, min/max-gain and comfort-noise math, the per-bin arrays and the tuning
 * tables are all float32 (pairwise sums via sk_pairwise_sum_tailfold_b_f32).
 * The caller owns all per-bin storage (passed at init); no allocation.
 * Python reference: modules/residual/suppression_gain.py. */
#ifndef SUPPRESSION_GAIN_H
#define SUPPRESSION_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Configuration. Power-domain floors are precomputed by the caller (10^(db/10)). */
typedef struct {
    int    n_bins;
    int    sr;
    int    hop_size;

    int    last_lf_smoothing_band;
    int    last_permanent_lf_smoothing_band;
    int    lf_smoothing_during_initial_phase;
    int    lf_clamp_bin;
    int    dne_lf_end;
    int    nearend_smoother_n;                   /* moving-average window in hops */

    int    aud_lf_end_bin;
    int    aud_mf_end_bin;
    float  floor_power;
    float  aud_thr_lf;
    float  aud_thr_mf;
    float  aud_thr_hf;
    float  low_render_limit;
    float  normal_render_limit;

    int    hf_lgb;
    int    hf_biq;
    int    conservative_hf;

    float  max_inc_normal;
    float  max_inc_nearend;
    float  max_dec_lf_normal;
    float  max_dec_lf_nearend;
    float  floor_first_increase;

    float  low_render_threshold;

    int    split_floor_enabled;
    float  split_floor_far_active;
    float  split_floor_far_silent;
    /* DT-gated floor, used instead of far_active when dt_protect_active is set. */
    float  split_floor_dt;
    float  split_floor_latch_power;

    int    soft_blend_enabled;
    int    soft_blend_per_bin;
    float  soft_blend_enr_thr;
    float  soft_blend_softness;

    float  dne_enr_threshold;
    float  dne_enr_exit_threshold;
    float  dne_snr_threshold;
    int    dne_use_during_initial_phase;
    int    dne_use_unbounded_echo;
    int    dne_lf_endpoint_bin;
    int    dne_trigger_threshold_hops;
    int    dne_hold_duration_hops;

    int    stat_aware_ne_proxy_enabled;
    float  stat_aware_ne_proxy_threshold;
} SuppressionGainConfig;

/* Per-bin tuning tables (n_bins each), supplied by the caller. */
typedef struct {
    const float *nearend_enr_tr;
    const float *nearend_enr_su;
    const float *nearend_emr_tr;
    const float *normal_enr_tr;
    const float *normal_enr_su;
    const float *normal_emr_tr;

    /* Length the per-rate lookup row baked the six tables at; must equal cfg->n_bins. */
    int table_len;
} SuppressionGainTuning;

/* Mutable per-instance state. All n_bins arrays are caller-owned. */
typedef struct {
    SuppressionGainConfig cfg;
    SuppressionGainTuning tun;

    float *last_gain;
    float *last_nearend;
    float *last_echo;
    float  low_render_avg_power;
    float  low_render_iir_decay;  /* set in suppression_gain_init() */
    float  low_render_iir_weight; /* set in suppression_gain_init() */
    int    far_active_latched;
    int    dt_protect_active;   /* set by the orchestrator before get_gain; lifts the floor to split_floor_dt */
    /* Runtime retarget of cfg.split_floor_far_active: cfg holds the target, the live value is what
     * get_min_gain applies; they differ only after the _set_split_floor_far_active_db setter. */
    float  split_floor_far_active_live;
    float  split_floor_ramp_ratio;  /* per-hop power-domain factor; 1 = idle */
    int    initial_state;
    float  stat_mask_frac;

    int    dne_trigger_counter;
    int    dne_hold_counter;
    int    dne_nearend_state;

    /* Moving-average ring of the nearend_smoother_n most recent spectra, oldest to newest. */
    float *ma_buf;
    int    ma_count;
    int    ma_head;

    float *nearend;
    float *weighted_residual;
    float *min_gain;
    float *max_gain;
    float *g_raw;
    float *gain;
    float *sum_scratch;
} SuppressionGain;

/* Initialise. The caller provides cfg, tuning and all state and scratch arrays. */
void suppression_gain_init(SuppressionGain *sg,
                           const SuppressionGainConfig *cfg,
                           const SuppressionGainTuning *tun,
                           float *last_gain, float *last_nearend,
                           float *last_echo, float *ma_buf,
                           float *nearend_scratch, float *weighted_residual,
                           float *min_gain, float *max_gain, float *g_raw,
                           float *gain_out, float *sum_scratch);

void suppression_gain_set_initial_state(SuppressionGain *sg, int state);

/* Retargets the far-active split floor on a running instance. `db` is range-checked to [-300, 50].
 * ramp_ms == 0 applies on the next hop; ramp_ms > 0 (at most 60 s) walks to the target one step per
 * get_gain() call, linearly in dB. A call during a ramp restarts from the live value. Call between
 * hops; not thread-safe. Returns 0, or -1 on a NULL instance or out-of-range argument (nothing is
 * written). */
int suppression_gain_set_split_floor_far_active_db(SuppressionGain *sg,
                                                   float db, float ramp_ms);

/* Retargets the DT-lifted floor (no ramp; applies from the next hop with dt_protect_active).
 * Returns 0, or -1 on a NULL instance or out-of-range db. */
int suppression_gain_set_split_floor_dt_db(SuppressionGain *sg, float db);

/* Enables or disables all three split floors at once (`enabled` coerced to 0/1, immediate).
 * Returns 0, or -1 on a NULL instance. */
int suppression_gain_set_split_floor_enabled(SuppressionGain *sg, int enabled);

/* Raw DominantNearend hold state; lags one hop. */
int suppression_gain_is_dominant_nearend(const SuppressionGain *sg);

/* State-updating prefix of get_gain(): advances the DominantNearend hold state without computing
 * the gain array, for callers that never consume the gain. get_gain() calls it internally. */
void suppression_gain_update_dominant_nearend(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise);

/* Far-end statistics of one hop as get_gain() derives them from its int16-scaled render block of n samples: the pairwise float32 sum of
 * the squares and the largest square; both are 0 for NULL, n <= 0 or n > 1024. */
void suppression_gain_render_stats(const float *render_block, int n,
                                   float *sum_sq, float *peak_sq);

/* One hop. Inputs are float32 spectra (n_bins) and the render block (hop_size). Returns
 * sg->gain. Equal to suppression_gain_render_stats() on the block followed by get_gain_stats(). */
const float *suppression_gain_get_gain(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise,
    const float *render_block,            /* [hop_size] time-domain render */
    int          clock_drift,
    int          saturated_echo);

/* One hop with the far-end statistics given directly (see suppression_gain_render_stats) instead
 * of the time-domain block; gain and state are exactly those get_gain() produces for a block with
 * these two numbers. Returns sg->gain. */
const float *suppression_gain_get_gain_stats(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise,
    float        render_sum_sq,           /* sum of squared render samples over the hop */
    float        render_peak_sq,          /* largest squared render sample of the hop */
    int          clock_drift,
    int          saturated_echo);

#ifdef __cplusplus
}
#endif

#endif
