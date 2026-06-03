/* suppression_gain.h — C port of python/modules/residual/suppression_gain.py
 * (WS5). Single-channel + single-band AEC3 SuppressionGain.
 *
 * Byte-equal port: produces bit-identical gain[n_bins] (amplitude domain,
 * sqrt'd) to the Python reference. fp64 rule: ENR/EMR/min/max/CNG math runs
 * in DOUBLE; per-bin arrays Python declares float32 (gain, tuning tables)
 * stay float, computed-in-double-then-(float)-truncated where the inputs are
 * f64. Pairwise float32 sums (np.sum / np.mean over float32) replicated by
 * f32_pairwise_sum().
 *
 * Build: gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Ic_impl/include
 *   link: suppression_gain.c freq_utils.c aec3_scale.c reverb_frequency_response.c
 *
 * The caller owns all per-bin storage (passed at init) to keep the port
 * malloc-free in the audio path, matching the other WS5 modules.
 */
#ifndef SUPPRESSION_GAIN_H
#define SUPPRESSION_GAIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All config that the balanced path (and the default-OFF levers) needs.
 * Power-domain floors are precomputed by the caller (10^(db/10)). */
typedef struct {
    int    n_bins;
    int    sr;
    int    hop_size;

    /* derived bin indices */
    int    last_lf_band;
    int    first_hf_band;
    int    last_lf_smoothing_band;
    int    last_permanent_lf_smoothing_band;
    int    lf_smoothing_during_initial_phase;   /* bool */
    int    dne_lf_end;                           /* D3/D5 LF sum endpoint bin */
    int    nearend_smoother_n;                   /* moving-avg window in hops */

    /* audibility (WeightEchoForAudibility) */
    int    aud_lf_end_bin;
    int    aud_mf_end_bin;
    double floor_power;
    double aud_thr_lf;
    double aud_thr_mf;
    double aud_thr_hf;
    double low_render_limit;
    double normal_render_limit;

    /* HF limiter */
    int    hf_lgb;
    int    hf_biq;
    int    conservative_hf;                      /* bool */

    /* gain ratchet (wallclock-scaled by caller) */
    float  max_inc_normal;
    float  max_inc_nearend;
    float  max_dec_lf_normal;
    float  max_dec_lf_nearend;
    double floor_first_increase;

    /* LowNoiseRender */
    double low_render_threshold;

    /* split min-gain floor (default ON) */
    int    split_floor_enabled;                  /* bool */
    double split_floor_far_active;               /* power */
    double split_floor_far_silent;               /* power */
    double split_floor_latch_power;

    /* soft nearend blend (default ON in balanced) */
    int    soft_blend_enabled;                   /* bool */
    int    soft_blend_per_bin;                   /* bool */
    float  soft_blend_enr_thr;
    float  soft_blend_softness;

    /* DominantNearend thresholds */
    double dne_enr_threshold;
    double dne_enr_exit_threshold;
    double dne_snr_threshold;
    int    dne_use_during_initial_phase;         /* bool */
    int    dne_use_unbounded_echo;               /* bool */
    int    dne_lf_endpoint_bin;
    int    dne_trigger_threshold_hops;
    int    dne_hold_duration_hops;
    /* W4 loud-nearend ENR relax (default OFF) */
    int    dne_loud_relax_enabled;               /* bool */
    double dne_loud_snr_factor;
    double dne_loud_enr_threshold;

    /* default-OFF levers */
    int    ser_floor_enabled;                    /* bool */
    float  ser_floor_strength;
    int    coh_gain_floor_enabled;               /* bool (always None feed) */
    float  coh_gain_floor_strength;
    int    d5_ne_floor_enabled;                  /* bool */
    float  d5_ne_floor_strength;
    int    hf_min_gain_floor_dne_enabled;        /* bool */
    double hf_min_gain_floor_dne_power;
    int    stat_aware_ne_proxy_enabled;          /* bool */
    double stat_aware_ne_proxy_threshold;
} SuppressionGainConfig;

/* Per-bin tuning tables (n_bins each), supplied by caller. */
typedef struct {
    const float *nearend_enr_tr;
    const float *nearend_enr_su;
    const float *nearend_emr_tr;
    const float *normal_enr_tr;
    const float *normal_enr_su;
    const float *normal_emr_tr;
} SuppressionGainTuning;

/* Mutable per-instance state. All n_bins arrays are caller-owned scratch. */
typedef struct {
    SuppressionGainConfig cfg;
    SuppressionGainTuning tun;

    /* persistent state across hops */
    float *last_gain;           /* [n_bins] */
    float *last_nearend;        /* [n_bins] */
    float *last_echo;           /* [n_bins] */
    double low_render_avg_power;
    int    far_active_latched;  /* bool */
    int    initial_state;       /* bool */
    double stat_mask_frac;

    /* DominantNearend state */
    int    dne_trigger_counter;
    int    dne_hold_counter;
    int    dne_nearend_state;   /* bool */

    /* moving-average ring buffer for the nearend smoother: stores up to
     * nearend_smoother_n most-recent input spectra, oldest..newest. */
    float *ma_buf;              /* [nearend_smoother_n * n_bins] */
    int    ma_count;            /* number of valid rows (<= n) */
    int    ma_head;             /* index of oldest row */

    /* scratch (caller-owned) */
    float *nearend;             /* [n_bins] smoothed input */
    float *weighted_residual;   /* [n_bins] */
    float *min_gain;            /* [n_bins] */
    float *max_gain;            /* [n_bins] */
    float *g_raw;               /* [n_bins] */
    float *gain;                /* [n_bins] output */
    float *sum_scratch;         /* [n_bins] pairwise-sum helper */
} SuppressionGain;

/* Initialise. Caller provides cfg/tuning and all the scratch/state arrays. */
void suppression_gain_init(SuppressionGain *sg,
                           const SuppressionGainConfig *cfg,
                           const SuppressionGainTuning *tun,
                           float *last_gain, float *last_nearend,
                           float *last_echo, float *ma_buf,
                           float *nearend_scratch, float *weighted_residual,
                           float *min_gain, float *max_gain, float *g_raw,
                           float *gain_out, float *sum_scratch);

void suppression_gain_set_initial_state(SuppressionGain *sg, int state);

/* is_dominant_nearend — returns the RAW DominantNearendDetector hold state
 * (== _dominant_nearend.is_nearend_state()). The orchestrator reads this BEFORE
 * get_gain (one-hop lag: it reflects the previous hop's get_gain update). */
int suppression_gain_is_dominant_nearend(const SuppressionGain *sg);

/* One hop. Inputs are the f32 spectra (n_bins) + the f32 render block
 * (hop_size). Returns the gain pointer (== sg->gain). */
const float *suppression_gain_get_gain(
    SuppressionGain *sg,
    const float *nearend_spectrum,        /* [n_bins] Y^2 / E^2 */
    const float *residual_echo,           /* [n_bins] R^2 */
    const float *residual_echo_unbounded, /* [n_bins] R^2_unb */
    const float *comfort_noise,           /* [n_bins] CN power */
    const float *render_block,            /* [hop_size] time-domain render */
    int          clock_drift,             /* bool */
    int          saturated_echo);         /* bool (aec_state.saturated_echo) */

/* numpy-1.26-bit-exact pairwise float32 sum (from reverb_frequency_response.c). */
float f32_pairwise_sum(const float *a, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* SUPPRESSION_GAIN_H */
