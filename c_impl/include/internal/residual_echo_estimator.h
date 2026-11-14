/* Per-bin residual echo power R^2 (single channel), float32:
 *  - Linear path (usable_linear): R^2 = S^2_linear / ERLE + reverb tail.
 *  - Non-linear path: R^2 = X^2 * echo_path_gain^2 (+ non-linear term) + reverb.
 *  - Saturated echo: R^2 = Y^2 (capture power) in either path.
 * Owns a ReverbModel and a ReverbFrequencyResponse. The caller must call ree_update_reverb_models()
 * before ree_estimate() every hop. The reverb decay is rescaled to the live grid in double
 * precision. Python reference: modules/residual/residual_echo_estimator.py. */
#ifndef RESIDUAL_ECHO_ESTIMATOR_H
#define RESIDUAL_ECHO_ESTIMATOR_H

#include "reverb_model.h"
#include "reverb_frequency_response.h"

#define REE_DELAY_BUF_SIZE 16

typedef struct {
    float  min_noise_floor_power;
    float  noise_gate_power;        /* used only when !use_aec_residual_noise_gate */
    float  noise_gate_slope;
    float  stationary_gate_slope;
    int    model_reverb_in_nonlinear_mode;
} ReeEchoModelConfig;

typedef struct {
    int    n_bins;

    ReeEchoModelConfig echo_model;
    float  default_gain_early;
    float  default_gain_late;
    float  tm_gain_early;
    float  tm_gain_late;
    int    erle_onset_comp_in_dominant;
    int    reverb_enabled;
    float  noise_floor_growth_per_hop; /* computed in ree_init() */
    /* Per-hop reverb decay for both dominant_nearend branches, resolved once in ree_init()
     * (0 when reverb is disabled). The mild value scales in double before the exponentiation, so it
     * cannot be derived from the steady one. */
    float  reverb_decay_per_hop;
    float  reverb_decay_per_hop_mild;
    float  reverb_tail_strength;
    int    use_aec_residual_noise_gate;
    int    use_stationarity_properties;  /* when set, the nonlinear-path residual noise gate is skipped */
    int    use_aec_echo_gen_window;
    int    nl_r2_enabled;
    float  nl_r2_alpha;
    float  nl_norm_power;
    float  nl_r2_cap_multiple;      /* <= 0 disables the cap */
    float  residual_noise_gate_power;
    int    noise_floor_hold_hops;
    int    render_pre_window_size;
    int    render_post_window_size;

    float *x2_noise_floor;
    int   *x2_noise_floor_counter;

    ReverbModel              reverb_model;
    ReverbFrequencyResponse  reverb_freq_resp;
    int                      have_reverb_freq_resp;

    /* ring render history, used only when !use_aec_echo_gen_window */
    float *render_history;
    int    render_history_size;
    int    render_history_idx;
    int    render_history_initialised;

    /* Delay-centred render deque stored as a ring of frames (index 0 = newest); delay_buf_count <=
     * 16. */
    float *delay_render_buf;
    int    delay_buf_count;
    int    delay_buf_head;

    float *reverb_render_history;
    int    reverb_buf_count;
    int    reverb_buf_head;

    float *last_r2_direct;
    float *last_r2_reverb;

    float *scratch;
} ResidualEchoEstimator;

/* Initialise. The caller supplies all backing storage (floats unless noted):
 * x2_noise_floor, reverb_model_storage, reverb_tail_storage, last_r2_direct_store,
 * last_r2_reverb_store, scratch_store: n_bins; x2_noise_floor_counter: n_bins int32;
 * render_history_storage: (pre + post + 1) x n_bins with pre = use_aec_echo_gen_window ? 1 : 0 and
 * post = 1; delay_render_storage, reverb_render_storage: REE_DELAY_BUF_SIZE x n_bins.
 * reverb_smoothing_base and reverb_use_conservative are supplied by the caller. */
void ree_init(ResidualEchoEstimator *r,
              int n_bins,
              int hop_size,
              int sample_rate,
              const ReeEchoModelConfig *echo_model,
              float default_gain, float tm_gain,
              int erle_onset_comp_in_dominant,
              double reverb_decay, double reverb_mild_decay_scale,
              int reverb_enabled, float reverb_tail_strength,
              int use_aec_residual_noise_gate,
              int use_stationarity_properties,
              int use_aec_echo_gen_window,
              int nl_r2_enabled, float nl_r2_alpha, float nl_norm_power,
              float nl_r2_cap_multiple,
              float residual_noise_gate_power,
              int noise_floor_hold_hops,
              int use_freq_response, int reverb_use_conservative,
              double reverb_smoothing_base,
              float *x2_noise_floor, int *x2_noise_floor_counter,
              float *reverb_model_storage, float *reverb_tail_storage,
              float *render_history_storage,
              float *delay_render_storage, float *reverb_render_storage,
              float *last_r2_direct_store, float *last_r2_reverb_store,
              float *scratch_store);

void ree_reset(ResidualEchoEstimator *r);

/* Refreshes the bound ReverbFrequencyResponse. frequency_response is row-major
 * (n_partitions x n_bins) float32; filter_quality_is_none != 0 skips the update. */
void ree_update_reverb_models(ResidualEchoEstimator *r,
                              const float *frequency_response,
                              int n_partitions,
                              int filter_delay_blocks,
                              double filter_quality,
                              int filter_quality_is_none,
                              int stationary_block);

/* Produces r2_out and r2_unbounded_out (n_bins floats each).
 * render_psd, capture_psd, s2_linear: float32[n_bins]; usable, saturated, transparent_mode: 0/1;
 * erle, erle_unbounded: float32[n_bins], read only on the linear non-saturated path.
 * force_nonlinear_path forces usable = 0. */
void ree_estimate(ResidualEchoEstimator *r,
                  const float *render_psd,
                  const float *capture_psd,
                  const float *s2_linear,
                  int dominant_nearend,
                  int usable, int saturated, int transparent_mode,
                  const float *erle, const float *erle_unbounded,
                  int filter_delay_blocks,
                  int filter_length_blocks,
                  int force_nonlinear_path,
                  float *r2_out, float *r2_unbounded_out);

#endif
