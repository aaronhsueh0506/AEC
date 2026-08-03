/* residual_echo_estimator.h — C port of
 * python/modules/residual/residual_echo_estimator.py
 * (mirrors AEC3 residual_echo_estimator.cc). Single-channel.
 *
 * Computes R² (residual echo power²) per-bin from:
 *   - Linear path  (usable_linear=True):  R² = S²_linear / ERLE + reverb tail
 *   - NonLinear   (usable_linear=False):  R² = X² · echo_path_gain² (+nl) + reverb
 *   - Saturated echo override: R² = Y² (capture power) in either path.
 *
 * PARITY (numpy 1.26 → C, captured from the real balanced pipeline):
 *   - render_psd / capture_psd / s2_linear / erle / erle_unbounded are all
 *     float32[257]; frequency_response is float32 (n_partitions × 257).
 *   - r2 / r2_unbounded / _last_r2_* are float32 buffers (`np.empty(f32)` then
 *     f32 stores). All the arithmetic feeding them is all-float32 (the inputs
 *     are f32 and every scalar multiplier — 1e-30, echo_path_gain=1.0,
 *     noise_gate_slope=0.3, etc. — is value-cast to f32 in the array op, the
 *     numpy<2 in-place / array-op rule, NOT the opposite scalar rule).
 *   - nl_r2 = (nl_alpha * x2**2 / norm).astype(f32): x2 is f32, alpha=0.1 and
 *     norm=1.07e7 are python floats in an ARRAY expression → all-float32
 *     (0.1*x2² then /norm in f32), then explicit (float) cast (no-op).
 *   - reverb tail comes from ReverbModel + ReverbFrequencyResponse C ports.
 *   - decay = aec3_per_block_growth_to_per_hop(0.83f, hop_size, sample_rate):
 *     wall-clock-preserving rescale (accounts for BOTH hop_size AND
 *     sample_rate) in float32, then used as a float scaling. The former
 *     formula (0.83f ** (hop/64), ignoring sample_rate) was correct only by
 *     coincidence at sr=16000 -- at 48 kHz it used an exponent 3x too large,
 *     decaying the reverb tail roughly 3x too fast in wall-clock time.
 *     Mirrors residual_echo_estimator.py's _reverb_decay(), fixed there
 *     first (see its docstring for the full derivation).
 *   Built with -ffp-contract=off so the per-op f32 roundings are not fused.
 *
 * The estimator OWNS a ReverbModel and a ReverbFrequencyResponse. The caller
 * must drive update_reverb_models() (freq-resp refresh) BEFORE estimate() each
 * hop, exactly as the orchestrator does.
 */
#ifndef RESIDUAL_ECHO_ESTIMATOR_H
#define RESIDUAL_ECHO_ESTIMATOR_H

#include "reverb_model.h"
#include "reverb_frequency_response.h"

#define REE_DELAY_BUF_SIZE 16   /* python _DELAY_BUF_SIZE */

/* EchoModelConfig subset (frozen in Python). */
typedef struct {
    float  min_noise_floor_power;   /* fft-density-scaled at init (6553600 @ fft512) */
    float  noise_gate_power;        /* legacy 27509562 (only used when !use_aec3_gate) */
    float  noise_gate_slope;        /* 0.3 */
    float  stationary_gate_slope;   /* 10.0 */
    int    model_reverb_in_nonlinear_mode; /* True */
} ReeEchoModelConfig;

typedef struct {
    int    n_bins;

    /* config */
    ReeEchoModelConfig echo_model;
    float  default_gain_early;      /* 1.0 */
    float  default_gain_late;       /* 1.0 */
    float  tm_gain_early;           /* 0.01 */
    float  tm_gain_late;            /* 0.01 */
    int    erle_onset_comp_in_dominant; /* False */
    float  reverb_decay;            /* 0.83 */
    float  reverb_mild_decay_scale; /* 1.0 */
    int    reverb_enabled;          /* True */
    int    hop_size;                /* 160 */
    int    sample_rate;             /* live rate; drives noise_floor_growth_per_hop
                                      * and the reverb-decay wall-clock rescale below */
    float  noise_floor_growth_per_hop; /* live-computed; see ree_init() */
    float  reverb_tail_strength;    /* 1.0 */
    int    use_aec3_residual_noise_gate; /* True */
    int    use_stationarity_properties;  /* True in production — when set, the
                                          * nonlinear-path residual noise gate
                                          * (cc:121-129) is SKIPPED (mirrors
                                          * Python `not _use_stationarity_properties`). */
    int    use_aec3_echo_gen_window;     /* True */
    int    nl_r2_enabled;           /* True */
    float  nl_r2_alpha;             /* 0.1 */
    float  nl_norm_power;           /* 1.07e7 */
    float  residual_noise_gate_power; /* per_bin_psd_threshold(27509.42,hop) */
    int    noise_floor_hold_hops;   /* 50 */
    int    render_pre_window_size;  /* 1 */
    int    render_post_window_size; /* 1 */

    /* min-statistics render noise floor state */
    float *x2_noise_floor;          /* owned; n_bins f32 */
    int   *x2_noise_floor_counter;  /* owned; n_bins int32 */

    /* reverb sub-models (storage owned by this struct) */
    ReverbModel              reverb_model;
    ReverbFrequencyResponse  reverb_freq_resp;
    int                      have_reverb_freq_resp; /* use_freq_response */

    /* legacy ring render history (only used when !use_aec3_echo_gen_window) */
    float *render_history;          /* (render_history_size × n_bins) f32 */
    int    render_history_size;
    int    render_history_idx;
    int    render_history_initialised;

    /* delay-centered render deque (index 0 = current after appendleft).
     * Stored as a ring of frames; `delay_buf_count` = current fill (<=16). */
    float *delay_render_buf;        /* (REE_DELAY_BUF_SIZE × n_bins) f32 */
    int    delay_buf_count;
    int    delay_buf_head;          /* index of element 0 (most recent) */

    /* reverb render history deque (index 0 = current after appendleft). */
    float *reverb_render_history;   /* (REE_DELAY_BUF_SIZE × n_bins) f32 */
    int    reverb_buf_count;
    int    reverb_buf_head;

    /* diagnostics — last estimate() */
    float *last_r2_direct;          /* n_bins f32 */
    float *last_r2_reverb;          /* n_bins f32 */

    /* per-instance scratch (x2 window walk + fallback scaling); n_bins f32 */
    float *scratch;
} ResidualEchoEstimator;

/* Initialise. Caller supplies all backing storage (no malloc inside).
 * Storage sizes (floats unless noted):
 *   x2_noise_floor:        n_bins
 *   x2_noise_floor_counter:n_bins (int32)
 *   reverb_model_storage:  n_bins
 *   reverb_tail_storage:   n_bins
 *   render_history_storage:(pre+post+1) × n_bins   (pre=use_aec3?1:0, post=1)
 *   delay_render_storage:  REE_DELAY_BUF_SIZE × n_bins
 *   reverb_render_storage: REE_DELAY_BUF_SIZE × n_bins
 *   last_r2_direct_store:  n_bins
 *   last_r2_reverb_store:  n_bins
 * reverb_freq_resp uses reverb_tail_storage; smoothing_base/use_conservative
 * supplied by the caller (orchestrator computes them).
 */
void ree_init(ResidualEchoEstimator *r,
              int n_bins,
              int hop_size,
              int sample_rate,
              const ReeEchoModelConfig *echo_model,
              float default_gain, float tm_gain,
              int erle_onset_comp_in_dominant,
              float reverb_decay, float reverb_mild_decay_scale,
              int reverb_enabled, float reverb_tail_strength,
              int use_aec3_residual_noise_gate,
              int use_stationarity_properties,
              int use_aec3_echo_gen_window,
              int nl_r2_enabled, float nl_r2_alpha, float nl_norm_power,
              float residual_noise_gate_power,
              int noise_floor_hold_hops,
              int use_freq_response, int reverb_use_conservative,
              float reverb_smoothing_base,
              float *x2_noise_floor, int *x2_noise_floor_counter,
              float *reverb_model_storage, float *reverb_tail_storage,
              float *render_history_storage,
              float *delay_render_storage, float *reverb_render_storage,
              float *last_r2_direct_store, float *last_r2_reverb_store,
              float *scratch_store);

void ree_reset(ResidualEchoEstimator *r);

/* Public wrapper for _reverb_decay(dominant_nearend) (static config path):
 * decay = reverb_decay (× mild_decay_scale if dominant_nearend), then
 * pow(decay, hop/64) when hop != 64. Returns 0 when reverb disabled. Used by
 * the orchestrator's avg-render-reverb step (decay_steady = dominant=False). */
float ree_reverb_decay_value(const ResidualEchoEstimator *r,
                             int dominant_nearend);

/* update_reverb_models — refreshes the bound ReverbFrequencyResponse.
 * (The adaptive ReverbDecayEstimator is not bound in the production path, so
 *  only the freq-resp branch is ported.)
 *   frequency_response : flat row-major (n_partitions × n_bins) float32
 *   filter_quality_is_none : 1 → Python None (skip update)
 */
void ree_update_reverb_models(ResidualEchoEstimator *r,
                              const float *frequency_response,
                              int n_partitions,
                              int filter_delay_blocks,
                              float filter_quality,
                              int filter_quality_is_none,
                              int stationary_block);

/* estimate — produces r2[n_bins] and r2_unbounded[n_bins] (both f32).
 *   render_psd / capture_psd / s2_linear : float32[n_bins]
 *   usable / saturated / transparent_mode : booleans (0/1)
 *   erle / erle_unbounded : float32[n_bins] (only read on the linear,
 *                           non-saturated path)
 * force_nonlinear_path forces usable=0 (matches Python).
 */
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

#endif /* RESIDUAL_ECHO_ESTIMATOR_H */
