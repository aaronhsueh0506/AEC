/* Render-activity, filter-convergence and double-talk detectors (float32).
 * Python reference: modules/detectors.py. */
#ifndef AEC_DETECTORS_H
#define AEC_DETECTORS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── RenderActivityDetector ────────────────────────────────── */
typedef struct RenderActivity {
    float  env_mean;
    float  env_var;
    int    active_prev;
    int    is_stationary;
    /* render_activity_update() scratch, caller-owned, length >= ceil(n/8) for the n passed to
     * render_activity_update (the instance hop_size). */
    float* pairwise_scratch;
    int    pairwise_scratch_len;
    /* Retimed from the authored 10 ms-hop value in render_activity_init. */
    float  alpha_cv;
} RenderActivity;

typedef struct RenderActivityResult {
    float  far_pwr;
    int    is_active;
    int    warmup_active;
} RenderActivityResult;

/* pairwise_scratch/pairwise_scratch_len: caller-owned scratch for the 8-wide pairwise-sum
 * reduction, length >= ceil(hop_size/8).
 * hop_size/sample_rate retime the wall-clock constants authored for hop=160 @ 16 kHz. */
void render_activity_init(RenderActivity* r, float* pairwise_scratch,
                           int pairwise_scratch_len,
                           int hop_size, int sample_rate);
void render_activity_reset(RenderActivity* r);
RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n);

/* ── FilterConvergenceAnalyzer ─────────────────────────────── */
typedef struct FilterConvergence {
    int    converged;
    int    once_converged;
    int    conv_counter;
    float  divergence;
    /* Retimed in filter_convergence_init; FC_CONV_FRAMES is an evidence count and is not retimed.
     * */
    float  div_alpha;
    float  div_alpha_c;   /* = 1.0f - div_alpha, cached at init */
    float  div_decay;
} FilterConvergence;

void filter_convergence_init(FilterConvergence* c, int hop_size, int sample_rate);
void filter_convergence_reset(FilterConvergence* c);
void filter_convergence_mark_diverged(FilterConvergence* c);
void filter_convergence_update_divergence(FilterConvergence* c,
                                             float near_power,
                                             float raw_error_power);
/* Returns 1 on the transition frame. */
int  filter_convergence_update_convergence(FilterConvergence* c,
                                              float near_power,
                                              float raw_error_power,
                                              int far_active, int warmup_done);

/* ── DoubleTalkAnalyzer ────────────────────────────────────── */
typedef struct DoubleTalk {
    float  dt_from_energy;
    float  dt_from_shadow;
    float  shadow_advantage;
    /* Config slice */
    float  shadow_dtd_offset;            /* default 1.5 */
    float  shadow_dtd_advantage_scale;   /* default 3.0 */
    /* Retimed in doubletalk_init. */
    int    shadow_frame_gate;
    float  dte_rise_old,  dte_rise_new;
    float  dte_decay_old, dte_decay_new;
    float  dts_old,       dts_new;
    float  dts_inactive_decay;
} DoubleTalk;

void doubletalk_init(DoubleTalk* d, float offset, float advantage_scale,
                        int hop_size, int sample_rate);
void doubletalk_reset(DoubleTalk* d);
void doubletalk_update_shadow_dt(DoubleTalk* d,
                                    int shadow_frame_count, int far_excited,
                                    float main_err_smooth,
                                    float shadow_err_smooth);
void doubletalk_update_energy_dt(DoubleTalk* d,
                                    int far_active, float far_pwr,
                                    float mic_pwr, float erl_estimate);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DETECTORS_H */
