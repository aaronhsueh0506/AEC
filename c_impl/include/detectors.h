/* Render-activity, filter-convergence and double-talk detectors. */
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
    /* De-stacked render_activity_update() scratch (formerly a fixed
     * `float blocks[1024]` local -- a stack-overflow hazard once the
     * runtime hop can require more than 1024/PAIR_BLOCK=8 blocks, e.g. a
     * higher-sample-rate build). Owned by caller; length >=
     * ceil(n/8) for the `n` passed to render_activity_update (n is always
     * the instance's hop_size at every call site). */
    float* pairwise_scratch;
    int    pairwise_scratch_len;
    /* Retimed from the authored 10 ms-hop value in render_activity_init. */
    float  alpha_cv;
} RenderActivity;

typedef struct RenderActivityResult {
    float  far_pwr;
    int    is_active;
    int    is_stationary;
    int    warmup_active;
} RenderActivityResult;

/* pairwise_scratch/pairwise_scratch_len: caller-owned scratch for the
 * internal 8-wide pairwise-sum reduction (see render_activity_update);
 * length must be >= ceil(hop_size/8). */
/* hop_size/sample_rate retime this detector's wall-clock constants off the
 * authored hop=160 @ 16000 (10 ms) grid. hop alone is not enough: hop=128 is
 * both 8 kHz (16.000 ms) and 16 kHz (8.000 ms). */
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
    /* Retimed in filter_convergence_init. conv_counter's threshold
     * (FC_CONV_FRAMES) is deliberately NOT here: it is an evidence count. */
    float  div_alpha;
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

/* ── FilterPlateauDetector (v3.10.0 + v3.10.3 F4/H3) ─────── */
/* Detects filter stuck below ERLE convergence threshold for sustained
 * time despite far activity. May fire up to max_attempts times per session.
 * Mirrors aec.py:2710-2848. */
typedef struct FilterPlateauDetector {
    /* Config */
    int    grace_frames;        /* 400 */
    float  erle_max_db;         /* 6.0 */
    float  far_active_ratio;    /* 0.5 */
    float  dt_signal_ratio;     /* 0.10 */
    int    max_attempts;        /* 2 */
    /* These counters are used in ratios. Widening them together avoids
     * long-running signed overflow without distorting those ratios. */
    int64_t frame_count;
    int64_t far_active_count;
    int64_t dt_signal_count;
    int    consecutive_match;
    int    attempts;
    int    cooldown_remaining;
    int64_t last_reset_frame;
} FilterPlateauDetector;

#define FILTER_PLATEAU_CONSECUTIVE_REQUIRED 50
#define FILTER_PLATEAU_POST_RESET_GRACE_FRAMES 200

void filter_plateau_init(FilterPlateauDetector* p);
void filter_plateau_reset(FilterPlateauDetector* p);
/* Returns 1 iff a recovery action should fire this frame. */
int  filter_plateau_update(FilterPlateauDetector* p,
                              int    far_active,
                              int    dt_signal_present,
                              float  erle_windowed_db,
                              int    once_converged);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DETECTORS_H */
