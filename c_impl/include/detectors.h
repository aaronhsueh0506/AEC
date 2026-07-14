/* detectors.h — Wave 1.5/1.6/1.7: 1:1 ports of
 *   RenderActivityDetector       (aec.py 2299-2358)
 *   FilterConvergenceAnalyzer    (aec.py 2369-2435)
 *   DoubleTalkAnalyzer           (aec.py 2437-2505)
 *
 * All three operate purely on float32 scalars + bool, so C mirrors with
 * `float` and `int` (0/1). float32-by-design — the original Python fp64
 * bit-exact parity target is retired; the drift this introduces is accepted.
 */
#ifndef AEC_DETECTORS_H
#define AEC_DETECTORS_H

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
void render_activity_init(RenderActivity* r, float* pairwise_scratch,
                           int pairwise_scratch_len);
void render_activity_reset(RenderActivity* r);
RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n);

/* ── FilterConvergenceAnalyzer ─────────────────────────────── */
typedef struct FilterConvergence {
    int    converged;
    int    once_converged;
    int    conv_counter;
    float  divergence;
} FilterConvergence;

void filter_convergence_init(FilterConvergence* c);
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
} DoubleTalk;

void doubletalk_init(DoubleTalk* d, float offset, float advantage_scale);
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
 * time despite far activity. Fires once per session (capped by max_attempts).
 * Mirrors aec.py:2710-2848. */
typedef struct FilterPlateauDetector {
    /* Config */
    int    grace_frames;        /* 400 */
    float  erle_max_db;         /* 6.0 */
    float  far_active_ratio;    /* 0.5 */
    float  dt_signal_ratio;     /* 0.10 */
    int    max_attempts;        /* 2 */
    /* State */
    int    frame_count;
    int    far_active_count;
    int    dt_signal_count;
    int    consecutive_match;
    int    attempts;
    int    cooldown_remaining;
    int    last_reset_frame;
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
