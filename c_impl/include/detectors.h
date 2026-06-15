/* detectors.h — Wave 1.5/1.6/1.7: 1:1 ports of
 *   RenderActivityDetector       (aec.py 2299-2358)
 *   FilterConvergenceAnalyzer    (aec.py 2369-2435)
 *   DoubleTalkAnalyzer           (aec.py 2437-2505)
 *
 * All three operate purely on Python `float` (fp64) scalars + bool, so C
 * mirrors with `double` and `int` (0/1).
 */
#ifndef AEC_DETECTORS_H
#define AEC_DETECTORS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── RenderActivityDetector ────────────────────────────────── */
typedef struct RenderActivity {
    double env_mean;
    double env_var;
    int    active_prev;
    int    is_stationary;
} RenderActivity;

typedef struct RenderActivityResult {
    double far_pwr;
    int    is_active;
    int    is_stationary;
    int    warmup_active;
} RenderActivityResult;

void render_activity_init(RenderActivity* r);
void render_activity_reset(RenderActivity* r);
RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n);

/* ── FilterConvergenceAnalyzer ─────────────────────────────── */
typedef struct FilterConvergence {
    int    converged;
    int    once_converged;
    int    conv_counter;
    double divergence;
    double erle_smooth_lin;  /* EMA of ERLE linear (alpha=0.7) for convergence check */
} FilterConvergence;

void filter_convergence_init(FilterConvergence* c);
void filter_convergence_reset(FilterConvergence* c);
void filter_convergence_mark_diverged(FilterConvergence* c);
void filter_convergence_update_divergence(FilterConvergence* c,
                                             double near_power,
                                             double raw_error_power);
/* Returns 1 on the transition frame. */
int  filter_convergence_update_convergence(FilterConvergence* c,
                                              double near_power,
                                              double raw_error_power,
                                              int far_active, int warmup_done);

/* ── DoubleTalkAnalyzer ────────────────────────────────────── */
typedef struct DoubleTalk {
    double dt_from_energy;
    double dt_from_shadow;
    double shadow_advantage;
    /* Config slice */
    double shadow_dtd_offset;            /* default 1.5 */
    double shadow_dtd_advantage_scale;   /* default 3.0 */
} DoubleTalk;

void doubletalk_init(DoubleTalk* d, double offset, double advantage_scale);
void doubletalk_reset(DoubleTalk* d);
void doubletalk_update_shadow_dt(DoubleTalk* d,
                                    int shadow_frame_count, int far_excited,
                                    double main_err_smooth,
                                    double shadow_err_smooth);
void doubletalk_update_energy_dt(DoubleTalk* d,
                                    int far_active, double far_pwr,
                                    double mic_pwr, double erl_estimate);

/* ── FilterPlateauDetector (v3.10.0 + v3.10.3 F4/H3) ─────── */
/* Detects filter stuck below ERLE convergence threshold for sustained
 * time despite far activity. Fires once per session (capped by max_attempts).
 * Mirrors aec.py:2710-2848. */
typedef struct FilterPlateauDetector {
    /* Config */
    int    grace_frames;        /* 400 */
    double erle_max_db;         /* 6.0 */
    double far_active_ratio;    /* 0.5 */
    double dt_signal_ratio;     /* 0.10 */
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
                              double erle_windowed_db,
                              int    once_converged);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DETECTORS_H */
