/* detectors_v2.h — Wave 1.5/1.6/1.7: 1:1 ports of
 *   RenderActivityDetector       (aec.py 2299-2358)
 *   FilterConvergenceAnalyzer    (aec.py 2369-2435)
 *   DoubleTalkAnalyzer           (aec.py 2437-2505)
 *
 * All three operate purely on Python `float` (fp64) scalars + bool, so C
 * mirrors with `double` and `int` (0/1).
 */
#ifndef AEC_V2_DETECTORS_H
#define AEC_V2_DETECTORS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── RenderActivityDetector ────────────────────────────────── */
typedef struct RenderActivityV2 {
    double env_mean;
    double env_var;
    int    active_prev;
    int    is_stationary;
} RenderActivityV2;

typedef struct RenderActivityResult {
    double far_pwr;
    int    is_active;
    int    is_stationary;
    int    warmup_active;
} RenderActivityResult;

void render_activity_v2_init(RenderActivityV2* r);
void render_activity_v2_reset(RenderActivityV2* r);
RenderActivityResult render_activity_v2_update(RenderActivityV2* r,
                                               const float* far_end, int n);

/* ── FilterConvergenceAnalyzer ─────────────────────────────── */
typedef struct FilterConvergenceV2 {
    int    converged;
    int    once_converged;
    int    conv_counter;
    double divergence;
} FilterConvergenceV2;

void filter_convergence_v2_init(FilterConvergenceV2* c);
void filter_convergence_v2_reset(FilterConvergenceV2* c);
void filter_convergence_v2_mark_diverged(FilterConvergenceV2* c);
void filter_convergence_v2_update_divergence(FilterConvergenceV2* c,
                                             double near_power,
                                             double raw_error_power);
/* Returns 1 on the transition frame. */
int  filter_convergence_v2_update_convergence(FilterConvergenceV2* c,
                                              double near_power,
                                              double raw_error_power,
                                              int far_active, int warmup_done);

/* ── DoubleTalkAnalyzer ────────────────────────────────────── */
typedef struct DoubleTalkV2 {
    double dt_from_energy;
    double dt_from_shadow;
    double shadow_advantage;
    /* Config slice */
    double shadow_dtd_offset;            /* default 1.5 */
    double shadow_dtd_advantage_scale;   /* default 3.0 */
} DoubleTalkV2;

void doubletalk_v2_init(DoubleTalkV2* d, double offset, double advantage_scale);
void doubletalk_v2_reset(DoubleTalkV2* d);
void doubletalk_v2_update_shadow_dt(DoubleTalkV2* d,
                                    int shadow_frame_count, int far_excited,
                                    double main_err_smooth,
                                    double shadow_err_smooth);
void doubletalk_v2_update_energy_dt(DoubleTalkV2* d,
                                    int far_active, double far_pwr,
                                    double mic_pwr, double erl_estimate);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_DETECTORS_H */
