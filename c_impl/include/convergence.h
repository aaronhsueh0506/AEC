/**
 * convergence.h - Filter convergence + divergence indicator state machine.
 * Port of Python aec.py FilterConvergenceAnalyzer (lines 2369-2434).
 */

#ifndef CONVERGENCE_H
#define CONVERGENCE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FilterConvergenceAnalyzer FilterConvergenceAnalyzer;

FilterConvergenceAnalyzer* fc_create(void);
void fc_destroy(FilterConvergenceAnalyzer* fc);
void fc_reset(FilterConvergenceAnalyzer* fc);

/**
 * Mark filter diverged (EPC / delay shift).
 * Drops _converged + restarts _conv_counter; preserves _once_converged.
 */
void fc_mark_diverged(FilterConvergenceAnalyzer* fc);

/**
 * Update divergence indicator EMA. Only fires post-convergence.
 */
void fc_update_divergence(FilterConvergenceAnalyzer* fc,
                          float near_power, float raw_error_power);

/**
 * Check convergence (10 consecutive far-active frames with inst-ERLE > 5dB).
 * Returns 1 on the transition frame, 0 otherwise.
 */
int fc_update_convergence(FilterConvergenceAnalyzer* fc,
                          float near_power, float raw_error_power,
                          int far_active, int warmup_done);

int   fc_is_converged(const FilterConvergenceAnalyzer* fc);
int   fc_is_once_converged(const FilterConvergenceAnalyzer* fc);
float fc_get_divergence(const FilterConvergenceAnalyzer* fc);

#ifdef __cplusplus
}
#endif
#endif /* CONVERGENCE_H */
