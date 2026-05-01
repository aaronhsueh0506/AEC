/**
 * dt_analyzer.h - DoubleTalkAnalyzer (energy + shadow DT signals)
 * Port of Python aec.py DoubleTalkAnalyzer (lines 2437-2504).
 */

#ifndef DT_ANALYZER_H
#define DT_ANALYZER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DoubleTalkAnalyzer DoubleTalkAnalyzer;

/**
 * Config-derived constants.
 * shadow_dtd_offset: AecConfig.shadow_dtd_offset (default 1.5)
 * shadow_dtd_advantage_scale: AecConfig.shadow_dtd_advantage_scale (default 3.0)
 */
DoubleTalkAnalyzer* dta_create(float shadow_dtd_offset,
                                float shadow_dtd_advantage_scale);
void dta_destroy(DoubleTalkAnalyzer* dta);
void dta_reset(DoubleTalkAnalyzer* dta);

/**
 * Update shadow-advantage DT signal. Runs after shadow filter process.
 * Gated: shadow_frame_count >= 50 AND far_excited.
 */
void dta_update_shadow_dt(DoubleTalkAnalyzer* dta,
                          int shadow_frame_count, int far_excited,
                          float main_err_smooth, float shadow_err_smooth);

/**
 * Update energy DT signal. Runs in RES block.
 * Gated: far_active AND far_pwr > 1e-4.
 */
void dta_update_energy_dt(DoubleTalkAnalyzer* dta,
                          int far_active, float far_pwr,
                          float mic_pwr, float erl_estimate);

float dta_get_dt_from_energy(const DoubleTalkAnalyzer* dta);
float dta_get_dt_from_shadow(const DoubleTalkAnalyzer* dta);
float dta_get_shadow_advantage(const DoubleTalkAnalyzer* dta);

#ifdef __cplusplus
}
#endif
#endif /* DT_ANALYZER_H */
