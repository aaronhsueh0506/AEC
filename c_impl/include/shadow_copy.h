/**
 * shadow_copy.h - ShadowCopyController (legacy energy-gate mode only)
 * Port of Python aec.py ShadowCopyController (lines 2852-3005, energy gate).
 *
 * Phase C1 ablation modes (coherence / coh_delay / streak_only) NOT
 * ported in initial pass — energy gate is v2.8.1 baseline + v3.x default.
 */

#ifndef SHADOW_COPY_H
#define SHADOW_COPY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShadowCopyController ShadowCopyController;

typedef struct {
    int pause_main;
    int boost_q;
    int reverse_copy;
} ShadowCopyDecision;

/**
 * Config from AecConfig:
 *   shadow_copy_threshold     (default 0.7)
 *   shadow_copy_hysteresis    (default 5)
 *   epc_hangover              (default 20, used as pause_resume countdown)
 */
ShadowCopyController* sc_create(float shadow_copy_threshold,
                                int   shadow_copy_hysteresis,
                                int   epc_hangover);
void sc_destroy(ShadowCopyController* sc);
void sc_reset(ShadowCopyController* sc);

/**
 * Per-frame update. Returns ShadowCopyDecision.
 * Inputs match Python ShadowCopyController.update() (energy gate mode).
 */
ShadowCopyDecision sc_update(ShadowCopyController* sc,
                              int   shadow_frame_count,
                              float far_pwr,
                              float main_err_smooth,
                              float shadow_err_smooth,
                              int   epc_active,
                              float saturation_level,
                              float dt_from_energy);

int   sc_main_paused(const ShadowCopyController* sc);
float sc_copy_err_baseline(const ShadowCopyController* sc);
int   sc_copy_counter(const ShadowCopyController* sc);

#ifdef __cplusplus
}
#endif
#endif /* SHADOW_COPY_H */
