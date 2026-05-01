/**
 * shadow_copy.c - port of Python ShadowCopyController energy gate
 * (aec.py 2852-3005). Other gate modes deferred.
 */

#include "shadow_copy.h"
#include <stdlib.h>

#define BASELINE_INIT   1e-6f
#define HYS_STREAK_MIN  10
#define DT_GATE_THR     0.3f


struct ShadowCopyController {
    float copy_err_baseline;
    int   copy_counter;
    int   streak;
    int   main_paused;
    int   pause_resume;

    /* Config */
    float shadow_copy_threshold;
    int   shadow_copy_hysteresis;
    int   epc_hangover;
};


ShadowCopyController* sc_create(float shadow_copy_threshold,
                                int   shadow_copy_hysteresis,
                                int   epc_hangover) {
    ShadowCopyController* sc = (ShadowCopyController*)calloc(1, sizeof(ShadowCopyController));
    if (!sc) return NULL;
    sc->shadow_copy_threshold = shadow_copy_threshold;
    sc->shadow_copy_hysteresis = shadow_copy_hysteresis;
    sc->epc_hangover = epc_hangover;
    sc_reset(sc);
    return sc;
}


void sc_destroy(ShadowCopyController* sc) { free(sc); }


void sc_reset(ShadowCopyController* sc) {
    if (!sc) return;
    sc->copy_err_baseline = BASELINE_INIT;
    sc->copy_counter = 0;
    sc->streak = 0;
    sc->main_paused = 0;
    sc->pause_resume = 0;
}


ShadowCopyDecision sc_update(ShadowCopyController* sc,
                              int   shadow_frame_count,
                              float far_pwr,
                              float main_err_smooth,
                              float shadow_err_smooth,
                              int   epc_active,
                              float saturation_level,
                              float dt_from_energy) {
    ShadowCopyDecision dec = {0, 0, 0};
    if (!sc) return dec;
    if (shadow_frame_count < 50) return dec;

    float threshold = sc->shadow_copy_threshold;
    int   far_active = (far_pwr > 1e-4f);

    float err_sum = main_err_smooth + shadow_err_smooth + 1e-10f;
    float diff = main_err_smooth - shadow_err_smooth;
    if (diff < 0.0f) diff = -diff;
    float err_balance = diff / err_sum;
    int is_stable_fs = far_active && (err_balance < 0.3f) && !epc_active;

    if (is_stable_fs) {
        float best_err = (main_err_smooth < shadow_err_smooth)
                            ? main_err_smooth : shadow_err_smooth;
        sc->copy_err_baseline = 0.995f * sc->copy_err_baseline + 0.005f * best_err;
    }

    int error_is_normal = (main_err_smooth < sc->copy_err_baseline * 4.0f + 1e-10f);
    int not_saturating  = (saturation_level < 0.3f);
    int dt_safe = (dt_from_energy < DT_GATE_THR);
    int copy_allowed = far_active && error_is_normal
                       && !epc_active && not_saturating && dt_safe;

    if (copy_allowed) {
        if (shadow_err_smooth < main_err_smooth * threshold) {
            sc->copy_counter++;
            sc->streak++;
        } else {
            sc->copy_counter = 0;
            sc->streak = 0;
        }
        if (sc->copy_counter >= sc->shadow_copy_hysteresis
                && sc->streak >= HYS_STREAK_MIN) {
            sc->copy_counter = 0;
            sc->streak = 0;
            sc->main_paused = 1;
            sc->pause_resume = sc->epc_hangover;
            dec.boost_q = 1;
        }
        if (sc->main_paused) {
            if (sc->pause_resume > 0) sc->pause_resume--;
            else sc->main_paused = 0;
        }
        if (main_err_smooth < shadow_err_smooth * threshold && error_is_normal) {
            dec.reverse_copy = 1;
        }
    } else {
        sc->copy_counter = 0;
        sc->streak = 0;
        sc->main_paused = 0;
    }

    dec.pause_main = sc->main_paused;
    return dec;
}


int   sc_main_paused(const ShadowCopyController* sc)       { return sc ? sc->main_paused : 0; }
float sc_copy_err_baseline(const ShadowCopyController* sc) { return sc ? sc->copy_err_baseline : BASELINE_INIT; }
int   sc_copy_counter(const ShadowCopyController* sc)      { return sc ? sc->copy_counter : 0; }
