/**
 * dt_analyzer.c - port of Python DoubleTalkAnalyzer (aec.py 2437-2504).
 */

#include "dt_analyzer.h"
#include <stdlib.h>

#define SHADOW_FRAME_GATE 50
#define ERL_CEILING_FLOOR 0.01f
#define SAFETY_MARGIN     2.0f
#define DTE_RISE_OLD      0.3f
#define DTE_RISE_NEW      0.7f
#define DTE_DECAY_OLD     0.9f
#define DTE_DECAY_NEW     0.1f
#define DTS_OLD           0.7f
#define DTS_NEW           0.3f
#define DTS_INACTIVE_DECAY 0.95f


struct DoubleTalkAnalyzer {
    float shadow_dtd_offset;
    float shadow_dtd_advantage_scale;
    float dt_from_energy;
    float dt_from_shadow;
    float shadow_advantage;
};


DoubleTalkAnalyzer* dta_create(float shadow_dtd_offset,
                                float shadow_dtd_advantage_scale) {
    DoubleTalkAnalyzer* d = (DoubleTalkAnalyzer*)calloc(1, sizeof(DoubleTalkAnalyzer));
    if (!d) return NULL;
    d->shadow_dtd_offset = shadow_dtd_offset;
    d->shadow_dtd_advantage_scale = shadow_dtd_advantage_scale;
    dta_reset(d);
    return d;
}


void dta_destroy(DoubleTalkAnalyzer* d) { free(d); }


void dta_reset(DoubleTalkAnalyzer* d) {
    if (!d) return;
    d->dt_from_energy = 0.0f;
    d->dt_from_shadow = 0.0f;
    d->shadow_advantage = 1.0f;
}


void dta_update_shadow_dt(DoubleTalkAnalyzer* d,
                          int shadow_frame_count, int far_excited,
                          float main_err_smooth, float shadow_err_smooth) {
    if (!d) return;
    if (shadow_frame_count >= SHADOW_FRAME_GATE && far_excited) {
        d->shadow_advantage = main_err_smooth / (shadow_err_smooth + 1e-10f);
        float raw = (d->shadow_advantage - d->shadow_dtd_offset)
                    / d->shadow_dtd_advantage_scale;
        if (raw < 0.0f) raw = 0.0f;
        if (raw > 1.0f) raw = 1.0f;
        d->dt_from_shadow = DTS_OLD * d->dt_from_shadow + DTS_NEW * raw;
    } else {
        d->dt_from_shadow *= DTS_INACTIVE_DECAY;
    }
}


void dta_update_energy_dt(DoubleTalkAnalyzer* d,
                          int far_active, float far_pwr,
                          float mic_pwr, float erl_estimate) {
    if (!d) return;
    float inst;
    if (far_active && far_pwr > 1e-4f) {
        float erl_ceil = (erl_estimate < ERL_CEILING_FLOOR) ? ERL_CEILING_FLOOR : erl_estimate;
        erl_ceil = 1.0f / erl_ceil;
        float max_echo = far_pwr * erl_ceil * SAFETY_MARGIN;
        inst = (mic_pwr - max_echo) / mic_pwr;
        if (inst < 0.0f) inst = 0.0f;
    } else {
        inst = 0.0f;
    }
    if (inst > d->dt_from_energy) {
        d->dt_from_energy = DTE_RISE_OLD * d->dt_from_energy + DTE_RISE_NEW * inst;
    } else {
        d->dt_from_energy = DTE_DECAY_OLD * d->dt_from_energy + DTE_DECAY_NEW * inst;
    }
}


float dta_get_dt_from_energy(const DoubleTalkAnalyzer* d) { return d ? d->dt_from_energy : 0.0f; }
float dta_get_dt_from_shadow(const DoubleTalkAnalyzer* d) { return d ? d->dt_from_shadow : 0.0f; }
float dta_get_shadow_advantage(const DoubleTalkAnalyzer* d) { return d ? d->shadow_advantage : 1.0f; }
