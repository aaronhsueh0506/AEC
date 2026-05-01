/**
 * epc_detector.c - port of Python EchoPathChangeDetector (aec.py 2726-2836).
 */

#include "epc_detector.h"
#include <stdlib.h>
#include <math.h>

#define EPV_FAST_TC  0.98f
#define EPV_SLOW_TC  0.999f
#define EPV_LOW      0.25f
#define EPV_HIGH     4.0f


struct EpcDetector {
    int   active;
    int   hangover;
    float epv_gain_fast;
    float epv_gain_slow;
    float prev_total_err;

    /* Config */
    int   epc_hangover_frames;
    float epc_total_rise;
    float epc_delta_threshold;
};


EpcDetector* epc_create(int epc_hangover, float epc_total_rise,
                        float epc_delta_threshold) {
    EpcDetector* d = (EpcDetector*)calloc(1, sizeof(EpcDetector));
    if (!d) return NULL;
    d->epc_hangover_frames = epc_hangover;
    d->epc_total_rise = epc_total_rise;
    d->epc_delta_threshold = epc_delta_threshold;
    epc_reset(d);
    return d;
}


void epc_destroy(EpcDetector* d) { free(d); }


void epc_reset(EpcDetector* d) {
    if (!d) return;
    d->active = 0;
    d->hangover = 0;
    d->epv_gain_fast = 0.0f;
    d->epv_gain_slow = 0.0f;
    d->prev_total_err = 0.0f;
}


EpcEvent epc_force_delay(EpcDetector* d) {
    EpcEvent ev = {0, EPC_SOURCE_NONE};
    if (!d) return ev;
    d->active = 1;
    d->hangover = d->epc_hangover_frames;
    ev.fired = 1; ev.source = EPC_SOURCE_DELAY;
    return ev;
}


EpcEvent epc_update_epv(EpcDetector* d, float far_pwr_global,
                        int filter_converged, int main_paused) {
    EpcEvent ev = {0, EPC_SOURCE_NONE};
    if (!d) return ev;
    if (far_pwr_global <= 1e-6f) return ev;

    if (d->epv_gain_fast < 1e-12f) {
        d->epv_gain_fast = d->epv_gain_slow = far_pwr_global;
    } else {
        d->epv_gain_fast = EPV_FAST_TC * d->epv_gain_fast
                         + (1.0f - EPV_FAST_TC) * far_pwr_global;
        d->epv_gain_slow = EPV_SLOW_TC * d->epv_gain_slow
                         + (1.0f - EPV_SLOW_TC) * far_pwr_global;
    }
    if (filter_converged && !d->active && !main_paused
            && d->epv_gain_slow > 1e-10f) {
        float ratio = d->epv_gain_fast / (d->epv_gain_slow + 1e-10f);
        if (ratio < EPV_LOW || ratio > EPV_HIGH) {
            d->active = 1;
            d->hangover = d->epc_hangover_frames;
            ev.fired = 1; ev.source = EPC_SOURCE_EPV;
            return ev;
        }
    }
    return ev;
}


EpcEvent epc_update_shadow_rise(EpcDetector* d,
                                float main_err_smooth,
                                float shadow_err_smooth,
                                int is_stationary) {
    EpcEvent ev = {0, EPC_SOURCE_NONE};
    if (!d) return ev;
    float total_err = main_err_smooth + shadow_err_smooth;
    float delta_ratio;
    if (total_err > 1e-10f) {
        float diff = main_err_smooth - shadow_err_smooth;
        if (diff < 0.0f) diff = -diff;
        delta_ratio = diff / total_err;
    } else {
        delta_ratio = 0.0f;
    }
    int errors_rising = (total_err > d->prev_total_err * d->epc_total_rise)
                        && (d->prev_total_err > 1e-10f);
    int is_echo_change = errors_rising
                         && (delta_ratio < d->epc_delta_threshold);
    if (is_echo_change && is_stationary) is_echo_change = 0;
    d->prev_total_err = total_err;
    if (is_echo_change) {
        d->active = 1;
        d->hangover = d->epc_hangover_frames;
        ev.fired = 1; ev.source = EPC_SOURCE_SHADOW_RISE;
    }
    return ev;
}


void epc_tick_hangover(EpcDetector* d) {
    if (!d) return;
    if (d->hangover > 0) {
        d->hangover--;
        d->active = 1;
    } else {
        d->active = 0;
    }
}


int   epc_is_active(const EpcDetector* d)    { return d ? d->active : 0; }
int   epc_get_hangover(const EpcDetector* d) { return d ? d->hangover : 0; }
float epc_get_gain_fast(const EpcDetector* d){ return d ? d->epv_gain_fast : 0.0f; }
float epc_get_gain_slow(const EpcDetector* d){ return d ? d->epv_gain_slow : 0.0f; }
