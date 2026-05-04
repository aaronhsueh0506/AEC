/* epc_shadow.c — port of aec.py EchoPathChangeDetector +
 * ShadowCopyController.
 *
 * All scalar fp64 (matches Python float) + ints.
 */
#include "epc_shadow.h"
#include <math.h>

/* ── EchoPathChangeDetector ───────────────────────────────────── */

static const double EPV_FAST_TC = 0.98;
static const double EPV_SLOW_TC = 0.999;
static const double EPV_LOW     = 0.25;
static const double EPV_HIGH    = 4.0;

void epc_init(EpcDetector* e, int hangover, double total_rise,
                 double delta_threshold) {
    e->active        = 0;
    e->hangover      = 0;
    e->epv_gain_fast = 0.0;
    e->epv_gain_slow = 0.0;
    e->prev_total_err = 0.0;
    e->epc_hangover  = hangover;
    e->epc_total_rise = total_rise;
    e->epc_delta_threshold = delta_threshold;
}
void epc_reset(EpcDetector* e) {
    e->active = 0; e->hangover = 0;
    e->epv_gain_fast = 0.0; e->epv_gain_slow = 0.0;
    e->prev_total_err = 0.0;
}

EpcEvent epc_force_delay(EpcDetector* e) {
    e->active   = 1;
    e->hangover = e->epc_hangover;
    EpcEvent ev = { .fired = 1, .source = EPC_DELAY };
    return ev;
}

EpcEvent epc_update_epv(EpcDetector* e, double far_pwr_global,
                             int filter_converged, int main_paused) {
    EpcEvent ev = { .fired = 0, .source = EPC_NONE };
    if (far_pwr_global <= 1e-6) return ev;
    if (e->epv_gain_fast < 1e-12) {
        e->epv_gain_fast = e->epv_gain_slow = far_pwr_global;
    } else {
        e->epv_gain_fast = EPV_FAST_TC * e->epv_gain_fast
                         + (1.0 - EPV_FAST_TC) * far_pwr_global;
        e->epv_gain_slow = EPV_SLOW_TC * e->epv_gain_slow
                         + (1.0 - EPV_SLOW_TC) * far_pwr_global;
    }
    if (filter_converged && !e->active && !main_paused
        && e->epv_gain_slow > 1e-10) {
        double ratio = e->epv_gain_fast / (e->epv_gain_slow + 1e-10);
        if (ratio < EPV_LOW || ratio > EPV_HIGH) {
            e->active   = 1;
            e->hangover = e->epc_hangover;
            ev.fired = 1; ev.source = EPC_EPV;
            return ev;
        }
    }
    return ev;
}

EpcEvent epc_update_shadow_rise(EpcDetector* e,
                                     double main_err_smooth,
                                     double shadow_err_smooth,
                                     int is_stationary) {
    EpcEvent ev = { .fired = 0, .source = EPC_NONE };
    double total_err = main_err_smooth + shadow_err_smooth;
    double delta_ratio;
    if (total_err > 1e-10) {
        delta_ratio = fabs(main_err_smooth - shadow_err_smooth) / total_err;
    } else {
        delta_ratio = 0.0;
    }
    int errors_rising = (total_err > e->prev_total_err * e->epc_total_rise)
                     && (e->prev_total_err > 1e-10);
    int is_echo_change = errors_rising
                      && (delta_ratio < e->epc_delta_threshold);
    if (is_echo_change && is_stationary) is_echo_change = 0;
    e->prev_total_err = total_err;
    if (is_echo_change) {
        e->active   = 1;
        e->hangover = e->epc_hangover;
        ev.fired = 1; ev.source = EPC_SHADOW_RISE;
    }
    return ev;
}

void epc_tick_hangover(EpcDetector* e) {
    if (e->hangover > 0) {
        e->hangover--;
        e->active = 1;
    } else {
        e->active = 0;
    }
}

/* ── ShadowCopyController ─────────────────────────────────────── */

static const double SC_BASELINE_INIT       = 1e-6;
static const int    SC_HYS_STREAK_MIN      = 10;
static const int    SC_AEC3_STREAK_FRAMES  = 5;

void shadow_copy_init(ShadowCopy* s, ShadowCopyGateMode mode,
                         double threshold, int hysteresis, int epc_hangover) {
    s->gate_mode = mode;
    s->copy_err_baseline = SC_BASELINE_INIT;
    s->copy_counter = 0;
    s->streak       = 0;
    s->main_paused  = 0;
    s->pause_resume = 0;
    s->shadow_copy_threshold  = threshold;
    s->shadow_copy_hysteresis = hysteresis;
    s->epc_hangover           = epc_hangover;
}
void shadow_copy_reset(ShadowCopy* s) {
    s->copy_err_baseline = SC_BASELINE_INIT;
    s->copy_counter = 0;
    s->streak       = 0;
    s->main_paused  = 0;
    s->pause_resume = 0;
}

static int dt_safe(const ShadowCopy* s,
                   double dt_from_energy, double dt_from_coh,
                   int delay_reliable) {
    switch (s->gate_mode) {
        case SC_GATE_ENERGY:    return dt_from_energy < 0.3;
        case SC_GATE_COHERENCE: return dt_from_coh < 0.4;
        case SC_GATE_COH_DELAY: return (dt_from_coh < 0.4) && delay_reliable;
        case SC_GATE_STREAK:    return 1;
    }
    return dt_from_energy < 0.3;
}

ShadowCopyDecision shadow_copy_update(
    ShadowCopy* s,
    int    shadow_frame_count,
    double far_pwr,
    double main_err_smooth,
    double shadow_err_smooth,
    int    epc_active,
    double saturation_level,
    double dt_from_energy,
    double dt_from_coherence,
    int    delay_reliable) {

    ShadowCopyDecision d = {0,0,0};
    if (shadow_frame_count < 50) return d;

    double threshold = s->shadow_copy_threshold;
    int    far_active = far_pwr > 1e-4;

    double err_sum     = main_err_smooth + shadow_err_smooth + 1e-10;
    double err_balance = fabs(main_err_smooth - shadow_err_smooth) / err_sum;
    int    is_stable_fs = far_active && (err_balance < 0.3) && !epc_active;
    if (is_stable_fs) {
        double best_err = (main_err_smooth < shadow_err_smooth) ? main_err_smooth : shadow_err_smooth;
        s->copy_err_baseline = 0.995 * s->copy_err_baseline + 0.005 * best_err;
    }
    int error_is_normal = main_err_smooth < (s->copy_err_baseline * 4.0 + 1e-10);
    int not_saturating  = saturation_level < 0.3;
    int dts             = dt_safe(s, dt_from_energy, dt_from_coherence, delay_reliable);
    int copy_allowed = far_active && error_is_normal
                    && !epc_active && not_saturating && dts;

    if (s->gate_mode == SC_GATE_STREAK) {
        if (copy_allowed && shadow_err_smooth < main_err_smooth * threshold) {
            s->streak++;
            if (s->streak >= SC_AEC3_STREAK_FRAMES) {
                s->streak = 0;
                s->main_paused = 1;
                s->pause_resume = s->epc_hangover;
                d.boost_q = 1;
            }
        } else {
            s->streak = 0;
        }
        if (s->main_paused) {
            if (s->pause_resume > 0) s->pause_resume--;
            else s->main_paused = 0;
        }
        if (copy_allowed
            && main_err_smooth < shadow_err_smooth * threshold
            && error_is_normal) {
            d.reverse_copy = 1;
        }
        d.pause_main = s->main_paused;
        return d;
    }

    if (copy_allowed) {
        if (shadow_err_smooth < main_err_smooth * threshold) {
            s->copy_counter++;
            s->streak++;
        } else {
            s->copy_counter = 0;
            s->streak       = 0;
        }
        if (s->copy_counter >= s->shadow_copy_hysteresis
            && s->streak >= SC_HYS_STREAK_MIN) {
            s->copy_counter = 0;
            s->streak       = 0;
            s->main_paused  = 1;
            s->pause_resume = s->epc_hangover;
            d.boost_q = 1;
        }
        if (s->main_paused) {
            if (s->pause_resume > 0) s->pause_resume--;
            else s->main_paused = 0;
        }
        if (main_err_smooth < shadow_err_smooth * threshold && error_is_normal) {
            d.reverse_copy = 1;
        }
    } else {
        s->copy_counter = 0;
        s->streak       = 0;
        s->main_paused  = 0;
    }
    d.pause_main = s->main_paused;
    return d;
}
