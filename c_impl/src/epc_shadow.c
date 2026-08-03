/* epc_shadow.c — port of aec.py EchoPathChangeDetector +
 * ShadowCopyController.
 *
 * All scalar math float32 (float32-by-design) + ints.
 */
#include "epc_shadow.h"
#include "aec3_scale.h"
#include <math.h>

/* ── EchoPathChangeDetector ───────────────────────────────────── */

/* Legacy-grid EPV time constants (TC ~50 frames / ~1000 frames @ the legacy
 * hop=160/sample_rate=16000 = 10 ms hop, i.e. ~500 ms / ~10 s wall-clock).
 * Project-native (NOT AEC3-sourced), authored before per-rate multi-grid
 * support existed, with zero rate conversion. Both use the RETENTION
 * convention (epc_update_epv(): ``fast <- EPV_FAST_TC*fast + (1-EPV_FAST_TC)
 * *new`` -- the constant multiplies the OLD state directly), so
 * aec3_growth_rehop (a direct power law), NOT
 * aec3_per_block_ema_alpha_to_per_hop's AEC3-new-sample-weight form, is the
 * correct helper. epc_init() retimes these into each EpcDetector instance's
 * epv_fast_tc/epv_slow_tc via aec3_growth_rehop() (2026-08 gap-fix) so the
 * wall-clock envelopes match at every grid; epc_update_epv() reads the
 * retimed per-instance fields, NOT these module constants directly. Must
 * match epc.py's EchoPathChangeDetector.EPV_FAST_TC/EPV_SLOW_TC (Python
 * side). */
static const float EPV_FAST_TC = 0.98f;
static const float EPV_SLOW_TC = 0.999f;
static const float EPV_LOW     = 0.25f;
static const float EPV_HIGH    = 4.0f;

void epc_init(EpcDetector* e, int hangover, float total_rise,
                 float delta_threshold, int hop_size, int sample_rate) {
    e->active        = 0;
    e->hangover      = 0;
    e->epv_gain_fast = 0.0f;
    e->epv_gain_slow = 0.0f;
    e->prev_total_err = 0.0f;
    e->epc_hangover  = hangover;
    e->epc_total_rise = total_rise;
    e->epc_delta_threshold = delta_threshold;
    if (hop_size > 0 && sample_rate > 0) {
        e->epv_fast_tc = aec3_growth_rehop(EPV_FAST_TC, 160, 16000, hop_size, sample_rate);
        e->epv_slow_tc = aec3_growth_rehop(EPV_SLOW_TC, 160, 16000, hop_size, sample_rate);
    } else {
        /* Invalid/unrecognized grid: fall back to the exact pre-fix
         * (unscaled) literals rather than divide-by-zero inside the rehop
         * helper -- mirrors aec.c's aec_legacy10ms_* fallback. */
        e->epv_fast_tc = EPV_FAST_TC;
        e->epv_slow_tc = EPV_SLOW_TC;
    }
}
void epc_reset(EpcDetector* e) {
    e->active = 0; e->hangover = 0;
    e->epv_gain_fast = 0.0f; e->epv_gain_slow = 0.0f;
    e->prev_total_err = 0.0f;
}

EpcEvent epc_force_delay(EpcDetector* e) {
    e->active   = 1;
    e->hangover = e->epc_hangover;
    EpcEvent ev = { .fired = 1, .source = EPC_DELAY };
    return ev;
}

EpcEvent epc_update_epv(EpcDetector* e, float far_pwr_global,
                             int filter_converged, int main_paused) {
    EpcEvent ev = { .fired = 0, .source = EPC_NONE };
    if (far_pwr_global <= 1e-6f) return ev;
    if (e->epv_gain_fast < 1e-12f) {
        e->epv_gain_fast = e->epv_gain_slow = far_pwr_global;
    } else {
        e->epv_gain_fast = e->epv_fast_tc * e->epv_gain_fast
                         + (1.0f - e->epv_fast_tc) * far_pwr_global;
        e->epv_gain_slow = e->epv_slow_tc * e->epv_gain_slow
                         + (1.0f - e->epv_slow_tc) * far_pwr_global;
    }
    if (filter_converged && !e->active && !main_paused
        && e->epv_gain_slow > 1e-10f) {
        float ratio = e->epv_gain_fast / (e->epv_gain_slow + 1e-10f);
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
                                     float main_err_smooth,
                                     float shadow_err_smooth,
                                     int is_stationary) {
    EpcEvent ev = { .fired = 0, .source = EPC_NONE };
    float total_err = main_err_smooth + shadow_err_smooth;
    float delta_ratio;
    if (total_err > 1e-10f) {
        delta_ratio = fabsf(main_err_smooth - shadow_err_smooth) / total_err;
    } else {
        delta_ratio = 0.0f;
    }
    int errors_rising = (total_err > e->prev_total_err * e->epc_total_rise)
                     && (e->prev_total_err > 1e-10f);
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

static const float SC_BASELINE_INIT       = 1e-6f;
static const int   SC_HYS_STREAK_MIN      = 10;
static const int   SC_AEC3_STREAK_FRAMES  = 5;

void shadow_copy_init(ShadowCopy* s, ShadowCopyGateMode mode,
                         float threshold, int hysteresis, int epc_hangover) {
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
                   float dt_from_energy, float dt_from_coh,
                   int delay_reliable) {
    switch (s->gate_mode) {
        case SC_GATE_ENERGY:    return dt_from_energy < 0.3f;
        case SC_GATE_COHERENCE: return dt_from_coh < 0.4f;
        case SC_GATE_COH_DELAY: return (dt_from_coh < 0.4f) && delay_reliable;
        case SC_GATE_STREAK:    return 1;
    }
    return dt_from_energy < 0.3f;
}

ShadowCopyDecision shadow_copy_update(
    ShadowCopy* s,
    int    shadow_frame_count,
    float  far_pwr,
    float  main_err_smooth,
    float  shadow_err_smooth,
    int    epc_active,
    float  saturation_level,
    float  dt_from_energy,
    float  dt_from_coherence,
    int    delay_reliable) {

    ShadowCopyDecision d = {0,0,0};
    if (shadow_frame_count < 50) return d;

    float threshold = s->shadow_copy_threshold;
    int   far_active = far_pwr > 1e-4f;

    float err_sum     = main_err_smooth + shadow_err_smooth + 1e-10f;
    float err_balance = fabsf(main_err_smooth - shadow_err_smooth) / err_sum;
    int   is_stable_fs = far_active && (err_balance < 0.3f) && !epc_active;
    if (is_stable_fs) {
        float best_err = (main_err_smooth < shadow_err_smooth) ? main_err_smooth : shadow_err_smooth;
        s->copy_err_baseline = 0.995f * s->copy_err_baseline + 0.005f * best_err;
    }
    int error_is_normal = main_err_smooth < (s->copy_err_baseline * 4.0f + 1e-10f);
    int not_saturating  = saturation_level < 0.3f;
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
