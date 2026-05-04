/* epc_shadow.h — Wave 3: ports of EchoPathChangeDetector +
 * ShadowCopyController (aec.py 2726-3006).
 */
#ifndef AEC_EPC_SHADOW_H
#define AEC_EPC_SHADOW_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── EchoPathChangeDetector ─────────────────────────────── */
typedef enum {
    EPC_NONE = 0,
    EPC_DELAY,
    EPC_EPV,
    EPC_SHADOW_RISE,
} EpcSource;

typedef struct EpcEvent {
    int       fired;     /* 0/1 */
    EpcSource source;
} EpcEvent;

typedef struct EpcDetector {
    int    active;       /* 0/1 */
    int    hangover;
    double epv_gain_fast;
    double epv_gain_slow;
    double prev_total_err;
    /* Config slice */
    int    epc_hangover;
    double epc_total_rise;
    double epc_delta_threshold;
} EpcDetector;

void epc_init(EpcDetector* e, int hangover, double total_rise,
                 double delta_threshold);
void epc_reset(EpcDetector* e);
EpcEvent epc_force_delay(EpcDetector* e);
EpcEvent epc_update_epv(EpcDetector* e, double far_pwr_global,
                             int filter_converged, int main_paused);
EpcEvent epc_update_shadow_rise(EpcDetector* e,
                                     double main_err_smooth,
                                     double shadow_err_smooth,
                                     int is_stationary);
void epc_tick_hangover(EpcDetector* e);

/* ── ShadowCopyController ───────────────────────────────── */
typedef enum {
    SC_GATE_ENERGY = 0,
    SC_GATE_COHERENCE,
    SC_GATE_COH_DELAY,
    SC_GATE_STREAK,
} ShadowCopyGateMode;

typedef struct ShadowCopyDecision {
    int pause_main;
    int boost_q;
    int reverse_copy;
} ShadowCopyDecision;

typedef struct ShadowCopy {
    ShadowCopyGateMode gate_mode;
    double copy_err_baseline;
    int    copy_counter;
    int    streak;
    int    main_paused;
    int    pause_resume;
    /* Config slice */
    double shadow_copy_threshold;
    int    shadow_copy_hysteresis;
    int    epc_hangover;
} ShadowCopy;

void shadow_copy_init(ShadowCopy* s, ShadowCopyGateMode mode,
                         double threshold, int hysteresis, int epc_hangover);
void shadow_copy_reset(ShadowCopy* s);
ShadowCopyDecision shadow_copy_update(
    ShadowCopy* s,
    int    shadow_frame_count,
    double far_pwr,
    double main_err_smooth,
    double shadow_err_smooth,
    int    epc_active,
    double saturation_level,
    double dt_from_energy,
    double dt_from_coherence,   /* default 0.0 */
    int    delay_reliable);     /* default 0 */

#ifdef __cplusplus
}
#endif

#endif /* AEC_EPC_SHADOW_H */
