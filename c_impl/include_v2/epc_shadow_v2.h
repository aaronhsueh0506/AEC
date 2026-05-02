/* epc_shadow_v2.h — Wave 3: ports of EchoPathChangeDetector +
 * ShadowCopyController (aec.py 2726-3006).
 */
#ifndef AEC_V2_EPC_SHADOW_H
#define AEC_V2_EPC_SHADOW_H

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

typedef struct EpcEventV2 {
    int       fired;     /* 0/1 */
    EpcSource source;
} EpcEventV2;

typedef struct EpcDetectorV2 {
    int    active;       /* 0/1 */
    int    hangover;
    double epv_gain_fast;
    double epv_gain_slow;
    double prev_total_err;
    /* Config slice */
    int    epc_hangover;
    double epc_total_rise;
    double epc_delta_threshold;
} EpcDetectorV2;

void epc_v2_init(EpcDetectorV2* e, int hangover, double total_rise,
                 double delta_threshold);
void epc_v2_reset(EpcDetectorV2* e);
EpcEventV2 epc_v2_force_delay(EpcDetectorV2* e);
EpcEventV2 epc_v2_update_epv(EpcDetectorV2* e, double far_pwr_global,
                             int filter_converged, int main_paused);
EpcEventV2 epc_v2_update_shadow_rise(EpcDetectorV2* e,
                                     double main_err_smooth,
                                     double shadow_err_smooth,
                                     int is_stationary);
void epc_v2_tick_hangover(EpcDetectorV2* e);

/* ── ShadowCopyController ───────────────────────────────── */
typedef enum {
    SC_GATE_ENERGY = 0,
    SC_GATE_COHERENCE,
    SC_GATE_COH_DELAY,
    SC_GATE_STREAK,
} ShadowCopyGateMode;

typedef struct ShadowCopyDecisionV2 {
    int pause_main;
    int boost_q;
    int reverse_copy;
} ShadowCopyDecisionV2;

typedef struct ShadowCopyV2 {
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
} ShadowCopyV2;

void shadow_copy_v2_init(ShadowCopyV2* s, ShadowCopyGateMode mode,
                         double threshold, int hysteresis, int epc_hangover);
void shadow_copy_v2_reset(ShadowCopyV2* s);
ShadowCopyDecisionV2 shadow_copy_v2_update(
    ShadowCopyV2* s,
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

#endif /* AEC_V2_EPC_SHADOW_H */
