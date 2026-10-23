/* EchoPathChangeDetector and ShadowCopy. Scalar math is float32.
 * Python reference: modules/epc.py. */
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
    float  epv_gain_fast;
    float  epv_gain_slow;
    float  prev_total_err;
    /* Config slice */
    int    epc_hangover;
    float  epc_total_rise;
    float  epc_delta_threshold;
    /* Config slice, fixed for the instance lifetime (epc_reset() leaves it alone);
     * EPV_FAST_TC/EPV_SLOW_TC are retimed to the live grid in epc_init(). */
    float  epv_fast_tc;
    float  epv_slow_tc;
} EpcDetector;

/* hop_size/sample_rate retime EPV_FAST_TC/EPV_SLOW_TC, authored for hop=160 @ 16 kHz. */
void epc_init(EpcDetector* e, int hangover, float total_rise,
                 float delta_threshold, int hop_size, int sample_rate);
void epc_reset(EpcDetector* e);
EpcEvent epc_force_delay(EpcDetector* e);
EpcEvent epc_update_epv(EpcDetector* e, float far_pwr_global,
                             int filter_converged, int main_paused);
EpcEvent epc_update_shadow_rise(EpcDetector* e,
                                     float main_err_smooth,
                                     float shadow_err_smooth,
                                     int is_stationary);
void epc_tick_hangover(EpcDetector* e);

/* ── ShadowCopy ─────────────────────────────────────────── */
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
    float  copy_err_baseline;
    int    copy_counter;
    /* Per-grid EMA retention, set at init and kept by shadow_copy_reset().
     * copy_counter serves both the energy and the streak gate. */
    float  copy_err_baseline_retention;
    int    main_paused;
    int    pause_resume;
    /* Config slice */
    float  shadow_copy_threshold;
    int    shadow_copy_hysteresis;
    int    epc_hangover;
} ShadowCopy;

void shadow_copy_init(ShadowCopy* s, ShadowCopyGateMode mode,
                      float threshold, int hysteresis, int epc_hangover);
void shadow_copy_init_for_grid(ShadowCopy* s, ShadowCopyGateMode mode,
                               float threshold, int hysteresis,
                               int epc_hangover, int hop_size,
                               int sample_rate);
void shadow_copy_reset(ShadowCopy* s);
ShadowCopyDecision shadow_copy_update(
    ShadowCopy* s,
    int    shadow_frame_count,
    int    warmup_hops,
    float  far_pwr,
    float  main_err_smooth,
    float  shadow_err_smooth,
    int    epc_active,
    float  saturation_level,
    float  dt_from_energy,
    float  dt_from_coherence,   /* default 0.0 */
    int    delay_reliable);     /* default 0 */

#ifdef __cplusplus
}
#endif

#endif /* AEC_EPC_SHADOW_H */
