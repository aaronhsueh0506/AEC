/**
 * epc_detector.h - EchoPathChangeDetector (3-trigger state machine)
 * Port of Python aec.py EchoPathChangeDetector (lines 2726-2836).
 */

#ifndef EPC_DETECTOR_H
#define EPC_DETECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EpcDetector EpcDetector;

typedef enum {
    EPC_SOURCE_NONE        = 0,
    EPC_SOURCE_DELAY       = 1,
    EPC_SOURCE_EPV         = 2,
    EPC_SOURCE_SHADOW_RISE = 3
} EpcSource;

typedef struct {
    int       fired;        /* 1 if event fired this frame */
    EpcSource source;       /* which trigger */
} EpcEvent;

/**
 * Config from AecConfig:
 *   epc_hangover           (default 20)
 *   epc_total_rise         (default 1.5)
 *   epc_delta_threshold    (default 0.3)
 */
EpcDetector* epc_create(int epc_hangover, float epc_total_rise,
                        float epc_delta_threshold);
void epc_destroy(EpcDetector* d);
void epc_reset(EpcDetector* d);

/* Trigger 1: caller-driven delay shift. */
EpcEvent epc_force_delay(EpcDetector* d);

/* Trigger 2: EPV gain ratio. */
EpcEvent epc_update_epv(EpcDetector* d, float far_pwr_global,
                        int filter_converged, int main_paused);

/* Trigger 3: shadow-rise (gated by stationary far). */
EpcEvent epc_update_shadow_rise(EpcDetector* d,
                                float main_err_smooth,
                                float shadow_err_smooth,
                                int is_stationary);

/* Tick hangover (call once per frame in original guard, NOT when
 * shadow_rise fired this frame — preserve Python if/elif/else). */
void epc_tick_hangover(EpcDetector* d);

int   epc_is_active(const EpcDetector* d);
int   epc_get_hangover(const EpcDetector* d);
float epc_get_gain_fast(const EpcDetector* d);
float epc_get_gain_slow(const EpcDetector* d);

#ifdef __cplusplus
}
#endif
#endif /* EPC_DETECTOR_H */
