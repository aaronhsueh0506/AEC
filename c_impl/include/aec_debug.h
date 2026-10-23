/* aec_debug.h -- debug level, log and per-frame CSV trace.
 * AEC_NO_STDIO: the header does not include <stdio.h>, declares no FILE*-based API and provides
 * static-inline no-op stubs, so a stdio-free library build links cleanly.
 * aec_debug_set_trace() arms the trace; with no trace file the hot path pays one NULL test. */
#ifndef AEC_DEBUG_H
#define AEC_DEBUG_H

#ifndef AEC_NO_STDIO
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AEC_NO_STDIO
/* State is global, one AEC per thread. */
void aec_debug_set_level(int level);
void aec_debug_set_log(FILE* fp);                /* NULL -> stderr */
void aec_debug_set_frame(int frame_idx, int hop, int sample_rate);
#else
/* AEC_NO_STDIO stubs. The FILE*-based setters are deliberately not declared, so a call under
 * this macro fails to compile. */
static inline void aec_debug_set_level(int level) { (void)level; }
static inline void aec_debug_set_frame(int frame_idx, int hop, int sample_rate) {
    (void)frame_idx; (void)hop; (void)sample_rate;
}
#endif

/* Per-frame trace: one CSV row per hop with
 * frame,delay,far_active,saturated_echo,usable_linear,dominant_nearend,filter_converged,
 * fullband_erle,erle_mean,r2_mean,gain_mean,comfort_noise_mean,near_pwr,raw_err_pwr */
typedef struct AecDebugTraceRow {
    int    delay;
    int    far_active;
    int    saturated_echo;
    int    usable_linear;
    int    dominant_nearend;
    int    filter_converged;
    float  fullband_erle;
    float  erle_mean;
    float  r2_mean;
    float  gain_mean;
    float  comfort_noise_mean;
    float  near_pwr;
    float  raw_err_pwr;
} AecDebugTraceRow;

#ifndef AEC_NO_STDIO
void aec_debug_set_trace(FILE* fp);  /* NULL -> trace off (default) */
int  aec_debug_trace_active(void);
void aec_debug_trace_row(const AecDebugTraceRow* row);
#else
static inline int  aec_debug_trace_active(void) { return 0; }
static inline void aec_debug_trace_row(const AecDebugTraceRow* row) { (void)row; }
#endif

#ifdef __cplusplus
}
#endif

#endif
