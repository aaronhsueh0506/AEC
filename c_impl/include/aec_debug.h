/* aec_debug.h — Timestamped debug log infrastructure (Plan §Debug Mode).
 *
 * Build switches:
 *   AEC_DEBUG    : enable debug log call sites (compile-time)
 *   NDEBUG       : strip log call sites entirely
 *
 * Runtime: aec_debug_set_level() sets per-call gate
 *   0 = off
 *   1 = summary every 100 frames
 *   2 = per-frame
 *   3 = per-frame per-module (everything)
 *
 * Format:
 *   [AEC][t=  1.234s][f=  154][PBFDKF] mu_mean=0.823 P_mean=0.0124 ...
 */
#ifndef AEC_DEBUG_H
#define AEC_DEBUG_H

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Global state — owned by aec_debug.c. Threaded use is single-AEC-per-thread
 * for the rewrite; no thread-local needed. */
void aec_debug_set_level(int level);
void aec_debug_set_log(FILE* fp);                /* NULL → stderr */
void aec_debug_set_frame(int frame_idx, int hop, int sample_rate);
int  aec_debug_level(void);

/* Core emit. Module is a short tag like "PBFDKF" / "ResFilter" / "State". */
void aec_debug_logf(const char* module, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* ── Per-frame structured trace ("logr") ──────────────────────────────────────
 * A separate, runtime-gated CSV stream that captures the post-filter internals
 * once per hop (mirrors the Python per-frame diagnostic schema). Independent of
 * the level-gated AEC_DEBUG_LOG path above: it is enabled solely by handing a
 * non-NULL FILE* to aec_debug_set_trace(), so when no trace file is set the
 * hot path pays nothing (a single NULL-pointer test).
 *
 * Schema (one row per hop):
 *   frame,delay,far_active,saturated_echo,usable_linear,dominant_nearend,
 *   filter_converged,fullband_erle,erle_mean,r2_mean,gain_mean,
 *   comfort_noise_mean,near_pwr,raw_err_pwr,limiter_gain
 */
typedef struct AecDebugTraceRow {
    int    delay;               /* min_direct_path_filter_delay (blocks)      */
    int    far_active;          /* active_render                              */
    int    saturated_echo;      /* aec_state.saturated_echo()                 */
    int    usable_linear;       /* aec_state.usable_linear_estimate()         */
    int    dominant_nearend;    /* suppression_gain dominant-nearend          */
    int    filter_converged;    /* AEC3 per-frame aec3_converged              */
    float  fullband_erle;       /* aec_state.fullband_erle_log2()             */
    float  erle_mean;           /* mean of per-bin subband ERLE               */
    float  r2_mean;             /* mean of per-bin residual-echo R^2          */
    float  gain_mean;           /* mean of per-bin SuppressionGain output     */
    float  comfort_noise_mean;  /* mean of per-bin comfort-noise PSD          */
    float  near_pwr;            /* near-power EMA                             */
    float  raw_err_pwr;         /* raw-error-power EMA                        */
    float  limiter_gain;        /* OLA output limiter gain                    */
} AecDebugTraceRow;

void aec_debug_set_trace(FILE* fp);  /* NULL → trace off (default)            */
int  aec_debug_trace_active(void);   /* 1 when a trace file is set            */
void aec_debug_trace_row(const AecDebugTraceRow* row);  /* emit one CSV row   */

/* Compile-time gate. In NDEBUG release builds the call site collapses to
 * nothing — no string literal is emitted, no branch is generated. */
#if defined(AEC_DEBUG) && !defined(NDEBUG)
  #define AEC_DEBUG_LOG(min_level, module, ...) \
      do { if (aec_debug_level() >= (min_level)) \
               aec_debug_logf((module), __VA_ARGS__); } while (0)
#else
  #define AEC_DEBUG_LOG(min_level, module, ...) ((void)0)
#endif

#ifdef __cplusplus
}
#endif

#endif /* AEC_DEBUG_H */
