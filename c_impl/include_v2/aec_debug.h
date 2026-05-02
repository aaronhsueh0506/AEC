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
#ifndef AEC_V2_DEBUG_H
#define AEC_V2_DEBUG_H

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

#endif /* AEC_V2_DEBUG_H */
