/* aec_debug.h — Timestamped debug log infrastructure (Plan §Debug Mode).
 *
 * Build switches:
 *   AEC_DEBUG    : enable debug log call sites (compile-time)
 *   NDEBUG       : strip log call sites entirely
 *   AEC_NO_STDIO : (round-3 review B03) board/no-stdio builds — this header
 *                  does not include <stdio.h> and exposes no FILE*-based API
 *                  at all; every aec_debug_* symbol a LIBRARY TU could call
 *                  becomes a static-inline no-op instead, so aec.c (the only
 *                  library TU that includes this header) still compiles
 *                  clean without pulling any stdio symbol into the archive.
 *                  src/aec_debug.c compiles to an empty TU under this macro
 *                  (see that file) and the Makefile excludes it from the
 *                  library sources entirely when NO_STDIO=1 — this header's
 *                  stubs mean that exclusion can never surface as a link
 *                  error even if some future library TU calls one of these.
 *                  The CLI (example/aec_wav.c) is never built with
 *                  AEC_NO_STDIO (see Makefile/README) and keeps the full
 *                  FILE*-based API unchanged when the macro is undefined —
 *                  default (NO_STDIO=0) builds are byte-identical to before
 *                  this switch existed.
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

#ifndef AEC_NO_STDIO
#include <stdio.h>
#endif
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef AEC_NO_STDIO
/* Global state — owned by aec_debug.c. Threaded use is single-AEC-per-thread
 * for the rewrite; no thread-local needed. */
void aec_debug_set_level(int level);
void aec_debug_set_log(FILE* fp);                /* NULL → stderr */
void aec_debug_set_frame(int frame_idx, int hop, int sample_rate);
int  aec_debug_level(void);

/* Core emit. Module is a short tag like "PBFDKF" / "ResFilter" / "State". */
void aec_debug_logf(const char* module, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
#else
/* AEC_NO_STDIO stubs — zero-cost (static inline, always unused in a library
 * build since the one library call site, aec.c's trace block, is itself
 * compiled out under this same macro). `aec_debug_set_log`/`aec_debug_set_
 * trace` (FILE*-based) are deliberately NOT declared here at all rather than
 * given a void* shim: the only caller today is the CLI (example/aec_wav.c),
 * which is never built with AEC_NO_STDIO, so an accidental library-side call
 * under this macro is a hard compile error instead of a silently-swallowed
 * runtime no-op. */
static inline void aec_debug_set_level(int level) { (void)level; }
static inline void aec_debug_set_frame(int frame_idx, int hop, int sample_rate) {
    (void)frame_idx; (void)hop; (void)sample_rate;
}
static inline int aec_debug_level(void) { return 0; }
static inline void aec_debug_logf(const char* module, const char* fmt, ...) {
    (void)module; (void)fmt;
}
#endif

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

#ifndef AEC_NO_STDIO
void aec_debug_set_trace(FILE* fp);  /* NULL → trace off (default)            */
int  aec_debug_trace_active(void);   /* 1 when a trace file is set            */
void aec_debug_trace_row(const AecDebugTraceRow* row);  /* emit one CSV row   */
#else
/* aec_debug_set_trace (FILE*-based) is unavailable under AEC_NO_STDIO — see
 * the header-level comment above. aec_debug_trace_active()/aec_debug_trace_
 * row() DO have a library call site (aec.c's per-frame trace block), but
 * that whole block is itself compiled out under this same macro (see
 * aec.c), so these stubs exist purely as a belt-and-braces fallback and are
 * never actually invoked in a NO_STDIO=1 library build. trace_active()
 * returns 0 (consistent with "trace off") in case that guarantee is ever
 * relied on directly. */
static inline int  aec_debug_trace_active(void) { return 0; }
static inline void aec_debug_trace_row(const AecDebugTraceRow* row) { (void)row; }
#endif

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
