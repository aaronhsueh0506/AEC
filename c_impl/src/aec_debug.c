/* aec_debug.c — see aec_debug.h.
 *
 * Implementation kept tiny on purpose: stderr fprintf, no ring buffer, no
 * async I/O. Maintenance > perf for debug code.
 *
 * Round-3 review B03 (AEC_NO_STDIO): this is the ONLY stdio translation unit
 * in the library (fprintf/vfprintf/fputc, plus the lazily-resolved stderr in
 * aec_debug_logf) — every symbol it defines has a static-inline no-op
 * replacement in aec_debug.h under AEC_NO_STDIO, so this whole file compiles
 * to nothing (not even an empty translation unit warning risk: no code, no
 * declarations) when that macro is set. Belt and braces with the Makefile,
 * which additionally excludes this file's object from the library sources
 * entirely under NO_STDIO=1 — either mechanism alone is sufficient to keep
 * libaec.a stdio-free; both together mean neither can regress silently. */
#ifndef AEC_NO_STDIO
#include "aec_debug.h"
#include <stdio.h>
#include <stdarg.h>

static int   g_level         = 0;
static FILE* g_fp            = NULL;   /* lazily resolved to stderr */
static int   g_frame_idx     = 0;
static int   g_hop           = 0;
static int   g_sr            = 16000;
static FILE* g_trace_fp      = NULL;   /* NULL → trace off */
static int   g_trace_header  = 0;      /* header written for current stream */

void aec_debug_set_level(int level) { g_level = level; }
void aec_debug_set_log(FILE* fp)    { g_fp = fp; }
int  aec_debug_level(void)          { return g_level; }

void aec_debug_set_trace(FILE* fp)  { g_trace_fp = fp; g_trace_header = 0; }
int  aec_debug_trace_active(void)   { return g_trace_fp != NULL; }

void aec_debug_trace_row(const AecDebugTraceRow* row) {
    if (g_trace_fp == NULL || row == NULL) return;
    if (!g_trace_header) {
        fprintf(g_trace_fp,
            "frame,delay,far_active,saturated_echo,usable_linear,"
            "dominant_nearend,filter_converged,fullband_erle,erle_mean,"
            "r2_mean,gain_mean,comfort_noise_mean,near_pwr,raw_err_pwr,"
            "limiter_gain\n");
        g_trace_header = 1;
    }
    fprintf(g_trace_fp,
        "%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
        g_frame_idx, row->delay, row->far_active, row->saturated_echo,
        row->usable_linear, row->dominant_nearend, row->filter_converged,
        row->fullband_erle, row->erle_mean, row->r2_mean, row->gain_mean,
        row->comfort_noise_mean, row->near_pwr, row->raw_err_pwr,
        row->limiter_gain);
}

void aec_debug_set_frame(int frame_idx, int hop, int sample_rate) {
    g_frame_idx = frame_idx;
    g_hop       = hop;
    g_sr        = (sample_rate > 0) ? sample_rate : 16000;
}

void aec_debug_logf(const char* module, const char* fmt, ...) {
    FILE* fp = (g_fp != NULL) ? g_fp : stderr;
    float t = (float)(g_frame_idx * g_hop) / (float)g_sr;
    fprintf(fp, "[AEC][t=%6.3fs][f=%5d][%s] ", t, g_frame_idx, module);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);

    fputc('\n', fp);
}
#endif /* AEC_NO_STDIO */
