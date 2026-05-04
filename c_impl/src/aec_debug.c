/* aec_debug.c — see aec_debug.h.
 *
 * Implementation kept tiny on purpose: stderr fprintf, no ring buffer, no
 * async I/O. Maintenance > perf for debug code.
 */
#include "aec_debug.h"
#include <stdio.h>
#include <stdarg.h>

static int   g_level       = 0;
static FILE* g_fp          = NULL;     /* lazily resolved to stderr */
static int   g_frame_idx   = 0;
static int   g_hop         = 0;
static int   g_sr          = 16000;

void aec_debug_set_level(int level) { g_level = level; }
void aec_debug_set_log(FILE* fp)    { g_fp = fp; }
int  aec_debug_level(void)          { return g_level; }

void aec_debug_set_frame(int frame_idx, int hop, int sample_rate) {
    g_frame_idx = frame_idx;
    g_hop       = hop;
    g_sr        = (sample_rate > 0) ? sample_rate : 16000;
}

void aec_debug_logf(const char* module, const char* fmt, ...) {
    FILE* fp = (g_fp != NULL) ? g_fp : stderr;
    double t = (double)(g_frame_idx * g_hop) / (double)g_sr;
    fprintf(fp, "[AEC][t=%6.3fs][f=%5d][%s] ", t, g_frame_idx, module);

    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);

    fputc('\n', fp);
}
