/* aec_debug.c -- see aec_debug.h. Compiles to nothing under AEC_NO_STDIO. */
#ifndef AEC_NO_STDIO
#include "aec_debug.h"
#include <stdio.h>

static int   g_frame_idx     = 0;
static FILE* g_trace_fp      = NULL;
static int   g_trace_header  = 0;      /* header written for the current stream */

void aec_debug_set_trace(FILE* fp)  { g_trace_fp = fp; g_trace_header = 0; }
int  aec_debug_trace_active(void)   { return g_trace_fp != NULL; }

void aec_debug_trace_row(const AecDebugTraceRow* row) {
    if (g_trace_fp == NULL || row == NULL) return;
    if (!g_trace_header) {
        fprintf(g_trace_fp,
            "frame,delay,far_active,saturated_echo,usable_linear,"
            "dominant_nearend,filter_converged,fullband_erle,erle_mean,"
            "r2_mean,gain_mean,comfort_noise_mean,near_pwr,raw_err_pwr\n");
        g_trace_header = 1;
    }
    fprintf(g_trace_fp,
        "%d,%d,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
        g_frame_idx, row->delay, row->far_active, row->saturated_echo,
        row->usable_linear, row->dominant_nearend, row->filter_converged,
        row->fullband_erle, row->erle_mean, row->r2_mean, row->gain_mean,
        row->comfort_noise_mean, row->near_pwr, row->raw_err_pwr);
}

void aec_debug_set_frame(int frame_idx, int hop, int sample_rate) {
    (void)hop; (void)sample_rate;
    g_frame_idx = frame_idx;
}

#endif
