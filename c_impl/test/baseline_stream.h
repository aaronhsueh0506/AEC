/* baseline_stream.h -- optional capture and comparison of a test's hashed line stream (no baseline is shipped).
 *
 * AEC_TEST_BASELINE_WRITE=<dir>  writes <dir>/<test>_simd{0,1}.txt (simd0 = SIMD=0 build, simd1 = NEON build).
 * AEC_TEST_BASELINE=<dir>        compares every line with that file.
 * Asking for either and not getting it (unsupported host build, unreadable or unwritable file, G2_ONLY set) is a failure.
 * Without either variable the test prints "baseline: none (structural checks only)" and runs its own checks only.
 *
 * The includer provides <stdio.h>, <stdlib.h>, <string.h> and utility/simd_kernels.h, calls baseline_load("<test>") once, feeds
 * every line to baseline_line(), and adds baseline_finish() to its failure count. g_idx / g_replay let it replay a span of lines
 * (the replay compares again and writes nothing). */
#ifndef AEC_TEST_BASELINE_STREAM_H
#define AEC_TEST_BASELINE_STREAM_H

static char* g_bl_buf;
static char** g_lines;
static long g_nlines, g_idx, g_compared, g_mismatch;
static int g_have_baseline, g_bl_error;
static FILE* g_wr;
static int g_replay;

static const char* baseline_path(const char* dir, const char* test) {
    static char path[1024];
#if defined(SIMD_KERNELS_FORCE_SCALAR)
    snprintf(path, sizeof path, "%s/%s_simd0.txt", dir, test);
#elif defined(SK_HAVE_NEON) && SK_HAVE_NEON
    snprintf(path, sizeof path, "%s/%s_simd1.txt", dir, test);
#else
    (void)dir; (void)test;
    return NULL;
#endif
    return path;
}

static int baseline_fail(const char* why, const char* path) {
    printf("FAIL: baseline %s%s%s\n", why, path ? ": " : "", path ? path : "");
    g_bl_error++;
    return 0;
}

/* 1 when the stream is being compared with a baseline, else 0 (writing, none requested, or failed: g_bl_error says which). */
static int baseline_load(const char* test) {
    const char* wdir = getenv("AEC_TEST_BASELINE_WRITE");
    const char* dir = getenv("AEC_TEST_BASELINE");
    const char* path;
    FILE* f;
    long size, i, n = 0;
    int want_wr = wdir && *wdir, want_cmp = dir && *dir;
    if (!want_wr && !want_cmp) { printf("baseline: none (structural checks only)\n"); return 0; }
    if (getenv("G2_ONLY")) return baseline_fail("capture or comparison is incompatible with G2_ONLY (it runs a subset of the cases)", NULL);
    if (want_wr) {
        path = baseline_path(wdir, test);
        if (!path) return baseline_fail("cannot be written on this host build (no SIMD=0 / NEON variant)", NULL);
        g_wr = fopen(path, "wb");
        if (!g_wr) return baseline_fail("cannot write", path);
        printf("baseline: writing %s\n", path);
        return 0;
    }
    path = baseline_path(dir, test);
    if (!path) return baseline_fail("cannot be compared on this host build (no SIMD=0 / NEON variant)", NULL);
    f = fopen(path, "rb");
    if (!f) return baseline_fail("not found", path);
    if (fseek(f, 0, SEEK_END) != 0 || (size = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return baseline_fail("unreadable", path); }
    g_bl_buf = (char*)malloc((size_t)size + 1);
    if (!g_bl_buf || fread(g_bl_buf, 1, (size_t)size, f) != (size_t)size) { fclose(f); return baseline_fail("short read", path); }
    fclose(f);
    g_bl_buf[size] = '\0';
    for (i = 0; i < size; ++i) if (g_bl_buf[i] == '\n') n++;
    g_lines = (char**)malloc((size_t)(n + 1) * sizeof(char*));
    if (!g_lines) return baseline_fail("out of memory", path);
    n = 0; g_lines[n++] = g_bl_buf;
    for (i = 0; i < size; ++i) if (g_bl_buf[i] == '\n') { g_bl_buf[i] = '\0'; if (i + 1 < size) g_lines[n++] = g_bl_buf + i + 1; }
    g_nlines = n;
    printf("baseline: %s\n", path);
    return 1;
}

static void baseline_line(const char* line) {
    if (g_wr && !g_replay) { fputs(line, g_wr); fputc('\n', g_wr); }
    if (!g_have_baseline) return;
    if (g_idx >= g_nlines) { if (g_mismatch++ < 10) printf("baseline ended early at: %s\n", line); return; }
    {
        const char* want = g_lines[g_idx++];
        g_compared++;
        if (strcmp(want, line) != 0 && g_mismatch++ < 10) printf("MISMATCH\n  want %s\n  got  %s\n", want, line);
    }
}

/* Failures to add to the test's count; closes the capture file. */
static int baseline_finish(void) {
    int fails = g_bl_error;
    if (g_have_baseline) {
        printf("baseline lines compared: %ld, mismatches: %ld (baseline has %ld lines, consumed %ld)\n",
               g_compared, g_mismatch, g_nlines, g_idx);
        if (g_mismatch) fails++;
        if (g_idx != g_nlines) { printf("FAIL: baseline lines not all consumed\n"); fails++; }
    }
    if (g_wr && fclose(g_wr) != 0) { printf("FAIL: baseline capture file not closed cleanly\n"); fails++; }
    g_wr = NULL;
    return fails;
}

#endif
