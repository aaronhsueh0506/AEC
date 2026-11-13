/* baseline2_common.h -- infrastructure shared by baseline2_mono.h and baseline2_array.h: hashing of complex vectors, the chained
 * per-group hash, line formatting, the optional per-hop verbose file (G2_VERBOSE=<file>), and the pool allocator.
 * The includer defines `static void g2_sink(const char* line)` before including (it writes a line to the baseline or compares
 * it with one), plus <aec.h> and <stdio.h>/<stdlib.h>/<string.h>. */
#ifndef AEC_TEST_GOLDEN2_COMMON_H
#define AEC_TEST_GOLDEN2_COMMON_H

#include "baseline2_signal.h"
#include "aec_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G2_STRIDE 16         /* a chained line is printed every G2_STRIDE (16) hops, on the last hop and on and after every event */
#define G2_NB_MAX 1025

enum { G2E_RESET = 1, G2E_LEVEL = 2, G2E_REALIGN = 3, G2E_REVERT = 4 };
typedef struct { int t_ms, type, a; } G2Ev;     /* level: a = AecLevel; realign: a = delta at 16 kHz; revert: a = samples at 16 kHz */

static inline uint64_t g2_hc(uint64_t h, const Complex* p, int n) { return g2_fnv_cxv(h, (const float*)p, n); }

static uint64_t g2_chain(uint64_t c, uint64_t h) {
    c = g2_fnv_u32(c, (uint32_t)(h & 0xFFFFFFFFu));
    return g2_fnv_u32(c, (uint32_t)(h >> 32));
}

static void g2_hex(char* buf, uint64_t h) {
    static const char d[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) buf[i] = d[(h >> (60 - 4 * i)) & 0xFu];
    buf[16] = '\0';
}

static FILE* g2_verbose_file(void) {
    static FILE* f; static int init;
    if (!init) { const char* p = getenv("G2_VERBOSE"); f = p ? fopen(p, "w") : NULL; init = 1; }
    return f;
}

static void g2_line(const char* grid, const char* cs, const char* field, int hop, uint64_t h) {
    char hex[17], line[160];
    g2_hex(hex, h);
    snprintf(line, sizeof line, "%s %s %s %d %s", grid, cs, field, hop, hex);
    g2_sink(line);
}

static void g2_vline(const char* grid, const char* cs, const char* field, int hop, uint64_t h) {
    FILE* f = g2_verbose_file();
    if (f) { char hex[17]; g2_hex(hex, h); fprintf(f, "%s %s %s %d %s\n", grid, cs, field, hop, hex); }
}

/* A 64-byte aligned pool, poisoned (G2_POISON=<byte>, default 0xA5): aec_init must not depend on its contents. */
static void* g2_pool(size_t bytes) {
    void* p = NULL;
    if (!bytes || posix_memalign(&p, 64u, bytes) != 0) return NULL;
    memset(p, getenv("G2_POISON") ? atoi(getenv("G2_POISON")) : 0xA5, bytes);
    return p;
}

#endif /* AEC_TEST_GOLDEN2_COMMON_H */
