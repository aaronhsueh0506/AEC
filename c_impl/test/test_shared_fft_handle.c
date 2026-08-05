/* test_shared_fft_handle.c — shared-FFT-handle regression test.
 *
 * aec_create()/aec_init() used to carve/construct THREE separate FftHandles
 * per instance (main_filter.base.fft, shadow_filter.fft, post_fft), one for
 * each sub-module, all at the identical fft_size. They now share exactly
 * ONE: aec_carve() constructs it once and hands the same pointer to
 * pbfdkf_init_static()/pbfdaf_init_static() via their new required
 * `shared_fft` parameter; pbfdaf_free()'s static-path branch no longer
 * destroys it (ownership stays with post_fft, torn down once by
 * aec_destroy()). This test proves:
 *
 *   1. All three FftHandle pointers on a live instance are the exact SAME
 *      address, not merely equal-sized/equivalent objects -- across all 3
 *      officially-contracted grids x shadow on/off. A revert to
 *      per-sub-module private handles would make this fail immediately
 *      (main_filter.base.fft/shadow_filter.fft would point at distinct
 *      heap/pool regions from post_fft).
 *   2. The shared identity survives aec_reset() (reset never re-carves the
 *      pool/arena, only clears state -- the handle pointers must be
 *      untouched).
 *   3. Both aec_create() (heap arena) and aec_init() (caller pool) share the
 *      same handle-identity property -- the sharing is not an accident of
 *      one construction path only.
 *   4. Sharing produces byte-identical AecResContext to a hypothetical
 *      per-sub-module-owned world: proven indirectly via the existing
 *      byte-equal aec_wav render gate (see CHANGELOG.md), not re-derived
 *      here -- this file's job is to pin the OWNERSHIP invariant a render
 *      diff cannot see (two builds with/without sharing produce identical
 *      audio either way; only the pointer identity differs).
 *
 * Build (standalone, from c_impl/ — mirrors test_process_context.c's
 * documented recipe):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_shared_fft_handle.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_shared_fft_handle
 * Run:
 *   ./bin/test_shared_fft_handle
 * Also wired into the Makefile: `make test-shared-fft-handle`.
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

static void check_shared_identity(const Aec* a, int enable_shadow, const char* label) {
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: post_fft is non-NULL", label);
    CHECK(a->post_fft != NULL, msg);

    snprintf(msg, sizeof(msg), "%s: main filter's FftHandle IS post_fft (same pointer)", label);
    CHECK(a->main_filter.base.fft == a->post_fft, msg);

    if (enable_shadow) {
        snprintf(msg, sizeof(msg), "%s: shadow filter's FftHandle IS post_fft (same pointer)", label);
        CHECK(a->shadow_filter.fft == a->post_fft, msg);
    }
}

/* aec_create() (heap arena) path, across all 3 officially-contracted grids
 * (see AEC/CLAUDE.md / pipelines/README.md "Parameter Alignment") x shadow
 * on/off. */
static void test_shared_identity_heap(int sample_rate, int fft_size, int enable_shadow) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sample_rate);
    if (fft_size > 0) cfg.fft_size = fft_size;
    cfg.enable_shadow = enable_shadow;

    Aec a;
    char label[128];
    snprintf(label, sizeof(label), "heap sr=%d/fft=%d/shadow=%d", sample_rate, cfg.fft_size, enable_shadow);
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: aec_create()", label);
    CHECK(aec_create(&a, &cfg) == 0, msg);

    check_shared_identity(&a, enable_shadow, label);

    /* Survives a hop of real processing (nothing about normal use should
     * ever reassign a->post_fft or the sub-modules' fft pointers). */
    int hop = aec_hop_size(&a);
    if (hop > 0 && hop <= 4096) {
        float mic[4096] = {0}, ref[4096] = {0}, out[4096];
        aec_process(&a, mic, ref, out);
        snprintf(msg, sizeof(msg), "%s: identity survives a real aec_process() hop", label);
        CHECK(a.main_filter.base.fft == a.post_fft, msg);
    }

    /* Survives aec_reset(): reset clears state, never re-carves the arena,
     * so the handle pointers themselves must be untouched. */
    aec_reset(&a);
    snprintf(msg, sizeof(msg), "%s: identity survives aec_reset()", label);
    CHECK(a.main_filter.base.fft == a.post_fft, msg);
    if (enable_shadow) {
        snprintf(msg, sizeof(msg), "%s: shadow identity survives aec_reset()", label);
        CHECK(a.shadow_filter.fft == a.post_fft, msg);
    }

    aec_destroy(&a);
}

/* aec_init() (caller-owned pool) path -- the sharing is not an accident of
 * the heap-arena construction path only. */
static void test_shared_identity_static(int sample_rate, int fft_size, int enable_shadow) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sample_rate);
    if (fft_size > 0) cfg.fft_size = fft_size;
    cfg.enable_shadow = enable_shadow;

    size_t need = aec_get_mem_size(&cfg);
    char label[128];
    snprintf(label, sizeof(label), "static sr=%d/fft=%d/shadow=%d", sample_rate, cfg.fft_size, enable_shadow);
    char msg[192];
    snprintf(msg, sizeof(msg), "%s: aec_get_mem_size() > 0", label);
    CHECK(need > 0, msg);
    if (need == 0) return;

    void* pool = NULL;
    if (posix_memalign(&pool, 16, need) != 0 || !pool) {
        snprintf(msg, sizeof(msg), "%s: posix_memalign", label);
        CHECK(0, msg);
        return;
    }

    Aec* a = aec_init(pool, need, &cfg);
    snprintf(msg, sizeof(msg), "%s: aec_init()", label);
    CHECK(a != NULL, msg);
    if (!a) { free(pool); return; }

    check_shared_identity(a, enable_shadow, label);

    aec_destroy(a);   /* genuine no-op on the pool path except the NE10
                        * twiddle-config teardown; caller still owns `pool`. */
    free(pool);
}

int main(void) {
    static const int grids[][1] = { {0}, {512}, {0} };
    static const int rates[]    = { 16000, 16000, 48000 };
    for (int g = 0; g < 3; ++g) {
        for (int shadow = 0; shadow <= 1; ++shadow) {
            test_shared_identity_heap(rates[g], grids[g][0], shadow);
            test_shared_identity_static(rates[g], grids[g][0], shadow);
        }
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
