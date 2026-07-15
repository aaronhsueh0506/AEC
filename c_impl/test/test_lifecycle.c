/* test_lifecycle.c — F03/F04 review remediation regression test.
 *
 * F03 was "dynamic AEC leaks 87 allocations / 184,144 B per create/destroy
 * lifecycle" (aec_create() used to do ~87 individual per-array mallocs that
 * aec_destroy() never freed). The fix (aec.c: aec_carve(), shared by
 * aec_create() and aec_init()) folds every one of those arrays into a
 * single arena; aec_destroy() now frees it with one free(). This test
 * repeatedly create/process/reset/destroy's an Aec so a leak checker
 * (`leaks --atExit`) run over it proves the fix: a real per-cycle leak
 * would show up as N-times-repeated allocations; a correct teardown stays
 * flat regardless of the iteration count.
 *
 * F04 was "no allocation-failure rollback; void nested inits" — this test
 * doesn't force an OOM (there's no portable way to do that here), but it
 * does exercise the two cheap, always-checkable safety properties the fix
 * guarantees:
 *   - aec_destroy(NULL) is a safe no-op.
 *   - Calling aec_destroy() twice on the same successfully-created instance
 *     is safe (idempotent: the second call frees nothing and dereferences
 *     nothing already freed) PROVIDED the instance is not used (aec_process
 *     etc.) between or after the two destroy calls — that is NOT
 *     guaranteed/tested here, only double-destroy-then-stop is.
 *
 * Build (standalone, from c_impl/ — mirrors test_config_validation.c's
 * documented recipe):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_lifecycle.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_lifecycle
 * Run:
 *   ./bin/test_lifecycle
 * Leak check (macOS):
 *   leaks --atExit -- ./bin/test_lifecycle
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

/* One create -> 5 hops of process -> reset -> destroy cycle. Returns 1 on
 * success (matching this test's own pass/fail convention), 0 on any
 * failure. Uses a fresh zeroed near/far hop each call (silence in, silence
 * out is a fine smoke signal here — this test is about the allocation
 * lifecycle, not audio content; that is covered elsewhere, e.g.
 * test_static_aec.c / the render byte-identity gate). */
static int run_one_cycle(const AecConfig* cfg) {
    Aec a;
    if (aec_create(&a, cfg) != 0) return 0;

    int hop = aec_hop_size(&a);
    if (hop <= 0 || hop > 4096) { aec_destroy(&a); return 0; }

    float mic[4096] = {0};
    float ref[4096] = {0};
    float out[4096];
    for (int i = 0; i < 5; ++i) {
        aec_process(&a, mic, ref, out);
    }

    aec_reset(&a);
    for (int i = 0; i < 5; ++i) {
        aec_process(&a, mic, ref, out);
    }

    aec_destroy(&a);
    return 1;
}

static void test_cycles(int n_iters) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

    int all_ok = 1;
    for (int i = 0; i < n_iters; ++i) {
        if (!run_one_cycle(&cfg)) { all_ok = 0; break; }
    }

    char label[96];
    snprintf(label, sizeof(label),
             "%d create->process(x5)->reset->process(x5)->destroy cycles all succeed",
             n_iters);
    CHECK(all_ok, label);
}

/* aec_destroy(NULL): documented as a safe no-op (aec.c: `if (!a) return;`). */
static void test_destroy_null(void) {
    aec_destroy(NULL);
    CHECK(1, "aec_destroy(NULL) does not crash");
}

/* Double-destroy: aec_destroy() is idempotent on a successfully-created
 * instance as long as nothing calls back into it between/after the two
 * destroy calls (aec_process/aec_reset after a destroy is NOT supported —
 * only "destroy twice, then stop" is). Covers both the heap-arena path
 * (aec_create) and the static-pool path (aec_init). */
static void test_double_destroy_heap(void) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

    Aec a;
    int created = (aec_create(&a, &cfg) == 0);
    CHECK(created, "double-destroy (heap-arena path): aec_create() succeeds");
    if (!created) return;

    int hop = aec_hop_size(&a);
    float mic[4096] = {0}, ref[4096] = {0}, out[4096];
    if (hop > 0 && hop <= 4096) aec_process(&a, mic, ref, out);

    aec_destroy(&a);
    aec_destroy(&a);   /* second call must not double-free / use-after-free */
    CHECK(1, "double-destroy (heap-arena path): second aec_destroy() does not crash");
}

static void test_double_destroy_static(void) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

    size_t sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "double-destroy (static-pool path): aec_get_mem_size() > 0");
    if (sz == 0) return;

    void* pool = malloc(sz);
    CHECK(pool != NULL, "double-destroy (static-pool path): malloc pool");
    if (!pool) return;

    Aec* a = aec_init(pool, sz, &cfg);
    CHECK(a != NULL, "double-destroy (static-pool path): aec_init() succeeds");
    if (!a) { free(pool); return; }

    int hop = aec_hop_size(a);
    float mic[4096] = {0}, ref[4096] = {0}, out[4096];
    if (hop > 0 && hop <= 4096) aec_process(a, mic, ref, out);

    aec_destroy(a);
    aec_destroy(a);   /* second call must not double-free / use-after-free */
    CHECK(1, "double-destroy (static-pool path): second aec_destroy() does not crash");

    free(pool);
}

int main(void) {
    int iters[] = { 1, 10, 1000 };
    for (size_t i = 0; i < sizeof(iters) / sizeof(iters[0]); ++i) {
        test_cycles(iters[i]);
    }

    test_destroy_null();
    test_double_destroy_heap();
    test_double_destroy_static();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
