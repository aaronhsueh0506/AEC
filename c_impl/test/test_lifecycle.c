/* test_lifecycle.c — create/destroy lifecycle regression test.
 *
 * Leak check: aec_create() allocates one block of aec_get_mem_size() bytes that holds the Aec and every sub-module array
 * (aec.c: aec_carve(), shared by aec_create() and aec_init()); aec_destroy() frees it with one free(). This test repeatedly
 * create/process/reset/destroy's an Aec (single-channel and array) so a leak checker
 * (`leaks --atExit`) run over it shows there is no per-cycle leak: a real
 * leak would show up as N-times-repeated allocations; a correct teardown
 * stays flat regardless of the iteration count.
 *
 * Allocation-failure safety: this test doesn't force an OOM (there's no
 * portable way to do that here), but it does exercise the two cheap,
 * always-checkable safety properties:
 *   - aec_destroy(NULL) is a safe no-op.
 *   - Calling aec_destroy() twice on the same aec_init() instance is safe (idempotent: the second call frees nothing and
 *     dereferences nothing already freed). An aec_create() instance is freed by aec_destroy(); its pointer is dead.
 *   - A destroyed pool instance is refused by the entry points (aec.h, "A handle that is not ready"): process calls answer
 *     AEC_ERR_ARG and aec_get_linear() zeroes *out.
 *
 * Run from c_impl/:  make test-lifecycle
 * Leak check (macOS):  leaks --atExit -- "$(make -s print-bin-dir)/test_lifecycle"
 */
#include "aec.h"
#include "aec_internal.h"
#include "aec_pool_test_util.h"
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
    Aec *a = NULL;
    if ((a = aec_create(cfg)) == NULL) return 0;

    int hop = aec_hop_size(a);
    if (hop <= 0 || hop > 4096) { aec_destroy(a); return 0; }

    float mic[4096] = {0};
    float ref[4096] = {0};
    float out[4096];
    for (int i = 0; i < 5; ++i) {
        aec_process(a, mic, ref, out);
    }

    aec_reset(a);
    for (int i = 0; i < 5; ++i) {
        aec_process(a, mic, ref, out);
    }

    aec_destroy(a);
    return 1;
}

/* The same cycle on a four-microphone array instance: the heap block holds the control block, the lanes and the estimator. */
#ifndef AEC_NO_ARRAY
static int run_one_array_cycle(const AecConfig* cfg) {
    Aec* a = aec_create(cfg);
    if (!a) return 0;
    int hop = aec_hop_size(a);
    if (hop <= 0 || hop > 1024) { aec_destroy(a); return 0; }
    static float mics[1024 * AEC_ARRAY_MICS], ref[1024];
    for (int i = 0; i < 5; ++i) {
        if (aec_process_linear(a, mics, ref) != AEC_OK) { aec_destroy(a); return 0; }
    }
    aec_reset(a);
    for (int i = 0; i < 5; ++i) {
        if (aec_process_linear(a, mics, ref) != AEC_OK) { aec_destroy(a); return 0; }
    }
    aec_destroy(a);
    return 1;
}

static void test_array_cycles(int n_iters) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.n_mics = AEC_ARRAY_MICS;
    cfg.residual = AEC_RESIDUAL_STATE;
    int all_ok = 1;
    for (int i = 0; i < n_iters; ++i) {
        if (!run_one_array_cycle(&cfg)) { all_ok = 0; break; }
    }
    char label[96];
    snprintf(label, sizeof(label), "%d array create->process_linear(x5)->reset->process_linear(x5)->destroy cycles all succeed", n_iters);
    CHECK(all_ok, label);
}
#endif

static void test_cycles(int n_iters) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

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

/* The process and view entry points on a destroyed pool instance (the pool is still alive). */
static void check_destroyed_single(Aec* a, const char* what) {
    static float mic[4096], ref[4096], out[4096];
    AecLinearView lv;
    char msg[96];
    size_t n, declared;
    memset(&lv, 0x5A, sizeof lv);
    lv.struct_size = sizeof lv;
    snprintf(msg, sizeof msg, "%s: aec_process is refused", what);
    CHECK(aec_process(a, mic, ref, out) == AEC_ERR_ARG, msg);
    snprintf(msg, sizeof msg, "%s: aec_process_linear is refused", what);
    CHECK(aec_process_linear(a, mic, ref) == AEC_ERR_ARG, msg);
    snprintf(msg, sizeof msg, "%s: aec_process_linear_shared_far_from answers AEC_ERR_ARG", what);
    CHECK(aec_process_linear_shared_far_from(a, mic, ref, NULL) == AEC_ERR_ARG, msg);
    snprintf(msg, sizeof msg, "%s: aec_get_linear is refused", what);
    CHECK(aec_get_linear(a, &lv) == AEC_ERR_ARG, msg);
    snprintf(msg, sizeof msg, "%s: aec_get_linear zeroes *out but struct_size", what);
    memcpy(&declared, &lv, sizeof declared);
    for (n = sizeof declared; n < sizeof lv; ++n) if (((unsigned char*)&lv)[n]) break;
    CHECK(declared == sizeof lv && n == sizeof lv, msg);
}

/* Heap path: aec_destroy() frees the block that holds the instance, so the pointer is dead afterwards; a new aec_create()
 * builds a working instance. Pool path: aec_destroy() is idempotent and leaves the handle refused. */
static void test_destroy_heap(void) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

    Aec *a = NULL;
    int created = ((a = aec_create(&cfg)) != NULL);
    CHECK(created, "destroy (heap path): aec_create() succeeds");
    if (!created) return;

    int hop = aec_hop_size(a);
    float mic[4096] = {0}, ref[4096] = {0}, out[4096];
    if (hop > 0 && hop <= 4096) aec_process(a, mic, ref, out);

    aec_destroy(a);
    created = ((a = aec_create(&cfg)) != NULL);
    CHECK(created && aec_hop_size(a) == hop, "destroy (heap path): aec_create() after a destroy builds the same instance");
    aec_destroy(a);
}

static void test_double_destroy_static(void) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);

    size_t sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "double-destroy (static-pool path): aec_get_mem_size() > 0");
    if (sz == 0) return;

    void* pool = aec_test_pool_alloc(sz);
    CHECK(pool != NULL, "double-destroy (static-pool path): aligned pool");
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
    check_destroyed_single(a, "destroyed static handle");

    free(pool);
}

int main(void) {
    int iters[] = { 1, 10, 1000 };
    for (size_t i = 0; i < sizeof(iters) / sizeof(iters[0]); ++i) {
        test_cycles(iters[i]);
    }
#ifndef AEC_NO_ARRAY
    test_array_cycles(10);
    test_array_cycles(100);
#endif

    test_destroy_null();
    test_destroy_heap();
    test_double_destroy_static();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
