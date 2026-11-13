/* test_handle_ready.c -- a handle that is not ready is refused by every public entry point that takes an Aec (aec.h).
 *
 * "Not ready" = a zero-filled Aec that was never built (static storage, calloc, a zeroed pool cast to Aec*), the handle a failed
 * aec_create() left behind, and a destroyed handle; single-channel and array, heap (aec_create) and caller pool (aec_init). A
 * live handle of each kind is the positive control: the same calls must not all be refused. The list of entries is the block
 * between the BEGIN / END markers in check_refused(); check_handle_entries.py compares it with the prototypes in aec.h.
 *
 * Run from c_impl/:  make test-handle-ready */
#include "aec.h"
#include "aec_internal.h"
#include "aec_pool_test_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_MAX 1025
#define HOP_MAX 512
#define ARRAY_CH 4

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static int view_zeroed_but_size(const void* view, size_t size) {
    const unsigned char* b = (const unsigned char*)view;
    size_t n, declared;
    memcpy(&declared, view, sizeof declared);
    if (declared != size) return 0;
    for (n = sizeof declared; n < size; ++n) if (b[n]) return 0;
    return 1;
}

static int all_zero(const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    size_t i;
    for (i = 0; i < n; ++i) if (b[i]) return 0;
    return 1;
}

/* Every call answers the refusal its prototype documents. aec_destroy() and aec_reset() are the repeatable no-ops. */
static void check_refused(Aec* a, const char* what) {
    static float mic[HOP_MAX * ARRAY_CH], ref[HOP_MAX], out[HOP_MAX];
    static Complex W[ARRAY_CH * NB_MAX], spec[NB_MAX];
    AecBeam beam = { W, NULL };
    AecResidualView rv; AecLinearView lv; AecDebugStatus ds; AecStageTiming tm;
    static float sentinel;
    const float* win = &sentinel; int len = 77;
    memset(&rv, 0x5A, sizeof rv); rv.struct_size = sizeof rv;      /* garbage in: a refusal leaves a zeroed record */
    memset(&lv, 0x5A, sizeof lv); lv.struct_size = sizeof lv;
    memset(&ds, 0x5A, sizeof ds);
    memset(&tm, 0x5A, sizeof tm);
#ifndef AEC_NO_ARRAY
    AecArrayView av; AecArrayTiming at;
    memset(&av, 0x5A, sizeof av); av.struct_size = sizeof av;
    memset(&at, 0x5A, sizeof at);
#endif
    /* BEGIN entries */
    CHECK(aec_process(a, mic, ref, out) == AEC_ERR_ARG, "%s: aec_process", what);
    CHECK(aec_process_linear(a, mic, ref) == AEC_ERR_ARG, "%s: aec_process_linear", what);
    CHECK(aec_process_residual(a, &beam) == AEC_ERR_ARG, "%s: aec_process_residual", what);
    CHECK(aec_process_linear_shared_far_from(a, mic, ref, NULL) == AEC_ERR_ARG, "%s: aec_process_linear_shared_far_from", what);
    CHECK(aec_get_linear(a, &lv) == AEC_ERR_ARG && view_zeroed_but_size(&lv, sizeof lv), "%s: aec_get_linear", what);
    CHECK(aec_get_residual(a, &rv) == AEC_ERR_ARG && view_zeroed_but_size(&rv, sizeof rv), "%s: aec_get_residual", what);
    CHECK(aec_apply_gain(a, NULL, spec) == AEC_ERR_ARG, "%s: aec_apply_gain", what);
    CHECK(aec_apply_cng(a, NULL, spec) == AEC_ERR_ARG, "%s: aec_apply_cng", what);
    CHECK(aec_set_level(a, AEC_LEVEL_MILD) == -1, "%s: aec_set_level", what);
    CHECK(aec_apply_external_realign(a, 3) == -1, "%s: aec_apply_external_realign", what);
    CHECK(aec_synthesis_window(a, &win, &len) == -1 && win == &sentinel && len == 77, "%s: aec_synthesis_window", what);
    CHECK(aec_post_reset_serial(a) == 0u, "%s: aec_post_reset_serial", what);
    CHECK(aec_far_fft_real_compute_count(a) == 0, "%s: aec_far_fft_real_compute_count", what);
    CHECK(aec_linear_is_cancelling(a) == 0, "%s: aec_linear_is_cancelling", what);
    CHECK(aec_hop_size(a) == 0, "%s: aec_hop_size", what);
    aec_debug_status(a, &ds);
    CHECK(all_zero(&ds, sizeof ds), "%s: aec_debug_status zeroes the record", what);
    aec_get_last_timing(a, &tm);
    CHECK(all_zero(&tm, sizeof tm), "%s: aec_get_last_timing zeroes the record", what);
#ifndef AEC_NO_ARRAY
    CHECK(aec_get_array(a, &av) == AEC_ERR_ARG && view_zeroed_but_size(&av, sizeof av), "%s: aec_get_array", what);
    CHECK(aec_array_realign_warm_lane_count(a) == 0, "%s: aec_array_realign_warm_lane_count", what);
    CHECK(aec_array_realign_soft_lane_count(a) == 0, "%s: aec_array_realign_soft_lane_count", what);
    CHECK(aec_array_matched_filter_count(a) == 0, "%s: aec_array_matched_filter_count", what);
    CHECK(aec_array_linear_aec_count(a) == 0, "%s: aec_array_linear_aec_count", what);
    aec_array_get_timing(a, &at);
    CHECK(all_zero(&at, sizeof at), "%s: aec_array_get_timing zeroes the record", what);
#endif
    aec_reset(a);
    aec_destroy(a);
    aec_destroy(a);
    /* END entries */
    CHECK(aec_hop_size(a) == 0, "%s: still refused after reset and repeated destroy", what);
}

/* The control: a ready handle is not refused by the calls that carry data. */
static void check_live(Aec* a, const char* what) {
    static float mic[HOP_MAX * ARRAY_CH], ref[HOP_MAX], out[HOP_MAX];
    const int hop = aec_hop_size(a);
    CHECK(hop > 0 && hop <= HOP_MAX, "%s: live handle has a hop size", what);
    if (hop <= 0 || hop > HOP_MAX) return;
    if (a->array) {
        CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "%s: live aec_process_linear", what);
    } else {
        CHECK(aec_process(a, mic, ref, out) == AEC_OK, "%s: live aec_process", what);
    }
    CHECK(aec_far_fft_real_compute_count(a) >= 0 && aec_linear_is_cancelling(a) >= 0, "%s: live counters", what);
}

static void invalid_cfg(AecConfig* cfg, int array) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, 16000);
    if (array) {
        cfg->n_mics = ARRAY_CH;
        cfg->residual = AEC_RESIDUAL_OFF;       /* an array needs a residual mode */
    } else {
        cfg->filter_length = 0;
    }
}

static void valid_cfg(AecConfig* cfg, int array) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, 16000);
    if (array) {
        cfg->n_mics = ARRAY_CH;
        cfg->residual = AEC_RESIDUAL_ESTIMATE;
    }
}

static Aec g_static_zero;

static void test_never_built(void) {
    Aec stack_zero;
    Aec* heap_zero = (Aec*)calloc(1, sizeof(Aec));
    void* pool_zero = calloc(1, 4096);          /* a caller pool that was never given to aec_init() */
    memset(&stack_zero, 0, sizeof stack_zero);
    check_refused(&g_static_zero, "zero Aec in static storage");
    check_refused(&stack_zero, "zero Aec on the stack");
    CHECK(heap_zero != NULL && pool_zero != NULL, "allocations");
    if (heap_zero) { check_refused(heap_zero, "calloc'd Aec"); free(heap_zero); }
    if (pool_zero) { check_refused((Aec*)pool_zero, "zero pool cast to Aec*"); free(pool_zero); }
}

static void test_failed_create(int array) {
    AecConfig bad, good;
    Aec a;
    const char* kind = array ? "array" : "single-channel";
    char what[96];
    invalid_cfg(&bad, array);
    valid_cfg(&good, array);
    memset(&a, 0, sizeof a);
    CHECK(aec_create(&a, &bad) == -1, "%s: aec_create(invalid cfg) fails", kind);
    snprintf(what, sizeof what, "%s: handle left by a failed aec_create", kind);
    check_refused(&a, what);
    /* a failed re-create over a destroyed handle leaves it refused, and a good create then builds it */
    CHECK(aec_create(&a, &good) == 0, "%s: aec_create(valid cfg)", kind);
    snprintf(what, sizeof what, "%s: created", kind);
    check_live(&a, what);
    aec_destroy(&a);
    CHECK(aec_create(&a, &bad) == -1, "%s: aec_create(invalid cfg) over a destroyed handle fails", kind);
    snprintf(what, sizeof what, "%s: failed re-create over a destroyed handle", kind);
    check_refused(&a, what);
    CHECK(aec_create(&a, &good) == 0, "%s: a refused handle can be created again", kind);
    check_live(&a, "re-created");
    aec_destroy(&a);
    snprintf(what, sizeof what, "%s: destroyed heap handle", kind);
    check_refused(&a, what);
}

static void test_destroyed_pool(int array) {
    AecConfig cfg;
    size_t sz;
    void* pool;
    Aec* a;
    const char* kind = array ? "array" : "single-channel";
    char what[96];
    valid_cfg(&cfg, array);
    sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "%s: pool size", kind);
    pool = sz ? aec_test_pool_alloc(sz) : NULL;
    CHECK(pool != NULL, "%s: pool", kind);
    if (!pool) return;
    memset(pool, 0xA5, sz);
    CHECK(aec_init(pool, sz - 1u, &cfg) == NULL, "%s: undersized pool is refused", kind);
    a = aec_init(pool, sz, &cfg);
    CHECK(a != NULL, "%s: aec_init", kind);
    if (a) {
        snprintf(what, sizeof what, "%s: static", kind);
        check_live(a, what);
        aec_destroy(a);
        snprintf(what, sizeof what, "%s: destroyed static handle", kind);
        check_refused(a, what);
    }
    free(pool);
}

int main(void) {
    test_never_built();
    test_failed_create(0);
    test_destroyed_pool(0);
#ifndef AEC_NO_ARRAY
    test_failed_create(1);
    test_destroyed_pool(1);
#endif
    printf("%ld checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
