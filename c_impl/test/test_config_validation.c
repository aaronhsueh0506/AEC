/* test_config_validation.c — negative-input tests for the F05/F07 review
 * remediation: aec_validate_config() (aec.c) is the single source of truth
 * for AecConfig bounds-checking, wired into all three public entry points
 * that derive sizes/state from a config. This test proves each entry point
 * actually rejects a bad config through its OWN existing failure
 * convention — no release-path asserts:
 *   - aec_get_mem_size(cfg)        -> 0
 *   - aec_init(mem, mem_size, cfg) -> NULL
 *   - aec_create(&a, cfg)          -> nonzero
 * plus the F07 misaligned-pool-base guard in aec_init(), and a smoke test
 * that the otherwise-valid 16 kHz balanced config still works end to end
 * (byte-identical-output gate lives elsewhere; this is just "didn't break").
 *
 * Build (standalone, from c_impl/). NOTE: neither this repo's Makefile nor
 * test_static_aec.c (c_impl/test_static_aec.c) is actually wired into a
 * `make` target today — despite a stale comment in test_static_aec.c
 * claiming otherwise, `grep -n test_static_aec Makefile` finds nothing, and
 * STATIC_MEMORY.md documents the manual gcc recipe below as the real way to
 * build it. This test mirrors that same manual recipe:
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_config_validation.c $(find src -name '*.c') \
 *       ../../audio_common/bin/kiss/libaudio_common.a -lm \
 *       -o bin/test_config_validation
 * Run:
 *   ./bin/test_config_validation
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

static void base_cfg(AecConfig* cfg) {
    aec_config_from_preset(cfg, AEC_PRESET_BALANCED, 16000);
}

/* Exercise all three public entry points against one deliberately-invalid
 * config, using an already-sized-and-aligned buffer (from a KNOWN-valid
 * config) so aec_init has something real to reject against even though
 * aec_get_mem_size(bad_cfg) itself is 0. */
static void check_rejected(const char* what, const AecConfig* bad_cfg,
                            void* aligned_mem, size_t aligned_mem_sz) {
    char label[192];

    snprintf(label, sizeof(label), "%s: aec_get_mem_size() == 0", what);
    CHECK(aec_get_mem_size(bad_cfg) == 0, label);

    snprintf(label, sizeof(label), "%s: aec_init() == NULL", what);
    CHECK(aec_init(aligned_mem, aligned_mem_sz, bad_cfg) == NULL, label);

    snprintf(label, sizeof(label), "%s: aec_create() != 0", what);
    {
        Aec a;
        CHECK(aec_create(&a, bad_cfg) != 0, label);
    }
}

static void test_sample_rate(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "sample_rate test setup: valid 16k config sizes > 0");
    void* mem = malloc(sz);
    CHECK(mem != NULL, "sample_rate test setup: malloc aligned pool");

    /* M5 (multi-rate campaign, review F01): the whitelist widened from
     * {16000} to {8000, 16000, 48000} — 8000/48000 moved out of this
     * negative list (see test_multi_rate_valid_smoke below for their
     * positive coverage). 44100 has no per-rate table and stays rejected,
     * alongside the always-invalid 0/-1/INT_MAX. */
    int bad_rates[] = { 0, -1, 44100, INT_MAX };
    for (size_t i = 0; i < sizeof(bad_rates) / sizeof(bad_rates[0]); ++i) {
        AecConfig cfg; base_cfg(&cfg);
        cfg.sample_rate = bad_rates[i];
        char what[64];
        snprintf(what, sizeof(what), "sample_rate=%d", bad_rates[i]);
        check_rejected(what, &cfg, mem, sz);
    }
    free(mem);
}

static void test_filter_length(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    void* mem = malloc(sz);
    CHECK(mem != NULL, "filter_length test setup: malloc aligned pool");

    int bad_lengths[] = { 0, -1, INT_MAX };
    for (size_t i = 0; i < sizeof(bad_lengths) / sizeof(bad_lengths[0]); ++i) {
        AecConfig cfg; base_cfg(&cfg);
        cfg.filter_length = bad_lengths[i];
        char what[64];
        snprintf(what, sizeof(what), "filter_length=%d", bad_lengths[i]);
        check_rejected(what, &cfg, mem, sz);
    }
    free(mem);
}

/* F07: aec_init() must reject a misaligned pool base BEFORE writing
 * anything to it. Prefill a raw buffer with a 0xA5 sentinel, hand aec_init
 * a deliberately-misaligned pointer +1..+15 into it, and assert (a) NULL
 * comes back and (b) not a single byte of the sentinel region changed. */
static void test_misaligned_base(void) {
    AecConfig cfg; base_cfg(&cfg);
    size_t sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "misaligned-base test setup: valid 16k config sizes > 0");

    size_t raw_sz = sz + 32;   /* headroom to both 16-align up and shift 1..15 */
    unsigned char* raw = (unsigned char*)malloc(raw_sz);
    CHECK(raw != NULL, "misaligned-base test setup: malloc raw buffer");
    if (!raw) return;
    memset(raw, 0xA5, raw_sz);

    uintptr_t base = (uintptr_t)raw;
    uintptr_t aligned = (base + 15u) & ~(uintptr_t)15u;
    unsigned char* aligned_ptr = (unsigned char*)aligned;
    CHECK((size_t)((raw + raw_sz) - aligned_ptr) >= sz,
          "misaligned-base test setup: aligned pointer has room for sz bytes");

    unsigned char* snapshot = (unsigned char*)malloc(raw_sz);
    for (int off = 1; off < 16; ++off) {
        unsigned char* mis = aligned_ptr + off;
        memcpy(snapshot, raw, raw_sz);   /* re-snapshot: previous iteration wrote nothing, but be explicit */

        Aec* a = aec_init(mis, sz, &cfg);
        char label[64];
        snprintf(label, sizeof(label), "misaligned base +%d rejected (aec_init == NULL)", off);
        CHECK(a == NULL, label);

        snprintf(label, sizeof(label), "misaligned base +%d: no byte written before rejection", off);
        CHECK(memcmp(raw, snapshot, raw_sz) == 0, label);
    }
    free(snapshot);
    free(raw);
}

/* Otherwise-valid 16 kHz balanced config: prove validation didn't collaterally
 * break the real path. get_mem_size>0, init OK, one hop of process runs.
 * aec_destroy() deliberately NOT called here — lifecycle/leak coverage is
 * test_static_aec.c's job (this test only touches config validation). */
static void test_valid_config_smoke(void) {
    AecConfig cfg; base_cfg(&cfg);

    size_t sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "valid 16k balanced: aec_get_mem_size() > 0");

    void* mem = malloc(sz);
    CHECK(mem != NULL, "valid 16k balanced: malloc pool");
    if (!mem) return;

    Aec* a = aec_init(mem, sz, &cfg);
    CHECK(a != NULL, "valid 16k balanced: aec_init() succeeds");

    if (a) {
        int hop = aec_hop_size(a);
        float* mic = (float*)calloc((size_t)hop, sizeof(float));
        float* ref = (float*)calloc((size_t)hop, sizeof(float));
        float* out = (float*)malloc((size_t)hop * sizeof(float));
        aec_process(a, mic, ref, out);
        CHECK(1, "valid 16k balanced: one hop of aec_process() runs");
        free(mic); free(ref); free(out);
    }
    free(mem);
}

/* M5: positive smoke at all three whitelisted rates -- get_mem_size>0,
 * aec_init/aec_create both succeed, one hop of aec_process runs clean at
 * 8000/16000/48000. Mirrors test_valid_config_smoke's 16k-only shape;
 * kept as a separate function since that one is referenced by name
 * elsewhere in prior review notes and its "didn't collaterally break"
 * framing is 16k-specific. */
static void test_multi_rate_valid_smoke(void) {
    int rates[] = { 8000, 16000, 48000 };
    for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
        int sr = rates[i];
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        char what[64];

        snprintf(what, sizeof(what), "sr=%d: aec_is_valid_sample_rate()", sr);
        CHECK(aec_is_valid_sample_rate(sr) == 1, what);

        size_t sz = aec_get_mem_size(&cfg);
        snprintf(what, sizeof(what), "sr=%d: aec_get_mem_size() > 0", sr);
        CHECK(sz > 0, what);

        void* mem = malloc(sz);
        snprintf(what, sizeof(what), "sr=%d: malloc pool", sr);
        CHECK(mem != NULL, what);
        if (!mem) continue;

        Aec* a = aec_init(mem, sz, &cfg);
        snprintf(what, sizeof(what), "sr=%d: aec_init() succeeds", sr);
        CHECK(a != NULL, what);

        if (a) {
            int hop = aec_hop_size(a);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            float* out = (float*)malloc((size_t)hop * sizeof(float));
            aec_process(a, mic, ref, out);
            snprintf(what, sizeof(what), "sr=%d (static): one hop of aec_process() runs", sr);
            CHECK(1, what);
            aec_destroy(a);
            free(mic); free(ref); free(out);
        }
        free(mem);

        /* Same rate through the dynamic (aec_create) constructor too --
         * heap and pool share aec_carve (arena-fication), but this proves
         * the aec_create entry point independently accepts the rate. */
        Aec dyn;
        int rc = aec_create(&dyn, &cfg);
        snprintf(what, sizeof(what), "sr=%d (dynamic): aec_create() == 0", sr);
        CHECK(rc == 0, what);
        if (rc == 0) {
            int hop = aec_hop_size(&dyn);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            float* out = (float*)malloc((size_t)hop * sizeof(float));
            aec_process(&dyn, mic, ref, out);
            snprintf(what, sizeof(what), "sr=%d (dynamic): one hop of aec_process() runs", sr);
            CHECK(1, what);
            aec_destroy(&dyn);
            free(mic); free(ref); free(out);
        }
    }
}

int main(void) {
    test_sample_rate();
    test_filter_length();
    test_misaligned_base();
    test_valid_config_smoke();
    test_multi_rate_valid_smoke();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
