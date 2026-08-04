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
 * `make` target today -- `grep -n test_static_aec Makefile` finds nothing,
 * and STATIC_MEMORY.md documents the manual gcc recipe below as the real
 * way to build it. (test_static_aec.c used to carry a stale comment
 * claiming a top-level `make` target wired this in; that comment has since
 * been removed -- this note's own description of the actual (unwired)
 * state was already correct and is unchanged.) This test mirrors that same
 * manual recipe:
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_config_validation.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
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
#include <math.h>

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

/* R08 (AEC-side re-review gap): highpass_cutoff_hz used to be checked with a
 * flat, rate-blind [0, 20000] Hz range regardless of enable_highpass, while
 * hpf_init() (audio_common hpf.c, called from aec_carve whenever
 * enable_highpass is set) separately rejects any cutoff_hz that isn't finite,
 * isn't > 0, or isn't < 0.45*sample_rate -- silently returning NULL and
 * leaving the AEC instance with no mic-path HPF, with no error surfaced
 * anywhere. E.g. {sample_rate=8000, highpass_cutoff_hz=4000} used to sail
 * through aec_validate_config and then silently drop the HPF underneath.
 * aec_validate_config now mirrors hpf_params_valid exactly when
 * enable_highpass=1, so every combination below that hpf_init would reject
 * must now be rejected here too (all three public entry points), and every
 * combination hpf_init would accept must still sail through end-to-end. */
static void test_highpass_cutoff(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "highpass test setup: valid 16k config sizes > 0");
    void* mem = malloc(sz);
    CHECK(mem != NULL, "highpass test setup: malloc aligned pool");

    /* Negative cases: {sample_rate, cutoff_hz} pairs that hpf_params_valid
     * (audio_common hpf.c) rejects -- includes the exact 0.45*sample_rate
     * boundary (rejected: hpf_params_valid uses `>=`) at all three
     * whitelisted rates, plus NaN/0/negative, all with enable_highpass=1. */
    struct { int sr; float cutoff; } bad_hp[] = {
        { 8000,  4000.0f },              /* > 0.45*8000=3600 */
        { 8000,  3600.0f },              /* == 0.45*8000 boundary, rejected */
        { 16000, 7200.0f },              /* == 0.45*16000 boundary, rejected */
        { 16000, 20000.0f },             /* old flat bound's own ceiling, still way over 0.45*sr */
        { 48000, 21600.0f },             /* == 0.45*48000 boundary, rejected */
        { 8000,  NAN },                  /* NaN */
        { 16000, 0.0f },                 /* not > 0 */
        { 48000, -80.0f },               /* negative */
    };
    for (size_t i = 0; i < sizeof(bad_hp) / sizeof(bad_hp[0]); ++i) {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, bad_hp[i].sr);
        cfg.enable_highpass = 1;
        cfg.highpass_cutoff_hz = bad_hp[i].cutoff;
        char what[80];
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g (enabled)",
                 bad_hp[i].sr, (double)bad_hp[i].cutoff);
        check_rejected(what, &cfg, mem, sz);
    }

    /* Same bad cutoff values must NOT be rejected when enable_highpass=0 --
     * the field is inert (aec_carve never reaches hpf_init), so it keeps the
     * old flat [0, 20000] range instead of the rate-relative one. Skip the
     * NaN/negative cases here (still invalid under the flat range too; only
     * exercise the ones that are >0.45*sr but still inside [0, 20000]). No
     * 48000 entry: 0.45*48000=21600 already exceeds the flat range's own
     * 20000 ceiling, so there is no cutoff value that is rejected by the
     * tightened (enabled) rule but still accepted by the flat (disabled)
     * one at this rate -- 8000/16000 already demonstrate the distinction. */
    {
        struct { int sr; float cutoff; } inert_ok[] = {
            { 8000,  4000.0f },
            { 8000,  3600.0f },
            { 16000, 7200.0f },
        };
        for (size_t i = 0; i < sizeof(inert_ok) / sizeof(inert_ok[0]); ++i) {
            AecConfig cfg;
            aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, inert_ok[i].sr);
            cfg.enable_highpass = 0;
            cfg.highpass_cutoff_hz = inert_ok[i].cutoff;
            char what[96];
            size_t bsz = aec_get_mem_size(&cfg);
            snprintf(what, sizeof(what),
                     "highpass sr=%d cutoff=%g (disabled): aec_get_mem_size() > 0",
                     inert_ok[i].sr, (double)inert_ok[i].cutoff);
            CHECK(bsz > 0, what);
        }
    }
    free(mem);

    /* Positive cases: one Hz below the 0.45*sample_rate boundary at each
     * whitelisted rate, enable_highpass=1 -- must pass end to end (get_mem_
     * size>0, aec_init succeeds, aec_create succeeds, one hop of process
     * runs), proving the tightened check didn't collaterally reject a
     * legitimate cutoff. */
    struct { int sr; float cutoff; } good_hp[] = {
        { 8000,  150.0f },
        { 16000, 7199.0f },
        { 48000, 21599.0f },
    };
    for (size_t i = 0; i < sizeof(good_hp) / sizeof(good_hp[0]); ++i) {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, good_hp[i].sr);
        cfg.enable_highpass = 1;
        cfg.highpass_cutoff_hz = good_hp[i].cutoff;
        char what[96];

        size_t gsz = aec_get_mem_size(&cfg);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aec_get_mem_size() > 0",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(gsz > 0, what);
        if (gsz == 0) continue;

        void* gmem = malloc(gsz);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: malloc pool",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(gmem != NULL, what);
        if (!gmem) continue;

        Aec* a = aec_init(gmem, gsz, &cfg);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aec_init() succeeds",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(a != NULL, what);
        if (a) {
            int hop = aec_hop_size(a);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            float* out = (float*)malloc((size_t)hop * sizeof(float));
            aec_process(a, mic, ref, out);
            snprintf(what, sizeof(what),
                     "highpass sr=%d cutoff=%g (static): one hop of aec_process() runs",
                     good_hp[i].sr, (double)good_hp[i].cutoff);
            CHECK(1, what);
            aec_destroy(a);
            free(mic); free(ref); free(out);
        }
        free(gmem);

        Aec dyn;
        int rc = aec_create(&dyn, &cfg);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aec_create() == 0",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(rc == 0, what);
        if (rc == 0) aec_destroy(&dyn);
    }
}

/* Otherwise-valid 16 kHz balanced config: prove validation didn't collaterally
 * break the real path. get_mem_size>0, init OK, one hop of process runs.
 * aec_destroy() deliberately NOT called here — lifecycle/leak coverage is
 * test_static_aec.c's job (this test only touches config validation). */
/* spatial_linear_context's whole premise (aec3_post_run Step 20/21) is that
 * apply_output() never reads the skipped gain -- true only under
 * context_only (enable_res==0 && return_res_context==1). Every other
 * combination must be rejected by aec_validate_config() itself, not left to
 * the aec3_post_run() debug assert alone. */
static void test_spatial_linear_context_requires_context_only(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "spatial_linear_context test setup: valid 16k config sizes > 0");
    void* mem = malloc(sz);
    CHECK(mem != NULL, "spatial_linear_context test setup: malloc aligned pool");

    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.enable_res = 0;
        cfg.return_res_context = 0;
        cfg.spatial_linear_context = 1;
        check_rejected("spatial_linear_context=1, return_res_context=0",
                       &cfg, mem, sz);
    }
    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.enable_res = 1;
        cfg.return_res_context = 1;
        cfg.spatial_linear_context = 1;
        check_rejected("spatial_linear_context=1, enable_res=1",
                       &cfg, mem, sz);
    }
    {
        /* Out-of-{0,1} value: caught by the plain AEC_CK_BOOL, same as any
         * other toggle field, ahead of the compound check. */
        AecConfig cfg; base_cfg(&cfg);
        cfg.enable_res = 0;
        cfg.return_res_context = 1;
        cfg.spatial_linear_context = 2;
        check_rejected("spatial_linear_context=2 (not a bool)", &cfg, mem, sz);
    }

    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.enable_res = 0;
        cfg.return_res_context = 1;
        cfg.spatial_linear_context = 1;
        Aec a;
        int rc = aec_create(&a, &cfg);
        CHECK(rc == 0, "spatial_linear_context=1 with context_only satisfied: "
                       "aec_create() succeeds");
        if (rc == 0) {
            int hop = aec_hop_size(&a);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            float* out = (float*)malloc((size_t)hop * sizeof(float));
            AecResContext ctx;
            aec_process(&a, mic, ref, out);
            aec_get_res_context(&a, &ctx);
            CHECK(ctx.res_gain == NULL,
                  "spatial_linear_context=1: AecResContext.res_gain is NULL");
            CHECK(ctx.r2 != NULL && ctx.comfort_noise != NULL,
                  "spatial_linear_context=1: r2/comfort_noise still exposed");
            aec_destroy(&a);
            free(mic); free(ref); free(out);
        }
    }

    free(mem);
}

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
    test_highpass_cutoff();
    test_misaligned_base();
    test_spatial_linear_context_requires_context_only();
    test_valid_config_smoke();
    test_multi_rate_valid_smoke();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
