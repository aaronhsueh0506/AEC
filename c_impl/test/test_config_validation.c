/* test_config_validation.c -- negative-input tests: aec_resolve() / aec_resolve_tuned() (aec.c) is the single source of truth
 * for AecConfig and AecTuning bounds-checking, wired into all three public entry points that derive sizes/state from a
 * config. This test proves each entry point rejects a bad config through its own failure convention (no release-path
 * asserts):
 *   - aec_get_mem_size(cfg)        -> 0
 *   - aec_init(mem, mem_size, cfg) -> NULL
 *   - aec_create(cfg)              -> NULL
 * A bad INTERNAL tuning value reaches the same three paths through the aec_testing_* hooks (check_rejected_tuned).
 * plus the misaligned-pool-base guard in aec_init(), and a smoke test that the otherwise-valid 16 kHz balanced config
 * still works end to end.
 *
 * Build (standalone, from c_impl/; `make test-config-validation` does the same):
 *   make -C ../../audiolib/common lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       test/test_config_validation.c $(find src -name '*.c') \
 *       $(make -s -C ../../audiolib/common print-lib-path) -lm \
 *       -o bin/test_config_validation
 * Run:
 *   ./bin/test_config_validation
 */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"
#include "aec_pool_test_util.h"
#include "res_gain_of.h"
/* AECB_RATE_TABLE / aecb_rate_cfg(): the tuning table carries its
 * own {n_bins, fft_size, block_size, hop_size} alongside ~40 tuning
 * constants. test_signal_grid_resolver() pins it against the shared
 * resolver so the instance can never be split across two grids. */
#include "aec_balanced_config.h"
#include "aec_ref_tables.h"   /* the twin's full tables: the oracle for what aec_init decodes into the pool */
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
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, 16000);
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
        Aec *a = NULL;
        CHECK((a = aec_create(bad_cfg)) == NULL, label);
    }
}

/* The same three paths for a bad tuning value (the hooks take a caller-built AecTuning). */
static void check_rejected_tuned(const char* what, const AecConfig* bad_cfg, const AecTuning* bad_tun,
                                 void* aligned_mem, size_t aligned_mem_sz) {
    char label[192];

    snprintf(label, sizeof(label), "%s: aec_testing_get_mem_size() == 0", what);
    CHECK(aec_testing_get_mem_size(bad_cfg, bad_tun) == 0, label);

    snprintf(label, sizeof(label), "%s: aec_testing_init() == NULL", what);
    CHECK(aec_testing_init(aligned_mem, aligned_mem_sz, bad_cfg, bad_tun) == NULL, label);

    snprintf(label, sizeof(label), "%s: aec_testing_create() == NULL", what);
    CHECK(aec_testing_create(bad_cfg, bad_tun) == NULL, label);

    snprintf(label, sizeof(label), "%s: aec_testing_tuning_valid() == 0 or the config is the fault", what);
    CHECK(aec_testing_tuning_valid(bad_tun, bad_cfg) == 0 || aec_get_mem_size(bad_cfg) == 0, label);
}

static void test_sample_rate(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "sample_rate test setup: valid 16k config sizes > 0");
    void* mem = aec_test_pool_alloc(sz);
    CHECK(mem != NULL, "sample_rate test setup: aligned pool");

    /* The whitelist is {8000, 16000, 48000} (see test_multi_rate_valid_smoke
     * below for the positive coverage). 44100 has no per-rate table and is
     * rejected, alongside the always-invalid 0/-1/INT_MAX. */
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
    void* mem = aec_test_pool_alloc(sz);
    CHECK(mem != NULL, "filter_length test setup: aligned pool");

    int bad_lengths[] = { 0, -1, INT_MAX };
    for (size_t i = 0; i < sizeof(bad_lengths) / sizeof(bad_lengths[0]); ++i) {
        AecTestCfg cfg; base_cfg(&cfg.c); aec_tcfg_refresh(&cfg);
        cfg.t.filter_length = bad_lengths[i];
        char what[64];
        snprintf(what, sizeof(what), "filter_length=%d", bad_lengths[i]);
        check_rejected_tuned(what, &cfg.c, &cfg.t, mem, sz);
    }
    free(mem);
}

/* The config factories take the rate as a plain int. An unsupported rate must come back as an unusable config (fft_size 0,
 * and a derived tuning with filter_length 0) that all three entry points reject, never as a value computed from an overflowing int product: before
 * the guard, sr * 64 overflowed int for sr >= 2^25 (undefined; the resulting filter_length depended on the optimisation
 * level) and a rate such as 44100 kept a plausible filter_length next to fft_size 0. A supported rate must keep its factory values exactly. */
static void test_config_factories_unsupported_rate(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    void* mem = aec_test_pool_alloc(sz);
    CHECK(sz > 0 && mem != NULL, "factory test setup: valid 16k config and aligned pool");
    if (!mem) return;

    const int bad_rates[] = { INT_MIN, -1, 0, 1, 44100, 33554431, 33554432, 100000000, INT_MAX };
    for (size_t i = 0; i < sizeof(bad_rates) / sizeof(bad_rates[0]); ++i) {
        char what[96];
        AecConfig cfg;
        AecTuning tn;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, bad_rates[i]);
        aec_tuning_defaults(&tn, &cfg);
        snprintf(what, sizeof(what), "aec_config_from_level(balanced, sr=%d): fft_size 0, filter_length 0", bad_rates[i]);
        CHECK(cfg.fft_size == 0 && tn.filter_length == 0, what);
        snprintf(what, sizeof(what), "aec_config_from_level(balanced, sr=%d)", bad_rates[i]);
        check_rejected(what, &cfg, mem, sz);

        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, bad_rates[i]);
        aec_tuning_defaults(&tn, &cfg);
        snprintf(what, sizeof(what), "aec_config_from_level(sr=%d): fft_size 0, filter_length 0", bad_rates[i]);
        CHECK(cfg.fft_size == 0 && tn.filter_length == 0, what);
        snprintf(what, sizeof(what), "aec_config_from_level(sr=%d)", bad_rates[i]);
        check_rejected(what, &cfg, mem, sz);
    }
    free(mem);

    aec_config_from_level(NULL, AEC_LEVEL_BALANCED, 16000);       /* a NULL cfg is ignored (a crash fails the run) */
    CHECK(1, "aec_config_from_level(NULL) returns");

    /* Supported rates: 52 ms below 44.1 kHz, 64 ms at 48 kHz. */
    const struct { int sr, fft, filter_length; } good[] = {
        { 8000, 256, 416 }, { 16000, 256, 832 }, { 48000, 1024, 3072 },
    };
    for (size_t i = 0; i < sizeof(good) / sizeof(good[0]); ++i) {
        char what[96];
        AecConfig cfg;
        AecTuning tn;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, good[i].sr);
        aec_tuning_defaults(&tn, &cfg);
        snprintf(what, sizeof(what), "aec_config_from_level(sr=%d): fft_size %d, derived filter_length %d unchanged",
                 good[i].sr, good[i].fft, good[i].filter_length);
        CHECK(cfg.fft_size == good[i].fft && tn.filter_length == good[i].filter_length, what);
        snprintf(what, sizeof(what), "aec_config_from_level(sr=%d) sizes > 0", good[i].sr);
        CHECK(aec_get_mem_size(&cfg) > 0, what);
    }
}

/* aec_init() must reject a misaligned pool base BEFORE writing
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

/* highpass_cutoff_hz must be checked against hpf_init()'s rate-relative rule, not a flat [0, 20000]
 * Hz range: hpf_init() (audiolib hpf.c, called from aec_carve whenever enable_highpass is set)
 * rejects any cutoff_hz that is not finite, not > 0, or not < 0.45*sample_rate, silently returning
 * NULL and leaving the instance without a mic-path HPF. E.g. {sample_rate=8000,
 * highpass_cutoff_hz=4000} would pass a flat range. aec_tuning_valid mirrors hpf_params_valid
 * when enable_highpass=1, so every combination hpf_init rejects is rejected here too (all three
 * entry points) and every combination it accepts still works end to end. */
static void test_highpass_cutoff(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "highpass test setup: valid 16k config sizes > 0");
    void* mem = aec_test_pool_alloc(sz);
    CHECK(mem != NULL, "highpass test setup: aligned pool");

    /* Negative cases: {sample_rate, cutoff_hz} pairs that hpf_params_valid
     * (audiolib hpf.c) rejects -- includes the exact 0.45*sample_rate
     * boundary (rejected: hpf_params_valid uses `>=`) at all three
     * whitelisted rates, plus NaN/0/negative, all with enable_highpass=1. */
    struct { int sr; float cutoff; } bad_hp[] = {
        { 8000,  4000.0f },              /* > 0.45*8000=3600 */
        { 8000,  3600.0f },              /* == 0.45*8000 boundary, rejected */
        { 16000, 7200.0f },              /* == 0.45*16000 boundary, rejected */
        { 16000, 20000.0f },             /* flat ceiling, still way over 0.45*sr */
        { 48000, 21600.0f },             /* == 0.45*48000 boundary, rejected */
        { 8000,  NAN },                  /* NaN */
        { 16000, 0.0f },                 /* not > 0 */
        { 48000, -80.0f },               /* negative */
    };
    for (size_t i = 0; i < sizeof(bad_hp) / sizeof(bad_hp[0]); ++i) {
        AecTestCfg cfg;
        aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, bad_hp[i].sr);
        cfg.t.enable_highpass = 1;
        cfg.t.highpass_cutoff_hz = bad_hp[i].cutoff;
        char what[80];
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g (enabled)",
                 bad_hp[i].sr, (double)bad_hp[i].cutoff);
        check_rejected_tuned(what, &cfg.c, &cfg.t, mem, sz);
    }

    /* The same bad cutoff values must NOT be rejected when enable_highpass=0: the field is inert
     * (aec_carve never reaches hpf_init), so it keeps the flat [0, 20000] range. Only values above
     * 0.45*sr but inside [0, 20000] are exercised. No 48000 entry: 0.45*48000=21600 already exceeds
     * the flat 20000 ceiling, so no cutoff is rejected by the enabled rule but accepted by the flat
     * one at that rate. */
    {
        struct { int sr; float cutoff; } inert_ok[] = {
            { 8000,  4000.0f },
            { 8000,  3600.0f },
            { 16000, 7200.0f },
        };
        for (size_t i = 0; i < sizeof(inert_ok) / sizeof(inert_ok[0]); ++i) {
            AecTestCfg cfg;
            aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, inert_ok[i].sr);
            cfg.t.enable_highpass = 0;
            cfg.t.highpass_cutoff_hz = inert_ok[i].cutoff;
            char what[96];
            size_t bsz = aec_tcfg_mem_size(&cfg);
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
        AecTestCfg cfg;
        aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, good_hp[i].sr);
        cfg.t.enable_highpass = 1;
        cfg.t.highpass_cutoff_hz = good_hp[i].cutoff;
        char what[96];

        size_t gsz = aec_tcfg_mem_size(&cfg);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aec_get_mem_size() > 0",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(gsz > 0, what);
        if (gsz == 0) continue;

        void* gmem = aec_test_pool_alloc(gsz);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aligned pool",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(gmem != NULL, what);
        if (!gmem) continue;

        Aec* a = aec_tcfg_init(gmem, gsz, &cfg);
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

        Aec *dyn = NULL;
        int rc = ((dyn = aec_tcfg_create(&cfg)) ? 0 : -1);
        snprintf(what, sizeof(what), "highpass sr=%d cutoff=%g: aec_create() == 0",
                 good_hp[i].sr, (double)good_hp[i].cutoff);
        CHECK(rc == 0, what);
        if (rc == 0) aec_destroy(dyn);
    }
}

/* Otherwise-valid 16 kHz balanced config: get_mem_size>0, init OK, one hop of process runs, so
 * validation did not break the real path. aec_destroy() is deliberately not called here:
 * lifecycle/leak coverage is test_static_aec.c's job. */
/* AecConfig.residual is range-checked ([OFF, FULL]); every in-range mode builds, and what the residual view exposes
 * follows the mode: OFF has no residual stage at all, STATE has the linear half and the rendering base but neither r2, the
 * comfort noise nor the gain, ESTIMATE adds r2 and the comfort noise, FULL adds the gain. */
static void test_residual_mode_range_and_exposure(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    void* mem = aec_test_pool_alloc(sz);
    CHECK(sz > 0 && mem != NULL, "residual mode test setup: pool allocated");

    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.residual = (AecResidualMode)(AEC_RESIDUAL_FULL + 1);
        check_rejected("residual=FULL+1 (no such mode)", &cfg, mem, sz);
    }
    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.residual = (AecResidualMode)-1;
        check_rejected("residual=-1", &cfg, mem, sz);
    }
    {
        AecConfig cfg; base_cfg(&cfg);
        cfg.residual = (AecResidualMode)7;
        check_rejected("residual=7", &cfg, mem, sz);
    }
    {
        static const AecResidualMode modes[4] = { AEC_RESIDUAL_OFF, AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE,
                                                  AEC_RESIDUAL_FULL };
        static const char* const names[4] = { "OFF", "STATE", "ESTIMATE", "FULL" };
        int m;
        for (m = 0; m < 4; ++m) {
            AecConfig cfg; base_cfg(&cfg);
            cfg.residual = modes[m];
            Aec *a = NULL;
            char what[160];
            int rc = ((a = aec_create(&cfg)) ? 0 : -1);
            snprintf(what, sizeof(what), "residual=%s: aec_create() succeeds", names[m]);
            CHECK(rc == 0, what);
            if (rc != 0) continue;
            int hop = aec_hop_size(a);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            AecResidualView rv;
            memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
            CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "residual mode: aec_process_linear() runs");
            if (modes[m] == AEC_RESIDUAL_OFF) {
                snprintf(what, sizeof(what), "residual=OFF: aec_process_residual() and aec_get_residual() are AEC_ERR_MODE");
                CHECK(aec_process_residual(a, NULL) == AEC_ERR_MODE && aec_get_residual(a, &rv) == AEC_ERR_MODE, what);
            } else {
                CHECK(aec_process_residual(a, NULL) == AEC_OK, "residual mode: aec_process_residual() runs");
                CHECK(aec_get_residual(a, &rv) == AEC_OK, "residual mode: aec_get_residual() succeeds");
                snprintf(what, sizeof(what), "residual=%s: the linear half and the rendering base are exposed", names[m]);
                CHECK(rv.formed_hop != NULL && rv.error_spec != NULL && rv.echo_spec != NULL && rv.near_spec != NULL &&
                      rv.far_spec != NULL && rv.base_spec != NULL, what);
                snprintf(what, sizeof(what), "residual=%s: r2 and comfort_noise are %s", names[m],
                         modes[m] == AEC_RESIDUAL_STATE ? "NULL" : "exposed");
                CHECK((rv.r2 != NULL) == (modes[m] != AEC_RESIDUAL_STATE) &&
                      (rv.comfort_noise != NULL) == (modes[m] != AEC_RESIDUAL_STATE), what);
                snprintf(what, sizeof(what), "residual=%s: res_gain is %s", names[m],
                         modes[m] == AEC_RESIDUAL_FULL ? "exposed" : "NULL");
                CHECK((rv.res_gain != NULL) == (modes[m] == AEC_RESIDUAL_FULL) &&
                      (res_gain_of(a) != NULL) == (modes[m] == AEC_RESIDUAL_FULL), what);
            }
            aec_destroy(a);
            free(mic); free(ref);
        }
    }
    free(mem);
}

static void test_valid_config_smoke(void) {
    AecConfig cfg; base_cfg(&cfg);

    size_t sz = aec_get_mem_size(&cfg);
    CHECK(sz > 0, "valid 16k balanced: aec_get_mem_size() > 0");

    void* mem = aec_test_pool_alloc(sz);
    CHECK(mem != NULL, "valid 16k balanced: aligned pool");
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

/* Positive smoke at all three whitelisted rates: get_mem_size>0, aec_init/aec_create succeed, one
 * hop of aec_process runs clean at 8000/16000/48000. Same shape as test_valid_config_smoke, which
 * is 16k-only. */
static void test_multi_rate_valid_smoke(void) {
    int rates[] = { 8000, 16000, 48000 };
    for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
        int sr = rates[i];
        AecConfig cfg;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
        char what[64];

        snprintf(what, sizeof(what), "sr=%d: aec_is_valid_sample_rate()", sr);
        CHECK(aec_is_valid_sample_rate(sr) == 1, what);

        size_t sz = aec_get_mem_size(&cfg);
        snprintf(what, sizeof(what), "sr=%d: aec_get_mem_size() > 0", sr);
        CHECK(sz > 0, what);

        void* mem = aec_test_pool_alloc(sz);
        snprintf(what, sizeof(what), "sr=%d: aligned pool", sr);
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
        Aec *dyn = NULL;
        int rc = ((dyn = aec_create(&cfg)) ? 0 : -1);
        snprintf(what, sizeof(what), "sr=%d (dynamic): aec_create() == 0", sr);
        CHECK(rc == 0, what);
        if (rc == 0) {
            int hop = aec_hop_size(dyn);
            float* mic = (float*)calloc((size_t)hop, sizeof(float));
            float* ref = (float*)calloc((size_t)hop, sizeof(float));
            float* out = (float*)malloc((size_t)hop * sizeof(float));
            aec_process(dyn, mic, ref, out);
            snprintf(what, sizeof(what), "sr=%d (dynamic): one hop of aec_process() runs", sr);
            CHECK(1, what);
            aec_destroy(dyn);
            free(mic); free(ref); free(out);
        }
    }
}

/* The three direct init entry points below sit under aec_create()'s validator, so a bad sample_rate
 * cannot reach them through the normal path, but they are public API and sample_rate is
 * load-bearing there: it retimes alpha_r/initial_state_threshold_hops and the saturation
 * attack/release pair. An unchecked sample_rate=0 would make hop_seconds infinite and powf() return
 * 0 or NaN, an EMA that never adapts or a filter full of NaN. The filters are pool-first
 * (pbfdaf_init_static / pbfdkf_init_static), so each guard must also leave the instance and the
 * caller-owned pool untouched when it refuses. */
static void test_direct_init_rejects_bad_rate(void) {
    PBFDAF f;
    PBFDAF f_before;
    PBFDKF k;
    PBFDKF k_before;
    Saturation s;

    {
        size_t af_bytes = pbfdaf_get_mem_size(256, 4, 128, 0);
        size_t kf_bytes = pbfdkf_get_mem_size(256, 4, 128);
        size_t pool_bytes = af_bytes > kf_bytes ? af_bytes : kf_bytes;
        unsigned char* raw = (unsigned char*)malloc(pool_bytes + 15u);
        unsigned char* snapshot = (unsigned char*)malloc(pool_bytes);
        unsigned char* pool = raw ? (unsigned char*)
            (((uintptr_t)raw + 15u) & ~(uintptr_t)15u) : NULL;
        FftHandle* shared_fft = fft_create(256);

        CHECK(raw != NULL && snapshot != NULL && shared_fft != NULL,
              "static direct-init guard test setup");
        if (raw && snapshot && shared_fft) {
            memset(pool, 0xA5, pool_bytes);
            memcpy(snapshot, pool, pool_bytes);

/* One refused call: it returns -1 and leaves the instance (pre-filled with a
 * recognisable pattern) and the pool exactly as they were. */
#define REFUSED_AF(label, ...) do { \
        memset(&f, 0xA5, sizeof(f)); \
        memcpy(&f_before, &f, sizeof(f)); \
        CHECK(pbfdaf_init_static(__VA_ARGS__) == -1, label " == -1"); \
        CHECK(memcmp(&f, &f_before, sizeof(f)) == 0 && \
              memcmp(pool, snapshot, pool_bytes) == 0, \
              label " writes neither instance nor pool"); \
    } while (0)
#define REFUSED_KF(label, ...) do { \
        memset(&k, 0xA5, sizeof(k)); \
        memcpy(&k_before, &k, sizeof(k)); \
        CHECK(pbfdkf_init_static(__VA_ARGS__) == -1, label " == -1"); \
        CHECK(memcmp(&k, &k_before, sizeof(k)) == 0 && \
              memcmp(pool, snapshot, pool_bytes) == 0, \
              label " writes neither instance nor pool"); \
    } while (0)

            REFUSED_AF("pbfdaf_init_static(sample_rate=0)",
                       &f, pool, af_bytes, 256, 4, 0.3f, 1e-6f, 128, 0, 0,
                       shared_fft);
            REFUSED_AF("pbfdaf_init_static(sample_rate<0)",
                       &f, pool, af_bytes, 256, 4, 0.3f, 1e-6f, 128, 0, -16000,
                       shared_fft);
            REFUSED_AF("pbfdaf_init_static(resolved hop=0)",
                       &f, pool, af_bytes, 0, 4, 0.3f, 1e-6f, 0, 0, 16000,
                       shared_fft);
            REFUSED_AF("pbfdaf_init_static(NULL)",
                       NULL, pool, af_bytes, 256, 4, 0.3f, 1e-6f, 128, 0, 16000,
                       shared_fft);

            REFUSED_KF("pbfdkf_init_static(sample_rate=0)",
                       &k, pool, kf_bytes, 256, 4, 0.3f, 1e-6f, 128, 0,
                       shared_fft);
            REFUSED_KF("pbfdkf_init_static(sample_rate<0)",
                       &k, pool, kf_bytes, 256, 4, 0.3f, 1e-6f, 128, -16000,
                       shared_fft);
            REFUSED_KF("pbfdkf_init_static(resolved hop=0)",
                       &k, pool, kf_bytes, 0, 4, 0.3f, 1e-6f, 0, 16000,
                       shared_fft);
            REFUSED_KF("pbfdkf_init_static(NULL)",
                       NULL, pool, kf_bytes, 256, 4, 0.3f, 1e-6f, 128, 16000,
                       shared_fft);
#undef REFUSED_AF
#undef REFUSED_KF

            /* A valid rate must still succeed -- otherwise the guards above
             * would be indistinguishable from "always fails". */
            CHECK(pbfdaf_init_static(&f, pool, af_bytes, 256, 4,
                                     0.3f, 1e-6f, 128, 0, 16000,
                                     shared_fft) == 0,
                  "pbfdaf_init_static(sample_rate=16000) == 0 (guard is not blanket)");
            pbfdaf_free(&f);
            CHECK(pbfdkf_init_static(&k, pool, kf_bytes, 256, 4,
                                     0.3f, 1e-6f, 128, 16000,
                                     shared_fft) == 0,
                  "pbfdkf_init_static(sample_rate=16000) == 0 (guard is not blanket)");
            pbfdkf_free(&k);
        }
        if (shared_fft) fft_destroy(shared_fft);
        free(snapshot);
        free(raw);
    }

    /* saturation_init returns void, so the fallback is the observable: the
     * retentions stay at their authoring values instead of going NaN. */
    saturation_init(&s, 0.95f, 0, 0);
    CHECK(s.alpha_attack == 0.3f && s.alpha_release == 0.98f,
          "saturation_init(hop=0, sr=0) falls back to authoring values, not NaN");
    saturation_init(&s, 0.95f, 128, 16000);
    CHECK(s.alpha_attack != 0.3f,
          "saturation_init(hop=128, sr=16000) still retimes (fallback is not blanket)");
}

/* ── shared signal-grid resolver: lockstep across every entry point ───────
 * (sample_rate, fft_size) admissibility (aec_resolve_tuned()), the `hop = fft/2` / `K = fft/2+1` derivation
 * (aec_derive_dims()) and AECB_RATE_TABLE's geometry columns all read one resolver. On all four supported grids:
 *   1. the resolver's frame/hop/n_freqs are what the grid table says, hard-coded rather than recomputed from `fft/2`;
 *   2. the grid aec_get_mem_size() sized against == the grid aec_init()/aec_create() initialised against == the hop
 *      aec_process() consumes (both construction paths) == the geometry AECB_RATE_TABLE hands the post-filter chain;
 *   3. fft_size == 0 ("auto") and every mismatched pair (16k/1024, 48k/256, 8k/512, ...) are rejected by all three entry
 *      points: 16 kHz has two production grids, so a library that guessed would silently pick one.
 * Mutations that must fail: editing one AEC_GRID_TABLE row's hop; making aec_derive_dims re-derive `hop = fft/2` itself.
 * Not caught here: removing the aec_resolve_signal_grid() call from aec_resolve_tuned() alone, since aecb_rate_cfg()
 * returns NULL for an unsupported pair and aec_get_mem_size() then returns 0. */
static void test_signal_grid_resolver(void) {
    struct { int sr, fft, frame, hop, K, legacy; const char* tag; } grids[] = {
        { 16000,  256,  256, 128, 129, 0, "16k/256 (product default)"   },
        { 16000,  512,  512, 256, 257, 0, "16k/512 (product alternate)" },
        { 48000, 1024, 1024, 512, 513, 0, "48k/1024 (product)"          },
        {  8000,  256,  256, 128, 129, 1, "8k/256 (legacy)"             },
    };
    char label[224];

    for (size_t i = 0; i < sizeof(grids) / sizeof(grids[0]); ++i) {
        AecSignalGrid g;
        memset(&g, 0, sizeof g);
        snprintf(label, sizeof(label), "%s: resolves", grids[i].tag);
        CHECK(aec_resolve_signal_grid(grids[i].sr, grids[i].fft, &g) == AEC_OK, label);
        snprintf(label, sizeof(label), "%s: frame/hop/n_freqs as tabled",
                 grids[i].tag);
        CHECK(g.sample_rate == grids[i].sr && g.fft_size == grids[i].fft &&
              g.frame_size == grids[i].frame && g.hop_size == grids[i].hop &&
              g.n_freqs == grids[i].K, label);
        snprintf(label, sizeof(label), "%s: is_legacy flag", grids[i].tag);
        CHECK(g.is_legacy == grids[i].legacy, label);

        /* The tuning table must describe the SAME geometry. It is not
         * a second resolver -- nothing consults it for admissibility -- but
         * it does hand the post chain its own n_bins/fft/block/hop, so a
         * divergence would split the instance across two grids. */
        const AecBalancedRateDims* rd = aecb_rate_cfg(grids[i].sr, grids[i].fft);
        snprintf(label, sizeof(label), "%s: AECB_RATE_TABLE agrees with the resolver",
                 grids[i].tag);
        CHECK(rd != NULL && rd->fft_size == g.fft_size &&
              rd->block_size == g.frame_size && rd->hop_size == g.hop_size &&
              rd->n_bins == g.n_freqs, label);

        /* LOCKSTEP: size, init, and process must all see this same grid. */
        AecConfig cfg;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, grids[i].sr);
        cfg.fft_size = grids[i].fft;
        size_t need = aec_get_mem_size(&cfg);
        snprintf(label, sizeof(label), "%s: aec_get_mem_size() > 0", grids[i].tag);
        CHECK(need > 0, label);

        void* pool = aec_test_pool_alloc(need);
        Aec* pa = aec_init(pool, need, &cfg);
        snprintf(label, sizeof(label), "%s: aec_init() != NULL", grids[i].tag);
        CHECK(pa != NULL, label);
        if (pa) {
            snprintf(label, sizeof(label),
                     "%s: init sees the resolver's grid", grids[i].tag);
            CHECK(pa->fft_size == g.fft_size && pa->block_size == g.frame_size &&
                  pa->hop_size == g.hop_size && pa->n_freqs == g.n_freqs, label);
            snprintf(label, sizeof(label),
                     "%s: aec_hop_size() == the resolved hop", grids[i].tag);
            CHECK(aec_hop_size(pa) == g.hop_size, label);
            /* The pool holds the decoded ENR/SU tables and the rebuilt window: equal to the reference tables. */
            {
                const AecRefTables* ref = NULL;
                const float* w = NULL;
                int wn = 0, j;
                for (j = 0; j < AECREF_N_GRIDS; ++j)
                    if (AECREF_GRIDS[j].sample_rate == grids[i].sr && AECREF_GRIDS[j].fft_size == grids[i].fft)
                        ref = &AECREF_GRIDS[j];
                const SuppressionGainTuning* tn = &pa->a3_sg.tun;
                size_t nb = ref ? (size_t)ref->n_bins * sizeof(float) : 0;
                snprintf(label, sizeof(label), "%s: pool ENR/SU tables equal the reference tables", grids[i].tag);
                CHECK(ref && tn->table_len == ref->n_bins &&
                      memcmp(tn->nearend_enr_tr, ref->nearend_enr_tr, nb) == 0 &&
                      memcmp(tn->nearend_enr_su, ref->nearend_enr_su, nb) == 0 &&
                      memcmp(tn->normal_enr_tr, ref->normal_enr_tr, nb) == 0 &&
                      memcmp(tn->normal_enr_su, ref->normal_enr_su, nb) == 0, label);
                snprintf(label, sizeof(label), "%s: pool window equals the reference window", grids[i].tag);
                CHECK(ref && aec_synthesis_window(pa, &w, &wn) == 0 && wn == grids[i].fft &&
                      memcmp(w, ref->window, (size_t)wn * sizeof(float)) == 0, label);
            }
            /* And the hop aec_process() actually consumes: hand it exactly
             * hop samples of a ramp and require the output hop to be
             * written end to end (a wrong hop would leave the tail at the
             * sentinel). */
            {
                int hop = aec_hop_size(pa);
                float* mic = (float*)calloc((size_t)hop, sizeof(float));
                float* ref = (float*)calloc((size_t)hop, sizeof(float));
                float* out = (float*)malloc((size_t)(hop + 1) * sizeof(float));
                for (int s = 0; s <= hop; ++s) out[s] = -12345.0f;
                for (int s = 0; s < hop; ++s) { mic[s] = 0.25f; ref[s] = 0.25f; }
                aec_process(pa, mic, ref, out);
                int wrote_all = 1;
                for (int s = 0; s < hop; ++s) if (out[s] == -12345.0f) wrote_all = 0;
                snprintf(label, sizeof(label),
                         "%s: aec_process() writes exactly the resolved hop",
                         grids[i].tag);
                CHECK(wrote_all && out[hop] == -12345.0f, label);
                free(mic); free(ref); free(out);
            }
            aec_destroy(pa);
        }
        free(pool);

        /* Same grid, other construction path -- and the SAME byte total. */
        Aec *dyn = NULL;
        snprintf(label, sizeof(label), "%s: aec_create() == 0", grids[i].tag);
        CHECK((dyn = aec_create(&cfg)) != NULL, label);
        snprintf(label, sizeof(label),
                 "%s: create sees the same grid as get_mem_size/init",
                 grids[i].tag);
        CHECK(dyn->fft_size == g.fft_size && dyn->block_size == g.frame_size &&
              dyn->hop_size == g.hop_size && dyn->n_freqs == g.n_freqs, label);
        CHECK(aec_get_mem_size(&dyn->cfg) == need,
              "grid lockstep: the instance's own config re-sizes identically");
        aec_destroy(dyn);
    }

    /* NO GUESSING: fft_size == 0 never resolves and is rejected everywhere. */
    {
        AecConfig valid_cfg; base_cfg(&valid_cfg);
        size_t sz = aec_get_mem_size(&valid_cfg);
        void* mem = aec_test_pool_alloc(sz);
        int rates[] = { 8000, 16000, 48000 };
        for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); ++i) {
            snprintf(label, sizeof(label),
                     "sr=%d: aec_resolve_signal_grid(fft=0) is refused", rates[i]);
            CHECK(aec_resolve_signal_grid(rates[i], 0, NULL) == AEC_ERR_ARG, label);
            AecConfig c; aec_config_from_level(&c, AEC_LEVEL_BALANCED, rates[i]);
            c.fft_size = 0;
            snprintf(label, sizeof(label), "sr=%d + fft_size=0 (auto)", rates[i]);
            check_rejected(label, &c, mem, sz);
        }
        /* Mismatched pairs: each fft is legal at SOME rate, just not this one. */
        struct { int sr, fft; } bad[] = {
            { 16000, 1024 }, { 48000, 256 }, { 48000, 512 }, { 8000, 512 },
            { 8000, 1024 }, { 16000, 128 }, { 16000, 255 }, { 16000, -256 },
        };
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
            snprintf(label, sizeof(label), "sr=%d + fft_size=%d does not resolve",
                     bad[i].sr, bad[i].fft);
            CHECK(aec_resolve_signal_grid(bad[i].sr, bad[i].fft, NULL) == AEC_ERR_ARG, label);
            AecConfig c; aec_config_from_level(&c, AEC_LEVEL_BALANCED, bad[i].sr);
            c.fft_size = bad[i].fft;
            snprintf(label, sizeof(label), "sr=%d + fft_size=%d",
                     bad[i].sr, bad[i].fft);
            check_rejected(label, &c, mem, sz);
        }
        /* An unsupported rate resolves at no fft at all. */
        CHECK(aec_resolve_signal_grid(44100, 1024, NULL) == AEC_ERR_ARG,
              "sr=44100 does not resolve at any fft_size");
        CHECK(aec_is_valid_sample_rate(44100) == 0,
              "sr=44100 is not a valid sample rate");
        /* A NULL out pointer is a pure validity query, not a crash. */
        CHECK(aec_resolve_signal_grid(16000, 256, NULL) == AEC_OK,
              "resolver accepts a NULL out (validity-only query)");
        /* On failure *out is left untouched -- a caller that ignores the
         * return value must not read a half-filled grid as if it were real. */
        {
            AecSignalGrid g; memset(&g, 0x5A, sizeof g);
            AecSignalGrid before = g;
            CHECK(aec_resolve_signal_grid(16000, 1024, &g) == AEC_ERR_ARG &&
                  memcmp(&g, &before, sizeof g) == 0,
                  "resolver leaves *out untouched on failure");
        }
        free(mem);
    }

    /* The convenience default factory is the ONE place a grid is guessed,
     * and it agrees with the resolver's first row per rate. */
    {
        struct { int sr, fft; } dflt[] = { { 8000, 256 }, { 16000, 256 },
                                           { 48000, 1024 } };
        for (size_t i = 0; i < sizeof(dflt) / sizeof(dflt[0]); ++i) {
            AecConfig c; aec_config_from_level(&c, AEC_LEVEL_BALANCED, dflt[i].sr);
            snprintf(label, sizeof(label),
                     "aec_config_from_level(balanced, sr=%d) fills fft_size=%d",
                     dflt[i].sr, dflt[i].fft);
            CHECK(c.fft_size == dflt[i].fft &&
                  aec_default_fft_size(dflt[i].sr) == dflt[i].fft, label);
            snprintf(label, sizeof(label),
                     "aec_config_from_level(balanced, sr=%d) is immediately resolvable",
                     dflt[i].sr);
            CHECK(aec_resolve_signal_grid(c.sample_rate, c.fft_size, NULL) == AEC_OK,
                  label);
        }
        /* An unsupported rate gets fft_size 0 -- which the core then
         * rejects. */
        AecConfig c; aec_config_from_level(&c, AEC_LEVEL_BALANCED, 44100);
        CHECK(c.fft_size == 0 && aec_default_fft_size(44100) == 0,
              "aec_config_from_level(balanced, sr=44100) leaves fft_size=0 (rejected later)");
    }
}

/* The library's default tuning per (rate, level), pinned as FNV-1a-32 over the AecTuning bytes (every field is 4 bytes, the
 * struct is zeroed first, so the bytes are the values). A one-unit change of any default or of a level's floors fails the
 * row of its rate; the baseline streams (AEC_TEST_BASELINE) are the second guard, this one needs no external directory.
 * Mutations that must fail: warmup_frames 100 -> 101; min_gain_floor_far_active_db of one level. Re-pin a row only together
 * with a deliberate default change. */
static uint32_t fnv32_bytes(const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < n; ++i) { h ^= b[i]; h *= 16777619u; }
    return h;
}

static void test_tuning_defaults_pin(void) {
    static const struct { int sr; uint32_t h[4]; } rows[] = {
        { 8000, { 0x39f16995u, 0x2a6346d5u, 0x64f50ffcu, 0x8fd8e0ccu } },
        { 16000, { 0xcebe373fu, 0xa6ab907fu, 0x89ea8192u, 0x72001496u } },
        { 48000, { 0xac519652u, 0x6a6f3a92u, 0xf4e4ae1fu, 0x41e4509bu } },
    };
    size_t i;
    int lv;
    CHECK(sizeof(AecTuning) % 4 == 0, "tuning pin: AecTuning has only 4-byte fields (no padding)");
    for (i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
        for (lv = 0; lv < 4; ++lv) {
            AecTestCfg x;
            uint32_t got;
            char label[96];
            aec_tcfg_from_level(&x, (AecLevel)lv, rows[i].sr);
            got = fnv32_bytes(&x.t, sizeof(x.t));
            snprintf(label, sizeof(label), "tuning pin: sr=%d level=%d defaults hash 0x%08x (pinned 0x%08x)", rows[i].sr, lv,
                     (unsigned)got, (unsigned)rows[i].h[lv]);
            CHECK(got == rows[i].h[lv], label);
        }
    }
}

/* ── delay-mode defaults + mode x field compatibility ─────────────────────
 * An illegal mode x field combination is REJECTED by all three entry points (aec_get_mem_size, aec_create, aec_init), not
 * normalised and not ignored. check_rejected() makes a fix to only ONE entry point fail here.
 * Mutations that must fail: dropping any arm of the validator's mode switch (the matrix fails); dropping the out-of-enum
 * delay_mode check (the delay_mode=7 row fails). */
static void test_delay_mode_defaults(void) {
    /* aec_config_from_level must spell fixed_delay_samples out rather than
     * leaving it at the memset 0 -- 0 is a LEGAL fixed delay, so a memset
     * default would make the shipped default config an illegal
     * MATCHED+fixed combination. */
    AecConfig cfg; aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    CHECK(cfg.delay_mode == AEC_DELAY_MATCHED,
          "defaults: delay_mode == MATCHED");
    CHECK(cfg.fixed_delay_samples == -1,
          "defaults: fixed_delay_samples == -1 (not the memset 0)");
    aec_config_from_level(&cfg, AEC_LEVEL_AGGRESSIVE, 48000);
    CHECK(cfg.delay_mode == AEC_DELAY_MATCHED &&
          cfg.fixed_delay_samples == -1,
          "from_level: keeps the default delay mode/fixed sentinel");
}

static void test_delay_mode_illegal_combinations(void) {
    AecConfig valid_cfg; base_cfg(&valid_cfg);
    size_t sz = aec_get_mem_size(&valid_cfg);
    CHECK(sz > 0, "delay-mode test setup: valid 16k config sizes > 0");
    void* mem = aec_test_pool_alloc(sz);
    CHECK(mem != NULL, "delay-mode test setup: aligned pool");

    /* A fixed delay is meaningless outside FIXED -- rejected, not dropped. */
    {
        AecConfig c; base_cfg(&c);
        c.delay_mode = AEC_DELAY_MATCHED; c.fixed_delay_samples = 1600;
        check_rejected("MATCHED + fixed_delay_samples=1600", &c, mem, sz);
        c.fixed_delay_samples = 0;
        check_rejected("MATCHED + fixed_delay_samples=0", &c, mem, sz);
        c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED; c.fixed_delay_samples = 1600;
        check_rejected("EXTERNAL_ALIGNED + fixed_delay_samples=1600", &c, mem, sz);
    }
    /* FIXED without a delay to apply. */
    {
        AecConfig c; base_cfg(&c);
        c.delay_mode = AEC_DELAY_FIXED; c.fixed_delay_samples = -1;
        check_rejected("FIXED + fixed_delay_samples=-1", &c, mem, sz);
        c.fixed_delay_samples = -1000;
        check_rejected("FIXED + negative fixed_delay_samples", &c, mem, sz);
        /* Rate-relative 120 s bound: in FIXED the ring is sized fixed+hop
         * (4096 is the MATCHED search ring's headroom, not this mode's), so
         * an unbounded value would drive an unbounded allocation. */
        c.fixed_delay_samples = 120 * 16000 + 1;
        check_rejected("FIXED + fixed_delay_samples past the 120 s bound",
                       &c, mem, sz);
    }
    /* delay_num_filters sizes the MATCHED bank only. */
    {
        AecConfig c; base_cfg(&c);
        c.delay_mode = AEC_DELAY_FIXED; c.fixed_delay_samples = 1600;
        c.delay_num_filters = 2;
        check_rejected("FIXED + delay_num_filters=2", &c, mem, sz);
        c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED; c.fixed_delay_samples = -1;
        check_rejected("EXTERNAL_ALIGNED + delay_num_filters=2", &c, mem, sz);
    }
    /* n=0 is NEVER a silent "delay estimation off" switch, in any mode. */
    {
        const AecDelayMode modes[3] = { AEC_DELAY_MATCHED, AEC_DELAY_FIXED,
                                        AEC_DELAY_EXTERNAL_ALIGNED };
        const char* names[3] = { "MATCHED", "FIXED", "EXTERNAL_ALIGNED" };
        char label[160];
        for (int i = 0; i < 3; ++i) {
            AecConfig c; base_cfg(&c);
            c.delay_mode = modes[i];
            c.fixed_delay_samples = (modes[i] == AEC_DELAY_FIXED) ? 1600 : -1;
            c.delay_num_filters = 0;
            snprintf(label, sizeof(label), "%s + delay_num_filters=0", names[i]);
            check_rejected(label, &c, mem, sz);
        }
    }
    /* Out-of-enum mode, and a non-boolean switch. */
    {
        AecConfig c; base_cfg(&c);
        c.delay_mode = (AecDelayMode)7;
        check_rejected("delay_mode=7 (out of enum)", &c, mem, sz);
        base_cfg(&c);
        c.enable_cng = 2;
        check_rejected("enable_cng=2 (non-boolean)", &c, mem, sz);
        base_cfg(&c);
        c.level = (AecLevel)4;
        check_rejected("level=4 (out of enum)", &c, mem, sz);
        c.level = (AecLevel)-1;
        check_rejected("level=-1 (out of enum)", &c, mem, sz);
        {
            AecTestCfg t; base_cfg(&t.c); aec_tcfg_refresh(&t);
            t.t.min_gain_split_floor_enabled = 2;
            check_rejected_tuned("min_gain_split_floor_enabled=2 (non-boolean)", &t.c, &t.t, mem, sz);
        }
    }

    /* The guard is not blanket: every LEGAL combination still constructs,
     * on all three entry points. Without this the matrix above would be
     * indistinguishable from "always rejects". */
    {
        struct { AecDelayMode m; int fixed; int n; const char* tag; } ok[] = {
            { AEC_DELAY_MATCHED,          -1,    5, "MATCHED n=5 (default)" },
            { AEC_DELAY_MATCHED,          -1,    1, "MATCHED n=1" },
            { AEC_DELAY_MATCHED,          -1,    3, "MATCHED n=3" },
            { AEC_DELAY_FIXED,          1600,    5, "FIXED 1600" },
            { AEC_DELAY_FIXED,             0,    5, "FIXED 0 (legal delay)" },
            { AEC_DELAY_FIXED,   120 * 16000,    5, "FIXED at the 120 s bound" },
            { AEC_DELAY_EXTERNAL_ALIGNED, -1,    5, "EXTERNAL_ALIGNED" },
        };
        char label[192];
        for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); ++i) {
            AecConfig c; base_cfg(&c);
            c.delay_mode = ok[i].m;
            c.fixed_delay_samples = ok[i].fixed;
            c.delay_num_filters = ok[i].n;
            size_t need = aec_get_mem_size(&c);
            snprintf(label, sizeof(label), "%s: aec_get_mem_size() > 0", ok[i].tag);
            CHECK(need > 0, label);
            Aec *a = NULL;
            snprintf(label, sizeof(label), "%s: aec_create() == 0", ok[i].tag);
            CHECK((a = aec_create(&c)) != NULL, label);
            snprintf(label, sizeof(label), "%s: instance keeps the mode",
                     ok[i].tag);
            CHECK(a->cfg.delay_mode == ok[i].m, label);
            snprintf(label, sizeof(label), "%s: estimator built iff MATCHED",
                     ok[i].tag);
            CHECK(a->has_delay == (ok[i].m == AEC_DELAY_MATCHED), label);
            snprintf(label, sizeof(label), "%s: ring carved iff not EXTERNAL",
                     ok[i].tag);
            CHECK((a->ref_ring != NULL) ==
                  (ok[i].m != AEC_DELAY_EXTERNAL_ALIGNED), label);
            if (ok[i].m == AEC_DELAY_FIXED) {
                snprintf(label, sizeof(label),
                         "%s: current_delay seeded from the config", ok[i].tag);
                CHECK(a->current_delay == ok[i].fixed, label);
                /* The FIXED ring is sized for THIS delay (the tight `fixed + hop` bound), not from
                 * the MATCHED search budget. The sizing rule and the exact-fit ring's alignment
                 * behaviour live in test_linear_context.c / test_delay_num_filters.c; this row only
                 * pins that a validated config lands on it. */
                snprintf(label, sizeof(label),
                         "%s: ring == fixed + hop (%d)", ok[i].tag,
                         ok[i].fixed + a->hop_size);
                CHECK(a->ref_ring_size == ok[i].fixed + a->hop_size, label);
            }
            aec_destroy(a);
            /* Same config through the caller-pool path. */
            void* pool = aec_test_pool_alloc(need);
            snprintf(label, sizeof(label), "%s: aec_init() != NULL", ok[i].tag);
            Aec* pa = aec_init(pool, need, &c);
            CHECK(pa != NULL, label);
            if (pa) {
                snprintf(label, sizeof(label),
                         "%s: pool path agrees on mode/ring", ok[i].tag);
                CHECK(pa->cfg.delay_mode == ok[i].m &&
                      pa->has_delay == (ok[i].m == AEC_DELAY_MATCHED) &&
                      (pa->ref_ring != NULL) ==
                          (ok[i].m != AEC_DELAY_EXTERNAL_ALIGNED), label);
                aec_destroy(pa);
            }
            free(pool);
        }
    }

    free(mem);
}

int main(void) {
    test_sample_rate();
    test_filter_length();
    test_config_factories_unsupported_rate();
    test_highpass_cutoff();
    test_misaligned_base();
    test_residual_mode_range_and_exposure();
    test_signal_grid_resolver();
    test_tuning_defaults_pin();
    test_delay_mode_defaults();
    test_delay_mode_illegal_combinations();
    test_valid_config_smoke();
    test_multi_rate_valid_smoke();
    test_direct_init_rejects_bad_rate();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
