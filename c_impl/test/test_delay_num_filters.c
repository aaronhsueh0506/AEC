/* test_delay_num_filters.c — permanent regression test for the configurable
 * matched-filter bank size (AecConfig::delay_num_filters ->
 * delay_aec3_init_ex -> DaMatchedFilter::num_filters).
 *
 * Background: DA_NUM_FILTERS used to be BOTH the static array bound and the
 * loop bound, so the bank was pinned at 5 with no way to trade reach for
 * MACs. 5 is correct for bench/dataset (it is what the published scores were
 * measured at), but the embedded target compensates the bring-up-measured
 * system delay out-of-band and only needs the matched filter to track the
 * residual, where each dropped filter removes ~4.2 MMAC/s of full-rate
 * search. DA_NUM_FILTERS is now only the array bound; the searched bank is
 * runtime.
 *
 * What this test proves (nothing else in the suite can):
 *   1. The default is still 5, on aec_config_defaults AND on all three
 *      presets, and a default-configured Aec really builds a 5-filter bank.
 *      (The byte-equality of the default PATH is covered by the existing
 *      golden/parity gates — this only pins the value the config carries.)
 *   2. The knob genuinely moves the bank's GEOMETRY, end to end from
 *      AecConfig through aec_create. Reliable reach is
 *      (n-1)*DA_FILTER_INTRA_SHIFT + (DA_FILTER_SIZE - 11) downsampled
 *      samples at 0.25 ms, i.e. 221 ms at n=2 and 509 ms at n=5. So:
 *        - a 150 ms echo locks at n=2   (inside reach; proves a short bank
 *          still works and the no-lock case below is about REACH, not about
 *          a broken estimator),
 *        - a 350 ms echo does NOT lock at n=2 (outside 221 ms),
 *        - the same 350 ms echo DOES lock at n=5 (inside 509 ms) — the
 *          control that makes the previous assertion meaningful.
 *      Mutation-checked: reverting the loop bound in
 *      da_matched_filter_update() to DA_NUM_FILTERS, or reverting aec.c's
 *      delay_aec3_init_ex() call to delay_aec3_init(), makes the n=2 /
 *      350 ms case lock and this test go red.
 *
 *      Why 350 ms and not 400: on a PURE single-tap echo (no reverb tail)
 *      the AEC3 pre-echo aggregator occasionally reports an onset one
 *      alignment shift early, and 400 ms at n=5 happens to be one of those
 *      points (reports 4800 instead of 6336, deterministically, on every
 *      seed tried). That is pre-existing upstream-port behaviour, unrelated
 *      to the bank size — verified identical on a build of the sources as
 *      they were before this knob existed — and matched_filter.py's own
 *      header already warns that detect_pre_echo can produce noisy
 *      estimates on material without real pre-echo. Sitting the control
 *      case on a known artifact would just make this test look flaky, so it
 *      uses the neighbouring clean point instead.
 *   3. Out-of-range config values are REJECTED by aec_validate_config
 *      (aec_create fails), matching the Python side's ValueError rather
 *      than silently normalising a caller's wrong mental model.
 *   4. The lower-level delay_aec3_init_ex() CLAMPS instead (defence in depth
 *      for direct users of that API), and plain delay_aec3_init() still
 *      means DA_NUM_FILTERS.
 *
 * Build (standalone, from c_impl/ — mirrors test_linear_context.c):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_delay_num_filters.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_delay_num_filters
 * Run:  ./bin/test_delay_num_filters
 * Also wired into the Makefile: `make test-delay-num-filters`.
 */
#include "aec.h"
#include <math.h>
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

/* 16 kHz / fft 256 -> hop 128. 3.0 s of audio is comfortably past the
 * aggregator's converged threshold at every bank size tested here. */
#define SR          16000
#define HOP         128
#define SECONDS     3
#define N_SAMPLES   (SR * SECONDS)
#define DELAY_NEAR  2400                 /* 150 ms @ 16 kHz */
#define DELAY_FAR   5600                 /* 350 ms @ 16 kHz */
#define MAX_DELAY   DELAY_FAR
/* 160 raw samples = 10 ms: far wider than the estimator's own quantisation
 * (32-sample headroom + 4-sample downsample granularity, ~64 in practice),
 * far tighter than the ~180 ms error an out-of-reach bank would have to
 * make to accidentally pass. */
#define LOCK_TOL    160

static float g_far[N_SAMPLES + MAX_DELAY];
static float g_near[N_SAMPLES + MAX_DELAY];
static float g_out[HOP];

/* Deterministic xorshift noise, same style as the sibling tests. */
static unsigned g_rng_state = 0x5EED1234u;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu;
    return amp * (2.0f * u - 1.0f);
}

/* far = white noise; near = far delayed by `delay` and attenuated. */
static void build_signals(int delay) {
    int i;
    g_rng_state = 0x5EED1234u;
    for (i = 0; i < N_SAMPLES + MAX_DELAY; ++i) g_far[i] = rng_uniform(0.35f);
    memset(g_near, 0, sizeof(g_near));
    for (i = delay; i < N_SAMPLES + MAX_DELAY; ++i) g_near[i] = 0.5f * g_far[i - delay];
}

/* Run the full engine at `num_filters` against a `delay`-sample echo.
 * Returns the raw matched-filter estimate (-1 if it never produced one) and
 * writes the bank size the engine actually built into *bank_out. */
static int run_case(int num_filters, int delay, int *bank_out) {
    AecConfig cfg;
    Aec a;
    int i, n_hops = N_SAMPLES / HOP;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    cfg.delay_num_filters = num_filters;
    if (aec_create(&a, &cfg) != 0) {
        *bank_out = -1;
        return -2;   /* distinct from "no estimate" so a failure is visible */
    }
    build_signals(delay);
    for (i = 0; i < n_hops; ++i)
        aec_process(&a, g_near + i * HOP, g_far + i * HOP, g_out);
    *bank_out = a.delay.est.matched_filter.num_filters;
    int got = delay_aec3_estimated_delay(&a.delay);
    aec_destroy(&a);
    return got;
}

/* ---------------------------------------------------------------- defaults */

static void test_default_is_five(void) {
    AecConfig cfg;
    Aec a;
    int bank;

    aec_config_defaults(&cfg, SR);
    CHECK(cfg.delay_num_filters == DA_NUM_FILTERS,
          "aec_config_defaults: delay_num_filters == DA_NUM_FILTERS (5)");
    CHECK(DA_NUM_FILTERS == 5, "DA_NUM_FILTERS is still 5 (the AEC3 default)");

    aec_config_from_preset(&cfg, AEC_PRESET_MILD, SR);
    CHECK(cfg.delay_num_filters == 5, "preset mild: delay_num_filters == 5");
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    CHECK(cfg.delay_num_filters == 5, "preset balanced: delay_num_filters == 5");
    aec_config_from_preset(&cfg, AEC_PRESET_AGGRESSIVE, SR);
    CHECK(cfg.delay_num_filters == 5, "preset aggressive: delay_num_filters == 5");

    aec_config_defaults(&cfg, SR);
    CHECK(aec_create(&a, &cfg) == 0, "default cfg: aec_create");
    bank = a.delay.est.matched_filter.num_filters;
    aec_destroy(&a);
    CHECK(bank == 5, "default cfg builds a 5-filter bank");
}

/* ---------------------------------------------------------------- geometry */

static void test_geometry(void) {
    char msg[192];
    int bank_a, bank_b, bank_c;
    int got;

    /* n=2, 150 ms: inside the 221 ms reach -> must lock. */
    got = run_case(2, DELAY_NEAR, &bank_a);
    snprintf(msg, sizeof msg, "n=2 built a 2-filter bank (got %d)", bank_a);
    CHECK(bank_a == 2, msg);
    snprintf(msg, sizeof msg,
             "n=2 locks a 150 ms echo: reported %d, true %d (|err| <= %d)",
             got, DELAY_NEAR, LOCK_TOL);
    CHECK(got >= 0 && abs(got - DELAY_NEAR) <= LOCK_TOL, msg);

    /* n=2, 350 ms: beyond the 221 ms reach -> must NOT lock. This is the
     * assertion the mutations in the header comment turn red. */
    got = run_case(2, DELAY_FAR, &bank_b);
    snprintf(msg, sizeof msg,
             "n=2 does NOT lock a 350 ms echo (out of 221 ms reach): reported %d",
             got);
    CHECK(bank_b == 2 && (got < 0 || abs(got - DELAY_FAR) > LOCK_TOL), msg);

    /* n=5, same 350 ms echo: inside the 509 ms reach -> must lock. Without
     * this control, the assertion above would also pass against an
     * estimator that simply never locks anything. */
    got = run_case(5, DELAY_FAR, &bank_c);
    snprintf(msg, sizeof msg,
             "n=5 locks the SAME 350 ms echo: reported %d, true %d (|err| <= %d)",
             got, DELAY_FAR, LOCK_TOL);
    CHECK(bank_c == 5 && got >= 0 && abs(got - DELAY_FAR) <= LOCK_TOL, msg);
}

/* -------------------------------------------------------------- validation */

static void test_config_validation(void) {
    AecConfig cfg;
    Aec a;
    int n;
    char msg[96];

    aec_config_defaults(&cfg, SR);
    cfg.delay_num_filters = 0;
    CHECK(aec_create(&a, &cfg) != 0, "delay_num_filters=0 is rejected");
    CHECK(aec_get_mem_size(&cfg) == 0, "delay_num_filters=0 -> mem_size 0");

    aec_config_defaults(&cfg, SR);
    cfg.delay_num_filters = DA_NUM_FILTERS + 1;
    CHECK(aec_create(&a, &cfg) != 0, "delay_num_filters=6 is rejected");

    aec_config_defaults(&cfg, SR);
    cfg.delay_num_filters = -1;
    CHECK(aec_create(&a, &cfg) != 0, "delay_num_filters=-1 is rejected");

    for (n = 1; n <= DA_NUM_FILTERS; ++n) {
        aec_config_defaults(&cfg, SR);
        cfg.delay_num_filters = n;
        snprintf(msg, sizeof msg, "delay_num_filters=%d is accepted and honoured", n);
        if (aec_create(&a, &cfg) != 0) { CHECK(0, msg); continue; }
        CHECK(a.delay.est.matched_filter.num_filters == n, msg);
        aec_destroy(&a);
    }
}

/* -------------------------------------------------- low-level init contract */

static void test_init_ex_contract(void) {
    static DelayAec3 d;   /* ~40 KB; static, not stack */

    delay_aec3_init(&d, SR);
    CHECK(d.est.matched_filter.num_filters == DA_NUM_FILTERS,
          "delay_aec3_init() still means the full DA_NUM_FILTERS bank");

    delay_aec3_init_ex(&d, SR, 3);
    CHECK(d.est.matched_filter.num_filters == 3, "init_ex(3) -> bank 3");

    delay_aec3_init_ex(&d, SR, 0);
    CHECK(d.est.matched_filter.num_filters == 1, "init_ex(0) clamps up to 1");

    delay_aec3_init_ex(&d, SR, -7);
    CHECK(d.est.matched_filter.num_filters == 1, "init_ex(-7) clamps up to 1");

    delay_aec3_init_ex(&d, SR, 99);
    CHECK(d.est.matched_filter.num_filters == DA_NUM_FILTERS,
          "init_ex(99) clamps down to DA_NUM_FILTERS");

    /* The bank size must survive a reset — delay_aec3_reset() clears the
     * signal chain, not the geometry. */
    delay_aec3_init_ex(&d, SR, 2);
    delay_aec3_reset(&d);
    CHECK(d.est.matched_filter.num_filters == 2, "bank size survives delay_aec3_reset()");
}

int main(void) {
    printf("=== test_delay_num_filters ===\n");
    test_default_is_five();
    test_geometry();
    test_config_validation();
    test_init_ex_contract();
    printf("---\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
