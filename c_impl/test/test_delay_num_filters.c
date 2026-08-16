/* test_delay_num_filters.c — permanent regression test for the configurable
 * matched-filter bank size (AecConfig::delay_num_filters -> the pool layout
 * -> DaMatchedFilter::num_filters).
 *
 * Background, in two steps. First DA_NUM_FILTERS stopped being the loop
 * bound: it used to be BOTH the static array bound and the loop bound, so the
 * bank was pinned at 5 with no way to trade reach for MACs. 5 is correct for
 * bench/dataset (it is what the published scores were measured at), but the
 * embedded target compensates the bring-up-measured system delay out-of-band
 * and only needs the matched filter to track the residual, where each dropped
 * filter removes ~4.2 MMAC/s of full-rate search.
 *
 * Then (productization plan step 2) it stopped sizing anything at all: the
 * bank, the render ring and both lag histograms are POOL-CARVED to the
 * configured n, so a short bank now saves BYTES as well as MACs.
 * DA_NUM_FILTERS is purely the validation upper bound.
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
 *      da_matched_filter_update() to DA_NUM_FILTERS, or hardcoding aec.c's
 *      delay_aec3_init() call to DA_NUM_FILTERS, makes the n=2 /
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
 *   4. The lower-level delay_aec3_get_mem_size()/delay_aec3_init() pair
 *      REJECTS out-of-range n as well (get_mem_size -> 0, init -> nonzero;
 *      external review 2026-08-16 -- a silent clamp here let direct callers
 *      run a different bank than requested), plus the pool rejection
 *      contract (NULL / misaligned base / one byte short).
 *   5. Plan §3.4.1: on any one grid mem(n=1) < ... < mem(n=5), on all four
 *      grids -- every one of those numbers was IDENTICAL before step 2.
 *      Non-MATCHED modes, which carve no estimator, land below even n=1.
 *   5b. Plan §3.2.8 / §3.4 (step 3): the FIXED pool then scales with
 *      fixed_delay_samples -- exactly EXTERNAL_ALIGNED plus one
 *      ALIGN16((fixed + hop) * 4) ring -- where before step 3 every FIXED
 *      delay produced the same number (the MATCHED search-ring size).
 *   6. Plan §3.4.3: aec_create() (heap arena) and aec_init() (caller pool)
 *      stay sample-exact at every bank size, now that the layout depends on
 *      the bank size.
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
#include "delay_pool_test_util.h"
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

/* Shared pool-first construction (delay_pool_test_util.h), reported through
 * this file's CHECK macro. */
static void *da_pool_init(DelayAec3 *d, int sr, int hop, int n) {
    const char *why = NULL;
    void *pool = delay_pool_init(d, sr, hop, n, &why);
    if (!pool) CHECK(0, why);
    return pool;
}

static void test_low_level_init_contract(void) {
    DelayAec3 d;
    void *pool;
    char msg[160];
    int n;

    /* Fail-fast contract at the low level too (external review 2026-08-16:
     * the earlier silent clamp let a direct caller run a different bank size
     * than requested with no error signal). get_mem_size returns 0 and init
     * returns nonzero for every out-of-range n; valid n still round-trips. */
    pool = da_pool_init(&d, SR, HOP, 3);
    if (pool) {
        CHECK(d.est.matched_filter.num_filters == 3, "init(n=3) -> bank 3");
        free(pool);
    }

    CHECK(delay_aec3_get_mem_size(SR, HOP, 0) == 0,  "get_mem_size(n=0) rejects (0)");
    CHECK(delay_aec3_get_mem_size(SR, HOP, -7) == 0, "get_mem_size(n=-7) rejects (0)");
    CHECK(delay_aec3_get_mem_size(SR, HOP, DA_NUM_FILTERS + 1) == 0,
          "get_mem_size(n=max+1) rejects (0)");
    CHECK(delay_aec3_get_mem_size(SR, HOP, 99) == 0, "get_mem_size(n=99) rejects (0)");
    {
        /* A validly-sized pool must still be refused when n is invalid --
         * rejection must come from validation, not from a size mismatch. */
        size_t ok_bytes = delay_aec3_get_mem_size(SR, HOP, DA_NUM_FILTERS);
        void *p = NULL;
        CHECK(ok_bytes > 0, "reference size for rejection probe");
        if (ok_bytes > 0 && posix_memalign(&p, 16, ok_bytes) == 0) {
            CHECK(delay_aec3_init(&d, p, ok_bytes, SR, HOP, 0) != 0,
                  "init(n=0) rejects");
            CHECK(delay_aec3_init(&d, p, ok_bytes, SR, HOP, 99) != 0,
                  "init(n=99) rejects");
            free(p);
        }
    }

    /* The bank size must survive a reset — delay_aec3_reset() clears the
     * signal chain, not the geometry. */
    pool = da_pool_init(&d, SR, HOP, 2);
    if (pool) {
        int cap = d.est.render_ring.capacity;
        int hpn = d.est.aggregator.highest_peak.hist_size;
        int pen = d.est.aggregator.pre_echo.hist_size;
        delay_aec3_reset(&d);
        CHECK(d.est.matched_filter.num_filters == 2, "bank size survives delay_aec3_reset()");
        CHECK(d.est.render_ring.capacity == cap &&
              d.est.aggregator.highest_peak.hist_size == hpn &&
              d.est.aggregator.pre_echo.hist_size == pen,
              "pool geometry survives delay_aec3_reset()");
        free(pool);
    }

    /* Rejection contract: NULL, misaligned base, and an undersized block are
     * all refused rather than carved past. */
    {
        size_t need = delay_aec3_get_mem_size(SR, HOP, DA_NUM_FILTERS);
        void *p = NULL;
        CHECK(delay_aec3_init(&d, NULL, need, SR, HOP, DA_NUM_FILTERS) != 0,
              "delay_aec3_init(NULL pool) rejected");
        if (posix_memalign(&p, 16, need + 16) == 0 && p) {
            CHECK(delay_aec3_init(&d, (char *)p + 8, need, SR, HOP, DA_NUM_FILTERS) != 0,
                  "delay_aec3_init(base not 16-byte aligned) rejected");
            CHECK(delay_aec3_init(&d, p, need - 1, SR, HOP, DA_NUM_FILTERS) != 0,
                  "delay_aec3_init(pool one byte short) rejected");
            CHECK(delay_aec3_init(&d, p, need, SR, HOP, DA_NUM_FILTERS) == 0,
                  "delay_aec3_init(exactly the queried size) accepted");
            free(p);
        }
        CHECK(delay_aec3_get_mem_size(0, HOP, DA_NUM_FILTERS) == 0 &&
              delay_aec3_get_mem_size(SR, 0, DA_NUM_FILTERS) == 0,
              "get_mem_size rejects a non-positive rate or hop");
    }

    /* End-to-end grid pass-through: aec_carve() must hand the estimator the
     * RESOLVED hop, not a guess. The 48 kHz sidechain scratch is the only
     * hop-dependent term, so it is the one place a wrong hop is observable
     * from outside -- and it is observable, which is the point of asserting
     * it here rather than trusting the call site.
     * Mutation-checked: passing any other hop to delay_aec3_init() in
     * aec_carve() turns this row red. */
    {
        static const struct { int sr, fft, hop, cap; } grids[] = {
            { 48000, 1024, 512, DA_RESAMPLE48_CAP_FOR(512) },
            { 16000,  256, 128, 0 },
            { 16000,  512, 256, 0 },
            {  8000,  256, 128, 0 },
        };
        unsigned g;
        for (g = 0; g < sizeof grids / sizeof grids[0]; ++g) {
            AecConfig cfg;
            Aec a;
            aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, grids[g].sr);
            cfg.fft_size = grids[g].fft;
            if (aec_create(&a, &cfg) != 0) {
                snprintf(msg, sizeof msg, "sr=%d fft=%d: aec_create",
                         grids[g].sr, grids[g].fft);
                CHECK(0, msg);
                continue;
            }
            snprintf(msg, sizeof msg,
                     "sr=%d fft=%d: estimator sized off the RESOLVED hop %d "
                     "(resample_cap %d, expected %d)",
                     grids[g].sr, grids[g].fft, grids[g].hop,
                     a.delay.resample_cap, grids[g].cap);
            CHECK(a.delay.resample_cap == grids[g].cap, msg);
            snprintf(msg, sizeof msg,
                     "sr=%d fft=%d: 48 kHz sidechain scratch present iff 48 kHz",
                     grids[g].sr, grids[g].fft);
            CHECK((a.delay.near16_scratch != NULL) == (grids[g].cap > 0) &&
                  (a.delay.far16_scratch != NULL) == (grids[g].cap > 0), msg);
            aec_destroy(&a);
        }
    }

    /* The 48 kHz sidechain scratch is 48 kHz-ONLY (plan §3.2.7): the same
     * bank costs strictly more at 48 kHz than at 16 kHz, and the difference
     * is exactly the two ALIGN16'd scratch buffers. */
    for (n = 1; n <= DA_NUM_FILTERS; ++n) {
        size_t m16 = delay_aec3_get_mem_size(16000, 128, n);
        size_t m48 = delay_aec3_get_mem_size(48000, 512, n);
        size_t scratch = 2 * (size_t)((DA_RESAMPLE48_CAP_FOR(512) * sizeof(float) + 15) & ~(size_t)15);
        snprintf(msg, sizeof msg,
                 "n=%d: 48 kHz costs exactly the two sidechain scratch buffers more "
                 "than 16 kHz (%zu - %zu == %zu)", n, m48, m16, scratch);
        CHECK(m48 > m16 && m48 - m16 == scratch, msg);
    }
}

/* ------------------------------------------- pool geometry, HAND-COMPUTED */

/* Plan §3.2/§3.4. The three n-dependent array lengths, pinned against
 * LITERALS derived by hand from the AEC3 geometry -- NOT against
 * DA_*_FOR(n), which is the very thing under test. An earlier draft of this
 * test read those macros back, which made every row a tautology: pinning the
 * macros to the old fixed n=5 bound (exactly the regression this file is
 * supposed to catch) left it fully green. Measured, then fixed.
 *
 *   ring capacity = (32 + 24*(n-1) + 1) * 16
 *   highest-peak bins = n*384 + 512 + 1
 *   pre-echo bins     = ((n*384 + 513) * 4) >> 6
 *
 * and the pool cost of ONE extra filter is a constant 5728 B at every grid:
 *   coefficients   512 f32          = 2048 B
 *   accum. error   128 f32          =  512 B
 *   render ring    384 f32          = 1536 B
 *   highest-peak   384 int          = 1536 B
 *   pre-echo        24 int          =   96 B
 * Every one of those five terms is a multiple of 16, so the per-field ALIGN16
 * padding is identical at every n and the delta really is exact, not
 * approximate. */
#define DA_STEP_BYTES 5728

static void test_pool_geometry_literals(void) {
    static const struct { int n, ring, hp, pe; } expect[] = {
        { 1,  528,  897,  56 },
        { 2,  912, 1281,  80 },
        { 3, 1296, 1665, 104 },
        { 4, 1680, 2049, 128 },
        { 5, 2064, 2433, 152 },
    };
    char msg[192];
    unsigned i;

    for (i = 0; i < sizeof expect / sizeof expect[0]; ++i) {
        AecConfig cfg;
        Aec a;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        cfg.delay_num_filters = expect[i].n;
        if (aec_create(&a, &cfg) != 0) {
            snprintf(msg, sizeof msg, "n=%d: aec_create for geometry check", expect[i].n);
            CHECK(0, msg);
            continue;
        }
        snprintf(msg, sizeof msg,
                 "n=%d: render ring carved to %d samples (hand-computed %d)",
                 expect[i].n, a.delay.est.render_ring.capacity, expect[i].ring);
        CHECK(a.delay.est.render_ring.capacity == expect[i].ring, msg);
        snprintf(msg, sizeof msg,
                 "n=%d: highest-peak histogram carved to %d bins (hand-computed %d)",
                 expect[i].n, a.delay.est.aggregator.highest_peak.hist_size, expect[i].hp);
        CHECK(a.delay.est.aggregator.highest_peak.hist_size == expect[i].hp, msg);
        snprintf(msg, sizeof msg,
                 "n=%d: pre-echo histogram carved to %d bins (hand-computed %d)",
                 expect[i].n, a.delay.est.aggregator.pre_echo.hist_size, expect[i].pe);
        CHECK(a.delay.est.aggregator.pre_echo.hist_size == expect[i].pe, msg);
        aec_destroy(&a);
    }
}

/* ----------------------------------------------- pool size vs the bank size */

/* Plan §3.4.1: on any one grid the pool must SHRINK monotonically as the bank
 * shrinks. Before plan step 2 every one of these numbers was identical (the
 * arrays were carved at the 5-filter bound), so this is the assertion that
 * pins the whole point of that step. Prints the measured table as it goes --
 * that printout is the source of the byte table in the manual, rather than
 * any hardcoded expectation here. */
static void test_mem_size_shrinks_with_bank(void) {
    static const struct { int sr, fft; const char *tag; } grids[] = {
        { 16000,  256, "16k/256"        },
        { 16000,  512, "16k/512"        },
        { 48000, 1024, "48k/1024"       },
        {  8000,  256, "8k/256 (legacy)"},
    };
    char msg[192];
    unsigned g;

    printf("--- aec_get_mem_size(balanced) by grid x delay_num_filters ---\n");
    for (g = 0; g < sizeof grids / sizeof grids[0]; ++g) {
        size_t prev = 0;
        int n;
        for (n = 1; n <= DA_NUM_FILTERS; ++n) {
            AecConfig cfg;
            size_t got;
            aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, grids[g].sr);
            cfg.fft_size = grids[g].fft;
            cfg.delay_num_filters = n;
            got = aec_get_mem_size(&cfg);
            printf("    %-16s n=%d  %zu B\n", grids[g].tag, n, got);
            snprintf(msg, sizeof msg, "%s: mem(n=%d) > 0", grids[g].tag, n);
            CHECK(got > 0, msg);
            if (n > 1) {
                /* Strictly monotonic AND by the exact hand-computed step --
                 * "> prev" alone would still pass if only some of the five
                 * n-dependent arrays actually shrank. */
                snprintf(msg, sizeof msg,
                         "%s: mem(n=%d) - mem(n=%d) == %d B exactly (%zu - %zu = %zu)",
                         grids[g].tag, n, n - 1, DA_STEP_BYTES, got, prev, got - prev);
                CHECK(got > prev && got - prev == (size_t)DA_STEP_BYTES, msg);
            }
            prev = got;
        }
        /* The estimator is a MATCHED-only cost: with no estimator to carve,
         * the pool must land strictly below even the smallest bank. Since
         * plan step 3 the FIXED figure below is also the FLOOR of that mode
         * (fixed_delay_samples=0, i.e. a hop-sized ring) rather than one
         * fixed number for the whole mode -- see
         * test_fixed_pool_scales_with_the_delay(). */
        {
            AecConfig c1, cf, ce;
            size_t m1, mf, me;
            aec_config_from_preset(&c1, AEC_PRESET_BALANCED, grids[g].sr);
            c1.fft_size = grids[g].fft; c1.delay_num_filters = 1;
            m1 = aec_get_mem_size(&c1);
            cf = c1; cf.delay_num_filters = DA_NUM_FILTERS;
            cf.delay_mode = AEC_DELAY_FIXED; cf.fixed_delay_samples = 0;
            mf = aec_get_mem_size(&cf);
            ce = c1; ce.delay_num_filters = DA_NUM_FILTERS;
            ce.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
            me = aec_get_mem_size(&ce);
            printf("    %-16s FIXED %zu B   EXTERNAL %zu B\n", grids[g].tag, mf, me);
            snprintf(msg, sizeof msg, "%s: FIXED (%zu) < MATCHED n=1 (%zu)",
                     grids[g].tag, mf, m1);
            CHECK(mf > 0 && mf < m1, msg);
            snprintf(msg, sizeof msg, "%s: EXTERNAL_ALIGNED (%zu) < FIXED (%zu)",
                     grids[g].tag, me, mf);
            CHECK(me > 0 && me < mf, msg);
        }
    }
}

/* Plan §3.2.8 / §3.4: the FIXED pool is a function of fixed_delay_samples,
 * not one flat number for the mode. Before plan step 3 every row of this
 * table was IDENTICAL (the ring was sized from the MATCHED search budget,
 * max(delay_buffer_ms, max_delay_ms + 4096), whatever the delay was), so
 * this is the assertion that pins the whole point of that step.
 *
 * Three separable claims:
 *   a) strictly monotonic in the delay, by the EXACT ALIGN16 ring step --
 *      "> prev" alone would still pass if the ring shrank by the wrong term;
 *   b) a small fixed delay is DRAMATICALLY cheaper than the old search-ring
 *      sizing -- pinned as a hard ratio against the MATCHED n=5 pool, not
 *      as a hardcoded KiB figure;
 *   c) the whole difference between EXTERNAL_ALIGNED and FIXED(d) is the
 *      ring array itself, which is what "EXTERNAL carries no delay bytes"
 *      means concretely.
 * The printed table is the source of the manual's per-mode byte figures. */
/* The FIXED ring as aec_carve() lays it out: ALIGN16((fixed + hop) * 4).
 * Spelled ONCE so the exact-composition claim and the monotonic-step claim
 * below cannot drift apart -- two hand-inlined copies of the same formula
 * could be edited into agreeing with each other and with neither the carve.
 * Deliberately re-derived here rather than read from a library helper: this
 * is the independent expectation the assertions compare aec_get_mem_size()
 * against. */
static size_t fixed_ring_bytes(int fixed, int hop) {
    return (((size_t)(fixed + hop) * sizeof(float)) + 15u) & ~(size_t)15u;
}

static void test_fixed_pool_scales_with_the_delay(void) {
    static const struct { int sr, fft, hop; const char *tag; } grids[] = {
        { 16000,  256, 128, "16k/256"  },
        { 16000,  512, 256, "16k/512"  },
        { 48000, 1024, 512, "48k/1024" },
    };
    /* ms of fixed delay -- converted per rate so every grid walks the same
     * physical delays (0 / 5 / 25 / 100 / 500 / 2500 ms). */
    static const float delay_ms[] = { 0.0f, 5.0f, 25.0f, 100.0f, 500.0f, 2500.0f };
    char msg[224];
    unsigned g, d;

    printf("--- aec_get_mem_size(balanced) FIXED by grid x fixed_delay ---\n");
    for (g = 0; g < sizeof grids / sizeof grids[0]; ++g) {
        AecConfig cm, ce;
        size_t m5, me, prev = 0;
        int prev_fixed = 0;

        aec_config_from_preset(&cm, AEC_PRESET_BALANCED, grids[g].sr);
        cm.fft_size = grids[g].fft;
        m5 = aec_get_mem_size(&cm);              /* MATCHED n=5 reference */
        ce = cm;
        ce.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        me = aec_get_mem_size(&ce);
        snprintf(msg, sizeof msg, "%s: MATCHED n=5 and EXTERNAL both size",
                 grids[g].tag);
        CHECK(m5 > 0 && me > 0, msg);

        for (d = 0; d < sizeof delay_ms / sizeof delay_ms[0]; ++d) {
            AecConfig cf;
            size_t got, ring_bytes;
            int fixed = (int)(delay_ms[d] * (float)grids[g].sr / 1000.0f);

            cf = cm;
            cf.delay_mode = AEC_DELAY_FIXED;
            cf.fixed_delay_samples = fixed;
            got = aec_get_mem_size(&cf);
            printf("    %-10s fixed=%-7d (%6.1f ms)  %zu B   (MATCHED n=5: %zu B)\n",
                   grids[g].tag, fixed, (double)delay_ms[d], got, m5);
            snprintf(msg, sizeof msg, "%s: FIXED(%d) sizes", grids[g].tag, fixed);
            CHECK(got > 0, msg);

            /* (c) exact composition: EXTERNAL + ALIGN16((fixed+hop)*4). */
            ring_bytes = fixed_ring_bytes(fixed, grids[g].hop);
            snprintf(msg, sizeof msg,
                     "%s: FIXED(%d) == EXTERNAL + ALIGN16((fixed+hop)*4) "
                     "(%zu == %zu + %zu)",
                     grids[g].tag, fixed, got, me, ring_bytes);
            CHECK(got == me + ring_bytes, msg);

            /* (a) monotonic by exactly the ring delta. */
            if (d > 0) {
                size_t step = fixed_ring_bytes(fixed, grids[g].hop)
                            - fixed_ring_bytes(prev_fixed, grids[g].hop);
                snprintf(msg, sizeof msg,
                         "%s: FIXED(%d) - FIXED(%d) == %zu B exactly (%zu - %zu)",
                         grids[g].tag, fixed, prev_fixed, step, got, prev);
                CHECK(got > prev && got - prev == step, msg);
            }
            prev = got; prev_fixed = fixed;
        }

        /* (b) the headline: a 25 ms fixed delay costs a small fraction of the
         * MATCHED n=5 pool, where before step 3 it cost within ~9% of it. */
        {
            AecConfig cf = cm;
            size_t got;
            cf.delay_mode = AEC_DELAY_FIXED;
            cf.fixed_delay_samples = (int)(25.0f * (float)grids[g].sr / 1000.0f);
            got = aec_get_mem_size(&cf);
            snprintf(msg, sizeof msg,
                     "%s: FIXED(25 ms) is under 70%% of MATCHED n=5 "
                     "(%zu vs %zu)", grids[g].tag, got, m5);
            CHECK(got * 10 < m5 * 7, msg);
        }
    }
}

/* ---------------------------------------- heap vs caller-pool, every bank size */

/* Plan §3.4.3. Runs the same signal through aec_create() (library-owned
 * arena) and aec_init() (caller-owned pool) at every bank size and demands
 * sample-exact agreement -- the layout is now n-dependent, so this is where a
 * carve that walked a different order on one of the two paths would show up. */
static void test_heap_vs_pool_parity_every_n(void) {
    char msg[160];
    int n;

    for (n = 1; n <= DA_NUM_FILTERS; ++n) {
        AecConfig cfg;
        Aec heap_inst;
        Aec *pool_inst;
        void *pool = NULL;
        size_t need;
        int i, n_hops = (SR * 2) / HOP, diffs = 0;
        static float out_heap[HOP], out_pool[HOP];

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        cfg.delay_num_filters = n;
        need = aec_get_mem_size(&cfg);
        if (need == 0 || aec_create(&heap_inst, &cfg) != 0) {
            snprintf(msg, sizeof msg, "n=%d: heap/pool parity setup", n);
            CHECK(0, msg);
            continue;
        }
        if (posix_memalign(&pool, 16, need) != 0 || !pool) {
            snprintf(msg, sizeof msg, "n=%d: pool alloc", n);
            CHECK(0, msg); aec_destroy(&heap_inst); continue;
        }
        memset(pool, 0xA5, need);   /* dirty: init must not rely on zeros */
        pool_inst = aec_init(pool, need, &cfg);
        if (!pool_inst) {
            snprintf(msg, sizeof msg, "n=%d: aec_init on a caller pool", n);
            CHECK(0, msg); aec_destroy(&heap_inst); free(pool); continue;
        }

        build_signals(DELAY_NEAR);
        for (i = 0; i < n_hops; ++i) {
            int k;
            aec_process(&heap_inst, g_near + i * HOP, g_far + i * HOP, out_heap);
            aec_process(pool_inst,  g_near + i * HOP, g_far + i * HOP, out_pool);
            for (k = 0; k < HOP; ++k) if (out_heap[k] != out_pool[k]) diffs++;
        }
        snprintf(msg, sizeof msg,
                 "n=%d: aec_create vs aec_init sample-exact over %d hops (%d diffs)",
                 n, n_hops, diffs);
        CHECK(diffs == 0, msg);

        aec_destroy(&heap_inst);
        aec_destroy(pool_inst);
        free(pool);
    }
}

int main(void) {
    printf("=== test_delay_num_filters ===\n");
    test_default_is_five();
    test_geometry();
    test_config_validation();
    test_low_level_init_contract();
    test_pool_geometry_literals();
    test_mem_size_shrinks_with_bank();
    test_fixed_pool_scales_with_the_delay();
    test_heap_vs_pool_parity_every_n();
    printf("---\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
