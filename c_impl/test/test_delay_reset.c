/* test_delay_reset.c — regression test: verify
 * delay_aec3_reset() clears the delay estimator's ENTIRE 48kHz signal chain
 * consistently -- both the OUTER DaResample48 anti-alias sidechain (filter
 * state + decimation phase) AND the INNER DaEstimator's decimators / ring
 * buffer / pending edge-chunker samples.
 *
 * Bug this guards against: reset() used to clear only the outer sidechain
 * (added alongside the sidechain itself), leaving the inner chain's biquad
 * memory / ring-buffer audio history / pending partial-block samples stale
 * -- a mixed fresh/stale reset. Call-site audit
 * (delay_aec3_reset's only caller is aec_reset(), a full top-level
 * cold-start-style reset -- see aec.c / test_lifecycle.c / bench_rtf.c, NOT
 * a lightweight per-echo-path-change nudge) established that the correct
 * fix is a FULL reset: everything in the chain gets cleared together.
 *
 * Two tests:
 *   1. Whitebox: feed one real hop, confirm the inner chain (ring buffer /
 *      decimator biquad state / pending_count) holds nonzero state, then
 *      confirm delay_aec3_reset() zeroes ALL of it (sidechain AND inner
 *      chain). This is the exact assertion the pre-fix code fails.
 *   2. Functional (reset mid-stream at 48kHz, then verify delay
 *      re-acquisition): lock onto a delay D1, reset() mid-stream, feed an
 *      unrelated delay D2 for a short post-reset window, and confirm the
 *      post-reset estimate matches a brand-new instance fed only that same
 *      D2 stream -- reset() should leave the object behaviourally
 *      equivalent to fresh for anything that follows.
 *
 * Build (standalone, from c_impl/ -- mirrors test/parity_delay.c's
 * documented recipe, does NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       -I../../audio_common/include \
 *       src/delay_aec3.c src/aec3_scale.c test/test_delay_reset.c -lm \
 *       -o bin/test_delay_reset
 * Run:
 *   ./bin/test_delay_reset
 */
#include "delay_aec3.h"
#include "delay_pool_test_util.h"

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

/* Deterministic xorshift32 PRNG -- avoids relying on libc rand()'s
 * platform-specific seeding/sequence. */
static unsigned int g_rng_state;
static void rng_seed(unsigned int s) { g_rng_state = s ? s : 1u; }
static float rng_next(void) {
    unsigned int x = g_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng_state = x;
    /* map to roughly [-0.2, 0.2]: well above the excitation gate
     * (DA_EXCITATION_LIMIT-derived x2_sum threshold), well below
     * DA_SATURATION_LIMIT. */
    return ((float)(x % 40000u) / 40000.0f - 0.5f) * 0.4f;
}

static int any_nonzero(const float *a, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) if (a[i] != 0.0f) return 1;
    return 0;
}

/* Shared pool-first construction (delay_pool_test_util.h), reported in this
 * file's printf + counter style. */
static void *da_new_pool(DelayAec3 *d, int sample_rate, int hop, int n) {
    const char *why = NULL;
    void *pool = delay_pool_init(d, sample_rate, hop, n, &why);
    if (!pool) { printf("FAIL: %s\n", why); g_fail++; }
    return pool;
}

/* Feed a delayed-copy-of-noise stream (near = far delayed by delay_samples)
 * through d in `hop`-sample chunks for n_hops hops. Returns the LAST
 * emitted delay_aec3_estimated_delay() (native-rate samples), or -1 if the
 * estimator never produced one. */
static int feed_delayed_noise(DelayAec3 *d, unsigned int seed, int n_hops,
                              int hop, int delay_samples) {
    int total = n_hops * hop + delay_samples;
    float *far = (float *)malloc((size_t)total * sizeof(float));
    float *near = (float *)calloc((size_t)total, sizeof(float));
    int i, last = -1;
    rng_seed(seed);
    for (i = 0; i < total; ++i) far[i] = rng_next();
    for (i = delay_samples; i < total; ++i) near[i] = far[i - delay_samples];
    for (i = 0; i < n_hops; ++i) {
        int cur;
        delay_aec3_accumulate(d, near + (size_t)i * hop, far + (size_t)i * hop, hop);
        cur = delay_aec3_estimated_delay(d);
        if (cur >= 0) last = cur;
    }
    free(far);
    free(near);
    return last;
}

static void test_reset_clears_inner_signal_chain_state(void) {
    DelayAec3 d;
    int hop = 480; /* 10 ms @ 48 kHz */
    void *pool = da_new_pool(&d, 48000, hop, DA_NUM_FILTERS);
    if (!pool) return;

    feed_delayed_noise(&d, 7u, 1, hop, 0);

    int pre_ring = any_nonzero(d.est.render_ring.buffer, (size_t)d.est.render_ring.capacity);
    int pre_pending = (d.est.pending_count != 0);
    int pre_dec = any_nonzero(&d.est.capture_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2)
               || any_nonzero(&d.est.render_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2)
               || any_nonzero(&d.est.capture_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2)
               || any_nonzero(&d.est.render_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2);
    CHECK(pre_ring || pre_pending || pre_dec,
          "pre-reset sanity: inner chain holds nonzero state after one hop");

    delay_aec3_reset(&d);

    CHECK(!any_nonzero(&d.resample48.near_lp.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.resample48.far_lp.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: outer resample48 biquad state cleared");
    CHECK(d.resample48.phase == 0, "reset: outer resample48 phase cleared");

    CHECK(!any_nonzero(d.est.render_ring.buffer, (size_t)d.est.render_ring.capacity),
          "reset: inner render_ring buffer cleared (was stale pre-fix)");
    CHECK(d.est.render_ring.write == 0, "reset: inner render_ring write cursor cleared");
    CHECK(d.est.pending_count == 0,
          "reset: inner pending_count cleared (was stale pre-fix)");
    CHECK(!any_nonzero(&d.est.capture_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.est.capture_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: inner capture_decimator biquad state cleared (was stale pre-fix)");
    CHECK(!any_nonzero(&d.est.render_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.est.render_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: inner render_decimator biquad state cleared (was stale pre-fix)");

    /* reset() clears STATE, never geometry: the pool pointers and every
     * runtime size must survive it, or a post-reset hop would walk a
     * differently-shaped buffer than the one that was carved. */
    CHECK(d.est.render_ring.capacity == DA_RING_CAPACITY_FOR(DA_NUM_FILTERS) &&
          d.est.matched_filter.num_filters == DA_NUM_FILTERS &&
          d.est.aggregator.highest_peak.hist_size == DA_HP_HIST_SIZE_FOR(DA_NUM_FILTERS) &&
          d.est.aggregator.pre_echo.hist_size == DA_PE_HIST_SIZE_FOR(DA_NUM_FILTERS),
          "reset: pool geometry (ring capacity, bank size, both histogram sizes) survives");
    CHECK(d.est.render_ring.buffer && d.est.matched_filter.filters &&
          d.est.matched_filter.accumulated_error &&
          d.est.matched_filter.instantaneous_error &&
          d.est.aggregator.highest_peak.histogram &&
          d.est.aggregator.highest_peak.ring &&
          d.est.aggregator.pre_echo.histogram &&
          d.est.aggregator.pre_echo.ring &&
          d.near16_scratch && d.far16_scratch,
          "reset: every pool pointer survives (incl. the 48 kHz sidechain scratch)");

    free(pool);
}

static void test_reset_mid_stream_reacquires_like_fresh(void) {
    int hop = 480;          /* 10 ms @ 48 kHz */
    int n_hops_d1 = 400;    /* 4 s: lock onto D1 before reset */
    int n_hops_d2 = 60;     /* 600 ms post-reset: short window maximises
                             * sensitivity to any leftover contamination */
    int d1_samples = 4800;  /* 100 ms */
    int d2_samples = 14400; /* 300 ms */
    int got1, got_after_reset, got_fresh;
    DelayAec3 d, fresh;
    void *pool, *pool_fresh;

    pool = da_new_pool(&d, 48000, hop, DA_NUM_FILTERS);
    if (!pool) return;
    got1 = feed_delayed_noise(&d, 11u, n_hops_d1, hop, d1_samples);
    CHECK(got1 >= 0, "pre-reset: delay estimate acquired before reset");

    delay_aec3_reset(&d);
    CHECK(delay_aec3_estimated_delay(&d) == -1, "reset: latest estimate cleared");

    got_after_reset = feed_delayed_noise(&d, 22u, n_hops_d2, hop, d2_samples);

    pool_fresh = da_new_pool(&fresh, 48000, hop, DA_NUM_FILTERS);
    if (!pool_fresh) { free(pool); return; }
    got_fresh = feed_delayed_noise(&fresh, 22u, n_hops_d2, hop, d2_samples);

    CHECK(got_after_reset >= 0, "post-reset: delay re-acquired after reset");
    CHECK(got_fresh >= 0, "fresh baseline: delay acquired");
    CHECK(abs(got_fresh - d2_samples) <= 192,
          "48kHz estimator acquires a 300-ms delay beyond the incorrect "
          "203-ms documentation limit");
    CHECK(got_after_reset == got_fresh,
          "post-reset estimate matches a fresh instance fed the same stream "
          "(no leftover contamination from the pre-reset stream)");

    free(pool);
    free(pool_fresh);
}

int main(void) {
    test_reset_clears_inner_signal_chain_state();
    test_reset_mid_stream_reacquires_like_fresh();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
