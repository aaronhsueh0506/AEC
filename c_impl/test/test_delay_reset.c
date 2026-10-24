/* test_delay_reset.c -- regression test: delay_aec_reset() clears the delay estimator's ENTIRE 48 kHz signal chain
 * consistently: the OUTER DaResample48 anti-alias sidechain (filter state + decimation phase) AND the INNER DaEstimator's
 * decimators / ring buffer / pending edge-chunker samples. A reset that cleared only the outer sidechain would leave a mixed
 * fresh/stale state. delay_aec_reset()'s only caller is aec_reset(), a full cold-start-style reset (not a per-echo-path-change
 * nudge), so everything in the chain is cleared together.
 *
 * Four tests:
 *   1. Whitebox: feed one real hop, confirm the inner chain (ring buffer / decimator biquad state / pending_count) holds
 *      nonzero state, then confirm delay_aec_reset() zeroes ALL of it (sidechain AND inner chain).
 *   2. Functional (reset mid-stream at 48 kHz, then delay re-acquisition): lock onto a delay D1, reset() mid-stream, feed an
 *      unrelated delay D2 for a short post-reset window, and confirm the post-reset estimate matches a brand-new instance fed
 *      only that D2 stream, i.e. reset() leaves the object behaviourally equivalent to fresh.
 *   3. Edge-chunker split invariance: the 64-sample block cadence belongs to the stream, not to how the caller slices it; the
 *      same samples fed one at a time, in odd-sized runs, or in whole hops leave the same carved state and the same pending
 *      samples.
 *   4. Edge-chunker overlap: an input that lives inside a pending array is appended sample by sample, so each store is
 *      visible to the next read.
 *
 * Build (standalone, from c_impl/; does NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       src/delay_aec.c src/aec_scale.c test/test_delay_reset.c -lm \
 *       -o bin/test_delay_reset
 * Run:
 *   ./bin/test_delay_reset
 */
#include "delay_aec.h"
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
static void *da_new_pool(DelayAec *d, int sample_rate, int hop, int n) {
    const char *why = NULL;
    void *pool = delay_pool_init(d, sample_rate, hop, n, &why);
    if (!pool) { printf("FAIL: %s\n", why); g_fail++; }
    return pool;
}

/* Feed a delayed-copy-of-noise stream (near = far delayed by delay_samples)
 * through d in `hop`-sample chunks for n_hops hops. Returns the LAST
 * emitted delay_aec_estimated_delay() (native-rate samples), or -1 if the
 * estimator never produced one. */
static int feed_delayed_noise(DelayAec *d, unsigned int seed, int n_hops,
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
        delay_aec_accumulate_ex(d, near + (size_t)i * hop, far + (size_t)i * hop, hop, 1);
        cur = delay_aec_estimated_delay(d);
        if (cur >= 0) last = cur;
    }
    free(far);
    free(near);
    return last;
}

static void test_reset_clears_inner_signal_chain_state(void) {
    DelayAec d;
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

    delay_aec_reset(&d);

    CHECK(!any_nonzero(&d.resample48.near_lp.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.resample48.far_lp.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: outer resample48 biquad state cleared");
    CHECK(d.resample48.phase == 0, "reset: outer resample48 phase cleared");

    CHECK(!any_nonzero(d.est.render_ring.buffer, (size_t)d.est.render_ring.capacity),
          "reset: inner render_ring buffer cleared");
    CHECK(d.est.render_ring.write == 0, "reset: inner render_ring write cursor cleared");
    CHECK(d.est.pending_count == 0,
          "reset: inner pending_count cleared");
    CHECK(!any_nonzero(&d.est.capture_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.est.capture_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: inner capture_decimator biquad state cleared");
    CHECK(!any_nonzero(&d.est.render_decimator.anti_alias.z[0][0], DA_BQ_MAX_SECTIONS * 2) &&
          !any_nonzero(&d.est.render_decimator.noise_reduction.z[0][0], DA_BQ_MAX_SECTIONS * 2),
          "reset: inner render_decimator biquad state cleared");

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
    DelayAec d, fresh;
    void *pool, *pool_fresh;

    pool = da_new_pool(&d, 48000, hop, DA_NUM_FILTERS);
    if (!pool) return;
    got1 = feed_delayed_noise(&d, 11u, n_hops_d1, hop, d1_samples);
    CHECK(got1 >= 0, "pre-reset: delay estimate acquired before reset");

    delay_aec_reset(&d);
    CHECK(delay_aec_estimated_delay(&d) == -1, "reset: latest estimate cleared");

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

/* Feed the stream in the given slicing on ONE pool/struct (re-initialised for
 * each run, so every pointer in the struct is the same across runs) and
 * snapshot the whole struct and pool. estimate_count counts calls that
 * produced an estimate, so it legitimately depends on the slicing and is
 * cleared from the snapshot. */
enum { SPLIT_HOP = 256, SPLIT_HOPS = 140, SPLIT_DELAY = 800 };

static int split_run(DelayAec *d, void *pool, size_t pool_bytes, const float *near,
                     const float *far, int mode, DelayAec *snap, void *pool_snap) {
    static const int ragged[] = {1, 63, 64, 65, 200, 3, 127, 64, 31, 256, 2};
    int total = SPLIT_HOPS * SPLIT_HOP, pos = 0, i = 0;
    memset(pool, 0xA5, pool_bytes);
    if (delay_aec_init(d, pool, pool_bytes, 16000, SPLIT_HOP, DA_NUM_FILTERS) != 0) return -1;
    while (pos < total) {
        int n = mode == 0 ? 1 : mode == 1 ? SPLIT_HOP : ragged[i++ % (int)(sizeof ragged / sizeof ragged[0])];
        if (n > total - pos) n = total - pos;
        delay_aec_accumulate_ex(d, near + pos, far + pos, n, 1);
        pos += n;
    }
    memcpy(snap, d, sizeof *snap);
    snap->estimate_count = 0;
    memcpy(pool_snap, pool, pool_bytes);
    return delay_aec_estimated_delay(d);
}

static void test_edge_chunker_split_invariance(void) {
    size_t pool_bytes = delay_aec_get_mem_size(16000, SPLIT_HOP, DA_NUM_FILTERS);
    int total = SPLIT_HOPS * SPLIT_HOP, i, locked;
    DelayAec d, snap[3];
    void *pool = malloc(pool_bytes), *psnap[3];
    float *far = (float *)malloc((size_t)(total + SPLIT_DELAY) * sizeof(float));
    float *near = (float *)calloc((size_t)(total + SPLIT_DELAY), sizeof(float));
    psnap[0] = malloc(pool_bytes); psnap[1] = malloc(pool_bytes); psnap[2] = malloc(pool_bytes);
    if (!pool || !far || !near || !psnap[0] || !psnap[1] || !psnap[2]) {
        printf("FAIL: allocation\n"); g_fail++; goto out;
    }
    rng_seed(5u);
    for (i = 0; i < total + SPLIT_DELAY; ++i) far[i] = rng_next();
    for (i = SPLIT_DELAY; i < total + SPLIT_DELAY; ++i) near[i] = far[i - SPLIT_DELAY];
    locked = split_run(&d, pool, pool_bytes, near, far, 0, &snap[0], psnap[0]);
    CHECK(locked >= 0, "split invariance: one-sample reference run acquired a delay");
    CHECK(split_run(&d, pool, pool_bytes, near, far, 1, &snap[1], psnap[1]) == locked &&
          !memcmp(&snap[0], &snap[1], sizeof snap[0]) && !memcmp(psnap[0], psnap[1], pool_bytes),
          "split invariance: whole hops leave the state of one-sample feeding");
    CHECK(split_run(&d, pool, pool_bytes, near, far, 2, &snap[2], psnap[2]) == locked &&
          !memcmp(&snap[0], &snap[2], sizeof snap[0]) && !memcmp(psnap[0], psnap[2], pool_bytes),
          "split invariance: ragged runs leave the state of one-sample feeding");
out:
    free(far); free(near); free(pool); free(psnap[0]); free(psnap[1]); free(psnap[2]);
}

/* The sample-by-sample append, spelled out on private copies. */
static void model_append(float *rp, float *cp, int pc, const float *rin, const float *cin, int n) {
    int i;
    for (i = 0; i < n; ++i) { rp[pc + i] = rin[i]; cp[pc + i] = cin[i]; }
}

static void test_edge_chunker_overlapping_input(void) {
    enum { HOP = 128 };
    /* in_render/in_capture: 1 = inside the matching pending array, 2 = inside
     * the other one; off: element offset of the input inside that array. */
    static const struct { int in_render, in_capture, off_r, off_c, n, pre; } cases[] = {
        {1, 0, 0, 0, 40, 10}, {0, 1, 0, 0, 40, 10}, {1, 1, 5, 9, 30, 20},
        {2, 2, 0, 8, 50, 3},  {1, 2, 12, 0, 20, 40}, {0, 2, 0, 4, 60, 0},
    };
    size_t k;
    for (k = 0; k < sizeof cases / sizeof cases[0]; ++k) {
        DelayAec d;
        void *pool = da_new_pool(&d, 16000, HOP, DA_NUM_FILTERS);
        float rp[DA_BLOCK_SIZE], cp[DA_BLOCK_SIZE], seed_in[DA_BLOCK_SIZE], outside[DA_BLOCK_SIZE];
        const float *rin, *cin;
        int j, pc, ok;
        char msg[96];
        if (!pool) return;
        rng_seed(100u + (unsigned)k);
        for (j = 0; j < DA_BLOCK_SIZE; ++j) { seed_in[j] = rng_next(); outside[j] = rng_next(); }
        if (cases[k].pre) delay_aec_accumulate_ex(&d, seed_in, seed_in, cases[k].pre, 1);
        pc = d.est.pending_count;
        memcpy(rp, d.est.render_pending, sizeof rp);
        memcpy(cp, d.est.capture_pending, sizeof cp);
        /* Inputs: inside this instance's arrays (real call) and the same
         * relation inside the model's private copies. */
        {
            const float *real_r = cases[k].in_render == 1 ? d.est.render_pending + cases[k].off_r
                                : cases[k].in_render == 2 ? d.est.capture_pending + cases[k].off_r : outside;
            const float *real_c = cases[k].in_capture == 1 ? d.est.capture_pending + cases[k].off_c
                                : cases[k].in_capture == 2 ? d.est.render_pending + cases[k].off_c : outside;
            rin = cases[k].in_render == 1 ? rp + cases[k].off_r
                : cases[k].in_render == 2 ? cp + cases[k].off_r : outside;
            cin = cases[k].in_capture == 1 ? cp + cases[k].off_c
                : cases[k].in_capture == 2 ? rp + cases[k].off_c : outside;
            /* delay_aec_accumulate_ex(d, near, far, ...): far is the render input. */
            delay_aec_accumulate_ex(&d, real_c, real_r, cases[k].n, 1);
        }
        model_append(rp, cp, pc, rin, cin, cases[k].n);
        ok = d.est.pending_count == pc + cases[k].n &&
             !memcmp(d.est.render_pending, rp, (size_t)(pc + cases[k].n) * sizeof(float)) &&
             !memcmp(d.est.capture_pending, cp, (size_t)(pc + cases[k].n) * sizeof(float));
        snprintf(msg, sizeof msg, "overlap case %d: pending samples equal the one-at-a-time append", (int)k);
        CHECK(ok, msg);
        free(pool);
    }
}

int main(void) {
    test_reset_clears_inner_signal_chain_state();
    test_reset_mid_stream_reacquires_like_fresh();
    test_edge_chunker_split_invariance();
    test_edge_chunker_overlapping_input();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
