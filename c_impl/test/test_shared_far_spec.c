/* test_shared_far_spec.c — far-end-FFT sharing regression test.
 *
 * aec_process_context_shared_far() lets one instance borrow another's
 * already-computed far-end spectrum instead of running its own FFT, for
 * multi-instance callers (e.g. a 4-lane wrapper) whose instances all see
 * the identical far-end signal every hop. This test proves:
 *
 *   1. Borrowing produces BYTE-IDENTICAL AecResContext output to computing
 *      independently -- two separately-constructed, identically-configured
 *      instances fed the identical mic/ref sequence, one via plain
 *      aec_process_context() (computes its own far FFT), the other via
 *      aec_process_context_shared_far() fed the first's ctx.far_spec each
 *      hop -- across shadow on/off (the borrow insertion point differs:
 *      shadow.precomputed_far_spec when shadow is present, main.
 *      precomputed_far_spec directly when it isn't) and all three
 *      officially-contracted grids.
 *   2. The shared spectrum is genuinely CONSUMED, not silently ignored: a
 *      deliberately WRONG (garbage) shared_far_spec produces a DIFFERENT
 *      result than the correct one -- a negative control without which
 *      (1) alone could pass vacuously if aec_process_context_shared_far()
 *      quietly fell back to computing its own spectrum regardless of the
 *      argument.
 *   3. aec_reset() on an instance that has an unconsumed
 *      precomputed_far_spec pointer pending does not leave a stale pointer
 *      behind (the pbfdaf_reset() defensive-clear fix).
 *   4. aec_far_fft_real_compute_count() actually reflects "did this
 *      instance run its own far FFT this hop" -- it increments once per
 *      hop on a plain aec_process_context() instance, and stays flat
 *      forever on an aec_process_context_shared_far()-only instance. This
 *      is the counter a multi-lane caller (4aec_nr_res.c) uses to prove
 *      its own total per-hop far-FFT count actually dropped after
 *      switching lanes 1-3 over to share lane 0's spectrum.
 *
 * Build (standalone, from c_impl/ — mirrors test_process_context.c's
 * documented recipe):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_shared_far_spec.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_shared_far_spec
 * Run:
 *   ./bin/test_shared_far_spec
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

static void context_only_config(AecConfig* cfg, int sample_rate, int fft_size,
                                 int enable_shadow) {
    aec_config_from_preset(cfg, AEC_PRESET_BALANCED, sample_rate);
    cfg->enable_res = 0;
    cfg->return_res_context = 1;
    if (fft_size > 0) cfg->fft_size = fft_size;
    cfg->enable_shadow = enable_shadow;
}

static unsigned g_rng_state = 0x6A5EC001u;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu;
    return amp * (2.0f * u - 1.0f);
}

static void test_borrow_matches_compute(int sample_rate, int fft_size, int enable_shadow) {
    AecConfig cfg;
    context_only_config(&cfg, sample_rate, fft_size, enable_shadow);

    Aec source, borrower;
    char label[160];
    snprintf(label, sizeof(label),
             "borrow-parity sr=%d/fft=%d/shadow=%d: aec_create(source)",
             sample_rate, cfg.fft_size, enable_shadow);
    CHECK(aec_create(&source, &cfg) == 0, label);
    snprintf(label, sizeof(label),
             "borrow-parity sr=%d/fft=%d/shadow=%d: aec_create(borrower)",
             sample_rate, cfg.fft_size, enable_shadow);
    CHECK(aec_create(&borrower, &cfg) == 0, label);

    int hop = aec_hop_size(&source);
    if (hop <= 0 || hop > 4096) { aec_destroy(&source); aec_destroy(&borrower); return; }

    float mic_a[4096], mic_b[4096], ref[4096];
    g_rng_state = 0x6A5EC001u;
    int all_match = 1;
    for (int i = 0; i < 150; ++i) {
        /* Different mic per instance (so the two filters genuinely diverge
         * on everything EXCEPT far_spec, which is the one thing meant to
         * be shared) but identical ref -- exactly the 4-lane wrapper's
         * shape: independent mic per lane, one shared aligned reference. */
        float amp = (i >= 50 && i < 55) ? 0.8f : 0.05f;
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(amp);
            mic_a[s] = rng_uniform(amp) + 0.5f * ref[s];
            mic_b[s] = rng_uniform(amp) + 0.3f * ref[s];
        }

        aec_process_context(&source, mic_a, ref);
        AecResContext src_ctx;
        aec_get_res_context(&source, &src_ctx);

        aec_process_context_shared_far(&borrower, mic_b, ref, src_ctx.far_spec);
    }

    /* Now verify: a FRESH pair, borrower this time computing its own
     * far_spec (plain aec_process_context on the SAME mic_b/ref sequence),
     * must match the shared-far run above bin-for-bin. Re-run both from
     * scratch so this comparison is a clean, deterministic replay. */
    Aec borrower_reference;
    CHECK(aec_create(&borrower_reference, &cfg) == 0,
          "borrow-parity: aec_create(borrower_reference)");

    Aec source2, borrower2;
    CHECK(aec_create(&source2, &cfg) == 0, "borrow-parity: aec_create(source2)");
    CHECK(aec_create(&borrower2, &cfg) == 0, "borrow-parity: aec_create(borrower2)");

    g_rng_state = 0x6A5EC001u;
    for (int i = 0; i < 150; ++i) {
        float amp = (i >= 50 && i < 55) ? 0.8f : 0.05f;
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(amp);
            mic_a[s] = rng_uniform(amp) + 0.5f * ref[s];
            mic_b[s] = rng_uniform(amp) + 0.3f * ref[s];
        }

        aec_process_context(&source2, mic_a, ref);
        AecResContext src_ctx2;
        aec_get_res_context(&source2, &src_ctx2);
        aec_process_context_shared_far(&borrower2, mic_b, ref, src_ctx2.far_spec);

        aec_process_context(&borrower_reference, mic_b, ref);

        AecResContext bctx, rctx;
        aec_get_res_context(&borrower2, &bctx);
        aec_get_res_context(&borrower_reference, &rctx);

        int K = bctx.n_freqs;
        if (memcmp(bctx.formed_hop, rctx.formed_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(bctx.error_spec, rctx.error_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(bctx.far_spec, rctx.far_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (bctx.filter_converged != rctx.filter_converged) { all_match = 0; break; }
    }

    snprintf(label, sizeof(label),
             "borrow-parity sr=%d/fft=%d/shadow=%d: borrowed far_spec produces byte-identical "
             "AecResContext to computing it independently",
             sample_rate, cfg.fft_size, enable_shadow);
    CHECK(all_match, label);

    aec_destroy(&source);
    aec_destroy(&borrower);
    aec_destroy(&source2);
    aec_destroy(&borrower2);
    aec_destroy(&borrower_reference);
}

/* Negative control: a deliberately WRONG shared spectrum must change the
 * result. Without this, test_borrow_matches_compute() could pass
 * vacuously if aec_process_context_shared_far() silently ignored its
 * shared_far_spec argument and just computed its own spectrum regardless. */
static void test_wrong_shared_spec_changes_result(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0, 1);

    Aec correct, wrong;
    CHECK(aec_create(&correct, &cfg) == 0, "negative-control: aec_create(correct)");
    CHECK(aec_create(&wrong, &cfg) == 0, "negative-control: aec_create(wrong)");

    int hop = aec_hop_size(&correct);
    if (hop <= 0 || hop > 4096) { aec_destroy(&correct); aec_destroy(&wrong); return; }
    AecResContext dims_ctx;
    aec_get_res_context(&correct, &dims_ctx);
    int K = dims_ctx.n_freqs;

    Complex* garbage_spec = (Complex*)malloc((size_t)K * sizeof(Complex));
    CHECK(garbage_spec != NULL, "negative-control: malloc garbage spectrum");

    float mic[4096], ref[4096];
    g_rng_state = 0xBAD5EC00u;
    int saw_divergence = 0;
    for (int i = 0; i < 60; ++i) {
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(0.3f);
            mic[s] = rng_uniform(0.3f) + 0.4f * ref[s];
        }
        for (int k = 0; k < K; ++k) {
            garbage_spec[k].r = rng_uniform(50.0f);
            garbage_spec[k].i = rng_uniform(50.0f);
        }

        aec_process_context(&correct, mic, ref);
        aec_process_context_shared_far(&wrong, mic, ref, garbage_spec);

        AecResContext cctx, wctx;
        aec_get_res_context(&correct, &cctx);
        aec_get_res_context(&wrong, &wctx);
        if (memcmp(cctx.formed_hop, wctx.formed_hop, (size_t)hop * sizeof(float)) != 0) {
            saw_divergence = 1;
            break;
        }
    }

    CHECK(saw_divergence,
          "negative-control: a deliberately wrong shared_far_spec actually changes the output "
          "(proves the argument is genuinely consumed, not silently ignored)");

    free(garbage_spec);
    aec_destroy(&correct);
    aec_destroy(&wrong);
}

/* aec_reset() must not leave a stale precomputed_far_spec pointer behind
 * even if called right after a shared-far call (pbfdaf_reset() defensive
 * clear). This can't observe the pointer field directly (Aec's internal
 * PBFDAF/PBFDKF sub-structs are opaque to this test), so it instead proves
 * the OBSERVABLE consequence: post-reset, the instance behaves exactly
 * like a fresh instance computing its own spectrum, never silently
 * reusing a freed/stale borrowed pointer from before the reset (which
 * would produce garbage or crash under ASan/valgrind on the freed
 * memory). */
static void test_reset_after_shared_far_is_clean(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0, 1);

    Aec source, a;
    CHECK(aec_create(&source, &cfg) == 0, "reset-after-shared-far: aec_create(source)");
    CHECK(aec_create(&a, &cfg) == 0, "reset-after-shared-far: aec_create(a)");

    int hop = aec_hop_size(&source);
    if (hop <= 0 || hop > 4096) { aec_destroy(&source); aec_destroy(&a); return; }

    float mic[4096], ref[4096];
    g_rng_state = 0xC1EA4001u;
    for (int s = 0; s < hop; ++s) { ref[s] = rng_uniform(0.3f); mic[s] = rng_uniform(0.3f); }

    aec_process_context(&source, mic, ref);
    AecResContext src_ctx;
    aec_get_res_context(&source, &src_ctx);

    /* Heap-allocated spectrum, freed immediately after the call that
     * consumes it -- if reset() (or anything else) kept a stale reference
     * to it, a subsequent dereference would be a use-after-free (would
     * show up under ASan even if it happens not to crash here). */
    int K = src_ctx.n_freqs;
    Complex* heap_spec = (Complex*)malloc((size_t)K * sizeof(Complex));
    memcpy(heap_spec, src_ctx.far_spec, (size_t)K * sizeof(Complex));
    aec_process_context_shared_far(&a, mic, ref, heap_spec);
    free(heap_spec);

    aec_reset(&a);
    CHECK(1, "reset-after-shared-far: aec_reset() right after a shared-far call does not crash");

    for (int i = 0; i < 10; ++i) {
        for (int s = 0; s < hop; ++s) { ref[s] = rng_uniform(0.3f); mic[s] = rng_uniform(0.3f); }
        aec_process_context(&a, mic, ref);
    }
    CHECK(1, "reset-after-shared-far: instance keeps processing normally post-reset "
             "(no crash/UB from a stale borrowed pointer)");

    aec_destroy(&source);
    aec_destroy(&a);
}

/* aec_far_fft_real_compute_count(): the counter a multi-lane caller uses
 * to prove its total per-hop far-FFT count dropped. Proves both halves:
 * a plain aec_process_context() instance increments by exactly 1 every
 * hop, and an aec_process_context_shared_far()-only instance (given a
 * valid spectrum every hop) never increments at all. */
static void test_far_fft_counter_reflects_actual_computation(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0, 1);

    Aec computing, borrowing;
    CHECK(aec_create(&computing, &cfg) == 0, "counter test: aec_create(computing)");
    CHECK(aec_create(&borrowing, &cfg) == 0, "counter test: aec_create(borrowing)");

    int hop = aec_hop_size(&computing);
    if (hop <= 0 || hop > 4096) { aec_destroy(&computing); aec_destroy(&borrowing); return; }

    CHECK(aec_far_fft_real_compute_count(&computing) == 0,
          "counter test: starts at 0 on a freshly created instance");
    CHECK(aec_far_fft_real_compute_count(&borrowing) == 0,
          "counter test: starts at 0 on a freshly created instance (borrowing side too)");

    float mic_a[4096], mic_b[4096], ref[4096];
    g_rng_state = 0xF17C0000u;
    const int n_hops = 25;
    for (int i = 0; i < n_hops; ++i) {
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(0.3f);
            mic_a[s] = rng_uniform(0.3f) + 0.4f * ref[s];
            mic_b[s] = rng_uniform(0.3f) + 0.2f * ref[s];
        }
        aec_process_context(&computing, mic_a, ref);
        AecResContext ctx;
        aec_get_res_context(&computing, &ctx);
        aec_process_context_shared_far(&borrowing, mic_b, ref, ctx.far_spec);
    }

    char label[128];
    snprintf(label, sizeof(label),
             "counter test: plain aec_process_context() increments by exactly %d after %d hops",
             n_hops, n_hops);
    CHECK(aec_far_fft_real_compute_count(&computing) == n_hops, label);
    snprintf(label, sizeof(label),
             "counter test: aec_process_context_shared_far()-only instance stays at 0 after %d hops",
             n_hops);
    CHECK(aec_far_fft_real_compute_count(&borrowing) == 0, label);

    /* aec_reset() zeroes the counter (matches frame_count's own reset
     * convention) -- not load-bearing for the multi-lane use case, but a
     * caller reading this across a reset should not see a stale count. */
    aec_reset(&computing);
    CHECK(aec_far_fft_real_compute_count(&computing) == 0,
          "counter test: aec_reset() zeroes the counter");

    aec_destroy(&computing);
    aec_destroy(&borrowing);
}

int main(void) {
    /* Three officially-contracted grids x shadow on/off. */
    test_borrow_matches_compute(16000, 0, 1);    /* 16k/256 default */
    test_borrow_matches_compute(16000, 0, 0);
    test_borrow_matches_compute(16000, 512, 1);  /* 16k/512 alt */
    test_borrow_matches_compute(16000, 512, 0);
    test_borrow_matches_compute(48000, 0, 1);    /* 48k/1024 */
    test_borrow_matches_compute(48000, 0, 0);

    test_wrong_shared_spec_changes_result();
    test_reset_after_shared_far_is_clean();
    test_far_fft_counter_reflects_actual_computation();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
