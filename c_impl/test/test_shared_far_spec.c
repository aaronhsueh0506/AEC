/* test_shared_far_spec.c -- far-end-FFT sharing regression test.
 *
 * aec_process_linear_shared_far_from() lets one instance borrow another's already-computed far-end spectrum (and its squared
 * magnitudes) instead of running its own FFT, for multi-instance callers (e.g. a 4-lane wrapper) whose instances see the
 * identical far-end signal every hop. This test proves:
 *   1. Borrowing produces BYTE-IDENTICAL AecResidualView output to computing independently: two separately constructed,
 *      identically configured instances fed the identical mic/ref sequence, one via aec_process_linear() (own far FFT), the
 *      other via aec_process_linear_shared_far_from() borrowing the first instance's spectrum each hop, across shadow on/off
 *      (the borrow insertion point differs: shadow.precomputed_far_spec when a shadow is present, main.precomputed_far_spec
 *      otherwise) and the three contracted grids.
 *   2. The shared spectrum is genuinely CONSUMED: borrowing from an owner that processed a DIFFERENT reference gives a
 *      DIFFERENT result than computing locally. Without this negative control, (1) could pass if the function quietly
 *      computed its own spectrum.
 *   3. aec_reset() on an instance that borrowed leaves no stale pointer behind (the pbfdaf_reset() defensive clear): the
 *      owner is destroyed after the reset and the instance keeps processing.
 *   4. aec_far_fft_real_compute_count() reflects "did this instance run its own far FFT this hop": it increments once per
 *      hop on an aec_process_linear() instance and stays flat on an aec_process_linear_shared_far_from()-only instance. A
 *      multi-lane caller (the array instance, aec_array.c) sums it over its four lanes; the total rises by one per
 *      hop, not four, once lanes 1-3 share lane 0's spectrum (a hop that mixes warm and soft realignments computes on all four lanes).
 *   5. The hop counters (frame_count, shadow_frame_count) saturate at LONG_MAX, and a shadow hop count past INT_MAX still
 *      reads as past the shadow warm-up gate where the detectors narrow it to int.
 *
 * Build (standalone, from c_impl/):
 *   make -C ../../audiolib/common BACKEND=ne10 lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       test/test_shared_far_spec.c $(find src -name '*.c') \
 *       $(make -s -C ../../audiolib/common BACKEND=ne10 print-lib-path) -lm \
 *       -o bin/test_shared_far_spec
 * Run:
 *   ./bin/test_shared_far_spec
 */
#include "aec.h"
#include "aec_internal.h"
#include <math.h>
#include <limits.h>
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

/* The split-entry configuration: residual FULL, driven by aec_process_linear() + aec_process_residual(). */
static void context_only_config(AecConfig* cfg, int sample_rate, int fft_size,
                                 int enable_shadow) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, sample_rate);
    cfg->residual = AEC_RESIDUAL_FULL;
    if (fft_size > 0) cfg->fft_size = fft_size;
    cfg->enable_shadow = enable_shadow;
}

/* One hop of a split-entry instance: the linear stage (far spectrum local, or borrowed from `owner`), then the residual
 * stage. */
static void hop_local(Aec* a, const float* mic, const float* ref) {
    if (aec_process_linear(a, mic, ref) != AEC_OK || aec_process_residual(a, NULL) != AEC_OK) {
        printf("FAIL: split hop refused\n"); g_fail++;
    }
}

static void hop_shared(Aec* a, const float* mic, const float* ref, const Aec* owner) {
    if (aec_process_linear_shared_far_from(a, mic, ref, owner) != AEC_OK || aec_process_residual(a, NULL) != AEC_OK) {
        printf("FAIL: shared split hop refused\n"); g_fail++;
    }
}

static void get_views(const Aec* a, AecLinearView* lv, AecResidualView* rv) {
    memset(lv, 0, sizeof *lv); lv->struct_size = sizeof *lv;
    memset(rv, 0, sizeof *rv); rv->struct_size = sizeof *rv;
    if (aec_get_linear(a, lv) != AEC_OK || aec_get_residual(a, rv) != AEC_OK) {
        printf("FAIL: view refused\n"); g_fail++;
    }
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

        hop_local(&source, mic_a, ref);

        hop_shared(&borrower, mic_b, ref, &source);
    }

    /* Now verify: a FRESH pair, borrower this time computing its own
     * far_spec (plain aec_process_linear + aec_process_residual on the SAME mic_b/ref sequence),
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

        hop_local(&source2, mic_a, ref);
        hop_shared(&borrower2, mic_b, ref, &source2);

        hop_local(&borrower_reference, mic_b, ref);

        AecLinearView blv, rlv;
        AecResidualView bctx, rctx;
        get_views(&borrower2, &blv, &bctx);
        get_views(&borrower_reference, &rlv, &rctx);

        int K = bctx.n_freqs;
        if (memcmp(bctx.formed_hop, rctx.formed_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(bctx.error_spec, rctx.error_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(bctx.far_spec, rctx.far_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (blv.filter_converged != rlv.filter_converged) { all_match = 0; break; }
    }

    snprintf(label, sizeof(label),
             "borrow-parity sr=%d/fft=%d/shadow=%d: borrowed far_spec produces byte-identical "
             "AecResidualView to computing it independently",
             sample_rate, cfg.fft_size, enable_shadow);
    CHECK(all_match, label);

    aec_destroy(&source);
    aec_destroy(&borrower);
    aec_destroy(&source2);
    aec_destroy(&borrower2);
    aec_destroy(&borrower_reference);
}

/* Negative control: borrowing the spectrum of an owner that processed a DIFFERENT
 * reference must change the result. Without this, test_borrow_matches_compute()
 * could pass vacuously if aec_process_linear_shared_far_from() silently ignored
 * the owner and just computed its own spectrum regardless. */
static void test_wrong_shared_spec_changes_result(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0, 1);

    Aec correct, wrong, other_owner;
    CHECK(aec_create(&correct, &cfg) == 0, "negative-control: aec_create(correct)");
    CHECK(aec_create(&wrong, &cfg) == 0, "negative-control: aec_create(wrong)");
    CHECK(aec_create(&other_owner, &cfg) == 0, "negative-control: aec_create(other_owner)");

    int hop = aec_hop_size(&correct);
    if (hop <= 0 || hop > 4096) {
        aec_destroy(&correct); aec_destroy(&wrong); aec_destroy(&other_owner);
        return;
    }

    float mic[4096], ref[4096], other_ref[4096];
    g_rng_state = 0xBAD5EC00u;
    int saw_divergence = 0;
    for (int i = 0; i < 60; ++i) {
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(0.3f);
            other_ref[s] = rng_uniform(0.3f);
            mic[s] = rng_uniform(0.3f) + 0.4f * ref[s];
        }

        hop_local(&correct, mic, ref);
        hop_local(&other_owner, mic, other_ref);
        hop_shared(&wrong, mic, ref, &other_owner);

        AecLinearView clv, wlv;
        AecResidualView cctx, wctx;
        get_views(&correct, &clv, &cctx);
        get_views(&wrong, &wlv, &wctx);
        if (memcmp(cctx.formed_hop, wctx.formed_hop, (size_t)hop * sizeof(float)) != 0) {
            saw_divergence = 1;
            break;
        }
    }

    CHECK(saw_divergence,
          "negative-control: borrowing a different reference's spectrum actually changes the output "
          "(proves the owner is genuinely consumed, not silently ignored)");

    aec_destroy(&correct);
    aec_destroy(&wrong);
    aec_destroy(&other_owner);
}

/* aec_reset() must not leave a stale precomputed_far_spec pointer behind, even right after a
 * shared-far call (pbfdaf_reset() defensive clear). The pointer field is not observable from here
 * (the sub-structs are opaque), so this proves the observable consequence: the owner whose
 * spectrum was borrowed is destroyed after the reset, and the instance keeps processing without
 * reusing a stale borrowed pointer (garbage output, or a crash under ASan/valgrind on the freed
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

    hop_local(&source, mic, ref);
    hop_shared(&a, mic, ref, &source);

    /* The owner's pool is freed right after the reset -- if reset() (or anything else) kept a
     * stale reference to its spectrum, a subsequent dereference would be a use-after-free (would
     * show up under ASan even if it happens not to crash here). */
    aec_reset(&a);
    CHECK(1, "reset-after-shared-far: aec_reset() right after a shared-far call does not crash");
    aec_destroy(&source);

    for (int i = 0; i < 10; ++i) {
        for (int s = 0; s < hop; ++s) { ref[s] = rng_uniform(0.3f); mic[s] = rng_uniform(0.3f); }
        hop_local(&a, mic, ref);
    }
    CHECK(1, "reset-after-shared-far: instance keeps processing normally post-reset "
             "(no crash/UB from a stale borrowed pointer)");

    aec_destroy(&a);
}

/* aec_far_fft_real_compute_count(): the counter a multi-lane caller uses
 * to prove its total per-hop far-FFT count dropped. Proves both halves:
 * a plain aec_process_linear() instance increments by exactly 1 every
 * hop, and an aec_process_linear_shared_far_from()-only instance (given a
 * valid owner every hop) never increments at all. */
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
        hop_local(&computing, mic_a, ref);
        hop_shared(&borrowing, mic_b, ref, &computing);
    }

    char label[128];
    snprintf(label, sizeof(label),
             "counter test: plain aec_process_linear() increments by exactly %d after %d hops",
             n_hops, n_hops);
    CHECK(aec_far_fft_real_compute_count(&computing) == n_hops, label);
    snprintf(label, sizeof(label),
             "counter test: aec_process_linear_shared_far_from()-only instance stays at 0 after %d hops",
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


static void test_far_fft_counter_saturation(void) {
    Aec a;
    memset(&a, 0, sizeof(a));
    a.ready = 1;                    /* a hand-built handle: only a ready one is read */
    a.main_filter.base.far_fft_real_compute_count = 7;
    a.shadow_filter.far_fft_real_compute_count = 11;
    a.observer_filter.far_fft_real_compute_count = 13;
    a.has_shadow = a.has_observer = 1;
    CHECK(aec_far_fft_real_compute_count(&a) == 31, "counter sum: ordinary values");
    a.main_filter.base.far_fft_real_compute_count = LONG_MAX - 3;
    a.shadow_filter.far_fft_real_compute_count = 2;
    a.observer_filter.far_fft_real_compute_count = 2;
    CHECK(aec_far_fft_real_compute_count(&a) == LONG_MAX, "counter sum: observer addition saturates");
    a.shadow_filter.far_fft_real_compute_count = LONG_MAX;
    CHECK(aec_far_fft_real_compute_count(&a) == LONG_MAX, "counter sum: shadow addition saturates");
    a.has_shadow = a.has_observer = 0;
    CHECK(aec_far_fft_real_compute_count(&a) == LONG_MAX - 3, "counter sum: disabled filters ignored");

    AecConfig cfg;
    context_only_config(&cfg, 16000, 0, 0);
    if (aec_create(&a, &cfg) != 0) { CHECK(0, "counter saturation: create"); return; }
    float zero[4096] = {0};
    a.main_filter.base.far_fft_real_compute_count = LONG_MAX - 1;
    hop_local(&a, zero, zero);
    CHECK(a.main_filter.base.far_fft_real_compute_count == LONG_MAX, "counter increment reaches LONG_MAX");
    hop_local(&a, zero, zero);
    CHECK(a.main_filter.base.far_fft_real_compute_count == LONG_MAX, "counter increment stays at LONG_MAX");
    aec_reset(&a);
    CHECK(aec_far_fft_real_compute_count(&a) == 0, "saturated counter resets to zero");
    aec_destroy(&a);
}

/* Hop counters hold at LONG_MAX instead of overflowing signed long (32 bits on
 * ILP32). Seeding one hop below the limit and running past it proves the clamp on the
 * real increment sites; each field must also return to zero on reset. */
static void test_hop_counter_saturation(void) {
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.enable_shadow = 1;
    Aec a;
    if (aec_create(&a, &cfg) != 0) { CHECK(0, "hop counter saturation: create"); return; }
    CHECK(a.has_shadow, "hop counter saturation: instance has a shadow filter");
    int hop = aec_hop_size(&a);
    float ref[4096] = {0}, mic[4096] = {0}, out[4096];
    if (hop <= 0 || hop > 4096) { aec_destroy(&a); return; }

    a.frame_count = LONG_MAX - 1;
    a.shadow_frame_count = LONG_MAX - 1;
    for (int i = 0; i < 3; ++i) aec_process(&a, mic, ref, out);
    CHECK(a.frame_count == LONG_MAX, "frame_count stops at LONG_MAX");
    CHECK(a.shadow_frame_count == LONG_MAX, "shadow_frame_count stops at LONG_MAX");
    aec_reset(&a);
    CHECK(a.frame_count == 0 && a.shadow_frame_count == 0, "saturated hop counters reset to zero");
    aec_destroy(&a);
}

/* The detectors take the shadow hop count as int. A count past INT_MAX must still read as
 * "past the warm-up gate": run two identical instances, one seeded far above the gate but
 * inside int range and one seeded past INT_MAX, and require identical state and output.
 * Narrowing modulo 2^32 would make the second count negative, so its shadow double-talk
 * and copy decisions would stay closed and the two would diverge. */
static void test_shadow_count_past_int_max(void) {
    if (sizeof(long) <= sizeof(int)) {
        CHECK(1, "shadow count past INT_MAX: not representable when long is 32 bits (skipped)");
        return;
    }
    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.enable_shadow = 1;
    Aec lo, hi;
    if (aec_create(&lo, &cfg) != 0 || aec_create(&hi, &cfg) != 0) {
        CHECK(0, "shadow count past INT_MAX: create");
        return;
    }
    int hop = aec_hop_size(&lo);
    if (hop <= 0 || hop > 4096) { aec_destroy(&lo); aec_destroy(&hi); return; }
    lo.shadow_frame_count = 100000;
    /* Built through unsigned so a 32-bit long does not trip -Woverflow on this unreachable arm. */
    hi.shadow_frame_count = (long)((unsigned long)INT_MAX + 5ul);

    float mic[4096], ref[4096], out_lo[4096], out_hi[4096];
    g_rng_state = 0x5AD0F00Du;
    int same = 1, moved = 0;
    for (int i = 0; i < 80; ++i) {
        for (int s = 0; s < hop; ++s) {
            ref[s] = rng_uniform(0.3f);
            mic[s] = rng_uniform(0.05f) + 0.5f * ref[s];
        }
        aec_process(&lo, mic, ref, out_lo);
        aec_process(&hi, mic, ref, out_hi);
        if (memcmp(out_lo, out_hi, (size_t)hop * sizeof(float)) != 0) same = 0;
        if (lo.dt_analyzer.shadow_advantage != 1.0f) moved = 1;
    }
    CHECK(moved, "shadow count past INT_MAX: control instance exercised the shadow double-talk update");
    CHECK(same, "shadow count past INT_MAX: output identical to an in-range count above the gate");
    CHECK(memcmp(&lo.dt_analyzer.shadow_advantage, &hi.dt_analyzer.shadow_advantage, sizeof(float)) == 0 &&
          memcmp(&lo.dt_analyzer.dt_from_shadow, &hi.dt_analyzer.dt_from_shadow, sizeof(float)) == 0,
          "shadow count past INT_MAX: shadow double-talk state identical");
    aec_destroy(&lo);
    aec_destroy(&hi);
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
    test_far_fft_counter_saturation();
    test_hop_counter_saturation();
    test_shadow_count_past_int_max();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
