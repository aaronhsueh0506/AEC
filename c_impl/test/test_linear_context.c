/* test_linear_context.c -- regression test for the linear-stage seam (aec_get_linear, AecLinearView): the read-only view an
 * external neural post-filter uses to consume the formed linear error plus the same time-domain aligned far the PBFDKF
 * consumed.
 *
 * What only this test proves (the byte-equal WAV gate cannot):
 *   1. aligned_far_hop aliases a->far_hop and, once LOCKED, is byte-identical to the caller's raw far delayed by the
 *      reported delay_samples.
 *   2. generation bumps exactly at ring-offset changes (first acquisition, confirmed shift, aec_reset) and at no other
 *      time; CHANGED is reported on the hop whose processing bumped it.
 *   3. delay_state never claims alignment it does not have: UNLOCKED before acquisition in MATCHED, and for the ring-fill
 *      window in FIXED even though the applied delay is known from hop 0.
 *   4. ref_ring_filled saturates at ref_ring_size (its unbounded form is signed-overflow UB after ~37 h of 16 kHz audio;
 *      the INT_MAX-seeded UBSan variant is in test_counter_saturation.c).
 *   5. Per-mode seam semantics (AecDelayMode). MATCHED as in 1-2. FIXED reports delay_samples == the configured delay,
 *      confidence 1.0, frozen generation, LOCKED once the ring can serve the offset, and the exposed far is byte-equal to
 *      the caller's far delayed by exactly that many samples. EXTERNAL_ALIGNED exposes the caller's own far byte for byte
 *      every hop, with delay_samples 0, LOCKED from hop 0, frozen generation.
 *   6. Per-mode reference-ring sizing. FIXED carves exactly `fixed_delay_samples + hop` and serves the caller's far delayed by exactly that many samples across hundreds of wraps, including
 *      through aec_reset()'s refill. EXTERNAL_ALIGNED carries no ring, and its pool is FIXED(0)'s minus one hop-sized
 *      array.
 *
 * Mutation checks (each reverts one thing and must go red here): drop the Path-A, Path-B or aec_reset generation bump;
 * report far_hop_aligned=1 unconditionally; serve the ring read one hop early or late; remove the ref_ring_filled clamp;
 * seed FIXED's current_delay from anything but cfg.fixed_delay_samples; run the ring write/read under the estimator gate
 * instead of the ring gate; report EXTERNAL_ALIGNED as UNLOCKED/-1; drop the `+ hop` from aec_ref_ring_samples()'s FIXED
 * arm; give EXTERNAL_ALIGNED a ring; bring a capture candidate back into linear_filter_select() (the far-active/no-echo
 * and analysis-seam scenarios go red).
 *
 * The end-to-end far-active/no-echo scenario proves that the rendering base (not the analysis seam) is bounded at the
 * capture while the estimate is unusable.
 *
 * Build (standalone, from c_impl/):
 *   make -C ../../audiolib/common lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       test/test_linear_context.c $(find src -name '*.c') \
 *       $(make -s -C ../../audiolib/common print-lib-path) -lm \
 *       -o bin/test_linear_context
 * Run:  ./bin/test_linear_context
 * Also wired into the Makefile: `make test-linear-context`.
 */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"
#include "aec_pool_test_util.h"
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

/* The split-entry configuration: residual FULL, every hop driven by aec_process_linear() + aec_process_residual(). */
static void context_only_config(AecConfig* cfg, int sample_rate, int fft_size) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, sample_rate);
    cfg->residual = AEC_RESIDUAL_FULL;
    if (fft_size > 0) cfg->fft_size = fft_size;
}

/* One hop of a split-entry instance (the library applies no gain and synthesizes nothing). */
static void split_hop(Aec* a, const float* mic, const float* far) {
    int st = aec_test_split_hop(a, mic, far);
    if (st != AEC_OK) { printf("FAIL: split hop refused (%d)\n", st); g_fail++; }
}

static void get_linear(const Aec* a, AecLinearView* v) {
    v->struct_size = sizeof *v;
    if (aec_get_linear(a, v) != AEC_OK) { printf("FAIL: aec_get_linear refused\n"); g_fail++; }
}

/* Deterministic xorshift noise, same style as the sibling tests. */
static unsigned g_rng_state = 0x11EC5EEDu;
static float rng_uniform(float amp) { return aec_test_rng_uniform(&g_rng_state, amp); }

/* One long shared far history so the test can reconstruct, from the CALLER
 * side, exactly which raw samples an aligned far_hop must contain.  Sized
 * for the largest scenario (48 kHz: 1500 hops x 512). */
/* Sized for the largest scenario: 500 acquisition + 5000 shift hops at 48 kHz/hop 512. The generous
 * shift budget is deliberate: with the ERLE watchdog live in this split-entry config, a confirmed delay
 * shift still needs seconds of audio to re-lock (ERLE collapse resumes full-rate matched-filter
 * analysis, then consensus must rebuild). The measured re-lock hop count is printed as info. */
#define MAX_SAMPLES (5600 * 512 + 8192)
static float g_far_hist[MAX_SAMPLES];

/* ONE definition of this file's stimulus: drive a single hop of far noise with mic = 0.5 * far[t -
 * true_delay], appending the far to g_far_hist so every scenario can byte-compare the aligned far
 * it gets back against the history. Returns the new `pushed` cursor, or `pushed` unchanged after
 * reporting a history overflow. It is per-hop rather than per-run because most scenarios inspect
 * their own locals between hops (fill windows, lock transitions, wrap classification). */
static long drive_hop(Aec* a, long pushed, int true_delay) {
    int hop = a->hop_size;
    float mic[1024];
    int i;
    if (pushed + hop > MAX_SAMPLES) {
        printf("FAIL: history overflow\n"); g_fail++; return pushed;
    }
    for (i = 0; i < hop; ++i) {
        long src;
        g_far_hist[pushed + i] = rng_uniform(0.05f);
        src = pushed + i - true_delay;
        mic[i] = (src >= 0) ? 0.5f * g_far_hist[src] : 0.0f;
    }
    split_hop(a, mic, g_far_hist + pushed);
    return pushed + hop;
}

/* Drive `hops` hops of the same stimulus, with an optional per-hop callback.
 * `*pushed` tracks the total samples generated so far so scenarios can chain
 * (e.g. change true_delay mid-stream) on one instance. */
static void run_hops(Aec* a, int hops, int true_delay, long* pushed,
                     void (*per_hop)(Aec*, long, void*), void* user) {
    for (int h = 0; h < hops; ++h) {
        long next = drive_hop(a, *pushed, true_delay);
        if (next == *pushed) return;   /* overflow, already reported */
        *pushed = next;
        if (per_hop) per_hop(a, *pushed, user);
    }
}

typedef struct LockWatch {
    int locked_seen;
    int changed_on_lock_hop;
    int changed_hops;
    int hops_seen;
    int first_changed_hop;
    int pre_lock_violation;
    int content_violation;
    unsigned gen_before;
    unsigned gen_at_lock;
    AecLinearView at_lock;
} LockWatch;

static void watch_hop(Aec* a, long pushed, void* user) {
    LockWatch* w = (LockWatch*)user;
    AecLinearView ctx;
    int hop = a->hop_size;
    long start;
    get_linear(a, &ctx);
    w->hops_seen++;
    /* Content proof on EVERY hop, not only at lock: UNLOCKED promises the caller's RAW far, and
     * from the first hop the ring can serve the offset the seam promises far[t - delay_samples].
     * Byte-compared against the history just pushed (amplitude 0.05 never soft-clips), not merely
     * non-NULL: aec_get_linear() assigns aligned_far_hop = a->far_hop unconditionally, so a
     * NULL test can never fail. */
    start = pushed - hop -
            (ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED ? 0 : ctx.delay_samples);
    if (start < 0 || memcmp(ctx.aligned_far_hop, g_far_hist + start,
                            (size_t)hop * sizeof(float)) != 0)
        w->content_violation = 1;
    if (ctx.delay_state == AEC_LINEAR_DELAY_CHANGED) {
        if (w->changed_hops == 0) w->first_changed_hop = w->hops_seen;
        w->changed_hops++;
    }
    if (!w->locked_seen) {
        if (ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED) {
            if (ctx.generation != w->gen_before) {
                /* generation may move only together with leaving UNLOCKED */
                w->pre_lock_violation = 1;
            }
        } else {
            w->locked_seen = 1;
            w->changed_on_lock_hop = (ctx.delay_state == AEC_LINEAR_DELAY_CHANGED);
            w->gen_at_lock = ctx.generation;
            w->at_lock = ctx;
        }
    }
}

static void scenario_acquire_and_verify(int sample_rate, int fft_size,
                                        int true_delay, const char* tag) {
    char msg[160];
    AecConfig cfg;
    context_only_config(&cfg, sample_rate, fft_size);
    Aec *a = NULL;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK((a = aec_create(&cfg)) != NULL, msg);
    int hop = a->hop_size;

    long pushed = 0;
    AecLinearView ctx0;
    get_linear(a, &ctx0);
    LockWatch w; memset(&w, 0, sizeof w);
    w.gen_before = ctx0.generation;

    run_hops(a, 500, true_delay, &pushed, watch_hop, &w);

    snprintf(msg, sizeof msg, "%s: estimator locks within 500 hops", tag);
    CHECK(w.locked_seen, msg);
    snprintf(msg, sizeof msg, "%s: pre-lock hops stay UNLOCKED with stable generation", tag);
    CHECK(!w.pre_lock_violation, msg);
    snprintf(msg, sizeof msg, "%s: exposed far is raw while UNLOCKED and "
                              "far[t-delay] once locked, every hop", tag);
    CHECK(!w.content_violation, msg);
    snprintf(msg, sizeof msg, "%s: first lock: CHANGED on the lock hop", tag);
    CHECK(w.changed_on_lock_hop, msg);
    snprintf(msg, sizeof msg, "%s: first lock: generation == start+1", tag);
    CHECK(w.gen_at_lock == w.gen_before + 1, msg);

    AecLinearView ctx;
    get_linear(a, &ctx);
    snprintf(msg, sizeof msg, "%s: steady state LOCKED after lock", tag);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_LOCKED, msg);
    snprintf(msg, sizeof msg, "%s: confidence >= 0.5 once locked", tag);
    CHECK(ctx.delay_confidence >= 0.5f, msg);
    snprintf(msg, sizeof msg, "%s: applied delay in (0, true_delay]", tag);
    CHECK(ctx.delay_samples > 0 && ctx.delay_samples <= true_delay, msg);
    /* Residual = quantization (one 64-sample bin at 16k-equivalent, x3 at
     * 48 kHz) + headroom (32..92 at 16k-equivalent). Generous cap: 512
     * samples at 16 kHz, 1536 at 48 kHz. */
    int residual_cap = (sample_rate == 48000) ? 1536 : 512;
    snprintf(msg, sizeof msg, "%s: under-compensation within headroom+quantization", tag);
    CHECK(true_delay - ctx.delay_samples <= residual_cap, msg);

    snprintf(msg, sizeof msg, "%s: aligned_far_hop aliases a->far_hop", tag);
    CHECK(ctx.aligned_far_hop == a->far_hop, msg);
    snprintf(msg, sizeof msg, "%s: residual view formed_hop aliases the formed output", tag);
    {
        AecResidualView rv;
        memset(&rv, 0, sizeof rv);
        rv.struct_size = sizeof rv;
        CHECK(aec_get_residual(a, &rv) == AEC_OK && rv.formed_hop == a->a3_lfs.e_form, msg);
    }

    /* Content proof: the exposed hop == raw far delayed by delay_samples,
     * byte for byte (no soft-clip at this amplitude, so the ring holds the
     * caller's exact samples). */
    long start = pushed - hop - ctx.delay_samples;
    snprintf(msg, sizeof msg, "%s: aligned far == caller far delayed by delay_samples (byte-equal)", tag);
    CHECK(start >= 0 && memcmp(ctx.aligned_far_hop, g_far_hist + start,
                               (size_t)hop * sizeof(float)) == 0, msg);

    /* Shift: move the true delay and expect exactly one more ring-offset
     * change (generation strictly increases; CHANGED observed again). */
    unsigned gen_locked = ctx.generation;
    int changed_before = w.changed_hops;
    LockWatch w2; memset(&w2, 0, sizeof w2);
    w2.gen_before = gen_locked;
    /* reuse watch via fresh struct: it counts CHANGED hops from here on */
    run_hops(a, 5000, true_delay + 10 * hop, &pushed, watch_hop, &w2);
    get_linear(a, &ctx);
    printf("info: %s: shift re-lock after %d hops (%.1f s audio)\n", tag,
           w2.first_changed_hop,
           (double)w2.first_changed_hop * hop / sample_rate);
    snprintf(msg, sizeof msg, "%s: shift: generation advanced", tag);
    CHECK(ctx.generation > gen_locked, msg);
    snprintf(msg, sizeof msg, "%s: shift: CHANGED observed on some hop", tag);
    CHECK(w2.changed_hops >= 1, msg);
    snprintf(msg, sizeof msg, "%s: shift: exposed far tracks the applied "
                              "offset on every hop", tag);
    CHECK(!w2.content_violation, msg);
    (void)changed_before;
    snprintf(msg, sizeof msg, "%s: shift: tracks the new delay", tag);
    CHECK(ctx.delay_samples > true_delay
          && ctx.delay_samples <= true_delay + 10 * hop, msg);
    long start2 = pushed - hop - ctx.delay_samples;
    snprintf(msg, sizeof msg, "%s: shift: aligned far still byte-equal at the new offset", tag);
    CHECK(start2 >= 0 && memcmp(ctx.aligned_far_hop, g_far_hist + start2,
                                (size_t)hop * sizeof(float)) == 0, msg);

    /* Reset: one more generation bump, back to UNLOCKED. */
    unsigned gen_shift = ctx.generation;
    aec_reset(a);
    get_linear(a, &ctx);
    snprintf(msg, sizeof msg, "%s: reset: generation advanced", tag);
    CHECK(ctx.generation > gen_shift, msg);
    snprintf(msg, sizeof msg, "%s: reset: UNLOCKED with delay_samples == -1", tag);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED && ctx.delay_samples == -1, msg);

    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;  /* keep scenarios independent of ordering */
}

/* EXTERNAL_ALIGNED: no estimator, no ring. The caller contracted that far is already aligned, so the
 * seam reports delay_samples 0, LOCKED from the first hop, frozen generation, and exposes the caller's
 * own hop byte for byte, never a ring read. */
static void scenario_external_aligned(void) {
    const char* tag = "external-aligned";
    char msg[160];
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    Aec *a = NULL;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK((a = aec_create(&cfg)) != NULL, msg);
    snprintf(msg, sizeof msg, "%s: resolves to EXTERNAL_ALIGNED", tag);
    CHECK(a->cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED, msg);
    snprintf(msg, sizeof msg, "%s: no ring carved", tag);
    CHECK(a->ref_ring == NULL && !a->has_delay, msg);

    int hop = a->hop_size;
    long pushed = 0;
    AecLinearView first, ctx;
    get_linear(a, &first);

    /* Walk EVERY hop: the far the seam exposes must equal the far this test
     * just handed in (modulo nothing -- amplitude 0.05 never soft-clips). */
    int content_ok = 1, state_ok = 1, gen_ok = 1;
    for (int h = 0; h < 120; ++h) {
        long base = pushed;
        pushed = drive_hop(a, pushed, 800);
        get_linear(a, &ctx);
        if (memcmp(ctx.aligned_far_hop, g_far_hist + base,
                   (size_t)hop * sizeof(float)) != 0) content_ok = 0;
        if (ctx.delay_state != AEC_LINEAR_DELAY_LOCKED ||
            ctx.delay_samples != 0 || ctx.delay_confidence != 1.0f) state_ok = 0;
        if (ctx.generation != first.generation) gen_ok = 0;
    }
    snprintf(msg, sizeof msg, "%s: exposed far IS the caller's far, every hop", tag);
    CHECK(content_ok, msg);
    snprintf(msg, sizeof msg, "%s: LOCKED / delay 0 / confidence 1.0, every hop", tag);
    CHECK(state_ok, msg);
    snprintf(msg, sizeof msg, "%s: generation frozen while processing", tag);
    CHECK(gen_ok, msg);
    snprintf(msg, sizeof msg, "%s: aligned_far_hop aliases a->far_hop", tag);
    CHECK(ctx.aligned_far_hop == a->far_hop, msg);
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

/* FIXED: no estimator, but a ring that applies the caller's MEASURED delay
 * from the first hop it can serve it. The seam must report that delay
 * verbatim (never -1, never the estimator's 0.0 confidence), must not claim
 * LOCKED while the ring is still filling, and must never bump generation --
 * there is nothing that could re-lock. */
static void scenario_fixed_delay(void) {
    const char* tag = "fixed";
    char msg[160];
    const int fixed = 1600;   /* 100 ms at 16 kHz */
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    cfg.fixed_delay_samples = fixed;
    cfg.delay_mode = AEC_DELAY_FIXED;
    Aec *a = NULL;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK((a = aec_create(&cfg)) != NULL, msg);
    snprintf(msg, sizeof msg, "%s: resolves to FIXED", tag);
    CHECK(a->cfg.delay_mode == AEC_DELAY_FIXED, msg);
    snprintf(msg, sizeof msg, "%s: ring carved, no estimator", tag);
    CHECK(a->ref_ring != NULL && !a->has_delay, msg);

    int hop = a->hop_size;
    AecLinearView first, ctx;
    get_linear(a, &first);
    snprintf(msg, sizeof msg, "%s: delay_samples reads the configured delay "
                              "before the first hop", tag);
    CHECK(first.delay_samples == fixed, msg);

    /* The ring cannot serve `fixed` until the samples already written cover
     * it, so hops 0..ceil(fixed/hop)-1 carry the caller's RAW far and hop
     * ceil(fixed/hop) is the FIRST aligned one. Spelled as that raw-hop count
     * directly: a ceil+1 "fill" count compared with a -1 leaves the first
     * aligned hop unasserted, which is exactly where an off-by-one lands. */
    int raw_hops = (fixed + hop - 1) / hop;
    long pushed = 0;
    int early_unlocked = 1, late_locked = 1, delay_ok = 1, conf_ok = 1, gen_ok = 1;
    int content_ok = 1, changed_seen = 0;
    for (int h = 0; h < 400; ++h) {
        pushed = drive_hop(a, pushed, fixed);
        get_linear(a, &ctx);
        if (ctx.delay_samples != fixed) delay_ok = 0;
        if (ctx.delay_confidence != 1.0f) conf_ok = 0;
        if (ctx.generation != first.generation) gen_ok = 0;
        if (ctx.delay_state == AEC_LINEAR_DELAY_CHANGED) changed_seen = 1;
        if (h < raw_hops) {
            if (ctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED) early_unlocked = 0;
            /* UNLOCKED means the content IS the caller's raw far, not
             * silence and not a partially served hop. */
            if (memcmp(ctx.aligned_far_hop, g_far_hist + pushed - hop,
                       (size_t)hop * sizeof(float)) != 0)
                content_ok = 0;
        } else {
            if (ctx.delay_state != AEC_LINEAR_DELAY_LOCKED) late_locked = 0;
            long start = pushed - hop - fixed;
            if (start < 0 || memcmp(ctx.aligned_far_hop, g_far_hist + start,
                                    (size_t)hop * sizeof(float)) != 0)
                content_ok = 0;
        }
    }
    snprintf(msg, sizeof msg, "%s: delay_samples == the configured delay, every hop", tag);
    CHECK(delay_ok, msg);
    snprintf(msg, sizeof msg, "%s: confidence 1.0 (measured, not estimated)", tag);
    CHECK(conf_ok, msg);
    snprintf(msg, sizeof msg, "%s: UNLOCKED until the ring can serve the offset", tag);
    CHECK(early_unlocked, msg);
    snprintf(msg, sizeof msg, "%s: LOCKED once the ring covers delay + hop", tag);
    CHECK(late_locked, msg);
    snprintf(msg, sizeof msg, "%s: aligned far is the caller far -- raw while "
                              "the ring fills, delayed by fixed from the first "
                              "servable hop", tag);
    CHECK(content_ok, msg);
    snprintf(msg, sizeof msg, "%s: generation frozen while processing", tag);
    CHECK(gen_ok, msg);
    snprintf(msg, sizeof msg, "%s: CHANGED is unreachable without an estimator", tag);
    CHECK(!changed_seen, msg);

    /* aec_reset re-zeroes the ring, so it DOES invalidate a consumer's cached
     * far -- the one legitimate generation bump in this mode -- and must
     * re-seed the applied delay from the config, not to -1. */
    unsigned gen_before = ctx.generation;
    aec_reset(a);
    get_linear(a, &ctx);
    snprintf(msg, sizeof msg, "%s: reset re-seeds the configured delay (not -1)", tag);
    CHECK(ctx.delay_samples == fixed && a->current_delay == fixed, msg);
    snprintf(msg, sizeof msg, "%s: reset returns to UNLOCKED (ring emptied)", tag);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED, msg);
    snprintf(msg, sizeof msg, "%s: reset bumps generation (ring content invalidated)", tag);
    CHECK(ctx.generation > gen_before, msg);
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

/* A FIXED delay of 0 is legal and distinct from EXTERNAL_ALIGNED: the ring
 * exists (so a later reset/realign has somewhere to read from) and the seam
 * is LOCKED from hop 0 because a zero offset needs no ring read at all. */
static void scenario_fixed_zero_delay(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = 0;
    Aec *a = NULL;
    CHECK((a = aec_create(&cfg)) != NULL, "fixed0: aec_create");
    CHECK(a->ref_ring != NULL, "fixed0: ring still carved (unlike EXTERNAL)");
    long pushed = 0;
    AecLinearView first, ctx;
    get_linear(a, &first);
    run_hops(a, 40, 0, &pushed, NULL, NULL);
    get_linear(a, &ctx);
    CHECK(ctx.delay_samples == 0 && ctx.delay_state == AEC_LINEAR_DELAY_LOCKED,
          "fixed0: LOCKED at delay 0 from the start");
    CHECK(ctx.generation == first.generation,
          "fixed0: generation frozen while processing");
    CHECK(memcmp(ctx.aligned_far_hop, g_far_hist + pushed - a->hop_size,
                 (size_t)a->hop_size * sizeof(float)) == 0,
          "fixed0: exposed far is the caller's latest hop");
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

/* ── the FIXED ring is sized for the delay, not for a search ──────────── */

/* FIXED carves exactly `fixed_delay_samples + hop` samples (the tight bound derived in
 * aec_ref_ring_samples()), not the MATCHED ring (the estimator's reach + hop), on every grid and
 * across the interesting fixed values (0, sub-hop, exact hop multiple, non-multiple, past the
 * MATCHED reach). */
static void scenario_fixed_ring_exact_size(void) {
    static const struct { int sr, fft, fixed; const char* tag; } rows[] = {
        { 16000,  256,     0, "16k/256 fixed=0 (degenerate, write-only ring)" },
        { 16000,  256,   100, "16k/256 fixed=100 (below one hop)" },
        { 16000,  256,   128, "16k/256 fixed=128 (exactly one hop)" },
        { 16000,  256,  1536, "16k/256 fixed=1536 (exact hop multiple)" },
        { 16000,  256,  1600, "16k/256 fixed=1600 (NOT a hop multiple)" },
        { 16000,  256, 40000, "16k/256 fixed=40000 (past the MATCHED reach)" },
        { 16000,  512,  1600, "16k/512 fixed=1600" },
        { 48000, 1024,  4801, "48k/1024 fixed=4801 (NOT a hop multiple)" },
    };
    char msg[224];
    unsigned r;
    printf("--- FIXED reference-ring capacity ---\n");
    for (r = 0; r < sizeof rows / sizeof rows[0]; ++r) {
        AecConfig cfg;
        Aec *a = NULL;
        size_t mem_a;
        int hop, want;

        context_only_config(&cfg, rows[r].sr, rows[r].fft);
        cfg.delay_mode = AEC_DELAY_FIXED;
        cfg.fixed_delay_samples = rows[r].fixed;
        mem_a = aec_get_mem_size(&cfg);
        snprintf(msg, sizeof msg, "%s: aec_create", rows[r].tag);
        CHECK(mem_a > 0 && (a = aec_create(&cfg)) != NULL, msg);
        hop = a->hop_size;
        want = rows[r].fixed + hop;
        printf("    %-44s ring=%d samples  pool=%zu B\n",
               rows[r].tag, a->ref_ring_size, mem_a);
        snprintf(msg, sizeof msg, "%s: ring == fixed + hop (%d), got %d",
                 rows[r].tag, want, a->ref_ring_size);
        CHECK(a->ref_ring_size == want, msg);
        snprintf(msg, sizeof msg, "%s: ring carved, no estimator", rows[r].tag);
        CHECK(a->ref_ring != NULL && !a->has_delay, msg);

        aec_destroy(a);
        g_rng_state = 0x11EC5EEDu;
    }
}

/* An exact-fit ring wraps constantly, so the delay-compensating read is a two-part (split) memcpy
 * on most hops. This walks a small and a large FIXED delay long enough to wrap many times, demands
 * byte-equality with the caller's far on every post-fill hop, and asserts that the split path was
 * actually taken, so the test cannot pass by never reaching the boundary. */
static void scenario_fixed_ring_wraps(int sr, int fft, int fixed, int hops,
                                      const char* tag) {
    char msg[224];
    AecConfig cfg;
    Aec *a = NULL;
    int hop, raw_hops, wrapped = 0, laps;
    long pushed = 0;
    int content_ok = 1, locked_ok = 1, h;

    context_only_config(&cfg, sr, fft);
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = fixed;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK((a = aec_create(&cfg)) != NULL, msg);
    hop = a->hop_size;
    /* Raw-far hops before the ring can serve the offset; hop raw_hops is the
     * first aligned one and is asserted like every later hop. */
    raw_hops = (fixed + hop - 1) / hop;

    for (h = 0; h < hops; ++h) {
        AecLinearView ctx;
        int read_pos;
        pushed = drive_hop(a, pushed, fixed);
        /* Recompute the read the engine just made, from the post-write
         * cursor, to classify contiguous vs split. */
        read_pos = (a->ref_ring_write - hop - a->current_delay) % a->ref_ring_size;
        if (read_pos < 0) read_pos += a->ref_ring_size;
        if (read_pos + hop > a->ref_ring_size) wrapped++;
        get_linear(a, &ctx);
        if (h < raw_hops) continue;
        if (ctx.delay_state != AEC_LINEAR_DELAY_LOCKED) locked_ok = 0;
        {
            long start = pushed - hop - fixed;
            if (start < 0 || memcmp(ctx.aligned_far_hop, g_far_hist + start,
                                    (size_t)hop * sizeof(float)) != 0)
                content_ok = 0;
        }
    }
    laps = (int)((long)hops * hop / a->ref_ring_size);
    printf("    %-30s ring=%d, %d laps, %d/%d hops split-read\n",
           tag, a->ref_ring_size, laps, wrapped, hops);
    snprintf(msg, sizeof msg, "%s: LOCKED for every post-fill hop", tag);
    CHECK(locked_ok, msg);
    snprintf(msg, sizeof msg,
             "%s: aligned far byte-equal to far[t-%d] on every post-fill hop",
             tag, fixed);
    CHECK(content_ok, msg);
    snprintf(msg, sizeof msg, "%s: the wrapped (split) read path was exercised "
                              "(%d hops)", tag, wrapped);
    CHECK(wrapped > 0, msg);
    snprintf(msg, sizeof msg, "%s: the ring wrapped many times (%d laps)",
             tag, laps);
    CHECK(laps >= 4, msg);
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

/* aec_reset() empties the exact-fit ring, so FIXED must serve the RAW far
 * again for exactly the same fill window it did at init, then re-lock and
 * read correctly -- there is no slack that could hide an
 * off-by-one refill. */
static void scenario_fixed_reset_refill(void) {
    const int fixed = 1600;
    AecConfig cfg;
    Aec *a = NULL;
    char msg[192];
    int hop, raw_hops, pass, h;
    long pushed = 0;

    context_only_config(&cfg, 16000, 0);
    cfg.delay_mode = AEC_DELAY_FIXED;
    cfg.fixed_delay_samples = fixed;
    CHECK((a = aec_create(&cfg)) != NULL, "fixed-refill: aec_create");
    hop = a->hop_size;
    raw_hops = (fixed + hop - 1) / hop;

    for (pass = 0; pass < 2; ++pass) {
        int unlocked_hops = 0, content_ok = 1, locked_ok = 1;
        for (h = 0; h < 3 * (raw_hops + 1); ++h) {
            AecLinearView ctx;
            pushed = drive_hop(a, pushed, fixed);
            get_linear(a, &ctx);
            if (ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED) {
                unlocked_hops++;
                continue;
            }
            if (ctx.delay_state != AEC_LINEAR_DELAY_LOCKED) locked_ok = 0;
            {
                long start = pushed - hop - fixed;
                if (start < 0 || memcmp(ctx.aligned_far_hop, g_far_hist + start,
                                        (size_t)hop * sizeof(float)) != 0)
                    content_ok = 0;
            }
        }
        snprintf(msg, sizeof msg,
                 "fixed-refill pass %d: exactly %d RAW-far hops (got %d)",
                 pass, raw_hops, unlocked_hops);
        CHECK(unlocked_hops == raw_hops, msg);
        snprintf(msg, sizeof msg,
                 "fixed-refill pass %d: LOCKED once refilled", pass);
        CHECK(locked_ok, msg);
        snprintf(msg, sizeof msg,
                 "fixed-refill pass %d: aligned far correct after refill", pass);
        CHECK(content_ok, msg);
        if (pass == 0) {
            aec_reset(a);
            CHECK(a->ref_ring_filled == 0 && a->current_delay == fixed,
                  "fixed-refill: reset empties the ring, keeps the delay");
        }
    }
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

/* EXTERNAL_ALIGNED carries no delay-attached allocation: no estimator, no ring, no ring-shaped
 * remnant. scenario_external_aligned() proves the far passthrough; this pins the absence of state,
 * so an edit that quietly hands this mode a ring is caught even though the audio would not change. */
static void scenario_external_has_no_delay_state(void) {
    static const struct { int sr, fft; const char* tag; } grids[] = {
        { 16000,  256, "16k/256"  },
        { 16000,  512, "16k/512"  },
        { 48000, 1024, "48k/1024" },
    };
    char msg[192];
    unsigned g;
    for (g = 0; g < sizeof grids / sizeof grids[0]; ++g) {
        AecConfig ce, cf;
        Aec *a = NULL;
        size_t mem_e, mem_f0, ring_bytes;
        context_only_config(&ce, grids[g].sr, grids[g].fft);
        ce.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        mem_e = aec_get_mem_size(&ce);
        snprintf(msg, sizeof msg, "%s external: aec_create", grids[g].tag);
        CHECK(mem_e > 0 && (a = aec_create(&ce)) != NULL, msg);
        snprintf(msg, sizeof msg,
                 "%s external: no ring, no estimator, delay 0", grids[g].tag);
        CHECK(a->ref_ring == NULL && a->ref_ring_size == 0 && !a->has_delay
              && a->current_delay == 0, msg);

        /* The ONLY thing FIXED-with-zero-delay buys over EXTERNAL is the
         * hop-sized ring, so the pool difference must be exactly that array
         * (16-byte aligned, as every pool field is). Anything else in the
         * pool that still scaled with "delay-ness" would break this. */
        cf = ce;
        cf.delay_mode = AEC_DELAY_FIXED;
        cf.fixed_delay_samples = 0;
        mem_f0 = aec_get_mem_size(&cf);
        ring_bytes = (((size_t)a->hop_size * sizeof(float)) + 15u) & ~(size_t)15u;
        snprintf(msg, sizeof msg,
                 "%s external: pool is FIXED(0) minus exactly one hop-ring "
                 "(%zu - %zu == %zu)", grids[g].tag, mem_f0, mem_e, ring_bytes);
        CHECK(mem_f0 > mem_e && mem_f0 - mem_e == ring_bytes, msg);
        aec_destroy(a);
        g_rng_state = 0x11EC5EEDu;
    }
}

/* The honest-UNLOCKED window: current_delay >= 0 but the ring does not yet hold delay+hop samples,
 * so far_hop stays RAW. It is unreachable through natural acquisition (locking takes ~0.3 s, by
 * which time the ring holds more than any in-range delay), so the state is poked directly. It is
 * the only scenario that observes far_hop_aligned=0 with a non-negative delay, so the mutation
 * "report aligned unconditionally" is caught here only. */
static void scenario_unfilled_ring_reads_unlocked(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    Aec *a = NULL;
    CHECK((a = aec_create(&cfg)) != NULL, "unfilled-ring: aec_create");
    int hop = a->hop_size;
    float mic[1024];
    memset(mic, 0, sizeof mic);

    /* Simulate an established lock whose ring content has just been wiped
     * (as after a reset that kept the delay -- an artificial but layout-legal
     * state). Silence on mic keeps the estimator from re-acquiring. */
    long pushed = 0;
    run_hops(a, 4, 0, &pushed, NULL, NULL);
    a->current_delay = 5000;
    a->ref_ring_filled = 64;

    long base = pushed;
    for (int i = 0; i < hop; ++i) g_far_hist[base + i] = rng_uniform(0.05f);
    split_hop(a, mic, g_far_hist + base);
    pushed = base + hop;

    AecLinearView ctx;
    get_linear(a, &ctx);
    CHECK(ctx.delay_samples == 5000, "unfilled-ring: delay reads as applied");
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED,
          "unfilled-ring: UNLOCKED while the ring cannot cover the delay");
    CHECK(memcmp(ctx.aligned_far_hop, g_far_hist + base,
                 (size_t)hop * sizeof(float)) == 0,
          "unfilled-ring: exposed far is the RAW hop, not a stale ring read");

    /* Once enough hops fill the ring past delay+hop, the same lock becomes
     * genuinely compensated and the state reads LOCKED again. */
    for (int h = 0; h < 5000 / hop + 4; ++h) {
        base = pushed;
        for (int i = 0; i < hop; ++i) g_far_hist[base + i] = rng_uniform(0.05f);
        split_hop(a, mic, g_far_hist + base);
        pushed = base + hop;
    }
    get_linear(a, &ctx);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_LOCKED,
          "unfilled-ring: LOCKED once the ring covers the delay");
    long start = pushed - hop - ctx.delay_samples;
    CHECK(start >= 0 && memcmp(ctx.aligned_far_hop, g_far_hist + start,
                               (size_t)hop * sizeof(float)) == 0,
          "unfilled-ring: compensated far byte-equal at the poked offset");
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}

static void scenario_ring_filled_saturates(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    Aec *a = NULL;
    CHECK((a = aec_create(&cfg)) != NULL, "ring-filled: aec_create");
    int rs = a->ref_ring_size;
    int hop = a->hop_size;
    long pushed = 0;
    int hops_to_fill = rs / hop + 8;
    run_hops(a, hops_to_fill, 1600, &pushed, NULL, NULL);
    CHECK(a->ref_ring_filled == rs, "filled saturates at ring size");
    run_hops(a, 100, 1600, &pushed, NULL, NULL);
    CHECK(a->ref_ring_filled == rs, "filled stays frozen at ring size");
    AecLinearView ctx;
    get_linear(a, &ctx);
    CHECK(ctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED || ctx.delay_samples < 0,
          "saturated fill never blocks a valid locked state");
    aec_destroy(a);
    g_rng_state = 0x11EC5EEDu;
}


/* far ACTIVE with NO echo path (near-end single talk). A reference this loud with nothing of it in
 * the capture gives the matched filter no peak to lock, so the delay never becomes solid and the
 * filtering-quality analyzer never calls the linear estimate usable; meanwhile the adaptive filter
 * fits far-shaped noise onto the near speech and subtracts something that was never there. The
 * analysis seam (formed_hop / error_spec) is the real residual and is NOT bounded; the rendering base
 * (aec_get_residual().base_spec) must not carry more windowed energy than the capture spectrum on any hop whose
 * usable verdict is false, and the guard must actually fire in this scenario. */
static void scenario_far_active_no_echo(int silent_mic, const char *label) {
    AecConfig cfg;
    Aec *a = NULL;
    int hop, i, h;
    const int HOPS = 900;
    float *mic, *far;
    double e_out = 0.0, e_mic = 0.0, worst = 0.0;
    long guard_hops = 0, unusable_hops = 0;

    g_rng_state = 0x11EC5EEDu;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.residual = AEC_RESIDUAL_FULL;   /* the seam the NN post-filter consumes (split entry points) */
    if ((a = aec_create(&cfg)) == NULL) { CHECK(0, "create"); return; }
    hop = aec_hop_size(a);
    mic = (float *)malloc((size_t)hop * sizeof(float));
    far = (float *)malloc((size_t)hop * sizeof(float));

    for (h = 0; h < HOPS; ++h) {
        AecResidualView ob;
        double eo = 0.0, em = 0.0;
        for (i = 0; i < hop; ++i) {
            /* Loud reference, quiet near, and NO path between them. */
            far[i] = rng_uniform(0.6f);
            mic[i] = silent_mic ? 0.0f : rng_uniform(0.05f);
        }
        split_hop(a, mic, far);
        memset(&ob, 0, sizeof ob);
        ob.struct_size = sizeof ob;
        if (aec_get_residual(a, &ob) != AEC_OK) { CHECK(0, "residual view"); break; }
        if (ob.guard_active) guard_hops++;
        if (ob.usable_linear) continue;
        unusable_hops++;
        for (i = 0; i < ob.n_freqs; ++i) {
            eo += (double)ob.base_spec[i].r * ob.base_spec[i].r
                + (double)ob.base_spec[i].i * ob.base_spec[i].i;
            em += (double)ob.near_spec[i].r * ob.near_spec[i].r
                + (double)ob.near_spec[i].i * ob.near_spec[i].i;
        }
        e_out += eo; e_mic += em;
        if (h > 8 && em > 1e-12 && eo / em > worst) worst = eo / em;
    }

    CHECK(unusable_hops > 100, "far-active/no-echo: the estimate stays unusable");
    if (!silent_mic)
        CHECK(guard_hops > 0, "far-active/no-echo: the output guard fires");
    if (silent_mic) {
        /* Nothing to preserve: the rendering base must stay at digital silence rather
         * than carrying the filter's synthesized reference. */
        CHECK(e_out < 1e-9,
              "far-active/no-echo, silent mic: rendering base stays silent");
    } else {
        CHECK(e_out <= e_mic * 1.0000001,
              "far-active/no-echo: rendering base carries no more energy than capture");
        CHECK(worst <= 1.0000001,
              "far-active/no-echo: no unusable hop's base exceeds the capture spectrum");
    }
    printf("  [%s] base/capture = %.3f, worst hop = %.2f dB, guard hops %ld of %ld unusable\n", label,
           e_mic > 0.0 ? e_out / e_mic : e_out,
           worst > 0.0 ? 10.0 * log10(worst) : 0.0, guard_hops, unusable_hops);

    free(mic); free(far);
    aec_destroy(a);
}

/* An instance that has not run a hop reads zeros: linear_hop, aligned_far_hop and the three spectra do not depend on what the
 * pool held (dirty caller pools, single channel and array, the three delay modes, three grids). */
static int all_zero_f(const float* x, int n) {
    int i;
    for (i = 0; i < n; ++i) if (x[i] != 0.0f) return 0;
    return 1;
}

#ifdef AEC_NO_ARRAY
#define N_MICS_MAX 1
#else
#define N_MICS_MAX 4
#endif
static void scenario_fresh_view_reads_zeros(void) {
    static const struct { int sr, fft; const char* tag; } grids[] = {
        { 16000,  256, "16k/256"  },
        { 16000,  512, "16k/512"  },
        { 48000, 1024, "48k/1024" },
    };
    static const AecDelayMode modes[3] = { AEC_DELAY_MATCHED, AEC_DELAY_FIXED, AEC_DELAY_EXTERNAL_ALIGNED };
    static const int fills[2] = { 0xAB, 0xFF };
    char msg[192];
    unsigned g;
    int m, f, n_mics;
    for (g = 0; g < sizeof grids / sizeof grids[0]; ++g)
        for (m = 0; m < 3; ++m)
            for (n_mics = 1; n_mics <= N_MICS_MAX; n_mics += 3)
                for (f = 0; f < 2; ++f) {
                    AecConfig cfg;
                    AecLinearView v;
                    void* mem = NULL;
                    Aec* a;
                    int ok;
                    context_only_config(&cfg, grids[g].sr, grids[g].fft);
                    cfg.delay_mode = modes[m];
                    if (modes[m] == AEC_DELAY_FIXED) cfg.fixed_delay_samples = 64;
                    cfg.n_mics = n_mics;
                    snprintf(msg, sizeof msg, "%s mode %d n_mics %d fill 0x%02X: fresh view reads zeros", grids[g].tag, m, n_mics,
                             fills[f]);
                    a = aec_test_new_aec(&cfg, &mem, fills[f]);
                    if (!a) { CHECK(0, msg); free(mem); continue; }
                    memset(&v, 0, sizeof v);
                    get_linear(a, &v);
                    ok = v.linear_hop && v.aligned_far_hop && all_zero_f(v.linear_hop, v.hop_size) &&
                         all_zero_f(v.aligned_far_hop, v.hop_size) && all_zero_f((const float*)v.echo_spec, 2 * v.n_freqs) &&
                         all_zero_f((const float*)v.near_spec, 2 * v.n_freqs) && all_zero_f((const float*)v.far_spec, 2 * v.n_freqs);
                    CHECK(ok, msg);
                    free(mem);
                }
}

int main(void) {
    scenario_fresh_view_reads_zeros();
    scenario_acquire_and_verify(16000, 0,   1600, "16k/fft256");
    scenario_acquire_and_verify(16000, 512, 1600, "16k/fft512");
    scenario_acquire_and_verify(48000, 0,   4800, "48k/fft1024");
    scenario_external_aligned();
    scenario_fixed_delay();
    scenario_fixed_zero_delay();
    scenario_fixed_ring_exact_size();
    scenario_fixed_ring_wraps(16000,  256,   100, 400, "16k/256 fixed=100");
    scenario_fixed_ring_wraps(16000,  256,  1600, 400, "16k/256 fixed=1600");
    scenario_fixed_ring_wraps(16000,  512,  1601, 300, "16k/512 fixed=1601");
    scenario_fixed_ring_wraps(48000, 1024,  4801, 300, "48k/1024 fixed=4801");
    scenario_fixed_reset_refill();
    scenario_external_has_no_delay_state();
    scenario_unfilled_ring_reads_unlocked();
    scenario_ring_filled_saturates();
    scenario_far_active_no_echo(/*silent_mic=*/0, "near speech");
    scenario_far_active_no_echo(/*silent_mic=*/1, "silent mic");
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
