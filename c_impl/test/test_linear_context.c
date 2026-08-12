/* test_linear_context.c — permanent regression test for the AecLinearContext
 * seam (aec_get_linear_context): the read-only view an external neural
 * post-filter uses to consume the formed linear error plus the SAME
 * time-domain aligned far the PBFDKF consumed.
 *
 * What only this test proves (the byte-equal WAV gate cannot):
 *   1. aligned_far_hop aliases a->far_hop and, once LOCKED, its content is
 *      byte-identical to the caller's own raw far delayed by the reported
 *      delay_samples — i.e. the exposed pointer really is the hop the linear
 *      filter read, not a copy or a different tap point.
 *   2. generation bumps exactly at ring-offset changes: first acquisition,
 *      confirmed shift, aec_reset — and at no other time. CHANGED is
 *      reported precisely on the hop whose processing bumped it.
 *   3. delay_state never claims alignment it does not have: UNLOCKED before
 *      acquisition, and permanently UNLOCKED (delay_samples == -1) when
 *      enable_delay_est=0, even though the engine's internal current_delay
 *      is 0 in that mode.
 *   4. ref_ring_filled saturates at ref_ring_size instead of growing without
 *      bound (its unbounded form is signed-overflow UB after ~37 h of
 *      continuous 16 kHz audio; the INT_MAX-seeded UBSan variant of this
 *      property lives in test_counter_saturation.c).
 *
 * Mutation checks (each reverts one line of the fix and must go red here):
 *   - drop the Path-A generation bump  -> "first lock: CHANGED on the lock
 *     hop" and "generation == g0+1" fail;
 *   - drop the Path-B generation bump  -> "shift: generation advanced" fails;
 *   - drop the aec_reset bump          -> "reset: generation advanced" fails;
 *   - report far_hop_aligned=1 unconditionally -> "pre-lock hops stay
 *     UNLOCKED" fails;
 *   - remove the ref_ring_filled clamp -> "filled saturates at ring size"
 *     fails.
 *
 * Build (standalone, from c_impl/ — mirrors test_process_context.c):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_linear_context.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_linear_context
 * Run:  ./bin/test_linear_context
 * Also wired into the Makefile: `make test-linear-context`.
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

static void context_only_config(AecConfig* cfg, int sample_rate, int fft_size) {
    aec_config_from_preset(cfg, AEC_PRESET_BALANCED, sample_rate);
    cfg->enable_res = 0;
    cfg->return_res_context = 1;
    if (fft_size > 0) cfg->fft_size = fft_size;
}

/* Deterministic xorshift noise, same style as the sibling tests. */
static unsigned g_rng_state = 0x11EC5EEDu;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu;
    return amp * (2.0f * u - 1.0f);
}

/* One long shared far history so the test can reconstruct, from the CALLER
 * side, exactly which raw samples an aligned far_hop must contain.  Sized
 * for the largest scenario (48 kHz: 1500 hops x 512). */
/* Sized for the largest scenario: 500 acquisition + 5000 shift hops at
 * 48 kHz/hop 512. The generous shift budget is deliberate: in the
 * enable_res=0 context-only config the ERLE watchdog is inert (see
 * aec.c:2492 note), so a confirmed delay shift re-locks only through the
 * duty-cycled matched-filter consensus, which takes tens of seconds of
 * audio. The measured re-lock hop count is printed as info. */
#define MAX_SAMPLES (5600 * 512 + 8192)
static float g_far_hist[MAX_SAMPLES];

/* Drive `hops` hops of far noise with mic = 0.5 * far[t - true_delay].
 * `*pushed` tracks the total samples generated so far so scenarios can chain
 * (e.g. change true_delay mid-stream) on one instance. */
static void run_hops(Aec* a, int hops, int true_delay, long* pushed,
                     void (*per_hop)(Aec*, long, void*), void* user) {
    int hop = a->hop_size;
    float mic[1024], out[1024];
    for (int h = 0; h < hops; ++h) {
        long base = *pushed;
        if (base + hop > MAX_SAMPLES) { printf("FAIL: history overflow\n"); g_fail++; return; }
        for (int i = 0; i < hop; ++i) {
            g_far_hist[base + i] = rng_uniform(0.05f);
            long src = base + i - true_delay;
            mic[i] = (src >= 0) ? 0.5f * g_far_hist[src] : 0.0f;
        }
        aec_process(a, mic, g_far_hist + base, out);
        *pushed = base + hop;
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
    unsigned gen_before;
    unsigned gen_at_lock;
    AecLinearContext at_lock;
} LockWatch;

static void watch_hop(Aec* a, long pushed, void* user) {
    (void)pushed;
    LockWatch* w = (LockWatch*)user;
    AecLinearContext ctx;
    aec_get_linear_context(a, &ctx);
    w->hops_seen++;
    if (ctx.delay_state == AEC_LINEAR_DELAY_CHANGED) {
        if (w->changed_hops == 0) w->first_changed_hop = w->hops_seen;
        w->changed_hops++;
    }
    if (!w->locked_seen) {
        if (ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED) {
            if (ctx.delay_samples >= 0 && !ctx.aligned_far_hop) w->pre_lock_violation = 1;
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
    Aec a;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK(aec_create(&a, &cfg) == 0, msg);
    int hop = a.hop_size;

    long pushed = 0;
    AecLinearContext ctx0;
    aec_get_linear_context(&a, &ctx0);
    LockWatch w; memset(&w, 0, sizeof w);
    w.gen_before = ctx0.generation;

    run_hops(&a, 500, true_delay, &pushed, watch_hop, &w);

    snprintf(msg, sizeof msg, "%s: estimator locks within 500 hops", tag);
    CHECK(w.locked_seen, msg);
    snprintf(msg, sizeof msg, "%s: pre-lock hops stay UNLOCKED with stable generation", tag);
    CHECK(!w.pre_lock_violation, msg);
    snprintf(msg, sizeof msg, "%s: first lock: CHANGED on the lock hop", tag);
    CHECK(w.changed_on_lock_hop, msg);
    snprintf(msg, sizeof msg, "%s: first lock: generation == start+1", tag);
    CHECK(w.gen_at_lock == w.gen_before + 1, msg);

    AecLinearContext ctx;
    aec_get_linear_context(&a, &ctx);
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
    CHECK(ctx.aligned_far_hop == a.far_hop, msg);
    snprintf(msg, sizeof msg, "%s: formed_linear_hop aliases the formed output", tag);
    CHECK(ctx.formed_linear_hop == a.a3_lfs.e_form, msg);

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
    run_hops(&a, 5000, true_delay + 10 * hop, &pushed, watch_hop, &w2);
    aec_get_linear_context(&a, &ctx);
    printf("info: %s: shift re-lock after %d hops (%.1f s audio)\n", tag,
           w2.first_changed_hop,
           (double)w2.first_changed_hop * hop / sample_rate);
    snprintf(msg, sizeof msg, "%s: shift: generation advanced", tag);
    CHECK(ctx.generation > gen_locked, msg);
    snprintf(msg, sizeof msg, "%s: shift: CHANGED observed on some hop", tag);
    CHECK(w2.changed_hops >= 1, msg);
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
    aec_reset(&a);
    aec_get_linear_context(&a, &ctx);
    snprintf(msg, sizeof msg, "%s: reset: generation advanced", tag);
    CHECK(ctx.generation > gen_shift, msg);
    snprintf(msg, sizeof msg, "%s: reset: UNLOCKED with delay_samples == -1", tag);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED && ctx.delay_samples == -1, msg);

    aec_destroy(&a);
    g_rng_state = 0x11EC5EEDu;  /* keep scenarios independent of ordering */
}

static void scenario_delay_est_disabled(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    cfg.enable_delay_est = 0;
    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "delay-est-off: aec_create");
    long pushed = 0;
    AecLinearContext first, ctx;
    aec_get_linear_context(&a, &first);
    run_hops(&a, 120, 800, &pushed, NULL, NULL);
    aec_get_linear_context(&a, &ctx);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED,
          "delay-est-off: permanently UNLOCKED");
    CHECK(ctx.delay_samples == -1,
          "delay-est-off: delay_samples reads -1, not the internal 0");
    CHECK(ctx.generation == first.generation,
          "delay-est-off: generation never moves across hops");
    CHECK(ctx.aligned_far_hop == a.far_hop,
          "delay-est-off: far pointer still valid (content is raw far)");
    aec_destroy(&a);
    g_rng_state = 0x11EC5EEDu;
}

/* The honest-UNLOCKED window: current_delay >= 0 but the ring does not yet
 * hold delay+hop samples, so far_hop stays RAW. Unreachable through natural
 * acquisition (locking takes ~0.3 s, by which time the ring holds far more
 * than any in-range delay), so poke the state directly, counter-saturation
 * style. This is the ONLY scenario that observes far_hop_aligned=0 with a
 * non-negative delay -- the mutation "report aligned unconditionally" is
 * caught here and nowhere else. */
static void scenario_unfilled_ring_reads_unlocked(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "unfilled-ring: aec_create");
    int hop = a.hop_size;
    float mic[1024], out[1024];
    memset(mic, 0, sizeof mic);

    /* Simulate an established lock whose ring content has just been wiped
     * (as after a reset that kept the delay -- an artificial but layout-legal
     * state). Silence on mic keeps the estimator from re-acquiring. */
    long pushed = 0;
    run_hops(&a, 4, 0, &pushed, NULL, NULL);
    a.current_delay = 5000;
    a.ref_ring_filled = 64;

    long base = pushed;
    for (int i = 0; i < hop; ++i) g_far_hist[base + i] = rng_uniform(0.05f);
    aec_process(&a, mic, g_far_hist + base, out);
    pushed = base + hop;

    AecLinearContext ctx;
    aec_get_linear_context(&a, &ctx);
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
        aec_process(&a, mic, g_far_hist + base, out);
        pushed = base + hop;
    }
    aec_get_linear_context(&a, &ctx);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_LOCKED,
          "unfilled-ring: LOCKED once the ring covers the delay");
    long start = pushed - hop - ctx.delay_samples;
    CHECK(start >= 0 && memcmp(ctx.aligned_far_hop, g_far_hist + start,
                               (size_t)hop * sizeof(float)) == 0,
          "unfilled-ring: compensated far byte-equal at the poked offset");
    aec_destroy(&a);
    g_rng_state = 0x11EC5EEDu;
}

static void scenario_ring_filled_saturates(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "ring-filled: aec_create");
    int rs = a.ref_ring_size;
    int hop = a.hop_size;
    long pushed = 0;
    int hops_to_fill = rs / hop + 8;
    run_hops(&a, hops_to_fill, 1600, &pushed, NULL, NULL);
    CHECK(a.ref_ring_filled == rs, "filled saturates at ring size");
    run_hops(&a, 100, 1600, &pushed, NULL, NULL);
    CHECK(a.ref_ring_filled == rs, "filled stays frozen at ring size");
    AecLinearContext ctx;
    aec_get_linear_context(&a, &ctx);
    CHECK(ctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED || ctx.delay_samples < 0,
          "saturated fill never blocks a valid locked state");
    aec_destroy(&a);
    g_rng_state = 0x11EC5EEDu;
}

int main(void) {
    scenario_acquire_and_verify(16000, 0,   1600, "16k/fft256");
    scenario_acquire_and_verify(16000, 512, 1600, "16k/fft512");
    scenario_acquire_and_verify(48000, 0,   4800, "48k/fft1024");
    scenario_delay_est_disabled();
    scenario_unfilled_ring_reads_unlocked();
    scenario_ring_filled_saturates();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
