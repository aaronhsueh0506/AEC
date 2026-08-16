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
 *      acquisition in MATCHED, and UNLOCKED for the ring-fill window in
 *      FIXED, even though the applied delay is known from hop 0 there.
 *   4. ref_ring_filled saturates at ref_ring_size instead of growing without
 *      bound (its unbounded form is signed-overflow UB after ~37 h of
 *      continuous 16 kHz audio; the INT_MAX-seeded UBSan variant of this
 *      property lives in test_counter_saturation.c).
 *   5. The PER-MODE seam semantics (AecDelayMode; productization plan
 *      appendix 11.1) — each of MATCHED / FIXED / EXTERNAL_ALIGNED reports
 *      exactly one story about delay_samples / confidence / state /
 *      generation, and the far content it exposes matches that story:
 *        MATCHED           unchanged (scenarios 1-2 above)
 *        FIXED             delay_samples == the configured delay, always;
 *                          confidence 1.0; generation frozen while
 *                          processing; LOCKED once the ring can serve the
 *                          offset, and the exposed far is then byte-equal
 *                          to the caller's far delayed by exactly that many
 *                          samples
 *        EXTERNAL_ALIGNED  the exposed far IS the caller's own far (byte
 *                          for byte, every hop); delay_samples == 0;
 *                          LOCKED from hop 0; generation frozen
 *
 * Mutation checks (each reverts one line of the fix and must go red here):
 *   - drop the Path-A generation bump  -> "first lock: CHANGED on the lock
 *     hop" and "generation == g0+1" fail;
 *   - drop the Path-B generation bump  -> "shift: generation advanced" fails;
 *   - drop the aec_reset bump          -> "reset: generation advanced" fails;
 *   - report far_hop_aligned=1 unconditionally -> "pre-lock hops stay
 *     UNLOCKED" and "fixed: UNLOCKED until the ring can serve" fail;
 *   - remove the ref_ring_filled clamp -> "filled saturates at ring size"
 *     fails;
 *   - seed FIXED's current_delay from anything but cfg.fixed_delay_samples
 *     -> "fixed: aligned far is the caller far delayed by fixed" fails;
 *   - run the ring write/read under the estimator gate instead of the
 *     ring gate -> every "fixed: ..." alignment row fails;
 *   - report EXTERNAL_ALIGNED as UNLOCKED/-1 (its pre-delay_mode reading)
 *     -> the external-aligned rows fail.
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
 * 48 kHz/hop 512. The generous shift budget is deliberate: even with the
 * ERLE watchdog LIVE in this enable_res=0 context-only config (since the
 * Python-parity fix that widened the last_erle_windowed cache condition to
 * enable_res || return_res_context, commit 87fe169 -- see the note above
 * that cache write in aec.c; the watchdog used to be dead here), a
 * confirmed delay shift still needs seconds of audio to re-lock (ERLE
 * collapse resumes full-rate matched-filter analysis, then consensus must
 * rebuild). The measured re-lock hop count is printed as info. */
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

/* EXTERNAL_ALIGNED (and its legacy spelling, enable_delay_est=0): no
 * estimator, no ring. The caller CONTRACTED that far is already aligned, so
 * the seam says so -- delay_samples 0, LOCKED from the first hop, frozen
 * generation -- and the exposed far must be the caller's own hop byte for
 * byte, every hop, never a ring read.
 *
 * `legacy` drives the SAME assertions through the deprecated
 * enable_delay_est=0 spelling, proving the translation layer lands on this
 * mode rather than merely compiling. */
static void scenario_external_aligned(int legacy) {
    const char* tag = legacy ? "external-aligned(legacy est=0)"
                             : "external-aligned";
    char msg[160];
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    if (legacy) cfg.enable_delay_est = 0;
    else        cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    Aec a;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK(aec_create(&a, &cfg) == 0, msg);
    snprintf(msg, sizeof msg, "%s: resolves to EXTERNAL_ALIGNED", tag);
    CHECK(a.cfg.delay_mode == AEC_DELAY_EXTERNAL_ALIGNED, msg);
    snprintf(msg, sizeof msg, "%s: no ring carved", tag);
    CHECK(a.ref_ring == NULL && !a.has_delay, msg);

    int hop = a.hop_size;
    long pushed = 0;
    AecLinearContext first, ctx;
    aec_get_linear_context(&a, &first);

    /* Walk EVERY hop: the far the seam exposes must equal the far this test
     * just handed in (modulo nothing -- amplitude 0.05 never soft-clips). */
    int content_ok = 1, state_ok = 1, gen_ok = 1;
    for (int h = 0; h < 120; ++h) {
        long base = pushed;
        float mic[1024], out[1024];
        for (int i = 0; i < hop; ++i) {
            g_far_hist[base + i] = rng_uniform(0.05f);
            long src = base + i - 800;
            mic[i] = (src >= 0) ? 0.5f * g_far_hist[src] : 0.0f;
        }
        aec_process(&a, mic, g_far_hist + base, out);
        pushed = base + hop;
        aec_get_linear_context(&a, &ctx);
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
    CHECK(ctx.aligned_far_hop == a.far_hop, msg);
    aec_destroy(&a);
    g_rng_state = 0x11EC5EEDu;
}

/* FIXED: no estimator, but a ring that applies the caller's MEASURED delay
 * from the first hop it can serve it. The seam must report that delay
 * verbatim (never -1, never the estimator's 0.0 confidence), must not claim
 * LOCKED while the ring is still filling, and must never bump generation --
 * there is nothing that could re-lock. */
static void scenario_fixed_delay(int legacy) {
    const char* tag = legacy ? "fixed(legacy est=0+fixed)" : "fixed";
    char msg[160];
    const int fixed = 1600;   /* 100 ms at 16 kHz */
    AecConfig cfg;
    context_only_config(&cfg, 16000, 0);
    cfg.fixed_delay_samples = fixed;
    if (legacy) cfg.enable_delay_est = 0;
    else        cfg.delay_mode = AEC_DELAY_FIXED;
    Aec a;
    snprintf(msg, sizeof msg, "%s: aec_create", tag);
    CHECK(aec_create(&a, &cfg) == 0, msg);
    snprintf(msg, sizeof msg, "%s: resolves to FIXED", tag);
    CHECK(a.cfg.delay_mode == AEC_DELAY_FIXED, msg);
    snprintf(msg, sizeof msg, "%s: ring carved, no estimator", tag);
    CHECK(a.ref_ring != NULL && !a.has_delay, msg);

    int hop = a.hop_size;
    AecLinearContext first, ctx;
    aec_get_linear_context(&a, &first);
    snprintf(msg, sizeof msg, "%s: delay_samples reads the configured delay "
                              "before the first hop", tag);
    CHECK(first.delay_samples == fixed, msg);

    /* The ring cannot serve `fixed` until it holds fixed + hop samples, so
     * the honest state is UNLOCKED for exactly ceil(fixed/hop)+1 hops. */
    int fill_hops = (fixed + hop - 1) / hop + 1;
    long pushed = 0;
    int early_unlocked = 1, late_locked = 1, delay_ok = 1, conf_ok = 1, gen_ok = 1;
    int content_ok = 1, changed_seen = 0;
    for (int h = 0; h < 400; ++h) {
        long base = pushed;
        float mic[1024], out[1024];
        for (int i = 0; i < hop; ++i) {
            g_far_hist[base + i] = rng_uniform(0.05f);
            long src = base + i - fixed;
            mic[i] = (src >= 0) ? 0.5f * g_far_hist[src] : 0.0f;
        }
        aec_process(&a, mic, g_far_hist + base, out);
        pushed = base + hop;
        aec_get_linear_context(&a, &ctx);
        if (ctx.delay_samples != fixed) delay_ok = 0;
        if (ctx.delay_confidence != 1.0f) conf_ok = 0;
        if (ctx.generation != first.generation) gen_ok = 0;
        if (ctx.delay_state == AEC_LINEAR_DELAY_CHANGED) changed_seen = 1;
        if (h < fill_hops - 1) {
            if (ctx.delay_state != AEC_LINEAR_DELAY_UNLOCKED) early_unlocked = 0;
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
    snprintf(msg, sizeof msg, "%s: aligned far is the caller far delayed by fixed", tag);
    CHECK(content_ok, msg);
    snprintf(msg, sizeof msg, "%s: generation frozen while processing", tag);
    CHECK(gen_ok, msg);
    snprintf(msg, sizeof msg, "%s: CHANGED is unreachable without an estimator", tag);
    CHECK(!changed_seen, msg);

    /* aec_reset re-zeroes the ring, so it DOES invalidate a consumer's cached
     * far -- the one legitimate generation bump in this mode -- and must
     * re-seed the applied delay from the config, not to -1. */
    unsigned gen_before = ctx.generation;
    aec_reset(&a);
    aec_get_linear_context(&a, &ctx);
    snprintf(msg, sizeof msg, "%s: reset re-seeds the configured delay (not -1)", tag);
    CHECK(ctx.delay_samples == fixed && a.current_delay == fixed, msg);
    snprintf(msg, sizeof msg, "%s: reset returns to UNLOCKED (ring emptied)", tag);
    CHECK(ctx.delay_state == AEC_LINEAR_DELAY_UNLOCKED, msg);
    snprintf(msg, sizeof msg, "%s: reset bumps generation (ring content invalidated)", tag);
    CHECK(ctx.generation > gen_before, msg);
    aec_destroy(&a);
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
    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "fixed0: aec_create");
    CHECK(a.ref_ring != NULL, "fixed0: ring still carved (unlike EXTERNAL)");
    long pushed = 0;
    AecLinearContext first, ctx;
    aec_get_linear_context(&a, &first);
    run_hops(&a, 40, 0, &pushed, NULL, NULL);
    aec_get_linear_context(&a, &ctx);
    CHECK(ctx.delay_samples == 0 && ctx.delay_state == AEC_LINEAR_DELAY_LOCKED,
          "fixed0: LOCKED at delay 0 from the start");
    CHECK(ctx.generation == first.generation,
          "fixed0: generation frozen while processing");
    CHECK(memcmp(ctx.aligned_far_hop, g_far_hist + pushed - a.hop_size,
                 (size_t)a.hop_size * sizeof(float)) == 0,
          "fixed0: exposed far is the caller's latest hop");
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
    scenario_external_aligned(/*legacy=*/0);
    scenario_external_aligned(/*legacy=*/1);
    scenario_fixed_delay(/*legacy=*/0);
    scenario_fixed_delay(/*legacy=*/1);
    scenario_fixed_zero_delay();
    scenario_unfilled_ring_reads_unlocked();
    scenario_ring_filled_saturates();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
