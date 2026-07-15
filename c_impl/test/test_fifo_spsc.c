/* test_fifo_spsc.c — F09 regression test (async render/capture FIFO data
 * race). Exercises the streaming API (aec_analyze_render() / aec_process_
 * capture(), aec.h ~F09 doc block) under real SPSC concurrency: one render
 * thread, one capture thread, both hammering the same Aec instance with no
 * external lock, exactly the shape aec.h's threading contract promises is
 * safe.
 *
 * This test targets the F09 Variant A' rewrite (drop-new + consumer
 * catch-up, see aec.h/aec.c): fifo_write is written ONLY by the render
 * thread and fifo_read is written ONLY by the capture thread (each a plain
 * monotonic unsigned cursor, published with a single acquire/release pair —
 * no RMW anywhere). This test PASSES ThreadSanitizer with ZERO suppressions
 * (see STEP 4 of the implementation report / the TSan build recipe below) —
 * that is the proof artifact for the redesign's race-freedom claim.
 *
 * Two modes:
 *   (no args)              — threaded SPSC PAYLOAD-INTEGRITY stress test:
 *                             producer pushes N_HOPS deterministic hops via
 *                             aec_analyze_render(), where every hop's full
 *                             content is a pure function of a monotonic
 *                             `call_seq` (the producer's own loop index) —
 *                             see stamp_hop() below. The consumer decodes
 *                             call_seq back out of a->far_hop after every
 *                             aec_process_capture() and REGENERATES the
 *                             expected hop independently, then requires a
 *                             bit-exact memcmp — this is what would catch a
 *                             torn/partial hop (two different call_seqs'
 *                             bytes mixed in one buffer) under a broken
 *                             SPSC implementation, not just "no crash".
 *                             Jitter (LCG-driven nanosleep on both sides,
 *                             plus a zero-jitter consumer "sprint" phase)
 *                             perturbs relative producer/consumer speed so
 *                             both the drop-new/catch-up (overrun) and
 *                             underrun paths are reliably exercised.
 *                             Asserts no crash, every output sample finite,
 *                             full payload integrity, strict call_seq
 *                             monotonicity, bounded staleness, and that the
 *                             FIFO's own bookkeeping (fifo_write/fifo_read/
 *                             fifo_count) matches the derived
 *                             produced/consumed/skipped arithmetic exactly
 *                             once both threads have joined.
 *   --singlethread <path>  — single-thread reference run: strict lockstep
 *                             (analyze_render immediately followed by
 *                             process_capture, alternating, same thread) on
 *                             the SAME deterministic LCG signal as before
 *                             the F09 Variant A' rewrite (this mode is
 *                             intentionally left byte-for-byte unchanged —
 *                             same lcg_next()/fill_hop(), same n_hops=4000 —
 *                             so a pre-rewrite dump and a post-rewrite dump
 *                             can be `cmp`'d: the streaming API's lockstep
 *                             behaviour must not change, aec.h's documented
 *                             lockstep-equivalence contract), dumping raw
 *                             float32 output to <path>.
 *
 * Build (standalone, from c_impl/ — mirrors test_lifecycle.c's documented
 * recipe, plus -pthread for the two worker threads):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -pthread \
 *       -Iinclude -Iexample -I../../audio_common/include \
 *       test/test_fifo_spsc.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_fifo_spsc
 * Run:
 *   ./bin/test_fifo_spsc
 *   ./bin/test_fifo_spsc --singlethread /tmp/out.bin
 *
 * ThreadSanitizer (same recipe, add -fsanitize=thread to BOTH compile and
 * link, -O1 recommended so TSan's stack traces stay readable):
 *   gcc -Wall -Wextra -O1 -ffp-contract=off -std=gnu99 -pthread \
 *       -fsanitize=thread -Iinclude -Iexample -I../../audio_common/include \
 *       test/test_fifo_spsc.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm -fsanitize=thread \
 *       -o bin/test_fifo_spsc_tsan
 *   ./bin/test_fifo_spsc_tsan
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

/* ── deterministic LCG signal source (--singlethread dump mode + this
 * test's mic/near-end content) ───────────────────────────────────────────
 * Two independent streams (far-end / near-end) so the mic and ref hops
 * aren't identical. Small amplitude (±0.1) — this test is about the FIFO
 * mechanism, not echo-cancellation quality, so any bounded, non-degenerate
 * signal is enough to keep aec_process()'s DSP well away from saturation/
 * NaN territory while still doing real (non-trivial) work per hop.
 * UNCHANGED by the F09 Variant A' rewrite: run_singlethread_dump() below
 * still drives both ref and mic from these two functions exactly as
 * before, so a pre-rewrite and post-rewrite dump remain comparable. */
static unsigned int lcg_state_far = 0x9E3779B9u;
static unsigned int lcg_state_near = 0x2545F491u;

static float lcg_next(unsigned int* state) {
    *state = (*state) * 1103515245u + 12345u;
    return ((float)((*state >> 8) & 0xFFFFu) / 65536.0f - 0.5f) * 0.2f;
}

static void fill_hop(float* buf, int hop, unsigned int* state) {
    for (int i = 0; i < hop; ++i) buf[i] = lcg_next(state);
}

/* ── threaded SPSC payload-integrity stress test ──────────────────────────
 */
#define N_HOPS 100000

/* Bounded-staleness slack (assertion 5): g_enq_watermark is a SEPARATE
 * test-side atomic from the FIFO's own fifo_write/fifo_read handshake (see
 * producer_main below) — there is no cross-variable ordering guarantee
 * between two independent atomics, so the watermark the consumer observes
 * for a given hop's enq_seq can lag very slightly behind that hop's own
 * publish. A few hops of slack absorbs that without weakening the check
 * (which is fundamentally about the ring's `cap` bound, not this slack). */
#define STALENESS_SLACK 8

/* Test-side diagnostic counter: mirrors the producer's own successful-push
 * ordinal (see ProducerArgs.enq_count) so the consumer can sanity-check how
 * stale a given hop is relative to total progress. NOT part of the FIFO's
 * correctness contract (aec.c's fifo_write/fifo_read ACQUIRE/RELEASE pair
 * is what actually guarantees the payload's visibility) — this is purely an
 * additional, independent assertion this test layers on top. */
static unsigned int g_enq_watermark = 0;

/* ── payload-integrity hop generator ──────────────────────────────────────
 * A pure function of call_seq — deliberately NOT the running lcg_state_far/
 * lcg_state_near streams above (those are reserved for run_singlethread_
 * dump()'s byte-identity contract, untouched by anything in this section).
 * f[0]/f[1] hold small non-negative integers (call_seq / enq_seq, both
 * N_HOPS=100000 << 2^24) scaled by 2^-24: an EXACT float32 round-trip
 * (multiplying/dividing by a power of two only moves the exponent, and any
 * integer < 2^24 is exactly representable in a float32's 24-bit mantissa),
 * so decode_seq() below recovers the exact integer, not an approximation.
 * f[2..hop-1] is LCG noise reseeded from call_seq alone, amplitude ±0.012 —
 * comparable magnitude to f[0]/f[1] and well inside aec_process()'s normal
 * operating range (no saturation/NaN risk). Because the WHOLE hop is a
 * deterministic function of call_seq (and, for f[1] only, enq_seq), the
 * consumer can independently regenerate the exact expected hop from a
 * decoded call_seq/enq_seq pair and catch a torn/partial hop (bytes from
 * two different pushes mixed in one buffer) with a bit-exact memcmp. */
static void stamp_hop(float* buf, int hop, long call_seq, long enq_seq) {
    buf[0] = (float)call_seq * 0x1p-24f;
    buf[1] = (float)enq_seq  * 0x1p-24f;
    unsigned int st = (unsigned int)call_seq * 2654435761u + 0x9E3779B9u;
    for (int i = 2; i < hop; ++i) {
        st = st * 1103515245u + 12345u;
        buf[i] = ((float)((st >> 8) & 0xFFFFu) / 65536.0f - 0.5f) * 0.024f;  /* ±0.012 */
    }
}

/* Decode an integer stamped by stamp_hop() back out of its scaled float
 * representation. The round-trip is exact for any value < 2^24 (see
 * stamp_hop's comment); lrintf is defensive rounding, not a correction for
 * an inexact conversion. */
static long decode_seq(float scaled) {
    return (long)lrintf(scaled * 16777216.0f);   /* undo the *0x1p-24f scale */
}

/* ── LCG-driven jitter (nanosleep) ────────────────────────────────────────
 * Perturbs relative producer/consumer speed so both the overrun (drop-new /
 * catch-up) and underrun paths get exercised reliably, rather than relying
 * entirely on whichever one the raw, unthrottled per-call DSP-cost gap
 * happens to produce on a given host. */
static void jitter_sleep_us(unsigned int* state, unsigned int max_us) {
    if (max_us == 0) return;
    *state = (*state) * 1103515245u + 12345u;
    unsigned int us = ((*state) >> 16) % (max_us + 1u);
    if (us == 0) return;
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = (long)us * 1000L;
    nanosleep(&ts, NULL);
}

typedef struct {
    Aec* a;
    int  hop;
    long overrun_count;   /* drop-new events (producer-observed) */
    long enq_count;       /* producer-private: successful pushes so far */
} ProducerArgs;

typedef struct {
    Aec* a;
    int  hop;
    long underrun_count;
    long overrun_count;        /* F09 Variant A': catch-up events (consumer-observed) */
    int  all_finite;
    int  integrity_ok;         /* every non-underrun hop matched its regenerated expected hop */
    int  monotonic_ok;         /* decoded call_seq strictly increasing across consumed hops */
    int  underrun_zero_ok;     /* every UNDERRUN hop's far_hop was exactly all-zero */
    int  staleness_ok;         /* bounded-staleness check never violated */
    long gaps_total;           /* sum of (call_seq - prev - 1) over consumed hops */
    long last_consumed_call_seq;  /* -1 == nothing consumed yet */
    int  have_consumed;
} ConsumerArgs;

static void* producer_main(void* arg) {
    ProducerArgs* pa = (ProducerArgs*)arg;
    float* ref = (float*)malloc((size_t)pa->hop * sizeof(float));
    unsigned int jlcg = 0xC0FFEE01u;
    for (long call_seq = 0; call_seq < N_HOPS; ++call_seq) {
        /* Stamp the WHOLE hop before the call -- a pure function of
         * call_seq (+ the producer's own running enq_count for f[1]) so it
         * is fully determined regardless of whether this push succeeds. */
        stamp_hop(ref, pa->hop, call_seq, pa->enq_count);
        AecBufferingEvent ev = aec_analyze_render(pa->a, ref);
        if (ev == AEC_BUF_RENDER_OVERRUN) {
            pa->overrun_count++;
        } else {
            pa->enq_count++;
            __atomic_store_n(&g_enq_watermark, (unsigned int)pa->enq_count, __ATOMIC_RELEASE);
        }
        if ((call_seq % 8) == 0) jitter_sleep_us(&jlcg, 50);
    }
    free(ref);
    return NULL;
}

static void* consumer_main(void* arg) {
    ConsumerArgs* ca = (ConsumerArgs*)arg;
    float* mic    = (float*)malloc((size_t)ca->hop * sizeof(float));
    float* out    = (float*)malloc((size_t)ca->hop * sizeof(float));
    float* expect = (float*)malloc((size_t)ca->hop * sizeof(float));
    ca->all_finite        = 1;
    ca->integrity_ok      = 1;
    ca->monotonic_ok      = 1;
    ca->underrun_zero_ok  = 1;
    ca->staleness_ok      = 1;
    ca->gaps_total        = 0;
    ca->last_consumed_call_seq = -1;
    ca->have_consumed     = 0;

    unsigned int jlcg = 0x13572468u;
    const long sprint_start = (long)(N_HOPS * 9L / 10L);   /* last 10%: zero-jitter sprint */
    const unsigned cap = (unsigned)ca->a->fifo_cap_hops;

    for (long i = 0; i < N_HOPS; ++i) {
        fill_hop(mic, ca->hop, &lcg_state_near);   /* near-end content only -- not integrity-checked */
        AecBufferingEvent ev = aec_process_capture(ca->a, mic, out);

        /* Same-thread read: aec_process()'s very first act is to memcpy the
         * consumed `ref` into a->far_hop (aec.c), so this is an exact byte
         * copy of whatever this capture call consumed -- race-free here
         * because only the capture thread ever reads far_hop and this IS
         * the capture thread. */
        const float* far = ca->a->far_hop;

        if (ev == AEC_BUF_RENDER_UNDERRUN) {
            ca->underrun_count++;
            /* Assertion 1: underrun processed the immutable fifo_zero_ref
             * -- far_hop must be exactly all-zero bits (not just small),
             * and the sequence chain (last_consumed_call_seq/gaps_total)
             * is untouched by this branch, by design. */
            for (int k = 0; k < ca->hop; ++k) {
                if (far[k] != 0.0f) { ca->underrun_zero_ok = 0; break; }
            }
        } else {
            if (ev == AEC_BUF_RENDER_OVERRUN) ca->overrun_count++;   /* catch-up */

            long call_seq = decode_seq(far[0]);
            long enq_seq  = decode_seq(far[1]);

            /* Assertion 2: regenerate the full expected hop from the
             * decoded call_seq/enq_seq and require a bit-exact memcmp --
             * this is what catches a torn/partial hop. */
            stamp_hop(expect, ca->hop, call_seq, enq_seq);
            if (memcmp(expect, far, (size_t)ca->hop * sizeof(float)) != 0)
                ca->integrity_ok = 0;

            /* Assertion 3: strictly increasing call_seq (no dup/backward). */
            if (ca->have_consumed && call_seq <= ca->last_consumed_call_seq)
                ca->monotonic_ok = 0;

            /* Assertion 4: accumulate the call_seq-domain gap (missing
             * call_seq values strictly between this hop and the previously
             * consumed one; the very first consumed hop's "gap" is
             * everything from 0 up to call_seq-1). */
            if (ca->have_consumed)
                ca->gaps_total += (call_seq - ca->last_consumed_call_seq - 1);
            else
                ca->gaps_total += call_seq;

            ca->last_consumed_call_seq = call_seq;
            ca->have_consumed = 1;

            /* Assertion 5: bounded staleness. watermark_snapshot can only
             * lag the true current successful-push count (see
             * g_enq_watermark's comment), so it can only UNDER-report how
             * far ahead the producer has gotten -- STALENESS_SLACK absorbs
             * that lag on top of the ring-derived `cap` bound. */
            unsigned wm = __atomic_load_n(&g_enq_watermark, __ATOMIC_ACQUIRE);
            long staleness = (long)wm - enq_seq;
            if (staleness > (long)cap + STALENESS_SLACK) ca->staleness_ok = 0;
        }

        for (int k = 0; k < ca->hop; ++k) {
            if (!isfinite(out[k])) { ca->all_finite = 0; break; }
        }

        /* Jitter: heavy periodic stalls for the first 90% of the loop
         * (forces the ring to fill -- overrun bursts), a zero-jitter
         * "sprint" for the last 10% (forces the consumer to run dry once
         * the producer's fixed N_HOPS budget is exhausted -- underruns). */
        if (i < sprint_start) {
            if ((i % 64) == 0) jitter_sleep_us(&jlcg, 500);
        }
    }
    free(mic); free(out); free(expect);
    return NULL;
}

static void run_stress_test(void) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    Aec a;
    int created = (aec_create(&a, &cfg) == 0);
    CHECK(created, "stress: aec_create() succeeds");
    if (!created) return;

    int hop = aec_hop_size(&a);
    CHECK(hop > 0 && hop <= 4096, "stress: hop_size sane (0 < hop <= 4096)");

    g_enq_watermark = 0;

    ProducerArgs pa; memset(&pa, 0, sizeof(pa)); pa.a = &a; pa.hop = hop;
    ConsumerArgs ca; memset(&ca, 0, sizeof(ca)); ca.a = &a; ca.hop = hop;

    /* Explicit large stack (16 MiB), not the platform default (512 KiB on
     * macOS pthreads): aec_process()'s call chain runs through the entire
     * AEC3 post-filter stack (FFTs, ResidualEchoEstimator, SuppressionGain,
     * ...), and instrumented builds (ThreadSanitizer in particular) inflate
     * per-frame stack usage well past what the default budget survives with
     * plain, uninstrumented builds — a stack overflow there is an artifact
     * of the test's default pthread attrs, not of the FIFO fix under test. */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16u * 1024u * 1024u);

    pthread_t t_prod, t_cons;
    int r1 = pthread_create(&t_prod, &attr, producer_main, &pa);
    int r2 = pthread_create(&t_cons, &attr, consumer_main, &ca);
    CHECK(r1 == 0 && r2 == 0, "stress: both threads spawned");
    pthread_attr_destroy(&attr);

    pthread_join(t_prod, NULL);
    pthread_join(t_cons, NULL);

    /* Everything below runs single-threaded again (both workers joined), so
     * plain reads of the Aec's internal fields are race-free here — this is
     * the test harness inspecting final state after synchronization, not a
     * third concurrent actor. */
    unsigned final_w = *(unsigned*)&a.fifo_write;
    unsigned final_r = *(unsigned*)&a.fifo_read;
    unsigned cap_u   = (unsigned)a.fifo_cap_hops;

    printf("render_call_count=%ld capture_call_count=%ld\n",
           a.render_call_count, a.capture_call_count);
    printf("producer overrun(drop-new)=%ld  consumer overrun(catch-up)=%ld  underrun=%ld\n",
           pa.overrun_count, ca.overrun_count, ca.underrun_count);
    printf("final fifo_write=%u fifo_read=%u (cap=%u)  fifo_count=%d\n",
           final_w, final_r, cap_u, a.fifo_count);

    CHECK(a.render_call_count == N_HOPS, "stress: render_call_count == N_HOPS");
    CHECK(a.capture_call_count == N_HOPS, "stress: capture_call_count == N_HOPS");
    CHECK(ca.all_finite, "stress: every captured output sample is finite");
    CHECK(ca.integrity_ok, "stress: every consumed hop's payload matches its decoded call_seq (no torn hops)");
    CHECK(ca.monotonic_ok, "stress: decoded call_seq strictly increasing across consumed hops");
    CHECK(ca.underrun_zero_ok, "stress: every UNDERRUN hop's far_hop is exactly all-zero");
    CHECK(ca.staleness_ok, "stress: bounded-staleness check never violated");

    /* Both buffering-event paths must actually be exercised (not just
     * "possible in theory") — an unthrottled producer racing a full-DSP
     * consumer fills the ring almost immediately (drop-new), then the
     * producer's fixed N_HOPS budget runs out long before the consumer's
     * does, so the ring drains and the consumer runs dry for the rest
     * (underrun); the jitter above makes both outcomes robust to exact
     * relative thread speed on whatever host runs this. F09 Variant A'
     * additionally gives the CONSUMER its own overrun path (catch-up, fired
     * when it finds the ring completely full) alongside the producer's. */
    CHECK(pa.overrun_count >= 1, "stress: drop-new (producer overrun) path was exercised");
    CHECK(ca.underrun_count >= 1, "stress: underrun path was exercised");
    CHECK(ca.overrun_count >= 1, "stress: catch-up (consumer overrun) path was exercised");

    /* Bookkeeping sanity: the ring invariant 0 <= fifo_write - fifo_read <=
     * cap must hold no matter how the two threads interleaved -- unsigned
     * subtraction makes the lower bound automatic, only the upper bound
     * needs an explicit check. */
    CHECK(final_w - final_r <= cap_u, "stress: 0 <= fifo_write - fifo_read <= cap");
    CHECK(a.fifo_count == 0, "stress: fifo_count RETIRED field stayed 0 (layout-stability field, see aec.h)");

    /* fifo_write identity: it advances by exactly 1 per successful push, so
     * its final value is exactly the render call count minus the drop-new
     * count. */
    CHECK((long)final_w == a.render_call_count - pa.overrun_count,
          "stress: fifo_write == renders - producer_drop_count");

    /* fifo_read identity: a NORMAL consume advances it by 1; an UNDERRUN
     * touches it not at all; a CATCH-UP jumps it by exactly `cap` in one
     * call (r=w-1 then released to w, i.e. +cap from the pre-call r, since
     * catch-up only fires when w-r==cap). So the final value is
     *   (normal consumes)*1 + (catch-ups)*cap
     * = (captures - underrun - catchups)*1 + catchups*cap
     * = captures - underrun + catchups*(cap-1)
     * This is exact and timing-independent (unlike the call_seq-domain
     * gaps_total reconciliation below, it needs no residual correction). */
    long skip_totals = ca.overrun_count * (long)(cap_u - 1);
    CHECK((long)final_r == a.capture_call_count - ca.underrun_count + skip_totals,
          "stress: fifo_read == captures - underrun_count + skip_totals (skip_totals = catchup_count*(cap-1))");

    /* Reconcile the call_seq-domain gap accumulator (assertion 4 in the
     * per-hop oracle) against the ring-domain counters above. gaps_total,
     * by construction, only sums gaps STRICTLY BETWEEN consumed call_seq
     * values -- it cannot see (a) producer drops that happen AFTER the
     * globally last-consumed call_seq (nothing subsequent creates a gap to
     * reveal them), or (b) hops still sitting unconsumed in the ring at
     * join time (the final fifo_write - fifo_read leftover, drained by
     * neither a normal consume nor a catch-up before the test ended).
     * Both of those together are exactly "every call_seq value greater
     * than last_consumed_call_seq that is not itself part of the
     * leftover-in-ring tail":
     *   trailing_drops = (N_HOPS - 1 - last_consumed_call_seq) - leftover
     * Accounting for that residual, the identity that always holds exactly
     * (regardless of relative thread speed / scheduling) is:
     *   gaps_total + trailing_drops == producer_drops + skip_totals
     * In the overwhelmingly likely case for this stress shape (the ring
     * drains completely and the very last render call of the whole run
     * happens to succeed), trailing_drops == 0 and this reduces to the
     * simple gaps_total == producer_drops + skip_totals; the general form
     * below holds unconditionally. */
    long leftover = (long)final_w - (long)final_r;
    long trailing_drops = (N_HOPS - 1 - ca.last_consumed_call_seq) - leftover;
    printf("gaps_total=%ld trailing_drops=%ld leftover=%ld last_consumed_call_seq=%ld\n",
           ca.gaps_total, trailing_drops, leftover, ca.last_consumed_call_seq);
    CHECK(trailing_drops >= 0, "stress: trailing_drops is non-negative (sanity on the residual accounting)");
    CHECK(ca.gaps_total + trailing_drops == pa.overrun_count + skip_totals,
          "stress: gaps_total + trailing_drops == producer_drops + skip_totals (F09 Variant A' identity)");

    aec_destroy(&a);
}

/* ── single-thread lockstep reference dump (byte-identity dance) ─────────
 * Strict alternation on ONE thread: aec_analyze_render(ref) immediately
 * followed by aec_process_capture(mic, out) — the FIFO stays pass-through
 * (aec.h's documented lockstep equivalence), so this must produce identical
 * output whether run against the pre-F09-Variant-A' or post-rewrite aec.c.
 * Left byte-for-byte unchanged by the F09 Variant A' rewrite (same
 * lcg_next()/fill_hop(), same n_hops=4000) so a pre-rewrite dump and a
 * post-rewrite dump can be `cmp`'d directly. */
static void run_singlethread_dump(const char* path) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    Aec a;
    if (aec_create(&a, &cfg) != 0) {
        fprintf(stderr, "aec_create failed\n");
        exit(1);
    }
    int hop = aec_hop_size(&a);

    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* out = (float*)malloc((size_t)hop * sizeof(float));

    const int n_hops = 4000;   /* a few thousand hops is plenty to exercise
                                 * the pass-through path repeatedly */
    for (int i = 0; i < n_hops; ++i) {
        fill_hop(ref, hop, &lcg_state_far);
        fill_hop(mic, hop, &lcg_state_near);
        aec_analyze_render(&a, ref);
        aec_process_capture(&a, mic, out);
        fwrite(out, sizeof(float), (size_t)hop, f);
    }

    free(ref); free(mic); free(out);
    fclose(f);
    aec_destroy(&a);
    printf("wrote %d hops (%d samples/hop) to %s\n", n_hops, hop, path);
}

int main(int argc, char** argv) {
    if (argc >= 3 && strcmp(argv[1], "--singlethread") == 0) {
        run_singlethread_dump(argv[2]);
        return 0;
    }

    run_stress_test();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
