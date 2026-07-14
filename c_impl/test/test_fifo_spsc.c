/* test_fifo_spsc.c — F09 regression test (async render/capture FIFO data
 * race). Exercises the streaming API (aec_analyze_render() / aec_process_
 * capture(), aec.h ~F09 doc block) under real SPSC concurrency: one render
 * thread, one capture thread, both hammering the same Aec instance with no
 * external lock, exactly the shape aec.h's threading contract promises is
 * safe. Before the F09 fix, the FIFO bookkeeping (fifo_read/fifo_count) was
 * plain non-atomic `int` read-modify-write — a data race (UB) the moment
 * this test's two threads actually run concurrently. After the fix, every
 * cross-thread touch goes through `__atomic_*_n` builtins (see aec.c).
 *
 * Two modes:
 *   (no args)              — threaded SPSC stress test: producer pushes
 *                             100k deterministic-LCG hops via
 *                             aec_analyze_render(), consumer pulls the same
 *                             count via aec_process_capture(). Asserts no
 *                             crash, every output sample finite, and the
 *                             FIFO bookkeeping is internally consistent
 *                             (fifo_count in range + matches the derived
 *                             produced/consumed arithmetic) once both
 *                             threads have joined. Reports the observed
 *                             overrun/underrun counts.
 *   --singlethread <path>  — single-thread reference run: strict lockstep
 *                             (analyze_render immediately followed by
 *                             process_capture, alternating, same thread) on
 *                             the same deterministic LCG signal, dumping raw
 *                             float32 output to <path>. Used for the
 *                             snapshot/rebuild byte-identity dance: run this
 *                             mode against a pre-F09-fix build and a
 *                             post-fix build and `cmp` the two files — the
 *                             streaming API's lockstep behaviour must not
 *                             change (aec.h's documented lockstep-equivalence
 *                             contract).
 *
 * Build (standalone, from c_impl/ — mirrors test_lifecycle.c's documented
 * recipe, plus -pthread for the two worker threads):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -pthread \
 *       -Iinclude -Iexample -I../../audio_common/include \
 *       test/test_fifo_spsc.c $(find src -name '*.c') \
 *       ../../audio_common/bin/kiss/libaudio_common.a -lm \
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
 *       ../../audio_common/bin/kiss/libaudio_common.a -lm -fsanitize=thread \
 *       -o bin/test_fifo_spsc_tsan
 *   ./bin/test_fifo_spsc_tsan
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

/* ── deterministic LCG signal source ──────────────────────────────────────
 * Two independent streams (far-end / near-end) so the mic and ref hops
 * aren't identical. Small amplitude (±0.1) — this test is about the FIFO
 * mechanism, not echo-cancellation quality, so any bounded, non-degenerate
 * signal is enough to keep aec_process()'s DSP well away from saturation/
 * NaN territory while still doing real (non-trivial) work per hop. */
static unsigned int lcg_state_far = 0x9E3779B9u;
static unsigned int lcg_state_near = 0x2545F491u;

static float lcg_next(unsigned int* state) {
    *state = (*state) * 1103515245u + 12345u;
    return ((float)((*state >> 8) & 0xFFFFu) / 65536.0f - 0.5f) * 0.2f;
}

static void fill_hop(float* buf, int hop, unsigned int* state) {
    for (int i = 0; i < hop; ++i) buf[i] = lcg_next(state);
}

/* ── threaded SPSC stress test ────────────────────────────────────────────
 */
#define N_HOPS 100000

typedef struct {
    Aec* a;
    int  hop;
    long overrun_count;
} ProducerArgs;

typedef struct {
    Aec* a;
    int  hop;
    long underrun_count;
    int  all_finite;
} ConsumerArgs;

static void* producer_main(void* arg) {
    ProducerArgs* pa = (ProducerArgs*)arg;
    float* ref = (float*)malloc((size_t)pa->hop * sizeof(float));
    for (long i = 0; i < N_HOPS; ++i) {
        fill_hop(ref, pa->hop, &lcg_state_far);
        AecBufferingEvent ev = aec_analyze_render(pa->a, ref);
        if (ev == AEC_BUF_RENDER_OVERRUN) pa->overrun_count++;
    }
    free(ref);
    return NULL;
}

static void* consumer_main(void* arg) {
    ConsumerArgs* ca = (ConsumerArgs*)arg;
    float* mic = (float*)malloc((size_t)ca->hop * sizeof(float));
    float* out = (float*)malloc((size_t)ca->hop * sizeof(float));
    ca->all_finite = 1;
    for (long i = 0; i < N_HOPS; ++i) {
        fill_hop(mic, ca->hop, &lcg_state_near);
        AecBufferingEvent ev = aec_process_capture(ca->a, mic, out);
        if (ev == AEC_BUF_RENDER_UNDERRUN) ca->underrun_count++;
        for (int k = 0; k < ca->hop; ++k) {
            if (!isfinite(out[k])) { ca->all_finite = 0; break; }
        }
    }
    free(mic); free(out);
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
    printf("render_call_count=%ld capture_call_count=%ld\n",
           a.render_call_count, a.capture_call_count);
    printf("overrun_count=%ld underrun_count=%ld\n", pa.overrun_count, ca.underrun_count);
    printf("final fifo_count=%d (cap=%d)\n", a.fifo_count, a.fifo_cap_hops);

    CHECK(a.render_call_count == N_HOPS, "stress: render_call_count == N_HOPS");
    CHECK(a.capture_call_count == N_HOPS, "stress: capture_call_count == N_HOPS");
    CHECK(ca.all_finite, "stress: every captured output sample is finite");

    /* Bookkeeping sanity: fifo_count must stay in [0, cap] no matter how the
     * two threads interleaved (this is exactly the invariant a corrupted/
     * racy FIFO would violate — going negative or past capacity). */
    CHECK(a.fifo_count >= 0 && a.fifo_count <= a.fifo_cap_hops,
          "stress: final fifo_count within [0, cap]");

    /* A NON-overrun render call grows the FIFO by exactly one item; an
     * overrun call nets to ZERO growth (it drops one to make room, then
     * inserts one — capacity was already full and stays full, see
     * aec_analyze_render()'s comment in aec.c). Symmetrically, a NON-
     * underrun capture call shrinks the FIFO by exactly one; an underrun
     * call touches nothing. So the final occupancy is fully determined by
     * the four counters alone; if the atomics were wrong this identity
     * would very likely be violated (it does not merely check "no crash",
     * it validates the produced vs. consumed arithmetic survived the
     * concurrency). */
    long expected_final_count = (a.render_call_count - pa.overrun_count) -
                                 (a.capture_call_count - ca.underrun_count);
    CHECK((long)a.fifo_count == expected_final_count,
          "stress: final fifo_count matches produced-minus-consumed arithmetic");

    /* Sanity on the counters themselves: overrun cannot exceed the number of
     * render calls, underrun cannot exceed the number of capture calls. An
     * unthrottled producer racing a full-DSP consumer is EXPECTED to overrun
     * heavily (the consumer is far slower per call) — that is normal load
     * behaviour for this stress shape, not a bug; the assertions only check
     * the counts are not corrupted (out of their possible range). */
    CHECK(pa.overrun_count >= 0 && pa.overrun_count <= N_HOPS,
          "stress: overrun_count within [0, N_HOPS]");
    CHECK(ca.underrun_count >= 0 && ca.underrun_count <= N_HOPS,
          "stress: underrun_count within [0, N_HOPS]");

    aec_destroy(&a);
}

/* ── single-thread lockstep reference dump (byte-identity dance) ─────────
 * Strict alternation on ONE thread: aec_analyze_render(ref) immediately
 * followed by aec_process_capture(mic, out) — the FIFO stays pass-through
 * (aec.h's documented lockstep equivalence), so this must produce identical
 * output whether run against the pre-F09-fix or post-fix aec.c. */
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
