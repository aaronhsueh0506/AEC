/* test_reset_parity.c — aec_reset() delivers a fresh instance.
 *
 * The contract on aec_reset() is that the instance it returns is
 * indistinguishable from one just built on the same config. That was prose,
 * not a gate, and it was false: state survived the call that a fresh
 * construction never has, and the first post-reset hop already diverged.
 *
 * The gate is byte identity of everything a caller can observe, over a run
 * long enough for the filter to converge and the delay estimator to lock:
 *   - out[hop]                    (aec_process; the audio a caller ships)
 *   - AecResContext.formed_hop    (the linear-AEC seam's time output)
 *   - AecResContext.error_spec    (its freq output, n_freqs Complex)
 * Reference instance: constructed, never warmed. Subject instance:
 * constructed, warmed WARM_HOPS on an unrelated echo path, then aec_reset().
 * Both are then fed the same TEST_HOPS of a second echo path and compared
 * hop by hop. A residue that steers any decision moves one of the three.
 *
 * Covered per run: the three product grids (16k/256, 16k/512, 48k/1024) x
 * both consumption modes (full cascade, and the enable_res=0 +
 * return_res_context=1 seam an external RES drives). The FFT backend is
 * whatever the build selected, so `make test-reset-parity` and
 * `make BACKEND=kiss test-reset-parity` are two distinct runs of the gate.
 * The caller-pool path needs no separate case here: test_static_aec.c
 * already pins aec_init byte-equal to aec_create on a real recording.
 *
 * Build (standalone, from c_impl/ -- mirrors test_process_context.c's recipe):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       -I../../audio_common/include \
 *       test/test_reset_parity.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_reset_parity
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Long enough that the main filter is well past its startup gates, the
 * delay estimator has locked, and the ERLE/leakage trackers hold real
 * values -- i.e. enough warm state that leaving any of it standing shows. */
#define WARM_HOPS  600
/* Long enough to catch a divergence that only opens once the reference
 * instance leaves ITS startup window, not just at hop 0. */
#define TEST_HOPS  400

static int g_fail = 0;
static int g_pass = 0;

/* Deterministic, self-contained noise -- no libc rand(), so the gate reads
 * the same on every platform and the two instances are fed identical bytes. */
typedef struct { unsigned int s; } Rng;
static void rng_seed(Rng* r, unsigned int seed) { r->s = seed; }
static float rng_next(Rng* r) {
    r->s = r->s * 1103515245u + 12345u;
    return (float)((r->s >> 9) & 0x7fffff) / 8388608.0f * 2.0f - 1.0f;
}

/* One hop of a far-end signal plus a mic that is a delayed, attenuated copy
 * of it with near-end noise on top -- an echo path the delay estimator can
 * actually lock onto. `hist` carries the far tail across hops so the delay
 * is real rather than a per-hop wrap. */
typedef struct {
    Rng   rng;
    float hist[4096];
    int   delay;
    float erl;
} Scene;

static void scene_init(Scene* s, unsigned int seed, int delay, float erl) {
    rng_seed(&s->rng, seed);
    memset(s->hist, 0, sizeof(s->hist));
    s->delay = delay;
    s->erl = erl;
}

static void scene_hop(Scene* s, float* mic, float* ref, int hop, int h) {
    int i;
    const int cap = (int)(sizeof(s->hist) / sizeof(s->hist[0]));
    /* Bursty far end: silence stretches are what make the delay estimator's
     * duty machine and the excitation gates actually change state. */
    float gain = ((h % 40) < 30) ? 0.4f : 0.0f;
    for (i = 0; i < hop; ++i) ref[i] = gain * rng_next(&s->rng);
    for (i = 0; i < hop; ++i) {
        int back = s->delay + (hop - 1 - i);
        float echo = (back < cap) ? s->hist[back] : 0.0f;
        /* Near-end talker on a slower cycle than the far-end bursts, so the
         * run spends time in single-talk and in double-talk. */
        float near = ((h % 130) < 35) ? 0.25f * rng_next(&s->rng) : 0.0f;
        mic[i] = s->erl * echo + near + 0.01f * rng_next(&s->rng);
    }
    /* Shift the far history back by one hop and append this one. */
    memmove(s->hist + hop, s->hist, (size_t)(cap - hop) * sizeof(float));
    for (i = 0; i < hop; ++i) s->hist[hop - 1 - i] = ref[i];
}

/* One (grid, mode) case. Returns 1 on byte identity. */
static int case_run(int sr, int fft_size, int ctx_only) {
    AecConfig cfg;
    Aec fresh, warmed;
    Scene sc_warm, sc_a, sc_b;
    float *mic, *ref, *out_a, *out_b, *formed_a;
    Complex *espec_a;
    int hop, n_freqs, h, ok = 1;
    long d_out = 0, d_formed = 0, d_espec = 0;
    int first_bad = -1;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
    cfg.fft_size = fft_size;
    if (ctx_only) { cfg.enable_res = 0; cfg.return_res_context = 1; }

    if (aec_create(&fresh, &cfg) != 0 || aec_create(&warmed, &cfg) != 0) {
        printf("FAIL: aec_create sr=%d fft=%d\n", sr, fft_size);
        g_fail++;
        return 0;
    }
    hop = aec_hop_size(&fresh);
    n_freqs = fft_size / 2 + 1;

    mic      = (float*)malloc((size_t)hop * sizeof(float));
    ref      = (float*)malloc((size_t)hop * sizeof(float));
    out_a    = (float*)malloc((size_t)hop * sizeof(float));
    out_b    = (float*)malloc((size_t)hop * sizeof(float));
    formed_a = (float*)malloc((size_t)hop * sizeof(float));
    espec_a  = (Complex*)malloc((size_t)n_freqs * sizeof(Complex));

    /* Warm the subject on an echo path the test phase does NOT reuse, so
     * anything it remembers is wrong for what follows. */
    scene_init(&sc_warm, 1234567u, 611, 0.65f);
    for (h = 0; h < WARM_HOPS; ++h) {
        scene_hop(&sc_warm, mic, ref, hop, h);
        if (ctx_only) aec_process_context(&warmed, mic, ref);
        else          aec_process(&warmed, mic, ref, out_b);
    }
    aec_reset(&warmed);

    /* Identical seeds: the two instances see the same bytes hop for hop. */
    scene_init(&sc_a, 0x89abcdefu, 293, 0.5f);
    scene_init(&sc_b, 0x89abcdefu, 293, 0.5f);
    for (h = 0; h < TEST_HOPS; ++h) {
        AecResContext ctx_a, ctx_b;

        scene_hop(&sc_a, mic, ref, hop, h);
        if (ctx_only) aec_process_context(&fresh, mic, ref);
        else          aec_process(&fresh, mic, ref, out_a);
        aec_get_res_context(&fresh, &ctx_a);
        /* The context pointers alias per-hop internals both instances
         * overwrite, so snapshot the reference side before driving the
         * subject. */
        memcpy(formed_a, ctx_a.formed_hop, (size_t)hop * sizeof(float));
        memcpy(espec_a, ctx_a.error_spec, (size_t)n_freqs * sizeof(Complex));

        scene_hop(&sc_b, mic, ref, hop, h);
        if (ctx_only) aec_process_context(&warmed, mic, ref);
        else          aec_process(&warmed, mic, ref, out_b);
        aec_get_res_context(&warmed, &ctx_b);

        if (!ctx_only && memcmp(out_a, out_b, (size_t)hop * sizeof(float))) {
            d_out++; if (first_bad < 0) first_bad = h;
        }
        if (memcmp(formed_a, ctx_b.formed_hop, (size_t)hop * sizeof(float))) {
            d_formed++; if (first_bad < 0) first_bad = h;
        }
        if (memcmp(espec_a, ctx_b.error_spec,
                   (size_t)n_freqs * sizeof(Complex))) {
            d_espec++; if (first_bad < 0) first_bad = h;
        }
    }

    if (d_out || d_formed || d_espec) ok = 0;
    printf("%s: sr=%-5d fft=%-4d %-9s out=%ld/%d formed=%ld/%d "
           "error_spec=%ld/%d first_bad_hop=%d\n",
           ok ? "pass" : "FAIL", sr, fft_size,
           ctx_only ? "ctx-only" : "full-res",
           d_out, TEST_HOPS, d_formed, TEST_HOPS, d_espec, TEST_HOPS,
           first_bad);
    if (ok) g_pass++; else g_fail++;

    aec_destroy(&fresh);
    aec_destroy(&warmed);
    free(mic); free(ref); free(out_a); free(out_b);
    free(formed_a); free(espec_a);
    return ok;
}

int main(void) {
    static const int grids[][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    int g, m;

    printf("aec_reset() fresh-instance parity: warm=%d hops, compare=%d hops\n",
           WARM_HOPS, TEST_HOPS);
    for (g = 0; g < (int)(sizeof(grids) / sizeof(grids[0])); ++g)
        for (m = 0; m < 2; ++m)
            case_run(grids[g][0], grids[g][1], m);

    printf("%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { printf(">>> PASS\n"); return 0; }
    printf(">>> FAIL\n");
    return 1;
}
