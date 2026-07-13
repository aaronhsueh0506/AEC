/* stream_sim.c — streaming API verification (no Python golden needed).
 *
 * Three checks:
 *   1. LOCKSTEP EQUIVALENCE — feeding the same (mic,ref) through aec_process()
 *      vs through aec_analyze_render()+aec_process_capture() in lockstep yields
 *      BYTE-IDENTICAL output. This is the bit-exact-preservation guarantee:
 *      the streaming wrapper does not alter the offline engine.
 *   2. OVERRUN — pushing > fifo_cap renders before any capture fires
 *      AEC_BUF_RENDER_OVERRUN and does not crash / overflow.
 *   3. UNDERRUN — capturing with an empty FIFO fires AEC_BUF_RENDER_UNDERRUN,
 *      processes (silent render), and normal operation resumes afterwards.
 *
 * Build (from c_impl/); the FFT wrapper now lives in the shared audio_common
 * archive:
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -I../../audio_common/include \
 *       $(find src -name '*.c') test/stream_sim.c \
 *       ../../audio_common/bin/kiss/libaudio_common.a -lm -o /tmp/stream_sim
 *   /tmp/stream_sim
 */
#include "aec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Deterministic LCG — no Math.random / time. */
static unsigned int g_lcg = 0x12345678u;
static float frnd(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return ((float)(g_lcg >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;  /* [-1,1) */
}

static void gen_signal(float *mic, float *ref, int n) {
    /* ref = tone+noise; mic = attenuated delayed ref (echo) + small near. */
    const int dly = 96;
    static float hist[4096];
    static int hp = 0;
    for (int i = 0; i < n; ++i) {
        float r = 0.6f * sinf(2.0f * 3.14159265f * 300.0f * (float)i / 16000.0f)
                  + 0.2f * frnd();
        ref[i] = r;
        hist[hp & 4095] = r; hp++;
        float echo = 0.5f * hist[(hp - 1 - dly) & 4095];
        float near = 0.05f * frnd();
        mic[i] = echo + near;
    }
}

int main(void) {
    const int sr = 16000;
    AecConfig cfg; aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);

    Aec a_off, a_str;
    if (aec_create(&a_off, &cfg) || aec_create(&a_str, &cfg)) {
        fprintf(stderr, "aec_create failed\n"); return 2;
    }
    const int hop = aec_hop_size(&a_off);
    const int n_hops = 400;
    float *mic = malloc(sizeof(float) * hop);
    float *ref = malloc(sizeof(float) * hop);
    float *out_off = malloc(sizeof(float) * hop);
    float *out_str = malloc(sizeof(float) * hop);

    /* ---- Check 1: lockstep equivalence ---- */
    long mismatches = 0;
    g_lcg = 0x12345678u; /* reset both signal streams identically per instance */
    /* We must drive both instances with the SAME per-hop data. Generate once
     * into buffers, feed both. Use a fresh LCG so the sequence is reproducible. */
    for (int h = 0; h < n_hops; ++h) {
        gen_signal(mic, ref, hop);
        aec_process(&a_off, mic, ref, out_off);                 /* offline */
        AecBufferingEvent ev1 = aec_analyze_render(&a_str, ref); /* streaming */
        AecBufferingEvent ev2 = aec_process_capture(&a_str, mic, out_str);
        if (ev1 != AEC_BUF_NONE || ev2 != AEC_BUF_NONE) {
            fprintf(stderr, "FAIL: unexpected event in lockstep h=%d (%d,%d)\n",
                    h, ev1, ev2);
            return 1;
        }
        if (memcmp(out_off, out_str, (size_t)hop * sizeof(float)) != 0)
            mismatches++;
    }
    if (mismatches) {
        printf("FAIL: lockstep equivalence — %ld/%d hops differ\n",
               mismatches, n_hops);
        return 1;
    }
    printf("PASS: lockstep equivalence — 0/%d hops differ (byte-exact)\n", n_hops);

    /* ---- Check 2: overrun (push > capacity renders before a capture) ---- */
    aec_reset(&a_str);
    int overrun_seen = 0;
    for (int h = 0; h < 100; ++h) {           /* fifo_cap is 64 */
        gen_signal(mic, ref, hop);
        if (aec_analyze_render(&a_str, ref) == AEC_BUF_RENDER_OVERRUN)
            overrun_seen = 1;
    }
    printf("%s: overrun detected after > capacity renders\n",
           overrun_seen ? "PASS" : "FAIL");
    if (!overrun_seen) return 1;

    /* ---- Check 3: underrun (capture with empty FIFO) + recovery ---- */
    aec_reset(&a_str);
    gen_signal(mic, ref, hop);
    AecBufferingEvent uev = aec_process_capture(&a_str, mic, out_str); /* empty */
    int underrun_ok = (uev == AEC_BUF_RENDER_UNDERRUN);
    /* recovery: a normal render+capture must now report no event */
    gen_signal(mic, ref, hop);
    aec_analyze_render(&a_str, ref);
    AecBufferingEvent rev = aec_process_capture(&a_str, mic, out_str);
    int recovered = (rev == AEC_BUF_NONE);
    printf("%s: underrun detected on empty FIFO; %s: recovered to NONE\n",
           underrun_ok ? "PASS" : "FAIL", recovered ? "PASS" : "FAIL");
    if (!underrun_ok || !recovered) return 1;

    aec_destroy(&a_off); aec_destroy(&a_str);
    free(mic); free(ref); free(out_off); free(out_str);
    printf("\nALL STREAMING CHECKS PASSED\n");
    return 0;
}
