/* stream_sim.c — streaming API verification (no Python golden needed).
 *
 * Five checks, run on every supported grid (16 kHz/256, 16 kHz/512,
 * 48 kHz/1024 and the legacy 8 kHz/256):
 *   0. CAPACITY — fifo_cap_hops is the documented 320 ms of render hops,
 *      rounded up to a power of two (aec_derive_dims). It differs per grid:
 *      the hop is 8 ms at 16 kHz/256, so 320 ms is 40 hops and the ring is 64;
 *      the other grids land on 32. The later checks size their pushes from
 *      the instance's own fifo_cap_hops, so this is the check that pins the
 *      capacity rule itself.
 *   1. LOCKSTEP EQUIVALENCE — feeding the same (mic,ref) through aec_process()
 *      vs through aec_analyze_render()+aec_process_capture() in lockstep yields
 *      BYTE-IDENTICAL output. This is the bit-exact-preservation guarantee:
 *      the streaming wrapper does not alter the offline engine.
 *   2. OVERRUN (drop-new) — pushing > fifo_cap renders before any capture
 *      fires AEC_BUF_RENDER_OVERRUN on every render past capacity (exactly
 *      cap of them succeed, the rest are dropped) and does not crash /
 *      overflow.
 *   3. UNDERRUN — capturing with an empty FIFO fires AEC_BUF_RENDER_UNDERRUN,
 *      processes (silent render), and normal operation resumes afterwards.
 *   4. CONSUMER CATCH-UP — once the ring is completely full, the next
 *      capture skips straight to the freshest buffered hop (instead of
 *      draining the backlog oldest-first), reports AEC_BUF_RENDER_OVERRUN,
 *      heals the occupancy to <=1, and the following render+capture pair is
 *      back to NONE/NONE.
 *
 * Build and run (from c_impl/; the FFT wrapper lives in the shared
 * audiolib archive, which the Makefile resolves):
 *   make test-stream-sim
 *   make test-stream-sim SIMD=0
 */
#include "aec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The documented render-FIFO span (README "320 ms render FIFO"; aec.c
 * AEC_STREAM_FIFO_MS). Restated here on purpose: the test pins the rule
 * rather than echoing whatever the library computed. */
#define FIFO_SPAN_MS 320
/* Renders pushed past capacity in the overrun check. */
#define OVERRUN_EXTRA 68

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

static int next_pow2_int(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

/* Runs all checks on one grid; returns 0 on success, 1 on a check failure,
 * 2 on a setup failure. */
static int run_grid(int sr, int fft) {
    AecConfig cfg; aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
    cfg.fft_size = fft;
    printf("---- grid %d Hz / fft %d ----\n", sr, fft);

    Aec a_off, a_str;
    if (aec_create(&a_off, &cfg) || aec_create(&a_str, &cfg)) {
        fprintf(stderr, "aec_create failed (%d Hz / fft %d)\n", sr, fft);
        return 2;
    }
    const int hop = aec_hop_size(&a_off);
    const int n_hops = 400;
    float *mic = malloc(sizeof(float) * hop);
    float *ref = malloc(sizeof(float) * hop);
    float *out_off = malloc(sizeof(float) * hop);
    float *out_str = malloc(sizeof(float) * hop);
    int rc = 1;

    /* ---- Check 0: FIFO capacity rule ---- */
    {
        int want = (FIFO_SPAN_MS * sr / 1000 + hop - 1) / hop;
        if (want < 2) want = 2;
        want = next_pow2_int(want);
        const int got = a_str.fifo_cap_hops;
        const double span_ms = 1000.0 * (double)got * (double)hop / (double)sr;
        if (got != want || span_ms < (double)FIFO_SPAN_MS) {
            printf("FAIL: FIFO capacity — hop %d, got fifo_cap_hops=%d (%.1f ms),"
                   " expected %d (320 ms of hops rounded up to a power of two)\n",
                   hop, got, span_ms, want);
            goto done;
        }
        printf("PASS: FIFO capacity — hop %d samples, fifo_cap_hops=%d (%.1f ms)\n",
               hop, got, span_ms);
    }
    const int cap = a_str.fifo_cap_hops;

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
            printf("FAIL: unexpected event in lockstep h=%d (%d,%d)\n",
                   h, ev1, ev2);
            goto done;
        }
        if (memcmp(out_off, out_str, (size_t)hop * sizeof(float)) != 0)
            mismatches++;
    }
    if (mismatches) {
        printf("FAIL: lockstep equivalence — %ld/%d hops differ\n",
               mismatches, n_hops);
        goto done;
    }
    printf("PASS: lockstep equivalence — 0/%d hops differ (byte-exact)\n", n_hops);

    /* ---- Check 2: overrun (push > capacity renders before a capture) ---- */
    aec_reset(&a_str);
    /* The streaming FIFO is drop-new: the ring fills after the first `cap`
     * renders (fifo_write - fifo_read reaches cap with zero capture calls to
     * drain it), and every render after that is dropped, so pushing
     * cap + OVERRUN_EXTRA renders with no intervening capture must report
     * exactly OVERRUN_EXTRA overruns -- not just "at least one". `cap` is
     * the capacity Check 0 just verified (64 at 16 kHz/256, 32 elsewhere);
     * it is NOT a constant. */
    int overrun_count = 0;
    for (int h = 0; h < cap + OVERRUN_EXTRA; ++h) {
        gen_signal(mic, ref, hop);
        AecBufferingEvent ev = aec_analyze_render(&a_str, ref);
        if (h < cap) {
            if (ev != AEC_BUF_NONE) {
                printf("FAIL: overrun count — render %d expected NONE, got %d\n", h, ev);
                goto done;
            }
        } else {
            if (ev != AEC_BUF_RENDER_OVERRUN) {
                printf("FAIL: overrun count — render %d expected OVERRUN, got %d\n", h, ev);
                goto done;
            }
            overrun_count++;
        }
    }
    int overrun_seen = (overrun_count == OVERRUN_EXTRA);
    printf("%s: overrun count exactly %d/%d (first %d NONE, then %d OVERRUN)\n",
           overrun_seen ? "PASS" : "FAIL", overrun_count, cap + OVERRUN_EXTRA,
           cap, OVERRUN_EXTRA);
    if (!overrun_seen) goto done;

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
    if (!underrun_ok || !recovered) goto done;

    /* ---- Check 4: consumer catch-up ---- */
    aec_reset(&a_str);
    /* Fill the ring to exactly capacity with no intervening captures --
     * every one of these must report NONE (see Check 2's byte-exact count
     * of the first `cap` renders). */
    for (int h = 0; h < a_str.fifo_cap_hops; ++h) {
        gen_signal(mic, ref, hop);
        AecBufferingEvent ev = aec_analyze_render(&a_str, ref);
        if (ev != AEC_BUF_NONE) {
            printf("FAIL: catch-up setup — render %d expected NONE, got %d\n", h, ev);
            goto done;
        }
    }
    /* Single-threaded peek (no atomics needed -- this is the test harness,
     * not a second concurrent actor): occupancy must read exactly `cap`. */
    unsigned occ_before = *(unsigned*)&a_str.fifo_write - *(unsigned*)&a_str.fifo_read;
    if (occ_before != (unsigned)a_str.fifo_cap_hops) {
        printf("FAIL: catch-up setup — ring not full (occupancy=%u, cap=%d)\n",
               occ_before, a_str.fifo_cap_hops);
        goto done;
    }
    AecBufferingEvent catchup_ev = aec_process_capture(&a_str, mic, out_str);
    int catchup_ok = (catchup_ev == AEC_BUF_RENDER_OVERRUN);
    unsigned occ_after = *(unsigned*)&a_str.fifo_write - *(unsigned*)&a_str.fifo_read;
    int healed = (occ_after <= 1u);
    printf("%s: catch-up capture reports OVERRUN; %s: occupancy heals to <=1 (occ=%u)\n",
           catchup_ok ? "PASS" : "FAIL", healed ? "PASS" : "FAIL", occ_after);
    if (!catchup_ok || !healed) goto done;

    /* The very next render+capture pair must be back to a clean NONE/NONE
     * lockstep -- the catch-up must not leave any lingering skew. */
    gen_signal(mic, ref, hop);
    AecBufferingEvent rev2 = aec_analyze_render(&a_str, ref);
    AecBufferingEvent cev2 = aec_process_capture(&a_str, mic, out_str);
    int recovered2 = (rev2 == AEC_BUF_NONE && cev2 == AEC_BUF_NONE);
    printf("%s: post-catch-up render+capture pair is NONE/NONE\n",
           recovered2 ? "PASS" : "FAIL");
    if (!recovered2) goto done;
    rc = 0;

done:
    aec_destroy(&a_off); aec_destroy(&a_str);
    free(mic); free(ref); free(out_off); free(out_str);
    return rc;
}

int main(void) {
    static const struct { int sr, fft; } grids[] = {
        { 16000, 256 }, { 16000, 512 }, { 48000, 1024 }, { 8000, 256 },
    };
    int rc = 0;
    for (size_t i = 0; i < sizeof grids / sizeof grids[0]; ++i) {
        int r = run_grid(grids[i].sr, grids[i].fft);
        if (r) {
            printf("FAIL: grid %d Hz / fft %d (rc=%d)\n", grids[i].sr, grids[i].fft, r);
            rc = 1;
        }
    }
    if (rc) return rc;
    printf("\nALL STREAMING CHECKS PASSED\n");
    return 0;
}
