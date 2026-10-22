/* test_neon_twins.c -- NEON-vs-scalar equivalence gates for kernels that the
 * whole-output hash cannot see.
 *
 *  T1  rsa_argmax_first (render_signal_analyzer.c) against its scalar twin
 *      sk_argmax_f32_scalar: tie order (first maximum wins), maxima at every lane
 *      and tail position, non-finite inputs. Real audio almost never produces
 *      an exact tie in a power spectrum, so the output hash passes even with
 *      last-match-wins semantics.
 *  T2  decrease_erle_for_low_render (subband_erle.c) against a verbatim copy
 *      of its scalar loop: hold-counter values at, one either side of, the
 *      release threshold and zero, which decide the onset re-arm.
 *  T3  rsa_max_abs against sk_max_abs_f32_scalar (the sk_max_abs_f32 pair), bit
 *      for bit, at the call shape render_signal_analyzer.c uses.
 *
 * The shipped translation units are included so the real source text runs.
 * When the NEON body is not compiled in (non-aarch64 or
 * SIMD_KERNELS_FORCE_SCALAR) the twins coincide: the checks still run and the
 * summary says so. */
#include "src/render_signal_analyzer.c"
#include "src/subband_erle.c"

#include <stdio.h>
#include <stdlib.h>

#if SK_HAVE_NEON
#define HAS_NEON_BODY 1
#else
#define HAS_NEON_BODY 0
#endif

static unsigned long long g_rs = 0x243F6A8885A308D3ULL;
static unsigned long long rnd(void) {
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17;
    return g_rs;
}

static long g_cases, g_fails;

static unsigned bits_of(float v) { unsigned u; memcpy(&u, &v, 4); return u; }
static float float_of(unsigned u) { float v; memcpy(&v, &u, 4); return v; }

/* ---- T1 ---- */
#define XMAX 600

static void check_argmax(const char* what, const float* x, int n) {
    int a = rsa_argmax_first(x, n), b = sk_argmax_f32_scalar(x, n);
    g_cases++;
    if (a != b) {
        g_fails++;
        if (g_fails <= 12) printf("  MISMATCH argmax [%s] n=%d first=%d scan=%d\n", what, n, a, b);
    }
}

static void check_max_abs(const char* what, const float* x, int n) {
    float a = rsa_max_abs(x, n), b = sk_max_abs_f32_scalar(x, n);
    g_cases++;
    if (bits_of(a) != bits_of(b)) {
        g_fails++;
        if (g_fails <= 12)
            printf("  MISMATCH max_abs [%s] n=%d %08x vs %08x\n", what, n, bits_of(a), bits_of(b));
    }
}

static void t1_t3(void) {
    static const int big[] = {129, 257, 513};
    static float x[XMAX];
    static const float pool[] = {0.0f, -0.0f, 1.0f, 2.0f, 1.5f, -1.0f, 1e-40f, 3.0e38f};
    int idx, n, p, q, r, it;

    for (idx = 0; idx < 70 + 3; ++idx) {
        n = idx < 70 ? idx + 1 : big[idx - 70];
        /* constant arrays */
        for (r = 0; r < 4; ++r) {
            float v = r == 0 ? 0.0f : r == 1 ? 1.0f : r == 2 ? -1.0f : -0.0f;
            for (p = 0; p < n; ++p) x[p] = v;
            check_argmax("constant", x, n);
            check_max_abs("constant", x, n);
        }
        /* a single maximum at every position; and two equal maxima p < q */
        for (p = 0; p < n; ++p) {
            for (q = 0; q < n; ++q) x[q] = 0.25f + (float)(q % 7) * 0.01f;
            x[p] = 5.0f;
            check_argmax("single", x, n);
            check_max_abs("single", x, n);
            if (n <= 40) {
                for (q = p + 1; q < n; ++q) {
                    x[q] = 5.0f;
                    check_argmax("pair", x, n);
                    x[q] = 0.25f + (float)(q % 7) * 0.01f;
                }
            } else {
                for (r = 0; r < 6; ++r) {
                    q = (int)(rnd() % (unsigned long long)n);
                    if (q == p) continue;
                    x[q] = 5.0f;
                    check_argmax("pair", x, n);
                    x[q] = 0.25f + (float)(q % 7) * 0.01f;
                }
            }
        }
        /* maximum repeated at every other / every fourth / every eighth slot */
        for (r = 1; r <= 8; r <<= 1) {
            for (p = 0; p < n; ++p) x[p] = 1.0f;
            for (p = n - 1; p >= 0; p -= r) x[p] = 7.0f;
            check_argmax("stride-tail", x, n);
            for (p = 0; p < n; ++p) x[p] = (p % r) == (r - 1) ? 7.0f : 1.0f;
            check_argmax("stride-late", x, n);
        }
        /* non-finite */
        for (p = 0; p < n; ++p) x[p] = 1.0f + (float)(p % 5);
        for (p = 0; p < n; ++p) {
            float save = x[p];
            x[p] = float_of(0x7FC00000u);          /* quiet NaN */
            check_argmax("nan", x, n);
            check_max_abs("nan", x, n);
            x[p] = float_of(0x7F800000u);          /* +inf */
            check_argmax("inf", x, n);
            check_max_abs("inf", x, n);
            if (p + 1 < n) {
                float save2 = x[p + 1];
                x[p + 1] = float_of(0x7F800000u);  /* two infinities */
                check_argmax("inf-pair", x, n);
                x[p + 1] = save2;
            }
            x[p] = float_of(0xFF800000u);          /* -inf */
            check_argmax("-inf", x, n);
            check_max_abs("-inf", x, n);
            x[p] = save;
        }
        for (p = 0; p < n; ++p) x[p] = float_of(0xFF800000u);
        check_argmax("all -inf", x, n);
        /* random draws from a small pool: ties are the common case */
        for (it = 0; it < (n <= 70 ? 2500 : 600); ++it) {
            for (p = 0; p < n; ++p) {
                unsigned long long u = rnd();
                x[p] = (u & 0xFF) == 0 ? float_of((unsigned)(u >> 8))
                                       : pool[(u >> 8) % (sizeof pool / sizeof pool[0])];
            }
            check_argmax("pool", x, n);
            check_max_abs("pool", x, n);
        }
    }
}

/* ---- T2 ---- */
#define PAD 16

/* Verbatim copy of the scalar tail of decrease_erle_for_low_render in
 * src/subband_erle.c; keep in step with it. */
static void ref_decrease(SubbandErle* s) {
    int k, n = s->n_bins;
    for (k = 1; k <= n - 2; ++k) {
        s->hold_counters[k] -= 1;
        if (s->hold_counters[k] <=
            (s->blocks_for_onset_detection - s->blocks_to_hold_erle)) {
            if (s->erle_onset_compensated[k] > s->erle_during_onsets[k]) {
                float decayed = s->onset_release_decay *
                                s->erle_onset_compensated[k];
                float during = s->erle_during_onsets[k];
                s->erle_onset_compensated[k] =
                    decayed > during ? decayed : during;
            }
            if (s->hold_counters[k] <= 0) {
                s->coming_onset[k] = 1;
                s->hold_counters[k] = 0;
            }
        }
    }
}

typedef struct {
    SubbandErle s;
    float max_erle[XMAX + PAD], erle[XMAX + PAD], oc[XMAX + PAD], unb[XMAX + PAD];
    float during[XMAX + PAD], y2[XMAX + PAD], e2[XMAX + PAD];
    unsigned char coming[XMAX + PAD], low[XMAX + PAD];
    int32_t hold[XMAX + PAD];
} Inst;

static void inst_init(Inst* a, int n, int hop, int sr) {
    memset(a, 0xA5, sizeof *a);                        /* canary beyond n */
    subband_erle_init(&a->s, n, 1.0f, 4.0f, 1.5f, 1, 1, hop, sr,
                      a->max_erle, a->erle, a->oc, a->unb, a->during,
                      a->coming, a->hold, a->y2, a->e2, a->low);
}

static int inst_equal(const Inst* a, const Inst* b) {
    /* whole storage, canaries included */
    return memcmp(a->max_erle, b->max_erle, sizeof a->max_erle) == 0 &&
           memcmp(a->erle, b->erle, sizeof a->erle) == 0 &&
           memcmp(a->oc, b->oc, sizeof a->oc) == 0 &&
           memcmp(a->unb, b->unb, sizeof a->unb) == 0 &&
           memcmp(a->during, b->during, sizeof a->during) == 0 &&
           memcmp(a->coming, b->coming, sizeof a->coming) == 0 &&
           memcmp(a->hold, b->hold, sizeof a->hold) == 0;
}

static void t2(void) {
    static const int grids[][2] = {{128, 16000}, {256, 16000}, {512, 48000}};
    static const int sizes[] = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                                26, 31, 32, 33, 40, 129, 257, 513};
    static Inst A, B;
    int gi, si, it, k, rep;

    for (gi = 0; gi < 3; ++gi) {
        for (si = 0; si < (int)(sizeof sizes / sizeof sizes[0]); ++si) {
            int n = sizes[si];
            inst_init(&A, n, grids[gi][0], grids[gi][1]);
            {
                int thr = A.s.blocks_for_onset_detection - A.s.blocks_to_hold_erle;
                int onset = A.s.blocks_for_onset_detection;
                int cand[] = {thr + 3, thr + 2, thr + 1, thr, thr - 1, thr - 2, 3, 2, 1, 0, -1, -2,
                              onset, onset + 1};
                int ncand = (int)(sizeof cand / sizeof cand[0]);
                for (it = 0; it < 60; ++it) {
                    inst_init(&A, n, grids[gi][0], grids[gi][1]);
                    for (k = 0; k < n; ++k) {
                        unsigned long long u = rnd();
                        float du = 1.0f + (float)((u >> 3) % 8) * 0.25f;
                        int mode = (int)((u >> 8) % 6);
                        float oc = mode == 0 ? du : mode == 1 ? du * 1.01f
                                 : mode == 2 ? du * 1.5f : mode == 3 ? du * 0.5f : du;
                        if (mode == 4) du = A.s.onset_release_decay * oc;  /* decayed == during */
                        A.hold[k] = (it % 3 == 0) ? (int)((u >> 16) % (unsigned)(onset + 6)) - 4
                                                  : cand[(u >> 16) % (unsigned)ncand];
                        A.during[k] = du;
                        A.oc[k] = oc;
                        A.coming[k] = (unsigned char)((u >> 40) & 1);
                    }
                    B = A;
                    A.s.max_erle = A.max_erle; A.s.erle = A.erle; A.s.erle_onset_compensated = A.oc;
                    A.s.erle_unbounded = A.unb; A.s.erle_during_onsets = A.during;
                    A.s.coming_onset = A.coming; A.s.hold_counters = A.hold;
                    A.s.y2_acc = A.y2; A.s.e2_acc = A.e2; A.s.low_render_energy = A.low;
                    B.s.max_erle = B.max_erle; B.s.erle = B.erle; B.s.erle_onset_compensated = B.oc;
                    B.s.erle_unbounded = B.unb; B.s.erle_during_onsets = B.during;
                    B.s.coming_onset = B.coming; B.s.hold_counters = B.hold;
                    B.s.y2_acc = B.y2; B.s.e2_acc = B.e2; B.s.low_render_energy = B.low;
                    for (rep = 0; rep < 3; ++rep) {
                        decrease_erle_for_low_render(&A.s);
                        ref_decrease(&B.s);
                        g_cases++;
                        if (!inst_equal(&A, &B)) {
                            g_fails++;
                            if (g_fails <= 12)
                                printf("  MISMATCH decrease hop=%d sr=%d n=%d it=%d rep=%d\n",
                                       grids[gi][0], grids[gi][1], n, it, rep);
                            break;
                        }
                    }
                }
            }
        }
    }
}

int main(void) {
    long c0;
    printf("test_neon_twins: NEON body %s\n", HAS_NEON_BODY ? "compiled in" : "absent (twins coincide)");
    t1_t3();
    c0 = g_cases;
    printf("  T1/T3 argmax + max_abs: %ld cases, %ld failures so far\n", c0, g_fails);
    t2();
    printf("  T2 decrease_erle_for_low_render: %ld cases\n", g_cases - c0);
    if (g_fails) { printf("FAIL: %ld mismatches\n", g_fails); return 1; }
    printf("PASS: test_neon_twins (%ld cases)\n", g_cases);
    return 0;
}
