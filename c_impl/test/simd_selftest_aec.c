/* Shared AEC kernel correctness and microbenchmark harness (utility/simd_aec.h).
 * Finite scalar/SIMD results and mutated state must match byte-for-byte.
 * The NaN corpus also checks classification; payload differences are allowed
 * only where check_bits_classify documents that contract. All hard mismatches
 * produce a nonzero exit. Generic kernels have their own audiolib self-test.
 */
#include "utility/simd_aec.h"
#include "utility/simd_test_mode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════════ config ═══════════════════════════════ */

#define SK_TEST_MAX_N 512
/* Includes n=0 (a zero-read/zero-write no-op, see the canary section below) and the complete 1..17
 * run, so every kernel's 4-lane NEON/scalar-tail boundary is crossed at every possible remainder,
 * plus hand-picked lane/leaf/split boundary values. */
static const int N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    128, 129, 160, 255, 256, 257, 512
};
#define N_LIST_COUNT ((int)(sizeof(N_LIST) / sizeof(N_LIST[0])))
#define TRIALS 8

#define BENCH_N 257
#define BENCH_REPS 200000

/* ═══════════════════════ input generation ═══════════════════════
 * Deterministic LCG over raw uint32 bit patterns mapped to floats, mixed with a curated
 * special-value pool (the special pool is drawn 25% of the time, see gen_float()). NaN bit patterns
 * from the raw-bits path are remapped to 0.0f: NaN payload propagation is outside the bit-exactness
 * contract. */

static uint32_t g_lcg = 0x9E3779B9u;

static uint32_t lcg_next(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return g_lcg;
}

static float bits_to_float(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

#define SPECIAL_POOL_COUNT 12
static float special_pool[SPECIAL_POOL_COUNT];

static void init_special_pool(void) {
    special_pool[0]  = 0.0f;
    special_pool[1]  = -0.0f;
    special_pool[2]  = FLT_MIN;
    special_pool[3]  = bits_to_float(0x00000001u); /* smallest subnormal */
    special_pool[4]  = bits_to_float(0x007FFFFFu); /* largest subnormal */
    special_pool[5]  = FLT_MAX;
    special_pool[6]  = 1.0f;
    special_pool[7]  = -1.0f;
    special_pool[8]  = 1e-30f;
    special_pool[9]  = 3e38f;
    special_pool[10] = (float)INFINITY;
    special_pool[11] = -(float)INFINITY;
}

static float gen_float(void) {
    uint32_t r = lcg_next();
    if ((r & 3u) == 0u) {
        uint32_t idx = (lcg_next() >> 8) % SPECIAL_POOL_COUNT;
        return special_pool[idx];
    } else {
        uint32_t bits = lcg_next();
        float f = bits_to_float(bits);
        if (f != f) f = 0.0f; /* exclude NaN */
        return f;
    }
}

static void gen_complex(Complex *c) {
    c->r = gen_float();
    c->i = gen_float();
}

static void fill_floats(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = gen_float();
}

static void fill_complex(Complex *a, int n) {
    int i;
    for (i = 0; i < n; ++i) gen_complex(&a[i]);
}

/* ═══════════════════════ int32 input generation (kernels 24-26) ═══════════════════════
 * hold_counters is int32: a separate generator from gen_float() (an int32 has no NaN/Inf, so no
 * exclusion logic is needed), with the same 25% special-pool / 75% raw-bits shape. Kernel 25
 * (sk_dec1_floorintmin_s32) decrements only while the element is > INT_MIN: an unconditional `-= 1`
 * on INT32_MIN is signed-overflow UB in C, and flooring at 0 would desync the raw value from
 * Python's unbounded counter (see that kernel's header comment in utility/simd_aec.h). INT32_MIN is
 * a safe input (the kernel leaves it unchanged), so it stays in both the special pool and the
 * raw-bits path; INT32_MIN+1 is also in the pool, to exercise the one lane where the NEON
 * speculative decrement fires the hop before the floor. */
#define SPECIAL_INT_POOL_COUNT 10
static int32_t special_int_pool[SPECIAL_INT_POOL_COUNT];

static void init_special_int_pool(void) {
    special_int_pool[0] = 0;
    special_int_pool[1] = 1;
    special_int_pool[2] = -1;
    special_int_pool[3] = 400;    /* typical hold_hops */
    special_int_pool[4] = -400;
    special_int_pool[5] = 2;
    special_int_pool[6] = -2;
    special_int_pool[7] = INT32_MAX;
    special_int_pool[8] = INT32_MIN;
    special_int_pool[9] = INT32_MIN + 1;
}

static int32_t gen_int32(void) {
    uint32_t r = lcg_next();
    if ((r & 3u) == 0u) {
        uint32_t idx = (lcg_next() >> 8) % SPECIAL_INT_POOL_COUNT;
        return special_int_pool[idx];
    } else {
        uint32_t bits = lcg_next();
        int32_t v;
        memcpy(&v, &bits, sizeof v);
        return v;   /* INT32_MIN included: safe under kernel 25's
                     * floor-at-INT_MIN contract (see the corpus comment above). */
    }
}

static void fill_ints(int *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = (int)gen_int32();
}

/* Separate, moderate-range generator for the microbenchmarks only -- keeps
 * the timing loops away from Inf/NaN-producing accumulation artifacts so
 * the reported numbers reflect typical-case throughput. Not used by any
 * correctness check. */
static float gen_bench_float(void) {
    uint32_t r = lcg_next();
    return ((float)(r % 2000001u) / 1000000.0f) - 1.0f; /* ~[-1, 1] */
}

static void fill_bench_floats(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) a[i] = gen_bench_float();
}

static void fill_bench_complex(Complex *a, int n) {
    int i;
    for (i = 0; i < n; ++i) { a[i].r = gen_bench_float(); a[i].i = gen_bench_float(); }
}

/* ═══════════════════════ NaN corpus ═══════════════════════
 * NaN-carrying input generation, separate from gen_float()'s special_pool (which excludes NaN). Two
 * uses: (1) the cabs_np/cmag2_np family (kernels 1/2/3/5, sharing the sk__cabs_np_neon4 helper):
 * full pattern x length sweep with STRICT bitwise memcmp (check_bits_or_die), gating the
 * compare/select NaN contract of that helper; (2) every other kernel: one NaN-sprinkled run per
 * kernel, checked with check_bits_classify (bit-exact / both-NaN-payload-unspecified / HARD FAIL).
 * Those kernels avoid vmaxq_f32/vminq_f32/vabsq_f32 (see the header's "NaN semantics" note), so a
 * HARD FAIL would be a finding outside the cabs family; a both-NaN classification is the
 * in-contract outcome for a multi-NaN-operand reduction whose scalar and NEON orders tie-break
 * payloads differently (see check_bits_classify). */

#define NAN_POOL_COUNT 6
static float nan_pool[NAN_POOL_COUNT];

static void init_nan_pool(void) {
    nan_pool[0] = bits_to_float(0x7fc12345u); /* qNaN, custom payload (the
                                                * repro pattern) */
    nan_pool[1] = bits_to_float(0x7fc00000u); /* qNaN, default payload */
    nan_pool[2] = bits_to_float(0x7fa00001u); /* sNaN */
    nan_pool[3] = bits_to_float(0xffc00000u); /* -NaN */
    nan_pool[4] = (float)INFINITY;
    nan_pool[5] = -(float)INFINITY;
}

/* n=1..17 straddles every kernel's 4-lane NEON/scalar-tail boundary many times over;
 * 128/129/160/255/256/257 straddle the pairwise-sum leaf/split cutovers (kernel 13/14's n<=128
 * leaf; kernels 21/22 share the n<=257 cutover, and their >257 recursion is covered by
 * PW_TAILFOLD_N_LIST). */
/* n=0 prepended: a NaN-sprinkled fill over zero elements is a no-op, the same zero-touch contract
 * as the finite corpus's n=0 case. */
static const int NAN_N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    128, 129, 160, 255, 256, 257
};
#define NAN_N_LIST_COUNT ((int)(sizeof(NAN_N_LIST) / sizeof(NAN_N_LIST[0])))

typedef enum {
    NANPAT_RE_ONLY = 0,     /* NaN in re only, im stays finite */
    NANPAT_IM_ONLY,         /* NaN in im only, re stays finite */
    NANPAT_INTERLEAVE3,     /* every 3rd element: BOTH re and im NaN */
    NANPAT_INF_NAN_MIX      /* alternating +-Inf paired with NaN */
} nan_pattern_t;

static const nan_pattern_t NAN_PATTERNS[] = {
    NANPAT_RE_ONLY, NANPAT_IM_ONLY, NANPAT_INTERLEAVE3, NANPAT_INF_NAN_MIX
};
#define NAN_PATTERN_COUNT ((int)(sizeof(NAN_PATTERNS) / sizeof(NAN_PATTERNS[0])))
static const char *NAN_PATTERN_NAMES[] = {
    "re_only", "im_only", "interleave3", "inf_nan_mix"
};

static void fill_complex_nan(Complex *z, int n, nan_pattern_t pat) {
    int i;
    for (i = 0; i < n; ++i) {
        switch (pat) {
        case NANPAT_RE_ONLY:
            z[i].r = nan_pool[i % NAN_POOL_COUNT];
            z[i].i = gen_float();
            break;
        case NANPAT_IM_ONLY:
            z[i].r = gen_float();
            z[i].i = nan_pool[i % NAN_POOL_COUNT];
            break;
        case NANPAT_INTERLEAVE3:
            if (i % 3 == 0) {
                z[i].r = nan_pool[i % NAN_POOL_COUNT];
                z[i].i = nan_pool[(i / 3 + 1) % NAN_POOL_COUNT];
            } else {
                z[i].r = gen_float();
                z[i].i = gen_float();
            }
            break;
        case NANPAT_INF_NAN_MIX:
        default:
            if (i & 1) {
                z[i].r = (i & 2) ? (float)INFINITY : -(float)INFINITY;
                z[i].i = nan_pool[i % NAN_POOL_COUNT];
            } else {
                z[i].r = nan_pool[i % NAN_POOL_COUNT];
                z[i].i = (i & 2) ? -(float)INFINITY : (float)INFINITY;
            }
            break;
        }
    }
}

/* Real-array NaN sprinkle (every 3rd element replaced by a NaN-pool value,
 * cycling qNaN/sNaN/-NaN/+-Inf), rest finite via gen_float(). Used by the
 * "every other kernel" sweep. */
static void fill_floats_nan_sprinkle(float *a, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        if (i % 3 == 0) a[i] = nan_pool[i % NAN_POOL_COUNT];
        else a[i] = gen_float();
    }
}

/* Complex-array NaN sprinkle: every 3rd element both-NaN, every (3rd+1)
 * element re-only NaN, remaining third left finite. */
static void fill_complex_nan_sprinkle(Complex *z, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        int m = i % 3;
        if (m == 0) {
            z[i].r = nan_pool[i % NAN_POOL_COUNT];
            z[i].i = nan_pool[(i + 1) % NAN_POOL_COUNT];
        } else if (m == 1) {
            z[i].r = nan_pool[i % NAN_POOL_COUNT];
            z[i].i = gen_float();
        } else {
            z[i].r = gen_float();
            z[i].i = gen_float();
        }
    }
}

/* ═══════════════════════ mismatch reporting ═══════════════════════
 * g_total_checks counts the individual bit-pattern comparisons performed, incremented at the
 * chokepoints every check funnels through (check_bits_or_die, check_scalar_bits_or_die,
 * check_bits_classify, check_mask_classify and the canary checks), so main()'s printout is a
 * reproducible measure of how much the run verified. */

static long g_total_checks = 0;

static int first_diff_bits(const float *a, const float *b, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        uint32_t ba, bb;
        memcpy(&ba, &a[i], sizeof ba);
        memcpy(&bb, &b[i], sizeof bb);
        if (ba != bb) return i;
    }
    return -1;
}

static void check_bits_or_die(const char *kernel, int n, int trial,
                               const float *simd, const float *scalar, int count) {
    int idx;
    g_total_checks += count;
    idx = first_diff_bits(simd, scalar, count);
    if (idx >= 0) {
        uint32_t gb, wb;
        memcpy(&gb, &simd[idx], sizeof gb);
        memcpy(&wb, &scalar[idx], sizeof wb);
        fprintf(stderr,
            "MISMATCH kernel=%s n=%d trial=%d idx=%d simd=0x%08x (%.9g) scalar=0x%08x (%.9g)\n",
            kernel, n, trial, idx, (unsigned)gb, (double)simd[idx], (unsigned)wb, (double)scalar[idx]);
        exit(1);
    }
}

static void check_scalar_bits_or_die(const char *kernel, int n, int trial,
                                      float simd_val, float scalar_val) {
    uint32_t gb, wb;
    g_total_checks++;
    memcpy(&gb, &simd_val, sizeof gb);
    memcpy(&wb, &scalar_val, sizeof wb);
    if (gb != wb) {
        fprintf(stderr,
            "MISMATCH kernel=%s n=%d trial=%d idx=0 simd=0x%08x (%.9g) scalar=0x%08x (%.9g)\n",
            kernel, n, trial, (unsigned)gb, (double)simd_val, (unsigned)wb, (double)scalar_val);
        exit(1);
    }
}

/* int32 counterpart of check_bits_or_die: hold_counters (kernels 24/25/26) have no NaN/payload
 * ambiguity, so the comparison is always strict bit-for-bit equality, with no classify escape
 * hatch. */
static void check_ints_or_die(const char *kernel, int n, int trial,
                               const int *simd, const int *scalar, int count) {
    int i;
    g_total_checks += count;
    for (i = 0; i < count; ++i) {
        if (simd[i] != scalar[i]) {
            fprintf(stderr,
                "MISMATCH kernel=%s n=%d trial=%d idx=%d simd=%d scalar=%d\n",
                kernel, n, trial, i, simd[i], scalar[i]);
            exit(1);
        }
    }
}

/* ═════════════════════ NaN classification gate ════════════════════════════
 * A pass/fail gate for the "every other kernel" NaN sweep. Each scalar-vs-NEON element comparison lands in exactly one bucket:
 *
 *   1. bit-exact  -- identical bits (the expected outcome for nearly all elements).
 *   2. both-NaN   -- bits differ, but BOTH values are NaN (tested via f32_is_nan(), an exponent/mantissa test, not `v != v`).
 *      This is IN CONTRACT: per utility/simd_aec.h and simd_kernels.h, the arithmetic kernels here (wupdate_nlms/wupdate_kf/
 *      ema_delta/n2_track/n2_initial_track/mask_zero and the four pairwise-sum-family reductions) promise "NaN in -> NaN
 *      out, PAYLOAD UNSPECIFIED". C leaves multi-NaN payload selection implementation-defined, so the scalar and NEON
 *      evaluation orders may surface different payloads; both sides still signal "invalid result". Tallied, not fatal.
 *   3. HARD FAIL  -- anything else: exactly one side NaN and the other finite/Inf (the class of bug the cabs_np/cmag2_np
 *      compare/select form exists to avoid), or both sides finite/Inf with different bits (an ordinary bit-exactness
 *      regression found through the NaN corpus). main() returns nonzero iff any kernel's HARD FAIL count is nonzero.
 */

typedef struct {
    char name[64];
    long bitexact;
    long both_nan;
    long fail;
} kernel_tally_t;

/* One tally row per distinct kernel name. The NaN corpus registers 15 names and the edge-case
 * matrix registers the same 15 again under a "_edge" suffix, so the table holds at least 30 rows;
 * running out is a FATAL ("kernel tally table full"), not a silent drop. */
#define MAX_TALLY_KERNELS 128
static kernel_tally_t g_tally[MAX_TALLY_KERNELS];
static int g_tally_count = 0;
static int g_hard_fail_count = 0;
/* Running total (both-NaN + HARD FAIL combined) used only by the per-kernel "(soft mismatches so
 * far: N)" progress prints; the real gate is g_hard_fail_count. */
static int g_nan_soft_mismatch_count = 0;

/* NaN test via the exponent/mantissa bit pattern, not the `v != v` idiom: classification must treat
 * a signaling NaN like a quiet NaN, and on some platforms an sNaN can be quieted by loading it into
 * a register before a C comparison runs, which would make `v != v` unreliable. Testing the raw bits
 * sidesteps that. */
static int f32_is_nan(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof b);
    return ((b & 0x7F800000u) == 0x7F800000u) && ((b & 0x007FFFFFu) != 0u);
}

static kernel_tally_t *get_tally(const char *name) {
    int i;
    for (i = 0; i < g_tally_count; ++i) {
        if (strcmp(g_tally[i].name, name) == 0) return &g_tally[i];
    }
    if (g_tally_count >= MAX_TALLY_KERNELS) {
        fprintf(stderr, "FATAL: kernel tally table full (raise MAX_TALLY_KERNELS)\n");
        exit(1);
    }
    {
        kernel_tally_t *t = &g_tally[g_tally_count++];
        memset(t, 0, sizeof *t);
        strncpy(t->name, name, sizeof(t->name) - 1);
        return t;
    }
}

/* Classifies and tallies every element of a float buffer pair (see the header comment above for the
 * three-way bit-exact/both-NaN/HARD-FAIL split). Only a HARD FAIL affects main()'s exit code;
 * both-NaN divergences are printed for auditability but excluded from the gate. */
static void check_bits_classify(const char *kernel, int n, int trial,
                                 const float *simd, const float *scalar, int count) {
    kernel_tally_t *t = get_tally(kernel);
    int i;
    g_total_checks += count;
    for (i = 0; i < count; ++i) {
        uint32_t bs, bc;
        memcpy(&bs, &simd[i], sizeof bs);
        memcpy(&bc, &scalar[i], sizeof bc);
        if (bs == bc) { t->bitexact++; continue; }
        if (f32_is_nan(simd[i]) && f32_is_nan(scalar[i])) {
            t->both_nan++;
            g_nan_soft_mismatch_count++;
            fprintf(stderr,
                "BOTH-NAN kernel=%s n=%d trial=%d idx=%d simd=0x%08x scalar=0x%08x "
                "(both NaN, differing payload -- in contract, not a HARD FAIL)\n",
                kernel, n, trial, i, (unsigned)bs, (unsigned)bc);
            continue;
        }
        t->fail++;
        g_nan_soft_mismatch_count++;
        g_hard_fail_count++;
        fprintf(stderr,
            "HARD-FAIL kernel=%s n=%d trial=%d idx=%d simd=0x%08x (%.9g) scalar=0x%08x (%.9g) "
            "-- contract violation: finite-vs-NaN divergence or unexplained bit mismatch\n",
            kernel, n, trial, i, (unsigned)bs, (double)simd[i], (unsigned)bc, (double)scalar[i]);
    }
}

/* Byte/mask counterpart, for the coherence-gate's derived 0/1 decision byte.
 * A mask bit has no "both-NaN, payload unspecified" escape hatch -- there is
 * no such thing as a NaN mask byte, the value is either 0 or 1 -- so ANY
 * divergence here is a HARD FAIL by construction: a real behavioral
 * difference in which frequency bins get gated, not an artifact this
 * classifier can excuse. */
static void check_mask_classify(const char *kernel, int n, int idx,
                                 unsigned char simd_val, unsigned char scalar_val) {
    kernel_tally_t *t = get_tally(kernel);
    g_total_checks++;
    if (simd_val == scalar_val) { t->bitexact++; return; }
    t->fail++;
    g_nan_soft_mismatch_count++;
    g_hard_fail_count++;
    fprintf(stderr,
        "HARD-FAIL kernel=%s n=%d idx=%d simd=%u scalar=%u -- mask decision diverged\n",
        kernel, n, idx, (unsigned)simd_val, (unsigned)scalar_val);
}

static void print_classification_summary(void) {
    int i;
    printf("\n--- NaN classification gate summary ---\n");
    printf("%-42s %10s %10s %10s\n", "kernel", "bitexact", "both-nan", "FAIL");
    for (i = 0; i < g_tally_count; ++i) {
        printf("%-42s %10ld %10ld %10ld\n",
               g_tally[i].name, g_tally[i].bitexact, g_tally[i].both_nan, g_tally[i].fail);
    }
    printf("TOTAL: hard fails=%d\n", g_hard_fail_count);
}

/* ═══════════════════════════ correctness: kernel 1 ═══════════════════════ */

static void test_cabs_np(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            sk_cabs_np_f32_scalar(z, out_scalar, n);
            sk_cabs_np_f32(z, out_simd, n);
            check_bits_or_die("cabs_np_f32", n, t, out_simd, out_scalar, n);
        }
    }
    printf("PASS cabs_np_f32\n");
}


/* ═══════════════════════════ correctness: kernel 2 ═══════════════════════ */

static void test_cmag2_np(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            sk_cmag2_np_f32_scalar(z, out_scalar, n);
            sk_cmag2_np_f32(z, out_simd, n);
            check_bits_or_die("cmag2_np_f32", n, t, out_simd, out_scalar, n);
        }
    }
    printf("PASS cmag2_np_f32\n");
}


/* ═══════════════════════════ correctness: kernel 3 ═══════════════════════ */

static void test_cmag2_np_acc(void) {
    Complex z[SK_TEST_MAX_N];
    float acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            fill_floats(acc_init, n);
            memcpy(acc_scalar, acc_init, (size_t)n * sizeof(float));
            memcpy(acc_simd, acc_init, (size_t)n * sizeof(float));
            sk_cmag2_np_acc_f32_scalar(z, acc_scalar, n);
            sk_cmag2_np_acc_f32(z, acc_simd, n);
            check_bits_or_die("cmag2_np_acc_f32", n, t, acc_simd, acc_scalar, n);
        }
    }
    printf("PASS cmag2_np_acc_f32\n");
}


/* ═══════════════════════ correctness: ring cmac ═══════════════════════════ */

static void test_cmac_np_ring(void) {
    enum { MAX_PART = 9, MAX_K = 130 };
    static Complex W[MAX_PART * MAX_K], X[MAX_PART * MAX_K];
    static Complex acc_init[MAX_K], acc_scalar[MAX_K], acc_simd[MAX_K];
    const int parts[] = {1, 2, 4, 6, 7, 9};
    const int ks[] = {1, 3, 4, 5, 8, 17, 64, 129};
    int pi, ki, t, c;
    for (pi = 0; pi < 6; ++pi) {
        for (ki = 0; ki < 8; ++ki) {
            int n_part = parts[pi], K = ks[ki];
            for (t = 0; t < TRIALS; ++t) {
                fill_complex(W, n_part * K);
                fill_complex(X, n_part * K);
                fill_complex(acc_init, K);
                for (c = 0; c < n_part; ++c) {
                    memcpy(acc_scalar, acc_init, (size_t)K * sizeof(Complex));
                    memcpy(acc_simd, acc_init, (size_t)K * sizeof(Complex));
                    sk_cmac_fma_ring_f32_scalar(acc_scalar, W, X, c, n_part, K);
                    sk_cmac_fma_ring_f32(acc_simd, W, X, c, n_part, K);
                    check_bits_classify("cmac_np_ring_f32", K, t, (const float *)acc_simd,
                                        (const float *)acc_scalar, 2 * K);
                }
            }
        }
    }
    printf("PASS cmac_np_ring_f32\n");
}


/* ═══════════════════════════ correctness: kernel 7 ═══════════════════════ */

static void test_wupdate_nlms(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu_eff[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(X, n);
            fill_complex(err, n);
            fill_complex(W_init, n);
            fill_floats(mu_eff, n);
            memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
            memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
            sk_wupdate_nlms_f32_scalar(W_scalar, X, err, mu_eff, n);
            sk_wupdate_nlms_f32(W_simd, X, err, mu_eff, n);
            check_bits_classify("wupdate_nlms_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
        }
    }
    printf("PASS wupdate_nlms_f32\n");
}


/* ═══════════════════════════ correctness: kernel 8 ═══════════════════════ */

static void test_wupdate_kf(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu[SK_TEST_MAX_N], mu_scale[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(X, n);
            fill_complex(err, n);
            fill_complex(W_init, n);
            fill_floats(mu, n);
            fill_floats(mu_scale, n);
            memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
            memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
            sk_wupdate_kf_f32_scalar(W_scalar, X, err, mu, mu_scale, n);
            sk_wupdate_kf_f32(W_simd, X, err, mu, mu_scale, n);
            check_bits_classify("wupdate_kf_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
        }
    }
    printf("PASS wupdate_kf_f32\n");
}


/* ═══════════════════════════ correctness: kernel 13 ══════════════════════ */

static void test_pairwise_sum(void) {
    float a[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_f32(a, (size_t)n);
                check_bits_classify("pairwise_sum_f32", n, t, &rn, &rs, 1);
            }
        }
    }
    printf("PASS pairwise_sum_f32\n");
}


/* ═══════════════════════════ correctness: kernel 14 ══════════════════════ */

static void test_sum_sq_pairwise(void) {
    float a[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_sum_sq_pairwise_f32_scalar(a, (size_t)n);
                float rn = sk_sum_sq_pairwise_f32(a, (size_t)n);
                check_bits_classify("sum_sq_pairwise_f32", n, t, &rn, &rs, 1);
            }
        }
    }
    printf("PASS sum_sq_pairwise_f32\n");
}


/* ═══════════════════ correctness: kernels 21/22 (tail-fold pairwise) ══════
 * Dedicated n-list covering the leaf/split boundaries specific to these two
 * kernels' recursion (127/128/129 straddle the n<=128 leaf cutover; 960
 * exercises >=2 levels of the half-rounded-to-a-multiple-of-8 split; 7/8/9
 * straddle the small-n vs. leaf cutover that differs between kernel 21 and
 * kernel 13 -- see kernel 21's header comment). Separate, larger backing
 * buffer (960) since this exceeds SK_TEST_MAX_N (512), used only here. */

/* 0 and the complete 1..17 run prepended (same rationale as N_LIST above), plus the
 * boundary-specific values. */
#define PW_TAILFOLD_MAX_N 960
static const int PW_TAILFOLD_N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    127, 128, 129, 160, 255, 256, 257, 512, 960
};
#define PW_TAILFOLD_N_LIST_COUNT ((int)(sizeof(PW_TAILFOLD_N_LIST) / sizeof(PW_TAILFOLD_N_LIST[0])))

static void test_pairwise_sum_tailfold(void) {
    static float a[PW_TAILFOLD_MAX_N];
    int ni, t;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_tailfold_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_f32(a, (size_t)n);
                check_bits_classify("pairwise_sum_tailfold_f32", n, t, &rn, &rs, 1);
            }
        }
    }
    /* dedicated signed-zero small-n checks (see header comment: this
     * kernel's 0.0f-seeded small-n accumulator normalizes -0.0f to +0.0f,
     * a bit pattern that must still round-trip scalar==NEON identically). */
    {
        float az1[1] = { -0.0f };
        float rs = sk_pairwise_sum_tailfold_f32_scalar(az1, 1);
        float rn = sk_pairwise_sum_tailfold_f32(az1, 1);
        check_scalar_bits_or_die("pairwise_sum_tailfold_f32_negzero_n1", 1, 0, rn, rs);
    }
    {
        float az5[5] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
        float rs = sk_pairwise_sum_tailfold_f32_scalar(az5, 5);
        float rn = sk_pairwise_sum_tailfold_f32(az5, 5);
        check_scalar_bits_or_die("pairwise_sum_tailfold_f32_negzero_n5", 5, 0, rn, rs);
    }
    printf("PASS pairwise_sum_tailfold_f32\n");
}


static void test_pairwise_sum_tailfold_b(void) {
    static float a[PW_TAILFOLD_MAX_N];
    int ni, t;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(a, n);
            {
                float rs = sk_pairwise_sum_tailfold_b_f32_scalar(a, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_b_f32(a, (size_t)n);
                check_bits_classify("pairwise_sum_tailfold_b_f32", n, t, &rn, &rs, 1);
            }
        }
    }
    /* n==0 explicit-return path. */
    {
        float dummy[1] = { 1.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(dummy, 0);
        float rn = sk_pairwise_sum_tailfold_b_f32(dummy, 0);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_n0", 0, 0, rn, rs);
    }
    /* dedicated signed-zero small-n checks (see header comment: this
     * kernel's a[0]-seeded small-n accumulator preserves -0.0f as-is,
     * unlike kernel 21 -- both must still be scalar==NEON internally). */
    {
        float az1[1] = { -0.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(az1, 1);
        float rn = sk_pairwise_sum_tailfold_b_f32(az1, 1);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_negzero_n1", 1, 0, rn, rs);
    }
    {
        float az5[5] = { -0.0f, -0.0f, -0.0f, -0.0f, -0.0f };
        float rs = sk_pairwise_sum_tailfold_b_f32_scalar(az5, 5);
        float rn = sk_pairwise_sum_tailfold_b_f32(az5, 5);
        check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32_negzero_n5", 5, 0, rn, rs);
    }
    printf("PASS pairwise_sum_tailfold_b_f32\n");
}

/* ═══════════ alignment + canary edge-case matrix ══════════════════════════
 * Layered on top of every per-kernel correctness test above (same design as audiolib/common/test/simd_selftest.c's
 * equivalent section; see its header comment for the full rationale):
 *
 *   - n=0 / n=1..17 are covered by the extended N_LIST/NAN_N_LIST/PW_TAILFOLD_N_LIST above. For n=0 the canary buffers
 *     below turn "the kernel touches nothing" into something checked.
 *   - Unaligned float/Complex-element offsets 1..15: every buffer lives in its own 64-byte-aligned arena (posix_memalign);
 *     each kernel is called at a +1..+15-element offset (naturally aligned for its element type, but not 16-/64-byte
 *     aligned). Three forms per kernel with distinct input/output roles: input-offset-only, output-offset-only and
 *     both-different (edge_offsets_for_form, shared with the common-file section). Kernels with more than one read-only
 *     input (wupdate_nlms/kf's err/mu/mu_scale, coherence_ema_gate's near_spec/abs_echo/abs_near/sye_im/syy/see) keep every
 *     non-primary array at offset 0. The four pairwise-sum-family reductions (kernels 13/14/21/22) return a float by value,
 *     so they sweep only the input array's offset. sk_mask_zero_f32 (kernel 20) is a single in-place buffer (see
 *     test_mask_zero_edge), like sk_clip_f32's edge test in the common file.
 *   - Canary guard: every arena is canary-filled before each call (float bit pattern 0x7fc0dead / mask byte 0xAA), then the
 *     payload window is overwritten with generated data. After the call every element OUTSIDE the window must still read
 *     back as the canary, in every arena the kernel touched (read-only ones too, as a cross-check against aliasing writes).
 *     This catches an out-of-bounds write on either side of the payload, or any write at all when n==0.
 *
 * The scalar-vs-NEON comparison is inherent to every check_bits_or_die/check_scalar_bits_or_die call below. */

#define EDGE_GUARD 32           /* guaranteed guard elements each side of the
                                 * payload window, regardless of offset/n */
#define EDGE_OFFSET_MAX 15      /* max float/Complex element offset under test */
#define EDGE_MAX_N SK_TEST_MAX_N
#define EDGE_ARENA_LEN (EDGE_GUARD + EDGE_OFFSET_MAX + EDGE_MAX_N + EDGE_GUARD)
/* Larger arena for the tail-fold pairwise-sum pair (kernels 21/22), whose
 * own PW_TAILFOLD_N_LIST reaches 960 (> SK_TEST_MAX_N). */
#define EDGE_ARENA_LEN_RED (EDGE_GUARD + EDGE_OFFSET_MAX + PW_TAILFOLD_MAX_N + EDGE_GUARD)
#define EDGE_CANARY_BITS 0x7fc0deadu
#define EDGE_MASK_CANARY 0xAAu
#define EDGE_FORM_COUNT 3

static float edge_canary_float(void) { return bits_to_float(EDGE_CANARY_BITS); }

static void *edge_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0 || p == NULL) {
        fprintf(stderr, "FATAL: posix_memalign(64, %zu) failed\n", bytes);
        exit(1);
    }
    return p;
}

static void edge_fill_canary_f(float *arena, int len) {
    int i;
    float c = edge_canary_float();
    for (i = 0; i < len; ++i) arena[i] = c;
}

/* Verifies every float in arena[0,len) OUTSIDE the payload window
 * [win_lo, win_lo+win_len) still holds the exact canary bit pattern -- an
 * empty window (win_len==0, i.e. n==0) means the ENTIRE arena must still be
 * canary, exactly the "n==0 performs zero reads/writes" contract this
 * section exists to check. exit(1) with a precise diagnostic on the first
 * violation, same house style as check_bits_or_die. */
static void edge_check_canary_f(const char *label, const float *arena, int len,
                                 int win_lo, int win_len) {
    int i;
    uint32_t want = EDGE_CANARY_BITS;
    int win_hi = win_lo + win_len;
    for (i = 0; i < len; ++i) {
        uint32_t got;
        if (i >= win_lo && i < win_hi) continue; /* payload window, not guarded */
        g_total_checks++;
        memcpy(&got, &arena[i], sizeof got);
        if (got != want) {
            fprintf(stderr,
                "CANARY VIOLATION %s: arena[%d]=0x%08x (want canary 0x%08x) "
                "-- out-of-bounds access, payload window=[%d,%d)\n",
                label, i, (unsigned)got, (unsigned)want, win_lo, win_hi);
            exit(1);
        }
    }
}

/* Complex-array counterparts: Complex is {float r; float i;} contiguous, so
 * a Complex arena is just a float arena with every length/offset doubled --
 * reuses the float helpers above instead of duplicating the canary logic. */
static void edge_fill_canary_c(Complex *arena, int len) {
    edge_fill_canary_f((float *)arena, len * 2);
}
static void edge_check_canary_c(const char *label, const Complex *arena, int len,
                                 int win_lo, int win_len) {
    edge_check_canary_f(label, (const float *)arena, len * 2, win_lo * 2, win_len * 2);
}

/* Byte-mask counterpart (coherence_ema_gate's mask output, mask_zero's mask
 * input): 0xAA is neither a legal 0 nor 1 mask value, so any leaked write
 * into the guard region is unambiguous, same intent as the float pattern. */
static void edge_fill_canary_b(unsigned char *arena, int len) {
    int i;
    for (i = 0; i < len; ++i) arena[i] = (unsigned char)EDGE_MASK_CANARY;
}
static void edge_check_canary_b(const char *label, const unsigned char *arena, int len,
                                 int win_lo, int win_len) {
    int i;
    int win_hi = win_lo + win_len;
    for (i = 0; i < len; ++i) {
        if (i >= win_lo && i < win_hi) continue;
        g_total_checks++;
        if (arena[i] != (unsigned char)EDGE_MASK_CANARY) {
            fprintf(stderr,
                "CANARY VIOLATION %s: mask[%d]=0x%02x (want canary 0x%02x) "
                "-- out-of-bounds access, payload window=[%d,%d)\n",
                label, i, (unsigned)arena[i], (unsigned)EDGE_MASK_CANARY, win_lo, win_hi);
            exit(1);
        }
    }
}

/* int32 counterpart (hold_counters, kernels 24/25/26): 0xDEADC0DE is not a
 * value any of these kernels' arithmetic would plausibly produce (nowhere
 * near hold_hops=400 or any of the special_int_pool values), same
 * unambiguous-leak-detector intent as the float/byte canary patterns above. */
#define EDGE_INT_CANARY ((int)0xDEADC0DEu)

static void edge_fill_canary_i(int *arena, int len) {
    int i;
    for (i = 0; i < len; ++i) arena[i] = EDGE_INT_CANARY;
}
static void edge_check_canary_i(const char *label, const int *arena, int len,
                                 int win_lo, int win_len) {
    int i;
    int win_hi = win_lo + win_len;
    for (i = 0; i < len; ++i) {
        if (i >= win_lo && i < win_hi) continue;
        g_total_checks++;
        if (arena[i] != EDGE_INT_CANARY) {
            fprintf(stderr,
                "CANARY VIOLATION %s: arena[%d]=%d (want canary %d) "
                "-- out-of-bounds access, payload window=[%d,%d)\n",
                label, i, arena[i], EDGE_INT_CANARY, win_lo, win_hi);
            exit(1);
        }
    }
}

/* Derives the (input-role, output-role) element offsets for the matrix's
 * three forms. Form 2 ("both, different") uses `((o+7)%15)+1` rather than
 * the obvious mirror `16-o`: the mirror collides with o itself at the
 * midpoint (o==8 -> 16-8==8). `(o+7)%15` is a fixed-point-free derangement
 * over {1..15} (o+7 == o (mod 15) requires 7 == 0 (mod 15), false for every
 * o), so out_off != in_off for every o in 1..15 by construction. */
static void edge_offsets_for_form(int form, int o, int *in_off, int *out_off) {
    switch (form) {
    case 0: *in_off = o; *out_off = 0; break;                   /* input offset only */
    case 1: *in_off = 0; *out_off = o; break;                   /* output offset only */
    default: *in_off = o; *out_off = ((o + 7) % 15) + 1; break;  /* both, different */
    }
}

static void test_cabs_np_edge(void) {
    Complex *z_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(z_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_complex(z_arena + in_off, n);

                sk_cabs_np_f32_scalar(z_arena + in_off, out_scalar_arena + out_off, n);
                sk_cabs_np_f32(z_arena + in_off, out_simd_arena + out_off, n);

                check_bits_or_die("cabs_np_f32_edge", n, form * 100 + o,
                                   out_simd_arena + out_off, out_scalar_arena + out_off, n);
                edge_check_canary_c("cabs_np_f32_edge:z", z_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("cabs_np_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("cabs_np_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(z_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS cabs_np_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_cmag2_np_edge(void) {
    Complex *z_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(z_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_complex(z_arena + in_off, n);

                sk_cmag2_np_f32_scalar(z_arena + in_off, out_scalar_arena + out_off, n);
                sk_cmag2_np_f32(z_arena + in_off, out_simd_arena + out_off, n);

                check_bits_or_die("cmag2_np_f32_edge", n, form * 100 + o,
                                   out_simd_arena + out_off, out_scalar_arena + out_off, n);
                edge_check_canary_c("cmag2_np_f32_edge:z", z_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("cmag2_np_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("cmag2_np_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(z_arena); free(out_scalar_arena); free(out_simd_arena);
    printf("PASS cmag2_np_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_cmag2_np_acc_edge(void) {
    Complex *z_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *acc_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *acc_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(z_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(acc_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(acc_simd_arena, EDGE_ARENA_LEN);
                fill_complex(z_arena + in_off, n);
                fill_floats(acc_scalar_arena + out_off, n);
                memcpy(acc_simd_arena + out_off, acc_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_cmag2_np_acc_f32_scalar(z_arena + in_off, acc_scalar_arena + out_off, n);
                sk_cmag2_np_acc_f32(z_arena + in_off, acc_simd_arena + out_off, n);

                check_bits_or_die("cmag2_np_acc_f32_edge", n, form * 100 + o,
                                   acc_simd_arena + out_off, acc_scalar_arena + out_off, n);
                edge_check_canary_c("cmag2_np_acc_f32_edge:z", z_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("cmag2_np_acc_f32_edge:acc_scalar", acc_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("cmag2_np_acc_f32_edge:acc_simd", acc_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(z_arena); free(acc_scalar_arena); free(acc_simd_arena);
    printf("PASS cmag2_np_acc_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_wupdate_nlms_edge(void) {
    Complex *X_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *err_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *mu_eff_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    Complex *W_scalar_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *W_simd_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(X_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(err_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(mu_eff_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(W_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(W_simd_arena, EDGE_ARENA_LEN);
                fill_complex(X_arena + in_off, n);
                fill_complex(err_arena, n);
                fill_floats(mu_eff_arena, n);
                fill_complex(W_scalar_arena + out_off, n);
                memcpy(W_simd_arena + out_off, W_scalar_arena + out_off, (size_t)n * sizeof(Complex));

                sk_wupdate_nlms_f32_scalar(W_scalar_arena + out_off, X_arena + in_off, err_arena, mu_eff_arena, n);
                sk_wupdate_nlms_f32(W_simd_arena + out_off, X_arena + in_off, err_arena, mu_eff_arena, n);

                /* classified, not strict: wupdate_nlms is one of the header's documented "NaN in ->
                 * NaN out, PAYLOAD UNSPECIFIED" kernels (grad = err*conj(X), a multi-operand fmaf
                 * term). The extreme-value special pool (+-Inf) can produce a NaN, and different
                 * codegen can select a different, still in-contract payload. */
                check_bits_classify("wupdate_nlms_f32_edge", n, form * 100 + o,
                                     (const float *)(W_simd_arena + out_off),
                                     (const float *)(W_scalar_arena + out_off), 2 * n);
                edge_check_canary_c("wupdate_nlms_f32_edge:X", X_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("wupdate_nlms_f32_edge:err", err_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("wupdate_nlms_f32_edge:mu_eff", mu_eff_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_c("wupdate_nlms_f32_edge:W_scalar", W_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_c("wupdate_nlms_f32_edge:W_simd", W_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(X_arena); free(err_arena); free(mu_eff_arena); free(W_scalar_arena); free(W_simd_arena);
    printf("PASS wupdate_nlms_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_wupdate_kf_edge(void) {
    Complex *X_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *err_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *mu_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *mu_scale_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    Complex *W_scalar_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *W_simd_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(X_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(err_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(mu_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(mu_scale_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(W_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(W_simd_arena, EDGE_ARENA_LEN);
                fill_complex(X_arena + in_off, n);
                fill_complex(err_arena, n);
                fill_floats(mu_arena, n);
                fill_floats(mu_scale_arena, n);
                fill_complex(W_scalar_arena + out_off, n);
                memcpy(W_simd_arena + out_off, W_scalar_arena + out_off, (size_t)n * sizeof(Complex));

                sk_wupdate_kf_f32_scalar(W_scalar_arena + out_off, X_arena + in_off, err_arena, mu_arena, mu_scale_arena, n);
                sk_wupdate_kf_f32(W_simd_arena + out_off, X_arena + in_off, err_arena, mu_arena, mu_scale_arena, n);

                /* classified, not strict; see wupdate_nlms_f32_edge's comment above (the
                 * finite-corpus test_wupdate_kf() hits the same class of divergence). */
                check_bits_classify("wupdate_kf_f32_edge", n, form * 100 + o,
                                     (const float *)(W_simd_arena + out_off),
                                     (const float *)(W_scalar_arena + out_off), 2 * n);
                edge_check_canary_c("wupdate_kf_f32_edge:X", X_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("wupdate_kf_f32_edge:err", err_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("wupdate_kf_f32_edge:mu", mu_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("wupdate_kf_f32_edge:mu_scale", mu_scale_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_c("wupdate_kf_f32_edge:W_scalar", W_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_c("wupdate_kf_f32_edge:W_simd", W_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(X_arena); free(err_arena); free(mu_arena); free(mu_scale_arena);
    free(W_scalar_arena); free(W_simd_arena);
    printf("PASS wupdate_kf_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_pairwise_sum_edge(void) {
    float *a_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(a_arena, EDGE_ARENA_LEN);
            fill_floats(a_arena + o, n);
            {
                float rs = sk_pairwise_sum_f32_scalar(a_arena + o, (size_t)n);
                float rn = sk_pairwise_sum_f32(a_arena + o, (size_t)n);
                /* classified, not strict -- pairwise_sum is one of the
                 * header's documented "payload unspecified" reduction
                 * kernels (matches test_pairwise_sum_nan()'s own pattern
                 * above: wrap the by-value result in a 1-element array). */
                check_bits_classify("pairwise_sum_f32_edge", n, o, &rn, &rs, 1);
            }
            edge_check_canary_f("pairwise_sum_f32_edge:a", a_arena, EDGE_ARENA_LEN, o, n);
        }
    }
    free(a_arena);
    printf("PASS pairwise_sum_f32_edge (n=0..17+existing x input offset 1..15, "
           "no output buffer for a by-value reduction, canary-guarded)\n");
}

static void test_sum_sq_pairwise_edge(void) {
    float *a_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(a_arena, EDGE_ARENA_LEN);
            fill_floats(a_arena + o, n);
            {
                float rs = sk_sum_sq_pairwise_f32_scalar(a_arena + o, (size_t)n);
                float rn = sk_sum_sq_pairwise_f32(a_arena + o, (size_t)n);
                /* classified, not strict -- see pairwise_sum_f32_edge's
                 * comment above. */
                check_bits_classify("sum_sq_pairwise_f32_edge", n, o, &rn, &rs, 1);
            }
            edge_check_canary_f("sum_sq_pairwise_f32_edge:a", a_arena, EDGE_ARENA_LEN, o, n);
        }
    }
    free(a_arena);
    printf("PASS sum_sq_pairwise_f32_edge (n=0..17+existing x input offset 1..15, "
           "no output buffer for a by-value reduction, canary-guarded)\n");
}

static void test_pairwise_sum_tailfold_edge(void) {
    float *a_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN_RED * sizeof(float));
    int ni, o;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(a_arena, EDGE_ARENA_LEN_RED);
            fill_floats(a_arena + o, n);
            {
                float rs = sk_pairwise_sum_tailfold_f32_scalar(a_arena + o, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_f32(a_arena + o, (size_t)n);
                /* classified, not strict -- see pairwise_sum_f32_edge's
                 * comment above. */
                check_bits_classify("pairwise_sum_tailfold_f32_edge", n, o, &rn, &rs, 1);
            }
            edge_check_canary_f("pairwise_sum_tailfold_f32_edge:a", a_arena, EDGE_ARENA_LEN_RED, o, n);
        }
    }
    free(a_arena);
    printf("PASS pairwise_sum_tailfold_f32_edge (n=0..17+existing incl. 960 x input offset 1..15, "
           "canary-guarded)\n");
}

static void test_pairwise_sum_tailfold_b_edge(void) {
    float *a_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN_RED * sizeof(float));
    int ni, o;
    for (ni = 0; ni < PW_TAILFOLD_N_LIST_COUNT; ++ni) {
        int n = PW_TAILFOLD_N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(a_arena, EDGE_ARENA_LEN_RED);
            fill_floats(a_arena + o, n);
            {
                float rs = sk_pairwise_sum_tailfold_b_f32_scalar(a_arena + o, (size_t)n);
                float rn = sk_pairwise_sum_tailfold_b_f32(a_arena + o, (size_t)n);
                /* classified, not strict -- see pairwise_sum_f32_edge's
                 * comment above. */
                check_bits_classify("pairwise_sum_tailfold_b_f32_edge", n, o, &rn, &rs, 1);
            }
            edge_check_canary_f("pairwise_sum_tailfold_b_f32_edge:a", a_arena, EDGE_ARENA_LEN_RED, o, n);
        }
    }
    free(a_arena);
    printf("PASS pairwise_sum_tailfold_b_f32_edge (n=0..17+existing incl. 960 x input offset 1..15, "
           "canary-guarded)\n");
}

static void test_coherence_ema_gate_edge(void) {
    Complex *echo_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *near_spec_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *abs_echo_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *abs_near_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *sye_re_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *sye_re_simd_arena   = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *sye_im_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *sye_im_simd_arena   = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *syy_scalar_arena    = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *syy_simd_arena      = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *see_scalar_arena    = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *see_simd_arena      = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    unsigned char *mask_scalar_arena = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    unsigned char *mask_simd_arena   = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    const float alpha = 0.05f, threshold = 0.5f;
    int ni, form, o;
    /* Primary input role = echo (Complex), primary output role = sye_re (float); every other buffer
     * (sye_im/syy/see/near_spec/abs_echo/abs_near/mask) stays at offset 0, since a 9-buffer kernel
     * does not fit the binary input/output framing. sye_im/syy/see are read-write, so each gets its
     * own scalar/simd arena pair even though its offset never varies (a shared arena would let the
     * scalar call's mutation leak into the simd call's "before" state). */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off, i;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(echo_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(near_spec_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(abs_echo_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(abs_near_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(sye_re_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(sye_re_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(sye_im_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(sye_im_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(syy_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(syy_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(see_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(see_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_b(mask_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_b(mask_simd_arena, EDGE_ARENA_LEN);

                fill_complex(echo_arena + in_off, n);
                fill_complex(near_spec_arena, n);
                fill_floats(abs_echo_arena, n);
                fill_floats(abs_near_arena, n);
                fill_floats(sye_re_scalar_arena + out_off, n);
                memcpy(sye_re_simd_arena + out_off, sye_re_scalar_arena + out_off, (size_t)n * sizeof(float));
                fill_floats(sye_im_scalar_arena, n);
                memcpy(sye_im_simd_arena, sye_im_scalar_arena, (size_t)n * sizeof(float));
                fill_floats(syy_scalar_arena, n);
                memcpy(syy_simd_arena, syy_scalar_arena, (size_t)n * sizeof(float));
                fill_floats(see_scalar_arena, n);
                memcpy(see_simd_arena, see_scalar_arena, (size_t)n * sizeof(float));

                sk_coherence_ema_gate_f32_scalar(
                    sye_re_scalar_arena + out_off, sye_im_scalar_arena, syy_scalar_arena, see_scalar_arena,
                    echo_arena + in_off, near_spec_arena, abs_echo_arena, abs_near_arena,
                    alpha, threshold, mask_scalar_arena, n);
                sk_coherence_ema_gate_f32(
                    sye_re_simd_arena + out_off, sye_im_simd_arena, syy_simd_arena, see_simd_arena,
                    echo_arena + in_off, near_spec_arena, abs_echo_arena, abs_near_arena,
                    alpha, threshold, mask_simd_arena, n);

                /* classified, not strict: echo/near_spec are drawn from the full special-value pool
                 * including +-Inf, and pr=er*nr+ei*ni / pi=ei*nr-er*ni can produce a NaN from
                 * finite/Inf operands (e.g. (+Inf)*(-Inf) + (+Inf)*(+Inf) = -Inf+Inf = NaN);
                 * test_coherence_ema_gate_nan() makes the same choice of
                 * check_bits_classify/check_mask_classify for this kernel. */
                check_bits_classify("coherence_ema_gate_f32_edge:sye_re", n, form * 100 + o,
                                     sye_re_simd_arena + out_off, sye_re_scalar_arena + out_off, n);
                check_bits_classify("coherence_ema_gate_f32_edge:sye_im", n, form * 100 + o,
                                     sye_im_simd_arena, sye_im_scalar_arena, n);
                check_bits_classify("coherence_ema_gate_f32_edge:syy", n, form * 100 + o,
                                     syy_simd_arena, syy_scalar_arena, n);
                check_bits_classify("coherence_ema_gate_f32_edge:see", n, form * 100 + o,
                                     see_simd_arena, see_scalar_arena, n);
                for (i = 0; i < n; ++i) {
                    check_mask_classify("coherence_ema_gate_f32_edge:mask", n, i,
                                         mask_simd_arena[i], mask_scalar_arena[i]);
                }

                edge_check_canary_c("coherence_ema_gate_f32_edge:echo", echo_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("coherence_ema_gate_f32_edge:near_spec", near_spec_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:abs_echo", abs_echo_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:abs_near", abs_near_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:sye_re_scalar", sye_re_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:sye_re_simd", sye_re_simd_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:sye_im_scalar", sye_im_scalar_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:sye_im_simd", sye_im_simd_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:syy_scalar", syy_scalar_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:syy_simd", syy_simd_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:see_scalar", see_scalar_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("coherence_ema_gate_f32_edge:see_simd", see_simd_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_b("coherence_ema_gate_f32_edge:mask_scalar", mask_scalar_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_b("coherence_ema_gate_f32_edge:mask_simd", mask_simd_arena, EDGE_ARENA_LEN, 0, n);
            }
        }
    }
    free(echo_arena); free(near_spec_arena); free(abs_echo_arena); free(abs_near_arena);
    free(sye_re_scalar_arena); free(sye_re_simd_arena);
    free(sye_im_scalar_arena); free(sye_im_simd_arena);
    free(syy_scalar_arena); free(syy_simd_arena);
    free(see_scalar_arena); free(see_simd_arena);
    free(mask_scalar_arena); free(mask_simd_arena);
    printf("PASS coherence_ema_gate_f32_edge (n=0..17+existing x echo/sye_re offset 1..15 x 3 forms; "
           "sye_im/syy/see/near_spec/abs_echo/abs_near/mask fixed@0, canary-guarded)\n");
}

static void test_ema_delta_edge(void) {
    float *x_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float alpha = 0.23156652857908377f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(x_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_simd_arena, EDGE_ARENA_LEN);
                fill_floats(x_arena + in_off, n);
                fill_floats(state_scalar_arena + out_off, n);
                memcpy(state_simd_arena + out_off, state_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_ema_delta_f32_scalar(state_scalar_arena + out_off, x_arena + in_off, alpha, n);
                sk_ema_delta_f32(state_simd_arena + out_off, x_arena + in_off, alpha, n);

                /* classified, not strict -- ema_delta is a documented
                 * "payload unspecified" kernel (x-state can be Inf-Inf=NaN
                 * for opposite-signed Inf operands drawn from the special
                 * pool). See wupdate_nlms_f32_edge's comment above. */
                check_bits_classify("ema_delta_f32_edge", n, form * 100 + o,
                                     state_simd_arena + out_off, state_scalar_arena + out_off, n);
                edge_check_canary_f("ema_delta_f32_edge:x", x_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("ema_delta_f32_edge:state_scalar", state_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("ema_delta_f32_edge:state_simd", state_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(x_arena); free(state_scalar_arena); free(state_simd_arena);
    printf("PASS ema_delta_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_n2_track_edge(void) {
    float *y2s_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *n2_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *n2_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float fresh = 0.9968377223398316f;
    const float retain = 0.003162277660168411f;
    const float g_up = 1.0005000750025f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(y2s_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(n2_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(n2_simd_arena, EDGE_ARENA_LEN);
                fill_floats(y2s_arena + in_off, n);
                fill_floats(n2_scalar_arena + out_off, n);
                memcpy(n2_simd_arena + out_off, n2_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_n2_track_f32_scalar(n2_scalar_arena + out_off, y2s_arena + in_off, fresh, retain, g_up, n);
                sk_n2_track_f32(n2_simd_arena + out_off, y2s_arena + in_off, fresh, retain, g_up, n);

                /* classified, not strict -- n2_track is a documented
                 * "payload unspecified" kernel (the "track" branch's
                 * fresh*y2s+retain*n2 can be Inf-Inf=NaN when selected).
                 * See wupdate_nlms_f32_edge's comment above. */
                check_bits_classify("n2_track_f32_edge", n, form * 100 + o,
                                     n2_simd_arena + out_off, n2_scalar_arena + out_off, n);
                edge_check_canary_f("n2_track_f32_edge:y2s", y2s_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("n2_track_f32_edge:n2_scalar", n2_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("n2_track_f32_edge:n2_simd", n2_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(y2s_arena); free(n2_scalar_arena); free(n2_simd_arena);
    printf("PASS n2_track_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_n2_initial_track_edge(void) {
    float *n2_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *n2i_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *n2i_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float alpha = 0.0024981253125391234f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(n2_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(n2i_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(n2i_simd_arena, EDGE_ARENA_LEN);
                fill_floats(n2_arena + in_off, n);
                fill_floats(n2i_scalar_arena + out_off, n);
                memcpy(n2i_simd_arena + out_off, n2i_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_n2_initial_track_f32_scalar(n2i_scalar_arena + out_off, n2_arena + in_off, alpha, n);
                sk_n2_initial_track_f32(n2i_simd_arena + out_off, n2_arena + in_off, alpha, n);

                /* classified, not strict -- n2_initial_track is a
                 * documented "payload unspecified" kernel (the "slow"
                 * branch's old+alpha*(n2-old) can be Inf-Inf=NaN when
                 * selected). See wupdate_nlms_f32_edge's comment above. */
                check_bits_classify("n2_initial_track_f32_edge", n, form * 100 + o,
                                     n2i_simd_arena + out_off, n2i_scalar_arena + out_off, n);
                edge_check_canary_f("n2_initial_track_f32_edge:n2", n2_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("n2_initial_track_f32_edge:n2i_scalar", n2i_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("n2_initial_track_f32_edge:n2i_simd", n2i_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(n2_arena); free(n2i_scalar_arena); free(n2i_simd_arena);
    printf("PASS n2_initial_track_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_mask_zero_edge(void) {
    float *x_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *x_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    unsigned char *mask_arena = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    int ni, o;
    /* sk_mask_zero_f32 is the AEC header's other documented alias form (kernel 20, "in-place" doc
     * comment in utility/simd_aec.h): it has no separate output parameter, so x is both the read
     * source and the write destination on every call, and every call in this matrix is already its
     * only in-place exercise. A single-buffer offset sweep, like sk_clip_f32's edge test in the
     * common file; mask stays at offset 0 (read-only, shared safely between the scalar and simd
     * calls) and is byte-canary-guarded. */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            int i;
            edge_fill_canary_f(x_scalar_arena, EDGE_ARENA_LEN);
            edge_fill_canary_f(x_simd_arena, EDGE_ARENA_LEN);
            edge_fill_canary_b(mask_arena, EDGE_ARENA_LEN);
            fill_floats(x_scalar_arena + o, n);
            memcpy(x_simd_arena + o, x_scalar_arena + o, (size_t)n * sizeof(float));
            for (i = 0; i < n; ++i) mask_arena[i] = (unsigned char)(lcg_next() & 1u);

            sk_mask_zero_f32_scalar(x_scalar_arena + o, mask_arena, n);
            sk_mask_zero_f32(x_simd_arena + o, mask_arena, n);

            /* classified for consistency with this section's other kernels (matching
             * test_mask_zero_nan()), though mask_zero is pure pass-through/zero-select with no
             * arithmetic, so a both-NaN classification should never fire; if it does, that is a
             * meaningful signal. */
            check_bits_classify("mask_zero_f32_edge", n, o, x_simd_arena + o, x_scalar_arena + o, n);
            edge_check_canary_f("mask_zero_f32_edge:x_scalar", x_scalar_arena, EDGE_ARENA_LEN, o, n);
            edge_check_canary_f("mask_zero_f32_edge:x_simd", x_simd_arena, EDGE_ARENA_LEN, o, n);
            edge_check_canary_b("mask_zero_f32_edge:mask", mask_arena, EDGE_ARENA_LEN, 0, n);
        }
    }
    free(x_scalar_arena); free(x_simd_arena); free(mask_arena);
    printf("PASS mask_zero_f32_edge (n=0..17+existing x offset 1..15, in-place alias form, canary-guarded)\n");
}

/* ═══════════════════════════ correctness: kernel 23 ══════════════════════ */

static void test_noise_spectrum_update_edge(void) {
    float *spec_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *noise_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *noise_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float alpha = 0.0024981253125391234f;
    const float min_noise = 1.0e-4f;
    int ni, form, o, m10;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                for (m10 = 0; m10 < 2; ++m10) {
                    int in_off, out_off;
                    edge_offsets_for_form(form, o, &in_off, &out_off);

                    edge_fill_canary_f(spec_arena, EDGE_ARENA_LEN);
                    edge_fill_canary_f(noise_scalar_arena, EDGE_ARENA_LEN);
                    edge_fill_canary_f(noise_simd_arena, EDGE_ARENA_LEN);
                    fill_floats(spec_arena + in_off, n);
                    fill_floats(noise_scalar_arena + out_off, n);
                    memcpy(noise_simd_arena + out_off, noise_scalar_arena + out_off,
                           (size_t)n * sizeof(float));

                    sk_noise_spectrum_update_f32_scalar(noise_scalar_arena + out_off,
                        spec_arena + in_off, alpha, m10, min_noise, n);
                    sk_noise_spectrum_update_f32(noise_simd_arena + out_off,
                        spec_arena + in_off, alpha, m10, min_noise, n);

                    /* classified, not strict -- like n2_track/n2_initial_track,
                     * the rising branch's alpha_inc*(pb-pn) combine can hit an
                     * Inf-Inf=NaN payload on extreme special-pool inputs; see
                     * n2_track_edge's comment above for the house rationale. */
                    check_bits_classify("noise_spectrum_update_f32_edge", n,
                                         form * 200 + o * 2 + m10,
                                         noise_simd_arena + out_off,
                                         noise_scalar_arena + out_off, n);
                    edge_check_canary_f("noise_spectrum_update_f32_edge:spec", spec_arena, EDGE_ARENA_LEN, in_off, n);
                    edge_check_canary_f("noise_spectrum_update_f32_edge:noise_scalar", noise_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                    edge_check_canary_f("noise_spectrum_update_f32_edge:noise_simd", noise_simd_arena, EDGE_ARENA_LEN, out_off, n);
                }
            }
        }
    }
    free(spec_arena); free(noise_scalar_arena); free(noise_simd_arena);
    printf("PASS noise_spectrum_update_f32_edge (n=0..17+existing x offset 1..15 x 3 forms x apply_mask10{0,1}, canary-guarded)\n");
}

/* ═══════════════════════════ correctness: kernel 24 ══════════════════════ */

static void test_erl_bin_update_edge(void) {
    float *x2_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *y2_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *erl_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *erl_simd_arena   = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int   *hold_scalar_arena = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    int   *hold_simd_arena   = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    const float x2_min = 44015068.0f;
    const int hold_hops = 400;
    const float min_erl = 0.01f;
    int ni, form, o;
    /* Primary input role = x2, primary output role = erl; y2 (secondary read-only input) and hold
     * (secondary in-place int32 array) stay at offset 0, the same >2-buffer scope decision as
     * test_coherence_ema_gate_edge above. hold is read-write, so it gets its own scalar/simd arena
     * pair even though its offset never varies (a shared arena would let the scalar call's mutation
     * leak into the simd call's "before" state). */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(x2_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(y2_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(erl_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(erl_simd_arena, EDGE_ARENA_LEN);
                edge_fill_canary_i(hold_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_i(hold_simd_arena, EDGE_ARENA_LEN);

                fill_floats(x2_arena + in_off, n);
                fill_floats(y2_arena, n);
                fill_floats(erl_scalar_arena + out_off, n);
                memcpy(erl_simd_arena + out_off, erl_scalar_arena + out_off, (size_t)n * sizeof(float));
                fill_ints(hold_scalar_arena, n);
                memcpy(hold_simd_arena, hold_scalar_arena, (size_t)n * sizeof(int));

                sk_erl_bin_update_f32_scalar(erl_scalar_arena + out_off, hold_scalar_arena,
                                              x2_arena + in_off, y2_arena,
                                              x2_min, hold_hops, min_erl, n);
                sk_erl_bin_update_f32(erl_simd_arena + out_off, hold_simd_arena,
                                       x2_arena + in_off, y2_arena,
                                       x2_min, hold_hops, min_erl, n);

                check_bits_classify("erl_bin_update_f32_edge:erl", n, form * 100 + o,
                                     erl_simd_arena + out_off, erl_scalar_arena + out_off, n);
                check_ints_or_die("erl_bin_update_f32_edge:hold", n, form * 100 + o,
                                   hold_simd_arena, hold_scalar_arena, n);

                edge_check_canary_f("erl_bin_update_f32_edge:x2", x2_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("erl_bin_update_f32_edge:y2", y2_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_f("erl_bin_update_f32_edge:erl_scalar", erl_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("erl_bin_update_f32_edge:erl_simd", erl_simd_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_i("erl_bin_update_f32_edge:hold_scalar", hold_scalar_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_i("erl_bin_update_f32_edge:hold_simd", hold_simd_arena, EDGE_ARENA_LEN, 0, n);
            }
        }
    }
    free(x2_arena); free(y2_arena); free(erl_scalar_arena); free(erl_simd_arena);
    free(hold_scalar_arena); free(hold_simd_arena);
    printf("PASS erl_bin_update_f32_edge (n=0..17+existing x x2/erl offset 1..15 x 3 forms; "
           "y2/hold fixed@0, canary-guarded)\n");
}

/* ═══════════════════════════ correctness: kernel 25 ══════════════════════ */

static void test_dec1_floorintmin_s32_edge(void) {
    int *x_scalar_arena = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    int *x_simd_arena   = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    int ni, o;
    /* Single in-place int32 buffer, no separate output parameter -- same
     * shape as sk_mask_zero_f32's edge test above (see that test's own
     * comment: no distinct input/output role to assign the 3-form matrix
     * to), so this sweeps only the one offset. */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_i(x_scalar_arena, EDGE_ARENA_LEN);
            edge_fill_canary_i(x_simd_arena, EDGE_ARENA_LEN);
            fill_ints(x_scalar_arena + o, n);
            memcpy(x_simd_arena + o, x_scalar_arena + o, (size_t)n * sizeof(int));

            sk_dec1_floorintmin_s32_scalar(x_scalar_arena + o, n);
            sk_dec1_floorintmin_s32(x_simd_arena + o, n);

            check_ints_or_die("dec1_floorintmin_s32_edge", n, o, x_simd_arena + o, x_scalar_arena + o, n);
            edge_check_canary_i("dec1_floorintmin_s32_edge:x_scalar", x_scalar_arena, EDGE_ARENA_LEN, o, n);
            edge_check_canary_i("dec1_floorintmin_s32_edge:x_simd", x_simd_arena, EDGE_ARENA_LEN, o, n);
        }
    }
    free(x_scalar_arena); free(x_simd_arena);
    printf("PASS dec1_floorintmin_s32_edge (n=0..17+existing x offset 1..15, in-place, canary-guarded)\n");
}

/* ═══════════════════════════ correctness: kernel 26 ══════════════════════ */

static void test_erl_hold_expire_edge(void) {
    int   *hold_arena = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    float *erl_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *erl_simd_arena   = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float max_erl = 1000.0f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_i(hold_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(erl_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(erl_simd_arena, EDGE_ARENA_LEN);
                fill_ints(hold_arena + in_off, n);
                fill_floats(erl_scalar_arena + out_off, n);
                memcpy(erl_simd_arena + out_off, erl_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_erl_hold_expire_f32_scalar(erl_scalar_arena + out_off, hold_arena + in_off, max_erl, n);
                sk_erl_hold_expire_f32(erl_simd_arena + out_off, hold_arena + in_off, max_erl, n);

                check_bits_classify("erl_hold_expire_f32_edge", n, form * 100 + o,
                                     erl_simd_arena + out_off, erl_scalar_arena + out_off, n);
                edge_check_canary_i("erl_hold_expire_f32_edge:hold", hold_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("erl_hold_expire_f32_edge:erl_scalar", erl_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("erl_hold_expire_f32_edge:erl_simd", erl_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(hold_arena); free(erl_scalar_arena); free(erl_simd_arena);
    printf("PASS erl_hold_expire_f32_edge (n=0..17+existing x hold/erl offset 1..15 x 3 forms, canary-guarded)\n");
}

/* ═══════════════════════════ correctness: kernel 27 ══════════════════════
 * Kernel 27 takes nine input arrays plus two scalars. The two scalars sweep
 * SG_CFG below: the shipped balanced pair (0.25/0.25) first, then softness
 * values that drive the sigmoid argument to +-Inf/NaN and a threshold that
 * is itself NaN -- the sigmoid clamps and the two 1e-30f denominator clamps
 * only differ between compare+select and FMIN/FMAX on operands like those,
 * so a corpus without them would pass a kernel written with vminq/vmaxq.
 * The tuning tables are drawn from the same gen_float() corpus as the
 * signals, which covers the degenerate enr_su == enr_tr case (d_lin == 0,
 * the low clamp's only in-domain trigger) by collision. */

#define SG_CFG_COUNT 4
static const float SG_CFG_THR[SG_CFG_COUNT]  = { 0.25f, 0.25f,  0.0f, 1e30f };
static const float SG_CFG_SOFT[SG_CFG_COUNT] = { 0.25f, 1e-30f, 0.25f, 1e-30f };

static void test_no_audible_echo_gain_edge(void) {
    float *ne_arena  = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *ec_arena  = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *mk_arena  = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *ntr_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *nsu_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *nem_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *otr_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *osu_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *oem_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *out_simd_arena   = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, form, o, c;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_f(ne_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(ec_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(mk_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(ntr_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(nsu_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(nem_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(otr_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(osu_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(oem_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(out_simd_arena, EDGE_ARENA_LEN);
                fill_floats(ne_arena + in_off, n);
                fill_floats(ec_arena + in_off, n);
                fill_floats(mk_arena + in_off, n);
                fill_floats(ntr_arena + in_off, n);
                fill_floats(nsu_arena + in_off, n);
                fill_floats(nem_arena + in_off, n);
                fill_floats(otr_arena + in_off, n);
                fill_floats(osu_arena + in_off, n);
                fill_floats(oem_arena + in_off, n);

                for (c = 0; c < SG_CFG_COUNT; ++c) {
                    sk_no_audible_echo_gain_f32_scalar(
                        ne_arena + in_off, ec_arena + in_off, mk_arena + in_off,
                        ntr_arena + in_off, nsu_arena + in_off, nem_arena + in_off,
                        otr_arena + in_off, osu_arena + in_off, oem_arena + in_off,
                        SG_CFG_THR[c], SG_CFG_SOFT[c], out_scalar_arena + out_off, n);
                    sk_no_audible_echo_gain_f32(
                        ne_arena + in_off, ec_arena + in_off, mk_arena + in_off,
                        ntr_arena + in_off, nsu_arena + in_off, nem_arena + in_off,
                        otr_arena + in_off, osu_arena + in_off, oem_arena + in_off,
                        SG_CFG_THR[c], SG_CFG_SOFT[c], out_simd_arena + out_off, n);
                    check_bits_classify("no_audible_echo_gain_f32_edge", n,
                                         (form * 100 + o) * 10 + c,
                                         out_simd_arena + out_off,
                                         out_scalar_arena + out_off, n);
                }
                edge_check_canary_f("no_audible_echo_gain_f32_edge:nearend", ne_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("no_audible_echo_gain_f32_edge:echo", ec_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("no_audible_echo_gain_f32_edge:masker", mk_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("no_audible_echo_gain_f32_edge:out_scalar", out_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("no_audible_echo_gain_f32_edge:out_simd", out_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(ne_arena); free(ec_arena); free(mk_arena);
    free(ntr_arena); free(nsu_arena); free(nem_arena);
    free(otr_arena); free(osu_arena); free(oem_arena);
    free(out_scalar_arena); free(out_simd_arena);
    printf("PASS no_audible_echo_gain_f32_edge (n=0..17+existing x offset 1..15 x 3 forms x %d cfg, canary-guarded)\n",
           SG_CFG_COUNT);
}

/* ═══════════════════════════════ microbench ═══════════════════════════════
 * n=257, ~200k reps, CLOCK_MONOTONIC. `g_bench_sink` (volatile) forces the
 * compiler to keep each call's result live, so the timing loop can't be
 * hoisted/eliminated as dead/invariant code. */

static volatile double g_bench_sink = 0.0;

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static void report_bench(const char *name, double ns_scalar, double ns_simd) {
    printf("kernel=%s ns_per_call_scalar=%.2f ns_per_call_simd=%.2f speedup=%.2f\n",
           name, ns_scalar, ns_simd,
           ns_simd > 0.0 ? ns_scalar / ns_simd : 0.0);
}
static void bench_cabs_np(void) {
    Complex z[BENCH_N]; float out[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cabs_np_f32_scalar(z, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cabs_np_f32(z, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("cabs_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_cmag2_np(void) {
    Complex z[BENCH_N]; float out[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_f32_scalar(z, out, BENCH_N); g_bench_sink += out[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_f32(z, out, BENCH_N); g_bench_sink += out[0]; }
            {
                double t3 = now_ns();
                report_bench("cmag2_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_cmag2_np_acc(void) {
    Complex z[BENCH_N]; float acc[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    fill_bench_floats(acc, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_acc_f32_scalar(z, acc, BENCH_N); g_bench_sink += acc[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmag2_np_acc_f32(z, acc, BENCH_N); g_bench_sink += acc[0]; }
            {
                double t3 = now_ns();
                report_bench("cmag2_np_acc_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}




static void bench_wupdate_nlms(void) {
    Complex X[BENCH_N], err[BENCH_N], W[BENCH_N];
    float mu_eff[BENCH_N];
    fill_bench_complex(X, BENCH_N);
    fill_bench_complex(err, BENCH_N);
    fill_bench_complex(W, BENCH_N);
    fill_bench_floats(mu_eff, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_nlms_f32_scalar(W, X, err, mu_eff, BENCH_N); g_bench_sink += W[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_nlms_f32(W, X, err, mu_eff, BENCH_N); g_bench_sink += W[0].r; }
            {
                double t3 = now_ns();
                report_bench("wupdate_nlms_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_wupdate_kf(void) {
    Complex X[BENCH_N], err[BENCH_N], W[BENCH_N];
    float mu[BENCH_N], mu_scale[BENCH_N];
    fill_bench_complex(X, BENCH_N);
    fill_bench_complex(err, BENCH_N);
    fill_bench_complex(W, BENCH_N);
    fill_bench_floats(mu, BENCH_N);
    fill_bench_floats(mu_scale, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_kf_f32_scalar(W, X, err, mu, mu_scale, BENCH_N); g_bench_sink += W[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_wupdate_kf_f32(W, X, err, mu, mu_scale, BENCH_N); g_bench_sink += W[0].r; }
            {
                double t3 = now_ns();
                report_bench("wupdate_kf_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_pairwise_sum(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_sum_sq_pairwise(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_sum_sq_pairwise_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_sum_sq_pairwise_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("sum_sq_pairwise_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_pairwise_sum_tailfold(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_tailfold_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_pairwise_sum_tailfold_b(void) {
    float a[BENCH_N];
    fill_bench_floats(a, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_b_f32_scalar(a, (size_t)BENCH_N);
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) g_bench_sink += sk_pairwise_sum_tailfold_b_f32(a, (size_t)BENCH_N);
            {
                double t3 = now_ns();
                report_bench("pairwise_sum_tailfold_b_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

/* ═══════════════════════════ correctness: kernel 16 ══════════════════════ */

static void test_coherence_ema_gate(void) {
    Complex echo[SK_TEST_MAX_N], near_spec[SK_TEST_MAX_N];
    float abs_echo[SK_TEST_MAX_N], abs_near[SK_TEST_MAX_N];
    float sye_re_init[SK_TEST_MAX_N], sye_im_init[SK_TEST_MAX_N];
    float syy_init[SK_TEST_MAX_N], see_init[SK_TEST_MAX_N];
    float sye_re_s[SK_TEST_MAX_N], sye_im_s[SK_TEST_MAX_N];
    float syy_s[SK_TEST_MAX_N], see_s[SK_TEST_MAX_N];
    float sye_re_n[SK_TEST_MAX_N], sye_im_n[SK_TEST_MAX_N];
    float syy_n[SK_TEST_MAX_N], see_n[SK_TEST_MAX_N];
    unsigned char mask_s[SK_TEST_MAX_N], mask_n[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.05f, threshold = 0.5f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(echo, n);
            fill_complex(near_spec, n);
            fill_floats(abs_echo, n);
            fill_floats(abs_near, n);
            fill_floats(sye_re_init, n);
            fill_floats(sye_im_init, n);
            fill_floats(syy_init, n);
            fill_floats(see_init, n);
            memcpy(sye_re_s, sye_re_init, (size_t)n * sizeof(float));
            memcpy(sye_im_s, sye_im_init, (size_t)n * sizeof(float));
            memcpy(syy_s, syy_init, (size_t)n * sizeof(float));
            memcpy(see_s, see_init, (size_t)n * sizeof(float));
            memcpy(sye_re_n, sye_re_init, (size_t)n * sizeof(float));
            memcpy(sye_im_n, sye_im_init, (size_t)n * sizeof(float));
            memcpy(syy_n, syy_init, (size_t)n * sizeof(float));
            memcpy(see_n, see_init, (size_t)n * sizeof(float));
            sk_coherence_ema_gate_f32_scalar(sye_re_s, sye_im_s, syy_s, see_s,
                                              echo, near_spec, abs_echo, abs_near,
                                              alpha, threshold, mask_s, n);
            sk_coherence_ema_gate_f32(sye_re_n, sye_im_n, syy_n, see_n,
                                       echo, near_spec, abs_echo, abs_near,
                                       alpha, threshold, mask_n, n);
            check_bits_classify("coherence_ema_gate_f32:sye_re", n, t, sye_re_n, sye_re_s, n);
            check_bits_classify("coherence_ema_gate_f32:sye_im", n, t, sye_im_n, sye_im_s, n);
            check_bits_classify("coherence_ema_gate_f32:syy", n, t, syy_n, syy_s, n);
            check_bits_classify("coherence_ema_gate_f32:see", n, t, see_n, see_s, n);
            {
                int idx;
                for (idx = 0; idx < n; ++idx) {
                    if (mask_s[idx] != mask_n[idx]) {
                        fprintf(stderr,
                            "MISMATCH kernel=coherence_ema_gate_f32:mask n=%d trial=%d idx=%d simd=%u scalar=%u\n",
                            n, t, idx, (unsigned)mask_n[idx], (unsigned)mask_s[idx]);
                        exit(1);
                    }
                }
            }
        }
    }
    printf("PASS coherence_ema_gate_f32\n");
}

/* ═══════════════════════════ correctness: kernel 17 ══════════════════════ */

static void test_ema_delta(void) {
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    float x[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.23156652857908377f; /* cng_y2_alpha */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(state_init, n);
            fill_floats(x, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_delta_f32_scalar(state_scalar, x, alpha, n);
            sk_ema_delta_f32(state_simd, x, alpha, n);
            check_bits_classify("ema_delta_f32", n, t, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_delta_f32\n");
}

/* ═══════════════════════════ correctness: kernel 18 ══════════════════════ */

static void test_n2_track(void) {
    float n2_init[SK_TEST_MAX_N], n2_scalar[SK_TEST_MAX_N], n2_simd[SK_TEST_MAX_N];
    float y2s[SK_TEST_MAX_N];
    int ni, t;
    const float fresh = 0.9968377223398316f;
    const float retain = 0.003162277660168411f;
    const float g_up = 1.0005000750025f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(n2_init, n);
            fill_floats(y2s, n);
            memcpy(n2_scalar, n2_init, (size_t)n * sizeof(float));
            memcpy(n2_simd, n2_init, (size_t)n * sizeof(float));
            sk_n2_track_f32_scalar(n2_scalar, y2s, fresh, retain, g_up, n);
            sk_n2_track_f32(n2_simd, y2s, fresh, retain, g_up, n);
            check_bits_classify("n2_track_f32", n, t, n2_simd, n2_scalar, n);
        }
    }
    printf("PASS n2_track_f32\n");
}

/* ═══════════════════════════ correctness: kernel 19 ══════════════════════ */

static void test_n2_initial_track(void) {
    float n2i_init[SK_TEST_MAX_N], n2i_scalar[SK_TEST_MAX_N], n2i_simd[SK_TEST_MAX_N];
    float n2[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.0024981253125391234f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(n2i_init, n);
            fill_floats(n2, n);
            memcpy(n2i_scalar, n2i_init, (size_t)n * sizeof(float));
            memcpy(n2i_simd, n2i_init, (size_t)n * sizeof(float));
            sk_n2_initial_track_f32_scalar(n2i_scalar, n2, alpha, n);
            sk_n2_initial_track_f32(n2i_simd, n2, alpha, n);
            check_bits_classify("n2_initial_track_f32", n, t, n2i_simd, n2i_scalar, n);
        }
    }
    printf("PASS n2_initial_track_f32\n");
}

/* ═══════════════════════════ correctness: kernel 20 ══════════════════════ */

static void test_mask_zero(void) {
    float x_init[SK_TEST_MAX_N], x_scalar[SK_TEST_MAX_N], x_simd[SK_TEST_MAX_N];
    unsigned char mask[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            int i;
            fill_floats(x_init, n);
            for (i = 0; i < n; ++i) mask[i] = (unsigned char)(lcg_next() & 1u);
            memcpy(x_scalar, x_init, (size_t)n * sizeof(float));
            memcpy(x_simd, x_init, (size_t)n * sizeof(float));
            sk_mask_zero_f32_scalar(x_scalar, mask, n);
            sk_mask_zero_f32(x_simd, mask, n);
            check_bits_classify("mask_zero_f32", n, t, x_simd, x_scalar, n);
        }
    }
    /* dedicated all-ones / all-zeros boundary check. */
    {
        float xa_init[8], xa_scalar[8], xa_simd[8];
        unsigned char mall1[8], mall0[8];
        int i;
        fill_floats(xa_init, 8);
        for (i = 0; i < 8; ++i) { mall1[i] = 1; mall0[i] = 0; }
        memcpy(xa_scalar, xa_init, sizeof(xa_init));
        memcpy(xa_simd, xa_init, sizeof(xa_init));
        sk_mask_zero_f32_scalar(xa_scalar, mall1, 8);
        sk_mask_zero_f32(xa_simd, mall1, 8);
        check_bits_or_die("mask_zero_f32_all1", 8, 0, xa_simd, xa_scalar, 8);
        memcpy(xa_scalar, xa_init, sizeof(xa_init));
        memcpy(xa_simd, xa_init, sizeof(xa_init));
        sk_mask_zero_f32_scalar(xa_scalar, mall0, 8);
        sk_mask_zero_f32(xa_simd, mall0, 8);
        check_bits_or_die("mask_zero_f32_all0", 8, 0, xa_simd, xa_scalar, 8);
    }
    printf("PASS mask_zero_f32\n");
}

/* ═══════════════════════════ correctness: kernel 23 ══════════════════════ */

static void test_noise_spectrum_update(void) {
    float noise_init[SK_TEST_MAX_N], noise_scalar[SK_TEST_MAX_N], noise_simd[SK_TEST_MAX_N];
    float spectrum[SK_TEST_MAX_N];
    int ni, t, m10;
    const float alpha = 0.0024981253125391234f;
    const float min_noise = 1.0e-4f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            /* apply_mask10 is a call-level (not per-lane) scalar in the real
             * caller -- exercise both values, not just one, since it selects
             * a genuinely different instruction sequence inside the rising
             * branch (see kernel 23's header comment). */
            for (m10 = 0; m10 < 2; ++m10) {
                fill_floats(noise_init, n);
                fill_floats(spectrum, n);
                memcpy(noise_scalar, noise_init, (size_t)n * sizeof(float));
                memcpy(noise_simd, noise_init, (size_t)n * sizeof(float));
                sk_noise_spectrum_update_f32_scalar(noise_scalar, spectrum, alpha, m10, min_noise, n);
                sk_noise_spectrum_update_f32(noise_simd, spectrum, alpha, m10, min_noise, n);
                check_bits_classify("noise_spectrum_update_f32", n, t * 2 + m10, noise_simd, noise_scalar, n);
            }
        }
    }
    printf("PASS noise_spectrum_update_f32\n");
}

/* ═══════════════════════════ correctness: kernel 24 ══════════════════════ */

static void test_erl_bin_update(void) {
    float erl_init[SK_TEST_MAX_N], erl_scalar[SK_TEST_MAX_N], erl_simd[SK_TEST_MAX_N];
    int hold_init[SK_TEST_MAX_N], hold_scalar[SK_TEST_MAX_N], hold_simd[SK_TEST_MAX_N];
    float x2[SK_TEST_MAX_N], y2[SK_TEST_MAX_N];
    int ni, t;
    const float x2_min = 44015068.0f;   /* ERL_X2_MIN */
    const int hold_hops = 400;          /* typical hold counter arm value */
    const float min_erl = 0.01f;        /* ERL_MIN_ERL */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(erl_init, n);
            fill_floats(x2, n);
            fill_floats(y2, n);
            fill_ints(hold_init, n);
            memcpy(erl_scalar, erl_init, (size_t)n * sizeof(float));
            memcpy(erl_simd, erl_init, (size_t)n * sizeof(float));
            memcpy(hold_scalar, hold_init, (size_t)n * sizeof(int));
            memcpy(hold_simd, hold_init, (size_t)n * sizeof(int));

            sk_erl_bin_update_f32_scalar(erl_scalar, hold_scalar, x2, y2,
                                          x2_min, hold_hops, min_erl, n);
            sk_erl_bin_update_f32(erl_simd, hold_simd, x2, y2,
                                   x2_min, hold_hops, min_erl, n);

            /* classified, not strict -- x2/y2 are drawn from the full
             * special-value pool including +-Inf/0, and the erl+delta blend
             * can legitimately produce a NaN from finite/Inf operands (e.g.
             * erl[k]==+Inf with new_erl==-Inf gives delta==-Inf, blend
             * (+Inf)+(-Inf)==NaN) -- see kernel 24's header comment. */
            check_bits_classify("erl_bin_update_f32:erl", n, t, erl_simd, erl_scalar, n);
            /* hold_counters is exact int32 -- always strict. */
            check_ints_or_die("erl_bin_update_f32:hold", n, t, hold_simd, hold_scalar, n);
        }
    }
    printf("PASS erl_bin_update_f32\n");
}

/* ═══════════════════════════ correctness: kernel 25 ══════════════════════ */

static void test_dec1_floorintmin_s32(void) {
    int x_init[SK_TEST_MAX_N], x_scalar[SK_TEST_MAX_N], x_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_ints(x_init, n);
            memcpy(x_scalar, x_init, (size_t)n * sizeof(int));
            memcpy(x_simd, x_init, (size_t)n * sizeof(int));
            sk_dec1_floorintmin_s32_scalar(x_scalar, n);
            sk_dec1_floorintmin_s32(x_simd, n);
            check_ints_or_die("dec1_floorintmin_s32", n, t, x_simd, x_scalar, n);
        }
    }
    printf("PASS dec1_floorintmin_s32\n");
}

/* ═══════════════════════════ correctness: kernel 26 ══════════════════════ */

static void test_erl_hold_expire(void) {
    float erl_init[SK_TEST_MAX_N], erl_scalar[SK_TEST_MAX_N], erl_simd[SK_TEST_MAX_N];
    int hold[SK_TEST_MAX_N];
    int ni, t;
    const float max_erl = 1000.0f; /* ERL_MAX_ERL */
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(erl_init, n);
            fill_ints(hold, n);
            memcpy(erl_scalar, erl_init, (size_t)n * sizeof(float));
            memcpy(erl_simd, erl_init, (size_t)n * sizeof(float));
            sk_erl_hold_expire_f32_scalar(erl_scalar, hold, max_erl, n);
            sk_erl_hold_expire_f32(erl_simd, hold, max_erl, n);
            check_bits_classify("erl_hold_expire_f32", n, t, erl_simd, erl_scalar, n);
        }
    }
    printf("PASS erl_hold_expire_f32\n");
}

/* ═══════════════════════════ correctness: kernel 27 ══════════════════════ */

static void test_no_audible_echo_gain(void) {
    float ne[SK_TEST_MAX_N], ec[SK_TEST_MAX_N], mk[SK_TEST_MAX_N];
    float ntr[SK_TEST_MAX_N], nsu[SK_TEST_MAX_N], nem[SK_TEST_MAX_N];
    float otr[SK_TEST_MAX_N], osu[SK_TEST_MAX_N], oem[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, t, c;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(ne, n); fill_floats(ec, n); fill_floats(mk, n);
            fill_floats(ntr, n); fill_floats(nsu, n); fill_floats(nem, n);
            fill_floats(otr, n); fill_floats(osu, n); fill_floats(oem, n);
            for (c = 0; c < SG_CFG_COUNT; ++c) {
                sk_no_audible_echo_gain_f32_scalar(ne, ec, mk, ntr, nsu, nem,
                                                    otr, osu, oem,
                                                    SG_CFG_THR[c], SG_CFG_SOFT[c],
                                                    out_scalar, n);
                sk_no_audible_echo_gain_f32(ne, ec, mk, ntr, nsu, nem,
                                             otr, osu, oem,
                                             SG_CFG_THR[c], SG_CFG_SOFT[c],
                                             out_simd, n);
                check_bits_classify("no_audible_echo_gain_f32", n, t * 10 + c,
                                     out_simd, out_scalar, n);
            }
        }
    }
    printf("PASS no_audible_echo_gain_f32 (%d cfg)\n", SG_CFG_COUNT);
}

/* ═══════════════════════ NaN corpus: cabs/cmag2 family (STRICT) ═══════════
 * sk__cabs_np_neon4 must bit-match sk__cabs_np_elem across every NaN
 * pattern x length combination. Any mismatch here means the compare/select
 * NaN contract is broken -- hard exit(1), same as the finite corpus. */

static void test_cabs_np_nan(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int pi, ni;
    for (pi = 0; pi < NAN_PATTERN_COUNT; ++pi) {
        for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
            int n = NAN_N_LIST[ni];
            char name[64];
            snprintf(name, sizeof name, "cabs_np_f32_nan:%s", NAN_PATTERN_NAMES[pi]);
            fill_complex_nan(z, n, NAN_PATTERNS[pi]);
            sk_cabs_np_f32_scalar(z, out_scalar, n);
            sk_cabs_np_f32(z, out_simd, n);
            check_bits_or_die(name, n, pi, out_simd, out_scalar, n);
        }
    }
    printf("PASS cabs_np_f32_nan\n");
}

static void test_cmag2_np_nan(void) {
    Complex z[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int pi, ni;
    for (pi = 0; pi < NAN_PATTERN_COUNT; ++pi) {
        for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
            int n = NAN_N_LIST[ni];
            char name[64];
            snprintf(name, sizeof name, "cmag2_np_f32_nan:%s", NAN_PATTERN_NAMES[pi]);
            fill_complex_nan(z, n, NAN_PATTERNS[pi]);
            sk_cmag2_np_f32_scalar(z, out_scalar, n);
            sk_cmag2_np_f32(z, out_simd, n);
            check_bits_or_die(name, n, pi, out_simd, out_scalar, n);
        }
    }
    printf("PASS cmag2_np_f32_nan\n");
}

static void test_cmag2_np_acc_nan(void) {
    Complex z[SK_TEST_MAX_N];
    float acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int pi, ni;
    for (pi = 0; pi < NAN_PATTERN_COUNT; ++pi) {
        for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
            int n = NAN_N_LIST[ni];
            char name[64];
            snprintf(name, sizeof name, "cmag2_np_acc_f32_nan:%s", NAN_PATTERN_NAMES[pi]);
            fill_complex_nan(z, n, NAN_PATTERNS[pi]);
            fill_floats(acc_init, n); /* finite accumulator seed, isolates the effect to cmag2 */
            memcpy(acc_scalar, acc_init, (size_t)n * sizeof(float));
            memcpy(acc_simd, acc_init, (size_t)n * sizeof(float));
            sk_cmag2_np_acc_f32_scalar(z, acc_scalar, n);
            sk_cmag2_np_acc_f32(z, acc_simd, n);
            check_bits_or_die(name, n, pi, acc_simd, acc_scalar, n);
        }
    }
    printf("PASS cmag2_np_acc_f32_nan\n");
}

/* ═══════════════════════ NaN corpus: every other kernel (CLASSIFIED) ═══════════════════════
 * Separate from the strict cabs/cmag2 gate above: these kernels do not use sk__cabs_np_neon4 and
 * avoid vmaxq_f32/vminq_f32/vabsq_f32 (see header). Every kernel in the file runs here;
 * check_bits_classify() sorts each divergence into bit-exact / both-NaN (in contract) / HARD FAIL,
 * and a HARD FAIL fails the build (see main()). */

static void test_wupdate_nlms_nan(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu_eff[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_complex_nan_sprinkle(X, n);
        fill_complex_nan_sprinkle(err, n);
        fill_complex(W_init, n);
        fill_floats_nan_sprinkle(mu_eff, n);
        memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
        memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
        sk_wupdate_nlms_f32_scalar(W_scalar, X, err, mu_eff, n);
        sk_wupdate_nlms_f32(W_simd, X, err, mu_eff, n);
        check_bits_classify("wupdate_nlms_f32_nan", n, 0, (const float *)W_simd, (const float *)W_scalar, 2 * n);
    }
    printf("PASS wupdate_nlms_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_wupdate_kf_nan(void) {
    Complex X[SK_TEST_MAX_N], err[SK_TEST_MAX_N];
    Complex W_init[SK_TEST_MAX_N], W_scalar[SK_TEST_MAX_N], W_simd[SK_TEST_MAX_N];
    float mu[SK_TEST_MAX_N], mu_scale[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_complex_nan_sprinkle(X, n);
        fill_complex_nan_sprinkle(err, n);
        fill_complex(W_init, n);
        fill_floats_nan_sprinkle(mu, n);
        fill_floats_nan_sprinkle(mu_scale, n);
        memcpy(W_scalar, W_init, (size_t)n * sizeof(Complex));
        memcpy(W_simd, W_init, (size_t)n * sizeof(Complex));
        sk_wupdate_kf_f32_scalar(W_scalar, X, err, mu, mu_scale, n);
        sk_wupdate_kf_f32(W_simd, X, err, mu, mu_scale, n);
        check_bits_classify("wupdate_kf_f32_nan", n, 0, (const float *)W_simd, (const float *)W_scalar, 2 * n);
    }
    printf("PASS wupdate_kf_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_pairwise_sum_nan(void) {
    float a[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(a, n);
        {
            float rs = sk_pairwise_sum_f32_scalar(a, (size_t)n);
            float rn = sk_pairwise_sum_f32(a, (size_t)n);
            check_bits_classify("pairwise_sum_f32_nan", n, 0, &rn, &rs, 1);
        }
    }
    printf("PASS pairwise_sum_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_sum_sq_pairwise_nan(void) {
    float a[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(a, n);
        {
            float rs = sk_sum_sq_pairwise_f32_scalar(a, (size_t)n);
            float rn = sk_sum_sq_pairwise_f32(a, (size_t)n);
            check_bits_classify("sum_sq_pairwise_f32_nan", n, 0, &rn, &rs, 1);
        }
    }
    printf("PASS sum_sq_pairwise_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_pairwise_sum_tailfold_nan(void) {
    float a[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(a, n);
        {
            float rs = sk_pairwise_sum_tailfold_f32_scalar(a, (size_t)n);
            float rn = sk_pairwise_sum_tailfold_f32(a, (size_t)n);
            check_bits_classify("pairwise_sum_tailfold_f32_nan", n, 0, &rn, &rs, 1);
        }
    }
    printf("PASS pairwise_sum_tailfold_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_pairwise_sum_tailfold_b_nan(void) {
    float a[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(a, n);
        {
            float rs = sk_pairwise_sum_tailfold_b_f32_scalar(a, (size_t)n);
            float rn = sk_pairwise_sum_tailfold_b_f32(a, (size_t)n);
            check_bits_classify("pairwise_sum_tailfold_b_f32_nan", n, 0, &rn, &rs, 1);
        }
    }
    printf("PASS pairwise_sum_tailfold_b_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_coherence_ema_gate_nan(void) {
    Complex echo[SK_TEST_MAX_N], near_spec[SK_TEST_MAX_N];
    float abs_echo[SK_TEST_MAX_N], abs_near[SK_TEST_MAX_N];
    float sye_re_init[SK_TEST_MAX_N], sye_im_init[SK_TEST_MAX_N];
    float syy_init[SK_TEST_MAX_N], see_init[SK_TEST_MAX_N];
    float sye_re_s[SK_TEST_MAX_N], sye_im_s[SK_TEST_MAX_N];
    float syy_s[SK_TEST_MAX_N], see_s[SK_TEST_MAX_N];
    float sye_re_n[SK_TEST_MAX_N], sye_im_n[SK_TEST_MAX_N];
    float syy_n[SK_TEST_MAX_N], see_n[SK_TEST_MAX_N];
    unsigned char mask_s[SK_TEST_MAX_N], mask_n[SK_TEST_MAX_N];
    int ni;
    const float alpha = 0.05f, threshold = 0.5f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_complex_nan_sprinkle(echo, n);
        fill_complex_nan_sprinkle(near_spec, n);
        fill_floats_nan_sprinkle(abs_echo, n);
        fill_floats_nan_sprinkle(abs_near, n);
        fill_floats(sye_re_init, n);
        fill_floats(sye_im_init, n);
        fill_floats(syy_init, n);
        fill_floats(see_init, n);
        memcpy(sye_re_s, sye_re_init, (size_t)n * sizeof(float));
        memcpy(sye_im_s, sye_im_init, (size_t)n * sizeof(float));
        memcpy(syy_s, syy_init, (size_t)n * sizeof(float));
        memcpy(see_s, see_init, (size_t)n * sizeof(float));
        memcpy(sye_re_n, sye_re_init, (size_t)n * sizeof(float));
        memcpy(sye_im_n, sye_im_init, (size_t)n * sizeof(float));
        memcpy(syy_n, syy_init, (size_t)n * sizeof(float));
        memcpy(see_n, see_init, (size_t)n * sizeof(float));
        sk_coherence_ema_gate_f32_scalar(sye_re_s, sye_im_s, syy_s, see_s,
                                          echo, near_spec, abs_echo, abs_near,
                                          alpha, threshold, mask_s, n);
        sk_coherence_ema_gate_f32(sye_re_n, sye_im_n, syy_n, see_n,
                                   echo, near_spec, abs_echo, abs_near,
                                   alpha, threshold, mask_n, n);
        check_bits_classify("coherence_ema_gate_f32_nan:sye_re", n, 0, sye_re_n, sye_re_s, n);
        check_bits_classify("coherence_ema_gate_f32_nan:sye_im", n, 0, sye_im_n, sye_im_s, n);
        check_bits_classify("coherence_ema_gate_f32_nan:syy", n, 0, syy_n, syy_s, n);
        check_bits_classify("coherence_ema_gate_f32_nan:see", n, 0, see_n, see_s, n);
        {
            int idx;
            for (idx = 0; idx < n; ++idx) {
                check_mask_classify("coherence_ema_gate_f32_nan:mask", n, idx,
                                     mask_n[idx], mask_s[idx]);
            }
        }
    }
    printf("PASS coherence_ema_gate_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_ema_delta_nan(void) {
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    float x[SK_TEST_MAX_N];
    int ni;
    const float alpha = 0.23156652857908377f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats(state_init, n);
        fill_floats_nan_sprinkle(x, n);
        memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
        memcpy(state_simd, state_init, (size_t)n * sizeof(float));
        sk_ema_delta_f32_scalar(state_scalar, x, alpha, n);
        sk_ema_delta_f32(state_simd, x, alpha, n);
        check_bits_classify("ema_delta_f32_nan", n, 0, state_simd, state_scalar, n);
    }
    printf("PASS ema_delta_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_n2_track_nan(void) {
    float n2_init[SK_TEST_MAX_N], n2_scalar[SK_TEST_MAX_N], n2_simd[SK_TEST_MAX_N];
    float y2s[SK_TEST_MAX_N];
    int ni;
    const float fresh = 0.9968377223398316f;
    const float retain = 0.003162277660168411f;
    const float g_up = 1.0005000750025f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats(n2_init, n);
        fill_floats_nan_sprinkle(y2s, n);
        memcpy(n2_scalar, n2_init, (size_t)n * sizeof(float));
        memcpy(n2_simd, n2_init, (size_t)n * sizeof(float));
        sk_n2_track_f32_scalar(n2_scalar, y2s, fresh, retain, g_up, n);
        sk_n2_track_f32(n2_simd, y2s, fresh, retain, g_up, n);
        check_bits_classify("n2_track_f32_nan", n, 0, n2_simd, n2_scalar, n);
    }
    printf("PASS n2_track_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_n2_initial_track_nan(void) {
    float n2i_init[SK_TEST_MAX_N], n2i_scalar[SK_TEST_MAX_N], n2i_simd[SK_TEST_MAX_N];
    float n2[SK_TEST_MAX_N];
    int ni;
    const float alpha = 0.0024981253125391234f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats(n2i_init, n);
        fill_floats_nan_sprinkle(n2, n);
        memcpy(n2i_scalar, n2i_init, (size_t)n * sizeof(float));
        memcpy(n2i_simd, n2i_init, (size_t)n * sizeof(float));
        sk_n2_initial_track_f32_scalar(n2i_scalar, n2, alpha, n);
        sk_n2_initial_track_f32(n2i_simd, n2, alpha, n);
        check_bits_classify("n2_initial_track_f32_nan", n, 0, n2i_simd, n2i_scalar, n);
    }
    printf("PASS n2_initial_track_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_mask_zero_nan(void) {
    float x_init[SK_TEST_MAX_N], x_scalar[SK_TEST_MAX_N], x_simd[SK_TEST_MAX_N];
    unsigned char mask[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        int i;
        fill_floats_nan_sprinkle(x_init, n);
        for (i = 0; i < n; ++i) mask[i] = (unsigned char)(lcg_next() & 1u);
        memcpy(x_scalar, x_init, (size_t)n * sizeof(float));
        memcpy(x_simd, x_init, (size_t)n * sizeof(float));
        sk_mask_zero_f32_scalar(x_scalar, mask, n);
        sk_mask_zero_f32(x_simd, mask, n);
        check_bits_classify("mask_zero_f32_nan", n, 0, x_simd, x_scalar, n);
    }
    printf("PASS mask_zero_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_noise_spectrum_update_nan(void) {
    float noise_init[SK_TEST_MAX_N], noise_scalar[SK_TEST_MAX_N], noise_simd[SK_TEST_MAX_N];
    float spectrum[SK_TEST_MAX_N];
    int ni, m10;
    const float alpha = 0.0024981253125391234f;
    const float min_noise = 1.0e-4f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        for (m10 = 0; m10 < 2; ++m10) {
            fill_floats(noise_init, n);
            fill_floats_nan_sprinkle(spectrum, n);
            memcpy(noise_scalar, noise_init, (size_t)n * sizeof(float));
            memcpy(noise_simd, noise_init, (size_t)n * sizeof(float));
            sk_noise_spectrum_update_f32_scalar(noise_scalar, spectrum, alpha, m10, min_noise, n);
            sk_noise_spectrum_update_f32(noise_simd, spectrum, alpha, m10, min_noise, n);
            check_bits_classify("noise_spectrum_update_f32_nan", n, m10, noise_simd, noise_scalar, n);
        }
    }
    printf("PASS noise_spectrum_update_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

static void test_erl_bin_update_nan(void) {
    float erl_init[SK_TEST_MAX_N], erl_scalar[SK_TEST_MAX_N], erl_simd[SK_TEST_MAX_N];
    int hold_init[SK_TEST_MAX_N], hold_scalar[SK_TEST_MAX_N], hold_simd[SK_TEST_MAX_N];
    float x2[SK_TEST_MAX_N], y2[SK_TEST_MAX_N];
    int ni;
    const float x2_min = 44015068.0f;
    const int hold_hops = 400;
    const float min_erl = 0.01f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats(erl_init, n);
        fill_floats_nan_sprinkle(x2, n);
        fill_floats_nan_sprinkle(y2, n);
        fill_ints(hold_init, n);
        memcpy(erl_scalar, erl_init, (size_t)n * sizeof(float));
        memcpy(erl_simd, erl_init, (size_t)n * sizeof(float));
        memcpy(hold_scalar, hold_init, (size_t)n * sizeof(int));
        memcpy(hold_simd, hold_init, (size_t)n * sizeof(int));

        sk_erl_bin_update_f32_scalar(erl_scalar, hold_scalar, x2, y2,
                                      x2_min, hold_hops, min_erl, n);
        sk_erl_bin_update_f32(erl_simd, hold_simd, x2, y2,
                               x2_min, hold_hops, min_erl, n);

        check_bits_classify("erl_bin_update_f32_nan:erl", n, 0, erl_simd, erl_scalar, n);
        check_ints_or_die("erl_bin_update_f32_nan:hold", n, 0, hold_simd, hold_scalar, n);
    }
    printf("PASS erl_bin_update_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

/* dec1_floorintmin_s32 has no NaN corpus test: it operates on int32 hold_counters, which have no
 * NaN/Inf, and is covered exhaustively (every special_int_pool value, INT32_MIN included) by
 * test_dec1_floorintmin_s32() and test_dec1_floorintmin_s32_edge() above. */

static void test_erl_hold_expire_nan(void) {
    float erl_init[SK_TEST_MAX_N], erl_scalar[SK_TEST_MAX_N], erl_simd[SK_TEST_MAX_N];
    int hold[SK_TEST_MAX_N];
    int ni;
    const float max_erl = 1000.0f;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(erl_init, n);
        fill_ints(hold, n);
        memcpy(erl_scalar, erl_init, (size_t)n * sizeof(float));
        memcpy(erl_simd, erl_init, (size_t)n * sizeof(float));
        sk_erl_hold_expire_f32_scalar(erl_scalar, hold, max_erl, n);
        sk_erl_hold_expire_f32(erl_simd, hold, max_erl, n);
        check_bits_classify("erl_hold_expire_f32_nan", n, 0, erl_simd, erl_scalar, n);
    }
    printf("PASS erl_hold_expire_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

/* Kernel 27's NaN pass is the one that actually pins the compare+select
 * discipline: with a NaN g_lin and a finite g_emr, the scalar ternary
 * yields g_emr while an FMAX would yield the NaN. Sprinkling NaN through
 * all nine inputs is what reaches that state. */
static void test_no_audible_echo_gain_nan(void) {
    float ne[SK_TEST_MAX_N], ec[SK_TEST_MAX_N], mk[SK_TEST_MAX_N];
    float ntr[SK_TEST_MAX_N], nsu[SK_TEST_MAX_N], nem[SK_TEST_MAX_N];
    float otr[SK_TEST_MAX_N], osu[SK_TEST_MAX_N], oem[SK_TEST_MAX_N];
    float out_scalar[SK_TEST_MAX_N], out_simd[SK_TEST_MAX_N];
    int ni, c;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(ne, n); fill_floats_nan_sprinkle(ec, n);
        fill_floats_nan_sprinkle(mk, n);
        fill_floats_nan_sprinkle(ntr, n); fill_floats_nan_sprinkle(nsu, n);
        fill_floats_nan_sprinkle(nem, n);
        fill_floats_nan_sprinkle(otr, n); fill_floats_nan_sprinkle(osu, n);
        fill_floats_nan_sprinkle(oem, n);
        for (c = 0; c < SG_CFG_COUNT; ++c) {
            sk_no_audible_echo_gain_f32_scalar(ne, ec, mk, ntr, nsu, nem,
                                                otr, osu, oem,
                                                SG_CFG_THR[c], SG_CFG_SOFT[c],
                                                out_scalar, n);
            sk_no_audible_echo_gain_f32(ne, ec, mk, ntr, nsu, nem,
                                         otr, osu, oem,
                                         SG_CFG_THR[c], SG_CFG_SOFT[c],
                                         out_simd, n);
            check_bits_classify("no_audible_echo_gain_f32_nan", n, c,
                                 out_simd, out_scalar, n);
        }
    }
    printf("PASS no_audible_echo_gain_f32_nan (soft mismatches so far: %d)\n",
           g_nan_soft_mismatch_count);
}

static void bench_coherence_ema_gate(void) {
    Complex echo[BENCH_N], near_spec[BENCH_N];
    float abs_echo[BENCH_N], abs_near[BENCH_N];
    float sye_re[BENCH_N], sye_im[BENCH_N], syy[BENCH_N], see[BENCH_N];
    unsigned char mask[BENCH_N];
    fill_bench_complex(echo, BENCH_N);
    fill_bench_complex(near_spec, BENCH_N);
    fill_bench_floats(abs_echo, BENCH_N);
    fill_bench_floats(abs_near, BENCH_N);
    fill_bench_floats(sye_re, BENCH_N);
    fill_bench_floats(sye_im, BENCH_N);
    fill_bench_floats(syy, BENCH_N);
    fill_bench_floats(see, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) {
            sk_coherence_ema_gate_f32_scalar(sye_re, sye_im, syy, see, echo, near_spec,
                                              abs_echo, abs_near, 0.05f, 0.5f, mask, BENCH_N);
            g_bench_sink += sye_re[0];
        }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) {
                sk_coherence_ema_gate_f32(sye_re, sye_im, syy, see, echo, near_spec,
                                           abs_echo, abs_near, 0.05f, 0.5f, mask, BENCH_N);
                g_bench_sink += sye_re[0];
            }
            {
                double t3 = now_ns();
                report_bench("coherence_ema_gate_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_ema_delta(void) {
    float state[BENCH_N], x[BENCH_N];
    fill_bench_floats(state, BENCH_N);
    fill_bench_floats(x, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_ema_delta_f32_scalar(state, x, 0.23f, BENCH_N); g_bench_sink += state[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_ema_delta_f32(state, x, 0.23f, BENCH_N); g_bench_sink += state[0]; }
            {
                double t3 = now_ns();
                report_bench("ema_delta_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_n2_track(void) {
    float n2[BENCH_N], y2s[BENCH_N];
    fill_bench_floats(n2, BENCH_N);
    fill_bench_floats(y2s, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_n2_track_f32_scalar(n2, y2s, 0.99f, 0.003f, 1.0005f, BENCH_N); g_bench_sink += n2[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_n2_track_f32(n2, y2s, 0.99f, 0.003f, 1.0005f, BENCH_N); g_bench_sink += n2[0]; }
            {
                double t3 = now_ns();
                report_bench("n2_track_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_n2_initial_track(void) {
    float n2i[BENCH_N], n2[BENCH_N];
    fill_bench_floats(n2i, BENCH_N);
    fill_bench_floats(n2, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_n2_initial_track_f32_scalar(n2i, n2, 0.0025f, BENCH_N); g_bench_sink += n2i[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_n2_initial_track_f32(n2i, n2, 0.0025f, BENCH_N); g_bench_sink += n2i[0]; }
            {
                double t3 = now_ns();
                report_bench("n2_initial_track_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_mask_zero(void) {
    float x[BENCH_N];
    unsigned char mask[BENCH_N];
    int i;
    fill_bench_floats(x, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) mask[i] = (unsigned char)(i & 1);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_mask_zero_f32_scalar(x, mask, BENCH_N); g_bench_sink += x[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_mask_zero_f32(x, mask, BENCH_N); g_bench_sink += x[0]; }
            {
                double t3 = now_ns();
                report_bench("mask_zero_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_noise_spectrum_update(void) {
    float noise[BENCH_N], spectrum[BENCH_N];
    fill_bench_floats(noise, BENCH_N);
    fill_bench_floats(spectrum, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_noise_spectrum_update_f32_scalar(noise, spectrum, 0.0025f, 1, 1.0e-4f, BENCH_N); g_bench_sink += noise[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_noise_spectrum_update_f32(noise, spectrum, 0.0025f, 1, 1.0e-4f, BENCH_N); g_bench_sink += noise[0]; }
            {
                double t3 = now_ns();
                report_bench("noise_spectrum_update_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_erl_bin_update(void) {
    float erl[BENCH_N], x2[BENCH_N], y2[BENCH_N];
    int hold[BENCH_N];
    int i;
    fill_bench_floats(erl, BENCH_N);
    fill_bench_floats(x2, BENCH_N);
    fill_bench_floats(y2, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) hold[i] = 0;
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_erl_bin_update_f32_scalar(erl, hold, x2, y2, 44015068.0f, 400, 0.01f, BENCH_N); g_bench_sink += erl[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_erl_bin_update_f32(erl, hold, x2, y2, 44015068.0f, 400, 0.01f, BENCH_N); g_bench_sink += erl[0]; }
            {
                double t3 = now_ns();
                report_bench("erl_bin_update_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_dec1_floorintmin_s32(void) {
    int x[BENCH_N];
    int i;
    for (i = 0; i < BENCH_N; ++i) x[i] = i;
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_dec1_floorintmin_s32_scalar(x, BENCH_N); g_bench_sink += x[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_dec1_floorintmin_s32(x, BENCH_N); g_bench_sink += x[0]; }
            {
                double t3 = now_ns();
                report_bench("dec1_floorintmin_s32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_erl_hold_expire(void) {
    float erl[BENCH_N];
    int hold[BENCH_N];
    int i;
    fill_bench_floats(erl, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) hold[i] = (i & 1) ? -1 : 1;
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_erl_hold_expire_f32_scalar(erl, hold, 1000.0f, BENCH_N); g_bench_sink += erl[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_erl_hold_expire_f32(erl, hold, 1000.0f, BENCH_N); g_bench_sink += erl[0]; }
            {
                double t3 = now_ns();
                report_bench("erl_hold_expire_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

static void bench_no_audible_echo_gain(void) {
    float ne[BENCH_N], ec[BENCH_N], mk[BENCH_N];
    float ntr[BENCH_N], nsu[BENCH_N], nem[BENCH_N];
    float otr[BENCH_N], osu[BENCH_N], oem[BENCH_N], out[BENCH_N];
    fill_bench_floats(ne, BENCH_N); fill_bench_floats(ec, BENCH_N);
    fill_bench_floats(mk, BENCH_N);
    fill_bench_floats(ntr, BENCH_N); fill_bench_floats(nsu, BENCH_N);
    fill_bench_floats(nem, BENCH_N);
    fill_bench_floats(otr, BENCH_N); fill_bench_floats(osu, BENCH_N);
    fill_bench_floats(oem, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) {
            sk_no_audible_echo_gain_f32_scalar(ne, ec, mk, ntr, nsu, nem,
                                                otr, osu, oem, 0.25f, 0.25f, out, BENCH_N);
            g_bench_sink += out[0];
        }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) {
                sk_no_audible_echo_gain_f32(ne, ec, mk, ntr, nsu, nem,
                                             otr, osu, oem, 0.25f, 0.25f, out, BENCH_N);
                g_bench_sink += out[0];
            }
            {
                double t3 = now_ns();
                report_bench("no_audible_echo_gain_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}

/* ═══════════════════════ additional kernels ═══════════════════════
 * Each test below runs, per kernel: the finite corpus (random bit patterns and the special-value
 * pool) at every length of N_LIST, the NaN-sprinkle corpus, and the offset + canary matrix (payload
 * at element offsets 1..15 of a canary-filled arena, untouched guard words checked afterwards).
 * Scalar and NEON results must be bit-identical; where a NaN can reach an output the comparison is
 * check_bits_classify, otherwise it is strict. */

#define NEW_BENCH(name, scalar_stmt, simd_stmt) do {                              \
        double t0_, t1_, t2_, t3_; int r_;                                        \
        t0_ = now_ns();                                                           \
        for (r_ = 0; r_ < BENCH_REPS; ++r_) { scalar_stmt; }                      \
        t1_ = now_ns();                                                           \
        t2_ = now_ns();                                                           \
        for (r_ = 0; r_ < BENCH_REPS; ++r_) { simd_stmt; }                        \
        t3_ = now_ns();                                                           \
        report_bench(name, (t1_ - t0_) / BENCH_REPS, (t3_ - t2_) / BENCH_REPS);   \
    } while (0)

/* The loop sk_max_abs_f32 stands for, written out, so the scalar twin has a reference
 * that does not share its code (with SIMD off the kernel is the scalar twin). */
static float ref_max_abs_loop(const float *x, int n) {
    float smax = 0.0f;
    int i;
    for (i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > smax) smax = a;
    }
    return smax;
}

static void check_max_abs_all(const char *kernel, const float *x, int n, int trial) {
    float ref = ref_max_abs_loop(x, n);
    check_scalar_bits_or_die(kernel, n, trial, sk_max_abs_f32_scalar(x, n), ref);
    check_scalar_bits_or_die(kernel, n, trial, sk_max_abs_f32(x, n), ref);
}

static void test_max_abs(void) {
    float x[SK_TEST_MAX_N];
    float *arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, t, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x, n);
            check_scalar_bits_or_die("max_abs_f32", n, t, sk_max_abs_f32(x, n), sk_max_abs_f32_scalar(x, n));
            check_max_abs_all("max_abs_f32_ref", x, n, t);
        }
    }
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(x, n);
        check_scalar_bits_or_die("max_abs_f32_nan", n, 0, sk_max_abs_f32(x, n), sk_max_abs_f32_scalar(x, n));
        check_max_abs_all("max_abs_f32_nan_ref", x, n, 0);
    }
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            edge_fill_canary_f(arena, EDGE_ARENA_LEN);
            fill_floats(arena + o, n);
            check_scalar_bits_or_die("max_abs_f32_edge", n, o,
                                      sk_max_abs_f32(arena + o, n), sk_max_abs_f32_scalar(arena + o, n));
            check_max_abs_all("max_abs_f32_edge_ref", arena + o, n, o);
            edge_check_canary_f("max_abs_f32_edge:x", arena, EDGE_ARENA_LEN, o, n);
        }
    }
    free(arena);
    printf("PASS max_abs_f32 (finite, NaN sprinkle strict, offsets 1..15 canary-guarded, written-out reference loop)\n");
}

/* Counter and PSD generator for the noise-floor tracker: counters straddle the
 * hold value so all three arms are taken, PSDs straddle the floors. */
static void fill_noise_floor_case(float *fl, int *cnt, float *psd, int n, int hold, int nan_mode) {
    int i;
    fill_floats(fl, n);
    fill_floats(psd, n);
    for (i = 0; i < n; ++i) {
        uint32_t r = lcg_next();
        if ((r & 7u) == 0u) cnt[i] = (int)gen_int32();
        else cnt[i] = (int)((r >> 8) % (hold > 0 ? 2ull * (uint64_t)hold + 3ull : 5ull)) - 1;
        if ((r & 24u) == 8u) psd[i] = fl[i];                          /* tie arm */
        else if ((r & 24u) == 16u) psd[i] = fl[i] * 0.5f;             /* below   */
        else if ((r & 24u) == 24u && fl[i] > 0.0f) psd[i] = fl[i] * 2.0f;
    }
    if (nan_mode) {
        for (i = 0; i < n; i += 3) {
            if (i % 2) fl[i] = nan_pool[i % NAN_POOL_COUNT];
            else psd[i] = nan_pool[i % NAN_POOL_COUNT];
        }
    }
}

static void test_noise_floor_track(void) {
    float fl_i[SK_TEST_MAX_N], fl_s[SK_TEST_MAX_N], fl_v[SK_TEST_MAX_N], psd[SK_TEST_MAX_N];
    int c_i[SK_TEST_MAX_N], c_s[SK_TEST_MAX_N], c_v[SK_TEST_MAX_N];
    float *fa_s = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *fa_v = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *pa = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int *ca_s = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    int *ca_v = (int *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int));
    static const int holds[] = { 0, 1, 50, 400, INT_MAX };
    static const float growths[] = { 1.0004f, 1.0f, 0.5f, 3.0e38f };
    static const float mins[] = { 1.0e-6f, 0.0f, 1.0e30f };
    int ni, t, o, hi;
    for (hi = 0; hi < (int)(sizeof holds / sizeof holds[0]); ++hi) {
        int hold = holds[hi];
        float growth = growths[hi % 4], min_floor = mins[hi % 3];
        for (ni = 0; ni < N_LIST_COUNT; ++ni) {
            int n = N_LIST[ni];
            for (t = 0; t < TRIALS; ++t) {
                fill_noise_floor_case(fl_i, c_i, psd, n, hold, 0);
                memcpy(fl_s, fl_i, (size_t)n * sizeof(float)); memcpy(fl_v, fl_i, (size_t)n * sizeof(float));
                memcpy(c_s, c_i, (size_t)n * sizeof(int));     memcpy(c_v, c_i, (size_t)n * sizeof(int));
                sk_noise_floor_track_f32_scalar(fl_s, c_s, psd, growth, min_floor, hold, n);
                sk_noise_floor_track_f32(fl_v, c_v, psd, growth, min_floor, hold, n);
                check_bits_classify("noise_floor_track_f32", n, t, fl_v, fl_s, n);
                check_ints_or_die("noise_floor_track_f32:count", n, t, c_v, c_s, n);
            }
        }
        for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
            int n = NAN_N_LIST[ni];
            fill_noise_floor_case(fl_i, c_i, psd, n, hold, 1);
            memcpy(fl_s, fl_i, (size_t)n * sizeof(float)); memcpy(fl_v, fl_i, (size_t)n * sizeof(float));
            memcpy(c_s, c_i, (size_t)n * sizeof(int));     memcpy(c_v, c_i, (size_t)n * sizeof(int));
            sk_noise_floor_track_f32_scalar(fl_s, c_s, psd, growth, min_floor, hold, n);
            sk_noise_floor_track_f32(fl_v, c_v, psd, growth, min_floor, hold, n);
            check_bits_classify("noise_floor_track_f32_nan", n, 0, fl_v, fl_s, n);
            check_ints_or_die("noise_floor_track_f32_nan:count", n, 0, c_v, c_s, n);
        }
        for (ni = 0; ni < N_LIST_COUNT; ++ni) {
            int n = N_LIST[ni];
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int po = ((o + 7) % 15) + 1;
                edge_fill_canary_f(fa_s, EDGE_ARENA_LEN); edge_fill_canary_f(fa_v, EDGE_ARENA_LEN);
                edge_fill_canary_f(pa, EDGE_ARENA_LEN);
                edge_fill_canary_i(ca_s, EDGE_ARENA_LEN); edge_fill_canary_i(ca_v, EDGE_ARENA_LEN);
                fill_noise_floor_case(fl_i, c_i, psd, n, hold, 0);
                memcpy(fa_s + o, fl_i, (size_t)n * sizeof(float)); memcpy(fa_v + o, fl_i, (size_t)n * sizeof(float));
                memcpy(ca_s + o, c_i, (size_t)n * sizeof(int));    memcpy(ca_v + o, c_i, (size_t)n * sizeof(int));
                memcpy(pa + po, psd, (size_t)n * sizeof(float));
                sk_noise_floor_track_f32_scalar(fa_s + o, ca_s + o, pa + po, growth, min_floor, hold, n);
                sk_noise_floor_track_f32(fa_v + o, ca_v + o, pa + po, growth, min_floor, hold, n);
                check_bits_classify("noise_floor_track_f32_edge", n, o, fa_v + o, fa_s + o, n);
                check_ints_or_die("noise_floor_track_f32_edge:count", n, o, ca_v + o, ca_s + o, n);
                edge_check_canary_f("noise_floor_track_f32_edge:floor_scalar", fa_s, EDGE_ARENA_LEN, o, n);
                edge_check_canary_f("noise_floor_track_f32_edge:floor_simd", fa_v, EDGE_ARENA_LEN, o, n);
                edge_check_canary_f("noise_floor_track_f32_edge:psd", pa, EDGE_ARENA_LEN, po, n);
                edge_check_canary_i("noise_floor_track_f32_edge:count_scalar", ca_s, EDGE_ARENA_LEN, o, n);
                edge_check_canary_i("noise_floor_track_f32_edge:count_simd", ca_v, EDGE_ARENA_LEN, o, n);
            }
        }
    }
    free(fa_s); free(fa_v); free(pa); free(ca_s); free(ca_v);
    printf("PASS noise_floor_track_f32 (5 hold values, finite + NaN sprinkle + offsets 1..15 canary-guarded)\n");
}

static void bench_noise_floor_track(void) {
    float fl0[BENCH_N], fl[BENCH_N], psd[BENCH_N];
    int c0[BENCH_N], c[BENCH_N];
    int i;
    fill_bench_floats(fl0, BENCH_N); fill_bench_floats(psd, BENCH_N);
    for (i = 0; i < BENCH_N; ++i) { fl0[i] = fabsf(fl0[i]) + 1.0f; psd[i] = fabsf(psd[i]) + 1.0f; c0[i] = i % 60; }
    memcpy(fl, fl0, sizeof fl); memcpy(c, c0, sizeof c);
    NEW_BENCH("noise_floor_track_f32",
              (sk_noise_floor_track_f32_scalar(fl, c, psd, 1.0004f, 1.0e-6f, 50, BENCH_N), g_bench_sink += fl[0]),
              (sk_noise_floor_track_f32(fl, c, psd, 1.0004f, 1.0e-6f, 50, BENCH_N), g_bench_sink += fl[0]));
}

static void test_raise_to(void) {
    float x_i[SK_TEST_MAX_N], lo[SK_TEST_MAX_N], x_s[SK_TEST_MAX_N], x_v[SK_TEST_MAX_N];
    float *xa_s = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *xa_v = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *la = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    int ni, t, o, i;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_floats(x_i, n);
            fill_floats(lo, n);
            for (i = 0; i < n; i += 5) lo[i] = x_i[i];                 /* ties, incl. +-0 pairs */
            for (i = 3; i < n; i += 7) lo[i] = -x_i[i];                /* sign-flipped, zero tie */
            memcpy(x_s, x_i, (size_t)n * sizeof(float)); memcpy(x_v, x_i, (size_t)n * sizeof(float));
            sk_raise_to_f32_scalar(x_s, lo, n);
            sk_raise_to_f32(x_v, lo, n);
            check_bits_classify("raise_to_f32", n, t, x_v, x_s, n);
        }
    }
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_floats_nan_sprinkle(x_i, n);
        fill_floats_nan_sprinkle(lo, n);
        for (i = 0; i < n; i += 2) lo[i] = gen_float();
        memcpy(x_s, x_i, (size_t)n * sizeof(float)); memcpy(x_v, x_i, (size_t)n * sizeof(float));
        sk_raise_to_f32_scalar(x_s, lo, n);
        sk_raise_to_f32(x_v, lo, n);
        check_bits_classify("raise_to_f32_nan", n, 0, x_v, x_s, n);
    }
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            int lo_off = ((o + 7) % 15) + 1;
            edge_fill_canary_f(xa_s, EDGE_ARENA_LEN); edge_fill_canary_f(xa_v, EDGE_ARENA_LEN);
            edge_fill_canary_f(la, EDGE_ARENA_LEN);
            fill_floats(x_i, n); fill_floats(lo, n);
            memcpy(xa_s + o, x_i, (size_t)n * sizeof(float)); memcpy(xa_v + o, x_i, (size_t)n * sizeof(float));
            memcpy(la + lo_off, lo, (size_t)n * sizeof(float));
            sk_raise_to_f32_scalar(xa_s + o, la + lo_off, n);
            sk_raise_to_f32(xa_v + o, la + lo_off, n);
            check_bits_classify("raise_to_f32_edge", n, o, xa_v + o, xa_s + o, n);
            edge_check_canary_f("raise_to_f32_edge:x_scalar", xa_s, EDGE_ARENA_LEN, o, n);
            edge_check_canary_f("raise_to_f32_edge:x_simd", xa_v, EDGE_ARENA_LEN, o, n);
            edge_check_canary_f("raise_to_f32_edge:lo", la, EDGE_ARENA_LEN, lo_off, n);
        }
    }
    free(xa_s); free(xa_v); free(la);
    printf("PASS raise_to_f32 (finite with ties and zero signs, NaN sprinkle, offsets 1..15 canary-guarded)\n");
}

static void bench_raise_to(void) {
    float x0[BENCH_N], x[BENCH_N], lo[BENCH_N];
    fill_bench_floats(x0, BENCH_N); fill_bench_floats(lo, BENCH_N);
    memcpy(x, x0, sizeof x);
    NEW_BENCH("raise_to_f32",
              (sk_raise_to_f32_scalar(x, lo, BENCH_N), g_bench_sink += x[0]),
              (sk_raise_to_f32(x, lo, BENCH_N), g_bench_sink += x[0]));
}

/* Flags carry any byte value (non-zero means set); hangovers are zero about
 * half the time and otherwise any int32, INT32_MIN and INT32_MAX included. */
static void fill_stationary_case(unsigned char *flags, int32_t *hang, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        uint32_t r = lcg_next();
        flags[i] = ((r & 3u) == 0u) ? 0 : (((r & 12u) == 0u) ? (unsigned char)(r >> 16) : 1);
        hang[i] = ((r & 0x10000u) != 0u) ? 0 : (int32_t)gen_int32();
    }
}

static void test_stationary_mask(void) {
    unsigned char f[SK_TEST_MAX_N], m_s[SK_TEST_MAX_N], m_v[SK_TEST_MAX_N];
    int32_t h[SK_TEST_MAX_N];
    unsigned char *fa = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    unsigned char *ma_s = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    unsigned char *ma_v = (unsigned char *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN);
    int32_t *ha = (int32_t *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(int32_t));
    int ni, t, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            int cs, cv, cv0;
            fill_stationary_case(f, h, n);
            memset(m_s, 0xAB, sizeof m_s); memset(m_v, 0xAB, sizeof m_v);
            cs = sk_stationary_mask_u8_scalar(f, h, m_s, n);
            cv = sk_stationary_mask_u8(f, h, m_v, n);
            cv0 = sk_stationary_mask_u8(f, h, NULL, n);
            g_total_checks += n + 2;
            if (cs != cv || cs != cv0 || memcmp(m_s, m_v, (size_t)n) != 0 ||
                sk_stationary_mask_u8_scalar(f, h, NULL, n) != cs) {
                fprintf(stderr, "MISMATCH kernel=stationary_mask_u8 n=%d trial=%d count %d/%d/%d\n", n, t, cs, cv, cv0);
                exit(1);
            }
        }
    }
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
            int ho = ((o + 7) % 15) + 1, cs, cv;
            memset(fa, 0xAA, (size_t)EDGE_ARENA_LEN);
            edge_fill_canary_b(ma_s, EDGE_ARENA_LEN); edge_fill_canary_b(ma_v, EDGE_ARENA_LEN);
            edge_fill_canary_i((int *)ha, EDGE_ARENA_LEN);
            fill_stationary_case(f, h, n);
            memcpy(fa + o, f, (size_t)n); memcpy(ha + ho, h, (size_t)n * sizeof(int32_t));
            cs = sk_stationary_mask_u8_scalar(fa + o, ha + ho, ma_s + o, n);
            cv = sk_stationary_mask_u8(fa + o, ha + ho, ma_v + o, n);
            g_total_checks += n + 1;
            if (cs != cv || memcmp(ma_s + o, ma_v + o, (size_t)n) != 0) {
                fprintf(stderr, "MISMATCH kernel=stationary_mask_u8_edge n=%d offset=%d\n", n, o);
                exit(1);
            }
            edge_check_canary_b("stationary_mask_u8_edge:mask_scalar", ma_s, EDGE_ARENA_LEN, o, n);
            edge_check_canary_b("stationary_mask_u8_edge:mask_simd", ma_v, EDGE_ARENA_LEN, o, n);
            edge_check_canary_i("stationary_mask_u8_edge:hang", (int *)ha, EDGE_ARENA_LEN, ho, n);
        }
    }
    free(fa); free(ma_s); free(ma_v); free(ha);
    printf("PASS stationary_mask_u8 (flags any byte, hangovers incl. INT32 extremes, NULL mask, offsets 1..15 canary-guarded)\n");
}

/* The loops sk_stationary_mask_u8 stands for, written out. */
static int ref_stationary_loop(const unsigned char *flags, const int32_t *hang,
                               unsigned char *mask, int n) {
    int k, count = 0;
    for (k = 0; k < n; ++k) {
        if (mask) mask[k] = (unsigned char)(flags[k] && hang[k] == 0);
        if (flags[k] && hang[k] == 0) count++;
    }
    return count;
}

/* The NEON form folds its byte accumulator after at most 255 steps of 16 bins and sums the
 * 16 lanes into an int: n >= 4096 puts 256 steps in one pass and 16 * 255 per fold exceeds
 * a byte. Random data seldom reaches either, so every bin set (flags 1 and 0xFF) and one
 * mixed pattern are checked at lengths around the fold limit against the written-out loops. */
static void test_stationary_mask_large(void) {
    static const int N_BIG[] = {16, 255, 256, 257, 4079, 4080, 4081, 4095, 4096, 4097, 4111,
                                4112, 8191, 8192, 8208, 65536, 65549};
    const int n_big = (int)(sizeof(N_BIG) / sizeof(N_BIG[0]));
    int max_n = 65549, ni, pass, k;
    unsigned char *f = (unsigned char *)malloc((size_t)max_n);
    unsigned char *m_ref = (unsigned char *)malloc((size_t)max_n);
    unsigned char *m_s = (unsigned char *)malloc((size_t)max_n);
    unsigned char *m_v = (unsigned char *)malloc((size_t)max_n);
    int32_t *h = (int32_t *)malloc((size_t)max_n * sizeof(int32_t));
    if (!f || !m_ref || !m_s || !m_v || !h) { fprintf(stderr, "alloc failed\n"); exit(1); }
    for (ni = 0; ni < n_big; ++ni) {
        int n = N_BIG[ni];
        size_t len = n > 0 ? (size_t)n : 0u;   /* a provably non-negative memset size */
        for (pass = 0; pass < 3; ++pass) {
            int cr, cs, cv, cv0;
            for (k = 0; k < n; ++k) {
                if (pass == 2) {
                    f[k] = (unsigned char)(lcg_next() & 0xFFu);
                    h[k] = ((lcg_next() & 3u) != 0u) ? 0 : (int32_t)(lcg_next() | 1u);
                } else {
                    f[k] = (unsigned char)(pass == 0 ? 0x01 : 0xFF);
                    h[k] = 0;
                }
            }
            memset(m_ref, 0xAB, len); memset(m_s, 0xAB, len); memset(m_v, 0xAB, len);
            cr = ref_stationary_loop(f, h, m_ref, n);
            cs = sk_stationary_mask_u8_scalar(f, h, m_s, n);
            cv = sk_stationary_mask_u8(f, h, m_v, n);
            cv0 = sk_stationary_mask_u8(f, h, NULL, n);
            g_total_checks += n + 3;
            if (pass < 2 && cr != n) { fprintf(stderr, "reference count %d != n=%d\n", cr, n); exit(1); }
            if (cr != cs || cr != cv || cr != cv0 || memcmp(m_ref, m_s, len) != 0 ||
                memcmp(m_ref, m_v, len) != 0) {
                fprintf(stderr,
                    "MISMATCH kernel=stationary_mask_u8_large n=%d pass=%d count ref/scalar/simd/simd-null %d/%d/%d/%d\n",
                    n, pass, cr, cs, cv, cv0);
                exit(1);
            }
        }
    }
    free(f); free(m_ref); free(m_s); free(m_v); free(h);
    printf("PASS stationary_mask_u8_large (all bins set up to n=65549, byte-fold and lane-sum limits, written-out reference loop)\n");
}

static void bench_stationary_mask(void) {
    unsigned char f[BENCH_N], m[BENCH_N];
    int32_t h[BENCH_N];
    int32_t *volatile hv = h;   /* opaque to the optimizer: the work cannot be hoisted */
    fill_stationary_case(f, h, BENCH_N);
    NEW_BENCH("stationary_mask_u8",
              (hv[r_ & 7] = (r_ & 1) ? 0 : 5, g_bench_sink += sk_stationary_mask_u8_scalar(f, hv, m, BENCH_N)),
              (hv[r_ & 7] = (r_ & 1) ? 0 : 5, g_bench_sink += sk_stationary_mask_u8(f, hv, m, BENCH_N)));
}

static void bench_max_abs(void) {
    float x[BENCH_N];
    fill_bench_floats(x, BENCH_N);
    NEW_BENCH("max_abs_f32", g_bench_sink += sk_max_abs_f32_scalar(x, BENCH_N),
                             g_bench_sink += sk_max_abs_f32(x, BENCH_N));
}

/* Inline AEC blocks: every lane boundary, real grid, canonical flag pattern,
 * threshold tie, non-finite predicate, and writable-array canary. */
static void check_inline_bytes(const char *name, int n, int trial,
                       const void *a, const void *b, size_t bytes) {
    g_total_checks += (long)bytes;
    if (memcmp(a, b, bytes)) {
        fprintf(stderr, "MISMATCH kernel=%s n=%d trial=%d\n", name, n, trial);
        ++g_hard_fail_count;
    }
}

/* Fixed answers exercise the vector body and scalar tail independently of
 * the twin implementation, including threshold ties and ordered NaN tests. */
static void test_inline_aec_oracles(void) {
    enum { N = 10 };
    for (int mode = 0; mode < 2; ++mode) {
        {
            const float initial[N] = {1,2,4,8,16,32,64,128,256,512};
            const float error[N] = {0,2,4,6,NAN,3,0,0,1,0};
            const float coarse[N] = {0,1,4,7,1,NAN,1,-1,1,1};
            const float erl[N] = {1,2,4,0,-1,8,2,4,8,2};
            const float expected[3][N] = {
                {1.5f,3,5,8.5f,17,36,64.5f,130,258,512.5f},
                {1.5f,2.5f,5,8.5f,16.5f,34,64.5f,129,258,512.5f},
                {1.25f,3,5,8,15.5f,36,64.5f,130,258,512.5f}
            };
            for (int t = 0; t < 3; ++t) {
                float h[N];
                int want_count = t == 1 ? 0 : 4;
                memcpy(h, initial, sizeof h);
                int count = mode
                    ? sk_h_error_refresh_f32(h,error,coarse,erl,0.25f,0.5f,
                                              t == 2 ? 0.0f : 2.0f,t == 1,N)
                    : sk_h_error_refresh_f32_scalar(h,error,coarse,erl,0.25f,0.5f,
                                                     t == 2 ? 0.0f : 2.0f,t == 1,N);
                check_bits_classify("h_error_fixed",N,mode,h,expected[t],N);
                check_inline_bytes("h_error_fixed_count",N,mode,
                                   &count,&want_count,sizeof count);
            }
        }
        {
            float ya[N] = {1,2,4,8,16,32,64,128,256,512};
            float ea[N] = {2,4,8,16,32,64,128,256,512,1024};
            const float y[N] = {2,4,8,16,32,64,128,256,512,1024};
            const float e[N] = {1,2,4,8,16,32,64,128,256,512};
            const float x[N] = {0,1,2,0.5f,NAN,INFINITY,-INFINITY,1,2,0};
            unsigned char low[N] = {0,1,0,1,0,0,0,0,1,0};
            const unsigned char want_low[N] = {1,1,0,1,0,0,1,0,1,1};
            const float expected[N] = {3,6,12,24,48,96,192,384,768,1536};
            if (mode) sk_erle_accumulate_f32(ya,ea,low,x,y,e,1.0f,N);
            else sk_erle_accumulate_f32_scalar(ya,ea,low,x,y,e,1.0f,N);
            check_bits_classify("erle_accumulate_y_fixed",N,mode,ya,expected,N);
            check_bits_classify("erle_accumulate_e_fixed",N,mode,ea,expected,N);
            check_inline_bytes("erle_accumulate_low_fixed",N,mode,
                               low,want_low,sizeof low);
        }
        {
            float onset[N] = {8,8,4,2,16,NAN,8,8,8,8};
            const float during[N] = {1,5,4,3,1,1,NAN,1,1,1};
            int32_t hold[N] = {4,3,2,1,0,-1,3,10,2,1};
            unsigned char coming[N] = {0,0,0,0,0,0,0,1,1,0};
            const float want_onset[N] = {8,5,4,2,8,NAN,8,8,4,4};
            const int32_t want_hold[N] = {3,2,1,0,0,0,2,9,1,0};
            const unsigned char want_coming[N] = {0,0,0,1,1,1,0,1,1,1};
            if (mode) sk_erle_release_f32(onset,during,hold,coming,2,0.5f,N);
            else sk_erle_release_f32_scalar(onset,during,hold,coming,2,0.5f,N);
            check_bits_classify("erle_release_fixed",N,mode,onset,want_onset,N);
            check_inline_bytes("erle_release_hold_fixed",N,mode,
                               hold,want_hold,sizeof hold);
            check_inline_bytes("erle_release_coming_fixed",N,mode,
                               coming,want_coming,sizeof coming);
        }
        {
            const float psd[N+2] = {1,4,1,2,8,2,NAN,8,1,4,1,0};
            int64_t counters[N];
            const int64_t expected[N] = {
                INT64_C(1099511627777),0,0,INT64_C(1099511627780),0,0,
                INT64_C(1099511627783),0,INT64_C(1099511627785),0
            };
            for (int k = 0; k < N; ++k) counters[k] = INT64_C(1099511627776)+k;
            if (mode) sk_peak_counters_i64(counters,psd,3.0f,N);
            else sk_peak_counters_i64_scalar(counters,psd,3.0f,N);
            check_inline_bytes("peak_counters_fixed",N,mode,
                               counters,expected,sizeof counters);
            float ties[N+2];
            const int64_t zero[N] = {0};
            for (int k = 0; k < N+2; ++k) ties[k] = 1.0f;
            if (mode) sk_peak_counters_i64(counters,ties,1.0f,N);
            else sk_peak_counters_i64_scalar(counters,ties,1.0f,N);
            check_inline_bytes("peak_counters_tie_fixed",N,mode,
                               counters,zero,sizeof counters);
        }
        {
            const float noise[N] = {1,1,2,2,0,-1,NAN,INFINITY,1,1};
            const float acc[N] = {7,8,15,16,0,1e-28f,0,INFINITY,-1,NAN};
            unsigned char flags[N];
            const unsigned char expected[N] = {1,0,1,0,1,0,1,0,1,0};
            if (mode) sk_stationarity_flags_u8(flags,acc,noise,2,4.0f,N);
            else sk_stationarity_flags_u8_scalar(flags,acc,noise,2,4.0f,N);
            check_inline_bytes("stationarity_flags_fixed",N,mode,
                               flags,expected,sizeof flags);
        }
    }
    /* Move the sole unset flag through EVERY lane, including the last bin. */
    const int sizes[] = {3,8,9,16,17,129,257,513};
    for (unsigned ni = 0; ni < sizeof sizes/sizeof sizes[0]; ++ni) {
        int n = sizes[ni];
        for (int missing = -1; missing < n; ++missing) {
            for (int mode = 0; mode < 2; ++mode) {
                unsigned char flags[513], want_flags[513];
                int32_t hang[513], want_hang[513];
                for (int k = 0; k < n; ++k) {
                    flags[k] = k != missing;
                    hang[k] = k % 5;
                    want_hang[k] = missing < 0 ? (k % 5 > 0 ? k % 5-1 : 0)
                                                : (k == missing ? 12 : k % 5);
                    int center = k == 0 ? 1 : k == n-1 ? n-2 : k;
                    want_flags[k] = missing < 0 || abs(center-missing) > 1;
                }
                if (mode) sk_stationarity_hangover_i32(flags,hang,12,n);
                else sk_stationarity_hangover_i32_scalar(flags,hang,12,n);
                check_inline_bytes("hangover_every_lane",n,missing,hang,
                                   want_hang,(size_t)n*sizeof *hang);
                if (mode) sk_stationarity_smooth_u8(flags,n);
                else sk_stationarity_smooth_u8_scalar(flags,n);
                check_inline_bytes("smooth_every_lane",n,missing,flags,
                                   want_flags,(size_t)n);
            }
        }
    }
    printf("PASS inline AEC fixed oracles (5 kernels, scalar/vector/tail; every unset-flag position)\n");
}

static void test_inline_aec_blocks(void) {
    enum { CAP = 530 };
    float x[CAP], y[CAP], e[CAP], z[CAP];
    float a[CAP], b[CAP], c[CAP], d[CAP];
    int32_t ha[CAP], hb[CAP];
    int64_t ca[CAP], cb[CAP];
    unsigned char fa[CAP], fb[CAP], ga[CAP], gb[CAP];
    const int grids[] = {129, 257, 513};
    const float pool[] = {0.0f, -0.0f, 1e-30f, 1e-31f, -1.0f, 0.8f,
                          0.8000001f, 1.0f, 1.0000001f, 2.0f, 4.0f,
                          INFINITY, -INFINITY, NAN};
    for (int ni = 0; ni < 39; ++ni) {
        int n = ni < 36 ? ni : grids[ni - 36];
        for (int t = 0; t < 64; ++t) {
            int off = 1 + t % 4;
            for (int k = 0; k < CAP; ++k) {
                x[k] = pool[lcg_next() % (sizeof pool / sizeof pool[0])];
                y[k] = pool[lcg_next() % (sizeof pool / sizeof pool[0])];
                e[k] = pool[lcg_next() % (sizeof pool / sizeof pool[0])];
                z[k] = pool[lcg_next() % (sizeof pool / sizeof pool[0])];
                a[k] = b[k] = 0.25f + (float)(k % 17);
                c[k] = d[k] = 0.5f + (float)(k % 11);
                fa[k] = fb[k] = (unsigned char)((k + t) % 2);
                ga[k] = gb[k] = (unsigned char)((k + t) % 3 != 0);
                ha[k] = hb[k] = (int32_t)((k + t) % 15) - 2;
                ca[k] = cb[k] = ((int64_t)1 << 40) + k;
            }
            /* Equal/error-above/error-below, floor off/on, and override. */
            for (int k = 0; k < n; ++k)
                if (t % 3 == 0) x[off + k] = y[off + k];
            int da = sk_h_error_refresh_f32(a + off, x + off, y + off, z + off,
                                            0.01f, 0.2f, t % 2 ? 0.1f : 0.0f, t % 4 == 0, n);
            int db = sk_h_error_refresh_f32_scalar(b + off, x + off, y + off, z + off,
                                                   0.01f, 0.2f, t % 2 ? 0.1f : 0.0f, t % 4 == 0, n);
            check_inline_bytes("h_error_count", n, t, &da, &db, sizeof da);
            check_bits_classify("h_error_refresh", n, t, a, b, CAP);
            sk_erle_accumulate_f32(a + off, c + off, fa + off,
                                    x + off, y + off, e + off, 1.0f, n);
            sk_erle_accumulate_f32_scalar(b + off, d + off, fb + off,
                                           x + off, y + off, e + off, 1.0f, n);
            check_bits_classify("erle_accumulate_y", n, t, a, b, CAP);
            check_bits_classify("erle_accumulate_e", n, t, c, d, CAP);
            check_inline_bytes("erle_accumulate_low", n, t, fa, fb, sizeof fa);
            sk_erle_release_f32(a + off, z + off, ha + off, ga + off, t % 8, 0.97f, n);
            sk_erle_release_f32_scalar(b + off, z + off, hb + off, gb + off, t % 8, 0.97f, n);
            check_bits_classify("erle_release", n, t, a, b, CAP);
            check_inline_bytes("erle_release_hold", n, t, ha, hb, sizeof ha);
            check_inline_bytes("erle_release_coming", n, t, ga, gb, sizeof ga);
            sk_peak_counters_i64(ca + off, x + off, 3.0f, n);
            sk_peak_counters_i64_scalar(cb + off, x + off, 3.0f, n);
            check_inline_bytes("peak_counters", n, t, ca, cb, sizeof ca);
            /* Exercise global all-stationary, isolated unset flags at every
             * vector lane, and decrement-to-zero hangover transitions. */
            for (int k = 0; k < n; ++k) {
                fa[off + k] = fb[off + k] = 1;
                ha[off + k] = hb[off + k] = (k + t) % 5;
            }
            if (n && t % 2) fa[off + t % n] = fb[off + t % n] = 0;
            sk_stationarity_hangover_i32(fa + off, ha + off, 12, n);
            sk_stationarity_hangover_i32_scalar(fb + off, hb + off, 12, n);
            check_inline_bytes("stationarity_hangover", n, t, ha, hb, sizeof ha);
            for (int k = 0; k < n; ++k) {
                int32_t expected = (k + t) % 5;
                if (t % 2 && k == t % n) expected = 12;
                if (!(t % 2) && expected > 0) --expected;
                check_inline_bytes("hangover_oracle", n, t, ha + off + k, &expected, sizeof expected);
            }
            sk_stationarity_smooth_u8(fa + off, n);
            sk_stationarity_smooth_u8_scalar(fb + off, n);
            check_inline_bytes("stationarity_smooth", n, t, fa, fb, sizeof fa);
            for (int k = 0; k < n; ++k) {
                int center = k == 0 ? 1 : k == n - 1 ? n - 2 : k;
                unsigned char expected = n >= 3;
                if (n >= 3 && t % 2 && abs(center - t % n) <= 1) expected = 0;
                check_inline_bytes("smooth_oracle", n, t, fa + off + k, &expected, sizeof expected);
            }
            sk_stationarity_flags_u8(fa + off, x + off, y + off, 5, 10.0f, n);
            sk_stationarity_flags_u8_scalar(fb + off, x + off, y + off, 5, 10.0f, n);
            check_inline_bytes("stationarity_flags", n, t, fa, fb, sizeof fa);
            /* Flat runs cross both vector blocks and their previous-sample
             * boundary. Other trials include strict threshold ties and NaN. */
            if (t % 4 == 0)
                for (int k = 0; k < n; ++k) x[off + k] = t % 8 ? 1.1f : 1.0f;
            long ac, ap, bc, bp;
            sk_saturation_counts_f32(x + off, (size_t)n, 1.0f, 0.8f, &ac, &ap);
            sk_saturation_counts_f32_scalar(x + off, (size_t)n, 1.0f, 0.8f, &bc, &bp);
            check_inline_bytes("saturation_clips", n, t, &ac, &bc, sizeof ac);
            check_inline_bytes("saturation_consecutive", n, t, &ap, &bp, sizeof ap);
            if (t % 4 == 0 && (ac != (t % 8 ? n : 0) || ap != (n > 0 ? n - 1 : 0))) {
                fprintf(stderr, "MISMATCH saturation plateau n=%d t=%d\n", n, t);
                exit(1);
            }
            ++g_total_checks;
            for (int k = 0; k < CAP; ++k) {
                if (k >= off && k < off + n) continue;
                check_scalar_bits_or_die("inline_canary_h", n, t, a[k], 0.25f + (float)(k % 17));
                check_scalar_bits_or_die("inline_canary_e", n, t, c[k], 0.5f + (float)(k % 11));
                int32_t hguard = (k + t) % 15 - 2;
                int64_t cguard = ((int64_t)1 << 40) + k;
                unsigned char fguard = (unsigned char)((k + t) % 2);
                unsigned char gguard = (unsigned char)((k + t) % 3 != 0);
                check_inline_bytes("inline_canary_hold", n, t, ha + k, &hguard, sizeof hguard);
                check_inline_bytes("inline_canary_count", n, t, ca + k, &cguard, sizeof cguard);
                check_inline_bytes("inline_canary_flags", n, t, fa + k, &fguard, sizeof fguard);
                check_inline_bytes("inline_canary_coming", n, t, ga + k, &gguard, sizeof gguard);
            }
        }
    }
    printf("PASS inline AEC blocks (8 kernels, n=0..35/129/257/513, 64 trials, offsets 1..4, canaries)\n");
}

/* ═══════════════════════════════════ main ══════════════════════════════════ */

int main(void) {
    SK_MATCH_SCALAR_TO_VECTOR_MODE();
    init_special_pool();
    init_nan_pool();
    init_special_int_pool();

    test_cabs_np();
    test_cmag2_np();
    test_cmag2_np_acc();
    test_cmac_np_ring();
    test_wupdate_nlms();
    test_wupdate_kf();
    test_pairwise_sum();
    test_sum_sq_pairwise();
    test_pairwise_sum_tailfold();
    test_pairwise_sum_tailfold_b();
    test_coherence_ema_gate();
    test_ema_delta();
    test_n2_track();
    test_n2_initial_track();
    test_mask_zero();
    test_noise_spectrum_update();
    test_erl_bin_update();
    test_dec1_floorintmin_s32();
    test_erl_hold_expire();
    test_no_audible_echo_gain();
    test_max_abs();
    test_noise_floor_track();
    test_raise_to();
    test_stationary_mask();
    test_stationary_mask_large();
    test_inline_aec_oracles();
    test_inline_aec_blocks();

    printf("\n--- NaN corpus ---\n");
    test_cabs_np_nan();
    test_cmag2_np_nan();
    test_cmag2_np_acc_nan();
    test_wupdate_nlms_nan();
    test_wupdate_kf_nan();
    test_pairwise_sum_nan();
    test_sum_sq_pairwise_nan();
    test_pairwise_sum_tailfold_nan();
    test_pairwise_sum_tailfold_b_nan();
    test_coherence_ema_gate_nan();
    test_ema_delta_nan();
    test_n2_track_nan();
    test_n2_initial_track_nan();
    test_mask_zero_nan();
    test_noise_spectrum_update_nan();
    test_erl_bin_update_nan();
    test_erl_hold_expire_nan();
    test_no_audible_echo_gain_nan();
    if (g_nan_soft_mismatch_count > 0) {
        printf("NAN SWEEP: %d mismatch(es) outside the cabs/cmag2 family "
               "-- see BOTH-NAN/HARD-FAIL lines above; classified below\n",
               g_nan_soft_mismatch_count);
    } else {
        printf("NAN SWEEP: 0 mismatches outside the cabs/cmag2 family\n");
    }
    print_classification_summary();

    printf("\n--- alignment + canary edge-case matrix ---\n");
    test_cabs_np_edge();
    test_cmag2_np_edge();
    test_cmag2_np_acc_edge();
    test_wupdate_nlms_edge();
    test_wupdate_kf_edge();
    test_pairwise_sum_edge();
    test_sum_sq_pairwise_edge();
    test_pairwise_sum_tailfold_edge();
    test_pairwise_sum_tailfold_b_edge();
    test_coherence_ema_gate_edge();
    test_ema_delta_edge();
    test_n2_track_edge();
    test_n2_initial_track_edge();
    test_mask_zero_edge();
    test_noise_spectrum_update_edge();
    test_erl_bin_update_edge();
    test_dec1_floorintmin_s32_edge();
    test_erl_hold_expire_edge();
    test_no_audible_echo_gain_edge();
    /* Second call: the table now also reflects the edge matrix's classify tallies.
     * g_tally/g_hard_fail_count are running totals for the whole process, so this printout is
     * cumulative with the first. */
    print_classification_summary();

    printf("\n--- microbenchmarks (n=%d, %d reps) ---\n", BENCH_N, BENCH_REPS);
    bench_cabs_np();
    bench_cmag2_np();
    bench_cmag2_np_acc();
    bench_wupdate_nlms();
    bench_wupdate_kf();
    bench_pairwise_sum();
    bench_sum_sq_pairwise();
    bench_pairwise_sum_tailfold();
    bench_pairwise_sum_tailfold_b();
    bench_coherence_ema_gate();
    bench_ema_delta();
    bench_n2_track();
    bench_n2_initial_track();
    bench_mask_zero();
    bench_noise_spectrum_update();
    bench_erl_bin_update();
    bench_dec1_floorintmin_s32();
    bench_erl_hold_expire();
    bench_no_audible_echo_gain();
    bench_max_abs();
    bench_noise_floor_track();
    bench_raise_to();
    bench_stationary_mask();

    if (g_hard_fail_count > 0) {
        fprintf(stderr,
            "\nGATE FAILED: %d HARD FAIL(s) -- genuine finite-vs-NaN divergence or "
            "unexplained bit mismatch outside the both-NaN-payload-unspecified "
            "contract (see HARD-FAIL lines above)\n", g_hard_fail_count);
        (void)g_bench_sink;
        return 1;
    }

    printf("\nALL PASS (SK_HAVE_NEON=%d, %d both-NaN payload-only divergences "
           "classified in-contract, 0 hard fails)\n",
           SK_HAVE_NEON, g_nan_soft_mismatch_count - g_hard_fail_count);
    printf("TOTAL CHECKS: %ld\n", g_total_checks);
    (void)g_bench_sink;
    return 0;
}
