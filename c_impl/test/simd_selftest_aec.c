/*
 * simd_selftest_aec.c - bitwise correctness + microbenchmark harness for
 * AEC/c_impl/include/aec_simd_kernels.h.
 *
 * AEC-only counterpart of audio_common/test/simd_selftest.c: same harness
 * pattern (LCG + special-value pool input generation, memcmp-based
 * bit-exactness checks, CLOCK_MONOTONIC microbench), scoped to the kernels
 * that moved into aec_simd_kernels.h (the numpy |z|/|z|**2 magnitude family,
 * the PBFDAF/PBFDKF per-bin filter-weight updates, the AEC3-post
 * coherence/CNG per-bin trackers, and the four numpy-pairwise-sum float32
 * reduction trees). The generic-kernel tests (min/clip/gain-apply/
 * complex-add/sq-scale/EMA/fast_sqrt) stay in audio_common's own
 * simd_selftest.c.
 *
 * For every sk_<name> kernel, runs sk_<name>() (NEON when available, else
 * scalar) and sk_<name>_scalar() on IDENTICAL copies of randomly-generated
 * input (mixed LCG bit patterns + a curated special-value pool, NaN
 * excluded) across a matrix of n values and trials, and memcmp's the full
 * output (plus accumulator/state buffers where relevant) bit-for-bit. Any
 * mismatch prints the kernel/n/trial/index and the two bit patterns, then
 * exit(1)s immediately -- there is no tolerance here, this is the
 * bit-exactness gate itself.
 *
 * After correctness, runs a small microbenchmark per kernel (n=257,
 * ~200k reps, CLOCK_MONOTONIC) and prints a one-line summary.
 *
 * Review finding F10 (SIMD NaN semantics) adds a dedicated NaN corpus after
 * the finite-corpus tests above (see "NaN corpus (review F10)" section):
 * qNaN/sNaN/-NaN/+-Inf patterns in re-only/im-only/interleaved-every-3rd/
 * Inf-mixed-with-NaN combinations, at lengths spanning every kernel's NEON
 * lane-count boundary. The cabs_np/cmag2_np family (the kernels
 * sk__cabs_np_neon4's fix targets) is checked STRICT (exit(1) on mismatch,
 * same discipline as the finite corpus); every other kernel in the file is
 * checked via `check_bits_classify()`.
 *
 * Re-review finding R07 (this revision) replaces what used to be a
 * report-only "soft" check on that second group with a real classified
 * pass/fail gate. The old `check_bits_soft()` tallied a raw mismatch count
 * that main() printed but never acted on -- no CI could fail on it even if
 * the count changed. `check_bits_classify()` (see its own header comment
 * below) instead sorts every scalar-vs-NEON element divergence into one of
 * three buckets -- bit-exact / both-NaN (payload unspecified, in contract) /
 * HARD FAIL (a genuine finite-vs-NaN divergence or an unexplained finite
 * bit mismatch) -- tallies each bucket per kernel, prints a summary table,
 * and main()'s exit code is now nonzero iff ANY kernel has a HARD FAIL. The
 * pre-existing 60-mismatch baseline for this file (across cmac_np/
 * wupdate_nlms/the four pairwise-sum kernels) classifies as 100% both-NaN --
 * see the classifier's own comment for why that is in contract, not a bug.
 */
#include "aec_simd_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════════ config ═══════════════════════════════ */

#define SK_TEST_MAX_N 512
static const int N_LIST[] = {1, 3, 4, 5, 8, 9, 16, 128, 129, 160, 255, 256, 257, 512};
#define N_LIST_COUNT ((int)(sizeof(N_LIST) / sizeof(N_LIST[0])))
#define TRIALS 8

#define BENCH_N 257
#define BENCH_REPS 200000

/* ═══════════════════════════ input generation ═══════════════════════════
 * Deterministic LCG over raw uint32 bit patterns mapped to floats (25% of
 * draws), mixed with a curated special-value pool (75%... actually the
 * other way: special pool draws 25% of the time, see gen_float()). NaN
 * bit patterns from the raw-bits path are remapped to 0.0f -- NaN payload
 * propagation is explicitly out of the bit-exactness contract. */

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

/* ═══════════════════════════ NaN corpus (review F10) ═══════════════════════
 * Dedicated NaN-carrying input generation, separate from gen_float()'s
 * special_pool (which deliberately EXCLUDES NaN, see that comment above) --
 * NaN inputs are the entire point of this section. Two uses:
 *
 *   1. The cabs_np/cmag2_np family (kernels 1/2/3/5, sharing the
 *      sk__cabs_np_neon4 helper this finding fixed): full pattern x length
 *      sweep, STRICT bitwise memcmp (check_bits_or_die, same zero-tolerance
 *      discipline as the finite corpus above) -- this is the actual
 *      regression gate for the F10 fix.
 *   2. Every other kernel in the file ("W-updates/EMA etc. process arbitrary
 *      floats too"): one NaN-sprinkled run per kernel, checked with
 *      check_bits_classify (classifies each divergence as bit-exact /
 *      both-NaN-payload-unspecified / HARD FAIL rather than a blanket
 *      report-and-continue) -- these kernels were audited to already avoid
 *      vmaxq_f32/vminq_f32/vabsq_f32 (see the header's "NaN semantics"
 *      note), so a HARD FAIL here would be a NEW finding outside the cabs
 *      family; a both-NaN classification is the expected, in-contract
 *      outcome for a multi-NaN-operand reduction (fmaf/pairwise-sum trees)
 *      tie-breaking payloads differently between scalar and NEON lane
 *      order -- see check_bits_classify's own comment. */

#define NAN_POOL_COUNT 6
static float nan_pool[NAN_POOL_COUNT];

static void init_nan_pool(void) {
    nan_pool[0] = bits_to_float(0x7fc12345u); /* qNaN, custom payload (the
                                                * reviewer's repro pattern) */
    nan_pool[1] = bits_to_float(0x7fc00000u); /* qNaN, default payload */
    nan_pool[2] = bits_to_float(0x7fa00001u); /* sNaN */
    nan_pool[3] = bits_to_float(0xffc00000u); /* -NaN */
    nan_pool[4] = (float)INFINITY;
    nan_pool[5] = -(float)INFINITY;
}

/* n=1..17 straddles every kernel's 4-lane NEON/scalar-tail boundary many
 * times over; 128/129/160/255/256/257 straddle the pairwise-sum leaf/split
 * cutovers (kernel 13/14's n<=128 leaf, kernel 21/22 share the same cutover
 * for n<=257 -- their own >257 recursion is covered by the finite corpus's
 * dedicated PW_TAILFOLD_N_LIST already, no separate NaN pass needed there). */
static const int NAN_N_LIST[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
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

/* ═══════════════════════════ mismatch reporting ═══════════════════════════ */

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
    int idx = first_diff_bits(simd, scalar, count);
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
    memcpy(&gb, &simd_val, sizeof gb);
    memcpy(&wb, &scalar_val, sizeof wb);
    if (gb != wb) {
        fprintf(stderr,
            "MISMATCH kernel=%s n=%d trial=%d idx=0 simd=0x%08x (%.9g) scalar=0x%08x (%.9g)\n",
            kernel, n, trial, (unsigned)gb, (double)simd_val, (unsigned)wb, (double)scalar_val);
        exit(1);
    }
}

/* ═══════════════ NaN classification gate (re-review R07) ══════════════════
 * Upgrades the "every other kernel" NaN sweep from a report-only tally into
 * a real pass/fail gate. For every scalar-vs-NEON element compared, sorts
 * the outcome into exactly one of three buckets:
 *
 *   1. bit-exact  -- simd and scalar bits are identical. The expected
 *      outcome for the overwhelming majority of elements (only a handful of
 *      NaN-pool draws land per call; everything else is an ordinary finite
 *      value that must still match bit-for-bit).
 *   2. both-NaN   -- bits differ, but BOTH values are NaN (checked via
 *      f32_is_nan(), an exponent-all-1s/mantissa-nonzero test, NOT `v != v`
 *      -- see that function's comment for why). This is IN CONTRACT: per
 *      aec_simd_kernels.h's and simd_kernels.h's header comments, the
 *      documented per-lane contract for the arithmetic kernels here
 *      (cmac_np/wupdate_nlms/wupdate_kf/ema_delta/n2_track/
 *      n2_initial_track/mask_zero, and the four pairwise-sum-family
 *      reductions) is "NaN in -> NaN out, PAYLOAD UNSPECIFIED". A
 *      multi-NaN-operand fmaf/add/pairwise-sum-tree is free to tie-break
 *      which operand's NaN payload survives differently between the
 *      scalar C evaluation order and the NEON lane order -- C leaves
 *      multi-NaN payload selection implementation-defined, so scalar and
 *      NEON computing the "same" reduction via different instruction
 *      sequences legitimately disagreeing on WHICH NaN bit pattern comes out
 *      is not a bug: both sides correctly signal "invalid result", they just
 *      don't have to agree on which invalid-result bit pattern. Tallied, not
 *      fatal.
 *   3. HARD FAIL  -- anything else: exactly one side is NaN and the other
 *      finite/Inf (a genuine finite-vs-NaN divergence -- the class of bug
 *      the F10 cabs_np/cmag2_np fix targeted, just not yet found in one of
 *      these kernels), or both sides are finite/Inf with different bits (an
 *      ordinary bit-exactness regression that happens to have been found via
 *      the NaN corpus rather than the finite one). This is the actual
 *      contract violation this gate exists to catch -- main() returns
 *      nonzero iff any kernel's HARD FAIL count is nonzero.
 *
 * Empirically, the pre-existing 60-mismatch baseline for this file (spread
 * across cmac_np_f32, wupdate_nlms_f32, and the pairwise_sum/sum_sq_pairwise/
 * pairwise_sum_tailfold family) classifies as 100% both-NaN, 0% HARD FAIL:
 * every one of the 60 old "NAN-SWEEP-MISMATCH" lines carried two differing
 * NaN bit patterns (e.g. simd=0x7fc12345 vs scalar=0x7fc00000), never a
 * NaN-vs-finite pair. No kernel needed fixing; this gate now proves that
 * fact mechanically on every run instead of asserting it in a comment. */

typedef struct {
    char name[64];
    long bitexact;
    long both_nan;
    long fail;
} kernel_tally_t;

#define MAX_TALLY_KERNELS 24
static kernel_tally_t g_tally[MAX_TALLY_KERNELS];
static int g_tally_count = 0;
static int g_hard_fail_count = 0;
/* Legacy running total (both-NaN + HARD FAIL combined) kept only so the
 * pre-existing per-kernel "(soft mismatches so far: N)" progress prints
 * keep working unchanged; the real gate is g_hard_fail_count. */
static int g_nan_soft_mismatch_count = 0;

/* NaN test via exponent/mantissa bit pattern, NOT the `v != v` idiom: this
 * file's classification must treat a signaling NaN identically to a quiet
 * NaN (both are "NaN" for contract purposes), and on some platforms an sNaN
 * can be quieted by the mere act of loading it into a register before a C
 * comparison ever executes, which would make `v != v` unreliable for
 * distinguishing "is this bit pattern a NaN" from "did the compiler quiet it
 * first". Testing the raw bits sidesteps that entirely. */
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

/* Classifies + tallies every element of a float buffer pair (see the header
 * comment above for the three-way bit-exact/both-NaN/HARD-FAIL split). Same
 * call signature as the old check_bits_soft it replaces, so every existing
 * call site is an in-place rename. Only a HARD FAIL affects main()'s exit
 * code; both-NaN divergences are printed (for auditability) but excluded
 * from the gate. */
static void check_bits_classify(const char *kernel, int n, int trial,
                                 const float *simd, const float *scalar, int count) {
    kernel_tally_t *t = get_tally(kernel);
    int i;
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
    printf("\n--- NaN classification gate summary (re-review R07) ---\n");
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


/* ═══════════════════════════ correctness: kernel 5 ═══════════════════════ */

static void test_ema_cmag2(void) {
    Complex z[SK_TEST_MAX_N];
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    int ni, t;
    const float alpha = 0.9f, beta = 0.1f;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(z, n);
            fill_floats(state_init, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_cmag2_f32_scalar(state_scalar, z, alpha, beta, n);
            sk_ema_cmag2_f32(state_simd, z, alpha, beta, n);
            check_bits_or_die("ema_cmag2_f32", n, t, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_cmag2_f32\n");
}


/* ═══════════════════════════ correctness: kernel 6 ═══════════════════════ */

static void test_cmac_np(void) {
    Complex w[SK_TEST_MAX_N], x[SK_TEST_MAX_N];
    Complex acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int ni, t;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (t = 0; t < TRIALS; ++t) {
            fill_complex(w, n);
            fill_complex(x, n);
            fill_complex(acc_init, n);
            memcpy(acc_scalar, acc_init, (size_t)n * sizeof(Complex));
            memcpy(acc_simd, acc_init, (size_t)n * sizeof(Complex));
            sk_cmac_np_f32_scalar(acc_scalar, w, x, n);
            sk_cmac_np_f32(acc_simd, w, x, n);
            check_bits_or_die("cmac_np_f32", n, t, (const float *)acc_simd, (const float *)acc_scalar, 2 * n);
        }
    }
    printf("PASS cmac_np_f32\n");
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
            check_bits_or_die("wupdate_nlms_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
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
            check_bits_or_die("wupdate_kf_f32", n, t, (const float *)W_simd, (const float *)W_scalar, 2 * n);
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
                check_scalar_bits_or_die("pairwise_sum_f32", n, t, rn, rs);
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
                check_scalar_bits_or_die("sum_sq_pairwise_f32", n, t, rn, rs);
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

#define PW_TAILFOLD_MAX_N 960
static const int PW_TAILFOLD_N_LIST[] = {1, 7, 8, 9, 127, 128, 129, 160, 255, 256, 257, 512, 960};
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
                check_scalar_bits_or_die("pairwise_sum_tailfold_f32", n, t, rn, rs);
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
                check_scalar_bits_or_die("pairwise_sum_tailfold_b_f32", n, t, rn, rs);
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


static void bench_ema_cmag2(void) {
    Complex z[BENCH_N]; float state[BENCH_N];
    fill_bench_complex(z, BENCH_N);
    fill_bench_floats(state, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_ema_cmag2_f32_scalar(state, z, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_ema_cmag2_f32(state, z, 0.9f, 0.1f, BENCH_N); g_bench_sink += state[0]; }
            {
                double t3 = now_ns();
                report_bench("ema_cmag2_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
            }
        }
    }
}


static void bench_cmac_np(void) {
    Complex w[BENCH_N], x[BENCH_N], acc[BENCH_N];
    fill_bench_complex(w, BENCH_N);
    fill_bench_complex(x, BENCH_N);
    fill_bench_complex(acc, BENCH_N);
    {
        double t0, t1; int r;
        t0 = now_ns();
        for (r = 0; r < BENCH_REPS; ++r) { sk_cmac_np_f32_scalar(acc, w, x, BENCH_N); g_bench_sink += acc[0].r; }
        t1 = now_ns();
        {
            double ns_scalar = (t1 - t0) / BENCH_REPS;
            double t2 = now_ns();
            for (r = 0; r < BENCH_REPS; ++r) { sk_cmac_np_f32(acc, w, x, BENCH_N); g_bench_sink += acc[0].r; }
            {
                double t3 = now_ns();
                report_bench("cmac_np_f32", ns_scalar, (t3 - t2) / BENCH_REPS);
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
            check_bits_or_die("coherence_ema_gate_f32:sye_re", n, t, sye_re_n, sye_re_s, n);
            check_bits_or_die("coherence_ema_gate_f32:sye_im", n, t, sye_im_n, sye_im_s, n);
            check_bits_or_die("coherence_ema_gate_f32:syy", n, t, syy_n, syy_s, n);
            check_bits_or_die("coherence_ema_gate_f32:see", n, t, see_n, see_s, n);
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
            check_bits_or_die("ema_delta_f32", n, t, state_simd, state_scalar, n);
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
            check_bits_or_die("n2_track_f32", n, t, n2_simd, n2_scalar, n);
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
            check_bits_or_die("n2_initial_track_f32", n, t, n2i_simd, n2i_scalar, n);
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
            check_bits_or_die("mask_zero_f32", n, t, x_simd, x_scalar, n);
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

/* ═══════════════════════ NaN corpus: cabs/cmag2 family (STRICT) ═══════════
 * The actual F10 regression gate: sk__cabs_np_neon4 must bit-match
 * sk__cabs_np_elem across every NaN pattern x length combination. Any
 * mismatch here means the fix regressed or is incomplete -- hard exit(1),
 * same as the finite corpus. */

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

static void test_ema_cmag2_nan(void) {
    Complex z[SK_TEST_MAX_N];
    float state_init[SK_TEST_MAX_N], state_scalar[SK_TEST_MAX_N], state_simd[SK_TEST_MAX_N];
    int pi, ni;
    const float alpha = 0.9f, beta = 0.1f;
    for (pi = 0; pi < NAN_PATTERN_COUNT; ++pi) {
        for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
            int n = NAN_N_LIST[ni];
            char name[64];
            snprintf(name, sizeof name, "ema_cmag2_f32_nan:%s", NAN_PATTERN_NAMES[pi]);
            fill_complex_nan(z, n, NAN_PATTERNS[pi]);
            fill_floats(state_init, n);
            memcpy(state_scalar, state_init, (size_t)n * sizeof(float));
            memcpy(state_simd, state_init, (size_t)n * sizeof(float));
            sk_ema_cmag2_f32_scalar(state_scalar, z, alpha, beta, n);
            sk_ema_cmag2_f32(state_simd, z, alpha, beta, n);
            check_bits_or_die(name, n, pi, state_simd, state_scalar, n);
        }
    }
    printf("PASS ema_cmag2_f32_nan\n");
}

/* ═══════════════ NaN corpus: every other kernel (CLASSIFIED, R07) ═════════
 * Not part of the original F10 fix's own regression gate -- these kernels
 * don't use sk__cabs_np_neon4 and were audited to already avoid
 * vmaxq_f32/vminq_f32/vabsq_f32 (see header). Run per the review's "every
 * kernel in this file" instruction; check_bits_classify() (re-review R07)
 * sorts each divergence into bit-exact / both-NaN (in contract) / HARD FAIL,
 * and a HARD FAIL here now actually fails the build (see main()) instead of
 * only being printed for a human to notice. */

static void test_cmac_np_nan(void) {
    Complex w[SK_TEST_MAX_N], x[SK_TEST_MAX_N];
    Complex acc_init[SK_TEST_MAX_N], acc_scalar[SK_TEST_MAX_N], acc_simd[SK_TEST_MAX_N];
    int ni;
    for (ni = 0; ni < NAN_N_LIST_COUNT; ++ni) {
        int n = NAN_N_LIST[ni];
        fill_complex_nan_sprinkle(w, n);
        fill_complex_nan_sprinkle(x, n);
        fill_complex(acc_init, n);
        memcpy(acc_scalar, acc_init, (size_t)n * sizeof(Complex));
        memcpy(acc_simd, acc_init, (size_t)n * sizeof(Complex));
        sk_cmac_np_f32_scalar(acc_scalar, w, x, n);
        sk_cmac_np_f32(acc_simd, w, x, n);
        check_bits_classify("cmac_np_f32_nan", n, 0, (const float *)acc_simd, (const float *)acc_scalar, 2 * n);
    }
    printf("PASS cmac_np_f32_nan (soft mismatches so far: %d)\n", g_nan_soft_mismatch_count);
}

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

/* ═══════════════════════════════════ main ══════════════════════════════════ */

int main(void) {
    init_special_pool();
    init_nan_pool();

    test_cabs_np();
    test_cmag2_np();
    test_cmag2_np_acc();
    test_ema_cmag2();
    test_cmac_np();
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

    printf("\n--- NaN corpus (review F10) ---\n");
    test_cabs_np_nan();
    test_cmag2_np_nan();
    test_cmag2_np_acc_nan();
    test_ema_cmag2_nan();
    test_cmac_np_nan();
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
    if (g_nan_soft_mismatch_count > 0) {
        printf("NAN SWEEP: %d mismatch(es) outside the cabs/cmag2 family "
               "-- see BOTH-NAN/HARD-FAIL lines above; classified below\n",
               g_nan_soft_mismatch_count);
    } else {
        printf("NAN SWEEP: 0 mismatches outside the cabs/cmag2 family\n");
    }
    print_classification_summary();

    printf("\n--- microbenchmarks (n=%d, %d reps) ---\n", BENCH_N, BENCH_REPS);
    bench_cabs_np();
    bench_cmag2_np();
    bench_cmag2_np_acc();
    bench_ema_cmag2();
    bench_cmac_np();
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
    (void)g_bench_sink;
    return 0;
}
