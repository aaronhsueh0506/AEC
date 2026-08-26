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
/* Extended to n=0 (must be a zero-read/zero-write no-op,
 * see the canary section below) plus the COMPLETE 1..17 run (every kernel's
 * 4-lane NEON/scalar-tail boundary crossed at every possible remainder, not
 * just a sparse sample) -- on top of the original hand-picked lane/leaf/
 * split boundary values, which stay for their own documented reasons. */
static const int N_LIST[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    128, 129, 160, 255, 256, 257, 512
};
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

/* ═══════════════════════ int32 input generation (kernels 24-26) ═══════════
 * hold_counters is int32 -- separate generator from gen_float()'s bit-LCG
 * (an int32 has no NaN/Inf concept, so no exclusion logic is needed there),
 * but same 25%-special-pool / 75%-raw-bits shape for consistency. Formerly
 * (pre-overflow-fix) the raw-bits path steered clear of INT32_MIN because
 * kernel 25 (then `sk_dec1_s32`) unconditionally subtracted 1 from every
 * element, and INT32_MIN-1 is signed-integer-overflow UB in C -- confirmed
 * live by UBSan ("signed integer overflow: -2147483648 - 1 cannot be
 * represented in type 'int'"). Kernel 25 went through TWO fixed contracts:
 * first `sk_dec1_floor0_s32` (decrement only while >0, floored at 0), later
 * corrected to today's `sk_dec1_floorintmin_s32` (decrement only while
 * >INT_MIN, floored at INT_MIN -- the floor-at-0 form was found to desync
 * test/historical/parity_erl_estimator.c's bit-exact golden, see that kernel's header
 * comment in aec_simd_kernels.h for the full argument). Under EITHER fixed
 * contract INT32_MIN is perfectly safe as an input (it's <=0 and also
 * ==INT_MIN, so both kernels leave it unchanged, never subtract from it) --
 * so INT32_MIN stays INCLUDED in both the special pool below and the
 * raw-bits path (no remapping), exercising the exact corner the fix
 * targets; INT32_MIN+1 is also in the special pool below specifically to
 * exercise the one lane where floor-at-INT_MIN's speculative decrement
 * actually fires the hop before hitting the floor. */
#define SPECIAL_INT_POOL_COUNT 10
static int32_t special_int_pool[SPECIAL_INT_POOL_COUNT];

static void init_special_int_pool(void) {
    special_int_pool[0] = 0;
    special_int_pool[1] = 1;
    special_int_pool[2] = -1;
    special_int_pool[3] = 400;    /* ERL_HOLD_HOPS */
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
        return v;   /* INT32_MIN included -- safe under kernel 25's
                     * floor-at-INT_MIN contract, see the corpus comment
                     * above. */
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
/* n=0 prepended: a NaN-sprinkled
 * fill over zero elements is a no-op, same zero-touch contract as the
 * finite corpus's n=0 case, so it belongs in this list too. */
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

/* ═══════════════════════════ mismatch reporting ═══════════════════════════
 * g_total_checks: a running count of individual
 * bit-pattern comparisons actually performed, incremented at each of the
 * chokepoints every check funnels through (check_bits_or_die,
 * check_scalar_bits_or_die, check_bits_classify, check_mask_classify, and
 * the new canary checks below) -- so main()'s printout is a real,
 * reproducible "how much did this run actually verify" number instead of a
 * hand count. */

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

/* int32 counterpart of check_bits_or_die: hold_counters (kernels 24/25/26)
 * has no NaN/payload-ambiguity concept at all -- an int32 is exact integer
 * arithmetic, always strict bit-for-bit equality expected, no classify
 * escape hatch needed or offered. */
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

/* Kernel tally table bumped 24 -> 40. The pre-existing NaN corpus already
 * registers 16 distinct tallied names (cmac_np_f32_nan, wupdate_nlms_f32_nan,
 * wupdate_kf_f32_nan, the 4 pairwise-sum_*_nan kernels,
 * coherence_ema_gate_f32_nan's 5 sub-buffers, ema_delta_f32_nan,
 * n2_track_f32_nan, n2_initial_track_f32_nan, mask_zero_f32_nan); the new
 * edge-case matrix below registers the same 16 kernels again under a
 * "_edge" suffix (its own classify calls, a separate tally row per name) --
 * 32 total, so 24 silently overflowed (FATAL: kernel tally table full) the
 * first time this file was run after adding the edge matrix. 40 leaves
 * headroom for future additions. */
#define MAX_TALLY_KERNELS 64
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
            /* Classified, not strict: the finite corpus's special pool
             * includes +-Inf, and this kernel's contract documents NaN
             * OUTPUT payloads as unspecified (multi-operand fma/reduction
             * tie-breaks) -- an Inf-fed lane can legitimately produce NaN
             * with a codegen-dependent payload (reproduced under UBSan
             * instrumentation). Same convention as the NaN sweep and the
             * B05 edge matrix for this kernel family. */
            sk_cmac_np_f32_scalar(acc_scalar, w, x, n);
            sk_cmac_np_f32(acc_simd, w, x, n);
            check_bits_classify("cmac_np_f32", n, t, (const float *)acc_simd, (const float *)acc_scalar, 2 * n);
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

/* 0 and the complete 1..17 run prepended (same
 * rationale as N_LIST above), on top of the original boundary-specific
 * values, which stay for their own documented reasons. */
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
 * This edge-case matrix is layered on top of every per-kernel
 * correctness test above (same design as audio_common/test/simd_selftest.c's
 * equivalent section -- see that file's header comment for the full
 * rationale; summarized here for this file's kernels):
 *
 *   - n=0 / n=1..17: already covered by the extended N_LIST/NAN_N_LIST/
 *     PW_TAILFOLD_N_LIST above. For n=0 specifically the canary buffers
 *     below turn "the kernel touches nothing" from a code-inspection claim
 *     into something actually checked.
 *   - Unaligned float/Complex-element offsets 1..15: every buffer lives in
 *     its own 64-byte-aligned arena (posix_memalign); each kernel is called
 *     at a deliberately +1..+15-element offset into it (always naturally
 *     aligned for its own element type -- a whole-element offset can never
 *     construct a misaligned pointer -- but deliberately NOT 16-/64-byte
 *     aligned). Three forms per kernel with a distinct input/output role:
 *     input-offset-only, output-offset-only, both-different (via
 *     edge_offsets_for_form, shared with the common-file section, same
 *     derangement argument). Kernels with more than one read-only input
 *     (cmac_np's x, wupdate_nlms/kf's err/mu/mu_scale, coherence_ema_gate's
 *     near_spec/abs_echo/abs_near/sye_im/syy/see) keep every non-primary
 *     array fixed at offset 0 -- the finding's own wording is binary
 *     ("input offset only, output offset only, both"), documented per
 *     kernel below rather than left implicit. The four pairwise-sum-family
 *     reductions (kernels 13/14/21/22) return a float BY VALUE -- there is
 *     no output array to offset, so those four sweep only the one input
 *     array's offset (no 3-form split). sk_mask_zero_f32 (kernel 20) is a
 *     single in-place buffer (no separate output parameter at all -- see
 *     test_mask_zero_edge's own comment), same shape as sk_clip_f32's edge
 *     test in the common file.
 *   - Canary guard: every arena is entirely canary-filled before each call
 *     (float bit pattern 0x7fc0dead / mask byte 0xAA), then the payload
 *     window is overwritten with real generated data. After the call, every
 *     element OUTSIDE the payload window must still read back as the
 *     untouched canary value, in EVERY arena the kernel touched (including
 *     read-only ones, as a defense-in-depth cross-check against unexpected
 *     aliasing writes) -- catches an out-of-bounds write on either side of
 *     the payload, or literally any write at all when n==0, without a
 *     separate front/back special case.
 *
 * scalar-vs-NEON comparison is inherent to every check_bits_or_die/
 * check_scalar_bits_or_die call below, same as the rest of this file. */

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
 * near ERL_HOLD_HOPS=400 or any of the special_int_pool values), same
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

static void test_ema_cmag2_edge(void) {
    Complex *z_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    float *state_scalar_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    float *state_simd_arena = (float *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(float));
    const float alpha = 0.9f, beta = 0.1f;
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(z_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_f(state_simd_arena, EDGE_ARENA_LEN);
                fill_complex(z_arena + in_off, n);
                fill_floats(state_scalar_arena + out_off, n);
                memcpy(state_simd_arena + out_off, state_scalar_arena + out_off, (size_t)n * sizeof(float));

                sk_ema_cmag2_f32_scalar(state_scalar_arena + out_off, z_arena + in_off, alpha, beta, n);
                sk_ema_cmag2_f32(state_simd_arena + out_off, z_arena + in_off, alpha, beta, n);

                check_bits_or_die("ema_cmag2_f32_edge", n, form * 100 + o,
                                   state_simd_arena + out_off, state_scalar_arena + out_off, n);
                edge_check_canary_c("ema_cmag2_f32_edge:z", z_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_f("ema_cmag2_f32_edge:state_scalar", state_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_f("ema_cmag2_f32_edge:state_simd", state_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(z_arena); free(state_scalar_arena); free(state_simd_arena);
    printf("PASS ema_cmag2_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
}

static void test_cmac_np_edge(void) {
    Complex *w_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *x_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *acc_scalar_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    Complex *acc_simd_arena = (Complex *)edge_aligned_alloc((size_t)EDGE_ARENA_LEN * sizeof(Complex));
    int ni, form, o;
    for (ni = 0; ni < N_LIST_COUNT; ++ni) {
        int n = N_LIST[ni];
        for (form = 0; form < EDGE_FORM_COUNT; ++form) {
            for (o = 1; o <= EDGE_OFFSET_MAX; ++o) {
                int in_off, out_off;
                edge_offsets_for_form(form, o, &in_off, &out_off);

                edge_fill_canary_c(w_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(x_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(acc_scalar_arena, EDGE_ARENA_LEN);
                edge_fill_canary_c(acc_simd_arena, EDGE_ARENA_LEN);
                fill_complex(w_arena + in_off, n);
                fill_complex(x_arena, n); /* fixed at offset 0, see section header */
                fill_complex(acc_scalar_arena + out_off, n);
                memcpy(acc_simd_arena + out_off, acc_scalar_arena + out_off, (size_t)n * sizeof(Complex));

                sk_cmac_np_f32_scalar(acc_scalar_arena + out_off, w_arena + in_off, x_arena, n);
                sk_cmac_np_f32(acc_simd_arena + out_off, w_arena + in_off, x_arena, n);

                /* classified, not strict: cmac_np is one of the header's
                 * documented "NaN in -> NaN out, PAYLOAD UNSPECIFIED"
                 * kernels (multi-operand fmaf accumulation) -- matches the
                 * strict-vs-classified split the pre-existing NaN corpus
                 * already draws for this file's kernels (see this section's
                 * header comment). Empirically found necessary under UBSan:
                 * the extreme-value special pool (+-Inf) can legitimately
                 * produce a NaN from finite/Inf operands via this kernel's
                 * arithmetic, and different codegen can select a different
                 * (still valid, still in-contract) NaN payload for it. */
                check_bits_classify("cmac_np_f32_edge", n, form * 100 + o,
                                     (const float *)(acc_simd_arena + out_off),
                                     (const float *)(acc_scalar_arena + out_off), 2 * n);
                edge_check_canary_c("cmac_np_f32_edge:w", w_arena, EDGE_ARENA_LEN, in_off, n);
                edge_check_canary_c("cmac_np_f32_edge:x", x_arena, EDGE_ARENA_LEN, 0, n);
                edge_check_canary_c("cmac_np_f32_edge:acc_scalar", acc_scalar_arena, EDGE_ARENA_LEN, out_off, n);
                edge_check_canary_c("cmac_np_f32_edge:acc_simd", acc_simd_arena, EDGE_ARENA_LEN, out_off, n);
            }
        }
    }
    free(w_arena); free(x_arena); free(acc_scalar_arena); free(acc_simd_arena);
    printf("PASS cmac_np_f32_edge (n=0..17+existing x offset 1..15 x 3 forms, canary-guarded)\n");
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

                /* classified, not strict -- see cmac_np_f32_edge's comment
                 * above; wupdate_nlms is the same documented "payload
                 * unspecified" class (grad = err*conj(X), a multi-operand
                 * fmaf term). */
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

                /* classified, not strict -- see cmac_np_f32_edge's comment
                 * above. Empirically confirmed necessary: reproduced under
                 * UBSan at n=128 (this file's own pre-existing finite-corpus
                 * test_wupdate_kf() hits the identical class of divergence,
                 * unrelated to this new matrix). */
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
    /* Primary input role = echo (Complex), primary output role = sye_re
     * (float); every other buffer (sye_im/syy/see/near_spec/abs_echo/
     * abs_near/mask) stays fixed at offset 0 -- a 9-buffer kernel doesn't
     * fit the finding's binary input/output framing without a scope
     * decision, documented here. sye_im/syy/see are still read-write, so
     * each gets its own scalar/simd arena pair even though their offset
     * never varies (a shared arena would let the scalar call's mutation
     * leak into the simd call's "before" state). */
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

                /* classified, not strict: echo/near_spec are drawn from the
                 * full special-value pool including +-Inf, and
                 * pr=er*nr+ei*ni / pi=ei*nr-er*ni can legitimately produce a
                 * NaN from finite/Inf operands (e.g. (+Inf)*(-Inf) +
                 * (+Inf)*(+Inf) = -Inf+Inf = NaN) -- matches the
                 * pre-existing test_coherence_ema_gate_nan()'s own choice of
                 * check_bits_classify/check_mask_classify for this kernel
                 * above, for the same reason. */
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
                 * pool). See cmac_np_f32_edge's comment above. */
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
                 * See cmac_np_f32_edge's comment above. */
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
                 * selected). See cmac_np_f32_edge's comment above. */
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
    /* sk_mask_zero_f32 is the AEC header's other documented alias form
     * (kernel 20's "in-place" doc comment in aec_simd_kernels.h) -- unlike
     * sk_capply_gain_f32's OPTIONAL out==z aliasing, this kernel has no
     * separate output parameter at all: x is unconditionally both the read
     * source and the write destination on every single call, by signature.
     * Every call in this matrix (n=0/1..17/existing x offset 1..15) IS
     * therefore already the only meaningful "in-place" exercise for this
     * kernel -- there is no separate out-of-place form of this kernel to
     * additionally test. Single-buffer offset sweep, same
     * shape as sk_clip_f32's edge test in the common file, for the same
     * reason (no distinct input/output roles to assign the 3-form matrix
     * to). mask stays fixed at offset 0 (read-only, shared safely between
     * the scalar and simd calls) and is itself byte-canary-guarded. */
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

            /* classified for consistency with this section's other kernels
             * (matches test_mask_zero_nan()'s own choice above), though in
             * practice mask_zero can never actually produce a NaN-payload
             * divergence: it is pure pass-through/zero-select with no
             * arithmetic combination of two independent operands, so a
             * both-NaN classification should never fire here -- if it ever
             * does, that itself would be a meaningful signal. */
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
    /* Primary input role = x2, primary output role = erl; y2 (secondary
     * read-only input) and hold (secondary in-place int32 array) stay fixed
     * at offset 0 -- same >2-buffer scope decision as
     * test_coherence_ema_gate_edge above. hold is still read-write, so it
     * gets its own scalar/simd arena pair even though its offset never
     * varies (a shared arena would let the scalar call's mutation leak into
     * the simd call's "before" state). */
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
    const float x2_min = 44015068.0f;   /* ERL_AEC3_X2_MIN */
    const int hold_hops = 400;          /* ERL_HOLD_HOPS */
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
 * kernel in this file" instruction; check_bits_classify()
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

/* dec1_floorintmin_s32 has no NaN corpus test: it operates on int32 hold_counters,
 * which has no NaN/Inf concept at all -- exact integer compare-then-subtract,
 * already covered exhaustively (incl. every special_int_pool value, now
 * including INT32_MIN) by test_dec1_floorintmin_s32() and
 * test_dec1_floorintmin_s32_edge() above. */

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

/* ═══════════════════════════════════ main ══════════════════════════════════ */

int main(void) {
    init_special_pool();
    init_nan_pool();
    init_special_int_pool();

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
    test_noise_spectrum_update();
    test_erl_bin_update();
    test_dec1_floorintmin_s32();
    test_erl_hold_expire();
    test_no_audible_echo_gain();

    printf("\n--- NaN corpus ---\n");
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
    test_ema_cmag2_edge();
    test_cmac_np_edge();
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
    /* Second call: the table below now reflects the edge
     * matrix's own classify tallies too, cumulative with the first printout
     * above -- g_tally/g_hard_fail_count are running totals for the whole
     * process, not reset between calls, so this is purely an additional
     * printout, not a second independent count. */
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
    bench_noise_spectrum_update();
    bench_erl_bin_update();
    bench_dec1_floorintmin_s32();
    bench_erl_hold_expire();
    bench_no_audible_echo_gain();

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
