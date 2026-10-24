/* Numeric contract of the matched-filter kernels (da_dot_f32,
 * da_nlms_update_f32, da_nlms_update_next_dot_f32) and of the whole
 * da_matched_filter_core, on every build flavour:
 *
 *   - AArch64 NEON, and 32-bit ARM NEON with fused FMA (VFPv4): the shipped
 *     kernels must equal a portable C model of the NEON contract bit for bit
 *     (fmaf per lane, four 4-lane accumulators at stride 16, reduction tree
 *     (acc0+acc1)+(acc2+acc3) then (l0+l1)+(l2+l3)). Since AArch64 and
 *     AArch32 are both checked against the same model, they are checked
 *     against each other; --dump writes the raw results so two binaries
 *     (e.g. an armhf one under qemu and a native aarch64 one) can be compared
 *     with cmp.
 *   - every other build (host scalar, SIMD=0, 32-bit ARM without FMA, or
 *     DA_DISABLE_ARM32_NEON): the scalar kernels must equal the scalar model.
 *     The test compiles and passes there; it just exercises the scalar path.
 *
 * Bit-exactness is asserted only for finite inputs with NORMAL intermediates.
 * ARMv7 NEON flushes subnormal inputs and results to zero and returns the
 * default NaN, whereas AArch64 NEON and the C model keep subnormals and
 * propagate NaN payloads. The bit-exact corpus therefore draws operands from
 * an exponent band whose products and sums cannot become subnormal; NaN/Inf
 * cases are compared by class (NaN vs NaN, identical Inf), never by payload;
 * a separate subnormal section only reports how many results differ from the
 * IEEE model and never fails.
 *
 * Build (standalone, from c_impl/; same recipe as test_delay_kernels):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -fno-math-errno -std=gnu99 -I. \
 *       -Iinclude -I../../audiolib/common/include -I../../audiolib/common \
 *       test/test_delay_matched_neon.c src/aec_scale.c -lm \
 *       -o bin/test_delay_matched_neon
 * 32-bit ARM with fused FMA: add -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard.
 * Run:  ./bin/test_delay_matched_neon [--dump out.bin]
 */
#include "src/delay_aec.c"

#include "delay_pool_test_util.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

/* ----------------------------------------------------------------- inputs */

static unsigned long long rs = 0x9E3779B97F4A7C15ULL;
static unsigned long long rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs;
}

/* A float with a random sign and mantissa and a biased exponent in
 * [lo_exp, hi_exp]. Integer-only, so the corpus is byte-identical on every
 * architecture. */
static float band_float(int lo_exp, int hi_exp) {
    unsigned long long r = rnd();
    unsigned int sign = (unsigned int)(r & 1u) << 31;
    unsigned int e = (unsigned int)(lo_exp + (int)((r >> 1) % (unsigned long long)(hi_exp - lo_exp + 1)));
    unsigned int bits = sign | (e << 23) | (unsigned int)((r >> 20) & 0x7FFFFFu);
    float f; memcpy(&f, &bits, sizeof f); return f;
}

/* Operands in [2^-24, 2^8): products stay above 2^-48, so neither a product
 * nor any sum of up to a few thousand of them can be subnormal (the exact sum
 * of two such products is a multiple of 2^-48-46, far above the subnormal
 * range). */
#define NORMAL_LO_EXP 103
#define NORMAL_HI_EXP 134
static float nf(void) { return band_float(NORMAL_LO_EXP, NORMAL_HI_EXP); }

static float uniform_f(float scale) {
    return scale * (2.0f * ((float)(rnd() >> 40) / 16777216.0f) - 1.0f);
}

/* -------------------------------------------------------------- compare */

static long checks = 0, nan_class_only = 0;
static int fails = 0;

static int same_bits(float a, float b) { return memcmp(&a, &b, sizeof a) == 0; }

/* Bitwise equal, or both NaN (payload may differ: ARMv7 NEON default NaN). */
static int same_or_both_nan(float a, float b) {
    if (isnan(a) && isnan(b)) { if (!same_bits(a, b)) ++nan_class_only; return 1; }
    return same_bits(a, b);
}

#define EXPECT(cond, ...)                                                   \
    do {                                                                    \
        ++checks;                                                           \
        if (!(cond)) {                                                      \
            ++fails;                                                        \
            if (fails <= 12) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
        }                                                                   \
    } while (0)

/* ---------------------------------------------------------- C models */

#if DA_MF_NEON

__attribute__((noinline)) static float model_dot(const float *a, const float *b, int n) {
    float acc[16] = {0};
    int i, k;
    for (i = 0; i + 16 <= n; i += 16)
        for (k = 0; k < 16; ++k) acc[k] = fmaf(a[i + k], b[i + k], acc[k]);
    {
        float t[4], s;
        for (k = 0; k < 4; ++k) t[k] = (acc[k] + acc[4 + k]) + (acc[8 + k] + acc[12 + k]);
        s = (t[0] + t[1]) + (t[2] + t[3]);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    }
}

__attribute__((noinline)) static void model_update(float *h, const float *x, float alpha, int n) {
    int i;
    for (i = 0; i + 4 <= n; i += 4) {
        int k;
        for (k = 0; k < 4; ++k) h[i + k] = fmaf(alpha, x[i + k], h[i + k]);
    }
    for (; i < n; ++i) h[i] += alpha * x[i];
}

#else  /* scalar model: four running sums over groups of 4, one tail sum */

__attribute__((noinline)) static float model_dot(const float *a, const float *b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s;
    int i;
    for (i = 0; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    s = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

__attribute__((noinline)) static void model_update(float *h, const float *x, float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) h[i] += alpha * x[i];
}

#endif

/* ----------------------------------------------- horizontal add equivalence */

#if DA_MF_NEON
static float hadd_macro(const float l[4]) { return DA_VADDV_F32(vld1q_f32(l)); }
static float hadd_c_tree(const float l[4]) { return (l[0] + l[1]) + (l[2] + l[3]); }
/* The two-step vpadd sequence written out; on AArch64 this is the same
 * dataflow as faddp + faddp. */
static float hadd_vpadd(const float l[4]) {
    float32x4_t v = vld1q_f32(l);
    float32x2_t p = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(p, p), 0);
}
#if SK_HAVE_NEON
static float hadd_across(const float l[4]) { return vaddvq_f32(vld1q_f32(l)); }
#endif

static void check_hadd(const float l[4], const char *what) {
    float m = hadd_macro(l), c = hadd_c_tree(l), p = hadd_vpadd(l);
    EXPECT(same_or_both_nan(m, c), "hadd macro vs C tree [%s] %a %a %a %a -> %a vs %a",
           what, l[0], l[1], l[2], l[3], m, c);
    EXPECT(same_or_both_nan(m, p), "hadd macro vs vpadd [%s]", what);
#if SK_HAVE_NEON
    EXPECT(same_or_both_nan(m, hadd_across(l)), "hadd macro vs vaddvq_f32 [%s]", what);
#endif
}

static void run_hadd(void) {
    static const float special[] = { 0.0f, -0.0f, 1.0f, -1.0f, INFINITY, -INFINITY,
                                     FLT_MAX, -FLT_MAX, FLT_MIN, -FLT_MIN, 1.0e8f, -1.0e8f, 3.0f };
    const int ns = (int)(sizeof special / sizeof special[0]);
    float l[4];
    int i, j, k, m;
    /* Order discriminators: every other summation order gives a different
     * float for this lane set, so a swapped tree cannot pass. */
    /* Read through volatile so the compiler cannot constant-fold the
     * reduction: a folded vaddvq_f32 is evaluated in the compiler's own
     * order, not by the faddp instruction under test. */
    {
        volatile float probe[4] = { 1.0e8f, 1.0f, -1.0e8f, 1.0f };
        for (i = 0; i < 4; ++i) l[i] = probe[i];
    }
    EXPECT(same_bits(hadd_macro(l), 0.0f), "hadd order probe (1e8,1,-1e8,1) must be (l0+l1)+(l2+l3) = 0, got %a",
           hadd_macro(l));
    EXPECT(!same_bits((l[0] + l[2]) + (l[1] + l[3]), 0.0f), "order probe separates (l0+l2)+(l1+l3)");
    EXPECT(!same_bits(((l[0] + l[1]) + l[2]) + l[3], 0.0f), "order probe separates the linear order");
    for (i = 0; i < ns; ++i) for (j = 0; j < ns; ++j) for (k = 0; k < ns; ++k) for (m = 0; m < ns; ++m) {
        l[0] = special[i]; l[1] = special[j]; l[2] = special[k]; l[3] = special[m];
        check_hadd(l, "special");
    }
    for (i = 0; i < 20000; ++i) {
        for (j = 0; j < 4; ++j) l[j] = band_float(40, 250);
        check_hadd(l, "wide-exponent");
    }
    for (i = 0; i < 20000; ++i) {
        for (j = 0; j < 4; ++j) l[j] = band_float(120, 134);
        if (i & 1) l[2] = -l[0];   /* cancelling pair */
        check_hadd(l, "cancelling");
    }
    l[0] = NAN; l[1] = 1.0f; l[2] = 2.0f; l[3] = 3.0f;
    check_hadd(l, "nan-class");
}
#else
static void run_hadd(void) {
    printf("note: no matched-filter NEON body in this build -- horizontal-add equivalence skipped\n");
}
#endif

/* ------------------------------------------------- kernel vs model sweeps */

static void fill_normal(float *p, int n) { for (int i = 0; i < n; ++i) p[i] = nf(); }

static void run_kernel_sweep(void) {
    enum { LEN = 600 };
    static const int lens[] = { 512, 512, 512, 16, 32, 20, 4, 500, 1, 0 };
    float a[LEN + 1], b[LEN + 1], h0[LEN], h1[LEN], h2[LEN];
    int t;
    for (t = 0; t < 4000; ++t) {
        int n = lens[t % (int)(sizeof lens / sizeof lens[0])];
        int off = (t >> 3) & 1;                   /* misaligned x as in x_window - 1 */
        float alpha = nf();
        float d, m;
        fill_normal(a, LEN + 1); fill_normal(b, LEN + 1); fill_normal(h0, LEN);
        d = da_dot_f32(a, b + off, n); m = model_dot(a, b + off, n);
        EXPECT(same_bits(d, m), "dot n=%d off=%d %a vs %a", n, off, d, m);

        memcpy(h1, h0, sizeof h0); memcpy(h2, h0, sizeof h0);
        da_nlms_update_f32(h1, a + off, alpha, n);
        model_update(h2, a + off, alpha, n);
        EXPECT(memcmp(h1, h2, sizeof h1) == 0, "nlms update n=%d off=%d", n, off);

#if DA_MF_NEON
        if (n == DA_FILTER_SIZE) {                /* fused kernel is 512-tap only */
            float f, ref;
            memcpy(h1, h0, sizeof h0); memcpy(h2, h0, sizeof h0);
            f = da_nlms_update_next_dot_f32(h1, a, b + off, alpha);
            model_update(h2, a, alpha, n);
            ref = model_dot(h2, b + off, n);
            EXPECT(same_bits(f, ref), "fused update+next dot %a vs %a", f, ref);
            EXPECT(memcmp(h1, h2, (size_t)n * sizeof h1[0]) == 0, "fused update taps");
        }
#endif
    }
    /* special-value classes: ±0 operands, then NaN / Inf classification */
    {
        float z[64], w[64], d;
        for (int i = 0; i < 64; ++i) { z[i] = (i & 1) ? 0.0f : -0.0f; w[i] = nf(); }
        d = da_dot_f32(z, w, 64);
        EXPECT(same_bits(d, model_dot(z, w, 64)), "dot with +-0 operands");
        for (int i = 0; i < 64; ++i) { z[i] = nf(); w[i] = nf(); }
        z[17] = INFINITY;
        d = da_dot_f32(z, w, 64);
        EXPECT(same_or_both_nan(d, model_dot(z, w, 64)) && (isinf(d) || isnan(d)), "dot with an Inf operand");
        z[17] = NAN;
        d = da_dot_f32(z, w, 64);
        EXPECT(isnan(d) && isnan(model_dot(z, w, 64)), "dot with a NaN operand is NaN");
        z[17] = INFINITY; w[17] = 0.0f;           /* Inf * 0 */
        d = da_dot_f32(z, w, 64);
        EXPECT(isnan(d) && isnan(model_dot(z, w, 64)), "dot with Inf*0 is NaN");
        z[17] = INFINITY; w[17] = 1.0f; z[18] = -INFINITY; w[18] = 1.0f;   /* Inf - Inf */
        d = da_dot_f32(z, w, 64);
        EXPECT(isnan(d) && isnan(model_dot(z, w, 64)), "dot with Inf - Inf is NaN");
    }
}

/* ------------------------------------------------------ whole matched core */

/* Reference core: da_matched_filter_core's control flow with the model
 * kernels in place of the shipped ones. */
static int ref_core(const DaRing *ring, int shift, float threshold, float smoothing,
                    const float *y, float *h, float *error_out) {
    float span[DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1];
    float error_sum = 0.0f, x2;
    int updated = 0;
    da_ring_gather_back(ring, shift, DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1, span);
    x2 = model_dot(span + DA_SUB_BLOCK_SIZE - 1, span + DA_SUB_BLOCK_SIZE - 1, DA_FILTER_SIZE);
    for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) {
        const float *x = span + DA_SUB_BLOCK_SIZE - 1 - i;
        float e;
        int saturated;
        if (i > 0) x2 += x[0] * x[0] - x[DA_FILTER_SIZE] * x[DA_FILTER_SIZE];
        e = y[i] - model_dot(h, x, DA_FILTER_SIZE);
        saturated = y[i] >= DA_SATURATION_LIMIT || y[i] <= -DA_SATURATION_LIMIT;
        error_sum += e * e;
        if (x2 > threshold && !saturated) {
            model_update(h, x, smoothing * e / x2, DA_FILTER_SIZE);
            updated = 1;
        }
    }
    *error_out = error_sum;
    return updated;
}

static void run_core_sweep(void) {
    static float ring[DA_RING_CAPACITY_FOR(DA_NUM_FILTERS)];
    float h[DA_FILTER_SIZE], ref[DA_FILTER_SIZE], y[DA_SUB_BLOCK_SIZE];
    const float gate = (float)DA_FILTER_SIZE * DA_EXCITATION_LIMIT * DA_EXCITATION_LIMIT;
    const int cap = (int)(sizeof ring / sizeof ring[0]);
    for (int trial = 0; trial < 3000; ++trial) {
        int variant = trial % 8;
        for (int i = 0; i < cap; ++i) ring[i] = uniform_f(0.1f);
        for (int i = 0; i < DA_FILTER_SIZE; ++i) h[i] = uniform_f(0.01f);
        for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = uniform_f(0.1f);
        if (variant == 0) memset(ring, 0, sizeof ring);
        if (variant == 1) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = (i & 1) ? 1.0f : -1.0f;
        if (variant == 2) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = i == DA_SUB_BLOCK_SIZE - 1 ? 0.1f : 1.0f;
        if (variant == 3) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = i == 0 ? 0.1f : -1.0f;
        if (variant == 4) for (int i = 0; i < cap; ++i) ring[i] = (i & 1) ? 0.0f : -0.0f;
        {
            DaRing r = { .buffer = ring, .capacity = cap, .write = (trial * 71) % cap };
            int shift = (trial % DA_NUM_FILTERS) * DA_FILTER_INTRA_SHIFT;
            float threshold = variant == 7 ? 1.0e20f : gate;
            float smoothing = (trial & 1) ? DA_SMOOTHING_FAST : DA_SMOOTHING_SLOW;
            float a, b;
            int ua, ub;
            memcpy(ref, h, sizeof h);
            ua = da_matched_filter_core(&r, shift, threshold, smoothing, y, h, &a);
            ub = ref_core(&r, shift, threshold, smoothing, y, ref, &b);
            EXPECT(ua == ub && same_bits(a, b) && memcmp(h, ref, sizeof h) == 0,
                   "matched core trial %d variant %d (updated %d/%d, err %a/%a)",
                   trial, variant, ua, ub, a, b);
        }
    }
}

/* -------------------------------------------- subnormal characterisation */

/* Informational: where the hardware flushes, a dot of subnormal operands differs from the C model
 * (the model's fmaf/mul-add keep subnormals). It never fails; it only states the behaviour so a
 * reader of the log knows which side of the contract a build is on. */
static void report_subnormal_behaviour(void) {
    float a[64], b[64];
    int diff = 0, n = 200;
    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < 64; ++i) {
            unsigned int ba = (unsigned int)(rnd() & 0x007FFFFFu), bb = 0x3F800000u;
            memcpy(&a[i], &ba, sizeof a[i]); memcpy(&b[i], &bb, sizeof b[i]);
        }
        if (!same_bits(da_dot_f32(a, b, 64), model_dot(a, b, 64))) ++diff;
    }
    printf("subnormal operands (informational, not a pass/fail): %d of %d dots differ from the C model"
           " -> %s\n", diff, n,
           diff ? "operands flushed to zero (ARMv7 NEON behaviour)" : "subnormals preserved");
}

/* ----------------------------------------------------- raw dump + scenario */

static FILE *g_dump = NULL;
static void dump_f(const float *p, size_t n) { if (g_dump) fwrite(p, sizeof *p, n, g_dump); }
static void dump_i(const int *p, size_t n)   { if (g_dump) fwrite(p, sizeof *p, n, g_dump); }

static void dump_kernel_corpus(void) {
    enum { LEN = 600 };
    float a[LEN + 1], b[LEN + 1], h0[LEN], h1[LEN];
    rs = 0x1234567887654321ULL;
    for (int t = 0; t < 3000; ++t) {
        float alpha = nf(), d;
        fill_normal(a, LEN + 1); fill_normal(b, LEN + 1); fill_normal(h0, LEN);
        d = da_dot_f32(a, b + (t & 1), (t % 3) ? 512 : 20);
        dump_f(&d, 1);
        memcpy(h1, h0, sizeof h0);
        da_nlms_update_f32(h1, a + (t & 1), alpha, (t % 3) ? 512 : 500);
        dump_f(h1, LEN);
#if DA_MF_NEON
        memcpy(h1, h0, sizeof h0);
        d = da_nlms_update_next_dot_f32(h1, a, b + (t & 1), alpha);
        dump_f(&d, 1); dump_f(h1, DA_FILTER_SIZE);
#endif
    }
    {   /* matched core over the same shapes as run_core_sweep */
        static float ring[DA_RING_CAPACITY_FOR(DA_NUM_FILTERS)];
        float h[DA_FILTER_SIZE], y[DA_SUB_BLOCK_SIZE];
        const float gate = (float)DA_FILTER_SIZE * DA_EXCITATION_LIMIT * DA_EXCITATION_LIMIT;
        const int cap = (int)(sizeof ring / sizeof ring[0]);
        for (int trial = 0; trial < 2000; ++trial) {
            DaRing r = { .buffer = ring, .capacity = cap, .write = (trial * 71) % cap };
            float err; int upd;
            for (int i = 0; i < cap; ++i) ring[i] = uniform_f(0.1f);
            for (int i = 0; i < DA_FILTER_SIZE; ++i) h[i] = uniform_f(0.01f);
            for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = uniform_f(0.1f);
            upd = da_matched_filter_core(&r, (trial % DA_NUM_FILTERS) * DA_FILTER_INTRA_SHIFT, gate,
                                         (trial & 1) ? DA_SMOOTHING_FAST : DA_SMOOTHING_SLOW, y, h, &err);
            dump_i(&upd, 1); dump_f(&err, 1); dump_f(h, DA_FILTER_SIZE);
        }
    }
}

static unsigned int g_lcg = 1u;
static float lcg_next(void) {          /* exact: integer / 2^14 */
    g_lcg = g_lcg * 1103515245u + 12345u;
    return (float)((int)((g_lcg >> 9) & 0x7FFFu) - 16384) / 16384.0f;
}

/* near = gain * far[n - delay] + small independent noise; far is white noise
 * with no silent gaps, so tap values and IIR state never decay into the
 * subnormal range. Returns the last estimate. */
static int run_estimator(int sample_rate, int hop, int n_hops, int delay) {
    DelayAec d;
    const char *why = NULL;
    void *pool = delay_pool_init(&d, sample_rate, hop, DA_NUM_FILTERS, &why);
    int total = n_hops * hop + delay, last = -1;
    float *far = (float *)malloc((size_t)total * sizeof(float));
    float *near = (float *)malloc((size_t)total * sizeof(float));
    if (!pool || !far || !near) { printf("  FAIL: estimator setup (%s)\n", why ? why : "alloc"); ++fails; free(pool); free(far); free(near); return -1; }
    g_lcg = 7u + (unsigned int)sample_rate;
    for (int i = 0; i < total; ++i) far[i] = lcg_next();
    for (int i = 0; i < total; ++i) near[i] = (i >= delay ? 0.5f * far[i - delay] : 0.0f) + 0.01f * lcg_next();
    for (int i = 0; i < n_hops; ++i) {
        int cur, nu;
        float conf;
        delay_aec_accumulate_ex(&d, near + (size_t)i * hop, far + (size_t)i * hop, hop, 1);
        cur = delay_aec_estimated_delay(&d);
        conf = delay_aec_confidence(&d);
        nu = delay_aec_n_updates(&d);
        if (cur >= 0) last = cur;
        dump_i(&cur, 1); dump_f(&conf, 1); dump_i(&nu, 1);
    }
    {
        const DaMatchedFilter *mf = &d.est.matched_filter;
        dump_f(mf->filters, (size_t)mf->num_filters * DA_FILTER_SIZE);
        dump_f(mf->accumulated_error, (size_t)mf->num_filters * DA_ACC_ERR_SIZE);
    }
    free(far); free(near); free(pool);
    return last;
}

int main(int argc, char **argv) {
    int e16, e48;
    printf("test_delay_matched_neon: matched-filter NEON body = %s\n",
#if SK_HAVE_NEON
           "AArch64 NEON"
#elif DA_MF_NEON
           "32-bit ARM NEON (fused FMA)"
#else
           "none (scalar kernels)"
#endif
    );
    if (argc == 3 && strcmp(argv[1], "--dump") == 0) {
        g_dump = fopen(argv[2], "wb");
        if (!g_dump) { printf("cannot open %s\n", argv[2]); return 2; }
    }
    run_hadd();
    run_kernel_sweep();
    run_core_sweep();
    report_subnormal_behaviour();
    if (g_dump) dump_kernel_corpus();
    e16 = run_estimator(16000, 256, 900, 3008);
    e48 = run_estimator(48000, 512, 900, 9024);
    printf("estimator: 16 kHz true 3008 -> %d, 48 kHz true 9024 -> %d\n", e16, e48);
    EXPECT(abs(e16 - 3008) <= 64, "16 kHz estimator locks near the true delay");
    EXPECT(abs(e48 - 9024) <= 192, "48 kHz estimator locks near the true delay");
    if (g_dump) fclose(g_dump);
    printf("%ld checks, %d failures, %ld NaN payload-only differences (class-equal)\n",
           checks, fails, nan_class_only);
    if (fails == 0) printf("test_delay_matched_neon: PASS\n");
    return fails != 0;
}
