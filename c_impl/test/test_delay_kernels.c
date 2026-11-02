/* Kernel and state equivalence gates for delay_aec.c.
 * The argmax's tie-order semantics are invisible to every other check in the repo: real audio never produces an exact h^2 tie
 * inside a 16-lane group, so the E2E dumps and the delay golden pass even with last-match-wins semantics; only this sweep
 * fails. The decimator half proves the register-specialized cascade against the generic walk across carried IIR state, which
 * a per-block re-zeroing comparison would not. The two-lane decimator and 48 kHz resampler kernels are checked against their
 * single-channel references the same way.
 *
 * On builds without the NEON body (SK_HAVE_NEON == 0: no NEON/FMA target, or SIMD_KERNELS_FORCE_SCALAR) the argmax half compares the scalar walk
 * against itself, and the two pair sweeps compare the single-call fallback (the only body that build has) against the generic
 * walks; they still run, and say so, so one `make test-delay-kernels` invocation is valid everywhere. The coefficient-equality
 * helper (da_decimator_same_coefficients) exists only in the NEON body, so the decimator pair sweep checks its verdict per
 * variant only there. The matched-core sweep checks the shipped core against separate update and dot passes, including
 * first-only, last-only and no-update gates; test_delay_matched_neon holds the kernels' bit-exact contract.
 * Includes the SHIPPED translation unit so the sweeps exercise real source. */
#include "src/delay_aec.c"
#include "utility/simd_test_mode.h"

#include <stdio.h>

#define SWEEP_HAS_NEON_BODY SK_HAVE_NEON


#define NMAX 600
static unsigned long long rs = 0x243F6A8885A308D3ULL;
static unsigned long long rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs;
}

static long cases = 0, neon_body_cases = 0, tie_cases = 0, fails = 0;

static void check(const char *what, const float *h, int n) {
    int a = da_max_square_peak_index(h, n);
    int b = sk_argmax_sq_f32_scalar(h, n);
    int i, hits = 0;
    float m;
    cases++;
    if (n >= 16) neon_body_cases++;
    if (n >= 2) {
        m = h[b] * h[b];
        for (i = 0; i < n; ++i) if (h[i] * h[i] == m) hits++;
        if (hits > 1) tie_cases++;
    }
    if (a != b) {
        fails++;
        if (fails <= 10)
            printf("  MISMATCH [%s] n=%d neon=%d scalar=%d\n", what, n, a, b);
    }
}

/* a float built from a random bit pattern inside a chosen exponent band */
static float band_float(int lo_exp, int hi_exp) {
    unsigned long long r = rnd();
    unsigned int sign = (unsigned int)(r & 1u) << 31;
    unsigned int exp = (unsigned int)(lo_exp + (int)((r >> 1) % (unsigned long long)(hi_exp - lo_exp + 1)));
    unsigned int man = (unsigned int)((r >> 20) & 0x7FFFFFu);
    unsigned int bits = sign | (exp << 23) | man;
    float f; memcpy(&f, &bits, sizeof f); return f;
}

static int run_argmax_sweep(void) {
    float h[NMAX];
    int i, k, n, trial;
    const int tail_ns[] = { 2, 3, 5, 15, 16, 17, 31, 63, 65, 127, 511, 513, 515 };

    /* ---- regime 1: broad random magnitudes, n = 512 (the shipped length) */
    for (trial = 0; trial < 400; ++trial) {
        for (i = 0; i < 512; ++i) h[i] = band_float(60, 180);
        check("random-wide", h, 512);
    }
    /* ---- regime 2: dense ties -- values from a 3-symbol alphabet, and
     * sign-flipped pairs (x and -x square identically)                    */
    for (trial = 0; trial < 400; ++trial) {
        float a = band_float(100, 140), b = a * 0.5f;
        for (i = 0; i < 512; ++i) {
            unsigned long long r = rnd() % 4;
            h[i] = (r == 0) ? a : (r == 1) ? -a : (r == 2) ? b : 0.0f;
        }
        check("dense-ties", h, 512);
    }
    /* ---- regime 3: denormals (and denormal ties)                        */
    for (trial = 0; trial < 300; ++trial) {
        for (i = 0; i < 512; ++i) {
            unsigned int bits = (unsigned int)(rnd() & 0x007FFFFFu) |
                                (unsigned int)((rnd() & 1u) << 31);
            memcpy(&h[i], &bits, sizeof h[i]);
        }
        check("denormal", h, 512);
    }
    /* ---- regime 4: denormals mixed with a single normal peak            */
    for (trial = 0; trial < 200; ++trial) {
        for (i = 0; i < 512; ++i) {
            unsigned int bits = (unsigned int)(rnd() & 0x007FFFFFu);
            memcpy(&h[i], &bits, sizeof h[i]);
        }
        h[(int)(rnd() % 512)] = band_float(120, 130);
        check("denormal+peak", h, 512);
    }
    /* ---- regime 5: all-equal windows (every index ties)                 */
    for (trial = 0; trial < 60; ++trial) {
        float v = (trial == 0) ? 0.0f : (trial == 1) ? -0.0f : band_float(1, 250);
        for (i = 0; i < 512; ++i) h[i] = v;
        check("all-equal", h, 512);
    }
    /* ---- regime 6: peak placed at every lane of the first and last NEON
     * group, and at lanes 0/1/last of the window, each with a tie behind it */
    for (k = 0; k < 512; ++k) {
        float base = 1.0f;
        for (i = 0; i < 512; ++i) h[i] = (i % 3 == 0) ? base : -base;
        h[k] = 4.0f;                 /* unique peak at k */
        check("unique-peak-at-k", h, 512);
        h[(k + 1) % 512] = -4.0f;    /* exact tie one lane later */
        check("tie-pair-at-k", h, 512);
    }
    /* ---- regime 7: ties pinned at lanes 0/1/last                        */
    for (trial = 0; trial < 60; ++trial) {
        for (i = 0; i < 512; ++i) h[i] = band_float(1, 120);
        h[0] = 3.0f; h[1] = -3.0f; h[511] = 3.0f;
        check("tie-0-1-last", h, 512);
        h[0] = 0.0f;
        check("tie-1-last", h, 512);
    }
    /* ---- regime 8: non-finite inputs (must take the scalar fallback)    */
    {
        float inf = 1.0f / 0.0f, nan = inf - inf;
        for (trial = 0; trial < 60; ++trial) {
            for (i = 0; i < 512; ++i) h[i] = band_float(100, 140);
            h[(int)(rnd() % 512)] = nan;
            check("with-nan", h, 512);
            h[(int)(rnd() % 512)] = inf;
            check("with-nan+inf", h, 512);
            for (i = 0; i < 512; ++i) h[i] = band_float(100, 140);
            h[(int)(rnd() % 512)] = 3.0e38f;   /* squares to +inf */
            check("overflow-square", h, 512);
        }
    }
    /* ---- regime 9: tail lengths (the 4-wide and scalar tails)           */
    for (trial = 0; trial < 40; ++trial) {
        for (k = 0; k < (int)(sizeof tail_ns / sizeof tail_ns[0]); ++k) {
            n = tail_ns[k];
            for (i = 0; i < n; ++i) h[i] = band_float(90, 150);
            check("tail-random", h, n);
            for (i = 0; i < n; ++i) h[i] = (i & 1) ? 2.0f : -2.0f;
            check("tail-allties", h, n);
            for (i = 0; i < n; ++i) h[i] = 0.0f;
            h[n - 1] = 1.0f;
            check("tail-peak-last", h, n);
        }
    }
    /* ---- regime 10: n < 2 (both must return 0)                          */
    h[0] = 7.0f;
    check("n=0", h, 0);
    check("n=1", h, 1);

    printf("argmax sweep: %ld cases (%ld exercising the NEON body, %ld with at least"
           " one tie at the peak), %ld mismatches -> %s\n",
           cases, neon_body_cases, tie_cases, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}



static float rnd_f(float scale) {
    return scale * (2.0f * ((float)(rnd() >> 40) / 16777216.0f) - 1.0f);
}

static long blocks = 0;

static int cmp_state(const DaBiquad *a, const DaBiquad *b) {
    int s;
    for (s = 0; s < a->n_sections; ++s)
        if (memcmp(a->z[s], b->z[s], sizeof a->z[s]) != 0) return 0;
    return 1;
}

static int run_decimator_sweep(void) {
    rs = 0x9E3779B97F4A7C15ULL;  /* fresh deterministic stream for this half */
    fails = 0;
    DaDecimator spec, gen;
    float in[DA_BLOCK_SIZE], o1[DA_SUB_BLOCK_SIZE], o2[DA_SUB_BLOCK_SIZE];
    int b, i;

    da_decimator_init(&spec);
    da_decimator_init(&gen);
    /* 2000 blocks of mixed regimes; state carries across all of them. */
    for (b = 0; b < 2000; ++b) {
        float scale = (b % 5 == 0) ? 1.0e-30f : (b % 7 == 0) ? 0.98f : 0.2f;
        for (i = 0; i < DA_BLOCK_SIZE; ++i) {
            if (b % 11 == 3) in[i] = (i & 1) ? scale : -scale;   /* nyquist tone */
            else if (b % 13 == 5) in[i] = 0.0f;                  /* silence */
            else in[i] = rnd_f(scale);
        }
        da_decimator_decimate(&spec, in, o1);
        da_decimator_decimate_generic(&gen, in, o2);
        blocks++;
        if (memcmp(o1, o2, sizeof o1) != 0) {
            if (++fails <= 5) printf("  MISMATCH out, block %d\n", b);
        }
        if (!cmp_state(&spec.anti_alias, &gen.anti_alias) ||
            !cmp_state(&spec.noise_reduction, &gen.noise_reduction)) {
            if (++fails <= 5) printf("  MISMATCH state, block %d\n", b);
        }
    }
    /* the shape guard: an off-shape cascade must route to the generic walk */
    {
        DaDecimator odd, ref;
        int agree = 1;
        da_decimator_init(&odd);
        da_decimator_init(&ref);
        odd.anti_alias.n_sections = 2;    /* not the 3-LP + 1-HP shape */
        ref.anti_alias.n_sections = 2;
        for (b = 0; b < 200; ++b) {
            for (i = 0; i < DA_BLOCK_SIZE; ++i) in[i] = rnd_f(0.5f);
            da_decimator_decimate(&odd, in, o1);
            da_decimator_decimate_generic(&ref, in, o2);
            if (memcmp(o1, o2, sizeof o1) != 0) agree = 0;
        }
        if (!agree) { printf("  MISMATCH off-shape fallback\n"); fails++; }
    }
    printf("decimator sweep: %ld state-carrying blocks + 200 off-shape blocks,"
           " %ld mismatches -> %s\n", blocks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

/* The pair kernel runs the render and capture cascades as the two lanes of
 * one vector; against two independent generic walks it must give identical
 * sub-blocks and identical carried filter state, block after block, across
 * the same regimes (denormal-scale, loud, tones, silence). Cascades whose
 * coefficients differ, or whose shape is off, must still agree through the
 * single-call fallback. */
static int run_decimator_pair_sweep(void) {
    rs = 0xD1B54A32D192ED03ULL;
    long pfails = 0, pblocks = 0;
    /* What da_decimator_same_coefficients() must answer for each variant
     * below: 0 = identical cascades, 1 = one coefficient differs, 2 = the
     * cascades have different section counts. A helper that wrongly said
     * "same" corrupts variant 1 (caught by the output comparison); one that
     * always said "different" would only silently cost the fast path, so
     * the verdict itself is asserted too. */
    static const int expect_same[3] = { 1, 0, 0 };
    DaDecimator pa, pb, ga, gb;
    float ina[DA_BLOCK_SIZE], inb[DA_BLOCK_SIZE];
    float oa[DA_SUB_BLOCK_SIZE], ob[DA_SUB_BLOCK_SIZE], ra[DA_SUB_BLOCK_SIZE], rb[DA_SUB_BLOCK_SIZE];
    int b, i, variant;
    for (variant = 0; variant < 3; ++variant) {
        da_decimator_init(&pa); da_decimator_init(&pb);
        da_decimator_init(&ga); da_decimator_init(&gb);
        if (variant == 1) {          /* coefficients differ: single-call fallback */
            pb.anti_alias.b[0][0] *= 1.0001f; gb.anti_alias.b[0][0] *= 1.0001f;
        } else if (variant == 2) {   /* off-shape cascade */
            pb.anti_alias.n_sections = 2; gb.anti_alias.n_sections = 2;
        }
        for (b = 0; b < 2000; ++b) {
            float sa = (b % 5 == 0) ? 1.0e-30f : (b % 7 == 0) ? 0.98f : 0.2f;
            float sb = (b % 3 == 0) ? 0.5f : (b % 11 == 0) ? 1.0e-30f : 0.05f;
            for (i = 0; i < DA_BLOCK_SIZE; ++i) {
                ina[i] = (b % 11 == 3) ? ((i & 1) ? sa : -sa) : (b % 13 == 5) ? 0.0f : rnd_f(sa);
                inb[i] = (b % 17 == 4) ? 0.0f : rnd_f(sb);
            }
#if SWEEP_HAS_NEON_BODY
            const int same = da_decimator_same_coefficients(&pa, &pb);
            if (same != expect_same[variant]) {
                if (++pfails <= 5)
                    printf("  MISMATCH same-coefficients verdict %d, expected %d, variant %d block %d\n",
                           same, expect_same[variant], variant, b);
            }
#else
            /* no helper in this build; the scalar pair function discards it */
            const int same = expect_same[variant];
#endif
            da_decimator_decimate_pair(&pa, ina, oa, &pb, inb, ob, same);
            da_decimator_decimate_generic(&ga, ina, ra);
            da_decimator_decimate_generic(&gb, inb, rb);
            pblocks++;
            if (memcmp(oa, ra, sizeof oa) != 0 || memcmp(ob, rb, sizeof ob) != 0) {
                if (++pfails <= 5) printf("  MISMATCH pair out, variant %d block %d\n", variant, b);
            }
            if (!cmp_state(&pa.anti_alias, &ga.anti_alias) ||
                !cmp_state(&pa.noise_reduction, &ga.noise_reduction) ||
                !cmp_state(&pb.anti_alias, &gb.anti_alias) ||
                !cmp_state(&pb.noise_reduction, &gb.noise_reduction)) {
                if (++pfails <= 5) printf("  MISMATCH pair state, variant %d block %d\n", variant, b);
            }
        }
    }
    printf("decimator pair sweep: %ld state-carrying blocks (3 coefficient/shape variants),"
           " %ld mismatches -> %s\n", pblocks, pfails, pfails ? "FAIL" : "PASS");
    return pfails ? 1 : 0;
}

/* The 48 kHz anti-alias pair runs the capture and render cascades as the two
 * lanes of one vector; against two da_resample48_channel() calls it must give
 * identical kept samples, kept count, carried filter state and phase, call
 * after call, for hop lengths that are not multiples of the decimation factor
 * or of the vector width, and when the output cap truncates. Cascades whose
 * coefficients differ, or whose shape is off, must agree through the
 * single-call fallback. */
static int run_resample48_pair_sweep(void) {
    rs = 0xA24BAED4963EE407ULL;
    long rfails = 0, calls = 0;
    DaBiquad pa, pb, ga, gb;
    float ina[1100], inb[1100];
    float oa[1100], ob[1100], ra[1100], rb[1100];
    static const int lens[] = { 0, 1, 2, 3, 4, 5, 7, 8, 255, 256, 257, 511, 512, 513, 1023, 1024, 1100 };
    int variant, c, i;
    for (variant = 0; variant < 4; ++variant) {
        int pph = 0, gph = 0;
        da_biquad_init_resample48(&pa); da_biquad_init_resample48(&pb);
        da_biquad_init_resample48(&ga); da_biquad_init_resample48(&gb);
        if (variant == 1) {          /* coefficients differ: single-call fallback */
            pb.b[1][0] *= 1.0001f; gb.b[1][0] *= 1.0001f;
        } else if (variant == 2) {   /* one off-shape cascade */
            pb.n_sections = 3; gb.n_sections = 3;
        } else if (variant == 3) {   /* both off-shape, equal coefficients */
            pa.n_sections = 3; ga.n_sections = 3;
            pb.n_sections = 3; gb.n_sections = 3;
        }
        for (c = 0; c < 3000; ++c) {
            int n = lens[c % (int)(sizeof lens / sizeof lens[0])];
            float sa = (c % 5 == 0) ? 1.0e-30f : (c % 7 == 0) ? 0.98f : 0.2f;
            float sb = (c % 3 == 0) ? 0.5f : (c % 11 == 0) ? 1.0e-30f : 0.05f;
            int cap = (c % 4 == 3) ? n / 3 : (n + 2) / 3 + 1;
            int pn, gn_a, gn_b, ph_a = gph, ph_b = gph;
            for (i = 0; i < n; ++i) {
                ina[i] = (c % 11 == 3) ? ((i & 1) ? sa : -sa) : (c % 13 == 5) ? 0.0f : rnd_f(sa);
                inb[i] = (c % 17 == 4) ? 0.0f : rnd_f(sb);
            }
            memset(oa, 0, sizeof oa); memset(ob, 0, sizeof ob);
            memset(ra, 0, sizeof ra); memset(rb, 0, sizeof rb);
            pn = da_resample48_pair(&pa, ina, oa, &pb, inb, ob, n, cap, &pph);
            gn_a = da_resample48_channel(&ga, ina, n, ra, cap, &ph_a);
            gn_b = da_resample48_channel(&gb, inb, n, rb, cap, &ph_b);
            gph = ph_a;
            calls++;
            if (pn != gn_a || gn_a != gn_b || pph != gph ||
                memcmp(oa, ra, sizeof oa) != 0 || memcmp(ob, rb, sizeof ob) != 0) {
                if (++rfails <= 5) printf("  MISMATCH resample out, variant %d call %d n=%d\n", variant, c, n);
            }
            if (!cmp_state(&pa, &ga) || !cmp_state(&pb, &gb)) {
                if (++rfails <= 5) printf("  MISMATCH resample state, variant %d call %d n=%d\n", variant, c, n);
            }
        }
    }
    printf("resample48 pair sweep: %ld state-carrying calls (4 coefficient/shape variants),"
           " %ld mismatches -> %s\n", calls, rfails, rfails ? "FAIL" : "PASS");
    return rfails ? 1 : 0;
}

/* Separate update/dot passes are the reference for the fused matched core. */
static int matched_core_unfused(const DaRing *ring, int shift, float threshold,
                                float smoothing, const float *y, float *h,
                                float *error_out, int *update_count) {
    float span[DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1];
    float error_sum = 0.0f;
    da_ring_gather_back(ring, shift, DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1, span);
    float x2 = da_dot_f32(span + DA_SUB_BLOCK_SIZE - 1,
                         span + DA_SUB_BLOCK_SIZE - 1, DA_FILTER_SIZE);
    *update_count = 0;
    for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) {
        const float *x = span + DA_SUB_BLOCK_SIZE - 1 - i;
        if (i > 0) x2 += x[0] * x[0] - x[DA_FILTER_SIZE] * x[DA_FILTER_SIZE];
        float e = y[i] - da_dot_f32(h, x, DA_FILTER_SIZE);
        int saturated = y[i] >= DA_SATURATION_LIMIT || y[i] <= -DA_SATURATION_LIMIT;
        error_sum += e * e;
        if (x2 > threshold && !saturated) {
            da_nlms_update_f32(h, x, smoothing * e / x2, DA_FILTER_SIZE);
            ++*update_count;
        }
    }
    *error_out = error_sum;
    return *update_count != 0;
}

static int run_matched_core_sweep(void) {
    float ring[DA_RING_CAPACITY_FOR(DA_NUM_FILTERS)];
    float h[DA_FILTER_SIZE], ref[DA_FILTER_SIZE], y[DA_SUB_BLOCK_SIZE];
    int first_only = 0, last_only = 0, no_update = 0;
    int nan_payload_only = 0;
    const float gate = (float)DA_FILTER_SIZE * DA_EXCITATION_LIMIT * DA_EXCITATION_LIMIT;
    for (int trial = 0; trial < 10000; ++trial) {
        int variant = trial % 10;
        for (int i = 0; i < (int)(sizeof ring / sizeof ring[0]); ++i) ring[i] = rnd_f(0.1f);
        for (int i = 0; i < DA_FILTER_SIZE; ++i) h[i] = rnd_f(0.01f);
        for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = rnd_f(0.1f);
        if (variant == 0) memset(ring, 0, sizeof ring);
        if (variant == 1) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = (i & 1) ? 1.0f : -1.0f;
        if (variant == 2) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = i == DA_SUB_BLOCK_SIZE - 1 ? 0.1f : 1.0f;
        if (variant == 3) for (int i = 0; i < DA_SUB_BLOCK_SIZE; ++i) y[i] = i == 0 ? 0.1f : -1.0f;
        if (variant == 4) for (int i = 0; i < (int)(sizeof ring / sizeof ring[0]); ++i) ring[i] *= 1.0e-20f;
        if (variant == 5) for (int i = 0; i < (int)(sizeof ring / sizeof ring[0]); ++i) ring[i] = (i & 1) ? 0.0f : -0.0f;
        if (variant == 6) y[trial % DA_SUB_BLOCK_SIZE] = NAN;
        if (variant == 7) y[trial % DA_SUB_BLOCK_SIZE] = INFINITY;
        if (variant == 8) h[trial % DA_FILTER_SIZE] = nanf("123");
        DaRing r = { .buffer = ring, .capacity = (int)(sizeof ring / sizeof ring[0]),
                     .write = (trial * 71) % (int)(sizeof ring / sizeof ring[0]) };
        int shift = (trial % DA_NUM_FILTERS) * DA_FILTER_INTRA_SHIFT;
        float threshold = variant == 9 ? 1.0e20f : gate;
        float smoothing = (trial & 1) ? DA_SMOOTHING_FAST : DA_SMOOTHING_SLOW;
        float a, b;
        int updates;
        memcpy(ref, h, sizeof h);
        int actual = da_matched_filter_core(&r, shift, threshold, smoothing, y, h, &a);
        int expected = matched_core_unfused(&r, shift, threshold, smoothing, y, ref, &b, &updates);
        int mismatch = actual != expected;
        if (memcmp(&a, &b, sizeof a)) {
            if (isnan(a) && isnan(b)) ++nan_payload_only;
            else mismatch = 1;
        }
        for (int i = 0; i < DA_FILTER_SIZE; ++i) {
            if (memcmp(h + i, ref + i, sizeof h[i])) {
                if (isnan(h[i]) && isnan(ref[i])) ++nan_payload_only;
                else mismatch = 1;
            }
        }
        if (mismatch) {
            printf("  MISMATCH matched core trial %d variant %d\n", trial, variant);
            return 1;
        }
        if (variant == 2 && updates == 1) ++last_only;
        if (variant == 3 && updates == 1) ++first_only;
        if (updates == 0) ++no_update;
    }
    if (first_only != 1000 || last_only != 1000 || no_update < 4000) {
        printf("  FAIL matched core premises: first=%d last=%d none=%d\n", first_only, last_only, no_update);
        return 1;
    }
    printf("matched core sweep: 10000 cases, first/last/no-update gates, full taps + error sum,"
           " %d NaN payload-only differences -> PASS\n", nan_payload_only);
    return 0;
}

int main(void) {
    int rc = 0;
    SK_MATCH_SCALAR_TO_VECTOR_MODE();
    if (!SWEEP_HAS_NEON_BODY)
        printf("note: scalar build -- argmax sweep degenerates to an identity check;"
               " the decimator and 48 kHz pair sweeps check the single-call fallback"
               " against the generic walks (no pair kernel, no coefficient-equality helper)\n");
    rc |= run_argmax_sweep();
    rc |= run_decimator_sweep();
    rc |= run_decimator_pair_sweep();
    rc |= run_resample48_pair_sweep();
    rc |= run_matched_core_sweep();
    if (rc == 0) printf("test_delay_kernels: PASS\n");
    return rc;
}
