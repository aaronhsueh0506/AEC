/* Permanent NEON-vs-scalar equivalence gates for delay_aec3.c's two local
 * kernels. Committed rather than left as throwaway sweeps because the
 * argmax's tie-order semantics are invisible to every other check in the
 * repo: real audio never produces an exact h^2 tie inside a 16-lane group,
 * so the E2E dumps and the delay golden pass even with last-match-wins
 * semantics -- only this sweep fails. The decimator half proves the
 * register-specialized cascade against the generic walk across carried IIR
 * state, which a per-block re-zeroing comparison would not.
 *
 * On builds without the NEON body (non-aarch64, or
 * SIMD_KERNELS_FORCE_SCALAR) the argmax half degenerates to comparing the
 * scalar walk against itself; it still runs -- and says so -- rather than
 * failing, so one `make test-delay-kernels` invocation is valid everywhere.
 * Includes the SHIPPED translation unit so both halves exercise the real
 * source text, not a transcription. */
#include "src/delay_aec3.c"

#include <stdio.h>

#if defined(__ARM_NEON) && defined(__aarch64__) && !defined(SIMD_KERNELS_FORCE_SCALAR)
#define SWEEP_HAS_NEON_BODY 1
#else
#define SWEEP_HAS_NEON_BODY 0
#endif


#define NMAX 600
static unsigned long long rs = 0x243F6A8885A308D3ULL;
static unsigned long long rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs;
}

static long cases = 0, neon_body_cases = 0, tie_cases = 0, fails = 0;

static void check(const char *what, const float *h, int n) {
    int a = da_max_square_peak_index(h, n);
    int b = da_max_square_peak_scan(h, n);
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
    float in[DA_AEC3_BLOCK_SIZE], o1[DA_SUB_BLOCK_SIZE], o2[DA_SUB_BLOCK_SIZE];
    int b, i;

    da_decimator_init(&spec);
    da_decimator_init(&gen);
    /* 2000 blocks of mixed regimes; state carries across all of them. */
    for (b = 0; b < 2000; ++b) {
        float scale = (b % 5 == 0) ? 1.0e-30f : (b % 7 == 0) ? 0.98f : 0.2f;
        for (i = 0; i < DA_AEC3_BLOCK_SIZE; ++i) {
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
            for (i = 0; i < DA_AEC3_BLOCK_SIZE; ++i) in[i] = rnd_f(0.5f);
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

int main(void) {
    int rc = 0;
    if (!SWEEP_HAS_NEON_BODY)
        printf("note: scalar build -- argmax sweep degenerates to an identity check\n");
    rc |= run_argmax_sweep();
    rc |= run_decimator_sweep();
    if (rc == 0) printf("test_delay_kernels: PASS\n");
    return rc;
}
