/* parity_fft.c — replay python/diag/gen_fft_golden.py through the C FFT wrapper
 * (fft_wrapper.c, backed by the vendored KISS FFT (float32)) and report
 * the BIT-EXACT mismatch count vs numpy's np.fft.rfft / np.fft.irfft at n=512.
 *
 * The decisive question: is the mismatch count 0? If yes, the C<->Python FFT is
 * bit-exact and the whole AEC pipeline can be bit-exact. Comparison is exact on
 * the stored 32-bit values (complex64 for rfft, float32 for irfft) — no rtol.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I<path-to-repo>/c_impl/include \
 *       -I<path-to-repo>/c_impl/lib/kiss_fft \
 *       <path-to-repo>/c_impl/src/fft_wrapper.c \
 *       <path-to-repo>/c_impl/lib/kiss_fft/kiss_fft.c \
 *       <path-to-repo>/c_impl/test/parity_fft.c \
 *       -lm -o /tmp/p_fft
 *   python3 .../python/diag/gen_fft_golden.py /tmp/fft_golden.bin
 *   /tmp/p_fft /tmp/fft_golden.bin
 */
#include "fft_wrapper.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* Count bit-exact mismatches between two float arrays (exact equality). Tracks
 * the worst absolute diff and the worst ULP distance for diagnostics. */
typedef struct { long mism; long total; double maxabs; long maxulp; } Acc;

static long ulp_dist_f32(float a, float b) {
    int32_t ia, ib;
    union { float f; int32_t i; } ua, ub;
    ua.f = a; ub.f = b;
    ia = ua.i; ib = ub.i;
    /* map to monotone ordering across sign */
    if (ia < 0) ia = (int32_t)0x80000000 - ia;
    if (ib < 0) ib = (int32_t)0x80000000 - ib;
    long d = (long)ia - (long)ib;
    return d < 0 ? -d : d;
}

static void cmp_f32(const float *got, const float *exp, int n, Acc *a) {
    for (int k = 0; k < n; ++k) {
        a->total++;
        if (got[k] != exp[k]) {
            a->mism++;
            double d = fabs((double)got[k] - (double)exp[k]);
            if (d > a->maxabs) a->maxabs = d;
            long u = ulp_dist_f32(got[k], exp[k]);
            if (u > a->maxulp) a->maxulp = u;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fft_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int32_t n, n_fwd, n_inv;
    if (!rd(f, &n, 4) || !rd(f, &n_fwd, 4) || !rd(f, &n_inv, 4)) {
        fprintf(stderr, "bad header\n"); return 2;
    }
    const int nf = n / 2 + 1;

    FftHandle *h = fft_create(n);
    if (!h) { fprintf(stderr, "fft_create(%d) failed\n", n); return 2; }

    float   *x   = malloc(sizeof(float) * n);
    float   *yo  = malloc(sizeof(float) * n);
    Complex *Xc  = malloc(sizeof(Complex) * nf);   /* C-computed spectrum */
    float   *Xg  = malloc(sizeof(float) * 2 * nf); /* golden spectrum (re,im) */
    Complex *Xin = malloc(sizeof(Complex) * nf);
    float   *Xinf = malloc(sizeof(float) * 2 * nf);
    float   *yg  = malloc(sizeof(float) * n);

    Acc fa = {0, 0, 0.0, 0}, ia = {0, 0, 0.0, 0};
    int rec_ok = 1;

    /* ---- forward: fft_forward vs np.fft.rfft (complex64) ---- */
    for (int r = 0; r < n_fwd && rec_ok; ++r) {
        if (!rd(f, x, sizeof(float) * n) ||
            !rd(f, Xg, sizeof(float) * 2 * nf)) { rec_ok = 0; break; }
        fft_forward(h, x, Xc);
        /* compare interleaved re,im against golden (both complex64) */
        for (int k = 0; k < nf; ++k) {
            cmp_f32(&Xc[k].r, &Xg[2 * k],     1, &fa);
            cmp_f32(&Xc[k].i, &Xg[2 * k + 1], 1, &fa);
        }
    }

    /* ---- inverse: fft_inverse vs np.fft.irfft (float32) ---- */
    for (int r = 0; r < n_inv && rec_ok; ++r) {
        if (!rd(f, Xinf, sizeof(float) * 2 * nf) ||
            !rd(f, yg, sizeof(float) * n)) { rec_ok = 0; break; }
        for (int k = 0; k < nf; ++k) {
            Xin[k].r = Xinf[2 * k];
            Xin[k].i = Xinf[2 * k + 1];
        }
        fft_inverse(h, Xin, yo);
        cmp_f32(yo, yg, n, &ia);
    }

    if (!rec_ok) { fprintf(stderr, "truncated golden file\n"); return 2; }

    printf("=== FFT bit-exact parity (n=%d) ===\n", n);
    printf("FORWARD  np.fft.rfft : records=%d  values=%ld  mismatches=%ld",
           n_fwd, fa.total, fa.mism);
    if (fa.mism) printf("  (maxabs=%.3e  maxulp=%ld)", fa.maxabs, fa.maxulp);
    printf("\n");
    printf("INVERSE  np.fft.irfft: records=%d  values=%ld  mismatches=%ld",
           n_inv, ia.total, ia.mism);
    if (ia.mism) printf("  (maxabs=%.3e  maxulp=%ld)", ia.maxabs, ia.maxulp);
    printf("\n");

    int ok = (fa.mism == 0 && ia.mism == 0);
    printf(">>> %s: C<->Python FFT is %sBIT-EXACT\n",
           ok ? "PASS" : "FAIL", ok ? "" : "NOT ");

    free(x); free(yo); free(Xc); free(Xg); free(Xin); free(Xinf); free(yg);
    fft_destroy(h);
    fclose(f);
    return ok ? 0 : 1;
}
