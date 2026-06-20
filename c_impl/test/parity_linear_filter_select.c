/* parity_linear_filter_select.c — replay the REAL-pipeline binary golden from
 * python/diag/gen_linear_filter_select_golden.py through the C
 * LinearFilterSelect port (AEC._aec3_select_linear_filter_output) and check
 * both selected_esw and selected_echo_spec match within the float32 FFT
 * tolerance across every captured hop of the DT case.
 *
 * This is a per-hop REPLAY (inputs fed from the golden each hop), so the only
 * source of drift is the single rfft of [e_old|e_form]·sqrt_hann (vendored
 * KISS FFT, float32) — one FFT deep, no recursion. The e2/y2/s2 sums are
 * np.sum-pairwise in f64 and the crossfade ramp + complex add/sub are reproduced
 * op-for-op in matching dtype (see linear_filter_output.c), so the divergence is
 * ULP-scale, not algorithmic. (Under the retired pocketfft backend this was a
 * strict 0/0 bit-pattern check; KISS trades that for the float32 tolerance.)
 *
 * Build (from c_impl/), STANDALONE (do NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Iinclude -Ilib/kiss_fft \
 *       src/linear_filter_output.c src/fft_wrapper.c lib/kiss_fft/kiss_fft.c \
 *       test/parity_linear_filter_select.c -lm -o /tmp/p_lfs
 *   python3 python/diag/gen_linear_filter_select_golden.py /tmp/lfs_golden.bin
 *   /tmp/p_lfs /tmp/lfs_golden.bin
 */
#include "linear_filter_output.h"
#include "fft_wrapper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* selected_esw/echo carry one rfft of windowed error; KISS float32 vs numpy
 * fp64 drifts at ULP scale. 1e-4 brackets it; a real regression is O(0.1). */
#define LFS_TOL 1.0e-4

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* |got-exp| > tol ?  (always updates *maxd so the worst diff is reported) */
static int tol_exceed(float got, float exp, double tol, double *maxd) {
    double d = fabs((double)got - (double)exp);
    if (d > *maxd) *maxd = d;
    return d > tol;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/lfs_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int hop, block_size, fft_size, K, n_hops;
    if (!rd(f, &hop, 4) || !rd(f, &block_size, 4) || !rd(f, &fft_size, 4) ||
        !rd(f, &K, 4) || !rd(f, &n_hops, 4)) return 2;

    printf("config: hop=%d block=%d fft=%d K=%d hops=%d\n",
           hop, block_size, fft_size, K, n_hops);

    float *sqrt_hann = malloc((size_t)block_size * sizeof(float));
    if (!rd(f, sqrt_hann, (size_t)block_size * 4)) { fprintf(stderr, "short sqrt_hann\n"); return 2; }

    FftHandle *fft = fft_create(fft_size);
    if (!fft) { fprintf(stderr, "fft_create failed\n"); return 2; }

    LinearFilterSelect s;
    if (linear_filter_select_init(&s, hop, block_size, fft_size, K) != 0) {
        fprintf(stderr, "init OOM\n"); return 2;
    }

    float   *e_refined = malloc((size_t)hop * sizeof(float));
    float   *near_end  = malloc((size_t)hop * sizeof(float));
    float   *e_coarse  = malloc((size_t)hop * sizeof(float));
    float   *re_buf    = malloc((size_t)K * sizeof(float));
    float   *im_buf    = malloc((size_t)K * sizeof(float));
    Complex *esw_in    = malloc((size_t)K * sizeof(Complex));
    Complex *echo_in   = malloc((size_t)K * sizeof(Complex));
    Complex *exp_esw   = malloc((size_t)K * sizeof(Complex));
    Complex *exp_echo  = malloc((size_t)K * sizeof(Complex));
    Complex *got_esw   = malloc((size_t)K * sizeof(Complex));
    Complex *got_echo  = malloc((size_t)K * sizeof(Complex));

    long mism_esw = 0, mism_echo = 0, total = 0;
    double maxd_esw = 0.0, maxd_echo = 0.0;
    int first_bad_hop = -1;

    for (int h = 0; h < n_hops; ++h) {
        if (!rd(f, e_refined, (size_t)hop * 4) ||
            !rd(f, near_end,  (size_t)hop * 4) ||
            !rd(f, e_coarse,  (size_t)hop * 4)) { fprintf(stderr, "short inputs hop %d\n", h); return 2; }
        /* error_spec_windowed: re[K] then im[K] */
        if (!rd(f, re_buf, (size_t)K * 4) || !rd(f, im_buf, (size_t)K * 4)) { fprintf(stderr, "short esw %d\n", h); return 2; }
        for (int k = 0; k < K; ++k) { esw_in[k].r = re_buf[k]; esw_in[k].i = im_buf[k]; }
        /* echo_spec */
        if (!rd(f, re_buf, (size_t)K * 4) || !rd(f, im_buf, (size_t)K * 4)) { fprintf(stderr, "short echo %d\n", h); return 2; }
        for (int k = 0; k < K; ++k) { echo_in[k].r = re_buf[k]; echo_in[k].i = im_buf[k]; }
        /* expected selected_esw */
        if (!rd(f, re_buf, (size_t)K * 4) || !rd(f, im_buf, (size_t)K * 4)) { fprintf(stderr, "short exp_esw %d\n", h); return 2; }
        for (int k = 0; k < K; ++k) { exp_esw[k].r = re_buf[k]; exp_esw[k].i = im_buf[k]; }
        /* expected selected_echo */
        if (!rd(f, re_buf, (size_t)K * 4) || !rd(f, im_buf, (size_t)K * 4)) { fprintf(stderr, "short exp_echo %d\n", h); return 2; }
        for (int k = 0; k < K; ++k) { exp_echo[k].r = re_buf[k]; exp_echo[k].i = im_buf[k]; }

        linear_filter_select(&s, e_refined, near_end, e_coarse,
                             esw_in, echo_in, sqrt_hann, fft,
                             got_esw, got_echo);

        for (int k = 0; k < K; ++k) {
            int bad = 0;
            /* bitwise | (not ||) so both r and i always update maxd */
            if (tol_exceed(got_esw[k].r, exp_esw[k].r, LFS_TOL, &maxd_esw) |
                tol_exceed(got_esw[k].i, exp_esw[k].i, LFS_TOL, &maxd_esw)) { mism_esw++; bad = 1; }
            if (tol_exceed(got_echo[k].r, exp_echo[k].r, LFS_TOL, &maxd_echo) |
                tol_exceed(got_echo[k].i, exp_echo[k].i, LFS_TOL, &maxd_echo)) { mism_echo++; bad = 1; }
            if (bad && first_bad_hop < 0) first_bad_hop = h;
        }
        total += K;
    }
    fclose(f);

    printf("linear_filter_select parity: %d hops, %d bins, %ld complex points\n",
           n_hops, K, total);
    printf("  selected_esw  tol-exceed=%ld (max=%.3e)\n", mism_esw, maxd_esw);
    printf("  selected_echo tol-exceed=%ld (max=%.3e)\n", mism_echo, maxd_echo);
    if (mism_esw || mism_echo) {
        printf("  first diverging hop=%d\n", first_bad_hop);
        printf(">>> FAIL (exceeds float32 FFT tolerance %.0e)\n", (double)LFS_TOL);
        return 1;
    }
    printf(">>> PASS (within float32 FFT tolerance %.0e; esw max=%.3e echo max=%.3e)\n",
           (double)LFS_TOL, maxd_esw, maxd_echo);

    linear_filter_select_free(&s);
    fft_destroy(fft);
    free(sqrt_hann); free(e_refined); free(near_end); free(e_coarse);
    free(re_buf); free(im_buf); free(esw_in); free(echo_in);
    free(exp_esw); free(exp_echo); free(got_esw); free(got_echo);
    return 0;
}
