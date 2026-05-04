/* parity_erle.c — Wave 1.4 gate.
 *
 * Synthetic-input parity for FilterErleEstimator + FullbandErleEstimator +
 * compute_erle_confidence. Bit-exact (ULP) where Python is fp64 throughout
 * (FullbandErle, confidence). FilterErle does fp64 EMA + fp32 store →
 * compare with rtol < 1e-6 on per-bin erle.
 */
#include "erle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int read_exact(FILE* f, void* b, size_t n) {
    return fread(b, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <erle.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t n_freqs, n_frames;
    if (read_exact(f, &n_freqs, 4) || read_exact(f, &n_frames, 4)) {
        fprintf(stderr, "header read failed\n"); return 2;
    }
    fprintf(stderr, "[erle] n_freqs=%d n_frames=%d\n", n_freqs, n_frames);

    FilterErleEst  fer; filter_erle_init(&fer, n_freqs);
    FullbandErleEst fbe; fullband_erle_init(&fbe);

    float* echo_re = (float*)malloc((size_t)n_freqs * sizeof(float));
    float* echo_im = (float*)malloc((size_t)n_freqs * sizeof(float));
    float* err_re  = (float*)malloc((size_t)n_freqs * sizeof(float));
    float* err_im  = (float*)malloc((size_t)n_freqs * sizeof(float));
    float* exp_erle = (float*)malloc((size_t)n_freqs * sizeof(float));

    double max_erle_abs = 0.0;
    double max_erle_rel = 0.0;
    double max_fb_abs   = 0.0;
    double max_conf_abs = 0.0;
    int    fail = 0;

    for (int fi = 0; fi < n_frames; ++fi) {
        size_t sz = (size_t)n_freqs * sizeof(float);
        if (read_exact(f, echo_re, sz) || read_exact(f, echo_im, sz) ||
            read_exact(f, err_re,  sz) || read_exact(f, err_im,  sz)) {
            fprintf(stderr, "frame %d input read failed\n", fi); fail = 1; break;
        }
        int32_t far_active;
        double dt_ind, near_pwr, err_pwr, exp_fb, exp_conf;
        if (read_exact(f, &far_active, 4) || read_exact(f, &dt_ind, 8) ||
            read_exact(f, &near_pwr, 8)   || read_exact(f, &err_pwr, 8)) {
            fprintf(stderr, "frame %d scalar read failed\n", fi); fail = 1; break;
        }
        if (read_exact(f, exp_erle, sz) ||
            read_exact(f, &exp_fb, 8) || read_exact(f, &exp_conf, 8)) {
            fprintf(stderr, "frame %d expected read failed\n", fi); fail = 1; break;
        }

        filter_erle_update(&fer, echo_re, echo_im, err_re, err_im,
                              far_active, dt_ind);
        fullband_erle_update(&fbe, near_pwr, err_pwr, far_active, dt_ind);
        double conf = erle_compute_confidence(fer.erle, n_freqs, fbe.fb_erle);

        for (int k = 0; k < n_freqs; ++k) {
            double a = fabs((double)fer.erle[k] - (double)exp_erle[k]);
            double r = (exp_erle[k] != 0.0f) ? a / fabs((double)exp_erle[k]) : a;
            if (a > max_erle_abs) max_erle_abs = a;
            if (r > max_erle_rel) max_erle_rel = r;
        }
        double fb_a   = fabs(fbe.fb_erle - exp_fb);
        double conf_a = fabs(conf - exp_conf);
        if (fb_a > max_fb_abs) max_fb_abs = fb_a;
        if (conf_a > max_conf_abs) max_conf_abs = conf_a;
    }
    fclose(f);
    free(echo_re); free(echo_im); free(err_re); free(err_im); free(exp_erle);
    filter_erle_free(&fer);

    fprintf(stderr,
            "[erle] erle abs=%.3e rel=%.3e  fb abs=%.3e  conf abs=%.3e\n",
            max_erle_abs, max_erle_rel, max_fb_abs, max_conf_abs);

    /* fp32 store-back gives ~1 ULP per-bin (~3e-7 rel) on FilterErle.
     * Confidence depends on np.mean of fp32 array → log/exp; absolute
     * conf diff stays well below 1e-6. fullband is bit-exact (fp64).  */
    if (fail || max_erle_rel > 1e-6 || max_fb_abs > 0.0 || max_conf_abs > 1e-6) {
        fprintf(stderr, "[erle] FAIL\n"); return 1;
    }
    fprintf(stderr, "[erle] PASS\n");
    return 0;
}
