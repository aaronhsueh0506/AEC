/**
 * parity_multi_erle_test.c - Phase 2a parity verifier
 *
 * Reads python/parity_multi_erle_input.bin (synthetic echo/error specs +
 * near/err powers + far_active/dt schedules), runs C FilterErle +
 * FullbandErle + compute_erle_confidence, dumps per-frame state.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multi_erle.h"

#define N_FREQS  257
#define N_FRAMES 200


int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    /* Allocate buffers (input arrays) */
    size_t spec_n = (size_t)N_FRAMES * N_FREQS;
    float* echo_re   = (float*)malloc(spec_n * sizeof(float));
    float* echo_im   = (float*)malloc(spec_n * sizeof(float));
    float* err_re    = (float*)malloc(spec_n * sizeof(float));
    float* err_im    = (float*)malloc(spec_n * sizeof(float));
    float* near_pwr  = (float*)malloc(N_FRAMES * sizeof(float));
    float* err_pwr   = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* far_act = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* dt_ind    = (float*)malloc(N_FRAMES * sizeof(float));

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 2; }
    fread(echo_re, sizeof(float), spec_n, fp);
    fread(echo_im, sizeof(float), spec_n, fp);
    fread(err_re,  sizeof(float), spec_n, fp);
    fread(err_im,  sizeof(float), spec_n, fp);
    fread(near_pwr, sizeof(float), N_FRAMES, fp);
    fread(err_pwr,  sizeof(float), N_FRAMES, fp);
    fread(far_act, sizeof(int32_t), N_FRAMES, fp);
    fread(dt_ind,  sizeof(float), N_FRAMES, fp);
    fclose(fp);

    FilterErleEstimator* fe = fe_create(N_FREQS);
    FullbandErleEstimator* fbe = fbe_create();
    if (!fe || !fbe) return 3;

    Complex* echo_spec = (Complex*)malloc(N_FREQS * sizeof(Complex));
    Complex* err_spec  = (Complex*)malloc(N_FREQS * sizeof(Complex));
    if (!echo_spec || !err_spec) return 4;

    /* Output binary */
    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 5; }
    int32_t hdr[2] = { N_FREQS, N_FRAMES };
    fwrite(hdr, sizeof(int32_t), 2, out);

    for (int f = 0; f < N_FRAMES; f++) {
        for (int k = 0; k < N_FREQS; k++) {
            echo_spec[k].r = echo_re[f * N_FREQS + k];
            echo_spec[k].i = echo_im[f * N_FREQS + k];
            err_spec[k].r  = err_re [f * N_FREQS + k];
            err_spec[k].i  = err_im [f * N_FREQS + k];
        }

        fe_update(fe, echo_spec, err_spec, far_act[f], dt_ind[f]);
        fbe_update(fbe, near_pwr[f], err_pwr[f], far_act[f], dt_ind[f]);

        const float* erle = fe_get_erle(fe);
        float fb = fbe_get_fb_erle(fbe);
        float conf = compute_erle_confidence(erle, N_FREQS, fb);

        fwrite(erle, sizeof(float), N_FREQS, out);
        fwrite(&fb, sizeof(float), 1, out);
        fwrite(&conf, sizeof(float), 1, out);
    }

    fclose(out);

    fe_destroy(fe);
    fbe_destroy(fbe);
    free(echo_re); free(echo_im); free(err_re); free(err_im);
    free(near_pwr); free(err_pwr); free(far_act); free(dt_ind);
    free(echo_spec); free(err_spec);

    printf("Wrote %s\n", argv[2]);
    return 0;
}
