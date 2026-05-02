/* parity_pbfdkf.c — Wave 2 gate.
 *
 * Drives PBFDKFV2 with Python-recorded near/far/mu_scale; compares output,
 * error_spec, echo_spec, P, R against Python state. Tolerances:
 *   FFT outputs (error_spec/echo_spec): rtol < 1e-4, abs < 1e-5
 *   Time output: rtol < 1e-4
 *   P / R (Kalman state): rtol < 1e-4 (compounds over frames; may need
 *                                       relax to 1e-3 if FFT drift dominates)
 */
#include "pbfdkf_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int rd(FILE* f, void* b, size_t n) { return fread(b,1,n,f)==n?0:-1; }

typedef struct { double abs_max, rel_max; int first_diff_frame, first_diff_idx; } StatT;

static void stat_update(StatT* s, double c, double e, int frame, int idx) {
    double a = fabs(c - e);
    double r = (fabs(e) > 1e-30) ? a / fabs(e) : a;
    if (a > s->abs_max) s->abs_max = a;
    if (r > s->rel_max) {
        s->rel_max = r;
        if (s->first_diff_frame < 0) {
            s->first_diff_frame = frame; s->first_diff_idx = idx;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <pbfdkf.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t hop, n_freqs, n_parts, n_frames;
    float   mu, delta;
    if (rd(f,&hop,4)||rd(f,&n_freqs,4)||rd(f,&n_parts,4)||rd(f,&n_frames,4)||
        rd(f,&mu,4)||rd(f,&delta,4)) { fprintf(stderr,"hdr fail\n"); return 2; }
    fprintf(stderr,
            "[pbfdkf] hop=%d n_freqs=%d n_parts=%d n_frames=%d mu=%.3f delta=%.1e\n",
            hop, n_freqs, n_parts, n_frames, mu, delta);

    PBFDKFV2 p;
    pbfdkf_v2_init(&p, 2*hop, n_parts, mu, delta, hop);

    float* near    = (float*)malloc((size_t)hop * sizeof(float));
    float* far_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* mu_arr  = (float*)malloc((size_t)n_freqs * sizeof(float));
    float* out_c   = (float*)malloc((size_t)hop * sizeof(float));
    float* exp_out  = (float*)malloc((size_t)hop * sizeof(float));
    float* exp_err  = (float*)malloc((size_t)2 * n_freqs * sizeof(float));
    float* exp_echo = (float*)malloc((size_t)2 * n_freqs * sizeof(float));
    float* exp_P    = (float*)malloc((size_t)n_parts * n_freqs * sizeof(float));
    float* exp_R    = (float*)malloc((size_t)n_freqs * sizeof(float));

    StatT s_out  = {0,0,-1,-1}, s_err = {0,0,-1,-1}, s_echo = {0,0,-1,-1};
    StatT s_P    = {0,0,-1,-1}, s_R = {0,0,-1,-1};

    for (int fi = 0; fi < n_frames; ++fi) {
        if (rd(f, near, (size_t)hop * sizeof(float)) ||
            rd(f, far_buf, (size_t)hop * sizeof(float))) break;
        int32_t is_arr; float mu_scalar = 1.0f;
        if (rd(f, &is_arr, 4)) break;
        if (is_arr) {
            if (rd(f, mu_arr, (size_t)n_freqs * sizeof(float))) break;
        } else {
            if (rd(f, &mu_scalar, 4)) break;
        }
        if (rd(f, exp_out, (size_t)hop * sizeof(float)) ||
            rd(f, exp_err, (size_t)2 * n_freqs * sizeof(float)) ||
            rd(f, exp_echo, (size_t)2 * n_freqs * sizeof(float)) ||
            rd(f, exp_P, (size_t)n_parts * n_freqs * sizeof(float)) ||
            rd(f, exp_R, (size_t)n_freqs * sizeof(float))) break;

        pbfdkf_v2_process(&p, near, far_buf,
                          is_arr ? mu_arr : NULL,
                          mu_scalar, out_c);

        for (int i = 0; i < hop; ++i) stat_update(&s_out, out_c[i], exp_out[i], fi, i);
        for (int k = 0; k < 2*n_freqs; ++k) {
            stat_update(&s_err, p.base.error_spec[k/2].r * (k%2==0?1:0)
                              + p.base.error_spec[k/2].i * (k%2==1?1:0),
                        exp_err[k], fi, k);
        }
        for (int k = 0; k < n_freqs; ++k) {
            stat_update(&s_err, p.base.error_spec[k].r, exp_err[2*k],   fi, 2*k);
            stat_update(&s_err, p.base.error_spec[k].i, exp_err[2*k+1], fi, 2*k+1);
            stat_update(&s_echo, p.base.echo_spec[k].r, exp_echo[2*k],   fi, 2*k);
            stat_update(&s_echo, p.base.echo_spec[k].i, exp_echo[2*k+1], fi, 2*k+1);
            stat_update(&s_R, p.R[k], exp_R[k], fi, k);
        }
        for (int k = 0; k < n_parts*n_freqs; ++k) {
            stat_update(&s_P, p.P[k], exp_P[k], fi, k);
        }
    }
    fclose(f);

    fprintf(stderr,
        "[pbfdkf] out  abs=%.3e rel=%.3e (first frame=%d idx=%d)\n",
        s_out.abs_max, s_out.rel_max, s_out.first_diff_frame, s_out.first_diff_idx);
    fprintf(stderr,
        "[pbfdkf] err  abs=%.3e rel=%.3e\n", s_err.abs_max, s_err.rel_max);
    fprintf(stderr,
        "[pbfdkf] echo abs=%.3e rel=%.3e\n", s_echo.abs_max, s_echo.rel_max);
    fprintf(stderr,
        "[pbfdkf] P    abs=%.3e rel=%.3e\n", s_P.abs_max, s_P.rel_max);
    fprintf(stderr,
        "[pbfdkf] R    abs=%.3e rel=%.3e\n", s_R.abs_max, s_R.rel_max);

    free(near); free(far_buf); free(mu_arr); free(out_c);
    free(exp_out); free(exp_err); free(exp_echo); free(exp_P); free(exp_R);
    pbfdkf_v2_free(&p);

    /* Gate (Wave 2 unit, no reset modeled — caller must keep frames within
     * pre-reset window):
     *   output / echo / P / R: bit-exact (rel = 0)
     *   error_spec: fp32 ULP (abs < 1e-6, rel may explode where exp ~ 0) */
    int fail = (s_out.rel_max > 0.0 || s_echo.rel_max > 0.0 ||
                s_P.rel_max > 0.0 || s_R.rel_max > 0.0 ||
                s_err.abs_max > 1e-6);
    if (fail) { fprintf(stderr, "[pbfdkf] FAIL\n"); return 1; }
    fprintf(stderr, "[pbfdkf] PASS (within FFT-drift tolerance)\n");
    return 0;
}
