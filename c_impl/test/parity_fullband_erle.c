/* parity_fullband_erle.c — replay the binary golden from
 * python/diag/gen_fullband_erle_golden.py through the C FullBandErleEstimator
 * and assert exact (bit-for-bit float64 + exact int/bool) match per frame.
 * WS5 Phase 5.2 gate (the AecState fullband-ERLE layer).
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -I/Users/mingyu/Desktop/novatek/SE/AEC/c_impl/include \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/fullband_erle.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/aec3_scale.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/test/parity_fullband_erle.c \
 *       -lm -o /tmp/p_fberle
 *   python3 .../python/diag/gen_fullband_erle_golden.py /tmp/fberle_golden.bin
 *   /tmp/p_fberle /tmp/fberle_golden.bin
 */
#include "fullband_erle.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int chk_f64(double got, double exp, const char *tag, int frame,
                   int *mism, double *max_abs_diff) {
    if (got != exp) {
        double d = got - exp;
        if (d < 0) d = -d;
        if (d > *max_abs_diff) *max_abs_diff = d;
        (*mism)++;
        if (*mism <= 12) {
            fprintf(stderr, "frame %d %s: C=%.17g golden=%.17g (diff=%.3e)\n",
                    frame, tag, got, exp, d);
        }
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fberle_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_freqs, n_frames, i;
    int mism = 0;
    double max_abs_diff = 0.0;
    double g_thr, g_min_log2, g_max_lf_log2;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_freqs, 4) || !rd(f, &n_frames, 4)) return 2;
    if (!rd(f, &g_thr, 8) || !rd(f, &g_min_log2, 8) || !rd(f, &g_max_lf_log2, 8)) return 2;

    {
        float *x2 = malloc((size_t)n_freqs * sizeof(float));
        float *y2 = malloc((size_t)n_freqs * sizeof(float));
        float *e2 = malloc((size_t)n_freqs * sizeof(float));
        FullBandErleEstimator s;

        fb_erle_init(&s, n_freqs, 1.0, 4.0, 160);

        /* Sanity: the precomputed scalars must match the golden header. */
        chk_f64(s.x2_band_energy_threshold, g_thr, "x2_band_energy_threshold", -1,
                &mism, &max_abs_diff);
        chk_f64(s.min_erle_log2, g_min_log2, "min_erle_log2", -1, &mism, &max_abs_diff);
        chk_f64(s.max_erle_lf_log2, g_max_lf_log2, "max_erle_lf_log2", -1, &mism,
                &max_abs_diff);

        for (i = 0; i < n_frames; ++i) {
            uint8_t conv, q_valid, iel_valid;
            double exp_td_log2, exp_q, exp_iel, exp_inst_q, exp_max, exp_min;
            double exp_y2acum, exp_e2acum;
            int32_t exp_num_points, exp_hold;
            double got_q;
            int got_q_valid = 0;

            if (!rd(f, x2, (size_t)n_freqs * 4) ||
                !rd(f, y2, (size_t)n_freqs * 4) ||
                !rd(f, e2, (size_t)n_freqs * 4) ||
                !rd(f, &conv, 1) ||
                !rd(f, &exp_td_log2, 8) ||
                !rd(f, &q_valid, 1) || !rd(f, &exp_q, 8) ||
                !rd(f, &iel_valid, 1) || !rd(f, &exp_iel, 8) ||
                !rd(f, &exp_inst_q, 8) ||
                !rd(f, &exp_max, 8) || !rd(f, &exp_min, 8) ||
                !rd(f, &exp_y2acum, 8) || !rd(f, &exp_e2acum, 8) ||
                !rd(f, &exp_num_points, 4) || !rd(f, &exp_hold, 4)) {
                fprintf(stderr, "short read frame %d\n", i); return 2;
            }

            fb_erle_update(&s, x2, y2, e2, (int)conv);

            chk_f64(fb_erle_log2(&s), exp_td_log2, "erle_td_log2", i,
                    &mism, &max_abs_diff);

            got_q = fb_erle_get_inst_linear_quality_estimate(&s, &got_q_valid);
            if (got_q_valid != (int)q_valid) {
                fprintf(stderr, "frame %d quality_valid: C=%d golden=%d\n",
                        i, got_q_valid, (int)q_valid);
                mism++;
            }
            if (q_valid) {
                chk_f64(got_q, exp_q, "linear_quality", i, &mism, &max_abs_diff);
            }

            if (s.inst.has_erle_log2 != (int)iel_valid) {
                fprintf(stderr, "frame %d inst_erle_valid: C=%d golden=%d\n",
                        i, s.inst.has_erle_log2, (int)iel_valid);
                mism++;
            }
            if (iel_valid) {
                chk_f64(s.inst.erle_log2, exp_iel, "inst_erle_log2", i,
                        &mism, &max_abs_diff);
            }
            chk_f64(s.inst.inst_quality_estimate, exp_inst_q, "inst_quality_est", i,
                    &mism, &max_abs_diff);
            chk_f64(s.inst.max_erle_log2, exp_max, "inst_max_erle_log2", i,
                    &mism, &max_abs_diff);
            chk_f64(s.inst.min_erle_log2, exp_min, "inst_min_erle_log2", i,
                    &mism, &max_abs_diff);
            chk_f64(s.inst.y2_acum, exp_y2acum, "inst_y2_acum", i,
                    &mism, &max_abs_diff);
            chk_f64(s.inst.e2_acum, exp_e2acum, "inst_e2_acum", i,
                    &mism, &max_abs_diff);
            if (s.inst.num_points != exp_num_points) {
                fprintf(stderr, "frame %d num_points: C=%d golden=%d\n",
                        i, s.inst.num_points, (int)exp_num_points);
                mism++;
            }
            if (s.hold_counter_inst_erle != exp_hold) {
                fprintf(stderr, "frame %d hold_counter: C=%d golden=%d\n",
                        i, s.hold_counter_inst_erle, (int)exp_hold);
                mism++;
            }
        }
        fclose(f);

        printf("fullband_erle parity: %d frames, %d bins, mismatches=%d "
               "(max_abs_diff=%.3e)\n", n_frames, n_freqs, mism, max_abs_diff);

        free(x2); free(y2); free(e2);
    }

    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
