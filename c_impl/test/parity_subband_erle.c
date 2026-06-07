/* parity_subband_erle.c — replay the binary golden from
 * python/diag/gen_subband_erle_golden.py through the C SubbandErle and assert
 * exact (bit-for-bit float32 + exact int/bool) match. WS5 Phase 5.2 gate.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I/Users/mingyu/Desktop/novatek/SE/AEC/c_impl/include \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/subband_erle.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/aec3_scale.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/test/parity_subband_erle.c \
 *       -lm -o /tmp/p_se
 *   python3 .../python/diag/gen_subband_erle_golden.py /tmp/se_golden.bin
 *   /tmp/p_se /tmp/se_golden.bin
 */
#include "subband_erle.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* Compare a float32 array bit-for-bit; bump *mism / *maxd on any difference. */
static void cmp_f32(const float *got, const float *exp, int n,
                    int *mism, double *maxd) {
    int k;
    for (k = 0; k < n; ++k) {
        if (got[k] != exp[k]) {
            double d = (double)got[k] - (double)exp[k];
            if (d < 0) d = -d;
            if (d > *maxd) *maxd = d;
            (*mism)++;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/se_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_variants, vi;
    int total_frames = 0, total_mism = 0;
    double max_abs_diff = 0.0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_variants, 4)) { fprintf(stderr, "short read header\n"); return 2; }

    for (vi = 0; vi < n_variants; ++vi) {
        int n_bins, n_frames;
        int use_onset, use_min_during, coh_active;
        float cfg_f[3];     /* max_erle_l, max_erle_h, min_erle */
        double x2_thr;
        int i, k;

        if (!rd(f, &n_bins, 4) || !rd(f, &n_frames, 4) ||
            !rd(f, &use_onset, 4) || !rd(f, &use_min_during, 4) ||
            !rd(f, &coh_active, 4) ||
            !rd(f, cfg_f, sizeof(cfg_f)) || !rd(f, &x2_thr, 8)) {
            fprintf(stderr, "short read variant %d header\n", vi); return 2;
        }

        {
            /* C-side storage */
            float *max_erle_st = malloc((size_t)n_bins * sizeof(float));
            float *erle_st     = malloc((size_t)n_bins * sizeof(float));
            float *erle_oc_st  = malloc((size_t)n_bins * sizeof(float));
            float *erle_unb_st = malloc((size_t)n_bins * sizeof(float));
            float *erle_dur_st = malloc((size_t)n_bins * sizeof(float));
            unsigned char *coming_st = malloc((size_t)n_bins);
            int32_t *hold_st   = malloc((size_t)n_bins * sizeof(int32_t));
            float *y2acc_st    = malloc((size_t)n_bins * sizeof(float));
            float *e2acc_st    = malloc((size_t)n_bins * sizeof(float));
            unsigned char *low_st = malloc((size_t)n_bins);
            /* golden input buffers */
            float *x2  = malloc((size_t)n_bins * sizeof(float));
            float *y2  = malloc((size_t)n_bins * sizeof(float));
            float *e2  = malloc((size_t)n_bins * sizeof(float));
            unsigned char *coh = malloc((size_t)n_bins);
            /* golden expected buffers */
            float *e_erle  = malloc((size_t)n_bins * sizeof(float));
            float *e_oc    = malloc((size_t)n_bins * sizeof(float));
            float *e_unb   = malloc((size_t)n_bins * sizeof(float));
            float *e_dur   = malloc((size_t)n_bins * sizeof(float));
            unsigned char *e_coming = malloc((size_t)n_bins);
            int32_t *e_hold = malloc((size_t)n_bins * sizeof(int32_t));
            unsigned char *e_low = malloc((size_t)n_bins);
            float *e_y2acc = malloc((size_t)n_bins * sizeof(float));
            float *e_e2acc = malloc((size_t)n_bins * sizeof(float));
            int32_t e_np;

            SubbandErle s;
            subband_erle_init(&s, n_bins,
                              cfg_f[2] /*min_erle*/, cfg_f[0] /*l*/, cfg_f[1] /*h*/,
                              use_onset, use_min_during, 160 /*hop*/,
                              max_erle_st, erle_st, erle_oc_st, erle_unb_st,
                              erle_dur_st, coming_st, hold_st,
                              y2acc_st, e2acc_st, low_st);

            if (s.x2_band_energy_threshold != x2_thr) {
                fprintf(stderr, "variant %d x2_thr mismatch: C %.17g golden %.17g\n",
                        vi, s.x2_band_energy_threshold, x2_thr);
                total_mism++;
            }

            for (i = 0; i < n_frames; ++i) {
                int conv_i;
                int32_t np_i;
                if (!rd(f, x2, (size_t)n_bins * 4) ||
                    !rd(f, y2, (size_t)n_bins * 4) ||
                    !rd(f, e2, (size_t)n_bins * 4) ||
                    !rd(f, &conv_i, 4)) {
                    fprintf(stderr, "short read v%d frame %d inputs\n", vi, i);
                    return 2;
                }
                if (coh_active && !rd(f, coh, (size_t)n_bins)) {
                    fprintf(stderr, "short read v%d frame %d coh\n", vi, i);
                    return 2;
                }
                if (!rd(f, e_erle, (size_t)n_bins * 4) ||
                    !rd(f, e_oc, (size_t)n_bins * 4) ||
                    !rd(f, e_unb, (size_t)n_bins * 4) ||
                    !rd(f, e_dur, (size_t)n_bins * 4) ||
                    !rd(f, e_coming, (size_t)n_bins) ||
                    !rd(f, e_hold, (size_t)n_bins * 4) ||
                    !rd(f, e_low, (size_t)n_bins) ||
                    !rd(f, e_y2acc, (size_t)n_bins * 4) ||
                    !rd(f, e_e2acc, (size_t)n_bins * 4) ||
                    !rd(f, &e_np, 4)) {
                    fprintf(stderr, "short read v%d frame %d expected\n", vi, i);
                    return 2;
                }

                subband_erle_update(&s, x2, y2, e2, conv_i,
                                    coh_active ? coh : NULL);

                cmp_f32(s.erle, e_erle, n_bins, &total_mism, &max_abs_diff);
                cmp_f32(s.erle_onset_compensated, e_oc, n_bins, &total_mism, &max_abs_diff);
                cmp_f32(s.erle_unbounded, e_unb, n_bins, &total_mism, &max_abs_diff);
                cmp_f32(s.erle_during_onsets, e_dur, n_bins, &total_mism, &max_abs_diff);
                cmp_f32(s.y2_acc, e_y2acc, n_bins, &total_mism, &max_abs_diff);
                cmp_f32(s.e2_acc, e_e2acc, n_bins, &total_mism, &max_abs_diff);
                for (k = 0; k < n_bins; ++k) {
                    if (s.coming_onset[k] != e_coming[k]) total_mism++;
                    if (s.hold_counters[k] != e_hold[k]) total_mism++;
                    if (s.low_render_energy[k] != e_low[k]) total_mism++;
                }
                np_i = (int32_t)s.num_points;
                if (np_i != e_np) total_mism++;
                total_frames++;
            }

            free(max_erle_st); free(erle_st); free(erle_oc_st); free(erle_unb_st);
            free(erle_dur_st); free(coming_st); free(hold_st);
            free(y2acc_st); free(e2acc_st); free(low_st);
            free(x2); free(y2); free(e2); free(coh);
            free(e_erle); free(e_oc); free(e_unb); free(e_dur);
            free(e_coming); free(e_hold); free(e_low); free(e_y2acc); free(e_e2acc);
        }
    }
    fclose(f);

    printf("subband_erle parity: %d variants, %d total frames, "
           "mismatches=%d (max_abs_diff=%.3e)\n",
           n_variants, total_frames, total_mism, max_abs_diff);

    if (total_mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
