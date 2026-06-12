/* parity_stationarity_estimator.c — replay the binary golden from
 * python/diag/gen_stationarity_estimator_golden.py through the C
 * StationarityEstimator and assert exact (bit-for-bit float32 + exact int/bool)
 * match. WS5 Phase 5.1 gate.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I<path-to-repo>/c_impl/include \
 *       <path-to-repo>/c_impl/src/stationarity_estimator.c \
 *       <path-to-repo>/c_impl/src/aec3_scale.c \
 *       <path-to-repo>/c_impl/test/parity_stationarity_estimator.c \
 *       -lm -o /tmp/p_stat
 *   python3 .../python/diag/gen_stationarity_estimator_golden.py /tmp/stat_golden.bin
 *   /tmp/p_stat /tmp/stat_golden.bin
 */
#include "stationarity_estimator.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/stat_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_freqs, n_frames, window_hops, hangover_hops;
    int i, k, mism = 0;
    double max_abs_diff = 0.0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_freqs, 4) || !rd(f, &n_frames, 4) ||
        !rd(f, &window_hops, 4) || !rd(f, &hangover_hops, 4)) return 2;

    {
        /* C-side storage */
        float        *noise_st   = malloc((size_t)n_freqs * sizeof(float));
        int32_t      *hang_st     = malloc((size_t)n_freqs * sizeof(int32_t));
        unsigned char *flags_st   = malloc((size_t)n_freqs);
        float        *hist_st     = malloc((size_t)window_hops * n_freqs * sizeof(float));
        unsigned char *mask_c     = malloc((size_t)n_freqs);
        /* golden buffers */
        float        *psd         = malloc((size_t)n_freqs * sizeof(float));
        float        *exp_noise   = malloc((size_t)n_freqs * sizeof(float));
        unsigned char *exp_flags  = malloc((size_t)n_freqs);
        int32_t      *exp_hang    = malloc((size_t)n_freqs * sizeof(int32_t));
        unsigned char *exp_mask   = malloc((size_t)n_freqs);
        int32_t       exp_blk;

        StationarityEstimator s;
        stationarity_estimator_init(&s, n_freqs, 160, 16000,
                                    noise_st, hang_st, flags_st, hist_st);

        if (s.window_hops != window_hops || s.hangover_hops != hangover_hops) {
            fprintf(stderr, "hop mismatch: C win=%d hang=%d golden win=%d hang=%d\n",
                    s.window_hops, s.hangover_hops, window_hops, hangover_hops);
            mism++;
        }

        for (i = 0; i < n_frames; ++i) {
            if (!rd(f, psd, (size_t)n_freqs * 4) ||
                !rd(f, exp_noise, (size_t)n_freqs * 4) ||
                !rd(f, exp_flags, (size_t)n_freqs) ||
                !rd(f, exp_hang, (size_t)n_freqs * 4) ||
                !rd(f, exp_mask, (size_t)n_freqs) ||
                !rd(f, &exp_blk, 4)) {
                fprintf(stderr, "short read frame %d\n", i); return 2;
            }

            stationarity_estimator_update_noise_estimator(&s, psd);
            stationarity_estimator_update_stationarity_flags(&s, psd, NULL);
            stationarity_estimator_band_stationary_mask(&s, mask_c);

            for (k = 0; k < n_freqs; ++k) {
                if (s.noise.noise[k] != exp_noise[k]) {
                    double d = (double)s.noise.noise[k] - (double)exp_noise[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) max_abs_diff = d;
                    mism++;
                }
                if (s.stationarity_flags[k] != exp_flags[k]) mism++;
                if (s.hangovers[k] != exp_hang[k]) mism++;
                if (mask_c[k] != exp_mask[k]) mism++;
            }
            {
                int blk_c = stationarity_estimator_is_block_stationary(&s);
                if (blk_c != (int)exp_blk) mism++;
            }
        }
        fclose(f);

        printf("stationarity_estimator parity: %d frames, %d bins, "
               "window=%d hangover=%d, mismatches=%d (max_abs_diff=%.3e)\n",
               n_frames, n_freqs, window_hops, hangover_hops, mism, max_abs_diff);

        free(noise_st); free(hang_st); free(flags_st); free(hist_st); free(mask_c);
        free(psd); free(exp_noise); free(exp_flags); free(exp_hang); free(exp_mask);
    }

    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
