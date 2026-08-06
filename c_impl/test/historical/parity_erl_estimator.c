/* Historical parity_erl_estimator.c — replay the binary golden from
 * python/diag/gen_erl_estimator_golden.py through the C ErlEstimator and assert
 * exact (bit-for-bit float32 + exact int + bit-for-bit float64) match.
 * WS5 Phase 5.2 gate (the AecState ERL layer).
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include \
 *       c_impl/src/erl_estimator.c \
 *       c_impl/src/reverb_frequency_response.c \
 *       c_impl/src/aec3_scale.c \
 *       c_impl/test/historical/parity_erl_estimator.c \
 *       -lm -o /tmp/p_erl
 *   python3 .../python/diag/gen_erl_estimator_golden.py /tmp/erl_golden.bin
 *   /tmp/p_erl /tmp/erl_golden.bin
 */
#include "erl_estimator.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/erl_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_bins, n_frames, startup_hops, hold_hops;
    double x2_min;
    int i, k, mism = 0;
    double max_abs_diff = 0.0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_bins, 4) || !rd(f, &n_frames, 4) ||
        !rd(f, &startup_hops, 4) || !rd(f, &hold_hops, 4) ||
        !rd(f, &x2_min, 8)) return 2;

    {
        /* C-side storage */
        float   *erl_st   = malloc((size_t)n_bins * sizeof(float));
        int     *hold_st  = malloc((size_t)(n_bins - 2) * sizeof(int));
        /* golden buffers */
        float   *x2       = malloc((size_t)n_bins * sizeof(float));
        float   *y2       = malloc((size_t)n_bins * sizeof(float));
        float   *exp_erl  = malloc((size_t)n_bins * sizeof(float));
        int32_t *exp_hold = malloc((size_t)(n_bins - 2) * sizeof(int32_t));
        int32_t  conv, exp_hold_td;
        double   exp_erl_td;

        ErlEstimator e;
        erl_estimator_init(&e, startup_hops, n_bins, 160, 16000, erl_st, hold_st);

        /* x2_min parity (init derives it from aec3_per_bin_psd_threshold). */
        if (e.x2_min != x2_min) {
            fprintf(stderr, "x2_min mismatch: C %.17g golden %.17g\n",
                    e.x2_min, x2_min);
            mism++;
        }

        for (i = 0; i < n_frames; ++i) {
            if (!rd(f, x2, (size_t)n_bins * 4) ||
                !rd(f, y2, (size_t)n_bins * 4) ||
                !rd(f, &conv, 4) ||
                !rd(f, exp_erl, (size_t)n_bins * 4) ||
                !rd(f, exp_hold, (size_t)(n_bins - 2) * 4) ||
                !rd(f, &exp_erl_td, 8) ||
                !rd(f, &exp_hold_td, 4)) {
                fprintf(stderr, "short read frame %d\n", i); return 2;
            }

            erl_estimator_update(&e, x2, y2, (int)conv);

            for (k = 0; k < n_bins; ++k) {
                if (e.erl[k] != exp_erl[k]) {
                    double d = (double)e.erl[k] - (double)exp_erl[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) max_abs_diff = d;
                    mism++;
                }
            }
            for (k = 0; k < n_bins - 2; ++k) {
                if (e.hold_counters[k] != exp_hold[k]) mism++;
            }
            if (e.erl_time_domain != exp_erl_td) {
                double d = e.erl_time_domain - exp_erl_td;
                if (d < 0) d = -d;
                if (d > max_abs_diff) max_abs_diff = d;
                mism++;
            }
            if (e.hold_counter_time_domain != (int)exp_hold_td) mism++;
        }
        fclose(f);

        printf("erl_estimator parity: %d frames, %d bins, startup=%d hold=%d, "
               "mismatches=%d (max_abs_diff=%.3e)\n",
               n_frames, n_bins, startup_hops, hold_hops, mism, max_abs_diff);

        free(erl_st); free(hold_st);
        free(x2); free(y2); free(exp_erl); free(exp_hold);
    }

    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
