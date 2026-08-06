/* Historical parity_filter_quality.c — replay the binary golden from
 * python/diag/gen_filter_quality_golden.py through the C
 * FilteringQualityAnalyzer and assert exact match on the full observable state
 * (linear_filter_usable / gate1 / gate2 / startup_blocks / reset_blocks /
 *  convergence_seen / convergence_hops_counter / overall_usable) over a
 * multi-frame sequence (state evolution incl. reset semantics). WS5 Phase 5.2.
 *
 * Build (standalone, from repo root):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include c_impl/src/filter_quality.c \
 *       c_impl/test/historical/parity_filter_quality.c -o /tmp/p_fq
 *   python3 python/diag/gen_filter_quality_golden.py /tmp/fq_golden.bin
 *   /tmp/p_fq /tmp/fq_golden.bin
 */
#include "filter_quality.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fq_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_steps, s, mism = 0;
    FilteringQualityAnalyzer m;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_steps, 4)) return 2;

    fq_init(&m, 1 /* use_linear_filter */, 160, 16000);

    for (s = 0; s < n_steps; ++s) {
        int32_t in[6];
        int32_t exp[8];
        int got[8];
        int k;

        if (!rd(f, in, sizeof(in))) {
            fprintf(stderr, "short read inputs step %d\n", s); return 2;
        }
        if (!rd(f, exp, sizeof(exp))) {
            fprintf(stderr, "short read expected step %d\n", s); return 2;
        }

        if (in[5]) fq_reset(&m);          /* do_reset BEFORE update */
        fq_update(&m,
                  in[0],                  /* active_render            */
                  in[1],                  /* transparent_mode         */
                  in[2],                  /* saturated_capture        */
                  in[3],                  /* external_delay_present   */
                  in[4]);                 /* any_filter_converged     */

        got[0] = fq_linear_filter_usable(&m);
        got[1] = fq_gate1_pass(&m);
        got[2] = fq_gate2_pass(&m);
        got[3] = fq_startup_blocks(&m);
        got[4] = fq_reset_blocks(&m);
        got[5] = m.convergence_seen;
        got[6] = m.convergence_hops_counter;
        got[7] = m.overall_usable;

        for (k = 0; k < 8; ++k) {
            if (got[k] != (int)exp[k]) {
                if (mism < 20)
                    fprintf(stderr, "step %d field[%d] c=%d golden=%d\n",
                            s, k, got[k], (int)exp[k]);
                mism++;
            }
        }
    }
    fclose(f);
    printf("filter_quality parity: %d steps, mismatches=%d\n", n_steps, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
