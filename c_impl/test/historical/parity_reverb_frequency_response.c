/* Historical parity_reverb_frequency_response.c — replay the binary golden from
 * python/diag/gen_reverb_frequency_response_golden.py through the C
 * ReverbFrequencyResponse and assert exact (bit-for-bit) match of
 * tail_response (float32) and average_decay (float64). WS5 Phase 5.1 gate.
 *
 * Build (standalone, from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       src/reverb_frequency_response.c test/historical/parity_reverb_frequency_response.c \
 *       -lm -o /tmp/p_rfr
 *   python3 ../python/diag/gen_reverb_frequency_response_golden.py /tmp/rfr_golden.bin
 *   /tmp/p_rfr /tmp/rfr_golden.bin
 */
#include "reverb_frequency_response.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/rfr_golden.bin";
    FILE *f = fopen(path, "rb");
    int32_t n_freqs, n_part, n_configs;
    int c, i, k;
    int mism = 0, calls = 0;
    float *fr, *tail_store, *exp_tail;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_freqs, 4) || !rd(f, &n_part, 4) || !rd(f, &n_configs, 4)) {
        fprintf(stderr, "short read header\n"); return 2;
    }

    fr        = malloc((size_t)n_part * (size_t)n_freqs * sizeof(float));
    tail_store= malloc((size_t)n_freqs * sizeof(float));
    exp_tail  = malloc((size_t)n_freqs * sizeof(float));
    if (!fr || !tail_store || !exp_tail) { fprintf(stderr, "oom\n"); return 2; }

    for (c = 0; c < n_configs; ++c) {
        int32_t use_cons, n_calls;
        double smoothing_base;
        ReverbFrequencyResponse r;

        if (!rd(f, &use_cons, 4) || !rd(f, &smoothing_base, 8) ||
            !rd(f, &n_calls, 4)) { fprintf(stderr, "short read cfg %d\n", c); return 2; }

        reverb_freq_resp_init(&r, tail_store, n_freqs, (int)use_cons, smoothing_base);

        for (i = 0; i < n_calls; ++i) {
            int32_t fdb, q_is_none, stationary;
            double quality, exp_decay;

            if (!rd(f, fr, (size_t)n_part * (size_t)n_freqs * sizeof(float)) ||
                !rd(f, &fdb, 4) || !rd(f, &q_is_none, 4) || !rd(f, &quality, 8) ||
                !rd(f, &stationary, 4) || !rd(f, &exp_decay, 8) ||
                !rd(f, exp_tail, (size_t)n_freqs * sizeof(float))) {
                fprintf(stderr, "short read cfg %d call %d\n", c, i); return 2;
            }

            reverb_freq_resp_update(&r, fr, (int)n_part, (int)fdb,
                                    quality, (int)q_is_none, (int)stationary);

            /* average_decay: exact float64 compare */
            if (memcmp(&r.average_decay, &exp_decay, sizeof(double)) != 0) {
                if (mism < 10)
                    fprintf(stderr, "cfg %d call %d: decay C=%.17g py=%.17g\n",
                            c, i, r.average_decay, exp_decay);
                mism++;
            }
            /* tail_response: exact float32 compare per bin */
            for (k = 0; k < n_freqs; ++k) {
                if (r.tail_response[k] != exp_tail[k]) {
                    if (mism < 10)
                        fprintf(stderr, "cfg %d call %d bin %d: C=%.9g py=%.9g\n",
                                c, i, k, (double)r.tail_response[k],
                                (double)exp_tail[k]);
                    mism++;
                }
            }
            calls++;
        }
    }
    fclose(f);
    free(fr); free(tail_store); free(exp_tail);

    printf("reverb_frequency_response parity: %d calls, %d freqs, %d configs, "
           "mismatches=%d\n", calls, n_freqs, n_configs, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
