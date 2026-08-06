/* Historical parity_filter_delay.c — replay the binary golden from
 * python/diag/gen_filter_delay_golden.py through the C FilterDelay and assert
 * exact (integer) match of filter_delays_blocks / min_direct_path /
 * external_delay_reported and the analyzer length-mismatch error path, over a
 * multi-frame state-evolution sequence. WS5 Phase 5.2 gate.
 *
 * Build (standalone, from repo root):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include c_impl/src/filter_delay.c \
 *       c_impl/test/historical/parity_filter_delay.c -lm -o /tmp/p_fd
 *   python3 python/diag/gen_filter_delay_golden.py /tmp/fd_golden.bin
 *   /tmp/p_fd /tmp/fd_golden.bin
 */
#include "filter_delay.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int rd32(FILE *f, int32_t *v) { return rd(f, v, 4); }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fd_golden.bin";
    FILE *f = fopen(path, "rb");
    int32_t n_configs;
    int c, calls_total = 0, frames_total = 0, mism = 0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd32(f, &n_configs)) { fprintf(stderr, "short read header\n"); return 2; }

    for (c = 0; c < n_configs; ++c) {
        int32_t headroom, nch, n_calls;
        int32_t init_min, init_ext, i, k;
        int *delays_store, *exp_delays, *analyzer;
        FilterDelay fd;

        if (!rd32(f, &headroom) || !rd32(f, &nch) || !rd32(f, &n_calls)) {
            fprintf(stderr, "short read cfg %d header\n", c); return 2;
        }

        delays_store = malloc((size_t)nch * sizeof(int));
        exp_delays   = malloc((size_t)nch * sizeof(int));
        analyzer     = malloc((size_t)(nch + 4) * sizeof(int)); /* slack for +2 */
        if (!delays_store || !exp_delays || !analyzer) {
            fprintf(stderr, "oom\n"); return 2;
        }

        filter_delay_init(&fd, delays_store, (int)headroom, (int)nch, 160, 16000);

        /* init snapshot */
        if (!rd32(f, &init_min) || !rd32(f, &init_ext)) {
            fprintf(stderr, "short read cfg %d init\n", c); return 2;
        }
        for (k = 0; k < nch; ++k) {
            if (!rd32(f, &exp_delays[k])) {
                fprintf(stderr, "short read cfg %d init delays\n", c); return 2;
            }
        }
        if (filter_delay_min_direct_path(&fd) != init_min) {
            if (mism < 20)
                fprintf(stderr, "cfg %d INIT min c=%d golden=%d\n",
                        c, filter_delay_min_direct_path(&fd), init_min);
            mism++;
        }
        if (filter_delay_external_reported(&fd) != init_ext) {
            if (mism < 20)
                fprintf(stderr, "cfg %d INIT ext c=%d golden=%d\n",
                        c, filter_delay_external_reported(&fd), init_ext);
            mism++;
        }
        for (k = 0; k < nch; ++k) {
            if (fd.filter_delays_blocks[k] != exp_delays[k]) {
                if (mism < 20)
                    fprintf(stderr, "cfg %d INIT delay[%d] c=%d golden=%d\n",
                            c, k, fd.filter_delays_blocks[k], exp_delays[k]);
                mism++;
            }
        }
        frames_total++;

        for (i = 0; i < n_calls; ++i) {
            int32_t has_an, an_len, ext_rep, ext_q, ext_d, blocks;
            int32_t exp_err, exp_min, exp_ext;
            FilterDelayEstimate ext;
            const int *an_ptr;
            int an_len_arg;
            int rc, j;

            if (!rd32(f, &has_an) || !rd32(f, &an_len)) {
                fprintf(stderr, "short read cfg %d call %d ahdr\n", c, i); return 2;
            }
            if (has_an) {
                for (j = 0; j < an_len; ++j) {
                    if (!rd32(f, &analyzer[j])) {
                        fprintf(stderr, "short read cfg %d call %d an\n", c, i);
                        return 2;
                    }
                }
            }
            if (!rd32(f, &ext_rep) || !rd32(f, &ext_q) || !rd32(f, &ext_d) ||
                !rd32(f, &blocks)) {
                fprintf(stderr, "short read cfg %d call %d in\n", c, i); return 2;
            }
            if (!rd32(f, &exp_err) || !rd32(f, &exp_min) || !rd32(f, &exp_ext)) {
                fprintf(stderr, "short read cfg %d call %d exp\n", c, i); return 2;
            }
            for (k = 0; k < nch; ++k) {
                if (!rd32(f, &exp_delays[k])) {
                    fprintf(stderr, "short read cfg %d call %d exp delays\n",
                            c, i); return 2;
                }
            }

            ext.reported = (int)ext_rep;
            ext.quality  = (int)ext_q;
            ext.delay    = (int)ext_d;

            an_ptr     = has_an ? analyzer : NULL;
            an_len_arg = has_an ? (int)an_len : 0;

            rc = filter_delay_update(&fd, an_ptr, an_len_arg,
                                     ext_rep ? &ext : NULL, (int)blocks);

            /* error path: rc == -1 must equal golden expect_error */
            {
                int c_err = (rc != 0) ? 1 : 0;
                if (c_err != exp_err) {
                    if (mism < 20)
                        fprintf(stderr, "cfg %d call %d ERR c=%d golden=%d\n",
                                c, i, c_err, exp_err);
                    mism++;
                }
            }
            if (filter_delay_min_direct_path(&fd) != exp_min) {
                if (mism < 20)
                    fprintf(stderr, "cfg %d call %d min c=%d golden=%d\n",
                            c, i, filter_delay_min_direct_path(&fd), exp_min);
                mism++;
            }
            if (filter_delay_external_reported(&fd) != exp_ext) {
                if (mism < 20)
                    fprintf(stderr, "cfg %d call %d ext c=%d golden=%d\n",
                            c, i, filter_delay_external_reported(&fd), exp_ext);
                mism++;
            }
            for (k = 0; k < nch; ++k) {
                if (fd.filter_delays_blocks[k] != exp_delays[k]) {
                    if (mism < 20)
                        fprintf(stderr,
                                "cfg %d call %d delay[%d] c=%d golden=%d\n",
                                c, i, k, fd.filter_delays_blocks[k],
                                exp_delays[k]);
                    mism++;
                }
            }
            calls_total++;
            frames_total++;
        }

        free(delays_store);
        free(exp_delays);
        free(analyzer);
    }
    fclose(f);

    printf("filter_delay parity: %d configs, %d update calls, %d state frames, "
           "mismatches=%d\n", (int)n_configs, calls_total, frames_total, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
