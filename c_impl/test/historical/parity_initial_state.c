/* Historical parity_initial_state.c — replay the binary golden from
 * python/diag/gen_initial_state_golden.py through the C InitialState and assert
 * exact (integer/bool) match across every frame of every config, including the
 * constructor threshold, the transition edge, saturation-gated counter, and a
 * mid-run reset(). WS5 Phase 5.2 gate.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include \
 *       c_impl/src/initial_state.c \
 *       c_impl/test/historical/parity_initial_state.c \
 *       -o /tmp/p_istate
 *   python3 .../python/diag/gen_initial_state_golden.py /tmp/initial_state_golden.bin
 *   /tmp/p_istate /tmp/initial_state_golden.bin
 */
#include "initial_state.h"

#include <stdint.h>
#include <stdio.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/initial_state_golden.bin";
    FILE *f = fopen(path, "rb");
    int32_t n_cfgs;
    int c, i, mism = 0;
    int total_frames = 0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_cfgs, 4)) return 2;

    for (c = 0; c < n_cfgs; ++c) {
        int32_t cons, thr_hops, cons_hops, n_frames, reset_at;
        double  secs;
        InitialState s;

        if (!rd(f, &cons, 4) || !rd(f, &secs, 8) || !rd(f, &thr_hops, 4) ||
            !rd(f, &cons_hops, 4) || !rd(f, &n_frames, 4) || !rd(f, &reset_at, 4)) {
            fprintf(stderr, "short header cfg %d\n", c); return 2;
        }

        initial_state_init(&s, cons, secs, 160, 16000);

        /* constructor-derived thresholds must match the Python instance */
        if (s.initial_state_hops != thr_hops) {
            fprintf(stderr, "cfg %d threshold mismatch: C=%d golden=%d (secs=%g)\n",
                    c, s.initial_state_hops, thr_hops, secs);
            mism++;
        }
        if (s.conservative_hops != cons_hops) {
            fprintf(stderr, "cfg %d conservative_hops mismatch: C=%d golden=%d\n",
                    c, s.conservative_hops, cons_hops);
            mism++;
        }

        for (i = 0; i < n_frames; ++i) {
            uint8_t active, sat, exp_init, exp_trig;
            int32_t exp_cnt;

            if (!rd(f, &active, 1) || !rd(f, &sat, 1) || !rd(f, &exp_cnt, 4) ||
                !rd(f, &exp_init, 1) || !rd(f, &exp_trig, 1)) {
                fprintf(stderr, "short read cfg %d frame %d\n", c, i); return 2;
            }

            if (reset_at >= 0 && i == reset_at) {
                initial_state_reset(&s);
            }
            initial_state_update(&s, active, sat);

            if (s.strong_not_saturated_render_blocks != exp_cnt) mism++;
            if (initial_state_initial_state_active(&s) != (int)exp_init) mism++;
            if (initial_state_transition_triggered(&s) != (int)exp_trig) mism++;
            total_frames++;
        }
    }
    fclose(f);

    printf("initial_state parity: %d configs, %d frames total, mismatches=%d\n",
           n_cfgs, total_frames, mism);

    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
