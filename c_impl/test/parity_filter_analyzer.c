/* parity_filter_analyzer.c — replay the binary golden from
 * python/diag/gen_filter_analyzer_golden.py through the C FilterAnalyzer and
 * assert bit-exact match on the high-pass-filtered taps + peak_index +
 * delay_blocks + gain + consistent flag + the ConsistentFilterDetector
 * internal accumulators, for every per-hop step. WS5 Phase 5.2 gate.
 *
 * Build (standalone, from repo root):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include c_impl/src/filter_analyzer.c \
 *       c_impl/test/parity_filter_analyzer.c -lm -o /tmp/p_fa
 *   python3 python/diag/gen_filter_analyzer_golden.py /tmp/fa_golden.bin
 *   /tmp/p_fa /tmp/fa_golden.bin
 *
 * The golden is fully self-describing: each step carries a `reset_before` flag
 * so the replay calls fa_reset() at exactly the position the Python generator
 * did (mid-sweep reset path coverage).
 */
#include "filter_analyzer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fa_golden.bin";
    FILE *f = fopen(path, "rb");
    int size, hop, n_steps;
    int s, k, mism = 0;
    float *taps, *block, *exp_h, *h_storage, *abs_scratch, *render_sq_scratch;
    FilterAnalyzer m;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &size, 4) || !rd(f, &hop, 4) || !rd(f, &n_steps, 4)) return 2;

    taps      = malloc((size_t)size * sizeof(float));
    block     = malloc((size_t)hop * sizeof(float));
    exp_h     = malloc((size_t)size * sizeof(float));
    h_storage = malloc((size_t)size * sizeof(float));
    /* M3: fa_update()'s former fixed-size stack scratch is now caller-owned
     * (pool-carved in production); this standalone harness mallocs its own
     * copies sized exactly to what fa_update requires (size / hop). */
    abs_scratch       = malloc((size_t)size * sizeof(float));
    render_sq_scratch = malloc((size_t)hop * sizeof(float));
    if (!taps || !block || !exp_h || !h_storage || !abs_scratch || !render_sq_scratch) {
        fprintf(stderr, "oom\n"); return 2;
    }

    /* Defaults match the Python FilterAnalyzer() ctor. */
    fa_init(&m, h_storage, size, 100.0 /*active_render_limit*/,
            0 /*bounded_erl*/, 1.0 /*default_gain*/,
            abs_scratch, size, render_sq_scratch, hop, hop, 16000);

    for (s = 0; s < n_steps; ++s) {
        int e_peak, e_delay, e_cons, e_min, e_any, e_sig, e_flo, e_fhi, e_dref;
        int reset_before;
        double e_gain, e_epg, e_facc, e_speak;
        int64_t e_counter;
        int got;

        if (!rd(f, &reset_before, 4)) {
            fprintf(stderr, "short read reset flag step %d\n", s); return 2;
        }
        if (reset_before) {
            fa_reset(&m);
        }
        if (!rd(f, taps, (size_t)size * sizeof(float)) ||
            !rd(f, block, (size_t)hop * sizeof(float)) ||
            !rd(f, exp_h, (size_t)size * sizeof(float))) {
            fprintf(stderr, "short read inputs step %d\n", s); return 2;
        }
        got = rd(f, &e_peak, 4) && rd(f, &e_delay, 4) && rd(f, &e_cons, 4) &&
              rd(f, &e_min, 4) && rd(f, &e_any, 4) &&
              rd(f, &e_gain, 8) && rd(f, &e_epg, 8) &&
              rd(f, &e_sig, 4) &&
              rd(f, &e_facc, 8) && rd(f, &e_speak, 8) &&
              rd(f, &e_flo, 4) && rd(f, &e_fhi, 4) &&
              rd(f, &e_counter, 8) && rd(f, &e_dref, 4);
        if (!got) { fprintf(stderr, "short read expected step %d\n", s); return 2; }

        fa_update(&m, taps, block, hop);

        /* high-pass taps (exact bit pattern) */
        for (k = 0; k < size; ++k) {
            if (m.h_highpass[k] != exp_h[k]) {
                if (mism < 20)
                    fprintf(stderr, "step %d h[%d] c=%.9g golden=%.9g\n",
                            s, k, (double)m.h_highpass[k], (double)exp_h[k]);
                mism++;
            }
        }
        /* peak_index / delay_blocks */
        if (m.peak_index != e_peak) {
            if (mism < 20) fprintf(stderr, "step %d peak_index c=%d golden=%d\n",
                                   s, m.peak_index, e_peak);
            mism++;
        }
        if (m.delay_blocks != e_delay) {
            if (mism < 20) fprintf(stderr, "step %d delay_blocks c=%d golden=%d\n",
                                   s, m.delay_blocks, e_delay);
            mism++;
        }
        if (m.consistent_estimate != e_cons) {
            if (mism < 20) fprintf(stderr, "step %d consistent c=%d golden=%d\n",
                                   s, m.consistent_estimate, e_cons);
            mism++;
        }
        if (fa_min_filter_delay_blocks(&m) != e_min) {
            if (mism < 20) fprintf(stderr, "step %d min_delay c=%d golden=%d\n",
                                   s, fa_min_filter_delay_blocks(&m), e_min);
            mism++;
        }
        if (fa_any_filter_consistent(&m) != e_any) {
            if (mism < 20) fprintf(stderr, "step %d any_consistent c=%d golden=%d\n",
                                   s, fa_any_filter_consistent(&m), e_any);
            mism++;
        }
        /* gain / max_echo_path_gain (exact f64 bit pattern) */
        if (m.gain != e_gain) {
            if (mism < 20) fprintf(stderr, "step %d gain c=%.17g golden=%.17g\n",
                                   s, m.gain, e_gain);
            mism++;
        }
        if (fa_max_echo_path_gain(&m) != e_epg) {
            if (mism < 20) fprintf(stderr, "step %d max_epg c=%.17g golden=%.17g\n",
                                   s, fa_max_echo_path_gain(&m), e_epg);
            mism++;
        }
        /* ConsistentFilterDetector internals */
        if (m.consistent.significant_peak != e_sig) {
            if (mism < 20) fprintf(stderr, "step %d cd_sig c=%d golden=%d\n",
                                   s, m.consistent.significant_peak, e_sig);
            mism++;
        }
        if (m.consistent.floor_accum != e_facc) {
            if (mism < 20) fprintf(stderr, "step %d cd_floor_accum c=%.17g golden=%.17g\n",
                                   s, m.consistent.floor_accum, e_facc);
            mism++;
        }
        if (m.consistent.secondary_peak != e_speak) {
            if (mism < 20) fprintf(stderr, "step %d cd_secondary c=%.17g golden=%.17g\n",
                                   s, m.consistent.secondary_peak, e_speak);
            mism++;
        }
        if (m.consistent.floor_low_limit != e_flo) {
            if (mism < 20) fprintf(stderr, "step %d cd_floor_low c=%d golden=%d\n",
                                   s, m.consistent.floor_low_limit, e_flo);
            mism++;
        }
        if (m.consistent.floor_high_limit != e_fhi) {
            if (mism < 20) fprintf(stderr, "step %d cd_floor_high c=%d golden=%d\n",
                                   s, m.consistent.floor_high_limit, e_fhi);
            mism++;
        }
        if (m.consistent.counter != e_counter) {
            if (mism < 20) fprintf(stderr, "step %d cd_counter c=%lld golden=%lld\n",
                                   s, (long long)m.consistent.counter,
                                   (long long)e_counter);
            mism++;
        }
        if (m.consistent.delay_ref != e_dref) {
            if (mism < 20) fprintf(stderr, "step %d cd_delay_ref c=%d golden=%d\n",
                                   s, m.consistent.delay_ref, e_dref);
            mism++;
        }
    }

    fclose(f);
    printf("filter_analyzer parity: %d steps, size %d, hop %d, mismatches=%d\n",
           n_steps, size, hop, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    free(taps); free(block); free(exp_h); free(h_storage);
    free(abs_scratch); free(render_sq_scratch);
    return 0;
}
