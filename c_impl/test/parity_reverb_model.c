/* parity_reverb_model.c — replay the binary golden from
 * python/diag/gen_reverb_golden.py through the C ReverbModel and assert exact
 * (bit-for-bit float32) match. WS5 Phase 5.1 gate.
 *
 * Build (from c_impl/):
 *   gcc -O2 -ffp-contract=off -std=c99 -Iinclude \
 *       src/reverb_model.c test/parity_reverb_model.c -lm -o /tmp/p_reverb
 *   python3 ../python/diag/gen_reverb_golden.py /tmp/reverb_golden.bin
 *   /tmp/p_reverb /tmp/reverb_golden.bin
 */
#include "reverb_model.h"

#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/reverb_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_bins, n_nfs, n_u, i, k, mism = 0, calls = 0;
    float *ps, *scaling, *expected, *rev;
    ReverbModel m;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_bins, 4) || !rd(f, &n_nfs, 4) || !rd(f, &n_u, 4)) return 2;

    ps = malloc(n_bins * sizeof(float));
    scaling = malloc(n_bins * sizeof(float));
    expected = malloc(n_bins * sizeof(float));
    rev = malloc(n_bins * sizeof(float));

    /* update_no_freq_shaping sequence */
    reverb_model_init(&m, rev, n_bins);
    for (i = 0; i < n_nfs; ++i) {
        float sc, decay;
        if (!rd(f, ps, n_bins * 4) || !rd(f, &sc, 4) || !rd(f, &decay, 4) ||
            !rd(f, expected, n_bins * 4)) { fprintf(stderr, "short read nfs %d\n", i); return 2; }
        reverb_model_update_no_freq_shaping(&m, ps, sc, decay);
        for (k = 0; k < n_bins; ++k) if (m.reverb[k] != expected[k]) mism++;
        calls++;
    }
    /* update sequence (per-bin scaling) */
    reverb_model_init(&m, rev, n_bins);
    for (i = 0; i < n_u; ++i) {
        float decay;
        if (!rd(f, ps, n_bins * 4) || !rd(f, scaling, n_bins * 4) || !rd(f, &decay, 4) ||
            !rd(f, expected, n_bins * 4)) { fprintf(stderr, "short read u %d\n", i); return 2; }
        reverb_model_update(&m, ps, scaling, decay);
        for (k = 0; k < n_bins; ++k) if (m.reverb[k] != expected[k]) mism++;
        calls++;
    }
    fclose(f);
    printf("reverb_model parity: %d calls, %d bins, mismatches=%d\n", calls, n_bins, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
