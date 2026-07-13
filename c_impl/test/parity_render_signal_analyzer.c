/* parity_render_signal_analyzer.c — replay the binary golden from
 * python/diag/gen_render_signal_analyzer_golden.py through the C
 * RenderSignalAnalyzer and assert exact match on counters / narrow_peak_band /
 * poor_signal_excitation / masked-mu. WS5 Phase 5.1 gate.
 *
 * Build (standalone, from repo root):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include c_impl/src/render_signal_analyzer.c \
 *       c_impl/test/parity_render_signal_analyzer.c -lm -o /tmp/p_rsa
 *   python3 python/diag/gen_render_signal_analyzer_golden.py /tmp/rsa_golden.bin
 *   /tmp/p_rsa /tmp/rsa_golden.bin
 */
#include "render_signal_analyzer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/rsa_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_freqs, hop, n_steps, n_counters;
    int s, k, mism = 0;
    float *psd, *block, *exp_mu, *mu;
    int64_t *exp_counters, *counters_storage;
    RenderSignalAnalyzer m;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_freqs, 4) || !rd(f, &hop, 4) ||
        !rd(f, &n_steps, 4) || !rd(f, &n_counters, 4)) return 2;

    psd          = malloc((size_t)n_freqs * sizeof(float));
    block        = malloc((size_t)hop * sizeof(float));
    exp_mu       = malloc((size_t)n_freqs * sizeof(float));
    mu           = malloc((size_t)n_freqs * sizeof(float));
    exp_counters = malloc((size_t)n_counters * sizeof(int64_t));
    counters_storage = malloc((size_t)n_counters * sizeof(int64_t));

    rsa_init(&m, counters_storage, n_freqs, 6 /* strong_peak_freeze_duration */);
    if (m.n_counters != n_counters) {
        fprintf(stderr, "n_counters mismatch: c=%d golden=%d\n",
                m.n_counters, n_counters);
        return 2;
    }

    for (s = 0; s < n_steps; ++s) {
        int has_psd, has_block, exp_npb, exp_poor;
        const float *psd_ptr, *block_ptr;
        int block_len;

        if (!rd(f, &has_psd, 4) || !rd(f, &has_block, 4)) {
            fprintf(stderr, "short read header step %d\n", s); return 2;
        }
        if (has_psd) {
            if (!rd(f, psd, (size_t)n_freqs * sizeof(float))) {
                fprintf(stderr, "short read psd step %d\n", s); return 2;
            }
        }
        if (has_block) {
            if (!rd(f, block, (size_t)hop * sizeof(float))) {
                fprintf(stderr, "short read block step %d\n", s); return 2;
            }
        }
        if (!rd(f, exp_counters, (size_t)n_counters * sizeof(int64_t)) ||
            !rd(f, &exp_npb, 4) || !rd(f, &exp_poor, 4) ||
            !rd(f, exp_mu, (size_t)n_freqs * sizeof(float))) {
            fprintf(stderr, "short read expected step %d\n", s); return 2;
        }

        psd_ptr   = has_psd ? psd : NULL;
        block_ptr = has_block ? block : NULL;
        block_len = has_block ? hop : 0;

        rsa_update(&m, psd_ptr, block_ptr, block_len);

        /* compare counters */
        for (k = 0; k < n_counters; ++k) {
            if (m.counters[k] != exp_counters[k]) {
                if (mism < 20)
                    fprintf(stderr, "step %d counter[%d] c=%lld golden=%lld\n",
                            s, k, (long long)m.counters[k],
                            (long long)exp_counters[k]);
                mism++;
            }
        }
        /* narrow_peak_band */
        if (m.narrow_peak_band != exp_npb) {
            if (mism < 20)
                fprintf(stderr, "step %d narrow_peak_band c=%d golden=%d\n",
                        s, m.narrow_peak_band, exp_npb);
            mism++;
        }
        /* poor_signal_excitation */
        {
            int poor = rsa_poor_signal_excitation(&m);
            if (poor != exp_poor) {
                if (mism < 20)
                    fprintf(stderr, "step %d poor_exc c=%d golden=%d\n",
                            s, poor, exp_poor);
                mism++;
            }
        }
        /* masked-mu on a ones() vector */
        for (k = 0; k < n_freqs; ++k) mu[k] = 1.0f;
        rsa_mask_regions_around_narrow_bands(&m, mu);
        for (k = 0; k < n_freqs; ++k) {
            if (mu[k] != exp_mu[k]) {
                if (mism < 20)
                    fprintf(stderr, "step %d mu[%d] c=%g golden=%g\n",
                            s, k, mu[k], exp_mu[k]);
                mism++;
            }
        }
    }
    fclose(f);
    printf("render_signal_analyzer parity: %d steps, %d freqs, %d counters, "
           "mismatches=%d\n", n_steps, n_freqs, n_counters, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
