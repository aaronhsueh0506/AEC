/* parity_aec_e2e.c -- replay the END-TO-END golden from python/diag/gen_aec_e2e_golden.py through the FULL top-level C
 * aec_process (aec_create(balanced) -> per-hop aec_process) and assert that out[hop] matches Python within the float32
 * tolerance over the whole doubletalk case (~4186 hops, at the rate the golden's header records; the checker reads hop/sr
 * straight back out of the file).
 *
 * The C chain is float32 end to end and its FFT is the float32 NE10 backend, while the golden comes from the fp64 Python spec
 * (numpy np.fft). The two therefore differ by float32 precision, amplified through the recursive adaptive filter and the steep
 * suppression-gain gates; the non-FFT logic is not bit-exact to Python either. PASS = both the linear and the final output stay
 * under TOL_E2E(sr).
 *
 * Measured (NE10 backend, 4186-hop doubletalk case; SIMD=1 and SIMD=0 give the same maxima to the digits shown):
 *
 *   rate    level                 linear max   out max     tolerance
 *   16 kHz  mild/balanced/aggr.   1.6e-4       4.3e-3      TOL_E2E     2e-2
 *   16 kHz  balanced --no-res     1.3e-5       1.3e-5      TOL_E2E     2e-2
 *   48 kHz  mild/balanced         1.2e-2       6.2e-2      TOL_E2E_48K 1e-1
 *   48 kHz  aggressive            1.2e-2       8.9e-2      TOL_E2E_48K 1e-1
 *    8 kHz  balanced/aggressive   1.3e-2       1.3e-1/1.6e-1   FAILS TOL_E2E 2e-2
 *
 * The 48 kHz row is wider because the transform and the filter are 4x larger (FFT 1024), so the rate-invariant tuning constants
 * regularise less and float32 rounding is amplified more through the adaptive recursion. The delay estimate is stable and the
 * extra error is confined to bounded, self-correcting spikes with no NaN/Inf, so TOL_E2E_48K is a measured ceiling.
 *
 * The 8 kHz golden exceeds TOL_E2E from the first hop of the linear output. A portable (non-NE10) FFT backend gave the same
 * maxima, so this is not an FFT-backend effect; it is a standing C/Python difference of that grid, recorded here rather than
 * hidden by loosening the constant.
 *
 * Build (standalone, from c_impl/; the FFT wrapper + fast_math live in the shared audiolib archive):
 *   make -C ../../audiolib/common lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       $(find src -name '*.c') test/parity_aec_e2e.c \
 *       $(make -s -C ../../audiolib/common print-lib-path) -lm -o bin/p_e2e
 *   python3 ../python/diag/gen_aec_e2e_golden.py bin/aec_e2e_golden.bin balanced --sr 16000
 *   bin/p_e2e bin/aec_e2e_golden.bin [level]      # level: 1|0|2|3 or balanced|mild|aggressive|no_floor
 *   # 8k/48k: --sr 8000 / --sr 48000 on the generator; the checker reads sr back out of the golden header and applies the
 *   # matching tolerance, so no extra argv is needed.
 *   # Linear-filter-only case (no post-filter block on either side): pass --no-res to BOTH, or the golden and the replay
 *   # disagree about which branch produced it:
 *   python3 ../python/diag/gen_aec_e2e_golden.py bin/g.bin balanced --sr 16000 --no-res
 *   bin/p_e2e bin/g.bin balanced --no-res
 *   # live observer (default off on either side): the same "pass to BOTH" rule, and it composes with --no-res:
 *   python3 ../python/diag/gen_aec_e2e_golden.py bin/g.bin balanced --sr 16000 --live-observer
 *   bin/p_e2e bin/g.bin balanced --live-observer
 */
#include "aec.h"

/* Max allowed |C - Python| per sample, per rate class (see the table in the
 * file header for the measured maxima each ceiling sits above). */
#define TOL_E2E      2.0e-2
#define TOL_E2E_48K  1.0e-1

static double tol_e2e_for(int sr) {
    return (sr >= 44100) ? TOL_E2E_48K : TOL_E2E;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/aec_e2e_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int hdr[3];
    if (!rd(f, hdr, sizeof(int) * 3)) { fprintf(stderr, "hdr\n"); return 2; }
    int hop = hdr[0], sr = hdr[1], n_hops = hdr[2];

    AecLevel level = AEC_LEVEL_BALANCED;
    if (argc > 2) {
        if (!strcmp(argv[2], "0") || !strcmp(argv[2], "mild")) level = AEC_LEVEL_MILD;
        else if (!strcmp(argv[2], "2") || !strcmp(argv[2], "aggressive")) level = AEC_LEVEL_AGGRESSIVE;
        else if (!strcmp(argv[2], "3") || !strcmp(argv[2], "no_floor")) level = AEC_LEVEL_NO_FLOOR;
        else if (strcmp(argv[2], "1") != 0 && strcmp(argv[2], "balanced") != 0) {
            fprintf(stderr, "unknown level '%s' (0|1|2|3 or "
                            "mild|balanced|aggressive|no_floor)\n", argv[2]);
            return 2;
        }
    }
    AecConfig cfg;
    aec_config_from_level(&cfg, level, sr);
    /* Golden produced with --no-res: the post-filter block never runs, so the
     * quantities that steer the filter from outside it are what this case
     * checks. `out` is then the linear residual, i.e. equal to raw. */
    /* Scan every remaining arg rather than hard-coding argv[3]/argv[4] positions: a fixed-position
     * check silently drops a flag if the two are passed in the other order, e.g. `--live-observer
     * --no-res` (the generator's argparse is order-independent). */
    for (int ai = 3; ai < argc; ++ai) {
        if (!strcmp(argv[ai], "--no-res")) cfg.enable_res = 0;
        else if (!strcmp(argv[ai], "--live-observer")) cfg.live_observer_enabled = 1;
    }
    Aec aec;
    if (aec_create(&aec, &cfg) != 0) { fprintf(stderr, "aec_create\n"); return 2; }
    if (aec_hop_size(&aec) != hop) {
        fprintf(stderr, "hop mismatch C=%d golden=%d\n", aec_hop_size(&aec), hop);
        return 2;
    }

    float *mic = malloc((size_t)hop * sizeof(float));
    float *ref = malloc((size_t)hop * sizeof(float));
    float *exp = malloc((size_t)hop * sizeof(float));
    float *exp_raw = malloc((size_t)hop * sizeof(float));
    float *out = malloc((size_t)hop * sizeof(float));

    long mism = 0, raw_mism = 0;
    double maxd = 0.0, raw_maxd = 0.0;
    int first_hop = -1, first_samp = -1, raw_first_hop = -1, raw_first_samp = -1;

    for (int hi = 0; hi < n_hops; ++hi) {
        if (!rd(f, mic, (size_t)hop * 4) ||
            !rd(f, ref, (size_t)hop * 4) ||
            !rd(f, exp, (size_t)hop * 4) ||
            !rd(f, exp_raw, (size_t)hop * 4)) { fprintf(stderr, "row %d\n", hi); return 2; }
        aec_process(&aec, mic, ref, out);
        const float *raw = aec.raw_output;   /* linear residual (struct exposed) */
        for (int k = 0; k < hop; ++k) {
            if (out[k] != exp[k]) {
                double d = (double)out[k] - (double)exp[k]; if (d < 0) d = -d;
                if (d > maxd) maxd = d;
                if (first_hop < 0) { first_hop = hi; first_samp = k; }
                mism++;
            }
            if (raw[k] != exp_raw[k]) {
                double d = (double)raw[k] - (double)exp_raw[k]; if (d < 0) d = -d;
                if (d > raw_maxd) raw_maxd = d;
                if (raw_first_hop < 0) { raw_first_hop = hi; raw_first_samp = k; }
                raw_mism++;
            }
        }
    }
    fclose(f);

    double tol = tol_e2e_for(sr);

    printf("aec_process E2E parity: %d hops hop=%d sr=%d (tol=%.1e)\n", n_hops, hop, sr, tol);
    printf("  raw_output(linear) mism=%ld (max=%.3e first_hop=%d first_samp=%d)\n",
           raw_mism, raw_maxd, raw_first_hop, raw_first_samp);
    printf("  >>> out[hop] mism=%ld (max=%.3e first_hop=%d first_samp=%d)\n",
           mism, maxd, first_hop, first_samp);

    aec_destroy(&aec);
    free(mic); free(ref); free(exp); free(exp_raw); free(out);

    if (raw_maxd < tol && maxd < tol) {
        printf(">>> PASS (within float32 FFT tolerance %.1e; linear max=%.3e out max=%.3e)\n",
               tol, raw_maxd, maxd);
        return 0;
    }
    printf(">>> FAIL (exceeds float32 FFT tolerance %.1e)\n", tol);
    return 1;
}
