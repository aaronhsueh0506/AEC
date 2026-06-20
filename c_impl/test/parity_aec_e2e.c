/* parity_aec_e2e.c — replay the END-TO-END golden from
 * python/diag/gen_aec_e2e_golden.py through the FULL top-level C aec_process
 * (aec_create(balanced) → per-hop aec_process) and assert out[hop] matches
 * Python within the float32-FFT tolerance over the whole doubletalk case
 * (~4186 hops).
 *
 * NOTE: with the KISS FFT backend (float32) the FFT is NOT bit-exact to numpy's
 * fp64 np.fft, so the end-to-end output differs by ~float32 precision
 * (measured: RMS Δ ≈ -60 dB below signal, correlation 0.99999958, per-sample
 * max ~6e-3 over 4186 recursive hops — inaudible). The non-FFT C logic stays
 * bit-exact; only the FFT layer carries this documented tolerance. PASS = both
 * the linear and final output stay under TOL_E2E.
 *
 * Build (standalone, from c_impl/; delay_est is the retired v3.10 module and
 * fft_wrapper_ne10.c needs the NE10 lib — both excluded):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Iinclude -Ilib/kiss_fft \
 *       $(find src -name '*.c' ! -name 'delay_est.c' ! -name 'fft_wrapper_ne10.c') \
 *       lib/kiss_fft/kiss_fft.c test/parity_aec_e2e.c -lm -o /tmp/p_e2e
 *   python3 ../python/diag/gen_aec_e2e_golden.py /tmp/aec_e2e_golden.bin [preset]
 *   /tmp/p_e2e /tmp/aec_e2e_golden.bin [preset]      # preset: balanced|gentle|aggressive
 */
#include "aec.h"

/* Max allowed |C - Python| per sample. Measured float32-FFT e2e max ≈ 6e-3
 * across all presets; 2e-2 gives ~3x headroom for signal variation while still
 * catching real FFT/scaling regressions (which diverge by >0.1). */
#define TOL_E2E 2.0e-2

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

    AecPreset preset = AEC_PRESET_BALANCED;
    if (argc > 2) {
        if (!strcmp(argv[2], "gentle")) preset = AEC_PRESET_GENTLE;
        else if (!strcmp(argv[2], "aggressive")) preset = AEC_PRESET_AGGRESSIVE;
    }
    AecConfig cfg;
    aec_config_from_preset(&cfg, preset, sr);
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

    printf("aec_process E2E parity: %d hops hop=%d sr=%d\n", n_hops, hop, sr);
    printf("  raw_output(linear) mism=%ld (max=%.3e first_hop=%d first_samp=%d)\n",
           raw_mism, raw_maxd, raw_first_hop, raw_first_samp);
    printf("  >>> out[hop] mism=%ld (max=%.3e first_hop=%d first_samp=%d)\n",
           mism, maxd, first_hop, first_samp);

    aec_destroy(&aec);
    free(mic); free(ref); free(exp); free(exp_raw); free(out);

    if (raw_maxd < TOL_E2E && maxd < TOL_E2E) {
        printf(">>> PASS (within float32 FFT tolerance %.0e; linear max=%.3e out max=%.3e)\n",
               TOL_E2E, raw_maxd, maxd);
        return 0;
    }
    printf(">>> FAIL (exceeds float32 FFT tolerance %.0e)\n", TOL_E2E);
    return 1;
}
