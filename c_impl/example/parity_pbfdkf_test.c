/**
 * parity_pbfdkf_test.c - Phase 1 parity verifier
 *
 * Reads python/parity_pbfdkf_input.bin (mic, ref, mu_schedule),
 * runs C PBFDKF with same config as Python parity_pbfdkf_gen.py,
 * dumps per-frame state to python/parity_pbfdkf_c.bin.
 *
 * Companion python/parity_pbfdkf_load_c.py converts the binary to
 * parity_pbfdkf_c.npz which parity_pbfdkf_check.py compares to
 * parity_pbfdkf_python.npz at rtol=1e-5 atol=1e-7.
 *
 * Usage:
 *   make parity_pbfdkf_test
 *   ./bin/parity_pbfdkf_test ../python/parity_pbfdkf_input.bin \
 *                            ../python/parity_pbfdkf_c.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pbfdkf.h"

/* Config — must match python/parity_pbfdkf_gen.py exactly */
#define HOP            160
#define BLOCK          (2 * HOP)         /* 320 */
#define N_FRAMES       200
#define SAMPLE_RATE    16000
#define FILTER_LEN     (SAMPLE_RATE * 52 / 1000)  /* 832 */
#define N_PARTITIONS   ((FILTER_LEN + HOP - 1) / HOP)  /* 6 */
#define DELTA          1e-8f
#define Q_HIGH         1e-4f
#define Q_LOW          1e-5f


static int read_input(const char* path, float* mic, float* ref, float* mu) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    size_t n = N_FRAMES * HOP;
    if (fread(mic, sizeof(float), n, fp) != n) goto err;
    if (fread(ref, sizeof(float), n, fp) != n) goto err;
    if (fread(mu,  sizeof(float), N_FRAMES, fp) != (size_t)N_FRAMES) goto err;
    fclose(fp);
    return 0;
err:
    fprintf(stderr, "input read short\n");
    fclose(fp);
    return -1;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <input.bin> <output.bin>\n"
                "Reads (mic, ref, mu_schedule) raw float32 from input.bin,\n"
                "runs C PBFDKF, dumps per-frame state to output.bin.\n",
                argv[0]);
        return 1;
    }

    /* Allocate input */
    size_t n_samples = N_FRAMES * HOP;
    float* mic = (float*)malloc(n_samples * sizeof(float));
    float* ref = (float*)malloc(n_samples * sizeof(float));
    float* mu  = (float*)malloc(N_FRAMES * sizeof(float));
    if (!mic || !ref || !mu) { fprintf(stderr, "OOM input\n"); return 2; }

    if (read_input(argv[1], mic, ref, mu) != 0) return 3;

    /* Create filter */
    Pbfdkf* f = pbfdkf_create(BLOCK, HOP, N_PARTITIONS, DELTA, Q_HIGH, Q_LOW);
    if (!f) { fprintf(stderr, "pbfdkf_create failed\n"); return 4; }

    int n_freqs = pbfdkf_get_n_freqs(f);
    int fft_size = pbfdkf_get_fft_size(f);
    printf("Config: hop=%d block=%d fft=%d n_freqs=%d partitions=%d frames=%d\n",
           HOP, BLOCK, fft_size, n_freqs, N_PARTITIONS, N_FRAMES);

    /* Open output binary */
    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 5; }

    /* Header: 7 int32 — must match Python parity_pbfdkf_load_c.py */
    int32_t hdr[7] = { HOP, BLOCK, fft_size, n_freqs, N_PARTITIONS, N_FRAMES, SAMPLE_RATE };
    fwrite(hdr, sizeof(int32_t), 7, out);

    /* Per-frame state buffers */
    float* output_buf = (float*)malloc(HOP * sizeof(float));
    if (!output_buf) { fprintf(stderr, "OOM output_buf\n"); return 6; }

    double mic_e = 0.0, out_e = 0.0;
    for (int frame = 0; frame < N_FRAMES; frame++) {
        const float* m = mic + frame * HOP;
        const float* r = ref + frame * HOP;

        if (pbfdkf_process(f, m, r, output_buf, mu[frame]) != 0) {
            fprintf(stderr, "pbfdkf_process frame %d failed\n", frame);
            return 7;
        }

        /* Dump output[hop] */
        fwrite(output_buf, sizeof(float), HOP, out);

        /* Dump error_spec[n_freqs] complex */
        const Complex* err = pbfdkf_get_error_spec(f);
        fwrite(err, sizeof(Complex), n_freqs, out);

        /* Dump echo_spec[n_freqs] complex */
        const Complex* echo = pbfdkf_get_echo_spec(f);
        fwrite(echo, sizeof(Complex), n_freqs, out);

        /* Dump P[partitions][n_freqs] */
        for (int p = 0; p < N_PARTITIONS; p++) {
            const float* P = pbfdkf_get_P(f, p);
            fwrite(P, sizeof(float), n_freqs, out);
        }

        /* Dump W[partitions][n_freqs] complex */
        for (int p = 0; p < N_PARTITIONS; p++) {
            const Complex* W = pbfdkf_get_W(f, p);
            fwrite(W, sizeof(Complex), n_freqs, out);
        }

        /* Dump power[n_freqs] */
        const float* pwr = pbfdkf_get_power(f);
        fwrite(pwr, sizeof(float), n_freqs, out);

        /* Energy diagnostics */
        for (int i = 0; i < HOP; i++) {
            mic_e += (double)m[i] * (double)m[i];
            out_e += (double)output_buf[i] * (double)output_buf[i];
        }
    }

    fclose(out);
    pbfdkf_destroy(f);
    free(mic); free(ref); free(mu); free(output_buf);

    printf("  mic energy : %.4e\n", mic_e / n_samples);
    printf("  output energy: %.4e\n", out_e / n_samples);
    printf("Wrote %s\n", argv[2]);
    return 0;
}
