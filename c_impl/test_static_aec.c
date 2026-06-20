/* test_static_aec.c — verify aec_init() (static) produces byte-equal output
 * to aec_create() (dynamic) on the same input.
 *
 * Build (standalone, KISS FFT backend; fft_wrapper_ne10.c is NE10-only):
 *   gcc -O2 -ffp-contract=off -std=c99 -Iinclude -Iexample -Ilib/kiss_fft \
 *       test_static_aec.c $(find src -name '*.c' ! -name 'fft_wrapper_ne10.c') \
 *       lib/kiss_fft/kiss_fft.c -lm -o bin/test_static_aec
 * Run:
 *   ./bin/test_static_aec mic.wav ref.wav
 */
#include "aec.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Read entire WAV into a float array.  Returns sample count; *out must be freed. */
static int read_wav_f32(const char* path, float** out) {
    WavReader* r = wav_open_read(path);
    if (!r) return -1;
    int total = 0, cap = 48000 * 60;
    *out = (float*)malloc((size_t)cap * sizeof(float));
    int got;
    while ((got = wav_read_float(r, *out + total, 1024)) > 0) {
        total += got;
        if (total + 1024 > cap) { cap *= 2; *out = (float*)realloc(*out, (size_t)cap * sizeof(float)); }
    }
    wav_close_read(r);
    return total;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <mic.wav> <ref.wav>\n", argv[0]); return 1; }

    float *mic = NULL, *ref = NULL;
    int n_mic = read_wav_f32(argv[1], &mic);
    int n_ref = read_wav_f32(argv[2], &ref);
    if (n_mic <= 0 || n_ref <= 0) { fprintf(stderr, "Error reading WAV\n"); return 1; }
    int n = n_mic < n_ref ? n_mic : n_ref;

    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);

    /* Dynamic path */
    Aec aec_dyn;
    aec_create(&aec_dyn, &cfg);
    int hop = aec_hop_size(&aec_dyn);
    int frames = n / hop;
    float* out_dyn = (float*)calloc((size_t)frames * hop, sizeof(float));
    for (int i = 0; i < frames; ++i)
        aec_process(&aec_dyn, mic + i*hop, ref + i*hop, out_dyn + i*hop);
    aec_destroy(&aec_dyn);

    /* Static path */
    size_t pool_sz = aec_get_mem_size(&cfg);
    void* pool = malloc(pool_sz);
    if (!pool) { fprintf(stderr, "pool alloc failed\n"); return 1; }
    Aec* aec_st = aec_init(pool, pool_sz, &cfg);
    if (!aec_st) { fprintf(stderr, "aec_init returned NULL\n"); return 1; }
    float* out_st = (float*)calloc((size_t)frames * hop, sizeof(float));
    for (int i = 0; i < frames; ++i)
        aec_process(aec_st, mic + i*hop, ref + i*hop, out_st + i*hop);

    /* Compare */
    int diffs = 0;
    for (int i = 0; i < frames * hop; ++i) {
        if (out_dyn[i] != out_st[i]) {
            if (diffs < 5)
                fprintf(stderr, "DIFF[%d]: dyn=%f st=%f\n", i, out_dyn[i], out_st[i]);
            diffs++;
        }
    }
    printf("Pool: %zu bytes (%.1f KB), frames: %d\n", pool_sz, pool_sz/1024.0, frames);
    if (diffs == 0)
        printf("PASS: all %d samples byte-equal (static == dynamic)\n", frames * hop);
    else
        printf("FAIL: %d/%d samples differ\n", diffs, frames * hop);

    free(out_dyn); free(out_st); free(pool); free(mic); free(ref);
    return diffs ? 1 : 0;
}
