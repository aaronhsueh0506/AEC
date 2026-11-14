/* test_static_aec.c -- verify aec_init() (static) produces byte-equal output to aec_create() (dynamic) on the same input.
 *
 * Build (standalone, NE10 FFT backend (default); the FFT wrapper + fast_math live in the shared audiolib archive):
 *   make -C ../../audiolib/common BACKEND=ne10 lib
 *   gcc -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       test/test_static_aec.c $(find src -name '*.c') \
 *       $(make -s -C ../../audiolib/common BACKEND=ne10 print-lib-path) -lm -o bin/test_static_aec
 * The normal entry point is `make test-static-aec`.
 * Run:
 *   ./bin/test_static_aec mic.wav ref.wav [sample_rate]
 *
 * The optional 3rd argv selects the sample rate for building AecConfig (default 16000). The WAV files themselves must already
 * be at that rate (this test does not resample).
 */
#include "aec.h"
#include "aec_pool_test_util.h"
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
    if (argc < 3) { fprintf(stderr, "Usage: %s <mic.wav> <ref.wav> [sample_rate]\n", argv[0]); return 1; }
    int sr = argc > 3 ? atoi(argv[3]) : 16000;

    float *mic = NULL, *ref = NULL;
    int n_mic = read_wav_f32(argv[1], &mic);
    int n_ref = read_wav_f32(argv[2], &ref);
    if (n_mic <= 0 || n_ref <= 0) { fprintf(stderr, "Error reading WAV\n"); return 1; }
    int n = n_mic < n_ref ? n_mic : n_ref;

    AecConfig cfg;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);

    /* Dynamic path */
    Aec *aec_dyn = NULL;
    if ((aec_dyn = aec_create(&cfg)) == NULL) {
        fprintf(stderr, "aec_create failed\n"); free(mic); free(ref); return 1;
    }
    int hop = aec_hop_size(aec_dyn);
    int frames = n / hop;
    float* out_dyn = (float*)calloc((size_t)frames * hop, sizeof(float));
    for (int i = 0; i < frames; ++i)
        aec_process(aec_dyn, mic + i*hop, ref + i*hop, out_dyn + i*hop);
    aec_destroy(aec_dyn);

    /* Static path */
    size_t pool_sz = aec_get_mem_size(&cfg);
    void* pool = aec_test_pool_alloc(pool_sz);
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

    /* The FFT handle and its twiddles live in the pool, so aec_destroy() frees
     * nothing here; the caller frees the pool. */
    aec_destroy(aec_st);
    free(out_dyn); free(out_st); free(pool);

    /* Leak-loop: repeat init -> process -> destroy so a leak checker
     * (`leaks --atExit -- ./bin/test_static_aec ...`) can see whether
     * anything grows with iteration count. A leak shows up here as
     * N-times-repeated allocations; a correct teardown stays flat. */
    {
        const int kIters = 50;
        const int loop_frames = frames < 20 ? frames : 20; /* keep it fast */
        for (int iter = 0; iter < kIters; ++iter) {
            size_t sz = aec_get_mem_size(&cfg);
            void* p = aec_test_pool_alloc(sz);
            if (!p) { fprintf(stderr, "leak-loop: pool alloc failed at iter %d\n", iter); return 1; }
            Aec* inst = aec_init(p, sz, &cfg);
            if (!inst) { fprintf(stderr, "leak-loop: aec_init failed at iter %d\n", iter); free(p); return 1; }
            float scratch_out[/*hop*/ 4096];
            for (int i = 0; i < loop_frames; ++i) {
                if (hop > (int)(sizeof(scratch_out) / sizeof(scratch_out[0]))) break;
                aec_process(inst, mic + i*hop, ref + i*hop, scratch_out);
            }
            aec_destroy(inst);
            free(p);
        }
        printf("Leak-loop: %d init->process->destroy cycles completed\n", kIters);
    }

    free(mic); free(ref);
    return diffs ? 1 : 0;
}
