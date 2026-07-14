/*
 * test_wav_io_shim.c - F06 remediation smoke test for THIS repo's
 * example/wav_io.h shim onto the shared audio_common/include/wav_io.h.
 *
 * The exhaustive negative-corpus coverage (malformed headers, odd-chunk
 * padding, bounds checks, float32 NaN/Inf sanitize, etc.) lives once in
 * audio_common/test/test_wav_io.c and is not duplicated here. This test
 * only checks the two things specific to THIS repo's shim:
 *   1. #include "wav_io.h" from this repo's include paths actually reaches
 *      the canonical audio_common header (not a stale/duplicate copy).
 *   2. WAV_IO_WRITER_STYLE resolves to WAV_IO_WRITER_AEC here, and both of
 *      AEC's historical writer paths (PCM16 round-half-away by default,
 *      IEEE float32 via the AEC_OUT_FLOAT=1 env var) still work end to end
 *      through the shim.
 *
 * Standalone by design (like test/simd_selftest_aec.c): does not link
 * against libaec.a or any src object, so it builds and runs
 * independently of the rest of this repo's C sources. Not wired into the
 * Makefile (kept out of scope for this change -- see the F06 task notes);
 * build directly, e.g.:
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -I./include -I./example -I../../audio_common/include \
 *       -o /tmp/test_wav_io_shim test/test_wav_io_shim.c -lm
 *   /tmp/test_wav_io_shim
 */
#include "wav_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_fail = 1; \
    } \
} while (0)

static void test_writer_style_is_aec(void) {
    CHECK(WAV_IO_WRITER_STYLE == WAV_IO_WRITER_AEC,
          "this repo's shim must select WAV_IO_WRITER_AEC");
}

static void test_pcm16_roundtrip(void) {
    unsetenv("AEC_OUT_FLOAT");
    const float in_samples[4] = { 0.0f, 0.5f, -0.5f, -1.0f };
    char path[] = "/tmp/aec_wav_io_shim_pcm16_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp for PCM16 round-trip");
    if (fd >= 0) close(fd);

    WavWriter* w = wav_open_write(path, 16000, 1);
    CHECK(w != NULL, "wav_open_write must succeed (PCM16 default)");
    if (w) {
        CHECK(w->info.is_float == 0, "AEC_OUT_FLOAT unset -> PCM16 output");
        CHECK(w->info.bits_per_sample == 16, "PCM16 -> bits_per_sample == 16");
        wav_write_float(w, in_samples, 4);
        wav_close_write(w);

        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must read back the PCM16 file");
        if (r) {
            CHECK(r->info.is_float == 0, "read-back PCM16 file must report is_float==0");
            CHECK(r->info.bits_per_sample == 16, "read-back PCM16 file must report bits_per_sample==16");
            CHECK(r->info.num_samples == 4, "read-back PCM16 file must report num_samples==4");
            float out[4];
            int n = wav_read_float(r, out, 4);
            CHECK(n == 4, "must read back all 4 PCM16 samples");
            wav_close_read(r);
        }
    }
    unlink(path);
}

static void test_float32_roundtrip_via_aec_out_float(void) {
    setenv("AEC_OUT_FLOAT", "1", 1);
    const float in_samples[3] = { 0.125f, -0.25f, 3.0f /* AEC's float path writes unquantized, no clamp */ };
    char path[] = "/tmp/aec_wav_io_shim_f32_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp for float32 round-trip");
    if (fd >= 0) close(fd);

    WavWriter* w = wav_open_write(path, 48000, 1);
    CHECK(w != NULL, "wav_open_write must succeed (AEC_OUT_FLOAT=1)");
    if (w) {
        CHECK(w->info.is_float == 1, "AEC_OUT_FLOAT=1 -> float32 output");
        CHECK(w->info.bits_per_sample == 32, "float32 output -> bits_per_sample == 32");
        wav_write_float(w, in_samples, 3);
        wav_close_write(w);

        WavReader* r = wav_open_read(path);
        CHECK(r != NULL, "wav_open_read must read back the float32 file");
        if (r) {
            CHECK(r->info.is_float == 1, "read-back float32 file must report is_float==1");
            CHECK(r->info.num_samples == 3, "read-back float32 file must report num_samples==3");
            float out[3];
            int n = wav_read_float(r, out, 3);
            CHECK(n == 3, "must read back all 3 float32 samples");
            /* AEC's float32 writer path is unquantized (raw fwrite), so
             * this must be exactly byte-equal, unlike the PCM16 path. */
            CHECK(out[0] == in_samples[0], "float32 round-trip sample 0 must be exact (no quantization)");
            CHECK(out[1] == in_samples[1], "float32 round-trip sample 1 must be exact (no quantization)");
            CHECK(out[2] == in_samples[2], "float32 round-trip sample 2 must be exact (no quantization, no clamp)");
            CHECK(r->nonfinite_sanitized == 0, "no NaN/Inf in this file -> nonfinite_sanitized must stay 0");
            wav_close_read(r);
        }
    }
    unlink(path);
    unsetenv("AEC_OUT_FLOAT");
}

int main(void) {
    test_writer_style_is_aec();
    test_pcm16_roundtrip();
    test_float32_roundtrip_via_aec_out_float();

    if (g_fail) {
        printf(">>> FAIL\n");
        return 1;
    }
    printf("ALL PASS\n>>> PASS\n");
    return 0;
}
