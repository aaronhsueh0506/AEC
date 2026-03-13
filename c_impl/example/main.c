/**
 * main.c - AEC example program
 *
 * Usage: aec_wav <mic.wav> <ref.wav> <output.wav> [options]
 *
 * Arguments:
 *   mic.wav    - Microphone input (near-end speech + echo)
 *   ref.wav    - Reference signal (far-end/loudspeaker playback)
 *   output.wav - Echo-cancelled output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
#include "wav_io.h"

static void print_usage(const char* program) {
    printf("AEC (Acoustic Echo Cancellation) — PBFDAF\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav> [options]\n\n", program);
    printf("Arguments:\n");
    printf("  mic.wav    - Microphone input (16-bit mono WAV)\n");
    printf("  ref.wav    - Reference/loudspeaker signal (16-bit mono WAV)\n");
    printf("  output.wav - Echo-cancelled output\n\n");
    printf("Options:\n");
    printf("  --mu <value>       - Step size (default: 0.3)\n");
    printf("  --filter <samples> - Filter length in samples (default: 1024)\n");
    printf("  --no-dtd           - Disable double-talk detection\n");
    printf("  --enable-res       - Enable residual echo suppressor (RES)\n");
    printf("  --res-gmin <dB>    - RES minimum gain in dB (default: -20)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    // Parse optional arguments
    float mu = 0.3f;
    int filter_length = 0;  // 0 = use default
    bool enable_dtd = true;
    bool enable_res = true;
    float res_gmin = -20.0f;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--mu") == 0 && i + 1 < argc) {
            mu = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-dtd") == 0) {
            enable_dtd = false;
        } else if (strcmp(argv[i], "--enable-res") == 0) {
            enable_res = true;
        } else if (strcmp(argv[i], "--no-res") == 0) {
            enable_res = false;
        } else if (strcmp(argv[i], "--res-gmin") == 0 && i + 1 < argc) {
            res_gmin = (float)atof(argv[++i]);
        }
    }

    // Open WAV files
    WavReader* mic_reader = wav_open_read(mic_path);
    if (!mic_reader) {
        fprintf(stderr, "Error: Failed to open %s\n", mic_path);
        return 1;
    }

    WavReader* ref_reader = wav_open_read(ref_path);
    if (!ref_reader) {
        fprintf(stderr, "Error: Failed to open %s\n", ref_path);
        wav_close_read(mic_reader);
        return 1;
    }

    if (mic_reader->info.sample_rate != ref_reader->info.sample_rate) {
        fprintf(stderr, "Error: Sample rate mismatch (mic=%d, ref=%d)\n",
                mic_reader->info.sample_rate, ref_reader->info.sample_rate);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int sample_rate = mic_reader->info.sample_rate;
    int num_samples = mic_reader->info.num_samples < ref_reader->info.num_samples
                    ? mic_reader->info.num_samples : ref_reader->info.num_samples;

    // Create AEC config
    AecConfig config = aec_default_config(sample_rate);
    config.mu = mu;
    config.enable_dtd = enable_dtd;
    config.enable_res = enable_res;
    config.res_g_min_db = res_gmin;
    if (filter_length > 0) {
        config.filter_length = filter_length;
    }

    AecDerivedParams params = aec_compute_params(&config);

    printf("AEC Processing (PBFDAF):\n");
    printf("  Microphone: %s (%d samples)\n", mic_path, mic_reader->info.num_samples);
    printf("  Reference:  %s (%d samples)\n", ref_path, ref_reader->info.num_samples);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate);
    printf("  Hop size: %d samples (%.1f ms)\n", params.hop_size,
           1000.0f * params.hop_size / sample_rate);
    printf("  Step size (mu): %.3f\n", mu);
    printf("  Filter length: %d samples (%.1f ms)\n",
           params.filter_length, 1000.0f * params.filter_length / sample_rate);
    printf("  FFT block size: %d\n", params.block_size);
    printf("  Partitions: %d\n", params.n_partitions);
    printf("  DTD: %s (warmup: %d frames)\n",
           enable_dtd ? "enabled" : "disabled", config.dtd_warmup_frames);
    printf("  RES: %s", enable_res ? "enabled" : "disabled");
    if (enable_res) printf(" (g_min=%.0f dB)", res_gmin);
    printf("\n\n");

    Aec* aec = aec_create(&config);
    if (!aec) {
        fprintf(stderr, "Error: Failed to create AEC\n");
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int hop_size = aec_get_hop_size(aec);

    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: Failed to create %s\n", out_path);
        aec_destroy(aec);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    float* mic_buf = (float*)malloc(hop_size * sizeof(float));
    float* ref_buf = (float*)malloc(hop_size * sizeof(float));
    float* out_buf = (float*)malloc(hop_size * sizeof(float));

    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(mic_buf); free(ref_buf); free(out_buf);
        aec_destroy(aec);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        wav_close_write(writer);
        return 1;
    }

    // Process
    int processed = 0;
    int dtd_frames = 0;
    float max_erle = 0.0f;

    while (processed + hop_size <= num_samples) {
        int mic_read = wav_read_float(mic_reader, mic_buf, hop_size);
        int ref_read = wav_read_float(ref_reader, ref_buf, hop_size);

        if (mic_read < hop_size || ref_read < hop_size) {
            for (int i = mic_read; i < hop_size; i++) mic_buf[i] = 0.0f;
            for (int i = ref_read; i < hop_size; i++) ref_buf[i] = 0.0f;
        }

        aec_process(aec, mic_buf, ref_buf, out_buf);
        wav_write_float(writer, out_buf, hop_size);

        if (aec_is_dtd_active(aec)) dtd_frames++;

        float erle = aec_get_erle(aec);
        if (erle > max_erle) max_erle = erle;

        processed += hop_size;

        if (processed % sample_rate == 0) {
            printf("  Processed: %.1f s, ERLE: %.1f dB\r",
                   (float)processed / sample_rate, erle);
            fflush(stdout);
        }
    }

    printf("\n\nResults:\n");
    printf("  Processed samples: %d\n", processed);
    printf("  Max ERLE: %.1f dB\n", max_erle);
    printf("  DTD active frames: %d (%.1f%%)\n",
           dtd_frames, 100.0f * dtd_frames * hop_size / (processed > 0 ? processed : 1));
    printf("\nOutput written to: %s\n", out_path);

    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    aec_destroy(aec);
    wav_close_read(mic_reader);
    wav_close_read(ref_reader);
    wav_close_write(writer);

    return 0;
}
