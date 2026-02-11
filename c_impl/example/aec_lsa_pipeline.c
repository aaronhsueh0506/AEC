/**
 * aec_lsa_pipeline.c - AEC + LSA Integration Example
 *
 * Demonstrates the complete speech enhancement pipeline:
 *   Microphone -> AEC -> LSA (NR) -> Output
 *
 * Compile:
 *   # First build both AEC and LSA libraries
 *   cd AEC/c_impl && make lib
 *   cd LSA/c_impl && make lib
 *
 *   # Then link together
 *   gcc -o aec_lsa_pipeline aec_lsa_pipeline.c \
 *       -I../include -I../../LSA/c_impl/include \
 *       -L../bin -laec -L../../LSA/c_impl/bin -lmmse_lsa -lm
 *
 * Usage:
 *   ./aec_lsa_pipeline <mic.wav> <ref.wav> <output.wav>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// AEC includes
#include "aec.h"
#include "wav_io.h"

// Note: For full integration, you would also include:
// #include "mmse_lsa_denoiser.h"
// #include "mmse_lsa_types.h"

// Stub for LSA integration (replace with actual LSA when linking)
typedef struct MmseLsaDenoiser MmseLsaDenoiser;
typedef struct {
    int sample_rate;
    float g_min_db;
} MmseLsaConfig;

// These would be the actual LSA functions
// MmseLsaDenoiser* mmse_lsa_create(const MmseLsaConfig* config);
// void mmse_lsa_destroy(MmseLsaDenoiser* denoiser);
// int mmse_lsa_process(MmseLsaDenoiser* denoiser, const float* input, float* output);
// int mmse_lsa_get_hop_size(const MmseLsaDenoiser* denoiser);

static void print_usage(const char* program) {
    printf("AEC + LSA Pipeline\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav> [options]\n\n", program);
    printf("Options:\n");
    printf("  --aec-mu <value>     AEC step size (default: 0.3)\n");
    printf("  --lsa-gain <dB>      LSA minimum gain (default: -12.5)\n");
    printf("  --aec-only           Run AEC only, skip LSA\n");
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    // Parse options
    float aec_mu = 0.3f;
    float lsa_g_min = -12.5f;
    int aec_only = 0;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--aec-mu") == 0 && i + 1 < argc) {
            aec_mu = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lsa-gain") == 0 && i + 1 < argc) {
            lsa_g_min = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--aec-only") == 0) {
            aec_only = 1;
        }
    }

    // Open WAV files
    WavReader* mic_reader = wav_open_read(mic_path);
    WavReader* ref_reader = wav_open_read(ref_path);

    if (!mic_reader || !ref_reader) {
        fprintf(stderr, "Error: Failed to open input files\n");
        return 1;
    }

    if (mic_reader->info.sample_rate != ref_reader->info.sample_rate) {
        fprintf(stderr, "Error: Sample rate mismatch\n");
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int sample_rate = mic_reader->info.sample_rate;
    int num_samples = (mic_reader->info.num_samples < ref_reader->info.num_samples)
                      ? mic_reader->info.num_samples : ref_reader->info.num_samples;

    printf("AEC + LSA Pipeline\n");
    printf("==================\n");
    printf("Input: %s, %s\n", mic_path, ref_path);
    printf("Output: %s\n", out_path);
    printf("Sample rate: %d Hz\n", sample_rate);
    printf("Duration: %.2f seconds\n", (float)num_samples / sample_rate);
    printf("\nAEC Settings:\n");
    printf("  Step size (mu): %.3f\n", aec_mu);
    printf("\nLSA Settings:\n");
    printf("  Min gain: %.1f dB\n", lsa_g_min);
    printf("  Mode: %s\n", aec_only ? "AEC only" : "AEC + LSA");
    printf("\n");

    // Create AEC
    AecConfig aec_cfg = aec_default_config(sample_rate);
    aec_cfg.mu = aec_mu;
    aec_cfg.enable_dtd = true;

    Aec* aec = aec_create(&aec_cfg);
    if (!aec) {
        fprintf(stderr, "Error: Failed to create AEC\n");
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    // Create LSA (commented out - replace with actual LSA integration)
    // MmseLsaConfig lsa_cfg = mmse_lsa_default_config(sample_rate);
    // lsa_cfg.g_min_db = lsa_g_min;
    // MmseLsaDenoiser* lsa = mmse_lsa_create(&lsa_cfg);

    int hop_size = aec_get_hop_size(aec);

    // Open output
    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: Failed to create output file\n");
        aec_destroy(aec);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    // Allocate buffers
    float* mic_buf = (float*)malloc(hop_size * sizeof(float));
    float* ref_buf = (float*)malloc(hop_size * sizeof(float));
    float* aec_out = (float*)malloc(hop_size * sizeof(float));
    float* final_out = (float*)malloc(hop_size * sizeof(float));

    // Process
    int processed = 0;
    float max_erle = 0.0f;

    printf("Processing");
    fflush(stdout);

    while (processed + hop_size <= num_samples) {
        // Read inputs
        wav_read_float(mic_reader, mic_buf, hop_size);
        wav_read_float(ref_reader, ref_buf, hop_size);

        // Stage 1: AEC
        aec_process(aec, mic_buf, ref_buf, aec_out);

        // Stage 2: LSA (NR)
        if (!aec_only) {
            // mmse_lsa_process(lsa, aec_out, final_out);
            // For now, just copy (replace with actual LSA)
            memcpy(final_out, aec_out, hop_size * sizeof(float));
        } else {
            memcpy(final_out, aec_out, hop_size * sizeof(float));
        }

        // Write output
        wav_write_float(writer, final_out, hop_size);

        float erle = aec_get_erle(aec);
        if (erle > max_erle) max_erle = erle;

        processed += hop_size;

        if (processed % sample_rate == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    printf(" Done!\n\n");
    printf("Results:\n");
    printf("  Processed: %d samples (%.2f s)\n", processed, (float)processed / sample_rate);
    printf("  Max ERLE: %.1f dB\n", max_erle);

    // Cleanup
    free(mic_buf);
    free(ref_buf);
    free(aec_out);
    free(final_out);
    aec_destroy(aec);
    // mmse_lsa_destroy(lsa);
    wav_close_read(mic_reader);
    wav_close_read(ref_reader);
    wav_close_write(writer);

    printf("\nOutput written to: %s\n", out_path);
    return 0;
}
