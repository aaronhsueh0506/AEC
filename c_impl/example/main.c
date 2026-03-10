/**
 * main.c - AEC example program
 *
 * Usage: aec_wav <mic.wav> <ref.wav> <output.wav>
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

// Mode name helper
static const char* mode_name(AecFilterMode mode) {
    switch (mode) {
        case AEC_MODE_TIME:    return "time";
        case AEC_MODE_FREQ:    return "freq";
        case AEC_MODE_SUBBAND: return "subband";
        case AEC_MODE_LMS:     return "lms";
        default:               return "unknown";
    }
}

// Print usage
static void print_usage(const char* program) {
    printf("AEC (Acoustic Echo Cancellation)\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav>\n\n", program);
    printf("Arguments:\n");
    printf("  mic.wav    - Microphone input (16-bit mono WAV)\n");
    printf("  ref.wav    - Reference/loudspeaker signal (16-bit mono WAV)\n");
    printf("  output.wav - Echo-cancelled output\n\n");
    printf("Options:\n");
    printf("  --mode <time|freq|subband|lms> - Filter mode (default: time)\n");
    printf("  --mu <value>                - Step size (default: 0.3)\n");
    printf("  --filter <samples>          - Filter length in samples (default: frame_size)\n");
    printf("  --no-dtd                    - Disable double-talk detection\n");
    printf("  --clear-history             - Clear TIME/LMS buffer each block (no carry-over)\n");
    printf("\nFilter modes:\n");
    printf("  time    - Time-domain NLMS (configurable filter length, default=frame_size)\n");
    printf("  freq    - Frequency-domain NLMS (filter=hop_size, overlap-save)\n");
    printf("  subband - Partitioned FDAF (P hops history, default filter=hop*4)\n");
    printf("  lms     - Time-domain LMS (configurable filter length, default=frame_size)\n");
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
    int filter_length = 0;  // 0 = use mode-dependent default
    bool user_set_filter = false;
    bool enable_dtd = true;
    bool clear_history = false;
    AecFilterMode mode = AEC_MODE_TIME;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "time") == 0) {
                mode = AEC_MODE_TIME;
            } else if (strcmp(argv[i], "freq") == 0) {
                mode = AEC_MODE_FREQ;
            } else if (strcmp(argv[i], "subband") == 0) {
                mode = AEC_MODE_SUBBAND;
            } else if (strcmp(argv[i], "lms") == 0) {
                mode = AEC_MODE_LMS;
            } else {
                fprintf(stderr, "Unknown mode: %s (use 'time', 'freq', 'subband', or 'lms')\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--mu") == 0 && i + 1 < argc) {
            mu = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter_length = atoi(argv[++i]);
            user_set_filter = true;
        } else if (strcmp(argv[i], "--no-dtd") == 0) {
            enable_dtd = false;
        } else if (strcmp(argv[i], "--clear-history") == 0) {
            clear_history = true;
        }
    }

    // Open microphone WAV
    WavReader* mic_reader = wav_open_read(mic_path);
    if (!mic_reader) {
        fprintf(stderr, "Error: Failed to open %s\n", mic_path);
        return 1;
    }

    // Open reference WAV
    WavReader* ref_reader = wav_open_read(ref_path);
    if (!ref_reader) {
        fprintf(stderr, "Error: Failed to open %s\n", ref_path);
        wav_close_read(mic_reader);
        return 1;
    }

    // Verify compatibility
    if (mic_reader->info.sample_rate != ref_reader->info.sample_rate) {
        fprintf(stderr, "Error: Sample rate mismatch (mic=%d, ref=%d)\n",
                mic_reader->info.sample_rate, ref_reader->info.sample_rate);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int sample_rate = mic_reader->info.sample_rate;
    int mic_samples = mic_reader->info.num_samples;
    int ref_samples = ref_reader->info.num_samples;
    int num_samples = (mic_samples < ref_samples) ? mic_samples : ref_samples;

    // LMS mode: auto-set small mu if user didn't override
    if (mode == AEC_MODE_LMS && mu == 0.3f) {
        mu = 0.01f;
    }

    // Create AEC config
    AecConfig config = aec_default_config(sample_rate);
    config.filter_mode = mode;
    config.mu = mu;
    config.enable_dtd = enable_dtd;

    // Set mode-dependent filter_length default if not specified
    if (!user_set_filter) {
        switch (mode) {
            case AEC_MODE_SUBBAND: filter_length = config.hop_size * 4; break;  // 1024 @ 16kHz
            default:               filter_length = config.frame_size;   break;  // 512 @ 16kHz
        }
    }
    config.filter_length = filter_length;
    config.clear_filter_history = clear_history;

    // Compute derived params to show accurate info
    AecDerivedParams params = aec_compute_params(&config);

    printf("AEC Processing:\n");
    printf("  Microphone: %s (%d samples)\n", mic_path, mic_samples);
    printf("  Reference:  %s (%d samples)\n", ref_path, ref_samples);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate);
    printf("  Filter mode: %s\n", mode_name(mode));
    printf("  Hop size: %d samples (%.1f ms)\n", params.hop_size,
           1000.0f * params.hop_size / sample_rate);
    printf("  Step size (mu): %.3f\n", mu);
    printf("  Filter length: %d samples (%.1f ms)\n",
           params.filter_length, 1000.0f * params.filter_length / sample_rate);
    if (mode == AEC_MODE_FREQ || mode == AEC_MODE_SUBBAND) {
        printf("  Partitions: %d\n", params.n_partitions);
    }
    printf("  DTD: %s\n", enable_dtd ? "enabled" : "disabled");
    printf("\n");

    Aec* aec = aec_create(&config);
    if (!aec) {
        fprintf(stderr, "Error: Failed to create AEC\n");
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    int hop_size = aec_get_hop_size(aec);

    // Open output WAV
    WavWriter* writer = wav_open_write(out_path, sample_rate, 1);
    if (!writer) {
        fprintf(stderr, "Error: Failed to create %s\n", out_path);
        aec_destroy(aec);
        wav_close_read(mic_reader);
        wav_close_read(ref_reader);
        return 1;
    }

    // Allocate processing buffers
    float* mic_buf = (float*)malloc(hop_size * sizeof(float));
    float* ref_buf = (float*)malloc(hop_size * sizeof(float));
    float* out_buf = (float*)malloc(hop_size * sizeof(float));

    if (!mic_buf || !ref_buf || !out_buf) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(mic_buf);
        free(ref_buf);
        free(out_buf);
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
        // Read input blocks
        int mic_read = wav_read_float(mic_reader, mic_buf, hop_size);
        int ref_read = wav_read_float(ref_reader, ref_buf, hop_size);

        if (mic_read < hop_size || ref_read < hop_size) {
            // Zero-pad if needed
            for (int i = mic_read; i < hop_size; i++) mic_buf[i] = 0.0f;
            for (int i = ref_read; i < hop_size; i++) ref_buf[i] = 0.0f;
        }

        // Process through AEC
        aec_process(aec, mic_buf, ref_buf, out_buf);

        // Write output
        wav_write_float(writer, out_buf, hop_size);

        if (aec_is_dtd_active(aec)) {
            dtd_frames++;
        }

        float erle = aec_get_erle(aec);
        if (erle > max_erle) {
            max_erle = erle;
        }

        processed += hop_size;

        // Progress
        if (processed % (sample_rate) == 0) {
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

    // Cleanup
    free(mic_buf);
    free(ref_buf);
    free(out_buf);
    aec_destroy(aec);
    wav_close_read(mic_reader);
    wav_close_read(ref_reader);
    wav_close_write(writer);

    return 0;
}
