/**
 * main.c - AEC example program
 *
 * Usage: aec_wav <mic.wav> <ref.wav> <output.wav> [options]
 *
 * Options:
 *   --preset balanced|aggressive|maximum  (default: balanced)
 *   --no-res       Disable RES post-filter
 *   --no-hpf       Disable high-pass filter
 *   --filter <N>   Filter length in samples (default: 512)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
#include "wav_io.h"

static void print_usage(const char* program) {
    printf("AEC (Acoustic Echo Cancellation) — PBFDKF + Shadow + WOLA RES\n");
    printf("Usage: %s <mic.wav> <ref.wav> <output.wav> [options]\n\n", program);
    printf("Arguments:\n");
    printf("  mic.wav    - Microphone input (16-bit mono WAV)\n");
    printf("  ref.wav    - Reference/loudspeaker signal (16-bit mono WAV)\n");
    printf("  output.wav - Echo-cancelled output\n\n");
    printf("Options:\n");
    printf("  --preset <name>    - balanced (default), aggressive, maximum\n");
    printf("  --no-res           - Disable residual echo suppressor\n");
    printf("  --no-hpf           - Disable high-pass filter\n");
    printf("  --filter <samples> - Filter length in samples (default: 512)\n");
    printf("  --enable-delay-est - Enable GCC-PHAT delay estimation\n");
    printf("  --no-delay-est     - Disable delay estimation (overrides preset)\n");
    printf("  --max-delay <ms>   - Max delay to search in ms (default: 250)\n");
}

static AecPreset parse_preset(const char* name) {
    if (strcmp(name, "aggressive") == 0) return AEC_PRESET_AGGRESSIVE;
    if (strcmp(name, "maximum") == 0)    return AEC_PRESET_MAXIMUM;
    return AEC_PRESET_BALANCED;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    /* Parse options */
    AecPreset preset = AEC_PRESET_BALANCED;
    int enable_res = 1;
    int enable_hpf = 1;
    int filter_length = 0; /* 0 = default */
    int delay_est_override = -1; /* -1 = use preset default, 0 = force off, 1 = force on */
    float max_delay_ms = 0.0f;  /* 0 = use default */

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--preset") == 0 && i + 1 < argc) {
            preset = parse_preset(argv[++i]);
        } else if (strcmp(argv[i], "--no-res") == 0) {
            enable_res = 0;
        } else if (strcmp(argv[i], "--no-hpf") == 0) {
            enable_hpf = 0;
        } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            filter_length = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--enable-delay-est") == 0) {
            delay_est_override = 1;
        } else if (strcmp(argv[i], "--no-delay-est") == 0) {
            delay_est_override = 0;
        } else if (strcmp(argv[i], "--max-delay") == 0 && i + 1 < argc) {
            max_delay_ms = (float)atof(argv[++i]);
        }
    }

    /* Open WAV files */
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

    /* Create config from preset */
    AecConfig config = aec_config_from_preset(preset, sample_rate);
    config.enable_res = enable_res;
    config.enable_highpass = enable_hpf;
    if (filter_length > 0) {
        config.filter_length = filter_length;
        config.n_partitions = (filter_length + config.hop_size - 1) / config.hop_size;
    }
    if (delay_est_override >= 0)
        config.enable_delay_est = delay_est_override;
    if (max_delay_ms > 0.0f)
        config.delay_est_max_ms = max_delay_ms;

    printf("AEC Processing (PBFDKF + Shadow + WOLA RES):\n");
    printf("  Preset: %s\n", aec_preset_name(preset));
    printf("  Microphone: %s (%d samples)\n", mic_path, mic_reader->info.num_samples);
    printf("  Reference:  %s (%d samples)\n", ref_path, ref_reader->info.num_samples);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate);
    printf("  Frame/Hop/FFT: %d/%d/%d (%.0f/%.0f ms)\n",
           config.frame_size, config.hop_size, config.fft_size,
           1000.0f * config.frame_size / sample_rate,
           1000.0f * config.hop_size / sample_rate);
    printf("  Filter: %d samples (%d partitions)\n",
           config.filter_length, config.n_partitions);
    printf("  Kalman: Q_high=%.1e, Q_low=%.1e\n",
           config.kalman_q_high, config.kalman_q_low);
    printf("  RES: %s (g_min=%.0f dB, base_over_sub=%.1f)\n",
           enable_res ? "enabled" : "disabled",
           config.res_g_min_db, config.res_over_sub_base);
    printf("  RES v2: echo=%s, gain=%s, enr_scale=%.2f\n",
           res_echo_method_name(config.res_echo_method),
           res_gain_type_name(config.res_gain_type),
           config.res_enr_scale);
    printf("  Reverb: %s (decay=%.1f, gain=%.1f)\n",
           config.res_enable_reverb ? "enabled" : "disabled",
           config.res_reverb_decay, config.res_reverb_gain);
    printf("  HPF: %s (%.0f Hz)\n",
           enable_hpf ? "enabled" : "disabled",
           config.highpass_cutoff_hz);
    printf("  Shadow: enabled (Q_ratio=%.1f)\n",
           config.shadow_q_ratio);
    printf("  Delay est: %s", config.enable_delay_est ? "enabled" : "disabled");
    if (config.enable_delay_est)
        printf(" (max=%.0f ms, init=%.1f s, period=%.1f s)",
               config.delay_est_max_ms, config.delay_est_init_s, config.delay_est_period_s);
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

    /* Process */
    int processed = 0;
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

        float erle = aec_get_erle(aec);
        if (erle > max_erle) max_erle = erle;

        processed += hop_size;

        if (processed % (sample_rate / 10 * 10) < hop_size) {
            int delay = aec_get_delay(aec);
            if (delay >= 0)
                printf("  Processed: %.1f s, ERLE: %.1f dB, delay: %d%s\r",
                       (float)processed / sample_rate, erle, delay,
                       aec_is_converged(aec) ? " [converged]" : "");
            else
                printf("  Processed: %.1f s, ERLE: %.1f dB%s\r",
                       (float)processed / sample_rate, erle,
                       aec_is_converged(aec) ? " [converged]" : "");
            fflush(stdout);
        }
    }

    printf("\n\nResults:\n");
    printf("  Processed samples: %d\n", processed);
    printf("  Max ERLE: %.1f dB\n", max_erle);
    printf("  Final ERLE: %.1f dB\n", aec_get_erle(aec));
    printf("  Converged: %s\n", aec_is_converged(aec) ? "yes" : "no");
    {
        int delay = aec_get_delay(aec);
        if (delay >= 0)
            printf("  Estimated delay: %d samples (%.1f ms)\n",
                   delay, 1000.0f * delay / sample_rate);
        else if (config.enable_delay_est)
            printf("  Estimated delay: not determined\n");
    }
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
