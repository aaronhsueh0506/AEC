/* aec_wav.c -- WAV processing CLI on top of aec.
 *  - --level takes 0 mild, 1 balanced, 2 aggressive or 3 no_floor (a whole decimal number); --preset is an alias.
 *  - The hop size comes from the library's signal grid (fft_size / 2).
 *  - --debug prints the periodic status line; --debug-trace writes the per-frame CSV trace. */
#include "aec.h"
#include "cli_level.h"
#include "aec_debug.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <mic.wav> <ref.wav> <out.wav> [options]\n\n"
        "Options:\n"
        "  --level {0|1|2|3}              0 mild, 1 balanced (default), 2 aggressive, 3 no_floor\n"
        "                                 (--preset is accepted as an alias)\n"
        "  --cng                          Enable comfort noise\n"
        "  --no-cng                       Explicitly disable CNG\n"
        "  --no-delay-est                 EXTERNAL_ALIGNED mode: no online delay\n"
        "                                 estimation (inputs assumed pre-aligned)\n"
        "  --no-res                       Disable residual echo suppressor\n"
        "  --fft-size <n>                 256/512 @16kHz; 1024 @48kHz\n"
        "  --delay-mode {matched|fixed|external}\n"
        "                                 alignment mode (default: matched)\n"
        "  --delay-num-filters <1..5>     MATCHED matched-filter bank size (default 5)\n"
        "  --fixed-delay <samples>        FIXED delay, native-rate samples (>=0)\n"
        "  --print-mem-size               print the resolved grid/mode/n and the\n"
        "                                 static-pool byte breakdown, then exit\n"
        "                                 without touching any audio\n"
        "  --debug-trace <path>           per-frame CSV trace (logr) of post-filter state\n"
        "  --debug                        print one aec_debug_status() line per second\n",
        prog);
}

static int parse_delay_mode(const char* s, AecDelayMode* out) {
    if (!strcmp(s, "matched"))  { *out = AEC_DELAY_MATCHED;          return 0; }
    if (!strcmp(s, "fixed"))    { *out = AEC_DELAY_FIXED;            return 0; }
    if (!strcmp(s, "external")) { *out = AEC_DELAY_EXTERNAL_ALIGNED; return 0; }
    return -1;
}

static const char* delay_mode_name(AecDelayMode m) {
    switch (m) {
        case AEC_DELAY_MATCHED:          return "matched";
        case AEC_DELAY_FIXED:            return "fixed";
        case AEC_DELAY_EXTERNAL_ALIGNED: return "external";
        default:                         return "?";
    }
}

/* Config rejection is reported the same way wherever it happens (aec_get_mem_breakdown() under
 * --print-mem-size, aec_create() otherwise): prints every delay field as resolved plus the legality
 * table. */
static void print_config_rejected(const char* who, int sr,
                                  const AecConfig* cfg) {
    fprintf(stderr,
            "ERROR: %s rejected the config (sr=%d, fft=%d, "
            "delay_mode=%s, delay_num_filters=%d, fixed_delay_samples=%d); "
            "grid must be 256/512 @16kHz, 1024 @48kHz, or 256 @8kHz (legacy); "
            "MATCHED needs delay_num_filters 1..5 and fixed_delay_samples=-1, "
            "FIXED needs fixed_delay_samples>=0 and delay_num_filters=5, "
            "external needs delay_num_filters=5 and fixed_delay_samples=-1\n",
            who, sr, cfg->fft_size, delay_mode_name(cfg->delay_mode),
            cfg->delay_num_filters, cfg->fixed_delay_samples);
}

static int bad_int(const char* option, const char* text) {
    fprintf(stderr, "ERROR: %s '%s' is not an integer\n", option, text);
    return 2;
}

int main(int argc, char* argv[]) {
    if (argc < 4) { print_usage(argv[0]); return 1; }
    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecLevel level = AEC_LEVEL_BALANCED;
    int explicit_cng = -1;
    int no_delay_est = 0, no_res = 0;
    const char* debug_trace = NULL;
    int debug_status = 0;
    int fft_size = 0;
    /* Delay-mode overrides use a separate "given" flag, and are handed to aec_create() unvalidated
     * so aec_resolve() is the single authority; illegal values fail the run instead of
     * being clamped. */
    AecDelayMode delay_mode = AEC_DELAY_MATCHED;  int have_delay_mode = 0;
    int delay_num_filters = 0;                    int have_delay_num_filters = 0;
    int fixed_delay_samples = 0;                  int have_fixed_delay = 0;
    int print_mem_size = 0;

    for (int i = 4; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "--level") || !strcmp(arg, "--preset")) && i + 1 < argc) {
            if (aec_cli_parse_level(argv[++i], &level) != 0) {
                fprintf(stderr, "ERROR: unknown level '%s'. Valid: 0 (mild) | 1 (balanced) | 2 (aggressive) | 3 (no_floor)\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--cng"))           explicit_cng = 1;
        else if (!strcmp(arg, "--no-cng"))          explicit_cng = 0;
        else if (!strcmp(arg, "--no-delay-est"))    no_delay_est = 1;
        else if (!strcmp(arg, "--no-res"))          no_res = 1;
        else if (!strcmp(arg, "--fft-size") && i + 1 < argc) {
            if (aec_cli_parse_int(argv[++i], &fft_size) != 0) return bad_int(arg, argv[i]);
        }
        else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (parse_delay_mode(argv[++i], &delay_mode) != 0) {
                fprintf(stderr, "ERROR: unknown --delay-mode '%s'. "
                                "Valid: matched|fixed|external\n", argv[i]);
                return 2;
            }
            have_delay_mode = 1;
        } else if (!strcmp(arg, "--delay-num-filters") && i + 1 < argc) {
            if (aec_cli_parse_int(argv[++i], &delay_num_filters) != 0) return bad_int(arg, argv[i]);
            have_delay_num_filters = 1;
        } else if (!strcmp(arg, "--fixed-delay") && i + 1 < argc) {
            if (aec_cli_parse_int(argv[++i], &fixed_delay_samples) != 0) return bad_int(arg, argv[i]);
            have_fixed_delay = 1;
        } else if (!strcmp(arg, "--print-mem-size"))
            print_mem_size = 1;
        else if (!strcmp(arg, "--debug-trace") && i + 1 < argc)
            debug_trace = argv[++i];
        else if (!strcmp(arg, "--debug"))
            debug_status = 1;
        else {
            fprintf(stderr, "ERROR: unknown option '%s'\n", arg);
            return 2;
        }
    }

    WavReader* mr = wav_open_read(mic_path);
    WavReader* rr = wav_open_read(ref_path);
    if (!mr || !rr) {
        fprintf(stderr, "ERROR: cannot open input wavs\n"); return 3;
    }

    /* Reject an unsupported sample rate (or a mic/ref rate mismatch) with a WAV-specific message. */
    if (!aec_is_valid_sample_rate(mr->info.sample_rate)) {
        fprintf(stderr,
            "ERROR: unsupported mic sample rate %d Hz (AEC whitelist: 8000/16000/48000 Hz)\n",
            mr->info.sample_rate);
        return 4;
    }
    if (!aec_is_valid_sample_rate(rr->info.sample_rate)) {
        fprintf(stderr,
            "ERROR: unsupported ref sample rate %d Hz (AEC whitelist: 8000/16000/48000 Hz)\n",
            rr->info.sample_rate);
        return 4;
    }
    if (mr->info.sample_rate != rr->info.sample_rate) {
        fprintf(stderr,
            "ERROR: mic/ref sample-rate mismatch (mic=%d Hz, ref=%d Hz)\n",
            mr->info.sample_rate, rr->info.sample_rate);
        return 4;
    }

    int sr = mr->info.sample_rate;
    int n  = mr->info.num_samples < rr->info.num_samples
           ? mr->info.num_samples : rr->info.num_samples;

    AecConfig cfg;
    aec_config_from_level(&cfg, level, sr);
    /* Grid resolution happens here: the fft_size handed to aec_create() is always concrete (the
     * rate's default unless --fft-size is given), since the core refuses fft_size == 0. */
    if (fft_size > 0) cfg.fft_size = fft_size;
    if (aec_resolve_signal_grid(sr, cfg.fft_size, NULL) != AEC_OK) {
        fprintf(stderr,
                "ERROR: unsupported AEC signal grid (sr=%d, fft=%d); "
                "use 256/512 @16kHz, 1024 @48kHz, or 256 @8kHz (legacy)\n",
                sr, cfg.fft_size);
        wav_close_read(mr); wav_close_read(rr);
        return 4;
    }
    if (explicit_cng >= 0) cfg.enable_cng = explicit_cng;
    if (no_res)       cfg.residual = AEC_RESIDUAL_OFF;
    if (have_delay_mode)         cfg.delay_mode = delay_mode;
    if (have_delay_num_filters)  cfg.delay_num_filters = delay_num_filters;
    if (have_fixed_delay)        cfg.fixed_delay_samples = fixed_delay_samples;
    if (no_delay_est && cfg.delay_mode == AEC_DELAY_MATCHED)
        cfg.delay_mode = cfg.fixed_delay_samples >= 0 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;

    /* --print-mem-size: report the static-pool size and exit before any audio is touched. */
    if (print_mem_size) {
        AecSignalGrid grid;
        aec_resolve_signal_grid(sr, cfg.fft_size, &grid);
        AecMemBreakdown mb;
        if (aec_get_mem_breakdown(&cfg, &mb) != AEC_OK) {
            print_config_rejected("aec_get_mem_breakdown", sr, &cfg);
            wav_close_read(mr); wav_close_read(rr);
            return 4;
        }
        printf("mem: sr=%d fft=%d hop=%d mode=%s n=%d fixed_delay_samples=%d "
               "total_bytes=%zu estimator_bytes=%zu ring_bytes=%zu\n",
               grid.sample_rate, grid.fft_size, grid.hop_size,
               delay_mode_name(cfg.delay_mode), cfg.delay_num_filters,
               cfg.fixed_delay_samples,
               mb.total_bytes, mb.estimator_bytes, mb.ring_bytes);
        wav_close_read(mr); wav_close_read(rr);
        return 0;
    }

    FILE* trace_fp = NULL;
    if (debug_trace) {
        trace_fp = fopen(debug_trace, "w");
        if (trace_fp) aec_debug_set_trace(trace_fp);
        else fprintf(stderr, "WARN: cannot open trace file '%s'\n", debug_trace);
    }

    Aec* aec = aec_create(&cfg);
    if (!aec) {
        print_config_rejected("aec_create", sr, &cfg);
        wav_close_read(mr); wav_close_read(rr);
        if (trace_fp) { aec_debug_set_trace(NULL); fclose(trace_fp); }
        return 4;
    }
    int hop = aec_hop_size(aec);
    aec_debug_set_frame(0, hop, sr);

    /* Default to float32 output; AEC_OUT_FLOAT=0 selects 16-bit PCM. */
    if (!getenv("AEC_OUT_FLOAT")) setenv("AEC_OUT_FLOAT", "1", 1);
    WavWriter* ww = wav_open_write(out_path, sr, 1);
    if (!ww) { fprintf(stderr, "ERROR: cannot write output\n"); return 3; }

    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    float* out = (float*)malloc((size_t)hop * sizeof(float));

    /* --debug cadence: about once per second. */
    int frames_per_sec = hop > 0 ? sr / hop : 0;
    if (frames_per_sec <= 0) frames_per_sec = 1;

    int frame_idx = 0;
    for (int i = 0; i + hop <= n; i += hop) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        aec_debug_set_frame(frame_idx, hop, sr);
        aec_process(aec, mic, ref, out);
        wav_write_float(ww, out, hop);
        frame_idx++;

        if (debug_status && frame_idx % frames_per_sec == 0) {
            AecDebugStatus st;
            aec_debug_status(aec, &st);
            fprintf(stderr,
                "[dbg %5.1fs] delay=%d (conf %.1f, upd %d) erle=%.1fdB "
                "usable_lin=%d conv=%d near=%.2e out=%.2e\n",
                (double)frame_idx / frames_per_sec,
                st.delay_samples, st.delay_confidence, st.delay_updates,
                st.erle_windowed_db, st.usable_linear, st.filter_converged,
                st.near_power, st.out_power);
        }
    }

    fprintf(stderr, "Processed %d frames @ hop=%d sr=%d level=%s cng=%d delay_est=%d "
            "delay_mode=%s num_filters=%d fixed_delay_samples=%d\n",
            frame_idx, hop, sr,
            aec_level_name(level),
            cfg.enable_cng, cfg.delay_mode == AEC_DELAY_MATCHED,
            delay_mode_name(cfg.delay_mode), cfg.delay_num_filters,
            cfg.fixed_delay_samples);

    /* Duty-cycle engagement census for this run (see the duty_hops_* fields in aec.h). */
    {
        AecDebugStatus st;
        aec_debug_status(aec, &st);
        if (st.duty_hops_total == 0) {
            /* No matched filter is constructed in this mode, so no saving is reported. */
            fprintf(stderr, "duty: matched_filter_ran=0/0 hops "
                            "(n/a -- no matched-filter estimator in delay_mode=%s)\n",
                    delay_mode_name(cfg.delay_mode));
        } else {
            double eng = 100.0 * (double)st.duty_hops_run
                       / (double)st.duty_hops_total;
            fprintf(stderr, "duty: matched_filter_ran=%llu/%llu hops "
                    "(%.2f%% engagement, %.2f%% saved)\n",
                    st.duty_hops_run, st.duty_hops_total, eng, 100.0 - eng);
        }
    }

    free(mic); free(ref); free(out);
    wav_close_write(ww);
    wav_close_read(mr); wav_close_read(rr);
    aec_destroy(aec);
    if (trace_fp) { aec_debug_set_trace(NULL); fclose(trace_fp); }
    return 0;
}
