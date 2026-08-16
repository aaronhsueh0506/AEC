/* aec_wav.c — WAV processing CLI on top of aec (Wave 5).
 *
 * User fixes:
 *   1. balanced reverb 0.85/1.6 — set in aec_config_from_preset
 *   2. CNG default OFF, --cng to enable (was: forced ON in old aec_dump)
 *   3. --preset mild now parses (old parse_preset silently fell back to balanced)
 *   6. Hop size dynamic (no [160] magic) — aec derives from sample_rate
 *
 * Plus --debug-level for log infra.
 */
#include "aec.h"
#include "aec_debug.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <mic.wav> <ref.wav> <out.wav> [options]\n\n"
        "Options:\n"
        "  --preset {mild|balanced|aggressive}   default: balanced\n"
        "  --cng                          Enable comfort noise\n"
        "  --no-cng                       Explicitly disable CNG\n"
        "  --no-delay-est                 EXTERNAL_ALIGNED mode: no online delay\n"
        "                                 estimation (inputs assumed pre-aligned)\n"
        "  --no-res                       Disable residual echo suppressor\n"
        "  --no-shadow                    Disable shadow filter\n"
        "  --no-hpf                       Disable 80Hz high-pass\n"
        "  --fft-size <n>                 256/512 @16kHz; 1024 @48kHz\n"
        "  --delay-mode {matched|fixed|external}\n"
        "                                 alignment mode (default: matched)\n"
        "  --delay-num-filters <1..5>     MATCHED matched-filter bank size (default 5)\n"
        "  --fixed-delay <samples>        FIXED delay, native-rate samples (>=0)\n"
        "  --print-mem-size               print the resolved grid/mode/n and the\n"
        "                                 static-pool byte breakdown, then exit\n"
        "                                 without touching any audio\n"
        "  --debug-level <0..3>           0=off, 1=summary, 2=per-frame, 3=full\n"
        "  --debug-log <path>             redirect log to file (default stderr)\n"
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

/* Config rejection is reported identically wherever it happens: the two
 * entry points that can reject (aec_get_mem_breakdown() under
 * --print-mem-size, aec_create() on the render path) run the SAME
 * aec_validate_config(), so a caller who hits one must not be told less
 * than a caller who hits the other. Prints every delay field as it was
 * actually resolved plus the full legality table, so the user can see which
 * of their flags is the illegal one without re-running under the other
 * entry point. */
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

static int parse_preset(const char* s, AecPreset* out) {
    if (!strcmp(s, "mild"))     { *out = AEC_PRESET_MILD;     return 0; }
    if (!strcmp(s, "balanced"))   { *out = AEC_PRESET_BALANCED;   return 0; }
    if (!strcmp(s, "aggressive")) { *out = AEC_PRESET_AGGRESSIVE; return 0; }
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc < 4) { print_usage(argv[0]); return 1; }
    const char* mic_path = argv[1];
    const char* ref_path = argv[2];
    const char* out_path = argv[3];

    AecPreset preset = AEC_PRESET_BALANCED;
    int explicit_cng = -1;
    int no_delay_est = 0, no_res = 0, no_shadow = 0, no_hpf = 0;
    int debug_level = 0;
    const char* debug_log = NULL;
    const char* debug_trace = NULL;
    int debug_status = 0;   /* --debug: periodic aec_debug_status() print */
    int fft_size = 0;
    /* Delay-mode overrides. A separate "was it given" flag rather than a
     * sentinel value for the same reason as --fft-size==0 above: overloading
     * a magic value as "not given" would make an explicit, legal user value
     * indistinguishable from "unset". Each is handed to aec_create()
     * UNVALIDATED on purpose -- aec_validate_config() is the single range
     * and mode/field-compatibility authority (productization plan §2.1), so
     * an illegal value or combination fails the run there rather than being
     * silently clamped or ignored into a different configuration than the
     * one asked for. */
    AecDelayMode delay_mode = AEC_DELAY_MATCHED;  int have_delay_mode = 0;
    int delay_num_filters = 0;                    int have_delay_num_filters = 0;
    int fixed_delay_samples = 0;                  int have_fixed_delay = 0;
    int print_mem_size = 0;

    for (int i = 4; i < argc; ++i) {
        const char* arg = argv[i];
        if (!strcmp(arg, "--preset") && i + 1 < argc) {
            if (parse_preset(argv[++i], &preset) != 0) {
                fprintf(stderr, "ERROR: unknown preset '%s'. "
                                "Valid: mild|balanced|aggressive\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--cng"))           explicit_cng = 1;
        else if (!strcmp(arg, "--no-cng"))          explicit_cng = 0;
        else if (!strcmp(arg, "--no-delay-est"))    no_delay_est = 1;
        else if (!strcmp(arg, "--no-res"))          no_res = 1;
        else if (!strcmp(arg, "--no-shadow"))       no_shadow = 1;
        else if (!strcmp(arg, "--no-hpf"))          no_hpf = 1;
        else if (!strcmp(arg, "--fft-size") && i + 1 < argc)
            fft_size = atoi(argv[++i]);
        else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (parse_delay_mode(argv[++i], &delay_mode) != 0) {
                fprintf(stderr, "ERROR: unknown --delay-mode '%s'. "
                                "Valid: matched|fixed|external\n", argv[i]);
                return 2;
            }
            have_delay_mode = 1;
        } else if (!strcmp(arg, "--delay-num-filters") && i + 1 < argc) {
            delay_num_filters = atoi(argv[++i]); have_delay_num_filters = 1;
        } else if (!strcmp(arg, "--fixed-delay") && i + 1 < argc) {
            fixed_delay_samples = atoi(argv[++i]); have_fixed_delay = 1;
        } else if (!strcmp(arg, "--print-mem-size"))
            print_mem_size = 1;
        else if (!strcmp(arg, "--debug-level") && i + 1 < argc)
            debug_level = atoi(argv[++i]);
        else if (!strcmp(arg, "--debug-log") && i + 1 < argc)
            debug_log = argv[++i];
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

    /* F05: reject an unsupported sample rate (or a mic/ref rate mismatch)
     * before ever constructing an AecConfig — aec_create would reject the
     * same rate today via aec_validate_config, but failing here gives a
     * clear, WAV-specific message instead of a bare aec_create() failure. */
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
    aec_config_from_preset(&cfg, preset, sr);
    /* Grid resolution happens HERE, in the CLI, before the library is
     * called: a user who did not pass --fft-size gets this rate's product
     * default (already filled by aec_config_from_preset via
     * aec_default_fft_size), and the value handed to aec_create() is always
     * a concrete nonzero fft_size. The core deliberately refuses
     * fft_size == 0 -- 16 kHz has two production grids and the library must
     * not guess which one a caller meant. Rejecting the unresolved pair
     * here (rather than letting aec_create() fail with the same message
     * further down) also catches a bad --fft-size before any allocation. */
    if (fft_size > 0) cfg.fft_size = fft_size;
    if (!aec_resolve_signal_grid(sr, cfg.fft_size, NULL)) {
        fprintf(stderr,
                "ERROR: unsupported AEC signal grid (sr=%d, fft=%d); "
                "use 256/512 @16kHz, 1024 @48kHz, or 256 @8kHz (legacy)\n",
                sr, cfg.fft_size);
        wav_close_read(mr); wav_close_read(rr);
        return 4;
    }
    if (explicit_cng >= 0) cfg.enable_cng = explicit_cng;
    if (no_delay_est) cfg.enable_delay_est = 0;
    if (no_res)       cfg.enable_res = 0;
    if (no_shadow)    cfg.enable_shadow = 0;
    if (no_hpf)       cfg.enable_highpass = 0;
    if (have_delay_mode)         cfg.delay_mode = delay_mode;
    if (have_delay_num_filters)  cfg.delay_num_filters = delay_num_filters;
    if (have_fixed_delay)        cfg.fixed_delay_samples = fixed_delay_samples;

    /* --print-mem-size: answer the static-pool question and exit, before
     * any debug/log setup or Aec construction — no audio is touched (plan
     * §3.4.6 RAM acceptance test 6). aec_get_mem_breakdown() runs the same
     * aec_validate_config() every other entry point does, so an illegal
     * mode/field combination fails HERE with the library's own validation
     * outcome (fail-fast), not a silently-different config. */
    if (print_mem_size) {
        AecSignalGrid grid;
        aec_resolve_signal_grid(sr, cfg.fft_size, &grid);   /* already proven valid above */
        AecMemBreakdown mb;
        if (!aec_get_mem_breakdown(&cfg, &mb)) {
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

    aec_debug_set_level(debug_level);
    if (debug_log) {
        FILE* fp = fopen(debug_log, "w");
        if (fp) aec_debug_set_log(fp);
    }
    FILE* trace_fp = NULL;
    if (debug_trace) {
        trace_fp = fopen(debug_trace, "w");
        if (trace_fp) aec_debug_set_trace(trace_fp);
        else fprintf(stderr, "WARN: cannot open trace file '%s'\n", debug_trace);
    }

    Aec aec;
    if (aec_create(&aec, &cfg) != 0) {
        print_config_rejected("aec_create", sr, &cfg);
        wav_close_read(mr); wav_close_read(rr);
        if (trace_fp) { aec_debug_set_trace(NULL); fclose(trace_fp); }
        return 4;
    }
    int hop = aec_hop_size(&aec);
    aec_debug_set_frame(0, hop, sr);

    /* Default to float32 output for parity-friendly comparison; override
     * via AEC_OUT_FLOAT=0 if 16-bit PCM needed. v3.10.4 fix: env var key
     * was AEC_FP32_WAV but wav_io.h:215 reads AEC_OUT_FLOAT — keys agree
     * now so the float32 default actually takes effect. */
    if (!getenv("AEC_OUT_FLOAT")) setenv("AEC_OUT_FLOAT", "1", 1);
    WavWriter* ww = wav_open_write(out_path, sr, 1);
    if (!ww) { fprintf(stderr, "ERROR: cannot write output\n"); return 3; }

    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));
    float* out = (float*)malloc((size_t)hop * sizeof(float));

    /* --debug cadence: approximately once per second on every signal grid. */
    int frames_per_sec = hop > 0 ? sr / hop : 0;
    if (frames_per_sec <= 0) frames_per_sec = 1;

    int frame_idx = 0;
    for (int i = 0; i + hop <= n; i += hop) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        aec_debug_set_frame(frame_idx, hop, sr);
        aec_process(&aec, mic, ref, out);
        wav_write_float(ww, out, hop);
        frame_idx++;

        if (debug_status && frame_idx % frames_per_sec == 0) {
            AecDebugStatus st;
            aec_debug_status(&aec, &st);
            fprintf(stderr,
                "[dbg %5.1fs] delay=%d (conf %.1f, upd %d) erle=%.1fdB "
                "usable_lin=%d conv=%d near=%.2e out=%.2e\n",
                (double)frame_idx / frames_per_sec,
                st.delay_samples, st.delay_confidence, st.delay_updates,
                st.erle_windowed_db, st.usable_linear, st.filter_converged,
                st.near_power, st.out_power);
        }
    }

    fprintf(stderr, "Processed %d frames @ hop=%d sr=%d preset=%s cng=%d delay_est=%d "
            "delay_mode=%s num_filters=%d fixed_delay_samples=%d\n",
            frame_idx, hop, sr,
            preset == AEC_PRESET_MILD ? "mild" :
            preset == AEC_PRESET_BALANCED ? "balanced" : "aggressive",
            cfg.enable_cng, cfg.enable_delay_est,
            delay_mode_name(cfg.delay_mode), cfg.delay_num_filters,
            cfg.fixed_delay_samples);

    /* Duty-cycle engagement census for this run (plan §7 measurement method
     * -- see the duty_hops_* doc in aec.h). Printed unconditionally: it is
     * one line, and the whole point is that the "~90% of the matched-filter
     * cost" design figure stops being an assumption and becomes a number a
     * bench can read off every run. */
    {
        AecDebugStatus st;
        aec_debug_status(&aec, &st);
        if (st.duty_hops_total == 0) {
            /* No matched filter was ever constructed this run (any mode
             * other than MATCHED, or the legacy enable_delay_est=0 mirror
             * of it) -- there was no matched-filter cost to decimate, so
             * reporting a "% saved" here would claim a saving for a machine
             * that never ran at all. */
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
    aec_destroy(&aec);
    if (trace_fp) { aec_debug_set_trace(NULL); fclose(trace_fp); }
    return 0;
}
