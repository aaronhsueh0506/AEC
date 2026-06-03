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
        "  --preset {gentle|balanced|aggressive}   default: balanced\n"
        "  --cng                          Enable comfort noise\n"
        "  --no-cng                       Explicitly disable CNG\n"
        "  --no-delay-est                 Disable online delay estimation\n"
        "  --no-res                       Disable residual echo suppressor\n"
        "  --no-shadow                    Disable shadow filter\n"
        "  --no-hpf                       Disable 80Hz high-pass\n"
        "  --debug-level <0..3>           0=off, 1=summary, 2=per-frame, 3=full\n"
        "  --debug-log <path>             redirect log to file (default stderr)\n",
        prog);
}

static int parse_preset(const char* s, AecPreset* out) {
    if (!strcmp(s, "gentle"))     { *out = AEC_PRESET_GENTLE;     return 0; }
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

    for (int i = 4; i < argc; ++i) {
        const char* arg = argv[i];
        if (!strcmp(arg, "--preset") && i + 1 < argc) {
            if (parse_preset(argv[++i], &preset) != 0) {
                fprintf(stderr, "ERROR: unknown preset '%s'. "
                                "Valid: gentle|balanced|aggressive\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--cng"))           explicit_cng = 1;
        else if (!strcmp(arg, "--no-cng"))          explicit_cng = 0;
        else if (!strcmp(arg, "--no-delay-est"))    no_delay_est = 1;
        else if (!strcmp(arg, "--no-res"))          no_res = 1;
        else if (!strcmp(arg, "--no-shadow"))       no_shadow = 1;
        else if (!strcmp(arg, "--no-hpf"))          no_hpf = 1;
        else if (!strcmp(arg, "--debug-level") && i + 1 < argc)
            debug_level = atoi(argv[++i]);
        else if (!strcmp(arg, "--debug-log") && i + 1 < argc)
            debug_log = argv[++i];
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
    int sr = mr->info.sample_rate;
    int n  = mr->info.num_samples < rr->info.num_samples
           ? mr->info.num_samples : rr->info.num_samples;

    AecConfig cfg;
    aec_config_from_preset(&cfg, preset, sr);
    if (explicit_cng >= 0) cfg.enable_cng = explicit_cng;
    if (no_delay_est) cfg.enable_delay_est = 0;
    if (no_res)       cfg.enable_res = 0;
    if (no_shadow)    cfg.enable_shadow = 0;
    if (no_hpf)       cfg.enable_highpass = 0;

    aec_debug_set_level(debug_level);
    if (debug_log) {
        FILE* fp = fopen(debug_log, "w");
        if (fp) aec_debug_set_log(fp);
    }

    Aec aec;
    aec_create(&aec, &cfg);
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

    int frame_idx = 0;
    for (int i = 0; i + hop <= n; i += hop) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        aec_debug_set_frame(frame_idx, hop, sr);
        aec_process(&aec, mic, ref, out);
        wav_write_float(ww, out, hop);
        frame_idx++;
    }

    fprintf(stderr, "Processed %d frames @ hop=%d sr=%d preset=%s cng=%d delay_est=%d\n",
            frame_idx, hop, sr,
            preset == AEC_PRESET_GENTLE ? "gentle" :
            preset == AEC_PRESET_BALANCED ? "balanced" : "aggressive",
            cfg.enable_cng, cfg.enable_delay_est);

    free(mic); free(ref); free(out);
    wav_close_write(ww);
    wav_close_read(mr); wav_close_read(rr);
    aec_destroy(&aec);
    return 0;
}
