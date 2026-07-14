/* bench_rtf.c — per-module real-time-factor (RTF) benchmark harness.
 *
 * Standalone, zero-host-dependency timing harness for the NEON optimization
 * campaign: measures wall-clock cost of aec_process() over a long synthetic
 * (or --wav) input, with and without the AEC3 post-filter (enable_res), so
 * the post-filter's own cost can be isolated as a derived third line.
 *
 * No Python, no corpus, no network required by default — a deterministic
 * LCG generates the far-end/near-end signals in-process. --wav is available
 * for spot-checking against a real recording but is not needed for routine
 * profiling.
 *
 * Build: `make bench` from c_impl/ (respects BACKEND=kiss|ne10 and
 * EXTRA_CFLAGS, same as bin/aec_wav). Cross-compiles the same way any other
 * c_impl target does (set CC / EXTRA_CFLAGS for the target toolchain).
 *
 * Usage:
 *   ./bin/bench_rtf [--seconds N] [--repeat M]
 *                    [--preset mild|balanced|aggressive]
 *                    [--wav mic.wav ref.wav]
 *
 * Output (one line per stage, machine-diffable):
 *   stage=<name> hops=<n> total_s=<x.xxxxxx> us_per_hop=<x.xx> rtf=<x.xxxxxx>
 *
 * Stages:
 *   full   — production config (preset's enable_res=1, enable_cng=1; the
 *            same config aec_wav's `--preset <p> --cng` path builds).
 *   linear — identical config with enable_res=0 (linear-filter-only path).
 *   post   — derived (full.total_s - linear.total_s); an approximation of
 *            the AEC3 post-filter chain's own cost, not an independently
 *            timed stage.
 */
#include "aec.h"
#include "wav_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── deterministic synthetic input ─────────────────────────────────────── */

/* Classic Numerical-Recipes LCG. Top 16 bits are far better distributed than
 * the low bits, so we take those as an int16-range sample and normalize the
 * same way a 16-bit PCM sample would be (/32768.0f). */
#define LCG_MULT 1664525u
#define LCG_INC  1013904223u

static uint32_t g_lcg_state;

static float lcg_next_sample(void) {
    g_lcg_state = g_lcg_state * LCG_MULT + LCG_INC;
    int16_t v = (int16_t)(g_lcg_state >> 16);
    return (float)v / 32768.0f;
}

/* far = white noise (full int16-normalized scale).
 * mic (near) = 0.3x-attenuated copy of far delayed 40ms (simulated echo
 * path) + a 440 Hz tone gated on 1s / off 1s at 0.1 amplitude (near-end
 * speech proxy). Fully deterministic — same output every run. */
static void gen_synthetic(float* mic, float* far, int n, int sample_rate) {
    g_lcg_state = 0u;
    const int   delay_samples = (int)(0.040 * sample_rate + 0.5);
    const float echo_gain     = 0.3f;
    const float tone_amp      = 0.1f;
    const float tone_hz       = 440.0f;

    for (int i = 0; i < n; ++i) far[i] = lcg_next_sample();

    for (int i = 0; i < n; ++i) {
        float echo = (i >= delay_samples) ? echo_gain * far[i - delay_samples] : 0.0f;
        float tone = 0.0f;
        if (((i / sample_rate) % 2) == 0) {
            double t = (double)i / (double)sample_rate;
            tone = tone_amp * (float)sin(2.0 * M_PI * tone_hz * t);
        }
        mic[i] = echo + tone;
    }
}

/* ── CLI ────────────────────────────────────────────────────────────────── */

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  --seconds N                          synthetic input length (default 60)\n"
        "  --repeat M                            passes per stage, report min (default 1)\n"
        "  --preset {mild|balanced|aggressive}   default: balanced\n"
        "  --wav <mic.wav> <ref.wav>             use a recording instead of synthetic input\n",
        prog);
}

static int parse_preset(const char* s, AecPreset* out) {
    if (!strcmp(s, "mild"))       { *out = AEC_PRESET_MILD;       return 0; }
    if (!strcmp(s, "balanced"))   { *out = AEC_PRESET_BALANCED;   return 0; }
    if (!strcmp(s, "aggressive")) { *out = AEC_PRESET_AGGRESSIVE; return 0; }
    return -1;
}

static const char* preset_name(AecPreset p) {
    switch (p) {
        case AEC_PRESET_MILD:       return "mild";
        case AEC_PRESET_AGGRESSIVE: return "aggressive";
        default:                    return "balanced";
    }
}

/* ── timing ─────────────────────────────────────────────────────────────── */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Runs `repeat` full passes of aec_process() over [mic,ref) x n_hops hops
 * under `cfg`, prints the stage line, and returns the minimum total_s
 * observed (least-noise estimate). One aec_create/aec_destroy per stage;
 * aec_reset() between repeats restores a cold-start state (no realloc) so
 * every repeat measures the same condition rather than a warmed-up tail. */
static double run_stage(const char* name, const AecConfig* cfg,
                         const float* mic_buf, const float* ref_buf,
                         int hop, int n_hops, int repeat, double audio_seconds) {
    Aec aec;
    if (aec_create(&aec, cfg) != 0) {
        fprintf(stderr, "ERROR: aec_create failed for stage '%s'\n", name);
        exit(4);
    }
    float* out = (float*)malloc((size_t)hop * sizeof(float));

    double best = -1.0;
    for (int r = 0; r < repeat; ++r) {
        aec_reset(&aec);
        const float* mp = mic_buf;
        const float* rp = ref_buf;

        double t0 = now_s();
        for (int h = 0; h < n_hops; ++h) {
            aec_process(&aec, mp, rp, out);
            mp += hop;
            rp += hop;
        }
        double dt = now_s() - t0;
        if (best < 0.0 || dt < best) best = dt;
    }

    free(out);
    aec_destroy(&aec);

    double us_per_hop = n_hops > 0 ? (best * 1e6) / (double)n_hops : 0.0;
    double rtf = audio_seconds > 0.0 ? best / audio_seconds : 0.0;
    printf("stage=%s hops=%d total_s=%.6f us_per_hop=%.2f rtf=%.6f\n",
           name, n_hops, best, us_per_hop, rtf);
    return best;
}

int main(int argc, char** argv) {
    int seconds = 60;
    int repeat = 1;
    AecPreset preset = AEC_PRESET_BALANCED;
    int use_wav = 0;
    const char* mic_path = NULL;
    const char* ref_path = NULL;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!strcmp(arg, "--seconds") && i + 1 < argc) {
            seconds = atoi(argv[++i]);
        } else if (!strcmp(arg, "--repeat") && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (!strcmp(arg, "--preset") && i + 1 < argc) {
            if (parse_preset(argv[++i], &preset) != 0) {
                fprintf(stderr, "ERROR: unknown preset '%s'. Valid: mild|balanced|aggressive\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--wav") && i + 2 < argc) {
            mic_path = argv[++i];
            ref_path = argv[++i];
            use_wav = 1;
        } else if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ERROR: unknown option '%s'\n", arg);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (seconds <= 0) { fprintf(stderr, "ERROR: --seconds must be > 0\n"); return 2; }
    if (repeat  <= 0) { fprintf(stderr, "ERROR: --repeat must be > 0\n");  return 2; }

    int sample_rate;
    int n_samples;
    float* mic_buf;
    float* ref_buf;

    if (use_wav) {
        WavReader* mr = wav_open_read(mic_path);
        WavReader* rr = wav_open_read(ref_path);
        if (!mr || !rr) {
            fprintf(stderr, "ERROR: cannot open input wavs\n");
            return 3;
        }
        sample_rate = mr->info.sample_rate;
        n_samples = mr->info.num_samples < rr->info.num_samples
                  ? mr->info.num_samples : rr->info.num_samples;

        mic_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        ref_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        wav_read_float(mr, mic_buf, n_samples);
        wav_read_float(rr, ref_buf, n_samples);
        wav_close_read(mr);
        wav_close_read(rr);
    } else {
        sample_rate = 16000;
        n_samples = seconds * sample_rate;
        mic_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        ref_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        gen_synthetic(mic_buf, ref_buf, n_samples, sample_rate);
    }

    AecConfig full_cfg;
    aec_config_from_preset(&full_cfg, preset, sample_rate);
    full_cfg.enable_cng = 1;   /* mirrors aec_wav's --cng path */

    /* Probe hop size once (hop is a pure function of sample_rate/preset,
     * identical for the full and linear configs below); this instance is
     * not part of either timed stage. */
    Aec probe;
    if (aec_create(&probe, &full_cfg) != 0) {
        fprintf(stderr, "ERROR: aec_create (probe) failed\n");
        return 4;
    }
    int hop = aec_hop_size(&probe);
    aec_destroy(&probe);

    int n_hops = hop > 0 ? n_samples / hop : 0;
    double audio_seconds = (double)n_hops * hop / (double)sample_rate;

    printf("# preset=%s input=%s sample_rate=%d hop=%d audio_s=%.3f\n",
           preset_name(preset), use_wav ? "wav" : "synthetic", sample_rate, hop, audio_seconds);
    printf("runs=%d\n", repeat);

    double full_s = run_stage("full", &full_cfg, mic_buf, ref_buf, hop, n_hops, repeat, audio_seconds);

    AecConfig linear_cfg = full_cfg;
    linear_cfg.enable_res = 0;
    double linear_s = run_stage("linear", &linear_cfg, mic_buf, ref_buf, hop, n_hops, repeat, audio_seconds);

    /* Derived: post-filter cost ≈ full - linear. Not an independently timed
     * stage (see file header); can be noisy/negative on a loaded machine. */
    double post_s = full_s - linear_s;
    double post_us_per_hop = n_hops > 0 ? (post_s * 1e6) / (double)n_hops : 0.0;
    double post_rtf = audio_seconds > 0.0 ? post_s / audio_seconds : 0.0;
    printf("stage=post hops=%d total_s=%.6f us_per_hop=%.2f rtf=%.6f\n",
           n_hops, post_s, post_us_per_hop, post_rtf);

    free(mic_buf);
    free(ref_buf);
    return 0;
}
