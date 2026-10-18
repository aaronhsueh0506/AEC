/* bench_rtf.c — per-module real-time-factor (RTF) benchmark harness.
 *
 * Standalone, zero-host-dependency timing harness: measures wall-clock cost
 * of aec_process() over a long synthetic (or --wav) input, with and without the AEC3 post-filter (enable_res), so
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
 *                    [--sr HZ] [--fft N] [--stage full|linear|both]
 *                    [--scene noise|bursts]
 *                    [--wav mic.wav ref.wav]
 *
 * Output (one line per stage, machine-diffable):
 *   stage=<name> hops=<n> total_s=<x.xxxxxx> us_per_hop=<x.xx> rtf=<x.xxxxxx>
 *                median_us_per_hop=<x.xx>
 *
 * total_s, us_per_hop and rtf come from the fastest pass; median_us_per_hop is
 * the median over all passes. A build with PROFILE=1 adds one line per stage,
 * "buckets_us_per_hop", with the mean of each stage-timing bucket over the
 * fastest pass.
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

/* Speech-like scene, repeating every 12 s: band-limited far-end bursts with
 * pauses, a 40-tap exponentially decaying echo path delayed 24 ms, voiced
 * near-end bursts (double talk while both are on), a level step and a
 * silence. Unlike the white-noise scene it leaves the far-end non-stationary,
 * so the adaptive update, the double-talk logic and the post-filter run their
 * active paths. Deterministic: xorshift64* seeded with a constant. */
#define SCENE_ECHO_TAPS 40

static uint64_t g_scene_rs;

static uint64_t scene_rnd(void) {
    g_scene_rs ^= g_scene_rs >> 12;
    g_scene_rs ^= g_scene_rs << 25;
    g_scene_rs ^= g_scene_rs >> 27;
    return g_scene_rs * 2685821657736338717ull;
}

static double scene_urand(void) {
    return (double)(scene_rnd() >> 11) / 9007199254740992.0;
}

static void gen_scene(float* mic, float* far, int n, int sample_rate) {
    g_scene_rs = 5ull * 0x9E3779B97F4A7C15ull + 3;
    const int delay = (int)(0.024 * sample_rate);
    const int ring = delay + SCENE_ECHO_TAPS + 8;
    double h[SCENE_ECHO_TAPS], lp1 = 0.0, lp2 = 0.0, ph = 0.0;
    float* hist = (float*)calloc((size_t)ring, sizeof(float));
    if (!hist) { fprintf(stderr, "ERROR: out of memory\n"); exit(4); }
    for (int i = 0; i < SCENE_ECHO_TAPS; ++i) h[i] = (scene_urand() * 2 - 1) * exp(-i / 9.0) * 0.5;

    int hp = 0;
    for (int i = 0; i < n; ++i) {
        double t = (double)i / sample_rate, tc = fmod(t, 12.0);
        int far_on = (tc < 4.0) || (tc >= 6.0 && tc < 10.0) || (tc >= 11.0);
        int near_on = (tc >= 4.0 && tc < 9.0);
        double farg = (tc >= 9.0 && tc < 10.0) ? 0.75 : 0.25;
        double nz = scene_urand() * 2 - 1;
        lp1 = 0.9 * lp1 + 0.1 * nz;
        lp2 = 0.6 * lp2 + 0.4 * nz;
        double x = far_on ? farg * (0.8 * lp1 * 3.0 + 0.2 * lp2) : 0.0;
        if (x > 0.95) x = 0.95;
        if (x < -0.95) x = -0.95;
        hist[hp] = (float)x;
        double echo = 0.0;
        for (int k = 0; k < SCENE_ECHO_TAPS; ++k)
            echo += h[k] * hist[((hp - delay - k) % ring + ring) % ring];
        hp = (hp + 1) % ring;
        ph += 2 * M_PI * (120 + 40 * sin(2 * M_PI * 0.3 * t)) / sample_rate;
        double sp = 0.0;
        for (int q = 1; q <= 12; ++q) sp += sin(q * ph) / q;
        double env = 0.5 + 0.5 * sin(2 * M_PI * 3.1 * t);
        double m = echo + (near_on ? 0.12 * sp * env : 0.0) + 0.002 * (scene_urand() * 2 - 1);
        if (m > 0.99) m = 0.99;
        if (m < -0.99) m = -0.99;
        mic[i] = (float)m;
        far[i] = (float)x;
    }
    free(hist);
}


/* ── CLI ────────────────────────────────────────────────────────────────── */

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  --seconds N                          synthetic input length (default 60)\n"
        "  --repeat M                            passes per stage, report min and median (default 1)\n"
        "  --preset {mild|balanced|aggressive}   default: balanced\n"
        "  --sr HZ                               sample rate of the synthetic input (default 16000)\n"
        "  --fft N                               FFT size (default: the preset's)\n"
        "  --stage {full|linear|both}            stages to run (default: both)\n"
        "  --scene {noise|bursts}                synthetic input: white noise (default) or the speech-like scene\n"
        "  --wav <mic.wav> <ref.wav>             use a recording instead of synthetic input\n"
        "  --live-observer                       A21 Stage 2 observer on (default off)\n",
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

static int cmp_double(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* Runs `repeat` full passes of aec_process() over [mic,ref) x n_hops hops
 * under `cfg`, prints the stage line, and returns the minimum total_s
 * observed (least-noise estimate); the line also carries the median over the
 * passes. With a PROFILE=1 build a second line gives the mean stage-timing
 * buckets of the fastest pass. One aec_create/aec_destroy per stage;
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
    double* times = (double*)malloc((size_t)repeat * sizeof(double));
    if (!out || !times) {
        fprintf(stderr, "ERROR: out of memory\n");
        exit(4);
    }

    double best = -1.0;
#if AEC_STAGE_TIMING
    double sum[5] = {0}, best_sum[5] = {0};
#endif
    for (int r = 0; r < repeat; ++r) {
        aec_reset(&aec);
        const float* mp = mic_buf;
        const float* rp = ref_buf;
#if AEC_STAGE_TIMING
        for (int k = 0; k < 5; ++k) sum[k] = 0.0;
#endif

        double t0 = now_s();
        for (int h = 0; h < n_hops; ++h) {
            aec_process(&aec, mp, rp, out);
#if AEC_STAGE_TIMING
            {
                AecStageTiming t;
                aec_get_last_timing(&aec, &t);
                sum[0] += t.delay_us;  sum[1] += t.frontend_us; sum[2] += t.linear_us;
                sum[3] += t.res_us;    sum[4] += t.steering_us;
            }
#endif
            mp += hop;
            rp += hop;
        }
        double dt = now_s() - t0;
        times[r] = dt;
        if (best < 0.0 || dt < best) {
            best = dt;
#if AEC_STAGE_TIMING
            for (int k = 0; k < 5; ++k) best_sum[k] = sum[k];
#endif
        }
    }

    qsort(times, (size_t)repeat, sizeof(double), cmp_double);
    double median = (repeat & 1) ? times[repeat / 2]
                                 : 0.5 * (times[repeat / 2 - 1] + times[repeat / 2]);

    free(out);
    free(times);
    aec_destroy(&aec);

    double us_per_hop = n_hops > 0 ? (best * 1e6) / (double)n_hops : 0.0;
    double median_us_per_hop = n_hops > 0 ? (median * 1e6) / (double)n_hops : 0.0;
    double rtf = audio_seconds > 0.0 ? best / audio_seconds : 0.0;
    printf("stage=%s hops=%d total_s=%.6f us_per_hop=%.2f rtf=%.6f median_us_per_hop=%.2f\n",
           name, n_hops, best, us_per_hop, rtf, median_us_per_hop);
#if AEC_STAGE_TIMING
    if (n_hops > 0)
        printf("buckets_us_per_hop stage=%s delay=%.3f frontend=%.3f linear=%.3f steering=%.3f res=%.3f\n",
               name, best_sum[0] / n_hops, best_sum[1] / n_hops, best_sum[2] / n_hops,
               best_sum[4] / n_hops, best_sum[3] / n_hops);
#endif
    return best;
}

int main(int argc, char** argv) {
    int seconds = 60;
    int repeat = 1;
    int synth_sr = 16000;
    int fft_override = 0;
    int stage_sel = 0;   /* 0 both, 1 full only, 2 linear only */
    int scene_bursts = 0;
    AecPreset preset = AEC_PRESET_BALANCED;
    int use_wav = 0;
    const char* mic_path = NULL;
    const char* ref_path = NULL;
    int live_observer = 0;   /* A21 Stage 2: real (not Python-level) RTF cost */

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
        } else if (!strcmp(arg, "--sr") && i + 1 < argc) {
            synth_sr = atoi(argv[++i]);
        } else if (!strcmp(arg, "--fft") && i + 1 < argc) {
            fft_override = atoi(argv[++i]);
        } else if (!strcmp(arg, "--stage") && i + 1 < argc) {
            const char* s = argv[++i];
            if (!strcmp(s, "full")) stage_sel = 1;
            else if (!strcmp(s, "linear")) stage_sel = 2;
            else if (strcmp(s, "both")) {
                fprintf(stderr, "ERROR: unknown stage '%s'. Valid: full|linear|both\n", s);
                return 2;
            }
        } else if (!strcmp(arg, "--scene") && i + 1 < argc) {
            const char* s = argv[++i];
            if (!strcmp(s, "bursts")) scene_bursts = 1;
            else if (!strcmp(s, "noise")) scene_bursts = 0;
            else {
                fprintf(stderr, "ERROR: unknown scene '%s'. Valid: noise|bursts\n", s);
                return 2;
            }
        } else if (!strcmp(arg, "--wav") && i + 2 < argc) {
            mic_path = argv[++i];
            ref_path = argv[++i];
            use_wav = 1;
        } else if (!strcmp(arg, "--live-observer")) {
            live_observer = 1;
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
        sample_rate = synth_sr;
        n_samples = seconds * sample_rate;
        mic_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        ref_buf = (float*)malloc((size_t)n_samples * sizeof(float));
        if (scene_bursts) gen_scene(mic_buf, ref_buf, n_samples, sample_rate);
        else gen_synthetic(mic_buf, ref_buf, n_samples, sample_rate);
    }

    AecConfig full_cfg;
    aec_config_from_preset(&full_cfg, preset, sample_rate);
    full_cfg.enable_cng = 1;   /* mirrors aec_wav's --cng path */
    if (fft_override > 0) full_cfg.fft_size = fft_override;
    full_cfg.live_observer_enabled = live_observer;

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

    double full_s = 0.0, linear_s = 0.0;
    if (stage_sel != 2)
        full_s = run_stage("full", &full_cfg, mic_buf, ref_buf, hop, n_hops, repeat, audio_seconds);

    AecConfig linear_cfg = full_cfg;
    linear_cfg.enable_res = 0;
    if (stage_sel != 1)
        linear_s = run_stage("linear", &linear_cfg, mic_buf, ref_buf, hop, n_hops, repeat, audio_seconds);

    /* Derived: post-filter cost ≈ full - linear. Not an independently timed
     * stage (see file header); can be noisy/negative on a loaded machine. */
    if (stage_sel == 0) {
        double post_s = full_s - linear_s;
        double post_us_per_hop = n_hops > 0 ? (post_s * 1e6) / (double)n_hops : 0.0;
        double post_rtf = audio_seconds > 0.0 ? post_s / audio_seconds : 0.0;
        printf("stage=post hops=%d total_s=%.6f us_per_hop=%.2f rtf=%.6f\n",
               n_hops, post_s, post_us_per_hop, post_rtf);
    }

    free(mic_buf);
    free(ref_buf);
    return 0;
}
