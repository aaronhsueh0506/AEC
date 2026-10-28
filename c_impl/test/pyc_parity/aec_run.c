/* aec_run.c -- C side of the Python<->C parity harness (test/pyc_parity_aec.sh).
 *
 * Feeds raw float32 mic/ref through the public aec_process() hop by hop and dumps, in files the
 * Python side writes in the SAME layout (python/diag/pyc_parity_aec_run.py), everything the
 * harness compares:
 *
 *   <pre>.out.f32    final output, hop after hop
 *   <pre>.raw.f32    linear (pre-RES) output, hop after hop
 *   <pre>.trace.txt  one line per hop: hop obs_call_counter main_call_counter converged
 *                    ctx_filter_converged epc_render_forced_remaining
 *                    (obs_call_counter is -1 when the live observer is off)
 *   <pre>.ctx.f32    only with --ctx: per hop [9 scalars][error_spec re/im][echo_spec re/im]
 *                    [res_gain][r2][comfort_noise] (spectra are zero-filled when the context
 *                    leaves them NULL). Scalars: far_power erle_factor dt_indicator divergence
 *                    saturation_level erl_estimate filter_converged res_floor_protect usable_linear
 *
 *   usage: aec_run mic.f32 ref.f32 outprefix sr fft level [--obs] [--no-hpf] [--no-res]
 *                  [--no-cng] [--ctx]
 *          aec_run --hpf-only in.f32 out.f32 sr [cutoff_hz]
 *   level: mild | balanced | aggressive | no_floor
 *
 * No algorithm lives here: it is a thin driver over aec.h. The hop loop matches the Python runner
 * (`for i in range(0, n - hop, hop)`), i.e. the same number of hops from the same samples. */
#include "aec.h"
#include "hpf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *load_f32(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "aec_run: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    float *x = (float *)malloc(bytes > 0 ? (size_t)bytes : 4u);
    if (!x || fread(x, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr, "aec_run: read %s\n", path); exit(2); }
    fclose(f);
    *n = bytes / 4;
    return x;
}

static FILE *open_out(const char *pre, const char *suffix, const char *mode) {
    char path[1024];
    snprintf(path, sizeof path, "%s%s", pre, suffix);
    FILE *f = fopen(path, mode);
    if (!f) { fprintf(stderr, "aec_run: cannot create %s\n", path); exit(2); }
    return f;
}

static int hpf_only(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: aec_run --hpf-only in.f32 out.f32 sr [cutoff_hz]\n"); return 2; }
    long n; float *x = load_f32(argv[2], &n);
    int sr = atoi(argv[4]);
    float cutoff = argc > 5 ? (float)atof(argv[5]) : 80.0f;
    /* One hop-sized call per block, like aec_process: the filter state carries across calls. */
    AecConfig cfg; aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    int hop = cfg.fft_size / 2;
    Hpf *h = hpf_create(cutoff, sr);
    if (!h) { fprintf(stderr, "aec_run: hpf_create rejected cutoff=%g sr=%d\n", (double)cutoff, sr); return 3; }
    for (long i = 0; i < n; i += hop) hpf_process(h, x + i, (int)((n - i) < hop ? (n - i) : hop));
    FILE *o = fopen(argv[3], "wb");
    if (!o) return 2;
    fwrite(x, 4, (size_t)n, o); fclose(o);
    hpf_destroy(h); free(x);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--hpf-only")) return hpf_only(argc, argv);
    if (argc < 7) {
        fprintf(stderr, "usage: aec_run mic.f32 ref.f32 outprefix sr fft level [--obs] [--no-hpf] [--no-res] [--no-cng] [--ctx]\n");
        return 2;
    }
    long nm, nr;
    float *mic = load_f32(argv[1], &nm), *ref = load_f32(argv[2], &nr);
    const char *pre = argv[3];
    int sr = atoi(argv[4]), fft = atoi(argv[5]);
    AecLevel level;
    if (!strcmp(argv[6], "mild")) level = AEC_LEVEL_MILD;
    else if (!strcmp(argv[6], "balanced")) level = AEC_LEVEL_BALANCED;
    else if (!strcmp(argv[6], "aggressive")) level = AEC_LEVEL_AGGRESSIVE;
    else if (!strcmp(argv[6], "no_floor")) level = AEC_LEVEL_NO_FLOOR;
    else { fprintf(stderr, "aec_run: unknown level %s (valid: mild|balanced|aggressive|no_floor)\n", argv[6]); return 2; }

    AecConfig cfg;
    aec_config_from_level(&cfg, level, sr);
    cfg.fft_size = fft;
    int want_ctx = 0;
    for (int i = 7; i < argc; ++i) {
        if (!strcmp(argv[i], "--obs")) cfg.live_observer_enabled = 1;
        else if (!strcmp(argv[i], "--no-hpf")) cfg.enable_highpass = 0;
        else if (!strcmp(argv[i], "--no-res")) cfg.enable_res = 0;
        else if (!strcmp(argv[i], "--no-cng")) cfg.enable_cng = 0;
        else if (!strcmp(argv[i], "--ctx")) { cfg.return_res_context = 1; want_ctx = 1; }
        else { fprintf(stderr, "aec_run: bad flag %s\n", argv[i]); return 2; }
    }
    Aec a;
    if (aec_create(&a, &cfg) != 0) { fprintf(stderr, "aec_run: aec_create failed (sr=%d fft=%d)\n", sr, fft); return 3; }
    int hop = aec_hop_size(&a);
    long n = nm < nr ? nm : nr;

    FILE *fo = open_out(pre, ".out.f32", "wb");
    FILE *fr = open_out(pre, ".raw.f32", "wb");
    FILE *ft = open_out(pre, ".trace.txt", "w");
    FILE *fc = want_ctx ? open_out(pre, ".ctx.f32", "wb") : NULL;
    float *out = (float *)malloc((size_t)hop * sizeof(float));
    float *zeros = (float *)calloc((size_t)(hop > 2 * (fft / 2 + 1) ? hop : 2 * (fft / 2 + 1)), sizeof(float));
    long h = 0;
    for (long i = 0; i + hop < n; i += hop, ++h) {
        aec_process(&a, mic + i, ref + i, out);
        fwrite(out, 4, (size_t)hop, fo);
        fwrite(a.raw_output, 4, (size_t)hop, fr);
        int ctx_conv = -1;
        if (want_ctx) {
            AecResContext c;
            aec_get_res_context(&a, &c);
            ctx_conv = c.filter_converged;
            const int K = c.n_freqs;
            const float sc[9] = { c.far_power, c.erle_factor, c.dt_indicator, c.divergence, c.saturation_level,
                                  c.erl_estimate, (float)c.filter_converged, (float)c.res_floor_protect,
                                  (float)c.usable_linear };
            fwrite(sc, 4, 9, fc);
            fwrite(c.error_spec ? (const float *)c.error_spec : zeros, 4, (size_t)(2 * K), fc);
            fwrite(c.echo_spec ? (const float *)c.echo_spec : zeros, 4, (size_t)(2 * K), fc);
            fwrite(c.res_gain ? c.res_gain : zeros, 4, (size_t)K, fc);
            fwrite(c.r2 ? c.r2 : zeros, 4, (size_t)K, fc);
            fwrite(c.comfort_noise ? c.comfort_noise : zeros, 4, (size_t)K, fc);
        }
        fprintf(ft, "%ld %ld %ld %d %d %d\n", h,
                a.has_observer ? (long)a.observer_filter.call_counter : -1L,
                (long)a.main_filter.base.call_counter, a.convergence.converged, ctx_conv,
                a.epc_render_forced_remaining);
    }
    fclose(fo); fclose(fr); fclose(ft);
    if (fc) fclose(fc);
    fprintf(stderr, "aec_run: %ld hops hop=%d sr=%d fft=%d level=%s obs=%d hpf=%d res=%d cng=%d ctx=%d\n", h, hop, sr, fft,
            argv[6], cfg.live_observer_enabled, cfg.enable_highpass, cfg.enable_res, cfg.enable_cng, want_ctx);
    aec_destroy(&a);
    free(out); free(zeros); free(mic); free(ref);
    return 0;
}
