/* parity_aec3_post.c — replay the binary golden from
 * python/diag/gen_aec3_post_golden.py through the C AEC3 post-filter DRIVER
 * (aec3_post.c) and assert bit-for-bit float32 match on the final out[hop] AND
 * every driver-assembled intermediate (PSDs / E1 capture_psd_erle /
 * x2_reverb_for_erle / coh_gate_mask / comfort_noise / pre-irfft e_out_spec)
 * across the full DT/FS/NE run (startup + the n2_initial transient + a
 * mid-stream reset). WS5 gate.
 *
 * VERDICT: the NON-FFT stages the DRIVER OWNS (PSDs / E1 capture_psd_erle /
 * x2_reverb_for_erle / coh_gate_mask / comfort_noise — all replayed from the
 * golden or OLA-only, no signal FFT) are asserted BIT-EXACT (hard PASS/FAIL).
 * The two FFT-derived outputs — the pre-irfft e_out_spec (carries the
 * windowed-error rfft) and the final out[hop] (adds the irfft + OLA) — drift at
 * ULP scale under the KISS float32 backend (fft_wrapper.c + lib/kiss_fft/
 * kiss_fft.c): measured e_out_spec max ~1.2e-7, out max ~6e-8. They are gated
 * within POST_TOL, not bit-for-bit. (Under the retired pocketfft backend both
 * were bit-exact.) A real driver regression diverges by O(0.1), well above tol.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I<path-to-repo>/c_impl/include \
 *       -I<path-to-repo>/c_impl/lib/kiss_fft \
 *       <path-to-repo>/c_impl/src/aec3_post.c \
 *       <path-to-repo>/c_impl/src/reverb_model.c \
 *       <path-to-repo>/c_impl/src/fft_wrapper.c \
 *       <path-to-repo>/c_impl/lib/kiss_fft/kiss_fft.c \
 *       <path-to-repo>/c_impl/test/parity_aec3_post.c \
 *       -lm -o /tmp/p_post
 *   python3 .../python/diag/gen_aec3_post_golden.py /tmp/aec3_post_golden.bin
 *   /tmp/p_post /tmp/aec3_post_golden.bin
 */
#include "aec3_post.h"
#include "fft_wrapper.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* FFT-derived e_out_spec/out drift: KISS float32 vs numpy fp64, measured
 * worst ~1.2e-7. 1e-4 brackets it; a real driver regression is O(0.1). */
#define POST_TOL 1.0e-4

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

typedef struct { int mism; double maxd; int first; } Acc;

static void cmp_f32(const float *got, const float *exp, int n, Acc *a) {
    int k;
    for (k = 0; k < n; ++k) {
        if (got[k] != exp[k]) {
            double d = (double)got[k] - (double)exp[k];
            if (d < 0) d = -d;
            if (d > a->maxd) a->maxd = d;
            if (a->first < 0) a->first = k;
            a->mism++;
        }
    }
}

static void cmp_u8(const unsigned char *got, const unsigned char *exp, int n,
                   Acc *a) {
    int k;
    for (k = 0; k < n; ++k) {
        if (got[k] != exp[k]) {
            if (a->first < 0) a->first = k;
            a->mism++;
        }
    }
}

static void rd_c64(FILE *f, Complex *c, int n) {
    /* interleaved (re,im) float32 */
    int k;
    float buf2[2];
    for (k = 0; k < n; ++k) {
        if (!rd(f, buf2, 8)) { fprintf(stderr, "short c64\n"); exit(2); }
        c[k].r = buf2[0];
        c[k].i = buf2[1];
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/aec3_post_golden.bin";
    FILE *f = fopen(path, "rb");
    int hdr_i[12];
    double hdr_d[8];
    int n_bins, fft_size, block_size, hop, n_part;
    int n_cases, ci;
    float *synth_window, *sqrt2_sin_lut;
    Aec3PostConfig cfg;
    Aec3Post post;
    FftHandle *fft;

    /* driver storage */
    float *avg_rev, *y2s, *n2, *n2i, *sye_re, *sye_im, *ola;
    double *syy, *see;
    float *np_, *fp_, *ep_, *erp_, *cpe_c, *x2r_c, *cn_c, *nf_;
    unsigned char *cgm_c;
    Complex *eout;
    float *eout_full;

    /* per-hop input buffers */
    Complex *near_spec, *echo_coh, *error_spec, *sel_echo;
    float *abs_near, *abs_far, *abs_sel_echo, *abs_error, *abs_echo_coh;
    float *abs_nsw_e1, *abs_ybase;
    float *x2_at_delay, *x2_past, *far_end, *gain, *out_c;
    /* expected golden */
    float *e_near_psd, *e_far_psd, *e_echo_psd, *e_error_psd, *e_cpe;
    float *e_x2r, *e_cn, *e_out;
    unsigned char *e_cgm;
    Complex *e_eout;

    Acc a_out = {0, 0.0, -1};
    Acc a_psd = {0, 0.0, -1};
    Acc a_cpe = {0, 0.0, -1};
    Acc a_x2r = {0, 0.0, -1};
    Acc a_cn = {0, 0.0, -1};
    Acc a_cgm = {0, 0.0, -1};
    Acc a_eout = {0, 0.0, -1};
    int total_hops = 0, total_resets = 0;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, hdr_i, sizeof(int) * 12)) { fprintf(stderr, "hdr i\n"); return 2; }
    n_bins = hdr_i[0]; fft_size = hdr_i[1]; block_size = hdr_i[2];
    hop = hdr_i[3]; n_part = hdr_i[4];
    (void)n_part;
    aec3_post_config_defaults(&cfg);
    cfg.n_bins = n_bins; cfg.fft_size = fft_size;
    cfg.block_size = block_size; cfg.hop_size = hop;
    cfg.erle_coh_gate_enabled = hdr_i[5];
    cfg.erle_windowed_capture_psd = hdr_i[6];
    cfg.erle_render_x2_psd_scale = hdr_i[7];
    cfg.output_capture_when_linear_unusable = hdr_i[8];
    cfg.enable_cng = hdr_i[9];
    cfg.cng_n2_update_onset_hops = hdr_i[10];
    cfg.cng_n2_initial_duration_hops = hdr_i[11];
    if (!rd(f, hdr_d, sizeof(double) * 8)) { fprintf(stderr, "hdr d\n"); return 2; }
    cfg.cng_y2_alpha = hdr_d[0];
    cfg.cng_n2_track_freshness = hdr_d[1];
    cfg.cng_n2_track_retention = hdr_d[2];
    cfg.cng_n2_slow_up = hdr_d[3];
    cfg.cng_n2_initial_alpha = hdr_d[4];
    cfg.noise_floor_int16sq = hdr_d[5];
    cfg.erle_coh_gate_alpha = hdr_d[6];
    cfg.erle_coh_gate_threshold = hdr_d[7];

    synth_window = malloc((size_t)block_size * sizeof(float));
    sqrt2_sin_lut = malloc(32 * sizeof(float));
    if (!rd(f, synth_window, (size_t)block_size * 4) ||
        !rd(f, sqrt2_sin_lut, 32 * 4)) { fprintf(stderr, "win/lut\n"); return 2; }
    if (!rd(f, &n_cases, 4)) { fprintf(stderr, "n_cases\n"); return 2; }

    fft = fft_create(fft_size);
    if (!fft) { fprintf(stderr, "fft\n"); return 2; }

#define A(p, n) ((p) = malloc((size_t)(n) * sizeof(*(p))))
    A(avg_rev, n_bins); A(y2s, n_bins); A(n2, n_bins); A(n2i, n_bins);
    A(sye_re, n_bins); A(sye_im, n_bins); A(ola, block_size);
    A(syy, n_bins); A(see, n_bins);
    A(np_, n_bins); A(fp_, n_bins); A(ep_, n_bins); A(erp_, n_bins);
    A(cpe_c, n_bins); A(x2r_c, n_bins); A(cn_c, n_bins); A(nf_, n_bins);
    A(cgm_c, n_bins);
    A(eout, n_bins); A(eout_full, fft_size);

    A(near_spec, n_bins); A(echo_coh, n_bins); A(error_spec, n_bins);
    A(sel_echo, n_bins);
    A(abs_near, n_bins); A(abs_far, n_bins); A(abs_sel_echo, n_bins);
    A(abs_error, n_bins); A(abs_echo_coh, n_bins); A(abs_nsw_e1, n_bins);
    A(abs_ybase, n_bins);
    A(x2_at_delay, n_bins); A(x2_past, n_bins); A(far_end, hop);
    A(gain, n_bins); A(out_c, hop);
    A(e_near_psd, n_bins); A(e_far_psd, n_bins); A(e_echo_psd, n_bins);
    A(e_error_psd, n_bins); A(e_cpe, n_bins); A(e_x2r, n_bins);
    A(e_cn, n_bins); A(e_out, hop); A(e_cgm, n_bins); A(e_eout, n_bins);
#undef A

    aec3_post_init(&post, &cfg, fft, synth_window, sqrt2_sin_lut, avg_rev,
                   y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                   np_, fp_, ep_, erp_, cpe_c, x2r_c, cgm_c, cn_c, nf_,
                   eout, eout_full);

    for (ci = 0; ci < n_cases; ++ci) {
        int n_frames, i;
        if (!rd(f, &n_frames, 4)) { fprintf(stderr, "n_frames\n"); return 2; }
        /* Each new case is a fresh AEC instance — re-init the driver state. */
        aec3_post_reset(&post);

        for (i = 0; i < n_frames; ++i) {
            int reset_before, x2_present, x2r_present, cpe_present, cgm_present;
            int sat_level_gt, usable, sat_echo;
            int hdr3[3];
            double decay_steady;
            Aec3PostAbs mag;

            if (!rd(f, &reset_before, 4)) { fprintf(stderr, "rb\n"); return 2; }
            rd_c64(f, near_spec, n_bins);
            rd_c64(f, echo_coh, n_bins);
            rd_c64(f, error_spec, n_bins);
            rd_c64(f, sel_echo, n_bins);
            if (!rd(f, abs_near, (size_t)n_bins * 4) ||
                !rd(f, abs_far, (size_t)n_bins * 4) ||
                !rd(f, abs_sel_echo, (size_t)n_bins * 4) ||
                !rd(f, abs_error, (size_t)n_bins * 4) ||
                !rd(f, abs_echo_coh, (size_t)n_bins * 4) ||
                !rd(f, abs_nsw_e1, (size_t)n_bins * 4) ||
                !rd(f, abs_ybase, (size_t)n_bins * 4)) {
                fprintf(stderr, "abs\n"); return 2;
            }
            if (!rd(f, &x2_present, 4)) { fprintf(stderr, "x2p\n"); return 2; }
            if (!rd(f, x2_at_delay, (size_t)n_bins * 4) ||
                !rd(f, x2_past, (size_t)n_bins * 4)) {
                fprintf(stderr, "x2\n"); return 2;
            }
            if (!rd(f, &decay_steady, 8)) { fprintf(stderr, "decay\n"); return 2; }
            if (!rd(f, hdr3, sizeof(int) * 3)) { fprintf(stderr, "s3\n"); return 2; }
            sat_level_gt = hdr3[0]; usable = hdr3[1]; sat_echo = hdr3[2];
            (void)sat_echo;
            if (!rd(f, gain, (size_t)n_bins * 4)) { fprintf(stderr, "g\n"); return 2; }
            if (!rd(f, e_near_psd, (size_t)n_bins * 4) ||
                !rd(f, e_far_psd, (size_t)n_bins * 4) ||
                !rd(f, e_echo_psd, (size_t)n_bins * 4) ||
                !rd(f, e_error_psd, (size_t)n_bins * 4) ||
                !rd(f, e_cpe, (size_t)n_bins * 4)) {
                fprintf(stderr, "psd-exp\n"); return 2;
            }
            if (!rd(f, &x2r_present, 4)) { fprintf(stderr, "x2rp\n"); return 2; }
            if (!rd(f, e_x2r, (size_t)n_bins * 4)) { fprintf(stderr, "x2r\n"); return 2; }
            if (!rd(f, &cpe_present, 4)) { fprintf(stderr, "cpep\n"); return 2; }
            if (!rd(f, &cgm_present, 4)) { fprintf(stderr, "cgmp\n"); return 2; }
            if (!rd(f, e_cgm, (size_t)n_bins)) { fprintf(stderr, "cgm\n"); return 2; }
            if (!rd(f, e_cn, (size_t)n_bins * 4)) { fprintf(stderr, "cn\n"); return 2; }
            rd_c64(f, e_eout, n_bins);
            if (!rd(f, e_out, (size_t)hop * 4)) { fprintf(stderr, "out\n"); return 2; }

            /* far_end is not stored in the golden (the output path ignores it). */
            (void)far_end;

            if (reset_before) {
                aec3_post_reset(&post);
                total_resets++;
            }

            mag.abs_near = abs_near;
            mag.abs_far = abs_far;
            mag.abs_sel_echo = abs_sel_echo;
            mag.abs_error = abs_error;
            mag.abs_echo_coh = abs_echo_coh;
            mag.abs_nsw_e1 = abs_nsw_e1;
            mag.abs_ybase = abs_ybase;

            aec3_post_process(&post, near_spec, error_spec, sel_echo, echo_coh,
                              &mag, x2_present, x2_at_delay, x2_past,
                              decay_steady, NULL,
                              sat_level_gt, usable, gain, out_c);

            /* per-stage assertions */
            cmp_f32(post.near_psd, e_near_psd, n_bins, &a_psd);
            cmp_f32(post.far_psd, e_far_psd, n_bins, &a_psd);
            cmp_f32(post.echo_psd, e_echo_psd, n_bins, &a_psd);
            cmp_f32(post.error_psd, e_error_psd, n_bins, &a_psd);
            if (cpe_present)
                cmp_f32(post.capture_psd_erle, e_cpe, n_bins, &a_cpe);
            if (x2r_present)
                cmp_f32(post.x2_reverb_for_erle, e_x2r, n_bins, &a_x2r);
            if (cgm_present)
                cmp_u8(post.coh_gate_mask, e_cgm, n_bins, &a_cgm);
            cmp_f32(post.comfort_noise, e_cn, n_bins, &a_cn);
            {
                /* compare post-CNG pre-irfft e_out_spec (interleaved). */
                int kk;
                for (kk = 0; kk < n_bins; ++kk) {
                    if (post.e_out_spec[kk].r != e_eout[kk].r) {
                        double dd = (double)post.e_out_spec[kk].r - e_eout[kk].r;
                        if (dd < 0) dd = -dd;
                        if (dd > a_eout.maxd) a_eout.maxd = dd;
                        if (a_eout.first < 0) a_eout.first = kk;
                        a_eout.mism++;
                    }
                    if (post.e_out_spec[kk].i != e_eout[kk].i) {
                        double dd = (double)post.e_out_spec[kk].i - e_eout[kk].i;
                        if (dd < 0) dd = -dd;
                        if (dd > a_eout.maxd) a_eout.maxd = dd;
                        if (a_eout.first < 0) a_eout.first = kk;
                        a_eout.mism++;
                    }
                }
            }
#ifdef DUMP_OUT
            {
                int kk, hopmis = 0;
                for (kk = 0; kk < hop; ++kk)
                    if (out_c[kk] != e_out[kk]) hopmis++;
                if (hopmis)
                    fprintf(stderr, "hop %d: %d mismatched samples\n",
                            total_hops, hopmis);
            }
#endif
            cmp_f32(out_c, e_out, hop, &a_out);
            total_hops++;
        }
    }
    fclose(f);

    printf("aec3_post DRIVER parity: %d cases, %d hops, %d resets\n",
           n_cases, total_hops, total_resets);
    printf("  PSD (near/far/echo/error) mism=%d (max=%.3e first_bin=%d)\n",
           a_psd.mism, a_psd.maxd, a_psd.first);
    printf("  capture_psd_erle (E1)     mism=%d (max=%.3e)\n",
           a_cpe.mism, a_cpe.maxd);
    printf("  x2_reverb_for_erle        mism=%d (max=%.3e)\n",
           a_x2r.mism, a_x2r.maxd);
    printf("  coh_gate_mask             mism=%d (first_bin=%d)\n",
           a_cgm.mism, a_cgm.first);
    printf("  comfort_noise             mism=%d (max=%.3e)\n",
           a_cn.mism, a_cn.maxd);
    printf("  e_out_spec (pre-irfft)    mism=%d (max=%.3e first_bin=%d)\n",
           a_eout.mism, a_eout.maxd, a_eout.first);
    printf("  >>> out[hop]              mism=%d (max=%.3e first=%d)\n",
           a_out.mism, a_out.maxd, a_out.first);

    /* DRIVER verdict: the NON-FFT stages the driver owns must be bit-exact (they
     * are replayed from the golden / OLA-only, no signal FFT). e_out_spec is
     * EXCLUDED here — it carries the windowed-error rfft and is gated by
     * tolerance below. */
    {
        int driver = a_psd.mism + a_cpe.mism + a_x2r.mism + a_cgm.mism + a_cn.mism;
        if (driver) {
            printf(">>> FAIL (non-FFT driver stage not bit-exact)\n");
            return 1;
        }
    }
    /* e_out_spec (pre-irfft) and out[hop] (post-irfft + OLA) carry the float32
     * FFT, so they drift at ULP scale under KISS (measured e_out_spec ~1.2e-7,
     * out ~6e-8). Gate within POST_TOL; a real regression diverges by O(0.1). */
    if (a_eout.maxd < POST_TOL && a_out.maxd < POST_TOL) {
        printf(">>> PASS (non-FFT driver bit-exact; e_out_spec/out within %.0e: "
               "e_out_spec max=%.3e out max=%.3e)\n",
               (double)POST_TOL, a_eout.maxd, a_out.maxd);
        return 0;
    }
    printf(">>> FAIL (e_out_spec/out exceed float32 FFT tolerance %.0e)\n",
           (double)POST_TOL);
    return 1;
}
