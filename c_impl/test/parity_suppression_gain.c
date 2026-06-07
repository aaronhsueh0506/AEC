/* parity_suppression_gain.c — replay the binary golden from
 * python/diag/gen_suppression_gain_golden.py through the C SuppressionGain and
 * assert exact bit-for-bit float32 match on the gain[n_bins] output across all
 * 3 real cases (doubletalk / farend_singletalk / nearend_singletalk). WS5 gate.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I/Users/mingyu/Desktop/novatek/SE/AEC/c_impl/include \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/suppression_gain.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/src/reverb_frequency_response.c \
 *       /Users/mingyu/Desktop/novatek/SE/AEC/c_impl/test/parity_suppression_gain.c \
 *       -lm -o /tmp/p_sg
 *   python3 .../python/diag/gen_suppression_gain_golden.py /tmp/sg_golden.bin
 *   /tmp/p_sg /tmp/sg_golden.bin
 */
#include "suppression_gain.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static void cmp_f32(const float *got, const float *exp, int n,
                    int *mism, double *maxd, int *first_bin) {
    int k;
    for (k = 0; k < n; ++k) {
        if (got[k] != exp[k]) {
            double d = (double)got[k] - (double)exp[k];
            if (d < 0) d = -d;
            if (d > *maxd) *maxd = d;
            if (*first_bin < 0) *first_bin = k;
            (*mism)++;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/sg_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_cases, ci;
    int total_frames = 0, total_mism = 0;
    double max_abs_diff = 0.0;
    int worst_first_bin = -1;

    int N_BINS = 257, HOP = 160;

    /* shared static config */
    double sf_active, sf_silent, sf_latch, low_render_thr;
    float ratchet[2];            /* max_inc, max_dec_lf */
    int32_t blend_flags[2];      /* enabled, per_bin */
    float blend_f[2];            /* enr_thr, softness */
    int32_t ibands[11];
    double aud_d[6];             /* floor_power, thr_lf/mf/hf, low_render_limit, normal_render_limit */
    int32_t hf_lim[2];           /* lgb, biq */
    double dne_d[4];             /* enr_thr, enr_exit, snr_thr, lf_endpoint_hz */
    int32_t dne_i[3];            /* use_during_initial, use_unbounded, lf_endpoint_bin */
    float *t_ne_enr_tr, *t_ne_enr_su, *t_ne_emr_tr;
    float *t_no_enr_tr, *t_no_enr_su, *t_no_emr_tr;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_cases, 4)) { fprintf(stderr, "short read header\n"); return 2; }
    if (!rd(f, &sf_active, 8) || !rd(f, &sf_silent, 8) || !rd(f, &sf_latch, 8) ||
        !rd(f, &low_render_thr, 8) || !rd(f, ratchet, sizeof(ratchet)) ||
        !rd(f, blend_flags, sizeof(blend_flags)) || !rd(f, blend_f, sizeof(blend_f)) ||
        !rd(f, ibands, sizeof(ibands)) || !rd(f, aud_d, sizeof(aud_d)) ||
        !rd(f, hf_lim, sizeof(hf_lim)) || !rd(f, dne_d, sizeof(dne_d)) ||
        !rd(f, dne_i, sizeof(dne_i))) {
        fprintf(stderr, "short read config block\n"); return 2;
    }

    t_ne_enr_tr = malloc((size_t)N_BINS * sizeof(float));
    t_ne_enr_su = malloc((size_t)N_BINS * sizeof(float));
    t_ne_emr_tr = malloc((size_t)N_BINS * sizeof(float));
    t_no_enr_tr = malloc((size_t)N_BINS * sizeof(float));
    t_no_enr_su = malloc((size_t)N_BINS * sizeof(float));
    t_no_emr_tr = malloc((size_t)N_BINS * sizeof(float));
    if (!rd(f, t_ne_enr_tr, (size_t)N_BINS * 4) ||
        !rd(f, t_ne_enr_su, (size_t)N_BINS * 4) ||
        !rd(f, t_ne_emr_tr, (size_t)N_BINS * 4) ||
        !rd(f, t_no_enr_tr, (size_t)N_BINS * 4) ||
        !rd(f, t_no_enr_su, (size_t)N_BINS * 4) ||
        !rd(f, t_no_emr_tr, (size_t)N_BINS * 4)) {
        fprintf(stderr, "short read tuning tables\n"); return 2;
    }

    for (ci = 0; ci < n_cases; ++ci) {
        int n_bins, n_frames, i;
        SuppressionGainConfig cfg;
        SuppressionGainTuning tun;
        SuppressionGain sg;
        float *last_gain, *last_nearend, *last_echo, *ma_buf;
        float *nearend_s, *wres, *min_g, *max_g, *g_raw, *gain_out, *sum_s;
        float *ne, *r2, *r2u, *cn, *rb, *e_gain;

        if (!rd(f, &n_bins, 4) || !rd(f, &n_frames, 4)) {
            fprintf(stderr, "short read case %d header\n", ci); return 2;
        }
        N_BINS = n_bins;

        /* assemble config */
        cfg.n_bins = n_bins;
        cfg.sr = 16000;
        cfg.hop_size = HOP;
        cfg.last_lf_band = ibands[0];
        cfg.first_hf_band = ibands[1];
        cfg.last_lf_smoothing_band = ibands[2];
        cfg.last_permanent_lf_smoothing_band = ibands[3];
        cfg.lf_smoothing_during_initial_phase = ibands[4];
        cfg.dne_lf_end = ibands[5];
        cfg.dne_trigger_threshold_hops = ibands[6];
        cfg.dne_hold_duration_hops = ibands[7];
        cfg.nearend_smoother_n = ibands[8];
        cfg.aud_lf_end_bin = ibands[9];
        cfg.aud_mf_end_bin = ibands[10];
        cfg.floor_power = aud_d[0];
        cfg.aud_thr_lf = aud_d[1];
        cfg.aud_thr_mf = aud_d[2];
        cfg.aud_thr_hf = aud_d[3];
        cfg.low_render_limit = aud_d[4];
        cfg.normal_render_limit = aud_d[5];
        cfg.hf_lgb = hf_lim[0];
        cfg.hf_biq = hf_lim[1];
        cfg.conservative_hf = 0;
        cfg.max_inc_normal = ratchet[0];
        cfg.max_inc_nearend = ratchet[0];      /* ne == normal under ratchet */
        cfg.max_dec_lf_normal = ratchet[1];
        cfg.max_dec_lf_nearend = ratchet[1];
        cfg.floor_first_increase = 0.00001;
        cfg.low_render_threshold = low_render_thr;
        cfg.split_floor_enabled = 1;
        cfg.split_floor_far_active = sf_active;
        cfg.split_floor_far_silent = sf_silent;
        cfg.split_floor_latch_power = sf_latch;
        cfg.soft_blend_enabled = blend_flags[0];
        cfg.soft_blend_per_bin = blend_flags[1];
        cfg.soft_blend_enr_thr = blend_f[0];
        cfg.soft_blend_softness = blend_f[1];
        cfg.dne_enr_threshold = dne_d[0];
        cfg.dne_enr_exit_threshold = dne_d[1];
        cfg.dne_snr_threshold = dne_d[2];
        cfg.dne_use_during_initial_phase = dne_i[0];
        cfg.dne_use_unbounded_echo = dne_i[1];
        cfg.dne_lf_endpoint_bin = dne_i[2];
        cfg.stat_aware_ne_proxy_enabled = 0;
        cfg.stat_aware_ne_proxy_threshold = 0.10;

        tun.nearend_enr_tr = t_ne_enr_tr;
        tun.nearend_enr_su = t_ne_enr_su;
        tun.nearend_emr_tr = t_ne_emr_tr;
        tun.normal_enr_tr = t_no_enr_tr;
        tun.normal_enr_su = t_no_enr_su;
        tun.normal_emr_tr = t_no_emr_tr;

        last_gain = malloc((size_t)n_bins * sizeof(float));
        last_nearend = malloc((size_t)n_bins * sizeof(float));
        last_echo = malloc((size_t)n_bins * sizeof(float));
        ma_buf = malloc((size_t)cfg.nearend_smoother_n * n_bins * sizeof(float));
        nearend_s = malloc((size_t)n_bins * sizeof(float));
        wres = malloc((size_t)n_bins * sizeof(float));
        min_g = malloc((size_t)n_bins * sizeof(float));
        max_g = malloc((size_t)n_bins * sizeof(float));
        g_raw = malloc((size_t)n_bins * sizeof(float));
        gain_out = malloc((size_t)n_bins * sizeof(float));
        sum_s = malloc((size_t)n_bins * sizeof(float));

        ne = malloc((size_t)n_bins * sizeof(float));
        r2 = malloc((size_t)n_bins * sizeof(float));
        r2u = malloc((size_t)n_bins * sizeof(float));
        cn = malloc((size_t)n_bins * sizeof(float));
        rb = malloc((size_t)HOP * sizeof(float));
        e_gain = malloc((size_t)n_bins * sizeof(float));

        suppression_gain_init(&sg, &cfg, &tun, last_gain, last_nearend,
                              last_echo, ma_buf, nearend_s, wres, min_g, max_g,
                              g_raw, gain_out, sum_s);

        for (i = 0; i < n_frames; ++i) {
            uint8_t sat, cd, rst, ist;
            const float *got;
            if (!rd(f, ne, (size_t)n_bins * 4) || !rd(f, r2, (size_t)n_bins * 4) ||
                !rd(f, r2u, (size_t)n_bins * 4) || !rd(f, cn, (size_t)n_bins * 4) ||
                !rd(f, rb, (size_t)HOP * 4) || !rd(f, &sat, 1) || !rd(f, &cd, 1) ||
                !rd(f, &rst, 1) || !rd(f, &ist, 1) ||
                !rd(f, e_gain, (size_t)n_bins * 4)) {
                fprintf(stderr, "short read case %d frame %d\n", ci, i);
                return 2;
            }
            /* Mid-stream reset: orchestrator recreated the SuppressionGain ->
             * smoother / split-floor latch / DNE counters / last_gain cleared.
             * Replicate by re-initialising the C state (config & tuning are
             * unchanged across the reset in balanced). */
            if (rst) {
                suppression_gain_init(&sg, &cfg, &tun, last_gain, last_nearend,
                                      last_echo, ma_buf, nearend_s, wres, min_g,
                                      max_g, g_raw, gain_out, sum_s);
            }
            /* initial_state is toggled externally by the orchestrator
             * (set_initial_state on delay_change / transition); feed the
             * captured value each frame. */
            suppression_gain_set_initial_state(&sg, (int)ist);
            got = suppression_gain_get_gain(&sg, ne, r2, r2u, cn, rb,
                                            (int)cd, (int)sat);
            cmp_f32(got, e_gain, n_bins, &total_mism, &max_abs_diff,
                    &worst_first_bin);
            total_frames++;
        }

        free(last_gain); free(last_nearend); free(last_echo); free(ma_buf);
        free(nearend_s); free(wres); free(min_g); free(max_g); free(g_raw);
        free(gain_out); free(sum_s);
        free(ne); free(r2); free(r2u); free(cn); free(rb); free(e_gain);
    }
    fclose(f);
    free(t_ne_enr_tr); free(t_ne_enr_su); free(t_ne_emr_tr);
    free(t_no_enr_tr); free(t_no_enr_su); free(t_no_emr_tr);

    printf("suppression_gain parity: %d cases, %d total frames, "
           "mismatches=%d (max_abs_diff=%.3e, first_bin=%d)\n",
           n_cases, total_frames, total_mism, max_abs_diff, worst_first_bin);

    if (total_mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
