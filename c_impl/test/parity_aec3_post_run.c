/* parity_aec3_post_run.c — replay the binary golden from
 * python/diag/gen_aec3_post_run_golden.py through the FULL-orchestration C
 * driver (aec3_post_run) and assert out[hop] BIT-EXACT (0 mismatches) across
 * the whole doubletalk case (~4186 hops). Also asserts usable_linear per hop
 * for localisation.
 *
 * This is the integration-core gate: aec3_post_run drives AecState +
 * ResidualEchoEstimator + SuppressionGain + StationarityEstimator +
 * LinearFilterSelect itself (gain / usable_linear / r2 are NOT injected), so a
 * mismatch in ANY sub-module call sequence shows up here.
 *
 * Build (standalone):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
 *       -I.../c_impl/include -I.../c_impl/lib/pocketfft \
 *       .../c_impl/src/aec3_post.c .../reverb_model.c \
 *       .../reverb_frequency_response.c .../aec3_scale.c .../freq_utils.c \
 *       .../aec_state.c .../erle_estimator.c .../subband_erle.c \
 *       .../fullband_erle.c .../erl_estimator.c .../filter_quality.c \
 *       .../saturation_detector.c .../filter_delay.c .../initial_state.c \
 *       .../filter_analyzer.c .../residual_echo_estimator.c \
 *       .../suppression_gain.c .../stationarity_estimator.c \
 *       .../linear_filter_output.c .../filter_state_bridge.c \
 *       .../fft_pocketfft.c .../lib/pocketfft/pocketfft.c \
 *       .../test/parity_aec3_post_run.c -lm -o /tmp/p_post_run
 *   python3 .../python/diag/gen_aec3_post_run_golden.py /tmp/aec3_post_run_golden.bin
 *   /tmp/p_post_run /tmp/aec3_post_run_golden.bin
 */
#include "aec3_post.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static void rd_c64(FILE *f, Complex *c, int n) {
    int k; float b2[2];
    for (k = 0; k < n; ++k) {
        if (!rd(f, b2, 8)) { fprintf(stderr, "short c64\n"); exit(2); }
        c[k].r = b2[0]; c[k].i = b2[1];
    }
}

/* numpy abs(c64) (same helper the driver exposes) for the |far|² stationarity
 * feed (the upstream process loop computes np.abs(far_spec)**2). */
static float cmag2(float re, float im) {
    float m = cabs_np(re, im);
    return m * m;
}

#define A(p, n) ((p) = malloc((size_t)(n) * sizeof(*(p))))

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/aec3_post_run_golden.bin";
    FILE *f = fopen(path, "rb");
    int hdr_g[5];
    int n_bins, fft_size, block_size, hop, n_part;
    int flag_i[6];
    int cnt_i[4];
    double cng_d[9];
    float *synth_window, *sqrt2_lut;
    int st_i[10];
    double st_d[4];
    int ree_i[11];
    double ree_d[13];
    double sg_d4[5];
    float sg_ratchet[2];
    int sg_blend_i[2];
    float sg_blend_f[2];
    int sg_bands[11];
    double sg_aud[6];
    int sg_hf[4];
    double sg_dne_d[3];
    int sg_dne_i[3];
    int n_hops, hi;

    Aec3PostConfig cfg;
    Aec3Post post;
    FftHandle *fft;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, hdr_g, sizeof(int) * 5)) { fprintf(stderr, "hdr_g\n"); return 2; }
    n_bins = hdr_g[0]; fft_size = hdr_g[1]; block_size = hdr_g[2];
    hop = hdr_g[3]; n_part = hdr_g[4];
    if (!rd(f, flag_i, sizeof(int) * 6)) { fprintf(stderr, "flags\n"); return 2; }
    if (!rd(f, cnt_i, sizeof(int) * 4)) { fprintf(stderr, "cnts\n"); return 2; }
    if (!rd(f, cng_d, sizeof(double) * 9)) { fprintf(stderr, "cng_d\n"); return 2; }

    A(synth_window, block_size); A(sqrt2_lut, 32);
    if (!rd(f, synth_window, (size_t)block_size * 4) ||
        !rd(f, sqrt2_lut, 32 * 4)) { fprintf(stderr, "win/lut\n"); return 2; }

    if (!rd(f, st_i, sizeof(int) * 10)) { fprintf(stderr, "st_i\n"); return 2; }
    if (!rd(f, st_d, sizeof(double) * 4)) { fprintf(stderr, "st_d\n"); return 2; }
    if (!rd(f, ree_i, sizeof(int) * 11)) { fprintf(stderr, "ree_i\n"); return 2; }
    if (!rd(f, ree_d, sizeof(double) * 13)) { fprintf(stderr, "ree_d\n"); return 2; }

    if (!rd(f, sg_d4, sizeof(double) * 5)) { fprintf(stderr, "sg_d4\n"); return 2; }
    if (!rd(f, sg_ratchet, sizeof(sg_ratchet))) { fprintf(stderr, "sg_r\n"); return 2; }
    if (!rd(f, sg_blend_i, sizeof(sg_blend_i))) { fprintf(stderr, "sg_bi\n"); return 2; }
    if (!rd(f, sg_blend_f, sizeof(sg_blend_f))) { fprintf(stderr, "sg_bf\n"); return 2; }
    if (!rd(f, sg_bands, sizeof(sg_bands))) { fprintf(stderr, "sg_b\n"); return 2; }
    if (!rd(f, sg_aud, sizeof(sg_aud))) { fprintf(stderr, "sg_a\n"); return 2; }
    if (!rd(f, sg_hf, sizeof(sg_hf))) { fprintf(stderr, "sg_hf\n"); return 2; }
    if (!rd(f, sg_dne_d, sizeof(sg_dne_d))) { fprintf(stderr, "sg_dd\n"); return 2; }
    if (!rd(f, sg_dne_i, sizeof(sg_dne_i))) { fprintf(stderr, "sg_di\n"); return 2; }

    float *t_ne_enr_tr, *t_ne_enr_su, *t_ne_emr_tr;
    float *t_no_enr_tr, *t_no_enr_su, *t_no_emr_tr;
    A(t_ne_enr_tr, n_bins); A(t_ne_enr_su, n_bins); A(t_ne_emr_tr, n_bins);
    A(t_no_enr_tr, n_bins); A(t_no_enr_su, n_bins); A(t_no_emr_tr, n_bins);
    if (!rd(f, t_ne_enr_tr, (size_t)n_bins * 4) ||
        !rd(f, t_ne_enr_su, (size_t)n_bins * 4) ||
        !rd(f, t_ne_emr_tr, (size_t)n_bins * 4) ||
        !rd(f, t_no_enr_tr, (size_t)n_bins * 4) ||
        !rd(f, t_no_enr_su, (size_t)n_bins * 4) ||
        !rd(f, t_no_emr_tr, (size_t)n_bins * 4)) { fprintf(stderr, "tun\n"); return 2; }

    /* ── Aec3Post driver ─────────────────────────────────────────────────── */
    fft = fft_create(fft_size);
    if (!fft) { fprintf(stderr, "fft\n"); return 2; }
    aec3_post_config_defaults(&cfg);
    cfg.n_bins = n_bins; cfg.fft_size = fft_size;
    cfg.block_size = block_size; cfg.hop_size = hop;
    cfg.erle_coh_gate_enabled = flag_i[0];
    cfg.erle_windowed_capture_psd = flag_i[1];
    cfg.erle_render_x2_psd_scale = flag_i[2];
    cfg.output_capture_when_linear_unusable = flag_i[3];
    cfg.enable_cng = flag_i[4];
    cfg.cng_n2_update_onset_hops = cnt_i[0];
    cfg.cng_n2_initial_duration_hops = cnt_i[1];
    cfg.cng_y2_alpha = cng_d[0];
    cfg.cng_n2_track_freshness = cng_d[1];
    cfg.cng_n2_track_retention = cng_d[2];
    cfg.cng_n2_slow_up = cng_d[3];
    cfg.cng_n2_initial_alpha = cng_d[4];
    cfg.noise_floor_int16sq = cng_d[5];
    cfg.erle_coh_gate_alpha = cng_d[6];
    cfg.erle_coh_gate_threshold = cng_d[7];

    float *avg_rev, *y2s, *n2, *n2i, *sye_re, *sye_im, *ola;
    double *syy, *see;
    float *np_, *fp_, *ep_, *erp_, *cpe_c, *x2r_c, *cn_c, *nf_;
    unsigned char *cgm_c;
    Complex *eout; float *eout_full;
    A(avg_rev, n_bins); A(y2s, n_bins); A(n2, n_bins); A(n2i, n_bins);
    A(sye_re, n_bins); A(sye_im, n_bins); A(ola, block_size);
    A(syy, n_bins); A(see, n_bins);
    A(np_, n_bins); A(fp_, n_bins); A(ep_, n_bins); A(erp_, n_bins);
    A(cpe_c, n_bins); A(x2r_c, n_bins); A(cn_c, n_bins); A(nf_, n_bins);
    A(cgm_c, n_bins); A(eout, n_bins); A(eout_full, fft_size);
    aec3_post_init(&post, &cfg, fft, synth_window, sqrt2_lut, avg_rev,
                   y2s, n2, n2i, sye_re, sye_im, syy, see, ola,
                   np_, fp_, ep_, erp_, cpe_c, x2r_c, cgm_c, cn_c, nf_,
                   eout, eout_full);

    /* ── AecState ────────────────────────────────────────────────────────── */
    AecStateConfig acfg;
    aec_state_config_defaults(&acfg);
    acfg.n_bins = st_i[0];
    acfg.num_capture_channels = st_i[1];
    acfg.hop_size = st_i[2];
    acfg.enable_filter_analyzer = st_i[3];
    acfg.erle_startup_hops = st_i[4];
    acfg.erl_startup_hops = st_i[5];
    acfg.echo_can_saturate = st_i[6];
    acfg.use_linear_filter = st_i[7];
    acfg.conservative_initial_phase = st_i[8];
    acfg.delay_headroom_samples = st_i[9];
    acfg.initial_state_seconds = st_d[0];
    acfg.erle_min = st_d[1];
    acfg.erle_max_l = st_d[2];
    acfg.erle_max_h = st_d[3];
    acfg.filter_taps_size = cnt_i[3] > 0 ? cnt_i[3] : 960;

    AecStateStorage ast;
    ast.erle_max = malloc((size_t)n_bins * sizeof(float));
    ast.erle = malloc((size_t)n_bins * sizeof(float));
    ast.erle_oc = malloc((size_t)n_bins * sizeof(float));
    ast.erle_unb = malloc((size_t)n_bins * sizeof(float));
    ast.erle_during = malloc((size_t)n_bins * sizeof(float));
    ast.erle_coming_onset = malloc((size_t)n_bins);
    ast.erle_hold = malloc((size_t)n_bins * sizeof(int32_t));
    ast.erle_y2_acc = malloc((size_t)n_bins * sizeof(float));
    ast.erle_e2_acc = malloc((size_t)n_bins * sizeof(float));
    ast.erle_low_render = malloc((size_t)n_bins);
    ast.erl = malloc((size_t)n_bins * sizeof(float));
    ast.erl_hold = malloc((size_t)(n_bins - 2) * sizeof(int));
    ast.filter_delays_blocks = malloc((size_t)acfg.num_capture_channels * sizeof(int));
    ast.fa_h_highpass = malloc((size_t)acfg.filter_taps_size * sizeof(float));
    AecState state;
    aec_state_init(&state, &acfg, &ast);

    /* ── ResidualEchoEstimator ───────────────────────────────────────────── */
    ReeEchoModelConfig em;
    em.min_noise_floor_power = ree_d[0];
    em.noise_gate_power = ree_d[1];
    em.noise_gate_slope = ree_d[2];
    em.stationary_gate_slope = ree_d[3];
    em.model_reverb_in_nonlinear_mode = ree_i[1];
    int ree_hop = ree_i[0];
    int rh = (ree_i[5] ? 1 : 0) + 1 + 1;   /* use_aec3_echo_gen_window pre + post + 1 */
    float *x2_nf = malloc((size_t)n_bins * sizeof(float));
    int *x2_nf_ctr = malloc((size_t)n_bins * sizeof(int));
    float *rm_st = malloc((size_t)n_bins * sizeof(float));
    float *rt_st = malloc((size_t)n_bins * sizeof(float));
    float *rh_st = malloc((size_t)rh * (size_t)n_bins * sizeof(float));
    float *drd_st = malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)n_bins * sizeof(float));
    float *rrd_st = malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)n_bins * sizeof(float));
    float *ld = malloc((size_t)n_bins * sizeof(float));
    float *lr = malloc((size_t)n_bins * sizeof(float));
    float *ree_scratch = malloc((size_t)n_bins * sizeof(float));
    ResidualEchoEstimator ree;
    ree_init(&ree, n_bins, ree_hop, &em, ree_d[4], ree_d[5], ree_i[2],
             ree_d[6], ree_d[7], ree_i[3], ree_d[8],
             ree_i[4], ree_i[10], ree_i[5], ree_i[6], ree_d[9], ree_d[10],
             ree_d[11], ree_i[7], ree_i[8], ree_i[9], ree_d[12],
             x2_nf, x2_nf_ctr, rm_st, rt_st, rh_st, drd_st, rrd_st,
             ld, lr, ree_scratch);

    /* ── SuppressionGain ─────────────────────────────────────────────────── */
    SuppressionGainConfig scfg;
    SuppressionGainTuning stun;
    memset(&scfg, 0, sizeof(scfg));
    scfg.n_bins = n_bins; scfg.sr = 16000; scfg.hop_size = hop;
    scfg.last_lf_band = sg_bands[0];
    scfg.first_hf_band = sg_bands[1];
    scfg.last_lf_smoothing_band = sg_bands[2];
    scfg.last_permanent_lf_smoothing_band = sg_bands[3];
    scfg.lf_smoothing_during_initial_phase = sg_bands[4];
    scfg.dne_lf_end = sg_bands[5];
    scfg.dne_trigger_threshold_hops = sg_bands[6];
    scfg.dne_hold_duration_hops = sg_bands[7];
    scfg.nearend_smoother_n = sg_bands[8];
    scfg.aud_lf_end_bin = sg_bands[9];
    scfg.aud_mf_end_bin = sg_bands[10];
    scfg.floor_power = sg_aud[0];
    scfg.aud_thr_lf = sg_aud[1];
    scfg.aud_thr_mf = sg_aud[2];
    scfg.aud_thr_hf = sg_aud[3];
    scfg.low_render_limit = sg_aud[4];
    scfg.normal_render_limit = sg_aud[5];
    scfg.hf_lgb = sg_hf[0];
    scfg.hf_biq = sg_hf[1];
    scfg.conservative_hf = sg_hf[2];
    scfg.lf_clamp_bin = sg_hf[3];   /* INFRA GAP fix: memset left this 0 →
                                     * LF gain clamp disabled → bin 6 (and 0-7)
                                     * unclamped → out divergence from hop 82. */
    scfg.max_inc_normal = sg_ratchet[0];
    scfg.max_inc_nearend = sg_ratchet[0];
    scfg.max_dec_lf_normal = sg_ratchet[1];
    scfg.max_dec_lf_nearend = sg_ratchet[1];
    scfg.floor_first_increase = 0.00001;
    scfg.low_render_threshold = sg_d4[3];
    scfg.split_floor_enabled = 1;
    scfg.split_floor_far_active = sg_d4[0];
    scfg.split_floor_far_silent = sg_d4[1];
    scfg.split_floor_latch_power = sg_d4[2];
    scfg.split_floor_dt = sg_d4[4];   /* INFRA GAP fix: memset left this 0 →
                                       * DT-protect floor wrong when the per-hop
                                       * dt_protect_active flag fires (~hop 130). */
    scfg.soft_blend_enabled = sg_blend_i[0];
    scfg.soft_blend_per_bin = sg_blend_i[1];
    scfg.soft_blend_enr_thr = sg_blend_f[0];
    scfg.soft_blend_softness = sg_blend_f[1];
    scfg.dne_enr_threshold = sg_dne_d[0];
    scfg.dne_enr_exit_threshold = sg_dne_d[1];
    scfg.dne_snr_threshold = sg_dne_d[2];
    scfg.dne_use_during_initial_phase = sg_dne_i[0];
    scfg.dne_use_unbounded_echo = sg_dne_i[1];
    scfg.dne_lf_endpoint_bin = sg_dne_i[2];
    scfg.stat_aware_ne_proxy_enabled = 0; scfg.stat_aware_ne_proxy_threshold = 0.10;
    stun.nearend_enr_tr = t_ne_enr_tr; stun.nearend_enr_su = t_ne_enr_su;
    stun.nearend_emr_tr = t_ne_emr_tr; stun.normal_enr_tr = t_no_enr_tr;
    stun.normal_enr_su = t_no_enr_su; stun.normal_emr_tr = t_no_emr_tr;

    SuppressionGain sg;
    float *sg_last_gain = malloc((size_t)n_bins * sizeof(float));
    float *sg_last_ne = malloc((size_t)n_bins * sizeof(float));
    float *sg_last_echo = malloc((size_t)n_bins * sizeof(float));
    float *sg_ma = malloc((size_t)scfg.nearend_smoother_n * n_bins * sizeof(float));
    float *sg_ne = malloc((size_t)n_bins * sizeof(float));
    float *sg_wr = malloc((size_t)n_bins * sizeof(float));
    float *sg_ming = malloc((size_t)n_bins * sizeof(float));
    float *sg_maxg = malloc((size_t)n_bins * sizeof(float));
    float *sg_graw = malloc((size_t)n_bins * sizeof(float));
    float *sg_gout = malloc((size_t)n_bins * sizeof(float));
    float *sg_sum = malloc((size_t)n_bins * sizeof(float));
    suppression_gain_init(&sg, &scfg, &stun, sg_last_gain, sg_last_ne,
                          sg_last_echo, sg_ma, sg_ne, sg_wr, sg_ming, sg_maxg,
                          sg_graw, sg_gout, sg_sum);

    /* ── StationarityEstimator ───────────────────────────────────────────── */
    StationarityEstimator stat;
    int win_hops = 5;   /* blocks_to_hops(13,160,16000); init() recomputes */
    float *stat_noise = malloc((size_t)n_bins * sizeof(float));
    int32_t *stat_hang = malloc((size_t)n_bins * sizeof(int32_t));
    unsigned char *stat_flags = malloc((size_t)n_bins);
    /* history needs window_hops × n_freqs; init() computes window_hops, but we
     * need to size the buffer beforehand. window_hops = blocks_to_hops(13,...) =
     * 5 at 16k/160. Allocate generously (16 rows) to be safe. */
    (void)win_hops;
    float *stat_hist = malloc((size_t)16 * (size_t)n_bins * sizeof(float));
    stationarity_estimator_init(&stat, n_bins, hop, 16000,
                                stat_noise, stat_hang, stat_flags, stat_hist);

    /* ── LinearFilterSelect ──────────────────────────────────────────────── */
    LinearFilterSelect lfs;
    if (linear_filter_select_init(&lfs, hop, block_size, fft_size, n_bins) != 0) {
        fprintf(stderr, "lfs init\n"); return 2;
    }

    Aec3PostRunObj obj;
    obj.state = &state; obj.ree = &ree; obj.sg = &sg;
    obj.stationarity = &stat; obj.lfs = &lfs; obj.fft = fft;

    /* ── run-scratch ─────────────────────────────────────────────────────── */
    Aec3PostRunScratch sc;
    A(sc.sel_esw, n_bins); A(sc.sel_echo, n_bins);
    A(sc.nsw_e1, n_bins); A(sc.ybase, n_bins);
    A(sc.abs_near, n_bins); A(sc.abs_far, n_bins); A(sc.abs_sel_echo, n_bins);
    A(sc.abs_error, n_bins); A(sc.abs_echo_coh, n_bins); A(sc.abs_nsw_e1, n_bins);
    A(sc.abs_ybase, n_bins);
    A(sc.x2_at_delay, n_bins); A(sc.x2_past, n_bins);
    sc.w_mag2 = malloc((size_t)n_part * (size_t)n_bins * sizeof(float));
    A(sc.render_block_scaled, hop);
    A(sc.bridge_taps, fft_size);
    A(sc.r2, n_bins); A(sc.r2_unb, n_bins); A(sc.nearend_pwr, n_bins);
    A(sc.stat_mask, n_bins);

    /* ── per-hop buffers ─────────────────────────────────────────────────── */
    Complex *near_spec, *far_spec, *echo_spec, *esw, *W0, *W_all, *X_buf;
    Complex *sh_err;
    float *sqrt_hann, *tdf, *lso, *raw_output, *near_end, *far_end;
    float *out_c, *e_out;
    A(near_spec, n_bins); A(far_spec, n_bins); A(echo_spec, n_bins);
    A(esw, n_bins); A(W0, n_bins);
    W_all = malloc((size_t)n_part * (size_t)n_bins * sizeof(Complex));
    X_buf = malloc((size_t)n_part * (size_t)n_bins * sizeof(Complex));
    A(sh_err, n_bins);
    A(sqrt_hann, block_size); A(lso, hop);
    tdf = malloc((size_t)(cnt_i[3] > 0 ? cnt_i[3] : 1) * sizeof(float));
    A(raw_output, hop); A(near_end, hop); A(far_end, hop);
    A(out_c, hop); A(e_out, hop);
    float *far_psd_st = malloc((size_t)n_bins * sizeof(float));

    if (!rd(f, &n_hops, 4)) { fprintf(stderr, "n_hops\n"); return 2; }

    long out_mism = 0, usable_mism = 0;
    double out_maxd = 0.0;
    int first_out_hop = -1, first_usable_hop = -1;
    int stat_updates = 0;
    int pgc = 0, pdc = -1;   /* C-side pending mirror (matches Python: starts cleared) */

    for (hi = 0; hi < n_hops; ++hi) {
        int part_i[3]; int sh_present, lso_present, dly_i[2], pend_i[3];
        double sd4[4], sat_d;
        int tdf_present, tdf_len;
        int stat_fired, e_usable, ree_reset_before, dt_protect_active;
        Aec3PostRunIn in;

        if (!rd(f, &ree_reset_before, 4)) { fprintf(stderr, "rrb\n"); return 2; }
        if (ree_reset_before) ree_reset(&ree);
        /* Per-hop dt_protect_active: the production caller (aec.c) sets this on
         * the SG before aec3_post_run from dt_aware_res_floor_enabled &
         * ne_recent_frames (upstream of _aec3_post). Replay the captured value. */
        if (!rd(f, &dt_protect_active, 4)) { fprintf(stderr, "dtp\n"); return 2; }
        sg.dt_protect_active = dt_protect_active;
        rd_c64(f, near_spec, n_bins);
        rd_c64(f, far_spec, n_bins);
        rd_c64(f, echo_spec, n_bins);
        rd_c64(f, esw, n_bins);
        rd_c64(f, W0, n_bins);
        rd_c64(f, W_all, n_part * n_bins);
        rd_c64(f, X_buf, n_part * n_bins);
        if (!rd(f, sqrt_hann, (size_t)block_size * 4)) { fprintf(stderr, "sh\n"); return 2; }
        if (!rd(f, part_i, sizeof(int) * 3)) { fprintf(stderr, "part\n"); return 2; }
        tdf_present = part_i[1]; tdf_len = part_i[2];
        if (tdf_present && tdf_len > 0) {
            if (!rd(f, tdf, (size_t)tdf_len * 4)) { fprintf(stderr, "tdf\n"); return 2; }
        }
        if (!rd(f, &sh_present, 4)) { fprintf(stderr, "shp\n"); return 2; }
        rd_c64(f, sh_err, n_bins);
        if (!rd(f, &lso_present, 4)) { fprintf(stderr, "lsop\n"); return 2; }
        if (!rd(f, lso, (size_t)hop * 4)) { fprintf(stderr, "lso\n"); return 2; }
        if (!rd(f, sd4, sizeof(double) * 4)) { fprintf(stderr, "sd4\n"); return 2; }
        if (!rd(f, raw_output, (size_t)hop * 4)) { fprintf(stderr, "raw\n"); return 2; }
        if (!rd(f, near_end, (size_t)hop * 4)) { fprintf(stderr, "near\n"); return 2; }
        if (!rd(f, far_end, (size_t)hop * 4)) { fprintf(stderr, "far\n"); return 2; }
        if (!rd(f, dly_i, sizeof(int) * 2)) { fprintf(stderr, "dly\n"); return 2; }
        if (!rd(f, &sat_d, 8)) { fprintf(stderr, "sat\n"); return 2; }
        if (!rd(f, pend_i, sizeof(int) * 3)) { fprintf(stderr, "pend\n"); return 2; }
        if (!rd(f, &e_usable, 4)) { fprintf(stderr, "eus\n"); return 2; }
        if (!rd(f, e_out, (size_t)hop * 4)) { fprintf(stderr, "eout\n"); return 2; }
        if (!rd(f, &stat_fired, 4)) { fprintf(stderr, "statf\n"); return 2; }

        /* ── replay the UPSTREAM stationarity update (process loop) ───────── */
        if (stat_fired) {
            int k;
            for (k = 0; k < n_bins; ++k)
                far_psd_st[k] = cmag2(far_spec[k].r, far_spec[k].i);
            stationarity_estimator_update_noise_estimator(&stat, far_psd_st);
            /* Python passes average_reverb=self._aec3_avg_render_reverb.reverb,
             * which is the avg-render-reverb state left by the PREVIOUS hop's
             * aec3_post_run (== post.avg_reverb.reverb here). Feeding NULL
             * desyncs the stationarity band mask → R² zeroing → out divergence
             * once the stationarity path engages (~hop 82). */
            stationarity_estimator_update_stationarity_flags(
                &stat, far_psd_st, post.avg_reverb.reverb);
            stat_updates++;
        }

        /* ── assemble inputs ──────────────────────────────────────────────── */
        memset(&in, 0, sizeof(in));
        in.near_spec = near_spec; in.far_spec = far_spec;
        in.echo_spec = echo_spec; in.error_spec_windowed = esw;
        in.W0 = W0; in.W_all = W_all; in.X_buf = X_buf;
        in.sqrt_hann = sqrt_hann;
        in.kalman_P = NULL; in.kalman_P_len = 0;   /* divergence_indicator dead */
        in.partition_idx = part_i[0];
        in.n_partitions = n_part;
        in.filter_taps_full = (tdf_present && tdf_len > 0) ? tdf : NULL;
        in.filter_taps_full_len = tdf_len;
        in.shadow_present = sh_present;
        in.shadow_error_spec = sh_err;
        in.last_shadow_output_time = lso_present ? lso : NULL;
        in.s_ref_max = sd4[0]; in.s_coa_max = sd4[1];
        in.main_error_energy = sd4[2]; in.shadow_error_energy = sd4[3];
        in.raw_output = raw_output; in.near_end = near_end; in.far_end = far_end;
        in.current_delay = dly_i[0]; in.delay_active = dly_i[1];
        in.saturation_level = sat_d;
        in.pending_gain_change = pend_i[0];
        in.pending_delay_change = pend_i[1];
        in.stationarity_active_hops = pend_i[2];
        in.stationarity_converge_hops = cnt_i[2];
        in.erle_coh_gate_enabled = flag_i[0];
        in.use_stationarity_properties = flag_i[5];
        in.active_render_threshold = cng_d[8];

        /* Sanity vs the C-side pending mirror (Python clears them post-event;
         * the captured pre-call values are the source of truth). */
        (void)pgc; (void)pdc;

        aec3_post_run(&post, &in, &obj, &sc, out_c, &pgc, &pdc);

        /* ── assertions ───────────────────────────────────────────────────── */
        {
            int k, g_usable = aec_state_usable_linear_estimate(&state);
            if (g_usable != e_usable) {
                if (first_usable_hop < 0) first_usable_hop = hi;
                usable_mism++;
            }
            for (k = 0; k < hop; ++k) {
                if (out_c[k] != e_out[k]) {
                    double d = (double)out_c[k] - (double)e_out[k];
                    if (d < 0) d = -d;
                    if (d > out_maxd) out_maxd = d;
                    if (first_out_hop < 0) first_out_hop = hi;
                    out_mism++;
                }
            }
        }
    }
    fclose(f);

    printf("aec3_post_run FULL parity: %d hops, n_bins=%d n_part=%d "
           "stat_updates=%d\n", n_hops, n_bins, n_part, stat_updates);
    printf("  usable_linear mism=%ld (first_hop=%d)\n",
           usable_mism, first_usable_hop);
    printf("  >>> out[hop]  mism=%ld (max=%.3e first_hop=%d)\n",
           out_mism, out_maxd, first_out_hop);

    if (out_mism == 0 && usable_mism == 0) {
        printf(">>> PASS (bit-exact end-to-end, incl. out[hop] + usable_linear)\n");
        return 0;
    }
    printf(">>> FAIL\n");
    return 1;
}
