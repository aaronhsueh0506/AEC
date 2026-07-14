/* parity_aec_state.c — replay the binary golden from
 * python/diag/gen_aec_state_golden.py through the C aec_state integration
 * layer and assert bit-exact agreement on every public-query output across the
 * FULL real doubletalk case (WS5 Phase 5.2 capstone).
 *
 * The golden hooks the REAL balanced pipeline: per update() it dumps every
 * input + every post-update query output; per handle_echo_path_change it dumps
 * a control row. This test rebuilds a fresh AecState from the same config,
 * replays the same control/update rows in order, and compares.
 *
 * Build (from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       src/aec_state.c src/erle_estimator.c src/subband_erle.c \
 *       src/fullband_erle.c src/erl_estimator.c src/filter_quality.c \
 *       src/saturation_detector.c src/filter_delay.c src/initial_state.c \
 *       src/filter_analyzer.c src/reverb_frequency_response.c \
 *       src/aec3_scale.c src/freq_utils.c \
 *       test/parity_aec_state.c -lm -o /tmp/p_aec_state
 *   python3 ../python/diag/gen_aec_state_golden.py /tmp/aec_state_golden.bin
 *   /tmp/p_aec_state /tmp/aec_state_golden.bin
 */
#include "aec_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* per-output mismatch counters */
typedef struct {
    long usable, active, sat_echo, min_delay, fb_log2, iq_valid, iq;
    long erle, erle_unb, erl, erl_td;
} MismCounts;

static double g_max_abs_diff = 0.0;

static void note(double d) { if (d > g_max_abs_diff) g_max_abs_diff = d; }

static long cmp_f32(const float *got, const float *exp, int n) {
    long m = 0;
    for (int k = 0; k < n; ++k) {
        if (got[k] != exp[k]) {
            note(fabs((double)got[k] - (double)exp[k]));
            m++;
        }
    }
    return m;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/aec_state_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int n_bins, filter_taps_size, num_ch, hop_size, enable_fa;
    int erle_startup, erl_startup, echo_can_sat, use_lin, cons_init;
    double init_sec;
    int delay_headroom;
    double erle_min, erle_max_l, erle_max_h;
    int n_hepc, n_rows;

    if (!rd(f, &n_bins, 4) || !rd(f, &filter_taps_size, 4) || !rd(f, &num_ch, 4)
        || !rd(f, &hop_size, 4) || !rd(f, &enable_fa, 4)
        || !rd(f, &erle_startup, 4) || !rd(f, &erl_startup, 4)
        || !rd(f, &echo_can_sat, 4) || !rd(f, &use_lin, 4)
        || !rd(f, &cons_init, 4) || !rd(f, &init_sec, 8)
        || !rd(f, &delay_headroom, 4)
        || !rd(f, &erle_min, 8) || !rd(f, &erle_max_l, 8) || !rd(f, &erle_max_h, 8)
        || !rd(f, &n_hepc, 4) || !rd(f, &n_rows, 4)) {
        fprintf(stderr, "short read header\n"); return 2;
    }

    /* ── build AecState with caller-owned storage ─────────────────────────── */
    AecStateConfig cfg;
    aec_state_config_defaults(&cfg);
    cfg.n_bins = n_bins;
    cfg.num_capture_channels = num_ch;
    cfg.hop_size = hop_size;
    cfg.enable_filter_analyzer = enable_fa;
    cfg.erle_startup_hops = erle_startup;
    cfg.erl_startup_hops = erl_startup;
    cfg.echo_can_saturate = echo_can_sat;
    cfg.use_linear_filter = use_lin;
    cfg.conservative_initial_phase = cons_init;
    cfg.initial_state_seconds = init_sec;
    cfg.delay_headroom_samples = delay_headroom;
    cfg.erle_min = erle_min;
    cfg.erle_max_l = erle_max_l;
    cfg.erle_max_h = erle_max_h;
    cfg.filter_taps_size = filter_taps_size > 0 ? filter_taps_size : 960;
    /* M3: FilterAnalyzer's de-stacked fa_update() scratch is now
     * caller-owned; this harness only knows filter_taps_size, so size
     * fa_abs_scratch to it directly (the runtime np*hop bound production
     * code uses is not available at this layer). */
    cfg.fa_scratch_size = cfg.filter_taps_size;

    AecStateStorage st;
    st.erle_max = malloc((size_t)n_bins * sizeof(float));
    st.erle = malloc((size_t)n_bins * sizeof(float));
    st.erle_oc = malloc((size_t)n_bins * sizeof(float));
    st.erle_unb = malloc((size_t)n_bins * sizeof(float));
    st.erle_during = malloc((size_t)n_bins * sizeof(float));
    st.erle_coming_onset = malloc((size_t)n_bins);
    st.erle_hold = malloc((size_t)n_bins * sizeof(int32_t));
    st.erle_y2_acc = malloc((size_t)n_bins * sizeof(float));
    st.erle_e2_acc = malloc((size_t)n_bins * sizeof(float));
    st.erle_low_render = malloc((size_t)n_bins);
    st.erl = malloc((size_t)n_bins * sizeof(float));
    st.erl_hold = malloc((size_t)(n_bins - 2) * sizeof(int));
    st.filter_delays_blocks = malloc((size_t)num_ch * sizeof(int));
    st.fa_h_highpass = malloc((size_t)cfg.filter_taps_size * sizeof(float));
    st.fa_abs_scratch = malloc((size_t)cfg.fa_scratch_size * sizeof(float));
    st.fa_render_sq_scratch = malloc((size_t)cfg.hop_size * sizeof(float));

    AecState state;
    aec_state_init(&state, &cfg, &st);

    /* scratch input buffers */
    float *render_psd = malloc((size_t)n_bins * sizeof(float));
    float *capture_psd = malloc((size_t)n_bins * sizeof(float));
    float *error_psd = malloc((size_t)n_bins * sizeof(float));
    float *echo_psd = malloc((size_t)n_bins * sizeof(float));
    float *x2r = malloc((size_t)n_bins * sizeof(float));
    float *cpe = malloc((size_t)n_bins * sizeof(float));
    unsigned char *cgm = malloc((size_t)n_bins);
    float *rb = NULL; int rb_cap = 0;
    float *ft = NULL; int ft_cap = 0;

    /* expected output buffers */
    float *exp_erle = malloc((size_t)n_bins * sizeof(float));
    float *exp_erle_unb = malloc((size_t)n_bins * sizeof(float));
    float *exp_erl = malloc((size_t)n_bins * sizeof(float));

    MismCounts m; memset(&m, 0, sizeof(m));
    long n_update = 0, n_hepc_seen = 0, n_rebuild_seen = 0;
    int first_conv = -1, first_usable = -1, idx_update = 0;
    int saw_fullreset = 0, saw_gainreset = 0;

    for (int row = 0; row < n_rows; ++row) {
        int kind;
        if (!rd(f, &kind, 4)) { fprintf(stderr, "short read kind row %d\n", row); return 2; }

        if (kind == 2) {
            /* orchestrator _reset_aec3_post — full AecState rebuild (zeroes
             * filter_update_blocks_since_start too, unlike _full_reset). Re-init
             * into the same caller-owned storage. */
            aec_state_init(&state, &cfg, &st);
            n_rebuild_seen++;
            continue;
        }

        if (kind == 1) {
            int gain_change, delay_change;
            if (!rd(f, &gain_change, 4) || !rd(f, &delay_change, 4)) {
                fprintf(stderr, "short read hepc row %d\n", row); return 2;
            }
            aec_state_handle_echo_path_change(&state, gain_change, delay_change);
            if (delay_change != 0) saw_fullreset = 1;
            else if (gain_change) saw_gainreset = 1;
            n_hepc_seen++;
            continue;
        }

        /* kind == 0: update row */
        int bfc, ext_rep, ext_q, ext_s;
        if (!rd(f, &bfc, 4) || !rd(f, &ext_rep, 4) || !rd(f, &ext_q, 4)
            || !rd(f, &ext_s, 4)) goto shortrow;
        if (!rd(f, render_psd, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, capture_psd, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, error_psd, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, echo_psd, (size_t)n_bins * sizeof(float))) goto shortrow;
        int active_render;
        double s_ref, s_coa, epg;
        if (!rd(f, &active_render, 4)) goto shortrow;
        if (!rd(f, &s_ref, 8) || !rd(f, &s_coa, 8) || !rd(f, &epg, 8)) goto shortrow;

        int rb_present, rb_len;
        if (!rd(f, &rb_present, 4) || !rd(f, &rb_len, 4)) goto shortrow;
        if (rb_present) {
            if (rb_len > rb_cap) { rb = realloc(rb, (size_t)rb_len * sizeof(float)); rb_cap = rb_len; }
            if (rb_len > 0 && !rd(f, rb, (size_t)rb_len * sizeof(float))) goto shortrow;
        }
        int ft_present, ft_len;
        if (!rd(f, &ft_present, 4) || !rd(f, &ft_len, 4)) goto shortrow;
        if (ft_present) {
            if (ft_len > ft_cap) { ft = realloc(ft, (size_t)ft_len * sizeof(float)); ft_cap = ft_len; }
            if (ft_len > 0 && !rd(f, ft, (size_t)ft_len * sizeof(float))) goto shortrow;
        }
        int x2r_present, cpe_present, cgm_present;
        if (!rd(f, &x2r_present, 4)) goto shortrow;
        if (x2r_present && !rd(f, x2r, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, &cpe_present, 4)) goto shortrow;
        if (cpe_present && !rd(f, cpe, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, &cgm_present, 4)) goto shortrow;
        if (cgm_present && !rd(f, cgm, (size_t)n_bins)) goto shortrow;
        int cap_sat_pre;
        if (!rd(f, &cap_sat_pre, 4)) goto shortrow;

        /* expected outputs */
        int e_usable, e_active, e_sat_echo, e_min_delay;
        double e_fb_log2;
        int e_iq_valid;
        double e_iq;
        double e_erl_td;
        if (!rd(f, &e_usable, 4) || !rd(f, &e_active, 4) || !rd(f, &e_sat_echo, 4)
            || !rd(f, &e_min_delay, 4)) goto shortrow;
        if (!rd(f, &e_fb_log2, 8)) goto shortrow;
        if (!rd(f, &e_iq_valid, 4)) goto shortrow;
        if (!rd(f, &e_iq, 8)) goto shortrow;
        if (!rd(f, exp_erle, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, exp_erle_unb, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, exp_erl, (size_t)n_bins * sizeof(float))) goto shortrow;
        if (!rd(f, &e_erl_td, 8)) goto shortrow;

        /* set capture saturation exactly as the orchestrator did before update */
        aec_state_update_capture_saturation(&state, cap_sat_pre);

        FilterDelayEstimate ext;
        ext.reported = ext_rep;
        ext.quality = ext_q;
        ext.delay = ext_s;

        aec_state_update(&state, bfc,
                         ext_rep ? &ext : NULL,
                         render_psd, capture_psd, error_psd, echo_psd,
                         active_render, s_ref, s_coa, epg,
                         rb_present ? rb : NULL, rb_present ? rb_len : 0,
                         ft_present ? ft : NULL, ft_present ? ft_len : 0,
                         x2r_present ? x2r : NULL,
                         cpe_present ? cpe : NULL,
                         cgm_present ? cgm : NULL);

        /* track milestones */
        if (bfc && first_conv < 0) first_conv = idx_update;

        /* compare outputs */
        int g_usable = aec_state_usable_linear_estimate(&state);
        int g_active = aec_state_active_render(&state);
        int g_sat = aec_state_saturated_echo(&state);
        int g_min = aec_state_min_direct_path_filter_delay(&state);
        double g_fb = aec_state_fullband_erle_log2(&state);
        int g_iq_valid = 0;
        double g_iq = aec_state_get_inst_linear_quality_estimate(&state, &g_iq_valid);
        const float *g_erle = aec_state_erle(&state, 0);
        const float *g_erle_unb = aec_state_erle_unbounded(&state);
        const float *g_erl = aec_state_erl(&state);
        double g_erl_td = aec_state_erl_time_domain(&state);

        if (g_usable && first_usable < 0) first_usable = idx_update;

        if (g_usable != e_usable) m.usable++;
        if (g_active != e_active) m.active++;
        if (g_sat != e_sat_echo) m.sat_echo++;
        if (g_min != e_min_delay) m.min_delay++;
        if (g_fb != e_fb_log2) { note(fabs(g_fb - e_fb_log2)); m.fb_log2++; }
        if (g_iq_valid != e_iq_valid) m.iq_valid++;
        /* only compare the value when both agree it is valid */
        if (g_iq_valid && e_iq_valid && g_iq != e_iq) { note(fabs(g_iq - e_iq)); m.iq++; }
        m.erle += cmp_f32(g_erle, exp_erle, n_bins);
        m.erle_unb += cmp_f32(g_erle_unb, exp_erle_unb, n_bins);
        m.erl += cmp_f32(g_erl, exp_erl, n_bins);
        if (g_erl_td != e_erl_td) { note(fabs(g_erl_td - e_erl_td)); m.erl_td++; }

        n_update++;
        idx_update++;
    }
    fclose(f);

    long total = m.usable + m.active + m.sat_echo + m.min_delay + m.fb_log2
               + m.iq_valid + m.iq + m.erle + m.erle_unb + m.erl + m.erl_td;

    printf("aec_state parity: rows=%d update_frames=%ld hepc_events=%ld "
           "rebuilds=%ld n_bins=%d filter_taps=%d\n",
           n_rows, n_update, n_hepc_seen, n_rebuild_seen, n_bins, filter_taps_size);
    printf("  startup transition: first_converged_frame=%d first_usable_frame=%d\n",
           first_conv, first_usable);
    printf("  reset cascade exercised: full_reset=%s gain_reset=%s "
           "(hepc events from golden=%d)\n",
           saw_fullreset ? "yes" : "no", saw_gainreset ? "yes" : "no", n_hepc);
    printf("  per-output mismatches:\n");
    printf("    usable_linear_estimate      : %ld\n", m.usable);
    printf("    active_render               : %ld\n", m.active);
    printf("    saturated_echo              : %ld\n", m.sat_echo);
    printf("    min_direct_path_filter_delay: %ld\n", m.min_delay);
    printf("    fullband_erle_log2          : %ld\n", m.fb_log2);
    printf("    inst_quality valid-flag     : %ld\n", m.iq_valid);
    printf("    inst_quality value          : %ld\n", m.iq);
    printf("    erle[257]                   : %ld\n", m.erle);
    printf("    erle_unbounded[257]         : %ld\n", m.erle_unb);
    printf("    erl[257]                    : %ld\n", m.erl);
    printf("    erl_time_domain             : %ld\n", m.erl_td);
    printf("  TOTAL mismatches=%ld  max_abs_diff=%g\n", total, g_max_abs_diff);

    if (total) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact across the full case)\n");
    return 0;

shortrow:
    fprintf(stderr, "short read in update row (update#%d)\n", idx_update);
    return 2;
}
