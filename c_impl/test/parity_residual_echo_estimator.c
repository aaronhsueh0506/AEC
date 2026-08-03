/* parity_residual_echo_estimator.c — replay the binary golden from
 * python/diag/gen_residual_echo_estimator_golden.py through the C
 * ResidualEchoEstimator and assert exact (bit-for-bit float32) match for
 * r2, r2_unbounded, the _last_r2_direct/_reverb decomposition, the bound
 * ReverbFrequencyResponse tail_response, and the average_decay (f64).
 *
 * Build (from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       src/residual_echo_estimator.c src/reverb_model.c \
 *       src/reverb_frequency_response.c \
 *       test/parity_residual_echo_estimator.c -lm -o /tmp/p_ree
 *   python3 ../python/diag/gen_residual_echo_estimator_golden.py /tmp/ree_golden.bin
 *   /tmp/p_ree /tmp/ree_golden.bin
 */
#include "residual_echo_estimator.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }
static int rdi(FILE *f, int *v) { return rd(f, v, 4); }
static int rdd(FILE *f, double *v) { return rd(f, v, 8); }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/ree_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_bins, n_cases, ci, fi, k;
    long mism_r2 = 0, mism_r2u = 0, mism_dir = 0, mism_rev = 0;
    long mism_tail = 0, mism_decay = 0, total_frames = 0;
    double max_abs_diff = 0.0;
    const char *worst = "none";

    /* config preamble */
    int hop_size, model_reverb_in_nl, erle_onset_comp, reverb_enabled;
    int use_aec3_residual_noise_gate, use_stationarity_properties;
    int use_aec3_echo_gen_window, nl_r2_enabled;
    int noise_floor_hold_hops, use_freq_response, reverb_use_conservative;
    double min_noise_floor_power, noise_gate_power_legacy, noise_gate_slope;
    double stationary_gate_slope, default_gain, tm_gain, reverb_decay;
    double reverb_mild_scale, reverb_tail_strength, nl_r2_alpha, nl_norm_power;
    double residual_noise_gate_power, reverb_smoothing_base;

    ResidualEchoEstimator ree;
    ReeEchoModelConfig em;
    float *x2_nf, *reverb_model_st, *reverb_tail_st, *render_hist_st;
    float *delay_render_st, *reverb_render_st, *ld, *lr, *scratch;
    int *x2_nf_ctr;
    float *r2, *r2u;
    float *fr, *rp, *cp, *s2, *erle, *erleu;
    float *exp_r2, *exp_r2u, *exp_dir, *exp_rev, *exp_tail;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    if (!rdi(f, &n_bins)) return 2;
    if (!rdi(f, &hop_size)) return 2;
    if (!rdd(f, &min_noise_floor_power)) return 2;
    if (!rdd(f, &noise_gate_power_legacy)) return 2;
    if (!rdd(f, &noise_gate_slope)) return 2;
    if (!rdd(f, &stationary_gate_slope)) return 2;
    if (!rdi(f, &model_reverb_in_nl)) return 2;
    if (!rdd(f, &default_gain)) return 2;
    if (!rdd(f, &tm_gain)) return 2;
    if (!rdi(f, &erle_onset_comp)) return 2;
    if (!rdd(f, &reverb_decay)) return 2;
    if (!rdd(f, &reverb_mild_scale)) return 2;
    if (!rdi(f, &reverb_enabled)) return 2;
    if (!rdd(f, &reverb_tail_strength)) return 2;
    if (!rdi(f, &use_aec3_residual_noise_gate)) return 2;
    if (!rdi(f, &use_stationarity_properties)) return 2;
    if (!rdi(f, &use_aec3_echo_gen_window)) return 2;
    if (!rdi(f, &nl_r2_enabled)) return 2;
    if (!rdd(f, &nl_r2_alpha)) return 2;
    if (!rdd(f, &nl_norm_power)) return 2;
    if (!rdd(f, &residual_noise_gate_power)) return 2;
    if (!rdi(f, &noise_floor_hold_hops)) return 2;
    if (!rdi(f, &use_freq_response)) return 2;
    if (!rdi(f, &reverb_use_conservative)) return 2;
    if (!rdd(f, &reverb_smoothing_base)) return 2;

    if (!rdi(f, &n_cases)) return 2;

    /* allocate working buffers (max_partitions known per case; alloc lazily) */
    x2_nf = malloc((size_t)n_bins * sizeof(float));
    x2_nf_ctr = malloc((size_t)n_bins * sizeof(int));
    reverb_model_st = malloc((size_t)n_bins * sizeof(float));
    reverb_tail_st = malloc((size_t)n_bins * sizeof(float));
    {
        int pre = use_aec3_echo_gen_window ? 1 : 0;
        int rh = pre + 1 + 1;
        render_hist_st = malloc((size_t)rh * (size_t)n_bins * sizeof(float));
    }
    delay_render_st = malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)n_bins * sizeof(float));
    reverb_render_st = malloc((size_t)REE_DELAY_BUF_SIZE * (size_t)n_bins * sizeof(float));
    ld = malloc((size_t)n_bins * sizeof(float));
    lr = malloc((size_t)n_bins * sizeof(float));
    scratch = malloc((size_t)n_bins * sizeof(float));
    r2 = malloc((size_t)n_bins * sizeof(float));
    r2u = malloc((size_t)n_bins * sizeof(float));
    rp = malloc((size_t)n_bins * sizeof(float));
    cp = malloc((size_t)n_bins * sizeof(float));
    s2 = malloc((size_t)n_bins * sizeof(float));
    erle = malloc((size_t)n_bins * sizeof(float));
    erleu = malloc((size_t)n_bins * sizeof(float));
    exp_r2 = malloc((size_t)n_bins * sizeof(float));
    exp_r2u = malloc((size_t)n_bins * sizeof(float));
    exp_dir = malloc((size_t)n_bins * sizeof(float));
    exp_rev = malloc((size_t)n_bins * sizeof(float));
    exp_tail = malloc((size_t)n_bins * sizeof(float));

    em.min_noise_floor_power = min_noise_floor_power;
    em.noise_gate_power = noise_gate_power_legacy;
    em.noise_gate_slope = noise_gate_slope;
    em.stationary_gate_slope = stationary_gate_slope;
    em.model_reverb_in_nonlinear_mode = model_reverb_in_nl;

    for (ci = 0; ci < n_cases; ++ci) {
        int n_frames, max_part;
        if (!rdi(f, &n_frames) || !rdi(f, &max_part)) return 2;

        /* fresh estimator per case (matches the per-case AEC() instantiation) */
        ree_init(&ree, n_bins, hop_size, 16000, &em, default_gain, tm_gain,
                 erle_onset_comp, reverb_decay, reverb_mild_scale,
                 reverb_enabled, reverb_tail_strength,
                 use_aec3_residual_noise_gate, use_stationarity_properties,
                 use_aec3_echo_gen_window,
                 nl_r2_enabled, nl_r2_alpha, nl_norm_power,
                 residual_noise_gate_power, noise_floor_hold_hops,
                 use_freq_response, reverb_use_conservative,
                 reverb_smoothing_base,
                 x2_nf, x2_nf_ctr, reverb_model_st, reverb_tail_st,
                 render_hist_st, delay_render_st, reverb_render_st,
                 ld, lr, scratch);

        fr = malloc((size_t)(max_part > 0 ? max_part : 1) *
                    (size_t)n_bins * sizeof(float));

        for (fi = 0; fi < n_frames; ++fi) {
            int reset_before;
            int n_part, fdb_urm, fq_is_none, stationary;
            int dominant_ne, usable, saturated, transparent;
            int fdb, flb, force_nl;
            double fq, avg_decay_exp;

            if (!rdi(f, &reset_before)) return 2;
            if (reset_before) {
                ree_reset(&ree);
            }
            if (!rdi(f, &n_part) || !rdi(f, &fdb_urm) || !rdi(f, &fq_is_none) ||
                !rdd(f, &fq) || !rdi(f, &stationary)) return 2;
            if (!rd(f, fr, (size_t)n_part * (size_t)n_bins * sizeof(float)))
                return 2;
            if (!rdi(f, &dominant_ne) || !rdi(f, &usable) ||
                !rdi(f, &saturated) || !rdi(f, &transparent) ||
                !rdi(f, &fdb) || !rdi(f, &flb) || !rdi(f, &force_nl))
                return 2;
            if (!rd(f, rp, (size_t)n_bins * sizeof(float)) ||
                !rd(f, cp, (size_t)n_bins * sizeof(float)) ||
                !rd(f, s2, (size_t)n_bins * sizeof(float)) ||
                !rd(f, erle, (size_t)n_bins * sizeof(float)) ||
                !rd(f, erleu, (size_t)n_bins * sizeof(float)))
                return 2;
            if (!rd(f, exp_r2, (size_t)n_bins * sizeof(float)) ||
                !rd(f, exp_r2u, (size_t)n_bins * sizeof(float)) ||
                !rd(f, exp_dir, (size_t)n_bins * sizeof(float)) ||
                !rd(f, exp_rev, (size_t)n_bins * sizeof(float)) ||
                !rd(f, exp_tail, (size_t)n_bins * sizeof(float)) ||
                !rdd(f, &avg_decay_exp))
                return 2;

            /* drive the estimator: urm then estimate (orchestrator order) */
            ree_update_reverb_models(&ree, fr, n_part, fdb_urm, fq,
                                     fq_is_none, stationary);
            ree_estimate(&ree, rp, cp, s2, dominant_ne, usable, saturated,
                         transparent, erle, erleu, fdb, flb, force_nl,
                         r2, r2u);

            for (k = 0; k < n_bins; ++k) {
                if (r2[k] != exp_r2[k]) {
                    double d = (double)r2[k] - (double)exp_r2[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) { max_abs_diff = d; worst = "r2"; }
                    mism_r2++;
                }
                if (r2u[k] != exp_r2u[k]) {
                    double d = (double)r2u[k] - (double)exp_r2u[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) { max_abs_diff = d; worst = "r2u"; }
                    mism_r2u++;
                }
                if (ree.last_r2_direct[k] != exp_dir[k]) {
                    double d = (double)ree.last_r2_direct[k] - (double)exp_dir[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) { max_abs_diff = d; worst = "direct"; }
                    mism_dir++;
                }
                if (ree.last_r2_reverb[k] != exp_rev[k]) {
                    double d = (double)ree.last_r2_reverb[k] - (double)exp_rev[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) { max_abs_diff = d; worst = "reverb"; }
                    mism_rev++;
                }
                if (ree.reverb_freq_resp.tail_response[k] != exp_tail[k]) {
                    double d = (double)ree.reverb_freq_resp.tail_response[k]
                             - (double)exp_tail[k];
                    if (d < 0) d = -d;
                    if (d > max_abs_diff) { max_abs_diff = d; worst = "tail"; }
                    mism_tail++;
                }
            }
            if (ree.reverb_freq_resp.average_decay != avg_decay_exp) {
                double d = ree.reverb_freq_resp.average_decay - avg_decay_exp;
                if (d < 0) d = -d;
                if (d > max_abs_diff) { max_abs_diff = d; worst = "avg_decay"; }
                mism_decay++;
            }
            total_frames++;
        }
        free(fr);
    }
    fclose(f);

    printf("residual_echo_estimator parity: %ld frames, %d bins\n",
           total_frames, n_bins);
    printf("  mismatches: r2=%ld r2_unb=%ld direct=%ld reverb=%ld tail=%ld "
           "avg_decay=%ld\n", mism_r2, mism_r2u, mism_dir, mism_rev,
           mism_tail, mism_decay);
    {
        long total = mism_r2 + mism_r2u + mism_dir + mism_rev + mism_tail +
                     mism_decay;
        if (total) {
            printf("  max_abs_diff=%.6g (worst field: %s)\n", max_abs_diff,
                   worst);
            printf(">>> FAIL (%ld mismatches)\n", total);
            return 1;
        }
    }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
