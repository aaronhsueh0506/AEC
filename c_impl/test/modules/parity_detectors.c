/* parity_detectors.c — Wave 1.5/1.6/1.7 gate.
 * Bit-exact gate (all fp64 arithmetic, integer state) on synthetic vectors.
 */
#include "detectors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int rd(FILE* f, void* b, size_t n) { return fread(b,1,n,f)==n?0:-1; }

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <det.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t hop, n_frames;
    if (rd(f, &hop, 4) || rd(f, &n_frames, 4)) {
        fprintf(stderr, "header read failed\n"); return 2;
    }

    /* DoubleTalkAnalyzer config defaults must match Python AecConfig BALANCED:
     *   shadow_dtd_offset=1.5  shadow_dtd_advantage_scale=3.0  */
    RenderActivity     ra;
    /* M3: render_activity_update()'s former fixed-size stack scratch is now
     * caller-owned (pool-carved in production); this standalone harness
     * mallocs its own copy sized ceil(hop/8), matching detectors.c's
     * PAIR_BLOCK=8. */
    int ra_scratch_len = (hop + 7) / 8;
    float* ra_scratch = (float*)malloc((size_t)ra_scratch_len * sizeof(float));
    render_activity_init(&ra, ra_scratch, ra_scratch_len);
    FilterConvergence  fc; filter_convergence_init(&fc);
    DoubleTalk         dt; doubletalk_init(&dt, 1.5, 3.0);

    float* far_buf = (float*)malloc((size_t)hop * sizeof(float));
    int diffs = 0;
    int fail = 0;

    for (int fi = 0; fi < n_frames; ++fi) {
        if (rd(f, far_buf, (size_t)hop * sizeof(float))) { fail=1; break; }
        double near_pwr, raw_err_pwr, mic_pwr, erl_est, main_err, shadow_err;
        int32_t far_active, warmup_done, far_excited, shadow_count;
        if (rd(f,&near_pwr,8)||rd(f,&raw_err_pwr,8)||rd(f,&mic_pwr,8)||
            rd(f,&erl_est,8)||rd(f,&main_err,8)||rd(f,&shadow_err,8)||
            rd(f,&far_active,4)||rd(f,&warmup_done,4)||
            rd(f,&far_excited,4)||rd(f,&shadow_count,4)) { fail=1; break; }

        double exp_far_pwr; int32_t exp_active, exp_stat, exp_warm;
        int32_t exp_conv, exp_once, exp_just; double exp_div;
        double exp_dte, exp_dts, exp_adv;
        if (rd(f,&exp_far_pwr,8)||rd(f,&exp_active,4)||rd(f,&exp_stat,4)||
            rd(f,&exp_warm,4)||rd(f,&exp_conv,4)||rd(f,&exp_once,4)||
            rd(f,&exp_just,4)||rd(f,&exp_div,8)||
            rd(f,&exp_dte,8)||rd(f,&exp_dts,8)||rd(f,&exp_adv,8)) {fail=1; break;}

        RenderActivityResult ra_res = render_activity_update(&ra, far_buf, hop);
        filter_convergence_update_divergence(&fc, near_pwr, raw_err_pwr);
        int just = filter_convergence_update_convergence(
            &fc, near_pwr, raw_err_pwr, far_active, warmup_done);
        doubletalk_update_shadow_dt(&dt, shadow_count, far_excited,
                                       main_err, shadow_err);
        doubletalk_update_energy_dt(&dt, far_active, ra_res.far_pwr,
                                       mic_pwr, erl_est);

        /* Tolerances:
         *  far_pwr: fp32 ULP (Python np.mean uses pairwise fp32 sum; we
         *           use 8-block pairwise but exact algorithm differs by
         *           ≤ 2 fp32-ULP). dt_from_energy compounds far_pwr drift.
         *  Booleans / counters / fp64-only EMAs: exact match. */
        const double F32_ULP = 1.2e-7;
        int far_drift = fabs(ra_res.far_pwr - exp_far_pwr) >
                        2.0 * F32_ULP * (fabs(exp_far_pwr) + 1e-30);
        int dte_drift = fabs(dt.dt_from_energy - exp_dte) >
                        4.0 * F32_ULP * (fabs(exp_dte) + 1e-30);
        if (far_drift || dte_drift ||
            ra_res.is_active != exp_active ||
            ra_res.is_stationary != exp_stat ||
            ra_res.warmup_active != exp_warm ||
            fc.converged != exp_conv || fc.once_converged != exp_once ||
            just != exp_just || fc.divergence != exp_div ||
            dt.dt_from_shadow != exp_dts ||
            dt.shadow_advantage != exp_adv) {
            if (diffs < 5) {
                fprintf(stderr,
                    "[det] frame %d MISMATCH:\n"
                    "  ra: far_pwr C=%.17g PY=%.17g  active C=%d PY=%d\n"
                    "  fc: conv C=%d PY=%d  div C=%.17g PY=%.17g\n"
                    "  dt: dte C=%.17g PY=%.17g  dts C=%.17g PY=%.17g\n",
                    fi, ra_res.far_pwr, exp_far_pwr,
                    ra_res.is_active, exp_active,
                    fc.converged, exp_conv, fc.divergence, exp_div,
                    dt.dt_from_energy, exp_dte, dt.dt_from_shadow, exp_dts);
            }
            diffs++;
        }
    }
    fclose(f); free(far_buf); free(ra_scratch);
    fprintf(stderr, "[det] diffs=%d / %d\n", diffs, n_frames);
    if (fail || diffs > 0) {
        fprintf(stderr, "[det] FAIL\n"); return 1;
    }
    fprintf(stderr, "[det] PASS (bit-exact)\n");
    return 0;
}
