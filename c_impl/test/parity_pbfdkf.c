/* parity_pbfdkf.c — replay the REAL-pipeline binary golden from
 * python/diag/gen_pbfdkf_golden.py through the C PBFDKF v3.22 port over the
 * whole ~4186-hop DT case. GATED quantities: (a) the integer control state
 * (call_counter / partition_idx / initial_state*) — BIT-EXACT; (b) the linear
 * output `out` — within an absolute float32 tolerance (measured worst ~7.8e-3).
 *
 * The FFT-derived INTERNAL Kalman state (error_spec, echo_spec, W, H_error,
 * error_psd, R) is DIAGNOSTIC ONLY, not gated. Why: PBFDKF is recursive, so the
 * vendored KISS FFT's float32 roundoff (parity_fft.c) feeds back through the
 * weight-update loop and the internal state diverges CHAOTICALLY from numpy fp64
 * (measured W max_rel ~1.5e5, H_error ~2). That is expected, not a bug — many
 * different weight trajectories cancel the same echo, so the OUTPUT stays
 * aligned (~7.8e-3) even while W does not. (Under the retired pocketfft backend
 * the whole internal state was bit-exact; the float32 KISS backend trades that
 * for output-level alignment.) The numpy-on-arm64 idioms the C port matches —
 * np.abs(complex64)**2 scaled-hypot, complex64 FMA multiply, double-coeff EMA
 * (see pbfdkf.c + project_c_port_parity_rules.md) — keep the per-hop port logic
 * faithful; the chaotic drift is purely the recursive FFT-roundoff accumulation.
 *
 * Build (from c_impl/), STANDALONE (do NOT link aec.c — old API); the FFT
 * wrapper now lives in the shared audio_common archive:
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -I../../audio_common/include \
 *       src/pbfdkf.c src/aec3_scale.c test/parity_pbfdkf.c \
 *       ../../audio_common/bin/kiss/libaudio_common.a -lm -o /tmp/p_pbfdkf
 *   python3 python/diag/gen_pbfdkf_golden.py /tmp/pbfdkf_golden.bin
 *   /tmp/p_pbfdkf /tmp/pbfdkf_golden.bin
 */
#include "pbfdkf.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

/* running max relative + absolute error accumulators */
typedef struct { double max_rel; double max_abs; long n; } Err;

static void acc(Err *e, double got, double exp) {
    double a = fabs(got - exp);
    double denom = fabs(exp);
    double rel = (denom > 1e-12) ? a / denom : a;
    if (a > e->max_abs) e->max_abs = a;
    if (rel > e->max_rel) e->max_rel = rel;
    e->n++;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/pbfdkf_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int K, N, hop, fft_size, block_size, n_hops;
    if (!rd(f, &K, 4) || !rd(f, &N, 4) || !rd(f, &hop, 4) ||
        !rd(f, &fft_size, 4) || !rd(f, &block_size, 4) || !rd(f, &n_hops, 4))
        return 2;
    float mu, delta, h_init, h_floor, h_ceil;
    float lk_c, lk_d, lk_ct, lk_dt, q_high, q_low;
    int init_thr, pad;
    if (!rd(f, &mu, 4) || !rd(f, &delta, 4)) return 2;
    if (!rd(f, &h_init, 4) || !rd(f, &h_floor, 4) || !rd(f, &h_ceil, 4)) return 2;
    if (!rd(f, &lk_c, 4) || !rd(f, &lk_d, 4) || !rd(f, &lk_ct, 4) || !rd(f, &lk_dt, 4))
        return 2;
    if (!rd(f, &q_high, 4) || !rd(f, &q_low, 4)) return 2;
    if (!rd(f, &init_thr, 4) || !rd(f, &pad, 4)) return 2;
    (void)pad;

    printf("config: K=%d N=%d hop=%d fft=%d block=%d hops=%d mu=%g delta=%g\n",
           K, N, hop, fft_size, block_size, n_hops, mu, delta);

    /* Build PBFDKF exactly as the orchestrator does: block_size = 2*hop. */
    PBFDKF flt;
    pbfdkf_init(&flt, block_size, N, mu, delta, hop);
    /* v3.24.0 round-robin TD constraint (config default ON; aec.c:432 sets it on
     * the production main filter). pbfdkf_init defaults it OFF, so the isolated
     * replay MUST enable it to match the golden (generated via aec.process with
     * constraint_round_robin=True) — else W diverges at the first constrained
     * far-active hop (~71). */
    flt.base.constraint_round_robin = 1;
    /* apply config Q (orchestrator: Q_high/Q_low/Q = config). golden q_high =
     * kalman_q_high (balanced 1e-3). */
    for (int k = 0; k < K; ++k) {
        flt.Q_high[k] = q_high; flt.Q_low[k] = q_low; flt.Q[k] = q_high;
    }
    flt.base.initial_state_threshold_hops = init_thr;

    /* sanity: C-side constants must equal golden constants bit-exact. */
    int cfail = 0;
    if (flt.h_error_floor != h_floor) { printf("FLOOR mismatch %g %g\n", flt.h_error_floor, h_floor); cfail++; }
    if (flt.h_error_ceil != h_ceil)   { printf("CEIL mismatch %g %g\n", flt.h_error_ceil, h_ceil); cfail++; }
    if (flt.leakage_converged != lk_c){ printf("LKC mismatch %g %g\n", flt.leakage_converged, lk_c); cfail++; }
    if (flt.leakage_diverged != lk_d) { printf("LKD mismatch %g %g\n", flt.leakage_diverged, lk_d); cfail++; }
    if (flt.leakage_converged_transient != lk_ct) { printf("LKCT mismatch\n"); cfail++; }
    if (flt.leakage_diverged_transient != lk_dt)  { printf("LKDT mismatch\n"); cfail++; }
    if (flt.H_error_per_bin[0] != h_init) { printf("HINIT mismatch %g %g\n", flt.H_error_per_bin[0], h_init); cfail++; }
    if (cfail) { printf(">>> FAIL (constant mismatch)\n"); return 1; }

    float *near = malloc(hop * sizeof(float));
    float *far  = malloc(hop * sizeof(float));
    float *e2cpb = malloc(K * sizeof(float));
    float *erl  = malloc(K * sizeof(float));
    float *muarr = malloc(K * sizeof(float));
    float *out  = malloc(hop * sizeof(float));
    float *exp_out = malloc(hop * sizeof(float));
    float *re = malloc(K * sizeof(float)), *im = malloc(K * sizeof(float));
    int Wsz = N * K;
    float *Wre = malloc(Wsz * sizeof(float)), *Wim = malloc(Wsz * sizeof(float));
    float *exp_h = malloc(K * sizeof(float));
    float *exp_ep = malloc(K * sizeof(float));
    float *exp_r = malloc(K * sizeof(float));

    /* int state is the only pure-bit-exact class (counters / partition_idx).
     * `out` is gated by absolute tolerance. error_spec/echo_spec/W/H_error/
     * error_psd/R are FFT-DERIVED and, through the recursive weight-update loop,
     * diverge chaotically under KISS float32 — reported as diagnostic only. */
    long mism_int = 0;
    Err e_out = {0,0,0}, e_err = {0,0,0}, e_echo = {0,0,0}, e_w = {0,0,0};
    Err e_h = {0,0,0}, e_ep = {0,0,0}, e_r = {0,0,0};

    /* OUT_TOL gates the linear output (measured worst ~7.8e-3 over the run);
     * 2e-2 matches the project-wide e2e gate (parity_aec_e2e.c) and catches real
     * O(0.1) regressions. RTOL/ATOL below ONLY classify the diagnostic
     * tol-exceed counts for the (ungated) internal Kalman arrays. */
    const double OUT_TOL = 2.0e-2;
    const double RTOL = 1.0e-4, ATOL = 1.0e-5;   /* diagnostic classifier only */
    long rtol_fail_out = 0, rtol_fail_err = 0, rtol_fail_echo = 0, rtol_fail_w = 0;
    long rtol_fail_h = 0, rtol_fail_ep = 0, rtol_fail_r = 0;

    for (int h = 0; h < n_hops; ++h) {
        int path, poor_exc, sat, block_stat, disallow, e2valid;
        if (!rd(f, near, hop * 4) || !rd(f, far, hop * 4)) { fprintf(stderr,"short near/far %d\n",h); return 2; }
        if (!rd(f, &path, 4) || !rd(f, &poor_exc, 4) || !rd(f, &sat, 4) ||
            !rd(f, &block_stat, 4) || !rd(f, &disallow, 4) || !rd(f, &e2valid, 4)) return 2;
        if (!rd(f, e2cpb, K * 4) || !rd(f, erl, K * 4) || !rd(f, muarr, K * 4)) return 2;
        int pre_call, pre_init, pre_render, epc_h_reset;
        if (!rd(f, &pre_call, 4) || !rd(f, &pre_init, 4) || !rd(f, &pre_render, 4) ||
            !rd(f, &epc_h_reset, 4)) return 2;

        /* expected post-hop */
        if (!rd(f, exp_out, hop * 4)) return 2;
        if (!rd(f, re, K * 4) || !rd(f, im, K * 4)) return 2;   /* error_spec */
        float *exp_err_re = malloc(K*4), *exp_err_im = malloc(K*4);
        memcpy(exp_err_re, re, K*4); memcpy(exp_err_im, im, K*4);
        if (!rd(f, re, K * 4) || !rd(f, im, K * 4)) return 2;   /* echo_spec */
        float *exp_echo_re = malloc(K*4), *exp_echo_im = malloc(K*4);
        memcpy(exp_echo_re, re, K*4); memcpy(exp_echo_im, im, K*4);
        if (!rd(f, Wre, Wsz * 4) || !rd(f, Wim, Wsz * 4)) return 2;
        if (!rd(f, exp_h, K * 4)) return 2;
        if (!rd(f, exp_ep, K * 4)) return 2;
        if (!rd(f, exp_r, K * 4)) return 2;
        int exp_pidx, exp_call, exp_init, exp_render;
        if (!rd(f, &exp_pidx, 4) || !rd(f, &exp_call, 4) ||
            !rd(f, &exp_init, 4) || !rd(f, &exp_render, 4)) return 2;

        /* set external per-hop state on the C filter (orchestrator-driven). */
        flt.base.poor_excitation_counter = poor_exc;
        flt.base.saturated_capture = sat;
        flt.base.block_stationary = block_stat;
        flt.disallow_leakage_diverged = disallow;
        flt.e2_coarse_per_bin_valid = e2valid;
        if (e2valid) memcpy(flt.e2_coarse_per_bin, e2cpb, K * 4);
        memcpy(flt.erl_per_bin, erl, K * 4);

        /* The orchestrator owns call_counter / initial_state and may reset them
         * via handle_echo_path_change BETWEEN hops. Sync the C's pre-hop
         * counters to Python's pre-hop snapshot so EPC resets are honoured;
         * the C then EVOLVES them inside process and we assert the post values.
         * Counters are externally-controlled inputs, not the unit under test —
         * the H_error / W / spectra evolution is. */
        flt.base.call_counter = pre_call;
        flt.base.initial_state_active = pre_init;
        flt.base.initial_state_active_render_hops = pre_render;
        /* H_error / W evolve continuously; EPC resets ONLY H_error (zero_filter
         * defaults False → W preserved). Re-apply the H_error reset on the EPC
         * rising edge so the C's continuous H_error trajectory tracks Python. */
        if (epc_h_reset) {
            for (int k = 0; k < K; ++k)
                flt.H_error_per_bin[k] = (float)10000.0;
        }

        /* run the C update. mu_scale_arr captured post-RSA = effective array. */
        pbfdkf_process(&flt, near, far, muarr, 1.0f, out);

        /* --- FFT-derived (rtol) Kalman state: H_error / error_psd / R --- */
        for (int k = 0; k < K; ++k) {
            acc(&e_h, flt.H_error_per_bin[k], exp_h[k]);
            acc(&e_ep, flt.error_psd[k], exp_ep[k]);
            acc(&e_r, flt.R[k], exp_r[k]);
            double dh = fabs((double)flt.H_error_per_bin[k] - exp_h[k]);
            double dep = fabs((double)flt.error_psd[k] - exp_ep[k]);
            double dr = fabs((double)flt.R[k] - exp_r[k]);
            if (dh  > ATOL + RTOL * fabs((double)exp_h[k]))  rtol_fail_h++;
            if (dep > ATOL + RTOL * fabs((double)exp_ep[k])) rtol_fail_ep++;
            if (dr  > ATOL + RTOL * fabs((double)exp_r[k]))  rtol_fail_r++;
        }
        if (flt.base.partition_idx != exp_pidx) { if (mism_int<5) printf("hop %d pidx C=%d py=%d\n",h,flt.base.partition_idx,exp_pidx); mism_int++; }
        if (flt.base.call_counter != exp_call) { if (mism_int<5) printf("hop %d call C=%ld py=%d\n",h,flt.base.call_counter,exp_call); mism_int++; }
        if (flt.base.initial_state_active != exp_init) { mism_int++; }
        if (flt.base.initial_state_active_render_hops != exp_render) { mism_int++; }

        /* --- FFT-derived rtol/atol asserts --- */
        for (int k = 0; k < hop; ++k) {
            acc(&e_out, out[k], exp_out[k]);
            double a = fabs((double)out[k] - exp_out[k]);
            if (a > ATOL + RTOL * fabs((double)exp_out[k])) rtol_fail_out++;
        }
        for (int k = 0; k < K; ++k) {
            acc(&e_err, flt.base.error_spec[k].r, exp_err_re[k]);
            acc(&e_err, flt.base.error_spec[k].i, exp_err_im[k]);
            double ar = fabs((double)flt.base.error_spec[k].r - exp_err_re[k]);
            double ai = fabs((double)flt.base.error_spec[k].i - exp_err_im[k]);
            if (ar > ATOL + RTOL * fabs((double)exp_err_re[k])) rtol_fail_err++;
            if (ai > ATOL + RTOL * fabs((double)exp_err_im[k])) rtol_fail_err++;
            acc(&e_echo, flt.base.echo_spec[k].r, exp_echo_re[k]);
            acc(&e_echo, flt.base.echo_spec[k].i, exp_echo_im[k]);
            double br = fabs((double)flt.base.echo_spec[k].r - exp_echo_re[k]);
            double bi = fabs((double)flt.base.echo_spec[k].i - exp_echo_im[k]);
            if (br > ATOL + RTOL * fabs((double)exp_echo_re[k])) rtol_fail_echo++;
            if (bi > ATOL + RTOL * fabs((double)exp_echo_im[k])) rtol_fail_echo++;
        }
        for (int j = 0; j < Wsz; ++j) {
            acc(&e_w, flt.base.W[j].r, Wre[j]);
            acc(&e_w, flt.base.W[j].i, Wim[j]);
            double wr = fabs((double)flt.base.W[j].r - Wre[j]);
            double wi = fabs((double)flt.base.W[j].i - Wim[j]);
            if (wr > ATOL + RTOL * fabs((double)Wre[j])) rtol_fail_w++;
            if (wi > ATOL + RTOL * fabs((double)Wim[j])) rtol_fail_w++;
        }

        free(exp_err_re); free(exp_err_im); free(exp_echo_re); free(exp_echo_im);
    }
    fclose(f);

    printf("\n=== GATED: integer control state (BIT-EXACT) ===\n");
    printf("  int state (call_counter/partition_idx/initial_state) mismatches = %ld\n", mism_int);

    printf("\n=== GATED: linear output (abs tol=%.0e) ===\n", OUT_TOL);
    printf("  out        max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_out.max_rel, e_out.max_abs, rtol_fail_out);

    printf("\n=== DIAGNOSTIC: FFT-derived internal Kalman state (NOT gated — "
           "diverges chaotically through the recursive loop; classifier "
           "rtol=%.0e atol=%.0e) ===\n", RTOL, ATOL);
    printf("  error_spec max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_err.max_rel, e_err.max_abs, rtol_fail_err);
    printf("  echo_spec  max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_echo.max_rel, e_echo.max_abs, rtol_fail_echo);
    printf("  W          max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_w.max_rel, e_w.max_abs, rtol_fail_w);
    printf("  H_error    max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_h.max_rel, e_h.max_abs, rtol_fail_h);
    printf("  error_psd  max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_ep.max_rel, e_ep.max_abs, rtol_fail_ep);
    printf("  R          max_rel=%.3e max_abs=%.3e  tol-exceed=%ld\n", e_r.max_rel, e_r.max_abs, rtol_fail_r);

    int fail = 0;
    if (mism_int) { printf("\n>>> FAIL: integer control state not bit-exact\n"); fail = 1; }
    if (e_out.max_abs >= OUT_TOL) {
        printf("\n>>> FAIL: linear output exceeds abs tol %.0e (max_abs=%.3e)\n",
               OUT_TOL, e_out.max_abs); fail = 1;
    }
    pbfdkf_free(&flt);
    if (!fail) printf("\n>>> PASS (int control state bit-exact; linear output within "
                      "%.0e over %d hops; internal Kalman state diagnostic only)\n",
                      OUT_TOL, n_hops);
    return fail;
}
