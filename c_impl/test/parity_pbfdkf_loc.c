/* parity_pbfdkf_loc.c — first-divergence localizer for the PBFDKF v3.22 C port.
 * Same golden as parity_pbfdkf.c, but instead of a gate it reports, per quantity,
 * the FIRST hop with any bit-exact (!=) mismatch and dumps the first few
 * diverging bins (C value, Py value, ULP gap). Because hop 0 starts from
 * identical init on both sides, a hop-0 mismatch pinpoints the op with no
 * compounding. This is a DIAGNOSTIC tool only — it always exits 0, never gates.
 *
 * NOTE: under the KISS FFT (float32) backend the C output is NOT bit-exact to
 * numpy fp64, so this localizer reports a "first mismatch" hop (~71) that is
 * merely the onset of expected float32 ULP drift, NOT a port bug. It was written
 * for the retired pocketfft backend (numpy-precision) where a hop-0 mismatch
 * really did pinpoint a wrong op; to use it that way again you need a
 * numpy-precision FFT. For the production gate see parity_pbfdkf.c (int state
 * bit-exact + output tolerance) and parity_aec_e2e.c.
 *
 * Build (from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Iinclude -Ilib/kiss_fft \
 *       src/pbfdkf.c src/fft_wrapper.c lib/kiss_fft/kiss_fft.c \
 *       src/aec3_scale.c test/parity_pbfdkf_loc.c -lm -o /tmp/p_loc
 *   /tmp/p_loc /tmp/pbfdkf_golden.bin
 */
#include "pbfdkf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int ulp_gap(float a, float b) {
    int32_t ia, ib;
    memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
    if (ia < 0) ia = 0x80000000 - ia;
    if (ib < 0) ib = 0x80000000 - ib;
    long d = (long)ia - (long)ib;
    return (int)(d < 0 ? -d : d);
}

typedef struct {
    const char *name;
    int first_hop;     /* -1 = none yet */
    long total_mism;
    long hops_with_mism;
    int max_ulp;
} Q;

static void note(Q *q, int hop, float got, float exp, int *hop_had, int *dumped) {
    if (got != exp) {
        if (q->first_hop < 0) q->first_hop = hop;
        q->total_mism++;
        int u = ulp_gap(got, exp);
        if (u > q->max_ulp) q->max_ulp = u;
        *hop_had = 1;
        if (q->first_hop == hop && *dumped < 6) {
            printf("    %-12s C=% .9e  Py=% .9e  ulp=%d\n", q->name, got, exp, u);
            (*dumped)++;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/pbfdkf_golden.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

    int K, N, hop, fft_size, block_size, n_hops;
    if (!rd(f, &K, 4) || !rd(f, &N, 4) || !rd(f, &hop, 4) ||
        !rd(f, &fft_size, 4) || !rd(f, &block_size, 4) || !rd(f, &n_hops, 4)) return 2;
    float mu, delta, h_init, h_floor, h_ceil, lk_c, lk_d, lk_ct, lk_dt, q_high, q_low;
    int init_thr, pad;
    if (!rd(f, &mu, 4) || !rd(f, &delta, 4)) return 2;
    if (!rd(f, &h_init, 4) || !rd(f, &h_floor, 4) || !rd(f, &h_ceil, 4)) return 2;
    if (!rd(f, &lk_c, 4) || !rd(f, &lk_d, 4) || !rd(f, &lk_ct, 4) || !rd(f, &lk_dt, 4)) return 2;
    if (!rd(f, &q_high, 4) || !rd(f, &q_low, 4)) return 2;
    if (!rd(f, &init_thr, 4) || !rd(f, &pad, 4)) return 2;
    (void)pad; (void)h_init; (void)h_floor; (void)h_ceil;
    (void)lk_c; (void)lk_d; (void)lk_ct; (void)lk_dt;

    printf("config: K=%d N=%d hop=%d fft=%d block=%d hops=%d\n",
           K, N, hop, fft_size, block_size, n_hops);

    PBFDKF flt;
    pbfdkf_init(&flt, block_size, N, mu, delta, hop);
    flt.base.constraint_round_robin = 1;  /* v3.24.0 default-ON (aec.c:432) — match golden */
    for (int k = 0; k < K; ++k) { flt.Q_high[k]=q_high; flt.Q_low[k]=q_low; flt.Q[k]=q_high; }
    flt.base.initial_state_threshold_hops = init_thr;

    float *near = malloc(hop*4), *far = malloc(hop*4);
    float *e2cpb = malloc(K*4), *erl = malloc(K*4), *muarr = malloc(K*4);
    float *out = malloc(hop*4), *exp_out = malloc(hop*4);
    float *re = malloc(K*4), *im = malloc(K*4);
    int Wsz = N*K;
    float *Wre = malloc(Wsz*4), *Wim = malloc(Wsz*4);
    float *exp_h = malloc(K*4), *exp_ep = malloc(K*4), *exp_r = malloc(K*4);
    float *err_re = malloc(K*4), *err_im = malloc(K*4);
    float *echo_re = malloc(K*4), *echo_im = malloc(K*4);

    Q q_out={"out",-1,0,0,0}, q_err={"error_spec",-1,0,0,0}, q_echo={"echo_spec",-1,0,0,0};
    Q q_w={"W",-1,0,0,0}, q_h={"H_error",-1,0,0,0}, q_ep={"error_psd",-1,0,0,0}, q_r={"R",-1,0,0,0};
    int first_any = -1;

    for (int h = 0; h < n_hops; ++h) {
        int p_, pe, sat, bs, dis, e2v;
        if (!rd(f, near, hop*4) || !rd(f, far, hop*4)) return 2;
        if (!rd(f, &p_,4)||!rd(f,&pe,4)||!rd(f,&sat,4)||!rd(f,&bs,4)||!rd(f,&dis,4)||!rd(f,&e2v,4)) return 2;
        if (!rd(f, e2cpb, K*4) || !rd(f, erl, K*4) || !rd(f, muarr, K*4)) return 2;
        int pre_call, pre_init, pre_render, epc_h_reset;
        if (!rd(f,&pre_call,4)||!rd(f,&pre_init,4)||!rd(f,&pre_render,4)||!rd(f,&epc_h_reset,4)) return 2;
        if (!rd(f, exp_out, hop*4)) return 2;
        if (!rd(f, err_re, K*4) || !rd(f, err_im, K*4)) return 2;
        if (!rd(f, echo_re, K*4) || !rd(f, echo_im, K*4)) return 2;
        if (!rd(f, Wre, Wsz*4) || !rd(f, Wim, Wsz*4)) return 2;
        if (!rd(f, exp_h, K*4) || !rd(f, exp_ep, K*4) || !rd(f, exp_r, K*4)) return 2;
        int e_pidx,e_call,e_init,e_render;
        if (!rd(f,&e_pidx,4)||!rd(f,&e_call,4)||!rd(f,&e_init,4)||!rd(f,&e_render,4)) return 2;
        (void)p_; (void)re; (void)im;

        flt.base.poor_excitation_counter = pe;
        flt.base.saturated_capture = sat;
        flt.base.block_stationary = bs;
        flt.disallow_leakage_diverged = dis;
        flt.e2_coarse_per_bin_valid = e2v;
        if (e2v) memcpy(flt.e2_coarse_per_bin, e2cpb, K*4);
        memcpy(flt.erl_per_bin, erl, K*4);
        flt.base.call_counter = pre_call;
        flt.base.initial_state_active = pre_init;
        flt.base.initial_state_active_render_hops = pre_render;
        if (epc_h_reset) for (int k=0;k<K;++k) flt.H_error_per_bin[k]=(float)10000.0;

        pbfdkf_process(&flt, near, far, muarr, 1.0f, out);

        int hh = 0, dumped = 0;
        for (int k=0;k<hop;++k) note(&q_out, h, out[k], exp_out[k], &hh, &dumped);
        for (int k=0;k<K;++k) {
            note(&q_err, h, flt.base.error_spec[k].r, err_re[k], &hh, &dumped);
            note(&q_err, h, flt.base.error_spec[k].i, err_im[k], &hh, &dumped);
        }
        for (int k=0;k<K;++k) {
            note(&q_echo, h, flt.base.echo_spec[k].r, echo_re[k], &hh, &dumped);
            note(&q_echo, h, flt.base.echo_spec[k].i, echo_im[k], &hh, &dumped);
        }
        for (int j=0;j<Wsz;++j) {
            note(&q_w, h, flt.base.W[j].r, Wre[j], &hh, &dumped);
            note(&q_w, h, flt.base.W[j].i, Wim[j], &hh, &dumped);
        }
        for (int k=0;k<K;++k) note(&q_h, h, flt.H_error_per_bin[k], exp_h[k], &hh, &dumped);
        for (int k=0;k<K;++k) note(&q_ep, h, flt.error_psd[k], exp_ep[k], &hh, &dumped);
        for (int k=0;k<K;++k) note(&q_r, h, flt.R[k], exp_r[k], &hh, &dumped);

        if (hh) {
            q_out.hops_with_mism += (q_out.first_hop==h)||(q_out.total_mism&&0); /* noop */
            if (first_any < 0) {
                first_any = h;
                printf("\n>>> FIRST diverging hop = %d  (path=%d e2valid=%d call=%d init=%d)\n",
                       h, p_, e2v, e_call, e_init);
            }
        }
    }
    fclose(f);

    printf("\n=== per-quantity bit-exact divergence ===\n");
    Q *qs[] = {&q_out,&q_echo,&q_err,&q_w,&q_h,&q_ep,&q_r};
    for (int i=0;i<7;++i)
        printf("  %-12s first_hop=%-6d total_mism=%-10ld max_ulp=%d\n",
               qs[i]->name, qs[i]->first_hop, qs[i]->total_mism, qs[i]->max_ulp);
    pbfdkf_free(&flt);
    return 0;
}
