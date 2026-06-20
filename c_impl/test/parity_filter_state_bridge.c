/* parity_filter_state_bridge.c — replay the binary golden from
 * python/diag/gen_filter_state_bridge_golden.py through the C
 * filter_state_bridge across a multi-frame sequence with evolving filter state.
 * WS5 Phase 5.2 gate.
 *
 * All control fields (regime / filter_converged / main_paused / mu_final /
 * external_delay / any_coarse / all_div / divergence_indicator) are replayed
 * from golden inputs and stay BIT-EXACT. The one FFT-derived output —
 * filter_taps = irfft(W0) — drifts by the KISS float32 FFT tolerance
 * (measured worst ~3e-8), so it is gated within FSB_TAPS_TOL, not bit-for-bit.
 * (Under the retired pocketfft backend filter_taps was bit-exact too.)
 *
 * Build (from c_impl/):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Iinclude -Ilib/kiss_fft \
 *       src/filter_state_bridge.c src/fft_wrapper.c lib/kiss_fft/kiss_fft.c \
 *       test/parity_filter_state_bridge.c -lm -o /tmp/p_fsb
 *   python3 ../python/diag/gen_filter_state_bridge_golden.py /tmp/fsb_golden.bin
 *   /tmp/p_fsb /tmp/fsb_golden.bin
 */
#include "filter_state_bridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* filter_taps = irfft(W0): KISS float32 vs numpy fp64, measured worst ~3e-8. */
#define FSB_TAPS_TOL 1.0e-4

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/fsb_golden.bin";
    FILE *f = fopen(path, "rb");
    int fft_size, n_freqs, p_len, n_frames, frame;
    int mism = 0;
    double max_abs_diff = 0.0;
    Complex *W0;
    float *P, *taps, *exp_taps;
    FftHandle *fft;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &fft_size, 4) || !rd(f, &n_freqs, 4) ||
        !rd(f, &p_len, 4) || !rd(f, &n_frames, 4)) {
        fprintf(stderr, "short read header\n"); return 2;
    }

    W0       = malloc((size_t)n_freqs * sizeof(Complex));
    P        = malloc((size_t)p_len * sizeof(float));
    taps     = malloc((size_t)fft_size * sizeof(float));  /* >= 1 */
    exp_taps = malloc((size_t)fft_size * sizeof(float));
    fft      = fft_create(fft_size);
    if (!W0 || !P || !taps || !exp_taps || !fft) {
        fprintf(stderr, "alloc/fft_create failed\n"); return 2;
    }

    for (frame = 0; frame < n_frames; ++frame) {
        int has_W, has_shadow, filter_converged, main_paused;
        int external_delay, any_coarse, all_div;
        double main_e, shadow_e, mu_final;
        int exp_taps_len, exp_regime, exp_fc, exp_mp, exp_ed, exp_ac, exp_ad;
        double exp_div, exp_mu;
        FilterStateBridge br;
        int k;

        /* --- inputs --- */
        if (!rd(f, &has_W, 4) || !rd(f, &has_shadow, 4)) goto shortread;
        if (!rd(f, W0, (size_t)n_freqs * sizeof(Complex))) goto shortread;
        if (!rd(f, P, (size_t)p_len * sizeof(float))) goto shortread;
        if (!rd(f, &main_e, 8) || !rd(f, &shadow_e, 8)) goto shortread;
        if (!rd(f, &filter_converged, 4) || !rd(f, &main_paused, 4)) goto shortread;
        if (!rd(f, &mu_final, 8)) goto shortread;
        if (!rd(f, &external_delay, 4)) goto shortread;
        if (!rd(f, &any_coarse, 4) || !rd(f, &all_div, 4)) goto shortread;

        /* --- expected outputs --- */
        if (!rd(f, &exp_taps_len, 4)) goto shortread;
        if (!rd(f, exp_taps, (size_t)exp_taps_len * sizeof(float))) goto shortread;
        if (!rd(f, &exp_div, 8)) goto shortread;
        if (!rd(f, &exp_regime, 4)) goto shortread;
        if (!rd(f, &exp_fc, 4) || !rd(f, &exp_mp, 4)) goto shortread;
        if (!rd(f, &exp_mu, 8)) goto shortread;
        if (!rd(f, &exp_ed, 4)) goto shortread;
        if (!rd(f, &exp_ac, 4) || !rd(f, &exp_ad, 4)) goto shortread;

        /* --- run --- */
        filter_state_bridge_build(
            &br, has_W ? fft : NULL,
            has_W ? W0 : NULL, fft_size, n_freqs,
            P, (size_t)p_len,
            has_shadow, main_e, shadow_e,
            filter_converged, main_paused, mu_final,
            external_delay, any_coarse, all_div,
            taps);

        /* --- compare --- */
        if (br.filter_taps_len != exp_taps_len) {
            printf("frame %d: taps_len %d != %d\n",
                   frame, br.filter_taps_len, exp_taps_len);
            mism++;
        } else {
            for (k = 0; k < exp_taps_len; ++k) {
                if (br.filter_taps[k] != exp_taps[k]) {
                    double d = fabs((double)br.filter_taps[k] - (double)exp_taps[k]);
                    if (d > max_abs_diff) max_abs_diff = d;
                    if (d > FSB_TAPS_TOL) mism++;  /* irfft float32-FFT drift below tol is OK */
                }
            }
        }
        if (br.divergence_indicator != exp_div) {
            double d = fabs(br.divergence_indicator - exp_div);
            if (d > max_abs_diff) max_abs_diff = d;
            printf("frame %d: div %.17g != %.17g (|d|=%g)\n",
                   frame, br.divergence_indicator, exp_div, d);
            mism++;
        }
        if (br.regime != exp_regime) { printf("frame %d: regime %d!=%d\n", frame, br.regime, exp_regime); mism++; }
        if (br.filter_converged != exp_fc) { printf("frame %d: fc %d!=%d\n", frame, br.filter_converged, exp_fc); mism++; }
        if (br.main_paused != exp_mp) { printf("frame %d: mp %d!=%d\n", frame, br.main_paused, exp_mp); mism++; }
        if (br.mu_final != exp_mu) { printf("frame %d: mu %.17g!=%.17g\n", frame, br.mu_final, exp_mu); mism++; }
        if (br.external_delay_samples != exp_ed) { printf("frame %d: ed %d!=%d\n", frame, br.external_delay_samples, exp_ed); mism++; }
        if (br.any_coarse_filter_converged != exp_ac) { printf("frame %d: ac %d!=%d\n", frame, br.any_coarse_filter_converged, exp_ac); mism++; }
        if (br.all_filters_diverged != exp_ad) { printf("frame %d: ad %d!=%d\n", frame, br.all_filters_diverged, exp_ad); mism++; }
    }

    fclose(f);
    fft_destroy(fft);
    printf("filter_state_bridge parity: %d frames, fft_size=%d, n_freqs=%d, "
           "p_len=%d, mismatches=%d, max_abs_diff=%g\n",
           n_frames, fft_size, n_freqs, p_len, mism, max_abs_diff);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (control fields bit-exact; filter_taps within %.0e, max=%g)\n",
           (double)FSB_TAPS_TOL, max_abs_diff);
    return 0;

shortread:
    fprintf(stderr, "short read frame %d\n", frame);
    return 2;
}
