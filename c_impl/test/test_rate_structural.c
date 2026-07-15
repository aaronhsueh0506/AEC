/* test_rate_structural.c — M5 (multi-rate campaign, review F01) per-rate
 * unity/impulse/COLA structural micro-checks. Standalone-gcc convention
 * (mirrors test_config_validation.c / test_static_aec.c's own header
 * recipe) -- does NOT link into any Makefile target.
 *
 * Three checks, run at each of the three whitelisted rates (8000/16000/48000):
 *
 *   (a) COLA: the rate's own synthesis window (Aec3BalancedRateDims::
 *       synth_window, looked up via aec3b_rate_cfg()) satisfies the
 *       sqrt-Hann 50%-overlap perfect-reconstruction identity --
 *       w[i]^2 + w[i+hop_size]^2 == 1 for i in [0, hop_size) -- within 1e-6.
 *       This is the actual OLA correctness property the synthesis stage
 *       depends on; M4 already asserted synth_window_len == block_size at
 *       carve time, this test asserts the WINDOW SHAPE itself reconstructs.
 *
 *   (b) impulse-through-linear-path: build an Aec at each rate with
 *       cfg.enable_res = 0 (linear-only output, same knob aec_wav.c's
 *       --no-res flips) and cfg.enable_cng = 0 (deterministic, no dither),
 *       drive it with a periodic impulse train on far + a delayed, scaled
 *       copy on mic (a trivial single-tap synthetic echo path), and assert:
 *         - every output sample is finite (no NaN/Inf) from hop 0 onward;
 *         - after enough hops to converge, the residual energy has dropped
 *           well below the mic energy (a generous ratio threshold -- this
 *           is structural sanity that the linear path actually adapts and
 *           cancels at every rate, not a quality bar).
 *
 *   (c) aec_get_mem_size monotonicity/consistency: pool size must increase
 *       8k < 16k < 48k (bigger FFT/filter/n_partitions at higher rates), and
 *       must be identical across the three presets at a fixed rate (mild/
 *       balanced/aggressive differ only in one float field, not a size-
 *       affecting one).
 *
 * Build (standalone, KISS FFT backend; from c_impl/):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_rate_structural.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_rate_structural
 * Run:
 *   ./bin/test_rate_structural
 */
#include "aec.h"
#include "aec3_balanced_config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

static const int RATES[] = { 8000, 16000, 48000 };
#define N_RATES (int)(sizeof(RATES) / sizeof(RATES[0]))

/* ── (a) COLA: sqrt-Hann 50%-overlap perfect reconstruction ───────────── */
static void test_cola(void) {
    for (int r = 0; r < N_RATES; ++r) {
        int sr = RATES[r];
        const Aec3BalancedRateDims *rd = aec3b_rate_cfg(sr);
        char what[96];
        snprintf(what, sizeof(what), "sr=%d: aec3b_rate_cfg() found", sr);
        CHECK(rd != NULL, what);
        if (!rd) continue;

        snprintf(what, sizeof(what), "sr=%d: synth_window_len == 2*hop_size", sr);
        CHECK(rd->synth_window_len == 2 * rd->hop_size, what);

        double max_err = 0.0;
        int hop = rd->hop_size;
        for (int i = 0; i < hop; ++i) {
            double w0 = rd->synth_window[i];
            double w1 = rd->synth_window[i + hop];
            double recon = w0 * w0 + w1 * w1;
            double err = fabs(recon - 1.0);
            if (err > max_err) max_err = err;
        }
        snprintf(what, sizeof(what),
                 "sr=%d: COLA sqrt-Hann 50%% overlap (max|recon-1|=%.3e <= 1e-6)",
                 sr, max_err);
        CHECK(max_err <= 1e-6, what);
    }
}

/* Small LCG, same shape as test_zero_heap_aec.c's lcg_sample -- deterministic,
 * no libc rand() portability concerns. Returns uniform [0, 1). */
static unsigned g_lcg_state;
static float lcg_uniform(void) {
    g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
    return (float)(g_lcg_state >> 8) / (float)(1u << 24);
}

/* ── (b) impulse-through-linear-path ───────────────────────────────────── */
static void test_impulse_linear_path(void) {
    /* A perfectly PERIODIC impulse train turned out to be an adversarial
     * input for the online delay estimator: an exactly-periodic comb is
     * ambiguous under any matched filter (every period-multiple offset
     * correlates equally well), and empirically this let the estimator
     * drift onto a spurious, non-alias, WRONG lag partway through a long
     * run at 16k/48k (delay jumped from a converging 1472 to a stuck 6336
     * around hop 900, killing cancellation for the remainder -- reproduced
     * and root-caused during M5 development, not a rate-specific bug: it's
     * a generic property of feeding any exactly-periodic signal to a
     * correlation-based delay estimator). Using an i.i.d. sparse random
     * impulse train instead (mean spacing PULSE_MEAN_PERIOD, LCG-driven
     * position AND amplitude) keeps the "impulse train" character (mostly-
     * zero signal, discrete non-zero impulses) while removing the exact
     * periodicity that made the true delay ambiguous -- and is in fact the
     * more standard choice for system-identification excitation for
     * exactly this reason. */
    const float PULSE_PROB = 1.0f / 37.0f;   /* mean spacing ~37 samples */
    const int ECHO_DELAY = 41;               /* samples */
    const float ECHO_SCALE = 0.5f;
    const int N_HOPS = 1500;    /* generous convergence budget at mu=0.3 */
    const int TAIL_HOPS = 200;  /* window used for the post-convergence ratio */

    for (int r = 0; r < N_RATES; ++r) {
        int sr = RATES[r];
        char what[128];
        g_lcg_state = 0x2026075Fu;   /* fixed seed per rate: reproducible */

        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.enable_res = 0;   /* linear-only output (aec_wav.c's --no-res knob) */
        cfg.enable_cng = 0;   /* deterministic -- no dither noise floor */

        Aec aec;
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d: aec_create() for impulse test", sr);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);
        float *far = (float*)malloc((size_t)hop * sizeof(float));
        float *mic = (float*)malloc((size_t)hop * sizeof(float));
        float *out = (float*)malloc((size_t)hop * sizeof(float));

        /* Ring history of far samples so mic can look back ECHO_DELAY
         * samples across a hop boundary regardless of hop size. */
        int hist_len = ECHO_DELAY + hop + 8;
        float *far_hist = (float*)calloc((size_t)hist_len, sizeof(float));
        long sample_idx = 0;

        int all_finite = 1;
        double mic_e_tail = 0.0, out_e_tail = 0.0;

        for (int hi = 0; hi < N_HOPS; ++hi) {
            for (int k = 0; k < hop; ++k) {
                long n = sample_idx + k;
                float f = (lcg_uniform() < PULSE_PROB) ? (0.5f + 0.4f * lcg_uniform()) : 0.0f;
                far[k] = f;
                far_hist[(n) % hist_len] = f;
                long echo_n = n - ECHO_DELAY;
                float echo_src = (echo_n >= 0)
                    ? far_hist[((echo_n % hist_len) + hist_len) % hist_len]
                    : 0.0f;
                /* NOTE: far_hist is being overwritten in the SAME pass as it's
                 * read at echo_n = n - ECHO_DELAY < n, and hist_len > ECHO_DELAY,
                 * so the value at echo_n has not yet been overwritten by n's
                 * write this hop (ring is strictly larger than the lookback). */
                mic[k] = ECHO_SCALE * echo_src;
            }
            aec_process(&aec, mic, far, out);

            for (int k = 0; k < hop; ++k) {
                if (!isfinite(out[k])) all_finite = 0;
            }
            if (hi >= N_HOPS - TAIL_HOPS) {
                for (int k = 0; k < hop; ++k) {
                    mic_e_tail += (double)mic[k] * (double)mic[k];
                    out_e_tail += (double)out[k] * (double)out[k];
                }
            }
            sample_idx += hop;
        }

        snprintf(what, sizeof(what), "sr=%d: impulse-path output finite for all %d hops", sr, N_HOPS);
        CHECK(all_finite, what);

        double ratio = (mic_e_tail > 0.0) ? (out_e_tail / mic_e_tail) : 0.0;
        snprintf(what, sizeof(what),
                 "sr=%d: echo attenuated after convergence (tail out/mic energy ratio=%.4f <= 0.5)",
                 sr, ratio);
        CHECK(ratio <= 0.5, what);

        aec_destroy(&aec);
        free(far); free(mic); free(out); free(far_hist);
    }
}

/* ── (c) aec_get_mem_size monotonicity/consistency across rates ───────── */
static void test_mem_size_consistency(void) {
    size_t sizes[N_RATES];
    for (int r = 0; r < N_RATES; ++r) {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, RATES[r]);
        sizes[r] = aec_get_mem_size(&cfg);
        char what[96];
        snprintf(what, sizeof(what), "sr=%d: aec_get_mem_size() > 0", RATES[r]);
        CHECK(sizes[r] > 0, what);
    }
    CHECK(sizes[0] < sizes[1], "mem_size(8000) < mem_size(16000)");
    CHECK(sizes[1] < sizes[2], "mem_size(16000) < mem_size(48000)");

    AecPreset presets[] = { AEC_PRESET_MILD, AEC_PRESET_BALANCED, AEC_PRESET_AGGRESSIVE };
    const char *preset_names[] = { "mild", "balanced", "aggressive" };
    for (int r = 0; r < N_RATES; ++r) {
        size_t base = 0;
        for (int p = 0; p < 3; ++p) {
            AecConfig cfg;
            aec_config_from_preset(&cfg, presets[p], RATES[r]);
            size_t sz = aec_get_mem_size(&cfg);
            if (p == 0) base = sz;
            char what[128];
            snprintf(what, sizeof(what),
                     "sr=%d preset=%s: mem_size == mild's mem_size (%zu == %zu, presets don't affect sizing)",
                     RATES[r], preset_names[p], sz, base);
            CHECK(sz == base, what);
        }
    }
}

int main(void) {
    test_cola();
    test_impulse_linear_path();
    test_mem_size_consistency();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
