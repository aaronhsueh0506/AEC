/* test_rate_structural.c — M5 (multi-rate campaign, review F01) per-rate
 * unity/impulse/COLA structural micro-checks. Standalone-gcc convention
 * (mirrors test_config_validation.c / test_static_aec.c's own header
 * recipe) -- does NOT link into any Makefile target.
 *
 * Five checks, run at each of the four whitelisted grids
 * (8000/256, 16000/256, 16000/512, 48000/1024):
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
 *   (b2) impulse-through-full-post-chain: identical excitation, cfg.enable_res
 *       = 1 -- the actual production path. Added alongside the 16kHz default
 *       grid flip (512/256 -> 256/128): (b) alone never touched the AEC3
 *       post-chain (SuppressionGain/ResidualEchoEstimator/CNG), which reads
 *       a per-(sample_rate, fft_size) tuning row from aec3b_rate_cfg() that
 *       nothing had exercised end-to-end before this.
 *
 *   (c) aec_get_mem_size monotonicity/consistency: pool size must increase
 *       8k < 16k < 48k (bigger FFT/filter/n_partitions at higher rates), and
 *       must be identical across the three presets at a fixed rate (mild/
 *       balanced/aggressive differ only in one float field, not a size-
 *       affecting one).
 *
 *   (c2) aec_init (static pool) byte-equal to aec_create (heap), full
 *       post-chain, every grid -- test_static_aec.c only ever exercised this
 *       at whichever fft_size happened to be the rate's *default*; this
 *       loops all four grids explicitly so a default flip can't silently
 *       leave the non-default grid's static path unchecked.
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

typedef struct { int sample_rate; int fft_size; } TestGrid;
static const TestGrid GRIDS[] = {
    { 8000, 256 }, { 16000, 256 }, { 16000, 512 }, { 48000, 1024 }
};
#define N_GRIDS (int)(sizeof(GRIDS) / sizeof(GRIDS[0]))

/* ── (a) COLA: sqrt-Hann 50%-overlap perfect reconstruction ───────────── */
static void test_cola(void) {
    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft_size = GRIDS[r].fft_size;
        const Aec3BalancedRateDims *rd = aec3b_rate_cfg(sr, fft_size);
        char what[96];
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec3b_rate_cfg() found", sr, fft_size);
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

/* ── (b) impulse-through-linear-path (and, parametrized, through the full
 * AEC3 post-chain) ────────────────────────────────────────────────────── */
static void run_impulse_test(int enable_res, const char *label) {
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

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        char what[128];
        g_lcg_state = 0x2026075Fu;   /* fixed seed per rate: reproducible */

        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = GRIDS[r].fft_size;
        cfg.enable_res = enable_res;
        cfg.enable_cng = 0;   /* deterministic -- no dither noise floor */

        Aec aec;
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d %s: aec_create() for impulse test", sr, label);
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

        snprintf(what, sizeof(what), "sr=%d %s: impulse-path output finite for all %d hops", sr, label, N_HOPS);
        CHECK(all_finite, what);

        double ratio = (mic_e_tail > 0.0) ? (out_e_tail / mic_e_tail) : 0.0;
        snprintf(what, sizeof(what),
                 "sr=%d %s: echo attenuated after convergence (tail out/mic energy ratio=%.4f <= 0.5)",
                 sr, label, ratio);
        CHECK(ratio <= 0.5, what);

        aec_destroy(&aec);
        free(far); free(mic); free(out); free(far_hist);
    }
}

static void test_impulse_linear_path(void) {
    run_impulse_test(0, "linear-only");   /* aec_wav.c's --no-res knob */
}

/* Same excitation, full AEC3 post-chain (RES) enabled -- the actual
 * production path (BALANCED ships with enable_res=1). test_impulse_linear_path
 * above only ever exercised the linear PBFDKF filter; M5 added a 16000/256
 * grid with its own aec3b_rate_cfg() tuning row (CNG/REE/SG parameters), and
 * nothing had run that row's post-chain end-to-end before this. */
static void test_impulse_res_enabled(void) {
    run_impulse_test(1, "RES-enabled");
}

/* ── (c2) aec_init (static pool) matches aec_create (heap), full post-chain,
 * every grid ───────────────────────────────────────────────────────────── */
static void test_static_pool_matches_heap(void) {
    const int N_HOPS = 50;   /* structural equality check, not a convergence test */

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft_size = GRIDS[r].fft_size;
        char what[128];
        g_lcg_state = 0x53746174u;   /* fixed seed, independent of the impulse test's */

        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft_size;
        cfg.enable_res = 1;   /* exercise the full post-chain, not just the linear filter */
        cfg.enable_cng = 0;   /* deterministic */

        Aec aec_heap;
        int rc = aec_create(&aec_heap, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_create() for static-vs-heap test", sr, fft_size);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        size_t pool_sz = aec_get_mem_size(&cfg);
        void *pool = malloc(pool_sz);
        Aec *aec_static = aec_init(pool, pool_sz, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_init() for static-vs-heap test", sr, fft_size);
        CHECK(aec_static != NULL, what);
        if (!aec_static) { aec_destroy(&aec_heap); free(pool); continue; }

        int hop = aec_hop_size(&aec_heap);
        float *far = (float*)malloc((size_t)hop * sizeof(float));
        float *mic = (float*)malloc((size_t)hop * sizeof(float));
        float *out_heap = (float*)malloc((size_t)hop * sizeof(float));
        float *out_static = (float*)malloc((size_t)hop * sizeof(float));
        int mismatch = 0;

        for (int hi = 0; hi < N_HOPS; ++hi) {
            for (int k = 0; k < hop; ++k) {
                far[k] = (lcg_uniform() < (1.0f / 37.0f)) ? (0.5f + 0.4f * lcg_uniform()) : 0.0f;
                mic[k] = 0.3f * far[k] + 0.02f * (lcg_uniform() - 0.5f);
            }
            aec_process(&aec_heap, mic, far, out_heap);
            aec_process(aec_static, mic, far, out_static);
            if (memcmp(out_heap, out_static, (size_t)hop * sizeof(float)) != 0) mismatch = 1;
        }

        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: aec_init (static) byte-equal to aec_create (heap) over %d hops, RES-enabled",
                 sr, fft_size, N_HOPS);
        CHECK(!mismatch, what);

        aec_destroy(&aec_heap);
        aec_destroy(aec_static);   /* B2 contract: required even for a pool instance
                                    * -- see test_static_aec.c's own regression guard,
                                    * some fields own resources outside the pool. */
        free(pool); free(far); free(mic); free(out_heap); free(out_static);
    }
}

/* ── (c) aec_get_mem_size monotonicity/consistency across rates ───────── */
static void test_mem_size_consistency(void) {
    size_t sizes[N_GRIDS];
    for (int r = 0; r < N_GRIDS; ++r) {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, GRIDS[r].sample_rate);
        cfg.fft_size = GRIDS[r].fft_size;
        sizes[r] = aec_get_mem_size(&cfg);
        char what[96];
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_get_mem_size() > 0",
                 GRIDS[r].sample_rate, GRIDS[r].fft_size);
        CHECK(sizes[r] > 0, what);
    }
    CHECK(sizes[0] < sizes[2], "mem_size(8000/256) < mem_size(16000/512)");
    CHECK(sizes[1] < sizes[2], "mem_size(16000/256) < mem_size(16000/512)");
    CHECK(sizes[2] < sizes[3], "mem_size(16000/512) < mem_size(48000/1024)");

    AecPreset presets[] = { AEC_PRESET_MILD, AEC_PRESET_BALANCED, AEC_PRESET_AGGRESSIVE };
    const char *preset_names[] = { "mild", "balanced", "aggressive" };
    for (int r = 0; r < N_GRIDS; ++r) {
        size_t base = 0;
        for (int p = 0; p < 3; ++p) {
            AecConfig cfg;
            aec_config_from_preset(&cfg, presets[p], GRIDS[r].sample_rate);
            cfg.fft_size = GRIDS[r].fft_size;
            size_t sz = aec_get_mem_size(&cfg);
            if (p == 0) base = sz;
            char what[128];
            snprintf(what, sizeof(what),
                     "sr=%d preset=%s: mem_size == mild's mem_size (%zu == %zu, presets don't affect sizing)",
                     GRIDS[r].sample_rate, preset_names[p], sz, base);
            CHECK(sz == base, what);
        }
    }
}

int main(void) {
    test_cola();
    test_impulse_linear_path();
    test_impulse_res_enabled();
    test_static_pool_matches_heap();
    test_mem_size_consistency();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
