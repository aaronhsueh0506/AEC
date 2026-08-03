/* test_rate_structural.c — per-rate unity/impulse/COLA structural checks.
 * Standalone-gcc convention
 * (mirrors test_config_validation.c / test_static_aec.c's own header
 * recipe). It is wired to the `make test-rate-structural` target.
 *
 * Seven checks, run at each of the four whitelisted grids
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
 *   (d) top-level (non-AEC3) constant retiming (2026-08 gap-fix): proves
 *       shadow_err_alpha / warmup_frames / epc_hangover / ne_recent_hold /
 *       filter_misadjustment_{stable,hangover}_frames and EpcDetector's
 *       EPV_FAST_TC/EPV_SLOW_TC now cover approximately the SAME
 *       wall-clock duration at every grid -- these are project-native
 *       (non-AEC3) constants that were literal hop counts / per-hop EMA
 *       constants frozen at the legacy hop=160/sample_rate=16000 (10 ms)
 *       grid with zero rate conversion, exactly the same bug class the
 *       AEC3-internal per-block/hop-count constant audit fixed for
 *       subband_erle/fullband_erle/etc, just missed in that pass because
 *       these live in aec.c/epc_shadow.c/config.py/epc.py instead. Builds
 *       via aec_config_from_preset() THEN overrides cfg.fft_size (mirrors
 *       aec_wav.c's --fft-size flag) before aec_create() -- the exact
 *       post-defaults grid-override path that forced retiming to live in
 *       aec_carve() rather than aec_config_defaults() (see aec.c's
 *       aec_legacy10ms_hops()/aec_legacy10ms_alpha() doc comment).
 *
 *   (e) external RES/NR seam WOLA identity: with internal RES disabled and
 *       return_res_context enabled, reconstruct ctx.error_spec with the
 *       matching sqrt-Hann synthesis/OLA and verify it produces the previous
 *       ctx.formed_hop. Runs with shadow selection both enabled and disabled,
 *       and also verifies near_spec == error_spec + echo_spec. This catches a
 *       wrong context pointer to PBFDKF's non-reconstructing estimator
 *       error_spec_windowed even though the synthesis window itself passes
 *       the standalone COLA check in (a).
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

/* ── (e) external RES/NR seam is a reconstructing 50%-overlap WOLA ─────── */
static void test_res_context_wola_identity(void) {
    const int N_HOPS = 80;

    for (int shadow = 0; shadow <= 1; ++shadow) {
        for (int r = 0; r < N_GRIDS; ++r) {
            int sr = GRIDS[r].sample_rate;
            int fft_size = GRIDS[r].fft_size;
            const Aec3BalancedRateDims *rd = aec3b_rate_cfg(sr, fft_size);
            AecConfig cfg;
            Aec aec;
            FftHandle *fft = NULL;
            float *far = NULL, *mic = NULL, *out = NULL;
            float *previous = NULL, *ifft = NULL, *ola = NULL;
            char what[192];
            float max_recon_error = 0.0f;
            float max_sum_error = 0.0f;
            int context_valid = 1;

            aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
            cfg.fft_size = fft_size;
            cfg.enable_shadow = shadow;
            cfg.enable_res = 0;
            cfg.return_res_context = 1;
            cfg.enable_cng = 0;
            cfg.enable_delay_est = 0;
            cfg.enable_highpass = 0;
            cfg.enable_saturation = 0;

            int rc = rd ? aec_create(&aec, &cfg) : -1;
            if (rc == 0) fft = fft_create(fft_size);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: construct WOLA seam test",
                     sr, fft_size, shadow);
            CHECK(rc == 0 && fft != NULL, what);
            if (rc != 0 || !fft) {
                if (rc == 0) aec_destroy(&aec);
                fft_destroy(fft);
                continue;
            }

            int hop = aec_hop_size(&aec);
            int n_freqs = fft_size / 2 + 1;
            far = (float*)malloc((size_t)hop * sizeof(float));
            mic = (float*)malloc((size_t)hop * sizeof(float));
            out = (float*)malloc((size_t)hop * sizeof(float));
            previous = (float*)calloc((size_t)hop, sizeof(float));
            ifft = (float*)malloc((size_t)fft_size * sizeof(float));
            ola = (float*)calloc((size_t)fft_size, sizeof(float));
            if (!far || !mic || !out || !previous || !ifft || !ola)
                context_valid = 0;

            g_lcg_state = 0x574F4C41u + (unsigned)(17 * r + shadow);
            for (int hi = 0; hi < N_HOPS && context_valid; ++hi) {
                AecResContext ctx;
                for (int i = 0; i < hop; ++i) {
                    far[i] = 0.18f * (2.0f * lcg_uniform() - 1.0f);
                    mic[i] = 0.45f * far[i]
                           + 0.025f * (2.0f * lcg_uniform() - 1.0f);
                }
                aec_process(&aec, mic, far, out);
                aec_get_res_context(&aec, &ctx);
                if (!ctx.error_spec || !ctx.echo_spec || !ctx.near_spec ||
                    !ctx.formed_hop || ctx.n_freqs != n_freqs ||
                    ctx.hop_size != hop) {
                    context_valid = 0;
                    break;
                }

                for (int k = 0; k < n_freqs; ++k) {
                    float sum_re = ctx.error_spec[k].r + ctx.echo_spec[k].r;
                    float sum_im = ctx.error_spec[k].i + ctx.echo_spec[k].i;
                    float dr = fabsf(ctx.near_spec[k].r - sum_re);
                    float di = fabsf(ctx.near_spec[k].i - sum_im);
                    if (dr > max_sum_error) max_sum_error = dr;
                    if (di > max_sum_error) max_sum_error = di;
                }

                fft_inverse(fft, ctx.error_spec, ifft);
                for (int i = 0; i < fft_size; ++i)
                    ola[i] += ifft[i] * rd->synth_window[i];
                for (int i = 0; i < hop; ++i) {
                    float d = fabsf(ola[i] - previous[i]);
                    if (d > max_recon_error) max_recon_error = d;
                }
                memmove(ola, ola + hop,
                        (size_t)(fft_size - hop) * sizeof(float));
                memset(ola + (fft_size - hop), 0,
                       (size_t)hop * sizeof(float));
                memcpy(previous, ctx.formed_hop,
                       (size_t)hop * sizeof(float));
            }

            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: RES context exposes complete WOLA fields",
                     sr, fft_size, shadow);
            CHECK(context_valid, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: unity-gain WOLA reconstructs formed hop "
                     "(max abs %.3g <= 1e-4)",
                     sr, fft_size, shadow, max_recon_error);
            CHECK(context_valid && max_recon_error <= 1e-4f, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: near == error + echo "
                     "(max abs %.3g <= 1e-6)",
                     sr, fft_size, shadow, max_sum_error);
            CHECK(context_valid && max_sum_error <= 1e-6f, what);

            free(far); free(mic); free(out); free(previous); free(ifft); free(ola);
            fft_destroy(fft);
            aec_destroy(&aec);
        }
    }
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

/* ── (d) top-level (non-AEC3) constant retiming (2026-08 gap-fix) ───────
 * See file header comment. Legacy target: the wall-clock duration/TC each
 * constant was authored to cover at the legacy hop=160/sample_rate=16000
 * (10 ms) grid, before per-rate multi-grid support existed. A relative
 * tolerance of 15% comfortably covers integer-hop-count rounding at the
 * smallest field (epc_hangover, ~12-25 hops depending on grid) while
 * still failing hard against the pre-fix bug class, where the SAME hop
 * COUNT was frozen at every grid (e.g. warmup_frames=100 covered 0.8 s /
 * 1.6 s / 1.067 s at 16k/256, 16k/512, 48k/1024 respectively -- up to 2x
 * apart, far outside a 15% band). */
static void check_close(const char *label, int sr, int fft, double actual_ms,
                        double target_ms, double rel_tol) {
    double rel_err = fabs(actual_ms - target_ms) / target_ms;
    char what[192];
    snprintf(what, sizeof(what),
             "sr=%d fft=%d: %s wall-clock %.2f ms ~= legacy target %.2f ms (rel_err=%.4f <= %.2f)",
             sr, fft, label, actual_ms, target_ms, rel_err, rel_tol);
    CHECK(rel_err <= rel_tol, what);
}

static void test_top_level_constant_retiming(void) {
    const double HOP_MS_TOL = 0.15;    /* integer-hop-count rounding budget */
    const double ALPHA_TOL  = 0.01;    /* growth_rehop is continuous (no
                                        * integer rounding), so this should
                                        * be near-exact modulo float32 error */

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;

        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        /* Post-defaults grid override -- mirrors aec_wav.c's --fft-size
         * flag / this file's own (b)/(c)/(c2) pattern, and is exactly the
         * path that forced retiming to live in aec_carve() rather than
         * aec_config_defaults() (see aec.c's aec_legacy10ms_hops()/
         * aec_legacy10ms_alpha() doc comment). */
        cfg.fft_size = fft;

        Aec aec;
        char what[128];
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_create() for retiming test", sr, fft);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);
        double hop_ms = 1000.0 * hop / sr;

        check_close("warmup_frames", sr, fft, aec.cfg.warmup_frames * hop_ms, 1000.0, HOP_MS_TOL);
        check_close("epc_hangover", sr, fft, aec.cfg.epc_hangover * hop_ms, 200.0, HOP_MS_TOL);
        check_close("ne_recent_hold", sr, fft, aec.cfg.ne_recent_hold * hop_ms, 1500.0, HOP_MS_TOL);
        check_close("filter_misadjustment_stable_frames", sr, fft,
                    aec.cfg.filter_misadjustment_stable_frames * hop_ms, 300.0, HOP_MS_TOL);
        check_close("filter_misadjustment_hangover_frames", sr, fft,
                    aec.cfg.filter_misadjustment_hangover_frames * hop_ms, 1000.0, HOP_MS_TOL);

        /* Time-constant form (EMA "RETENTION" convention -- see
         * aec3_growth_rehop's doc comment) for the three alpha-shaped
         * constants: TC = -hop_seconds / ln(retention). Continuous (no
         * integer rounding), so this should track the legacy target far
         * tighter than the hop-count fields above. */
        double tc_shadow = -(hop_ms / 1000.0) / log((double)aec.cfg.shadow_err_alpha) * 1000.0;
        check_close("shadow_err_alpha (TC)", sr, fft, tc_shadow, 44.82, ALPHA_TOL);

        double tc_fast = -(hop_ms / 1000.0) / log((double)aec.epc.epv_fast_tc) * 1000.0;
        check_close("EPV_FAST_TC (TC)", sr, fft, tc_fast, 495.0, ALPHA_TOL);

        double tc_slow = -(hop_ms / 1000.0) / log((double)aec.epc.epv_slow_tc) * 1000.0;
        check_close("EPV_SLOW_TC (TC)", sr, fft, tc_slow, 9995.0, ALPHA_TOL);

        aec_destroy(&aec);
    }

    /* ne_recent_sustain: genuine consecutive-event count, NOT a duration --
     * confirm it stays the untouched literal 3 at every grid (the negative
     * space of this test: proving it was correctly left OUT of the
     * retiming batch, not merely that it wasn't exercised above). */
    for (int r = 0; r < N_GRIDS; ++r) {
        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, GRIDS[r].sample_rate);
        cfg.fft_size = GRIDS[r].fft_size;
        char what[128];
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: ne_recent_sustain stays literal 3 (event count, not retimed)",
                 GRIDS[r].sample_rate, GRIDS[r].fft_size);
        CHECK(cfg.ne_recent_sustain == 3, what);
    }
}

int main(void) {
    test_cola();
    test_impulse_linear_path();
    test_impulse_res_enabled();
    test_res_context_wola_identity();
    test_static_pool_matches_heap();
    test_mem_size_consistency();
    test_top_level_constant_retiming();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
