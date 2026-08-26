/* test_rate_structural.c — per-rate unity/impulse/COLA structural checks.
 * Standalone-gcc convention
 * (mirrors test_config_validation.c / test_static_aec.c's own header
 * recipe). It is wired to the `make test-rate-structural` target.
 *
 * Ten checks, run at each of the four whitelisted grids
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
 *   (d2) adaptation-constant retiming (2026-08-06 batch): the same property
 *       for the five constants that live in the ADAPTATION path rather than
 *       in AecConfig -- Aec::alpha_pow (per-SAMPLE), alpha_erl_tracking/
 *       alpha_erl_converged, PBFDKF::alpha_r, and
 *       Saturation::alpha_attack/alpha_release. (d)
 *       missed them because they are assigned in aec_carve()/pbfdkf_init()/
 *       saturation_init(), not read out of the config struct. This is the C
 *       mirror of python/tests/test_detector_timing_effective_values.py and
 *       exists so the C/Python agreement is a REGRESSION GATE rather than a
 *       one-off manual comparison. Includes both halves of the negative space:
 *       the 16 ms-authored saturation pair must NOT move on a 16 ms grid, and
 *       the 10 ms/per-sample constants MUST move where the grids differ --
 *       including alpha_r, whose anchor moved to 10 ms on 2026-08-07, so a
 *       16 ms grid is now one of the places it has to move.
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
 *   (e2) spatial_linear_context: an Aec configured to skip computing its own
 *       (otherwise-unused) per-lane suppression gain must still expose an
 *       identical R²/comfort_noise/near_spec/echo_spec/error_spec/formed_hop
 *       to a baseline instance fed the same inputs -- only res_gain differs
 *       (NULL instead of the computed array). Runs with shadow selection
 *       both enabled and disabled.
 *
 *   (e3) suppression_gain_update_dominant_nearend() vs get_gain(): drives two
 *       SuppressionGain instances (via two Aec's a3_sg) through a
 *       deliberately-constructed enter/hold/early-exit cycle of the
 *       DominantNearend state machine and asserts dne_trigger_counter/
 *       dne_hold_counter/dne_nearend_state match every hop. This is the
 *       actual "removing the state-updating prefix must fail" mutation
 *       proof for the spatial_linear_context design in (e2) -- deterministic
 *       by construction rather than dependent on an incidental signal
 *       reaching these states. Mutation-tested by temporarily dropping
 *       get_gain()'s call to suppression_gain_update_dominant_nearend():
 *       confirmed to fail at all 4 grids (unlike (e2)'s dependence on
 *       whether a specific synthetic signal happens to reach these states).
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
#include "aec3_scale.h"
#include "aec_test_hooks.h"   /* AEC_TESTING-only; see check (d6) */

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

/* ── (e2) spatial_linear_context: skipping the unused per-lane G_res leaves
 * every OTHER context field byte-identical, every grid, shadow on/off ────
 * A multi-channel wrapper (e.g. pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c) sets
 * this to stop computing a per-lane suppression gain that only a downstream
 * fused/beamformed stage ever reads, while the DominantNearend hold-state
 * (suppression_gain_update_dominant_nearend(), still called every hop) keeps
 * advancing so the NEXT hop's ERLE onset decision, and therefore r2, are
 * unaffected. comfort_noise is a separate, independent Step 19 computation
 * (one of DominantNearend's own inputs, not something it affects) and is
 * checked here for completeness, not because it depends on this mechanism.
 * Proves that "keep the state-updating prefix, skip everything else" is a
 * true no-op on every field except res_gain itself, end to end through a
 * real Aec instance.
 *
 * The DominantNearend state machine itself (enter/hold/early-exit) is
 * mutation-tested in isolation, at every grid, in
 * test_suppression_gain_dne_prefix_matches_get_gain() below -- see that
 * test for the actual "removing the prefix must fail" proof; this test's
 * job is bit-exactness of the mode as a whole, not of that one mechanism. */
static void test_spatial_linear_context_matches_baseline(void) {
    /* 200, not 80: at the default 16k/256 grid, DominantNearend does not
     * reach its ACTIVE hold state within 80 hops on this synthetic signal --
     * a run that never activates DNE would not actually exercise the one
     * state transition (dne_nearend_state, fed forward into the next hop's
     * ERLE onset decision) this test exists to protect. 200 hops was
     * verified (by hand, across every grid x shadow combination below) to
     * reach dne_nearend_state==1 at least once; saw_dominant_nearend below
     * asserts that, so a future signal-generation change that regresses
     * back to "never activates" fails loudly instead of silently testing
     * less than it claims to. */
    const int N_HOPS = 200;

    for (int shadow = 0; shadow <= 1; ++shadow) {
        for (int r = 0; r < N_GRIDS; ++r) {
            int sr = GRIDS[r].sample_rate;
            int fft_size = GRIDS[r].fft_size;
            AecConfig cfg;
            Aec base, spatial;
            float *far = NULL, *mic = NULL, *out_base = NULL, *out_spatial = NULL;
            char what[192];
            int base_res_gain_ever_null = 0;
            int spatial_res_gain_seen_null = 1;
            int r2_identical = 1, cn_identical = 1, near_identical = 1;
            int echo_identical = 1, error_identical = 1, formed_identical = 1;
            int out_identical = 1;
            int all_finite = 1;
            int saw_dominant_nearend = 0;
            int valid = 1;

            aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
            cfg.fft_size = fft_size;
            cfg.enable_shadow = shadow;
            cfg.enable_res = 0;
            cfg.return_res_context = 1;
            cfg.enable_cng = 0;
            cfg.enable_delay_est = 0;
            cfg.enable_highpass = 0;
            cfg.enable_saturation = 0;

            int rc_base = aec_create(&base, &cfg);
            cfg.spatial_linear_context = 1;
            int rc_spatial = aec_create(&spatial, &cfg);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: construct spatial_linear_context pair",
                     sr, fft_size, shadow);
            CHECK(rc_base == 0 && rc_spatial == 0, what);
            if (rc_base != 0 || rc_spatial != 0) {
                if (rc_base == 0) aec_destroy(&base);
                if (rc_spatial == 0) aec_destroy(&spatial);
                continue;
            }

            int hop = aec_hop_size(&base);
            int n_freqs = fft_size / 2 + 1;
            far = (float*)malloc((size_t)hop * sizeof(float));
            mic = (float*)malloc((size_t)hop * sizeof(float));
            out_base = (float*)malloc((size_t)hop * sizeof(float));
            out_spatial = (float*)malloc((size_t)hop * sizeof(float));
            if (!far || !mic || !out_base || !out_spatial) valid = 0;

            /* Bursty far-end (loud/near-silent, ~300 ms per half-cycle) rather
             * than stationary noise: a constant-level signal never creates a
             * genuine render-level transition, which is what feeds
             * SubbandErle's coming_onset/hold_counter machinery -- with a
             * flat signal, erle_onset_compensated tracks erle exactly, so
             * onset never has anything to select between and a broken
             * dominant_ne feed goes undetected. Authored in wall-clock ms and
             * retimed per grid (this codebase's own convention for
             * warmup_frames/epc_hangover/etc., aec.c's aec_carve()) rather
             * than a fixed hop count: a fixed 20-hop period was verified (by
             * hand) to only trigger the divergence at half the grids here --
             * 8000/256 and 16000/512 both have a 16 ms hop and caught it,
             * 16000/256 (8 ms hop) and 48000/1024 (10.67 ms hop) did not,
             * because 20 hops covers a different wall-clock burst length at
             * each rate. */
            const int burst_hops_local =
                ((300 * sr) / (1000 * hop)) > 0 ? (300 * sr) / (1000 * hop) : 1;
            g_lcg_state = 0x53504331u + (unsigned)(17 * r + shadow);
            for (int hi = 0; hi < N_HOPS && valid; ++hi) {
                AecResContext ctx_base, ctx_spatial;
                int burst_on = ((hi / burst_hops_local) % 2) == 0;
                float far_scale = burst_on ? 0.18f : 0.002f;
                for (int i = 0; i < hop; ++i) {
                    far[i] = far_scale * (2.0f * lcg_uniform() - 1.0f);
                    mic[i] = 0.45f * far[i]
                           + 0.025f * (2.0f * lcg_uniform() - 1.0f);
                }
                /* Identical inputs fed to both instances every hop -- mic/far
                 * are read-only, so one pair of buffers is safe to reuse. */
                aec_process(&base, mic, far, out_base);
                aec_process(&spatial, mic, far, out_spatial);
                aec_get_res_context(&base, &ctx_base);
                aec_get_res_context(&spatial, &ctx_spatial);

                if (!ctx_base.r2 || !ctx_base.comfort_noise ||
                    !ctx_base.near_spec || !ctx_base.echo_spec ||
                    !ctx_base.error_spec || !ctx_base.formed_hop ||
                    !ctx_spatial.r2 || !ctx_spatial.comfort_noise ||
                    !ctx_spatial.near_spec || !ctx_spatial.echo_spec ||
                    !ctx_spatial.error_spec || !ctx_spatial.formed_hop) {
                    valid = 0;
                    break;
                }
                /* Accumulate across ALL hops, not just the last one -- the
                 * pointer doesn't actually move hop to hop, but asserting
                 * that every hop stayed non-NULL is the honest claim, not an
                 * incidental one-hop snapshot. */
                if (ctx_base.res_gain == NULL) base_res_gain_ever_null = 1;
                if (ctx_spatial.res_gain != NULL) spatial_res_gain_seen_null = 0;

                if (suppression_gain_is_dominant_nearend(&base.a3_sg))
                    saw_dominant_nearend = 1;

                /* Genuine byte-identical checks (memcmp over the raw arrays),
                 * not a numeric-equality proxy -- fabsf(a-b) > 0 would also
                 * silently pass a NaN divergence (fabsf(NaN) compares false
                 * against any threshold), so finiteness is checked
                 * separately below rather than folded into the diff. */
                for (int k = 0; k < n_freqs && all_finite; ++k) {
                    if (!isfinite(ctx_base.r2[k]) || !isfinite(ctx_spatial.r2[k]) ||
                        !isfinite(ctx_base.comfort_noise[k]) ||
                        !isfinite(ctx_spatial.comfort_noise[k]) ||
                        !isfinite(ctx_base.near_spec[k].r) ||
                        !isfinite(ctx_base.near_spec[k].i) ||
                        !isfinite(ctx_spatial.near_spec[k].r) ||
                        !isfinite(ctx_spatial.near_spec[k].i) ||
                        !isfinite(ctx_base.echo_spec[k].r) ||
                        !isfinite(ctx_base.echo_spec[k].i) ||
                        !isfinite(ctx_spatial.echo_spec[k].r) ||
                        !isfinite(ctx_spatial.echo_spec[k].i) ||
                        !isfinite(ctx_base.error_spec[k].r) ||
                        !isfinite(ctx_base.error_spec[k].i) ||
                        !isfinite(ctx_spatial.error_spec[k].r) ||
                        !isfinite(ctx_spatial.error_spec[k].i)) {
                        all_finite = 0;
                    }
                }
                for (int i = 0; i < hop && all_finite; ++i) {
                    if (!isfinite(ctx_base.formed_hop[i]) ||
                        !isfinite(ctx_spatial.formed_hop[i]) ||
                        !isfinite(out_base[i]) || !isfinite(out_spatial[i])) {
                        all_finite = 0;
                    }
                }

                if (memcmp(ctx_base.r2, ctx_spatial.r2,
                          (size_t)n_freqs * sizeof(float)) != 0)
                    r2_identical = 0;
                if (memcmp(ctx_base.comfort_noise, ctx_spatial.comfort_noise,
                          (size_t)n_freqs * sizeof(float)) != 0)
                    cn_identical = 0;
                if (memcmp(ctx_base.near_spec, ctx_spatial.near_spec,
                          (size_t)n_freqs * sizeof(Complex)) != 0)
                    near_identical = 0;
                if (memcmp(ctx_base.echo_spec, ctx_spatial.echo_spec,
                          (size_t)n_freqs * sizeof(Complex)) != 0)
                    echo_identical = 0;
                if (memcmp(ctx_base.error_spec, ctx_spatial.error_spec,
                          (size_t)n_freqs * sizeof(Complex)) != 0)
                    error_identical = 0;
                if (memcmp(ctx_base.formed_hop, ctx_spatial.formed_hop,
                          (size_t)hop * sizeof(float)) != 0)
                    formed_identical = 0;
                if (memcmp(out_base, out_spatial, (size_t)hop * sizeof(float)) != 0)
                    out_identical = 0;
            }

            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: baseline exposes res_gain, spatial_linear_context exposes NULL",
                     sr, fft_size, shadow);
            CHECK(valid && !base_res_gain_ever_null && spatial_res_gain_seen_null, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: reached dne_nearend_state==1 at least once "
                     "(exercises the state transition this test protects)",
                     sr, fft_size, shadow);
            CHECK(valid && saw_dominant_nearend, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: all compared fields stayed finite",
                     sr, fft_size, shadow);
            CHECK(valid && all_finite, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: r2/comfort_noise/near/echo/error/formed_hop "
                     "byte-identical (memcmp) every hop",
                     sr, fft_size, shadow);
            CHECK(valid && r2_identical && cn_identical && near_identical &&
                  echo_identical && error_identical && formed_identical, what);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d shadow=%d: synthesized output byte-identical (memcmp) every hop",
                     sr, fft_size, shadow);
            CHECK(valid && out_identical, what);

            free(far); free(mic); free(out_base); free(out_spatial);
            aec_destroy(&base);
            aec_destroy(&spatial);
        }
    }
}

/* ── (e3) suppression_gain_update_dominant_nearend() reproduces get_gain()'s
 * DominantNearend trajectory exactly, through a deliberately-driven
 * enter -> hold -> early-exit cycle, at every grid ─────────────────────────
 * Deterministic contract test: two Aec instances at the same grid, one
 * driven through the full suppression_gain_get_gain(), the other through
 * only the standalone prefix. Both are fed identical, hand-crafted
 * nearend/echo/comfort_noise spectra designed (from the DominantNearend
 * trigger formula in suppression_gain.c: trigger_active = enr_pass &&
 * snr_pass, where enr_pass = echo_sum < dne_enr_threshold*ne_sum and
 * snr_pass = ne_sum > dne_snr_threshold*noise_sum; early_exit =
 * echo_sum > dne_enr_exit_threshold*ne_sum && echo_sum >
 * dne_snr_threshold*noise_sum) to force ENTER for dne_trigger_threshold_hops
 * hops, then hold under NEUTRAL input (neither condition satisfied) partway
 * into dne_hold_duration_hops, then force an EARLY EXIT. dne_trigger_counter/
 * dne_hold_counter/dne_nearend_state are compared every hop -- unlike
 * test_spatial_linear_context_matches_baseline() above (which depends on a
 * signal incidentally reaching these states), this one drives the exact
 * state machine transitions by construction, so it needs no by-hand
 * verification of coverage and mutation-catches at every grid. */
static void test_suppression_gain_dne_prefix_matches_get_gain(void) {
    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft_size = GRIDS[r].fft_size;
        AecConfig cfg;
        Aec full, prefix_only;
        float *nearend = NULL, *echo = NULL, *echo_unb = NULL, *cn = NULL;
        float *render_block = NULL;
        char what[192];
        int trig_identical = 1, hold_identical = 1, state_identical = 1;
        int saw_enter = 0, saw_hold = 0, saw_exit = 0;
        int valid = 1;

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft_size;
        cfg.enable_cng = 0;
        cfg.enable_delay_est = 0;

        int rc_full = aec_create(&full, &cfg);
        int rc_prefix = aec_create(&prefix_only, &cfg);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: construct DNE-prefix pair", sr, fft_size);
        CHECK(rc_full == 0 && rc_prefix == 0, what);
        if (rc_full != 0 || rc_prefix != 0) {
            if (rc_full == 0) aec_destroy(&full);
            if (rc_prefix == 0) aec_destroy(&prefix_only);
            continue;
        }
        /* dne_use_during_initial_phase=1 in balanced makes trigger_initial_
         * gate always true regardless -- forcing initial_state off just
         * removes that as a variable to reason about. */
        suppression_gain_set_initial_state(&full.a3_sg, 0);
        suppression_gain_set_initial_state(&prefix_only.a3_sg, 0);

        int n_bins = full.a3_sg.cfg.n_bins;
        int hop = aec_hop_size(&full);
        int trig_hops = full.a3_sg.cfg.dne_trigger_threshold_hops;
        int hold_hops = full.a3_sg.cfg.dne_hold_duration_hops;
        nearend = (float*)malloc((size_t)n_bins * sizeof(float));
        echo = (float*)malloc((size_t)n_bins * sizeof(float));
        echo_unb = (float*)malloc((size_t)n_bins * sizeof(float));
        cn = (float*)malloc((size_t)n_bins * sizeof(float));
        render_block = (float*)calloc((size_t)hop, sizeof(float));
        if (!nearend || !echo || !echo_unb || !cn || !render_block) valid = 0;

        /* Phases, driven by the trigger/exit formula's exact inequalities
         * (thresholds in balanced: dne_enr_threshold=0.25,
         * dne_enr_exit_threshold=10, dne_snr_threshold=30 -- see
         * aec3_balanced_config.h -- margins below are wide enough to hold
         * regardless of the exact per-rate table row). */
        int enter_end = trig_hops + 2;
        int hold_check = enter_end + (hold_hops > 2 ? hold_hops / 2 : 1);
        int exit_start = enter_end + (hold_hops > 0 ? hold_hops - 1 : 1);
        int total_hops = exit_start + 5;

        for (int hi = 0; hi < total_hops && valid; ++hi) {
            float ne_val, echo_val, cn_val;
            if (hi < enter_end) {
                /* ENTER: echo_sum << 0.25*ne_sum, ne_sum >> 30*noise_sum. */
                ne_val = 1.0f; echo_val = 0.001f; cn_val = 0.0001f;
            } else if (hi < exit_start) {
                /* NEUTRAL: neither trigger_active nor early_exit -- lets
                 * dne_hold_counter decay by exactly 1/hop instead of being
                 * re-armed (trigger_active) or zeroed (early_exit). */
                ne_val = 1.0f; echo_val = 1.0f; cn_val = 1.0f;
            } else {
                /* EARLY EXIT: echo_sum > 10*ne_sum AND echo_sum > 30*noise_sum. */
                ne_val = 1.0f; echo_val = 1000.0f; cn_val = 0.001f;
            }
            for (int k = 0; k < n_bins; ++k) {
                nearend[k] = ne_val;
                echo[k] = echo_val;
                echo_unb[k] = echo_val;
                cn[k] = cn_val;
            }

            suppression_gain_get_gain(&full.a3_sg, nearend, echo, echo_unb, cn,
                                      render_block, /*clock_drift=*/0,
                                      /*saturated_echo=*/0);
            suppression_gain_update_dominant_nearend(&prefix_only.a3_sg,
                                                      nearend, echo, echo_unb, cn);

            if (full.a3_sg.dne_trigger_counter != prefix_only.a3_sg.dne_trigger_counter)
                trig_identical = 0;
            if (full.a3_sg.dne_hold_counter != prefix_only.a3_sg.dne_hold_counter)
                hold_identical = 0;
            if (full.a3_sg.dne_nearend_state != prefix_only.a3_sg.dne_nearend_state)
                state_identical = 0;

            if (hi == enter_end - 1 && full.a3_sg.dne_nearend_state) saw_enter = 1;
            if (hi == hold_check && full.a3_sg.dne_nearend_state) saw_hold = 1;
            if (hi == total_hops - 1 && !full.a3_sg.dne_nearend_state) saw_exit = 1;
        }

        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: driven signal actually reaches enter(%d)/hold(%d)/exit(%d)",
                 sr, fft_size, saw_enter, saw_hold, saw_exit);
        CHECK(valid && saw_enter && saw_hold && saw_exit, what);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: dne_trigger_counter/hold_counter/nearend_state "
                 "identical every hop through enter/hold/exit",
                 sr, fft_size);
        CHECK(valid && trig_identical && hold_identical && state_identical, what);

        free(nearend); free(echo); free(echo_unb); free(cn); free(render_block);
        aec_destroy(&full);
        aec_destroy(&prefix_only);
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

/* ── (d2) adaptation-constant retiming (2026-08-06 batch) ───────────────
 * The C mirror of python/tests/test_detector_timing_effective_values.py.
 *
 * (d) above covers the constants that live in AecConfig/EpcDetector. This
 * covers the five that live in the ADAPTATION path -- Aec's own EMAs, the
 * PBFDAF/PBFDKF filter EMAs and the saturation detector's attack/release --
 * which the earlier pass missed because they are set in aec_carve()/
 * pbfdkf_init()/saturation_init() rather than read out of the config struct.
 *
 * This asserts the value each instance ACTUALLY ENDS UP WITH at every grid,
 * not that a retiming call appears in the source. That distinction is the
 * reason the file exists: an external reviewer read the sources and concluded
 * the retiming had never been applied, because two of the seven values are
 * legitimately IDENTICAL to their authored literal on two of the four grids.
 * Only an effective-value assertion separates "retimed" from "not retimed".
 *
 * The reference grids are NOT uniform, verified per constant from git
 * provenance -- retiming a 16 ms constant off the 10 ms reference is wrong by
 * exactly 1.6x, which is inside no sane tolerance but is invisible if you only
 * ever sample the 16 ms grids:
 *   alpha_pow                  per-SAMPLE, authored at sr=16000
 *   alpha_erl_{tracking,conv}  per-hop, 10 ms   (5407e71)
 *   alpha_r     (PBFDKF)       per-hop, 10 ms   (introduced 16 ms at e9cb383;
 *                                               the default moved to 10 ms at
 *                                               83ced18 and every validating
 *                                               commit since kept 0.95 there)
 *   saturation attack/release  per-hop, 16 ms   (243d67c, frame 512/hop 256)
 *
 * Targets are the wall-clock TC each retention covers on its OWN reference
 * grid, so a correct retiming reproduces the same number at every rate. */
static void check_moved(const char *label, int sr, int fft,
                        double actual, double authored) {
    char what[192];
    snprintf(what, sizeof(what),
             "sr=%d fft=%d: %s = %.9f actually moved off authored %.9f",
             sr, fft, label, actual, authored);
    CHECK(fabs(actual - authored) > 1e-6, what);
}

static void check_identity(const char *label, int sr, int fft,
                           double actual, double authored) {
    char what[192];
    snprintf(what, sizeof(what),
             "sr=%d fft=%d: %s = %.9f stays authored %.9f (already on its own "
             "reference grid)", sr, fft, label, actual, authored);
    CHECK(fabs(actual - authored) <= 1e-6, what);
}

static void test_adaptation_constant_retiming(void) {
    /* growth_rehop is continuous (no integer-hop rounding), so the only error
     * here is float32 representation of the retention -- worst case ~4e-5
     * relative at alpha_erl_converged, where the retention is nearest 1. */
    const double ALPHA_TOL = 0.01;

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;

        AecConfig cfg;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft;             /* post-defaults override; see (d) */
        cfg.enable_shadow = 1;          /* the shadow is built through its own
                                         * call site; a retime applied to only
                                         * one of the two must not pass */
        cfg.enable_saturation = 1;      /* else sat_mic/sat_ref are never
                                         * initialized and would read as zero */

        Aec aec;
        char what[160];
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: aec_create() for adaptation retiming test", sr, fft);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);
        double hop_s = (double)hop / sr;

        /* Per-SAMPLE constant: its TC is one SAMPLE period over ln(alpha), not
         * one hop period. Feeding hop_s here would "pass" at 16k/256 by
         * coincidence and be wrong everywhere else. */
        double tc_alpha_pow =
            -(1.0 / sr) / log((double)aec.alpha_pow) * 1000.0;
        check_close("alpha_pow (per-sample TC)", sr, fft, tc_alpha_pow,
                    1.218483, ALPHA_TOL);

        /* Per-hop, 10 ms reference. */
        check_close("alpha_erl_tracking (TC)", sr, fft,
                    -hop_s / log((double)aec.alpha_erl_tracking) * 1000.0,
                    994.991625, ALPHA_TOL);
        check_close("alpha_erl_converged (TC)", sr, fft,
                    -hop_s / log((double)aec.alpha_erl_converged) * 1000.0,
                    9994.999166, ALPHA_TOL);
        snprintf(what, sizeof(what), "sr=%d fft=%d: shadow filter present", sr, fft);
        CHECK(aec.has_shadow, what);

        check_close("alpha_r (TC)", sr, fft,
                    -hop_s / log((double)aec.main_filter.alpha_r) * 1000.0,
                    194.957302, ALPHA_TOL);

        /* Per-hop, 16 ms reference. */
        check_close("saturation alpha_attack (TC)", sr, fft,
                    -hop_s / log((double)aec.sat_mic.alpha_attack) * 1000.0,
                    13.289337, ALPHA_TOL);
        check_close("saturation alpha_release (TC)", sr, fft,
                    -hop_s / log((double)aec.sat_mic.alpha_release) * 1000.0,
                    791.973063, ALPHA_TOL);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: sat_ref retimed identically to sat_mic", sr, fft);
        CHECK(aec.sat_ref.alpha_attack == aec.sat_mic.alpha_attack &&
              aec.sat_ref.alpha_release == aec.sat_mic.alpha_release, what);

        /* Negative space, half 1 -- the 16 ms family MUST NOT move on a 16 ms
         * grid (8000/256 and 16000/512 both have a 16.000 ms hop). A reviewer
         * seeing these unchanged is looking at correct behaviour, and retiming
         * them off the 10 ms reference by mistake would move them HERE. */
        if (hop * 1000 == 16 * sr) {
            check_identity("saturation alpha_attack", sr, fft,
                           aec.sat_mic.alpha_attack, 0.3);
            check_identity("saturation alpha_release", sr, fft,
                           aec.sat_mic.alpha_release, 0.98);
        }

        /* Negative space, half 2 -- guard against a whole-table identity map.
         * Every check_close above still passes if nothing was retimed AND the
         * reference grid happens to equal the live grid, so pin the cases that
         * cannot be explained that way. */
        if (sr == 16000 && fft == 256) {          /* 8 ms hop vs 10 ms ref */
            check_moved("alpha_erl_tracking", sr, fft, aec.alpha_erl_tracking, 0.99);
            check_moved("saturation alpha_attack", sr, fft,
                        aec.sat_mic.alpha_attack, 0.3);
        }
        if (sr == 48000) {                        /* per-sample constant */
            check_moved("alpha_pow", sr, fft, aec.alpha_pow, 0.95);
        }
        if (hop * 1000 == 16 * sr) {
            /* alpha_r moved from the 16 ms family to the 10 ms family on
             * 2026-08-07, so a 16 ms grid is no longer its identity point --
             * it must MOVE here. This assertion is the one that fails if the
             * anchor is ever silently reverted to 256/16000. */
            check_moved("alpha_r (10 ms anchor)", sr, fft,
                        aec.main_filter.alpha_r, 0.95);
        }

        aec_destroy(&aec);
    }
}

/* ── (d3) F2.4 mu-holdoff re-arm regression ─────────────────────────────
 * The holdoff counter must arm only on a FRESH double-talk onset. Ongoing DT
 * must let it run down, or marginal-DT oscillation re-arms it every hop and
 * the adaptation step size never releases.
 *
 * This shipped correct, was silently reverted by a "remove dead flag branches"
 * cleanup (2f3699f, 2026-05-27) that collapsed
 *     if (!cfg.mu_holdoff_no_reset || holdoff == 0)
 * to an unconditional arm -- keeping the arm the flag had DISABLED, because
 * mu_holdoff_no_reset lived in _LEGACY_HARDCODE_TRUE -- and stayed reverted for
 * ten weeks in both ports. Nothing caught it: the counter is internal state, no
 * test asserted on it, and a mu freeze under marginal DT does not move a bucket
 * average enough to break the benchmark.
 *
 * The observable signature is exact and stimulus-independent: with the guard,
 * the counter only ever goes 0 -> armed, or n -> n-1. Without it, a hop inside
 * an active window jumps it back UP (measured: ... 18, 20, 19 ...). So the
 * assertion is "an increase is legal only from zero" -- no threshold, no
 * tolerance, and it cannot be satisfied by a stimulus that simply fails to
 * reach the branch, because coverage is asserted separately below. */
/* ── (d4) retimed constants must reach the AUDIO PATH, not just the struct ──
 * Check (d2) asserts the VALUE each instance ends up with. That is necessary
 * and was not sufficient: a retimed retention can be computed into a field,
 * asserted on, and then read by nothing -- which is exactly what happened to
 * PBFDAF::alpha_power, whose far-power EMA was maintained on every hop of
 * both filters and consumed by no one. That EMA and its retention are gone;
 * this check now measures the retention that DOES reach audio.
 *
 * Aec::alpha_pow is the per-SAMPLE power EMA behind near_power /
 * raw_error_power, and those two drive convergence detection, the
 * instantaneous-ERLE ring and the windowed-ERLE guard. It is recovered from
 * the filter's own state with no access to the coefficient itself:
 *
 *   applying `P = a*P + (1-a)*c²` once per sample over one hop of a CONSTANT
 *   near level c gives  P_new = a^hop * P_old + (1 - a^hop) * c²
 *   =>  a = ((P_new - c²) / (P_old - c²)) ^ (1/hop)
 *
 * Two near levels make that well conditioned: settle at the first, then run a
 * single hop at the second, so the denominator is the gap between them rather
 * than a converged residual. The mic HPF is off, so near_hop is the constant
 * the caller passed, sample for sample. */
static void test_retimed_constants_reach_the_audio_path(void) {
    const double TOL = 2e-3;    /* fp32 EMA recovery over one hop */
    const float  C_SETTLE = 0.5f;
    const float  C_STEP   = 0.1f;
    const int    SETTLE_HOPS = 40;

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;
        AecConfig cfg;
        Aec aec;
        char what[224];

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft;
        cfg.enable_delay_est = 0;
        cfg.enable_highpass = 0;    /* a DC near level must survive to near_hop */
        cfg.enable_saturation = 0;
        cfg.enable_cng = 0;
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_create() for audio-path test", sr, fft);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);
        float *mic = (float*)malloc((size_t)hop * sizeof(float));
        float *far = (float*)calloc((size_t)hop, sizeof(float));
        float *out = (float*)malloc((size_t)hop * sizeof(float));
        snprintf(what, sizeof(what), "sr=%d fft=%d: audio-path test allocations", sr, fft);
        CHECK(mic && far && out, what);
        if (!mic || !far || !out) {
            free(mic); free(far); free(out); aec_destroy(&aec); continue;
        }

        for (int i = 0; i < hop; ++i) mic[i] = C_SETTLE;
        for (int h = 0; h < SETTLE_HOPS; ++h) aec_process(&aec, mic, far, out);
        const double p_old = (double)aec.near_power;

        for (int i = 0; i < hop; ++i) mic[i] = C_STEP;
        aec_process(&aec, mic, far, out);
        const double p_new = (double)aec.near_power;

        const double target = (double)C_STEP * (double)C_STEP;
        const double den = p_old - target;
        const double num = p_new - target;
        const double a_field_d = (double)aec.alpha_pow;

        /* Both halves are asserted, not assumed: a settle phase that failed to
         * move near_power, or a step hop that overshot the target, would make
         * the recovery meaningless rather than wrong. */
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: near-power step is well conditioned "
                 "(P_old=%.6g P_new=%.6g target=%.6g)", sr, fft, p_old, p_new, target);
        CHECK(den > 1e-3 && num > 0.0 && num < den, what);
        if (!(den > 1e-3 && num > 0.0 && num < den)) {
            free(mic); free(far); free(out); aec_destroy(&aec); continue;
        }

        const double a_meas = pow(num / den, 1.0 / (double)hop);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: near-power EMA actually applied alpha=%.6f, "
                 "matching the retimed field %.6f (not the 0.95 literal)",
                 sr, fft, a_meas, a_field_d);
        CHECK(fabs(a_meas - a_field_d) <= TOL, what);

        free(mic); free(far); free(out);
        aec_destroy(&aec);
    }
}

static void test_mu_holdoff_rearm_guard(void) {
    const int N_HOPS = 120;

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;
        AecConfig cfg;
        Aec aec;
        char what[192];

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft;
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_create() for holdoff test", sr, fft);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);
        float *mic = (float*)malloc((size_t)hop * sizeof(float));
        float *far = (float*)malloc((size_t)hop * sizeof(float));
        float *out = (float*)malloc((size_t)hop * sizeof(float));
        if (!mic || !far || !out) { free(mic); free(far); free(out); aec_destroy(&aec); continue; }

        int prev = aec.simple_mu_holdoff;
        int illegal_rearm = 0, armed_hops = 0, saw_decrement = 0;

        g_lcg_state = 0x4632u + (unsigned)r;
        for (int h = 0; h < N_HOPS; ++h) {
            /* Near-end dominant with a small far-end: far_power << error_power,
             * so ratio stays below simple_mu_ratio and every hop takes the
             * attack branch -- the exact condition the guard governs. */
            for (int i = 0; i < hop; ++i) {
                far[i] = 0.002f * (2.0f * lcg_uniform() - 1.0f);
                mic[i] = 0.5f   * (2.0f * lcg_uniform() - 1.0f);
            }
            aec_process(&aec, mic, far, out);

            int v = aec.simple_mu_holdoff;
            if (v > prev && prev != 0) illegal_rearm++;   /* the defect */
            if (v < prev) saw_decrement = 1;
            if (v > 0) armed_hops++;
            prev = v;
        }

        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: holdoff never re-armed from a nonzero value "
                 "(%d illegal re-arms)", sr, fft, illegal_rearm);
        CHECK(illegal_rearm == 0, what);

        /* Coverage guards: without these the check above passes trivially on a
         * stimulus that never arms the counter at all. */
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: stimulus actually armed the holdoff (%d/%d hops)",
                 sr, fft, armed_hops, N_HOPS);
        CHECK(armed_hops > 0, what);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: stimulus actually exercised the countdown", sr, fft);
        CHECK(saw_decrement, what);

        free(mic); free(far); free(out);
        aec_destroy(&aec);
    }
}

static void test_render_activity_first_observation(void) {
    float scratch[16];
    float active[128];
    float silent[128] = {0};
    for (int i = 0; i < 128; ++i) active[i] = 0.1f;

    RenderActivity detector;
    render_activity_init(&detector, scratch, 16, 128, 16000);

    RenderActivityResult first =
        render_activity_update(&detector, active, 128);
    CHECK(first.is_active && !first.is_stationary,
          "render activity needs two observations before stationarity");

    RenderActivityResult second =
        render_activity_update(&detector, active, 128);
    CHECK(second.is_active && second.is_stationary,
          "constant render becomes stationary on the second observation");

    (void)render_activity_update(&detector, silent, 128);
    RenderActivityResult after_silence =
        render_activity_update(&detector, active, 128);
    CHECK(after_silence.is_active && !after_silence.is_stationary,
          "stationarity requires a fresh second observation after silence");
}

/* ── (d5) alpha_r is a live constant of the DIRECT PBFDKF API ───────────────
 *
 * alpha_r cannot be pinned the way alpha_pow is pinned by (d4), because it
 * does not reach audio through `Aec` at all -- and it took two wrong readings
 * to establish why, so the mechanism is recorded here rather than in a commit
 * message:
 *
 *   Aec + shadow ON (default)  error_psd is never read. The orchestrator
 *                              publishes e2_coarse_per_bin every hop, and the
 *                              live per-bin branch of the H_error refresh
 *                              reads error_spec directly.
 *   Aec + shadow OFF           the scalar fallback DOES run and DOES read the
 *                              smoothed error_psd -- but e2_coarse_for_refresh
 *                              is likewise assigned only inside
 *                              `if (a->has_shadow)` (aec.c), so it stays 0.0f
 *                              and `use_conv = (e2_ref_sum <= e2_coa_sum)` is
 *                              constant-false whatever the coefficient is.
 *                              Confirmed empirically: 90 cases x 2 grids with
 *                              --no-shadow are byte-identical
 *                              (eval/ab_evidence/2026-08-07-alpha-r/).
 *   direct pbfdkf_process()    the caller supplies its own
 *                              e2_coarse_for_refresh, both sides of the
 *                              comparison vary, and alpha_r selects the
 *                              leakage branch and the adaptation that follows.
 *
 * So no wrapper-level A/B can observe this constant, and a zero delta from one
 * is not evidence that it is inert. This check drives PBFDKF directly, which is
 * a supported entry point (pbfdkf.h), and pins two things:
 *
 *   (1) the coefficient the EMA actually APPLIED, recovered from two runs that
 *       differ only in error_psd's starting value -- not the stored field,
 *       which is what the pre-5232ab6 tests asserted while the loop ran on a
 *       hardcoded literal;
 *   (2) that the recovered coefficient selects the leakage branch, so the
 *       constant has a downstream effect and not merely a value.
 *
 * Mutations that must fail: reverting the use site to `0.95f`; reverting the
 * reference hop to 256 (the introduce grid); deleting the EMA. */
typedef struct {
    double psd_sum;         /* sum(error_psd) after the measured hop */
    float  div_frac;        /* last_leakage_div_frac after the measured hop */
    double h_error_mean;    /* mean H_error_per_bin after the measured hop */
} DirectProbe;

/* One measured hop on a fresh PBFDKF. `old` seeds error_psd; `e2_coa` is the
 * scalar the refresh compares against. Deterministic: same seed, same signal,
 * same warm-up, so two probes differ only in the arguments. */
static DirectProbe pbfdkf_direct_probe(int fft, int hop, int sr, unsigned seed,
                                       float old, float e2_coa) {
    const int N_PART = 4;
    const int WARM = 8;
    DirectProbe out;
    PBFDKF p;
    int K = fft / 2 + 1;
    memset(&out, 0, sizeof(out));

    if (pbfdkf_init(&p, fft / 2, N_PART, 0.3f, 1e-6f, hop, sr) != 0)
        return out;

    float *mic = (float*)calloc((size_t)hop, sizeof(float));
    float *far = (float*)malloc((size_t)hop * sizeof(float));
    float *res = (float*)malloc((size_t)hop * sizeof(float));
    if (!mic || !far || !res) {
        free(mic); free(far); free(res); pbfdkf_free(&p); return out;
    }
    g_lcg_state = seed;
    for (int i = 0; i < hop; ++i) {
        far[i] = 0.25f * (2.0f * lcg_uniform() - 1.0f);
        mic[i] = 0.10f * (2.0f * lcg_uniform() - 1.0f);
    }
    for (int h = 0; h < WARM; ++h)
        pbfdkf_process(&p, mic, far, NULL, 1.0f, res);

    /* Open every gate that would divert the hop to refresh-only, so the
     * measured hop takes the full update path (EMA, then refresh). These are
     * set directly rather than driven into place: the gates are not what this
     * check is about, and driving them would make the probe stimulus-dependent.
     * poor_excitation_counter is only reset by an explicit echo-path-change
     * call, never per hop, so forcing it here is stable. */
    p.base.call_counter = 10L * N_PART;
    p.base.poor_excitation_counter = 10 * N_PART;
    p.base.saturated_capture = 0;
    p.base.block_stationary = 0;
    p.base.initial_state_active = 0;     /* steady leakage pair, not transient */

    p.e2_coarse_per_bin_valid = 0;       /* the scalar fallback under test */
    p.disallow_leakage_diverged = 0;     /* else use_conv is forced true */
    p.h_error_refresh_erl_floor = 0.0f;
    p.e2_coarse_for_refresh = e2_coa;
    p.leakage_converged = 1.25e-4f;
    p.leakage_diverged  = 0.125f;
    for (int k = 0; k < K; ++k) {
        p.error_psd[k] = old;
        p.H_error_per_bin[k] = 1.0f;     /* clear of both clamp bounds */
        p.erl_per_bin[k] = 0.5f;
    }

    pbfdkf_process(&p, mic, far, NULL, 1.0f, res);

    double psd = 0.0, he = 0.0;
    for (int k = 0; k < K; ++k) {
        psd += (double)p.error_psd[k];
        he  += (double)p.H_error_per_bin[k];
    }
    out.psd_sum = psd;
    out.h_error_mean = he / K;
    out.div_frac = p.last_leakage_div_frac;

    free(mic); free(far); free(res);
    pbfdkf_free(&p);
    return out;
}

static void test_alpha_r_reaches_the_direct_pbfdkf_path(void) {
    const double TOL = 5e-4;          /* fp32 EMA recovery over one hop */
    const double TC_TARGET_MS = 194.957302;   /* 0.95 at a 10 ms hop */

    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;
        int hop = fft / 2;
        int K = fft / 2 + 1;
        unsigned seed = 0x5A17u + (unsigned)r;
        char what[240];

        /* Probe 1 pins the stimulus term: with error_psd starting at 0 the
         * result is (1-a)*sum(e2). */
        DirectProbe p1 = pbfdkf_direct_probe(fft, hop, sr, seed, 0.0f, 0.0f);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: direct PBFDKF probe produced a non-zero "
                 "error_psd sum (%.6g)", sr, fft, p1.psd_sum);
        CHECK(p1.psd_sum > 0.0, what);
        if (!(p1.psd_sum > 0.0)) continue;

        /* Probe 2 differs ONLY in the starting value, chosen large enough that
         * K*old dominates sum(e2) -- otherwise the two coefficients produce
         * nearly the same sum and the branch test below is ill-conditioned. */
        const double old2 = 1.0 + 30.0 * p1.psd_sum / K;
        DirectProbe p2 = pbfdkf_direct_probe(fft, hop, sr, seed, (float)old2,
                                             0.0f);

        /* sum(old) = a*K*old + (1-a)*sum(e2), so the stimulus term cancels and
         * `a` comes out of the two sums alone -- no assumption about the
         * signal, the filter state, or the number of bins that converged. */
        const double a_meas = (p2.psd_sum - p1.psd_sum) / (K * old2);
        const double a_field = (double)aec3_growth_rehop(0.95f, 160, 16000,
                                                         hop, sr);

        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: error_psd EMA actually applied alpha=%.6f, "
                 "matching the 10 ms-anchored %.6f", sr, fft, a_meas, a_field);
        CHECK(fabs(a_meas - a_field) <= TOL, what);

        /* The field-vs-applied check above still passes if BOTH move together,
         * which is exactly what reverting the reference hop to 256 would do.
         * Pin the wall-clock span itself. */
        double tc_ms = -((double)hop / sr) * 1000.0 / log(a_meas);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: applied alpha_r covers TC %.3f ms, the 10 ms "
                 "anchor's %.3f ms (a 16 ms reference would give %.3f)",
                 sr, fft, tc_ms, TC_TARGET_MS, TC_TARGET_MS * 1.6);
        CHECK(fabs(tc_ms - TC_TARGET_MS) <= 1.0, what);

        /* No grid has a 10 ms hop, so the retimed value must differ from the
         * authored literal everywhere -- i.e. this check cannot be satisfied by
         * a use site that ignores the field. */
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: applied alpha_r %.6f is not the 0.95 literal",
                 sr, fft, a_meas);
        CHECK(fabs(a_meas - 0.95) > 1e-3, what);

        /* ── downstream: the coefficient selects the leakage branch ──────── */
        const double s_e2 = p1.psd_sum / (1.0 - a_meas);
        const double sum_retimed = p2.psd_sum;
        const double sum_literal = 0.95 * K * old2 + 0.05 * s_e2;
        const double mid = 0.5 * (sum_retimed + sum_literal);

        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: the two coefficients give separable error_psd "
                 "sums (%.6g vs %.6g)", sr, fft, sum_retimed, sum_literal);
        CHECK(fabs(sum_retimed - sum_literal) > 1e-3 * fabs(mid), what);

        /* With the threshold set between them, `use_conv = (sum <= e2_coa)`
         * resolves one way for the retimed coefficient and the other way for
         * the literal. div_frac is 0.0 or 1.0 on the scalar path. */
        const float expect_retimed = (sum_retimed <= mid) ? 0.0f : 1.0f;
        const float expect_literal = (sum_literal <= mid) ? 0.0f : 1.0f;
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: setup separates the branches (%.1f vs %.1f)",
                 sr, fft, expect_retimed, expect_literal);
        CHECK(expect_retimed != expect_literal, what);

        DirectProbe p3 = pbfdkf_direct_probe(fft, hop, sr, seed, (float)old2,
                                             (float)mid);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: leakage branch follows the retimed alpha "
                 "(div_frac %.1f, expected %.1f; the 0.95 literal would give "
                 "%.1f)", sr, fft, p3.div_frac, expect_retimed, expect_literal);
        CHECK(p3.div_frac == expect_retimed, what);

        /* ...and the branch moves H_error, so this is a real downstream effect
         * and not a diagnostic counter. Same probe, threshold pushed to the far
         * side, and the two H_error means must differ by (ld - lc)*erl. */
        DirectProbe p4 = pbfdkf_direct_probe(
            fft, hop, sr, seed, (float)old2,
            (float)(expect_retimed == 1.0f ? mid * 1e6 : -1.0));
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: forcing the other branch flips div_frac "
                 "(%.1f -> %.1f)", sr, fft, p3.div_frac, p4.div_frac);
        CHECK(p4.div_frac != p3.div_frac, what);

        const double expect_gap = (0.125 - 1.25e-4) * 0.5;   /* (ld-lc)*erl */
        const double gap = fabs(p3.h_error_mean - p4.h_error_mean);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: the branch moves H_error by %.6f, the expected "
                 "(leakage_diverged - leakage_converged)*erl = %.6f",
                 sr, fft, gap, expect_gap);
        CHECK(fabs(gap - expect_gap) <= 1e-3, what);
    }
}

/* ── (d6) simple-mu frozen exception: four constants, one mechanism ──
 *
 * The F2.4 holdoff and its three alphas are hop-authored. Retiming the four
 * values as a group failed the two-grid blind A/B gate, so production keeps
 * 20 and 0.3/0.99/0.95 together. The test recovers what each branch actually
 * applies and separately checks the arm site; no stored copy can pass it. */
static void mu_step(Aec* a, float far_scale, float err_scale, unsigned *seed) {
    int hop = aec_hop_size(a);
    float far[2048], out[2048];
    if (hop > 2048) return;
    for (int i = 0; i < hop; ++i) {
        *seed = *seed * 1664525u + 1013904223u;
        float u1 = (float)(*seed >> 8) / (float)(1u << 24);
        *seed = *seed * 1664525u + 1013904223u;
        float u2 = (float)(*seed >> 8) / (float)(1u << 24);
        far[i] = far_scale * (2.0f * u1 - 1.0f);
        out[i] = err_scale * (2.0f * u2 - 1.0f);
    }
    aec_testing_update_simple_mu_ratio(a, out, far, hop);
}

static void test_simple_mu_frozen_exception(void) {
    static const float expected_alpha[3] = { 0.3f, 0.99f, 0.95f };
    for (int r = 0; r < N_GRIDS; ++r) {
        int sr = GRIDS[r].sample_rate;
        int fft = GRIDS[r].fft_size;
        AecConfig cfg;
        Aec aec;
        char what[224];

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
        cfg.fft_size = fft;
        int rc = aec_create(&aec, &cfg);
        snprintf(what, sizeof(what), "sr=%d fft=%d: aec_create() for simple-mu test",
                 sr, fft);
        CHECK(rc == 0, what);
        if (rc != 0) continue;

        int hop = aec_hop_size(&aec);

        /* Prove the default grid distinguishes the frozen policy from the
         * rejected wall-clock retime. */
        if (sr == 16000 && fft == 256) {
            CHECK(aec3_ms_to_hops(200.0f, hop, sr) != 20,
                  "default grid distinguishes frozen and retimed holdoff");
            for (int i = 0; i < 3; ++i) {
                float retimed = aec3_growth_rehop(expected_alpha[i], 160,
                                                   16000, hop, sr);
                CHECK(fabsf(retimed - expected_alpha[i]) > 1e-5f,
                      "default grid distinguishes frozen and retimed alpha");
            }
        }

        /* Reset owns the runtime pair. */
        aec.simple_mu_holdoff = 9;
        aec.simple_mu_ratio = 0.123f;
        aec_reset(&aec);
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: reset clears the runtime pair", sr, fft);
        CHECK(aec.simple_mu_holdoff == 0 && aec.simple_mu_ratio == 1.0f, what);
        /* ── the coefficient each branch ACTUALLY APPLIED ─────────────────
         * ratio_new = a*ratio_old + (1-a)*r, and r depends on the frame and
         * the filter state but NOT on ratio_old, so running the same stimulus
         * from two starting points cancels it:  a = (new2-new1)/(s2-s1). */
        for (int b = 0; b < 3; ++b) {
            const char *name = (b == 0) ? "attack" : (b == 1) ? "hold" : "release";
            float want = expected_alpha[b];
            /* attack needs ratio_in < ratio_old (error >> far); hold/release
             * need ratio_in >= ratio_old (far >> error). */
            float fs = (b == 0) ? 0.001f : 0.5f;
            float es = (b == 0) ? 0.5f   : 0.001f;
            double s[2] = { 0.90, 0.95 }, got[2];
            for (int j = 0; j < 2; ++j) {
                aec_reset(&aec);
                aec.simple_mu_ratio = (float)s[j];
                aec.simple_mu_holdoff = (b == 1) ? 5 : 0;
                unsigned seed = 0xC0FFEEu + (unsigned)r;
                mu_step(&aec, fs, es, &seed);
                got[j] = aec.simple_mu_ratio;
            }
            double applied = (got[1] - got[0]) / (s[1] - s[0]);
            snprintf(what, sizeof(what),
                     "sr=%d fft=%d: %s branch applied frozen %.6f",
                     sr, fft, name, applied);
            CHECK(fabs(applied - (double)want) <= 2e-4, what);
        }

        /* The arm site is independent of the alpha recovery above. */
        aec_reset(&aec);
        aec.simple_mu_ratio = 1.0f;
        aec.simple_mu_holdoff = 0;
        {
            unsigned seed = 0xA55A00u + (unsigned)r;
            mu_step(&aec, 0.001f, 0.5f, &seed);   /* error >> far -> attack */
        }
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: a fresh attack arms frozen 20 hops (got %d)",
                 sr, fft, aec.simple_mu_holdoff);
        CHECK(aec.simple_mu_holdoff == 20, what);

        /* ── all three branches reachable, identified without instrumentation
         * The holdoff counter's transition names the branch, so branch coverage
         * is checkable on a real corpus with no diagnostic fields added to the
         * public struct:
         *   up            attack, fresh onset      down    hold
         *   nonzero flat  attack, ongoing          0 flat  release
         * (an attack at 0 would arm it, and the limit is always >= 1.) */
        int hit_attack = 0, hit_hold = 0, hit_release = 0;
        aec_reset(&aec);
        {
            unsigned seed = 0xBEEF01u + (unsigned)r;
            int prev = aec.simple_mu_holdoff;
            for (int i = 0; i < 400; ++i) {
                int dt = ((i / 7) % 2) == 0;
                mu_step(&aec, dt ? 0.001f : 0.5f, dt ? 0.5f : 0.001f, &seed);
                int now = aec.simple_mu_holdoff;
                if (now > prev) hit_attack = 1;
                else if (now < prev) hit_hold = 1;
                else if (now == 0) hit_release = 1;
                else hit_attack = 1;
                prev = now;
            }
        }
        snprintf(what, sizeof(what),
                 "sr=%d fft=%d: attack/hold/release all reached "
                 "(%d/%d/%d)",
                 sr, fft, hit_attack, hit_hold, hit_release);
        CHECK(hit_attack && hit_hold && hit_release, what);

        aec_destroy(&aec);
    }
}

int main(void) {
    test_cola();
    test_impulse_linear_path();
    test_impulse_res_enabled();
    test_res_context_wola_identity();
    test_spatial_linear_context_matches_baseline();
    test_suppression_gain_dne_prefix_matches_get_gain();
    test_static_pool_matches_heap();
    test_mem_size_consistency();
    test_top_level_constant_retiming();
    test_adaptation_constant_retiming();
    test_retimed_constants_reach_the_audio_path();
    test_alpha_r_reaches_the_direct_pbfdkf_path();
    test_simple_mu_frozen_exception();
    test_mu_holdoff_rearm_guard();
    test_render_activity_first_observation();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
