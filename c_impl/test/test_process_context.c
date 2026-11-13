/* test_process_context.c -- split-entry processing regression test.
 *
 * aec_process() and the pair aec_process_linear() + aec_process_residual() run the same two stages (linear filter, then the
 * post-filter/RES block + power EMAs + convergence detection); aec_process() adds, with AEC_RESIDUAL_FULL, the gain, the
 * comfort noise and the library's own synthesis into `out`. This test proves three things a byte-equal WAV render alone
 * cannot:
 *   1. The two forms may be interleaved freely on ONE instance: they advance identical filter/detector state and differ
 *      only by the library's synthesis (its overlap-add tail and comfort-noise generator move only in aec_process()). An
 *      interleaved instance is compared bit-for-bit with one driven by aec_process() throughout, on the residual view of
 *      every hop (formed hop, error spectrum, rendering base, r2, gain); a stage that only one form advances makes the
 *      two diverge and fails here. Below FULL (ESTIMATE, STATE), where aec_process() synthesizes nothing and `out` is the
 *      linear residual, `out` itself is compared bit for bit on the aec_process() hops.
 *   2. EVERY field of the two views a caller could read (including the linear hop, far_spec, res_gain, saturation_level) is
 *      BYTE-IDENTICAL between two separately constructed, identically configured, identically fed instances, one driven by
 *      aec_process() and one by the split pair, across the three contracted grids x shadow on/off x AecConfig.residual
 *      FULL / ESTIMATE (a multi-lane wrapper runs its lanes without a gain). Every populated field is also checked finite.
 *   3. aec_reset() on a split-driven instance restores the state a fresh aec_create() would produce.
 *   4. AEC_RESIDUAL_STATE skips the residual-echo half of the post block (R^2, comfort noise, suppression gain). Nothing it
 *      skips feeds the linear filter, so against an ESTIMATE / FULL instance fed identically every linear output is
 *      byte-identical, while res_gain / r2 / comfort_noise read NULL.
 *
 * Build (standalone, from c_impl/):
 *   make -C ../../audiolib/common lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audiolib/common/include \
 *       -I../../audiolib/common \
 *       test/test_process_context.c $(find src -name '*.c') \
 *       $(make -s -C ../../audiolib/common print-lib-path) -lm \
 *       -o bin/test_process_context
 * Run:
 *   ./bin/test_process_context
 * Also wired into the Makefile: `make test-process-context`.
 */
#include "aec.h"
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

/* Linear stage + residual stage of one hop, then both views. */
static void split_hop(Aec* a, const float* mic, const float* ref) {
    int st = aec_process_linear(a, mic, ref);
    if (st == AEC_OK) st = aec_process_residual(a, NULL);
    if (st != AEC_OK) { printf("FAIL: split hop refused (%d)\n", st); g_fail++; }
}

static void both_views(const Aec* a, AecLinearView* lv, AecResidualView* rv) {
    memset(lv, 0, sizeof *lv); lv->struct_size = sizeof *lv;
    memset(rv, 0, sizeof *rv); rv->struct_size = sizeof *rv;
    if (aec_get_linear(a, lv) != AEC_OK || aec_get_residual(a, rv) != AEC_OK) {
        printf("FAIL: view refused\n"); g_fail++;
    }
}

static void split_config(AecConfig* cfg, int sample_rate, AecResidualMode mode) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, sample_rate);
    cfg->residual = mode;
}

/* fft_size=0 means "use the level's default grid for this sample_rate". */
static void split_config_grid(AecConfig* cfg, int sample_rate, int fft_size, AecResidualMode mode) {
    split_config(cfg, sample_rate, mode);
    if (fft_size > 0) cfg->fft_size = fft_size;
}

static int all_finite(const float* buf, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(buf[i])) return 0;
    return 1;
}

static int all_finite_complex(const Complex* buf, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(buf[i].r) || !isfinite(buf[i].i)) return 0;
    return 1;
}

/* Deterministic pseudo-random hop generator (no <random.h> dependency,
 * matches this repo's existing xorshift-in-tests style). */
static unsigned g_rng_state = 0x2AEC5EEDu;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu; /* [0,1) */
    return amp * (2.0f * u - 1.0f);
}

/* The two forms may be MIXED FREELY on one instance.
 *
 * aec_process() and the split pair advance the same filter and detector state and differ only by the library's synthesis
 * (overlap-add tail, comfort-noise generator), which moves only in aec_process(). So an instance whose hops alternate
 * between the two must expose, on EVERY hop, a residual view bit-identical to an instance that called aec_process() every
 * hop. Re-introducing any stateful stage that only one of the two forms advances would make the interleaved state diverge
 * and fail this. The burst assertion below keeps it from passing vacuously: the signal actually reaches the regime (output
 * peak above mic peak) in which such a stage would have acted. */
static void test_entry_points_may_be_freely_interleaved(void) {
    AecConfig cfg;
    split_config(&cfg, 16000, AEC_RESIDUAL_FULL);

    Aec ref_inst, mixed;
    CHECK(aec_create(&ref_inst, &cfg) == 0, "free-mixing test: aec_create(reference instance)");
    CHECK(aec_create(&mixed, &cfg) == 0, "free-mixing test: aec_create(interleaved instance)");

    int hop = aec_hop_size(&ref_inst);
    CHECK(hop > 0 && hop <= 4096, "free-mixing test: valid hop size");
    if (hop <= 0 || hop > 4096) { aec_destroy(&ref_inst); aec_destroy(&mixed); return; }

    float mic[4096], ref[4096], out_ref[4096], out_mixed[4096];
    int mismatches = 0, compared = 0, saw_overshoot = 0;
    g_rng_state = 0x2AEC5EEDu;
    for (int i = 0; i < 200; ++i) {
        /* Quiet baseline, loud burst frames 40-49, quiet tail; mic/ref are drawn independently so
         * the adaptive filter has no real echo path to converge onto, and the abrupt quiet->loud
         * transition forces a transient output overshoot above the near peak. */
        float amp = (i >= 40 && i < 50) ? 0.9f : 0.02f;
        float near_peak = 0.0f;
        for (int s = 0; s < hop; ++s) {
            mic[s] = rng_uniform(amp);
            ref[s] = 0.3f * rng_uniform(amp);
            float am = mic[s] < 0.0f ? -mic[s] : mic[s];
            if (am > near_peak) near_peak = am;
        }
        if (aec_process(&ref_inst, mic, ref, out_ref) != AEC_OK) { printf("FAIL: aec_process\n"); g_fail++; }
        if ((i & 1) == 0) {
            if (aec_process(&mixed, mic, ref, out_mixed) != AEC_OK) { printf("FAIL: aec_process\n"); g_fail++; }
        } else {
            split_hop(&mixed, mic, ref);
        }
        AecLinearView la, lb;
        AecResidualView ra, rb;
        both_views(&ref_inst, &la, &ra);
        both_views(&mixed, &lb, &rb);
        compared++;
        if (memcmp(ra.formed_hop, rb.formed_hop, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(ra.error_spec, rb.error_spec, (size_t)ra.n_freqs * sizeof(Complex)) != 0 ||
            memcmp(ra.base_spec, rb.base_spec, (size_t)ra.n_freqs * sizeof(Complex)) != 0 ||
            memcmp(ra.r2, rb.r2, (size_t)ra.n_freqs * sizeof(float)) != 0 ||
            memcmp(ra.res_gain, rb.res_gain, (size_t)ra.n_freqs * sizeof(float)) != 0 ||
            memcmp(la.linear_hop, lb.linear_hop, (size_t)hop * sizeof(float)) != 0 ||
            ra.usable_linear != rb.usable_linear || la.filter_converged != lb.filter_converged) mismatches++;
        float out_peak = 0.0f;
        for (int s = 0; s < hop; ++s) {
            float ao = out_ref[s] < 0.0f ? -out_ref[s] : out_ref[s];
            if (ao > out_peak) out_peak = ao;
        }
        if (out_peak > near_peak && near_peak > 1e-6f) saw_overshoot = 1;
    }

    CHECK(compared == 200, "free-mixing test: compared the expected number of hops");
    CHECK(mismatches == 0,
          "interleaving aec_process() and the split pair on one instance leaves a residual view bit-identical "
          "to using aec_process() throughout (no entry-point-specific state remains)");
    CHECK(saw_overshoot == 1,
          "free-mixing test: the signal actually reached the regime in which a stage would act "
          "(output peak above mic peak), so a re-introduced entry-point-specific stage would have been caught");

    aec_destroy(&ref_inst);
    aec_destroy(&mixed);
}

/* The same property where `out` IS comparable: below AEC_RESIDUAL_FULL the library applies and synthesizes nothing, so
 * aec_process() writes the linear residual and has no overlap-add tail or comfort-noise generator of its own. An instance
 * that alternates aec_process() and the split pair must then produce, on every aec_process() hop, `out` bit-identical to an
 * instance that called aec_process() every hop (and the same views on every hop). */
static void test_entry_points_interleave_below_full(AecResidualMode mode) {
    AecConfig cfg;
    split_config(&cfg, 16000, mode);

    Aec ref_inst, mixed;
    char label[160];
    snprintf(label, sizeof(label), "interleave mode=%d: aec_create", (int)mode);
    CHECK(aec_create(&ref_inst, &cfg) == 0 && aec_create(&mixed, &cfg) == 0, label);

    int hop = aec_hop_size(&ref_inst);
    if (hop <= 0 || hop > 4096) { aec_destroy(&ref_inst); aec_destroy(&mixed); return; }

    float mic[4096], ref[4096], out_ref[4096], out_mixed[4096];
    int out_mismatch = 0, view_mismatch = 0, out_compared = 0;
    g_rng_state = 0x2AEC5EEDu;
    for (int i = 0; i < 200; ++i) {
        float amp = (i >= 40 && i < 50) ? 0.9f : 0.02f;
        for (int s = 0; s < hop; ++s) {
            mic[s] = rng_uniform(amp);
            ref[s] = 0.3f * rng_uniform(amp);
        }
        if (aec_process(&ref_inst, mic, ref, out_ref) != AEC_OK) { printf("FAIL: aec_process\n"); g_fail++; }
        if ((i & 1) == 0) {
            if (aec_process(&mixed, mic, ref, out_mixed) != AEC_OK) { printf("FAIL: aec_process\n"); g_fail++; }
            out_compared++;
            if (memcmp(out_ref, out_mixed, (size_t)hop * sizeof(float)) != 0) out_mismatch++;
        } else {
            split_hop(&mixed, mic, ref);
        }
        AecLinearView la, lb;
        AecResidualView ra, rb;
        both_views(&ref_inst, &la, &ra);
        both_views(&mixed, &lb, &rb);
        if (memcmp(ra.formed_hop, rb.formed_hop, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(ra.error_spec, rb.error_spec, (size_t)ra.n_freqs * sizeof(Complex)) != 0 ||
            memcmp(ra.base_spec, rb.base_spec, (size_t)ra.n_freqs * sizeof(Complex)) != 0 ||
            memcmp(la.linear_hop, lb.linear_hop, (size_t)hop * sizeof(float)) != 0 ||
            ra.usable_linear != rb.usable_linear || la.filter_converged != lb.filter_converged) view_mismatch++;
    }
    snprintf(label, sizeof(label),
             "interleave mode=%d: out of the aec_process() hops (%d) is bit-identical to aec_process() throughout", (int)mode,
             out_compared);
    CHECK(out_compared == 100 && out_mismatch == 0, label);
    snprintf(label, sizeof(label), "interleave mode=%d: views bit-identical on every hop", (int)mode);
    CHECK(view_mismatch == 0, label);

    aec_destroy(&ref_inst);
    aec_destroy(&mixed);
}

/* Two separately constructed, identically configured, identically fed instances, one driven exclusively by aec_process()
 * and one by the split pair, must expose byte-identical views on every hop across every field a real caller (mono
 * pipeline, multi-lane wrapper) reads, with the residual mode FULL and ESTIMATE. Every populated field is also checked
 * finite, since a byte-equal comparison of two NaN payloads would otherwise pass vacuously. */
static void test_split_matches_full_process(
        int sample_rate, int fft_size, int enable_shadow, AecResidualMode mode) {
    AecConfig cfg;
    split_config_grid(&cfg, sample_rate, fft_size, mode);
    cfg.enable_shadow = enable_shadow;
    const int with_gain = (mode == AEC_RESIDUAL_FULL);

    Aec via_full, via_split;
    char label[160];
    snprintf(label, sizeof(label), "split-parity sr=%d/fft=%d/shadow=%d/mode=%d: aec_create(via_full)",
             sample_rate, cfg.fft_size, enable_shadow, (int)mode);
    CHECK(aec_create(&via_full, &cfg) == 0, label);
    snprintf(label, sizeof(label), "split-parity sr=%d/fft=%d/shadow=%d/mode=%d: aec_create(via_split)",
             sample_rate, cfg.fft_size, enable_shadow, (int)mode);
    CHECK(aec_create(&via_split, &cfg) == 0, label);

    int hop = aec_hop_size(&via_full);
    if (hop <= 0 || hop > 4096) { aec_destroy(&via_full); aec_destroy(&via_split); return; }

    float mic[4096], ref[4096], out[4096];
    g_rng_state = 0x51DEC0DEu;
    int all_match = 1;
    int all_finite_ok = 1;
    int res_gain_null_as_expected = 1;
    for (int i = 0; i < 150; ++i) {
        float amp = (i >= 50 && i < 55) ? 0.8f : 0.05f; /* occasional burst too */
        for (int s = 0; s < hop; ++s) {
            float v = rng_uniform(amp);
            mic[s] = v;
            ref[s] = 0.6f * v;
        }
        if (aec_process(&via_full, mic, ref, out) != AEC_OK) { printf("FAIL: aec_process\n"); g_fail++; }
        split_hop(&via_split, mic, ref);

        AecLinearView la, lb;
        AecResidualView ca, cb;
        both_views(&via_full, &la, &ca);
        both_views(&via_split, &lb, &cb);

        if (ca.n_freqs != cb.n_freqs || ca.hop_size != cb.hop_size) { all_match = 0; break; }
        int K = ca.n_freqs;
        if (memcmp(ca.formed_hop, cb.formed_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(la.linear_hop, lb.linear_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(ca.far_spec, cb.far_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.error_spec, cb.error_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.echo_spec, cb.echo_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.near_spec, cb.near_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.base_spec, cb.base_spec, (size_t)K * sizeof(Complex)) != 0 ||
            ca.guard_active != cb.guard_active) { all_match = 0; break; }
        if (memcmp(ca.r2, cb.r2, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(ca.comfort_noise, cb.comfort_noise, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        /* res_gain exists only at AEC_RESIDUAL_FULL; there it must be byte-identical like every other per-bin field. */
        if ((ca.res_gain == NULL) != (cb.res_gain == NULL)) { all_match = 0; break; }
        if ((ca.res_gain == NULL) == with_gain) res_gain_null_as_expected = 0;
        if (ca.res_gain != NULL &&
            memcmp(ca.res_gain, cb.res_gain, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        if (la.far_power != lb.far_power || la.erle_factor != lb.erle_factor
                || la.dt_indicator != lb.dt_indicator || la.divergence != lb.divergence
                || la.saturation_level != lb.saturation_level
                || la.erl_estimate != lb.erl_estimate || la.shadow_dt != lb.shadow_dt
                || la.is_stationary_dt != lb.is_stationary_dt
                || la.filter_converged != lb.filter_converged
                || la.filter_once_converged != lb.filter_once_converged
                || la.epc_active != lb.epc_active
                || ca.usable_linear != cb.usable_linear
                || ca.res_floor_protect != cb.res_floor_protect) { all_match = 0; break; }

        if (!all_finite(ca.formed_hop, hop) || !all_finite(la.linear_hop, hop)
                || !all_finite_complex(ca.far_spec, K) || !all_finite_complex(ca.error_spec, K)
                || !all_finite_complex(ca.echo_spec, K) || !all_finite_complex(ca.near_spec, K)
                || !all_finite_complex(ca.base_spec, K)
                || !all_finite(ca.r2, K) || !all_finite(ca.comfort_noise, K)
                || (ca.res_gain != NULL && !all_finite(ca.res_gain, K))
                || !isfinite(la.far_power) || !isfinite(la.erle_factor)
                || !isfinite(la.dt_indicator) || !isfinite(la.divergence)
                || !isfinite(la.saturation_level) || !isfinite(la.erl_estimate)
                || !isfinite(la.shadow_dt)) { all_finite_ok = 0; break; }
    }

    snprintf(label, sizeof(label),
             "split-parity sr=%d/fft=%d/shadow=%d/mode=%d: both views byte-identical every hop, "
             "aec_process() vs the split pair",
             sample_rate, cfg.fft_size, enable_shadow, (int)mode);
    CHECK(all_match, label);

    snprintf(label, sizeof(label),
             "split-parity sr=%d/fft=%d/shadow=%d/mode=%d: every populated view field stays finite",
             sample_rate, cfg.fft_size, enable_shadow, (int)mode);
    CHECK(all_finite_ok, label);

    snprintf(label, sizeof(label),
             "split-parity sr=%d/fft=%d/shadow=%d/mode=%d: res_gain is NULL iff the mode is below FULL",
             sample_rate, cfg.fft_size, enable_shadow, (int)mode);
    CHECK(res_gain_null_as_expected, label);

    aec_destroy(&via_full);
    aec_destroy(&via_split);
}

/* AEC_RESIDUAL_STATE skips the residual-echo half of the post block (R^2, comfort noise,
 * suppression gain). Nothing it skips feeds the linear filter, so against an instance configured as
 * `base_mode` (FULL or ESTIMATE) and fed identically, every linear output must be byte-identical on every hop,
 * including the AecLinearView a downstream model reads and the usable-linear verdict, while
 * res_gain / r2 / comfort_noise read NULL. The input carries a delayed, scaled copy of the far end
 * plus near-end bursts, so the filter converges, the alignment locks and the verdicts change during
 * the run; a skipped step that did feed back would diverge. */
static void test_state_mode_matches_full(
        int sample_rate, int fft_size, int enable_shadow, AecResidualMode base_mode) {
    AecConfig cfg_full, cfg_lin;
    split_config_grid(&cfg_full, sample_rate, fft_size, base_mode);
    cfg_full.enable_shadow = enable_shadow;
    cfg_lin = cfg_full;
    cfg_lin.residual = AEC_RESIDUAL_STATE;

    Aec full, lin;
    char label[160];
    snprintf(label, sizeof(label), "state-mode sr=%d/fft=%d/shadow=%d/base=%d: aec_create(base)",
             sample_rate, cfg_full.fft_size, enable_shadow, (int)base_mode);
    CHECK(aec_create(&full, &cfg_full) == 0, label);
    snprintf(label, sizeof(label), "state-mode sr=%d/fft=%d/shadow=%d/base=%d: aec_create(STATE)",
             sample_rate, cfg_full.fft_size, enable_shadow, (int)base_mode);
    CHECK(aec_create(&lin, &cfg_lin) == 0, label);

    int hop = aec_hop_size(&full);
    if (hop <= 0 || hop > 1024) { aec_destroy(&full); aec_destroy(&lin); return; }

    enum { HOPS = 600, DELAY_HOPS = 3 };
    static float far_hist[DELAY_HOPS + 1][1024];
    float mic[1024], far_now[1024];
    int all_match = 1, nulls_ok = 1, usable_seen[2] = {0, 0}, hops_checked = 0;
    g_rng_state = 0x71A3C0DEu;
    for (int i = 0; i < HOPS; ++i) {
        float burst = (i >= 300 && i < 330) ? 0.5f : 0.02f;   /* double-talk-ish burst */
        for (int s = 0; s < hop; ++s) far_now[s] = rng_uniform(0.3f);
        memmove(far_hist[1], far_hist[0], DELAY_HOPS * sizeof(far_hist[0]));
        memcpy(far_hist[0], far_now, (size_t)hop * sizeof(float));
        for (int s = 0; s < hop; ++s)
            mic[s] = 0.5f * far_hist[DELAY_HOPS][s] + rng_uniform(burst);

        split_hop(&full, mic, far_now);
        split_hop(&lin,  mic, far_now);

        AecLinearView la, lb;
        AecResidualView ca, cb;
        both_views(&full, &la, &ca);
        both_views(&lin, &lb, &cb);
        int K = ca.n_freqs;

        if (memcmp(ca.formed_hop, cb.formed_hop, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(la.linear_hop, lb.linear_hop, (size_t)hop * sizeof(float)) != 0 ||
            memcmp(ca.far_spec,   cb.far_spec,   (size_t)K * sizeof(Complex)) != 0 ||
            memcmp(ca.error_spec, cb.error_spec, (size_t)K * sizeof(Complex)) != 0 ||
            memcmp(ca.echo_spec,  cb.echo_spec,  (size_t)K * sizeof(Complex)) != 0 ||
            memcmp(ca.near_spec,  cb.near_spec,  (size_t)K * sizeof(Complex)) != 0 ||
            memcmp(ca.base_spec,  cb.base_spec,  (size_t)K * sizeof(Complex)) != 0 ||
            ca.guard_active != cb.guard_active ||
            ca.usable_linear != cb.usable_linear ||
            la.far_power != lb.far_power || la.erle_factor != lb.erle_factor ||
            la.dt_indicator != lb.dt_indicator || la.divergence != lb.divergence ||
            la.saturation_level != lb.saturation_level ||
            la.erl_estimate != lb.erl_estimate || la.shadow_dt != lb.shadow_dt ||
            la.is_stationary_dt != lb.is_stationary_dt ||
            la.filter_converged != lb.filter_converged ||
            la.filter_once_converged != lb.filter_once_converged ||
            la.epc_active != lb.epc_active ||
            la.delay_samples != lb.delay_samples ||
            la.delay_confidence != lb.delay_confidence ||
            la.delay_state != lb.delay_state || la.generation != lb.generation ||
            memcmp(la.aligned_far_hop, lb.aligned_far_hop, (size_t)hop * sizeof(float)) != 0) {
            all_match = 0; break;
        }
        if (cb.res_gain != NULL || cb.r2 != NULL || cb.comfort_noise != NULL ||
            cb.res_floor_protect != 0 ||
            ca.r2 == NULL || ca.comfort_noise == NULL) nulls_ok = 0;
        usable_seen[ca.usable_linear != 0] = 1;
        ++hops_checked;
    }

    snprintf(label, sizeof(label),
             "state-mode sr=%d/fft=%d/shadow=%d/base=%d: every linear output and the AecLinearView "
             "are byte-identical to the base mode on all hops",
             sample_rate, cfg_full.fft_size, enable_shadow, (int)base_mode);
    CHECK(all_match && hops_checked == HOPS, label);
    snprintf(label, sizeof(label),
             "state-mode sr=%d/fft=%d/shadow=%d/base=%d: res_gain/r2/comfort_noise are NULL and "
             "res_floor_protect is 0 (the base mode still exposes r2/comfort_noise)",
             sample_rate, cfg_full.fft_size, enable_shadow, (int)base_mode);
    CHECK(nulls_ok, label);
    snprintf(label, sizeof(label),
             "state-mode sr=%d/fft=%d/shadow=%d/base=%d: the run exercised both usable_linear verdicts "
             "(the comparison is not vacuous)",
             sample_rate, cfg_full.fft_size, enable_shadow, (int)base_mode);
    CHECK(usable_seen[0] && usable_seen[1], label);

    aec_destroy(&full);
    aec_destroy(&lin);
}

/* The split pair must also work correctly across a reset (state fully re-zeroed, next hop behaves like a fresh
 * instance) -- exercises the same reset path aec_process() itself already relies on. */
static void test_split_survives_reset(void) {
    AecConfig cfg;
    split_config(&cfg, 16000, AEC_RESIDUAL_FULL);

    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "reset test: aec_create()");
    int hop = aec_hop_size(&a);
    if (hop <= 0 || hop > 4096) { aec_destroy(&a); return; }

    float mic[4096], ref[4096];
    g_rng_state = 0x8ACE0001u;
    for (int i = 0; i < 30; ++i) {
        for (int s = 0; s < hop; ++s) { mic[s] = rng_uniform(0.3f); ref[s] = rng_uniform(0.2f); }
        split_hop(&a, mic, ref);
    }
    CHECK(a.frame_count == 30, "reset test: frame_count reaches 30 via the split pair alone");

    aec_reset(&a);
    CHECK(a.frame_count == 0, "reset test: aec_reset() zeroes frame_count");

    for (int i = 0; i < 5; ++i) {
        for (int s = 0; s < hop; ++s) { mic[s] = rng_uniform(0.3f); ref[s] = rng_uniform(0.2f); }
        split_hop(&a, mic, ref);
    }
    CHECK(a.frame_count == 5, "reset test: the split pair runs cleanly post-reset");

    aec_destroy(&a);
}

int main(void) {
    test_entry_points_may_be_freely_interleaved();
    test_entry_points_interleave_below_full(AEC_RESIDUAL_ESTIMATE);
    test_entry_points_interleave_below_full(AEC_RESIDUAL_STATE);

    /* The three contracted grids (see pipelines/README.md "Parameter Alignment") x shadow on/off x residual mode
     * FULL / ESTIMATE (a multi-lane wrapper runs its lanes below FULL). */
    static const int grids[][1] = { {0}, {512}, {0} };
    static const int rates[]    = { 16000, 16000, 48000 };
    static const AecResidualMode modes[] = { AEC_RESIDUAL_FULL, AEC_RESIDUAL_ESTIMATE };
    for (int g = 0; g < 3; ++g)
        for (int shadow = 0; shadow <= 1; ++shadow)
            for (int m = 0; m < 2; ++m)
                test_split_matches_full_process(rates[g], grids[g][0], shadow, modes[m]);

    for (int g = 0; g < 3; ++g)
        for (int shadow = 0; shadow <= 1; ++shadow)
            for (int m = 0; m < 2; ++m)
                test_state_mode_matches_full(rates[g], grids[g][0], shadow, modes[m]);

    test_split_survives_reset();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
