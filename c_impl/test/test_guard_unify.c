/* test_guard_unify.c -- one AEC algorithm behind every API split.
 *
 * What it proves. Rendering, synthesis and every compared quantity go through the PUBLIC API only (aec_process,
 * aec_process_context, aec_get_res_context, aec_get_output_base, aec_post_reset_serial, aec_cng_fill, aec_debug_status,
 * audiolib ac_istft_*; no aec_post_apply_output, no private reset counter). Internal state is only READ, to classify
 * which branch decided a delay event (near-end-recent hold, inst-ERLE ring, reject list) and to check that the
 * comfort-noise LCG was not reseeded on a revert; the forced-verdict scenario (9) is the one place that WRITES internal
 * state (the filter-quality counters, the same way on every instance, like a patched verdict):
 *   1. Full mode == split mode, bit for bit. aec_process() (enable_res=1) against an external application that runs
 *      aec_process_context() and renders base_spec * res_gain (+ aec_cng_fill + ac_istft_push), clearing its OLA tail
 *      and reseeding its CNG to AEC_CNG_SEED BEFORE the CNG draw and the ac_istft_push of any hop on which
 *      aec_post_reset_serial() moved. Same for a full-config instance driven through aec_process_context(). No
 *      tolerance anywhere in this comparison.
 *   2. Every mode publishes the same analysis seam and the same rendering base: enable_res, return_res_context-only,
 *      AEC_CONTEXT_NO_GAIN and AEC_CONTEXT_LINEAR_ONLY lanes agree on error_spec / echo_spec / near_spec / formed_hop /
 *      usable_linear / guard decision / base_spec / serial each hop.
 *   3. The rendering base obeys its definition (guard on: base_spec == near_spec; off: base_spec == error_spec) and the
 *      guard decision agrees with an independent double-precision recomputation, except inside a relative 1e-6 tie band
 *      where the float32 pairwise reduction may legitimately round differently (counted, reported, and required to be 0
 *      outside the band).
 *   4. The analysis seam keeps its pairing on EVERY hop, guard hops included and with no exception: unity ac_istft of
 *      error_spec reproduces the previous formed_hop (one framing hop of delay), and after a hard reset it reproduces
 *      zeros (the reset clears the previous-hop memory with the OLA).
 *   5. post_reset_serial moves on hard resets only. Every delay event of the context instance is classified from the
 *      state that decides it (Path A first acquisition: warm tap transfer / DT-aware soft acquisition / hard; Path B
 *      shift: DT-aware soft realign / hard; bad-relock revert, detected both by its public delay_samples signature and
 *      by the reject list) and the serial must move by exactly 1 on the hard branches and 0 on every other one;
 *      aec_reset() moves it by 1, an external realign never. A serial move without a hard event fails.
 *   6. The bad-relock revert is soft: serial unchanged, the full-mode comfort-noise LCG continues (not reseeded), and an
 *      application that cleared its OLA tail and reseeded its CNG on the revert hop would NOT match aec_process()
 *      there (so the identity above really covers the kept OLA/CNG state). Clip-list rows can require a minimum number
 *      of reverts (fifth column), so the revert path is reached on every grid.
 *   7. aec_cng_fill() against an independent scalar reference (no NR and NR paths), the draw schedule, and agreement of
 *      the library's internal CNG with the public helper through the full-vs-split identity above.
 *   8. The near-threshold float32 fixture for the guard rule (equal energy, +/- 1 ulp), flag/usable gating, and a
 *      reduction-order fixture family: multi-bin E / Y pairs with an exactly tied double energy whose decision under the
 *      float32 pairwise reduction differs from the sequential one (expected decisions from an independent pairwise
 *      replica in this file), in both directions.
 *   9. Forced verdict switching (filter-quality counters written before each hop, identically on every instance) on a
 *      stream whose residual is louder than the capture: many hops where the usable verdict flips while E > Y, so a
 *      guard evaluated on the previous hop's verdict fails there.
 * Scenarios: synthetic (init, never usable, unusable -> usable, moving echo path with delay changes, DT-aware soft
 * acquisition and soft shifts, aec_reset and replay, external realign, comfort noise off, forced verdict switching) on
 * 16k/256, 16k/512, 48k/1024, plus the real blind-set clips of test/guard_unify_clips.tsv (never-usable clips and
 * bad-relock revert clips included) which must exercise guard firing, switching, hard resets and reverts.
 *
 * Usage: test_guard_unify [wav_root] [clips.tsv]    (defaults ../wav and test/guard_unify_clips.tsv)
 *   The blind-set corpora are not in git. When a clip's corpus directory (first path component under wav_root) is
 *   absent, its rows are skipped with a loud SKIP banner and the real-clip coverage checks of that grid are not run;
 *   the synthetic part always runs and the exit status reflects it. When the corpus directory is present, a missing or
 *   unreadable clip, a coverage shortfall or a missing revert is a failure.
 *   GUARD_UNIFY_ALL=<cases.tsv> additionally runs every case of such a list (stem, bucket, mic, ref, sr columns) at the
 *   grid given by GUARD_UNIFY_ALL_FFT; there a missing wav is a failure. */
#include "aec.h"
#include "aec_post.h"
#include "aec_res.h"
#include "aec_balanced_config.h"
#include "audio_stft.h"
#include "utility/fast_math.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } \
        else         { g_pass++; } \
    } while (0)

/* ------------------------------------------------------------------------------------------------------------------ */
/* deterministic signals                                                                                              */
/* ------------------------------------------------------------------------------------------------------------------ */

static uint32_t g_rs = 0x2545F491u;
static float rnd(void) {                      /* uniform in [-1, 1) */
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 17; g_rs ^= g_rs << 5;
    return (float)(g_rs & 0xFFFFFFu) / (float)0x800000u - 1.0f;
}

typedef struct {
    const char* name;
    int sr, fft, hop, nh;
    float* mic;
    float* far;
    int* reset_at;      /* hops (ascending) on which every instance gets aec_reset() BEFORE the process call, or NULL */
    int n_reset;
    int* realign_at;    /* hops on which aec_apply_external_realign(realign_delta[i]) is applied to every instance */
    int* realign_delta;
    int n_realign;
    int external_aligned;
    int enable_cng;
    int warm_transfer_off;  /* cfg.delay_acquire_warm_transfer = 0: a Path A acquisition is DT-aware soft or hard */
    int force_verdict;      /* > 0: usable verdict forced to ((hop / force_verdict) % 3 == 0) on every instance */
    int replay_from;    /* >= 0: after the last reset event, hops [0, replay_len) are replayed and compared to pass 1 */
    int replay_len;
} Stream;

/* Piecewise echo path: delay changes at 1/3 and 2/3 of the stream. gain/colouring is fixed. */
static void gen_echo_stream(Stream* s, int speech_far, int echo, int near_burst, int silent_head_hops) {
    int n = s->nh * s->hop, i, seg = s->sr / 4;
    float env = 0.2f, lp = 0.0f;
    float* nearv = (float*)calloc((size_t)n, sizeof(float));
    int d1 = 480 * (s->sr / 16000), d2 = 1100 * (s->sr / 16000), d3 = 240 * (s->sr / 16000);
    float nlp = 0.0f;
    s->mic = (float*)calloc((size_t)n, sizeof(float));
    s->far = (float*)calloc((size_t)n, sizeof(float));
    for (i = 0; i < n; ++i) {
        float v;
        if (i % seg == 0) { float r = rnd(); env = speech_far ? (r > 0.3f ? 0.35f : (r > -0.4f ? 0.12f : 0.0f)) : 0.45f; }
        lp = 0.6f * lp + 0.4f * rnd();
        v = env * lp * 2.0f;
        s->far[i] = (i < silent_head_hops * s->hop) ? 0.0f : v;
        nlp = 0.7f * nlp + 0.3f * rnd();
        nearv[i] = (near_burst && i > n / 3 && i < n / 3 + n / 6) ? 0.25f * nlp * 2.0f : 0.004f * nlp;
    }
    for (i = 0; i < n; ++i) {
        int d = i < n / 3 ? d1 : (i < 2 * n / 3 ? d2 : d3);
        float e = 0.0f;
        if (echo) {
            if (i - d >= 0) e += 0.45f * s->far[i - d];
            if (i - d - 37 >= 0) e += 0.2f * s->far[i - d - 37];
        }
        s->mic[i] = nearv[i] + e;
    }
    free(nearv);
}

/* Double talk that keeps the DT-aware near-end-recent hold armed: every 0.8 s a 0.15 s window with a quiet but active
 * far end and a loud near end, otherwise a loud far end whose echo (gain echo_gain) follows the same piecewise delay as
 * gen_echo_stream. Every delay acquisition and shift then happens while the hold is armed (soft branches). */
static void gen_dt_soft_stream(Stream* s, float echo_gain) {
    int n = s->nh * s->hop, i, per = (int)(0.8f * (float)s->sr), win = (int)(0.15f * (float)s->sr);
    int d1 = 480 * (s->sr / 16000), d2 = 1100 * (s->sr / 16000), d3 = 240 * (s->sr / 16000);
    float lp = 0.0f, nlp = 0.0f;
    s->mic = (float*)calloc((size_t)n, sizeof(float));
    s->far = (float*)calloc((size_t)n, sizeof(float));
    for (i = 0; i < n; ++i) {
        lp = 0.6f * lp + 0.4f * rnd();
        s->far[i] = (i % per) < win ? 0.05f * rnd() : lp;
    }
    for (i = 0; i < n; ++i) {
        int d = i < n / 3 ? d1 : (i < 2 * n / 3 ? d2 : d3);
        float e = 0.0f;
        if (i - d >= 0) e += echo_gain * s->far[i - d];
        if (i - d - 37 >= 0) e += 0.44f * echo_gain * s->far[i - d - 37];
        nlp = 0.7f * nlp + 0.3f * rnd();
        s->mic[i] = e + ((i % per) < win ? 0.8f * nlp : 0.004f * nlp);
    }
}

/* Far end and capture uncorrelated (no echo path at all): the linear filter only adds energy, so E > Y on most hops. */
static void gen_uncorrelated_stream(Stream* s) {
    int n = s->nh * s->hop, i;
    float lp = 0.0f;
    s->mic = (float*)calloc((size_t)n, sizeof(float));
    s->far = (float*)calloc((size_t)n, sizeof(float));
    for (i = 0; i < n; ++i) {
        lp = 0.6f * lp + 0.4f * rnd();
        s->far[i] = 0.8f * lp;
        s->mic[i] = 0.1f * rnd();
    }
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* the external application                                                                                           */
/* ------------------------------------------------------------------------------------------------------------------ */

typedef struct {
    AcIstft* is;
    void* mem;
    uint32_t rng;
    uint32_t serial;
    int cng;
    Complex* spec;
} App;

/* The application synthesizes with the library's synthesis window (aec_synthesis_window), borrowed by ac_istft; the
 * audiolib default sqrt-Hann differs from that table in the last float bit. */
static int app_init(App* ap, int fft, int cng, const Aec* a) {
    AcStftConfig sc;
    size_t sz;
    const float* win = NULL;
    int win_n = 0;
    if (aec_synthesis_window(a, &win, &win_n) != 0 || win_n != fft) return -1;
    ac_stft_config_default(&sc, fft);
    sc.window = win;
    sz = ac_istft_get_mem_size(&sc);
    if (!sz || posix_memalign(&ap->mem, 16, sz) != 0) return -1;
    ap->is = ac_istft_init(ap->mem, sz, &sc);
    if (!ap->is) return -1;
    ap->rng = AEC_CNG_SEED;
    ap->serial = aec_post_reset_serial(a);
    ap->cng = cng;
    ap->spec = (Complex*)calloc((size_t)(fft / 2 + 1), sizeof(Complex));
    return 0;
}
static void app_free(App* ap) { free(ap->mem); free(ap->spec); }

/* One hop of external rendering. Only public getters. force_reset makes the application clear its OLA tail and reseed
 * its CNG on this hop although the serial did not move (the counterfactual of a hard reset). */
static int app_synth(App* ap, const Aec* a, float* out, int force_reset) {
    AecResContext c;
    AecOutputBase ob;
    uint32_t s = aec_post_reset_serial(a);
    int k, n;
    if (s != ap->serial || force_reset) {   /* before the first CNG draw and before the ac_istft_push of this hop */
        ac_istft_reset(ap->is);
        ap->rng = AEC_CNG_SEED;
        ap->serial = s;
    }
    aec_get_res_context(a, &c);
    memset(&ob, 0, sizeof ob);
    ob.struct_size = sizeof ob;
    if (aec_get_output_base(a, &ob) != 0) return -1;
    n = ob.n_freqs;
    for (k = 0; k < n; ++k) {
        ap->spec[k].r = ob.base_spec[k].r * c.res_gain[k];
        ap->spec[k].i = ob.base_spec[k].i * c.res_gain[k];
    }
    if (ap->cng) aec_cng_fill(&ap->rng, ap->spec, c.comfort_noise, c.res_gain, NULL, 0.0f, n);
    ac_istft_push(ap->is, ap->spec, out);
    return 0;
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* one run                                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

/* Unity WOLA of the sqrt-Hann STFT pair reproduces the previous formed hop to float rounding (measured 1.2e-7 at the
 * 115e10c baseline); a guard hop that wrongly published a different spectrum is off by ~1e-1. */
#define PROBE_TOL 5e-7

typedef struct {
    long hops, guard_hops, guard_toggles, serial_changes, first_usable;
    long guard_before_usable, guard_after_usable;
    long full_vs_split_diff_hops, full_vs_ctxfull_diff_hops, seam_diff_hops, lane_diff_hops, base_def_bad_hops;
    long oracle_mismatch, oracle_ties, serial_lane_diff, realign_serial_moves, realign_warm, realign_cold;
    long probe_bad_hops, probe_guard_hops_checked;
    double probe_max_err;
    long replay_diff_hops;
    long first_bad_hop;
    /* delay events of the context instance, classified from the deciding state; serial_bad: serial delta != expected */
    long pa_warm, pa_soft, pa_hard, pb_soft, pb_hard, reverts, serial_bad, unexplained_serial;
    long revert_sig_bad, revert_cng_bad, revert_cf_checked, revert_cf_same, revert_split_diff;
    long first_revert_hop;
    long verdict_flips_e_gt_y;     /* hops whose usable verdict differs from the previous hop's while E > Y (oracle) */
} Stats;

static void make_cfg(AecConfig* c, const Stream* s, int role) {
    aec_config_from_level(c, AEC_LEVEL_BALANCED, s->sr);
    c->fft_size = s->fft;
    c->enable_cng = s->enable_cng;
    if (s->warm_transfer_off) c->delay_acquire_warm_transfer = 0;
    if (s->external_aligned) { c->delay_mode = AEC_DELAY_EXTERNAL_ALIGNED; c->enable_delay_est = 0; }
    switch (role) {
    case 0: c->enable_res = 1; c->return_res_context = 0; break;                       /* F: full */
    case 1: c->enable_res = 0; c->return_res_context = 1; break;                       /* K: context only */
    case 2: c->enable_res = 1; c->return_res_context = 1; break;                       /* N: full config, context API */
    case 3: c->enable_res = 0; c->return_res_context = 1; c->spatial_linear_context = AEC_CONTEXT_NO_GAIN; break;
    default: c->enable_res = 0; c->return_res_context = 1; c->spatial_linear_context = AEC_CONTEXT_LINEAR_ONLY; break;
    }
}

static int bits_eq_f(const float* a, const float* b, int n) { return memcmp(a, b, (size_t)n * sizeof(float)) == 0; }
static int bits_eq_c(const Complex* a, const Complex* b, int n) { return memcmp(a, b, (size_t)n * sizeof(Complex)) == 0; }

/* Independent guard recomputation: float32 magnitudes and squares as the rule is specified, double accumulation. */
static double energy_abs_sq(const Complex* x, int n) {
    double e = 0.0;
    int k;
    for (k = 0; k < n; ++k) {
        float m = sqrtf(x[k].r * x[k].r + x[k].i * x[k].i);
        e += (double)(m * m);
    }
    return e;
}

#define NROLE 5

/* n LCG steps of the comfort-noise generator (the documented schedule: one step per drawn bin). */
static uint32_t lcg_advance(uint32_t seed, int n) {
    int i;
    for (i = 0; i < n; ++i) seed = (seed * 69069u + 1u) & 0x7FFFFFFFu;
    return seed;
}

/* Inst-ERLE ring peak over the last 15 entries, read from the instance as the Path A warm-transfer test reads it. */
static float erle_ring_peak15(const Aec* a) {
    int n = a->erle_slope_len < 15 ? a->erle_slope_len : 15, i, idx = a->erle_slope_head, got = 0;
    float m = 0.0f;
    for (i = 0; i < n; ++i) {
        idx = (idx - 1 + a->erle_slope_cap) % a->erle_slope_cap;
        if (!got || a->erle_slope_buf[idx] > m) { m = a->erle_slope_buf[idx]; got = 1; }
    }
    return m;
}

/* Forces the filter-quality verdict of the coming hop: counters far past both gates and convergence seen (usable), or
 * the since-reset counter at 0 (unusable for reset_hops more hops). */
static void force_verdict(Aec* a, int usable) {
    FilteringQualityAnalyzer* q = &a->a3_state.filter_quality;
    q->filter_update_blocks_since_start = 1 << 20;
    q->filter_update_blocks_since_reset = usable ? (1 << 20) : 0;
    q->convergence_seen = 1;
}

static void run_stream(const Stream* s, Stats* st) {
    Aec* A[NROLE];
    App app_k, app_n, app_x;
    AcIstft* probe = NULL;
    void* probe_mem = NULL;
    AcStftConfig psc;
    int hop = s->hop, nb = s->fft / 2 + 1, role, h, k, ev = 0, ra = 0;
    float *oF, *oK, *oN, *oP, *oX, *prev_formed;
    float* pass1 = NULL;
    uint32_t last_serial = 0, probe_serial = 0;
    int prev_guard = 0, prev_usable = -1, cf_active = 1;
    int replay_phase = 0;
    int last_pb_hop = -1, last_pb_old = -2, last_pb_new = -2;   /* last Path B shift of the context instance */
    memset(st, 0, sizeof *st);
    st->first_usable = -1;
    st->first_bad_hop = -1;
    st->first_revert_hop = -1;
    for (role = 0; role < NROLE; ++role) {
        AecConfig c;
        make_cfg(&c, s, role);
        A[role] = (Aec*)calloc(1, sizeof(Aec));
        if (aec_create(A[role], &c) != 0) { CHECK(0, "%s: create role %d", s->name, role); return; }
    }
    CHECK(aec_hop_size(A[0]) == hop, "%s: hop size", s->name);
    if (app_init(&app_k, s->fft, s->enable_cng, A[1]) || app_init(&app_n, s->fft, s->enable_cng, A[2])
        || app_init(&app_x, s->fft, s->enable_cng, A[1])) {
        CHECK(0, "%s: app init", s->name);
        return;
    }
    ac_stft_config_default(&psc, s->fft);
    {
        size_t sz = ac_istft_get_mem_size(&psc);
        if (posix_memalign(&probe_mem, 16, sz) != 0) return;
        probe = ac_istft_init(probe_mem, sz, &psc);
    }
    oF = (float*)calloc((size_t)hop, sizeof(float));
    oK = (float*)calloc((size_t)hop, sizeof(float));
    oN = (float*)calloc((size_t)hop, sizeof(float));
    oP = (float*)calloc((size_t)hop, sizeof(float));
    oX = (float*)calloc((size_t)hop, sizeof(float));
    prev_formed = (float*)calloc((size_t)hop, sizeof(float));
    pass1 = (float*)calloc((size_t)s->nh * hop, sizeof(float));

    for (h = 0; h < s->nh + (s->replay_from >= 0 ? s->replay_len : 0); ++h) {
        const float* mic;
        const float* far;
        AecResContext cF, cK, cN, cG, cL;
        AecOutputBase oBF, oBK, oBN, oBG, oBL;
        AecDebugStatus ds;
        int src_h = h, bad = 0, d0, d1, rej0, warm_pred, revert_hop = 0;
        uint32_t s0, s1, cng0 = 0;
        if (h >= s->nh) {                       /* replay pass: reset everything, feed hops 0..replay_len-1 again */
            if (!replay_phase) {
                replay_phase = 1;
                for (role = 0; role < NROLE; ++role) aec_reset(A[role]);
                last_pb_hop = -1;
            }
            src_h = h - s->nh;
        }
        mic = s->mic + (size_t)src_h * hop;
        far = s->far + (size_t)src_h * hop;
        if (!replay_phase) {
            while (ev < s->n_reset && s->reset_at[ev] == h) {
                for (role = 0; role < NROLE; ++role) aec_reset(A[role]);
                last_pb_hop = -1;
                ev++;
            }
            while (ra < s->n_realign && s->realign_at[ra] == h) {
                uint32_t before = aec_post_reset_serial(A[1]);
                for (role = 0; role < NROLE; ++role) {
                    int rr = aec_apply_external_realign(A[role], s->realign_delta[ra]);
                    if (role == 1) { if (rr == 1) st->realign_warm++; else st->realign_cold++; }
                }
                if (aec_post_reset_serial(A[1]) != before) st->realign_serial_moves++;
                ra++;
            }
        }
        if (s->force_verdict > 0) {
            int want = (h / s->force_verdict) % 3 == 0;
            for (role = 0; role < NROLE; ++role) force_verdict(A[role], want);
        }
        /* state that decides this hop's delay event (all instances are identical up to their post-chain mode) */
        aec_debug_status(A[1], &ds);
        d0 = ds.delay_samples;
        s0 = aec_post_reset_serial(A[1]);
        rej0 = A[1]->delay_reject_count;
        cng0 = A[0]->post.cng_seed;
        {
            const Aec* a1 = A[1];
            int span = pbfdaf_tap_span(&a1->main_filter.base);
            warm_pred = a1->cfg.delay_acquire_warm_transfer
                        && erle_ring_peak15(a1) > a1->cfg.delay_acquire_inst_erle_db
                        && span <= 4096;                    /* pbfdaf_warm_shift_ir refuses a span above its scratch */
            if (warm_pred) warm_pred = span;                /* the new delay must also be in (0, span): checked below */
        }
        aec_process(A[0], mic, far, oF);
        aec_process_context(A[1], mic, far);
        aec_process_context(A[2], mic, far);
        aec_process_context(A[3], mic, far);
        aec_process_context(A[4], mic, far);
        aec_get_res_context(A[0], &cF); aec_get_res_context(A[1], &cK); aec_get_res_context(A[2], &cN);
        aec_get_res_context(A[3], &cG); aec_get_res_context(A[4], &cL);
        memset(&oBF, 0, sizeof oBF); oBF.struct_size = sizeof oBF; oBK = oBN = oBG = oBL = oBF;
        if (aec_get_output_base(A[0], &oBF) || aec_get_output_base(A[1], &oBK) || aec_get_output_base(A[2], &oBN)
            || aec_get_output_base(A[3], &oBG) || aec_get_output_base(A[4], &oBL)) {
            CHECK(0, "%s: aec_get_output_base failed", s->name);
            break;
        }

        /* 5./6. classify this hop's delay event and check the serial delta against the branch */
        aec_debug_status(A[1], &ds);
        d1 = ds.delay_samples;
        s1 = aec_post_reset_serial(A[1]);
        {
            int ne_soft = A[1]->cfg.dt_aware_recovery_soft && A[1]->ne_recent_frames > 0;
            int rej = A[1]->delay_reject_count > rej0;
            uint32_t want_delta = 0u;
            int event = 1;
            if (rej) {                                   /* bad-relock revert */
                revert_hop = 1;
                st->reverts++;
                if (st->first_revert_hop < 0) st->first_revert_hop = h;
                if (!(d1 != d0 && h - last_pb_hop == AEC_DELAY_SHIFT_JUDGMENT_HOPS && d1 == last_pb_old
                      && d0 == last_pb_new)) {
                    st->revert_sig_bad++; bad = 1;
                    if (getenv("GU_DEBUG")) printf("DBG revert sig hop %d d %d->%d last pb hop %d %d->%d\n", h, d0, d1, last_pb_hop, last_pb_old, last_pb_new);
                }
                {   /* full-mode CNG: continued (one schedule step) or idle, never reseeded */
                    uint32_t c1 = A[0]->post.cng_seed;
                    if (!(c1 == cng0 || c1 == lcg_advance(cng0, nb - 2))) { st->revert_cng_bad++; bad = 1; }
                }
            } else if (d1 != d0 && d0 < 0) {             /* Path A first acquisition */
                if (warm_pred && d1 > 0 && d1 < warm_pred) st->pa_warm++;
                else if (ne_soft) st->pa_soft++;
                else { st->pa_hard++; want_delta = 1u; }
            } else if (d1 != d0) {                       /* Path B shift */
                if (ne_soft) st->pb_soft++;
                else { st->pb_hard++; want_delta = 1u; }
                last_pb_hop = h; last_pb_old = d0; last_pb_new = d1;
            } else {
                event = 0;
            }
            if (event && s1 - s0 != want_delta) {
                st->serial_bad++; bad = 1;
                if (getenv("GU_DEBUG")) printf("DBG %s hop %d delay %d->%d rej %d ne %d serial +%u want +%u\n", s->name, h, d0, d1, rej, ne_soft, s1 - s0, want_delta);
            }
            if (!event && s1 != s0) { st->unexplained_serial++; bad = 1; }
            for (role = 0; role < NROLE; ++role) {
                AecDebugStatus dr;
                aec_debug_status(A[role], &dr);
                if (dr.delay_samples != d1) { st->lane_diff_hops++; bad = 1; break; }
            }
        }

        /* 1. full vs split audio, bit for bit */
        if (app_synth(&app_k, A[1], oK, 0) || app_synth(&app_n, A[2], oN, 0)) { CHECK(0, "%s: synth", s->name); break; }
        if (!bits_eq_f(oF, oK, hop)) { st->full_vs_split_diff_hops++; bad = 1; if (revert_hop) st->revert_split_diff++;
            if (getenv("GU_DEBUG") && st->full_vs_split_diff_hops < 4) { double m = 0; for (k = 0; k < hop; ++k) { double d = fabs((double)oF[k] - oK[k]); if (d > m) m = d; } printf("DBG hop %d maxdiff %.3e oF[3]=%.9g oK[3]=%.9g serial %u\n", h, m, oF[3], oK[3], aec_post_reset_serial(A[1])); } }
        if (!bits_eq_f(oF, oN, hop)) { st->full_vs_ctxfull_diff_hops++; bad = 1; }
        /* 6. counterfactual application: identical to app_k up to the first revert, where it clears and reseeds as if the
         * serial had moved; that must change the audio of that hop (else the identity could not see a cleared OLA/CNG). */
        if (cf_active) {
            if (app_synth(&app_x, A[1], oX, revert_hop)) { CHECK(0, "%s: synth", s->name); break; }
            if (revert_hop) {
                st->revert_cf_checked++;
                if (bits_eq_f(oF, oX, hop)) { st->revert_cf_same++; bad = 1; }
                cf_active = 0;
            }
        }
        if (!replay_phase) memcpy(pass1 + (size_t)h * hop, oF, (size_t)hop * sizeof(float));
        else if (src_h < s->replay_len) {
            if (!bits_eq_f(oF, pass1 + (size_t)src_h * hop, hop)) { st->replay_diff_hops++; bad = 1; }
        }

        /* 2. same analysis seam, rendering base and serial in every mode */
        if (!bits_eq_c(cF.error_spec, cK.error_spec, nb) || !bits_eq_c(cF.echo_spec, cK.echo_spec, nb)
            || !bits_eq_c(cF.near_spec, cK.near_spec, nb) || !bits_eq_f(cF.formed_hop, cK.formed_hop, hop)
            || cF.usable_linear != cK.usable_linear || !bits_eq_f(cF.res_gain, cK.res_gain, nb)
            || !bits_eq_f(cF.r2, cK.r2, nb) || !bits_eq_f(cF.comfort_noise, cK.comfort_noise, nb)
            || !bits_eq_c(cN.error_spec, cK.error_spec, nb) || !bits_eq_c(cN.near_spec, cK.near_spec, nb)
            || !bits_eq_f(cN.formed_hop, cK.formed_hop, hop)) { st->seam_diff_hops++; bad = 1; }
        if (oBF.guard_active != oBK.guard_active || oBN.guard_active != oBK.guard_active
            || oBG.guard_active != oBK.guard_active || oBL.guard_active != oBK.guard_active
            || !bits_eq_c(oBF.base_spec, oBK.base_spec, nb) || !bits_eq_c(oBN.base_spec, oBK.base_spec, nb)
            || !bits_eq_c(oBG.base_spec, oBK.base_spec, nb) || !bits_eq_c(oBL.base_spec, oBK.base_spec, nb)
            || !bits_eq_c(cG.error_spec, cK.error_spec, nb) || !bits_eq_f(cG.formed_hop, cK.formed_hop, hop)
            || !bits_eq_c(cG.near_spec, cK.near_spec, nb) || cG.usable_linear != cK.usable_linear
            || !bits_eq_c(cL.error_spec, cK.error_spec, nb) || !bits_eq_f(cL.formed_hop, cK.formed_hop, hop)
            || !bits_eq_c(cL.near_spec, cK.near_spec, nb) || cL.usable_linear != cK.usable_linear) {
            st->lane_diff_hops++; bad = 1;
        }
        {
            uint32_t sk0 = aec_post_reset_serial(A[0]);
            for (role = 1; role < NROLE; ++role)
                if (aec_post_reset_serial(A[role]) != sk0) { st->serial_lane_diff++; bad = 1; break; }
        }

        /* 3. definition of the rendering base and independent guard decision (this hop's verdict) */
        {
            int fired = oBK.guard_active;
            long base_def_before = st->base_def_bad_hops;
            double se = energy_abs_sq(cK.error_spec, nb), sy = energy_abs_sq(cK.near_spec, nb);
            double gap = fabs(se - sy) / (se > sy ? se : sy > 0.0 ? sy : 1.0);
            if (fired ? !bits_eq_c(oBK.base_spec, cK.near_spec, nb) : !bits_eq_c(oBK.base_spec, cK.error_spec, nb))
                { st->base_def_bad_hops++; bad = 1; }
            if (cK.usable_linear && fired) { st->base_def_bad_hops++; bad = 1; }
            if (getenv("GU_DEBUG") && (st->base_def_bad_hops != base_def_before))
                printf("DBG %s hop %d base def: fired %d usable %d base==near %d base==error %d\n", s->name, h, fired,
                       cK.usable_linear, bits_eq_c(oBK.base_spec, cK.near_spec, nb), bits_eq_c(oBK.base_spec, cK.error_spec, nb));
            if (!cK.usable_linear) {
                int want = se > sy;
                if (want != fired) { if (gap < 1e-6) st->oracle_ties++; else { st->oracle_mismatch++; bad = 1; } }
            }
            if (prev_usable >= 0 && cK.usable_linear != prev_usable && se > sy && gap >= 1e-6) st->verdict_flips_e_gt_y++;
            prev_usable = cK.usable_linear;
            if (fired) {
                st->guard_hops++;
                if (st->first_usable < 0) st->guard_before_usable++; else st->guard_after_usable++;
            }
            if (fired != prev_guard) st->guard_toggles++;
            prev_guard = fired;
            if (cK.usable_linear && st->first_usable < 0) st->first_usable = h;
        }
        {
            uint32_t sr_now = aec_post_reset_serial(A[1]);
            if (sr_now != last_serial) st->serial_changes++;
            last_serial = sr_now;
        }

        /* 4. analysis pairing with NO guard-hop exception: unity synthesis of error_spec == previous formed hop.
         * After a hard reset the previous-hop memory is cleared together with the OLA, so the expectation is zeros. */
        {
            uint32_t sn = aec_post_reset_serial(A[1]);
            double mx = 0.0;
            if (sn != probe_serial) { ac_istft_reset(probe); memset(prev_formed, 0, (size_t)hop * sizeof(float)); probe_serial = sn; }
            ac_istft_push(probe, cK.error_spec, oP);
            for (k = 0; k < hop; ++k) {
                double d = fabs((double)oP[k] - (double)prev_formed[k]);
                if (!(d <= mx)) mx = d;           /* NaN propagates into the maximum */
            }
            if (mx > st->probe_max_err) st->probe_max_err = mx;
            if (mx > PROBE_TOL || mx != mx) { st->probe_bad_hops++; bad = 1; }
            if (oBK.guard_active) st->probe_guard_hops_checked++;
            memcpy(prev_formed, cK.formed_hop, (size_t)hop * sizeof(float));
        }
        st->hops++;
        if (bad && st->first_bad_hop < 0) st->first_bad_hop = h;
    }
    app_free(&app_k); app_free(&app_n); app_free(&app_x);
    free(probe_mem);
    for (role = 0; role < NROLE; ++role) { aec_destroy(A[role]); free(A[role]); }
    free(oF); free(oK); free(oN); free(oP); free(oX); free(prev_formed); free(pass1);
}


/* ------------------------------------------------------------------------------------------------------------------ */
/* reporting and scenarios                                                                                            */
/* ------------------------------------------------------------------------------------------------------------------ */

static void report(const Stream* s, const Stats* st) {
    CHECK(st->full_vs_split_diff_hops == 0, "%s: aec_process() vs context+getters+aec_cng_fill+ac_istft differs on %ld of %ld hops (first %ld)",
          s->name, st->full_vs_split_diff_hops, st->hops, st->first_bad_hop);
    CHECK(st->full_vs_ctxfull_diff_hops == 0, "%s: full config via aec_process_context() differs from aec_process() on %ld hops", s->name,
          st->full_vs_ctxfull_diff_hops);
    CHECK(st->seam_diff_hops == 0, "%s: analysis seam / gain / r2 / comfort noise differs between modes on %ld hops (first %ld)", s->name,
          st->seam_diff_hops, st->first_bad_hop);
    CHECK(st->lane_diff_hops == 0, "%s: rendering base / guard / NO_GAIN / LINEAR_ONLY lanes differ on %ld hops (first %ld)", s->name,
          st->lane_diff_hops, st->first_bad_hop);
    CHECK(st->base_def_bad_hops == 0, "%s: rendering base violates its definition on %ld hops", s->name, st->base_def_bad_hops);
    CHECK(st->oracle_mismatch == 0, "%s: guard decision disagrees with the independent recomputation on %ld hops outside the tie band", s->name,
          st->oracle_mismatch);
    CHECK(st->serial_lane_diff == 0, "%s: post_reset_serial differs between modes", s->name);
    CHECK(st->probe_bad_hops == 0, "%s: error_spec does not pair with the previous formed hop on %ld hops (max err %.3g, first %ld)", s->name,
          st->probe_bad_hops, st->probe_max_err, st->first_bad_hop);
    CHECK(st->replay_diff_hops == 0, "%s: reset + replay differs from the first pass on %ld hops", s->name, st->replay_diff_hops);
    CHECK(st->realign_serial_moves == 0, "%s: an external realign moved post_reset_serial", s->name);
    CHECK(st->serial_bad == 0, "%s: post_reset_serial delta disagrees with the delay-event branch on %ld events (hard +1, "
          "warm / soft / revert 0)", s->name, st->serial_bad);
    CHECK(st->unexplained_serial == 0, "%s: post_reset_serial moved inside %ld processing calls without a hard delay event",
          s->name, st->unexplained_serial);
    CHECK(st->revert_sig_bad == 0, "%s: %ld reverts without the public signature (back to the pre-shift delay %d hops after "
          "the shift)", s->name, st->revert_sig_bad, AEC_DELAY_SHIFT_JUDGMENT_HOPS);
    CHECK(st->revert_cng_bad == 0, "%s: the full-mode comfort-noise LCG was reseeded on %ld revert hops", s->name,
          st->revert_cng_bad);
    CHECK(st->revert_cf_same == 0, "%s: clearing the OLA tail and reseeding the CNG on the revert hop leaves the audio "
          "unchanged (the identity cannot see a reset there)", s->name);
    printf("  %-34s %5ld hops guard %5ld (before usable %5ld, after %5ld, switches %4ld) hard resets %3ld first usable %5ld "
           "probe max %.2e on %ld guard hops, ties %ld\n",
           s->name, st->hops, st->guard_hops, st->guard_before_usable, st->guard_after_usable, st->guard_toggles,
           st->serial_changes, st->first_usable, st->probe_max_err, st->probe_guard_hops_checked, st->oracle_ties);
    if (st->pa_warm + st->pa_soft + st->pa_hard + st->pb_soft + st->pb_hard + st->reverts + st->verdict_flips_e_gt_y)
        printf("    delay events: acquire warm %ld soft %ld hard %ld, shift soft %ld hard %ld, revert %ld (first hop %ld); "
               "verdict flips with E > Y %ld\n", st->pa_warm, st->pa_soft, st->pa_hard, st->pb_soft, st->pb_hard,
               st->reverts, st->first_revert_hop, st->verdict_flips_e_gt_y);
}

static Stream make_stream(const char* name, int sr, int fft, int seconds) {
    Stream s;
    memset(&s, 0, sizeof s);
    s.name = name; s.sr = sr; s.fft = fft; s.hop = fft / 2;
    s.nh = seconds * sr / s.hop;
    s.enable_cng = 1;
    s.replay_from = -1;
    return s;
}
static void free_stream(Stream* s) { free(s->mic); free(s->far); }

static void scenarios_synthetic(int sr, int fft) {
    char nm[96];
    Stream s;
    Stats st;
    const int tag_len = 16;
    (void)tag_len;

    /* moving echo path with delay changes, double talk burst: guard on/off/switching, unusable -> usable, hard resets */
    snprintf(nm, sizeof nm, "syn move %d/%d", sr / 1000, fft);
    g_rs = 0x2545F491u; s = make_stream(nm, sr, fft, 14);
    gen_echo_stream(&s, 1, 1, 1, 0);
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.guard_hops > 0 && st.guard_before_usable > 0, "%s: the guard fires while the estimate is not usable", nm);
    CHECK(st.first_usable >= 0, "%s: the estimate becomes usable (unusable -> usable covered)", nm);
    CHECK(st.guard_toggles >= 2, "%s: guard switches on and off", nm);
    free_stream(&s);

    /* the same stream with comfort noise off: no CNG draw at all, still identical */
    snprintf(nm, sizeof nm, "syn move nocng %d/%d", sr / 1000, fft);
    g_rs = 0x2545F491u; s = make_stream(nm, sr, fft, 8);
    s.enable_cng = 0;
    gen_echo_stream(&s, 1, 1, 1, 0);
    run_stream(&s, &st); report(&s, &st);
    free_stream(&s);

    /* init: silence head, then loud far end without any echo path: never usable, guard fires, nothing to cancel */
    snprintf(nm, sizeof nm, "syn never-usable %d/%d", sr / 1000, fft);
    g_rs = 0x1234567u; s = make_stream(nm, sr, fft, 10);
    gen_echo_stream(&s, 0, 0, 0, 30);
    { int i, n = s.nh * s.hop; for (i = 0; i < n; ++i) s.mic[i] += 0.03f * rnd(); }
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.guard_hops > 0, "%s: guard fires on a never-usable stream", nm);
    CHECK(st.first_usable < 0, "%s: stays unusable", nm);
    free_stream(&s);

    /* aec_reset events (hard reset) and replay from the start: reset == fresh, serial moves, outputs identical */
    snprintf(nm, sizeof nm, "syn reset+replay %d/%d", sr / 1000, fft);
    g_rs = 0x2545F491u; s = make_stream(nm, sr, fft, 10);
    gen_echo_stream(&s, 1, 1, 1, 0);
    { static int rs[3]; rs[0] = s.nh / 4; rs[1] = s.nh / 2; rs[2] = 3 * s.nh / 5; s.reset_at = rs; s.n_reset = 3; }
    s.replay_from = 0; s.replay_len = s.nh / 4;
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.serial_changes >= 4, "%s: aec_reset moves post_reset_serial (saw %ld)", nm, st.serial_changes);
    free_stream(&s);
}

/* DT-aware soft acquisition and soft shifts (warm transfer off, so a Path A acquisition is soft or hard), the bad-relock
 * revert on a synthetic stream, and forced verdict switching. */
static void scenarios_delay_branches(int sr, int fft) {
    char nm[96];
    Stream s;
    Stats st;

    snprintf(nm, sizeof nm, "syn dt-soft %d/%d", sr / 1000, fft);
    g_rs = 0x2545F491u; s = make_stream(nm, sr, fft, 14);
    s.warm_transfer_off = 1;
    gen_dt_soft_stream(&s, 0.8f);
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.pa_soft >= 1, "%s: a DT-aware soft acquisition is reached (%ld)", nm, st.pa_soft);
    CHECK(st.pb_soft >= 2, "%s: DT-aware soft shifts are reached (%ld)", nm, st.pb_soft);
    CHECK(st.serial_changes == 0, "%s: soft acquisition / soft shifts never move post_reset_serial (saw %ld)", nm,
          st.serial_changes);
    free_stream(&s);

    /* moving echo path without double talk: hard acquisition and shifts; on the 16 kHz grids the shift to the third
     * delay is judged bad and reverted */
    snprintf(nm, sizeof nm, "syn relock %d/%d", sr / 1000, fft);
    g_rs = 0x2545F491u; s = make_stream(nm, sr, fft, 14);
    gen_echo_stream(&s, 1, 1, 0, 0);
    run_stream(&s, &st); report(&s, &st);
    if (sr == 16000) CHECK(st.reverts >= 1, "%s: the bad-relock revert is reached (%ld)", nm, st.reverts);
    CHECK(st.pa_hard + st.pa_warm >= 1 && st.pb_hard >= 1, "%s: acquisition and hard shifts are reached", nm);
    free_stream(&s);

    /* forced verdict switching on an uncorrelated stream: the verdict flips every 7 hops (usable on one phase of three)
     * while the residual is louder than the capture */
    snprintf(nm, sizeof nm, "syn forced-verdict %d/%d", sr / 1000, fft);
    g_rs = 0x31415926u; s = make_stream(nm, sr, fft, 6);
    s.force_verdict = 7;
    gen_uncorrelated_stream(&s);
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.verdict_flips_e_gt_y >= 20, "%s: hops whose verdict flips while E > Y (%ld)", nm, st.verdict_flips_e_gt_y);
    CHECK(st.guard_hops > 0 && st.guard_hops < st.hops, "%s: guard on and off", nm);
    free_stream(&s);
}

static void scenario_external_realign(int sr, int fft) {
    char nm[96];
    Stream s;
    Stats st;
    snprintf(nm, sizeof nm, "syn ext-realign %d/%d", sr / 1000, fft);
    g_rs = 0x77AA55u; s = make_stream(nm, sr, fft, 12);
    gen_echo_stream(&s, 1, 1, 0, 0);
    /* EXTERNAL_ALIGNED: the caller's far is taken as aligned, so the echo path is a fixed small lag; realigns are soft
     * or cold filter restarts that never touch the post chain. */
    s.external_aligned = 1;
    { static int at[3]; static int dl[3]; at[0] = s.nh / 4; at[1] = s.nh / 2; at[2] = 3 * s.nh / 4; dl[0] = 40; dl[1] = -24; dl[2] = 96;
      s.realign_at = at; s.realign_delta = dl; s.n_realign = 3; }
    run_stream(&s, &st); report(&s, &st);
    CHECK(st.serial_changes == 0, "%s: soft/warm realign never moves post_reset_serial (saw %ld)", nm, st.serial_changes);
    CHECK(st.realign_warm + st.realign_cold == 3, "%s: all realign events applied", nm);
    printf("    realigns: %ld warm-transfer, %ld filter restart\n", st.realign_warm, st.realign_cold);
    CHECK(st.guard_hops > 0, "%s: guard exercised", nm);
    free_stream(&s);
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* real clips                                                                                                         */
/* ------------------------------------------------------------------------------------------------------------------ */

static short* read_wav16(const char* path, long* n, int* sr) {
    FILE* f = fopen(path, "rb");
    unsigned char h[12];
    short* data = NULL;
    *n = 0; *sr = 0;
    if (!f) return NULL;
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) { fclose(f); return NULL; }
    for (;;) {
        unsigned char ch[8];
        uint32_t sz;
        if (fread(ch, 1, 8, f) != 8) break;
        sz = ch[4] | (ch[5] << 8) | (ch[6] << 16) | ((uint32_t)ch[7] << 24);
        if (!memcmp(ch, "fmt ", 4)) {
            unsigned char fm[64];
            if (sz > sizeof fm || fread(fm, 1, sz, f) != sz) break;
            if ((fm[0] | (fm[1] << 8)) != 1 || (fm[2] | (fm[3] << 8)) != 1 || (fm[14] | (fm[15] << 8)) != 16) break;
            *sr = fm[4] | (fm[5] << 8) | (fm[6] << 16) | (fm[7] << 24);
        } else if (!memcmp(ch, "data", 4)) {
            data = (short*)malloc((size_t)sz + 2);
            if (data && fread(data, 1, sz, f) == sz) *n = (long)(sz / 2); else { free(data); data = NULL; }
            break;
        } else {
            if (fseek(f, (long)(sz + (sz & 1)), SEEK_CUR) != 0) break;
        }
    }
    fclose(f);
    return data;
}

typedef struct {
    long clips, firing, never_usable_firing, reset_firing, after_usable, switching, no_guard, hops, guard_hops;
    long reverts, revert_clips, skipped;
} ClipTotals;

/* min_reverts: the clip must hit the bad-relock revert at least this often (clip-list fifth column; 0 = no demand). */
static int run_clip(const char* wav_root, const char* stem, int fft, const char* mic_rel, const char* ref_rel, ClipTotals* tot,
                    const char* abs_mic, const char* abs_ref, int min_reverts) {
    char pm[2048], pr[2048];
    long nm, nr, n;
    int srm, srr, i;
    short *m16, *r16;
    Stream s;
    Stats st;
    if (abs_mic) { snprintf(pm, sizeof pm, "%s", abs_mic); snprintf(pr, sizeof pr, "%s", abs_ref); }
    else { snprintf(pm, sizeof pm, "%s/%s", wav_root, mic_rel); snprintf(pr, sizeof pr, "%s/%s", wav_root, ref_rel); }
    m16 = read_wav16(pm, &nm, &srm);
    r16 = read_wav16(pr, &nr, &srr);
    if (!m16 || !r16 || srm != srr) {
        CHECK(0, "clip %s: cannot read %s / %s (the corpus is present, so a missing clip is a failure)", stem, pm, pr);
        free(m16); free(r16);
        return -1;
    }
    memset(&s, 0, sizeof s);
    s.name = stem; s.sr = srm; s.fft = fft; s.hop = fft / 2; s.enable_cng = 1; s.replay_from = -1;
    n = nm < nr ? nm : nr;
    s.nh = (int)(n / s.hop);
    s.mic = (float*)malloc((size_t)s.nh * s.hop * sizeof(float));
    s.far = (float*)malloc((size_t)s.nh * s.hop * sizeof(float));
    for (i = 0; i < s.nh * s.hop; ++i) { s.mic[i] = (float)m16[i] / 32768.0f; s.far[i] = (float)r16[i] / 32768.0f; }
    free(m16); free(r16);
    run_stream(&s, &st);
    {
        char nm2[160];
        snprintf(nm2, sizeof nm2, "clip %d/%.40s", fft, stem);
        s.name = nm2;
        report(&s, &st);
        CHECK(st.reverts >= min_reverts, "%s: bad-relock reverts %ld, the clip list requires >= %d", nm2, st.reverts,
              min_reverts);
    }
    tot->clips++; tot->hops += st.hops; tot->guard_hops += st.guard_hops;
    tot->reverts += st.reverts;
    if (st.reverts > 0) tot->revert_clips++;
    if (st.guard_hops > 0) tot->firing++;
    else tot->no_guard++;
    if (st.guard_hops > 0 && st.first_usable < 0) tot->never_usable_firing++;
    if (st.guard_hops > 0 && st.serial_changes > 0) tot->reset_firing++;
    if (st.guard_after_usable > 0) tot->after_usable++;
    if (st.guard_toggles >= 6) tot->switching++;
    free_stream(&s);
    return 0;
}

/* The corpus of a clip-list row is the first path component of its mic path under wav_root. */
static int corpus_present(const char* wav_root, const char* mic_rel) {
    char dir[2048];
    struct stat sb;
    size_t l = strcspn(mic_rel, "/");
    snprintf(dir, sizeof dir, "%s/%.*s", wav_root, (int)l, mic_rel);
    return stat(dir, &sb) == 0 && S_ISDIR(sb.st_mode);
}

static void clips_from_list(const char* wav_root, const char* list) {
    FILE* f = fopen(list, "r");
    char line[8192];
    int fft_seen[3] = {0, 0, 0};
    ClipTotals tot[3];
    int g;
    memset(tot, 0, sizeof tot);
    if (!f) { CHECK(0, "cannot open the clip list %s", list); return; }
    while (fgets(line, sizeof line, f)) {
        char* fld[5];
        int nf = 0, fft, gi, min_rev;
        char* t;
        if (line[0] == '#' || line[0] == '\n') continue;
        for (t = strtok(line, "\t\n"); t && nf < 5; t = strtok(NULL, "\t\n")) fld[nf++] = t;
        if (nf < 4) continue;
        fft = atoi(fld[0]);
        gi = fft == 256 ? 0 : (fft == 512 ? 1 : 2);
        fft_seen[gi] = 1;
        min_rev = nf >= 5 ? atoi(fld[4]) : 0;
        if (!corpus_present(wav_root, fld[2])) {
            printf("  SKIP clip %d/%s: corpus directory of %s absent under %s\n", fft, fld[1], fld[2], wav_root);
            tot[gi].skipped++;
            continue;
        }
        run_clip(wav_root, fld[1], fft, fld[2], fld[3], &tot[gi], NULL, NULL, min_rev);
    }
    fclose(f);
    for (g = 0; g < 3; ++g) {
        const char* gn[3] = {"16k/256", "16k/512", "48k/1024"};
        if (tot[g].skipped) {
            printf("\n  ******************************************************************************************\n"
                   "  *** SKIP %s: %ld listed real clips not run, their blind-set corpus is absent (not in git).\n"
                   "  *** The real-clip coverage of this grid (guard firing, switching, hard resets, bad-relock\n"
                   "  *** reverts) is NOT verified by this run; the synthetic part above still is.\n"
                   "  ******************************************************************************************\n\n",
                   gn[g], tot[g].skipped);
            CHECK(tot[g].clips == 0, "%s: part of the grid's corpus present and part absent (%ld run, %ld skipped)", gn[g],
                  tot[g].clips, tot[g].skipped);
            continue;
        }
        CHECK(fft_seen[g] && tot[g].clips >= 8, "%s: at least 8 real clips ran (%ld)", gn[g], tot[g].clips);
        CHECK(tot[g].never_usable_firing >= 2, "%s: never-usable clips that fire the guard (%ld)", gn[g], tot[g].never_usable_firing);
        CHECK(tot[g].reset_firing >= 1, "%s: clips with hard resets and guard firing (%ld)", gn[g], tot[g].reset_firing);
        CHECK(tot[g].after_usable >= 1, "%s: clips that fire the guard after the first usable verdict (%ld)", gn[g], tot[g].after_usable);
        CHECK(tot[g].switching >= 1, "%s: clips with guard switching (%ld)", gn[g], tot[g].switching);
        CHECK(tot[g].no_guard >= 1, "%s: clips that never fire (%ld)", gn[g], tot[g].no_guard);
        CHECK(tot[g].revert_clips >= 2, "%s: clips that hit the bad-relock revert (%ld)", gn[g], tot[g].revert_clips);
        printf("  %s totals: %ld clips, %ld hops, %ld guard hops; firing %ld, never-usable+firing %ld, reset+firing %ld, after-usable %ld, switching %ld, silent %ld, reverts %ld on %ld clips\n",
               gn[g], tot[g].clips, tot[g].hops, tot[g].guard_hops, tot[g].firing, tot[g].never_usable_firing,
               tot[g].reset_firing, tot[g].after_usable, tot[g].switching, tot[g].no_guard, tot[g].reverts,
               tot[g].revert_clips);
    }
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* aec_cng_fill against an independent scalar reference                                                               */
/* ------------------------------------------------------------------------------------------------------------------ */

static void test_cng_reference(void) {
    static const int sizes[3] = {129, 257, 513};
    int si, k, exact_sqrt = 1, variant;
    /* the phase table is checked against its definition sqrt(2) * sin(2 pi i / 32), then used as data by the reference */
    for (k = 0; k < 32; ++k)
        CHECK(fabs((double)AECB_SQRT2_SIN_LUT[k] - sqrt(2.0) * sin(2.0 * 3.14159265358979323846 * k / 32.0)) < 1e-6,
              "phase table entry %d matches sqrt(2) sin(2 pi i / 32)", k);
    /* bit-exact comparison is required wherever fast_sqrt is the correctly rounded sqrtf (every AArch64 build) */
    g_rs = 99u;
    for (k = 0; k < 4000; ++k) {
        float x = 1.0e-3f + 1.0e6f * (rnd() * 0.5f + 0.5f);
        if (fast_sqrt(x) != sqrtf(x)) { exact_sqrt = 0; break; }
    }
    printf("  fast_sqrt == sqrtf on this build: %s\n", exact_sqrt ? "yes (bit-exact comparison)" : "no (1e-5 relative comparison)");
    for (si = 0; si < 3; ++si) {
        int n = sizes[si];
        for (variant = 0; variant < 4; ++variant) {
            float *cn = (float*)malloc((size_t)n * 4), *gr = (float*)malloc((size_t)n * 4), *gn = (float*)malloc((size_t)n * 4);
            Complex *sp = (Complex*)malloc((size_t)n * sizeof(Complex)), *rf = (Complex*)malloc((size_t)n * sizeof(Complex));
            uint32_t rng = 0x00C0FFEEu + (uint32_t)variant, seed = rng;
            const float nr_floor = variant >= 2 ? 0.25f : 0.0f;
            int use_nr = variant & 1, bad = 0;
            g_rs = 0xABCDEF01u + (uint32_t)si * 7u + (uint32_t)variant;
            for (k = 0; k < n; ++k) {
                float u = rnd();
                cn[k] = k == 5 ? 0.0f : (k == 6 ? -1234.5f : 1.0e9f * (u * u * u * u + 1e-9f));   /* wide range, zero, negative */
                gr[k] = k == 7 ? 1.3f : 0.5f * (rnd() + 1.0f);                                      /* includes g > 1 */
                gn[k] = k == 8 ? 0.0f : 1.2f * (rnd() * 0.5f + 0.5f);
                sp[k].r = rnd(); sp[k].i = rnd();
                rf[k] = sp[k];
            }
            aec_cng_fill(&rng, sp, cn, gr, use_nr ? gn : NULL, nr_floor, n);
            /* independent reference: plain scalar C, libm sqrtf, the documented product order */
            for (k = 1; k < n - 1; ++k) {
                uint32_t ix;
                float amp, ng, cr, ci;
                seed = (seed * 69069u + 1u) & 0x7FFFFFFFu;
                ix = seed >> 26;
                amp = cn[k] / (32768.0f * 32768.0f);
                amp = amp > 0.0f ? sqrtf(amp) : 0.0f;
                ng = 1.0f - gr[k] * gr[k];
                ng = ng > 0.0f ? sqrtf(ng) : 0.0f;
                if (use_nr) ng = ng * fmaxf(gn[k], nr_floor);
                cr = amp * AECB_SQRT2_SIN_LUT[ix];
                ci = amp * AECB_SQRT2_SIN_LUT[(ix + 8u) & 31u];
                rf[k].r += ng * cr;
                rf[k].i += ng * ci;
            }
            for (k = 0; k < n; ++k) {
                if (memcmp(&sp[k], &rf[k], sizeof(Complex)) != 0) {
                    /* exact when fast_sqrt is the correctly rounded sqrtf (any AArch64 build); otherwise 1e-5 relative */
                    double dr = fabs((double)sp[k].r - rf[k].r), di = fabs((double)sp[k].i - rf[k].i);
                    double sc = fabs((double)rf[k].r) + fabs((double)rf[k].i) + 1e-3;
                    if (exact_sqrt || (dr + di) / sc > 1e-5) bad++;
                }
            }
            CHECK(bad == 0, "aec_cng_fill n=%d nr=%d floor=%.2f: %d bins differ from the scalar reference", n, use_nr, nr_floor, bad);
            CHECK(rng == seed, "aec_cng_fill n=%d: rng advanced by exactly n-2 draws", n);
            CHECK(memcmp(&sp[0], &rf[0], sizeof(Complex)) == 0 && memcmp(&sp[n - 1], &rf[n - 1], sizeof(Complex)) == 0,
                  "aec_cng_fill n=%d: bins 0 and n-1 untouched", n);
            free(cn); free(gr); free(gn); free(sp); free(rf);
        }
    }
    {   /* argument contract: nothing touched, no draw */
        uint32_t rng = 7u;
        Complex sp[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
        float cn[4] = {1e9f, 1e9f, 1e9f, 1e9f}, g[4] = {0, 0, 0, 0};
        Complex ref[4];
        memcpy(ref, sp, sizeof sp);
        aec_cng_fill(&rng, sp, cn, g, NULL, 0.0f, 2);
        aec_cng_fill(NULL, sp, cn, g, NULL, 0.0f, 4);
        aec_cng_fill(&rng, NULL, cn, g, NULL, 0.0f, 4);
        aec_cng_fill(&rng, sp, NULL, g, NULL, 0.0f, 4);
        aec_cng_fill(&rng, sp, cn, NULL, NULL, 0.0f, 4);
        CHECK(rng == 7u && memcmp(sp, ref, sizeof sp) == 0, "aec_cng_fill: n < 3 or a NULL argument touches nothing");
    }
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* near-threshold float32 fixture for the guard rule                                                                  */
/* ------------------------------------------------------------------------------------------------------------------ */

/* Single non-zero bin so that the float32 pairwise reduction is exact and the decision depends only on the rounded
 * per-bin squares: |E| = 1 against |Y| = 1, nextafter(1, 0), nextafter(1, 2). Expected values come from plain float
 * arithmetic in this file, not from the code under test. */
static void test_guard_threshold_fixture(int sr, int fft) {
    AecConfig cfg;
    Aec a;
    int nb, k, c, mode;
    const float one = 1.0f;
    const float ys[3] = { 1.0f, nextafterf(one, 0.0f), nextafterf(one, 2.0f) };
    const int want[3] = { 0, 1, 0 };            /* strict: equal energy does not fire; Y one ulp below fires */
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    cfg.fft_size = fft;
    cfg.enable_res = 0; cfg.return_res_context = 1;
    if (aec_create(&a, &cfg) != 0) { CHECK(0, "fixture: create"); return; }
    nb = a.n_freqs;
    for (mode = 0; mode < 4; ++mode) {          /* 0 normal, 1 usable, 2 flag off, 3 energies swapped (E one ulp below) */
        for (c = 0; c < 3; ++c) {
            AecPostRunScratch* sc = &a.a3_sc;
            float e = one, y = ys[c];
            int expect = want[c], got;
            float se_ref, sy_ref;
            if (mode == 3) { e = ys[c]; y = one; expect = (e * e > y * y) ? 1 : 0; }
            for (k = 0; k < nb; ++k) {
                sc->sel_esw[k].r = sc->sel_esw[k].i = 0.0f;
                sc->sel_echo[k].r = sc->sel_echo[k].i = 0.0f;
                sc->ybase[k].r = sc->ybase[k].i = 0.0f;
                sc->output_base[k].r = sc->output_base[k].i = -7.0f;
                sc->abs_error[k] = 0.0f;
                sc->abs_ybase[k] = 123.0f;
            }
            sc->sel_esw[5].r = e;
            sc->sel_echo[5].r = y - e;
            sc->ybase[5].r = y;
            sc->abs_error[5] = e;
            se_ref = e * e; sy_ref = sqrtf(y * y); sy_ref = sy_ref * sy_ref;
            if (mode == 0 || mode == 3) CHECK((se_ref > sy_ref) == expect, "fixture reference sanity (c=%d mode=%d)", c, mode);
            a.post.cfg.output_capture_when_linear_unusable = (mode != 2);
            aec_post_eval_output_guard(&a.post, sc, mode == 1);
            got = a.post.output_guard_active;
            if (mode == 1 || mode == 2) expect = 0;
            CHECK(got == expect, "fixture %d/%d c=%d mode=%d: guard %d, expected %d", sr / 1000, fft, c, mode, got, expect);
            if (got) CHECK(sc->output_base[5].r == y && sc->output_base[4].r == 0.0f && sc->output_base[5].i == 0.0f,
                           "fixture: fired base is error + echo");
            else CHECK(sc->output_base[5].r == -7.0f, "fixture: an unfired guard leaves the dedicated base untouched");
        }
    }
    a.post.cfg.output_capture_when_linear_unusable = 1;
    aec_destroy(&a);
}

/* Reduction-order fixtures. The decision is specified on the float32 pairwise reduction (numpy's pairwise sum: n <= 8
 * sequential; n <= 128 eight interleaved accumulators combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) plus the
 * sequential sum of the n % 8 tail; above that a split at half - half % 8). The replica below is written from that
 * description. Each trial draws |E| over the n_freqs bins with a wide range and sets |Y| to a random permutation of
 * |E|: the per-bin float32 squares are the same multiset, so the exact energies tie and only the reduction order
 * decides. Trials are kept only when the pairwise and the sequential float32 decisions differ (both directions). */
static float pairwise_ref(const float* a, size_t n) {
    if (n <= 8) {
        float t = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) t += a[i];
        return t;
    }
    if (n <= 128) {
        float r[8], tail = 0.0f, res;
        size_t i, j;
        for (j = 0; j < 8; ++j) r[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) r[j] += a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) tail += a[i];
        return res + tail;
    }
    {
        size_t half = n / 2;
        half -= half % 8;
        return pairwise_ref(a, half) + pairwise_ref(a + half, n - half);
    }
}
static float sequential_ref(const float* a, size_t n) {
    float t = 0.0f;
    size_t i;
    for (i = 0; i < n; ++i) t += a[i];
    return t;
}

static void test_guard_reduction_order(int sr, int fft) {
    AecConfig cfg;
    Aec a;
    int nb, k, trial, kept[2] = {0, 0}, bad = 0, premise_bad = 0;
    const int want_each = 16;
    float *e, *y, *e2, *y2;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    cfg.fft_size = fft;
    cfg.enable_res = 0; cfg.return_res_context = 1;
    if (aec_create(&a, &cfg) != 0) { CHECK(0, "reduction fixture: create"); return; }
    nb = a.n_freqs;
    e = (float*)malloc((size_t)nb * sizeof(float)); y = (float*)malloc((size_t)nb * sizeof(float));
    e2 = (float*)malloc((size_t)nb * sizeof(float)); y2 = (float*)malloc((size_t)nb * sizeof(float));
    a.post.cfg.output_capture_when_linear_unusable = 1;
    g_rs = 0x5EED0000u + (uint32_t)fft;
    for (trial = 0; trial < 200000 && (kept[0] < want_each || kept[1] < want_each); ++trial) {
        int dp, ds, got;
        AecPostRunScratch* sc = &a.a3_sc;
        for (k = 0; k < nb; ++k) {
            float u = 0.5f * (rnd() + 1.0f), v = 0.5f * (rnd() + 1.0f);
            e[k] = ldexpf(1.0f + 0.5f * v, -(int)(14.0f * u));
            y[k] = e[k];
        }
        for (k = nb - 1; k > 0; --k) {               /* Fisher-Yates permutation of |E| into |Y| */
            int j = (int)((0.5f * (rnd() + 1.0f)) * (float)(k + 1));
            float t;
            if (j > k) j = k;
            t = y[k]; y[k] = y[j]; y[j] = t;
        }
        for (k = 0; k < nb; ++k) { e2[k] = e[k] * e[k]; y2[k] = y[k] * y[k]; }
        dp = pairwise_ref(e2, (size_t)nb) > pairwise_ref(y2, (size_t)nb);
        ds = sequential_ref(e2, (size_t)nb) > sequential_ref(y2, (size_t)nb);
        if (dp == ds || kept[dp] >= want_each) continue;
        kept[dp]++;
        for (k = 0; k < nb; ++k) {
            sc->sel_esw[k].r = e[k]; sc->sel_esw[k].i = 0.0f;
            sc->sel_echo[k].r = y[k] - e[k]; sc->sel_echo[k].i = 0.0f;
            sc->ybase[k].r = y[k]; sc->ybase[k].i = 0.0f;
            sc->abs_error[k] = e[k];
        }
        aec_post_eval_output_guard(&a.post, sc, 0);
        for (k = 0; k < nb; ++k) if (sc->abs_ybase[k] != y[k]) { premise_bad++; break; }
        got = a.post.output_guard_active;
        if (got != dp) bad++;
    }
    printf("  reduction-order fixture %d/%d: %d trials kept (pairwise fires %d, sequential fires %d) of %d drawn\n",
           sr / 1000, fft, kept[0] + kept[1], kept[1], kept[0], trial);
    CHECK(kept[0] >= want_each && kept[1] >= want_each, "reduction fixture %d/%d: discriminating trials in both directions",
          sr / 1000, fft);
    CHECK(premise_bad == 0, "reduction fixture %d/%d: |Y| magnitudes reproduce the drawn values", sr / 1000, fft);
    CHECK(bad == 0, "reduction fixture %d/%d: guard decision differs from the float32 pairwise reduction on %d of %d trials",
          sr / 1000, fft, bad, kept[0] + kept[1]);
    free(e); free(y); free(e2); free(y2);
    aec_destroy(&a);
}

/* ------------------------------------------------------------------------------------------------------------------ */
/* ABI / lifetime                                                                                                     */
/* ------------------------------------------------------------------------------------------------------------------ */

static void test_getter_contract(void) {
    AecConfig cfg;
    Aec a, b;
    AecOutputBase ob;
    float mic[256], far[256];
    int i, h;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.enable_res = 0; cfg.return_res_context = 1;
    if (aec_create(&a, &cfg) != 0) { CHECK(0, "getter: create"); return; }
    {   /* a caller struct declared smaller than AecOutputBase (an older 16-byte layout, and one byte short) sits in a
         * canary-filled buffer: refused, and not one byte written, the declared struct included */
        static const size_t short_sizes[2] = { 16u, sizeof(AecOutputBase) - 1u };
        unsigned char buf[sizeof(AecOutputBase) + 64], ref[sizeof(AecOutputBase) + 64];
        int t;
        for (t = 0; t < 2; ++t) {
            size_t dsz = short_sizes[t];
            memset(buf, 0xAB, sizeof buf);
            memcpy(buf, &dsz, sizeof dsz);
            memcpy(ref, buf, sizeof buf);
            CHECK(aec_get_output_base(&a, (AecOutputBase*)(void*)buf) == -1, "getter: struct_size %zu refused", dsz);
            CHECK(memcmp(buf, ref, sizeof buf) == 0, "getter: struct_size %zu: nothing written (declared struct and canary)",
                  dsz);
        }
        /* a larger, newer caller struct: accepted, only the known prefix written, the rest of the declared size kept */
        memset(buf, 0xAB, sizeof buf);
        { size_t dsz = sizeof buf; memcpy(buf, &dsz, sizeof dsz); }
        memcpy(ref, buf, sizeof buf);
        CHECK(aec_get_output_base(&a, (AecOutputBase*)(void*)buf) == 0, "getter: larger struct_size accepted");
        CHECK(memcmp(buf + sizeof(AecOutputBase), ref + sizeof(AecOutputBase), sizeof buf - sizeof(AecOutputBase)) == 0,
              "getter: larger struct_size: bytes past sizeof(AecOutputBase) untouched");
        memcpy(&ob, buf, sizeof ob);
        CHECK(ob.struct_size == sizeof ob && ob.n_freqs == a.n_freqs && ob.base_spec != NULL,
              "getter: larger struct_size: prefix filled, struct_size reports sizeof(AecOutputBase)");
    }
    memset(&ob, 0xAB, sizeof ob);
    ob.struct_size = sizeof ob;
    CHECK(aec_get_output_base(NULL, &ob) == -1 && ob.base_spec == NULL && ob.n_freqs == 0 && ob.guard_active == 0
          && ob.struct_size == sizeof ob, "getter: NULL instance refused, prefix zeroed");
    CHECK(aec_get_output_base(&a, NULL) == -1, "getter: NULL out refused");
    CHECK(aec_post_reset_serial(NULL) == 0u && aec_post_reset_serial(&a) == 0u, "serial: 0 on NULL and on a fresh instance");
    g_rs = 5u;
    for (h = 0; h < 40; ++h) {
        for (i = 0; i < 256; ++i) { mic[i] = 0.1f * rnd(); far[i] = 0.3f * rnd(); }
        aec_process_context(&a, mic, far);
    }
    {
        uint32_t s0 = aec_post_reset_serial(&a), s1, s2;
        s1 = aec_post_reset_serial(&a);
        CHECK(s0 == s1, "serial: reading does not consume it");
        aec_reset(&a);
        s2 = aec_post_reset_serial(&a);
        CHECK(s2 == s0 + 1u, "serial: aec_reset advances it by one");
        CHECK(aec_post_reset_serial(&a) == s2, "serial: repeatable read after a reset");
    }
    aec_destroy(&a);
    /* neither the post chain nor the context seam on: no base */
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.enable_res = 0; cfg.return_res_context = 0;
    if (aec_create(&b, &cfg) == 0) {
        ob.struct_size = sizeof ob;
        CHECK(aec_get_output_base(&b, &ob) == -1 && ob.base_spec == NULL, "getter: refused when neither enable_res nor return_res_context");
        aec_destroy(&b);
    }
}

/* ------------------------------------------------------------------------------------------------------------------ */

int main(int argc, char** argv) {
    const char* wav_root = argc > 1 ? argv[1] : "../wav";
    const char* list = argc > 2 ? argv[2] : "test/guard_unify_clips.tsv";
    const int grids[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    int g;
    const char* all = getenv("GUARD_UNIFY_ALL");

    printf("== getter contract\n");
    test_getter_contract();
    printf("== aec_cng_fill reference\n");
    test_cng_reference();
    printf("== guard threshold fixture\n");
    for (g = 0; g < 3; ++g) test_guard_threshold_fixture(grids[g][0], grids[g][1]);
    for (g = 0; g < 3; ++g) test_guard_reduction_order(grids[g][0], grids[g][1]);
    printf("== synthetic scenarios\n");
    for (g = 0; g < 3; ++g) {
        scenarios_synthetic(grids[g][0], grids[g][1]);
        scenarios_delay_branches(grids[g][0], grids[g][1]);
        scenario_external_realign(grids[g][0], grids[g][1]);
    }
    printf("== real blind-set clips\n");
    clips_from_list(wav_root, list);
    if (all) {
        FILE* f = fopen(all, "r");
        char line[8192];
        ClipTotals tot;
        const char* fs = getenv("GUARD_UNIFY_ALL_FFT");
        int fft = fs ? atoi(fs) : 512;
        memset(&tot, 0, sizeof tot);
        printf("== GUARD_UNIFY_ALL %s at fft %d\n", all, fft);
        if (!f) { CHECK(0, "cannot open %s", all); }
        else {
            while (fgets(line, sizeof line, f)) {
                char* fld[5];
                int nf = 0;
                char* t;
                for (t = strtok(line, "\t\n"); t && nf < 5; t = strtok(NULL, "\t\n")) fld[nf++] = t;
                if (nf < 4) continue;
                run_clip(wav_root, fld[0], fft, NULL, NULL, &tot, fld[2], fld[3], 0);
            }
            fclose(f);
            printf("  ALL totals: %ld clips, %ld hops, %ld guard hops; firing %ld, never-usable+firing %ld, reset+firing %ld, after-usable %ld, switching %ld, silent %ld\n",
                   tot.clips, tot.hops, tot.guard_hops, tot.firing, tot.never_usable_firing, tot.reset_firing, tot.after_usable, tot.switching, tot.no_guard);
        }
    }
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
