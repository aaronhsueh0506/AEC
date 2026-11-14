/* baseline2_mono.h -- the second single-channel scene set: case table and runner of the test of the entry points
 * (test/test_split_entry2.c; test/test_split_entry3.c adds baseline3_mono.h on top).
 *
 * The includer provides, before including this header: <aec.h> and the header of the stand-alone residual gain stage
 * (AecRes: src/aec_res_internal.h), <utility/simd_kernels.h>, and (the cases switch internal tuning, so it is built with AEC_TESTING=1)
 *   static void g2_mode_cfg(AecConfig* c, int ctx);     ctx = 1: the lane that exposes its residual view with every field
 *                                                       (residual FULL, driven by aec_process_linear + aec_process_residual);
 *                                                       ctx = 0: the instance that applies its own gain (aec_process)
 *   static int  g2_step_ctx(Aec* a, const float* mic, const float* ref, G2View* v);   0 on success
 *   static int  g2_step_out(Aec* a, const float* mic, const float* ref, float* out);  0 on success
 *   static void g2_sink(const char* line);              one finished line (no newline)
 * and may define G2_HOOK_APPLY(a, v, hop, gx, p1, p2, p3, p4) to cross-check the explicit-array apply computation.
 *
 * Output: per case, lines "<grid> <case> <field> <hop> <16 hex>"; the per-hop group hashes are chained (FNV over the
 * previous chain and the hop's hash) and a line is printed every G2_STRIDE hops, on the last hop and on the hop after every
 * event, so that any difference stays visible in every later line. G2_VERBOSE=<file> writes the unchained per-hop hash of
 * every group, in the same line format, to locate the first diverging hop and field with diff. Also read:
 * G2_ONLY=<substring of "<grid> <case>"> (run only those cases), G2_TRACE=1 (a per-hop status line on stderr), G2_POISON=<byte>
 * (the byte the instance pools are filled with, default 0xA5). */
#ifndef AEC_TEST_GOLDEN2_MONO_H
#define AEC_TEST_GOLDEN2_MONO_H

#include "baseline2_common.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ---- neutral view of one hop --------------------------------------------------------------------------------------- */
struct G2View {
    int n_freqs, hop_size;
    const float*   linear_hop;
    const Complex* far_spec;
    const float*   lin_formed;      /* the formed linear hop of the linear context */
    const float*   aligned_far;
    int            delay_samples; float delay_conf; int delay_state; unsigned generation;
    const Complex *error_spec, *echo_spec, *near_spec, *base_spec;
    const float   *formed_hop, *r2, *comfort_noise, *res_gain;
    int   usable_linear, filter_converged, epc_active, is_stationary_dt, res_floor_protect, guard_active, filter_once_converged;
    int   clock_drift;
    float dt_indicator, far_power, erle_factor, saturation_level, erl_estimate, divergence, shadow_dt;
};
typedef struct G2View G2View;

/* One bad-relock judgment as the library took it: `cur` is the windowed ERLE and `has_est` the (error power < near power) reading
 * of the hop before the judging hop, `pre` the value stored at the shift. Expected form: the outcome, the estimate flag and a
 * range for cur - pre. */
typedef struct { int revert, has_est; float d_lo, d_hi; } G2JudgeExp;
typedef struct { int shift_hop, hop, old_delay, new_delay, soft, has_est, reverted; float cur, pre, np, ep; } G2JudgeRec;
#define G2_MAX_JUDGE 16
static G2JudgeRec g2_jrec[G2_MAX_JUDGE];
static int g2_jn;
#ifndef G2_JOUT
#define G2_JOUT stderr
#endif

/* ---- cases --------------------------------------------------------------------------------------------------------- */
enum { G2T_NOSHADOW = 1, G2T_NOHPF = 2, G2T_NOCNG = 4, G2T_NOSAT = 8, G2T_OBSERVER = 16, G2T_N2WIN = 32 };

typedef struct G2MonoCase {
    char  name[28];
    int   sr, fft, dur_ms;
    int   delay_mode;          /* 0 MATCHED, 1 FIXED, 2 EXTERNAL_ALIGNED */
    int   level;               /* AecLevel of the configuration */
    int   tog;                 /* G2T_* */
    float duty_s;              /* delay_est_period_s */
    int   ares;                /* 1: also drive a stand-alone gain stage (with forced inputs below) */
    int   fd_t0, fd_t1;        /* ares: clock_drift forced to 1 in [t0, t1) ms */
    int   fs_t0, fs_t1;        /* ares: max_saturation forced to 0.9 */
    int   fdt_t0, fdt_t1;      /* ares: max_dt forced to 0.5 */
    G2Stim stim;
    int   n_ev; G2Ev ev[8];
    int   judge;               /* 1: log every bad-relock judgment (EV_judge line, G2JudgeRec census); 0 for the second set */
    int   n_exp; G2JudgeExp exp[4];     /* judge: the expected judgments in order (checked by g3m_judge_check, not part of the line stream) */
} G2MonoCase;

#define G2M_MAX_CASES 64

static G2MonoCase* g2m_new(G2MonoCase* out, int* n, const char* name, int sr, int fft, int dur_ms) {
    G2MonoCase* c = &out[(*n)++];
    memset(c, 0, sizeof *c);
    snprintf(c->name, sizeof c->name, "%s", name);
    c->sr = sr; c->fft = fft; c->dur_ms = dur_ms; c->level = 1;
    return c;
}

static int g2m_cases(G2MonoCase* out) {
    static const int grid[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    int n = 0;
    for (int g = 0; g < 3; ++g) {                       /* the three product grids */
        const int sr = grid[g][0], fft = grid[g][1];
        G2MonoCase* c;
        c = g2m_new(out, &n, "satdt", sr, fft, 8200);
        g2_stim_satdt(&c->stim);
        c->ares = 1; c->fd_t0 = 2300; c->fd_t1 = 8000; c->fs_t0 = 800; c->fs_t1 = 3000; c->fdt_t0 = 3600; c->fdt_t1 = 4400;
        c = g2m_new(out, &n, "bad_hard", sr, fft, 11000);
        g2_stim_bad(&c->stim, 0);
        c->n_ev = 2; c->ev[0].t_ms = 2800; c->ev[0].type = G2E_REVERT; c->ev[0].a = 160;
        c->ev[1].t_ms = 6800; c->ev[1].type = G2E_REVERT; c->ev[1].a = -160;
        c = g2m_new(out, &n, "bad_soft", sr, fft, 11000);
        g2_stim_bad(&c->stim, 1);
        c->n_ev = 2; c->ev[0].t_ms = 2800; c->ev[0].type = G2E_REVERT; c->ev[0].a = 160;
        c->ev[1].t_ms = 6800; c->ev[1].type = G2E_REVERT; c->ev[1].a = -160;
        c = g2m_new(out, &n, "bad_adv", sr, fft, 11000);
        g2_stim_bad(&c->stim, 0);
        c->n_ev = 1; c->ev[0].t_ms = 2800; c->ev[0].type = G2E_REVERT; c->ev[0].a = -160;      /* the revert goes forward in delay */
        c = g2m_new(out, &n, "jump_hard", sr, fft, 9000);
        g2_stim_jump(&c->stim, 0);
    }
    for (int g = 0; g < 3; g += 2) {                    /* 16k/256 and 48k/1024 */
        const int sr = grid[g][0], fft = grid[g][1];
        G2MonoCase* c;
        c = g2m_new(out, &n, "drift", sr, fft, 12000);
        memset(&c->stim, 0, sizeof c->stim);
        g2_seg(c->stim.far, &c->stim.n_far, 100, 11900); c->stim.far_amp = 0.12f;
        for (int k = 0; k < 8; ++k) g2_dly(&c->stim, k ? 2000 + 1200 * (k - 1) : 0, 640 + 4 * k);
        c = g2m_new(out, &n, "reset_mid", sr, fft, 8200);
        g2_stim_satdt(&c->stim);
        c->n_ev = 2;
        c->ev[0].t_ms = 3000; c->ev[0].type = G2E_RESET;
        c->ev[1].t_ms = 6500; c->ev[1].type = G2E_RESET;
        c = g2m_new(out, &n, "external", sr, fft, 10000);
        c->delay_mode = 2;
        memset(&c->stim, 0, sizeof c->stim);
        g2_seg(c->stim.far, &c->stim.n_far, 100, 9900); c->stim.far_amp = 0.12f;
        g2_seg(c->stim.near, &c->stim.n_near, 5000, 6200); c->stim.near_amp = 0.25f;
        g2_dly(&c->stim, 0, 0); g2_dly(&c->stim, 400, 8); g2_dly(&c->stim, 3000, 24); g2_dly(&c->stim, 5200, 14);
        g2_dly(&c->stim, 6500, 914); g2_dly(&c->stim, 8000, 14);
        c->n_ev = 6;
        c->ev[0].t_ms = 400;  c->ev[0].type = G2E_REALIGN; c->ev[0].a = 8;      /* not cancelling yet: filter reset */
        c->ev[1].t_ms = 600;  c->ev[1].type = G2E_REALIGN; c->ev[1].a = 0;      /* no change */
        c->ev[2].t_ms = 3000; c->ev[2].type = G2E_REALIGN; c->ev[2].a = 16;     /* cancelling: warm */
        c->ev[3].t_ms = 5200; c->ev[3].type = G2E_REALIGN; c->ev[3].a = -10;    /* warm, backwards */
        c->ev[4].t_ms = 6500; c->ev[4].type = G2E_REALIGN; c->ev[4].a = 900;    /* beyond the tap span: filter reset */
        c->ev[5].t_ms = 8000; c->ev[5].type = G2E_REALIGN; c->ev[5].a = -900;
    }
    for (int g = 0; g < 2; ++g) {                       /* FIXED with levels in flight, 16k/256 and 16k/512 */
        G2MonoCase* c = g2m_new(out, &n, "fixed", grid[g][0], grid[g][1], 8200);
        c->delay_mode = 1;
        g2_stim_satdt(&c->stim);
        c->n_ev = 2;
        c->ev[0].t_ms = 2000; c->ev[0].type = G2E_LEVEL; c->ev[0].a = 2;
        c->ev[1].t_ms = 4500; c->ev[1].type = G2E_LEVEL; c->ev[1].a = 0;
    }
    for (int k = 0; k < 4; ++k) {                       /* levels 0..3 with ramps in flight */
        char nm[28]; G2MonoCase* c;
        snprintf(nm, sizeof nm, "level%d", k);
        c = g2m_new(out, &n, nm, 16000, 256, 8200);
        c->level = k;
        g2_stim_satdt(&c->stim);
        c->n_ev = 4;
        c->ev[0].t_ms = 1500; c->ev[0].type = G2E_LEVEL; c->ev[0].a = (k + 1) % 4;
        c->ev[1].t_ms = 3800; c->ev[1].type = G2E_LEVEL; c->ev[1].a = (k + 2) % 4;
        c->ev[2].t_ms = 4600; c->ev[2].type = G2E_LEVEL; c->ev[2].a = (k + 3) % 4;
        c->ev[3].t_ms = 5200; c->ev[3].type = G2E_LEVEL; c->ev[3].a = 1;
    }
    {
        static const struct { const char* name; int tog; } tg[5] = {
            { "noshadow", G2T_NOSHADOW }, { "nohpf", G2T_NOHPF }, { "nocng", G2T_NOCNG }, { "nosat", G2T_NOSAT },
            { "n2win", G2T_N2WIN } };
        for (int k = 0; k < 5; ++k) {
            G2MonoCase* c = g2m_new(out, &n, tg[k].name, 16000, 256, 8200);
            c->tog = tg[k].tog;
            g2_stim_satdt(&c->stim);
        }
        /* the live observer reads the far end 24 ms before the delay in force: it needs an earlier path to corroborate */
        for (int k = 0; k < 2; ++k) {
            G2MonoCase* c = g2m_new(out, &n, k ? "observer" : "observer_off", 16000, 256, 8200);
            c->tog = k ? G2T_OBSERVER : 0;
            g2_stim_satdt(&c->stim);
            c->stim.pre_gain = 0.7f; c->stim.pre_adv16 = 384;
        }
    }
    {
        G2MonoCase* c = g2m_new(out, &n, "duty", 16000, 256, 9000);
        c->duty_s = 1.0f;
        g2_stim_jump(&c->stim, 0);
        c = g2m_new(out, &n, "grid8k", 8000, 256, 8200);
        g2_stim_satdt(&c->stim);
    }
    for (int g = 1; g < 3; ++g) {                       /* the families above on the other two grids */
        static const struct { const char* name; int tog; } tg2[7] = {
            { "noshadow", G2T_NOSHADOW }, { "nohpf", G2T_NOHPF }, { "nocng", G2T_NOCNG }, { "nosat", G2T_NOSAT },
            { "n2win", G2T_N2WIN }, { "observer_off", 0 }, { "observer", G2T_OBSERVER } };
        const int sr = grid[g][0], fft = grid[g][1];
        G2MonoCase* c = g2m_new(out, &n, "level1", sr, fft, 8200);
        g2_stim_satdt(&c->stim);
        c->n_ev = 4;
        c->ev[0].t_ms = 1500; c->ev[0].type = G2E_LEVEL; c->ev[0].a = 2;
        c->ev[1].t_ms = 3800; c->ev[1].type = G2E_LEVEL; c->ev[1].a = 3;
        c->ev[2].t_ms = 4600; c->ev[2].type = G2E_LEVEL; c->ev[2].a = 0;
        c->ev[3].t_ms = 5200; c->ev[3].type = G2E_LEVEL; c->ev[3].a = 1;
        for (int k = 0; k < 7; ++k) {
            c = g2m_new(out, &n, tg2[k].name, sr, fft, 8200);
            c->tog = tg2[k].tog;
            g2_stim_satdt(&c->stim);
            if (k >= 5) { c->stim.pre_gain = 0.7f; c->stim.pre_adv16 = 384; }
        }
        c = g2m_new(out, &n, "duty", sr, fft, 9000);
        c->duty_s = 1.0f;
        g2_stim_jump(&c->stim, 0);
        if (g == 2) {
            c = g2m_new(out, &n, "fixed", sr, fft, 8200);
            c->delay_mode = 1;
            g2_stim_satdt(&c->stim);
            c->n_ev = 2;
            c->ev[0].t_ms = 2000; c->ev[0].type = G2E_LEVEL; c->ev[0].a = 2;
            c->ev[1].t_ms = 4500; c->ev[1].type = G2E_LEVEL; c->ev[1].a = 0;
        }
    }
    return n;
}

/* Forces the bad-relock judgment of the matched-delay path to fire on the next hop: the snapshot (taps, shadow taps, H_error)
 * is taken from the instance now (taps halved so that a restore is visible), the pre-shift delay is `back` samples (at 16 kHz,
 * may be negative) before the delay in force and the pre-shift ERLE is out of reach. Writes the instance's public fields,
 * which are the same in the old and the new tree. No effect before the first acquisition or outside MATCHED. */
static void g2_force_bad_relock(Aec* a, int back16, int sr) {
    if (!a->has_delay || a->current_delay < 0 || !a->delay_shift_judgment_pre_W || !a->delay_shift_judgment_pre_H_error) return;
    const size_t w_elems = (size_t)a->n_partitions * (size_t)a->n_freqs;
    memcpy(a->delay_shift_judgment_pre_W, a->main_filter.base.W, w_elems * sizeof(Complex));
    for (size_t i = 0; i < w_elems; ++i) { a->delay_shift_judgment_pre_W[i].r *= 0.5f; a->delay_shift_judgment_pre_W[i].i *= 0.5f; }
    if (a->has_shadow && a->delay_shift_judgment_pre_shadow_W) {
        memcpy(a->delay_shift_judgment_pre_shadow_W, a->shadow_filter.W, w_elems * sizeof(Complex));
        for (size_t i = 0; i < w_elems; ++i) { a->delay_shift_judgment_pre_shadow_W[i].r *= 0.5f; a->delay_shift_judgment_pre_shadow_W[i].i *= 0.5f; }
    }
    memcpy(a->delay_shift_judgment_pre_H_error, a->main_filter.H_error_per_bin, (size_t)a->n_freqs * sizeof(float));
    a->delay_shift_judgment_pre_delay = a->current_delay - (int)((int64_t)back16 * sr / 16000);
    if (a->delay_shift_judgment_pre_delay < 0) a->delay_shift_judgment_pre_delay = 0;
    a->delay_shift_judgment_pre_erle = 1000.0f;
    a->delay_shift_judgment_hops_left = 1;
    a->delay_shift_judgment_active = 1;
}

/* ---- hashing helpers ------------------------------------------------------------------------------------------------ */
/* The delay / shift-judgment / near-end-recent / duty state of the instance (public struct fields). */
static uint64_t g2_hash_state(const Aec* a) {
    uint64_t h = G2_FNV_BASIS;
    h = g2_fnv_u32(h, (uint32_t)a->current_delay);
    h = g2_fnv_u32(h, (uint32_t)a->has_pending);
    h = g2_fnv_u32(h, (uint32_t)a->pending_delay);
    h = g2_fnv_u32(h, (uint32_t)a->pending_delay_ttl);
    h = g2_fnv_u32(h, (uint32_t)a->delay_shift_judgment_active);
    h = g2_fnv_u32(h, (uint32_t)a->delay_shift_judgment_pre_delay);
    h = g2_fnv_u32(h, (uint32_t)a->delay_shift_judgment_hops_left);
    h = g2_fnv_f32(h, a->delay_shift_judgment_pre_erle);
    h = g2_fnv_u32(h, (uint32_t)a->delay_reject_count);
    for (int i = 0; i < AEC_DELAY_REJECT_CAP; ++i) h = g2_fnv_u32(h, (uint32_t)a->delay_reject_values[i]);
    h = g2_fnv_u32(h, a->delay_generation);
    h = g2_fnv_u32(h, (uint32_t)a->ne_recent_frames);
    h = g2_fnv_u32(h, (uint32_t)a->ne_above);
    h = g2_fnv_u32(h, (uint32_t)a->duty.active);
    h = g2_fnv_u32(h, (uint32_t)a->duty.pos);
    h = g2_fnv_u32(h, (uint32_t)a->a3_sg.dt_protect_active);
    return h;
}

enum { G2G_LIN, G2G_FLAGS, G2G_STATE, G2G_SPEC, G2G_GAIN, G2G_APPLY, G2G_OUT, G2G_ARES, G2G_COUNT };
static const char* const G2G_NAME[G2G_COUNT] = { "lin", "flags", "state", "spec", "gain", "apply", "out", "ares" };

/* A synthetic denoiser gain (no libm): a triangle wave in [0.35, 0.95] that moves with k and the hop. */
static void g2_extra_gain(float* gx, int n, int hop) {
    for (int k = 0; k < n; ++k) {
        int ph = (k * 3 + hop * 5) % 40, tri = ph < 20 ? ph : 40 - ph;
        gx[k] = 0.35f + 0.6f * ((float)tri / 20.0f);
    }
}

typedef struct { uint32_t rng, rng_null, serial; } G2ApplyState;

/* The gain / comfort-noise computations of an application over the base spectrum; returns one hash. */
static uint64_t g2_apply_hash(G2ApplyState* st, const Aec* a, int hop, const float* gain, const Complex* base,
                              const float* cn, int n, const void* hook_view) {
    float gx[G2_NB_MAX], gm[G2_NB_MAX];
    Complex p1[G2_NB_MAX], p2[G2_NB_MAX], p3[G2_NB_MAX], p4[G2_NB_MAX];
    uint64_t h = G2_FNV_BASIS;
    uint32_t serial = aec_post_reset_serial(a);
    (void)hook_view;
    if (serial != st->serial) { st->rng = st->rng_null = AEC_CNG_SEED; st->serial = serial; }
    g2_extra_gain(gx, n, hop);
    sk_min_f32(gm, gain, gx, (size_t)n);
    sk_capply_gain_f32(p1, base, gm, (size_t)n);
    memcpy(p2, p1, (size_t)n * sizeof(Complex));
    aec_cng_fill(&st->rng, p2, cn, gain, gx, 0.31622777f, n);
    sk_capply_gain_f32(p3, base, gain, (size_t)n);
    memcpy(p4, p3, (size_t)n * sizeof(Complex));
    aec_cng_fill(&st->rng_null, p4, cn, gain, NULL, 0.0f, n);
    h = g2_hc(h, p1, n); h = g2_hc(h, p2, n); h = g2_hc(h, p3, n); h = g2_hc(h, p4, n);
#ifdef G2_HOOK_APPLY
    G2_HOOK_APPLY(a, hook_view, hop, gx, p1, p2, p3, p4);
#endif
    return h;
}

/* ---- census of what a case reached (stderr; not part of the line stream) --------------------------------------------------- */
typedef struct {
    int hops, dt02, rfp, sat05, drift, guard, usable, conv, ne_recent, gen_changes, serial_moves, judged, reverts;
    int ares_drift, ares_sat, ares_dt;
    int rc_warm, rc_reset, rc_zero;
} G2Census;

static void g2_census_print(const char* grid, const char* cs, const G2Census* c, const Aec* a) {
    fprintf(stderr, "  census %-10s %-12s hops %5d dt>0.2 %4d rfp %4d sat>0.5 %4d drift %4d guard %4d usable %4d conv %4d "
            "ne_recent %4d gen+ %2d serial+ %2d judged %d reverts %d rejects %d realign(w/r/0) %d/%d/%d ares(d/s/dt) %d/%d/%d duty %llu/%llu\n",
            grid, cs, c->hops, c->dt02, c->rfp, c->sat05, c->drift, c->guard, c->usable, c->conv, c->ne_recent,
            c->gen_changes, c->serial_moves, c->judged, c->reverts, a->delay_reject_count, c->rc_warm, c->rc_reset, c->rc_zero,
            c->ares_drift, c->ares_sat, c->ares_dt, a->duty.hops_run, a->duty_hops_total);
}

/* ---- one case ------------------------------------------------------------------------------------------------------- */
static void g2m_apply_cfg(AecTestCfg* tc, const G2MonoCase* c, int ctx) {
    aec_tcfg_from_level(tc, (AecLevel)c->level, c->sr);
    tc->c.fft_size = c->fft;
    tc->c.delay_mode = c->delay_mode == 0 ? AEC_DELAY_MATCHED : c->delay_mode == 1 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    tc->c.fixed_delay_samples = c->delay_mode == 1 ? (int)((int64_t)c->stim.dly[0].d16 * c->sr / 16000) : -1;
    if (c->tog & G2T_NOSHADOW)  tc->t.enable_shadow = 0;
    if (c->tog & G2T_NOHPF)     tc->t.enable_highpass = 0;
    if (c->tog & G2T_NOCNG)     tc->c.enable_cng = 0;
    if (c->tog & G2T_NOSAT)     tc->t.enable_saturation = 0;
    if (c->tog & G2T_OBSERVER)  tc->t.live_observer_enabled = 1;
    if (c->tog & G2T_N2WIN)     tc->t.cng_n2_windowed_psd = 1;
    tc->t.delay_est_period_s = c->duty_s;
    g2_mode_cfg(&tc->c, ctx);
}

/* G2_ONLY=<substring>: run only the cases whose "<grid> <name>" contains it (development aid; take a baseline without it). */
static int g2m_want(const G2MonoCase* c) {
    const char* f = getenv("G2_ONLY");
    char id[64];
    if (!f || !*f) return 1;
    snprintf(id, sizeof id, "%d/%d %s", c->sr, c->fft, c->name);
    return strstr(id, f) != NULL;
}

static int g2m_run_case(const G2MonoCase* c) {
    AecTestCfg cfg_c, cfg_o;
    void *mem_c = NULL, *mem_o = NULL, *mem_r = NULL;
    AecRes* res = NULL;
    char grid[24];
    G2Census cen; memset(&cen, 0, sizeof cen);
    snprintf(grid, sizeof grid, "%d/%d", c->sr, c->fft);
    g2m_apply_cfg(&cfg_c, c, 1);
    g2m_apply_cfg(&cfg_o, c, 0);
    size_t sz_c = aec_tcfg_mem_size(&cfg_c), sz_o = aec_tcfg_mem_size(&cfg_o);
    Aec* a = sz_c ? (mem_c = g2_pool(sz_c), mem_c ? aec_tcfg_init(mem_c, sz_c, &cfg_c) : NULL) : NULL;
    Aec* o = sz_o ? (mem_o = g2_pool(sz_o), mem_o ? aec_tcfg_init(mem_o, sz_o, &cfg_o) : NULL) : NULL;
    if (!a || !o) { fprintf(stderr, "aec_init failed %s %s\n", grid, c->name); return -1; }
    const int hop = c->fft / 2;
    const int n_hops = (int)((int64_t)c->dur_ms * c->sr / 1000 / hop);
    const int ns = n_hops * hop;
    float* ref = (float*)malloc((size_t)ns * sizeof(float));
    float* mic = (float*)malloc((size_t)ns * sizeof(float));
    float out[1024];
    G2View v;
    G2ApplyState ast = { AEC_CNG_SEED, AEC_CNG_SEED, 0u }, rst = { AEC_CNG_SEED, AEC_CNG_SEED, 0u };
    uint64_t chain[G2G_COUNT], hh[G2G_COUNT];
    int ev_hop[8], emit_next = 0, ev_now;
    unsigned last_gen = 0; uint32_t last_serial = 0; int last_judged = 0, last_rejects = 0;
    int jwas = 0, j_rej = 0, j_shift_h = 0, j_old = 0, j_new = 0, j_soft = 0; uint32_t j_serial = 0;
    float j_pe = 0.0f, j_np = 0.0f, j_ep = 0.0f;
    g2_jn = 0;
    if (!ref || !mic) return -1;
    g2_mono_signal(&c->stim, c->sr, ns, ref, mic);
    for (int g = 0; g < G2G_COUNT; ++g) chain[g] = G2_FNV_BASIS;
    for (int e = 0; e < c->n_ev; ++e) ev_hop[e] = (int)((int64_t)c->ev[e].t_ms * c->sr / 1000 / hop);
    if (c->ares) {
        size_t rsz = aec_res_get_mem_size(&cfg_c.c);
        mem_r = g2_pool(rsz);
        res = (rsz && mem_r) ? aec_res_init(mem_r, rsz, &cfg_c.c) : NULL;
        if (!res) { fprintf(stderr, "aec_res_init failed\n"); return -1; }
    }
    {
        uint64_t hi = G2_FNV_BASIS;
        hi = g2_fnv_f32v(hi, ref, ns); hi = g2_fnv_f32v(hi, mic, ns);
        g2_line(grid, c->name, "INPUT", 0, hi);
        g2_line(grid, c->name, "META_hops", 0, (uint64_t)n_hops);
        g2_line(grid, c->name, "META_hop", 0, (uint64_t)hop);
    }
    for (int h = 0; h < n_hops; ++h) {
        const float* r = ref + (size_t)h * hop;
        const float* m = mic + (size_t)h * hop;
        ev_now = 0;
        for (int e = 0; e < c->n_ev; ++e) {
            if (ev_hop[e] != h) continue;
            ev_now = 1;
            if (c->ev[e].type == G2E_RESET) {
                aec_reset(a); aec_reset(o);
                if (res) aec_res_reset(res);
            } else if (c->ev[e].type == G2E_LEVEL) {
                int r1 = aec_set_level(a, (AecLevel)c->ev[e].a);
                int r2 = aec_set_level(o, (AecLevel)c->ev[e].a);
                if (res) aec_res_set_level(res, (AecLevel)c->ev[e].a);
                g2_line(grid, c->name, "EV_level", h, (uint64_t)(uint32_t)(r1 * 2 + r2 + 7));
            } else if (c->ev[e].type == G2E_REVERT) {
                g2_force_bad_relock(a, c->ev[e].a, c->sr);
                g2_force_bad_relock(o, c->ev[e].a, c->sr);
            } else {
                int delta = (int)((int64_t)c->ev[e].a * c->sr / 16000);
                int r1 = aec_apply_external_realign(a, delta);
                int r2 = aec_apply_external_realign(o, delta);
                if (r1 == 1) cen.rc_warm++; else if (r1 == 0 && delta != 0) cen.rc_reset++; else cen.rc_zero++;
                g2_line(grid, c->name, "EV_realign", h, (uint64_t)(uint32_t)(r1 + 3) * 16u + (uint64_t)(uint32_t)(r2 + 3));
            }
        }
        if (g2_step_ctx(a, m, r, &v) != 0) { fprintf(stderr, "ctx step failed %s %s hop %d\n", grid, c->name, h); return -1; }
        if (g2_step_out(o, m, r, out) != 0) { fprintf(stderr, "out step failed %s %s hop %d\n", grid, c->name, h); return -1; }
        {
            const int n = v.n_freqs;
            uint64_t x;
            x = G2_FNV_BASIS;
            x = g2_fnv_f32v(x, v.linear_hop, v.hop_size);
            x = g2_hc(x, v.far_spec, n);
            x = g2_fnv_f32v(x, v.lin_formed, v.hop_size);
            x = g2_fnv_f32v(x, v.aligned_far, v.hop_size);
            x = g2_fnv_u32(x, (uint32_t)v.hop_size); x = g2_fnv_u32(x, (uint32_t)v.delay_samples);
            x = g2_fnv_f32(x, v.delay_conf); x = g2_fnv_u32(x, (uint32_t)v.delay_state); x = g2_fnv_u32(x, v.generation);
            hh[G2G_LIN] = x;
            x = G2_FNV_BASIS;
            x = g2_fnv_u32(x, (uint32_t)v.usable_linear); x = g2_fnv_u32(x, (uint32_t)v.filter_converged);
            x = g2_fnv_u32(x, (uint32_t)v.epc_active); x = g2_fnv_u32(x, (uint32_t)v.is_stationary_dt);
            x = g2_fnv_u32(x, (uint32_t)v.res_floor_protect); x = g2_fnv_u32(x, (uint32_t)v.guard_active);
            x = g2_fnv_u32(x, (uint32_t)v.filter_once_converged); x = g2_fnv_u32(x, (uint32_t)v.clock_drift);
            x = g2_fnv_f32(x, v.dt_indicator); x = g2_fnv_f32(x, v.far_power); x = g2_fnv_f32(x, v.erle_factor);
            x = g2_fnv_f32(x, v.saturation_level); x = g2_fnv_f32(x, v.erl_estimate); x = g2_fnv_f32(x, v.divergence);
            x = g2_fnv_f32(x, v.shadow_dt);
            x = g2_fnv_u32(x, aec_post_reset_serial(a));
            hh[G2G_FLAGS] = x;
            hh[G2G_STATE] = g2_hash_state(a);
            x = G2_FNV_BASIS;
            x = g2_hc(x, v.error_spec, n); x = g2_hc(x, v.echo_spec, n); x = g2_hc(x, v.near_spec, n);
            x = g2_fnv_f32v(x, v.formed_hop, v.hop_size); x = g2_hc(x, v.base_spec, n);
            hh[G2G_SPEC] = x;
            x = G2_FNV_BASIS;
            x = g2_fnv_f32v(x, v.r2, n); x = g2_fnv_f32v(x, v.comfort_noise, n); x = g2_fnv_f32v(x, v.res_gain, n);
            hh[G2G_GAIN] = x;
            hh[G2G_APPLY] = g2_apply_hash(&ast, a, h, v.res_gain, v.base_spec, v.comfort_noise, n, &v);
            hh[G2G_OUT] = g2_fnv_f32v(G2_FNV_BASIS, out, hop);
            hh[G2G_ARES] = 0;
            if (res) {
                AecResInput in;
                const int tms = (int)((int64_t)h * hop * 1000 / c->sr);
                const int fd = (c->fd_t1 > c->fd_t0 && tms >= c->fd_t0 && tms < c->fd_t1);
                const int fs = (c->fs_t1 > c->fs_t0 && tms >= c->fs_t0 && tms < c->fs_t1);
                const int fdt = (c->fdt_t1 > c->fdt_t0 && tms >= c->fdt_t0 && tms < c->fdt_t1);
                memset(&in, 0, sizeof in);
                in.error_spec = v.error_spec; in.near_spec = v.near_spec; in.r2 = v.r2; in.comfort_noise = v.comfort_noise;
                in.render_hop = r; in.all_converged = v.filter_converged;
                in.max_dt = fdt ? 0.5f : v.dt_indicator;
                in.max_saturation = fs ? 0.9f : v.saturation_level;
                in.clock_drift = fd || v.clock_drift;
                cen.ares_drift += in.clock_drift != 0; cen.ares_sat += in.max_saturation > 0.5f; cen.ares_dt += in.max_dt > 0.2f;
                const float* gr = aec_res_process(res, &in);
                if (!gr) { fprintf(stderr, "aec_res_process NULL\n"); return -1; }
                x = g2_fnv_f32v(G2_FNV_BASIS, gr, n);
                x = g2_chain(x, g2_apply_hash(&rst, a, h, gr, v.base_spec, v.comfort_noise, n, NULL));
                hh[G2G_ARES] = x;
            }
            if (getenv("G2_TRACE"))
                fprintf(stderr, "T %s %s h%d t%d dt %.3f epc %d conv %d once %d usable %d sat %.2f rfp %d ner %d guard %d drift %d gen %u cd %d stat %d erle %.2f pre %.2f act %d left %d np %.2e rp %.2e\n",
                        grid, c->name, h, (int)((int64_t)h * hop * 1000 / c->sr), v.dt_indicator, v.epc_active, v.filter_converged,
                        v.filter_once_converged, v.usable_linear, v.saturation_level, v.res_floor_protect, a->ne_recent_frames,
                        v.guard_active, v.clock_drift, v.generation, a->current_delay, v.is_stationary_dt, a->last_erle_windowed,
                        a->delay_shift_judgment_pre_erle, a->delay_shift_judgment_active, a->delay_shift_judgment_hops_left, a->near_power, a->raw_error_power);
            /* census */
            cen.hops++; cen.dt02 += v.dt_indicator > 0.2f; cen.rfp += v.res_floor_protect; cen.sat05 += v.saturation_level > 0.5f;
            cen.drift += v.clock_drift; cen.guard += v.guard_active; cen.usable += v.usable_linear; cen.conv += v.filter_converged;
            cen.ne_recent += a->ne_recent_frames > 0;
            if (v.generation != last_gen) { cen.gen_changes++; last_gen = v.generation; }
            if (aec_post_reset_serial(a) != last_serial) { cen.serial_moves++; last_serial = aec_post_reset_serial(a); }
            if (a->delay_shift_judgment_active && !last_judged) cen.judged++;
            last_judged = a->delay_shift_judgment_active;
            if (a->delay_reject_count != last_rejects) { cen.reverts++; last_rejects = a->delay_reject_count; }
            if (c->judge) {
                /* The judgment runs at the start of a hop on the previous hop's readings (j_*). */
                if (jwas && !a->delay_shift_judgment_active && g2_jn < G2_MAX_JUDGE) {
                    G2JudgeRec* r = &g2_jrec[g2_jn++];
                    uint64_t x = G2_FNV_BASIS;
                    r->shift_hop = j_shift_h; r->hop = h; r->old_delay = j_old; r->new_delay = j_new; r->soft = j_soft;
                    r->cur = j_pe; r->pre = a->delay_shift_judgment_pre_erle; r->np = j_np; r->ep = j_ep;
                    r->has_est = j_ep < j_np; r->reverted = a->delay_reject_count > j_rej;
                    x = g2_fnv_u32(x, (uint32_t)r->reverted); x = g2_fnv_u32(x, (uint32_t)r->has_est);
                    x = g2_fnv_f32(x, r->cur); x = g2_fnv_f32(x, r->pre); x = g2_fnv_f32(x, r->np); x = g2_fnv_f32(x, r->ep);
                    x = g2_fnv_u32(x, (uint32_t)a->current_delay); x = g2_fnv_u32(x, (uint32_t)a->delay_reject_count);
                    x = g2_fnv_u32(x, a->delay_generation);
                    g2_line(grid, c->name, "EV_judge", h, x);
                }
                if (!jwas && a->delay_shift_judgment_active) {
                    j_shift_h = h; j_old = a->delay_shift_judgment_pre_delay; j_new = a->current_delay;
                    j_soft = aec_post_reset_serial(a) == j_serial;      /* the hard reset moves the serial, the soft realign does not */
                }
                jwas = a->delay_shift_judgment_active; j_rej = a->delay_reject_count; j_serial = aec_post_reset_serial(a);
                j_pe = a->last_erle_windowed; j_np = a->near_power; j_ep = a->raw_error_power;
            }
        }
        for (int g = 0; g < G2G_COUNT; ++g) {
            if (g == G2G_ARES && !res) continue;
            g2_vline(grid, c->name, G2G_NAME[g], h, hh[g]);
            chain[g] = g2_chain(chain[g], hh[g]);
        }
        if ((h % G2_STRIDE) == G2_STRIDE - 1 || h == n_hops - 1 || emit_next || ev_now) {
            for (int g = 0; g < G2G_COUNT; ++g) {
                if (g == G2G_ARES && !res) continue;
                g2_line(grid, c->name, G2G_NAME[g], h, chain[g]);
            }
        }
        emit_next = ev_now;      /* the hop after an event is printed too */
    }
    g2_line(grid, c->name, "END_generation", 0, (uint64_t)a->delay_generation);
    g2_line(grid, c->name, "END_reject_count", 0, (uint64_t)(uint32_t)a->delay_reject_count);
    g2_line(grid, c->name, "END_serial", 0, (uint64_t)aec_post_reset_serial(a));
    g2_line(grid, c->name, "END_current_delay", 0, (uint64_t)(uint32_t)a->current_delay);
    g2_line(grid, c->name, "END_far_fft", 0, (uint64_t)aec_far_fft_real_compute_count(a));
    g2_line(grid, c->name, "END_duty_total", 0, (uint64_t)a->duty_hops_total);
    g2_line(grid, c->name, "END_duty_run", 0, (uint64_t)a->duty.hops_run);
    g2_census_print(grid, c->name, &cen, a);
    if (c->judge) {
        int rv = 0;
        for (int i = 0; i < g2_jn; ++i) rv += g2_jrec[i].reverted;
        fprintf(G2_JOUT, "  judge %-9s %-12s %d judgments: %d natural reverts, %d kept\n", grid, c->name, g2_jn, rv, g2_jn - rv);
        for (int i = 0; i < g2_jn; ++i) {
            const G2JudgeRec* r = &g2_jrec[i];
            fprintf(G2_JOUT, "    #%d shift hop %d (%d ms) delay %d -> %d %s, judged hop %d (%d ms): cur %.2f pre %.2f delta %+.2f dB, "
                    "estimate %d (err/near %.2f) -> %s\n", i, r->shift_hop, (int)((int64_t)r->shift_hop * hop * 1000 / c->sr),
                    r->old_delay, r->new_delay, r->soft ? "soft" : "hard", r->hop, (int)((int64_t)r->hop * hop * 1000 / c->sr),
                    r->cur, r->pre, r->cur - r->pre, r->has_est, r->np > 0.0f ? r->ep / r->np : 0.0f,
                    r->reverted ? "REVERTED" : "kept");
        }
    }
    free(ref); free(mic); free(mem_c); free(mem_o); free(mem_r);
    return 0;
}

#endif /* AEC_TEST_GOLDEN2_MONO_H */
