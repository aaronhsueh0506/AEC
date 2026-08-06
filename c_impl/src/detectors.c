/* detectors.c — RenderActivity / FilterConvergence / DoubleTalk ports.
 * All scalar math in float32 by design (Python fp64 bit-exact parity
 * retired; the resulting drift is accepted).
 */
#include "detectors.h"
#include "aec3_scale.h"
#include <math.h>

/* Every wall-clock constant below is authored against this repo's legacy
 * hop=160 @ 16000 Hz (10.000 ms) grid and retimed to the live grid in the
 * *_init functions. Mirrors python/modules/detectors.py; the values are
 * gated against it by test/modules/parity_detectors.c. */
#define DET_REF_HOP 160
#define DET_REF_SR  16000
#define DET_REHOP(v, hop, sr) \
    aec3_growth_rehop((v), DET_REF_HOP, DET_REF_SR, (hop), (sr))

/* ── RenderActivityDetector ──────────────────────────────────── */

static const float RA_ALPHA_CV_REF = 0.99f;   /* TC ~ 1 s at the 10 ms ref hop */
/* Dimensionless CV^2 level gate -- no time content, so not retimed. */
static const float RA_STATIONARY_C = 0.02f;

void render_activity_init(RenderActivity* r, float* pairwise_scratch,
                           int pairwise_scratch_len,
                           int hop_size, int sample_rate) {
    r->alpha_cv = DET_REHOP(RA_ALPHA_CV_REF, hop_size, sample_rate);
    r->env_mean      = 1e-10f;
    r->env_var       = 0.0f;
    r->active_prev   = 0;
    r->is_stationary = 0;
    r->pairwise_scratch     = pairwise_scratch;
    r->pairwise_scratch_len = pairwise_scratch_len;
}
void render_activity_reset(RenderActivity* r) {
    /* Scalar state only -- pairwise_scratch/pairwise_scratch_len are
     * caller-owned and set once at render_activity_init; reset must not
     * clobber them (there is no pool-carved buffer to re-supply here). */
    r->env_mean      = 1e-10f;
    r->env_var       = 0.0f;
    r->active_prev   = 0;
    r->is_stationary = 0;
}

RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n) {
    /* Guard (clang-analyzer): n<=0 would leave `blocks[0]` unwritten below
     * (n_blocks would be 0, or negative n would skip the fill loop entirely)
     * and then read it uninitialized at `far_pwr_raw = blocks[0] / n`. Every
     * real call site passes n == hop_size (always > 0), so this never fires
     * in production — pure guard, no behavior change for valid inputs.
     * Returns current state unchanged, far_pwr floored the same way the
     * silent-far branch below would (`far_pwr_raw=0` + epsilon). */
    if (n <= 0) {
        RenderActivityResult res0 = {
            .far_pwr       = 1e-10f,
            .is_active     = r->active_prev,
            .is_stationary = r->is_stationary,
            .warmup_active = 0,
        };
        return res0;
    }
    /* far_pwr_raw = mean(far_end ** 2), computed in float32.
     *
     * np.mean of an fp32 array uses pairwise summation in fp32 then divides
     * (also fp32). To reproduce numpy's pairwise summation we use 8-element
     * blocks summed linearly, then a tree reduction over the block sums —
     * this mirrors numpy's `pairwise_sum_FLOAT`
     * (`numpy/core/src/umath/loops_utils.h.src`).
     */
    enum { PAIR_BLOCK = 8 };
    int n_blocks = (n + PAIR_BLOCK - 1) / PAIR_BLOCK;
    /* Guard: pairwise_scratch is caller-owned (pool-carved by aec_carve,
     * sized ceil(hop/PAIR_BLOCK) from the same runtime hop passed as `n`
     * here), so this never fires in production -- but a corrupted or
     * mis-sized carve must not silently overrun caller memory. Mirrors the
     * n<=0 guard above: return current state unchanged. */
    if (n_blocks > r->pairwise_scratch_len) {
        RenderActivityResult res_guard = {
            .far_pwr       = 1e-10f,
            .is_active     = r->active_prev,
            .is_stationary = r->is_stationary,
            .warmup_active = 0,
        };
        return res_guard;
    }
    float* blocks = r->pairwise_scratch;
    for (int b = 0; b < n_blocks; ++b) {
        int start = b * PAIR_BLOCK;
        int end   = start + PAIR_BLOCK;
        if (end > n) end = n;
        float s = 0.0f;
        for (int i = start; i < end; ++i) {
            float sq = far_end[i] * far_end[i];
            s += sq;
        }
        blocks[b] = s;
    }
    /* Tree reduction over blocks (fp32) */
    int len = n_blocks;
    while (len > 1) {
        int half = len / 2;
        for (int i = 0; i < half; ++i) {
            blocks[i] = blocks[2 * i] + blocks[2 * i + 1];
        }
        if (len & 1) blocks[half] = blocks[len - 1], half++;
        len = half;
    }
    float far_pwr_raw = blocks[0] / (float)n;
    float far_pwr     = far_pwr_raw + 1e-10f;
    int   warmup_active = far_pwr_raw > 1e-6f ? 1 : 0;

    if (far_pwr > 1e-6f) {
        if (!r->active_prev) {
            r->env_mean    = far_pwr;
            r->env_var     = 0.0f;
            r->active_prev = 1;
            /* One observation has no variance estimate. Match the Python
             * detector and wait for the next active hop before declaring
             * the render stationary. */
            r->is_stationary = 0;
        } else {
            float old_mean = r->env_mean;
            r->env_mean = r->alpha_cv * r->env_mean
                        + (1.0f - r->alpha_cv) * far_pwr;
            float diff = far_pwr - old_mean;
            r->env_var = r->alpha_cv * r->env_var
                       + (1.0f - r->alpha_cv) * (diff * diff);
            float far_cv2 = r->env_var /
                            (r->env_mean * r->env_mean + 1e-10f);
            r->is_stationary = (far_cv2 < RA_STATIONARY_C) ? 1 : 0;
        }
    } else {
        r->active_prev   = 0;
        r->is_stationary = 0;
    }
    RenderActivityResult res = {
        .far_pwr       = far_pwr,
        .is_active     = r->active_prev,
        .is_stationary = r->is_stationary,
        .warmup_active = warmup_active,
    };
    return res;
}

/* ── FilterConvergenceAnalyzer ───────────────────────────────── */

static const float FC_CONV_ERLE_DB = 5.0f;
static const int   FC_CONV_FRAMES  = 10;
static const float FC_DIV_ERLE_LIN = 0.63f;
static const float FC_DIV_ALPHA_REF = 0.9f;
static const float FC_DIV_DECAY_REF = 0.95f;

void filter_convergence_init(FilterConvergence* c, int hop_size, int sample_rate) {
    c->div_alpha = DET_REHOP(FC_DIV_ALPHA_REF, hop_size, sample_rate);
    c->div_decay = DET_REHOP(FC_DIV_DECAY_REF, hop_size, sample_rate);
    filter_convergence_reset(c);
}
void filter_convergence_reset(FilterConvergence* c) {
    /* State only. div_alpha/div_decay are grid-derived and set once at init;
     * reset must not clobber them (there is no hop/sample_rate to re-derive
     * from here). Same rule as render_activity_reset's scratch pointers. */
    c->converged       = 0;
    c->once_converged  = 0;
    c->conv_counter    = 0;
    c->divergence      = 0.0f;
}
void filter_convergence_mark_diverged(FilterConvergence* c) {
    c->converged    = 0;
    c->conv_counter = 0;
}
void filter_convergence_update_divergence(FilterConvergence* c,
                                             float near_power,
                                             float raw_error_power) {
    if (c->converged && near_power > 1e-8f) {
        float inst_erle_lin = near_power / (raw_error_power + 1e-10f);
        float is_div        = (inst_erle_lin < FC_DIV_ERLE_LIN) ? 1.0f : 0.0f;
        c->divergence = c->div_alpha * c->divergence
                      + (1.0f - c->div_alpha) * is_div;
    } else {
        c->divergence *= c->div_decay;
    }
}
int filter_convergence_update_convergence(FilterConvergence* c,
                                             float near_power,
                                             float raw_error_power,
                                             int far_active, int warmup_done) {
    if (c->converged || near_power <= 1e-8f || !warmup_done || !far_active) {
        return 0;
    }
    float inst_erle_db = 10.0f * log10f(near_power / (raw_error_power + 1e-10f));
    if (inst_erle_db > FC_CONV_ERLE_DB) c->conv_counter += 1;
    else                                c->conv_counter  = 0;
    if (c->conv_counter >= FC_CONV_FRAMES) {
        c->converged       = 1;
        c->once_converged  = 1;
        return 1;
    }
    return 0;
}

/* ── DoubleTalkAnalyzer ──────────────────────────────────────── */

/* Shadow-filter DT blind period; kept as a duration so all grids and both
 * implementations use the same wall-clock gate. */
static const float DT_SHADOW_FRAME_GATE_MS = 200.0f;
static const float DT_ERL_CEILING_FLOOR = 0.01f;
static const float DT_SAFETY_MARGIN     = 2.0f;
static const float DTE_RISE_OLD_REF  = 0.3f;
static const float DTE_DECAY_OLD_REF = 0.9f;  /* the "TC~90ms" hangover */
static const float DTS_OLD_REF = 0.7f;
static const float DTS_INACTIVE_DECAY_REF = 0.95f;

void doubletalk_init(DoubleTalk* d, float offset, float advantage_scale,
                        int hop_size, int sample_rate) {
    d->dt_from_energy   = 0.0f;
    d->dt_from_shadow   = 0.0f;
    d->shadow_advantage = 1.0f;
    d->shadow_dtd_offset           = offset;
    d->shadow_dtd_advantage_scale  = advantage_scale;
    d->shadow_frame_gate = aec3_ms_to_hops(DT_SHADOW_FRAME_GATE_MS,
                                           hop_size, sample_rate);
    /* Only the OLD (retention) term is retimed; NEW is 1-OLD by construction
     * so each pair keeps summing to 1 on every grid. */
    d->dte_rise_old  = DET_REHOP(DTE_RISE_OLD_REF, hop_size, sample_rate);
    d->dte_rise_new  = 1.0f - d->dte_rise_old;
    d->dte_decay_old = DET_REHOP(DTE_DECAY_OLD_REF, hop_size, sample_rate);
    d->dte_decay_new = 1.0f - d->dte_decay_old;
    d->dts_old       = DET_REHOP(DTS_OLD_REF, hop_size, sample_rate);
    d->dts_new       = 1.0f - d->dts_old;
    d->dts_inactive_decay = DET_REHOP(DTS_INACTIVE_DECAY_REF,
                                      hop_size, sample_rate);
}
void doubletalk_reset(DoubleTalk* d) {
    d->dt_from_energy   = 0.0f;
    d->dt_from_shadow   = 0.0f;
    d->shadow_advantage = 1.0f;
}

void doubletalk_update_shadow_dt(DoubleTalk* d,
                                    int shadow_frame_count, int far_excited,
                                    float main_err_smooth,
                                    float shadow_err_smooth) {
    if (shadow_frame_count >= d->shadow_frame_gate && far_excited) {
        d->shadow_advantage = main_err_smooth / (shadow_err_smooth + 1e-10f);
        float raw = (d->shadow_advantage - d->shadow_dtd_offset)
                   / d->shadow_dtd_advantage_scale;
        if (raw < 0.0f) raw = 0.0f;
        if (raw > 1.0f) raw = 1.0f;
        d->dt_from_shadow = d->dts_old * d->dt_from_shadow + d->dts_new * raw;
    } else {
        d->dt_from_shadow *= d->dts_inactive_decay;
    }
}

void doubletalk_update_energy_dt(DoubleTalk* d,
                                    int far_active, float far_pwr,
                                    float mic_pwr, float erl_estimate) {
    float inst;
    if (far_active && far_pwr > 1e-4f) {
        float erl = (erl_estimate > DT_ERL_CEILING_FLOOR)
                     ? erl_estimate : DT_ERL_CEILING_FLOOR;
        float erl_ceiling      = 1.0f / erl;
        float max_echo_expected = far_pwr * erl_ceiling * DT_SAFETY_MARGIN;
        inst = (mic_pwr - max_echo_expected) / mic_pwr;
        if (inst < 0.0f) inst = 0.0f;
    } else {
        inst = 0.0f;
    }
    if (inst > d->dt_from_energy) {
        d->dt_from_energy = d->dte_rise_old * d->dt_from_energy
                          + d->dte_rise_new * inst;
    } else {
        d->dt_from_energy = d->dte_decay_old * d->dt_from_energy
                          + d->dte_decay_new * inst;
    }
}

/* ── FilterPlateauDetector (v3.10.0 + v3.10.3 F4/H3) ─────────── */

void filter_plateau_init(FilterPlateauDetector* p) {
    p->grace_frames     = 400;
    p->erle_max_db      = 6.0f;
    p->far_active_ratio = 0.5f;
    p->dt_signal_ratio  = 0.10f;
    p->max_attempts     = 2;
    filter_plateau_reset(p);
}

void filter_plateau_reset(FilterPlateauDetector* p) {
    p->frame_count        = 0;
    p->far_active_count   = 0;
    p->dt_signal_count    = 0;
    p->consecutive_match  = 0;
    p->attempts           = 0;
    p->cooldown_remaining = 0;
    p->last_reset_frame   = -1;
}

int filter_plateau_update(FilterPlateauDetector* p,
                             int    far_active,
                             int    dt_signal_present,
                             float  erle_windowed_db,
                             int    once_converged) {
    p->frame_count++;
    if (far_active)        p->far_active_count++;
    if (dt_signal_present) p->dt_signal_count++;

    if (p->cooldown_remaining > 0) {
        p->cooldown_remaining--;
        return 0;
    }
    if (once_converged) { p->consecutive_match = 0; return 0; }
    if (p->attempts >= p->max_attempts) return 0;
    if (p->frame_count <= p->grace_frames) return 0;

    int64_t denom = p->frame_count > 1 ? p->frame_count : (int64_t)1;
    float far_ratio = (float)p->far_active_count / (float)denom;
    float dt_ratio  = (float)p->dt_signal_count  / (float)denom;

    /* v3.10.3 F4: require dt_signal_present in the current frame too. */
    int criteria_met = (far_ratio > p->far_active_ratio)
                    && (erle_windowed_db < p->erle_max_db)
                    && (dt_ratio > p->dt_signal_ratio)
                    && dt_signal_present;

    if (!criteria_met) { p->consecutive_match = 0; return 0; }

    p->consecutive_match++;
    if (p->consecutive_match < FILTER_PLATEAU_CONSECUTIVE_REQUIRED) return 0;

    /* Fire — also reset cumulative counters (v3.10.3 H3). */
    p->consecutive_match  = 0;
    p->attempts++;
    p->cooldown_remaining = FILTER_PLATEAU_POST_RESET_GRACE_FRAMES;
    p->last_reset_frame   = p->frame_count;
    p->frame_count        = 0;
    p->far_active_count   = 0;
    p->dt_signal_count    = 0;
    return 1;
}
