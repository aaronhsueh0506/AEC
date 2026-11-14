/* detectors.c -- RenderActivity, FilterConvergence and DoubleTalk detectors (float32). */
#include "detectors.h"
#include "aec_scale.h"
#include <math.h>

/* Wall-clock constants are authored for hop=160 @ 16 kHz and retimed in the *_init functions. */
#define DET_REF_HOP 160
#define DET_REF_SR  16000
#define DET_REHOP(v, hop, sr) \
    aec_growth_rehop((v), DET_REF_HOP, DET_REF_SR, (hop), (sr))

/* ── RenderActivityDetector ──────────────────────────────────── */

static const float RA_ALPHA_CV_REF = 0.99f;   /* TC ~ 1 s at the reference hop */
/* Dimensionless CV^2 gate; not retimed. */
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
    /* Scalar state only; the caller-owned scratch pointers set at init are kept. */
    r->env_mean      = 1e-10f;
    r->env_var       = 0.0f;
    r->active_prev   = 0;
    r->is_stationary = 0;
}

RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n) {
    /* Guard: n <= 0 returns the current state with the power floored. */
    if (n <= 0) {
        RenderActivityResult res0 = {
            .far_pwr       = 1e-10f,
            .is_active     = r->active_prev,
            .warmup_active = 0,
        };
        return res0;
    }
    /* Mean power via 8-element block sums and a tree reduction over the block sums. */
    enum { PAIR_BLOCK = 8 };
    int n_blocks = (n + PAIR_BLOCK - 1) / PAIR_BLOCK;
    /* Guard: scratch too small for n returns the current state. */
    if (n_blocks > r->pairwise_scratch_len) {
        RenderActivityResult res_guard = {
            .far_pwr       = 1e-10f,
            .is_active     = r->active_prev,
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
    /* Tree reduction over the block sums. */
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
            /* One observation has no variance estimate; wait for the next active hop. */
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
    c->div_alpha_c = 1.0f - c->div_alpha;
    c->div_decay = DET_REHOP(FC_DIV_DECAY_REF, hop_size, sample_rate);
    filter_convergence_reset(c);
}
void filter_convergence_reset(FilterConvergence* c) {
    /* State only; the grid-derived div_alpha/div_decay are kept. */
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
                      + c->div_alpha_c * is_div;
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

/* Shadow-filter DT blind period, kept as a duration so it is the same on every grid. */
static const float DT_SHADOW_FRAME_GATE_MS = 200.0f;
static const float DT_ERL_CEILING_FLOOR = 0.01f;
static const float DT_SAFETY_MARGIN     = 2.0f;
static const float DTE_RISE_OLD_REF  = 0.3f;
static const float DTE_DECAY_OLD_REF = 0.9f;  /* ~90 ms hangover */
static const float DTS_OLD_REF = 0.7f;
static const float DTS_INACTIVE_DECAY_REF = 0.95f;

void doubletalk_init(DoubleTalk* d, float offset, float advantage_scale,
                        int hop_size, int sample_rate) {
    d->dt_from_energy   = 0.0f;
    d->dt_from_shadow   = 0.0f;
    d->shadow_advantage = 1.0f;
    d->shadow_dtd_offset           = offset;
    d->shadow_dtd_advantage_scale  = advantage_scale;
    d->shadow_frame_gate = aec_ms_to_hops(DT_SHADOW_FRAME_GATE_MS,
                                          hop_size, sample_rate);
    /* Only the retention term is retimed; the new-sample weight is 1 - retention. */
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
