/* detectors.c — RenderActivity / FilterConvergence / DoubleTalk ports.
 * All scalar math in fp64 to match Python.
 */
#include "detectors.h"
#include <math.h>

/* ── RenderActivityDetector ──────────────────────────────────── */

static const double RA_ALPHA_CV       = 0.99;
static const double RA_STATIONARY_C = 0.02;

void render_activity_init(RenderActivity* r) {
    r->env_mean      = 1e-10;
    r->env_var       = 0.0;
    r->active_prev   = 0;
    r->is_stationary = 0;
}
void render_activity_reset(RenderActivity* r) {
    render_activity_init(r);
}

RenderActivityResult render_activity_update(RenderActivity* r,
                                               const float* far_end, int n) {
    /* far_pwr_raw = float(np.mean(far_end ** 2)).
     *
     * np.mean of an fp32 array uses pairwise summation in fp32 then divides
     * (also fp32), then float() promotes the fp32 scalar to fp64. To
     * reproduce numpy's pairwise summation we use 8-element blocks summed
     * linearly, then a tree reduction over the block sums — this mirrors
     * numpy's `pairwise_sum_FLOAT` (`numpy/core/src/umath/loops_utils.h.src`).
     */
    enum { PAIR_BLOCK = 8 };
    int n_blocks = (n + PAIR_BLOCK - 1) / PAIR_BLOCK;
    /* Stack buffer: max hop in our pipeline ≤ 8192 → ≤ 1024 blocks. */
    float blocks[1024];
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
    float mean_f32 = blocks[0] / (float)n;
    double far_pwr_raw = (double)mean_f32;
    double far_pwr     = far_pwr_raw + 1e-10;
    int    warmup_active = far_pwr_raw > 1e-6 ? 1 : 0;

    if (far_pwr > 1e-6) {
        if (!r->active_prev) {
            r->env_mean    = far_pwr;
            r->env_var     = 0.0;
            r->active_prev = 1;
        } else {
            double old_mean = r->env_mean;
            r->env_mean = RA_ALPHA_CV * r->env_mean
                        + (1.0 - RA_ALPHA_CV) * far_pwr;
            double diff = far_pwr - old_mean;
            r->env_var = RA_ALPHA_CV * r->env_var
                       + (1.0 - RA_ALPHA_CV) * (diff * diff);
        }
        double far_cv2 = r->env_var / (r->env_mean * r->env_mean + 1e-10);
        r->is_stationary = (far_cv2 < RA_STATIONARY_C) ? 1 : 0;
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

static const double FC_CONV_ERLE_DB    = 5.0;
static const int    FC_CONV_FRAMES     = 10;
static const double FC_CONV_ERLE_ALPHA = 0.7;  /* TC≈3 frames; immune to 1-2 frame mic-silence */
static const double FC_DIV_ERLE_LIN = 0.63;
static const double FC_DIV_ALPHA    = 0.9;
static const double FC_DIV_DECAY    = 0.95;

void filter_convergence_init(FilterConvergence* c) {
    c->converged       = 0;
    c->once_converged  = 0;
    c->conv_counter    = 0;
    c->divergence      = 0.0;
    c->erle_smooth_lin = 1.0;
}
void filter_convergence_reset(FilterConvergence* c) {
    filter_convergence_init(c);
}
void filter_convergence_mark_diverged(FilterConvergence* c) {
    c->converged       = 0;
    c->conv_counter    = 0;
    c->erle_smooth_lin = 1.0;
}
void filter_convergence_update_divergence(FilterConvergence* c,
                                             double near_power,
                                             double raw_error_power) {
    if (c->converged && near_power > 1e-8) {
        double inst_erle_lin = near_power / (raw_error_power + 1e-10);
        double is_div        = (inst_erle_lin < FC_DIV_ERLE_LIN) ? 1.0 : 0.0;
        c->divergence = FC_DIV_ALPHA * c->divergence
                      + (1.0 - FC_DIV_ALPHA) * is_div;
    } else {
        c->divergence *= FC_DIV_DECAY;
    }
}
int filter_convergence_update_convergence(FilterConvergence* c,
                                             double near_power,
                                             double raw_error_power,
                                             int far_active, int warmup_done) {
    if (c->converged || near_power <= 1e-8 || !warmup_done || !far_active) {
        return 0;
    }
    double inst_erle_lin = near_power / (raw_error_power + 1e-10);
    c->erle_smooth_lin   = FC_CONV_ERLE_ALPHA * c->erle_smooth_lin
                         + (1.0 - FC_CONV_ERLE_ALPHA) * inst_erle_lin;
    double erle_smooth_db = 10.0 * log10(c->erle_smooth_lin + 1e-10);
    if (erle_smooth_db > FC_CONV_ERLE_DB) c->conv_counter += 1;
    else                                   c->conv_counter  = 0;
    if (c->conv_counter >= FC_CONV_FRAMES) {
        c->converged       = 1;
        c->once_converged  = 1;
        return 1;
    }
    return 0;
}

/* ── DoubleTalkAnalyzer ──────────────────────────────────────── */

static const int    DT_SHADOW_FRAME_GATE = 50;
static const double DT_ERL_CEILING_FLOOR = 0.01;
static const double DT_SAFETY_MARGIN     = 2.0;
static const double DTE_RISE_OLD = 0.3, DTE_RISE_NEW = 0.7;
static const double DTE_DECAY_OLD = 0.9, DTE_DECAY_NEW = 0.1;
static const double DTS_OLD = 0.7, DTS_NEW = 0.3;
static const double DTS_INACTIVE_DECAY = 0.95;

void doubletalk_init(DoubleTalk* d, double offset, double advantage_scale) {
    d->dt_from_energy   = 0.0;
    d->dt_from_shadow   = 0.0;
    d->shadow_advantage = 1.0;
    d->shadow_dtd_offset           = offset;
    d->shadow_dtd_advantage_scale  = advantage_scale;
}
void doubletalk_reset(DoubleTalk* d) {
    d->dt_from_energy   = 0.0;
    d->dt_from_shadow   = 0.0;
    d->shadow_advantage = 1.0;
}

void doubletalk_update_shadow_dt(DoubleTalk* d,
                                    int shadow_frame_count, int far_excited,
                                    double main_err_smooth,
                                    double shadow_err_smooth) {
    if (shadow_frame_count >= DT_SHADOW_FRAME_GATE && far_excited) {
        d->shadow_advantage = main_err_smooth / (shadow_err_smooth + 1e-10);
        double raw = (d->shadow_advantage - d->shadow_dtd_offset)
                   / d->shadow_dtd_advantage_scale;
        if (raw < 0.0) raw = 0.0;
        if (raw > 1.0) raw = 1.0;
        d->dt_from_shadow = DTS_OLD * d->dt_from_shadow + DTS_NEW * raw;
    } else {
        d->dt_from_shadow *= DTS_INACTIVE_DECAY;
    }
}

void doubletalk_update_energy_dt(DoubleTalk* d,
                                    int far_active, double far_pwr,
                                    double mic_pwr, double erl_estimate) {
    double inst;
    if (far_active && far_pwr > 1e-4) {
        double erl = (erl_estimate > DT_ERL_CEILING_FLOOR)
                     ? erl_estimate : DT_ERL_CEILING_FLOOR;
        double erl_ceiling      = 1.0 / erl;
        double max_echo_expected = far_pwr * erl_ceiling * DT_SAFETY_MARGIN;
        inst = (mic_pwr - max_echo_expected) / mic_pwr;
        if (inst < 0.0) inst = 0.0;
    } else {
        inst = 0.0;
    }
    if (inst > d->dt_from_energy) {
        d->dt_from_energy = DTE_RISE_OLD * d->dt_from_energy
                          + DTE_RISE_NEW * inst;
    } else {
        d->dt_from_energy = DTE_DECAY_OLD * d->dt_from_energy
                          + DTE_DECAY_NEW * inst;
    }
}

/* ── FilterPlateauDetector (v3.10.0 + v3.10.3 F4/H3) ─────────── */

void filter_plateau_init(FilterPlateauDetector* p) {
    p->grace_frames     = 400;
    p->erle_max_db      = 6.0;
    p->far_active_ratio = 0.5;
    p->dt_signal_ratio  = 0.10;
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
                             double erle_windowed_db,
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

    int denom = p->frame_count > 1 ? p->frame_count : 1;
    double far_ratio = (double)p->far_active_count / (double)denom;
    double dt_ratio  = (double)p->dt_signal_count  / (double)denom;

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
