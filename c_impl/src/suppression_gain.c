/* suppression_gain.c -- see suppression_gain.h. Float32 throughout, pairwise float32 sums for
 * reductions. */
#include "suppression_gain.h"

#include <assert.h>
#include <string.h>
#include "utility/fast_math.h"
#include "utility/simd_aec.h"
#include "aec_scale.h"


/* Float32 sum of a[lo:hi) via the pairwise sum. */
static float f32_sum_slice(const float *a, int lo, int hi) {
    if (hi <= lo) return 0.0f;
    return sk_pairwise_sum_tailfold_b_f32(a + lo, (size_t)(hi - lo));
}

/* helpers */

void suppression_gain_init(SuppressionGain *sg,
                           const SuppressionGainConfig *cfg,
                           const SuppressionGainTuning *tun,
                           float *last_gain, float *last_nearend,
                           float *last_echo, float *ma_buf,
                           float *nearend_scratch, float *weighted_residual,
                           float *min_gain, float *max_gain, float *g_raw,
                           float *gain_out, float *sum_scratch) {
    int k;
    memset(sg, 0, sizeof(*sg));
    sg->cfg = *cfg;
    sg->tun = *tun;
    /* table_len must equal cfg->n_bins: the per-bin loops read the six tuning tables at
     * [0, n_bins). Construction-time invariant, so a debug assert is the guard. */
    assert(tun->table_len == cfg->n_bins);
    /* The live floor starts equal to its target. */
    sg->split_floor_far_active_live = cfg->split_floor_far_active;
    sg->split_floor_ramp_ratio = 1.0f;
    sg->last_gain = last_gain;
    sg->last_nearend = last_nearend;
    sg->last_echo = last_echo;
    sg->ma_buf = ma_buf;
    sg->nearend = nearend_scratch;
    sg->weighted_residual = weighted_residual;
    sg->min_gain = min_gain;
    sg->max_gain = max_gain;
    sg->g_raw = g_raw;
    sg->gain = gain_out;
    sg->sum_scratch = sum_scratch;

    /* The gain arrays may all be NULL when only the DominantNearend state is used (suppression_gain_get_gain() is then never
     * called). */
    if (sg->last_gain && sg->last_nearend && sg->last_echo) {
        for (k = 0; k < cfg->n_bins; ++k) {
            sg->last_gain[k] = 1.0f;
            sg->last_nearend[k] = 0.0f;
            sg->last_echo[k] = 0.0f;
        }
    }
    sg->low_render_avg_power = 32768.0f * 32768.0f;
    /* The power IIR is the literal 0.9 / 0.1 at every grid. */
    sg->low_render_iir_decay = 0.9f;
    sg->low_render_iir_weight = 0.1f;
    sg->far_active_latched = 0;
    sg->dt_protect_active = 0;
    sg->initial_state = 1;
    sg->stat_mask_frac = 0.0f;
    sg->dne_trigger_counter = 0;
    sg->dne_hold_counter = 0;
    sg->dne_nearend_state = 0;
    sg->ma_count = 0;
    sg->ma_head = 0;
}

void suppression_gain_set_initial_state(SuppressionGain *sg, int state) {
    sg->initial_state = state ? 1 : 0;
}

/* Bounds for the runtime retarget; the dB range matches the config validator. */
#define SG_SPLIT_FLOOR_DB_MIN   (-300.0f)
#define SG_SPLIT_FLOOR_DB_MAX     (50.0f)
#define SG_SPLIT_FLOOR_RAMP_MS_MAX (60000.0f)

/* Range-checked dB -> power conversion shared by the split-floor setters. Returns 0 and writes
 * *power_out, or -1 (nothing written) for a non-finite or out-of-range db. */
static int sg_split_floor_db_to_power(float db, float *power_out) {
    if (!isfinite(db) || db < SG_SPLIT_FLOOR_DB_MIN || db > SG_SPLIT_FLOOR_DB_MAX) {
        return -1;
    }
    *power_out = powf(10.0f, db / 10.0f);
    return 0;
}

/* One step of the power-geometric (dB-linear) walk toward the target; idle when live == target. */
static void advance_split_floor_ramp(SuppressionGain *sg) {
    float live = sg->split_floor_far_active_live;
    float target = sg->cfg.split_floor_far_active;
    float value;
    if (live == target) return;
    value = live * sg->split_floor_ramp_ratio;
    /* Land exactly on the target rather than approaching it asymptotically. */
    if (sg->split_floor_ramp_ratio == 1.0f ||
        (sg->split_floor_ramp_ratio > 1.0f ? (value >= target)
                                           : (value <= target))) {
        value = target;
    }
    sg->split_floor_far_active_live = value;
}

int suppression_gain_set_split_floor_far_active_db(SuppressionGain *sg,
                                                   float db, float ramp_ms) {
    float target, live, ratio;
    int hops;

    if (!sg) return -1;
    if (sg_split_floor_db_to_power(db, &target) != 0) return -1;
    if (!isfinite(ramp_ms) || ramp_ms < 0.0f ||
        ramp_ms > SG_SPLIT_FLOOR_RAMP_MS_MAX) {
        return -1;
    }

    live = sg->split_floor_far_active_live;

    hops = 0;
    if (ramp_ms > 0.0f && target > 0.0f && live > 0.0f && live != target) {
        /* The ramp_ms bound above keeps the float->int conversion in aec_ms_to_hops() in range
         * (worst case 60 s * 48000/128 = 22500 hops). */
        hops = aec_ms_to_hops(ramp_ms, sg->cfg.hop_size, sg->cfg.sr);
        if (hops < 1) hops = 1;
    }

    /* Everything is validated; write now. */
    sg->cfg.split_floor_far_active = target;
    if (hops == 0) {
        sg->split_floor_far_active_live = target;
        sg->split_floor_ramp_ratio = 1.0f;
        return 0;
    }
    ratio = powf(target / live, 1.0f / (float)hops);
    if (ratio == 1.0f) {
        /* Ratio is exactly 1 (step too small): land now rather than stall. */
        sg->split_floor_far_active_live = target;
        sg->split_floor_ramp_ratio = 1.0f;
        return 0;
    }
    sg->split_floor_ramp_ratio = ratio;
    return 0;
}

/* The DT-lifted floor is a hard per-hop override with no ramp state, so it steps immediately. */
int suppression_gain_set_split_floor_dt_db(SuppressionGain *sg, float db) {
    float target;
    if (!sg) return -1;
    if (sg_split_floor_db_to_power(db, &target) != 0) return -1;
    sg->cfg.split_floor_dt = target;
    return 0;
}

/* Plain coerce-and-store; a boolean gate steps rather than ramps. */
int suppression_gain_set_split_floor_enabled(SuppressionGain *sg, int enabled) {
    if (!sg) return -1;
    sg->cfg.split_floor_enabled = enabled ? 1 : 0;
    return 0;
}

int suppression_gain_is_dominant_nearend(const SuppressionGain *sg) {
    return sg->dne_nearend_state ? 1 : 0;
}

/* Raw DominantNearend state unless the stat-aware proxy is enabled (off in the shipped levels). */
static int ne_state_for_gain_rules(const SuppressionGain *sg) {
    if (!sg->cfg.stat_aware_ne_proxy_enabled) return sg->dne_nearend_state;
    if (sg->dne_nearend_state) return 1;
    return sg->stat_mask_frac > sg->cfg.stat_aware_ne_proxy_threshold;
}

/* Largest hop the render statistics support (scratch size of the squared-sample array). */
#define SG_RENDER_STATS_MAX_HOP 1024

void suppression_gain_render_stats(const float *render_block, int n,
                                   float *sum_sq, float *peak_sq) {
    float x2[SG_RENDER_STATS_MAX_HOP];
    int i;
    *sum_sq = 0.0f;
    *peak_sq = 0.0f;
    if (!render_block || n <= 0 || n > SG_RENDER_STATS_MAX_HOP) return;
    /* Squared samples (scale 1). */
    sk_sq_scale_f32(render_block, 1.0f, x2, n);
    *sum_sq = sk_pairwise_sum_tailfold_b_f32(x2, (size_t)n);
    *peak_sq = x2[0];
    for (i = 1; i < n; ++i) if (x2[i] > *peak_sq) *peak_sq = x2[i];
}

/* Low-noise render detector: returns the flag and updates the average power. A hop outside the
 * supported range leaves the average untouched and reports no low-noise render. */
static int low_noise_render_detect(SuppressionGain *sg, int hop,
                                   float x2_sum, float x2_max) {
    const SuppressionGainConfig *c = &sg->cfg;
    int low_noise;
    if (hop <= 0 || hop > SG_RENDER_STATS_MAX_HOP) return 0;
    /* Normalise the average power to per-sample (reference block is 64 samples) before the peak
     * comparison. */
    {
        float avg_per_sample = sg->low_render_avg_power * (64.0f / (float)hop);
        low_noise = (sg->low_render_avg_power < c->low_render_threshold)
                    && (x2_max < 3.0f * avg_per_sample);
    }
    sg->low_render_avg_power = sg->low_render_avg_power * sg->low_render_iir_decay
                              + x2_sum * sg->low_render_iir_weight;
    return low_noise ? 1 : 0;
}

/* DominantNearend update; sums are float32 pairwise. */
static void dominant_nearend_update(SuppressionGain *sg, const float *nearend,
                                    const float *echo, const float *cn,
                                    int initial_state) {
    const SuppressionGainConfig *c = &sg->cfg;
    int lf = c->dne_lf_endpoint_bin;
    float ne_sum = f32_sum_slice(nearend, 1, lf);
    float echo_sum = f32_sum_slice(echo, 1, lf);
    float noise_sum = f32_sum_slice(cn, 1, lf);
    int trigger_initial_gate = (!initial_state) || c->dne_use_during_initial_phase;
    int trigger_enr_pass = echo_sum < c->dne_enr_threshold * ne_sum;
    int trigger_snr_pass = ne_sum > c->dne_snr_threshold * noise_sum;
    int trigger_active = trigger_initial_gate && trigger_enr_pass
                         && trigger_snr_pass;
    int early_exit;

    if (trigger_active) {
        sg->dne_trigger_counter += 1;
        if (sg->dne_trigger_counter >= c->dne_trigger_threshold_hops) {
            sg->dne_hold_counter = c->dne_hold_duration_hops;
            sg->dne_trigger_counter = c->dne_trigger_threshold_hops;
        }
    } else {
        sg->dne_trigger_counter = sg->dne_trigger_counter - 1;
        if (sg->dne_trigger_counter < 0) sg->dne_trigger_counter = 0;
    }

    early_exit = (echo_sum > c->dne_enr_exit_threshold * ne_sum)
                 && (echo_sum > c->dne_snr_threshold * noise_sum);
    if (early_exit) sg->dne_hold_counter = 0;

    sg->dne_hold_counter -= 1;
    if (sg->dne_hold_counter < 0) sg->dne_hold_counter = 0;
    sg->dne_nearend_state = sg->dne_hold_counter > 0;
}

/* Append the spectrum to the ring and write the per-bin mean over the most recent rows to out. */
static void nearend_smoother_average(SuppressionGain *sg, const float *spec,
                                     float *out) {
    int n = sg->cfg.nearend_smoother_n;
    int nb = sg->cfg.n_bins;
    int idx, k;
    /* Append into the slot after the newest, evicting the oldest. */
    if (sg->ma_count < n) {
        idx = (sg->ma_head + sg->ma_count) % n;
        sg->ma_count += 1;
    } else {
        idx = sg->ma_head;
        sg->ma_head = (sg->ma_head + 1) % n;
    }
    memcpy(sg->ma_buf + (size_t)idx * nb, spec, (size_t)nb * sizeof(float));

    /* n == 1: the mean is the single row, so copy instead of dividing. */
    if (n == 1) {
        memcpy(out, sg->ma_buf, (size_t)nb * sizeof(float));
        return;
    }

    /* Per-bin mean over ma_count rows, summed sequentially in float32. */
    for (k = 0; k < nb; ++k) {
        int r;
        float acc;
        int first = sg->ma_head;
        acc = sg->ma_buf[(size_t)first * nb + k];
        for (r = 1; r < sg->ma_count; ++r) {
            int ri = (sg->ma_head + r) % n;
            acc = acc + sg->ma_buf[(size_t)ri * nb + k];
        }
        out[k] = acc / (float)sg->ma_count;
    }
}

/* Echo weighted for audibility, one band. */
static void weigh_band(const SuppressionGainConfig *c, const float *echo,
                       float *out, float threshold, int begin, int end) {
    int k;
    float normalizer = 1.0f / (threshold - c->floor_power);
    if (begin >= end) return;
    for (k = begin; k < end; ++k) {
        float seg = echo[k];
        if (seg < threshold) {
            float tmp = (threshold - seg) * normalizer;
            float w = 1.0f - tmp * tmp;
            if (w < 0.0f) w = 0.0f;
            out[k] = seg * w;
        } else {
            out[k] = seg;
        }
    }
}

static void weight_echo_for_audibility(const SuppressionGainConfig *c,
                                       const float *echo, float *out) {
    int n = c->n_bins;
    int lf_end = c->aud_lf_end_bin < n ? c->aud_lf_end_bin : n;
    int mf_end = c->aud_mf_end_bin < n ? c->aud_mf_end_bin : n;
    weigh_band(c, echo, out, c->floor_power * c->aud_thr_lf, 0, lf_end);
    weigh_band(c, echo, out, c->floor_power * c->aud_thr_mf, lf_end, mf_end);
    weigh_band(c, echo, out, c->floor_power * c->aud_thr_hf, mf_end, n);
}

/* Maximum gain: last_gain * increase, clipped to [floor_first_increase, 1]. */
static void get_max_gain(SuppressionGain *sg, float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int is_ne = ne_state_for_gain_rules(sg);
    float inc = is_ne ? c->max_inc_nearend : c->max_inc_normal;
    float ffi = c->floor_first_increase;
    sk_scale_f32(out, sg->last_gain, inc, c->n_bins);
    sk_clip_f32(out, ffi, 1.0f, c->n_bins);
}

/* Minimum gain. */
static void get_min_gain(SuppressionGain *sg, const float *weighted_residual,
                         int low_noise_render, int saturated_echo, float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int k;
    float min_echo_power;
    if (saturated_echo) {
        for (k = 0; k < n; ++k) out[k] = 0.0f;
        return;
    }
    min_echo_power = low_noise_render ? c->low_render_limit
                                      : c->normal_render_limit;
    {
        for (k = 0; k < n; ++k) {
            float wr = weighted_residual[k];
            if (wr > 0.0f) {
                float denom = wr < 1e-30f ? 1e-30f : wr;
                out[k] = min_echo_power / denom;
            } else {
                out[k] = 1.0f;
            }
            if (out[k] > 1.0f) out[k] = 1.0f;
        }
    }
    /* LF smoothing block */
    if (!sg->initial_state || c->lf_smoothing_during_initial_phase) {
        int is_ne = ne_state_for_gain_rules(sg);
        float dec = is_ne ? c->max_dec_lf_nearend : c->max_dec_lf_normal;
        int end = c->last_lf_smoothing_band + 1;
        int permanent = c->last_permanent_lf_smoothing_band;
        if (end > n) end = n;
        for (k = 0; k < end; ++k) {
            if (sg->last_nearend[k] > sg->last_echo[k] || k <= permanent) {
                /* last_gain[k] * dec */
                float prod = sg->last_gain[k] * dec;
                float mg = out[k];
                float chosen = mg;
                if (prod > mg) chosen = prod;
                if (1.0f < chosen) chosen = 1.0f;
                out[k] = chosen;
            }
        }
    }
    /* Split min-gain floor. */
    if (c->split_floor_enabled) {
        float base_floor;
        if (sg->far_active_latched) {
            /* In double-talk the DT floor lifts the far-active floor. fmaxf() keeps the DT floor
             * from dropping below the far-active floor while a ramp is in flight (the DT setter
             * steps, the far-active setter ramps); at steady state it is a no-op. */
            base_floor = sg->dt_protect_active
                             ? fmaxf(c->split_floor_dt, sg->split_floor_far_active_live)
                             : sg->split_floor_far_active_live;
        } else {
            base_floor = c->split_floor_far_silent;
        }
        /* Clip to [base_floor, 1]. */
        sk_clip_f32(out, base_floor, 1.0f, n);
    }
}

/* Gain to no audible echo. */
static void gain_to_no_audible_echo(SuppressionGain *sg, const float *nearend,
                                    const float *echo, const float *masker,
                                    float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int is_ne;
    int k;
    float ne_w = 0.0f;
    const float *enr_tr_tab, *enr_su_tab, *emr_tr_tab;
    int have_scalar_blend_inputs;

    /* Per-bin soft blend (the shipped configuration) is handled by one SIMD kernel and checked
     * first. */
    if (c->soft_blend_enabled && c->soft_blend_per_bin) {
        sk_no_audible_echo_gain_f32(nearend, echo, masker,
                                    sg->tun.nearend_enr_tr,
                                    sg->tun.nearend_enr_su,
                                    sg->tun.nearend_emr_tr,
                                    sg->tun.normal_enr_tr,
                                    sg->tun.normal_enr_su,
                                    sg->tun.normal_emr_tr,
                                    c->soft_blend_enr_thr,
                                    c->soft_blend_softness, out, n);
        return;
    }

    is_ne = ne_state_for_gain_rules(sg);
    have_scalar_blend_inputs = c->soft_blend_enabled;

    if (have_scalar_blend_inputs) {
        /* ne_lf / echo_lf: pairwise sums over [1, dne_lf_end) */
        float ne_lf = f32_sum_slice(nearend, 1, c->dne_lf_end);
        float echo_lf = f32_sum_slice(echo, 1, c->dne_lf_end);
        float enr_lf = echo_lf / (ne_lf + 1.0f);
        float sig_arg = (enr_lf - c->soft_blend_enr_thr) / c->soft_blend_softness;
        if (sig_arg < -50.0f) sig_arg = -50.0f;
        if (sig_arg > 50.0f) sig_arg = 50.0f;
        ne_w = 1.0f / (1.0f + fast_exp(sig_arg));
    }

    /* Resolve the tuning tables; with soft blend the blended values are computed per bin below. */
    if (c->soft_blend_enabled) {
        enr_tr_tab = NULL;
        enr_su_tab = NULL;
        emr_tr_tab = NULL;
    } else {
        enr_tr_tab = is_ne ? sg->tun.nearend_enr_tr : sg->tun.normal_enr_tr;
        enr_su_tab = is_ne ? sg->tun.nearend_enr_su : sg->tun.normal_enr_su;
        emr_tr_tab = is_ne ? sg->tun.nearend_emr_tr : sg->tun.normal_emr_tr;
    }

    for (k = 0; k < n; ++k) {
        float enr_tr_k, enr_su_k, emr_tr_k;
        float enr, emr, g;

        if (c->soft_blend_enabled) {
            float ne_wb;
            if (c->soft_blend_per_bin) {
                /* Per-bin sigmoid weight from echo/(nearend + 1). */
                float enr_bin = echo[k] / (nearend[k] + 1.0f);
                float sig = (enr_bin - c->soft_blend_enr_thr)
                            / c->soft_blend_softness;
                if (sig < -50.0f) sig = -50.0f;
                if (sig > 50.0f) sig = 50.0f;
                ne_wb = 1.0f / (1.0f + fast_exp(sig));
            } else {
                ne_wb = ne_w;
            }
            /* Blend nearend and normal tables. */
            enr_tr_k = (ne_wb * sg->tun.nearend_enr_tr[k]
                        + (1.0f - ne_wb) * sg->tun.normal_enr_tr[k]);
            enr_su_k = (ne_wb * sg->tun.nearend_enr_su[k]
                        + (1.0f - ne_wb) * sg->tun.normal_enr_su[k]);
            emr_tr_k = (ne_wb * sg->tun.nearend_emr_tr[k]
                        + (1.0f - ne_wb) * sg->tun.normal_emr_tr[k]);
        } else {
            enr_tr_k = enr_tr_tab[k];
            enr_su_k = enr_su_tab[k];
            emr_tr_k = emr_tr_tab[k];
        }

        enr = echo[k] / (nearend[k] + 1.0f);
        emr = echo[k] / (masker[k] + 1.0f);
        g = 1.0f;
        if (enr > enr_tr_k && emr > emr_tr_k) {
            float d_lin = enr_su_k - enr_tr_k;
            float denom_lin = d_lin < 1e-30f ? 1e-30f : d_lin;
            float g_lin = (enr_su_k - enr) / denom_lin;
            float denom_emr = emr < 1e-30f ? 1e-30f : emr;
            float g_emr = emr_tr_k / denom_emr;
            g = g_lin > g_emr ? g_lin : g_emr;
        }
        out[k] = g;
    }
}

/* Limit LF gains to the value at lf_clamp_bin (hz_to_bin(250 Hz)). */
static void limit_lf_gains(float *gain, int n, int lf_clamp_bin) {
    if (lf_clamp_bin > 0 && lf_clamp_bin < n) {
        float anchor = gain[lf_clamp_bin];
        sk_clip_f32(gain, -(float)INFINITY, anchor, lf_clamp_bin);
    }
}

/* Limit HF gains. */
static void limit_hf_gains(SuppressionGain *sg, float *gain) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int lgb = c->hf_lgb;
    int biq = c->hf_biq;
    int k;
    if (biq > 0 && lgb + biq <= n) {
        /* Minimum over gain[lgb : lgb + biq], then clip the rest to it. */
        float mug = gain[lgb];
        for (k = lgb + 1; k < lgb + biq; ++k) if (gain[k] < mug) mug = gain[k];
        sk_clip_f32(gain + lgb + 1, -(float)INFINITY, mug, n - (lgb + 1));
    }
    if (n >= 2) gain[n - 1] = gain[n - 2];
    (void)c;
}

/* get_gain */

void suppression_gain_update_dominant_nearend(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise) {
    const SuppressionGainConfig *c = &sg->cfg;
    const float *echo_for_det;

    /* No stationary mask: stat_mask_frac = 0. */
    sg->stat_mask_frac = 0.0f;

    echo_for_det = c->dne_use_unbounded_echo ? residual_echo_unbounded
                                             : residual_echo;
    dominant_nearend_update(sg, nearend_spectrum, echo_for_det, comfort_noise,
                            sg->initial_state);
}

const float *suppression_gain_get_gain(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise,
    const float *render_block,
    int clock_drift,
    int saturated_echo) {
    float sum_sq, peak_sq;
    suppression_gain_render_stats(render_block, sg->cfg.hop_size, &sum_sq, &peak_sq);
    return suppression_gain_get_gain_stats(sg, nearend_spectrum, residual_echo,
                                           residual_echo_unbounded, comfort_noise,
                                           sum_sq, peak_sq, clock_drift, saturated_echo);
}

const float *suppression_gain_get_gain_stats(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise,
    float render_sum_sq,
    float render_peak_sq,
    int clock_drift,
    int saturated_echo) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int low_noise_render;
    int hf_lim_applied;

    assert(sg->gain);       /* NULL arrays: an instance that only tracks DominantNearend */

    /* A hop outside the supported range has no statistics (what get_gain() reads from it). */
    if (c->hop_size <= 0 || c->hop_size > SG_RENDER_STATS_MAX_HOP) render_sum_sq = 0.0f;

    advance_split_floor_ramp(sg);

    suppression_gain_update_dominant_nearend(
        sg, nearend_spectrum, residual_echo, residual_echo_unbounded,
        comfort_noise);

    low_noise_render = low_noise_render_detect(sg, c->hop_size, render_sum_sq,
                                                render_peak_sq);

    /* Split-floor far-active latch. */
    if (c->split_floor_enabled && !sg->far_active_latched) {
        float mean_pow = render_sum_sq / (float)c->hop_size;
        if (mean_pow > c->split_floor_latch_power) sg->far_active_latched = 1;
    }

    /* Lower-band gain */
    /* Step 1: max gain */
    get_max_gain(sg, sg->max_gain);
    /* Step 2: smoothed nearend */
    nearend_smoother_average(sg, nearend_spectrum, sg->nearend);
    /* Step 3: weighted residual */
    weight_echo_for_audibility(c, residual_echo, sg->weighted_residual);
    /* Step 4: min gain (uses last_nearend / last_echo) */
    get_min_gain(sg, sg->weighted_residual, low_noise_render, saturated_echo,
                 sg->min_gain);
    /* Step 5: gain that leaves no audible echo */
    gain_to_no_audible_echo(sg, sg->nearend, sg->weighted_residual,
                            comfort_noise, sg->g_raw);
    /* Step 6: minimum(G_raw, max_gain), then maximum with min_gain; min_gain wins when it exceeds
     * max_gain (saturated-echo recovery). */
    sk_min_f32(sg->gain, sg->max_gain, sg->g_raw, n);
    sk_raise_to_f32(sg->gain, sg->min_gain, n);

    /* Step 7: LF + HF limiters */
    limit_lf_gains(sg->gain, n, sg->cfg.lf_clamp_bin);
    hf_lim_applied = (!ne_state_for_gain_rules(sg)) || clock_drift
                     || c->conservative_hf;
    if (hf_lim_applied) limit_hf_gains(sg, sg->gain);

    /* Stash for the next hop. last_gain is copied because sg->gain must keep naming the same
     * buffer; nearend and weighted_residual are rewritten in full each hop, so their stash is a
     * pointer swap. */
    memcpy(sg->last_gain, sg->gain, (size_t)n * sizeof(float));
    {
        float *t = sg->last_nearend;
        sg->last_nearend = sg->nearend;
        sg->nearend = t;
    }
    {
        float *t = sg->last_echo;
        sg->last_echo = sg->weighted_residual;
        sg->weighted_residual = t;
    }

    /* Step 8: sqrt to amplitude domain. The explicit clamp at 0 keeps negatives from becoming NaN
     * when fast_sqrt is plain sqrtf. */
    sk_clip_f32(sg->gain, 0.0f, (float)INFINITY, n);
    sk_fast_sqrt_f32(sg->gain, sg->gain, n);
    return sg->gain;
}
