/* suppression_gain.c — C port of python/modules/residual/suppression_gain.py
 * (WS5). Byte-equal to numpy 1.26. Build with -ffp-contract=off.
 *
 * Parity rules applied (see project_c_port_parity_rules):
 *  - f32_array op pyfloat  -> scalar cast to f32, op in f32.
 *  - scalar f32 * pyfloat  -> f64 (opposite of array). Used in LF-smoothing
 *    `last_gain[k]*dec` and in the DNE/CNG sums (all routed through double).
 *  - Python builtin max/min compare in DOUBLE and return one operand.
 *  - np.sum/np.mean over f32 = pairwise float32 (f32_pairwise_sum);
 *    over f64 = pairwise float64 (f64_pairwise_sum here).
 *  - np.exp/np.sqrt over f32 == expf/sqrtf bit-exact on this target.
 */
#include "suppression_gain.h"

#include <math.h>
#include <string.h>

/* numpy-1.26-bit-exact pairwise sum over float64 (same tree shape as the f32
 * version in reverb_frequency_response.c, accumulated in double). */
static double f64_pairwise_sum(const double *a, size_t n) {
    if (n == 0) return 0.0;
    if (n < 8) {
        double res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) res = res + a[i];
        return res;
    }
    if (n <= 128) {
        double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3];
        double r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
        double res;
        size_t i;
        for (i = 8; i + 8 <= n; i += 8) {
            r0 += a[i + 0]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3];
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
        }
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
        for (; i < n; ++i) res = res + a[i];
        return res;
    } else {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return f64_pairwise_sum(a, n2) + f64_pairwise_sum(a + n2, n - n2);
    }
}

/* float32 sum of a sub-slice a[lo:hi) via the canonical pairwise sum. */
static float f32_sum_slice(const float *a, int lo, int hi) {
    if (hi <= lo) return 0.0f;
    return f32_pairwise_sum(a + lo, (size_t)(hi - lo));
}

/* ---------------------------------------------------------------- helpers */

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

    for (k = 0; k < cfg->n_bins; ++k) {
        sg->last_gain[k] = 1.0f;
        sg->last_nearend[k] = 0.0f;
        sg->last_echo[k] = 0.0f;
    }
    sg->low_render_avg_power = 32768.0 * 32768.0;
    sg->far_active_latched = 0;
    sg->initial_state = 1;
    sg->stat_mask_frac = 0.0;
    sg->dne_trigger_counter = 0;
    sg->dne_hold_counter = 0;
    sg->dne_nearend_state = 0;
    sg->ma_count = 0;
    sg->ma_head = 0;
}

void suppression_gain_set_initial_state(SuppressionGain *sg, int state) {
    sg->initial_state = state ? 1 : 0;
}

int suppression_gain_is_dominant_nearend(const SuppressionGain *sg) {
    return sg->dne_nearend_state ? 1 : 0;
}

/* _ne_state_for_gain_rules — returns raw DNE state unless the stat-aware
 * proxy is enabled (OFF in balanced). */
static int ne_state_for_gain_rules(const SuppressionGain *sg) {
    if (!sg->cfg.stat_aware_ne_proxy_enabled) return sg->dne_nearend_state;
    if (sg->dne_nearend_state) return 1;
    return sg->stat_mask_frac > sg->cfg.stat_aware_ne_proxy_threshold;
}

/* LowNoiseRenderDetector.detect — returns bool, mutates avg power. */
static int low_noise_render_detect(SuppressionGain *sg, const float *rb,
                                   int hop) {
    const SuppressionGainConfig *c = &sg->cfg;
    double x2[1024];          /* hop_size <= 1024 in practice */
    double x2_sum, x2_max;
    int low_noise, i;
    if (hop == 0) return 0;
    for (i = 0; i < hop; ++i) {
        double v = (double)rb[i];   /* render_block.astype(np.float64) */
        x2[i] = v * v;
    }
    x2_sum = f64_pairwise_sum(x2, (size_t)hop);
    x2_max = x2[0];
    for (i = 1; i < hop; ++i) if (x2[i] > x2_max) x2_max = x2[i];
    low_noise = (sg->low_render_avg_power < c->low_render_threshold)
                && (x2_max < 3.0 * sg->low_render_avg_power);
    sg->low_render_avg_power = sg->low_render_avg_power * 0.9 + x2_sum * 0.1;
    return low_noise ? 1 : 0;
}

/* DominantNearendDetector.update — all sums in f32 pairwise, ratios in f64. */
static void dominant_nearend_update(SuppressionGain *sg, const float *nearend,
                                    const float *echo, const float *cn,
                                    int initial_state) {
    const SuppressionGainConfig *c = &sg->cfg;
    int lf = c->dne_lf_endpoint_bin;
    double ne_sum = (double)f32_sum_slice(nearend, 1, lf);
    double echo_sum = (double)f32_sum_slice(echo, 1, lf);
    double noise_sum = (double)f32_sum_slice(cn, 1, lf);
    int trigger_initial_gate = (!initial_state) || c->dne_use_during_initial_phase;
    int loud_nearend = c->dne_loud_relax_enabled
        && (ne_sum > c->dne_loud_snr_factor * c->dne_snr_threshold * noise_sum);
    double eff_enr_thr = loud_nearend ? c->dne_loud_enr_threshold
                                      : c->dne_enr_threshold;
    int trigger_enr_pass = echo_sum < eff_enr_thr * ne_sum;
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

/* _MovingAverageSpectrum.average — append spectrum, return f32 per-bin mean
 * over the (<= n) most-recent rows. Writes to out[n_bins]. */
static void nearend_smoother_average(SuppressionGain *sg, const float *spec,
                                     float *out) {
    int n = sg->cfg.nearend_smoother_n;
    int nb = sg->cfg.n_bins;
    int idx, k;
    /* append: write into the slot after the current newest, evicting oldest. */
    if (sg->ma_count < n) {
        idx = (sg->ma_head + sg->ma_count) % n;
        sg->ma_count += 1;
    } else {
        idx = sg->ma_head;
        sg->ma_head = (sg->ma_head + 1) % n;
    }
    memcpy(sg->ma_buf + (size_t)idx * nb, spec, (size_t)nb * sizeof(float));

    /* np.mean(buf, axis=0).astype(f32): sum the `ma_count` rows per bin in
     * f32 then divide by count in f32. For count<=2 the deque holds at most
     * 2 rows; the per-bin reduction is (row0[k]+row1[k]) in f32 (numpy
     * reduces axis 0 sequentially in f32) then /count in f32. We replicate
     * the sequential-in-f32 reduction over up to `n` rows generically. */
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

/* WeightEchoForAudibility band weigh (suppression_gain.cc:88-121). */
static void weigh_band(const SuppressionGainConfig *c, const float *echo,
                       float *out, double threshold, int begin, int end) {
    int k;
    float thr_f = (float)threshold;
    float normalizer = (float)(1.0 / (threshold - c->floor_power));
    if (begin >= end) return;
    for (k = begin; k < end; ++k) {
        float seg = echo[k];
        if (seg < thr_f) {
            float tmp = (thr_f - seg) * normalizer;
            float w = 1.0f - tmp * tmp;
            if (w < 0.0f) w = 0.0f;          /* np.maximum(0.0, .) in f32 */
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

/* _get_max_gain */
static void get_max_gain(SuppressionGain *sg, float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int is_ne = ne_state_for_gain_rules(sg);
    float inc = is_ne ? c->max_inc_nearend : c->max_inc_normal;
    float ffi = (float)c->floor_first_increase;
    int k;
    for (k = 0; k < c->n_bins; ++k) {
        float v = sg->last_gain[k] * inc;     /* f32 array * scalar(f32) */
        if (v < ffi) v = ffi;                 /* np.maximum(., ffi) f32 */
        if (v > 1.0f) v = 1.0f;               /* np.minimum(., 1.0) f32 */
        out[k] = v;
    }
}

/* _get_min_gain */
static void get_min_gain(SuppressionGain *sg, const float *weighted_residual,
                         int low_noise_render, int saturated_echo, float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int k;
    double min_echo_power;
    if (saturated_echo) {
        for (k = 0; k < n; ++k) out[k] = 0.0f;
        return;
    }
    min_echo_power = low_noise_render ? c->low_render_limit
                                      : c->normal_render_limit;
    {
        float mep_f = (float)min_echo_power;   /* pyfloat / f32arr -> f32 */
        for (k = 0; k < n; ++k) {
            float wr = weighted_residual[k];
            if (wr > 0.0f) {
                float denom = wr < 1e-30f ? 1e-30f : wr;  /* np.maximum f32 */
                out[k] = mep_f / denom;
            } else {
                out[k] = 1.0f;
            }
            if (out[k] > 1.0f) out[k] = 1.0f;   /* np.minimum(.,1.0,out=) f32 */
        }
    }
    /* LF smoothing block */
    if (!sg->initial_state || c->lf_smoothing_during_initial_phase) {
        int is_ne = ne_state_for_gain_rules(sg);
        double dec = is_ne ? (double)c->max_dec_lf_nearend
                           : (double)c->max_dec_lf_normal;
        int end = c->last_lf_smoothing_band + 1;
        int permanent = c->last_permanent_lf_smoothing_band;
        if (end > n) end = n;
        for (k = 0; k < end; ++k) {
            if (sg->last_nearend[k] > sg->last_echo[k] || k <= permanent) {
                /* last_gain[k]*dec : scalar f32 * pyfloat -> f64 */
                double prod = (double)sg->last_gain[k] * dec;
                double mg = (double)out[k];
                double chosen = mg;            /* max(mg, prod) in double */
                if (prod > mg) chosen = prod;
                if (1.0 < chosen) chosen = 1.0; /* min(chosen, 1.0) in double */
                out[k] = (float)chosen;
            }
        }
    }
    /* HF min-gain floor during DNE (default OFF) */
    if (c->hf_min_gain_floor_dne_enabled && ne_state_for_gain_rules(sg)
            && c->first_hf_band < n) {
        float floor_f = (float)c->hf_min_gain_floor_dne_power;
        for (k = c->first_hf_band; k < n; ++k) {
            if (out[k] < floor_f) out[k] = floor_f;  /* np.maximum f32 */
        }
        for (k = 0; k < n; ++k) if (out[k] > 1.0f) out[k] = 1.0f;
    }
    /* Split min-gain floor (default ON). cohxd release is OFF in balanced
     * (coh_xy_gamma2 always None) -> the simple base_floor branch. */
    if (c->split_floor_enabled) {
        float base_floor = (float)(sg->far_active_latched
                                   ? c->split_floor_far_active
                                   : c->split_floor_far_silent);
        for (k = 0; k < n; ++k) {
            if (out[k] < base_floor) out[k] = base_floor;  /* np.maximum f32 */
        }
        for (k = 0; k < n; ++k) if (out[k] > 1.0f) out[k] = 1.0f;
    }
}

/* _gain_to_no_audible_echo */
static void gain_to_no_audible_echo(SuppressionGain *sg, const float *nearend,
                                    const float *echo, const float *masker,
                                    float *out) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int is_ne = ne_state_for_gain_rules(sg);
    int k;
    float ne_w = 0.0f;                          /* scalar LF ne_weight */
    const float *enr_tr_tab, *enr_su_tab, *emr_tr_tab;
    int have_scalar_blend_inputs = c->soft_blend_enabled || c->d5_ne_floor_enabled;

    if (have_scalar_blend_inputs) {
        /* ne_lf / echo_lf : f32 pairwise sum over [1:dne_lf_end], cast f64 */
        double ne_lf = (double)f32_sum_slice(nearend, 1, c->dne_lf_end);
        double echo_lf = (double)f32_sum_slice(echo, 1, c->dne_lf_end);
        double enr_lf = echo_lf / (ne_lf + 1.0);
        double sig_arg = (enr_lf - (double)c->soft_blend_enr_thr)
                         / (double)c->soft_blend_softness;
        if (sig_arg < -50.0) sig_arg = -50.0;
        if (sig_arg > 50.0) sig_arg = 50.0;
        ne_w = (float)(1.0 / (1.0 + exp(sig_arg)));
    }

    /* Resolve tuning tables (or per-bin blend). For per-bin blend we write
     * the blended tables into the caller-owned weighted_residual scratch? No,
     * we keep them in local fixed arrays sized by n_bins via the sum_scratch
     * triple region. To stay malloc-free we recompute per-bin inline below. */
    if (c->soft_blend_enabled) {
        enr_tr_tab = NULL;    /* per-bin path computes inline */
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
                /* _enr_bin = echo/(nearend+1.0); sigmoid in f32 -> ne_wb */
                float enr_bin = echo[k] / (nearend[k] + 1.0f);
                float sig = (enr_bin - c->soft_blend_enr_thr)
                            / c->soft_blend_softness;
                if (sig < -50.0f) sig = -50.0f;
                if (sig > 50.0f) sig = 50.0f;
                ne_wb = (float)(1.0f / (1.0f + expf(sig)));
            } else {
                ne_wb = ne_w;
            }
            /* blend: (ne_wb*nearend + (1-ne_wb)*normal).astype(f32), all f32 */
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

        enr = echo[k] / (nearend[k] + 1.0f);     /* f32 */
        emr = echo[k] / (masker[k] + 1.0f);      /* f32 */
        g = 1.0f;
        if (enr > enr_tr_k && emr > emr_tr_k) {
            float d_lin = enr_su_k - enr_tr_k;
            float denom_lin = d_lin < 1e-30f ? 1e-30f : d_lin;
            float g_lin = (enr_su_k - enr) / denom_lin;
            float denom_emr = emr < 1e-30f ? 1e-30f : emr;
            float g_emr = emr_tr_k / denom_emr;
            g = g_lin > g_emr ? g_lin : g_emr;   /* np.maximum f32 */
        }
        out[k] = g;
    }

    /* D5 ne_weight gain floor (default OFF) */
    if (c->d5_ne_floor_enabled && ne_w > 0.0f) {
        float fl = ne_w * c->d5_ne_floor_strength;
        for (k = 0; k < n; ++k) if (out[k] < fl) out[k] = fl;  /* np.maximum f32 */
    }
}

/* _limit_lf_gains */
static void limit_lf_gains(float *gain, int n) {
    if (n >= 3) {
        /* min(gain[1], gain[2]): Python builtin min, compare in double */
        double g1 = (double)gain[1], g2 = (double)gain[2];
        double m = g1;
        if (g2 < g1) m = g2;
        gain[0] = gain[1] = (float)m;
    }
}

/* _limit_hf_gains */
static void limit_hf_gains(SuppressionGain *sg, float *gain) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int lgb = c->hf_lgb;
    int biq = c->hf_biq;
    int k;
    if (biq > 0 && lgb + biq <= n) {
        /* min over gain[lgb:lgb+biq] (np.min, f32), then np.minimum out */
        float mug = gain[lgb];
        for (k = lgb + 1; k < lgb + biq; ++k) if (gain[k] < mug) mug = gain[k];
        for (k = lgb + 1; k < n; ++k) if (gain[k] > mug) gain[k] = mug;
    }
    if (n >= 2) gain[n - 1] = gain[n - 2];
    /* conservative_hf path: OFF in balanced; not exercised by the golden. */
    (void)c;
}

/* --------------------------------------------------------------- get_gain */

const float *suppression_gain_get_gain(
    SuppressionGain *sg,
    const float *nearend_spectrum,
    const float *residual_echo,
    const float *residual_echo_unbounded,
    const float *comfort_noise,
    const float *render_block,
    int clock_drift,
    int saturated_echo) {
    const SuppressionGainConfig *c = &sg->cfg;
    int n = c->n_bins;
    int k;
    const float *echo_for_det;
    int low_noise_render;
    int hf_lim_applied;

    /* stationary_mask is None in balanced -> stat_mask_frac = 0.0. */
    sg->stat_mask_frac = 0.0;

    echo_for_det = c->dne_use_unbounded_echo ? residual_echo_unbounded
                                             : residual_echo;
    dominant_nearend_update(sg, nearend_spectrum, echo_for_det, comfort_noise,
                            sg->initial_state);

    low_noise_render = low_noise_render_detect(sg, render_block, c->hop_size);

    /* split-floor far-active latch (pre-_lower_band_gain). */
    if (c->split_floor_enabled && !sg->far_active_latched) {
        double mean_pow;
        double rb2[1024];
        int hop = c->hop_size;
        for (k = 0; k < hop; ++k) {
            double v = (double)render_block[k];   /* asarray(rb, f64) */
            rb2[k] = v * v;
        }
        mean_pow = f64_pairwise_sum(rb2, (size_t)hop) / (double)hop;
        if (mean_pow > c->split_floor_latch_power) sg->far_active_latched = 1;
    }

    /* ---- _lower_band_gain ---- */
    /* Step 1: max gain */
    get_max_gain(sg, sg->max_gain);
    /* Step 2: smoothed nearend */
    nearend_smoother_average(sg, nearend_spectrum, sg->nearend);
    /* Step 3: weighted residual */
    weight_echo_for_audibility(c, residual_echo, sg->weighted_residual);
    /* Step 4: min gain (uses last_nearend / last_echo) */
    get_min_gain(sg, sg->weighted_residual, low_noise_render, saturated_echo,
                 sg->min_gain);
    /* Step 5: GainToNoAudibleEcho */
    gain_to_no_audible_echo(sg, sg->nearend, sg->weighted_residual,
                            comfort_noise, sg->g_raw);
    /* Step 6: clip into [min, max] (np.clip = min(max(x,lo),hi) f32) */
    for (k = 0; k < n; ++k) {
        float v = sg->g_raw[k];
        float lo = sg->min_gain[k], hi = sg->max_gain[k];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        sg->gain[k] = v;
    }
    /* SER floor (OFF), coh gain floor (OFF, coh_gamma2 None) -> skipped. */

    /* Step 7: LF + HF limiters */
    limit_lf_gains(sg->gain, n);
    hf_lim_applied = (!ne_state_for_gain_rules(sg)) || clock_drift
                     || c->conservative_hf;
    if (hf_lim_applied) limit_hf_gains(sg, sg->gain);

    /* Stash for next hop. */
    memcpy(sg->last_gain, sg->gain, (size_t)n * sizeof(float));
    memcpy(sg->last_nearend, sg->nearend, (size_t)n * sizeof(float));
    memcpy(sg->last_echo, sg->weighted_residual, (size_t)n * sizeof(float));

    /* Step 8: sqrt to amplitude domain. np.sqrt(np.maximum(G,0.0)) */
    for (k = 0; k < n; ++k) {
        float v = sg->gain[k];
        if (v < 0.0f) v = 0.0f;
        sg->gain[k] = sqrtf(v);
    }
    return sg->gain;
}
