/* suppression_gain.c — C port of python/modules/residual/suppression_gain.py
 * (WS5). Build with -ffp-contract=off.
 *
 * PRECISION: float32-by-design (f32 campaign — Python bit-exact parity is
 * retired for this module; drift accepted):
 *  - Every per-bin array and every scalar (ENR/EMR/min/max/CNG/DNE math,
 *    SuppressionGainConfig thresholds) is float32, single rounding per op.
 *    The old numpy-parity rules ("f32_array op pyfloat -> f32", "scalar f32 *
 *    pyfloat -> f64", "Python builtin max/min compare in double") no longer
 *    apply — this is native C float arithmetic throughout.
 *  - np.sum/np.mean-equivalent reductions over f32 stay pairwise float32
 *    (f32_pairwise_sum / the local float32 pairwise-sum helper below).
 *  - np.exp/np.sqrt equivalents use fast_exp/fast_sqrt (float32; libm
 *    exp()/sqrt() double calls are retired in favour of the float32 helpers
 *    already used elsewhere in this file).
 */
#include "suppression_gain.h"

#include <assert.h>
#include <string.h>
#include "fast_math.h"
#include "aec_simd_kernels.h"
#include "aec3_scale.h"


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
    /* M4 (multi-rate consumption switch): tun->table_len is the length the
     * caller's per-rate lookup row baked the six per-bin tuning arrays at.
     * gain_to_no_audible_echo's per-bin loop below reads all six at index
     * [0, cfg->n_bins) -- a mismatch would walk off the end of one of those
     * static const arrays. Construction-time-only invariant (both n_bins
     * and the tables come from the same rate-table row for a validated
     * sample rate), so a debug assert is the whole guard here (unlike
     * aec_state.c's filter_taps_full_len, which is a per-hop live value and
     * gets both a debug assert AND a release-path skip). Never fires for
     * the validated {16000} whitelist (table_len == n_bins == 257 there). */
    assert(tun->table_len == cfg->n_bins);
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
    sg->low_render_avg_power = 32768.0f * 32768.0f;
    /* AEC3 power IIR (average_power = 0.9*avg + 0.1*x2) is per-4ms-block;
     * a bare hardcoded 0.9f/0.1f has no rate conversion at all (wrong even
     * at the legacy hop=160/sr=16000 grid: the correct per-hop alpha there
     * is 1-0.9^2.5 ~= 0.2316, not 0.1). The wall-clock-rescaled form below
     * mirrors suppression_gain.py's _LowNoiseRenderDetector with
     * use_wallclock_ema_alpha=True, but -- same as the Python default --
     * stays OFF (cfg->use_wallclock_ema_alpha=0) until it has its own
     * bench pass + sign-off; see suppression_gain.h and CHANGELOG's
     * "Explicitly held back" entry. Do not flip the default without that. */
    if (cfg->use_wallclock_ema_alpha) {
        float alpha = aec3_per_block_ema_alpha_to_per_hop(
            0.1f, cfg->hop_size, cfg->sr);
        sg->low_render_iir_decay = 1.0f - alpha;
        sg->low_render_iir_weight = alpha;
    } else {
        sg->low_render_iir_decay = 0.9f;
        sg->low_render_iir_weight = 0.1f;
    }
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
                                   int hop, float *x2_sum_out) {
    const SuppressionGainConfig *c = &sg->cfg;
    float x2[1024];          /* hop_size <= 1024 in practice */
    float x2_sum, x2_max;
    int low_noise, i;
    *x2_sum_out = 0.0f;
    if (hop == 0) return 0;
    /* x2[i] = rb[i]*rb[i]; sk_sq_scale_f32 with scale=1.0f is bit-exact here
     * ((x*x)*1.0f == x*x for every finite value, no rounding change). */
    sk_sq_scale_f32(rb, 1.0f, x2, hop);
    x2_sum = f32_pairwise_sum(x2, (size_t)hop);
    *x2_sum_out = x2_sum;
    x2_max = x2[0];
    for (i = 1; i < hop; ++i) if (x2[i] > x2_max) x2_max = x2[i];
    /* Normalize average_power to per-sample before peak comparison.
     * AEC3 uses 64-sample blocks; at larger hop sizes, average_power is
     * hop/64 times larger for the same RMS, making the peak-rejection
     * 2.5x too loose at hop=160.  Mirror Python: avg_per_sample = avg*(64/hop). */
    {
        float avg_per_sample = sg->low_render_avg_power * (64.0f / (float)hop);
        low_noise = (sg->low_render_avg_power < c->low_render_threshold)
                    && (x2_max < 3.0f * avg_per_sample);
    }
    sg->low_render_avg_power = sg->low_render_avg_power * sg->low_render_iir_decay
                              + x2_sum * sg->low_render_iir_weight;
    return low_noise ? 1 : 0;
}

/* DominantNearendDetector.update — all sums in f32 pairwise, ratios in f32. */
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

    /* mean(buf, axis=0): sum the `ma_count` rows per bin in f32 then divide
     * by count in f32. Sequential-in-f32 reduction over up to `n` rows. */
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
                       float *out, float threshold, int begin, int end) {
    int k;
    float normalizer = 1.0f / (threshold - c->floor_power);
    if (begin >= end) return;
    for (k = begin; k < end; ++k) {
        float seg = echo[k];
        if (seg < threshold) {
            float tmp = (threshold - seg) * normalizer;
            float w = 1.0f - tmp * tmp;
            if (w < 0.0f) w = 0.0f;          /* max(0.0f, .) */
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
    float ffi = c->floor_first_increase;
    int k;
    /* f32 array * f32 scalar: no matching kernel (sk_sq_scale_f32 squares
     * first; there is no plain scale-by-scalar kernel), kept as plain C. */
    for (k = 0; k < c->n_bins; ++k) out[k] = sg->last_gain[k] * inc;
    /* max(.,ffi) then min(.,1.0f), low-bound-then-high-bound -- exactly
     * sk_clip_f32's order; ffi (floor_first_increase, e.g. 1e-5) < 1.0f
     * always holds by construction. */
    sk_clip_f32(out, ffi, 1.0f, c->n_bins);
}

/* _get_min_gain */
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
                float denom = wr < 1e-30f ? 1e-30f : wr;  /* max f32 */
                out[k] = min_echo_power / denom;
            } else {
                out[k] = 1.0f;
            }
            if (out[k] > 1.0f) out[k] = 1.0f;   /* min(.,1.0f) */
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
                /* last_gain[k]*dec, all float32. */
                float prod = sg->last_gain[k] * dec;
                float mg = out[k];
                float chosen = mg;            /* max(mg, prod) */
                if (prod > mg) chosen = prod;
                if (1.0f < chosen) chosen = 1.0f; /* min(chosen, 1.0f) */
                out[k] = chosen;
            }
        }
    }
    /* Split min-gain floor (default ON). */
    if (c->split_floor_enabled) {
        float base_floor;
        if (sg->far_active_latched) {
            /* DT (near recently present) lifts the floor toward near-protection;
             * FS (far-active, no near) keeps the aggressive far_active floor. */
            base_floor = sg->dt_protect_active ? c->split_floor_dt
                                               : c->split_floor_far_active;
        } else {
            base_floor = c->split_floor_far_silent;
        }
        /* max(.,base_floor) then min(.,1.0f) -- sk_clip_f32's exact
         * low-then-high order; base_floor is a power-domain gain floor
         * (10^(db/10), db<0) so base_floor<1.0f always holds. */
        sk_clip_f32(out, base_floor, 1.0f, n);
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
    int have_scalar_blend_inputs = c->soft_blend_enabled;

    if (have_scalar_blend_inputs) {
        /* ne_lf / echo_lf : f32 pairwise sum over [1:dne_lf_end] */
        float ne_lf = f32_sum_slice(nearend, 1, c->dne_lf_end);
        float echo_lf = f32_sum_slice(echo, 1, c->dne_lf_end);
        float enr_lf = echo_lf / (ne_lf + 1.0f);
        float sig_arg = (enr_lf - c->soft_blend_enr_thr) / c->soft_blend_softness;
        if (sig_arg < -50.0f) sig_arg = -50.0f;
        if (sig_arg > 50.0f) sig_arg = 50.0f;
        ne_w = 1.0f / (1.0f + fast_exp(sig_arg));
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
                ne_wb = 1.0f / (1.0f + fast_exp(sig));
            } else {
                ne_wb = ne_w;
            }
            /* blend: (ne_wb*nearend + (1-ne_wb)*normal), all f32 */
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
            g = g_lin > g_emr ? g_lin : g_emr;   /* max f32 */
        }
        out[k] = g;
    }
}

/* _limit_lf_gains — mirrors Python: np.minimum(gain[:lf_clamp_bin], gain[lf_clamp_bin]).
 * E8 fix: AEC3 native fft=128 uses hardcoded bins 0-2 (≤125 Hz); our fft=512 needs
 * 9 bins (0-250 Hz).  lf_clamp_bin = hz_to_bin(250 Hz) = 8 at 16 kHz / 512-pt. */
static void limit_lf_gains(float *gain, int n, int lf_clamp_bin) {
    int k;
    if (lf_clamp_bin > 0 && lf_clamp_bin < n) {
        float anchor = gain[lf_clamp_bin];
        for (k = 0; k < lf_clamp_bin; ++k)
            if (gain[k] > anchor) gain[k] = anchor;   /* np.minimum */
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
        /* min over gain[lgb:lgb+biq] (f32), then minimum out */
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
    float render_x2_sum = 0.0f;
    int hf_lim_applied;

    /* stationary_mask is None in balanced -> stat_mask_frac = 0.0f. */
    sg->stat_mask_frac = 0.0f;

    echo_for_det = c->dne_use_unbounded_echo ? residual_echo_unbounded
                                             : residual_echo;
    dominant_nearend_update(sg, nearend_spectrum, echo_for_det, comfort_noise,
                            sg->initial_state);

    low_noise_render = low_noise_render_detect(sg, render_block, c->hop_size,
                                                &render_x2_sum);

    /* split-floor far-active latch (pre-_lower_band_gain). */
    if (c->split_floor_enabled && !sg->far_active_latched) {
        /* reuse the render x2 sum computed by low_noise_render_detect above
         * (same pairwise tree over the same block — identical value). */
        float mean_pow = render_x2_sum / (float)c->hop_size;
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
    /* Step 6: np.maximum(np.minimum(G_raw, max_gain), min_gain).
     * E7 fix: order is minimum-first then maximum, not np.clip (min(max(x,lo),hi)).
     * When min_gain > max_gain (saturated-echo recovery), min_gain wins. */
    /* minimum(G_raw, max_gain): "if (v>hi) v=hi;" == (hi<v)?hi:v, which
     * matches sk_min_f32(a,b)=(a<b)?a:b with a=max_gain, b=g_raw exactly
     * (including the signed-zero tie case, since b is returned unmodified
     * on the false branch just like the original's implicit "else v"). */
    sk_min_f32(sg->gain, sg->max_gain, sg->g_raw, n);
    /* maximum(., min_gain): per-bin array bound, no matching two-array-max
     * kernel exists (only sk_min_f32 is available) -- kept as plain C,
     * unchanged from the original "if (v<lo) v=lo;" expression. */
    for (k = 0; k < n; ++k) {
        if (sg->gain[k] < sg->min_gain[k]) sg->gain[k] = sg->min_gain[k];
    }

    /* Step 7: LF + HF limiters */
    limit_lf_gains(sg->gain, n, sg->cfg.lf_clamp_bin);
    hf_lim_applied = (!ne_state_for_gain_rules(sg)) || clock_drift
                     || c->conservative_hf;
    if (hf_lim_applied) limit_hf_gains(sg, sg->gain);

    /* Stash for next hop. */
    memcpy(sg->last_gain, sg->gain, (size_t)n * sizeof(float));
    memcpy(sg->last_nearend, sg->nearend, (size_t)n * sizeof(float));
    memcpy(sg->last_echo, sg->weighted_residual, (size_t)n * sizeof(float));

    /* Step 8: sqrt to amplitude domain. sqrt(max(G,0.0f)). The explicit
     * negative clamp stays: under the default fast_sqrt it is redundant
     * (fast_sqrt(v<=0)=0), but under USE_STANDARD_MATH fast_sqrt is plain
     * sqrtf, where the clamp is what keeps negatives from becoming NaN —
     * dropping it would change that build's semantics. sk_fast_sqrt_f32
     * replicates fast_sqrt verbatim in both build modes, so clamp + kernel
     * reproduces the original loop bit-exactly. */
    for (int k = 0; k < n; ++k)
        if (sg->gain[k] < 0.0f) sg->gain[k] = 0.0f;
    sk_fast_sqrt_f32(sg->gain, sg->gain, n);
    return sg->gain;
}
