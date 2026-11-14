/* aec_post.c -- post-filter driver; see aec_post.h for the stages. */
#include "aec_post.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include "utility/fast_math.h"
#include "utility/simd_aec.h"
#include "aec_scale.h"
#include "aec.h"
#include "aec_internal.h"

#define PSD_SCALE (32768.0 * 32768.0)   /* int16 max^2 */

void aec_post_config_defaults(AecPostConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->n_bins = 257;
    cfg->fft_size = 512;
    cfg->block_size = 320;
    cfg->hop_size = 160;
    cfg->erle_coh_gate_enabled = 1;
    cfg->erle_windowed_capture_psd = 1;
    cfg->erle_render_x2_psd_scale = 1;
    cfg->output_capture_when_linear_unusable = 1;
    cfg->enable_cng = 1;
    cfg->cng_n2_windowed_psd = 0;
    cfg->erle_coh_gate_alpha = 0.05f;
    cfg->erle_coh_gate_threshold = 0.5f;
    cfg->cng_y2_alpha = 0.23156652857908377f;
    cfg->cng_n2_track_freshness = 0.9968377223398316f;
    cfg->cng_n2_track_retention = 0.003162277660168411f;
    cfg->cng_n2_slow_up = 1.0005000750025f;
    cfg->cng_n2_initial_alpha = 0.0024981253125391234f;
    cfg->cng_n2_update_onset_hops = 20;
    cfg->cng_n2_initial_duration_hops = 400;
    cfg->noise_floor_int16sq = 68.50682420305405f;
    /* Default is block_size; aec.c overrides it with the per-rate value. */
    cfg->synth_window_len = cfg->block_size;
}

void aec_post_init(AecPost *p, const AecPostConfig *cfg,
                   FftHandle *fft,
                   const float *synth_window,
                   float *avg_reverb_storage,
                   float *y2_smoothed, float *n2, float *n2_initial,
                   float *coh_sye_re, float *coh_sye_im,
                   float *coh_syy, float *coh_see,
                   float *ola_buf,
                   float *near_psd, float *far_psd, float *echo_psd,
                   float *error_psd, float *capture_psd_win,
                   float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                   float *comfort_noise, float *nf,
                   Complex *e_out_spec, float *e_out_full) {
    p->cfg = *cfg;
    p->far_active = 0;
    /* Convergence-flags y2 threshold: 50^2 * 64 / 32768^2 at the reference grid, scaled to the live
     * hop_size (it is compared against y2_time summed over the hop). */
    p->y2_thr = aec_block_energy_scale(50.0f * 50.0f * 64.0f, cfg->hop_size)
              / (32768.0f * 32768.0f);
    p->fft = fft;
    p->synth_window = synth_window;
    reverb_model_init(&p->avg_reverb, avg_reverb_storage, cfg->n_bins);
    p->y2_smoothed = y2_smoothed;
    p->n2 = n2;
    p->n2_initial = n2_initial;
    p->coh_sye_re = coh_sye_re;
    p->coh_sye_im = coh_sye_im;
    p->coh_syy = coh_syy;
    p->coh_see = coh_see;
    p->ola_buf = ola_buf;
    p->near_psd = near_psd;
    p->far_psd = far_psd;
    p->echo_psd = echo_psd;
    p->error_psd = error_psd;
    p->capture_psd_win = capture_psd_win;
    p->x2_reverb_for_erle = x2_reverb_for_erle;
    p->coh_gate_mask = coh_gate_mask;
    p->comfort_noise = comfort_noise;
    p->nf = nf;
    p->e_out_spec = e_out_spec;
    p->e_out_full = e_out_full;
    aec_post_reset(p);
    p->reset_serial = 0u;
}

void aec_post_reset(AecPost *p) {
    int k;
    int nb = p->cfg.n_bins;
    /* Reset: OLA, CNG, coherence and avg-reverb cleared. */
    if (p->ola_buf) memset(p->ola_buf, 0, (size_t)p->cfg.block_size * sizeof(float));
    p->far_active = 0;
    p->cng_seed = AEC_CNG_SEED;
    p->output_guard_active = 0;
    p->reset_serial += 1u;
    p->noise_initialized = 0;
    p->n2_counter = 0;
    if (p->y2_smoothed) {   /* NULL on a STATE lane, which never runs the comfort-noise tracker */
        for (k = 0; k < nb; ++k) {
            p->y2_smoothed[k] = 0.0f;
            p->n2[k] = 1.0e6f;
            p->n2_initial[k] = 0.0f;
        }
    }
    for (k = 0; k < nb; ++k) {
        p->coh_sye_re[k] = 0.0f;
        p->coh_sye_im[k] = 0.0f;
        p->coh_syy[k] = 1.0e-30f;
        p->coh_see[k] = 1.0e-30f;
    }
    reverb_model_reset(&p->avg_reverb);
    /* Live observer corroboration EMAs. */
    p->observer_obs_err_ema = 0.0;
    p->observer_main_err_ema = 0.0;
}

static void psd_from_abs(const float *mag, int n, float *out) {
    sk_sq_scale_f32(mag, (float)PSD_SCALE, out, n);
}

/* Stage 1-2: PSDs and capture_psd_win. A NULL mag->abs_far leaves p->far_psd untouched. */
static void aec_post_compute_psds(AecPost *p, const AecPostAbs *mag) {
    int nb = p->cfg.n_bins;
    psd_from_abs(mag->abs_near, nb, p->near_psd);
    if (mag->abs_far) psd_from_abs(mag->abs_far, nb, p->far_psd);
    psd_from_abs(mag->abs_sel_echo, nb, p->echo_psd);
    psd_from_abs(mag->abs_error, nb, p->error_psd);
    /* capture_psd_win is read by both erle_windowed_capture_psd and cng_n2_windowed_psd; form it
     * when either is set. */
    if (p->cfg.erle_windowed_capture_psd || p->cfg.cng_n2_windowed_psd) {
        psd_from_abs(mag->abs_nsw_e1, nb, p->capture_psd_win);
    }
}

void aec_post_avg_reverb_slots(int n_part, int partition_idx, int delay,
                               int *delay_idx, int *past_idx) {
    int curr_p = ((partition_idx - 1) % n_part + n_part) % n_part;
    if (delay > n_part - 2) delay = n_part - 2;
    if (delay < 0) delay = 0;
    *delay_idx = ((curr_p - delay) % n_part + n_part) % n_part;
    *past_idx = ((curr_p - delay - 1) % n_part + n_part) % n_part;
}

/* Stage 3: avg-render-reverb -> p->x2_reverb_for_erle (only when x2_present). */
static void aec_post_compute_x2_reverb(AecPost *p, int x2_present,
                                       const float *x2_at_delay, const float *x2_past,
                                       float decay_steady) {
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins, k;
    if (!x2_present) return;
    reverb_model_update_no_freq_shaping(&p->avg_reverb, x2_past,
                                        1.0f, decay_steady);
    for (k = 0; k < nb; ++k) {
        float v = x2_at_delay[k] + p->avg_reverb.reverb[k];
        if (c->erle_render_x2_psd_scale) v = (float)(v * (float)PSD_SCALE);
        p->x2_reverb_for_erle[k] = v;
    }
}

/* Stage 4: coherence EMA -> p->coh_gate_mask (when erle_coh_gate). */
static void aec_post_compute_coherence(AecPost *p,
                                       const Complex *echo_spec_for_coh,
                                       const Complex *near_spec,
                                       const AecPostAbs *mag) {
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins;
    if (!c->erle_coh_gate_enabled) return;
    sk_coherence_ema_gate_f32(p->coh_sye_re, p->coh_sye_im,
                              p->coh_syy, p->coh_see,
                              echo_spec_for_coh, near_spec,
                              mag->abs_echo_coh, mag->abs_near,
                              c->erle_coh_gate_alpha, c->erle_coh_gate_threshold,
                              p->coh_gate_mask, nb);
}

/* Stage 5: CNG N2 tracking -> p->comfort_noise. */
static void aec_post_compute_comfort_noise(AecPost *p, int saturated_capture) {
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins;
    /* With cng_n2_windowed_psd, track the windowed capture PSD instead of the rectangular near_psd
     * (which runs about 3 dB high against the windowed reference). */
    const float *cng_in = (c->cng_n2_windowed_psd) ? p->capture_psd_win
                                                    : p->near_psd;
    if (!p->noise_initialized) {
        memcpy(p->y2_smoothed, cng_in, (size_t)nb * sizeof(float));
        p->noise_initialized = 1;
    }
    if (!saturated_capture) {
        float y2a = c->cng_y2_alpha;
        float fresh = c->cng_n2_track_freshness;
        float retain = c->cng_n2_track_retention;
        float g_up = c->cng_n2_slow_up;
        float ia = c->cng_n2_initial_alpha;
        float nfloor = c->noise_floor_int16sq;
        int dur = c->cng_n2_initial_duration_hops;
        sk_ema_delta_f32(p->y2_smoothed, cng_in, y2a, nb);
        if (p->n2_counter > c->cng_n2_update_onset_hops) {
            sk_n2_track_f32(p->n2, p->y2_smoothed, fresh, retain, g_up, nb);
        }
        if (p->n2_counter < dur) {
            p->n2_counter += 1;
            if (p->n2_counter < dur) {
                sk_n2_initial_track_f32(p->n2_initial, p->n2, ia, nb);
            }
        }
        sk_clip_f32(p->n2, nfloor, (float)INFINITY, nb);
        if (p->n2_counter < dur) {
            sk_clip_f32(p->n2_initial, nfloor, (float)INFINITY, nb);
        }
    }
    {
        const float *cn = (p->n2_counter < c->cng_n2_initial_duration_hops)
                          ? p->n2_initial : p->n2;
        memcpy(p->comfort_noise, cn, (size_t)nb * sizeof(float));
    }
}

/* Stage 6: gain apply on the rendering base, CNG inject (aec_cng_fill), irfft, OLA. output_base is the base chosen by this hop's
 * output guard: the linear residual spectrum, or error + echo when p->output_guard_active. */
static void aec_post_apply_output(AecPost *p,
                                  const Complex *output_base,
                                  const float *gain,
                                  float *out) {
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins;
    int bs = c->block_size;
    int hop = c->hop_size;
    int k;

    assert(p->ola_buf && p->e_out_spec && p->e_out_full);
    sk_capply_gain_f32(p->e_out_spec, output_base, gain, nb);

    /* aec_cng_fill() is the single comfort-noise implementation (an external synthesizer calls the
     * same function); the output stage weights the noise with the total gain it just applied. */
    if (c->enable_cng)
        aec_cng_fill(&p->cng_seed, p->e_out_spec, p->comfort_noise, gain, NULL, 0.0f, nb);

    fft_inverse(p->fft, p->e_out_spec, p->e_out_full);

    /* synth_window_len must equal block_size (asserted); the win_n clamp keeps a build without the
     * assert from reading past the shorter buffer. */
    assert(c->synth_window_len == bs);
    {
        int win_n = (c->synth_window_len < bs) ? c->synth_window_len : bs;
        sk_wola_accumulate_f32(p->ola_buf, p->e_out_full, p->synth_window, win_n);
    }
    for (k = 0; k < hop; ++k) out[k] = p->ola_buf[k];
    memmove(p->ola_buf, p->ola_buf + hop,
            (size_t)(bs - hop) * sizeof(float));
    memset(p->ola_buf + (bs - hop), 0, (size_t)hop * sizeof(float));
}

/* aec_post_run: full orchestration of the post-filter step, driving the sub-modules in order. */

static float sum_sq_f32_pairwise(const float *a, size_t n) {
    return sk_sum_sq_pairwise_f32(a, n);
}

/* Live observer injection step. Runs after the stationarity R^2 zeroing and before the comfort-noise
 * and gain steps, and needs this hop's dominant_ne and usable verdicts, so aec_post_run calls it in
 * place. */
static void aec_post_run_observer_step(AecPost *p,
                                       const AecPostRunIn *in,
                                       const AecPostRunObj *obj,
                                       AecPostRunScratch *sc,
                                       int nb, int dominant_ne, int usable,
                                       const unsigned char *stat_mask) {
    if (!(p->cfg.observer_enabled && obj->observer_filter && in->observer_far_end))
        return;
    int lo_trustworthy = in->render_activity_is_active && usable && !dominant_ne;
    float lo_mu_scale = lo_trustworthy ? 1.0f : 0.0f;
    pbfdaf_process(obj->observer_filter, in->near_end, in->observer_far_end, NULL,
                   lo_mu_scale, sc->observer_out);
    int kmax = p->cfg.observer_max_bin;
    if (kmax > nb) kmax = nb;
    if (kmax < 0) kmax = 0;
    /* |sel_esw|^2 bounded to [0, kmax): bins at or above kmax never reach r2. It feeds both the
     * e_main EMA input and the corroborated-branch cap. */
    sk_cmag2_np_f32(sc->sel_esw, sc->observer_cap_mag2, kmax);
    sk_cmag2_np_f32(obj->observer_filter->error_spec, sc->observer_diff_mag2, kmax);
    float e_obs = sk_pairwise_sum_f32(sc->observer_diff_mag2, (size_t)kmax);
    float e_main = sk_pairwise_sum_f32(sc->observer_cap_mag2, (size_t)kmax);
    double ea = p->cfg.observer_err_ema_alpha;
    p->observer_obs_err_ema =
        ea * p->observer_obs_err_ema + (1.0 - ea) * (double)e_obs;
    p->observer_main_err_ema =
        ea * p->observer_main_err_ema + (1.0 - ea) * (double)e_main;
    int corroborated = (p->observer_obs_err_ema
                         <= p->cfg.observer_corrob_margin * p->observer_main_err_ema);
    if (corroborated && lo_trustworthy) {
        sk_csub_f32(sc->observer_diff_spec, obj->observer_filter->echo_spec,
                   sc->sel_echo, kmax);
        sk_cmag2_np_f32(sc->observer_diff_spec, sc->observer_diff_mag2, kmax);
        sk_min_f32(sc->observer_diff_mag2, sc->observer_diff_mag2,
                  sc->observer_cap_mag2, kmax);
        sk_scale_f32(sc->observer_diff_mag2, sc->observer_diff_mag2,
                    (float)PSD_SCALE, kmax);
        if (stat_mask)
            sk_mask_zero_f32(sc->observer_diff_mag2, stat_mask, kmax);
        sk_raise_to_f32(sc->r2, sc->observer_diff_mag2, kmax);
    }
}

/* Output guard, evaluated once per hop after this hop's usable verdict is final (in every API mode):
 * when output_capture_when_linear_unusable is set, the linear estimate is unusable and the windowed
 * residual carries more energy than the windowed capture (sum|E|^2 > sum|Y|^2, strictly, from the
 * float32 magnitudes with the pairwise reduction), the rendering base is the capture spectrum
 * Y = E + echo, written to sc->output_base. Otherwise the base is the residual E (sc->sel_esw).
 * The analysis spectra (sel_esw, sel_echo, ybase) and the formed hop are never altered. */
void aec_post_eval_output_guard(AecPost *p, AecPostRunScratch *sc, int usable) {
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins;
    float *sq = p->nf;
    float se, sy;
    p->output_guard_active = 0;
    if (!c->output_capture_when_linear_unusable || usable) return;
    sk_cabs_np_f32(sc->ybase, sc->abs_ybase, nb);
    sk_sq_scale_f32(sc->abs_error, 1.0f, sq, nb);
    se = sk_pairwise_sum_f32(sq, (size_t)nb);
    sk_sq_scale_f32(sc->abs_ybase, 1.0f, sq, nb);
    sy = sk_pairwise_sum_f32(sq, (size_t)nb);
    if (se > sy) {
        sk_cadd_f32(sc->output_base, sc->sel_esw, sc->sel_echo, nb);
        p->output_guard_active = 1;
    }
}

void aec_post_run(AecPost *p,
                  const AecPostRunIn *in,
                  const AecPostRunObj *obj,
                  AecPostRunScratch *sc,
                  float *out,
                  int *out_pending_gain_change,
                  int *out_pending_delay_change)
{
    const AecPostConfig *c = &p->cfg;
    int nb = c->n_bins;
    int fft_size = c->fft_size;
    int hop = c->hop_size;
    int n_part = in->n_partitions;
    int k;
    AecPostAbs mag;
    int saturated_capture = (in->saturation_level > 0.5f);
    const float *far_psd, *render_block;
    int output_usable;
    int pgc = in->pending_gain_change;
    int pdc = in->pending_delay_change;

    p->output_guard_active = 0;

    /* Form the linear output on the post-filter WOLA grid. error_spec_windowed is an internal
     * estimator quantity and cannot pair with the synthesis window, so FormLinearFilterOutput
     * always runs; with a shadow filter it also selects refined/coarse, without one it is a
     * stateful WOLA analysis of the refined output. */
    linear_filter_select(obj->lfs,
                         in->raw_output, in->near_end,
                         (in->shadow_present && in->last_shadow_output_time != NULL)
                             ? in->last_shadow_output_time
                             : in->raw_output,
                         in->error_spec_windowed, in->echo_spec,
                         in->sqrt_hann, obj->fft,
                         sc->sel_esw, sc->sel_echo);

    /* near_spec_win = error_spec_windowed + echo_spec (original); y_base = sel_esw + sel_echo
     * (selected). */
    sk_cadd_f32(sc->nsw_e1, in->error_spec_windowed, in->echo_spec, nb);
    sk_cadd_f32(sc->ybase, sc->sel_esw, sc->sel_echo, nb);

    sk_cabs_np_f32(in->near_spec, sc->abs_near, nb);
    if (!in->shared_far_psd) sk_cabs_np_f32(in->far_spec, sc->abs_far, nb);
    sk_cabs_np_f32(sc->sel_echo, sc->abs_sel_echo, nb);
    sk_cabs_np_f32(sc->sel_esw, sc->abs_error, nb);
    sk_cabs_np_f32(in->echo_spec, sc->abs_echo_coh, nb);
    sk_cabs_np_f32(sc->nsw_e1, sc->abs_nsw_e1, nb);
    mag.abs_near = sc->abs_near;
    mag.abs_far = in->shared_far_psd ? NULL : sc->abs_far;
    mag.abs_sel_echo = sc->abs_sel_echo;
    mag.abs_error = sc->abs_error;
    mag.abs_echo_coh = sc->abs_echo_coh;
    mag.abs_nsw_e1 = sc->abs_nsw_e1;

    aec_post_compute_psds(p, &mag);

    if (!in->shared_far_psd) sk_scale_f32(sc->render_block_scaled, in->far_end, 32768.0f, hop);
    far_psd = in->shared_far_psd ? in->shared_far_psd : p->far_psd;
    render_block = in->shared_far_psd ? in->shared_render_block : sc->render_block_scaled;

    {
        float y2_time = sum_sq_f32_pairwise(in->near_end, (size_t)hop);
        float e2_refined = sum_sq_f32_pairwise(in->raw_output, (size_t)hop);
        float y2_thr = p->y2_thr;
        float e2_coarse = 0.0f;
        int refined_conv, coarse_conv = 0;
        int aec_converged;

        refined_conv = (e2_refined < 0.5f * y2_time) && (y2_time > y2_thr);
        if (in->shadow_present) {
            /* Parseval map: (2 * sum|E[1:-1]|^2 + |E[0]|^2 + |E[-1]|^2) / fft_size. */
            const Complex *es = in->shadow_error_spec;
            float inner = 0.0f;
            /* Per-bin |c|^2 via sk_cmag2_np_f32 into x2_at_delay, which is free here (rewritten
             * later in this run); summed serially in order. */
            sk_cmag2_np_f32(es + 1, sc->x2_at_delay, nb - 2);
            for (k = 0; k < nb - 2; ++k) inner += sc->x2_at_delay[k];
            {
                float m0 = sk__cabs_np_elem(es[0].r, es[0].i);
                float mn = sk__cabs_np_elem(es[nb - 1].r, es[nb - 1].i);
                e2_coarse = (2.0f * inner + m0 * m0 + mn * mn)
                          / (float)fft_size;
            }
            coarse_conv = (e2_coarse < 0.05f * y2_time) && (y2_time > y2_thr);
        }
        aec_converged = refined_conv || coarse_conv;

        {
            FilterDelayEstimate ext;
            const FilterDelayEstimate *ext_p = NULL;
            if (in->delay_active && in->current_delay >= 0) {
                ext.reported = 1;
                ext.delay = in->current_delay;
                ext_p = &ext;
            }

            if (pgc || pdc != -1) {
                int was_delay = (pdc != -1);
                int gain_change = pgc ? 1 : 0;
                int delay_change = (pdc != -1) ? pdc : AEC_POST_DELAY_ADJ_NONE;
                aec_state_handle_echo_path_change(obj->state, gain_change,
                                                  delay_change);
                if (was_delay)
                    suppression_gain_set_initial_state(obj->sg, 1);
                pgc = 0;
                pdc = -1;
            }

            aec_state_update_capture_saturation(obj->state,
                                                saturated_capture);

            {
                int x2_present = 0;
                float decay_steady = 0.0f;
                int delay = aec_state_min_direct_path_filter_delay(obj->state);
                int delay_idx = 0, past_idx = 0;
                const float *x2d = NULL, *x2p = NULL;
                if (n_part > 0)
                    aec_post_avg_reverb_slots(n_part, in->partition_idx, delay,
                                              &delay_idx, &past_idx);
                if (in->x2_cache != NULL && n_part > 0) {
                    /* x2 = |X|^2 per bin: both rows are read in place from the filter's x2_cache mirror. */
                    x2d = in->x2_cache + (size_t)delay_idx * nb;
                    x2p = in->x2_cache + (size_t)past_idx * nb;
                    x2_present = 1;
                }
                decay_steady = obj->ree->reverb_decay_per_hop;
                aec_post_compute_x2_reverb(p, x2_present, x2d, x2p, decay_steady);
            }

            if (in->erle_coh_gate_enabled) {
                aec_post_compute_coherence(p, in->echo_spec, in->near_spec,
                                           &mag);
            }

            {
                int active_render = in->shared_far_psd
                    ? in->shared_far_active
                    : (sum_sq_f32_pairwise(in->far_end, (size_t)hop) / (float)hop > in->active_render_threshold);
                p->far_active = active_render;
                int x2r_present_state =
                    (in->x2_cache != NULL && n_part > 0);
                aec_state_update(obj->state,
                                 aec_converged,
                                 ext_p,
                                 far_psd, p->near_psd, p->error_psd,
                                 active_render,
                                 in->s_ref_max, in->s_coa_max,
                                  1.0,
                                 render_block, hop,
                                 in->filter_taps_full, in->filter_taps_full_len,
                                 x2r_present_state ? p->x2_reverb_for_erle : NULL,
                                 c->erle_windowed_capture_psd ? p->capture_psd_win : NULL,
                                 in->erle_coh_gate_enabled ? p->coh_gate_mask : NULL);
            }

            if (aec_state_transition_triggered(obj->state))
                suppression_gain_set_initial_state(obj->sg, 0);

            /* Rendering base: the output guard on this hop's final usable verdict, before any context
             * or linear-only exit, so every API mode publishes the same decision. */
            output_usable = aec_state_usable_linear_estimate(obj->state);
            aec_post_eval_output_guard(p, sc, output_usable);

            /* Linear-only context: the remaining stages only produce R^2, comfort noise and G_res
             * and advance the state that only they read, so stop here; the linear filter and
             * formed output are unaffected. */
            if (in->linear_context_only) {
                assert(in->context_only);
                memcpy(out, in->raw_output, (size_t)hop * sizeof(float));
                goto post_done;
            }

            {
                int converged_enough = (in->stationarity_active_hops
                                        >= in->stationarity_converge_hops);
                int dominant_ne =
                    suppression_gain_is_dominant_nearend(obj->sg);

                {
                    int delay_blocks =
                        aec_state_min_direct_path_filter_delay(obj->state);
                    int fq_valid = 0;
                    float filter_q =
                        aec_state_get_inst_linear_quality_estimate(
                            obj->state, &fq_valid);
                    int usable = output_usable;

                    /* DT-gated min-gain floor lift (read when the gain is computed below). The
                     * energy-based near-recent latch decides it; the result is exported via
                     * AecResidualView.res_floor_protect so fused consumers do not derive a second rule. */
                    obj->sg->dt_protect_active =
                        (in->dt_floor_enabled && in->nearend_recent) ? 1 : 0;

                    /* |W|^2 (cmag2_np) as an n_part x nb block, but only the two rows the reverb
                     * update reads (the direct-path row delay_blocks and the tail row n_part - 1)
                     * are materialized, under the same guards as that consumer. Other rows are left
                     * stale and not carried across hops. */
                    if (!in->stationary_block && fq_valid &&
                        delay_blocks >= 0 && delay_blocks < n_part) {
                        int tail_row = n_part - 1;
                        size_t doff = (size_t)delay_blocks * (size_t)nb;
                        sk_cmag2_np_f32(in->W_all + doff, sc->w_mag2 + doff, nb);
                        if (tail_row != delay_blocks) {
                            size_t toff = (size_t)tail_row * (size_t)nb;
                            sk_cmag2_np_f32(in->W_all + toff,
                                            sc->w_mag2 + toff, nb);
                        }
                    }
                    ree_update_reverb_models(obj->ree, sc->w_mag2, n_part,
                                             delay_blocks, filter_q,
                                             !fq_valid,
                                             in->stationary_block);

                    {
                        int saturated = aec_state_saturated_echo(obj->state);
                        int onset_comp = obj->ree->erle_onset_comp_in_dominant;
                        int onset = onset_comp || !dominant_ne;
                        const float *erle =
                            aec_state_erle(obj->state, onset);
                        const float *erle_unb =
                            aec_state_erle_unbounded(obj->state);
                        ree_estimate(obj->ree,
                                     far_psd, p->near_psd, p->echo_psd,
                                     dominant_ne, usable, saturated, /*transparent_mode=*/0,
                                     erle, erle_unb,
                                     delay_blocks, n_part,
                                     0,
                                     sc->r2, sc->r2_unb);

                        /* Stationarity R^2 zeroing */
                        {
                            int use_stat = in->use_stationarity_properties;
                            int need = use_stat && converged_enough;
                            const unsigned char *stat_mask = NULL;
                            if (need) {
                                stationarity_estimator_band_stationary_mask(
                                    obj->stationarity, sc->stat_mask);
                                stat_mask = sc->stat_mask;
                            }
                            if (use_stat && converged_enough && stat_mask) {
                                int any = 0;
                                for (k = 0; k < nb; ++k)
                                    if (stat_mask[k]) { any = 1; break; }
                                if (any) {
                                    sk_mask_zero_f32(sc->r2, stat_mask, nb);
                                    sk_mask_zero_f32(sc->r2_unb, stat_mask, nb);
                                }
                            }

                            /* Live observer, after the stationarity R^2 zeroing. */
                            aec_post_run_observer_step(p, in, obj, sc, nb,
                                                       dominant_ne, usable,
                                                       stat_mask);

                            if (usable) {
                                sk_min_f32(sc->nearend_pwr, p->error_psd,
                                          p->near_psd, nb);
                            } else {
                                memcpy(sc->nearend_pwr, p->near_psd,
                                      (size_t)nb * sizeof(float));
                            }

                            aec_post_compute_comfort_noise(p,
                                                           saturated_capture);

                            {
                                const float *gain;
                                int sat_echo = saturated;
                                if (in->hold_state_only) {
                                    /* G_res is unused in this mode (context_only is required), but
                                     * the DominantNearend hold state must keep advancing: the next
                                     * hop's ERLE onset flag and r2 depend on it. */
                                    suppression_gain_update_dominant_nearend(
                                        obj->sg, sc->nearend_pwr, sc->r2,
                                        sc->r2_unb, p->comfort_noise);
                                    gain = NULL;
                                } else {
                                    gain = suppression_gain_get_gain(
                                        obj->sg, sc->nearend_pwr, sc->r2, sc->r2_unb,
                                        p->comfort_noise, render_block,
                                        0, sat_echo);
                                }

                                /* hold_state_only implies context_only
                                 * (aec_resolve()), so the NULL gain is never
                                 * dereferenced by aec_post_apply_output(). */
                                assert(!in->hold_state_only ||
                                       in->context_only);
                                if (in->context_only) {
                                    /* context_only: the external joint stage consumes E(f), gain,
                                     * R^2 and comfort_noise and does its own CNG/IFFT/OLA, so
                                     * return the linear residual and leave the private OLA/CNG
                                     * phase alone. */
                                    memcpy(out, in->raw_output,
                                           (size_t)hop * sizeof(float));
                                } else {
                                    aec_post_apply_output(p,
                                        p->output_guard_active ? sc->output_base
                                                               : sc->sel_esw,
                                        gain, out);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

post_done:
    if (out_pending_gain_change) *out_pending_gain_change = pgc;
    if (out_pending_delay_change) *out_pending_delay_change = pdc;
}
