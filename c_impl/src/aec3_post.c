/* aec3_post.c — C port of the AEC3 post-filter DRIVER (AEC._aec3_post).
 * WS5 Phase 5.5. See aec3_post.h for the stage breakdown + parity rules.
 *
 * Build: gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99
 *        -Ic_impl/include -Ic_impl/lib/pocketfft
 *   link: aec3_post.c reverb_model.c fft_pocketfft.c pocketfft.c -lm
 */
#include "aec3_post.h"

#include <math.h>
#include <string.h>
#include "fast_math.h"

#define PSD_SCALE (32768.0 * 32768.0)   /* int16 max^2 (Python _PSD_SCALE) */

/* numpy complex64 |z| — scaled-hypot with a FUSED multiply-add (= the `m` inside
 * cmag2_np in pbfdkf.c). This is `np.abs(c64)`. IEEE-754 correctly-rounded, so
 * deterministic + portable; reproduce np.abs everywhere the .py uses it. */
float cabs_np(float re, float im) {
    float ar = re < 0.0f ? -re : re;
    float ai = im < 0.0f ? -im : im;
    float larger  = ar > ai ? ar : ai;
    float smaller = ar > ai ? ai : ar;
    float ratio;
    if (larger == 0.0f) return 0.0f;
    ratio = smaller / larger;
    return larger * sqrtf(fmaf(ratio, ratio, 1.0f));
}

/* numpy-1.26-bit-exact pairwise float32 sum (block <=128, unrolled by 8,
 * recursive split) — matches np.sum over a float32 array. */
static float pairwise_sum_f32(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) {
        float acc[8];
        size_t i, j;
        for (j = 0; j < 8; ++j) acc[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) acc[j] += a[i + j];
        {
            float s = 0.0f, r;
            for (; i < n; ++i) s += a[i];
            r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
              + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            return r + s;
        }
    }
    {
        size_t half = n / 2;
        half -= half % 8;  /* keep the split on an 8-boundary (numpy semantics) */
        return pairwise_sum_f32(a, half) + pairwise_sum_f32(a + half, n - half);
    }
}

float aec3_post_pairwise_sum_f32(const float *a, size_t n) {
    return pairwise_sum_f32(a, n);
}

void aec3_post_config_defaults(Aec3PostConfig *cfg) {
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
}

void aec3_post_init(Aec3Post *p, const Aec3PostConfig *cfg,
                    FftHandle *fft,
                    const float *synth_window, const float *sqrt2_sin_lut,
                    float *avg_reverb_storage,
                    float *y2_smoothed, float *n2, float *n2_initial,
                    float *coh_sye_re, float *coh_sye_im,
                    float *coh_syy, float *coh_see,
                    float *ola_buf,
                    float *near_psd, float *far_psd, float *echo_psd,
                    float *error_psd, float *capture_psd_erle,
                    float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                    float *comfort_noise, float *nf,
                    Complex *e_out_spec, float *e_out_full) {
    p->cfg = *cfg;
    p->trace.aec3_converged = 0;
    p->trace.far_active = 0;
    p->trace.gain_mean = 0.0f;
    p->fft = fft;
    p->synth_window = synth_window;
    p->sqrt2_sin_lut = sqrt2_sin_lut;
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
    p->capture_psd_erle = capture_psd_erle;
    p->x2_reverb_for_erle = x2_reverb_for_erle;
    p->coh_gate_mask = coh_gate_mask;
    p->comfort_noise = comfort_noise;
    p->nf = nf;
    p->e_out_spec = e_out_spec;
    p->e_out_full = e_out_full;
    aec3_post_reset(p);
}

void aec3_post_reset(Aec3Post *p) {
    int k;
    int nb = p->cfg.n_bins;
    /* mirrors _reset_aec3_post (filter-derived path): OLA / CNG / coherence /
     * avg-reverb cleared. */
    memset(p->ola_buf, 0, (size_t)p->cfg.block_size * sizeof(float));
    p->cng_seed = 42u;
    p->noise_initialized = 0;
    p->n2_counter = 0;
    for (k = 0; k < nb; ++k) {
        p->y2_smoothed[k] = 0.0f;
        p->n2[k] = 1.0e6f;
        p->n2_initial[k] = 0.0f;
        p->coh_sye_re[k] = 0.0f;     /* _reset_coherence_state: sye=0 */
        p->coh_sye_im[k] = 0.0f;
        p->coh_syy[k] = 1.0e-30f;    /* syy=see=1e-30 */
        p->coh_see[k] = 1.0e-30f;
    }
    reverb_model_reset(&p->avg_reverb);
}

/* PSD = (mag*mag) in f32, then * _PSD_SCALE in f32 (numpy<2 array*pyfloat rule),
 * .astype(f32) is a no-op. mag is the captured np.abs(c64) f32 value. */
static void psd_from_abs(const float *mag, int n, float *out) {
    int k;
    for (k = 0; k < n; ++k) {
        float a2 = mag[k] * mag[k];              /* |c|^2 in f32 */
        out[k] = (float)(a2 * (float)PSD_SCALE); /* * _PSD_SCALE in f32 */
    }
}

/* ── stage 1-2: PSDs (+ E1 capture_psd_erle) ───────────────────────────── */
void aec3_post_compute_psds(Aec3Post *p, const Aec3PostAbs *mag) {
    int nb = p->cfg.n_bins;
    psd_from_abs(mag->abs_near, nb, p->near_psd);
    psd_from_abs(mag->abs_far, nb, p->far_psd);
    psd_from_abs(mag->abs_sel_echo, nb, p->echo_psd);
    psd_from_abs(mag->abs_error, nb, p->error_psd);
    if (p->cfg.erle_windowed_capture_psd) {
        psd_from_abs(mag->abs_nsw_e1, nb, p->capture_psd_erle);
    }
}

/* ── stage 3: avg-render-reverb x2_reverb_for_erle ─────────────────────── */
void aec3_post_compute_x2_reverb(Aec3Post *p, int x2_present,
                                 const float *x2_at_delay, const float *x2_past,
                                 float decay_steady) {
    const Aec3PostConfig *c = &p->cfg;
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

/* ── stage 4: coherence Γ²(Ŷ,Y) EMA + ERLE coh-gate mask ───────────────── */
void aec3_post_compute_coherence(Aec3Post *p,
                                 const Complex *echo_spec_for_coh,
                                 const Complex *near_spec,
                                 const Aec3PostAbs *mag) {
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins, k;
    float a = c->erle_coh_gate_alpha;
    float af = a;
    float omaf = 1.0f - a;
    if (!c->erle_coh_gate_enabled) return;
    for (k = 0; k < nb; ++k) {
        float er = echo_spec_for_coh[k].r, ei = echo_spec_for_coh[k].i;
        float nr = near_spec[k].r, ni = near_spec[k].i;
        float pr = er * nr + ei * ni;      /* re = ac + bd (echo·conj(near)) */
        float pi = ei * nr - er * ni;      /* im = bc - ad */
        float echo_abs2, near_abs2;
        p->coh_sye_re[k] = omaf * p->coh_sye_re[k] + af * pr;
        p->coh_sye_im[k] = omaf * p->coh_sye_im[k] + af * pi;
        echo_abs2 = mag->abs_echo_coh[k] * mag->abs_echo_coh[k];
        near_abs2 = mag->abs_near[k] * mag->abs_near[k];
        p->coh_syy[k] = (1.0f - a) * p->coh_syy[k] + af * echo_abs2;
        p->coh_see[k] = (1.0f - a) * p->coh_see[k] + af * near_abs2;
    }
    for (k = 0; k < nb; ++k) {
        float sye2 = p->coh_sye_re[k] * p->coh_sye_re[k]
                   + p->coh_sye_im[k] * p->coh_sye_im[k];
        float denom = p->coh_syy[k] * p->coh_see[k];
        float g2;
        if (denom < 1.0e-30f) denom = 1.0e-30f;
        g2 = sye2 / denom;
        p->coh_gate_mask[k] =
            (g2 >= c->erle_coh_gate_threshold) ? 1u : 0u;
    }
}

/* ── stage 5: CNG N2 tracking → comfort_noise ──────────────────────────── */
void aec3_post_compute_comfort_noise(Aec3Post *p, int saturated_capture) {
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins, k;
    if (!p->noise_initialized) {
        for (k = 0; k < nb; ++k) p->y2_smoothed[k] = p->near_psd[k];
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
        for (k = 0; k < nb; ++k) {
            p->y2_smoothed[k] = p->y2_smoothed[k]
                + y2a * (p->near_psd[k] - p->y2_smoothed[k]);
        }
        if (p->n2_counter > c->cng_n2_update_onset_hops) {
            for (k = 0; k < nb; ++k) {
                float track = (fresh * p->y2_smoothed[k]
                               + retain * p->n2[k]) * g_up;
                float up = p->n2[k] * g_up;
                p->n2[k] = (p->y2_smoothed[k] < p->n2[k]) ? track : up;
            }
        }
        if (p->n2_counter < dur) {
            p->n2_counter += 1;
            if (p->n2_counter < dur) {
                for (k = 0; k < nb; ++k) {
                    float slow = p->n2_initial[k]
                               + ia * (p->n2[k] - p->n2_initial[k]);
                    p->n2_initial[k] = (p->n2[k] > p->n2_initial[k])
                                       ? slow : p->n2[k];
                }
            }
        }
        for (k = 0; k < nb; ++k) if (p->n2[k] < nfloor) p->n2[k] = nfloor;
        if (p->n2_counter < dur) {
            for (k = 0; k < nb; ++k)
                if (p->n2_initial[k] < nfloor) p->n2_initial[k] = nfloor;
        }
    }
    {
        const float *cn = (p->n2_counter < c->cng_n2_initial_duration_hops)
                          ? p->n2_initial : p->n2;
        for (k = 0; k < nb; ++k) p->comfort_noise[k] = cn[k];
    }
}

/* ── stage 6: E2 select + gain apply + CNG inject + irfft + OLA ─────────── */
void aec3_post_apply_output(Aec3Post *p,
                            const Complex *error_spec,
                            const Complex *echo_spec_sel,
                            const Aec3PostAbs *mag,
                            int usable_linear,
                            const float *gain,
                            float *out) {
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins;
    int bs = c->block_size;
    int hop = c->hop_size;
    int k;

    /* E2: _out_base = error_spec; switch to y_base = error_spec + sel_echo when
     * (output_capture_when_linear_unusable && !usable && |E|>|Y|). */
    {
        const Complex *out_base = error_spec;
        if (c->output_capture_when_linear_unusable && !usable_linear) {
            float *se2 = p->nf;
            float se, sy;
            for (k = 0; k < nb; ++k)
                se2[k] = mag->abs_error[k] * mag->abs_error[k];
            se = pairwise_sum_f32(se2, (size_t)nb);
            for (k = 0; k < nb; ++k)
                se2[k] = mag->abs_ybase[k] * mag->abs_ybase[k];
            sy = pairwise_sum_f32(se2, (size_t)nb);
            if (se > sy) {
                for (k = 0; k < nb; ++k) {
                    p->e_out_spec[k].r = error_spec[k].r + echo_spec_sel[k].r;
                    p->e_out_spec[k].i = error_spec[k].i + echo_spec_sel[k].i;
                }
                out_base = p->e_out_spec;
            }
        }
        for (k = 0; k < nb; ++k) {
            p->e_out_spec[k].r = out_base[k].r * gain[k];
            p->e_out_spec[k].i = out_base[k].i * gain[k];
        }
    }

    if (c->enable_cng) {
        int n_random = nb - 2;
        uint32_t seed = p->cng_seed;
        static const float inv_psd_scale = 1.0f / (float)PSD_SCALE;  /* 2^-30, exact */
        for (k = 0; k < nb; ++k) {
            float v = p->comfort_noise[k] * inv_psd_scale;
            if (v < 0.0f) v = 0.0f;
            p->nf[k] = fast_sqrt(v);
        }
        for (k = 0; k < n_random; ++k) {
            uint32_t ix;
            int re_idx, im_idx, bin = k + 1;
            float ng, cn_re, cn_im;
            seed = (seed * 69069u + 1u) & 0x7FFFFFFFu;
            ix = seed >> 26;
            re_idx = (int)ix;
            im_idx = (int)((ix + 8u) & 31u);
            cn_re = p->nf[bin] * p->sqrt2_sin_lut[re_idx];
            cn_im = p->nf[bin] * p->sqrt2_sin_lut[im_idx];
            {
                float g2 = gain[bin] * gain[bin];
                float t = 1.0f - g2;
                if (t < 0.0f) t = 0.0f;
                ng = fast_sqrt(t);
            }
            p->e_out_spec[bin].r += ng * cn_re;
            p->e_out_spec[bin].i += ng * cn_im;
        }
        p->cng_seed = seed;
    }

    fft_inverse(p->fft, p->e_out_spec, p->e_out_full);

    for (k = 0; k < bs; ++k) {
        float windowed = p->e_out_full[k] * p->synth_window[k];
        p->ola_buf[k] = p->ola_buf[k] + windowed;
    }
    for (k = 0; k < hop; ++k) out[k] = p->ola_buf[k];
    memmove(p->ola_buf, p->ola_buf + hop,
            (size_t)(bs - hop) * sizeof(float));
    memset(p->ola_buf + (bs - hop), 0, (size_t)hop * sizeof(float));
}

void aec3_post_process(Aec3Post *p,
                       const Complex *near_spec,
                       const Complex *error_spec,
                       const Complex *echo_spec_sel,
                       const Complex *echo_spec_for_coh,
                       const Aec3PostAbs *mag,
                       int x2_present,
                       const float *x2_at_delay, const float *x2_past,
                       float decay_steady,
                       const float *far_end,
                       int saturated_capture,
                       int usable_linear,
                       const float *gain,
                       float *out)
{
    (void)far_end;  /* render_block_scaled feeds injected sub-modules only */
    aec3_post_compute_psds(p, mag);
    aec3_post_compute_x2_reverb(p, x2_present, x2_at_delay, x2_past, decay_steady);
    aec3_post_compute_coherence(p, echo_spec_for_coh, near_spec, mag);
    aec3_post_compute_comfort_noise(p, saturated_capture);
    aec3_post_apply_output(p, error_spec, echo_spec_sel, mag, usable_linear,
                           gain, out);
}

/* ─────────────────────────────────────────────────────────────────────────
 * aec3_post_run — full orchestration of AEC._aec3_post (orchestrator.py
 * lines 3001-3689). Drives the sub-modules in the exact Python order.
 * ──────────────────────────────────────────────────────────────────────── */

/* f32 pairwise sum of x[i]² (Stage 3a: the former f64 upcast used to match
 * float(np.sum(arr.astype(np.float64) ** 2)) is retired — accumulates in f32
 * throughout, same pairwise/unrolled-by-8 structure and split order as
 * pairwise_sum_f32 above; drift vs the numpy reference accepted). */
static float sum_sq_f32_pairwise(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) { float v = a[i]; s += v * v; }
        return s;
    }
    if (n <= 128) {
        float acc[8];
        size_t i, j;
        for (j = 0; j < 8; ++j) { float v = a[j]; acc[j] = v * v; }
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) {
                float v = a[i + j];
                acc[j] += v * v;
            }
        {
            float s = 0.0f, r;
            for (; i < n; ++i) { float v = a[i]; s += v * v; }
            r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
              + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            return r + s;
        }
    }
    {
        size_t half = n / 2;
        half -= half % 8;
        return sum_sq_f32_pairwise(a, half) + sum_sq_f32_pairwise(a + half, n - half);
    }
}

int aec3_post_run(Aec3Post *p,
                  const Aec3PostRunIn *in,
                  const Aec3PostRunObj *obj,
                  Aec3PostRunScratch *sc,
                  float *out,
                  int *out_pending_gain_change,
                  int *out_pending_delay_change)
{
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins;
    int fft_size = c->fft_size;
    int hop = c->hop_size;
    int n_part = in->n_partitions;
    int k, pp;
    Aec3PostAbs mag;
    int saturated_capture = (in->saturation_level > 0.5);
    int pgc = in->pending_gain_change;
    int pdc = in->pending_delay_change;   /* -1 == None */

    /* ── Step 1: shadow-output selection (orchestrator 3029-3036) ─────────── */
    if (in->shadow_present && in->last_shadow_output_time != NULL) {
        linear_filter_select(obj->lfs,
                             in->raw_output, in->near_end,
                             in->last_shadow_output_time,
                             in->error_spec_windowed, in->echo_spec,
                             in->sqrt_hann, obj->fft,
                             sc->sel_esw, sc->sel_echo);
    } else {
        for (k = 0; k < nb; ++k) {
            sc->sel_esw[k] = in->error_spec_windowed[k];
            sc->sel_echo[k] = in->echo_spec[k];
        }
    }

    /* ── Step 2: precompute the np.abs magnitude arrays (3037-3052,
     *            3267-3270, 3629-3630). sel_esw/sel_echo come from Step 1.   */
    for (k = 0; k < nb; ++k) {
        /* E1 near_spec_win = error_spec_windowed + echo_spec (ORIGINAL). */
        sc->nsw_e1[k].r = in->error_spec_windowed[k].r + in->echo_spec[k].r;
        sc->nsw_e1[k].i = in->error_spec_windowed[k].i + in->echo_spec[k].i;
        /* E2 y_base = sel_esw + sel_echo (SELECTED). */
        sc->ybase[k].r = sc->sel_esw[k].r + sc->sel_echo[k].r;
        sc->ybase[k].i = sc->sel_esw[k].i + sc->sel_echo[k].i;
    }
    for (k = 0; k < nb; ++k) {
        sc->abs_near[k]     = cabs_np(in->near_spec[k].r, in->near_spec[k].i);
        sc->abs_far[k]      = cabs_np(in->far_spec[k].r, in->far_spec[k].i);
        sc->abs_sel_echo[k] = cabs_np(sc->sel_echo[k].r, sc->sel_echo[k].i);
        sc->abs_error[k]    = cabs_np(sc->sel_esw[k].r, sc->sel_esw[k].i);
        sc->abs_echo_coh[k] = cabs_np(in->echo_spec[k].r, in->echo_spec[k].i);
        sc->abs_nsw_e1[k]   = cabs_np(sc->nsw_e1[k].r, sc->nsw_e1[k].i);
        sc->abs_ybase[k]    = cabs_np(sc->ybase[k].r, sc->ybase[k].i);
    }
    mag.abs_near = sc->abs_near;
    mag.abs_far = sc->abs_far;
    mag.abs_sel_echo = sc->abs_sel_echo;
    mag.abs_error = sc->abs_error;
    mag.abs_echo_coh = sc->abs_echo_coh;
    mag.abs_nsw_e1 = sc->abs_nsw_e1;
    mag.abs_ybase = sc->abs_ybase;

    /* ── Step 3: PSDs + E1 (3037-3052) ───────────────────────────────────── */
    aec3_post_compute_psds(p, &mag);

    /* render_block_scaled = (far_end * 32768).astype(f32) (3056). */
    for (k = 0; k < hop; ++k)
        sc->render_block_scaled[k] = (float)(in->far_end[k] * 32768.0f);
    /* far_pwr = mean(far_end²) (pairwise f32; 3053). */
    {
        float *fsq = sc->nearend_pwr;   /* borrow scratch */
        for (k = 0; k < hop; ++k) fsq[k] = in->far_end[k] * in->far_end[k];
        (void)fsq;
    }

    /* ── Step 4: convergence flags (3075-3106) ────────────────────────────── */
    {
        float y2_time = sum_sq_f32_pairwise(in->near_end, (size_t)hop);
        float e2_refined = sum_sq_f32_pairwise(in->raw_output, (size_t)hop);
        float y2_thr = 3.73e-4f;
        float y2_thr_low = y2_thr * (20.0f / 50.0f) * (20.0f / 50.0f);
        float y2_thr_div = y2_thr * (30.0f / 50.0f) * (30.0f / 50.0f);
        float e2_coarse = 0.0f;
        int refined_conv, coarse_conv = 0, coarse_conv_relaxed = 0;
        int aec3_converged, all_diverged;
        float min_e2;

        refined_conv = (e2_refined < 0.5f * y2_time) && (y2_time > y2_thr);
        if (in->shadow_present) {
            /* Parseval map: (2·Σ|E[1:-1]|² + |E[0]|² + |E[-1]|²)/fft_size,
             * cmag2_np per bin, f32 sums (Stage 3a; was f64 sums, 3095-3098). */
            const Complex *es = in->shadow_error_spec;
            float inner = 0.0f;
            /* (np.abs(c)²) per bin (cmag2_np), NOT er*er+ei*ei. */
            for (k = 1; k < nb - 1; ++k) {
                float m = cabs_np(es[k].r, es[k].i);
                inner += m * m;
            }
            {
                float m0 = cabs_np(es[0].r, es[0].i);
                float mn = cabs_np(es[nb - 1].r, es[nb - 1].i);
                e2_coarse = (2.0f * inner + m0 * m0 + mn * mn)
                          / (float)fft_size;
            }
            coarse_conv = (e2_coarse < 0.05f * y2_time) && (y2_time > y2_thr);
            coarse_conv_relaxed = (e2_coarse < 0.3f * y2_time)
                                  && (y2_time > y2_thr_low);
        }
        aec3_converged = refined_conv || coarse_conv;
        p->trace.aec3_converged = aec3_converged;   /* audio-passive trace stash */
        min_e2 = in->shadow_present
                 ? (e2_refined < e2_coarse ? e2_refined : e2_coarse)
                 : e2_refined;
        all_diverged = (min_e2 > 1.5f * y2_time) && (y2_time > y2_thr_div);

        /* ── Step 5: filter_state_bridge (3109-3118) ─────────────────────── */
        {
            FilterStateBridge bridge;
            int ext_delay_samples = (in->delay_active ? in->current_delay : -1);
            filter_state_bridge_build(&bridge, obj->fft, in->W0,
                                      fft_size, nb,
                                      in->kalman_P, in->kalman_P_len,
                                      in->shadow_present,
                                      in->main_error_energy,
                                      in->shadow_error_energy,
                                      aec3_converged,
                                      /*main_paused=*/0,
                                      /*mu_final=*/1.0,
                                      ext_delay_samples,
                                      coarse_conv_relaxed,
                                      all_diverged,
                                      sc->bridge_taps);
            /* AecState.update consumes only bridge.filter_converged (the
             * all_filters_diverged / divergence_indicator path feeds
             * TransparentMode, permanently None in balanced). The build is kept
             * for line-by-line fidelity (read-only; only side-effect is irfft). */
            (void)bridge;

            /* ── Step 6: ext_delay (3126-3131) ───────────────────────────── */
            FilterDelayEstimate ext;
            const FilterDelayEstimate *ext_p = NULL;
            if (in->delay_active && in->current_delay >= 0) {
                ext.reported = 1;
                ext.quality = AEC3_DELAY_QUALITY_REFINED;
                ext.delay = in->current_delay;
                ext_p = &ext;
            }

            /* ── Step 7: handle_echo_path_change (3154-3170) ─────────────── */
            if (pgc || pdc != -1) {
                int was_delay = (pdc != -1);
                int gain_change = pgc ? 1 : 0;
                int delay_change = (pdc != -1) ? pdc : AEC3_DELAY_ADJ_NONE;
                aec_state_handle_echo_path_change(obj->state, gain_change,
                                                  delay_change);
                if (was_delay)
                    suppression_gain_set_initial_state(obj->sg, 1);
                pgc = 0;
                pdc = -1;
            }

            /* ── Step 8: update_capture_saturation (3172) ────────────────── */
            aec_state_update_capture_saturation(obj->state,
                                                saturated_capture);

            /* ── Step 9: avg-render-reverb (3195-3220) ───────────────────── */
            {
                int x2_present = 0;
                float decay_steady = 0.0f;
                int curr_p = ((in->partition_idx - 1) % n_part + n_part) % n_part;
                int delay = aec_state_min_direct_path_filter_delay(obj->state);
                int delay_idx, past_idx;
                if (delay < 0) delay = 0;
                if (delay > n_part - 1) delay = n_part - 1;
                delay_idx = ((curr_p - delay) % n_part + n_part) % n_part;
                past_idx = ((curr_p - delay - 1) % n_part + n_part) % n_part;
                if (in->X_buf != NULL && n_part > 0) {
                    const Complex *Xd = in->X_buf + (size_t)delay_idx * nb;
                    const Complex *Xp = in->X_buf + (size_t)past_idx * nb;
                    /* x2 = (np.abs(X)**2).astype(f32) = cmag2_np per bin. */
                    for (k = 0; k < nb; ++k) {
                        float md = cabs_np(Xd[k].r, Xd[k].i);
                        float mp = cabs_np(Xp[k].r, Xp[k].i);
                        sc->x2_at_delay[k] = md * md;
                        sc->x2_past[k] = mp * mp;
                    }
                    x2_present = 1;
                }
                /* decay_steady = ree._reverb_decay(dominant_nearend=False) —
                 * powf(reverb_decay, hop/64) at our hop (NOT verbatim 0.83). */
                decay_steady = ree_reverb_decay_value(obj->ree, 0);
                aec3_post_compute_x2_reverb(p, x2_present, sc->x2_at_delay,
                                            sc->x2_past, decay_steady);
            }

            /* ── Step 10: coherence Γ²(Ŷ,Y) (3259-3279) ──────────────────── */
            if (in->erle_coh_gate_enabled) {
                aec3_post_compute_coherence(p, in->echo_spec, in->near_spec,
                                            &mag);
            }

            /* ── Step 11: aec_state.update (3302-3319) ───────────────────── */
            {
                /* far_pwr = mean(far_end²), f32 pairwise-sum / n (Stage 3a: the
                 * former f64 widen used to match numpy's float(np.mean(...))
                 * exactly is retired — plain f32 compare now, drift accepted). */
                float far_pwr = pairwise_sum_f32(sc->nearend_pwr,
                                                 (size_t)hop) / (float)hop;
                int active_render = (far_pwr > in->active_render_threshold);
                p->trace.far_active = active_render;   /* audio-passive trace stash */
                int x2r_present_state =
                    (in->X_buf != NULL && n_part > 0);
                aec_state_update(obj->state,
                                 aec3_converged,
                                 ext_p,
                                 p->far_psd, p->near_psd, p->error_psd,
                                 p->echo_psd,
                                 active_render,
                                 in->s_ref_max, in->s_coa_max,
                                 /*echo_path_gain (unused; analyzer drives)*/ 1.0,
                                 sc->render_block_scaled, hop,
                                 in->filter_taps_full, in->filter_taps_full_len,
                                 x2r_present_state ? p->x2_reverb_for_erle : NULL,
                                 c->erle_windowed_capture_psd ? p->capture_psd_erle : NULL,
                                 in->erle_coh_gate_enabled ? p->coh_gate_mask : NULL);
            }

            /* ── Step 12: transition_triggered → set_initial_state(0) (3327) */
            if (aec_state_transition_triggered(obj->state))
                suppression_gain_set_initial_state(obj->sg, 0);

            /* ── Step 13: converged_enough + dominant_ne (3344-3349) ─────── */
            {
                int converged_enough = (in->stationarity_active_hops
                                        >= in->stationarity_converge_hops);
                int dominant_ne =
                    suppression_gain_is_dominant_nearend(obj->sg);

                /* ── Step 14: ree update_reverb_models (3354-3400) ───────── */
                /* attach_reverb_decay_estimator: no-op (use_adaptive_decay=False). */
                /* _w_mag2 = |filter.W|² (cmag2_np), shape n_part × nb. */
                for (pp = 0; pp < n_part; ++pp) {
                    const Complex *Wp = in->W_all + (size_t)pp * nb;
                    float *row = sc->w_mag2 + (size_t)pp * nb;
                    for (k = 0; k < nb; ++k) {
                        float m = cabs_np(Wp[k].r, Wp[k].i);
                        row[k] = m * m;
                    }
                }
                {
                    int delay_blocks =
                        aec_state_min_direct_path_filter_delay(obj->state);
                    int fq_valid = 0;
                    float filter_q =
                        aec_state_get_inst_linear_quality_estimate(
                            obj->state, &fq_valid);
                    int stationary_block =
                        stationarity_estimator_is_block_stationary(obj->stationarity);
                    int usable =
                        aec_state_usable_linear_estimate(obj->state);
                    ree_update_reverb_models(obj->ree, sc->w_mag2, n_part,
                                             delay_blocks, filter_q,
                                             /*fq_is_none=*/!fq_valid,
                                             stationary_block);

                    /* ── Step 15: ree.estimate (3402-3411) ───────────────── */
                    {
                        int saturated = aec_state_saturated_echo(obj->state);
                        int transparent =
                            aec_state_transparent_mode_active(obj->state);
                        /* onset = erle_onset_comp_in_dominant or not dominant_ne.
                         * erle_onset_comp_in_dominant is False in balanced. */
                        int onset_comp = obj->ree->erle_onset_comp_in_dominant;
                        int onset = onset_comp || !dominant_ne;
                        const float *erle =
                            aec_state_erle(obj->state, onset);
                        const float *erle_unb =
                            aec_state_erle_unbounded(obj->state);
                        ree_estimate(obj->ree,
                                     p->far_psd, p->near_psd, p->echo_psd,
                                     dominant_ne, usable, saturated, transparent,
                                     erle, erle_unb,
                                     delay_blocks, n_part,
                                     /*force_nonlinear=*/0,
                                     sc->r2, sc->r2_unb);

                        /* ── Step 17: stationarity R² zeroing (3462-3476) ──── */
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
                                    for (k = 0; k < nb; ++k) {
                                        if (stat_mask[k]) {
                                            sc->r2[k] = 0.0f;
                                            sc->r2_unb[k] = 0.0f;
                                        }
                                    }
                                }
                            }

                            /* ── Step 18: nearend_pwr (3484-3488) ──────────── */
                            if (usable) {
                                for (k = 0; k < nb; ++k) {
                                    float e = p->error_psd[k];
                                    float y = p->near_psd[k];
                                    /* np.minimum compares in float. */
                                    sc->nearend_pwr[k] = (e < y) ? e : y;
                                }
                            } else {
                                for (k = 0; k < nb; ++k)
                                    sc->nearend_pwr[k] = p->near_psd[k];
                            }

                            /* ── Step 19: comfort_noise (3502-3570) ────────── */
                            aec3_post_compute_comfort_noise(p,
                                                            saturated_capture);

                            /* ── Step 20: suppression_gain.get_gain (3574) ─── */
                            {
                                const float *gain;
                                int sat_echo = saturated;  /* aec_state.saturated_echo() */
                                gain = suppression_gain_get_gain(
                                    obj->sg, sc->nearend_pwr, sc->r2, sc->r2_unb,
                                    p->comfort_noise, sc->render_block_scaled,
                                    /*clock_drift=*/0, sat_echo);

                                /* audio-passive trace stash: mean per-bin gain. */
                                {
                                    float gsum = 0.0f;
                                    for (k = 0; k < nb; ++k) gsum += gain[k];
                                    p->trace.gain_mean = (nb > 0) ? gsum / (float)nb : 0.0f;
                                }

                                /* ── Step 21: apply_output (3606-3689) ──────── */
                                aec3_post_apply_output(p, sc->sel_esw,
                                                       sc->sel_echo, &mag,
                                                       usable, gain, out);
                            }
                        }
                    }
                }
            }
        }
    }

    if (out_pending_gain_change) *out_pending_gain_change = pgc;
    if (out_pending_delay_change) *out_pending_delay_change = pdc;
    return 0;
}
