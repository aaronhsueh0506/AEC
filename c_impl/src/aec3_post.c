/* aec3_post.c — C port of the AEC3 post-filter DRIVER (AEC._aec3_post).
 * WS5 Phase 5.5. See aec3_post.h for the stage breakdown + parity rules.
 *
 * Build: gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -Ic_impl/include
 *   link: aec3_post.c reverb_model.c fft_fp64.c -lm
 */
#include "aec3_post.h"

#include <math.h>
#include <string.h>

#define PSD_SCALE (32768.0 * 32768.0)   /* int16 max^2 (Python _PSD_SCALE) */

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
    cfg->erle_coh_gate_alpha = 0.05;
    cfg->erle_coh_gate_threshold = 0.5;
    cfg->cng_y2_alpha = 0.23156652857908377;
    cfg->cng_n2_track_freshness = 0.9968377223398316;
    cfg->cng_n2_track_retention = 0.003162277660168411;
    cfg->cng_n2_slow_up = 1.0005000750025;
    cfg->cng_n2_initial_alpha = 0.0024981253125391234;
    cfg->cng_n2_update_onset_hops = 20;
    cfg->cng_n2_initial_duration_hops = 400;
    cfg->noise_floor_int16sq = 68.50682420305405;
}

void aec3_post_init(Aec3Post *p, const Aec3PostConfig *cfg,
                    FftHandle *fft,
                    const float *synth_window, const float *sqrt2_sin_lut,
                    float *avg_reverb_storage,
                    float *y2_smoothed, float *n2, float *n2_initial,
                    float *coh_sye_re, float *coh_sye_im,
                    double *coh_syy, double *coh_see,
                    float *ola_buf,
                    float *near_psd, float *far_psd, float *echo_psd,
                    float *error_psd, float *capture_psd_erle,
                    float *x2_reverb_for_erle, unsigned char *coh_gate_mask,
                    float *comfort_noise, float *nf,
                    Complex *e_out_spec, float *e_out_full) {
    p->cfg = *cfg;
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
        p->coh_syy[k] = 1.0e-30;     /* syy=see=1e-30 */
        p->coh_see[k] = 1.0e-30;
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

void aec3_post_process(Aec3Post *p,
                       const Complex *near_spec,
                       const Complex *error_spec,
                       const Complex *echo_spec_sel,
                       const Complex *echo_spec_for_coh,
                       const Aec3PostAbs *mag,
                       int x2_present,
                       const float *x2_at_delay, const float *x2_past,
                       double decay_steady,
                       const float *far_end,
                       int saturated_capture,
                       int usable_linear,
                       const float *gain,
                       float *out)
{
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins;
    int fft_size = c->fft_size;
    int bs = c->block_size;
    int hop = c->hop_size;
    int k;

    (void)far_end;  /* render_block_scaled feeds injected sub-modules only */

    /* ---- stage 1: PSDs from the captured magnitudes ----------------------- */
    psd_from_abs(mag->abs_near, nb, p->near_psd);
    psd_from_abs(mag->abs_far, nb, p->far_psd);
    psd_from_abs(mag->abs_sel_echo, nb, p->echo_psd);
    psd_from_abs(mag->abs_error, nb, p->error_psd);

    /* ---- stage 2: E1 windowed-capture Y² --------------------------------- */
    if (c->erle_windowed_capture_psd) {
        psd_from_abs(mag->abs_nsw_e1, nb, p->capture_psd_erle);
    }

    /* ---- stage 3: avg-render-reverb x2_reverb_for_erle -------------------- *
     * _avg.update_no_freq_shaping(x2_past, scaling=1.0, decay=decay_steady);
     * x2_reverb = (x2_at_delay + _avg.reverb).astype(f32)
     * (* _PSD_SCALE when erle_render_x2_psd_scale).                          */
    if (x2_present) {
        reverb_model_update_no_freq_shaping(&p->avg_reverb, x2_past,
                                            1.0f, (float)decay_steady);
        for (k = 0; k < nb; ++k) {
            float v = x2_at_delay[k] + p->avg_reverb.reverb[k];
            if (c->erle_render_x2_psd_scale) {
                v = (float)(v * (float)PSD_SCALE);
            }
            p->x2_reverb_for_erle[k] = v;
        }
    }

    /* ---- stage 4: coherence Γ²(Ŷ,Y) EMA + ERLE coh-gate mask -------------- *
     * sye (complex64) += component-wise f32 EMA; syy/see (float64).
     * echo_c = filter.echo_spec (echo_spec_for_coh), near_c = near_spec.     */
    if (c->erle_coh_gate_enabled) {
        double a = c->erle_coh_gate_alpha;
        float af = (float)a;
        float omaf = (float)(1.0 - a);
        for (k = 0; k < nb; ++k) {
            float er = echo_spec_for_coh[k].r, ei = echo_spec_for_coh[k].i;
            float nr = near_spec[k].r, ni = near_spec[k].i;
            /* prod = echo_c * conj(near_c)  (complex64) */
            float pr = er * nr + ei * ni;      /* re = ac + bd */
            float pi = ei * nr - er * ni;      /* im = bc - ad */
            float echo_abs2, near_abs2;
            p->coh_sye_re[k] = omaf * p->coh_sye_re[k] + af * pr;
            p->coh_sye_im[k] = omaf * p->coh_sye_im[k] + af * pi;
            /* syy/see: f64 EMA; the (a · |·|²) term is f32 then widened to f64.
             * |·|² uses the captured magnitudes (numpy abs), squared in f32. */
            echo_abs2 = mag->abs_echo_coh[k] * mag->abs_echo_coh[k];
            near_abs2 = mag->abs_near[k] * mag->abs_near[k];
            p->coh_syy[k] = (1.0 - a) * p->coh_syy[k]
                          + (double)(af * echo_abs2);
            p->coh_see[k] = (1.0 - a) * p->coh_see[k]
                          + (double)(af * near_abs2);
        }
        for (k = 0; k < nb; ++k) {
            double sye2 = (double)p->coh_sye_re[k] * p->coh_sye_re[k]
                        + (double)p->coh_sye_im[k] * p->coh_sye_im[k];
            double denom = p->coh_syy[k] * p->coh_see[k];
            float g2;
            if (denom < 1.0e-30) denom = 1.0e-30;   /* np.maximum(., 1e-30) */
            g2 = (float)(sye2 / denom);             /* .astype(f32) */
            p->coh_gate_mask[k] =
                (g2 >= (float)c->erle_coh_gate_threshold) ? 1u : 0u;
        }
    }

    /* ---- stage 5: CNG N2 tracking → comfort_noise ------------------------ */
    if (!p->noise_initialized) {
        for (k = 0; k < nb; ++k) p->y2_smoothed[k] = p->near_psd[k];
        p->noise_initialized = 1;
    }
    if (!saturated_capture) {
        float y2a = (float)c->cng_y2_alpha;
        float fresh = (float)c->cng_n2_track_freshness;
        float retain = (float)c->cng_n2_track_retention;
        float g_up = (float)c->cng_n2_slow_up;
        float ia = (float)c->cng_n2_initial_alpha;
        float nfloor = (float)c->noise_floor_int16sq;
        int dur = c->cng_n2_initial_duration_hops;

        /* y2_smoothed EMA (all-f32): a += α·(b - a). */
        for (k = 0; k < nb; ++k) {
            p->y2_smoothed[k] = p->y2_smoothed[k]
                + y2a * (p->near_psd[k] - p->y2_smoothed[k]);
        }
        /* N2 update after warm-up (all-f32 track + slow-up). */
        if (p->n2_counter > c->cng_n2_update_onset_hops) {
            for (k = 0; k < nb; ++k) {
                float track = (fresh * p->y2_smoothed[k]
                               + retain * p->n2[k]) * g_up;
                float up = p->n2[k] * g_up;
                p->n2[k] = (p->y2_smoothed[k] < p->n2[k]) ? track : up;
            }
        }
        /* N2_initial transient (all-f32). */
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
        /* Clamp to noise floor (np.maximum compares in float). */
        for (k = 0; k < nb; ++k) {
            if (p->n2[k] < nfloor) p->n2[k] = nfloor;
        }
        if (p->n2_counter < dur) {
            for (k = 0; k < nb; ++k) {
                if (p->n2_initial[k] < nfloor) p->n2_initial[k] = nfloor;
            }
        }
    }
    /* Select N2 to use (N2_initial during the transient, else N2). */
    {
        const float *cn = (p->n2_counter < c->cng_n2_initial_duration_hops)
                          ? p->n2_initial : p->n2;
        for (k = 0; k < nb; ++k) p->comfort_noise[k] = cn[k];
    }

    /* ---- stage 6: output base select + gain apply + CNG + irfft + OLA ----- */
    /* E2: _out_base = error_spec; switch to y_base = error_spec + sel_echo when
     * (output_capture_when_linear_unusable && !usable && |E|>|Y|). */
    {
        const Complex *out_base = error_spec;
        Complex *yb = NULL;  /* lazily formed when E2 fires */
        if (c->output_capture_when_linear_unusable && !usable_linear) {
            /* sum(|error_spec|^2) vs sum(|y_base|^2): pairwise-f32, compare f64.
             * |error_spec| and |y_base| are the captured magnitudes. */
            float *se2 = p->nf;          /* reuse scratch for the squared mags */
            double se, sy;
            for (k = 0; k < nb; ++k)
                se2[k] = mag->abs_error[k] * mag->abs_error[k];
            se = (double)pairwise_sum_f32(se2, (size_t)nb);
            for (k = 0; k < nb; ++k)
                se2[k] = mag->abs_ybase[k] * mag->abs_ybase[k];
            sy = (double)pairwise_sum_f32(se2, (size_t)nb);
            if (se > sy) {
                /* form y_base = error_spec + sel_echo (complex add, f32). */
                for (k = 0; k < nb; ++k) {
                    p->e_out_spec[k].r = error_spec[k].r + echo_spec_sel[k].r;
                    p->e_out_spec[k].i = error_spec[k].i + echo_spec_sel[k].i;
                }
                yb = p->e_out_spec;
                out_base = yb;
            }
        }
        /* e_out_spec = out_base * gain (complex64 * f32). */
        for (k = 0; k < nb; ++k) {
            p->e_out_spec[k].r = out_base[k].r * gain[k];
            p->e_out_spec[k].i = out_base[k].i * gain[k];
        }
        (void)yb;
    }

    /* CNG injection. */
    if (c->enable_cng) {
        int n_random = nb - 2;
        uint32_t seed = p->cng_seed;
        /* N_float = sqrt(max(comfort_noise / _PSD_SCALE, 0)) in f32. */
        for (k = 0; k < nb; ++k) {
            float v = (float)(p->comfort_noise[k] / (float)PSD_SCALE);
            if (v < 0.0f) v = 0.0f;
            p->nf[k] = sqrtf(v);
        }
        /* For each random bin 1..nb-2 build CN, mul by noise_gain, add. DC and
         * Nyquist (k=0, k=nb-1) get CN=0 (cn_re/cn_im zero). noise_gain still
         * multiplies them but CN is 0, so they only get the gain-applied base. */
        for (k = 0; k < n_random; ++k) {
            uint32_t ix;
            int re_idx, im_idx, bin = k + 1;
            float ng, cn_re, cn_im;
            seed = (seed * 69069u + 1u) & 0x7FFFFFFFu;
            ix = seed >> 26;            /* top 5 bits, 0..31 */
            re_idx = (int)ix;
            im_idx = (int)((ix + 8u) & 31u);
            cn_re = p->nf[bin] * p->sqrt2_sin_lut[re_idx];
            cn_im = p->nf[bin] * p->sqrt2_sin_lut[im_idx];
            /* noise_gain = sqrt(max(1 - gain^2, 0)) in f32. */
            {
                float g2 = gain[bin] * gain[bin];
                float t = 1.0f - g2;
                if (t < 0.0f) t = 0.0f;
                ng = sqrtf(t);
            }
            p->e_out_spec[bin].r += ng * cn_re;
            p->e_out_spec[bin].i += ng * cn_im;
        }
        p->cng_seed = seed;
    }

    /* irfft (fp64-internal, numpy 1/N normalization) → e_out_full[fft_size]. */
    fft_inverse(p->fft, p->e_out_spec, p->e_out_full);

    /* windowed = e_out_full[:bs] * synth_window; OLA. */
    for (k = 0; k < bs; ++k) {
        float windowed = p->e_out_full[k] * p->synth_window[k];
        p->ola_buf[k] = p->ola_buf[k] + windowed;
    }
    for (k = 0; k < hop; ++k) out[k] = p->ola_buf[k];   /* out = ola[:hop] */
    /* shift: ola[:-hop] = ola[hop:]; ola[-hop:] = 0. */
    memmove(p->ola_buf, p->ola_buf + hop,
            (size_t)(bs - hop) * sizeof(float));
    memset(p->ola_buf + (bs - hop), 0, (size_t)hop * sizeof(float));

    (void)fft_size;
}
