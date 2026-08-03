/* aec3_post.c — C port of the AEC3 post-filter DRIVER (AEC._aec3_post).
 * WS5 Phase 5.5. See aec3_post.h for the stage breakdown + parity rules.
 *
 * Build: gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99
 *        -Ic_impl/include -Ic_impl/lib/pocketfft
 *   link: aec3_post.c reverb_model.c fft_pocketfft.c pocketfft.c -lm
 */
#include "aec3_post.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include "fast_math.h"
#include "aec_simd_kernels.h"
#include "aec3_scale.h"

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
 * recursive split) — matches np.sum over a float32 array. Delegates to the
 * shared kernel in simd_kernels.h (sk_pairwise_sum_f32 replicates this exact
 * tree verbatim — this file is in fact its documented reference source). */
static float pairwise_sum_f32(const float *a, size_t n) {
    return sk_pairwise_sum_f32(a, n);
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
    /* M4: default == block_size (the 16 kHz row's synth_window_len); the
     * caller (aec.c) overrides with the per-rate lookup value, same as
     * every other cfg field below block_size above. */
    cfg->synth_window_len = cfg->block_size;
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
    /* Convergence-flags y2 threshold (step 4 below): was a bare literal
     * 3.73e-4f = 50^2*64/32768^2, correct only at the legacy hop=160
     * grid but compared against y2_time summed over the LIVE hop_size.
     * Rescaled live via the same helper suppression_gain.c already uses
     * for this identical AEC3 constant (aec3_block_energy_scale is a pure
     * sample-count ratio -- hop_size only, no sample_rate). Mirrors
     * orchestrator.py's _y2_threshold. */
    p->y2_thr = aec3_block_energy_scale(50.0f * 50.0f * 64.0f, cfg->hop_size)
              / (32768.0f * 32768.0f);
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
    /* sk_sq_scale_f32(mag, scale, ...) computes out[i]=(mag[i]*mag[i])*scale
     * -- textually identical op sequence to this loop's body, and
     * (float)PSD_SCALE (2^30, exact in f32) folds to the same constant
     * regardless of call path. Same established bit-exact scale-multiply
     * pattern already used at lines 270/272 in this file and in
     * suppression_gain.c:112 (there with scale=1.0f). */
    sk_sq_scale_f32(mag, (float)PSD_SCALE, out, n);
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
    int nb = c->n_bins;
    if (!c->erle_coh_gate_enabled) return;
    /* Was 2 separate per-bin loops (EMA update, then threshold gate); fused
     * into sk_coherence_ema_gate_f32's single per-bin pass -- the gate loop
     * only ever reads coh_sye_re/im/syy/see at the SAME index k the EMA loop
     * just wrote, never a different index, so per-k fusion is order-
     * preserving (see the kernel's doc comment in simd_kernels.h). */
    sk_coherence_ema_gate_f32(p->coh_sye_re, p->coh_sye_im,
                              p->coh_syy, p->coh_see,
                              echo_spec_for_coh, near_spec,
                              mag->abs_echo_coh, mag->abs_near,
                              c->erle_coh_gate_alpha, c->erle_coh_gate_threshold,
                              p->coh_gate_mask, nb);
}

/* ── stage 5: CNG N2 tracking → comfort_noise ──────────────────────────── */
void aec3_post_compute_comfort_noise(Aec3Post *p, int saturated_capture) {
    const Aec3PostConfig *c = &p->cfg;
    int nb = c->n_bins;
    if (!p->noise_initialized) {
        /* straight copy -- memcpy is bit-exact by construction (no rounding
         * involved, so no NEON kernel is needed for this one). */
        memcpy(p->y2_smoothed, p->near_psd, (size_t)nb * sizeof(float));
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
        /* delta-form EMA: y2_smoothed += y2a*(near_psd-y2_smoothed) --
         * sk_ema_delta_f32 mirrors the source's separate sub/mul/add. */
        sk_ema_delta_f32(p->y2_smoothed, p->near_psd, y2a, nb);
        if (p->n2_counter > c->cng_n2_update_onset_hops) {
            /* data-dependent track/up select on n2 -- sk_n2_track_f32. */
            sk_n2_track_f32(p->n2, p->y2_smoothed, fresh, retain, g_up, nb);
        }
        if (p->n2_counter < dur) {
            p->n2_counter += 1;
            if (p->n2_counter < dur) {
                /* data-dependent slow/raw select on n2_initial --
                 * sk_n2_initial_track_f32. */
                sk_n2_initial_track_f32(p->n2_initial, p->n2, ia, nb);
            }
        }
        /* floor clamp: `if (x[k]<nfloor) x[k]=nfloor;` with no upper bound
         * at all -- sk_clip_f32(x, nfloor, +inf, n) reproduces this exactly:
         * `x[k] > +inf` is unreachable for any finite x (and false even for
         * x==+inf), so the clip's high branch degenerates to a no-op and
         * only the low-bound compare+select fires, verbatim. */
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
            /* |abs|^2 == (abs*abs)*1.0f bit-for-bit (multiplying by 1.0f is
             * an exact IEEE no-op, incl. sign of ±0.0f), so sk_sq_scale_f32
             * with scale=1.0f reproduces the source square verbatim.
             * pairwise_sum_f32 now delegates to sk_pairwise_sum_f32 (see its
             * definition above). */
            sk_sq_scale_f32(mag->abs_error, 1.0f, se2, nb);
            se = pairwise_sum_f32(se2, (size_t)nb);
            sk_sq_scale_f32(mag->abs_ybase, 1.0f, se2, nb);
            sy = pairwise_sum_f32(se2, (size_t)nb);
            if (se > sy) {
                sk_cadd_f32(p->e_out_spec, error_spec, echo_spec_sel, nb);
                out_base = p->e_out_spec;
            }
        }
        /* out_base may alias p->e_out_spec (the se>sy branch above just
         * wrote it there) -- sk_capply_gain_f32 explicitly supports
         * out == z in-place (full load before store, no cross-iteration
         * reuse of a not-yet-overwritten element), which is exactly this
         * case. */
        sk_capply_gain_f32(p->e_out_spec, out_base, gain, nb);
    }

    if (c->enable_cng) {
        int n_random = nb - 2;
        uint32_t seed = p->cng_seed;
        static const float inv_psd_scale = 1.0f / (float)PSD_SCALE;  /* 2^-30, exact */
        /* v = comfort_noise[k]*inv_psd_scale is pure float32*float32 --
         * inv_psd_scale is a compile-time-folded float32 constant (2^-30
         * exact); PSD_SCALE's double literal never touches the per-element
         * math. Split into: plain scalar multiply into p->nf as scratch
         * (provably dead here -- whatever nf held before, whether leftover
         * se2 scratch from the E2 block above or a previous hop's contents,
         * is never read again: every element is overwritten by this
         * multiply before anything reads nf back), then the floor clamp via
         * sk_clip_f32(v, 0, +inf) (the source has no upper bound either --
         * same +inf-degenerates-to-floor-only trick as the CNG N2 clamps
         * above), then a single batched sk_fast_sqrt_f32 in place. */
        for (k = 0; k < nb; ++k)
            p->nf[k] = p->comfort_noise[k] * inv_psd_scale;
        sk_clip_f32(p->nf, 0.0f, (float)INFINITY, nb);
        sk_fast_sqrt_f32(p->nf, p->nf, nb);
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

    /* M4 (multi-rate consumption switch): synth_window is caller-owned,
     * sized to c->synth_window_len (the per-rate lookup row's window
     * length) -- asserted equal to block_size right before the read below
     * (both come from the same Aec3BalancedRateDims row for a validated
     * sample rate, so this never fires for the {16000} whitelist: 320==320).
     * The win_n clamp is the release-path guard for a hypothetical NDEBUG
     * build that strips the assert: it degrades to a partial OLA window
     * instead of reading past whichever buffer is shorter, rather than an
     * OOB read. At every validated rate win_n == bs, so the loop below is
     * byte-identical to the unguarded version it replaces. */
    assert(c->synth_window_len == bs);
    {
        int win_n = (c->synth_window_len < bs) ? c->synth_window_len : bs;
        for (k = 0; k < win_n; ++k) {
            float windowed = p->e_out_full[k] * p->synth_window[k];
            p->ola_buf[k] = p->ola_buf[k] + windowed;
        }
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
 * pairwise_sum_f32 above; drift vs the numpy reference accepted). Delegates
 * to the shared kernel (sk_sum_sq_pairwise_f32 replicates this exact tree
 * verbatim — this file is its documented reference source). */
static float sum_sq_f32_pairwise(const float *a, size_t n) {
    return sk_sum_sq_pairwise_f32(a, n);
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
    int k;
    Aec3PostAbs mag;
    int saturated_capture = (in->saturation_level > 0.5f);
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
    /* E1 near_spec_win = error_spec_windowed + echo_spec (ORIGINAL); E2
     * y_base = sel_esw + sel_echo (SELECTED). Fissioned into 2 independent
     * elementwise complex adds -- disjoint destinations (nsw_e1/ybase are
     * separate caller-owned scratch buffers), so splitting the single
     * per-k loop into 2 full-range sk_cadd_f32 calls changes nothing:
     * sk_cadd_f32 reproduces `out[k].r = a[k].r+b[k].r; out[k].i =
     * a[k].i+b[k].i` expression-for-expression. */
    sk_cadd_f32(sc->nsw_e1, in->error_spec_windowed, in->echo_spec, nb);
    sk_cadd_f32(sc->ybase, sc->sel_esw, sc->sel_echo, nb);

    /* 7 independent np.abs magnitude arrays. Fissioned into 7 per-array
     * sk_cabs_np_f32 calls -- each out[k] = cabs_np(z[k].r, z[k].i)
     * elementwise with no loop-carried dependency across k, so splitting
     * the single loop into 7 full passes over disjoint destinations is
     * order-preserving; sk_cabs_np_f32's scalar reference is the same
     * ar/ai/larger/smaller/ratio/sqrtf(fmaf(...)) sequence as cabs_np()
     * above, verbatim. */
    sk_cabs_np_f32(in->near_spec, sc->abs_near, nb);
    sk_cabs_np_f32(in->far_spec, sc->abs_far, nb);
    sk_cabs_np_f32(sc->sel_echo, sc->abs_sel_echo, nb);
    sk_cabs_np_f32(sc->sel_esw, sc->abs_error, nb);
    sk_cabs_np_f32(in->echo_spec, sc->abs_echo_coh, nb);
    sk_cabs_np_f32(sc->nsw_e1, sc->abs_nsw_e1, nb);
    sk_cabs_np_f32(sc->ybase, sc->abs_ybase, nb);
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
    /* far_pwr = mean(far_end²) (pairwise f32; 3053). sk_sq_scale_f32 with
     * scale=1.0f is bit-exact to x*x (exact IEEE no-op multiply) -- same
     * established pattern as lines 270/272 above and suppression_gain.c:112. */
    {
        float *fsq = sc->nearend_pwr;   /* borrow scratch */
        sk_sq_scale_f32(in->far_end, 1.0f, fsq, hop);
        (void)fsq;
    }

    /* ── Step 4: convergence flags (3075-3106) ────────────────────────────── */
    {
        float y2_time = sum_sq_f32_pairwise(in->near_end, (size_t)hop);
        float e2_refined = sum_sq_f32_pairwise(in->raw_output, (size_t)hop);
        float y2_thr = p->y2_thr;
        float e2_coarse = 0.0f;
        int refined_conv, coarse_conv = 0;
        int aec3_converged;

        refined_conv = (e2_refined < 0.5f * y2_time) && (y2_time > y2_thr);
        if (in->shadow_present) {
            /* Parseval map: (2·Σ|E[1:-1]|² + |E[0]|² + |E[-1]|²)/fft_size,
             * cmag2_np per bin, f32 sums (Stage 3a; was f64 sums, 3095-3098). */
            const Complex *es = in->shadow_error_spec;
            float inner = 0.0f;
            /* (np.abs(c)²) per bin (cmag2_np), NOT er*er+ei*ei. Batched via
             * sk_cmag2_np_f32 into sc->x2_at_delay as scratch -- provably
             * dead here: within this call, x2_at_delay is next written by
             * Step 9 below (aec3_post_compute_x2_reverb consumes it
             * immediately after), and the previous hop's Step 9 already
             * consumed its former contents the same way before returning;
             * nothing reads x2_at_delay between that consume and this
             * point. The manual serial sum below keeps the exact k=1..nb-2
             * left-to-right accumulation order of the original loop. */
            sk_cmag2_np_f32(es + 1, sc->x2_at_delay, nb - 2);
            for (k = 0; k < nb - 2; ++k) inner += sc->x2_at_delay[k];
            {
                float m0 = cabs_np(es[0].r, es[0].i);
                float mn = cabs_np(es[nb - 1].r, es[nb - 1].i);
                e2_coarse = (2.0f * inner + m0 * m0 + mn * mn)
                          / (float)fft_size;
            }
            coarse_conv = (e2_coarse < 0.05f * y2_time) && (y2_time > y2_thr);
        }
        aec3_converged = refined_conv || coarse_conv;
        p->trace.aec3_converged = aec3_converged;   /* audio-passive trace stash */

        /* Step 5 (filter_state_bridge, formerly here) was dead code: every
         * output field of the FilterStateBridge it built (including the
         * unconditional per-hop IRFFT into what was then sc->bridge_taps) was
         * discarded -- the only downstream consumer, AecState.update, already
         * reads the live aec3_converged local directly (see Step 11 below),
         * which is bit-for-bit what bridge.filter_converged would have held
         * anyway (a pure passthrough in filter_state_bridge_build). Removed;
         * see filter_state_bridge_build itself (still exercised directly by
         * test/parity_filter_state_bridge.c) for that function's own
         * behaviour. The now-write-only-by-nobody sc->bridge_taps scratch
         * field ([fft_size] float, in Aec3PostRunScratch) was later removed
         * from the struct entirely (aec.c's pool-size accounting + carve, and
         * this driver's caller struct in aec3_post.h). */
        {
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
                    /* x2 = (np.abs(X)**2).astype(f32) = cmag2_np per bin.
                     * Fissioned into 2 independent elementwise cmag2 calls:
                     * disjoint destinations (x2_at_delay/x2_past), disjoint
                     * nb-contiguous sources (Xd/Xp are two different
                     * partition-slices of X_buf) -- no ordering dependency
                     * between them. */
                    sk_cmag2_np_f32(Xd, sc->x2_at_delay, nb);
                    sk_cmag2_np_f32(Xp, sc->x2_past, nb);
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
                /* _w_mag2 = |filter.W|² (cmag2_np), shape n_part × nb.
                 * in->W_all and sc->w_mag2 are verified single contiguous
                 * flat blocks with stride exactly nb at every call site
                 * (aec.c dynamic malloc+memcpy / static-arena carve of
                 * W_all and w_mag2; parity_aec3_post_run.c test harness) --
                 * so the (partition, bin) double loop collapses to one
                 * flat elementwise cmag2 pass over n_part*nb elements
                 * (cmag2_np_elem is per-index, not a reduction, so there is
                 * no reordering risk in merging the two loop levels). */
                sk_cmag2_np_f32(in->W_all, sc->w_mag2, n_part * nb);
                {
                    int delay_blocks =
                        aec_state_min_direct_path_filter_delay(obj->state);
                    int fq_valid = 0;
                    float filter_q =
                        aec_state_get_inst_linear_quality_estimate(
                            obj->state, &fq_valid);
                    int usable =
                        aec_state_usable_linear_estimate(obj->state);
                    ree_update_reverb_models(obj->ree, sc->w_mag2, n_part,
                                             delay_blocks, filter_q,
                                             /*fq_is_none=*/!fq_valid,
                                             in->stationary_block);

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
                                    /* r2/r2_unb are disjoint destinations
                                     * sharing the same mask + index range --
                                     * no ordering dependency between the two
                                     * calls, so each converts independently
                                     * to sk_mask_zero_f32 (verbatim
                                     * `if (mask[k]) x[k]=0.0f;`). */
                                    sk_mask_zero_f32(sc->r2, stat_mask, nb);
                                    sk_mask_zero_f32(sc->r2_unb, stat_mask, nb);
                                }
                            }

                            /* ── Step 18: nearend_pwr (3484-3488) ──────────── */
                            if (usable) {
                                /* np.minimum compares in float -- exact shape
                                 * of sk_min_f32: (a<b)?a:b. */
                                sk_min_f32(sc->nearend_pwr, p->error_psd,
                                          p->near_psd, nb);
                            } else {
                                memcpy(sc->nearend_pwr, p->near_psd,
                                      (size_t)nb * sizeof(float));
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
                                if (in->context_only) {
                                    /* The external joint NR/RES consumes E(f),
                                     * gain, R² and comfort_noise, then performs
                                     * its own CNG/IFFT/OLA. Preserve the run()
                                     * output contract by returning the linear
                                     * residual, without advancing the otherwise
                                     * private AEC synthesis OLA/CNG phase. */
                                    memcpy(out, in->raw_output,
                                           (size_t)hop * sizeof(float));
                                } else {
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
    }

    if (out_pending_gain_change) *out_pending_gain_change = pgc;
    if (out_pending_delay_change) *out_pending_delay_change = pdc;
    return 0;
}
