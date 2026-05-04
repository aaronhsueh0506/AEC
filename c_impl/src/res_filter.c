/* res_filter.c — ResFilter port (direct echo + ENR gain only).
 *
 * Numerical strategy:
 *   - PSD arrays stored fp32 (matches np.float32). Ops use fp32 to mirror
 *     Python broadcast semantics; scalar EMAs (far_activity, etc.) fp64.
 *   - User Fix 4 (final fullband noise gate at old aec.c:1419): NOT ported.
 *
 * NOTE: this is a faithful 1:1 port of the ENR/direct path. Unported branches:
 *   - echo_method='coherence'  (legacy)
 *   - gain_type='spectral_sub' (legacy)
 *   - gain_type='wiener'       (legacy)
 *   - _capture_stages diagnostic (debug-only)
 *   - _stats DT accumulators   (debug-only)
 */
#include "res_filter.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI_AEC
#define M_PI_AEC 3.14159265358979323846
#endif

static int next_pow2_at_least(int x) { int n = 1; while (n < x) n <<= 1; return n; }

/* Apply scalar config (not the buffer pointers). Used by both init paths. */
static void res_filter_apply_config(ResFilter* r, const ResFilterConfig* cfg) {
    r->block_size  = cfg->block_size;
    r->sample_rate = cfg->sample_rate;
    r->frame_size  = cfg->frame_size > 0 ? cfg->frame_size : cfg->block_size;
    r->hop_size    = cfg->hop_size   > 0 ? cfg->hop_size   : r->frame_size / 2;
    r->n_freqs     = cfg->n_freqs;
    r->g_min       = (float)pow(10.0, cfg->g_min_db / 20.0);
    r->over_sub    = cfg->over_sub;
    r->alpha       = cfg->alpha;
    r->ne_protect_db = cfg->ne_protect_db;
    r->alpha_echo_psd  = cfg->alpha_echo_psd;
    r->alpha_error_psd = cfg->alpha_error_psd;
    r->enr_scale       = cfg->enr_scale;
    r->startup_dt_min_ne_scale         = cfg->startup_dt_min_ne_scale;
    r->startup_dt_gain_floor           = cfg->startup_dt_gain_floor;
    r->startup_dt_noise_floor_scale    = cfg->startup_dt_noise_floor_scale;
    r->render_min_ne_factor            = cfg->render_min_ne_factor;
    r->render_dt_gain_ceil             = cfg->render_dt_gain_ceil;
    r->enable_reverb        = cfg->enable_reverb;
    r->enable_cng           = cfg->enable_cng;
    r->enable_spectral_floor = cfg->enable_spectral_floor;
    r->spectral_floor_ratio = (float)pow(10.0, cfg->spectral_floor_db / 20.0);
    r->reverb_decay         = cfg->reverb_decay;
    r->reverb_gain          = cfg->reverb_gain;
    r->alpha_coh      = 0.65f;
    r->alpha_noise    = 0.98f;
    r->max_drop_ratio = (float)pow(10.0, cfg->max_drop_db_per_frame / 20.0);
    r->max_rise_ratio = (float)pow(10.0, cfg->max_rise_db_per_frame / 20.0);
    r->alpha_envelope = 0.95f;
}

/* Compute per-bin masks + bin indices. Run after buffers are placed. */
static void res_filter_compute_tables(ResFilter* r, const ResFilterConfig* cfg) {
    int K = r->n_freqs;
    double freq_res = (double)cfg->sample_rate / (double)cfg->block_size;
    memset(r->stat_dt_mask, 0, (size_t)K * sizeof(float));
    memset(r->enr_blend,    0, (size_t)K * sizeof(float));
    for (int k = 0; k < K; ++k) {
        double f = k * freq_res;
        if (f >= 300.0 && f <= 3000.0) r->stat_dt_mask[k] = 0.8f;
        else if (f > 100.0 && f < 300.0) r->stat_dt_mask[k] = (float)(0.8 * (f - 100.0) / 200.0);
        else if (f > 3000.0 && f < 4000.0) r->stat_dt_mask[k] = (float)(0.8 * (4000.0 - f) / 1000.0);
        double bv = ((double)k - 5.0) / 5.0;
        if (bv < 0.0) bv = 0.0;
        if (bv > 1.0) bv = 1.0;
        r->enr_blend[k] = (float)bv;
    }
    int hf_cap = (int)(500.0 / freq_res);
    if (hf_cap > K - 1) hf_cap = K - 1;
    r->hf_cap_bin = hf_cap;
    r->harm_lf_start = (int)(100.0 / freq_res); if (r->harm_lf_start < 1) r->harm_lf_start = 1;
    r->harm_lf_end   = (int)(500.0 / freq_res); if (r->harm_lf_end > K - 1) r->harm_lf_end = K - 1;
    r->harm_hf_start = (int)(1000.0 / freq_res);
    r->harm_hf_end   = (int)(4000.0 / freq_res); if (r->harm_hf_end > K - 1) r->harm_hf_end = K - 1;

    /* sqrt-Hann window over frame_size: np.hanning uses (N-1) denom */
    for (int i = 0; i < r->frame_size; ++i) {
        double h = 0.5 * (1.0 - cos(2.0 * M_PI_AEC * (double)i / (double)(r->frame_size - 1)));
        r->window[i] = (float)sqrt(h);
    }
    for (int k = 0; k < K; ++k) r->gain_smooth[k] = r->g_min;
    memset(r->echo_psd, 0, (size_t)K * sizeof(float));
    memset(r->error_psd, 0, (size_t)K * sizeof(float));
    memset(r->S_fe, 0, (size_t)K * sizeof(Complex));
    memset(r->S_ff, 0, (size_t)K * sizeof(float));
    memset(r->S_ee, 0, (size_t)K * sizeof(float));
    memset(r->near_psd, 0, (size_t)K * sizeof(float));
    memset(r->near_psd_buf, 0, (size_t)4 * K * sizeof(float));
    r->near_psd_idx = 0;
    memset(r->reverb_psd, 0, (size_t)K * sizeof(float));
    memset(r->noise_psd, 0, (size_t)K * sizeof(float));
    r->noise_initialized = 0;
    memset(r->coh2_smooth, 0, (size_t)K * sizeof(float));
    for (int k = 0; k < K; ++k) r->error_envelope[k] = 1.0f;
    memset(r->input_buf, 0, (size_t)r->frame_size * sizeof(float));
    memset(r->ola_buf,   0, (size_t)r->frame_size * sizeof(float));
    memset(r->smooth_cn_gain, 0, (size_t)K * sizeof(float));
    r->far_activity = 0.0;
    r->nonlinear_frames = 0;
    r->last_effective_dt = 0.0;
    int fft_size = next_pow2_at_least(r->block_size);
    memset(r->time_scratch, 0, (size_t)fft_size * sizeof(float));
    memset(r->spec_synth, 0, (size_t)K * sizeof(Complex));
    memset(r->enhanced_spec, 0, (size_t)K * sizeof(Complex));
    r->cng_rng_state = 0xAEC0DEC0u;
}

void res_filter_init(ResFilter* r, const ResFilterConfig* cfg) {
    memset(r, 0, sizeof(*r));
    res_filter_apply_config(r, cfg);
    int K = r->n_freqs;
    int fft_size = next_pow2_at_least(r->block_size);

    r->stat_dt_mask = (float*)calloc((size_t)K, sizeof(float));
    r->enr_blend    = (float*)calloc((size_t)K, sizeof(float));
    r->window       = (float*)malloc((size_t)r->frame_size * sizeof(float));
    r->gain_smooth  = (float*)malloc((size_t)K * sizeof(float));
    r->echo_psd     = (float*)calloc((size_t)K, sizeof(float));
    r->error_psd    = (float*)calloc((size_t)K, sizeof(float));
    r->S_fe         = (Complex*)calloc((size_t)K, sizeof(Complex));
    r->S_ff         = (float*)calloc((size_t)K, sizeof(float));
    r->S_ee         = (float*)calloc((size_t)K, sizeof(float));
    r->near_psd     = (float*)calloc((size_t)K, sizeof(float));
    r->near_psd_buf = (float*)calloc((size_t)4 * K, sizeof(float));
    r->reverb_psd   = (float*)calloc((size_t)K, sizeof(float));
    r->noise_psd    = (float*)calloc((size_t)K, sizeof(float));
    r->coh2_smooth  = (float*)calloc((size_t)K, sizeof(float));
    r->error_envelope = (float*)malloc((size_t)K * sizeof(float));
    r->input_buf    = (float*)calloc((size_t)r->frame_size, sizeof(float));
    r->ola_buf      = (float*)calloc((size_t)r->frame_size, sizeof(float));
    r->smooth_cn_gain = (float*)calloc((size_t)K, sizeof(float));
    filter_erle_init(&r->filter_erle, K);
    fullband_erle_init(&r->fb_erle);
    residual_echo_init(&r->residual_est, K);
    r->fft = fft_create(fft_size);
    r->time_scratch  = (float*)calloc((size_t)fft_size, sizeof(float));
    r->spec_synth    = (Complex*)calloc((size_t)K, sizeof(Complex));
    r->enhanced_spec = (Complex*)calloc((size_t)K, sizeof(Complex));

    res_filter_compute_tables(r, cfg);
    r->is_static = 0;
}

size_t res_filter_get_mem_size(const ResFilterConfig* cfg) {
    if (!cfg) return 0;
    int K   = cfg->n_freqs;
    int fr  = cfg->frame_size > 0 ? cfg->frame_size : cfg->block_size;
    int fft = next_pow2_at_least(cfg->block_size);
    if (K <= 0 || fr <= 0 || fft <= 0) return 0;
    size_t total = 0;
    total += ALIGN16((size_t)K * sizeof(float));      /* stat_dt_mask */
    total += ALIGN16((size_t)K * sizeof(float));      /* enr_blend */
    total += ALIGN16((size_t)fr * sizeof(float));     /* window */
    total += ALIGN16((size_t)K * sizeof(float));      /* gain_smooth */
    total += ALIGN16((size_t)K * sizeof(float));      /* echo_psd */
    total += ALIGN16((size_t)K * sizeof(float));      /* error_psd */
    total += ALIGN16((size_t)K * sizeof(Complex));    /* S_fe */
    total += ALIGN16((size_t)K * sizeof(float));      /* S_ff */
    total += ALIGN16((size_t)K * sizeof(float));      /* S_ee */
    total += ALIGN16((size_t)K * sizeof(float));      /* near_psd */
    total += ALIGN16((size_t)4 * K * sizeof(float));  /* near_psd_buf */
    total += ALIGN16((size_t)K * sizeof(float));      /* reverb_psd */
    total += ALIGN16((size_t)K * sizeof(float));      /* noise_psd */
    total += ALIGN16((size_t)K * sizeof(float));      /* coh2_smooth */
    total += ALIGN16((size_t)K * sizeof(float));      /* error_envelope */
    total += ALIGN16((size_t)fr * sizeof(float));     /* input_buf */
    total += ALIGN16((size_t)fr * sizeof(float));     /* ola_buf */
    total += ALIGN16((size_t)K * sizeof(float));      /* smooth_cn_gain */
    total += filter_erle_get_mem_size(K);             /* nested erle */
    total += fft_get_mem_size(fft);                   /* nested fft */
    total += ALIGN16((size_t)fft * sizeof(float));    /* time_scratch */
    total += ALIGN16((size_t)K * sizeof(Complex));    /* spec_synth */
    total += ALIGN16((size_t)K * sizeof(Complex));    /* enhanced_spec */
    return total;
}

void res_filter_init_static(ResFilter* r, void* mem, size_t mem_size,
                             const ResFilterConfig* cfg) {
    if (!r || !mem || !cfg) return;
    if (mem_size < res_filter_get_mem_size(cfg)) return;

    memset(r, 0, sizeof(*r));
    res_filter_apply_config(r, cfg);
    int K   = r->n_freqs;
    int fr  = r->frame_size;
    int fft = next_pow2_at_least(r->block_size);

    uint8_t* ptr = (uint8_t*)mem;
    r->stat_dt_mask = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->enr_blend    = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->window       = (float*)ptr; ptr += ALIGN16((size_t)fr * sizeof(float));
    r->gain_smooth  = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->echo_psd     = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->error_psd    = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->S_fe         = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    r->S_ff         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->S_ee         = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->near_psd     = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->near_psd_buf = (float*)ptr; ptr += ALIGN16((size_t)4 * K * sizeof(float));
    r->reverb_psd   = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->noise_psd    = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->coh2_smooth  = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->error_envelope = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    r->input_buf    = (float*)ptr; ptr += ALIGN16((size_t)fr * sizeof(float));
    r->ola_buf      = (float*)ptr; ptr += ALIGN16((size_t)fr * sizeof(float));
    r->smooth_cn_gain = (float*)ptr; ptr += ALIGN16((size_t)K * sizeof(float));
    size_t erle_sz = filter_erle_get_mem_size(K);
    filter_erle_init_static(&r->filter_erle, ptr, erle_sz, K); ptr += erle_sz;
    fullband_erle_init(&r->fb_erle);
    residual_echo_init(&r->residual_est, K);
    size_t fft_sz = fft_get_mem_size(fft);
    r->fft = fft_init(ptr, fft_sz, fft); ptr += fft_sz;
    r->time_scratch  = (float*)ptr; ptr += ALIGN16((size_t)fft * sizeof(float));
    r->spec_synth    = (Complex*)ptr; ptr += ALIGN16((size_t)K * sizeof(Complex));
    r->enhanced_spec = (Complex*)ptr; /* last */

    res_filter_compute_tables(r, cfg);
    r->is_static = 1;
}

void res_filter_free(ResFilter* r) {
    if (!r) return;
    if (r->is_static) {
        filter_erle_free(&r->filter_erle);  /* no-op when nested static */
        if (r->fft) fft_destroy(r->fft);    /* no-op when nested static */
        return;
    }
    free(r->stat_dt_mask); free(r->enr_blend);
    free(r->window);
    free(r->gain_smooth); free(r->echo_psd); free(r->error_psd);
    free(r->S_fe); free(r->S_ff); free(r->S_ee);
    free(r->near_psd); free(r->near_psd_buf);
    free(r->reverb_psd); free(r->noise_psd);
    free(r->coh2_smooth); free(r->error_envelope);
    free(r->input_buf); free(r->ola_buf); free(r->smooth_cn_gain);
    filter_erle_free(&r->filter_erle);
    free(r->time_scratch); free(r->spec_synth); free(r->enhanced_spec);
    if (r->fft) fft_destroy(r->fft);
}

void res_filter_reset(ResFilter* r) {
    int K = r->n_freqs;
    for (int k = 0; k < K; ++k) r->gain_smooth[k] = r->g_min;
    memset(r->echo_psd, 0, (size_t)K * sizeof(float));
    memset(r->error_psd, 0, (size_t)K * sizeof(float));
    memset(r->S_fe, 0, (size_t)K * sizeof(Complex));
    memset(r->S_ff, 0, (size_t)K * sizeof(float));
    memset(r->S_ee, 0, (size_t)K * sizeof(float));
    memset(r->near_psd, 0, (size_t)K * sizeof(float));
    memset(r->near_psd_buf, 0, (size_t)4 * K * sizeof(float));
    r->near_psd_idx = 0;
    memset(r->reverb_psd, 0, (size_t)K * sizeof(float));
    memset(r->noise_psd, 0, (size_t)K * sizeof(float));
    r->noise_initialized = 0;
    memset(r->coh2_smooth, 0, (size_t)K * sizeof(float));
    for (int k = 0; k < K; ++k) r->error_envelope[k] = 1.0f;
    memset(r->input_buf, 0, (size_t)r->frame_size * sizeof(float));
    memset(r->ola_buf, 0, (size_t)r->frame_size * sizeof(float));
    memset(r->smooth_cn_gain, 0, (size_t)K * sizeof(float));
    r->far_activity = 0.0;
    r->nonlinear_frames = 0;
    residual_echo_reset(&r->residual_est);
    filter_erle_reset(&r->filter_erle);
    fullband_erle_reset(&r->fb_erle);
}

/* Helper: pairwise-mean of fp32 array (matches numpy np.mean for fp32) */
static double mean_f32(const float* a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)a[i];
    return s / (double)n;
}

/* Box-Muller-ish randn via xorshift32 + Marsaglia polar — compact. */
static float u01(unsigned int* st) {
    unsigned int x = *st;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *st = x;
    return (float)((x & 0x00FFFFFFu) / (float)0x01000000u);
}
static float randn(unsigned int* st) {
    /* Marsaglia polar method */
    float u, v, s;
    do {
        u = 2.0f * u01(st) - 1.0f;
        v = 2.0f * u01(st) - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    s = sqrtf(-2.0f * logf(s) / s);
    return u * s;
}

void res_filter_process(ResFilter* r, const ResProcessInputs* in,
                           float* output_hop) {
    const int hop = r->hop_size;
    const int frame = r->frame_size;
    const int K = r->n_freqs;
    const float eps = 1e-10f;

    /* Slide input_buf */
    memmove(r->input_buf, r->input_buf + hop, (size_t)(frame - hop) * sizeof(float));
    memcpy(r->input_buf + (frame - hop), in->error_hop, (size_t)hop * sizeof(float));

    /* Windowed analysis FFT (zero-padded to block_size) */
    for (int i = 0; i < frame; ++i)
        r->time_scratch[i] = r->input_buf[i] * r->window[i];
    for (int i = frame; i < r->block_size; ++i) r->time_scratch[i] = 0.0f;
    fft_forward(r->fft, r->time_scratch, r->spec_synth);

    const Complex* spec = (in->error_spec_from_filter)
                            ? in->error_spec_from_filter
                            : r->spec_synth;

    /* Power spectra */
    float echo_pwr_linear[8192];
    float error_pwr[8192];
    for (int k = 0; k < K; ++k) {
        float er = in->echo_spec[k].r, ei = in->echo_spec[k].i;
        echo_pwr_linear[k] = er * er + ei * ei;
        float sr = spec[k].r, si = spec[k].i;
        error_pwr[k] = sr * sr + si * si;
    }

    /* Coherence */
    float coh2[8192];
    for (int k = 0; k < K; ++k) coh2[k] = 0.0f;
    if (in->far_spec && in->far_power > 1e-4) {
        float a = r->alpha_coh, oa = 1.0f - a;
        for (int k = 0; k < K; ++k) {
            float fr = in->far_spec[k].r, fi = in->far_spec[k].i;
            float sr = spec[k].r, si = spec[k].i;
            /* spec * conj(far) = (sr*fr + si*fi) + j*(si*fr - sr*fi) */
            float cr = sr * fr + si * fi;
            float ci = si * fr - sr * fi;
            r->S_fe[k].r = a * r->S_fe[k].r + oa * cr;
            r->S_fe[k].i = a * r->S_fe[k].i + oa * ci;
            r->S_ff[k]   = a * r->S_ff[k] + oa * (fr * fr + fi * fi);
            r->S_ee[k]   = a * r->S_ee[k] + oa * error_pwr[k];
        }
        float a_rise = in->filter_converged ? 0.90f : 0.80f;
        float a_drop = 0.50f;
        for (int k = 0; k < K; ++k) {
            float sxr = r->S_fe[k].r, sxi = r->S_fe[k].i;
            float num = sxr * sxr + sxi * sxi;
            float den = r->S_ff[k] * r->S_ee[k] + 1e-10f;
            float c2  = num / den;
            if (c2 > 1.0f) c2 = 1.0f;
            float ac = (c2 < r->coh2_smooth[k]) ? a_drop : a_rise;
            r->coh2_smooth[k] = ac * r->coh2_smooth[k] + (1.0f - ac) * c2;
            coh2[k] = r->coh2_smooth[k];
        }
    } else if (in->far_power <= 1e-4) {
        for (int k = 0; k < K; ++k) {
            r->S_fe[k].r *= 0.5f; r->S_fe[k].i *= 0.5f;
            r->S_ff[k] *= 0.5f; r->S_ee[k] *= 0.5f;
        }
    }

    /* Cold start of echo_psd / error_psd on first far-active frame */
    double echo_sum = 0.0;
    for (int k = 0; k < K; ++k) echo_sum += (double)r->echo_psd[k];
    if (in->far_power > 1e-4 && echo_sum < 1e-10) {
        memcpy(r->echo_psd, echo_pwr_linear, (size_t)K * sizeof(float));
        memcpy(r->error_psd, error_pwr, (size_t)K * sizeof(float));
    }
    float aE = r->alpha_echo_psd, oE = 1.0f - aE;
    float aR = r->alpha_error_psd, oR = 1.0f - aR;
    for (int k = 0; k < K; ++k) {
        r->echo_psd[k]  = aE * r->echo_psd[k]  + oE * echo_pwr_linear[k];
        r->error_psd[k] = aR * r->error_psd[k] + oR * error_pwr[k];
    }

    /* Multi-ERLE update */
    int far_active = in->far_power > 1e-4;
    {
        float echo_re[8192], echo_im[8192], err_re[8192], err_im[8192];
        for (int k = 0; k < K; ++k) {
            echo_re[k] = in->echo_spec[k].r;
            echo_im[k] = in->echo_spec[k].i;
            err_re[k]  = spec[k].r;
            err_im[k]  = spec[k].i;
        }
        filter_erle_update(&r->filter_erle, echo_re, echo_im, err_re, err_im,
                              far_active, in->dt_indicator);
    }
    double near_power_broad = (in->near_spec) ? mean_f32(r->near_psd, K) : 0.0;
    double error_power_broad = mean_f32(error_pwr, K);
    fullband_erle_update(&r->fb_erle, near_power_broad, error_power_broad,
                            far_active, in->dt_indicator);

    if (in->far_power < 1e-4) {
        for (int k = 0; k < K; ++k) r->echo_psd[k] *= 0.3f;
    }

    /* Dynamic far_activity */
    double is_far_active = (in->far_power > 1e-4) ? 1.0 : 0.0;
    if (is_far_active > r->far_activity) {
        r->far_activity = 0.7 * r->far_activity + 0.3 * is_far_active;
    } else {
        r->far_activity = 0.98 * r->far_activity + 0.02 * is_far_active;
    }
    float effective_g_min = r->g_min;

    /* Quiet mask */
    double err_mean = mean_f32(r->error_psd, K);
    float signal_floor = (float)(err_mean * 0.001 + 1e-8);

    /* Stationary-DT virtual indicator */
    double dt_for_fs = in->is_stationary_dt ? 0.8 : in->dt_indicator;
    double effective_dt = (dt_for_fs > in->shadow_dt) ? dt_for_fs : in->shadow_dt;
    r->last_effective_dt = effective_dt;
    int epc_dt = in->epc_active && (effective_dt > 0.35);

    /* === Residual echo PSD (direct method) === */
    float residual_echo_psd[8192];
    if (in->near_spec) {
        /* near_pwr from near_spec */
        float near_pwr[8192];
        for (int k = 0; k < K; ++k) {
            float nr = in->near_spec[k].r, ni = in->near_spec[k].i;
            near_pwr[k] = nr * nr + ni * ni;
        }
        /* Update 4-block near_psd_buf circular */
        memcpy(r->near_psd_buf + (size_t)r->near_psd_idx * K, near_pwr,
               (size_t)K * sizeof(float));
        r->near_psd_idx = (r->near_psd_idx + 1) % 4;
        for (int k = 0; k < K; ++k) {
            float s = 0.0f;
            for (int b = 0; b < 4; ++b) s += r->near_psd_buf[(size_t)b * K + k];
            r->near_psd[k] = s * 0.25f;
        }

        /* Stage1+2 attribution via ResidualEchoEstimator */
        residual_echo_attribute_legacy(
            &r->residual_est,
            r->echo_psd, r->error_psd, coh2,
            in->far_spec,
            in->far_power, in->erle_factor, dt_for_fs, r->far_activity,
            in->epc_active, in->saturation_level, in->filter_converged,
            in->erl_estimate,
            r->filter_erle.erle, r->fb_erle.fb_erle,
            residual_echo_psd);

        /* Nonlinear hangover */
        if (in->saturation_level > 0.3) r->nonlinear_frames++;
        else if (r->nonlinear_frames > 0) r->nonlinear_frames--;
        int is_nonlinear = r->nonlinear_frames > 5;
        if (is_nonlinear && in->far_power > 1e-4) {
            float boost = (float)(1.0 + 1.0 * in->saturation_level);
            for (int k = 0; k < K; ++k) residual_echo_psd[k] *= boost;
        }
        /* Harmonic distortion mapping */
        if (in->saturation_level > 0.05 && in->far_power > 1e-4) {
            int lf0 = r->harm_lf_start, lf1 = r->harm_lf_end;
            int hf0 = r->harm_hf_start, hf1 = r->harm_hf_end;
            if (lf1 > lf0 && hf1 > hf0) {
                double sum = 0.0;
                for (int k = lf0; k < lf1; ++k) sum += residual_echo_psd[k];
                float lf_mean = (float)(sum / (lf1 - lf0));
                float dist = (float)(0.1 + 0.4 * in->saturation_level);
                float floor_v = lf_mean * dist;
                for (int k = hf0; k < hf1; ++k) {
                    if (residual_echo_psd[k] < floor_v) residual_echo_psd[k] = floor_v;
                }
            }
        }
        /* Cap cascade */
        int using_render = r->residual_est.using_render_based;
        if (!using_render) {
            for (int k = 0; k < K; ++k) {
                float c = r->echo_psd[k] * 2.0f;
                if (residual_echo_psd[k] > c) residual_echo_psd[k] = c;
            }
        }
        float err_cap_mult = using_render ? 1.5f : 1.0f;
        for (int k = 0; k < K; ++k) {
            float c = r->error_psd[k] * err_cap_mult;
            if (residual_echo_psd[k] > c) residual_echo_psd[k] = c;
        }
        if (!using_render) {
            float dt2 = (float)(dt_for_fs * dt_for_fs);
            float dts = 1.0f - dt2;
            if (dts < 0.1f) dts = 0.1f;
            if (dts > 1.0f) dts = 1.0f;
            for (int k = 0; k < K; ++k) {
                float c = r->error_psd[k] * dts;
                if (residual_echo_psd[k] > c) residual_echo_psd[k] = c;
            }
        }
        if (in->far_spec && in->far_power > 1e-4 && in->erl_estimate > 0.0) {
            double erl_mul = in->erl_estimate * 2.0;
            if (erl_mul > 1.0) erl_mul = 1.0;
            if (!using_render) {
                for (int k = 0; k < K; ++k) {
                    float fr = in->far_spec[k].r, fi = in->far_spec[k].i;
                    float far_psd_k = fr * fr + fi * fi;
                    float ceil = (float)(far_psd_k * erl_mul);
                    if (residual_echo_psd[k] > ceil) residual_echo_psd[k] = ceil;
                }
            }
        }
        /* Reverb tail */
        if (r->enable_reverb) {
            for (int k = 0; k < K; ++k) {
                float far_psd_k;
                if (in->far_spec) {
                    float fr = in->far_spec[k].r, fi = in->far_spec[k].i;
                    far_psd_k = fr * fr + fi * fi;
                } else {
                    far_psd_k = echo_pwr_linear[k];
                }
                r->reverb_psd[k] = r->reverb_decay * r->reverb_psd[k]
                                 + (1.0f - r->reverb_decay) * far_psd_k;
            }
            if (!in->is_stationary_dt) {
                double ne_reverb_factor = 0.7 + 0.3 * r->far_activity * (1.0 - dt_for_fs);
                double reverb_gate = r->far_activity * ne_reverb_factor;
                if (r->far_activity < 0.1) reverb_gate = 0.0;
                for (int k = 0; k < K; ++k) {
                    residual_echo_psd[k] += (float)(r->reverb_gain * r->reverb_psd[k] * reverb_gate);
                }
            }
        }
    } else {
        /* No near_spec → cannot use direct path. Set residual = echo_psd as a
         * conservative fallback (matches Python when echo_method not "direct";
         * we skip the gain branch below similarly). */
        memcpy(residual_echo_psd, r->echo_psd, (size_t)K * sizeof(float));
    }

    /* Per-bin echo boost */
    if (in->far_power > 1e-4 && in->erle_factor > 0.3) {
        if (dt_for_fs < 0.2) {
            for (int k = 0; k < K; ++k) {
                float boost = 1.0f + 0.5f * coh2[k];
                residual_echo_psd[k] *= boost;
            }
        }
    }

    /* Spectral floor */
    float spectral_g_min[8192];
    if (r->enable_spectral_floor && in->far_power > 1e-4) {
        float env_max = 1e-10f;
        for (int k = 0; k < K; ++k) {
            float em = sqrtf(error_pwr[k] + 1e-10f);
            r->error_envelope[k] = r->alpha_envelope * r->error_envelope[k]
                                + (1.0f - r->alpha_envelope) * em;
            if (r->error_envelope[k] > env_max) env_max = r->error_envelope[k];
        }
        env_max += 1e-10f;
        for (int k = 0; k < K; ++k) {
            float env_norm = r->error_envelope[k] / env_max;
            float v = effective_g_min + (1.0f - effective_g_min) * env_norm * r->spectral_floor_ratio;
            if (v < effective_g_min) v = effective_g_min;
            spectral_g_min[k] = v;
        }
    } else {
        for (int k = 0; k < K; ++k) spectral_g_min[k] = effective_g_min;
    }

    double fs_confidence = r->far_activity * pow(1.0 - effective_dt, 2.0);

    /* NE protection */
    double ne_erle_gate = (in->erle_factor > 0.3) ? in->erle_factor : 0.3;
    double ne_g_min_ceil = pow(10.0, r->ne_protect_db / 20.0);
    for (int k = 0; k < K; ++k) {
        double ne_protection = (1.0 - coh2[k]) * ne_erle_gate * (1.0 - fs_confidence);
        double ne_g_floor = effective_g_min + (ne_g_min_ceil - effective_g_min) * ne_protection;
        if (ne_g_floor < effective_g_min) ne_g_floor = effective_g_min;
        if (spectral_g_min[k] < ne_g_floor) spectral_g_min[k] = (float)ne_g_floor;
    }

    int startup_dt_curr = (effective_dt > 0.35) && (in->far_power > 1e-4) && !in->filter_converged;
    if (startup_dt_curr && r->startup_dt_gain_floor < 1.0f) {
        for (int k = 0; k < K; ++k) {
            if (spectral_g_min[k] > r->startup_dt_gain_floor) spectral_g_min[k] = r->startup_dt_gain_floor;
        }
    }

    /* === Gain (ENR path) === */
    float g[8192];
    if (in->near_spec) {
        float raw_nearend[8192];
        float dt_per_bin[8192];
        for (int k = 0; k < K; ++k) {
            float v = r->error_psd[k] - residual_echo_psd[k];
            if (v < 0.0f) v = 0.0f;
            raw_nearend[k] = v;
            float a = (float)effective_dt;
            float b = 1.0f - coh2[k];
            float dp = a > b ? a : b;
            dt_per_bin[k] = dp;
        }
        if (in->is_stationary_dt) {
            for (int k = 0; k < K; ++k) {
                if (r->stat_dt_mask[k] > dt_per_bin[k]) dt_per_bin[k] = r->stat_dt_mask[k];
            }
        }
        float dt_shaped[8192];
        for (int k = 0; k < K; ++k) {
            dt_shaped[k] = (float)pow(dt_per_bin[k], 1.1);
        }
        float noise_floor_psd_scalar = (float)(mean_f32(r->error_psd, K) * 0.01 + 1e-10);
        float nearend_est[8192];
        float min_ne_from_dt[8192];
        int startup_dt_cond = (effective_dt > 0.35) && (in->far_power > 1e-4) && !in->filter_once_converged;
        for (int k = 0; k < K; ++k) {
            float ne = raw_nearend[k] * dt_shaped[k];
            if (ne < noise_floor_psd_scalar) ne = noise_floor_psd_scalar;
            nearend_est[k] = ne;
            float m = r->error_psd[k] * dt_shaped[k];
            if (startup_dt_cond && r->startup_dt_min_ne_scale != 1.0f) m *= r->startup_dt_min_ne_scale;
            if (r->residual_est.using_render_based) m *= r->render_min_ne_factor;
            min_ne_from_dt[k] = m;
            if (nearend_est[k] < min_ne_from_dt[k]) nearend_est[k] = min_ne_from_dt[k];
            float phys = r->error_psd[k] * 0.05f;
            if (nearend_est[k] < phys) nearend_est[k] = phys;
        }

        /* ENR */
        float enr[8192];
        for (int k = 0; k < K; ++k) enr[k] = residual_echo_psd[k] / (nearend_est[k] + 1e-10f);

        const float *blend_arr = r->enr_blend;
        float scale = r->enr_scale;
        float dt_enr_relax = 1.0f;
        if (effective_dt > 0.4) {
            dt_enr_relax = (float)(1.0 + (effective_dt - 0.4) / 0.6 * 0.5);
        }
        for (int k = 0; k < K; ++k) {
            float blend_k = blend_arr[k];
            float enr_t_ne = (1.0f - blend_k) * 2.0f + blend_k * 1.5f;
            float enr_s_ne = (1.0f - blend_k) * 3.0f + blend_k * 2.5f;
            float enr_t_fs = (1.0f - blend_k) * (0.3f * scale) + blend_k * (0.07f * scale);
            float enr_s_fs = (1.0f - blend_k) * (0.4f * scale) + blend_k * (0.1f * scale);
            enr_t_ne *= dt_enr_relax;
            enr_s_ne *= dt_enr_relax;
            float ne_conf = dt_per_bin[k];
            float enr_t = ne_conf * enr_t_ne + (1.0f - ne_conf) * enr_t_fs;
            float enr_s = ne_conf * enr_s_ne + (1.0f - ne_conf) * enr_s_fs;
            float enr_s_safe = enr_t + 0.2f;
            if (enr_s > enr_s_safe) enr_s_safe = enr_s;
            float gv;
            if (enr[k] > enr_t) {
                float num = enr_s_safe - enr[k];
                float den = enr_s_safe - enr_t + eps;
                gv = num / den;
                if (gv < 0.0f) gv = 0.0f;
                if (gv > 1.0f) gv = 1.0f;
            } else {
                gv = 1.0f;
            }
            g[k] = gv;
        }

        /* EMR */
        double npsd_sum = 0.0;
        for (int k = 0; k < K; ++k) npsd_sum += r->noise_psd[k];
        if (npsd_sum > 0.0) {
            const float emr_transp = 0.3f;
            for (int k = 0; k < K; ++k) {
                float emr = residual_echo_psd[k] / (r->noise_psd[k] + 1e-10f);
                float gem = emr_transp / (emr + 1e-10f);
                if (gem < 0.0f) gem = 0.0f;
                if (gem > 1.0f) gem = 1.0f;
                if (gem > g[k]) g[k] = gem;
            }
        }
        for (int k = 0; k < K; ++k) {
            if (g[k] < spectral_g_min[k]) g[k] = spectral_g_min[k];
        }
    } else {
        /* ENR path requires near_spec — fallback: pass-through gain */
        for (int k = 0; k < K; ++k) g[k] = 1.0f;
    }

    if (epc_dt) {
        for (int k = 0; k < K; ++k) if (g[k] > 0.85f) g[k] = 0.85f;
    }
    /* Quiet mask */
    for (int k = 0; k < K; ++k) {
        if (r->echo_psd[k] < signal_floor && r->error_psd[k] < signal_floor) g[k] = 1.0f;
    }
    /* Postprocess */
    if (in->far_power > 1e-4) {
        /* 3-bin smooth */
        float gtmp[8192];
        for (int k = 0; k < K; ++k) {
            float left  = (k > 0)     ? g[k - 1] : 0.0f;
            float mid   = g[k];
            float right = (k < K - 1) ? g[k + 1] : 0.0f;
            gtmp[k] = 0.25f * left + 0.5f * mid + 0.25f * right;
        }
        memcpy(g, gtmp, (size_t)K * sizeof(float));
        if (K > 2) {
            float bcap = (g[1] < g[2]) ? g[1] : g[2];
            g[0] = bcap; g[1] = bcap;
        }
        if (K > r->hf_cap_bin + 1 && effective_dt < 0.5 && !in->is_stationary_dt) {
            float hf_cap = g[r->hf_cap_bin];
            for (int k = r->hf_cap_bin + 1; k < K; ++k) {
                if (g[k] > hf_cap) g[k] = hf_cap;
            }
        }
    }

    if (in->divergence > 0.3) {
        float dgain = (float)(0.01 + (1.0 - 0.01) * (1.0 - in->divergence));
        for (int k = 0; k < K; ++k) if (g[k] > dgain) g[k] = dgain;
    }

    /* Temporal smoothing */
    double dt_temporal = in->is_stationary_dt ? 0.8 :
                          ((in->dt_indicator > effective_dt * 0.5)
                           ? in->dt_indicator : effective_dt * 0.5);
    float alpha_fast = (float)(0.3 + 0.2 * (1.0 - in->erle_factor));
    float alpha_slow = (float)(0.85 + 0.1 * (1.0 - in->erle_factor));
    float alpha_attack = (float)(alpha_slow + (alpha_fast - alpha_slow) * fs_confidence);
    if (in->is_stationary_dt) {
        double f = 1.0 - dt_temporal * dt_temporal;
        if (f < 0.1) f = 0.1; if (f > 1.0) f = 1.0;
        alpha_attack = (float)(alpha_attack * f);
    }
    float alpha_release_light = (float)(0.5 - 0.2 * dt_temporal);
    float smoothed[8192];
    for (int k = 0; k < K; ++k) {
        if (g[k] < r->gain_smooth[k]) {
            smoothed[k] = alpha_attack * r->gain_smooth[k] + (1.0f - alpha_attack) * g[k];
        } else {
            smoothed[k] = alpha_release_light * r->gain_smooth[k] + (1.0f - alpha_release_light) * g[k];
        }
    }

    /* Rate limiting */
    float activity_scale = (float)(0.5 + 0.5 * r->far_activity);
    float eff_drop = (float)pow(r->max_drop_ratio, activity_scale);
    float rise_exp = (float)(0.5 + 0.5 * (1.0 - r->far_activity));
    if (dt_temporal > 0.3) {
        float boost = (float)(1.0 + dt_temporal);
        rise_exp /= boost;
    }
    float eff_rise = (float)pow(r->max_rise_ratio, rise_exp);

    float gain_floor[8192], gain_ceil[8192];
    for (int k = 0; k < K; ++k) {
        gain_floor[k] = r->gain_smooth[k] / eff_drop;
        gain_ceil[k]  = r->gain_smooth[k] * eff_rise;
    }
    if (fs_confidence < 0.9) {
        int lf_limit = K < 8 ? K : 8;
        double lf_factor = 0.25 * (1.0 - fs_confidence * 2.0);
        if (lf_factor < 0.0) lf_factor = 0.0;
        if (lf_factor > 0.01) {
            for (int k = 0; k < lf_limit; ++k) {
                float v = (float)(r->gain_smooth[k] * lf_factor);
                if (gain_floor[k] < v) gain_floor[k] = v;
            }
        }
    }
    for (int k = 0; k < K; ++k) {
        if (smoothed[k] < gain_floor[k]) smoothed[k] = gain_floor[k];
        if (smoothed[k] > gain_ceil[k])  smoothed[k] = gain_ceil[k];
        if (smoothed[k] < spectral_g_min[k]) smoothed[k] = spectral_g_min[k];
        if (smoothed[k] > 1.0f) smoothed[k] = 1.0f;
    }
    if (r->residual_est.using_render_based && r->far_activity > 0.3) {
        for (int k = 0; k < K; ++k) {
            if (smoothed[k] > r->render_dt_gain_ceil) smoothed[k] = r->render_dt_gain_ceil;
        }
    }
    memcpy(r->gain_smooth, smoothed, (size_t)K * sizeof(float));

    /* Min-stat noise tracker */
    if (!r->noise_initialized) {
        for (int k = 0; k < K; ++k) r->noise_psd[k] = r->error_psd[k] + 1e-8f;
        r->noise_initialized = 1;
    }
    int is_learning_safe = (r->far_activity < 0.01) && (in->dt_indicator < 0.1);
    float a_down = 0.98f;
    float a_up   = is_learning_safe ? 0.998f : 1.0f;
    for (int k = 0; k < K; ++k) {
        float an = (r->error_psd[k] > r->noise_psd[k]) ? a_up : a_down;
        r->noise_psd[k] = an * r->noise_psd[k] + (1.0f - an) * r->error_psd[k];
    }

    /* Dynamic noise floor on gain */
    float noise_floor_gain[8192];
    for (int k = 0; k < K; ++k) {
        float sr = r->spec_synth[k].r, si = r->spec_synth[k].i;
        float spec_pwr = sr * sr + si * si + 1e-10f;
        float v = sqrtf(r->noise_psd[k] / spec_pwr);
        if (v < effective_g_min) v = effective_g_min;
        if (v > 1.0f)              v = 1.0f;
        noise_floor_gain[k] = v;
    }
    int startup_dt_nfl = (effective_dt > 0.35) && (in->far_power > 1e-4) && !in->filter_once_converged;
    if (startup_dt_nfl && r->startup_dt_noise_floor_scale < 1.0f) {
        if (r->startup_dt_noise_floor_scale > 0.0f) {
            for (int k = 0; k < K; ++k) {
                float v = noise_floor_gain[k] * r->startup_dt_noise_floor_scale;
                if (r->gain_smooth[k] < v) r->gain_smooth[k] = v;
            }
        }
    } else {
        for (int k = 0; k < K; ++k) {
            if (r->gain_smooth[k] < noise_floor_gain[k]) r->gain_smooth[k] = noise_floor_gain[k];
        }
    }

    /* Apply gain to spec_synth */
    for (int k = 0; k < K; ++k) {
        r->enhanced_spec[k].r = r->gain_smooth[k] * r->spec_synth[k].r;
        r->enhanced_spec[k].i = r->gain_smooth[k] * r->spec_synth[k].i;
    }

    /* CNG */
    double npsd_sum = 0.0;
    for (int k = 0; k < K; ++k) npsd_sum += r->noise_psd[k];
    if (r->enable_cng && npsd_sum > 1e-7) {
        for (int k = 0; k < K; ++k) {
            float gs = r->gain_smooth[k];
            float t = 1.0f - gs * gs;
            if (t < 0.0f) t = 0.0f;
            float target = sqrtf(t) * 0.4f;
            r->smooth_cn_gain[k] = 0.8f * r->smooth_cn_gain[k] + 0.2f * target;
            float std = sqrtf(r->noise_psd[k] / 2.0f);
            float cng_real = randn(&r->cng_rng_state) * std;
            float cng_imag = randn(&r->cng_rng_state) * std;
            r->enhanced_spec[k].r += r->smooth_cn_gain[k] * cng_real;
            r->enhanced_spec[k].i += r->smooth_cn_gain[k] * cng_imag;
        }
    }

    /* IFFT enhanced */
    fft_inverse(r->fft, r->enhanced_spec, r->time_scratch);
    for (int i = 0; i < r->frame_size; ++i) r->time_scratch[i] *= r->window[i];

    /* OLA */
    for (int i = 0; i < r->frame_size; ++i) r->ola_buf[i] += r->time_scratch[i];
    memcpy(output_hop, r->ola_buf, (size_t)hop * sizeof(float));
    memmove(r->ola_buf, r->ola_buf + hop, (size_t)(r->frame_size - hop) * sizeof(float));
    memset(r->ola_buf + (r->frame_size - hop), 0, (size_t)hop * sizeof(float));

    /* Diagnostics */
    double gsum = 0.0; float gmin_v = 1.0f;
    for (int k = 0; k < K; ++k) { gsum += r->gain_smooth[k]; if (r->gain_smooth[k] < gmin_v) gmin_v = r->gain_smooth[k]; }
    r->diag_gain_mean = gsum / K;
    r->diag_gain_min  = gmin_v;
    r->diag_effective_g_min = effective_g_min;
    r->diag_far_activity    = r->far_activity;
    r->diag_echo_psd_mean   = mean_f32(r->echo_psd, K);
    r->diag_error_psd_mean  = mean_f32(r->error_psd, K);
}
